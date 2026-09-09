#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Cell-exact mixed condensation and sparse Q2 mechanics preconditioning.

The root remains matrix-free. Its setup uses only 11-by-11 material-point
Jacobians, 81-by-81 element matrices, eight-mode cell LU factors, and a fixed
mechanical CSR graph. CPU-bound Plans use the native JAX-CPU sparse LU action;
other dynamic Plans retain bounded element-Schwarz factors and quasistatic
fallback retains a multiplicative triangular sweep. Every reverse action is the
exact coordinate transpose of its forward action. No constitutive term is omitted.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....ein import contract
from ....linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    AbstractPreconditionerBuilder,
    AbstractSparseLinearOperator,
    analyze_sparse_triangular,
    ArraySpace,
    BlockFactorizationPreconditioner,
    BlockLinearOperator,
    BlockSpace,
    DifferentiationPolicy,
    GMRES,
    LinearCapabilityError,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    LocalBlockDiagonalLinearOperator,
    LocalBlockPreconditioner,
    OperatorCapabilities,
    OperatorProperties,
    PreconditionerCostEstimate,
    PreconditionerProperties,
    PreconditioningPolicy,
    prepare as prepare_linear,
    PreparedLinearSolve,
    PyTreeSpace,
    SchurComplementLinearOperator,
    solve as solve_linear,
    solve_local_blocks,
    SparseLU,
    SparseStorage,
    SparseTriangularAnalysis,
    SparseTriangularFactor,
    SparseTriangularStatus,
    TolerancePolicy,
    TransposeLinearOperator,
)
from ....linalg._costs import _array_tree_storage_bytes
from ....sparse import EdgeRelation, SparseCoordinateOperator
from ._almonacid_2024_material import almonacid_2024_material_response


_RECIPE = "almonacid-2024/cell-exact-q2-dgpm1/ldu-element-schwarz"


def _properties():
    # Row equilibration and the dynamic active tangent are not self-adjoint.
    return PreconditionerProperties(
        linear=True,
        stationary=True,
        evidence={"linear": "construction", "stationary": "construction"},
    )


class Almonacid2024SymbolicPreconditioning(StrictModule, NonTrainableState):
    """Host-prepared mesh graph; all Newton refreshes change values only."""

    mechanical_template: SparseCoordinateOperator
    upper_template: SparseCoordinateOperator
    lower_template: SparseCoordinateOperator
    full_storage_template: SparseStorage
    full_a_entries: Array
    full_a_targets: Array
    full_b_entries: Array
    full_b_targets: Array
    full_c_entries: Array
    full_c_targets: Array
    full_d_targets: Array
    full_transpose_entries: Array
    mechanical_storage_template: SparseStorage
    mechanical_transpose_entries: Array
    mechanical_lower_analysis: SparseTriangularAnalysis
    mechanical_upper_analysis: SparseTriangularAnalysis
    mechanical_lower_positions: Array
    mechanical_upper_positions: Array
    mechanical_patch_gathers: Array
    mechanical_patch_valid: Array
    mechanical_patch_weights: Array
    assembly_entries: Array
    assembly_targets: Array
    symbolic_id: str = eqx.field(static=True)


def _triangular_symbolic(operator, triangle):
    storage = operator.sparse_storage()
    indices = np.asarray(storage.indices)
    indptr = np.asarray(storage.indptr)
    rows = np.repeat(np.arange(storage.shape[0]), np.diff(indptr))
    selected = indices <= rows if triangle == "lower" else indices >= rows
    positions = np.flatnonzero(selected)
    counts = np.bincount(rows[selected], minlength=storage.shape[0])
    triangular = SparseStorage(
        jnp.ones((positions.size,), dtype=storage.values.dtype),
        jnp.asarray(indices[selected], dtype=storage.indices.dtype),
        jnp.asarray(np.concatenate(([0], np.cumsum(counts))), dtype=storage.indptr.dtype),
        shape=storage.shape,
    )
    return analyze_sparse_triangular(triangular, triangle=triangle), jnp.asarray(
        positions, dtype=storage.indices.dtype
    )


def almonacid_2024_prepare_symbolic(geometry):
    """Analyze the constrained Q2 graph once, outside JAX tracing.

    The internal two-field order is ((p_cell[4], j_cell[4]), u_free[3]).
    Mechanical element entries use (cell, node, component, node, component).
    CSR coefficients are canonical row-major. Fixed/pulling displacement
    columns and rows are removed, not replaced by artificial diagonal entries.
    """
    cells = np.asarray(geometry.displacement_dofs, dtype=np.int64)
    free = np.asarray(geometry.free_dofs, dtype=np.int64)
    global_to_free = np.full(geometry.displacement_dof_count, -1, dtype=np.int64)
    global_to_free[free] = np.arange(free.size)
    local_nodes = global_to_free[cells]
    local_u = np.where(
        local_nodes[..., None] >= 0,
        3 * local_nodes[..., None] + np.arange(3),
        -1,
    ).reshape((geometry.cell_count, -1))
    local_size = local_u.shape[1]
    mechanical_size = 3 * free.size
    patch_valid = local_u >= 0
    patch_gathers = np.maximum(local_u, 0)
    patch_counts = np.bincount(patch_gathers[patch_valid], minlength=mechanical_size)
    patch_weights = np.where(patch_valid, 1.0 / patch_counts[patch_gathers], 0.0)
    rows = np.broadcast_to(
        local_u[:, :, None], (geometry.cell_count, local_size, local_size)
    )
    columns = np.broadcast_to(local_u[:, None, :], rows.shape)
    valid = ((rows >= 0) & (columns >= 0)).reshape((-1,))
    entries = np.flatnonzero(valid)
    keys, targets = np.unique(
        (
            rows.reshape((-1,))[entries] * mechanical_size
            + columns.reshape((-1,))[entries]
        ),
        return_inverse=True,
    )
    csr_rows, csr_columns = keys // mechanical_size, keys % mechanical_size
    mechanical_transpose = np.searchsorted(keys, csr_columns * mechanical_size + csr_rows)
    dtype = geometry.weights_m3.dtype
    u_space = ArraySpace((mechanical_size,), dtype=dtype)
    y_space = ArraySpace((geometry.cell_count, 8), dtype=dtype)
    identifier = canonical_fingerprint(
        {
            "recipe": _RECIPE,
            "fem": geometry.discretization.prepared_id,
            "free_nodes": free.tolist(),
            "cells": cells.tolist(),
        }
    )
    mechanical = SparseCoordinateOperator(
        EdgeRelation(
            csr_columns,
            csr_rows,
            source_size=mechanical_size,
            target_size=mechanical_size,
        ),
        jnp.zeros((keys.size,), dtype=dtype),
        source=u_space,
        target=u_space,
        operator_id=f"{identifier}/mechanical",
    )
    mechanical_storage = mechanical.sparse_storage()
    y_indices = np.arange(8 * geometry.cell_count).reshape((geometry.cell_count, 8))
    coupling_shape = (geometry.cell_count, local_size, 8)
    coupling_u = np.broadcast_to(local_u[:, :, None], coupling_shape).reshape((-1,))
    coupling_y = np.broadcast_to(y_indices[:, None, :], coupling_shape).reshape((-1,))
    relation = EdgeRelation(
        coupling_y,
        coupling_u,
        source_size=y_space.size,
        target_size=u_space.size,
        valid=coupling_u >= 0,
    )
    lower = SparseCoordinateOperator(
        relation,
        jnp.zeros(relation.route_shape, dtype=dtype),
        source=y_space,
        target=u_space,
        operator_id=f"{identifier}/u-y",
    )
    upper = SparseCoordinateOperator(
        relation.transpose(),
        jnp.zeros(relation.route_shape, dtype=dtype),
        source=u_space,
        target=y_space,
        operator_id=f"{identifier}/y-u",
    )
    cell = np.arange(geometry.cell_count)[:, None]
    mode = np.arange(4)[None, :]
    external_y = np.concatenate(
        (
            mechanical_size + 4 * cell + mode,
            mechanical_size + 4 * geometry.cell_count + 4 * cell + mode,
        ),
        axis=1,
    )
    full_size = mechanical_size + 8 * geometry.cell_count
    b_rows = np.broadcast_to(local_u[:, :, None], (geometry.cell_count, local_size, 8))
    b_columns = np.broadcast_to(external_y[:, None, :], b_rows.shape)
    b_valid = (b_rows >= 0).reshape((-1,))
    b_entries = np.flatnonzero(b_valid)
    c_rows = np.broadcast_to(external_y[:, :, None], (geometry.cell_count, 8, local_size))
    c_columns = np.broadcast_to(local_u[:, None, :], c_rows.shape)
    c_valid = (c_columns >= 0).reshape((-1,))
    c_entries = np.flatnonzero(c_valid)
    d_rows = np.broadcast_to(external_y[:, :, None], (geometry.cell_count, 8, 8))
    d_columns = np.broadcast_to(external_y[:, None, :], d_rows.shape)
    key_parts = (
        rows.reshape((-1,))[entries] * full_size + columns.reshape((-1,))[entries],
        b_rows.reshape((-1,))[b_entries] * full_size
        + b_columns.reshape((-1,))[b_entries],
        c_rows.reshape((-1,))[c_entries] * full_size
        + c_columns.reshape((-1,))[c_entries],
        d_rows.reshape((-1,)) * full_size + d_columns.reshape((-1,)),
    )
    full_keys, full_targets = np.unique(np.concatenate(key_parts), return_inverse=True)
    boundaries = np.cumsum([part.size for part in key_parts])
    full_a_targets, full_b_targets, full_c_targets, full_d_targets = np.split(
        full_targets, boundaries[:-1]
    )
    full_rows, full_columns = full_keys // full_size, full_keys % full_size
    full_space = ArraySpace((full_size,), dtype=dtype)
    full_template = SparseCoordinateOperator(
        EdgeRelation(
            full_columns,
            full_rows,
            source_size=full_size,
            target_size=full_size,
        ),
        jnp.zeros((full_keys.size,), dtype=dtype),
        source=full_space,
        target=full_space,
        operator_id=f"{identifier}/full-mixed",
    )
    full_storage = full_template.sparse_storage()
    full_transpose = np.searchsorted(full_keys, full_columns * full_size + full_rows)
    lower_analysis, lower_positions = _triangular_symbolic(mechanical, "lower")
    upper_analysis, upper_positions = _triangular_symbolic(mechanical, "upper")
    return Almonacid2024SymbolicPreconditioning(
        mechanical_template=mechanical,
        upper_template=upper,
        lower_template=lower,
        full_storage_template=full_storage,
        full_a_entries=jnp.asarray(entries, dtype=jnp.int32),
        full_a_targets=jnp.asarray(full_a_targets, dtype=jnp.int32),
        full_b_entries=jnp.asarray(b_entries, dtype=jnp.int32),
        full_b_targets=jnp.asarray(full_b_targets, dtype=jnp.int32),
        full_c_entries=jnp.asarray(c_entries, dtype=jnp.int32),
        full_c_targets=jnp.asarray(full_c_targets, dtype=jnp.int32),
        full_d_targets=jnp.asarray(full_d_targets, dtype=jnp.int32),
        full_transpose_entries=jnp.asarray(full_transpose, dtype=jnp.int32),
        mechanical_storage_template=mechanical_storage,
        mechanical_transpose_entries=jnp.asarray(mechanical_transpose, dtype=jnp.int32),
        mechanical_lower_analysis=lower_analysis,
        mechanical_upper_analysis=upper_analysis,
        mechanical_lower_positions=lower_positions,
        mechanical_upper_positions=upper_positions,
        mechanical_patch_gathers=jnp.asarray(patch_gathers, dtype=jnp.int32),
        mechanical_patch_valid=jnp.asarray(patch_valid),
        mechanical_patch_weights=jnp.asarray(patch_weights, dtype=dtype),
        assembly_entries=jnp.asarray(entries, dtype=jnp.int32),
        assembly_targets=jnp.asarray(targets, dtype=jnp.int32),
        symbolic_id=identifier,
    )


def _to_blocks(vector):
    mechanical, pressure, dilation = vector
    return jnp.concatenate((pressure, dilation), axis=1), mechanical.reshape((-1,))


def _from_blocks(blocks):
    scalars, mechanical = blocks
    return mechanical.reshape((-1, 3)), scalars[:, :4], scalars[:, 4:]


class _CanonicalSparseOperator(AbstractSparseLinearOperator):
    """Trace-safe canonical CSR storage paired with one structured action."""

    action: AbstractLinearOperator
    storage: SparseStorage

    def __init__(self, action, storage, /):
        if storage.shape != (action.target.size, action.source.size):
            raise ValueError("Canonical sparse storage must match the action spaces.")
        self.action = action
        self.storage = storage
        self.source = action.source
        self.target = action.target
        self.properties = action.properties
        self.capabilities = action.capabilities
        self.batch_shape = action.batch_shape
        self.operator_id = f"{action.operator_id}/canonical-csr"

    def sparse_storage(self, /):
        return self.storage

    def mv(self, vector, /):
        return self.action.mv(vector)

    def transpose_mv(self, vector, /):
        return self.action.transpose_mv(vector)

    def adjoint_mv(self, vector, /):
        return self.action.adjoint_mv(vector)

    def _assemble_diagonal(self, /):
        return self.action._assemble_diagonal()

    def _materialize(self, /):
        return self.action._materialize()


class _Almonacid2024SetupOperator(AbstractLinearOperator):
    """Exact sparse/local tangent on the root's structured Q2/DGPM1 space."""

    block_operator: BlockLinearOperator
    scalar_inverse: LocalBlockPreconditioner
    mechanical_schur: SparseCoordinateOperator
    full_storage: SparseStorage
    full_transpose_entries: Array
    mechanical_storage_template: SparseStorage
    mechanical_transpose_entries: Array
    mechanical_lower_analysis: SparseTriangularAnalysis
    mechanical_upper_analysis: SparseTriangularAnalysis
    mechanical_lower_positions: Array
    mechanical_upper_positions: Array
    mechanical_patch_blocks: Array
    mechanical_patch_gathers: Array
    mechanical_patch_valid: Array
    mechanical_patch_weights: Array
    dynamic: bool = eqx.field(static=True)

    def __init__(
        self,
        blocks,
        scalar_inverse,
        mechanical_schur,
        patch_blocks,
        full_storage,
        symbolic,
        coordinates,
        *,
        dynamic,
    ):
        self.block_operator = blocks
        self.scalar_inverse = scalar_inverse
        self.mechanical_schur = mechanical_schur
        self.full_storage = full_storage
        self.full_transpose_entries = symbolic.full_transpose_entries
        self.mechanical_storage_template = symbolic.mechanical_storage_template
        self.mechanical_transpose_entries = symbolic.mechanical_transpose_entries
        self.mechanical_lower_analysis = symbolic.mechanical_lower_analysis
        self.mechanical_upper_analysis = symbolic.mechanical_upper_analysis
        self.mechanical_lower_positions = symbolic.mechanical_lower_positions
        self.mechanical_upper_positions = symbolic.mechanical_upper_positions
        self.mechanical_patch_blocks = patch_blocks
        self.mechanical_patch_gathers = symbolic.mechanical_patch_gathers
        self.mechanical_patch_valid = symbolic.mechanical_patch_valid
        self.mechanical_patch_weights = symbolic.mechanical_patch_weights
        self.dynamic = bool(dynamic)
        self.source = PyTreeSpace(coordinates)
        self.target = self.source
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False
        )
        self.batch_shape = ()
        self.operator_id = f"{symbolic.symbolic_id}/root-setup"

    def mv(self, vector, /):
        blocks = _to_blocks(self.source.validate(vector))
        return _from_blocks(self.block_operator.mv(blocks))

    def transpose_mv(self, vector, /):
        blocks = _to_blocks(self.target.validate(vector))
        return _from_blocks(self.block_operator.transpose_mv(blocks))

    def adjoint_mv(self, vector, /):
        return jax.tree.map(jnp.conj, self.transpose_mv(jax.tree.map(jnp.conj, vector)))

    def _materialize(self, /):
        raise LinearCapabilityError(
            "Almonacid setup cannot materialize a global dense Jacobian."
        )


def _element_tangents(model, coordinates, control):
    """Differentiate at material points, then integrate bounded-size elements."""
    g = model.geometry
    length, stress = model.plan.geometry.muscle_length_m, model.plan.stress_scale_Pa
    dt = control.time_s - model.state.time_s
    displacement, pressure, dilation = coordinates
    deformation = model.deformation(
        model._displacement(displacement, control.engineering_strain)
    )
    previous = (
        model.state.deformation_gradient
        if model.plan.dynamic
        else jnp.broadcast_to(jnp.eye(3, dtype=deformation.dtype), deformation.shape)
    )
    points = jnp.concatenate(
        (
            deformation.reshape(deformation.shape[:2] + (9,)),
            contract("qa,ca->cq", g.scalar_basis, pressure)[..., None],
            contract("qa,ca->cq", g.scalar_basis, dilation)[..., None],
        ),
        axis=-1,
    )

    def point_residual(point, parameters, previous_F, direction, tissue):
        response = almonacid_2024_material_response(
            parameters,
            point[:9].reshape((3, 3)),
            previous_F,
            point[9] * stress,
            point[10],
            control.activation,
            dt,
            direction,
            tissue,
            dynamic=model.plan.dynamic,
        )
        return jnp.concatenate(
            (
                (response.first_piola_Pa / stress).reshape((9,)),
                jnp.stack(
                    (response.volume_constraint, response.dilation_residual_Pa / stress)
                ),
            )
        )

    point_tangent = jax.jacfwd(point_residual)

    def cell(parameters, values, last, direction, tissue):
        return jax.vmap(
            lambda point, old: point_tangent(point, parameters, old, direction, tissue)
        )(values, last)

    tangent = jax.vmap(cell)(
        model._tissue_parameters(g.tissue_ids),
        points,
        previous,
        g.reference_directions,
        g.tissue_ids,
    )
    tangent = eqx.error_if(
        tangent,
        jnp.any(~jnp.isfinite(tangent)),
        "Almonacid material-point tangent is nonfinite.",
    )
    cq = tangent.shape[:2]
    mechanical = length**2 * contract(
        "cqiJkL,cqaJ,cqbL,cq->caibk",
        tangent[:, :, :9, :9].reshape(cq + (3, 3, 3, 3)),
        g.gradients,
        g.gradients,
        g.weights_m3,
    )
    if model.plan.dynamic:
        mass = (
            model.parameters.density_kg_per_m3
            * length**2
            / (stress * dt**2)
            * contract(
                "qa,qb,cq->cab",
                g.basis,
                g.basis,
                g.weights_m3,
            )
        )
        mechanical = (
            mechanical
            + mass[:, :, None, :, None]
            * jnp.eye(3, dtype=mass.dtype)[None, None, :, None, :]
        )
    lower = length * contract(
        "cqiJt,cqaJ,qb,cq->caitb",
        tangent[:, :, :9, 9:].reshape(cq + (3, 3, 2)),
        g.gradients,
        g.scalar_basis,
        g.weights_m3,
    )
    upper = length * contract(
        "cqtkL,qa,cqbL,cq->ctabk",
        tangent[:, :, 9:, :9].reshape(cq + (2, 3, 3)),
        g.scalar_basis,
        g.gradients,
        g.weights_m3,
    )
    scalar = contract(
        "cqst,qa,qb,cq->csatb",
        tangent[:, :, 9:, 9:],
        g.scalar_basis,
        g.scalar_basis,
        g.weights_m3,
    )
    # Exactly _root_residual's fixed reference row equilibration, including
    # shared-node volumes; a per-cell nodal denominator would be incorrect.
    cell_volume = jnp.sum(g.weights_m3, axis=1)
    nodal = contract("qa,qa,cq->ca", g.basis, g.basis, g.weights_m3)
    nodal_volume = (
        jnp.zeros((g.displacement_dof_count,), dtype=tangent.dtype)
        .at[g.displacement_dofs]
        .add(nodal)
    )
    local_volume = jnp.repeat(nodal_volume[g.displacement_dofs], 3, axis=1)
    count = g.cell_count
    mechanical = mechanical.reshape((count, 81, 81)) / local_volume[:, :, None]
    lower = lower.reshape((count, 81, 8)) / local_volume[:, :, None]
    upper = upper.reshape((count, 8, 81)) / cell_volume[:, None, None]
    scalar = scalar.reshape((count, 8, 8)) / cell_volume[:, None, None]
    return mechanical, lower, upper, scalar


def almonacid_2024_setup_operator(model, coordinates, control):
    """Refresh the exact mixed blocks and condensed sparse mechanical values."""
    symbolic = model.linear_symbolic
    mechanical, lower, upper, scalar = _element_tangents(model, coordinates, control)
    scalar_inverse = LocalBlockPreconditioner(
        scalar, preconditioner_id=f"{symbolic.symbolic_id}/cell-lu"
    )
    eliminated, failed = solve_local_blocks(scalar_inverse.factorization, upper)
    eliminated = eqx.error_if(eliminated, failed, "Almonacid cell condensation failed.")
    schur = mechanical - contract("cik,ckj->cij", lower, eliminated)
    patch_valid = symbolic.mechanical_patch_valid
    patch_blocks = jnp.where(
        patch_valid[:, :, None] & patch_valid[:, None, :], schur, 0.0
    )
    patch_blocks = patch_blocks + (
        (~patch_valid)[:, :, None]
        * jnp.eye(patch_blocks.shape[-1], dtype=patch_blocks.dtype)[None, :, :]
    )
    full_coefficients = jnp.zeros_like(symbolic.full_storage_template.values)
    full_coefficients = full_coefficients.at[symbolic.full_a_targets].add(
        mechanical.reshape((-1,))[symbolic.full_a_entries]
    )
    full_coefficients = full_coefficients.at[symbolic.full_b_targets].add(
        lower.reshape((-1,))[symbolic.full_b_entries]
    )
    full_coefficients = full_coefficients.at[symbolic.full_c_targets].add(
        upper.reshape((-1,))[symbolic.full_c_entries]
    )
    full_coefficients = full_coefficients.at[symbolic.full_d_targets].add(
        scalar.reshape((-1,))
    )
    full_storage = eqx.tree_at(
        lambda value: value.values,
        symbolic.full_storage_template,
        full_coefficients,
    )

    def assemble(values):
        coefficients = (
            jnp.zeros_like(symbolic.mechanical_template.coefficients)
            .at[symbolic.assembly_targets]
            .add(
                values.reshape((-1,))[symbolic.assembly_entries],
            )
        )
        return eqx.tree_at(
            lambda operator: operator.coefficients,
            symbolic.mechanical_template,
            coefficients,
        )

    mechanical_operator = assemble(mechanical)
    schur_operator = assemble(schur)
    lower_operator = eqx.tree_at(
        lambda operator: operator.coefficients,
        symbolic.lower_template,
        lower.reshape((-1,)),
    )
    upper_operator = eqx.tree_at(
        lambda operator: operator.coefficients,
        symbolic.upper_template,
        jnp.swapaxes(upper, -1, -2).reshape((-1,)),
    )
    scalar_operator = LocalBlockDiagonalLinearOperator(
        scalar, operator_id=f"{symbolic.symbolic_id}/scalar"
    )
    block_space = BlockSpace((scalar_operator.source, mechanical_operator.source))
    blocks = BlockLinearOperator(
        ((scalar_operator, upper_operator), (lower_operator, mechanical_operator)),
        source=block_space,
        target=block_space,
        operator_id=f"{symbolic.symbolic_id}/mixed-blocks",
    )
    return _Almonacid2024SetupOperator(
        blocks,
        scalar_inverse,
        schur_operator,
        patch_blocks,
        full_storage,
        symbolic,
        coordinates,
        dynamic=model.plan.dynamic,
    )


class _SparseDirectPreconditioner(AbstractPreconditioner):
    """Native JAX-CPU sparse solve used as a fixed linear action."""

    prepared: PreparedLinearSolve

    def __init__(self, prepared, /):
        self.prepared = prepared
        self.space = prepared.problem.operator.source
        self.properties = _properties()
        self.preconditioner_id = (
            f"{prepared.problem.operator.operator_id}/jax-cpu-sparse-lu"
        )

    def apply(self, residual, /, *, iteration=None):
        del iteration
        result = solve_linear(self.prepared, self.space.validate(residual))
        return eqx.error_if(
            result.value,
            result.status != int(LinearSolveStatus.SUCCESS),
            "Almonacid sparse direct preconditioner solve failed.",
        )


class _MechanicalPatchPreconditioner(AbstractPreconditioner):
    """Weighted element-Schwarz action and its exact coordinate transpose."""

    local_inverse: LocalBlockPreconditioner
    gathers: Array
    valid: Array
    weights: Array
    transpose_action: bool = eqx.field(static=True)

    def __init__(
        self, blocks, gathers, valid, weights, space, /, *, transpose_action=False
    ):
        transpose_action = bool(transpose_action)
        factored = jnp.swapaxes(blocks, -1, -2) if transpose_action else blocks
        self.local_inverse = LocalBlockPreconditioner(
            factored, preconditioner_id=f"{_RECIPE}/patch-local"
        )
        self.gathers = gathers
        self.valid = valid
        self.weights = weights
        self.transpose_action = transpose_action
        self.space = space
        self.properties = _properties()
        suffix = "transpose" if transpose_action else "forward"
        self.preconditioner_id = f"{_RECIPE}/patch/{suffix}"

    def apply(self, residual, /, *, iteration=None):
        del iteration
        right_hand_side = self.space.validate(residual)
        local = right_hand_side[self.gathers]
        if self.transpose_action:
            local = local * self.weights
        local = jnp.where(self.valid, local, 0.0)
        correction = self.local_inverse.apply(local)
        if not self.transpose_action:
            correction = correction * self.weights
        correction = jnp.where(self.valid, correction, 0.0)
        return jnp.zeros_like(right_hand_side).at[self.gathers].add(correction)


class _MechanicalSweepPreconditioner(AbstractPreconditioner):
    """Repeated two-triangle corrections with an exact coordinate transpose."""

    operator: SparseCoordinateOperator
    lower: SparseTriangularFactor
    upper: SparseTriangularFactor
    transpose_action: bool = eqx.field(static=True)
    sweeps: int = eqx.field(static=True)

    def __init__(self, operator, lower, upper, /, *, transpose_action=False, sweeps=4):
        self.operator = operator
        self.lower = lower
        self.upper = upper
        self.transpose_action = bool(transpose_action)
        self.sweeps = int(sweeps)
        self.space = operator.source
        self.properties = _properties()
        suffix = "transpose" if self.transpose_action else "forward"
        self.preconditioner_id = (
            f"{operator.operator_id}/two-triangle/{self.sweeps}/{suffix}"
        )

    @staticmethod
    def _solve(factor, right_hand_side, *, transpose=False):
        result = factor.solve(right_hand_side, transpose=transpose)
        return eqx.error_if(
            result.value,
            result.status != int(SparseTriangularStatus.SUCCESS),
            "Almonacid mechanical triangular sweep failed.",
        )

    def _one_sweep(self, right_hand_side):
        if not self.transpose_action:
            first = self._solve(self.lower, right_hand_side)
            defect = right_hand_side - self.operator.mv(first)
            second = self._solve(self.upper, defect)
        else:
            first = self._solve(self.upper, right_hand_side, transpose=True)
            defect = right_hand_side - self.operator.transpose_mv(first)
            second = self._solve(self.lower, defect, transpose=True)
        return first + second

    def apply(self, residual, /, *, iteration=None):
        del iteration
        right_hand_side = self.space.validate(residual)

        def correction(_, current):
            image = (
                self.operator.transpose_mv(current)
                if self.transpose_action
                else self.operator.mv(current)
            )
            return current + self._one_sweep(right_hand_side - image)

        return jax.lax.fori_loop(
            0, self.sweeps, correction, jnp.zeros_like(right_hand_side)
        )


class _Almonacid2024Preconditioner(AbstractPreconditioner):
    """Permutation adapter around the native fixed linear LDU action."""

    inner: BlockFactorizationPreconditioner

    def __init__(self, inner, setup):
        self.inner = inner
        self.space = setup.source
        self.properties = inner.properties
        self.preconditioner_id = f"{setup.operator_id}/ldu"

    def apply(self, residual, /, *, iteration=None):
        blocks = _to_blocks(self.space.validate(residual))
        return _from_blocks(self.inner.apply(blocks, iteration=iteration))


def _source_setup(operator):
    source = (
        operator.operator if isinstance(operator, TransposeLinearOperator) else operator
    )
    if not isinstance(source, _Almonacid2024SetupOperator):
        raise TypeError(
            "Almonacid preconditioning requires its cell-condensed setup operator."
        )
    return source


class _Almonacid2024PreconditionerBuilder(AbstractPreconditionerBuilder):
    mechanical_solver: str = eqx.field(static=True)

    def __init__(self, mechanical_solver=None):
        if mechanical_solver is None:
            selected = (
                "full-mixed-jax-cpu"
                if jax.default_backend() == "cpu"
                else "element-schwarz"
            )
        else:
            selected = str(mechanical_solver)
        if selected not in (
            "full-mixed-jax-cpu",
            "condensed-jax-cpu",
            "element-schwarz",
        ):
            raise ValueError("Unknown Almonacid mechanical solver policy.")
        if selected.endswith("jax-cpu") and jax.default_backend() != "cpu":
            raise ValueError("JAX-CPU sparse policies require the CPU backend.")
        self.mechanical_solver = selected

    @property
    def builder_id(self):
        return f"{_RECIPE}/{self.mechanical_solver}"

    @property
    def default_refresh(self):
        return "numeric"

    def properties_for(self, setup_operator, /):
        _source_setup(setup_operator)
        return _properties()

    def cost_for(self, setup_operator, /, *, materialization=None):
        del materialization
        source = _source_setup(setup_operator)
        itemsize = source.mechanical_schur.coefficients.dtype.itemsize
        direct = (
            self.mechanical_solver
            in (
                "full-mixed-jax-cpu",
                "condensed-jax-cpu",
            )
            and source.dynamic
        )
        if direct:
            storage = (
                source.full_storage
                if self.mechanical_solver == "full-mixed-jax-cpu"
                else source.mechanical_storage_template
            )
            sparse_bytes = (
                storage.values.nbytes + storage.indices.nbytes + storage.indptr.nbytes
            )
            factor_storage_bytes = 0
            preparation_workspace_bytes = 8 * sparse_bytes
            reason = f"{self.mechanical_solver} native sparse LU"
        else:
            factor_storage_bytes = (
                source.mechanical_patch_blocks.nbytes
                if source.dynamic
                else (
                    source.mechanical_lower_positions.size
                    + source.mechanical_upper_positions.size
                )
                * itemsize
            )
            preparation_workspace_bytes = factor_storage_bytes
            reason = (
                "cell LU plus dynamic element Schwarz or quasistatic "
                "sparse triangular mechanics"
            )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=_array_tree_storage_bytes(source) + factor_storage_bytes,
            preparation_workspace_bytes=preparation_workspace_bytes,
            apply_workspace_bytes_per_rhs=8 * source.source.size * itemsize,
            reason=reason,
        )

    def prepare(self, setup_operator, /, *, materialization):
        del materialization
        source = _source_setup(setup_operator)
        transpose_action = isinstance(setup_operator, TransposeLinearOperator)
        direct = (
            self.mechanical_solver
            in (
                "full-mixed-jax-cpu",
                "condensed-jax-cpu",
            )
            and source.dynamic
        )
        if direct:
            full_mixed = self.mechanical_solver == "full-mixed-jax-cpu"
            action = source if full_mixed else source.mechanical_schur
            storage = (
                source.full_storage if full_mixed else source.mechanical_storage_template
            )
            coefficients = (
                source.full_storage.values
                if full_mixed
                else source.mechanical_schur.coefficients
            )
            if transpose_action:
                action = TransposeLinearOperator(action)
                positions = (
                    source.full_transpose_entries
                    if full_mixed
                    else source.mechanical_transpose_entries
                )
                coefficients = coefficients[positions]
            storage = eqx.tree_at(
                lambda value: value.values,
                storage,
                coefficients,
            )
            direct_operator = _CanonicalSparseOperator(action, storage)
            prepared_direct = prepare_linear(
                LinearSystem(direct_operator),
                LinearSolvePolicy(
                    SparseLU(provider="jax-cpu"),
                    tolerance=TolerancePolicy(
                        relative=1e-7,
                        absolute=1e-10,
                    ),
                    differentiation=DifferentiationPolicy("none"),
                ),
            )
            sparse_inverse = _SparseDirectPreconditioner(prepared_direct)
            if full_mixed:
                return sparse_inverse

        properties = _properties()
        blocks = source.block_operator.blocks
        scalar_inverse = source.scalar_inverse
        mechanical, lower, upper = blocks[1][1], blocks[1][0], blocks[0][1]
        if self.mechanical_solver == "condensed-jax-cpu" and source.dynamic:
            mechanical_inverse = sparse_inverse
        elif source.dynamic:
            mechanical_inverse = _MechanicalPatchPreconditioner(
                source.mechanical_patch_blocks,
                source.mechanical_patch_gathers,
                source.mechanical_patch_valid,
                source.mechanical_patch_weights,
                source.mechanical_schur.source,
                transpose_action=transpose_action,
            )
        else:
            lower_factor = SparseTriangularFactor(
                source.mechanical_lower_analysis,
                source.mechanical_schur.coefficients[source.mechanical_lower_positions],
            )
            upper_factor = SparseTriangularFactor(
                source.mechanical_upper_analysis,
                source.mechanical_schur.coefficients[source.mechanical_upper_positions],
            )
            mechanical_inverse = _MechanicalSweepPreconditioner(
                source.mechanical_schur,
                lower_factor,
                upper_factor,
                transpose_action=transpose_action,
            )
        if transpose_action:
            scalar_inverse = LocalBlockPreconditioner(
                jnp.swapaxes(blocks[0][0].blocks, -1, -2),
                preconditioner_id=f"{setup_operator.operator_id}/cell-lu",
            )
            mechanical = TransposeLinearOperator(mechanical)
            lower = TransposeLinearOperator(upper)
            upper = TransposeLinearOperator(blocks[1][0])
        schur = SchurComplementLinearOperator(
            mechanical,
            lower,
            scalar_inverse,
            upper,
            operator_id=f"{setup_operator.operator_id}/schur",
        )
        inner = BlockFactorizationPreconditioner(
            schur,
            mechanical_inverse,
            "ldu",
            properties=properties,
            space=source.block_operator.source,
            preconditioner_id=f"{setup_operator.operator_id}/block-ldu",
        )
        return _Almonacid2024Preconditioner(inner, setup_operator)

    def refresh(self, preconditioner, setup_operator, /, *, materialization):
        if not isinstance(
            preconditioner,
            (_SparseDirectPreconditioner, _Almonacid2024Preconditioner),
        ):
            raise TypeError("Almonacid refresh requires its prepared sparse action.")
        if not preconditioner.space.compatible(setup_operator.source):
            raise ValueError(
                "Almonacid numeric refresh cannot change the symbolic root space."
            )
        return self.prepare(setup_operator, materialization=materialization)


def almonacid_2024_linear_policy(mechanical_solver=None):
    """Source-scaled right-preconditioned GMRES with numeric setup refresh."""
    return LinearSolvePolicy(
        GMRES(restart=128),
        tolerance=TolerancePolicy(
            relative=1e-7,
            absolute=1e-11,
            max_steps=2048,
        ),
        preconditioning=PreconditioningPolicy(
            _Almonacid2024PreconditionerBuilder(mechanical_solver),
            side="right",
            refresh="numeric",
        ),
    )


__all__ = [
    "Almonacid2024SymbolicPreconditioning",
    "almonacid_2024_linear_policy",
    "almonacid_2024_prepare_symbolic",
    "almonacid_2024_setup_operator",
]
