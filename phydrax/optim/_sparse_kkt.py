#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import OperatorProperties
from ..sparse import EdgeRelation, SparseCoordinateOperator, SparseLinearMap
from ._structured_nonlinear import StructuredNonlinearTemplate


class SparseAugmentedKKTPlan(StrictModule):
    """Static full-symmetric CSR routes for one bound-form primal-dual KKT."""

    template: StructuredNonlinearTemplate
    relation: EdgeRelation
    equality_indices: Array
    general_indices: Array
    hessian_positions: Array
    hessian_sources: Array
    equality_jacobian_positions: Array
    equality_jacobian_sources: Array
    general_jacobian_positions: Array
    general_jacobian_sources: Array
    primal_diagonal_positions: Array
    slack_diagonal_positions: Array
    equality_dual_diagonal_positions: Array
    general_dual_diagonal_positions: Array
    slack_constraint_positions: Array
    num_primal: int = eqx.field(static=True)
    num_slacks: int = eqx.field(static=True)
    num_equalities: int = eqx.field(static=True)
    kkt_dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _position_map(entries: set[tuple[int, int]], /):
    ordered = tuple(sorted(entries))
    lookup = {entry: index for index, entry in enumerate(ordered)}
    rows = np.asarray([row for row, _ in ordered], dtype=np.int32)
    columns = np.asarray([column for _, column in ordered], dtype=np.int32)
    return ordered, lookup, rows, columns


def plan_sparse_augmented_kkt(
    template: StructuredNonlinearTemplate,
    /,
) -> SparseAugmentedKKTPlan:
    if not isinstance(template, StructuredNonlinearTemplate):
        raise TypeError("template must be a StructuredNonlinearTemplate.")
    program = template.program
    if program.hessian_plan is None:
        raise ValueError("Sparse augmented KKT requires an exact Hessian plan.")
    hessian_pattern = program.hessian_plan.pattern
    if not hessian_pattern.symmetric:
        raise ValueError(
            "Sparse augmented KKT requires an explicit symmetric Hessian pattern."
        )

    equality_indices = np.asarray(program.equality_indices, dtype=np.int32)
    general_indices = np.unique(
        np.concatenate(
            (
                np.asarray(program.lower_indices, dtype=np.int32),
                np.asarray(program.upper_indices, dtype=np.int32),
            )
        )
    )
    equality_lookup = {int(value): index for index, value in enumerate(equality_indices)}
    general_lookup = {int(value): index for index, value in enumerate(general_indices)}
    n = program.num_variables
    nc = int(equality_indices.size)
    nd = int(general_indices.size)
    x_offset = 0
    slack_offset = n
    equality_offset = n + nd
    general_offset = equality_offset + nc
    dimension = n + nd + nc + nd

    h_rows = np.asarray(hessian_pattern.rows, dtype=np.int32)
    h_cols = np.asarray(hessian_pattern.cols, dtype=np.int32)
    j_rows = np.asarray(program.jacobian_plan.pattern.rows, dtype=np.int32)
    j_cols = np.asarray(program.jacobian_plan.pattern.cols, dtype=np.int32)
    entries: set[tuple[int, int]] = set(
        zip(h_rows.tolist(), h_cols.tolist(), strict=True)
    )
    entries.update((index, index) for index in range(n))
    entries.update((slack_offset + index, slack_offset + index) for index in range(nd))
    entries.update(
        (equality_offset + index, equality_offset + index) for index in range(nc)
    )
    entries.update(
        (general_offset + index, general_offset + index) for index in range(nd)
    )

    equality_routes = []
    general_routes = []
    for source_position, (constraint, variable) in enumerate(
        zip(j_rows.tolist(), j_cols.tolist(), strict=True)
    ):
        if constraint in equality_lookup:
            row = equality_offset + equality_lookup[constraint]
            entries.add((row, x_offset + variable))
            entries.add((x_offset + variable, row))
            equality_routes.append((source_position, row, x_offset + variable))
        if constraint in general_lookup:
            row = general_offset + general_lookup[constraint]
            entries.add((row, x_offset + variable))
            entries.add((x_offset + variable, row))
            general_routes.append((source_position, row, x_offset + variable))
    for index in range(nd):
        slack = slack_offset + index
        constraint = general_offset + index
        entries.add((slack, constraint))
        entries.add((constraint, slack))

    _, lookup, rows, columns = _position_map(entries)
    hessian_positions = np.asarray(
        [
            lookup[(int(row), int(column))]
            for row, column in zip(h_rows, h_cols, strict=True)
        ],
        dtype=np.int32,
    )
    hessian_sources = np.arange(h_rows.size, dtype=np.int32)

    equality_positions = []
    equality_sources = []
    for source_position, row, column in equality_routes:
        equality_positions.extend((lookup[(row, column)], lookup[(column, row)]))
        equality_sources.extend((source_position, source_position))
    general_positions = []
    general_sources = []
    for source_position, row, column in general_routes:
        general_positions.extend((lookup[(row, column)], lookup[(column, row)]))
        general_sources.extend((source_position, source_position))

    primal_diagonal = np.asarray(
        [lookup[(index, index)] for index in range(n)],
        dtype=np.int32,
    )
    slack_diagonal = np.asarray(
        [lookup[(slack_offset + index, slack_offset + index)] for index in range(nd)],
        dtype=np.int32,
    )
    equality_dual_diagonal = np.asarray(
        [
            lookup[(equality_offset + index, equality_offset + index)]
            for index in range(nc)
        ],
        dtype=np.int32,
    )
    general_dual_diagonal = np.asarray(
        [lookup[(general_offset + index, general_offset + index)] for index in range(nd)],
        dtype=np.int32,
    )
    slack_constraint_positions = []
    for index in range(nd):
        slack = slack_offset + index
        constraint = general_offset + index
        slack_constraint_positions.extend(
            (lookup[(slack, constraint)], lookup[(constraint, slack)])
        )

    relation = EdgeRelation(
        jnp.asarray(columns),
        jnp.asarray(rows),
        source_size=dimension,
        target_size=dimension,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "sparse-augmented-kkt",
            "template": template.template_id,
            "equality_indices": equality_indices.tolist(),
            "general_indices": general_indices.tolist(),
            "rows": rows.tolist(),
            "columns": columns.tolist(),
        }
    )
    return SparseAugmentedKKTPlan(
        template,
        relation,
        jnp.asarray(equality_indices),
        jnp.asarray(general_indices),
        jnp.asarray(hessian_positions),
        jnp.asarray(hessian_sources),
        jnp.asarray(equality_positions, dtype=jnp.int32),
        jnp.asarray(equality_sources, dtype=jnp.int32),
        jnp.asarray(general_positions, dtype=jnp.int32),
        jnp.asarray(general_sources, dtype=jnp.int32),
        jnp.asarray(primal_diagonal),
        jnp.asarray(slack_diagonal),
        jnp.asarray(equality_dual_diagonal),
        jnp.asarray(general_dual_diagonal),
        jnp.asarray(slack_constraint_positions, dtype=jnp.int32),
        n,
        nd,
        nc,
        dimension,
        plan_id,
    )


def _vector(value: ArrayLike, size: int, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape[-1:] != (size,):
        raise ValueError(f"{name} must end in shape {(size,)}; got {array.shape}.")
    return array


def assemble_sparse_augmented_kkt(
    plan: SparseAugmentedKKTPlan,
    hessian: SparseCoordinateOperator,
    jacobian: SparseCoordinateOperator,
    sigma_primal: ArrayLike,
    sigma_slack: ArrayLike,
    /,
    *,
    primal_regularization: ArrayLike,
    dual_regularization: ArrayLike,
) -> SparseLinearMap:
    if not isinstance(plan, SparseAugmentedKKTPlan):
        raise TypeError("plan must be a SparseAugmentedKKTPlan.")
    if not isinstance(hessian, SparseCoordinateOperator):
        raise TypeError("hessian must be a SparseCoordinateOperator.")
    if not isinstance(jacobian, SparseCoordinateOperator):
        raise TypeError("jacobian must be a SparseCoordinateOperator.")
    hessian_values = jnp.asarray(hessian.coefficients)
    jacobian_values = jnp.asarray(jacobian.coefficients)
    sigma_x = _vector(sigma_primal, plan.num_primal, "sigma_primal")
    sigma_s = _vector(sigma_slack, plan.num_slacks, "sigma_slack")
    primal_shift = jnp.asarray(primal_regularization, dtype=hessian_values.dtype)
    dual_shift = jnp.asarray(dual_regularization, dtype=hessian_values.dtype)
    if primal_shift.shape != () or dual_shift.shape != ():
        raise ValueError("KKT regularization values must be scalar.")
    batch_shape = jnp.broadcast_shapes(
        hessian_values.shape[:-1],
        jacobian_values.shape[:-1],
        sigma_x.shape[:-1],
        sigma_s.shape[:-1],
    )
    values = jnp.zeros(
        batch_shape + (plan.relation.capacity,),
        dtype=jnp.result_type(hessian_values, jacobian_values, sigma_x, sigma_s),
    )
    hessian_values = jnp.broadcast_to(
        hessian_values,
        batch_shape + hessian_values.shape[-1:],
    )
    jacobian_values = jnp.broadcast_to(
        jacobian_values,
        batch_shape + jacobian_values.shape[-1:],
    )
    sigma_x = jnp.broadcast_to(sigma_x, batch_shape + (plan.num_primal,))
    sigma_s = jnp.broadcast_to(sigma_s, batch_shape + (plan.num_slacks,))
    values = values.at[..., plan.hessian_positions].add(
        hessian_values[..., plan.hessian_sources]
    )
    values = values.at[..., plan.equality_jacobian_positions].add(
        jacobian_values[..., plan.equality_jacobian_sources]
    )
    values = values.at[..., plan.general_jacobian_positions].add(
        jacobian_values[..., plan.general_jacobian_sources]
    )
    values = values.at[..., plan.primal_diagonal_positions].add(sigma_x + primal_shift)
    values = values.at[..., plan.slack_diagonal_positions].add(sigma_s)
    values = values.at[..., plan.equality_dual_diagonal_positions].add(-dual_shift)
    values = values.at[..., plan.general_dual_diagonal_positions].add(-dual_shift)
    values = values.at[..., plan.slack_constraint_positions].add(-1.0)
    return SparseLinearMap(
        plan.relation,
        values,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
        operator_id=plan.plan_id,
    )


__all__ = [
    "SparseAugmentedKKTPlan",
    "assemble_sparse_augmented_kkt",
    "plan_sparse_augmented_kkt",
]
