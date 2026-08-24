#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import scipy.sparse as sp

from ...backends import ClarabelPlan, prepare_clarabel, PreparedClarabel
from ._cones import (
    AbstractConvexCone,
    NonnegativeCone,
    ProductCone,
    RotatedSecondOrderCone,
    SecondOrderCone,
    ZeroCone,
)
from ._policy import ClarabelInteriorPoint, ConvexSolvePolicy
from ._problem import ConicProgram
from ._quadratic import _max_abs, ConvexProgramResult
from ._types import (
    ConvexProgramCertificate,
    ConvexProgramProvenance,
    ConvexProgramStatus,
)


@dataclass(frozen=True, slots=True)
class _ClarabelStructure:
    cones: tuple[object, ...]
    transforms: tuple[np.ndarray | None, ...]
    slices: tuple[slice, ...]
    fixed_indices: np.ndarray
    lower_indices: np.ndarray
    upper_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedClarabelProgram:
    provider: PreparedClarabel
    structure: _ClarabelStructure


def _cone_blocks(cone: AbstractConvexCone, /):
    if isinstance(cone, ProductCone):
        return cone.cones, cone.slices
    return (cone,), (slice(0, cone.dimension),)


def _rotated_transform(dimension: int, /) -> np.ndarray:
    transform = np.eye(dimension)
    scale = 1.0 / sqrt(2.0)
    transform[0, :2] = scale
    transform[1, 0] = scale
    transform[1, 1] = -scale
    return transform


def _clarabel_cones(prepared, cone: AbstractConvexCone, /):
    module = prepared.module
    mapped = []
    transforms = []
    _, slices = _cone_blocks(cone)
    for block, block_slice in zip(*_cone_blocks(cone), strict=True):
        dimension = block_slice.stop - block_slice.start
        if isinstance(block, ZeroCone):
            mapped.append(module.ZeroConeT(dimension))
            transforms.append(None)
        elif isinstance(block, NonnegativeCone):
            mapped.append(module.NonnegativeConeT(dimension))
            transforms.append(None)
        elif isinstance(block, SecondOrderCone):
            mapped.append(module.SecondOrderConeT(dimension))
            transforms.append(None)
        elif isinstance(block, RotatedSecondOrderCone):
            mapped.append(module.SecondOrderConeT(dimension))
            transforms.append(_rotated_transform(dimension))
        else:
            raise TypeError(f"Clarabel does not support cone {type(block).__name__!r}.")
    return tuple(mapped), tuple(transforms), tuple(slices)


def _bound_structure(program: ConicProgram, /):
    lower = np.asarray(program.lower_bounds).reshape((-1, program.num_variables))
    upper = np.asarray(program.upper_bounds).reshape((-1, program.num_variables))
    lower_finite = np.isfinite(lower)
    upper_finite = np.isfinite(upper)
    fixed = lower_finite & upper_finite & (lower == upper)
    roles = np.stack((lower_finite, upper_finite, fixed), axis=-1)
    if not np.all(roles == roles[:1]):
        raise ValueError("Clarabel requires shared bound roles across a program batch.")
    fixed_indices = np.flatnonzero(fixed[0])
    lower_indices = np.flatnonzero(lower_finite[0] & ~fixed[0])
    upper_indices = np.flatnonzero(upper_finite[0] & ~fixed[0])
    return fixed_indices, lower_indices, upper_indices


def _provider_plan(policy: ConvexSolvePolicy, /) -> ClarabelPlan:
    method = policy.method
    if not isinstance(method, ClarabelInteriorPoint):
        raise TypeError("Clarabel adapter requires ClarabelInteriorPoint.")
    source = method.plan
    return ClarabelPlan(
        max_iterations=policy.termination.maximum_steps,
        tolerance_gap_abs=max(policy.termination.absolute, 1e-12),
        tolerance_gap_rel=max(policy.termination.relative, 1e-12),
        tolerance_feasibility=max(policy.termination.absolute, 1e-12),
        presolve=source.presolve,
        verbose=source.verbose,
    )


def _prepare_structure(
    program: ConicProgram,
    policy: ConvexSolvePolicy,
    /,
) -> _PreparedClarabelProgram:
    prepared = prepare_clarabel(_provider_plan(policy))
    cones, transforms, slices = _clarabel_cones(prepared, program.cone)
    fixed, lower, upper = _bound_structure(program)
    module = prepared.module
    cones = cones + (
        (() if fixed.size == 0 else (module.ZeroConeT(int(fixed.size)),))
        + (() if lower.size == 0 else (module.NonnegativeConeT(int(lower.size)),))
        + (() if upper.size == 0 else (module.NonnegativeConeT(int(upper.size)),))
    )
    return _PreparedClarabelProgram(
        prepared,
        _ClarabelStructure(
            cones=cones,
            transforms=transforms,
            slices=slices,
            fixed_indices=fixed,
            lower_indices=lower,
            upper_indices=upper,
        ),
    )


def prepare_clarabel_policy(
    program: ConicProgram,
    policy: ConvexSolvePolicy,
    /,
) -> _PreparedClarabelProgram:
    """Prepare Clarabel settings and immutable cone/bound structure."""

    if not isinstance(program, ConicProgram):
        raise TypeError("program must be a ConicProgram.")
    return _prepare_structure(program, policy)


def _transformed_constraint_data(
    program: ConicProgram,
    structure: _ClarabelStructure,
    batch_index: int,
    /,
):
    count = int(np.prod(program.batch_shape)) if program.batch_shape else 1
    matrix = np.asarray(program.constraint_matrix).reshape(
        (count, program.num_constraints, program.num_variables)
    )[batch_index]
    rhs = np.asarray(program.constraint_rhs).reshape((count, program.num_constraints))[
        batch_index
    ]
    blocks = []
    rhs_blocks = []
    for block_slice, transform in zip(
        structure.slices, structure.transforms, strict=True
    ):
        block_matrix = matrix[block_slice]
        block_rhs = rhs[block_slice]
        if transform is not None:
            block_matrix = transform @ block_matrix
            block_rhs = transform @ block_rhs
        blocks.append(block_matrix)
        rhs_blocks.append(block_rhs)
    lower = np.asarray(program.lower_bounds).reshape((count, program.num_variables))[
        batch_index
    ]
    upper = np.asarray(program.upper_bounds).reshape((count, program.num_variables))[
        batch_index
    ]
    identity = np.eye(program.num_variables)
    if structure.fixed_indices.size:
        blocks.append(identity[structure.fixed_indices])
        rhs_blocks.append(lower[structure.fixed_indices])
    if structure.lower_indices.size:
        blocks.append(-identity[structure.lower_indices])
        rhs_blocks.append(-lower[structure.lower_indices])
    if structure.upper_indices.size:
        blocks.append(identity[structure.upper_indices])
        rhs_blocks.append(upper[structure.upper_indices])
    return np.concatenate(blocks, axis=0), np.concatenate(rhs_blocks, axis=0)


def _restore_cone_vector(
    value: np.ndarray,
    structure: _ClarabelStructure,
    original_dimension: int,
    /,
) -> np.ndarray:
    restored = np.asarray(value[:original_dimension]).copy()
    for block_slice, transform in zip(
        structure.slices, structure.transforms, strict=True
    ):
        if transform is not None:
            restored[block_slice] = transform.T @ restored[block_slice]
    return restored


def _bound_duals(
    value: np.ndarray,
    structure: _ClarabelStructure,
    original_dimension: int,
    variables: int,
    /,
):
    lower = np.zeros(variables)
    upper = np.zeros(variables)
    cursor = original_dimension
    if structure.fixed_indices.size:
        fixed = value[cursor : cursor + structure.fixed_indices.size]
        cursor += structure.fixed_indices.size
        lower[structure.fixed_indices] = np.maximum(-fixed, 0.0)
        upper[structure.fixed_indices] = np.maximum(fixed, 0.0)
    if structure.lower_indices.size:
        lower[structure.lower_indices] = value[
            cursor : cursor + structure.lower_indices.size
        ]
        cursor += structure.lower_indices.size
    if structure.upper_indices.size:
        upper[structure.upper_indices] = value[
            cursor : cursor + structure.upper_indices.size
        ]
    return lower, upper


def _recession_bound_residual(program: ConicProgram, ray, /):
    lower_finite = jnp.isfinite(program.lower_bounds)
    upper_finite = jnp.isfinite(program.upper_bounds)
    fixed = lower_finite & upper_finite & (program.lower_bounds == program.upper_bounds)
    violation = jnp.where(
        fixed | (lower_finite & upper_finite),
        jnp.abs(ray),
        jnp.where(
            lower_finite,
            jnp.maximum(-ray, 0.0),
            jnp.where(upper_finite, jnp.maximum(ray, 0.0), 0.0),
        ),
    )
    return _max_abs(violation)


def _audit_result(
    program: ConicProgram,
    primal,
    slack,
    dual,
    lower_dual,
    upper_dual,
    backend_status,
    iterations,
    policy,
    backend_version,
):
    dtype = program.linear.dtype
    quadratic = (
        jnp.zeros(
            program.batch_shape + (program.num_variables, program.num_variables),
            dtype=dtype,
        )
        if program.quadratic is None
        else program.quadratic
    )
    objective = 0.5 * oe.contract(
        "...i,...ij,...j->...", primal, quadratic, primal
    ) + jnp.sum(program.linear * primal, axis=-1)
    primal_residual = (
        oe.contract("...ij,...j->...i", program.constraint_matrix, primal)
        + slack
        - program.constraint_rhs
    )
    cone_projection = program.cone.project(slack)
    dual_projection = program.cone.project_dual(dual)
    cone_violation = slack - cone_projection
    dual_violation = dual - dual_projection
    stationarity = (
        oe.contract("...ij,...j->...i", quadratic, primal)
        + program.linear
        + oe.contract("...ji,...j->...i", program.constraint_matrix, dual)
        - lower_dual
        + upper_dual
    )
    solver_stationarity = stationarity + policy.regularization * primal
    bound_violation = jnp.maximum(
        jnp.maximum(program.lower_bounds - primal, primal - program.upper_bounds),
        0.0,
    )
    cone_complementarity = (
        program.cone.block_complementarity(slack, dual)
        if isinstance(program.cone, ProductCone)
        else program.cone.complementarity(slack, dual)[..., None]
    )
    lower_slack = jnp.where(
        jnp.isfinite(program.lower_bounds),
        primal - program.lower_bounds,
        0.0,
    )
    upper_slack = jnp.where(
        jnp.isfinite(program.upper_bounds),
        program.upper_bounds - primal,
        0.0,
    )
    bound_complementarity = jnp.concatenate(
        (lower_slack * lower_dual, upper_slack * upper_dual),
        axis=-1,
    )
    complementarity = jnp.concatenate(
        (cone_complementarity, bound_complementarity),
        axis=-1,
    )
    primal_norm = jnp.maximum(
        _max_abs(primal_residual),
        jnp.maximum(_max_abs(cone_violation), _max_abs(bound_violation)),
    )
    dual_norm = jnp.maximum(_max_abs(stationarity), _max_abs(dual_violation))
    solver_dual_norm = jnp.maximum(
        _max_abs(solver_stationarity), _max_abs(dual_violation)
    )
    complementarity_norm = _max_abs(complementarity)
    kkt_norm = jnp.maximum(
        jnp.maximum(primal_norm, solver_dual_norm), complementarity_norm
    )
    optimality_scale = jnp.maximum(
        1.0,
        jnp.maximum(
            jnp.abs(objective),
            jnp.maximum(_max_abs(program.linear), _max_abs(program.constraint_rhs)),
        ),
    )
    audit_tolerance = (
        policy.termination.absolute + policy.termination.relative * optimality_scale
    )
    finite = (
        jnp.all(jnp.isfinite(primal), axis=-1)
        & jnp.all(jnp.isfinite(slack), axis=-1)
        & jnp.all(jnp.isfinite(dual), axis=-1)
        & jnp.all(jnp.isfinite(lower_dual), axis=-1)
        & jnp.all(jnp.isfinite(upper_dual), axis=-1)
        & jnp.isfinite(kkt_norm)
    )
    converged = finite & (kkt_norm <= audit_tolerance)

    primal_ray_scale = jnp.maximum(1.0, _max_abs(primal))
    primal_ray = primal / primal_ray_scale[..., None]
    quadratic_ray = _max_abs(oe.contract("...ij,...j->...i", quadratic, primal_ray))
    recession_slack = -oe.contract(
        "...ij,...j->...i", program.constraint_matrix, primal_ray
    )
    primal_ray_residual = jnp.maximum(
        quadratic_ray,
        jnp.maximum(
            program.cone.residual(recession_slack),
            _recession_bound_residual(program, primal_ray),
        ),
    )
    primal_ray_objective = jnp.sum(program.linear * primal_ray, axis=-1)
    primal_ray_tolerance = jnp.asarray(
        policy.termination.dual_infeasible,
        dtype=dtype,
    )
    primal_ray_valid = (primal_ray_residual <= primal_ray_tolerance) & (
        primal_ray_objective < -primal_ray_tolerance
    )

    dual_ray_scale = jnp.maximum(
        1.0,
        jnp.maximum(
            _max_abs(dual),
            jnp.maximum(_max_abs(lower_dual), _max_abs(upper_dual)),
        ),
    )
    dual_ray = dual / dual_ray_scale[..., None]
    lower_ray = lower_dual / dual_ray_scale[..., None]
    upper_ray = upper_dual / dual_ray_scale[..., None]
    dual_ray_stationarity = (
        oe.contract("...ji,...j->...i", program.constraint_matrix, dual_ray)
        - lower_ray
        + upper_ray
    )
    lower_term = jnp.where(
        jnp.isfinite(program.lower_bounds), program.lower_bounds * lower_ray, 0.0
    )
    upper_term = jnp.where(
        jnp.isfinite(program.upper_bounds), program.upper_bounds * upper_ray, 0.0
    )
    dual_ray_objective = (
        jnp.sum(program.constraint_rhs * dual_ray, axis=-1)
        - jnp.sum(lower_term, axis=-1)
        + jnp.sum(upper_term, axis=-1)
    )
    dual_ray_residual = jnp.maximum(
        _max_abs(dual_ray_stationarity), program.cone.dual_residual(dual_ray)
    )
    dual_ray_tolerance = jnp.asarray(
        policy.termination.primal_infeasible,
        dtype=dtype,
    )
    dual_ray_valid = (dual_ray_residual <= dual_ray_tolerance) & (
        dual_ray_objective < -dual_ray_tolerance
    )
    status = jnp.where(
        converged,
        int(ConvexProgramStatus.OPTIMAL),
        jnp.where(
            dual_ray_valid,
            int(ConvexProgramStatus.PRIMAL_INFEASIBLE),
            jnp.where(
                primal_ray_valid,
                int(ConvexProgramStatus.DUAL_INFEASIBLE),
                jnp.where(
                    finite,
                    int(ConvexProgramStatus.ITERATION_LIMIT),
                    int(ConvexProgramStatus.NONFINITE_OUTPUT),
                ),
            ),
        ),
    ).astype(jnp.int32)
    certificate = ConvexProgramCertificate(
        primal_ray=primal_ray,
        equality_dual_ray=jnp.empty(program.batch_shape + (0,), dtype=dtype),
        inequality_dual_ray=dual_ray,
        lower_bound_dual_ray=lower_ray,
        upper_bound_dual_ray=upper_ray,
        primal_ray_residual_norm=primal_ray_residual,
        dual_ray_residual_norm=dual_ray_residual,
        primal_ray_objective=primal_ray_objective,
        dual_ray_objective=dual_ray_objective,
        primal_ray_valid=primal_ray_valid,
        dual_ray_valid=dual_ray_valid,
    )
    provenance = ConvexProgramProvenance(
        numeric_version=0,
        problem_id=program.problem_id,
        structure_id=program.structure_id,
        policy_id=policy.policy_id,
        method_id=policy.method.method_id,
        backend="clarabel",
        backend_version=backend_version,
        convexity_evidence=program.convexity_evidence,
        regularization=policy.regularization,
    )
    empty = jnp.empty(program.batch_shape + (0,), dtype=dtype)
    return ConvexProgramResult(
        primal=primal,
        equality_dual=empty,
        inequality_dual=empty,
        inequality_slack=empty,
        cone_slack=slack,
        cone_dual=dual,
        cone_primal_residual=primal_residual,
        cone_violation=cone_violation,
        cone_dual_violation=dual_violation,
        cone_complementarity=cone_complementarity,
        lower_bound_dual=lower_dual,
        upper_bound_dual=upper_dual,
        objective=objective,
        stationarity_residual=stationarity,
        solver_stationarity_residual=solver_stationarity,
        equality_residual=empty,
        inequality_residual=empty,
        inequality_violation=empty,
        complementarity_residual=complementarity,
        primal_residual_norm=primal_norm,
        dual_residual_norm=dual_norm,
        solver_dual_residual_norm=solver_dual_norm,
        complementarity_gap=jnp.sum(complementarity, axis=-1),
        kkt_residual_norm=kkt_norm,
        iterations=iterations,
        backend_converged=backend_status,
        valid=status == int(ConvexProgramStatus.OPTIMAL),
        status=status,
        certificate=certificate,
        provenance=provenance,
        batch_shape=program.batch_shape,
        method=policy.method.method_id,
        backend=f"clarabel-{backend_version}",
        regularization=policy.regularization,
        tolerance=policy.termination.absolute,
        max_iterations=policy.termination.maximum_steps,
    )


def solve_clarabel_program(
    program: ConicProgram,
    policy: ConvexSolvePolicy,
    /,
    *,
    prepared_backend: _PreparedClarabelProgram | None = None,
) -> ConvexProgramResult:
    """Solve a canonical conic program through Clarabel and audit original data."""

    if not isinstance(program, ConicProgram):
        raise TypeError("program must be a ConicProgram.")
    selected = (
        _prepare_structure(program, policy)
        if prepared_backend is None
        else prepared_backend
    )
    if not isinstance(selected, _PreparedClarabelProgram):
        raise TypeError(
            "prepared_backend must be prepared Clarabel program state or None."
        )
    expected_plan = _provider_plan(policy)
    if selected.provider.plan.plan_id != expected_plan.plan_id:
        raise ValueError("Prepared Clarabel state does not match the solve policy.")
    prepared = selected.provider
    structure = selected.structure
    count = int(np.prod(program.batch_shape)) if program.batch_shape else 1
    linear = np.asarray(program.linear).reshape((count, program.num_variables))
    quadratic = (
        np.zeros((count, program.num_variables, program.num_variables))
        if program.quadratic is None
        else np.asarray(program.quadratic).reshape(
            (count, program.num_variables, program.num_variables)
        )
    )
    quadratic = quadratic + policy.regularization * np.eye(program.num_variables)
    primal_values = []
    slack_values = []
    dual_values = []
    lower_values = []
    upper_values = []
    solved_values = []
    iteration_values = []
    for index in range(count):
        matrix, rhs = _transformed_constraint_data(program, structure, index)
        solver = prepared.module.DefaultSolver(
            sp.triu(sp.csc_matrix(quadratic[index])).tocsc(),
            linear[index],
            sp.csc_matrix(matrix),
            rhs,
            list(structure.cones),
            prepared.settings,
        )
        solution = solver.solve()
        primal_values.append(np.asarray(solution.x))
        slack_values.append(
            _restore_cone_vector(
                np.asarray(solution.s), structure, program.num_constraints
            )
        )
        dual_values.append(
            _restore_cone_vector(
                np.asarray(solution.z), structure, program.num_constraints
            )
        )
        lower_dual, upper_dual = _bound_duals(
            np.asarray(solution.z),
            structure,
            program.num_constraints,
            program.num_variables,
        )
        lower_values.append(lower_dual)
        upper_values.append(upper_dual)
        solved_values.append(str(solution.status) == "Solved")
        iteration_values.append(int(solution.iterations))
    shape = program.batch_shape
    primal = jnp.asarray(np.stack(primal_values)).reshape(
        shape + (program.num_variables,)
    )
    slack = jnp.asarray(np.stack(slack_values)).reshape(
        shape + (program.num_constraints,)
    )
    dual = jnp.asarray(np.stack(dual_values)).reshape(shape + (program.num_constraints,))
    lower_dual = jnp.asarray(np.stack(lower_values)).reshape(
        shape + (program.num_variables,)
    )
    upper_dual = jnp.asarray(np.stack(upper_values)).reshape(
        shape + (program.num_variables,)
    )
    return _audit_result(
        program,
        primal,
        slack,
        dual,
        lower_dual,
        upper_dual,
        jnp.asarray(solved_values, dtype=bool).reshape(shape),
        jnp.asarray(iteration_values, dtype=jnp.int32).reshape(shape),
        policy,
        prepared.backend_version,
    )


__all__ = ["prepare_clarabel_policy", "solve_clarabel_program"]
