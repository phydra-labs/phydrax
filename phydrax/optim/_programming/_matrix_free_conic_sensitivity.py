#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._strict import StrictModule
from ...linalg import (
    adjoint,
    FailureMode,
    JacobianLinearOperator,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveResult,
    prepare_linearization,
    solve as solve_linear,
    StabilityLowerBound,
)
from ._cones import AbstractConvexCone, NonnegativeCone, ProductCone
from ._conic_sensitivity import (
    _active_set_evidence,
    _resolve_status,
    ConicActiveSetEvidence,
    ConicProgramData,
    ConicSensitivityResult,
    ConicSensitivityStatus,
)
from ._exponential_cone import ExponentialCone
from ._lifecycle import ConvexProgramExecution, PreparedConvexProgram
from ._policy import ConicGeneralizedDerivativePolicy
from ._power_cone import PowerCone
from ._problem import (
    _conic_matrix_mv,
    _conic_matrix_transpose_mv,
    _conic_quadratic_mv,
    ConicProgram,
)


class PreparedMatrixFreeConicSensitivity(StrictModule):
    """Audited state for matrix-free projection-KKT sensitivities of one program."""

    original_data: ConicProgramData
    state: Array
    operator: JacobianLinearOperator
    cone: AbstractConvexCone
    stability: StabilityLowerBound
    active_set: ConicActiveSetEvidence
    linear_policy: LinearSolvePolicy
    numeric_version: Array
    generalized: ConicGeneralizedDerivativePolicy | None
    num_variables: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    regularity_tolerance: float = eqx.field(static=True)
    failure_mode: FailureMode = eqx.field(static=True)
    convex_plan_id: str = eqx.field(static=True)
    numeric_binding_id: str = eqx.field(static=True)


def _selected_projection(
    cone: AbstractConvexCone,
    value: Array,
    generalized: ConicGeneralizedDerivativePolicy | None,
) -> Array:
    if generalized is None:
        return cone.project_dual(value)
    policy = generalized

    @jax.custom_jvp
    def selected(candidate: Array) -> Array:
        return cone.project_dual(candidate)

    @selected.defjvp
    def selected_jvp(
        primals: tuple[Array], tangents: tuple[Array]
    ) -> tuple[Array, Array]:
        (candidate,), (candidate_dot,) = primals, tangents
        shifted = candidate
        if policy.approach_direction:
            if len(policy.approach_direction) != cone.dimension:
                raise ValueError("approach_direction must match cone dimension.")
            shifted = candidate + policy.approach_scale * jnp.asarray(
                policy.approach_direction, dtype=candidate.dtype
            )
        jacobian = jax.jacfwd(cone.project_dual)(shifted)
        blocks = cone.cones if isinstance(cone, ProductCone) else (cone,)
        slices = (
            cone.slices if isinstance(cone, ProductCone) else (slice(0, cone.dimension),)
        )
        for block, block_slice in zip(blocks, slices, strict=True):
            if isinstance(block, NonnegativeCone):
                indices = jnp.arange(block_slice.start, block_slice.stop)
                diagonal = jnp.where(
                    candidate[indices] > 0.0,
                    1.0,
                    jnp.where(
                        candidate[indices] < 0.0,
                        0.0,
                        policy.orthant_zero_value,
                    ),
                )
                jacobian = jacobian.at[indices, indices].set(diagonal)
        return cone.project_dual(candidate), jacobian @ candidate_dot

    return selected(value)


def _residual(
    data: ConicProgramData,
    state: Array,
    cone: AbstractConvexCone,
    variables: int,
    regularization: float,
    generalized: ConicGeneralizedDerivativePolicy | None,
) -> Array:
    primal = state[:variables]
    dual = state[variables:]
    projection_point = (
        dual + _conic_matrix_mv(data.constraint_matrix, primal) - data.constraint_rhs
    )
    # The executed map includes the solve policy's explicit regularization.
    stationarity = (
        _conic_quadratic_mv(data.quadratic, primal)
        + regularization * primal
        + data.linear
        + _conic_matrix_transpose_mv(data.constraint_matrix, dual)
    )
    projection = _selected_projection(cone, projection_point, generalized)
    return jnp.concatenate((stationarity, dual - projection))


def prepare_matrix_free_conic_sensitivity(
    prepared: PreparedConvexProgram,
    execution: ConvexProgramExecution,
    /,
    *,
    linear: LinearSolvePolicy,
    stability: Callable[[JacobianLinearOperator], StabilityLowerBound] | None,
    regularity_tolerance: float,
    generalized: ConicGeneralizedDerivativePolicy | None,
    failure_mode: FailureMode,
    fixed_active_set: ConicActiveSetEvidence | None,
) -> PreparedMatrixFreeConicSensitivity:
    program = prepared.program
    if not isinstance(program, ConicProgram) or program.batch_shape:
        raise ValueError("Matrix-free sensitivity requires one ConicProgram.")
    if (
        program.fixed_bound_indices
        or program.lower_bound_indices
        or program.upper_bound_indices
    ):
        raise ValueError(
            "Matrix-free sensitivity requires bounds expressed as cone rows."
        )
    if not callable(stability):
        raise TypeError("stability must build evidence for the exact Jacobian.")
    if generalized is not None and not isinstance(
        generalized, ConicGeneralizedDerivativePolicy
    ):
        raise TypeError("generalized has the wrong policy type.")
    data = ConicProgramData(
        program.quadratic,
        program.linear,
        program.constraint_matrix,
        program.constraint_rhs,
        program.lower_bounds,
        program.upper_bounds,
    )
    policy = prepared.plan.policy
    regularization = policy.regularization
    result = execution.result
    state = jnp.concatenate((result.primal, result.cone_dual))
    linearization = prepare_linearization(
        lambda candidate: _residual(
            data,
            candidate,
            program.cone,
            program.num_variables,
            regularization,
            generalized,
        ),
        state,
        linearization_id=f"conic-projection-kkt:{prepared.numeric_binding_id}",
    )
    operator = JacobianLinearOperator(linearization)
    evidence = stability(operator)
    if (
        not isinstance(evidence, StabilityLowerBound)
        or evidence.evidence not in ("construction", "verified")
        or not evidence.matches(operator)
    ):
        raise ValueError("Matching constructive/verified stability evidence is required.")
    residual = linearization.primal
    root_norm = jnp.max(jnp.abs(residual), initial=0.0)
    active_set = _active_set_evidence(
        program,
        result,
        regularization=regularization,
        termination=policy.termination,
        tolerance=regularity_tolerance,
        projection_residual_norm=root_norm,
        projection_finite=jnp.all(jnp.isfinite(state)) & jnp.isfinite(root_norm),
        kkt_nonsingular=evidence.valid,
        reference=fixed_active_set,
    )
    if generalized is not None:
        blocks = (
            program.cone.cones
            if isinstance(program.cone, ProductCone)
            else (program.cone,)
        )
        ambiguous = bool(
            active_set.status == int(ConicSensitivityStatus.AMBIGUOUS_ACTIVE_SET)
        )
        if ambiguous and any(
            isinstance(block, (ExponentialCone, PowerCone)) for block in blocks
        ):
            raise ValueError(
                "Nonsmooth exponential/power generalized strata are unsupported."
            )
    return PreparedMatrixFreeConicSensitivity(
        data,
        state,
        operator,
        program.cone,
        evidence,
        active_set,
        linear,
        prepared.numeric_version,
        generalized,
        num_variables=program.num_variables,
        regularization=regularization,
        regularity_tolerance=regularity_tolerance,
        failure_mode=failure_mode,
        convex_plan_id=prepared.plan.plan_id,
        numeric_binding_id=prepared.numeric_binding_id,
    )


def _resolution(
    prepared: PreparedMatrixFreeConicSensitivity,
    linear_result: LinearSolveResult,
    residual: Array,
) -> tuple[Array, Array]:
    # With the certified stability constant sigma, the exact derivative system
    # solution differs from the computed one by at most ||residual|| / sigma.
    # A positive but numerically negligible sigma therefore never certifies a
    # least-squares solution of a (near-)singular system.
    value = linear_result.value
    error_bound = jnp.linalg.norm(residual) / prepared.stability.lower_bound
    certified = error_bound <= prepared.regularity_tolerance * jnp.maximum(
        1.0, jnp.linalg.norm(value)
    )
    linear_regular = (
        linear_result.successful
        & linear_result.diagnostics.finite
        & linear_result.diagnostics.converged
        & jnp.all(jnp.isfinite(value))
        & certified
    )
    return _resolve_status(
        prepared.active_set.status, linear_regular, prepared.generalized is not None
    )


def _data_residual(
    prepared: PreparedMatrixFreeConicSensitivity,
) -> tuple[Callable[[ConicProgramData], Array], ConicProgramData]:
    # Sparse operators carry integer topology leaves; only inexact numerical
    # coordinates are differentiated, the fixed topology is closed over.
    numeric, topology = eqx.partition(prepared.original_data, eqx.is_inexact_array)

    def residual(candidate: ConicProgramData) -> Array:
        return _residual(
            eqx.combine(candidate, topology),
            prepared.state,
            prepared.cone,
            prepared.num_variables,
            prepared.regularization,
            prepared.generalized,
        )

    return residual, numeric


def matrix_free_conic_primal_jvp(
    prepared: PreparedMatrixFreeConicSensitivity, tangent: ConicProgramData
) -> ConicSensitivityResult:
    if not isinstance(tangent, ConicProgramData):
        raise TypeError("tangent must be a ConicProgramData.")
    residual, numeric = _data_residual(prepared)
    tangent_numeric, _ = eqx.partition(tangent, eqx.is_inexact_array)
    tangent_finite = jnp.all(
        jnp.stack(
            [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tangent_numeric)]
        )
    )
    _, action = jax.jvp(residual, (numeric,), (tangent_numeric,))
    action = eqx.error_if(action, ~tangent_finite, "Conic tangent must be finite.")
    linear_result = solve_linear(
        LeastSquaresProblem(prepared.operator, problem_id="matrix-free-conic-jvp"),
        -action,
        policy=prepared.linear_policy,
    )
    status, available = _resolution(
        prepared,
        linear_result,
        jnp.asarray(prepared.operator.mv(linear_result.value)) + action,
    )
    value = jnp.where(available, linear_result.value[: prepared.num_variables], jnp.nan)
    if prepared.failure_mode == "error":
        value = eqx.error_if(
            value, ~available, "Conic matrix-free JVP has no available derivative."
        )
    return _result(prepared, linear_result, status, available, value)


def matrix_free_conic_primal_vjp(
    prepared: PreparedMatrixFreeConicSensitivity, cotangent: ArrayLike
) -> ConicSensitivityResult:
    cotangent_ = jnp.asarray(cotangent, dtype=prepared.state.dtype)
    if cotangent_.shape != (prepared.num_variables,):
        raise ValueError("cotangent has the wrong shape.")
    cotangent_ = eqx.error_if(
        cotangent_,
        jnp.any(~jnp.isfinite(cotangent_)),
        "Conic primal cotangent must be finite.",
    )
    state_cotangent = jnp.concatenate(
        (
            cotangent_,
            jnp.zeros(
                prepared.state.shape[0] - prepared.num_variables,
                dtype=prepared.state.dtype,
            ),
        )
    )
    adjoint_operator = adjoint(prepared.operator)
    linear_result = solve_linear(
        LeastSquaresProblem(adjoint_operator, problem_id="matrix-free-conic-vjp"),
        state_cotangent,
        policy=prepared.linear_policy,
    )
    residual, numeric = _data_residual(prepared)
    _, pullback = jax.vjp(residual, numeric)
    status, available = _resolution(
        prepared,
        linear_result,
        jnp.asarray(adjoint_operator.mv(linear_result.value)) - state_cotangent,
    )
    numeric_cotangent = jax.tree.map(
        lambda leaf: jnp.where(available, -leaf, jnp.full_like(leaf, jnp.nan)),
        pullback(linear_result.value)[0],
    )
    if prepared.failure_mode == "error":
        leaves, structure = jax.tree.flatten(numeric_cotangent)
        leaves[0] = eqx.error_if(
            leaves[0], ~available, "Conic matrix-free VJP has no available derivative."
        )
        numeric_cotangent = jax.tree.unflatten(structure, leaves)
    _, topology = eqx.partition(prepared.original_data, eqx.is_inexact_array)
    value = eqx.combine(numeric_cotangent, topology)
    return _result(prepared, linear_result, status, available, value)


def _result(
    prepared: PreparedMatrixFreeConicSensitivity,
    linear_result: LinearSolveResult,
    status: Array,
    available: Array,
    value: PyTree[Array],
) -> ConicSensitivityResult:
    return ConicSensitivityResult(
        value,
        status,
        available,
        prepared.active_set,
        linear_result.status,
        linear_result.diagnostics,
        prepared.numeric_version,
        convex_plan_id=prepared.convex_plan_id,
        linear_plan_id=linear_result.provenance.plan_id,
        representation="matrix-free",
        generalized_selection=(
            "smooth"
            if prepared.generalized is None
            else (
                f"orthant={prepared.generalized.orthant_zero_value};approach={prepared.generalized.approach_direction}"
            )
        ),
        numeric_binding_id=prepared.numeric_binding_id,
    )


__all__ = [
    "PreparedMatrixFreeConicSensitivity",
    "matrix_free_conic_primal_jvp",
    "matrix_free_conic_primal_vjp",
    "prepare_matrix_free_conic_sensitivity",
]
