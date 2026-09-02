#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import (
    adjoint,
    JacobianLinearOperator,
    LeastSquaresProblem,
    LinearSolvePolicy,
    prepare_linearization,
    solve as solve_linear,
    StabilityLowerBound,
)
from ._cones import AbstractConvexCone, NonnegativeCone, ProductCone
from ._conic_sensitivity import ConicProgramData, ConicSensitivityResult
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
    original_data: ConicProgramData
    state: Array
    operator: JacobianLinearOperator
    cone: AbstractConvexCone
    stability: StabilityLowerBound
    forward_valid: Array
    projection_margin: Array
    projection_regular: Array
    root_residual_norm: Array
    linear_policy: LinearSolvePolicy
    numeric_version: Array
    generalized: ConicGeneralizedDerivativePolicy | None
    num_variables: int = eqx.field(static=True)
    regularity_tolerance: float = eqx.field(static=True)
    failure_mode: str = eqx.field(static=True)
    convex_plan_id: str = eqx.field(static=True)
    numeric_binding_id: str = eqx.field(static=True)


def _selected_projection(cone, value, generalized):
    if generalized is None:
        return cone.project_dual(value)

    @jax.custom_jvp
    def selected(candidate):
        return cone.project_dual(candidate)

    @selected.defjvp
    def selected_jvp(primals, tangents):
        (candidate,), (candidate_dot,) = primals, tangents
        shifted = candidate
        if generalized.approach_direction:
            if len(generalized.approach_direction) != cone.dimension:
                raise ValueError("approach_direction must match cone dimension.")
            shifted = candidate + generalized.approach_scale * jnp.asarray(
                generalized.approach_direction, dtype=candidate.dtype
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
                        generalized.orthant_zero_value,
                    ),
                )
                jacobian = jacobian.at[indices, indices].set(diagonal)
        return cone.project_dual(candidate), jacobian @ candidate_dot

    return selected(value)


def _residual(data, state, cone, variables, generalized):
    primal = state[:variables]
    dual = state[variables:]
    projection_point = (
        dual + _conic_matrix_mv(data.constraint_matrix, primal) - data.constraint_rhs
    )
    stationarity = (
        _conic_quadratic_mv(data.quadratic, primal)
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
    stability: Callable[[JacobianLinearOperator], StabilityLowerBound],
    regularity_tolerance: float,
    generalized: ConicGeneralizedDerivativePolicy | None,
    failure_mode: str,
) -> PreparedMatrixFreeConicSensitivity:
    program = prepared.program
    if not isinstance(program, ConicProgram) or program.batch_shape:
        raise ValueError("Matrix-free sensitivity requires one ConicProgram.")
    if (
        program.fixed_bound_indices
        or program.lower_bound_indices
        or program.upper_bound_indices
    ):
        raise ValueError("Matrix-free sensitivity currently requires no bounds.")
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
    result = execution.result
    state = jnp.concatenate((result.primal, result.cone_dual))
    linearization = prepare_linearization(
        lambda candidate: _residual(
            data,
            candidate,
            program.cone,
            program.num_variables,
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
    point = (
        result.cone_dual
        + _conic_matrix_mv(program.constraint_matrix, result.primal)
        - program.constraint_rhs
    )
    margin = program.cone.dual_projection_smoothness_margin(point)
    scale = jnp.maximum(jnp.max(jnp.abs(point), initial=0.0), 1.0)
    if generalized is not None:
        blocks = (
            program.cone.cones
            if isinstance(program.cone, ProductCone)
            else (program.cone,)
        )
        if any(isinstance(block, (ExponentialCone, PowerCone)) for block in blocks):
            if bool(margin <= regularity_tolerance * scale):
                raise ValueError(
                    "Nonsmooth exponential/power generalized strata are unsupported."
                )
    projection_regular = (margin > regularity_tolerance * scale) | (
        generalized is not None
    )
    termination = prepared.plan.policy.termination
    data_scale = jnp.maximum(
        1.0,
        jnp.maximum(
            jnp.max(jnp.abs(program.linear), initial=0.0),
            jnp.max(jnp.abs(program.constraint_rhs), initial=0.0),
        ),
    )
    forward_valid = (
        result.successful
        & jnp.all(jnp.isfinite(state))
        & (root_norm <= termination.absolute + termination.relative * data_scale)
    )
    return PreparedMatrixFreeConicSensitivity(
        data,
        state,
        operator,
        program.cone,
        evidence,
        forward_valid,
        margin,
        projection_regular,
        root_norm,
        linear,
        prepared.numeric_version,
        generalized,
        program.num_variables,
        regularity_tolerance,
        failure_mode,
        prepared.plan.plan_id,
        prepared.numeric_binding_id,
    )


def _regular(prepared, linear_result):
    return (
        prepared.forward_valid
        & prepared.projection_regular
        & prepared.stability.valid
        & linear_result.successful
        & linear_result.diagnostics.finite
        & linear_result.diagnostics.converged
        & jnp.all(jnp.isfinite(linear_result.value))
    )


def matrix_free_conic_primal_jvp(prepared, tangent):
    _, action = jax.jvp(
        lambda data: _residual(
            data,
            prepared.state,
            prepared.cone,
            prepared.num_variables,
            prepared.generalized,
        ),
        (prepared.original_data,),
        (tangent,),
    )
    linear_result = solve_linear(
        LeastSquaresProblem(prepared.operator, problem_id="matrix-free-conic-jvp"),
        -action,
        policy=prepared.linear_policy,
    )
    regular = _regular(prepared, linear_result)
    value = jnp.where(regular, linear_result.value[: prepared.num_variables], jnp.nan)
    if prepared.failure_mode == "error":
        value = eqx.error_if(value, ~regular, "Conic matrix-free JVP is not regular.")
    return _result(prepared, linear_result, regular, value)


def matrix_free_conic_primal_vjp(prepared, cotangent: ArrayLike):
    cotangent_ = jnp.asarray(cotangent, dtype=prepared.state.dtype)
    if cotangent_.shape != (prepared.num_variables,):
        raise ValueError("cotangent has the wrong shape.")
    state_cotangent = jnp.concatenate(
        (
            cotangent_,
            jnp.zeros(
                prepared.state.shape[0] - prepared.num_variables,
                dtype=prepared.state.dtype,
            ),
        )
    )
    linear_result = solve_linear(
        LeastSquaresProblem(
            adjoint(prepared.operator), problem_id="matrix-free-conic-vjp"
        ),
        state_cotangent,
        policy=prepared.linear_policy,
    )
    _, pullback = jax.vjp(
        lambda data: _residual(
            data,
            prepared.state,
            prepared.cone,
            prepared.num_variables,
            prepared.generalized,
        ),
        prepared.original_data,
    )
    value = jax.tree.map(jnp.negative, pullback(linear_result.value)[0])
    regular = _regular(prepared, linear_result)
    value = jax.tree.map(
        lambda leaf: (
            jnp.where(regular, leaf, jnp.full_like(leaf, jnp.nan))
            if jnp.issubdtype(leaf.dtype, jnp.inexact)
            else leaf
        ),
        value,
    )
    if prepared.failure_mode == "error":
        leaves, structure = jax.tree.flatten(value)
        leaves[0] = eqx.error_if(
            leaves[0], ~regular, "Conic matrix-free VJP is not regular."
        )
        value = jax.tree.unflatten(structure, leaves)
    return _result(prepared, linear_result, regular, value)


def _result(prepared, linear_result, regular, value):
    return ConicSensitivityResult(
        value,
        prepared.forward_valid,
        prepared.projection_margin,
        prepared.projection_regular,
        prepared.root_residual_norm,
        linear_result.status,
        linear_result.diagnostics,
        regular,
        prepared.numeric_version,
        convex_plan_id=prepared.convex_plan_id,
        linear_plan_id=linear_result.provenance.plan_id,
        representation="matrix-free",
        generalized_selection=(
            "smooth"
            if prepared.generalized is None
            else (
                f"orthant={prepared.generalized.orthant_zero_value};"
                f"approach={prepared.generalized.approach_direction}"
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
