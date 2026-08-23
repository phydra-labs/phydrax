#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    prepare as prepare_linear,
    solve as solve_linear,
)
from ._constrained_model import prepare_constrained_model
from ._iterative import MinimizationProblem


ConstrainedSensitivityMode: TypeAlias = Literal["fixed-active", "barrier"]


class ConstrainedSensitivityResult(StrictModule):
    value: PyTree[Array]
    condition_estimate: Array
    active_constraints: Array
    regular: Array
    mode: ConstrainedSensitivityMode = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    linear_plan_id: str = eqx.field(static=True)


def _linear_policy(
    linear: LinearSolvePolicy | None,
    precision: NonlinearPrecisionPolicy,
    /,
) -> LinearSolvePolicy:
    linear_ = LinearSolvePolicy(DenseSVD()) if linear is None else linear
    if not isinstance(linear_, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    return precision.bind_linear(linear_)


def _least_squares(matrix, right, linear, precision, /):
    return solve_linear(
        LeastSquaresProblem(DenseLinearOperator(precision.accumulation(matrix))),
        precision.accumulation(right),
        policy=_linear_policy(linear, precision),
    )


def _sensitivity_system(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    mode: ConstrainedSensitivityMode,
    active_tolerance: float,
    barrier: float,
    linear: LinearSolvePolicy | None,
    precision: NonlinearPrecisionPolicy,
):
    model_parameters = precision.state(parameters)
    prepared = prepare_constrained_model(problem, model_parameters, args=args)
    evaluation = prepared.evaluate(model_parameters, args)
    coordinates = evaluation.coordinates
    equality_count = evaluation.equalities.size
    if mode == "fixed-active":
        active_mask = evaluation.inequality_slacks <= active_tolerance
    else:
        active_mask = jnp.ones_like(evaluation.inequality_slacks, dtype=jnp.bool_)
    lower_jacobian = evaluation.constraint_jacobian[prepared.lower_indices]
    upper_jacobian = -evaluation.constraint_jacobian[prepared.upper_indices]
    inequality_jacobian = jnp.concatenate([lower_jacobian, upper_jacobian], axis=0)
    active_jacobian = inequality_jacobian[active_mask]
    equality_jacobian = evaluation.constraint_jacobian[prepared.equality_indices]
    if mode == "fixed-active":
        multiplier_matrix = jnp.concatenate(
            [jnp.conj(equality_jacobian.T), -jnp.conj(active_jacobian.T)],
            axis=1,
        )
        multipliers = (
            _least_squares(
                multiplier_matrix,
                -evaluation.gradient,
                linear,
                precision,
            ).value
            if multiplier_matrix.shape[1]
            else jnp.empty((0,), dtype=evaluation.gradient.dtype)
        )
        equality_multipliers = multipliers[:equality_count]
        active_multipliers = multipliers[equality_count:]
    else:
        equality_multipliers = (
            _least_squares(
                jnp.conj(equality_jacobian.T),
                -evaluation.gradient,
                linear,
                precision,
            ).value
            if equality_jacobian.shape[0]
            else jnp.empty((0,), dtype=evaluation.gradient.dtype)
        )
        active_multipliers = barrier / jnp.maximum(
            evaluation.inequality_slacks[active_mask], 1e-12
        )
    initial = jnp.concatenate([coordinates, equality_multipliers, active_multipliers])

    def residual(combined, current_args):
        x = combined[: coordinates.size]
        equality_dual = combined[coordinates.size : coordinates.size + equality_count]
        inequality_dual = combined[coordinates.size + equality_count :]
        point = prepared.unflatten(x)
        current = prepared.evaluate(point, current_args)
        lower_j = current.constraint_jacobian[prepared.lower_indices]
        upper_j = -current.constraint_jacobian[prepared.upper_indices]
        inequality_j = jnp.concatenate([lower_j, upper_j], axis=0)[active_mask]
        equality_j = current.constraint_jacobian[prepared.equality_indices]
        stationarity = (
            current.gradient
            + jnp.conj(equality_j.T) @ equality_dual
            - jnp.conj(inequality_j.T) @ inequality_dual
        )
        if mode == "fixed-active":
            active_values = current.inequality_slacks[active_mask]
        else:
            active_values = (
                current.inequality_slacks[active_mask] * inequality_dual - barrier
            )
        return jnp.concatenate([stationarity, current.equalities, active_values])

    matrix = jax.jacfwd(lambda value: residual(value, args))(initial)
    system = prepare_linear(
        LeastSquaresProblem(DenseLinearOperator(precision.accumulation(matrix))),
        _linear_policy(linear, precision),
    )
    return (
        prepared,
        initial,
        residual,
        matrix,
        system,
        active_mask,
        model_parameters,
        evaluation.gradient,
    )


def constrained_solution_jvp(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    tangent_args: Any,
    /,
    *,
    mode: ConstrainedSensitivityMode = "fixed-active",
    active_tolerance: float = 1e-7,
    barrier: float = 1e-8,
    linear: LinearSolvePolicy | None = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> ConstrainedSensitivityResult:
    if mode not in ("fixed-active", "barrier"):
        raise ValueError("Unknown constrained sensitivity mode.")
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    (
        prepared,
        initial,
        residual,
        matrix,
        system,
        active_mask,
        model_parameters,
        gradient,
    ) = _sensitivity_system(
        problem,
        parameters,
        args,
        mode,
        active_tolerance,
        barrier,
        linear,
        precision_,
    )
    _, argument_action = jax.jvp(
        lambda current_args: residual(initial, current_args),
        (args,),
        (tangent_args,),
    )
    linear_result = solve_linear(
        system,
        precision_.accumulation(-argument_action),
    )
    direction = precision_.direction(linear_result.value)
    condition = precision_.decision(linear_result.diagnostics.condition_estimate)
    regular = jnp.isfinite(condition) & (condition < 1e12)
    tangent = prepared.unflatten(direction[: prepared.template_coordinates.size])
    tangent = jax.tree.map(
        lambda value: jnp.where(regular, value, jnp.full_like(value, jnp.nan)),
        tangent,
    )
    return ConstrainedSensitivityResult(
        tangent,
        condition,
        jnp.sum(active_mask, dtype=jnp.int32),
        regular,
        mode=mode,
        precision_evidence=precision_.evidence_for(
            model_parameters,
            gradient,
        ),
        linear_plan_id=linear_result.provenance.plan_id,
    )


def constrained_solution_vjp(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    cotangent_parameters: PyTree[Any],
    /,
    *,
    mode: ConstrainedSensitivityMode = "fixed-active",
    active_tolerance: float = 1e-7,
    barrier: float = 1e-8,
    linear: LinearSolvePolicy | None = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> ConstrainedSensitivityResult:
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    (
        prepared,
        initial,
        residual,
        matrix,
        _,
        active_mask,
        model_parameters,
        gradient,
    ) = _sensitivity_system(
        problem,
        parameters,
        args,
        mode,
        active_tolerance,
        barrier,
        linear,
        precision_,
    )
    cotangent, _ = ravel_pytree(cotangent_parameters)
    right = jnp.concatenate(
        [
            cotangent,
            jnp.zeros((matrix.shape[0] - cotangent.size,), dtype=cotangent.dtype),
        ]
    )
    linear_result = _least_squares(
        jnp.conj(matrix.T),
        right,
        linear,
        precision_,
    )
    adjoint = jnp.asarray(linear_result.value, dtype=initial.dtype)
    condition = precision_.decision(linear_result.diagnostics.condition_estimate)
    regular = jnp.isfinite(condition) & (condition < 1e12)
    _, pullback = jax.vjp(lambda current_args: residual(initial, current_args), args)
    argument_cotangent = jax.tree.map(jnp.negative, pullback(adjoint)[0])
    argument_cotangent = jax.tree.map(
        lambda value: jnp.where(regular, value, jnp.full_like(value, jnp.nan)),
        argument_cotangent,
    )
    return ConstrainedSensitivityResult(
        argument_cotangent,
        condition,
        jnp.sum(active_mask, dtype=jnp.int32),
        regular,
        mode=mode,
        precision_evidence=precision_.evidence_for(
            model_parameters,
            gradient,
        ),
        linear_plan_id=linear_result.provenance.plan_id,
    )


__all__ = [
    "ConstrainedSensitivityMode",
    "ConstrainedSensitivityResult",
    "constrained_solution_jvp",
    "constrained_solution_vjp",
]
