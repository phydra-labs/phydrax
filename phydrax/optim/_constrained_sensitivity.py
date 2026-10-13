#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree

from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    AdjointLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    DenseSVD,
    FunctionLinearOperator,
    GeneralizedLSMR,
    inverse_residual_certificate,
    InverseResidualCertificate,
    JacobianLinearOperator,
    LeastSquaresProblem,
    LinearSolveDiagnostics,
    LinearSolvePolicy,
    LinearSolveResult,
    MaterializationPolicy,
    materialize,
    prepare as prepare_linear,
    prepare_linearization,
    PreparedLinearSolve,
    solve as solve_linear,
    solve_many as solve_linear_many,
    TolerancePolicy,
)
from ..typing import parse
from ._constrained_model import (
    prepare_constrained_model_actions,
    PreparedConstrainedModel,
)
from ._iterative import MinimizationProblem


ConstrainedSensitivityMode: TypeAlias = Literal["fixed-active", "barrier"]
ConstrainedSensitivityRoute: TypeAlias = Literal["matrix-free", "dense-svd"]
_SensitivitySystem: TypeAlias = tuple[
    PreparedConstrainedModel,
    Array,
    Callable[[Array, Any], Array],
    AbstractLinearOperator,
    PreparedLinearSolve,
    Array,
    PyTree[Any],
    Array,
]


class ConstrainedSensitivityResult(StrictModule):
    value: PyTree[Array]
    condition_estimate: Array
    active_constraints: Array
    regular: Array
    mode: ConstrainedSensitivityMode = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    linear_plan_id: str = eqx.field(static=True)
    route: ConstrainedSensitivityRoute = eqx.field(static=True)
    linear_diagnostics: LinearSolveDiagnostics
    regularity_evidence: InverseResidualCertificate | None


def _linear_policy(
    linear: LinearSolvePolicy | None,
    precision: NonlinearPrecisionPolicy,
    /,
    *,
    route: ConstrainedSensitivityRoute = "matrix-free",
    materialization: MaterializationPolicy | None = None,
) -> LinearSolvePolicy:
    if linear is None:
        match route:
            case "matrix-free":
                linear = LinearSolvePolicy(
                    GeneralizedLSMR(condition_limit=1e12),
                    tolerance=TolerancePolicy(
                        relative=1e-10, absolute=1e-12, max_steps=1000
                    ),
                )
            case "dense-svd":
                linear = LinearSolvePolicy(DenseSVD(), materialization=materialization)
    if not isinstance(linear, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    if route == "matrix-free" and isinstance(linear.method, DenseSVD):
        raise ValueError("Dense sensitivity requires route='dense-svd'.")
    if route == "dense-svd" and not isinstance(linear.method, DenseSVD):
        raise ValueError("route='dense-svd' requires a DenseSVD linear policy.")
    return precision.bind_linear(linear)


def _prepare_system(
    operator: AbstractLinearOperator,
    policy: LinearSolvePolicy,
    route: ConstrainedSensitivityRoute,
    precision: NonlinearPrecisionPolicy,
    /,
) -> PreparedLinearSolve:
    if route == "dense-svd":
        matrix_dtype = jax.tree.leaves(operator.target.structure())[0].dtype
        if precision.accumulation_dtype is not None:
            matrix_dtype = jnp.result_type(matrix_dtype, precision.accumulation_dtype)
        policy.materialization.require(
            operator.source.size * operator.target.size, matrix_dtype
        )
        operator = DenseLinearOperator(
            materialize(operator, policy.materialization).astype(matrix_dtype)
        )
    elif policy.method.name.startswith("dense-"):
        raise ValueError("Dense sensitivity requires route='dense-svd'.")
    return prepare_linear(LeastSquaresProblem(operator), policy)


def _linear_rhs(system: PreparedLinearSolve, value: Array, /) -> Array:
    dtype = jax.tree.leaves(system.problem.operator.target.structure())[0].dtype
    return jnp.asarray(value, dtype=dtype)


def _least_squares(
    operator: AbstractLinearOperator,
    right: Array,
    policy: LinearSolvePolicy,
    route: ConstrainedSensitivityRoute,
    precision: NonlinearPrecisionPolicy,
    /,
) -> LinearSolveResult:
    system = _prepare_system(operator, policy, route, precision)
    return solve_linear(system, _linear_rhs(system, right))


def _sensitivity_system(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    mode: ConstrainedSensitivityMode,
    active_tolerance: float,
    barrier: float,
    linear: LinearSolvePolicy | None,
    precision: NonlinearPrecisionPolicy,
    *,
    route: ConstrainedSensitivityRoute = "matrix-free",
    materialization: MaterializationPolicy | None = None,
    adjoint: bool = False,
) -> _SensitivitySystem:
    mode = parse(mode, ConstrainedSensitivityMode, "mode")
    route = parse(route, ConstrainedSensitivityRoute, "route")
    policy = _linear_policy(
        linear, precision, route=route, materialization=materialization
    )
    model_parameters = precision.state(parameters)
    prepared = prepare_constrained_model_actions(problem, model_parameters, args=args)
    evaluation = prepared.evaluate_actions(model_parameters, args)
    values = evaluation.values
    coordinates = values.coordinates
    equality_count = values.equalities.size
    active_mask = (
        values.inequality_slacks <= active_tolerance
        if mode == "fixed-active"
        else jnp.ones_like(values.inequality_slacks, dtype=jnp.bool_)
    )
    # Active ownership is immutable host preparation, not a runtime-sized selection.
    active_indices = jnp.asarray(jnp.flatnonzero(active_mask), dtype=jnp.int32)
    active_count = active_indices.size
    lower_count = prepared.lower_indices.size

    def constraint_dual(equality: Array, inequality: Array) -> Array:
        all_inequality = (
            jnp.zeros_like(values.inequality_slacks)
            .at[active_indices]
            .set(inequality, unique_indices=True)
        )
        raw_dual = jnp.zeros_like(values.raw_constraints)
        raw_dual = raw_dual.at[prepared.equality_indices].add(equality)
        raw_dual = raw_dual.at[prepared.lower_indices].add(-all_inequality[:lower_count])
        return raw_dual.at[prepared.upper_indices].add(all_inequality[lower_count:])

    def multiplier_action(dual: Array) -> Array:
        return evaluation.constraint_operator.adjoint_mv(
            constraint_dual(dual[:equality_count], dual[equality_count:])
        )

    dual_space = ArraySpace((equality_count + active_count,), dtype=coordinates.dtype)
    coordinate_space = ArraySpace(coordinates.shape, dtype=coordinates.dtype)
    multiplier_operator = FunctionLinearOperator(
        multiplier_action,
        source=dual_space,
        target=coordinate_space,
    )
    if mode == "fixed-active":
        multipliers = (
            _least_squares(
                multiplier_operator, -evaluation.gradient, policy, route, precision
            ).value
            if dual_space.size
            else dual_space.zeros()
        )
        equality_multipliers = multipliers[:equality_count]
        active_multipliers = multipliers[equality_count:]
    else:
        active_multipliers = barrier / jnp.maximum(
            values.inequality_slacks[active_indices], 1e-12
        )
        equality_right = -evaluation.gradient - multiplier_action(
            jnp.concatenate(
                (
                    jnp.zeros((equality_count,), dtype=coordinates.dtype),
                    active_multipliers,
                )
            )
        )
        equality_operator = FunctionLinearOperator(
            lambda dual: multiplier_action(
                jnp.concatenate(
                    (dual, jnp.zeros((active_count,), dtype=coordinates.dtype))
                )
            ),
            source=ArraySpace((equality_count,), dtype=coordinates.dtype),
            target=coordinate_space,
        )
        equality_multipliers = (
            _least_squares(
                equality_operator, equality_right, policy, route, precision
            ).value
            if equality_count
            else jnp.empty((0,), dtype=coordinates.dtype)
        )
    initial = jnp.concatenate(
        (
            coordinates,
            jnp.asarray(equality_multipliers, dtype=coordinates.dtype),
            jnp.asarray(active_multipliers, dtype=coordinates.dtype),
        )
    )

    def residual(combined: Array, current_args: Any) -> Array:
        x = combined[: coordinates.size]
        equality_dual = combined[coordinates.size : coordinates.size + equality_count]
        inequality_dual = combined[coordinates.size + equality_count :]
        current = prepared.evaluate_actions(prepared.unflatten(x), current_args)
        stationarity = current.gradient + current.constraint_operator.adjoint_mv(
            constraint_dual(equality_dual, inequality_dual)
        )
        slacks = current.values.inequality_slacks[active_indices]
        active_values = (
            slacks if mode == "fixed-active" else slacks * inequality_dual - barrier
        )
        return jnp.concatenate((stationarity, current.values.equalities, active_values))

    derivative = prepare_linearization(lambda value: residual(value, args), initial)
    operator: AbstractLinearOperator = JacobianLinearOperator(derivative)
    if adjoint:
        operator = AdjointLinearOperator(operator)
    system = _prepare_system(operator, policy, route, precision)
    return (
        prepared,
        initial,
        residual,
        operator,
        system,
        active_mask,
        model_parameters,
        evaluation.gradient,
    )


def _regularity(
    operator: AbstractLinearOperator,
    system: PreparedLinearSolve,
    result: LinearSolveResult,
    route: ConstrainedSensitivityRoute,
    precision: NonlinearPrecisionPolicy,
    /,
) -> tuple[Array, Array, InverseResidualCertificate | None]:
    """Require native all-direction rank and condition evidence."""
    usable = result.successful & result.diagnostics.converged & result.diagnostics.finite
    if route == "dense-svd":
        condition = result.diagnostics.condition_estimate
        regular = (
            usable
            & (result.diagnostics.rank == operator.source.size)
            & jnp.isfinite(condition)
            & (condition < 1e12)
        )
        return condition, regular, None

    def inverse_action(columns: Array) -> tuple[Array, Array]:
        solved = solve_linear_many(system, columns)
        return solved.value, (
            solved.successful & solved.diagnostics.converged & solved.diagnostics.finite
        )

    certificate = inverse_residual_certificate(
        operator,
        inverse_action,
        accumulation_dtype=precision.accumulation_dtype,
    )
    return certificate.condition_upper_bound, usable & certificate.valid, certificate


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
    route: ConstrainedSensitivityRoute = "matrix-free",
    materialization: MaterializationPolicy | None = None,
) -> ConstrainedSensitivityResult:
    mode = parse(mode, ConstrainedSensitivityMode, "mode")
    route = parse(route, ConstrainedSensitivityRoute, "route")
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    prepared, initial, residual, operator, system, active, model_parameters, gradient = (
        _sensitivity_system(
            problem,
            parameters,
            args,
            mode,
            active_tolerance,
            barrier,
            linear,
            precision_,
            route=route,
            materialization=materialization,
        )
    )
    _, argument_action = jax.jvp(
        lambda current_args: residual(initial, current_args), (args,), (tangent_args,)
    )
    result = solve_linear(system, _linear_rhs(system, -argument_action))
    condition, regular, certificate = _regularity(
        operator, system, result, route, precision_
    )
    tangent = prepared.unflatten(
        precision_.direction(result.value)[: prepared.template_coordinates.size]
    )
    tangent = jax.tree.map(
        lambda value: jnp.where(regular, value, jnp.full_like(value, jnp.nan)), tangent
    )
    return ConstrainedSensitivityResult(
        tangent,
        precision_.decision(condition),
        jnp.sum(active, dtype=jnp.int32),
        regular,
        mode=mode,
        precision_evidence=precision_.evidence_for(model_parameters, gradient),
        linear_plan_id=result.provenance.plan_id,
        route=route,
        linear_diagnostics=result.diagnostics,
        regularity_evidence=certificate,
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
    route: ConstrainedSensitivityRoute = "matrix-free",
    materialization: MaterializationPolicy | None = None,
) -> ConstrainedSensitivityResult:
    mode = parse(mode, ConstrainedSensitivityMode, "mode")
    route = parse(route, ConstrainedSensitivityRoute, "route")
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    if jax.tree.structure(cotangent_parameters) != jax.tree.structure(parameters) or any(
        jnp.shape(cotangent) != jnp.shape(parameter)
        for cotangent, parameter in zip(
            jax.tree.leaves(cotangent_parameters),
            jax.tree.leaves(parameters),
            strict=True,
        )
    ):
        raise ValueError(
            "cotangent_parameters must match the parameter PyTree and leaf shapes."
        )
    prepared, initial, residual, operator, system, active, model_parameters, gradient = (
        _sensitivity_system(
            problem,
            parameters,
            args,
            mode,
            active_tolerance,
            barrier,
            linear,
            precision_,
            route=route,
            materialization=materialization,
            adjoint=True,
        )
    )
    cotangent, _ = ravel_pytree(cotangent_parameters)
    if cotangent.size != prepared.template_coordinates.size:
        raise ValueError("cotangent_parameters has the wrong coordinate size.")
    right = jnp.concatenate(
        (cotangent, jnp.zeros((initial.size - cotangent.size,), dtype=cotangent.dtype))
    )
    result = solve_linear(system, _linear_rhs(system, right))
    condition, regular, certificate = _regularity(
        operator, system, result, route, precision_
    )
    _, pullback = jax.vjp(lambda current_args: residual(initial, current_args), args)
    argument_cotangent = jax.tree.map(
        jnp.negative, pullback(jnp.asarray(result.value, dtype=initial.dtype))[0]
    )
    argument_cotangent = jax.tree.map(
        lambda value: jnp.where(regular, value, jnp.full_like(value, jnp.nan)),
        argument_cotangent,
    )
    return ConstrainedSensitivityResult(
        argument_cotangent,
        precision_.decision(condition),
        jnp.sum(active, dtype=jnp.int32),
        regular,
        mode=mode,
        precision_evidence=precision_.evidence_for(model_parameters, gradient),
        linear_plan_id=result.provenance.plan_id,
        route=route,
        linear_diagnostics=result.diagnostics,
        regularity_evidence=certificate,
    )


__all__ = [
    "ConstrainedSensitivityMode",
    "ConstrainedSensitivityRoute",
    "ConstrainedSensitivityResult",
    "constrained_solution_jvp",
    "constrained_solution_vjp",
]
