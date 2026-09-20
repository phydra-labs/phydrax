#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._tree_math import (
    tree_add,
    tree_allfinite,
    tree_negative,
    tree_subtract,
)
from ..linalg import (
    DenseSVD,
    JacobianLinearOperator,
    LeastSquaresProblem,
    LinearSolvePolicy,
    prepare as prepare_linear,
    prepare_linearization,
    PreparedLinearization,
    PreparedLinearSolve,
    PyTreeSpace,
    solve as solve_linear,
    transpose,
)
from ._precision import NonlinearPrecisionPolicy
from ._types import NonlinearSystemProblem


SensitivityMode: TypeAlias = Literal[
    "implicit-forward",
    "implicit-reverse",
    "unrolled",
    "truncated",
    "dlm",
    "unsupported",
]


class SensitivityStatus(IntEnum):
    SUCCESS = 0
    PRIMAL_FAILED = 1
    SINGULAR = 2
    CONDITION_LIMIT = 3
    NONFINITE = 4
    UNSUPPORTED = 5
    LINEAR_FAILED = 6


class SensitivityPolicy(StrictModule):
    mode: SensitivityMode = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    truncation: int = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    perturbation: float = eqx.field(static=True)
    primal_residual_tolerance: float = eqx.field(static=True)
    linear: LinearSolvePolicy
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        mode: SensitivityMode = "implicit-reverse",
        /,
        *,
        iterations: int = 16,
        truncation: int = 4,
        condition_limit: float = 1e12,
        perturbation: float = 1e-3,
        primal_residual_tolerance: float = 1e-8,
        linear: LinearSolvePolicy | None = None,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        if mode not in (
            "implicit-forward",
            "implicit-reverse",
            "unrolled",
            "truncated",
            "dlm",
            "unsupported",
        ):
            raise ValueError("Unknown sensitivity mode.")
        iterations_ = int(iterations)
        truncation_ = int(truncation)
        limit = float(condition_limit)
        perturbation_ = float(perturbation)
        primal_tolerance = float(primal_residual_tolerance)
        if iterations_ < 1 or not 0 <= truncation_ <= iterations_:
            raise ValueError("Sensitivity iteration/truncation counts are invalid.")
        if not isfinite(limit) or limit <= 1.0:
            raise ValueError("condition_limit must be finite and exceed one.")
        if not isfinite(perturbation_) or perturbation_ <= 0.0:
            raise ValueError("perturbation must be finite and positive.")
        if not isfinite(primal_tolerance) or primal_tolerance < 0.0:
            raise ValueError("primal_residual_tolerance must be finite and non-negative.")
        linear_ = LinearSolvePolicy(DenseSVD()) if linear is None else linear
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        precision_.validate_tolerance(primal_tolerance)
        self.mode = mode
        self.iterations = iterations_
        self.truncation = truncation_
        self.condition_limit = limit
        self.perturbation = perturbation_
        self.primal_residual_tolerance = primal_tolerance
        self.linear = linear_
        self.precision = precision_


class SensitivityEvidence(StrictModule):
    status: Array
    condition_estimate: Array
    residual_norm: Array
    finite: Array
    primal_residual_norm: Array
    primal_residual_tolerance: Array
    primal_valid: Array
    linear_status: Array
    mode: SensitivityMode = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope | None = eqx.field(static=True)
    linear_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: Any,
        condition_estimate: Any,
        residual_norm: Any,
        finite: Any,
        /,
        *,
        mode: SensitivityMode,
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
        linear_plan_id: str = "",
        primal_residual_norm: Any = jnp.nan,
        primal_residual_tolerance: Any = jnp.nan,
        primal_valid: Any = True,
        linear_status: Any = -1,
    ):
        if precision_evidence is not None and not isinstance(
            precision_evidence,
            PrecisionEvidenceEnvelope,
        ):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.condition_estimate = jnp.asarray(condition_estimate)
        self.residual_norm = jnp.asarray(residual_norm)
        self.finite = jnp.asarray(finite, dtype=jnp.bool_)
        self.mode = mode
        self.precision_evidence = precision_evidence
        self.linear_plan_id = str(linear_plan_id)
        self.primal_residual_norm = jnp.asarray(primal_residual_norm)
        self.primal_residual_tolerance = jnp.asarray(primal_residual_tolerance)
        self.primal_valid = jnp.asarray(primal_valid, dtype=jnp.bool_)
        self.linear_status = jnp.asarray(linear_status, dtype=jnp.int32)

    @property
    def successful(self):
        return self.status == int(SensitivityStatus.SUCCESS)


class SolutionMapDerivative(StrictModule):
    value: PyTree[Array]
    evidence: SensitivityEvidence


def _coordinate_norm(value: Array, precision: NonlinearPrecisionPolicy, /) -> Array:
    return precision.decision(jnp.linalg.norm(precision.accumulation(value)))


class _RootSystem(NamedTuple):
    linearization: PreparedLinearization
    operator: JacobianLinearOperator
    prepared: PreparedLinearSolve
    residual: PyTree[Array]
    primal_residual_norm: Array
    primal_finite: Array
    primal_valid: Array


def _root_system(problem, state, args, policy):
    source = PyTreeSpace(state) if problem.state_space is None else problem.state_space
    linearization = prepare_linearization(
        lambda candidate: problem.residual(candidate, args),
        state,
        source=source,
        target=problem.residual_space,
        linearization_id=f"{problem.problem_id}:sensitivity-linearization",
    )
    residual_tree = linearization.primal
    policy.precision.validate_trees(state, residual_tree)
    if linearization.source.size != linearization.target.size:
        raise ValueError("Implicit root sensitivity requires a square Jacobian.")
    operator = JacobianLinearOperator(
        linearization,
        operator_id=f"{problem.problem_id}:sensitivity-jacobian",
    )
    prepared = prepare_linear(
        LeastSquaresProblem(operator),
        policy.precision.bind_linear(policy.linear),
    )
    primal_residual_norm = _coordinate_norm(
        linearization.target.flatten(residual_tree),
        policy.precision,
    )
    primal_finite = tree_allfinite(residual_tree) & jnp.isfinite(primal_residual_norm)
    primal_valid = primal_finite & (
        primal_residual_norm <= policy.primal_residual_tolerance
    )
    return _RootSystem(
        linearization,
        operator,
        prepared,
        residual_tree,
        primal_residual_norm,
        primal_finite,
        primal_valid,
    )


def _implicit_status(system, linear_result, finite, condition, policy):
    linear_ok = (
        jnp.all(linear_result.successful)
        & jnp.all(linear_result.diagnostics.finite)
        & jnp.all(linear_result.diagnostics.converged)
    )
    condition_finite = jnp.isfinite(condition)
    return jnp.where(
        ~system.primal_valid,
        int(SensitivityStatus.PRIMAL_FAILED),
        jnp.where(
            ~finite,
            int(SensitivityStatus.NONFINITE),
            jnp.where(
                ~linear_ok,
                int(SensitivityStatus.LINEAR_FAILED),
                jnp.where(
                    ~condition_finite,
                    int(SensitivityStatus.SINGULAR),
                    jnp.where(
                        condition > policy.condition_limit,
                        int(SensitivityStatus.CONDITION_LIMIT),
                        int(SensitivityStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)


def _root_solution_jvp_from_system(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    args: Any,
    tangent_args: Any,
    policy: SensitivityPolicy,
    system: _RootSystem,
    /,
) -> SolutionMapDerivative:
    _, argument_action = jax.jvp(
        lambda current_args: problem.residual(state, current_args),
        (args,),
        (tangent_args,),
    )
    linear_result = solve_linear(
        system.prepared,
        tree_negative(argument_action),
    )
    tangent = policy.precision.direction(linear_result.value)
    derivative_residual = tree_add(system.operator.mv(tangent), argument_action)
    finite = tree_allfinite(tangent) & tree_allfinite(derivative_residual)
    condition = policy.precision.decision(
        jnp.max(linear_result.diagnostics.condition_estimate)
    )
    status = _implicit_status(system, linear_result, finite, condition, policy)
    successful = status == int(SensitivityStatus.SUCCESS)
    tangent = jax.tree.map(
        lambda value: jnp.where(
            successful,
            value,
            jnp.full_like(value, jnp.nan),
        ),
        tangent,
    )
    return SolutionMapDerivative(
        tangent,
        SensitivityEvidence(
            status,
            condition,
            _coordinate_norm(
                system.linearization.target.flatten(derivative_residual),
                policy.precision,
            ),
            finite,
            mode="implicit-forward",
            precision_evidence=policy.precision.evidence_for(
                state,
                system.residual,
            ),
            linear_plan_id=linear_result.provenance.plan_id,
            primal_residual_norm=system.primal_residual_norm,
            primal_residual_tolerance=policy.primal_residual_tolerance,
            primal_valid=system.primal_valid,
            linear_status=jnp.max(linear_result.status),
        ),
    )


def root_solution_jvp(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    args: Any,
    tangent_args: Any,
    /,
    *,
    policy: SensitivityPolicy | None = None,
) -> SolutionMapDerivative:
    policy_ = SensitivityPolicy("implicit-forward") if policy is None else policy
    if policy_.mode not in ("implicit-forward", "implicit-reverse"):
        raise ValueError("root_solution_jvp requires an implicit sensitivity mode.")
    system = _root_system(problem, state, args, policy_)
    return _root_solution_jvp_from_system(
        problem,
        state,
        args,
        tangent_args,
        policy_,
        system,
    )


def root_solution_vjp(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    args: Any,
    cotangent_state: PyTree[Any],
    /,
    *,
    policy: SensitivityPolicy | None = None,
) -> SolutionMapDerivative:
    policy_ = SensitivityPolicy("implicit-reverse") if policy is None else policy
    if policy_.mode not in ("implicit-forward", "implicit-reverse"):
        raise ValueError("root_solution_vjp requires an implicit sensitivity mode.")
    system = _root_system(problem, state, args, policy_)
    cotangent = system.linearization.source.validate(cotangent_state)
    transposed = transpose(system.operator)
    prepared_transpose = prepare_linear(
        LeastSquaresProblem(transposed),
        policy_.precision.bind_linear(policy_.linear),
    )
    linear_result = solve_linear(prepared_transpose, cotangent)
    adjoint_value = policy_.precision.direction(linear_result.value)
    condition = policy_.precision.decision(
        jnp.max(linear_result.diagnostics.condition_estimate)
    )
    _, pullback = jax.vjp(
        lambda current_args: problem.residual(state, current_args),
        args,
    )
    argument_cotangent = jax.tree.map(jnp.negative, pullback(adjoint_value)[0])
    derivative_residual = tree_subtract(
        transposed.mv(adjoint_value),
        cotangent,
    )
    finite = tree_allfinite(argument_cotangent) & tree_allfinite(derivative_residual)
    status = _implicit_status(system, linear_result, finite, condition, policy_)
    successful = status == int(SensitivityStatus.SUCCESS)
    argument_cotangent = jax.tree.map(
        lambda value: jnp.where(
            successful,
            value,
            jnp.full_like(value, jnp.nan),
        ),
        argument_cotangent,
    )
    return SolutionMapDerivative(
        argument_cotangent,
        SensitivityEvidence(
            status,
            condition,
            _coordinate_norm(
                transposed.target.flatten(derivative_residual),
                policy_.precision,
            ),
            finite,
            mode="implicit-reverse",
            precision_evidence=policy_.precision.evidence_for(
                state,
                system.residual,
            ),
            linear_plan_id=linear_result.provenance.plan_id,
            primal_residual_norm=system.primal_residual_norm,
            primal_residual_tolerance=policy_.primal_residual_tolerance,
            primal_valid=system.primal_valid,
            linear_status=jnp.max(linear_result.status),
        ),
    )


def differentiate_iterations_jvp(
    iteration: Callable[[PyTree[Any], Any], PyTree[Any]],
    initial_state: PyTree[Any],
    args: Any,
    tangent_args: Any,
    /,
    *,
    policy: SensitivityPolicy,
) -> SolutionMapDerivative:
    if policy.mode not in ("unrolled", "truncated"):
        raise ValueError("Iteration JVP requires unrolled or truncated mode.")
    cutoff = policy.iterations - policy.truncation

    def solve(current_args):
        state = initial_state
        for index in range(policy.iterations):
            if policy.mode == "truncated" and index == cutoff:
                state = jax.lax.stop_gradient(state)
            state = iteration(state, current_args)
        return state

    value, tangent = jax.jvp(solve, (args,), (tangent_args,))
    del value
    tangent = jax.tree.map(policy.precision.output, tangent)
    finite = jax.tree.reduce(
        lambda left, right: left & right,
        jax.tree.map(lambda leaf: jnp.all(jnp.isfinite(leaf)), tangent),
        jnp.asarray(True),
    )
    return SolutionMapDerivative(
        tangent,
        SensitivityEvidence(
            jnp.where(
                finite,
                int(SensitivityStatus.SUCCESS),
                int(SensitivityStatus.NONFINITE),
            ),
            jnp.asarray(jnp.nan),
            jnp.asarray(0.0),
            finite,
            mode=policy.mode,
        ),
    )


def direct_loss_minimization_gradient(
    solve_perturbed: Callable[[Any, float], PyTree[Any]],
    args: Any,
    loss: Callable[[PyTree[Any], Any], Any],
    /,
    *,
    policy: SensitivityPolicy | None = None,
) -> SolutionMapDerivative:
    policy_ = SensitivityPolicy("dlm") if policy is None else policy
    if policy_.mode != "dlm":
        raise ValueError("Direct loss minimization requires mode='dlm'.")
    positive = solve_perturbed(args, policy_.perturbation)
    negative = solve_perturbed(args, -policy_.perturbation)
    positive_loss = policy_.precision.accumulation(loss(positive, args))
    negative_loss = policy_.precision.accumulation(loss(negative, args))
    gradient = policy_.precision.output(
        (positive_loss - negative_loss) / (2.0 * policy_.perturbation)
    )
    finite = jnp.isfinite(gradient)
    return SolutionMapDerivative(
        gradient,
        SensitivityEvidence(
            jnp.where(
                finite,
                int(SensitivityStatus.SUCCESS),
                int(SensitivityStatus.NONFINITE),
            ),
            jnp.asarray(jnp.nan),
            jnp.asarray(0.0),
            finite,
            mode="dlm",
        ),
    )


def root_solution_second_jvp(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    args: Any,
    tangent_args: Any,
    /,
    *,
    second_tangent_args: Any | None = None,
    policy: SensitivityPolicy | None = None,
) -> SolutionMapDerivative:
    policy_ = SensitivityPolicy("implicit-forward") if policy is None else policy
    if policy_.mode not in ("implicit-forward", "implicit-reverse"):
        raise ValueError(
            "root_solution_second_jvp requires an implicit sensitivity mode."
        )
    system = _root_system(problem, state, args, policy_)
    first = _root_solution_jvp_from_system(
        problem,
        state,
        args,
        tangent_args,
        policy_,
        system,
    )
    second_args = (
        jax.tree.map(jnp.zeros_like, tangent_args)
        if second_tangent_args is None
        else second_tangent_args
    )

    def path_residual(time):
        state_at_time = jax.tree.map(
            lambda value, tangent: value + time * tangent,
            state,
            first.value,
        )
        args_at_time = jax.tree.map(
            lambda value, tangent, second: (
                value + time * tangent + 0.5 * time * time * second
            ),
            args,
            tangent_args,
            second_args,
        )
        return problem.residual(state_at_time, args_at_time)

    def first_path_derivative(time):
        return jax.jvp(
            path_residual,
            (time,),
            (jnp.asarray(1.0, dtype=time.dtype),),
        )[1]

    zero = jnp.asarray(0.0)
    forcing = jax.jvp(
        first_path_derivative,
        (zero,),
        (jnp.asarray(1.0),),
    )[1]
    linear_result = solve_linear(
        system.prepared,
        tree_negative(forcing),
    )
    second = policy_.precision.direction(linear_result.value)
    derivative_residual = tree_add(system.operator.mv(second), forcing)
    finite = tree_allfinite(second) & tree_allfinite(derivative_residual)
    condition = policy_.precision.decision(
        jnp.max(linear_result.diagnostics.condition_estimate)
    )
    local_status = _implicit_status(
        system,
        linear_result,
        finite,
        condition,
        policy_,
    )
    status = jnp.where(
        first.evidence.successful,
        local_status,
        first.evidence.status,
    ).astype(jnp.int32)
    successful = status == int(SensitivityStatus.SUCCESS)
    second = jax.tree.map(
        lambda value: jnp.where(
            successful,
            value,
            jnp.full_like(value, jnp.nan),
        ),
        second,
    )
    return SolutionMapDerivative(
        second,
        SensitivityEvidence(
            status,
            condition,
            _coordinate_norm(
                system.linearization.target.flatten(derivative_residual),
                policy_.precision,
            ),
            finite,
            mode="implicit-forward",
            precision_evidence=policy_.precision.evidence_for(
                state,
                system.residual,
            ),
            linear_plan_id=linear_result.provenance.plan_id,
            primal_residual_norm=system.primal_residual_norm,
            primal_residual_tolerance=policy_.primal_residual_tolerance,
            primal_valid=system.primal_valid,
            linear_status=jnp.max(linear_result.status),
        ),
    )


def minimizer_solution_jvp(
    objective: Callable[[PyTree[Any], Any], Any],
    solution: PyTree[Any],
    args: Any,
    tangent_args: Any,
    /,
    *,
    policy: SensitivityPolicy | None = None,
) -> SolutionMapDerivative:
    policy_ = SensitivityPolicy("implicit-forward") if policy is None else policy
    if policy_.mode not in ("implicit-forward", "implicit-reverse"):
        raise ValueError("minimizer_solution_jvp requires an implicit mode.")

    def gradient_function(point, current_args):
        return jax.grad(lambda item: objective(item, current_args))(point)

    source = PyTreeSpace(solution)
    linearization = prepare_linearization(
        lambda point: gradient_function(point, args),
        solution,
        source=source,
        linearization_id="minimizer-sensitivity-linearization",
    )
    gradient_tree = linearization.primal
    policy_.precision.validate_trees(solution, gradient_tree)
    operator = JacobianLinearOperator(
        linearization,
        operator_id="minimizer-sensitivity-hessian",
    )
    prepared = prepare_linear(
        LeastSquaresProblem(operator),
        policy_.precision.bind_linear(policy_.linear),
    )
    primal_residual_norm = _coordinate_norm(
        linearization.target.flatten(gradient_tree),
        policy_.precision,
    )
    primal_finite = tree_allfinite(gradient_tree) & jnp.isfinite(primal_residual_norm)
    system = _RootSystem(
        linearization,
        operator,
        prepared,
        gradient_tree,
        primal_residual_norm,
        primal_finite,
        primal_finite & (primal_residual_norm <= policy_.primal_residual_tolerance),
    )
    _, forcing = jax.jvp(
        lambda current_args: gradient_function(solution, current_args),
        (args,),
        (tangent_args,),
    )
    linear_result = solve_linear(prepared, tree_negative(forcing))
    tangent = policy_.precision.direction(linear_result.value)
    derivative_residual = tree_add(operator.mv(tangent), forcing)
    finite = tree_allfinite(tangent) & tree_allfinite(derivative_residual)
    condition = policy_.precision.decision(
        jnp.max(linear_result.diagnostics.condition_estimate)
    )
    status = _implicit_status(system, linear_result, finite, condition, policy_)
    successful = status == int(SensitivityStatus.SUCCESS)
    tangent = jax.tree.map(
        lambda value: jnp.where(
            successful,
            value,
            jnp.full_like(value, jnp.nan),
        ),
        tangent,
    )
    return SolutionMapDerivative(
        tangent,
        SensitivityEvidence(
            status,
            condition,
            _coordinate_norm(
                linearization.target.flatten(derivative_residual),
                policy_.precision,
            ),
            finite,
            mode="implicit-forward",
            precision_evidence=policy_.precision.evidence_for(
                solution,
                gradient_tree,
            ),
            linear_plan_id=linear_result.provenance.plan_id,
            primal_residual_norm=system.primal_residual_norm,
            primal_residual_tolerance=policy_.primal_residual_tolerance,
            primal_valid=system.primal_valid,
            linear_status=jnp.max(linear_result.status),
        ),
    )


__all__ = [
    "SensitivityEvidence",
    "SensitivityMode",
    "SensitivityPolicy",
    "SensitivityStatus",
    "SolutionMapDerivative",
    "differentiate_iterations_jvp",
    "direct_loss_minimization_gradient",
    "minimizer_solution_jvp",
    "root_solution_jvp",
    "root_solution_second_jvp",
    "root_solution_vjp",
]
