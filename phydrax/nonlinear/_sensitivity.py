#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any, Callable, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

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


class SensitivityPolicy(StrictModule):
    mode: SensitivityMode = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    truncation: int = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    perturbation: float = eqx.field(static=True)
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
        if iterations_ < 1 or not 0 <= truncation_ <= iterations_:
            raise ValueError("Sensitivity iteration/truncation counts are invalid.")
        if not isfinite(limit) or limit <= 1.0:
            raise ValueError("condition_limit must be finite and exceed one.")
        if not isfinite(perturbation_) or perturbation_ <= 0.0:
            raise ValueError("perturbation must be finite and positive.")
        linear_ = LinearSolvePolicy(DenseSVD()) if linear is None else linear
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.mode = mode
        self.iterations = iterations_
        self.truncation = truncation_
        self.condition_limit = limit
        self.perturbation = perturbation_
        self.linear = linear_
        self.precision = precision_


class SensitivityEvidence(StrictModule):
    status: Array
    condition_estimate: Array
    residual_norm: Array
    finite: Array
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
        self.finite = jnp.asarray(finite, dtype=bool)
        self.mode = mode
        self.precision_evidence = precision_evidence
        self.linear_plan_id = str(linear_plan_id)

    @property
    def successful(self):
        return self.status == int(SensitivityStatus.SUCCESS)


class SolutionMapDerivative(StrictModule):
    value: PyTree[Array]
    evidence: SensitivityEvidence


def _coordinate_norm(value: Array, precision: NonlinearPrecisionPolicy, /) -> Array:
    return precision.decision(jnp.linalg.norm(precision.accumulation(value)))


def _root_system(problem, state, args, policy):
    coordinates, unflatten = ravel_pytree(state)
    residual_tree = problem.residual(state, args)
    policy.precision.validate_trees(state, residual_tree)

    def coordinate_residual(value, current_args):
        residual = problem.residual(unflatten(value), current_args)
        return ravel_pytree(residual)[0]

    matrix = jax.jacfwd(lambda value: coordinate_residual(value, args))(coordinates)
    prepared = prepare_linear(
        LeastSquaresProblem(DenseLinearOperator(matrix)),
        policy.precision.bind_linear(policy.linear),
    )
    return (
        coordinates,
        unflatten,
        coordinate_residual,
        matrix,
        residual_tree,
        prepared,
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
    (
        coordinates,
        unflatten,
        residual,
        matrix,
        residual_tree,
        prepared,
    ) = _root_system(problem, state, args, policy_)
    _, argument_action = jax.jvp(
        lambda current_args: residual(coordinates, current_args),
        (args,),
        (tangent_args,),
    )
    linear_result = solve_linear(prepared, -argument_action)
    tangent_coordinates = policy_.precision.direction(linear_result.value)
    tangent = unflatten(tangent_coordinates)
    finite = jnp.all(jnp.isfinite(tangent_coordinates))
    condition = policy_.precision.decision(linear_result.diagnostics.condition_estimate)
    regular = jnp.isfinite(condition) & (condition <= policy_.condition_limit)
    status = jnp.where(
        ~finite,
        int(SensitivityStatus.NONFINITE),
        jnp.where(
            ~jnp.isfinite(condition),
            int(SensitivityStatus.SINGULAR),
            jnp.where(
                ~regular,
                int(SensitivityStatus.CONDITION_LIMIT),
                int(SensitivityStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    tangent = jax.tree.map(
        lambda value: jnp.where(regular & finite, value, jnp.full_like(value, jnp.nan)),
        tangent,
    )
    return SolutionMapDerivative(
        tangent,
        SensitivityEvidence(
            status,
            condition,
            _coordinate_norm(
                matrix @ tangent_coordinates + argument_action,
                policy_.precision,
            ),
            finite,
            mode="implicit-forward",
            precision_evidence=policy_.precision.evidence_for(
                state,
                residual_tree,
            ),
            linear_plan_id=linear_result.provenance.plan_id,
        ),
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
    (
        coordinates,
        _,
        residual,
        matrix,
        residual_tree,
        prepared,
    ) = _root_system(problem, state, args, policy_)
    cotangent, _ = ravel_pytree(cotangent_state)
    linear_result = solve_linear(
        LeastSquaresProblem(DenseLinearOperator(jnp.conj(matrix.T))),
        cotangent,
        policy=policy_.precision.bind_linear(policy_.linear),
    )
    adjoint = policy_.precision.direction(linear_result.value)
    condition = policy_.precision.decision(linear_result.diagnostics.condition_estimate)
    _, pullback = jax.vjp(lambda current_args: residual(coordinates, current_args), args)
    argument_cotangent = jax.tree.map(jnp.negative, pullback(adjoint)[0])
    finite = jax.tree.reduce(
        lambda left, right: left & right,
        jax.tree.map(lambda value: jnp.all(jnp.isfinite(value)), argument_cotangent),
        jnp.asarray(True),
    )
    regular = jnp.isfinite(condition) & (condition <= policy_.condition_limit)
    status = jnp.where(
        finite & regular,
        int(SensitivityStatus.SUCCESS),
        jnp.where(
            ~jnp.isfinite(condition),
            int(SensitivityStatus.SINGULAR),
            jnp.where(
                ~regular,
                int(SensitivityStatus.CONDITION_LIMIT),
                int(SensitivityStatus.NONFINITE),
            ),
        ),
    ).astype(jnp.int32)
    argument_cotangent = jax.tree.map(
        lambda value: jnp.where(finite & regular, value, jnp.full_like(value, jnp.nan)),
        argument_cotangent,
    )
    return SolutionMapDerivative(
        argument_cotangent,
        SensitivityEvidence(
            status,
            condition,
            _coordinate_norm(
                jnp.conj(matrix.T) @ adjoint - cotangent,
                policy_.precision,
            ),
            finite,
            mode="implicit-reverse",
            precision_evidence=policy_.precision.evidence_for(
                state,
                residual_tree,
            ),
            linear_plan_id=linear_result.provenance.plan_id,
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
    first = root_solution_jvp(
        problem,
        state,
        args,
        tangent_args,
        policy=policy_,
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
    (
        _,
        unflatten,
        _,
        matrix,
        residual_tree,
        prepared,
    ) = _root_system(problem, state, args, policy_)
    forcing_coordinates, _ = ravel_pytree(forcing)
    linear_result = solve_linear(prepared, -forcing_coordinates)
    second_coordinates = policy_.precision.direction(linear_result.value)
    second = unflatten(second_coordinates)
    finite = jnp.all(jnp.isfinite(second_coordinates))
    condition = policy_.precision.decision(linear_result.diagnostics.condition_estimate)
    regular = (
        first.evidence.successful
        & jnp.isfinite(condition)
        & (condition <= policy_.condition_limit)
    )
    second = jax.tree.map(
        lambda value: jnp.where(
            finite & regular,
            value,
            jnp.full_like(value, jnp.nan),
        ),
        second,
    )
    return SolutionMapDerivative(
        second,
        SensitivityEvidence(
            jnp.where(
                finite & regular,
                int(SensitivityStatus.SUCCESS),
                int(SensitivityStatus.CONDITION_LIMIT),
            ),
            condition,
            _coordinate_norm(
                matrix @ second_coordinates + forcing_coordinates,
                policy_.precision,
            ),
            finite,
            mode="implicit-forward",
            precision_evidence=policy_.precision.evidence_for(
                state,
                residual_tree,
            ),
            linear_plan_id=linear_result.provenance.plan_id,
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
    coordinates, unflatten = ravel_pytree(solution)

    def gradient_coordinates(value, current_args):
        point = unflatten(value)
        gradient = jax.grad(lambda item: objective(item, current_args))(point)
        return ravel_pytree(gradient)[0]

    gradient_tree = jax.grad(lambda item: objective(item, args))(solution)
    policy_.precision.validate_trees(solution, gradient_tree)

    hessian = jax.jacfwd(lambda value: gradient_coordinates(value, args))(coordinates)
    _, forcing = jax.jvp(
        lambda current_args: gradient_coordinates(
            coordinates,
            current_args,
        ),
        (args,),
        (tangent_args,),
    )
    prepared = prepare_linear(
        LeastSquaresProblem(DenseLinearOperator(hessian)),
        policy_.precision.bind_linear(policy_.linear),
    )
    linear_result = solve_linear(prepared, -forcing)
    tangent_coordinates = policy_.precision.direction(linear_result.value)
    condition = policy_.precision.decision(linear_result.diagnostics.condition_estimate)
    finite = jnp.all(jnp.isfinite(tangent_coordinates))
    regular = jnp.isfinite(condition) & (condition <= policy_.condition_limit)
    tangent = unflatten(tangent_coordinates)
    tangent = jax.tree.map(
        lambda value: jnp.where(
            finite & regular,
            value,
            jnp.full_like(value, jnp.nan),
        ),
        tangent,
    )
    return SolutionMapDerivative(
        tangent,
        SensitivityEvidence(
            jnp.where(
                finite & regular,
                int(SensitivityStatus.SUCCESS),
                int(SensitivityStatus.CONDITION_LIMIT),
            ),
            condition,
            _coordinate_norm(
                hessian @ tangent_coordinates + forcing,
                policy_.precision,
            ),
            finite,
            mode="implicit-forward",
            precision_evidence=policy_.precision.evidence_for(
                solution,
                gradient_tree,
            ),
            linear_plan_id=linear_result.provenance.plan_id,
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
    "root_solution_jvp",
    "root_solution_vjp",
    "minimizer_solution_jvp",
    "root_solution_second_jvp",
]
