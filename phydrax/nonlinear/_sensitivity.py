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

from .._strict import StrictModule
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

    def __init__(
        self,
        mode: SensitivityMode = "implicit-reverse",
        /,
        *,
        iterations: int = 16,
        truncation: int = 4,
        condition_limit: float = 1e12,
        perturbation: float = 1e-3,
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
        self.mode = mode
        self.iterations = iterations_
        self.truncation = truncation_
        self.condition_limit = limit
        self.perturbation = perturbation_


class SensitivityEvidence(StrictModule):
    status: Array
    condition_estimate: Array
    residual_norm: Array
    finite: Array
    mode: SensitivityMode = eqx.field(static=True)

    @property
    def successful(self):
        return self.status == int(SensitivityStatus.SUCCESS)


class SolutionMapDerivative(StrictModule):
    value: PyTree[Array]
    evidence: SensitivityEvidence


def _root_system(problem, state, args):
    coordinates, unflatten = ravel_pytree(state)

    def coordinate_residual(value, current_args):
        residual = problem.residual(unflatten(value), current_args)
        return ravel_pytree(residual)[0]

    matrix = jax.jacfwd(lambda value: coordinate_residual(value, args))(coordinates)
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    condition = singular_values[0] / jnp.maximum(singular_values[-1], 1e-30)
    return coordinates, unflatten, coordinate_residual, matrix, condition


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
    coordinates, unflatten, residual, matrix, condition = _root_system(
        problem, state, args
    )
    _, argument_action = jax.jvp(
        lambda current_args: residual(coordinates, current_args),
        (args,),
        (tangent_args,),
    )
    tangent_coordinates = jnp.linalg.solve(matrix, -argument_action)
    tangent = unflatten(tangent_coordinates)
    finite = jnp.all(jnp.isfinite(tangent_coordinates))
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
            jnp.linalg.norm(matrix @ tangent_coordinates + argument_action),
            finite,
            mode="implicit-forward",
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
    coordinates, _, residual, matrix, condition = _root_system(problem, state, args)
    cotangent, _ = ravel_pytree(cotangent_state)
    adjoint = jnp.linalg.solve(jnp.conj(matrix.T), cotangent)
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
            jnp.linalg.norm(jnp.conj(matrix.T) @ adjoint - cotangent),
            finite,
            mode="implicit-reverse",
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
    positive_loss = jnp.asarray(loss(positive, args))
    negative_loss = jnp.asarray(loss(negative, args))
    gradient = (positive_loss - negative_loss) / (2.0 * policy_.perturbation)
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
    _, unflatten, _, matrix, condition = _root_system(
        problem,
        state,
        args,
    )
    forcing_coordinates, _ = ravel_pytree(forcing)
    second_coordinates = jnp.linalg.solve(matrix, -forcing_coordinates)
    second = unflatten(second_coordinates)
    finite = jnp.all(jnp.isfinite(second_coordinates))
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
            jnp.linalg.norm(matrix @ second_coordinates + forcing_coordinates),
            finite,
            mode="implicit-forward",
        ),
    )


def minimizer_solution_jvp(
    objective: Callable[[PyTree[Any], Any], Any],
    solution: PyTree[Any],
    args: Any,
    tangent_args: Any,
    /,
    *,
    condition_limit: float = 1e12,
) -> SolutionMapDerivative:
    coordinates, unflatten = ravel_pytree(solution)

    def gradient_coordinates(value, current_args):
        point = unflatten(value)
        gradient = jax.grad(lambda item: objective(item, current_args))(point)
        return ravel_pytree(gradient)[0]

    hessian = jax.jacfwd(lambda value: gradient_coordinates(value, args))(coordinates)
    _, forcing = jax.jvp(
        lambda current_args: gradient_coordinates(
            coordinates,
            current_args,
        ),
        (args,),
        (tangent_args,),
    )
    tangent_coordinates = jnp.linalg.solve(hessian, -forcing)
    singular_values = jnp.linalg.svd(hessian, compute_uv=False)
    condition = singular_values[0] / jnp.maximum(
        singular_values[-1],
        1e-30,
    )
    finite = jnp.all(jnp.isfinite(tangent_coordinates))
    regular = jnp.isfinite(condition) & (condition <= condition_limit)
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
            jnp.linalg.norm(hessian @ tangent_coordinates + forcing),
            finite,
            mode="implicit-forward",
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
