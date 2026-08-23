#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._quantum_jump import StateVectorOperator


class NeuralJumpProjectionProblem(StrictModule):
    state_function: Callable[[Array], Array]
    parameters: Array
    jump_operator: StateVectorOperator
    problem_id: str

    def __init__(
        self,
        state_function: Callable[[Array], Array],
        parameters: ArrayLike,
        jump_operator: StateVectorOperator,
        /,
        *,
        problem_id: str = "neural-jump-projection",
    ):
        if not callable(state_function):
            raise TypeError("state_function must be callable.")
        parameters_ = jnp.asarray(parameters)
        state = jnp.asarray(state_function(parameters_))
        if state.shape != (jump_operator.dimension,):
            raise ValueError("Neural state and jump-operator dimensions differ.")
        self.state_function = state_function
        self.parameters = parameters_
        self.jump_operator = jump_operator
        self.problem_id = str(problem_id)


class NeuralJumpProjectionResult(StrictModule):
    parameters: Array
    projected_state: Array
    target_state: Array
    infidelity_history: Array
    residual: Array
    valid: Array
    problem_id: str

    def __init__(
        self,
        parameters: ArrayLike,
        projected_state: ArrayLike,
        target_state: ArrayLike,
        infidelity_history: ArrayLike,
        /,
        *,
        problem_id: str,
    ):
        self.parameters = jnp.asarray(parameters)
        self.projected_state = jnp.asarray(projected_state)
        self.target_state = jnp.asarray(target_state)
        self.infidelity_history = jnp.asarray(infidelity_history)
        self.residual = jnp.sqrt(jnp.maximum(self.infidelity_history[-1], 0.0))
        self.valid = (
            jnp.all(jnp.isfinite(self.projected_state))
            & jnp.all(jnp.isfinite(self.target_state))
            & jnp.all(jnp.isfinite(self.infidelity_history))
        )
        self.problem_id = str(problem_id)


def solve_neural_jump_projection(
    problem: NeuralJumpProjectionProblem,
    /,
    *,
    learning_rate: float = 0.05,
    iterations: int = 50,
) -> NeuralJumpProjectionResult:
    source = jnp.asarray(problem.state_function(problem.parameters))
    source = source / jnp.linalg.norm(source)
    target = problem.jump_operator(source)
    target = target / jnp.linalg.norm(target)

    def loss(parameters):
        state = jnp.asarray(problem.state_function(parameters))
        state = state / jnp.linalg.norm(state)
        overlap = jnp.vdot(target, state)
        return 1.0 - jnp.abs(overlap) ** 2

    value_and_grad = jax.value_and_grad(loss)
    parameters = problem.parameters
    history = []
    for _ in range(int(iterations)):
        value, gradient = value_and_grad(parameters)
        parameters = parameters - float(learning_rate) * gradient
        history.append(value)
    projected = jnp.asarray(problem.state_function(parameters))
    projected = projected / jnp.linalg.norm(projected)
    return NeuralJumpProjectionResult(
        parameters,
        projected,
        target,
        jnp.stack(history),
        problem_id=problem.problem_id,
    )


__all__ = [
    "NeuralJumpProjectionProblem",
    "NeuralJumpProjectionResult",
    "solve_neural_jump_projection",
]
