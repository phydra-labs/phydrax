#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ..metrix import InformationMetricOperator
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
    count = int(iterations)
    if count < 1 or not jnp.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("iterations and learning_rate must be positive and finite.")
    for _ in range(count):
        value, gradient = value_and_grad(parameters)
        direction = jnp.conj(gradient) if jnp.iscomplexobj(gradient) else gradient
        parameters = parameters - float(learning_rate) * direction
        history.append(value)
    final_value = loss(parameters)
    history.append(final_value)
    projected = jnp.asarray(problem.state_function(parameters))
    projected = projected / jnp.linalg.norm(projected)
    return NeuralJumpProjectionResult(
        parameters,
        projected,
        target,
        jnp.stack(history),
        problem_id=problem.problem_id,
    )


class NeuralNoJumpTDVPProblem(StrictModule):
    parameters: Array
    qgt_action: Callable[[Array, Array], Array]
    force: Callable[[Array], Array]
    channel_rates: Callable[[Array], Array]
    jump_projection: Callable[[int, Array], tuple[Array, Array]]
    problem_id: str

    def __init__(
        self,
        parameters: ArrayLike,
        qgt_action: Callable[[Array, Array], Array],
        force: Callable[[Array], Array],
        channel_rates: Callable[[Array], Array],
        jump_projection: Callable[[int, Array], tuple[Array, Array]],
        /,
        *,
        problem_id: str = "neural-no-jump-tdvp",
    ):
        for value in (qgt_action, force, channel_rates, jump_projection):
            if not callable(value):
                raise TypeError("Neural trajectory actions must be callable.")
        self.parameters = jnp.asarray(parameters)
        self.qgt_action = qgt_action
        self.force = force
        self.channel_rates = channel_rates
        self.jump_projection = jump_projection
        self.problem_id = str(problem_id)


class NeuralNoJumpTDVPResult(StrictModule):
    parameters: Array
    parameter_history: Array
    rate_history: Array
    event_times: Array
    event_channels: Array
    projection_residuals: Array
    active_events: Array
    valid: Array
    saturated: Array
    successful: Array
    problem_id: str

    def __init__(
        self,
        parameters: ArrayLike,
        parameter_history: ArrayLike,
        rate_history: ArrayLike,
        event_times: ArrayLike,
        event_channels: ArrayLike,
        projection_residuals: ArrayLike,
        active_events: ArrayLike,
        /,
        *,
        problem_id: str,
        saturated: bool = False,
        successful: bool = True,
    ):
        self.parameters = jnp.asarray(parameters)
        self.parameter_history = jnp.asarray(parameter_history)
        self.rate_history = jnp.asarray(rate_history)
        self.event_times = jnp.asarray(event_times)
        self.event_channels = jnp.asarray(event_channels, dtype=jnp.int32)
        self.projection_residuals = jnp.asarray(projection_residuals)
        self.active_events = jnp.asarray(active_events, dtype=bool)
        self.saturated = jnp.asarray(saturated, dtype=bool)
        self.successful = jnp.asarray(successful, dtype=bool)
        self.valid = (
            jnp.all(jnp.isfinite(self.parameter_history))
            & jnp.all(jnp.isfinite(self.rate_history))
            & jnp.all(self.rate_history >= 0.0)
            & jnp.all(
                jnp.where(
                    self.active_events,
                    jnp.isfinite(self.projection_residuals),
                    True,
                )
            )
            & ~self.saturated
            & self.successful
        )
        self.problem_id = str(problem_id)


def solve_neural_no_jump_tdvp(
    problem: NeuralNoJumpTDVPProblem,
    key: Array,
    /,
    *,
    step_size: float,
    steps: int,
    maximum_events: int = 64,
    damping: float = 1e-6,
) -> NeuralNoJumpTDVPResult:
    count = int(steps)
    capacity = int(maximum_events)
    step = float(step_size)
    if (
        count < 1
        or capacity < 1
        or not jnp.isfinite(step)
        or step <= 0.0
        or not jnp.isfinite(damping)
        or damping < 0.0
    ):
        raise ValueError("Neural trajectory steps, capacity, and scales are invalid.")
    parameters = problem.parameters
    history = [parameters]
    rates_history = []
    event_times = jnp.zeros((capacity,))
    event_channels = -jnp.ones((capacity,), dtype=jnp.int32)
    projection_residuals = jnp.zeros((capacity,))
    active = jnp.zeros((capacity,), dtype=bool)
    threshold_address = SampleAddress(
        "neural-quantum-trajectory",
        "jump-threshold",
        target=problem.problem_id,
        role="threshold",
    )
    channel_address = SampleAddress(
        "neural-quantum-trajectory",
        "jump-channel",
        target=problem.problem_id,
        role="channel",
    )
    event_count = 0
    threshold = -jnp.log(
        jnp.maximum(
            1.0 - jax.random.uniform(derive_key(key, threshold_address, 0)),
            1e-30,
        )
    )
    hazard = jnp.asarray(0.0)
    saturated = False
    successful = True
    for index in range(count):
        force = jnp.asarray(problem.force(parameters))
        metric = InformationMetricOperator(
            lambda vector: problem.qgt_action(parameters, vector),
            parameters,
            damping=damping,
            metric_id=f"{problem.problem_id}:qgt",
        )
        solve_result = metric.solve(force)
        if not bool(jax.device_get(jnp.all(solve_result.successful))):
            successful = False
            break
        parameters = parameters + step * solve_result.value
        rates = jnp.asarray(problem.channel_rates(parameters))
        if (
            jnp.iscomplexobj(rates)
            or rates.ndim != 1
            or not bool(jax.device_get(jnp.all(jnp.isfinite(rates) & (rates >= 0.0))))
        ):
            successful = False
            break
        rates_history.append(rates)
        total = jnp.sum(rates)
        if not bool(jax.device_get(jnp.isfinite(total) & (total >= 0.0))):
            successful = False
            break
        hazard = hazard + step * total
        while bool(jax.device_get((total > 0.0) & (hazard >= threshold))):
            if event_count >= capacity:
                saturated = True
                break
            hazard = hazard - threshold
            event_time = (index + 1) * step - hazard / total
            channel = jax.random.categorical(
                derive_key(key, channel_address, event_count),
                jnp.log(jnp.maximum(rates / total, 1e-30)),
            )
            parameters, residual = problem.jump_projection(int(channel), parameters)
            residual = jnp.asarray(residual)
            if residual.shape != () or not bool(jax.device_get(jnp.isfinite(residual))):
                successful = False
                break
            event_times = event_times.at[event_count].set(event_time)
            event_channels = event_channels.at[event_count].set(
                jnp.asarray(channel, dtype=jnp.int32)
            )
            projection_residuals = projection_residuals.at[event_count].set(residual)
            active = active.at[event_count].set(True)
            event_count += 1
            rates = jnp.asarray(problem.channel_rates(parameters))
            if (
                jnp.iscomplexobj(rates)
                or rates.ndim != 1
                or not bool(jax.device_get(jnp.all(jnp.isfinite(rates) & (rates >= 0.0))))
            ):
                successful = False
                break
            total = jnp.sum(rates)
            threshold = -jnp.log(
                jnp.maximum(
                    1.0
                    - jax.random.uniform(derive_key(key, threshold_address, event_count)),
                    1e-30,
                )
            )
        history.append(parameters)
        if saturated or not successful:
            break
    rate_array = (
        jnp.stack(rates_history)
        if rates_history
        else jnp.zeros((0, 0), dtype=parameters.real.dtype)
    )
    return NeuralNoJumpTDVPResult(
        parameters,
        jnp.stack(history),
        rate_array,
        event_times,
        event_channels,
        projection_residuals,
        active,
        problem_id=problem.problem_id,
        saturated=saturated,
        successful=successful,
    )


__all__ = [
    "NeuralJumpProjectionProblem",
    "NeuralJumpProjectionResult",
    "NeuralNoJumpTDVPProblem",
    "NeuralNoJumpTDVPResult",
    "solve_neural_jump_projection",
    "solve_neural_no_jump_tdvp",
]
