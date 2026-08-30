#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import (
    FiniteElementAcceptedStepSchedule,
    FiniteElementAttemptResult,
    FiniteElementStepPolicy,
)


class IncompressibleFlowOperators(StrictModule, NonTrainableState):
    advection: Callable
    velocity_solve: Callable
    divergence: Callable
    pressure_solve: Callable
    gradient: Callable
    subcycle: Callable | None
    operators_id: str = eqx.field(static=True)

    def __init__(
        self,
        advection: Callable,
        velocity_solve: Callable,
        divergence: Callable,
        pressure_solve: Callable,
        gradient: Callable,
        /,
        *,
        subcycle: Callable | None = None,
        operators_id: str = "incompressible-flow-operators",
    ):
        callables = (advection, velocity_solve, divergence, pressure_solve, gradient)
        if not all(callable(value) for value in callables):
            raise TypeError("Every incompressible-flow operator must be callable.")
        if subcycle is not None and not callable(subcycle):
            raise TypeError("subcycle must be callable or None.")
        identifier = str(operators_id)
        if not identifier:
            raise ValueError("operators_id must be non-empty.")
        self.advection = advection
        self.velocity_solve = velocity_solve
        self.divergence = divergence
        self.pressure_solve = pressure_solve
        self.gradient = gradient
        self.subcycle = subcycle
        self.operators_id = canonical_fingerprint(
            {
                "kind": "incompressible-flow-operators",
                "declared_id": identifier,
                "has_subcycle": subcycle is not None,
            }
        )


class IncompressibleFlowPolicy(StrictModule, NonTrainableState):
    pressure_increment: bool = eqx.field(static=True)
    advection_subcycles: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        pressure_increment: bool = True,
        advection_subcycles: int = 1,
    ):
        subcycles = int(advection_subcycles)
        if subcycles < 1:
            raise ValueError("Advection subcycles must be positive.")
        self.pressure_increment = bool(pressure_increment)
        self.advection_subcycles = subcycles
        self.policy_id = canonical_fingerprint(
            {
                "kind": "incompressible-flow-policy",
                "pressure_increment": bool(pressure_increment),
                "advection_subcycles": subcycles,
            }
        )


class IncompressibleFlowState(StrictModule):
    velocity: Array
    pressure: Array
    velocity_history: tuple[Array, ...]

    def __init__(
        self,
        velocity: ArrayLike,
        pressure: ArrayLike,
        /,
        *,
        velocity_history: Sequence[ArrayLike] = (),
    ):
        velocity_ = jnp.asarray(velocity)
        pressure_ = jnp.asarray(pressure)
        history = tuple(jnp.asarray(value) for value in velocity_history)
        if not jnp.issubdtype(velocity_.dtype, jnp.inexact) or not jnp.issubdtype(
            pressure_.dtype, jnp.inexact
        ):
            raise TypeError("Incompressible-flow state must use inexact arrays.")
        if any(value.shape != velocity_.shape for value in history):
            raise ValueError("Velocity history must preserve velocity shape.")
        self.velocity = velocity_
        self.pressure = pressure_
        self.velocity_history = history


class IncompressibleFlowDiagnostics(StrictModule):
    divergence_before: Array
    divergence_after: Array
    pressure_increment_norm: Array
    successful: Array


def oifs_history_combination(
    history: Sequence[ArrayLike],
    coefficients: ArrayLike,
    /,
) -> Array:
    values = tuple(jnp.asarray(value) for value in history)
    weights = jnp.asarray(coefficients)
    if not values or weights.shape != (len(values),):
        raise ValueError("OIFS history and coefficients are incompatible.")
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("OIFS history values must share one shape.")
    combined = jnp.zeros_like(values[0])
    accumulated = jnp.asarray(0.0, dtype=weights.dtype)
    for value, weight in zip(reversed(values), reversed(weights), strict=True):
        denominator = weight + accumulated
        combined = jnp.where(
            denominator != 0.0,
            weight / denominator * value + accumulated / denominator * combined,
            combined,
        )
        accumulated = denominator
    return combined


def pressure_correction_step(
    state: IncompressibleFlowState,
    step_size: ArrayLike,
    operators: IncompressibleFlowOperators,
    policy: IncompressibleFlowPolicy,
    time: ArrayLike,
    args: object = None,
    /,
) -> tuple[IncompressibleFlowState, IncompressibleFlowDiagnostics]:
    if (
        not isinstance(state, IncompressibleFlowState)
        or not isinstance(operators, IncompressibleFlowOperators)
        or not isinstance(policy, IncompressibleFlowPolicy)
    ):
        raise TypeError("Pressure-correction step inputs are invalid.")
    dt = jnp.asarray(step_size)
    time_ = jnp.asarray(time)
    if dt.shape != () or time_.shape != () or bool(dt <= 0.0):
        raise ValueError("Pressure-correction time data are invalid.")
    if policy.advection_subcycles > 1:
        if operators.subcycle is None:
            raise ValueError("Advection subcycling requires a subcycle operator.")
        advected = operators.subcycle(
            state.velocity,
            time_,
            dt,
            policy.advection_subcycles,
            args,
        )
        advection = (state.velocity - advected) / dt
    else:
        advection = operators.advection(state.velocity, time_, args)
    right_hand_side = state.velocity / dt - advection
    if policy.pressure_increment:
        right_hand_side = right_hand_side - operators.gradient(
            state.pressure, time_, args
        )
    predictor = operators.velocity_solve(right_hand_side, 1.0 / dt, time_, args)
    divergence_before = operators.divergence(predictor, time_, args)
    pressure_correction = operators.pressure_solve(
        -divergence_before / dt,
        time_,
        args,
    )
    if policy.pressure_increment:
        pressure = state.pressure + pressure_correction
        correction_gradient = operators.gradient(pressure_correction, time_, args)
    else:
        pressure = pressure_correction
        correction_gradient = operators.gradient(pressure, time_, args)
    velocity = predictor - dt * correction_gradient
    divergence_after = operators.divergence(velocity, time_ + dt, args)
    finite = (
        jnp.all(jnp.isfinite(velocity))
        & jnp.all(jnp.isfinite(pressure))
        & jnp.all(jnp.isfinite(divergence_after))
    )
    next_history = (state.velocity, *state.velocity_history[:2])
    return IncompressibleFlowState(
        velocity,
        pressure,
        velocity_history=next_history,
    ), IncompressibleFlowDiagnostics(
        divergence_before=jnp.sqrt(jnp.sum(jnp.abs(divergence_before) ** 2)),
        divergence_after=jnp.sqrt(jnp.sum(jnp.abs(divergence_after) ** 2)),
        pressure_increment_norm=jnp.sqrt(jnp.sum(jnp.abs(pressure_correction) ** 2)),
        successful=finite,
    )


def incompressible_flow_schedule(
    operators: IncompressibleFlowOperators,
    flow_policy: IncompressibleFlowPolicy | None = None,
    /,
    *,
    step_policy: FiniteElementStepPolicy | None = None,
) -> FiniteElementAcceptedStepSchedule:
    selected = IncompressibleFlowPolicy() if flow_policy is None else flow_policy
    if not isinstance(selected, IncompressibleFlowPolicy):
        raise TypeError("flow_policy must be IncompressibleFlowPolicy or None.")

    def attempt(accepted, start, end, time_law, args):
        state = IncompressibleFlowState(
            accepted.fields[0],
            accepted.fields[1],
            velocity_history=accepted.fields[2:],
        )
        updated, diagnostics = pressure_correction_step(
            state,
            end - start,
            operators,
            selected,
            start,
            args,
        )
        fields = (updated.velocity, updated.pressure, *updated.velocity_history)
        return FiniteElementAttemptResult(
            fields,
            diagnostics.successful,
            retry_requested=~diagnostics.successful,
            suggested_step=0.5 * (end - start),
            diagnostics=diagnostics,
        )

    return FiniteElementAcceptedStepSchedule(
        attempt,
        policy=step_policy,
        schedule_id="incompressible-flow-pressure-correction",
    )


__all__ = [
    "IncompressibleFlowDiagnostics",
    "IncompressibleFlowOperators",
    "IncompressibleFlowPolicy",
    "IncompressibleFlowState",
    "incompressible_flow_schedule",
    "oifs_history_combination",
    "pressure_correction_step",
]
