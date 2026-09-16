#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class DrivenWorkLedgerPlan(StrictModule):
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-6,
    ):
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not math.isfinite(absolute)
            or absolute < 0.0
            or not math.isfinite(relative)
            or relative < 0.0
        ):
            raise ValueError("Driven-work tolerances must be finite and nonnegative.")
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "driven-work-ledger",
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "stress_sign": "tension-positive-cauchy",
                "heat_sign": "positive-into-system",
            }
        )

    def initialize(self, initial_flow_power: ArrayLike = 0.0, /) -> DrivenWorkLedgerState:
        power = jnp.asarray(initial_flow_power).reshape(())
        return DrivenWorkLedgerState(
            jnp.zeros_like(power),
            jnp.zeros_like(power),
            jnp.zeros_like(power),
            power,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(True),
            self.plan_id,
        )


class DrivenWorkLedgerState(StrictModule):
    accumulated_flow_work: Array
    accumulated_thermostat_heat: Array
    cumulative_balance_residual: Array
    previous_flow_power: Array
    accepted_steps: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DrivenWorkLedgerStepResult(StrictModule):
    candidate_state: DrivenWorkLedgerState
    accepted_state: DrivenWorkLedgerState
    flow_work_increment: Array
    thermostat_heat_increment: Array
    internal_energy_change: Array
    balance_residual: Array
    tolerance: Array
    accepted: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def driven_work_ledger_step(
    plan: DrivenWorkLedgerPlan,
    state: DrivenWorkLedgerState,
    next_flow_power: ArrayLike,
    time_step: float,
    internal_energy_change: ArrayLike,
    /,
    *,
    thermostat_heat_into_system: ArrayLike = 0.0,
) -> DrivenWorkLedgerStepResult:
    if not isinstance(plan, DrivenWorkLedgerPlan) or not isinstance(
        state, DrivenWorkLedgerState
    ):
        raise TypeError("plan and state must be driven-work ledger objects.")
    if state.plan_id != plan.plan_id:
        raise ValueError("Driven-work state does not belong to this plan.")
    step = float(time_step)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("time_step must be finite and positive.")
    power = jnp.asarray(next_flow_power, dtype=state.previous_flow_power.dtype).reshape(
        ()
    )
    energy = jnp.asarray(internal_energy_change, dtype=power.dtype).reshape(())
    heat = jnp.asarray(thermostat_heat_into_system, dtype=power.dtype).reshape(())
    work = 0.5 * step * (state.previous_flow_power + power)
    residual = energy - work - heat
    scale = jnp.maximum(jnp.maximum(jnp.abs(energy), jnp.abs(work)), jnp.abs(heat))
    tolerance = plan.absolute_tolerance + plan.relative_tolerance * scale
    finite = (
        jnp.isfinite(power)
        & jnp.isfinite(energy)
        & jnp.isfinite(heat)
        & jnp.isfinite(work)
        & jnp.isfinite(residual)
    )
    accepted = state.successful & finite & (jnp.abs(residual) <= tolerance)
    candidate = DrivenWorkLedgerState(
        state.accumulated_flow_work + work,
        state.accumulated_thermostat_heat + heat,
        state.cumulative_balance_residual + residual,
        power,
        state.accepted_steps + 1,
        state.successful & accepted,
        plan.plan_id,
    )
    accepted_state = DrivenWorkLedgerState(
        jnp.where(accepted, candidate.accumulated_flow_work, state.accumulated_flow_work),
        jnp.where(
            accepted,
            candidate.accumulated_thermostat_heat,
            state.accumulated_thermostat_heat,
        ),
        jnp.where(
            accepted,
            candidate.cumulative_balance_residual,
            state.cumulative_balance_residual,
        ),
        jnp.where(accepted, candidate.previous_flow_power, state.previous_flow_power),
        jnp.where(accepted, candidate.accepted_steps, state.accepted_steps),
        state.successful,
        plan.plan_id,
    )
    return DrivenWorkLedgerStepResult(
        candidate,
        accepted_state,
        work,
        heat,
        energy,
        residual,
        tolerance,
        accepted,
        accepted,
        plan.plan_id,
    )


__all__ = [
    "DrivenWorkLedgerPlan",
    "DrivenWorkLedgerState",
    "DrivenWorkLedgerStepResult",
    "driven_work_ledger_step",
]
