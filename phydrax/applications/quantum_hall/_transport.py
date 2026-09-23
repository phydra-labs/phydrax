#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Named multi-terminal Hall-bar observables."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import (
    MultiTerminalCoherentProblem,
    MultiTerminalCoherentResult,
    solve_multiterminal_coherent,
)


_ELEMENTARY_CHARGE_SI = 1.602_176_634e-19


class HallBarPlan(StrictModule, NonTrainableState):
    problem: MultiTerminalCoherentProblem
    source_contact: int = eqx.field(static=True)
    drain_contact: int = eqx.field(static=True)
    hall_contacts: tuple[int, int] = eqx.field(static=True)
    longitudinal_contacts: tuple[int, int] = eqx.field(static=True)
    current_floor_ampere: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: MultiTerminalCoherentProblem,
        source_contact: int,
        drain_contact: int,
        hall_contacts: tuple[int, int],
        longitudinal_contacts: tuple[int, int],
        /,
        *,
        current_floor_ampere: float = 1.0e-18,
    ):
        if not isinstance(problem, MultiTerminalCoherentProblem):
            raise TypeError("problem must be MultiTerminalCoherentProblem.")
        source = int(source_contact)
        drain = int(drain_contact)
        hall = tuple(int(value) for value in hall_contacts)
        longitudinal = tuple(int(value) for value in longitudinal_contacts)
        floor = float(current_floor_ampere)
        count = len(problem.contacts)
        selected = (source, drain, *hall, *longitudinal)
        if (
            source == drain
            or len(hall) != 2
            or len(longitudinal) != 2
            or hall[0] == hall[1]
            or longitudinal[0] == longitudinal[1]
            or any(value < 0 or value >= count for value in selected)
            or not isfinite(floor)
            or floor <= 0.0
        ):
            raise ValueError("Hall-bar contact layout or current floor is invalid.")
        self.problem = problem
        self.source_contact = source
        self.drain_contact = drain
        self.hall_contacts = hall
        self.longitudinal_contacts = longitudinal
        self.current_floor_ampere = floor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hall-bar-plan",
                "problem": problem.problem_id,
                "source_contact": source,
                "drain_contact": drain,
                "hall_contacts": hall,
                "longitudinal_contacts": longitudinal,
                "current_floor_ampere": floor,
            }
        )


class QuantumHallTransportResult(StrictModule, NonTrainableState):
    coherent: MultiTerminalCoherentResult
    hall_voltage_volt: jnp.ndarray
    longitudinal_voltage_volt: jnp.ndarray
    source_current_ampere: jnp.ndarray
    hall_resistance_ohm: jnp.ndarray
    longitudinal_resistance_ohm: jnp.ndarray
    current_conservation_residual_ampere: jnp.ndarray
    successful: jnp.ndarray
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def solve_quantum_hall_transport(
    plan: HallBarPlan,
    /,
) -> QuantumHallTransportResult:
    if not isinstance(plan, HallBarPlan):
        raise TypeError("plan must be HallBarPlan.")
    coherent = solve_multiterminal_coherent(plan.problem)
    chemical = plan.problem.chemical_potentials_joule
    hall_voltage = (
        chemical[plan.hall_contacts[0]] - chemical[plan.hall_contacts[1]]
    ) / _ELEMENTARY_CHARGE_SI
    longitudinal_voltage = (
        chemical[plan.longitudinal_contacts[0]] - chemical[plan.longitudinal_contacts[1]]
    ) / _ELEMENTARY_CHARGE_SI
    source_current = coherent.charge_current_ampere[plan.source_contact]
    resolved_current = jnp.abs(source_current) > plan.current_floor_ampere
    safe_current = jnp.where(resolved_current, source_current, 1.0)
    hall_resistance = jnp.where(
        resolved_current,
        hall_voltage / safe_current,
        jnp.nan,
    )
    longitudinal_resistance = jnp.where(
        resolved_current,
        longitudinal_voltage / safe_current,
        jnp.nan,
    )
    conservation = jnp.abs(jnp.sum(coherent.charge_current_ampere))
    successful = (
        coherent.successful
        & resolved_current
        & jnp.isfinite(hall_resistance)
        & jnp.isfinite(longitudinal_resistance)
    )
    return QuantumHallTransportResult(
        coherent,
        hall_voltage,
        longitudinal_voltage,
        source_current,
        hall_resistance,
        longitudinal_resistance,
        conservation,
        successful,
        plan.plan_id,
        canonical_fingerprint(
            {
                "kind": "quantum-hall-transport-result",
                "plan": plan.plan_id,
                "coherent": coherent.result_id,
            }
        ),
    )


__all__ = [
    "HallBarPlan",
    "QuantumHallTransportResult",
    "solve_quantum_hall_transport",
]
