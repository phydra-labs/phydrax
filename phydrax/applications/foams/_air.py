#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-region pressure models for explicit foam dynamics."""

from __future__ import annotations

from enum import IntEnum
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...bubble_dynamics import (
    AbstractBubbleCompartmentGasLaw,
    BubbleEnvironment,
    BubbleGasEvaluation,
    BubbleGasState,
)
from ...typing import parse


RegionPressureAirRoute: TypeAlias = Literal["incompressible", "compartment-gas"]


class FoamAirModel(IntEnum):
    """Resolved air model selected by a foam workflow."""

    REGION_PRESSURE = 0
    VORTEX_SHEET = 1


@final
class RegionPressureAirEvidence(StrictModule):
    """Pressure, caloric balance, and admissibility of one air update."""

    pressures: Array
    temperatures: Array
    pressure_work: Array
    amount_residual: Array
    internal_energy_residual: Array
    admissible: Array
    route: RegionPressureAirRoute = eqx.field(static=True)
    gas_law_id: str | None = eqx.field(static=True)


@final
class RegionPressureAirPlan(StrictModule):
    """Incompressible target volumes or a caloric finite-cell gas law.

    Exactly one route is selected. Incompressible pressure is the multiplier of
    the independent volume-constraint basis. Compressible pressure and state
    rates come directly from ``AbstractBubbleCompartmentGasLaw``; amount and
    internal energy remain extensive state variables.
    """

    route: RegionPressureAirRoute = eqx.field(static=True)
    target_volumes: Array | None
    gas_law: AbstractBubbleCompartmentGasLaw | None
    environment: BubbleEnvironment | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        target_volumes: ArrayLike | None = None,
        gas_law: AbstractBubbleCompartmentGasLaw | None = None,
        environment: BubbleEnvironment | None = None,
    ) -> None:
        incompressible = target_volumes is not None
        compressible = gas_law is not None or environment is not None
        if incompressible == compressible:
            raise ValueError(
                "Select exactly one air route: target_volumes or gas_law plus environment."
            )
        if incompressible:
            targets = jnp.asarray(target_volumes, dtype=jnp.float64)
            host = np.asarray(targets)
            if targets.ndim != 1 or targets.size == 0:
                raise ValueError("target_volumes must be one nonempty vector.")
            if not np.all(np.isfinite(host)) or np.any(host <= 0.0):
                raise ValueError("target_volumes must be finite and strictly positive.")
            self.route = parse("incompressible", RegionPressureAirRoute, "route")
            self.target_volumes = targets
            self.gas_law = None
            self.environment = None
            payload: dict[str, object] = {
                "kind": "region-pressure-air-plan",
                "route": self.route,
                "targets": array_tree_fingerprint(host),
            }
        else:
            if not isinstance(gas_law, AbstractBubbleCompartmentGasLaw):
                raise TypeError("gas_law must be an AbstractBubbleCompartmentGasLaw.")
            if not isinstance(environment, BubbleEnvironment):
                raise TypeError("environment must be a BubbleEnvironment.")
            self.route = parse("compartment-gas", RegionPressureAirRoute, "route")
            self.target_volumes = None
            self.gas_law = gas_law
            self.environment = environment
            payload = {
                "kind": "region-pressure-air-plan",
                "route": self.route,
                "gas_law": gas_law.law_id,
            }
        self.plan_id = canonical_fingerprint(payload)

    @classmethod
    def incompressible(cls, target_volumes: ArrayLike, /) -> RegionPressureAirPlan:
        """Construct the constrained-volume route."""
        return cls(target_volumes=target_volumes)

    @classmethod
    def compressible(
        cls,
        gas_law: AbstractBubbleCompartmentGasLaw,
        environment: BubbleEnvironment,
        /,
    ) -> RegionPressureAirPlan:
        """Construct the finite caloric gas-cell route."""
        return cls(gas_law=gas_law, environment=environment)

    def require_region_count(self, count: int, /) -> None:
        """Validate the topology-dependent number of finite gas cells."""
        if int(count) < 1:
            raise ValueError("Foam dynamics requires at least one finite region.")
        if self.target_volumes is not None and self.target_volumes.shape != (count,):
            raise ValueError(
                "target_volumes must list every finite region in table order."
            )

    def initialize(
        self,
        volumes: ArrayLike,
        reference_pressures: ArrayLike,
        /,
    ) -> BubbleGasState | None:
        """Initialize one extensive gas state per finite region."""
        volume = jnp.asarray(volumes, dtype=jnp.float64)
        pressure = jnp.asarray(reference_pressures, dtype=jnp.float64)
        if volume.ndim != 1 or pressure.shape != volume.shape:
            raise ValueError("volumes and reference_pressures must be aligned vectors.")
        self.require_region_count(volume.size)
        if self.route == "incompressible":
            return None
        law = self.gas_law
        environment = self.environment
        if law is None or environment is None:
            raise RuntimeError("Compressible air plan lost its gas law or environment.")
        state = jax.vmap(lambda v, p: law.initialize(v, p, environment))(volume, pressure)
        if state.internal_energy is None:
            raise ValueError(
                "Compartment gas laws used by foams must store internal energy."
            )
        return state

    def evaluate(
        self,
        volumes: Array,
        volume_rates: Array,
        state: BubbleGasState,
        /,
    ) -> BubbleGasEvaluation:
        """Evaluate every compressible gas compartment without hidden closure."""
        if self.route != "compartment-gas":
            raise ValueError("Gas evaluation is unavailable for incompressible air.")
        law = self.gas_law
        environment = self.environment
        if law is None or environment is None or state.internal_energy is None:
            raise RuntimeError("Compressible air state is incomplete.")
        if volumes.ndim != 1 or volume_rates.shape != volumes.shape:
            raise ValueError("volumes and volume_rates must be aligned vectors.")
        if state.amount.shape != volumes.shape or state.internal.shape[0] != volumes.size:
            raise ValueError("Gas state does not match the finite-region axis.")

        def one(
            volume: Array,
            rate: Array,
            amount: Array,
            energy: Array,
            internal: Array,
        ) -> BubbleGasEvaluation:
            return law.evaluate(
                volume,
                rate,
                BubbleGasState(amount, energy, internal),
                environment,
            )

        return jax.vmap(one)(
            volumes,
            volume_rates,
            state.amount,
            state.internal_energy,
            state.internal,
        )

    def advance(
        self,
        volumes: Array,
        volume_rates: Array,
        state: BubbleGasState,
        step_size: Array,
        /,
    ) -> tuple[BubbleGasState, RegionPressureAirEvidence]:
        """Explicitly advance extensive gas state and retain its exact rate ledger."""
        evaluation = self.evaluate(volumes, volume_rates, state)
        if state.internal_energy is None or evaluation.energy_rate is None:
            raise RuntimeError("Compartment gas update requires caloric energy state.")
        candidate = BubbleGasState(
            state.amount + step_size * evaluation.amount_rate,
            state.internal_energy + step_size * evaluation.energy_rate,
            state.internal + step_size * evaluation.internal_rate,
        )
        candidate_energy = candidate.internal_energy
        if candidate_energy is None:
            raise RuntimeError("Compartment gas candidate lost its caloric energy.")
        amount_residual = jnp.sum(candidate.amount - state.amount) - step_size * jnp.sum(
            evaluation.amount_rate
        )
        energy_residual = jnp.sum(candidate_energy - state.internal_energy) - (
            step_size * jnp.sum(evaluation.energy_rate)
        )
        pressure_work = step_size * jnp.sum(evaluation.pressure * volume_rates)
        admissible = (
            jnp.all(evaluation.admissible)
            & jnp.all(candidate.finite)
            & jnp.all(candidate.amount >= 0.0)
            & jnp.all(candidate_energy > 0.0)
        )
        evidence = RegionPressureAirEvidence(
            pressures=evaluation.pressure,
            temperatures=evaluation.temperature,
            pressure_work=pressure_work,
            amount_residual=amount_residual,
            internal_energy_residual=energy_residual,
            admissible=admissible,
            route=self.route,
            gas_law_id=None if self.gas_law is None else self.gas_law.law_id,
        )
        return candidate, evidence


__all__ = [
    "FoamAirModel",
    "RegionPressureAirEvidence",
    "RegionPressureAirPlan",
    "RegionPressureAirRoute",
]
