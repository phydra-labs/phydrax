#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conservative execution of scheduled manufacturing events."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ._activation import MaterialActivationState
from ._schedule import ProcessSchedule


_GAUSS_ABSCISSA = jnp.asarray((-0.7745966692414834, 0.0, 0.7745966692414834))
_GAUSS_WEIGHT = jnp.asarray((5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0))


@dataclass(frozen=True, slots=True)
class ManufacturingRuntimeState:
    time_s: float
    deposited_mass_kg: Array
    supplied_energy_j: Array
    activation: MaterialActivationState

    @classmethod
    def initialize(cls, control_volume_count: int, /, *, time_s: float = 0.0):
        if control_volume_count <= 0 or not np.isfinite(time_s):
            raise ValueError("Manufacturing runtime initialization is invalid.")
        zeros = jnp.zeros((control_volume_count,))
        return cls(
            float(time_s),
            zeros,
            zeros,
            MaterialActivationState(
                jnp.zeros((control_volume_count,), dtype=jnp.bool_),
                jnp.full((control_volume_count,), -jnp.inf),
            ),
        )


@dataclass(frozen=True, slots=True)
class ManufacturingStepResult:
    state: ManufacturingRuntimeState
    commanded_mass_kg: Array
    commanded_energy_j: Array
    mass_balance_residual_kg: Array
    energy_balance_residual_j: Array


@dataclass(frozen=True, slots=True)
class ManufacturingRuntime:
    """Control-volume runtime for deposition, removal, and moving heat events."""

    schedule: ProcessSchedule
    coordinates_m: Array
    control_volumes_m3: Array
    interaction_radius_m: float

    @classmethod
    def create(
        cls,
        schedule: ProcessSchedule,
        coordinates_m: ArrayLike,
        control_volumes_m3: ArrayLike,
        interaction_radius_m: float,
        /,
    ) -> ManufacturingRuntime:
        coordinates = np.asarray(coordinates_m, dtype=np.float64)
        volumes = np.asarray(control_volumes_m3, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[0] == 0:
            raise ValueError("Manufacturing coordinates require shape (cell, dimension).")
        if volumes.shape != (coordinates.shape[0],) or np.any(volumes <= 0):
            raise ValueError("Manufacturing control volumes must align and be positive.")
        if not np.isfinite(interaction_radius_m) or interaction_radius_m <= 0:
            raise ValueError("Manufacturing interaction radius must be positive.")
        if any(len(event.start) != coordinates.shape[1] for event in schedule.events):
            raise ValueError(
                "Schedule and manufacturing coordinates use different dimensions."
            )
        return cls(
            schedule,
            jnp.asarray(coordinates),
            jnp.asarray(volumes),
            float(interaction_radius_m),
        )

    def _distribution(self, center: Array) -> Array:
        delta = self.coordinates_m - center
        kernel = jnp.exp(
            -2 * contract("qd,qd->q", delta, delta) / self.interaction_radius_m**2
        )
        weighted = kernel * self.control_volumes_m3
        normalization = jnp.sum(weighted)
        return jnp.where(normalization > 0, weighted / normalization, 0)

    def _integrated_distribution(self, event, lower: float, upper: float) -> Array:
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        sample_times = midpoint + half_width * _GAUSS_ABSCISSA
        distributions = jnp.stack(
            tuple(self._distribution(event.position(t)) for t in sample_times)
        )
        return 0.5 * contract("g,gq->q", _GAUSS_WEIGHT, distributions)

    def advance(
        self,
        state: ManufacturingRuntimeState,
        end_time_s: float,
        /,
    ) -> ManufacturingStepResult:
        if state.deposited_mass_kg.shape != self.control_volumes_m3.shape:
            raise ValueError("Manufacturing state does not match runtime discretization.")
        if end_time_s <= state.time_s or not np.isfinite(end_time_s):
            raise ValueError("Manufacturing runtime requires an increasing finite time.")

        mass = state.deposited_mass_kg
        energy = state.supplied_energy_j
        activation = state.activation
        commanded_mass = jnp.asarray(0.0)
        commanded_energy = jnp.asarray(0.0)
        initial_mass = jnp.sum(mass)
        initial_energy = jnp.sum(energy)

        for event in self.schedule.events:
            lower = max(state.time_s, event.start_time_s)
            upper = min(float(end_time_s), event.end_time_s)
            if upper <= lower:
                continue
            duration = upper - lower
            distribution = self._integrated_distribution(event, lower, upper)
            event_energy = jnp.asarray(event.power_w * duration)
            energy = energy + event_energy * distribution
            commanded_energy = commanded_energy + event_energy

            event_mass = jnp.asarray(event.mass_rate_kg_s * duration)
            if event.kind == "remove":
                requested = event_mass * distribution
                removed = jnp.minimum(mass, requested)
                mass = mass - removed
                commanded_mass = commanded_mass - jnp.sum(removed)
                activation = activation.remove((removed > 0) & (mass <= 0))
            elif event.kind == "deposit":
                increment = event_mass * distribution
                mass = mass + increment
                commanded_mass = commanded_mass + event_mass
                activation = activation.activate(increment > 0, upper)

        next_state = ManufacturingRuntimeState(
            float(end_time_s), mass, energy, activation
        )
        return ManufacturingStepResult(
            next_state,
            commanded_mass,
            commanded_energy,
            jnp.sum(mass) - initial_mass - commanded_mass,
            jnp.sum(energy) - initial_energy - commanded_energy,
        )


__all__ = [
    "ManufacturingRuntime",
    "ManufacturingRuntimeState",
    "ManufacturingStepResult",
]
