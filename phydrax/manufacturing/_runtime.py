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
    candidate_state: ManufacturingRuntimeState
    commanded_mass_kg: Array
    applied_mass_kg: Array
    commanded_energy_j: Array
    mass_balance_residual_kg: Array
    energy_balance_residual_j: Array

    successful: Array


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
        if (
            coordinates.ndim != 2
            or coordinates.shape[0] == 0
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError(
                "Manufacturing coordinates require finite shape (cell, dimension)."
            )
        if (
            volumes.shape != (coordinates.shape[0],)
            or not np.all(np.isfinite(volumes))
            or np.any(volumes <= 0)
        ):
            raise ValueError(
                "Manufacturing control volumes must align and be finite/positive."
            )
        if not np.isfinite(interaction_radius_m) or interaction_radius_m <= 0:
            raise ValueError(
                "Manufacturing interaction radius must be finite and positive."
            )
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
        expected_shape = self.control_volumes_m3.shape
        if (
            state.deposited_mass_kg.shape != expected_shape
            or state.supplied_energy_j.shape != expected_shape
            or state.activation.active.shape != expected_shape
            or state.activation.activation_time_s.shape != expected_shape
        ):
            raise ValueError("Manufacturing state does not match runtime discretization.")
        if end_time_s <= state.time_s or not np.isfinite(end_time_s):
            raise ValueError("Manufacturing runtime requires an increasing finite time.")
        mass = jnp.asarray(state.deposited_mass_kg)
        energy = jnp.asarray(state.supplied_energy_j)
        if (
            not bool(jnp.all(jnp.isfinite(mass)))
            or not bool(jnp.all(jnp.isfinite(energy)))
            or bool(jnp.any(mass < 0))
            or bool(jnp.any(energy < 0))
            or bool(
                jnp.any(
                    state.activation.active
                    & ~jnp.isfinite(state.activation.activation_time_s)
                )
            )
        ):
            raise ValueError("Manufacturing runtime state is nonfinite or nonphysical.")
        activation = state.activation
        commanded_mass = jnp.asarray(0.0)
        applied_mass = jnp.asarray(0.0)
        successful = jnp.asarray(True)
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
                removal_supported = jnp.all(requested <= mass + 1e-12)
                removed = jnp.minimum(mass, requested)
                mass = mass - removed
                commanded_mass = commanded_mass - event_mass
                applied_mass = applied_mass - jnp.sum(removed)
                successful = successful & removal_supported
                activation = activation.remove((removed > 0) & (mass <= 0))
            elif event.kind == "deposit":
                increment = event_mass * distribution
                mass = mass + increment
                commanded_mass = commanded_mass + event_mass
                applied_mass = applied_mass + jnp.sum(increment)
                activation = activation.activate(increment > 0, upper)

        candidate_state = ManufacturingRuntimeState(
            float(end_time_s), mass, energy, activation
        )
        mass_residual = jnp.sum(mass) - initial_mass - commanded_mass
        energy_residual = jnp.sum(energy) - initial_energy - commanded_energy
        successful = (
            successful
            & jnp.all(jnp.isfinite(mass))
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(mass >= 0)
            & jnp.all(energy >= 0)
            & jnp.isfinite(mass_residual)
            & (jnp.abs(mass_residual) <= 1e-10)
            & jnp.isfinite(energy_residual)
            & (jnp.abs(energy_residual) <= 1e-10)
        )
        if bool(successful):
            accepted_state = candidate_state
        else:
            accepted_state = state
        return ManufacturingStepResult(
            accepted_state,
            candidate_state,
            commanded_mass,
            applied_mass,
            commanded_energy,
            mass_residual,
            energy_residual,
            successful,
        )


__all__ = [
    "ManufacturingRuntime",
    "ManufacturingRuntimeState",
    "ManufacturingStepResult",
]
