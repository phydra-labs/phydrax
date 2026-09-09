#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._gas_dynamics import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
)
from ...equations._nonequilibrium_gas import (
    TwoTemperatureMixtureNavierStokesSystem,
)
from ...equations._spalart_allmaras import (
    SpalartAllmarasArguments,
    SpalartAllmarasCompressibleSystem,
)


class RarefactionHysteresisState(StrictModule):
    kinetic_recommended: Array
    plan_id: str = eqx.field(static=True)


class GradientLengthKnudsenEvidence(StrictModule):
    mean_free_path: Array
    density_knudsen: Array
    temperature_knudsen: Array
    velocity_knudsen: Array
    species_knudsen: Array
    mode_temperature_knudsen: Array
    maximum_knudsen: Array
    triggering_component: Array
    kinetic_recommended: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class GradientLengthKnudsenPlan(StrictModule, NonTrainableState):
    """Continuum gradient-length evidence with static hysteresis thresholds."""

    enter_threshold: float = eqx.field(static=True)
    leave_threshold: float = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    temperature_floor: float = eqx.field(static=True)
    velocity_floor: float = eqx.field(static=True)
    species_floor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        enter_threshold: float = 0.05,
        leave_threshold: float = 0.02,
        density_floor: float = 1.0e-12,
        temperature_floor: float = 1.0,
        velocity_floor: float = 1.0,
        species_floor: float = 1.0e-8,
    ):
        values = tuple(
            float(value)
            for value in (
                enter_threshold,
                leave_threshold,
                density_floor,
                temperature_floor,
                velocity_floor,
                species_floor,
            )
        )
        if (
            any(not np.isfinite(value) or value <= 0.0 for value in values)
            or values[1] >= values[0]
        ):
            raise ValueError("Knudsen thresholds and normalization floors are invalid.")
        self.enter_threshold = values[0]
        self.leave_threshold = values[1]
        self.density_floor = values[2]
        self.temperature_floor = values[3]
        self.velocity_floor = values[4]
        self.species_floor = values[5]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gradient-length-knudsen",
                "enter_threshold": values[0],
                "leave_threshold": values[1],
                "floors": values[2:],
            }
        )

    @staticmethod
    def _supported(system: Any, /) -> bool:
        return isinstance(
            system,
            (
                HomogeneousMixtureCompressibleNavierStokesSystem,
                TwoTemperatureMixtureNavierStokesSystem,
                SpalartAllmarasCompressibleSystem,
            ),
        )

    @staticmethod
    def _gas_system_and_state(system: Any, state: Array, /):
        if isinstance(system, SpalartAllmarasCompressibleSystem):
            return system.base, system.gas_state(state)
        return system, state

    def evaluate(
        self,
        system: Any,
        state: ArrayLike,
        conserved_gradient: ArrayLike,
        args: Any = None,
        /,
        *,
        previous: RarefactionHysteresisState | None = None,
    ) -> tuple[GradientLengthKnudsenEvidence, RarefactionHysteresisState]:
        if not self._supported(system):
            raise TypeError("Gradient-length Knudsen evidence requires viscous gas flow.")
        value = jnp.asarray(state)
        gradient = jnp.asarray(conserved_gradient)
        if gradient.shape != value.shape + (system.dimension,):
            raise ValueError("Knudsen evidence requires full conserved gradients.")
        gas_system, gas_state = self._gas_system_and_state(system, value)
        transport_args = (
            args.transport_args
            if isinstance(system, SpalartAllmarasCompressibleSystem)
            and isinstance(args, SpalartAllmarasArguments)
            else args
        )
        gas_gradient = gradient[..., : gas_system.component_count, :]
        density = gas_system.density(gas_state)
        pressure = gas_system.pressure(gas_state)
        temperature = gas_system.temperature(gas_state)
        properties = gas_system.transport_properties(gas_state, transport_args)
        mean_free_path = (
            properties.dynamic_viscosity
            / pressure
            * jnp.sqrt(jnp.pi * pressure / (2.0 * density))
        )
        flat_state = value.reshape((-1, system.component_count))
        primitive_jacobian = jax.vmap(jax.jacfwd(system.conserved_to_primitive))(
            flat_state
        ).reshape(value.shape[:-1] + (system.component_count, system.component_count))
        primitive_gradient = contract(
            "...pc,...cd->...pd", primitive_jacobian, gradient, backend="jax"
        )
        density_gradient = jnp.sum(
            gas_gradient[..., : gas_system.species_count, :], axis=-2
        )
        density_knudsen = (
            mean_free_path
            * jnp.sqrt(jnp.sum(density_gradient * density_gradient, axis=-1))
            / jnp.maximum(jnp.abs(density), self.density_floor)
        )
        temperature_gradient = primitive_gradient[..., gas_system.energy_index, :]
        temperature_knudsen = (
            mean_free_path
            * jnp.sqrt(jnp.sum(temperature_gradient * temperature_gradient, axis=-1))
            / jnp.maximum(jnp.abs(temperature), self.temperature_floor)
        )
        primitive = system.conserved_to_primitive(value)
        velocity = system.primitive_velocity(primitive)
        velocity_gradient = primitive_gradient[..., system.momentum_slice, :]
        speed_scale = jnp.maximum(
            jnp.sqrt(jnp.sum(velocity * velocity, axis=-1)),
            self.velocity_floor,
        )
        velocity_knudsen = (
            mean_free_path
            * jnp.sqrt(jnp.sum(velocity_gradient * velocity_gradient, axis=(-2, -1)))
            / speed_scale
        )
        species_density = gas_state[..., : gas_system.species_count]
        mass_fraction = species_density / density[..., None]
        mass_fraction_gradient = (
            gas_gradient[..., : gas_system.species_count, :]
            - mass_fraction[..., :, None] * density_gradient[..., None, :]
        ) / density[..., None, None]
        species_knudsen = (
            mean_free_path[..., None]
            * jnp.sqrt(jnp.sum(mass_fraction_gradient * mass_fraction_gradient, axis=-1))
            / jnp.maximum(jnp.abs(mass_fraction), self.species_floor)
        )
        if isinstance(gas_system, TwoTemperatureMixtureNavierStokesSystem):
            mode_temperature = gas_system.mode_temperatures(gas_state)
            mode_gradient = primitive_gradient[..., gas_system.mode_slice, :]
            mode_temperature_knudsen = (
                mean_free_path[..., None]
                * jnp.sqrt(jnp.sum(mode_gradient * mode_gradient, axis=-1))
                / jnp.maximum(jnp.abs(mode_temperature), self.temperature_floor)
            )
        else:
            mode_temperature_knudsen = jnp.zeros(
                value.shape[:-1] + (0,), dtype=value.dtype
            )
        components = jnp.concatenate(
            (
                density_knudsen[..., None],
                temperature_knudsen[..., None],
                velocity_knudsen[..., None],
                species_knudsen,
                mode_temperature_knudsen,
            ),
            axis=-1,
        )
        maximum = jnp.max(components, axis=-1)
        trigger = jnp.argmax(components, axis=-1)
        if previous is None:
            previous_mask = jnp.zeros(maximum.shape, dtype=bool)
        else:
            if (
                not isinstance(previous, RarefactionHysteresisState)
                or previous.plan_id != self.plan_id
                or previous.kinetic_recommended.shape != maximum.shape
            ):
                raise ValueError("Rarefaction hysteresis state does not match this plan.")
            previous_mask = previous.kinetic_recommended
        recommended = jnp.where(
            maximum >= self.enter_threshold,
            True,
            jnp.where(maximum <= self.leave_threshold, False, previous_mask),
        )
        finite = jnp.isfinite(mean_free_path) & jnp.all(jnp.isfinite(components), axis=-1)
        successful = finite & (mean_free_path > 0.0) & gas_system.admissible(gas_state)
        state_ = RarefactionHysteresisState(recommended, self.plan_id)
        evidence = GradientLengthKnudsenEvidence(
            mean_free_path,
            density_knudsen,
            temperature_knudsen,
            velocity_knudsen,
            species_knudsen,
            mode_temperature_knudsen,
            maximum,
            trigger,
            recommended,
            finite,
            successful,
            self.plan_id,
        )
        return evidence, state_


__all__ = [
    "GradientLengthKnudsenEvidence",
    "GradientLengthKnudsenPlan",
    "RarefactionHysteresisState",
]
