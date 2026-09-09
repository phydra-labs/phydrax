#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._ionized_gas import (
    IonizedMultitemperatureEulerSystem,
    IonizedMultitemperatureNavierStokesSystem,
)
from ..equations._nonlte_radiation import NonLTERadiationCoefficientPlan


class RadiationMatterLedger(StrictModule):
    gas_energy_change: Array
    radiation_energy_change: Array
    combined_energy_defect: Array
    minimum_radiation_energy: Array
    minimum_electron_energy: Array
    finite: Array
    successful: Array


class MultigroupRadiationMatterResult(StrictModule):
    gas_candidate: Array
    radiation_candidate: Array
    gas_accepted: Array
    radiation_accepted: Array
    ledger: RadiationMatterLedger
    plan_id: str = eqx.field(static=True)


class MultigroupRadiationMatterProcessPlan(StrictModule, NonTrainableState):
    """Exact frozen-coefficient multigroup exchange with one gas-energy ledger."""

    coefficients: NonLTERadiationCoefficientPlan
    transport_light_speed: float = eqx.field(static=True)
    matter_light_speed: float = eqx.field(static=True)
    subcycles: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: NonLTERadiationCoefficientPlan,
        /,
        *,
        transport_light_speed: float = 299792458.0,
        matter_light_speed: float = 299792458.0,
        subcycles: int = 4,
    ):
        transport_speed = float(transport_light_speed)
        matter_speed = float(matter_light_speed)
        count = int(subcycles)
        if (
            not isinstance(coefficients, NonLTERadiationCoefficientPlan)
            or not np.isfinite(transport_speed)
            or transport_speed <= 0.0
            or not np.isfinite(matter_speed)
            or matter_speed <= 0.0
            or count <= 0
        ):
            raise ValueError("Radiation-matter coefficients or light speeds are invalid.")
        self.coefficients = coefficients
        self.transport_light_speed = transport_speed
        self.matter_light_speed = matter_speed
        self.subcycles = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multigroup-radiation-matter-process",
                "coefficients": coefficients.plan_id,
                "transport_light_speed": transport_speed,
                "matter_light_speed": matter_speed,
                "subcycles": count,
            }
        )

    def advance(
        self,
        system: IonizedMultitemperatureEulerSystem
        | IonizedMultitemperatureNavierStokesSystem,
        gas_state: ArrayLike,
        radiation_energy: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        departure_coefficients: ArrayLike | None = None,
    ) -> MultigroupRadiationMatterResult:
        if not isinstance(
            system,
            (
                IonizedMultitemperatureEulerSystem,
                IonizedMultitemperatureNavierStokesSystem,
            ),
        ):
            raise TypeError("Radiation-matter exchange requires ionized gas.")
        gas = jnp.asarray(gas_state)
        radiation = jnp.asarray(radiation_energy, dtype=gas.dtype)
        step = jnp.asarray(step_size, dtype=gas.dtype)
        group_count = self.coefficients.group_count
        if (
            gas.shape[-1] != system.component_count
            or radiation.shape != gas.shape[:-1] + (group_count,)
            or step.shape != ()
        ):
            raise ValueError("Radiation-matter state or step shape is invalid.")
        initial_gas = gas
        initial_radiation = radiation
        electron_mode = (
            system.energy_index + 1 + system.thermodynamics.electron_mode_index
        )
        substep = step / self.subcycles

        def body(_, carry):
            gas_value, radiation_value = carry
            recovered = system.recover_thermodynamics(gas_value)
            species_molar = gas_value[
                ..., : system.species_count
            ] / system.thermodynamics.schema.molar_masses.astype(gas.dtype)
            coefficient = self.coefficients.evaluate(
                species_molar,
                recovered.electron_temperature,
                departure_coefficients=departure_coefficients,
            )
            absorption_rate = self.matter_light_speed * coefficient.absorption
            attenuation = jnp.exp(-absorption_rate * substep)
            equilibrium = coefficient.emission_power_density / jnp.maximum(
                absorption_rate, jnp.finfo(gas.dtype).tiny
            )
            absorbed = equilibrium + (radiation_value - equilibrium) * attenuation
            transparent = coefficient.absorption <= 0.0
            updated_radiation = jnp.where(
                transparent,
                radiation_value + substep * coefficient.emission_power_density,
                absorbed,
            )
            radiation_change = jnp.sum(updated_radiation - radiation_value, axis=-1)
            electron_before = gas_value[..., electron_mode]
            requested_electron = electron_before - radiation_change
            lower_fraction = jnp.where(
                requested_electron < 0.0,
                0.9
                * electron_before
                / jnp.maximum(radiation_change, jnp.finfo(gas.dtype).tiny),
                1.0,
            )
            fraction = jnp.clip(lower_fraction, 0.0, 1.0)
            accepted_change = fraction * radiation_change
            updated_radiation = radiation_value + fraction[..., None] * (
                updated_radiation - radiation_value
            )
            gas_value = gas_value.at[..., system.energy_index].add(-accepted_change)
            gas_value = gas_value.at[..., electron_mode].add(-accepted_change)
            return gas_value, updated_radiation

        gas_candidate, radiation_candidate = jax.lax.fori_loop(
            0, self.subcycles, body, (gas, radiation)
        )
        gas_change = (
            gas_candidate[..., system.energy_index]
            - initial_gas[..., system.energy_index]
        )
        radiation_change = jnp.sum(radiation_candidate - initial_radiation, axis=-1)
        defect = gas_change + radiation_change
        finite = (
            jnp.all(jnp.isfinite(gas_candidate), axis=-1)
            & jnp.all(jnp.isfinite(radiation_candidate), axis=-1)
            & jnp.isfinite(defect)
        )
        successful_local = (
            finite
            & system.admissible(gas_candidate)
            & jnp.all(radiation_candidate >= 0.0, axis=-1)
            & (
                jnp.abs(defect)
                <= 512.0
                * jnp.finfo(gas.dtype).eps
                * jnp.maximum(jnp.abs(gas_change) + jnp.abs(radiation_change), 1.0)
            )
        )
        successful = jnp.all(successful_local)
        gas_accepted = jnp.where(successful, gas_candidate, initial_gas)
        radiation_accepted = jnp.where(successful, radiation_candidate, initial_radiation)
        electron_energy = gas_candidate[..., electron_mode]
        ledger = RadiationMatterLedger(
            gas_change,
            radiation_change,
            defect,
            jnp.min(radiation_candidate, axis=-1),
            electron_energy,
            jnp.all(finite),
            successful,
        )
        return MultigroupRadiationMatterResult(
            gas_candidate,
            radiation_candidate,
            gas_accepted,
            radiation_accepted,
            ledger,
            self.plan_id,
        )


__all__ = [
    "MultigroupRadiationMatterProcessPlan",
    "MultigroupRadiationMatterResult",
    "RadiationMatterLedger",
]
