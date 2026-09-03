#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._gas_dynamics import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
    HomogeneousMixtureEulerSystem,
)


class CompressibleForcingResult(StrictModule):
    """Conservative source and exact species/momentum/energy work ledger."""

    source: Array
    species_mass_source: Array
    mass_source: Array
    momentum_source: Array
    total_energy_source: Array
    acceleration_work: Array
    injected_total_energy: Array
    volumetric_heating: Array
    work_identity_residual: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class CompressibleForcingPlan(StrictModule, NonTrainableState):
    """Body acceleration, canonical-mixture mass injection, and heat source."""

    system: (
        HomogeneousMixtureEulerSystem | HomogeneousMixtureCompressibleNavierStokesSystem
    )
    acceleration: tuple[float, ...] = eqx.field(static=True)
    injection_velocity: tuple[float, ...] = eqx.field(static=True)
    injection_mass_fractions: tuple[float, ...] = eqx.field(static=True)
    mass_rate: float = eqx.field(static=True)
    injection_density: float = eqx.field(static=True)
    injection_temperature: float = eqx.field(static=True)
    volumetric_heating: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    species_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: HomogeneousMixtureEulerSystem
        | HomogeneousMixtureCompressibleNavierStokesSystem,
        /,
        *,
        acceleration: Sequence[float] | None = None,
        mass_rate: float = 0.0,
        injection_mass_fractions: Sequence[float] | None = None,
        injection_velocity: Sequence[float] | None = None,
        injection_density: float = 1.0,
        injection_temperature: float = 300.0,
        volumetric_heating: float = 0.0,
    ):
        if not isinstance(
            system,
            (
                HomogeneousMixtureEulerSystem,
                HomogeneousMixtureCompressibleNavierStokesSystem,
            ),
        ):
            raise TypeError("system must be a canonical homogeneous-mixture gas system.")
        dimension = system.dimension
        species_count = system.species_count
        acceleration_ = (
            (0.0,) * dimension
            if acceleration is None
            else tuple(float(value) for value in acceleration)
        )
        injection_velocity_ = (
            (0.0,) * dimension
            if injection_velocity is None
            else tuple(float(value) for value in injection_velocity)
        )
        fractions = (
            (1.0 / species_count,) * species_count
            if injection_mass_fractions is None
            else tuple(float(value) for value in injection_mass_fractions)
        )
        mass_rate_ = float(mass_rate)
        injection_density_ = float(injection_density)
        injection_temperature_ = float(injection_temperature)
        heating = float(volumetric_heating)
        if (
            len(acceleration_) != dimension
            or len(injection_velocity_) != dimension
            or len(fractions) != species_count
            or any(
                not np.isfinite(value) for value in (*acceleration_, *injection_velocity_)
            )
            or any(not np.isfinite(value) or value < 0.0 for value in fractions)
            or not np.isclose(sum(fractions), 1.0)
            or not np.isfinite(mass_rate_)
            or mass_rate_ < 0.0
            or not np.isfinite(injection_density_)
            or injection_density_ <= 0.0
            or not np.isfinite(injection_temperature_)
            or injection_temperature_ <= 0.0
            or not np.isfinite(heating)
        ):
            raise ValueError("Compressible forcing values are invalid.")
        injection_species_density = injection_density_ * jnp.asarray(fractions)
        injection_state = system.thermodynamics.evaluate_density_temperature(
            injection_species_density, jnp.asarray(injection_temperature_)
        )
        if not bool(injection_state.evidence.successful):
            raise ValueError("Injection state lacks canonical thermodynamic evidence.")
        self.system = system
        self.dimension = dimension
        self.species_count = species_count
        self.acceleration = acceleration_
        self.mass_rate = mass_rate_
        self.injection_mass_fractions = fractions
        self.injection_velocity = injection_velocity_
        self.injection_density = injection_density_
        self.injection_temperature = injection_temperature_
        self.volumetric_heating = heating
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compressible-conservative-mixture-forcing",
                "system": system.system_id,
                "acceleration": acceleration_,
                "mass_rate": mass_rate_,
                "injection_mass_fractions": fractions,
                "injection_velocity": injection_velocity_,
                "injection_density": injection_density_,
                "injection_temperature": injection_temperature_,
                "volumetric_heating": heating,
            }
        )

    def evaluate(
        self,
        conserved: ArrayLike,
        /,
        *,
        acceleration: ArrayLike | None = None,
        mass_rate: ArrayLike | None = None,
        injection_mass_fractions: ArrayLike | None = None,
        injection_velocity: ArrayLike | None = None,
        injection_density: ArrayLike | None = None,
        injection_temperature: ArrayLike | None = None,
        volumetric_heating: ArrayLike | None = None,
    ) -> CompressibleForcingResult:
        state = jnp.asarray(conserved)
        if state.ndim < 1 or state.shape[-1] != self.system.component_count:
            raise ValueError("Compressible forcing state has the wrong component count.")
        species_density = state[..., : self.species_count]
        density = jnp.sum(species_density, axis=-1)
        state = eqx.error_if(
            state,
            jnp.any(~jnp.isfinite(state))
            | jnp.any(species_density < 0.0)
            | jnp.any(density <= 0.0),
            "Compressible forcing state must be finite with positive total density.",
        )
        field_shape = state.shape[:-1]
        momentum = state[..., self.species_count : -1]
        acceleration_value = jnp.broadcast_to(
            jnp.asarray(
                self.acceleration if acceleration is None else acceleration,
                dtype=state.dtype,
            ),
            field_shape + (self.dimension,),
        )
        injection_velocity_value = jnp.broadcast_to(
            jnp.asarray(
                self.injection_velocity
                if injection_velocity is None
                else injection_velocity,
                dtype=state.dtype,
            ),
            field_shape + (self.dimension,),
        )
        fractions = jnp.broadcast_to(
            jnp.asarray(
                self.injection_mass_fractions
                if injection_mass_fractions is None
                else injection_mass_fractions,
                dtype=state.dtype,
            ),
            field_shape + (self.species_count,),
        )
        rate = jnp.broadcast_to(
            jnp.asarray(
                self.mass_rate if mass_rate is None else mass_rate, dtype=state.dtype
            ),
            field_shape,
        )
        injection_density_value = jnp.broadcast_to(
            jnp.asarray(
                self.injection_density
                if injection_density is None
                else injection_density,
                dtype=state.dtype,
            ),
            field_shape,
        )
        injection_temperature_value = jnp.broadcast_to(
            jnp.asarray(
                self.injection_temperature
                if injection_temperature is None
                else injection_temperature,
                dtype=state.dtype,
            ),
            field_shape,
        )
        heating = jnp.broadcast_to(
            jnp.asarray(
                self.volumetric_heating
                if volumetric_heating is None
                else volumetric_heating,
                dtype=state.dtype,
            ),
            field_shape,
        )
        fractions = eqx.error_if(
            fractions,
            jnp.any(~jnp.isfinite(fractions) | (fractions < 0.0))
            | jnp.any(jnp.abs(jnp.sum(fractions, axis=-1) - 1.0) > 1.0e-10)
            | jnp.any(~jnp.isfinite(rate) | (rate < 0.0))
            | jnp.any(
                ~jnp.isfinite(injection_density_value) | (injection_density_value <= 0.0)
            )
            | jnp.any(
                ~jnp.isfinite(injection_temperature_value)
                | (injection_temperature_value <= 0.0)
            ),
            "Injection density, temperature, composition, and rate are invalid.",
        )
        injection_species_density = injection_density_value[..., None] * fractions
        injection_thermodynamics = (
            self.system.thermodynamics.evaluate_density_temperature(
                injection_species_density, injection_temperature_value
            )
        )
        injection_thermodynamics = eqx.error_if(
            injection_thermodynamics,
            jnp.any(~injection_thermodynamics.evidence.successful),
            "Injection state lacks canonical thermodynamic evidence.",
        )
        injection_molar_density = jnp.sum(
            injection_species_density
            / self.system.thermodynamics.schema.molar_masses.astype(state.dtype),
            axis=-1,
        )
        injection_specific_internal_energy = (
            injection_molar_density * injection_thermodynamics.molar_internal_energy
        ) / injection_density_value
        species_mass_source = rate[..., None] * fractions
        acceleration_work = contract(
            "...d,...d->...", momentum, acceleration_value, backend="jax"
        )
        momentum_source = (
            density[..., None] * acceleration_value
            + rate[..., None] * injection_velocity_value
        )
        injection_kinetic = 0.5 * contract(
            "...d,...d->...",
            injection_velocity_value,
            injection_velocity_value,
            backend="jax",
        )
        injected_energy = rate * (injection_specific_internal_energy + injection_kinetic)
        energy_source = acceleration_work + injected_energy + heating
        source = jnp.concatenate(
            (species_mass_source, momentum_source, energy_source[..., None]), axis=-1
        )
        residual = energy_source - acceleration_work - injected_energy - heating
        finite = jnp.all(jnp.isfinite(source))
        return CompressibleForcingResult(
            source,
            species_mass_source,
            rate,
            momentum_source,
            energy_source,
            acceleration_work,
            injected_energy,
            heating,
            residual,
            finite,
            self.plan_id,
        )

    __call__ = evaluate


__all__ = ["CompressibleForcingPlan", "CompressibleForcingResult"]
