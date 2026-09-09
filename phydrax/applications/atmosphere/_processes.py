#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stage-evaluated global forcing, phase change and conservative reservoirs."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._column import conservative_radiation, conservative_vertical_mixing
from ._global_surface import GlobalSurfacePhysics
from ._moist import MoistThermodynamicPlan


class GlobalHeldForcing(StrictModule):
    equilibrium_temperature: Array
    radiative_rate: Array
    drag_rate: Array
    sensible_heat_flux: Array
    evaporation_flux: Array
    solar_down: Array


class GlobalProcessRates(StrictModule):
    """Physical rates plus one-way internal phase-conversion throughput.

    ``phase_conversion_mass_rate`` is one half the sum of absolute
    vapor/liquid/ice relaxation rates in kg/(m² s), integrated vertically per
    column. It excludes precipitation and numerical phase projection.
    """

    temperature: Array
    east: Array
    north: Array
    mass: Array
    water: tuple[Array, Array, Array]
    surface_water: Array
    surface_energy: Array
    environment_energy: Array
    phase_conversion_mass_rate: Array
    successful: Array
    energy_power: Array


class GlobalAtmosphereProcesses(StrictModule):
    """Minimal dry/moist process composition, with no post-step repair.

    Without ``surface_physics``, evaporation/sensible fluxes are prescribed.
    The optional physical boundary instead derives SST and exchanges from the
    existing surface inventories; prescribed radiation/heat/water cannot also
    own that boundary. Water and donor enthalpy come from those reservoirs.
    Condensate precipitation has an exponential-removal *tendency* and deposits phase
    enthalpy and water at the surface. It is not a cloud/fall-speed model.
    Saturation relaxation conserves local moist enthalpy using the real
    mixed-phase isobaric equilibrium closure. Exchanges are evaluated at each
    Runge--Kutta stage; only external forcing fields have a held cadence.
    """

    thermodynamics: MoistThermodynamicPlan | None
    surface_physics: GlobalSurfacePhysics | None
    held_suarez: bool = eqx.field(static=True)
    equilibrium_temperature: float = eqx.field(static=True)
    radiative_timescale: float = eqx.field(static=True)
    mixing_rate: float = eqx.field(static=True)
    condensation_timescale: float = eqx.field(static=True)
    precipitation_timescale: float = eqx.field(static=True)
    sensible_heat_flux: float = eqx.field(static=True)
    evaporation_flux: float = eqx.field(static=True)
    cadence: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        thermodynamics=None,
        surface_physics=None,
        held_suarez=False,
        equilibrium_temperature=260.0,
        radiative_timescale=0.0,
        mixing_rate=0.0,
        condensation_timescale=300.0,
        precipitation_timescale=1800.0,
        sensible_heat_flux=0.0,
        evaporation_flux=0.0,
        cadence=1,
    ):
        if thermodynamics is not None and not isinstance(
            thermodynamics, MoistThermodynamicPlan
        ):
            raise TypeError("thermodynamics must be MoistThermodynamicPlan or None.")
        if surface_physics is not None:
            if not isinstance(surface_physics, GlobalSurfacePhysics):
                raise TypeError("surface_physics must be GlobalSurfacePhysics or None.")
            if thermodynamics is None:
                raise ValueError(
                    "Interactive surface physics requires moist thermodynamics."
                )
            if surface_physics.slab.thermodynamics.plan_id != thermodynamics.plan_id:
                raise ValueError(
                    "Global air and slab must share a thermodynamic reference."
                )
            if (
                held_suarez
                or radiative_timescale != 0
                or sensible_heat_flux != 0
                or evaporation_flux != 0
            ):
                raise ValueError(
                    "Interactive surface physics is the sole radiative and lower-boundary owner."
                )
        values = (
            equilibrium_temperature,
            radiative_timescale,
            mixing_rate,
            condensation_timescale,
            precipitation_timescale,
            sensible_heat_flux,
            evaporation_flux,
        )
        if any(not np.isfinite(x) for x in values):
            raise ValueError("Global process parameters must be finite.")
        if (
            equilibrium_temperature <= 0
            or min(radiative_timescale, mixing_rate, evaporation_flux) < 0
        ):
            raise ValueError(
                "Invalid temperature, relaxation, mixing or evaporation parameter."
            )
        if (
            min(condensation_timescale, precipitation_timescale) <= 0
            or int(cadence) != cadence
            or cadence < 1
        ):
            raise ValueError(
                "Phase-change timescales and integer cadence must be positive."
            )
        if thermodynamics is None and evaporation_flux != 0:
            raise ValueError("Evaporation requires moist thermodynamics.")
        self.thermodynamics, self.held_suarez = thermodynamics, bool(held_suarez)
        self.surface_physics = surface_physics
        self.equilibrium_temperature, self.radiative_timescale = (
            float(equilibrium_temperature),
            float(radiative_timescale),
        )
        self.mixing_rate, self.condensation_timescale = (
            float(mixing_rate),
            float(condensation_timescale),
        )
        self.precipitation_timescale = float(precipitation_timescale)
        self.sensible_heat_flux, self.evaporation_flux = (
            float(sensible_heat_flux),
            float(evaporation_flux),
        )
        self.cadence = int(cadence)
        self.process_id = canonical_fingerprint(
            {
                "kind": "global-atmosphere-stage-processes",
                "held_suarez": held_suarez,
                "thermodynamics": None
                if thermodynamics is None
                else thermodynamics.plan_id,
                "surface_physics": None
                if surface_physics is None
                else surface_physics.plan_id,
                "parameters": list(values),
                "cadence": self.cadence,
            }
        )

    def admit_step(self, dt: float) -> None:
        if self.thermodynamics is not None and dt > min(
            self.condensation_timescale, self.precipitation_timescale
        ):
            raise ValueError(
                "Explicit phase processes require dt no larger than their relaxation timescale."
            )
        if dt * self.mixing_rate > 0.5:
            raise ValueError("Explicit vertical mixing requires dt * mixing_rate <= 0.5.")
        if self.radiative_timescale > 0 and dt > self.radiative_timescale:
            raise ValueError(
                "Explicit radiative relaxation requires dt <= radiative_timescale."
            )

    @property
    def active(self) -> bool:
        return (
            self.thermodynamics is not None
            or self.held_suarez
            or self.radiative_timescale > 0
            or self.mixing_rate > 0
            or self.sensible_heat_flux != 0
        )

    def thermodynamic_coefficients(self, water, dry_gas_constant, dry_heat_capacity):
        if self.thermodynamics is None:
            return jnp.full_like(water[0], dry_gas_constant), jnp.full_like(
                water[0], dry_heat_capacity
            )
        return (
            self.thermodynamics.gas_constant(*water),
            self.thermodynamics.heat_capacity(*water, at_constant_pressure=True),
        )

    def internal_energy(self, temperature, water, dry_gas_constant, dry_heat_capacity):
        if self.thermodynamics is None:
            return (dry_heat_capacity - dry_gas_constant) * temperature
        # Caloric mixture energy is independent of density in this ideal mixture.
        return self.thermodynamics.energy(jnp.ones_like(temperature), temperature, *water)

    def forcing(self, view, colatitude: Array) -> GlobalHeldForcing:
        sigma = view.pressure / view.surface_pressure[..., None]
        if self.held_suarez:
            sin_lat = jnp.cos(colatitude)[:, None, None]
            cos2 = 1.0 - sin_lat**2
            eq = jnp.maximum(
                200.0,
                (
                    315.0
                    - 60.0 * sin_lat**2
                    - 10.0 * jnp.log(view.pressure / 100000.0) * cos2
                )
                * (view.pressure / 100000.0) ** (2.0 / 7.0),
            )
            boundary = jnp.maximum(0.0, (sigma - 0.7) / 0.3)
            cooling = (
                1.0 / (40 * 86400)
                + (1.0 / (4 * 86400) - 1.0 / (40 * 86400)) * boundary * cos2**2
            )
            drag = boundary / 86400.0
        else:
            eq = jnp.full_like(view.temperature, self.equilibrium_temperature)
            cooling = jnp.full_like(
                view.temperature,
                0.0 if self.radiative_timescale == 0 else 1.0 / self.radiative_timescale,
            )
            drag = jnp.zeros_like(view.temperature)
        surface_shape = view.surface_pressure.shape
        return GlobalHeldForcing(
            eq,
            cooling,
            drag,
            jnp.full(surface_shape, self.sensible_heat_flux),
            jnp.full(surface_shape, self.evaporation_flux),
            jnp.zeros(surface_shape)
            if self.surface_physics is None
            else self.surface_physics.solar_forcing(colatitude, surface_shape),
        )

    def tendencies(
        self, view, surface_water, surface_energy, held: GlobalHeldForcing
    ) -> GlobalProcessRates:
        mass, temp, cp = view.layer_mass, view.temperature, view.heat_capacity
        boundary = None
        if self.surface_physics is not None:
            boundary = self.surface_physics.evaluate(
                self.thermodynamics, view, surface_water, surface_energy, held.solar_down
            )
        zero = jnp.zeros_like(temp)
        water_rate = (zero, zero, zero)
        mass_rate = zero
        surface_mass = jnp.zeros_like(surface_water)
        surface_energy = jnp.zeros_like(surface_water)
        phase_conversion = jnp.zeros_like(surface_water)
        if boundary is None:
            # Disabled columns use finite operands, not an infinite timescale.
            timescale = jnp.where(
                held.radiative_rate > 0,
                1.0 / jnp.where(held.radiative_rate > 0, held.radiative_rate, 1.0),
                1.0,
            )
            radiation, _ = conservative_radiation(
                temp, mass, cp, held.equilibrium_temperature, timescale
            )
            radiation = jnp.where(held.radiative_rate > 0, radiation, 0.0)
            environment = -jnp.sum(radiation, axis=-1)
            temperature_rate = radiation / (mass * cp)
        east = -held.drag_rate * view.east
        north = -held.drag_rate * view.north
        successful = jnp.asarray(True)
        sensible = held.sensible_heat_flux
        if boundary is not None:
            radiation = boundary.radiation.heating
            temperature_rate = radiation / (mass * cp)
            surface_energy = boundary.radiation.surface_heating
            environment = boundary.radiation.space_heating
            sensible = boundary.exchange.sensible_heat
            successful = jnp.all(boundary.successful)
        if self.thermodynamics is not None:
            thermo = self.thermodynamics
            qv, ql, qi = view.water
            enthalpy = thermo.enthalpy(temp, qv, ql, qi)
            hd, hv, hl, hi = thermo.phase_enthalpies(temp)
            adjusted = thermo.adjust_isobaric(view.pressure, qv + ql + qi, enthalpy)
            phase_rates = tuple(
                mass * (target - current) / self.condensation_timescale
                for target, current in zip(
                    (adjusted.vapor, adjusted.liquid, adjusted.ice),
                    view.water,
                    strict=True,
                )
            )
            phase_conversion = 0.5 * jnp.sum(
                jnp.sum(jnp.abs(jnp.stack(phase_rates)), axis=0),
                axis=-1,
            )
            # Enthalpy derivative, not merely (T_equilibrium - T)/tau, closes
            # the budget when mixture heat capacities change during relaxation.
            latent = sum(
                (h - hd) * rate for h, rate in zip((hv, hl, hi), phase_rates, strict=True)
            )
            temperature_rate = temperature_rate - latent / (mass * cp)
            rain, snow = (
                mass * ql / self.precipitation_timescale,
                mass * qi / self.precipitation_timescale,
            )
            water_rate = (phase_rates[0], phase_rates[1] - rain, phase_rates[2] - snow)
            mass_rate = -rain - snow
            surface_mass = jnp.sum(rain + snow, axis=-1)
            transported_mechanical = view.geopotential + 0.5 * (
                view.east**2 + view.north**2
            )
            surface_energy = surface_energy + jnp.sum(
                rain * (hl + transported_mechanical)
                + snow * (hi + transported_mechanical),
                axis=-1,
            )
            evaporation = (
                held.evaporation_flux
                if boundary is None
                else boundary.exchange.water_mass
            )
            donor_enthalpy = (
                evaporation * hv[..., -1]
                if boundary is None
                else boundary.exchange.water_enthalpy
            )
            # The moving-pressure-coordinate source already injects h_v(T_air).
            # Only the donor/receiver difference belongs in direct heating.
            temperature_rate = temperature_rate.at[..., -1].add(
                (donor_enthalpy - evaporation * hv[..., -1])
                / (mass[..., -1] * cp[..., -1])
            )
            mass_rate = mass_rate.at[..., -1].add(evaporation)
            water_rate = (
                water_rate[0].at[..., -1].add(evaporation),
                water_rate[1],
                water_rate[2],
            )
            surface_mass = surface_mass - evaporation
            # Co-moving mass enters at the bottom cell's geopotential and speed.
            # The slab explicitly supplies lift/kinetic work in addition to
            # donor vapor enthalpy; precipitation releases these terms above.
            surface_energy = (
                surface_energy
                - donor_enthalpy
                - evaporation * (transported_mechanical[..., -1])
            )
            successful = (
                successful & jnp.all(adjusted.successful) & jnp.all(surface_water >= 0)
            )
            if self.mixing_rate > 0:
                mixed_water = tuple(
                    conservative_vertical_mixing(q, mass, self.mixing_rate)
                    for q in view.water
                )
                mixed_h = conservative_vertical_mixing(enthalpy, mass, self.mixing_rate)
                temperature_rate = temperature_rate + (
                    mixed_h
                    - sum(
                        (h - hd) * r
                        for h, r in zip((hv, hl, hi), mixed_water, strict=True)
                    )
                ) / (mass * cp)
                water_rate = tuple(
                    a + b for a, b in zip(water_rate, mixed_water, strict=True)
                )
        elif self.mixing_rate > 0:
            temperature_rate = (
                temperature_rate
                + conservative_vertical_mixing(temp, mass, self.mixing_rate) / mass
            )
        if self.mixing_rate > 0:
            east = (
                east
                + conservative_vertical_mixing(view.east, mass, self.mixing_rate) / mass
            )
            north = (
                north
                + conservative_vertical_mixing(view.north, mass, self.mixing_rate) / mass
            )
        # Explicit dissipative momentum work leaves the resolved atmosphere and
        # enters its environment reservoir; it is never hidden as a correction.
        environment = environment - jnp.sum(
            mass * (view.east * east + view.north * north), axis=-1
        )
        temperature_rate = temperature_rate.at[..., -1].add(
            sensible / (mass[..., -1] * cp[..., -1])
        )
        surface_energy = surface_energy - sensible
        return GlobalProcessRates(
            temperature_rate,
            east,
            north,
            mass_rate,
            water_rate,
            surface_mass,
            surface_energy,
            environment,
            phase_conversion,
            successful,
            jnp.asarray(0.0),
        )


__all__ = ["GlobalAtmosphereProcesses", "GlobalHeldForcing", "GlobalProcessRates"]
