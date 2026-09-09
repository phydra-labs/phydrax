#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-rate, fixed-volume moist physics over one interactive wet slab.

Cells have unit horizontal area and run TOP TO BOTTOM. Layer volumes therefore
also give thicknesses in metres. This is a forced column, not a hydrostatic or
momentum solver: imposed ventilation and shear supply unresolved stirring.
Internal-energy budgets exclude kinetic and gravitational energy; the omitted
terminal-fall gravitational power and turbulent temperature-variance destruction
are reported explicitly, never restored as unrecorded heat.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_archive import read_array_archive, write_array_archive
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...solver._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ._moist import MoistThermodynamicPlan
from ._radiation import ColumnRadiationPlan
from ._surface import BulkSurfaceExchangePlan, WetSlabPlan, WetSlabState


_LAYER_FIELDS = (
    "dry_mass",
    "vapor_mass",
    "cloud_liquid_mass",
    "cloud_ice_mass",
    "rain_mass",
    "snow_mass",
    "internal_energy",
    "layer_volume",
)
_SCALAR_FIELDS = (
    "environment_energy",
    "external_energy",
    "precipitated_water",
    "precipitated_energy",
    "evaporated_water",
    "fall_potential_energy",
    "mixing_temperature_variance_loss",
    "time",
    "step_count",
)
_PARAMETER_FIELDS = (
    "condensation_timescale",
    "autoconversion_timescale",
    "cloud_threshold",
    "rain_evaporation_timescale",
    "phase_conversion_timescale",
    "rain_fall_speed",
    "snow_fall_speed",
    "background_diffusivity",
    "mixing_length",
    "critical_richardson",
)


class InteractiveMoistColumnState(StrictModule):
    dry_mass: Array
    vapor_mass: Array
    cloud_liquid_mass: Array
    cloud_ice_mass: Array
    rain_mass: Array
    snow_mass: Array
    internal_energy: Array
    layer_volume: Array
    slab: WetSlabState
    environment_energy: Array
    external_energy: Array
    precipitated_water: Array
    precipitated_energy: Array
    evaporated_water: Array
    fall_potential_energy: Array
    mixing_temperature_variance_loss: Array
    time: Array
    step_count: Array
    plan_id: str = eqx.field(static=True)

    @property
    def water_mass(self) -> Array:
        return (
            self.vapor_mass
            + self.cloud_liquid_mass
            + self.cloud_ice_mass
            + self.rain_mass
            + self.snow_mass
        )

    @property
    def layer_mass(self) -> Array:
        return self.dry_mass + self.water_mass

    @property
    def total_water(self) -> Array:
        return jnp.sum(self.water_mass) + self.slab.water_mass

    @property
    def total_energy(self) -> Array:
        """Caloric energy including the signed radiative environment reservoir."""
        return jnp.sum(self.internal_energy) + self.slab.energy + self.environment_energy


class InteractiveColumnDiagnostics(StrictModule):
    temperature: Array
    density: Array
    pressure: Array
    relative_humidity: Array
    surface_temperature: Array
    successful: Array
    derivative_valid: Array


class InteractiveColumnStepResult(StrictModule):
    state: InteractiveMoistColumnState
    successful: Array
    derivative_valid: Array
    stable_step: Array
    water_residual: Array
    energy_residual: Array
    dry_mass_residual: Array
    precipitated_water: Array
    precipitated_energy: Array
    rain_evaporated_water: Array
    sensible_heat: Array
    surface_water_flux: Array
    surface_water_enthalpy: Array
    radiative_heating: Array
    surface_radiative_heating: Array
    space_radiative_heating: Array
    upward_radiative_flux: Array
    downward_radiative_flux: Array
    turbulent_diffusivity: Array
    buoyancy_frequency_squared: Array
    fall_potential_power: Array
    mixing_temperature_variance_dissipation: Array


def _divergence(flux, template):
    zero = jnp.zeros_like(template[:1])
    return jnp.concatenate((zero, flux)) - jnp.concatenate((flux, zero))


def _temperature(thermo, masses, energy):
    """Invert native linear calorics at CURRENT composition, including precipitation."""
    dry, vapor, liquid, ice, rain, snow = masses
    capacity = (
        dry * thermo.dry_cv
        + vapor * thermo.vapor_cv
        + (liquid + rain) * thermo.liquid_heat_capacity
        + (ice + snow) * thermo.ice_heat_capacity
    )
    ed, ev, el, ei = thermo.phase_energies(thermo.reference_temperature)
    reference = dry * ed + vapor * ev + (liquid + rain) * el + (ice + snow) * ei
    return thermo.reference_temperature + (energy - reference) / capacity, capacity


def _masses(state):
    return (
        state.dry_mass,
        state.vapor_mass,
        state.cloud_liquid_mass,
        state.cloud_ice_mass,
        state.rain_mass,
        state.snow_mass,
    )


def _temperature_interior(temperature, minimum, maximum):
    """Strict derivative domain, distinct from inclusive physical admission."""
    margin = (
        64 * jnp.finfo(temperature.dtype).eps * jnp.maximum(jnp.abs(temperature), 1.0)
    )
    return (temperature - minimum > margin) & (maximum - temperature > margin)


def _at_state_dtype(module, dtype):
    """Bind numeric physics to its state owner's precision without detaching AD."""
    return jax.tree.map(
        lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
        module,
    )


class InteractiveMoistColumnPlan(StrictModule):
    """Explicit conservative splitting with physical source/transport CFL rejection.

    All numerical process coefficients are scalar JAX leaves. ``None`` radiation
    or surface exchange genuinely disables that component; the wet slab remains
    the sole owner of lower-boundary water and energy. No condensate projection,
    mass clipping, hidden substepping or equilibrium precipitation removal occurs.
    """

    thermodynamics: MoistThermodynamicPlan
    radiation: ColumnRadiationPlan | None
    surface_exchange: BulkSurfaceExchangePlan | None
    slab: WetSlabPlan
    condensation_timescale: Array
    autoconversion_timescale: Array
    cloud_threshold: Array
    rain_evaporation_timescale: Array
    phase_conversion_timescale: Array
    rain_fall_speed: Array
    snow_fall_speed: Array
    background_diffusivity: Array
    mixing_length: Array
    critical_richardson: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: MoistThermodynamicPlan | None = None,
        radiation: ColumnRadiationPlan | None = None,
        surface_exchange: BulkSurfaceExchangePlan | None = None,
        slab: WetSlabPlan | None = None,
        *,
        condensation_timescale: ArrayLike = 60.0,
        autoconversion_timescale: ArrayLike = 600.0,
        cloud_threshold: ArrayLike = 1e-4,
        rain_evaporation_timescale: ArrayLike = 120.0,
        phase_conversion_timescale: ArrayLike = 120.0,
        rain_fall_speed: ArrayLike = 5.0,
        snow_fall_speed: ArrayLike = 1.0,
        background_diffusivity: ArrayLike = 0.0,
        mixing_length: ArrayLike = 50.0,
        critical_richardson: ArrayLike = 0.25,
    ):
        self.thermodynamics = (
            MoistThermodynamicPlan() if thermodynamics is None else thermodynamics
        )
        if not isinstance(self.thermodynamics, MoistThermodynamicPlan):
            raise TypeError("thermodynamics must be MoistThermodynamicPlan.")
        if radiation is not None and not isinstance(radiation, ColumnRadiationPlan):
            raise TypeError("radiation must be ColumnRadiationPlan or None.")
        if surface_exchange is not None and not isinstance(
            surface_exchange, BulkSurfaceExchangePlan
        ):
            raise TypeError("surface_exchange must be BulkSurfaceExchangePlan or None.")
        self.radiation = radiation
        self.surface_exchange = surface_exchange
        self.slab = WetSlabPlan(self.thermodynamics) if slab is None else slab
        if not isinstance(self.slab, WetSlabPlan):
            raise TypeError("slab must be WetSlabPlan.")
        values = (
            condensation_timescale,
            autoconversion_timescale,
            cloud_threshold,
            rain_evaporation_timescale,
            phase_conversion_timescale,
            rain_fall_speed,
            snow_fall_speed,
            background_diffusivity,
            mixing_length,
            critical_richardson,
        )
        for name, value in zip(_PARAMETER_FIELDS, values):
            array = jnp.asarray(value, dtype=jnp.result_type(value, jnp.float32))
            if array.shape != ():
                raise ValueError("Column process parameters must be scalars.")
            object.__setattr__(self, name, array)
        # Numeric leaves may be changed by inference. Runtime certification, not a
        # stale static fingerprint, enforces their domain on every attempted step.
        self.plan_id = canonical_fingerprint(
            {
                "kind": "interactive-fixed-volume-moist-column",
                "thermodynamics": self.thermodynamics.plan_id,
                "radiation": None if radiation is None else radiation.plan_id,
                "surface_exchange": None
                if surface_exchange is None
                else surface_exchange.plan_id,
                "slab": self.slab.plan_id,
                "ordering": "top-to-bottom",
                "transport": "balanced-volume-donor-enthalpy",
            }
        )

    def _parameters_valid(self):
        positive = (
            self.condensation_timescale,
            self.autoconversion_timescale,
            self.rain_evaporation_timescale,
            self.phase_conversion_timescale,
            self.critical_richardson,
        )
        nonnegative = (
            self.cloud_threshold,
            self.rain_fall_speed,
            self.snow_fall_speed,
            self.background_diffusivity,
            self.mixing_length,
        )
        return (
            jnp.all(jnp.isfinite(jnp.stack(positive + nonnegative)))
            & jnp.all(jnp.stack(positive) > 0)
            & jnp.all(jnp.stack(nonnegative) >= 0)
        )

    def initialize(
        self,
        dry_mass: ArrayLike,
        vapor_mass: ArrayLike,
        temperature: ArrayLike,
        layer_volume: ArrayLike,
        *,
        cloud_liquid_mass: ArrayLike = 0.0,
        cloud_ice_mass: ArrayLike = 0.0,
        rain_mass: ArrayLike = 0.0,
        snow_mass: ArrayLike = 0.0,
        surface_temperature: ArrayLike = 290.0,
        surface_water_mass: ArrayLike = 1000.0,
    ) -> InteractiveMoistColumnState:
        """Initialize explicit kg/m² inventories without a saturation adjustment."""
        values = jnp.broadcast_arrays(
            *map(
                jnp.asarray,
                (
                    dry_mass,
                    vapor_mass,
                    cloud_liquid_mass,
                    cloud_ice_mass,
                    rain_mass,
                    snow_mass,
                    temperature,
                    layer_volume,
                ),
            )
        )
        dtype = jnp.result_type(*values, jnp.float32)
        dry, vapor, liquid, ice, rain, snow, temp, volume = (
            v.astype(dtype) for v in values
        )
        if dry.ndim != 1 or dry.size < 1:
            raise ValueError("Interactive column requires a nonempty layer axis.")
        ed, ev, el, ei = self.thermodynamics.phase_energies(temp)
        energy = dry * ed + vapor * ev + (liquid + rain) * el + (ice + snow) * ei
        lower = _at_state_dtype(self.slab, dtype).initialize(
            jnp.asarray(surface_temperature, dtype),
            jnp.asarray(surface_water_mass, dtype),
        )
        if lower.water_mass.shape != () or lower.energy.shape != ():
            raise ValueError("Column wet slab inventories must be scalar.")
        zero = jnp.zeros((), dtype)
        state = InteractiveMoistColumnState(
            dry,
            vapor,
            liquid,
            ice,
            rain,
            snow,
            energy,
            volume,
            lower,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            jnp.zeros((), jnp.int32),
            self.plan_id,
        )
        valid = jnp.all(self.diagnose(state).successful) & self._parameters_valid()
        return eqx.error_if(state, ~valid, "Initial interactive column is inadmissible.")

    def diagnose(
        self, state: InteractiveMoistColumnState
    ) -> InteractiveColumnDiagnostics:
        if state.plan_id != self.plan_id:
            raise ValueError(
                "Column state belongs to a different structural process plan."
            )
        thermo = self.thermodynamics
        masses = _masses(state)
        temperature, _ = _temperature(thermo, masses, state.internal_energy)
        mass = state.layer_mass
        density = mass / state.layer_volume
        pressure = (
            (
                state.dry_mass * thermo.dry_gas_constant
                + state.vapor_mass * thermo.vapor_gas_constant
            )
            * temperature
            / state.layer_volume
        )
        frozen = temperature < thermo.reference_temperature
        saturation = jnp.where(
            frozen,
            thermo.saturation_pressure(temperature, phase="ice"),
            thermo.saturation_pressure(temperature),
        )
        humidity = (
            state.vapor_mass
            * thermo.vapor_gas_constant
            * temperature
            / (state.layer_volume * saturation)
        )
        slab_plan = _at_state_dtype(self.slab, state.internal_energy.dtype)
        surface_temperature = slab_plan.temperature(state.slab, thermo)
        scalar_valid = (
            slab_plan.admissible(state.slab, thermo)
            & jnp.all(
                jnp.isfinite(
                    jnp.stack(tuple(getattr(state, name) for name in _SCALAR_FIELDS))
                )
            )
            & (state.time >= 0)
            & (state.step_count >= 0)
        )
        valid = (
            jnp.all(jnp.isfinite(jnp.stack(masses)), axis=0)
            & jnp.all(jnp.stack(masses) >= 0, axis=0)
            & (state.dry_mass > 0)
            & jnp.isfinite(state.layer_volume)
            & (state.layer_volume > 0)
            & jnp.isfinite(temperature)
            & (temperature >= thermo.minimum_temperature)
            & (temperature <= thermo.maximum_temperature)
            & jnp.isfinite(pressure)
            & (pressure > 0)
            & scalar_valid
        )
        margin = 64 * jnp.finfo(temperature.dtype).eps
        regular = (
            valid
            & _temperature_interior(
                temperature, thermo.minimum_temperature, thermo.maximum_temperature
            )
            & _temperature_interior(
                surface_temperature,
                slab_plan.minimum_temperature,
                slab_plan.maximum_temperature,
            )
            & (jnp.abs(temperature - thermo.reference_temperature) > margin * temperature)
        )
        return InteractiveColumnDiagnostics(
            temperature, density, pressure, humidity, surface_temperature, valid, regular
        )

    def _mixing(self, state, diagnosed, ventilation, shear):
        thickness = state.layer_volume
        distance = 0.5 * (thickness[:-1] + thickness[1:])
        # Virtual potential temperature includes condensate loading. Gravity and
        # reference pressure only parameterize stability, not prognostic dynamics.
        thermo = self.thermodynamics
        gas = (
            state.dry_mass * thermo.dry_gas_constant
            + state.vapor_mass * thermo.vapor_gas_constant
        ) / state.layer_mass
        theta = (
            diagnosed.temperature
            * gas
            / thermo.dry_gas_constant
            * (1e5 / diagnosed.pressure)
            ** (thermo.dry_gas_constant / (thermo.dry_cv + thermo.dry_gas_constant))
        )
        n2 = (
            9.80665
            * (theta[:-1] - theta[1:])
            / (distance * 0.5 * (theta[:-1] + theta[1:]))
        )
        length = self.mixing_length
        frequency2 = shear**2 + (ventilation / jnp.maximum(length, 1.0)) ** 2
        denominator = frequency2 + jnp.maximum(n2, 0.0) / self.critical_richardson
        suppression = jnp.where(
            denominator > 0,
            frequency2 / jnp.where(denominator > 0, denominator, 1.0),
            1.0,
        )
        stirring2 = frequency2 + jnp.maximum(-n2, 0.0)
        stirring = jnp.where(
            stirring2 > 0, jnp.sqrt(jnp.where(stirring2 > 0, stirring2, 1.0)), 0.0
        )
        diffusivity = self.background_diffusivity + suppression * (
            length * ventilation + length**2 * stirring
        )
        volume_flux = diffusivity / distance
        outflow = jnp.concatenate(
            (volume_flux, jnp.zeros_like(thickness[:1]))
        ) + jnp.concatenate((jnp.zeros_like(thickness[:1]), volume_flux))
        # cp/cv bounds the enthalpy transport amplification relative to caloric
        # capacity. This also enforces the species donor-volume CFL.
        ratio = max(
            (thermo.dry_cv + thermo.dry_gas_constant) / thermo.dry_cv,
            (thermo.vapor_cv + thermo.vapor_gas_constant) / thermo.vapor_cv,
            1.0,
        )
        stable = jnp.min(
            jnp.where(
                outflow > 0,
                0.5 * thickness / (ratio * jnp.where(outflow > 0, outflow, 1.0)),
                jnp.inf,
            )
        )
        masses = _masses(state)
        rates = tuple(
            _divergence(
                volume_flux * (m[:-1] / thickness[:-1] - m[1:] / thickness[1:]), m
            )
            for m in masses
        )
        hd, hv, hl, hi = thermo.phase_enthalpies(diagnosed.temperature)
        enthalpy_density = (
            masses[0] * hd
            + masses[1] * hv
            + (masses[2] + masses[4]) * hl
            + (masses[3] + masses[5]) * hi
        ) / thickness
        energy_rate = _divergence(
            volume_flux * (enthalpy_density[:-1] - enthalpy_density[1:]), thickness
        )
        variance_loss = 2.0 * jnp.sum(volume_flux * jnp.diff(diagnosed.temperature) ** 2)
        margin = 64 * jnp.finfo(theta.dtype).eps
        floor_regular = (
            jnp.abs(length - 1.0) > margin * jnp.maximum(jnp.abs(length), 1.0)
        ) | jnp.all(ventilation == 0)
        regular = ((length == 0) | jnp.all(jnp.abs(n2) > margin)) & floor_regular
        return rates, energy_rate, diffusivity, n2, stable, variance_loss, regular

    def _microphysics(self, masses, energy, volume, dt):
        thermo = self.thermodynamics
        dry, vapor, liquid, ice, rain, snow = masses
        temperature, capacity = _temperature(thermo, masses, energy)
        frozen_mass = ice + snow
        liquid_mass = liquid + rain
        thermal = (
            capacity * (temperature - thermo.reference_temperature) / thermo.latent_fusion
        )
        fraction = dt / self.phase_conversion_timescale
        # Fractions retain the correct dilute-phase tangent at zero donor mass;
        # multiplying a total conversion by a zero-mass ratio would lose it.
        melt_cap = jnp.where(
            frozen_mass > 0,
            jnp.maximum(thermal, 0.0) / jnp.where(frozen_mass > 0, frozen_mass, 1.0),
            fraction,
        )
        freeze_cap = jnp.where(
            liquid_mass > 0,
            jnp.maximum(-thermal, 0.0) / jnp.where(liquid_mass > 0, liquid_mass, 1.0),
            fraction,
        )
        melt_fraction = jnp.where(thermal > 0, jnp.minimum(fraction, melt_cap), 0.0)
        freeze_fraction = jnp.where(thermal < 0, jnp.minimum(fraction, freeze_cap), 0.0)
        cloud_conversion = ice * melt_fraction - liquid * freeze_fraction
        falling_conversion = snow * melt_fraction - rain * freeze_fraction
        liquid, ice = liquid + cloud_conversion, ice - cloud_conversion
        rain, snow = rain + falling_conversion, snow - falling_conversion
        temperature, capacity = _temperature(
            thermo, (dry, vapor, liquid, ice, rain, snow), energy
        )
        phase_regular = jnp.all(
            jnp.abs(temperature - thermo.reference_temperature)
            > 64 * jnp.finfo(energy.dtype).eps * temperature
        ) & jnp.all(
            _temperature_interior(
                temperature, thermo.minimum_temperature, thermo.maximum_temperature
            )
        )
        frozen = temperature < thermo.reference_temperature
        es = jnp.where(
            frozen,
            thermo.saturation_pressure(temperature, phase="ice"),
            thermo.saturation_pressure(temperature),
        )
        saturated_mass = es * volume / (thermo.vapor_gas_constant * temperature)
        excess = vapor - saturated_mass
        condensed = dt / self.condensation_timescale * jnp.maximum(excess, 0.0)
        vapor = vapor - condensed
        liquid = liquid + jnp.where(frozen, 0.0, condensed)
        ice = ice + jnp.where(frozen, condensed, 0.0)
        _, hv, hl, hi = thermo.phase_enthalpies(temperature)
        latent_h = hv - jnp.where(frozen, hi, hl)
        latent_e = latent_h - thermo.vapor_gas_constant * temperature
        slope = saturated_mass * (
            latent_h / (thermo.vapor_gas_constant * temperature**2) - 1 / temperature
        )
        limit = (
            0.5
            * self.condensation_timescale
            / (1 + jnp.maximum(slope * latent_e / capacity, 0.0))
        )
        stable = jnp.min(jnp.where(excess > 0, limit, jnp.inf))
        margin = 64 * jnp.finfo(energy.dtype).eps
        regular = phase_regular & jnp.all(
            jnp.abs(excess) > margin * jnp.maximum(vapor + saturated_mass, 1.0)
        )
        # Cloud droplets evaporate faster than falling precipitation. Separate
        # phase saturation laws also permit physically supercooled liquid.
        hydrometeors = [liquid, ice, rain, snow]
        rain_evaporated = jnp.zeros((), energy.dtype)
        for index in range(4):
            temperature, capacity = _temperature(
                thermo, (dry, vapor, *hydrometeors), energy
            )
            regular = regular & jnp.all(
                _temperature_interior(
                    temperature, thermo.minimum_temperature, thermo.maximum_temperature
                )
            )
            phase = "liquid" if index % 2 == 0 else "ice"
            es = thermo.saturation_pressure(temperature, phase=phase)
            saturated_mass = es * volume / (thermo.vapor_gas_constant * temperature)
            deficit = jnp.maximum(saturated_mass - vapor, 0.0)
            tau = (
                self.condensation_timescale
                if index < 2
                else self.rain_evaporation_timescale
            )
            requested = dt / tau * deficit
            evaporated = jnp.minimum(hydrometeors[index], requested)
            # Donor exhaustion is an explicit kinetic branch, not post-step clipping.
            active = hydrometeors[index] > 0
            regular = regular & jnp.all(
                ~active
                | (
                    (
                        jnp.abs(saturated_mass - vapor)
                        > margin * jnp.maximum(saturated_mass + vapor, 1.0)
                    )
                    & (
                        jnp.abs(hydrometeors[index] - requested)
                        > margin * jnp.maximum(hydrometeors[index] + requested, 1.0)
                    )
                )
            )
            vapor = vapor + evaporated
            hydrometeors[index] = hydrometeors[index] - evaporated
            if index >= 2:
                rain_evaporated = rain_evaporated + jnp.sum(evaporated)
            _, hv, hl, hi = thermo.phase_enthalpies(temperature)
            latent_h = hv - (hl if index % 2 == 0 else hi)
            latent_e = latent_h - thermo.vapor_gas_constant * temperature
            slope = saturated_mass * (
                latent_h / (thermo.vapor_gas_constant * temperature**2) - 1 / temperature
            )
            limit = 0.5 * tau / (1 + jnp.maximum(slope * latent_e / capacity, 0.0))
            stable = jnp.minimum(stable, jnp.min(jnp.where(active, limit, jnp.inf)))
        liquid, ice, rain, snow = hydrometeors
        threshold = self.cloud_threshold * dry
        cloud_total = liquid + ice
        fraction = jnp.where(
            cloud_total >= threshold,
            dt
            / self.autoconversion_timescale
            * (1 - threshold / jnp.where(cloud_total > 0, cloud_total, 1.0)),
            0.0,
        )
        liquid_conversion, ice_conversion = liquid * fraction, ice * fraction
        regular = regular & jnp.all(
            (cloud_total == 0)
            | (
                jnp.abs(cloud_total - threshold)
                > margin * jnp.maximum(cloud_total + threshold, 1.0)
            )
        )
        stable = jnp.minimum(
            stable,
            jnp.minimum(self.phase_conversion_timescale, self.autoconversion_timescale),
        )
        return (
            (
                dry,
                vapor,
                liquid - liquid_conversion,
                ice - ice_conversion,
                rain + liquid_conversion,
                snow + ice_conversion,
            ),
            stable,
            rain_evaporated,
            regular,
        )

    def step(
        self,
        state: InteractiveMoistColumnState,
        dt: ArrayLike,
        *,
        solar_down: ArrayLike = 0.0,
        wind_speed: ArrayLike = 5.0,
        measurement_height: ArrayLike = 10.0,
        ventilation: ArrayLike = 0.0,
        shear: ArrayLike = 0.0,
        heating_rate: ArrayLike = 0.0,
        water_flux: ArrayLike = 0.0,
        energy_flux: ArrayLike = 0.0,
    ) -> InteractiveColumnStepResult:
        """Attempt one first-order physical step; any failure rolls back ALL state.

        Solar input and heat rates are W/m²; wind/ventilation are m/s, shear s⁻¹.
        Ventilation and shear broadcast to the n-1 internal interfaces. Prescribed
        heating has its own cumulative external-energy ledger. Surface water and
        energy, fallout, radiation and clock advance only on acceptance.
        Additional interior water/total-energy fluxes are positive downward,
        kg/(m² s) and W/m². Water enters vapor; energy ALREADY includes its
        enthalpy. These use the same closed divergence and donor/domain veto.
        All inexact coefficients and exchanged rates are bound differentiably to
        the initialized state's dtype, including when x64 is globally enabled.
        """
        self = _at_state_dtype(self, state.internal_energy.dtype)
        diagnosed = self.diagnose(state)
        dtype = state.internal_energy.dtype
        dt = jnp.asarray(dt, dtype)
        solar = jnp.asarray(solar_down, dtype)
        wind = jnp.asarray(wind_speed, dtype)
        height = jnp.asarray(measurement_height, dtype)
        if any(x.shape != () for x in (dt, solar, wind, height)):
            raise ValueError("Column dt, solar and surface forcing must be scalar.")
        ventilation = jnp.broadcast_to(
            jnp.asarray(ventilation, dtype), state.dry_mass[:-1].shape
        )
        shear = jnp.broadcast_to(jnp.asarray(shear, dtype), state.dry_mass[:-1].shape)
        water_flux = jnp.broadcast_to(
            jnp.asarray(water_flux, dtype), state.dry_mass[:-1].shape
        )
        energy_flux = jnp.broadcast_to(
            jnp.asarray(energy_flux, dtype), state.dry_mass[:-1].shape
        )
        heating = jnp.broadcast_to(jnp.asarray(heating_rate, dtype), state.dry_mass.shape)
        zero = jnp.zeros((), dtype)
        zeros = jnp.zeros_like(state.dry_mass)
        interfaces = jnp.zeros((state.dry_mass.size + 1,), dtype)
        if self.radiation is None:
            radiation_heat, surface_radiation, space_radiation = zeros, zero, zero
            upward, downward, radiation_valid = interfaces, interfaces, jnp.asarray(True)
        else:
            radiation = _at_state_dtype(
                self.radiation.evaluate(
                    diagnosed.temperature,
                    state.layer_mass,
                    state.vapor_mass,
                    state.cloud_liquid_mass,
                    state.cloud_ice_mass,
                    diagnosed.surface_temperature,
                    solar,
                    rain_mass=state.rain_mass,
                    snow_mass=state.snow_mass,
                ),
                dtype,
            )
            radiation_heat, surface_radiation, space_radiation = (
                radiation.heating,
                radiation.surface_heating,
                radiation.space_heating,
            )
            upward, downward, radiation_valid = (
                radiation.upward_flux,
                radiation.downward_flux,
                jnp.all(radiation.successful),
            )
            if (
                radiation_heat.shape != state.dry_mass.shape
                or upward.shape != interfaces.shape
                or downward.shape != interfaces.shape
                or surface_radiation.shape != ()
                or space_radiation.shape != ()
            ):
                raise ValueError(
                    "An interactive column requires unbatched radiation parameters."
                )
        if self.surface_exchange is None:
            sensible, surface_water, surface_enthalpy, surface_valid = (
                zero,
                zero,
                zero,
                jnp.asarray(True),
            )
            surface_regular = jnp.asarray(True)
        else:
            gas_mass = state.dry_mass[-1] + state.vapor_mass[-1]
            surface = _at_state_dtype(
                self.surface_exchange.evaluate(
                    self.thermodynamics,
                    diagnosed.temperature[-1],
                    gas_mass / state.layer_volume[-1],
                    state.vapor_mass[-1] / gas_mass,
                    diagnosed.pressure[-1],
                    diagnosed.surface_temperature,
                    wind,
                    height,
                ),
                dtype,
            )
            sensible, surface_water, surface_enthalpy = (
                surface.sensible_heat,
                surface.water_mass,
                surface.water_enthalpy,
            )
            surface_valid = jnp.all(surface.successful)
            surface_regular = jnp.all(surface.derivative_valid)
            if any(x.shape != () for x in (sensible, surface_water, surface_enthalpy)):
                raise ValueError(
                    "An interactive column requires scalar bulk exchange parameters."
                )
        rates, mixing_heat, diffusivity, n2, mixing_step, variance, mixing_regular = (
            self._mixing(state, diagnosed, ventilation, shear)
        )
        masses = tuple(m + dt * rate for m, rate in zip(_masses(state), rates))
        dry, vapor, liquid, ice, rain, snow = masses
        vapor = vapor + dt * _divergence(water_flux, vapor)
        vapor = vapor.at[-1].add(dt * surface_water)
        energy = state.internal_energy + dt * (
            mixing_heat
            + radiation_heat
            + heating
            + _divergence(energy_flux, state.internal_energy)
        )
        energy = energy.at[-1].add(dt * (sensible + surface_enthalpy))
        intermediate = (dry, vapor, liquid, ice, rain, snow)
        intermediate_temperature, _ = _temperature(
            self.thermodynamics, intermediate, energy
        )
        transport_valid = (
            jnp.all(jnp.stack(intermediate) >= 0)
            & jnp.all(jnp.isfinite(intermediate_temperature))
            & jnp.all(intermediate_temperature >= self.thermodynamics.minimum_temperature)
            & jnp.all(intermediate_temperature <= self.thermodynamics.maximum_temperature)
            & (state.slab.water_mass - dt * surface_water >= 0)
        )
        transport_regular = jnp.all(
            _temperature_interior(
                intermediate_temperature,
                self.thermodynamics.minimum_temperature,
                self.thermodynamics.maximum_temperature,
            )
        )
        outgoing = jnp.concatenate(
            (jnp.maximum(water_flux, 0), zeros[:1])
        ) + jnp.concatenate((zeros[:1], jnp.maximum(-water_flux, 0)))
        outgoing = outgoing.at[-1].add(jnp.maximum(-surface_water, 0))
        flux_step = jnp.min(
            jnp.where(
                outgoing > 0,
                state.vapor_mass / jnp.where(outgoing > 0, outgoing, 1.0),
                jnp.inf,
            )
        )
        masses, micro_step, rain_evaporated, micro_regular = self._microphysics(
            (dry, vapor, liquid, ice, rain, snow), energy, state.layer_volume, dt
        )
        temperature, _ = _temperature(self.thermodynamics, masses, energy)
        micro_valid = (
            jnp.all(jnp.stack(masses) >= 0)
            & jnp.all(jnp.isfinite(temperature))
            & jnp.all(temperature >= self.thermodynamics.minimum_temperature)
            & jnp.all(temperature <= self.thermodynamics.maximum_temperature)
        )
        micro_regular = micro_regular & jnp.all(
            _temperature_interior(
                temperature,
                self.thermodynamics.minimum_temperature,
                self.thermodynamics.maximum_temperature,
            )
        )
        dry, vapor, liquid, ice, rain, snow = masses
        rain_out = dt * self.rain_fall_speed * rain / state.layer_volume
        snow_out = dt * self.snow_fall_speed * snow / state.layer_volume
        _, _, hl, hi = self.thermodynamics.phase_enthalpies(temperature)
        energy_out = rain_out * hl + snow_out * hi
        rain = rain - rain_out + jnp.concatenate((zeros[:1], rain_out[:-1]))
        snow = snow - snow_out + jnp.concatenate((zeros[:1], snow_out[:-1]))
        energy = energy - energy_out + jnp.concatenate((zeros[:1], energy_out[:-1]))
        precipitation = rain_out[-1] + snow_out[-1]
        precipitation_energy = energy_out[-1]
        # Every flux moves at most one adjacent cell per accepted step. The
        # first-order upwind arrival distribution has a finite-volume time-of-flight.
        travel = jnp.concatenate(
            (
                0.5 * (state.layer_volume[:-1] + state.layer_volume[1:]),
                0.5 * state.layer_volume[-1:],
            )
        )
        potential = 9.80665 * jnp.sum((rain_out + snow_out) * travel)
        slab = WetSlabState(
            state.slab.water_mass - dt * surface_water + precipitation,
            state.slab.energy
            + dt * (surface_radiation - sensible - surface_enthalpy)
            + precipitation_energy,
        )
        candidate = InteractiveMoistColumnState(
            dry,
            vapor,
            liquid,
            ice,
            rain,
            snow,
            energy,
            state.layer_volume,
            slab,
            state.environment_energy + dt * space_radiation,
            state.external_energy + dt * jnp.sum(heating),
            state.precipitated_water + precipitation,
            state.precipitated_energy + precipitation_energy,
            state.evaporated_water + dt * surface_water,
            state.fall_potential_energy + potential,
            state.mixing_temperature_variance_loss + dt * variance,
            state.time + dt,
            state.step_count + 1,
            self.plan_id,
        )
        final_diagnosed = self.diagnose(candidate)
        fall_speed = jnp.maximum(self.rain_fall_speed, self.snow_fall_speed)
        fall_step = jnp.min(
            jnp.where(
                fall_speed > 0,
                state.layer_volume / jnp.where(fall_speed > 0, fall_speed, 1.0),
                jnp.inf,
            )
        )
        stable = jnp.minimum(
            flux_step, jnp.minimum(mixing_step, jnp.minimum(micro_step, fall_step))
        )
        water_residual = candidate.total_water - state.total_water
        external = dt * jnp.sum(heating)
        energy_residual = candidate.total_energy - state.total_energy - external
        dry_residual = jnp.sum(candidate.dry_mass) - jnp.sum(state.dry_mass)
        eps = jnp.finfo(dtype).eps
        energy_scale = (
            jnp.sum(jnp.abs(state.internal_energy))
            + jnp.abs(state.slab.energy)
            + jnp.abs(state.environment_energy)
            + jnp.abs(external)
            + dt
            * (
                jnp.sum(jnp.abs(radiation_heat))
                + jnp.abs(surface_radiation)
                + jnp.abs(space_radiation)
            )
        )
        successful = (
            jnp.all(diagnosed.successful)
            & jnp.all(final_diagnosed.successful)
            & self._parameters_valid()
            & radiation_valid
            & surface_valid
            & transport_valid
            & micro_valid
            & jnp.all(jnp.isfinite(water_flux))
            & jnp.all(jnp.isfinite(energy_flux))
            & jnp.isfinite(dt)
            & (dt > 0)
            & (dt <= stable)
            & jnp.all(jnp.isfinite(heating))
            & jnp.isfinite(solar)
            & (solar >= 0)
            & jnp.isfinite(wind)
            & (wind >= 0)
            & jnp.isfinite(height)
            & (height > 0)
            & jnp.all(jnp.isfinite(ventilation) & (ventilation >= 0))
            & jnp.all(jnp.isfinite(shear))
            & (jnp.abs(water_residual) <= 128 * eps * jnp.maximum(state.total_water, 1.0))
            & (
                jnp.abs(dry_residual)
                <= 128 * eps * jnp.maximum(jnp.sum(state.dry_mass), 1.0)
            )
            & (jnp.abs(energy_residual) <= 256 * eps * jnp.maximum(energy_scale, 1.0))
        )
        chosen = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        regular = (
            successful
            & jnp.all(diagnosed.derivative_valid)
            & jnp.all(final_diagnosed.derivative_valid)
            & transport_regular
            & micro_regular
            & mixing_regular
            & surface_regular
            & (dt < stable * (1 - 64 * eps))
            & (slab.water_mass > 64 * eps)
        )
        accepted = lambda value: jnp.where(successful, value, jnp.zeros_like(value))
        return InteractiveColumnStepResult(
            chosen,
            successful,
            regular,
            stable,
            water_residual,
            energy_residual,
            dry_residual,
            accepted(precipitation),
            accepted(precipitation_energy),
            accepted(rain_evaporated),
            accepted(sensible),
            accepted(surface_water),
            accepted(surface_enthalpy),
            accepted(radiation_heat),
            accepted(surface_radiation),
            accepted(space_radiation),
            accepted(upward),
            accepted(downward),
            diffusivity,
            n2,
            accepted(potential / dt),
            accepted(variance),
        )

    def fixed_step_method(self) -> InteractiveColumnFixedStepMethod:
        """Adapt to native fixed-step replay, continuation, retention and lifecycle."""
        return InteractiveColumnFixedStepMethod(self)

    def advance(
        self, state: InteractiveMoistColumnState, dt: ArrayLike, steps: int, **forcing
    ):
        """Convenience scan; after the first rejection the continuation is frozen."""
        if int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer.")

        def body(carry, _):
            current, valid = carry
            result = self.step(current, dt, **forcing)
            selected = jax.tree.map(
                lambda new, old: jnp.where(valid, new, old), result.state, current
            )
            valid = valid & result.successful
            return (selected, valid), valid

        (final, _), valid = jax.lax.scan(
            body, (state, jnp.asarray(True)), None, length=int(steps)
        )
        return final, valid

    def save_checkpoint(
        self, path: str | Path, state: InteractiveMoistColumnState
    ) -> Path:
        if state.plan_id != self.plan_id or not bool(
            jnp.all(self.diagnose(state).successful)
        ):
            raise ValueError("Cannot checkpoint an incompatible or inadmissible column.")
        arrays = {name: getattr(state, name) for name in _LAYER_FIELDS + _SCALAR_FIELDS}
        arrays.update(
            slab_water_mass=state.slab.water_mass, slab_energy=state.slab.energy
        )
        arrays.update(
            {f"parameter_{i}": leaf for i, leaf in enumerate(jax.tree.leaves(self))}
        )
        return write_array_archive(
            path,
            manifest={
                "kind": "interactive-moist-column-checkpoint",
                "plan_id": self.plan_id,
            },
            arrays=arrays,
        )

    def load_checkpoint(self, path: str | Path) -> InteractiveMoistColumnState:
        manifest, arrays = read_array_archive(path)
        if (
            set(manifest) != {"kind", "plan_id", "arrays"}
            or manifest["kind"] != "interactive-moist-column-checkpoint"
            or manifest["plan_id"] != self.plan_id
        ):
            raise ValueError(
                "Checkpoint belongs to a different interactive process plan."
            )
        parameters = {
            f"parameter_{i}": np.asarray(leaf)
            for i, leaf in enumerate(jax.tree.leaves(self))
        }
        names = set(_LAYER_FIELDS + _SCALAR_FIELDS) | {"slab_water_mass", "slab_energy"}
        if set(arrays) != names | set(parameters):
            raise ValueError("Checkpoint has incomplete or unknown inventories.")
        for name, value in parameters.items():
            if (
                not np.array_equal(arrays[name], value)
                or arrays[name].dtype != value.dtype
            ):
                raise ValueError(
                    "Checkpoint numeric physics parameters differ from this plan."
                )
        shape = arrays["dry_mass"].shape
        if len(shape) != 1 or shape[0] < 1:
            raise ValueError("Checkpoint requires a nonempty layer axis.")
        for name in names:
            value = arrays[name]
            expected_shape = shape if name in _LAYER_FIELDS else ()
            kind = "iu" if name == "step_count" else "f"
            if (
                value.shape != expected_shape
                or value.dtype.kind not in kind
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(f"Invalid checkpoint inventory {name!r}.")
            if np.dtype(jax.dtypes.canonicalize_dtype(value.dtype)) != value.dtype:
                raise ValueError("Checkpoint requires unavailable JAX precision.")
        state = InteractiveMoistColumnState(
            **{
                name: jnp.asarray(arrays[name]) for name in _LAYER_FIELDS + _SCALAR_FIELDS
            },
            slab=WetSlabState(
                jnp.asarray(arrays["slab_water_mass"]), jnp.asarray(arrays["slab_energy"])
            ),
            plan_id=self.plan_id,
        )
        if not bool(jnp.all(self.diagnose(state).successful)):
            raise ValueError("Checkpoint thermodynamic inventories are inadmissible.")
        return state


class InteractiveColumnFixedStepMethod(AbstractFixedStepMethod):
    """Native method adapter; ``args`` is a dictionary of physical forcing values."""

    plan: InteractiveMoistColumnPlan
    method_id: str = eqx.field(static=True)

    def __init__(self, plan: InteractiveMoistColumnPlan):
        self.plan = plan
        self.method_id = canonical_fingerprint(
            {"kind": "interactive-column-fixed-step", "plan": plan.plan_id}
        )

    def step(self, step_index, time, state, step_size, args, /) -> FixedStepResult:
        del step_index
        size = jnp.asarray(step_size, state.internal_energy.dtype)
        scheduled_time = jnp.asarray(time, state.time.dtype)
        result = self.plan.step(state, size, **({} if args is None else args))
        tolerance = (
            64
            * jnp.finfo(state.time.dtype).eps
            * jnp.maximum(jnp.abs(scheduled_time), 1.0)
        )
        consistent = jnp.abs(scheduled_time - state.time) <= tolerance
        successful = result.successful & consistent
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), result.state, state
        )
        return FixedStepResult(
            result.state,
            accepted,
            successful,
            jnp.maximum(size - result.stable_step, 0.0),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(1, jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), state.time.dtype),
        )


__all__ = [
    "InteractiveMoistColumnState",
    "InteractiveColumnDiagnostics",
    "InteractiveColumnStepResult",
    "InteractiveMoistColumnPlan",
    "InteractiveColumnFixedStepMethod",
]
