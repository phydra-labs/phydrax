#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Interactive liquid slabs and signed, ventilated air--water exchange.

Numeric kernels use SI, like ``MoistThermodynamicPlan``: temperatures K,
pressure Pa, density kg/m3, height m, speed m/s, inventories kg/m2 and J/m2.
Surface-to-air heat rates are W/m2 and water rates are kg/m2/s. No momentum
stress, wind evolution, sea ice, or additional latent-heat source is implied.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._moist import MoistAdjustmentResult, MoistThermodynamicPlan


class WetSlabState(StrictModule):
    """Liquid water kg/m2 and total slab energy J/m2 in the owning plan's reference."""

    water_mass: Array
    energy: Array


class WetSlabPlan(StrictModule):
    """Dry heat capacity plus liquid-water calorics with one derived temperature.

    ``dry_heat_capacity`` is positive J/m2/K. Energy is
    ``C_dry*(T-T_ref) + M_water*h_liquid(T)``; the native incompressible liquid
    has equal internal energy and enthalpy. Bounds must stay within the moist
    thermodynamic domain and at or above its freezing reference. This model
    rejects freezing rather than silently creating ice or clamping temperature.
    Boiling depends on pressure and is checked by the bulk exchange operator.
    """

    thermodynamics: MoistThermodynamicPlan
    dry_heat_capacity: Array
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: MoistThermodynamicPlan | None = None,
        *,
        dry_heat_capacity: ArrayLike = 2.0e6,
        minimum_temperature: float | None = None,
        maximum_temperature: float | None = None,
    ):
        if thermodynamics is not None and not isinstance(
            thermodynamics, MoistThermodynamicPlan
        ):
            raise TypeError("thermodynamics must be a MoistThermodynamicPlan.")
        thermo = MoistThermodynamicPlan() if thermodynamics is None else thermodynamics
        lower = (
            thermo.reference_temperature
            if minimum_temperature is None
            else float(minimum_temperature)
        )
        upper = (
            thermo.maximum_temperature
            if maximum_temperature is None
            else float(maximum_temperature)
        )
        capacity = jnp.asarray(
            dry_heat_capacity, dtype=jnp.result_type(dry_heat_capacity, 0.0)
        )
        if capacity.shape != ():
            raise ValueError("Dry slab heat capacity must be a scalar.")
        capacity = eqx.error_if(
            capacity,
            ~jnp.isfinite(capacity) | (capacity <= 0),
            "Dry slab heat capacity must be positive and finite.",
        )
        if not (
            math.isfinite(lower)
            and math.isfinite(upper)
            and thermo.reference_temperature
            <= lower
            < upper
            <= thermo.maximum_temperature
        ):
            raise ValueError(
                "Slab bounds must lie in the positive liquid temperature domain."
            )
        self.thermodynamics = thermo
        self.dry_heat_capacity = capacity
        self.minimum_temperature, self.maximum_temperature = lower, upper
        self.plan_id = canonical_fingerprint(
            {
                "kind": "liquid-wet-slab",
                "thermodynamics": thermo.plan_id,
                "minimum_temperature": lower,
                "maximum_temperature": upper,
            }
        )

    def initialize(self, temperature: ArrayLike, water_mass: ArrayLike) -> WetSlabState:
        temperature_, water = jnp.broadcast_arrays(
            jnp.asarray(temperature), jnp.asarray(water_mass)
        )
        _, _, liquid_energy, _ = self.thermodynamics.phase_energies(temperature_)
        energy = (
            self.dry_heat_capacity
            * (temperature_ - self.thermodynamics.reference_temperature)
            + water * liquid_energy
        )
        valid = (
            jnp.isfinite(temperature_)
            & (temperature_ >= self.minimum_temperature)
            & (temperature_ <= self.maximum_temperature)
            & jnp.isfinite(water)
            & (water >= 0)
            & jnp.isfinite(energy)
            & jnp.isfinite(self.dry_heat_capacity)
            & (self.dry_heat_capacity > 0)
        )
        energy = eqx.error_if(
            energy, ~jnp.all(valid), "Initial wet slab is inadmissible."
        )
        return WetSlabState(water, energy)

    def temperature(
        self, state: WetSlabState, thermodynamics: MoistThermodynamicPlan
    ) -> Array:
        """Derive K without certifying candidate inventories; use ``admissible``.

        Reference mismatch is a structural error, including under JIT. State
        arrays never carry a second, independently evolving temperature.
        """
        if thermodynamics.plan_id != self.thermodynamics.plan_id:
            raise ValueError("Wet slab thermodynamic reference does not match.")
        capacity = (
            self.dry_heat_capacity
            + jnp.asarray(state.water_mass) * thermodynamics.liquid_heat_capacity
        )
        # Invalid negative inventories may have zero capacity. Keep the candidate
        # calculation defined; admissible rejects them, never commits a repair.
        denominator = jnp.where(capacity > 0, capacity, 1.0)
        return (
            thermodynamics.reference_temperature + jnp.asarray(state.energy) / denominator
        )

    def admissible(
        self, state: WetSlabState, thermodynamics: MoistThermodynamicPlan
    ) -> Array:
        """Elementwise certification, suitable for atomic coupled proposals."""
        temperature = self.temperature(state, thermodynamics)
        return (
            jnp.isfinite(state.water_mass)
            & (state.water_mass >= 0)
            & jnp.isfinite(state.energy)
            & jnp.isfinite(temperature)
            & (temperature >= self.minimum_temperature)
            & (temperature <= self.maximum_temperature)
            & jnp.isfinite(self.dry_heat_capacity)
            & (self.dry_heat_capacity > 0)
        )


class SurfaceExchangeRates(StrictModule):
    """Signed surface-to-air fluxes, not a finite-step availability guarantee.

    ``sensible_heat`` and ``water_enthalpy`` are W/m2; ``water_mass`` is
    kg/m2/s. The latter energy flux already includes phase-change enthalpy:
    total air heating is exactly sensible_heat + water_enthalpy.
    """

    sensible_heat: Array
    water_mass: Array
    water_enthalpy: Array
    successful: Array
    derivative_valid: Array


class BulkSurfaceExchangePlan(StrictModule):
    """Forced, shallow surface-layer exchange with identifiable numeric coefficients.

    ``air_vapor`` is specific humidity of condensate-free boundary air, not a
    dry-air mixing ratio. Both boundary states use ``air_pressure``; saturated
    surface vapor comes from the native liquid Clausius--Clapeyron law.

    ``stability='bulk-richardson'`` uses virtual temperatures and a first-order
    dry-adiabatic height correction to the air temperature. Its declared
    empirical multiplier is (1+5 Ri)^-2 for stable Ri >= 0 and
    (1-16 Ri)^1/4 for unstable Ri < 0. This is not an iterative Monin--Obukhov
    solution. The unstable branch tends to zero total exchange as wind tends
    to zero; natural convection and gustiness are not modeled. ``neutral``
    applies unmodified coefficients. No stress or mechanical work is returned.

    Heat and moisture coefficients are nonnegative scalar JAX leaves, fixed
    during a forward trajectory but differentiable calibration parameters.
    ``derivative_valid`` excludes active donor, stability and calm-wind branch
    boundaries rather than treating an arbitrary selected AD tangent as valid.
    """

    heat_transfer_coefficient: Array
    moisture_transfer_coefficient: Array
    stability: str = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        heat_transfer_coefficient: ArrayLike = 1.2e-3,
        moisture_transfer_coefficient: ArrayLike = 1.2e-3,
        stability: str = "bulk-richardson",
        gravity: float = 9.80665,
    ):
        heat = jnp.asarray(
            heat_transfer_coefficient,
            dtype=jnp.result_type(heat_transfer_coefficient, 0.0),
        )
        moisture = jnp.asarray(
            moisture_transfer_coefficient,
            dtype=jnp.result_type(moisture_transfer_coefficient, 0.0),
        )
        if heat.shape != () or moisture.shape != ():
            raise ValueError("Heat and moisture transfer coefficients must be scalars.")
        heat = eqx.error_if(
            heat,
            ~jnp.isfinite(heat) | (heat < 0),
            "Heat transfer coefficient must be finite and nonnegative.",
        )
        moisture = eqx.error_if(
            moisture,
            ~jnp.isfinite(moisture) | (moisture < 0),
            "Moisture transfer coefficient must be finite and nonnegative.",
        )
        gravity_ = float(gravity)
        if not math.isfinite(gravity_) or gravity_ <= 0:
            raise ValueError("Gravity must be finite and positive.")
        if stability not in ("neutral", "bulk-richardson"):
            raise ValueError("stability must be neutral or bulk-richardson.")
        self.heat_transfer_coefficient, self.moisture_transfer_coefficient = (
            heat,
            moisture,
        )
        self.stability, self.gravity = stability, gravity_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ventilated-liquid-surface-exchange",
                "stability": stability,
                "gravity": gravity_,
            }
        )

    def evaluate(
        self,
        thermodynamics: MoistThermodynamicPlan,
        air_temperature: ArrayLike,
        air_density: ArrayLike,
        air_vapor: ArrayLike,
        air_pressure: ArrayLike,
        surface_temperature: ArrayLike,
        wind_speed: ArrayLike,
        measurement_height: ArrayLike,
    ) -> SurfaceExchangeRates:
        air_t, density, vapor, pressure, surface_t, speed, height = jnp.broadcast_arrays(
            jnp.asarray(air_temperature),
            jnp.asarray(air_density),
            jnp.asarray(air_vapor),
            jnp.asarray(air_pressure),
            jnp.asarray(surface_temperature),
            jnp.asarray(wind_speed),
            jnp.asarray(measurement_height),
        )
        saturation_pressure = thermodynamics.saturation_pressure(
            surface_t, phase="liquid"
        )
        epsilon = thermodynamics.dry_gas_constant / thermodynamics.vapor_gas_constant
        denominator = pressure - (1.0 - epsilon) * saturation_pressure
        surface_vapor = (
            epsilon * saturation_pressure / jnp.where(denominator > 0, denominator, 1.0)
        )
        cp_air = thermodynamics.heat_capacity(vapor, 0.0, 0.0, at_constant_pressure=True)
        if self.stability == "neutral":
            ventilation = speed
        else:
            virtual_factor = (
                thermodynamics.vapor_gas_constant / thermodynamics.dry_gas_constant - 1.0
            )
            virtual_air = (air_t + self.gravity * height / cp_air) * (
                1.0 + virtual_factor * vapor
            )
            virtual_surface = surface_t * (1.0 + virtual_factor * surface_vapor)
            speed_squared = jnp.where(speed > 0, speed * speed, 1.0)
            richardson = (
                self.gravity
                * height
                * (virtual_air - virtual_surface)
                / (virtual_air * speed_squared)
            )
            # Both inactive branches remain real and differentiable. There is
            # no speed floor, hence no artificial exchange in windless air.
            stable = jnp.maximum(richardson, 0.0)
            unstable = jnp.minimum(richardson, 0.0)
            multiplier = jnp.where(
                richardson >= 0,
                (1.0 + 5.0 * stable) ** -2,
                (1.0 - 16.0 * unstable) ** 0.25,
            )
            ventilation = jnp.where(speed > 0, speed * multiplier, 0.0)
        sensible = (
            density
            * cp_air
            * self.heat_transfer_coefficient
            * ventilation
            * (surface_t - air_t)
        )
        humidity_difference = surface_vapor - vapor
        water = (
            density
            * self.moisture_transfer_coefficient
            * ventilation
            * humidity_difference
        )
        # The disequilibrium selects the donor even at a zero transfer
        # coefficient, preserving its admissible positive-sided derivative.
        donor_temperature = jnp.where(humidity_difference >= 0, surface_t, air_t)
        _, vapor_enthalpy, _, _ = thermodynamics.phase_enthalpies(donor_temperature)
        water_enthalpy = water * vapor_enthalpy
        successful = (
            jnp.isfinite(air_t)
            & (air_t >= thermodynamics.minimum_temperature)
            & (air_t <= thermodynamics.maximum_temperature)
            & jnp.isfinite(surface_t)
            & (surface_t >= thermodynamics.reference_temperature)
            & (surface_t <= thermodynamics.maximum_temperature)
            & jnp.isfinite(density)
            & (density > 0)
            & jnp.isfinite(vapor)
            & (vapor >= 0)
            & (vapor < 1)
            & jnp.isfinite(pressure)
            & (pressure > saturation_pressure)
            & jnp.isfinite(speed)
            & (speed >= 0)
            & jnp.isfinite(height)
            & (height > 0)
            & jnp.isfinite(sensible)
            & jnp.isfinite(water)
            & jnp.isfinite(water_enthalpy)
            & jnp.isfinite(self.heat_transfer_coefficient)
            & (self.heat_transfer_coefficient >= 0)
            & jnp.isfinite(self.moisture_transfer_coefficient)
            & (self.moisture_transfer_coefficient >= 0)
        )
        active = (self.heat_transfer_coefficient > 0) | (
            self.moisture_transfer_coefficient > 0
        )
        derivative_valid = (
            successful
            & (air_t > thermodynamics.minimum_temperature)
            & (air_t < thermodynamics.maximum_temperature)
            & (surface_t > thermodynamics.reference_temperature)
            & (surface_t < thermodynamics.maximum_temperature)
            & (vapor > 0)
            & (~active | (speed > 0))
            & (
                (self.moisture_transfer_coefficient == 0)
                | (humidity_difference != 0)
                | (surface_t == air_t)
            )
        )
        if self.stability == "bulk-richardson":
            derivative_valid = derivative_valid & (~active | (richardson != 0))
        return SurfaceExchangeRates(
            sensible, water, water_enthalpy, successful, derivative_valid
        )


class SurfaceTransferResult(StrictModule):
    """Committed paired inventories and transfers; failed elements are unchanged."""

    slab_state: WetSlabState
    air_water_mass: Array
    air_internal_energy: Array
    air: MoistAdjustmentResult
    water_mass: Array
    energy: Array
    successful: Array


def paired_surface_transfer(
    thermodynamics: MoistThermodynamicPlan,
    slab_plan: WetSlabPlan,
    slab_state: WetSlabState,
    air_dry_mass: ArrayLike,
    air_water_mass: ArrayLike,
    air_internal_energy: ArrayLike,
    air_volume: ArrayLike,
    *,
    water_mass: ArrayLike,
    energy: ArrayLike,
) -> SurfaceTransferResult:
    """Atomically apply already integrated water kg/m2 and total energy J/m2.

    Transfers are positive surface-to-air. For constant rates pass
    ``water_mass=dt*rates.water_mass`` and
    ``energy=dt*(rates.sensible_heat+rates.water_enthalpy)`` after certifying
    ``rates.successful``. A time integrator may instead supply its quadratures.
    The identical transfers are added to air and subtracted from the slab;
    no second latent-heat term, energy correction, or availability cap exists.

    Air is a fixed-volume moist cell with nonzero dry mass, native equilibrium
    adjustment, and volume per area in m. A dew impulse cannot remove more
    than its donor vapor inventory. Each batch element is its own transaction;
    slab depletion, out-of-domain calorics, or invalid air rejects both sides.
    """
    dry, air_water, air_energy, volume, slab_water, slab_energy, water, heat = (
        jnp.broadcast_arrays(
            jnp.asarray(air_dry_mass),
            jnp.asarray(air_water_mass),
            jnp.asarray(air_internal_energy),
            jnp.asarray(air_volume),
            jnp.asarray(slab_state.water_mass),
            jnp.asarray(slab_state.energy),
            jnp.asarray(water_mass),
            jnp.asarray(energy),
        )
    )
    original_slab = WetSlabState(slab_water, slab_energy)
    slab_valid = slab_plan.admissible(original_slab, thermodynamics)
    mass = dry + air_water
    safe_mass = jnp.where(mass > 0, mass, 1.0)
    safe_volume = jnp.where(volume > 0, volume, 1.0)
    before = thermodynamics.adjust(
        mass / safe_volume, air_water / safe_mass, air_energy / safe_mass
    )
    candidate_water = air_water + water
    candidate_energy = air_energy + heat
    candidate_mass = dry + candidate_water
    safe_candidate_mass = jnp.where(candidate_mass > 0, candidate_mass, 1.0)
    after = thermodynamics.adjust(
        candidate_mass / safe_volume,
        candidate_water / safe_candidate_mass,
        candidate_energy / safe_candidate_mass,
    )
    candidate_slab = WetSlabState(slab_water - water, slab_energy - heat)
    successful = (
        slab_valid
        & slab_plan.admissible(candidate_slab, thermodynamics)
        & before.successful
        & after.successful
        & jnp.isfinite(dry)
        & (dry > 0)
        & jnp.isfinite(air_water)
        & (air_water >= 0)
        & jnp.isfinite(air_energy)
        & jnp.isfinite(volume)
        & (volume > 0)
        & jnp.isfinite(water)
        & jnp.isfinite(heat)
        & (candidate_water >= 0)
        & (candidate_mass > 0)
        & (-water <= mass * before.vapor)
    )
    selected_slab = jax.tree.map(
        lambda new, old: jnp.where(successful, new, old), candidate_slab, original_slab
    )
    selected_air = jax.tree.map(
        lambda new, old: jnp.where(successful, new, old), after, before
    )
    return SurfaceTransferResult(
        selected_slab,
        jnp.where(successful, candidate_water, air_water),
        jnp.where(successful, candidate_energy, air_energy),
        selected_air,
        jnp.where(successful, water, 0.0),
        jnp.where(successful, heat, 0.0),
        successful,
    )


__all__ = [
    "BulkSurfaceExchangePlan",
    "SurfaceExchangeRates",
    "SurfaceTransferResult",
    "WetSlabPlan",
    "WetSlabState",
    "paired_surface_transfer",
]
