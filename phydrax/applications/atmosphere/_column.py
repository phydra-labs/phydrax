#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-volume moist column and paired water/energy process transfers.

The column has fixed geometric volumes, dry masses, and no mechanical motion.
It evolves water mass and internal energy under prescribed heat, surface
exchange, radiation, mixing, and fallout. It is a thermodynamic column, not a
hydrostatic or convective dynamics solver. Vertical axis is last, top to bottom.
"""

from __future__ import annotations

import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_archive import read_array_archive, write_array_archive
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._moist import MoistAdjustmentResult, MoistThermodynamicPlan


def conservative_radiation(
    temperature: ArrayLike,
    layer_mass: ArrayLike,
    heat_capacity: ArrayLike,
    equilibrium_temperature: ArrayLike,
    relaxation_time: ArrayLike,
) -> tuple[Array, Array]:
    """Newtonian cooling W/area and its opposite environmental-reservoir rate.

    ``heat_capacity`` is J/(kg K); this is a grey relaxation parameterization,
    not spectrally resolved radiation. The environment receives exactly the
    negative sum of layer heat sources. ``relaxation_time`` must be positive.
    """
    temperature, mass, capacity, equilibrium, timescale = jnp.broadcast_arrays(
        jnp.asarray(temperature),
        jnp.asarray(layer_mass),
        jnp.asarray(heat_capacity),
        jnp.asarray(equilibrium_temperature),
        jnp.asarray(relaxation_time),
    )
    rate = mass * capacity * (equilibrium - temperature) / timescale
    valid = (mass > 0) & (capacity > 0) & (timescale > 0)
    rate = jnp.where(valid, rate, jnp.nan)
    return rate, -jnp.sum(rate, axis=-1)


def _exchange_flux(quantity, mass, exchange_rate):
    rate = jnp.broadcast_to(jnp.asarray(exchange_rate), quantity[..., :-1].shape)
    interface_mass = (
        2.0 * mass[..., :-1] * mass[..., 1:] / (mass[..., :-1] + mass[..., 1:])
    )
    flux = rate * interface_mass * (quantity[..., :-1] - quantity[..., 1:])
    return jnp.where(
        (rate >= 0) & (mass[..., :-1] > 0) & (mass[..., 1:] > 0), flux, jnp.nan
    )


def _flux_divergence(flux, template):
    boundary = jnp.zeros_like(template[..., :1])
    return jnp.concatenate((boundary, flux), axis=-1) - jnp.concatenate(
        (flux, boundary), axis=-1
    )


def conservative_vertical_mixing(
    specific_quantity: ArrayLike, layer_mass: ArrayLike, exchange_rate: ArrayLike
) -> Array:
    """Closed-boundary conservative exchange of an extensive inventory.

    ``exchange_rate`` is nonnegative s^-1 on the internal interfaces, broadcast
    to (..., layers-1). A harmonic neighboring mass gives the exchange mass
    scale; the returned rate has units (layer_mass * specific_quantity)/s.
    Positivity is a timestep responsibility of the caller, never a clipped sink.
    """
    quantity, mass = jnp.broadcast_arrays(
        jnp.asarray(specific_quantity), jnp.asarray(layer_mass)
    )
    if quantity.ndim < 1:
        raise ValueError("Vertical mixing needs a final layer axis.")
    return _flux_divergence(_exchange_flux(quantity, mass, exchange_rate), quantity)


class MoistPrecipitationResult(StrictModule):
    density: Array
    total_water: Array
    specific_internal_energy: Array
    thermodynamics: MoistAdjustmentResult
    precipitated_water: Array
    precipitated_energy: Array
    successful: Array


def precipitate(
    thermodynamics: MoistThermodynamicPlan,
    density: ArrayLike,
    total_water: ArrayLike,
    specific_internal_energy: ArrayLike,
    layer_volume: ArrayLike,
    *,
    fraction: ArrayLike,
) -> MoistPrecipitationResult:
    """Remove a condensate fraction and its enthalpy, renormalizing total mass.

    Condensed phases have zero specific volume in this model, so their enthalpy
    equals their internal energy. The caller owns the receiving water/energy
    reservoir and must add the returned extensive transfers. Each failed cell
    returns its original inventories and zero transfer, not a partial removal.
    """
    rho, qt, energy, volume, fraction_ = jnp.broadcast_arrays(
        jnp.asarray(density),
        jnp.asarray(total_water),
        jnp.asarray(specific_internal_energy),
        jnp.asarray(layer_volume),
        jnp.asarray(fraction),
    )
    before = thermodynamics.adjust(rho, qt, energy)
    mass = rho * volume
    liquid = mass * before.liquid * fraction_
    ice = mass * before.ice * fraction_
    water = liquid + ice
    _, _, hl, hi = thermodynamics.phase_enthalpies(before.temperature)
    removed_energy = liquid * hl + ice * hi
    remaining_mass = mass - water
    candidate_rho = remaining_mass / volume
    candidate_water = (mass * qt - water) / remaining_mass
    candidate_energy = (mass * energy - removed_energy) / remaining_mass
    after = thermodynamics.adjust(candidate_rho, candidate_water, candidate_energy)
    successful = (
        before.successful
        & after.successful
        & (volume > 0)
        & jnp.isfinite(volume)
        & (fraction_ >= 0)
        & (fraction_ <= 1)
        & (remaining_mass > 0)
    )
    selected = jax.tree.map(
        lambda new, old: jnp.where(successful, new, old), after, before
    )
    return MoistPrecipitationResult(
        jnp.where(successful, candidate_rho, rho),
        jnp.where(successful, candidate_water, qt),
        jnp.where(successful, candidate_energy, energy),
        selected,
        jnp.where(successful, water, 0.0),
        jnp.where(successful, removed_energy, 0.0),
        successful,
    )


class MoistColumnState(StrictModule):
    dry_mass: Array
    vapor_mass: Array
    liquid_mass: Array
    ice_mass: Array
    internal_energy: Array
    layer_volume: Array
    reservoir_water: Array
    reservoir_energy: Array
    environment_energy: Array
    external_energy: Array
    time: Array
    step_count: Array
    cadence_phase: Array
    held_heating: Array
    held_surface_vapor: Array
    held_surface_energy: Array
    plan_id: str = eqx.field(static=True)

    @property
    def water_mass(self) -> Array:
        return self.vapor_mass + self.liquid_mass + self.ice_mass

    @property
    def layer_mass(self) -> Array:
        return self.dry_mass + self.water_mass

    @property
    def total_water(self) -> Array:
        return jnp.sum(self.water_mass, axis=-1) + self.reservoir_water

    @property
    def total_energy(self) -> Array:
        return (
            jnp.sum(self.internal_energy, axis=-1)
            + self.reservoir_energy
            + self.environment_energy
        )


class MoistColumnStepResult(StrictModule):
    state: MoistColumnState
    successful: Array
    water_residual: Array
    energy_residual: Array
    precipitated_water: Array
    precipitated_energy: Array


class MoistColumnPlan(StrictModule, NonTrainableState):
    thermodynamics: MoistThermodynamicPlan
    precipitation_timescale: float = eqx.field(static=True)
    radiation_timescale: float = eqx.field(static=True)
    radiation_temperature: float = eqx.field(static=True)
    mixing_rate: float = eqx.field(static=True)
    surface_temperature: float = eqx.field(static=True)
    forcing_cadence: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: MoistThermodynamicPlan | None = None,
        *,
        precipitation_timescale: float = 600.0,
        radiation_timescale: float = 86400.0,
        radiation_temperature: float = 250.0,
        mixing_rate: float = 0.0,
        surface_temperature: float = 290.0,
        forcing_cadence: int = 1,
    ):
        if thermodynamics is not None and not isinstance(
            thermodynamics, MoistThermodynamicPlan
        ):
            raise TypeError("thermodynamics must be a MoistThermodynamicPlan.")
        values = (
            precipitation_timescale,
            radiation_timescale,
            radiation_temperature,
            surface_temperature,
        )
        if (
            any(not math.isfinite(v) or v <= 0 for v in values)
            or not math.isfinite(mixing_rate)
            or mixing_rate < 0
        ):
            raise ValueError(
                "Column timescales/temperatures must be positive and mixing nonnegative."
            )
        if int(forcing_cadence) != forcing_cadence or forcing_cadence < 1:
            raise ValueError("forcing_cadence must be a positive integer.")
        self.thermodynamics = (
            MoistThermodynamicPlan() if thermodynamics is None else thermodynamics
        )
        self.precipitation_timescale = float(precipitation_timescale)
        self.radiation_timescale = float(radiation_timescale)
        self.radiation_temperature = float(radiation_temperature)
        self.mixing_rate = float(mixing_rate)
        self.surface_temperature = float(surface_temperature)
        self.forcing_cadence = int(forcing_cadence)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-volume-moist-column",
                "thermodynamics": self.thermodynamics.plan_id,
                "precipitation_timescale": self.precipitation_timescale,
                "radiation_timescale": self.radiation_timescale,
                "radiation_temperature": self.radiation_temperature,
                "mixing_rate": self.mixing_rate,
                "surface_temperature": self.surface_temperature,
                "forcing_cadence": self.forcing_cadence,
            }
        )

    def initialize(
        self,
        density: ArrayLike,
        temperature: ArrayLike,
        total_water: ArrayLike,
        layer_volume: ArrayLike,
        *,
        reservoir_water: ArrayLike = 0.0,
        reservoir_energy: ArrayLike = 0.0,
    ) -> MoistColumnState:
        rho, temperature_, qt, volume = jnp.broadcast_arrays(
            jnp.asarray(density),
            jnp.asarray(temperature),
            jnp.asarray(total_water),
            jnp.asarray(layer_volume),
        )
        dtype = jnp.result_type(
            rho.dtype, temperature_.dtype, qt.dtype, volume.dtype, jnp.float32
        )
        rho, temperature_, qt, volume = (
            x.astype(dtype) for x in (rho, temperature_, qt, volume)
        )
        if rho.ndim != 1 or rho.size < 1:
            raise ValueError("A moist column requires one nonempty layer axis.")
        state = self.thermodynamics.equilibrium(rho, temperature_, qt)
        mass = rho * volume
        reservoir_water_ = jnp.asarray(reservoir_water, dtype=mass.dtype)
        reservoir_energy_ = jnp.asarray(reservoir_energy, dtype=mass.dtype)
        if reservoir_water_.shape != () or reservoir_energy_.shape != ():
            raise ValueError("Column reservoir inventories must be scalars.")
        valid = (
            jnp.all(state.successful)
            & jnp.all(jnp.isfinite(volume) & (volume > 0))
            & jnp.isfinite(reservoir_water_)
            & (reservoir_water_ >= 0)
            & jnp.isfinite(reservoir_energy_)
        )
        mass = eqx.error_if(
            mass, ~valid, "Initial moist column or reservoir is inadmissible."
        )
        zero = jnp.zeros((), dtype=mass.dtype)
        count = jnp.zeros((), dtype=jnp.int32)
        return MoistColumnState(
            mass * (1.0 - qt),
            mass * state.vapor,
            mass * state.liquid,
            mass * state.ice,
            mass
            * self.thermodynamics.energy(
                rho, temperature_, state.vapor, state.liquid, state.ice
            ),
            volume,
            reservoir_water_,
            reservoir_energy_,
            zero,
            zero,
            zero,
            count,
            count,
            jnp.zeros_like(mass),
            zero,
            zero,
            self.plan_id,
        )

    def save_checkpoint(self, path: str | Path, state: MoistColumnState) -> Path:
        """Atomically archive all numeric state with its actual static plan identity."""
        if state.plan_id != self.plan_id:
            raise ValueError("Column checkpoint belongs to a different process plan.")
        arrays = {
            "dry_mass": state.dry_mass,
            "vapor_mass": state.vapor_mass,
            "liquid_mass": state.liquid_mass,
            "ice_mass": state.ice_mass,
            "internal_energy": state.internal_energy,
            "layer_volume": state.layer_volume,
            "reservoir_water": state.reservoir_water,
            "reservoir_energy": state.reservoir_energy,
            "environment_energy": state.environment_energy,
            "external_energy": state.external_energy,
            "time": state.time,
            "step_count": state.step_count,
            "cadence_phase": state.cadence_phase,
            "held_heating": state.held_heating,
            "held_surface_vapor": state.held_surface_vapor,
            "held_surface_energy": state.held_surface_energy,
        }
        return write_array_archive(
            path,
            manifest={
                "kind": "moist-column-checkpoint",
                "plan_id": state.plan_id,
                "thermodynamics_id": self.thermodynamics.plan_id,
            },
            arrays=arrays,
        )

    def load_checkpoint(self, path: str | Path) -> MoistColumnState:
        """Read a checksummed, bounded native archive; never trust a template identity."""
        manifest, arrays = read_array_archive(path)
        if (
            set(manifest) != {"kind", "plan_id", "thermodynamics_id", "arrays"}
            or manifest["kind"] != "moist-column-checkpoint"
        ):
            raise ValueError("Archive is not a canonical moist-column checkpoint.")
        if (
            manifest["plan_id"] != self.plan_id
            or manifest["thermodynamics_id"] != self.thermodynamics.plan_id
        ):
            raise ValueError("Column checkpoint belongs to a different process plan.")
        layer_fields = {
            "dry_mass",
            "vapor_mass",
            "liquid_mass",
            "ice_mass",
            "internal_energy",
            "layer_volume",
            "held_heating",
        }
        scalar_fields = {
            "reservoir_water",
            "reservoir_energy",
            "environment_energy",
            "external_energy",
            "time",
            "held_surface_vapor",
            "held_surface_energy",
        }
        integer_fields = {"step_count", "cadence_phase"}
        if set(arrays) != layer_fields | scalar_fields | integer_fields:
            raise ValueError("Column checkpoint has an incomplete or unknown inventory.")
        shape = arrays["dry_mass"].shape
        if len(shape) != 1 or shape[0] < 1:
            raise ValueError("Column checkpoint requires a nonempty layer axis.")
        for name, value in arrays.items():
            expected_shape = shape if name in layer_fields else ()
            expected_kind = "iu" if name in integer_fields else "f"
            if (
                value.shape != expected_shape
                or value.dtype.kind not in expected_kind
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(
                    f"Column checkpoint inventory {name!r} has invalid shape, dtype, or values."
                )
            if np.dtype(jax.dtypes.canonicalize_dtype(value.dtype)) != value.dtype:
                raise ValueError(
                    "Column checkpoint dtype is unavailable under the active JAX precision configuration."
                )
        if (
            np.any(arrays["dry_mass"] <= 0)
            or np.any(arrays["layer_volume"] <= 0)
            or any(
                np.any(arrays[name] < 0)
                for name in ("vapor_mass", "liquid_mass", "ice_mass", "reservoir_water")
            )
            or arrays["time"] < 0
            or arrays["step_count"] < 0
            or arrays["cadence_phase"] != arrays["step_count"] % self.forcing_cadence
        ):
            raise ValueError("Column checkpoint inventories or cadence are inadmissible.")
        return MoistColumnState(
            **{name: jnp.asarray(value) for name, value in arrays.items()},
            plan_id=manifest["plan_id"],
        )

    def diagnose(self, state: MoistColumnState) -> MoistAdjustmentResult:
        if state.plan_id != self.plan_id:
            raise ValueError("Column restart belongs to a different process plan.")
        mass = state.layer_mass
        return self.thermodynamics.adjust(
            mass / state.layer_volume,
            state.water_mass / mass,
            state.internal_energy / mass,
        )

    def step(
        self,
        state: MoistColumnState,
        dt: ArrayLike,
        *,
        heating_rate: ArrayLike = 0.0,
        surface_vapor_flux: ArrayLike = 0.0,
        surface_energy_flux: ArrayLike = 0.0,
    ) -> MoistColumnStepResult:
        """One transactional forward-Euler process step, with exact fallout fraction.

        Heat input is W/m² per layer. Surface vapor is kg/(m² s), positive
        reservoir-to-air; surface energy is W/m², positive reservoir-to-air in
        addition to vapor enthalpy. Only accepted steps advance cadence/time.
        All held forcing and cumulative reservoir inventories live in the state,
        and the native checkpoint envelope also preserves the static plan identity.
        """
        diagnosed = self.diagnose(state)
        mass = state.layer_mass
        dt_ = jnp.asarray(dt, dtype=mass.dtype)
        surface_vapor = jnp.asarray(surface_vapor_flux, dtype=mass.dtype)
        surface_energy = jnp.asarray(surface_energy_flux, dtype=mass.dtype)
        if dt_.shape != () or surface_vapor.shape != () or surface_energy.shape != ():
            raise ValueError("Column dt and surface fluxes must be scalars.")
        heating = jnp.broadcast_to(
            jnp.asarray(heating_rate, dtype=mass.dtype), mass.shape
        )
        refresh = state.cadence_phase == 0
        heating = jnp.where(refresh, heating, state.held_heating)
        surface_vapor = jnp.where(refresh, surface_vapor, state.held_surface_vapor)
        surface_energy = jnp.where(refresh, surface_energy, state.held_surface_energy)
        capacity = self.thermodynamics.heat_capacity(
            diagnosed.vapor, diagnosed.liquid, diagnosed.ice
        )
        radiation, environment = conservative_radiation(
            diagnosed.temperature,
            mass,
            capacity,
            self.radiation_temperature,
            self.radiation_timescale,
        )
        water_flux = _exchange_flux(state.water_mass / mass, mass, self.mixing_rate)
        _, ev, el, ei = self.thermodynamics.phase_energies(diagnosed.temperature)
        water_energy = (
            state.vapor_mass * ev + state.liquid_mass * el + state.ice_mass * ei
        ) / jnp.where(state.water_mass > 0, state.water_mass, 1.0)
        donor_energy = jnp.where(water_flux >= 0, water_energy[:-1], water_energy[1:])
        # Dry mass stays fixed: water carries its own energy, not (e_water-e_dry).
        conductive_flux = (
            _exchange_flux(diagnosed.temperature, mass, self.mixing_rate)
            * 0.5
            * (capacity[:-1] + capacity[1:])
        )
        energy_flux = water_flux * donor_energy + conductive_flux
        water_rate = _flux_divergence(water_flux, mass)
        energy_rate = _flux_divergence(energy_flux, mass)
        vapor_enthalpy = self.thermodynamics.phase_enthalpies(self.surface_temperature)[1]
        surface_heat = surface_energy + surface_vapor * vapor_enthalpy
        water = state.water_mass + dt_ * water_rate
        water = water.at[-1].add(dt_ * surface_vapor)
        energy = state.internal_energy + dt_ * (energy_rate + heating + radiation)
        energy = energy.at[-1].add(dt_ * surface_heat)
        candidate_mass = state.dry_mass + water
        candidate_rho = candidate_mass / state.layer_volume
        # Use the shared helper to keep fallout mass renormalization and calorics identical.
        fallout = precipitate(
            self.thermodynamics,
            candidate_rho,
            water / candidate_mass,
            energy / candidate_mass,
            state.layer_volume,
            fraction=-jnp.expm1(-dt_ / self.precipitation_timescale),
        )
        final_mass = fallout.density * state.layer_volume
        final = fallout.thermodynamics
        rain_water = jnp.sum(fallout.precipitated_water)
        rain_energy = jnp.sum(fallout.precipitated_energy)
        reservoir_water = state.reservoir_water - dt_ * surface_vapor + rain_water
        reservoir_energy = state.reservoir_energy - dt_ * surface_heat + rain_energy
        candidate = MoistColumnState(
            state.dry_mass,
            final_mass * final.vapor,
            final_mass * final.liquid,
            final_mass * final.ice,
            final_mass * fallout.specific_internal_energy,
            state.layer_volume,
            reservoir_water,
            reservoir_energy,
            state.environment_energy + dt_ * environment,
            state.external_energy + dt_ * jnp.sum(heating),
            state.time + dt_,
            state.step_count + 1,
            (state.cadence_phase + 1) % self.forcing_cadence,
            heating,
            surface_vapor,
            surface_energy,
            self.plan_id,
        )
        water_residual = candidate.total_water - state.total_water
        energy_residual = (
            candidate.total_energy - state.total_energy - dt_ * jnp.sum(heating)
        )
        eps = jnp.finfo(mass.dtype).eps
        water_tolerance = 64.0 * eps * jnp.maximum(jnp.abs(state.total_water), 1.0)
        energy_scale = (
            jnp.sum(jnp.abs(state.internal_energy))
            + jnp.abs(state.reservoir_energy)
            + jnp.abs(state.environment_energy)
            + jnp.abs(dt_ * jnp.sum(heating))
        )
        energy_tolerance = 128.0 * eps * jnp.maximum(energy_scale, 1.0)
        successful = (
            jnp.all(diagnosed.successful)
            & jnp.all(fallout.successful)
            & jnp.isfinite(dt_)
            & (dt_ > 0)
            & jnp.all(jnp.isfinite(heating))
            & jnp.isfinite(surface_vapor)
            & jnp.isfinite(surface_energy)
            & (reservoir_water >= 0)
            & jnp.isfinite(reservoir_energy)
            & (jnp.abs(water_residual) <= water_tolerance)
            & (jnp.abs(energy_residual) <= energy_tolerance)
        )
        selected = jax.tree.map(
            lambda new, old: (
                jnp.where(successful, new, old) if eqx.is_array(new) else new
            ),
            candidate,
            state,
        )
        return MoistColumnStepResult(
            selected,
            successful,
            water_residual,
            energy_residual,
            jnp.where(successful, rain_water, 0.0),
            jnp.where(successful, rain_energy, 0.0),
        )

    def advance(
        self,
        state: MoistColumnState,
        dt: ArrayLike,
        steps: int,
        *,
        heating_rate: ArrayLike = 0.0,
        surface_vapor_flux: ArrayLike = 0.0,
        surface_energy_flux: ArrayLike = 0.0,
    ) -> tuple[MoistColumnState, Array]:
        """Scan accepted/rejected steps; return final restart and per-step acceptance."""
        if int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer.")

        def body(current, _):
            result = self.step(
                current,
                dt,
                heating_rate=heating_rate,
                surface_vapor_flux=surface_vapor_flux,
                surface_energy_flux=surface_energy_flux,
            )
            return result.state, result.successful

        return jax.lax.scan(body, state, None, length=int(steps))


__all__ = [
    "MoistColumnPlan",
    "MoistColumnState",
    "MoistColumnStepResult",
    "MoistPrecipitationResult",
    "conservative_radiation",
    "conservative_vertical_mixing",
    "precipitate",
]
