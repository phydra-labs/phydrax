#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One physical lower boundary for the existing global atmosphere reservoirs."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._moist import MoistThermodynamicPlan
from ._radiation import ColumnRadiationPlan, ColumnRadiationResult
from ._surface import (
    BulkSurfaceExchangePlan,
    SurfaceExchangeRates,
    WetSlabPlan,
    WetSlabState,
)


class GlobalSurfaceFluxes(StrictModule):
    """Stage-local diagnostics; these are not a second prognostic slab."""

    surface_temperature: Array
    radiation: ColumnRadiationResult
    exchange: SurfaceExchangeRates
    successful: Array


class GlobalSurfacePhysics(StrictModule):
    """Grey radiative transfer and ventilated exchange over independent wet slabs.

    The time-independent incident diffuse TOA flux is
    ``solar_constant / 4 * (1 + solar_p2 * P2(sin(latitude)))``. Its spherical
    mean is exactly solar_constant/4 on a grid integrating degree two. This is
    a declared idealized annual-mean distribution, not an orbital/beam model.
    Solar parameters enter the held-forcing payload; SST and all physical
    exchange/radiation fluxes are recomputed from current inventories each stage.

    The bottom model cell represents boundary air. Its gas-only density and
    humidity are passed to bulk exchange, excluding condensate loading. There
    is no unresolved vertical profile interpolation. Prescribed subgrid RMS
    ventilation combines in quadrature with resolved wind, not a wind floor.
    """

    slab: WetSlabPlan
    surface_exchange: BulkSurfaceExchangePlan
    radiation: ColumnRadiationPlan
    solar_constant: float = eqx.field(static=True)
    solar_p2: float = eqx.field(static=True)
    subgrid_wind_speed: float = eqx.field(static=True)
    measurement_height: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        slab: WetSlabPlan,
        surface_exchange: BulkSurfaceExchangePlan,
        radiation: ColumnRadiationPlan,
        *,
        solar_constant: float = 1361.0,
        solar_p2: float = -0.48,
        subgrid_wind_speed: float = 5.0,
        measurement_height: float = 10.0,
    ):
        if not isinstance(slab, WetSlabPlan):
            raise TypeError("slab must be WetSlabPlan.")
        if not isinstance(surface_exchange, BulkSurfaceExchangePlan):
            raise TypeError("surface_exchange must be BulkSurfaceExchangePlan.")
        if not isinstance(radiation, ColumnRadiationPlan):
            raise TypeError("radiation must be ColumnRadiationPlan.")
        values = tuple(
            float(x)
            for x in (solar_constant, solar_p2, subgrid_wind_speed, measurement_height)
        )
        if not all(np.isfinite(x) for x in values):
            raise ValueError("Global surface forcing parameters must be finite.")
        if values[0] < 0 or not -1 <= values[1] <= 2 or values[2] < 0 or values[3] <= 0:
            raise ValueError(
                "Require nonnegative solar/wind, solar_p2 in [-1,2], and positive height."
            )
        self.slab, self.surface_exchange, self.radiation = (
            slab,
            surface_exchange,
            radiation,
        )
        (
            self.solar_constant,
            self.solar_p2,
            self.subgrid_wind_speed,
            self.measurement_height,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "global-wet-slab-grey-surface-boundary",
                "slab": slab.plan_id,
                "surface_exchange": surface_exchange.plan_id,
                "radiation": radiation.plan_id,
                # Numeric component leaves are bound by the native global checkpoint
                # at save/load, not hashed here (constructors may receive tracers).
                "solar_distribution": "diffuse-annual-mean-one-plus-P2",
                "parameters": values,
                "boundary_air": "bottom-cell-gas-only",
                "water_mechanics": "co-moving-injection-instantaneous-precipitation",
            }
        )

    def initialize(self, surface_temperature, surface_water) -> WetSlabState:
        """Return values to place in the global state's existing two reservoirs."""
        return self.slab.initialize(surface_temperature, surface_water)

    def admissible(self, surface_water, surface_energy, thermodynamics) -> Array:
        return self.slab.admissible(
            WetSlabState(surface_water, surface_energy), thermodynamics
        )

    def temperature(self, surface_water, surface_energy, thermodynamics) -> Array:
        return self.slab.temperature(
            WetSlabState(surface_water, surface_energy), thermodynamics
        )

    def solar_forcing(self, colatitude, surface_shape) -> Array:
        p2 = 0.5 * (3.0 * jnp.cos(colatitude) ** 2 - 1.0)
        return jnp.broadcast_to(
            (0.25 * self.solar_constant * (1.0 + self.solar_p2 * p2))[:, None],
            surface_shape,
        )

    def evaluate(
        self,
        thermodynamics: MoistThermodynamicPlan,
        view,
        surface_water,
        surface_energy,
        solar_down,
    ) -> GlobalSurfaceFluxes:
        temperature = self.temperature(surface_water, surface_energy, thermodynamics)
        mass = view.layer_mass
        qv, ql, qi = view.water
        radiation = self.radiation.evaluate(
            view.temperature,
            mass,
            mass * qv,
            mass * ql,
            mass * qi,
            temperature,
            solar_down,
        )
        gas_fraction = 1.0 - ql[..., -1] - qi[..., -1]
        density = view.pressure[..., -1] / (
            view.gas_constant[..., -1] * view.temperature[..., -1]
        )
        wind = jnp.sqrt(
            view.east[..., -1] ** 2
            + view.north[..., -1] ** 2
            + self.subgrid_wind_speed**2
        )
        exchange = self.surface_exchange.evaluate(
            thermodynamics,
            view.temperature[..., -1],
            density * gas_fraction,
            qv[..., -1] / gas_fraction,
            view.pressure[..., -1],
            temperature,
            wind,
            self.measurement_height,
        )
        successful = (
            self.admissible(surface_water, surface_energy, thermodynamics)
            & radiation.successful
            & exchange.successful
        )
        return GlobalSurfaceFluxes(temperature, radiation, exchange, successful)


__all__ = ["GlobalSurfacePhysics", "GlobalSurfaceFluxes"]
