#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coolant boundary laws on the waterside surface of the cladding.

``PrescribedWallTemperature`` imposes a tabulated waterside surface temperature
per axial zone and time; it does not predict the cladding surface temperature.

``SingleChannelCoolant`` is the single closed-channel model of FRAPCON-4.0
(Geelhood et al., PNNL-19418 Vol. 1 Rev. 2, 2015, Section 2.3): a steady axial
enthalpy balance, forced convection by Dittus-Boelter (Eq. (2.7)) with crud in
series (Eq. (2.8)), and subcooled nucleate boiling by Jens-Lottes (Eq. (2.9);
Jens and Lottes, ANL-4627, 1951, Section IV). FRAPCON takes the surface
temperature as the minimum of the forced-convection and nucleate-boiling
temperatures (Eqs. (2.4)-(2.5)); both are increasing in the heat flux, so the
flux at a given wall temperature is the maximum of the two branch fluxes.
Oxide conduction (the ``Delta T_ox`` term shared by both FRAPCON branches) is
computed by the cladding model: ``wall_temperature`` here is the waterside
surface of the oxide. Water properties are IAPWS-IF97 region 1 with IAPWS
industrial transport (`phydrax.materials`). Bulk boiling and zero or reversed
flow are outside the model and reported by `SingleChannelCoolant.admissible`;
``bulk_enthalpy`` divides by the mass flux and requires it to be positive.
"""

from __future__ import annotations

import math
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...materials import (
    iapws97_liquid_enthalpy,
    iapws97_liquid_state,
    iapws97_liquid_temperature,
    iapws97_saturation_temperature,
    PropertyValidityRange,
)
from ...typing import as_host_array, ConvertibleToArray, Dim, Float64, HostFloat64, Scope
from ._constants import SECONDS_PER_HOUR
from ._identity import AxialLayerDim


# Exact unit definitions for evaluating Jens-Lottes in its source variables:
# pound-force (0.45359237 kg * 9.80665 m/s^2) per square inch (0.0254 m)^2.
_PASCAL_PER_PSI = 0.45359237 * 9.80665 / (0.0254 * 0.0254)
# International Table Btu (1055.05585262 J) per hour per square foot
# (0.09290304 m^2).
_WATT_PER_SQUARE_METER_PER_BTU_HOUR_SQUARE_FOOT = 1055.05585262 / (
    SECONDS_PER_HOUR * 0.09290304
)
_KELVIN_PER_DEGREE_FAHRENHEIT = 5.0 / 9.0
# 0 F on the Rankine scale.
_RANKINE_AT_ZERO_FAHRENHEIT = 459.67
# ANL-4627 Section IV: Delta t_sat = 60 (q''/10^6)^(1/4) / exp(p/900),
# q'' in Btu/(hr ft^2), p in psia, Delta t_sat in F.
_JENS_LOTTES_FLUX_SCALE_BTU_HOUR_SQUARE_FOOT = 1.0e6

# IAPWS-IF97 region 1 bounds (R7-97(2012), Section 5.1).
_REGION1_MINIMUM_TEMPERATURE_K = 273.15
_REGION1_MAXIMUM_TEMPERATURE_K = 623.15
_REGION1_MINIMUM_PRESSURE_PA = 611.213
_REGION1_MAXIMUM_PRESSURE_PA = 100.0e6


class WallTemperatureBreakpointDim(Dim):
    """Time breakpoints of a prescribed wall-temperature table."""


@final
class PrescribedWallTemperature(StrictModule, NonTrainableState):
    """Waterside surface temperature, piecewise linear in time per axial zone.

    ``times`` [s] are strictly increasing breakpoints and ``wall_temperatures``
    [K] the surface temperature of each axial zone at each breakpoint. Between
    breakpoints the temperature is the exact linear interpolant; evaluation
    outside ``[times[0], times[-1]]`` is refused.
    """

    __strict_contract__ = True

    times: Float64[WallTemperatureBreakpointDim]
    wall_temperatures: Float64[WallTemperatureBreakpointDim, AxialLayerDim]
    table_id: str = eqx.field(static=True)

    def __init__(
        self, times: ConvertibleToArray, wall_temperatures: ConvertibleToArray, /
    ) -> None:
        scope = Scope()
        host_times = as_host_array(
            times, HostFloat64[WallTemperatureBreakpointDim], "times", scope=scope
        )
        host_temperatures = as_host_array(
            wall_temperatures,
            HostFloat64[WallTemperatureBreakpointDim, AxialLayerDim],
            "wall_temperatures",
            scope=scope,
        )
        if host_times.shape[0] < 2:
            raise ValueError("A wall-temperature table needs at least two breakpoints.")
        if not np.all(np.isfinite(host_times)) or np.any(np.diff(host_times) <= 0.0):
            raise ValueError(
                "Wall-temperature times must be finite and strictly increasing."
            )
        if host_temperatures.shape[1] < 1:
            raise ValueError("A wall-temperature table needs at least one axial zone.")
        if not np.all(np.isfinite(host_temperatures)) or np.any(host_temperatures <= 0.0):
            raise ValueError(
                "Wall temperatures must be finite and positive kelvin values."
            )
        self.times = jnp.asarray(host_times)
        self.wall_temperatures = jnp.asarray(host_temperatures)
        self.table_id = canonical_fingerprint(
            {
                "kind": "prescribed-wall-temperature",
                "times": host_times,
                "wall_temperatures": host_temperatures,
            }
        )

    def wall_temperature(self, time: ArrayLike, /) -> Array:
        """Surface temperature [K] of every axial zone at ``time`` [s].

        A query of shape ``S`` returns shape ``S + (zones,)``.
        """

        return linear_interpolate(
            self.times, self.wall_temperatures, time, axis=0, bounds="error"
        ).values


def _host_parameter(value: ArrayLike, name: str, /, *, positive: bool) -> float:
    host = np.asarray(value, dtype=np.float64)
    if host.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    number = float(host)
    if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {requirement}.")
    return number


def _kelvin_from_fahrenheit(temperature_degf: float, /) -> float:
    return (
        temperature_degf + _RANKINE_AT_ZERO_FAHRENHEIT
    ) * _KELVIN_PER_DEGREE_FAHRENHEIT


def _jens_lottes_validity() -> PropertyValidityRange:
    # ANL-4627 Section IV: UCLA and MIT local-boiling data cover 85-2500 psia,
    # water temperatures 300-600 F, and heat fluxes to 2,500,000 Btu/(hr ft^2).
    return PropertyValidityRange(
        {
            "pressure": (85.0 * _PASCAL_PER_PSI, 2500.0 * _PASCAL_PER_PSI, "Pa"),
            "bulk_temperature": (
                _kelvin_from_fahrenheit(300.0),
                _kelvin_from_fahrenheit(600.0),
                "K",
            ),
            "heat_flux": (
                0.0,
                2.5e6 * _WATT_PER_SQUARE_METER_PER_BTU_HOUR_SQUARE_FOOT,
                "W/m^2",
            ),
        },
        source_id="jens-lottes-anl-4627-1951:section-iv",
    )


@final
class SingleChannelCoolant(StrictModule):
    """FRAPCON-4.0 single closed-channel coolant for one rod.

    Geometry: ``hydraulic_diameter`` [m], ``flow_area_per_rod`` [m^2] and the
    ``crud_resistance`` [m^2 K/W] (FRAPCON Eq. (2.8) writes it as
    crud thickness over crud conductivity). Correlation parameters default to
    their source values: Dittus-Boelter ``0.023 Re^0.8 Pr^0.4`` (FRAPCON-4.0
    Eq. (2.7)) and Jens-Lottes ``60 F`` at ``10^6 Btu/(hr ft^2)`` with pressure
    scale ``900 psia`` (ANL-4627 Section IV; FRAPCON-4.0 Eq. (2.9)).

    Operating conditions (inlet temperature [K], pressure [Pa], mass flux
    [kg/(m^2 s)]) vary in time and are method arguments; every method
    broadcasts elementwise.
    """

    hydraulic_diameter: Array
    flow_area_per_rod: Array
    crud_resistance: Array
    dittus_boelter_coefficient: Array
    reynolds_exponent: Array
    prandtl_exponent: Array
    jens_lottes_coefficient_degf: Array
    jens_lottes_pressure_scale_psia: Array
    nucleate_boiling_validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        hydraulic_diameter: ArrayLike,
        flow_area_per_rod: ArrayLike,
        crud_resistance: ArrayLike = 0.0,
        dittus_boelter_coefficient: ArrayLike = 0.023,
        reynolds_exponent: ArrayLike = 0.8,
        prandtl_exponent: ArrayLike = 0.4,
        jens_lottes_coefficient_degf: ArrayLike = 60.0,
        jens_lottes_pressure_scale_psia: ArrayLike = 900.0,
    ) -> None:
        parameters = {
            "hydraulic_diameter": _host_parameter(
                hydraulic_diameter, "hydraulic_diameter", positive=True
            ),
            "flow_area_per_rod": _host_parameter(
                flow_area_per_rod, "flow_area_per_rod", positive=True
            ),
            "crud_resistance": _host_parameter(
                crud_resistance, "crud_resistance", positive=False
            ),
            "dittus_boelter_coefficient": _host_parameter(
                dittus_boelter_coefficient, "dittus_boelter_coefficient", positive=True
            ),
            "reynolds_exponent": _host_parameter(
                reynolds_exponent, "reynolds_exponent", positive=True
            ),
            "prandtl_exponent": _host_parameter(
                prandtl_exponent, "prandtl_exponent", positive=True
            ),
            "jens_lottes_coefficient_degf": _host_parameter(
                jens_lottes_coefficient_degf,
                "jens_lottes_coefficient_degf",
                positive=True,
            ),
            "jens_lottes_pressure_scale_psia": _host_parameter(
                jens_lottes_pressure_scale_psia,
                "jens_lottes_pressure_scale_psia",
                positive=True,
            ),
        }
        self.hydraulic_diameter = jnp.asarray(parameters["hydraulic_diameter"])
        self.flow_area_per_rod = jnp.asarray(parameters["flow_area_per_rod"])
        self.crud_resistance = jnp.asarray(parameters["crud_resistance"])
        self.dittus_boelter_coefficient = jnp.asarray(
            parameters["dittus_boelter_coefficient"]
        )
        self.reynolds_exponent = jnp.asarray(parameters["reynolds_exponent"])
        self.prandtl_exponent = jnp.asarray(parameters["prandtl_exponent"])
        self.jens_lottes_coefficient_degf = jnp.asarray(
            parameters["jens_lottes_coefficient_degf"]
        )
        self.jens_lottes_pressure_scale_psia = jnp.asarray(
            parameters["jens_lottes_pressure_scale_psia"]
        )
        self.nucleate_boiling_validity = _jens_lottes_validity()
        self.source_id = "frapcon-4.0-pnnl-19418-vol1-rev2:section-2.3.1-2.3.2"
        self.law_id = canonical_fingerprint(
            {"kind": "single-channel-coolant", "parameters": parameters}
        )

    def bulk_enthalpy(
        self,
        *,
        inlet_temperature: ArrayLike,
        pressure: ArrayLike,
        mass_flux: ArrayLike,
        cumulative_power_to_coolant: ArrayLike,
    ) -> Array:
        """Bulk specific enthalpy [J/kg] from the steady channel energy balance.

        ``h = h_in(T_in, p) + Q / (G A)`` with ``Q`` [W] the power delivered to
        the coolant upstream of the point, ``G`` the mass flux and ``A`` the
        flow area per rod (the enthalpy form of FRAPCON-4.0 Eq. (2.1)). The
        mass flux must be positive.
        """

        inlet = iapws97_liquid_enthalpy(inlet_temperature, pressure)
        power, mass_flux_, inlet = jnp.broadcast_arrays(
            jnp.asarray(cumulative_power_to_coolant), jnp.asarray(mass_flux), inlet
        )
        return inlet + power / (mass_flux_ * self.flow_area_per_rod)

    def bulk_temperature(
        self, *, pressure: ArrayLike, specific_enthalpy: ArrayLike
    ) -> Array:
        """Bulk temperature [K] by the IAPWS-IF97 region 1 backward T(p, h)."""

        return iapws97_liquid_temperature(pressure, specific_enthalpy)

    def wall_heat_flux(
        self,
        *,
        wall_temperature: ArrayLike,
        bulk_temperature: ArrayLike,
        pressure: ArrayLike,
        mass_flux: ArrayLike,
    ) -> Array:
        """Heat flux [W/m^2] from the waterside surface into the coolant.

        Forced convection: ``q = (T_w - T_b) / (1/h_DB + R_crud)`` with
        Dittus-Boelter ``h_DB`` from bulk properties. Above saturation the
        Jens-Lottes flux ``q_JL(T_w - T_sat)`` (no crud resistance: FRAPCON
        assumes boiling through the crud) competes, and FRAPCON's minimum
        surface-temperature rule becomes ``q = max(q_FC, q_JL)``. The branches
        meet continuously at ``T_w = T_sat`` where ``q_JL`` vanishes with zero
        slope.
        """

        wall, bulk, pressure_, mass_flux_ = jnp.broadcast_arrays(
            jnp.asarray(wall_temperature),
            jnp.asarray(bulk_temperature),
            jnp.asarray(pressure),
            jnp.asarray(mass_flux),
        )
        water = iapws97_liquid_state(bulk, pressure_)
        reynolds = mass_flux_ * self.hydraulic_diameter / water.dynamic_viscosity
        prandtl = (
            water.isobaric_heat_capacity
            * water.dynamic_viscosity
            / water.thermal_conductivity
        )
        film = (
            self.dittus_boelter_coefficient
            * water.thermal_conductivity
            / self.hydraulic_diameter
            * reynolds**self.reynolds_exponent
            * prandtl**self.prandtl_exponent
        )
        forced = (wall - bulk) / (1.0 / film + self.crud_resistance)
        saturation = iapws97_saturation_temperature(pressure_)
        boiling = wall > saturation
        superheat_degf = (
            jnp.where(boiling, wall - saturation, 0.0) / _KELVIN_PER_DEGREE_FAHRENHEIT
        )
        pressure_psia = pressure_ / _PASCAL_PER_PSI
        # Inverse of ANL-4627: q'' = 10^6 [Delta t_sat e^(p/900) / 60]^4 Btu/(hr ft^2).
        nucleate = (
            _WATT_PER_SQUARE_METER_PER_BTU_HOUR_SQUARE_FOOT
            * _JENS_LOTTES_FLUX_SCALE_BTU_HOUR_SQUARE_FOOT
            * (
                superheat_degf
                * jnp.exp(pressure_psia / self.jens_lottes_pressure_scale_psia)
                / self.jens_lottes_coefficient_degf
            )
            ** 4
        )
        return jnp.where(boiling, jnp.maximum(forced, nucleate), forced)

    def nucleate_boiling_supported(
        self,
        *,
        wall_temperature: ArrayLike,
        bulk_temperature: ArrayLike,
        pressure: ArrayLike,
        heat_flux: ArrayLike,
    ) -> Array:
        """Whether nucleate boiling is absent or inside the Jens-Lottes data range.

        True for finite inputs where the wall is at or below saturation, or where
        pressure, bulk temperature, and heat flux [W/m^2] lie inside
        ``nucleate_boiling_validity``. Forced convection is not assessed.
        """

        wall, bulk, pressure_, heat_flux_ = jnp.broadcast_arrays(
            jnp.asarray(wall_temperature),
            jnp.asarray(bulk_temperature),
            jnp.asarray(pressure),
            jnp.asarray(heat_flux),
        )
        finite = (
            jnp.isfinite(wall)
            & jnp.isfinite(bulk)
            & jnp.isfinite(pressure_)
            & jnp.isfinite(heat_flux_)
        )
        within_data = self.nucleate_boiling_validity.contains(
            pressure=pressure_,
            bulk_temperature=bulk,
            heat_flux=heat_flux_,
        )
        return finite & (
            (wall <= iapws97_saturation_temperature(pressure_)) | within_data
        )

    def saturation_margin(
        self, *, pressure: ArrayLike, specific_enthalpy: ArrayLike
    ) -> Array:
        """Bulk subcooling margin [J/kg] to the region 1 liquid limit.

        ``h_limit(p) - h`` with ``h_limit`` the IAPWS-IF97 region 1 enthalpy at
        ``min(T_sat(p), 623.15 K)``: the saturated-liquid enthalpy where the
        saturation line bounds region 1, and the region 1 upper temperature
        bound above ``p_s(623.15 K)``.
        """

        pressure_, enthalpy = jnp.broadcast_arrays(
            jnp.asarray(pressure), jnp.asarray(specific_enthalpy)
        )
        limit_temperature = jnp.minimum(
            iapws97_saturation_temperature(pressure_), _REGION1_MAXIMUM_TEMPERATURE_K
        )
        return iapws97_liquid_enthalpy(limit_temperature, pressure_) - enthalpy

    def bulk_quality_admissible(
        self, *, pressure: ArrayLike, specific_enthalpy: ArrayLike
    ) -> Array:
        """Whether the bulk state is subcooled liquid inside IAPWS-IF97 region 1.

        False at or beyond saturated liquid (bulk boiling, quality >= 0, is
        outside the model), below the region 1 minimum temperature, or outside
        the region 1 pressure range. NaN inputs are inadmissible.
        """

        pressure_, enthalpy = jnp.broadcast_arrays(
            jnp.asarray(pressure), jnp.asarray(specific_enthalpy)
        )
        minimum_enthalpy = iapws97_liquid_enthalpy(
            jnp.full_like(pressure_, _REGION1_MINIMUM_TEMPERATURE_K), pressure_
        )
        return (
            (self.saturation_margin(pressure=pressure_, specific_enthalpy=enthalpy) > 0.0)
            & (enthalpy >= minimum_enthalpy)
            & (pressure_ >= _REGION1_MINIMUM_PRESSURE_PA)
            & (pressure_ <= _REGION1_MAXIMUM_PRESSURE_PA)
        )

    def admissible(
        self, *, pressure: ArrayLike, mass_flux: ArrayLike, specific_enthalpy: ArrayLike
    ) -> Array:
        """Whether the channel state is inside the model's range.

        Requires positive finite forced flow (zero and natural-circulation flow
        are outside the model) and `bulk_quality_admissible`.
        """

        pressure_, mass_flux_, enthalpy = jnp.broadcast_arrays(
            jnp.asarray(pressure), jnp.asarray(mass_flux), jnp.asarray(specific_enthalpy)
        )
        return (
            jnp.isfinite(mass_flux_)
            & (mass_flux_ > 0.0)
            & self.bulk_quality_admissible(pressure=pressure_, specific_enthalpy=enthalpy)
        )


__all__ = [
    "PrescribedWallTemperature",
    "SingleChannelCoolant",
    "WallTemperatureBreakpointDim",
]
