"""Coolant boundary laws: prescribed wall temperature and FRAPCON single channel.

Independent references: exact linear interpolation of the declared table; the
steady energy balance h = h_in + Q/(G A); Dittus-Boelter evaluated by hand from
IAPWS properties (themselves verified in tests/unit/materials/test_water_iapws.py);
Jens-Lottes in its ANL-4627 source units (F, Btu/(hr ft^2), psia) and its SI
restatement in the INL BISON CoolantChannelMaterial documentation
(25 (q'' 1e-6)^0.25 / exp(p / 6.2e6)); FRAPCON-4.0 Eqs. (2.4)-(2.5) minimum rule.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.nuclear_fuel._coolant import (
    PrescribedWallTemperature,
    SingleChannelCoolant,
)
from phydrax.materials import iapws97_liquid_state, iapws97_saturation_temperature


pytestmark = pytest.mark.strict_jax

_PRESSURE = 15.5e6
_MASS_FLUX = 3500.0
_HYDRAULIC_DIAMETER = 0.0118
_FLOW_AREA = 8.8e-5
_PASCAL_PER_PSI = 6894.757293168361
_WATT_PER_SQUARE_METER_PER_BTU_HOUR_SQUARE_FOOT = 3.1545907450533
_BULK_TEMPERATURE = 580.0


def _channel(crud_resistance: float = 0.0) -> SingleChannelCoolant:
    return SingleChannelCoolant(
        hydraulic_diameter=_HYDRAULIC_DIAMETER,
        flow_area_per_rod=_FLOW_AREA,
        crud_resistance=crud_resistance,
    )


def _dittus_boelter_by_hand(bulk_temperature: float, pressure: float) -> float:
    state = iapws97_liquid_state(jnp.asarray(bulk_temperature), jnp.asarray(pressure))
    viscosity = float(state.dynamic_viscosity)
    conductivity = float(state.thermal_conductivity)
    heat_capacity = float(state.isobaric_heat_capacity)
    reynolds = _MASS_FLUX * _HYDRAULIC_DIAMETER / viscosity
    prandtl = heat_capacity * viscosity / conductivity
    return 0.023 * conductivity / _HYDRAULIC_DIAMETER * reynolds**0.8 * prandtl**0.4


def _jens_lottes_superheat_source_units(
    heat_flux: np.ndarray, pressure: float
) -> np.ndarray:
    """ANL-4627 Section IV: Delta t_sat [F] = 60 (q''/1e6)^(1/4) / e^(p/900)."""

    flux_btu = heat_flux / _WATT_PER_SQUARE_METER_PER_BTU_HOUR_SQUARE_FOOT
    superheat_degf = (
        60.0 * (flux_btu / 1.0e6) ** 0.25 / np.exp(pressure / _PASCAL_PER_PSI / 900.0)
    )
    return superheat_degf * 5.0 / 9.0


def test_prescribed_wall_temperature_interpolates_each_layer_exactly() -> None:
    table = PrescribedWallTemperature(
        [0.0, 10.0, 30.0], [[560.0, 570.0], [580.0, 600.0], [580.0, 590.0]]
    )
    np.testing.assert_allclose(table.wall_temperature(5.0), (570.0, 585.0), rtol=1e-15)
    np.testing.assert_allclose(
        table.wall_temperature(jnp.asarray((10.0, 20.0, 30.0))),
        ((580.0, 600.0), (580.0, 595.0), (580.0, 590.0)),
        rtol=1e-15,
    )


def test_prescribed_wall_temperature_refuses_time_outside_table() -> None:
    table = PrescribedWallTemperature([0.0, 10.0], [[560.0], [580.0]])
    with pytest.raises(eqx.EquinoxRuntimeError, match="outside the node interval"):
        jax.block_until_ready(table.wall_temperature(10.5))


@pytest.mark.parametrize(
    ("times", "temperatures"),
    [
        pytest.param([0.0, 0.0], [[560.0], [580.0]], id="non-increasing-times"),
        pytest.param([0.0], [[560.0]], id="single-breakpoint"),
        pytest.param([0.0, 1.0], [[560.0], [-1.0]], id="non-positive-temperature"),
        pytest.param([0.0, np.nan], [[560.0], [580.0]], id="non-finite-time"),
    ],
)
def test_prescribed_wall_temperature_refuses_invalid_tables(
    times: list[float], temperatures: list[list[float]]
) -> None:
    with pytest.raises(ValueError):
        PrescribedWallTemperature(times, temperatures)


def test_single_channel_refuses_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="crud_resistance"):
        SingleChannelCoolant(
            hydraulic_diameter=_HYDRAULIC_DIAMETER,
            flow_area_per_rod=_FLOW_AREA,
            crud_resistance=-1.0e-6,
        )
    with pytest.raises(ValueError, match="flow_area_per_rod"):
        SingleChannelCoolant(
            hydraulic_diameter=_HYDRAULIC_DIAMETER, flow_area_per_rod=0.0
        )


def test_bulk_enthalpy_rise_equals_integrated_power_over_mass_flow() -> None:
    channel = _channel()
    power = jnp.asarray((0.0, 2.5e4, 6.0e4))
    enthalpy = channel.bulk_enthalpy(
        inlet_temperature=jnp.asarray(565.0),
        pressure=jnp.asarray(_PRESSURE),
        mass_flux=jnp.asarray(_MASS_FLUX),
        cumulative_power_to_coolant=power,
    )
    inlet = iapws97_liquid_state(
        jnp.asarray(565.0), jnp.asarray(_PRESSURE)
    ).specific_enthalpy
    np.testing.assert_allclose(enthalpy[0], inlet, rtol=1e-15)
    np.testing.assert_allclose(
        enthalpy - inlet,
        np.asarray(power) / (_MASS_FLUX * _FLOW_AREA),
        rtol=1e-12,
        atol=1e-8,
    )
    # The IF97 backward equation recovers the inlet temperature within its
    # 25 mK consistency bound (R7-97(2012) Eq. (12)).
    recovered = channel.bulk_temperature(
        pressure=jnp.asarray(_PRESSURE), specific_enthalpy=inlet
    )
    assert abs(float(recovered) - 565.0) <= 0.025


@pytest.mark.parametrize("crud", [0.0, 2.0e-5], ids=["clean", "crud"])
def test_forced_convection_matches_hand_computed_dittus_boelter(crud: float) -> None:
    wall = 600.0  # below T_sat(15.5 MPa) = 617.9 K: forced convection only
    flux = _channel(crud).wall_heat_flux(
        wall_temperature=jnp.asarray(wall),
        bulk_temperature=jnp.asarray(_BULK_TEMPERATURE),
        pressure=jnp.asarray(_PRESSURE),
        mass_flux=jnp.asarray(_MASS_FLUX),
    )
    film = _dittus_boelter_by_hand(_BULK_TEMPERATURE, _PRESSURE)
    expected = (wall - _BULK_TEMPERATURE) / (1.0 / film + crud)
    np.testing.assert_allclose(flux, expected, rtol=1e-12)


def test_nucleate_boiling_flux_reproduces_jens_lottes_superheat() -> None:
    saturation = float(iapws97_saturation_temperature(jnp.asarray(_PRESSURE)))
    superheat = np.asarray((3.0, 5.0, 8.0))
    flux = _channel(2.0e-5).wall_heat_flux(
        wall_temperature=jnp.asarray(saturation + superheat),
        bulk_temperature=jnp.asarray(_BULK_TEMPERATURE),
        pressure=jnp.asarray(_PRESSURE),
        mass_flux=jnp.asarray(_MASS_FLUX),
    )
    flux_host = np.asarray(flux)
    np.testing.assert_allclose(
        _jens_lottes_superheat_source_units(flux_host, _PRESSURE), superheat, rtol=1e-12
    )
    # BISON rounds the exact conversions 25.01 K and 900 psia = 6.2053 MPa; at
    # 15.5 MPa the rounding moves the superheat by 0.26 %.
    bison_si = 25.0 * (flux_host * 1.0e-6) ** 0.25 / np.exp(_PRESSURE / 6.2e6)
    np.testing.assert_allclose(bison_si, superheat, rtol=3.0e-3)


@pytest.mark.parametrize("crud", [0.0, 2.0e-5], ids=["clean", "crud"])
def test_wall_heat_flux_satisfies_frapcon_minimum_temperature_rule(crud: float) -> None:
    saturation = float(iapws97_saturation_temperature(jnp.asarray(_PRESSURE)))
    wall = np.linspace(_BULK_TEMPERATURE + 0.5, saturation + 6.0, 241)
    channel = _channel(crud)

    def flux_at(temperature: jax.Array) -> jax.Array:
        return channel.wall_heat_flux(
            wall_temperature=temperature,
            bulk_temperature=jnp.asarray(_BULK_TEMPERATURE),
            pressure=jnp.asarray(_PRESSURE),
            mass_flux=jnp.asarray(_MASS_FLUX),
        )

    flux = np.asarray(flux_at(jnp.asarray(wall)))
    film = _dittus_boelter_by_hand(_BULK_TEMPERATURE, _PRESSURE)
    # FRAPCON-4.0 Eq. (2.4) (no oxide: this surface is the oxide's waterside face)
    # and Eq. (2.5) (no crud term during nucleate boiling).
    forced_wall = _BULK_TEMPERATURE + flux * (1.0 / film + crud)
    boiling_wall = saturation + _jens_lottes_superheat_source_units(flux, _PRESSURE)
    np.testing.assert_allclose(np.minimum(forced_wall, boiling_wall), wall, rtol=1e-12)
    assert np.any(forced_wall < boiling_wall) and np.any(boiling_wall < forced_wall)
    slope = np.asarray(jax.vmap(jax.grad(flux_at))(jnp.asarray(wall)))
    assert np.all(np.isfinite(slope)) and np.all(slope > 0.0)


def test_bulk_boiling_and_stagnant_flow_are_refused() -> None:
    channel = _channel()
    pressure = jnp.asarray(_PRESSURE)
    saturation = iapws97_saturation_temperature(pressure)
    saturated_liquid = iapws97_liquid_state(saturation, pressure).specific_enthalpy
    enthalpy = saturated_liquid + jnp.asarray((-1.0e3, 0.0, 1.0e3))
    margin = channel.saturation_margin(pressure=pressure, specific_enthalpy=enthalpy)
    np.testing.assert_allclose(margin, (1.0e3, 0.0, -1.0e3), atol=1e-6)
    np.testing.assert_array_equal(
        channel.bulk_quality_admissible(pressure=pressure, specific_enthalpy=enthalpy),
        (True, False, False),
    )
    subcooled = saturated_liquid - 1.0e4
    np.testing.assert_array_equal(
        channel.admissible(
            pressure=pressure,
            mass_flux=jnp.asarray((_MASS_FLUX, 0.0, -_MASS_FLUX, np.nan)),
            specific_enthalpy=subcooled,
        ),
        (True, False, False, False),
    )


@pytest.mark.parametrize(
    ("bulk", "heat_flux", "supported"),
    [(580.0, 1.0e6, True), (600.0, 1.0e6, False), (580.0, 1.0e7, False)],
    ids=["measured-domain", "bulk-outside-measurements", "flux-outside-measurements"],
)
def test_nucleate_boiling_source_support_is_distinct_from_liquid_admissibility(
    bulk: float,
    heat_flux: float,
    supported: bool,
) -> None:
    channel = _channel()
    pressure = jnp.asarray(_PRESSURE, dtype=jnp.float64)
    saturation = iapws97_saturation_temperature(pressure)
    enthalpy = iapws97_liquid_state(
        jnp.asarray(bulk, dtype=jnp.float64), pressure
    ).specific_enthalpy
    assert bool(
        channel.admissible(
            pressure=pressure, mass_flux=_MASS_FLUX, specific_enthalpy=enthalpy
        )
    )
    actual = channel.nucleate_boiling_supported(
        wall_temperature=saturation + 1.0,
        bulk_temperature=bulk,
        pressure=pressure,
        heat_flux=heat_flux,
    )
    assert bool(actual) is supported
