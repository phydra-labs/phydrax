"""IAPWS-IF97 region 1/4 liquid water and IAPWS industrial transport.

Reference values are the computer-program verification tables of the IAPWS
releases: R7-97(2012) Tables 5, 7, 35 and 36; R12-08 Table 4; R15-11 Table 7.
The releases give nine significant digits and ask for agreement at 8-byte
precision, so relative tolerances are 1e-8.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.materials import (
    iapws97_liquid_state,
    iapws97_liquid_temperature,
    iapws97_region1_contains,
    iapws97_saturation_pressure,
    iapws97_saturation_temperature,
    iapws2008_viscosity,
)


pytestmark = pytest.mark.strict_jax

_RTOL = 1.0e-8


# R7-97(2012) Table 5: (T [K], p [Pa], v [m^3/kg], h [J/kg], c_p [J/(kg K)]).
@pytest.mark.parametrize(
    ("temperature", "pressure", "volume", "enthalpy", "heat_capacity"),
    [
        pytest.param(
            300.0, 3.0e6, 0.100215168e-2, 0.115331273e6, 0.417301218e4, id="300K-3MPa"
        ),
        pytest.param(
            300.0, 80.0e6, 0.971180894e-3, 0.184142828e6, 0.401008987e4, id="300K-80MPa"
        ),
        pytest.param(
            500.0, 3.0e6, 0.120241800e-2, 0.975542239e6, 0.465580682e4, id="500K-3MPa"
        ),
    ],
)
def test_region1_basic_equation_matches_table5(
    temperature: float,
    pressure: float,
    volume: float,
    enthalpy: float,
    heat_capacity: float,
) -> None:
    state = iapws97_liquid_state(jnp.asarray(temperature), jnp.asarray(pressure))
    np.testing.assert_allclose(1.0 / np.asarray(state.density), volume, rtol=_RTOL)
    np.testing.assert_allclose(state.specific_enthalpy, enthalpy, rtol=_RTOL)
    np.testing.assert_allclose(state.isobaric_heat_capacity, heat_capacity, rtol=_RTOL)


def test_saturation_pressure_matches_table35() -> None:
    temperature = jnp.asarray((300.0, 500.0, 600.0))
    expected = np.asarray((0.353658941e-2, 0.263889776e1, 0.123443146e2)) * 1.0e6
    np.testing.assert_allclose(
        iapws97_saturation_pressure(temperature), expected, rtol=_RTOL
    )


def test_saturation_temperature_matches_table36() -> None:
    pressure = jnp.asarray((0.1e6, 1.0e6, 10.0e6))
    expected = np.asarray((0.372755919e3, 0.453035632e3, 0.584149488e3))
    np.testing.assert_allclose(
        iapws97_saturation_temperature(pressure), expected, rtol=_RTOL
    )


def test_backward_temperature_matches_table7() -> None:
    pressure = jnp.asarray((3.0e6, 80.0e6, 80.0e6))
    enthalpy = jnp.asarray((500.0e3, 500.0e3, 1500.0e3))
    expected = np.asarray((0.391798509e3, 0.378108626e3, 0.611041229e3))
    np.testing.assert_allclose(
        iapws97_liquid_temperature(pressure, enthalpy), expected, rtol=_RTOL
    )


def test_backward_temperature_is_consistent_with_basic_equation() -> None:
    # R7-97(2012) Eq. (12): |T(p, h(T, p)) - T| <= 25 mK over region 1.
    temperature = jnp.linspace(273.15, 623.15, 71)[:, None]
    pressure = jnp.linspace(16.6e6, 100.0e6, 41)[None, :]
    enthalpy = iapws97_liquid_state(temperature, pressure).specific_enthalpy
    recovered = iapws97_liquid_temperature(pressure, enthalpy)
    assert float(jnp.max(jnp.abs(recovered - temperature))) <= 0.025


# R12-08 Table 4 liquid points (industrial form, mu_2 = 1): (T, rho, mu [Pa s]).
@pytest.mark.parametrize(
    ("temperature", "density", "viscosity"),
    [
        pytest.param(298.15, 998.0, 889.735100e-6, id="298K-998"),
        pytest.param(298.15, 1200.0, 1437.649467e-6, id="298K-1200"),
        pytest.param(373.15, 1000.0, 307.883622e-6, id="373K-1000"),
        pytest.param(433.15, 1000.0, 217.685358e-6, id="433K-1000"),
    ],
)
def test_viscosity_matches_r12_table4(
    temperature: float, density: float, viscosity: float
) -> None:
    value = iapws2008_viscosity(jnp.asarray(temperature), jnp.asarray(density))
    np.testing.assert_allclose(value, viscosity, rtol=_RTOL)


# R15-11 Table 7, region 1 industrial verification (including lambda_2):
# (p [Pa], T [K], lambda [W/(m K)], rho [kg/m^3], c_p [J/(kg K)], mu [Pa s]).
@pytest.mark.parametrize(
    ("pressure", "temperature", "conductivity", "density", "heat_capacity", "viscosity"),
    [
        pytest.param(
            20.0e6,
            620.0,
            0.481485195,
            0.613227777e3,
            0.763433705e4,
            0.709051068e-4,
            id="620K-20MPa",
        ),
        pytest.param(
            50.0e6,
            620.0,
            0.545038940,
            0.699226043e3,
            0.532047725e4,
            0.841527945e-4,
            id="620K-50MPa",
        ),
    ],
)
def test_industrial_transport_matches_r15_table7(
    pressure: float,
    temperature: float,
    conductivity: float,
    density: float,
    heat_capacity: float,
    viscosity: float,
) -> None:
    state = iapws97_liquid_state(jnp.asarray(temperature), jnp.asarray(pressure))
    np.testing.assert_allclose(state.thermal_conductivity, conductivity, rtol=_RTOL)
    np.testing.assert_allclose(state.density, density, rtol=_RTOL)
    np.testing.assert_allclose(state.isobaric_heat_capacity, heat_capacity, rtol=_RTOL)
    np.testing.assert_allclose(state.dynamic_viscosity, viscosity, rtol=_RTOL)


def test_region1_lower_pressure_bound_follows_saturation() -> None:
    temperature = jnp.asarray((600.0, 600.0, 273.15, 623.15, 623.16))
    saturation = iapws97_saturation_pressure(temperature)
    pressure = saturation * jnp.asarray((0.999, 1.001, 1.001, 1.001, 1.001))
    contained = iapws97_region1_contains(temperature, pressure)
    np.testing.assert_array_equal(contained, (False, True, True, True, False))
    assert not bool(iapws97_region1_contains(jnp.asarray(300.0), jnp.asarray(100.1e6)))


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        pytest.param(300.0, 3.0e6, id="no-critical-enhancement"),
        pytest.param(620.0, 20.0e6, id="critical-enhancement"),
    ],
)
def test_state_temperature_derivatives_are_finite_and_consistent(
    temperature: float, pressure: float
) -> None:
    def fields(t: jax.Array) -> jax.Array:
        state = iapws97_liquid_state(t, jnp.asarray(pressure))
        return jnp.stack(
            (
                state.density,
                state.specific_enthalpy,
                state.isobaric_heat_capacity,
                state.dynamic_viscosity,
                state.thermal_conductivity,
            )
        )

    t = jnp.asarray(temperature)
    derivative = jax.jacfwd(fields)(t)
    step = 1.0e-3
    central = (fields(t + step) - fields(t - step)) / (2.0 * step)
    assert bool(jnp.all(jnp.isfinite(derivative)))
    np.testing.assert_allclose(derivative, central, rtol=1.0e-5)
