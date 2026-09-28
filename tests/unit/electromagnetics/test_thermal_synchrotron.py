#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, special

from phydrax import ElectromagneticScaleContract, RelativityScaleContract
from phydrax.electromagnetics import ThermalSynchrotronModel
from phydrax.electromagnetics._thermal_synchrotron import _validated_log_bessel_k2


jax.config.update("jax_enable_x64", True)

# CODATA 2022 SI constants, written independently of the scale contract.
_E = 1.602176634e-19
_ME = 9.1093837139e-31
_C = 299792458.0
_EPS0 = 8.8541878188e-12
_KB = 1.380649e-23
_H = 6.62607015e-34


def _mny96_emissivity(
    density: float, temperature: float, field: float, frequency: Any
) -> Any:
    """MNY96 eq. 31 angle-averaged j_nu (W m^-3 Hz^-1 sr^-1) with exact K_2."""
    theta = _KB * temperature / (_ME * _C**2)
    nu_s = 1.5 * _E * field / (2.0 * np.pi * _ME) * theta**2
    x = frequency / nu_s
    shape = (
        4.0505
        * x ** (-1.0 / 6.0)
        * (1.0 + 0.40 * x ** (-0.25) + 0.5316 * x ** (-0.5))
        * np.exp(-1.8899 * x ** (1.0 / 3.0))
    )
    return (
        density
        * _E**2
        * frequency
        / (4.0 * np.pi * _EPS0 * _C * np.sqrt(3.0) * special.kn(2, 1.0 / theta))
        * shape
    )


def _planck(frequency: Any, temperature: float) -> Any:
    return (
        2.0 * _H * frequency**3 / _C**2 / np.expm1(_H * frequency / (_KB * temperature))
    )


def test_validated_log_bessel_k2_matches_trusted_values() -> None:
    arguments = jnp.asarray([1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0])
    trusted_log_values = jnp.asarray(
        [
            14.508657488524674,
            9.903462555643179,
            5.295834109025258,
            0.4854086715656462,
            -10.74700112206937,
            -102.05813713541278,
            -1003.2262122239944,
        ]
    )
    evaluated = eqx.filter_jit(_validated_log_bessel_k2)(arguments)
    np.testing.assert_allclose(evaluated, trusted_log_values, atol=1.0e-9, rtol=0.0)


def test_thermal_synchrotron_attribution_and_scoped_qualification() -> None:
    model = ThermalSynchrotronModel()
    coefficients = eqx.filter_jit(model.evaluate)(
        jnp.asarray(1.0e6),
        jnp.asarray(4.0e10),
        jnp.asarray(1.0),
        jnp.asarray(1.0e11),
        jnp.asarray(0.5),
    )
    assert model.units.number_density_unit.symbol == "m^-3"
    assert model.units.frequency_unit.symbol == "Hz"
    assert model.reference.doi == "10.1086/177422"
    assert model.reference.equation == "31"
    assert model.reference.maximum_shape_relative_error == 0.027
    assert model.reference.authors == (
        "Rohan Mahadevan",
        "Ramesh Narayan",
        "Insu Yi",
    )
    assert model.reference.polarization_status == (
        "unqualified-independent-approximation"
    )
    assert bool(coefficients.evidence.k2_approximation_valid)
    assert bool(coefficients.evidence.emission_reference_valid)
    assert bool(coefficients.evidence.emission_derivative_valid)
    assert not bool(coefficients.evidence.polarization_reference_valid)
    assert not bool(coefficients.evidence.faraday_reference_valid)
    assert not bool(coefficients.evidence.qualified)
    assert not bool(coefficients.evidence.derivative_valid)
    assert coefficients.emission[0] > 0.0
    assert coefficients.emission[1] < 0.0
    assert coefficients.absorption_i > 0.0
    assert coefficients.faraday_rotation > 0.0
    assert coefficients.faraday_conversion > 0.0

    log_temperature_derivative = jax.grad(
        lambda log_temperature: jnp.log(
            model.evaluate(
                1.0e6,
                jnp.exp(log_temperature),
                1.0,
                1.0e11,
                0.5,
            ).emission[0]
        )
    )(jnp.log(jnp.asarray(4.0e10)))
    assert bool(jnp.isfinite(log_temperature_derivative))

    below_published_support = model.evaluate(1.0e6, 1.0e10, 1.0, 1.0e11, 0.5)
    assert bool(below_published_support.evidence.physically_valid)
    assert not bool(below_published_support.evidence.in_domain)
    assert not bool(below_published_support.evidence.emission_reference_valid)

    vacuum = model.evaluate(0.0, 1.0e10, 1.0, 1.0e10, 0.5)
    np.testing.assert_allclose(vacuum.emission, 0.0)
    np.testing.assert_allclose(vacuum.propagation_matrix, 0.0)
    assert bool(vacuum.evidence.qualified)
    assert not bool(vacuum.evidence.derivative_valid)

    invalid = model.evaluate(1.0e6, 4.0e10, 1.0, 1.0e11, 1.1)
    assert not bool(invalid.evidence.physically_valid)
    assert not bool(invalid.evidence.qualified)
    assert bool(jnp.all(jnp.isnan(invalid.emission)))


def test_mny96_route_is_equation_31_with_codata_2022_constants() -> None:
    frequency = np.geomspace(1.0e10, 1.0e13, 7)
    coefficients = ThermalSynchrotronModel().evaluate(
        2.0e12, 1.0e11, 3.0e-2, jnp.asarray(frequency), 0.3
    )
    expected = _mny96_emissivity(2.0e12, 1.0e11, 3.0e-2, frequency)
    np.testing.assert_allclose(coefficients.emission[:, 0], expected, rtol=1.0e-9)
    # Kirchhoff with the Planck function.
    np.testing.assert_allclose(
        coefficients.absorption_i, expected / _planck(frequency, 1.0e11), rtol=1.0e-9
    )


def test_mny96_route_refuses_non_si_scales() -> None:
    code = ElectromagneticScaleContract.code_units(
        RelativityScaleContract.si().dimensional_scale,
        ElectromagneticScaleContract.si().charge_unit,
        gravitational_constant=1,
        speed_of_light=1,
        reduced_planck_constant=1,
        boltzmann_constant=1,
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=1,
        constant_set_id="unit-test",
    )
    with pytest.raises(ValueError, match="SI"):
        ThermalSynchrotronModel(scale=code)


def test_mny96_gray_means_match_spectral_quadrature_and_flag_rosseland() -> None:
    density, temperature, field, radiation = 1.0e13, 1.0e11, 1.0e-1, 5.0e10
    means = ThermalSynchrotronModel().gray_means(density, temperature, field, radiation)
    theta = _KB * temperature / (_ME * _C**2)
    nu_s = 1.5 * _E * field / (2.0 * np.pi * _ME) * theta**2
    log_lower, log_upper = np.log(1.0e-6 * nu_s), np.log(1.0e6 * nu_s)

    def emission(log_nu: float) -> float:
        nu = np.exp(log_nu)
        return float(_mny96_emissivity(density, temperature, field, nu) * nu)

    def absorption_at_radiation(log_nu: float) -> float:
        nu = np.exp(log_nu)
        alpha = _mny96_emissivity(density, temperature, field, nu) / _planck(
            nu, temperature
        )
        return float(alpha * _planck(nu, radiation) * nu)

    stefan_over_pi = 2.0 * np.pi**4 * _KB**4 / (15.0 * _H**3 * _C**2)
    planck_emission = integrate.quad(emission, log_lower, log_upper, limit=400)[0]
    planck_absorption = integrate.quad(
        absorption_at_radiation, log_lower, log_upper, limit=400
    )[0]
    np.testing.assert_allclose(
        means.planck_emission,
        planck_emission / (stefan_over_pi * temperature**4),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        means.planck_absorption,
        planck_absorption / (stefan_over_pi * radiation**4),
        rtol=1.0e-6,
    )
    assert bool(means.emission_supported)
    assert bool(means.absorption_supported)
    # ∂B/∂T peaks near hν ≈ 4kT, far above the fit window: not supported.
    assert not bool(means.rosseland_supported)
