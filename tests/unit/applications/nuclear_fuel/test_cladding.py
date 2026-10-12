"""Stress-relief-annealed Zircaloy-4 cladding laws against published values.

Independent references:
- MATPRO, NUREG/CR-6150 Rev. 2 Vol. 4 (2001): measured moduli Tables 4-15 and
  4-18, typical-tube thermal strains Eqs. (4-37)-(4-39), conductivity Eq.
  (4-26), enthalpy Table 4-5, texture growth rule Eq. (4-104), CHUPTK Eq.
  (A2-14), ZODEN oxide density (Section 5.5.2), ZOEMIS Eqs. (5-25)-(5-28) and
  Table 5-6.
- PNNL-19417 Rev. 2 (2015): CCP Table 3.2, CAGROW Eq. (3.44), and the
  FRAPCON-4.0 CREPR listing (Appendix A.2.8) re-evaluated here in its native
  units (F, MPa, hours).
- FRAPCON-4.0, PNNL-19418 Vol. 1 Rev. 2 (2015): creep strain Eq. (2.91), the
  cubic oxide integral Eq. (2.191), and the Garzarolli step integral Eq. (2.193).
- PNNL-17700 (2008): yield stress Eq. (3) and the strain-rate sensitivity
  statement of Section 2.4.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.nuclear_fuel._cladding_deformation import (
    ZircaloyCreep,
    ZircaloyCreepState,
    ZircaloyIrradiationGrowth,
    ZircaloyPlasticity,
)
from phydrax.applications.nuclear_fuel._cladding_properties import (
    ZIRCALOY_DENSITY,
    ZircaloyElasticity,
    ZircaloyHeatCapacity,
    ZircaloyTexture,
    ZircaloyThermalConductivity,
    ZircaloyThermalExpansion,
)
from phydrax.applications.nuclear_fuel._cladding_surface import (
    ZircaloyEmissivity,
    ZircaloyHydrogenPickup,
    ZircaloyWatersideCorrosion,
    ZirconiaThermalConductivity,
)


pytestmark = pytest.mark.strict_jax

_SECONDS_PER_DAY = 86400.0


@pytest.mark.parametrize(
    ("temperature", "youngs", "shear"),
    [
        # MATPRO Table 4-15 (Busby, Zircaloy-4, 0 % effective cold work).
        pytest.param(297.0, 9.686e10, None, id="busby-297K"),
        pytest.param(516.0, 8.018e10, None, id="busby-516K"),
        pytest.param(644.0, 7.515e10, None, id="busby-644K"),
        # MATPRO Table 4-18 (Northwood et al., annealed Zircaloy-2).
        pytest.param(293.0, 9.67e10, 3.48e10, id="northwood-293K"),
        pytest.param(573.0, 7.99e10, 2.94e10, id="northwood-573K"),
        pytest.param(773.0, 6.78e10, 2.53e10, id="northwood-773K"),
    ],
)
def test_moduli_reproduce_measured_data_within_published_standard_error(
    temperature: float, youngs: float, shear: float | None
) -> None:
    law = ZircaloyElasticity()
    predicted = float(law.youngs_modulus(temperature, 0.0, 0.0, 0.0))
    # MATPRO Section 4.6.1: standard errors 6.4 GPa (CELMOD), 9 GPa (CSHEAR).
    assert abs(predicted - youngs) < 6.4e9
    if shear is not None:
        assert abs(float(law.shear_modulus(temperature, 0.0, 0.0, 0.0)) - shear) < 9.0e9


def test_moduli_follow_published_fluence_saturation_and_phase_boundaries() -> None:
    law = ZircaloyElasticity()
    fresh = float(law.youngs_modulus(600.0, 0.0, 0.0, 0.0))
    saturated = float(law.youngs_modulus(600.0, 1.0e28, 0.0, 0.0))
    # MATPRO Eq. (4-75): K3 -> 0.88 at saturating fluence.
    np.testing.assert_allclose(saturated, fresh / 0.88, rtol=1e-12)
    # MATPRO Eq. (4-72): beta phase above the alpha+beta range.
    np.testing.assert_allclose(
        float(law.youngs_modulus(1300.0, 0.0, 0.0, 0.0)), 9.21e10 - 4.05e7 * 1300.0
    )
    # As-received boundaries: 1094 K and 392.46 (0.1242807^2 + 3.1417) K.
    upper = 392.46 * (0.1242807**2 + 3.1417)
    for boundary in (1094.0, upper):
        below, above = law.youngs_modulus(
            jnp.asarray([boundary - 1.0e-7, boundary + 1.0e-7]), 0.0, 0.0, 0.0
        )
        np.testing.assert_allclose(float(below), float(above), rtol=1e-8)
    assert 0.0 < float(law.poisson_ratio(600.0, 1.0e25, 0.5, 0.0)) < 0.5


def test_thermal_strains_reproduce_matpro_typical_tube() -> None:
    # MATPRO Eqs. (4-37)-(4-39) hold for <cos^2 theta> = 0.71013 and
    # <sin^2 phi> = 0.30822 (hoop weight sin^2 theta cos^2 phi).
    radial = 0.71013
    texture = ZircaloyTexture(
        radial=radial,
        hoop=(1.0 - radial) * (1.0 - 0.30822),
        axial=(1.0 - radial) * 0.30822,
    )
    law = ZircaloyThermalExpansion(texture=texture)
    temperatures = np.array([400.0, 700.0, 1000.0])
    strains = law.principal_strains(temperatures)
    for strain, slope, intercept, slope_digit in (
        (strains.hoop, 6.48e-6, -1.95e-3, 1.0e-8),
        (strains.axial, 5.63e-6, -1.69e-3, 1.0e-8),
        (strains.radial, 1.04e-5, -3.11e-3, 1.0e-7),
    ):
        published = slope * temperatures + intercept
        # Half a unit in the last printed digit of each coefficient.
        tolerance = 0.5 * slope_digit * temperatures + 0.5e-5
        assert np.all(np.abs(np.asarray(strain) - published) <= tolerance)


def test_thermal_expansion_is_zero_at_300K_and_refuses_beyond_alpha_phase() -> None:
    law = ZircaloyThermalExpansion()
    at_reference = law.principal_strains(300.0)
    for strain in (at_reference.radial, at_reference.hoop, at_reference.axial):
        assert abs(float(strain)) < 1e-15
    assert law.validity.exceedance == "refuse"
    inside = law.validity.contains(temperature=jnp.asarray([300.0, 1083.0, 1100.0]))
    np.testing.assert_array_equal(np.asarray(inside), [True, True, False])


def test_invalid_texture_is_refused() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        ZircaloyTexture(radial=0.7, hoop=0.3, axial=0.1)
    with pytest.raises(ValueError, match="one third"):
        ZircaloyIrradiationGrowth(
            texture=ZircaloyTexture(radial=0.2, hoop=0.3, axial=0.5)
        )


def test_conductivity_matches_matpro_rounded_fit() -> None:
    temperatures = np.array([400.0, 800.0, 1600.0])
    predicted = np.asarray(
        ZircaloyThermalConductivity().thermal_conductivity(temperatures)
    )
    # MATPRO Eq. (4-26) prints the fit to three significant figures.
    matpro = (
        7.51
        + 2.09e-2 * temperatures
        - 1.45e-5 * temperatures**2
        + 7.67e-9 * temperatures**3
    )
    rounding = (
        0.005
        + 0.005e-2 * temperatures
        + 0.005e-5 * temperatures**2
        + 0.005e-9 * temperatures**3
    )
    assert np.all(np.abs(predicted - matpro) <= rounding)


@pytest.mark.parametrize(
    ("temperature", "enthalpy"),
    [
        # MATPRO Table 4-5 (units of 1e4 J/kg). Entries from 1233 K upward
        # carry an 800 J/kg arithmetic slip against Table 4-2 and are omitted.
        pytest.param(400.0, 2.915e4, id="400K"),
        pytest.param(640.0, 10.511e4, id="640K"),
        pytest.param(1090.0, 26.396e4, id="1090K"),
        pytest.param(1093.0, 26.52755e4, id="1093K"),
        pytest.param(1153.0, 30.15855e4, id="1153K"),
        pytest.param(1213.0, 34.66855e4, id="1213K"),
    ],
)
def test_enthalpy_matches_matpro_table(temperature: float, enthalpy: float) -> None:
    law = ZircaloyHeatCapacity()
    np.testing.assert_allclose(
        float(law.specific_enthalpy(temperature)), enthalpy, atol=0.1
    )


def test_heat_capacity_is_enthalpy_derivative_and_table_value() -> None:
    law = ZircaloyHeatCapacity()
    # PNNL-19417 Table 3.2.
    np.testing.assert_allclose(float(law.specific_heat_capacity(640.0)), 331.0)
    for temperature in (350.0, 800.0, 1100.0, 1240.0, 1500.0):
        slope = jax.grad(lambda t: law.specific_enthalpy(t))(jnp.asarray(temperature))
        np.testing.assert_allclose(
            float(slope), float(law.specific_heat_capacity(temperature)), rtol=1e-12
        )


def _crepr_listing_rate_per_hour(
    stress_mpa: float, temperature_f: float, flux: float, fluence: float, hours: float
) -> float:
    """FRAPCON-4.0 CREPR (PNNL-19417 Appendix A.2.8), SRA branch, native units."""

    tcak = (temperature_f + 459.67) / 1.8
    fluence_limback = fluence / 10000.0
    modulus = 1.148e5 - 59.9 * tcak
    ai = 650.0 * (1.0 - 0.56 * (1.0 - math.exp(-1.4e-27 * fluence_limback**1.3)))
    thermal = (
        1.08e9
        * (modulus / tcak)
        * math.sinh(ai * abs(stress_mpa) / modulus) ** 2.0
        * math.exp(-201000.0 / (8.314 * tcak))
    )
    tmod = min(max(tcak, 570.0), 625.0)
    irradiation = 4.0985e-24 * flux**0.85 * abs(stress_mpa) * (-7.0237 + 0.0136 * tmod)
    if hours >= 0.1:
        steady = thermal + irradiation
        saturation = 0.0216 * steady**0.109 * (2.0 - math.tanh(3.55e4 * steady)) ** -2.05
        thermal += (
            0.5
            * saturation
            * 52.0
            * steady**0.5
            / hours**0.5
            * math.exp(-52.0 * steady**0.5 * hours**0.5)
        )
    return (thermal + irradiation) * 1.1547


@pytest.mark.parametrize("hours", [0.0, 0.05, 10.0, 1000.0], ids=lambda h: f"{h}h")
@pytest.mark.parametrize("stress_mpa", [40.0, 60.0, 120.0], ids=lambda s: f"{s}MPa")
def test_creep_rate_matches_frapcon_crepr_listing(
    stress_mpa: float, hours: float
) -> None:
    # FRAPCON-4.0 Figure 2.9 condition: 630 F, fast flux 1e18 n/(m^2 s).
    temperature_f, flux, fluence = 630.0, 1.0e18, 3.0e25
    rate = ZircaloyCreep().equivalent_creep_rate(
        stress_mpa * 1.0e6,
        (temperature_f + 459.67) / 1.8,
        flux,
        fluence,
        hours * 3600.0,
    )
    reference = _crepr_listing_rate_per_hour(
        stress_mpa, temperature_f, flux, fluence, hours
    )
    np.testing.assert_allclose(float(rate) * 3600.0, reference, rtol=1e-12)


def test_primary_creep_integrates_to_limback_strain_history() -> None:
    law = ZircaloyCreep()
    stress, temperature, flux, fluence = 80.0e6, 600.0, 5.0e17, 1.0e25
    steady = float(law.equivalent_creep_rate(stress, temperature, flux, fluence, 0.0))
    steady_per_hour = steady * 3600.0 / 1.1547
    saturation = (
        0.0216
        * steady_per_hour**0.109
        * (2.0 - math.tanh(3.55e4 * steady_per_hour)) ** -2.05
    )

    def strain(hours: float) -> float:
        # FRAPCON-4.0 Eq. (2.91), scaled by the CREPR factor 1.1547.
        return 1.1547 * (
            saturation * (1.0 - math.exp(-52.0 * math.sqrt(steady_per_hour * hours)))
            + steady_per_hour * hours
        )

    # t = u^2 removes the t^(-1/2) singularity of the primary term.
    roots = jnp.linspace(1.0, math.sqrt(2000.0), 40001)
    rates = law.equivalent_creep_rate(
        stress, temperature, flux, fluence, roots**2 * 3600.0
    )
    integrand = np.asarray(rates) * 3600.0 * 2.0 * np.asarray(roots)
    integral = np.trapezoid(integrand, np.asarray(roots))
    np.testing.assert_allclose(integral, strain(2000.0) - strain(1.0), rtol=1e-6)


def test_creep_history_partition_and_signed_stress_reversal() -> None:
    law = ZircaloyCreep()
    stress = jnp.diag(jnp.asarray([-80.0e6 / 3.0, 160.0e6 / 3.0, -80.0e6 / 3.0]))
    initial = ZircaloyCreepState.initialize(stress)
    first = law.advance(stress, 600.0, 5.0e17, 1.0e25, initial, 36000.0)
    second = law.advance(stress, 600.0, 5.0e17, 1.0e25, first.state, 36000.0)
    whole = law.advance(stress, 600.0, 5.0e17, 1.0e25, initial, 72000.0)
    # FRAPCON Eq. (2.91): the frozen-rate strain integral telescopes.
    np.testing.assert_allclose(
        np.asarray(first.strain_increment + second.strain_increment),
        np.asarray(whole.strain_increment),
        rtol=1e-12,
    )
    np.testing.assert_allclose(float(second.state.creep_time), 72000.0)
    # FRAPCON §2.4.1.1: reversal starts a new primary episode using the
    # effective stress change (160 MPa), not the new stress magnitude (80 MPa).
    reversed_step = law.advance(-stress, 600.0, 5.0e17, 1.0e25, second.state, 36000.0)
    np.testing.assert_allclose(
        float(reversed_step.state.primary_equivalent_stress), 160.0e6
    )
    np.testing.assert_allclose(float(reversed_step.state.creep_time), 36000.0)
    assert float(reversed_step.strain_increment[1, 1]) < 0.0
    assert float(reversed_step.state.accumulated_equivalent_strain) > float(
        second.state.accumulated_equivalent_strain
    )
    # Candidate calculation does not commit or overwrite its input history.
    np.testing.assert_allclose(float(initial.creep_time), 0.0)


def _fabrication_creep_strain(scale: jax.Array) -> jax.Array:
    # Zero fluence and flux with zero primary-creep reference stress.
    stress = jnp.diag(jnp.asarray([-2.0e6 / 3.0, 4.0e6 / 3.0, -2.0e6 / 3.0]))
    initial = ZircaloyCreepState.initialize(stress)
    return (
        ZircaloyCreep()
        .advance(scale * stress, 600.0, 0.0, 0.0, initial, 15.0)
        .strain_increment
    )


def _fabrication_growth_strain(fluence: jax.Array) -> jax.Array:
    return ZircaloyIrradiationGrowth().principal_strains(fluence).axial


# Fabrication starts with exactly zero fluence, flux and primary-creep stress.
# Fractional powers there have infinite derivatives; ``inf * 0`` must not leak
# NaN into forward sensitivities even along zero directions.
@pytest.mark.parametrize("direction", [0.0, 1.0], ids=["zero-direction", "unit"])
@pytest.mark.parametrize(
    ("function", "point"),
    [(_fabrication_creep_strain, 1.0), (_fabrication_growth_strain, 0.0)],
    ids=["creep-at-zero-fluence", "growth-at-zero-fluence"],
)
def test_fabrication_state_cladding_laws_have_finite_forward_derivatives(
    function: Callable[[jax.Array], jax.Array], point: float, direction: float
) -> None:
    value, tangent = jax.jvp(
        function,
        (jnp.asarray(point, dtype=jnp.float64),),
        (jnp.asarray(direction, dtype=jnp.float64),),
    )
    assert bool(jnp.all(jnp.isfinite(value)))
    assert bool(jnp.all(jnp.isfinite(tangent)))


def _creep_rate(*, flux: float = 5.0e17, fluence: float = 1.0e25) -> jax.Array:
    return ZircaloyCreep().equivalent_creep_rate(80.0e6, 600.0, flux, fluence, 0.0)


def _creep_increment_of_nan_stress() -> jax.Array:
    stress = jnp.full((3, 3), math.nan, dtype=jnp.float64)
    return (
        ZircaloyCreep()
        .advance(
            stress, 600.0, 5.0e17, 1.0e25, ZircaloyCreepState.initialize(stress), 3600.0
        )
        .equivalent_strain_increment
    )


# Fractional powers and the von Mises root guard only an exactly zero argument:
# NaN and negative inputs must not silently become zero.
@pytest.mark.parametrize(
    "evaluate",
    [
        lambda: ZircaloyIrradiationGrowth().principal_strains(math.nan).axial,
        lambda: ZircaloyIrradiationGrowth().principal_strains(-1.0e25).axial,
        lambda: _creep_rate(fluence=math.nan),
        lambda: _creep_rate(fluence=-1.0e25),
        lambda: _creep_rate(flux=math.nan),
        lambda: _creep_rate(flux=-5.0e17),
        _creep_increment_of_nan_stress,
    ],
    ids=[
        "growth-nan-fluence",
        "growth-negative-fluence",
        "creep-nan-fluence",
        "creep-negative-fluence",
        "creep-nan-flux",
        "creep-negative-flux",
        "creep-nan-stress",
    ],
)
def test_invalid_cladding_inputs_propagate_nan(
    evaluate: Callable[[], jax.Array],
) -> None:
    assert bool(jnp.isnan(evaluate()))


def test_zero_stress_creep_increment_is_zero_and_differentiable() -> None:
    law = ZircaloyCreep()
    direction = jnp.diag(
        jnp.asarray([-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0], dtype=jnp.float64)
    )
    initial = ZircaloyCreepState.initialize(direction)

    def increment(scale: jax.Array) -> jax.Array:
        return law.advance(
            scale * direction, 600.0, 5.0e17, 1.0e25, initial, 3600.0
        ).equivalent_strain_increment

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    value, tangent = jax.jvp(increment, (zero,), (jnp.ones_like(zero),))
    assert float(value) == 0.0
    assert bool(jnp.isfinite(tangent))


def test_primary_creep_reset_uses_significant_change_from_episode_reference() -> None:
    law = ZircaloyCreep()
    direction = jnp.diag(jnp.asarray([-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0]))
    start = law.advance(
        80.0e6 * direction,
        600.0,
        5.0e17,
        1.0e25,
        ZircaloyCreepState.initialize(direction),
        3600.0,
    )
    small = law.advance(84.0e6 * direction, 600.0, 5.0e17, 1.0e25, start.state, 3600.0)
    significant = law.advance(
        86.0e6 * direction, 600.0, 5.0e17, 1.0e25, small.state, 3600.0
    )
    np.testing.assert_allclose(float(small.state.creep_time), 7200.0)
    np.testing.assert_allclose(float(significant.state.creep_time), 3600.0)
    np.testing.assert_allclose(float(significant.state.primary_equivalent_stress), 6.0e6)


# PNNL-17700 Table 1 PWR condition (fluence 1.2e26 n/m^2, 50 % cold work) at 600 K.
_TEMPERATURE = 600.0
_FLUENCE = 1.2e26
_COLD_WORK = 0.5


def _power_law_parameters(law: ZircaloyPlasticity) -> tuple[float, float, float, float]:
    return (
        float(law.strength_coefficient(_TEMPERATURE, _FLUENCE, _COLD_WORK)),
        float(law.strain_hardening_exponent(_TEMPERATURE, _FLUENCE)),
        float(law.strain_rate_exponent(_TEMPERATURE)),
        float(law.elasticity.youngs_modulus(_TEMPERATURE, _FLUENCE, _COLD_WORK, 0.0)),
    )


def test_zero_strain_flow_stress_is_pnnl_yield_stress() -> None:
    law = ZircaloyPlasticity()
    strength, hardening, sensitivity, modulus = _power_law_parameters(law)
    rate = 1.0e-4
    # PNNL-17700 Eq. (3): intersection of Hooke's law and the power law.
    yield_stress = (strength / modulus**hardening * (rate / 1.0e-3) ** sensitivity) ** (
        1.0 / (1.0 - hardening)
    )
    predicted = law.flow_stress(0.0, rate, _TEMPERATURE, _FLUENCE, _COLD_WORK)
    np.testing.assert_allclose(float(predicted), yield_stress, rtol=1e-10)


def test_hardened_flow_stress_solves_frapcon_stress_strain_relation() -> None:
    law = ZircaloyPlasticity()
    strength, hardening, sensitivity, modulus = _power_law_parameters(law)
    strains = np.array([0.0, 1.0e-3, 1.0e-2, 5.0e-2])
    rate = 1.0e-4

    def flow(strain: jax.Array) -> jax.Array:
        return law.flow_stress(strain, rate, _TEMPERATURE, _FLUENCE, _COLD_WORK)

    stress = np.asarray(flow(jnp.asarray(strains)))
    # FRAPCON-4.0 Eq. (2.134): s = K (s / E + eps_p)^n (rate / 1e-3)^m.
    relation = (
        strength
        * (stress / modulus + strains) ** hardening
        * (rate / 1.0e-3) ** sensitivity
    )
    np.testing.assert_allclose(stress, relation, rtol=1e-11)
    assert np.all(np.diff(stress) > 0.0)
    tangent = jax.grad(flow)(jnp.asarray(1.0e-2))
    step = 1.0e-7
    difference = (
        float(flow(jnp.asarray(1.0e-2 + step))) - float(flow(jnp.asarray(1.0e-2 - step)))
    ) / (2.0 * step)
    np.testing.assert_allclose(float(tangent), difference, rtol=1e-6)


def test_yield_stress_rate_sensitivity_matches_pnnl_statement() -> None:
    flow = eqx.filter_jit(ZircaloyPlasticity().flow_stress)
    slow = float(flow(0.0, 1.0e-4, _TEMPERATURE, _FLUENCE, _COLD_WORK))
    fast = float(flow(0.0, 1.0, _TEMPERATURE, _FLUENCE, _COLD_WORK))
    # PNNL-17700 Section 2.4: 1e-4/s -> 1/s raises yield strength about 15 %.
    assert 1.10 < fast / slow < 1.20


@pytest.mark.parametrize("trial_stress", [100.0e6, 900.0e6], ids=["elastic", "plastic"])
def test_radial_return_preserves_elastic_branch_and_enforces_flow_law(
    trial_stress: float,
) -> None:
    law = ZircaloyPlasticity()
    direction = jnp.diag(jnp.asarray([-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0]))
    result = law.return_map(
        trial_stress * direction,
        0.002,
        1.0e-4,
        _TEMPERATURE,
        _FLUENCE,
        _COLD_WORK,
        10.0,
    )
    assert bool(result.converged)
    assert abs(float(result.residual)) <= 1e-13
    q = float(jnp.sqrt(1.5 * jnp.sum(result.deviatoric_stress**2)))
    ep = float(result.equivalent_plastic_strain)
    if trial_stress == 100.0e6:
        np.testing.assert_allclose(q, trial_stress)
        np.testing.assert_allclose(ep, 0.002)
        np.testing.assert_array_equal(
            np.asarray(result.plastic_strain_increment), np.zeros((3, 3))
        )
    else:
        # FRAPCON Eq. (2.134) plus Prandtl-Reuss elastic stress correction.
        strength, hardening, sensitivity, youngs = _power_law_parameters(law)
        reference = (
            strength * (q / youngs + ep) ** hardening * (1.0e-4 / 1.0e-3) ** sensitivity
        )
        np.testing.assert_allclose(q, reference, rtol=1e-11)
        shear = float(
            law.elasticity.shear_modulus(_TEMPERATURE, _FLUENCE, _COLD_WORK, 0.0)
        )
        np.testing.assert_allclose(
            trial_stress - q, 3.0 * shear * (ep - 0.002), rtol=1e-11
        )
        assert ep > 0.002
        np.testing.assert_allclose(
            float(jnp.trace(result.plastic_strain_increment)), 0.0, atol=1e-16
        )


def test_moduli_record_and_explicit_return_map_match_elasticity_bitwise() -> None:
    law = ZircaloyPlasticity()
    elasticity = law.elasticity
    moduli = elasticity.moduli(_TEMPERATURE, _FLUENCE, _COLD_WORK, 0.0)
    youngs = elasticity.youngs_modulus(_TEMPERATURE, _FLUENCE, _COLD_WORK, 0.0)
    shear = elasticity.shear_modulus(_TEMPERATURE, _FLUENCE, _COLD_WORK, 0.0)
    poisson = elasticity.poisson_ratio(_TEMPERATURE, _FLUENCE, _COLD_WORK, 0.0)
    np.testing.assert_array_equal(moduli.youngs, youngs)
    np.testing.assert_array_equal(moduli.shear, shear)
    np.testing.assert_array_equal(moduli.bulk, youngs / (3.0 * (1.0 - 2.0 * poisson)))
    direction = jnp.diag(jnp.asarray([-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0]))
    arguments = (900.0e6 * direction, 0.002, 1.0e-4, _TEMPERATURE, _FLUENCE, _COLD_WORK)
    default = law.return_map(*arguments, 10.0)
    explicit = law.return_map(*arguments, 10.0, moduli=moduli)
    for actual, expected in zip(
        jax.tree.leaves(explicit), jax.tree.leaves(default), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_axial_growth_matches_frapcon_franklin_model() -> None:
    fluence = np.array([1.0e24, 3.0e25, 1.0e26])
    strains = ZircaloyIrradiationGrowth().principal_strains(fluence)
    # PNNL-19417 Eq. (3.44) with fluence in n/cm^2; ~0.85 % at 1e22 n/cm^2
    # (Figure 3.8).
    np.testing.assert_allclose(
        np.asarray(strains.axial), 2.18e-21 * (fluence / 1.0e4) ** 0.845, rtol=1e-12
    )
    assert 0.0084 < float(strains.axial[-1]) < 0.0086


def test_growth_components_follow_matpro_texture_rule_and_conserve_volume() -> None:
    texture = ZircaloyTexture(radial=0.66, hoop=0.28, axial=0.06)
    strains = ZircaloyIrradiationGrowth(texture=texture).principal_strains(6.0e25)
    radial, hoop, axial = float(strains.radial), float(strains.hoop), float(strains.axial)
    # MATPRO Eq. (4-104): growth along i proportional to 1 - 3 f_i.
    np.testing.assert_allclose(hoop / axial, (1.0 - 3.0 * 0.28) / (1.0 - 3.0 * 0.06))
    np.testing.assert_allclose(radial / axial, (1.0 - 3.0 * 0.66) / (1.0 - 3.0 * 0.06))
    assert abs(radial + hoop + axial) < 1e-15


def _integrate(
    rate: Callable[[jax.Array], jax.Array], start: float, days: float, steps: int
) -> float:
    step = days * _SECONDS_PER_DAY / steps

    def body(_: int, thickness: jax.Array) -> jax.Array:
        k1 = rate(thickness)
        k2 = rate(thickness + 0.5 * step * k1)
        k3 = rate(thickness + 0.5 * step * k2)
        k4 = rate(thickness + step * k3)
        return thickness + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return float(
        jax.jit(lambda s0: jax.lax.fori_loop(0, steps, body, s0))(jnp.asarray(start))
    )


def test_pre_transition_oxide_follows_frapcon_cubic_integral() -> None:
    law = ZircaloyWatersideCorrosion()
    interface, days, start_um = 600.0, 100.0, 0.5
    final = _integrate(
        lambda s: law.oxide_thickness_rate(s, interface, 5.0e17),
        start_um * 1.0e-6,
        days,
        4000,
    )
    # FRAPCON-4.0 Eq. (2.191): s^3 = 3 A exp(-Q1/(R T1)) dt + s_i^3 [um, days].
    cubic = (
        3.0 * 6.3e9 * math.exp(-32289.0 / (1.98 * interface)) * days + start_um**3
    ) ** (1.0 / 3.0)
    assert cubic < 2.0
    np.testing.assert_allclose(final * 1.0e6, cubic, rtol=1e-8)


def test_post_transition_oxide_with_heat_flux_matches_garzarolli_integral() -> None:
    law = ZircaloyWatersideCorrosion()
    water_side, heat_flux, conductivity = 590.0, 60.0, 0.02  # K, W/cm^2, W/(cm K)
    flux_cm, days, start = 1.0e14, 200.0, 10.0e-6
    gamma, q2, gas = 0.6789, 27354.0, 1.98

    def rate(thickness: jax.Array) -> jax.Array:
        interface = water_side + heat_flux * (thickness * 100.0) / conductivity
        return law.oxide_thickness_rate(thickness, interface, flux_cm * 1.0e4)

    final = _integrate(rate, start, days, 4000)
    # FRAPCON-4.0 Eq. (2.193), weight gain w [g/cm^2], s = w gamma / 100 [m].
    k0 = 11863.0 + 3.5e4 * (1.91e-15 * flux_cm) ** 0.24
    a = gamma * q2 * heat_flux / (gas * water_side**2 * conductivity)
    w0 = start * 100.0 / gamma
    w1 = (
        w0
        - math.log(
            1.0 - a * k0 * math.exp(-q2 / (gas * water_side)) * math.exp(a * w0) * days
        )
        / a
    )
    # Eq. (2.193) linearizes 1/T1 about the water-side temperature.
    np.testing.assert_allclose(final - start, w1 * gamma / 100.0 - start, rtol=5e-3)


def test_oxide_advance_handles_bare_metal_and_transition_crossing() -> None:
    law = ZircaloyWatersideCorrosion()
    temperature, flux = 620.0, 5.0e17
    # FRAPCON Eqs. (2.191)-(2.194), independent native-unit integration.
    cubic_um_per_day = 3.0 * 6.3e9 * math.exp(-32289.0 / (1.98 * temperature))
    transition_days = 2.0**3 / cubic_um_per_day
    pre_days = 0.5 * transition_days
    pre = law.advance(0.0, temperature, flux, pre_days * _SECONDS_PER_DAY)
    np.testing.assert_allclose(
        float(pre) * 1.0e6, (cubic_um_per_day * pre_days) ** (1.0 / 3.0)
    )
    post_days = 20.0
    post_um_per_day = (
        (11863.0 + 3.5e4 * (1.91e-15 * flux / 1.0e4) ** 0.24)
        * math.exp(-27354.0 / (1.98 * temperature))
        * 0.6789
        * 1.0e4
    )
    final = law.advance(
        0.0, temperature, flux, (transition_days + post_days) * _SECONDS_PER_DAY
    )
    np.testing.assert_allclose(
        float(final) * 1.0e6, 2.0 + post_um_per_day * post_days, rtol=1e-12
    )
    partitioned = law.advance(
        pre,
        temperature,
        flux,
        (transition_days - pre_days + post_days) * _SECONDS_PER_DAY,
    )
    np.testing.assert_allclose(float(partitioned), float(final), rtol=1e-12)


def test_oxide_conductivity_matches_pnnl_zotcon_reference() -> None:
    # PNNL-19417 Eq. (3.7), ZOTCON listing A.2.3, evaluated at 600 K.
    np.testing.assert_allclose(
        float(ZirconiaThermalConductivity().thermal_conductivity(600.0)),
        2.0047464,
        rtol=1e-12,
    )


def test_oxide_mass_balance_and_pilling_bedworth_ratio() -> None:
    law = ZircaloyWatersideCorrosion()
    thickness = 1.0e-5
    oxygen = float(law.oxygen_areal_mass(thickness))
    metal = float(law.metal_thickness_consumed(thickness))
    oxide_density = (oxygen + metal * ZIRCALOY_DENSITY) / thickness
    # MATPRO Section 5.5.2: black oxide density 5800 kg/m^3; Zr PBR 1.56.
    np.testing.assert_allclose(oxide_density, 5800.0, rtol=0.03)
    np.testing.assert_allclose(thickness / metal, 1.56, atol=0.005)


def test_hydrogen_increment_matches_matpro_chuptk() -> None:
    corrosion = ZircaloyWatersideCorrosion()
    pickup = ZircaloyHydrogenPickup()
    outer, inner, growth = 9.5e-3, 8.36e-3, 10.0e-6
    metal_areal_mass = ZIRCALOY_DENSITY * (outer**2 - inner**2) / (4.0 * outer)
    ppm = float(
        pickup.hydrogen_content_increment(
            corrosion.oxygen_areal_mass(growth), metal_areal_mass
        )
    )
    # MATPRO Eq. (A2-14): H = 9.0e5 d_o / (d_o^2 - d_i^2) (b / 8) dx, b = 0.15.
    matpro = 9.0e5 * outer / (outer**2 - inner**2) * (0.15 / 8.0) * growth
    np.testing.assert_allclose(ppm, matpro, rtol=1.5e-2)
    absorbed = float(
        pickup.absorbed_hydrogen_areal_mass(corrosion.oxygen_areal_mass(growth))
    )
    np.testing.assert_allclose(ppm, 1.0e6 * absorbed / metal_areal_mass, rtol=1e-12)


def test_emissivity_follows_zoemis() -> None:
    law = ZircaloyEmissivity()
    below, above = law.emissivity(jnp.asarray([3.88e-6 - 1.0e-12, 3.88e-6]), 600.0)
    # MATPRO Eq. (5-26) footnote: branches match at 3.88e-6 m.
    np.testing.assert_allclose(float(below), float(above), atol=1e-6)
    # MATPRO Table 5-6 (Murphy and Havelock, 0.9 um film, 100-400 C:
    # 0.414-0.434) within the standard error 0.1 of Eq. (5-28).
    thin = np.asarray(law.emissivity(0.9e-6, jnp.asarray([373.15, 673.15])))
    assert np.all(np.abs(thin - 0.424) < 0.1)
    hot = np.asarray(law.emissivity(1.0e-4, jnp.asarray([1600.0, 1800.0])))
    thick = 0.808642 - 50.0 * 1.0e-4
    np.testing.assert_allclose(hot, [thick * math.exp(-100.0 / 300.0), 0.325])


def test_emissivity_measurement_temperature_is_distinct_from_historical_maximum() -> None:
    law = ZircaloyEmissivity()
    # PNNL-19417 §3.4.1: historical Tmax only changes the degradation branch.
    # Measurement applicability concerns instantaneous surface temperature.
    assert bool(law.validity.contains(oxide_thickness=0.0, surface_temperature=600.0))
    assert not bool(law.validity.contains(oxide_thickness=0.0, surface_temperature=300.0))


@pytest.mark.parametrize(
    ("law_type", "inside", "outside"),
    [
        pytest.param(
            ZircaloyCreep,
            {"temperature": 600.0, "equivalent_stress": 80.0e6, "fast_flux": 5.0e17},
            {"temperature": 650.0, "equivalent_stress": 80.0e6, "fast_flux": 5.0e17},
            id="creep-temperature",
        ),
        pytest.param(
            ZircaloyWatersideCorrosion,
            {"interface_temperature": 620.0},
            {"interface_temperature": 700.0},
            id="corrosion-interface-temperature",
        ),
        pytest.param(
            ZircaloyIrradiationGrowth,
            {"fast_fluence": 1.0e26},
            {"fast_fluence": 1.3e26},
            id="growth-fluence",
        ),
        pytest.param(
            ZircaloyPlasticity,
            {
                "temperature": 600.0,
                "fast_fluence": 1.0e26,
                "cold_work": 0.5,
                "equivalent_plastic_strain_rate": 1.0e-4,
            },
            {
                "temperature": 600.0,
                "fast_fluence": 1.0e26,
                "cold_work": 0.5,
                "equivalent_plastic_strain_rate": 1.0e-7,
            },
            id="plasticity-strain-rate",
        ),
    ],
)
def test_validity_ranges_bound_published_data(
    law_type: type[
        ZircaloyCreep
        | ZircaloyWatersideCorrosion
        | ZircaloyIrradiationGrowth
        | ZircaloyPlasticity
    ],
    inside: dict[str, float],
    outside: dict[str, float],
) -> None:
    law = law_type()
    assert bool(law.validity.contains(**inside))
    assert not bool(law.validity.contains(**outside))
