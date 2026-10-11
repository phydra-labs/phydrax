"""UO2 pellet property, eigenstrain, and creep laws against published values.

References (full citations in the law modules):
IAEA-TECDOC-1496 (2006) Section 6.1.1.1 Tables 1 and 3 (Fink 2000 UO2 enthalpy
and heat capacity) and the thermal expansion Table 1 (Martin 1988);
PNNL-19417 Rev. 2 (FRAPCON-4.0 material properties); PNNL-19418 Vol. 1 Rev. 2
(FRAPCON-4.0 models, Fig. 2.16); MatLib-1.2.1 PNNL-35702 Eq. (2-17);
NUREG/CR-6150 Vol. 4 Section 2.6 Table 2-12 (FELMOD); NUREG/CR-0497 Section
A-6 (FCREEP).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.applications.nuclear_fuel._fuel_creep import FuelCreepState, UO2Creep
from phydrax.applications.nuclear_fuel._fuel_eigenstrain import (
    FuelRelocation,
    UO2Densification,
    UO2SolidSwelling,
)
from phydrax.applications.nuclear_fuel._fuel_properties import (
    UO2Elasticity,
    UO2Emissivity,
    UO2HeatCapacity,
    UO2ThermalConductivity,
    UO2ThermalExpansion,
)
from phydrax.materials import PropertyValidityRange


pytestmark = pytest.mark.strict_jax


# IAEA-TECDOC-1496 Table 3 (per kg of UO2, from Fink Eqs. (1)-(2)):
# (T [K], H - H(298.15) [kJ/kg], half unit of last digit, cp [J/(kg K)]).
@pytest.mark.parametrize(
    ("temperature", "enthalpy_kj", "enthalpy_tolerance", "heat_capacity"),
    [
        pytest.param(300.0, 0.43, 0.005, 235.0, id="300K"),
        pytest.param(400.0, 25.7, 0.05, 266.0, id="400K"),
        pytest.param(1000.0, 203.0, 0.5, 312.0, id="1000K"),
        pytest.param(2000.0, 535.0, 0.5, 373.0, id="2000K"),
        pytest.param(2500.0, 749.0, 0.5, 500.0, id="2500K"),
        pytest.param(3000.0, 1052.0, 0.5, 726.0, id="3000K"),
        pytest.param(3120.0, 1143.0, 0.5, 792.0, id="3120K"),
    ],
)
def test_heat_capacity_matches_fink_per_kilogram_table(
    temperature: float,
    enthalpy_kj: float,
    enthalpy_tolerance: float,
    heat_capacity: float,
) -> None:
    law = UO2HeatCapacity()
    t = jnp.asarray(temperature)
    np.testing.assert_allclose(
        law.enthalpy(t) / 1.0e3, enthalpy_kj, atol=enthalpy_tolerance + 1.0e-9
    )
    np.testing.assert_allclose(law.heat_capacity(t), heat_capacity, atol=0.5 + 1.0e-9)


# IAEA-TECDOC-1496 Table 1 (per mole of UO2): (T [K], kJ/mol, J/(mol K)).
@pytest.mark.parametrize(
    ("temperature", "enthalpy_kj_per_mol", "heat_capacity_per_mol"),
    [
        pytest.param(600.0, 22.1, 78.9, id="600K"),
        pytest.param(1500.0, 98.0, 88.4, id="1500K"),
        pytest.param(2700.0, 231.4, 156.4, id="2700K"),
    ],
)
def test_heat_capacity_matches_fink_per_mole_table(
    temperature: float, enthalpy_kj_per_mol: float, heat_capacity_per_mol: float
) -> None:
    law = UO2HeatCapacity()
    t = jnp.asarray(temperature)
    np.testing.assert_allclose(
        law.enthalpy(t) * law.molar_mass / 1.0e3, enthalpy_kj_per_mol, atol=0.05 + 1.0e-9
    )
    np.testing.assert_allclose(
        law.heat_capacity(t) * law.molar_mass, heat_capacity_per_mol, atol=0.05 + 1.0e-9
    )


def test_heat_capacity_is_exact_temperature_derivative_of_enthalpy() -> None:
    law = UO2HeatCapacity()
    temperature = jnp.linspace(298.15, 3120.0, 97)
    derivative = jax.vmap(jax.grad(law.enthalpy))(temperature)
    np.testing.assert_allclose(law.heat_capacity(temperature), derivative, rtol=1.0e-12)
    np.testing.assert_allclose(law.enthalpy(jnp.asarray(298.15)), 0.0, atol=1.0e-12)


def test_calibrated_enthalpy_keeps_reference_and_derivative_contract() -> None:
    # The reference enthalpy must remain a gauge choice even when calibration
    # lowers the defect activation temperature enough that its cold term matters.
    law = UO2HeatCapacity(defect_temperature=3000.0)
    reference = jnp.asarray(298.15)
    np.testing.assert_array_equal(law.enthalpy(reference), 0.0)
    np.testing.assert_allclose(
        jax.grad(law.enthalpy)(reference), law.heat_capacity(reference), rtol=1.0e-12
    )


# PNNL-19417 Rev. 2 Eq. (2.19) evaluated by hand at 95 %TD (porosity factor
# 1.0789 * 0.95 / 1.025 = 0.99995610): K95 = 1/(0.0452 + 1.1599 gad + 2.46e-4 T
# + 0.00187 Bu + (1 - 0.9 e^(-0.04 Bu)) 0.038 Bu^0.28 / (1 + 396 e^(-6380/T)))
# + 3.5e9/T^2 e^(-16361/T), Bu in GWd/MTU (= MWd/kgHM for UO2).
@pytest.mark.parametrize(
    ("temperature", "burnup", "gadolinia", "k95"),
    [
        pytest.param(1000.0, 0.0, 0.0, 3.4343404553527224, id="1000K-fresh"),
        pytest.param(1000.0, 30.0, 0.0, 2.562701316188946, id="1000K-30GWd"),
        pytest.param(600.0, 45.0, 0.0, 2.702876087067777, id="600K-45GWd"),
        pytest.param(2000.0, 20.0, 0.05, 1.818318493899907, id="2000K-20GWd-5wtGd"),
    ],
)
def test_conductivity_matches_modified_nfi_equation(
    temperature: float, burnup: float, gadolinia: float, k95: float
) -> None:
    law = UO2ThermalConductivity()
    value = law.conductivity(
        jnp.asarray(temperature),
        jnp.asarray(burnup),
        jnp.asarray(0.05),
        jnp.asarray(gadolinia),
    )
    np.testing.assert_allclose(value, k95 * 0.9999560975609756, rtol=1.0e-12)
    # Eq. (2.20): the Lucuta factor is referenced to 95 %TD.
    np.testing.assert_allclose(value, k95, rtol=5.0e-5)


def test_conductivity_degrades_with_burnup_porosity_and_gadolinia() -> None:
    law = UO2ThermalConductivity()
    temperature = jnp.asarray([400.0, 800.0, 1200.0, 1600.0, 2000.0, 2600.0])[:, None]
    burnup = jnp.linspace(0.0, 62.0, 32)[None, :]
    by_burnup = law.conductivity(temperature, burnup, jnp.asarray(0.05), jnp.asarray(0.0))
    assert bool(jnp.all(jnp.diff(by_burnup, axis=1) < 0.0))
    by_porosity = law.conductivity(
        jnp.asarray(1000.0),
        jnp.asarray(10.0),
        jnp.linspace(0.03, 0.08, 6),
        jnp.asarray(0.0),
    )
    assert bool(jnp.all(jnp.diff(by_porosity) < 0.0))
    by_gadolinia = law.conductivity(
        jnp.asarray(1000.0),
        jnp.asarray(10.0),
        jnp.asarray(0.05),
        jnp.linspace(0.0, 0.1, 6),
    )
    assert bool(jnp.all(jnp.diff(by_gadolinia) < 0.0))


def test_conductivity_burnup_power_guards_only_exact_zero() -> None:
    # Bu^0.28 is evaluated zero-safe: zero burnup has a finite value and
    # derivative, while NaN and negative burnup propagate as NaN.
    law = UO2ThermalConductivity()

    def conductivity(burnup: jax.Array) -> jax.Array:
        return law.conductivity(1000.0, burnup, 0.05, 0.0)

    for invalid in (math.nan, -1.0):
        assert bool(jnp.isnan(conductivity(jnp.asarray(invalid, dtype=jnp.float64))))
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    value, tangent = jax.jvp(conductivity, (zero,), (jnp.ones_like(zero),))
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(tangent))


# Martin (1988) recommended linear expansion, IAEA-TECDOC-1496 Table 1
# (dL/L273 in %), re-referenced to 300 K: (m_T - 0.027) / (100 + 0.027). The
# FRAPCON-4.0 constants were refit to these data (PNNL-19417 Rev. 2 Section
# 2.5.1) with a +/-10 % one-sigma band (Section 2.5.2).
@pytest.mark.parametrize(
    ("temperature", "martin_percent"),
    [
        pytest.param(1000.0, 0.730, id="1000K"),
        pytest.param(1500.0, 1.305, id="1500K"),
        pytest.param(2000.0, 2.034, id="2000K"),
        pytest.param(2500.0, 3.006, id="2500K"),
        pytest.param(3000.0, 4.314, id="3000K"),
    ],
)
def test_thermal_expansion_agrees_with_martin_within_source_uncertainty(
    temperature: float, martin_percent: float
) -> None:
    law = UO2ThermalExpansion()
    reference = (martin_percent - 0.027) / (100.0 + 0.027)
    np.testing.assert_allclose(
        law.linear_strain(jnp.asarray(temperature)), reference, rtol=0.10
    )


def test_thermal_expansion_vanishes_at_declared_reference_and_increases() -> None:
    law = UO2ThermalExpansion(reference_temperature=293.15)
    np.testing.assert_allclose(law.linear_strain(jnp.asarray(293.15)), 0.0, atol=1.0e-15)
    strain = law.linear_strain(jnp.linspace(300.0, 3111.15, 64))
    assert bool(jnp.all(jnp.diff(strain) > 0.0))


# NUREG/CR-6150 Vol. 4 Table 2-12: per-sample least-squares intercepts a
# [1e10 N/m^2] of E = a (1 + b T); FELMOD's intercept is its value at T = 0.
# Standard error 0.06e11 N/m^2 (Eq. (2-50)).
@pytest.mark.parametrize(
    ("density", "intercept"),
    [
        pytest.param(0.911, 17.605, id="padel-0.911"),
        pytest.param(0.935, 19.221, id="padel-0.935"),
        pytest.param(0.959, 20.549, id="padel-0.959"),
        pytest.param(0.93, 18.742, id="belle-lustman-0.93"),
        pytest.param(0.947, 20.175, id="hall-0.947"),
    ],
)
def test_youngs_modulus_matches_felmod_data_fits(
    density: float, intercept: float
) -> None:
    law = UO2Elasticity()
    value = law.youngs_modulus(jnp.asarray(0.0), jnp.asarray(1.0 - density))
    np.testing.assert_allclose(value, intercept * 1.0e10, atol=0.06e11)


def test_elastic_moduli_reproduce_wachtman_shear_ratio_and_isotropy() -> None:
    law = UO2Elasticity()
    temperature = jnp.asarray([300.0, 900.0, 1500.0])
    porosity = jnp.asarray(0.05)
    young = law.youngs_modulus(temperature, porosity)
    shear = law.shear_modulus(temperature, porosity)
    bulk = law.bulk_modulus(temperature, porosity)
    # FPOIR: nu = 0.316 from Wachtman E = 2.30e11 Pa and G = 0.874e11 Pa.
    np.testing.assert_allclose(shear / young, 0.874 / 2.30, rtol=1.0e-3)
    np.testing.assert_allclose(
        9.0 * bulk * shear / (3.0 * bulk + shear), young, rtol=1.0e-12
    )
    np.testing.assert_allclose(law.poisson_ratio(temperature, porosity), 0.316, rtol=0.0)
    assert bool(jnp.all(jnp.diff(young) < 0.0))


def test_elastic_moduli_record_matches_individual_moduli_bitwise() -> None:
    law = UO2Elasticity()
    temperature = jnp.asarray([300.0, 900.0, 1500.0])
    porosity = jnp.asarray(0.05)
    moduli = law.moduli(temperature, porosity)
    np.testing.assert_array_equal(moduli.shear, law.shear_modulus(temperature, porosity))
    np.testing.assert_array_equal(moduli.bulk, law.bulk_modulus(temperature, porosity))


# PNNL-19417 Rev. 2 Eq. (2.23): e = 0.78557 + 1.5263e-5 T.
def test_emissivity_matches_femiss() -> None:
    law = UO2Emissivity()
    np.testing.assert_allclose(
        law.emissivity(jnp.asarray([1000.0, 2400.0])), [0.800833, 0.8222012], rtol=1.0e-12
    )


def test_densification_starts_at_zero_and_is_bounded_by_resintering_limit() -> None:
    law = UO2Densification()
    density = jnp.asarray(10400.0)
    change = jnp.asarray([10.0, 50.0, 150.0])[:, None]
    burnup = jnp.concatenate([jnp.asarray([0.0, 0.019]), jnp.linspace(0.02, 20.0, 60)])[
        None, :
    ]
    strain = law.linear_strain(burnup, change, density)
    limit = law.maximum_linear_strain(change, density)
    # FUDENS returns no densification below 1728 MW s/kgU = 0.02 MWd/kgU.
    np.testing.assert_array_equal(strain[:, :2], 0.0)
    assert bool(jnp.all(jnp.diff(strain, axis=1) <= 0.0))
    assert bool(jnp.all(strain >= limit))
    # FRAPCON-4.0 Section 2.4: the ultimate change is approached by ~5 GWd/MTU.
    at_five = law.linear_strain(jnp.asarray(5.0), change[:, 0], density)
    assert bool(jnp.all(at_five / limit[:, 0] > 0.99))
    np.testing.assert_allclose(
        law.maximum_linear_strain(change[:, 0], density), -change[:, 0] / (3.0 * 10400.0)
    )
    np.testing.assert_array_equal(
        law.linear_strain(burnup, jnp.asarray(0.0), density), jnp.zeros_like(burnup)
    )


def test_densification_offset_satisfies_zero_burnup_boundary_condition() -> None:
    # PNNL-19417 Rev. 2 Eq. (2.32): B is determined so that dL/L = 0 at FBU = 0.
    law = UO2Densification(onset_burnup=0.0)
    change = jnp.asarray([1.0, 50.0, 400.0, 900.0])
    strain = law.linear_strain(jnp.asarray(0.0), change, jnp.asarray(10400.0))
    np.testing.assert_allclose(strain, 0.0, atol=1.0e-14)


def test_densification_update_never_reverses_committed_shrinkage() -> None:
    # PNNL-19417 Rev. 2 Appendix A.1.6, FUDENS final prvden comparison:
    # a smaller predicted shrinkage cannot undo an accepted shrinkage.
    law = UO2Densification()
    previous = jnp.asarray(-0.004)
    candidate = law.update(
        jnp.asarray(0.1), jnp.asarray(150.0), jnp.asarray(10400.0), previous
    )
    np.testing.assert_array_equal(candidate.linear_strain, previous)
    np.testing.assert_array_equal(candidate.linear_strain_increment, 0.0)
    assert bool(candidate.root_converged)
    np.testing.assert_allclose(candidate.root_residual, 0.0, atol=1.0e-13)


def test_densification_update_reports_additional_shrinkage_and_root_evidence() -> None:
    law = UO2Densification()
    previous = jnp.asarray([0.0, -0.001])
    candidate = law.update(
        jnp.asarray([5.0, 5.0]),
        jnp.asarray([0.0, 150.0]),
        jnp.asarray(10400.0),
        previous,
    )
    np.testing.assert_allclose(
        candidate.linear_strain, [0.0, -150.0 / (3.0 * 10400.0)], rtol=1.0e-6
    )
    np.testing.assert_allclose(
        candidate.linear_strain_increment,
        candidate.linear_strain - previous,
        atol=1.0e-15,
    )
    assert bool(jnp.all(candidate.root_converged))
    np.testing.assert_allclose(candidate.root_residual, 0.0, atol=1.0e-13)


def test_densification_root_failure_below_onset_does_not_fail_update() -> None:
    # A negative resintering change makes the burnup-offset root NaN; below the
    # FUDENS onset the strain is zero, so only the active site reports failure.
    law = UO2Densification()
    previous = jnp.zeros(2)
    candidate = law.update(
        jnp.asarray([0.0, 5.0]), jnp.asarray(-150.0), jnp.asarray(10400.0), previous
    )
    np.testing.assert_array_equal(candidate.root_converged, [True, False])
    np.testing.assert_array_equal(candidate.linear_strain[0], 0.0)
    np.testing.assert_array_equal(candidate.linear_strain_increment[0], 0.0)


# MatLib-1.2.1 Eq. (2-17); FRAPCON-4.0 Section 2.4: none to 6 GWd/MTU, then
# 0.062 %/GWd/MTU to 80 GWd/MTU and 0.086 %/GWd/MTU beyond.
@pytest.mark.parametrize(
    ("burnup", "strain"),
    [
        pytest.param(0.0, 0.0, id="fresh"),
        pytest.param(6.0, 0.0, id="onset"),
        pytest.param(50.0, 0.00062 * 44.0, id="50GWd"),
        pytest.param(80.0, 0.00062 * 74.0, id="transition"),
        pytest.param(90.0, 0.00062 * 74.0 + 0.00086 * 10.0, id="90GWd"),
    ],
)
def test_solid_swelling_follows_frapcon_rates(burnup: float, strain: float) -> None:
    law = UO2SolidSwelling()
    np.testing.assert_allclose(
        law.volumetric_strain(jnp.asarray(burnup)), strain, rtol=1.0e-12, atol=1.0e-15
    )


# FRAPCON-4.0 Fig. 2.16: plateaus 0.40, 0.45, 0.50 of the as-fabricated gap at
# 15, 30, 45 kW/m (beyond ~6 GWd/MTU) and 0.055 at the start of life.
@pytest.mark.parametrize(
    ("linear_heat_rate", "plateau"),
    [
        pytest.param(15.0e3, 0.40, id="15kW/m"),
        pytest.param(30.0e3, 0.45, id="30kW/m"),
        pytest.param(45.0e3, 0.50, id="45kW/m"),
    ],
)
def test_relocation_reproduces_frapcon_figure(
    linear_heat_rate: float, plateau: float
) -> None:
    law = FuelRelocation()
    power = jnp.asarray(linear_heat_rate)
    np.testing.assert_allclose(
        law.gap_closure_fraction(power, jnp.asarray([6.0, 8.0])), plateau, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        law.gap_closure_fraction(power, jnp.asarray([0.0, 0.09])), 0.055, rtol=1.0e-12
    )
    closure = law.gap_closure_fraction(power, jnp.linspace(0.1, 8.0, 40))
    assert bool(jnp.all(jnp.diff(closure) >= 0.0))


def test_relocation_strain_scales_gap_and_keeps_permanent_fraction() -> None:
    law = FuelRelocation()
    power = jnp.asarray(30.0e3)
    burnup = jnp.asarray(10.0)
    gap = jnp.asarray(170.0e-6)
    diameter = jnp.asarray(8.19e-3)
    full = law.diametral_strain(power, burnup, gap, diameter)
    np.testing.assert_allclose(full, 0.45 * 170.0e-6 / 8.19e-3, rtol=1.0e-12)
    # FRAPCON-4.0: 50 % of the relocation is permanent for the rigid pellet.
    np.testing.assert_allclose(
        law.mechanical_diametral_strain(power, burnup, gap, diameter),
        0.5 * full,
        rtol=1.0e-12,
    )


def test_creep_stress_exponent_switches_at_transition_stress() -> None:
    law = UO2Creep()
    grain = jnp.asarray(10.0e-6)
    # NUREG/CR-0497 Eq. (A-6.1): sigma_t = 1.6547e7 / G^0.5714, G in um.
    transition = law.transition_stress(grain)
    np.testing.assert_allclose(transition, 1.6547e7 / 10.0**0.5714, rtol=1.0e-12)

    def rate(stress: Array) -> Array:
        return law.equivalent_creep_rate(
            stress,
            jnp.asarray(1900.0),
            jnp.asarray(0.0),
            jnp.asarray(0.95),
            grain,
            jnp.asarray(2.0),
        )

    low = jnp.asarray([0.1, 0.2, 0.4]) * transition
    np.testing.assert_allclose(rate(low) / low, rate(low[0]) / low[0], rtol=1.0e-12)
    high = jnp.asarray([20.0e6, 40.0e6])
    capped_linear = rate(transition)
    power_law = rate(high) - capped_linear
    np.testing.assert_allclose(power_law[1] / power_law[0], 2.0**4.5, rtol=1.0e-10)


def test_creep_fission_enhancement_is_fivefold_at_perrin_fission_rate() -> None:
    # NUREG/CR-0497 Section A-6.2.2: a fivefold increase of the diffusional
    # creep rate at 1.2e19 fissions/(m^3 s); at 1900 K the athermal fission
    # induced term is below 0.5 % of it.
    law = UO2Creep()
    arguments = (
        jnp.asarray(2.0e6),
        jnp.asarray(1900.0),
        jnp.asarray(0.95),
        jnp.asarray(10.0e-6),
        jnp.asarray(2.0),
    )
    stress, temperature, density, grain, ratio = arguments
    unirradiated = law.equivalent_creep_rate(
        stress, temperature, jnp.asarray(0.0), density, grain, ratio
    )
    irradiated = law.equivalent_creep_rate(
        stress, temperature, jnp.asarray(1.2e19), density, grain, ratio
    )
    np.testing.assert_allclose(irradiated / unirradiated, 5.0, rtol=5.0e-3)


def test_creep_increases_with_oxygen_to_metal_ratio() -> None:
    # NUREG/CR-0497 Section A-6.1: increasing O/M increases the creep rate;
    # the FCREEP listing treats hypostoichiometric fuel as O/M = 2.
    law = UO2Creep()
    ratio = jnp.asarray([1.98, 2.0, 2.001, 2.01, 2.05, 2.1])
    rate = law.equivalent_creep_rate(
        jnp.asarray(2.0e6),
        jnp.asarray(1600.0),
        jnp.asarray(0.0),
        jnp.asarray(0.95),
        jnp.asarray(10.0e-6),
        ratio,
    )
    np.testing.assert_allclose(rate[0], rate[1], rtol=0.0)
    assert bool(jnp.all(jnp.diff(rate[1:]) > 0.0))


@pytest.mark.parametrize(
    ("previous_stress", "candidate_stress", "previous_age", "candidate_age"),
    [
        pytest.param(1.0e6, 2.0e6, 5.0e5, 3600.0, id="increase-resets-age"),
        pytest.param(2.0e6, 2.0e6, 5.0e5, 503600.0, id="hold-advances-age"),
        pytest.param(3.0e6, 2.0e6, 5.0e5, 503600.0, id="unload-advances-age"),
    ],
)
def test_primary_creep_candidate_follows_matpro_listing_history(
    previous_stress: float,
    candidate_stress: float,
    previous_age: float,
    candidate_age: float,
) -> None:
    # MATPRO-11 Table A-6.1: XTIME=TLAST on FSTRES>PSTRES, then
    # FTIME=TIME-XTIME, UPRURY=1+2.5 EXP(-1.2e-6 FTIME).
    law = UO2Creep()
    accepted = FuelCreepState(jnp.asarray(previous_stress), jnp.asarray(previous_age))
    arguments = (
        jnp.asarray(candidate_stress),
        jnp.asarray(1600.0),
        jnp.asarray(1.0e19),
        jnp.asarray(0.96),
        jnp.asarray(10.0e-6),
        jnp.asarray(2.0),
    )
    result = law.update(*arguments, accepted, jnp.asarray(3600.0))
    steady = law.equivalent_creep_rate(*arguments)
    np.testing.assert_allclose(
        result.equivalent_creep_rate / steady,
        1.0 + 2.5 * math.exp(-1.2e-6 * candidate_age),
        rtol=1.0e-12,
    )
    np.testing.assert_array_equal(result.state.time_since_stress_increase, candidate_age)
    np.testing.assert_array_equal(
        result.state.previous_equivalent_stress, candidate_stress
    )
    np.testing.assert_allclose(
        result.equivalent_creep_strain_increment,
        3600.0 * result.equivalent_creep_rate,
        rtol=1.0e-12,
    )
    np.testing.assert_array_equal(accepted.time_since_stress_increase, previous_age)


def test_creep_validity_excludes_singular_stoichiometry_endpoint() -> None:
    law = UO2Creep()
    assert not bool(
        law.validity.contains(
            density_fraction=jnp.asarray(0.96),
            oxygen_to_metal_ratio=jnp.asarray(2.99999),
        )
    )


@pytest.mark.parametrize(
    ("validity", "inside", "outside"),
    [
        pytest.param(
            UO2HeatCapacity().validity,
            {"temperature": 298.15},
            {"temperature": 3200.0},
            id="heat-capacity",
        ),
        pytest.param(
            UO2ThermalConductivity().validity,
            {
                "temperature": 1000.0,
                "burnup_mwd_per_kghm": 62.0,
                "as_fabricated_porosity": 0.05,
                "gadolinia_weight_fraction": 0.0,
            },
            {
                "temperature": 1000.0,
                "burnup_mwd_per_kghm": 70.0,
                "as_fabricated_porosity": 0.05,
                "gadolinia_weight_fraction": 0.0,
            },
            id="conductivity",
        ),
        pytest.param(
            UO2ThermalExpansion().validity,
            {"temperature": 3111.15},
            {"temperature": 3150.0},
            id="thermal-expansion",
        ),
        pytest.param(
            UO2Elasticity().validity,
            {"temperature": 1600.0, "porosity": 0.1},
            {"temperature": 1600.0, "porosity": 0.2},
            id="elasticity",
        ),
        pytest.param(
            UO2Emissivity().validity,
            {"temperature": 2500.0},
            {"temperature": 2600.0},
            id="emissivity",
        ),
        pytest.param(
            UO2Densification().validity,
            {"resintering_density_fraction": 0.05},
            {"resintering_density_fraction": 0.09},
            id="densification",
        ),
        pytest.param(
            UO2SolidSwelling().validity,
            {"burnup_mwd_per_kghm": 100.0},
            {"burnup_mwd_per_kghm": 110.0},
            id="solid-swelling",
        ),
        pytest.param(
            FuelRelocation().validity,
            {"burnup_mwd_per_kghm": 0.0},
            {"burnup_mwd_per_kghm": 63.0},
            id="relocation",
        ),
        pytest.param(
            UO2Creep().validity,
            {"density_fraction": 0.95, "oxygen_to_metal_ratio": 2.0},
            {"density_fraction": 0.9, "oxygen_to_metal_ratio": 2.0},
            id="creep",
        ),
    ],
)
def test_validity_ranges_bound_the_source_envelope(
    validity: PropertyValidityRange, inside: dict[str, float], outside: dict[str, float]
) -> None:
    assert bool(validity.contains(**{k: jnp.asarray(v) for k, v in inside.items()}))
    assert not bool(validity.contains(**{k: jnp.asarray(v) for k, v in outside.items()}))


def _assert_jit_and_grad[L: eqx.Module](
    law: L, evaluate: Callable[[L, Array], Array]
) -> None:
    x = jnp.asarray([0.0, 0.25, 1.0])
    eager = evaluate(law, x)
    compiled = eqx.filter_jit(evaluate)(law, x)
    np.testing.assert_allclose(compiled, eager, rtol=1.0e-12)
    input_gradient = jax.vmap(jax.grad(lambda value: evaluate(law, value)))(x)
    assert bool(jnp.all(jnp.isfinite(input_gradient)))
    parameter_gradient = eqx.filter_grad(lambda model: jnp.sum(evaluate(model, x)))(law)
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(parameter_gradient, eqx.is_inexact_array)
    )
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def _enthalpy(law: UO2HeatCapacity, x: Array) -> Array:
    return law.enthalpy(1500.0 + x)


def _conductivity(law: UO2ThermalConductivity, x: Array) -> Array:
    return law.conductivity(1200.0 + x, 20.0 + x, jnp.asarray(0.05), jnp.asarray(0.0))


def _expansion(law: UO2ThermalExpansion, x: Array) -> Array:
    return law.linear_strain(1200.0 + x)


def _bulk_modulus(law: UO2Elasticity, x: Array) -> Array:
    return law.bulk_modulus(900.0 + x, jnp.asarray(0.05))


def _emissivity(law: UO2Emissivity, x: Array) -> Array:
    return law.emissivity(1200.0 + x)


def _densification(law: UO2Densification, x: Array) -> Array:
    return law.linear_strain(jnp.asarray(0.5), 50.0 + x, jnp.asarray(10400.0))


def _swelling(law: UO2SolidSwelling, x: Array) -> Array:
    return law.volumetric_strain(30.0 + x)


def _relocation(law: FuelRelocation, x: Array) -> Array:
    return law.diametral_strain(
        25.0e3 + 1.0e3 * x, 2.0 + x, jnp.asarray(170.0e-6), jnp.asarray(8.19e-3)
    )


def _creep(law: UO2Creep, x: Array) -> Array:
    return law.equivalent_creep_rate(
        20.0e6 + x,
        1500.0 + x,
        jnp.asarray(1.0e19),
        jnp.asarray(0.95),
        jnp.asarray(10.0e-6),
        jnp.asarray(2.01),
    )


@pytest.mark.parametrize(
    "check",
    [
        pytest.param(
            lambda: _assert_jit_and_grad(UO2HeatCapacity(), _enthalpy), id="enthalpy"
        ),
        pytest.param(
            lambda: _assert_jit_and_grad(UO2ThermalConductivity(), _conductivity),
            id="conductivity",
        ),
        pytest.param(
            lambda: _assert_jit_and_grad(UO2ThermalExpansion(), _expansion),
            id="expansion",
        ),
        pytest.param(
            lambda: _assert_jit_and_grad(UO2Elasticity(), _bulk_modulus), id="elasticity"
        ),
        pytest.param(
            lambda: _assert_jit_and_grad(UO2Emissivity(), _emissivity), id="emissivity"
        ),
        pytest.param(
            lambda: _assert_jit_and_grad(UO2Densification(), _densification),
            id="densification",
        ),
        pytest.param(
            lambda: _assert_jit_and_grad(UO2SolidSwelling(), _swelling), id="swelling"
        ),
        pytest.param(
            lambda: _assert_jit_and_grad(FuelRelocation(), _relocation), id="relocation"
        ),
        pytest.param(lambda: _assert_jit_and_grad(UO2Creep(), _creep), id="creep"),
    ],
)
def test_laws_are_jit_and_grad_compatible(check: Callable[[], None]) -> None:
    check()


@pytest.mark.parametrize(
    ("factory", "keyword"),
    [
        pytest.param(UO2HeatCapacity, "molar_mass", id="heat-capacity"),
        pytest.param(UO2ThermalConductivity, "phonon_constant", id="conductivity"),
        pytest.param(UO2ThermalExpansion, "defect_energy", id="expansion"),
        pytest.param(UO2Elasticity, "poisson_ratio", id="elasticity"),
        pytest.param(UO2Emissivity, "slope", id="emissivity"),
        pytest.param(UO2Densification, "fast_rate", id="densification"),
        pytest.param(UO2SolidSwelling, "low_burnup_rate", id="swelling"),
        pytest.param(FuelRelocation, "recoverable_fraction", id="relocation"),
        pytest.param(UO2Creep, "dislocation_stress_exponent", id="creep"),
    ],
)
def test_laws_refuse_nonfinite_parameters(
    factory: Callable[..., object], keyword: str
) -> None:
    with pytest.raises(ValueError, match=keyword):
        factory(**{keyword: math.nan})
