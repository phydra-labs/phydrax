#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import factorial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, special

from phydrax import ElectromagneticScaleContract
from phydrax.electromagnetics import (
    born_thermal_gaunt,
    ColdPlasmaDielectric,
    KappaDistribution,
    MagnetobremsstrahlungPlan,
    MagnetobremsstrahlungResult,
    MagnetobremsstrahlungRoute,
    MagnetobremsstrahlungStatus,
    PlasmaWaveMode,
    PowerLawDistribution,
    TabulatedGyrotropicDistribution,
    ThermalFreeFreeModel,
    ThermalJuttnerDistribution,
)


jax.config.update("jax_enable_x64", True)

# CODATA 2022 SI constants, written independently of the scale contract.
_E = 1.602176634e-19
_ME = 9.1093837139e-31
_C = 299792458.0
_EPS0 = 8.8541878188e-12
_KB = 1.380649e-23
_H = 6.62607015e-34
_SCALE = ElectromagneticScaleContract.si()
_GYRO = _E / _ME  # electron cyclotron frequency at 1 T


def _plasma(density: float, field: float = 1.0) -> ColdPlasmaDielectric:
    return ColdPlasmaDielectric(
        _SCALE,
        densities=[density],
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        magnetic_field=[0.0, 0.0, field],
    )


def _line_integral(
    plan: MagnetobremsstrahlungPlan, lower: float, upper: float, angle: float
) -> tuple[MagnetobremsstrahlungResult, np.ndarray]:
    omega = np.linspace(lower, upper, 129)
    result = eqx.filter_jit(plan.evaluate)(
        jnp.asarray(omega), jnp.full(omega.shape, angle)
    )
    assert not bool(jnp.any(result.status != 0))
    return result, omega


@pytest.mark.parametrize("harmonic", [2, 3], ids=["second", "third"])
def test_thermal_cyclotron_harmonic_matches_bekefi(harmonic: int) -> None:
    # Bekefi (1966): optically thin s-th harmonic of a nonrelativistic Maxwellian,
    # both modes, line integrated:
    # e² ω_s² N/(8π² ε₀ c) (1 + cos²θ) sin^{2s−2}θ s^{2s} (θ_e/2)^s / s!.
    theta_e, density, angle = 1.0e-5, 1.0e15, np.pi / 3.0
    plan = MagnetobremsstrahlungPlan(
        _plasma(1.0e14),
        ThermalJuttnerDistribution(theta_e),
        emitter_density=density,
        maximum_harmonics=4,
    )
    width = harmonic * _GYRO * np.sqrt(theta_e) * np.cos(angle)
    center = harmonic * _GYRO
    result, omega = _line_integral(
        plan, center - 8.0 * width, center + 8.0 * width, angle
    )
    total = np.trapezoid(np.asarray(jnp.sum(result.emission, axis=-1)), omega)
    expected = (
        _E**2
        * center**2
        * density
        / (8.0 * np.pi**2 * _EPS0 * _C)
        * (1.0 + np.cos(angle) ** 2)
        * np.sin(angle) ** (2 * harmonic - 2)
        * harmonic ** (2 * harmonic)
        * (theta_e / 2.0) ** harmonic
        / factorial(harmonic)
    )
    np.testing.assert_allclose(total, expected, rtol=1.0e-3)
    assert bool(jnp.all(result.harmonic_count == 1))
    np.testing.assert_array_equal(result.harmonic_range[..., 0], float(harmonic))


def test_perpendicular_ordinary_to_extraordinary_ratio_is_temperature() -> None:
    # At θ = π/2 the O mode couples only to v∥ J_s and the X mode to v⊥ J_s′, so
    # the line-integrated O/X ratio is ⟨β∥²⟩ = θ_e at leading order.
    theta_e = 1.0e-5
    plan = MagnetobremsstrahlungPlan(
        _plasma(1.0e14),
        ThermalJuttnerDistribution(theta_e),
        emitter_density=1.0e15,
        maximum_harmonics=4,
    )
    result, omega = _line_integral(
        plan, 2.0 * _GYRO * (1.0 - 40.0 * theta_e), 2.0 * _GYRO, np.pi / 2.0
    )
    ordinary = np.trapezoid(
        np.asarray(result.select(PlasmaWaveMode.ORDINARY, result.emission)), omega
    )
    extraordinary = np.trapezoid(
        np.asarray(result.select(PlasmaWaveMode.EXTRAORDINARY, result.emission)), omega
    )
    np.testing.assert_allclose(ordinary / extraordinary, theta_e, rtol=1.0e-3)


def _stix_null_vectors(x: float, y: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Magnetoionic modes (electrons) from a direct numpy null-vector solve."""
    r = 1.0 - x / (1.0 - y)
    l_ = 1.0 - x / (1.0 + y)
    p = 1.0 - x
    s, d = 0.5 * (r + l_), 0.5 * (r - l_)
    epsilon = np.array([[s, -1j * d, 0.0], [1j * d, s, 0.0], [0.0, 0.0, p]])
    kappa = np.array([np.sin(angle), 0.0, np.cos(angle)])
    sin2 = np.sin(angle) ** 2
    root = np.sqrt(0.25 * y**4 * sin2**2 + (1.0 - x) ** 2 * y**2 * np.cos(angle) ** 2)
    # Appleton–Hartree: the "+" root is the ordinary mode below the X cutoff.
    n2_ordinary = 1.0 - x * (1.0 - x) / (1.0 - x - 0.5 * y**2 * sin2 + root)
    n2_extraordinary = 1.0 - x * (1.0 - x) / (1.0 - x - 0.5 * y**2 * sin2 - root)
    vectors = []
    for n2 in (n2_ordinary, n2_extraordinary):
        operator = n2 * (np.outer(kappa, kappa) - np.eye(3)) + epsilon
        vectors.append(np.linalg.svd(operator)[2][-1].conj())
    return vectors[0], vectors[1]


def test_oblique_mode_projection_matches_magnetoionic_polarization() -> None:
    # Leading order in β: e*·V ∝ ē_x + i ē_y for electrons, so the O/X ratio is
    # |e_x − i e_y|²/|e_T|² of the two independent null vectors.
    theta_e, angle, background = 1.0e-6, np.pi / 4.0, 1.0e14
    plan = MagnetobremsstrahlungPlan(
        _plasma(background),
        ThermalJuttnerDistribution(theta_e),
        emitter_density=1.0e15,
        maximum_harmonics=4,
    )
    width = 2.0 * _GYRO * np.sqrt(theta_e) * np.cos(angle)
    result, omega = _line_integral(
        plan, 2.0 * _GYRO - 8.0 * width, 2.0 * _GYRO + 8.0 * width, angle
    )
    ordinary = np.trapezoid(
        np.asarray(result.select(PlasmaWaveMode.ORDINARY, result.emission)), omega
    )
    extraordinary = np.trapezoid(
        np.asarray(result.select(PlasmaWaveMode.EXTRAORDINARY, result.emission)), omega
    )
    x = background * _E**2 / (_EPS0 * _ME) / (2.0 * _GYRO) ** 2
    kappa = np.array([np.sin(angle), 0.0, np.cos(angle)])

    def coupling(vector: np.ndarray) -> float:
        transverse = 1.0 - abs(kappa @ vector) ** 2
        return abs(vector[0] - 1j * vector[1]) ** 2 / transverse

    e_ordinary, e_extraordinary = _stix_null_vectors(x, 0.5, angle)
    np.testing.assert_allclose(
        ordinary / extraordinary,
        coupling(e_ordinary) / coupling(e_extraordinary),
        rtol=1.0e-3,
    )


@pytest.mark.parametrize(
    ("route", "theta_e", "harmonics"),
    [("harmonic-sum", 0.02, (2.3, 3.1, 4.4)), ("continuous-harmonic", 1.0, (40.0, 90.0))],
    ids=["harmonic-sum", "continuous-harmonic"],
)
def test_thermal_emission_obeys_kirchhoff_in_a_dispersive_plasma(
    route: MagnetobremsstrahlungRoute, theta_e: float, harmonics: tuple[float, ...]
) -> None:
    plan = MagnetobremsstrahlungPlan(
        _plasma(3.0e18),
        ThermalJuttnerDistribution(theta_e),
        emitter_density=1.0e15,
        route=route,
        maximum_harmonics=32,
    )
    omega = np.asarray(harmonics) * _GYRO
    angle = np.linspace(0.4, 1.4, omega.size)
    result = plan.evaluate(jnp.asarray(omega), jnp.asarray(angle))
    assert bool(jnp.all(result.supported))
    index = np.asarray(result.faraday.wave.refractive_index.real)
    assert np.max(np.abs(index - 1.0)) > 1.0e-5
    temperature = theta_e * _ME * _C**2 / _KB
    rayleigh_jeans = _KB * temperature * omega[:, None] ** 2 / (8.0 * np.pi**3 * _C**2)
    np.testing.assert_allclose(
        result.emission, index**2 * rayleigh_jeans * result.absorption, rtol=1.0e-9
    )


def test_harmonic_set_includes_anomalous_doppler_harmonics_when_superluminal() -> None:
    # A dense plasma below the electron cyclotron frequency supports the whistler
    # with N∥ > 1, whose resonance hyperbola admits s ≤ 0.
    plasma = _plasma(1.0e20, 1.0)
    omega, angle = 0.3 * _GYRO, 0.2
    wave = plasma.refractive_indices(omega, angle)
    whistler = float(wave.select(PlasmaWaveMode.RIGHT, wave.refractive_index.real))
    assert whistler * np.cos(angle) > 1.0
    harmonic = MagnetobremsstrahlungPlan(
        plasma,
        ThermalJuttnerDistribution(0.05),
        emitter_density=1.0e15,
        maximum_harmonics=64,
    ).evaluate(omega, angle)
    lowest = harmonic.select(PlasmaWaveMode.RIGHT, harmonic.harmonic_range[..., 0])
    assert float(lowest) <= -1.0
    assert bool(harmonic.select(PlasmaWaveMode.RIGHT, harmonic.supported))
    assert float(harmonic.select(PlasmaWaveMode.RIGHT, harmonic.emission)) > 0.0
    continuous = MagnetobremsstrahlungPlan(
        plasma,
        ThermalJuttnerDistribution(0.05),
        emitter_density=1.0e15,
        route="continuous-harmonic",
    ).evaluate(omega, angle)
    status = int(continuous.select(PlasmaWaveMode.RIGHT, continuous.status))
    assert status & MagnetobremsstrahlungStatus.ANOMALOUS_DOPPLER
    assert np.isnan(float(continuous.select(PlasmaWaveMode.RIGHT, continuous.emission)))


def test_evanescent_modes_and_capacity_are_reported_not_zero_filled() -> None:
    plasma = _plasma(1.0e20, 1.0)
    plasma_frequency = np.sqrt(1.0e20 * _E**2 / (_EPS0 * _ME))
    plan = MagnetobremsstrahlungPlan(
        plasma,
        ThermalJuttnerDistribution(1.0e-3),
        emitter_density=1.0e15,
        maximum_harmonics=2,
    )
    # Above the electron cyclotron frequency and below ω_p and the left cutoff,
    # the ordinary (P < 0) root is evanescent.
    omega = 1.1 * _GYRO
    assert omega < plasma_frequency
    result = plan.evaluate(omega, np.pi / 2.0)
    ordinary = int(result.select(PlasmaWaveMode.ORDINARY, result.status))
    assert ordinary & MagnetobremsstrahlungStatus.EVANESCENT
    assert np.isnan(float(result.select(PlasmaWaveMode.ORDINARY, result.emission)))
    assert bool(jnp.all(jnp.isnan(result.stokes_emission)))

    tenuous = MagnetobremsstrahlungPlan(
        _plasma(1.0e14),
        ThermalJuttnerDistribution(0.2),
        emitter_density=1.0e15,
        maximum_harmonics=2,
    ).evaluate(5.0 * _GYRO, 1.0)
    assert bool(
        jnp.all(tenuous.status & MagnetobremsstrahlungStatus.HARMONIC_CAPACITY_EXCEEDED)
    )
    assert bool(jnp.all(tenuous.harmonic_count > 2))
    assert bool(jnp.all(jnp.isnan(tenuous.emission)))


def test_harmonic_count_matches_resonance_ellipse_intersection() -> None:
    # Near vacuum (N∥ = cos θ), s is resonant iff (γ − N∥u∥)Y⁻¹ = s has a solution
    # with |u| ≤ u_max: s ∈ [⌈√(1 − N∥²)/Y⌉, ⌊(γ_max + |N∥| u_max)/Y⌋].
    distribution = ThermalJuttnerDistribution(0.05, tail_tolerance=1.0e-12)
    angle, harmonic = 0.7, 3.4
    result = MagnetobremsstrahlungPlan(
        _plasma(1.0e14),
        distribution,
        emitter_density=1.0e15,
        maximum_harmonics=64,
    ).evaluate(harmonic * _GYRO, angle)
    _, upper = distribution.momentum_support()
    u_max = float(upper)
    n_par = np.asarray(result.faraday.wave.refractive_index.real) * np.cos(angle)
    y = 1.0 / harmonic
    lowest = np.ceil(np.sqrt(1.0 - n_par**2) / y)
    highest = np.floor((np.sqrt(1.0 + u_max**2) + np.abs(n_par) * u_max) / y)
    np.testing.assert_array_equal(result.harmonic_range[..., 0], lowest)
    np.testing.assert_array_equal(result.harmonic_range[..., 1], highest)
    np.testing.assert_array_equal(result.harmonic_count, highest - lowest + 1)
    assert float(result.tail_mass_bound) <= 1.0e-12
    # The omitted thermal mass is bounded by the reported tail bound.
    theta = 0.05
    gamma_max = np.sqrt(1.0 + u_max**2)
    full = integrate.quad(
        lambda g: g * np.sqrt(g * g - 1.0) * np.exp(-(g - 1.0) / theta), 1.0, np.inf
    )[0]
    tail = integrate.quad(
        lambda g: g * np.sqrt(g * g - 1.0) * np.exp(-(g - 1.0) / theta), gamma_max, np.inf
    )[0]
    assert tail / full <= float(result.tail_mass_bound)


def test_continuous_route_converges_to_harmonic_sum_with_harmonic_number() -> None:
    plasma = _plasma(1.0e16)
    distribution = ThermalJuttnerDistribution(0.3, tail_tolerance=1.0e-10)
    angle = 1.1
    errors = []
    effective = []
    for harmonic in (6.0, 24.0):
        omega = harmonic * _GYRO
        exact = MagnetobremsstrahlungPlan(
            plasma,
            distribution,
            emitter_density=1.0e15,
            maximum_harmonics=512,
        ).evaluate(omega, angle)
        fast = MagnetobremsstrahlungPlan(
            plasma,
            distribution,
            emitter_density=1.0e15,
            route="continuous-harmonic",
        ).evaluate(omega, angle)
        assert bool(jnp.all(exact.supported)) and bool(jnp.all(fast.supported))
        errors.append(float(jnp.max(jnp.abs(fast.emission / exact.emission - 1.0))))
        effective.append(float(jnp.min(fast.effective_harmonic)))
    assert effective[1] > effective[0]
    assert errors[1] < errors[0]
    assert errors[1] < 2.0e-2


def _bessel_k_five_thirds(t: float) -> float:
    return float(special.kv(np.float64(5.0 / 3.0), np.float64(t)))


def _synchrotron_function(x: float) -> float:
    return x * integrate.quad(_bessel_k_five_thirds, x, np.inf)[0]


@pytest.mark.parametrize("razin", [0.05, 1.0], ids=["tenuous", "razin-suppressed"])
def test_continuous_route_reproduces_razin_single_particle_spectrum(razin: float) -> None:
    # Ultra-relativistic single particle in a plasma of index n: the vacuum
    # spectrum with the phase mismatch 1/γ² → 2(1 − nβ), i.e. x = (ω/ω_c) R^{3/2}
    # and amplitude n R^{−1/2} with R = 2γ²(1 − nβ), which tends to the Razin
    # factor 1 + γ²ω_p²/ω² for n² = 1 − ω_p²/ω² and γ ≫ 1.
    # Isotropic shell: j = N P(ω, α = θ)/(4π).
    field, angle, shell = 1.0e-4, np.pi / 3.0, (19.9, 20.1)
    gyro = _E * field / _ME
    critical = 1.5 * 400.0 * gyro * np.sin(angle)
    plasma_frequency = razin * critical / 20.0
    background = plasma_frequency**2 * _EPS0 * _ME / _E**2
    plan = MagnetobremsstrahlungPlan(
        _plasma(background, field),
        PowerLawDistribution(0.0, *shell),
        emitter_density=1.0,
        route="continuous-harmonic",
        continuous_panels=4,
    )
    omega = np.asarray([0.5, 1.0, 3.0]) * critical
    result = plan.evaluate(jnp.asarray(omega), jnp.full(omega.shape, angle))
    assert bool(jnp.all(result.supported))
    index = np.asarray(jnp.mean(result.faraday.wave.refractive_index.real, axis=-1))
    expected = []
    for w, n in zip(omega, index, strict=True):

        def single(u: float, w: float = w, n: float = n) -> float:
            gamma2 = 1.0 + u * u
            razin_factor = 2.0 * gamma2 * (1.0 - n * np.sqrt(1.0 - 1.0 / gamma2))
            x = w / (1.5 * gamma2 * gyro * np.sin(angle)) * razin_factor**1.5
            return n * razin_factor**-0.5 * _synchrotron_function(x)

        mean = integrate.quad(single, *shell)[0] / (shell[1] - shell[0])
        expected.append(
            np.sqrt(3.0)
            * _E**2
            * gyro
            * np.sin(angle)
            / (8.0 * np.pi**2 * _EPS0 * _C)
            * mean
            / (4.0 * np.pi)
        )
    np.testing.assert_allclose(jnp.sum(result.emission, axis=-1), expected, rtol=1.0e-2)


def test_distributions_are_normalized_with_independent_quadrature() -> None:
    kappa = KappaDistribution(0.3, 4.0, 200.0)
    power_law = PowerLawDistribution(2.5, 0.5, 30.0)
    for distribution, upper in ((kappa, 200.0), (power_law, 30.0)):
        mass = integrate.quad(
            lambda u, d=distribution: 4.0 * np.pi * u * u * float(d.density(u, 0.0)),
            0.0,
            upper,
            points=(0.5, 1.0, 5.0),
            limit=400,
        )[0]
        np.testing.assert_allclose(mass, 1.0, rtol=1.0e-7)
    # Kappa tail bound dominates the true omitted mass.
    y = lambda g: 1.0 + (g - 1.0) / (4.0 * 0.3)
    total = integrate.quad(
        lambda g: g * np.sqrt(g * g - 1.0) * y(g) ** -5.0, 1.0, np.inf, limit=400
    )[0]
    gamma_max = np.sqrt(1.0 + 200.0**2)
    tail = integrate.quad(
        lambda g: g * np.sqrt(g * g - 1.0) * y(g) ** -5.0, gamma_max, np.inf, limit=400
    )[0]
    assert tail / total <= float(kappa.tail_mass_bound())


def test_tabulated_thermal_distribution_reproduces_juttner_emission() -> None:
    theta = 0.1
    momenta = np.geomspace(1.0e-3, 5.0, 160)
    cosines = np.linspace(-1.0, 1.0, 9)
    kinetic = momenta**2 / (1.0 + np.sqrt(1.0 + momenta**2))
    table = np.broadcast_to((-kinetic / theta)[:, None], (momenta.size, cosines.size))
    tabulated = TabulatedGyrotropicDistribution(momenta, cosines, table)
    thermal = ThermalJuttnerDistribution(theta)
    plasma = _plasma(1.0e16)
    omega = jnp.asarray([2.4, 3.6]) * _GYRO
    angle = jnp.asarray([0.9, 1.3])
    reference = MagnetobremsstrahlungPlan(
        plasma, thermal, emitter_density=1.0e15, maximum_harmonics=64
    ).evaluate(omega, angle)
    table_result = MagnetobremsstrahlungPlan(
        plasma, tabulated, emitter_density=1.0e15, maximum_harmonics=64
    ).evaluate(omega, angle)
    np.testing.assert_allclose(table_result.emission, reference.emission, rtol=2.0e-3)
    np.testing.assert_allclose(table_result.absorption, reference.absorption, rtol=2.0e-2)


def test_free_free_coefficients_match_rybicki_lightman_with_born_gaunt() -> None:
    model = ThermalFreeFreeModel(_SCALE, ion_charge_number=2.0)
    electrons, ions, temperature = 1.0e20, 5.0e19, 2.0e7
    frequency = np.asarray([1.0e12, 1.0e15, 1.0e18])
    result = model.evaluate(electrons, ions, temperature, 2.0 * np.pi * frequency)
    u = _H * frequency / (_KB * temperature)
    gaunt = np.sqrt(3.0) / np.pi * special.kve(0, 0.5 * u)
    epsilon_nu = (
        32.0
        * np.pi
        * _E**6
        / (3.0 * _ME * _C**3 * (4.0 * np.pi * _EPS0) ** 3)
        * np.sqrt(2.0 * np.pi / (3.0 * _KB * _ME * temperature))
        * 4.0
        * electrons
        * ions
        * np.exp(-u)
        * gaunt
    )
    j_nu = epsilon_nu / (4.0 * np.pi)
    np.testing.assert_allclose(result.emission[:, 0], j_nu / (2.0 * np.pi), rtol=1.0e-8)
    planck = 2.0 * _H * frequency**3 / _C**2 / np.expm1(u)
    np.testing.assert_allclose(result.absorption, j_nu / planck, rtol=1.0e-8)
    assert bool(jnp.all(result.evidence.qualified))
    # Born limits: (√3/π) ln(4/(e^{γ_E} u)) as u → 0 and √(3/(π u)) as u → ∞.
    small, large = 1.0e-6, 400.0
    np.testing.assert_allclose(
        born_thermal_gaunt(small),
        np.sqrt(3.0) / np.pi * np.log(4.0 / (np.exp(np.euler_gamma) * small)),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        born_thermal_gaunt(large), np.sqrt(3.0 / (np.pi * large)), rtol=5.0e-3
    )
    cold = model.evaluate(electrons, ions, 1.0e4, 2.0 * np.pi * 1.0e14)
    assert not bool(cold.evidence.born_valid)
    assert not bool(cold.evidence.qualified)
