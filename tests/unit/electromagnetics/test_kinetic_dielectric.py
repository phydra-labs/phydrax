#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, optimize, special

from phydrax import ElectromagneticScaleContract
from phydrax.electromagnetics import (
    AbstractGyrotropicDistribution,
    ColdPlasmaDielectric,
    HorseshoeDistribution,
    KineticDispersionProblem,
    KineticDispersionStatus,
    KineticPlasmaDielectric,
    KineticSusceptibilityModel,
    KineticSusceptibilityStatus,
    LossConeDistribution,
    PlasmaWaveMode,
    RelativisticWeakGrowthPlan,
    RingDistribution,
    ThermalJuttnerDistribution,
    WeakGrowthStatus,
)


SCALE = ElectromagneticScaleContract.si()
E = float(SCALE.elementary_charge)
M_E = float(SCALE.electron_mass)
EPS0 = float(SCALE.vacuum_permittivity)
C = float(SCALE.speed_of_light)
PROTON_RATIO = 1836.152673426

pytestmark = pytest.mark.strict_jax


def _plasma_frequency(density: float) -> float:
    return math.sqrt(density * E * E / (EPS0 * M_E))


def _electron_plasma(
    density: float,
    field: float,
    temperature: float,
    *,
    perpendicular_temperature: float | None = None,
    drift: float = 0.0,
    model: KineticSusceptibilityModel = "nonrelativistic",
    harmonic_count: int = 8,
) -> KineticPlasmaDielectric:
    return KineticPlasmaDielectric(
        SCALE,
        densities=[density],
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        magnetic_field=[0.0, 0.0, field],
        parallel_temperatures=[temperature],
        perpendicular_temperatures=[
            temperature
            if perpendicular_temperature is None
            else perpendicular_temperature
        ],
        parallel_drifts=[drift],
        model=model,
        harmonic_count=harmonic_count,
    )


def _landau_integral_below_axis(zeta: complex) -> complex:
    """``Z(ζ)`` for ``Im ζ < 0``: direct Hilbert integral plus the Landau residue."""

    def part(t: float, component: int) -> float:
        value = math.exp(-t * t) / (t - zeta) / math.sqrt(math.pi)
        return value.real if component == 0 else value.imag

    real = integrate.quad(part, -np.inf, np.inf, args=(0,), epsabs=1e-13, limit=400)[0]
    imag = integrate.quad(part, -np.inf, np.inf, args=(1,), epsabs=1e-13, limit=400)[0]
    return complex(real, imag) + 2j * math.sqrt(math.pi) * np.exp(-zeta * zeta)


def test_landau_damping_roots_match_published_langmuir_values() -> None:
    density = 1.0e18
    temperature = 100.0 * E
    plasma = _electron_plasma(density, 0.5, temperature)
    omega_p = _plasma_frequency(density)
    debye = math.sqrt(EPS0 * temperature / (density * E * E))
    problem = KineticDispersionProblem(plasma, model="electrostatic")
    result = problem.solve(
        jnp.asarray([0.3, 0.4, 0.5]) / debye,
        jnp.asarray(0.0),
        jnp.asarray(1.16 * omega_p + 0j),
    )
    # Langmuir roots of 1 + (1 + ζZ(ζ))/(kλ_D)² = 0 (Fried & Conte 1961 tables;
    # Canosa 1973): ω/ω_p at kλ_D = 0.3, 0.4, 0.5.
    reference = np.asarray([1.1598 - 0.0126j, 1.2850 - 0.0661j, 1.4157 - 0.1534j])
    frequencies = np.asarray(result.frequencies) / omega_p
    assert np.all(np.asarray(result.converged))
    assert np.all(np.asarray(result.status) == 0)
    np.testing.assert_allclose(frequencies.real, reference.real, atol=1.0e-4)
    np.testing.assert_allclose(frequencies.imag, reference.imag, atol=1.0e-4)


def test_bernstein_mode_matches_independent_perpendicular_dispersion() -> None:
    field = 0.1
    omega_c = E * field / M_E
    density = 4.0 * omega_c**2 * EPS0 * M_E / (E * E)  # ω_p = 2 |Ω|
    temperature = 1.0e3 * E
    plasma = _electron_plasma(density, field, temperature, harmonic_count=30)
    thermal = math.sqrt(2.0 * temperature / M_E)
    ratio = 4.0  # ω_p² / Ω²
    wavenumbers = np.asarray([0.8, 1.0, 1.2]) * math.sqrt(2.0) * omega_c / thermal

    def bernstein(x: float, lam: float) -> float:
        orders = np.arange(1, 60)
        terms = 2.0 * orders**2 * special.ive(orders, lam) / (x * x - orders**2)
        return 1.0 - ratio / lam * float(np.sum(terms))

    reference = []
    for k in wavenumbers:
        lam = (k * thermal / omega_c) ** 2 / 2.0
        reference.append(optimize.brentq(bernstein, 2.0 + 1e-9, 3.0 - 1e-9, args=(lam,)))
    problem = KineticDispersionProblem(plasma, model="electrostatic")
    result = problem.solve(
        jnp.asarray(wavenumbers),
        jnp.asarray(0.5 * math.pi),
        jnp.asarray(reference[0] * omega_c * (1.0 + 1e-3) + 0j),
    )
    frequencies = np.asarray(result.frequencies) / omega_c
    assert np.all(np.asarray(result.converged))
    np.testing.assert_allclose(frequencies.real, reference, rtol=1.0e-9)
    np.testing.assert_allclose(frequencies.imag, 0.0, atol=1.0e-9)
    assert np.all(np.asarray(result.truncation_ratio) < 1.0e-8)


def _whistler_reference(
    omega_p: float,
    omega_c: float,
    thermal: float,
    anisotropy: float,
    k: float,
    guess: complex,
) -> complex:
    """Parallel R-mode root of the scalar bi-Maxwellian dispersion relation (Gary 1993)."""

    def residual(omega: complex) -> complex:
        zeta = (omega - omega_c) / (k * thermal)  # electron resonance at ω = |Ω|
        z = 1j * math.sqrt(math.pi) * special.wofz(np.complex128(zeta))
        chi = (omega_p**2 / omega**2) * (
            omega / (k * thermal) * z + (anisotropy - 1.0) * (1.0 + zeta * z)
        )
        return 1.0 - (C * k / omega) ** 2 + chi

    def split(x: np.ndarray) -> list[float]:
        value = residual(complex(x[0], x[1]) * omega_c)
        return [value.real, value.imag]

    solution = optimize.root(
        split, [guess.real / omega_c, guess.imag / omega_c], tol=1e-13
    )
    root = complex(solution.x[0], solution.x[1]) * omega_c
    assert abs(residual(root)) < 1.0e-10
    return root


def test_whistler_anisotropy_instability_matches_scalar_dispersion() -> None:
    field = 1.0e-3
    omega_c = E * field / M_E
    omega_p = 4.0 * omega_c
    density = omega_p**2 * EPS0 * M_E / (E * E)
    parallel_temperature = 1.0e3 * E
    thermal = math.sqrt(2.0 * parallel_temperature / M_E)
    k = 1.0 * omega_p / C
    anisotropic = _electron_plasma(
        density,
        field,
        parallel_temperature,
        perpendicular_temperature=3.0 * parallel_temperature,
    )
    isotropic = _electron_plasma(density, field, parallel_temperature)
    guess = 0.54 * omega_c + 0.018j * omega_c
    for plasma, anisotropy, sign in ((anisotropic, 3.0, 1.0), (isotropic, 1.0, -1.0)):
        reference = _whistler_reference(omega_p, omega_c, thermal, anisotropy, k, guess)
        result = KineticDispersionProblem(plasma).solve(
            jnp.asarray([k]), jnp.asarray(0.0), jnp.asarray(reference * (1.0 + 2e-3))
        )
        root = complex(np.asarray(result.frequencies)[0])
        assert bool(np.asarray(result.converged)[0])
        assert abs(root - reference) / abs(reference) < 1.0e-8
        assert sign * root.imag > 0.0


@pytest.mark.parametrize(
    ("model", "temperature_ev", "tolerance"),
    [("nonrelativistic", 1.0e-5, 1.0e-8), ("weakly-relativistic", 1.0e-2, 1.0e-5)],
    ids=["nonrelativistic", "weakly-relativistic"],
)
def test_zero_temperature_limit_recovers_cold_stix_tensor(
    model: KineticSusceptibilityModel, temperature_ev: float, tolerance: float
) -> None:
    density = 1.0e18
    field = 0.5
    cold = ColdPlasmaDielectric(
        SCALE,
        densities=[density, density],
        charge_numbers=[-1.0, 1.0],
        mass_ratios=[1.0, PROTON_RATIO],
        magnetic_field=[0.0, 0.0, field],
    )
    temperature = temperature_ev * E
    hot = KineticPlasmaDielectric(
        SCALE,
        densities=[density, density],
        charge_numbers=[-1.0, 1.0],
        mass_ratios=[1.0, PROTON_RATIO],
        magnetic_field=[0.0, 0.0, field],
        parallel_temperatures=[temperature, temperature],
        perpendicular_temperatures=[temperature, temperature],
        model=model,
        harmonic_count=3,
    )
    omega = np.asarray([0.6, 1.7, 3.1]) * _plasma_frequency(density)
    k = 1.3 * omega / C
    theta = 0.6
    result = hot.susceptibility(
        jnp.asarray(omega),
        jnp.asarray(k * math.cos(theta)),
        jnp.asarray(k * math.sin(theta)),
    )
    expected = np.asarray(cold.dielectric_tensor(jnp.asarray(omega)))
    actual = np.asarray(result.dielectric_tensor)
    scale = np.max(np.abs(expected), axis=(-2, -1), keepdims=True)
    assert np.max(np.abs(actual - expected) / scale) < tolerance
    assert np.all(np.asarray(result.status) == 0)


def test_analytic_continuation_below_real_axis_for_both_parallel_signs() -> None:
    density = 1.0e18
    temperature = 50.0 * E
    drift = 2.0e6
    plasma = _electron_plasma(density, 0.5, temperature, drift=drift)
    thermal = math.sqrt(2.0 * temperature / M_E)
    omega_p = _plasma_frequency(density)
    k = 0.4 * omega_p / thermal
    omega = (1.3 - 0.4j) * omega_p

    def longitudinal(k_parallel: float, velocity: float) -> complex:
        # ε_zz(k∥ = ±k, k⊥ = 0) = 1 + (2ω_p²/k²w²)[1 + ζ Z(ζ)] with the causal ζ.
        zeta = (omega - k_parallel * velocity) / (abs(k_parallel) * thermal)
        z = _landau_integral_below_axis(zeta)
        return 1.0 + 2.0 * omega_p**2 / (k * k * thermal * thermal) * (1.0 + zeta * z)

    for k_parallel in (k, -k):
        result = plasma.susceptibility(
            jnp.asarray(omega), jnp.asarray(k_parallel), jnp.asarray(0.0)
        )
        # A drift V at k∥ < 0 is the mirror image of −V at |k∥|.
        expected = longitudinal(abs(k_parallel), drift if k_parallel > 0.0 else -drift)
        actual = complex(np.asarray(result.dielectric_tensor)[2, 2])
        assert abs(actual - expected) / abs(expected) < 1.0e-9


def test_parallel_wavenumber_sign_mirrors_the_tensor() -> None:
    plasma = _electron_plasma(
        1.0e18, 0.5, 200.0 * E, perpendicular_temperature=400.0 * E, harmonic_count=12
    )
    omega = jnp.asarray((1.1 - 0.05j) * _plasma_frequency(1.0e18))
    k_parallel = 3.0e3
    k_perp = jnp.asarray(2.0e3)
    forward = np.asarray(
        plasma.susceptibility(omega, jnp.asarray(k_parallel), k_perp).dielectric_tensor
    )
    backward = np.asarray(
        plasma.susceptibility(omega, jnp.asarray(-k_parallel), k_perp).dielectric_tensor
    )
    mirror = np.diag([1.0, 1.0, -1.0])
    np.testing.assert_allclose(
        backward, mirror @ forward @ mirror, rtol=1e-11, atol=1e-14
    )


def test_harmonic_truncation_is_reported_and_converges() -> None:
    density = 1.0e18
    field = 0.5
    temperature = 2.0e3 * E
    omega_c = E * field / M_E
    thermal = math.sqrt(2.0 * temperature / M_E)
    k_perp = 4.0 * omega_c / thermal  # λ = 8
    omega = jnp.asarray(2.5 * omega_c + 0j)
    args = (omega, jnp.asarray(1.0e2), jnp.asarray(k_perp))
    coarse = _electron_plasma(
        density, field, temperature, harmonic_count=3
    ).susceptibility(*args)
    fine = _electron_plasma(
        density, field, temperature, harmonic_count=40
    ).susceptibility(*args)
    finer = _electron_plasma(
        density, field, temperature, harmonic_count=60
    ).susceptibility(*args)
    assert float(np.asarray(coarse.truncation_ratio)[0]) > 1.0e-3
    assert int(coarse.status) & KineticSusceptibilityStatus.HARMONIC_TRUNCATION
    assert float(np.asarray(fine.truncation_ratio)[0]) < 1.0e-8
    assert int(fine.status) == 0
    np.testing.assert_allclose(
        np.asarray(fine.dielectric_tensor),
        np.asarray(finer.dielectric_tensor),
        rtol=1e-10,
        atol=1e-12,
    )


def test_root_failure_is_reported_without_zero_filling() -> None:
    density = 1.0e18
    plasma = _electron_plasma(density, 0.5, 100.0 * E)
    omega_p = _plasma_frequency(density)
    debye = math.sqrt(EPS0 * 100.0 * E / (density * E * E))
    problem = KineticDispersionProblem(plasma, model="electrostatic", maximum_steps=1)
    result = problem.solve(
        jnp.asarray([0.5 / debye]),
        jnp.asarray(0.0),
        jnp.asarray(3.0 * omega_p + 0.5j * omega_p),
    )
    assert not bool(np.asarray(result.converged)[0])
    assert int(np.asarray(result.status)[0]) & KineticDispersionStatus.NOT_CONVERGED
    frequency = complex(np.asarray(result.frequencies)[0])
    assert np.isfinite(frequency) and frequency != 0.0
    assert float(np.asarray(result.residual_norm)[0]) > 1.0e-6


def _loss_cone_growth(
    omega: float, omega_c: float, n_perp: float, hot_density: float, width: float
) -> float:
    """Closed-form O-mode fundamental growth at θ = π/2 for the j = 1 DGH loss cone.

    On the resonance circle ``u = √(γ_R² − 1)``, ``γ_R = |Ω|/ω``, the small-FLR
    ``J₁² ≈ a²/4`` makes ``χᴬ_zz`` a polynomial integral in ``u∥``; the cold
    ``P`` gives ``γ = −(ω/2) χᴬ_zz``.
    """
    y = omega_c / omega
    radius = math.sqrt(y * y - 1.0)
    normalization = 1.0 / (math.pi**1.5 * width**3)
    envelope = math.exp(-radius * radius / width**2)
    hot = hot_density * E * E / (EPS0 * M_E)
    bracket = 4.0 * radius**5 / (15.0 * width**2) - 16.0 * radius**7 / (105.0 * width**4)
    chi = (
        -2.0
        * math.pi**2
        * (hot / omega**2)
        * y
        * (n_perp**2 / (4.0 * y * y))
        * 2.0
        * normalization
        * envelope
        * bracket
    )
    return -0.5 * omega * chi


def test_electron_cyclotron_maser_growth_matches_analytic_loss_cone() -> None:
    field = 0.1
    omega_c = E * field / M_E
    density = 1.0e15  # ω_p ≈ 0.1 |Ω|
    background = ColdPlasmaDielectric(
        SCALE,
        densities=[density],
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        magnetic_field=[0.0, 0.0, field],
    )
    width = 0.05
    plan = RelativisticWeakGrowthPlan(
        background, LossConeDistribution(width, index=1), density=1.0e13, harmonic_count=3
    )
    omega = 0.999 * omega_c
    result = plan.evaluate(
        jnp.asarray(omega), jnp.asarray(0.5 * math.pi), PlasmaWaveMode.ORDINARY
    )
    reference = _loss_cone_growth(
        omega, omega_c, float(result.refractive_index), 1.0e13, width
    )
    growth = float(result.growth_rate)
    assert reference > 0.0
    assert abs(growth - reference) / reference < 2.0e-3
    assert int(result.status) == 0
    # Electrons (Ω < 0) resonate only at s < 0 below |Ω|; the s = −2, −3 circles
    # lie at u ≈ 1.7, 2.8 where the loss cone is empty.
    assert np.asarray(result.resonant).tolist() == [True, True, True] + [False] * 4
    harmonic = np.asarray(result.harmonic_growth_rates)
    assert harmonic[2] == pytest.approx(growth, rel=1e-12)


def test_weak_growth_refuses_non_elliptic_resonance() -> None:
    field = 0.1
    omega_c = E * field / M_E
    background = ColdPlasmaDielectric(
        SCALE,
        densities=[1.0e19],
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        magnetic_field=[0.0, 0.0, field],
    )
    plan = RelativisticWeakGrowthPlan(
        background,
        RingDistribution(0.1, perpendicular_spread=0.02, parallel_spread=0.02),
        density=1.0e15,
    )
    result = plan.evaluate(
        jnp.asarray(0.3 * omega_c), jnp.asarray(0.2), PlasmaWaveMode.RIGHT
    )
    assert int(result.status) & WeakGrowthStatus.RESONANCE_NOT_ELLIPTIC
    assert np.isnan(float(result.growth_rate))


def test_weakly_relativistic_perpendicular_absorption_matches_dnestrovskii_form() -> None:
    field = 1.0
    omega_c = E * field / M_E
    density = 1.0e18
    temperature = 1.0e3 * E
    mu = M_E * C * C / temperature
    plasma = _electron_plasma(
        density, field, temperature, model="weakly-relativistic", harmonic_count=3
    )
    omega = 0.998 * omega_c
    k_perp = omega / C
    result = plasma.susceptibility(
        jnp.asarray(omega), jnp.asarray(0.0), jnp.asarray(k_perp)
    )
    # Only n = −1 resonates: z = μ(1 − |Ω|/ω) < 0 and Im F_q(z, 0) = −π(−z)^{q−1}eᶻ/Γ(q).
    z = mu * (1.0 - omega_c / omega)
    b = C * k_perp / omega_c
    omega_p2 = density * E * E / (EPS0 * M_E)
    expected = (
        (omega_p2 / omega**2)
        * 0.5
        * b
        * b
        * math.pi
        * (-z) ** 2.5
        * math.exp(z)
        / math.gamma(3.5)
    )
    actual = float(np.asarray(result.species_susceptibility)[0, 2, 2].imag)
    assert abs(actual - expected) / expected < 1.0e-10


def test_weak_growth_agrees_with_shkarofsky_thermal_absorption() -> None:
    field = 1.0
    omega_c = E * field / M_E
    background_density = 1.0e17
    background = ColdPlasmaDielectric(
        SCALE,
        densities=[background_density],
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        magnetic_field=[0.0, 0.0, field],
    )
    temperature = 1.0e3 * E
    theta_t = temperature / (M_E * C * C)
    hot_density = 1.0e16
    plan = RelativisticWeakGrowthPlan(
        background,
        ThermalJuttnerDistribution(theta_t),
        density=hot_density,
        harmonic_count=3,
    )
    omega = 0.998 * omega_c
    growth = plan.evaluate(
        jnp.asarray(omega), jnp.asarray(0.5 * math.pi), PlasmaWaveMode.ORDINARY
    )
    n_perp = float(growth.refractive_index)
    shkarofsky = _electron_plasma(
        hot_density, field, temperature, model="weakly-relativistic", harmonic_count=3
    ).susceptibility(
        jnp.asarray(omega), jnp.asarray(0.0), jnp.asarray(n_perp * omega / C)
    )
    relativistic = float(np.asarray(growth.anti_hermitian_susceptibility)[2, 2].real)
    weakly = float(np.asarray(shkarofsky.species_susceptibility)[0, 2, 2].imag)
    assert float(growth.growth_rate) < 0.0
    # Agreement to the weakly relativistic model error O(T/mc²) ≈ 2e-3.
    assert abs(relativistic - weakly) / weakly < 1.0e-2


@pytest.mark.parametrize(
    "distribution",
    [
        LossConeDistribution(0.08, index=2),
        RingDistribution(0.2, perpendicular_spread=0.03, parallel_spread=0.05),
        HorseshoeDistribution(0.3, spread=0.04, loss_cone_angle=0.5, edge_width=0.05),
    ],
    ids=["loss-cone", "ring", "horseshoe"],
)
def test_driver_distributions_are_normalized_within_support(
    distribution: AbstractGyrotropicDistribution,
) -> None:
    lower, upper = (float(value) for value in distribution.momentum_support())
    rule_u, weight_u = np.polynomial.legendre.leggauss(200)
    rule_m, weight_m = np.polynomial.legendre.leggauss(400)
    u = 0.5 * (upper - lower) * (rule_u + 1.0) + lower
    cosine = rule_m
    radius, pitch = np.meshgrid(u, cosine, indexing="ij")
    density = np.asarray(
        distribution.density(
            jnp.asarray(radius * np.sqrt(1.0 - pitch**2)), jnp.asarray(radius * pitch)
        )
    )
    total = (
        2.0
        * math.pi
        * 0.5
        * (upper - lower)
        * np.einsum("i,j,ij->", weight_u * u * u, weight_m, density)
    )
    assert abs(total - 1.0) < 1.0e-8
    assert float(distribution.tail_mass_bound()) < 1.0e-12
