#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import ElectromagneticScaleContract
from phydrax._external_runtime import pin_executable
from phydrax.electromagnetics import (
    ColdPlasmaDielectric,
    KappaDistribution,
    MagnetobremsstrahlungPlan,
    MagnetobremsstrahlungResult,
    MagnetobremsstrahlungRoute,
    PlasmaWaveMode,
    PowerLawDistribution,
    read_symphony_output,
    read_ufgc_output,
    run_symphony,
    run_ufgc,
    symphony_input,
    SymphonyProvider,
    TabulatedGyrotropicDistribution,
    ThermalJuttnerDistribution,
    ufgc_input,
    UFGCProvider,
)
from phydrax.interchange import AdapterStatus


jax.config.update("jax_enable_x64", True)

# CODATA 2022 SI constants, written independently of the scale contract.
_E = 1.602176634e-19
_ME = 9.1093837139e-31
_C = 299792458.0
_MU0 = 1.25663706127e-6
_KB = 1.380649e-23
_REST_MEV = 0.51099895069
_SCALE = ElectromagneticScaleContract.si()
_GYRO = _E / _ME  # electron cyclotron angular frequency at 1 T
_DATA = Path(__file__).resolve().parents[2] / "data" / "providers"


def _plasma(density: float, field: float) -> ColdPlasmaDielectric:
    return ColdPlasmaDielectric(
        _SCALE,
        densities=[density],
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        magnetic_field=[0.0, 0.0, field],
    )


def _momentum(kinetic_mev: float) -> float:
    gamma = kinetic_mev / _REST_MEV + 1.0
    return float(np.sqrt(gamma * gamma - 1.0))


def _gauss(tesla: float) -> float:
    # B_G²/(8π) erg cm⁻³ = B_T²/(2μ₀) J m⁻³ with 1 J m⁻³ = 10 erg cm⁻³.
    return tesla * np.sqrt(4.0 * np.pi * 10.0 / _MU0)


def _power_law_plan(
    route: MagnetobremsstrahlungRoute = "harmonic-sum",
) -> MagnetobremsstrahlungPlan:
    # Solar-flare gyrosynchrotron: 100 G, n₀ + n_b = 1.01e9 cm⁻³, n_b = 1e7 cm⁻³,
    # 0.1–10 MeV electrons with dN/du ∝ u⁻³ (δ_p = 5 per d³p).
    return MagnetobremsstrahlungPlan(
        _plasma(1.01e15, 0.01),
        PowerLawDistribution(3.0, _momentum(0.1), _momentum(10.0)),
        emitter_density=1.0e13,
        route=route,
        maximum_harmonics=512,
    )


def _thermal_plan(
    density: float = 1.0e14, field: float = 0.01, temperature: float = 0.02
) -> MagnetobremsstrahlungPlan:
    return MagnetobremsstrahlungPlan(
        _plasma(density, field),
        ThermalJuttnerDistribution(temperature),
        emitter_density=density,
        maximum_harmonics=64,
    )


def _deck(inputs: dict[str, bytes], name: str) -> dict[str, Any]:
    return json.loads(inputs[name])


# -- deck generation ------------------------------------------------------------------


def test_ufgc_deck_maps_power_law_plan_to_momentum_power_law_in_cgs() -> None:
    omega = 2.0 * np.pi * np.array([1.0e9, 3.0e9])
    theta = np.array([np.pi / 3.0, 2.0 * np.pi / 3.0])
    deck = _deck(
        ufgc_input(_power_law_plan(), angular_frequencies=omega, angles=theta),
        "ufgc_input.json",
    )
    np.testing.assert_allclose(deck["frequencies_ghz"], [1.0, 3.0], rtol=1.0e-15)
    voxels = np.asarray(deck["voxel_parameters"])
    assert voxels.shape == (4, 24)
    # Two lines of sight per angle, in angle order.
    np.testing.assert_allclose(voxels[:, 4], [60.0, 60.0, 120.0, 120.0], rtol=1.0e-14)
    np.testing.assert_allclose(voxels[:, 3], _gauss(0.01), rtol=1.0e-12)
    # UFGC's plasma density is n₀ + n_b, so the background excludes the emitters.
    np.testing.assert_allclose(voxels[:, 2], 1.0e9, rtol=1.0e-12)
    np.testing.assert_allclose(voxels[:, 7], 1.0e7, rtol=1.0e-12)
    np.testing.assert_allclose(voxels[:, 9], 0.1, rtol=1.0e-9)
    np.testing.assert_allclose(voxels[:, 10], 10.0, rtol=1.0e-9)
    # dN/du ∝ u⁻³ is f(p) ∝ p⁻⁵ per d³p.
    np.testing.assert_array_equal(voxels[:, 12], 5.0)


def test_ufgc_route_selects_exact_or_continuous_code_over_the_whole_band() -> None:
    omega = 2.0 * np.pi * np.array([1.0e9, 3.0e9])
    gyrofrequency = _E * 0.01 / (2.0 * np.pi * _ME)
    exact = _deck(
        ufgc_input(_power_law_plan(), angular_frequencies=omega, angles=[1.0]),
        "ufgc_input.json",
    )
    continuous = _deck(
        ufgc_input(
            _power_law_plan("continuous-harmonic"),
            angular_frequencies=omega,
            angles=[1.0],
        ),
        "ufgc_input.json",
    )
    # f_C, f_WH in gyrofrequency units: exact below, continuous above.
    boundaries = np.asarray(exact["pixel_parameters"])[:, 3:5]
    assert np.all(boundaries > 3.0e9 / gyrofrequency)
    np.testing.assert_array_equal(np.asarray(continuous["pixel_parameters"])[:, 3], 0.0)


def test_ufgc_kappa_width_and_cutoff_follow_the_plan_distribution() -> None:
    theta_e, kappa, u_max = 0.01, 4.0, 3.0
    plan = MagnetobremsstrahlungPlan(
        _plasma(1.0e14, 0.01),
        KappaDistribution(theta_e, kappa, u_max),
        emitter_density=1.0e14,
    )
    deck = _deck(
        ufgc_input(plan, angular_frequencies=[1.0e10], angles=[1.0]), "ufgc_input.json"
    )
    voxel = np.asarray(deck["voxel_parameters"])[0]
    rest_kelvin = _ME * _C * _C / _KB
    # UFGC's (κ − 3/2) k_B T₀/(mc²) equals the plan's κθ.
    np.testing.assert_allclose(
        voxel[1], kappa * theta_e / (kappa - 1.5) * rest_kelvin, rtol=1.0e-12
    )
    np.testing.assert_allclose(voxel[2], 1.0e8, rtol=1.0e-12)
    np.testing.assert_allclose(voxel[8], kappa)
    np.testing.assert_allclose(
        voxel[10], (np.sqrt(1.0 + u_max * u_max) - 1.0) * _REST_MEV, rtol=1.0e-9
    )


def test_symphony_deck_rescales_power_law_to_its_unbounded_normalization() -> None:
    index, u_low, u_high = 3.0, 3.0, 1.0e3
    plan = MagnetobremsstrahlungPlan(
        _plasma(1.0e6, 0.1),
        PowerLawDistribution(index, u_low, u_high),
        emitter_density=1.0e12,
    )
    omega = _GYRO * 0.1 * np.array([300.0, 3000.0])
    deck = _deck(
        symphony_input(plan, angular_frequencies=omega, angles=[np.pi / 3.0]),
        "symphony_input.json",
    )
    np.testing.assert_allclose(deck["frequencies_hz"], omega / (2.0 * np.pi), rtol=1e-15)
    np.testing.assert_allclose(deck["magnetic_field_gauss"], _gauss(0.1), rtol=1.0e-12)
    # n_s (p − 1) γ^{−p} on [1, ∞) equals N (p − 1) u^{−p}/(u₁^{1−p} − u₂^{1−p}).
    expected = 1.0e6 / (u_low ** (1.0 - index) - u_high ** (1.0 - index))
    np.testing.assert_allclose(deck["electron_density_cgs"], expected, rtol=1.0e-12)
    model = deck["distribution"]
    assert model["kind"] == "power-law"
    np.testing.assert_allclose(model["gamma_min"], np.sqrt(10.0), rtol=1.0e-15)
    np.testing.assert_allclose(model["power_law_p"], index)


# -- refusal --------------------------------------------------------------------------


def _tabulated_plan() -> MagnetobremsstrahlungPlan:
    momenta = np.geomspace(1.0, 10.0, 4)
    cosines = np.linspace(-1.0, 1.0, 3)
    return MagnetobremsstrahlungPlan(
        _plasma(1.0e14, 0.01),
        TabulatedGyrotropicDistribution(momenta, cosines, np.zeros((4, 3))),
        emitter_density=1.0e14,
    )


_UFGC_REFUSALS = {
    "tabulated": (_tabulated_plan, [1.0], "supports thermal"),
    "positrons": (
        lambda: MagnetobremsstrahlungPlan(
            _plasma(1.0e14, 0.01),
            ThermalJuttnerDistribution(0.02),
            emitter_density=1.0e14,
            emitter_charge_number=1.0,
        ),
        [1.0],
        "electron emitters",
    ),
    "two-species": (
        lambda: MagnetobremsstrahlungPlan(
            ColdPlasmaDielectric(
                _SCALE,
                densities=[1.0e14, 1.0e14],
                charge_numbers=[-1.0, 1.0],
                mass_ratios=[1.0, 1836.15267343],
                magnetic_field=[0.0, 0.0, 0.01],
            ),
            ThermalJuttnerDistribution(0.02),
            emitter_density=1.0e14,
        ),
        [1.0],
        "one electron species",
    ),
    "thermal-not-plasma": (
        lambda: MagnetobremsstrahlungPlan(
            _plasma(1.0e14, 0.01),
            ThermalJuttnerDistribution(0.02),
            emitter_density=1.0e12,
        ),
        [1.0],
        "must equal the plasma density",
    ),
    "emitters-exceed-plasma": (
        lambda: MagnetobremsstrahlungPlan(
            _plasma(1.0e12, 0.01),
            PowerLawDistribution(3.0, 1.0, 10.0),
            emitter_density=1.0e13,
        ),
        [1.0],
        "must not exceed",
    ),
    "angle-pi": (_thermal_plan, [np.pi], "angles must lie"),
}


@pytest.mark.parametrize("case", sorted(_UFGC_REFUSALS), ids=sorted(_UFGC_REFUSALS))
def test_ufgc_refuses_inputs_outside_its_subset(case: str) -> None:
    factory, angles, message = _UFGC_REFUSALS[case]
    with pytest.raises(ValueError, match=message):
        ufgc_input(factory(), angular_frequencies=[1.0e10], angles=angles)


_SYMPHONY_REFUSALS = {
    "tabulated": (_tabulated_plan, [1.0], [1.0e10], "supports thermal"),
    "shallow-power-law": (
        lambda: MagnetobremsstrahlungPlan(
            _plasma(1.0e6, 0.1),
            PowerLawDistribution(1.0, 3.0, 10.0),
            emitter_density=1.0e12,
        ),
        [1.0],
        [1.0e10],
        "index > 1",
    ),
    "obtuse-angle": (_thermal_plan, [2.0], [1.0e10], "angles must lie"),
    "perpendicular": (_thermal_plan, [0.5 * np.pi], [1.0e10], "angles must lie"),
    "decreasing-frequencies": (
        _thermal_plan,
        [1.0],
        [2.0e10, 1.0e10],
        "strictly increasing",
    ),
}


@pytest.mark.parametrize(
    "case", sorted(_SYMPHONY_REFUSALS), ids=sorted(_SYMPHONY_REFUSALS)
)
def test_symphony_refuses_inputs_outside_its_subset(case: str) -> None:
    factory, angles, frequencies, message = _SYMPHONY_REFUSALS[case]
    with pytest.raises(ValueError, match=message):
        symphony_input(factory(), angular_frequencies=frequencies, angles=angles)


# -- parsers on real provider output --------------------------------------------------


def _rayleigh_jeans(temperature: float, omega: np.ndarray) -> np.ndarray:
    """``k T ω²/(8π³c²)``: one mode's thermal ``j_ω/α`` in vacuum (SI)."""
    return temperature * _ME * _C * _C * omega**2 / (8.0 * np.pi**3 * _C * _C)


def test_ufgc_parser_recovers_thermal_mode_coefficients() -> None:
    """Parse real UFGC output: Kirchhoff per mode and θ ↔ π − θ mode identity.

    ``tests/data/providers/ufgc/thermal_output.json`` was produced on 2026-09-28
    by ``run_ufgc`` with UFGC revision 5e014ba (``MWTransferArr_arm64.so``,
    sha256 29a7af56…22cddb, clang++ -O3, Homebrew libomp 22.1.8) under Python
    3.11.15, for the plan below at ``ω = (2.5, 5, 9.5) eB/mₑ`` and
    ``θ = (π/3, 2π/3)``. At ``ω_p²/ω² < 2·10⁻⁸`` both modes have ``n² = 1``, so
    Kirchhoff's law gives ``j_σ/κ_σ = kTω²/(8π³c²)``; a gyrotropic population
    radiates each mode identically at ``θ`` and ``π − θ``, where UFGC swaps its
    left/right rows.
    """
    plan = MagnetobremsstrahlungPlan(
        _plasma(1.0e8, 0.01), ThermalJuttnerDistribution(0.02), emitter_density=1.0e8
    )
    data = (_DATA / "ufgc" / "thermal_output.json").read_bytes()
    coefficients = read_ufgc_output(data, plan)
    omega = _GYRO * 0.01 * np.array([2.5, 5.0, 9.5])
    np.testing.assert_allclose(coefficients.angular_frequencies, omega, rtol=1e-15)
    ratio = coefficients.emission / coefficients.absorption
    expected = _rayleigh_jeans(0.02, omega)[None, :, None]
    np.testing.assert_allclose(ratio, np.broadcast_to(expected, ratio.shape), rtol=1e-7)
    np.testing.assert_allclose(
        coefficients.emission[0], coefficients.emission[1], rtol=1e-12
    )
    # The extraordinary mode dominates at low harmonics of a mildly relativistic plasma.
    ordinary = coefficients.select(PlasmaWaveMode.ORDINARY, coefficients.emission)
    extraordinary = coefficients.select(
        PlasmaWaveMode.EXTRAORDINARY, coefficients.emission
    )
    assert np.all(extraordinary > ordinary)


def test_symphony_parser_recovers_thermal_stokes_coefficients() -> None:
    """Parse real Symphony output: Kirchhoff per Stokes parameter and basis sign.

    ``tests/data/providers/symphony/thermal_output.json`` was produced on
    2026-09-28 by ``run_symphony`` with Symphony revision a869c6b
    (``symphonyPy.so``, sha256 55b03bd2…582b42, GSL 2.8, NumPy 1.26.4) under
    Python 3.11.15, for ``θ = 1`` electrons at ``ω = (30, 300) eB/mₑ`` in 0.1 T
    and ``θ = (π/3, 1.2)``. Vacuum Kirchhoff gives ``j_S = B_ω α_S`` for every
    Stokes ``S`` with ``B_ω = kTω²/(4π³c²)`` (both polarizations, per unit
    angular frequency), within Symphony's 10⁻³ integration tolerance; synchrotron
    emission is polarized across the projected field, so ``Q < 0`` when
    ``ê₁`` lies in the ``k``–``B₀`` plane.
    """
    plan = MagnetobremsstrahlungPlan(
        _plasma(1.0e16, 0.1), ThermalJuttnerDistribution(1.0), emitter_density=1.0e12
    )
    data = (_DATA / "symphony" / "thermal_output.json").read_bytes()
    coefficients = read_symphony_output(data, plan)
    omega = _GYRO * 0.1 * np.array([30.0, 300.0])
    np.testing.assert_allclose(coefficients.angular_frequencies, omega, rtol=1e-15)
    planck = 2.0 * _rayleigh_jeans(1.0, omega)[None, :]
    for stokes in (0, 1, 3):
        np.testing.assert_allclose(
            coefficients.stokes_emission[..., stokes]
            / coefficients.stokes_absorption[..., stokes],
            np.broadcast_to(planck, (2, 2)),
            rtol=5.0e-3,
        )
    assert np.all(coefficients.stokes_emission[..., 1] < 0.0)
    np.testing.assert_array_equal(coefficients.stokes_emission[..., 2], 0.0)


# -- live oracle comparisons ----------------------------------------------------------


_UFGC_ENVIRONMENT = (
    "PHYDRAX_UFGC_PYTHON",
    "PHYDRAX_UFGC_PYTHON_VERSION",
    "PHYDRAX_UFGC_LIBRARY",
    "PHYDRAX_UFGC_VERSION",
)
_SYMPHONY_ENVIRONMENT = (
    "PHYDRAX_SYMPHONY_PYTHON",
    "PHYDRAX_SYMPHONY_PYTHON_VERSION",
    "PHYDRAX_SYMPHONY_MODULE",
    "PHYDRAX_SYMPHONY_VERSION",
)


def _environment(names: tuple[str, ...], what: str) -> tuple[str, ...]:
    if any(name not in os.environ for name in names):
        pytest.skip(f"set {', '.join(names)} to a pinned {what}")
    return tuple(os.environ[name] for name in names)


def _ufgc() -> UFGCProvider:
    python, python_version, library, version = _environment(
        _UFGC_ENVIRONMENT, "UFGC build"
    )
    return UFGCProvider(
        pin_executable(python, version=python_version, license_id="PSF-2.0"),
        pin_executable(library, version=version, license_id="GPL-3.0-only"),
    )


def _symphony() -> SymphonyProvider:
    python, python_version, module, version = _environment(
        _SYMPHONY_ENVIRONMENT, "Symphony build"
    )
    return SymphonyProvider(
        pin_executable(python, version=python_version, license_id="PSF-2.0"),
        pin_executable(module, version=version, license_id="GPL-3.0-only"),
    )


def _reference(
    plan: MagnetobremsstrahlungPlan, omega: np.ndarray, theta: np.ndarray
) -> MagnetobremsstrahlungResult:
    result = plan.evaluate(jnp.asarray(omega)[None, :], jnp.asarray(theta)[:, None])
    assert bool(jnp.all(result.supported))
    return result


def _assert_modes_match(
    provider: UFGCProvider,
    plan: MagnetobremsstrahlungPlan,
    omega: np.ndarray,
    theta: np.ndarray,
    rtol: float,
    directory: Path,
) -> None:
    result = run_ufgc(provider, plan, directory, angular_frequencies=omega, angles=theta)
    reference = _reference(plan, omega, theta)
    for mode in (PlasmaWaveMode.ORDINARY, PlasmaWaveMode.EXTRAORDINARY):
        coefficients = result.coefficients
        np.testing.assert_allclose(
            coefficients.select(mode, coefficients.emission),
            np.asarray(reference.select(mode, reference.emission)),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            coefficients.select(mode, coefficients.absorption),
            np.asarray(reference.select(mode, reference.absorption)),
            rtol=rtol,
        )
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    assert result.report.source_id == result.output_sha256
    assert result.executable_sha256 == provider.library.sha256
    assert result.license_id == "GPL-3.0-only"


def test_ufgc_thermal_harmonic_sum_matches_pinned_ufgc(tmp_path: Path) -> None:
    """Mildly relativistic (θ = 0.02) thermal gyroresonance at 2.5–9.5 ω_B.

    Both codes sum exact integer harmonics with exact Bessel functions in the
    same magnetoionic modes (ω_p/ω_B = 0.32); UFGC's energy quadrature agrees
    with the embedded Gauss–Kronrod sum to a few 10⁻³ (observed ≤ 2.3·10⁻³).
    """
    omega = _GYRO * 0.01 * np.array([2.5, 5.0, 9.5])
    theta = np.array([np.pi / 3.0, 1.3, 2.0 * np.pi / 3.0])
    _assert_modes_match(_ufgc(), _thermal_plan(), omega, theta, 5.0e-3, tmp_path)


def test_ufgc_power_law_harmonic_sum_matches_pinned_ufgc(tmp_path: Path) -> None:
    """0.1–10 MeV momentum power law at 1–2 GHz (3.6–7.1 ω_B, up to ~200 harmonics).

    Exact codes on both sides agree to their quadrature accuracy (observed
    ≤ 1.4·10⁻⁴).
    """
    omega = 2.0 * np.pi * np.array([1.0e9, 2.0e9])
    _assert_modes_match(
        _ufgc(), _power_law_plan(), omega, np.array([np.pi / 3.0]), 1.0e-3, tmp_path
    )


def test_ufgc_continuous_code_matches_continuous_route(tmp_path: Path) -> None:
    """Same power law at 3–30 GHz with both continuous-harmonic codes.

    UFGC's continuous code uses approximate Bessel functions (Fleishman &
    Kuznetsov 2010, accurate to a few per cent), while the plan integrates exact
    real-order Bessel functions (it matches UFGC's exact code to 10⁻⁴ here);
    observed differences ≤ 2.4 %.
    """
    omega = 2.0 * np.pi * np.array([3.0e9, 10.0e9, 30.0e9])
    _assert_modes_match(
        _ufgc(),
        _power_law_plan("continuous-harmonic"),
        omega,
        np.array([np.pi / 3.0, 1.3]),
        5.0e-2,
        tmp_path,
    )


def test_ufgc_kappa_harmonic_sum_matches_pinned_ufgc(tmp_path: Path) -> None:
    """κ = 4, θ = 0.01 kappa population truncated at u = 3 (observed ≤ 5·10⁻⁴)."""
    plan = MagnetobremsstrahlungPlan(
        _plasma(1.0e14, 0.01),
        KappaDistribution(0.01, 4.0, 3.0),
        emitter_density=1.0e14,
        maximum_harmonics=128,
    )
    omega = _GYRO * 0.01 * np.array([3.5, 7.5])
    _assert_modes_match(_ufgc(), plan, omega, np.array([np.pi / 3.0]), 2.0e-3, tmp_path)


def _assert_stokes_intensity_match(
    plan: MagnetobremsstrahlungPlan,
    omega: np.ndarray,
    theta: np.ndarray,
    rtol: float,
    directory: Path,
) -> None:
    """Stokes ``I`` coefficients agree; ``V`` has the same sense.

    At ``ω_p/ω ≤ 3·10⁻³`` the plan's modes carry ``n_σ² − 1 < 10⁻⁵`` and their
    incoherent sum gives vacuum ``j_I`` and ``α_I = (α_O + α_X)/2``. Linear
    polarization is not comparable: quasi-circular modes carry no ``Q``, whereas
    Symphony's vacuum emission does, and their ellipticity scales ``V``.
    """
    provider = _symphony()
    result = run_symphony(
        provider, plan, directory, angular_frequencies=omega, angles=theta
    )
    reference = _reference(plan, omega, theta)
    coefficients = result.coefficients
    np.testing.assert_allclose(
        coefficients.stokes_emission[..., 0],
        np.asarray(reference.stokes_emission[..., 0]),
        rtol=rtol,
    )
    np.testing.assert_allclose(
        coefficients.stokes_absorption[..., 0],
        np.asarray(reference.propagation_matrix[..., 0, 0]),
        rtol=rtol,
    )
    np.testing.assert_array_equal(
        np.sign(coefficients.stokes_emission[..., 3]),
        np.sign(np.asarray(reference.stokes_emission[..., 3])),
    )
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    assert result.executable_sha256 == provider.module.sha256


def test_symphony_thermal_matches_pinned_symphony(tmp_path: Path) -> None:
    """θ = 1 thermal synchrotron at 30 and 300 ω_B (observed ≤ 0.36 %).

    Symphony's nested QAG integrals at relative tolerance 10⁻³, its continuous
    harmonic integral beyond n = 30 and the plasma's ``n_σ − 1`` set the 1 %
    tolerance.
    """
    plan = MagnetobremsstrahlungPlan(
        _plasma(1.0e16, 0.1),
        ThermalJuttnerDistribution(1.0),
        emitter_density=1.0e12,
        route="continuous-harmonic",
    )
    omega = _GYRO * 0.1 * np.array([30.0, 300.0])
    _assert_stokes_intensity_match(
        plan, omega, np.array([np.pi / 3.0, 1.2]), 1.0e-2, tmp_path
    )


def test_symphony_kappa_matches_pinned_symphony(tmp_path: Path) -> None:
    """κ = 4, w = 1 kappa synchrotron at 100 and 1000 ω_B (observed ≤ 0.14 %).

    The plan truncates at u = 10⁴ and Symphony does not; the declared loss
    reports the omitted tail mass.
    """
    plan = MagnetobremsstrahlungPlan(
        _plasma(1.0e16, 0.1),
        KappaDistribution(1.0, 4.0, 1.0e4),
        emitter_density=1.0e12,
        route="continuous-harmonic",
        continuous_panels=8,
    )
    omega = _GYRO * 0.1 * np.array([100.0, 1000.0])
    _assert_stokes_intensity_match(plan, omega, np.array([np.pi / 3.0]), 1.0e-2, tmp_path)


def test_symphony_power_law_matches_inside_its_window(tmp_path: Path) -> None:
    """dN/du ∝ u⁻³ on u ∈ [3, 10³] at 300 and 3000 ω_B (observed ≤ 2.2 %).

    The pinned Symphony integrates γ⁻³ over γ ∈ [1, ∞) (a declared loss): the
    electrons it adds below u = 3 and above u = 10³, and the difference between
    ``dN/du`` and ``dN/dγ`` near ``u₁``, perturb the coefficients by a few per cent
    at ``γ₁² ω_B ≪ ω ≪ γ₂² ω_B``.
    """
    plan = MagnetobremsstrahlungPlan(
        _plasma(1.0e16, 0.1),
        PowerLawDistribution(3.0, 3.0, 1.0e3),
        emitter_density=1.0e12,
        route="continuous-harmonic",
        continuous_panels=8,
    )
    omega = _GYRO * 0.1 * np.array([300.0, 3000.0])
    _assert_stokes_intensity_match(plan, omega, np.array([np.pi / 3.0]), 5.0e-2, tmp_path)
