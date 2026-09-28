#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-independent averaged FEL against independent high-gain references.

References: the cold 1-D FEL cubic ``μ³ = i(2ρk_u)³`` whose growing root has
field amplitude rate ``√3 ρ k_u`` and power gain length ``λ_u/(4π√3ρ)``
(Bonifacio, Pellegrini & Narducci, Opt. Commun. 50, 373, 1984); the linearized
1-D Vlasov dispersion relation ``μ̂ = i ∫ f(η̂) / (μ̂ + iη̂)² dη̂`` for a Gaussian
energy distribution, solved here by Gauss–Hermite quadrature and Newton
iteration; the 1-D saturated power ``≈ 1.4 ρ P_beam`` of the universal scaling;
Ming Xie's fitted 3-D gain length (Xie, PAC 1995; NIM A 445, 59, 2000), stated
to reproduce the exact eigenmode growth within about ten percent; Poisson
bunching ``⟨|b_h|²⟩ = 1/N_λ`` (Fawley, PRST-AB 5, 070701, 2002); SciPy Bessel
functions for ``[JJ]_h``; the ``P_3 ∝ P_1³`` law of nonlinear harmonic
generation; exact linear betatron transfer matrices; and a closed-form wake
potential of a constant wake.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import constants, special

from phydrax import ElectromagneticScaleContract
from phydrax.applications.accelerator import InsertionDeviceField, WakeFunctionPlan
from phydrax.applications.accelerator.fel import (
    fel_scaling_estimate,
    FELBeamSlices,
    FELLoading,
    FELPlan,
    FELResult,
    FELScalingEstimate,
    FELSeed,
    FELStatus,
    FELUndulatorLattice,
    FELUndulatorSegment,
    FELWakeLoss,
    undulator_coupling_factors,
)
from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import AngularSpectrumPlan, PlaneFieldSpace


SCALE = ElectromagneticScaleContract.si()
LIGHT = constants.c
CHARGE = constants.e
MASS = constants.m_e
PERIOD = 0.03
WAVENUMBER = 2.0 * math.pi / PERIOD
GAMMA = 1.0e4
DEFLECTION = 3.5
CURRENT = 3000.0
BEAM_SIZE = 30.0e-6
# Tiny emittance with a matching large beta keeps the 1-D slice area 2πσ²
# while its angular spread is negligible.
COLD_EMITTANCE = 1.0e-9
COLD_BETA = BEAM_SIZE**2 * GAMMA / COLD_EMITTANCE


def _peak_field(deflection: float) -> float:
    return deflection * 2.0 * math.pi * MASS * LIGHT / (CHARGE * PERIOD)


def _lattice(
    peaks: tuple[float, ...],
    *,
    periods: int,
    polarization: str = "planar",
    step: float = 0.05,
    drift_length: float = 0.0,
    smooth_focusing_gradient: float = 0.0,
    quadrupole_integrated_gradient: float = 0.0,
) -> FELUndulatorLattice:
    segments = tuple(
        FELUndulatorSegment(
            InsertionDeviceField(
                peak,
                PERIOD,
                periods,
                polarization=polarization,  # ty: ignore[invalid-argument-type]
                center=0.0,
                aperture=(1.0e-2, 1.0e-2),
            ),
            drift_length=drift_length,
            smooth_focusing_gradient=smooth_focusing_gradient,
            quadrupole_integrated_gradient=quadrupole_integrated_gradient,
        )
        for peak in peaks
    )
    return FELUndulatorLattice(SCALE, segments, step_length=step)


def _cold_slices(spread: float = 0.0) -> FELBeamSlices:
    return FELBeamSlices(
        [0.0],
        [CURRENT],
        [GAMMA],
        [spread],
        [[COLD_EMITTANCE, COLD_EMITTANCE]],
        [[COLD_BETA, COLD_BETA]],
        [[0.0, 0.0]],
    )


def _solve(plan: FELPlan, slices: FELBeamSlices, seed: int = 0) -> FELResult:
    return eqx.filter_jit(plan.solve)(slices, jax.random.key(seed))


def _window_rate(power: np.ndarray, positions: np.ndarray) -> float:
    """Power e-folding rate between 1e-5 and 1e-2 of the peak."""
    first = int(np.searchsorted(power, 1.0e-5 * power.max()))
    last = int(np.searchsorted(power, 1.0e-2 * power.max()))
    return float(
        (np.log(power[last]) - np.log(power[first]))
        / (positions[last] - positions[first])
    )


@pytest.fixture(scope="module")
def cold_lattice() -> FELUndulatorLattice:
    return _lattice((_peak_field(DEFLECTION),), periods=1200)


@pytest.fixture(scope="module")
def wavelength(cold_lattice: FELUndulatorLattice) -> float:
    return cold_lattice.resonant_wavelength(GAMMA)


@pytest.fixture(scope="module")
def cold_run(
    cold_lattice: FELUndulatorLattice, wavelength: float
) -> tuple[FELResult, FELScalingEstimate]:
    plan = FELPlan(
        cold_lattice,
        wavelength,
        loading=FELLoading(32, 6, shot_noise="quiet"),
        harmonics=(1, 3),
        seed=FELSeed(1.0e3, waist=BEAM_SIZE),
    )
    slices = _cold_slices()
    return _solve(plan, slices), fel_scaling_estimate(cold_lattice, slices, wavelength)


def test_coupling_factors_match_bessel_closed_form() -> None:
    deflections = np.asarray([0.5, 1.0, 3.5])
    harmonics = (1, 3, 5)
    xi = deflections**2 / (4.0 + 2.0 * deflections**2)
    expected = np.stack(
        [
            (-1.0) ** ((h - 1) // 2)
            * (special.jv((h - 1) // 2, h * xi) - special.jv((h + 1) // 2, h * xi))
            for h in harmonics
        ],
        axis=-1,
    )
    planar = undulator_coupling_factors(deflections, harmonics, "planar")
    np.testing.assert_allclose(planar, expected, rtol=1.0e-12, atol=1.0e-14)
    np.testing.assert_array_equal(
        undulator_coupling_factors(deflections, (1,), "helical"), np.ones((3, 1))
    )


def test_one_dimensional_cold_growth_rate(
    cold_run: tuple[FELResult, FELScalingEstimate],
) -> None:
    result, scaling = cold_run
    rho = float(scaling.pierce_parameter[0])
    power = np.asarray(result.power[0, :, 0])
    amplitude_rate = 0.5 * _window_rate(power, np.asarray(result.positions))
    assert amplitude_rate == pytest.approx(math.sqrt(3.0) * rho * WAVENUMBER, rel=0.01)
    assert bool(result.gain.fit_valid[0])
    assert float(result.gain.fitted_gain_length[0]) == pytest.approx(
        PERIOD / (4.0 * math.pi * math.sqrt(3.0) * rho), rel=0.01
    )


def test_one_dimensional_saturation_power_scale(
    cold_run: tuple[FELResult, FELScalingEstimate],
) -> None:
    result, scaling = cold_run
    beam_power = GAMMA * MASS * LIGHT**2 * CURRENT / CHARGE
    assert float(scaling.beam_power[0]) == pytest.approx(beam_power, rel=1.0e-12)
    ratio = float(result.gain.saturation_power[0]) / (
        float(scaling.pierce_parameter[0]) * beam_power
    )
    assert 1.2 < ratio < 1.6


def test_energy_ledger_closes_particle_field_exchange(
    cold_run: tuple[FELResult, FELScalingEstimate],
) -> None:
    result, _ = cold_run
    ledger = result.ledger
    assert int(result.evidence.status[0]) == FELStatus.SUCCESS
    assert float(ledger.field_power_change[0]) > 1.0e9
    assert float(ledger.beam_power_change[0]) < 0.0
    assert float(ledger.relative_defect[0]) < 1.0e-7
    assert float(ledger.defect[0]) == pytest.approx(
        float(ledger.beam_power_change[0] + ledger.field_power_change[0]), abs=1.0
    )


def test_nonlinear_third_harmonic_grows_three_times_faster(
    cold_run: tuple[FELResult, FELScalingEstimate],
) -> None:
    result, _ = cold_run
    positions = np.asarray(result.positions)
    fundamental = np.asarray(result.power[0, :, 0])
    third = np.asarray(result.power[0, :, 1])
    first = int(np.searchsorted(fundamental, 1.0e-5 * fundamental.max()))
    last = int(np.searchsorted(fundamental, 1.0e-2 * fundamental.max()))
    span = positions[last] - positions[first]
    rate_first = np.log(fundamental[last] / fundamental[first]) / span
    rate_third = np.log(third[last] / third[first]) / span
    assert rate_third / rate_first == pytest.approx(3.0, rel=0.05)


def test_slice_spectrum_lines_sit_at_harmonic_frequencies(
    cold_run: tuple[FELResult, FELScalingEstimate], wavelength: float
) -> None:
    result, _ = cold_run
    np.testing.assert_allclose(
        np.asarray(result.angular_frequencies),
        2.0 * math.pi * LIGHT / wavelength * np.asarray([1.0, 3.0]),
        rtol=1.0e-12,
    )
    assert np.asarray(result.exit_field).shape == (1, 2)


def test_energy_spread_growth_matches_vlasov_dispersion(
    wavelength: float,
) -> None:
    lattice = _lattice((_peak_field(DEFLECTION),), periods=1600)
    slices = _cold_slices()
    rho = float(fel_scaling_estimate(lattice, slices, wavelength).pierce_parameter[0])
    normalized_spread = 0.5
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / math.sqrt(2.0 * math.pi)
    root = complex(np.exp(1j * math.pi / 6.0))
    for _ in range(60):
        denominator = root + 1j * normalized_spread * nodes
        residual = root - 1j * np.sum(weights / denominator**2)
        root -= residual / (1.0 + 2j * np.sum(weights / denominator**3))
    expected = 2.0 * rho * WAVENUMBER * root.real
    plan = FELPlan(
        lattice,
        wavelength,
        loading=FELLoading(2048, 4, shot_noise="quiet"),
        seed=FELSeed(1.0e3, waist=BEAM_SIZE),
    )
    result = _solve(plan, _cold_slices(normalized_spread * rho))
    power = np.asarray(result.power[0, :, 0])
    rate = 0.5 * _window_rate(power, np.asarray(result.positions))
    assert expected < 0.85 * math.sqrt(3.0) * rho * WAVENUMBER
    assert rate == pytest.approx(expected, rel=0.03)


def test_ming_xie_gain_length_and_far_field_power() -> None:
    helical = DEFLECTION / math.sqrt(2.0)
    beta = 9.0
    natural = (helical * WAVENUMBER) ** 2 / (2.0 * GAMMA**2)
    gradient = (1.0 / beta**2 - natural) * GAMMA * MASS * LIGHT / CHARGE
    lattice = _lattice(
        (_peak_field(helical),),
        periods=1200,
        polarization="helical",
        smooth_focusing_gradient=gradient,
    )
    wavelength = lattice.resonant_wavelength(GAMMA)
    emittance = BEAM_SIZE**2 / beta * GAMMA
    slices = FELBeamSlices(
        [0.0], [CURRENT], [GAMMA], [1.0e-4], [[emittance] * 2], [[beta] * 2], [[0.0] * 2]
    )
    grid = TensorGridPlan(
        (UniformAxisSpec(64), UniformAxisSpec(64)), axis_names=("x", "y")
    ).prepare(jnp.asarray([[-240.0e-6, -240.0e-6], [240.0e-6, 240.0e-6]]))
    plan = FELPlan(
        lattice,
        wavelength,
        loading=FELLoading(2048, 4, shot_noise="quiet"),
        transverse="angular-spectrum",
        field_space=PlaneFieldSpace(grid, RigidFrame.identity(3), "finite-window"),
        propagation=AngularSpectrumPlan(16),
        seed=FELSeed(1.0e3, waist=40.0e-6),
    )
    result = _solve(plan, slices)
    scaling = result.gain.scaling
    ming_xie = float(scaling.ming_xie_gain_length[0])
    fitted = float(result.gain.fitted_gain_length[0])
    assert int(result.evidence.status[0]) == FELStatus.SUCCESS
    assert ming_xie > 1.1 * float(scaling.one_dimensional_gain_length[0])
    assert fitted == pytest.approx(ming_xie, rel=0.12)
    assert float(result.ledger.relative_defect[0]) < 1.0e-7
    ratio = float(result.gain.saturation_power[0]) / (
        float(scaling.pierce_parameter[0]) * float(scaling.beam_power[0])
    )
    assert 0.3 < ratio < 1.6
    assert result.far_field_angles is not None
    assert result.far_field_intensity is not None
    angles_x, angles_y = result.far_field_angles[0]
    solid_angle = float((angles_x[1] - angles_x[0]) * (angles_y[1] - angles_y[0]))
    radiated = float(jnp.sum(result.far_field_intensity[0, :, :, 0])) * solid_angle
    assert radiated == pytest.approx(float(result.power[0, -1, 0]), rel=1.0e-9)


def test_quiet_start_and_fawley_shot_noise(wavelength: float) -> None:
    count = 256
    slices = FELBeamSlices(
        np.arange(count) * wavelength,
        np.full(count, CURRENT),
        np.full(count, GAMMA),
        np.full(count, 1.0e-4),
        np.full((count, 2), 1.0e-6),
        np.full((count, 2), 10.0),
        np.zeros((count, 2)),
    )
    lattice = _lattice((_peak_field(DEFLECTION),), periods=1, step=PERIOD)
    electrons = CURRENT * wavelength / (CHARGE * LIGHT)
    for noise in ("quiet", "fawley"):
        plan = FELPlan(
            lattice,
            wavelength,
            loading=FELLoading(64, 6, shot_noise=noise),
            harmonics=(1, 3),
            slice_batch=16,
        )
        result = _solve(plan, slices, seed=3)
        normalized = np.mean(np.abs(np.asarray(result.bunching[:, 0, :])) ** 2, axis=0)
        normalized = normalized * electrons
        assert np.all(np.asarray(result.evidence.status) == FELStatus.SUCCESS)
        if noise == "quiet":
            assert np.all(normalized < 1.0e-20)
        else:
            # 256 exponential samples: 1σ of the mean is 1/16.
            np.testing.assert_allclose(normalized, 1.0, rtol=0.25)


def test_identity_addressed_loading_is_order_and_batch_invariant(
    wavelength: float,
) -> None:
    count = 12
    order = np.random.default_rng(7).permutation(count)

    def slices(identities: np.ndarray) -> FELBeamSlices:
        return FELBeamSlices(
            identities * wavelength,
            CURRENT * (1.0 + 0.01 * identities),
            np.full(count, GAMMA),
            np.full(count, 1.0e-4),
            np.full((count, 2), 1.0e-6),
            np.full((count, 2), 10.0),
            np.zeros((count, 2)),
            identities=identities,
        )

    lattice = _lattice((_peak_field(DEFLECTION),), periods=20, step=0.1)
    loading = FELLoading(16, 4, shot_noise="fawley")
    reference = _solve(
        FELPlan(lattice, wavelength, loading=loading, slice_batch=4),
        slices(np.arange(count)),
    )
    permuted = _solve(
        FELPlan(lattice, wavelength, loading=loading, slice_batch=5), slices(order)
    )
    np.testing.assert_allclose(
        np.asarray(permuted.bunching), np.asarray(reference.bunching)[order], rtol=1.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(permuted.particles.phases),
        np.asarray(reference.particles.phases)[order],
        rtol=1.0e-12,
    )


def test_taper_extends_growth_past_saturation(wavelength: float) -> None:
    def run(taper: float) -> FELResult:
        peaks = tuple(
            _peak_field(DEFLECTION) * (1.0 - taper * max(0, index - 4))
            for index in range(8)
        )
        plan = FELPlan(
            _lattice(peaks, periods=150),
            wavelength,
            loading=FELLoading(64, 4, shot_noise="quiet"),
            seed=FELSeed(1.0e3, waist=BEAM_SIZE),
        )
        return _solve(plan, _cold_slices())

    untapered = run(0.0)
    tapered = run(2.0e-3)
    assert float(tapered.power[0, -1, 0]) > 3.0 * float(jnp.max(untapered.power[0, :, 0]))
    assert float(tapered.ledger.relative_defect[0]) < 1.0e-7
    assert float(tapered.mean_lorentz_factor[0, -1]) < float(
        untapered.mean_lorentz_factor[0, -1]
    )


def test_betatron_transfer_natural_focusing_and_thin_quadrupole(
    wavelength: float,
) -> None:
    periods = 300
    drift = 1.0
    integrated_gradient = 0.5
    lattice = _lattice(
        (_peak_field(DEFLECTION),),
        periods=periods,
        drift_length=drift,
        quadrupole_integrated_gradient=integrated_gradient,
    )
    slices = FELBeamSlices(
        [0.0], [1.0e-6], [GAMMA], [0.0], [[1.0e-6] * 2], [[10.0] * 2], [[0.0] * 2]
    )
    result = _solve(
        FELPlan(lattice, wavelength, loading=FELLoading(16, 2, shot_noise="quiet")),
        slices,
    )
    start = np.asarray(result.initial_particles.positions[0])
    start_momenta = np.asarray(result.initial_particles.transverse_momenta[0])
    gamma = np.asarray(result.initial_particles.lorentz_factors[0])
    length = periods * PERIOD
    # Planar natural focusing acts in y only: k_y = a_w k_u / γ, a_w = K/√2.
    focusing = DEFLECTION / math.sqrt(2.0) * WAVENUMBER / gamma
    y = start[:, 1] * np.cos(focusing * length) + start_momenta[:, 1] / (
        gamma * focusing
    ) * np.sin(focusing * length)
    uy = -gamma * focusing * start[:, 1] * np.sin(focusing * length) + start_momenta[
        :, 1
    ] * np.cos(focusing * length)
    x = start[:, 0] + start_momenta[:, 0] / gamma * length
    kick = CHARGE * integrated_gradient / (MASS * LIGHT)
    ux = start_momenta[:, 0] - kick * x
    uy = uy + kick * y
    expected_positions = np.stack((x + ux / gamma * drift, y + uy / gamma * drift), -1)
    np.testing.assert_allclose(
        np.asarray(result.particles.positions[0]),
        expected_positions,
        rtol=1.0e-10,
        atol=1.0e-12 * np.max(np.abs(expected_positions)),
    )
    np.testing.assert_allclose(
        np.asarray(result.particles.transverse_momenta[0]),
        np.stack((ux, uy), -1),
        rtol=1.0e-10,
        atol=1.0e-12 * np.max(np.abs(ux)),
    )


def test_per_slice_wake_energy_loss(wavelength: float) -> None:
    count = 8
    spacing = 1.0e-6
    wake_value = 1.0e13
    structure = 2.0
    wake = WakeFunctionPlan(
        "longitudinal",
        np.asarray([0.0, 1.0]),
        np.asarray([wake_value, wake_value]),
        units="V/C",
        causality_convention="behind-positive",
    )
    lattice = _lattice((_peak_field(DEFLECTION),), periods=100)
    slices = FELBeamSlices(
        np.arange(count) * spacing,
        np.full(count, CURRENT),
        np.full(count, GAMMA),
        np.zeros(count),
        np.full((count, 2), 1.0e-6),
        np.full((count, 2), 10.0),
        np.zeros((count, 2)),
    )
    plan = FELPlan(
        lattice,
        wavelength,
        loading=FELLoading(8, 2, shot_noise="quiet"),
        wake=FELWakeLoss(wake, structure_length=structure),
        slice_batch=4,
    )
    result = _solve(plan, slices)
    charge = CURRENT * spacing / LIGHT
    potential = wake_value * charge * (np.arange(count) + 0.5)
    expected = -CHARGE * potential * lattice.length / (MASS * LIGHT**2 * structure)
    change = np.asarray(
        result.mean_lorentz_factor[:, -1] - result.mean_lorentz_factor[:, 0]
    )
    np.testing.assert_allclose(change, expected, rtol=1.0e-5)
    np.testing.assert_allclose(
        np.asarray(result.ledger.wake_loss),
        CURRENT * potential * lattice.length / structure,
        rtol=1.0e-12,
    )
    assert np.all(np.asarray(result.ledger.relative_defect) < 1.0e-6)


@pytest.mark.parametrize(
    ("polarization", "harmonics", "particles", "message"),
    [
        ("planar", (1, 2), 4, "Even planar"),
        ("helical", (1, 3), 6, "only its fundamental"),
        ("planar", (1, 3), 4, "particles_per_beamlet >= 2"),
    ],
    ids=["even-planar-harmonic", "helical-harmonic", "too-few-quiet-particles"],
)
def test_unsupported_harmonic_configurations_are_refused(
    polarization: str, harmonics: tuple[int, ...], particles: int, message: str
) -> None:
    lattice = _lattice((_peak_field(1.0),), periods=10, polarization=polarization)
    with pytest.raises(ValueError, match=message):
        FELPlan(
            lattice,
            1.0e-9,
            loading=FELLoading(4, particles, shot_noise="quiet"),
            harmonics=harmonics,
        )


def test_configuration_refusals() -> None:
    planar = InsertionDeviceField(
        _peak_field(1.0),
        PERIOD,
        10,
        polarization="planar",
        center=0.0,
        aperture=(1e-2, 1e-2),
    )
    helical = InsertionDeviceField(
        _peak_field(1.0),
        PERIOD,
        10,
        polarization="helical",
        center=0.0,
        aperture=(1e-2, 1e-2),
    )
    with pytest.raises(ValueError, match="share one polarization"):
        FELUndulatorLattice(
            SCALE,
            (FELUndulatorSegment(planar), FELUndulatorSegment(helical)),
            step_length=0.1,
        )
    with pytest.raises(ValueError, match="positive drift_length"):
        FELUndulatorSegment(planar, quadrupole_integrated_gradient=1.0)
    lattice = FELUndulatorLattice(SCALE, (FELUndulatorSegment(planar),), step_length=0.1)
    loading = FELLoading(4, 2, shot_noise="quiet")
    with pytest.raises(TypeError, match="requires a PlaneFieldSpace"):
        FELPlan(lattice, 1.0e-9, loading=loading, transverse="angular-spectrum")
    grid = TensorGridPlan(
        (UniformAxisSpec(8), UniformAxisSpec(8)), axis_names=("x", "y")
    ).prepare(jnp.asarray([[-1.0e-4, -1.0e-4], [1.0e-4, 1.0e-4]]))
    with pytest.raises(ValueError, match="takes no field_space"):
        FELPlan(
            lattice,
            1.0e-9,
            loading=loading,
            field_space=PlaneFieldSpace(grid, RigidFrame.identity(3), "finite-window"),
        )
    with pytest.raises(ValueError, match="longitudinal wakes only"):
        FELWakeLoss(
            WakeFunctionPlan(
                "dipolar-x",
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, 1.0]),
                units="V/C/m",
                causality_convention="behind-positive",
            ),
            structure_length=1.0,
        )
    wake = FELWakeLoss(
        WakeFunctionPlan(
            "longitudinal",
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, 1.0]),
            units="V/C",
            causality_convention="behind-positive",
        ),
        structure_length=1.0,
    )
    irregular = FELBeamSlices(
        [0.0, 1.0e-6, 3.0e-6],
        [1.0, 1.0, 1.0],
        [GAMMA] * 3,
        [0.0] * 3,
        [[1.0e-6] * 2] * 3,
        [[10.0] * 2] * 3,
        [[0.0] * 2] * 3,
    )
    with pytest.raises(ValueError, match="uniformly spaced"):
        FELPlan(lattice, 1.0e-9, loading=loading, wake=wake).solve(
            irregular, jax.random.key(0)
        )
    with pytest.raises(ValueError, match="currents must be positive"):
        FELBeamSlices(
            [0.0], [0.0], [GAMMA], [0.0], [[1e-6] * 2], [[10.0] * 2], [[0.0] * 2]
        )
