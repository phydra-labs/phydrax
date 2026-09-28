#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-dependent averaged FEL against independent references.

References: the time-independent solver on identical slices (a uniform,
infinitely long beam is a periodic window of identical slices); exact
translation of sampled and band-limited fields; the linearized 1-D FEL
equations with slippage, ``∂_z a = i s q a + C(b − P)``, ``∂_z b = −2ik_u P``,
``∂_z P = D a`` per window Fourier mode ``q`` (Bonifacio, Pellegrini &
Narducci 1984; Saldin, Schneidmiller & Yurkov, *The Physics of Free Electron
Lasers*, 2000), integrated by SciPy's matrix exponential; the SASE pulse-energy
gamma distribution with ``M = (Σ G_q)² / Σ G_q²`` modes for independent
Poisson start-up bunching (Saldin et al. 2000, §6); the HGHG bunching
``|J_h(hAB)| e^{−h²B²/2}`` (Yu, PRA 44, 5178, 1991) and the EEHG sum
``Σ_m J_m(hA₂B₂) J_{−(h+m)}(A₁Y_m) e^{−Y_m²/2}``, ``Y_m = (h+m)B₁ + hB₂``
(Xiang & Stupakov, PRST-AB 12, 030702, 2009), derived for the documented
``ψ ← ψ + B p`` map; the closed-form potential of a constant wake; the
longitudinal field of a Gaussian bunch from its one-dimensional integral
representation; and a pinned Genesis 1.3 version 4 run when available.
"""

from __future__ import annotations

import math
import os
import shutil
import typing
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from scipy import constants, integrate, linalg, special

from phydrax import ElectromagneticScaleContract
from phydrax._external_runtime import pin_executable
from phydrax.applications.accelerator import (
    AcceleratorConvention,
    InsertionDeviceField,
    SpaceChargeIGFPlan,
    SymplecticMapPlan,
    WakeFunctionPlan,
)
from phydrax.applications.accelerator.fel import (
    FELBeamSlices,
    FELLoading,
    FELModulator,
    FELParticles,
    FELPlan,
    FELPrebunching,
    FELPulseSeed,
    FELSeed,
    FELSlippageRoute,
    FELSpaceCharge,
    FELStatus,
    FELTimeDependentPlan,
    FELTimeDependentResult,
    FELTransverseSpaceCharge,
    FELUndulatorLattice,
    FELUndulatorSegment,
    FELWakeLoss,
    Genesis4GaussianSeed,
    run_genesis4,
)
from phydrax.discretization import TensorGridPlan, UniformAxisSpec, UniformCellAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import (
    AngularSpectrumPlan,
    PlaneFieldSpace,
    PulseEnvelopeField,
    PulseTimeSpace,
)


SCALE = ElectromagneticScaleContract.si()
LIGHT = constants.c
CHARGE = constants.e
MASS = constants.m_e
PERMITTIVITY = constants.epsilon_0
PERIOD = 0.03
UNDULATOR_WAVENUMBER = 2.0 * math.pi / PERIOD
GAMMA = 1.0e4
DEFLECTION = 3.5 / math.sqrt(2.0)
CURRENT = 3000.0
# Natural helical focusing k_β = K k_u / (√2 γ) matches this beta function.
BETA = math.sqrt(2.0) * GAMMA / (DEFLECTION * UNDULATOR_WAVENUMBER)
PLANE = PlaneFieldSpace(
    TensorGridPlan(
        (UniformAxisSpec(2), UniformAxisSpec(2)), axis_names=("x", "y")
    ).prepare(jnp.asarray([[-1.0, -1.0], [1.0, 1.0]])),
    RigidFrame.identity(3),
    "finite-window",
)


def _lattice(
    periods: int,
    *,
    step: float = 2.0 * PERIOD,
    drift_length: float = 0.0,
) -> FELUndulatorLattice:
    peak = DEFLECTION * 2.0 * math.pi * MASS * LIGHT / (CHARGE * PERIOD)
    device = InsertionDeviceField(
        peak, PERIOD, periods, polarization="helical", center=0.0, aperture=(1e-2, 1e-2)
    )
    segment = FELUndulatorSegment(device, drift_length=drift_length)
    return FELUndulatorLattice(SCALE, (segment,), step_length=step)


def _helical_lattice(
    deflection: float,
    period: float,
    periods: int,
    *,
    step: float | None = None,
    smooth_focusing: float = 0.0,
) -> FELUndulatorLattice:
    """One helical module (``a_w = K``) stepped per period unless ``step`` is given."""
    peak = deflection * 2.0 * math.pi * MASS * LIGHT / (CHARGE * period)
    device = InsertionDeviceField(
        peak, period, periods, polarization="helical", center=0.0, aperture=(1e-3, 1e-3)
    )
    segment = FELUndulatorSegment(device, smooth_focusing_gradient=smooth_focusing)
    return FELUndulatorLattice(
        SCALE, (segment,), step_length=period if step is None else step
    )


def _slices(
    count: int,
    spacing: float,
    *,
    current: float,
    emittance: float,
    spread: float = 0.0,
    gamma: float = GAMMA,
    beta: float = BETA,
) -> FELBeamSlices:
    return FELBeamSlices(
        np.arange(count) * spacing,
        np.full(count, current),
        np.full(count, gamma),
        np.full(count, spread),
        np.full((count, 2), emittance),
        np.full((count, 2), beta),
        np.zeros((count, 2)),
    )


def _plane_wave_seed(
    values: np.ndarray, spacing: float, carrier: float, *, reference: float = 0.0
) -> FELPulseSeed:
    count = values.shape[0]
    step = spacing / LIGHT
    time = PulseTimeSpace(
        TensorGridPlan((UniformAxisSpec(count),), axis_names=("t",)).prepare(
            jnp.asarray([[0.0], [(count - 1) * step]])
        ),
        topology="finite-window",
    )
    field = np.broadcast_to(values, (2, 2, count)).astype(np.complex128)
    return FELPulseSeed(
        PulseEnvelopeField(PLANE, time, field, carrier, 0.0), reference_position=reference
    )


def _solve(
    plan: FELTimeDependentPlan, slices: FELBeamSlices, seed: int = 0
) -> FELTimeDependentResult:
    return eqx.filter_jit(plan.solve)(slices, jax.random.key(seed))


def _linear_matrix(
    lattice: FELUndulatorLattice,
    wavelength: float,
    current: float,
    area: float,
    mode: float,
) -> np.ndarray:
    """Cold 1-D linear FEL generator for ``(a, b, P)`` at window wavenumber ``q``."""
    coupling = lattice.fundamental_couplings[0]
    source = coupling * current / (PERMITTIVITY * LIGHT * area * GAMMA)
    drive = -CHARGE * coupling / (2.0 * MASS * LIGHT**2 * GAMMA**2)
    slip = wavelength / PERIOD
    return np.asarray(
        [
            [1j * slip * mode, source, -source],
            [0.0, 0.0, -2j * UNDULATOR_WAVENUMBER],
            [drive, 0.0, 0.0],
        ]
    )


def _propagator(generator: np.ndarray) -> np.ndarray:
    matrix: npt.NDArray[np.complex128] = np.asarray(generator, dtype=np.complex128)
    return linalg.expm(matrix)


def _frequency(wavelength: float) -> float:
    return 2.0 * math.pi * LIGHT / wavelength


# --------------------------------------------------------------- reduction


@pytest.mark.parametrize("route", ["commensurate", "spectral"])
def test_uniform_periodic_beam_reduces_to_time_independent_solver(
    route: FELSlippageRoute,
) -> None:
    # A cold matched beam (tiny emittance, current scaled with the area) makes
    # identical slices; a periodic window of them is an infinitely long beam.
    # Into the high-gain regime, before saturation amplifies slice roundoff.
    lattice = _lattice(500)
    wavelength = lattice.resonant_wavelength(GAMMA)
    emittance = 1.0e-12
    area_ratio = emittance * BETA / GAMMA / (30.0e-6) ** 2
    # Two-period steps slip 2λ: one slot of the 2λ slice spacing.
    slices = _slices(
        8, 2.0 * wavelength, current=CURRENT * area_ratio, emittance=emittance
    )
    core = FELPlan(
        lattice,
        wavelength,
        loading=FELLoading(4, 8, shot_noise="quiet"),
        seed=FELSeed(1.0e3 * area_ratio, waist=30.0e-6),
        slice_batch=8,
    )
    reference = eqx.filter_jit(core.solve)(slices, jax.random.key(0))
    result = _solve(
        FELTimeDependentPlan(core, slippage=route, boundary="periodic"), slices
    )
    expected = np.asarray(reference.power[:, :, 0])
    power = np.asarray(result.power[:, :, 0]).T
    assert expected[0, -1] / expected[0, 0] > 1.0e5
    scale = expected.max(axis=0)
    np.testing.assert_array_less(np.abs(power - expected) / scale, 1.0e-5)
    np.testing.assert_allclose(
        np.asarray(result.bunching[:, :, 0]).T,
        np.asarray(reference.bunching[:, :, 0]),
        atol=1.0e-5,
    )
    assert int(result.evidence.status) == FELStatus.SUCCESS
    assert float(result.ledger.relative_defect) < 1.0e-6


# ---------------------------------------------------------- exact slippage


def test_commensurate_slippage_is_an_exact_roll() -> None:
    lattice = _lattice(10, step=PERIOD)
    wavelength = lattice.resonant_wavelength(GAMMA)
    count = 32
    rng = np.random.default_rng(0)
    values = 1.0e6 * (rng.normal(size=count) + 1j * rng.normal(size=count))
    plan = FELTimeDependentPlan(
        FELPlan(lattice, wavelength, loading=FELLoading(2, 4, shot_noise="quiet")),
        slippage="commensurate",
        boundary="periodic",
        pulse_seed=_plane_wave_seed(values, wavelength, _frequency(wavelength)),
    )
    result = _solve(plan, _slices(count, wavelength, current=1.0e-9, emittance=1e-7))
    # Ten periods slip ten wavelengths: slot j receives the seed of slot j + 10.
    np.testing.assert_allclose(
        np.asarray(result.exit_field[:, 0]), np.roll(values, -10), rtol=1.0e-9
    )
    assert float(result.evidence.slippage_slots) == pytest.approx(10.0, rel=1e-12)


def test_spectral_slippage_is_an_exact_band_limited_translation() -> None:
    drift = 0.37
    lattice = _lattice(10, step=PERIOD, drift_length=drift)
    wavelength = lattice.resonant_wavelength(GAMMA)
    count = 32
    length = count * wavelength
    positions = np.arange(count) * wavelength
    modes = {-3: 1.0 + 0.5j, 1: 2.0, 2: -0.7j, 5: 0.3}

    def field(coordinate: np.ndarray) -> np.ndarray:
        return 1.0e6 * sum(
            amplitude * np.exp(2j * np.pi * mode * coordinate / length)
            for mode, amplitude in modes.items()
        )

    core = FELPlan(lattice, wavelength, loading=FELLoading(2, 4, shot_noise="quiet"))
    seed = _plane_wave_seed(field(positions), wavelength, _frequency(wavelength))
    plan = FELTimeDependentPlan(
        core, slippage="spectral", boundary="periodic", pulse_seed=seed
    )
    result = _solve(plan, _slices(count, wavelength, current=1.0e-9, emittance=1e-7))
    # One wavelength per period plus the break's d/(2γ_r²), 2γ_r² = λ_u(1+K²)/λ.
    slip = 10.0 * wavelength + drift * wavelength / (PERIOD * (1.0 + DEFLECTION**2))
    np.testing.assert_allclose(
        np.asarray(result.exit_field[:, 0]), field(positions + slip), rtol=1.0e-9
    )
    assert float(result.evidence.total_slippage) == pytest.approx(slip, rel=1e-12)
    with pytest.raises(ValueError, match="integer number of slice spacings"):
        FELTimeDependentPlan(
            core, slippage="commensurate", boundary="periodic", pulse_seed=seed
        ).solve(
            _slices(count, wavelength, current=1.0e-9, emittance=1e-7), jax.random.key(0)
        )


def test_exit_spectrum_places_a_detuned_seed_at_its_carrier() -> None:
    lattice = _lattice(4, step=PERIOD)
    wavelength = lattice.resonant_wavelength(GAMMA)
    count = 64
    slot_time = wavelength / LIGHT
    detuning = 2.0 * math.pi * 3.0 / (count * slot_time)
    carrier = _frequency(wavelength) + detuning
    plan = FELTimeDependentPlan(
        FELPlan(lattice, wavelength, loading=FELLoading(2, 4, shot_noise="quiet")),
        slippage="commensurate",
        boundary="periodic",
        pulse_seed=_plane_wave_seed(np.full(count, 1.0e6), wavelength, carrier),
    )
    result = _solve(plan, _slices(count, wavelength, current=1.0e-9, emittance=1e-7))
    spectrum = result.spectrum
    frequencies = np.asarray(spectrum.angular_frequencies[:, 0])
    energy = np.asarray(spectrum.spectral_energy[:, 0])
    assert frequencies[np.argmax(energy)] == pytest.approx(carrier, rel=1e-12)
    spacing = frequencies[1] - frequencies[0]
    assert np.sum(energy) * spacing == pytest.approx(
        float(result.pulse_energy[-1, 0]), rel=1e-10
    )
    # A single mode is coherent over the whole periodic window.
    assert float(spectrum.coherence_time[0]) == pytest.approx(count * slot_time, rel=1e-9)
    assert float(spectrum.rms_bandwidth[0]) < 1.0e-6 * spacing


# ---------------------------------------------------------- window padding


@pytest.mark.parametrize("route", ["commensurate", "spectral"])
def test_open_window_padding_evidence_and_exit_ledger(route: FELSlippageRoute) -> None:
    lattice = _lattice(40, step=PERIOD)
    wavelength = lattice.resonant_wavelength(GAMMA)
    count = 48
    slots = np.arange(count)
    pulse = 1.0e6 * np.exp(-(((slots - 36.0) / 4.0) ** 2))
    slices = _slices(count, wavelength, current=1.0e-9, emittance=1e-7)
    core = FELPlan(lattice, wavelength, loading=FELLoading(2, 4, shot_noise="quiet"))
    seed = _plane_wave_seed(pulse, wavelength, _frequency(wavelength))

    def run(padding: int) -> FELTimeDependentResult:
        return _solve(
            FELTimeDependentPlan(
                core,
                slippage=route,
                boundary="open",
                head_padding=padding,
                pulse_seed=seed,
            ),
            slices,
        )

    # 40 slots of slip carry the pulse center from slot 36 past the head.
    truncated = run(0)
    evidence = truncated.evidence
    assert not bool(evidence.padding_sufficient)
    assert float(evidence.exit_fraction) > 0.4
    assert int(evidence.status) & FELStatus.WINDOW_TRUNCATED
    initial, final = np.asarray(truncated.pulse_energy[[0, -1], 0])
    assert float(truncated.ledger.exit_energy) == pytest.approx(initial - final, rel=1e-6)
    assert float(truncated.ledger.relative_defect) < 1.0e-6
    padded = run(40)
    assert bool(padded.evidence.padding_sufficient)
    assert float(padded.evidence.exit_fraction) < 1.0e-3
    assert int(padded.evidence.status) == FELStatus.SUCCESS
    assert int(np.argmax(np.asarray(padded.power[-1, :, 0]))) == 36
    assert float(padded.pulse_energy[-1, 0]) == pytest.approx(initial, rel=1e-3)


# ---------------------------------------------------- seeded amplification


def test_seeded_amplification_spectrum_matches_linear_theory() -> None:
    lattice = _lattice(200, step=PERIOD)
    wavelength = lattice.resonant_wavelength(GAMMA)
    spacing = wavelength
    count = 128
    emittance = 1.0e-10
    current = CURRENT * emittance / 1.0e-7
    slices = _slices(count, spacing, current=current, emittance=emittance)
    slot_time = spacing / LIGHT
    times = np.arange(count) * slot_time
    detuning = 1.0e-3 * _frequency(wavelength)
    envelope = 1.0e6 * np.exp(-(((times - 64.0 * slot_time) / (12.0 * slot_time)) ** 2))
    plan = FELTimeDependentPlan(
        FELPlan(lattice, wavelength, loading=FELLoading(4, 4, shot_noise="quiet")),
        slippage="commensurate",
        boundary="periodic",
        pulse_seed=_plane_wave_seed(envelope, spacing, _frequency(wavelength) + detuning),
    )
    result = _solve(plan, slices)
    area = float(result.scaling.beam_area[0])
    initial = np.fft.fft(envelope * np.exp(-1j * detuning * times))
    final = np.fft.fft(np.asarray(result.exit_field[:, 0]))
    modes = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    transfer = np.asarray(
        [
            _propagator(
                _linear_matrix(lattice, wavelength, current, area, mode) * lattice.length
            )[0, 0]
            for mode in modes
        ]
    )
    significant = np.abs(initial) > 1.0e-3 * np.abs(initial).max()
    assert np.abs(transfer[significant]).max() > 20.0
    error = np.abs(final[significant] / initial[significant] - transfer[significant])
    np.testing.assert_array_less(error / np.abs(transfer[significant]), 5.0e-3)
    assert int(result.evidence.status) == FELStatus.SUCCESS
    assert float(result.ledger.relative_defect) < 1.0e-6


# ------------------------------------------------------------------- SASE


def _sase(count: int) -> tuple[FELTimeDependentResult, np.ndarray]:
    lattice = _lattice(200)
    wavelength = lattice.resonant_wavelength(GAMMA)
    spacing = 4.0 * wavelength
    slices = _slices(count, spacing, current=CURRENT, emittance=1.0e-7)
    plan = FELTimeDependentPlan(
        FELPlan(
            lattice,
            wavelength,
            loading=FELLoading(1, 4, shot_noise="fawley"),
            slice_batch=256,
        ),
        slippage="spectral",
        boundary="periodic",
    )
    keys = jax.random.split(jax.random.key(11), 512)
    result = eqx.filter_jit(jax.vmap(plan.solve, in_axes=(None, 0)))(slices, keys)
    area = float(result.scaling.beam_area[0, 0])
    modes = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    gain = np.asarray(
        [
            abs(
                _propagator(
                    _linear_matrix(lattice, wavelength, CURRENT, area, mode)
                    * lattice.length
                )[0, 1]
            )
            ** 2
            for mode in modes
        ]
    )
    return result, gain


def test_sase_pulse_energy_statistics_and_coherence_time() -> None:
    short, gain = _sase(96)
    long, long_gain = _sase(192)
    for result in (short, long):
        assert np.all(np.asarray(result.evidence.status) == FELStatus.SUCCESS)
    energy = np.asarray(short.pulse_energy[:, -1, 0])
    modes = gain.sum() ** 2 / np.sum(gain**2)
    # Gamma statistics: M = ⟨W⟩²/σ_W²; 512 shots give ≈ 8 % sampling error.
    assert energy.mean() ** 2 / energy.var() == pytest.approx(modes, rel=0.2)
    slot_time = float(short.window_positions[0, 1] - short.window_positions[0, 0]) / LIGHT
    for result, curve in ((short, gain), (long, long_gain)):
        spectrum = np.asarray(result.spectrum.spectral_energy[:, :, 0]).mean(axis=0)
        count = spectrum.shape[0]
        measured = slot_time * count * np.sum(spectrum**2) / np.sum(spectrum) ** 2
        expected = slot_time * count * np.sum(curve**2) / np.sum(curve) ** 2
        assert measured == pytest.approx(expected, rel=0.05)
    # Spikes scale with the window length over the coherence time.
    ratio = (
        np.asarray(long.spectrum.spike_count[:, 0]).mean()
        / np.asarray(short.spectrum.spike_count[:, 0]).mean()
    )
    assert ratio == pytest.approx(2.0, rel=0.15)


# ------------------------------------------------------ HGHG / EEHG stages


def _chicane(r56: float) -> SymplecticMapPlan:
    matrix = np.eye(6)
    matrix[4, 5] = r56
    return SymplecticMapPlan(
        matrix, np.zeros(6), AcceleratorConvention(), element_id="chicane"
    )


def _prebunched(
    stages: tuple[FELModulator | SymplecticMapPlan, ...], harmonic: int
) -> FELTimeDependentResult:
    lattice = _lattice(2)
    wavelength = lattice.resonant_wavelength(GAMMA)
    plan = FELTimeDependentPlan(
        FELPlan(
            lattice,
            wavelength,
            loading=FELLoading(4096, 4 * harmonic, shot_noise="quiet"),
            slice_batch=8,
        ),
        slippage="commensurate",
        boundary="periodic",
        prebunching=FELPrebunching(
            stages, harmonic=harmonic, reference_lorentz_factor=GAMMA
        ),
    )
    slices = _slices(8, wavelength, current=1.0, emittance=1.0e-9, spread=1.0e-4)
    return _solve(plan, slices, seed=1)


def _dispersion(strength: float, harmonic: int, wavelength: float) -> float:
    """``R₅₆`` giving ``B = −k_b R₅₆ σ_δ`` for a positive-late map."""
    return -strength / (2.0 * math.pi / (harmonic * wavelength) * 1.0e-4)


def test_hghg_bunching_matches_bessel_formula() -> None:
    harmonic, amplitude, strength = 4, 4.0, 0.3
    wavelength = _lattice(2).resonant_wavelength(GAMMA)
    result = _prebunched(
        (
            FELModulator(amplitude * 1.0e-4 * GAMMA),
            _chicane(_dispersion(strength, harmonic, wavelength)),
        ),
        harmonic,
    )
    bunching = abs(complex(np.asarray(result.bunching[0, :, 0]).mean()))
    expected = float(
        np.abs(
            special.jv(np.float64(harmonic), np.float64(harmonic * amplitude * strength))
        )
    ) * math.exp(-0.5 * (harmonic * strength) ** 2)
    assert bunching == pytest.approx(expected, abs=5.0e-3)


def test_eehg_bunching_matches_echo_formula() -> None:
    harmonic, first, second, strong = 7, 3.0, 3.0, 6.0
    weak = 0.8125
    wavelength = _lattice(2).resonant_wavelength(GAMMA)
    spread = 1.0e-4 * GAMMA
    result = _prebunched(
        (
            FELModulator(first * spread),
            _chicane(_dispersion(strong, harmonic, wavelength)),
            FELModulator(second * spread),
            _chicane(_dispersion(weak, harmonic, wavelength)),
        ),
        harmonic,
    )
    orders = np.arange(-60, 61)
    shifted = (harmonic + orders) * strong + harmonic * weak
    expected = abs(
        np.sum(
            special.jv(orders, harmonic * second * weak)
            * special.jv(-(harmonic + orders), first * shifted)
            * np.exp(-0.5 * shifted**2)
        )
    )
    bunching = abs(complex(np.asarray(result.bunching[0, :, 0]).mean()))
    assert expected > 0.05
    assert bunching == pytest.approx(expected, abs=5.0e-3)
    assert float(np.max(result.evidence.maximum_prebunching_displacement)) > wavelength


# ------------------------------------------------------- collective effects


def test_per_slice_wake_energy_loss() -> None:
    count, spacing, wake_value, structure = 8, 1.0e-6, 1.0e13, 2.0
    lattice = _lattice(40)
    wavelength = lattice.resonant_wavelength(GAMMA)
    wake = WakeFunctionPlan(
        "longitudinal",
        np.asarray([0.0, 1.0]),
        np.asarray([wake_value, wake_value]),
        units="V/C",
        causality_convention="behind-positive",
    )
    plan = FELTimeDependentPlan(
        FELPlan(
            lattice,
            wavelength,
            loading=FELLoading(8, 2, shot_noise="quiet"),
            wake=FELWakeLoss(wake, structure_length=structure),
        ),
        slippage="spectral",
        boundary="open",
    )
    result = _solve(plan, _slices(count, spacing, current=CURRENT, emittance=1e-7))
    charge = CURRENT * spacing / LIGHT
    potential = wake_value * charge * (np.arange(count) + 0.5)
    expected = -CHARGE * potential * lattice.length / (MASS * LIGHT**2 * structure)
    change = np.asarray(result.mean_lorentz_factor[-1] - result.mean_lorentz_factor[0])
    np.testing.assert_allclose(change, expected, rtol=1.0e-5)
    assert float(result.ledger.wake_loss) == pytest.approx(
        np.sum(CURRENT * potential) * lattice.length / structure * spacing / LIGHT,
        rel=1e-12,
    )
    assert float(result.ledger.relative_defect) < 1.0e-6


def _gaussian_bunch_case(
    gamma: float,
    radius: float,
    rest_length: float,
    peak_current: float,
    lattice: FELUndulatorLattice,
    *,
    strength: float,
    count: int,
    beamlets: int,
    beta: float,
    transverse_cells: int,
    transverse: FELTransverseSpaceCharge,
    transverse_tolerance: float = 1.0e-2,
) -> tuple[FELTimeDependentPlan, FELBeamSlices, float]:
    """Gaussian current profile of rest-frame length ``rest_length``.

    ``strength`` is the first module's ``a_w²``, which sets the mean-motion
    frame ``γ_z = γ/√(1 + a_w²)``.
    """
    frame = gamma / math.sqrt(1.0 + strength)
    length = rest_length / frame
    wavelength = lattice.resonant_wavelength(gamma)
    spacing = 8.0 * length / count
    positions = (np.arange(count) - 0.5 * (count - 1)) * spacing
    currents = peak_current * np.exp(-0.5 * (positions / length) ** 2)
    slices = FELBeamSlices(
        positions,
        currents,
        np.full(count, gamma),
        np.zeros(count),
        np.full((count, 2), radius**2 * gamma / beta),
        np.full((count, 2), beta),
        np.zeros((count, 2)),
    )
    # Rest-frame grid of the mean-motion frame γ_z: one longitudinal cell per
    # slice, ±6σ transversely.
    # A guard cell at each end keeps the edge slices' assignment inside the grid.
    extent = 0.5 * frame * (count + 2) * spacing
    grid = TensorGridPlan(
        (
            UniformCellAxisSpec(transverse_cells),
            UniformCellAxisSpec(transverse_cells),
            UniformCellAxisSpec(count + 2),
        )
    ).prepare(
        np.asarray(
            [
                [-6.0 * radius, -6.0 * radius, -extent],
                [6.0 * radius, 6.0 * radius, extent],
            ]
        )
    )
    loading = FELLoading(beamlets, 4, shot_noise="quiet")
    plan = FELTimeDependentPlan(
        FELPlan(lattice, wavelength, loading=loading, slice_batch=count),
        slippage="spectral",
        boundary="open",
        space_charge=FELSpaceCharge(
            SpaceChargeIGFPlan(SCALE, grid, capacity=count * loading.particle_count),
            transverse=transverse,
            transverse_tolerance=transverse_tolerance,
        ),
    )
    return plan, slices, frame


def test_bunch_scale_space_charge_matches_gaussian_bunch_field_in_mean_motion_frame() -> (
    None
):
    # Inside the undulator the bunch is quasi-static in the frame of its mean
    # longitudinal motion, γ_z = γ/√(1 + a_w²): a rest-frame-round bunch there
    # has lab length σ_r/γ_z.
    radius, peak_current = 100.0e-6, 100.0
    lattice = _lattice(20, step=PERIOD)
    plan, slices, frame = _gaussian_bunch_case(
        GAMMA,
        radius,
        radius,
        peak_current,
        lattice,
        strength=DEFLECTION**2,
        count=64,
        beamlets=1024,
        beta=math.sqrt(2.0) * GAMMA / (DEFLECTION * UNDULATOR_WAVENUMBER),
        transverse_cells=48,
        transverse="omitted",
    )
    result = _solve(plan, slices)
    positions = np.asarray(slices.positions)
    spacing = float(positions[1] - positions[0])
    total = -float(np.sum(np.asarray(slices.currents))) * spacing / LIGHT

    def longitudinal_field(rest_position: float) -> float:
        # ⟨E_z⟩ of a Gaussian bunch averaged over the Gaussian witness plane,
        # integrated over q = e^v so every width scale is resolved.
        def integrand(logarithm: float) -> float:
            parameter = math.exp(logarithm)
            width = 2.0 * radius**2 + parameter
            return (
                2.0
                * rest_position
                / width**1.5
                * math.exp(-(rest_position**2) / width)
                / (4.0 * radius**2 + parameter)
                * parameter
            )

        value, _ = integrate.quad(integrand, -80.0, 20.0, limit=1000)
        return total / (4.0 * math.pi * PERMITTIVITY * math.sqrt(math.pi)) * value

    # Positive-late ζ trails, so the rest-frame forward coordinate is −γ_z ζ.
    expected = np.asarray(
        [
            -CHARGE * longitudinal_field(float(-frame * position)) / (MASS * LIGHT**2)
            for position in positions
        ]
    )
    evidence = result.evidence
    assert bool(evidence.space_charge_accepted)
    assert int(evidence.status) == FELStatus.SUCCESS
    assert np.all(np.asarray(evidence.space_charge_cells_per_sigma) >= 3.0)
    gain = np.asarray(result.space_charge_gain)
    # The first step's slice-mean energy change is the entrance field × Δz.
    np.testing.assert_allclose(
        gain[1] / PERIOD, expected, atol=0.05 * np.abs(expected).max()
    )
    change = np.asarray(result.mean_lorentz_factor - result.mean_lorentz_factor[0])
    # Mean-γ roundoff at γ = 10⁴ is ~10⁻⁸; the collective change is ~10⁻².
    np.testing.assert_allclose(change, gain, rtol=0.0, atol=1e-7)
    # Negligible at γ = 10⁴ (∝ 1/γ_z²), and reported as such.
    assert float(np.max(evidence.transverse_space_charge_ratio)) < 1.0e-3
    # The exchange is ~10⁻⁶ of the beam energy, so roundoff bounds the ledger.
    ledger = result.ledger
    assert float(ledger.space_charge_loss) == pytest.approx(
        -float(ledger.beam_energy_change), rel=1e-4
    )


def test_transverse_space_charge_kick_scales_as_inverse_gamma_z_squared() -> None:
    # A bunch 100× longer than wide in its rest frame: central slices feel the
    # line-charge field E_r = λ(1 − e^{−r²/2σ²})/(2πε₀ r); in the lab the force
    # q(E⊥ − v_z B) = qE⊥/γ_z² kicks Δu⊥ = qE⊥Δz/(mₑc²γ_z²) outward.
    gamma, radius, peak_current, beta = 1.0e3, 30.0e-6, 100.0, 10.0
    count, steps, step = 32, 10, 0.01
    center = count // 2
    current = peak_current * math.exp(
        -0.5 * ((center - 0.5 * (count - 1)) * 8.0 / count) ** 2
    )
    results: dict[tuple[float, FELTransverseSpaceCharge], FELTimeDependentResult] = {}
    for deflection in (1.0, 1.0e-3):
        lattice = _helical_lattice(deflection, step, steps)
        for mode in typing.get_args(FELTransverseSpaceCharge):
            plan, slices, frame = _gaussian_bunch_case(
                gamma,
                radius,
                100.0 * radius,
                peak_current,
                lattice,
                strength=deflection**2,
                count=count,
                beamlets=1024,
                beta=beta,
                transverse_cells=48,
                transverse=mode,
                transverse_tolerance=1.0e-3,
            )
            results[deflection, mode] = _solve(plan, slices)
        applied, omitted = results[deflection, "applied"], results[deflection, "omitted"]
        # Rest-frame cells are 25σ⊥ long: the central slice's field is that of
        # its own (sampled) charge line, centered on its own centroid.
        particles = np.asarray(applied.initial_particles.positions[center])
        offsets = particles - particles.mean(axis=0)
        separation = np.sqrt(np.sum(offsets**2, axis=-1))
        field = (
            current
            / (2.0 * math.pi * PERMITTIVITY * LIGHT * separation)
            * (1.0 - np.exp(-(separation**2) / (2.0 * radius**2)))
        )
        kick = steps * CHARGE * field * step / (MASS * LIGHT**2 * frame**2)
        expected = (kick / separation)[:, None] * offsets
        difference = np.asarray(
            applied.particles.transverse_momenta[center]
            - omitted.particles.transverse_momenta[center]
        )
        scale = float(np.sqrt(np.mean(np.sum(expected**2, axis=-1))))
        # Kicks of the smooth Gaussian line; the sampled line (1024 beamlets)
        # scatters them by a few percent of the rms kick, and deposit + gather
        # at four cells per σ lower the X3 field by ≈ 3 %.
        fitted = float(np.sum(difference * expected) / np.sum(expected**2))
        assert fitted == pytest.approx(1.0, abs=0.05)
        residual = difference - expected
        assert float(np.sqrt(np.mean(np.sum(residual**2, axis=-1)))) < 0.1 * scale
        momenta = np.asarray(omitted.initial_particles.transverse_momenta[center])
        entrance = float(np.sqrt(np.mean(np.sum(momenta**2, axis=-1))))
        ratio = float(omitted.evidence.transverse_space_charge_ratio[center])
        assert ratio == pytest.approx(scale / entrance, rel=0.05)
    strong = float(results[1.0, "omitted"].evidence.transverse_space_charge_ratio[center])
    weak = float(
        results[1.0e-3, "omitted"].evidence.transverse_space_charge_ratio[center]
    )
    # γ_z² = γ²/(1 + a_w²): the undulator kick is (1 + a_w²)× the drift kick.
    assert strong / weak == pytest.approx(2.0 / (1.0 + 1.0e-6), rel=0.02)
    # Omitting a kick above the declared tolerance is flagged; applying it is not.
    assert strong > 1.0e-3
    omitted_status = int(results[1.0, "omitted"].evidence.status)
    assert omitted_status == FELStatus.TRANSVERSE_SPACE_CHARGE_OMITTED
    assert int(results[1.0, "applied"].evidence.status) == FELStatus.SUCCESS


def _bunched(gamma: float, wavelength: float, modulation: float) -> FELPrebunching:
    """Cold-beam bunching ``b₀ = J₁(X)`` from a modulator and a 1 mm chicane."""
    r56 = 1.0e-3
    energy = modulation * gamma * wavelength / (2.0 * math.pi * r56)
    return FELPrebunching(
        (FELModulator(energy), _chicane(r56)), harmonic=1, reference_lorentz_factor=gamma
    )


def _cold_slices(
    count: int,
    spacing: float,
    *,
    current: float,
    gamma: float,
    radius: float,
    beta: float,
) -> FELBeamSlices:
    return _slices(
        count,
        spacing,
        current=current,
        emittance=radius**2 * gamma / beta,
        gamma=gamma,
        beta=beta,
    )


def _modulations(particles: FELParticles, slot: int) -> tuple[complex, complex, float]:
    """Slice bunching ``⟨e^{−iθ}⟩``, energy modulation ``⟨Δγ e^{−iθ}⟩``, area 2πσ_xσ_y."""
    phases = np.asarray(particles.phases[slot])
    gamma = np.asarray(particles.lorentz_factors[slot])
    positions = np.asarray(particles.positions[slot])
    rotation = np.exp(-1j * phases)
    deviation = np.std(positions, axis=0)
    area = 2.0 * math.pi * float(deviation[0] * deviation[1])
    return (
        complex(np.mean(rotation)),
        complex(np.mean((gamma - gamma.mean()) * rotation)),
        area,
    )


def _disk_reduction(wavenumber: float, area: float, frame: float) -> float:
    """Transversely averaged LSC of a uniform disk: 1 − 2 I₁(ξ)K₁(ξ), ξ = k r_b/γ_z."""
    argument = wavenumber * math.sqrt(area / math.pi) / frame
    return float(
        1.0
        - 2.0
        * special.ive(np.float64(1.0), np.float64(argument))
        * special.kve(np.float64(1.0), np.float64(argument))
    )


def test_longitudinal_space_charge_plasma_oscillation_at_the_plasma_wavelength() -> None:
    # A weak undulator (a_w = 10⁻³) is a drift for the beam; an initially
    # bunched cold beam performs longitudinal plasma oscillations at
    # λ_p = 2πγ^{3/2}c/(ω_p √(F(1 + a_w²))), ω_p² = e²n/(ε₀mₑ), n = I/(ecA), with
    # F the disk reduction of the one-dimensional model. The bunching and the
    # energy modulation ⟨Δγ e^{−iθ}⟩ exchange amplitude with period λ_p.
    gamma, deflection, current, radius, beta = 300.0, 1.0e-3, 100.0, 3.0e-6, 1.0e3
    lattice = _helical_lattice(deflection, 0.01, 200)
    wavelength = lattice.resonant_wavelength(gamma)
    wavenumber = 2.0 * math.pi / wavelength
    # The negligible FEL exchange (~10⁻¹² of the beam energy) sits at the γ
    # roundoff of the energy sums, so the ledger tolerance is relaxed.
    plan = FELTimeDependentPlan(
        FELPlan(
            lattice,
            wavelength,
            loading=FELLoading(1024, 4, shot_noise="quiet"),
            ledger_tolerance=1.0e-3,
        ),
        slippage="commensurate",
        boundary="periodic",
        prebunching=_bunched(gamma, wavelength, 0.02),
        space_charge=FELSpaceCharge(transverse="omitted", harmonics=1),
    )
    slices = _cold_slices(
        2, wavelength, current=current, gamma=gamma, radius=radius, beta=beta
    )
    result = _solve(plan, slices)
    assert int(result.evidence.status) == FELStatus.SUCCESS
    bunching, modulation, area = _modulations(result.initial_particles, 0)
    stretch = 1.0 + deflection**2
    reduction = _disk_reduction(wavenumber, area, gamma / math.sqrt(stretch))
    plasma = math.sqrt(
        current / (CHARGE * LIGHT * area) * CHARGE**2 / (PERMITTIVITY * MASS)
    )
    plasma_wavelength = (
        2.0 * math.pi * LIGHT * gamma**1.5 / (plasma * math.sqrt(reduction * stretch))
    )
    lattice_length = float(lattice.length)
    assert lattice_length > 1.25 * plasma_wavelength
    plasma_wavenumber = 2.0 * math.pi / plasma_wavelength
    dispersion = wavenumber * stretch / gamma**3  # dθ'/dγ
    positions = np.asarray(result.positions)
    phase = plasma_wavenumber * positions
    expected = bunching * np.cos(phase) - 1j * dispersion * modulation / (
        plasma_wavenumber
    ) * np.sin(phase)
    np.testing.assert_allclose(
        np.asarray(result.bunching[:, 0, 0]), expected, atol=1.0e-2 * abs(bunching)
    )
    # Energy modulation after 1.5 plasma periods, amplitude k_p|b₀|/κ.
    _, final_modulation, _ = _modulations(result.particles, 0)
    amplitude = plasma_wavenumber / dispersion
    end = plasma_wavenumber * lattice_length
    expected_modulation = modulation * math.cos(
        end
    ) - 1j * amplitude * bunching * math.sin(end)
    assert abs(final_modulation - expected_modulation) < 2.0e-2 * amplitude * abs(
        bunching
    )


def _lsc_generator(
    lattice: FELUndulatorLattice,
    segment: int,
    wavelength: float,
    gamma: float,
    current: float,
    areas: tuple[float, float],
    reduction: float,
    betatron_squared: float,
) -> np.ndarray:
    """Cold 1-D linear generator of ``(a, b, P = ⟨(Δγ/γ)e^{−iθ}⟩)`` with LSC.

    Coherent emission and FEL drive (Bonifacio, Pellegrini & Narducci 1984)
    plus the longitudinal space-charge field ``−i (I e/(ε₀ c k mₑc²)) F b/A``;
    the mean betatron detuning ``−k⟨u⊥²⟩/(2γ²)`` rotates the bunching.
    """
    radiation_area, beam_area = areas
    strength = lattice.deflections[segment] ** 2  # helical a_w = K
    coupling = lattice.fundamental_couplings[segment]
    wavenumber = 2.0 * math.pi / wavelength
    source = coupling * current / (PERMITTIVITY * LIGHT * radiation_area * gamma)
    drive = -CHARGE * coupling / (2.0 * MASS * LIGHT**2 * gamma**2)
    field = CHARGE * current / (PERMITTIVITY * LIGHT * wavenumber * MASS * LIGHT**2)
    return np.asarray(
        [
            [0.0, source, -source],
            [
                0.0,
                0.5j * wavenumber * betatron_squared / gamma**2,
                -1j * wavenumber * (1.0 + strength) / gamma**2,
            ],
            [drive, -1j * field * reduction / (beam_area * gamma), 0.0],
        ],
        dtype=np.complex128,
    )


def test_space_charge_microbunching_in_drift_and_undulator_matches_linear_theory() -> (
    None
):
    # 1 m of weak undulator (a drift) turns a small density modulation into an
    # LSC energy modulation that a 20-period undulator's dispersion turns back
    # into bunching, while the undulator radiates coherently. The bunching
    # follows the cold linear theory of both sections (LSC with its γ_z and
    # disk reduction, dispersion, FEL coupling); without LSC it would differ
    # by more than the initial bunching.
    gamma, current, radius = 300.0, 100.0, 10.0e-6
    strong, weak = 1.0, 1.0e-3
    strong_period = 0.01
    weak_period = strong_period * (1.0 + strong**2) / (1.0 + weak**2)
    # Helical natural focusing k_β² = a_w²k_u²/(2γ²) matched by smooth focusing
    # in the weak section: constant beam size in both.
    focusing = (strong * 2.0 * math.pi / strong_period / gamma) ** 2 / 2.0
    beta = 1.0 / math.sqrt(focusing)
    gradient = focusing * gamma * MASS * LIGHT / CHARGE

    def module(deflection: float, period: float, periods: int) -> InsertionDeviceField:
        peak = deflection * 2.0 * math.pi * MASS * LIGHT / (CHARGE * period)
        return InsertionDeviceField(
            peak,
            period,
            periods,
            polarization="helical",
            center=0.0,
            aperture=(1e-3, 1e-3),
        )

    lattice = FELUndulatorLattice(
        SCALE,
        (
            FELUndulatorSegment(
                module(weak, weak_period, 50), smooth_focusing_gradient=gradient
            ),
            FELUndulatorSegment(module(strong, strong_period, 20)),
        ),
        step_length=strong_period,
    )
    wavelength = lattice.resonant_wavelength(gamma)
    wavenumber = 2.0 * math.pi / wavelength
    slices = _cold_slices(
        2, wavelength, current=current, gamma=gamma, radius=radius, beta=beta
    )

    def run(space_charge: FELSpaceCharge | None) -> FELTimeDependentResult:
        return _solve(
            FELTimeDependentPlan(
                FELPlan(
                    lattice, wavelength, loading=FELLoading(16384, 4, shot_noise="quiet")
                ),
                slippage="spectral",
                boundary="periodic",
                prebunching=_bunched(gamma, wavelength, 0.02),
                space_charge=space_charge,
            ),
            slices,
        )

    result = run(FELSpaceCharge(transverse="omitted", harmonics=1))
    assert int(result.evidence.status) == FELStatus.SUCCESS
    bunching, modulation, area = _modulations(result.initial_particles, 0)
    areas = (float(result.scaling.beam_area[0]), area)
    momenta = np.asarray(result.initial_particles.transverse_momenta[0])
    betatron_squared = float(np.mean(np.sum(momenta**2, axis=-1)))
    generators = [
        _lsc_generator(
            lattice,
            segment,
            wavelength,
            gamma,
            current,
            areas,
            _disk_reduction(wavenumber, area, gamma / math.sqrt(1.0 + deflection**2)),
            betatron_squared,
        )
        for segment, deflection in enumerate((weak, strong))
    ]
    drift = 50 * weak_period
    initial = np.asarray([0.0, bunching, modulation / gamma], dtype=np.complex128)
    boundary = _propagator(generators[0] * drift) @ initial
    positions = np.asarray(result.positions)
    expected = np.asarray(
        [
            (_propagator(generators[0] * z) @ initial)[1]
            if z <= drift + 1.0e-12
            else (_propagator(generators[1] * (z - drift)) @ boundary)[1]
            for z in positions
        ]
    )
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(
        np.asarray(result.bunching[:, 0, 0]), expected, atol=2.0e-2 * scale
    )
    free = run(None)
    assert abs(complex(free.bunching[-1, 0, 0]) - expected[-1]) > abs(bunching)


def test_radial_space_charge_matches_gaussian_beam_reduction() -> None:
    # Grid model: the radial solve of [1 − (γ_z/k)²∇⊥²]E = S for a round
    # Gaussian beam averages to ⟨E⟩ = −i (I e/(ε₀ c k mₑc²)) b (e^x E₁(x))/(4πκ),
    # κ = (γ_z/k)², x = σ²/κ (Fourier–Bessel transform of the Gaussian source).
    gamma, deflection, current, radius, beta = 300.0, 1.0e-3, 100.0, 2.5e-6, 1.0e3
    lattice = _helical_lattice(deflection, 0.01, 1)
    wavelength = lattice.resonant_wavelength(gamma)
    wavenumber = 2.0 * math.pi / wavelength
    plan = FELTimeDependentPlan(
        FELPlan(
            lattice,
            wavelength,
            loading=FELLoading(16384, 4, shot_noise="quiet"),
            transverse="angular-spectrum",
            field_space=_grid_space(161, 200.0e-6),
            propagation=AngularSpectrumPlan(16),
            ledger_tolerance=1.0e-3,
        ),
        slippage="commensurate",
        boundary="periodic",
        prebunching=_bunched(gamma, wavelength, 0.02),
        space_charge=FELSpaceCharge(transverse="omitted", harmonics=1),
    )
    result = _solve(
        plan,
        _cold_slices(
            2, wavelength, current=current, gamma=gamma, radius=radius, beta=beta
        ),
    )
    assert int(result.evidence.status) == FELStatus.SUCCESS
    bunching, initial, _ = _modulations(result.initial_particles, 0)
    _, final, _ = _modulations(result.particles, 0)
    positions = np.asarray(result.initial_particles.positions[0])
    variance = float(np.mean(np.var(positions, axis=0)))
    screening = (gamma / math.sqrt(1.0 + deflection**2) / wavenumber) ** 2
    ratio = variance / screening
    field = CHARGE * current / (PERMITTIVITY * LIGHT * wavenumber * MASS * LIGHT**2)
    average = (
        -1j
        * field
        * bunching
        * math.exp(ratio)
        * special.exp1(np.float64(ratio))
        / (4.0 * math.pi * screening)
    )
    assert 0.3 < ratio < 3.0  # the transverse reduction is order one
    expected = 0.01 * average
    assert abs(final - initial - expected) < 3.0e-2 * abs(expected)


# -------------------------------------------------------- grid-model ledger


def _grid_space(count: int, half: float) -> PlaneFieldSpace:
    grid = TensorGridPlan(
        (UniformAxisSpec(count), UniformAxisSpec(count)), axis_names=("x", "y")
    ).prepare(jnp.asarray([[-half, -half], [half, half]]))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "finite-window")


def _gaussian_pulse(
    space: PlaneFieldSpace,
    count: int,
    spacing: float,
    wavelength: float,
    *,
    power: float,
    center: float,
    rms: float,
    waist: float,
) -> FELPulseSeed:
    slot_time = spacing / LIGHT
    time = PulseTimeSpace(
        TensorGridPlan((UniformAxisSpec(count),), axis_names=("t",)).prepare(
            jnp.asarray([[0.0], [(count - 1) * slot_time]])
        ),
        topology="finite-window",
    )
    radius = np.sum(np.asarray(space.transverse_coordinates) ** 2, axis=-1)
    positions = np.arange(count) * spacing
    amplitude = math.sqrt(4.0 * power / (PERMITTIVITY * LIGHT * math.pi * waist**2))
    values = (
        amplitude
        * np.exp(-radius / waist**2)[:, :, None]
        * np.exp(-((positions - center) ** 2) / (4.0 * rms**2))[None, None, :]
    )
    return FELPulseSeed(
        PulseEnvelopeField(
            space, time, values.astype(np.complex128), _frequency(wavelength), 0.0
        )
    )


def _grid_case(
    count: int, periods: int, grid_points: int
) -> tuple[FELTimeDependentPlan, FELBeamSlices, float]:
    lattice = _lattice(periods, step=PERIOD)
    wavelength = lattice.resonant_wavelength(GAMMA)
    size = 30.0e-6
    slices = _slices(
        count, wavelength, current=CURRENT, emittance=size**2 * GAMMA / BETA, spread=1e-4
    )
    space = _grid_space(grid_points, 240.0e-6)
    plan = FELTimeDependentPlan(
        FELPlan(
            lattice,
            wavelength,
            loading=FELLoading(256, 4, shot_noise="quiet"),
            transverse="angular-spectrum",
            field_space=space,
            propagation=AngularSpectrumPlan(16),
            slice_batch=64,
        ),
        slippage="spectral",
        boundary="open",
        pulse_seed=_gaussian_pulse(
            space,
            count,
            wavelength,
            wavelength,
            power=1.0e5,
            center=(count - 56) * wavelength,
            rms=20.0 * wavelength,
            waist=40.0e-6,
        ),
    )
    return plan, slices, wavelength


def test_grid_model_ledger_closes_with_diffraction_and_exit() -> None:
    plan, slices, _ = _grid_case(96, 60, 17)
    result = _solve(plan, slices)
    ledger = result.ledger
    assert float(ledger.diffraction_loss) > 0.0
    assert float(ledger.exit_energy) > 0.0
    assert float(ledger.relative_defect) < 1.0e-6
    assert bool(result.evidence.diffraction_accepted)


# -------------------------------------------------------- Genesis 4 oracle


def _genesis4() -> tuple[str, str]:
    executable = shutil.which(os.environ.get("PHYDRAX_GENESIS4", "genesis4"))
    version = os.environ.get("PHYDRAX_GENESIS4_VERSION")
    if executable is None or version is None:
        pytest.skip(
            "requires a pinned Genesis 1.3 version 4 executable (set PHYDRAX_GENESIS4 "
            "and PHYDRAX_GENESIS4_VERSION)"
        )
    return executable, version


def test_genesis4_oracle_seeded_time_dependent_amplifier(tmp_path: Path) -> None:
    executable, version = _genesis4()
    count = 256
    plan, slices, wavelength = _grid_case(count, 150, 33)
    pinned = pin_executable(executable, version=version, license_id="GPL-3.0-only")
    # Space charge on both sides: the intra-slice fundamental on Genesis' radial
    # grid, and the bunch-scale field (X3 here, Genesis' uniform-disk long-range
    # model there) in the mean-motion frame γ_z; X3 cells of four slices, ±6σ.
    frame = GAMMA / math.sqrt(1.0 + DEFLECTION**2)
    cells = count // 4 + 2
    extent = 2.0 * frame * cells * wavelength
    grid = TensorGridPlan(
        (UniformCellAxisSpec(48), UniformCellAxisSpec(48), UniformCellAxisSpec(cells))
    ).prepare(np.asarray([[-180.0e-6, -180.0e-6, -extent], [180.0e-6, 180.0e-6, extent]]))
    space_charge = FELSpaceCharge(
        SpaceChargeIGFPlan(
            SCALE, grid, capacity=count * plan.core.loading.particle_count
        ),
        transverse="omitted",
        harmonics=1,
    )
    seed = Genesis4GaussianSeed(
        1.0e5,
        center_position=(count - 56) * wavelength,
        rms_length=20.0 * wavelength,
        waist=40.0e-6,
    )
    ours: dict[bool, FELTimeDependentResult] = {}
    theirs = {}
    for collective in (False, True):
        charge = space_charge if collective else None
        destination = tmp_path / ("space-charge" if collective else "free")
        destination.mkdir()
        # Genesis receives the same lattice, beam, window, and space charge; its
        # seed is the same Gaussian pulse declared in Genesis' own profile form.
        theirs[collective] = run_genesis4(
            pinned,
            FELTimeDependentPlan(
                plan.core, slippage="spectral", boundary="open", space_charge=charge
            ),
            slices,
            destination,
            seed=seed,
        )
        ours[collective] = _solve(
            FELTimeDependentPlan(
                plan.core,
                slippage="spectral",
                boundary="open",
                pulse_seed=plan.pulse_seed,
                space_charge=charge,
            ),
            slices,
        )
        energy = np.asarray(ours[collective].pulse_energy[:, 0])
        oracle = np.asarray(theirs[collective].pulse_energy)
        assert oracle.shape == energy.shape
        assert oracle[-1] > 3.0 * oracle[0]
        np.testing.assert_allclose(energy[::25], oracle[::25], rtol=0.05)
        correlation = np.corrcoef(
            np.asarray(ours[collective].power[-1, :, 0]), theirs[collective].power[-1]
        )[0, 1]
        assert correlation > 0.99
    # Both codes lower the output by the same ≈ 0.25 % through space charge.
    effect = float(ours[True].pulse_energy[-1, 0] / ours[False].pulse_energy[-1, 0])
    oracle_effect = float(theirs[True].pulse_energy[-1] / theirs[False].pulse_energy[-1])
    assert oracle_effect < 0.999
    assert effect == pytest.approx(oracle_effect, abs=5.0e-4)
    # Genesis has no transverse space charge; here it reaches ≈ 10 % of the rms
    # transverse momentum over the undulator, which the evidence reports.
    evidence = ours[True].evidence
    assert bool(evidence.space_charge_accepted)
    assert float(np.max(evidence.transverse_space_charge_ratio)) > 1.0e-2
    assert int(evidence.status) & FELStatus.TRANSVERSE_SPACE_CHARGE_OMITTED


# -------------------------------------------------------------- refusals


def test_time_dependent_configuration_refusals() -> None:
    lattice = _lattice(4)
    wavelength = lattice.resonant_wavelength(GAMMA)
    loading = FELLoading(2, 4, shot_noise="quiet")
    seeded = FELPlan(lattice, wavelength, loading=loading, seed=FELSeed(1.0, waist=1e-5))
    with pytest.raises(ValueError, match="refuses the continuous-wave FELSeed"):
        FELTimeDependentPlan(seeded, slippage="spectral", boundary="open")
    core = FELPlan(lattice, wavelength, loading=loading)
    with pytest.raises(ValueError, match="uniformly spaced"):
        FELTimeDependentPlan(core, slippage="spectral", boundary="periodic").solve(
            _slices(1, wavelength, current=1.0, emittance=1e-7), jax.random.key(0)
        )
    ramp = _plane_wave_seed(np.linspace(1.0, 2.0, 8), wavelength, _frequency(wavelength))
    with pytest.raises(ValueError, match="time spacing times c"):
        FELTimeDependentPlan(
            core, slippage="spectral", boundary="periodic", pulse_seed=ramp
        ).solve(
            _slices(8, 2.0 * wavelength, current=1.0, emittance=1e-7), jax.random.key(0)
        )
    nonuniform = np.ones((2, 2, 8), dtype=np.complex128)
    nonuniform[0, 0] = 2.0
    envelope = PulseEnvelopeField(
        PLANE,
        ramp.envelope.time_space,
        nonuniform,
        _frequency(wavelength),
        0.0,
    )
    with pytest.raises(ValueError, match="transversely uniform"):
        FELTimeDependentPlan(
            core,
            slippage="spectral",
            boundary="periodic",
            pulse_seed=FELPulseSeed(envelope),
        )
    with pytest.raises(ValueError, match="2 \\* prebunching.harmonic"):
        FELTimeDependentPlan(
            core,
            slippage="spectral",
            boundary="periodic",
            prebunching=FELPrebunching(
                (FELModulator(1.0),), harmonic=3, reference_lorentz_factor=GAMMA
            ),
        )


def test_space_charge_configuration_refusals() -> None:
    lattice = _lattice(4)
    wavelength = lattice.resonant_wavelength(GAMMA)
    core = FELPlan(lattice, wavelength, loading=FELLoading(2, 4, shot_noise="quiet"))
    grid = TensorGridPlan((UniformCellAxisSpec(4),) * 3).prepare(
        np.asarray([[-1e-3] * 3, [1e-3] * 3])
    )
    bunch = SpaceChargeIGFPlan(SCALE, grid, capacity=3)
    with pytest.raises(ValueError, match="bunch-scale plan or at least one harmonic"):
        FELSpaceCharge(transverse="omitted")
    with pytest.raises(ValueError, match="transverse kick comes from the bunch-scale"):
        FELSpaceCharge(transverse="applied", harmonics=1)
    with pytest.raises(ValueError, match="particles_per_beamlet >= 2 \\* harmonics"):
        FELTimeDependentPlan(
            core,
            slippage="spectral",
            boundary="periodic",
            space_charge=FELSpaceCharge(transverse="omitted", harmonics=3),
        )
    # The free-space bunch field of a finite window is not that of an
    # infinitely long periodic beam.
    with pytest.raises(ValueError, match="needs an open window"):
        FELTimeDependentPlan(
            core,
            slippage="spectral",
            boundary="periodic",
            space_charge=FELSpaceCharge(bunch, transverse="omitted"),
        )
    with pytest.raises(ValueError, match="slices × particles per slice"):
        FELTimeDependentPlan(
            core,
            slippage="spectral",
            boundary="open",
            space_charge=FELSpaceCharge(bunch, transverse="omitted"),
        ).solve(_slices(4, wavelength, current=1.0, emittance=1e-7), jax.random.key(0))
