#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Full-wave boosted-frame FEL against independent references.

References: the period-averaged time-independent FEL (`FELPlan`, X5a) for the
small-signal gain in its 1-D, low-gain overlap regime; the analytic lab energy
``ε₀ A ∫ E_x² dξ`` of the declared seed plane wave; A1 trajectory radiation of
the same recorded tracks times the energy factor of the order-one spline
deposit evaluated on the boosted track (the linear-hat weights of the grid
nodes, not the solver) for the spontaneous spectrum; the steady coherent field
``κ I L |F| / (ε₀ c A γ)`` of a prebunched beam with ``κ = a_w [JJ]/√2`` and
``[JJ] = J₀(ξ) − J₁(ξ)``, ``ξ = K²/(4 + 2K²)`` (Kroll, Morton & Rosenbluth, IEEE
JQE 17, 1436, 1981) from SciPy Bessel functions, and the fundamental
bunching ``J₁(a)`` of the phase modulation ``θ = ψ − a sin ψ``; and a lab-frame
electromagnetic PIC run assembled from the public PIC API on the same beam.

Tolerances follow measured convergence: the Huygens surface quadrature and the
boosted grid dispersion leave ``O((k′h)²)`` errors (3 % at eight cells per
boosted wavelength about the spectral peak, 1 % at twelve); the gain carries
the averaged model's ``γ ≫ 1`` resonance and undulator end-ramp differences
(+3.6 % here, falling with resolution and ramp length).
"""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from phydrax import ElectromagneticScaleContract
from phydrax.applications.accelerator import InsertionDeviceField
from phydrax.applications.accelerator.fel import (
    FELBeamSlices,
    FELFullWaveBeam,
    FELFullWaveHuygens,
    FELFullWavePlan,
    FELFullWaveResult,
    FELFullWaveSeed,
    FELFullWaveStatus,
    FELFullWaveTracks,
    FELLoading,
    FELPlan,
    FELSeed,
    FELUndulatorLattice,
    FELUndulatorSegment,
    PreparedFELFullWave,
)
from phydrax.discretization import (
    ChargedParticlePlan,
    ParticlePopulationPlan,
    ParticleSetPlan,
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.discretization.pic import (
    ChargeConservingCurrentPlan,
    PIC_CODE_RELATIVITY,
    PICChargeModelPlan,
    PICParticleCochainTransferPlan,
    PICSpeciesPlan,
)
from phydrax.electromagnetics import RadiationObserverPlan, TrajectoryRadiationPlan
from phydrax.solver import ElectromagneticPICPlan
from phydrax.solver.maxwell.spectral import SpectralMaxwellPlan
from phydrax.units import CHARGE, UnitDefinition


SCALE = ElectromagneticScaleContract.code_units(
    PIC_CODE_RELATIVITY.dimensional_scale,
    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
    gravitational_constant=1,
    speed_of_light=1,
    reduced_planck_constant=1,
    boltzmann_constant=1,
    elementary_charge=1,
    electron_mass=1,
    vacuum_permittivity=1,
    constant_set_id="fel-full-wave-test",
)
DEFLECTION = 1.0
PERIOD = 1.0
# The 1-D cross section: area 2πσ² of the averaged model's slice.
SIGMA = 0.01
AREA = 2.0 * math.pi * SIGMA**2
WIDTH = math.sqrt(AREA)


def _lattice(
    periods: int, *, ramp_periods: float, aperture: tuple[float, float]
) -> FELUndulatorLattice:
    device = InsertionDeviceField(
        DEFLECTION * 2.0 * math.pi / PERIOD,
        PERIOD,
        periods,
        polarization="planar",
        center=0.0,
        aperture=aperture,
        ramp_periods=ramp_periods,
    )
    return FELUndulatorLattice(
        SCALE, (FELUndulatorSegment(device),), step_length=PERIOD / 16.0
    )


def _resonant_boost(gamma: float) -> float:
    return gamma / math.sqrt(1.0 + 0.5 * DEFLECTION**2)


def _coupling() -> float:
    """``κ = a_w [JJ]/√2`` of the planar fundamental (SciPy Bessel functions)."""
    xi = DEFLECTION**2 / (4.0 + 2.0 * DEFLECTION**2)
    bessel = special.jv(np.asarray([0.0, 1.0]), np.asarray([xi, xi]))
    return (DEFLECTION / math.sqrt(2.0)) * float(bessel[0] - bessel[1]) / math.sqrt(2.0)


def _bunching(modulation: float) -> float:
    """Fundamental bunching ``J₁(a)`` of the phase modulation ``θ = ψ − a sin ψ``."""
    return float(special.jv(np.asarray([1.0]), np.asarray([modulation]))[0])


def _smooth_step(value: np.ndarray) -> np.ndarray:
    u = np.clip(value, 0.0, 1.0)
    rise = np.where(u > 0.0, np.exp(-1.0 / np.maximum(u, 1.0e-300)), 0.0)
    fall = np.where(u < 1.0, np.exp(-1.0 / np.maximum(1.0 - u, 1.0e-300)), 0.0)
    return rise / (rise + fall)


# -- seeded small-signal gain ----------------------------------------------------------

GAIN_GAMMA = 20.0
GAIN_PERIODS = 6
GAIN_CURRENT = 4.0e-3
GAIN_SEED = 0.3
# Detuned to the averaged model's gain maximum.
GAIN_DETUNING = 0.07


def _gain_plan(lattice: FELUndulatorLattice, wavelength: float) -> FELFullWavePlan:
    return FELFullWavePlan(
        lattice,
        wavelength,
        boost_lorentz_factor=_resonant_boost(GAIN_GAMMA),
        transverse_size=(WIDTH, WIDTH),
        cells_per_wavelength=24,
        steps_per_period=48,
        seed=FELFullWaveSeed(GAIN_SEED),
    )


def _averaged_gain(lattice: FELUndulatorLattice, wavelength: float) -> float:
    power = 0.5 * GAIN_SEED**2 * AREA
    emittance = 1.0e-6
    beta = GAIN_GAMMA * SIGMA**2 / emittance
    result = FELPlan(
        lattice,
        wavelength,
        loading=FELLoading(1, 16, shot_noise="quiet"),
        seed=FELSeed(power, waist=1.0),
    ).solve(
        FELBeamSlices(
            [0.0],
            [GAIN_CURRENT],
            [GAIN_GAMMA],
            [0.0],
            [[emittance, emittance]],
            [[beta, beta]],
            [[0.0, 0.0]],
        ),
        jax.random.key(0),
    )
    power_trace = np.asarray(result.power[0, :, 0])
    return float(power_trace[-1] / power_trace[0] - 1.0)


def _gain_setup() -> tuple[FELUndulatorLattice, float]:
    lattice = _lattice(GAIN_PERIODS, ramp_periods=0.125, aperture=(0.3, 0.09))
    return lattice, lattice.resonant_wavelength(GAIN_GAMMA) * (1.0 + GAIN_DETUNING)


def test_seeded_small_signal_gain_matches_the_averaged_fel() -> None:
    lattice, wavelength = _gain_setup()
    plan = _gain_plan(lattice, wavelength)
    beam = plan.flat_top_beam(
        GAIN_GAMMA,
        GAIN_CURRENT,
        wavelengths=22,
        particles_per_wavelength=8,
        taper_wavelengths=3.0,
    )
    result = plan.prepare(beam).run()
    assert int(result.evidence.status) == FELFullWaveStatus.SUCCESS
    assert result.steady_gain is not None and result.seed_amplitude is not None
    reference = _averaged_gain(lattice, wavelength)
    assert abs(float(result.steady_gain) / reference - 1.0) < 0.05
    # The FEL takes the field's gain out of the beam: with the discrete push's
    # lab-energy error, the ledger closes to its tolerance.
    assert float(result.ledger.beam_energy_change) < 0.0
    assert float(result.ledger.relative_defect) < plan.ledger_tolerance


def test_seed_antenna_launches_the_declared_plane_wave_and_closes_the_ledger() -> None:
    lattice, wavelength = _gain_setup()
    plan = _gain_plan(lattice, wavelength)
    # A test-charge beam: the seed crosses the undulator undisturbed.
    beam = plan.flat_top_beam(
        GAIN_GAMMA,
        1.0e-9,
        wavelengths=22,
        particles_per_wavelength=8,
        taper_wavelengths=3.0,
    )
    prepared = plan.prepare(beam)
    result = prepared.run()
    assert int(result.evidence.status) == FELFullWaveStatus.SUCCESS
    assert result.seed_amplitude is not None and result.steady_gain is not None
    assert abs(float(result.seed_amplitude) / GAIN_SEED - 1.0) < 1.0e-3
    assert abs(float(result.steady_gain)) < 1.0e-4
    # Lab energy ε₀ A ∫ E_x² dξ = ½ E₀² A ∫ g² dξ of the declared envelope, whose
    # flat top spans every electron's slippage interval and the two margins.
    seed = plan.seed
    assert seed is not None
    lower, upper = plan.undulator_span
    positions = np.asarray(beam.positions)[:, 2]
    speed = np.asarray(beam.velocities)[:, 2]
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    mean = np.sqrt(1.0 - (1.0 + 0.5 * DEFLECTION**2) / gamma**2)
    entry = lower - (lower - positions) / speed
    slip = (upper - lower) * (1.0 / mean - 1.0)
    margin = seed.margin_wavelengths * wavelength
    ramp = seed.ramp_wavelengths * wavelength
    flat = (np.min(entry - slip) - margin, np.max(entry) + margin)
    xi = np.linspace(flat[0] - ramp, flat[1] + ramp, 400_001)
    envelope = _smooth_step((xi - (flat[0] - ramp)) / ramp) * _smooth_step(
        ((flat[1] + ramp) - xi) / ramp
    )
    energy = 0.5 * GAIN_SEED**2 * AREA * np.trapezoid(envelope**2, xi)
    assert float(result.ledger.injected_energy) == pytest.approx(energy, rel=2.0e-3)
    assert float(result.ledger.relative_defect) < 1.0e-9


# -- spontaneous emission: boosted Huygens against A1 ----------------------------------

SPONTANEOUS_GAMMA = 10.0


def _spontaneous_run(offset: float) -> tuple[PreparedFELFullWave, FELFullWaveResult]:
    lattice = _lattice(2, ramp_periods=0.25, aperture=(0.3, 0.15))
    wavelength = lattice.resonant_wavelength(SPONTANEOUS_GAMMA)
    boost = _resonant_boost(SPONTANEOUS_GAMMA)
    beta_b = math.sqrt(1.0 - 1.0 / boost**2)
    # Boosted frequencies about the rest-frame oscillation γ_b β_b c k_u.
    frequencies = np.linspace(0.5, 1.5, 21) * boost * beta_b * 2.0 * math.pi / PERIOD
    radiation = TrajectoryRadiationPlan(
        SCALE,
        RadiationObserverPlan(np.asarray([[0.0, 0.0, 1.0]]), np.asarray([0.0, 1.0, 0.0])),
        frequencies * boost * (1.0 + beta_b),
        coherence="coherent",
        route="segment-exact",
    )
    boosted_wavelength = wavelength * boost * (1.0 + beta_b)
    plan = FELFullWavePlan(
        lattice,
        wavelength,
        boost_lorentz_factor=boost,
        transverse_size=(8.0 * boosted_wavelength, 8.0 * boosted_wavelength),
        transverse_cells=(64, 64),
        cells_per_wavelength=8,
        tracks=FELFullWaveTracks(radiation, (0,)),
        huygens=FELFullWaveHuygens(frequencies, np.asarray([[0.0, 0.0, 1.0]])),
    )
    beta = math.sqrt(1.0 - 1.0 / SPONTANEOUS_GAMMA**2)
    beam = FELFullWaveBeam(
        np.asarray([[offset, offset, plan.undulator_span[0] - 0.01]]),
        np.asarray([[0.0, 0.0, beta]]),
        np.asarray([1.0e-12]),
    )
    prepared = plan.prepare(beam)
    return prepared, prepared.run()


def _deposit_factor(
    prepared: PreparedFELFullWave, result: FELFullWaveResult, frequencies: np.ndarray
) -> np.ndarray:
    """Forward energy factor of the order-one spline deposit of the actual track.

    The boosted track ``(t′, x′, z′)`` weights the two grid nodes around ``z′``
    by the linear hat; the ratio of the resulting forward far field to the
    continuum one (at ``k = ω′/c``) is the grid's radiated-energy factor.
    """
    frame = prepared.plan.frame
    gamma_b, beta_b = frame.lorentz_factor, frame.beta
    track = prepared.boosted.lab_trajectory(result.final_state, 0, SCALE)
    time = np.asarray(track.times)[:, 0]
    position = np.asarray(track.positions)[:, 0]
    boosted_time = gamma_b * (time - beta_b * position[:, 2])
    boosted_z = gamma_b * (position[:, 2] - beta_b * time)
    velocity = np.diff(position[:, 0]) / np.diff(boosted_time)
    weight = np.diff(boosted_time)
    middle_time = 0.5 * (boosted_time[1:] + boosted_time[:-1])
    middle_z = 0.5 * (boosted_z[1:] + boosted_z[:-1])
    origin = prepared.frame_evidence.grid_origin[2]
    spacing = prepared.frame_evidence.grid_spacing[2]
    cell = np.floor((middle_z - origin) / spacing)
    fraction = (middle_z - origin) / spacing - cell
    node = origin + cell * spacing
    factors = []
    for omega in frequencies:
        source = weight * velocity * np.exp(1j * omega * middle_time)
        continuum = np.sum(source * np.exp(-1j * omega * middle_z))
        deposited = np.sum(
            source
            * (
                (1.0 - fraction) * np.exp(-1j * omega * node)
                + fraction * np.exp(-1j * omega * (node + spacing))
            )
        )
        factors.append(abs(deposited) ** 2 / abs(continuum) ** 2)
    return np.asarray(factors)


@pytest.mark.parametrize(
    "offset",
    [0.0, 0.5 * 8.0 / 64.0],
    ids=["axis-at-cell-center", "axis-at-node"],
)
def test_spontaneous_huygens_spectrum_matches_trajectory_radiation(
    offset: float,
) -> None:
    lattice = _lattice(2, ramp_periods=0.25, aperture=(0.3, 0.15))
    boost = _resonant_boost(SPONTANEOUS_GAMMA)
    beta_b = math.sqrt(1.0 - 1.0 / boost**2)
    boosted_wavelength = (
        lattice.resonant_wavelength(SPONTANEOUS_GAMMA) * boost * (1.0 + beta_b)
    )
    # The offset moves the electron by half a transverse cell: from a cell
    # center onto a node of the staggered grid.
    prepared, result = _spontaneous_run(offset * boosted_wavelength)
    assert int(result.evidence.status) == FELFullWaveStatus.SUCCESS
    assert result.huygens_spectrum is not None
    assert result.trajectory_spectrum is not None
    huygens = np.asarray(result.huygens_spectrum.spectral_energy)[:, 0]
    trajectory = np.asarray(result.trajectory_spectrum.spectral_energy)[:, 0]
    frequencies = np.asarray(result.huygens_spectrum.angular_frequencies)[:, 0] / (
        boost * (1.0 + beta_b)
    )
    factor = _deposit_factor(prepared, result, frequencies)
    # Within ±20 % of the spectral peak: the O((k′h)²) Huygens quadrature and grid
    # dispersion error at eight cells per boosted wavelength rises from 2 % at
    # the peak to 5.2 % at +20 % (3.4 % at twelve cells).
    peak = slice(6, 15)
    ratio = huygens[peak] / (trajectory[peak] * factor[peak])
    np.testing.assert_allclose(ratio, 1.0, atol=0.06)


# -- prebunched coherent emission --------------------------------------------------------

PREBUNCHED_GAMMA = 20.0
PREBUNCHED_PERIODS = 6


def _prebunched(modulation: float, current: float) -> FELFullWaveResult:
    lattice = _lattice(PREBUNCHED_PERIODS, ramp_periods=0.125, aperture=(0.3, 0.09))
    plan = FELFullWavePlan(
        lattice,
        lattice.resonant_wavelength(PREBUNCHED_GAMMA),
        boost_lorentz_factor=_resonant_boost(PREBUNCHED_GAMMA),
        transverse_size=(WIDTH, WIDTH),
        nci_energy_fraction=0.1,
    )
    beam = plan.flat_top_beam(
        PREBUNCHED_GAMMA,
        current,
        wavelengths=22,
        particles_per_wavelength=8,
        taper_wavelengths=3.0,
        phase_modulation=modulation,
    )
    return plan.prepare(beam).run()


@pytest.fixture(scope="module")
def prebunched_pair() -> tuple[FELFullWaveResult, FELFullWaveResult]:
    return _prebunched(0.2, 4.0e-3), _prebunched(0.4, 8.0e-3)


def test_prebunched_steady_amplitude_matches_kmr(
    prebunched_pair: tuple[FELFullWaveResult, FELFullWaveResult],
) -> None:
    result, _ = prebunched_pair
    assert int(result.evidence.status) == FELFullWaveStatus.SUCCESS
    assert result.steady_amplitude is not None
    length = PREBUNCHED_PERIODS * PERIOD
    expected = _coupling() * 4.0e-3 * length * _bunching(0.2) / (AREA * PREBUNCHED_GAMMA)
    assert float(result.steady_amplitude) == pytest.approx(expected, rel=0.03)


def test_prebunched_coherent_power_scales_as_current_times_bunching_squared(
    prebunched_pair: tuple[FELFullWaveResult, FELFullWaveResult],
) -> None:
    base, doubled = prebunched_pair
    assert base.steady_amplitude is not None and doubled.steady_amplitude is not None
    power_ratio = (float(doubled.steady_amplitude) / float(base.steady_amplitude)) ** 2
    expected = (2.0 * _bunching(0.4) / _bunching(0.2)) ** 2
    assert power_ratio == pytest.approx(expected, rel=0.03)


# -- boosted frame against the lab frame ---------------------------------------------------


def _lab_forward_amplitude(
    plan: FELFullWavePlan, beam: FELFullWaveBeam, window: tuple[float, float]
) -> tuple[float, float]:
    """Lab-frame PIC of the same pair beam: steady forward amplitude and mean
    electron ``Δγ``.

    A periodic box long enough that nothing wraps, sixteen cells per lab
    wavelength, the undulator gathered directly, and the forward wave
    ``(E_x + cB_y)/2`` with ``B_y`` moved onto the ``E_x`` nodes.
    """
    wavelength = plan.wavelength
    positions = np.asarray(beam.positions)
    velocities = np.asarray(beam.velocities)
    electrons = np.asarray(beam.electrons)
    lower, upper = plan.undulator_span
    gamma = 1.0 / np.sqrt(1.0 - np.sum(velocities**2, axis=1))
    mean = np.sqrt(1.0 - (1.0 + 0.5 * DEFLECTION**2) / gamma**2)
    stop = float(
        np.max((lower - positions[:, 2]) / velocities[:, 2] + (upper - lower) / mean)
    )
    spacing = wavelength / 16.0
    low = float(positions[:, 2].min()) - stop - 4.0 * wavelength
    count = int(
        math.ceil(
            (float(positions[:, 2].max()) + stop + 4.0 * wavelength - low) / spacing
        )
    )
    width_x, width_y = plan.transverse_size
    cells_x, cells_y = plan.transverse_cells
    corner = (
        -0.5 * width_x + 0.5 * width_x / cells_x,
        -0.5 * width_y + 0.5 * width_y / cells_y,
    )
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(n, periodic=True) for n in (cells_x, cells_y, count)),
        axis_names=("x", "y", "z"),
    ).prepare(
        jnp.asarray(
            [
                [corner[0], corner[1], low],
                [corner[0] + width_x, corner[1] + width_y, low + count * spacing],
            ]
        )
    )
    bridge = StructuredCochainBridge(grid)
    size = positions.shape[0]
    species, charged = [], []
    for index, (name, sign) in enumerate((("electron", -1.0), ("positron", 1.0))):
        support = ParticleSetPlan(
            jnp.arange(index * size, (index + 1) * size),
            jnp.ones((size,)),
            ambient_dimension=3,
        ).prepare()
        charged.append(
            ChargedParticlePlan(sign * jnp.ones((size,)), name).prepare(support)
        )
        species.append(
            PICSpeciesPlan(
                ParticlePopulationPlan(support),
                PICChargeModelPlan(
                    sign,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    transfer = PICParticleCochainTransferPlan(bridge, shape_order=1)
    transfers = tuple(transfer.prepare(value) for value in charged)
    currents = tuple(ChargeConservingCurrentPlan(value) for value in transfers)
    solver = SpectralMaxwellPlan(bridge, grid="staggered").prepare(transfers, currents)
    density = float(electrons.max()) / (spacing * width_x / cells_x * width_y / cells_y)
    pic = ElectromagneticPICPlan(
        solver,
        species=tuple(species),
        external_fields=tuple(segment.device for segment in plan.lattice.segments),
        continuity_tolerance=max(1.0e-9, 1.0e-7 * density),
        constraint_tolerance=max(1.0e-8, 1.0e-6 * density),
    )
    step = min(0.4 * spacing, float(solver.stable_step))
    steps = int(math.ceil(stop / step))
    state = pic.initialize(
        (jnp.asarray(positions), jnp.asarray(positions)),
        (jnp.asarray(velocities), jnp.asarray(velocities)),
        step,
        masses=(jnp.asarray(electrons), jnp.asarray(electrons)),
    )

    @eqx.filter_jit
    def advance(runtime: ElectromagneticPICPlan, value: Any) -> tuple[Any, jax.Array]:
        def body(current: Any, _: None) -> tuple[Any, jax.Array]:
            advanced = runtime.step_detailed(current, step)
            return advanced.accepted_state, advanced.successful

        return jax.lax.scan(body, value, None, length=steps)

    final, successful = advance(pic, state)
    assert bool(jnp.all(successful))
    electric = np.asarray(jnp.mean(final.field.electric, axis=(0, 1)))[:, 0]
    magnetic = np.asarray(jnp.mean(final.field.magnetic, axis=(0, 1)))[:, 1]
    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    # B_y sits half a cell above E_x along z.
    magnetic = np.real(
        np.fft.ifft(np.fft.fft(magnetic) * np.exp(-0.5j * wavenumbers * spacing))
    )
    forward = 0.5 * (electric + magnetic)
    target = 2.0 * math.pi / wavelength
    band = np.where(
        wavenumbers > 0.0, np.exp(-0.5 * ((wavenumbers / target - 1.0) / 0.3) ** 2), 0.0
    )
    envelope = np.abs(np.fft.ifft(2.0 * band * np.fft.fft(forward)))
    xi = low + spacing * np.arange(count) - float(final.time)
    inside = (xi >= window[0]) & (xi <= window[1])
    lorentz = np.sqrt(
        1.0 + np.sum(np.asarray(final.species[0].particles.proper_velocity) ** 2, axis=-1)
    )
    return float(np.mean(envelope[inside])), float(np.mean(lorentz - gamma))


def test_boosted_run_matches_a_lab_frame_pic_run() -> None:
    gamma = 2.0
    lattice = _lattice(3, ramp_periods=0.25, aperture=(0.3, 0.15))
    wavelength = lattice.resonant_wavelength(gamma)
    boost = _resonant_boost(gamma)
    boosted_wavelength = wavelength * boost * (1.0 + math.sqrt(1.0 - 1.0 / boost**2))
    plan = FELFullWavePlan(
        lattice,
        wavelength,
        boost_lorentz_factor=boost,
        transverse_size=(boosted_wavelength / 8.0, boosted_wavelength / 8.0),
        cells_per_wavelength=24,
    )
    beam = plan.flat_top_beam(
        gamma,
        1.0e-3,
        wavelengths=24,
        particles_per_wavelength=8,
        taper_wavelengths=2.0,
        phase_modulation=0.3,
    )
    prepared = plan.prepare(beam)
    result = prepared.run()
    assert int(result.evidence.status) == FELFullWaveStatus.SUCCESS
    assert result.steady_amplitude is not None and prepared.steady_window is not None
    amplitude, energy_change = _lab_forward_amplitude(plan, beam, prepared.steady_window)
    assert float(result.steady_amplitude) == pytest.approx(amplitude, rel=0.05)
    boosted_change = float(
        jnp.mean(result.final_lorentz_factors[0] - result.initial_lorentz_factors[0])
    )
    # The mean energy change carries both runs' push discretization (6 % here).
    assert boosted_change == pytest.approx(energy_change, rel=0.1)


# -- refusals ------------------------------------------------------------------------------


def _small_plan(
    *,
    boost: float = 5.0,
    transverse_cells: tuple[int, int] = (2, 2),
    seed: FELFullWaveSeed | None = None,
    huygens: FELFullWaveHuygens | None = None,
    absorber_cells: int = 16,
) -> FELFullWavePlan:
    lattice = _lattice(2, ramp_periods=0.25, aperture=(0.3, 0.15))
    return FELFullWavePlan(
        lattice,
        lattice.resonant_wavelength(6.0),
        boost_lorentz_factor=boost,
        transverse_size=(WIDTH, WIDTH),
        transverse_cells=transverse_cells,
        seed=seed,
        huygens=huygens,
        absorber_cells=absorber_cells,
    )


def test_plans_outside_code_units_are_refused() -> None:
    device = InsertionDeviceField(
        1.0, 0.03, 4, polarization="planar", center=0.0, aperture=(0.01, 0.01)
    )
    lattice = FELUndulatorLattice(
        ElectromagneticScaleContract.si(),
        (FELUndulatorSegment(device),),
        step_length=0.01,
    )
    with pytest.raises(ValueError, match="PIC code units"):
        FELFullWavePlan(
            lattice, 1.0e-9, boost_lorentz_factor=5.0, transverse_size=(1.0e-4, 1.0e-4)
        )


def test_lab_frame_boosts_are_refused() -> None:
    with pytest.raises(ValueError, match="must exceed one"):
        _small_plan(boost=1.0)


def test_seed_antenna_with_huygens_extraction_is_refused() -> None:
    with pytest.raises(ValueError, match="no Huygens surface is current-free"):
        _small_plan(
            seed=FELFullWaveSeed(0.1),
            huygens=FELFullWaveHuygens(np.asarray([1.0]), np.asarray([[0.0, 0.0, 1.0]])),
        )


def test_single_cell_absorbers_are_refused() -> None:
    with pytest.raises(ValueError, match="absorber_cells must be at least two"):
        _small_plan(absorber_cells=1)


def test_seed_on_a_helical_lattice_is_refused() -> None:
    device = InsertionDeviceField(
        2.0 * math.pi,
        PERIOD,
        4,
        polarization="helical",
        center=0.0,
        aperture=(0.3, 0.3),
    )
    lattice = FELUndulatorLattice(SCALE, (FELUndulatorSegment(device),), step_length=0.1)
    with pytest.raises(ValueError, match="planar lattices only"):
        FELFullWavePlan(
            lattice,
            lattice.resonant_wavelength(6.0),
            boost_lorentz_factor=4.0,
            transverse_size=(WIDTH, WIDTH),
            seed=FELFullWaveSeed(0.1),
        )


def test_seed_without_a_reference_margin_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two"):
        FELFullWaveSeed(0.1, margin_wavelengths=1.0)


def test_flat_top_beam_needs_the_one_dimensional_cross_section() -> None:
    plan = _small_plan(transverse_cells=(2, 4))
    with pytest.raises(ValueError, match="two vertical cells"):
        plan.flat_top_beam(6.0, 1.0e-3, wavelengths=4)


def _single(plan: FELFullWavePlan, z: float, electrons: float) -> FELFullWaveBeam:
    beta = math.sqrt(1.0 - 1.0 / 36.0)
    return FELFullWaveBeam(
        np.asarray([[0.0, 0.0, z]]),
        np.asarray([[0.0, 0.0, beta]]),
        np.asarray([electrons]),
    )


def test_beams_inside_the_undulator_are_refused() -> None:
    plan = _small_plan()
    with pytest.raises(ValueError, match="upstream of the lattice support"):
        plan.prepare(_single(plan, 0.0, 1.0e-12))


def test_charged_electron_only_boxes_are_refused() -> None:
    plan = _small_plan()
    with pytest.raises(ValueError, match="charged periodic box"):
        plan.prepare(_single(plan, plan.undulator_span[0] - 0.01, 1.0e6))


def test_huygens_boxes_reached_by_transverse_images_are_refused() -> None:
    plan = _small_plan(
        transverse_cells=(16, 16),
        huygens=FELFullWaveHuygens(np.asarray([1.0]), np.asarray([[0.0, 0.0, 1.0]])),
    )
    with pytest.raises(ValueError, match="Periodic images|does not fit"):
        plan.prepare(_single(plan, plan.undulator_span[0] - 0.01, 1.0e-16))
