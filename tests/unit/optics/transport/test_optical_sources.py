#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cherenkov and scintillation photon sources from charged steps.

Independent references: the Frank--Tamm formula with CODATA 2022 ``alpha``
(Frank and Tamm 1937; Jackson, Classical Electrodynamics, 3rd ed., §
13.5), adaptive ``scipy.integrate.quad`` for dispersive and path integrals,
Birks (1951) quenching, and the bi-exponential scintillation time law. The
Poisson and Kolmogorov--Smirnov tolerances are five standard deviations or a
``1e-3`` p-value floor at fixed seeds.
"""

from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy import integrate, stats

import phydrax as phx
from phydrax.optics.geometric import NonSequentialSurfaceKind
from phydrax.optics.transport import (
    charged_steps_from_trajectory,
    charged_steps_from_transport,
    ChargedOpticalSteps,
    CherenkovEmission,
    detect_optical_arrivals,
    emit_optical_photons,
    ExplicitPhotonSource,
    OpticalEmissionProcess,
    OpticalEmissionStatus,
    OpticalMonteCarloPlan,
    OpticalPhotodetector,
    OpticalPhotonSourcePlan,
    PhotodetectionPlan,
    prepare_optical_monte_carlo,
    ScintillationEmission,
    simulate_optical_photons,
    SpectralOpticalMedium,
    TissueOpticalMedium,
)


# CODATA 2022 fine-structure constant; the package derives alpha from its
# scale contract's constants, which agree to about 1e-9 relative.
_ALPHA = 7.2973525643e-3
_RELATIVITY = phx.RelativityScaleContract.si()
_C = 299_792_458.0
_BAND = (300e-9, 600e-9)


def _water(index: float = 1.33) -> TissueOpticalMedium:
    return TissueOpticalMedium(
        np.zeros(1), np.zeros(1), np.zeros(1), np.asarray((index,))
    )


def _dispersive_medium() -> SpectralOpticalMedium:
    grid = np.linspace(250e-9, 650e-9, 81)
    index = 1.36 - 0.03 * (grid - 300e-9) / 300e-9
    return SpectralOpticalMedium(grid, index[None], np.full((1, grid.size), np.inf))


def _cherenkov_plan(
    medium: Any, nodes: np.ndarray, capacity: int = 50_000
) -> OpticalPhotonSourcePlan:
    return OpticalPhotonSourcePlan(
        relativity=_RELATIVITY,
        photon_capacity=capacity,
        cherenkov=CherenkovEmission(medium, nodes),
    )


def _straight_steps(
    start_beta: np.ndarray,
    end_beta: np.ndarray,
    lengths: np.ndarray,
    *,
    ids: np.ndarray | None = None,
    deposits: np.ndarray | None = None,
) -> ChargedOpticalSteps:
    """Parents along +z; step k starts where and when step k - 1 ends.

    A step with speed linear in path takes ``L ln(b1 / b0) / (c (b1 - b0))``.
    """

    parents, slots = start_beta.shape
    offsets = np.concatenate(
        (np.zeros((parents, 1)), np.cumsum(lengths, axis=1)[:, :-1]), axis=1
    )
    starts = np.zeros((parents, slots, 3))
    starts[..., 2] = offsets
    ends = starts.copy()
    ends[..., 2] += lengths
    words = np.arange(parents, dtype=np.uint32) if ids is None else ids
    spread = end_beta - start_beta
    same = np.abs(spread) < 1e-15
    transit = np.where(
        same,
        lengths / (_C * start_beta),
        lengths * np.log(end_beta / start_beta) / (_C * np.where(same, 1.0, spread)),
    )
    times = np.concatenate(
        (np.zeros((parents, 1)), np.cumsum(transit, axis=1)[:, :-1]), axis=1
    )
    return ChargedOpticalSteps(
        starts,
        ends,
        times,
        start_beta,
        end_beta,
        np.zeros((parents, slots), dtype=np.int32),
        np.ones((parents, slots), dtype=np.bool_),
        (np.zeros(parents, dtype=np.uint32), words.astype(np.uint32)),
        speed_of_light=_C,
        deposited_energy=deposits,
    )


def _frank_tamm_per_length(beta: float, index: float) -> float:
    return (
        2.0
        * np.pi
        * _ALPHA
        * (1.0 / _BAND[0] - 1.0 / _BAND[1])
        * (1.0 - 1.0 / (beta * beta * index * index))
    )


def _active(emission: Any) -> np.ndarray:
    return np.asarray(emission.active)


def test_water_frank_tamm_yield_per_centimeter() -> None:
    parents, beta = 64, 0.99999
    plan = _cherenkov_plan(_water(), np.linspace(*_BAND, 513))
    steps = _straight_steps(
        np.full((parents, 4), beta),
        np.full((parents, 4), beta),
        np.full((parents, 4), 0.0025),
    )
    emission = emit_optical_photons(plan, steps, jr.key(0))
    reference = _frank_tamm_per_length(beta, 1.33) * 0.01

    assert int(emission.status) == int(OpticalEmissionStatus.SUCCESS)
    np.testing.assert_allclose(emission.cherenkov_expected, reference, rtol=2e-5)
    counts = np.asarray(emission.cherenkov_counts, dtype=np.float64)
    assert abs(counts.mean() - reference) < 5.0 * np.sqrt(reference / parents)
    assert int(emission.allocated_count) == int(counts.sum())
    assert int(_active(emission).sum()) == int(counts.sum())
    wavelengths = np.asarray(emission.state.wavelengths)[_active(emission)]
    assert wavelengths.min() >= _BAND[0] and wavelengths.max() <= _BAND[1]


def test_below_threshold_steps_emit_nothing() -> None:
    plan = _cherenkov_plan(_dispersive_medium(), np.linspace(*_BAND, 65))
    beta = np.full((8, 3), 0.7)  # beta * n_max = 0.952 < 1
    emission = emit_optical_photons(
        plan, _straight_steps(beta, beta, np.full((8, 3), 0.01)), jr.key(1)
    )

    assert bool(emission.successful)
    np.testing.assert_array_equal(emission.cherenkov_expected, 0.0)
    assert int(emission.requested_count) == 0
    assert not _active(emission).any()
    np.testing.assert_array_equal(emission.state.weights, 0.0)


def test_cone_angle_follows_local_speed_and_dispersion() -> None:
    nodes = np.linspace(*_BAND, 129)
    medium = _dispersive_medium()
    plan = _cherenkov_plan(medium, nodes)
    first, last, length = 0.95, 0.80, 0.03
    steps = _straight_steps(
        np.full((16, 1), first), np.full((16, 1), last), np.full((16, 1), length)
    )
    emission = emit_optical_photons(plan, steps, jr.key(2))
    active = _active(emission)
    wavelengths = np.asarray(emission.state.wavelengths)[active]
    positions = np.asarray(emission.state.positions)[active]
    directions = np.asarray(emission.state.directions)[active]
    index = np.interp(wavelengths, nodes, 1.36 - 0.03 * (nodes - 300e-9) / 300e-9)
    speed = first + (last - first) * positions[:, 2] / length

    assert active.sum() > 500
    np.testing.assert_allclose(directions[:, 2], 1.0 / (speed * index), atol=1e-9)
    assert np.all(speed * index >= 1.0 - 1e-12)
    azimuth = np.arctan2(directions[:, 1], directions[:, 0])
    assert stats.kstest(azimuth, stats.uniform(-np.pi, 2 * np.pi).cdf).pvalue > 1e-3


def test_cherenkov_polarization_is_transverse_and_radial() -> None:
    plan = _cherenkov_plan(_water(), np.linspace(*_BAND, 33))
    beta = np.full((4, 1), 0.99)
    emission = emit_optical_photons(
        plan, _straight_steps(beta, beta, np.full((4, 1), 0.01)), jr.key(3)
    )
    active = _active(emission)
    direction = np.asarray(emission.state.directions)[active]
    field = np.asarray(emission.state.transverse_axes)[active]
    jones = np.asarray(emission.state.jones_vectors)[active]
    axis = np.asarray((0.0, 0.0, 1.0))
    expected = np.cross(direction, np.cross(direction, axis))
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)

    np.testing.assert_allclose(np.sum(field * direction, axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(np.sum(field * expected, axis=-1)), 1.0, atol=1e-12)
    # Radial: the field lies in the (track, photon) plane, pointing off-axis.
    np.testing.assert_allclose(
        np.sum(field * np.cross(direction, axis), axis=-1), 0.0, atol=1e-12
    )
    assert np.all(np.sum(field * axis, axis=-1) < 0.0)
    np.testing.assert_allclose(jones, np.tile((1.0 + 0j, 0.0 + 0j), (active.sum(), 1)))


def test_dispersive_band_integral_and_spectrum_match_quadrature() -> None:
    nodes = np.linspace(*_BAND, 513)
    beta = 0.745  # beta * n crosses one inside the band.
    table = 1.36 - 0.03 * (nodes - 300e-9) / 300e-9

    def density(wavelength: float) -> float:
        index = np.interp(wavelength, nodes, table)
        return (
            2.0
            * np.pi
            * _ALPHA
            / wavelength**2
            * max(1.0 - 1.0 / (beta * beta * index * index), 0.0)
        )

    kink = nodes[np.argmin(np.abs(table * beta - 1.0))]
    reference = 0.05 * sum(
        integrate.quad(density, a, b, epsabs=0.0, epsrel=1e-12, limit=400)[0]
        for a, b in ((_BAND[0], kink), (kink, _BAND[1]))
    )
    plan = _cherenkov_plan(_dispersive_medium(), nodes, capacity=60_000)
    speeds = np.full((256, 1), beta)
    emission = emit_optical_photons(
        plan, _straight_steps(speeds, speeds, np.full((256, 1), 0.05)), jr.key(4)
    )

    np.testing.assert_allclose(emission.cherenkov_expected, reference, rtol=1e-4)
    wavelengths = np.asarray(emission.state.wavelengths)[_active(emission)]
    grid = np.linspace(*_BAND, 2001)
    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(
                [
                    integrate.quad(density, a, b)[0]
                    for a, b in zip(grid.tolist()[:-1], grid.tolist()[1:])
                ]
            ),
        )
    )
    cdf = cumulative / cumulative[-1]
    assert wavelengths.size > 1000
    assert stats.kstest(wavelengths, lambda x: np.interp(x, grid, cdf)).pvalue > 1e-3


def test_step_spectral_yield_integrates_linear_speed_exactly() -> None:
    length, first, last, index, wavelength = 0.02, 0.9, 0.7, 1.33, 400e-9
    density, valid = phx.equations.cherenkov_step_spectral_yield(
        length, first, last, index, wavelength
    )

    def integrand(s: float) -> float:
        speed = first + (last - first) * s / length
        return max(1.0 - 1.0 / (speed * speed * index * index), 0.0)

    threshold = length * (1.0 / index - first) / (last - first)
    path = integrate.quad(integrand, 0.0, threshold, epsrel=1e-13)[0]
    assert bool(valid)
    # This checks the exact path integral, so the prefactor uses the package's
    # own alpha (SI scale contract); CODATA 2022 alpha differs by 6.2e-10
    # relative, far above the 1e-13 quadrature error bound.
    alpha = float(phx.ElectromagneticScaleContract.si().fine_structure)
    np.testing.assert_allclose(
        density, 2.0 * np.pi * alpha / wavelength**2 * path, rtol=1e-12
    )
    invalid, flag = phx.equations.cherenkov_step_spectral_yield(
        length, 1.2, 0.5, index, wavelength
    )
    assert not bool(flag) and np.isnan(float(invalid))


def _scintillator(
    *, birks: float = 1.26e-10, photon_yield: float = 1e-3
) -> ScintillationEmission:
    grid = np.asarray((380e-9, 420e-9, 480e-9))
    spectra = np.asarray((((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),))
    return ScintillationEmission(
        np.asarray((photon_yield,)),
        np.asarray((birks,)),
        np.asarray(((0.7, 0.3),)),
        np.asarray(((0.5e-9, 0.0),)),
        np.asarray(((2.1e-9, 14e-9),)),
        grid,
        spectra,
    )


def _scintillation_plan(
    emission: ScintillationEmission, capacity: int
) -> OpticalPhotonSourcePlan:
    return OpticalPhotonSourcePlan(
        relativity=_RELATIVITY, photon_capacity=capacity, scintillation=emission
    )


def test_scintillation_birks_mean_yield() -> None:
    parents, deposit, length, birks = 128, 2.0e5, 1.0e-3, 1.26e-10
    plan = _scintillation_plan(_scintillator(birks=birks), 40_000)
    speed = np.full((parents, 1), 0.5)
    steps = _straight_steps(
        speed,
        speed,
        np.full((parents, 1), length),
        deposits=np.full((parents, 1), deposit),
    )
    emission = emit_optical_photons(plan, steps, jr.key(5))
    visible = deposit / (1.0 + birks * deposit / length)
    mean = 1e-3 * visible

    assert bool(emission.successful)
    np.testing.assert_allclose(emission.visible_energy, visible, rtol=1e-14)
    np.testing.assert_allclose(emission.deposited_energy, deposit)
    np.testing.assert_allclose(emission.scintillation_expected, mean, rtol=1e-14)
    counts = np.asarray(emission.scintillation_counts, dtype=np.float64)
    assert abs(counts.mean() - mean) < 5.0 * np.sqrt(mean / parents)
    processes = np.asarray(emission.processes)[_active(emission)]
    assert np.all(processes == int(OpticalEmissionProcess.SCINTILLATION))


def test_scintillation_timing_and_spectrum_follow_declared_laws() -> None:
    plan = _scintillation_plan(_scintillator(birks=0.0), 30_000)
    speed = np.full((64, 1), 0.5)
    steps = _straight_steps(
        speed, speed, np.full((64, 1), 1e-9), deposits=np.full((64, 1), 4.0e5)
    )
    emission = emit_optical_photons(plan, steps, jr.key(6))
    active = _active(emission)
    times = np.asarray(emission.state.times)[active]
    fractions, rise, decay = (0.7, 0.3), (0.5e-9, 0.0), (2.1e-9, 14e-9)

    def cdf(t: np.ndarray) -> np.ndarray:
        total = np.zeros_like(t)
        for weight, tau_r, tau_d in zip(fractions, rise, decay):
            tail = (
                tau_d * np.exp(-t / tau_d)
                - tau_r * np.exp(-t / tau_r if tau_r else -np.inf)
            ) / (tau_d - tau_r)
            total += weight * (1.0 - tail)
        return total

    mean = sum(w * (r + d) for w, r, d in zip(fractions, rise, decay))
    second = sum(
        w * (2 * r * r + 2 * d * d + 2 * r * d) for w, r, d in zip(fractions, rise, decay)
    )
    assert times.size > 20_000
    np.testing.assert_allclose(
        times.mean(), mean, rtol=5 * np.sqrt(second / times.size) / mean
    )
    assert stats.kstest(times, cdf).pvalue > 1e-3
    wavelengths = np.asarray(emission.state.wavelengths)[active]

    def triangle(x: np.ndarray) -> np.ndarray:
        left = (x - 380e-9) ** 2 / (40e-9 * 100e-9)
        right = 1.0 - (480e-9 - x) ** 2 / (60e-9 * 100e-9)
        return np.clip(np.where(x < 420e-9, left, right), 0.0, 1.0)

    assert stats.kstest(wavelengths, triangle).pvalue > 1e-3
    direction = np.asarray(emission.state.directions)[active]
    field = np.asarray(emission.state.transverse_axes)[active]
    assert stats.kstest(direction[:, 2], stats.uniform(-1.0, 2.0).cdf).pvalue > 1e-3
    np.testing.assert_allclose(np.sum(field * direction, axis=-1), 0.0, atol=1e-12)


def test_emission_is_invariant_to_step_subdivision() -> None:
    nodes = np.linspace(*_BAND, 65)
    plan = OpticalPhotonSourcePlan(
        relativity=_RELATIVITY,
        photon_capacity=150_000,
        cherenkov=CherenkovEmission(_water(), nodes),
        scintillation=_scintillator(photon_yield=1e-4),
    )
    parents, length, first, last, deposit = 96, 0.02, 0.99, 0.72, 4.0e6
    whole = _straight_steps(
        np.full((parents, 1), first),
        np.full((parents, 1), last),
        np.full((parents, 1), length),
        deposits=np.full((parents, 1), deposit),
    )
    cuts = np.linspace(0.0, 1.0, 5)
    speeds = first + (last - first) * cuts
    split = _straight_steps(
        np.tile(speeds[:-1], (parents, 1)),
        np.tile(speeds[1:], (parents, 1)),
        np.full((parents, 4), length / 4),
        ids=np.arange(parents, dtype=np.uint32) + 1000,
        deposits=np.full((parents, 4), deposit / 4),
    )
    one = emit_optical_photons(plan, whole, jr.key(7))
    four = emit_optical_photons(plan, split, jr.key(7))

    assert bool(one.successful) and bool(four.successful)
    np.testing.assert_allclose(
        four.cherenkov_expected, one.cherenkov_expected, rtol=1e-12
    )
    np.testing.assert_allclose(
        four.scintillation_expected, one.scintillation_expected, rtol=1e-12
    )
    for counts in ("cherenkov_counts", "scintillation_counts"):
        a = np.asarray(getattr(one, counts), dtype=np.float64)
        b = np.asarray(getattr(four, counts), dtype=np.float64)
        assert abs(a.mean() - b.mean()) < 5.0 * np.sqrt((a.var() + b.var()) / parents)

    def cherenkov(emission: Any) -> np.ndarray:
        mask = _active(emission) & (
            np.asarray(emission.processes) == int(OpticalEmissionProcess.CHERENKOV)
        )
        return np.asarray(emission.state.positions)[mask]

    left, right = cherenkov(one), cherenkov(four)
    assert stats.ks_2samp(left[:, 2], right[:, 2]).pvalue > 1e-3
    left_times = np.asarray(one.state.times)[_active(one)]
    right_times = np.asarray(four.state.times)[_active(four)]
    assert stats.ks_2samp(left_times, right_times).pvalue > 1e-3


def test_emission_is_invariant_to_parent_order() -> None:
    plan = OpticalPhotonSourcePlan(
        relativity=_RELATIVITY,
        photon_capacity=20_000,
        cherenkov=CherenkovEmission(_water(), np.linspace(*_BAND, 33)),
        scintillation=_scintillator(),
        batch_size=7,
    )
    rng = np.random.default_rng(8)
    first = rng.uniform(0.8, 0.99, (6, 3))
    last = first - rng.uniform(0.0, 0.1, (6, 3))
    lengths = rng.uniform(1e-3, 5e-3, (6, 3))
    deposits = rng.uniform(1e4, 1e5, (6, 3))
    ids = np.asarray((9, 3, 12, 5, 1, 7), dtype=np.uint32)
    permutation = np.asarray((4, 2, 5, 0, 3, 1))
    base = emit_optical_photons(
        plan, _straight_steps(first, last, lengths, ids=ids, deposits=deposits), jr.key(9)
    )
    shuffled = emit_optical_photons(
        plan,
        _straight_steps(
            first[permutation],
            last[permutation],
            lengths[permutation],
            ids=ids[permutation],
            deposits=deposits[permutation],
        ),
        jr.key(9),
    )

    assert bool(base.successful) and int(base.allocated_count) > 100
    for field in ("positions", "directions", "transverse_axes", "wavelengths", "times"):
        np.testing.assert_allclose(
            getattr(shuffled.state, field), getattr(base.state, field), atol=1e-15
        )
    np.testing.assert_array_equal(shuffled.state.id_lo, base.state.id_lo)
    np.testing.assert_array_equal(shuffled.parent_lo, base.parent_lo)
    np.testing.assert_array_equal(shuffled.parent_steps, base.parent_steps)
    np.testing.assert_array_equal(
        shuffled.cherenkov_counts, np.asarray(base.cherenkov_counts)[permutation]
    )
    np.testing.assert_allclose(
        shuffled.scintillation_photon_energy,
        np.asarray(base.scintillation_photon_energy)[permutation],
        rtol=1e-14,
    )


def test_identity_lineage_and_energy_ledger() -> None:
    plan = _cherenkov_plan(_water(), np.linspace(*_BAND, 33), capacity=4000)
    speed = np.full((3, 2), 0.99)
    ids = np.asarray((20, 10, 30), dtype=np.uint32)
    first_identity = (1, 2**32 - 5)
    emission = emit_optical_photons(
        plan,
        _straight_steps(speed, speed, np.full((3, 2), 0.002), ids=ids),
        jr.key(10),
        first_identity=first_identity,
    )
    active = _active(emission)
    count = int(emission.allocated_count)
    identity = (
        np.asarray(emission.state.id_hi, dtype=np.uint64)[active] << np.uint64(32)
    ) | np.asarray(emission.state.id_lo, dtype=np.uint64)[active]
    first = 2**32 + 2**32 - 5
    parents = np.asarray(emission.parent_lo)[active]

    assert active[:count].all() and not active[count:].any()
    np.testing.assert_array_equal(
        identity, np.arange(first, first + count, dtype=np.uint64)
    )
    assert np.all(np.diff(parents) >= 0) and set(parents) <= {10, 20, 30}
    next_identity = (int(emission.next_id_hi) << 32) | int(emission.next_id_lo)
    assert next_identity == first + count
    # h c / lambda with the exact SI Planck constant.
    photon_energy = 6.62607015e-34 * _C / np.asarray(emission.state.wavelengths)[active]
    for slot, word in enumerate(ids):
        np.testing.assert_allclose(
            emission.cherenkov_photon_energy[slot],
            photon_energy[parents == word].sum(),
            rtol=1e-12,
        )
        assert int(emission.cherenkov_counts[slot]) == int(np.sum(parents == word))


def test_capacity_refusal_is_atomic() -> None:
    nodes = np.linspace(*_BAND, 33)
    speed = np.full((4, 2), 0.99)
    steps = _straight_steps(speed, speed, np.full((4, 2), 0.005))
    roomy = emit_optical_photons(
        _cherenkov_plan(_water(), nodes, 10_000), steps, jr.key(11)
    )
    tight = emit_optical_photons(_cherenkov_plan(_water(), nodes, 64), steps, jr.key(11))

    assert int(roomy.requested_count) > 64
    assert int(tight.status) == int(OpticalEmissionStatus.CAPACITY_EXHAUSTED)
    assert not bool(tight.successful)
    assert int(tight.requested_count) == int(roomy.requested_count)
    assert int(tight.allocated_count) == 0
    assert not _active(tight).any()
    np.testing.assert_array_equal(tight.state.weights, 0.0)
    np.testing.assert_array_equal(tight.cherenkov_photon_energy, 0.0)
    np.testing.assert_array_equal(tight.cherenkov_counts, roomy.cherenkov_counts)
    assert (int(tight.next_id_hi), int(tight.next_id_lo)) == (0, 0)


def test_invalid_and_exhausted_inputs_refuse_the_bank() -> None:
    plan = _cherenkov_plan(_water(), np.linspace(*_BAND, 33), capacity=10_000)
    speed = np.full((2, 1), 0.99)
    duplicate = _straight_steps(
        speed, speed, np.full((2, 1), 0.01), ids=np.zeros(2, dtype=np.uint32)
    )
    near_reserved = emit_optical_photons(
        plan,
        _straight_steps(speed, speed, np.full((2, 1), 0.01)),
        jr.key(12),
        first_identity=(2**32 - 1, 2**32 - 10),
    )

    assert int(emit_optical_photons(plan, duplicate, jr.key(12)).status) == int(
        OpticalEmissionStatus.INVALID_STEPS
    )
    assert int(near_reserved.status) == int(OpticalEmissionStatus.IDENTITY_EXHAUSTED)
    assert not _active(near_reserved).any()
    with pytest.raises(ValueError, match="deposited energy"):
        emit_optical_photons(
            _scintillation_plan(_scintillator(), 100),
            _straight_steps(speed, speed, np.full((2, 1), 0.01)),
            jr.key(12),
        )


def test_trajectory_lanes_become_optical_steps() -> None:
    scale = phx.ElectromagneticScaleContract.si()
    beta, dt = 0.999, 1e-11
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    samples = 5
    times = np.arange(samples) * dt
    positions = np.zeros((samples, 1, 3))
    positions[:, 0, 2] = beta * _C * times
    velocities = np.zeros((samples, 1, 3))
    velocities[:, 0, 2] = gamma * beta * _C
    trajectory = phx.electromagnetics.ChargedTrajectory(
        times,
        positions,
        velocities,
        np.asarray((-float(scale.elementary_charge),)),
        np.ones(1),
        np.ones((samples, 1), dtype=np.bool_),
        (np.zeros(1, dtype=np.uint32), np.full(1, 7, dtype=np.uint32)),
    )
    steps = charged_steps_from_trajectory(trajectory, scale, 0)
    plan = _cherenkov_plan(_water(), np.linspace(*_BAND, 513), capacity=1000)
    emission = emit_optical_photons(plan, steps, jr.key(13))
    path = beta * _C * dt * (samples - 1)

    np.testing.assert_allclose(steps.start_beta, beta, rtol=1e-12)
    np.testing.assert_allclose(steps.charge_numbers, -1.0)
    np.testing.assert_allclose(steps.start_times[0], times[:-1])
    np.testing.assert_allclose(
        emission.cherenkov_expected, _frank_tamm_per_length(beta, 1.33) * path, rtol=2e-5
    )
    active = _active(emission)
    emitted = np.asarray(emission.state.times)[active]
    z = np.asarray(emission.state.positions)[active, 2]
    np.testing.assert_allclose(emitted, z / (beta * _C), rtol=1e-9, atol=1e-22)


def _manifest() -> Any:
    return phx.qualification.ReferenceArtifactManifest(
        "optical-source-water",
        checksum_algorithm="sha256",
        checksum="f" * 64,
        size_bytes=1,
        license_id="synthetic-permissive",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy_eV": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("synthetic:water",),
    )


def _charged_water_transport(step_bank_capacity: int) -> Any:
    materials = phx.equations.ChargedRadiationMaterialLibrary(
        jnp.asarray((10.0, 2.0e7)),
        jnp.full((1, 2), 2.0e8),  # ~2 MeV/cm collision stopping power.
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        ("water",),
        _manifest(),
    )
    geometry = phx.discretization.VoxelRadiationGeometryPlan(
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((0.04, 0.04, 0.04)),
        jnp.zeros((1, 1, 1), dtype=jnp.int32),
        material_count=1,
    )
    return phx.solver.ChargedParticleTransportPlan(
        geometry,
        materials,
        maximum_steps=512,
        maximum_step_length=1e-3,
        maximum_fractional_energy_loss=0.05,
        cutoff_energy_ev=1.0e4,
        step_bank_capacity=step_bank_capacity,
    )


def _electron(plan: Any) -> Any:
    return plan.simulate(
        jnp.asarray(((0.02, 0.02, 0.005),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((3.0e6,)),
        jnp.asarray((int(phx.equations.ChargedRadiationParticleKind.ELECTRON),)),
        jr.key(14),
    )


def test_incomplete_step_bank_refuses_emission() -> None:
    steps = charged_steps_from_transport(
        _electron(_charged_water_transport(4)), (0,), _RELATIVITY
    )
    emission = emit_optical_photons(
        _cherenkov_plan(_water(), np.linspace(*_BAND, 33), capacity=5000),
        steps,
        jr.key(15),
    )

    assert not bool(steps.complete[0])
    assert int(emission.status) & int(OpticalEmissionStatus.INCOMPLETE_STEPS)
    assert not _active(emission).any()


def _water_cube_surfaces() -> Any:
    low, high = np.zeros(3), np.full(3, 0.04)
    vertices, triangles = [], []
    for axis in range(3):
        for side, bound in enumerate((low, high)):
            u, v = (value for value in range(3) if value != axis)
            corners = []
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                corner = np.empty(3)
                corner[axis], corner[u], corner[v] = (
                    bound[axis],
                    (low, high)[a][u],
                    (low, high)[b][v],
                )
                corners.append(corner)
            outward = np.zeros(3)
            outward[axis] = 1.0 if side else -1.0
            base = len(vertices)
            vertices.extend(corners)
            for triangle in ((0, 1, 2), (0, 2, 3)):
                p = [corners[index] for index in triangle]
                forward = np.cross(p[1] - p[0], p[2] - p[0]) @ outward > 0.0
                triangles.append(
                    [base + i for i in (triangle if forward else triangle[::-1])]
                )
    return phx.optics.geometric.NonSequentialSurfaceTable(
        np.asarray(vertices),
        np.asarray(triangles),
        np.zeros(12, dtype=np.int32),
        np.ones(12, dtype=np.int32),
        np.asarray((1.33, 1.0)),
        surface_ids=np.repeat(np.arange(6), 2),
        surface_kinds=np.full(12, int(NonSequentialSurfaceKind.DETECTOR)),
        detector_indices=np.zeros(12, dtype=np.int32),
    )


def test_charged_steps_drive_optical_transport_to_detector_hits() -> None:
    transport = _electron(_charged_water_transport(512))
    steps = charged_steps_from_transport(transport, (0,), _RELATIVITY)
    medium = TissueOpticalMedium(
        np.zeros(2), np.zeros(2), np.zeros(2), np.asarray((1.33, 1.0))
    )
    emission = emit_optical_photons(
        _cherenkov_plan(medium, np.linspace(*_BAND, 65), capacity=4096),
        steps,
        jr.key(16),
    )
    prepared = prepare_optical_monte_carlo(
        OpticalMonteCarloPlan(
            _water_cube_surfaces(),
            medium,
            relativity=_RELATIVITY,
            maximum_interactions=8,
            detector_arrival_capacity=1,
            photon_batch_size=1024,
        )
    )
    optical = simulate_optical_photons(
        prepared, ExplicitPhotonSource(emission.state), jr.key(17)
    )
    efficiency = np.asarray((0.2, 0.25, 0.3, 0.25, 0.2))
    nodes = np.linspace(*_BAND, 5)
    detection = detect_optical_arrivals(
        PhotodetectionPlan(
            OpticalPhotodetector(nodes, efficiency[None], np.zeros((1, 3))),
            phx.applications.detector.SensitiveHitPlan(
                np.asarray((0,)), channel_count=1, conditions_id="water-cube"
            ),
            gate=(0.0, 1e-6),
            hit_capacity=4096,
        ),
        optical.detector_arrivals,
        jr.key(18),
        event_ids=np.asarray((0,)),
        photon_events=np.zeros(emission.state.weights.shape[0], dtype=np.int64),
    )
    arrivals = optical.detector_arrivals
    arrived = np.asarray(arrivals.active)[:, 0]
    wavelengths = np.asarray(arrivals.wavelengths)[:, 0][arrived]

    assert bool(transport.all_successful) and bool(emission.successful)
    assert int(emission.allocated_count) > 200
    assert bool(optical.all_successful) and bool(detection.all_successful)
    np.testing.assert_array_equal(arrived, _active(emission))
    np.testing.assert_allclose(
        detection.expected_photoelectrons[0, 0],
        np.interp(wavelengths, nodes, efficiency).sum(),
        rtol=1e-12,
    )
    hits = int(detection.photoelectron_counts[0, 0])
    expected = float(detection.expected_photoelectrons[0, 0])
    assert abs(hits - expected) < 5.0 * np.sqrt(expected)
    assert np.all(
        np.asarray(arrivals.times)[:, 0][arrived]
        >= np.asarray(emission.state.times)[_active(emission)]
    )
