#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.applications.detector import SensitiveHitPlan
from phydrax.optics.transport import (
    detect_optical_arrivals,
    OpticalDetectorArrivals,
    OpticalPhotodetector,
    PhotodetectionPlan,
    PhotodetectionResult,
    PhotodetectionStatus,
)


_NODES = np.asarray([300e-9, 400e-9, 500e-9, 600e-9])
_QE = np.asarray([0.1, 0.3, 0.2, 0.05])


def _arrivals(
    detectors: np.ndarray,
    times: np.ndarray,
    *,
    wavelength: float = 450e-9,
    weight: float = 1.0,
    active: bool = True,
    dropped: int = 0,
) -> OpticalDetectorArrivals:
    """One arrival slot per photon with identities ``0, 1, ...``."""

    count = detectors.shape[0]
    identities = np.arange(count, dtype=np.uint64) + np.uint64(3 << 32)
    return OpticalDetectorArrivals(
        jnp.asarray(detectors.reshape(count, 1), dtype=jnp.int32),
        jnp.asarray(np.tile(np.asarray([0.0, 0.0, 1.0]), (count, 1, 1))),
        jnp.asarray(times.reshape(count, 1), dtype=jnp.float64),
        jnp.full((count, 1), wavelength),
        jnp.full((count, 1), weight),
        jnp.ones((count, 1)),
        jnp.full((count, 1), active),
        jnp.full((count,), dropped, dtype=jnp.int32),
        jnp.asarray((identities >> np.uint64(32)).astype(np.uint32)),
        jnp.asarray((identities & np.uint64(0xFFFFFFFF)).astype(np.uint32)),
    )


def _plan(
    *,
    detectors: int = 1,
    hit_capacity: int = 64,
    dark_count_capacity: int = 0,
    gate: tuple[float, float] = (0.0, 1e-6),
    **photodetector: Any,
) -> PhotodetectionPlan:
    return PhotodetectionPlan(
        OpticalPhotodetector(_NODES, _QE, np.zeros((detectors, 3)), **photodetector),
        SensitiveHitPlan(
            np.arange(detectors), channel_count=detectors, conditions_id="optical-run"
        ),
        gate=gate,
        hit_capacity=hit_capacity,
        dark_count_capacity=dark_count_capacity,
    )


def _detect(
    plan: PhotodetectionPlan,
    arrivals: OpticalDetectorArrivals,
    events: int,
    *,
    seed: int = 0,
) -> PhotodetectionResult:
    photons = arrivals.detector_indices.shape[0]
    return detect_optical_arrivals(
        plan,
        arrivals,
        jr.key(seed),
        event_ids=np.arange(100, 100 + events),
        photon_events=np.arange(photons) % events,
    )


def test_photoelectron_counts_are_binomial_in_qe_times_collection() -> None:
    events, per_event = 400, 50
    photons = events * per_event
    plan = _plan(collection_efficiency=0.8)
    result = _detect(plan, _arrivals(np.zeros(photons), np.full(photons, 1e-8)), events)
    # QE is linear between the 400 nm (0.3) and 500 nm (0.2) nodes.
    probability = 0.25 * 0.8
    counts = np.asarray(result.photoelectron_counts)[:, 0]
    np.testing.assert_allclose(
        np.asarray(result.expected_photoelectrons)[:, 0], per_event * probability
    )
    mean = per_event * probability
    variance = mean * (1.0 - probability)
    assert abs(np.mean(counts) - mean) < 5.0 * np.sqrt(variance / events)
    assert abs(np.var(counts, ddof=1) / variance - 1.0) < 5.0 * np.sqrt(2.0 / events)
    np.testing.assert_array_equal(np.sum(np.asarray(result.hits.active), axis=-1), counts)
    assert bool(result.all_successful)


def test_transit_time_spread_and_single_photoelectron_charge_moments() -> None:
    photons = 20000
    # Recorded weight 4 at QE 0.25 makes every arrival a photoelectron.
    plan = _plan(
        hit_capacity=photons,
        transit_times=40e-9,
        transit_time_spreads=0.6e-9,
        single_photoelectron_charges=1.6,
        single_photoelectron_spreads=0.5,
    )
    arrival_time = 5e-9
    arrivals = _arrivals(np.zeros(photons), np.full(photons, arrival_time), weight=4.0)
    result = _detect(plan, arrivals, 1)
    active = np.asarray(result.hits.active)[0]
    assert np.sum(active) == photons
    delays = np.asarray(result.hits.times)[0] - arrival_time
    charges = np.asarray(result.hits.energies)[0]
    assert abs(np.mean(delays) - 40e-9) < 5.0 * 0.6e-9 / np.sqrt(photons)
    assert abs(np.std(delays, ddof=1) / 0.6e-9 - 1.0) < 5.0 * np.sqrt(0.5 / photons)
    assert abs(np.mean(charges) - 1.6) < 5.0 * 0.5 / np.sqrt(photons)
    # Gamma law: relative variance of the sample variance is 2/N + excess/N
    # with excess kurtosis 6 / shape.
    shape = (1.6 / 0.5) ** 2
    spread = np.sqrt((2.0 + 6.0 / shape) / photons)
    assert abs(np.var(charges, ddof=1) / 0.25 - 1.0) < 5.0 * spread
    assert np.all(charges > 0.0)
    assert np.all(np.diff(np.asarray(result.hits.times)[0]) >= 0.0)


def test_dark_counts_are_poisson_and_uniform_in_the_gate() -> None:
    events = 2000
    rate, window = 2.0e8, 50e-9
    plan = _plan(
        detectors=2,
        gate=(0.0, window),
        dark_count_rates=(rate, 0.5 * rate),
        dark_count_capacity=64,
    )
    arrivals = _arrivals(np.zeros(events), np.zeros(events), active=False)
    result = _detect(plan, arrivals, events, seed=4)
    counts = np.asarray(result.dark_counts)
    np.testing.assert_allclose(
        np.asarray(result.expected_dark_counts), (rate * window, 0.5 * rate * window)
    )
    for column, mean in enumerate((rate * window, 0.5 * rate * window)):
        assert abs(np.mean(counts[:, column]) - mean) < 5.0 * np.sqrt(mean / events)
        dispersion = np.var(counts[:, column], ddof=1) / mean
        assert abs(dispersion - 1.0) < 5.0 * np.sqrt((2.0 + 1.0 / mean) / events)
    active = np.asarray(result.hits.active)
    assert np.all(np.asarray(result.dark) == active)
    np.testing.assert_array_equal(np.sum(active, axis=-1), np.sum(counts, axis=-1))
    times = np.asarray(result.hits.times)[active]
    assert abs(np.mean(times) - 0.5 * window) < 5.0 * window / np.sqrt(12.0 * times.size)
    assert np.all((times >= 0.0) & (times < window))
    assert np.all(np.asarray(result.photoelectron_counts) == 0)


def test_hit_banks_are_invariant_to_photon_order_and_event_slot_order() -> None:
    rng = np.random.default_rng(2)
    photons, events = 300, 6
    detectors = rng.integers(0, 3, photons)
    times = rng.uniform(0.0, 20e-9, photons)
    plan = _plan(
        detectors=3,
        dark_count_rates=1e8,
        dark_count_capacity=16,
        transit_time_spreads=1e-9,
        single_photoelectron_spreads=0.3,
    )
    arrivals = _arrivals(detectors, times)
    owners = np.arange(photons) % events
    event_ids = np.asarray([7, 11, 13, 17, 19, 23])
    reference = detect_optical_arrivals(
        plan, arrivals, jr.key(8), event_ids=event_ids, photon_events=owners
    )
    permutation = rng.permutation(photons)
    permuted = detect_optical_arrivals(
        plan,
        jax.tree_util.tree_map(lambda value: value[permutation], arrivals),
        jr.key(8),
        event_ids=event_ids,
        photon_events=owners[permutation],
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(reference),
        jax.tree_util.tree_leaves(permuted),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    reversed_events = event_ids[::-1]
    relabeled = detect_optical_arrivals(
        plan,
        arrivals,
        jr.key(8),
        event_ids=reversed_events,
        photon_events=events - 1 - owners,
    )
    np.testing.assert_array_equal(
        np.asarray(relabeled.hits.times)[::-1], np.asarray(reference.hits.times)
    )
    np.testing.assert_array_equal(
        np.asarray(relabeled.hits.energies)[::-1], np.asarray(reference.hits.energies)
    )
    assert np.sum(np.asarray(reference.dark)) > 0


@pytest.mark.parametrize(
    ("arrival", "plan_kwargs", "status"),
    [
        ({"dropped": 1}, {}, PhotodetectionStatus.INCOMPLETE_ARRIVALS),
        ({"wavelength": 700e-9}, {}, PhotodetectionStatus.UNSUPPORTED_WAVELENGTH),
        ({"weight": 5.0}, {}, PhotodetectionStatus.DETECTION_PROBABILITY_EXCEEDS_UNITY),
        (
            {"active": False},
            {"dark_count_rates": 1e10, "dark_count_capacity": 1},
            PhotodetectionStatus.DARK_COUNT_CAPACITY_EXHAUSTED,
        ),
        (
            {"weight": 4.0},
            {"hit_capacity": 1},
            PhotodetectionStatus.HIT_CAPACITY_EXHAUSTED,
        ),
    ],
    ids=["incomplete", "unsupported", "probability", "dark-capacity", "hit-capacity"],
)
def test_photodetection_failures_are_reported_per_event(
    arrival: dict[str, Any], plan_kwargs: dict[str, Any], status: PhotodetectionStatus
) -> None:
    arrivals = _arrivals(np.zeros(8), np.full(8, 1e-8), **arrival)
    result = _detect(_plan(**plan_kwargs), arrivals, 1)
    assert int(result.status[0]) == int(status)
    assert not bool(result.all_successful)
    if status == PhotodetectionStatus.HIT_CAPACITY_EXHAUSTED:
        assert int(result.dropped_hits[0]) == 7


def test_invalid_photodetector_and_plan_declarations_are_refused() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        OpticalPhotodetector(_NODES[::-1], _QE, np.zeros((1, 3)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        OpticalPhotodetector(_NODES, 1.5 * np.ones(4), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="positive"):
        OpticalPhotodetector(
            _NODES, _QE, np.zeros((1, 3)), single_photoelectron_charges=0.0
        )
    detector = OpticalPhotodetector(_NODES, _QE, np.zeros((2, 3)), dark_count_rates=1.0)
    hits = SensitiveHitPlan(np.arange(2), channel_count=2, conditions_id="optical-run")
    with pytest.raises(ValueError, match="dark_count_capacity"):
        PhotodetectionPlan(detector, hits, gate=(0.0, 1.0), hit_capacity=4)
    with pytest.raises(ValueError, match="one element per photodetector"):
        PhotodetectionPlan(
            detector,
            SensitiveHitPlan(np.arange(3), channel_count=3, conditions_id="optical-run"),
            gate=(0.0, 1.0),
            hit_capacity=4,
            dark_count_capacity=4,
        )
    with pytest.raises(ValueError, match="gate"):
        PhotodetectionPlan(
            detector, hits, gate=(1.0, 1.0), hit_capacity=4, dark_count_capacity=4
        )
    plan = _plan()
    with pytest.raises(ValueError, match="unique"):
        detect_optical_arrivals(
            plan,
            _arrivals(np.zeros(2), np.zeros(2)),
            jr.key(0),
            event_ids=np.asarray([1, 1]),
            photon_events=np.asarray([0, 1]),
        )
    with pytest.raises(ValueError, match="photon_events"):
        detect_optical_arrivals(
            plan,
            _arrivals(np.zeros(2), np.zeros(2)),
            jr.key(0),
            event_ids=np.asarray([1]),
            photon_events=np.asarray([0, 1]),
        )
