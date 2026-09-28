#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Photodetection of optical detector arrivals into canonical sensitive hits.

An accepted arrival of recorded weight ``w`` and wavelength ``lambda`` on
detector ``d`` releases a photoelectron with probability
``w · QE_d(lambda) · CE_d``: quantum efficiency interpolated linearly on the
declared wavelength nodes times the collection efficiency. For unit-weight
photons the photoelectron count is binomial; for implicit-capture weights the
Bernoulli draw keeps the expected count exact. Only photoelectrons emitted
inside the gate ``[start, end)`` are read out. The anode time is the emission
time plus the mean transit time and a Gaussian transit-time spread; the
charge follows the single-photoelectron gain distribution, a gamma (Polya)
law with the declared mean and standard deviation. Dark counts are Poisson
with mean ``rate_d · (end - start)`` per event and detector, uniform in the
gate, and pass through the same transit and gain. References: Hamamatsu
Photonics, *Photomultiplier Tubes: Basics and Applications*, 4th ed. (2017),
ch. 4; G. F. Knoll, *Radiation Detection and Measurement*, 4th ed. (2010),
ch. 9; J. R. Prescott, Nucl. Instrum. Methods 39, 173 (1966).

Hits are ordered within each event by anode time, then photon hits before
dark counts, then photon identity and arrival slot (dark: detector and
ordinal), so the bank is independent of photon order and batching. Every
draw is keyed by photon identity and arrival slot, or by event identity,
detector, and dark-count ordinal.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_stencil
from ..._sampling import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...applications.detector._core import SensitiveHitBank
from ...applications.detector._digitization import SensitiveHitPlan
from ...typing import ConvertibleToArray, Dim, Float64, parse, PRNGKey, Size
from ._optical_monte_carlo import OpticalDetectorArrivals


_DETECTION_ADDRESS = SampleAddress("optics", "photodetection", target="photon")
_DARK_COUNT_ADDRESS = SampleAddress("optics", "photodetection", target="dark-count")
_DARK_HIT_ADDRESS = SampleAddress("optics", "photodetection", target="dark-hit")
_WORD = 1 << 32


class _DetectorDim(Dim, minimum=1):
    """Photodetector elements."""


class _WavelengthDim(Dim, minimum=2):
    """Quantum-efficiency wavelength nodes."""


class PhotodetectionStatus(IntEnum):
    """Per-event photodetection status."""

    SUCCESS = 0
    INCOMPLETE_ARRIVALS = 1
    UNSUPPORTED_WAVELENGTH = 2
    DETECTION_PROBABILITY_EXCEEDS_UNITY = 3
    DARK_COUNT_CAPACITY_EXHAUSTED = 4
    HIT_CAPACITY_EXHAUSTED = 5
    NONFINITE_RESULT = 6


def _per_detector(value: ConvertibleToArray, name: str, count: int, /) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real-valued.")
    if array.ndim > 1 or (array.ndim == 1 and array.shape != (count,)):
        raise ValueError(f"{name} must be a scalar or have one value per detector.")
    result = np.broadcast_to(array.astype(np.float64), (count,)).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


class OpticalPhotodetector(StrictModule, NonTrainableState):
    """Photocathode, collection, timing, gain, and dark-count response per detector.

    Wavelengths and times use the optical transport's units; dark-count
    rates are per that time unit and charges are in the unit the
    digitization calibration consumes. ``reference_positions`` locate dark
    counts, which have no photon position.
    """

    __strict_contract__ = True

    wavelength_nodes: Float64[_WavelengthDim]
    quantum_efficiency: Float64[_DetectorDim, _WavelengthDim]
    collection_efficiency: Float64[_DetectorDim]
    transit_times: Float64[_DetectorDim]
    transit_time_spreads: Float64[_DetectorDim]
    single_photoelectron_charges: Float64[_DetectorDim]
    single_photoelectron_spreads: Float64[_DetectorDim]
    dark_count_rates: Float64[_DetectorDim]
    reference_positions: Float64[_DetectorDim, Literal[3]]
    detector_count: Size[_DetectorDim] = eqx.field(static=True)
    photodetector_id: str = eqx.field(static=True)

    def __init__(
        self,
        wavelength_nodes: ConvertibleToArray,
        quantum_efficiency: ConvertibleToArray,
        reference_positions: ConvertibleToArray,
        /,
        *,
        collection_efficiency: ConvertibleToArray = 1.0,
        transit_times: ConvertibleToArray = 0.0,
        transit_time_spreads: ConvertibleToArray = 0.0,
        single_photoelectron_charges: ConvertibleToArray = 1.0,
        single_photoelectron_spreads: ConvertibleToArray = 0.0,
        dark_count_rates: ConvertibleToArray = 0.0,
    ) -> None:
        nodes = np.asarray(wavelength_nodes, dtype=np.float64)
        positions = np.asarray(reference_positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or positions.shape[0] < 1:
            raise ValueError("reference_positions must have shape (detectors, 3).")
        if not np.all(np.isfinite(positions)):
            raise ValueError("reference_positions must be finite.")
        count = positions.shape[0]
        if nodes.ndim != 1 or nodes.size < 2:
            raise ValueError("wavelength_nodes must be rank one with at least two nodes.")
        if not np.all(np.isfinite(nodes)) or np.any(nodes <= 0.0):
            raise ValueError("wavelength_nodes must be finite and positive.")
        if np.any(np.diff(nodes) <= 0.0):
            raise ValueError("wavelength_nodes must be strictly increasing.")
        efficiency = np.asarray(quantum_efficiency, dtype=np.float64)
        if efficiency.shape == nodes.shape:
            efficiency = np.broadcast_to(efficiency, (count, nodes.size)).copy()
        if efficiency.shape != (count, nodes.size):
            raise ValueError(
                "quantum_efficiency must have shape (nodes,) or (detectors, nodes)."
            )
        if not np.all(np.isfinite(efficiency)) or np.any(
            (efficiency < 0.0) | (efficiency > 1.0)
        ):
            raise ValueError("quantum_efficiency must lie in [0, 1].")
        collection = _per_detector(collection_efficiency, "collection_efficiency", count)
        transit = _per_detector(transit_times, "transit_times", count)
        spread = _per_detector(transit_time_spreads, "transit_time_spreads", count)
        charge = _per_detector(
            single_photoelectron_charges, "single_photoelectron_charges", count
        )
        charge_spread = _per_detector(
            single_photoelectron_spreads, "single_photoelectron_spreads", count
        )
        dark = _per_detector(dark_count_rates, "dark_count_rates", count)
        if np.any((collection < 0.0) | (collection > 1.0)):
            raise ValueError("collection_efficiency must lie in [0, 1].")
        if np.any(spread < 0.0) or np.any(charge_spread < 0.0) or np.any(dark < 0.0):
            raise ValueError("Spreads and dark-count rates must be non-negative.")
        if np.any(charge <= 0.0):
            raise ValueError("single_photoelectron_charges must be positive.")
        self.wavelength_nodes = jnp.asarray(nodes)
        self.quantum_efficiency = jnp.asarray(efficiency)
        self.collection_efficiency = jnp.asarray(collection)
        self.transit_times = jnp.asarray(transit)
        self.transit_time_spreads = jnp.asarray(spread)
        self.single_photoelectron_charges = jnp.asarray(charge)
        self.single_photoelectron_spreads = jnp.asarray(charge_spread)
        self.dark_count_rates = jnp.asarray(dark)
        self.reference_positions = jnp.asarray(positions)
        self.detector_count = count
        self.photodetector_id = canonical_fingerprint(
            {
                "kind": "optical-photodetector",
                "content": array_tree_fingerprint(
                    {
                        "wavelength_nodes": nodes,
                        "quantum_efficiency": efficiency,
                        "collection_efficiency": collection,
                        "transit_times": transit,
                        "transit_time_spreads": spread,
                        "single_photoelectron_charges": charge,
                        "single_photoelectron_spreads": charge_spread,
                        "dark_count_rates": dark,
                        "reference_positions": positions,
                    }
                ),
            }
        )


class PhotodetectionPlan(StrictModule, NonTrainableState):
    """Photodetector, element-to-channel map, readout gate, and hit capacities.

    ``hit_capacity`` bounds the hits kept per event; ``dark_count_capacity``
    bounds the dark counts sampled per event and must be positive when any
    dark-count rate is.
    """

    photodetector: OpticalPhotodetector
    hit_plan: SensitiveHitPlan
    gate_start: float = eqx.field(static=True)
    gate_end: float = eqx.field(static=True)
    hit_capacity: int = eqx.field(static=True)
    dark_count_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        photodetector: OpticalPhotodetector,
        hit_plan: SensitiveHitPlan,
        /,
        *,
        gate: tuple[float, float],
        hit_capacity: int,
        dark_count_capacity: int = 0,
    ) -> None:
        if not isinstance(photodetector, OpticalPhotodetector):
            raise TypeError("photodetector must be an OpticalPhotodetector.")
        if not isinstance(hit_plan, SensitiveHitPlan):
            raise TypeError("hit_plan must be a SensitiveHitPlan.")
        if hit_plan.element_count != photodetector.detector_count:
            raise ValueError("hit_plan must map exactly one element per photodetector.")
        start, end = (float(value) for value in gate)
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            raise ValueError("gate must be a finite increasing (start, end) pair.")
        if type(hit_capacity) is not int or type(dark_count_capacity) is not int:
            raise TypeError("Hit and dark-count capacities must be ints.")
        if hit_capacity < 1 or dark_count_capacity < 0:
            raise ValueError(
                "hit_capacity must be positive and dark_count_capacity non-negative."
            )
        if dark_count_capacity == 0 and bool(
            np.any(np.asarray(photodetector.dark_count_rates) > 0.0)
        ):
            raise ValueError("Positive dark-count rates need a dark_count_capacity.")
        self.photodetector = photodetector
        self.hit_plan = hit_plan
        self.gate_start = start
        self.gate_end = end
        self.hit_capacity = hit_capacity
        self.dark_count_capacity = dark_count_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "photodetection-plan",
                "photodetector": photodetector.photodetector_id,
                "hit_plan": hit_plan.plan_id,
                "gate": [start, end],
                "hit_capacity": hit_capacity,
                "dark_count_capacity": dark_count_capacity,
            }
        )


class PhotodetectionResult(StrictModule, NonTrainableState):
    """Sensitive hits per ``(event, hit)`` with provenance and evidence.

    ``hits.energies`` hold anode charges and ``hits.source_step_indices`` the
    photon arrival slot (dark counts: ordinal on their detector). ``dark``
    marks dark-count hits and ``photon_id_hi``/``photon_id_lo`` identify the
    photon of every other hit. ``expected_photoelectrons`` sums the detection
    probabilities of in-gate supported arrivals per ``(event, detector)``;
    ``photoelectron_counts`` and ``dark_counts`` are the sampled counts before
    the hit capacity, and ``expected_dark_counts`` the per-event Poisson
    means.
    """

    hits: SensitiveHitBank
    dark: Array
    photon_id_hi: Array
    photon_id_lo: Array
    expected_photoelectrons: Array
    photoelectron_counts: Array
    expected_dark_counts: Array
    dark_counts: Array
    dropped_hits: Array
    status: Array
    successful: Array
    all_successful: Array
    plan_id: str = eqx.field(static=True)


def _single_photoelectron_charges(keys: Array, means: Array, spreads: Array) -> Array:
    """Gamma (Polya) charges with the declared mean and standard deviation."""

    spread = spreads > 0.0
    shape = jnp.where(spread, (means / jnp.where(spread, spreads, 1.0)) ** 2, 1.0)
    gamma = jax.vmap(lambda key, value: jr.gamma(key, value, dtype=jnp.float64))(
        keys, shape
    )
    return jnp.where(spread, gamma * means / shape, means)


class _Candidates(StrictModule):
    """Flat hit candidates before per-event ordering and capacity."""

    events: Array
    times: Array
    dark: Array
    first_word: Array
    second_word: Array
    slots: Array
    detectors: Array
    channels: Array
    positions: Array
    charges: Array
    valid: Array


def _photon_candidates(
    plan: PhotodetectionPlan,
    arrivals: OpticalDetectorArrivals,
    owners: Array,
    event_count: int,
    key: PRNGKey,
) -> tuple[_Candidates, Array, Array, Array, Array, Array]:
    """Photoelectrons from arrivals, expected/sampled counts, and event flags."""

    detector = plan.photodetector
    photons, slots = arrivals.detector_indices.shape
    count = detector.detector_count
    indices = arrivals.detector_indices.reshape(-1)
    in_range = (indices >= 0) & (indices < count)
    elements = jnp.clip(indices, 0, count - 1)
    stencil = linear_stencil(
        detector.wavelength_nodes, arrivals.wavelengths.reshape(-1), bounds="fill"
    )
    rows = detector.quantum_efficiency[elements[:, None], stencil.indices]
    efficiency = jnp.sum(jnp.where(stencil.valid, stencil.weights * rows, 0.0), axis=-1)
    probability = (
        arrivals.weights.reshape(-1)
        * efficiency
        * detector.collection_efficiency[elements]
    )
    times = arrivals.times.reshape(-1)
    in_gate = (times >= plan.gate_start) & (times < plan.gate_end)
    eligible = arrivals.active.reshape(-1) & in_range & in_gate
    supported = eligible & stencil.support
    unsupported = eligible & ~stencil.support
    exceeding = supported & (probability > 1.0)

    def arrival_keys(id_hi: Array, id_lo: Array) -> Array:
        return jax.vmap(
            lambda slot: derive_key(key, _DETECTION_ADDRESS, id_hi, id_lo, slot)
        )(jnp.arange(slots, dtype=jnp.uint32))

    keys = jax.vmap(arrival_keys)(arrivals.id_hi, arrivals.id_lo).reshape(-1)
    split = jax.vmap(lambda value: jr.split(value, 3))(keys)
    uniforms = jax.vmap(lambda value: jr.uniform(value, dtype=jnp.float64))(split[:, 0])
    normals = jax.vmap(lambda value: jr.normal(value, dtype=jnp.float64))(split[:, 1])
    detected = supported & (uniforms < probability)
    anode_times = (
        times
        + detector.transit_times[elements]
        + detector.transit_time_spreads[elements] * normals
    )
    charges = _single_photoelectron_charges(
        split[:, 2],
        detector.single_photoelectron_charges[elements],
        detector.single_photoelectron_spreads[elements],
    )
    channels = plan.hit_plan.element_to_channel[elements]
    photon_events = jnp.repeat(owners, slots)
    by_event = jnp.zeros((event_count, count), dtype=jnp.float64)
    expected = by_event.at[photon_events, elements].add(
        jnp.where(supported, probability, 0.0)
    )
    sampled = (
        jnp.zeros((event_count, count), dtype=jnp.int32)
        .at[photon_events, elements]
        .add(detected.astype(jnp.int32))
    )

    def any_per_event(slots_: Array, values: Array) -> Array:
        tally = jnp.zeros((event_count,), dtype=jnp.int32)
        return tally.at[slots_].add(values.astype(jnp.int32)) > 0

    incomplete = any_per_event(owners, arrivals.dropped > 0)
    unsupported_events = any_per_event(photon_events, unsupported)
    exceeding_events = any_per_event(photon_events, exceeding)
    candidates = _Candidates(
        photon_events,
        anode_times,
        jnp.zeros_like(detected),
        jnp.repeat(arrivals.id_hi, slots),
        jnp.repeat(arrivals.id_lo, slots),
        jnp.tile(jnp.arange(slots, dtype=jnp.uint32), photons),
        elements,
        channels,
        arrivals.positions.reshape((-1, 3)),
        charges,
        detected & (channels >= 0),
    )
    return candidates, expected, sampled, incomplete, unsupported_events, exceeding_events


def _dark_candidates(
    plan: PhotodetectionPlan,
    event_words: Array,
    key: PRNGKey,
) -> tuple[_Candidates, Array, Array, Array]:
    """Dark counts per event: sampled counts, candidates, and overflow."""

    detector = plan.photodetector
    count = detector.detector_count
    capacity = plan.dark_count_capacity
    window = plan.gate_end - plan.gate_start
    means = detector.dark_count_rates * window
    detectors = jnp.arange(count, dtype=jnp.uint32)

    def event_counts(word: Array) -> Array:
        def one(element: Array, mean: Array) -> Array:
            return jr.poisson(
                derive_key(key, _DARK_COUNT_ADDRESS, word, element), mean
            ).astype(jnp.int32)

        return jax.vmap(one)(detectors, means)

    counts = jax.vmap(event_counts)(event_words)
    totals = jnp.sum(counts, axis=-1)
    overflow = totals > capacity
    cumulative = jnp.cumsum(counts, axis=-1)
    ordinals = jnp.arange(capacity, dtype=jnp.int32)
    elements = jax.vmap(lambda edges: jnp.searchsorted(edges, ordinals, side="right"))(
        cumulative
    ).astype(jnp.int32)
    elements = jnp.minimum(elements, count - 1)
    starts = jnp.take_along_axis(cumulative - counts, elements, axis=-1)
    detector_ordinals = (ordinals[None, :] - starts).astype(jnp.uint32)
    occupied = ordinals[None, :] < totals[:, None]

    def slot_keys(word: Array, element: Array, ordinal: Array) -> Array:
        return derive_key(key, _DARK_HIT_ADDRESS, word, element, ordinal)

    keys = jax.vmap(
        lambda word, element_row, ordinal_row: jax.vmap(
            lambda element, ordinal: slot_keys(word, element, ordinal)
        )(element_row, ordinal_row)
    )(event_words, elements.astype(jnp.uint32), detector_ordinals).reshape(-1)
    split = jax.vmap(lambda value: jr.split(value, 3))(keys)
    flat_elements = elements.reshape(-1)
    emission = plan.gate_start + window * jax.vmap(
        lambda value: jr.uniform(value, dtype=jnp.float64)
    )(split[:, 0])
    normals = jax.vmap(lambda value: jr.normal(value, dtype=jnp.float64))(split[:, 1])
    anode_times = (
        emission
        + detector.transit_times[flat_elements]
        + detector.transit_time_spreads[flat_elements] * normals
    )
    charges = _single_photoelectron_charges(
        split[:, 2],
        detector.single_photoelectron_charges[flat_elements],
        detector.single_photoelectron_spreads[flat_elements],
    )
    channels = plan.hit_plan.element_to_channel[flat_elements]
    event_count = event_words.shape[0]
    candidates = _Candidates(
        jnp.repeat(jnp.arange(event_count, dtype=jnp.int32), capacity),
        anode_times,
        jnp.ones((event_count * capacity,), dtype=jnp.bool_),
        flat_elements.astype(jnp.uint32),
        detector_ordinals.reshape(-1),
        detector_ordinals.reshape(-1),
        flat_elements,
        channels,
        detector.reference_positions[flat_elements],
        charges,
        occupied.reshape(-1) & (channels >= 0),
    )
    return candidates, counts, overflow, means


def detect_optical_arrivals(
    plan: PhotodetectionPlan,
    arrivals: OpticalDetectorArrivals,
    key: PRNGKey,
    /,
    *,
    event_ids: ConvertibleToArray,
    photon_events: ConvertibleToArray,
) -> PhotodetectionResult:
    """Convert detector arrivals and dark counts into ``(event, hit)`` hits.

    ``event_ids`` are the unique non-negative 32-bit event identities of the
    output event axis and ``photon_events[i]`` the event slot of photon ``i``
    of ``arrivals``. The returned ``hits`` feed
    :func:`phydrax.applications.detector.digitize_sensitive_hits` under the
    hit plan's conditions.
    """

    if not isinstance(plan, PhotodetectionPlan):
        raise TypeError("plan must be a PhotodetectionPlan.")
    if not isinstance(arrivals, OpticalDetectorArrivals):
        raise TypeError("arrivals must be OpticalDetectorArrivals.")
    key_ = parse(key, PRNGKey, "key")
    events = np.asarray(event_ids)
    if events.ndim != 1 or events.size < 1 or not np.issubdtype(events.dtype, np.integer):
        raise ValueError("event_ids must be a non-empty rank-one integer array.")
    if np.any(events < 0) or np.any(events >= _WORD):
        raise ValueError("event_ids must lie in [0, 2**32).")
    if np.unique(events).size != events.size:
        raise ValueError("event_ids must be unique.")
    photon_count = arrivals.detector_indices.shape[0]
    owners = np.asarray(photon_events)
    if owners.shape != (photon_count,) or not np.issubdtype(owners.dtype, np.integer):
        raise ValueError("photon_events must hold one integer event slot per photon.")
    if np.any(owners < 0) or np.any(owners >= events.size):
        raise ValueError("photon_events must index the event_ids axis.")
    event_count = events.size
    event_words = jnp.asarray(events.astype(np.uint32))
    (
        photon,
        expected,
        sampled,
        incomplete,
        unsupported,
        exceeding,
    ) = _photon_candidates(
        plan, arrivals, jnp.asarray(owners, dtype=jnp.int32), event_count, key_
    )
    count = plan.photodetector.detector_count
    if plan.dark_count_capacity > 0:
        dark, dark_counts, dark_overflow, dark_means = _dark_candidates(
            plan, event_words, key_
        )
        candidates = jax.tree_util.tree_map(
            lambda first, second: jnp.concatenate((first, second), axis=0),
            photon,
            dark,
        )
    else:
        candidates = photon
        dark_counts = jnp.zeros((event_count, count), dtype=jnp.int32)
        dark_overflow = jnp.zeros((event_count,), dtype=jnp.bool_)
        dark_means = jnp.zeros((count,), dtype=jnp.float64)
    hits, dark_hits, id_hi, id_lo, dropped, finite = _bank(
        plan, candidates, jnp.asarray(events), event_count
    )
    status = jnp.select(
        (incomplete, unsupported, exceeding, dark_overflow, dropped > 0, ~finite),
        (
            int(PhotodetectionStatus.INCOMPLETE_ARRIVALS),
            int(PhotodetectionStatus.UNSUPPORTED_WAVELENGTH),
            int(PhotodetectionStatus.DETECTION_PROBABILITY_EXCEEDS_UNITY),
            int(PhotodetectionStatus.DARK_COUNT_CAPACITY_EXHAUSTED),
            int(PhotodetectionStatus.HIT_CAPACITY_EXHAUSTED),
            int(PhotodetectionStatus.NONFINITE_RESULT),
        ),
        int(PhotodetectionStatus.SUCCESS),
    ).astype(jnp.int32)
    successful = status == int(PhotodetectionStatus.SUCCESS)
    return PhotodetectionResult(
        hits,
        dark_hits,
        id_hi,
        id_lo,
        expected,
        sampled,
        dark_means,
        dark_counts,
        dropped,
        status,
        successful,
        jnp.all(successful),
        plan.plan_id,
    )


def _bank(
    plan: PhotodetectionPlan,
    candidates: _Candidates,
    event_ids: Array,
    event_count: int,
) -> tuple[SensitiveHitBank, Array, Array, Array, Array, Array]:
    """Order candidates per event and keep the first ``hit_capacity``."""

    capacity = plan.hit_capacity
    sort_events = jnp.where(candidates.valid, candidates.events, event_count)
    order = jnp.lexsort(
        (
            candidates.slots,
            candidates.second_word,
            candidates.first_word,
            candidates.dark,
            candidates.times,
            sort_events,
        )
    )
    ordered = jax.tree_util.tree_map(lambda values: values[order], candidates)
    ordered_events = sort_events[order]
    rank = jnp.arange(ordered_events.shape[0], dtype=jnp.int32) - jnp.searchsorted(
        ordered_events, ordered_events, side="left"
    ).astype(jnp.int32)
    kept = ordered.valid & (rank < capacity)
    rows = jnp.where(kept, ordered_events, event_count)

    def place(values: Array, fill: Array) -> Array:
        empty = jnp.broadcast_to(fill, (event_count, capacity) + values.shape[1:])
        return empty.at[rows, rank].set(values, mode="drop")

    zero_word = jnp.zeros((), dtype=jnp.uint32)
    active = place(kept, jnp.zeros((), dtype=jnp.bool_))
    valid_per_event = (
        jnp.zeros((event_count,), dtype=jnp.int32)
        .at[jnp.where(ordered.valid, ordered_events, event_count)]
        .add(1, mode="drop")
    )
    dropped = valid_per_event - jnp.sum(active, axis=-1, dtype=jnp.int32)
    dark = place(ordered.dark, jnp.zeros((), dtype=jnp.bool_)) & active
    times = place(ordered.times, jnp.zeros((), dtype=jnp.float64))
    charges = place(ordered.charges, jnp.zeros((), dtype=jnp.float64))
    positions = place(ordered.positions, jnp.zeros((), dtype=jnp.float64))
    finite = jnp.all(
        jnp.where(active, jnp.isfinite(times) & jnp.isfinite(charges), True), axis=-1
    ) & jnp.all(
        jnp.where(active[..., None], jnp.isfinite(positions), True), axis=(-2, -1)
    )
    hits = SensitiveHitBank(
        event_ids=event_ids,
        hit_ids=jnp.broadcast_to(
            jnp.arange(capacity, dtype=jnp.int32), (event_count, capacity)
        ),
        detector_element_ids=place(ordered.detectors, jnp.zeros((), dtype=jnp.int32)),
        channel_ids=place(
            jnp.maximum(ordered.channels, 0), jnp.zeros((), dtype=jnp.int32)
        ),
        source_step_indices=place(
            ordered.slots.astype(jnp.int32), jnp.zeros((), dtype=jnp.int32)
        ),
        positions=positions,
        times=times,
        energies=charges,
        active=active,
        conditions_id=plan.hit_plan.conditions_id,
    )
    id_hi = jnp.where(active & ~dark, place(ordered.first_word, zero_word), zero_word)
    id_lo = jnp.where(active & ~dark, place(ordered.second_word, zero_word), zero_word)
    return hits, dark, id_hi, id_lo, dropped, finite


__all__ = [
    "OpticalPhotodetector",
    "PhotodetectionPlan",
    "PhotodetectionResult",
    "PhotodetectionStatus",
    "detect_optical_arrivals",
]
