#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Metal qualification of prepared Morton queries, ordered sparse relations and Fourier resampling.

Every case runs in a process started with ``JAX_PLATFORMS=mps`` and
``JAX_ENABLE_X64=1``. It places explicitly float32/complex64 geometry and values,
plus full-width int64/uint64 identities, on the one MPS device, calls the public
Phydrax API, and records the device that holds the actual results. References
are analytical formulas or NumPy host arithmetic applied to the same inputs
after they have been rounded once. No reference runs through JAX, so there is
no implicit second backend. Floating-point bounds are derived below from the
float32 unit roundoff before any measurement; exact orderings, identities and
status values are compared bit for bit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.spatial import (
    morton_decode_integer,
    morton_encode_integer,
    MortonAddressPlan,
    MortonNeighborQueryPlan,
    MortonNeighborQueryStatus,
    MortonRadiusRelationPlan,
    MortonSourcePlan,
)
from phydrax.signal import fourier_resample
from tests._support.metal_qualification import metal_device, record_metal_execution


pytestmark = pytest.mark.metal_native

RecordProperty = Callable[[str, object], None]

_EPS32 = float(np.finfo(np.float32).eps)
_SCALAR32 = jax.ShapeDtypeStruct((), jnp.float32)


def _put(value: Any, device: jax.Device) -> Array:
    """Place host data, already in its declared dtype, on the selected device."""
    return jax.device_put(np.asarray(value), device)


def _results(*trees: Any) -> tuple[Array, ...]:
    return tuple(
        leaf
        for tree in trees
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _interleave(coordinates: Sequence[int], depth: int) -> int:
    """Host Morton code from Python integers; never passes through floating point."""
    dimension = len(coordinates)
    code = 0
    for bit in range(depth):
        for axis, value in enumerate(coordinates):
            code |= ((int(value) >> bit) & 1) << (dimension * bit + axis)
    return code


def _squared_distances(
    targets: np.ndarray, sources: np.ndarray, periodic: Sequence[bool]
) -> np.ndarray:
    """Float64 minimum-image distances of the already rounded float32 inputs."""
    relative = targets.astype(np.float64)[:, None, :] - sources.astype(np.float64)[None]
    wrap = np.asarray(periodic, dtype=np.bool_)
    relative = np.where(wrap, relative - np.round(relative), relative)
    return np.sum(relative * relative, axis=-1)


def _host_nearest(
    targets: np.ndarray,
    sources: np.ndarray,
    ids: np.ndarray,
    *,
    neighbors: int,
    periodic: Sequence[bool],
    active: np.ndarray | None = None,
    radius: float | None = None,
) -> list[np.ndarray]:
    """Brute-force (distance, stable ID) order of the eligible sources per target."""
    distances = _squared_distances(targets, sources, periodic)
    eligible = np.ones(distances.shape, dtype=np.bool_)
    if active is not None:
        eligible &= active[None, :]
    if radius is not None:
        eligible &= distances <= radius * radius
    rows = []
    for row in range(targets.shape[0]):
        candidates = np.flatnonzero(eligible[row])
        order = np.lexsort((ids[candidates], distances[row, candidates]))
        rows.append(candidates[order][:neighbors])
    return rows


# --- Morton addressing ------------------------------------------------------


@pytest.mark.parametrize(
    "dimension,depth,coordinates",
    [
        (1, 63, [[2**63 - 1], [2**62 + 2**53 + 1], [2**53 + 1], [1]]),
        (2, 31, [[2**31 - 1, 2**30 + 5], [2**30, 1]]),
        (3, 21, [[2**21 - 1, 0, 2**20 + 1], [2**20, 2**21 - 1, 3]]),
    ],
    ids=["1d-depth63", "2d-depth31", "3d-depth21"],
)
def test_metal_morton_integer_codes_keep_every_high_valid_bit(
    record_property: RecordProperty,
    dimension: int,
    depth: int,
    coordinates: list[list[int]],
) -> None:
    device = metal_device()
    host = np.asarray(coordinates, dtype=np.int64)
    expected = np.asarray(
        [_interleave(row, depth) for row in coordinates], dtype=np.uint64
    )
    integer = _put(host, device)

    codes = morton_encode_integer(integer, depth)
    compiled = jax.jit(morton_encode_integer, static_argnums=1)(integer, depth)
    decoded = morton_decode_integer(codes, dimension, depth)
    plan = MortonAddressPlan((0.0,) * dimension, (1.0,) * dimension, depth)
    level = depth // 2
    shift = dimension * (depth - level)
    prefix = plan.prefix(codes, level)
    start, end = plan.descendant_interval(
        prefix, _put(np.full(host.shape[0], level, dtype=np.int32), device)
    )

    assert codes.dtype == jnp.uint64 and decoded.dtype == jnp.int64
    np.testing.assert_array_equal(np.asarray(codes), expected)
    np.testing.assert_array_equal(np.asarray(compiled), expected)
    np.testing.assert_array_equal(np.asarray(decoded), host)
    expected_prefix = np.asarray(
        [int(code) >> shift for code in expected], dtype=np.uint64
    )
    np.testing.assert_array_equal(np.asarray(prefix), expected_prefix)
    np.testing.assert_array_equal(
        np.asarray(start),
        np.asarray([int(value) << shift for value in expected_prefix], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        np.asarray(end),
        np.asarray(
            [(int(value) + 1) << shift for value in expected_prefix], dtype=np.uint64
        ),
    )
    record_metal_execution(
        record_property, *_results(codes, compiled, decoded, prefix, start, end)
    )


def test_metal_morton_float32_encoding_is_exact_at_full_depth_and_periodic_seams(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    resolution = 2**21
    cells = np.asarray([[2**21 - 1, 0, 2**20 + 1], [2**20, 2**21 - 1, 3]], np.int64)
    # Cell centres (i + 1/2) / 2^21 need at most 22 significand bits: exact in float32.
    centres = ((cells + 0.5) / resolution).astype(np.float32)
    assert np.array_equal(
        np.floor(centres.astype(np.float64) * resolution).astype(np.int64), cells
    )
    invalid = np.asarray([[1.0, 0.5, 0.5], [np.nan, 0.5, 0.5]], dtype=np.float32)
    spatial = _put(np.concatenate((centres, invalid)), device)
    plan = MortonAddressPlan((0.0,) * 3, (1.0,) * 3, 21)

    encoded = plan.encode(spatial)
    compiled = eqx.filter_jit(plan.encode)(spatial)

    expected = [_interleave(row, 21) for row in cells.tolist()] + [0, 0]
    np.testing.assert_array_equal(
        np.asarray(encoded.codes), np.asarray(expected, np.uint64)
    )
    np.testing.assert_array_equal(
        np.asarray(encoded.integer_coordinates),
        np.concatenate((cells, np.zeros((2, 3), np.int64))),
    )
    np.testing.assert_array_equal(encoded.in_domain, [True, True, False, False])
    np.testing.assert_array_equal(encoded.finite, [True, True, True, False])
    assert not bool(encoded.successful)
    assert encoded.coordinates.dtype == jnp.float32
    for actual, reference in zip(
        jax.tree.leaves(compiled), jax.tree.leaves(encoded), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))

    seam = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10, periodic_axes=(True, False))
    wrapped = seam.encode(
        _put(np.asarray([[1.25, 0.5], [-0.25, 0.75], [0.5, 1.0]], np.float32), device)
    )
    np.testing.assert_array_equal(wrapped.in_domain, [True, True, False])
    np.testing.assert_array_equal(
        np.asarray(wrapped.coordinates)[:2], np.asarray([[0.25, 0.5], [0.75, 0.75]])
    )
    np.testing.assert_array_equal(
        np.asarray(wrapped.codes),
        np.asarray(
            [_interleave((256, 512), 10), _interleave((768, 768), 10), 0], np.uint64
        ),
    )
    record_metal_execution(record_property, *_results(encoded, compiled, wrapped))


# --- Prepared Morton neighbour queries -------------------------------------

# Three coincident sources (0, 2, 4) and four equidistant sources (1, 3, 5, 6)
# around the target (0.5, 0.5). The identities differ only below float64
# resolution or across the signed/unsigned boundary, so routing an identity
# through floating point, signed reinterpretation or 32-bit truncation changes
# the expected order.
_TIE_POINTS = np.asarray(
    [
        [0.5, 0.5],
        [0.25, 0.5],
        [0.5, 0.5],
        [0.75, 0.5],
        [0.5, 0.5],
        [0.5, 0.25],
        [0.5, 0.75],
    ],
    dtype=np.float32,
)
_TIE_IDS: dict[str, tuple[np.dtype, list[int], list[int]]] = {
    "int64": (
        np.dtype(np.int64),
        [
            2**60 + 3,
            -(2**63),
            2**60 + 1,
            2**63 - 1,
            2**60 + 2,
            -(2**60) - 1,
            -(2**60) - 2,
        ],
        [2, 4, 0, 1, 6, 5, 3],
    ),
    "uint64": (
        np.dtype(np.uint64),
        [2**64 - 1, 2**63 + 1, 2**64 - 3, 2**63 - 1, 2**64 - 2, 2**53 + 1, 0],
        [2, 4, 0, 6, 5, 3, 1],
    ),
}


@pytest.mark.parametrize("id_kind", ["int64", "uint64"])
def test_metal_prepared_float32_nearest_orders_ties_by_full_width_stable_ids(
    record_property: RecordProperty, id_kind: str
) -> None:
    device = metal_device()
    dtype, id_values, expected_order = _TIE_IDS[id_kind]
    host_ids = np.asarray(id_values, dtype=dtype)
    target_host = _TIE_POINTS[[2]]
    oracle = _host_nearest(
        target_host, _TIE_POINTS, host_ids, neighbors=7, periodic=(False, False)
    )[0]
    # Guard the oracle itself: the hand-derived full-width order.
    np.testing.assert_array_equal(oracle, expected_order)

    source_plan = MortonSourcePlan(MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10), 7)
    points = _put(_TIE_POINTS, device)
    ids = _put(host_ids, device)
    target = _put(target_host, device)
    sources = source_plan.prepare(points, stable_ids=ids)
    complete = MortonNeighborQueryPlan(source_plan, 1, 7)
    eager = complete.query(sources, target)
    compiled = eqx.filter_jit(complete.query)(sources, target)
    prefix = MortonNeighborQueryPlan(source_plan, 1, 3).query(sources, target)
    excluded = MortonNeighborQueryPlan(source_plan, 1, 6).query(
        sources,
        target,
        target_stable_ids=_put(host_ids[[2]], device),
        exclude_self=True,
    )

    integer = np.floor(_TIE_POINTS.astype(np.float64) * 1024).astype(np.int64)
    codes = np.asarray([_interleave(row, 10) for row in integer.tolist()], np.uint64)
    storage = np.lexsort((host_ids, codes))
    assert sources.stable_ids.dtype == dtype and sources.points.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(sources.stable_ids), host_ids)
    np.testing.assert_array_equal(np.asarray(sources.storage_to_logical), storage)
    np.testing.assert_array_equal(np.asarray(sources.sorted_codes), codes[storage])
    np.testing.assert_array_equal(
        np.asarray(sources.sorted_stable_ids), host_ids[storage]
    )
    for result, expected in (
        (eager, oracle),
        (compiled, oracle),
        (prefix, oracle[:3]),
        (excluded, oracle[oracle != 2]),
    ):
        assert bool(result.evidence.successful)
        np.testing.assert_array_equal(result.status, MortonNeighborQueryStatus.COMPLETE)
        np.testing.assert_array_equal(result.counts, [expected.size])
        np.testing.assert_array_equal(np.asarray(result.source_indices)[0], expected)
        np.testing.assert_array_equal(result.valid, True)
    record_metal_execution(
        record_property, *_results(sources, eager, compiled, prefix, excluded)
    )


def test_metal_prepared_float32_nearest_honors_masks_radius_and_periodic_seam(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    source_host = np.asarray(
        [[0.98, 0.5], [0.1, 0.5], [0.5, 0.5], [0.75, 0.5]], dtype=np.float32
    )
    target_host = np.asarray([[0.02, 0.5], [0.5, 0.5]], dtype=np.float32)
    active = np.asarray([True, True, False, True])
    periodic = (True, False)
    source_plan = MortonSourcePlan(
        MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10, periodic_axes=periodic), 4
    )
    result = MortonNeighborQueryPlan(source_plan, 2, 3).query(
        source_plan.prepare(_put(source_host, device), active_mask=_put(active, device)),
        _put(target_host, device),
        target_mask=_put(np.asarray([True, False]), device),
        radius=0.2,
    )

    expected = _host_nearest(
        target_host[:1],
        source_host,
        np.arange(4, dtype=np.int64),
        neighbors=3,
        periodic=periodic,
        active=active,
        radius=0.2,
    )[0]
    # The seam neighbour 0.98 is nearest to 0.02 only through the periodic image.
    np.testing.assert_array_equal(expected, [0, 1])
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(np.asarray(result.source_indices)[0, :2], expected)
    np.testing.assert_array_equal(result.valid, [[True, True, False], [False] * 3])
    np.testing.assert_array_equal(result.counts, [2, 0])
    np.testing.assert_array_equal(
        result.status,
        [MortonNeighborQueryStatus.COMPLETE, MortonNeighborQueryStatus.INACTIVE_TARGET],
    )
    record_metal_execution(record_property, *_results(result))


def test_metal_prepared_float32_candidate_capacity_overflow_refuses_rows(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    source_host = np.asarray([[0.3, 0.3]] * 8 + [[0.9, 0.9], [0.92, 0.9]], np.float32)
    target_host = np.asarray([[0.3, 0.3], [0.9, 0.91]], np.float32)
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10)
    source_plan = MortonSourcePlan(address, 10)
    sources = source_plan.prepare(_put(source_host, device))
    nearest = MortonNeighborQueryPlan(source_plan, 2, 2, maximum_candidates=4).query(
        sources, _put(target_host, device)
    )
    relation = MortonRadiusRelationPlan(source_plan, 10, 64, maximum_candidates=4).query(
        sources, _put(source_host, device), 0.05, exclude_self=True
    )

    expected = _host_nearest(
        target_host[1:],
        source_host,
        np.arange(10, dtype=np.int64),
        neighbors=2,
        periodic=(False, False),
    )[0]
    assert not bool(nearest.evidence.successful)
    assert int(nearest.evidence.overflow_rows) == 1
    assert int(nearest.evidence.required_candidates) >= 8
    np.testing.assert_array_equal(
        nearest.status,
        [
            MortonNeighborQueryStatus.CANDIDATE_OVERFLOW,
            MortonNeighborQueryStatus.COMPLETE,
        ],
    )
    np.testing.assert_array_equal(nearest.valid[0], False)
    np.testing.assert_array_equal(np.asarray(nearest.source_indices)[1], expected)
    assert not bool(relation.evidence.successful)
    assert not bool(relation.evidence.complete)
    assert int(relation.evidence.overflow_rows) == 8
    np.testing.assert_array_equal(relation.relation.valid, False)
    record_metal_execution(record_property, *_results(sources, nearest, relation))


def test_metal_prepared_float32_invalid_geometry_duplicates_and_empty_sets_fail_closed(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    source_plan = MortonSourcePlan(MortonAddressPlan((0.0,), (1.0,), 12), 4)
    host_points = np.asarray([[0.125], [0.375], [0.625], [np.nan]], np.float32)
    points = _put(host_points, device)
    targets = _put(np.asarray([[0.125], [0.375]], np.float32), device)
    query = MortonNeighborQueryPlan(source_plan, 2, 2)
    # Distinct only at full uint64 width: float64 would merge 2^64 - 1 and 2^64 - 2.
    ids = _put(
        np.asarray([2**64 - 1, 2**64 - 2, 2**64 - 2, 2**64 - 1], np.uint64), device
    )
    mask = np.asarray([True, True, False, False])

    masked = source_plan.prepare(points, active_mask=_put(mask, device), stable_ids=ids)
    admitted = query.query(masked, targets)
    duplicate = query.query(
        source_plan.prepare(
            points,
            active_mask=_put(np.asarray([True, True, True, False]), device),
            stable_ids=ids,
        ),
        targets,
    )
    nonfinite = query.query(source_plan.prepare(points), targets)
    outside = query.query(
        source_plan.prepare(_put(host_points[[0, 1, 2, 0]], device)),
        _put(np.asarray([[0.125], [1.5]], np.float32), device),
    )
    empty_sources = source_plan.prepare(
        _put(np.full((4, 1), np.nan, np.float32), device),
        active_mask=_put(np.zeros(4, np.bool_), device),
    )
    empty = MortonNeighborQueryPlan(source_plan, 2, 3).query(empty_sources, targets)
    empty_radius = MortonRadiusRelationPlan(source_plan, 2, 0).query(
        empty_sources, targets, 0.125
    )

    assert bool(admitted.evidence.successful)
    assert int(masked.active_count) == 2 and not bool(masked.finite[3])
    np.testing.assert_array_equal(np.asarray(admitted.source_indices), [[0, 1], [1, 0]])
    assert not bool(duplicate.evidence.sources_valid)
    assert not bool(jnp.any(duplicate.valid))
    np.testing.assert_array_equal(
        duplicate.status, MortonNeighborQueryStatus.INVALID_SOURCES
    )
    assert not bool(nonfinite.evidence.finite)
    assert int(nonfinite.evidence.invalid_sources) == 1
    np.testing.assert_array_equal(
        nonfinite.status, MortonNeighborQueryStatus.INVALID_SOURCES
    )
    np.testing.assert_array_equal(nonfinite.valid, False)
    assert not bool(outside.evidence.successful)
    assert int(outside.evidence.invalid_targets) == 1
    np.testing.assert_array_equal(
        outside.status,
        [MortonNeighborQueryStatus.COMPLETE, MortonNeighborQueryStatus.INVALID_TARGET],
    )
    assert bool(empty.evidence.successful)
    np.testing.assert_array_equal(empty.counts, 0)
    assert not bool(jnp.any(empty.valid))
    assert bool(empty_radius.evidence.successful)
    assert int(empty_radius.evidence.required_pairs) == 0
    record_metal_execution(
        record_property,
        *_results(masked, admitted, duplicate, nonfinite, outside, empty, empty_radius),
    )


def test_metal_morton_radius_relation_periodic_pairs_boundary_and_pair_overflow(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    host_points = np.asarray([[0.98], [0.03], [0.4], [0.65]], np.float32)
    host_ids = np.asarray([2**60 + 40, 2**60 + 10, 2**60 + 30, 2**60 + 20], np.int64)
    periodic_plan = MortonRadiusRelationPlan(
        MortonSourcePlan(MortonAddressPlan((0.0,), (1.0,), 16, periodic_axes=(True,)), 4),
        4,
        6,
    )
    points = _put(host_points, device)
    ids = _put(host_ids, device)
    periodic = periodic_plan.query(
        periodic_plan.source_plan.prepare(points, stable_ids=ids),
        points,
        0.26,
        target_stable_ids=ids,
        exclude_self=True,
        pair_once=True,
    )
    distances = _squared_distances(host_points, host_points, (True,))
    lower, upper = np.nonzero(
        (distances <= 0.26**2) & (host_ids[:, None] < host_ids[None, :])
    )
    order = np.lexsort((host_ids[upper], host_ids[lower]))
    valid = np.asarray(periodic.relation.valid)
    np.testing.assert_array_equal(host_ids[lower[order]], [2**60 + 10, 2**60 + 20])
    np.testing.assert_array_equal(host_ids[upper[order]], [2**60 + 40, 2**60 + 30])
    assert bool(periodic.evidence.successful)
    np.testing.assert_array_equal(
        np.asarray(periodic.relation.source_indices)[valid], lower[order]
    )
    np.testing.assert_array_equal(
        np.asarray(periodic.relation.target_indices)[valid], upper[order]
    )

    address = MortonAddressPlan((0.0,), (1.0,), 8)
    lattice = _put(np.asarray([[0.0], [0.25], [0.5], [0.75]], np.float32), device)
    lattice_sources = MortonSourcePlan(address, 4).prepare(lattice)

    def boundary(maximum_pairs: int, inclusive: bool) -> Any:
        return MortonRadiusRelationPlan(
            MortonSourcePlan(address, 4), 4, maximum_pairs, inclusive=inclusive
        ).query(lattice_sources, lattice, 0.25, exclude_self=True, pair_once=True)

    # Lattice spacing and radius are exact in float32: three pairs lie on the sphere.
    closed, opened, overflow = boundary(3, True), boundary(3, False), boundary(2, True)
    assert bool(closed.evidence.successful)
    assert int(closed.evidence.required_pairs) == 3
    assert bool(opened.evidence.successful)
    assert int(opened.evidence.required_pairs) == 0
    assert not bool(overflow.evidence.successful)
    assert bool(overflow.evidence.pair_overflow)
    assert int(overflow.evidence.required_pairs) == 3
    np.testing.assert_array_equal(overflow.relation.valid, False)
    record_metal_execution(
        record_property, *_results(periodic, lattice_sources, closed, opened, overflow)
    )


def test_metal_prepared_float32_refresh_reuses_rebuilds_and_rolls_back(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    source_plan = MortonSourcePlan(MortonAddressPlan((0.0,), (1.0,), 12), 4)
    host_points = np.asarray([[0.125], [0.375], [0.625], [0.875]], np.float32)
    host_ids = np.asarray([2**60 + 40, 2**60 + 10, 2**60 + 30, 2**60 + 20], np.int64)
    host_mask = np.asarray([True, True, True, False])
    points = _put(host_points, device)
    ids = _put(host_ids, device)
    mask = _put(host_mask, device)
    previous = source_plan.prepare(points, active_mask=mask, stable_ids=ids)
    refresh = eqx.filter_jit(source_plan.refresh)

    moved = _put(host_points + np.float32(0.01), device)
    reused = refresh(previous, moved)
    fresh = source_plan.prepare(moved, active_mask=mask, stable_ids=ids)
    exchanged = _put(np.asarray([[0.65], [0.375], [0.1], [0.875]], np.float32), device)
    rebuilt = refresh(previous, exchanged)
    activated = refresh(previous, points, active_mask=_put(np.ones(4, np.bool_), device))
    renamed = refresh(previous, points, stable_ids=_put(host_ids.copy() + 50, device))
    duplicated_ids = host_ids.copy()
    duplicated_ids[0] = host_ids[1]
    rejections = [
        refresh(
            previous,
            _put(
                host_points + np.asarray([[0.975], [0], [0], [0]], dtype=np.float32),
                device,
            ),
        ),
        refresh(previous, points, stable_ids=_put(duplicated_ids, device)),
    ]
    gradient = jax.grad(
        lambda values: jnp.sum(
            source_plan.refresh(previous, values).accepted.sorted_coordinates ** 2
        )
    )(points)

    assert bool(reused.successful) and bool(reused.reused_order)
    assert reused.accepted.points.dtype == jnp.float32
    assert reused.accepted.sorted_coordinates.dtype == jnp.float32
    assert reused.accepted.stable_ids.dtype == jnp.int64
    np.testing.assert_array_equal(np.asarray(reused.accepted.stable_ids), host_ids)
    np.testing.assert_array_equal(
        np.asarray(reused.accepted.storage_to_logical),
        np.asarray(previous.storage_to_logical),
    )
    for actual, expected in zip(
        jax.tree.leaves(reused.candidate), jax.tree.leaves(fresh), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert bool(rebuilt.successful) and not bool(rebuilt.reused_order)
    # Active points ordered by Morton cell: logical 2 (0.1), 1 (0.375), 0 (0.65).
    np.testing.assert_array_equal(
        np.asarray(rebuilt.accepted.storage_to_logical)[:3], [2, 1, 0]
    )
    assert bool(activated.successful) and not bool(activated.reused_order)
    assert bool(renamed.successful) and not bool(renamed.reused_order)
    for rejected in rejections:
        assert not bool(rejected.successful) and not bool(rejected.reused_order)
        np.testing.assert_array_equal(np.asarray(rejected.accepted.points), host_points)
        np.testing.assert_array_equal(np.asarray(rejected.accepted.stable_ids), host_ids)
    # Sort-free refresh keeps live float32 geometry: d/dx sum(x^2) = 2x exactly.
    assert gradient.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(gradient), 2 * host_points)
    with pytest.raises(ValueError, match="stable-ID dtype"):
        source_plan.refresh(
            previous, points, stable_ids=_put(np.arange(4, dtype=np.int32), device)
        )
    record_metal_execution(
        record_property,
        *_results(
            previous, reused, fresh, rebuilt, activated, renamed, rejections, gradient
        ),
    )


def test_metal_prepared_float32_wrapped_geometry_is_differentiable(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    source_plan = MortonSourcePlan(
        MortonAddressPlan((0.0,), (1.0,), 12, periodic_axes=(True,)), 4
    )
    query = MortonNeighborQueryPlan(source_plan, 1, 1)
    # Source 0 sits one period above 0.125; the target is 0.2.
    points = _put(np.asarray([[1.125], [0.375], [0.625], [0.875]], np.float32), device)
    target = _put(np.asarray([[0.2]], np.float32), device)

    def squared_distance(values: Array) -> Array:
        sources = source_plan.prepare(values)
        result = query.query(sources, target)
        relative = target[0] - sources.points[result.source_indices[0, 0]]
        relative = relative - jnp.round(relative)
        return jnp.sum(relative * relative)

    gradient = jax.grad(squared_distance)(points)
    wrapped = jax.grad(
        lambda values: jnp.sum(source_plan.prepare(values).sorted_coordinates)
    )(points)

    # d/dx (0.2 - (x - 1))^2 at x = 1.125: -2 * 0.075. Bound from the float32
    # rounding of 0.2 and the wrap: a few ulps of the operands.
    assert gradient.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(gradient)[:, 0], [-0.15, 0.0, 0.0, 0.0], rtol=2e-6, atol=2e-7
    )
    np.testing.assert_array_equal(np.asarray(wrapped), np.ones((4, 1), np.float32))
    record_metal_execution(record_property, *_results(gradient, wrapped))


# --- Ordered sparse relations ----------------------------------------------


def _sequential_float32(values: Iterable[np.float32]) -> np.float32:
    """Left-to-right float32 recursive sum, the deterministic policy's contract."""
    total = np.float32(0.0)
    for value in values:
        total = np.float32(total + np.float32(value))
    return total


def _single_receiver_total(
    device: jax.Device,
    events: Array,
    *,
    edge_tile: int,
    accumulation: phx.sparse.RelationAccumulation,
) -> phx.sparse.StreamedRelationResult:
    relation = phx.sparse.EdgeRelation(
        np.zeros(4, dtype=np.int32),
        np.zeros(4, dtype=np.int32),
        source_size=1,
        target_size=1,
    )
    prepared = phx.sparse.StreamedRelationPlan(
        receiver_tile=1, edge_tile=edge_tile, accumulation=accumulation
    ).prepare(relation, owner_id="metal-cancellation")
    zeros = _put(np.zeros(1, np.float32), device)
    return prepared.evaluate(
        phx.sparse.StreamedPayloadSpec(_SCALAR32, _SCALAR32),
        lambda p, s, r, e: e,
        lambda p, r, a: a,
        None,
        zeros,
        zeros,
        events,
    )


@pytest.mark.parametrize("edge_tile", [1, 4], ids=["fragment-per-event", "one-fragment"])
def test_metal_sparse_float32_cancellation_keeps_seeded_compensation(
    record_property: RecordProperty, edge_tile: int
) -> None:
    device = metal_device()
    # 2^25 + 1 rounds to 2^25 in float32 (ulp 4): a left-to-right sum yields 1,
    # an exact or correctly compensated sum yields 2.
    host = np.asarray([2.0**25, 1.0, -(2.0**25), 1.0], np.float32)
    assert _sequential_float32(host) == np.float32(1.0)
    assert math.fsum(float(value) for value in host) == 2.0
    events = _put(host, device)

    compensated = _single_receiver_total(
        device, events, edge_tile=edge_tile, accumulation="compensated"
    )
    deterministic = _single_receiver_total(
        device, events, edge_tile=edge_tile, accumulation="deterministic"
    )
    replay = jax.jit(
        lambda values: (
            _single_receiver_total(
                device, values, edge_tile=edge_tile, accumulation="compensated"
            ).receiver_outputs
        )
    )
    replays = [replay(events) for _ in range(3)]

    for result in (compensated, deterministic):
        assert bool(result.evidence.successful)
        assert result.receiver_outputs.dtype == jnp.float32
    assert float(compensated.receiver_outputs[0]) == 2.0
    assert float(deterministic.receiver_outputs[0]) == float(_sequential_float32(host))
    for value in replays:
        np.testing.assert_array_equal(np.asarray(value), [2.0])
    record_metal_execution(
        record_property, *_results(compensated, deterministic, replays)
    )


def _ordered_relation_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Receivers, mixed-magnitude float32 events, int64 IDs and a storage permutation.

    ``id = 2^60 + (pi // 2) * 2^32 + sigma`` puts every ID beyond 2^53 and makes
    pairs share their high word: float64 conversion merges them, 32-bit
    truncation orders by ``sigma``, and only full-width comparison orders by
    ``pi``. Seed 0 is order-sensitive in float32 (checked in the test).
    """
    rng = np.random.default_rng(0)
    count = 24
    receivers = rng.integers(0, 3, count).astype(np.int32)
    events = (rng.normal(size=count) * 10.0 ** rng.integers(-4, 5, size=count)).astype(
        np.float32
    )
    pi, sigma = rng.permutation(count), rng.permutation(count)
    ids = np.asarray(
        [2**60 + ((int(pi[k]) // 2) << 32) + int(sigma[k]) for k in range(count)],
        dtype=np.int64,
    )
    return receivers, events, ids, rng.permutation(count)


def test_metal_sparse_float32_ordered_policies_follow_int64_stable_ids(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    receivers, events, ids, permutation = _ordered_relation_data()

    def host_order(keys: np.ndarray, receiver: int) -> np.ndarray:
        order = np.argsort(keys, kind="stable")
        return order[receivers[order] == receiver]

    expected = np.asarray(
        [_sequential_float32(events[host_order(ids, row)]) for row in range(3)]
    )
    by_storage = [_sequential_float32(events[receivers == row]) for row in range(3)]
    by_low_word = [
        _sequential_float32(events[host_order(ids & 0xFFFFFFFF, row)]) for row in range(3)
    ]
    # Deliberate-fault guard: a wrong event order changes these float32 sums.
    assert not np.array_equal(expected, by_storage)
    assert not np.array_equal(expected, by_low_word)

    def evaluate(
        storage: np.ndarray, accumulation: phx.sparse.RelationAccumulation
    ) -> phx.sparse.StreamedRelationResult:
        relation = phx.sparse.EdgeRelation(
            np.zeros(storage.size, np.int32),
            receivers[storage],
            source_size=1,
            target_size=3,
        )
        prepared = phx.sparse.StreamedRelationPlan(
            receiver_tile=2, edge_tile=3, accumulation=accumulation
        ).prepare(
            relation,
            owner_id="metal-order",
            stable_route_ids=_put(ids[storage], device),
        )
        return prepared.evaluate(
            phx.sparse.StreamedPayloadSpec(_SCALAR32, _SCALAR32),
            lambda p, s, r, e: e,
            lambda p, r, a: a,
            None,
            _put(np.zeros(1, np.float32), device),
            _put(np.zeros(3, np.float32), device),
            _put(events[storage], device),
        )

    identity = np.arange(events.size)
    results = {
        (policy, name): evaluate(storage, policy)
        for policy in ("deterministic", "compensated")
        for name, storage in (("identity", identity), ("permuted", permutation))
    }
    for result in results.values():
        assert bool(result.evidence.successful)
        assert result.receiver_outputs.dtype == jnp.float32
    for name in ("identity", "permuted"):
        np.testing.assert_array_equal(
            np.asarray(results["deterministic", name].receiver_outputs), expected
        )
    np.testing.assert_array_equal(
        np.asarray(results["compensated", "identity"].receiver_outputs),
        np.asarray(results["compensated", "permuted"].receiver_outputs),
    )
    # Compensated summation bound (Higham, Accuracy and Stability, 4.3):
    # |s - S| <= 2u|S| + O(n^2 u^2) sum|x|, with u = eps/2, n events per receiver.
    unit = _EPS32 / 2
    compensated = np.asarray(results["compensated", "identity"].receiver_outputs)
    for row in range(3):
        terms = [float(value) for value in events[receivers == row]]
        exact = math.fsum(terms)
        bound = 2 * unit * abs(exact) + 4 * len(terms) ** 2 * unit**2 * math.fsum(
            abs(value) for value in terms
        )
        assert abs(float(compensated[row]) - exact) <= bound, (
            row,
            compensated[row],
            exact,
        )
    record_metal_execution(record_property, *_results(list(results.values())))


def test_metal_sparse_packed_key_groups_order_int64_ids_beyond_float64(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    maximum = 2**32 - 1
    keys = np.asarray(
        [
            [maximum, 0, 3, 1],
            [0, maximum, 3, 1],
            [0, maximum, 3, 2],
            [maximum, 0, 3, 1],
            [maximum, maximum, maximum, maximum],
        ],
        dtype=np.uint32,
    )
    # Rows 0 and 3 share a key; IDs 2^60 + 5 and 2^60 + 1 are equal in float64.
    stable_ids = np.asarray([2**60 + k for k in (5, 4, 3, 1, 2)], dtype=np.int64)
    groups = jax.jit(phx.sparse.KeyGroupPlan(5, 5, (maximum,) * 4).build)(
        _put(keys, device),
        _put(np.ones(5, np.bool_), device),
        stable_ids=_put(stable_ids, device),
    )
    query = np.asarray(
        [
            [maximum, maximum, maximum, maximum],
            [maximum, 0, 3, 2],
            [0, maximum, 3, 1],
            [0, maximum, 4, 1],
        ],
        dtype=np.uint32,
    )
    lookup = eqx.filter_jit(groups.lookup)(_put(query, device))

    assert bool(groups.evidence.successful)
    np.testing.assert_array_equal(np.asarray(groups.storage_to_logical), [1, 2, 3, 0, 4])
    np.testing.assert_array_equal(np.asarray(groups.group_keys)[:4], keys[[1, 2, 0, 4]])
    np.testing.assert_array_equal(np.asarray(groups.group_counts), [1, 1, 2, 1, 0])
    np.testing.assert_array_equal(lookup.supported, [True, False, True, False])
    np.testing.assert_array_equal(np.asarray(lookup.group_slots), [3, 0, 0, 0])
    record_metal_execution(record_property, *_results(groups, lookup))


@pytest.mark.parametrize("accumulation", ["deterministic", "compensated"])
@pytest.mark.parametrize("edge_tile", [1, 2], ids=["seeded-fragments", "one-fragment"])
def test_metal_sparse_float32_valid_zero_events_keep_jvp_vjp_and_curvature(
    record_property: RecordProperty,
    accumulation: phx.sparse.RelationAccumulation,
    edge_tile: int,
) -> None:
    device = metal_device()
    relation = phx.sparse.EdgeRelation(
        np.asarray([0, 1, 0], dtype=np.int32),
        np.asarray([0, 0, 0], dtype=np.int32),
        source_size=2,
        target_size=1,
        valid=np.asarray([True, True, False]),
    )
    prepared = phx.sparse.StreamedRelationPlan(
        receiver_tile=1, edge_tile=edge_tile, accumulation=accumulation
    ).prepare(relation, owner_id="metal-zero-event-curvature")
    payload = phx.sparse.StreamedPayloadSpec(_SCALAR32, _SCALAR32, _SCALAR32)
    residual = _put(np.asarray([0.5], np.float32), device)
    # The invalid route's NaN edge row is padding: it must contribute nothing.
    edges = _put(np.asarray([0.0, 0.0, np.nan], np.float32), device)
    host_labels = np.asarray([0.3, -0.7], np.float32)
    theta = _put(np.float32(1.0), device)
    x = _put(np.zeros(2, np.float32), device)
    labels = _put(host_labels, device)
    direction = _put(np.asarray([1.0, 2.0], np.float32), device)

    def node_edge(t: Array, positions: Array) -> tuple[Array, Array]:
        result = prepared.evaluate(
            payload,
            lambda p, source, r, e: ((p - 1.0) + source + e, source * source),
            lambda p, r, aggregate: (aggregate + r) ** 2,
            t,
            positions,
            residual,
            edges,
        )
        assert result.edge_outputs is not None
        return result.receiver_outputs[0], jnp.sum(result.edge_outputs[:2] ** 2)

    def energy(t: Array, positions: Array) -> Array:
        return node_edge(t, positions)[0]

    def loss(t: Array, positions: Array) -> Array:
        first, second = node_edge(t, positions)
        return first + second

    def force_loss(t: Array) -> Array:
        forces = -jax.grad(loss, argnums=1)(t, x)
        return jnp.sum((forces - labels) ** 2) + loss(t, x) ** 2

    coordinate_gradient = jax.grad(energy, argnums=1)
    observed = {
        "energy": energy(theta, x),
        "vjp": coordinate_gradient(theta, x),
        "jvp": jax.jvp(lambda y: energy(theta, y), (x,), (direction,))[1],
        "forward-over-reverse": jax.jacfwd(coordinate_gradient, argnums=1)(theta, x),
        "reverse-over-reverse": jax.jacrev(coordinate_gradient, argnums=1)(theta, x),
        "combined-hvp": jax.jvp(
            lambda y: jax.grad(loss, argnums=1)(theta, y), (x,), (direction,)
        )[1],
        "mixed": jax.jacfwd(coordinate_gradient, argnums=0)(theta, x),
        "force-loss": jax.grad(force_loss)(theta),
    }
    # Every valid message is exactly zero at theta = 1, x = 0, yet both events
    # own their derivatives. E = (2 (theta - 1) + x0 + x1 + r)^2, the edge term
    # is x0^4 + x1^4, s = 0.5 and ds/dtheta = 2:
    #   dE/dx = 2 s = 1; d2E/dx2 = 2; d2E/dx dtheta = 4; (H + 12 x^2) d = 2 (d0 + d1);
    #   force loss = 8 (2 + l0 + l1) + 2 E dE/dtheta = 8 (2 + l0 + l1) + 1.
    expected = {
        "energy": 0.25,
        "vjp": [1.0, 1.0],
        "jvp": 3.0,
        "forward-over-reverse": np.full((2, 2), 2.0),
        "reverse-over-reverse": np.full((2, 2), 2.0),
        "combined-hvp": [6.0, 6.0],
        "mixed": [4.0, 4.0],
        "force-loss": 8.0 * (2.0 + float(host_labels[0]) + float(host_labels[1])) + 1.0,
    }
    for name, value in observed.items():
        assert value.dtype == jnp.float32, name
        # At most a handful of float32 roundings of O(10) magnitudes per entry.
        np.testing.assert_allclose(
            np.asarray(value), expected[name], rtol=8 * _EPS32, atol=0, err_msg=name
        )
    np.testing.assert_array_equal(np.asarray(observed["forward-over-reverse"]), 2.0)
    np.testing.assert_array_equal(np.asarray(observed["vjp"]), [1.0, 1.0])
    record_metal_execution(record_property, *_results(observed))


_FAILURE_SOURCES = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 3, 0, 2], dtype=np.int32)
_FAILURE_TARGETS = np.asarray([2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 3, 3, 9], dtype=np.int32)
_FAILURE_VALID = np.asarray([True] * 12 + [False])


def test_metal_sparse_float32_failed_traced_schedule_invalidates_derivatives(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    rng = np.random.default_rng(23)
    host_positions = rng.normal(size=(5, 3)).astype(np.float32)
    host_edges = rng.uniform(0.5, 1.5, size=13).astype(np.float32)
    host_theta = np.float32(0.7)
    plan = phx.sparse.StreamedRelationPlan(
        receiver_tile=2, edge_tile=3, accumulation="deterministic"
    )
    payload = phx.sparse.StreamedPayloadSpec(_SCALAR32, _SCALAR32)
    edges = _put(host_edges, device)
    theta = _put(host_theta, device)

    def energy(positions: Array, ids: Array) -> tuple[Array, Array]:
        prepared = plan.prepare(
            phx.sparse.EdgeRelation(
                _FAILURE_SOURCES,
                _FAILURE_TARGETS,
                source_size=5,
                target_size=5,
                valid=_FAILURE_VALID,
            ),
            owner_id="metal-failure",
            stable_route_ids=ids,
        )
        result = prepared.evaluate(
            payload,
            lambda t, s, r, e: t * e * jnp.sum((s - r) ** 2),
            lambda t, r, aggregate: aggregate,
            theta,
            positions,
            positions,
            edges,
        )
        return jnp.sum(result.receiver_outputs), result.evidence.successful

    gradient = jax.jit(jax.grad(energy, has_aux=True))
    positions = _put(host_positions, device)
    duplicate_ids = _put(np.full(13, 2**60, dtype=np.int64), device)
    distinct_ids = _put(2**60 + np.arange(13, dtype=np.int64), device)
    failed, failed_successful = gradient(positions, duplicate_ids)
    replays = [gradient(positions, distinct_ids) for _ in range(3)]

    # Host gradient of sum_k theta e_k |x_s - x_r|^2 over the valid routes.
    expected = np.zeros((5, 3))
    magnitude = np.zeros((5, 3))
    for route in np.flatnonzero(_FAILURE_VALID):
        source, target = _FAILURE_SOURCES[route], _FAILURE_TARGETS[route]
        term = (
            2.0
            * float(host_theta)
            * float(host_edges[route])
            * (host_positions[source].astype(np.float64) - host_positions[target])
        )
        expected[source] += term
        expected[target] -= term
        magnitude[source] += np.abs(term)
        magnitude[target] += np.abs(term)
    # Each term takes ~5 float32 roundings and each node sums at most 12 of them.
    bound = 32 * _EPS32 * magnitude
    assert not bool(failed_successful)
    assert bool(jnp.all(jnp.isnan(failed)))
    first, first_successful = replays[0]
    assert bool(first_successful)
    assert first.dtype == jnp.float32
    assert np.all(np.abs(np.asarray(first, np.float64) - expected) <= bound)
    for value, successful in replays[1:]:
        assert bool(successful)
        np.testing.assert_array_equal(np.asarray(value), np.asarray(first))
    record_metal_execution(record_property, *_results(failed, failed_successful, replays))


# --- Fourier resampling ----------------------------------------------------


def _fourier_basis(
    size: int, queries: np.ndarray, *, derivative: bool = False
) -> np.ndarray:
    """Band-limited interpolation basis with the symmetric even-Nyquist split.

    Column ``j`` multiplies the forward-normalized DFT coefficient ``j``. An even
    Nyquist coefficient is shared equally by modes ``+-N/2``: ``cos(pi N q)``.
    """
    q = np.asarray(queries, dtype=np.float64)[:, None]
    frequencies = np.fft.fftfreq(size) * size
    basis = np.exp(2j * np.pi * frequencies[None, :] * q)
    if derivative:
        basis = basis * (2j * np.pi * frequencies[None, :])
    if size % 2 == 0:
        nyquist = size // 2
        column = (
            -np.pi * size * np.sin(np.pi * size * q[:, 0])
            if derivative
            else np.cos(np.pi * size * q[:, 0])
        )
        basis[:, nyquist] = column
    return basis


def _spectral_bound(
    terms: int, basis: np.ndarray, coefficients: np.ndarray, peak: float
) -> float:
    """A priori float32 bound for forward + inverse transforms of ``terms`` total size.

    Each coefficient and each output is a sum of at most ``terms`` rounded
    products (gamma_n ~ n eps per sum). Coefficient errors are at most
    ``terms * eps * peak`` and are propagated by the interpolation basis.
    """
    scale = float(np.max(np.abs(basis) @ (np.abs(coefficients) + peak)))
    return 2.0 * (terms + 2) * _EPS32 * scale


@pytest.mark.strict_jax
@pytest.mark.parametrize("source_size,target_size", [(5, 8), (8, 5)])
def test_metal_fourier_complex64_preserves_representable_modes(
    record_property: RecordProperty, source_size: int, target_size: int
) -> None:
    device = metal_device()
    host = np.exp(2j * np.pi * np.arange(source_size) / source_size).astype(np.complex64)
    output = fourier_resample(_put(host, device), (target_size,))

    targets = np.arange(target_size) / target_size
    coefficients = np.fft.fft(host.astype(np.complex128)) / source_size
    basis = _fourier_basis(source_size, targets)
    bound = _spectral_bound(source_size + target_size, basis, coefficients, 1.0)
    assert output.dtype == jnp.complex64
    np.testing.assert_allclose(
        np.asarray(output), np.exp(2j * np.pi * targets), rtol=0, atol=bound
    )
    record_metal_execution(record_property, output)


@pytest.mark.strict_jax
def test_metal_fourier_float32_even_nyquist_splits_and_merges(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    host = ((-1.0) ** np.arange(8)).astype(np.float32)
    source = _put(host, device)
    fine = fourier_resample(source, (13,))
    restored = fourier_resample(fine, (8,))

    targets = np.arange(13) / 13
    coefficients = np.fft.fft(host.astype(np.float64)) / 8
    basis = _fourier_basis(8, targets)
    # Nyquist alone: cos(2 pi 4 m / 13) on the fine grid, then exactly (-1)^n.
    np.testing.assert_allclose(
        basis @ coefficients, np.cos(8 * np.pi * targets), atol=1e-12
    )
    bound = _spectral_bound(8 + 13 + 13 + 8, basis, coefficients, 1.0)
    assert fine.dtype == jnp.float32 and restored.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(fine), np.cos(8 * np.pi * targets), rtol=0, atol=bound
    )
    np.testing.assert_allclose(np.asarray(restored), host, rtol=0, atol=bound)
    record_metal_execution(record_property, fine, restored)


@pytest.mark.strict_jax
def test_metal_fourier_complex64_multiaxis_phase_shift_matches_direct_evaluation(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    rng = np.random.default_rng(12)
    source_shape, target_shape = (8, 6), (9, 11)
    host = (
        rng.normal(size=source_shape + (2,)) + 1j * rng.normal(size=source_shape + (2,))
    ).astype(np.complex64)
    host_offsets = (np.float32(0.137), np.float32(-0.219))
    values = _put(host, device)
    offsets = tuple(_put(offset, device) for offset in host_offsets)

    def shifted(array: Array, first: Array, second: Array) -> Array:
        return fourier_resample(
            array, target_shape, axes=(0, 1), phase_offsets=(first, second)
        )

    eager = shifted(values, *offsets)
    compiled = jax.jit(shifted)(values, *offsets)

    coefficients = np.fft.fftn(host.astype(np.complex128), axes=(0, 1)) / math.prod(
        source_shape
    )
    bases = [
        _fourier_basis(size, float(offset) + np.arange(target) / target)
        for size, target, offset in zip(
            source_shape, target_shape, host_offsets, strict=True
        )
    ]
    reference = np.einsum("ai,bj,ijp->abp", bases[0], bases[1], coefficients)
    peak = float(np.max(np.abs(host)))
    magnitude = np.einsum(
        "ai,bj,ijp->abp", np.abs(bases[0]), np.abs(bases[1]), np.abs(coefficients) + peak
    )
    bound = (
        2.0
        * (math.prod(source_shape) + math.prod(target_shape) + 2)
        * _EPS32
        * float(np.max(magnitude))
    )
    for output in (eager, compiled):
        assert output.shape == target_shape + (2,)
        assert output.dtype == jnp.complex64
        np.testing.assert_allclose(np.asarray(output), reference, rtol=0, atol=bound)
    record_metal_execution(record_property, eager, compiled)


@pytest.mark.strict_jax
def test_metal_fourier_float32_jit_phase_derivatives_and_adjoint(
    record_property: RecordProperty,
) -> None:
    device = metal_device()
    rng = np.random.default_rng(31)
    source_size, target_size = 6, 9
    host = rng.normal(size=source_size).astype(np.float32)
    tangent_host = rng.normal(size=source_size).astype(np.float32)
    cotangent_host = rng.normal(size=target_size).astype(np.float32)
    host_offset = np.float32(0.173)
    values = _put(host, device)
    offset = _put(host_offset, device)

    @jax.jit
    def resample(array: Array, shift: Array) -> Array:
        return fourier_resample(array, (target_size,), phase_offsets=(shift,))

    output = resample(values, offset)
    total_gradient = jax.jit(jax.grad(lambda v, s: jnp.sum(resample(v, s))))(
        values, offset
    )
    _, offset_tangent = jax.jvp(
        lambda s: resample(values, s), (offset,), (_put(np.float32(1.0), device),)
    )
    _, value_tangent = jax.jvp(
        lambda v: resample(v, offset), (values,), (_put(tangent_host, device),)
    )
    _, pullback = jax.vjp(lambda v: resample(v, offset), values)
    (cotangent,) = pullback(_put(cotangent_host, device))

    queries = float(host_offset) + np.arange(target_size) / target_size
    basis = _fourier_basis(source_size, queries)
    slope = _fourier_basis(source_size, queries, derivative=True)
    coefficients = np.fft.fft(host.astype(np.float64)) / source_size
    tangent_coefficients = np.fft.fft(tangent_host.astype(np.float64)) / source_size
    terms = source_size + 1 + target_size
    peak = float(np.max(np.abs(host)))
    # The real linear operator A: x -> Re(B F x / N); its exact transpose on the host.
    operator = np.real(basis @ (np.fft.fft(np.eye(source_size), axis=0) / source_size))

    for value in (output, total_gradient, offset_tangent, value_tangent, cotangent):
        assert value.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(output),
        np.real(basis @ coefficients),
        rtol=0,
        atol=_spectral_bound(terms, basis, coefficients, peak),
    )
    # sum_m f(q_m) = target_size * c_0 for |k| < target_size: gradient 9 / 6.
    np.testing.assert_allclose(
        np.asarray(total_gradient),
        np.full(source_size, target_size / source_size),
        rtol=0,
        atol=_spectral_bound(
            terms, np.ones((1, source_size)) * target_size, np.zeros(source_size), 1.0
        ),
    )
    np.testing.assert_allclose(
        np.asarray(offset_tangent),
        np.real(slope @ coefficients),
        rtol=0,
        atol=_spectral_bound(terms, slope, coefficients, peak),
    )
    np.testing.assert_allclose(
        np.asarray(value_tangent),
        np.real(basis @ tangent_coefficients),
        rtol=0,
        atol=_spectral_bound(
            terms, basis, tangent_coefficients, float(np.max(np.abs(tangent_host)))
        ),
    )
    np.testing.assert_allclose(
        np.asarray(cotangent),
        operator.T @ cotangent_host.astype(np.float64),
        rtol=0,
        atol=_spectral_bound(
            terms,
            np.abs(operator.T),
            np.abs(cotangent_host.astype(np.float64)),
            float(np.max(np.abs(cotangent_host))),
        ),
    )
    record_metal_execution(
        record_property, output, total_gradient, offset_tangent, value_tangent, cotangent
    )
