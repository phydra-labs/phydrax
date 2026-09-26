#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from itertools import product
from math import ceil
from typing import Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.sparse import EdgeRelation

from ._morton import (
    _canonical_morton_point_order,
    _MortonPointOrder,
    morton_encode_integer,
    MortonAddressPlan,
)


SpatialDistanceBackend: TypeAlias = Literal["jax", "pallas"]

_UINT64_MAX = np.iinfo(np.uint64).max
# Candidate slots evaluated per target chunk when no chunk size is requested.
_CHUNK_SLOT_BUDGET = 1 << 18
_MINIMUM_DEFAULT_CANDIDATES = 64
_DEFAULT_CANDIDATES_PER_RESULT = 32
# Cell faces and point cells are computed in floating point; certification
# subtracts a conservative multiple of the coordinate rounding scale.
_ROUNDING_FACTOR = 16.0


class MortonNeighborQueryStatus(IntEnum):
    """Per-target outcome of a Morton neighbor or radius query."""

    COMPLETE = 0
    INACTIVE_TARGET = 1
    INVALID_TARGET = 2
    INVALID_SOURCES = 3
    CANDIDATE_OVERFLOW = 4
    UNCERTIFIED = 5


class MortonNeighborQueryEvidence(NonTrainableState, StrictModule):
    """Completeness and resource evidence for one exact neighbor query."""

    successful: jax.Array
    complete: jax.Array
    finite: jax.Array
    sources_valid: jax.Array
    invalid_sources: jax.Array
    invalid_targets: jax.Array
    required_candidates: jax.Array
    candidate_capacity: jax.Array
    overflow_rows: jax.Array
    uncertified_rows: jax.Array


class MortonNeighborQueryResult(NonTrainableState, StrictModule):
    """Fixed-width exact source indices in target-logical order."""

    source_indices: jax.Array
    valid: jax.Array
    counts: jax.Array
    status: jax.Array
    evidence: MortonNeighborQueryEvidence


class MortonRadiusRelationEvidence(NonTrainableState, StrictModule):
    """Completeness and resource evidence for one exact radius relation."""

    successful: jax.Array
    complete: jax.Array
    finite: jax.Array
    sources_valid: jax.Array
    invalid_sources: jax.Array
    invalid_targets: jax.Array
    cell_level: jax.Array
    maximum_cell_occupancy: jax.Array
    required_candidates: jax.Array
    candidate_capacity: jax.Array
    overflow_rows: jax.Array
    uncertified_rows: jax.Array
    required_pairs: jax.Array
    pair_capacity: jax.Array
    pair_overflow: jax.Array


class MortonRadiusRelationResult(NonTrainableState, StrictModule):
    """Exact fixed-capacity source-to-target radius relation."""

    relation: EdgeRelation
    status: jax.Array
    evidence: MortonRadiusRelationEvidence
    storage_to_logical: jax.Array
    logical_to_storage: jax.Array
    logical_cell_slots: jax.Array
    cell_counts: jax.Array
    cell_offsets: jax.Array


class _SortedSources(NamedTuple):
    order: _MortonPointOrder
    search_codes: jax.Array
    coordinates: jax.Array


class _TargetRows(NamedTuple):
    coordinates: jax.Array
    integer_coordinates: jax.Array
    stable_ids: jax.Array
    valid: jax.Array


class _NearestRows(NamedTuple):
    storage: jax.Array
    valid: jax.Array
    bound: jax.Array
    certified: jax.Array
    overflow: jax.Array
    required: jax.Array


class _RadiusRows(NamedTuple):
    sources: jax.Array
    counts: jax.Array
    certified: jax.Array
    overflow: jax.Array
    required: jax.Array


def _minimum_image(
    relative: jax.Array,
    address_plan: MortonAddressPlan,
) -> jax.Array:
    lengths = jnp.asarray(address_plan.upper, dtype=relative.dtype) - jnp.asarray(
        address_plan.lower, dtype=relative.dtype
    )
    periodic = jnp.asarray(address_plan.periodic_axes, dtype=jnp.bool_)
    wrapped = relative - jnp.round(relative / lengths) * lengths
    return jnp.where(periodic, wrapped, relative)


def _squared_norm(
    relative: jax.Array,
    *,
    backend: SpatialDistanceBackend,
    pallas_interpret: bool,
) -> jax.Array:
    match backend:
        case "jax":
            return jnp.sum(relative * relative, axis=-1)
        case "pallas":
            from phydrax.backends.spatial import spatial_squared_norm

            return spatial_squared_norm(
                relative,
                backend="pallas",
                pallas_interpret=pallas_interpret,
            )
        case _:
            raise ValueError("distance_backend must be 'jax' or 'pallas'.")


def _validated_capacities(
    address_plan: MortonAddressPlan,
    source_capacity: int,
    target_capacity: int,
    target_chunk_size: int | None,
    distance_backend: SpatialDistanceBackend,
) -> tuple[int, int]:
    if not isinstance(address_plan, MortonAddressPlan):
        raise TypeError("address_plan must be a MortonAddressPlan.")
    sources = int(source_capacity)
    targets = int(target_capacity)
    if sources < 1 or targets < 1:
        raise ValueError("source_capacity and target_capacity must be positive.")
    if target_chunk_size is not None and int(target_chunk_size) < 1:
        raise ValueError("target_chunk_size must be positive when supplied.")
    if distance_backend not in ("jax", "pallas"):
        raise ValueError("distance_backend must be 'jax' or 'pallas'.")
    return sources, targets


def _resolved_chunk_size(
    target_chunk_size: int | None, targets: int, candidates: int
) -> int:
    if target_chunk_size is None:
        return max(1, min(targets, _CHUNK_SLOT_BUDGET // candidates))
    return min(int(target_chunk_size), targets)


def _floating_points(name: str, points: jax.Array, shape: tuple[int, int]) -> jax.Array:
    values = jnp.asarray(points)
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError(f"{name} must have floating dtype.")
    # Neighbor selection is a discrete decision; coordinate derivatives belong
    # to consumers that evaluate geometry on the returned indices.
    return jax.lax.stop_gradient(values)


def _target_mask(mask: jax.Array | None, capacity: int) -> jax.Array:
    if mask is None:
        return jnp.ones((capacity,), dtype=jnp.bool_)
    active = jnp.asarray(mask, dtype=jnp.bool_)
    if active.shape != (capacity,):
        raise ValueError("target_mask must match target_capacity.")
    return active


def _target_ids(stable_ids: jax.Array | None, capacity: int) -> jax.Array:
    if stable_ids is None:
        return jnp.arange(capacity, dtype=jnp.int64)
    identifiers = jnp.asarray(stable_ids)
    if identifiers.shape != (capacity,):
        raise ValueError("target_stable_ids must match target_capacity.")
    if not jnp.issubdtype(identifiers.dtype, jnp.integer):
        raise TypeError("target_stable_ids must have integer dtype.")
    return identifiers


def _rounding_margin(address_plan: MortonAddressPlan, dtype: jnp.dtype) -> float:
    scale = max(
        abs(lower) + abs(upper)
        for lower, upper in zip(address_plan.lower, address_plan.upper, strict=True)
    )
    epsilon = float(np.finfo(np.dtype(dtype)).eps)
    return _ROUNDING_FACTOR * epsilon * address_plan.dimension * scale


def _minimum_extent(address_plan: MortonAddressPlan) -> float:
    return min(
        upper - lower
        for lower, upper in zip(address_plan.lower, address_plan.upper, strict=True)
    )


def _static_level_wider_than(
    address_plan: MortonAddressPlan, bound: float, margin: float
) -> int:
    """Finest level whose narrowest cell exceeds ``bound`` after rounding."""
    extent = _minimum_extent(address_plan)
    level = 0
    while (
        level < address_plan.maximum_depth
        and extent * 0.5 ** (level + 1) - margin > bound
    ):
        level += 1
    return level


def _dynamic_level_wider_than(
    address_plan: MortonAddressPlan, bound: jax.Array, margin: float
) -> jax.Array:
    exponents = np.arange(1, address_plan.maximum_depth + 1, dtype=np.float64)
    widths = jnp.asarray(
        _minimum_extent(address_plan) * np.exp2(-exponents), dtype=bound.dtype
    )
    return jnp.sum(widths[None, :] - margin > bound[:, None], axis=1, dtype=jnp.int32)


def _stencil_offsets(dimension: int) -> np.ndarray:
    return np.asarray(tuple(product((-1, 0, 1), repeat=dimension)), dtype=np.int64)


def _prepare_sources(
    address_plan: MortonAddressPlan,
    points: jax.Array,
    *,
    capacity: int,
    active_mask: jax.Array | None,
    stable_ids: jax.Array | None,
) -> _SortedSources:
    # One Morton sort serves every prefix level: each coarse cell at any level
    # is one contiguous span of the sorted codes, found by binary search.
    order = _canonical_morton_point_order(
        address_plan,
        points,
        point_capacity=capacity,
        active_mask=active_mask,
        stable_ids=stable_ids,
    )
    search_codes = jnp.where(
        order.sorted_active,
        order.sorted_codes,
        jnp.asarray(_UINT64_MAX, dtype=jnp.uint64),
    )
    return _SortedSources(
        order=order,
        search_codes=search_codes,
        coordinates=order.encoding.coordinates[order.storage_to_logical],
    )


def _stencil_spans(
    address_plan: MortonAddressPlan,
    sources: _SortedSources,
    rows: _TargetRows,
    level: jax.Array,
    prune: jax.Array,
    margin: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return visited 3^d-stencil spans and each row's certified search radius.

    A stencil cell is visited when its box lies within ``prune`` of the target.
    Every unvisited source is at least the certified radius from its target:
    points beyond the stencil block are separated by the block faces, and points
    inside pruned cells are separated by those cells' box distances.
    """
    dimension = address_plan.dimension
    depth = address_plan.maximum_depth
    coordinates = rows.coordinates
    dtype = coordinates.dtype
    offsets = jnp.asarray(_stencil_offsets(dimension))
    lower = jnp.asarray(address_plan.lower, dtype=dtype)
    extent = jnp.asarray(address_plan.upper, dtype=dtype) - lower
    periodic = jnp.asarray(address_plan.periodic_axes, dtype=jnp.bool_)
    infinity = jnp.asarray(jnp.inf, dtype=dtype)

    shift = (depth - level).astype(jnp.int64)
    cells = rows.integer_coordinates >> shift[:, None]
    resolution = (jnp.int64(1) << level.astype(jnp.int64))[:, None]
    scale = jnp.exp2(-level.astype(dtype))[:, None]
    width = extent * scale
    cell_lower = lower + extent * (cells.astype(dtype) * scale)
    below = jnp.maximum(coordinates - cell_lower, 0)
    above = jnp.maximum(cell_lower + width - coordinates, 0)
    # A periodic axis with at most three cells is covered completely by the
    # distinct stencil offsets; its neighbor distances are bounded below by 0.
    wraps = periodic & (resolution <= 3)

    neighbors = cells[:, None, :] + offsets
    axis_resolution = resolution[:, None, :]
    in_range = (neighbors >= 0) & (neighbors < axis_resolution)
    distinct = (
        (offsets == 0)
        | ((offsets == -1) & (axis_resolution >= 2))
        | ((offsets == 1) & (axis_resolution >= 3))
    )
    axis_valid = jnp.where(periodic, distinct, in_range)
    neighbors = jnp.where(periodic, jnp.mod(neighbors, axis_resolution), neighbors)
    axis_distance = jnp.where(
        offsets < 0,
        below[:, None, :],
        jnp.where(offsets > 0, above[:, None, :], 0),
    )
    axis_distance = jnp.where(wraps[:, None, :], 0, axis_distance)
    cell_valid = jnp.all(axis_valid, axis=-1)
    box_distance = jnp.sqrt(jnp.sum(axis_distance * axis_distance, axis=-1))
    visited = cell_valid & (box_distance - margin <= prune[:, None])
    pruned_radius = jnp.min(
        jnp.where(cell_valid & ~visited, box_distance, infinity), axis=1
    )

    lower_open = jnp.where(periodic, wraps, cells <= 1)
    upper_open = jnp.where(periodic, wraps, cells >= resolution - 2)
    lower_gap = jnp.where(lower_open, infinity, below + width)
    upper_gap = jnp.where(upper_open, infinity, above + width)
    block_radius = jnp.min(jnp.minimum(lower_gap, upper_gap), axis=1)
    certified_radius = jnp.minimum(block_radius, pruned_radius) - margin

    cell_codes = jnp.where(axis_valid, neighbors, 0).astype(jnp.uint64)
    first = morton_encode_integer(
        cell_codes << shift.astype(jnp.uint64)[:, None, None], depth
    )
    span = jnp.uint64(1) << (dimension * shift).astype(jnp.uint64)
    start = jnp.searchsorted(sources.search_codes, first, side="left")
    stop = jnp.searchsorted(sources.search_codes, first + span[:, None], side="left")
    counts = jnp.where(visited, stop - start, 0).astype(jnp.int32)
    return start.astype(jnp.int32), counts, certified_radius


def _gather_slots(
    starts: jax.Array, counts: jax.Array, capacity: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pack visited spans into a fixed-width buffer of Morton storage slots."""
    cumulative = jnp.cumsum(counts, axis=1)
    required = cumulative[:, -1]
    slots = jnp.arange(capacity, dtype=jnp.int32)
    span = jax.vmap(lambda row: jnp.searchsorted(row, slots, side="right"))(cumulative)
    span = jnp.minimum(span, counts.shape[1] - 1)
    before = jnp.take_along_axis(cumulative - counts, span, axis=1)
    storage = jnp.take_along_axis(starts, span, axis=1) + slots - before
    filled = slots < required[:, None]
    return jnp.where(filled, storage, 0), filled, required


def _finest_populated_level(
    address_plan: MortonAddressPlan,
    sources: _SortedSources,
    rows: _TargetRows,
    requested: jax.Array,
    prune: jax.Array,
    margin: float,
    minimum_level: int,
) -> jax.Array:
    """Binary-search the finest level whose visited stencil holds ``requested``."""
    depth = address_plan.maximum_depth
    batch = rows.valid.shape[0]

    def refine(_, bounds):
        low, high = bounds
        middle = (low + high + 1) // 2
        _, counts, _ = _stencil_spans(address_plan, sources, rows, middle, prune, margin)
        enough = jnp.sum(counts, axis=1) >= requested
        return jnp.where(enough, middle, low), jnp.where(enough, high, middle - 1)

    bounds = (
        jnp.full((batch,), minimum_level, dtype=jnp.int32),
        jnp.full((batch,), depth, dtype=jnp.int32),
    )
    iterations = (depth - minimum_level).bit_length()
    low, _ = jax.lax.fori_loop(0, iterations, refine, bounds)
    return low


def _map_target_chunks(
    function: Callable[[_TargetRows], tuple[jax.Array, ...]],
    rows: _TargetRows,
    chunk_size: int,
) -> tuple[jax.Array, ...]:
    count = rows.valid.shape[0]
    chunks = ceil(count / chunk_size)
    padding = chunks * chunk_size - count

    def split(value: jax.Array) -> jax.Array:
        widths = ((0, padding),) + ((0, 0),) * (value.ndim - 1)
        padded = jnp.pad(value, widths)
        return padded.reshape((chunks, chunk_size) + value.shape[1:])

    outputs = jax.lax.map(function, jax.tree.map(split, rows))
    return jax.tree.map(
        lambda value: value.reshape((chunks * chunk_size,) + value.shape[2:])[:count],
        outputs,
    )


class _QueryInputs(NamedTuple):
    sources: _SortedSources
    rows: _TargetRows
    target_active: jax.Array
    target_in_domain: jax.Array
    target_finite: jax.Array
    margin: float


def _prepare_query(
    address_plan: MortonAddressPlan,
    source_points: jax.Array,
    target_points: jax.Array,
    *,
    source_capacity: int,
    target_capacity: int,
    source_mask: jax.Array | None,
    target_mask: jax.Array | None,
    source_stable_ids: jax.Array | None,
    target_stable_ids: jax.Array | None,
) -> _QueryInputs:
    dimension = address_plan.dimension
    sources = _floating_points(
        "source_points", source_points, (source_capacity, dimension)
    )
    targets = _floating_points(
        "target_points", target_points, (target_capacity, dimension)
    )
    target_active = _target_mask(target_mask, target_capacity)
    target_ids = _target_ids(target_stable_ids, target_capacity)
    prepared = _prepare_sources(
        address_plan,
        sources,
        capacity=source_capacity,
        active_mask=source_mask,
        stable_ids=source_stable_ids,
    )
    encoding = address_plan.encode(targets)
    dtype = jnp.result_type(sources.dtype, targets.dtype)
    rows = _TargetRows(
        coordinates=encoding.coordinates.astype(dtype),
        integer_coordinates=encoding.integer_coordinates,
        stable_ids=target_ids,
        valid=target_active & encoding.in_domain,
    )
    return _QueryInputs(
        sources=prepared._replace(
            coordinates=prepared.coordinates.astype(dtype),
        ),
        rows=rows,
        target_active=target_active,
        target_in_domain=encoding.in_domain,
        target_finite=encoding.finite,
        margin=_rounding_margin(address_plan, dtype),
    )


def _row_outcomes(
    inputs: _QueryInputs,
    overflow: jax.Array,
    certified: jax.Array,
    required: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Resolve per-target status and the evidence shared by both query kinds."""
    order = inputs.sources.order
    sources_valid = (order.invalid_points == 0) & order.stable_ids_unique
    status = jnp.where(
        certified,
        jnp.int32(MortonNeighborQueryStatus.COMPLETE),
        jnp.int32(MortonNeighborQueryStatus.UNCERTIFIED),
    )
    status = jnp.where(
        overflow, jnp.int32(MortonNeighborQueryStatus.CANDIDATE_OVERFLOW), status
    )
    status = jnp.where(
        sources_valid, status, jnp.int32(MortonNeighborQueryStatus.INVALID_SOURCES)
    )
    status = jnp.where(
        inputs.target_in_domain,
        status,
        jnp.int32(MortonNeighborQueryStatus.INVALID_TARGET),
    )
    status = jnp.where(
        inputs.target_active,
        status,
        jnp.int32(MortonNeighborQueryStatus.INACTIVE_TARGET),
    )
    overflow_rows = jnp.sum(
        status == MortonNeighborQueryStatus.CANDIDATE_OVERFLOW, dtype=jnp.int32
    )
    uncertified_rows = jnp.sum(
        status == MortonNeighborQueryStatus.UNCERTIFIED, dtype=jnp.int32
    )
    return status, {
        "complete": (overflow_rows == 0) & (uncertified_rows == 0),
        "finite": jnp.all(inputs.target_finite | ~inputs.target_active)
        & jnp.all(order.encoding.finite | ~order.active),
        "sources_valid": sources_valid,
        "invalid_sources": order.invalid_points,
        "invalid_targets": jnp.sum(
            inputs.target_active & ~inputs.target_in_domain, dtype=jnp.int32
        ),
        "required_candidates": jnp.max(
            jnp.where(inputs.rows.valid, required, 0), initial=0
        ).astype(jnp.int32),
        "overflow_rows": overflow_rows,
        "uncertified_rows": uncertified_rows,
    }


class MortonNeighborQueryPlan(StrictModule):
    """Exact low-dimensional k-nearest-neighbor query over coarse Morton cells.

    Sources are sorted once by Morton code. Each target visits the 3^d stencil
    around its cell at the finest level holding enough sources, packs at most
    ``maximum_candidates`` sources into a fixed-width buffer, and certifies the
    selection when the k-th distance is strictly inside the visited region.
    Uncertified rows retry once at the coarser level implied by that distance;
    rows that still cannot be certified or that overflow the buffer are
    reported through ``status`` and never return neighbors.
    """

    address_plan: MortonAddressPlan
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    maximum_neighbors: int = eqx.field(static=True)
    maximum_candidates: int = eqx.field(static=True)
    target_chunk_size: int = eqx.field(static=True)
    distance_backend: SpatialDistanceBackend = eqx.field(static=True)
    pallas_interpret: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        source_capacity: int,
        target_capacity: int,
        maximum_neighbors: int,
        *,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
        distance_backend: SpatialDistanceBackend = "jax",
        pallas_interpret: bool = False,
    ) -> None:
        sources, targets = _validated_capacities(
            address_plan,
            source_capacity,
            target_capacity,
            target_chunk_size,
            distance_backend,
        )
        neighbors = int(maximum_neighbors)
        if neighbors < 1 or neighbors > sources:
            raise ValueError("maximum_neighbors must lie in [1, source_capacity].")
        candidates = (
            min(
                sources,
                max(
                    _MINIMUM_DEFAULT_CANDIDATES,
                    _DEFAULT_CANDIDATES_PER_RESULT * neighbors,
                ),
            )
            if maximum_candidates is None
            else int(maximum_candidates)
        )
        if candidates < neighbors or candidates > sources:
            raise ValueError(
                "maximum_candidates must lie in [maximum_neighbors, source_capacity]."
            )
        chunk = _resolved_chunk_size(target_chunk_size, targets, candidates)

        self.address_plan = address_plan
        self.source_capacity = sources
        self.target_capacity = targets
        self.maximum_neighbors = neighbors
        self.maximum_candidates = candidates
        self.target_chunk_size = chunk
        self.distance_backend = distance_backend
        self.pallas_interpret = bool(pallas_interpret)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "morton-neighbor-query-plan",
                "address_plan_id": address_plan.plan_id,
                "source_capacity": sources,
                "target_capacity": targets,
                "maximum_neighbors": neighbors,
                "maximum_candidates": candidates,
                "target_chunk_size": chunk,
                "distance_backend": distance_backend,
                "pallas_interpret": bool(pallas_interpret),
            }
        )

    def _nearest_attempt(
        self,
        inputs: _QueryInputs,
        rows: _TargetRows,
        level: jax.Array,
        prune: jax.Array,
        *,
        exclude_self: bool,
        radius: float | None,
    ) -> _NearestRows:
        sources = inputs.sources
        starts, counts, certified_radius = _stencil_spans(
            self.address_plan, sources, rows, level, prune, inputs.margin
        )
        storage, candidate_valid, required = _gather_slots(
            starts, counts, self.maximum_candidates
        )
        relative = _minimum_image(
            rows.coordinates[:, None, :] - sources.coordinates[storage],
            self.address_plan,
        )
        distance_squared = _squared_norm(
            relative,
            backend=self.distance_backend,
            pallas_interpret=self.pallas_interpret,
        )
        source_ids = sources.order.sorted_stable_ids[storage]
        if exclude_self:
            candidate_valid = candidate_valid & (source_ids != rows.stable_ids[:, None])
        if radius is not None:
            candidate_valid = candidate_valid & (distance_squared <= radius**2)

        maximum_id = jnp.asarray(jnp.iinfo(source_ids.dtype).max, dtype=source_ids.dtype)
        infinity = jnp.asarray(jnp.inf, dtype=distance_squared.dtype)
        ordered_distance, _, ordered_storage, ordered_valid = jax.lax.sort(
            (
                jnp.where(candidate_valid, distance_squared, infinity),
                jnp.where(candidate_valid, source_ids, maximum_id),
                storage,
                candidate_valid,
            ),
            dimension=1,
            num_keys=2,
        )
        k = self.maximum_neighbors
        found = jnp.sum(candidate_valid, axis=1)
        bound = jnp.where(found >= k, jnp.sqrt(ordered_distance[:, k - 1]), infinity)
        if radius is not None:
            bound = jnp.minimum(bound, radius)
        return _NearestRows(
            storage=ordered_storage[:, :k],
            valid=ordered_valid[:, :k],
            bound=bound,
            certified=(bound < certified_radius) | jnp.isposinf(certified_radius),
            overflow=required > self.maximum_candidates,
            required=required,
        )

    def _nearest_chunk(
        self,
        inputs: _QueryInputs,
        rows: _TargetRows,
        requested: jax.Array,
        *,
        exclude_self: bool,
        radius: float | None,
    ) -> _NearestRows:
        dtype = rows.coordinates.dtype
        batch = rows.valid.shape[0]
        if radius is None:
            initial_prune = jnp.full((batch,), jnp.inf, dtype=dtype)
            minimum_level = 0
        else:
            initial_prune = jnp.full((batch,), radius, dtype=dtype)
            minimum_level = _static_level_wider_than(
                self.address_plan, radius, inputs.margin
            )
        level = _finest_populated_level(
            self.address_plan,
            inputs.sources,
            rows,
            requested,
            initial_prune,
            inputs.margin,
            minimum_level,
        )
        first = self._nearest_attempt(
            inputs,
            rows,
            level,
            initial_prune,
            exclude_self=exclude_self,
            radius=radius,
        )
        retry = rows.valid & ~first.overflow & ~first.certified

        def coarser(_) -> _NearestRows:
            wider = _dynamic_level_wider_than(
                self.address_plan, first.bound, inputs.margin
            )
            coarse_level = jnp.maximum(jnp.minimum(level - 1, wider), 0)
            return self._nearest_attempt(
                inputs,
                rows,
                coarse_level.astype(jnp.int32),
                first.bound,
                exclude_self=exclude_self,
                radius=radius,
            )

        second = jax.lax.cond(jnp.any(retry), coarser, lambda _: first, None)
        return jax.tree.map(
            lambda retried, initial: jnp.where(
                retry.reshape(retry.shape + (1,) * (initial.ndim - 1)),
                retried,
                initial,
            ),
            second,
            first,
        )

    def query(
        self,
        source_points: jax.Array,
        target_points: jax.Array,
        *,
        source_mask: jax.Array | None = None,
        target_mask: jax.Array | None = None,
        source_stable_ids: jax.Array | None = None,
        target_stable_ids: jax.Array | None = None,
        exclude_self: bool = False,
        radius: float | None = None,
    ) -> MortonNeighborQueryResult:
        radius_value = None if radius is None else float(radius)
        if radius_value is not None and (
            not np.isfinite(radius_value) or radius_value <= 0
        ):
            raise ValueError("radius must be finite and positive when supplied.")
        if (
            exclude_self
            and target_stable_ids is None
            and (self.source_capacity != self.target_capacity)
        ):
            raise ValueError(
                "exclude_self with unequal capacities requires target_stable_ids."
            )
        inputs = _prepare_query(
            self.address_plan,
            source_points,
            target_points,
            source_capacity=self.source_capacity,
            target_capacity=self.target_capacity,
            source_mask=source_mask,
            target_mask=target_mask,
            source_stable_ids=source_stable_ids,
            target_stable_ids=target_stable_ids,
        )
        requested = jnp.minimum(
            self.maximum_neighbors + int(bool(exclude_self)),
            inputs.sources.order.active_count,
        )
        nearest = _map_target_chunks(
            lambda rows: self._nearest_chunk(
                inputs,
                rows,
                requested,
                exclude_self=bool(exclude_self),
                radius=radius_value,
            ),
            inputs.rows,
            self.target_chunk_size,
        )
        status, evidence = _row_outcomes(
            inputs, nearest.overflow, nearest.certified, nearest.required
        )
        valid = nearest.valid & (status == MortonNeighborQueryStatus.COMPLETE)[:, None]
        logical = inputs.sources.order.storage_to_logical[nearest.storage]
        successful = (
            evidence["complete"]
            & evidence["sources_valid"]
            & (evidence["invalid_targets"] == 0)
        )
        return MortonNeighborQueryResult(
            source_indices=jnp.where(valid, logical, 0).astype(jnp.int32),
            valid=valid,
            counts=jnp.sum(valid, axis=1, dtype=jnp.int32),
            status=status,
            evidence=MortonNeighborQueryEvidence(
                successful=successful,
                candidate_capacity=jnp.asarray(self.maximum_candidates, dtype=jnp.int32),
                **evidence,
            ),
        )


def _cell_table(
    address_plan: MortonAddressPlan, order: _MortonPointOrder, level: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Occupied level cells as contiguous spans of the canonical Morton order."""
    capacity = order.sorted_codes.shape[0]
    prefixes = address_plan.prefix(order.sorted_codes, level)
    position = jnp.arange(capacity, dtype=jnp.int32)
    changed = jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), prefixes[1:] != prefixes[:-1])
    )
    head = order.sorted_active & changed
    slot = jnp.cumsum(head, dtype=jnp.int32) - 1
    sorted_slot = jnp.where(order.sorted_active, slot, -1)
    offsets = (
        jnp.zeros((capacity,), dtype=jnp.int32)
        .at[jnp.where(head, slot, capacity)]
        .set(position, mode="drop")
    )
    counts = (
        jnp.zeros((capacity,), dtype=jnp.int32)
        .at[jnp.where(order.sorted_active, slot, capacity)]
        .add(1, mode="drop")
    )
    return sorted_slot[order.logical_to_storage], counts, offsets


class MortonRadiusRelationPlan(StrictModule):
    """Exact low-dimensional radius relation over coarse Morton cells.

    The cell level is the finest whose cells are wider than the radius, so each
    target visits only stencil cells within the radius. Candidate buffers are
    bounded by ``maximum_candidates`` and pairs by ``maximum_pairs``; any
    overflow or failed certificate invalidates the relation and is reported.
    """

    address_plan: MortonAddressPlan
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    maximum_candidates: int = eqx.field(static=True)
    target_chunk_size: int = eqx.field(static=True)
    inclusive: bool = eqx.field(static=True)
    distance_backend: SpatialDistanceBackend = eqx.field(static=True)
    pallas_interpret: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        source_capacity: int,
        target_capacity: int,
        maximum_pairs: int,
        *,
        inclusive: bool = True,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
        distance_backend: SpatialDistanceBackend = "jax",
        pallas_interpret: bool = False,
    ) -> None:
        sources, targets = _validated_capacities(
            address_plan,
            source_capacity,
            target_capacity,
            target_chunk_size,
            distance_backend,
        )
        pairs = int(maximum_pairs)
        if pairs < 0:
            raise ValueError("maximum_pairs must be nonnegative.")
        candidates = (
            min(
                sources,
                max(
                    _MINIMUM_DEFAULT_CANDIDATES,
                    _DEFAULT_CANDIDATES_PER_RESULT * ceil(pairs / targets),
                ),
            )
            if maximum_candidates is None
            else int(maximum_candidates)
        )
        if candidates < 1 or candidates > sources:
            raise ValueError("maximum_candidates must lie in [1, source_capacity].")
        chunk = _resolved_chunk_size(target_chunk_size, targets, candidates)

        self.address_plan = address_plan
        self.source_capacity = sources
        self.target_capacity = targets
        self.maximum_pairs = pairs
        self.maximum_candidates = candidates
        self.target_chunk_size = chunk
        self.inclusive = bool(inclusive)
        self.distance_backend = distance_backend
        self.pallas_interpret = bool(pallas_interpret)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "morton-radius-relation-plan",
                "address_plan_id": address_plan.plan_id,
                "source_capacity": sources,
                "target_capacity": targets,
                "maximum_pairs": pairs,
                "maximum_candidates": candidates,
                "target_chunk_size": chunk,
                "inclusive": bool(inclusive),
                "distance_backend": distance_backend,
                "pallas_interpret": bool(pallas_interpret),
            }
        )

    def _radius_chunk(
        self,
        inputs: _QueryInputs,
        rows: _TargetRows,
        level: int,
        radius: float,
        *,
        exclude_self: bool,
        pair_once: bool,
    ) -> _RadiusRows:
        sources = inputs.sources
        batch = rows.valid.shape[0]
        dtype = rows.coordinates.dtype
        starts, counts, certified_radius = _stencil_spans(
            self.address_plan,
            sources,
            rows,
            jnp.full((batch,), level, dtype=jnp.int32),
            jnp.full((batch,), radius, dtype=dtype),
            inputs.margin,
        )
        storage, candidate_valid, required = _gather_slots(
            starts, counts, self.maximum_candidates
        )
        relative = _minimum_image(
            rows.coordinates[:, None, :] - sources.coordinates[storage],
            self.address_plan,
        )
        distance_squared = _squared_norm(
            relative,
            backend=self.distance_backend,
            pallas_interpret=self.pallas_interpret,
        )
        if self.inclusive:
            candidate_valid = candidate_valid & (distance_squared <= radius**2)
        else:
            candidate_valid = candidate_valid & (distance_squared < radius**2)
        source_ids = sources.order.sorted_stable_ids[storage]
        if exclude_self:
            candidate_valid = candidate_valid & (source_ids != rows.stable_ids[:, None])
        if pair_once:
            # Each unordered pair is emitted by its smaller-ID endpoint row.
            candidate_valid = candidate_valid & (source_ids > rows.stable_ids[:, None])
        maximum_id = jnp.asarray(jnp.iinfo(source_ids.dtype).max, dtype=source_ids.dtype)
        _, _, ordered_storage = jax.lax.sort(
            (
                (~candidate_valid).astype(jnp.int32),
                jnp.where(candidate_valid, source_ids, maximum_id),
                storage,
            ),
            dimension=1,
            num_keys=2,
        )
        return _RadiusRows(
            sources=sources.order.storage_to_logical[ordered_storage],
            counts=jnp.sum(candidate_valid, axis=1, dtype=jnp.int32),
            certified=(radius < certified_radius) | jnp.isposinf(certified_radius),
            overflow=required > self.maximum_candidates,
            required=required,
        )

    def _pack_pairs(
        self,
        rows: _RadiusRows,
        row_counts: jax.Array,
        target_ids: jax.Array,
        successful: jax.Array,
        *,
        pair_once: bool,
    ) -> EdgeRelation:
        # Rows are emitted in target stable-ID order and each row is sorted by
        # source stable ID, so the packed pairs are canonical without a global sort.
        row_order = jnp.argsort(target_ids, stable=True).astype(jnp.int32)
        ordered_counts = row_counts[row_order]
        cumulative = jnp.cumsum(ordered_counts)
        slots = jnp.arange(self.maximum_pairs, dtype=jnp.int32)
        row = jnp.minimum(
            jnp.searchsorted(cumulative, slots, side="right"),
            self.target_capacity - 1,
        )
        rank = slots - (cumulative[row] - ordered_counts[row])
        target = row_order[row]
        source = rows.sources[target, jnp.clip(rank, 0, self.maximum_candidates - 1)]
        valid = (slots < cumulative[-1]) & successful
        first, second = (target, source) if pair_once else (source, target)
        return EdgeRelation(
            jnp.where(valid, first, 0).astype(jnp.int32),
            jnp.where(valid, second, 0).astype(jnp.int32),
            source_size=self.source_capacity,
            target_size=self.target_capacity,
            valid=valid,
        )

    def query(
        self,
        source_points: jax.Array,
        target_points: jax.Array,
        radius: float,
        *,
        source_mask: jax.Array | None = None,
        target_mask: jax.Array | None = None,
        source_stable_ids: jax.Array | None = None,
        target_stable_ids: jax.Array | None = None,
        exclude_self: bool = False,
        pair_once: bool = False,
    ) -> MortonRadiusRelationResult:
        radius_value = float(radius)
        if not np.isfinite(radius_value) or radius_value <= 0:
            raise ValueError("radius must be finite and positive.")
        if pair_once and self.source_capacity != self.target_capacity:
            raise ValueError("pair_once requires equal source and target capacities.")
        inputs = _prepare_query(
            self.address_plan,
            source_points,
            target_points,
            source_capacity=self.source_capacity,
            target_capacity=self.target_capacity,
            source_mask=source_mask,
            target_mask=target_mask,
            source_stable_ids=source_stable_ids,
            target_stable_ids=target_stable_ids,
        )
        level = _static_level_wider_than(self.address_plan, radius_value, inputs.margin)
        rows = _map_target_chunks(
            lambda chunk: self._radius_chunk(
                inputs,
                chunk,
                level,
                radius_value,
                exclude_self=bool(exclude_self),
                pair_once=bool(pair_once),
            ),
            inputs.rows,
            self.target_chunk_size,
        )
        order = inputs.sources.order
        status, evidence = _row_outcomes(
            inputs, rows.overflow, rows.certified, rows.required
        )
        row_counts = jnp.where(
            status == MortonNeighborQueryStatus.COMPLETE, rows.counts, 0
        )
        required_pairs = jnp.sum(row_counts, dtype=jnp.int32)
        pair_overflow = required_pairs > self.maximum_pairs
        identity_compatible = (
            jnp.all(order.stable_ids == inputs.rows.stable_ids)
            if pair_once
            else jnp.asarray(True)
        )
        successful = (
            evidence["complete"]
            & evidence["sources_valid"]
            & (evidence["invalid_targets"] == 0)
            & identity_compatible
            & ~pair_overflow
        )
        relation = self._pack_pairs(
            rows,
            row_counts,
            inputs.rows.stable_ids,
            successful,
            pair_once=bool(pair_once),
        )
        cell_slots, cell_counts, cell_offsets = _cell_table(
            self.address_plan, order, level
        )
        return MortonRadiusRelationResult(
            relation=relation,
            status=status,
            evidence=MortonRadiusRelationEvidence(
                successful=successful,
                cell_level=jnp.asarray(level, dtype=jnp.int32),
                maximum_cell_occupancy=jnp.max(cell_counts, initial=0),
                candidate_capacity=jnp.asarray(self.maximum_candidates, dtype=jnp.int32),
                required_pairs=required_pairs,
                pair_capacity=jnp.asarray(self.maximum_pairs, dtype=jnp.int32),
                pair_overflow=pair_overflow,
                **evidence,
            ),
            storage_to_logical=order.storage_to_logical,
            logical_to_storage=order.logical_to_storage,
            logical_cell_slots=cell_slots,
            cell_counts=cell_counts,
            cell_offsets=cell_offsets,
        )


__all__ = [
    "MortonNeighborQueryEvidence",
    "MortonNeighborQueryPlan",
    "MortonNeighborQueryResult",
    "MortonNeighborQueryStatus",
    "MortonRadiusRelationEvidence",
    "MortonRadiusRelationPlan",
    "MortonRadiusRelationResult",
    "SpatialDistanceBackend",
]
