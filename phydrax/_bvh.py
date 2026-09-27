#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Packed axis-aligned bounding-volume hierarchies.

`prepare_bvh` is the single host construction entry point (median, Morton
radix-tree, or binned surface-area-heuristic splits).  Every build produces the
same level-ordered `PackedBVH`, whose device queries use explicit bounded stacks
and whose fixed topology is refitted to new item bounds with one vectorized
update per tree level.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from enum import StrEnum
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ._strict import StrictModule
from ._trainable import NonTrainableState


class BVHBuildKind(StrEnum):
    """Host construction strategy of a packed BVH."""

    MEDIAN = "median"
    MORTON = "morton"
    SAH = "sah"


_MORTON_CODE_BITS = (30, 63)
_MAXIMUM_SAH_BINS = 256
# Upper bound on item-pair tests evaluated together by the host overlap search.
_HOST_ITEM_PAIR_BLOCK = 1 << 20
_HOST_NODE_PAIR_BLOCK = 1 << 16


class BVHBuildPolicy(StrictModule):
    """Construction strategy, leaf payload capacity, and SAH bin budget.

    `MEDIAN` splits the longest node-box axis at the median item center.
    `MORTON` builds the Karras (2012) radix tree over sorted `morton_code_bits`
    (30 or 63) Morton codes of item centers and collapses subtrees that fit one
    leaf.  `SAH` minimizes the binned surface-area heuristic over `sah_bins`
    candidate planes per axis.  Every strategy splits each node holding more than
    `leaf_size` items, so leaf payloads stay bounded, and orders ties by item index.
    """

    kind: BVHBuildKind = eqx.field(static=True)
    leaf_size: int = eqx.field(static=True)
    sah_bins: int = eqx.field(static=True)
    morton_code_bits: int = eqx.field(static=True)

    def __init__(
        self,
        kind: BVHBuildKind = BVHBuildKind.MEDIAN,
        leaf_size: int = 16,
        sah_bins: int = 16,
        *,
        morton_code_bits: int = 63,
    ) -> None:
        if not isinstance(kind, BVHBuildKind):
            raise TypeError("kind must be a BVHBuildKind.")
        for name, value in (
            ("leaf_size", leaf_size),
            ("sah_bins", sah_bins),
            ("morton_code_bits", morton_code_bits),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be positive, got {leaf_size}.")
        if not 2 <= sah_bins <= _MAXIMUM_SAH_BINS:
            raise ValueError(
                f"sah_bins must lie in [2, {_MAXIMUM_SAH_BINS}], got {sah_bins}."
            )
        if morton_code_bits not in _MORTON_CODE_BITS:
            raise ValueError(
                f"morton_code_bits must be one of {_MORTON_CODE_BITS}, "
                f"got {morton_code_bits}."
            )
        self.kind = kind
        self.leaf_size = int(leaf_size)
        self.sah_bins = int(sah_bins)
        self.morton_code_bits = int(morton_code_bits)


class PackedBVH(StrictModule, NonTrainableState):
    """Level-ordered packed binary AABB hierarchy with fixed-size leaf payloads.

    Node 0 is the root and the nodes of depth `d` occupy
    `level_offsets[d]:level_offsets[d + 1]`, so every child lies in the level
    after its parent.  Internal nodes have `leaf_id == -1`; leaves have
    `left == right == -1` and list at most `leaf_size` item indices (padded with
    `-1`) in `leaf_items`.  Node bounds contain the stored item bounds.
    """

    bbox_min: Array
    bbox_max: Array
    left: Array
    right: Array
    leaf_id: Array
    leaf_items: Array
    leaf_node: Array
    item_bbox_min: Array
    item_bbox_max: Array
    leaf_size: int = eqx.field(static=True)
    max_depth: int = eqx.field(static=True)
    level_offsets: tuple[int, ...] = eqx.field(static=True)
    build_kind: BVHBuildKind = eqx.field(static=True)

    @property
    def node_count(self) -> int:
        return self.left.shape[0]

    @property
    def item_count(self) -> int:
        return self.item_bbox_min.shape[0]

    @property
    def dimension(self) -> int:
        return self.bbox_min.shape[1]


# Host construction.  Builders return per-level node lists in breadth-first
# order: item permutation `order` plus, for each level, the node ranges
# `[start, stop)` into `order` and the child node ids (`-1` at leaves).
_Levels = tuple[
    np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]
]


def _segment_positions(
    starts: np.ndarray, stops: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = stops - starts
    offsets = np.concatenate((np.zeros((1,), dtype=np.int64), np.cumsum(counts)[:-1]))
    segment = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    positions = (
        starts[segment] + np.arange(segment.shape[0], dtype=np.int64) - offsets[segment]
    )
    return positions, segment, offsets


def _build_split_levels(
    item_count: int,
    leaf_size: int,
    split: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    /,
) -> _Levels:
    """Level-synchronous top-down build; `split` reorders segments in place."""
    order = np.arange(item_count, dtype=np.int64)
    level_start = np.zeros((1,), dtype=np.int64)
    level_stop = np.full((1,), item_count, dtype=np.int64)
    starts: list[np.ndarray] = []
    stops: list[np.ndarray] = []
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    node_count = 1
    while level_start.shape[0] > 0:
        divide = level_stop - level_start > leaf_size
        left = np.full(level_start.shape, -1, dtype=np.int64)
        right = np.full(level_start.shape, -1, dtype=np.int64)
        segment_start = level_start[divide]
        segment_stop = level_stop[divide]
        middle = (
            split(order, segment_start, segment_stop)
            if segment_start.shape[0] > 0
            else segment_start
        )
        children = node_count + np.arange(2 * segment_start.shape[0], dtype=np.int64)
        left[divide] = children[0::2]
        right[divide] = children[1::2]
        node_count += children.shape[0]
        starts.append(level_start)
        stops.append(level_stop)
        lefts.append(left)
        rights.append(right)
        level_start = np.stack((segment_start, middle), axis=1).reshape((-1,))
        level_stop = np.stack((middle, segment_stop), axis=1).reshape((-1,))
    return order, starts, stops, lefts, rights


def _median_split(lower: np.ndarray, upper: np.ndarray, centers: np.ndarray, /) -> Any:
    def split(order: Any, segment_start: Any, segment_stop: Any) -> Any:
        positions, segment, offsets = _segment_positions(segment_start, segment_stop)
        items = order[positions]
        box_min = np.minimum.reduceat(lower[items], offsets, axis=0)
        box_max = np.maximum.reduceat(upper[items], offsets, axis=0)
        axis = np.argmax(box_max - box_min, axis=1)
        key = centers[items, axis[segment]]
        order[positions] = items[np.lexsort((items, key, segment))]
        return segment_start + (segment_stop - segment_start) // 2

    return split


def _half_area(extent: np.ndarray, /) -> np.ndarray:
    """Half the boundary measure of boxes; comparisons only need proportionality."""
    dimension = extent.shape[-1]
    if dimension == 1:
        return extent[..., 0]
    area = np.zeros(extent.shape[:-1], dtype=extent.dtype)
    for first in range(dimension):
        for second in range(first + 1, dimension):
            area = area + extent[..., first] * extent[..., second]
    return area


def _sah_split(
    lower: np.ndarray, upper: np.ndarray, centers: np.ndarray, bins: int, /
) -> Any:
    def split(order: Any, segment_start: Any, segment_stop: Any) -> Any:
        positions, segment, offsets = _segment_positions(segment_start, segment_stop)
        items = order[positions]
        counts = segment_stop - segment_start
        segment_count = counts.shape[0]
        dimension = centers.shape[1]
        item_centers = centers[items]
        item_lower = lower[items]
        item_upper = upper[items]
        center_min = np.minimum.reduceat(item_centers, offsets, axis=0)
        center_extent = np.maximum.reduceat(item_centers, offsets, axis=0) - center_min
        best_cost = np.full((segment_count,), np.inf, dtype=np.float64)
        best_axis = np.zeros((segment_count,), dtype=np.int64)
        best_plane = np.zeros((segment_count,), dtype=np.int64)
        item_bins = np.empty((items.shape[0], dimension), dtype=np.int64)
        for axis in range(dimension):
            extent = center_extent[:, axis]
            scale = np.where(
                extent > 0.0, bins / np.where(extent > 0.0, extent, 1.0), 0.0
            )
            bin_index = np.clip(
                np.floor(
                    (item_centers[:, axis] - center_min[segment, axis]) * scale[segment]
                ).astype(np.int64),
                0,
                bins - 1,
            )
            item_bins[:, axis] = bin_index
            flat = segment * bins + bin_index
            bin_count = np.bincount(flat, minlength=segment_count * bins).reshape(
                (segment_count, bins)
            )
            bin_min = np.full((segment_count * bins, dimension), np.inf)
            bin_max = np.full((segment_count * bins, dimension), -np.inf)
            np.minimum.at(bin_min, flat, item_lower)
            np.maximum.at(bin_max, flat, item_upper)
            bin_min = bin_min.reshape((segment_count, bins, dimension))
            bin_max = bin_max.reshape((segment_count, bins, dimension))
            left_count = np.cumsum(bin_count, axis=1)[:, :-1]
            right_count = np.cumsum(bin_count[:, ::-1], axis=1)[:, ::-1][:, 1:]
            left_extent = (
                np.maximum.accumulate(bin_max, axis=1)[:, :-1]
                - np.minimum.accumulate(bin_min, axis=1)[:, :-1]
            )
            right_extent = (
                np.maximum.accumulate(bin_max[:, ::-1], axis=1)[:, ::-1][:, 1:]
                - np.minimum.accumulate(bin_min[:, ::-1], axis=1)[:, ::-1][:, 1:]
            )
            separable = (left_count > 0) & (right_count > 0)
            cost = np.where(
                separable,
                _half_area(np.where(separable[..., None], left_extent, 0.0)) * left_count
                + _half_area(np.where(separable[..., None], right_extent, 0.0))
                * right_count,
                np.inf,
            )
            plane = np.argmin(cost, axis=1)
            axis_cost = cost[np.arange(segment_count), plane]
            better = axis_cost < best_cost
            best_cost = np.where(better, axis_cost, best_cost)
            best_axis = np.where(better, axis, best_axis)
            best_plane = np.where(better, plane, best_plane)
        # Centers sharing one bin on every axis admit no separating plane; such
        # segments split at their current median position instead.
        separable = np.isfinite(best_cost)[segment]
        rank = np.arange(items.shape[0], dtype=np.int64) - offsets[segment]
        side = np.where(
            separable,
            item_bins[np.arange(items.shape[0]), best_axis[segment]]
            > best_plane[segment],
            rank >= counts[segment] // 2,
        )
        order[positions] = items[np.lexsort((items, side, segment))]
        left_count = np.bincount(
            segment, weights=(~side).astype(np.float64), minlength=segment_count
        ).astype(np.int64)
        return segment_start + left_count

    return split


def _bit_length(values: np.ndarray, /) -> np.ndarray:
    remaining = values.astype(np.uint64, copy=True)
    length = np.zeros(values.shape, dtype=np.int64)
    for shift in (32, 16, 8, 4, 2, 1):
        high = (remaining >> np.uint64(shift)) != 0
        length = length + np.where(high, shift, 0)
        remaining = np.where(high, remaining >> np.uint64(shift), remaining)
    return length + (remaining != 0)


def _morton_codes(centers: np.ndarray, code_bits: int, /) -> np.ndarray:
    # Lazy import: the discretization package imports this module at load time.
    from .discretization.spatial._morton import morton_encode_integer

    dimension = centers.shape[1]
    depth = code_bits // dimension
    lower = np.min(centers, axis=0)
    extent = np.max(centers, axis=0) - lower
    normalized = (centers - lower) / np.where(extent > 0.0, extent, 1.0)
    resolution = np.float64(2.0) ** depth
    integer = np.floor(
        np.minimum(normalized * resolution, np.nextafter(resolution, 0.0))
    ).astype(np.uint64)
    return np.asarray(
        morton_encode_integer(jnp.asarray(integer, dtype=jnp.uint64), depth)
    )


def _build_morton_levels(
    lower: np.ndarray, upper: np.ndarray, leaf_size: int, code_bits: int, /
) -> _Levels:
    """Karras (2012) radix tree over sorted Morton codes, collapsed to leaves."""
    item_count, dimension = lower.shape
    if dimension > 3:
        raise ValueError("MORTON BVH construction supports dimensions 1, 2, and 3.")
    codes = _morton_codes(0.5 * (lower + upper), code_bits)
    order = np.lexsort((np.arange(item_count, dtype=np.int64), codes)).astype(np.int64)
    keys = codes[order]
    internal_count = item_count - 1
    # Radix-tree arrays have the fixed capacity 2N - 1: internal nodes first,
    # then one single-item leaf per sorted position.
    range_start = np.concatenate(
        (np.zeros((internal_count,), dtype=np.int64), np.arange(item_count))
    )
    range_stop = np.concatenate(
        (np.zeros((internal_count,), dtype=np.int64), np.arange(1, item_count + 1))
    )
    tree_left = np.full((2 * item_count - 1,), -1, dtype=np.int64)
    tree_right = np.full((2 * item_count - 1,), -1, dtype=np.int64)
    if internal_count > 0:
        node = np.arange(internal_count, dtype=np.int64)

        def common_prefix(other: np.ndarray) -> np.ndarray:
            inside = (other >= 0) & (other < item_count)
            safe = np.clip(other, 0, item_count - 1)
            difference = keys[node] ^ keys[safe]
            # Equal codes are disambiguated by their sorted position (Karras 2012).
            prefix = np.where(
                difference == 0,
                128 - _bit_length(node.astype(np.uint64) ^ safe.astype(np.uint64)),
                64 - _bit_length(difference),
            )
            return np.where(inside, prefix, -1)

        direction = np.where(common_prefix(node + 1) > common_prefix(node - 1), 1, -1)
        minimum_prefix = common_prefix(node - direction)
        bound = np.full((internal_count,), 2, dtype=np.int64)
        grow = common_prefix(node + bound * direction) > minimum_prefix
        while np.any(grow):
            bound = np.where(grow, 2 * bound, bound)
            grow = grow & (common_prefix(node + bound * direction) > minimum_prefix)
        length = np.zeros((internal_count,), dtype=np.int64)
        step = bound // 2
        while np.any(step >= 1):
            take = (step >= 1) & (
                common_prefix(node + (length + step) * direction) > minimum_prefix
            )
            length = np.where(take, length + step, length)
            step = step // 2
        other = node + length * direction
        node_prefix = common_prefix(other)
        split = np.zeros((internal_count,), dtype=np.int64)
        divisor = 2
        while True:
            step = (length + divisor - 1) // divisor
            take = common_prefix(node + (split + step) * direction) > node_prefix
            split = np.where(take, split + step, split)
            if np.all(step <= 1):
                break
            divisor *= 2
        gamma = node + split * direction + np.minimum(direction, 0)
        first = np.minimum(node, other)
        last = np.maximum(node, other)
        range_start[:internal_count] = first
        range_stop[:internal_count] = last + 1
        tree_left[:internal_count] = np.where(
            first == gamma, internal_count + gamma, gamma
        )
        tree_right[:internal_count] = np.where(
            last == gamma + 1, internal_count + gamma + 1, gamma + 1
        )
    starts: list[np.ndarray] = []
    stops: list[np.ndarray] = []
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    frontier = np.zeros((1,), dtype=np.int64)
    node_count = 1
    while frontier.shape[0] > 0:
        start = range_start[frontier]
        stop = range_stop[frontier]
        divide = stop - start > leaf_size
        left = np.full(frontier.shape, -1, dtype=np.int64)
        right = np.full(frontier.shape, -1, dtype=np.int64)
        children = node_count + np.arange(2 * np.count_nonzero(divide), dtype=np.int64)
        left[divide] = children[0::2]
        right[divide] = children[1::2]
        node_count += children.shape[0]
        starts.append(start)
        stops.append(stop)
        lefts.append(left)
        rights.append(right)
        frontier = np.stack(
            (tree_left[frontier[divide]], tree_right[frontier[divide]]), axis=1
        ).reshape((-1,))
    return order, starts, stops, lefts, rights


def _outward_cast(
    lower: np.ndarray, upper: np.ndarray, dtype: np.dtype, /
) -> tuple[np.ndarray, np.ndarray]:
    """Cast bounds to `dtype`, rounding outward so boxes stay conservative."""
    cast_lower = lower.astype(dtype)
    cast_upper = upper.astype(dtype)
    cast_lower = np.where(
        cast_lower.astype(np.float64) > lower,
        np.nextafter(cast_lower, dtype.type(-np.inf)),
        cast_lower,
    )
    cast_upper = np.where(
        cast_upper.astype(np.float64) < upper,
        np.nextafter(cast_upper, dtype.type(np.inf)),
        cast_upper,
    )
    return cast_lower, cast_upper


def _pack(
    lower: np.ndarray,
    upper: np.ndarray,
    levels: _Levels,
    policy: BVHBuildPolicy,
    dtype: np.dtype,
    /,
) -> PackedBVH:
    order, starts, stops, lefts, rights = levels
    start = np.concatenate(starts)
    stop = np.concatenate(stops)
    left = np.concatenate(lefts)
    right = np.concatenate(rights)
    level_offsets = tuple(
        int(value)
        for value in np.concatenate(
            ((0,), np.cumsum([level.shape[0] for level in starts]))
        )
    )
    leaf_node = np.flatnonzero(left < 0)
    leaf_id = np.full(left.shape, -1, dtype=np.int64)
    leaf_id[leaf_node] = np.arange(leaf_node.shape[0])
    slots = start[leaf_node, None] + np.arange(policy.leaf_size)[None, :]
    leaf_items = np.where(
        slots < stop[leaf_node, None],
        order[np.minimum(slots, order.shape[0] - 1)],
        -1,
    )
    item_lower, item_upper = _outward_cast(lower, upper, dtype)
    # Every node covers one contiguous range of `order`; the trailing sentinel
    # row keeps `reduceat` indices in range for ranges that end at the last item.
    boundaries = np.stack((start, stop), axis=1).reshape((-1,))
    sorted_lower = np.concatenate((item_lower[order], item_lower[order[:1]]))
    sorted_upper = np.concatenate((item_upper[order], item_upper[order[:1]]))
    node_lower = np.minimum.reduceat(sorted_lower, boundaries, axis=0)[0::2]
    node_upper = np.maximum.reduceat(sorted_upper, boundaries, axis=0)[0::2]
    return PackedBVH(
        bbox_min=jnp.asarray(node_lower, dtype=dtype),
        bbox_max=jnp.asarray(node_upper, dtype=dtype),
        left=jnp.asarray(left, dtype=jnp.int32),
        right=jnp.asarray(right, dtype=jnp.int32),
        leaf_id=jnp.asarray(leaf_id, dtype=jnp.int32),
        leaf_items=jnp.asarray(leaf_items, dtype=jnp.int32),
        leaf_node=jnp.asarray(leaf_node, dtype=jnp.int32),
        item_bbox_min=jnp.asarray(item_lower, dtype=dtype),
        item_bbox_max=jnp.asarray(item_upper, dtype=dtype),
        leaf_size=policy.leaf_size,
        max_depth=len(starts) - 1,
        level_offsets=level_offsets,
        build_kind=policy.kind,
    )


def prepare_bvh(
    item_bbox_min: ArrayLike,
    item_bbox_max: ArrayLike,
    /,
    *,
    policy: BVHBuildPolicy = BVHBuildPolicy(),
    dtype: DTypeLike = jnp.float32,
) -> PackedBVH:
    """Build a level-ordered packed BVH over item bounding boxes (host NumPy).

    Bounds narrower than float64 are rounded outward, so stored boxes contain the
    supplied ones.  Point sets use `prepare_bvh(points, points, ...)`.
    """
    if not isinstance(policy, BVHBuildPolicy):
        raise TypeError("policy must be a BVHBuildPolicy.")
    storage = np.dtype(dtype)
    if not np.issubdtype(storage, np.floating):
        raise TypeError("dtype must be a floating dtype.")
    lower = np.asarray(item_bbox_min, dtype=np.float64)
    upper = np.asarray(item_bbox_max, dtype=np.float64)
    if lower.ndim != 2 or upper.shape != lower.shape or lower.shape[1] == 0:
        raise ValueError(
            "item_bbox_min/max must share shape (item_count, dimension > 0)."
        )
    if lower.shape[0] == 0:
        raise ValueError("cannot build a BVH over an empty item set.")
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("item bounds must be finite.")
    if np.any(lower > upper):
        raise ValueError("every item_bbox_min component must be <= item_bbox_max.")
    centers = 0.5 * (lower + upper)
    item_count = lower.shape[0]
    match policy.kind:
        case BVHBuildKind.MEDIAN:
            levels = _build_split_levels(
                item_count, policy.leaf_size, _median_split(lower, upper, centers)
            )
        case BVHBuildKind.SAH:
            levels = _build_split_levels(
                item_count,
                policy.leaf_size,
                _sah_split(lower, upper, centers, policy.sah_bins),
            )
        case BVHBuildKind.MORTON:
            levels = _build_morton_levels(
                lower, upper, policy.leaf_size, policy.morton_code_bits
            )
        case _:
            raise ValueError(f"Unsupported BVH build kind {policy.kind!r}.")
    return _pack(lower, upper, levels, policy, storage)


def reduce_packed_bvh_nodes(
    bvh: PackedBVH,
    item_values: ArrayLike,
    /,
    *,
    reduction: Literal["sum", "min", "max"],
) -> Array:
    """Reduce floating item values over every node subtree, one level at a time.

    Returns shape `(node_count, *item_values.shape[1:])`.  The level sweep is
    differentiable with respect to `item_values`.
    """
    if not isinstance(bvh, PackedBVH):
        raise TypeError("bvh must be a PackedBVH.")
    values = jnp.asarray(item_values)
    if values.ndim < 1 or values.shape[0] != bvh.item_count:
        raise ValueError(f"item_values must have leading dimension {bvh.item_count}.")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("item_values must be floating.")
    match reduction:
        case "sum":
            identity = jnp.asarray(0.0, dtype=values.dtype)
            combine = jnp.add
            reduce = jnp.sum
        case "min":
            identity = jnp.asarray(jnp.inf, dtype=values.dtype)
            combine = jnp.minimum
            reduce = jnp.min
        case "max":
            identity = jnp.asarray(-jnp.inf, dtype=values.dtype)
            combine = jnp.maximum
            reduce = jnp.max
        case _:
            raise ValueError(f"Unsupported BVH node reduction {reduction!r}.")
    trailing = (1,) * (values.ndim - 1)
    valid = bvh.leaf_items >= 0
    gathered = values[jnp.maximum(bvh.leaf_items, 0)]
    leaf_values = reduce(
        jnp.where(valid.reshape(valid.shape + trailing), gathered, identity), axis=1
    )
    nodes = jnp.full((bvh.node_count,) + values.shape[1:], identity, dtype=values.dtype)
    nodes = nodes.at[bvh.leaf_node].set(leaf_values)
    # Children occupy the level after their parent, so one vectorized update per
    # level (deepest first) finalizes every internal node.
    for level in range(bvh.max_depth - 1, -1, -1):
        start = bvh.level_offsets[level]
        stop = bvh.level_offsets[level + 1]
        left = bvh.left[start:stop]
        internal = (left >= 0).reshape((stop - start,) + trailing)
        combined = combine(
            nodes[jnp.maximum(left, 0)], nodes[jnp.maximum(bvh.right[start:stop], 0)]
        )
        nodes = nodes.at[start:stop].set(jnp.where(internal, combined, nodes[start:stop]))
    return nodes


def refit_packed_bvh_bounds(
    bvh: PackedBVH,
    item_bbox_min: ArrayLike,
    item_bbox_max: ArrayLike,
    /,
) -> PackedBVH:
    """Refit fixed BVH topology to differentiable current item bounds."""
    if not isinstance(bvh, PackedBVH):
        raise TypeError("bvh must be a PackedBVH.")
    item_min = jnp.asarray(item_bbox_min, dtype=bvh.bbox_min.dtype)
    item_max = jnp.asarray(item_bbox_max, dtype=bvh.bbox_min.dtype)
    expected = bvh.item_bbox_min.shape
    if item_min.shape != expected or item_max.shape != expected:
        raise ValueError(f"item_bbox_min/max must have shape {expected}.")
    return eqx.tree_at(
        lambda tree: (
            tree.bbox_min,
            tree.bbox_max,
            tree.item_bbox_min,
            tree.item_bbox_max,
        ),
        bvh,
        (
            reduce_packed_bvh_nodes(bvh, item_min, reduction="min"),
            reduce_packed_bvh_nodes(bvh, item_max, reduction="max"),
            item_min,
            item_max,
        ),
    )


def aabb_dist2(p: Array, bmin: Array, bmax: Array, /) -> Array:
    z = jnp.asarray(0.0, dtype=p.dtype)
    d = jnp.maximum(z, jnp.maximum(bmin - p, p - bmax))
    return jnp.sum(d * d, axis=-1)


def beam_select_nodes(
    points: Array,
    /,
    *,
    bvh: PackedBVH,
    beam_width: int,
    steps: int,
) -> Array:
    """Beam traverse BVH by AABB lower bounds. Returns node ids with shape (N, B)."""
    if beam_width <= 0:
        raise ValueError(f"beam_width must be positive, got {beam_width}.")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")

    pts = jnp.asarray(points, dtype=bvh.bbox_min.dtype)
    if pts.ndim == 1:
        pts = pts.reshape((1, -1))
    if pts.ndim != 2:
        raise ValueError("points must have shape (N, dim) or (dim,).")

    B = int(beam_width)
    inf = jnp.asarray(jnp.inf, dtype=pts.dtype)

    bbox_min = bvh.bbox_min
    bbox_max = bvh.bbox_max
    left = bvh.left
    right = bvh.right
    leaf_id = bvh.leaf_id

    nodes = jnp.full((pts.shape[0], B), jnp.int32(-1))
    nodes = nodes.at[:, 0].set(jnp.int32(0))

    def _step(_: Any, nodes: Any) -> Any:
        valid = nodes >= 0
        safe = jnp.where(valid, nodes, jnp.int32(0))
        is_leaf = valid & (leaf_id[safe] >= 0)

        lch = jnp.where(valid, left[safe], jnp.int32(-1))
        rch = jnp.where(valid, right[safe], jnp.int32(-1))

        cand0 = jnp.where(is_leaf, safe, lch)
        cand1 = jnp.where(is_leaf, jnp.int32(-1), rch)
        cand_nodes = jnp.concatenate([cand0, cand1], axis=1)  # (N, 2B)

        cand_safe = jnp.where(cand_nodes >= 0, cand_nodes, jnp.int32(0))
        bmin = bbox_min[cand_safe]
        bmax = bbox_max[cand_safe]
        d2 = aabb_dist2(pts[:, None, :], bmin, bmax)
        d2 = jnp.where(cand_nodes >= 0, d2, inf)

        _, idx = jax.lax.top_k(-d2, B)
        return jnp.take_along_axis(cand_nodes, idx, axis=1)

    nodes = jax.lax.fori_loop(0, int(steps), _step, nodes)
    return nodes


def beam_select_leaf_items(
    points: Array,
    /,
    *,
    bvh: PackedBVH,
    beam_width: int,
    steps: int,
) -> tuple[Array, Array]:
    """Return candidate leaf items (and validity mask) using beam BVH traversal."""
    pts = jnp.asarray(points, dtype=bvh.bbox_min.dtype)
    is_single = pts.ndim == 1
    if is_single:
        pts = pts.reshape((1, -1))
    if pts.ndim != 2:
        raise ValueError("points must have shape (N, dim) or (dim,).")

    nodes = beam_select_nodes(pts, bvh=bvh, beam_width=beam_width, steps=steps)

    safe_nodes = jnp.where(nodes >= 0, nodes, jnp.int32(0))
    lids = bvh.leaf_id[safe_nodes]
    valid_leaf = (nodes >= 0) & (lids >= 0)
    safe_lids = jnp.where(valid_leaf, lids, jnp.int32(0))

    items = bvh.leaf_items[safe_lids]  # (N, B, leaf_size)
    items = jnp.where(valid_leaf[..., None], items, jnp.int32(-1))
    items = items.reshape((pts.shape[0], int(beam_width * bvh.leaf_size)))

    valid = items >= 0
    safe_items = jnp.where(valid, items, jnp.int32(0))
    if is_single:
        return safe_items.reshape((-1,)), valid.reshape((-1,))
    return safe_items, valid


def _bounded_leaf_candidates(
    overlaps: Any,
    bvh: PackedBVH,
    maximum_candidates: int,
    /,
) -> tuple[Array, Array, Array]:
    capacity = int(maximum_candidates)
    if capacity <= 0:
        raise ValueError("maximum_candidates must be positive.")
    node_count = bvh.left.shape[0]
    stack_capacity = min(node_count, max(1, int(bvh.max_depth) + 2))
    stack = jnp.full((stack_capacity,), -1, dtype=jnp.int32).at[0].set(0)
    candidates = jnp.full((capacity,), -1, dtype=jnp.int32)

    def visit_leaf(leaf_items: Any, state: Any) -> Any:
        values, count = state

        def append_item(slot: Any, carry: Any) -> Any:
            current, current_count = carry
            item = leaf_items[slot]
            item_valid = item >= 0
            has_space = current_count < capacity
            target = jnp.minimum(current_count, capacity - 1)
            current = current.at[target].set(
                jnp.where(item_valid & has_space, item, current[target])
            )
            return current, current_count + item_valid.astype(jnp.int32)

        return jax.lax.fori_loop(
            0,
            bvh.leaf_size,
            append_item,
            (values, count),
        )

    def continue_traversal(state: Any) -> Any:
        _, top, _, _, visits, stack_overflow = state
        return (top > 0) & (visits < node_count) & ~stack_overflow

    def traverse(state: Any) -> Any:
        current_stack, top, values, count, visits, stack_overflow = state
        popped_top = top - 1
        node = current_stack[popped_top]
        hit = overlaps(
            bvh.bbox_min[node],
            bvh.bbox_max[node],
        )
        leaf = hit & (bvh.leaf_id[node] >= 0)
        internal = hit & ~leaf
        leaf_index = jnp.maximum(bvh.leaf_id[node], 0)
        values, count = jax.lax.cond(
            leaf,
            lambda carry: visit_leaf(bvh.leaf_items[leaf_index], carry),
            lambda carry: carry,
            (values, count),
        )
        can_push = popped_top + 1 < stack_capacity
        push = internal & can_push
        stack_overflow = stack_overflow | (internal & ~can_push)
        left = jnp.maximum(bvh.left[node], 0)
        right = jnp.maximum(bvh.right[node], 0)
        current_stack = current_stack.at[popped_top].set(
            jnp.where(push, left, current_stack[popped_top])
        )
        second_slot = jnp.minimum(popped_top + 1, stack_capacity - 1)
        current_stack = current_stack.at[second_slot].set(
            jnp.where(push, right, current_stack[second_slot])
        )
        next_top = jnp.where(push, popped_top + 2, popped_top)
        return (
            current_stack,
            next_top,
            values,
            count,
            visits + 1,
            stack_overflow,
        )

    _, top, candidates, count, _, stack_overflow = jax.lax.while_loop(
        continue_traversal,
        traverse,
        (
            stack,
            jnp.asarray(1, dtype=jnp.int32),
            candidates,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        ),
    )
    retained = jnp.minimum(count, capacity)
    valid = jnp.arange(capacity, dtype=jnp.int32) < retained
    complete = (count <= capacity) & ~stack_overflow & (top == 0)
    return jnp.maximum(candidates, 0), valid, complete


def _map_bounded_queries(query: Any, arguments: Any, query_batch_capacity: int, /) -> Any:
    count = arguments[0].shape[0]
    capacity = int(query_batch_capacity)
    if capacity <= 0:
        raise ValueError("query_batch_capacity must be positive.")
    if count <= capacity:
        return jax.vmap(query)(*arguments)
    chunk_count = (count + capacity - 1) // capacity
    padded_count = chunk_count * capacity
    padding = padded_count - count
    padded = tuple(
        jnp.concatenate(
            (
                value,
                jnp.broadcast_to(value[-1], (padding,) + value.shape[1:]),
            ),
            axis=0,
        ).reshape((chunk_count, capacity) + value.shape[1:])
        for value in arguments
    )
    mapped = jax.lax.map(lambda chunk: jax.vmap(query)(*chunk), padded)
    return jax.tree.map(
        lambda value: value.reshape((padded_count,) + value.shape[2:])[:count],
        mapped,
    )


def point_select_leaf_items(
    points: ArrayLike,
    /,
    *,
    bvh: PackedBVH,
    maximum_candidates: int,
    tolerance: ArrayLike = 0.0,
    query_batch_capacity: int = 64,
) -> tuple[Array, Array, Array]:
    """Return all bounded leaf candidates whose node boxes contain each point."""
    values = jnp.asarray(points, dtype=bvh.bbox_min.dtype)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[-1] != bvh.bbox_min.shape[-1]:
        raise ValueError("points must have shape (queries, dimension) or (dimension,).")
    padding = jnp.asarray(tolerance, dtype=values.dtype)
    if padding.shape != ():
        raise ValueError("tolerance must be a scalar.")
    if isinstance(tolerance, (int, float)) and tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")

    def query(point: Any) -> Any:
        return _bounded_leaf_candidates(
            lambda lower, upper: jnp.all(
                (point >= lower - padding) & (point <= upper + padding)
            ),
            bvh,
            maximum_candidates,
        )

    if single:
        return query(values[0])
    return _map_bounded_queries(
        query,
        (values,),
        query_batch_capacity,
    )


def ray_select_leaf_items(
    origins: ArrayLike,
    directions: ArrayLike,
    /,
    *,
    bvh: PackedBVH,
    maximum_candidates: int,
    minimum_parameter: ArrayLike = 0.0,
    maximum_parameter: ArrayLike = jnp.inf,
    query_batch_capacity: int = 64,
) -> tuple[Array, Array, Array]:
    """Return bounded leaf candidates intersected by each ray."""
    ray_origins = jnp.asarray(origins, dtype=bvh.bbox_min.dtype)
    ray_directions = jnp.asarray(directions, dtype=ray_origins.dtype)
    single = ray_origins.ndim == 1
    if single:
        ray_origins = ray_origins[None, :]
        ray_directions = ray_directions[None, :]
    if (
        ray_origins.ndim != 2
        or ray_origins.shape != ray_directions.shape
        or ray_origins.shape[-1] != bvh.bbox_min.shape[-1]
    ):
        raise ValueError(
            "origins and directions must share shape (rays, dimension) or (dimension,)."
        )
    parameter_shape = (ray_origins.shape[0],)
    lower_parameter = jnp.broadcast_to(
        jnp.asarray(minimum_parameter, dtype=ray_origins.dtype),
        parameter_shape,
    )
    upper_parameter = jnp.broadcast_to(
        jnp.asarray(maximum_parameter, dtype=ray_origins.dtype),
        parameter_shape,
    )

    def query(
        origin: Any, direction: Any, parameter_lower: Any, parameter_upper: Any
    ) -> Any:
        def overlaps(lower: Any, upper: Any) -> Any:
            parallel = jnp.abs(direction) <= jnp.finfo(direction.dtype).tiny
            outside = parallel & ((origin < lower) | (origin > upper))
            inverse = jnp.where(parallel, 1.0, 1.0 / direction)
            first = (lower - origin) * inverse
            second = (upper - origin) * inverse
            axis_lower = jnp.where(parallel, -jnp.inf, jnp.minimum(first, second))
            axis_upper = jnp.where(parallel, jnp.inf, jnp.maximum(first, second))
            entry = jnp.maximum(jnp.max(axis_lower), parameter_lower)
            exit = jnp.minimum(jnp.min(axis_upper), parameter_upper)
            return ~jnp.any(outside) & (entry <= exit)

        return _bounded_leaf_candidates(overlaps, bvh, maximum_candidates)

    if single:
        return query(
            ray_origins[0],
            ray_directions[0],
            lower_parameter[0],
            upper_parameter[0],
        )
    return _map_bounded_queries(
        query,
        (
            ray_origins,
            ray_directions,
            lower_parameter,
            upper_parameter,
        ),
        query_batch_capacity,
    )


class BVHNearestResult(StrictModule):
    """Exact nearest items per query ordered by (squared distance, item index).

    `items` is `-1` and `distance_squared` is infinite where fewer than `k`
    items have a finite distance.
    """

    items: Array
    distance_squared: Array


def _item_box_distance(bvh: PackedBVH, /) -> Any:
    def distance(point: Array, items: Array) -> Array:
        return aabb_dist2(point, bvh.item_bbox_min[items], bvh.item_bbox_max[items])

    return distance


def _nearest_items_one(bvh: PackedBVH, point: Array, k: int, distance: Any, /) -> Any:
    infinity = jnp.asarray(jnp.inf, dtype=point.dtype)
    sentinel = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    # Depth-first order keeps at most one pending sibling per depth, so
    # `max_depth + 2` slots bound the stack exactly.
    stack = jnp.zeros((bvh.max_depth + 2,), dtype=jnp.int32)

    def visit_leaf(node: Any, state: Any) -> Any:
        stack_, top, best_distance, best_key = state
        items = bvh.leaf_items[jnp.maximum(bvh.leaf_id[node], 0)]
        candidate = distance(point, jnp.maximum(items, 0)).astype(point.dtype)
        valid = (items >= 0) & (candidate < infinity)
        merged_distance = jnp.concatenate(
            (best_distance, jnp.where(valid, candidate, infinity))
        )
        merged_key = jnp.concatenate((best_key, jnp.where(valid, items, sentinel)))
        order = jnp.lexsort((merged_key, merged_distance))[:k]
        return stack_, top, merged_distance[order], merged_key[order]

    def visit_internal(node: Any, state: Any) -> Any:
        stack_, top, best_distance, best_key = state
        left = bvh.left[node]
        right = bvh.right[node]
        left_bound = aabb_dist2(point, bvh.bbox_min[left], bvh.bbox_max[left])
        right_bound = aabb_dist2(point, bvh.bbox_min[right], bvh.bbox_max[right])
        near_left = left_bound <= right_bound
        stack_ = stack_.at[top].set(jnp.where(near_left, right, left))
        stack_ = stack_.at[top + 1].set(jnp.where(near_left, left, right))
        return stack_, top + 2, best_distance, best_key

    def body(state: Any) -> Any:
        stack_, top, best_distance, best_key = state
        top = top - 1
        node = stack_[top]
        bound = aabb_dist2(point, bvh.bbox_min[node], bvh.bbox_max[node])
        # Ties at the k-th distance are explored so equal distances resolve by
        # item index.
        visit = bound <= best_distance[k - 1]

        def active(carry: Any) -> Any:
            return jax.lax.cond(
                bvh.leaf_id[node] >= 0,
                lambda value: visit_leaf(node, value),
                lambda value: visit_internal(node, value),
                carry,
            )

        return jax.lax.cond(
            visit, active, lambda carry: carry, (stack_, top, best_distance, best_key)
        )

    _, _, best_distance, best_key = jax.lax.while_loop(
        lambda state: state[1] > 0,
        body,
        (
            stack,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.full((k,), infinity),
            jnp.full((k,), sentinel),
        ),
    )
    return jnp.where(best_key == sentinel, -1, best_key), best_distance


def bvh_nearest_items(
    bvh: PackedBVH,
    points: ArrayLike,
    /,
    *,
    k: int = 1,
    item_distance_squared: Callable[[Array, Array], Array] | None = None,
    query_batch_capacity: int = 64,
) -> BVHNearestResult:
    """Return the exact `k` nearest items of each point by branch and bound.

    `item_distance_squared(point, items)` maps one point `(dimension,)` and leaf
    item indices `(leaf_size,)` to squared distances; it must be bounded below
    by the squared distance to each item's box.  The default is the squared
    distance to the stored item boxes, which is exact for point sets.
    """
    if not isinstance(bvh, PackedBVH):
        raise TypeError("bvh must be a PackedBVH.")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}.")
    values = jnp.asarray(points, dtype=bvh.bbox_min.dtype)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != bvh.dimension:
        raise ValueError("points must have shape (queries, dimension) or (dimension,).")
    distance = (
        _item_box_distance(bvh)
        if item_distance_squared is None
        else item_distance_squared
    )

    def query(point: Any) -> Any:
        return _nearest_items_one(bvh, point, k, distance)

    if single:
        items, distance_squared = query(values[0])
    else:
        items, distance_squared = _map_bounded_queries(
            query, (values,), query_batch_capacity
        )
    return BVHNearestResult(items=items, distance_squared=distance_squared)


def bvh_hierarchical_sum(
    bvh: PackedBVH,
    points: ArrayLike,
    /,
    *,
    open_node: Callable[[Array, Array], Array],
    far_value: Callable[[Array, Array], Array],
    leaf_value: Callable[[Array, Array], Array],
    query_batch_capacity: int = 64,
) -> Array:
    """Sum per-query contributions over the BVH cut selected by `open_node`.

    Traversal starts at the root.  A node for which `open_node(point, node)` is
    false contributes `far_value(point, node)`; an opened leaf contributes
    `leaf_value(point, leaf_index)`; an opened internal node is replaced by its
    two children.  Every item lies below exactly one cut node, so the sum is
    exact whenever `far_value` reproduces its subtree's contribution exactly and
    a Barnes-Hut style approximation otherwise.
    """
    if not isinstance(bvh, PackedBVH):
        raise TypeError("bvh must be a PackedBVH.")
    values = jnp.asarray(points, dtype=bvh.bbox_min.dtype)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != bvh.dimension:
        raise ValueError("points must have shape (queries, dimension) or (dimension,).")

    def query(point: Any) -> Any:
        def far(state: Any) -> Any:
            stack, top, total, node = state
            return stack, top, total + far_value(point, node).astype(total.dtype), node

        def leaf(state: Any) -> Any:
            stack, top, total, node = state
            contribution = leaf_value(point, bvh.leaf_id[node]).astype(total.dtype)
            return stack, top, total + contribution, node

        def internal(state: Any) -> Any:
            stack, top, total, node = state
            stack = stack.at[top].set(bvh.right[node]).at[top + 1].set(bvh.left[node])
            return stack, top + 2, total, node

        def body(state: Any) -> Any:
            stack, top, total = state
            top = top - 1
            node = stack[top]
            branch = jnp.where(
                open_node(point, node), jnp.where(bvh.leaf_id[node] >= 0, 1, 2), 0
            )
            stack, top, total, _ = jax.lax.switch(
                branch, (far, leaf, internal), (stack, top, total, node)
            )
            return stack, top, total

        _, _, total = jax.lax.while_loop(
            lambda state: state[1] > 0,
            body,
            (
                jnp.zeros((bvh.max_depth + 2,), dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
                jnp.zeros((), dtype=point.dtype),
            ),
        )
        return total

    if single:
        return query(values[0])
    return _map_bounded_queries(query, (values,), query_batch_capacity)


class BVHPairResult(StrictModule):
    """Bounded overlapping item pairs of two BVHs in lexicographic order.

    `count` (int64) is the number of overlapping pairs found; pairs beyond the
    output capacity are dropped.  `overflow` is set when pairs were dropped or
    when the traversal stack capacity was exhausted (then `count` is a lower
    bound).
    """

    first_items: Array
    second_items: Array
    valid: Array
    count: Array
    overflow: Array


def _boxes_overlap(
    first_min: Any,
    first_max: Any,
    second_min: Any,
    second_max: Any,
    padding: Any,
    include_touching: bool,
    /,
) -> Any:
    extent = (
        jnp.minimum(first_max, second_max) - jnp.maximum(first_min, second_min) + padding
    )
    if include_touching:
        return jnp.all(extent >= 0.0, axis=-1)
    return jnp.all(extent > 0.0, axis=-1)


def bvh_overlap_pairs(
    first: PackedBVH,
    second: PackedBVH,
    /,
    *,
    capacity: int,
    include_touching: bool = False,
    padding: ArrayLike = 0.0,
    traversal_width: int = 64,
) -> BVHPairResult:
    """Return overlapping item pairs by simultaneous bounded traversal (JAX).

    A pair overlaps when, on every axis, `min(max) - max(min) + padding` is
    positive (non-negative with `include_touching`).  Each step expands up to
    `traversal_width` node pairs from a stack of capacity
    `traversal_width * (first.max_depth + second.max_depth + 2)`, always
    descending the larger internal node.
    """
    if not isinstance(first, PackedBVH) or not isinstance(second, PackedBVH):
        raise TypeError("first and second must be PackedBVH instances.")
    if first.dimension != second.dimension:
        raise ValueError("first and second must share their spatial dimension.")
    for name, value in (("capacity", capacity), ("traversal_width", traversal_width)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}.")
    if not isinstance(include_touching, bool):
        raise TypeError("include_touching must be a bool.")
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "bvh_overlap_pairs counts pairs in int64; enable jax_enable_x64."
        )
    dtype = jnp.result_type(first.bbox_min.dtype, second.bbox_min.dtype)
    pad = jnp.asarray(padding, dtype=dtype)
    if pad.shape != ():
        raise ValueError("padding must be a scalar.")
    width = traversal_width
    stack_capacity = width * (first.max_depth + second.max_depth + 2)
    lanes = jnp.arange(width, dtype=jnp.int32)
    first_size = jnp.sum(first.bbox_max - first.bbox_min, axis=-1)
    second_size = jnp.sum(second.bbox_max - second.bbox_min, axis=-1)

    pair_shape = (width, first.leaf_size, second.leaf_size)

    def node_overlap(first_nodes: Any, second_nodes: Any) -> Any:
        return _boxes_overlap(
            first.bbox_min[first_nodes],
            first.bbox_max[first_nodes],
            second.bbox_min[second_nodes],
            second.bbox_max[second_nodes],
            pad,
            include_touching,
        )

    def emit(state: Any, first_nodes: Any, second_nodes: Any, both_leaf: Any) -> Any:
        first_out, second_out, count = state
        first_items = first.leaf_items[jnp.maximum(first.leaf_id[first_nodes], 0)]
        second_items = second.leaf_items[jnp.maximum(second.leaf_id[second_nodes], 0)]
        safe_first = jnp.maximum(first_items, 0)[:, :, None]
        safe_second = jnp.maximum(second_items, 0)[:, None, :]
        hit = (
            both_leaf[:, None, None]
            & (first_items >= 0)[:, :, None]
            & (second_items >= 0)[:, None, :]
            & _boxes_overlap(
                first.item_bbox_min[safe_first],
                first.item_bbox_max[safe_first],
                second.item_bbox_min[safe_second],
                second.item_bbox_max[safe_second],
                pad,
                include_touching,
            )
        ).reshape((-1,))
        position = count + jnp.cumsum(hit, dtype=jnp.int64) - 1
        target = jnp.where(hit & (position < capacity), position, capacity)
        first_out = first_out.at[target].set(
            jnp.broadcast_to(safe_first, pair_shape).reshape((-1,)), mode="drop"
        )
        second_out = second_out.at[target].set(
            jnp.broadcast_to(safe_second, pair_shape).reshape((-1,)), mode="drop"
        )
        return first_out, second_out, count + jnp.sum(hit, dtype=jnp.int64)

    def body(state: Any) -> Any:
        first_stack, second_stack, top, first_out, second_out, count, exhausted = state
        start = jnp.maximum(top - width, 0)
        first_nodes = jax.lax.dynamic_slice(first_stack, (start,), (width,))
        second_nodes = jax.lax.dynamic_slice(second_stack, (start,), (width,))
        active = start + lanes < top
        first_leaf = first.leaf_id[first_nodes] >= 0
        second_leaf = second.leaf_id[second_nodes] >= 0
        both_leaf = active & first_leaf & second_leaf
        first_out, second_out, count = emit(
            (first_out, second_out, count), first_nodes, second_nodes, both_leaf
        )
        descend_first = (
            active
            & ~first_leaf
            & (second_leaf | (first_size[first_nodes] >= second_size[second_nodes]))
        )
        descend_second = active & ~both_leaf & ~descend_first
        child_first = jnp.stack(
            (
                jnp.where(descend_first, first.left[first_nodes], first_nodes),
                jnp.where(descend_first, first.right[first_nodes], first_nodes),
            ),
            axis=1,
        ).reshape((-1,))
        child_second = jnp.stack(
            (
                jnp.where(descend_second, second.left[second_nodes], second_nodes),
                jnp.where(descend_second, second.right[second_nodes], second_nodes),
            ),
            axis=1,
        ).reshape((-1,))
        expanded = jnp.repeat(descend_first | descend_second, 2)
        push = expanded & node_overlap(
            jnp.maximum(child_first, 0), jnp.maximum(child_second, 0)
        )
        position = start + jnp.cumsum(push, dtype=jnp.int32) - 1
        fits = push & (position < stack_capacity)
        target = jnp.where(fits, position, stack_capacity)
        first_stack = first_stack.at[target].set(child_first, mode="drop")
        second_stack = second_stack.at[target].set(child_second, mode="drop")
        exhausted = exhausted | jnp.any(push & ~fits)
        top = start + jnp.sum(fits, dtype=jnp.int32)
        return first_stack, second_stack, top, first_out, second_out, count, exhausted

    root = node_overlap(jnp.asarray(0), jnp.asarray(0))
    initial = (
        jnp.zeros((stack_capacity,), dtype=jnp.int32),
        jnp.zeros((stack_capacity,), dtype=jnp.int32),
        root.astype(jnp.int32),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int64),
        jnp.asarray(False),
    )
    _, _, _, first_out, second_out, count, exhausted = jax.lax.while_loop(
        lambda state: (state[2] > 0) & ~state[6], body, initial
    )
    valid = jnp.arange(capacity, dtype=jnp.int32) < jnp.minimum(count, capacity)
    sentinel = jnp.iinfo(jnp.int32).max
    order = jnp.lexsort(
        (
            jnp.where(valid, second_out, sentinel),
            jnp.where(valid, first_out, sentinel),
        )
    )
    return BVHPairResult(
        first_items=jnp.where(valid, first_out[order], -1),
        second_items=jnp.where(valid, second_out[order], -1),
        valid=valid,
        count=count,
        overflow=(count > capacity) | exhausted,
    )


def _host_boxes_overlap(
    first_min: np.ndarray,
    first_max: np.ndarray,
    second_min: np.ndarray,
    second_max: np.ndarray,
    include_touching: bool,
    absolute_tolerance: float,
    relative_tolerance: float,
    /,
) -> np.ndarray:
    extent = np.minimum(first_max, second_max) - np.maximum(first_min, second_min)
    if not include_touching:
        return np.all(extent > 0.0, axis=-1)
    scale = np.maximum(
        1.0,
        np.maximum(
            np.abs(first_min),
            np.maximum(
                np.abs(first_max), np.maximum(np.abs(second_min), np.abs(second_max))
            ),
        ),
    )
    return np.all(extent >= -(absolute_tolerance + relative_tolerance * scale), axis=-1)


def _host_arrays(bvh: PackedBVH, /) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(bvh.bbox_min, dtype=np.float64),
        np.asarray(bvh.bbox_max, dtype=np.float64),
        np.asarray(bvh.left, dtype=np.int64),
        np.asarray(bvh.right, dtype=np.int64),
        np.asarray(bvh.leaf_id, dtype=np.int64),
        np.asarray(bvh.leaf_items, dtype=np.int64),
        np.asarray(bvh.item_bbox_min, dtype=np.float64),
        np.asarray(bvh.item_bbox_max, dtype=np.float64),
    )


def _host_pair_blocks(
    first: tuple[np.ndarray, ...],
    second: tuple[np.ndarray, ...],
    test: Callable[..., np.ndarray],
    /,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    first_min, first_max, first_left, first_right, first_leaf, first_items = first[:6]
    second_min, second_max, second_left, second_right, second_leaf, second_items = second[
        :6
    ]
    first_item_min, first_item_max = first[6:]
    second_item_min, second_item_max = second[6:]
    first_size = np.sum(first_max - first_min, axis=-1)
    second_size = np.sum(second_max - second_min, axis=-1)
    leaf_pair_block = max(
        1, _HOST_ITEM_PAIR_BLOCK // (first_items.shape[1] * second_items.shape[1])
    )
    root = np.zeros((1,), dtype=np.int64)
    pending = (
        [(root, root)]
        if test(first_min[:1], first_max[:1], second_min[:1], second_max[:1])[0]
        else []
    )
    # Last-in first-out blocks keep the pending working set proportional to the
    # tree depth times the block size.
    while pending:
        first_nodes, second_nodes = pending.pop()
        if first_nodes.shape[0] > _HOST_NODE_PAIR_BLOCK:
            pending.append(
                (
                    first_nodes[_HOST_NODE_PAIR_BLOCK:],
                    second_nodes[_HOST_NODE_PAIR_BLOCK:],
                )
            )
            first_nodes = first_nodes[:_HOST_NODE_PAIR_BLOCK]
            second_nodes = second_nodes[:_HOST_NODE_PAIR_BLOCK]
        first_is_leaf = first_leaf[first_nodes] >= 0
        second_is_leaf = second_leaf[second_nodes] >= 0
        both_leaf = first_is_leaf & second_is_leaf
        leaf_first = first_leaf[first_nodes[both_leaf]]
        leaf_second = second_leaf[second_nodes[both_leaf]]
        for offset in range(0, leaf_first.shape[0], leaf_pair_block):
            items_a = first_items[leaf_first[offset : offset + leaf_pair_block]]
            items_b = second_items[leaf_second[offset : offset + leaf_pair_block]]
            shape = (items_a.shape[0], items_a.shape[1], items_b.shape[1])
            items_a = np.broadcast_to(items_a[:, :, None], shape).reshape((-1,))
            items_b = np.broadcast_to(items_b[:, None, :], shape).reshape((-1,))
            present = (items_a >= 0) & (items_b >= 0)
            items_a = items_a[present]
            items_b = items_b[present]
            hit = test(
                first_item_min[items_a],
                first_item_max[items_a],
                second_item_min[items_b],
                second_item_max[items_b],
            )
            if np.any(hit):
                yield items_a[hit], items_b[hit]
        descend_first = ~first_is_leaf & (
            second_is_leaf | (first_size[first_nodes] >= second_size[second_nodes])
        )
        descend_second = ~both_leaf & ~descend_first
        first_parents = first_nodes[descend_first]
        second_parents = second_nodes[descend_second]
        child_first = np.concatenate(
            (
                np.stack(
                    (first_left[first_parents], first_right[first_parents]), axis=1
                ).reshape((-1,)),
                np.repeat(first_nodes[descend_second], 2),
            )
        )
        child_second = np.concatenate(
            (
                np.repeat(second_nodes[descend_first], 2),
                np.stack(
                    (second_left[second_parents], second_right[second_parents]), axis=1
                ).reshape((-1,)),
            )
        )
        keep = test(
            first_min[child_first],
            first_max[child_first],
            second_min[child_second],
            second_max[child_second],
        )
        if np.any(keep):
            pending.append((child_first[keep], child_second[keep]))


def _validate_host_pair_arguments(
    first: PackedBVH,
    second: PackedBVH,
    include_touching: bool,
    absolute_tolerance: float,
    relative_tolerance: float,
    /,
) -> Callable[..., np.ndarray]:
    if not isinstance(first, PackedBVH) or not isinstance(second, PackedBVH):
        raise TypeError("first and second must be PackedBVH instances.")
    if first.dimension != second.dimension:
        raise ValueError("first and second must share their spatial dimension.")
    if not isinstance(include_touching, bool):
        raise TypeError("include_touching must be a bool.")
    tolerances = np.asarray((absolute_tolerance, relative_tolerance), dtype=np.float64)
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0.0):
        raise ValueError("tolerances must be finite and non-negative.")
    if not include_touching and np.any(tolerances > 0.0):
        raise ValueError("tolerances apply only when include_touching is True.")
    atol, rtol = (float(value) for value in tolerances)

    def test(first_min: Any, first_max: Any, second_min: Any, second_max: Any) -> Any:
        return _host_boxes_overlap(
            first_min, first_max, second_min, second_max, include_touching, atol, rtol
        )

    return test


def bvh_overlap_pair_blocks(
    first: PackedBVH,
    second: PackedBVH,
    /,
    *,
    include_touching: bool = False,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Enumerate exact overlapping item pairs in bounded host blocks.

    Blocks arrive in deterministic traversal order and together contain every
    overlapping pair exactly once; consumers may stop early to enforce resource
    limits.  See `bvh_overlap_pairs_host` for the overlap predicate.
    """
    test = _validate_host_pair_arguments(
        first, second, include_touching, absolute_tolerance, relative_tolerance
    )
    return _host_pair_blocks(_host_arrays(first), _host_arrays(second), test)


def bvh_overlap_pairs_host(
    first: PackedBVH,
    second: PackedBVH,
    /,
    *,
    include_touching: bool = False,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return every overlapping item pair, sorted by (first, second) item index.

    Items `i` and `j` overlap when `extent = min(max_i, max_j) - max(min_i, min_j)`
    is positive on every axis or, with `include_touching`, when
    `extent >= -(absolute_tolerance + relative_tolerance * scale)` with
    `scale = max(1, |coordinates of both boxes on that axis|)`.  Tolerances must
    be zero otherwise.  Evaluation uses float64 copies of the stored bounds, so
    float64 builds are exact with respect to their input boxes.  A self query
    returns every ordered pair, including `(i, i)`.
    """
    blocks = list(
        bvh_overlap_pair_blocks(
            first,
            second,
            include_touching=include_touching,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    )
    if not blocks:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy()
    first_items = np.concatenate([block[0] for block in blocks])
    second_items = np.concatenate([block[1] for block in blocks])
    order = np.lexsort((second_items, first_items))
    return first_items[order], second_items[order]


__all__ = [
    "BVHBuildKind",
    "BVHBuildPolicy",
    "BVHNearestResult",
    "BVHPairResult",
    "PackedBVH",
    "aabb_dist2",
    "beam_select_leaf_items",
    "beam_select_nodes",
    "bvh_hierarchical_sum",
    "bvh_nearest_items",
    "bvh_overlap_pair_blocks",
    "bvh_overlap_pairs",
    "bvh_overlap_pairs_host",
    "point_select_leaf_items",
    "prepare_bvh",
    "ray_select_leaf_items",
    "reduce_packed_bvh_nodes",
    "refit_packed_bvh_bounds",
]
