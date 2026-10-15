#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Capacity-fixed device MEDIAN sources and exact bounded point queries."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax._bvh import (
    _build_median_bvh,
    _map_bounded_queries,
    _MedianBVHLayout,
    _nearest_items_one,
    PackedBVH,
)
from phydrax._fingerprint import canonical_fingerprint
from phydrax._jax_config import x64_enabled
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState


# Per-chunk working-slot budget for default target chunking.
_TARGET_SLOT_BUDGET = 1 << 18


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")


def _points(points: ArrayLike, capacity: int, dimension: int) -> Array:
    values = jnp.asarray(points)
    if values.shape != (capacity, dimension):
        raise ValueError(f"points must have shape {(capacity, dimension)}.")
    if values.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("points must have float32 or float64 dtype.")
    return values


def _mask(mask: ArrayLike | None, capacity: int) -> Array:
    if mask is None:
        return jnp.ones((capacity,), dtype=jnp.bool_)
    values = jnp.asarray(mask, dtype=jnp.bool_)
    if values.shape != (capacity,):
        raise ValueError("mask must match capacity.")
    return values


def _ids(ids: ArrayLike | None, capacity: int) -> Array:
    if ids is None:
        return jnp.arange(capacity, dtype=jnp.int64)
    values = jnp.asarray(ids)
    if values.shape != (capacity,):
        raise ValueError("stable_ids must match capacity.")
    if not jnp.issubdtype(values.dtype, jnp.integer):
        raise TypeError("stable_ids must have integer dtype.")
    return values


def _geometry_valid(points: Array) -> Array:
    # |s-t| <= 2M, sum_d (s-t)^2 <= max/4. The margin also
    # admits the periodic half-length addition without overflow. This is an
    # arithmetic admission bound, NOT a pruning epsilon.
    maximum = jnp.asarray(
        np.sqrt(np.finfo(points.dtype).max / (16 * points.shape[1])),
        dtype=points.dtype,
    )
    return jnp.all(jnp.isfinite(points) & (jnp.abs(points) <= maximum), axis=1)


def _ordered_squared_norm(relative: Array) -> Array:
    """One common monotone arithmetic schedule for candidates and bounds.

    Subtraction, squaring and each nonnegative addition are separated by
    optimization barriers: no FMA or reassociation can change the schedule.
    Correctly rounded subtraction is monotone; on a non-periodic axis the
    nearest box endpoint has no larger absolute rounded displacement than any
    contained point. Squaring and the identical ordered positive additions
    preserve that inequality (including underflow). Periodic axes contribute
    zero to bounds. Thus the computed bound <= the computed candidate distance,
    without an epsilon, even at a one-ulp tied boundary.
    """
    total = jnp.zeros(relative.shape[:-1], dtype=relative.dtype)
    for axis in range(relative.shape[-1]):
        component = jax.lax.optimization_barrier(relative[..., axis])
        square = jax.lax.optimization_barrier(component * component)
        total = jax.lax.optimization_barrier(total + square)
    return total


@final
class BVHSourcePlan(StrictModule):
    """Static MEDIAN topology and point-domain metadata, reusable under JIT."""

    dimension: int = eqx.field(static=True)
    point_capacity: int = eqx.field(static=True)
    leaf_size: int = eqx.field(static=True)
    periodic_lengths: tuple[float | None, ...] | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    _layout: _MedianBVHLayout = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        point_capacity: int,
        leaf_size: int = 8,
        *,
        periodic_lengths: tuple[float | None, ...] | None = None,
    ) -> None:
        _positive_integer("dimension", dimension)
        _positive_integer("point_capacity", point_capacity)
        _positive_integer("leaf_size", leaf_size)
        if periodic_lengths is not None:
            if len(periodic_lengths) != dimension:
                raise ValueError("periodic_lengths must match dimension.")
            if any(
                length is not None and (not isfinite(length) or length <= 0)
                for length in periodic_lengths
            ):
                raise ValueError("periodic lengths must be finite and positive.")
            periodic_lengths = tuple(
                None if length is None else float(length) for length in periodic_lengths
            )
        self.dimension = dimension
        self.point_capacity = point_capacity
        self.leaf_size = leaf_size
        self.periodic_lengths = periodic_lengths
        self._layout = _MedianBVHLayout(point_capacity, leaf_size)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bvh-source-plan",
                "dimension": dimension,
                "point_capacity": point_capacity,
                "leaf_size": leaf_size,
                "periodic_lengths": periodic_lengths,
            }
        )

    def prepare(
        self,
        points: ArrayLike,
        *,
        active_mask: ArrayLike | None = None,
        stable_ids: ArrayLike | None = None,
    ) -> PreparedBVHSources:
        values = _points(points, self.point_capacity, self.dimension)
        if self.periodic_lengths is not None:
            maximum = np.sqrt(np.finfo(values.dtype).max / (16 * self.dimension))
            if any(
                length is not None
                and (length > maximum or np.asarray(length, dtype=values.dtype) == 0)
                for length in self.periodic_lengths
            ):
                raise ValueError(
                    "periodic length is not representable in geometry precision."
                )
        active = _mask(active_mask, self.point_capacity)
        ids = _ids(stable_ids, self.point_capacity)
        geometry = jax.lax.stop_gradient(values)
        finite = jnp.all(jnp.isfinite(geometry), axis=1)
        admissible = _geometry_valid(geometry)
        admitted = active & admissible
        tree = _build_median_bvh(self._layout, geometry, geometry, active_mask=admitted)
        # Occupancy and ID minima are separate: the maximum ID remains valid.
        items = tree.leaf_items
        present = items >= 0
        empty = jnp.ones((tree.node_count,), dtype=jnp.bool_)
        empty = empty.at[tree.leaf_node].set(~jnp.any(present, axis=1))
        sentinel = jnp.asarray(jnp.iinfo(ids.dtype).max, dtype=ids.dtype)
        minima = jnp.full((tree.node_count,), sentinel, dtype=ids.dtype)
        minima = minima.at[tree.leaf_node].set(
            jnp.min(jnp.where(present, ids[jnp.maximum(items, 0)], sentinel), axis=1)
        )
        for level in range(tree.max_depth - 1, -1, -1):
            start, stop = tree.level_offsets[level : level + 2]
            left, right = tree.left[start:stop], tree.right[start:stop]
            internal = left >= 0
            left, right = jnp.maximum(left, 0), jnp.maximum(right, 0)
            empty = empty.at[start:stop].set(
                jnp.where(internal, empty[left] & empty[right], empty[start:stop])
            )
            minima = minima.at[start:stop].set(
                jnp.where(
                    internal, jnp.minimum(minima[left], minima[right]), minima[start:stop]
                )
            )
        sorted_active, sorted_ids = jax.lax.sort((~active, ids), num_keys=2)
        unique = jnp.all(
            sorted_active[1:] | sorted_active[:-1] | (sorted_ids[1:] != sorted_ids[:-1])
        )
        invalid = jnp.sum(active & ~admissible, dtype=jnp.int32)
        return PreparedBVHSources(
            source_plan=self,
            points=values,
            active=active,
            stable_ids=ids,
            stable_ids_declared=stable_ids is not None,
            tree=tree,
            node_empty=empty,
            node_min_id=minima,
            finite=finite,
            invalid_points=invalid,
            stable_ids_unique=unique,
            build_successful=(invalid == 0) & unique,
        )


@final
class PreparedBVHSources(NonTrainableState, StrictModule):
    """Logical live geometry with private nondifferentiable masked tree state."""

    source_plan: BVHSourcePlan = eqx.field(static=True)
    points: Array
    active: Array
    stable_ids: Array
    stable_ids_declared: bool = eqx.field(static=True)
    tree: PackedBVH
    node_empty: Array
    node_min_id: Array
    finite: Array
    invalid_points: Array
    stable_ids_unique: Array
    build_successful: Array

    @property
    def point_capacity(self) -> int:
        return self.source_plan.point_capacity


class BVHNeighborQueryStatus(IntEnum):
    """Inactive targets precede invalid targets, sources, and traversal failures."""

    COMPLETE = 0
    INACTIVE_TARGET = 1
    INVALID_TARGET = 2
    INVALID_SOURCES = 3
    TRAVERSAL_FAILURE = 4


@final
class BVHNeighborQueryEvidence(NonTrainableState, StrictModule):
    """Completion, validity, and traversal-work evidence for one BVH query.

    ``node_visits`` counts visited hierarchy nodes; ``failed_rows`` counts rows
    whose bounded traversal could not certify a complete answer.
    """

    successful: Array
    complete: Array
    finite: Array
    sources_valid: Array
    invalid_sources: Array
    invalid_targets: Array
    failed_rows: Array
    node_visits: Array


@final
class BVHNeighborQueryResult(NonTrainableState, StrictModule):
    """Logical source indices per target; rows that are not complete are empty."""

    source_indices: Array
    valid: Array
    counts: Array
    status: Array
    evidence: BVHNeighborQueryEvidence


@final
class BVHNeighborQueryPlan(StrictModule):
    """Exact logical point neighbors ordered by (distance, full stable ID).

    Periodic pruning uses zero on periodic axes and is exact but weak. Selection
    is nondifferentiable; consumers recompute geometry from live sources.points.
    """

    source_plan: BVHSourcePlan = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    maximum_neighbors: int = eqx.field(static=True)
    target_chunk_size: int = eqx.field(static=True)

    def __init__(
        self,
        source_plan: BVHSourcePlan,
        target_capacity: int,
        maximum_neighbors: int,
        *,
        target_chunk_size: int | None = None,
    ) -> None:
        if not isinstance(source_plan, BVHSourcePlan):
            raise TypeError("source_plan must be a BVHSourcePlan.")
        _positive_integer("target_capacity", target_capacity)
        _positive_integer("maximum_neighbors", maximum_neighbors)
        if target_chunk_size is not None:
            _positive_integer("target_chunk_size", target_chunk_size)
        self.source_plan = source_plan
        self.target_capacity = target_capacity
        self.maximum_neighbors = maximum_neighbors
        # Each traversal lane holds its stack, k (distance, ID) slots, and one
        # leaf of candidates. Divergent lanes run until the slowest finishes, so
        # the default chunk fills the same slot budget as the Morton candidate
        # buffers instead of serializing many short traversal loops.
        lane_slots = (
            source_plan._layout.max_depth
            + 2
            + 2 * maximum_neighbors
            + source_plan.leaf_size
        )
        default_chunk = max(1, _TARGET_SLOT_BUDGET // lane_slots)
        self.target_chunk_size = min(target_capacity, target_chunk_size or default_chunk)

    @property
    def source_capacity(self) -> int:
        return self.source_plan.point_capacity

    def query(
        self,
        sources: PreparedBVHSources,
        target_points: ArrayLike,
        *,
        target_mask: ArrayLike | None = None,
        target_stable_ids: ArrayLike | None = None,
        exclude_self: bool = False,
        radius: float | None = None,
        target_radii: ArrayLike | None = None,
    ) -> BVHNeighborQueryResult:
        if not isinstance(sources, PreparedBVHSources):
            raise TypeError("sources must be PreparedBVHSources.")
        if sources.source_plan.plan_id != self.source_plan.plan_id:
            raise ValueError("sources must match the query source plan.")
        targets = _points(target_points, self.target_capacity, self.source_plan.dimension)
        if targets.dtype != sources.points.dtype:
            raise TypeError("target and prepared source precision must match.")
        if radius is not None and (not isfinite(radius) or radius <= 0):
            raise ValueError("radius must be finite and positive when supplied.")
        if exclude_self and target_stable_ids is None:
            if self.source_capacity != self.target_capacity:
                raise ValueError(
                    "exclude_self with unequal capacities requires target_stable_ids."
                )
        target_ids = _ids(target_stable_ids, self.target_capacity)
        active = _mask(target_mask, self.target_capacity)
        radii = jnp.full((self.target_capacity,), jnp.inf, dtype=targets.dtype)
        if target_radii is not None:
            radii = jnp.asarray(target_radii, dtype=targets.dtype)
            if radii.shape != (self.target_capacity,):
                raise ValueError("target_radii must hold one radius per target slot.")
        if radius is not None:
            radii = jnp.minimum(radii, jnp.asarray(radius, dtype=targets.dtype))
        return _compiled_bvh_query(
            self, sources, targets, active, target_ids, radii, exclude_self
        )


def _bvh_query(
    plan: BVHNeighborQueryPlan,
    sources: PreparedBVHSources,
    targets: Array,
    active: Array,
    target_ids: Array,
    radii: Array,
    exclude_self: bool,
) -> BVHNeighborQueryResult:
    targets = jax.lax.stop_gradient(targets)
    points = jax.lax.stop_gradient(sources.points)
    radii = jax.lax.stop_gradient(radii)
    lengths = plan.source_plan.periodic_lengths
    valid_target = _geometry_valid(targets) & (radii >= 0) & ~jnp.isnan(radii)

    def node_distance(point: Array, node: Array) -> Array:
        lower, upper = sources.tree.bbox_min[node], sources.tree.bbox_max[node]
        relative = jnp.where(
            point < lower, lower - point, jnp.where(point > upper, upper - point, 0)
        )
        if lengths is not None:
            periodic = jnp.asarray(
                tuple(length is not None for length in lengths), dtype=jnp.bool_
            )
            relative = jnp.where(periodic, jnp.asarray(0, dtype=point.dtype), relative)
        return _ordered_squared_norm(relative)

    def query(
        point: Array, target_id: Array, radius: Array, runnable: Array
    ) -> tuple[Array, Array, Array, Array]:

        def distance(value: Array, items: Array) -> Array:
            # Same target-minus-source minimum image as the Morton and graph routes.
            relative = jax.lax.optimization_barrier(value - points[items])
            if lengths is not None:
                periods = jnp.asarray(
                    tuple(1.0 if length is None else length for length in lengths),
                    dtype=value.dtype,
                )
                periodic = jnp.asarray(
                    tuple(length is not None for length in lengths), dtype=jnp.bool_
                )
                wrapped = relative - jnp.round(relative / periods) * periods
                relative = jnp.where(periodic, wrapped, relative)
            return _ordered_squared_norm(relative)

        def admit(items: Array) -> Array:
            # Admission is evaluated only on visited leaf items, never as a
            # per-target array over every source.
            admitted = sources.active[items]
            if not exclude_self:
                return admitted
            source_ids = sources.stable_ids[items]
            source_signed = jnp.issubdtype(source_ids.dtype, jnp.signedinteger)
            target_signed = jnp.issubdtype(target_id.dtype, jnp.signedinteger)
            if source_signed == target_signed:
                same_id = source_ids == target_id
            else:
                # Never promote signed/unsigned full-width IDs to floating point.
                unsigned = jnp.uint64 if x64_enabled() else jnp.uint32
                same_id = source_ids.astype(unsigned) == target_id.astype(unsigned)
                same_id &= source_ids >= 0 if source_signed else target_id >= 0
            return admitted & ~same_id

        return _nearest_items_one(
            sources.tree,
            point,
            plan.maximum_neighbors,
            distance,
            item_filter=admit,
            stable_ids=sources.stable_ids,
            node_empty=sources.node_empty,
            node_min_id=sources.node_min_id,
            node_distance=node_distance,
            maximum_distance=radius * radius,
            stop_selection_gradient=True,
            runnable=runnable,
        )

    indices, _, complete, visits = _map_bounded_queries(
        query,
        (targets, target_ids, radii, active & valid_target & sources.build_successful),
        plan.target_chunk_size,
    )
    status = jnp.where(
        complete,
        jnp.int32(BVHNeighborQueryStatus.COMPLETE),
        jnp.int32(BVHNeighborQueryStatus.TRAVERSAL_FAILURE),
    )
    status = jnp.where(
        sources.build_successful,
        status,
        jnp.int32(BVHNeighborQueryStatus.INVALID_SOURCES),
    )
    status = jnp.where(
        valid_target, status, jnp.int32(BVHNeighborQueryStatus.INVALID_TARGET)
    )
    status = jnp.where(active, status, jnp.int32(BVHNeighborQueryStatus.INACTIVE_TARGET))
    valid = (indices >= 0) & (status[:, None] == BVHNeighborQueryStatus.COMPLETE)
    indices = jnp.where(valid, indices, -1)
    invalid_targets = jnp.sum(active & ~valid_target, dtype=jnp.int32)
    failed_rows = jnp.sum(
        status == BVHNeighborQueryStatus.TRAVERSAL_FAILURE, dtype=jnp.int32
    )
    finite = jnp.all(sources.finite | ~sources.active) & jnp.all(
        jnp.all(jnp.isfinite(targets), axis=1) | ~active
    )
    evidence = BVHNeighborQueryEvidence(
        successful=sources.build_successful & (invalid_targets == 0) & (failed_rows == 0),
        complete=failed_rows == 0,
        finite=finite,
        sources_valid=sources.build_successful,
        invalid_sources=sources.invalid_points,
        invalid_targets=invalid_targets,
        failed_rows=failed_rows,
        node_visits=visits,
    )
    return BVHNeighborQueryResult(
        source_indices=indices,
        valid=valid,
        counts=jnp.sum(valid, axis=1, dtype=jnp.int32),
        status=status,
        evidence=evidence,
    )


_compiled_bvh_query = eqx.filter_jit(_bvh_query)

__all__ = [
    "BVHSourcePlan",
    "PreparedBVHSources",
    "BVHNeighborQueryPlan",
    "BVHNeighborQueryResult",
    "BVHNeighborQueryEvidence",
    "BVHNeighborQueryStatus",
]
