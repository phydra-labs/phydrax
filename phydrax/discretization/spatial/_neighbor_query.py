#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.sparse import EdgeRelation

from ._morton import MortonAddressPlan
from ._plane_schedule import MortonPlaneSchedulePlan


SpatialDistanceBackend: TypeAlias = Literal["jax", "pallas"]


class MortonNeighborQueryEvidence(NonTrainableState, StrictModule):
    """Completeness and resource evidence for one exact neighbor query."""

    successful: jax.Array
    topology_successful: jax.Array
    finite: jax.Array
    invalid_targets: jax.Array
    required_nodes: jax.Array
    node_capacity: jax.Array
    required_leaf_interactions: jax.Array
    maximum_leaf_interactions: jax.Array
    required_candidates: jax.Array
    candidate_capacity: jax.Array
    row_complete: jax.Array
    complete: jax.Array


class MortonNeighborQueryResult(NonTrainableState, StrictModule):
    """Fixed-width exact source indices in target-logical order."""

    source_indices: jax.Array
    valid: jax.Array
    counts: jax.Array
    evidence: MortonNeighborQueryEvidence


class MortonRadiusRelationEvidence(NonTrainableState, StrictModule):
    """Completeness and resource evidence for one exact radius relation."""

    successful: jax.Array
    topology_successful: jax.Array
    finite: jax.Array
    invalid_sources: jax.Array
    invalid_targets: jax.Array
    required_nodes: jax.Array
    node_capacity: jax.Array
    maximum_leaf_occupancy: jax.Array
    required_leaf_interactions: jax.Array
    required_pairs: jax.Array
    pair_capacity: jax.Array
    pair_overflow: jax.Array


class MortonRadiusRelationResult(NonTrainableState, StrictModule):
    """Exact fixed-capacity source-to-target radius relation."""

    relation: EdgeRelation
    evidence: MortonRadiusRelationEvidence
    storage_to_logical: jax.Array
    logical_to_storage: jax.Array
    logical_leaf_slots: jax.Array
    leaf_counts: jax.Array
    leaf_offsets: jax.Array


def _minimum_image(
    relative: jax.Array,
    address_plan: MortonAddressPlan,
) -> jax.Array:
    lengths = jnp.asarray(address_plan.upper, dtype=relative.dtype) - jnp.asarray(
        address_plan.lower, dtype=relative.dtype
    )
    periodic = jnp.asarray(address_plan.periodic_axes, dtype=bool)
    wrapped = relative - jnp.round(relative / lengths) * lengths
    return jnp.where(periodic, wrapped, relative)


def _squared_norm(
    relative: jax.Array,
    *,
    backend: SpatialDistanceBackend,
    pallas_interpret: bool,
) -> jax.Array:
    if backend == "jax":
        return jnp.sum(relative * relative, axis=-1)
    from phydrax.backends.spatial import spatial_squared_norm

    return spatial_squared_norm(
        relative,
        backend="pallas",
        pallas_interpret=pallas_interpret,
    )


def _point_box_distance_bounds(
    points: jax.Array,
    centers: jax.Array,
    half_widths: jax.Array,
    node_active: jax.Array,
    address_plan: MortonAddressPlan,
) -> tuple[jax.Array, jax.Array]:
    relative = points[:, None, :] - centers[None, :, :]
    lengths = jnp.asarray(address_plan.upper, dtype=relative.dtype) - jnp.asarray(
        address_plan.lower, dtype=relative.dtype
    )
    periodic = jnp.asarray(address_plan.periodic_axes, dtype=bool)
    wrapped = relative - jnp.round(relative / lengths) * lengths
    center_distance = jnp.where(periodic, jnp.abs(wrapped), jnp.abs(relative))
    lower_axis = jnp.maximum(center_distance - half_widths[None, :, :], 0)
    ordinary_upper = center_distance + half_widths[None, :, :]
    upper_axis = jnp.where(
        periodic,
        jnp.minimum(ordinary_upper, 0.5 * lengths),
        ordinary_upper,
    )
    lower_squared = jnp.sum(lower_axis * lower_axis, axis=-1)
    upper_squared = jnp.sum(upper_axis * upper_axis, axis=-1)
    infinity = jnp.asarray(jnp.inf, dtype=lower_squared.dtype)
    return (
        jnp.where(node_active[None, :], lower_squared, infinity),
        jnp.where(node_active[None, :], upper_squared, infinity),
    )


class MortonNeighborQueryPlan(StrictModule):
    """Exact low-dimensional neighbor query over compact Morton leaves."""

    address_plan: MortonAddressPlan
    schedule_plan: MortonPlaneSchedulePlan
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    maximum_neighbors: int = eqx.field(static=True)
    maximum_candidates: int = eqx.field(static=True)
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
        maximum_nodes: int | None = None,
        maximum_leaf_occupancy: int = 32,
        coarsening_factor: int = 8,
        target_top_nodes: int = 1024,
        distance_backend: SpatialDistanceBackend = "jax",
        pallas_interpret: bool = False,
    ) -> None:
        sources = int(source_capacity)
        targets = int(target_capacity)
        neighbors = int(maximum_neighbors)
        candidates = sources if maximum_candidates is None else int(maximum_candidates)
        if sources < 1 or targets < 1:
            raise ValueError("source_capacity and target_capacity must be positive.")
        if neighbors < 1 or neighbors > sources:
            raise ValueError("maximum_neighbors must lie in [1, source_capacity].")
        if candidates < neighbors or candidates > sources:
            raise ValueError(
                "maximum_candidates must lie in [maximum_neighbors, source_capacity]."
            )
        if distance_backend not in ("jax", "pallas"):
            raise ValueError("distance_backend must be 'jax' or 'pallas'.")
        schedule = MortonPlaneSchedulePlan(
            address_plan,
            sources,
            node_capacity=maximum_nodes,
            maximum_leaf_occupancy=maximum_leaf_occupancy,
            coarsening_factor=coarsening_factor,
            target_top_nodes=target_top_nodes,
        )
        if schedule.node_capacity < schedule.plane_capacities[0]:
            raise ValueError("maximum_nodes must retain every possible leaf node.")

        object.__setattr__(self, "address_plan", address_plan)
        object.__setattr__(self, "schedule_plan", schedule)
        object.__setattr__(self, "source_capacity", sources)
        object.__setattr__(self, "target_capacity", targets)
        object.__setattr__(self, "maximum_neighbors", neighbors)
        object.__setattr__(self, "maximum_candidates", candidates)
        object.__setattr__(self, "distance_backend", distance_backend)
        object.__setattr__(self, "pallas_interpret", bool(pallas_interpret))
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "morton-neighbor-query-plan",
                    "address_plan_id": address_plan.plan_id,
                    "source_capacity": sources,
                    "target_capacity": targets,
                    "maximum_neighbors": neighbors,
                    "maximum_candidates": candidates,
                    "schedule_plan_id": schedule.plan_id,
                    "distance_backend": distance_backend,
                    "pallas_interpret": bool(pallas_interpret),
                }
            ),
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
        sources = jnp.asarray(source_points)
        targets = jnp.asarray(target_points)
        source_shape = (self.source_capacity, self.address_plan.dimension)
        target_shape = (self.target_capacity, self.address_plan.dimension)
        if sources.shape != source_shape:
            raise ValueError(f"source_points must have shape {source_shape}.")
        if targets.shape != target_shape:
            raise ValueError(f"target_points must have shape {target_shape}.")
        if radius is not None and float(radius) <= 0:
            raise ValueError("radius must be positive when supplied.")

        if target_mask is None:
            target_active = jnp.ones((self.target_capacity,), dtype=bool)
        else:
            target_active = jnp.asarray(target_mask, dtype=bool)
            if target_active.shape != (self.target_capacity,):
                raise ValueError("target_mask must match target_capacity.")

        if target_stable_ids is None:
            target_ids = jnp.arange(self.target_capacity, dtype=jnp.int64)
        else:
            target_ids = jnp.asarray(target_stable_ids)
            if target_ids.shape != (self.target_capacity,):
                raise ValueError("target_stable_ids must match target_capacity.")
            if not jnp.issubdtype(target_ids.dtype, jnp.integer):
                raise TypeError("target_stable_ids must have integer dtype.")
        if (
            exclude_self
            and target_stable_ids is None
            and (self.source_capacity != self.target_capacity)
        ):
            raise ValueError(
                "exclude_self with unequal capacities requires target_stable_ids."
            )

        schedule = self.schedule_plan.build(
            sources,
            active_mask=source_mask,
            stable_ids=source_stable_ids,
        )
        target_encoding = self.address_plan.encode(targets)
        target_valid = target_active & target_encoding.in_domain
        invalid_targets = jnp.sum(
            target_active & ~target_encoding.in_domain, dtype=jnp.int32
        )

        leaf_capacity = self.schedule_plan.plane_capacities[0]
        leaf_active = schedule.node_active[:leaf_capacity]
        leaf_centers = schedule.node_centers[:leaf_capacity]
        leaf_half_widths = schedule.node_half_widths[:leaf_capacity]
        leaf_counts = schedule.node_item_counts[:leaf_capacity]
        leaf_starts = schedule.node_item_starts[:leaf_capacity]
        lower_squared, upper_squared = _point_box_distance_bounds(
            target_encoding.coordinates,
            leaf_centers,
            leaf_half_widths,
            leaf_active,
            self.address_plan,
        )

        upper_order = jnp.argsort(upper_squared, axis=1, stable=True)
        ordered_upper = jnp.take_along_axis(upper_squared, upper_order, axis=1)
        ordered_counts = jnp.take_along_axis(
            jnp.broadcast_to(leaf_counts, upper_order.shape), upper_order, axis=1
        )
        self_extra = int(bool(exclude_self))
        requested = jnp.minimum(
            self.maximum_neighbors + self_extra,
            schedule.evidence.active_points,
        )
        cumulative = jnp.cumsum(ordered_counts, axis=1)
        reaches_requested = cumulative >= requested
        radius_slot = jnp.argmax(reaches_requested, axis=1)
        search_radius_squared = jnp.take_along_axis(
            ordered_upper, radius_slot[:, None], axis=1
        )[:, 0]
        search_radius_squared = jnp.where(requested > 0, search_radius_squared, 0)

        leaf_candidate = (
            leaf_active[None, :]
            & target_valid[:, None]
            & (lower_squared <= search_radius_squared[:, None])
        )
        if radius is not None:
            leaf_candidate = leaf_candidate & (lower_squared <= float(radius) ** 2)
        required_leaf_interactions = jnp.sum(leaf_candidate, dtype=jnp.int32)
        maximum_leaf_interactions = jnp.max(
            jnp.sum(leaf_candidate, axis=1, dtype=jnp.int32), initial=0
        )

        width = self.schedule_plan.maximum_leaf_occupancy
        point_offsets = jnp.arange(width, dtype=jnp.int32)
        candidate_storage = leaf_starts[:, None] + point_offsets[None, :]
        candidate_storage = jnp.minimum(candidate_storage, self.source_capacity - 1)
        leaf_point_valid = point_offsets[None, :] < leaf_counts[:, None]
        expanded_valid = leaf_candidate[:, :, None] & leaf_point_valid[None, :, :]
        flattened_valid = expanded_valid.reshape((self.target_capacity, -1))
        flattened_storage = jnp.broadcast_to(
            candidate_storage.reshape((1, -1)), flattened_valid.shape
        )
        required_candidates_by_row = jnp.sum(flattened_valid, axis=1, dtype=jnp.int32)
        row_complete = (~target_active) | (
            required_candidates_by_row <= self.maximum_candidates
        )

        def select_positions(mask: jax.Array) -> jax.Array:
            return jnp.nonzero(
                mask,
                size=self.maximum_candidates,
                fill_value=0,
            )[0].astype(jnp.int32)

        selected_positions = jax.vmap(select_positions)(flattened_valid)
        candidate_storage = jnp.take_along_axis(
            flattened_storage, selected_positions, axis=1
        )
        candidate_rank = jnp.arange(self.maximum_candidates, dtype=jnp.int32)
        candidate_valid = candidate_rank[None, :] < required_candidates_by_row[:, None]
        candidate_valid = (
            candidate_valid
            & row_complete[:, None]
            & target_valid[:, None]
            & schedule.evidence.successful
        )

        safe_storage = jnp.minimum(candidate_storage, self.source_capacity - 1)
        source_logical = schedule.point_order.storage_to_logical[safe_storage]
        source_coordinates = schedule.point_order.encoding.coordinates[source_logical]
        relative = target_encoding.coordinates[:, None, :] - source_coordinates
        relative = _minimum_image(relative, self.address_plan)
        distance_squared = _squared_norm(
            relative,
            backend=self.distance_backend,
            pallas_interpret=self.pallas_interpret,
        )
        source_ids = schedule.point_order.sorted_stable_ids[safe_storage]
        if exclude_self:
            candidate_valid = candidate_valid & (source_ids != target_ids[:, None])
        if radius is not None:
            candidate_valid = candidate_valid & (distance_squared <= float(radius) ** 2)

        maximum_id = jnp.asarray(jnp.iinfo(source_ids.dtype).max, dtype=source_ids.dtype)
        sortable_distance = jnp.where(candidate_valid, distance_squared, jnp.inf)
        sortable_id = jnp.where(candidate_valid, source_ids, maximum_id)
        candidate_order = jnp.lexsort(
            (sortable_id, sortable_distance),
            axis=1,
        )
        selected = candidate_order[:, : self.maximum_neighbors]
        source_indices = jnp.take_along_axis(source_logical, selected, axis=1)
        valid = jnp.take_along_axis(candidate_valid, selected, axis=1)
        source_indices = jnp.where(valid, source_indices, 0).astype(jnp.int32)
        complete = (
            schedule.evidence.successful & (invalid_targets == 0) & jnp.all(row_complete)
        )
        evidence = MortonNeighborQueryEvidence(
            successful=complete,
            topology_successful=schedule.evidence.successful,
            finite=jnp.all(target_encoding.finite | ~target_active),
            invalid_targets=invalid_targets,
            required_nodes=schedule.evidence.required_nodes,
            node_capacity=schedule.evidence.node_capacity,
            required_leaf_interactions=required_leaf_interactions,
            maximum_leaf_interactions=maximum_leaf_interactions,
            required_candidates=jnp.max(required_candidates_by_row, initial=0),
            candidate_capacity=jnp.asarray(self.maximum_candidates, dtype=jnp.int32),
            row_complete=row_complete,
            complete=complete,
        )
        return MortonNeighborQueryResult(
            source_indices=source_indices,
            valid=valid,
            counts=jnp.sum(valid, axis=1, dtype=jnp.int32),
            evidence=evidence,
        )


class MortonRadiusRelationPlan(StrictModule):
    """Exact low-dimensional radius relation over compact Morton leaves."""

    address_plan: MortonAddressPlan
    schedule_plan: MortonPlaneSchedulePlan
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
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
        maximum_nodes: int | None = None,
        maximum_leaf_occupancy: int = 32,
        coarsening_factor: int = 8,
        target_top_nodes: int = 1024,
        distance_backend: SpatialDistanceBackend = "jax",
        pallas_interpret: bool = False,
    ) -> None:
        sources = int(source_capacity)
        targets = int(target_capacity)
        pairs = int(maximum_pairs)
        if sources < 1 or targets < 1:
            raise ValueError("source_capacity and target_capacity must be positive.")
        if pairs < 0:
            raise ValueError("maximum_pairs must be nonnegative.")
        if distance_backend not in ("jax", "pallas"):
            raise ValueError("distance_backend must be 'jax' or 'pallas'.")
        schedule = MortonPlaneSchedulePlan(
            address_plan,
            sources,
            node_capacity=maximum_nodes,
            maximum_leaf_occupancy=maximum_leaf_occupancy,
            coarsening_factor=coarsening_factor,
            target_top_nodes=target_top_nodes,
        )
        if schedule.node_capacity < schedule.plane_capacities[0]:
            raise ValueError("maximum_nodes must retain every possible leaf node.")
        object.__setattr__(self, "address_plan", address_plan)
        object.__setattr__(self, "schedule_plan", schedule)
        object.__setattr__(self, "source_capacity", sources)
        object.__setattr__(self, "target_capacity", targets)
        object.__setattr__(self, "maximum_pairs", pairs)
        object.__setattr__(self, "inclusive", bool(inclusive))
        object.__setattr__(self, "distance_backend", distance_backend)
        object.__setattr__(self, "pallas_interpret", bool(pallas_interpret))
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "morton-radius-relation-plan",
                    "address_plan_id": address_plan.plan_id,
                    "source_capacity": sources,
                    "target_capacity": targets,
                    "maximum_pairs": pairs,
                    "inclusive": bool(inclusive),
                    "schedule_plan_id": schedule.plan_id,
                    "distance_backend": distance_backend,
                    "pallas_interpret": bool(pallas_interpret),
                }
            ),
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
        sources = jnp.asarray(source_points)
        targets = jnp.asarray(target_points)
        source_shape = (self.source_capacity, self.address_plan.dimension)
        target_shape = (self.target_capacity, self.address_plan.dimension)
        if sources.shape != source_shape:
            raise ValueError(f"source_points must have shape {source_shape}.")
        if targets.shape != target_shape:
            raise ValueError(f"target_points must have shape {target_shape}.")
        radius_value = float(radius)
        if radius_value <= 0:
            raise ValueError("radius must be positive.")
        if pair_once and self.source_capacity != self.target_capacity:
            raise ValueError("pair_once requires equal source and target capacities.")

        if target_mask is None:
            target_active = jnp.ones((self.target_capacity,), dtype=bool)
        else:
            target_active = jnp.asarray(target_mask, dtype=bool)
            if target_active.shape != (self.target_capacity,):
                raise ValueError("target_mask must match target_capacity.")
        if target_stable_ids is None:
            target_ids = jnp.arange(self.target_capacity, dtype=jnp.int64)
        else:
            target_ids = jnp.asarray(target_stable_ids)
            if target_ids.shape != (self.target_capacity,):
                raise ValueError("target_stable_ids must match target_capacity.")
            if not jnp.issubdtype(target_ids.dtype, jnp.integer):
                raise TypeError("target_stable_ids must have integer dtype.")

        schedule = self.schedule_plan.build(
            sources,
            active_mask=source_mask,
            stable_ids=source_stable_ids,
        )
        target_encoding = self.address_plan.encode(targets)
        target_valid = target_active & target_encoding.in_domain
        invalid_targets = jnp.sum(
            target_active & ~target_encoding.in_domain, dtype=jnp.int32
        )
        leaf_capacity = self.schedule_plan.plane_capacities[0]
        leaf_active = schedule.node_active[:leaf_capacity]
        leaf_counts = schedule.node_item_counts[:leaf_capacity]
        leaf_starts = schedule.node_item_starts[:leaf_capacity]
        lower_squared, _ = _point_box_distance_bounds(
            target_encoding.coordinates,
            schedule.node_centers[:leaf_capacity],
            schedule.node_half_widths[:leaf_capacity],
            leaf_active,
            self.address_plan,
        )
        leaf_candidate = (
            target_valid[:, None]
            & leaf_active[None, :]
            & (lower_squared <= radius_value**2)
        )
        required_leaf_interactions = jnp.sum(leaf_candidate, dtype=jnp.int32)

        width = self.schedule_plan.maximum_leaf_occupancy
        point_offsets = jnp.arange(width, dtype=jnp.int32)
        source_storage = leaf_starts[:, None] + point_offsets[None, :]
        source_storage = jnp.minimum(source_storage, self.source_capacity - 1)
        leaf_point_valid = point_offsets[None, :] < leaf_counts[:, None]
        route_candidate = leaf_candidate[:, :, None] & leaf_point_valid[None, :, :]
        source_storage = jnp.broadcast_to(
            source_storage[None, :, :], route_candidate.shape
        )
        source_logical = schedule.point_order.storage_to_logical[source_storage]
        target_logical = jnp.broadcast_to(
            jnp.arange(self.target_capacity, dtype=jnp.int32)[:, None, None],
            route_candidate.shape,
        )
        source_coordinates = schedule.point_order.encoding.coordinates[source_logical]
        target_coordinates = target_encoding.coordinates[target_logical]
        relative = _minimum_image(
            target_coordinates - source_coordinates,
            self.address_plan,
        )
        distance_squared = _squared_norm(
            relative,
            backend=self.distance_backend,
            pallas_interpret=self.pallas_interpret,
        )
        if self.inclusive:
            route_candidate = route_candidate & (distance_squared <= radius_value**2)
        else:
            route_candidate = route_candidate & (distance_squared < radius_value**2)

        source_ids = schedule.point_order.stable_ids[source_logical]
        route_target_ids = target_ids[target_logical]
        if exclude_self:
            route_candidate = route_candidate & (source_ids != route_target_ids)
        if pair_once:
            route_candidate = route_candidate & (source_ids < route_target_ids)
        identity_compatible = (
            jnp.all(schedule.point_order.stable_ids == target_ids)
            if pair_once
            else jnp.asarray(True)
        )

        flat_valid = route_candidate.reshape((-1,))
        flat_source = source_logical.reshape((-1,))
        flat_target = target_logical.reshape((-1,))
        flat_source_ids = source_ids.reshape((-1,))
        flat_target_ids = route_target_ids.reshape((-1,))
        major_ids = flat_source_ids if pair_once else flat_target_ids
        minor_ids = flat_target_ids if pair_once else flat_source_ids
        route_order = jnp.lexsort(
            (
                minor_ids,
                major_ids,
                (~flat_valid).astype(jnp.int32),
            )
        )
        selected = route_order[: self.maximum_pairs]
        required_pairs = jnp.sum(flat_valid, dtype=jnp.int32)
        pair_overflow = required_pairs > self.maximum_pairs
        complete = (
            schedule.evidence.successful
            & (invalid_targets == 0)
            & identity_compatible
            & ~pair_overflow
        )
        route_rank = jnp.arange(self.maximum_pairs, dtype=jnp.int32)
        selected_valid = (
            (route_rank < required_pairs) & complete
            if self.maximum_pairs
            else jnp.zeros((0,), dtype=bool)
        )
        selected_source = flat_source[selected].astype(jnp.int32)
        selected_target = flat_target[selected].astype(jnp.int32)
        relation = EdgeRelation(
            jnp.where(selected_valid, selected_source, 0),
            jnp.where(selected_valid, selected_target, 0),
            source_size=self.source_capacity,
            target_size=self.target_capacity,
            valid=selected_valid,
        )
        evidence = MortonRadiusRelationEvidence(
            successful=complete,
            topology_successful=schedule.evidence.successful,
            finite=(
                jnp.all(target_encoding.finite | ~target_active)
                & schedule.evidence.invalid_points
                == 0
            ),
            invalid_sources=schedule.evidence.invalid_points,
            invalid_targets=invalid_targets,
            required_nodes=schedule.evidence.required_nodes,
            node_capacity=schedule.evidence.node_capacity,
            maximum_leaf_occupancy=schedule.evidence.maximum_leaf_occupancy,
            required_leaf_interactions=required_leaf_interactions,
            required_pairs=required_pairs,
            pair_capacity=jnp.asarray(self.maximum_pairs, dtype=jnp.int32),
            pair_overflow=pair_overflow,
        )
        return MortonRadiusRelationResult(
            relation=relation,
            evidence=evidence,
            storage_to_logical=schedule.point_order.storage_to_logical,
            logical_to_storage=schedule.point_order.logical_to_storage,
            logical_leaf_slots=schedule.logical_point_leaf_slots,
            leaf_counts=schedule.node_item_counts[:leaf_capacity],
            leaf_offsets=schedule.node_item_starts[:leaf_capacity],
        )


__all__ = [
    "MortonNeighborQueryEvidence",
    "MortonNeighborQueryPlan",
    "MortonNeighborQueryResult",
    "MortonRadiusRelationEvidence",
    "MortonRadiusRelationPlan",
    "MortonRadiusRelationResult",
    "SpatialDistanceBackend",
]
