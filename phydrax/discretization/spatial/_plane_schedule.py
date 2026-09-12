#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import ceil

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._morton import _canonical_morton_point_order, _MortonPointOrder, MortonAddressPlan


class MortonPlaneBuildEvidence(NonTrainableState, StrictModule):
    """Completion and resource evidence for one compact plane build."""

    successful: jax.Array
    active_points: jax.Array
    invalid_points: jax.Array
    stable_ids_unique: jax.Array
    required_nodes: jax.Array
    node_capacity: jax.Array
    active_nodes: jax.Array
    active_leaves: jax.Array
    maximum_leaf_occupancy: jax.Array
    oversized_terminal_buckets: jax.Array
    maximum_terminal_bucket_occupancy: jax.Array
    minimum_scale_exponent: jax.Array
    maximum_scale_exponent: jax.Array
    invalid_scales: jax.Array


class MortonPlaneScheduleState(NonTrainableState, StrictModule):
    """Compact execution planes over one canonical Morton point order."""

    point_order: _MortonPointOrder
    plane_offsets: jax.Array
    plane_active_counts: jax.Array
    node_active: jax.Array
    node_planes: jax.Array
    node_bit_levels: jax.Array
    node_prefixes: jax.Array
    node_parents: jax.Array
    node_child_starts: jax.Array
    node_child_counts: jax.Array
    node_item_starts: jax.Array
    node_item_counts: jax.Array
    node_centers: jax.Array
    node_half_widths: jax.Array
    node_scales: jax.Array
    sorted_point_leaf_slots: jax.Array
    logical_point_leaf_slots: jax.Array
    epoch: jax.Array
    evidence: MortonPlaneBuildEvidence


class MortonPlaneTransition(NonTrainableState, StrictModule):
    """Atomic compact-plane refresh result."""

    candidate: MortonPlaneScheduleState
    accepted: MortonPlaneScheduleState
    accepted_candidate: jax.Array
    refitted: jax.Array
    rebuilt: jax.Array


def _common_prefix(
    first: jax.Array,
    last: jax.Array,
    active: jax.Array,
    *,
    total_bits: int,
) -> tuple[jax.Array, jax.Array]:
    xor = jnp.asarray(first, dtype=jnp.uint64) ^ jnp.asarray(last, dtype=jnp.uint64)
    shifts = jnp.arange(total_bits - 1, -1, -1, dtype=jnp.uint64)
    differs = ((xor[:, None] >> shifts[None, :]) & jnp.uint64(1)).astype(bool)
    has_difference = jnp.any(differs, axis=1)
    first_difference = jnp.argmax(differs, axis=1).astype(jnp.int32)
    bit_levels = jnp.where(has_difference, first_difference, total_bits)
    suffix_bits = total_bits - bit_levels
    prefixes = jnp.asarray(first, dtype=jnp.uint64) >> suffix_bits.astype(jnp.uint64)
    return (
        jnp.where(active, bit_levels, 0).astype(jnp.int32),
        jnp.where(active, prefixes, jnp.uint64(0)),
    )


def _group_bounds(
    coordinates: jax.Array,
    valid: jax.Array,
    *,
    group_count: int,
    group_size: int,
) -> tuple[jax.Array, jax.Array]:
    padding = group_count * group_size - int(coordinates.shape[0])
    padded_coordinates = jnp.pad(coordinates, ((0, padding), (0, 0)))
    padded_valid = jnp.pad(valid, (0, padding))
    grouped_coordinates = padded_coordinates.reshape(
        (group_count, group_size, int(coordinates.shape[1]))
    )
    grouped_valid = padded_valid.reshape((group_count, group_size))
    positive = jnp.asarray(jnp.inf, dtype=coordinates.dtype)
    negative = jnp.asarray(-jnp.inf, dtype=coordinates.dtype)
    lower = jnp.min(
        jnp.where(grouped_valid[..., None], grouped_coordinates, positive), axis=1
    )
    upper = jnp.max(
        jnp.where(grouped_valid[..., None], grouped_coordinates, negative), axis=1
    )
    active = jnp.any(grouped_valid, axis=1)
    return (
        jnp.where(active[:, None], lower, jnp.zeros_like(lower)),
        jnp.where(active[:, None], upper, jnp.zeros_like(upper)),
    )


class MortonPlaneSchedulePlan(StrictModule):
    """Build compact Morton-ordered execution planes with contiguous children."""

    address_plan: MortonAddressPlan
    point_capacity: int = eqx.field(static=True)
    node_capacity: int = eqx.field(static=True)
    maximum_leaf_occupancy: int = eqx.field(static=True)
    coarsening_factor: int = eqx.field(static=True)
    target_top_nodes: int = eqx.field(static=True)
    plane_capacities: tuple[int, ...] = eqx.field(static=True)
    rectangular_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        point_capacity: int,
        *,
        node_capacity: int | None = None,
        maximum_leaf_occupancy: int = 32,
        coarsening_factor: int = 8,
        target_top_nodes: int = 1024,
    ) -> None:
        points = int(point_capacity)
        leaf_occupancy = int(maximum_leaf_occupancy)
        coarse = int(coarsening_factor)
        top_nodes = int(target_top_nodes)
        if points < 1:
            raise ValueError("point_capacity must be positive.")
        if leaf_occupancy < 1:
            raise ValueError("maximum_leaf_occupancy must be positive.")
        if coarse < 2:
            raise ValueError("coarsening_factor must be at least two.")
        if top_nodes < 1:
            raise ValueError("target_top_nodes must be positive.")

        capacities = [ceil(points / leaf_occupancy)]
        while capacities[-1] > top_nodes:
            capacities.append(ceil(capacities[-1] / coarse))
        rectangular = sum(capacities)
        nodes = rectangular if node_capacity is None else int(node_capacity)
        if nodes < 1 or nodes > rectangular:
            raise ValueError(
                "node_capacity must be positive and no larger than the compact "
                "plane capacity."
            )

        object.__setattr__(self, "address_plan", address_plan)
        object.__setattr__(self, "point_capacity", points)
        object.__setattr__(self, "node_capacity", nodes)
        object.__setattr__(self, "maximum_leaf_occupancy", leaf_occupancy)
        object.__setattr__(self, "coarsening_factor", coarse)
        object.__setattr__(self, "target_top_nodes", top_nodes)
        object.__setattr__(self, "plane_capacities", tuple(capacities))
        object.__setattr__(self, "rectangular_capacity", rectangular)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "morton-plane-schedule-plan",
                    "address_plan_id": address_plan.plan_id,
                    "point_capacity": points,
                    "node_capacity": nodes,
                    "maximum_leaf_occupancy": leaf_occupancy,
                    "coarsening_factor": coarse,
                    "target_top_nodes": top_nodes,
                }
            ),
        )

    @property
    def plane_count(self) -> int:
        return len(self.plane_capacities)

    def build(
        self,
        points: jax.Array,
        *,
        active_mask: jax.Array | None = None,
        stable_ids: jax.Array | None = None,
        epoch: int | jax.Array = 0,
    ) -> MortonPlaneScheduleState:
        point_order = _canonical_morton_point_order(
            self.address_plan,
            points,
            point_capacity=self.point_capacity,
            active_mask=active_mask,
            stable_ids=stable_ids,
        )
        sorted_coordinates = point_order.encoding.coordinates[
            point_order.storage_to_logical
        ]
        sorted_codes = point_order.sorted_codes
        active_count = point_order.active_count
        total_bits = self.address_plan.dimension * self.address_plan.maximum_depth

        plane_active: list[jax.Array] = []
        plane_starts: list[jax.Array] = []
        plane_counts: list[jax.Array] = []
        plane_lower: list[jax.Array] = []
        plane_upper: list[jax.Array] = []
        plane_bit_levels: list[jax.Array] = []
        plane_prefixes: list[jax.Array] = []

        leaf_capacity = self.plane_capacities[0]
        leaf_slots = jnp.arange(leaf_capacity, dtype=jnp.int32)
        leaf_starts = leaf_slots * self.maximum_leaf_occupancy
        leaf_counts = jnp.clip(
            active_count - leaf_starts,
            0,
            self.maximum_leaf_occupancy,
        ).astype(jnp.int32)
        leaf_active = leaf_counts > 0
        leaf_lower, leaf_upper = _group_bounds(
            sorted_coordinates,
            point_order.sorted_active,
            group_count=leaf_capacity,
            group_size=self.maximum_leaf_occupancy,
        )
        safe_first = jnp.minimum(leaf_starts, self.point_capacity - 1)
        safe_last = jnp.minimum(
            leaf_starts + jnp.maximum(leaf_counts, 1) - 1,
            self.point_capacity - 1,
        )
        bit_levels, prefixes = _common_prefix(
            sorted_codes[safe_first],
            sorted_codes[safe_last],
            leaf_active,
            total_bits=total_bits,
        )
        plane_active.append(leaf_active)
        plane_starts.append(jnp.where(leaf_active, leaf_starts, 0))
        plane_counts.append(leaf_counts)
        plane_lower.append(leaf_lower)
        plane_upper.append(leaf_upper)
        plane_bit_levels.append(bit_levels)
        plane_prefixes.append(prefixes)

        for plane in range(1, self.plane_count):
            capacity = self.plane_capacities[plane]
            child_capacity = self.plane_capacities[plane - 1]
            slots = jnp.arange(capacity, dtype=jnp.int32)
            child_starts = slots * self.coarsening_factor
            child_counts = jnp.clip(
                jnp.sum(plane_active[plane - 1], dtype=jnp.int32) - child_starts,
                0,
                self.coarsening_factor,
            ).astype(jnp.int32)
            active = child_counts > 0

            padding = capacity * self.coarsening_factor - child_capacity
            padded_active = jnp.pad(plane_active[plane - 1], (0, padding))
            grouped_active = padded_active.reshape((capacity, self.coarsening_factor))
            lower_values = jnp.pad(
                plane_lower[plane - 1], ((0, padding), (0, 0))
            ).reshape((capacity, self.coarsening_factor, self.address_plan.dimension))
            upper_values = jnp.pad(
                plane_upper[plane - 1], ((0, padding), (0, 0))
            ).reshape((capacity, self.coarsening_factor, self.address_plan.dimension))
            positive = jnp.asarray(jnp.inf, dtype=sorted_coordinates.dtype)
            negative = jnp.asarray(-jnp.inf, dtype=sorted_coordinates.dtype)
            lower = jnp.min(
                jnp.where(grouped_active[..., None], lower_values, positive), axis=1
            )
            upper = jnp.max(
                jnp.where(grouped_active[..., None], upper_values, negative), axis=1
            )
            lower = jnp.where(active[:, None], lower, jnp.zeros_like(lower))
            upper = jnp.where(active[:, None], upper, jnp.zeros_like(upper))

            item_starts = child_starts * (
                self.maximum_leaf_occupancy * self.coarsening_factor ** (plane - 1)
            )
            item_counts = jnp.clip(
                active_count - item_starts,
                0,
                self.maximum_leaf_occupancy * self.coarsening_factor**plane,
            ).astype(jnp.int32)
            safe_first = jnp.minimum(item_starts, self.point_capacity - 1)
            safe_last = jnp.minimum(
                item_starts + jnp.maximum(item_counts, 1) - 1,
                self.point_capacity - 1,
            )
            bit_levels, prefixes = _common_prefix(
                sorted_codes[safe_first],
                sorted_codes[safe_last],
                active,
                total_bits=total_bits,
            )
            plane_active.append(active)
            plane_starts.append(jnp.where(active, item_starts, 0))
            plane_counts.append(item_counts)
            plane_lower.append(lower)
            plane_upper.append(upper)
            plane_bit_levels.append(bit_levels)
            plane_prefixes.append(prefixes)

        rectangular_offsets = []
        running = 0
        for capacity in self.plane_capacities:
            rectangular_offsets.append(running)
            running += capacity
        sentinel = self.rectangular_capacity

        flat_active = jnp.concatenate(plane_active)
        flat_starts = jnp.concatenate(plane_starts)
        flat_counts = jnp.concatenate(plane_counts)
        flat_lower = jnp.concatenate(plane_lower)
        flat_upper = jnp.concatenate(plane_upper)
        flat_bit_levels = jnp.concatenate(plane_bit_levels)
        flat_prefixes = jnp.concatenate(plane_prefixes)
        flat_planes = jnp.concatenate(
            [
                jnp.full((capacity,), plane, dtype=jnp.int32)
                for plane, capacity in enumerate(self.plane_capacities)
            ]
        )

        parent_parts: list[jax.Array] = []
        child_start_parts: list[jax.Array] = []
        child_count_parts: list[jax.Array] = []
        for plane, capacity in enumerate(self.plane_capacities):
            slots = jnp.arange(capacity, dtype=jnp.int32)
            if plane + 1 < self.plane_count:
                parent_parts.append(
                    rectangular_offsets[plane + 1] + slots // self.coarsening_factor
                )
            else:
                parent_parts.append(jnp.full((capacity,), sentinel, dtype=jnp.int32))
            if plane == 0:
                child_start_parts.append(jnp.full((capacity,), sentinel, dtype=jnp.int32))
                child_count_parts.append(jnp.zeros((capacity,), dtype=jnp.int32))
            else:
                child_start_parts.append(
                    rectangular_offsets[plane - 1] + slots * self.coarsening_factor
                )
                child_count_parts.append(
                    jnp.clip(
                        jnp.sum(plane_active[plane - 1], dtype=jnp.int32)
                        - slots * self.coarsening_factor,
                        0,
                        self.coarsening_factor,
                    ).astype(jnp.int32)
                )

        flat_parents = jnp.concatenate(parent_parts)
        flat_child_starts = jnp.concatenate(child_start_parts)
        flat_child_counts = jnp.concatenate(child_count_parts)
        required_nodes = jnp.sum(flat_active, dtype=jnp.int32)
        selected = jnp.nonzero(
            flat_active,
            size=self.node_capacity,
            fill_value=sentinel,
        )[0].astype(jnp.int32)
        packed_slots = jnp.arange(self.node_capacity, dtype=jnp.int32)
        packed_active = packed_slots < jnp.minimum(required_nodes, self.node_capacity)
        safe_selected = jnp.minimum(selected, self.rectangular_capacity - 1)

        rectangular_to_packed = jnp.full(
            (self.rectangular_capacity + 1,), -1, dtype=jnp.int32
        )
        rectangular_to_packed = rectangular_to_packed.at[
            jnp.where(packed_active, selected, sentinel)
        ].set(jnp.where(packed_active, packed_slots, -1))
        parent_rectangular = jnp.where(
            packed_active, flat_parents[safe_selected], sentinel
        )
        child_rectangular = jnp.where(
            packed_active, flat_child_starts[safe_selected], sentinel
        )
        child_counts = jnp.where(
            packed_active, flat_child_counts[safe_selected], 0
        ).astype(jnp.int32)

        active_per_plane = jnp.asarray(
            [jnp.sum(active, dtype=jnp.int32) for active in plane_active],
            dtype=jnp.int32,
        )
        plane_offsets = jnp.concatenate(
            (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(active_per_plane))
        )
        node_lower = jnp.where(packed_active[:, None], flat_lower[safe_selected], 0)
        node_upper = jnp.where(packed_active[:, None], flat_upper[safe_selected], 0)
        node_half_widths = 0.5 * (node_upper - node_lower)
        node_radii = jnp.sqrt(jnp.sum(node_half_widths * node_half_widths, axis=-1))
        scale_floor = jnp.asarray(
            jnp.finfo(sorted_coordinates.dtype).tiny,
            dtype=sorted_coordinates.dtype,
        )
        safe_radii = jnp.maximum(node_radii, scale_floor)
        scale_exponents = jnp.ceil(jnp.log2(safe_radii))
        node_scales = jnp.where(
            packed_active,
            jnp.exp2(scale_exponents),
            jnp.asarray(1.0, dtype=sorted_coordinates.dtype),
        )
        valid_scale = (~packed_active) | (jnp.isfinite(node_scales) & (node_scales > 0))

        sorted_slots = jnp.arange(self.point_capacity, dtype=jnp.int32)
        leaf_rectangular = sorted_slots // self.maximum_leaf_occupancy
        sorted_point_leaf_slots = jnp.where(
            point_order.sorted_active & (leaf_rectangular < self.node_capacity),
            rectangular_to_packed[leaf_rectangular],
            -1,
        ).astype(jnp.int32)
        logical_point_leaf_slots = (
            jnp.full((self.point_capacity,), -1, dtype=jnp.int32)
            .at[point_order.storage_to_logical]
            .set(sorted_point_leaf_slots)
        )

        run_boundary = point_order.sorted_active & jnp.concatenate(
            (
                jnp.asarray([True]),
                (~point_order.sorted_active[:-1])
                | (sorted_codes[1:] != sorted_codes[:-1]),
            )
        )
        run_count = jnp.sum(run_boundary, dtype=jnp.int32)
        run_starts = jnp.nonzero(
            run_boundary,
            size=self.point_capacity,
            fill_value=self.point_capacity,
        )[0].astype(jnp.int32)
        run_slots = jnp.arange(self.point_capacity, dtype=jnp.int32)
        run_valid = run_slots < run_count
        run_next = jnp.concatenate((run_starts[1:], active_count[None]))
        run_lengths = jnp.where(run_valid, run_next - run_starts, 0)

        successful = (
            (point_order.invalid_points == 0)
            & point_order.stable_ids_unique
            & (required_nodes <= self.node_capacity)
            & jnp.all(valid_scale)
        )
        evidence = MortonPlaneBuildEvidence(
            successful=successful,
            active_points=active_count,
            invalid_points=point_order.invalid_points,
            stable_ids_unique=point_order.stable_ids_unique,
            required_nodes=required_nodes,
            node_capacity=jnp.asarray(self.node_capacity, dtype=jnp.int32),
            active_nodes=jnp.minimum(required_nodes, self.node_capacity),
            active_leaves=active_per_plane[0],
            maximum_leaf_occupancy=jnp.max(leaf_counts, initial=0),
            oversized_terminal_buckets=jnp.sum(
                run_valid & (run_lengths > self.maximum_leaf_occupancy),
                dtype=jnp.int32,
            ),
            maximum_terminal_bucket_occupancy=jnp.max(run_lengths, initial=0),
            minimum_scale_exponent=jnp.where(
                required_nodes > 0,
                jnp.min(
                    jnp.where(packed_active, scale_exponents, jnp.inf),
                    initial=jnp.inf,
                ),
                0.0,
            ),
            maximum_scale_exponent=jnp.where(
                required_nodes > 0,
                jnp.max(
                    jnp.where(packed_active, scale_exponents, -jnp.inf),
                    initial=-jnp.inf,
                ),
                0.0,
            ),
            invalid_scales=jnp.sum(~valid_scale, dtype=jnp.int32),
        )
        return MortonPlaneScheduleState(
            point_order=point_order,
            plane_offsets=plane_offsets,
            plane_active_counts=active_per_plane,
            node_active=packed_active,
            node_planes=jnp.where(packed_active, flat_planes[safe_selected], 0),
            node_bit_levels=jnp.where(packed_active, flat_bit_levels[safe_selected], 0),
            node_prefixes=jnp.where(
                packed_active, flat_prefixes[safe_selected], jnp.uint64(0)
            ),
            node_parents=rectangular_to_packed[parent_rectangular],
            node_child_starts=rectangular_to_packed[child_rectangular],
            node_child_counts=child_counts,
            node_item_starts=jnp.where(packed_active, flat_starts[safe_selected], 0),
            node_item_counts=jnp.where(packed_active, flat_counts[safe_selected], 0),
            node_centers=0.5 * (node_lower + node_upper),
            node_half_widths=node_half_widths,
            node_scales=node_scales,
            sorted_point_leaf_slots=sorted_point_leaf_slots,
            logical_point_leaf_slots=logical_point_leaf_slots,
            epoch=jnp.asarray(epoch, dtype=jnp.int32),
            evidence=evidence,
        )

    def refresh(
        self,
        previous: MortonPlaneScheduleState,
        points: jax.Array,
        *,
        active_mask: jax.Array | None = None,
        stable_ids: jax.Array | None = None,
    ) -> MortonPlaneTransition:
        candidate = self.build(
            points,
            active_mask=active_mask,
            stable_ids=stable_ids,
            epoch=previous.epoch + 1,
        )
        same_topology = (
            jnp.array_equal(candidate.node_planes, previous.node_planes)
            & jnp.array_equal(candidate.node_bit_levels, previous.node_bit_levels)
            & jnp.array_equal(candidate.node_prefixes, previous.node_prefixes)
            & jnp.array_equal(
                candidate.point_order.sorted_stable_ids,
                previous.point_order.sorted_stable_ids,
            )
        )
        accepted_candidate = candidate.evidence.successful
        accepted = jax.lax.cond(
            accepted_candidate,
            lambda _: candidate,
            lambda _: previous,
            operand=None,
        )
        return MortonPlaneTransition(
            candidate=candidate,
            accepted=accepted,
            accepted_candidate=accepted_candidate,
            refitted=accepted_candidate & same_topology,
            rebuilt=accepted_candidate & ~same_topology,
        )


__all__ = [
    "MortonPlaneBuildEvidence",
    "MortonPlaneSchedulePlan",
    "MortonPlaneScheduleState",
    "MortonPlaneTransition",
]
