#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import ceil
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._morton import MortonAddressPlan
from ._neighbor_sources import MortonSourcePlan, PreparedMortonSources


@final
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
    invalid_padding: jax.Array


@final
class MortonPlaneScheduleState(NonTrainableState, StrictModule):
    """Compact execution planes over one canonical Morton point order."""

    point_order: PreparedMortonSources
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
    bounding_padding: jax.Array
    epoch: jax.Array
    evidence: MortonPlaneBuildEvidence


@final
class MortonPlaneTransition(NonTrainableState, StrictModule):
    """Atomic compact-plane refresh result."""

    candidate: MortonPlaneScheduleState
    accepted: MortonPlaneScheduleState
    accepted_candidate: jax.Array
    refitted: jax.Array
    rebuilt: jax.Array


@final
class _MortonPlaneGeometry(NonTrainableState, StrictModule):
    """Current numerical geometry on frozen contiguous plane groups."""

    node_bit_levels: jax.Array
    node_prefixes: jax.Array
    node_centers: jax.Array
    node_half_widths: jax.Array
    node_scales: jax.Array
    scale_exponents: jax.Array
    valid_scales: jax.Array
    bounding_padding: jax.Array
    padding_valid: jax.Array


def _common_prefix(
    first: jax.Array,
    last: jax.Array,
    active: jax.Array,
    *,
    total_bits: int,
) -> tuple[jax.Array, jax.Array]:
    xor = jnp.asarray(first, dtype=jnp.uint64) ^ jnp.asarray(last, dtype=jnp.uint64)
    shifts = jnp.arange(total_bits - 1, -1, -1, dtype=jnp.uint64)
    differs = ((xor[:, None] >> shifts[None, :]) & jnp.uint64(1)).astype(jnp.bool_)
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
    padding = group_count * group_size - coordinates.shape[0]
    padded_coordinates = jnp.pad(coordinates, ((0, padding), (0, 0)))
    padded_valid = jnp.pad(valid, (0, padding))
    grouped_coordinates = padded_coordinates.reshape(
        (group_count, group_size, coordinates.shape[1])
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


@final
class MortonPlaneSchedulePlan(StrictModule):
    """Build compact Morton-ordered execution planes with contiguous children."""

    address_plan: MortonAddressPlan
    source_plan: MortonSourcePlan
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
                "node_capacity must be positive and no larger than the compact plane capacity."
            )

        self.source_plan = MortonSourcePlan(address_plan, points)
        self.address_plan = address_plan
        self.point_capacity = points
        self.node_capacity = nodes
        self.maximum_leaf_occupancy = leaf_occupancy
        self.coarsening_factor = coarse
        self.target_top_nodes = top_nodes
        self.plane_capacities = tuple(capacities)
        self.rectangular_capacity = rectangular
        self.plan_id = canonical_fingerprint(
            {
                "kind": "morton-plane-schedule-plan",
                "address_plan_id": address_plan.plan_id,
                "source_plan_id": self.source_plan.plan_id,
                "point_capacity": points,
                "node_capacity": nodes,
                "maximum_leaf_occupancy": leaf_occupancy,
                "coarsening_factor": coarse,
                "target_top_nodes": top_nodes,
            }
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
        bounding_padding: jax.Array | float = 0.0,
        epoch: int | jax.Array = 0,
    ) -> MortonPlaneScheduleState:
        point_order = self.source_plan.prepare(
            points, active_mask=active_mask, stable_ids=stable_ids
        )
        return self._build_prepared(
            point_order, bounding_padding=bounding_padding, epoch=epoch
        )

    def _build_prepared(
        self,
        point_order: PreparedMortonSources,
        *,
        bounding_padding: jax.Array | float,
        epoch: int | jax.Array,
    ) -> MortonPlaneScheduleState:
        active_count = point_order.active_count
        plane_active: list[jax.Array] = []
        plane_starts: list[jax.Array] = []
        plane_counts: list[jax.Array] = []
        leaf_capacity = self.plane_capacities[0]
        leaf_slots = jnp.arange(leaf_capacity, dtype=jnp.int32)
        leaf_starts = leaf_slots * self.maximum_leaf_occupancy
        leaf_counts = jnp.clip(
            active_count - leaf_starts,
            0,
            self.maximum_leaf_occupancy,
        ).astype(jnp.int32)
        leaf_active = leaf_counts > 0
        plane_active.append(leaf_active)
        plane_starts.append(jnp.where(leaf_active, leaf_starts, 0))
        plane_counts.append(leaf_counts)

        for plane in range(1, self.plane_count):
            capacity = self.plane_capacities[plane]
            slots = jnp.arange(capacity, dtype=jnp.int32)
            child_starts = slots * self.coarsening_factor
            child_counts = jnp.clip(
                jnp.sum(plane_active[plane - 1], dtype=jnp.int32) - child_starts,
                0,
                self.coarsening_factor,
            ).astype(jnp.int32)
            active = child_counts > 0

            item_starts = child_starts * (
                self.maximum_leaf_occupancy * self.coarsening_factor ** (plane - 1)
            )
            item_counts = jnp.clip(
                active_count - item_starts,
                0,
                self.maximum_leaf_occupancy * self.coarsening_factor**plane,
            ).astype(jnp.int32)
            plane_active.append(active)
            plane_starts.append(jnp.where(active, item_starts, 0))
            plane_counts.append(item_counts)

        rectangular_offsets = []
        running = 0
        for capacity in self.plane_capacities:
            rectangular_offsets.append(running)
            running += capacity
        sentinel = self.rectangular_capacity

        flat_active = jnp.concatenate(plane_active)
        flat_starts = jnp.concatenate(plane_starts)
        flat_counts = jnp.concatenate(plane_counts)
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
        node_planes = jnp.where(packed_active, flat_planes[safe_selected], 0)
        node_item_starts = jnp.where(packed_active, flat_starts[safe_selected], 0)
        node_item_counts = jnp.where(packed_active, flat_counts[safe_selected], 0)
        geometry = self._geometry(
            point_order,
            node_active=packed_active,
            node_planes=node_planes,
            node_item_starts=node_item_starts,
            node_item_counts=node_item_counts,
            bounding_padding=bounding_padding,
        )

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

        evidence = self._evidence(
            point_order,
            geometry,
            required_nodes=required_nodes,
            node_active=packed_active,
            active_leaves=active_per_plane[0],
            maximum_leaf_occupancy=jnp.max(leaf_counts, initial=0),
        )
        return MortonPlaneScheduleState(
            point_order=point_order,
            plane_offsets=plane_offsets,
            plane_active_counts=active_per_plane,
            node_active=packed_active,
            node_planes=node_planes,
            node_bit_levels=geometry.node_bit_levels,
            node_prefixes=geometry.node_prefixes,
            node_parents=rectangular_to_packed[parent_rectangular],
            node_child_starts=rectangular_to_packed[child_rectangular],
            node_child_counts=child_counts,
            node_item_starts=node_item_starts,
            node_item_counts=node_item_counts,
            node_centers=geometry.node_centers,
            node_half_widths=geometry.node_half_widths,
            node_scales=geometry.node_scales,
            sorted_point_leaf_slots=sorted_point_leaf_slots,
            logical_point_leaf_slots=logical_point_leaf_slots,
            bounding_padding=geometry.bounding_padding,
            epoch=jnp.asarray(epoch, dtype=jnp.int32),
            evidence=evidence,
        )

    def _geometry(
        self,
        point_order: PreparedMortonSources,
        *,
        node_active: jax.Array,
        node_planes: jax.Array,
        node_item_starts: jax.Array,
        node_item_counts: jax.Array,
        bounding_padding: jax.Array | float,
    ) -> _MortonPlaneGeometry:
        coordinates = point_order.sorted_coordinates
        padding = jnp.asarray(bounding_padding, dtype=coordinates.dtype)
        if padding.ndim == 0:
            padding = jnp.full(
                (self.address_plan.dimension,), padding, dtype=padding.dtype
            )
        if padding.shape != (self.address_plan.dimension,):
            raise ValueError("bounding_padding must be scalar or match dimension.")
        padding_valid = jnp.all(jnp.isfinite(padding) & (padding >= 0))
        safe_padding = jnp.where(padding_valid, padding, 0.0)

        leaf_capacity = self.plane_capacities[0]
        lower, upper = _group_bounds(
            coordinates,
            point_order.sorted_active,
            group_count=leaf_capacity,
            group_size=self.maximum_leaf_occupancy,
        )
        active = (
            jnp.arange(leaf_capacity, dtype=jnp.int32) * self.maximum_leaf_occupancy
            < point_order.active_count
        )
        lower_by_plane = [lower]
        upper_by_plane = [upper]
        offsets = [0]
        group_sizes = [self.maximum_leaf_occupancy]
        for plane in range(1, self.plane_count):
            capacity = self.plane_capacities[plane]
            child_capacity = self.plane_capacities[plane - 1]
            group_padding = capacity * self.coarsening_factor - child_capacity
            grouped_active = jnp.pad(active, (0, group_padding)).reshape(
                (capacity, self.coarsening_factor)
            )
            grouped_lower = jnp.pad(lower, ((0, group_padding), (0, 0))).reshape(
                (capacity, self.coarsening_factor, self.address_plan.dimension)
            )
            grouped_upper = jnp.pad(upper, ((0, group_padding), (0, 0))).reshape(
                (capacity, self.coarsening_factor, self.address_plan.dimension)
            )
            lower = jnp.min(
                jnp.where(
                    grouped_active[..., None],
                    grouped_lower,
                    jnp.asarray(jnp.inf, dtype=coordinates.dtype),
                ),
                axis=1,
            )
            upper = jnp.max(
                jnp.where(
                    grouped_active[..., None],
                    grouped_upper,
                    jnp.asarray(-jnp.inf, dtype=coordinates.dtype),
                ),
                axis=1,
            )
            active = jnp.any(grouped_active, axis=1)
            lower = jnp.where(active[:, None], lower, jnp.zeros_like(lower))
            upper = jnp.where(active[:, None], upper, jnp.zeros_like(upper))
            lower_by_plane.append(lower)
            upper_by_plane.append(upper)
            offsets.append(offsets[-1] + child_capacity)
            group_sizes.append(group_sizes[-1] * self.coarsening_factor)

        rectangular_slots = (
            jnp.asarray(offsets, dtype=jnp.int32)[node_planes]
            + node_item_starts // jnp.asarray(group_sizes, dtype=jnp.int32)[node_planes]
        )
        node_lower = jnp.where(
            node_active[:, None],
            jnp.concatenate(lower_by_plane)[rectangular_slots] - safe_padding,
            0,
        )
        node_upper = jnp.where(
            node_active[:, None],
            jnp.concatenate(upper_by_plane)[rectangular_slots] + safe_padding,
            0,
        )
        half_widths = 0.5 * (node_upper - node_lower)
        radii = jax.lax.stop_gradient(
            jnp.sqrt(jnp.sum(half_widths * half_widths, axis=-1))
        )
        scale_floor = jnp.asarray(
            jnp.finfo(coordinates.dtype).tiny, dtype=coordinates.dtype
        )
        scale_exponents = jnp.ceil(jnp.log2(jnp.maximum(radii, scale_floor)))
        scales = jnp.where(
            node_active,
            jnp.exp2(scale_exponents),
            jnp.asarray(1.0, dtype=coordinates.dtype),
        )
        first = jnp.minimum(node_item_starts, self.point_capacity - 1)
        last = jnp.minimum(
            node_item_starts + jnp.maximum(node_item_counts, 1) - 1,
            self.point_capacity - 1,
        )
        bit_levels, prefixes = _common_prefix(
            point_order.sorted_codes[first],
            point_order.sorted_codes[last],
            node_active,
            total_bits=self.address_plan.dimension * self.address_plan.maximum_depth,
        )
        return _MortonPlaneGeometry(
            node_bit_levels=bit_levels,
            node_prefixes=prefixes,
            node_centers=0.5 * (node_lower + node_upper),
            node_half_widths=half_widths,
            node_scales=scales,
            scale_exponents=scale_exponents,
            valid_scales=(~node_active) | (jnp.isfinite(scales) & (scales > 0)),
            bounding_padding=safe_padding,
            padding_valid=padding_valid,
        )

    def _evidence(
        self,
        point_order: PreparedMortonSources,
        geometry: _MortonPlaneGeometry,
        *,
        required_nodes: jax.Array,
        node_active: jax.Array,
        active_leaves: jax.Array,
        maximum_leaf_occupancy: jax.Array,
    ) -> MortonPlaneBuildEvidence:
        run_boundary = point_order.sorted_active & jnp.concatenate(
            (
                jnp.asarray([True], dtype=jnp.bool_),
                (~point_order.sorted_active[:-1])
                | (point_order.sorted_codes[1:] != point_order.sorted_codes[:-1]),
            )
        )
        run_count = jnp.sum(run_boundary, dtype=jnp.int32)
        run_starts = jnp.nonzero(
            run_boundary,
            size=self.point_capacity,
            fill_value=self.point_capacity,
        )[0].astype(jnp.int32)
        run_valid = jnp.arange(self.point_capacity, dtype=jnp.int32) < run_count
        run_next = jnp.concatenate((run_starts[1:], point_order.active_count[None]))
        run_lengths = jnp.where(run_valid, run_next - run_starts, 0)
        successful = (
            (point_order.invalid_points == 0)
            & point_order.stable_ids_unique
            & (required_nodes <= self.node_capacity)
            & jnp.all(geometry.valid_scales)
            & geometry.padding_valid
        )
        return MortonPlaneBuildEvidence(
            successful=successful,
            active_points=point_order.active_count,
            invalid_points=point_order.invalid_points,
            stable_ids_unique=point_order.stable_ids_unique,
            required_nodes=required_nodes,
            node_capacity=jnp.asarray(self.node_capacity, dtype=jnp.int32),
            active_nodes=jnp.minimum(required_nodes, self.node_capacity),
            active_leaves=active_leaves,
            maximum_leaf_occupancy=maximum_leaf_occupancy,
            oversized_terminal_buckets=jnp.sum(
                run_valid & (run_lengths > self.maximum_leaf_occupancy),
                dtype=jnp.int32,
            ),
            maximum_terminal_bucket_occupancy=jnp.max(run_lengths, initial=0),
            minimum_scale_exponent=jnp.where(
                required_nodes > 0,
                jnp.min(
                    jnp.where(node_active, geometry.scale_exponents, jnp.inf),
                    initial=jnp.inf,
                ),
                0.0,
            ),
            maximum_scale_exponent=jnp.where(
                required_nodes > 0,
                jnp.max(
                    jnp.where(node_active, geometry.scale_exponents, -jnp.inf),
                    initial=-jnp.inf,
                ),
                0.0,
            ),
            invalid_scales=jnp.sum(~geometry.valid_scales, dtype=jnp.int32),
            invalid_padding=(~geometry.padding_valid).astype(jnp.int32),
        )

    def refresh(
        self,
        previous: MortonPlaneScheduleState,
        points: jax.Array,
        *,
        active_mask: jax.Array | None = None,
        stable_ids: jax.Array | None = None,
        bounding_padding: jax.Array | float = 0.0,
    ) -> MortonPlaneTransition:
        source_transition = self.source_plan.refresh(
            previous.point_order,
            points,
            active_mask=active_mask,
            stable_ids=stable_ids,
        )
        point_order = source_transition.candidate
        # Leaf groups are fixed-size sorted ranges; every parent groups a fixed
        # number of adjacent children. Unchanged canonical order and activity
        # therefore preserve all group memberships and connectivity, regardless
        # of changes to their current common Morton prefixes.
        reuse_topology = previous.evidence.successful & source_transition.reused_order

        def refit(_: None) -> MortonPlaneScheduleState:
            geometry = self._geometry(
                point_order,
                node_active=previous.node_active,
                node_planes=previous.node_planes,
                node_item_starts=previous.node_item_starts,
                node_item_counts=previous.node_item_counts,
                bounding_padding=bounding_padding,
            )
            evidence = self._evidence(
                point_order,
                geometry,
                required_nodes=previous.evidence.required_nodes,
                node_active=previous.node_active,
                active_leaves=previous.evidence.active_leaves,
                maximum_leaf_occupancy=previous.evidence.maximum_leaf_occupancy,
            )
            return MortonPlaneScheduleState(
                point_order=point_order,
                plane_offsets=previous.plane_offsets,
                plane_active_counts=previous.plane_active_counts,
                node_active=previous.node_active,
                node_planes=previous.node_planes,
                node_bit_levels=geometry.node_bit_levels,
                node_prefixes=geometry.node_prefixes,
                node_parents=previous.node_parents,
                node_child_starts=previous.node_child_starts,
                node_child_counts=previous.node_child_counts,
                node_item_starts=previous.node_item_starts,
                node_item_counts=previous.node_item_counts,
                node_centers=geometry.node_centers,
                node_half_widths=geometry.node_half_widths,
                node_scales=geometry.node_scales,
                sorted_point_leaf_slots=previous.sorted_point_leaf_slots,
                logical_point_leaf_slots=previous.logical_point_leaf_slots,
                bounding_padding=geometry.bounding_padding,
                epoch=previous.epoch + 1,
                evidence=evidence,
            )

        candidate = jax.lax.cond(
            reuse_topology,
            refit,
            lambda _: self._build_prepared(
                point_order,
                bounding_padding=bounding_padding,
                epoch=previous.epoch + 1,
            ),
            operand=None,
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
            refitted=accepted_candidate & reuse_topology,
            rebuilt=accepted_candidate & ~reuse_topology,
        )


__all__ = [
    "MortonPlaneBuildEvidence",
    "MortonPlaneSchedulePlan",
    "MortonPlaneScheduleState",
    "MortonPlaneTransition",
]
