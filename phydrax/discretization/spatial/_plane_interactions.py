#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.sparse import EdgeRelation

from ._plane_schedule import MortonPlaneSchedulePlan, MortonPlaneScheduleState


class MortonPlaneInteractionEvidence(NonTrainableState, StrictModule):
    """Capacity and geometric evidence for one dual-tree traversal."""

    successful: jax.Array
    schedule_successful: jax.Array
    required_queue: jax.Array
    queue_capacity: jax.Array
    queue_overflow: jax.Array
    required_far: jax.Array
    far_capacity: jax.Array
    far_overflow: jax.Array
    required_near: jax.Array
    near_capacity: jax.Array
    near_overflow: jax.Array
    maximum_accepted_ratio: jax.Array
    finite: jax.Array
    complete: jax.Array


class MortonPlaneInteractionState(NonTrainableState, StrictModule):
    """Accepted node interactions and exact leaf-completion routes."""

    far: EdgeRelation
    near: EdgeRelation
    evidence: MortonPlaneInteractionEvidence


def _pack_queue(
    sources: jax.Array,
    targets: jax.Array,
    valid: jax.Array,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    flat_sources = sources.reshape((-1,)).astype(jnp.int32)
    flat_targets = targets.reshape((-1,)).astype(jnp.int32)
    flat_valid = valid.reshape((-1,))
    required = jnp.sum(flat_valid, dtype=jnp.int32)
    selected = jnp.nonzero(flat_valid, size=capacity, fill_value=0)[0]
    active = jnp.arange(capacity, dtype=jnp.int32) < jnp.minimum(required, capacity)
    return flat_sources[selected], flat_targets[selected], active, required


def _append_far(
    buffer_sources: jax.Array,
    buffer_targets: jax.Array,
    required: jax.Array,
    sources: jax.Array,
    targets: jax.Array,
    valid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Append the stable accepted prefix without retaining traversal steps."""
    valid = valid.reshape((-1,))
    ranks = jnp.cumsum(valid, dtype=jnp.int32)
    slots = jnp.where(valid, required + ranks - 1, buffer_sources.size)
    return (
        buffer_sources.at[slots].set(sources.reshape((-1,)), mode="drop"),
        buffer_targets.at[slots].set(targets.reshape((-1,)), mode="drop"),
        required + jnp.sum(valid, dtype=jnp.int32),
    )


class MortonPlaneInteractionPlan(StrictModule):
    """Build exact far/near coverage by deterministic dual-tree refinement."""

    source_schedule_plan: MortonPlaneSchedulePlan
    target_schedule_plan: MortonPlaneSchedulePlan
    opening_angle: float = eqx.field(static=True)
    queue_capacity: int = eqx.field(static=True)
    frontier_widths: tuple[int, ...] = eqx.field(static=True)
    far_capacity: int = eqx.field(static=True)
    near_capacity: int = eqx.field(static=True)
    maximum_node_radius: float | None = eqx.field(static=True)
    interaction_cutoff: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_schedule_plan: MortonPlaneSchedulePlan,
        target_schedule_plan: MortonPlaneSchedulePlan | None = None,
        *,
        opening_angle: float,
        queue_capacity: int,
        far_capacity: int,
        near_capacity: int,
        maximum_node_radius: float | None = None,
        interaction_cutoff: float | None = None,
        frontier_widths: tuple[int, ...] | None = None,
    ) -> None:
        theta = float(opening_angle)
        queue = int(queue_capacity)
        far = int(far_capacity)
        near = int(near_capacity)
        node_radius = None if maximum_node_radius is None else float(maximum_node_radius)
        cutoff = None if interaction_cutoff is None else float(interaction_cutoff)
        target_plan = (
            source_schedule_plan if target_schedule_plan is None else target_schedule_plan
        )
        if (
            source_schedule_plan.address_plan.dimension
            != target_plan.address_plan.dimension
        ):
            raise ValueError("Source and target schedule dimensions must match.")
        if (
            source_schedule_plan.address_plan.lower != target_plan.address_plan.lower
            or source_schedule_plan.address_plan.upper != target_plan.address_plan.upper
            or source_schedule_plan.address_plan.periodic_axes
            != target_plan.address_plan.periodic_axes
        ):
            raise ValueError(
                "Source and target schedules must share one physical address domain."
            )
        if not 0.0 < theta < 1.0:
            raise ValueError("opening_angle must lie strictly between zero and one.")
        if queue < 1 or far < 1 or near < 1:
            raise ValueError("Dual-tree interaction capacities must be positive.")
        widths = (queue,) if frontier_widths is None else frontier_widths
        if (
            not isinstance(widths, tuple)
            or not widths
            or any(type(width) is not int or width < 1 for width in widths)
            or any(left >= right for left, right in zip(widths, widths[1:]))
            or widths[-1] != queue
        ):
            raise ValueError(
                "frontier_widths must be increasing positive integer widths ending "
                "at queue_capacity."
            )
        if node_radius is not None and (
            not math.isfinite(node_radius) or node_radius <= 0.0
        ):
            raise ValueError("maximum_node_radius must be finite and positive.")
        if cutoff is not None and (not math.isfinite(cutoff) or cutoff <= 0.0):
            raise ValueError("interaction_cutoff must be finite and positive.")
        self.source_schedule_plan = source_schedule_plan
        self.target_schedule_plan = target_plan
        self.opening_angle = theta
        self.queue_capacity = queue
        self.frontier_widths = widths
        self.far_capacity = far
        self.near_capacity = near
        self.maximum_node_radius = node_radius
        self.interaction_cutoff = cutoff
        self.plan_id = canonical_fingerprint(
            {
                "kind": "morton-plane-interaction-plan",
                "source_schedule_plan_id": source_schedule_plan.plan_id,
                "target_schedule_plan_id": target_plan.plan_id,
                "opening_angle": theta,
                "queue_capacity": queue,
                "frontier_widths": widths,
                "far_capacity": far,
                "near_capacity": near,
                "maximum_node_radius": node_radius,
                "interaction_cutoff": cutoff,
            }
        )

    def _classify(
        self,
        source_schedule: MortonPlaneScheduleState,
        target: MortonPlaneScheduleState,
        source_nodes: jax.Array,
        target_nodes: jax.Array,
        active: jax.Array,
        same: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        source_center = source_schedule.node_centers[source_nodes]
        target_center = target.node_centers[target_nodes]
        source_half_width = source_schedule.node_half_widths[source_nodes]
        target_half_width = target.node_half_widths[target_nodes]
        geometry_finite = (
            jnp.all(jnp.isfinite(source_center), axis=-1)
            & jnp.all(jnp.isfinite(target_center), axis=-1)
            & jnp.all(jnp.isfinite(source_half_width), axis=-1)
            & jnp.all(jnp.isfinite(target_half_width), axis=-1)
        )
        center_distance = jnp.sqrt(jnp.sum((target_center - source_center) ** 2, axis=-1))
        extent = jnp.sqrt(jnp.sum((source_half_width + target_half_width) ** 2, axis=-1))
        ratio = extent / jnp.maximum(
            center_distance, jnp.asarray(jnp.finfo(center_distance.dtype).tiny)
        )
        geometry_finite = (
            geometry_finite
            & jnp.isfinite(center_distance)
            & jnp.isfinite(extent)
            & jnp.isfinite(ratio)
        )
        source_radius = jnp.sqrt(jnp.sum(source_half_width**2, axis=-1))
        target_radius = jnp.sqrt(jnp.sum(target_half_width**2, axis=-1))
        resolved = (
            jnp.asarray(True)
            if self.maximum_node_radius is None
            else (source_radius <= self.maximum_node_radius)
            & (target_radius <= self.maximum_node_radius)
        )
        distinct = jnp.asarray(not same) | (source_nodes != target_nodes)
        if self.interaction_cutoff is None:
            outside = jnp.zeros_like(active)
            fully_inside = jnp.ones_like(active)
        else:
            outside = jnp.maximum(center_distance - extent, 0.0) > self.interaction_cutoff
            fully_inside = center_distance + extent <= self.interaction_cutoff
        accept = (
            active
            & geometry_finite
            & distinct
            & resolved
            & fully_inside
            & (extent < self.opening_angle * center_distance)
        )
        return (
            accept,
            active & ~accept & ~outside,
            ratio,
            jnp.all(~active | geometry_finite),
        )

    def recheck(
        self,
        source_schedule: MortonPlaneScheduleState,
        target_schedule: MortonPlaneScheduleState | None = None,
        /,
        *,
        interactions: MortonPlaneInteractionState,
        same_support: bool | None = None,
    ) -> MortonPlaneInteractionEvidence:
        """Recheck retained far pairs against current geometry.

        The caller must establish unchanged schedule topology and membership.
        Cutoff traversals require rebuilding: retained pairs cannot certify
        coverage of previously omitted pairs after motion.
        """
        target = source_schedule if target_schedule is None else target_schedule
        same = target_schedule is None if same_support is None else bool(same_support)
        self._validate_schedules(source_schedule, target, target_schedule, same)
        if (
            interactions.far.source_size != self.source_schedule_plan.node_capacity
            or interactions.far.target_size != self.target_schedule_plan.node_capacity
            or interactions.near.source_size != self.source_schedule_plan.node_capacity
            or interactions.near.target_size != self.target_schedule_plan.node_capacity
        ):
            raise ValueError("interactions do not match the interaction plan.")
        accepted, _, ratios, finite = self._classify(
            source_schedule,
            target,
            interactions.far.source_indices,
            interactions.far.target_indices,
            interactions.far.valid,
            same,
        )
        finite = finite & self._geometry_finite(source_schedule, target)
        schedule_successful = (
            source_schedule.evidence.successful & target.evidence.successful
        )
        complete = (
            interactions.evidence.complete
            & interactions.evidence.successful
            & schedule_successful
            & finite
            & jnp.all(~interactions.far.valid | accepted)
            & jnp.asarray(self.interaction_cutoff is None)
        )
        return eqx.tree_at(
            lambda evidence: (
                evidence.successful,
                evidence.schedule_successful,
                evidence.maximum_accepted_ratio,
                evidence.finite,
                evidence.complete,
            ),
            interactions.evidence,
            (
                complete,
                schedule_successful,
                jnp.max(jnp.where(interactions.far.valid, ratios, 0.0), initial=0.0),
                finite,
                complete,
            ),
        )

    def _geometry_finite(
        self,
        source: MortonPlaneScheduleState,
        target: MortonPlaneScheduleState,
    ) -> jax.Array:
        return jnp.all(
            ~source.node_active
            | (
                jnp.all(jnp.isfinite(source.node_centers), axis=-1)
                & jnp.all(jnp.isfinite(source.node_half_widths), axis=-1)
            )
        ) & jnp.all(
            ~target.node_active
            | (
                jnp.all(jnp.isfinite(target.node_centers), axis=-1)
                & jnp.all(jnp.isfinite(target.node_half_widths), axis=-1)
            )
        )

    def _validate_schedules(
        self,
        source_schedule: MortonPlaneScheduleState,
        target: MortonPlaneScheduleState,
        target_schedule: MortonPlaneScheduleState | None,
        same: bool,
    ) -> None:
        if (
            same
            and target_schedule is not None
            and target_schedule is not source_schedule
        ):
            raise ValueError("same_support requires one aliased source/target schedule.")
        if source_schedule.node_active.shape != (
            self.source_schedule_plan.node_capacity,
        ):
            raise ValueError("source_schedule does not match the interaction plan.")
        if target.node_active.shape != (self.target_schedule_plan.node_capacity,):
            raise ValueError("target_schedule does not match the interaction plan.")

    def build(
        self,
        source_schedule: MortonPlaneScheduleState,
        target_schedule: MortonPlaneScheduleState | None = None,
        /,
        *,
        same_support: bool | None = None,
    ) -> MortonPlaneInteractionState:
        target = source_schedule if target_schedule is None else target_schedule
        same = target_schedule is None if same_support is None else bool(same_support)
        self._validate_schedules(source_schedule, target, target_schedule, same)
        source_top_plane = self.source_schedule_plan.plane_count - 1
        target_top_plane = self.target_schedule_plan.plane_count - 1
        source_top_capacity = self.source_schedule_plan.plane_capacities[-1]
        target_top_capacity = self.target_schedule_plan.plane_capacities[-1]
        source_top_rank = jnp.arange(source_top_capacity, dtype=jnp.int32)
        target_top_rank = jnp.arange(target_top_capacity, dtype=jnp.int32)
        source_top_nodes = (
            source_schedule.plane_offsets[source_top_plane] + source_top_rank
        )
        target_top_nodes = target.plane_offsets[target_top_plane] + target_top_rank
        source_top_valid = (
            source_top_rank < source_schedule.plane_active_counts[source_top_plane]
        )
        target_top_valid = target_top_rank < target.plane_active_counts[target_top_plane]
        seed_sources = jnp.broadcast_to(
            source_top_nodes[None, :],
            (target_top_capacity, source_top_capacity),
        )
        seed_targets = jnp.broadcast_to(
            target_top_nodes[:, None],
            (target_top_capacity, source_top_capacity),
        )
        seed_valid = target_top_valid[:, None] & source_top_valid[None, :]
        sources, targets, queue_valid, seed_required = _pack_queue(
            seed_sources,
            seed_targets,
            seed_valid,
            self.queue_capacity,
        )
        maximum_required_queue = seed_required
        queue_overflow = seed_required > self.queue_capacity
        far_sources = jnp.zeros((self.far_capacity,), dtype=jnp.int32)
        far_targets = jnp.zeros((self.far_capacity,), dtype=jnp.int32)
        required_far = jnp.asarray(0, dtype=jnp.int32)
        maximum_ratio = jnp.asarray(0.0, dtype=source_schedule.node_centers.dtype)
        finite = self._geometry_finite(source_schedule, target)
        target_offsets = jnp.arange(
            self.target_schedule_plan.coarsening_factor, dtype=jnp.int32
        )
        source_offsets = jnp.arange(
            self.source_schedule_plan.coarsening_factor, dtype=jnp.int32
        )

        def split_target(
            source_nodes: jax.Array,
            target_nodes: jax.Array,
            active: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            has_children = target.node_planes[target_nodes] > 0
            starts = jnp.where(
                has_children,
                target.node_child_starts[target_nodes],
                target_nodes,
            )
            counts = jnp.where(
                has_children,
                target.node_child_counts[target_nodes],
                1,
            )
            expanded_targets = starts[:, None] + target_offsets[None, :]
            expanded_sources = jnp.broadcast_to(
                source_nodes[:, None], expanded_targets.shape
            )
            expanded_valid = active[:, None] & (target_offsets[None, :] < counts[:, None])
            return expanded_sources, expanded_targets, expanded_valid

        def split_source(
            source_nodes: jax.Array,
            target_nodes: jax.Array,
            active: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            has_children = source_schedule.node_planes[source_nodes] > 0
            starts = jnp.where(
                has_children,
                source_schedule.node_child_starts[source_nodes],
                source_nodes,
            )
            counts = jnp.where(
                has_children,
                source_schedule.node_child_counts[source_nodes],
                1,
            )
            expanded_sources = starts[:, None] + source_offsets[None, :]
            expanded_targets = jnp.broadcast_to(
                target_nodes[:, None], expanded_sources.shape
            )
            expanded_valid = active[:, None] & (source_offsets[None, :] < counts[:, None])
            return expanded_sources, expanded_targets, expanded_valid

        def step(
            state: tuple[jax.Array, ...],
            split: Callable[
                [jax.Array, jax.Array, jax.Array],
                tuple[jax.Array, jax.Array, jax.Array],
            ]
            | None,
        ) -> tuple[jax.Array, ...]:
            def branch_for_width(
                width: int,
            ) -> Callable[[tuple[jax.Array, ...]], tuple[jax.Array, ...]]:
                def branch(operand: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
                    (
                        current_sources,
                        current_targets,
                        current_valid,
                        maximum_required,
                        overflow,
                        buffer_sources,
                        buffer_targets,
                        far_required,
                        ratio_maximum,
                        geometry_finite,
                    ) = operand
                    expanded_sources = current_sources[:width]
                    expanded_targets = current_targets[:width]
                    expanded_valid = current_valid[:width]
                    if split is not None:
                        expanded_sources, expanded_targets, expanded_valid = split(
                            expanded_sources, expanded_targets, expanded_valid
                        )
                    accept, opened, ratios, step_finite = self._classify(
                        source_schedule,
                        target,
                        expanded_sources,
                        expanded_targets,
                        expanded_valid,
                        same,
                    )
                    buffer_sources, buffer_targets, far_required = _append_far(
                        buffer_sources,
                        buffer_targets,
                        far_required,
                        expanded_sources,
                        expanded_targets,
                        accept,
                    )
                    next_sources, next_targets, next_valid, required = _pack_queue(
                        expanded_sources,
                        expanded_targets,
                        opened,
                        self.queue_capacity,
                    )
                    return (
                        next_sources,
                        next_targets,
                        next_valid,
                        jnp.maximum(maximum_required, required),
                        overflow | (required > self.queue_capacity),
                        buffer_sources,
                        buffer_targets,
                        far_required,
                        jnp.maximum(
                            ratio_maximum,
                            jnp.max(jnp.where(accept, ratios, 0.0), initial=0.0),
                        ),
                        geometry_finite & step_finite,
                    )

                return branch

            if len(self.frontier_widths) == 1:
                return branch_for_width(self.queue_capacity)(state)
            live_count = jnp.sum(state[2], dtype=jnp.int32)
            rung = jnp.sum(
                live_count > jnp.asarray(self.frontier_widths[:-1], dtype=jnp.int32),
                dtype=jnp.int32,
            )
            return jax.lax.switch(
                rung,
                tuple(branch_for_width(width) for width in self.frontier_widths),
                state,
            )

        state = (
            sources,
            targets,
            queue_valid,
            maximum_required_queue,
            queue_overflow,
            far_sources,
            far_targets,
            required_far,
            maximum_ratio,
            finite,
        )
        state = step(state, None)
        for _ in range(self.target_schedule_plan.plane_count - 1):
            state = step(state, split_target)
        for _ in range(self.source_schedule_plan.plane_count - 1):
            state = step(state, split_source)
        (
            sources,
            targets,
            queue_valid,
            maximum_required_queue,
            queue_overflow,
            far_sources,
            far_targets,
            required_far,
            maximum_ratio,
            finite,
        ) = state
        far_valid = jnp.arange(self.far_capacity, dtype=jnp.int32) < jnp.minimum(
            required_far, self.far_capacity
        )
        far_overflow = required_far > self.far_capacity

        required_near = jnp.sum(queue_valid, dtype=jnp.int32)
        near_selected = jnp.nonzero(queue_valid, size=self.near_capacity, fill_value=0)[0]
        near_valid = jnp.arange(self.near_capacity, dtype=jnp.int32) < jnp.minimum(
            required_near, self.near_capacity
        )
        near_overflow = required_near > self.near_capacity
        finite = finite & jnp.isfinite(maximum_ratio)
        complete = (
            source_schedule.evidence.successful
            & target.evidence.successful
            & ~queue_overflow
            & ~far_overflow
            & ~near_overflow
            & finite
        )
        far_valid = far_valid & complete
        near_valid = near_valid & complete
        far = EdgeRelation(
            jnp.where(far_valid, far_sources, 0),
            jnp.where(far_valid, far_targets, 0),
            source_size=self.source_schedule_plan.node_capacity,
            target_size=self.target_schedule_plan.node_capacity,
            valid=far_valid,
        )
        near = EdgeRelation(
            jnp.where(near_valid, sources[near_selected], 0),
            jnp.where(near_valid, targets[near_selected], 0),
            source_size=self.source_schedule_plan.node_capacity,
            target_size=self.target_schedule_plan.node_capacity,
            valid=near_valid,
        )
        evidence = MortonPlaneInteractionEvidence(
            successful=complete,
            schedule_successful=(
                source_schedule.evidence.successful & target.evidence.successful
            ),
            required_queue=maximum_required_queue,
            queue_capacity=jnp.asarray(self.queue_capacity, dtype=jnp.int32),
            queue_overflow=queue_overflow,
            required_far=required_far,
            far_capacity=jnp.asarray(self.far_capacity, dtype=jnp.int32),
            far_overflow=far_overflow,
            required_near=required_near,
            near_capacity=jnp.asarray(self.near_capacity, dtype=jnp.int32),
            near_overflow=near_overflow,
            maximum_accepted_ratio=maximum_ratio,
            finite=finite,
            complete=complete,
        )
        return MortonPlaneInteractionState(far=far, near=near, evidence=evidence)


__all__ = [
    "MortonPlaneInteractionEvidence",
    "MortonPlaneInteractionPlan",
    "MortonPlaneInteractionState",
]
