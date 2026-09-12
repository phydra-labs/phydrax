#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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


class MortonPlaneInteractionPlan(StrictModule):
    """Build exact far/near coverage by deterministic dual-tree refinement."""

    schedule_plan: MortonPlaneSchedulePlan
    opening_angle: float = eqx.field(static=True)
    queue_capacity: int = eqx.field(static=True)
    far_capacity: int = eqx.field(static=True)
    near_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule_plan: MortonPlaneSchedulePlan,
        *,
        opening_angle: float,
        queue_capacity: int,
        far_capacity: int,
        near_capacity: int,
    ) -> None:
        theta = float(opening_angle)
        queue = int(queue_capacity)
        far = int(far_capacity)
        near = int(near_capacity)
        if not 0.0 < theta < 1.0:
            raise ValueError("opening_angle must lie strictly between zero and one.")
        if queue < 1 or far < 1 or near < 1:
            raise ValueError("Dual-tree interaction capacities must be positive.")
        object.__setattr__(self, "schedule_plan", schedule_plan)
        object.__setattr__(self, "opening_angle", theta)
        object.__setattr__(self, "queue_capacity", queue)
        object.__setattr__(self, "far_capacity", far)
        object.__setattr__(self, "near_capacity", near)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "morton-plane-interaction-plan",
                    "schedule_plan_id": schedule_plan.plan_id,
                    "opening_angle": theta,
                    "queue_capacity": queue,
                    "far_capacity": far,
                    "near_capacity": near,
                }
            ),
        )

    def build(
        self,
        schedule: MortonPlaneScheduleState,
        /,
    ) -> MortonPlaneInteractionState:
        if schedule.node_active.shape != (self.schedule_plan.node_capacity,):
            raise ValueError("schedule does not match the interaction plan.")
        top_plane = self.schedule_plan.plane_count - 1
        top_capacity = self.schedule_plan.plane_capacities[-1]
        top_start = schedule.plane_offsets[top_plane]
        top_rank = jnp.arange(top_capacity, dtype=jnp.int32)
        top_nodes = top_start + top_rank
        top_valid = top_rank < schedule.plane_active_counts[top_plane]
        seed_sources = jnp.broadcast_to(top_nodes[None, :], (top_capacity, top_capacity))
        seed_targets = jnp.broadcast_to(top_nodes[:, None], (top_capacity, top_capacity))
        seed_valid = top_valid[:, None] & top_valid[None, :]
        sources, targets, queue_valid, seed_required = _pack_queue(
            seed_sources,
            seed_targets,
            seed_valid,
            self.queue_capacity,
        )
        maximum_required_queue = seed_required
        queue_overflow = seed_required > self.queue_capacity
        accepted_sources: list[jax.Array] = []
        accepted_targets: list[jax.Array] = []
        accepted_valid: list[jax.Array] = []
        accepted_ratios: list[jax.Array] = []

        def classify(
            source_nodes: jax.Array,
            target_nodes: jax.Array,
            active: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            source_center = schedule.node_centers[source_nodes]
            target_center = schedule.node_centers[target_nodes]
            center_distance = jnp.sqrt(
                jnp.sum((target_center - source_center) ** 2, axis=-1)
            )
            combined_half_width = (
                schedule.node_half_widths[source_nodes]
                + schedule.node_half_widths[target_nodes]
            )
            extent = jnp.sqrt(jnp.sum(combined_half_width**2, axis=-1))
            ratio = extent / jnp.maximum(
                center_distance,
                jnp.asarray(jnp.finfo(center_distance.dtype).tiny),
            )
            accept = (
                active
                & (source_nodes != target_nodes)
                & (extent < self.opening_angle * center_distance)
            )
            return accept, active & ~accept, ratio

        def retain_open(
            source_nodes: jax.Array,
            target_nodes: jax.Array,
            active: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            accept, opened, ratio = classify(source_nodes, target_nodes, active)
            accepted_sources.append(source_nodes.reshape((-1,)))
            accepted_targets.append(target_nodes.reshape((-1,)))
            accepted_valid.append(accept.reshape((-1,)))
            accepted_ratios.append(jnp.where(accept, ratio, 0.0).reshape((-1,)))
            next_sources, next_targets, next_valid, required = _pack_queue(
                source_nodes,
                target_nodes,
                opened,
                self.queue_capacity,
            )
            return next_sources, next_targets, next_valid, required

        sources, targets, queue_valid, required = retain_open(
            sources, targets, queue_valid
        )
        maximum_required_queue = jnp.maximum(maximum_required_queue, required)
        queue_overflow = queue_overflow | (required > self.queue_capacity)
        offsets = jnp.arange(self.schedule_plan.coarsening_factor, dtype=jnp.int32)

        def split_target(
            source_nodes: jax.Array,
            target_nodes: jax.Array,
            active: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            has_children = schedule.node_planes[target_nodes] > 0
            starts = jnp.where(
                has_children,
                schedule.node_child_starts[target_nodes],
                target_nodes,
            )
            counts = jnp.where(
                has_children,
                schedule.node_child_counts[target_nodes],
                1,
            )
            expanded_targets = starts[:, None] + offsets[None, :]
            expanded_sources = jnp.broadcast_to(
                source_nodes[:, None], expanded_targets.shape
            )
            expanded_valid = active[:, None] & (offsets[None, :] < counts[:, None])
            return expanded_sources, expanded_targets, expanded_valid

        def split_source(
            source_nodes: jax.Array,
            target_nodes: jax.Array,
            active: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            has_children = schedule.node_planes[source_nodes] > 0
            starts = jnp.where(
                has_children,
                schedule.node_child_starts[source_nodes],
                source_nodes,
            )
            counts = jnp.where(
                has_children,
                schedule.node_child_counts[source_nodes],
                1,
            )
            expanded_sources = starts[:, None] + offsets[None, :]
            expanded_targets = jnp.broadcast_to(
                target_nodes[:, None], expanded_sources.shape
            )
            expanded_valid = active[:, None] & (offsets[None, :] < counts[:, None])
            return expanded_sources, expanded_targets, expanded_valid

        for _ in range(self.schedule_plan.plane_count - 1):
            expanded_source, expanded_target, expanded_valid = split_target(
                sources, targets, queue_valid
            )
            sources, targets, queue_valid, required = retain_open(
                expanded_source, expanded_target, expanded_valid
            )
            maximum_required_queue = jnp.maximum(maximum_required_queue, required)
            queue_overflow = queue_overflow | (required > self.queue_capacity)

        for _ in range(self.schedule_plan.plane_count - 1):
            expanded_source, expanded_target, expanded_valid = split_source(
                sources, targets, queue_valid
            )
            sources, targets, queue_valid, required = retain_open(
                expanded_source, expanded_target, expanded_valid
            )
            maximum_required_queue = jnp.maximum(maximum_required_queue, required)
            queue_overflow = queue_overflow | (required > self.queue_capacity)

        all_far_sources = jnp.concatenate(accepted_sources)
        all_far_targets = jnp.concatenate(accepted_targets)
        all_far_valid = jnp.concatenate(accepted_valid)
        all_far_ratios = jnp.concatenate(accepted_ratios)
        required_far = jnp.sum(all_far_valid, dtype=jnp.int32)
        far_selected = jnp.nonzero(all_far_valid, size=self.far_capacity, fill_value=0)[0]
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
        finite = jnp.all(jnp.isfinite(all_far_ratios))
        complete = (
            schedule.evidence.successful
            & ~queue_overflow
            & ~far_overflow
            & ~near_overflow
            & finite
        )
        far_valid = far_valid & complete
        near_valid = near_valid & complete
        far = EdgeRelation(
            jnp.where(far_valid, all_far_sources[far_selected], 0),
            jnp.where(far_valid, all_far_targets[far_selected], 0),
            source_size=self.schedule_plan.node_capacity,
            target_size=self.schedule_plan.node_capacity,
            valid=far_valid,
        )
        near = EdgeRelation(
            jnp.where(near_valid, sources[near_selected], 0),
            jnp.where(near_valid, targets[near_selected], 0),
            source_size=self.schedule_plan.node_capacity,
            target_size=self.schedule_plan.node_capacity,
            valid=near_valid,
        )
        evidence = MortonPlaneInteractionEvidence(
            successful=complete,
            schedule_successful=schedule.evidence.successful,
            required_queue=maximum_required_queue,
            queue_capacity=jnp.asarray(self.queue_capacity, dtype=jnp.int32),
            queue_overflow=queue_overflow,
            required_far=required_far,
            far_capacity=jnp.asarray(self.far_capacity, dtype=jnp.int32),
            far_overflow=far_overflow,
            required_near=required_near,
            near_capacity=jnp.asarray(self.near_capacity, dtype=jnp.int32),
            near_overflow=near_overflow,
            maximum_accepted_ratio=jnp.max(all_far_ratios, initial=0.0),
            finite=finite,
            complete=complete,
        )
        return MortonPlaneInteractionState(far=far, near=near, evidence=evidence)


__all__ = [
    "MortonPlaneInteractionEvidence",
    "MortonPlaneInteractionPlan",
    "MortonPlaneInteractionState",
]
