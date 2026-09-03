#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..spatial import MortonPrimitiveBoundsState
from ._footprint import SurfelFootprintPlan
from ._geometry import SurfelGeometryState


class SurfelRayQueryEvidence(StrictModule):
    node_visits: Array
    primitive_visits: Array
    hit_count: Array
    returned_hits: Array
    hit_overflow: Array
    traversal_complete: Array
    finite: Array
    successful: Array


class SurfelRayQueryResult(StrictModule):
    surfel_slots: Array
    surfel_ids: Array
    distance: Array
    position: Array
    normal: Array
    tangent_coordinates: Array
    front_facing: Array
    valid: Array
    evidence: SurfelRayQueryEvidence
    query_id: str = eqx.field(static=True)


class SurfelRayQueryPlan(StrictModule):
    """Exact bounded surfel intersections with deterministic hit storage."""

    bounds: MortonPrimitiveBoundsState
    geometry: SurfelGeometryState
    footprint_plan: SurfelFootprintPlan
    maximum_hits_per_ray: int = eqx.field(static=True)
    parallel_tolerance: float = eqx.field(static=True)
    near_distance: float = eqx.field(static=True)
    far_distance: float = eqx.field(static=True)
    stack_capacity: int = eqx.field(static=True)
    ray_batch_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bounds: MortonPrimitiveBoundsState,
        geometry: SurfelGeometryState,
        /,
        *,
        maximum_hits_per_ray: int,
        parallel_tolerance: float = 1.0e-12,
        near_distance: float = 0.0,
        far_distance: float = float("inf"),
        ray_batch_size: int = 32,
    ) -> None:
        if not isinstance(bounds, MortonPrimitiveBoundsState):
            raise TypeError("bounds must be MortonPrimitiveBoundsState.")
        if not isinstance(geometry, SurfelGeometryState):
            raise TypeError("geometry must be SurfelGeometryState.")
        if not bool(bounds.evidence.successful) or not bool(geometry.evidence.successful):
            raise ValueError(
                "Surfel ray queries require successful geometry and primitive bounds."
            )
        hits = int(maximum_hits_per_ray)
        batch_size = int(ray_batch_size)
        tolerance = float(parallel_tolerance)
        near = float(near_distance)
        far = float(far_distance)
        if (
            hits < 1
            or batch_size < 1
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or not np.isfinite(near)
            or near < 0.0
            or far <= near
        ):
            raise ValueError("Surfel ray-query limits are invalid.")
        if bounds.item_lower.shape != geometry.position.shape:
            raise ValueError("Surfel bounds and geometry capacities disagree.")
        if not np.array_equal(
            np.sort(
                np.asarray(bounds.hierarchy.sorted_stable_ids)[
                    np.asarray(bounds.hierarchy.sorted_active)
                ]
            ),
            np.sort(
                np.asarray(geometry.discretization.surfel_ids)[
                    np.asarray(geometry.active_mask)
                ]
            ),
        ):
            raise ValueError("Surfel bounds and geometry identities disagree.")
        maximum_depth = int(np.max(np.asarray(bounds.hierarchy.node_levels)))
        branching = int(bounds.hierarchy.node_children.shape[1])
        self.bounds = bounds
        self.geometry = geometry
        self.footprint_plan = SurfelFootprintPlan(
            geometry.ambient_dimension,
            maximum_condition=float(geometry.evidence.maximum_tangent_condition) * 2.0
            + 1.0,
        )
        self.maximum_hits_per_ray = hits
        self.parallel_tolerance = tolerance
        self.near_distance = near
        self.far_distance = far
        self.stack_capacity = 1 + max(branching - 1, 1) * maximum_depth
        self.ray_batch_size = batch_size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "surfel-ray-query-plan",
                "bounds": bounds.bounds_id,
                "geometry": geometry.geometry_id,
                "maximum_hits_per_ray": hits,
                "ray_batch_size": batch_size,
                "parallel_tolerance": tolerance,
                "near_distance": near,
                "far_distance": None if np.isinf(far) else far,
            }
        )

    def query(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        /,
    ) -> SurfelRayQueryResult:
        origin = jnp.asarray(origins, dtype=self.geometry.position.dtype)
        direction = jnp.asarray(directions, dtype=origin.dtype)
        dimension = self.geometry.ambient_dimension
        if origin.ndim != 2 or origin.shape[1] != dimension:
            raise ValueError("origins must have shape (ray_count,ambient_dimension).")
        if direction.shape != origin.shape:
            raise ValueError("directions must match origins.")
        direction_norm = jnp.sqrt(jnp.sum(direction**2, axis=-1))
        finite_ray = (
            jnp.all(jnp.isfinite(origin), axis=-1)
            & jnp.all(jnp.isfinite(direction), axis=-1)
            & (direction_norm > self.parallel_tolerance)
        )
        unit_direction = (
            direction / jnp.where(direction_norm > 0.0, direction_norm, 1.0)[:, None]
        )
        hierarchy = self.bounds.hierarchy
        sorted_logical = hierarchy.storage_to_logical
        maximum_hits = self.maximum_hits_per_ray

        def aabb_hit(ray_origin, ray_direction, lower, upper):
            nonparallel = jnp.abs(ray_direction) > self.parallel_tolerance
            inverse = 1.0 / jnp.where(nonparallel, ray_direction, 1.0)
            first = (lower - ray_origin) * inverse
            second = (upper - ray_origin) * inverse
            axis_enter = jnp.minimum(first, second)
            axis_exit = jnp.maximum(first, second)
            parallel_inside = (ray_origin >= lower) & (ray_origin <= upper)
            axis_enter = jnp.where(nonparallel, axis_enter, -jnp.inf)
            axis_exit = jnp.where(nonparallel, axis_exit, jnp.inf)
            hit = jnp.all(nonparallel | parallel_inside)
            enter = jnp.maximum(jnp.max(axis_enter), self.near_distance)
            exit_value = jnp.minimum(jnp.min(axis_exit), self.far_distance)
            return hit & (exit_value >= enter)

        def select_routes(inputs):
            ray_origin, ray_direction, ray_finite = inputs
            stack = jnp.zeros((self.stack_capacity,), dtype=jnp.int32)
            has_root = ray_finite & (hierarchy.root_slot >= 0)
            stack = stack.at[0].set(jnp.maximum(hierarchy.root_slot, 0))
            initial = (
                stack,
                has_root.astype(jnp.int32),
                jnp.full((maximum_hits,), -1, dtype=jnp.int32),
                jnp.full((maximum_hits,), jnp.inf, dtype=origin.dtype),
                jnp.full(
                    (maximum_hits,),
                    jnp.iinfo(jnp.int64).max,
                    dtype=jnp.int64,
                ),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
            )

            def traversal_condition(state):
                return (state[1] > 0) & ~state[8]

            def traversal_body(state):
                (
                    current_stack,
                    top,
                    hit_slots,
                    hit_distances,
                    hit_ids,
                    total_hits,
                    node_visits,
                    primitive_visits,
                    stack_overflow,
                ) = state
                next_top = top - 1
                node = current_stack[next_top]
                node_hit = self.bounds.node_bounded[node] & aabb_hit(
                    ray_origin,
                    ray_direction,
                    self.bounds.node_lower[node],
                    self.bounds.node_upper[node],
                )

                def process_leaf(leaf_state):
                    slots, distances, ids, hit_count, primitive_count = leaf_state
                    start = hierarchy.node_item_starts[node]
                    count = hierarchy.node_item_counts[node]

                    def item_condition(item_state):
                        return item_state[0] < count

                    def item_body(item_state):
                        offset, current_slots, current_distances, current_ids, found = (
                            item_state
                        )
                        storage_slot = start + offset
                        surfel_slot = sorted_logical[storage_slot]
                        surfel_active = self.geometry.active_mask[surfel_slot]
                        surfel_normal = self.geometry.normal[surfel_slot]
                        denominator = jnp.sum(surfel_normal * ray_direction)
                        numerator = jnp.sum(
                            surfel_normal
                            * (self.geometry.position[surfel_slot] - ray_origin)
                        )
                        distance = numerator / jnp.where(
                            jnp.abs(denominator) > self.parallel_tolerance,
                            denominator,
                            1.0,
                        )
                        candidate_point = ray_origin + distance * ray_direction
                        footprint = self.footprint_plan.evaluate(
                            self.geometry,
                            candidate_point[None, :],
                            jnp.asarray([surfel_slot], dtype=jnp.int32),
                        )
                        sided = (
                            denominator < -self.parallel_tolerance
                            if self.geometry.certificate.one_sided
                            else jnp.abs(denominator) > self.parallel_tolerance
                        )
                        valid_hit = (
                            surfel_active
                            & sided
                            & (distance >= self.near_distance)
                            & (distance <= self.far_distance)
                            & footprint.inside[0]
                        )
                        identifier = self.geometry.discretization.surfel_ids[surfel_slot]
                        available = found < maximum_hits
                        insert_slot = jnp.minimum(found, maximum_hits - 1)
                        worst_slot = jnp.argmax(current_distances)
                        better_than_worst = (distance < current_distances[worst_slot]) | (
                            (distance == current_distances[worst_slot])
                            & (identifier < current_ids[worst_slot])
                        )
                        replace = valid_hit & (~available) & better_than_worst
                        write = valid_hit & (available | replace)
                        destination = jnp.where(available, insert_slot, worst_slot)
                        current_slots = current_slots.at[destination].set(
                            jnp.where(
                                write,
                                surfel_slot,
                                current_slots[destination],
                            )
                        )
                        current_distances = current_distances.at[destination].set(
                            jnp.where(
                                write,
                                distance,
                                current_distances[destination],
                            )
                        )
                        current_ids = current_ids.at[destination].set(
                            jnp.where(
                                write,
                                identifier,
                                current_ids[destination],
                            )
                        )
                        return (
                            offset + 1,
                            current_slots,
                            current_distances,
                            current_ids,
                            found + valid_hit.astype(jnp.int32),
                        )

                    _, slots, distances, ids, found = jax.lax.while_loop(
                        item_condition,
                        item_body,
                        (
                            jnp.asarray(0, dtype=jnp.int32),
                            slots,
                            distances,
                            ids,
                            hit_count,
                        ),
                    )
                    return (
                        slots,
                        distances,
                        ids,
                        found,
                        primitive_count + count,
                    )

                leaf_state = (
                    hit_slots,
                    hit_distances,
                    hit_ids,
                    total_hits,
                    primitive_visits,
                )
                (
                    hit_slots,
                    hit_distances,
                    hit_ids,
                    total_hits,
                    primitive_visits,
                ) = jax.lax.cond(
                    node_hit & hierarchy.node_is_leaf[node],
                    process_leaf,
                    lambda value: value,
                    leaf_state,
                )
                descend = node_hit & ~hierarchy.node_is_leaf[node]
                children = hierarchy.node_children[node]
                for child_index in range(children.shape[0]):
                    child = children[child_index]
                    push = descend & (child >= 0)
                    has_capacity = next_top < self.stack_capacity
                    write = push & has_capacity
                    safe_top = jnp.minimum(next_top, self.stack_capacity - 1)
                    current_stack = current_stack.at[safe_top].set(
                        jnp.where(write, child, current_stack[safe_top])
                    )
                    next_top = next_top + write.astype(jnp.int32)
                    stack_overflow = stack_overflow | (push & ~has_capacity)
                return (
                    current_stack,
                    next_top,
                    hit_slots,
                    hit_distances,
                    hit_ids,
                    total_hits,
                    node_visits + hierarchy.node_active[node].astype(jnp.int32),
                    primitive_visits,
                    stack_overflow,
                )

            final = jax.lax.while_loop(
                traversal_condition,
                traversal_body,
                initial,
            )
            (
                _,
                remaining,
                hit_slots,
                hit_distances,
                hit_ids,
                total_hits,
                node_visits,
                primitive_visits,
                stack_overflow,
            ) = final
            valid = jnp.arange(maximum_hits, dtype=jnp.int32) < jnp.minimum(
                total_hits, maximum_hits
            )
            order = jnp.lexsort(
                (
                    hit_ids,
                    hit_distances,
                    (~valid).astype(jnp.int32),
                )
            ).astype(jnp.int32)
            return (
                hit_slots[order],
                valid[order],
                total_hits,
                node_visits,
                primitive_visits,
                stack_overflow,
                remaining == 0,
            )

        def select_branchless(inputs):
            ray_origin, ray_direction, ray_finite = inputs
            surfel_slots = jnp.arange(self.geometry.capacity, dtype=jnp.int32)
            normal = self.geometry.normal
            denominator = jnp.sum(normal * ray_direction[None, :], axis=-1)
            numerator = jnp.sum(
                normal * (self.geometry.position - ray_origin[None, :]),
                axis=-1,
            )
            distance = numerator / jnp.where(
                jnp.abs(denominator) > self.parallel_tolerance,
                denominator,
                1.0,
            )
            candidate_points = (
                ray_origin[None, :] + distance[:, None] * ray_direction[None, :]
            )
            footprint = self.footprint_plan.evaluate(
                self.geometry,
                candidate_points,
                surfel_slots,
            )
            sided = (
                denominator < -self.parallel_tolerance
                if self.geometry.certificate.one_sided
                else jnp.abs(denominator) > self.parallel_tolerance
            )
            valid = (
                ray_finite
                & self.geometry.active_mask
                & sided
                & (distance >= self.near_distance)
                & (distance <= self.far_distance)
                & footprint.inside
            )
            total_hits = jnp.sum(valid, dtype=jnp.int32)
            order = jnp.lexsort(
                (
                    self.geometry.discretization.surfel_ids,
                    distance,
                    (~valid).astype(jnp.int32),
                )
            ).astype(jnp.int32)
            padded_order = jnp.pad(
                order,
                (0, max(maximum_hits - self.geometry.capacity, 0)),
            )
            selected = padded_order[:maximum_hits]
            selected_valid = (
                jnp.arange(maximum_hits, dtype=jnp.int32) < self.geometry.capacity
            ) & valid[selected]
            return (
                selected,
                selected_valid,
                total_hits,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.sum(self.geometry.active_mask, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(True),
            )

        if self.geometry.capacity <= 4096:
            (
                selected_slots,
                selected_valid,
                hit_count,
                node_visits,
                primitive_visits,
                stack_overflow,
                traversal_empty,
            ) = jax.lax.map(
                select_branchless,
                (origin, unit_direction, finite_ray),
                batch_size=min(self.ray_batch_size, origin.shape[0]),
            )
        else:
            (
                selected_slots,
                selected_valid,
                hit_count,
                node_visits,
                primitive_visits,
                stack_overflow,
                traversal_empty,
            ) = jax.vmap(select_routes)((origin, unit_direction, finite_ray))
        selected_slots = jax.lax.stop_gradient(selected_slots)
        selected_valid = jax.lax.stop_gradient(selected_valid)
        safe_slots = jnp.maximum(selected_slots, 0)
        selected_position = self.geometry.position[safe_slots]
        selected_normal = self.geometry.normal[safe_slots]
        denominator = jnp.sum(selected_normal * unit_direction[:, None, :], axis=-1)
        numerator = jnp.sum(
            selected_normal * (selected_position - origin[:, None, :]),
            axis=-1,
        )
        distance = numerator / jnp.where(
            jnp.abs(denominator) > self.parallel_tolerance,
            denominator,
            1.0,
        )
        hit_position = (
            origin[:, None, :] + distance[..., None] * unit_direction[:, None, :]
        )
        footprint = self.footprint_plan.evaluate(
            self.geometry,
            hit_position.reshape((-1, dimension)),
            safe_slots.reshape((-1,)),
        )
        tangent_coordinates = footprint.tangent_coordinates.reshape(
            (origin.shape[0], maximum_hits, dimension - 1)
        )
        footprint_successful = footprint.successful.reshape(
            (origin.shape[0], maximum_hits)
        )
        finite_hit = (
            jnp.isfinite(distance)
            & jnp.all(jnp.isfinite(hit_position), axis=-1)
            & jnp.all(jnp.isfinite(tangent_coordinates), axis=-1)
        )
        valid = selected_valid & finite_hit & footprint_successful
        hit_overflow = hit_count > maximum_hits
        traversal_complete = traversal_empty & ~stack_overflow
        successful_ray = (
            finite_ray
            & traversal_complete
            & ~hit_overflow
            & self.bounds.evidence.successful
            & self.geometry.evidence.successful
        )
        evidence = SurfelRayQueryEvidence(
            node_visits=node_visits,
            primitive_visits=primitive_visits,
            hit_count=hit_count,
            returned_hits=jnp.sum(valid, axis=1, dtype=jnp.int32),
            hit_overflow=hit_overflow,
            traversal_complete=traversal_complete,
            finite=finite_ray
            & jnp.all(
                ~selected_valid | (finite_hit & footprint_successful),
                axis=1,
            ),
            successful=successful_ray,
        )
        return SurfelRayQueryResult(
            surfel_slots=jnp.where(valid, selected_slots, -1),
            surfel_ids=jnp.where(
                valid,
                self.geometry.discretization.surfel_ids[safe_slots],
                -1,
            ),
            distance=jnp.where(valid, distance, jnp.inf),
            position=jnp.where(valid[..., None], hit_position, 0.0),
            normal=jnp.where(valid[..., None], selected_normal, 0.0),
            tangent_coordinates=jnp.where(valid[..., None], tangent_coordinates, 0.0),
            front_facing=valid & (denominator < 0.0),
            valid=valid,
            evidence=evidence,
            query_id=canonical_fingerprint(
                {"kind": "surfel-ray-query", "plan": self.plan_id}
            ),
        )


__all__ = [
    "SurfelRayQueryEvidence",
    "SurfelRayQueryPlan",
    "SurfelRayQueryResult",
]
