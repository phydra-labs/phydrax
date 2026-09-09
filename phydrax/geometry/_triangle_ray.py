#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._bvh import build_packed_bvh, refit_packed_bvh_bounds
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._ray_intersection import RayIntersectionResult, RayIntersectionStatus


class TriangleRayIntersectionStatus(IntEnum):
    """Status of a conservative nearest oriented-triangle query."""

    SUCCESS = 0
    MISS = 1
    NONFINITE_INPUT = 2
    DEGENERATE_DIRECTION = 3
    AMBIGUOUS_HIT = 4
    TRAVERSAL_CAPACITY_EXHAUSTED = 5
    INVALID_GEOMETRY = 6


class TriangleRayQueryPlan(StrictModule, NonTrainableState):
    """Immutable oriented triangle query definition.

    ``entity_ids`` identify triangles belonging to the same physical interface.
    A nearest-distance tie is valid only when every tied triangle has the same
    entity ID. This permits a consistently labelled triangulated face to share
    edges without hiding coincident interfaces.
    """

    vertices: Array
    triangles: Array
    entity_ids: Array
    leaf_size: int = eqx.field(static=True)
    traversal_stack_capacity: int = eqx.field(static=True)
    acceleration: Literal["bvh", "exhaustive"] = eqx.field(static=True)
    determinant_tolerance: float = eqx.field(static=True)
    barycentric_tolerance: float = eqx.field(static=True)
    forward_tolerance: float = eqx.field(static=True)
    tie_tolerance: float = eqx.field(static=True)

    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        triangles: ArrayLike,
        /,
        *,
        entity_ids: ArrayLike | None = None,
        leaf_size: int = 8,
        traversal_stack_capacity: int = 64,
        acceleration: Literal["bvh", "exhaustive"] = "bvh",
        determinant_tolerance: float = 1e-12,
        barycentric_tolerance: float = 1e-10,
        forward_tolerance: float = 1e-9,
        tie_tolerance: float = 1e-9,
    ):
        vertices_host = np.asarray(vertices)
        triangles_host = np.asarray(triangles)
        if vertices_host.ndim != 2 or vertices_host.shape[1:] != (3,):
            raise ValueError("vertices must have shape (n_vertices, 3).")
        if triangles_host.ndim != 2 or triangles_host.shape[1:] != (3,):
            raise ValueError("triangles must have shape (n_triangles, 3).")
        if vertices_host.shape[0] < 3 or triangles_host.shape[0] < 1:
            raise ValueError("An oriented triangle query requires at least one triangle.")
        if not np.issubdtype(vertices_host.dtype, np.number) or np.iscomplexobj(
            vertices_host
        ):
            raise TypeError("vertices must be real-valued.")
        if not np.all(np.isfinite(vertices_host)):
            raise ValueError("vertices must be finite.")
        if not np.issubdtype(triangles_host.dtype, np.integer):
            raise TypeError("triangles must contain integer vertex indices.")
        if np.any(triangles_host < 0) or np.any(triangles_host >= vertices_host.shape[0]):
            raise ValueError("triangles contains an out-of-range vertex index.")
        triangle_vertices = vertices_host[triangles_host]
        area_vectors = np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        if np.any(np.linalg.norm(area_vectors, axis=-1) == 0.0):
            raise ValueError("triangles must be nondegenerate and oriented.")
        count = triangles_host.shape[0]
        if entity_ids is None:
            entity_host = np.arange(count, dtype=np.int32)
        else:
            entity_host = np.asarray(entity_ids)
            if entity_host.shape != (count,) or not np.issubdtype(
                entity_host.dtype, np.integer
            ):
                raise ValueError("entity_ids must be an integer (n_triangles,) array.")
            if np.any(entity_host < 0):
                raise ValueError("entity_ids must be non-negative.")
            entity_host = entity_host.astype(np.int32, copy=False)
        leaf_size_ = int(leaf_size)
        stack_capacity_ = int(traversal_stack_capacity)
        if leaf_size_ < 1 or stack_capacity_ < 1:
            raise ValueError("leaf_size and traversal_stack_capacity must be positive.")
        if acceleration not in ("bvh", "exhaustive"):
            raise ValueError("acceleration must be 'bvh' or 'exhaustive'.")
        tolerances = (
            determinant_tolerance,
            barycentric_tolerance,
            forward_tolerance,
            tie_tolerance,
        )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0 for value in tolerances
        ):
            raise ValueError("Triangle query tolerances must be finite and non-negative.")
        if determinant_tolerance == 0.0:
            raise ValueError("determinant_tolerance must be positive.")

        dtype = jnp.result_type(vertices_host, 0.0)
        self.vertices = jnp.asarray(vertices_host, dtype=dtype)
        self.triangles = jnp.asarray(triangles_host, dtype=jnp.int32)
        self.entity_ids = jnp.asarray(entity_host, dtype=jnp.int32)
        self.leaf_size = leaf_size_
        self.traversal_stack_capacity = stack_capacity_
        self.acceleration = acceleration
        self.determinant_tolerance = float(determinant_tolerance)
        self.barycentric_tolerance = float(barycentric_tolerance)
        self.forward_tolerance = float(forward_tolerance)
        self.tie_tolerance = float(tie_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "triangle-ray-query-plan",
                "vertices": array_tree_fingerprint(vertices_host),
                "triangles": array_tree_fingerprint(triangles_host),
                "entities": array_tree_fingerprint(entity_host),
                "leaf_size": leaf_size_,
                "stack_capacity": stack_capacity_,
                "acceleration": acceleration,
                "tolerances": [float(value) for value in tolerances],
            }
        )


class TriangleRayGeometryState(StrictModule):
    """Differentiable triangle geometry and conservatively refitted bounds."""

    triangle_vertices: Array
    edge_one: Array
    edge_two: Array
    normals: Array
    bbox_min: Array
    bbox_max: Array
    finite: Array
    nondegenerate: Array
    geometry_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.finite & self.nondegenerate


class PreparedTriangleRayQuery(StrictModule, NonTrainableState):
    """Fixed triangle topology, entity labels, and conservative BVH routes."""

    triangles: Array
    entity_ids: Array
    left: Array
    right: Array
    leaf_id: Array
    leaf_items: Array
    leaf_node: Array
    reference_geometry: TriangleRayGeometryState
    leaf_size: int = eqx.field(static=True)
    traversal_stack_capacity: int = eqx.field(static=True)
    acceleration: Literal["bvh", "exhaustive"] = eqx.field(static=True)
    determinant_tolerance: float = eqx.field(static=True)
    barycentric_tolerance: float = eqx.field(static=True)
    forward_tolerance: float = eqx.field(static=True)
    tie_tolerance: float = eqx.field(static=True)
    triangle_count: int = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    required_stack_capacity: int = eqx.field(static=True)
    exact_when_successful: bool = eqx.field(static=True)
    exhaustive_reference: bool = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class TriangleRayIntersectionResult(StrictModule, NonTrainableState):
    """Nearest triangle hit plus exactness, tie, and traversal evidence."""

    intersection: RayIntersectionResult
    triangle_indices: Array
    entity_ids: Array
    barycentric_coordinates: Array
    oriented_normals: Array
    front_facing: Array
    tie_count: Array
    uniqueness_margin: Array
    traversal_steps: Array
    triangle_tests: Array
    status: Array
    successful: Array


def prepare_triangle_ray_query(
    plan: TriangleRayQueryPlan,
    /,
) -> PreparedTriangleRayQuery:
    """Prepare fixed topology and its initial exact geometry state."""
    if not isinstance(plan, TriangleRayQueryPlan):
        raise TypeError("plan must be TriangleRayQueryPlan.")
    vertices_host = np.asarray(plan.vertices)
    triangles_host = np.asarray(plan.triangles)
    triangle_vertices = vertices_host[triangles_host]
    edge_one = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge_two = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    area_vectors = np.cross(edge_one, edge_two)
    normals = area_vectors / np.linalg.norm(area_vectors, axis=-1, keepdims=True)
    bbox_min = np.minimum.reduce(triangle_vertices, axis=1)
    bbox_max = np.maximum.reduce(triangle_vertices, axis=1)
    scale = np.maximum(1.0, np.max(np.abs(triangle_vertices), axis=(1, 2)))
    padding = np.finfo(vertices_host.dtype).eps * 16.0 * scale
    bbox_min = bbox_min - padding[:, None]
    bbox_max = bbox_max + padding[:, None]
    bvh = build_packed_bvh(
        bbox_min,
        bbox_max,
        np.mean(triangle_vertices, axis=1),
        leaf_size=plan.leaf_size,
        dtype=plan.vertices.dtype,
    )
    geometry = TriangleRayGeometryState(
        jnp.asarray(triangle_vertices, dtype=plan.vertices.dtype),
        jnp.asarray(edge_one, dtype=plan.vertices.dtype),
        jnp.asarray(edge_two, dtype=plan.vertices.dtype),
        jnp.asarray(normals, dtype=plan.vertices.dtype),
        bvh.bbox_min,
        bvh.bbox_max,
        jnp.asarray(True),
        jnp.asarray(True),
        f"{plan.plan_id}:reference",
    )
    arrays = (
        np.asarray(plan.triangles),
        np.asarray(plan.entity_ids),
        np.asarray(bvh.left),
        np.asarray(bvh.right),
        np.asarray(bvh.leaf_id),
        np.asarray(bvh.leaf_items),
        np.asarray(bvh.leaf_node),
        triangle_vertices,
        edge_one,
        edge_two,
        normals,
        np.asarray(bvh.bbox_min),
        np.asarray(bvh.bbox_max),
    )
    storage_bytes = sum(value.nbytes for value in arrays)
    required_stack_capacity = int(bvh.max_depth) + 1
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-triangle-ray-query",
            "plan": plan.plan_id,
            "node_count": int(bvh.left.shape[0]),
            "required_stack_capacity": required_stack_capacity,
        }
    )
    return PreparedTriangleRayQuery(
        plan.triangles,
        plan.entity_ids,
        bvh.left,
        bvh.right,
        bvh.leaf_id,
        bvh.leaf_items,
        bvh.leaf_node,
        geometry,
        plan.leaf_size,
        plan.traversal_stack_capacity,
        plan.acceleration,
        plan.determinant_tolerance,
        plan.barycentric_tolerance,
        plan.forward_tolerance,
        plan.tie_tolerance,
        int(triangle_vertices.shape[0]),
        int(vertices_host.shape[0]),
        int(bvh.left.shape[0]),
        required_stack_capacity,
        True,
        plan.acceleration == "exhaustive",
        int(storage_bytes),
        prepared_id,
    )


def refit_triangle_ray_geometry(
    prepared: PreparedTriangleRayQuery,
    vertices: ArrayLike,
    /,
    *,
    geometry_id: str,
) -> TriangleRayGeometryState:
    """Refit one fixed triangle topology to differentiable current vertices."""
    if not isinstance(prepared, PreparedTriangleRayQuery):
        raise TypeError("prepared must be PreparedTriangleRayQuery.")
    vertices_ = jnp.asarray(
        vertices, dtype=prepared.reference_geometry.triangle_vertices.dtype
    )
    if vertices_.shape != (prepared.vertex_count, 3):
        raise ValueError(f"vertices must have shape ({prepared.vertex_count}, 3).")
    if (
        not isinstance(geometry_id, str)
        or not geometry_id
        or geometry_id.strip() != geometry_id
    ):
        raise ValueError("geometry_id must be non-empty canonical text.")
    finite = jnp.all(jnp.isfinite(vertices_))
    safe_vertices = jnp.where(jnp.isfinite(vertices_), vertices_, 0.0)
    triangle_vertices = safe_vertices[prepared.triangles]
    edge_one = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge_two = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    area_vectors = jnp.cross(edge_one, edge_two)
    area_norms = jnp.sqrt(jnp.sum(area_vectors * area_vectors, axis=-1))
    nondegenerate_triangles = area_norms > jnp.finfo(vertices_.dtype).eps
    normals = area_vectors / jnp.where(
        nondegenerate_triangles[:, None], area_norms[:, None], 1.0
    )
    item_min = jnp.min(triangle_vertices, axis=1)
    item_max = jnp.max(triangle_vertices, axis=1)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(triangle_vertices), axis=(1, 2)))
    padding = jnp.finfo(vertices_.dtype).eps * 16.0 * scale
    item_min = item_min - padding[:, None]
    item_max = item_max + padding[:, None]
    bbox_min, bbox_max, _, _ = refit_packed_bvh_bounds(
        item_min,
        item_max,
        left=prepared.left,
        right=prepared.right,
        leaf_id=prepared.leaf_id,
        leaf_items=prepared.leaf_items,
        leaf_node=prepared.leaf_node,
    )
    return TriangleRayGeometryState(
        triangle_vertices,
        edge_one,
        edge_two,
        normals,
        bbox_min,
        bbox_max,
        finite,
        jnp.all(nondegenerate_triangles),
        geometry_id,
    )


def _intersect_triangles(
    prepared: PreparedTriangleRayQuery,
    geometry: TriangleRayGeometryState,
    origin: Array,
    direction: Array,
    triangle_indices: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    valid_index = triangle_indices >= 0
    safe_indices = jnp.where(valid_index, triangle_indices, 0)
    vertices = geometry.triangle_vertices[safe_indices]
    edge_one = geometry.edge_one[safe_indices]
    edge_two = geometry.edge_two[safe_indices]
    relative = origin - vertices[:, 0]
    p = jnp.cross(direction[None, :], edge_two)
    determinant = jnp.sum(edge_one * p, axis=-1)
    determinant_scale = jnp.sqrt(
        jnp.sum(edge_one * edge_one, axis=-1) * jnp.sum(edge_two * edge_two, axis=-1)
    )
    determinant_ok = (
        jnp.abs(determinant) > prepared.determinant_tolerance * determinant_scale
    )
    safe_determinant = jnp.where(determinant_ok, determinant, 1.0)
    u = jnp.sum(relative * p, axis=-1) / safe_determinant
    q = jnp.cross(relative, edge_one)
    v = jnp.sum(direction[None, :] * q, axis=-1) / safe_determinant
    distance = jnp.sum(edge_two * q, axis=-1) / safe_determinant
    tolerance = prepared.barycentric_tolerance
    hit = (
        valid_index
        & determinant_ok
        & (u >= -tolerance)
        & (v >= -tolerance)
        & (u + v <= 1.0 + tolerance)
        & (distance > prepared.forward_tolerance)
        & jnp.isfinite(distance)
    )
    barycentric = jnp.stack((1.0 - u - v, u, v), axis=-1)
    entities = prepared.entity_ids[safe_indices]
    return hit, distance, barycentric, safe_indices, entities


def _ray_box_hit(
    origin: Array,
    direction: Array,
    bbox_min: Array,
    bbox_max: Array,
    maximum_distance: Array,
    tolerance: float,
) -> tuple[Array, Array]:
    nonzero = direction != 0.0
    inverse = jnp.where(nonzero, 1.0 / direction, 0.0)
    lower = (bbox_min - origin) * inverse
    upper = (bbox_max - origin) * inverse
    slab_near = jnp.minimum(lower, upper)
    slab_far = jnp.maximum(lower, upper)
    inside_parallel = (origin >= bbox_min) & (origin <= bbox_max)
    slab_near = jnp.where(nonzero, slab_near, -jnp.inf)
    slab_far = jnp.where(nonzero, slab_far, jnp.inf)
    parallel_ok = jnp.all(nonzero | inside_parallel)
    near = jnp.maximum(jnp.max(slab_near), 0.0)
    far = jnp.min(slab_far)
    distance_tolerance = tolerance * jnp.maximum(1.0, jnp.abs(maximum_distance))
    hit = parallel_ok & (far >= near) & (near <= maximum_distance + distance_tolerance)
    return hit, near


def _empty_candidate(dtype: jnp.dtype) -> tuple[Array, ...]:
    return (
        jnp.asarray(jnp.inf, dtype=dtype),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.zeros((3,), dtype=dtype),
        jnp.zeros((3,), dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(jnp.inf, dtype=dtype),
    )


def _merge_for_ray(
    prepared: PreparedTriangleRayQuery,
    geometry: TriangleRayGeometryState,
    origin: Array,
    direction: Array,
    candidate: tuple[Array, ...],
    indices: Array,
) -> tuple[tuple[Array, ...], Array]:
    hit, distances, barycentric, triangle_indices, entities = _intersect_triangles(
        prepared, geometry, origin, direction, indices
    )
    (
        best_distance,
        best_triangle,
        best_entity,
        best_barycentric,
        best_normal,
        tie_count,
        ambiguous,
        second_distance,
    ) = candidate
    inf = jnp.asarray(jnp.inf, dtype=distances.dtype)
    nearest = jnp.min(jnp.where(hit, distances, inf))
    has_hit = jnp.isfinite(nearest)
    local_scale = jnp.maximum(1.0, jnp.abs(nearest))
    local_tied = hit & (
        jnp.abs(distances - nearest) <= prepared.tie_tolerance * local_scale
    )
    local_second = jnp.min(jnp.where(hit & ~local_tied, distances, inf))
    sentinel = jnp.asarray(prepared.triangle_count, dtype=jnp.int32)
    winner_triangle = jnp.min(jnp.where(local_tied, triangle_indices, sentinel))
    winner_triangle = jnp.where(has_hit, winner_triangle, -1)
    winner_position = jnp.argmin(
        jnp.where(
            local_tied & (triangle_indices == winner_triangle), triangle_indices, sentinel
        )
    )
    winner_entity = jnp.where(has_hit, entities[winner_position], -1)
    winner_barycentric = barycentric[winner_position]
    winner_normal = geometry.normals[jnp.maximum(winner_triangle, 0)]
    local_count = jnp.sum(local_tied, dtype=jnp.int32)
    local_ambiguous = has_hit & jnp.any(local_tied & (entities != winner_entity))
    comparison_scale = jnp.maximum(
        1.0, jnp.maximum(jnp.abs(best_distance), jnp.abs(nearest))
    )
    close = (
        has_hit
        & jnp.isfinite(best_distance)
        & (jnp.abs(nearest - best_distance) <= prepared.tie_tolerance * comparison_scale)
    )
    better = has_hit & ~close & (nearest < best_distance)
    use_local = better | (close & (winner_triangle < best_triangle))
    merged = (
        jnp.where(use_local, nearest, best_distance),
        jnp.where(use_local, winner_triangle, best_triangle),
        jnp.where(use_local, winner_entity, best_entity),
        jnp.where(use_local, winner_barycentric, best_barycentric),
        jnp.where(use_local, winner_normal, best_normal),
        jnp.where(
            better, local_count, jnp.where(close, tie_count + local_count, tie_count)
        ),
        jnp.where(
            better,
            local_ambiguous,
            jnp.where(
                close,
                ambiguous | local_ambiguous | (winner_entity != best_entity),
                ambiguous,
            ),
        ),
        jnp.where(
            better,
            jnp.minimum(jnp.minimum(best_distance, second_distance), local_second),
            jnp.where(
                close,
                jnp.minimum(second_distance, local_second),
                jnp.where(
                    has_hit,
                    jnp.minimum(jnp.minimum(second_distance, nearest), local_second),
                    second_distance,
                ),
            ),
        ),
    )
    return merged, jnp.sum(indices >= 0, dtype=jnp.int32)


def _query_exhaustive_one(
    prepared: PreparedTriangleRayQuery,
    geometry: TriangleRayGeometryState,
    origin: Array,
    direction: Array,
) -> tuple[tuple[Array, ...], Array, Array, Array]:
    candidate, tests = _merge_for_ray(
        prepared,
        geometry,
        origin,
        direction,
        _empty_candidate(origin.dtype),
        jnp.arange(prepared.triangle_count, dtype=jnp.int32),
    )
    return (
        candidate,
        jnp.asarray(1, dtype=jnp.int32),
        tests,
        jnp.asarray(False),
    )


def _query_bvh_one(
    prepared: PreparedTriangleRayQuery,
    geometry: TriangleRayGeometryState,
    origin: Array,
    direction: Array,
) -> tuple[tuple[Array, ...], Array, Array, Array]:
    stack = jnp.full((prepared.traversal_stack_capacity,), -1, dtype=jnp.int32)
    stack = stack.at[0].set(0)
    initial = (
        stack,
        jnp.asarray(1, dtype=jnp.int32),
        _empty_candidate(origin.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    )

    def body(_, state):
        stack_, size, candidate, steps, tests, exhausted = state

        def visit(active_state):
            stack__, size_, candidate_, steps_, tests_, exhausted_ = active_state
            next_size = size_ - 1
            node = stack__[next_size]
            stack__ = stack__.at[next_size].set(-1)
            maximum = candidate_[0]
            node_hit, _ = _ray_box_hit(
                origin,
                direction,
                geometry.bbox_min[node],
                geometry.bbox_max[node],
                maximum,
                prepared.tie_tolerance,
            )

            def inspect_node(node_state):
                stack___, size__, candidate__, tests__, exhausted__ = node_state
                leaf = prepared.leaf_id[node]

                def inspect_leaf(leaf_state):
                    stack____, size___, candidate___, tests___, exhausted___ = leaf_state
                    merged, tested = _merge_for_ray(
                        prepared,
                        geometry,
                        origin,
                        direction,
                        candidate___,
                        prepared.leaf_items[leaf],
                    )
                    return stack____, size___, merged, tests___ + tested, exhausted___

                def inspect_branch(branch_state):
                    stack____, size___, candidate___, tests___, exhausted___ = (
                        branch_state
                    )
                    left = prepared.left[node]
                    right = prepared.right[node]
                    left_hit, left_near = _ray_box_hit(
                        origin,
                        direction,
                        geometry.bbox_min[left],
                        geometry.bbox_max[left],
                        candidate___[0],
                        prepared.tie_tolerance,
                    )
                    right_hit, right_near = _ray_box_hit(
                        origin,
                        direction,
                        geometry.bbox_min[right],
                        geometry.bbox_max[right],
                        candidate___[0],
                        prepared.tie_tolerance,
                    )
                    count = left_hit.astype(jnp.int32) + right_hit.astype(jnp.int32)
                    room = size___ + count <= prepared.traversal_stack_capacity
                    exhausted___ = exhausted___ | (~room & (count > 0))
                    near_node = jnp.where(left_near <= right_near, left, right)
                    far_node = jnp.where(left_near <= right_near, right, left)
                    near_hit = jnp.where(left_near <= right_near, left_hit, right_hit)
                    far_hit = jnp.where(left_near <= right_near, right_hit, left_hit)
                    write_far = room & far_hit
                    stack____ = stack____.at[
                        jnp.minimum(size___, prepared.traversal_stack_capacity - 1)
                    ].set(
                        jnp.where(
                            write_far,
                            far_node,
                            stack____[
                                jnp.minimum(
                                    size___, prepared.traversal_stack_capacity - 1
                                )
                            ],
                        )
                    )
                    size_after_far = size___ + write_far.astype(jnp.int32)
                    write_near = room & near_hit
                    near_position = jnp.minimum(
                        size_after_far, prepared.traversal_stack_capacity - 1
                    )
                    stack____ = stack____.at[near_position].set(
                        jnp.where(write_near, near_node, stack____[near_position])
                    )
                    return (
                        stack____,
                        size_after_far + write_near.astype(jnp.int32),
                        candidate___,
                        tests___,
                        exhausted___,
                    )

                return jax.lax.cond(
                    leaf >= 0,
                    inspect_leaf,
                    inspect_branch,
                    (stack___, size__, candidate__, tests__, exhausted__),
                )

            stack__, next_size, candidate_, tests_, exhausted_ = jax.lax.cond(
                node_hit,
                inspect_node,
                lambda value: value,
                (stack__, next_size, candidate_, tests_, exhausted_),
            )
            return stack__, next_size, candidate_, steps_ + 1, tests_, exhausted_

        return jax.lax.cond(size > 0, visit, lambda value: value, state)

    stack, size, candidate, steps, tests, exhausted = jax.lax.fori_loop(
        0, prepared.node_count, body, initial
    )
    exhausted = exhausted | (size > 0)
    return candidate, steps, tests, exhausted


def _query_one(
    prepared: PreparedTriangleRayQuery,
    geometry: TriangleRayGeometryState,
    origin: Array,
    direction: Array,
) -> tuple[Array, ...]:
    finite = jnp.all(jnp.isfinite(origin)) & jnp.all(jnp.isfinite(direction))
    safe_direction = jnp.where(finite, direction, 0.0)
    norm = jnp.sqrt(jnp.sum(safe_direction * safe_direction))
    direction_ok = norm > 0.0
    unit_direction = safe_direction / jnp.where(direction_ok, norm, 1.0)
    if prepared.acceleration == "exhaustive":
        candidate, steps, tests, exhausted = _query_exhaustive_one(
            prepared, geometry, origin, unit_direction
        )
    else:
        candidate, steps, tests, exhausted = _query_bvh_one(
            prepared, geometry, origin, unit_direction
        )
    (
        distance,
        triangle,
        entity,
        barycentric,
        normal,
        tie_count,
        ambiguous,
        second_distance,
    ) = candidate
    hit = jnp.isfinite(distance)
    status = jnp.where(
        ~geometry.successful,
        int(TriangleRayIntersectionStatus.INVALID_GEOMETRY),
        jnp.where(
            ~finite,
            int(TriangleRayIntersectionStatus.NONFINITE_INPUT),
            jnp.where(
                ~direction_ok,
                int(TriangleRayIntersectionStatus.DEGENERATE_DIRECTION),
                jnp.where(
                    exhausted,
                    int(TriangleRayIntersectionStatus.TRAVERSAL_CAPACITY_EXHAUSTED),
                    jnp.where(
                        ambiguous,
                        int(TriangleRayIntersectionStatus.AMBIGUOUS_HIT),
                        jnp.where(
                            hit,
                            int(TriangleRayIntersectionStatus.SUCCESS),
                            int(TriangleRayIntersectionStatus.MISS),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    successful = status == int(TriangleRayIntersectionStatus.SUCCESS)
    safe_distance = jnp.where(successful, distance, 0.0)
    point = jnp.where(successful, origin + safe_distance * unit_direction, origin)
    safe_triangle = jnp.where(successful, triangle, -1).astype(jnp.int32)
    safe_entity = jnp.where(successful, entity, -1).astype(jnp.int32)
    safe_barycentric = jnp.where(successful, barycentric, 0.0)
    safe_normal = jnp.where(successful, normal, 0.0)
    front_facing = successful & (jnp.sum(unit_direction * normal) < 0.0)
    denominator_margin = jnp.where(
        successful,
        jnp.abs(jnp.sum(unit_direction * normal)) - prepared.determinant_tolerance,
        0.0,
    )
    uniqueness_margin = jnp.where(
        successful & jnp.isfinite(second_distance),
        second_distance - distance,
        jnp.where(successful, jnp.inf, 0.0),
    )
    base_status = jnp.where(
        ~finite,
        int(RayIntersectionStatus.NONFINITE_INPUT),
        jnp.where(
            ~direction_ok,
            int(RayIntersectionStatus.DEGENERATE_DIRECTION),
            jnp.where(
                successful,
                int(RayIntersectionStatus.SUCCESS),
                int(RayIntersectionStatus.PARALLEL),
            ),
        ),
    ).astype(jnp.int32)
    return (
        point,
        safe_distance,
        denominator_margin,
        successful,
        base_status,
        safe_triangle,
        safe_entity,
        safe_barycentric,
        safe_normal,
        front_facing,
        tie_count,
        uniqueness_margin,
        steps,
        tests,
        status,
        successful,
    )


def intersect_triangle_rays(
    prepared: PreparedTriangleRayQuery,
    origins: ArrayLike,
    directions: ArrayLike,
    /,
    *,
    geometry: TriangleRayGeometryState | None = None,
) -> TriangleRayIntersectionResult:
    """Return exact nearest oriented-triangle hits for a fixed ray batch.

    A successful BVH result has exhaustive semantics: every node that can
    contain an equal or nearer hit has been visited. Insufficient stack capacity
    never falls back or returns a possibly incomplete hit; it reports
    ``TRAVERSAL_CAPACITY_EXHAUSTED``. No positional ray-origin nudge is used.
    """
    if not isinstance(prepared, PreparedTriangleRayQuery):
        raise TypeError("prepared must be PreparedTriangleRayQuery.")
    geometry_ = prepared.reference_geometry if geometry is None else geometry
    if not isinstance(geometry_, TriangleRayGeometryState):
        raise TypeError("geometry must be TriangleRayGeometryState or None.")
    origins_ = jnp.asarray(origins, dtype=geometry_.triangle_vertices.dtype)
    directions_ = jnp.asarray(directions, dtype=geometry_.triangle_vertices.dtype)
    if origins_.shape != directions_.shape or origins_.shape[-1:] != (3,):
        raise ValueError("origins and directions must have matching shape B + (3,).")
    batch_shape = origins_.shape[:-1]
    flat_origins = origins_.reshape((-1, 3))
    flat_directions = directions_.reshape((-1, 3))
    values = jax.vmap(
        lambda origin, direction: _query_one(prepared, geometry_, origin, direction)
    )(flat_origins, flat_directions)
    (
        points,
        distances,
        margins,
        valid,
        base_status,
        triangles,
        entities,
        barycentric,
        normals,
        front_facing,
        tie_count,
        uniqueness_margin,
        steps,
        tests,
        status,
        successful,
    ) = values
    base = RayIntersectionResult(
        points.reshape(batch_shape + (3,)),
        distances.reshape(batch_shape),
        margins.reshape(batch_shape),
        valid.reshape(batch_shape),
        base_status.reshape(batch_shape),
    )
    return TriangleRayIntersectionResult(
        base,
        triangles.reshape(batch_shape),
        entities.reshape(batch_shape),
        barycentric.reshape(batch_shape + (3,)),
        normals.reshape(batch_shape + (3,)),
        front_facing.reshape(batch_shape),
        tie_count.reshape(batch_shape),
        uniqueness_margin.reshape(batch_shape),
        steps.reshape(batch_shape),
        tests.reshape(batch_shape),
        status.reshape(batch_shape),
        successful.reshape(batch_shape),
    )


__all__ = [
    "PreparedTriangleRayQuery",
    "TriangleRayGeometryState",
    "TriangleRayIntersectionResult",
    "TriangleRayIntersectionStatus",
    "TriangleRayQueryPlan",
    "intersect_triangle_rays",
    "prepare_triangle_ray_query",
    "refit_triangle_ray_geometry",
]
