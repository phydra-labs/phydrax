#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._bvh import build_packed_bvh
from ..._strict import StrictModule
from ._mesh import _closest_points_on_triangles, MeshQueryResult, TriangleMesh


class TriangleBVH(StrictModule):
    """Exact stack-traversed packed AABB hierarchy over immutable triangles."""

    mesh: TriangleMesh
    bbox_min: Array
    bbox_max: Array
    left: Array
    right: Array
    leaf_id: Array
    leaf_items: Array
    num_nodes: int = eqx.field(static=True)

    def __init__(self, mesh: TriangleMesh, *, leaf_size: int = 8):
        if not isinstance(mesh, TriangleMesh):
            raise TypeError("TriangleBVH requires a TriangleMesh.")
        triangles = np.asarray(mesh.triangles)
        packed = build_packed_bvh(
            np.min(triangles, axis=1),
            np.max(triangles, axis=1),
            np.mean(triangles, axis=1),
            leaf_size=leaf_size,
            dtype=mesh.vertices.dtype,
        )
        self.mesh = mesh
        self.bbox_min = packed.bbox_min
        self.bbox_max = packed.bbox_max
        self.left = packed.left
        self.right = packed.right
        self.leaf_id = packed.leaf_id
        self.leaf_items = packed.leaf_items
        self.num_nodes = int(packed.left.shape[0])

    def _query_one(self, point: Array) -> tuple[Array, Array, Array]:
        triangles = self.mesh.triangles
        stack = jnp.zeros((self.num_nodes,), dtype=jnp.int32).at[0].set(0)
        initial = (
            stack,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=point.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros((3,), dtype=point.dtype),
        )

        def condition(state):
            return state[1] > 0

        def body(state):
            stack_, top, best_distance_sq, best_face, best_point = state
            top = top - 1
            node = stack_[top]
            delta = jnp.maximum(
                0.0,
                jnp.maximum(self.bbox_min[node] - point, point - self.bbox_max[node]),
            )
            lower_bound = jnp.sum(delta * delta)
            active = lower_bound <= best_distance_sq
            leaf = self.leaf_id[node]

            def visit_leaf(leaf_state):
                stack_l, top_l, distance_l, face_l, point_l = leaf_state
                safe_leaf = jnp.maximum(leaf, 0)
                items = self.leaf_items[safe_leaf]
                valid = items >= 0
                safe_items = jnp.maximum(items, 0)
                closest = _closest_points_on_triangles(point, triangles[safe_items])
                distance_sq = jnp.sum((closest - point) ** 2, axis=-1)
                distance_sq = jnp.where(valid, distance_sq, jnp.inf)
                local = jnp.argmin(distance_sq)
                candidate_distance = distance_sq[local]
                improve = candidate_distance < distance_l
                return (
                    stack_l,
                    top_l,
                    jnp.where(improve, candidate_distance, distance_l),
                    jnp.where(improve, safe_items[local], face_l),
                    jnp.where(improve, closest[local], point_l),
                )

            def visit_internal(internal_state):
                stack_i, top_i, distance_i, face_i, point_i = internal_state
                left = self.left[node]
                right = self.right[node]
                left_delta = jnp.maximum(
                    0.0,
                    jnp.maximum(
                        self.bbox_min[left] - point,
                        point - self.bbox_max[left],
                    ),
                )
                right_delta = jnp.maximum(
                    0.0,
                    jnp.maximum(
                        self.bbox_min[right] - point,
                        point - self.bbox_max[right],
                    ),
                )
                left_distance = jnp.sum(left_delta * left_delta)
                right_distance = jnp.sum(right_delta * right_delta)
                near = jnp.where(left_distance <= right_distance, left, right)
                far = jnp.where(left_distance <= right_distance, right, left)
                stack_i = stack_i.at[top_i].set(far)
                stack_i = stack_i.at[top_i + 1].set(near)
                return stack_i, top_i + 2, distance_i, face_i, point_i

            def visit(active_state):
                return jax.lax.cond(
                    leaf >= 0,
                    visit_leaf,
                    visit_internal,
                    active_state,
                )

            return jax.lax.cond(
                active,
                visit,
                lambda inactive_state: inactive_state,
                (stack_, top, best_distance_sq, best_face, best_point),
            )

        _, _, distance_sq, face, closest = jax.lax.while_loop(
            condition,
            body,
            initial,
        )
        return closest, jnp.sqrt(distance_sq), face

    def query(self, points: Array, /) -> MeshQueryResult:
        points_ = jnp.asarray(points, dtype=self.mesh.vertices.dtype)
        if points_.ndim == 0 or points_.shape[-1] != 3:
            raise ValueError("points must have trailing dimension 3.")
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))
        closest, distance, face = jax.vmap(self._query_one)(flat)
        normal = self.mesh.face_normals[face]
        return MeshQueryResult(
            closest_point=closest.reshape((*leading, 3)),
            distance=distance.reshape(leading),
            face_index=face.reshape(leading),
            normal=normal.reshape((*leading, 3)),
        )


__all__ = ["TriangleBVH"]
