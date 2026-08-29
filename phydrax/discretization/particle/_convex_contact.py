#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._rigid_body import quaternion_rotation_matrix
from ._rigid_contact import RigidContactGeometry


class ConvexShapePlan(StrictModule, NonTrainableState):
    vertices: Array
    triangles: Array
    edges: Array
    material_id: int = eqx.field(static=True)
    shape_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        triangles: ArrayLike,
        material_id: int,
        /,
        *,
        volume_tolerance: float = 1.0e-14,
        shape_id: str | None = None,
    ):
        vertices_ = np.asarray(vertices)
        triangles_ = np.asarray(triangles)
        material = int(material_id)
        if vertices_.ndim != 2 or vertices_.shape[1] != 3 or vertices_.shape[0] < 4:
            raise ValueError("Convex vertices must have shape (vertices>=4,3).")
        if (
            triangles_.ndim != 2
            or triangles_.shape[1] != 3
            or not np.issubdtype(triangles_.dtype, np.integer)
        ):
            raise TypeError("Convex triangles must be integer triples.")
        if (
            np.any(~np.isfinite(vertices_))
            or np.any(triangles_ < 0)
            or np.any(triangles_ >= vertices_.shape[0])
            or material < 0
        ):
            raise ValueError("Convex shape geometry/material is invalid.")
        faces = vertices_[triangles_]
        cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        if np.any(np.linalg.norm(cross, axis=-1) <= volume_tolerance):
            raise ValueError("Convex shape contains a degenerate face.")
        center = np.mean(vertices_, axis=0)
        normals = cross / np.linalg.norm(cross, axis=-1, keepdims=True)
        face_centers = np.mean(faces, axis=1)
        inward = np.sum(normals * (center - face_centers), axis=-1) > 0.0
        triangles_ = triangles_.copy()
        triangles_[inward, 1], triangles_[inward, 2] = (
            triangles_[inward, 2].copy(),
            triangles_[inward, 1].copy(),
        )
        edge_set = {
            tuple(sorted((int(left), int(right))))
            for triangle in triangles_
            for left, right in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            )
        }
        edges = np.asarray(sorted(edge_set), dtype=np.int32)
        generated = canonical_fingerprint(
            {
                "kind": "convex-shape-plan",
                "values": array_tree_fingerprint(
                    {"vertices": vertices_, "triangles": triangles_, "edges": edges}
                ),
                "material_id": material,
            }
        )
        self.vertices = jnp.asarray(vertices_)
        self.triangles = jnp.asarray(triangles_, dtype=jnp.int32)
        self.edges = jnp.asarray(edges, dtype=jnp.int32)
        self.material_id = material
        self.shape_id = generated if shape_id is None else str(shape_id)
        if not self.shape_id:
            raise ValueError("shape_id must be nonempty.")

    def prepare(self) -> PreparedConvexShape:
        return PreparedConvexShape(self)


class PreparedConvexShape(StrictModule, NonTrainableState):
    plan: ConvexShapePlan
    face_normals: Array
    local_center: Array
    bounding_radius: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ConvexShapePlan, /):
        faces = plan.vertices[plan.triangles]
        cross = jnp.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        normals = cross / jnp.linalg.norm(cross, axis=-1, keepdims=True)
        center = jnp.mean(plan.vertices, axis=0)
        self.plan = plan
        self.face_normals = normals
        self.local_center = center
        self.bounding_radius = jnp.max(jnp.linalg.norm(plan.vertices - center, axis=-1))
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-convex-shape", "plan": plan.shape_id}
        )


class ConvexContactResult(StrictModule):
    geometry: RigidContactGeometry
    separating_axis_count: Array
    minimum_overlap: Array
    successful: Array


def _world_shape(shape: PreparedConvexShape, position: Array, orientation: Array, /):
    rotation = quaternion_rotation_matrix(orientation[None, :])[0]
    vertices = contract("ij,kj->ki", rotation, shape.plan.vertices) + position
    normals = contract("ij,kj->ki", rotation, shape.face_normals)
    edge_vectors = vertices[shape.plan.edges[:, 1]] - vertices[shape.plan.edges[:, 0]]
    center = contract("ij,j->i", rotation, shape.local_center) + position
    return vertices, normals, edge_vectors, center


def convex_sat_contact(
    left: PreparedConvexShape,
    right: PreparedConvexShape,
    left_position: Array,
    right_position: Array,
    left_orientation: Array,
    right_orientation: Array,
    left_velocity: Array,
    right_velocity: Array,
    left_angular_velocity: Array,
    right_angular_velocity: Array,
    contact_key: Array,
    /,
    *,
    axis_tolerance: float = 1.0e-12,
) -> ConvexContactResult:
    """Qualified convex-polyhedron contact using complete SAT face/edge axes."""

    lv, ln, le, lc = _world_shape(left, left_position, left_orientation)
    rv, rn, re, rc = _world_shape(right, right_position, right_orientation)
    edge_axes = jnp.cross(le[:, None, :], re[None, :, :]).reshape((-1, 3))
    axes = jnp.concatenate((ln, rn, edge_axes), axis=0)
    norms = jnp.linalg.norm(axes, axis=-1)
    axis_valid = norms > axis_tolerance
    axes = axes / jnp.where(axis_valid, norms, 1.0)[:, None]
    left_projection = contract("ad,vd->av", axes, lv)
    right_projection = contract("ad,vd->av", axes, rv)
    overlap = jnp.minimum(
        jnp.max(left_projection, axis=-1), jnp.max(right_projection, axis=-1)
    ) - jnp.maximum(jnp.min(left_projection, axis=-1), jnp.min(right_projection, axis=-1))
    overlap = jnp.where(axis_valid, overlap, jnp.inf)
    intersecting = jnp.all(overlap >= 0.0)
    selected = jnp.argmin(overlap)
    axis = axes[selected]
    center_direction = lc - rc
    normal = jnp.where(jnp.dot(axis, center_direction) >= 0.0, axis, -axis)
    penetration = jnp.maximum(overlap[selected], 0.0)
    left_support_projection = contract("vd,d->v", lv, normal)
    right_support_projection = contract("vd,d->v", rv, normal)
    left_witness = lv[jnp.argmin(left_support_projection)]
    right_witness = rv[jnp.argmax(right_support_projection)]
    contact_point = 0.5 * (left_witness + right_witness)
    left_arm = contact_point - left_position
    right_arm = contact_point - right_position
    left_contact_velocity = left_velocity + jnp.cross(left_angular_velocity, left_arm)
    right_contact_velocity = right_velocity + jnp.cross(right_angular_velocity, right_arm)
    relative = left_contact_velocity - right_contact_velocity
    normal_velocity = jnp.dot(relative, normal)
    tangent = relative - normal_velocity * normal
    valid = (
        intersecting & jnp.all(jnp.isfinite(normal)) & (norms[selected] > axis_tolerance)
    )
    geometry = RigidContactGeometry(
        normal[None, :],
        jnp.asarray([-penetration]),
        jnp.asarray([penetration]),
        jnp.zeros((1,)),
        contact_point[None, :],
        left_arm[None, :],
        right_arm[None, :],
        left_arm[None, :],
        right_arm[None, :],
        relative[None, :],
        jnp.asarray([normal_velocity]),
        tangent[None, :],
        left_angular_velocity[None, :],
        right_angular_velocity[None, :],
        jnp.asarray([contact_key], dtype=jnp.int64),
        jnp.asarray([selected], dtype=jnp.int32),
        jnp.asarray([selected], dtype=jnp.int32),
        valid[None],
        jnp.where(valid, 0, 1).astype(jnp.int32)[None],
        jnp.asarray([jnp.min(jnp.abs(overlap))]),
        valid,
        "rigid-contact:convex-sat",
    )
    return ConvexContactResult(
        geometry,
        jnp.sum(axis_valid, dtype=jnp.int32),
        penetration,
        valid,
    )


__all__ = [
    "ConvexContactResult",
    "ConvexShapePlan",
    "PreparedConvexShape",
    "convex_sat_contact",
]
