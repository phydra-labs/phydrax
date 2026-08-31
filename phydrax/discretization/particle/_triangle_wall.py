#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._pair_state import particle_wall_interaction_keys
from ._rigid_contact import RigidContactGeometry
from ._rigid_sphere import (
    PreparedRigidSphereSet,
    RigidSphereKinematics,
    sphere_spin_velocity,
)


class TriangleWallPlan(StrictModule, NonTrainableState):
    vertices: Array
    triangles: Array
    triangle_material: Array
    vertex_ids: Array
    triangle_ids: Array
    two_sided: bool = eqx.field(static=True)
    wall_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        triangles: ArrayLike,
        triangle_material: ArrayLike,
        /,
        *,
        two_sided: bool = False,
        vertex_ids: ArrayLike | None = None,
        triangle_ids: ArrayLike | None = None,
        area_tolerance: float = 1.0e-14,
        wall_id: str | None = None,
    ):
        vertices_ = np.asarray(vertices)
        triangles_ = np.asarray(triangles)
        materials = np.asarray(triangle_material)
        vertex_ids_ = (
            np.arange(vertices_.shape[0], dtype=np.int64)
            if vertex_ids is None
            else np.asarray(vertex_ids)
        )
        triangle_ids_ = (
            np.arange(triangles_.shape[0], dtype=np.int64)
            if triangle_ids is None
            else np.asarray(triangle_ids)
        )
        if vertices_.ndim != 2 or vertices_.shape[1] != 3:
            raise ValueError("vertices must have shape (vertices,3).")
        if (
            triangles_.ndim != 2
            or triangles_.shape[1] != 3
            or not np.issubdtype(triangles_.dtype, np.integer)
        ):
            raise TypeError("triangles must have shape (faces,3) with integer indices.")
        if materials.shape != (triangles_.shape[0],) or not np.issubdtype(
            materials.dtype, np.integer
        ):
            raise TypeError("triangle_material must have face shape and integer dtype.")
        if (
            np.any(~np.isfinite(vertices_))
            or np.any(triangles_ < 0)
            or np.any(triangles_ >= vertices_.shape[0])
            or np.any(materials < 0)
            or vertex_ids_.shape != (vertices_.shape[0],)
            or triangle_ids_.shape != (triangles_.shape[0],)
            or not np.issubdtype(vertex_ids_.dtype, np.integer)
            or not np.issubdtype(triangle_ids_.dtype, np.integer)
            or np.any(vertex_ids_ < 0)
            or np.any(triangle_ids_ < 0)
            or np.unique(vertex_ids_).size != vertex_ids_.size
            or np.unique(triangle_ids_).size != triangle_ids_.size
        ):
            raise ValueError("Triangle wall geometry/material/stable IDs are invalid.")
        selected = vertices_[triangles_]
        cross = np.cross(selected[:, 1] - selected[:, 0], selected[:, 2] - selected[:, 0])
        doubled_area = np.linalg.norm(cross, axis=-1)
        if np.any(doubled_area <= float(area_tolerance)):
            raise ValueError("Triangle wall contains a degenerate face.")
        generated = canonical_fingerprint(
            {
                "kind": "triangle-wall-plan",
                "values": array_tree_fingerprint(
                    {
                        "vertices": vertices_,
                        "triangles": triangles_,
                        "materials": materials,
                        "vertex_ids": vertex_ids_.astype(np.int64),
                        "triangle_ids": triangle_ids_.astype(np.int64),
                    }
                ),
                "two_sided": bool(two_sided),
            }
        )
        self.vertices = jnp.asarray(vertices_)
        self.triangles = jnp.asarray(triangles_, dtype=jnp.int32)
        self.triangle_material = jnp.asarray(materials, dtype=jnp.int32)
        self.vertex_ids = jnp.asarray(vertex_ids_, dtype=jnp.int64)
        self.triangle_ids = jnp.asarray(triangle_ids_, dtype=jnp.int64)
        self.two_sided = bool(two_sided)
        self.wall_id = generated if wall_id is None else str(wall_id)
        if not self.wall_id:
            raise ValueError("wall_id must be nonempty.")

    def prepare(self) -> PreparedTriangleWall:
        return PreparedTriangleWall(self)


class PreparedTriangleWall(StrictModule, NonTrainableState):
    plan: TriangleWallPlan
    face_vertices: Array
    normals: Array
    aabb_lower: Array
    aabb_upper: Array
    triangle_ids: Array
    edge_ids: Array
    edge_owner_triangle_ids: Array
    vertex_owner_triangle_ids: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: TriangleWallPlan, /):
        if not isinstance(plan, TriangleWallPlan):
            raise TypeError("plan must be a TriangleWallPlan.")
        face_vertices = plan.vertices[plan.triangles]
        cross = jnp.cross(
            face_vertices[:, 1] - face_vertices[:, 0],
            face_vertices[:, 2] - face_vertices[:, 0],
        )
        normals = cross / jnp.linalg.norm(cross, axis=-1, keepdims=True)
        self.plan = plan
        self.face_vertices = face_vertices
        self.normals = normals
        self.aabb_lower = jnp.min(face_vertices, axis=1)
        self.aabb_upper = jnp.max(face_vertices, axis=1)
        self.triangle_ids = plan.triangle_ids
        local_edges = np.asarray(plan.triangles)[:, ((0, 1), (1, 2), (2, 0))]
        local_vertex_ids = np.asarray(plan.vertex_ids)[local_edges]
        canonical_edges = np.sort(local_vertex_ids, axis=-1)
        unique_edges = sorted(
            {tuple(value) for value in canonical_edges.reshape((-1, 2)).tolist()}
        )
        edge_lookup = {value: index for index, value in enumerate(unique_edges)}
        edge_ids = np.asarray(
            [[edge_lookup[tuple(value)] for value in row] for row in canonical_edges],
            dtype=np.int64,
        )
        triangle_ids_host = np.asarray(plan.triangle_ids, dtype=np.int64)
        edge_owner = np.full((len(unique_edges),), np.iinfo(np.int64).max, dtype=np.int64)
        for face_index, row in enumerate(edge_ids):
            for edge_id in row:
                edge_owner[edge_id] = min(
                    edge_owner[edge_id], triangle_ids_host[face_index]
                )
        vertex_owner = np.full(
            (plan.vertices.shape[0],), np.iinfo(np.int64).max, dtype=np.int64
        )
        for face_index, row in enumerate(np.asarray(plan.triangles)):
            for vertex in row:
                vertex_owner[vertex] = min(
                    vertex_owner[vertex], triangle_ids_host[face_index]
                )
        self.edge_ids = jnp.asarray(edge_ids, dtype=jnp.int64)
        self.edge_owner_triangle_ids = jnp.asarray(edge_owner, dtype=jnp.int64)
        self.vertex_owner_triangle_ids = jnp.asarray(vertex_owner, dtype=jnp.int64)
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-triangle-wall", "plan": plan.wall_id}
        )

    @property
    def face_count(self) -> int:
        return int(self.face_vertices.shape[0])


def _segment_closest(point, start, end, /):
    direction = end - start
    denominator = jnp.sum(direction * direction, axis=-1)
    parameter = jnp.sum((point - start) * direction, axis=-1) / denominator
    parameter = jnp.clip(parameter, 0.0, 1.0)
    return start + parameter[:, None] * direction, parameter


def sphere_triangle_contact_geometry(
    bodies: PreparedRigidSphereSet,
    kinematics: RigidSphereKinematics,
    wall: PreparedTriangleWall,
    /,
    *,
    feature_tolerance: float = 1.0e-10,
    distance_tolerance: float = 1.0e-12,
) -> TriangleWallContactResult:
    if bodies.ambient_dimension != 3:
        raise ValueError("Triangle wall contact currently requires 3-D spheres.")
    count = bodies.capacity
    faces = wall.face_count
    owner = jnp.repeat(jnp.arange(count, dtype=jnp.int32), faces)
    face = jnp.tile(jnp.arange(faces, dtype=jnp.int32), count)
    center = kinematics.position[owner]
    radius = bodies.radii[owner]
    vertices = wall.face_vertices[face]
    a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    face_normal = wall.normals[face]
    signed_plane = jnp.sum((center - a) * face_normal, axis=-1)
    projection = center - signed_plane[:, None] * face_normal
    v0 = b - a
    v1 = c - a
    v2 = projection - a
    d00 = jnp.sum(v0 * v0, axis=-1)
    d01 = jnp.sum(v0 * v1, axis=-1)
    d11 = jnp.sum(v1 * v1, axis=-1)
    d20 = jnp.sum(v2 * v0, axis=-1)
    d21 = jnp.sum(v2 * v1, axis=-1)
    denominator = d00 * d11 - d01 * d01
    bary_b = (d11 * d20 - d01 * d21) / denominator
    bary_c = (d00 * d21 - d01 * d20) / denominator
    bary_a = 1.0 - bary_b - bary_c
    inside = (
        (bary_a >= -feature_tolerance)
        & (bary_b >= -feature_tolerance)
        & (bary_c >= -feature_tolerance)
    )
    ab, tab = _segment_closest(center, a, b)
    bc, tbc = _segment_closest(center, b, c)
    ca, tca = _segment_closest(center, c, a)
    candidates = jnp.stack((projection, ab, bc, ca), axis=1)
    candidate_valid = jnp.stack(
        (inside, jnp.ones_like(inside), jnp.ones_like(inside), jnp.ones_like(inside)),
        axis=1,
    )
    distance_squared = jnp.sum((center[:, None, :] - candidates) ** 2, axis=-1)
    distance_squared = jnp.where(candidate_valid, distance_squared, jnp.inf)
    selected = jnp.argmin(distance_squared, axis=1)
    closest = jnp.take_along_axis(candidates, selected[:, None, None], axis=1)[:, 0]
    edge_parameter = jnp.stack((jnp.zeros_like(tab), tab, tbc, tca), axis=1)
    selected_parameter = jnp.take_along_axis(edge_parameter, selected[:, None], axis=1)[
        :, 0
    ]
    feature = selected.astype(jnp.int32)
    feature = jnp.where(
        (selected > 0) & (selected_parameter <= feature_tolerance),
        jnp.asarray((0, 4, 5, 6), dtype=jnp.int32)[selected],
        feature,
    )
    feature = jnp.where(
        (selected > 0) & (selected_parameter >= 1.0 - feature_tolerance),
        jnp.asarray((0, 5, 6, 4), dtype=jnp.int32)[selected],
        feature,
    )
    separation = center - closest
    distance = jnp.sqrt(jnp.sum(separation * separation, axis=-1))
    positive = distance > distance_tolerance
    separation_normal = separation / jnp.where(positive, distance, 1.0)[:, None]
    normal = jnp.where(
        wall.plan.two_sided,
        separation_normal,
        face_normal,
    )
    front = wall.plan.two_sided | (signed_plane >= -feature_tolerance)
    gap = distance - radius
    overlap = jnp.maximum(-gap, 0.0)
    active_owner = bodies.particles.active_mask[owner]
    degenerate = active_owner & (overlap > 0.0) & ~positive
    edge_index = jnp.clip(feature - 1, 0, 2)
    vertex_local = jnp.clip(feature - 4, 0, 2)
    edge_id = wall.edge_ids[face, edge_index]
    vertex_index = wall.plan.triangles[face, vertex_local]
    vertex_id = wall.plan.vertex_ids[vertex_index]
    feature_kind = jnp.where(feature == 0, 0, jnp.where(feature <= 3, 1, 2))
    feature_id = jnp.where(
        feature == 0,
        wall.triangle_ids[face],
        jnp.where(feature <= 3, edge_id, vertex_id),
    )
    feature_owned = jnp.where(
        feature == 0,
        True,
        jnp.where(
            feature <= 3,
            wall.edge_owner_triangle_ids[edge_id] == wall.triangle_ids[face],
            wall.vertex_owner_triangle_ids[vertex_index] == wall.triangle_ids[face],
        ),
    )
    valid = active_owner & front & ~degenerate & feature_owned
    owner_arm = closest - center
    contact_velocity = kinematics.velocity[owner] + sphere_spin_velocity(
        kinematics.angular_velocity[owner], owner_arm, 3
    )
    normal_velocity = jnp.sum(contact_velocity * normal, axis=-1)
    tangent_velocity = contact_velocity - normal_velocity[:, None] * normal
    object_id = jnp.where(feature == 0, wall.triangle_ids[face], 0)
    key = particle_wall_interaction_keys(
        bodies.particles.particle_ids[owner],
        object_id,
        feature_kind,
        feature_id,
        valid,
    )
    geometry = RigidContactGeometry(
        jnp.where(valid[:, None], normal, 0.0),
        jnp.where(valid, gap, 0.0),
        jnp.where(valid, overlap, 0.0),
        jnp.where(valid, radius, 0.0),
        jnp.where(valid[:, None], closest, 0.0),
        jnp.where(valid[:, None], owner_arm, 0.0),
        jnp.zeros_like(owner_arm),
        jnp.where(valid[:, None], owner_arm, 0.0),
        jnp.zeros_like(owner_arm),
        jnp.where(valid[:, None], contact_velocity, 0.0),
        jnp.where(valid, normal_velocity, 0.0),
        jnp.where(valid[:, None], tangent_velocity, 0.0),
        kinematics.angular_velocity[owner],
        jnp.zeros_like(kinematics.angular_velocity[owner]),
        key,
        feature,
        feature,
        valid,
        degenerate.astype(jnp.int32),
        jnp.minimum(
            jnp.min(jnp.abs(jnp.stack((bary_a, bary_b, bary_c), axis=-1)), axis=-1),
            jnp.abs(distance),
        ),
        ~jnp.any(degenerate),
        "rigid-contact:sphere-triangle-wall",
    )
    return TriangleWallContactResult(
        geometry,
        owner,
        face,
        bodies.material_ids[owner],
        wall.plan.triangle_material[face],
        wall.prepared_id,
    )


class TriangleWallContactResult(StrictModule):
    geometry: RigidContactGeometry
    owner_indices: Array
    triangle_indices: Array
    particle_material: Array
    wall_material: Array
    wall_id: str = eqx.field(static=True)


__all__ = [
    "PreparedTriangleWall",
    "TriangleWallContactResult",
    "TriangleWallPlan",
    "sphere_triangle_contact_geometry",
]
