#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse
from ._core import ParticleDiscretization
from ._pair_state import ParticlePairKeySpace
from ._pairwise import ParticlePairGeometry, ParticlePairRelation
from ._rigid_body import (
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
    RigidBodyKinematics,
    RigidBodySetPlan,
)
from ._rigid_contact import RigidContactGeometry
from ._rigid_shape import (
    AbstractPreparedRigidContactShape,
    AbstractRigidContactShapePlan,
    bounded_contact_geometry,
    CompliantContactShapeKind,
    RigidContactGeometryBatch,
)
from ._signed_field import (
    exact_signed_field_result,
    posed_signed_field_coordinates,
    PosedSignedFieldQueryResult,
)


@final
class ConvexShapePlan(StrictModule, NonTrainableState):
    vertices: Array
    triangles: Array
    edges: Array
    center_of_mass: Array
    volume: Array
    inertia_per_mass: Array
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
    ) -> None:
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
        scale = float(
            np.max(np.linalg.norm(vertices_ - np.mean(vertices_, axis=0), axis=-1))
        )
        if np.any(np.linalg.norm(cross, axis=-1) <= volume_tolerance * scale**2):
            raise ValueError("Convex shape contains a degenerate face.")
        center, volume, inertia = _closed_shell_mass_properties(
            vertices_, triangles_, float(volume_tolerance)
        )
        normals = cross / np.linalg.norm(cross, axis=-1, keepdims=True)
        distances = np.sum(
            normals[:, None, :] * (vertices_[None, :, :] - faces[:, :1, :]), axis=-1
        )
        scale = np.max(np.linalg.norm(vertices_ - center, axis=-1))
        if np.any(distances > volume_tolerance * scale):
            raise ValueError(
                "Triangle shell must bound a convex solid with outward faces."
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
        self.center_of_mass = jnp.asarray(center)
        self.volume = jnp.asarray(volume)
        self.inertia_per_mass = jnp.asarray(inertia)
        self.material_id = material
        self.shape_id = generated if shape_id is None else str(shape_id)
        if not self.shape_id:
            raise ValueError("shape_id must be nonempty.")

    def prepare(self) -> PreparedConvexShape:
        return PreparedConvexShape(self)


@final
class PreparedConvexShape(StrictModule, NonTrainableState):
    plan: ConvexShapePlan
    face_normals: Array
    local_center: Array
    bounding_radius: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ConvexShapePlan, /) -> None:
        faces = plan.vertices[plan.triangles]
        cross = jnp.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        normals = cross / jnp.linalg.norm(cross, axis=-1, keepdims=True)
        center = plan.center_of_mass
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


def _world_shape(
    shape: PreparedConvexShape, position: Array, orientation: Array, /
) -> tuple[Array, Array, Array, Array]:
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

    lv, ln, le, _ = _world_shape(left, left_position, left_orientation)
    rv, rn, re, _ = _world_shape(right, right_position, right_orientation)
    left_packed = (
        lv,
        ln,
        le,
        jnp.ones((lv.shape[0],), dtype=jnp.bool_),
        jnp.ones((ln.shape[0],), dtype=jnp.bool_),
        jnp.ones((le.shape[0],), dtype=jnp.bool_),
    )
    right_packed = (
        rv,
        rn,
        re,
        jnp.ones((rv.shape[0],), dtype=jnp.bool_),
        jnp.ones((rn.shape[0],), dtype=jnp.bool_),
        jnp.ones((re.shape[0],), dtype=jnp.bool_),
    )
    normal, gap, contact_point, selected, margin, finite, axis_count = _packed_sat(
        left_packed, right_packed, axis_tolerance=axis_tolerance
    )
    penetration = jnp.maximum(-gap, 0.0)
    left_arm = contact_point - left_position
    right_arm = contact_point - right_position
    left_contact_velocity = left_velocity + jnp.cross(left_angular_velocity, left_arm)
    right_contact_velocity = right_velocity + jnp.cross(right_angular_velocity, right_arm)
    relative = left_contact_velocity - right_contact_velocity
    normal_velocity = jnp.dot(relative, normal)
    tangent = relative - normal_velocity * normal
    valid = finite & (gap <= 0.0)
    geometry = RigidContactGeometry(
        normal[None, :],
        jnp.asarray([gap]),
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
        margin[None],
        valid,
        "rigid-contact:convex-sat",
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.ones((1,), dtype=jnp.int32),
        normal[None, :],
        jnp.ones((1,), dtype=normal.dtype),
        jnp.zeros((1,), dtype=jnp.bool_),
        margin[None],
        jnp.zeros((1,), dtype=normal.dtype),
    )
    return ConvexContactResult(
        geometry,
        axis_count,
        penetration,
        valid,
    )


def _closed_shell_mass_properties(
    vertices: np.ndarray, triangles: np.ndarray, tolerance: float, /
) -> tuple[np.ndarray, float, np.ndarray]:
    """Integrate signed tetrahedra of a closed outward triangle shell."""
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("volume_tolerance must be finite and positive.")
    directed: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in triangles:
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            lower, upper = sorted((int(left), int(right)))
            edge = (lower, upper)
            directed.setdefault(edge, []).append((int(left), int(right)))
    if any(len(uses) != 2 or uses[0] != uses[1][::-1] for uses in directed.values()):
        raise ValueError(
            "Convex triangles must form a closed consistently oriented shell."
        )
    reference = np.mean(vertices, axis=0)
    faces = vertices[triangles] - reference
    signed = np.sum(faces[:, 0] * np.cross(faces[:, 1], faces[:, 2]), axis=-1) / 6.0
    volume = float(np.sum(signed))
    scale = float(np.max(np.linalg.norm(vertices - reference, axis=-1)))
    if volume <= tolerance * scale**3:
        raise ValueError("Convex shell must have positive outward-oriented volume.")
    total = np.sum(faces, axis=1)
    center_relative = np.sum(signed[:, None] * total, axis=0) / (4.0 * volume)
    second = np.zeros((3, 3), dtype=np.float64)
    for weight, face, vector in zip(signed, faces, total, strict=True):
        second += weight * (face.T @ face + np.outer(vector, vector)) / 20.0
    covariance = second / volume - np.outer(center_relative, center_relative)
    inertia = np.trace(covariance) * np.eye(3, dtype=np.float64) - covariance
    if np.any(~np.isfinite(inertia)) or np.any(np.linalg.eigvalsh(inertia) <= 0.0):
        raise ValueError("Convex shell COM inertia must be positive definite.")
    return reference + center_relative, volume, inertia


@final
class ConvexShapeSetPlan(AbstractRigidContactShapePlan):
    templates: tuple[ConvexShapePlan, ...]
    owner_template_ids: Array
    owner_material_ids: Array
    fixed_mask: Array
    shape_kind: CompliantContactShapeKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        templates: Sequence[ConvexShapePlan],
        owner_template_ids: ArrayLike,
        owner_material_ids: ArrayLike,
        /,
        *,
        fixed_mask: ArrayLike | None = None,
        plan_id: str | None = None,
    ) -> None:
        templates_ = tuple(templates)
        if not templates_ or any(
            not isinstance(value, ConvexShapePlan) for value in templates_
        ):
            raise TypeError("templates must contain ConvexShapePlan values.")
        selected = np.asarray(owner_template_ids)
        material = np.asarray(owner_material_ids)
        if selected.ndim != 1 or selected.size == 0 or material.shape != selected.shape:
            raise ValueError(
                "Owner template/material IDs must be matching nonempty vectors."
            )
        if not np.issubdtype(selected.dtype, np.integer) or not np.issubdtype(
            material.dtype, np.integer
        ):
            raise TypeError("Owner template/material IDs must be integer arrays.")
        if (
            np.any(selected < 0)
            or np.any(selected >= len(templates_))
            or np.any(material < 0)
        ):
            raise ValueError("Owner template/material IDs are out of range.")
        fixed = (
            np.zeros(selected.shape, dtype=np.bool_)
            if fixed_mask is None
            else np.asarray(fixed_mask, dtype=np.bool_)
        )
        if fixed.shape != selected.shape:
            raise ValueError("fixed_mask must have owner-capacity shape.")
        generated = canonical_fingerprint(
            {
                "kind": "convex-shape-set-plan",
                "templates": [value.shape_id for value in templates_],
                "owners": array_tree_fingerprint(
                    {"templates": selected, "materials": material, "fixed": fixed}
                ),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.templates = templates_
        self.owner_template_ids = jnp.asarray(selected, dtype=jnp.int32)
        self.owner_material_ids = jnp.asarray(material, dtype=jnp.int32)
        self.fixed_mask = jnp.asarray(fixed)
        self.shape_kind = parse("convex", CompliantContactShapeKind, "shape_kind")
        self.plan_id = identifier

    def rigid_body_plan(self, particles: ParticleDiscretization, /) -> RigidBodySetPlan:
        if particles.ambient_dimension != 3 or self.owner_template_ids.shape != (
            particles.capacity,
        ):
            raise ValueError(
                "Convex shapes require a matching three-dimensional population."
            )
        selected = np.asarray(self.owner_template_ids)
        unit_inertia = np.stack(
            [np.asarray(self.templates[index].inertia_per_mass) for index in selected]
        )
        inertia = np.asarray(particles.safe_masses)[:, None, None] * unit_inertia
        return RigidBodySetPlan(
            self.owner_material_ids,
            inertia,
            fixed_mask=self.fixed_mask,
            name="convex-rigid-bodies",
        )

    def prepare(self, particles: ParticleDiscretization, /) -> PreparedConvexShapeSet:
        return PreparedConvexShapeSet(self, particles)


@final
class PreparedConvexShapeSet(AbstractPreparedRigidContactShape):
    plan: ConvexShapeSetPlan
    bodies: PreparedRigidBodySet
    vertices: Array
    face_normals: Array
    triangle_indices: Array
    edge_vectors: Array
    vertex_valid: Array
    face_valid: Array
    edge_valid: Array
    bounding_radii: Array
    volumes: Array
    prepared_id: str = eqx.field(static=True)
    maximum_vertices: int = eqx.field(static=True)
    maximum_faces: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)

    def __init__(
        self, plan: ConvexShapeSetPlan, particles: ParticleDiscretization, /
    ) -> None:
        bodies = plan.rigid_body_plan(particles).prepare(particles)
        count = len(plan.templates)
        nv = max(value.vertices.shape[0] for value in plan.templates)
        nf = max(value.triangles.shape[0] for value in plan.templates)
        ne = max(value.edges.shape[0] for value in plan.templates)
        vertices = np.zeros((count, nv, 3), dtype=np.float64)
        triangles = np.zeros((count, nf, 3), dtype=np.int32)
        normals = np.zeros((count, nf, 3), dtype=np.float64)
        edges = np.zeros((count, ne, 3), dtype=np.float64)
        vv = np.zeros((count, nv), dtype=np.bool_)
        fv = np.zeros((count, nf), dtype=np.bool_)
        ev = np.zeros((count, ne), dtype=np.bool_)
        bounds = np.zeros(count, dtype=np.float64)
        volumes = np.zeros(count, dtype=np.float64)
        for index, template in enumerate(plan.templates):
            local = np.asarray(template.vertices) - np.asarray(template.center_of_mass)
            faces = local[np.asarray(template.triangles)]
            normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
            normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
            edge = np.asarray(template.edges)
            vertices[index, : local.shape[0]] = local
            normals[index, : normal.shape[0]] = normal
            triangles[index, : normal.shape[0]] = np.asarray(template.triangles)
            edges[index, : edge.shape[0]] = local[edge[:, 1]] - local[edge[:, 0]]
            vv[index, : local.shape[0]] = True
            fv[index, : normal.shape[0]] = True
            ev[index, : edge.shape[0]] = True
            bounds[index] = np.max(np.linalg.norm(local, axis=-1))
            volumes[index] = np.asarray(template.volume)
        selected = np.asarray(plan.owner_template_ids)
        dtype = particles.safe_masses.dtype
        self.plan = plan
        self.bodies = bodies
        self.vertices = jnp.asarray(vertices[selected], dtype=dtype)
        self.face_normals = jnp.asarray(normals[selected], dtype=dtype)
        self.triangle_indices = jnp.asarray(triangles[selected], dtype=jnp.int32)
        self.edge_vectors = jnp.asarray(edges[selected], dtype=dtype)
        self.vertex_valid = jnp.asarray(vv[selected])
        self.face_valid = jnp.asarray(fv[selected])
        self.edge_valid = jnp.asarray(ev[selected])
        self.bounding_radii = jnp.where(
            particles.active_mask, jnp.asarray(bounds[selected], dtype=dtype), 0.0
        )
        self.volumes = jnp.where(
            particles.active_mask, jnp.asarray(volumes[selected], dtype=dtype), 0.0
        )
        self.maximum_vertices = nv
        self.maximum_faces = nf
        self.maximum_edges = ne
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-convex-shape-set",
                "plan": plan.plan_id,
                "bodies": bodies.prepared_id,
                "max_vertices": nv,
                "max_faces": nf,
                "max_edges": ne,
            }
        )

    @property
    def particles(self) -> ParticleDiscretization:
        return self.bodies.particles

    @property
    def material_ids(self) -> Array:
        return self.bodies.material_ids

    @property
    def contact_material_ids(self) -> Array:
        return self.material_ids

    @property
    def shape_kind(self) -> CompliantContactShapeKind:
        return self.plan.shape_kind

    @property
    def shape_id(self) -> str:
        return self.prepared_id

    @property
    def routes_per_pair(self) -> int:
        return 1

    def evaluate_signed_field_query(
        self,
        kinematics: RigidBodyKinematics,
        body_indices: Array,
        world_points: Array,
        /,
        *,
        shape_scale: Array | None = None,
    ) -> PosedSignedFieldQueryResult:
        """Exact Euclidean signed distance to the closed convex triangle shell."""
        local, indices, scale, supported = posed_signed_field_coordinates(
            kinematics,
            body_indices,
            world_points,
            self.bodies.particles.active_mask,
            shape_scale=shape_scale,
        )

        def query(route: tuple[Array, Array]) -> Array:
            point, body = route
            return _convex_signed_distance(
                point,
                self.vertices[body][self.triangle_indices[body]],
                self.face_normals[body],
                self.face_valid[body],
            )

        # One prepared maximum-face working set, not queries × maximum_faces.
        value = scale * jax.lax.map(query, (local, indices))
        return exact_signed_field_result(value, supported, self.shape_id)

    def pair_geometry(
        self,
        kinematics: RigidBodyKinematics,
        shape_scale: Array,
        pairs: ParticlePairRelation,
        pair_key_space: ParticlePairKeySpace,
        /,
        *,
        maximum_contact_routes: int,
        center_geometry: ParticlePairGeometry | None = None,
        distance_tolerance: float = 1.0e-12,
    ) -> RigidContactGeometryBatch:
        del center_geometry, distance_tolerance
        left, right = pairs.left_indices, pairs.right_indices
        rotation = quaternion_rotation_matrix(kinematics.orientation)
        vertices = kinematics.position[:, None, :] + contract(
            "...ij,...vj->...vi", rotation, self.vertices * shape_scale[:, None, None]
        )
        normals = contract("...ij,...fj->...fi", rotation, self.face_normals)
        edges = contract("...ij,...ej->...ei", rotation, self.edge_vectors)

        def packed(indices: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
            return (
                vertices[indices],
                normals[indices],
                edges[indices],
                self.vertex_valid[indices],
                self.face_valid[indices],
                self.edge_valid[indices],
            )

        normal, gap, point, feature, margin, finite, _ = jax.vmap(_packed_sat)(
            packed(left), packed(right)
        )
        valid = pairs.valid & (left != right) & finite
        left_arm, right_arm = (
            point - kinematics.position[left],
            point - kinematics.position[right],
        )
        relative = (
            kinematics.velocity[left]
            + jnp.cross(kinematics.angular_velocity[left], left_arm)
            - kinematics.velocity[right]
            - jnp.cross(kinematics.angular_velocity[right], right_arm)
        )
        normal_velocity = jnp.sum(relative * normal, axis=-1)
        keys = pair_key_space.keys(pairs)
        geometry = RigidContactGeometry(
            normal,
            gap,
            jnp.where(valid, jnp.maximum(-gap, 0.0), 0.0),
            jnp.zeros_like(gap),
            point,
            left_arm,
            right_arm,
            left_arm,
            right_arm,
            relative,
            normal_velocity,
            relative - normal_velocity[:, None] * normal,
            kinematics.angular_velocity[left],
            kinematics.angular_velocity[right],
            keys.keys,
            feature,
            feature,
            valid,
            (~finite).astype(jnp.int32),
            margin,
            jnp.all(~pairs.valid | valid),
            "rigid-contact:convex-sat",
            left,
            right,
            normal,
            jnp.ones_like(gap),
            jnp.zeros_like(valid),
            margin,
            jnp.zeros_like(gap),
        )
        return bounded_contact_geometry(
            geometry,
            keys,
            maximum_contact_routes,
            self.shape_id,
            left_material_ids=self.material_ids[left],
            right_material_ids=self.material_ids[right],
        )


def _convex_signed_distance(
    point: Array, triangles: Array, normals: Array, face_valid: Array, /
) -> Array:
    """Closest points lie on a face interior or one of its closed edges."""
    first = triangles[:, 0, :]
    signed_plane = jnp.sum((point[None, :] - first) * normals, axis=-1)
    projection = point[None, :] - signed_plane[:, None] * normals
    ends = jnp.roll(triangles, -1, axis=1)
    edges = ends - triangles
    edge_cross = jnp.cross(edges, projection[:, None, :] - triangles)
    projected_inside = jnp.all(
        jnp.sum(edge_cross * normals[:, None, :], axis=-1) >= 0.0, axis=-1
    )
    edge_squared = jnp.sum(edges * edges, axis=-1)
    fraction = jnp.sum((point[None, None, :] - triangles) * edges, axis=-1) / jnp.where(
        face_valid[:, None], edge_squared, 1.0
    )
    closest_edge = triangles + jnp.clip(fraction, 0.0, 1.0)[:, :, None] * edges
    edge_distance_squared = jnp.min(
        jnp.sum((point[None, None, :] - closest_edge) ** 2, axis=-1), axis=-1
    )
    face_distance_squared = jnp.where(
        projected_inside, signed_plane**2, edge_distance_squared
    )
    distance = jnp.sqrt(jnp.min(jnp.where(face_valid, face_distance_squared, jnp.inf)))
    inside = jnp.all(~face_valid | (signed_plane <= 0.0))
    return jnp.where(inside, -distance, distance)


def _packed_sat(
    left: tuple[Array, Array, Array, Array, Array, Array],
    right: tuple[Array, Array, Array, Array, Array, Array],
    /,
    *,
    axis_tolerance: float = 1.0e-12,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    lv, ln, le, lvv, lfv, lev = left
    rv, rn, re, rvv, rfv, rev = right
    left_norm = jnp.sqrt(jnp.sum(le * le, axis=-1))
    right_norm = jnp.sqrt(jnp.sum(re * re, axis=-1))
    le = le / jnp.where(lev, left_norm, 1.0)[:, None]
    re = re / jnp.where(rev, right_norm, 1.0)[:, None]
    edge_axes = jnp.cross(le[:, None, :], re[None, :, :]).reshape((-1, 3))
    axes = jnp.concatenate((ln, rn, edge_axes), axis=0)
    norms = jnp.sqrt(jnp.sum(axes * axes, axis=-1))
    valid = jnp.concatenate((lfv, rfv, (lev[:, None] & rev[None, :]).reshape((-1,)))) & (
        norms > axis_tolerance
    )
    axes = axes / jnp.where(valid, norms, 1.0)[:, None]
    lp = contract("ad,vd->av", axes, lv)
    rp = contract("ad,vd->av", axes, rv)
    lmin = jnp.min(jnp.where(lvv[None, :], lp, jnp.inf), axis=-1)
    lmax = jnp.max(jnp.where(lvv[None, :], lp, -jnp.inf), axis=-1)
    rmin = jnp.min(jnp.where(rvv[None, :], rp, jnp.inf), axis=-1)
    rmax = jnp.max(jnp.where(rvv[None, :], rp, -jnp.inf), axis=-1)
    positive_gap, negative_gap = lmin - rmax, rmin - lmax
    gap = jnp.where(valid, jnp.maximum(positive_gap, negative_gap), -jnp.inf)
    selected = jnp.argmax(gap)
    normal = jnp.where(
        positive_gap[selected] >= negative_gap[selected], axes[selected], -axes[selected]
    )
    left_projection = lv @ normal
    right_projection = rv @ normal
    left_witness = lv[jnp.argmin(jnp.where(lvv, left_projection, jnp.inf))]
    right_witness = rv[jnp.argmax(jnp.where(rvv, right_projection, -jnp.inf))]
    point = 0.5 * (left_witness + right_witness)
    margin = jnp.min(jnp.where(valid, jnp.abs(gap - gap[selected]), jnp.inf))
    finite = (
        jnp.any(valid)
        & jnp.all(jnp.isfinite(normal))
        & jnp.isfinite(gap[selected])
        & jnp.all(jnp.isfinite(point))
    )
    return (
        normal,
        gap[selected],
        point,
        selected.astype(jnp.int32),
        margin,
        finite,
        jnp.sum(valid, dtype=jnp.int32),
    )


__all__ = [
    "ConvexContactResult",
    "ConvexShapeSetPlan",
    "PreparedConvexShapeSet",
    "ConvexShapePlan",
    "PreparedConvexShape",
    "convex_sat_contact",
]
