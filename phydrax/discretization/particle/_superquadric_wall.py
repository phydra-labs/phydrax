#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._pair_state import particle_wall_interaction_keys
from ._rigid_body import quaternion_rotation_matrix, RigidBodyKinematics
from ._rigid_contact import RigidContactGeometry
from ._superquadric_contact import (
    _principal_curvature,
    _support_local,
    PreparedSuperquadricSet,
)
from ._triangle_wall import PreparedTriangleWall


class SuperquadricTriangleContactPlan(StrictModule, NonTrainableState):
    iterations: int = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    edge_rounding_radius: float = eqx.field(static=True)
    vertex_rounding_radius: float = eqx.field(static=True)
    interaction_range: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        iterations: int = 24,
        relaxation: float = 0.5,
        residual_tolerance: float = 1.0e-6,
        edge_rounding_radius: float = 0.0,
        vertex_rounding_radius: float = 0.0,
        interaction_range: float = 0.0,
    ):
        count = int(iterations)
        relax = float(relaxation)
        tolerance = float(residual_tolerance)
        edge = float(edge_rounding_radius)
        vertex = float(vertex_rounding_radius)
        reach = float(interaction_range)
        if (
            count <= 0
            or not 0.0 < relax <= 1.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or not np.isfinite(edge)
            or edge < 0.0
            or not np.isfinite(vertex)
            or vertex < 0.0
            or not np.isfinite(reach)
            or reach < 0.0
        ):
            raise ValueError("Superquadric wall contact controls are invalid.")
        self.iterations = count
        self.relaxation = relax
        self.residual_tolerance = tolerance
        self.edge_rounding_radius = edge
        self.vertex_rounding_radius = vertex
        self.interaction_range = reach
        self.plan_id = canonical_fingerprint(
            {
                "kind": "superquadric-triangle-contact-plan",
                "iterations": count,
                "relaxation": relax,
                "residual_tolerance": tolerance,
                "edge_rounding_radius": edge,
                "vertex_rounding_radius": vertex,
                "interaction_range": reach,
            }
        )


class SuperquadricWallContactResult(StrictModule):
    geometry: RigidContactGeometry
    owner_indices: Array
    triangle_indices: Array
    particle_material: Array
    wall_material: Array
    feature_kind: Array
    feature_id: Array
    witness_residual: Array
    feature_tie_margin: Array
    curvature_valid: Array
    broadphase_valid: Array
    plan_id: str = eqx.field(static=True)
    wall_id: str = eqx.field(static=True)


def _segment_closest(point, start, end):
    direction = end - start
    denominator = jnp.sum(direction * direction, axis=-1)
    parameter = jnp.sum((point - start) * direction, axis=-1) / jnp.maximum(
        denominator, 1.0e-30
    )
    parameter = jnp.clip(parameter, 0.0, 1.0)
    return start + parameter[:, None] * direction, parameter


def _support(rotation, axes, first, second, direction):
    local_direction = contract("...ji,...j->...i", rotation, direction)
    local = jax.vmap(_support_local)(local_direction, axes, first, second)
    return contract("...ij,...j->...i", rotation, local)


def _barycentric(point, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = jnp.sum(v0 * v0, axis=-1)
    d01 = jnp.sum(v0 * v1, axis=-1)
    d11 = jnp.sum(v1 * v1, axis=-1)
    d20 = jnp.sum(v2 * v0, axis=-1)
    d21 = jnp.sum(v2 * v1, axis=-1)
    denominator = d00 * d11 - d01 * d01
    beta = (d11 * d20 - d01 * d21) / denominator
    gamma = (d00 * d21 - d01 * d20) / denominator
    alpha = 1.0 - beta - gamma
    return jnp.stack((alpha, beta, gamma), axis=-1)


def _feature_candidate(
    plan,
    center,
    rotation,
    axes,
    first,
    second,
    start,
    end,
    initial_direction,
    is_vertex,
):
    axis = initial_direction / jnp.maximum(
        jnp.linalg.norm(initial_direction, axis=-1, keepdims=True), 1.0e-30
    )

    def iteration(_, current):
        body_witness = center + _support(rotation, axes, first, second, -current)
        wall_witness, _ = _segment_closest(body_witness, start, end)
        wall_witness = jnp.where(is_vertex[:, None], start, wall_witness)
        separation = body_witness - wall_witness
        axial = jnp.sum(separation * current, axis=-1, keepdims=True)
        tangent = separation - axial * current
        scale = jnp.maximum(jnp.linalg.norm(separation, axis=-1, keepdims=True), 1.0e-30)
        candidate = current + plan.relaxation * tangent / scale
        return candidate / jnp.maximum(
            jnp.linalg.norm(candidate, axis=-1, keepdims=True), 1.0e-30
        )

    axis = jax.lax.fori_loop(0, plan.iterations, iteration, axis)
    body_witness = center + _support(rotation, axes, first, second, -axis)
    wall_witness, parameter = _segment_closest(body_witness, start, end)
    wall_witness = jnp.where(is_vertex[:, None], start, wall_witness)
    separation = body_witness - wall_witness
    gap = jnp.sum(separation * axis, axis=-1)
    tangent = separation - gap[:, None] * axis
    residual = jnp.linalg.norm(tangent, axis=-1) / jnp.maximum(
        jnp.min(axes, axis=-1), 1.0e-30
    )
    return axis, gap, body_witness, wall_witness, parameter, residual


def superquadric_triangle_contact_geometry(
    plan: SuperquadricTriangleContactPlan,
    shapes: PreparedSuperquadricSet,
    kinematics: RigidBodyKinematics,
    wall: PreparedTriangleWall,
    /,
) -> SuperquadricWallContactResult:
    if not isinstance(plan, SuperquadricTriangleContactPlan):
        raise TypeError("plan must be SuperquadricTriangleContactPlan.")
    if not isinstance(shapes, PreparedSuperquadricSet):
        raise TypeError("shapes must be PreparedSuperquadricSet.")
    if not isinstance(kinematics, RigidBodyKinematics):
        raise TypeError("kinematics must be RigidBodyKinematics.")
    if not isinstance(wall, PreparedTriangleWall):
        raise TypeError("wall must be PreparedTriangleWall.")
    particle_count = shapes.particles.capacity
    face_count = wall.face_count
    owner = jnp.repeat(jnp.arange(particle_count, dtype=jnp.int32), face_count)
    face = jnp.tile(jnp.arange(face_count, dtype=jnp.int32), particle_count)
    center = kinematics.position[owner]
    rotation = quaternion_rotation_matrix(kinematics.orientation[owner])
    axes = shapes.semi_axes[owner]
    first = shapes.first_blockiness[owner]
    second = shapes.second_blockiness[owner]
    vertices = wall.face_vertices[face]
    a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    face_normal = wall.normals[face]
    signed_center = jnp.sum((center - a) * face_normal, axis=-1)
    oriented_face_normal = jnp.where(
        (wall.plan.two_sided & (signed_center < 0.0))[:, None],
        -face_normal,
        face_normal,
    )
    body_face = center + _support(rotation, axes, first, second, -oriented_face_normal)
    face_gap = jnp.sum((body_face - a) * oriented_face_normal, axis=-1)
    wall_face = body_face - face_gap[:, None] * oriented_face_normal
    barycentric = _barycentric(wall_face, a, b, c)
    face_inside = jnp.all(barycentric >= -plan.residual_tolerance, axis=-1)
    face_front = wall.plan.two_sided | (signed_center >= -plan.residual_tolerance)
    face_valid = face_inside & face_front
    face_residual = jnp.maximum(-jnp.min(barycentric, axis=-1), 0.0)

    starts = jnp.stack((a, b, c), axis=1)
    ends = jnp.stack((b, c, a), axis=1)
    center_repeated = jnp.repeat(center, 3, axis=0)
    rotation_repeated = jnp.repeat(rotation, 3, axis=0)
    axes_repeated = jnp.repeat(axes, 3, axis=0)
    first_repeated = jnp.repeat(first, 3, axis=0)
    second_repeated = jnp.repeat(second, 3, axis=0)
    starts_flat = starts.reshape((-1, 3))
    ends_flat = ends.reshape((-1, 3))
    center_edge, _ = _segment_closest(center_repeated, starts_flat, ends_flat)
    edge_direction = center_repeated - center_edge
    edge_axis, edge_gap, edge_body, edge_wall, edge_parameter, edge_residual = (
        _feature_candidate(
            plan,
            center_repeated,
            rotation_repeated,
            axes_repeated,
            first_repeated,
            second_repeated,
            starts_flat,
            ends_flat,
            edge_direction,
            jnp.zeros((center_repeated.shape[0],), dtype=bool),
        )
    )
    edge_axis = edge_axis.reshape((-1, 3, 3))
    edge_gap = edge_gap.reshape((-1, 3))
    edge_body = edge_body.reshape((-1, 3, 3))
    edge_wall = edge_wall.reshape((-1, 3, 3))
    edge_parameter = edge_parameter.reshape((-1, 3))
    edge_residual = edge_residual.reshape((-1, 3))
    edge_front = wall.plan.two_sided | (
        jnp.sum(edge_axis * face_normal[:, None, :], axis=-1) >= -plan.residual_tolerance
    )
    edge_interior = (edge_parameter > plan.residual_tolerance) & (
        edge_parameter < 1.0 - plan.residual_tolerance
    )
    edge_owned = (
        wall.edge_owner_triangle_ids[wall.edge_ids[face]] == wall.triangle_ids[face, None]
    )
    edge_valid = (
        edge_front
        & edge_interior
        & edge_owned
        & (edge_residual <= plan.residual_tolerance)
    )

    vertex_points = vertices.reshape((-1, 3))
    center_vertex = jnp.repeat(center, 3, axis=0)
    vertex_direction = center_vertex - vertex_points
    vertex_axis, vertex_gap, vertex_body, vertex_wall, _, vertex_residual = (
        _feature_candidate(
            plan,
            center_vertex,
            rotation_repeated,
            axes_repeated,
            first_repeated,
            second_repeated,
            vertex_points,
            vertex_points + jnp.asarray([1.0, 0.0, 0.0]),
            vertex_direction,
            jnp.ones((center_vertex.shape[0],), dtype=bool),
        )
    )
    vertex_axis = vertex_axis.reshape((-1, 3, 3))
    vertex_gap = vertex_gap.reshape((-1, 3))
    vertex_body = vertex_body.reshape((-1, 3, 3))
    vertex_wall = vertex_wall.reshape((-1, 3, 3))
    vertex_residual = vertex_residual.reshape((-1, 3))
    vertex_front = wall.plan.two_sided | (
        jnp.sum(vertex_axis * face_normal[:, None, :], axis=-1)
        >= -plan.residual_tolerance
    )
    vertex_indices = wall.plan.triangles[face]
    vertex_owned = (
        wall.vertex_owner_triangle_ids[vertex_indices] == wall.triangle_ids[face, None]
    )
    vertex_valid = (
        vertex_front & vertex_owned & (vertex_residual <= plan.residual_tolerance)
    )

    gaps = jnp.concatenate((face_gap[:, None], edge_gap, vertex_gap), axis=1)
    valid_candidates = jnp.concatenate(
        (face_valid[:, None], edge_valid, vertex_valid), axis=1
    )
    candidate_gaps = jnp.where(valid_candidates, gaps, jnp.inf)
    selected = jnp.argmin(candidate_gaps, axis=1)
    sorted_gaps = jnp.sort(candidate_gaps, axis=1)
    tie_margin = sorted_gaps[:, 1] - sorted_gaps[:, 0]
    normals = jnp.concatenate(
        (oriented_face_normal[:, None, :], edge_axis, vertex_axis), axis=1
    )
    body_witnesses = jnp.concatenate(
        (body_face[:, None, :], edge_body, vertex_body), axis=1
    )
    wall_witnesses = jnp.concatenate(
        (wall_face[:, None, :], edge_wall, vertex_wall), axis=1
    )
    residuals = jnp.concatenate(
        (face_residual[:, None], edge_residual, vertex_residual), axis=1
    )
    normal = jnp.take_along_axis(normals, selected[:, None, None], axis=1)[:, 0]
    body_witness = jnp.take_along_axis(body_witnesses, selected[:, None, None], axis=1)[
        :, 0
    ]
    wall_witness = jnp.take_along_axis(wall_witnesses, selected[:, None, None], axis=1)[
        :, 0
    ]
    gap = jnp.take_along_axis(gaps, selected[:, None], axis=1)[:, 0]
    residual = jnp.take_along_axis(residuals, selected[:, None], axis=1)[:, 0]
    feature_kind = jnp.where(selected == 0, 0, jnp.where(selected <= 3, 1, 2))
    edge_selected = jnp.clip(selected - 1, 0, 2)
    vertex_selected = jnp.clip(selected - 4, 0, 2)
    edge_id = wall.edge_ids[face, edge_selected]
    selected_vertex_index = wall.plan.triangles[face, vertex_selected]
    vertex_id = wall.plan.vertex_ids[selected_vertex_index]
    feature_id = jnp.where(
        selected == 0,
        wall.triangle_ids[face],
        jnp.where(selected <= 3, edge_id, vertex_id),
    )
    object_id = jnp.where(selected == 0, wall.triangle_ids[face], 0)
    broadphase_distance = jnp.linalg.norm(
        jnp.maximum(
            jnp.maximum(wall.aabb_lower[face] - center, center - wall.aabb_upper[face]),
            0.0,
        ),
        axis=-1,
    )
    broadphase = broadphase_distance <= (
        shapes.bounding_radii[owner] + plan.interaction_range
    )
    active = shapes.particles.active_mask[owner]
    selected_valid = jnp.take_along_axis(valid_candidates, selected[:, None], axis=1)[
        :, 0
    ]
    finite = (
        jnp.all(jnp.isfinite(normal), axis=-1)
        & jnp.isfinite(gap)
        & jnp.isfinite(residual)
        & jnp.isfinite(tie_margin)
    )
    valid = active & broadphase & selected_valid & finite
    local_body = contract("...ji,...j->...i", rotation, body_witness - center)
    body_curvature, body_curvature_valid, curvature_margin = jax.vmap(
        _principal_curvature
    )(local_body, axes, first, second)
    body_mean_curvature = 0.5 * jnp.sum(body_curvature, axis=-1)
    feature_radius = jnp.where(
        feature_kind == 0,
        jnp.inf,
        jnp.where(
            feature_kind == 1,
            plan.edge_rounding_radius,
            plan.vertex_rounding_radius,
        ),
    )
    rounded = (feature_kind == 0) | (feature_radius > 0.0)
    wall_curvature = jnp.where(
        feature_kind == 0, 0.0, 1.0 / jnp.maximum(feature_radius, 1.0e-30)
    )
    curvature_sum = body_mean_curvature + wall_curvature
    effective_radius = jnp.where(curvature_sum > 0.0, 1.0 / curvature_sum, 0.0)
    curvature_valid = rounded & body_curvature_valid & (effective_radius > 0.0)
    contact_point = 0.5 * (body_witness + wall_witness)
    owner_arm = contact_point - center
    contact_velocity = kinematics.velocity[owner] + jnp.cross(
        kinematics.angular_velocity[owner], owner_arm
    )
    normal_velocity = jnp.sum(contact_velocity * normal, axis=-1)
    tangent_velocity = contact_velocity - normal_velocity[:, None] * normal
    key = particle_wall_interaction_keys(
        shapes.particles.particle_ids[owner],
        object_id,
        feature_kind,
        feature_id,
        valid,
    )
    geometry = RigidContactGeometry(
        jnp.where(valid[:, None], normal, 0.0),
        jnp.where(valid, gap, 0.0),
        jnp.where(valid, jnp.maximum(-gap, 0.0), 0.0),
        jnp.where(valid, effective_radius, 0.0),
        jnp.where(valid[:, None], contact_point, 0.0),
        jnp.where(valid[:, None], owner_arm, 0.0),
        jnp.zeros_like(owner_arm),
        jnp.where(valid[:, None], body_witness - center, 0.0),
        jnp.where(valid[:, None], wall_witness - contact_point, 0.0),
        jnp.where(valid[:, None], contact_velocity, 0.0),
        jnp.where(valid, normal_velocity, 0.0),
        jnp.where(valid[:, None], tangent_velocity, 0.0),
        kinematics.angular_velocity[owner],
        jnp.zeros_like(kinematics.angular_velocity[owner]),
        key,
        selected.astype(jnp.int32),
        selected.astype(jnp.int32),
        valid,
        (~selected_valid).astype(jnp.int32),
        jnp.minimum(
            jnp.minimum(tie_margin, plan.residual_tolerance - residual),
            curvature_margin,
        ),
        ~jnp.any(active & broadphase & ~valid),
        "rigid-contact:superquadric-triangle-wall",
    )
    return SuperquadricWallContactResult(
        geometry,
        owner,
        face,
        shapes.material_ids[owner],
        wall.plan.triangle_material[face],
        feature_kind,
        feature_id,
        residual,
        tie_margin,
        curvature_valid,
        broadphase,
        plan.plan_id,
        wall.prepared_id,
    )


__all__ = [
    "SuperquadricTriangleContactPlan",
    "SuperquadricWallContactResult",
    "superquadric_triangle_contact_geometry",
]
