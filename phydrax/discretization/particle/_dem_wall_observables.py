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
from ._dem_contact import DEMContactResponse
from ._superquadric_wall import SuperquadricWallContactResult
from ._triangle_wall import (
    PreparedTriangleWall,
    TriangleWallContactResult,
    TriangleWallPlan,
)


class DEMWallFacetObservables(StrictModule):
    facet_force: Array
    facet_torque: Array
    normal_traction: Array
    tangential_traction: Array
    contact_count: Array
    mechanical_power: Array
    heat_rate: Array
    total_force: Array
    total_torque: Array
    force_residual: Array
    successful: Array
    wall_id: str = eqx.field(static=True)


def evaluate_wall_facet_observables(
    wall: PreparedTriangleWall,
    geometry: TriangleWallContactResult | SuperquadricWallContactResult,
    contact: DEMContactResponse,
    /,
    *,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
    wall_velocity: ArrayLike | None = None,
    heat_rate: ArrayLike | None = None,
) -> DEMWallFacetObservables:
    if not isinstance(wall, PreparedTriangleWall):
        raise TypeError("wall must be a PreparedTriangleWall.")
    if not isinstance(
        geometry, (TriangleWallContactResult, SuperquadricWallContactResult)
    ):
        raise TypeError("geometry must be a supported triangle-wall contact result.")
    if not isinstance(contact, DEMContactResponse):
        raise TypeError("contact must be a DEMContactResponse.")
    faces = wall.face_count
    face = geometry.triangle_indices.astype(jnp.int32)
    valid = geometry.geometry.valid & contact.active
    reaction = jnp.where(valid[:, None], -contact.pair_force, 0.0)
    point = geometry.geometry.contact_point
    reference = jnp.asarray(reference_point, dtype=point.dtype)
    if reference.shape != (3,):
        raise ValueError("reference_point must be a three-vector.")
    edge_torque = jnp.cross(point - reference, reaction) - contact.right_torque
    facet_force = jnp.zeros((faces, 3), dtype=point.dtype).at[face].add(reaction)
    facet_torque = (
        jnp.zeros((faces, 3), dtype=point.dtype)
        .at[face]
        .add(jnp.where(valid[:, None], edge_torque, 0.0))
    )
    count = jnp.zeros((faces,), dtype=jnp.int32).at[face].add(valid.astype(jnp.int32))
    vertices = wall.face_vertices
    area = 0.5 * jnp.linalg.norm(
        jnp.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=-1,
    )
    normal_force = jnp.sum(facet_force * wall.normals, axis=-1)
    tangent_force = facet_force - normal_force[:, None] * wall.normals
    normal_traction = normal_force / area
    tangential_traction = jnp.linalg.norm(tangent_force, axis=-1) / area
    velocity = (
        jnp.zeros_like(point)
        if wall_velocity is None
        else jnp.asarray(wall_velocity, dtype=point.dtype)
    )
    if velocity.shape == (3,):
        velocity = jnp.broadcast_to(velocity, point.shape)
    if velocity.shape != point.shape:
        raise ValueError("wall_velocity must have contact shape or be a three-vector.")
    edge_power = jnp.sum(reaction * velocity, axis=-1)
    power = (
        jnp.zeros((faces,), dtype=point.dtype)
        .at[face]
        .add(jnp.where(valid, edge_power, 0.0))
    )
    edge_heat = (
        jnp.zeros((face.shape[0],), dtype=point.dtype)
        if heat_rate is None
        else jnp.asarray(heat_rate, dtype=point.dtype)
    )
    if edge_heat.shape != face.shape:
        raise ValueError("heat_rate must have contact-edge shape.")
    facet_heat = (
        jnp.zeros((faces,), dtype=point.dtype)
        .at[face]
        .add(jnp.where(valid, edge_heat, 0.0))
    )
    total_force = jnp.sum(facet_force, axis=0)
    total_torque = jnp.sum(facet_torque, axis=0)
    force_residual = total_force - jnp.sum(reaction, axis=0)
    successful = (
        contact.successful
        & geometry.geometry.successful
        & jnp.all(jnp.isfinite(facet_force))
        & jnp.all(jnp.isfinite(facet_torque))
        & jnp.all(jnp.isfinite(normal_traction))
        & jnp.all(jnp.isfinite(tangential_traction))
        & (
            jnp.linalg.norm(force_residual)
            <= 64.0
            * jnp.finfo(point.dtype).eps
            * jnp.maximum(jnp.linalg.norm(total_force), 1.0)
        )
    )
    return DEMWallFacetObservables(
        facet_force,
        facet_torque,
        normal_traction,
        tangential_traction,
        count,
        power,
        facet_heat,
        total_force,
        total_torque,
        force_residual,
        successful,
        wall.prepared_id,
    )


class DEMWearState(StrictModule):
    wear_depth: Array
    cumulative_removed_volume: Array
    accepted_steps: Array


class DEMWearEvaluation(StrictModule):
    wear_rate: Array
    removed_volume_rate: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DEMWearStepResult(StrictModule):
    candidate_state: DEMWearState
    accepted_state: DEMWearState
    evaluation: DEMWearEvaluation
    successful: Array


class FinnieWearPlan(StrictModule, NonTrainableState):
    wear_coefficient: Array
    hardness: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        wear_coefficient: ArrayLike,
        hardness: ArrayLike,
        /,
        *,
        plan_id: str | None = None,
    ):
        coefficient = np.asarray(wear_coefficient)
        hardness_ = np.asarray(hardness)
        if (
            coefficient.ndim != 2
            or coefficient.shape[0] != coefficient.shape[1]
            or hardness_.shape != coefficient.shape
            or not np.array_equal(coefficient, coefficient.T)
            or not np.array_equal(hardness_, hardness_.T)
            or np.any(~np.isfinite(coefficient))
            or np.any(coefficient < 0.0)
            or np.any(~np.isfinite(hardness_))
            or np.any(hardness_ <= 0.0)
        ):
            raise ValueError(
                "Finnie wear parameters must be symmetric valid pair tables."
            )
        generated = canonical_fingerprint(
            {
                "kind": "finnie-wear-plan",
                "values": array_tree_fingerprint(
                    {"wear_coefficient": coefficient, "hardness": hardness_}
                ),
            }
        )
        self.wear_coefficient = jnp.asarray(coefficient)
        self.hardness = jnp.asarray(hardness_)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def initialize(self, wall: PreparedTriangleWall, dtype=None, /) -> DEMWearState:
        selected_dtype = wall.face_vertices.dtype if dtype is None else dtype
        return DEMWearState(
            jnp.zeros((wall.face_count,), dtype=selected_dtype),
            jnp.zeros((), dtype=selected_dtype),
            jnp.zeros((), dtype=jnp.int32),
        )

    def evaluate(
        self,
        wall: PreparedTriangleWall,
        geometry: TriangleWallContactResult | SuperquadricWallContactResult,
        contact: DEMContactResponse,
        /,
    ) -> DEMWearEvaluation:
        if not isinstance(
            geometry, (TriangleWallContactResult, SuperquadricWallContactResult)
        ):
            raise TypeError("geometry must be a supported triangle-wall contact result.")
        face = geometry.triangle_indices.astype(jnp.int32)
        valid = geometry.geometry.valid & contact.active
        particle_material = geometry.particle_material.astype(jnp.int32)
        wall_material = geometry.wall_material.astype(jnp.int32)
        coefficient = self.wear_coefficient[particle_material, wall_material]
        hardness = self.hardness[particle_material, wall_material]
        normal_speed = jnp.abs(geometry.geometry.normal_velocity)
        tangent_speed = jnp.linalg.norm(geometry.geometry.tangential_velocity, axis=-1)
        speed = jnp.sqrt(normal_speed**2 + tangent_speed**2)
        angle = jnp.arctan2(normal_speed, jnp.maximum(tangent_speed, 1.0e-30))
        sine = jnp.sin(angle)
        cosine = jnp.cos(angle)
        angular = jnp.where(
            jnp.tan(angle) <= (1.0 / 3.0),
            jnp.maximum(jnp.sin(2.0 * angle) - 3.0 * sine**2, 0.0),
            cosine**2 / 3.0,
        )
        force = jnp.linalg.norm(contact.pair_force, axis=-1)
        edge_volume_rate = jnp.where(
            valid,
            coefficient * force * speed * angular / hardness,
            0.0,
        )
        vertices = wall.face_vertices
        area = 0.5 * jnp.linalg.norm(
            jnp.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=-1,
        )
        volume_rate = (
            jnp.zeros((wall.face_count,), dtype=force.dtype)
            .at[face]
            .add(edge_volume_rate)
        )
        wear_rate = volume_rate / area
        successful = (
            contact.successful
            & geometry.geometry.successful
            & jnp.all(jnp.isfinite(wear_rate))
            & jnp.all(wear_rate >= 0.0)
        )
        return DEMWearEvaluation(wear_rate, volume_rate, successful, self.plan_id)

    def step(
        self,
        wall: PreparedTriangleWall,
        geometry: TriangleWallContactResult | SuperquadricWallContactResult,
        contact: DEMContactResponse,
        state: DEMWearState,
        step_size: Array,
        /,
    ) -> DEMWearStepResult:
        evaluation = self.evaluate(wall, geometry, contact)
        dt = jnp.asarray(step_size, dtype=state.wear_depth.dtype)
        successful = evaluation.successful & jnp.isfinite(dt) & (dt > 0.0)
        candidate = DEMWearState(
            state.wear_depth + dt * evaluation.wear_rate,
            state.cumulative_removed_volume
            + dt * jnp.sum(evaluation.removed_volume_rate),
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
        )
        accepted = eqx.tree_at(
            lambda value: value.wear_depth,
            candidate,
            jnp.where(successful, candidate.wear_depth, state.wear_depth),
        )
        accepted = eqx.tree_at(
            lambda value: value.cumulative_removed_volume,
            accepted,
            jnp.where(
                successful,
                candidate.cumulative_removed_volume,
                state.cumulative_removed_volume,
            ),
        )
        accepted = eqx.tree_at(
            lambda value: value.accepted_steps,
            accepted,
            jnp.where(successful, candidate.accepted_steps, state.accepted_steps),
        )
        return DEMWearStepResult(candidate, accepted, evaluation, successful)


class DEMWearCommitResult(StrictModule):
    wall: TriangleWallPlan
    state: DEMWearState
    quality_margin: Array
    successful: Array


def commit_triangle_wear(
    wall: PreparedTriangleWall,
    state: DEMWearState,
    /,
    *,
    maximum_edge_fraction: float = 0.1,
) -> DEMWearCommitResult:
    fraction = float(maximum_edge_fraction)
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("maximum_edge_fraction must be finite and positive.")
    triangles = wall.plan.triangles
    vertex_count = wall.plan.vertices.shape[0]
    vertex_depth = jnp.zeros((vertex_count,), dtype=state.wear_depth.dtype)
    vertex_count_values = jnp.zeros((vertex_count,), dtype=state.wear_depth.dtype)
    vertex_normal = jnp.zeros((vertex_count, 3), dtype=state.wear_depth.dtype)
    for local in range(3):
        vertices = triangles[:, local]
        vertex_depth = vertex_depth.at[vertices].add(state.wear_depth)
        vertex_count_values = vertex_count_values.at[vertices].add(1.0)
        vertex_normal = vertex_normal.at[vertices].add(wall.normals)
    vertex_depth = vertex_depth / jnp.maximum(vertex_count_values, 1.0)
    vertex_normal = vertex_normal / jnp.maximum(
        jnp.linalg.norm(vertex_normal, axis=-1, keepdims=True), 1.0e-30
    )
    candidate_vertices = wall.plan.vertices - vertex_depth[:, None] * vertex_normal
    face_vertices = candidate_vertices[triangles]
    edge_lengths = jnp.stack(
        (
            jnp.linalg.norm(face_vertices[:, 1] - face_vertices[:, 0], axis=-1),
            jnp.linalg.norm(face_vertices[:, 2] - face_vertices[:, 1], axis=-1),
            jnp.linalg.norm(face_vertices[:, 0] - face_vertices[:, 2], axis=-1),
        ),
        axis=-1,
    )
    minimum_edge = jnp.min(edge_lengths)
    maximum_depth = jnp.max(state.wear_depth)
    quality_margin = fraction * minimum_edge - maximum_depth
    finite = jnp.all(jnp.isfinite(candidate_vertices)) & (quality_margin >= 0.0)
    if bool(np.asarray(finite)):
        accepted_wall = TriangleWallPlan(
            candidate_vertices,
            wall.plan.triangles,
            wall.plan.triangle_material,
            two_sided=wall.plan.two_sided,
        )
        accepted_state = DEMWearState(
            jnp.zeros_like(state.wear_depth),
            state.cumulative_removed_volume,
            state.accepted_steps,
        )
    else:
        accepted_wall = wall.plan
        accepted_state = state
    return DEMWearCommitResult(accepted_wall, accepted_state, quality_margin, finite)


__all__ = [
    "DEMWallFacetObservables",
    "DEMWearCommitResult",
    "DEMWearEvaluation",
    "DEMWearState",
    "DEMWearStepResult",
    "FinnieWearPlan",
    "commit_triangle_wear",
    "evaluate_wall_facet_observables",
]
