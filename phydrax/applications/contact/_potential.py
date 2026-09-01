#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact import (
    ContactCandidateEpoch,
    evaluate_contact_stencils,
    point_edge_distance,
    point_point_distance,
    PreparedCollisionScene,
)
from ._barrier import physical_clamped_log_barrier


class ConvergentContactPotentialPlan(StrictModule, NonTrainableState):
    """Area-weighted improved-max contact potential with a physical barrier."""

    activation_distance: float = eqx.field(static=True)
    stiffness: Array
    geometry_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        activation_distance: float,
        stiffness: ArrayLike,
        /,
        *,
        geometry_tolerance: float = 1.0e-12,
        plan_id: str | None = None,
    ):
        activation = float(activation_distance)
        tolerance = float(geometry_tolerance)
        stiffness_ = jnp.asarray(stiffness)
        if not np.isfinite(activation) or activation <= 0.0:
            raise ValueError("activation_distance must be finite and positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("geometry_tolerance must be finite and positive.")
        if (
            stiffness_.shape != ()
            or not bool(jnp.isfinite(stiffness_))
            or stiffness_ <= 0.0
        ):
            raise ValueError("stiffness must be one positive finite scalar.")
        generated = canonical_fingerprint(
            {
                "kind": "convergent-contact-potential-plan",
                "activation_distance": activation.hex(),
                "stiffness": float(stiffness_).hex(),
                "geometry_tolerance": tolerance.hex(),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty or None.")
        self.activation_distance = activation
        self.stiffness = stiffness_
        self.geometry_tolerance = tolerance
        self.plan_id = identifier

    def prepare(
        self, scene: PreparedCollisionScene, /
    ) -> PreparedConvergentContactPotential:
        return PreparedConvergentContactPotential(self, scene)


class ContactPotentialEvaluation(StrictModule):
    energy: Array
    surface_force: Array
    state_force: PyTree[Array]
    minimum_gap: Array
    active_contacts: Array
    action_reaction_residual: Array
    moment_residual: Array
    complementarity_defect: Array
    minimum_feature_margin: Array
    finite: Array
    nonnegative_energy: Array
    successful: Array
    epoch_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PreparedConvergentContactPotential(StrictModule, NonTrainableState):
    """Prepared convergent node potential plus mollified edge barrier."""

    plan: ConvergentContactPotentialPlan
    scene: PreparedCollisionScene
    face_edges: Array
    edge_face_count: Array
    vertex_edge_count: Array
    vertex_face_count: Array
    internal_vertex: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: ConvergentContactPotentialPlan, scene: PreparedCollisionScene, /
    ):
        if not isinstance(plan, ConvergentContactPotentialPlan):
            raise TypeError("plan must be ConvergentContactPotentialPlan.")
        if not isinstance(scene, PreparedCollisionScene):
            raise TypeError("scene must be PreparedCollisionScene.")
        edges = np.asarray(scene.edges, dtype=np.int32)
        faces = np.asarray(scene.faces, dtype=np.int32)
        vertex_edge_count = np.zeros((scene.vertex_count,), dtype=np.int32)
        for edge in edges:
            vertex_edge_count[edge] += 1
        edge_lookup = {
            tuple(sorted(edge.tolist())): index for index, edge in enumerate(edges)
        }
        edge_face_count = np.zeros((scene.edge_count,), dtype=np.int32)
        vertex_face_count = np.zeros((scene.vertex_count,), dtype=np.int32)
        face_edges = np.empty((faces.shape[0], 3), dtype=np.int32)
        for face_index, face in enumerate(faces):
            vertex_face_count[face] += 1
            local_edges = ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
            for local_index, edge in enumerate(local_edges):
                edge_index = edge_lookup[tuple(sorted((int(edge[0]), int(edge[1]))))]
                face_edges[face_index, local_index] = edge_index
                edge_face_count[edge_index] += 1
        if scene.ambient_dimension == 2:
            internal_vertex = vertex_edge_count == 2
        else:
            boundary_edge = edge_face_count == 1
            boundary_vertex = np.zeros((scene.vertex_count,), dtype=bool)
            if np.any(boundary_edge):
                boundary_vertex[np.unique(edges[boundary_edge])] = True
            internal_vertex = ~boundary_vertex
        self.plan = plan
        self.scene = scene
        self.face_edges = jnp.asarray(face_edges, dtype=jnp.int32)
        self.edge_face_count = jnp.asarray(edge_face_count, dtype=jnp.int32)
        self.vertex_edge_count = jnp.asarray(vertex_edge_count, dtype=jnp.int32)
        self.vertex_face_count = jnp.asarray(vertex_face_count, dtype=jnp.int32)
        self.internal_vertex = jnp.asarray(internal_vertex)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-convergent-contact-potential",
                "plan": plan.plan_id,
                "scene": scene.scene_id,
            }
        )

    def _rest_positions(self) -> Array:
        return jnp.concatenate(
            tuple(surface.rest_positions for surface in self.scene.surfaces), axis=0
        )

    def _measures(self, rest_positions: Array, /) -> tuple[Array, Array]:
        edges = self.scene.edges
        edge_vectors = rest_positions[edges[:, 1]] - rest_positions[edges[:, 0]]
        edge_lengths = jnp.sqrt(jnp.sum(edge_vectors * edge_vectors, axis=-1))
        vertex_measure = jnp.zeros((self.scene.vertex_count,), dtype=rest_positions.dtype)
        if self.scene.ambient_dimension == 2 or self.scene.face_count == 0:
            vertex_measure = vertex_measure.at[edges[:, 0]].add(0.5 * edge_lengths)
            vertex_measure = vertex_measure.at[edges[:, 1]].add(0.5 * edge_lengths)
            return vertex_measure, edge_lengths
        faces = self.scene.faces
        selected = rest_positions[faces]
        cross = jnp.cross(
            selected[:, 1] - selected[:, 0], selected[:, 2] - selected[:, 0]
        )
        face_area = 0.5 * jnp.sqrt(jnp.sum(cross * cross, axis=-1))
        for local in range(3):
            vertex_measure = vertex_measure.at[faces[:, local]].add(face_area / 3.0)
        edge_measure = jnp.zeros((self.scene.edge_count,), dtype=rest_positions.dtype)
        for local in range(3):
            edge_measure = edge_measure.at[self.face_edges[:, local]].add(face_area / 3.0)
        return vertex_measure, edge_measure

    def _barrier(self, squared: Array, separation: Array, valid: Array, /) -> Array:
        dtype = squared.dtype
        d_hat = jnp.asarray(self.plan.activation_distance, dtype=dtype)
        activation_squared = (separation + d_hat) ** 2
        safe_actual = jnp.maximum(
            squared,
            separation * separation + jnp.finfo(dtype).tiny,
        )
        safe_squared = jnp.where(valid, safe_actual, activation_squared)
        active = valid & (squared < activation_squared)
        value = physical_clamped_log_barrier(safe_squared, d_hat, separation)
        value = jnp.where(valid & (squared <= separation * separation), jnp.inf, value)
        return jnp.where(active, value, 0.0)

    def _edge_vertex_energy(
        self,
        positions: Array,
        rest_positions: Array,
        epoch: ContactCandidateEpoch,
        vertex_measure: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        batch = epoch.edge_vertex
        if batch.capacity == 0:
            zero = jnp.asarray(0.0, dtype=positions.dtype)
            return (
                zero,
                jnp.asarray(jnp.inf, dtype=positions.dtype),
                zero.astype(jnp.int32),
            )
        evaluation = evaluate_contact_stencils(
            batch, positions, rest_positions, tolerance=self.plan.geometry_tolerance
        )
        indices = batch.vertex_indices
        query = indices[:, 0]
        base = self._barrier(
            evaluation.distance.squared_distance,
            evaluation.minimum_separation,
            evaluation.valid,
        )
        endpoints = indices[:, 1:3]
        query_points = positions[query]
        endpoint_points = positions[endpoints]
        endpoint_squared = jnp.sum(
            (query_points[:, None, :] - endpoint_points) ** 2, axis=-1
        )
        endpoint_internal = self.internal_vertex[endpoints]
        incidence = jnp.maximum(self.vertex_edge_count[endpoints], 1)
        correction = (
            self._barrier(
                endpoint_squared,
                evaluation.minimum_separation[:, None],
                evaluation.valid[:, None] & endpoint_internal,
            )
            / incidence
        )
        weight = 0.5 * vertex_measure[query]
        energy = jnp.sum(weight * (base - jnp.sum(correction, axis=-1)))
        gap = (
            jnp.sqrt(jnp.maximum(evaluation.distance.squared_distance, 0.0))
            - evaluation.minimum_separation
        )
        minimum_gap = jnp.min(jnp.where(evaluation.valid, gap, jnp.inf), initial=jnp.inf)
        return energy, minimum_gap, jnp.sum(evaluation.valid, dtype=jnp.int32)

    def _face_vertex_energy(
        self,
        positions: Array,
        rest_positions: Array,
        epoch: ContactCandidateEpoch,
        vertex_measure: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        batch = epoch.face_vertex
        if batch.capacity == 0:
            zero = jnp.asarray(0.0, dtype=positions.dtype)
            return (
                zero,
                jnp.asarray(jnp.inf, dtype=positions.dtype),
                zero.astype(jnp.int32),
                jnp.asarray(jnp.inf, dtype=positions.dtype),
            )
        evaluation = evaluate_contact_stencils(
            batch, positions, rest_positions, tolerance=self.plan.geometry_tolerance
        )
        indices = batch.vertex_indices
        query = indices[:, 0]
        face_vertices = indices[:, 1:4]
        base = self._barrier(
            evaluation.distance.squared_distance,
            evaluation.minimum_separation,
            evaluation.valid,
        )
        local_edges = (
            (face_vertices[:, 0], face_vertices[:, 1]),
            (face_vertices[:, 1], face_vertices[:, 2]),
            (face_vertices[:, 2], face_vertices[:, 0]),
        )
        face_index = batch.right_feature_ids - (
            self.scene.vertex_count + self.scene.edge_count
        )
        safe_face = jnp.clip(face_index, 0, max(self.scene.face_count - 1, 0)).astype(
            jnp.int32
        )
        edge_correction = jnp.zeros((batch.capacity,), dtype=positions.dtype)
        for local, (first, second) in enumerate(local_edges):
            distance = point_edge_distance(
                positions[query],
                positions[first],
                positions[second],
                tolerance=self.plan.geometry_tolerance,
            )
            edge_index = self.face_edges[safe_face, local]
            internal = self.edge_face_count[edge_index] == 2
            value = self._barrier(
                distance.squared_distance,
                evaluation.minimum_separation,
                evaluation.valid & distance.nondegenerate & internal,
            ) / jnp.maximum(self.edge_face_count[edge_index], 1)
            edge_correction = edge_correction + value
        vertex_correction = jnp.zeros((batch.capacity,), dtype=positions.dtype)
        for local in range(3):
            vertex = face_vertices[:, local]
            distance = point_point_distance(
                positions[query],
                positions[vertex],
                tolerance=self.plan.geometry_tolerance,
            )
            value = self._barrier(
                distance.squared_distance,
                evaluation.minimum_separation,
                evaluation.valid & self.internal_vertex[vertex],
            ) / jnp.maximum(self.vertex_face_count[vertex], 1)
            vertex_correction = vertex_correction + value
        weight = 0.5 * vertex_measure[query]
        energy = jnp.sum(weight * (base - edge_correction + vertex_correction))
        gap = (
            jnp.sqrt(jnp.maximum(evaluation.distance.squared_distance, 0.0))
            - evaluation.minimum_separation
        )
        minimum_gap = jnp.min(jnp.where(evaluation.valid, gap, jnp.inf), initial=jnp.inf)
        minimum_feature = jnp.min(
            jnp.where(evaluation.valid, evaluation.distance.feature_margin, jnp.inf),
            initial=jnp.inf,
        )
        return (
            energy,
            minimum_gap,
            jnp.sum(evaluation.valid, dtype=jnp.int32),
            minimum_feature,
        )

    def _edge_edge_energy(
        self,
        positions: Array,
        rest_positions: Array,
        epoch: ContactCandidateEpoch,
        edge_measure: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        batch = epoch.edge_edge
        if batch.capacity == 0:
            zero = jnp.asarray(0.0, dtype=positions.dtype)
            return (
                zero,
                jnp.asarray(jnp.inf, dtype=positions.dtype),
                zero.astype(jnp.int32),
                jnp.asarray(jnp.inf, dtype=positions.dtype),
            )
        evaluation = evaluate_contact_stencils(
            batch, positions, rest_positions, tolerance=self.plan.geometry_tolerance
        )
        edge_offset = self.scene.vertex_count
        left_edge = jnp.clip(
            batch.left_feature_ids - edge_offset, 0, max(self.scene.edge_count - 1, 0)
        ).astype(jnp.int32)
        right_edge = jnp.clip(
            batch.right_feature_ids - edge_offset, 0, max(self.scene.edge_count - 1, 0)
        ).astype(jnp.int32)
        value = (
            self._barrier(
                evaluation.distance.squared_distance,
                evaluation.minimum_separation,
                evaluation.valid,
            )
            * evaluation.mollifier
        )
        weight = 0.25 * (edge_measure[left_edge] + edge_measure[right_edge])
        energy = jnp.sum(weight * value)
        gap = (
            jnp.sqrt(jnp.maximum(evaluation.distance.squared_distance, 0.0))
            - evaluation.minimum_separation
        )
        minimum_gap = jnp.min(jnp.where(evaluation.valid, gap, jnp.inf), initial=jnp.inf)
        minimum_feature = jnp.min(
            jnp.where(
                evaluation.valid,
                jnp.minimum(
                    evaluation.distance.feature_margin, evaluation.mollifier_margin
                ),
                jnp.inf,
            ),
            initial=jnp.inf,
        )
        return (
            energy,
            minimum_gap,
            jnp.sum(evaluation.valid, dtype=jnp.int32),
            minimum_feature,
        )

    def energy(
        self,
        positions: ArrayLike,
        epoch: ContactCandidateEpoch,
        /,
        *,
        rest_positions: ArrayLike | None = None,
        stiffness: ArrayLike | None = None,
    ) -> Array:
        if not isinstance(epoch, ContactCandidateEpoch):
            raise TypeError("epoch must be ContactCandidateEpoch.")
        current = jnp.asarray(
            positions, dtype=self.scene.surfaces[0].precision.geometry_dtype
        )
        expected = (self.scene.vertex_count, self.scene.ambient_dimension)
        if current.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        rest = (
            self._rest_positions()
            if rest_positions is None
            else jnp.asarray(rest_positions, dtype=current.dtype)
        )
        if rest.shape != expected:
            raise ValueError(f"rest_positions must have shape {expected}.")
        scale = (
            self.plan.stiffness
            if stiffness is None
            else jnp.asarray(stiffness, dtype=current.dtype)
        )
        if scale.shape != ():
            raise ValueError("stiffness must be scalar.")
        vertex_measure, edge_measure = self._measures(rest)
        edge_vertex = self._edge_vertex_energy(current, rest, epoch, vertex_measure)[0]
        face_vertex = self._face_vertex_energy(current, rest, epoch, vertex_measure)[0]
        edge_edge = self._edge_edge_energy(current, rest, epoch, edge_measure)[0]
        value = scale.astype(current.dtype) * (edge_vertex + face_vertex + edge_edge)
        return jnp.where(
            epoch.successful, value, jnp.asarray(jnp.inf, dtype=current.dtype)
        )

    def evaluate(
        self,
        positions: ArrayLike,
        epoch: ContactCandidateEpoch,
        /,
        *,
        rest_positions: ArrayLike | None = None,
        stiffness: ArrayLike | None = None,
    ) -> ContactPotentialEvaluation:
        current = jnp.asarray(
            positions, dtype=self.scene.surfaces[0].precision.geometry_dtype
        )
        rest = (
            self._rest_positions()
            if rest_positions is None
            else jnp.asarray(rest_positions, dtype=current.dtype)
        )
        scale = (
            self.plan.stiffness
            if stiffness is None
            else jnp.asarray(stiffness, dtype=current.dtype)
        )

        def objective(value):
            return self.energy(value, epoch, rest_positions=rest, stiffness=scale)

        energy, gradient = jax.value_and_grad(objective)(current)
        surface_force = -gradient
        state_force = self.scene.pullback(surface_force)
        vertex_measure, edge_measure = self._measures(rest)
        ev = self._edge_vertex_energy(current, rest, epoch, vertex_measure)
        fv = self._face_vertex_energy(current, rest, epoch, vertex_measure)
        ee = self._edge_edge_energy(current, rest, epoch, edge_measure)
        minimum_gap = jnp.min(jnp.stack((ev[1], fv[1], ee[1])))
        active_contacts = ev[2] + fv[2] + ee[2]
        minimum_feature = jnp.min(jnp.stack((fv[3], ee[3])))
        action_reaction = jnp.sum(surface_force, axis=0)
        if self.scene.ambient_dimension == 3:
            moment = jnp.sum(jnp.cross(current, surface_force), axis=0)
        else:
            moment = jnp.sum(
                current[:, 0] * surface_force[:, 1] - current[:, 1] * surface_force[:, 0]
            )[None]
        complementarity = jnp.asarray(0.0, dtype=current.dtype)
        finite = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(surface_force))
            & jnp.all(jnp.isfinite(action_reaction))
            & jnp.all(jnp.isfinite(moment))
        )
        tolerance = jnp.finfo(current.dtype).eps * max(64, 8 * self.scene.vertex_count)
        nonnegative = energy >= -tolerance * jnp.maximum(1.0, jnp.abs(energy))
        successful = epoch.successful & finite & nonnegative & (minimum_gap > 0.0)
        return ContactPotentialEvaluation(
            energy,
            surface_force,
            state_force,
            minimum_gap,
            active_contacts,
            action_reaction,
            moment,
            complementarity,
            minimum_feature,
            finite,
            nonnegative,
            successful,
            epoch.epoch_id,
            self.prepared_id,
        )

    def hessian_action(
        self,
        positions: ArrayLike,
        direction: ArrayLike,
        epoch: ContactCandidateEpoch,
        /,
        *,
        rest_positions: ArrayLike | None = None,
        stiffness: ArrayLike | None = None,
    ) -> Array:
        current = jnp.asarray(
            positions, dtype=self.scene.surfaces[0].precision.geometry_dtype
        )
        tangent = jnp.asarray(direction, dtype=current.dtype)
        if tangent.shape != current.shape:
            raise ValueError("direction must match positions.")
        rest = (
            self._rest_positions()
            if rest_positions is None
            else jnp.asarray(rest_positions, dtype=current.dtype)
        )
        scale = (
            self.plan.stiffness
            if stiffness is None
            else jnp.asarray(stiffness, dtype=current.dtype)
        )
        gradient = jax.grad(
            lambda value: self.energy(value, epoch, rest_positions=rest, stiffness=scale)
        )
        return jax.jvp(gradient, (current,), (tangent,))[1]


__all__ = [
    "ContactPotentialEvaluation",
    "ConvergentContactPotentialPlan",
    "PreparedConvergentContactPotential",
]
