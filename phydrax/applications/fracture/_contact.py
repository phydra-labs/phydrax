#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact import (
    CollisionSurfacePlan,
    contact_tangent_basis,
    ContactCandidateEpoch,
    ContactKinematicsBatch,
    ContactKinematicsEpoch,
    ContactKinematicsEvidence,
    ContactPairPolicy,
    ContactParticipantScene,
    ContactStencilBatch,
    DenseContactSearchPlan,
    evaluate_contact_kinematics,
    LinearContactParticipant,
    PreparedCollisionSurface,
    selection_collision_operator,
)
from ...linalg import ArraySpace
from ..contact import (
    AbstractNormalContactLaw,
    AbstractTangentialContactLaw,
    assemble_smooth_contact,
    ContactClosurePlan,
    ContactMaterialPairTable,
    ContactRouteState,
    ContactStateTransferPlan,
    ContactStateTransferResult,
    CrossDiscretizationContactResult,
    evaluate_contact_closure,
    remap_contact_route_state,
    transfer_contact_route_state,
)
from ._geometry import SharpCrackTopology


def _complete_exclusions(vertex_count: int, /) -> np.ndarray:
    if vertex_count < 2:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(
        [
            (left, right)
            for left in range(vertex_count)
            for right in range(left + 1, vertex_count)
        ],
        dtype=np.int64,
    )


def _surface_participant(
    topology: SharpCrackTopology,
    vertex_ids: np.ndarray,
    body_id: int,
    material_id: int,
    /,
) -> LinearContactParticipant:
    geometry = topology.geometry
    coordinates = np.asarray(geometry.vertices)
    vertex_count = coordinates.shape[0]
    policy = ContactPairPolicy(
        vertex_count,
        excluded_vertex_pairs=_complete_exclusions(vertex_count),
    )
    plan = CollisionSurfacePlan(
        vertex_ids,
        ambient_dimension=2,
        edges=geometry.segments,
        body_ids=np.full((vertex_count,), body_id, dtype=np.int64),
        material_ids=np.full((vertex_count,), material_id, dtype=np.int64),
        pair_policy=policy,
        topology_id=canonical_fingerprint(
            {
                "kind": "sharp-crack-collision-surface",
                "topology": topology.topology_id,
                "body": body_id,
            }
        ),
    )
    space = ArraySpace(coordinates.shape, dtype=coordinates.dtype)
    surface = PreparedCollisionSurface(
        plan,
        coordinates,
        selection_collision_operator(
            space,
            np.arange(vertex_count, dtype=np.int32),
            operator_id=canonical_fingerprint(
                {
                    "kind": "sharp-crack-trace-selection",
                    "topology": topology.topology_id,
                    "body": body_id,
                }
            ),
        ),
    )
    return LinearContactParticipant(surface)


def _filtered_crack_epoch(
    scene: ContactParticipantScene,
    search: DenseContactSearchPlan,
    reference_positions: np.ndarray,
    vertex_count: int,
    segments: np.ndarray,
    segment_ids: np.ndarray,
    nodal_weights: np.ndarray,
    /,
) -> tuple[ContactCandidateEpoch, np.ndarray, np.ndarray, np.ndarray]:
    source = search.build(scene, reference_positions)
    batch = source.edge_vertex
    indices = np.asarray(batch.vertex_indices)
    source_valid = np.asarray(batch.valid)
    desired = (
        source_valid
        & (indices[:, 0] >= 0)
        & (indices[:, 0] < vertex_count)
        & (indices[:, 1] >= vertex_count)
        & (indices[:, 2] >= vertex_count)
    )
    selected = np.zeros((batch.capacity,), dtype=bool)
    route_segment = np.zeros((batch.capacity, 2), dtype=np.int32)
    route_node_ids = np.full((batch.capacity,), -1, dtype=np.int64)
    route_segment_ids = np.full((batch.capacity,), -1, dtype=np.int64)
    segment_keys = {
        tuple(sorted((int(first), int(second)))): index
        for index, (first, second) in enumerate(segments.tolist())
    }
    for point_index in range(vertex_count):
        rows = np.flatnonzero(desired & (indices[:, 0] == point_index))
        if rows.size == 0:
            continue
        distances = []
        stable_ids = []
        for row in rows:
            first = indices[row, 1] - vertex_count
            second = indices[row, 2] - vertex_count
            segment_index = segment_keys[tuple(sorted((first, second)))]
            a = reference_positions[vertex_count + first]
            b = reference_positions[vertex_count + second]
            tangent = b - a
            parameter = np.clip(
                np.dot(reference_positions[point_index] - a, tangent)
                / np.dot(tangent, tangent),
                0.0,
                1.0,
            )
            witness = a + parameter * tangent
            delta = reference_positions[point_index] - witness
            distances.append(float(np.dot(delta, delta)))
            stable_ids.append(int(segment_ids[segment_index]))
        order = np.lexsort((np.asarray(stable_ids), np.asarray(distances)))
        row = int(rows[order[0]])
        first = int(indices[row, 1] - vertex_count)
        second = int(indices[row, 2] - vertex_count)
        segment_index = segment_keys[tuple(sorted((first, second)))]
        selected[row] = True
        route_segment[row] = segments[segment_index]
        route_node_ids[row] = point_index
        route_segment_ids[row] = segment_index
    weights = np.zeros((batch.capacity,), dtype=nodal_weights.dtype)
    weights[selected] = nodal_weights[indices[selected, 0]]
    filtered_batch = ContactStencilBatch(
        batch.kind,
        batch.vertex_indices,
        batch.left_feature_ids,
        batch.right_feature_ids,
        capacity=batch.capacity,
        feature_indices=jnp.stack(
            (batch.left_feature_indices, batch.right_feature_indices), axis=1
        ),
        weights=weights,
        minimum_separation=batch.minimum_separation,
        valid=selected,
        actual_count=int(np.sum(selected)),
        overflow_count=0,
        route_keys=batch.route_keys,
    )
    epoch_id = canonical_fingerprint(
        {
            "kind": "sharp-crack-contact-candidate-epoch",
            "scene": scene.scene_id,
            "search": search.plan_id,
            "batch": filtered_batch.batch_id,
        }
    )
    epoch = ContactCandidateEpoch(
        filtered_batch,
        source.edge_edge,
        source.face_vertex,
        source.reference_positions,
        source.envelope_radius,
        jnp.asarray(np.sum(selected), dtype=jnp.int32),
        jnp.asarray(np.sum(selected) * 96, dtype=jnp.int64),
        source.elapsed_seconds,
        source.status,
        source.complete,
        search.plan_id,
        epoch_id,
    )
    return epoch, route_segment, route_node_ids, route_segment_ids


class CrackFaceContactAdapter(StrictModule, NonTrainableState):
    """Canonical participant, epoch, closure, and route state for crack faces."""

    topology: SharpCrackTopology
    scene: ContactParticipantScene
    search: DenseContactSearchPlan
    closure_plan: ContactClosurePlan
    candidate_epoch: ContactCandidateEpoch
    reference_coordinates: Array
    rest_positions: Array
    plus_node_ids: Array
    minus_node_ids: Array
    segment_ids: Array
    route_segments: Array
    route_node_ids: Array
    route_segment_ids: Array
    topology_id: str = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: SharpCrackTopology,
        normal_law: AbstractNormalContactLaw,
        material_table: ContactMaterialPairTable,
        /,
        *,
        tangential_law: AbstractTangentialContactLaw | None = None,
        search_radius: float = math.inf,
        plus_material_id: int = 0,
        minus_material_id: int = 0,
        adapter_id: str = "sharp-crack-face-contact",
    ):
        if not isinstance(topology, SharpCrackTopology):
            raise TypeError("topology must be SharpCrackTopology.")
        if not isinstance(normal_law, AbstractNormalContactLaw):
            raise TypeError("normal_law must implement AbstractNormalContactLaw.")
        if not isinstance(material_table, ContactMaterialPairTable):
            raise TypeError("material_table must be ContactMaterialPairTable.")
        if tangential_law is not None and not isinstance(
            tangential_law, AbstractTangentialContactLaw
        ):
            raise TypeError("tangential_law must implement AbstractTangentialContactLaw.")
        declared_id = str(adapter_id)
        if not declared_id:
            raise ValueError("adapter_id must be nonempty.")
        geometry = topology.geometry
        coordinates = np.asarray(geometry.vertices)
        segments = np.asarray(geometry.segments, dtype=np.int32)
        segment_ids = np.asarray(geometry.segment_ids, dtype=np.int64)
        vertex_count = coordinates.shape[0]
        plus_node_ids = 2 * np.arange(vertex_count, dtype=np.int64)
        minus_node_ids = plus_node_ids + 1
        plus = _surface_participant(
            topology,
            plus_node_ids,
            0,
            int(plus_material_id),
        )
        minus = _surface_participant(
            topology,
            minus_node_ids,
            1,
            int(minus_material_id),
        )
        scene = ContactParticipantScene((plus, minus))
        rest_positions = np.concatenate((coordinates, coordinates), axis=0)
        if math.isinf(search_radius):
            extent = np.ptp(coordinates, axis=0)
            radius = max(float(np.linalg.norm(extent)), 1.0)
        else:
            radius = float(search_radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("search_radius must be positive or infinite.")
        capacity = max(1, 4 * vertex_count * segments.shape[0])
        search = DenseContactSearchPlan(
            edge_vertex_capacity=capacity,
            edge_edge_capacity=0,
            face_vertex_capacity=0,
            activation_distance=radius,
        )
        lengths = np.linalg.norm(
            coordinates[segments[:, 1]] - coordinates[segments[:, 0]], axis=1
        )
        nodal_weights = np.zeros((vertex_count,), dtype=coordinates.dtype)
        np.add.at(nodal_weights, segments[:, 0], 0.5 * lengths)
        np.add.at(nodal_weights, segments[:, 1], 0.5 * lengths)
        if np.any(nodal_weights <= 0.0):
            raise ValueError("Every crack-face vertex must carry positive trace measure.")
        epoch, route_segments, route_node_ids, route_segment_indices = (
            _filtered_crack_epoch(
                scene,
                search,
                rest_positions,
                vertex_count,
                segments,
                segment_ids,
                nodal_weights,
            )
        )
        closure_plan = ContactClosurePlan(
            normal_law,
            material_table,
            tangential=tangential_law,
        )
        route_segment_ids = np.full(route_segment_indices.shape, -1, dtype=np.int64)
        valid_route = route_segment_indices >= 0
        route_segment_ids[valid_route] = segment_ids[route_segment_indices[valid_route]]
        self.topology = topology
        self.scene = scene
        self.search = search
        self.closure_plan = closure_plan
        self.candidate_epoch = epoch
        self.reference_coordinates = jnp.asarray(coordinates)
        self.rest_positions = jnp.asarray(rest_positions)
        self.plus_node_ids = jnp.asarray(plus_node_ids)
        self.minus_node_ids = jnp.asarray(minus_node_ids)
        self.segment_ids = jnp.asarray(segment_ids)
        self.route_segments = jnp.asarray(route_segments)
        self.route_node_ids = jnp.asarray(route_node_ids)
        self.route_segment_ids = jnp.asarray(route_segment_ids)
        self.topology_id = topology.topology_id
        self.mapping_id = canonical_fingerprint(
            {
                "kind": "sharp-crack-face-contact-mapping",
                "declared_id": declared_id,
                "topology": topology.topology_id,
                "scene": scene.scene_id,
                "epoch": epoch.epoch_id,
                "closure": closure_plan.closure_id,
                "segment_ids": segment_ids.tolist(),
            }
        )

    def initial_state(self, /) -> ContactRouteState:
        return ContactRouteState.empty(0, 1, self.closure_plan.closure_id)

    def current_coordinates(
        self,
        plus_displacement: ArrayLike,
        minus_displacement: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        plus = jnp.asarray(plus_displacement)
        minus = jnp.asarray(minus_displacement)
        if plus.shape != self.reference_coordinates.shape or minus.shape != plus.shape:
            raise ValueError(
                "Crack-face displacements must preserve the crack-vertex layout."
            )
        return self.reference_coordinates + plus, self.reference_coordinates + minus

    def _oriented_kinematics(
        self,
        states: tuple[Array, Array],
        positions: Array,
        /,
    ) -> ContactKinematicsEpoch:
        zero_rates = tuple(jnp.zeros_like(state) for state in states)
        canonical = evaluate_contact_kinematics(
            self.scene,
            self.candidate_epoch,
            positions,
            self.scene.velocities(states, zero_rates),
            1.0,
            rest_positions=self.rest_positions,
            activation_distance=self.search.activation_distance,
        )
        batch = canonical.batches[0]
        safe_segments = jnp.clip(
            self.route_segments, 0, self.reference_coordinates.shape[0] - 1
        )
        minus_offset = self.reference_coordinates.shape[0]
        first = positions[minus_offset + safe_segments[:, 0]]
        second = positions[minus_offset + safe_segments[:, 1]]
        tangent = second - first
        norm = jnp.linalg.norm(tangent, axis=-1)
        orientation = jnp.asarray(
            self.topology.geometry.orientation, dtype=positions.dtype
        )
        normal = orientation * jnp.stack((-tangent[:, 1], tangent[:, 0]), axis=-1)
        normal = normal / jnp.where(norm > 0.0, norm, 1.0)[:, None]
        coefficients = batch.coefficients
        point = positions[jnp.clip(batch.vertex_indices[:, 0], 0, positions.shape[0] - 1)]
        closest = -(coefficients[:, 1, None] * first + coefficients[:, 2, None] * second)
        gap = jnp.sum((point - closest) * normal, axis=-1)
        valid = batch.valid & (self.route_node_ids >= 0) & (norm > 0.0)
        velocity = self.scene.velocities(states, zero_rates)
        safe = jnp.clip(batch.vertex_indices, 0, positions.shape[0] - 1)
        gathered = velocity[safe]
        relative_velocity = jnp.sum(coefficients[..., None] * gathered, axis=1)
        tangent_basis = contact_tangent_basis(normal)
        normal_velocity = jnp.sum(relative_velocity * normal, axis=-1)
        tangential_velocity = jnp.sum(
            tangent_basis * relative_velocity[..., :, None], axis=-2
        )
        finite_per_route = (
            jnp.isfinite(gap)
            & jnp.all(jnp.isfinite(normal), axis=-1)
            & jnp.isfinite(normal_velocity)
            & jnp.all(jnp.isfinite(tangential_velocity), axis=-1)
        )
        valid = valid & finite_per_route
        oriented_batch = ContactKinematicsBatch(
            batch.vertex_indices,
            batch.left_feature_ids,
            batch.right_feature_ids,
            batch.left_feature_indices,
            batch.right_feature_indices,
            batch.route_keys,
            batch.left_participant_ids,
            batch.right_participant_ids,
            batch.left_body_ids,
            batch.right_body_ids,
            batch.left_material_ids,
            batch.right_material_ids,
            batch.left_patch_ids,
            batch.right_patch_ids,
            batch.coefficients,
            normal,
            tangent_basis,
            batch.distance,
            gap,
            normal_velocity,
            tangential_velocity,
            tangential_velocity,
            batch.quadrature_weight,
            batch.minimum_separation,
            batch.feature,
            batch.feature_margin,
            valid,
            jnp.all((~batch.valid) | finite_per_route),
            batch.kind,
            canonical_fingerprint(
                {
                    "kind": "oriented-sharp-crack-kinematics-batch",
                    "candidate": self.candidate_epoch.epoch_id,
                }
            ),
        )
        active = jnp.sum(valid, dtype=jnp.int32)
        minimum_gap = jnp.min(jnp.where(valid, gap, jnp.inf), initial=jnp.inf)
        finite = oriented_batch.finite
        successful = self.candidate_epoch.successful & finite
        evidence = ContactKinematicsEvidence(
            active,
            minimum_gap,
            canonical.evidence.minimum_feature_margin,
            finite,
            self.candidate_epoch.successful,
            successful,
            self.candidate_epoch.epoch_id,
        )
        return ContactKinematicsEpoch(
            (oriented_batch,),
            evidence,
            canonical_fingerprint(
                {
                    "kind": "oriented-sharp-crack-kinematics-epoch",
                    "candidate": self.candidate_epoch.epoch_id,
                }
            ),
        )

    def evaluate(
        self,
        accepted: ContactRouteState,
        plus_displacement: ArrayLike,
        minus_displacement: ArrayLike,
        /,
    ) -> CrossDiscretizationContactResult:
        if not isinstance(accepted, ContactRouteState):
            raise TypeError("accepted must be ContactRouteState.")
        states = (jnp.asarray(plus_displacement), jnp.asarray(minus_displacement))
        self.current_coordinates(*states)
        positions = self.scene.positions(states)
        velocities = jnp.zeros_like(positions)
        kinematics = self._oriented_kinematics(states, positions)
        transition = remap_contact_route_state(accepted, kinematics)
        closure = evaluate_contact_closure(
            self.closure_plan,
            kinematics,
            transition.candidate,
        )
        assembly = assemble_smooth_contact(kinematics, closure, positions)
        generalized = self.scene.effort_pullback(states, assembly.surface_force)
        successful = (
            self.candidate_epoch.successful
            & kinematics.evidence.successful
            & transition.successful
            & closure.evidence.successful
            & assembly.successful
        )
        return CrossDiscretizationContactResult(
            positions,
            velocities,
            self.candidate_epoch,
            kinematics,
            transition,
            closure,
            assembly,
            generalized,
            successful,
            self.scene.scene_id,
        )

    def transfer_state(
        self,
        previous_adapter: "CrackFaceContactAdapter",
        previous: ContactRouteState,
        /,
    ) -> ContactStateTransferResult:
        if not isinstance(previous_adapter, CrackFaceContactAdapter):
            raise TypeError("previous_adapter must be CrackFaceContactAdapter.")
        if not isinstance(previous, ContactRouteState):
            raise TypeError("previous must be ContactRouteState.")
        if previous.closure_id != self.closure_plan.closure_id:
            raise ValueError("Crack contact closures differ across topology transfer.")
        old_identity = {
            (int(node), int(segment)): slot
            for slot, (node, segment) in enumerate(
                zip(
                    np.asarray(previous_adapter.route_node_ids),
                    np.asarray(previous_adapter.route_segment_ids),
                    strict=True,
                )
            )
            if node >= 0
            and segment >= 0
            and slot < previous.capacity
            and bool(previous.valid[slot])
        }
        capacity = self.candidate_epoch.edge_vertex.capacity
        parents = np.full((capacity, 1), -1, dtype=np.int32)
        weights = np.zeros((capacity, 1), dtype=float)
        valid = np.zeros((capacity,), dtype=bool)
        for slot, identity in enumerate(
            zip(
                np.asarray(self.route_node_ids),
                np.asarray(self.route_segment_ids),
                strict=True,
            )
        ):
            parent = old_identity.get((int(identity[0]), int(identity[1])))
            if parent is not None:
                parents[slot, 0] = parent
                weights[slot, 0] = 1.0
                valid[slot] = True
        plan = ContactStateTransferPlan(
            self.candidate_epoch.edge_vertex.route_keys,
            parents,
            weights,
            valid=valid,
        )
        return transfer_contact_route_state(plan, previous)


__all__ = ["CrackFaceContactAdapter"]
