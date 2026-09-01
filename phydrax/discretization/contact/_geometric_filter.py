#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._kinematics import ContactKinematicsEpoch
from ._stencils import ContactStencilKind
from ._surface import CollisionSurfacePlan, PreparedCollisionScene


class ClosedSurfaceCertificate(StrictModule):
    edge_incidence_valid: Array
    orientation_consistent: Array
    nondegenerate: Array
    codimensional_free: Array
    successful: Array
    topology_id: str = eqx.field(static=True)


def certify_closed_oriented_surface(
    plan: CollisionSurfacePlan,
    rest_positions: ArrayLike,
    /,
) -> ClosedSurfaceCertificate:
    if not isinstance(plan, CollisionSurfacePlan):
        raise TypeError("plan must be CollisionSurfacePlan.")
    positions = np.asarray(rest_positions, dtype=float)
    faces = np.asarray(plan.faces, dtype=np.int32)
    edges = np.asarray(plan.edges, dtype=np.int32)
    edge_lookup = {
        tuple(sorted(edge.tolist())): index for index, edge in enumerate(edges)
    }
    incidence = np.zeros((edges.shape[0],), dtype=np.int32)
    oriented_sum = np.zeros((edges.shape[0],), dtype=np.int32)
    nondegenerate = True
    for face in faces:
        first = positions[face[1]] - positions[face[0]]
        second = positions[face[2]] - positions[face[0]]
        nondegenerate = (
            nondegenerate
            and np.dot(np.cross(first, second), np.cross(first, second)) > 0.0
        )
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            key = tuple(sorted((int(left), int(right))))
            edge = edge_lookup[key]
            incidence[edge] += 1
            oriented_sum[edge] += 1 if left < right else -1
    incidence_valid = faces.shape[0] > 0 and np.all(incidence == 2)
    orientation = incidence_valid and np.all(oriented_sum == 0)
    codimensional = not bool(jnp.any(plan.codimensional_mask))
    successful = incidence_valid and orientation and nondegenerate and codimensional
    return ClosedSurfaceCertificate(
        jnp.asarray(incidence_valid),
        jnp.asarray(orientation),
        jnp.asarray(nondegenerate),
        jnp.asarray(codimensional),
        jnp.asarray(successful),
        plan.topology_id,
    )


class GeometricContactFilterPlan(StrictModule, NonTrainableState):
    normal_alignment: float = eqx.field(static=True)
    feature_tolerance: float = eqx.field(static=True)
    require_closed_surface: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        normal_alignment: float = 0.0,
        feature_tolerance: float = 1.0e-10,
        require_closed_surface: bool = True,
    ):
        alignment = float(normal_alignment)
        tolerance = float(feature_tolerance)
        if not -1.0 <= alignment <= 1.0 or tolerance < 0.0:
            raise ValueError("Geometric contact filter controls are invalid.")
        self.normal_alignment = alignment
        self.feature_tolerance = tolerance
        self.require_closed_surface = bool(require_closed_surface)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "geometric-contact-filter-plan",
                "normal_alignment": alignment.hex(),
                "feature_tolerance": tolerance.hex(),
                "require_closed_surface": bool(require_closed_surface),
            }
        )


class GeometricContactFilterEvidence(StrictModule):
    manifold_certified: Array
    input_contacts: Array
    exterior_contacts: Array
    local_minimum_contacts: Array
    output_contacts: Array
    minimum_margin: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class GeometricContactFilterResult(StrictModule):
    kinematics: ContactKinematicsEpoch
    evidence: GeometricContactFilterEvidence


def _safe_normal(value):
    norm = jnp.sqrt(jnp.sum(value * value, axis=-1, keepdims=True))
    return value / jnp.maximum(norm, jnp.finfo(value.dtype).eps)


def filter_geometric_contacts(
    plan: GeometricContactFilterPlan,
    scene: PreparedCollisionScene,
    kinematics: ContactKinematicsEpoch,
    positions: ArrayLike,
    /,
) -> GeometricContactFilterResult:
    if not isinstance(plan, GeometricContactFilterPlan):
        raise TypeError("plan must be GeometricContactFilterPlan.")
    if not isinstance(scene, PreparedCollisionScene):
        raise TypeError("scene must be PreparedCollisionScene.")
    current = jnp.asarray(positions)
    if current.shape != (scene.vertex_count, scene.ambient_dimension):
        raise ValueError("Geometric contact positions have invalid shape.")
    certificates = tuple(
        certify_closed_oriented_surface(surface.plan, surface.rest_positions)
        for surface in scene.surfaces
        if surface.plan.face_count
    )
    manifold = (
        jnp.all(jnp.stack(tuple(value.successful for value in certificates)))
        if certificates
        else jnp.asarray(not plan.require_closed_surface)
    )
    vertex_normal = jnp.zeros_like(current)
    if scene.face_count:
        faces = scene.faces
        face_vector = jnp.cross(
            current[faces[:, 1]] - current[faces[:, 0]],
            current[faces[:, 2]] - current[faces[:, 0]],
        )
        face_normal = _safe_normal(face_vector)
        for local in range(3):
            vertex_normal = vertex_normal.at[faces[:, local]].add(face_normal)
        vertex_normal = _safe_normal(vertex_normal)
    filtered_batches = []
    input_count = jnp.asarray(0, dtype=jnp.int32)
    exterior_count = jnp.asarray(0, dtype=jnp.int32)
    local_count = jnp.asarray(0, dtype=jnp.int32)
    output_count = jnp.asarray(0, dtype=jnp.int32)
    margins = []
    for batch in kinematics.batches:
        input_count = input_count + jnp.sum(batch.valid, dtype=jnp.int32)
        indices = jnp.clip(batch.vertex_indices, 0, scene.vertex_count - 1)
        if batch.kind == ContactStencilKind.FACE_VERTEX:
            first = current[indices[:, 2]] - current[indices[:, 1]]
            second = current[indices[:, 3]] - current[indices[:, 1]]
            pseudonormal = _safe_normal(jnp.cross(first, second))
            alignment = jnp.sum(pseudonormal * batch.normal, axis=-1)
            exterior = alignment >= plan.normal_alignment
        elif batch.kind == ContactStencilKind.EDGE_EDGE:
            left_normal = _safe_normal(
                vertex_normal[indices[:, 0]] + vertex_normal[indices[:, 1]]
            )
            right_normal = _safe_normal(
                vertex_normal[indices[:, 2]] + vertex_normal[indices[:, 3]]
            )
            exterior = (
                jnp.sum(left_normal * batch.normal, axis=-1) >= plan.normal_alignment
            ) & (jnp.sum(right_normal * -batch.normal, axis=-1) >= plan.normal_alignment)
        elif batch.kind == ContactStencilKind.EDGE_VERTEX:
            edge = current[indices[:, 2]] - current[indices[:, 1]]
            pseudonormal = _safe_normal(jnp.stack((-edge[:, 1], edge[:, 0]), axis=-1))
            exterior = (
                jnp.sum(pseudonormal * batch.normal, axis=-1) >= plan.normal_alignment
            )
        else:
            exterior = jnp.ones_like(batch.valid)
        local_minimum = batch.feature_margin > plan.feature_tolerance
        valid = batch.valid & exterior & local_minimum
        if plan.require_closed_surface:
            valid = valid & manifold
        filtered_batches.append(eqx.tree_at(lambda value: value.valid, batch, valid))
        exterior_count = exterior_count + jnp.sum(batch.valid & exterior, dtype=jnp.int32)
        local_count = local_count + jnp.sum(batch.valid & local_minimum, dtype=jnp.int32)
        output_count = output_count + jnp.sum(valid, dtype=jnp.int32)
        margins.append(
            jnp.min(
                jnp.where(valid, batch.feature_margin, jnp.inf),
                initial=jnp.inf,
            )
        )
    filtered = eqx.tree_at(
        lambda value: value.batches,
        kinematics,
        tuple(filtered_batches),
    )
    minimum_margin = (
        jnp.min(jnp.stack(tuple(margins))) if margins else jnp.asarray(jnp.inf)
    )
    finite = jnp.all(jnp.isfinite(current)) & ~jnp.isnan(minimum_margin)
    successful = (
        kinematics.evidence.successful
        & finite
        & (manifold | jnp.asarray(not plan.require_closed_surface))
    )
    evidence = GeometricContactFilterEvidence(
        manifold,
        input_count,
        exterior_count,
        local_count,
        output_count,
        minimum_margin,
        finite,
        successful,
        plan.plan_id,
    )
    return GeometricContactFilterResult(filtered, evidence)


__all__ = [
    "ClosedSurfaceCertificate",
    "GeometricContactFilterEvidence",
    "GeometricContactFilterPlan",
    "GeometricContactFilterResult",
    "certify_closed_oriented_surface",
    "filter_geometric_contacts",
]
