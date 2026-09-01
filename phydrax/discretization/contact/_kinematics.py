#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._distance import contact_tangent_basis
from ._participant import ContactParticipantScene
from ._search import ContactCandidateEpoch
from ._stencils import (
    ContactStencilBatch,
    ContactStencilKind,
    evaluate_contact_stencils,
)
from ._surface import PreparedCollisionScene


ContactKinematicsScene = PreparedCollisionScene | ContactParticipantScene


class ContactKinematicsBatch(StrictModule):
    """Fixed-capacity local interface kinematics independent of contact law."""

    vertex_indices: Array
    route_keys: Array
    left_body_ids: Array
    right_body_ids: Array
    left_material_ids: Array
    right_material_ids: Array
    coefficients: Array
    normal: Array
    tangent_basis: Array
    distance: Array
    gap: Array
    normal_velocity: Array
    tangential_velocity: Array
    tangential_slip_increment: Array
    quadrature_weight: Array
    minimum_separation: Array
    feature: Array
    feature_margin: Array
    valid: Array
    finite: Array
    kind: ContactStencilKind = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    @property
    def capacity(self) -> int:
        return int(self.valid.size)


class ContactKinematicsEvidence(StrictModule):
    active_contacts: Array
    minimum_gap: Array
    minimum_feature_margin: Array
    finite: Array
    search_complete: Array
    successful: Array
    epoch_id: str = eqx.field(static=True)


class ContactKinematicsEpoch(StrictModule):
    batches: tuple[ContactKinematicsBatch, ...]
    evidence: ContactKinematicsEvidence
    epoch_id: str = eqx.field(static=True)


def _right_endpoint(batch: ContactStencilBatch, /) -> Array:
    if batch.kind == ContactStencilKind.EDGE_EDGE:
        return batch.vertex_indices[:, 2]
    return batch.vertex_indices[:, 1]


def evaluate_contact_kinematics_batch(
    scene: ContactKinematicsScene,
    batch: ContactStencilBatch,
    positions: ArrayLike,
    velocities: ArrayLike,
    rest_positions: ArrayLike,
    step_size: ArrayLike,
    /,
    *,
    activation_distance: float | None = None,
    tolerance: float = 1.0e-12,
) -> ContactKinematicsBatch:
    if not isinstance(scene, (PreparedCollisionScene, ContactParticipantScene)):
        raise TypeError(
            "scene must be PreparedCollisionScene or ContactParticipantScene."
        )
    if not isinstance(batch, ContactStencilBatch):
        raise TypeError("batch must be ContactStencilBatch.")
    current = jnp.asarray(positions)
    velocity = jnp.asarray(velocities, dtype=current.dtype)
    rest = jnp.asarray(rest_positions, dtype=current.dtype)
    dt = jnp.asarray(step_size, dtype=current.dtype)
    expected = (scene.vertex_count, scene.ambient_dimension)
    if current.shape != expected or velocity.shape != expected or rest.shape != expected:
        raise ValueError(
            "Contact positions, velocities, and rest positions changed shape."
        )
    if dt.shape != ():
        raise ValueError("Contact kinematics step_size must be scalar.")
    evaluation = evaluate_contact_stencils(
        batch,
        current,
        rest,
        tolerance=tolerance,
    )
    safe = jnp.clip(batch.vertex_indices, 0, scene.vertex_count - 1)
    gathered_velocity = velocity[safe]
    relative_velocity = jnp.sum(
        evaluation.distance.coefficients[..., None] * gathered_velocity,
        axis=1,
    )
    normal = evaluation.distance.normal
    tangent = contact_tangent_basis(normal)
    normal_velocity = jnp.sum(relative_velocity * normal, axis=-1)
    tangential_velocity = jnp.sum(tangent * relative_velocity[..., :, None], axis=-2)
    distance = jnp.sqrt(
        jnp.maximum(
            evaluation.distance.squared_distance,
            jnp.asarray(0.0, dtype=current.dtype),
        )
    )
    gap = distance - evaluation.minimum_separation
    valid = evaluation.valid
    if activation_distance is not None:
        threshold = jnp.asarray(activation_distance, dtype=current.dtype)
        valid = valid & (gap < threshold)
    left_vertex = jnp.clip(batch.vertex_indices[:, 0], 0, scene.vertex_count - 1)
    right_vertex = jnp.clip(_right_endpoint(batch), 0, scene.vertex_count - 1)
    finite_per_route = (
        jnp.isfinite(distance)
        & jnp.isfinite(gap)
        & jnp.isfinite(normal_velocity)
        & jnp.all(jnp.isfinite(tangential_velocity), axis=-1)
    )
    valid = valid & finite_per_route
    identifier = canonical_fingerprint(
        {
            "kind": "contact-kinematics-batch",
            "source_batch": batch.batch_id,
            "step_size_shape": tuple(dt.shape),
        }
    )
    return ContactKinematicsBatch(
        batch.vertex_indices,
        batch.route_keys,
        scene.vertex_body_ids[left_vertex],
        scene.vertex_body_ids[right_vertex],
        scene.vertex_material_ids[left_vertex],
        scene.vertex_material_ids[right_vertex],
        evaluation.distance.coefficients,
        normal,
        tangent,
        distance,
        gap,
        normal_velocity,
        tangential_velocity,
        dt * tangential_velocity,
        batch.weights.astype(current.dtype),
        evaluation.minimum_separation,
        evaluation.distance.feature,
        jnp.minimum(
            evaluation.distance.feature_margin,
            evaluation.mollifier_margin,
        ),
        valid,
        jnp.all((~batch.valid) | finite_per_route),
        batch.kind,
        identifier,
    )


def evaluate_contact_kinematics(
    scene: ContactKinematicsScene,
    epoch: ContactCandidateEpoch,
    positions: ArrayLike,
    velocities: ArrayLike,
    step_size: ArrayLike,
    /,
    *,
    rest_positions: ArrayLike | None = None,
    activation_distance: float | None = None,
    tolerance: float = 1.0e-12,
) -> ContactKinematicsEpoch:
    if not isinstance(epoch, ContactCandidateEpoch):
        raise TypeError("epoch must be ContactCandidateEpoch.")
    current = jnp.asarray(positions)
    if rest_positions is None:
        if not isinstance(scene, PreparedCollisionScene):
            raise ValueError(
                "Independent participant scenes require explicit rest_positions."
            )
        rest = jnp.concatenate(
            tuple(surface.rest_positions for surface in scene.surfaces),
            axis=0,
        )
    else:
        rest = jnp.asarray(rest_positions, dtype=current.dtype)
    batches = tuple(
        evaluate_contact_kinematics_batch(
            scene,
            batch,
            current,
            velocities,
            rest,
            step_size,
            activation_distance=activation_distance,
            tolerance=tolerance,
        )
        for batch in epoch.active_batches
    )
    if batches:
        active = sum(
            (jnp.sum(batch.valid, dtype=jnp.int32) for batch in batches),
            start=jnp.asarray(0, dtype=jnp.int32),
        )
        minimum_gap = jnp.min(
            jnp.stack(
                tuple(
                    jnp.min(
                        jnp.where(batch.valid, batch.gap, jnp.inf),
                        initial=jnp.inf,
                    )
                    for batch in batches
                )
            )
        )
        minimum_feature = jnp.min(
            jnp.stack(
                tuple(
                    jnp.min(
                        jnp.where(
                            batch.valid,
                            batch.feature_margin,
                            jnp.inf,
                        ),
                        initial=jnp.inf,
                    )
                    for batch in batches
                )
            )
        )
        finite = jnp.all(jnp.stack(tuple(batch.finite for batch in batches)))
    else:
        active = jnp.asarray(0, dtype=jnp.int32)
        minimum_gap = jnp.asarray(jnp.inf, dtype=current.dtype)
        minimum_feature = jnp.asarray(jnp.inf, dtype=current.dtype)
        finite = jnp.asarray(True)
    successful = epoch.successful & finite
    evidence = ContactKinematicsEvidence(
        active,
        minimum_gap,
        minimum_feature,
        finite,
        epoch.successful,
        successful,
        epoch.epoch_id,
    )
    return ContactKinematicsEpoch(
        batches,
        evidence,
        canonical_fingerprint(
            {
                "kind": "contact-kinematics-epoch",
                "candidate_epoch": epoch.epoch_id,
                "batches": [batch.batch_id for batch in batches],
            }
        ),
    )


__all__ = [
    "ContactKinematicsBatch",
    "ContactKinematicsEpoch",
    "ContactKinematicsEvidence",
    "evaluate_contact_kinematics",
    "evaluate_contact_kinematics_batch",
]
