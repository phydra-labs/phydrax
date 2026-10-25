#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

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


type ContactKinematicsScene = PreparedCollisionScene | ContactParticipantScene


class ContactKinematicsBatch(StrictModule):
    """Fixed-capacity local interface kinematics independent of contact law.

    Rows mirror the canonical route batch. ``left_witness``/``right_witness``
    are the world witness points, ``normal`` points from right to left and
    ``tangent_basis`` completes it to a right-handed local frame. ``failed``
    marks candidate witnesses whose narrow phase was nonfinite, degenerate or
    contained; they are excluded from ``valid`` but never silently dropped:
    any failure makes the kinematics epoch unsuccessful.
    """

    vertex_indices: Array
    left_feature_ids: Array
    right_feature_ids: Array
    left_feature_indices: Array
    right_feature_indices: Array
    route_keys: Array
    witness_slots: Array
    left_participant_ids: Array
    right_participant_ids: Array
    left_body_ids: Array
    right_body_ids: Array
    left_material_ids: Array
    right_material_ids: Array
    left_patch_ids: Array
    right_patch_ids: Array
    coefficients: Array
    left_witness: Array
    right_witness: Array
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
    failed: Array
    valid: Array
    finite: Array
    kind: ContactStencilKind = eqx.field(static=True)
    catalogue_id: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    @property
    def capacity(self) -> int:
        return self.valid.size


class ContactKinematicsEvidence(StrictModule):
    active_contacts: Array
    failed_contacts: Array
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
    if batch.catalogue_id != scene.feature_catalogue.catalogue_id:
        raise ValueError("Route batch belongs to another feature catalogue.")
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
        scene=scene,
        tolerance=tolerance,
    )
    safe = jnp.clip(batch.vertex_indices, 0, scene.vertex_count - 1)
    present = batch.vertex_indices >= 0
    gathered_velocity = jnp.where(present[..., None], velocity[safe], 0.0)
    relative_velocity = jnp.sum(
        evaluation.distance.coefficients[..., None] * gathered_velocity,
        axis=1,
    )
    normal = evaluation.distance.normal
    tangent = contact_tangent_basis(normal)
    normal_velocity = jnp.sum(relative_velocity * normal, axis=-1)
    tangential_velocity = jnp.sum(tangent * relative_velocity[..., :, None], axis=-2)
    if batch.kind in (ContactStencilKind.VERTEX_PLANE, ContactStencilKind.EDGE_PLANE):
        # Plane rows keep the sign: cores at or behind the plane give gaps
        # below minus the radius; admissibility is the path certificate's job.
        distance = jnp.sum(
            evaluation.distance.distance_vector * evaluation.distance.normal, axis=-1
        )
    else:
        distance = jnp.sqrt(
            jnp.maximum(
                evaluation.distance.squared_distance,
                jnp.asarray(0.0, dtype=current.dtype),
            )
        )
    gap = distance - evaluation.minimum_separation
    tangential_slip = dt * tangential_velocity
    step_valid = jnp.isfinite(dt) & (dt > 0.0)
    finite_per_route = (
        jnp.isfinite(distance)
        & jnp.isfinite(gap)
        & jnp.isfinite(normal_velocity)
        & jnp.all(jnp.isfinite(tangential_velocity), axis=-1)
        & jnp.all(jnp.isfinite(tangential_slip), axis=-1)
    )
    considered = batch.valid & evaluation.active
    failed = evaluation.failed | (considered & ~finite_per_route)
    valid = evaluation.valid & finite_per_route & step_valid
    if activation_distance is not None:
        threshold = jnp.asarray(activation_distance, dtype=current.dtype)
        valid = valid & (gap < threshold)
    feature_limit = scene.feature_count - 1
    left_feature = jnp.clip(batch.left_feature_indices, 0, feature_limit)
    right_feature = jnp.clip(batch.right_feature_indices, 0, feature_limit)
    identifier = canonical_fingerprint(
        {
            "kind": "contact-kinematics-batch",
            "source_batch": batch.batch_id,
            "step_size_shape": tuple(dt.shape),
        }
    )
    return ContactKinematicsBatch(
        batch.vertex_indices,
        batch.left_feature_ids,
        batch.right_feature_ids,
        batch.left_feature_indices,
        batch.right_feature_indices,
        batch.route_keys,
        batch.witness_slots,
        scene.feature_participant_ids[left_feature],
        scene.feature_participant_ids[right_feature],
        scene.feature_body_ids[left_feature],
        scene.feature_body_ids[right_feature],
        scene.feature_material_ids[left_feature],
        scene.feature_material_ids[right_feature],
        scene.feature_patch_ids[left_feature],
        scene.feature_patch_ids[right_feature],
        evaluation.distance.coefficients,
        evaluation.distance.left_witness,
        evaluation.distance.right_witness,
        normal,
        tangent,
        distance,
        gap,
        normal_velocity,
        tangential_velocity,
        tangential_slip,
        batch.weights.astype(current.dtype),
        evaluation.minimum_separation,
        evaluation.distance.feature,
        jnp.minimum(
            evaluation.distance.feature_margin,
            evaluation.mollifier_margin,
        ),
        failed,
        valid,
        step_valid & ~jnp.any(failed),
        batch.kind,
        batch.catalogue_id,
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
    """Narrow phase and local frames of every candidate of a canonical epoch.

    Success requires a successful search epoch of this scene's catalogue,
    finite inputs, and no failed candidate witness in any batch.
    """
    if not isinstance(epoch, ContactCandidateEpoch):
        raise TypeError("epoch must be ContactCandidateEpoch.")
    # Foreign catalogue/topology/routing raises; changed radii or planes make
    # the candidate set incomplete and the epoch unbound (unsuccessful).
    bound = epoch.binds(scene)
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
    zero = jnp.asarray(0, dtype=jnp.int32)
    if batches:
        active = sum((jnp.sum(batch.valid, dtype=jnp.int32) for batch in batches), zero)
        failed = sum((jnp.sum(batch.failed, dtype=jnp.int32) for batch in batches), zero)
        minimum_gap = jnp.min(
            jnp.stack(
                tuple(
                    jnp.min(jnp.where(batch.valid, batch.gap, jnp.inf), initial=jnp.inf)
                    for batch in batches
                )
            )
        )
        minimum_feature = jnp.min(
            jnp.stack(
                tuple(
                    jnp.min(
                        jnp.where(batch.valid, batch.feature_margin, jnp.inf),
                        initial=jnp.inf,
                    )
                    for batch in batches
                )
            )
        )
        finite = jnp.all(jnp.stack(tuple(batch.finite for batch in batches)))
    else:
        active = zero
        failed = zero
        minimum_gap = jnp.asarray(jnp.inf, dtype=current.dtype)
        minimum_feature = jnp.asarray(jnp.inf, dtype=current.dtype)
        finite = jnp.asarray(True)
    successful = epoch.successful & bound & finite & (failed == 0)
    evidence = ContactKinematicsEvidence(
        active,
        failed,
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
