#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..camera._rig import CameraRig
from ..imaging._photometry import (
    CameraStackRenderResult,
    ParticleImageFormation,
    render_camera_stack,
)
from ..imaging._types import ImageGeometry2D
from ._association import (
    associate_multiview,
    MultiViewAssociationPlan,
    MultiViewAssociationResult,
)
from ._detection import detect_particles, ParticleDetectionPlan
from ._reconstruction import (
    ParticleReconstructionResult,
    reconstruct_particles,
    TriangulationPlan,
)
from ._types import ParticleDetections


IPR_SUCCESS = 0
IPR_NO_CANDIDATE = 1
IPR_SUBSET_REJECTED = 2
IPR_CAPACITY_EXHAUSTED = 3
IPR_NO_RESIDUAL_REDUCTION = 4
IPR_NONFINITE = 5


class IPRPlan(StrictModule, NonTrainableState):
    """Fixed-resource iterative particle reconstruction policy."""

    detection: ParticleDetectionPlan
    association: MultiViewAssociationPlan
    triangulation: TriangulationPlan
    particle_capacity: int = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    duplicate_distance: float = eqx.field(static=True)
    minimum_candidate_intensity: float = eqx.field(static=True)
    minimum_loss_reduction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        detection: ParticleDetectionPlan,
        association: MultiViewAssociationPlan,
        triangulation: TriangulationPlan,
        *,
        particle_capacity: int,
        iterations: int = 2,
        duplicate_distance: float = 0.05,
        minimum_candidate_intensity: float = 0.0,
        minimum_loss_reduction: float = 0.0,
    ):
        if not isinstance(detection, ParticleDetectionPlan):
            raise TypeError("detection must be ParticleDetectionPlan.")
        if not isinstance(association, MultiViewAssociationPlan):
            raise TypeError("association must be MultiViewAssociationPlan.")
        if not isinstance(triangulation, TriangulationPlan):
            raise TypeError("triangulation must be TriangulationPlan.")
        capacity = int(particle_capacity)
        iterations_ = int(iterations)
        duplicate = float(duplicate_distance)
        minimum_intensity = float(minimum_candidate_intensity)
        minimum_reduction = float(minimum_loss_reduction)
        if capacity <= 0 or iterations_ <= 0:
            raise ValueError("particle_capacity and iterations must be positive.")
        if not isfinite(duplicate) or duplicate <= 0.0:
            raise ValueError("duplicate_distance must be finite and positive.")
        if not isfinite(minimum_intensity) or minimum_intensity < 0.0:
            raise ValueError(
                "minimum_candidate_intensity must be finite and non-negative."
            )
        if not isfinite(minimum_reduction) or minimum_reduction < 0.0:
            raise ValueError("minimum_loss_reduction must be finite and non-negative.")
        self.detection = detection
        self.association = association
        self.triangulation = triangulation
        self.particle_capacity = capacity
        self.iterations = iterations_
        self.duplicate_distance = duplicate
        self.minimum_candidate_intensity = minimum_intensity
        self.minimum_loss_reduction = minimum_reduction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iterative-particle-reconstruction",
                "detection_plan_id": detection.plan_id,
                "association_plan_id": association.plan_id,
                "triangulation_plan_id": triangulation.plan_id,
                "particle_capacity": capacity,
                "iterations": iterations_,
                "duplicate_distance": duplicate,
                "minimum_candidate_intensity": minimum_intensity,
                "minimum_loss_reduction": minimum_reduction,
            }
        )


class IPRIterationEvidence(StrictModule):
    """Candidate selection, duplicate, subset, and capacity evidence."""

    candidate_count: Array
    accepted_count: Array
    duplicate_rejected_count: Array
    subset_rejected_count: Array
    capacity_rejected_count: Array
    loss_before: Array
    loss_after: Array
    status: Array


class IPRIterationResult(StrictModule):
    """One residual detect-associate-reconstruct-select iteration."""

    detections_by_camera: tuple[ParticleDetections, ...]
    association: MultiViewAssociationResult
    reconstruction: ParticleReconstructionResult
    accepted_candidates: Array
    evidence: IPRIterationEvidence


class IPRResult(StrictModule):
    """Fixed-capacity reconstructed particle support and residual images."""

    positions_xyz: Array
    amplitude: Array
    active: Array
    render: CameraStackRenderResult
    residual: Array
    initial_loss: Array
    final_loss: Array
    iterations: tuple[IPRIterationResult, ...]
    accepted_count: Array
    duplicate_rejected_count: Array
    subset_rejected_count: Array
    capacity_rejected_count: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _masked_squared_image_loss(
    observed: Array,
    predicted: Array,
    valid: Array,
    /,
) -> Array:
    residual = jnp.where(valid, observed - predicted, 0.0)
    count = jnp.maximum(jnp.sum(valid, dtype=predicted.dtype), 1.0)
    return jnp.sum(residual * residual) / count


def iterative_particle_reconstruction(
    plan: IPRPlan,
    formation: ParticleImageFormation,
    rig: CameraRig,
    geometry: ImageGeometry2D,
    observed_images: ArrayLike,
    positions_xyz: ArrayLike,
    amplitude: ArrayLike,
    sigma: ArrayLike,
    active: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
) -> IPRResult:
    """Recover missing point tracers by residual multi-view reconstruction."""
    if not isinstance(plan, IPRPlan):
        raise TypeError("plan must be IPRPlan.")
    if not isinstance(formation, ParticleImageFormation):
        raise TypeError("formation must be ParticleImageFormation.")
    if formation.response.stochastic:
        raise ValueError("IPR requires a deterministic photometric response.")
    if not isinstance(rig, CameraRig):
        raise TypeError("rig must be CameraRig.")
    if not isinstance(geometry, ImageGeometry2D):
        raise TypeError("geometry must be ImageGeometry2D.")
    positions = jnp.asarray(positions_xyz)
    if positions.shape != (plan.particle_capacity, 3):
        raise ValueError(
            "positions_xyz must match IPR particle_capacity and dimension 3."
        )
    if not jnp.issubdtype(positions.dtype, jnp.inexact):
        positions = positions.astype(float)
    amplitudes = jnp.asarray(amplitude, dtype=positions.dtype)
    sigmas = jnp.asarray(sigma, dtype=positions.dtype)
    active_ = jnp.asarray(active, dtype=bool)
    expected_particle_shape = (plan.particle_capacity,)
    if (
        amplitudes.shape != expected_particle_shape
        or sigmas.shape != expected_particle_shape
        or active_.shape != expected_particle_shape
    ):
        raise ValueError("amplitude, sigma, and active must match IPR particle_capacity.")
    invalid_input = active_ & (
        ~jnp.all(jnp.isfinite(positions), axis=-1)
        | ~jnp.isfinite(amplitudes)
        | ~jnp.isfinite(sigmas)
        | (amplitudes < 0.0)
        | (sigmas <= 0.0)
    )
    active_ = active_ & ~invalid_input
    observed = jnp.asarray(observed_images, dtype=positions.dtype)
    image_shape = (rig.capacity,) + geometry.image_shape
    if observed.shape != image_shape:
        raise ValueError(
            "observed_images must have shape (camera_capacity, rows, columns)."
        )
    finite_observed = jnp.isfinite(observed)
    valid = (
        finite_observed
        if valid_mask is None
        else finite_observed & jnp.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != observed.shape:
        raise ValueError("valid_mask must have the observed image shape.")
    valid = valid & rig.camera_valid[:, None, None]
    observed = jnp.where(valid, observed, 0.0)

    current_positions = jnp.where(active_[:, None], positions, 0.0)
    current_amplitudes = jnp.where(active_, amplitudes, 0.0)
    current_active = active_
    current_render = render_camera_stack(
        formation,
        rig,
        geometry,
        current_positions,
        current_amplitudes,
        sigmas,
        current_active,
    )
    current_loss = _masked_squared_image_loss(observed, current_render.images, valid)
    initial_loss = current_loss
    iteration_results: list[IPRIterationResult] = []

    for iteration in range(plan.iterations):
        positive_residual = jnp.where(
            valid, jnp.maximum(observed - current_render.images, 0.0), 0.0
        )
        detections = tuple(
            detect_particles(
                positive_residual[camera],
                geometry,
                plan.detection,
                valid_mask=valid[camera],
                frame_id=f"ipr:{plan.plan_id}:{iteration}:{camera}",
            )
            for camera in range(rig.capacity)
        )
        association = associate_multiview(detections, rig, plan.association)
        reconstruction = reconstruct_particles(
            detections, rig, association, plan.triangulation
        )
        candidate_positions = reconstruction.positions_xyz
        candidate_amplitudes = reconstruction.intensity.astype(positions.dtype)
        candidate_valid = (
            reconstruction.valid
            & jnp.all(jnp.isfinite(candidate_positions), axis=-1)
            & jnp.isfinite(candidate_amplitudes)
            & (candidate_amplitudes >= plan.minimum_candidate_intensity)
        )

        def insert_candidate(carry, candidate):
            trial_positions, trial_amplitudes, trial_active, trial_loss = carry
            candidate_position, candidate_amplitude, eligible = candidate
            delta = trial_positions - candidate_position
            distance_squared = jnp.sum(delta * delta, axis=-1)
            world_duplicate = eligible & jnp.any(
                trial_active & (distance_squared < plan.duplicate_distance**2)
            )
            free = ~trial_active
            has_capacity = jnp.any(free)
            slot = jnp.argmax(free.astype(jnp.int32))
            eligible_insert = eligible & ~world_duplicate & has_capacity
            proposed_positions = trial_positions.at[slot].set(
                jnp.where(
                    eligible_insert,
                    candidate_position,
                    trial_positions[slot],
                )
            )
            proposed_amplitudes = trial_amplitudes.at[slot].set(
                jnp.where(
                    eligible_insert,
                    candidate_amplitude,
                    trial_amplitudes[slot],
                )
            )
            proposed_active = trial_active.at[slot].set(
                trial_active[slot] | eligible_insert
            )
            proposed_render = render_camera_stack(
                formation,
                rig,
                geometry,
                proposed_positions,
                proposed_amplitudes,
                sigmas,
                proposed_active,
            )
            proposed_loss = _masked_squared_image_loss(
                observed,
                proposed_render.images,
                valid,
            )
            candidate_pixels = proposed_render.projection_pixels[:, slot]
            candidate_projection_valid = proposed_render.projection_valid[:, slot]
            projection_delta = (
                proposed_render.projection_pixels - candidate_pixels[:, None, :]
            )
            projection_distance_squared = jnp.sum(
                projection_delta * projection_delta,
                axis=-1,
            )
            shared_projection = (
                proposed_render.projection_valid
                & candidate_projection_valid[:, None]
                & trial_active[None, :]
            )
            image_separation = jnp.maximum(sigmas[slot], sigmas)[None, :]
            projection_close = (
                projection_distance_squared < image_separation * image_separation
            )
            image_duplicate = eligible_insert & jnp.any(
                trial_active
                & (jnp.sum(shared_projection, axis=0) >= 2)
                & ~jnp.any(shared_projection & ~projection_close, axis=0)
            )
            duplicate = world_duplicate | image_duplicate
            finite_proposal = (
                jnp.isfinite(proposed_loss)
                & jnp.all(jnp.isfinite(proposed_positions))
                & jnp.all(jnp.isfinite(proposed_amplitudes))
            )
            improved = proposed_loss + plan.minimum_loss_reduction < trial_loss
            candidate_admissible = eligible_insert & ~image_duplicate
            accept = candidate_admissible & finite_proposal & improved
            updated_positions = jnp.where(
                accept,
                proposed_positions,
                trial_positions,
            )
            updated_amplitudes = jnp.where(
                accept,
                proposed_amplitudes,
                trial_amplitudes,
            )
            updated_active = jnp.where(accept, proposed_active, trial_active)
            updated_loss = jnp.where(accept, proposed_loss, trial_loss)
            capacity_rejected = eligible & ~duplicate & ~has_capacity
            no_reduction_rejected = candidate_admissible & finite_proposal & ~improved
            nonfinite_rejected = candidate_admissible & ~finite_proposal
            return (
                updated_positions,
                updated_amplitudes,
                updated_active,
                updated_loss,
            ), (
                accept,
                duplicate,
                capacity_rejected,
                no_reduction_rejected,
                nonfinite_rejected,
            )

        proposed, decisions = jax.lax.scan(
            insert_candidate,
            (
                current_positions,
                current_amplitudes,
                current_active,
                current_loss,
            ),
            (candidate_positions, candidate_amplitudes, candidate_valid),
        )
        (
            proposed_positions,
            proposed_amplitudes,
            proposed_active,
            proposed_loss,
        ) = proposed
        (
            accepted,
            duplicate_rejected,
            capacity_rejected,
            no_reduction_rejected,
            nonfinite_rejected,
        ) = decisions
        proposed_render = render_camera_stack(
            formation,
            rig,
            geometry,
            proposed_positions,
            proposed_amplitudes,
            sigmas,
            proposed_active,
        )
        finite_proposal = (
            ~jnp.any(nonfinite_rejected)
            & jnp.isfinite(proposed_loss)
            & jnp.all(jnp.isfinite(proposed_positions))
            & jnp.all(jnp.isfinite(proposed_amplitudes))
        )
        loss_before = current_loss
        committed_candidates = accepted
        current_positions = proposed_positions
        current_amplitudes = proposed_amplitudes
        current_active = proposed_active
        current_loss = proposed_loss
        current_render = proposed_render
        below_threshold = reconstruction.valid & ~candidate_valid
        subset_rejected = below_threshold | duplicate_rejected
        candidate_count = jnp.sum(reconstruction.valid, dtype=jnp.int32)
        accepted_count = jnp.sum(committed_candidates, dtype=jnp.int32)
        duplicate_count = jnp.sum(duplicate_rejected, dtype=jnp.int32)
        subset_count = jnp.sum(subset_rejected, dtype=jnp.int32)
        capacity_count = jnp.sum(capacity_rejected, dtype=jnp.int32)
        no_reduction_count = jnp.sum(no_reduction_rejected, dtype=jnp.int32)
        status = jnp.where(
            ~finite_proposal,
            IPR_NONFINITE,
            jnp.where(
                capacity_count > 0,
                IPR_CAPACITY_EXHAUSTED,
                jnp.where(
                    candidate_count == 0,
                    IPR_NO_CANDIDATE,
                    jnp.where(
                        no_reduction_count > 0,
                        IPR_NO_RESIDUAL_REDUCTION,
                        jnp.where(
                            subset_count > 0,
                            IPR_SUBSET_REJECTED,
                            IPR_SUCCESS,
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        evidence = IPRIterationEvidence(
            candidate_count,
            accepted_count,
            duplicate_count,
            subset_count,
            capacity_count,
            loss_before,
            current_loss,
            status,
        )
        iteration_results.append(
            IPRIterationResult(
                detections,
                association,
                reconstruction,
                committed_candidates,
                evidence,
            )
        )

    residual = jnp.where(valid, observed - current_render.images, 0.0)
    accepted_total = sum(
        (result.evidence.accepted_count for result in iteration_results),
        jnp.asarray(0, dtype=jnp.int32),
    )
    duplicate_total = sum(
        (result.evidence.duplicate_rejected_count for result in iteration_results),
        jnp.asarray(0, dtype=jnp.int32),
    )
    subset_total = sum(
        (result.evidence.subset_rejected_count for result in iteration_results),
        jnp.asarray(0, dtype=jnp.int32),
    )
    capacity_total = sum(
        (result.evidence.capacity_rejected_count for result in iteration_results),
        jnp.asarray(0, dtype=jnp.int32),
    )
    no_reduction = jnp.any(
        jnp.stack(
            tuple(
                result.evidence.status == IPR_NO_RESIDUAL_REDUCTION
                for result in iteration_results
            )
        )
    )
    iteration_nonfinite = jnp.any(
        jnp.stack(
            tuple(result.evidence.status == IPR_NONFINITE for result in iteration_results)
        )
    )
    nonfinite = jnp.any(invalid_input) | iteration_nonfinite | ~jnp.isfinite(current_loss)
    status = jnp.where(
        nonfinite,
        IPR_NONFINITE,
        jnp.where(
            capacity_total > 0,
            IPR_CAPACITY_EXHAUSTED,
            jnp.where(
                subset_total > 0,
                IPR_SUBSET_REJECTED,
                jnp.where(
                    no_reduction,
                    IPR_NO_RESIDUAL_REDUCTION,
                    jnp.where(accepted_total == 0, IPR_NO_CANDIDATE, IPR_SUCCESS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return IPRResult(
        current_positions,
        current_amplitudes,
        current_active,
        current_render,
        residual,
        initial_loss,
        current_loss,
        tuple(iteration_results),
        accepted_total,
        duplicate_total,
        subset_total,
        capacity_total,
        status,
        ~nonfinite & (current_loss <= initial_loss),
        plan.plan_id,
    )


__all__ = [
    "IPR_CAPACITY_EXHAUSTED",
    "IPR_NONFINITE",
    "IPR_NO_CANDIDATE",
    "IPR_NO_RESIDUAL_REDUCTION",
    "IPR_SUBSET_REJECTED",
    "IPR_SUCCESS",
    "IPRIterationEvidence",
    "IPRIterationResult",
    "IPRPlan",
    "IPRResult",
    "iterative_particle_reconstruction",
]
