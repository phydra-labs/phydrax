#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._objects import ReconstructedParticleBank
from ._tracking import ReconstructedTrackBank
from .calorimetry import CalorimeterClusterBank


class PrimaryVertexPlan(StrictModule, NonTrainableState):
    minimum_tracks: int = eqx.field(static=True)
    maximum_variance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, minimum_tracks: int = 2, maximum_variance: float = 1.0e6):
        minimum = int(minimum_tracks)
        variance = float(maximum_variance)
        if minimum < 1 or not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("Primary-vertex track and variance policy are invalid.")
        self.minimum_tracks = minimum
        self.maximum_variance = variance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-association-primary-vertex-plan",
                "minimum_tracks": minimum,
                "maximum_variance": variance,
            }
        )


class ReconstructedVertexBank(StrictModule, NonTrainableState):
    event_ids: Array
    positions: Array
    covariance: Array
    track_counts: Array
    chi_square: Array
    active: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def fit_primary_vertices(
    plan: PrimaryVertexPlan,
    tracks: ReconstructedTrackBank,
    /,
) -> ReconstructedVertexBank:
    """Fit one bounded primary vertex from caller-resolved track associations."""
    if not isinstance(plan, PrimaryVertexPlan) or not isinstance(
        tracks, ReconstructedTrackBank
    ):
        raise TypeError("plan and tracks must use detector reconstruction types.")
    positions = tracks.parameters[..., :3]
    diagonal_variance = jnp.diagonal(tracks.covariance[..., :3, :3], axis1=-2, axis2=-1)
    admitted = (
        tracks.active
        & tracks.valid
        & jnp.all(
            jnp.isfinite(diagonal_variance)
            & (diagonal_variance > 0.0)
            & (diagonal_variance <= plan.maximum_variance),
            axis=-1,
        )
    )
    precision = jnp.where(admitted[..., None], 1.0 / diagonal_variance, 0.0)
    total_precision = jnp.sum(precision, axis=1)
    vertex = jnp.sum(precision * positions, axis=1) / jnp.maximum(
        total_precision, jnp.finfo(positions.dtype).tiny
    )
    covariance = jnp.zeros((positions.shape[0], 3, 3), dtype=positions.dtype)
    covariance = covariance.at[:, jnp.arange(3), jnp.arange(3)].set(
        1.0 / jnp.maximum(total_precision, jnp.finfo(positions.dtype).tiny)
    )
    residual = positions - vertex[:, None, :]
    chi_square = jnp.sum(
        jnp.where(admitted[..., None], residual * residual / diagonal_variance, 0.0),
        axis=(1, 2),
    )
    count = jnp.sum(admitted, axis=1, dtype=jnp.int32)
    active = count >= plan.minimum_tracks
    valid = (
        active
        & jnp.all(jnp.isfinite(vertex), axis=-1)
        & jnp.all(jnp.isfinite(covariance), axis=(-1, -2))
    )
    return ReconstructedVertexBank(
        tracks.event_ids,
        jnp.where(active[:, None], vertex, 0.0),
        jnp.where(active[:, None, None], covariance, 0.0),
        count,
        chi_square,
        active,
        valid,
        plan.plan_id,
    )


class ParticleFlowPlan(StrictModule, NonTrainableState):
    maximum_association_distance: float = eqx.field(static=True)
    neutral_pdg_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, maximum_association_distance: float, /, *, neutral_pdg_id: int = 22
    ):
        distance = float(maximum_association_distance)
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("maximum_association_distance must be finite and positive.")
        self.maximum_association_distance = distance
        self.neutral_pdg_id = int(neutral_pdg_id)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-association-particle-flow-plan",
                "maximum_distance": distance,
                "neutral_pdg_id": int(neutral_pdg_id),
            }
        )


class ParticleFlowResult(StrictModule, NonTrainableState):
    candidates: ReconstructedParticleBank
    cluster_to_track: Array
    association_distance: Array
    association_valid: Array
    plan_id: str = eqx.field(static=True)


def build_particle_flow_candidates(
    plan: ParticleFlowPlan,
    tracks: ReconstructedTrackBank,
    charged: ReconstructedParticleBank,
    clusters: CalorimeterClusterBank,
    /,
) -> ParticleFlowResult:
    """Build a fixed nearest-association charged-plus-neutral particle-flow reference."""
    if (
        not isinstance(plan, ParticleFlowPlan)
        or not isinstance(tracks, ReconstructedTrackBank)
        or not isinstance(charged, ReconstructedParticleBank)
        or not isinstance(clusters, CalorimeterClusterBank)
    ):
        raise TypeError("Particle-flow inputs must use detector reconstruction types.")
    if not jnp.array_equal(tracks.event_ids, charged.event_ids) or not jnp.array_equal(
        tracks.event_ids, clusters.event_ids
    ):
        raise ValueError("Particle-flow event identities must align.")
    track_positions = tracks.parameters[..., :3]
    delta = clusters.positions[:, :, None, :] - track_positions[:, None, :, :]
    distance = jnp.linalg.norm(delta, axis=-1)
    eligible = (
        clusters.active[:, :, None] & tracks.active[:, None, :] & tracks.valid[:, None, :]
    )
    masked_distance = jnp.where(eligible, distance, jnp.inf)
    nearest_track = jnp.argmin(masked_distance, axis=-1)
    minimum_distance = jnp.min(masked_distance, axis=-1)
    association_valid = minimum_distance <= plan.maximum_association_distance
    neutral_active = clusters.active & clusters.valid & ~association_valid
    event_count, track_capacity = charged.active.shape
    cluster_capacity = clusters.active.shape[1]
    output_capacity = track_capacity + cluster_capacity
    pdg = jnp.zeros((event_count, output_capacity), dtype=jnp.int32)
    charges = jnp.zeros((event_count, output_capacity), dtype=charged.charges.dtype)
    momenta = jnp.zeros((event_count, output_capacity, 3), dtype=charged.momenta.dtype)
    energies = jnp.zeros((event_count, output_capacity), dtype=charged.energies.dtype)
    source_tracks = jnp.full((event_count, output_capacity), -1, dtype=jnp.int32)
    source_clusters = jnp.full((event_count, output_capacity), -1, dtype=jnp.int32)
    active = jnp.zeros((event_count, output_capacity), dtype=bool)
    pdg = pdg.at[:, :track_capacity].set(charged.pdg_hypotheses)
    charges = charges.at[:, :track_capacity].set(charged.charges)
    momenta = momenta.at[:, :track_capacity].set(charged.momenta)
    energies = energies.at[:, :track_capacity].set(charged.energies)
    source_tracks = source_tracks.at[:, :track_capacity].set(charged.source_track_indices)
    source_clusters = source_clusters.at[:, :track_capacity].set(
        charged.source_cluster_indices
    )
    active = active.at[:, :track_capacity].set(charged.active & charged.valid)
    neutral_direction = clusters.positions / jnp.maximum(
        jnp.linalg.norm(clusters.positions, axis=-1, keepdims=True),
        jnp.finfo(clusters.positions.dtype).tiny,
    )
    neutral_momentum = neutral_direction * clusters.energies[..., None]
    cluster_indices = jnp.broadcast_to(
        jnp.arange(cluster_capacity, dtype=jnp.int32), neutral_active.shape
    )
    pdg = pdg.at[:, track_capacity:].set(plan.neutral_pdg_id)
    momenta = momenta.at[:, track_capacity:].set(neutral_momentum)
    energies = energies.at[:, track_capacity:].set(clusters.energies)
    source_clusters = source_clusters.at[:, track_capacity:].set(cluster_indices)
    active = active.at[:, track_capacity:].set(neutral_active)
    candidates = ReconstructedParticleBank(
        event_ids=tracks.event_ids,
        pdg_hypotheses=pdg,
        charges=charges,
        momenta=momenta,
        energies=energies,
        source_track_indices=source_tracks,
        source_cluster_indices=source_clusters,
        active=active,
        provider_id=plan.plan_id,
    )
    return ParticleFlowResult(
        candidates, nearest_track, minimum_distance, association_valid, plan.plan_id
    )


__all__ = [
    "ParticleFlowPlan",
    "ParticleFlowResult",
    "PrimaryVertexPlan",
    "ReconstructedVertexBank",
    "build_particle_flow_candidates",
    "fit_primary_vertices",
]
