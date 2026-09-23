#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._tracking import ReconstructedTrackBank


class ReconstructedParticleBank(StrictModule, NonTrainableState):
    event_ids: Array
    pdg_hypotheses: Array
    charges: Array
    momenta: Array
    energies: Array
    source_track_indices: Array
    source_cluster_indices: Array
    active: Array
    valid: Array
    provider_id: str = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        event_ids: ArrayLike,
        pdg_hypotheses: ArrayLike,
        charges: ArrayLike,
        momenta: ArrayLike,
        energies: ArrayLike,
        source_track_indices: ArrayLike,
        source_cluster_indices: ArrayLike,
        active: ArrayLike,
        provider_id: str,
        speed_of_light: float,
    ):
        event_ids_ = jnp.asarray(event_ids)
        pdg = jnp.asarray(pdg_hypotheses, dtype=jnp.int32)
        charges_ = jnp.asarray(charges)
        momenta_ = jnp.asarray(momenta, dtype=charges_.dtype)
        energies_ = jnp.asarray(energies, dtype=charges_.dtype)
        tracks = jnp.asarray(source_track_indices, dtype=jnp.int32)
        clusters = jnp.asarray(source_cluster_indices, dtype=jnp.int32)
        active_ = jnp.asarray(active, dtype=jnp.bool_)
        if pdg.ndim != 2:
            raise ValueError("Reconstructed particles require shape (event, particle).")
        expected = pdg.shape
        if (
            any(
                value.shape != expected
                for value in (charges_, energies_, tracks, clusters, active_)
            )
            or momenta_.shape != expected + (3,)
            or event_ids_.shape != (expected[0],)
        ):
            raise ValueError("Reconstructed particle fields must align.")
        provider = str(provider_id).strip()
        light = float(speed_of_light)
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        if not math.isfinite(light) or light <= 0.0:
            raise ValueError("speed_of_light must be finite and positive.")
        valid = (
            jnp.all(jnp.isfinite(momenta_), axis=-1)
            & jnp.isfinite(energies_)
            & (energies_ >= light * jnp.linalg.norm(momenta_, axis=-1))
            & jnp.isfinite(charges_)
            & ((tracks >= 0) | (clusters >= 0))
        )
        self.event_ids = event_ids_
        self.pdg_hypotheses = pdg
        self.charges = charges_
        self.momenta = momenta_
        self.energies = energies_
        self.source_track_indices = tracks
        self.source_cluster_indices = clusters
        self.active = active_
        self.valid = jnp.where(active_, valid, True)
        self.provider_id = provider
        self.speed_of_light = light


def particles_from_straight_tracks(
    tracks: ReconstructedTrackBank,
    /,
    *,
    pdg_hypothesis: int,
    rest_energy: float,
    charge: float,
    speed_of_light: float = 1.0,
) -> ReconstructedParticleBank:
    """Construct charged candidates from the velocity part of fixed-association fits."""
    if not isinstance(tracks, ReconstructedTrackBank):
        raise TypeError("tracks must be ReconstructedTrackBank.")
    velocity = tracks.parameters[..., 3:6]
    light = float(speed_of_light)
    rest = float(rest_energy)
    if not math.isfinite(light) or light <= 0.0:
        raise ValueError("speed_of_light must be finite and positive.")
    if not math.isfinite(rest) or rest <= 0.0:
        raise ValueError("rest_energy must be finite and positive.")
    speed_squared = jnp.sum(velocity * velocity, axis=-1)
    subluminal = speed_squared < light * light
    gamma = jnp.where(
        subluminal,
        1.0 / jnp.sqrt(jnp.maximum(1.0 - speed_squared / light**2, 0.0)),
        jnp.nan,
    )
    momenta = rest * gamma[..., None] * velocity / light**2
    energy = rest * gamma
    shape = tracks.active.shape
    indices = jnp.broadcast_to(jnp.arange(shape[1], dtype=jnp.int32), shape)
    return ReconstructedParticleBank(
        event_ids=tracks.event_ids,
        pdg_hypotheses=jnp.full(shape, int(pdg_hypothesis), dtype=jnp.int32),
        charges=jnp.full(shape, float(charge)),
        momenta=momenta,
        energies=energy,
        source_track_indices=indices,
        source_cluster_indices=jnp.full(shape, -1, dtype=jnp.int32),
        active=tracks.active & tracks.valid,
        speed_of_light=light,
        provider_id=tracks.plan_id,
    )


__all__ = ["ReconstructedParticleBank", "particles_from_straight_tracks"]
