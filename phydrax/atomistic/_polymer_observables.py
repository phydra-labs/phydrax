#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class PolymerChainLayoutPlan(StrictModule, NonTrainableState):
    """Fixed chain-to-particle slot layout for polymer observables."""

    particle_indices: Array
    chain_mask: Array
    first_indices: Array
    last_indices: Array
    maximum_frames: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_indices: ArrayLike,
        chain_mask: ArrayLike,
        /,
        *,
        maximum_frames: int,
    ):
        indices = np.asarray(particle_indices, dtype=np.int32)
        mask = np.asarray(chain_mask, dtype=bool)
        frames = int(maximum_frames)
        if (
            indices.ndim != 2
            or indices.shape != mask.shape
            or indices.shape[0] == 0
            or indices.shape[1] == 0
            or frames <= 0
            or np.any(indices[mask] < 0)
            or np.any(np.sum(mask, axis=1) == 0)
        ):
            raise ValueError("Polymer chain layout and frame capacity are invalid.")
        normalized = np.where(mask, indices, 0)
        first = np.argmax(mask, axis=1)
        last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
        self.particle_indices = jnp.asarray(normalized)
        self.chain_mask = jnp.asarray(mask)
        self.first_indices = jnp.asarray(normalized[np.arange(indices.shape[0]), first])
        self.last_indices = jnp.asarray(normalized[np.arange(indices.shape[0]), last])
        self.maximum_frames = frames
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polymer-chain-layout-plan",
                "particle_indices": normalized.tolist(),
                "chain_mask": mask.tolist(),
                "maximum_frames": frames,
            }
        )


class PolymerConformationResult(StrictModule):
    centers_of_mass: Array
    end_to_end_vectors: Array
    end_to_end_squared: Array
    gyration_tensors: Array
    radius_of_gyration_squared: Array
    chain_masses: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def polymer_conformation(
    plan: PolymerChainLayoutPlan,
    positions: ArrayLike,
    /,
    *,
    particle_masses: ArrayLike | None = None,
) -> PolymerConformationResult:
    """Evaluate conformation tensors from explicitly unwrapped coordinates."""

    if not isinstance(plan, PolymerChainLayoutPlan):
        raise TypeError("plan must be PolymerChainLayoutPlan.")
    values = jnp.asarray(positions)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("positions must have shape (frames, particles, 3).")
    frame_count, particle_count = values.shape[:2]
    if frame_count > plan.maximum_frames:
        raise ValueError("Polymer conformation input exceeds its frame capacity.")
    if int(np.max(np.asarray(plan.particle_indices))) >= particle_count:
        raise ValueError("Polymer chain layout references an absent particle slot.")
    masses = (
        jnp.ones((particle_count,), dtype=values.dtype)
        if particle_masses is None
        else jnp.asarray(particle_masses, dtype=values.dtype)
    )
    if masses.shape != (particle_count,):
        raise ValueError("particle_masses must align with the particle axis.")
    chain_positions = values[:, plan.particle_indices, :]
    chain_masses = masses[plan.particle_indices] * plan.chain_mask
    total_mass = jnp.sum(chain_masses, axis=-1)
    centers = jnp.sum(
        chain_positions * chain_masses[None, :, :, None], axis=2
    ) / jnp.maximum(total_mass[None, :, None], jnp.finfo(values.dtype).tiny)
    centered = chain_positions - centers[:, :, None, :]
    weighted_centered = centered * chain_masses[None, :, :, None]
    gyration = contract("fcni,fcnj->fcij", weighted_centered, centered) / jnp.maximum(
        total_mass[None, :, None, None], jnp.finfo(values.dtype).tiny
    )
    end_to_end = values[:, plan.last_indices, :] - values[:, plan.first_indices, :]
    end_to_end_squared = contract("fci,fci->fc", end_to_end, end_to_end)
    radius_squared = jnp.trace(gyration, axis1=-2, axis2=-1)
    successful = (
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(masses))
        & jnp.all(masses >= 0.0)
        & jnp.all(total_mass > 0.0)
        & jnp.all(jnp.isfinite(gyration))
    )
    return PolymerConformationResult(
        jnp.where(successful, centers, jnp.nan),
        jnp.where(successful, end_to_end, jnp.nan),
        jnp.where(successful, end_to_end_squared, jnp.nan),
        jnp.where(successful, gyration, jnp.nan),
        jnp.where(successful, radius_squared, jnp.nan),
        total_mass,
        successful,
        plan.plan_id,
    )


class PolymerContourStatisticsPlan(StrictModule, NonTrainableState):
    layout: PolymerChainLayoutPlan
    maximum_separation: int = eqx.field(static=True)
    contact_distance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: PolymerChainLayoutPlan,
        maximum_separation: int,
        contact_distance: float,
        /,
    ):
        separation = int(maximum_separation)
        contact = float(contact_distance)
        if (
            not isinstance(layout, PolymerChainLayoutPlan)
            or separation <= 0
            or separation >= layout.chain_mask.shape[1]
            or not math.isfinite(contact)
            or contact <= 0.0
        ):
            raise ValueError("Polymer contour-statistics configuration is invalid.")
        self.layout = layout
        self.maximum_separation = separation
        self.contact_distance = contact
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polymer-contour-statistics-plan",
                "layout": layout.plan_id,
                "maximum_separation": separation,
                "contact_distance": contact,
            }
        )


class PolymerContourStatisticsResult(StrictModule):
    contour_separations: Array
    internal_distance_squared: Array
    contact_probability: Array
    pair_counts: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def polymer_contour_statistics(
    plan: PolymerContourStatisticsPlan,
    positions: ArrayLike,
    /,
) -> PolymerContourStatisticsResult:
    """Evaluate per-chain contour-distance and contact statistics."""

    if not isinstance(plan, PolymerContourStatisticsPlan):
        raise TypeError("plan must be PolymerContourStatisticsPlan.")
    values = jnp.asarray(positions)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("positions must have shape (frames, particles, 3).")
    if values.shape[0] > plan.layout.maximum_frames:
        raise ValueError("Contour-statistics input exceeds its frame capacity.")
    if int(np.max(np.asarray(plan.layout.particle_indices))) >= values.shape[1]:
        raise ValueError("Polymer chain layout references an absent particle slot.")
    chain_positions = values[:, plan.layout.particle_indices, :]
    distance_outputs = []
    contact_outputs = []
    count_outputs = []
    cutoff_squared = plan.contact_distance * plan.contact_distance
    for separation in range(1, plan.maximum_separation + 1):
        valid = (
            plan.layout.chain_mask[:, :-separation]
            & plan.layout.chain_mask[:, separation:]
        )
        displacement = (
            chain_positions[:, :, separation:, :] - chain_positions[:, :, :-separation, :]
        )
        squared = jnp.sum(displacement * displacement, axis=-1)
        counts = jnp.sum(valid, axis=-1)
        denominator = jnp.maximum(counts, 1)
        distance_outputs.append(
            jnp.sum(jnp.where(valid[None, :, :], squared, 0.0), axis=-1)
            / denominator[None, :]
        )
        contact_outputs.append(
            jnp.sum(
                jnp.where(valid[None, :, :], squared <= cutoff_squared, False),
                axis=-1,
            )
            / denominator[None, :]
        )
        count_outputs.append(counts)
    distances = jnp.stack(distance_outputs, axis=-1)
    contacts = jnp.stack(contact_outputs, axis=-1)
    counts = jnp.stack(count_outputs, axis=-1)
    populated = counts > 0
    distances = jnp.where(populated[None, :, :], distances, jnp.nan)
    contacts = jnp.where(populated[None, :, :], contacts, jnp.nan)
    successful = jnp.all(jnp.isfinite(values)) & jnp.any(populated)
    return PolymerContourStatisticsResult(
        jnp.arange(1, plan.maximum_separation + 1, dtype=jnp.int32),
        distances,
        contacts,
        counts,
        successful,
        plan.plan_id,
    )


class DebyeScatteringPlan(StrictModule, NonTrainableState):
    wave_numbers: Array
    maximum_frames: int = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        wave_numbers: ArrayLike,
        /,
        *,
        maximum_frames: int,
        maximum_particles: int,
        block_size: int = 64,
    ):
        wave = np.asarray(wave_numbers, dtype=float)
        frames = int(maximum_frames)
        particles = int(maximum_particles)
        block = int(block_size)
        if (
            wave.ndim != 1
            or wave.size == 0
            or np.any(~np.isfinite(wave))
            or np.any(wave < 0.0)
            or frames <= 0
            or particles <= 0
            or block <= 0
        ):
            raise ValueError("Debye scattering probes and resource bounds are invalid.")
        self.wave_numbers = jnp.asarray(wave)
        self.maximum_frames = frames
        self.maximum_particles = particles
        self.block_size = block
        self.plan_id = canonical_fingerprint(
            {
                "kind": "debye-scattering-plan",
                "wave_numbers": wave.tolist(),
                "maximum_frames": frames,
                "maximum_particles": particles,
                "block_size": block,
            }
        )


class DebyeScatteringResult(StrictModule):
    wave_numbers: Array
    values: Array
    frame_values: Array
    active_frames: Array
    active_particles: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def debye_scattering(
    plan: DebyeScatteringPlan,
    positions: ArrayLike,
    /,
    *,
    particle_mask: ArrayLike | None = None,
    sample_mask: ArrayLike | None = None,
    scattering_lengths: ArrayLike | None = None,
) -> DebyeScatteringResult:
    """Evaluate isotropic Debye scattering with bounded pair-block memory."""

    if not isinstance(plan, DebyeScatteringPlan):
        raise TypeError("plan must be DebyeScatteringPlan.")
    values = jnp.asarray(positions)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("positions must have shape (frames, particles, 3).")
    frame_count, particle_count = values.shape[:2]
    if frame_count > plan.maximum_frames or particle_count > plan.maximum_particles:
        raise ValueError("Debye input exceeds its declared resource bounds.")
    particles = (
        jnp.ones((particle_count,), dtype=bool)
        if particle_mask is None
        else jnp.asarray(particle_mask, dtype=bool)
    )
    samples = (
        jnp.ones((frame_count,), dtype=bool)
        if sample_mask is None
        else jnp.asarray(sample_mask, dtype=bool)
    )
    scattering = (
        jnp.ones((particle_count,), dtype=values.dtype)
        if scattering_lengths is None
        else jnp.asarray(scattering_lengths, dtype=values.dtype)
    )
    if (
        particles.shape != (particle_count,)
        or samples.shape != (frame_count,)
        or scattering.shape != (particle_count,)
    ):
        raise ValueError("Debye masks and scattering lengths are misaligned.")
    block = plan.block_size
    padded_count = ((particle_count + block - 1) // block) * block
    pad = padded_count - particle_count
    padded_positions = jnp.pad(values, ((0, 0), (0, pad), (0, 0)))
    padded_weights = jnp.pad(jnp.where(particles, scattering, 0.0), (0, pad))
    block_count = padded_count // block
    wave = plan.wave_numbers.astype(values.dtype)
    initial = jnp.zeros((frame_count, wave.size), dtype=values.dtype)

    def accumulate_left(left_block, total):
        left_start = left_block * block
        left_positions = jax.lax.dynamic_slice(
            padded_positions, (0, left_start, 0), (frame_count, block, 3)
        )
        left_weights = jax.lax.dynamic_slice(padded_weights, (left_start,), (block,))

        def accumulate_right(right_block, subtotal):
            right_start = right_block * block
            right_positions = jax.lax.dynamic_slice(
                padded_positions, (0, right_start, 0), (frame_count, block, 3)
            )
            right_weights = jax.lax.dynamic_slice(
                padded_weights, (right_start,), (block,)
            )
            displacement = left_positions[:, :, None, :] - right_positions[:, None, :, :]
            distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
            kernel = jnp.sinc(
                wave[None, :, None, None] * distance[:, None, :, :] / jnp.pi
            )
            pair_weights = left_weights[:, None] * right_weights[None, :]
            return subtotal + jnp.sum(
                kernel * pair_weights[None, None, :, :], axis=(-2, -1)
            )

        return jax.lax.fori_loop(0, block_count, accumulate_right, total)

    frame_values = jax.lax.fori_loop(0, block_count, accumulate_left, initial)
    normalization = jnp.sum(jnp.where(particles, scattering * scattering, 0.0))
    frame_values = frame_values / jnp.maximum(normalization, jnp.finfo(values.dtype).tiny)
    active_frames = jnp.sum(samples, dtype=jnp.int32)
    averaged = jnp.sum(
        jnp.where(samples[:, None], frame_values, 0.0), axis=0
    ) / jnp.maximum(active_frames, 1)
    successful = (
        jnp.all(jnp.isfinite(jnp.where(samples[:, None, None], values, 0.0)))
        & jnp.all(jnp.isfinite(scattering))
        & jnp.all(jnp.isfinite(frame_values))
        & (active_frames > 0)
        & (jnp.sum(particles, dtype=jnp.int32) > 0)
        & (normalization > 0.0)
    )
    return DebyeScatteringResult(
        plan.wave_numbers,
        jnp.where(successful, averaged, jnp.nan),
        jnp.where(samples[:, None], frame_values, 0.0),
        active_frames,
        jnp.sum(particles, dtype=jnp.int32),
        successful,
        plan.plan_id,
    )


class PartialStructureFactorPlan(StrictModule, NonTrainableState):
    wave_vectors: Array
    site_type_count: int = eqx.field(static=True)
    maximum_frames: int = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        wave_vectors: ArrayLike,
        site_type_count: int,
        /,
        *,
        maximum_frames: int,
        maximum_particles: int,
    ):
        vectors = np.asarray(wave_vectors, dtype=float)
        types = int(site_type_count)
        frames = int(maximum_frames)
        particles = int(maximum_particles)
        if (
            vectors.ndim != 2
            or vectors.shape[0] == 0
            or vectors.shape[1] != 3
            or np.any(~np.isfinite(vectors))
            or types <= 0
            or frames <= 0
            or particles <= 0
        ):
            raise ValueError("Partial structure-factor configuration is invalid.")
        self.wave_vectors = jnp.asarray(vectors)
        self.site_type_count = types
        self.maximum_frames = frames
        self.maximum_particles = particles
        self.plan_id = canonical_fingerprint(
            {
                "kind": "partial-structure-factor-plan",
                "wave_vectors": vectors.tolist(),
                "site_type_count": types,
                "maximum_frames": frames,
                "maximum_particles": particles,
            }
        )


class PartialStructureFactorResult(StrictModule):
    wave_vectors: Array
    values: Array
    frame_values: Array
    site_counts: Array
    active_frames: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def partial_structure_factors(
    plan: PartialStructureFactorPlan,
    positions: ArrayLike,
    site_type_ids: ArrayLike,
    /,
    *,
    particle_mask: ArrayLike | None = None,
    sample_mask: ArrayLike | None = None,
    scattering_lengths: ArrayLike | None = None,
) -> PartialStructureFactorResult:
    """Return S_ab(k)=Re[A_a(k) A_b(k)*]/N for declared site channels."""

    if not isinstance(plan, PartialStructureFactorPlan):
        raise TypeError("plan must be PartialStructureFactorPlan.")
    values = jnp.asarray(positions)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("positions must have shape (frames, particles, 3).")
    frame_count, particle_count = values.shape[:2]
    if frame_count > plan.maximum_frames or particle_count > plan.maximum_particles:
        raise ValueError("Partial structure-factor input exceeds resource bounds.")
    site_types = jnp.asarray(site_type_ids, dtype=jnp.int32)
    particles = (
        jnp.ones((particle_count,), dtype=bool)
        if particle_mask is None
        else jnp.asarray(particle_mask, dtype=bool)
    )
    samples = (
        jnp.ones((frame_count,), dtype=bool)
        if sample_mask is None
        else jnp.asarray(sample_mask, dtype=bool)
    )
    scattering = (
        jnp.ones((particle_count,), dtype=values.dtype)
        if scattering_lengths is None
        else jnp.asarray(scattering_lengths, dtype=values.dtype)
    )
    if (
        site_types.shape != (particle_count,)
        or particles.shape != (particle_count,)
        or samples.shape != (frame_count,)
        or scattering.shape != (particle_count,)
    ):
        raise ValueError("Partial structure-factor metadata are misaligned.")
    valid_types = (site_types >= 0) & (site_types < plan.site_type_count)
    active = particles & valid_types
    safe_types = jnp.clip(site_types, 0, plan.site_type_count - 1)
    channels = jax.nn.one_hot(safe_types, plan.site_type_count, dtype=values.dtype)
    channels = channels * jnp.where(active, scattering, 0.0)[:, None]
    phase = contract("tnd,kd->tkn", values, plan.wave_vectors.astype(values.dtype))
    amplitudes = contract("tkn,ns->tks", jnp.exp(1j * phase), channels)
    frame_values = jnp.real(
        amplitudes[..., :, None] * jnp.conj(amplitudes[..., None, :])
    ) / jnp.maximum(jnp.sum(active), 1)
    active_frames = jnp.sum(samples, dtype=jnp.int32)
    averaged = jnp.sum(
        jnp.where(samples[:, None, None, None], frame_values, 0.0), axis=0
    ) / jnp.maximum(active_frames, 1)
    site_counts = jnp.sum(
        jax.nn.one_hot(safe_types, plan.site_type_count, dtype=jnp.int32)
        * active[:, None],
        axis=0,
    )
    successful = (
        jnp.all(~particles | valid_types)
        & jnp.all(jnp.isfinite(jnp.where(samples[:, None, None], values, 0.0)))
        & jnp.all(jnp.isfinite(scattering))
        & jnp.all(jnp.isfinite(frame_values))
        & (active_frames > 0)
        & (jnp.sum(active) > 0)
    )
    return PartialStructureFactorResult(
        plan.wave_vectors,
        jnp.where(successful, averaged, jnp.nan),
        jnp.where(samples[:, None, None, None], frame_values, 0.0),
        site_counts,
        active_frames,
        successful,
        plan.plan_id,
    )


__all__ = [
    "DebyeScatteringPlan",
    "DebyeScatteringResult",
    "PartialStructureFactorPlan",
    "PartialStructureFactorResult",
    "PolymerChainLayoutPlan",
    "PolymerConformationResult",
    "PolymerContourStatisticsPlan",
    "PolymerContourStatisticsResult",
    "debye_scattering",
    "partial_structure_factors",
    "polymer_conformation",
    "polymer_contour_statistics",
]
