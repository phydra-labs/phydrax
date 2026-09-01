#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._types import PICCollisionResult


def _isotropic_directions(key, count, dtype):
    samples = jr.normal(key, (count, 3), dtype=dtype)
    norm = jnp.sqrt(jnp.sum(samples * samples, axis=-1))
    fallback = jnp.asarray([1.0, 0.0, 0.0], dtype=dtype)
    return jnp.where((norm > 0.0)[:, None], samples / norm[:, None], fallback)


class CoulombCollisionPlan(StrictModule, NonTrainableState):
    """Deterministic-pair stochastic binary relaxation with exact pair invariants."""

    collision_frequency: float = eqx.field(static=True)
    maximum_probability: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        collision_frequency: float,
        /,
        *,
        maximum_probability: float = 0.25,
    ):
        frequency = float(collision_frequency)
        maximum = float(maximum_probability)
        if not np.isfinite(frequency) or frequency < 0.0:
            raise ValueError("collision_frequency must be finite and nonnegative.")
        if not np.isfinite(maximum) or not 0.0 < maximum <= 1.0:
            raise ValueError("maximum_probability must lie in (0,1].")
        self.collision_frequency = frequency
        self.maximum_probability = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coulomb-binary-collision",
                "frequency": frequency,
                "maximum_probability": maximum,
            }
        )

    def collide(
        self,
        velocity: ArrayLike,
        mass: ArrayLike,
        active_mask: ArrayLike,
        incarnation: ArrayLike,
        key,
        step_size: ArrayLike,
        /,
        cell_ids: ArrayLike | None = None,
    ) -> PICCollisionResult:
        values = jnp.asarray(velocity)
        masses = jnp.asarray(mass, dtype=values.dtype)
        active = jnp.asarray(active_mask, dtype=bool)
        generation = jnp.asarray(incarnation, dtype=jnp.int32)
        if values.ndim != 2 or values.shape[-1] != 3:
            raise ValueError("velocity must have shape (capacity,3).")
        if masses.shape != active.shape or masses.shape != generation.shape:
            raise ValueError("Collision particle arrays must preserve capacity.")
        dt = jnp.asarray(step_size, dtype=values.dtype).reshape(())
        probability = 1.0 - jnp.exp(-self.collision_frequency * dt)
        stable = (
            jnp.isfinite(probability)
            & (probability >= 0.0)
            & (probability <= self.maximum_probability)
        )
        capacity = values.shape[0]
        scores = jr.uniform(key, (capacity,), dtype=values.dtype)
        identity_tie = (
            jnp.arange(capacity, dtype=values.dtype) + generation
        ) * jnp.finfo(values.dtype).eps
        cells = (
            jnp.zeros((capacity,), dtype=jnp.int32)
            if cell_ids is None
            else jnp.asarray(cell_ids, dtype=jnp.int32)
        )
        if cells.shape != (capacity,):
            raise ValueError("cell_ids must have particle-capacity shape.")
        cells = jnp.where(active, cells, jnp.iinfo(jnp.int32).max)
        order = jnp.lexsort((scores + identity_tie, cells))
        pair_count = capacity // 2
        left = order[: 2 * pair_count : 2]
        right = order[1 : 2 * pair_count : 2]
        pair_valid = active[left] & active[right] & (cells[left] == cells[right])
        collision_key, direction_key = jr.split(key)
        collided = pair_valid & (
            jr.uniform(collision_key, (pair_count,), dtype=values.dtype) < probability
        )
        m_left = masses[left]
        m_right = masses[right]
        m_total = jnp.where(m_left + m_right > 0.0, m_left + m_right, 1.0)
        center = (
            m_left[:, None] * values[left] + m_right[:, None] * values[right]
        ) / m_total[:, None]
        relative = values[left] - values[right]
        relative_speed = jnp.sqrt(jnp.sum(relative * relative, axis=-1))
        direction = _isotropic_directions(direction_key, pair_count, values.dtype)
        rotated = relative_speed[:, None] * direction
        candidate_left = center + (m_right / m_total)[:, None] * rotated
        candidate_right = center - (m_left / m_total)[:, None] * rotated
        candidate = values.at[left].set(
            jnp.where(collided[:, None], candidate_left, values[left])
        )
        candidate = candidate.at[right].set(
            jnp.where(collided[:, None], candidate_right, values[right])
        )
        candidate = jnp.where(active[:, None], candidate, 0.0)
        momentum_before = jnp.sum(masses[:, None] * values, axis=0)
        momentum_after = jnp.sum(masses[:, None] * candidate, axis=0)
        energy_before = 0.5 * jnp.sum(masses * jnp.sum(values * values, axis=-1))
        energy_after = 0.5 * jnp.sum(masses * jnp.sum(candidate * candidate, axis=-1))
        momentum_defect = jnp.sqrt(jnp.sum((momentum_after - momentum_before) ** 2))
        energy_defect = energy_after - energy_before
        finite = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.isfinite(momentum_defect)
            & jnp.isfinite(energy_defect)
        )
        tolerance = 256.0 * jnp.finfo(values.dtype).eps
        conservative = (
            momentum_defect
            <= tolerance * jnp.maximum(1.0, jnp.sqrt(jnp.sum(momentum_before**2)))
        ) & (
            jnp.abs(energy_defect) <= tolerance * jnp.maximum(1.0, jnp.abs(energy_before))
        )
        successful = stable & finite & conservative
        accepted = jnp.where(successful, candidate, values)
        return PICCollisionResult(
            candidate,
            accepted,
            collided,
            jnp.sum(pair_valid, dtype=jnp.int32),
            momentum_defect,
            energy_defect,
            jnp.zeros((3,), dtype=values.dtype),
            jnp.zeros((), dtype=values.dtype),
            probability,
            finite,
            stable,
            successful,
            self.plan_id,
        )


__all__ = ["CoulombCollisionPlan"]
