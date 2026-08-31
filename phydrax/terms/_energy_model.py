#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


class EnergyTarget(StrictModule):
    """Unnormalized finite-dimensional energy target without a partition-function claim."""

    energy: Any
    support: Any
    event_shape: tuple[int, ...] = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    normalizer_status: str = eqx.field(static=True)

    def __init__(
        self,
        energy,
        event_shape,
        /,
        *,
        support=None,
        temperature: float = 1.0,
        target_id: str | None = None,
        normalizer_status: str = "unknown",
    ):
        if not callable(energy):
            raise TypeError("energy must be callable.")
        shape = tuple(int(size) for size in event_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("event_shape must contain positive dimensions.")
        value = float(temperature)
        if not jnp.isfinite(value) or value <= 0.0:
            raise ValueError("temperature must be finite and positive.")
        if normalizer_status not in ("unknown", "estimated", "exact"):
            raise ValueError("Unknown normalizer status.")
        identifier = target_id or canonical_fingerprint(
            {
                "kind": "energy-target",
                "event_shape": list(shape),
                "temperature": value,
                "normalizer_status": normalizer_status,
            }
        )
        self.energy = energy
        self.support = support
        self.event_shape = shape
        self.temperature = value
        self.target_id = identifier
        self.normalizer_status = normalizer_status

    def contains(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        rank = len(self.event_shape)
        if array.ndim < rank or tuple(array.shape[-rank:]) != self.event_shape:
            raise ValueError("Energy value has an incompatible event shape.")
        finite = jnp.all(
            jnp.isfinite(array), axis=tuple(range(array.ndim - rank, array.ndim))
        )
        if self.support is None:
            return finite
        return finite & jnp.asarray(self.support(array), dtype=bool)

    def energy_value(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        energy = jnp.asarray(self.energy(array))
        expected = array.shape[: -len(self.event_shape)]
        if energy.shape != expected:
            raise ValueError("Energy callable must return one scalar per event.")
        return jnp.where(self.contains(array), energy, jnp.inf)

    def log_unnormalized(self, value: ArrayLike, /) -> Array:
        return -self.energy_value(value) / self.temperature

    def score(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if array.shape != self.event_shape:
            raise ValueError("One energy score evaluation requires one unbatched event.")
        return -jax.grad(lambda current: self.energy_value(current))(array) / self.temperature


class PersistentEnergyState(StrictModule):
    particles: Array
    step_index: Array
    root_key: Array
    valid: Array
    target_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)


class PersistentContrastiveDivergence(StrictModule):
    """Fixed-capacity replayable unadjusted Langevin negative sampler."""

    target: EnergyTarget
    reference_sampler: Any
    step_size: float = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    refresh_probability: float = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)

    def __init__(
        self,
        target: EnergyTarget,
        reference_sampler,
        /,
        *,
        step_size: float,
        num_steps: int,
        refresh_probability: float = 0.05,
    ):
        if not isinstance(target, EnergyTarget) or not callable(reference_sampler):
            raise TypeError("PCD requires an EnergyTarget and reference sampler.")
        step = float(step_size)
        count = int(num_steps)
        refresh = float(refresh_probability)
        if not jnp.isfinite(step) or step <= 0.0 or count <= 0:
            raise ValueError("PCD step_size and num_steps must be positive.")
        if not jnp.isfinite(refresh) or refresh < 0.0 or refresh > 1.0:
            raise ValueError("refresh_probability must lie in [0, 1].")
        self.target = target
        self.reference_sampler = reference_sampler
        self.step_size = step
        self.num_steps = count
        self.refresh_probability = refresh
        self.sampler_id = canonical_fingerprint(
            {
                "kind": "persistent-contrastive-divergence",
                "target_id": target.target_id,
                "step_size": step,
                "num_steps": count,
                "refresh_probability": refresh,
            }
        )

    def initialize(self, key: Key[Array, ""], particle_count: int, /):
        count = int(particle_count)
        if count <= 0:
            raise ValueError("particle_count must be positive.")
        particles = jnp.asarray(self.reference_sampler(key, (count,)))
        expected = (count,) + self.target.event_shape
        if particles.shape != expected:
            raise ValueError(f"Reference sampler must return shape {expected}.")
        valid = self.target.contains(particles)
        identifier = canonical_fingerprint(
            {
                "kind": "persistent-energy-state",
                "sampler_id": self.sampler_id,
                "particle_count": count,
            }
        )
        return PersistentEnergyState(
            particles,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(key),
            valid,
            self.target.target_id,
            identifier,
        )

    def advance(self, state: PersistentEnergyState, /) -> PersistentEnergyState:
        if not isinstance(state, PersistentEnergyState) or state.target_id != self.target.target_id:
            raise ValueError("Persistent energy state does not match this target.")
        refresh_key = jr.fold_in(state.root_key, state.step_index * 3)
        dynamics_key = jr.fold_in(state.root_key, state.step_index * 3 + 1)
        mask_key = jr.fold_in(state.root_key, state.step_index * 3 + 2)
        fresh = jnp.asarray(self.reference_sampler(refresh_key, (state.particles.shape[0],)))
        refresh = jr.bernoulli(
            mask_key,
            self.refresh_probability,
            (state.particles.shape[0],),
        )
        expanded = refresh.reshape(refresh.shape + (1,) * len(self.target.event_shape))
        initial = jnp.where(expanded, fresh, state.particles)

        def one_step(particles, key):
            keys = jr.split(key, particles.shape[0])
            gradients = jax.vmap(
                lambda particle: jax.grad(self.target.energy_value)(particle)
            )(particles)
            noise = jax.vmap(
                lambda local_key: jr.normal(
                    local_key,
                    self.target.event_shape,
                    dtype=particles.dtype,
                )
            )(keys)
            updated = particles - self.step_size * gradients / self.target.temperature
            return updated + jnp.sqrt(2.0 * self.step_size) * noise, None

        step_keys = jr.split(dynamics_key, self.num_steps)
        final, _ = jax.lax.scan(one_step, initial, step_keys)
        valid = self.target.contains(final)
        return PersistentEnergyState(
            final,
            state.step_index + 1,
            state.root_key,
            valid,
            state.target_id,
            state.state_id,
        )

    def contrastive_loss(self, data: ArrayLike, state: PersistentEnergyState, /) -> Array:
        if (
            not isinstance(state, PersistentEnergyState)
            or state.target_id != self.target.target_id
        ):
            raise ValueError("Persistent energy state does not match this target.")
        observations = jnp.asarray(data)
        if observations.ndim != len(self.target.event_shape) + 1:
            raise ValueError("Energy training data require one leading sample axis.")
        expected = (observations.shape[0],) + self.target.event_shape
        if observations.shape != expected:
            raise ValueError("Energy training data require one leading sample axis.")
        if not bool(jnp.all(self.target.contains(observations))):
            raise ValueError("Energy training data contain invalid events.")
        negatives = jax.lax.stop_gradient(state.particles)
        return jnp.mean(self.target.energy_value(observations)) - jnp.mean(
            self.target.energy_value(negatives)
        )


__all__ = [
    "EnergyTarget",
    "PersistentContrastiveDivergence",
    "PersistentEnergyState",
]
