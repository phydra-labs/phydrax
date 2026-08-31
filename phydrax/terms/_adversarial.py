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


class ImplicitGenerator(StrictModule):
    """Sample-only generator with no normalized-density claim."""

    generator: Any
    event_shape: tuple[int, ...] = eqx.field(static=True)
    generator_id: str = eqx.field(static=True)

    def __init__(self, generator, event_shape, /, *, generator_id: str):
        if not callable(generator) or not generator_id:
            raise TypeError("generator must be callable with a non-empty ID.")
        shape = tuple(int(size) for size in event_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("event_shape must contain positive dimensions.")
        self.generator = generator
        self.event_shape = shape
        self.generator_id = generator_id

    def sample(self, key: Key[Array, ""], sample_shape, /) -> Array:
        samples = tuple(int(size) for size in sample_shape)
        value = jnp.asarray(self.generator(key, samples))
        expected = samples + self.event_shape
        if value.shape != expected:
            raise ValueError(f"Implicit generator must return shape {expected}.")
        return value


class AdversarialEvaluation(StrictModule):
    critic_loss: Array
    generator_loss: Array
    gradient_penalty: Array
    real_score: Array
    fake_score: Array
    finite: Array
    objective_id: str = eqx.field(static=True)


def wasserstein_adversarial_evaluation(
    critic: Any,
    real: ArrayLike,
    fake: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    gradient_penalty_weight: float = 0.0,
    objective_id: str | None = None,
) -> AdversarialEvaluation:
    """Evaluate WGAN losses and optional interpolation gradient penalty."""
    if not callable(critic):
        raise TypeError("critic must be callable.")
    real_array = jnp.asarray(real)
    if jnp.iscomplexobj(real_array) or not jnp.issubdtype(
        real_array.dtype, jnp.floating
    ):
        raise TypeError("Wasserstein adversarial evaluation requires real floating events.")
    fake_array = jnp.asarray(fake, dtype=real_array.dtype)
    if real_array.shape != fake_array.shape or real_array.ndim < 2:
        raise ValueError("Real and fake events require identical sample-first shapes.")
    weight = float(gradient_penalty_weight)
    if not jnp.isfinite(weight) or weight < 0.0:
        raise ValueError("gradient_penalty_weight must be finite and nonnegative.")
    real_keys = jr.split(jr.fold_in(key, 0), real_array.shape[0])
    fake_keys = jr.split(jr.fold_in(key, 1), fake_array.shape[0])
    real_score = jax.vmap(lambda value, local: jnp.asarray(critic(value, key=local)).reshape(()))(
        real_array, real_keys
    )
    fake_score = jax.vmap(lambda value, local: jnp.asarray(critic(value, key=local)).reshape(()))(
        fake_array, fake_keys
    )
    penalty = jnp.asarray(0.0, dtype=real_array.dtype)
    if weight > 0.0:
        alpha = jr.uniform(
            jr.fold_in(key, 2),
            (real_array.shape[0],) + (1,) * (real_array.ndim - 1),
            dtype=real_array.dtype,
        )
        interpolated = alpha * real_array + (1.0 - alpha) * fake_array
        penalty_keys = jr.split(jr.fold_in(key, 3), real_array.shape[0])

        def gradient_norm(value, local):
            gradient = jax.grad(
                lambda current: jnp.asarray(critic(current, key=local)).reshape(())
            )(value)
            return jnp.sqrt(jnp.sum(gradient**2))

        norms = jax.vmap(gradient_norm)(interpolated, penalty_keys)
        penalty = weight * jnp.mean((norms - 1.0) ** 2)
    critic_loss = jnp.mean(fake_score) - jnp.mean(real_score) + penalty
    generator_loss = -jnp.mean(fake_score)
    finite = (
        jnp.isfinite(critic_loss)
        & jnp.isfinite(generator_loss)
        & jnp.isfinite(penalty)
    )
    identifier = objective_id or canonical_fingerprint(
        {
            "kind": "wasserstein-adversarial-objective",
            "event_shape": list(real_array.shape[1:]),
            "gradient_penalty_weight": weight,
        }
    )
    return AdversarialEvaluation(
        critic_loss,
        generator_loss,
        penalty,
        real_score,
        fake_score,
        finite,
        identifier,
    )


__all__ = [
    "AdversarialEvaluation",
    "ImplicitGenerator",
    "wasserstein_adversarial_evaluation",
]
