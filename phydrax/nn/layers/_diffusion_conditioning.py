#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


class SinusoidalTimeEmbedding(eqx.Module):
    """Fixed log-spaced sine/cosine embedding for scalar diffusion time."""

    frequencies: Array
    dimension: int = eqx.field(static=True)
    maximum_frequency: float = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, maximum_frequency: float = 10_000.0):
        size = int(dimension)
        maximum = float(maximum_frequency)
        if size <= 0 or size % 2:
            raise ValueError("Time embedding dimension must be positive and even.")
        if not isfinite(maximum) or maximum <= 1.0:
            raise ValueError("maximum_frequency must be finite and exceed one.")
        half = size // 2
        frequencies = jnp.exp(jnp.linspace(0.0, jnp.log(maximum), half))
        self.frequencies = frequencies
        self.dimension = size
        self.maximum_frequency = maximum

    def __call__(self, time: ArrayLike, /) -> Array:
        value = jnp.asarray(time, dtype=self.frequencies.dtype)
        phase = value[..., None] * self.frequencies
        return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)


class TimeConditionedVectorModel(eqx.Module):
    """Adapt a vector model to a state/time score callable by feature concatenation."""

    model: Any
    embedding: SinusoidalTimeEmbedding
    state_dimension: int = eqx.field(static=True)

    def __init__(self, model: Any, state_dimension: int, embedding_dimension: int, /):
        size = int(state_dimension)
        if size <= 0 or not callable(model):
            raise ValueError("model must be callable and state_dimension positive.")
        self.model = model
        self.embedding = SinusoidalTimeEmbedding(embedding_dimension)
        self.state_dimension = size

    def __call__(self, state: ArrayLike, time: ArrayLike, /, *, key=None) -> Array:
        value = jnp.asarray(state)
        if value.shape[-1:] != (self.state_dimension,):
            raise ValueError("State does not match the conditioned model dimension.")
        embedding = self.embedding(time)
        leading = value.shape[:-1]
        embedding = jnp.broadcast_to(embedding, leading + (self.embedding.dimension,))
        features = jnp.concatenate((value, embedding), axis=-1)
        result = jnp.asarray(self.model(features, key=key))
        if result.shape != value.shape:
            raise ValueError("Conditioned score model must return the complete state shape.")
        return result


__all__ = ["SinusoidalTimeEmbedding", "TimeConditionedVectorModel"]
