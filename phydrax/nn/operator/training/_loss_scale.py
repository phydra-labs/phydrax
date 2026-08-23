#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...._strict import StrictModule


class OperatorLossScaleState(StrictModule):
    """Dynamic loss scale and finite-update evidence carried through checkpoints."""

    scale: Array
    consecutive_finite_updates: Array
    nonfinite_microsteps: Array

    def __init__(
        self,
        scale: Any,
        consecutive_finite_updates: Any = 0,
        nonfinite_microsteps: Any = 0,
    ):
        scale_array = jnp.asarray(scale)
        finite_updates = jnp.asarray(consecutive_finite_updates, dtype=jnp.int32)
        nonfinite = jnp.asarray(nonfinite_microsteps, dtype=jnp.int32)
        if scale_array.ndim != 0:
            raise ValueError("Loss scale must be scalar.")
        if finite_updates.ndim != 0 or nonfinite.ndim != 0:
            raise ValueError("Loss-scale counters must be scalar.")
        self.scale = scale_array
        self.consecutive_finite_updates = finite_updates
        self.nonfinite_microsteps = nonfinite


@dataclass(frozen=True, slots=True)
class OperatorLossScalePolicy:
    """Static or dynamic scaling for float16 operator gradients."""

    initial_scale: float = 32768.0
    dynamic: bool = True
    growth_interval: int = 2000
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    minimum_scale: float = 1.0
    maximum_scale: float = 16777216.0

    def __post_init__(self):
        values = (
            self.initial_scale,
            self.growth_factor,
            self.backoff_factor,
            self.minimum_scale,
            self.maximum_scale,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Loss-scale configuration must be finite.")
        if self.initial_scale <= 0.0:
            raise ValueError("initial_scale must be positive.")
        if int(self.growth_interval) <= 0:
            raise ValueError("growth_interval must be positive.")
        if self.growth_factor <= 1.0:
            raise ValueError("growth_factor must exceed one.")
        if not 0.0 < self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must lie strictly between zero and one.")
        if self.minimum_scale <= 0.0 or self.maximum_scale < self.minimum_scale:
            raise ValueError("Loss-scale bounds are invalid.")
        if not self.minimum_scale <= self.initial_scale <= self.maximum_scale:
            raise ValueError("initial_scale must lie within the configured bounds.")

    def initial_state(self, dtype: Any, /) -> OperatorLossScaleState:
        return OperatorLossScaleState(jnp.asarray(self.initial_scale, dtype=dtype))

    def scale_loss(
        self,
        loss: Array,
        state: OperatorLossScaleState,
        /,
    ) -> Array:
        return loss * state.scale

    def unscale_gradients(
        self,
        gradients: Any,
        state: OperatorLossScaleState,
        /,
    ) -> Any:
        inverse = jnp.reciprocal(state.scale)
        return jax.tree_util.tree_map(
            lambda gradient: (gradient * inverse).astype(gradient.dtype),
            gradients,
        )

    def on_finite_update(
        self,
        state: OperatorLossScaleState,
        /,
    ) -> OperatorLossScaleState:
        if not self.dynamic:
            return state
        count = state.consecutive_finite_updates + jnp.asarray(1, dtype=jnp.int32)
        grow = count >= int(self.growth_interval)
        candidate = state.scale * jnp.asarray(
            self.growth_factor,
            dtype=state.scale.dtype,
        )
        maximum = jnp.asarray(self.maximum_scale, dtype=state.scale.dtype)
        scale = jnp.where(grow, jnp.minimum(candidate, maximum), state.scale)
        count = jnp.where(grow, jnp.zeros_like(count), count)
        return OperatorLossScaleState(scale, count, state.nonfinite_microsteps)

    def on_nonfinite_microstep(
        self,
        state: OperatorLossScaleState,
        /,
    ) -> OperatorLossScaleState:
        if not self.dynamic:
            return state
        minimum = jnp.asarray(self.minimum_scale, dtype=state.scale.dtype)
        backed_off = state.scale * jnp.asarray(
            self.backoff_factor,
            dtype=state.scale.dtype,
        )
        return OperatorLossScaleState(
            jnp.maximum(minimum, backed_off),
            jnp.zeros_like(state.consecutive_finite_updates),
            state.nonfinite_microsteps + jnp.asarray(1, dtype=jnp.int32),
        )


def tree_all_finite(tree: Any, /) -> Array:
    """Return one scalar JAX predicate over every numeric PyTree leaf."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))


__all__ = [
    "OperatorLossScalePolicy",
    "OperatorLossScaleState",
]
