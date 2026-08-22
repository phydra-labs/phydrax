#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._fingerprint import canonical_fingerprint
from ._strict import AbstractAttribute, StrictModule


class AbstractFlowMatchingMetric(StrictModule):
    """One scalar velocity error for an unbatched interpolant state."""

    metric_id: AbstractAttribute[str]

    @abstractmethod
    def __call__(
        self,
        state: Array,
        predicted_velocity: Array,
        target_velocity: Array,
        /,
    ) -> Array:
        raise NotImplementedError


class EuclideanFlowMatchingMetric(AbstractFlowMatchingMetric):
    """Squared Euclidean velocity error over the complete event."""

    normalize_event: bool = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    def __init__(self, *, normalize_event: bool = False):
        self.normalize_event = bool(normalize_event)
        self.metric_id = canonical_fingerprint(
            {
                "kind": "euclidean-flow-matching-metric-v1",
                "normalize_event": self.normalize_event,
            }
        )

    def __call__(
        self,
        state: Array,
        predicted_velocity: Array,
        target_velocity: Array,
        /,
    ) -> Array:
        if not (state.shape == predicted_velocity.shape == target_velocity.shape):
            raise ValueError("Flow-matching state and velocities must have one shape.")
        value = jnp.sum(jnp.abs(predicted_velocity - target_velocity) ** 2)
        if self.normalize_event:
            value = value / float(max(state.size, 1))
        return jnp.asarray(value, dtype=float).reshape(())


__all__ = ["AbstractFlowMatchingMetric", "EuclideanFlowMatchingMetric"]
