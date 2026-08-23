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
from .metrix import AbstractRiemannianManifold, RiemannianMetric


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


class RiemannianFlowMatchingMetric(AbstractFlowMatchingMetric):
    """Pointwise metric velocity error in one coordinate chart."""

    metric: RiemannianMetric
    metric_id: str = eqx.field(static=True)

    def __init__(self, metric: RiemannianMetric, /):
        if not isinstance(metric, RiemannianMetric):
            raise TypeError("metric must be a RiemannianMetric.")
        self.metric = metric
        self.metric_id = canonical_fingerprint(
            {
                "kind": "riemannian-flow-matching-metric-v1",
                "chart": metric.chart.name,
                "coordinates": metric.chart.coordinates,
            }
        )

    def __call__(
        self,
        state: Array,
        predicted_velocity: Array,
        target_velocity: Array,
        /,
    ) -> Array:
        expected = (self.metric.chart.dimension,)
        if not (
            state.shape == predicted_velocity.shape == target_velocity.shape == expected
        ):
            raise ValueError(
                "Riemannian flow matching requires one chart-sized state and "
                "two chart-sized tangent vectors."
            )
        difference = predicted_velocity - target_velocity
        return jnp.asarray(
            self.metric.inner(difference, difference, state),
            dtype=float,
        ).reshape(())


class ManifoldFlowMatchingMetric(AbstractFlowMatchingMetric):
    """Intrinsic tangent error under an array-manifold metric."""

    geometry: AbstractRiemannianManifold
    metric_id: str = eqx.field(static=True)

    def __init__(self, geometry: AbstractRiemannianManifold, /):
        if not isinstance(geometry, AbstractRiemannianManifold):
            raise TypeError("geometry must be an AbstractRiemannianManifold.")
        self.geometry = geometry
        self.metric_id = canonical_fingerprint(
            {
                "kind": "manifold-flow-matching-metric-v1",
                "geometry": geometry.manifold_id,
            }
        )

    def __call__(
        self,
        state: Array,
        predicted_velocity: Array,
        target_velocity: Array,
        /,
    ) -> Array:
        if not (
            state.shape
            == predicted_velocity.shape
            == target_velocity.shape
            == self.geometry.point_shape
        ):
            raise ValueError(
                "Manifold flow matching requires one point-shaped state and tangents."
            )
        difference = self.geometry.project_tangent(
            state, predicted_velocity - target_velocity
        )
        return jnp.asarray(
            self.geometry.inner(state, difference, difference), dtype=float
        ).reshape(())


__all__ = [
    "AbstractFlowMatchingMetric",
    "EuclideanFlowMatchingMetric",
    "ManifoldFlowMatchingMetric",
    "RiemannianFlowMatchingMetric",
]
