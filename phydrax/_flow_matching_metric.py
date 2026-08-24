#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ._fingerprint import canonical_fingerprint
from ._geometry_precision import GeometryPrecisionPolicy
from ._strict import AbstractAttribute, StrictModule
from .metrix import AbstractRiemannianManifold, RiemannianMetric


class AbstractFlowMatchingMetric(StrictModule):
    """One scalar velocity error for an unbatched interpolant state."""

    metric_id: AbstractAttribute[str]
    precision: AbstractAttribute[GeometryPrecisionPolicy]

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
    precision: GeometryPrecisionPolicy
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        normalize_event: bool = False,
        precision: GeometryPrecisionPolicy | None = None,
    ):
        precision_ = GeometryPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
        self.normalize_event = bool(normalize_event)
        self.precision = precision_
        self.metric_id = canonical_fingerprint(
            {
                "kind": "euclidean-flow-matching-metric-v2",
                "normalize_event": self.normalize_event,
                "precision_policy_id": precision_.policy_id,
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
        self.precision.validate_coordinates(state)
        difference = self.precision.compute(predicted_velocity) - self.precision.compute(
            target_velocity
        )
        value = self.precision.sum(jnp.abs(self.precision.accumulation(difference)) ** 2)
        if self.normalize_event:
            value = value / jnp.asarray(max(state.size, 1), dtype=value.dtype)
        return self.precision.decision(value).reshape(())


class RiemannianFlowMatchingMetric(AbstractFlowMatchingMetric):
    """Pointwise metric velocity error in one coordinate chart."""

    metric: RiemannianMetric
    precision: GeometryPrecisionPolicy
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        metric: RiemannianMetric,
        /,
        *,
        precision: GeometryPrecisionPolicy | None = None,
    ):
        if not isinstance(metric, RiemannianMetric):
            raise TypeError("metric must be a RiemannianMetric.")
        precision_ = GeometryPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
        self.metric = metric
        self.precision = precision_
        self.metric_id = canonical_fingerprint(
            {
                "kind": "riemannian-flow-matching-metric-v2",
                "chart": metric.chart.name,
                "coordinates": metric.chart.coordinates,
                "precision_policy_id": precision_.policy_id,
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
        self.precision.validate_coordinates(state)
        coordinates = self.precision.compute(state)
        difference = self.precision.compute(predicted_velocity) - self.precision.compute(
            target_velocity
        )
        accumulated_difference = self.precision.accumulation(difference)
        matrix = self.precision.accumulation(
            self.precision.compute(self.metric(coordinates))
        )
        value = oe.contract(
            "i,ij,j->",
            jnp.conj(accumulated_difference),
            matrix,
            accumulated_difference,
        )
        return self.precision.decision(jnp.real(value)).reshape(())


class ManifoldFlowMatchingMetric(AbstractFlowMatchingMetric):
    """Intrinsic tangent error under an array-manifold metric."""

    geometry: AbstractRiemannianManifold
    precision: GeometryPrecisionPolicy
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: AbstractRiemannianManifold,
        /,
        *,
        precision: GeometryPrecisionPolicy | None = None,
    ):
        if not isinstance(geometry, AbstractRiemannianManifold):
            raise TypeError("geometry must be an AbstractRiemannianManifold.")
        precision_ = GeometryPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
        self.geometry = geometry
        self.precision = precision_
        self.metric_id = canonical_fingerprint(
            {
                "kind": "manifold-flow-matching-metric-v2",
                "geometry": geometry.manifold_id,
                "precision_policy_id": precision_.policy_id,
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
        self.precision.validate_coordinates(state)
        computed_state = self.precision.accumulation(self.precision.compute(state))
        difference = self.geometry.project_tangent(
            computed_state,
            self.precision.accumulation(
                self.precision.compute(predicted_velocity)
                - self.precision.compute(target_velocity)
            ),
        )
        value = self.geometry.inner(
            computed_state,
            difference,
            difference,
        )
        return self.precision.decision(jnp.real(value)).reshape(())


__all__ = [
    "AbstractFlowMatchingMetric",
    "EuclideanFlowMatchingMetric",
    "ManifoldFlowMatchingMetric",
    "RiemannianFlowMatchingMetric",
]
