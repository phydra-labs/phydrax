#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._connection import geodesic_rhs
from ._metric import RiemannianMetric


class MetricGeodesicResult(StrictModule):
    """Endpoint of one fixed-step coordinate geodesic integration."""

    endpoint: Array
    final_velocity: Array
    precision_evidence: PrecisionEvidenceEnvelope
    steps: int = eqx.field(static=True)
    duration: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        endpoint: ArrayLike,
        final_velocity: ArrayLike,
        /,
        *,
        steps: int,
        duration: float,
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
    ):
        endpoint_ = jnp.asarray(endpoint)
        evidence = (
            GeometryPrecisionPolicy().evidence_for(endpoint_)
            if precision_evidence is None
            else precision_evidence
        )
        if not isinstance(evidence, PrecisionEvidenceEnvelope):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        self.endpoint = endpoint_
        self.final_velocity = jnp.asarray(final_velocity)
        self.precision_evidence = evidence
        self.steps = int(steps)
        self.duration = float(duration)
        self.method_id = "fixed-rk4-coordinate-geodesic"


class _RK4GeodesicStep(StrictModule):
    metric: RiemannianMetric
    precision: GeometryPrecisionPolicy
    step_size: float

    def __init__(
        self,
        metric: RiemannianMetric,
        step_size: float,
        precision: GeometryPrecisionPolicy,
        /,
    ):
        self.metric = metric
        self.step_size = float(step_size)
        self.precision = precision

    def __call__(self, _, state: Array) -> Array:
        self.precision.validate_coordinates(state)
        staged = self.precision.compute(state)
        step = self.precision.compute(jnp.asarray(self.step_size, dtype=state.dtype))
        first = self.precision.compute(geodesic_rhs(self.metric, staged))
        second_state = self.precision.compute(
            self.precision.accumulation(staged)
            + self.precision.accumulation(0.5 * step * first)
        )
        second = self.precision.compute(geodesic_rhs(self.metric, second_state))
        third_state = self.precision.compute(
            self.precision.accumulation(staged)
            + self.precision.accumulation(0.5 * step * second)
        )
        third = self.precision.compute(geodesic_rhs(self.metric, third_state))
        fourth_state = self.precision.compute(
            self.precision.accumulation(staged)
            + self.precision.accumulation(step * third)
        )
        fourth = self.precision.compute(geodesic_rhs(self.metric, fourth_state))
        increment = (
            self.precision.accumulation(first)
            + self.precision.accumulation(2.0 * second)
            + self.precision.accumulation(2.0 * third)
            + self.precision.accumulation(fourth)
        ) / 6.0
        result = self.precision.accumulation(staged) + self.precision.accumulation(
            step * increment
        )
        return jnp.asarray(result, dtype=state.dtype)


def integrate_metric_geodesic(
    metric: RiemannianMetric,
    point: ArrayLike,
    tangent: ArrayLike,
    /,
    *,
    duration: float = 1.0,
    steps: int = 64,
    precision: GeometryPrecisionPolicy | None = None,
) -> MetricGeodesicResult:
    """Integrate a coordinate geodesic to a fixed endpoint with RK4."""
    if not isinstance(metric, RiemannianMetric):
        raise TypeError("integrate_metric_geodesic requires a RiemannianMetric.")
    count = int(steps)
    if count <= 0:
        raise ValueError("steps must be positive.")
    duration_ = float(duration)
    if not jnp.isfinite(duration_):
        raise ValueError("duration must be finite.")
    precision_ = GeometryPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, GeometryPrecisionPolicy):
        raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
    point_array = jnp.asarray(point)
    tangent_array = jnp.asarray(tangent)
    expected = point_array.shape[:-1] + (metric.chart.dimension,)
    if point_array.shape[-1:] != (metric.chart.dimension,):
        raise ValueError("Geodesic points must match the metric chart dimension.")
    if tangent_array.shape != expected:
        raise ValueError(
            f"Geodesic tangent must have shape {expected}; got {tangent_array.shape}."
        )
    if point_array.dtype != tangent_array.dtype:
        raise TypeError("Geodesic point and tangent must have one dtype.")
    precision_.validate_coordinates(point_array)
    precision_.validate_coordinates(tangent_array)
    leading = point_array.shape[:-1]
    flat_points = point_array.reshape((-1, metric.chart.dimension))
    flat_tangents = tangent_array.reshape((-1, metric.chart.dimension))
    stepper = _RK4GeodesicStep(metric, duration_ / count, precision_)

    def integrate_one(initial_point: Array, initial_tangent: Array) -> Array:
        state = jnp.concatenate((initial_point, initial_tangent), axis=-1)
        return jax.lax.fori_loop(0, count, stepper, state)

    final = jax.vmap(integrate_one)(flat_points, flat_tangents)
    final = final.reshape(leading + (2 * metric.chart.dimension,))
    return MetricGeodesicResult(
        precision_.output(final[..., : metric.chart.dimension]),
        precision_.output(final[..., metric.chart.dimension :]),
        steps=count,
        duration=duration_,
        precision_evidence=precision_.evidence_for(point_array),
    )


__all__ = ["MetricGeodesicResult", "integrate_metric_geodesic"]
