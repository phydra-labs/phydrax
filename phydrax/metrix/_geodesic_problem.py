#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._connection import geodesic_rhs
from ._metric import RiemannianMetric


class MetricGeodesicResult(StrictModule):
    """Endpoint of one fixed-step coordinate geodesic integration."""

    endpoint: Array
    final_velocity: Array
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
    ):
        self.endpoint = jnp.asarray(endpoint)
        self.final_velocity = jnp.asarray(final_velocity)
        self.steps = int(steps)
        self.duration = float(duration)
        self.method_id = "fixed-rk4-coordinate-geodesic"


class _RK4GeodesicStep(StrictModule):
    metric: RiemannianMetric
    step_size: float

    def __init__(self, metric: RiemannianMetric, step_size: float, /):
        self.metric = metric
        self.step_size = float(step_size)

    def __call__(self, _, state: Array) -> Array:
        step = jnp.asarray(self.step_size, dtype=state.dtype)
        first = geodesic_rhs(self.metric, state)
        second = geodesic_rhs(self.metric, state + 0.5 * step * first)
        third = geodesic_rhs(self.metric, state + 0.5 * step * second)
        fourth = geodesic_rhs(self.metric, state + step * third)
        return state + step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0


def integrate_metric_geodesic(
    metric: RiemannianMetric,
    point: ArrayLike,
    tangent: ArrayLike,
    /,
    *,
    duration: float = 1.0,
    steps: int = 64,
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
    point_array = jnp.asarray(point)
    tangent_array = jnp.asarray(tangent)
    expected = point_array.shape[:-1] + (metric.chart.dimension,)
    if point_array.shape[-1:] != (metric.chart.dimension,):
        raise ValueError("Geodesic points must match the metric chart dimension.")
    if tangent_array.shape != expected:
        raise ValueError(
            f"Geodesic tangent must have shape {expected}; got {tangent_array.shape}."
        )
    leading = point_array.shape[:-1]
    flat_points = point_array.reshape((-1, metric.chart.dimension))
    flat_tangents = tangent_array.reshape((-1, metric.chart.dimension))
    stepper = _RK4GeodesicStep(metric, duration_ / count)

    def integrate_one(initial_point: Array, initial_tangent: Array) -> Array:
        state = jnp.concatenate((initial_point, initial_tangent), axis=-1)
        return jax.lax.fori_loop(0, count, stepper, state)

    final = jax.vmap(integrate_one)(flat_points, flat_tangents)
    final = final.reshape(leading + (2 * metric.chart.dimension,))
    return MetricGeodesicResult(
        final[..., : metric.chart.dimension],
        final[..., metric.chart.dimension :],
        steps=count,
        duration=duration_,
    )


__all__ = ["MetricGeodesicResult", "integrate_metric_geodesic"]
