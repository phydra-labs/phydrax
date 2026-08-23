#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import ChartTransition, CoordinateChart
from ._map import DifferentiableMap
from ._metric import AbstractSemiRiemannianMetric
from ._utils import _pointwise_array


class VolumeDensity(StrictModule):
    """Positive coordinate coefficient of a volume density in one chart."""

    coefficient_function: Callable[[Array], Array]
    log_coefficient_function: Callable[[Array], Array] | None
    chart: CoordinateChart

    def __init__(
        self,
        coefficient: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
        log_coefficient: Callable[[Array], Array] | None = None,
    ):
        if not callable(coefficient):
            raise TypeError("Volume density coefficient must be callable.")
        if log_coefficient is not None and not callable(log_coefficient):
            raise TypeError("log_coefficient must be callable when supplied.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Volume density chart must be a CoordinateChart.")
        self.coefficient_function = coefficient
        self.log_coefficient_function = log_coefficient
        self.chart = chart

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        values = _pointwise_array(
            self.coefficient_function,
            coordinates,
            self.chart.dimension,
        )
        points = jnp.asarray(coordinates)
        expected = points.shape[:-1]
        if values.shape != expected:
            raise ValueError(
                f"Volume density must be scalar-valued with shape {expected}; "
                f"got {values.shape}."
            )
        return values

    def log_value(self, coordinates: ArrayLike, /) -> Array:
        if self.log_coefficient_function is None:
            values = self(coordinates)
            return jnp.log(values)
        result = _pointwise_array(
            self.log_coefficient_function,
            coordinates,
            self.chart.dimension,
        )
        points = jnp.asarray(coordinates)
        expected = points.shape[:-1]
        if result.shape != expected:
            raise ValueError(
                "Log volume density must be scalar-valued with shape "
                f"{expected}; got {result.shape}."
            )
        return result


class _MetricDensityCoefficient(StrictModule):
    metric: AbstractSemiRiemannianMetric

    def __init__(self, metric: AbstractSemiRiemannianMetric, /):
        self.metric = metric

    def __call__(self, coordinates: Array, /) -> Array:
        return self.metric.volume_density(coordinates)


class _MetricLogDensityCoefficient(StrictModule):
    metric: AbstractSemiRiemannianMetric

    def __init__(self, metric: AbstractSemiRiemannianMetric, /):
        self.metric = metric

    def __call__(self, coordinates: Array, /) -> Array:
        return self.metric.log_volume_density(coordinates)


class _PullbackDensityCoefficient(StrictModule):
    density: VolumeDensity
    map: DifferentiableMap | ChartTransition
    logarithmic: bool

    def __init__(
        self,
        density: VolumeDensity,
        map: DifferentiableMap | ChartTransition,
        /,
        *,
        logarithmic: bool,
    ):
        self.density = density
        self.map = map
        self.logarithmic = bool(logarithmic)

    def __call__(self, coordinates: Array, /) -> Array:
        target_coordinates = self.map.map_function(coordinates)
        jacobian = self.map.jacobian(coordinates)
        _, log_absolute_determinant = jnp.linalg.slogdet(jacobian)
        if self.logarithmic:
            return self.density.log_value(target_coordinates) + log_absolute_determinant
        return self.density(target_coordinates) * jnp.exp(log_absolute_determinant)


def metric_volume_density(metric: AbstractSemiRiemannianMetric, /) -> VolumeDensity:
    """Return the canonical positive density ``sqrt(abs(det(g)))`` of a metric."""
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("metric_volume_density requires a nondegenerate metric.")
    return VolumeDensity(
        _MetricDensityCoefficient(metric),
        chart=metric.chart,
        log_coefficient=_MetricLogDensityCoefficient(metric),
    )


def pullback_density(
    density: VolumeDensity,
    map: DifferentiableMap | ChartTransition,
    /,
) -> VolumeDensity:
    """Pull an equal-dimensional volume density into the source chart."""
    if not isinstance(density, VolumeDensity):
        raise TypeError("density must be a VolumeDensity.")
    if not isinstance(map, (DifferentiableMap, ChartTransition)):
        raise TypeError("map must be a DifferentiableMap or ChartTransition.")
    if not map.target.compatible_with(density.chart):
        raise ValueError("Map target chart must match the volume-density chart.")
    if map.source.dimension != map.target.dimension:
        raise ValueError("Volume-density pullback requires equal chart dimensions.")
    return VolumeDensity(
        _PullbackDensityCoefficient(density, map, logarithmic=False),
        chart=map.source,
        log_coefficient=_PullbackDensityCoefficient(density, map, logarithmic=True),
    )


class VolumeDensityValidationReport(StrictModule):
    """Aggregate diagnostics for a positive coordinate density."""

    valid: Array
    finite: Array
    positive: Array
    finite_log: Array
    maximum_log_residual: Array

    def __init__(
        self,
        *,
        valid: ArrayLike,
        finite: ArrayLike,
        positive: ArrayLike,
        finite_log: ArrayLike,
        maximum_log_residual: ArrayLike,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.positive = jnp.asarray(positive, dtype=bool)
        self.finite_log = jnp.asarray(finite_log, dtype=bool)
        self.maximum_log_residual = jnp.asarray(maximum_log_residual)


def validate_volume_density(
    density: VolumeDensity,
    points: ArrayLike,
    /,
    *,
    log_consistency_tolerance: float = 1e-8,
    raise_on_error: bool = True,
) -> VolumeDensityValidationReport:
    """Validate positivity, finiteness, and an explicit log coefficient."""
    if not isinstance(density, VolumeDensity):
        raise TypeError("validate_volume_density requires a VolumeDensity.")
    if log_consistency_tolerance < 0.0:
        raise ValueError("log_consistency_tolerance must be non-negative.")
    coefficients = density(points)
    log_coefficients = density.log_value(points)
    finite = jnp.all(jnp.isfinite(coefficients))
    positive = jnp.all(coefficients > 0)
    finite_log = jnp.all(jnp.isfinite(log_coefficients))
    if density.log_coefficient_function is None:
        maximum_log_residual = jnp.asarray(0.0, dtype=coefficients.dtype)
    else:
        safe_coefficients = jnp.where(coefficients > 0, coefficients, 1)
        maximum_log_residual = jnp.max(
            jnp.abs(jnp.log(safe_coefficients) - log_coefficients)
        )
    valid = (
        finite
        & positive
        & finite_log
        & (maximum_log_residual <= log_consistency_tolerance)
    )
    report = VolumeDensityValidationReport(
        valid=valid,
        finite=finite,
        positive=positive,
        finite_log=finite_log,
        maximum_log_residual=maximum_log_residual,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Volume-density validation failed: "
            f"finite={bool(jax.device_get(finite))}, "
            f"positive={bool(jax.device_get(positive))}, "
            f"finite_log={bool(jax.device_get(finite_log))}, "
            "maximum_log_residual="
            f"{float(jax.device_get(maximum_log_residual))}."
        )
    return report


__all__ = [
    "VolumeDensity",
    "VolumeDensityValidationReport",
    "metric_volume_density",
    "pullback_density",
    "validate_volume_density",
]
