#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._curvature import ricci_tensor
from ._density import VolumeDensity
from ._metric import RiemannianMetric
from ._operator_kernels import density_divergence
from ._operators import covariant_hessian, gradient
from ._utils import _pointwise_array


class _WeightedDensityCoefficient(StrictModule):
    measure: WeightedRiemannianMeasure

    def __init__(self, measure: WeightedRiemannianMeasure, /):
        self.measure = measure

    def __call__(self, coordinates: Array, /) -> Array:
        return jnp.exp(self.measure.log_coordinate_density(coordinates))


class _WeightedLogDensityCoefficient(StrictModule):
    measure: WeightedRiemannianMeasure

    def __init__(self, measure: WeightedRiemannianMeasure, /):
        self.measure = measure

    def __call__(self, coordinates: Array, /) -> Array:
        return self.measure.log_coordinate_density(coordinates)


class _WeightedGradientField(StrictModule):
    field: Callable[[Array], Array]
    measure: WeightedRiemannianMeasure

    def __init__(
        self,
        field: Callable[[Array], Array],
        measure: WeightedRiemannianMeasure,
        /,
    ):
        self.field = field
        self.measure = measure

    def __call__(self, coordinates: Array, /) -> Array:
        return gradient(self.field, self.measure.metric, coordinates)


class WeightedRiemannianMeasure(StrictModule):
    """A positive measure ``exp(log_weight) dvol_g`` on one metric chart."""

    metric: RiemannianMetric
    log_weight_function: Callable[[Array], Array]

    def __init__(
        self,
        metric: RiemannianMetric,
        log_weight: Callable[[Array], Array],
        /,
    ):
        if not isinstance(metric, RiemannianMetric):
            raise TypeError("WeightedRiemannianMeasure requires a RiemannianMetric.")
        if not callable(log_weight):
            raise TypeError("log_weight must be callable.")
        self.metric = metric
        self.log_weight_function = log_weight

    @property
    def chart(self):
        return self.metric.chart

    def _log_weight_point(self, coordinates: Array, /) -> Array:
        value = jnp.asarray(self.log_weight_function(coordinates))
        if value.shape != ():
            raise ValueError("Pointwise log_weight must be scalar-valued.")
        return value

    def log_weight(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            self._log_weight_point,
            coordinates,
            self.chart.dimension,
        )

    def log_coordinate_density(self, coordinates: ArrayLike, /) -> Array:
        return self.metric.log_volume_density(coordinates) + self.log_weight(coordinates)

    def coordinate_density(self) -> VolumeDensity:
        return VolumeDensity(
            _WeightedDensityCoefficient(self),
            chart=self.chart,
            log_coefficient=_WeightedLogDensityCoefficient(self),
        )

    def gradient(
        self,
        field: Callable[[Array], Array],
        coordinates: ArrayLike,
        /,
    ) -> Array:
        return gradient(field, self.metric, coordinates)

    def divergence(
        self,
        field: Callable[[Array], Array],
        coordinates: ArrayLike,
        /,
    ) -> Array:
        return density_divergence(field, self.coordinate_density(), coordinates)

    def laplacian(
        self,
        field: Callable[[Array], Array],
        coordinates: ArrayLike,
        /,
    ) -> Array:
        return density_divergence(
            _WeightedGradientField(field, self),
            self.coordinate_density(),
            coordinates,
        )

    def reversible_drift(self, coordinates: ArrayLike, /) -> Array:
        return gradient(self._log_weight_point, self.metric, coordinates)

    def bakry_emery_ricci(self, coordinates: ArrayLike, /) -> Array:
        return ricci_tensor(self.metric, coordinates) - covariant_hessian(
            self._log_weight_point,
            self.metric,
            coordinates,
        )

    def score(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            jax.grad(self._log_weight_point),
            coordinates,
            self.chart.dimension,
        )


__all__ = ["WeightedRiemannianMeasure"]
