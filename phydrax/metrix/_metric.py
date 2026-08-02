#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import ChartTransition, CoordinateChart
from ._utils import _pointwise_array


class RiemannianMetric(StrictModule):
    """A symmetric positive-definite metric tensor in one coordinate chart."""

    matrix_function: Callable[[Array], Array]
    chart: CoordinateChart

    def __init__(
        self,
        matrix: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
    ):
        if not callable(matrix):
            raise TypeError("Metric matrix must be callable.")
        self.matrix_function = matrix
        self.chart = chart

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        matrix = _pointwise_array(
            self.matrix_function,
            coordinates,
            self.chart.dimension,
        )
        expected = (self.chart.dimension, self.chart.dimension)
        if matrix.shape[-2:] != expected:
            raise ValueError(
                f"Metric matrix must have trailing shape {expected}; got {matrix.shape}."
            )
        return matrix

    def inverse(self, coordinates: ArrayLike, /) -> Array:
        matrix = self(coordinates)
        identity = jnp.broadcast_to(
            jnp.eye(self.chart.dimension, dtype=matrix.dtype),
            matrix.shape,
        )
        return jnp.linalg.solve(matrix, identity)

    def volume_density(self, coordinates: ArrayLike, /) -> Array:
        factor = jnp.linalg.cholesky(self(coordinates))
        return jnp.prod(jnp.diagonal(factor, axis1=-2, axis2=-1), axis=-1)

    def log_volume_density(self, coordinates: ArrayLike, /) -> Array:
        factor = jnp.linalg.cholesky(self(coordinates))
        diagonal = jnp.diagonal(factor, axis1=-2, axis2=-1)
        return jnp.sum(jnp.log(diagonal), axis=-1)

    def inner(
        self,
        left: ArrayLike,
        right: ArrayLike,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        dimension = self.chart.dimension
        if left_.shape[-1:] != (dimension,) or right_.shape[-1:] != (dimension,):
            raise ValueError(
                f"Metric inner product requires vector trailing dimension {dimension}."
            )
        return oe.contract(
            "...i,...ij,...j->...",
            left_,
            self(coordinates),
            right_,
        )

    def norm_squared(self, vector: ArrayLike, coordinates: ArrayLike, /) -> Array:
        return self.inner(vector, vector, coordinates)


class _EuclideanMetricMap(StrictModule):
    dimension: int

    def __init__(self, dimension: int, /):
        self.dimension = int(dimension)

    def __call__(self, coordinates: Array, /) -> Array:
        return jnp.eye(self.dimension, dtype=coordinates.dtype)


class _DiagonalMetricMap(StrictModule):
    diagonal: Callable[[Array], Array]
    dimension: int

    def __init__(self, diagonal: Callable[[Array], Array], dimension: int, /):
        self.diagonal = diagonal
        self.dimension = int(dimension)

    def __call__(self, coordinates: Array, /) -> Array:
        values = jnp.asarray(self.diagonal(coordinates))
        if values.shape != (self.dimension,):
            raise ValueError(
                "Pointwise diagonal metric output must have shape "
                f"{(self.dimension,)}; got {values.shape}."
            )
        return jnp.diag(values)


class _CholeskyMetricMap(StrictModule):
    model: Callable[[Array], Array]
    dimension: int
    minimum_diagonal: float

    def __init__(
        self,
        model: Callable[[Array], Array],
        dimension: int,
        minimum_diagonal: float,
        /,
    ):
        if minimum_diagonal < 0.0:
            raise ValueError("minimum_diagonal must be non-negative.")
        self.model = model
        self.dimension = int(dimension)
        self.minimum_diagonal = float(minimum_diagonal)

    def __call__(self, coordinates: Array, /) -> Array:
        raw = jnp.asarray(self.model(coordinates))
        expected = (self.dimension, self.dimension)
        if raw.shape != expected:
            raise ValueError(
                f"Pointwise Cholesky model must return shape {expected}; got {raw.shape}."
            )
        diagonal = jax.nn.softplus(jnp.diagonal(raw)) + self.minimum_diagonal
        factor = jnp.tril(raw, k=-1) + jnp.diag(diagonal)
        return factor @ jnp.swapaxes(factor, -1, -2)


class _PullbackMetricMap(StrictModule):
    target_metric: RiemannianMetric
    transition: ChartTransition

    def __init__(
        self,
        target_metric: RiemannianMetric,
        transition: ChartTransition,
        /,
    ):
        self.target_metric = target_metric
        self.transition = transition

    def __call__(self, coordinates: Array, /) -> Array:
        target_coordinates = self.transition.map_function(coordinates)
        jacobian = jax.jacfwd(self.transition.map_function)(coordinates)
        target_matrix = self.target_metric.matrix_function(target_coordinates)
        return oe.contract("ai,ab,bj->ij", jacobian, target_matrix, jacobian)


def euclidean_metric(chart: CoordinateChart, /) -> RiemannianMetric:
    return RiemannianMetric(_EuclideanMetricMap(chart.dimension), chart=chart)


def diagonal_metric(
    diagonal: Callable[[Array], Array],
    /,
    *,
    chart: CoordinateChart,
) -> RiemannianMetric:
    return RiemannianMetric(
        _DiagonalMetricMap(diagonal, chart.dimension),
        chart=chart,
    )


def cholesky_metric(
    model: Callable[[Array], Array],
    /,
    *,
    chart: CoordinateChart,
    minimum_diagonal: float = 0.0,
) -> RiemannianMetric:
    return RiemannianMetric(
        _CholeskyMetricMap(model, chart.dimension, minimum_diagonal),
        chart=chart,
    )


def pullback_metric(
    metric: RiemannianMetric,
    transition: ChartTransition,
    /,
) -> RiemannianMetric:
    if not transition.target.compatible_with(metric.chart):
        raise ValueError(
            "Pullback transition target chart must match the metric chart; got "
            f"{transition.target.name!r} and {metric.chart.name!r}."
        )
    return RiemannianMetric(
        _PullbackMetricMap(metric, transition),
        chart=transition.source,
    )
