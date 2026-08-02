#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._metric import RiemannianMetric
from ._utils import _coordinates


class MetricJet(StrictModule):
    """Metric values and coordinate derivatives evaluated at common points."""

    matrix: Array
    inverse: Array
    determinant: Array
    volume_density: Array
    log_volume_density: Array
    first_derivative: Array | None
    second_derivative: Array | None
    order: int

    def __init__(
        self,
        *,
        matrix: Array,
        inverse: Array,
        determinant: Array,
        volume_density: Array,
        log_volume_density: Array,
        first_derivative: Array | None,
        second_derivative: Array | None,
        order: int,
    ):
        self.matrix = matrix
        self.inverse = inverse
        self.determinant = determinant
        self.volume_density = volume_density
        self.log_volume_density = log_volume_density
        self.first_derivative = first_derivative
        self.second_derivative = second_derivative
        self.order = int(order)


class _MetricJetEvaluator(StrictModule):
    metric_function: Callable[[Array], Array]
    dimension: int
    order: int

    def __init__(self, metric: RiemannianMetric, order: int, /):
        self.metric_function = metric.matrix_function
        self.dimension = metric.chart.dimension
        self.order = int(order)

    def __call__(self, coordinates: Array, /):
        matrix = jnp.asarray(self.metric_function(coordinates))
        expected = (self.dimension, self.dimension)
        if matrix.shape != expected:
            raise ValueError(
                f"Pointwise metric matrix must have shape {expected}; got {matrix.shape}."
            )
        identity = jnp.eye(self.dimension, dtype=matrix.dtype)
        inverse = jnp.linalg.solve(matrix, identity)
        factor = jnp.linalg.cholesky(matrix)
        diagonal = jnp.diagonal(factor)
        volume_density = jnp.prod(diagonal)
        log_volume_density = jnp.sum(jnp.log(diagonal))
        determinant = volume_density * volume_density
        if self.order == 0:
            return (
                matrix,
                inverse,
                determinant,
                volume_density,
                log_volume_density,
            )
        derivative_function = jax.jacfwd(self.metric_function)
        first_derivative = derivative_function(coordinates)
        if self.order == 1:
            return (
                matrix,
                inverse,
                determinant,
                volume_density,
                log_volume_density,
                first_derivative,
            )
        second_derivative = jax.jacfwd(derivative_function)(coordinates)
        return (
            matrix,
            inverse,
            determinant,
            volume_density,
            log_volume_density,
            first_derivative,
            second_derivative,
        )


def _evaluate_jet(
    evaluator: _MetricJetEvaluator,
    coordinates: Array,
    dimension: int,
    /,
):
    if coordinates.ndim == 1:
        return evaluator(coordinates)
    leading_shape = coordinates.shape[:-1]
    flattened = coordinates.reshape((-1, dimension))
    values = jax.vmap(evaluator)(flattened)
    return tuple(value.reshape(leading_shape + value.shape[1:]) for value in values)


def metric_jet(
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
    *,
    order: int = 2,
) -> MetricJet:
    """Evaluate a reusable metric jet up to coordinate derivative order two."""

    order_ = int(order)
    if order_ not in (0, 1, 2):
        raise ValueError("Metric jet order must be 0, 1, or 2.")
    points = _coordinates(coordinates, metric.chart.dimension)
    values = _evaluate_jet(
        _MetricJetEvaluator(metric, order_),
        points,
        metric.chart.dimension,
    )
    first_derivative = None if order_ == 0 else values[5]
    second_derivative = None if order_ < 2 else values[6]
    return MetricJet(
        matrix=values[0],
        inverse=values[1],
        determinant=values[2],
        volume_density=values[3],
        log_volume_density=values[4],
        first_derivative=first_derivative,
        second_derivative=second_derivative,
        order=order_,
    )
