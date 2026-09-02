#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._metric import _metric_inverse, AbstractSemiRiemannianMetric, RiemannianMetric
from ._utils import _coordinates


class MetricJet(StrictModule):
    """Metric values and coordinate derivatives evaluated at common points."""

    matrix: Array
    inverse: Array
    determinant: Array
    determinant_sign: Array
    log_abs_determinant: Array
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
        determinant_sign: Array,
        log_abs_determinant: Array,
        volume_density: Array,
        log_volume_density: Array,
        first_derivative: Array | None,
        second_derivative: Array | None,
        order: int,
    ):
        self.matrix = matrix
        self.inverse = inverse
        self.determinant = determinant
        self.determinant_sign = determinant_sign
        self.log_abs_determinant = log_abs_determinant
        self.volume_density = volume_density
        self.log_volume_density = log_volume_density
        self.first_derivative = first_derivative
        self.second_derivative = second_derivative
        self.order = int(order)


class _MetricJetEvaluator(StrictModule):
    metric_function: Callable[[Array], Array]
    dimension: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    positive_definite: bool = eqx.field(static=True)

    def __init__(self, metric: AbstractSemiRiemannianMetric, order: int, /):
        self.metric_function = metric.matrix_function
        self.dimension = metric.chart.dimension
        self.order = int(order)
        self.positive_definite = isinstance(metric, RiemannianMetric)

    def __call__(self, coordinates: Array, /):
        matrix = jnp.asarray(self.metric_function(coordinates))
        expected = (self.dimension, self.dimension)
        if matrix.shape != expected:
            raise ValueError(
                f"Pointwise metric matrix must have shape {expected}; got {matrix.shape}."
            )
        inverse = _metric_inverse(
            matrix,
            positive_definite=self.positive_definite,
        )
        if self.positive_definite:
            factor = jnp.linalg.cholesky(matrix)
            log_abs_determinant = 2.0 * jnp.sum(jnp.log(jnp.diagonal(factor)))
            determinant_sign = jnp.asarray(1.0, dtype=matrix.dtype)
        else:
            determinant_sign, log_abs_determinant = jnp.linalg.slogdet(matrix)
        log_volume_density = 0.5 * log_abs_determinant
        volume_density = jnp.exp(log_volume_density)
        determinant = determinant_sign * jnp.exp(log_abs_determinant)
        if self.order == 0:
            return (
                matrix,
                inverse,
                determinant,
                determinant_sign,
                log_abs_determinant,
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
                determinant_sign,
                log_abs_determinant,
                volume_density,
                log_volume_density,
                first_derivative,
            )
        second_derivative = jax.jacfwd(derivative_function)(coordinates)
        return (
            matrix,
            inverse,
            determinant,
            determinant_sign,
            log_abs_determinant,
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
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
    *,
    order: int = 2,
) -> MetricJet:
    """Evaluate a reusable nondegenerate metric jet through derivative order two."""
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("metric_jet requires a nondegenerate metric.")
    order_value = int(order)
    if order_value not in (0, 1, 2):
        raise ValueError("Metric jet order must be 0, 1, or 2.")
    points = _coordinates(coordinates, metric.chart.dimension)
    values = _evaluate_jet(
        _MetricJetEvaluator(metric, order_value),
        points,
        metric.chart.dimension,
    )
    first_derivative = None if order_value == 0 else values[7]
    second_derivative = None if order_value < 2 else values[8]
    return MetricJet(
        matrix=values[0],
        inverse=values[1],
        determinant=values[2],
        determinant_sign=values[3],
        log_abs_determinant=values[4],
        volume_density=values[5],
        log_volume_density=values[6],
        first_derivative=first_derivative,
        second_derivative=second_derivative,
        order=order_value,
    )


__all__ = ["MetricJet", "metric_jet"]
