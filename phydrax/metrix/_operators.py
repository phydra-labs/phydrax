#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from string import ascii_lowercase

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ._connection import LeviCivitaConnection
from ._metric import RiemannianMetric
from ._tensor import TensorType
from ._utils import _pointwise_array


def _scalar_value(field: Callable[[Array], Array], coordinates: Array, /) -> Array:
    value = jnp.asarray(field(coordinates))
    if value.shape != ():
        raise ValueError(f"Expected a pointwise scalar field; got shape {value.shape}.")
    return value


def _gradient_point(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: Array,
    /,
) -> Array:
    _scalar_value(field, coordinates)
    differential = jax.jacfwd(field)(coordinates)
    return oe.contract("ij,j->i", metric.inverse(coordinates), differential)


def gradient(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Riemannian gradient of a pointwise scalar field."""

    return _pointwise_array(
        lambda point: _gradient_point(field, metric, point),
        coordinates,
        metric.chart.dimension,
    )


def _covariant_hessian_point(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: Array,
    /,
) -> Array:
    _scalar_value(field, coordinates)
    differential = jax.jacfwd(field)(coordinates)
    second = jax.jacfwd(jax.jacfwd(field))(coordinates)
    coefficients = LeviCivitaConnection(metric).coefficients(coordinates)
    correction = oe.contract("kij,k->ij", coefficients, differential)
    return second - correction


def covariant_hessian(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Covariant Hessian ``∇_i∇_j field`` of a scalar field."""

    return _pointwise_array(
        lambda point: _covariant_hessian_point(field, metric, point),
        coordinates,
        metric.chart.dimension,
    )


def laplace_beltrami(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Metric trace of the covariant Hessian of a scalar field."""

    def _point(point: Array) -> Array:
        hessian = _covariant_hessian_point(field, metric, point)
        return oe.contract("ij,ij->", metric.inverse(point), hessian)

    return _pointwise_array(_point, coordinates, metric.chart.dimension)


def _divergence_point(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: Array,
    /,
) -> Array:
    values = jnp.asarray(field(coordinates))
    dimension = metric.chart.dimension
    if values.shape != (dimension,):
        raise ValueError(
            f"Divergence requires a pointwise vector shape {(dimension,)}; "
            f"got {values.shape}."
        )
    derivative = jax.jacfwd(field)(coordinates)
    coefficients = LeviCivitaConnection(metric).coefficients(coordinates)
    return jnp.trace(derivative) + oe.contract("iik,k->", coefficients, values)


def divergence(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Riemannian divergence of a contravariant vector field."""

    return _pointwise_array(
        lambda point: _divergence_point(field, metric, point),
        coordinates,
        metric.chart.dimension,
    )


def _covariant_derivative_point(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    tensor_type: TensorType,
    coordinates: Array,
    /,
) -> Array:
    values = jnp.asarray(field(coordinates))
    dimension = metric.chart.dimension
    expected = (dimension,) * tensor_type.rank
    if values.shape != expected:
        raise ValueError(
            f"Tensor field of rank {tensor_type.rank} must have pointwise shape "
            f"{expected}; got {values.shape}."
        )
    derivative = jax.jacfwd(field)(coordinates)
    if tensor_type.rank == 0:
        return derivative
    coefficients = LeviCivitaConnection(metric).coefficients(coordinates)
    letters = tuple(letter for letter in ascii_lowercase if letter not in ("x", "y"))[
        : tensor_type.rank
    ]
    output = "".join(letters) + "x"
    result = derivative
    for slot, variance in enumerate(tensor_type.variance):
        input_letters = list(letters)
        input_letters[slot] = "y"
        tensor_subscript = "".join(input_letters)
        if variance == "contravariant":
            connection_subscript = f"{letters[slot]}xy"
            sign = 1.0
        else:
            connection_subscript = f"yx{letters[slot]}"
            sign = -1.0
        correction = oe.contract(
            f"{connection_subscript},{tensor_subscript}->{output}",
            coefficients,
            values,
        )
        result = result + sign * correction
    return result


def covariant_derivative(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    tensor_type: TensorType,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Covariant derivative with its new covariant derivative axis appended last."""

    return _pointwise_array(
        lambda point: _covariant_derivative_point(
            field,
            metric,
            tensor_type,
            point,
        ),
        coordinates,
        metric.chart.dimension,
    )
