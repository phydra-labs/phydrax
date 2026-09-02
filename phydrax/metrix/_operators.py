#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from string import ascii_lowercase

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ._connection import AbstractAffineConnection, LeviCivitaConnection
from ._metric import AbstractSemiRiemannianMetric, RiemannianMetric
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
    return ein.contract("ij,j->i", metric.inverse(coordinates), differential)


def gradient(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Riemannian gradient of a pointwise scalar field."""
    if not isinstance(metric, RiemannianMetric):
        raise TypeError("gradient requires a RiemannianMetric.")
    return _pointwise_array(
        lambda point: _gradient_point(field, metric, point),
        coordinates,
        metric.chart.dimension,
    )


def _connection_hessian_point(
    field: Callable[[Array], Array],
    connection: AbstractAffineConnection,
    coordinates: Array,
    /,
) -> Array:
    _scalar_value(field, coordinates)
    differential = jax.jacfwd(field)(coordinates)
    second = jax.jacfwd(jax.jacfwd(field))(coordinates)
    correction = ein.contract(
        "kij,k->ij", connection.coefficients(coordinates), differential
    )
    return second - correction


def connection_covariant_hessian(
    field: Callable[[Array], Array],
    connection: AbstractAffineConnection,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Covariant Hessian induced by an explicit affine connection."""
    if not isinstance(connection, AbstractAffineConnection):
        raise TypeError("connection must be an AbstractAffineConnection.")
    return _pointwise_array(
        lambda point: _connection_hessian_point(field, connection, point),
        coordinates,
        connection.chart.dimension,
    )


def covariant_hessian(
    field: Callable[[Array], Array],
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Covariant Hessian induced by a metric's Levi-Civita connection."""
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("covariant_hessian requires a nondegenerate metric.")
    return connection_covariant_hessian(field, LeviCivitaConnection(metric), coordinates)


def laplace_beltrami(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Metric trace of the Riemannian covariant Hessian."""
    if not isinstance(metric, RiemannianMetric):
        raise TypeError("laplace_beltrami requires a RiemannianMetric.")

    def pointwise(point: Array) -> Array:
        hessian = _connection_hessian_point(field, LeviCivitaConnection(metric), point)
        return ein.contract("ij,ij->", metric.inverse(point), hessian)

    return _pointwise_array(pointwise, coordinates, metric.chart.dimension)


def _connection_divergence_point(
    field: Callable[[Array], Array],
    connection: AbstractAffineConnection,
    coordinates: Array,
    /,
) -> Array:
    values = jnp.asarray(field(coordinates))
    dimension = connection.chart.dimension
    if values.shape != (dimension,):
        raise ValueError(
            f"Divergence requires a pointwise vector shape {(dimension,)}; "
            f"got {values.shape}."
        )
    derivative = jax.jacfwd(field)(coordinates)
    coefficients = connection.coefficients(coordinates)
    return jnp.trace(derivative) + ein.contract("iik,k->", coefficients, values)


def connection_divergence(
    field: Callable[[Array], Array],
    connection: AbstractAffineConnection,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Trace the covariant derivative of a vector under an affine connection."""
    if not isinstance(connection, AbstractAffineConnection):
        raise TypeError("connection must be an AbstractAffineConnection.")
    return _pointwise_array(
        lambda point: _connection_divergence_point(field, connection, point),
        coordinates,
        connection.chart.dimension,
    )


def divergence(
    field: Callable[[Array], Array],
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Divergence under a metric's Levi-Civita connection."""
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("divergence requires a nondegenerate metric.")
    return connection_divergence(field, LeviCivitaConnection(metric), coordinates)


def _affine_covariant_derivative_point(
    field: Callable[[Array], Array],
    connection: AbstractAffineConnection,
    tensor_type: TensorType,
    coordinates: Array,
    /,
) -> Array:
    values = jnp.asarray(field(coordinates))
    dimension = connection.chart.dimension
    expected = (dimension,) * tensor_type.rank
    if values.shape != expected:
        raise ValueError(
            f"Tensor field of rank {tensor_type.rank} must have pointwise shape "
            f"{expected}; got {values.shape}."
        )
    derivative = jax.jacfwd(field)(coordinates)
    coefficients = connection.coefficients(coordinates)
    result = derivative
    if tensor_type.rank:
        letters = tuple(letter for letter in ascii_lowercase if letter not in ("x", "y"))[
            : tensor_type.rank
        ]
        output = "".join(letters) + "x"
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
            correction = ein.contract(
                f"{connection_subscript},{tensor_subscript}->{output}",
                coefficients,
                values,
            )
            result = result + sign * correction
    if tensor_type.density_weight != 0.0:
        connection_trace = ein.contract("aax->x", coefficients)
        result = (
            result - tensor_type.density_weight * values[..., None] * connection_trace
        )
    return result


def affine_covariant_derivative(
    field: Callable[[Array], Array],
    connection: AbstractAffineConnection,
    tensor_type: TensorType,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Covariantly differentiate a tensor using an explicit affine connection."""
    if not isinstance(connection, AbstractAffineConnection):
        raise TypeError("connection must be an AbstractAffineConnection.")
    return _pointwise_array(
        lambda point: _affine_covariant_derivative_point(
            field, connection, tensor_type, point
        ),
        coordinates,
        connection.chart.dimension,
    )


def covariant_derivative(
    field: Callable[[Array], Array],
    metric: AbstractSemiRiemannianMetric,
    tensor_type: TensorType,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Metric covariant derivative with derivative axis appended last."""
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("covariant_derivative requires a nondegenerate metric.")
    return affine_covariant_derivative(
        field,
        LeviCivitaConnection(metric),
        tensor_type,
        coordinates,
    )


__all__ = [
    "affine_covariant_derivative",
    "connection_covariant_hessian",
    "connection_divergence",
    "covariant_derivative",
    "covariant_hessian",
    "divergence",
    "gradient",
    "laplace_beltrami",
]
