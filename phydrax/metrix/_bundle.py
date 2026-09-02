#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import inverse as matrix_inverse
from ._chart import CoordinateChart
from ._utils import _pointwise_array


class VectorBundleConnection(StrictModule):
    """Connection coefficients on a fixed-rank trivialized vector bundle."""

    coefficient_function: Callable[[Array], Array]
    chart: CoordinateChart
    fiber_dimension: int

    def __init__(
        self,
        coefficients: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
        fiber_dimension: int,
    ):
        if not callable(coefficients):
            raise TypeError("Bundle-connection coefficients must be callable.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Bundle connection requires a CoordinateChart.")
        dimension = int(fiber_dimension)
        if dimension < 1:
            raise ValueError("Bundle fiber_dimension must be positive.")
        self.coefficient_function = coefficients
        self.chart = chart
        self.fiber_dimension = dimension

    def _coefficients_point(self, coordinates: Array, /) -> Array:
        values = jnp.asarray(self.coefficient_function(coordinates))
        expected = (
            self.fiber_dimension,
            self.fiber_dimension,
            self.chart.dimension,
        )
        if values.shape != expected:
            raise ValueError(
                f"Bundle connection coefficients must have shape {expected}; "
                f"got {values.shape}."
            )
        return values

    def coefficients(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            self._coefficients_point,
            coordinates,
            self.chart.dimension,
        )

    def derivative(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            jax.jacfwd(self._coefficients_point),
            coordinates,
            self.chart.dimension,
        )


class _GaugeTransformedCoefficients(StrictModule):
    connection: VectorBundleConnection
    gauge: Callable[[Array], Array]

    def __init__(
        self,
        connection: VectorBundleConnection,
        gauge: Callable[[Array], Array],
        /,
    ):
        self.connection = connection
        self.gauge = gauge

    def _gauge(self, coordinates: Array, /) -> Array:
        value = jnp.asarray(self.gauge(coordinates))
        expected = (
            self.connection.fiber_dimension,
            self.connection.fiber_dimension,
        )
        if value.shape != expected:
            raise ValueError(f"Gauge transformation must have shape {expected}.")
        return value

    def __call__(self, coordinates: Array, /) -> Array:
        gauge = self._gauge(coordinates)
        inverse_result = matrix_inverse(gauge)
        inverse = eqx.error_if(
            inverse_result.value,
            ~inverse_result.successful,
            "Gauge transformation must be nonsingular.",
        )
        derivative = jax.jacfwd(self._gauge)(coordinates)
        coefficients = self.connection._coefficients_point(coordinates)
        conjugated = oe.contract("ac,cdi,db->abi", inverse, coefficients, gauge)
        inhomogeneous = oe.contract("ac,cbi->abi", inverse, derivative)
        return conjugated + inhomogeneous


def bundle_covariant_derivative(
    section: Callable[[Array], Array],
    connection: VectorBundleConnection,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return ``∇section[..., fiber, base]`` in one trivialization."""
    if not callable(section):
        raise TypeError("section must be callable.")
    if not isinstance(connection, VectorBundleConnection):
        raise TypeError("connection must be a VectorBundleConnection.")

    def section_point(point: Array) -> Array:
        value = jnp.asarray(section(point))
        expected = (connection.fiber_dimension,)
        if value.shape != expected:
            raise ValueError(f"Bundle section must have shape {expected}.")
        return value

    def evaluate(point: Array) -> Array:
        value = section_point(point)
        derivative = jax.jacfwd(section_point)(point)
        correction = oe.contract(
            "abi,b->ai", connection._coefficients_point(point), value
        )
        return derivative + correction

    return _pointwise_array(evaluate, coordinates, connection.chart.dimension)


def bundle_curvature(
    connection: VectorBundleConnection,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return curvature ``F[..., fiber, fiber, i, j]``."""
    if not isinstance(connection, VectorBundleConnection):
        raise TypeError("bundle_curvature requires a VectorBundleConnection.")

    def evaluate(point: Array) -> Array:
        coefficients = connection._coefficients_point(point)
        derivative = jax.jacfwd(connection._coefficients_point)(point)
        first_derivative = jnp.swapaxes(derivative, -1, -2) - derivative
        first_product = oe.contract("aci,cbj->abij", coefficients, coefficients)
        second_product = oe.contract("acj,cbi->abij", coefficients, coefficients)
        return first_derivative + first_product - second_product

    return _pointwise_array(evaluate, coordinates, connection.chart.dimension)


def gauge_transform_connection(
    connection: VectorBundleConnection,
    gauge: Callable[[Array], Array],
    /,
) -> VectorBundleConnection:
    """Transform ``A`` as ``G⁻¹ A G + G⁻¹ dG``."""
    if not isinstance(connection, VectorBundleConnection):
        raise TypeError("connection must be a VectorBundleConnection.")
    if not callable(gauge):
        raise TypeError("gauge must be callable.")
    return VectorBundleConnection(
        _GaugeTransformedCoefficients(connection, gauge),
        chart=connection.chart,
        fiber_dimension=connection.fiber_dimension,
    )


def gauge_curvature_residual(
    connection: VectorBundleConnection,
    gauge: Callable[[Array], Array],
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the residual of ``F' = G⁻¹ F G``."""
    transformed = gauge_transform_connection(connection, gauge)

    def evaluate(point: Array) -> Array:
        matrix = jnp.asarray(gauge(point))
        inverse_result = matrix_inverse(matrix)
        inverse = eqx.error_if(
            inverse_result.value,
            ~inverse_result.successful,
            "Gauge transformation must be nonsingular.",
        )
        curvature = bundle_curvature(connection, point)
        expected = oe.contract("ac,cdij,db->abij", inverse, curvature, matrix)
        actual = bundle_curvature(transformed, point)
        return jnp.max(jnp.abs(actual - expected))

    return _pointwise_array(evaluate, coordinates, connection.chart.dimension)


__all__ = [
    "VectorBundleConnection",
    "bundle_covariant_derivative",
    "bundle_curvature",
    "gauge_curvature_residual",
    "gauge_transform_connection",
]
