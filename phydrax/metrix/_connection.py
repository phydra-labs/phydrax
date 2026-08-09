#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule
from ._chart import CoordinateChart
from ._jet import metric_jet, MetricJet
from ._map import DifferentiableMap
from ._metric import AbstractSemiRiemannianMetric
from ._utils import _pointwise_array


def christoffel_from_metric_jet(jet: MetricJet, /) -> Array:
    """Construct ``Γ[..., k, i, j]`` from a first-order metric jet."""
    derivative = jet.first_derivative
    if derivative is None:
        raise ValueError(
            "Christoffel symbols require a metric jet of order at least one."
        )
    first = oe.contract("...kl,...jli->...kij", jet.inverse, derivative)
    second = oe.contract("...kl,...ilj->...kij", jet.inverse, derivative)
    third = oe.contract("...kl,...ijl->...kij", jet.inverse, derivative)
    return 0.5 * (first + second - third)


class AbstractAffineConnection(StrictModule):
    """Coordinate coefficients of a linear connection on a tangent bundle."""

    chart: AbstractAttribute[CoordinateChart]

    @abstractmethod
    def coefficients(self, coordinates: ArrayLike, /) -> Array:
        """Return coefficients ``Γ[..., k, i, j]``."""
        raise NotImplementedError

    def derivative(self, coordinates: ArrayLike, /) -> Array:
        """Return ``∂_l Γ^k_ij`` with the derivative axis last."""
        return _pointwise_array(
            jax.jacfwd(self._coefficients_point),
            coordinates,
            self.chart.dimension,
        )

    def _coefficients_point(self, coordinates: Array, /) -> Array:
        return self.coefficients(coordinates)

    def torsion(self, coordinates: ArrayLike, /) -> Array:
        coefficients = self.coefficients(coordinates)
        return coefficients - jnp.swapaxes(coefficients, -1, -2)


class CallableAffineConnection(AbstractAffineConnection):
    """An affine connection supplied by coordinate coefficient callables."""

    coefficient_function: Callable[[Array], Array]
    chart: CoordinateChart

    def __init__(
        self,
        coefficients: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
    ):
        if not callable(coefficients):
            raise TypeError("Connection coefficients must be callable.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Connection chart must be a CoordinateChart.")
        self.coefficient_function = coefficients
        self.chart = chart

    def coefficients(self, coordinates: ArrayLike, /) -> Array:
        values = _pointwise_array(
            self.coefficient_function,
            coordinates,
            self.chart.dimension,
        )
        expected = (self.chart.dimension,) * 3
        if values.shape[-3:] != expected:
            raise ValueError(
                "Connection coefficients must have trailing shape "
                f"{expected}; got {values.shape}."
            )
        return values


class LeviCivitaConnection(AbstractAffineConnection):
    """The unique torsion-free, metric-compatible connection of a metric."""

    metric: AbstractSemiRiemannianMetric
    chart: CoordinateChart

    def __init__(self, metric: AbstractSemiRiemannianMetric, /):
        if not isinstance(metric, AbstractSemiRiemannianMetric):
            raise TypeError("LeviCivitaConnection requires a nondegenerate metric.")
        self.metric = metric
        self.chart = metric.chart

    def coefficients(self, coordinates: ArrayLike, /) -> Array:
        return christoffel_from_metric_jet(metric_jet(self.metric, coordinates, order=1))


class _PullbackConnectionCoefficients(StrictModule):
    connection: AbstractAffineConnection
    map: DifferentiableMap

    def __init__(
        self,
        connection: AbstractAffineConnection,
        map: DifferentiableMap,
        /,
    ):
        self.connection = connection
        self.map = map

    def __call__(self, coordinates: Array, /) -> Array:
        target_coordinates = self.map.map_function(coordinates)
        jacobian = self.map.jacobian(coordinates)
        inverse_jacobian = jnp.linalg.solve(
            jacobian,
            jnp.eye(self.map.source.dimension, dtype=jacobian.dtype),
        )
        second_derivative = jax.jacfwd(jax.jacfwd(self.map.map_function))(coordinates)
        target = self.connection.coefficients(target_coordinates)
        transformed = (
            oe.contract("abc,bj,ck->ajk", target, jacobian, jacobian) + second_derivative
        )
        return oe.contract("ia,ajk->ijk", inverse_jacobian, transformed)


def pullback_affine_connection(
    connection: AbstractAffineConnection,
    map: DifferentiableMap,
    /,
) -> CallableAffineConnection:
    """Pull an affine connection through an equal-dimensional coordinate map."""
    if not isinstance(connection, AbstractAffineConnection):
        raise TypeError("connection must be an AbstractAffineConnection.")
    if not isinstance(map, DifferentiableMap):
        raise TypeError("map must be a DifferentiableMap.")
    if not map.target.compatible_with(connection.chart):
        raise ValueError("Map target chart must match the connection chart.")
    if map.source.dimension != map.target.dimension:
        raise ValueError("Connection pullback requires equal chart dimensions.")
    return CallableAffineConnection(
        _PullbackConnectionCoefficients(connection, map),
        chart=map.source,
    )


def connection_transformation_residual(
    source_connection: AbstractAffineConnection,
    target_connection: AbstractAffineConnection,
    map: DifferentiableMap,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the maximum residual of the inhomogeneous connection law."""
    transformed = pullback_affine_connection(target_connection, map)
    difference = source_connection.coefficients(coordinates) - transformed.coefficients(
        coordinates
    )
    return jnp.max(jnp.abs(difference), axis=(-3, -2, -1))


def torsion_tensor(
    connection: AbstractAffineConnection,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return ``Tᵏᵢⱼ = Γᵏᵢⱼ - Γᵏⱼᵢ``."""
    if not isinstance(connection, AbstractAffineConnection):
        raise TypeError("connection must be an AbstractAffineConnection.")
    coefficients = connection.coefficients(coordinates)
    return coefficients - jnp.swapaxes(coefficients, -1, -2)


def nonmetricity_tensor(
    connection: AbstractAffineConnection,
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the covariant derivative ``∇ᵢgⱼₖ`` of a metric."""
    if not isinstance(connection, AbstractAffineConnection):
        raise TypeError("connection must be an AbstractAffineConnection.")
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("metric must be an AbstractSemiRiemannianMetric.")
    if not connection.chart.compatible_with(metric.chart):
        raise ValueError("Connection and metric charts must match.")

    def evaluate(point: Array) -> Array:
        matrix = metric(point)
        derivative = jnp.moveaxis(jax.jacfwd(metric)(point), -1, 0)
        coefficients = connection.coefficients(point)
        first = oe.contract("lij,lk->ijk", coefficients, matrix)
        second = oe.contract("lik,jl->ijk", coefficients, matrix)
        return derivative - first - second

    return _pointwise_array(evaluate, coordinates, connection.chart.dimension)


def connection_geodesic_acceleration(
    connection: AbstractAffineConnection,
    coordinates: ArrayLike,
    velocity: ArrayLike,
    /,
) -> Array:
    velocity_array = jnp.asarray(velocity)
    dimension = connection.chart.dimension
    if velocity_array.shape[-1:] != (dimension,):
        raise ValueError(
            f"Geodesic velocity must have trailing dimension {dimension}; "
            f"got {velocity_array.shape}."
        )
    return -oe.contract(
        "...kij,...i,...j->...k",
        connection.coefficients(coordinates),
        velocity_array,
        velocity_array,
    )


def geodesic_acceleration(
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    velocity: ArrayLike,
    /,
) -> Array:
    return connection_geodesic_acceleration(
        LeviCivitaConnection(metric), coordinates, velocity
    )


def connection_geodesic_rhs(
    connection: AbstractAffineConnection,
    state: ArrayLike,
    /,
) -> Array:
    """First-order affine-geodesic system for ``[..., (q, velocity)]``."""
    state_array = jnp.asarray(state)
    dimension = connection.chart.dimension
    if state_array.shape[-1:] != (2 * dimension,):
        raise ValueError(
            f"Geodesic state must have trailing dimension {2 * dimension}; "
            f"got {state_array.shape}."
        )
    coordinates = state_array[..., :dimension]
    velocity = state_array[..., dimension:]
    acceleration = connection_geodesic_acceleration(connection, coordinates, velocity)
    return jnp.concatenate((velocity, acceleration), axis=-1)


def geodesic_rhs(
    metric: AbstractSemiRiemannianMetric,
    state: ArrayLike,
    /,
) -> Array:
    return connection_geodesic_rhs(LeviCivitaConnection(metric), state)


def connection_parallel_transport_rhs(
    connection: AbstractAffineConnection,
    coordinates: ArrayLike,
    velocity: ArrayLike,
    transported: ArrayLike,
    /,
) -> Array:
    velocity_array = jnp.asarray(velocity)
    transported_array = jnp.asarray(transported)
    dimension = connection.chart.dimension
    if velocity_array.shape[-1:] != (dimension,):
        raise ValueError(
            f"Path velocity must have trailing dimension {dimension}; "
            f"got {velocity_array.shape}."
        )
    if transported_array.shape[-1:] != (dimension,):
        raise ValueError(
            "Transported vector must have trailing dimension "
            f"{dimension}; got {transported_array.shape}."
        )
    return -oe.contract(
        "...kij,...i,...j->...k",
        connection.coefficients(coordinates),
        velocity_array,
        transported_array,
    )


def parallel_transport_rhs(
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    velocity: ArrayLike,
    transported: ArrayLike,
    /,
) -> Array:
    return connection_parallel_transport_rhs(
        LeviCivitaConnection(metric),
        coordinates,
        velocity,
        transported,
    )


__all__ = [
    "AbstractAffineConnection",
    "CallableAffineConnection",
    "LeviCivitaConnection",
    "christoffel_from_metric_jet",
    "connection_geodesic_acceleration",
    "connection_geodesic_rhs",
    "connection_parallel_transport_rhs",
    "connection_transformation_residual",
    "nonmetricity_tensor",
    "geodesic_acceleration",
    "geodesic_rhs",
    "parallel_transport_rhs",
    "pullback_affine_connection",
    "torsion_tensor",
]
