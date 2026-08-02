#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._jet import metric_jet, MetricJet
from ._metric import RiemannianMetric
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


class LeviCivitaConnection(StrictModule):
    """The unique torsion-free, metric-compatible connection of a metric."""

    metric: RiemannianMetric

    def __init__(self, metric: RiemannianMetric, /):
        self.metric = metric

    def coefficients(self, coordinates: ArrayLike, /) -> Array:
        return christoffel_from_metric_jet(metric_jet(self.metric, coordinates, order=1))

    def derivative(self, coordinates: ArrayLike, /) -> Array:
        """Return ``∂_l Γ^k_ij`` with the derivative axis last."""

        return _pointwise_array(
            jax.jacfwd(self._coefficients_point),
            coordinates,
            self.metric.chart.dimension,
        )

    def _coefficients_point(self, coordinates: Array, /) -> Array:
        return christoffel_from_metric_jet(metric_jet(self.metric, coordinates, order=1))


def geodesic_acceleration(
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    velocity: ArrayLike,
    /,
) -> Array:
    velocity_ = jnp.asarray(velocity)
    dimension = metric.chart.dimension
    if velocity_.shape[-1:] != (dimension,):
        raise ValueError(
            f"Geodesic velocity must have trailing dimension {dimension}; "
            f"got {velocity_.shape}."
        )
    coefficients = LeviCivitaConnection(metric).coefficients(coordinates)
    return -oe.contract(
        "...kij,...i,...j->...k",
        coefficients,
        velocity_,
        velocity_,
    )


def geodesic_rhs(
    metric: RiemannianMetric,
    state: ArrayLike,
    /,
) -> Array:
    """First-order geodesic system for state ``[..., (q, velocity)]``."""

    state_ = jnp.asarray(state)
    dimension = metric.chart.dimension
    if state_.shape[-1:] != (2 * dimension,):
        raise ValueError(
            f"Geodesic state must have trailing dimension {2 * dimension}; "
            f"got {state_.shape}."
        )
    coordinates = state_[..., :dimension]
    velocity = state_[..., dimension:]
    acceleration = geodesic_acceleration(metric, coordinates, velocity)
    return jnp.concatenate((velocity, acceleration), axis=-1)


def parallel_transport_rhs(
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    velocity: ArrayLike,
    transported: ArrayLike,
    /,
) -> Array:
    velocity_ = jnp.asarray(velocity)
    transported_ = jnp.asarray(transported)
    dimension = metric.chart.dimension
    if velocity_.shape[-1:] != (dimension,):
        raise ValueError(
            f"Path velocity must have trailing dimension {dimension}; got {velocity_.shape}."
        )
    if transported_.shape[-1:] != (dimension,):
        raise ValueError(
            "Transported vector must have trailing dimension "
            f"{dimension}; got {transported_.shape}."
        )
    coefficients = LeviCivitaConnection(metric).coefficients(coordinates)
    return -oe.contract(
        "...kij,...i,...j->...k",
        coefficients,
        velocity_,
        transported_,
    )
