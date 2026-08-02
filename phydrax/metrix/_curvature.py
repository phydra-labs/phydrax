#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ._connection import LeviCivitaConnection
from ._metric import RiemannianMetric


def _connection_values(
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> tuple[Array, Array]:
    connection = LeviCivitaConnection(metric)
    return connection.coefficients(coordinates), connection.derivative(coordinates)


def riemann_tensor(
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return ``R[..., l, k, i, j] = Rˡ_kij``.

    The convention is chosen so that the standard round sphere has positive scalar
    curvature.
    """

    gamma, derivative = _connection_values(metric, coordinates)
    first_derivative = oe.contract("...ljki->...lkij", derivative)
    second_derivative = oe.contract("...likj->...lkij", derivative)
    first_product = oe.contract("...lim,...mjk->...lkij", gamma, gamma)
    second_product = oe.contract("...ljm,...mik->...lkij", gamma, gamma)
    return first_derivative - second_derivative + first_product - second_product


def ricci_tensor(
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the Ricci contraction without materializing the full Riemann tensor."""

    gamma, derivative = _connection_values(metric, coordinates)
    first_derivative = oe.contract("...ljkl->...kj", derivative)
    second_derivative = oe.contract("...llkj->...kj", derivative)
    first_product = oe.contract("...llm,...mjk->...kj", gamma, gamma)
    second_product = oe.contract("...ljm,...mlk->...kj", gamma, gamma)
    return first_derivative - second_derivative + first_product - second_product


def scalar_curvature(
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the metric trace of the Ricci tensor."""

    return oe.contract(
        "...ij,...ij->...",
        metric.inverse(coordinates),
        ricci_tensor(metric, coordinates),
    )


def einstein_tensor(
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return ``Ric - 1/2 scalar_curvature * metric``."""

    ricci = ricci_tensor(metric, coordinates)
    scalar = oe.contract("...ij,...ij->...", metric.inverse(coordinates), ricci)
    return ricci - 0.5 * scalar[..., None, None] * metric(coordinates)


def sectional_curvature(
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    first: ArrayLike,
    second: ArrayLike,
    /,
) -> Array:
    """Sectional curvature of the plane spanned by two tangent vectors."""

    first_ = jnp.asarray(first)
    second_ = jnp.asarray(second)
    dimension = metric.chart.dimension
    if first_.shape[-1:] != (dimension,) or second_.shape[-1:] != (dimension,):
        raise ValueError(f"Section vectors must have trailing dimension {dimension}.")
    matrix = metric(coordinates)
    riemann = riemann_tensor(metric, coordinates)
    lowered = oe.contract("...al,...lkij->...akij", matrix, riemann)
    numerator = oe.contract(
        "...akij,...a,...k,...i,...j->...",
        lowered,
        first_,
        second_,
        first_,
        second_,
    )
    first_norm = oe.contract("...i,...ij,...j->...", first_, matrix, first_)
    second_norm = oe.contract("...i,...ij,...j->...", second_, matrix, second_)
    cross = oe.contract("...i,...ij,...j->...", first_, matrix, second_)
    denominator = first_norm * second_norm - cross * cross
    return numerator / denominator
