#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ._connection import AbstractAffineConnection, LeviCivitaConnection
from ._metric import AbstractSemiRiemannianMetric


def _connection_values(
    connection: AbstractAffineConnection,
    coordinates: ArrayLike,
    /,
) -> tuple[Array, Array]:
    return connection.coefficients(coordinates), connection.derivative(coordinates)


def connection_riemann_tensor(
    connection: AbstractAffineConnection,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return ``R[..., l, k, i, j] = Rˡ_kij`` for an affine connection."""
    gamma, derivative = _connection_values(connection, coordinates)
    first_derivative = ein.contract("...ljki->...lkij", derivative)
    second_derivative = ein.contract("...likj->...lkij", derivative)
    first_product = ein.contract("...lim,...mjk->...lkij", gamma, gamma)
    second_product = ein.contract("...ljm,...mik->...lkij", gamma, gamma)
    return first_derivative - second_derivative + first_product - second_product


def riemann_tensor(
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the Riemann tensor of a metric's Levi-Civita connection."""
    return connection_riemann_tensor(LeviCivitaConnection(metric), coordinates)


def connection_ricci_tensor(
    connection: AbstractAffineConnection,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the Ricci contraction without materializing full curvature."""
    gamma, derivative = _connection_values(connection, coordinates)
    first_derivative = ein.contract("...ljkl->...kj", derivative)
    second_derivative = ein.contract("...llkj->...kj", derivative)
    first_product = ein.contract("...llm,...mjk->...kj", gamma, gamma)
    second_product = ein.contract("...ljm,...mlk->...kj", gamma, gamma)
    return first_derivative - second_derivative + first_product - second_product


def ricci_tensor(
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    return connection_ricci_tensor(LeviCivitaConnection(metric), coordinates)


def scalar_curvature(
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the metric trace of the Ricci tensor."""
    return ein.contract(
        "...ij,...ij->...",
        metric.inverse(coordinates),
        ricci_tensor(metric, coordinates),
    )


def einstein_tensor(
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return ``Ric - 1/2 scalar_curvature * metric``."""
    ricci = ricci_tensor(metric, coordinates)
    scalar = ein.contract("...ij,...ij->...", metric.inverse(coordinates), ricci)
    return ricci - 0.5 * scalar[..., None, None] * metric(coordinates)


def sectional_curvature(
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    first: ArrayLike,
    second: ArrayLike,
    /,
    *,
    degeneracy_tolerance: float = 1e-12,
) -> Array:
    """Return curvature of a nondegenerate tangent two-plane."""
    if degeneracy_tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be non-negative.")
    first_array = jnp.asarray(first)
    second_array = jnp.asarray(second)
    dimension = metric.chart.dimension
    if first_array.shape[-1:] != (dimension,) or second_array.shape[-1:] != (dimension,):
        raise ValueError(f"Section vectors must have trailing dimension {dimension}.")
    matrix = metric(coordinates)
    riemann = riemann_tensor(metric, coordinates)
    lowered = ein.contract("...al,...lkij->...akij", matrix, riemann)
    numerator = ein.contract(
        "...akij,...a,...k,...i,...j->...",
        lowered,
        first_array,
        second_array,
        first_array,
        second_array,
    )
    first_square = ein.contract("...i,...ij,...j->...", first_array, matrix, first_array)
    second_square = ein.contract(
        "...i,...ij,...j->...", second_array, matrix, second_array
    )
    cross = ein.contract("...i,...ij,...j->...", first_array, matrix, second_array)
    denominator = first_square * second_square - cross * cross
    scale = jnp.maximum(
        jnp.abs(first_square * second_square) + cross * cross,
        jnp.finfo(denominator.dtype).tiny,
    )
    denominator = eqx.error_if(
        denominator,
        jnp.any(jnp.abs(denominator) <= degeneracy_tolerance * scale),
        "Sectional curvature is undefined for a degenerate tangent two-plane.",
    )
    return numerator / denominator


__all__ = [
    "connection_ricci_tensor",
    "connection_riemann_tensor",
    "einstein_tensor",
    "ricci_tensor",
    "riemann_tensor",
    "scalar_curvature",
    "sectional_curvature",
]
