#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ._connection import LeviCivitaConnection
from ._metric import RiemannianMetric
from ._operators import laplace_beltrami
from ._utils import _pointwise_array


def _vector_value(
    field: Callable[[Array], Array],
    coordinates: Array,
    dimension: int,
    name: str,
    /,
) -> Array:
    values = jnp.asarray(field(coordinates))
    if values.shape != (dimension,):
        raise ValueError(
            f"{name} must have pointwise shape {(dimension,)}; got {values.shape}."
        )
    return values


def _diffusion_value(
    diffusion: Callable[[Array], Array],
    coordinates: Array,
    dimension: int,
    /,
) -> Array:
    values = jnp.asarray(diffusion(coordinates))
    if values.ndim != 2 or values.shape[0] != dimension:
        raise ValueError(
            "diffusion must have pointwise shape "
            f"({dimension}, noise_dim); got {values.shape}."
        )
    return values


def _covariance_value(
    covariance: Callable[[Array], Array],
    coordinates: Array,
    dimension: int,
    /,
) -> Array:
    values = jnp.asarray(covariance(coordinates))
    expected = (dimension, dimension)
    if values.shape != expected:
        raise ValueError(
            f"covariance must have pointwise shape {expected}; got {values.shape}."
        )
    return values


def _resolve_covariance(
    diffusion: Callable[[Array], Array] | None,
    covariance: Callable[[Array], Array] | None,
    coordinates: Array,
    dimension: int,
    /,
) -> Array | None:
    if diffusion is not None and covariance is not None:
        raise ValueError("Provide either diffusion or covariance, not both.")
    if diffusion is not None:
        sigma = _diffusion_value(diffusion, coordinates, dimension)
        return sigma @ jnp.swapaxes(sigma, -1, -2)
    if covariance is not None:
        return _covariance_value(covariance, coordinates, dimension)
    return None


def _coordinate_stratonovich_to_ito_drift_point(
    drift: Callable[[Array], Array],
    diffusion: Callable[[Array], Array],
    coordinates: Array,
    /,
) -> Array:
    dimension = int(coordinates.shape[0])
    drift_value = _vector_value(drift, coordinates, dimension, "drift")
    sigma = _diffusion_value(diffusion, coordinates, dimension)
    derivative = jax.jacfwd(diffusion)(coordinates)
    expected = (dimension, int(sigma.shape[1]), dimension)
    if derivative.shape != expected:
        raise ValueError(
            f"diffusion derivative must have pointwise shape {expected}; "
            f"got {derivative.shape}."
        )
    correction = 0.5 * ein.contract("jk,ikj->i", sigma, derivative)
    return drift_value + correction


def coordinate_stratonovich_to_ito_drift(
    drift: Callable[[Array], Array],
    diffusion: Callable[[Array], Array],
    coordinates: ArrayLike,
    /,
) -> Array:
    """Convert coordinate Stratonovich drift coefficients to Itô coefficients.

    This is the chart-local correction ``b_I^i = b_S^i + 1/2 σ_a^j ∂_j σ_a^i``.
    Itô drift coefficients are not vector components under nonlinear coordinate
    changes; use :func:`coordinate_to_covariant_drift` before a covariant generator.
    """

    coordinates_ = jnp.asarray(coordinates)
    if coordinates_.ndim < 1:
        raise ValueError("coordinates must have a trailing coordinate axis.")
    dimension = int(coordinates_.shape[-1])
    return _pointwise_array(
        lambda point: _coordinate_stratonovich_to_ito_drift_point(
            drift,
            diffusion,
            point,
        ),
        coordinates_,
        dimension,
    )


def _coordinate_to_covariant_drift_point(
    drift: Callable[[Array], Array],
    covariance: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: Array,
    /,
) -> Array:
    dimension = metric.chart.dimension
    drift_value = _vector_value(drift, coordinates, dimension, "drift")
    covariance_value = _covariance_value(covariance, coordinates, dimension)
    coefficients = LeviCivitaConnection(metric).coefficients(coordinates)
    correction = 0.5 * ein.contract("kij,ij->k", coefficients, covariance_value)
    return drift_value + correction


def coordinate_to_covariant_drift(
    drift: Callable[[Array], Array],
    covariance: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Convert coordinate Itô drift to the vector drift in a covariant generator.

    If ``a`` is the contravariant diffusion covariance, the returned vector is
    ``b^k + 1/2 Γ^k_ij a^ij``. Consequently
    ``b_I^i ∂_i + 1/2 a^ij ∂_i∂_j`` equals
    ``b_cov^i ∇_i + 1/2 a^ij ∇_i∇_j`` on scalar observables.
    """

    return _pointwise_array(
        lambda point: _coordinate_to_covariant_drift_point(
            drift,
            covariance,
            metric,
            point,
        ),
        coordinates,
        metric.chart.dimension,
    )


def _covariant_hessian_point(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: Array,
    /,
) -> tuple[Array, Array]:
    differential = jax.jacfwd(field)(coordinates)
    second = jax.jacfwd(jax.jacfwd(field))(coordinates)
    dimension = metric.chart.dimension
    if differential.shape[-1:] != (dimension,) or second.shape[-2:] != (
        dimension,
        dimension,
    ):
        raise ValueError("observable derivatives do not end in the coordinate axes.")
    coefficients = LeviCivitaConnection(metric).coefficients(coordinates)
    correction = ein.contract("kij,...k->...ij", coefficients, differential)
    return differential, second - correction


def _covariant_kolmogorov_generator_point(
    observable: Callable[[Array], Array],
    drift: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: Array,
    diffusion: Callable[[Array], Array] | None,
    covariance: Callable[[Array], Array] | None,
    /,
) -> Array:
    dimension = metric.chart.dimension
    drift_value = _vector_value(drift, coordinates, dimension, "drift")
    differential, covariant_hessian = _covariant_hessian_point(
        observable,
        metric,
        coordinates,
    )
    result = ein.contract("i,...i->...", drift_value, differential)
    covariance_value = _resolve_covariance(
        diffusion,
        covariance,
        coordinates,
        dimension,
    )
    if covariance_value is None:
        return result
    return result + 0.5 * ein.contract(
        "ij,...ij->...",
        covariance_value,
        covariant_hessian,
    )


def covariant_kolmogorov_generator(
    observable: Callable[[Array], Array],
    drift: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
    *,
    diffusion: Callable[[Array], Array] | None = None,
    covariance: Callable[[Array], Array] | None = None,
) -> Array:
    """Apply an intrinsic backward Kolmogorov generator.

    ``drift`` is a contravariant vector field and ``covariance`` is a symmetric
    rank-two contravariant field. The operator is
    ``b^i ∇_i u + 1/2 a^ij ∇_i∇_j u`` and acts componentwise on non-scalar
    observables. A rectangular diffusion factor may be supplied instead of ``a``.
    """

    if diffusion is not None and covariance is not None:
        raise ValueError("Provide either diffusion or covariance, not both.")
    return _pointwise_array(
        lambda point: _covariant_kolmogorov_generator_point(
            observable,
            drift,
            metric,
            point,
            diffusion,
            covariance,
        ),
        coordinates,
        metric.chart.dimension,
    )


def _covariant_derivative_rank_two_point(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: Array,
    /,
) -> Array:
    dimension = metric.chart.dimension
    values = jnp.asarray(field(coordinates))
    expected = (dimension, dimension)
    if values.shape != expected:
        raise ValueError(
            f"tensor field must have pointwise shape {expected}; got {values.shape}."
        )
    derivative = jax.jacfwd(field)(coordinates)
    coefficients = LeviCivitaConnection(metric).coefficients(coordinates)
    first = ein.contract("ixy,yj->ijx", coefficients, values)
    second = ein.contract("jxy,iy->ijx", coefficients, values)
    return derivative + first + second


def _divergence_rank_two_point(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: Array,
    /,
) -> Array:
    derivative = _covariant_derivative_rank_two_point(field, metric, coordinates)
    return ein.contract("ijj->i", derivative)


def _divergence_vector_point(
    field: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: Array,
    /,
) -> Array:
    dimension = metric.chart.dimension
    values = _vector_value(field, coordinates, dimension, "vector field")
    derivative = jax.jacfwd(field)(coordinates)
    coefficients = LeviCivitaConnection(metric).coefficients(coordinates)
    return jnp.trace(derivative) + ein.contract("iik,k->", coefficients, values)


def _covariant_fokker_planck_operator_point(
    density: Callable[[Array], Array],
    drift: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: Array,
    diffusion: Callable[[Array], Array] | None,
    covariance: Callable[[Array], Array] | None,
    /,
) -> Array:
    dimension = metric.chart.dimension

    def drift_flux(point: Array) -> Array:
        density_value = jnp.asarray(density(point))
        if density_value.shape != ():
            raise ValueError(
                f"density must be pointwise scalar; got {density_value.shape}."
            )
        return density_value * _vector_value(drift, point, dimension, "drift")

    result = -_divergence_vector_point(drift_flux, metric, coordinates)
    if diffusion is None and covariance is None:
        return result

    def diffusion_flux(point: Array) -> Array:
        density_value = jnp.asarray(density(point))
        if density_value.shape != ():
            raise ValueError(
                f"density must be pointwise scalar; got {density_value.shape}."
            )
        covariance_at_point = _resolve_covariance(
            diffusion,
            covariance,
            point,
            dimension,
        )
        assert covariance_at_point is not None
        return density_value * covariance_at_point

    def first_divergence(point: Array) -> Array:
        return _divergence_rank_two_point(diffusion_flux, metric, point)

    return result + 0.5 * _divergence_vector_point(
        first_divergence,
        metric,
        coordinates,
    )


def covariant_fokker_planck_operator(
    density: Callable[[Array], Array],
    drift: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
    *,
    diffusion: Callable[[Array], Array] | None = None,
    covariance: Callable[[Array], Array] | None = None,
) -> Array:
    """Apply the Fokker--Planck operator relative to Riemannian volume.

    The result is ``-∇_i(b^i p) + 1/2 ∇_i∇_j(a^ij p)``. Thus ``density`` is the
    scalar density with respect to ``dvol_g`` rather than coordinate Lebesgue
    measure. ``drift`` and ``covariance`` use contravariant tensor components.
    """

    if diffusion is not None and covariance is not None:
        raise ValueError("Provide either diffusion or covariance, not both.")
    return _pointwise_array(
        lambda point: _covariant_fokker_planck_operator_point(
            density,
            drift,
            metric,
            point,
            diffusion,
            covariance,
        ),
        coordinates,
        metric.chart.dimension,
    )


def brownian_generator(
    observable: Callable[[Array], Array],
    metric: RiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Apply the Riemannian Brownian generator ``1/2 Δ_g``."""

    return 0.5 * laplace_beltrami(observable, metric, coordinates)


__all__ = [
    "brownian_generator",
    "coordinate_stratonovich_to_ito_drift",
    "coordinate_to_covariant_drift",
    "covariant_fokker_planck_operator",
    "covariant_kolmogorov_generator",
]
