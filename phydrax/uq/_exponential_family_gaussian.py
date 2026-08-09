#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._exponential_family import (
    ExponentialFamilyConversionResult,
    ExponentialFamilyLaw,
    MultivariateNormalFamily,
)
from .._exponential_family._symmetric_coordinates import svec
from ._gaussian_factor import gaussian_factor_from_covariance, GaussianFactor


def multivariate_normal_from_gaussian_factor(
    family: MultivariateNormalFamily,
    location: ArrayLike,
    factor: GaussianFactor,
    /,
) -> ExponentialFamilyConversionResult:
    """Convert a location and unregularized Gaussian factor to family coordinates."""
    if not isinstance(family, MultivariateNormalFamily):
        raise TypeError("family must be a MultivariateNormalFamily.")
    if not isinstance(factor, GaussianFactor):
        raise TypeError("factor must be a GaussianFactor.")
    if factor.event_size != family.event_size:
        raise ValueError("factor event_size does not match the family.")
    if jnp.issubdtype(factor.factor.dtype, jnp.complexfloating):
        raise TypeError("MultivariateNormalFamily requires a real GaussianFactor.")
    location_array = jnp.asarray(location)
    if jnp.issubdtype(location_array.dtype, jnp.complexfloating):
        raise TypeError("location must be real-valued.")
    if location_array.ndim == 0 or int(location_array.shape[-1]) != family.event_size:
        raise ValueError(
            f"location must end in event_size={family.event_size}; got {location_array.shape}."
        )
    checked_factor = eqx.error_if(
        factor.factor,
        jnp.any(~factor.valid),
        "Cannot convert an invalid GaussianFactor.",
    )
    checked_factor = eqx.error_if(
        checked_factor,
        factor.regularization != 0.0,
        "GaussianFactor regularization must be zero for an exact family conversion.",
    )
    batch_shape = jnp.broadcast_shapes(
        location_array.shape[:-1], checked_factor.shape[:-2]
    )
    dtype = jnp.result_type(location_array, checked_factor, 0.0)
    location_array = jnp.broadcast_to(
        location_array.astype(dtype), batch_shape + (family.event_size,)
    )
    checked_factor = jnp.broadcast_to(
        checked_factor.astype(dtype),
        batch_shape + (family.event_size, checked_factor.shape[-1]),
    )
    covariance = checked_factor @ jnp.swapaxes(checked_factor, -1, -2)
    second = covariance + jnp.einsum("...i,...j->...ij", location_array, location_array)
    mean = family.mean(jnp.concatenate((location_array, svec(second)), axis=-1))
    return family.natural_from_mean(mean)


def gaussian_factor_from_multivariate_normal(
    law: ExponentialFamilyLaw,
    /,
    *,
    rank_tolerance: ArrayLike = 0.0,
) -> tuple[Array, GaussianFactor]:
    """Return the location and exact covariance factor of a Normal family law."""
    if not isinstance(law, ExponentialFamilyLaw) or not isinstance(
        law.family, MultivariateNormalFamily
    ):
        raise TypeError(
            "law must be an ExponentialFamilyLaw backed by MultivariateNormalFamily."
        )
    location, covariance = law.family.location_covariance_from_natural(law.natural)
    dtype = covariance.dtype
    scale = jnp.maximum(jnp.max(jnp.abs(covariance)), 1.0)
    tolerance = 64.0 * jnp.finfo(dtype).eps * scale
    factor = gaussian_factor_from_covariance(
        covariance,
        rank_tolerance=rank_tolerance,
        hermitian_tolerance=tolerance,
        factor_id="multivariate-normal-covariance-factor",
    )
    return location, factor


__all__ = [
    "gaussian_factor_from_multivariate_normal",
    "multivariate_normal_from_gaussian_factor",
]
