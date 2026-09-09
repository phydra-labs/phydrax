#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Numerically stable standard-Normal distribution functions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import Array
from jax.typing import ArrayLike

from ._dtype import promote_real


_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)


@jax.custom_jvp
def _normal_pdf_kernel(x: Array, /) -> Array:
    return jnp.asarray(_INV_SQRT_TWO_PI, dtype=x.dtype) * jnp.exp(
        jnp.asarray(-0.5, dtype=x.dtype) * jnp.square(x)
    )


@_normal_pdf_kernel.defjvp
def _normal_pdf_kernel_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    (x,) = primals
    (tangent,) = tangents
    value = _normal_pdf_kernel(x)
    derivative = jnp.where(jnp.isinf(x), jnp.zeros_like(x), -x * value)
    return value, tangent * derivative


def normal_pdf(x: ArrayLike, /) -> Array:
    """Return the standard-Normal probability density.

    Integer inputs use JAX's configured default floating dtype, while float16 and
    bfloat16 inputs are widened to float32. Infinite arguments have exact zero
    density and a zero derivative.
    """
    (value,) = promote_real("normal_pdf", x)
    return _normal_pdf_kernel(value)


def normal_logpdf(x: ArrayLike, /) -> Array:
    """Return the standard-Normal log density without forming the density."""
    (value,) = promote_real("normal_logpdf", x)
    return jnp.asarray(-0.5, dtype=value.dtype) * jnp.square(value) - jnp.asarray(
        _LOG_SQRT_TWO_PI, dtype=value.dtype
    )


def normal_cdf(x: ArrayLike, /) -> Array:
    """Return the standard-Normal cumulative distribution function."""
    (value,) = promote_real("normal_cdf", x)
    return jsp.ndtr(value)


def normal_logcdf(x: ArrayLike, /) -> Array:
    """Return the log standard-Normal CDF, including far-left tails."""
    (value,) = promote_real("normal_logcdf", x)
    return jsp.log_ndtr(value)


def normal_survival(x: ArrayLike, /) -> Array:
    """Return ``P[Z > x]`` without subtracting the CDF from one."""
    (value,) = promote_real("normal_survival", x)
    return jsp.ndtr(-value)


def normal_logsurvival(x: ArrayLike, /) -> Array:
    """Return the log standard-Normal survival probability in the right tail."""
    (value,) = promote_real("normal_logsurvival", x)
    return jsp.log_ndtr(-value)


def normal_quantile(p: ArrayLike, /) -> Array:
    """Return the standard-Normal quantile for probabilities in ``[0, 1]``.

    The endpoints map exactly to negative and positive infinity. Values outside
    the closed probability interval, NaNs included, return NaN so the function
    remains JIT-, VMAP-, and differentiation-compatible.
    """
    (probability,) = promote_real("normal_quantile", p)
    quantile = jsp.ndtri(probability)
    valid = (probability >= 0.0) & (probability <= 1.0)
    return jnp.where(valid, quantile, jnp.asarray(jnp.nan, dtype=probability.dtype))


__all__ = [
    "normal_cdf",
    "normal_logcdf",
    "normal_logpdf",
    "normal_logsurvival",
    "normal_pdf",
    "normal_quantile",
    "normal_survival",
]
