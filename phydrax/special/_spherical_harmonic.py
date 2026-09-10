#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# The normalized reduced-Legendre recurrence and Cartesian formulation are
# adapted from spexial at commit 6946494322d105edf84490529460553ac7c79b09,
# licensed under MIT. See NOTICE and LICENSES/SPEXIAL-MIT.txt.
#

"""Normalized spherical Legendre functions and spherical harmonics."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Literal

import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from ._dtype import (
    _exact_zero,
    _positive_log,
    _positive_subnormal,
    _signbit,
    promote_real,
)


AssociatedLegendreNormalization = Literal[
    "unnormalized",
    "schmidt",
    "fully_normalized",
]


def _degree_order(n: int, m: int, /) -> tuple[int, int]:
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be a static integer degree")
    if isinstance(m, bool) or not isinstance(m, Integral):
        raise TypeError("m must be a static integer order")
    degree, order = int(n), int(m)
    if degree < 0 or abs(order) > degree:
        raise ValueError(f"require n >= 0 and abs(m) <= n; got n={degree}, m={order}")
    return degree, order


def _seed(order: int, reference: Array, /) -> Array:
    log_seed = (
        0.5 * math.log((2 * order + 1) / (4 * math.pi))
        + 0.5 * math.lgamma(2 * order + 1)
        - order * math.log(2.0)
        - math.lgamma(order + 1)
    )
    return jnp.full_like(reference, (-1.0) ** order * math.exp(log_seed))


def _step(degree: int, order: int, /) -> tuple[float, float]:
    first = math.sqrt((4 * degree * degree - 1) / (degree * degree - order * order))
    second = (
        math.sqrt(((degree - 1) ** 2 - order * order) / (4 * (degree - 1) ** 2 - 1))
        if degree >= 2
        else 0.0
    )
    return first, second


def _reduced_spherical_legendre(
    degree: int,
    order: int,
    argument: Array,
    /,
) -> Array:
    previous = jnp.zeros_like(argument)
    current = _seed(order, argument)
    for current_degree in range(order + 1, degree + 1):
        first, second = _step(current_degree, order)
        previous, current = current, first * (argument * current - second * previous)
    return current


def _spherical_legendre_array(
    degree: int,
    order: int,
    theta: Array,
    /,
) -> Array:
    absolute_order = abs(order)
    value = _reduced_spherical_legendre(
        degree,
        absolute_order,
        jnp.cos(theta),
    ) * lax.integer_pow(jnp.sin(theta), absolute_order)
    return value if order >= 0 else (-1.0) ** absolute_order * value


def _azimuth_power(
    order: int,
    x: Array,
    y: Array,
    /,
) -> tuple[Array, Array]:
    real = jnp.ones_like(x)
    imaginary = jnp.zeros_like(x)
    for _ in range(order):
        real, imaginary = real * x - imaginary * y, real * y + imaginary * x
    return real, imaginary


def _normalize_directions(vector: ArrayLike, /) -> tuple[Array, Array]:
    (values,) = promote_real("sph_harm_y_cart", vector)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError(
            f"Cartesian spherical-harmonic vectors must end in dimension 3; got {values.shape}"
        )

    finite = jnp.all(jnp.isfinite(values), axis=-1)
    nonzero = jnp.any(~_exact_zero(values), axis=-1)
    valid = finite & nonzero
    fallback = jnp.asarray((0.0, 0.0, 1.0), dtype=values.dtype)
    safe = jnp.where(valid[..., None], values, fallback)

    scale = jnp.max(jnp.abs(safe), axis=-1, keepdims=True)
    use_log_scaling = _positive_subnormal(scale)
    direct = safe / jnp.where(use_log_scaling, jnp.ones_like(scale), scale)

    log_safe = jnp.where(use_log_scaling, safe, fallback)
    magnitude = jnp.abs(log_safe)
    zero_component = _exact_zero(magnitude)
    logarithm = _positive_log(
        jnp.where(zero_component, jnp.ones_like(magnitude), magnitude)
    )
    logarithm = jnp.where(
        zero_component,
        jnp.full_like(logarithm, -jnp.inf),
        logarithm,
    )
    maximum_logarithm = jnp.max(logarithm, axis=-1, keepdims=True)
    scaled_magnitude = jnp.where(
        zero_component,
        jnp.zeros_like(magnitude),
        jnp.exp(logarithm - maximum_logarithm),
    )
    signed_magnitude = jnp.where(
        _signbit(log_safe),
        -scaled_magnitude,
        scaled_magnitude,
    )
    scaled = jnp.where(use_log_scaling, signed_magnitude, direct)
    norm = jnp.sqrt(jnp.sum(scaled * scaled, axis=-1, keepdims=True))
    return scaled / norm, valid


def _spherical_harmonic_cartesian_unit(
    degree: int,
    order: int,
    unit_vector: Array,
    /,
) -> Array:
    x, y, z = unit_vector[..., 0], unit_vector[..., 1], unit_vector[..., 2]
    absolute_order = abs(order)
    reduced = _reduced_spherical_legendre(degree, absolute_order, z)
    real_azimuth, imaginary_azimuth = _azimuth_power(absolute_order, x, y)
    real, imaginary = reduced * real_azimuth, reduced * imaginary_azimuth
    if order < 0:
        sign = (-1.0) ** absolute_order
        real, imaginary = sign * real, -sign * imaginary
    return lax.complex(real, imaginary)


def sph_legendre_p(n: int, m: int, theta: ArrayLike, /) -> Array:
    """Evaluate the normalized spherical associated Legendre function."""
    degree, order = _degree_order(n, m)
    (angle,) = promote_real("sph_legendre_p", theta)
    return _spherical_legendre_array(degree, order, angle)


def sph_harm_y(
    n: int,
    m: int,
    theta: ArrayLike,
    phi: ArrayLike,
    /,
) -> Array:
    """Evaluate the orthonormal Condon--Shortley spherical harmonic."""
    degree, order = _degree_order(n, m)
    polar, azimuth = promote_real("sph_harm_y", theta, phi)
    polar, azimuth = jnp.broadcast_arrays(polar, azimuth)
    legendre = _spherical_legendre_array(degree, order, polar)
    phase = order * azimuth
    return lax.complex(legendre * jnp.cos(phase), legendre * jnp.sin(phase))


def sph_harm_y_cart(n: int, m: int, vector: ArrayLike, /) -> Array:
    """Evaluate a spherical harmonic from a finite nonzero Cartesian vector."""
    degree, order = _degree_order(n, m)
    unit_vector, valid = _normalize_directions(vector)
    value = _spherical_harmonic_cartesian_unit(degree, order, unit_vector)
    nan = lax.complex(
        jnp.full_like(unit_vector[..., 0], jnp.nan),
        jnp.full_like(unit_vector[..., 0], jnp.nan),
    )
    return jnp.where(valid, value, nan)


def _spherical_harmonic_synthesis(
    coefficients: Array,
    directions: ArrayLike,
    /,
    *,
    bandlimit: int,
    real_output: bool,
) -> Array:
    unit_vector, valid = _normalize_directions(directions)
    point_shape = unit_vector.shape[:-1]
    payload_shape = coefficients.shape[2:]
    point_count = math.prod(point_shape) if point_shape else 1
    payload_count = math.prod(payload_shape) if payload_shape else 1
    unit = unit_vector.reshape((point_count, 3))
    modal = coefficients.reshape((bandlimit, 2 * bandlimit - 1, payload_count))
    center = bandlimit - 1

    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    if real_output:
        result = jnp.zeros((point_count, payload_count), dtype=x.dtype)
    else:
        result = jnp.zeros(
            (point_count, payload_count),
            dtype=jnp.result_type(coefficients.dtype, 1j),
        )

    real_azimuth = jnp.ones_like(x)
    imaginary_azimuth = jnp.zeros_like(x)
    for order in range(bandlimit):
        if order > 0:
            real_azimuth, imaginary_azimuth = (
                real_azimuth * x - imaginary_azimuth * y,
                real_azimuth * y + imaginary_azimuth * x,
            )
        previous = jnp.zeros_like(z)
        current = _seed(order, z)
        for degree in range(order, bandlimit):
            if degree > order:
                first, second = _step(degree, order)
                previous, current = current, first * (z * current - second * previous)
            harmonic = lax.complex(
                current * real_azimuth,
                current * imaginary_azimuth,
            )
            positive = modal[degree, center + order]
            if real_output:
                multiplicity = 1.0 if order == 0 else 2.0
                result = result + multiplicity * jnp.real(
                    harmonic[:, None] * positive[None, :]
                )
            else:
                result = result + harmonic[:, None] * positive[None, :]
                if order > 0:
                    sign = (-1.0) ** order
                    negative_harmonic = sign * jnp.conj(harmonic)
                    negative = modal[degree, center - order]
                    result = result + negative_harmonic[:, None] * negative[None, :]

    valid_flat = valid.reshape((point_count, 1))
    if real_output:
        invalid = jnp.full_like(result, jnp.nan)
    else:
        invalid = lax.complex(
            jnp.full_like(jnp.real(result), jnp.nan),
            jnp.full_like(jnp.real(result), jnp.nan),
        )
    result = jnp.where(valid_flat, result, invalid)
    return result.reshape(point_shape + payload_shape)


def _real_harmonic_scale(
    degree: int,
    order: int,
    normalization: AssociatedLegendreNormalization,
    condon_shortley: bool,
    /,
) -> float:
    multiplicity = 2.0 if order > 0 else 1.0
    if normalization == "unnormalized":
        log_scale = 0.5 * (
            math.log(4.0 * math.pi / (2 * degree + 1))
            + math.lgamma(degree + order + 1)
            - math.lgamma(degree - order + 1)
        )
    elif normalization == "schmidt":
        log_scale = 0.5 * math.log(multiplicity * 4.0 * math.pi / (2 * degree + 1))
    elif normalization == "fully_normalized":
        log_scale = 0.5 * math.log(multiplicity * 4.0 * math.pi)
    else:
        raise ValueError(
            f"Unsupported associated-Legendre normalization: {normalization!r}"
        )
    phase = 1.0 if condon_shortley or order % 2 == 0 else -1.0
    return phase * math.exp(log_scale)


def _real_spherical_harmonic_table(
    maximum_degree: int,
    unit_vector: ArrayLike,
    /,
    *,
    normalization: AssociatedLegendreNormalization,
    condon_shortley: bool,
) -> tuple[Array, Array]:
    if isinstance(maximum_degree, bool) or not isinstance(maximum_degree, Integral):
        raise TypeError("maximum_degree must be a static integer")
    limit = int(maximum_degree)
    if limit < 0:
        raise ValueError("maximum_degree must be nonnegative")
    if normalization not in ("unnormalized", "schmidt", "fully_normalized"):
        raise ValueError(
            f"Unsupported associated-Legendre normalization: {normalization!r}"
        )

    (unit_vector,) = promote_real("spherical_harmonic_table", unit_vector)
    if unit_vector.shape != (3,):
        raise ValueError("Application spherical-harmonic synthesis requires one vector")
    x, y, z = unit_vector
    cosine = jnp.zeros((limit + 1, limit + 1), dtype=unit_vector.dtype)
    sine = jnp.zeros_like(cosine)

    real_azimuth = jnp.ones_like(x)
    imaginary_azimuth = jnp.zeros_like(x)
    for order in range(limit + 1):
        if order > 0:
            real_azimuth, imaginary_azimuth = (
                real_azimuth * x - imaginary_azimuth * y,
                real_azimuth * y + imaginary_azimuth * x,
            )
        previous = jnp.zeros_like(z)
        current = _seed(order, z)
        for degree in range(order, limit + 1):
            if degree > order:
                first, second = _step(degree, order)
                previous, current = current, first * (z * current - second * previous)
            scale = _real_harmonic_scale(
                degree,
                order,
                normalization,
                condon_shortley,
            )
            cosine = cosine.at[degree, order].set(scale * current * real_azimuth)
            sine = sine.at[degree, order].set(scale * current * imaginary_azimuth)

    return cosine, sine


__all__ = ["sph_harm_y", "sph_harm_y_cart", "sph_legendre_p"]
