#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Regular and irregular orthonormal solid harmonics."""

from __future__ import annotations

import math
from typing import Literal

import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from ._dtype import _positive_log, promote_real
from ._spherical_harmonic import (
    _azimuth_power,
    _degree_order,
    _normalize_directions,
    _seed,
    _step,
)


SolidHarmonicKind = Literal["regular", "irregular"]


def _cartesian_vectors(name: str, vector: ArrayLike, /) -> Array:
    (values,) = promote_real(name, vector)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError(
            f"Cartesian solid-harmonic vectors must end in dimension 3; got {values.shape}"
        )
    return values


def _solid_harmonic_recurrence(
    degree: int,
    order: int,
    x: Array,
    y: Array,
    z: Array,
    radius_squared: Array,
    /,
) -> Array:
    absolute_order = abs(order)
    real_azimuth, imaginary_azimuth = _azimuth_power(absolute_order, x, y)
    seed = _seed(absolute_order, x)
    previous = lax.complex(jnp.zeros_like(x), jnp.zeros_like(x))
    current = lax.complex(seed * real_azimuth, seed * imaginary_azimuth)
    for current_degree in range(absolute_order + 1, degree + 1):
        first, second = _step(current_degree, absolute_order)
        previous, current = (
            current,
            first * (z * current - second * radius_squared * previous),
        )
    if order < 0:
        current = (-1.0) ** absolute_order * jnp.conj(current)
    return current


def _solid_harmonic_regular_array(
    degree: int,
    order: int,
    vector: Array,
    /,
) -> Array:
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    radius_squared = x * x + y * y + z * z
    return _solid_harmonic_recurrence(
        degree,
        order,
        x,
        y,
        z,
        radius_squared,
    )


def _scaled_direction_and_log_radius(vector: Array, /) -> tuple[Array, Array, Array]:
    unit, valid = _normalize_directions(vector)
    fallback = jnp.asarray((0.0, 0.0, 1.0), dtype=vector.dtype)
    safe = jnp.where(valid[..., None], vector, fallback)
    scale = jnp.max(jnp.abs(safe), axis=-1)
    scaled = safe / scale[..., None]
    scaled_norm = jnp.sqrt(jnp.sum(scaled * scaled, axis=-1))
    log_radius = _positive_log(scale) + jnp.log(scaled_norm)
    return unit, log_radius, valid


def _complex_nan(reference: Array, /) -> Array:
    nan = jnp.full_like(reference, jnp.nan)
    return lax.complex(nan, nan)


def solid_harmonic_regular(n: int, m: int, vector: ArrayLike, /) -> Array:
    """Evaluate ``r**n Y_n^m`` in orthonormal Condon--Shortley convention."""
    degree, order = _degree_order(n, m)
    values = _cartesian_vectors("solid_harmonic_regular", vector)
    return _solid_harmonic_regular_array(degree, order, values)


def solid_harmonic_irregular(n: int, m: int, vector: ArrayLike, /) -> Array:
    """Evaluate ``r**(-n-1) Y_n^m`` on finite nonzero Cartesian vectors."""
    degree, order = _degree_order(n, m)
    values = _cartesian_vectors("solid_harmonic_irregular", vector)
    unit, log_radius, valid = _scaled_direction_and_log_radius(values)
    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
    angular = _solid_harmonic_recurrence(
        degree,
        order,
        x,
        y,
        z,
        jnp.ones_like(x),
    )
    radial = jnp.exp(-(degree + 1) * log_radius)
    value = angular * radial
    return jnp.where(valid, value, _complex_nan(jnp.real(value)))


def _synthesis_dtype(coefficients: Array, displacements: Array, /) -> jnp.dtype:
    coefficient_real_dtype = jnp.real(jnp.zeros((), dtype=coefficients.dtype)).dtype
    dtype = jnp.result_type(coefficient_real_dtype, displacements.dtype)
    if dtype == jnp.float16 or dtype == jnp.bfloat16:
        return jnp.dtype(jnp.float32)
    if not jnp.issubdtype(dtype, jnp.inexact):
        return jnp.asarray(0.0).dtype
    return jnp.dtype(dtype)


def _solid_harmonic_synthesis(
    coefficients: Array,
    displacements: Array,
    /,
    *,
    bandlimit: int,
    kind: SolidHarmonicKind,
    real_output: bool,
) -> Array:
    """Fuse solid-harmonic generation and contraction without a mode table."""
    limit = int(bandlimit)
    if limit <= 0:
        raise ValueError("bandlimit must be positive.")
    if kind not in ("regular", "irregular"):
        raise ValueError("kind must be 'regular' or 'irregular'.")

    modal = jnp.asarray(coefficients)
    coefficient_shape = (limit, 2 * limit - 1)
    if modal.ndim < 2 or tuple(modal.shape[:2]) != coefficient_shape:
        raise ValueError(
            "Solid-harmonic coefficients must begin with shape "
            f"{coefficient_shape}; got {modal.shape}."
        )
    points = _cartesian_vectors("solid_harmonic_synthesis", displacements)
    real_dtype = _synthesis_dtype(modal, points)
    points = points.astype(real_dtype)
    complex_dtype = jnp.result_type(real_dtype, 1j)
    modal = modal.astype(complex_dtype)

    degrees = jnp.arange(limit, dtype=jnp.int32)[:, None]
    orders = jnp.arange(-(limit - 1), limit, dtype=jnp.int32)[None, :]
    valid_modes = jnp.abs(orders) <= degrees
    payload_ndim = modal.ndim - 2
    modal = jnp.where(
        valid_modes.reshape(coefficient_shape + (1,) * payload_ndim),
        modal,
        jnp.zeros((), dtype=modal.dtype),
    )

    point_shape = points.shape[:-1]
    payload_shape = modal.shape[2:]
    point_count = math.prod(point_shape) if point_shape else 1
    payload_count = math.prod(payload_shape) if payload_shape else 1
    geometry = points.reshape((point_count, 3))
    flattened_modal = modal.reshape(coefficient_shape + (payload_count,))
    center = limit - 1

    if kind == "irregular":
        geometry, log_radius, valid_points = _scaled_direction_and_log_radius(geometry)
        radius_squared = jnp.ones((point_count,), dtype=real_dtype)
        inverse_radius = jnp.exp(-log_radius)
    else:
        x_, y_, z_ = geometry[:, 0], geometry[:, 1], geometry[:, 2]
        radius_squared = x_ * x_ + y_ * y_ + z_ * z_
        log_radius = jnp.zeros((point_count,), dtype=real_dtype)
        inverse_radius = jnp.ones((point_count,), dtype=real_dtype)
        valid_points = jnp.ones((point_count,), dtype=bool)

    x, y, z = geometry[:, 0], geometry[:, 1], geometry[:, 2]
    if real_output:
        result = jnp.zeros((point_count, payload_count), dtype=real_dtype)
    else:
        result = jnp.zeros((point_count, payload_count), dtype=complex_dtype)

    real_azimuth = jnp.ones_like(x)
    imaginary_azimuth = jnp.zeros_like(x)
    for order in range(limit):
        if order > 0:
            real_azimuth, imaginary_azimuth = (
                real_azimuth * x - imaginary_azimuth * y,
                real_azimuth * y + imaginary_azimuth * x,
            )
        seed = _seed(order, x)
        previous = lax.complex(jnp.zeros_like(x), jnp.zeros_like(x))
        current = lax.complex(seed * real_azimuth, seed * imaginary_azimuth)
        radial = jnp.exp(-(order + 1) * log_radius)
        for degree in range(order, limit):
            if degree > order:
                first, second = _step(degree, order)
                previous, current = (
                    current,
                    first * (z * current - second * radius_squared * previous),
                )
                radial = radial * inverse_radius
            harmonic = current * radial if kind == "irregular" else current
            positive = flattened_modal[degree, center + order]
            if real_output:
                multiplicity = 1.0 if order == 0 else 2.0
                result = result + multiplicity * jnp.real(
                    harmonic[:, None] * positive[None, :]
                )
            else:
                result = result + harmonic[:, None] * positive[None, :]
                if order > 0:
                    negative_harmonic = (-1.0) ** order * jnp.conj(harmonic)
                    negative = flattened_modal[degree, center - order]
                    result = result + negative_harmonic[:, None] * negative[None, :]

    if kind == "irregular":
        valid = valid_points[:, None]
        if real_output:
            invalid = jnp.full_like(result, jnp.nan)
        else:
            invalid = _complex_nan(jnp.real(result))
        result = jnp.where(valid, result, invalid)
    return result.reshape(point_shape + payload_shape)


__all__ = [
    "SolidHarmonicKind",
    "solid_harmonic_irregular",
    "solid_harmonic_regular",
]
