#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fused point evaluation of scalar and spin-spherical expansions."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike
from s2fft.recursions.price_mcewen import compute_all_slices_jax
from s2fft.recursions.risbo_jax import compute_full as _wigner_small_d

import phydrax.ein as ein

from ...special._dtype import _exact_zero, promote_real
from ...special._spherical_harmonic import _normalize_directions, _seed, _step


_RISBO_SPIN_THRESHOLD = 8


def _invalid_like(result: Array, /) -> Array:
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        return lax.complex(
            jnp.full_like(jnp.real(result), jnp.nan),
            jnp.full_like(jnp.real(result), jnp.nan),
        )
    return jnp.full_like(result, jnp.nan)


def _scalar_synthesis_cartesian(
    coefficients: Array,
    directions: ArrayLike,
    /,
    *,
    bandlimit: int,
    real_output: bool,
) -> Array:
    """Stream scalar harmonics into coefficients without a basis table."""
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
                    sign = -1.0 if order % 2 else 1.0
                    negative_harmonic = sign * jnp.conj(harmonic)
                    negative = modal[degree, center - order]
                    result = result + negative_harmonic[:, None] * negative[None, :]

    valid_flat = valid.reshape((point_count, 1))
    result = jnp.where(valid_flat, result, _invalid_like(result))
    return result.reshape(point_shape + payload_shape)


def _accumulate_degree(
    result: Array,
    modal: Array,
    wigner_slice: Array,
    azimuth_phase: Array,
    degree: int,
    spin: int,
    /,
) -> Array:
    normalization = math.sqrt((2 * degree + 1) / (4 * math.pi))
    spin_phase = -1.0 if spin % 2 else 1.0
    harmonic = spin_phase * normalization * wigner_slice * azimuth_phase
    return result + ein.contract("pm,mc->pc", harmonic, modal[degree], backend="jax")


def _price_mcewen_synthesis(
    modal: Array,
    theta: Array,
    phi: Array,
    /,
    *,
    bandlimit: int,
    spin: int,
) -> Array:
    width = 2 * bandlimit - 1
    payload_count = modal.shape[-1]
    orders = jnp.arange(-(bandlimit - 1), bandlimit, dtype=theta.dtype)
    azimuth_phase = jnp.exp(1j * phi[:, None] * orders[None, :])
    slices = compute_all_slices_jax(theta, bandlimit, -spin)
    result = jnp.zeros(
        (theta.size, payload_count),
        dtype=jnp.result_type(modal.dtype, azimuth_phase.dtype),
    )
    for degree in range(abs(spin), bandlimit):
        # Price--McEwen stores the m axis in descending order.
        wigner_slice = jnp.flip(slices[:, :, degree], axis=0).T
        result = _accumulate_degree(
            result,
            modal,
            wigner_slice.reshape((theta.size, width)),
            azimuth_phase,
            degree,
            spin,
        )
    convention_phase = -1.0 if abs(spin) % 2 == 0 else 1.0
    return convention_phase * result


def _risbo_synthesis(
    modal: Array,
    theta: Array,
    phi: Array,
    /,
    *,
    bandlimit: int,
    spin: int,
) -> Array:
    width = 2 * bandlimit - 1
    center = bandlimit - 1
    payload_count = modal.shape[-1]
    orders = jnp.arange(-center, center + 1, dtype=theta.dtype)
    azimuth_phase = jnp.exp(1j * phi[:, None] * orders[None, :])
    result = jnp.zeros(
        (theta.size, payload_count),
        dtype=jnp.result_type(modal.dtype, azimuth_phase.dtype),
    )
    planes = jnp.zeros((theta.size, width, width), dtype=theta.dtype)
    for degree in range(bandlimit):
        planes = jax.vmap(
            lambda plane, angle, degree=degree: _wigner_small_d(
                plane, angle, bandlimit, degree
            )
        )(planes, theta)
        if degree >= abs(spin):
            wigner_slice = planes[:, :, center - spin]
            result = _accumulate_degree(
                result,
                modal,
                wigner_slice,
                azimuth_phase,
                degree,
                spin,
            )
    return result


def _polar_synthesis(
    modal: Array,
    phi: Array,
    /,
    *,
    bandlimit: int,
    spin: int,
) -> tuple[Array, Array]:
    center = bandlimit - 1
    north_coefficient = jnp.zeros((modal.shape[-1],), dtype=modal.dtype)
    south_coefficient = jnp.zeros_like(north_coefficient)
    spin_phase = -1.0 if spin % 2 else 1.0
    for degree in range(abs(spin), bandlimit):
        normalization = math.sqrt((2 * degree + 1) / (4 * math.pi))
        north_coefficient = (
            north_coefficient + spin_phase * normalization * modal[degree, center - spin]
        )
        degree_phase = -1.0 if degree % 2 else 1.0
        south_coefficient = (
            south_coefficient
            + degree_phase * normalization * modal[degree, center + spin]
        )
    north = jnp.exp(-1j * spin * phi)[:, None] * north_coefficient[None, :]
    south = jnp.exp(1j * spin * phi)[:, None] * south_coefficient[None, :]
    return north, south


def _spin_synthesis_angles(
    coefficients: Array,
    theta: Array,
    phi: Array,
    frame_angle: Array,
    /,
    *,
    bandlimit: int,
    spin: int,
) -> Array:
    point_shape = theta.shape
    payload_shape = coefficients.shape[2:]
    point_count = math.prod(point_shape) if point_shape else 1
    payload_count = math.prod(payload_shape) if payload_shape else 1
    polar = theta.reshape((point_count,))
    azimuth = phi.reshape((point_count,))
    frame = frame_angle.reshape((point_count,))
    valid = jnp.isfinite(polar) & jnp.isfinite(azimuth) & jnp.isfinite(frame)

    safe_polar = jnp.where(valid, polar, jnp.asarray(math.pi / 2, dtype=polar.dtype))
    safe_azimuth = jnp.where(valid, azimuth, jnp.zeros_like(azimuth))
    safe_frame = jnp.where(valid, frame, jnp.zeros_like(frame))
    north_pole = valid & (polar == jnp.asarray(0.0, dtype=polar.dtype))
    south_pole = valid & (polar == jnp.asarray(math.pi, dtype=polar.dtype))
    recursion_polar = jnp.where(
        north_pole | south_pole,
        jnp.asarray(math.pi / 2, dtype=polar.dtype),
        safe_polar,
    )
    modal = coefficients.reshape((bandlimit, 2 * bandlimit - 1, payload_count))

    if abs(spin) < _RISBO_SPIN_THRESHOLD:
        result = _price_mcewen_synthesis(
            modal,
            recursion_polar,
            safe_azimuth,
            bandlimit=bandlimit,
            spin=spin,
        )
    else:
        result = _risbo_synthesis(
            modal,
            safe_polar,
            safe_azimuth,
            bandlimit=bandlimit,
            spin=spin,
        )

    north, south = _polar_synthesis(
        modal,
        safe_azimuth,
        bandlimit=bandlimit,
        spin=spin,
    )
    result = jnp.where(north_pole[:, None], north, result)
    result = jnp.where(south_pole[:, None], south, result)
    result = result * jnp.exp(-1j * spin * safe_frame)[:, None]
    result = jnp.where(valid[:, None], result, _invalid_like(result))
    return result.reshape(point_shape + payload_shape)


def _spherical_synthesis_angles(
    coefficients: Array,
    theta: ArrayLike,
    phi: ArrayLike,
    /,
    *,
    bandlimit: int,
    spin: int,
    real_output: bool,
    frame_angle: ArrayLike = 0.0,
) -> Array:
    polar, azimuth, frame = promote_real(
        "spherical point evaluation", theta, phi, frame_angle
    )
    polar, azimuth, frame = jnp.broadcast_arrays(polar, azimuth, frame)
    if spin == 0:
        sine = jnp.sin(polar)
        directions = jnp.stack(
            (
                sine * jnp.cos(azimuth),
                sine * jnp.sin(azimuth),
                jnp.cos(polar),
            ),
            axis=-1,
        )
        result = _scalar_synthesis_cartesian(
            coefficients,
            directions,
            bandlimit=bandlimit,
            real_output=real_output,
        )
        valid_frame = jnp.isfinite(frame)
        payload_ndim = coefficients.ndim - 2
        valid_frame = valid_frame.reshape(frame.shape + (1,) * payload_ndim)
        return jnp.where(valid_frame, result, _invalid_like(result))
    if real_output:
        raise ValueError("Nonzero-spin point evaluation requires complex output.")
    return _spin_synthesis_angles(
        coefficients,
        polar,
        azimuth,
        frame,
        bandlimit=bandlimit,
        spin=spin,
    )


def _frame_angle(
    unit: Array,
    valid_direction: Array,
    east: Array,
    north: Array,
    baseline_east: Array,
    baseline_north: Array,
    /,
) -> Array:
    finite = jnp.all(jnp.isfinite(east), axis=-1) & jnp.all(jnp.isfinite(north), axis=-1)
    safe_east = jnp.where(finite[..., None], east, baseline_east)
    safe_north = jnp.where(finite[..., None], north, baseline_north)
    east_norm = jnp.sqrt(jnp.sum(safe_east * safe_east, axis=-1))
    north_norm = jnp.sqrt(jnp.sum(safe_north * safe_north, axis=-1))
    east_radial = jnp.sum(safe_east * unit, axis=-1)
    north_radial = jnp.sum(safe_north * unit, axis=-1)
    orthogonality = jnp.sum(safe_east * safe_north, axis=-1)
    orientation = jnp.sum(jnp.cross(safe_east, safe_north) * unit, axis=-1)
    tolerance = 512.0 * jnp.finfo(unit.dtype).eps
    frame_valid = (
        finite
        & (jnp.abs(east_norm - 1.0) <= tolerance)
        & (jnp.abs(north_norm - 1.0) <= tolerance)
        & (jnp.abs(east_radial) <= tolerance)
        & (jnp.abs(north_radial) <= tolerance)
        & (jnp.abs(orthogonality) <= tolerance)
        & (orientation >= 1.0 - tolerance)
    )
    cosine = 0.5 * (
        jnp.sum(safe_east * baseline_east, axis=-1)
        + jnp.sum(safe_north * baseline_north, axis=-1)
    )
    sine = 0.5 * (
        jnp.sum(safe_east * baseline_north, axis=-1)
        - jnp.sum(safe_north * baseline_east, axis=-1)
    )
    cosine = jnp.where(frame_valid, cosine, jnp.ones_like(cosine))
    sine = jnp.where(frame_valid, sine, jnp.zeros_like(sine))
    angle = jnp.arctan2(sine, cosine)
    return eqx.error_if(
        angle,
        jnp.any(valid_direction & ~frame_valid),
        "A tangent frame must be finite, orthonormal, tangent, and satisfy "
        "east cross north = radial.",
    )


def _spherical_synthesis_cartesian(
    coefficients: Array,
    directions: ArrayLike,
    /,
    *,
    bandlimit: int,
    spin: int,
    real_output: bool,
    tangent_frame: tuple[ArrayLike, ArrayLike] | None = None,
) -> Array:
    if spin == 0:
        return _scalar_synthesis_cartesian(
            coefficients,
            directions,
            bandlimit=bandlimit,
            real_output=real_output,
        )
    if real_output:
        raise ValueError("Nonzero-spin point evaluation requires complex output.")

    if tangent_frame is None:
        (vectors,) = promote_real("spherical point evaluation", directions)
    else:
        if not isinstance(tangent_frame, tuple) or len(tangent_frame) != 2:
            raise TypeError("tangent_frame must be an (east, north) tuple.")
        vectors, east, north = promote_real(
            "spherical point evaluation",
            directions,
            tangent_frame[0],
            tangent_frame[1],
        )
        for name, vector in (("east", east), ("north", north)):
            if vector.ndim == 0 or vector.shape[-1] != 3:
                raise ValueError(
                    f"Tangent-frame {name} vectors must end in dimension 3; "
                    f"got {vector.shape}."
                )
        vectors, east, north = jnp.broadcast_arrays(vectors, east, north)

    unit, valid = _normalize_directions(vectors)
    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
    horizontal = jnp.sqrt(x * x + y * y)
    pole = _exact_zero(horizontal)
    safe_x = jnp.where(pole, jnp.ones_like(x), x)
    safe_y = jnp.where(pole, jnp.zeros_like(y), y)
    phi = jnp.arctan2(safe_y, safe_x)
    theta = jnp.arctan2(horizontal, z)
    baseline_east = jnp.stack((-jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)), axis=-1)
    baseline_north = jnp.cross(unit, baseline_east)

    if tangent_frame is None:
        theta = eqx.error_if(
            theta,
            jnp.any(valid & pole),
            "Nonzero-spin evaluation at a pole requires an explicit tangent_frame.",
        )
        angle = jnp.zeros_like(theta)
    else:
        angle = _frame_angle(
            unit,
            valid,
            east,
            north,
            baseline_east,
            baseline_north,
        )

    result = _spherical_synthesis_angles(
        coefficients,
        theta,
        phi,
        bandlimit=bandlimit,
        spin=spin,
        real_output=False,
        frame_angle=angle,
    )
    payload_ndim = coefficients.ndim - 2
    valid = valid.reshape(valid.shape + (1,) * payload_ndim)
    return jnp.where(valid, result, _invalid_like(result))


__all__: list[str] = []
