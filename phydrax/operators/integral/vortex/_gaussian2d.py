#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._strict import StrictModule


_TWO_PI = 2.0 * jnp.pi
_SERIES_THRESHOLD = 2.0e-2


class GaussianVortexKernelEvaluation2D(StrictModule):
    """Unit-circulation Gaussian-blob fields for source-to-target displacements."""

    velocity: Array
    velocity_gradient: Array
    vorticity: Array


def _validated_inputs(
    displacement: ArrayLike,
    core_radius: ArrayLike,
    /,
) -> tuple[Array, Array]:
    radius = jnp.asarray(core_radius)
    value = jnp.asarray(
        displacement,
        dtype=jnp.result_type(displacement, radius, jnp.float32),
    )
    radius = jnp.asarray(radius, dtype=value.dtype)
    if value.ndim < 1 or value.shape[-1] != 2:
        raise ValueError(
            "Gaussian 2-D vortex displacement must have trailing shape (2,)."
        )
    leading = value.shape[:-1]
    if radius.shape not in ((), leading):
        raise ValueError(
            "Gaussian 2-D core_radius must be scalar or match displacement leading shape."
        )
    radius = jnp.broadcast_to(radius, leading)
    invalid = jnp.any(~jnp.isfinite(radius) | (radius <= 0.0))
    radius = eqx.error_if(
        radius,
        invalid,
        "Gaussian 2-D core radii must be finite and strictly positive.",
    )
    value = eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(value)),
        "Gaussian 2-D displacements must be finite.",
    )
    return value, radius


def _radial_functions(scaled_squared_distance: Array, /) -> tuple[Array, Array, Array]:
    """Return phi(z), phi'(z), and exp(-z), with analytic zero limits."""

    z = scaled_squared_distance
    z2 = z * z
    z3 = z2 * z
    z4 = z2 * z2
    phi_series = 1.0 - 0.5 * z + z2 / 6.0 - z3 / 24.0 + z4 / 120.0
    derivative_series = -0.5 + z / 3.0 - z2 / 8.0 + z3 / 30.0 - z4 / 144.0
    small = z < _SERIES_THRESHOLD
    safe_z = jnp.where(small, 1.0, z)
    exponential = jnp.exp(-z)
    phi_regular = -jnp.expm1(-z) / safe_z
    derivative_regular = ((safe_z + 1.0) * exponential - 1.0) / (safe_z * safe_z)
    phi = jnp.where(small, phi_series, phi_regular)
    derivative = jnp.where(small, derivative_series, derivative_regular)
    return phi, derivative, exponential


def gaussian_vortex_kernel_2d(
    displacement: ArrayLike,
    core_radius: ArrayLike,
    /,
) -> GaussianVortexKernelEvaluation2D:
    """Evaluate the Lamb--Oseen/Gaussian unit-circulation blob kernel.

    ``displacement`` is target minus source.  The core convention is
    ``omega(r) = Gamma exp(-|r|^2 / sigma^2) / (pi sigma^2)``.
    """

    value, radius = _validated_inputs(displacement, core_radius)
    radius_squared = radius * radius
    squared_distance = jnp.sum(value * value, axis=-1)
    scaled_squared = squared_distance / radius_squared
    phi, derivative, exponential = _radial_functions(scaled_squared)
    rotated = jnp.stack((-value[..., 1], value[..., 0]), axis=-1)
    inverse_scale = 1.0 / (_TWO_PI * radius_squared)
    velocity = inverse_scale[..., None] * phi[..., None] * rotated

    rotation = jnp.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=value.dtype)
    outer = contract("...i,...j->...ij", rotated, value)
    velocity_gradient = inverse_scale[..., None, None] * (
        phi[..., None, None] * rotation
        + (2.0 * derivative / radius_squared)[..., None, None] * outer
    )
    vorticity = exponential / (jnp.pi * radius_squared)
    return GaussianVortexKernelEvaluation2D(
        velocity,
        velocity_gradient,
        vorticity,
    )


def _strength(
    strength: ArrayLike,
    leading_shape: tuple[int, ...],
    dtype: jnp.dtype,
    /,
) -> Array:
    value = jnp.asarray(strength, dtype=dtype)
    if value.shape not in ((), leading_shape):
        raise ValueError(
            "Gaussian 2-D circulation must be scalar or match displacement leading shape."
        )
    value = jnp.broadcast_to(value, leading_shape)
    return eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(value)),
        "Gaussian 2-D circulations must be finite.",
    )


def gaussian_vortex_velocity_2d(
    displacement: ArrayLike,
    circulation: ArrayLike,
    core_radius: ArrayLike,
    /,
) -> Array:
    """Velocity induced by Gaussian blobs of scalar circulation."""

    value, radius = _validated_inputs(displacement, core_radius)
    gamma = _strength(circulation, value.shape[:-1], value.dtype)
    radius_squared = radius * radius
    scaled_squared = jnp.sum(value * value, axis=-1) / radius_squared
    phi, _, _ = _radial_functions(scaled_squared)
    rotated = jnp.stack((-value[..., 1], value[..., 0]), axis=-1)
    return (gamma / (_TWO_PI * radius_squared) * phi)[..., None] * rotated


def gaussian_vortex_velocity_gradient_2d(
    displacement: ArrayLike,
    circulation: ArrayLike,
    core_radius: ArrayLike,
    /,
) -> Array:
    """Target-coordinate gradient of Gaussian-blob velocity."""

    value, radius = _validated_inputs(displacement, core_radius)
    gamma = _strength(circulation, value.shape[:-1], value.dtype)
    radius_squared = radius * radius
    scaled_squared = jnp.sum(value * value, axis=-1) / radius_squared
    phi, derivative, _ = _radial_functions(scaled_squared)
    rotated = jnp.stack((-value[..., 1], value[..., 0]), axis=-1)
    rotation = jnp.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=value.dtype)
    outer = contract("...i,...j->...ij", rotated, value)
    scale = gamma / (_TWO_PI * radius_squared)
    return scale[..., None, None] * (
        phi[..., None, None] * rotation
        + (2.0 * derivative / radius_squared)[..., None, None] * outer
    )


def gaussian_vortex_vorticity_2d(
    displacement: ArrayLike,
    circulation: ArrayLike,
    core_radius: ArrayLike,
    /,
) -> Array:
    """Scalar Gaussian-blob vorticity at target displacements."""

    value, radius = _validated_inputs(displacement, core_radius)
    gamma = _strength(circulation, value.shape[:-1], value.dtype)
    radius_squared = radius * radius
    scaled_squared = jnp.sum(value * value, axis=-1) / radius_squared
    return gamma * jnp.exp(-scaled_squared) / (jnp.pi * radius_squared)


__all__ = [
    "GaussianVortexKernelEvaluation2D",
    "gaussian_vortex_kernel_2d",
    "gaussian_vortex_velocity_2d",
    "gaussian_vortex_velocity_gradient_2d",
    "gaussian_vortex_vorticity_2d",
]
