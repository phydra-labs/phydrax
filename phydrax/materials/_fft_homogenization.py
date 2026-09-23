#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def scalar_fft_effective_conductivity(
    conductivity: ArrayLike, imposed_gradient: ArrayLike, iterations: int = 64, /
) -> tuple[Array, Array]:
    k = jnp.asarray(conductivity)
    gradient = jnp.asarray(imposed_gradient)
    if (
        k.ndim == 0
        or gradient.shape != (k.ndim,)
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise ValueError(
            "Conductivity grid dimension and iteration controls are invalid."
        )
    k = eqx.error_if(
        k,
        jnp.any(~jnp.isfinite(k) | (k <= 0))
        | jnp.any(~jnp.isfinite(gradient))
        | (jnp.linalg.norm(gradient) <= 0),
        "Conductivity must be finite/positive and imposed gradient finite/nonzero.",
    )
    reference = 0.5 * (jnp.min(k) + jnp.max(k))
    field = jnp.broadcast_to(gradient, (*k.shape, gradient.size))
    frequencies = tuple(jnp.fft.fftfreq(n) * 2 * jnp.pi for n in k.shape)
    wave = jnp.stack(jnp.meshgrid(*frequencies, indexing="ij"), axis=-1)
    norm = jnp.sum(wave * wave, axis=-1)
    for _ in range(iterations):
        polarization = (k - reference)[..., None] * field
        transformed = jnp.fft.fftn(polarization, axes=tuple(range(k.ndim)))
        projection = (
            wave
            * jnp.sum(wave * transformed, axis=-1)[..., None]
            / jnp.where(norm[..., None] > 0, norm[..., None], 1)
        )
        correction = jnp.fft.ifftn(projection, axes=tuple(range(k.ndim))).real / reference
        field = jnp.broadcast_to(gradient, field.shape) - correction
    flux = k[..., None] * field
    effective_flux = jnp.mean(flux, axis=tuple(range(k.ndim)))
    effective = jnp.vdot(effective_flux, gradient) / jnp.vdot(gradient, gradient)
    residual = jnp.linalg.norm(jnp.mean(field, axis=tuple(range(k.ndim))) - gradient)
    return effective, residual


__all__ = ["scalar_fft_effective_conductivity"]
