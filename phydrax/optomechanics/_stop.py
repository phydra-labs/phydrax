#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike


def optical_path_difference(
    displacement_m: ArrayLike,
    surface_normal: ArrayLike,
    refractive_index: ArrayLike = 1.0,
    /,
):
    displacement = jnp.asarray(displacement_m)
    normal = jnp.asarray(surface_normal)
    index = jnp.asarray(refractive_index)
    if displacement.shape != normal.shape or displacement.ndim == 0:
        raise ValueError(
            "Optical displacement and surface normal must have matching vector shape."
        )
    normal_norm = jnp.linalg.norm(normal, axis=-1)
    displacement = eqx.error_if(
        displacement,
        jnp.any(
            ~jnp.isfinite(displacement)
            | ~jnp.isfinite(normal)
            | ~jnp.isfinite(index)
            | (index <= 0)
            | ~jnp.isclose(normal_norm, 1.0, atol=1e-6, rtol=1e-6)
        ),
        "Optical displacement/index must be finite and normals unit length.",
    )
    return 2 * index * jnp.sum(displacement * normal, axis=-1)


def rms_wavefront_error(optical_path_difference_m: ArrayLike, weights: ArrayLike, /):
    opd = jnp.asarray(optical_path_difference_m)
    weights_ = jnp.asarray(weights)
    if opd.shape != weights_.shape or opd.size == 0:
        raise ValueError("Wavefront weights must match a nonempty OPD array.")
    weights_ = eqx.error_if(
        weights_,
        jnp.any(~jnp.isfinite(opd) | ~jnp.isfinite(weights_) | (weights_ <= 0)),
        "Wavefront OPD/weights must be finite and weights positive.",
    )
    w = weights_ / jnp.sum(weights_)
    mean = jnp.sum(w * opd)
    return jnp.sqrt(jnp.sum(w * jnp.abs(opd - mean) ** 2))


__all__ = ["optical_path_difference", "rms_wavefront_error"]
