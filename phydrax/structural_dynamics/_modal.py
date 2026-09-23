#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike


def mass_normalize_modes(modes: ArrayLike, mass: ArrayLike, /):
    phi = jnp.asarray(modes)
    m = jnp.asarray(mass)
    if phi.ndim != 2 or m.shape != (phi.shape[0], phi.shape[0]):
        raise ValueError("Modes and mass matrix have incompatible shapes.")
    quadratic = jnp.real(jnp.sum(jnp.conj(phi) * (m @ phi), axis=0))
    quadratic = eqx.error_if(
        quadratic,
        jnp.any(~jnp.isfinite(quadratic) | (quadratic <= 0)),
        "Modal mass norms must be finite and positive.",
    )
    return phi / jnp.sqrt(quadratic)


def modal_damping_matrix(
    modes: ArrayLike,
    mass: ArrayLike,
    damping_ratios: ArrayLike,
    angular_frequencies: ArrayLike,
    /,
):
    phi = jnp.asarray(modes)
    ratios = jnp.asarray(damping_ratios)
    frequencies = jnp.asarray(angular_frequencies)
    if ratios.shape != frequencies.shape or ratios.shape != (phi.shape[1],):
        raise ValueError("Modal damping and frequency vectors must align with modes.")
    ratios = eqx.error_if(
        ratios,
        jnp.any(
            ~jnp.isfinite(ratios)
            | ~jnp.isfinite(frequencies)
            | (ratios < 0)
            | (frequencies <= 0)
        ),
        "Modal damping must be finite/nonnegative and frequencies finite/positive.",
    )
    phi = mass_normalize_modes(modes, mass)
    return (
        jnp.asarray(mass)
        @ phi
        @ jnp.diag(2 * ratios * frequencies)
        @ jnp.conj(phi).T
        @ jnp.asarray(mass)
    )


__all__ = ["mass_normalize_modes", "modal_damping_matrix"]
