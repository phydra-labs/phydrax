#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def mass_normalize_modes(modes: ArrayLike, mass: ArrayLike, /):
    phi = jnp.asarray(modes)
    m = jnp.asarray(mass)
    norms = jnp.sqrt(jnp.real(jnp.sum(jnp.conj(phi) * (m @ phi), axis=0)))
    return phi / norms


def modal_damping_matrix(
    modes: ArrayLike,
    mass: ArrayLike,
    damping_ratios: ArrayLike,
    angular_frequencies: ArrayLike,
    /,
):
    phi = mass_normalize_modes(modes, mass)
    return (
        jnp.asarray(mass)
        @ phi
        @ jnp.diag(2 * jnp.asarray(damping_ratios) * jnp.asarray(angular_frequencies))
        @ jnp.conj(phi).T
        @ jnp.asarray(mass)
    )


__all__ = ["mass_normalize_modes", "modal_damping_matrix"]
