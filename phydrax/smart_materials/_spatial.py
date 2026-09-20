#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def piezoelectric_block_matrix(
    mechanical_stiffness: ArrayLike,
    piezoelectric_coupling: ArrayLike,
    dielectric_stiffness: ArrayLike,
    /,
):
    k = jnp.asarray(mechanical_stiffness)
    e = jnp.asarray(piezoelectric_coupling)
    d = jnp.asarray(dielectric_stiffness)
    return jnp.block([[k, -jnp.swapaxes(e, -1, -2)], [e, d]])


__all__ = ["piezoelectric_block_matrix"]
