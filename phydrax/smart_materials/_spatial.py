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
    if (
        k.ndim != 2
        or k.shape[0] != k.shape[1]
        or d.ndim != 2
        or d.shape[0] != d.shape[1]
        or e.shape != (k.shape[0], d.shape[0])
    ):
        raise ValueError("Piezoelectric block operators have incompatible shapes.")
    return jnp.block([[k, -e], [-jnp.swapaxes(e, -1, -2), -d]])


__all__ = ["piezoelectric_block_matrix"]
