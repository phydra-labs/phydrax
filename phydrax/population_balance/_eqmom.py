#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class GaussianEQMOM:
    weight: Array
    mean: Array
    variance: Array


def gaussian_eqmom_one_node(moments: ArrayLike, /):
    m = jnp.asarray(moments)
    if m.shape != (3,):
        raise ValueError("One-node Gaussian EQMOM requires m0,m1,m2.")
    mean = m[1] / m[0]
    variance = jnp.maximum(m[2] / m[0] - mean**2, 0)
    return GaussianEQMOM(m[0], mean, variance)


__all__ = ["GaussianEQMOM", "gaussian_eqmom_one_node"]
