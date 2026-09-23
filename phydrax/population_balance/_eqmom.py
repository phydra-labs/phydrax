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
    realizable: Array


def gaussian_eqmom_one_node(moments: ArrayLike, /):
    m = jnp.asarray(moments)
    if m.shape != (3,):
        raise ValueError("One-node Gaussian EQMOM requires m0,m1,m2.")
    valid = (
        jnp.all(jnp.isfinite(m))
        & (m[0] > 0)
        & (m[1] >= 0)
        & (m[2] >= 0)
        & (m[0] * m[2] >= m[1] ** 2)
    )
    safe_m0 = jnp.where(m[0] > 0, m[0], 1.0)
    mean = m[1] / safe_m0
    variance = m[2] / safe_m0 - mean**2
    valid = (
        valid
        & jnp.isfinite(mean)
        & (mean >= 0)
        & jnp.isfinite(variance)
        & (variance >= 0)
    )
    return GaussianEQMOM(m[0], mean, jnp.maximum(variance, 0), valid)


__all__ = ["GaussianEQMOM", "gaussian_eqmom_one_node"]
