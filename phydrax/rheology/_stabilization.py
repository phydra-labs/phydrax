#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class ConformationTransformResult:
    value: Array
    positive_definite: Array


def log_conformation_with_status(conformation: ArrayLike, /):
    values, vectors = jnp.linalg.eigh(jnp.asarray(conformation))
    positive = jnp.all(values > 0)
    safe = jnp.where(positive, values, 1.0)
    result = (vectors * jnp.log(safe)[..., None, :]) @ jnp.swapaxes(vectors, -1, -2)
    return ConformationTransformResult(
        jnp.where(positive, result, jnp.zeros_like(result)), positive
    )


__all__ = ["ConformationTransformResult", "log_conformation_with_status"]
