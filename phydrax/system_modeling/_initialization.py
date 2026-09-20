#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class InitializationResult:
    state: Array
    residual_norm: Array
    successful: Array


def initialization_result(state: ArrayLike, residual: ArrayLike, tolerance: float, /):
    value = jnp.asarray(state)
    norm = jnp.linalg.norm(jnp.asarray(residual))
    return InitializationResult(
        value, norm, jnp.isfinite(norm) & (norm <= float(tolerance))
    )


__all__ = ["InitializationResult", "initialization_result"]
