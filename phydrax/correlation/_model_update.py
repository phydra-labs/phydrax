#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)


def gauss_newton_update(
    jacobian: ArrayLike, residual: ArrayLike, regularization: float = 0.0, /
):
    j = jnp.asarray(jacobian)
    r = jnp.asarray(residual)
    if not isfinite(regularization) or regularization < 0:
        raise ValueError("Gauss-Newton regularization must be finite and nonnegative.")
    if j.ndim != 2 or r.shape != (j.shape[0],):
        raise ValueError("Gauss-Newton Jacobian and residual are incompatible.")
    j = eqx.error_if(
        j,
        jnp.any(~jnp.isfinite(j) | ~jnp.isfinite(r[:, None])),
        "Gauss-Newton Jacobian and residual must be finite.",
    )
    normal = j.T @ j + float(regularization) * jnp.eye(j.shape[1])
    right = -j.T @ r
    space = ArraySpace((normal.shape[0],), dtype=normal.dtype)
    return solve(
        LinearSystem(DenseLinearOperator(normal, source=space, target=space)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    )


__all__ = ["gauss_newton_update"]
