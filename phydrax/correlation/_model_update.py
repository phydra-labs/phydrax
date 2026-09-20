#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
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
    normal = j.T @ j + float(regularization) * jnp.eye(j.shape[1])
    right = -j.T @ r
    space = ArraySpace((normal.shape[0],), dtype=normal.dtype)
    return solve(
        LinearSystem(DenseLinearOperator(normal, source=space, target=space)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    ).value


__all__ = ["gauss_newton_update"]
