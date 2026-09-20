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


def solve_helmholtz(
    stiffness: ArrayLike,
    mass: ArrayLike,
    source: ArrayLike,
    angular_frequency_rad_s: float,
    /,
):
    matrix = jnp.asarray(stiffness).astype(complex) - float(
        angular_frequency_rad_s
    ) ** 2 * jnp.asarray(mass)
    rhs = jnp.asarray(source).astype(complex)
    space = ArraySpace((rhs.size,), dtype=matrix.dtype)
    return solve(
        LinearSystem(DenseLinearOperator(matrix, source=space, target=space)),
        rhs,
        policy=LinearSolvePolicy(DenseLU()),
    )


__all__ = ["solve_helmholtz"]
