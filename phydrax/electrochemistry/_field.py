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
    LinearSolveResult,
    LinearSystem,
    solve,
)


def solve_current_distribution(
    conductance_matrix: ArrayLike, current_source: ArrayLike, /
) -> LinearSolveResult:
    matrix = jnp.asarray(conductance_matrix)
    source = jnp.asarray(current_source)
    space = ArraySpace((source.size,), dtype=matrix.dtype)
    return solve(
        LinearSystem(DenseLinearOperator(matrix, source=space, target=space)),
        source,
        policy=LinearSolvePolicy(DenseLU()),
    )


__all__ = ["solve_current_distribution"]
