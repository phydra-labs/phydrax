#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve,
)


def implicit_thermal_step(
    temperature_k: ArrayLike,
    capacity_j_k: ArrayLike,
    conductance_laplacian_w_k: ArrayLike,
    heat_input_w: ArrayLike,
    step_size_s: float,
    /,
) -> LinearSolveResult:
    old = jnp.asarray(temperature_k)
    capacity = jnp.asarray(capacity_j_k)
    matrix = jnp.diag(capacity / float(step_size_s)) + jnp.asarray(
        conductance_laplacian_w_k
    )
    right = capacity / float(step_size_s) * old + jnp.asarray(heat_input_w)
    space = ArraySpace((old.size,), dtype=matrix.dtype)
    return solve(
        LinearSystem(DenseLinearOperator(matrix, source=space, target=space)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    )


__all__ = ["implicit_thermal_step"]
