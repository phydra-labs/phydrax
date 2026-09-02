#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    DenseSVD,
    FailurePolicy,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
)


def regularized_least_squares(
    design: Array,
    right_hand_side: Array,
    regularization: float,
    /,
) -> Array:
    """Solve one finite local least-squares problem through phydrax.linalg."""
    matrix = jnp.asarray(design)
    right = jnp.asarray(right_hand_side)
    if matrix.ndim != 2 or right.shape[0] != matrix.shape[0]:
        raise ValueError("Local least-squares design and right-hand side do not agree.")
    ridge = float(regularization)
    if ridge <= 0.0:
        raise ValueError("Local least-squares regularization must be positive.")
    dtype = jnp.result_type(matrix, right)
    matrix = matrix.astype(dtype)
    right = right.astype(dtype)
    result = solve(
        LeastSquaresProblem(DenseLinearOperator(matrix)),
        right,
        policy=LinearSolvePolicy(
            DenseSVD(damping=ridge**0.5),
            failure=FailurePolicy("status"),
        ),
    )
    return eqx.error_if(
        result.value,
        ~jnp.all(jnp.isfinite(result.value)),
        "Regularized local least-squares solve produced nonfinite values.",
    )


def solve_spd(matrix: Array, right_hand_side: Array, /) -> Array:
    """Solve one finite SPD system through phydrax.linalg."""
    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "asserted",
                "positive_definite": "asserted",
                "positive_semidefinite": "asserted",
            },
        ),
    )
    return solve(
        LinearSystem(operator),
        right_hand_side,
        policy=LinearSolvePolicy(
            DenseLU(),
            failure=FailurePolicy("error"),
        ),
    ).value


__all__ = ["regularized_least_squares", "solve_spd"]
