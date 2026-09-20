#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)


@dataclass(frozen=True, slots=True)
class LinearMaterialCalibrationResult:
    parameters: Array
    residual_norm: Array
    normal_condition: Array


def calibrate_linear_material(
    design: ArrayLike, observations: ArrayLike, weights: ArrayLike, /
) -> LinearMaterialCalibrationResult:
    matrix = jnp.asarray(design)
    target = jnp.asarray(observations)
    w = jnp.asarray(weights)
    normal = jnp.conj(matrix).T @ (w[:, None] * matrix)
    right = jnp.conj(matrix).T @ (w * target)
    space = ArraySpace((matrix.shape[1],), dtype=normal.dtype)
    value = solve(
        LinearSystem(DenseLinearOperator(normal, source=space, target=space)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    ).value
    residual = matrix @ value - target
    return LinearMaterialCalibrationResult(
        value, jnp.linalg.norm(jnp.sqrt(w) * residual), jnp.linalg.cond(normal)
    )


__all__ = ["LinearMaterialCalibrationResult", "calibrate_linear_material"]
