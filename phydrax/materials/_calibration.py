#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import equinox as eqx
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
    successful: Array


def calibrate_linear_material(
    design: ArrayLike, observations: ArrayLike, weights: ArrayLike, /
) -> LinearMaterialCalibrationResult:
    matrix = jnp.asarray(design)
    target = jnp.asarray(observations)
    w = jnp.asarray(weights)
    if matrix.ndim != 2 or target.shape != (matrix.shape[0],) or w.shape != target.shape:
        raise ValueError(
            "Calibration design, observations, and weights are incompatible."
        )
    matrix = eqx.error_if(
        matrix,
        jnp.any(
            ~jnp.isfinite(matrix)
            | ~jnp.isfinite(target[:, None])
            | ~jnp.isfinite(w[:, None])
            | (w[:, None] < 0)
        )
        | (jnp.sum(w) <= 0),
        "Calibration data must be finite with nonnegative nonzero weights.",
    )
    normal = jnp.conj(matrix).T @ (w[:, None] * matrix)
    right = jnp.conj(matrix).T @ (w * target)
    space = ArraySpace((matrix.shape[1],), dtype=normal.dtype)
    solved = solve(
        LinearSystem(DenseLinearOperator(normal, source=space, target=space)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    )
    value = solved.value
    residual = matrix @ value - target
    residual_norm = jnp.linalg.norm(jnp.sqrt(w) * residual)
    condition = jnp.linalg.cond(normal)
    successful = (
        solved.successful
        & jnp.all(jnp.isfinite(value))
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(condition)
    )
    return LinearMaterialCalibrationResult(
        value,
        residual_norm,
        condition,
        successful,
    )


__all__ = ["LinearMaterialCalibrationResult", "calibrate_linear_material"]
