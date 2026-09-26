#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve,
)


def solve_electric_potential(
    operator_matrix: ArrayLike, charge_source: ArrayLike, /
) -> LinearSolveResult:
    matrix = jnp.asarray(operator_matrix)
    source = jnp.asarray(charge_source)
    space = ArraySpace((source.size,), dtype=matrix.dtype)
    return solve(
        LinearSystem(DenseLinearOperator(matrix, source=space, target=space)),
        source,
        policy=LinearSolvePolicy(DenseLU()),
    )


def drift_diffusion_current(
    charge_density: ArrayLike,
    electric_field: ArrayLike,
    charge_gradient: ArrayLike,
    mobility: float,
    diffusivity: float,
    velocity: ArrayLike = 0.0,
    /,
) -> Array:
    return (
        float(mobility)
        * jnp.asarray(charge_density)[..., None]
        * jnp.asarray(electric_field)
        - float(diffusivity) * jnp.asarray(charge_gradient)
        + jnp.asarray(charge_density)[..., None] * jnp.asarray(velocity)
    )


__all__ = ["drift_diffusion_current", "solve_electric_potential"]
