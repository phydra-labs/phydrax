#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from typing import Literal, TypeAlias

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


HomogenizationBound: TypeAlias = Literal["voigt", "reuss", "hill"]


def homogenize_scalar(
    properties: ArrayLike, fractions: ArrayLike, bound: HomogenizationBound = "hill"
) -> Array:
    p = jnp.asarray(properties)
    f = jnp.asarray(fractions)
    if p.shape != f.shape:
        raise ValueError("Properties and fractions must align.")
    voigt = jnp.sum(f * p, axis=-1)
    reuss = 1 / jnp.sum(f / p, axis=-1)
    if bound == "voigt":
        return voigt
    if bound == "reuss":
        return reuss
    if bound == "hill":
        return 0.5 * (voigt + reuss)
    raise ValueError("Unknown bound.")


def _inverse(value: ArrayLike) -> Array:
    matrix = jnp.asarray(value)
    size = matrix.shape[-1]
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise ValueError("This homogenization inverse requires one square matrix.")
    space = ArraySpace((size,), dtype=matrix.dtype)
    operator = DenseLinearOperator(matrix, source=space, target=space)
    columns = tuple(
        solve(
            LinearSystem(operator),
            jnp.eye(size, dtype=matrix.dtype)[:, index],
            policy=LinearSolvePolicy(DenseLU()),
        ).value
        for index in range(size)
    )
    return jnp.stack(columns, axis=1)


def reuss_tensor(stiffness: ArrayLike, fractions: ArrayLike, /) -> Array:
    values = jnp.asarray(stiffness)
    if values.ndim != 3:
        raise ValueError("Reuss tensor requires shape (phase, component, component).")
    compliance = jnp.stack(
        tuple(_inverse(values[index]) for index in range(values.shape[0]))
    )
    average = jnp.sum(jnp.asarray(fractions)[..., None, None] * compliance, axis=-3)
    return _inverse(average)


def voigt_tensor(stiffness: ArrayLike, fractions: ArrayLike, /) -> Array:
    return jnp.sum(
        jnp.asarray(fractions)[..., None, None] * jnp.asarray(stiffness), axis=-3
    )


__all__ = ["HomogenizationBound", "homogenize_scalar", "reuss_tensor", "voigt_tensor"]
