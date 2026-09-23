#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass
from typing import Literal, TypeAlias

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


HomogenizationBound: TypeAlias = Literal["voigt", "reuss", "hill"]


@dataclass(frozen=True, slots=True)
class TensorHomogenizationResult:
    value: Array
    successful: Array


def homogenize_scalar(
    properties: ArrayLike, fractions: ArrayLike, bound: HomogenizationBound = "hill"
) -> Array:
    p = jnp.asarray(properties)
    f = jnp.asarray(fractions)
    if p.shape != f.shape:
        raise ValueError("Properties and fractions must align.")
    p = eqx.error_if(
        p,
        jnp.any(~jnp.isfinite(p) | ~jnp.isfinite(f) | (p <= 0) | (f < 0))
        | jnp.any(~jnp.isclose(jnp.sum(f, axis=-1), 1.0)),
        "Homogenization properties must be finite/positive and fractions normalized/nonnegative.",
    )
    voigt = jnp.sum(f * p, axis=-1)
    reuss = 1 / jnp.sum(f / p, axis=-1)
    if bound == "voigt":
        return voigt
    if bound == "reuss":
        return reuss
    if bound == "hill":
        return 0.5 * (voigt + reuss)
    raise ValueError("Unknown bound.")


def _inverse(value: ArrayLike) -> TensorHomogenizationResult:
    matrix = jnp.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("This homogenization inverse requires one square matrix.")
    size = matrix.shape[0]
    matrix = eqx.error_if(
        matrix,
        jnp.any(~jnp.isfinite(matrix)),
        "Homogenization matrix must be finite.",
    )
    space = ArraySpace((size,), dtype=matrix.dtype)
    operator = DenseLinearOperator(matrix, source=space, target=space)
    results = tuple(
        solve(
            LinearSystem(operator),
            jnp.eye(size, dtype=matrix.dtype)[:, index],
            policy=LinearSolvePolicy(DenseLU()),
        )
        for index in range(size)
    )
    inverse = jnp.stack(tuple(result.value for result in results), axis=1)
    successful = jnp.all(
        jnp.stack(tuple(result.successful for result in results))
    ) & jnp.all(jnp.isfinite(inverse))
    return TensorHomogenizationResult(inverse, successful)


def reuss_tensor(
    stiffness: ArrayLike, fractions: ArrayLike, /
) -> TensorHomogenizationResult:
    values = jnp.asarray(stiffness)
    fraction = jnp.asarray(fractions)
    if (
        values.ndim != 3
        or values.shape[1] != values.shape[2]
        or fraction.shape != (values.shape[0],)
    ):
        raise ValueError("Reuss tensor requires phase-aligned square stiffness tensors.")
    values = eqx.error_if(
        values,
        jnp.any(
            ~jnp.isfinite(values)
            | ~jnp.isfinite(fraction[:, None, None])
            | (fraction[:, None, None] < 0)
        )
        | ~jnp.isclose(jnp.sum(fraction), 1.0)
        | jnp.any(jnp.linalg.eigvalsh(values) <= 0),
        "Reuss stiffness must be positive definite and fractions normalized/nonnegative.",
    )
    inverses = tuple(_inverse(values[index]) for index in range(values.shape[0]))
    compliance = jnp.stack(tuple(result.value for result in inverses))
    phase_successful = jnp.all(jnp.stack(tuple(result.successful for result in inverses)))
    average = jnp.sum(fraction[..., None, None] * compliance, axis=-3)
    effective = _inverse(average)
    return TensorHomogenizationResult(
        effective.value,
        phase_successful & effective.successful,
    )


def voigt_tensor(
    stiffness: ArrayLike, fractions: ArrayLike, /
) -> TensorHomogenizationResult:
    values = jnp.asarray(stiffness)
    fraction = jnp.asarray(fractions)
    if (
        values.ndim != 3
        or values.shape[1] != values.shape[2]
        or fraction.shape != (values.shape[0],)
    ):
        raise ValueError("Voigt tensor requires phase-aligned square stiffness tensors.")
    valid = (
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(fraction))
        & jnp.all(fraction >= 0)
        & jnp.isclose(jnp.sum(fraction), 1.0)
        & jnp.all(jnp.linalg.eigvalsh(values) > 0)
    )
    value = jnp.sum(fraction[..., None, None] * values, axis=-3)
    return TensorHomogenizationResult(value, valid)


__all__ = [
    "HomogenizationBound",
    "TensorHomogenizationResult",
    "homogenize_scalar",
    "reuss_tensor",
    "voigt_tensor",
]
