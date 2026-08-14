#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class AbstractParameterTransform(StrictModule):
    """Differentiable map from unconstrained coordinates to physical parameters."""

    preserves_shape: ClassVar[bool] = False

    @abstractmethod
    def __call__(self, raw: Any, /) -> Array:
        raise NotImplementedError


class IdentityTransform(AbstractParameterTransform):
    """Leave unconstrained coordinates unchanged."""

    preserves_shape: ClassVar[bool] = True

    def __call__(self, raw: ArrayLike, /) -> Array:
        return jnp.asarray(raw)


class PositiveTransform(AbstractParameterTransform):
    """Map real coordinates to values strictly above a configured minimum."""

    preserves_shape: ClassVar[bool] = True
    minimum: float = eqx.field(static=True)

    def __init__(self, minimum: float = 0.0):
        minimum_value = float(minimum)
        if not math.isfinite(minimum_value):
            raise ValueError("minimum must be finite.")
        self.minimum = minimum_value

    def __call__(self, raw: ArrayLike, /) -> Array:
        value = jnp.asarray(raw)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("PositiveTransform requires real coordinates.")
        return jnp.asarray(self.minimum, dtype=value.dtype) + jax.nn.softplus(value)


class IntervalTransform(AbstractParameterTransform):
    """Map real coordinates into an open finite interval."""

    preserves_shape: ClassVar[bool] = True
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)

    def __init__(self, lower: float, upper: float):
        lower_value = float(lower)
        upper_value = float(upper)
        if not math.isfinite(lower_value) or not math.isfinite(upper_value):
            raise ValueError("interval bounds must be finite.")
        if upper_value <= lower_value:
            raise ValueError("upper must exceed lower.")
        self.lower = lower_value
        self.upper = upper_value

    def __call__(self, raw: ArrayLike, /) -> Array:
        value = jnp.asarray(raw)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("IntervalTransform requires real coordinates.")
        lower = jnp.asarray(self.lower, dtype=value.dtype)
        width = jnp.asarray(self.upper - self.lower, dtype=value.dtype)
        return lower + width * jax.nn.sigmoid(value)


class SimplexTransform(AbstractParameterTransform):
    """Map additive log-ratio coordinates into the interior of a simplex.

    A raw trailing axis of length ``k - 1`` produces ``k`` positive
    coordinates whose sum is one. The final logit is fixed at zero, removing
    the translation degeneracy of an unconstrained ``k``-logit softmax.
    """

    def __call__(self, raw: ArrayLike, /) -> Array:
        value = jnp.asarray(raw)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("SimplexTransform requires real coordinates.")
        if value.ndim < 1 or int(value.shape[-1]) < 1:
            raise ValueError(
                "SimplexTransform requires a non-empty trailing coordinate axis."
            )
        reference = jnp.zeros(value.shape[:-1] + (1,), dtype=value.dtype)
        return jax.nn.softmax(jnp.concatenate((value, reference), axis=-1), axis=-1)


def _square_matrix(raw: ArrayLike, /, *, name: str) -> Array:
    value = jnp.asarray(raw)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise TypeError(f"{name} requires real coordinates.")
    if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
        raise ValueError(f"{name} requires a square matrix in the trailing axes.")
    return value


class SymmetricTransform(AbstractParameterTransform):
    """Project a square real matrix onto the symmetric matrices."""

    preserves_shape: ClassVar[bool] = True

    def __call__(self, raw: ArrayLike, /) -> Array:
        value = _square_matrix(raw, name="SymmetricTransform")
        return 0.5 * (value + jnp.swapaxes(value, -1, -2))


class SkewSymmetricTransform(AbstractParameterTransform):
    """Project a square real matrix onto the skew-symmetric matrices."""

    preserves_shape: ClassVar[bool] = True

    def __call__(self, raw: ArrayLike, /) -> Array:
        value = _square_matrix(raw, name="SkewSymmetricTransform")
        return 0.5 * (value - jnp.swapaxes(value, -1, -2))


def _packed_dimension(size: int, /, *, name: str) -> int:
    discriminant = 1 + 8 * int(size)
    root = math.isqrt(discriminant)
    dimension = (root - 1) // 2
    if root * root != discriminant or dimension * (dimension + 1) // 2 != size:
        raise ValueError(
            f"{name} requires a trailing packed-triangle size n(n+1)/2; got {size}."
        )
    return dimension


def _strict_packed_dimension(size: int, /, *, name: str) -> int:
    discriminant = 1 + 8 * int(size)
    root = math.isqrt(discriminant)
    dimension = (1 + root) // 2
    if root * root != discriminant or dimension * (dimension - 1) // 2 != size:
        raise ValueError(
            f"{name} requires a trailing strict-triangle size n(n-1)/2; got {size}."
        )
    return dimension


class PackedSkewSymmetricTransform(AbstractParameterTransform):
    """Construct a skew-symmetric matrix from packed strict-lower coordinates."""

    def __call__(self, raw: ArrayLike, /) -> Array:
        value = jnp.asarray(raw)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("PackedSkewSymmetricTransform requires real coordinates.")
        if value.ndim < 1:
            raise ValueError(
                "PackedSkewSymmetricTransform requires a packed trailing axis."
            )
        dimension = _strict_packed_dimension(
            int(value.shape[-1]), name="PackedSkewSymmetricTransform"
        )
        row, column = jnp.tril_indices(dimension, k=-1)
        matrix = jnp.zeros(value.shape[:-1] + (dimension, dimension), dtype=value.dtype)
        matrix = matrix.at[..., row, column].set(value)
        return matrix.at[..., column, row].set(-value)


class PositiveSemidefiniteTransform(AbstractParameterTransform):
    """Construct a PSD matrix from packed lower-triangular factor coordinates."""

    def factor(self, raw: ArrayLike, /) -> Array:
        value = jnp.asarray(raw)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("PositiveSemidefiniteTransform requires real coordinates.")
        if value.ndim < 1:
            raise ValueError(
                "PositiveSemidefiniteTransform requires a packed trailing axis."
            )
        dimension = _packed_dimension(
            int(value.shape[-1]), name="PositiveSemidefiniteTransform"
        )
        row, column = jnp.tril_indices(dimension)
        factor = jnp.zeros(value.shape[:-1] + (dimension, dimension), dtype=value.dtype)
        return factor.at[..., row, column].set(value)

    def __call__(self, raw: ArrayLike, /) -> Array:
        factor = self.factor(raw)
        return factor @ jnp.swapaxes(factor, -1, -2)


class PositiveDefiniteTransform(AbstractParameterTransform):
    """Construct an SPD matrix from packed lower-triangular coordinates."""

    minimum_diagonal: float = eqx.field(static=True)

    def __init__(self, minimum_diagonal: float = 1e-6):
        minimum = float(minimum_diagonal)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_diagonal must be finite and positive.")
        self.minimum_diagonal = minimum

    def factor(self, raw: ArrayLike, /) -> Array:
        value = jnp.asarray(raw)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("PositiveDefiniteTransform requires real coordinates.")
        if value.ndim < 1:
            raise ValueError("PositiveDefiniteTransform requires a packed trailing axis.")
        dimension = _packed_dimension(
            int(value.shape[-1]), name="PositiveDefiniteTransform"
        )
        row, column = jnp.tril_indices(dimension)
        factor = jnp.zeros(value.shape[:-1] + (dimension, dimension), dtype=value.dtype)
        factor = factor.at[..., row, column].set(value)
        diagonal = jnp.diagonal(factor, axis1=-2, axis2=-1)
        positive = jax.nn.softplus(diagonal) + jnp.asarray(
            self.minimum_diagonal, dtype=value.dtype
        )
        diagonal_index = jnp.arange(dimension)
        return factor.at[..., diagonal_index, diagonal_index].set(positive)

    def __call__(self, raw: ArrayLike, /) -> Array:
        factor = self.factor(raw)
        return factor @ jnp.swapaxes(factor, -1, -2)


class HurwitzTransform(AbstractParameterTransform):
    """Construct a matrix with strictly negative-definite symmetric part."""

    minimum_damping: float = eqx.field(static=True)

    def __init__(self, minimum_damping: float = 1e-6):
        minimum = float(minimum_damping)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_damping must be finite and positive.")
        self.minimum_damping = minimum

    def __call__(self, raw: tuple[ArrayLike, ArrayLike], /) -> Array:
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise TypeError("HurwitzTransform requires (skew_raw, damping_raw).")
        skew = SkewSymmetricTransform()(raw[0])
        damping = PositiveDefiniteTransform(self.minimum_damping)(raw[1])
        if skew.shape != damping.shape:
            raise ValueError(
                "Hurwitz skew and damping coordinates imply different shapes."
            )
        return skew - damping


class SchurStableTransform(AbstractParameterTransform):
    """Map Hurwitz coordinates into a strictly Schur-stable matrix."""

    minimum_damping: float = eqx.field(static=True)
    step: float = eqx.field(static=True)

    def __init__(self, *, minimum_damping: float = 1e-6, step: float = 1.0):
        minimum = float(minimum_damping)
        step_value = float(step)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_damping must be finite and positive.")
        if not math.isfinite(step_value) or step_value <= 0.0:
            raise ValueError("step must be finite and positive.")
        self.minimum_damping = minimum
        self.step = step_value

    def __call__(self, raw: tuple[ArrayLike, ArrayLike], /) -> Array:
        generator = HurwitzTransform(self.minimum_damping)(raw)
        dimension = int(generator.shape[-1])
        identity = jnp.eye(dimension, dtype=generator.dtype)
        step = jnp.asarray(self.step, dtype=generator.dtype)
        return jnp.linalg.solve(
            identity - step * generator,
            identity + step * generator,
        )


class StiefelTransform(AbstractParameterTransform):
    """Map a full-rank matrix to the Stiefel manifold by canonicalized thin QR."""

    preserves_shape: ClassVar[bool] = True

    def __call__(self, raw: ArrayLike, /) -> Array:
        value = jnp.asarray(raw)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("StiefelTransform requires real coordinates.")
        if value.ndim < 2 or int(value.shape[-2]) < int(value.shape[-1]):
            raise ValueError(
                "StiefelTransform requires trailing shape (rows, columns) with rows >= columns."
            )
        orthogonal, triangular = jnp.linalg.qr(value, mode="reduced")
        diagonal = jnp.diagonal(triangular, axis1=-2, axis2=-1)
        signs = jnp.where(diagonal < 0.0, -1.0, 1.0)
        return orthogonal * signs[..., None, :]


__all__ = [
    "AbstractParameterTransform",
    "HurwitzTransform",
    "IdentityTransform",
    "IntervalTransform",
    "PackedSkewSymmetricTransform",
    "PositiveDefiniteTransform",
    "PositiveSemidefiniteTransform",
    "PositiveTransform",
    "SchurStableTransform",
    "SimplexTransform",
    "SkewSymmetricTransform",
    "StiefelTransform",
    "SymmetricTransform",
]
