#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class AbstractPositiveDefiniteKernel(StrictModule):
    """Real scalar positive-definite kernel over coordinate vectors."""

    @abstractmethod
    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Evaluate the kernel at two individual points."""
        raise NotImplementedError

    @abstractmethod
    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Evaluate a Gram or cross-Gram matrix over two point designs."""
        raise NotImplementedError

    @abstractmethod
    def diagonal(self, points: ArrayLike, /) -> Array:
        """Evaluate the Gram diagonal without materializing the Gram matrix."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_derivative_order(self) -> int | None:
        """Return the certified mean-square derivative order, or no finite limit."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_unit_diagonal(self) -> bool:
        """Whether the kernel guarantees a unit diagonal for every valid point."""
        raise NotImplementedError

    @property
    @abstractmethod
    def kernel_id(self) -> str:
        """Return stable diagnostic metadata without reconstructive semantics."""
        raise NotImplementedError

    def __add__(self, other: Any) -> AbstractPositiveDefiniteKernel:
        if not isinstance(other, AbstractPositiveDefiniteKernel):
            raise TypeError("Positive-definite kernels may only be added to kernels.")
        from ._algebra import SumKernel

        return SumKernel((self, other))

    def __mul__(self, other: Any) -> AbstractPositiveDefiniteKernel:
        from ._algebra import ProductKernel, ScaleKernel

        if isinstance(other, AbstractPositiveDefiniteKernel):
            return ProductKernel((self, other))
        return ScaleKernel(self, other)

    def __rmul__(self, other: Any) -> AbstractPositiveDefiniteKernel:
        from ._algebra import ScaleKernel

        return ScaleKernel(self, other)


class AbstractUnitDiagonalKernel(AbstractPositiveDefiniteKernel):
    """Positive-definite correlation kernel with an exact unit diagonal."""

    def diagonal(self, points: ArrayLike, /) -> Array:
        point_design = _as_points(points, name="points")
        return jnp.ones((point_design.shape[0],), dtype=point_design.dtype)

    @property
    def is_unit_diagonal(self) -> bool:
        return True


def _pairwise_matrix(
    kernel: AbstractPositiveDefiniteKernel,
    left: ArrayLike,
    right: ArrayLike,
    /,
) -> Array:
    left_points = _as_points(left, name="left")
    right_points = _as_points(right, name="right")
    if left_points.shape[1] != right_points.shape[1]:
        raise ValueError("Kernel point designs must have equal coordinate size.")
    return jax.vmap(
        lambda point: jax.vmap(lambda other: kernel.pairwise(point, other))(right_points)
    )(left_points)


def _as_point(value: ArrayLike, /, *, name: str) -> Array:
    point = jnp.asarray(value, dtype=float)
    if point.ndim == 0:
        point = point.reshape((1,))
    if point.ndim != 1 or point.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty coordinate vector.")
    return eqx.error_if(
        point,
        jnp.any(~jnp.isfinite(point)),
        f"{name} must contain only finite coordinates.",
    )


def _as_points(value: ArrayLike, /, *, name: str) -> Array:
    points = jnp.asarray(value, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape (point, coordinate) with nonempty axes."
        )
    return eqx.error_if(
        points,
        jnp.any(~jnp.isfinite(points)),
        f"{name} must contain only finite coordinates.",
    )


__all__ = ["AbstractPositiveDefiniteKernel", "AbstractUnitDiagonalKernel"]
