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
    """Real scalar positive-definite kernel over declared array inputs."""

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
    def input_ndim(self) -> int:
        """Number of trailing axes forming one kernel input."""
        return 1

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
        point_design = _as_inputs(
            points,
            input_ndim=self.input_ndim,
            name="points",
        )
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


def _as_input(
    value: ArrayLike,
    /,
    *,
    input_ndim: int,
    name: str,
) -> Array:
    rank = int(input_ndim)
    if rank <= 0:
        raise ValueError("input_ndim must be positive.")
    sample = jnp.asarray(value, dtype=float)
    if rank == 1 and sample.ndim == 0:
        sample = sample.reshape((1,))
    if sample.ndim != rank or any(int(size) <= 0 for size in sample.shape):
        raise ValueError(
            f"{name} must have {rank} nonempty input axes; got shape {sample.shape}."
        )
    return eqx.error_if(
        sample,
        jnp.any(~jnp.isfinite(sample)),
        f"{name} must contain only finite values.",
    )


def _as_inputs(
    value: ArrayLike,
    /,
    *,
    input_ndim: int,
    name: str,
) -> Array:
    rank = int(input_ndim)
    if rank <= 0:
        raise ValueError("input_ndim must be positive.")
    samples = jnp.asarray(value, dtype=float)
    if rank == 1 and samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != rank + 1 or any(int(size) <= 0 for size in samples.shape):
        raise ValueError(
            f"{name} must have one design axis followed by {rank} nonempty "
            f"input axes; got shape {samples.shape}."
        )
    return eqx.error_if(
        samples,
        jnp.any(~jnp.isfinite(samples)),
        f"{name} must contain only finite values.",
    )


def _as_point(value: ArrayLike, /, *, name: str) -> Array:
    return _as_input(value, input_ndim=1, name=name)


def _as_points(value: ArrayLike, /, *, name: str) -> Array:
    return _as_inputs(value, input_ndim=1, name=name)


__all__ = ["AbstractPositiveDefiniteKernel", "AbstractUnitDiagonalKernel"]
