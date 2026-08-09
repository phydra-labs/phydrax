#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._base import AbstractPositiveDefiniteKernel


class SumKernel(AbstractPositiveDefiniteKernel):
    """Finite sum of positive-definite kernels."""

    kernels: tuple[AbstractPositiveDefiniteKernel, ...]

    def __init__(self, kernels: tuple[AbstractPositiveDefiniteKernel, ...], /):
        flattened: list[AbstractPositiveDefiniteKernel] = []
        for kernel in kernels:
            if not isinstance(kernel, AbstractPositiveDefiniteKernel):
                raise TypeError("SumKernel children must be positive-definite kernels.")
            if isinstance(kernel, SumKernel):
                flattened.extend(kernel.kernels)
            else:
                flattened.append(kernel)
        if not flattened:
            raise ValueError("SumKernel requires at least one child kernel.")
        input_ndim = flattened[0].input_ndim
        if any(kernel.input_ndim != input_ndim for kernel in flattened[1:]):
            raise ValueError("SumKernel children must have equal input_ndim.")
        self.kernels = tuple(flattened)

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        value = self.kernels[0].pairwise(left, right)
        for kernel in self.kernels[1:]:
            value = value + kernel.pairwise(left, right)
        return value

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        value = self.kernels[0].matrix(left, right)
        for kernel in self.kernels[1:]:
            value = value + kernel.matrix(left, right)
        return value

    def diagonal(self, points: ArrayLike, /) -> Array:
        value = self.kernels[0].diagonal(points)
        for kernel in self.kernels[1:]:
            value = value + kernel.diagonal(points)
        return value

    @property
    def input_ndim(self) -> int:
        return self.kernels[0].input_ndim

    @property
    def max_derivative_order(self) -> int | None:
        return _minimum_regularity(self.kernels)

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        children = ",".join(kernel.kernel_id for kernel in self.kernels)
        return f"SumKernel[{children}]"


class ProductKernel(AbstractPositiveDefiniteKernel):
    """Pointwise product of positive-definite kernels."""

    kernels: tuple[AbstractPositiveDefiniteKernel, ...]

    def __init__(self, kernels: tuple[AbstractPositiveDefiniteKernel, ...], /):
        flattened: list[AbstractPositiveDefiniteKernel] = []
        for kernel in kernels:
            if not isinstance(kernel, AbstractPositiveDefiniteKernel):
                raise TypeError(
                    "ProductKernel children must be positive-definite kernels."
                )
            if isinstance(kernel, ProductKernel):
                flattened.extend(kernel.kernels)
            else:
                flattened.append(kernel)
        if not flattened:
            raise ValueError("ProductKernel requires at least one child kernel.")
        input_ndim = flattened[0].input_ndim
        if any(kernel.input_ndim != input_ndim for kernel in flattened[1:]):
            raise ValueError("ProductKernel children must have equal input_ndim.")
        self.kernels = tuple(flattened)

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        value = self.kernels[0].pairwise(left, right)
        for kernel in self.kernels[1:]:
            value = value * kernel.pairwise(left, right)
        return value

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        value = self.kernels[0].matrix(left, right)
        for kernel in self.kernels[1:]:
            value = value * kernel.matrix(left, right)
        return value

    def diagonal(self, points: ArrayLike, /) -> Array:
        value = self.kernels[0].diagonal(points)
        for kernel in self.kernels[1:]:
            value = value * kernel.diagonal(points)
        return value

    @property
    def input_ndim(self) -> int:
        return self.kernels[0].input_ndim

    @property
    def max_derivative_order(self) -> int | None:
        return _minimum_regularity(self.kernels)

    @property
    def is_unit_diagonal(self) -> bool:
        return all(kernel.is_unit_diagonal for kernel in self.kernels)

    @property
    def kernel_id(self) -> str:
        children = ",".join(kernel.kernel_id for kernel in self.kernels)
        return f"ProductKernel[{children}]"


class ScaleKernel(AbstractPositiveDefiniteKernel):
    """Nonnegative covariance scaling of a positive-definite kernel."""

    kernel: AbstractPositiveDefiniteKernel
    scale: Array

    def __init__(
        self,
        kernel: AbstractPositiveDefiniteKernel,
        scale: ArrayLike,
        /,
    ):
        if not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be a positive-definite kernel.")
        value = jnp.asarray(scale, dtype=float)
        if value.ndim != 0:
            raise ValueError("Kernel scale must be scalar.")
        self.kernel = kernel
        self.scale = eqx.error_if(
            value,
            ~jnp.isfinite(value) | (value < 0.0),
            "Kernel scale must be finite and nonnegative.",
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.scale * self.kernel.pairwise(left, right)

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.scale * self.kernel.matrix(left, right)

    def diagonal(self, points: ArrayLike, /) -> Array:
        return self.scale * self.kernel.diagonal(points)

    @property
    def input_ndim(self) -> int:
        return self.kernel.input_ndim

    @property
    def max_derivative_order(self) -> int | None:
        return self.kernel.max_derivative_order

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return f"ScaleKernel[{self.kernel.kernel_id}]"


class AmplitudeKernel(AbstractPositiveDefiniteKernel):
    """Standard-deviation amplitude scaling, returning amplitude squared times k."""

    kernel: AbstractPositiveDefiniteKernel
    amplitude: Array

    def __init__(
        self,
        kernel: AbstractPositiveDefiniteKernel,
        amplitude: ArrayLike,
        /,
    ):
        if not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be a positive-definite kernel.")
        value = jnp.asarray(amplitude, dtype=float)
        if value.ndim != 0:
            raise ValueError("Kernel amplitude must be scalar.")
        self.kernel = kernel
        self.amplitude = eqx.error_if(
            value,
            ~jnp.isfinite(value) | (value < 0.0),
            "Kernel amplitude must be finite and nonnegative.",
        )

    @property
    def variance_scale(self) -> Array:
        return self.amplitude * self.amplitude

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.variance_scale * self.kernel.pairwise(left, right)

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.variance_scale * self.kernel.matrix(left, right)

    def diagonal(self, points: ArrayLike, /) -> Array:
        return self.variance_scale * self.kernel.diagonal(points)

    @property
    def input_ndim(self) -> int:
        return self.kernel.input_ndim

    @property
    def max_derivative_order(self) -> int | None:
        return self.kernel.max_derivative_order

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return f"AmplitudeKernel[{self.kernel.kernel_id}]"


class NormalizedKernel(AbstractPositiveDefiniteKernel):
    """Unit-diagonal normalization of a strictly positive-diagonal kernel."""

    kernel: AbstractPositiveDefiniteKernel

    def __init__(self, kernel: AbstractPositiveDefiniteKernel, /):
        if not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be a positive-definite kernel.")
        self.kernel = kernel

    @staticmethod
    def _checked_diagonal(diagonal: Array, /) -> Array:
        return eqx.error_if(
            diagonal,
            jnp.any(~jnp.isfinite(diagonal)) | jnp.any(diagonal <= 0.0),
            "NormalizedKernel requires a finite strictly positive child diagonal.",
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_diagonal = self._checked_diagonal(self.kernel.pairwise(left, left))
        right_diagonal = self._checked_diagonal(self.kernel.pairwise(right, right))
        return self.kernel.pairwise(left, right) / jnp.sqrt(
            left_diagonal * right_diagonal
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_diagonal = self._checked_diagonal(self.kernel.diagonal(left))
        right_diagonal = self._checked_diagonal(self.kernel.diagonal(right))
        return self.kernel.matrix(left, right) / jnp.sqrt(
            left_diagonal[:, None] * right_diagonal[None, :]
        )

    def diagonal(self, points: ArrayLike, /) -> Array:
        diagonal = self._checked_diagonal(self.kernel.diagonal(points))
        return diagonal / diagonal

    @property
    def input_ndim(self) -> int:
        return self.kernel.input_ndim

    @property
    def max_derivative_order(self) -> int | None:
        return self.kernel.max_derivative_order

    @property
    def is_unit_diagonal(self) -> bool:
        return True

    @property
    def kernel_id(self) -> str:
        return f"NormalizedKernel[{self.kernel.kernel_id}]"


def _minimum_regularity(
    kernels: tuple[AbstractPositiveDefiniteKernel, ...], /
) -> int | None:
    finite = tuple(
        order for kernel in kernels if (order := kernel.max_derivative_order) is not None
    )
    return None if not finite else min(finite)


__all__ = [
    "AmplitudeKernel",
    "NormalizedKernel",
    "ProductKernel",
    "ScaleKernel",
    "SumKernel",
]
