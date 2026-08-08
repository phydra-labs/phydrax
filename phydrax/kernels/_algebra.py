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
    def max_derivative_order(self) -> int | None:
        return self.kernel.max_derivative_order

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return f"AmplitudeKernel[{self.kernel.kernel_id}]"


def _minimum_regularity(
    kernels: tuple[AbstractPositiveDefiniteKernel, ...], /
) -> int | None:
    finite = tuple(
        order for kernel in kernels if (order := kernel.max_derivative_order) is not None
    )
    return None if not finite else min(finite)


__all__ = [
    "AmplitudeKernel",
    "ProductKernel",
    "ScaleKernel",
    "SumKernel",
]
