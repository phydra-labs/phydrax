#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._base import (
    _as_input,
    _as_inputs,
    _as_point,
    _as_points,
    AbstractPositiveDefiniteKernel,
)


class InputTransformedKernel(AbstractPositiveDefiniteKernel):
    """Positive-definite pullback through a deterministic input transform."""

    kernel: AbstractPositiveDefiniteKernel
    transform_function: Callable[[Array], Array]
    transform_id: str = eqx.field(static=True)
    transform_derivative_order: int | None = eqx.field(static=True)
    _input_ndim: int = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractPositiveDefiniteKernel,
        transform_function: Callable[[Array], Array],
        /,
        *,
        transform_id: str,
        max_derivative_order: int | None = 0,
        input_ndim: int = 1,
    ):
        if not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be a positive-definite kernel.")
        if not callable(transform_function):
            raise TypeError("transform_function must be callable.")
        if not isinstance(transform_id, str) or not transform_id:
            raise ValueError("transform_id must be a nonempty string.")
        if max_derivative_order is not None and int(max_derivative_order) < 0:
            raise ValueError("max_derivative_order must be nonnegative or None.")
        resolved_input_ndim = int(input_ndim)
        if resolved_input_ndim <= 0:
            raise ValueError("input_ndim must be positive.")
        self.kernel = kernel
        self.transform_function = transform_function
        self.transform_id = transform_id
        self.transform_derivative_order = (
            None if max_derivative_order is None else int(max_derivative_order)
        )
        self._input_ndim = resolved_input_ndim

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_input = _as_input(left, input_ndim=self.input_ndim, name="left")
        right_input = _as_input(right, input_ndim=self.input_ndim, name="right")
        return self.kernel.pairwise(
            self._transform_input(left_input),
            self._transform_input(right_input),
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_inputs = _as_inputs(left, input_ndim=self.input_ndim, name="left")
        right_inputs = _as_inputs(right, input_ndim=self.input_ndim, name="right")
        left_features = jax.vmap(self._transform_input)(left_inputs)
        right_features = jax.vmap(self._transform_input)(right_inputs)
        return self.kernel.matrix(left_features, right_features)

    def diagonal(self, points: ArrayLike, /) -> Array:
        inputs = _as_inputs(points, input_ndim=self.input_ndim, name="points")
        features = jax.vmap(self._transform_input)(inputs)
        return self.kernel.diagonal(features)

    def _transform_input(self, value: Array, /) -> Array:
        return _as_input(
            self.transform_function(value),
            input_ndim=self.kernel.input_ndim,
            name="transformed input",
        )

    @property
    def input_ndim(self) -> int:
        return self._input_ndim

    @property
    def max_derivative_order(self) -> int | None:
        base = self.kernel.max_derivative_order
        transform = self.transform_derivative_order
        if base is None:
            return transform
        if transform is None:
            return base
        return min(base, transform)

    @property
    def is_unit_diagonal(self) -> bool:
        return self.kernel.is_unit_diagonal

    @property
    def kernel_id(self) -> str:
        return f"InputTransformedKernel[{self.transform_id},{self.kernel.kernel_id}]"


class AffineInputTransform(StrictModule):
    """Coordinatewise affine map ``point -> (point - offset) / scale``."""

    offset: Array
    scale: Array

    def __init__(self, offset: ArrayLike, scale: ArrayLike, /):
        offset_value = jnp.asarray(offset, dtype=float)
        scale_value = jnp.asarray(scale, dtype=float)
        if offset_value.ndim > 1 or scale_value.ndim > 1:
            raise ValueError("Affine offset and scale must be scalar or vectors.")
        if (
            offset_value.shape != scale_value.shape
            and offset_value.ndim != 0
            and scale_value.ndim != 0
        ):
            raise ValueError("Affine offset and scale vectors must have equal shape.")
        offset_value = eqx.error_if(
            offset_value,
            jnp.any(~jnp.isfinite(offset_value)),
            "Affine input offset must be finite.",
        )
        scale_value = eqx.error_if(
            scale_value,
            jnp.any(~jnp.isfinite(scale_value)) | jnp.any(scale_value <= 0.0),
            "Affine input scale must be finite and strictly positive.",
        )
        self.offset = offset_value
        self.scale = scale_value

    def __call__(self, point: ArrayLike, /) -> Array:
        value = _as_point(point, name="point")
        if self.offset.ndim == 1 and self.offset.shape[0] not in (1, value.shape[0]):
            raise ValueError("Affine offset must match the coordinate size.")
        if self.scale.ndim == 1 and self.scale.shape[0] not in (1, value.shape[0]):
            raise ValueError("Affine scale must match the coordinate size.")
        return (value - self.offset) / self.scale

    @classmethod
    def from_points(
        cls,
        points: ArrayLike,
        /,
        *,
        minimum_scale: ArrayLike = 1e-12,
    ) -> AffineInputTransform:
        design = _as_points(points, name="points")
        floor = jnp.asarray(minimum_scale, dtype=float)
        if floor.ndim != 0:
            raise ValueError("minimum_scale must be scalar.")
        floor = eqx.error_if(
            floor,
            ~jnp.isfinite(floor) | (floor <= 0.0),
            "minimum_scale must be finite and strictly positive.",
        )
        offset = jnp.mean(design, axis=0)
        scale = jnp.maximum(jnp.std(design, axis=0), floor)
        return cls(offset, scale)


__all__ = ["AffineInputTransform", "InputTransformedKernel"]
