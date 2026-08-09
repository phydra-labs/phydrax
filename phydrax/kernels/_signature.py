#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import factorial
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._base import _as_input, _as_inputs, AbstractPositiveDefiniteKernel


class SignaturePDEKernel(AbstractPositiveDefiniteKernel):
    """Truncated signature kernel evaluated through its Goursat PDE recurrence.

    The recurrence propagates monomial boundary coefficients over every pair of
    path intervals. An additional level axis truncates the global Picard series,
    so order ``m`` is exactly the positive-definite inner product of signatures
    truncated after tensor level ``m`` rather than a locally truncated PDE
    approximation.
    """

    static_kernel: AbstractPositiveDefiniteKernel
    polynomial_order: int = eqx.field(static=True)
    pair_block_size: int = eqx.field(static=True)

    def __init__(
        self,
        static_kernel: AbstractPositiveDefiniteKernel,
        /,
        *,
        polynomial_order: int = 5,
        pair_block_size: int = 64,
    ):
        if not isinstance(static_kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("static_kernel must be a positive-definite kernel.")
        if static_kernel.input_ndim != 1:
            raise ValueError("static_kernel.input_ndim must be 1.")
        if not isinstance(polynomial_order, Integral) or isinstance(
            polynomial_order, bool
        ):
            raise TypeError("polynomial_order must be an integer.")
        if not isinstance(pair_block_size, Integral) or isinstance(pair_block_size, bool):
            raise TypeError("pair_block_size must be an integer.")
        resolved_order = int(polynomial_order)
        resolved_block_size = int(pair_block_size)
        if resolved_order <= 0:
            raise ValueError("polynomial_order must be positive.")
        if resolved_block_size <= 0:
            raise ValueError("pair_block_size must be positive.")
        self.static_kernel = static_kernel
        self.polynomial_order = resolved_order
        self.pair_block_size = resolved_block_size

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_path = _as_input(left, input_ndim=2, name="left")
        right_path = _as_input(right, input_ndim=2, name="right")
        _validate_path_channels(left_path, right_path)
        return self._pairwise_paths(left_path, right_path)

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_paths = _as_inputs(left, input_ndim=2, name="left")
        right_paths = _as_inputs(right, input_ndim=2, name="right")
        _validate_path_channels(left_paths, right_paths)
        left_count = int(left_paths.shape[0])
        right_count = int(right_paths.shape[0])
        if left_count == 0 or right_count == 0:
            return jnp.empty(
                (left_count, right_count),
                dtype=jnp.result_type(left_paths, right_paths),
            )

        pair_count = left_count * right_count
        block_size = self.pair_block_size
        block_count = (pair_count + block_size - 1) // block_size
        block_offsets = jnp.arange(block_count, dtype=jnp.int32) * block_size
        within_block = jnp.arange(block_size, dtype=jnp.int32)

        def evaluate_block(offset: Array, /) -> Array:
            indices = offset + within_block
            valid = indices < pair_count
            safe_indices = jnp.minimum(indices, pair_count - 1)
            left_indices = safe_indices // right_count
            right_indices = safe_indices % right_count
            values = jax.vmap(self._pairwise_paths)(
                left_paths[left_indices], right_paths[right_indices]
            )
            return jnp.where(valid, values, 0.0)

        blocks = jax.lax.map(evaluate_block, block_offsets)
        return blocks.reshape((-1,))[:pair_count].reshape((left_count, right_count))

    def diagonal(self, points: ArrayLike, /) -> Array:
        paths = _as_inputs(points, input_ndim=2, name="points")
        return jax.vmap(self._pairwise_paths)(paths, paths)

    def _pairwise_paths(self, left_path: Array, right_path: Array, /) -> Array:
        left_segment_count = int(left_path.shape[0]) - 1
        right_segment_count = int(right_path.shape[0]) - 1
        if left_segment_count == 0 or right_segment_count == 0:
            return jnp.asarray(1.0, dtype=jnp.result_type(left_path, right_path))

        point_gram = self.static_kernel.matrix(left_path, right_path)
        increment_gram = jnp.diff(jnp.diff(point_gram, axis=0), axis=1)
        order = self.polynomial_order
        identity = (
            jnp.zeros((order + 1, order + 1), dtype=increment_gram.dtype)
            .at[0, 0]
            .set(1.0)
        )
        horizontal_boundaries = jnp.broadcast_to(
            identity, (left_segment_count,) + identity.shape
        )

        def propagate_row(
            bottom_boundaries: Array, increment_column: Array, /
        ) -> tuple[Array, Array]:
            def propagate_cell(
                left_boundary: Array, cell: tuple[Array, Array], /
            ) -> tuple[Array, Array]:
                bottom_boundary, increment = cell
                top_boundary, right_boundary = self._propagate_rectangle(
                    bottom_boundary, left_boundary, increment
                )
                return right_boundary, top_boundary

            right_boundary, top_boundaries = jax.lax.scan(
                propagate_cell,
                identity,
                (bottom_boundaries, increment_column),
            )
            return top_boundaries, right_boundary

        top_boundaries, right_boundaries = jax.lax.scan(
            propagate_row,
            horizontal_boundaries,
            jnp.swapaxes(increment_gram, 0, 1),
        )
        top_value = jnp.sum(top_boundaries[-1])
        right_value = jnp.sum(right_boundaries[-1])
        return 0.5 * (top_value + right_value)

    def _propagate_rectangle(
        self,
        bottom: Array,
        left: Array,
        increment: Array,
        /,
    ) -> tuple[Array, Array]:
        order = self.polynomial_order
        dtype = jnp.result_type(bottom, left, increment)
        same_coefficients, cross_coefficients = _propagation_coefficients(
            order, dtype=dtype
        )
        powers = [jnp.ones((), dtype=dtype)]
        for _ in range(order):
            powers.append(powers[-1] * increment)
        powers_array = jnp.stack(powers)
        top = jnp.zeros_like(bottom)
        right = jnp.zeros_like(left)
        cross_from_left = cross_coefficients @ left
        cross_from_bottom = cross_coefficients @ bottom

        for shift in range(order + 1):
            size = order + 1 - shift
            same_weights = jnp.diagonal(same_coefficients[shift:, :size])[:, None]
            top = top.at[shift:, shift:].add(
                powers_array[shift] * same_weights * bottom[:size, :size]
            )
            right = right.at[shift:, shift:].add(
                powers_array[shift] * same_weights * left[:size, :size]
            )
            top = top.at[shift, shift:].add(
                powers_array[shift] * cross_from_left[shift, :size]
            )
            right = right.at[shift, shift:].add(
                powers_array[shift] * cross_from_bottom[shift, :size]
            )
        return top, right

    @property
    def input_ndim(self) -> int:
        return 2

    @property
    def max_derivative_order(self) -> int:
        return 0

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return (
            f"SignaturePDEKernel[order={self.polynomial_order},"
            f"static={self.static_kernel.kernel_id}]"
        )


def _propagation_coefficients(order: int, /, *, dtype: jnp.dtype) -> tuple[Array, Array]:
    same = tuple(
        tuple(
            (factorial(k) / (factorial(n - k) * factorial(n)) if k <= n else 0.0)
            for k in range(order + 1)
        )
        for n in range(order + 1)
    )
    cross = tuple(
        tuple(
            (factorial(k) / (factorial(n + k) * factorial(n)) if k >= 1 else 0.0)
            for k in range(order + 1)
        )
        for n in range(order + 1)
    )
    return jnp.asarray(same, dtype=dtype), jnp.asarray(cross, dtype=dtype)


def _validate_path_channels(left: Array, right: Array, /) -> None:
    if left.shape[-1] != right.shape[-1]:
        raise ValueError("Path channel dimensions must be equal.")


__all__ = ["SignaturePDEKernel"]
