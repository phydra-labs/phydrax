#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...kernels import AbstractPositiveDefiniteKernel


def validate_kernel(kernel: Any) -> Any:
    if not isinstance(kernel, AbstractPositiveDefiniteKernel) and not callable(kernel):
        raise TypeError(
            "kernel must be a Phydrax positive-definite kernel or a callable."
        )
    return kernel


def kernel_matrix(kernel: Any, left: Array, right: Array) -> Array:
    """Evaluate one two-dimensional cross-Gram matrix through shared kernel algebra."""
    if isinstance(kernel, AbstractPositiveDefiniteKernel):
        if jnp.iscomplexobj(left) or jnp.iscomplexobj(right):
            raise TypeError(
                "Phydrax positive-definite kernels require real coordinates; "
                "use an explicit callable kernel for complex coordinates."
            )
        return kernel.matrix(left, right)
    return jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(right))(left)


def kernel_diagonal(kernel: Any, points: Array) -> Array:
    if isinstance(kernel, AbstractPositiveDefiniteKernel):
        if jnp.iscomplexobj(points):
            raise TypeError(
                "Phydrax positive-definite kernels require real coordinates; "
                "use an explicit callable kernel for complex coordinates."
            )
        return kernel.diagonal(points)
    return jax.vmap(lambda x: kernel(x, x))(points)


def case_kernel_matrix(
    kernel: Any, left: Array, right: Array, case_shape: tuple[int, ...]
) -> Array:
    """Evaluate case-aligned Gram matrices without crossing independent cases."""
    cases = _size(case_shape)
    left_cases = left.reshape((cases, left.shape[-2], left.shape[-1]))
    right_cases = right.reshape((cases, right.shape[-2], right.shape[-1]))
    matrices = jax.vmap(lambda x, y: kernel_matrix(kernel, x, y))(left_cases, right_cases)
    return matrices.reshape(case_shape + matrices.shape[-2:])


def query_kernel_matrix(
    kernel: Any,
    query: Array,
    support: Array,
    case_shape: tuple[int, ...],
) -> tuple[Array, tuple[int, ...]]:
    """Return case/query/support cross-Grams and the query shape."""
    x = jnp.asarray(query)
    if x.shape[-1] != support.shape[-1]:
        raise ValueError("Query feature size does not match the fitted support.")
    if case_shape:
        if x.shape[: len(case_shape)] != case_shape:
            raise ValueError(f"Query must begin with fitted case shape {case_shape}.")
        query_shape = x.shape[len(case_shape) : -1]
        if not query_shape:
            query_shape = ()
            flat = x.reshape(case_shape + (1, x.shape[-1]))
        else:
            flat = x.reshape(case_shape + (_size(query_shape), x.shape[-1]))
        matrix = case_kernel_matrix(kernel, flat, support, case_shape)
        if query_shape:
            matrix = matrix.reshape(case_shape + query_shape + (support.shape[-2],))
        else:
            matrix = matrix[..., 0, :]
        return matrix, tuple(int(s) for s in query_shape)
    query_shape = x.shape[:-1]
    flat = x.reshape((-1, x.shape[-1]))
    matrix = kernel_matrix(kernel, flat, support)
    if query_shape:
        matrix = matrix.reshape(query_shape + (support.shape[-2],))
    else:
        matrix = matrix[0]
    return matrix, tuple(int(s) for s in query_shape)


def flatten_targets(
    target: Array, sample_shape: tuple[int, ...]
) -> tuple[Array, tuple[int, ...]]:
    output_shape = target.shape[len(sample_shape) :]
    return target.reshape(sample_shape + (-1,)), tuple(int(s) for s in output_shape)


def finite_array(value: Array) -> Array:
    return jnp.isfinite(jnp.real(value)) & jnp.isfinite(jnp.imag(value))


def validated_weights(value: Array) -> Array:
    weights = jnp.asarray(value, dtype=float)
    return eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
        "Sample and measure weights must be finite and nonnegative.",
    )


def _size(shape: tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


__all__ = [
    "case_kernel_matrix",
    "finite_array",
    "flatten_targets",
    "kernel_diagonal",
    "kernel_matrix",
    "query_kernel_matrix",
    "validate_kernel",
    "validated_weights",
]
