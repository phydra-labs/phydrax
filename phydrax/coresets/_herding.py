#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.kernels import (
    AbstractPositiveDefiniteKernel,
    SquaredExponentialKernel,
)

from .._strict import StrictModule
from ._kernel_reductions import _weighted_kernel_mean, _weighted_kernel_sum
from ._types import CoresetSelection, KernelHerdingDiagnostics
from ._weights import log_weights_from_normalized, normalized_weights


class KernelHerding(StrictModule):
    """Fixed-size greedy minimization of weighted empirical MMD."""

    kernel: AbstractPositiveDefiniteKernel
    num_points: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    unique: bool = eqx.field(static=True)

    def __init__(
        self,
        num_points: int,
        /,
        *,
        kernel: AbstractPositiveDefiniteKernel | None = None,
        block_size: int = 256,
        unique: bool = True,
    ):
        count = int(num_points)
        block = int(block_size)
        if count <= 0:
            raise ValueError("KernelHerding num_points must be positive.")
        if block <= 0:
            raise ValueError("KernelHerding block_size must be positive.")
        if kernel is not None and not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be an AbstractPositiveDefiniteKernel or None.")
        self.num_points = count
        self.kernel = SquaredExponentialKernel() if kernel is None else kernel
        self.block_size = block
        self.unique = bool(unique)


def _mmd_from_weights(
    source_points: Array,
    source_weights: Array,
    comparison_points: Array,
    comparison_weights: Array,
    /,
    *,
    kernel: AbstractPositiveDefiniteKernel,
    block_size: int,
) -> Array:
    source_self = _weighted_kernel_sum(
        kernel,
        source_points,
        source_weights,
        source_points,
        source_weights,
        block_size=block_size,
    )
    comparison_self = _weighted_kernel_sum(
        kernel,
        comparison_points,
        comparison_weights,
        comparison_points,
        comparison_weights,
        block_size=block_size,
    )
    cross = _weighted_kernel_sum(
        kernel,
        source_points,
        source_weights,
        comparison_points,
        comparison_weights,
        block_size=block_size,
    )
    squared = jnp.maximum(source_self + comparison_self - 2.0 * cross, 0.0)
    return jnp.sqrt(squared)


def weighted_mmd(
    source_points: Array,
    comparison_points: Array,
    /,
    *,
    source_log_weights: Array | None = None,
    comparison_log_weights: Array | None = None,
    source_mask: Array | None = None,
    comparison_mask: Array | None = None,
    kernel: AbstractPositiveDefiniteKernel | None = None,
    block_size: int = 256,
) -> Array:
    """Compute blockwise MMD between two finite nonnegative measures."""
    source = jnp.asarray(source_points, dtype=float)
    comparison = jnp.asarray(comparison_points, dtype=float)
    kernel_ = SquaredExponentialKernel() if kernel is None else kernel
    if not isinstance(kernel_, AbstractPositiveDefiniteKernel):
        raise TypeError("kernel must be an AbstractPositiveDefiniteKernel or None.")
    expected_rank = kernel_.input_ndim + 1
    if source.ndim != expected_rank or comparison.ndim != expected_rank:
        raise ValueError(
            "MMD inputs must have one design axis followed by "
            f"{kernel_.input_ndim} kernel input axes."
        )
    source_rows = jnp.all(jnp.isfinite(source), axis=tuple(range(1, source.ndim)))
    comparison_rows = jnp.all(
        jnp.isfinite(comparison), axis=tuple(range(1, comparison.ndim))
    )
    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive.")
    source_weights, _, source_valid, _ = normalized_weights(
        int(source.shape[0]),
        log_weights=source_log_weights,
        mask=source_mask,
        rows_valid=source_rows,
    )
    comparison_weights, _, comparison_valid, _ = normalized_weights(
        int(comparison.shape[0]),
        log_weights=comparison_log_weights,
        mask=comparison_mask,
        rows_valid=comparison_rows,
    )
    value = _mmd_from_weights(
        jnp.nan_to_num(source),
        source_weights,
        jnp.nan_to_num(comparison),
        comparison_weights,
        kernel=kernel_,
        block_size=block,
    )
    return jnp.where(source_valid & comparison_valid, value, jnp.nan)


def kernel_herd(
    points: Array,
    method: KernelHerding,
    /,
    *,
    log_weights: Array | None = None,
    mask: Array | None = None,
) -> CoresetSelection:
    """Select a fixed-capacity weighted empirical kernel herding coreset."""
    if not isinstance(method, KernelHerding):
        raise TypeError("method must be a KernelHerding.")
    values = jnp.asarray(points, dtype=float)
    expected_rank = method.kernel.input_ndim + 1
    if values.ndim != expected_rank:
        raise ValueError(
            "points must have one source axis followed by "
            f"{method.kernel.input_ndim} kernel input axes."
        )
    source_points = int(values.shape[0])
    input_shape = tuple(int(size) for size in values.shape[1:])
    if source_points < 1:
        raise ValueError("Kernel herding requires at least one source point.")
    if method.unique and method.num_points > source_points:
        raise ValueError("Unique kernel herding cannot select more points than supplied.")
    rows_valid = jnp.all(jnp.isfinite(values), axis=tuple(range(1, values.ndim)))
    weights, active_source, input_valid, log_source_mass = normalized_weights(
        source_points,
        log_weights=log_weights,
        mask=mask,
        rows_valid=rows_valid,
    )
    safe_points = jnp.nan_to_num(values)
    target_mean = _weighted_kernel_mean(
        method.kernel,
        safe_points,
        weights,
        block_size=method.block_size,
    )
    capacity = method.num_points
    initial = (
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=bool),
        jnp.zeros((source_points,), dtype=safe_points.dtype),
        jnp.zeros((source_points,), dtype=bool),
    )

    def body(iteration, state):
        indices, output_mask, penalty, selected = state
        denominator = jnp.asarray(iteration + 1, dtype=safe_points.dtype)
        objective = target_mean - penalty / denominator
        eligible = active_source & (~selected if method.unique else True)
        candidate = jnp.asarray(
            jnp.argmax(jnp.where(eligible, objective, -jnp.inf)),
            dtype=jnp.int32,
        )
        valid_choice = input_valid & jnp.any(eligible)
        indices = indices.at[iteration].set(candidate)
        output_mask = output_mask.at[iteration].set(valid_choice)
        update = method.kernel.matrix(safe_points, safe_points[candidate][None, ...])[
            :, 0
        ]
        penalty = penalty + jnp.where(valid_choice, update, 0.0)
        selected = selected.at[candidate].set(selected[candidate] | valid_choice)
        return indices, output_mask, penalty, selected

    indices, output_mask, _, _ = jax.lax.fori_loop(0, capacity, body, initial)
    active_points = jnp.sum(output_mask, dtype=jnp.int32)
    equal_weight = jnp.where(
        active_points > 0,
        1.0 / active_points.astype(safe_points.dtype),
        0.0,
    )
    selected_weights = jnp.where(output_mask, equal_weight, 0.0)
    selected_points = safe_points[indices]
    discrepancy = _mmd_from_weights(
        safe_points,
        weights,
        selected_points,
        selected_weights,
        kernel=method.kernel,
        block_size=method.block_size,
    )
    output_valid = input_valid & (active_points > 0) & jnp.isfinite(discrepancy)
    output_mask = output_mask & output_valid
    selected_weights = jnp.where(output_mask, selected_weights, 0.0)
    minimum_weight = jnp.min(
        jnp.where(output_mask, selected_weights, jnp.inf),
        initial=jnp.inf,
    )
    diagnostics = KernelHerdingDiagnostics(
        valid=output_valid,
        active_points=active_points,
        mmd=discrepancy,
        minimum_weight=minimum_weight,
        log_source_mass=log_source_mass,
        source_points=source_points,
        capacity=capacity,
        input_shape=input_shape,
        kernel_id=method.kernel.kernel_id,
    )
    return CoresetSelection(
        indices,
        log_weights_from_normalized(selected_weights, output_mask),
        output_mask,
        diagnostics,
        method="kernel-herding",
    )


__all__ = ["KernelHerding", "kernel_herd", "weighted_mmd"]
