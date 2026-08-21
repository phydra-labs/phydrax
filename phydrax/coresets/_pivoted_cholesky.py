#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax.kernels import (
    AbstractPositiveDefiniteKernel,
    SquaredExponentialKernel,
)

from .._doc import DOC_KEY0
from .._measure_weights import log_weights_from_normalized
from .._strict import StrictModule
from ._types import CoresetSelection, PivotedCholeskyDiagnostics


class RandomizedPivotedCholesky(StrictModule):
    """Residual-diagonal randomized pivoting for kernel column selection."""

    kernel: AbstractPositiveDefiniteKernel
    num_points: int = eqx.field(static=True)

    def __init__(
        self,
        num_points: int,
        /,
        *,
        kernel: AbstractPositiveDefiniteKernel | None = None,
    ):
        count = int(num_points)
        if count <= 0:
            raise ValueError("num_points must be positive.")
        if kernel is not None and not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be an AbstractPositiveDefiniteKernel or None.")
        self.num_points = count
        self.kernel = SquaredExponentialKernel() if kernel is None else kernel


def randomized_pivoted_cholesky(
    points: Array,
    method: RandomizedPivotedCholesky,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> CoresetSelection:
    """Select source rows using randomized residual Cholesky pivots."""
    if not isinstance(method, RandomizedPivotedCholesky):
        raise TypeError("method must be a RandomizedPivotedCholesky.")
    values = jnp.asarray(points, dtype=float)
    expected_rank = method.kernel.input_ndim + 1
    if values.ndim != expected_rank:
        raise ValueError(
            "points must have one source axis followed by "
            f"{method.kernel.input_ndim} kernel input axes."
        )
    source_points = int(values.shape[0])
    if source_points < 1:
        raise ValueError("Pivoted Cholesky requires at least one source point.")
    if method.num_points > source_points:
        raise ValueError("Cannot select more inducing points than source points.")
    rows_valid = jnp.all(jnp.isfinite(values), axis=tuple(range(1, values.ndim)))
    input_valid = jnp.all(rows_valid)
    safe_points = jnp.nan_to_num(values)
    diagonal = method.kernel.diagonal(safe_points)
    diagonal = jnp.where(rows_valid, jnp.maximum(diagonal, 0.0), 0.0)
    initial_trace = jnp.sum(diagonal)
    capacity = method.num_points
    columns = jnp.zeros((source_points, capacity), dtype=safe_points.dtype)
    indices = jnp.zeros((capacity,), dtype=jnp.int32)
    output_mask = jnp.zeros((capacity,), dtype=bool)
    selected = jnp.zeros((source_points,), dtype=bool)
    tolerance = jnp.asarray(
        jnp.finfo(safe_points.dtype).eps * jnp.maximum(initial_trace, 1.0) * 32.0
    )

    def body(iteration, state):
        residual, factors, chosen, active, used = state
        eligible_residual = jnp.where(~used, jnp.maximum(residual, 0.0), 0.0)
        total = jnp.sum(eligible_residual)
        valid_choice = input_valid & (total > tolerance)
        fallback = jnp.full(
            (source_points,),
            1.0 / float(source_points),
            dtype=safe_points.dtype,
        )
        probabilities = jnp.where(
            total > 0.0,
            eligible_residual / total,
            fallback,
        )
        pivot = jnp.asarray(
            jr.choice(jr.fold_in(key, iteration), source_points, p=probabilities),
            dtype=jnp.int32,
        )
        kernel_column = method.kernel.matrix(safe_points, safe_points[pivot][None, ...])[
            :, 0
        ]
        previous = factors @ factors[pivot]
        pivot_residual = jnp.maximum(residual[pivot], tolerance)
        column = (kernel_column - previous) / jnp.sqrt(pivot_residual)
        column = jnp.where(valid_choice, column, 0.0)
        factors = factors.at[:, iteration].set(column)
        residual = jnp.maximum(residual - column * column, 0.0)
        residual = residual.at[pivot].set(jnp.where(valid_choice, 0.0, residual[pivot]))
        chosen = chosen.at[iteration].set(pivot)
        active = active.at[iteration].set(valid_choice)
        used = used.at[pivot].set(used[pivot] | valid_choice)
        return residual, factors, chosen, active, used

    residual, _, indices, output_mask, _ = jax.lax.fori_loop(
        0,
        capacity,
        body,
        (diagonal, columns, indices, output_mask, selected),
    )
    active_points = jnp.sum(output_mask, dtype=jnp.int32)
    residual_trace = jnp.sum(residual)
    explained = jnp.where(
        initial_trace > 0.0,
        jnp.clip(1.0 - residual_trace / initial_trace, 0.0, 1.0),
        0.0,
    )
    output_valid = input_valid & (active_points == capacity)
    output_mask = output_mask & output_valid
    equal_weight = jnp.where(
        active_points > 0,
        1.0 / active_points.astype(safe_points.dtype),
        0.0,
    )
    weights = jnp.where(output_mask, equal_weight, 0.0)
    diagnostics = PivotedCholeskyDiagnostics(
        valid=output_valid,
        active_points=active_points,
        initial_trace=initial_trace,
        residual_trace=residual_trace,
        explained_trace_fraction=explained,
        source_points=source_points,
        capacity=capacity,
        kernel_id=method.kernel.kernel_id,
    )
    return CoresetSelection(
        indices,
        log_weights_from_normalized(weights, output_mask),
        output_mask,
        diagnostics,
        method="randomized-pivoted-cholesky",
    )


__all__ = ["RandomizedPivotedCholesky", "randomized_pivoted_cholesky"]
