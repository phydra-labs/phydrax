#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    RankPolicy,
    solve,
)


LeastSquaresStatus: TypeAlias = Literal[0, 1, 2, 3]

LEAST_SQUARES_SUCCESS = 0
LEAST_SQUARES_INSUFFICIENT_SAMPLES = 1
LEAST_SQUARES_RANK_DEFICIENT = 2
LEAST_SQUARES_NONFINITE = 3


class NormalizedLeastSquaresDesign(StrictModule):
    """Weighted, masked design normalization with explicit rank diagnostics."""

    values: Array
    valid_rows: Array
    weights: Array
    offset: Array
    scale: Array
    singular_values: Array
    sample_count: Array
    weight_sum: Array
    rank: Array
    condition_number: Array
    num_samples: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    scaled: bool = eqx.field(static=True)


class WeightedLeastSquaresResult(StrictModule):
    """Diagnosed weighted least-squares solution in normalized and raw coordinates."""

    coefficients: Array
    raw_coefficients: Array
    intercept: Array
    prediction: Array
    residual: Array
    valid_rows: Array
    singular_values: Array
    sample_count: Array
    weight_sum: Array
    rank: Array
    condition_number: Array
    normal_equation_error: Array
    valid: Array
    status: Array
    ridge: float = eqx.field(static=True)
    rcond: float = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)


def _real_dtype(dtype) -> jnp.dtype:
    return jnp.empty((), dtype=dtype).real.dtype


def _rcond(value: float | None, dtype, rows: int, columns: int, /) -> float:
    if value is None:
        return float(max(rows, columns) * jnp.finfo(_real_dtype(dtype)).eps)
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError("rcond must be finite and nonnegative or None.")
    return resolved


def _ridge(value: float, /) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError("ridge must be finite and nonnegative.")
    return resolved


def _mask(value: ArrayLike | None, size: int, /) -> Array:
    if value is None:
        return jnp.ones((size,), dtype=bool)
    result = jnp.asarray(value, dtype=bool)
    if result.shape != (size,):
        raise ValueError(f"mask must have shape ({size},); got {result.shape}.")
    return result


def _weights(value: ArrayLike | None, size: int, dtype, /) -> Array:
    if value is None:
        return jnp.ones((size,), dtype=_real_dtype(dtype))
    result = jnp.asarray(value)
    if result.shape != (size,):
        raise ValueError(f"weights must have shape ({size},); got {result.shape}.")
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError("weights must be real-valued.")
    return result.astype(_real_dtype(dtype))


class _PreparedLeastSquaresDesign(StrictModule):
    """Validated weighted design before rank diagnostics or factorization."""

    values: Array
    valid_rows: Array
    weights: Array
    offset: Array
    scale: Array
    sample_count: Array
    weight_sum: Array
    num_samples: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    scaled: bool = eqx.field(static=True)


def _prepare_least_squares_design(
    design: ArrayLike,
    /,
    *,
    mask: ArrayLike | None,
    weights: ArrayLike | None,
    center: bool,
    scale: bool,
    max_features: int | None,
) -> _PreparedLeastSquaresDesign:
    values = jnp.asarray(design)
    if values.ndim != 2:
        raise ValueError("design must have shape (samples, features).")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)
    samples, features = (int(size) for size in values.shape)
    if samples < 1 or features < 1:
        raise ValueError("design must contain at least one sample and one feature.")
    if max_features is not None:
        maximum = int(max_features)
        if maximum < 1:
            raise ValueError("max_features must be positive or None.")
        if features > maximum:
            raise ValueError(
                f"design has {features} features; max_features is {maximum}."
            )

    requested = _mask(mask, samples)
    raw_weights = _weights(weights, samples, values.dtype)
    finite_design = jnp.all(jnp.isfinite(values), axis=-1)
    finite_weights = jnp.isfinite(raw_weights) & (raw_weights >= 0.0)
    valid_rows = requested & finite_design & finite_weights & (raw_weights > 0.0)
    safe_weights = jnp.where(valid_rows, raw_weights, 0.0)
    weight_sum = jnp.sum(safe_weights)
    denominator = jnp.maximum(weight_sum, jnp.asarray(1.0, dtype=safe_weights.dtype))
    safe_values = jnp.where(valid_rows[:, None], values, jnp.zeros((), values.dtype))
    weighted_mean = jnp.sum(safe_weights[:, None] * safe_values, axis=0) / denominator
    around_mean = values - weighted_mean
    safe_around_mean = jnp.where(
        valid_rows[:, None], around_mean, jnp.zeros((), values.dtype)
    )
    variance = (
        jnp.sum(safe_weights[:, None] * jnp.abs(safe_around_mean) ** 2, axis=0)
        / denominator
    )
    threshold = jnp.finfo(_real_dtype(values.dtype)).eps
    varying = variance > threshold
    offset = (
        jnp.where(varying, weighted_mean, 0.0)
        if bool(center)
        else jnp.zeros_like(weighted_mean)
    )
    centered_values = values - offset
    safe_centered = jnp.where(
        valid_rows[:, None], centered_values, jnp.zeros((), values.dtype)
    )
    second_moment = (
        jnp.sum(safe_weights[:, None] * jnp.abs(safe_centered) ** 2, axis=0) / denominator
    )
    resolved_scale = jnp.where(second_moment > threshold, jnp.sqrt(second_moment), 1.0)
    scales = resolved_scale if bool(scale) else jnp.ones_like(resolved_scale)
    return _PreparedLeastSquaresDesign(
        values=centered_values / scales,
        valid_rows=valid_rows,
        weights=safe_weights,
        offset=offset,
        scale=scales,
        sample_count=jnp.sum(valid_rows).astype(jnp.int32),
        weight_sum=weight_sum,
        num_samples=samples,
        num_features=features,
        centered=bool(center),
        scaled=bool(scale),
    )


def _diagnose_prepared_design(
    design: _PreparedLeastSquaresDesign,
    /,
    *,
    rcond: float | None,
) -> NormalizedLeastSquaresDesign:
    denominator = jnp.maximum(
        design.weight_sum,
        jnp.asarray(1.0, dtype=design.weights.dtype),
    )
    weighted = jnp.sqrt(design.weights / denominator)[:, None] * jnp.where(
        design.valid_rows[:, None],
        design.values,
        jnp.zeros((), design.values.dtype),
    )
    singular_values = jnp.linalg.svd(weighted, compute_uv=False)
    largest = jnp.max(singular_values, initial=0.0)
    tolerance = largest * _rcond(
        rcond,
        design.values.dtype,
        design.num_samples,
        design.num_features,
    )
    retained = singular_values > tolerance
    rank = jnp.sum(retained).astype(jnp.int32)
    smallest = jnp.min(
        jnp.where(retained, singular_values, jnp.inf),
        initial=jnp.inf,
    )
    condition = jnp.where(
        rank == design.num_features,
        largest / smallest,
        jnp.inf,
    )
    return NormalizedLeastSquaresDesign(
        values=design.values,
        valid_rows=design.valid_rows,
        weights=design.weights,
        offset=design.offset,
        scale=design.scale,
        singular_values=singular_values,
        sample_count=design.sample_count,
        weight_sum=design.weight_sum,
        rank=rank,
        condition_number=condition,
        num_samples=design.num_samples,
        num_features=design.num_features,
        centered=design.centered,
        scaled=design.scaled,
    )


def normalize_least_squares_design(
    design: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    center: bool = False,
    scale: bool = False,
    rcond: float | None = None,
    max_features: int | None = None,
) -> NormalizedLeastSquaresDesign:
    """Normalize one rank-two design with explicit standalone diagnostics."""
    prepared = _prepare_least_squares_design(
        design,
        mask=mask,
        weights=weights,
        center=center,
        scale=scale,
        max_features=max_features,
    )
    return _diagnose_prepared_design(prepared, rcond=rcond)


def _solve_prepared_least_squares(
    design: _PreparedLeastSquaresDesign,
    target: ArrayLike,
    /,
    *,
    mask: ArrayLike | None,
    ridge: float,
    rcond: float | None,
    min_samples: int | None,
    feature_mask: ArrayLike | None,
) -> WeightedLeastSquaresResult:
    response = jnp.asarray(target)
    if response.ndim < 1 or int(response.shape[0]) != design.num_samples:
        raise ValueError(
            "target must have one leading entry per normalized design sample."
        )
    if not jnp.issubdtype(response.dtype, jnp.inexact):
        response = response.astype(float)
    dtype = jnp.result_type(design.values, response)
    matrix = design.values.astype(dtype)
    response = response.astype(dtype)
    output_shape = tuple(int(size) for size in response.shape[1:])
    flat_response = response.reshape((design.num_samples, -1))
    requested = _mask(mask, design.num_samples)
    finite_target = jnp.all(jnp.isfinite(flat_response), axis=-1)
    valid_rows = design.valid_rows & requested & finite_target
    weights = jnp.where(valid_rows, design.weights, 0.0)
    weight_sum = jnp.sum(weights)
    denominator = jnp.maximum(weight_sum, jnp.asarray(1.0, dtype=weights.dtype))

    if feature_mask is None:
        active = jnp.ones((design.num_features,), dtype=bool)
    else:
        active = jnp.asarray(feature_mask, dtype=bool)
        if active.shape != (design.num_features,):
            raise ValueError(
                f"feature_mask must have shape ({design.num_features},); got "
                f"{active.shape}."
            )
    active_count = jnp.sum(active).astype(jnp.int32)
    safe_matrix = jnp.where(valid_rows[:, None], matrix, jnp.zeros((), dtype))
    safe_response = jnp.where(valid_rows[:, None], flat_response, jnp.zeros((), dtype))
    root_weights = jnp.sqrt(weights / denominator)
    weighted_matrix = root_weights[:, None] * safe_matrix * active[None, :]
    weighted_response = root_weights[:, None] * safe_response

    resolved_rcond = _rcond(rcond, dtype, design.num_samples, design.num_features)
    ridge_value = _ridge(ridge)
    linear_result = solve(
        LeastSquaresProblem(DenseLinearOperator(weighted_matrix)),
        weighted_response,
        policy=LinearSolvePolicy(
            DenseSVD(damping=ridge_value**0.5),
            rank=RankPolicy(relative_cutoff=resolved_rcond),
        ),
    )
    singular_values = linear_result.diagnostics.singular_values
    assert singular_values is not None
    rank = jnp.asarray(linear_result.diagnostics.rank).reshape(-1)[0]
    condition_estimate = jnp.asarray(
        linear_result.diagnostics.condition_estimate
    ).reshape(-1)[0]
    coefficients_flat = linear_result.value * active[:, None]
    prediction_flat = matrix @ coefficients_flat
    residual_flat = flat_response - prediction_flat
    weighted_residual = jnp.where(
        valid_rows[:, None], residual_flat, jnp.zeros((), dtype)
    )
    normal_moment = (
        jnp.swapaxes(jnp.conj(safe_matrix), -1, -2)
        @ (weights[:, None] * weighted_residual)
        / denominator
        - ridge_value * coefficients_flat
    )
    normal_moment = normal_moment * active[:, None]
    normal_error = jnp.max(jnp.abs(normal_moment), initial=0.0)
    condition = jnp.where(
        rank == active_count,
        condition_estimate,
        jnp.inf,
    )
    sample_count = jnp.sum(valid_rows).astype(jnp.int32)
    required = design.num_features if min_samples is None else int(min_samples)
    if required < 1:
        raise ValueError("min_samples must be positive or None.")
    finite = (
        jnp.all(jnp.isfinite(coefficients_flat))
        & jnp.isfinite(normal_error)
        & jnp.isfinite(weight_sum)
        & jnp.all(linear_result.diagnostics.finite)
    )
    enough = sample_count >= required
    full_rank = rank == active_count
    valid = enough & (active_count > 0) & finite & ((ridge_value > 0.0) | full_rank)
    status = jnp.where(
        ~finite,
        LEAST_SQUARES_NONFINITE,
        jnp.where(
            ~enough,
            LEAST_SQUARES_INSUFFICIENT_SAMPLES,
            jnp.where(
                (active_count == 0) | ((ridge_value == 0.0) & ~full_rank),
                LEAST_SQUARES_RANK_DEFICIENT,
                LEAST_SQUARES_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)

    raw_coefficients_flat = coefficients_flat / design.scale[:, None]
    intercept_flat = -jnp.sum(design.offset[:, None] * raw_coefficients_flat, axis=0)
    coefficient_shape = (design.num_features,) + output_shape
    coefficients = coefficients_flat.reshape(coefficient_shape)
    raw_coefficients = raw_coefficients_flat.reshape(coefficient_shape)
    intercept = intercept_flat.reshape(output_shape)
    return WeightedLeastSquaresResult(
        coefficients=coefficients,
        raw_coefficients=raw_coefficients,
        intercept=intercept,
        prediction=prediction_flat.reshape(response.shape),
        residual=residual_flat.reshape(response.shape),
        valid_rows=valid_rows,
        singular_values=singular_values,
        sample_count=sample_count,
        weight_sum=weight_sum,
        rank=rank,
        condition_number=condition,
        normal_equation_error=normal_error,
        valid=valid,
        status=status,
        ridge=ridge_value,
        rcond=resolved_rcond,
        output_shape=output_shape,
    )


def solve_normalized_least_squares(
    design: NormalizedLeastSquaresDesign,
    target: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
    ridge: float = 0.0,
    rcond: float | None = None,
    min_samples: int | None = None,
    feature_mask: ArrayLike | None = None,
) -> WeightedLeastSquaresResult:
    """Solve a normalized weighted least-squares problem by one economy SVD."""
    if not isinstance(design, NormalizedLeastSquaresDesign):
        raise TypeError("design must be a NormalizedLeastSquaresDesign.")
    prepared = _PreparedLeastSquaresDesign(
        values=design.values,
        valid_rows=design.valid_rows,
        weights=design.weights,
        offset=design.offset,
        scale=design.scale,
        sample_count=design.sample_count,
        weight_sum=design.weight_sum,
        num_samples=design.num_samples,
        num_features=design.num_features,
        centered=design.centered,
        scaled=design.scaled,
    )
    return _solve_prepared_least_squares(
        prepared,
        target,
        mask=mask,
        ridge=ridge,
        rcond=rcond,
        min_samples=min_samples,
        feature_mask=feature_mask,
    )


def solve_weighted_least_squares(
    design: ArrayLike,
    target: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    center: bool = False,
    scale: bool = False,
    ridge: float = 0.0,
    rcond: float | None = None,
    min_samples: int | None = None,
    feature_mask: ArrayLike | None = None,
    max_features: int | None = None,
) -> WeightedLeastSquaresResult:
    """Prepare and solve one weighted least-squares problem with one SVD."""
    prepared = _prepare_least_squares_design(
        design,
        mask=mask,
        weights=weights,
        center=center,
        scale=scale,
        max_features=max_features,
    )
    return _solve_prepared_least_squares(
        prepared,
        target,
        mask=None,
        ridge=ridge,
        rcond=rcond,
        min_samples=min_samples,
        feature_mask=feature_mask,
    )


__all__ = [
    "LEAST_SQUARES_INSUFFICIENT_SAMPLES",
    "LEAST_SQUARES_NONFINITE",
    "LEAST_SQUARES_RANK_DEFICIENT",
    "LEAST_SQUARES_SUCCESS",
    "LeastSquaresStatus",
    "NormalizedLeastSquaresDesign",
    "WeightedLeastSquaresResult",
    "normalize_least_squares_design",
    "solve_normalized_least_squares",
    "solve_weighted_least_squares",
]
