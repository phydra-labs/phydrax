#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._contracts import ML_INFEASIBLE, ML_NONFINITE, ML_RANK_DEFICIENT, ML_SUCCESS


class LeastSquaresResult(StrictModule):
    """Weighted affine least-squares solution with rank diagnostics."""

    coefficients: Array
    intercept: Array
    prediction: Array
    residual_sum_squares: Array
    singular_values: Array
    rank: Array
    condition: Array
    valid: Array
    status: Array
    method: str = eqx.field(static=True)


def _solve_one(
    design: Array,
    target: Array,
    weights: Array,
    ridge: Array,
    fit_intercept: bool,
    regularize_intercept: bool,
    rcond: float,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    valid_weight = jnp.isfinite(weights) & (weights >= 0.0)
    active = valid_weight & (weights > 0.0)
    safe_weights = jnp.where(valid_weight, weights, 0.0)
    safe_design = jnp.where(active[:, None], design, 0)
    safe_target = jnp.where(active[:, None], target, 0)
    if fit_intercept:
        augmented_design = jnp.concatenate(
            (safe_design, jnp.ones((design.shape[0], 1), dtype=design.dtype)), axis=1
        )
    else:
        augmented_design = safe_design
    sqrt_weights = jnp.sqrt(safe_weights)
    weighted_design = sqrt_weights[:, None] * augmented_design
    weighted_target = sqrt_weights[:, None] * safe_target

    parameter_count = augmented_design.shape[1]
    penalty = jnp.ones((parameter_count,), dtype=weighted_design.real.dtype)
    if fit_intercept and not regularize_intercept:
        penalty = penalty.at[-1].set(0.0)
    regularizer = jnp.sqrt(jnp.maximum(ridge, 0.0)) * jnp.diag(penalty).astype(
        weighted_design.dtype
    )
    solve_design = jnp.concatenate((weighted_design, regularizer), axis=0)
    solve_target = jnp.concatenate(
        (
            weighted_target,
            jnp.zeros((parameter_count, target.shape[1]), dtype=target.dtype),
        ),
        axis=0,
    )
    u, singular_values, vh = jnp.linalg.svd(solve_design, full_matrices=False)
    largest = jnp.max(singular_values, initial=0.0)
    threshold = largest * float(rcond)
    retained = singular_values > threshold
    inverse = jnp.where(retained, 1.0 / singular_values, 0.0)
    parameters = jnp.conj(vh).T @ (inverse[:, None] * (jnp.conj(u).T @ solve_target))
    if fit_intercept:
        coefficients = parameters[:-1]
        intercept = parameters[-1]
    else:
        coefficients = parameters
        intercept = jnp.zeros((target.shape[1],), dtype=target.dtype)
    prediction = design @ coefficients + intercept
    residual = jnp.where(active[:, None], prediction - target, 0)
    rss = jnp.sum(safe_weights[:, None] * jnp.real(residual * jnp.conj(residual)), axis=0)
    rank = jnp.sum(retained, dtype=jnp.int32)
    smallest = jnp.min(jnp.where(retained, singular_values, jnp.inf), initial=jnp.inf)
    condition = jnp.where(
        smallest < jnp.inf,
        largest / jnp.maximum(smallest, jnp.finfo(float).tiny),
        jnp.inf,
    )
    finite_inputs = jnp.all(jnp.isfinite(weights)) & jnp.isfinite(ridge)
    feasible = jnp.all(weights >= 0.0) & (ridge >= 0.0)
    finite_solution = (
        jnp.all(jnp.isfinite(jnp.real(parameters)))
        & jnp.all(jnp.isfinite(jnp.imag(parameters)))
        & jnp.all(jnp.isfinite(rss))
    )
    finite = finite_inputs & finite_solution
    full_rank = rank == parameter_count
    valid = finite & feasible & full_rank
    status = jnp.where(
        ~finite,
        ML_NONFINITE,
        jnp.where(
            ~feasible,
            ML_INFEASIBLE,
            jnp.where(full_rank, ML_SUCCESS, ML_RANK_DEFICIENT),
        ),
    ).astype(jnp.int32)
    return (
        coefficients,
        intercept,
        prediction,
        rss,
        singular_values,
        rank,
        condition,
        valid,
        status,
    )


def solve_weighted_least_squares(
    design: ArrayLike,
    target: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    ridge: ArrayLike = 0.0,
    fit_intercept: bool = True,
    regularize_intercept: bool = False,
    rcond: float | None = None,
) -> LeastSquaresResult:
    """Solve batched weighted affine regression through an augmented SVD."""
    x = jnp.asarray(design)
    y = jnp.asarray(target)
    w = jnp.asarray(weights, dtype=float)
    if x.ndim < 2 or w.shape != x.shape[:-1]:
        raise ValueError("design and weights must end in (sample, feature) and sample.")
    sample_shape = x.shape[:-1]
    if y.shape[: len(sample_shape)] != sample_shape:
        raise ValueError("target must begin with the design sample shape.")
    target_shape = y.shape[len(sample_shape) :]
    y_flat = y.reshape(sample_shape + (-1,)) if target_shape else y[..., None]
    case_shape = x.shape[:-2]
    cases = 1
    for size in case_shape:
        cases *= int(size)
    x_cases = x.reshape((cases, x.shape[-2], x.shape[-1]))
    y_cases = y_flat.reshape((cases, y_flat.shape[-2], y_flat.shape[-1]))
    w_cases = w.reshape((cases, w.shape[-1]))
    ridge_ = jnp.broadcast_to(jnp.asarray(ridge, dtype=float), case_shape or ())
    ridge_cases = ridge_.reshape((cases,))
    cutoff = (
        max(x.shape[-2], x.shape[-1]) * jnp.finfo(x.real.dtype).eps
        if rcond is None
        else float(rcond)
    )
    outputs = jax.vmap(
        lambda a, b, c, d: _solve_one(
            a,
            b,
            c,
            d,
            bool(fit_intercept),
            bool(regularize_intercept),
            float(cutoff),
        )
    )(x_cases, y_cases, w_cases, ridge_cases)
    coefficients, intercept, prediction, rss, singular, rank, condition, valid, status = (
        outputs
    )
    output_shape = target_shape or ()
    coefficients = coefficients.reshape(case_shape + (x.shape[-1],) + output_shape)
    intercept = intercept.reshape(case_shape + output_shape)
    prediction = prediction.reshape(sample_shape + output_shape)
    rss = rss.reshape(case_shape + output_shape)
    return LeastSquaresResult(
        coefficients=coefficients,
        intercept=intercept,
        prediction=prediction,
        residual_sum_squares=rss,
        singular_values=singular.reshape(case_shape + (singular.shape[-1],)),
        rank=rank.reshape(case_shape),
        condition=condition.reshape(case_shape),
        valid=valid.reshape(case_shape),
        status=status.reshape(case_shape),
        method="augmented-svd",
    )


__all__ = ["LeastSquaresResult", "solve_weighted_least_squares"]
