#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule


class ChainLengthScalingPlan(StrictModule):
    minimum_chain_lengths: int = eqx.field(static=True)
    minimum_r_squared: float = eqx.field(static=True)
    minimum_exponent: float = eqx.field(static=True)
    maximum_exponent: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_chain_lengths: int = 3,
        minimum_r_squared: float = 0.95,
        exponent_interval: tuple[float, float] = (-math.inf, math.inf),
    ):
        if int(minimum_chain_lengths) < 3:
            raise ValueError("minimum_chain_lengths must be at least three.")
        if not 0.0 <= float(minimum_r_squared) <= 1.0:
            raise ValueError("minimum_r_squared must lie in [0, 1].")
        lower, upper = map(float, exponent_interval)
        if not lower < upper:
            raise ValueError("exponent_interval must be ordered.")
        self.minimum_chain_lengths = int(minimum_chain_lengths)
        self.minimum_r_squared = float(minimum_r_squared)
        self.minimum_exponent = lower
        self.maximum_exponent = upper
        self.plan_id = canonical_fingerprint(
            {
                "kind": "chain-length-scaling",
                "points": self.minimum_chain_lengths,
                "r2": self.minimum_r_squared,
                "interval": [
                    "negative_infinity" if math.isinf(lower) and lower < 0.0 else lower,
                    "infinity" if math.isinf(upper) else upper,
                ],
            }
        )


class ChainLengthScalingResult(StrictModule):
    exponent: Array
    exponent_standard_error: Array
    prefactor: Array
    r_squared: Array
    fitted_values: Array
    standardized_residuals: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def fit_chain_length_scaling(
    plan: ChainLengthScalingPlan,
    chain_lengths: ArrayLike,
    observable: ArrayLike,
    /,
    *,
    standard_error: ArrayLike | None = None,
) -> ChainLengthScalingResult:
    if not isinstance(plan, ChainLengthScalingPlan):
        raise TypeError("plan must be ChainLengthScalingPlan.")
    lengths_host = np.asarray(chain_lengths, dtype=np.float64)
    values_host = np.asarray(observable, dtype=np.float64)
    if lengths_host.ndim != 1 or values_host.shape != lengths_host.shape:
        raise ValueError("chain_lengths and observable must be matching rank-1 arrays.")
    if lengths_host.size < plan.minimum_chain_lengths:
        raise ValueError("Too few chain lengths for the requested scaling fit.")
    if np.any(~np.isfinite(lengths_host)) or np.any(lengths_host <= 0.0):
        raise ValueError("chain_lengths must be finite and positive.")
    if np.any(~np.isfinite(values_host)) or np.any(values_host <= 0.0):
        raise ValueError("observable values must be finite and positive.")
    if np.unique(lengths_host).size != lengths_host.size:
        raise ValueError("chain_lengths must be unique.")
    order = np.argsort(lengths_host)
    x_host = np.log(lengths_host[order])
    y_host = np.log(values_host[order])
    if standard_error is None:
        weights_host = np.ones_like(y_host)
    else:
        errors = np.asarray(standard_error, dtype=np.float64)
        if (
            errors.shape != values_host.shape
            or np.any(~np.isfinite(errors))
            or np.any(errors <= 0.0)
        ):
            raise ValueError(
                "standard_error must be finite, positive, and match observable."
            )
        log_errors = errors[order] / values_host[order]
        weights_host = 1.0 / (log_errors * log_errors)

    x = jnp.asarray(x_host)
    y = jnp.asarray(y_host)
    weights = jnp.asarray(weights_host)
    weight_sum = jnp.sum(weights)
    x_mean = jnp.sum(weights * x) / weight_sum
    y_mean = jnp.sum(weights * y) / weight_sum
    centered_x = x - x_mean
    centered_y = y - y_mean
    denominator = jnp.sum(weights * centered_x * centered_x)
    exponent = jnp.sum(weights * centered_x * centered_y) / denominator
    intercept = y_mean - exponent * x_mean
    fitted_log = intercept + exponent * x
    residual = y - fitted_log
    residual_sum = jnp.sum(weights * residual * residual)
    total_sum = jnp.sum(weights * centered_y * centered_y)
    r_squared = 1.0 - residual_sum / jnp.maximum(total_sum, jnp.finfo(y.dtype).tiny)
    degrees = lengths_host.size - 2
    variance = residual_sum / degrees
    exponent_error = jnp.sqrt(variance / denominator)
    standardized = residual / jnp.sqrt(
        jnp.maximum(variance / weights, jnp.finfo(y.dtype).tiny)
    )
    fitted_sorted = jnp.exp(fitted_log)
    inverse_order = np.argsort(order)
    fitted = fitted_sorted[jnp.asarray(inverse_order)]
    standardized = standardized[jnp.asarray(inverse_order)]
    prefactor = jnp.exp(intercept)
    finite = (
        jnp.isfinite(exponent)
        & jnp.isfinite(exponent_error)
        & jnp.isfinite(prefactor)
        & jnp.isfinite(r_squared)
        & jnp.all(jnp.isfinite(standardized))
    )
    successful = (
        finite
        & (r_squared >= plan.minimum_r_squared)
        & (exponent >= plan.minimum_exponent)
        & (exponent <= plan.maximum_exponent)
    )
    return ChainLengthScalingResult(
        exponent,
        exponent_error,
        prefactor,
        r_squared,
        fitted,
        standardized,
        finite,
        successful,
        plan.plan_id,
    )


__all__ = [
    "ChainLengthScalingPlan",
    "ChainLengthScalingResult",
    "fit_chain_length_scaling",
]
