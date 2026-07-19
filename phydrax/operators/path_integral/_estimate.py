#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import jax.scipy.special as jsp
from jaxtyping import Array


def _estimate_positive_log_weights(
    log_weights: Array,
    /,
    *,
    scale: Array,
) -> PathIntegralEstimate:
    values = jnp.asarray(log_weights, dtype=float)
    if values.ndim < 1:
        raise ValueError("log_weights must have a trailing path axis.")
    count = int(values.shape[-1])
    if count < 1:
        raise ValueError("log_weights must contain at least one path.")

    log_sum = jsp.logsumexp(values, axis=-1)
    log_sum_sq = jsp.logsumexp(2.0 * values, axis=-1)
    return _estimate_positive_log_sums(
        log_sum,
        log_sum_sq,
        count=count,
        scale=scale,
    )


def _estimate_positive_log_sums(
    log_sum: Array,
    log_sum_sq: Array,
    /,
    *,
    count: int,
    scale: Array,
) -> PathIntegralEstimate:
    if int(count) < 1:
        raise ValueError("count must be at least one.")
    log_n = jnp.log(jnp.asarray(float(count)))
    log_mean = log_sum - log_n
    log_second_moment = log_sum_sq - log_n
    scale_arr = jnp.asarray(scale)
    log_abs_scale = jnp.log(jnp.abs(scale_arr))
    value = jnp.sign(scale_arr) * jnp.exp(log_abs_scale + log_mean)
    if int(count) == 1:
        standard_error = jnp.full_like(value, jnp.nan)
    else:
        log_mean_sq_ratio = jnp.minimum(
            2.0 * log_mean - log_second_moment,
            0.0,
        )
        variance_fraction = -jnp.expm1(log_mean_sq_ratio)
        log_population_variance = log_second_moment + jnp.log(variance_fraction)
        standard_error = jnp.exp(
            log_abs_scale
            + 0.5 * (log_population_variance - jnp.log(jnp.asarray(float(count - 1))))
        )
    ess = jnp.exp(jnp.minimum(2.0 * log_sum - log_sum_sq, log_n))
    return PathIntegralEstimate(
        value=value,
        standard_error=standard_error,
        effective_sample_size=ess,
        log_mean_weight=log_mean,
        num_paths=int(count),
    )


class PathIntegralEstimate(NamedTuple):
    """A path-estimator value together with sampling diagnostics."""

    value: Array
    standard_error: Array
    effective_sample_size: Array
    log_mean_weight: Array
    num_paths: int


__all__ = ["PathIntegralEstimate"]
