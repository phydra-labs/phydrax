#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule


class RateMixtureStatus(IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    NEGATIVE_RATE = 2
    INVALID_WEIGHTS = 3
    ZERO_MEAN_RATE = 4
    INVALID_INVARIANT_PROPORTION = 5


class RateMixtureEvidence(StrictModule):
    """Normalization and support evidence for a discrete site-rate law."""

    finite: Array
    nonnegative_rates: Array
    nonnegative_weights: Array
    weight_sum: Array
    mean_rate: Array
    normalized: Array
    invariant_weight: Array


class DiscreteRateMixture(StrictModule):
    """Finite site-rate distribution, including optional zero-rate mass."""

    rates: Array
    weights: Array
    valid: Array
    status: Array
    evidence: RateMixtureEvidence
    category_count: int = eqx.field(static=True)
    mixture_name: str = eqx.field(static=True)


def discrete_rate_mixture(
    rates: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    normalize_mean: bool = False,
    mixture_name: str = "discrete-rates",
) -> DiscreteRateMixture:
    """Construct a checked finite rate mixture without implicit truncation."""

    rate_values = jnp.asarray(rates)
    weight_values = jnp.asarray(weights, dtype=rate_values.dtype)
    if rate_values.ndim != 1 or rate_values.shape[0] == 0:
        raise ValueError("rates must be a non-empty rank-one array.")
    if weight_values.shape != rate_values.shape:
        raise ValueError("weights must have the same shape as rates.")
    if not jnp.issubdtype(rate_values.dtype, jnp.inexact):
        rate_values = rate_values.astype(jnp.float32)
        weight_values = weight_values.astype(jnp.float32)

    weight_sum = jnp.sum(weight_values)
    safe_weight_sum = jnp.where(
        jnp.isfinite(weight_sum) & (weight_sum > 0.0), weight_sum, 1.0
    )
    normalized_weights = weight_values / safe_weight_sum
    mean_rate = jnp.sum(normalized_weights * rate_values)
    safe_mean = jnp.where(jnp.isfinite(mean_rate) & (mean_rate > 0.0), mean_rate, 1.0)
    normalized_rates = rate_values / safe_mean if normalize_mean else rate_values
    output_mean = jnp.sum(normalized_weights * normalized_rates)

    finite = jnp.all(jnp.isfinite(rate_values)) & jnp.all(jnp.isfinite(weight_values))
    nonnegative_rates = jnp.all(rate_values >= 0.0)
    nonnegative_weights = jnp.all(weight_values >= 0.0)
    positive_weight_sum = weight_sum > 0.0
    positive_mean = mean_rate > 0.0
    valid = (
        finite
        & nonnegative_rates
        & nonnegative_weights
        & positive_weight_sum
        & (~jnp.asarray(normalize_mean) | positive_mean)
    )
    status = jnp.where(
        ~finite,
        int(RateMixtureStatus.NONFINITE),
        jnp.where(
            ~nonnegative_rates,
            int(RateMixtureStatus.NEGATIVE_RATE),
            jnp.where(
                ~nonnegative_weights | ~positive_weight_sum,
                int(RateMixtureStatus.INVALID_WEIGHTS),
                jnp.where(
                    jnp.asarray(normalize_mean) & ~positive_mean,
                    int(RateMixtureStatus.ZERO_MEAN_RATE),
                    int(RateMixtureStatus.SUCCESS),
                ),
            ),
        ),
    )
    evidence = RateMixtureEvidence(
        finite=finite,
        nonnegative_rates=nonnegative_rates,
        nonnegative_weights=nonnegative_weights,
        weight_sum=weight_sum,
        mean_rate=output_mean,
        normalized=jnp.abs(jnp.sum(normalized_weights) - 1.0)
        <= 64.0 * jnp.finfo(normalized_weights.dtype).eps,
        invariant_weight=jnp.sum(
            jnp.where(normalized_rates == 0.0, normalized_weights, 0.0)
        ),
    )
    return DiscreteRateMixture(
        rates=normalized_rates,
        weights=normalized_weights,
        valid=valid,
        status=jnp.asarray(status, dtype=jnp.int32),
        evidence=evidence,
        category_count=int(rate_values.shape[0]),
        mixture_name=str(mixture_name),
    )


def unit_rate_mixture(*, dtype: jnp.dtype = jnp.float32) -> DiscreteRateMixture:
    """Single exact unit-rate category."""

    return discrete_rate_mixture(
        jnp.ones((1,), dtype=dtype),
        jnp.ones((1,), dtype=dtype),
        mixture_name="unit-rate",
    )


def invariant_rate_mixture(
    invariant_proportion: ArrayLike,
    /,
    *,
    dtype: jnp.dtype | None = None,
) -> DiscreteRateMixture:
    """Invariant-plus-variable law with overall mean rate one."""

    proportion = jnp.asarray(invariant_proportion, dtype=dtype)
    if proportion.shape != ():
        raise ValueError("invariant_proportion must be scalar.")
    resolved_dtype = jnp.result_type(proportion, jnp.float32)
    variable_weight = 1.0 - proportion
    safe_variable_weight = jnp.where(variable_weight > 0.0, variable_weight, 1.0)
    mixture = discrete_rate_mixture(
        jnp.asarray([0.0, 1.0 / safe_variable_weight], dtype=resolved_dtype),
        jnp.asarray([proportion, variable_weight], dtype=resolved_dtype),
        mixture_name="invariant-plus-unit-mean-variable",
    )
    proportion_valid = (proportion >= 0.0) & (proportion < 1.0)
    status = jnp.where(
        proportion_valid,
        mixture.status,
        int(RateMixtureStatus.INVALID_INVARIANT_PROPORTION),
    )
    return eqx.tree_at(
        lambda value: (value.valid, value.status),
        mixture,
        (mixture.valid & proportion_valid, jnp.asarray(status, dtype=jnp.int32)),
    )


def with_invariant_sites(
    mixture: DiscreteRateMixture,
    invariant_proportion: ArrayLike,
    /,
) -> DiscreteRateMixture:
    """Add zero-rate mass while preserving the original overall mean rate."""

    if not isinstance(mixture, DiscreteRateMixture):
        raise TypeError("mixture must be a DiscreteRateMixture.")
    proportion = jnp.asarray(invariant_proportion, dtype=mixture.rates.dtype)
    if proportion.shape != ():
        raise ValueError("invariant_proportion must be scalar.")
    variable_weight = 1.0 - proportion
    safe_variable_weight = jnp.where(variable_weight > 0.0, variable_weight, 1.0)
    combined = discrete_rate_mixture(
        jnp.concatenate(
            (
                jnp.zeros((1,), dtype=mixture.rates.dtype),
                mixture.rates / safe_variable_weight,
            )
        ),
        jnp.concatenate(
            (
                proportion[None],
                variable_weight * mixture.weights,
            )
        ),
        mixture_name=f"invariant-plus-{mixture.mixture_name}",
    )
    proportion_valid = (proportion >= 0.0) & (proportion < 1.0)
    status = jnp.where(
        proportion_valid & mixture.valid,
        combined.status,
        int(RateMixtureStatus.INVALID_INVARIANT_PROPORTION),
    )
    return eqx.tree_at(
        lambda value: (value.valid, value.status),
        combined,
        (
            combined.valid & mixture.valid & proportion_valid,
            jnp.asarray(status, dtype=jnp.int32),
        ),
    )


RateMixture = DiscreteRateMixture


__all__ = [
    "DiscreteRateMixture",
    "RateMixture",
    "RateMixtureEvidence",
    "RateMixtureStatus",
    "discrete_rate_mixture",
    "invariant_rate_mixture",
    "unit_rate_mixture",
    "with_invariant_sites",
]
