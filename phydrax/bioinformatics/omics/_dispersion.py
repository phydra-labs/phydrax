#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._numerics import solve_weighted_least_squares
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._assay import CountAssay


DISPERSION_SUCCESS = 0
DISPERSION_INSUFFICIENT_SAMPLES = 1
DISPERSION_ALL_ZERO = 2
DISPERSION_BOUNDARY = 3
DISPERSION_RANK_DEFICIENT = 4
DISPERSION_NONFINITE = 5


def _dispersion_contract(name: str) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        name,
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "NB2 dispersion is represented on a bounded positive interval; "
            "shrinkage is performed in log dispersion."
        ),
        truncation_statement="No features are truncated; unusable features remain explicitly invalid.",
        capacity_semantics="Feature capacity is the fixed assay width.",
        assumptions=(
            "Within-feature normalized samples share a mean-dispersion relationship.",
        ),
        nondifferentiable_outputs=("valid", "status", "sample_count"),
    )


def _bounds(minimum: float, maximum: float, /) -> tuple[float, float]:
    lower = float(minimum)
    upper = float(maximum)
    if not (0.0 < lower < upper):
        raise ValueError("Dispersion bounds must satisfy 0 < minimum < maximum.")
    return lower, upper


class FeatureDispersionEstimate(StrictModule):
    """Feature-wise method-of-moments NB2 dispersion evidence."""

    mean: Array
    variance: Array
    raw_dispersion: Array
    sample_count: Array
    degrees_of_freedom: Array
    at_lower_bound: Array
    at_upper_bound: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    minimum: float = eqx.field(static=True)
    maximum: float = eqx.field(static=True)


class DispersionTrendResult(StrictModule):
    """Parametric alpha(mu) = intercept + reciprocal / mu trend."""

    coefficients: Array
    fitted_dispersion: Array
    feature_mask: Array
    rank: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    minimum: float = eqx.field(static=True)
    maximum: float = eqx.field(static=True)


class DispersionShrinkageResult(StrictModule):
    """Log-scale feature dispersions shrunk toward a mean trend."""

    dispersion: Array
    raw_log_dispersion: Array
    trend_log_dispersion: Array
    posterior_log_dispersion: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    prior_degrees_of_freedom: float = eqx.field(static=True)


def estimate_feature_dispersion(
    assay: CountAssay,
    /,
    *,
    size_factors: ArrayLike | None = None,
    minimum: float = 1.0e-8,
    maximum: float = 1.0e4,
) -> FeatureDispersionEstimate:
    """Estimate raw NB2 dispersion from observed normalized first two moments."""

    if not isinstance(assay, CountAssay):
        raise TypeError("assay must be a CountAssay.")
    lower, upper = _bounds(minimum, maximum)
    count_values, observed, _, _ = assay.dense_components()
    counts = count_values.astype(float)
    if size_factors is None:
        factors = jnp.ones((assay.num_samples,), dtype=counts.dtype)
    else:
        factors = jnp.asarray(size_factors, dtype=counts.dtype)
        if factors.shape != (assay.num_samples,):
            raise ValueError(f"size_factors must have shape ({assay.num_samples},).")
        factors = eqx.error_if(
            factors,
            jnp.any(~jnp.isfinite(factors) | (factors <= 0.0)),
            "size_factors must be finite and positive.",
        )
    normalized = counts / factors[:, None]
    sample_count = jnp.sum(observed, axis=0).astype(jnp.int32)
    denominator = jnp.maximum(sample_count, 1).astype(counts.dtype)
    mean = compensated_sum(jnp.where(observed, normalized, 0.0), axis=0) / denominator
    centered = jnp.where(observed, normalized - mean[None, :], 0.0)
    variance = compensated_sum(centered * centered, axis=0) / jnp.maximum(
        sample_count - 1, 1
    ).astype(counts.dtype)
    unbounded = (variance - mean) / jnp.maximum(mean * mean, lower)
    raw = jnp.clip(unbounded, lower, upper)
    enough = sample_count > 1
    nonzero = mean > 0.0
    finite = jnp.isfinite(mean) & jnp.isfinite(variance) & jnp.isfinite(raw)
    at_lower = unbounded <= lower
    at_upper = unbounded >= upper
    boundary = at_lower | at_upper
    valid = enough & nonzero & finite
    status = jnp.where(
        ~finite,
        DISPERSION_NONFINITE,
        jnp.where(
            ~enough,
            DISPERSION_INSUFFICIENT_SAMPLES,
            jnp.where(
                ~nonzero,
                DISPERSION_ALL_ZERO,
                jnp.where(boundary, DISPERSION_BOUNDARY, DISPERSION_SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = jnp.stack(
        (
            mean,
            variance,
            unbounded,
            sample_count.astype(counts.dtype),
        ),
        axis=1,
    )
    return FeatureDispersionEstimate(
        mean=mean,
        variance=variance,
        raw_dispersion=raw,
        sample_count=sample_count,
        degrees_of_freedom=jnp.maximum(sample_count - 1, 0),
        at_lower_bound=at_lower,
        at_upper_bound=at_upper,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_dispersion_contract("nb2-moment-dispersion"),
        minimum=lower,
        maximum=upper,
    )


def fit_dispersion_trend(
    estimates: FeatureDispersionEstimate,
    /,
    *,
    rcond: float | None = None,
) -> DispersionTrendResult:
    """Fit a positive parametric dispersion trend using native least squares."""

    if not isinstance(estimates, FeatureDispersionEstimate):
        raise TypeError("estimates must be a FeatureDispersionEstimate.")
    mean = estimates.mean
    raw = estimates.raw_dispersion
    feature_mask = estimates.valid & jnp.isfinite(mean) & jnp.isfinite(raw)
    predictor = jnp.stack(
        (jnp.ones_like(mean), 1.0 / jnp.maximum(mean, estimates.minimum)), axis=1
    )
    fit = solve_weighted_least_squares(
        predictor,
        raw,
        mask=feature_mask,
        weights=estimates.degrees_of_freedom.astype(mean.dtype),
        ridge=1.0e-8,
        rcond=rcond,
        min_samples=2,
        max_features=2,
    )
    coefficients = jnp.maximum(fit.raw_coefficients, 0.0)
    fitted = jnp.clip(
        predictor @ coefficients,
        estimates.minimum,
        estimates.maximum,
    )
    finite = jnp.all(jnp.isfinite(coefficients)) & jnp.all(jnp.isfinite(fitted))
    full_rank = fit.rank == 2
    enough = jnp.sum(feature_mask) >= 2
    valid = finite & full_rank & enough
    status = jnp.where(
        ~finite,
        DISPERSION_NONFINITE,
        jnp.where(
            ~enough,
            DISPERSION_INSUFFICIENT_SAMPLES,
            jnp.where(~full_rank, DISPERSION_RANK_DEFICIENT, DISPERSION_SUCCESS),
        ),
    ).astype(jnp.int32)
    evidence = jnp.stack(
        (
            jnp.sum(feature_mask).astype(mean.dtype),
            fit.rank.astype(mean.dtype),
            fit.condition_number.astype(mean.dtype),
        )
    )
    return DispersionTrendResult(
        coefficients=coefficients,
        fitted_dispersion=fitted,
        feature_mask=feature_mask,
        rank=fit.rank,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_dispersion_contract("parametric-dispersion-trend"),
        minimum=estimates.minimum,
        maximum=estimates.maximum,
    )


def shrink_dispersion(
    estimates: FeatureDispersionEstimate,
    trend: DispersionTrendResult,
    /,
    *,
    prior_degrees_of_freedom: float = 10.0,
) -> DispersionShrinkageResult:
    """Shrink raw feature dispersions toward the parametric trend in log space."""

    if not isinstance(estimates, FeatureDispersionEstimate):
        raise TypeError("estimates must be a FeatureDispersionEstimate.")
    if not isinstance(trend, DispersionTrendResult):
        raise TypeError("trend must be a DispersionTrendResult.")
    if estimates.raw_dispersion.shape != trend.fitted_dispersion.shape:
        raise ValueError("estimate and trend feature dimensions do not match.")
    prior_df = float(prior_degrees_of_freedom)
    if not jnp.isfinite(prior_df) or prior_df <= 0.0:
        raise ValueError("prior_degrees_of_freedom must be finite and positive.")
    raw_log = jnp.log(estimates.raw_dispersion)
    trend_log = jnp.log(trend.fitted_dispersion)
    feature_df = estimates.degrees_of_freedom.astype(raw_log.dtype)
    posterior_log = (feature_df * raw_log + prior_df * trend_log) / (
        feature_df + prior_df
    )
    dispersion = jnp.clip(jnp.exp(posterior_log), estimates.minimum, estimates.maximum)
    finite = jnp.isfinite(dispersion)
    valid = estimates.valid & trend.valid & finite
    status = jnp.where(
        ~finite,
        DISPERSION_NONFINITE,
        jnp.where(
            ~estimates.valid,
            estimates.status,
            jnp.where(~trend.valid, trend.status, DISPERSION_SUCCESS),
        ),
    ).astype(jnp.int32)
    evidence = jnp.stack((feature_df, jnp.full_like(feature_df, prior_df)), axis=1)
    return DispersionShrinkageResult(
        dispersion=dispersion,
        raw_log_dispersion=raw_log,
        trend_log_dispersion=trend_log,
        posterior_log_dispersion=posterior_log,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_dispersion_contract("log-dispersion-shrinkage"),
        prior_degrees_of_freedom=prior_df,
    )


def estimate_shrunk_dispersion(
    assay: CountAssay,
    /,
    *,
    size_factors: ArrayLike | None = None,
    minimum: float = 1.0e-8,
    maximum: float = 1.0e4,
    prior_degrees_of_freedom: float = 10.0,
    rcond: float | None = None,
) -> DispersionShrinkageResult:
    """Run moment estimation, trend fitting, and log-scale shrinkage."""

    estimates = estimate_feature_dispersion(
        assay,
        size_factors=size_factors,
        minimum=minimum,
        maximum=maximum,
    )
    trend = fit_dispersion_trend(estimates, rcond=rcond)
    return shrink_dispersion(
        estimates,
        trend,
        prior_degrees_of_freedom=prior_degrees_of_freedom,
    )


__all__ = [
    "DISPERSION_ALL_ZERO",
    "DISPERSION_BOUNDARY",
    "DISPERSION_INSUFFICIENT_SAMPLES",
    "DISPERSION_NONFINITE",
    "DISPERSION_RANK_DEFICIENT",
    "DISPERSION_SUCCESS",
    "DispersionShrinkageResult",
    "DispersionTrendResult",
    "FeatureDispersionEstimate",
    "estimate_feature_dispersion",
    "estimate_shrunk_dispersion",
    "fit_dispersion_trend",
    "shrink_dispersion",
]
