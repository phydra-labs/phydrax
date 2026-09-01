#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

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


NORMALIZATION_SUCCESS = 0
NORMALIZATION_ZERO_DEPTH = 1
NORMALIZATION_INSUFFICIENT_FEATURES = 2


def _normalization_contract(name: str) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        name,
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement="Size factors are centered to geometric mean one.",
        truncation_statement="All declared assay features are considered; none are truncated.",
        capacity_semantics="Work is bounded by the fixed assay sample-feature shape.",
        assumptions=("Observed counts are nonnegative integer measurements.",),
        nondifferentiable_outputs=("valid", "status", "feature_valid"),
    )


class CountNormalizationResult(StrictModule):
    """Normalized counts, offsets, and explicit usable-feature evidence."""

    size_factors: Array
    log_offsets: Array
    normalized_counts: Array
    observed_mask: Array
    feature_valid: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract


def _center_positive(values: Array, valid: Array, /) -> Array:
    safe_log = jnp.where(valid, jnp.log(jnp.maximum(values, 1.0e-30)), 0.0)
    count = jnp.sum(valid)
    center = jnp.exp(compensated_sum(safe_log) / jnp.maximum(count, 1))
    return jnp.where(valid, values / jnp.maximum(center, 1.0e-30), 1.0)


def library_size_normalization(assay: CountAssay, /) -> CountNormalizationResult:
    """Normalize by observed library depth without treating missing cells as zero."""

    if not isinstance(assay, CountAssay):
        raise TypeError("assay must be a CountAssay.")
    count_values, observed, _, _ = assay.dense_components()
    counts = count_values.astype(float)
    totals = compensated_sum(jnp.where(observed, counts, 0.0), axis=1)
    valid = totals > 0.0
    factors = _center_positive(totals, valid)
    normalized = jnp.where(observed, counts / factors[:, None], 0.0)
    feature_valid = jnp.any(observed, axis=0)
    status = jnp.where(valid, NORMALIZATION_SUCCESS, NORMALIZATION_ZERO_DEPTH).astype(
        jnp.int32
    )
    evidence = jnp.stack((totals, jnp.sum(observed, axis=1).astype(counts.dtype)), axis=1)
    return CountNormalizationResult(
        size_factors=factors,
        log_offsets=jnp.log(factors),
        normalized_counts=normalized,
        observed_mask=observed,
        feature_valid=feature_valid,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_normalization_contract("library-size-normalization"),
    )


def median_ratio_normalization(
    assay: CountAssay,
    /,
    *,
    minimum_positive_samples: int = 2,
) -> CountNormalizationResult:
    """Composition-robust positive-count median-ratio normalization.

    Feature reference abundances use the geometric mean over positive observed
    counts. Ratios from biological zeros, structural absences, and missing cells
    are excluded rather than conflated.
    """

    if not isinstance(assay, CountAssay):
        raise TypeError("assay must be a CountAssay.")
    minimum = int(minimum_positive_samples)
    if minimum < 1:
        raise ValueError("minimum_positive_samples must be positive.")
    count_values, observed, _, _ = assay.dense_components()
    counts = count_values.astype(float)
    positive = observed & (counts > 0.0)
    positive_count = jnp.sum(positive, axis=0)
    feature_valid = positive_count >= minimum
    log_counts = jnp.where(positive, jnp.log(jnp.maximum(counts, 1.0)), 0.0)
    geometric_mean = jnp.exp(
        compensated_sum(log_counts, axis=0) / jnp.maximum(positive_count, 1)
    )
    ratio_valid = positive & feature_valid[None, :]
    ratios = jnp.where(
        ratio_valid,
        counts / jnp.maximum(geometric_mean[None, :], 1.0e-30),
        jnp.nan,
    )
    raw_factors = jnp.nanmedian(ratios, axis=1)
    used = jnp.sum(ratio_valid, axis=1)
    valid = jnp.isfinite(raw_factors) & (raw_factors > 0.0) & (used > 0)
    factors = _center_positive(jnp.where(valid, raw_factors, 1.0), valid)
    normalized = jnp.where(observed, counts / factors[:, None], 0.0)
    status = jnp.where(
        valid, NORMALIZATION_SUCCESS, NORMALIZATION_INSUFFICIENT_FEATURES
    ).astype(jnp.int32)
    library_depth = compensated_sum(jnp.where(observed, counts, 0.0), axis=1)
    evidence = jnp.stack((library_depth, used.astype(counts.dtype), raw_factors), axis=1)
    return CountNormalizationResult(
        size_factors=factors,
        log_offsets=jnp.log(factors),
        normalized_counts=normalized,
        observed_mask=observed,
        feature_valid=feature_valid,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_normalization_contract("median-ratio-normalization"),
    )


__all__ = [
    "CountNormalizationResult",
    "NORMALIZATION_INSUFFICIENT_FEATURES",
    "NORMALIZATION_SUCCESS",
    "NORMALIZATION_ZERO_DEPTH",
    "library_size_normalization",
    "median_ratio_normalization",
]
