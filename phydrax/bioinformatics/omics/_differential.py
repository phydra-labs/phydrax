#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, gammaincc
from jaxtyping import Array

from ..._numerics import solve_weighted_least_squares
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._count_models import NegativeBinomialGLMFit
from ._design import DesignContrast


TEST_SUCCESS = 0
TEST_NOT_ESTIMABLE = 1
TEST_INVALID_FIT = 2
TEST_NONFINITE = 3
TEST_NOT_NESTED = 4
TEST_NO_DF_GAIN = 5


def _test_contract(name: str) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        name,
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.PROBABILISTIC,
        conditioning_statement="Inference uses each feature's observed-row rank and fitted covariance.",
        truncation_statement="All features are returned; non-estimable tests remain explicitly invalid.",
        capacity_semantics="Test capacity equals the fixed feature count of the fitted model.",
        assumptions=(
            "Wald statistics use their asymptotic standard-normal reference law.",
            "Likelihood-ratio statistics use their asymptotic chi-square reference law.",
        ),
        nondifferentiable_outputs=(
            "p_value",
            "estimable",
            "degrees_of_freedom",
            "valid",
            "status",
        ),
    )


class WaldTestResult(StrictModule):
    """Feature-wise linear-contrast Wald inference."""

    effect: Array
    log2_fold_change: Array
    standard_error: Array
    statistic: Array
    p_value: Array
    estimable: Array
    estimability_error: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    contrast: DesignContrast


class LikelihoodRatioTestResult(StrictModule):
    """Feature-wise nested-model likelihood-ratio inference."""

    log_likelihood_full: Array
    log_likelihood_reduced: Array
    statistic: Array
    degrees_of_freedom: Array
    p_value: Array
    nested: Array
    nesting_error: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract


def wald_test(
    fit: NegativeBinomialGLMFit,
    contrast: DesignContrast,
    /,
    *,
    estimability_tolerance: float = 1.0e-7,
) -> WaldTestResult:
    """Test one estimable coefficient contrast for every fitted feature."""

    if not isinstance(fit, NegativeBinomialGLMFit):
        raise TypeError("fit must be a NegativeBinomialGLMFit.")
    if not isinstance(contrast, DesignContrast):
        raise TypeError("contrast must be a DesignContrast.")
    if contrast.num_coefficients != fit.num_coefficients:
        raise ValueError("contrast width does not match the fitted design.")
    tolerance = float(estimability_tolerance)
    if tolerance < 0.0:
        raise ValueError("estimability_tolerance must be nonnegative.")
    weights = contrast.weights.astype(fit.coefficients.dtype)
    projected = fit.coefficient_projection @ weights
    residual = weights[None, :] - projected
    estimability_error = jnp.max(jnp.abs(residual), axis=1)
    scale = jnp.maximum(jnp.max(jnp.abs(weights), initial=0.0), 1.0)
    estimable = contrast.estimable & (estimability_error <= tolerance * scale)
    effect = fit.coefficients @ weights
    safe_covariance = jnp.where(fit.valid[:, None, None], fit.covariance, 0.0)
    contrast_variance = jnp.sum(weights[None, :] * (safe_covariance @ weights), axis=1)
    standard_error = jnp.sqrt(jnp.maximum(contrast_variance, 0.0))
    finite = jnp.isfinite(effect) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    valid = fit.valid & estimable & finite
    statistic = jnp.where(valid, effect / standard_error, jnp.nan)
    p_value = jnp.where(
        valid,
        erfc(jnp.abs(statistic) / jnp.sqrt(jnp.asarray(2.0, statistic.dtype))),
        jnp.nan,
    )
    status = jnp.where(
        ~fit.valid,
        TEST_INVALID_FIT,
        jnp.where(
            ~estimable,
            TEST_NOT_ESTIMABLE,
            jnp.where(~finite, TEST_NONFINITE, TEST_SUCCESS),
        ),
    ).astype(jnp.int32)
    evidence = jnp.stack(
        (
            fit.rank.astype(effect.dtype),
            fit.residual_degrees_of_freedom.astype(effect.dtype),
            estimability_error,
            contrast_variance,
        ),
        axis=1,
    )
    return WaldTestResult(
        effect=effect,
        log2_fold_change=effect / jnp.log(jnp.asarray(2.0, effect.dtype)),
        standard_error=standard_error,
        statistic=statistic,
        p_value=jax.lax.stop_gradient(p_value),
        estimable=jax.lax.stop_gradient(estimable),
        estimability_error=estimability_error,
        valid=jax.lax.stop_gradient(valid),
        status=jax.lax.stop_gradient(status),
        evidence=evidence,
        method_contract=_test_contract("negative-binomial-wald-test"),
        contrast=contrast,
    )


def likelihood_ratio_test(
    full: NegativeBinomialGLMFit,
    reduced: NegativeBinomialGLMFit,
    /,
    *,
    nesting_tolerance: float = 1.0e-7,
    rcond: float | None = None,
) -> LikelihoodRatioTestResult:
    """Compare explicitly nested NB2 GLMs using feature-specific rank gain."""

    if not isinstance(full, NegativeBinomialGLMFit) or not isinstance(
        reduced, NegativeBinomialGLMFit
    ):
        raise TypeError("full and reduced must be NegativeBinomialGLMFit values.")
    if (
        full.num_samples != reduced.num_samples
        or full.num_features != reduced.num_features
    ):
        raise ValueError("full and reduced fits must cover the same assay shape.")
    if full.likelihood_data_id != reduced.likelihood_data_id:
        raise ValueError(
            "full and reduced fits must use identical counts, observed rows, "
            "offsets, and dispersions."
        )
    tolerance = float(nesting_tolerance)
    if tolerance < 0.0:
        raise ValueError("nesting_tolerance must be nonnegative.")
    nesting_errors: list[Array] = []
    nested_values: list[Array] = []
    for feature in range(full.num_features):
        included = full.observed_mask[:, feature] & reduced.observed_mask[:, feature]
        projection = solve_weighted_least_squares(
            full.design_matrix,
            reduced.design_matrix,
            mask=included,
            rcond=rcond,
            min_samples=1,
            max_features=full.num_coefficients,
        )
        residual = jnp.where(
            included[:, None],
            reduced.design_matrix - projection.prediction,
            0.0,
        )
        error = jnp.max(jnp.abs(residual), initial=0.0)
        target_scale = jnp.maximum(
            jnp.max(jnp.abs(reduced.design_matrix), initial=0.0), 1.0
        )
        nesting_errors.append(error)
        nested_values.append(error <= tolerance * target_scale)
    nesting_error = jnp.stack(nesting_errors)
    nested = jnp.stack(nested_values)
    degrees_of_freedom = full.rank - reduced.rank
    gain = degrees_of_freedom > 0
    statistic = jnp.maximum(2.0 * (full.log_likelihood - reduced.log_likelihood), 0.0)
    finite = jnp.isfinite(statistic)
    valid = full.valid & reduced.valid & nested & gain & finite
    safe_df = jnp.maximum(degrees_of_freedom, 1).astype(statistic.dtype)
    p_value = jnp.where(
        valid,
        gammaincc(0.5 * safe_df, 0.5 * statistic),
        jnp.nan,
    )
    status = jnp.where(
        ~full.valid | ~reduced.valid,
        TEST_INVALID_FIT,
        jnp.where(
            ~nested,
            TEST_NOT_NESTED,
            jnp.where(
                ~gain,
                TEST_NO_DF_GAIN,
                jnp.where(~finite, TEST_NONFINITE, TEST_SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = jnp.stack(
        (
            full.rank.astype(statistic.dtype),
            reduced.rank.astype(statistic.dtype),
            degrees_of_freedom.astype(statistic.dtype),
            nesting_error,
        ),
        axis=1,
    )
    return LikelihoodRatioTestResult(
        log_likelihood_full=full.log_likelihood,
        log_likelihood_reduced=reduced.log_likelihood,
        statistic=statistic,
        degrees_of_freedom=jax.lax.stop_gradient(degrees_of_freedom),
        p_value=jax.lax.stop_gradient(p_value),
        nested=jax.lax.stop_gradient(nested),
        nesting_error=nesting_error,
        valid=jax.lax.stop_gradient(valid),
        status=jax.lax.stop_gradient(status),
        evidence=evidence,
        method_contract=_test_contract("negative-binomial-likelihood-ratio-test"),
    )


__all__ = [
    "LikelihoodRatioTestResult",
    "TEST_INVALID_FIT",
    "TEST_NO_DF_GAIN",
    "TEST_NONFINITE",
    "TEST_NOT_ESTIMABLE",
    "TEST_NOT_NESTED",
    "TEST_SUCCESS",
    "WaldTestResult",
    "likelihood_ratio_test",
    "wald_test",
]
