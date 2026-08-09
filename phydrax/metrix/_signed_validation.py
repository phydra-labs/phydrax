#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._metric import AbstractSemiRiemannianMetric, LorentzianMetric


class SignedMetricValidationReport(StrictModule):
    """Representative-point diagnostics for a declared nondegenerate metric."""

    valid: Array
    finite: Array
    maximum_asymmetry: Array
    signature_matches: Array
    observed_positive: Array
    observed_negative: Array
    observed_near_zero: Array
    minimum_absolute_eigenvalue: Array
    maximum_condition_number: Array

    def __init__(
        self,
        *,
        valid: Array,
        finite: Array,
        maximum_asymmetry: Array,
        signature_matches: Array,
        observed_positive: Array,
        observed_negative: Array,
        observed_near_zero: Array,
        minimum_absolute_eigenvalue: Array,
        maximum_condition_number: Array,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.maximum_asymmetry = jnp.asarray(maximum_asymmetry)
        self.signature_matches = jnp.asarray(signature_matches, dtype=bool)
        self.observed_positive = jnp.asarray(observed_positive, dtype=jnp.int32)
        self.observed_negative = jnp.asarray(observed_negative, dtype=jnp.int32)
        self.observed_near_zero = jnp.asarray(observed_near_zero, dtype=jnp.int32)
        self.minimum_absolute_eigenvalue = jnp.asarray(minimum_absolute_eigenvalue)
        self.maximum_condition_number = jnp.asarray(maximum_condition_number)


def validate_semi_riemannian_metric(
    metric: AbstractSemiRiemannianMetric,
    points: ArrayLike,
    /,
    *,
    symmetry_tolerance: float = 1e-10,
    eigenvalue_tolerance: float = 1e-10,
    maximum_condition_number: float | None = None,
    raise_on_error: bool = False,
) -> SignedMetricValidationReport:
    """Validate symmetry, nondegeneracy, and constant declared signature."""
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("metric must be an AbstractSemiRiemannianMetric.")
    if symmetry_tolerance < 0.0 or eigenvalue_tolerance < 0.0:
        raise ValueError("Metric validation tolerances must be non-negative.")
    if maximum_condition_number is not None and maximum_condition_number <= 0.0:
        raise ValueError("maximum_condition_number must be positive when supplied.")
    matrices = metric(points)
    finite = jnp.all(jnp.isfinite(matrices))
    asymmetry = jnp.max(jnp.abs(matrices - jnp.swapaxes(matrices, -1, -2)))
    symmetric = asymmetry <= symmetry_tolerance
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (matrices + jnp.swapaxes(matrices, -1, -2)))
    magnitude = jnp.max(jnp.abs(eigenvalues), axis=-1, keepdims=True)
    threshold = eigenvalue_tolerance * jnp.maximum(magnitude, 1.0)
    positive = eigenvalues > threshold
    negative = eigenvalues < -threshold
    near_zero = ~(positive | negative)
    positive_count = jnp.sum(positive, axis=-1, dtype=jnp.int32)
    negative_count = jnp.sum(negative, axis=-1, dtype=jnp.int32)
    near_zero_count = jnp.sum(near_zero, axis=-1, dtype=jnp.int32)
    signature_matches = jnp.all(
        (positive_count == metric.signature.positive)
        & (negative_count == metric.signature.negative)
        & (near_zero_count == 0)
    )
    absolute_eigenvalues = jnp.abs(eigenvalues)
    minimum_absolute_eigenvalue = jnp.min(absolute_eigenvalues)
    condition = jnp.max(absolute_eigenvalues, axis=-1) / jnp.min(
        absolute_eigenvalues, axis=-1
    )
    observed_maximum_condition = jnp.max(condition)
    condition_valid = (
        jnp.asarray(True)
        if maximum_condition_number is None
        else observed_maximum_condition <= maximum_condition_number
    )
    valid = finite & symmetric & signature_matches & condition_valid
    report = SignedMetricValidationReport(
        valid=valid,
        finite=finite,
        maximum_asymmetry=asymmetry,
        signature_matches=signature_matches,
        observed_positive=positive_count,
        observed_negative=negative_count,
        observed_near_zero=near_zero_count,
        minimum_absolute_eigenvalue=minimum_absolute_eigenvalue,
        maximum_condition_number=observed_maximum_condition,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Signed metric validation failed: "
            f"finite={bool(jax.device_get(finite))}, "
            f"maximum_asymmetry={float(jax.device_get(asymmetry))}, "
            f"signature_matches={bool(jax.device_get(signature_matches))}, "
            "minimum_absolute_eigenvalue="
            f"{float(jax.device_get(minimum_absolute_eigenvalue))}, "
            "maximum_condition_number="
            f"{float(jax.device_get(observed_maximum_condition))}."
        )
    return report


def validate_lorentzian_metric(
    metric: LorentzianMetric,
    points: ArrayLike,
    /,
    **kwargs,
) -> SignedMetricValidationReport:
    """Validate a Lorentzian metric at representative points."""
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("validate_lorentzian_metric requires a LorentzianMetric.")
    return validate_semi_riemannian_metric(metric, points, **kwargs)


__all__ = [
    "SignedMetricValidationReport",
    "validate_lorentzian_metric",
    "validate_semi_riemannian_metric",
]
