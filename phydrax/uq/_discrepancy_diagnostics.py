#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class DiscrepancyIdentifiabilityThresholds(StrictModule):
    """Release gates for repeated-data model-discrepancy validation."""

    min_repeats: int = eqx.field(static=True)
    max_fixed_bias_ratio: float = eqx.field(static=True)
    max_joint_bias_ratio: float = eqx.field(static=True)
    min_nll_improvement: float = eqx.field(static=True)
    min_crps_improvement: float = eqx.field(static=True)
    min_coverage: float = eqx.field(static=True)
    max_abs_parameter_gp_correlation: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        min_repeats: int = 5,
        max_fixed_bias_ratio: float = 0.75,
        max_joint_bias_ratio: float = 0.75,
        min_nll_improvement: float = 0.0,
        min_crps_improvement: float = 0.0,
        min_coverage: float = 0.85,
        max_abs_parameter_gp_correlation: float = 0.95,
    ):
        if int(min_repeats) < 2:
            raise ValueError("min_repeats must be at least two.")
        for name, value in (
            ("max_fixed_bias_ratio", max_fixed_bias_ratio),
            ("max_joint_bias_ratio", max_joint_bias_ratio),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie between zero and one.")
        if not 0.0 <= float(min_coverage) <= 1.0:
            raise ValueError("min_coverage must lie between zero and one.")
        if not 0.0 < float(max_abs_parameter_gp_correlation) <= 1.0:
            raise ValueError("max_abs_parameter_gp_correlation must lie in (0, 1].")
        self.min_repeats = int(min_repeats)
        self.max_fixed_bias_ratio = float(max_fixed_bias_ratio)
        self.max_joint_bias_ratio = float(max_joint_bias_ratio)
        self.min_nll_improvement = float(min_nll_improvement)
        self.min_crps_improvement = float(min_crps_improvement)
        self.min_coverage = float(min_coverage)
        self.max_abs_parameter_gp_correlation = float(max_abs_parameter_gp_correlation)


class DiscrepancyIdentifiabilityReport(StrictModule):
    """Exact metrics and failures from repeated discrepancy comparisons."""

    passed: bool = eqx.field(static=True)
    failures: tuple[str, ...] = eqx.field(static=True)
    num_repeats: int = eqx.field(static=True)
    baseline_parameter_bias: Array
    fixed_gp_parameter_bias: Array
    joint_gp_parameter_bias: Array
    nll_improvement: Array
    crps_improvement: Array
    mean_coverage: Array
    max_abs_parameter_gp_correlation: Array

    def __init__(
        self,
        *,
        failures: tuple[str, ...],
        num_repeats: int,
        baseline_parameter_bias: ArrayLike,
        fixed_gp_parameter_bias: ArrayLike,
        joint_gp_parameter_bias: ArrayLike,
        nll_improvement: ArrayLike,
        crps_improvement: ArrayLike,
        mean_coverage: ArrayLike,
        max_abs_parameter_gp_correlation: ArrayLike,
    ):
        self.passed = not failures
        self.failures = tuple(failures)
        self.num_repeats = int(num_repeats)
        self.baseline_parameter_bias = jnp.asarray(baseline_parameter_bias)
        self.fixed_gp_parameter_bias = jnp.asarray(fixed_gp_parameter_bias)
        self.joint_gp_parameter_bias = jnp.asarray(joint_gp_parameter_bias)
        self.nll_improvement = jnp.asarray(nll_improvement)
        self.crps_improvement = jnp.asarray(crps_improvement)
        self.mean_coverage = jnp.asarray(mean_coverage)
        self.max_abs_parameter_gp_correlation = jnp.asarray(
            max_abs_parameter_gp_correlation
        )

    def raise_on_failure(self) -> None:
        """Raise with every failed identifiability gate."""
        if not self.passed:
            raise RuntimeError(
                "Discrepancy identifiability gates failed: " + "; ".join(self.failures)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "num_repeats": self.num_repeats,
            "baseline_parameter_bias": float(self.baseline_parameter_bias),
            "fixed_gp_parameter_bias": float(self.fixed_gp_parameter_bias),
            "joint_gp_parameter_bias": float(self.joint_gp_parameter_bias),
            "nll_improvement": float(self.nll_improvement),
            "crps_improvement": float(self.crps_improvement),
            "mean_coverage": float(self.mean_coverage),
            "max_abs_parameter_gp_correlation": float(
                self.max_abs_parameter_gp_correlation
            ),
        }


def discrepancy_identifiability_report(
    *,
    true_parameters: ArrayLike,
    baseline_parameter_estimates: ArrayLike,
    fixed_gp_parameter_estimates: ArrayLike,
    joint_gp_parameter_estimates: ArrayLike,
    baseline_nll: ArrayLike,
    fixed_gp_nll: ArrayLike,
    baseline_crps: ArrayLike,
    fixed_gp_crps: ArrayLike,
    fixed_gp_coverage: ArrayLike,
    joint_parameter_gp_correlations: ArrayLike,
    thresholds: DiscrepancyIdentifiabilityThresholds | None = None,
) -> DiscrepancyIdentifiabilityReport:
    """Gate discrepancy use on repeated bias, score, coverage, and confounding tests."""
    limits = thresholds or DiscrepancyIdentifiabilityThresholds()
    if not isinstance(limits, DiscrepancyIdentifiabilityThresholds):
        raise TypeError("thresholds must be DiscrepancyIdentifiabilityThresholds.")
    baseline = _parameter_estimates(
        baseline_parameter_estimates,
        name="baseline_parameter_estimates",
    )
    fixed = _parameter_estimates(
        fixed_gp_parameter_estimates,
        name="fixed_gp_parameter_estimates",
    )
    joint = _parameter_estimates(
        joint_gp_parameter_estimates,
        name="joint_gp_parameter_estimates",
    )
    if baseline.shape != fixed.shape or baseline.shape != joint.shape:
        raise ValueError("All parameter estimate arrays must have identical shape.")
    repeats = int(baseline.shape[0])
    truth = jnp.asarray(true_parameters, dtype=float)
    if truth.ndim == 0:
        truth = truth[None]
    if truth.shape != baseline.shape[1:]:
        raise ValueError("true_parameters must align with each repeated estimate.")

    baseline_nll_array = _repeat_metric(baseline_nll, repeats, "baseline_nll")
    fixed_nll_array = _repeat_metric(fixed_gp_nll, repeats, "fixed_gp_nll")
    baseline_crps_array = _repeat_metric(baseline_crps, repeats, "baseline_crps")
    fixed_crps_array = _repeat_metric(fixed_gp_crps, repeats, "fixed_gp_crps")
    coverage = _repeat_metric(fixed_gp_coverage, repeats, "fixed_gp_coverage")
    correlations = jnp.asarray(joint_parameter_gp_correlations, dtype=float)
    if correlations.ndim < 2 or int(correlations.shape[0]) != repeats:
        raise ValueError(
            "joint_parameter_gp_correlations must have a leading repeat axis."
        )
    arrays = (
        baseline,
        fixed,
        joint,
        baseline_nll_array,
        fixed_nll_array,
        baseline_crps_array,
        fixed_crps_array,
        coverage,
        correlations,
    )
    if any(not bool(jnp.all(jnp.isfinite(value))) for value in arrays):
        raise ValueError("Identifiability inputs must be finite.")
    if bool(jnp.any((coverage < 0.0) | (coverage > 1.0))):
        raise ValueError("fixed_gp_coverage must lie between zero and one.")
    if bool(jnp.any(jnp.abs(correlations) > 1.0 + 1e-7)):
        raise ValueError("Correlations must lie between negative and positive one.")

    baseline_bias = _bias(baseline, truth)
    fixed_bias = _bias(fixed, truth)
    joint_bias = _bias(joint, truth)
    nll_improvement = jnp.mean(baseline_nll_array - fixed_nll_array)
    crps_improvement = jnp.mean(baseline_crps_array - fixed_crps_array)
    mean_coverage = jnp.mean(coverage)
    max_correlation = jnp.max(jnp.abs(correlations))
    failures: list[str] = []
    if repeats < limits.min_repeats:
        failures.append(f"num_repeats={repeats} < min_repeats={limits.min_repeats}")
    bias_floor = jnp.finfo(baseline.dtype).eps
    bias_denominator = jnp.maximum(baseline_bias, bias_floor)
    fixed_ratio = fixed_bias / bias_denominator
    joint_ratio = joint_bias / bias_denominator
    if float(fixed_ratio) > limits.max_fixed_bias_ratio:
        failures.append(
            "fixed GP parameter-bias ratio "
            f"{float(fixed_ratio):.6g} > {limits.max_fixed_bias_ratio:.6g}"
        )
    if float(joint_ratio) > limits.max_joint_bias_ratio:
        failures.append(
            "joint GP parameter-bias ratio "
            f"{float(joint_ratio):.6g} > {limits.max_joint_bias_ratio:.6g}"
        )
    if float(nll_improvement) < limits.min_nll_improvement:
        failures.append(
            f"NLL improvement {float(nll_improvement):.6g} < "
            f"{limits.min_nll_improvement:.6g}"
        )
    if float(crps_improvement) < limits.min_crps_improvement:
        failures.append(
            f"CRPS improvement {float(crps_improvement):.6g} < "
            f"{limits.min_crps_improvement:.6g}"
        )
    if float(mean_coverage) < limits.min_coverage:
        failures.append(
            f"coverage {float(mean_coverage):.6g} < {limits.min_coverage:.6g}"
        )
    if float(max_correlation) > limits.max_abs_parameter_gp_correlation:
        failures.append(
            "parameter/GP correlation "
            f"{float(max_correlation):.6g} > "
            f"{limits.max_abs_parameter_gp_correlation:.6g}"
        )

    return DiscrepancyIdentifiabilityReport(
        failures=tuple(failures),
        num_repeats=repeats,
        baseline_parameter_bias=baseline_bias,
        fixed_gp_parameter_bias=fixed_bias,
        joint_gp_parameter_bias=joint_bias,
        nll_improvement=nll_improvement,
        crps_improvement=crps_improvement,
        mean_coverage=mean_coverage,
        max_abs_parameter_gp_correlation=max_correlation,
    )


def _parameter_estimates(value: ArrayLike, *, name: str) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or int(array.shape[0]) < 1:
        raise ValueError(f"{name} must have shape (repeat, parameter).")
    return array


def _repeat_metric(value: ArrayLike, repeats: int, name: str) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape != (repeats,):
        raise ValueError(f"{name} must have shape ({repeats},).")
    return array


def _bias(estimates: Array, truth: Array) -> Array:
    return jnp.mean(jnp.abs(jnp.mean(estimates, axis=0) - truth))


__all__ = [
    "DiscrepancyIdentifiabilityReport",
    "DiscrepancyIdentifiabilityThresholds",
    "discrepancy_identifiability_report",
]
