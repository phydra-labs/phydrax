#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from phydrax._fingerprint import canonical_fingerprint

from ._runtime import DurationDistribution


PerformanceObjective = Literal["minimize", "maximize"]


@dataclass(frozen=True, slots=True)
class PerformancePolicy:
    """Practical and statistical requirements for one performance decision."""

    objective: PerformanceObjective
    relative_tolerance: float | None = None
    absolute_tolerance: float | None = None
    confidence: float = 0.95
    bootstrap_resamples: int = 10_000
    minimum_samples: int = 5

    def __post_init__(self) -> None:
        if self.objective not in ("minimize", "maximize"):
            raise ValueError("Performance objective must be 'minimize' or 'maximize'.")
        tolerances = (self.relative_tolerance, self.absolute_tolerance)
        if all(value is None for value in tolerances):
            raise ValueError("At least one practical performance tolerance is required.")
        if any(
            value is not None and (not math.isfinite(value) or value < 0.0)
            for value in tolerances
        ):
            raise ValueError("Performance tolerances must be finite and nonnegative.")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one.")
        if self.bootstrap_resamples <= 0 or self.minimum_samples <= 0:
            raise ValueError("Bootstrap and minimum sample counts must be positive.")


@dataclass(frozen=True, slots=True)
class PerformanceComparison:
    """Observed and bootstrapped degradation evidence."""

    baseline: DurationDistribution
    candidate: DurationDistribution
    paired: bool
    absolute_degradation: float
    absolute_interval: tuple[float, float] | None
    relative_degradation: float | None
    relative_interval: tuple[float, float] | None
    sufficient_samples: bool
    regressed: bool | None
    reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_seconds_dict(),
            "candidate": self.candidate.to_seconds_dict(),
            "paired": self.paired,
            "absolute_degradation": self.absolute_degradation,
            "absolute_interval": self.absolute_interval,
            "relative_degradation": self.relative_degradation,
            "relative_interval": self.relative_interval,
            "sufficient_samples": self.sufficient_samples,
            "regressed": self.regressed,
            "reason": self.reason,
        }


def compare_performance(
    baseline: DurationDistribution,
    candidate: DurationDistribution,
    policy: PerformancePolicy,
    /,
    *,
    comparison_id: str,
    baseline_pair_ids: tuple[str, ...] | None = None,
    candidate_pair_ids: tuple[str, ...] | None = None,
) -> PerformanceComparison:
    """Compare raw samples without inventing pairing or environment compatibility."""
    if not comparison_id:
        raise ValueError("comparison_id must be non-empty.")
    paired = _validate_pairing(
        baseline,
        candidate,
        baseline_pair_ids,
        candidate_pair_ids,
    )
    baseline_median = _required_median(baseline, "baseline")
    candidate_median = _required_median(candidate, "candidate")
    orientation = 1.0 if policy.objective == "minimize" else -1.0
    absolute = orientation * (candidate_median - baseline_median)
    relative = None if baseline_median == 0.0 else absolute / abs(baseline_median)
    sufficient = (
        baseline.count >= policy.minimum_samples
        and candidate.count >= policy.minimum_samples
    )
    if not sufficient:
        return PerformanceComparison(
            baseline,
            candidate,
            paired,
            absolute,
            None,
            relative,
            None,
            False,
            None,
            "insufficient samples for the declared policy",
        )
    seed = int(
        canonical_fingerprint(
            {
                "comparison": comparison_id,
                "objective": policy.objective,
                "confidence": policy.confidence,
                "resamples": policy.bootstrap_resamples,
                "paired": paired,
            }
        )[:16],
        16,
    )
    generator = np.random.default_rng(seed)
    absolute_samples, relative_samples = _bootstrap_degradation(
        np.asarray(baseline.samples_seconds, dtype=np.float64),
        np.asarray(candidate.samples_seconds, dtype=np.float64),
        orientation=orientation,
        resamples=policy.bootstrap_resamples,
        paired=paired,
        generator=generator,
    )
    alpha = 1.0 - policy.confidence
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    absolute_interval = tuple(
        float(value) for value in np.quantile(absolute_samples, quantiles)
    )
    relative_interval = (
        None
        if relative_samples is None
        else tuple(float(value) for value in np.quantile(relative_samples, quantiles))
    )
    decisions: list[bool] = []
    if policy.absolute_tolerance is not None:
        decisions.append(_exceeds(absolute_interval[0], policy.absolute_tolerance))
    if policy.relative_tolerance is not None and relative_interval is not None:
        decisions.append(_exceeds(relative_interval[0], policy.relative_tolerance))
    if (
        policy.relative_tolerance is not None
        and relative_interval is None
        and policy.absolute_tolerance is None
    ):
        return PerformanceComparison(
            baseline,
            candidate,
            paired,
            absolute,
            absolute_interval,
            None,
            None,
            True,
            None,
            "zero baseline requires an absolute practical tolerance",
        )
    return PerformanceComparison(
        baseline,
        candidate,
        paired,
        absolute,
        absolute_interval,
        relative,
        relative_interval,
        True,
        all(decisions),
        None,
    )


def _validate_pairing(
    baseline: DurationDistribution,
    candidate: DurationDistribution,
    baseline_pair_ids: tuple[str, ...] | None,
    candidate_pair_ids: tuple[str, ...] | None,
) -> bool:
    if baseline_pair_ids is None and candidate_pair_ids is None:
        return False
    if baseline_pair_ids is None or candidate_pair_ids is None:
        raise ValueError("Both baseline and candidate pair IDs are required for pairing.")
    if baseline_pair_ids != candidate_pair_ids:
        raise ValueError("Paired sample IDs must match exactly and in order.")
    if (
        len(baseline_pair_ids) != baseline.count
        or len(candidate_pair_ids) != candidate.count
    ):
        raise ValueError("Pair IDs must cover every timing sample.")
    if len(set(baseline_pair_ids)) != len(baseline_pair_ids):
        raise ValueError("Pair IDs must be unique.")
    return True


def _required_median(distribution: DurationDistribution, owner: str, /) -> float:
    value = distribution.median_seconds
    if value is None:
        raise ValueError(f"{owner} samples must be non-empty.")
    return value


def _exceeds(value: float, threshold: float, /) -> bool:
    return value > threshold and not math.isclose(
        value,
        threshold,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )


def _bootstrap_degradation(
    baseline: np.ndarray,
    candidate: np.ndarray,
    /,
    *,
    orientation: float,
    resamples: int,
    paired: bool,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    if paired:
        indices = generator.integers(0, baseline.size, size=(resamples, baseline.size))
        baseline_medians = np.median(baseline[indices], axis=1)
        candidate_medians = np.median(candidate[indices], axis=1)
    else:
        baseline_indices = generator.integers(
            0,
            baseline.size,
            size=(resamples, baseline.size),
        )
        candidate_indices = generator.integers(
            0,
            candidate.size,
            size=(resamples, candidate.size),
        )
        baseline_medians = np.median(baseline[baseline_indices], axis=1)
        candidate_medians = np.median(candidate[candidate_indices], axis=1)
    absolute = orientation * (candidate_medians - baseline_medians)
    if np.any(baseline_medians == 0.0):
        return absolute, None
    return absolute, absolute / np.abs(baseline_medians)


__all__ = [
    "PerformanceComparison",
    "PerformanceObjective",
    "PerformancePolicy",
    "compare_performance",
]
