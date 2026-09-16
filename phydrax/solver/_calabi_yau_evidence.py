#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent held-out evidence for frozen Calabi–Yau metric candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..geometry.complex import (
    HypersurfaceKahlerGeometry,
    ProjectiveHypersurface,
    ProjectiveLineSamples,
)
from ._calabi_yau import CalabiYauMetricResult


class CalabiYauMetricEvidencePlan(StrictModule):
    """Frozen independent sample ancestry, weights, tolerances, and resources."""

    training_samples: ProjectiveLineSamples
    heldout_samples: ProjectiveLineSamples
    heldout_weights: Array
    quantiles: tuple[float, ...] = eqx.field(static=True)
    batch_count: int = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)
    residual_rms_tolerance: float = eqx.field(static=True)
    positivity_floor: float = eqx.field(static=True)
    minimum_valid_fraction: float = eqx.field(static=True)
    ricci_tolerance: float = eqx.field(static=True)
    require_ricci: bool = eqx.field(static=True)
    training_sample_id: str = eqx.field(static=True)
    heldout_sample_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        training_samples: ProjectiveLineSamples,
        heldout_samples: ProjectiveLineSamples,
        /,
        *,
        heldout_weights: ArrayLike | None = None,
        quantiles: Sequence[float] = (0.5, 0.9, 0.99),
        batch_count: int = 4,
        maximum_samples: int = 100_000,
        residual_rms_tolerance: float = 1e-3,
        positivity_floor: float = 1e-8,
        minimum_valid_fraction: float = 0.9,
        ricci_tolerance: float = 1e-3,
        require_ricci: bool = False,
    ):
        if not isinstance(training_samples, ProjectiveLineSamples) or not isinstance(
            heldout_samples, ProjectiveLineSamples
        ):
            raise TypeError(
                "training_samples and heldout_samples must be projective samples."
            )
        training_count = int(training_samples.homogeneous_points.shape[0])
        heldout_count = int(heldout_samples.homogeneous_points.shape[0])
        maximum = int(maximum_samples)
        batches = int(batch_count)
        quantile_values = tuple(float(value) for value in quantiles)
        residual_tolerance = float(residual_rms_tolerance)
        positivity = float(positivity_floor)
        valid_fraction = float(minimum_valid_fraction)
        ricci = float(ricci_tolerance)
        if training_count < 1 or heldout_count < 1:
            raise ValueError("Training and held-out sample sets must be nonempty.")
        if training_count > maximum or heldout_count > maximum or maximum < 1:
            raise ValueError("Calabi-Yau evidence samples exceed maximum_samples.")
        if batches < 1 or batches > heldout_count:
            raise ValueError(
                "batch_count must lie between one and held-out sample count."
            )
        if (
            not quantile_values
            or any(not 0.0 < value < 1.0 for value in quantile_values)
            or tuple(sorted(set(quantile_values))) != quantile_values
        ):
            raise ValueError("quantiles must be unique, increasing, and lie in (0, 1).")
        if (
            not all(
                np.isfinite(value)
                for value in (residual_tolerance, positivity, valid_fraction, ricci)
            )
            or residual_tolerance < 0.0
            or positivity < 0.0
            or not 0.0 < valid_fraction <= 1.0
            or ricci < 0.0
        ):
            raise ValueError("Calabi-Yau evidence tolerances are invalid.")
        training_points = np.asarray(training_samples.homogeneous_points)
        heldout_points = np.asarray(heldout_samples.homogeneous_points)
        if training_points.shape[1:] != heldout_points.shape[1:]:
            raise ValueError("Training and held-out projective coordinate shapes differ.")
        training_rows = {
            np.ascontiguousarray(point).view(np.uint8).tobytes()
            for point in training_points
        }
        duplicates = any(
            np.ascontiguousarray(point).view(np.uint8).tobytes() in training_rows
            for point in heldout_points
        )
        if duplicates:
            raise ValueError("Training and held-out sample sets share an exact point.")
        raw_weights = (
            np.ones((heldout_count,), dtype=float)
            if heldout_weights is None
            else np.asarray(heldout_weights, dtype=float)
        )
        if raw_weights.shape != (heldout_count,) or not np.all(np.isfinite(raw_weights)):
            raise ValueError("heldout_weights must be one finite sample vector.")
        if np.any(raw_weights < 0.0) or np.sum(raw_weights) <= 0.0:
            raise ValueError("heldout_weights must be nonnegative with positive mass.")
        weights = raw_weights / np.sum(raw_weights)
        training_id = canonical_fingerprint(
            {
                "kind": "calabi-yau-training-samples",
                "arrays": array_tree_fingerprint(
                    (
                        training_samples.homogeneous_points,
                        training_samples.chart_indices,
                        training_samples.pivot_indices,
                        training_samples.line_ids,
                        training_samples.root_ids,
                    )
                ),
            }
        )
        heldout_id = canonical_fingerprint(
            {
                "kind": "calabi-yau-heldout-samples",
                "arrays": array_tree_fingerprint(
                    (
                        heldout_samples.homogeneous_points,
                        heldout_samples.chart_indices,
                        heldout_samples.pivot_indices,
                        heldout_samples.line_ids,
                        heldout_samples.root_ids,
                    )
                ),
            }
        )
        if training_id == heldout_id:
            raise ValueError("Training and held-out sample identities must differ.")
        self.training_samples = training_samples
        self.heldout_samples = heldout_samples
        self.heldout_weights = jnp.asarray(weights)
        self.quantiles = quantile_values
        self.batch_count = batches
        self.maximum_samples = maximum
        self.residual_rms_tolerance = residual_tolerance
        self.positivity_floor = positivity
        self.minimum_valid_fraction = valid_fraction
        self.ricci_tolerance = ricci
        self.require_ricci = bool(require_ricci)
        self.training_sample_id = training_id
        self.heldout_sample_id = heldout_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "calabi-yau-metric-evidence-plan",
                "training_samples": training_id,
                "heldout_samples": heldout_id,
                "weights": array_tree_fingerprint(weights),
                "quantiles": quantile_values,
                "batch_count": batches,
                "maximum_samples": maximum,
                "residual_rms_tolerance": residual_tolerance,
                "positivity_floor": positivity,
                "minimum_valid_fraction": valid_fraction,
                "ricci_tolerance": ricci,
                "require_ricci": bool(require_ricci),
            }
        )


class CalabiYauMetricEvidence(StrictModule):
    residuals: Array
    positivity_margins: Array
    volume_ratio_residuals: Array
    sample_valid: Array
    normalized_weights: Array
    weighted_mean_residual: Array
    weighted_rms_residual: Array
    maximum_absolute_residual: Array
    absolute_residual_quantiles: Array
    minimum_positivity_margin: Array
    valid_fraction: Array
    effective_sample_size: Array
    batch_rms_residuals: Array
    batch_standard_deviation: Array
    minimum_smoothness_margin: Array
    minimum_chart_margin: Array
    ricci_residuals: Array
    maximum_ricci_residual: Array
    ricci_available: Array
    kahler_by_construction: Array
    accepted: Array
    precision_evidence: PrecisionEvidenceEnvelope
    plan_id: str = eqx.field(static=True)
    hypersurface_id: str = eqx.field(static=True)
    training_sample_id: str = eqx.field(static=True)
    heldout_sample_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple[float, ...],
    /,
) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative = cumulative / cumulative[-1]
    return np.asarray(
        [
            sorted_values[min(np.searchsorted(cumulative, q), len(values) - 1)]
            for q in quantiles
        ]
    )


def evaluate_calabi_yau_metric_evidence(
    result: CalabiYauMetricResult,
    hypersurface: ProjectiveHypersurface,
    plan: CalabiYauMetricEvidencePlan,
    /,
    *,
    ricci_evaluator: Callable[[Array, int, int], Array] | None = None,
) -> CalabiYauMetricEvidence:
    """Evaluate a solved metric on independent points and retain all raw axes."""
    if not isinstance(result, CalabiYauMetricResult):
        raise TypeError("result must be CalabiYauMetricResult.")
    if not isinstance(hypersurface, ProjectiveHypersurface):
        raise TypeError("hypersurface must be ProjectiveHypersurface.")
    if not isinstance(plan, CalabiYauMetricEvidencePlan):
        raise TypeError("plan must be CalabiYauMetricEvidencePlan.")
    if result.hypersurface_id != hypersurface.hypersurface_id:
        raise ValueError("Result and hypersurface identities differ.")
    if plan.require_ricci and ricci_evaluator is None:
        raise ValueError("This evidence plan requires an explicit Ricci evaluator.")
    if ricci_evaluator is not None and not callable(ricci_evaluator):
        raise TypeError("ricci_evaluator must be callable or None.")
    geometry = HypersurfaceKahlerGeometry(
        hypersurface,
        result.potential_model,
        normalization=result.normalization,
        positivity_floor=plan.positivity_floor,
    )
    residuals = []
    margins = []
    volume_residuals = []
    valid = []
    ricci_values = []
    samples = plan.heldout_samples
    for index in range(samples.homogeneous_points.shape[0]):
        evaluation = geometry.evaluate(
            samples.homogeneous_points[index],
            chart_index=int(samples.chart_indices[index]),
            pivot_index=int(samples.pivot_indices[index]),
        )
        residuals.append(evaluation.monge_ampere_residual)
        margins.append(evaluation.positivity_margin)
        volume_residuals.append(jnp.expm1(evaluation.monge_ampere_residual))
        valid.append(samples.valid[index] & evaluation.valid)
        if ricci_evaluator is not None:
            ricci_values.append(
                jnp.asarray(
                    ricci_evaluator(
                        samples.homogeneous_points[index],
                        int(samples.chart_indices[index]),
                        int(samples.pivot_indices[index]),
                    )
                ).reshape(())
            )
    residual = jnp.stack(residuals)
    margin = jnp.stack(margins)
    volume = jnp.stack(volume_residuals)
    validity = jnp.stack(valid) & jnp.isfinite(residual) & jnp.isfinite(margin)
    raw_weights = jnp.where(validity, plan.heldout_weights, 0.0)
    mass = jnp.sum(raw_weights)
    normalized = jnp.where(mass > 0.0, raw_weights / mass, 0.0)
    mean = jnp.sum(normalized * residual)
    rms = jnp.sqrt(jnp.sum(normalized * residual**2))
    maximum = jnp.max(jnp.where(validity, jnp.abs(residual), 0.0))
    minimum_margin = jnp.min(jnp.where(validity, margin, jnp.inf))
    fraction = jnp.mean(validity.astype(float))
    ess = jnp.where(
        mass > 0.0,
        1.0 / jnp.sum(normalized**2),
        0.0,
    )
    host_valid = np.asarray(validity)
    host_values = np.abs(np.asarray(residual)[host_valid])
    host_weights = np.asarray(normalized)[host_valid]
    quantile_values = (
        np.full((len(plan.quantiles),), np.nan)
        if host_values.size == 0
        else _weighted_quantiles(host_values, host_weights, plan.quantiles)
    )
    index_batches = np.array_split(np.arange(residual.shape[0]), plan.batch_count)
    batch_values = []
    for indices in index_batches:
        batch_weights = normalized[jnp.asarray(indices)]
        batch_mass = jnp.sum(batch_weights)
        batch_values.append(
            jnp.where(
                batch_mass > 0.0,
                jnp.sqrt(
                    jnp.sum(batch_weights * residual[jnp.asarray(indices)] ** 2)
                    / batch_mass
                ),
                jnp.nan,
            )
        )
    batch_rms = jnp.stack(batch_values)
    batch_std = jnp.nanstd(batch_rms)
    smoothness = jnp.min(jnp.where(samples.valid, samples.smoothness_margins, jnp.inf))
    points = samples.homogeneous_points
    chart_values = jnp.abs(
        jnp.take_along_axis(points, samples.chart_indices[:, None], axis=1)[:, 0]
    )
    chart_margin = jnp.min(jnp.where(samples.valid, chart_values, jnp.inf))
    if ricci_values:
        ricci_residuals = jnp.stack(ricci_values)
        maximum_ricci = jnp.max(jnp.abs(ricci_residuals))
        ricci_available = jnp.asarray(True)
    else:
        ricci_residuals = jnp.full(residual.shape, jnp.nan)
        maximum_ricci = jnp.asarray(jnp.nan, dtype=residual.dtype)
        ricci_available = jnp.asarray(False)
    finite = (
        (mass > 0.0)
        & jnp.isfinite(mean)
        & jnp.isfinite(rms)
        & jnp.isfinite(maximum)
        & jnp.isfinite(minimum_margin)
        & jnp.all(jnp.isfinite(jnp.asarray(quantile_values)))
    )
    ricci_ok = jnp.asarray(not plan.require_ricci) | (
        ricci_available
        & jnp.isfinite(maximum_ricci)
        & (maximum_ricci <= plan.ricci_tolerance)
    )
    accepted = (
        finite
        & (rms <= plan.residual_rms_tolerance)
        & (minimum_margin >= plan.positivity_floor)
        & (fraction >= plan.minimum_valid_fraction)
        & ricci_ok
    )
    precision_evidence = result.precision.evidence_for(residual)
    evidence_id = canonical_fingerprint(
        {
            "kind": "calabi-yau-metric-evidence",
            "plan": plan.plan_id,
            "hypersurface": hypersurface.hypersurface_id,
            "training_samples": plan.training_sample_id,
            "heldout_samples": plan.heldout_sample_id,
            "residuals": array_tree_fingerprint(residual),
            "positivity": array_tree_fingerprint(margin),
            "validity": array_tree_fingerprint(validity),
            "precision": precision_evidence.evidence_id,
        }
    )
    return CalabiYauMetricEvidence(
        residuals=residual,
        positivity_margins=margin,
        volume_ratio_residuals=volume,
        sample_valid=validity,
        normalized_weights=normalized,
        weighted_mean_residual=mean,
        weighted_rms_residual=rms,
        maximum_absolute_residual=maximum,
        absolute_residual_quantiles=jnp.asarray(quantile_values),
        minimum_positivity_margin=minimum_margin,
        valid_fraction=fraction,
        effective_sample_size=ess,
        batch_rms_residuals=batch_rms,
        batch_standard_deviation=batch_std,
        minimum_smoothness_margin=smoothness,
        minimum_chart_margin=chart_margin,
        ricci_residuals=ricci_residuals,
        maximum_ricci_residual=maximum_ricci,
        ricci_available=ricci_available,
        kahler_by_construction=jnp.asarray(True),
        accepted=accepted,
        precision_evidence=precision_evidence,
        plan_id=plan.plan_id,
        hypersurface_id=hypersurface.hypersurface_id,
        training_sample_id=plan.training_sample_id,
        heldout_sample_id=plan.heldout_sample_id,
        evidence_id=evidence_id,
        claim="independent-heldout-approximate-metric-evidence-no-exact-ricci-flat-claim",
    )


__all__ = [
    "CalabiYauMetricEvidence",
    "CalabiYauMetricEvidencePlan",
    "evaluate_calabi_yau_metric_evidence",
]
