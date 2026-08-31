#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..imaging import DenseDisplacementField2D


def _safe_ratio(numerator: Array, denominator: Array, /) -> Array:
    denominator_ = jnp.asarray(denominator)
    return jnp.where(denominator_ > 0, numerator / jnp.maximum(denominator_, 1), 0.0)


def _masked_mean(values: Array, valid: Array, /) -> Array:
    values_ = jnp.asarray(values)
    base_valid = jnp.asarray(valid, dtype=bool)
    valid_ = base_valid
    while valid_.ndim < values_.ndim:
        valid_ = valid_[..., None]
    axes = tuple(range(base_valid.ndim))
    numerator = jnp.sum(jnp.where(valid_, values_, 0.0), axis=axes)
    denominator = jnp.sum(base_valid, dtype=values_.dtype)
    return jnp.where(denominator > 0, numerator / jnp.maximum(denominator, 1), 0.0)


class QualificationEvidence(StrictModule, NonTrainableState):
    """One metric value, its support, and finite-computation evidence."""

    metric: str = eqx.field(static=True)
    value: Array
    support: Array
    finite: Array
    status: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        metric: str,
        value: ArrayLike,
        support: ArrayLike,
        /,
        *,
        finite: ArrayLike,
        status: str,
        source_id: str,
    ):
        metric_ = str(metric)
        status_ = str(status)
        source = str(source_id)
        if not metric_ or not status_ or not source:
            raise ValueError("Qualification evidence identifiers must be non-empty.")
        value_ = jnp.asarray(value)
        support_ = jnp.asarray(support, dtype=jnp.int32).reshape(())
        if bool(jnp.any(support_ < 0)):
            raise ValueError("Qualification evidence support must be non-negative.")
        self.metric = metric_
        self.value = value_
        self.support = support_
        self.finite = jnp.asarray(finite, dtype=bool).reshape(())
        self.status = status_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "velocimetry-qualification-evidence",
                "metric": metric_,
                "source": source,
                "status": status_,
            }
        )


class PIVQualificationResult(StrictModule, NonTrainableState):
    """Dense displacement bias, endpoint error, and coverage metrics."""

    bias_rc: Array
    mean_endpoint_error: Array
    root_mean_square_endpoint_error: Array
    p95_endpoint_error: Array
    coverage: Array
    valid_count: Array
    finite: Array
    evidence: tuple[QualificationEvidence, ...]
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        bias_rc: ArrayLike,
        mean_endpoint_error: ArrayLike,
        root_mean_square_endpoint_error: ArrayLike,
        p95_endpoint_error: ArrayLike,
        coverage: ArrayLike,
        valid_count: ArrayLike,
        finite: ArrayLike,
        evidence: Sequence[QualificationEvidence],
        /,
        *,
        source_id: str,
    ):
        evidence_ = tuple(evidence)
        if any(not isinstance(item, QualificationEvidence) for item in evidence_):
            raise TypeError("evidence must contain QualificationEvidence values.")
        self.bias_rc = jnp.asarray(bias_rc).reshape((2,))
        self.mean_endpoint_error = jnp.asarray(mean_endpoint_error).reshape(())
        self.root_mean_square_endpoint_error = jnp.asarray(
            root_mean_square_endpoint_error
        ).reshape(())
        self.p95_endpoint_error = jnp.asarray(p95_endpoint_error).reshape(())
        self.coverage = jnp.asarray(coverage).reshape(())
        self.valid_count = jnp.asarray(valid_count, dtype=jnp.int32).reshape(())
        self.finite = jnp.asarray(finite, dtype=bool).reshape(())
        self.evidence = evidence_
        self.result_id = canonical_fingerprint(
            {
                "kind": "piv-qualification-result",
                "source": str(source_id),
                "evidence": [item.evidence_id for item in evidence_],
            }
        )


def qualify_piv(
    estimate: DenseDisplacementField2D,
    truth: DenseDisplacementField2D,
    /,
) -> PIVQualificationResult:
    """Evaluate an estimated row/column displacement field on shared valid support."""
    if not isinstance(estimate, DenseDisplacementField2D) or not isinstance(
        truth, DenseDisplacementField2D
    ):
        raise TypeError("estimate and truth must be DenseDisplacementField2D values.")
    if estimate.geometry_id != truth.geometry_id:
        raise ValueError("PIV estimate and truth must share one image geometry.")
    if estimate.displacement_rc.shape != truth.displacement_rc.shape:
        raise ValueError("PIV estimate and truth displacement shapes must match.")
    if estimate.positions_rc.shape != truth.positions_rc.shape:
        raise ValueError("PIV estimate and truth coordinate shapes must match.")
    if not bool(jnp.array_equal(estimate.positions_rc, truth.positions_rc)):
        raise ValueError("PIV estimate and truth coordinates must be identical.")

    finite_vectors = jnp.all(jnp.isfinite(estimate.displacement_rc), axis=-1) & jnp.all(
        jnp.isfinite(truth.displacement_rc), axis=-1
    )
    valid = estimate.valid & truth.valid & finite_vectors
    truth_count = jnp.sum(truth.valid, dtype=jnp.int32)
    valid_count = jnp.sum(valid, dtype=jnp.int32)
    error = estimate.displacement_rc - truth.displacement_rc
    endpoint_error = jnp.sqrt(jnp.sum(error * error, axis=-1))
    bias = _masked_mean(error, valid)
    mean_epe = _masked_mean(endpoint_error, valid)
    root_mean_square_epe = jnp.sqrt(_masked_mean(endpoint_error * endpoint_error, valid))
    flattened = jnp.sort(jnp.where(valid, endpoint_error, jnp.inf).reshape((-1,)))
    percentile_index = jnp.clip(
        jnp.ceil(0.95 * valid_count).astype(jnp.int32) - 1,
        0,
        max(0, flattened.size - 1),
    )
    p95 = jnp.where(valid_count > 0, flattened[percentile_index], 0.0)
    coverage = _safe_ratio(valid_count, truth_count)
    finite = (valid_count > 0) & jnp.all(
        jnp.isfinite(
            jnp.concatenate(
                (
                    bias.reshape((-1,)),
                    jnp.stack((mean_epe, root_mean_square_epe, p95, coverage)),
                )
            )
        )
    )
    source_id = canonical_fingerprint(
        {
            "kind": "piv-qualification-source",
            "estimate": estimate.field_id,
            "truth": truth.field_id,
        }
    )
    evidence = (
        QualificationEvidence(
            "endpoint-error",
            jnp.stack((mean_epe, root_mean_square_epe, p95)),
            valid_count,
            finite=finite,
            status="computed",
            source_id=source_id,
        ),
        QualificationEvidence(
            "bias-rc",
            bias,
            valid_count,
            finite=finite,
            status="computed",
            source_id=source_id,
        ),
        QualificationEvidence(
            "coverage",
            coverage,
            truth_count,
            finite=jnp.isfinite(coverage),
            status="computed",
            source_id=source_id,
        ),
    )
    return PIVQualificationResult(
        bias,
        mean_epe,
        root_mean_square_epe,
        p95,
        coverage,
        valid_count,
        finite,
        evidence,
        source_id=source_id,
    )


class PTVQualificationResult(StrictModule, NonTrainableState):
    """Detection, triangulation, and trajectory-association metrics."""

    detection_precision: Array
    detection_recall: Array
    detection_f1: Array
    triangulation_bias_xyz: Array
    triangulation_root_mean_square_error: Array
    triangulation_coverage: Array
    track_identity_accuracy: Array
    track_completeness: Array
    matched_count: Array
    finite: Array
    evidence: tuple[QualificationEvidence, ...]
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        detection_precision: ArrayLike,
        detection_recall: ArrayLike,
        detection_f1: ArrayLike,
        triangulation_bias_xyz: ArrayLike,
        triangulation_root_mean_square_error: ArrayLike,
        triangulation_coverage: ArrayLike,
        track_identity_accuracy: ArrayLike,
        track_completeness: ArrayLike,
        matched_count: ArrayLike,
        finite: ArrayLike,
        evidence: Sequence[QualificationEvidence],
        /,
        *,
        source_id: str,
    ):
        evidence_ = tuple(evidence)
        if any(not isinstance(item, QualificationEvidence) for item in evidence_):
            raise TypeError("evidence must contain QualificationEvidence values.")
        self.detection_precision = jnp.asarray(detection_precision).reshape(())
        self.detection_recall = jnp.asarray(detection_recall).reshape(())
        self.detection_f1 = jnp.asarray(detection_f1).reshape(())
        self.triangulation_bias_xyz = jnp.asarray(triangulation_bias_xyz).reshape((3,))
        self.triangulation_root_mean_square_error = jnp.asarray(
            triangulation_root_mean_square_error
        ).reshape(())
        self.triangulation_coverage = jnp.asarray(triangulation_coverage).reshape(())
        self.track_identity_accuracy = jnp.asarray(track_identity_accuracy).reshape(())
        self.track_completeness = jnp.asarray(track_completeness).reshape(())
        self.matched_count = jnp.asarray(matched_count, dtype=jnp.int32).reshape(())
        self.finite = jnp.asarray(finite, dtype=bool).reshape(())
        self.evidence = evidence_
        self.result_id = canonical_fingerprint(
            {
                "kind": "ptv-qualification-result",
                "source": str(source_id),
                "evidence": [item.evidence_id for item in evidence_],
            }
        )


def qualify_ptv(
    reconstructed_xyz: ArrayLike,
    reconstructed_valid: ArrayLike,
    matched_truth_indices: ArrayLike,
    truth_xyz: ArrayLike,
    truth_valid: ArrayLike,
    /,
    *,
    reconstructed_track_ids: ArrayLike | None = None,
    truth_track_ids: ArrayLike | None = None,
    source_id: str = "ptv-evaluation",
) -> PTVQualificationResult:
    """Evaluate padded per-frame reconstructions against finite 3-D track truth.

    ``matched_truth_indices[t, i]`` names the truth slot associated with reconstructed
    slot ``i`` at frame ``t``; ``-1`` denotes an unmatched detection. Matching indices
    must be unique within each frame so detection evidence cannot be double counted.
    """
    reconstructed = jnp.asarray(reconstructed_xyz)
    reconstructed_mask = jnp.asarray(reconstructed_valid, dtype=bool)
    matched = jnp.asarray(matched_truth_indices, dtype=jnp.int32)
    truth = jnp.asarray(truth_xyz)
    truth_mask = jnp.asarray(truth_valid, dtype=bool)
    if reconstructed.ndim != 3 or reconstructed.shape[-1] != 3:
        raise ValueError("reconstructed_xyz must have shape (frames, capacity, 3).")
    if truth.ndim != 3 or truth.shape[-1] != 3:
        raise ValueError("truth_xyz must have shape (frames, capacity, 3).")
    if reconstructed.shape[0] != truth.shape[0]:
        raise ValueError("Reconstruction and truth frame counts must match.")
    if (
        reconstructed_mask.shape != reconstructed.shape[:-1]
        or matched.shape != reconstructed_mask.shape
    ):
        raise ValueError(
            "Reconstruction masks and match indices must match padded slots."
        )
    if truth_mask.shape != truth.shape[:-1]:
        raise ValueError("truth_valid must match truth_xyz slots.")
    truth_capacity = truth.shape[1]
    if bool(jnp.any((matched < -1) | (matched >= truth_capacity))):
        raise ValueError("matched_truth_indices contain an out-of-range truth slot.")
    for frame in range(reconstructed.shape[0]):
        assigned = matched[frame][reconstructed_mask[frame] & (matched[frame] >= 0)]
        if int(jnp.unique(assigned).size) != int(assigned.size):
            raise ValueError("Truth matches must be unique within each frame.")

    clipped = jnp.clip(matched, 0, max(0, truth_capacity - 1))
    frame_indices = jnp.arange(reconstructed.shape[0])[:, None]
    gathered_truth = truth[frame_indices, clipped]
    gathered_valid = truth_mask[frame_indices, clipped]
    finite_reconstruction = jnp.all(jnp.isfinite(reconstructed), axis=-1)
    finite_truth = jnp.all(jnp.isfinite(gathered_truth), axis=-1)
    true_positive = (
        reconstructed_mask
        & (matched >= 0)
        & gathered_valid
        & finite_reconstruction
        & finite_truth
    )
    predicted_count = jnp.sum(reconstructed_mask, dtype=jnp.int32)
    truth_count = jnp.sum(truth_mask, dtype=jnp.int32)
    matched_count = jnp.sum(true_positive, dtype=jnp.int32)
    precision = _safe_ratio(matched_count, predicted_count)
    recall = _safe_ratio(matched_count, truth_count)
    f1 = jnp.where(
        precision + recall > 0, 2.0 * precision * recall / (precision + recall), 0.0
    )
    error = reconstructed - gathered_truth
    bias = _masked_mean(error, true_positive)
    squared_distance = jnp.sum(error * error, axis=-1)
    triangulation_rmse = jnp.sqrt(_masked_mean(squared_distance, true_positive))
    triangulation_coverage = recall

    matched_one_hot = (
        jax.nn.one_hot(clipped, truth_capacity, dtype=bool) & true_positive[..., None]
    )
    truth_detected = jnp.any(matched_one_hot, axis=1)
    active_per_track = jnp.sum(truth_mask, axis=0)
    detected_per_track = jnp.sum(truth_detected & truth_mask, axis=0)
    track_has_truth = active_per_track > 0
    per_track_completeness = jnp.where(
        track_has_truth,
        detected_per_track / jnp.maximum(active_per_track, 1),
        0.0,
    )
    track_completeness = _masked_mean(per_track_completeness, track_has_truth)

    if (reconstructed_track_ids is None) != (truth_track_ids is None):
        raise ValueError(
            "Both reconstructed_track_ids and truth_track_ids are required together."
        )
    if reconstructed_track_ids is None:
        identity_accuracy = jnp.asarray(0.0)
        identity_support = jnp.asarray(0, dtype=jnp.int32)
        identity_status = "not-provided"
    else:
        reconstruction_ids = jnp.asarray(reconstructed_track_ids, dtype=jnp.int32)
        truth_ids = jnp.asarray(truth_track_ids, dtype=jnp.int32)
        if reconstruction_ids.shape != reconstructed_mask.shape:
            raise ValueError("reconstructed_track_ids must match reconstructed slots.")
        if truth_ids.shape != (truth_capacity,):
            raise ValueError("truth_track_ids must have shape (truth_capacity,).")
        gathered_ids = truth_ids[clipped]
        identity_support = matched_count
        identity_accuracy = _safe_ratio(
            jnp.sum(true_positive & (reconstruction_ids == gathered_ids)),
            identity_support,
        )
        identity_status = "computed"

    metric_values = jnp.concatenate(
        (
            jnp.stack(
                (
                    precision,
                    recall,
                    f1,
                    triangulation_rmse,
                    triangulation_coverage,
                    identity_accuracy,
                    track_completeness,
                )
            ),
            bias,
        )
    )
    finite = (matched_count > 0) & jnp.all(jnp.isfinite(metric_values))
    source = str(source_id)
    if not source:
        raise ValueError("source_id must be non-empty.")
    evidence = (
        QualificationEvidence(
            "detection",
            jnp.stack((precision, recall, f1)),
            truth_count,
            finite=jnp.all(jnp.isfinite(jnp.stack((precision, recall, f1)))),
            status="computed",
            source_id=source,
        ),
        QualificationEvidence(
            "triangulation",
            jnp.concatenate(
                (bias, jnp.stack((triangulation_rmse, triangulation_coverage)))
            ),
            matched_count,
            finite=jnp.all(
                jnp.isfinite(jnp.concatenate((bias, triangulation_rmse[None])))
            ),
            status="computed",
            source_id=source,
        ),
        QualificationEvidence(
            "track-identity",
            identity_accuracy,
            identity_support,
            finite=jnp.isfinite(identity_accuracy),
            status=identity_status,
            source_id=source,
        ),
        QualificationEvidence(
            "track-completeness",
            track_completeness,
            jnp.sum(track_has_truth),
            finite=jnp.isfinite(track_completeness),
            status="computed",
            source_id=source,
        ),
    )
    return PTVQualificationResult(
        precision,
        recall,
        f1,
        bias,
        triangulation_rmse,
        triangulation_coverage,
        identity_accuracy,
        track_completeness,
        matched_count,
        finite,
        evidence,
        source_id=source,
    )


class STBQualificationResult(StrictModule, NonTrainableState):
    """Image-space Shake-The-Box reconstruction residual evidence."""

    residual_bias: Array
    residual_mean_absolute_error: Array
    residual_root_mean_square_error: Array
    explained_energy: Array
    coverage: Array
    valid_count: Array
    finite: Array
    evidence: tuple[QualificationEvidence, ...]
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual_bias: ArrayLike,
        residual_mean_absolute_error: ArrayLike,
        residual_root_mean_square_error: ArrayLike,
        explained_energy: ArrayLike,
        coverage: ArrayLike,
        valid_count: ArrayLike,
        finite: ArrayLike,
        evidence: Sequence[QualificationEvidence],
        /,
        *,
        source_id: str,
    ):
        evidence_ = tuple(evidence)
        if any(not isinstance(item, QualificationEvidence) for item in evidence_):
            raise TypeError("evidence must contain QualificationEvidence values.")
        self.residual_bias = jnp.asarray(residual_bias).reshape(())
        self.residual_mean_absolute_error = jnp.asarray(
            residual_mean_absolute_error
        ).reshape(())
        self.residual_root_mean_square_error = jnp.asarray(
            residual_root_mean_square_error
        ).reshape(())
        self.explained_energy = jnp.asarray(explained_energy).reshape(())
        self.coverage = jnp.asarray(coverage).reshape(())
        self.valid_count = jnp.asarray(valid_count, dtype=jnp.int32).reshape(())
        self.finite = jnp.asarray(finite, dtype=bool).reshape(())
        self.evidence = evidence_
        self.result_id = canonical_fingerprint(
            {
                "kind": "stb-qualification-result",
                "source": str(source_id),
                "evidence": [item.evidence_id for item in evidence_],
            }
        )


def qualify_stb(
    reconstructed_images: ArrayLike,
    observed_images: ArrayLike,
    /,
    *,
    valid_mask: ArrayLike | None = None,
    source_id: str = "stb-evaluation",
) -> STBQualificationResult:
    """Measure masked multiview, multiframe image reconstruction residuals."""
    reconstructed = jnp.asarray(reconstructed_images)
    observed = jnp.asarray(observed_images)
    if reconstructed.shape != observed.shape or reconstructed.ndim < 2:
        raise ValueError(
            "STB reconstructed and observed image stacks must share a shape."
        )
    if valid_mask is None:
        requested_valid = jnp.ones(observed.shape, dtype=bool)
    else:
        requested_valid = jnp.asarray(valid_mask, dtype=bool)
        if requested_valid.shape != observed.shape:
            raise ValueError("valid_mask must exactly match the STB image stack shape.")
    finite_pixels = jnp.isfinite(reconstructed) & jnp.isfinite(observed)
    valid = requested_valid & finite_pixels
    valid_count = jnp.sum(valid, dtype=jnp.int32)
    requested_count = jnp.sum(requested_valid, dtype=jnp.int32)
    residual = reconstructed - observed
    bias = _masked_mean(residual, valid)
    mean_absolute_error = _masked_mean(jnp.abs(residual), valid)
    residual_energy = _masked_mean(residual * residual, valid)
    root_mean_square_error = jnp.sqrt(residual_energy)
    observed_energy = _masked_mean(observed * observed, valid)
    explained_energy = jnp.where(
        observed_energy > 0,
        1.0 - residual_energy / observed_energy,
        jnp.where(residual_energy == 0, 1.0, 0.0),
    )
    coverage = _safe_ratio(valid_count, requested_count)
    metrics = jnp.stack(
        (bias, mean_absolute_error, root_mean_square_error, explained_energy, coverage)
    )
    finite = (valid_count > 0) & jnp.all(jnp.isfinite(metrics))
    source = str(source_id)
    if not source:
        raise ValueError("source_id must be non-empty.")
    evidence = (
        QualificationEvidence(
            "stb-image-residual",
            metrics[:4],
            valid_count,
            finite=finite,
            status="computed",
            source_id=source,
        ),
        QualificationEvidence(
            "coverage",
            coverage,
            requested_count,
            finite=jnp.isfinite(coverage),
            status="computed",
            source_id=source,
        ),
    )
    return STBQualificationResult(
        bias,
        mean_absolute_error,
        root_mean_square_error,
        explained_energy,
        coverage,
        valid_count,
        finite,
        evidence,
        source_id=source,
    )


__all__ = [
    "PIVQualificationResult",
    "PTVQualificationResult",
    "QualificationEvidence",
    "STBQualificationResult",
    "qualify_piv",
    "qualify_ptv",
    "qualify_stb",
]
