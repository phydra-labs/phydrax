#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..imaging import DenseDisplacementField2D


class PIVPreparationReport(StrictModule, NonTrainableState):
    """Auditable shapes, capacities, retention, and numeric resolution."""

    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    image_shape: tuple[int, int] = eqx.field(static=True)
    grid_shapes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    window_counts: tuple[int, ...] = eqx.field(static=True)
    padded_window_counts: tuple[int, ...] = eqx.field(static=True)
    maximum_working_bytes: int = eqx.field(static=True)
    resource_limit_bytes: int = eqx.field(static=True)
    requested_compute_dtype: str = eqx.field(static=True)
    resolved_compute_dtype: str = eqx.field(static=True)
    fft_complex_dtype: str = eqx.field(static=True)
    correlation_mode: str = eqx.field(static=True)
    retained_correlation: bool = eqx.field(static=True)


class WindowGrid2D(StrictModule, NonTrainableState):
    """Prepared fixed grid of interrogation-window centers."""

    centers_rc: Array
    active: Array
    grid_shape: tuple[int, int] = eqx.field(static=True)
    window_size: tuple[int, int] = eqx.field(static=True)
    search_margin: tuple[int, int] = eqx.field(static=True)
    spacing: tuple[int, int] = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


class CorrelationBatch(StrictModule):
    """Fixed-shape mask-aware correlation surfaces and overlap evidence."""

    values: Array
    overlap: Array
    valid: Array
    lags_rc: Array
    mode: str = eqx.field(static=True)


class PeakBatch(StrictModule):
    """Deterministically ordered peak candidates and local-fit evidence."""

    offsets_rc: Array
    values: Array
    valid: Array
    curvature_rc: Array
    covariance_rc: Array
    method: str = eqx.field(static=True)


class PIVQuality2D(StrictModule):
    """Signal-quality metrics; these are not uncertainty estimates."""

    primary_peak: Array
    secondary_peak: Array
    peak_ratio: Array
    peak_to_rms: Array
    overlap_fraction: Array


class PIVUncertainty2D(StrictModule):
    """Local displacement covariance inferred from peak curvature."""

    covariance_rc: Array
    valid: Array
    method: str = eqx.field(static=True)


class ValidationEvidence2D(StrictModule):
    """Per-vector evidence retained for every validation decision."""

    finite: Array
    within_displacement_limit: Array
    correlation_accepted: Array
    peak_ratio_accepted: Array
    local_consistency_accepted: Array
    neighbor_count: Array
    local_median_rc: Array
    local_residual: Array
    local_threshold: Array
    valid: Array


class ReplacementEvidence2D(StrictModule):
    """Per-vector provenance for non-mutating neighborhood replacement."""

    originally_valid: Array
    replaced: Array
    replacement_iteration: Array
    contributing_neighbors: Array
    unresolved: Array


class PIVStatus2D(StrictModule):
    """JIT-safe terminal status bits for every fixed-capacity vector."""

    code: Array
    correlated: Array
    peak_fitted: Array
    validated: Array
    replaced: Array


class PIVRetention(StrictModule, NonTrainableState):
    """Optional retained final-pass correlation state and numeric provenance."""

    correlation: Array
    overlap: Array
    lags_rc: Array
    retained: bool = eqx.field(static=True)
    pair_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    requested_compute_dtype: str = eqx.field(static=True)
    resolved_compute_dtype: str = eqx.field(static=True)
    fft_complex_dtype: str = eqx.field(static=True)


class PIVResult(StrictModule, NonTrainableState):
    """Raw, validated, and replaced fields with quality and decision evidence."""

    raw: DenseDisplacementField2D
    validated: DenseDisplacementField2D
    replaced: DenseDisplacementField2D
    quality: PIVQuality2D
    uncertainty: PIVUncertainty2D
    validation_evidence: ValidationEvidence2D
    replacement_evidence: ReplacementEvidence2D
    status: PIVStatus2D
    retention: PIVRetention
    pair_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    requested_compute_dtype: str = eqx.field(static=True)
    resolved_compute_dtype: str = eqx.field(static=True)
    fft_complex_dtype: str = eqx.field(static=True)

    @property
    def field(self) -> DenseDisplacementField2D:
        """Return the replacement-complete field without hiding earlier stages."""
        return self.replaced


class PhysicalPIVResult2D(StrictModule, NonTrainableState):
    """Right-handed physical ``(x, y)`` displacement and velocity vectors."""

    positions_xy: Array
    displacement_xy: Array
    velocity_xy: Array
    valid: Array
    source_field_id: str = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)
    spatial_unit: str = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)


class EnsemblePIVAccumulator(StrictModule, NonTrainableState):
    """Mergeable fixed-shape sums of retained correlation evidence."""

    correlation_sum: Array
    overlap_sum: Array
    sample_count: Array
    valid_count: Array
    lags_rc: Array
    prepared_id: str = eqx.field(static=True)
    resolved_compute_dtype: str = eqx.field(static=True)


class ResidualDisparityDiagnostics2D(StrictModule):
    """Brightness residual after applying one measured displacement field."""

    residual: Array
    valid: Array
    absolute_residual: Array
    squared_residual: Array
    valid_fraction: Array
    mean_residual: Array
    root_mean_square: Array
    maximum_absolute: Array
    source_field_id: str = eqx.field(static=True)


__all__ = [
    "CorrelationBatch",
    "EnsemblePIVAccumulator",
    "PIVPreparationReport",
    "PIVQuality2D",
    "PIVResult",
    "PIVRetention",
    "PIVStatus2D",
    "PIVUncertainty2D",
    "PeakBatch",
    "PhysicalPIVResult2D",
    "ReplacementEvidence2D",
    "ResidualDisparityDiagnostics2D",
    "ValidationEvidence2D",
    "WindowGrid2D",
]
