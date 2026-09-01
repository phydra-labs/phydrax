#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


CELL_QC_SUCCESS = 0
CELL_QC_EMPTY_SAMPLE = 1
CELL_QC_NONFINITE = 2
CELL_QC_NO_ACCEPTED_CELLS = 3
CELL_QC_INVALID_SAMPLE_INDEX = 4


def cell_qc_status_name(status: int, /) -> str:
    """Return the stable name of a sample-level cell-QC status code."""
    names = (
        "success",
        "empty_sample",
        "nonfinite_cell_measurement",
        "no_accepted_cells",
        "invalid_sample_index",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown cell-QC status {code}.")
    return names[code]


def _qc_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "sample_cell_quality_control",
        MethodKind.HEURISTIC,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Threshold decisions are conditioned on the supplied per-cell count, "
            "feature, mitochondrial, and doublet summaries."
        ),
        truncation_statement="No cells or samples are truncated.",
        capacity_semantics="The declared sample count is exact and all cells are retained.",
        assumptions=(
            "Each cell belongs to one biological sample.",
            "Mitochondrial counts are a subset of total counts.",
        ),
        nondifferentiable_outputs=("accepted", "decision_reason_counts", "status"),
    )


class CellQCThresholds(StrictModule):
    """Fixed sample-aware thresholds for per-cell descriptive QC decisions."""

    minimum_total_counts: float = eqx.field(static=True)
    minimum_detected_features: int = eqx.field(static=True)
    maximum_mitochondrial_fraction: float = eqx.field(static=True)
    maximum_doublet_score: float = eqx.field(static=True)
    minimum_accepted_cells_per_sample: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_total_counts: float = 1.0,
        minimum_detected_features: int = 1,
        maximum_mitochondrial_fraction: float = 1.0,
        maximum_doublet_score: float = 1.0,
        minimum_accepted_cells_per_sample: int = 1,
    ):
        minimum_counts = float(minimum_total_counts)
        minimum_features = int(minimum_detected_features)
        maximum_mitochondrial = float(maximum_mitochondrial_fraction)
        maximum_doublet = float(maximum_doublet_score)
        minimum_cells = int(minimum_accepted_cells_per_sample)
        if (
            not math.isfinite(minimum_counts)
            or minimum_counts < 0.0
            or minimum_features < 0
            or not math.isfinite(maximum_mitochondrial)
            or maximum_mitochondrial < 0.0
            or maximum_mitochondrial > 1.0
            or not math.isfinite(maximum_doublet)
            or maximum_doublet < 0.0
            or minimum_cells < 1
        ):
            raise ValueError("Cell-QC thresholds must be finite and physically valid.")
        self.minimum_total_counts = minimum_counts
        self.minimum_detected_features = minimum_features
        self.maximum_mitochondrial_fraction = maximum_mitochondrial
        self.maximum_doublet_score = maximum_doublet
        self.minimum_accepted_cells_per_sample = minimum_cells


class SampleCellQCEvidence(StrictModule):
    """Auditable sample-level evidence behind every QC decision."""

    input_indices_valid: Array
    cells_observed: Array
    cells_accepted: Array
    decision_reason_counts: Array
    thresholds: CellQCThresholds = eqx.field(static=True)
    decision_reason_names: tuple[str, ...] = eqx.field(static=True)
    replicate_unit: str = eqx.field(static=True)


class SampleCellQCSummary(StrictModule):
    """Sample-level summaries; cells are never treated as independent replicates."""

    mean_total_counts: Array
    mean_detected_features: Array
    mean_mitochondrial_fraction: Array
    mean_doublet_score: Array
    accepted_fraction: Array
    accepted: Array
    valid: Array
    status: Array
    evidence: SampleCellQCEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def summarize_cell_qc(
    total_counts: ArrayLike,
    detected_features: ArrayLike,
    mitochondrial_counts: ArrayLike,
    doublet_score: ArrayLike,
    sample_index: ArrayLike,
    /,
    *,
    sample_count: int,
    thresholds: CellQCThresholds | None = None,
    method_contract: BioinformaticsMethodContract | None = None,
) -> SampleCellQCSummary:
    """Aggregate cell-level QC into decision-bearing biological-sample summaries."""
    totals = jnp.asarray(total_counts)
    detected = jnp.asarray(detected_features)
    mitochondrial = jnp.asarray(mitochondrial_counts)
    doublet = jnp.asarray(doublet_score)
    samples = jnp.asarray(sample_index)
    count = int(sample_count)
    if count < 1:
        raise ValueError("sample_count must be positive.")
    if any(
        value.ndim != 1 for value in (totals, detected, mitochondrial, doublet, samples)
    ):
        raise ValueError("Cell-QC inputs must be rank-one arrays.")
    if not (
        totals.shape
        == detected.shape
        == mitochondrial.shape
        == doublet.shape
        == samples.shape
    ):
        raise ValueError("Cell-QC inputs must contain the same number of cells.")
    if not jnp.issubdtype(samples.dtype, jnp.integer):
        raise TypeError("sample_index must have an integer dtype.")
    if not jnp.issubdtype(detected.dtype, jnp.integer):
        raise TypeError("detected_features must have an integer dtype.")

    plan = thresholds if thresholds is not None else CellQCThresholds()
    contract = method_contract if method_contract is not None else _qc_contract()
    samples = samples.astype(jnp.int32)
    in_bounds = (samples >= 0) & (samples < count)
    safe_samples = jnp.where(in_bounds, samples, 0)
    finite = jnp.isfinite(totals) & jnp.isfinite(mitochondrial) & jnp.isfinite(doublet)
    physical = (
        (totals >= 0) & (detected >= 0) & (mitochondrial >= 0) & (mitochondrial <= totals)
    )
    measurable = in_bounds & finite & physical
    safe_total = jnp.where(measurable, totals, 0.0)
    safe_detected = jnp.where(measurable, detected, 0)
    mitochondrial_fraction = jnp.where(
        measurable & (totals > 0),
        mitochondrial / jnp.where(totals > 0, totals, 1.0),
        0.0,
    )
    safe_doublet = jnp.where(measurable, doublet, 0.0)

    low_counts = measurable & (totals < plan.minimum_total_counts)
    low_features = measurable & (detected < plan.minimum_detected_features)
    high_mitochondrial = measurable & (
        mitochondrial_fraction > plan.maximum_mitochondrial_fraction
    )
    high_doublet = measurable & (doublet > plan.maximum_doublet_score)
    invalid_measurement = in_bounds & ~(finite & physical)
    accepted_cell = measurable & ~(
        low_counts | low_features | high_mitochondrial | high_doublet
    )

    observed_cells = (
        jnp.zeros((count,), dtype=jnp.int32)
        .at[safe_samples]
        .add(in_bounds.astype(jnp.int32))
    )
    accepted_cells = (
        jnp.zeros((count,), dtype=jnp.int32)
        .at[safe_samples]
        .add(accepted_cell.astype(jnp.int32))
    )
    denominator = jnp.maximum(observed_cells, 1)
    total_sum = (
        jnp.zeros((count,), dtype=safe_total.dtype).at[safe_samples].add(safe_total)
    )
    feature_sum = (
        jnp.zeros((count,), dtype=safe_total.dtype)
        .at[safe_samples]
        .add(safe_detected.astype(safe_total.dtype))
    )
    mito_sum = (
        jnp.zeros((count,), dtype=safe_total.dtype)
        .at[safe_samples]
        .add(mitochondrial_fraction)
    )
    doublet_sum = (
        jnp.zeros((count,), dtype=safe_total.dtype).at[safe_samples].add(safe_doublet)
    )
    reason_masks = jnp.stack(
        (
            low_counts,
            low_features,
            high_mitochondrial,
            high_doublet,
            invalid_measurement,
        ),
        axis=-1,
    )
    reason_counts = (
        jnp.zeros((count, 5), dtype=jnp.int32)
        .at[safe_samples]
        .add(reason_masks.astype(jnp.int32))
    )
    indices_valid = jnp.all(in_bounds)
    sample_has_nonfinite = reason_counts[:, 4] > 0
    sample_accepted = accepted_cells >= plan.minimum_accepted_cells_per_sample
    valid = indices_valid & (observed_cells > 0) & ~sample_has_nonfinite & sample_accepted
    status = jnp.where(
        ~indices_valid,
        CELL_QC_INVALID_SAMPLE_INDEX,
        jnp.where(
            observed_cells == 0,
            CELL_QC_EMPTY_SAMPLE,
            jnp.where(
                sample_has_nonfinite,
                CELL_QC_NONFINITE,
                jnp.where(sample_accepted, CELL_QC_SUCCESS, CELL_QC_NO_ACCEPTED_CELLS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = SampleCellQCEvidence(
        indices_valid,
        observed_cells,
        accepted_cells,
        reason_counts,
        plan,
        (
            "low_total_counts",
            "low_detected_features",
            "high_mitochondrial_fraction",
            "high_doublet_score",
            "nonfinite_or_physical_invalidity",
        ),
        "sample",
    )
    return SampleCellQCSummary(
        total_sum / denominator,
        feature_sum / denominator,
        mito_sum / denominator,
        doublet_sum / denominator,
        accepted_cells / denominator,
        sample_accepted,
        valid,
        status,
        evidence,
        contract,
        "heuristic_qc_decision",
    )


__all__ = [
    "CELL_QC_EMPTY_SAMPLE",
    "CELL_QC_INVALID_SAMPLE_INDEX",
    "CELL_QC_NONFINITE",
    "CELL_QC_NO_ACCEPTED_CELLS",
    "CELL_QC_SUCCESS",
    "CellQCThresholds",
    "SampleCellQCEvidence",
    "SampleCellQCSummary",
    "cell_qc_status_name",
    "summarize_cell_qc",
]
