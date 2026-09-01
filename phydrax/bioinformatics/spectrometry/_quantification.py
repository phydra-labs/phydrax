#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class QuantificationSampleKind(IntEnum):
    """Experimental role of one spectrometry injection."""

    BIOLOGICAL = 0
    BLANK = 1
    QUALITY_CONTROL = 2


class QuantificationStatus(IntEnum):
    """Status of bounded replicate-aware quantification."""

    SUCCESS = 0
    CAPACITY_EXCEEDED = 1
    NONFINITE = 2
    NO_BIOLOGICAL_SAMPLES = 3


class QuantificationEvidence(IntFlag):
    """Corrections and missing-data evidence retained by quantification."""

    NONE = 0
    BLANK_CORRECTED = 1
    RUN_ORDER_CORRECTED = 2
    CENSORED_INTERVAL = 4
    MISSING_VALUES = 8
    REPLICATE_SUMMARY = 16
    INSUFFICIENT_BLANKS = 32
    INSUFFICIENT_QUALITY_CONTROLS = 64


_QUANTIFICATION_CONTRACT = BioinformaticsMethodContract(
    "replicate-aware censored spectrometry quantification",
    MethodKind.APPROXIMATE_MODEL,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Blank subtraction uses uncensored blanks; optional log-linear drift uses "
        "uncensored QC injections; replicate intervals propagate detection-limit "
        "censoring."
    ),
    truncation_statement=(
        "No feature or sample is dropped. Missing, censored, and observed-zero "
        "states remain distinct."
    ),
    capacity_semantics="Feature, sample, and replicate-group capacities are fixed by the batch and plan.",
    assumptions=(
        "QC drift is approximately log-linear over run order.",
        "Detection limits bound censored signals after correction.",
    ),
    nondifferentiable_outputs=("masks", "counts", "status", "evidence"),
)


class QuantificationBatch(StrictModule):
    """Feature-by-injection values with explicit missing and censored states."""

    feature_ids: Array
    sample_ids: Array
    intensity: Array
    detection_limit: Array
    present_mask: Array
    censored_mask: Array
    feature_mask: Array
    sample_mask: Array
    sample_kind: Array
    run_order: Array
    replicate_ids: Array
    batch_ids: Array
    feature_capacity: int = eqx.field(static=True)
    sample_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        feature_ids: ArrayLike,
        sample_ids: ArrayLike,
        intensity: ArrayLike,
        detection_limit: ArrayLike,
        present_mask: ArrayLike,
        censored_mask: ArrayLike,
        feature_mask: ArrayLike,
        sample_mask: ArrayLike,
        sample_kind: ArrayLike,
        run_order: ArrayLike,
        replicate_ids: ArrayLike,
        batch_ids: ArrayLike,
        /,
    ):
        features = np.asarray(feature_ids)
        samples = np.asarray(sample_ids)
        values = np.asarray(intensity)
        limits = np.asarray(detection_limit)
        present = np.asarray(present_mask, dtype=bool)
        censored = np.asarray(censored_mask, dtype=bool)
        active_features = np.asarray(feature_mask, dtype=bool)
        active_samples = np.asarray(sample_mask, dtype=bool)
        roles = np.asarray(sample_kind)
        order = np.asarray(run_order)
        replicates = np.asarray(replicate_ids)
        batches = np.asarray(batch_ids)
        if (
            features.ndim != 1
            or features.size == 0
            or samples.ndim != 1
            or samples.size == 0
        ):
            raise ValueError("Feature and sample identifiers must be non-empty vectors.")
        expected = (features.size, samples.size)
        if any(value.shape != expected for value in (values, limits, present, censored)):
            raise ValueError(
                "Measurement arrays must have shape (feature_capacity, sample_capacity)."
            )
        if (
            active_features.shape != features.shape
            or active_samples.shape != samples.shape
        ):
            raise ValueError("Feature and sample masks must match their identifiers.")
        if any(
            value.shape != samples.shape for value in (roles, order, replicates, batches)
        ):
            raise ValueError("Sample metadata must match sample_ids.")
        if not all(
            np.issubdtype(value.dtype, np.integer)
            for value in (features, samples, roles, order, replicates, batches)
        ):
            raise TypeError(
                "Identifiers, sample roles, run order, and groups must be integers."
            )
        feature_count = int(np.count_nonzero(active_features))
        sample_count = int(np.count_nonzero(active_samples))
        if not np.all(active_features[:feature_count]) or np.any(
            active_features[feature_count:]
        ):
            raise ValueError("feature_mask must be a left-prefix mask.")
        if not np.all(active_samples[:sample_count]) or np.any(
            active_samples[sample_count:]
        ):
            raise ValueError("sample_mask must be a left-prefix mask.")
        active_cells = active_features[:, None] & active_samples[None, :]
        if np.any(present & ~active_cells) or np.any(censored & ~present):
            raise ValueError(
                "Present cells require active axes and censored cells must be present."
            )
        observed = present & ~censored
        if np.any(~np.isfinite(values[observed])) or np.any(values[observed] < 0.0):
            raise ValueError("Observed intensities must be finite and nonnegative.")
        if np.any(values[~observed] != 0.0):
            raise ValueError(
                "Missing and censored intensities must use zero payload with explicit masks."
            )
        if np.any(~np.isfinite(limits[present])) or np.any(limits[present] < 0.0):
            raise ValueError(
                "Present-cell detection limits must be finite and nonnegative."
            )
        if np.any(limits[~present] != 0.0):
            raise ValueError("Missing-cell detection limits must be zero padding.")
        valid_roles = np.isin(
            roles[active_samples],
            [
                int(QuantificationSampleKind.BIOLOGICAL),
                int(QuantificationSampleKind.BLANK),
                int(QuantificationSampleKind.QUALITY_CONTROL),
            ],
        )
        if not np.all(valid_roles):
            raise ValueError("Active sample_kind contains an unsupported value.")
        if (
            np.any(order[active_samples] < 0)
            or np.unique(order[active_samples]).size != sample_count
        ):
            raise ValueError("Active run_order values must be unique and nonnegative.")
        biological = active_samples & (roles == int(QuantificationSampleKind.BIOLOGICAL))
        if np.any(replicates[biological] < 0):
            raise ValueError("Biological replicate identifiers must be nonnegative.")
        if np.any(batches[active_samples] < 0):
            raise ValueError("Active batch identifiers must be nonnegative.")
        if (
            np.unique(features[active_features]).size != feature_count
            or np.unique(samples[active_samples]).size != sample_count
        ):
            raise ValueError("Active feature and sample identifiers must be unique.")
        for value in (features, samples):
            mask = active_features if value is features else active_samples
            if np.any(value[~mask] != 0):
                raise ValueError("Inactive identifiers must be zero padding.")
        for value in (roles, order, replicates, batches):
            if np.any(value[~active_samples] != 0):
                raise ValueError("Inactive sample metadata must be zero padding.")
        self.feature_ids = jnp.asarray(features, dtype=jnp.int64)
        self.sample_ids = jnp.asarray(samples, dtype=jnp.int64)
        self.intensity = jnp.asarray(values)
        self.detection_limit = jnp.asarray(limits)
        self.present_mask = jnp.asarray(present)
        self.censored_mask = jnp.asarray(censored)
        self.feature_mask = jnp.asarray(active_features)
        self.sample_mask = jnp.asarray(active_samples)
        self.sample_kind = jnp.asarray(roles, dtype=jnp.int32)
        self.run_order = jnp.asarray(order, dtype=jnp.int32)
        self.replicate_ids = jnp.asarray(replicates, dtype=jnp.int64)
        self.batch_ids = jnp.asarray(batches, dtype=jnp.int64)
        self.feature_capacity = int(features.size)
        self.sample_capacity = int(samples.size)


class QuantificationPlan(StrictModule):
    """Bounded replicate identities and blank/QC correction policy."""

    replicate_group_ids: Array
    group_mask: Array
    minimum_blank_count: int = eqx.field(static=True)
    minimum_qc_count: int = eqx.field(static=True)
    correct_run_order: bool = eqx.field(static=True)
    group_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        replicate_group_ids: ArrayLike,
        group_mask: ArrayLike,
        /,
        *,
        minimum_blank_count: int = 1,
        minimum_qc_count: int = 2,
        correct_run_order: bool = True,
    ):
        groups = np.asarray(replicate_group_ids)
        mask = np.asarray(group_mask, dtype=bool)
        if groups.ndim != 1 or groups.size == 0 or mask.shape != groups.shape:
            raise ValueError("Replicate groups and mask must be equal non-empty vectors.")
        if not np.issubdtype(groups.dtype, np.integer):
            raise TypeError("replicate_group_ids must contain integers.")
        count = int(np.count_nonzero(mask))
        if not np.all(mask[:count]) or np.any(mask[count:]):
            raise ValueError("group_mask must be a left-prefix mask.")
        if np.any(groups[mask] < 0) or np.unique(groups[mask]).size != count:
            raise ValueError(
                "Active replicate group identifiers must be unique and nonnegative."
            )
        if np.any(groups[~mask] != 0):
            raise ValueError("Inactive replicate group identifiers must be zero padding.")
        blanks = int(minimum_blank_count)
        qcs = int(minimum_qc_count)
        if blanks < 0 or qcs < 2:
            raise ValueError(
                "minimum_blank_count must be nonnegative and minimum_qc_count at least two."
            )
        self.replicate_group_ids = jnp.asarray(groups, dtype=jnp.int64)
        self.group_mask = jnp.asarray(mask)
        self.minimum_blank_count = blanks
        self.minimum_qc_count = qcs
        self.correct_run_order = bool(correct_run_order)
        self.group_capacity = int(groups.size)


class QuantificationResult(StrictModule):
    """Corrected cell values and replicate summaries with finite-state missingness."""

    corrected_intensity: Array
    lower_bound: Array
    upper_bound: Array
    blank_estimate: Array
    drift_factor: Array
    point_estimate_mask: Array
    censored_mask: Array
    missing_mask: Array
    replicate_mean: Array
    replicate_lower_bound: Array
    replicate_upper_bound: Array
    replicate_observed_count: Array
    replicate_censored_count: Array
    replicate_missing_count: Array
    replicate_group_mask: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def quantify_replicates(
    batch: QuantificationBatch,
    plan: QuantificationPlan,
    /,
) -> QuantificationResult:
    """Blank-correct, drift-correct, and summarize replicates without zero imputation."""
    if not isinstance(batch, QuantificationBatch):
        raise TypeError("batch must be QuantificationBatch.")
    if not isinstance(plan, QuantificationPlan):
        raise TypeError("plan must be QuantificationPlan.")
    biological_samples = batch.sample_mask & (
        batch.sample_kind == int(QuantificationSampleKind.BIOLOGICAL)
    )
    represented = jnp.any(
        (batch.replicate_ids[:, None] == plan.replicate_group_ids[None, :])
        & plan.group_mask[None, :],
        axis=1,
    )
    capacity_ok = ~jnp.any(biological_samples & ~represented)

    observed = batch.present_mask & ~batch.censored_mask
    blank_samples = batch.sample_mask & (
        batch.sample_kind == int(QuantificationSampleKind.BLANK)
    )
    qc_samples = batch.sample_mask & (
        batch.sample_kind == int(QuantificationSampleKind.QUALITY_CONTROL)
    )
    blank_cells = observed & blank_samples[None, :] & batch.feature_mask[:, None]
    blank_count = jnp.sum(blank_cells, axis=1, dtype=jnp.int32)
    blank_estimate = jnp.sum(
        jnp.where(blank_cells, batch.intensity, 0.0), axis=1
    ) / jnp.maximum(blank_count, 1)
    blank_estimate = jnp.where(blank_count > 0, blank_estimate, 0.0)
    blank_corrected = jnp.maximum(batch.intensity - blank_estimate[:, None], 0.0)
    limit_corrected = jnp.maximum(batch.detection_limit - blank_estimate[:, None], 0.0)

    qc_cells = observed & qc_samples[None, :] & batch.feature_mask[:, None]
    qc_count = jnp.sum(qc_cells, axis=1, dtype=jnp.int32)
    order = batch.run_order.astype(batch.intensity.dtype)
    order_mean = jnp.sum(qc_cells * order[None, :], axis=1) / jnp.maximum(qc_count, 1)
    centered_order = order[None, :] - order_mean[:, None]
    log_signal = jnp.log1p(blank_corrected)
    signal_mean = jnp.sum(jnp.where(qc_cells, log_signal, 0.0), axis=1) / jnp.maximum(
        qc_count, 1
    )
    numerator = jnp.sum(
        jnp.where(
            qc_cells,
            centered_order * (log_signal - signal_mean[:, None]),
            0.0,
        ),
        axis=1,
    )
    denominator = jnp.sum(
        jnp.where(qc_cells, centered_order * centered_order, 0.0), axis=1
    )
    drift_available = (
        (qc_count >= plan.minimum_qc_count)
        & (denominator > jnp.finfo(batch.intensity.dtype).eps)
        & batch.feature_mask
        & plan.correct_run_order
    )
    slope = jnp.where(drift_available, numerator / jnp.maximum(denominator, 1.0), 0.0)
    drift_factor = jnp.exp(slope[:, None] * centered_order)
    corrected_values = blank_corrected / drift_factor
    corrected_limits = limit_corrected / drift_factor

    point_mask = observed & batch.feature_mask[:, None] & batch.sample_mask[None, :]
    censored = (
        batch.censored_mask & batch.feature_mask[:, None] & batch.sample_mask[None, :]
    )
    missing = (
        (~batch.present_mask) & batch.feature_mask[:, None] & batch.sample_mask[None, :]
    )
    nan = jnp.asarray(jnp.nan, dtype=batch.intensity.dtype)
    point_values = jnp.where(point_mask, corrected_values, nan)
    lower = jnp.where(point_mask, corrected_values, jnp.where(censored, 0.0, nan))
    upper = jnp.where(
        point_mask, corrected_values, jnp.where(censored, corrected_limits, nan)
    )

    membership = (
        biological_samples[None, :]
        & plan.group_mask[:, None]
        & (plan.replicate_group_ids[:, None] == batch.replicate_ids[None, :])
    )
    observed_group = point_mask[:, None, :] & membership[None, :, :]
    censored_group = censored[:, None, :] & membership[None, :, :]
    missing_group = missing[:, None, :] & membership[None, :, :]
    observed_count = jnp.sum(observed_group, axis=-1, dtype=jnp.int32)
    censored_count = jnp.sum(censored_group, axis=-1, dtype=jnp.int32)
    missing_count = jnp.sum(missing_group, axis=-1, dtype=jnp.int32)
    measured_count = observed_count + censored_count
    observed_sum = jnp.sum(
        jnp.where(observed_group, corrected_values[:, None, :], 0.0), axis=-1
    )
    censored_upper_sum = jnp.sum(
        jnp.where(censored_group, corrected_limits[:, None, :], 0.0), axis=-1
    )
    replicate_mean = jnp.where(
        observed_count > 0, observed_sum / jnp.maximum(observed_count, 1), nan
    )
    bounded = (measured_count > 0) & (missing_count == 0)
    replicate_lower = jnp.where(
        bounded, observed_sum / jnp.maximum(measured_count, 1), nan
    )
    replicate_upper = jnp.where(
        bounded,
        (observed_sum + censored_upper_sum) / jnp.maximum(measured_count, 1),
        nan,
    )

    finite = jnp.all(
        jnp.isfinite(jnp.where(point_mask, corrected_values, 0.0))
    ) & jnp.all(jnp.isfinite(jnp.where(censored, corrected_limits, 0.0)))
    has_biological = jnp.any(biological_samples)
    valid = capacity_ok & finite & has_biological
    status = jnp.where(
        ~capacity_ok,
        int(QuantificationStatus.CAPACITY_EXCEEDED),
        jnp.where(
            ~has_biological,
            int(QuantificationStatus.NO_BIOLOGICAL_SAMPLES),
            jnp.where(
                finite,
                int(QuantificationStatus.SUCCESS),
                int(QuantificationStatus.NONFINITE),
            ),
        ),
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        int(QuantificationEvidence.REPLICATE_SUMMARY), dtype=jnp.uint32
    )
    evidence = evidence | jnp.where(
        jnp.any(
            batch.feature_mask
            & (blank_count > 0)
            & (blank_count >= plan.minimum_blank_count)
        ),
        int(QuantificationEvidence.BLANK_CORRECTED),
        0,
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        jnp.any(drift_available), int(QuantificationEvidence.RUN_ORDER_CORRECTED), 0
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        jnp.any(censored), int(QuantificationEvidence.CENSORED_INTERVAL), 0
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        jnp.any(missing), int(QuantificationEvidence.MISSING_VALUES), 0
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        jnp.any(batch.feature_mask & (blank_count < plan.minimum_blank_count)),
        int(QuantificationEvidence.INSUFFICIENT_BLANKS),
        0,
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        jnp.any(batch.feature_mask & (qc_count < plan.minimum_qc_count)),
        int(QuantificationEvidence.INSUFFICIENT_QUALITY_CONTROLS),
        0,
    ).astype(jnp.uint32)
    evidence = jnp.where(
        capacity_ok,
        evidence,
        jnp.asarray(int(QuantificationEvidence.NONE), dtype=jnp.uint32),
    )
    return QuantificationResult(
        corrected_intensity=jnp.where(capacity_ok, point_values, nan),
        lower_bound=jnp.where(capacity_ok, lower, nan),
        upper_bound=jnp.where(capacity_ok, upper, nan),
        blank_estimate=jnp.where(capacity_ok, blank_estimate, 0.0),
        drift_factor=jnp.where(capacity_ok, drift_factor, 1.0),
        point_estimate_mask=point_mask & capacity_ok,
        censored_mask=censored & capacity_ok,
        missing_mask=missing & capacity_ok,
        replicate_mean=jnp.where(capacity_ok, replicate_mean, nan),
        replicate_lower_bound=jnp.where(capacity_ok, replicate_lower, nan),
        replicate_upper_bound=jnp.where(capacity_ok, replicate_upper, nan),
        replicate_observed_count=jnp.where(capacity_ok, observed_count, 0),
        replicate_censored_count=jnp.where(capacity_ok, censored_count, 0),
        replicate_missing_count=jnp.where(capacity_ok, missing_count, 0),
        replicate_group_mask=plan.group_mask,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_QUANTIFICATION_CONTRACT,
    )


__all__ = [
    "QuantificationBatch",
    "QuantificationEvidence",
    "QuantificationPlan",
    "QuantificationResult",
    "QuantificationSampleKind",
    "QuantificationStatus",
    "quantify_replicates",
]
