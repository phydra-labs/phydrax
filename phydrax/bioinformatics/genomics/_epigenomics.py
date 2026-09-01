#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


EPIGENOMICS_SUCCESS = 0
EPIGENOMICS_CAPACITY_EXCEEDED = 1
EPIGENOMICS_INVALID_COORDINATE = 2
EPIGENOMICS_MISSING_CONTROL = 3
EPIGENOMICS_ALL_FRAGMENTS_BLACKLISTED = 4
EPIGENOMICS_INSUFFICIENT_COVERAGE = 5
EPIGENOMICS_INVALID_CONTEXT = 6
EPIGENOMICS_CONVERSION_FAILED = 7
EPIGENOMICS_MM_ML_ORIENTATION_INVALID = 8
EPIGENOMICS_ML_UNCALIBRATED = 9
EPIGENOMICS_INVALID_ML_SCORE = 10


def epigenomics_status_name(status: int, /) -> str:
    """Return the stable name of an epigenomics status code."""
    names = (
        "success",
        "declared_capacity_exceeded",
        "invalid_coordinate_or_sample",
        "required_control_missing",
        "all_fragments_blacklisted",
        "insufficient_methylation_coverage",
        "invalid_methylation_context",
        "conversion_control_failed",
        "mm_ml_orientation_invalid",
        "ml_probability_uncalibrated",
        "invalid_ml_score",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown epigenomics status {code}.")
    return names[code]


def _fragment_contract(assay_kind: str, /) -> BioinformaticsMethodContract:
    method_name = (
        "atac_fragment_peak_blacklist_statistics"
        if assay_kind == "atac"
        else "chip_fragment_peak_control_blacklist_statistics"
    )
    return BioinformaticsMethodContract(
        method_name,
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Statistics condition on explicit fragment, peak, control, and blacklist "
            "intervals in one reference coordinate system."
        ),
        truncation_statement=(
            "Fragment and interval capacities are preflighted; exceedance is returned "
            "and never silently truncates overlaps."
        ),
        capacity_semantics="maximum_fragments and maximum_intervals are hard capacities.",
        assumptions=(
            "Fragments are half-open reference intervals.",
            (
                "ATAC peak occupancy is descriptive; controls are optional."
                if assay_kind == "atac"
                else "ChIP enrichment requires a distinct declared control interval set."
            ),
        ),
        nondifferentiable_outputs=("counts", "status", "valid"),
    )


def _methylation_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "bisulfite_methylation_context_statistics",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Methylation fractions condition on covered methylated/total calls, "
            "sequence context, and supplied conversion controls."
        ),
        truncation_statement="Call capacity exceedance is explicit and no calls are dropped.",
        capacity_semantics="maximum_calls is a hard preflight capacity.",
        assumptions=(
            "Context codes 0, 1, and 2 denote CG, CHG, and CHH.",
            "Conversion controls report methylated and total control counts per sample.",
        ),
        nondifferentiable_outputs=("coverage", "status", "valid"),
    )


def _modification_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "sam_mm_ml_native_modification_statistics",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Modification probabilities condition on array-lowered SAM MM orientation, "
            "ML scores, alignment orientation, and a declared calibration."
        ),
        truncation_statement="Call capacity exceedance is explicit and no calls are dropped.",
        capacity_semantics="maximum_calls is a hard preflight capacity.",
        assumptions=(
            "MM positions have already been lowered relative to stored SEQ orientation.",
            "ML entries align one-for-one with lowered MM calls.",
        ),
        nondifferentiable_outputs=("orientation", "status", "valid"),
    )


def _integer_vector(name: str, value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 1 or not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError(f"{name} must be a rank-one integer array.")
    return array.astype(jnp.int32)


class ChromatinFragmentBatch(StrictModule):
    """Array-lowered ATAC or ChIP fragments in half-open reference coordinates."""

    contig_index: Array
    start: Array
    end: Array
    sample_index: Array
    valid_mask: Array
    assay_kind: str = eqx.field(static=True)
    paired_end: bool = eqx.field(static=True)

    def __init__(
        self,
        contig_index: ArrayLike,
        start: ArrayLike,
        end: ArrayLike,
        sample_index: ArrayLike,
        /,
        *,
        assay_kind: str,
        valid_mask: ArrayLike | None = None,
        paired_end: bool = True,
    ):
        contig = _integer_vector("contig_index", contig_index)
        start_ = _integer_vector("start", start)
        end_ = _integer_vector("end", end)
        sample = _integer_vector("sample_index", sample_index)
        if not (contig.shape == start_.shape == end_.shape == sample.shape):
            raise ValueError("Fragment arrays must have matching shapes.")
        if assay_kind not in ("atac", "chip"):
            raise ValueError("assay_kind must be 'atac' or 'chip'.")
        valid = (
            jnp.ones(contig.shape, dtype=bool)
            if valid_mask is None
            else jnp.asarray(valid_mask, dtype=bool)
        )
        if valid.shape != contig.shape:
            raise ValueError("valid_mask must match fragment arrays.")
        self.contig_index = contig
        self.start = start_
        self.end = end_
        self.sample_index = sample
        self.valid_mask = valid
        self.assay_kind = str(assay_kind)
        self.paired_end = bool(paired_end)


class PeakControlBlacklistPlan(StrictModule):
    """Distinct peak, control, and blacklist interval collections."""

    peak_contig: Array
    peak_start: Array
    peak_end: Array
    control_contig: Array
    control_start: Array
    control_end: Array
    blacklist_contig: Array
    blacklist_start: Array
    blacklist_end: Array
    maximum_fragments: int = eqx.field(static=True)
    maximum_intervals: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        peak_contig: ArrayLike,
        peak_start: ArrayLike,
        peak_end: ArrayLike,
        control_contig: ArrayLike,
        control_start: ArrayLike,
        control_end: ArrayLike,
        blacklist_contig: ArrayLike,
        blacklist_start: ArrayLike,
        blacklist_end: ArrayLike,
        /,
        *,
        maximum_fragments: int,
        maximum_intervals: int,
    ):
        interval_groups = []
        for prefix, values in (
            ("peak", (peak_contig, peak_start, peak_end)),
            ("control", (control_contig, control_start, control_end)),
            ("blacklist", (blacklist_contig, blacklist_start, blacklist_end)),
        ):
            arrays = tuple(
                _integer_vector(f"{prefix}_{name}", value)
                for name, value in zip(("contig", "start", "end"), values, strict=True)
            )
            if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
                raise ValueError(f"{prefix} interval arrays must match.")
            interval_groups.append(arrays)
        fragments = int(maximum_fragments)
        intervals = int(maximum_intervals)
        if fragments < 1 or intervals < 1:
            raise ValueError("Fragment and interval capacities must be positive.")
        (
            (self.peak_contig, self.peak_start, self.peak_end),
            (self.control_contig, self.control_start, self.control_end),
            (self.blacklist_contig, self.blacklist_start, self.blacklist_end),
        ) = interval_groups
        self.maximum_fragments = fragments
        self.maximum_intervals = intervals
        self.plan_id = canonical_fingerprint(
            {
                "kind": "peak-control-blacklist-plan",
                "maximum_fragments": fragments,
                "maximum_intervals": intervals,
                "arrays": array_tree_fingerprint(
                    tuple(array for group in interval_groups for array in group)
                ),
            }
        )


class ChromatinFragmentEvidence(StrictModule):
    """Peak/control/blacklist and capacity evidence per sample."""

    input_coordinates_valid: Array
    capacities_satisfied: Array
    control_declared: Array
    fragments_observed: Array
    fragments_blacklisted: Array
    fragments_retained: Array
    fragments_in_peak: Array
    assay_kind: str = eqx.field(static=True)


class ChromatinFragmentStatistics(StrictModule):
    """ATAC/ChIP fragment summaries with distinct peak and control counts."""

    peak_counts: Array
    control_counts: Array
    fraction_fragments_in_peaks: Array
    blacklist_fraction: Array
    peak_control_enrichment: Array
    valid: Array
    status: Array
    evidence: ChromatinFragmentEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def _interval_overlap(
    fragments: ChromatinFragmentBatch,
    contig: Array,
    start: Array,
    end: Array,
) -> Array:
    return (
        (fragments.contig_index[:, None] == contig[None, :])
        & (fragments.start[:, None] < end[None, :])
        & (fragments.end[:, None] > start[None, :])
    )


def chromatin_fragment_statistics(
    fragments: ChromatinFragmentBatch,
    intervals: PeakControlBlacklistPlan,
    /,
    *,
    sample_count: int,
    method_contract: BioinformaticsMethodContract | None = None,
) -> ChromatinFragmentStatistics:
    """Count ATAC/ChIP fragment overlaps without conflating controls or blacklist."""
    if not isinstance(fragments, ChromatinFragmentBatch):
        raise TypeError("fragments must be ChromatinFragmentBatch.")
    if not isinstance(intervals, PeakControlBlacklistPlan):
        raise TypeError("intervals must be PeakControlBlacklistPlan.")
    samples = int(sample_count)
    if samples < 1:
        raise ValueError("sample_count must be positive.")

    interval_coordinate_valid = (
        jnp.all(intervals.peak_start >= 0)
        & jnp.all(intervals.peak_end > intervals.peak_start)
        & jnp.all(intervals.control_start >= 0)
        & jnp.all(intervals.control_end > intervals.control_start)
        & jnp.all(intervals.blacklist_start >= 0)
        & jnp.all(intervals.blacklist_end > intervals.blacklist_start)
    )
    fragment_coordinate_valid = (
        (fragments.contig_index >= 0)
        & (fragments.start >= 0)
        & (fragments.end > fragments.start)
        & (fragments.sample_index >= 0)
        & (fragments.sample_index < samples)
    )
    input_valid = (
        jnp.all(~fragments.valid_mask | fragment_coordinate_valid)
        & interval_coordinate_valid
    )
    safe_sample = jnp.where(fragment_coordinate_valid, fragments.sample_index, 0)
    fragment_active = fragments.valid_mask & fragment_coordinate_valid
    peak_overlap = _interval_overlap(
        fragments, intervals.peak_contig, intervals.peak_start, intervals.peak_end
    )
    control_overlap = _interval_overlap(
        fragments,
        intervals.control_contig,
        intervals.control_start,
        intervals.control_end,
    )
    blacklist_overlap = _interval_overlap(
        fragments,
        intervals.blacklist_contig,
        intervals.blacklist_start,
        intervals.blacklist_end,
    )
    blacklisted = fragment_active & jnp.any(blacklist_overlap, axis=1)
    retained = fragment_active & ~blacklisted
    peak_membership = retained[:, None] & peak_overlap
    control_membership = retained[:, None] & control_overlap
    sample_membership = jax.nn.one_hot(safe_sample, samples, dtype=jnp.int32).T
    peak_counts = sample_membership @ peak_membership.astype(jnp.int32)
    control_counts = sample_membership @ control_membership.astype(jnp.int32)
    observed_count = sample_membership @ fragment_active.astype(jnp.int32)
    blacklisted_count = sample_membership @ blacklisted.astype(jnp.int32)
    retained_count = sample_membership @ retained.astype(jnp.int32)
    in_peak_count = sample_membership @ (retained & jnp.any(peak_overlap, axis=1)).astype(
        jnp.int32
    )
    fraction_in_peak = in_peak_count / jnp.maximum(retained_count, 1)
    blacklist_fraction = blacklisted_count / jnp.maximum(observed_count, 1)
    peak_rate = jnp.sum(peak_counts, axis=1) / max(int(intervals.peak_contig.size), 1)
    control_rate = jnp.sum(control_counts, axis=1) / max(
        int(intervals.control_contig.size), 1
    )
    peak_control_enrichment = jnp.where(control_rate > 0.0, peak_rate / control_rate, 0.0)
    control_declared = jnp.asarray(
        fragments.assay_kind == "atac" or int(intervals.control_contig.size) > 0
    )
    capacities = jnp.asarray(
        int(fragments.start.size) <= intervals.maximum_fragments
        and (
            int(intervals.peak_start.size)
            + int(intervals.control_start.size)
            + int(intervals.blacklist_start.size)
        )
        <= intervals.maximum_intervals
    )
    sample_retained = retained_count > 0
    valid = input_valid & capacities & control_declared & sample_retained
    status = jnp.where(
        ~capacities,
        EPIGENOMICS_CAPACITY_EXCEEDED,
        jnp.where(
            ~input_valid,
            EPIGENOMICS_INVALID_COORDINATE,
            jnp.where(
                ~control_declared,
                EPIGENOMICS_MISSING_CONTROL,
                jnp.where(
                    sample_retained,
                    EPIGENOMICS_SUCCESS,
                    EPIGENOMICS_ALL_FRAGMENTS_BLACKLISTED,
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = ChromatinFragmentEvidence(
        input_valid,
        capacities,
        control_declared,
        observed_count,
        blacklisted_count,
        retained_count,
        in_peak_count,
        fragments.assay_kind,
    )
    return ChromatinFragmentStatistics(
        peak_counts,
        control_counts,
        fraction_in_peak,
        blacklist_fraction,
        peak_control_enrichment,
        valid,
        status,
        evidence,
        method_contract
        if method_contract is not None
        else _fragment_contract(fragments.assay_kind),
        "exact_descriptive",
    )


class BisulfiteCallBatch(StrictModule):
    """Covered methylated/total calls with explicit sequence-context codes."""

    methylated_count: Array
    total_count: Array
    context_code: Array
    sample_index: Array
    valid_mask: Array

    def __init__(
        self,
        methylated_count: ArrayLike,
        total_count: ArrayLike,
        context_code: ArrayLike,
        sample_index: ArrayLike,
        /,
        *,
        valid_mask: ArrayLike | None = None,
    ):
        methylated = _integer_vector("methylated_count", methylated_count)
        total = _integer_vector("total_count", total_count)
        context = _integer_vector("context_code", context_code)
        sample = _integer_vector("sample_index", sample_index)
        if not (methylated.shape == total.shape == context.shape == sample.shape):
            raise ValueError("Bisulfite call arrays must have matching shapes.")
        valid = (
            jnp.ones(total.shape, dtype=bool)
            if valid_mask is None
            else jnp.asarray(valid_mask, dtype=bool)
        )
        if valid.shape != total.shape:
            raise ValueError("valid_mask must match bisulfite call arrays.")
        self.methylated_count = methylated
        self.total_count = total
        self.context_code = context
        self.sample_index = sample
        self.valid_mask = valid


class BisulfiteMethylationPlan(StrictModule):
    """Coverage, context, conversion, and capacity requirements."""

    minimum_coverage_per_context: int = eqx.field(static=True)
    minimum_conversion_rate: float = eqx.field(static=True)
    maximum_calls: int = eqx.field(static=True)
    context_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_coverage_per_context: int,
        minimum_conversion_rate: float,
        maximum_calls: int,
        context_names: tuple[str, ...] = ("CG", "CHG", "CHH"),
    ):
        coverage = int(minimum_coverage_per_context)
        conversion = float(minimum_conversion_rate)
        capacity = int(maximum_calls)
        names = tuple(str(name) for name in context_names)
        if (
            coverage < 1
            or not math.isfinite(conversion)
            or not 0.0 <= conversion <= 1.0
            or capacity < 1
        ):
            raise ValueError(
                "Bisulfite coverage, conversion, and capacity must be valid."
            )
        if names != ("CG", "CHG", "CHH"):
            raise ValueError("Canonical bisulfite context names must be CG, CHG, CHH.")
        self.minimum_coverage_per_context = coverage
        self.minimum_conversion_rate = conversion
        self.maximum_calls = capacity
        self.context_names = names
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bisulfite-methylation-plan",
                "minimum_coverage_per_context": coverage,
                "minimum_conversion_rate": conversion,
                "maximum_calls": capacity,
                "context_names": names,
            }
        )


class BisulfiteMethylationEvidence(StrictModule):
    """Coverage, context validity, and conversion-control evidence."""

    coverage: Array
    context_covered: Array
    calls_valid: Array
    conversion_control_coverage: Array
    conversion_rate: Array
    conversion_passed: Array
    capacities_satisfied: Array
    context_names: tuple[str, ...] = eqx.field(static=True)


class BisulfiteMethylationStatistics(StrictModule):
    """Coverage-aware methylation fractions by sample and canonical context."""

    methylated_count: Array
    coverage: Array
    methylation_fraction: Array
    conversion_rate: Array
    valid: Array
    status: Array
    evidence: BisulfiteMethylationEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def bisulfite_methylation_statistics(
    calls: BisulfiteCallBatch,
    conversion_methylated_count: ArrayLike,
    conversion_total_count: ArrayLike,
    plan: BisulfiteMethylationPlan,
    /,
    *,
    sample_count: int,
    method_contract: BioinformaticsMethodContract | None = None,
) -> BisulfiteMethylationStatistics:
    """Aggregate context-aware methylation with explicit conversion controls."""
    if not isinstance(calls, BisulfiteCallBatch):
        raise TypeError("calls must be BisulfiteCallBatch.")
    if not isinstance(plan, BisulfiteMethylationPlan):
        raise TypeError("plan must be BisulfiteMethylationPlan.")
    samples = int(sample_count)
    conversion_methylated = _integer_vector(
        "conversion_methylated_count", conversion_methylated_count
    )
    conversion_total = _integer_vector("conversion_total_count", conversion_total_count)
    if (
        samples < 1
        or conversion_methylated.shape != (samples,)
        or conversion_total.shape != (samples,)
    ):
        raise ValueError(
            "Conversion controls must contain one pair of counts per sample."
        )

    call_physical = (
        (calls.methylated_count >= 0)
        & (calls.total_count > 0)
        & (calls.methylated_count <= calls.total_count)
    )
    context_valid = (calls.context_code >= 0) & (calls.context_code < 3)
    sample_valid = (calls.sample_index >= 0) & (calls.sample_index < samples)
    call_valid = calls.valid_mask & call_physical & context_valid & sample_valid
    all_calls_valid = jnp.all(
        ~calls.valid_mask | (call_physical & context_valid & sample_valid)
    )
    safe_sample = jnp.where(sample_valid, calls.sample_index, 0)
    safe_context = jnp.where(context_valid, calls.context_code, 0)
    methylated = (
        jnp.zeros((samples, 3), dtype=jnp.int32)
        .at[safe_sample, safe_context]
        .add(jnp.where(call_valid, calls.methylated_count, 0))
    )
    coverage = (
        jnp.zeros((samples, 3), dtype=jnp.int32)
        .at[safe_sample, safe_context]
        .add(jnp.where(call_valid, calls.total_count, 0))
    )
    methylation_fraction = jnp.where(
        coverage > 0,
        methylated / jnp.where(coverage > 0, coverage, 1),
        0.0,
    )
    context_covered = coverage >= plan.minimum_coverage_per_context
    conversion_physical = (
        (conversion_methylated >= 0)
        & (conversion_total > 0)
        & (conversion_methylated <= conversion_total)
    )
    conversion_rate = jnp.where(
        conversion_physical,
        1.0
        - conversion_methylated / jnp.where(conversion_total > 0, conversion_total, 1),
        0.0,
    )
    conversion_passed = conversion_physical & (
        conversion_rate >= plan.minimum_conversion_rate
    )
    capacities = jnp.asarray(int(calls.total_count.size) <= plan.maximum_calls)
    sample_context_covered = jnp.all(context_covered, axis=1)
    valid = all_calls_valid & capacities & sample_context_covered & conversion_passed
    status = jnp.where(
        ~capacities,
        EPIGENOMICS_CAPACITY_EXCEEDED,
        jnp.where(
            ~all_calls_valid,
            jnp.where(
                jnp.any(calls.valid_mask & ~context_valid),
                EPIGENOMICS_INVALID_CONTEXT,
                EPIGENOMICS_INVALID_COORDINATE,
            ),
            jnp.where(
                ~sample_context_covered,
                EPIGENOMICS_INSUFFICIENT_COVERAGE,
                jnp.where(
                    conversion_passed, EPIGENOMICS_SUCCESS, EPIGENOMICS_CONVERSION_FAILED
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = BisulfiteMethylationEvidence(
        coverage,
        context_covered,
        all_calls_valid,
        conversion_total,
        conversion_rate,
        conversion_passed,
        capacities,
        plan.context_names,
    )
    return BisulfiteMethylationStatistics(
        methylated,
        coverage,
        methylation_fraction,
        conversion_rate,
        valid,
        status,
        evidence,
        method_contract if method_contract is not None else _methylation_contract(),
        "exact_descriptive",
    )


class MMMLModificationBatch(StrictModule):
    """Array-lowered SAM MM/ML calls; SAM text and records never enter the PyTree."""

    read_index: Array
    sequence_position: Array
    reference_position: Array
    modification_code_index: Array
    mm_strand: Array
    ml_score: Array
    valid_mask: Array
    read_length: Array
    read_sample_index: Array
    alignment_reverse: Array

    def __init__(
        self,
        read_index: ArrayLike,
        sequence_position: ArrayLike,
        reference_position: ArrayLike,
        modification_code_index: ArrayLike,
        mm_strand: ArrayLike,
        ml_score: ArrayLike,
        read_length: ArrayLike,
        read_sample_index: ArrayLike,
        alignment_reverse: ArrayLike,
        /,
        *,
        valid_mask: ArrayLike | None = None,
    ):
        call_arrays = tuple(
            _integer_vector(name, value)
            for name, value in (
                ("read_index", read_index),
                ("sequence_position", sequence_position),
                ("reference_position", reference_position),
                ("modification_code_index", modification_code_index),
                ("mm_strand", mm_strand),
                ("ml_score", ml_score),
            )
        )
        if any(array.shape != call_arrays[0].shape for array in call_arrays[1:]):
            raise ValueError("Lowered MM/ML call arrays must have matching shapes.")
        lengths = _integer_vector("read_length", read_length)
        samples = _integer_vector("read_sample_index", read_sample_index)
        reverse = jnp.asarray(alignment_reverse, dtype=bool)
        if lengths.shape != samples.shape or reverse.shape != lengths.shape:
            raise ValueError("Read metadata arrays must have matching shapes.")
        valid = (
            jnp.ones(call_arrays[0].shape, dtype=bool)
            if valid_mask is None
            else jnp.asarray(valid_mask, dtype=bool)
        )
        if valid.shape != call_arrays[0].shape:
            raise ValueError("valid_mask must match lowered MM/ML calls.")
        (
            self.read_index,
            self.sequence_position,
            self.reference_position,
            self.modification_code_index,
            self.mm_strand,
            self.ml_score,
        ) = call_arrays
        self.valid_mask = valid
        self.read_length = lengths
        self.read_sample_index = samples
        self.alignment_reverse = reverse


class MMMLCalibrationPlan(StrictModule):
    """Affine ML calibration and explicit modification-code vocabulary."""

    modification_codes: tuple[str, ...] = eqx.field(static=True)
    probability_scale: float = eqx.field(static=True)
    probability_offset: float = eqx.field(static=True)
    call_threshold: float = eqx.field(static=True)
    maximum_calls: int = eqx.field(static=True)
    calibrated: bool = eqx.field(static=True)
    calibration_provenance_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        modification_codes: tuple[str, ...],
        /,
        *,
        probability_scale: float = 1.0,
        probability_offset: float = 0.0,
        call_threshold: float = 0.5,
        maximum_calls: int,
        calibrated: bool,
        calibration_provenance_id: str,
    ):
        codes = tuple(str(code) for code in modification_codes)
        scale = float(probability_scale)
        offset = float(probability_offset)
        threshold = float(call_threshold)
        capacity = int(maximum_calls)
        if not codes or any(not code for code in codes) or len(set(codes)) != len(codes):
            raise ValueError("Modification codes must be unique and non-empty.")
        if (
            not all(math.isfinite(value) for value in (scale, offset, threshold))
            or scale <= 0.0
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("ML calibration parameters must be finite and valid.")
        if capacity < 1 or not calibration_provenance_id:
            raise ValueError("ML capacity and calibration provenance must be declared.")
        self.modification_codes = codes
        self.probability_scale = scale
        self.probability_offset = offset
        self.call_threshold = threshold
        self.maximum_calls = capacity
        self.calibrated = bool(calibrated)
        self.calibration_provenance_id = str(calibration_provenance_id)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mm-ml-calibration-plan",
                "modification_codes": codes,
                "probability_scale": scale,
                "probability_offset": offset,
                "call_threshold": threshold,
                "maximum_calls": capacity,
                "calibrated": self.calibrated,
                "calibration_provenance_id": self.calibration_provenance_id,
            }
        )


class MMMLModificationEvidence(StrictModule):
    """MM/ML pairing, orientation, calibration, and capacity evidence."""

    call_orientation_valid: Array
    ml_score_valid: Array
    reference_forward_strand: Array
    calibration_declared: Array
    capacities_satisfied: Array
    calls_observed: Array
    calls_passing_threshold: Array
    modification_codes: tuple[str, ...] = eqx.field(static=True)
    calibration_provenance_id: str = eqx.field(static=True)


class MMMLModificationStatistics(StrictModule):
    """Calibrated native-modification statistics by sample and modification code."""

    call_count: Array
    expected_modified_count: Array
    threshold_call_fraction: Array
    forward_strand_count: Array
    reverse_strand_count: Array
    calibrated_probability: Array
    valid: Array
    status: Array
    evidence: MMMLModificationEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def mm_ml_modification_statistics(
    calls: MMMLModificationBatch,
    calibration: MMMLCalibrationPlan,
    /,
    *,
    sample_count: int,
    method_contract: BioinformaticsMethodContract | None = None,
) -> MMMLModificationStatistics:
    """Orient lowered MM calls to reference strand and calibrate paired ML scores."""
    if not isinstance(calls, MMMLModificationBatch):
        raise TypeError("calls must be MMMLModificationBatch.")
    if not isinstance(calibration, MMMLCalibrationPlan):
        raise TypeError("calibration must be MMMLCalibrationPlan.")
    samples = int(sample_count)
    if samples < 1:
        raise ValueError("sample_count must be positive.")
    read_count = int(calls.read_length.size)
    code_count = len(calibration.modification_codes)
    read_valid = (calls.read_index >= 0) & (calls.read_index < read_count)
    safe_read = jnp.where(read_valid, calls.read_index, 0)
    sequence_position_valid = (calls.sequence_position >= 0) & (
        calls.sequence_position < calls.read_length[safe_read]
    )
    reference_position_valid = calls.reference_position >= 0
    strand_valid = (calls.mm_strand == 1) | (calls.mm_strand == -1)
    read_metadata_valid = (
        (calls.read_length[safe_read] > 0)
        & (calls.read_sample_index[safe_read] >= 0)
        & (calls.read_sample_index[safe_read] < samples)
    )
    code_valid = (calls.modification_code_index >= 0) & (
        calls.modification_code_index < code_count
    )
    orientation_valid = (
        read_valid
        & sequence_position_valid
        & reference_position_valid
        & strand_valid
        & read_metadata_valid
        & code_valid
    )
    ml_valid = (calls.ml_score >= 0) & (calls.ml_score <= 255)
    active = calls.valid_mask & orientation_valid & ml_valid
    safe_sample = jnp.where(read_metadata_valid, calls.read_sample_index[safe_read], 0)
    safe_code = jnp.where(code_valid, calls.modification_code_index, 0)
    raw_probability = jnp.clip(calls.ml_score, 0, 255) / 255.0
    probability = jnp.clip(
        calibration.probability_offset + calibration.probability_scale * raw_probability,
        0.0,
        1.0,
    )
    mm_plus = calls.mm_strand == 1
    reference_forward = jnp.logical_xor(mm_plus, calls.alignment_reverse[safe_read])
    call_count = (
        jnp.zeros((samples, code_count), dtype=jnp.int32)
        .at[safe_sample, safe_code]
        .add(active.astype(jnp.int32))
    )
    expected = (
        jnp.zeros((samples, code_count), dtype=probability.dtype)
        .at[safe_sample, safe_code]
        .add(jnp.where(active, probability, 0.0))
    )
    threshold_count = (
        jnp.zeros((samples, code_count), dtype=jnp.int32)
        .at[safe_sample, safe_code]
        .add((active & (probability >= calibration.call_threshold)).astype(jnp.int32))
    )
    forward_count = (
        jnp.zeros((samples, code_count), dtype=jnp.int32)
        .at[safe_sample, safe_code]
        .add((active & reference_forward).astype(jnp.int32))
    )
    reverse_count = (
        jnp.zeros((samples, code_count), dtype=jnp.int32)
        .at[safe_sample, safe_code]
        .add((active & ~reference_forward).astype(jnp.int32))
    )
    fraction = threshold_count / jnp.maximum(call_count, 1)
    capacities = jnp.asarray(int(calls.ml_score.size) <= calibration.maximum_calls)
    all_orientation_valid = jnp.all(~calls.valid_mask | orientation_valid)
    all_ml_valid = jnp.all(~calls.valid_mask | ml_valid)
    calibration_declared = jnp.asarray(calibration.calibrated)
    sample_observed = jnp.sum(call_count, axis=1, dtype=jnp.int32) > 0
    valid = (
        sample_observed
        & capacities
        & all_orientation_valid
        & all_ml_valid
        & calibration_declared
    )
    status = jnp.where(
        ~capacities,
        EPIGENOMICS_CAPACITY_EXCEEDED,
        jnp.where(
            ~all_orientation_valid,
            EPIGENOMICS_MM_ML_ORIENTATION_INVALID,
            jnp.where(
                ~all_ml_valid,
                EPIGENOMICS_INVALID_ML_SCORE,
                jnp.where(
                    calibration_declared, EPIGENOMICS_SUCCESS, EPIGENOMICS_ML_UNCALIBRATED
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = MMMLModificationEvidence(
        orientation_valid,
        ml_valid,
        reference_forward,
        calibration_declared,
        capacities,
        call_count,
        threshold_count,
        calibration.modification_codes,
        calibration.calibration_provenance_id,
    )
    return MMMLModificationStatistics(
        call_count,
        expected,
        fraction,
        forward_count,
        reverse_count,
        probability,
        valid,
        status,
        evidence,
        method_contract if method_contract is not None else _modification_contract(),
        "calibrated_probability_descriptive",
    )


__all__ = [
    "EPIGENOMICS_ALL_FRAGMENTS_BLACKLISTED",
    "EPIGENOMICS_CAPACITY_EXCEEDED",
    "EPIGENOMICS_CONVERSION_FAILED",
    "EPIGENOMICS_INSUFFICIENT_COVERAGE",
    "EPIGENOMICS_INVALID_CONTEXT",
    "EPIGENOMICS_INVALID_COORDINATE",
    "EPIGENOMICS_INVALID_ML_SCORE",
    "EPIGENOMICS_MISSING_CONTROL",
    "EPIGENOMICS_ML_UNCALIBRATED",
    "EPIGENOMICS_MM_ML_ORIENTATION_INVALID",
    "EPIGENOMICS_SUCCESS",
    "BisulfiteCallBatch",
    "BisulfiteMethylationEvidence",
    "BisulfiteMethylationPlan",
    "BisulfiteMethylationStatistics",
    "ChromatinFragmentBatch",
    "ChromatinFragmentEvidence",
    "ChromatinFragmentStatistics",
    "MMMLCalibrationPlan",
    "MMMLModificationBatch",
    "MMMLModificationEvidence",
    "MMMLModificationStatistics",
    "PeakControlBlacklistPlan",
    "bisulfite_methylation_statistics",
    "chromatin_fragment_statistics",
    "epigenomics_status_name",
    "mm_ml_modification_statistics",
]
