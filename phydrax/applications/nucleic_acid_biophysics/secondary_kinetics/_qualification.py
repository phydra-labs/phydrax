#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Locked raw-trace comparison and scientific qualification evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ...._fingerprint import canonical_fingerprint
from ....qualification import (
    QualificationEvidence,
    ScientificCampaign,
    ScientificClaimProfile,
)
from ..interchange._strand_displacement import (
    FluorescenceTimeTrace,
    StrandDisplacementCohort,
)
from ._fluorescence import (
    EffectiveDisplacementRateModel,
    FluorescencePrediction,
    MechanisticDisplacementRateModel,
    predict_locked_fluorescence,
    ReporterObservationModel,
    StrandDisplacementForwardModel,
    trace_log_probability,
)


_LOG_SCORE_UNIT = "natural-log-unit-per-observation"
_DIMENSIONLESS_UNIT = "dimensionless"
_METRIC_AGGREGATIONS = {
    "raw-trace-log-score": "pooled",
    "family-macro-log-score": "independent_unit_macro",
    "preparation-worst-log-score": "worst_stratum",
    "predictive-interval-coverage-95": "pooled",
    "family-macro-predictive-interval-coverage-95": "independent_unit_macro",
    "log-half-time-rate-mean-absolute-error": "independent_unit_macro",
    "matched-rate-ordering-concordance": "independent_unit_macro",
    "concentration-order-concordance": "independent_unit_macro",
    "reporter-standardized-residual-mean-absolute": "pooled",
    "reporter-standardized-residual-root-mean-square": "pooled",
    "mechanistic-log-score-improvement": "independent_unit_macro",
}
_METRIC_UNITS = {
    "raw-trace-log-score": _LOG_SCORE_UNIT,
    "family-macro-log-score": _LOG_SCORE_UNIT,
    "preparation-worst-log-score": _LOG_SCORE_UNIT,
    "predictive-interval-coverage-95": _DIMENSIONLESS_UNIT,
    "family-macro-predictive-interval-coverage-95": _DIMENSIONLESS_UNIT,
    "log-half-time-rate-mean-absolute-error": "natural-log-second-inverse",
    "matched-rate-ordering-concordance": _DIMENSIONLESS_UNIT,
    "concentration-order-concordance": _DIMENSIONLESS_UNIT,
    "reporter-standardized-residual-mean-absolute": _DIMENSIONLESS_UNIT,
    "reporter-standardized-residual-root-mean-square": _DIMENSIONLESS_UNIT,
    "mechanistic-log-score-improvement": _LOG_SCORE_UNIT,
}
_MODEL_SPECIFIC_STAGES = frozenset(
    ("parameter-identifiability", "predictive-calibration")
)


def _role_cases(campaign: ScientificCampaign, names: Sequence[str], /) -> frozenset[str]:
    selected = frozenset(names)
    return frozenset(
        case_id
        for role in campaign.roles
        if role.name in selected
        for case_id in role.case_ids
    )


@dataclass(frozen=True, slots=True)
class GroupedTraceScore:
    """One explicit macro stratum; cases remain the independent trace axis."""

    group_kind: str
    group_id: str
    case_ids: tuple[str, ...]
    mean_log_score_per_observation: float
    interval_coverage_95: float


@dataclass(frozen=True, slots=True)
class LockedModelEvaluation:
    """Raw scores, grouped evidence and an exact locked-prediction stage."""

    model_id: str
    model_kind: str
    predictions: tuple[FluorescencePrediction, ...]
    case_log_probabilities: tuple[float, ...]
    family_scores: tuple[GroupedTraceScore, ...]
    preparation_scores: tuple[GroupedTraceScore, ...]
    metric_values: tuple[tuple[str, float], ...]
    metric_units: tuple[tuple[str, str], ...]
    metric_aggregations: tuple[tuple[str, str], ...]
    metric_gaps: tuple[str, ...]
    execution_evidence: QualificationEvidence


@dataclass(frozen=True, slots=True)
class StrandDisplacementQualificationResult:
    """Side-by-side effective/mechanistic evidence without a universal winner claim."""

    campaign_id: str
    effective: LockedModelEvaluation
    mechanistic: LockedModelEvaluation
    effective_claim_evidence: QualificationEvidence
    mechanistic_claim_evidence: QualificationEvidence
    predictive_winner_model_id: str | None
    comparison_id: str


def _metadata(
    *,
    build_id: str,
    environment_id: str,
    backend: str,
    topology: str,
    precision: str,
    reduction: str,
    replay_id: str,
    reviewer_id: str,
    issued_at: int,
    expires_at: int,
) -> dict[str, object]:
    values = {
        "build_id": build_id,
        "environment_id": environment_id,
        "backend": backend,
        "topology": topology,
        "precision": precision,
        "reduction": reduction,
        "replay_id": replay_id,
        "reviewer_id": reviewer_id,
    }
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in values.values()
    ):
        raise ValueError(
            "Qualification execution metadata requires canonical identifiers."
        )
    if (
        type(issued_at) is not int
        or type(expires_at) is not int
        or expires_at <= issued_at
    ):
        raise ValueError("Qualification evidence timestamps must be ordered integers.")
    return {**values, "issued_at": issued_at, "expires_at": expires_at}


def _stage_evidence(
    outcome: str,
    reason: str,
    campaign_id: str,
    model_id: str,
    raw_artifact_ids: Sequence[str],
    metadata: Mapping[str, object],
    /,
    *,
    criteria_ids: Sequence[str] = ("locked-prediction",),
) -> QualificationEvidence:
    return QualificationEvidence(
        "scientific",
        outcome,
        (campaign_id, model_id),
        build_id=metadata["build_id"],
        environment_id=metadata["environment_id"],
        backend=metadata["backend"],
        topology=metadata["topology"],
        precision=metadata["precision"],
        reduction=metadata["reduction"],
        replay_id=metadata["replay_id"],
        criteria_ids=tuple(criteria_ids),
        raw_artifact_ids=raw_artifact_ids,
        reviewer_id=metadata["reviewer_id"],
        issued_at=metadata["issued_at"],
        expires_at=metadata["expires_at"],
        reason=reason,
        requalification_triggers=(
            "model-parameters",
            "observation-law",
            "operating-domain",
            "source-artifact",
        ),
        campaign_start_record_ids=(),
        campaign_observation_record_ids=(),
    )


def _support_reasons(
    model: StrandDisplacementForwardModel,
    observation_model: ReporterObservationModel,
    traces: Sequence[FluorescenceTimeTrace],
    /,
) -> tuple[str, ...]:
    reasons = [reason for trace in traces for reason in model.support_reasons(trace)]
    reasons.extend(observation_model.uncertainty_limitations)
    reasons.extend(model.uncertainty_limitations)
    for trace in traces:
        if trace.injection_reference_seconds is None:
            reasons.append("injection-reference-unknown")
        if trace.reporter_id != observation_model.calibration.reporter_id:
            reasons.append("reporter-outside-observation-model-support")
        if trace.intensity_unit_id != observation_model.intensity_unit_id:
            reasons.append("intensity-unit-outside-observation-model-support")
        if (
            trace.has_saturated_observations
            and trace.saturation_threshold_intensity is None
        ):
            reasons.append("saturation-threshold-unknown")
        if trace.has_saturated_observations and (
            observation_model.calibration.covariance is not None
            or model.fit.has_epistemic_uncertainty
        ):
            reasons.append("correlated-epistemic-uncertainty-with-censoring-unsupported")
    return tuple(sorted(set(reasons)))


def _coverage(
    trace: FluorescenceTimeTrace, prediction: FluorescencePrediction, /
) -> tuple[int, int]:
    observed = np.asarray(trace.intensity)
    active = ~np.asarray(trace.saturation_mask)
    lower = np.asarray(prediction.lower_95_intensity)
    upper = np.asarray(prediction.upper_95_intensity)
    covered = active & (observed >= lower) & (observed <= upper)
    return int(np.count_nonzero(covered)), int(np.count_nonzero(active))


def _group_scores(
    traces: Sequence[FluorescenceTimeTrace],
    predictions: Sequence[FluorescencePrediction],
    case_scores: Sequence[float],
    /,
    *,
    kind: str,
) -> tuple[GroupedTraceScore, ...]:
    if kind == "family":
        coordinate = lambda trace: trace.sequence_family_id
    elif kind == "preparation":
        coordinate = lambda trace: trace.identity.preparation_id
    else:
        raise ValueError("Trace grouping kind must be family or preparation.")
    groups = sorted({coordinate(trace) for trace in traces})
    result: list[GroupedTraceScore] = []
    for group in groups:
        indices = tuple(i for i, trace in enumerate(traces) if coordinate(trace) == group)
        observation_count = sum(int(traces[i].time_seconds.size) for i in indices)
        log_score = sum(case_scores[i] for i in indices) / observation_count
        coverage = tuple(_coverage(traces[i], predictions[i]) for i in indices)
        covered = sum(value[0] for value in coverage)
        active = sum(value[1] for value in coverage)
        result.append(
            GroupedTraceScore(
                kind,
                group,
                tuple(traces[i].case_id for i in indices),
                float(log_score),
                float(covered / active) if active else float("nan"),
            )
        )
    return tuple(result)


def _half_time(time: np.ndarray, values: np.ndarray, /) -> float | None:
    active = time > 0.0
    if np.count_nonzero(active) < 2:
        return None
    post_time = time[active]
    post = values[active]
    baseline = float(post[0])
    amplitude = float(post[-1] - baseline)
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        return None
    threshold = baseline + 0.5 * amplitude
    crossing = np.flatnonzero(post >= threshold)
    if not crossing.size or crossing[0] == 0:
        return None
    index = int(crossing[0])
    before_time, after_time = float(post_time[index - 1]), float(post_time[index])
    before, after = float(post[index - 1]), float(post[index])
    if after <= before:
        return None
    fraction = (threshold - before) / (after - before)
    result = before_time + fraction * (after_time - before_time)
    return result if result > 0.0 and math.isfinite(result) else None


def _pairwise_concordance(
    observed: Mapping[str, float], predicted: Mapping[str, float], /
) -> float | None:
    shared = tuple(sorted(set(observed) & set(predicted)))
    concordant = 0
    informative = 0
    for i, first in enumerate(shared):
        for second in shared[i + 1 :]:
            observed_difference = observed[first] - observed[second]
            predicted_difference = predicted[first] - predicted[second]
            if observed_difference == 0.0 or predicted_difference == 0.0:
                continue
            informative += 1
            concordant += int(observed_difference * predicted_difference > 0.0)
    return None if informative == 0 else concordant / informative


def _secondary_trace_metrics(
    traces: Sequence[FluorescenceTimeTrace],
    predictions: Sequence[FluorescencePrediction],
    /,
) -> tuple[dict[str, float], tuple[str, ...]]:
    observed_rates: dict[str, float] = {}
    predicted_rates: dict[str, float] = {}
    for trace, prediction in zip(traces, predictions, strict=True):
        observed_time = _half_time(
            np.asarray(trace.time_seconds),
            np.asarray(trace.intensity),
        )
        predicted_time = _half_time(
            np.asarray(prediction.time_seconds),
            np.asarray(prediction.mean_intensity),
        )
        if observed_time is not None:
            observed_rates[trace.case_id] = 1.0 / observed_time
        if predicted_time is not None:
            predicted_rates[trace.case_id] = 1.0 / predicted_time
    shared = tuple(sorted(set(observed_rates) & set(predicted_rates)))
    metrics: dict[str, float] = {}
    gaps: list[str] = []
    if shared:
        metrics["log-half-time-rate-mean-absolute-error"] = float(
            np.mean(
                [
                    abs(math.log(predicted_rates[case]) - math.log(observed_rates[case]))
                    for case in shared
                ]
            )
        )
    else:
        gaps.append("log-half-time-rate:no-shared-resolved-half-times")
    concordance = _pairwise_concordance(observed_rates, predicted_rates)
    if concordance is None:
        gaps.append("matched-rate-ordering:no-informative-case-pair")
    else:
        metrics["matched-rate-ordering-concordance"] = float(concordance)

    observed_endpoints: dict[str, float] = {}
    predicted_endpoints: dict[str, float] = {}
    limiting_concentration: dict[str, float] = {}
    for trace, prediction in zip(traces, predictions, strict=True):
        unsaturated = np.flatnonzero(~np.asarray(trace.saturation_mask))
        if unsaturated.size:
            observed_endpoints[trace.case_id] = float(
                np.asarray(trace.intensity)[unsaturated[-1]]
            )
            predicted_endpoints[trace.case_id] = float(
                np.asarray(prediction.mean_intensity)[unsaturated[-1]]
            )
            limiting_concentration[trace.case_id] = float(
                np.min(np.asarray(trace.initial_concentrations_molar))
            )
    concordant = 0
    informative = 0
    for i, first in enumerate(traces):
        for second in traces[i + 1 :]:
            if (
                first.sequence_family_id != second.sequence_family_id
                or first.construct_ids != second.construct_ids
                or first.case_id not in observed_endpoints
                or second.case_id not in observed_endpoints
            ):
                continue
            concentration_difference = (
                limiting_concentration[first.case_id]
                - limiting_concentration[second.case_id]
            )
            observed_difference = (
                observed_endpoints[first.case_id] - observed_endpoints[second.case_id]
            )
            predicted_difference = (
                predicted_endpoints[first.case_id] - predicted_endpoints[second.case_id]
            )
            if (
                concentration_difference == 0.0
                or observed_difference == 0.0
                or predicted_difference == 0.0
            ):
                continue
            informative += 1
            concordant += int(
                concentration_difference * observed_difference > 0.0
                and concentration_difference * predicted_difference > 0.0
            )
    if informative:
        metrics["concentration-order-concordance"] = concordant / informative
    else:
        gaps.append("concentration-order:no-matched-informative-concentration-pair")
    return metrics, tuple(gaps)


def _completed_evaluation(
    model: StrandDisplacementForwardModel,
    model_kind: str,
    observation_model: ReporterObservationModel,
    traces: Sequence[FluorescenceTimeTrace],
    campaign_id: str,
    raw_artifact_ids: Sequence[str],
    metadata: Mapping[str, object],
    /,
) -> LockedModelEvaluation:
    prediction = predict_locked_fluorescence(model, observation_model, traces)
    case_scores = tuple(
        float(trace_log_probability(trace, predicted))
        for trace, predicted in zip(traces, prediction.predictions, strict=True)
    )
    family = _group_scores(
        traces,
        prediction.predictions,
        case_scores,
        kind="family",
    )
    preparation = _group_scores(
        traces,
        prediction.predictions,
        case_scores,
        kind="preparation",
    )
    total_observations = sum(int(trace.time_seconds.size) for trace in traces)
    active_coverages = tuple(
        _coverage(trace, predicted)
        for trace, predicted in zip(traces, prediction.predictions, strict=True)
    )
    covered = sum(value[0] for value in active_coverages)
    active = sum(value[1] for value in active_coverages)
    residual_parts = tuple(
        (
            np.asarray(trace.intensity)[~np.asarray(trace.saturation_mask)]
            - np.asarray(predicted.mean_intensity)[~np.asarray(trace.saturation_mask)]
        )
        / np.asarray(predicted.standard_deviation_intensity)[
            ~np.asarray(trace.saturation_mask)
        ]
        for trace, predicted in zip(traces, prediction.predictions, strict=True)
    )
    metrics = {
        "raw-trace-log-score": float(sum(case_scores) / total_observations),
        "family-macro-log-score": float(
            np.mean([item.mean_log_score_per_observation for item in family])
        ),
        "preparation-worst-log-score": float(
            min(item.mean_log_score_per_observation for item in preparation)
        ),
    }
    gaps: tuple[str, ...] = tuple(
        "uncertainty:" + limitation
        for limitation in observation_model.uncertainty_limitations
    )
    if active:
        residuals = np.concatenate(residual_parts)
        metrics.update(
            {
                "predictive-interval-coverage-95": float(covered / active),
                "family-macro-predictive-interval-coverage-95": float(
                    np.nanmean(
                        [
                            item.interval_coverage_95
                            for item in family
                            if math.isfinite(item.interval_coverage_95)
                        ]
                    )
                ),
                "reporter-standardized-residual-mean-absolute": float(
                    abs(np.mean(residuals))
                ),
                "reporter-standardized-residual-root-mean-square": float(
                    np.sqrt(np.mean(residuals**2))
                ),
            }
        )
    else:
        gaps = tuple((*gaps, "reporter-residuals:no-unsaturated-observations"))
    secondary, secondary_gaps = _secondary_trace_metrics(traces, prediction.predictions)
    gaps = tuple((*gaps, *secondary_gaps))
    metrics.update(secondary)
    evidence = _stage_evidence(
        "passed",
        "locked-raw-trace-prediction-completed",
        campaign_id,
        model.model_id,
        raw_artifact_ids,
        metadata,
    )
    metric_names = tuple(sorted(metrics))
    return LockedModelEvaluation(
        model.model_id,
        model_kind,
        prediction.predictions,
        case_scores,
        family,
        preparation,
        tuple((name, metrics[name]) for name in metric_names),
        tuple((name, _METRIC_UNITS[name]) for name in metric_names),
        tuple((name, _METRIC_AGGREGATIONS[name]) for name in metric_names),
        gaps,
        evidence,
    )


def _inconclusive_evaluation(
    model: StrandDisplacementForwardModel,
    model_kind: str,
    reasons: Sequence[str],
    campaign_id: str,
    raw_artifact_ids: Sequence[str],
    metadata: Mapping[str, object],
    /,
) -> LockedModelEvaluation:
    reasons_ = tuple(sorted(set(reasons)))
    uncertainty_incomplete = any(
        reason.endswith("uncertainty-unquantified") for reason in reasons_
    )
    evidence = _stage_evidence(
        "inconclusive",
        "locked-prediction-unavailable:" + ",".join(reasons_),
        campaign_id,
        model.model_id,
        raw_artifact_ids,
        metadata,
        criteria_ids=(
            ("locked-prediction", "predictive-calibration")
            if uncertainty_incomplete
            else ("locked-prediction",)
        ),
    )
    return LockedModelEvaluation(
        model.model_id,
        model_kind,
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        reasons_,
        evidence,
    )


def _claim_evidence(
    profile: ScientificClaimProfile,
    evaluation: LockedModelEvaluation,
    stage_evidence: Sequence[QualificationEvidence],
    raw_artifact_ids: Sequence[str],
    metadata: Mapping[str, object],
    /,
) -> QualificationEvidence:
    if any("locked-prediction" in item.criteria_ids for item in stage_evidence):
        raise ValueError(
            "The workflow owns locked-prediction evidence; do not supply a second record."
        )
    applicable_stage_evidence = tuple(
        item
        for item in stage_evidence
        if not _MODEL_SPECIFIC_STAGES.intersection(item.criteria_ids)
        or evaluation.model_id in item.subject_ids
    )
    return profile.evaluate(
        dict(evaluation.metric_values),
        applicable_stage_evidence + (evaluation.execution_evidence,),
        metric_units=dict(evaluation.metric_units),
        metric_aggregations=dict(evaluation.metric_aggregations),
        build_id=metadata["build_id"],
        environment_id=metadata["environment_id"],
        backend=metadata["backend"],
        topology=metadata["topology"],
        precision=metadata["precision"],
        reduction=metadata["reduction"],
        replay_id=metadata["replay_id"],
        raw_artifact_ids=raw_artifact_ids,
        reviewer_id=metadata["reviewer_id"],
        issued_at=metadata["issued_at"],
        expires_at=metadata["expires_at"],
    )


def qualify_strand_displacement_models(
    effective_model: EffectiveDisplacementRateModel,
    mechanistic_model: MechanisticDisplacementRateModel,
    observation_model: ReporterObservationModel,
    cohort: StrandDisplacementCohort,
    effective_claim_profile: ScientificClaimProfile,
    mechanistic_claim_profile: ScientificClaimProfile,
    stage_evidence: Sequence[QualificationEvidence],
    /,
    *,
    build_id: str,
    environment_id: str,
    backend: str,
    topology: str,
    precision: str,
    reduction: str,
    replay_id: str,
    reviewer_id: str,
    issued_at: int,
    expires_at: int,
) -> StrandDisplacementQualificationResult:
    """Compare locked models on held-out wells and return pass/fail/inconclusive claims.

    Support/capacity/measurement limitations become inconclusive scientific
    evidence.  Role leakage remains invalid workflow input and is refused.
    """

    if not isinstance(cohort, StrandDisplacementCohort):
        raise TypeError("cohort must be a StrandDisplacementCohort.")
    if not isinstance(cohort.campaign, ScientificCampaign):
        raise TypeError("cohort campaign must be a ScientificCampaign.")
    if not isinstance(effective_model, EffectiveDisplacementRateModel):
        raise TypeError("effective_model must be an EffectiveDisplacementRateModel.")
    if not isinstance(mechanistic_model, MechanisticDisplacementRateModel):
        raise TypeError("mechanistic_model must be a MechanisticDisplacementRateModel.")
    if not isinstance(observation_model, ReporterObservationModel):
        raise TypeError("observation_model must be a ReporterObservationModel.")
    if not isinstance(effective_claim_profile, ScientificClaimProfile) or not isinstance(
        mechanistic_claim_profile, ScientificClaimProfile
    ):
        raise TypeError("claim profiles must be ScientificClaimProfile values.")
    profiles = (effective_claim_profile, mechanistic_claim_profile)
    campaign = cohort.campaign
    conditions = tuple(sorted({trace.condition_id for trace in cohort.traces}))
    chemistry_directions = {trace.chemistry_direction for trace in cohort.traces}
    if len(chemistry_directions) != 1:
        raise ValueError("Strand cohort must declare one exact chemistry direction.")
    chemistry_direction = next(iter(chemistry_directions))
    if any(
        profile.capability_name != "nucleic.strand-displacement"
        or profile.observable_ids != ("raw-fluorescence-trace",)
        or profile.condition_domain_ids != conditions
        or dict(profile.support.attributes)
        != {"chemistry_direction": chemistry_direction}
        or profile.frozen_criteria_ids != campaign.criteria_ids
        for profile in profiles
    ):
        raise ValueError(
            "Strand claim profiles must exactly bind capability, raw observable, "
            "conditions, chemistry support, and frozen campaign criteria."
        )
    profile_criterion_ids = {
        criterion.criterion_id for profile in profiles for criterion in profile.criteria
    }
    if profile_criterion_ids != set(campaign.criteria_ids):
        raise ValueError(
            "Campaign criteria must exactly equal the compared claim criteria."
        )
    if observation_model.campaign_id != cohort.campaign.campaign_id:
        raise ValueError(
            "Reporter observation model must bind the exact cohort campaign."
        )
    if effective_model.chemistry_direction != mechanistic_model.chemistry_direction:
        raise ValueError(
            "Compared models must declare one identical chemistry direction."
        )
    common_stage_ids = frozenset(
        (
            "source-admission",
            "measurement-calibration",
            "predictive-calibration",
            "locked-prediction",
        )
    )
    if set(effective_claim_profile.required_stage_ids) != common_stage_ids:
        raise ValueError(
            "Effective qualification must require the exact source, measurement, "
            "predictive-calibration, and locked-prediction stages."
        )
    if set(mechanistic_claim_profile.required_stage_ids) != common_stage_ids | {
        "parameter-identifiability"
    }:
        raise ValueError(
            "Mechanistic qualification must require the effective stages plus "
            "parameter-identifiability."
        )
    mechanistic_criteria = {
        criterion.metric_id: criterion for criterion in mechanistic_claim_profile.criteria
    }
    improvement_criterion = mechanistic_criteria.get("mechanistic-log-score-improvement")
    if (
        "parameter-identifiability" not in mechanistic_claim_profile.required_stage_ids
        or improvement_criterion is None
        or improvement_criterion.direction != "at_least"
        or improvement_criterion.lower is None
        or improvement_criterion.lower <= 0.0
        or improvement_criterion.aggregation != "independent_unit_macro"
    ):
        raise ValueError(
            "Mechanistic qualification must require parameter-identifiability and "
            "a strictly positive locked log-score improvement."
        )
    campaign = cohort.campaign
    if (
        effective_claim_profile.campaign_id != campaign.campaign_id
        or mechanistic_claim_profile.campaign_id != campaign.campaign_id
    ):
        raise ValueError("Scientific claim profiles must bind the exact cohort campaign.")
    if any(not isinstance(item, QualificationEvidence) for item in stage_evidence):
        raise TypeError("stage_evidence must contain QualificationEvidence values.")
    locked_ids = _role_cases(campaign, ("locked_evaluation",))
    calibration_ids = tuple(sorted(_role_cases(campaign, ("calibration",))))
    selection_ids = tuple(sorted(_role_cases(campaign, ("model_selection",))))
    cohort_traces_by_case = {trace.case_id: trace for trace in cohort.traces}
    if not set((*calibration_ids, *selection_ids, *locked_ids)) <= set(
        cohort_traces_by_case
    ):
        raise ValueError("Cohort must retain every campaign fit and evaluation trace.")
    for model in (effective_model, mechanistic_model):
        expected_source_traces = tuple(
            sorted(
                cohort_traces_by_case[case_id].trace_id
                for case_id in (*calibration_ids, *selection_ids)
            )
        )
        if (
            model.fit.campaign_id != campaign.campaign_id
            or model.fit.fit_case_ids != calibration_ids
            or model.fit.model_selection_case_ids != selection_ids
            or model.fit.source_trace_ids != expected_source_traces
            or model.fit.selected_prepared.observation_model.observation_model_id
            != observation_model.observation_model_id
        ):
            raise ValueError(
                "Locked models must derive from exact campaign fit/model-selection "
                "traces and the exact reporter observation model."
            )
    models_by_id = {
        effective_model.model_id: effective_model,
        mechanistic_model.model_id: mechanistic_model,
    }
    for item in stage_evidence:
        if not _MODEL_SPECIFIC_STAGES.intersection(item.criteria_ids):
            continue
        matching = tuple(
            model
            for model_id, model in models_by_id.items()
            if model_id in item.subject_ids
        )
        if (
            len(matching) != 1
            or (
                "parameter-identifiability" in item.criteria_ids
                and matching[0] is not mechanistic_model
            )
            or set(item.subject_ids)
            != {
                campaign.campaign_id,
                matching[0].model_id,
                matching[0].fit.fit_id,
            }
            or set(item.raw_artifact_ids) != set(matching[0].fit.source_manifest_ids)
        ):
            raise ValueError(
                "Model-specific strand evidence must bind the exact applicable "
                "campaign, forward model, fit artifact, and fit/model-selection sources."
            )
    traces = tuple(trace for trace in cohort.traces if trace.case_id in locked_ids)
    if {trace.case_id for trace in traces} != set(locked_ids):
        raise ValueError(
            "Every locked campaign case must have exactly one admitted raw trace."
        )
    metadata = _metadata(
        build_id=build_id,
        environment_id=environment_id,
        backend=backend,
        topology=topology,
        precision=precision,
        reduction=reduction,
        replay_id=replay_id,
        reviewer_id=reviewer_id,
        issued_at=issued_at,
        expires_at=expires_at,
    )
    raw_artifact_ids = tuple(
        sorted({manifest for trace in traces for manifest in trace.source_manifest_ids})
    )
    effective_reasons = _support_reasons(effective_model, observation_model, traces)
    effective = (
        _inconclusive_evaluation(
            effective_model,
            "effective-mass-action",
            effective_reasons,
            campaign.campaign_id,
            raw_artifact_ids,
            metadata,
        )
        if effective_reasons
        else _completed_evaluation(
            effective_model,
            "effective-mass-action",
            observation_model,
            traces,
            campaign.campaign_id,
            raw_artifact_ids,
            metadata,
        )
    )
    mechanistic_reasons = _support_reasons(mechanistic_model, observation_model, traces)
    mechanistic = (
        _inconclusive_evaluation(
            mechanistic_model,
            "exhaustive-secondary-kinetics",
            mechanistic_reasons,
            campaign.campaign_id,
            raw_artifact_ids,
            metadata,
        )
        if mechanistic_reasons
        else _completed_evaluation(
            mechanistic_model,
            "exhaustive-secondary-kinetics",
            observation_model,
            traces,
            campaign.campaign_id,
            raw_artifact_ids,
            metadata,
        )
    )
    winner: str | None = None
    if effective.case_log_probabilities and mechanistic.case_log_probabilities:
        effective_family_scores = {
            item.group_id: item.mean_log_score_per_observation
            for item in effective.family_scores
        }
        mechanistic_family_scores = {
            item.group_id: item.mean_log_score_per_observation
            for item in mechanistic.family_scores
        }
        if set(effective_family_scores) != set(mechanistic_family_scores):
            raise ValueError("Compared models must score the same held-out families.")
        improvement = float(
            np.mean(
                [
                    mechanistic_family_scores[family_id]
                    - effective_family_scores[family_id]
                    for family_id in sorted(effective_family_scores)
                ]
            )
        )
        mechanistic_metrics = dict(mechanistic.metric_values)
        mechanistic_metrics["mechanistic-log-score-improvement"] = float(improvement)
        metric_names = tuple(sorted(mechanistic_metrics))
        mechanistic = LockedModelEvaluation(
            mechanistic.model_id,
            mechanistic.model_kind,
            mechanistic.predictions,
            mechanistic.case_log_probabilities,
            mechanistic.family_scores,
            mechanistic.preparation_scores,
            tuple((name, mechanistic_metrics[name]) for name in metric_names),
            tuple((name, _METRIC_UNITS[name]) for name in metric_names),
            tuple((name, _METRIC_AGGREGATIONS[name]) for name in metric_names),
            mechanistic.metric_gaps,
            mechanistic.execution_evidence,
        )
        winner = mechanistic.model_id if improvement > 0.0 else effective.model_id
    effective_claim = _claim_evidence(
        effective_claim_profile,
        effective,
        stage_evidence,
        raw_artifact_ids,
        metadata,
    )
    mechanistic_claim = _claim_evidence(
        mechanistic_claim_profile,
        mechanistic,
        stage_evidence,
        raw_artifact_ids,
        metadata,
    )
    comparison_id = canonical_fingerprint(
        {
            "kind": "locked-strand-displacement-model-comparison",
            "campaign": campaign.campaign_id,
            "effective_model": effective.model_id,
            "mechanistic_model": mechanistic.model_id,
            "observation_model": observation_model.observation_model_id,
            "effective_claim": effective_claim.evidence_id,
            "mechanistic_claim": mechanistic_claim.evidence_id,
            "winner_by_raw_log_score_only": winner,
        }
    )
    return StrandDisplacementQualificationResult(
        campaign.campaign_id,
        effective,
        mechanistic,
        effective_claim,
        mechanistic_claim,
        winner,
        comparison_id,
    )


__all__ = [
    "GroupedTraceScore",
    "LockedModelEvaluation",
    "StrandDisplacementQualificationResult",
    "qualify_strand_displacement_models",
]
