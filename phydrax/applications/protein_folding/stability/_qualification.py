#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Grouped qualification and double-mutant challenge for protein stability."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ...._fingerprint import canonical_fingerprint
from ....qualification import (
    CampaignRole,
    QualificationEvidence,
    ScientificCampaign,
    ScientificCase,
    ScientificClaimProfile,
    ScientificMetricCriterion,
    SupportTuple,
)
from ....uq import SplitConformal
from .._construct import ProteinConstruct
from ..interchange._megascale import (
    ProteinStabilityCohort,
)
from ._features import ProteinMutationFeatures
from ._models import (
    _feature_matches_measurement,
    AbstractProteinStabilityPredictor,
    DoubleMutantCase,
    GlobalSubstitutionBaseline,
    ProteinStabilityModelFit,
    ProteinStabilityModelSelectionRecord,
    RegularizedPairInteractionModel,
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _role_case_ids(campaign: ScientificCampaign, role_name: str, /) -> tuple[str, ...]:
    for role in campaign.roles:
        if role.name == role_name:
            return role.case_ids
    return ()


def _evidence(
    campaign_id: str,
    stage_id: str,
    outcome: str,
    reason: str,
    /,
    *,
    build_id: str,
    environment_id: str,
    backend: str,
    topology: str,
    precision: str,
    reduction: str,
    replay_id: str,
    raw_artifact_ids: Sequence[str],
    reviewer_id: str,
    issued_at: int,
    expires_at: int,
) -> QualificationEvidence:
    return QualificationEvidence(
        "scientific",
        outcome,
        (campaign_id,),
        build_id=build_id,
        environment_id=environment_id,
        backend=backend,
        topology=topology,
        precision=precision,
        reduction=reduction,
        replay_id=replay_id,
        criteria_ids=(stage_id,),
        raw_artifact_ids=tuple(raw_artifact_ids),
        reviewer_id=reviewer_id,
        issued_at=issued_at,
        expires_at=expires_at,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class ProteinStabilityThresholds:
    """Predeclared operating tolerances; no universal threshold is embedded."""

    maximum_family_macro_mae_kcal_per_mol: float
    minimum_interval_coverage: float
    maximum_mean_interval_width_kcal_per_mol: float
    minimum_baseline_benefit_kcal_per_mol: float
    minimum_prediction_coverage: float
    interval_miscoverage: float = 0.1

    def __post_init__(self) -> None:
        values = (
            self.maximum_family_macro_mae_kcal_per_mol,
            self.maximum_mean_interval_width_kcal_per_mol,
            self.minimum_baseline_benefit_kcal_per_mol,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Protein stability thresholds must be finite.")
        if (
            self.maximum_family_macro_mae_kcal_per_mol <= 0.0
            or self.maximum_mean_interval_width_kcal_per_mol <= 0.0
            or self.minimum_baseline_benefit_kcal_per_mol < 0.0
        ):
            raise ValueError(
                "Error/width thresholds must be positive and benefit non-negative."
            )
        if not 0.0 < self.minimum_interval_coverage <= 1.0:
            raise ValueError("minimum_interval_coverage must lie in (0,1].")
        if not 0.0 < self.minimum_prediction_coverage <= 1.0:
            raise ValueError("minimum_prediction_coverage must lie in (0,1].")
        if not 0.0 < self.interval_miscoverage < 1.0:
            raise ValueError("interval_miscoverage must lie in (0,1).")


@dataclass(frozen=True, slots=True)
class GroupedStabilityPrediction:
    case_id: str
    independent_group_id: str
    domain_family_id: str
    background_id: str
    observed_kcal_per_mol: float
    predicted_kcal_per_mol: float | None
    baseline_kcal_per_mol: float | None
    interval_lower_kcal_per_mol: float | None
    interval_upper_kcal_per_mol: float | None
    valid: bool
    abstention_reason: str | None


@dataclass(frozen=True, slots=True)
class GroupedStabilityMetrics:
    family_macro_mae_kcal_per_mol: float | None
    pooled_mae_kcal_per_mol: float | None
    baseline_family_macro_mae_kcal_per_mol: float | None
    baseline_benefit_kcal_per_mol: float | None
    interval_coverage: float | None
    mean_interval_width_kcal_per_mol: float | None
    prediction_coverage: float
    independent_group_count: int
    evaluated_case_count: int

    def criterion_values(self) -> dict[str, float]:
        values: dict[str, float] = {
            "protein-stability-prediction-coverage": self.prediction_coverage,
        }
        optional = {
            "protein-stability-family-macro-mae": self.family_macro_mae_kcal_per_mol,
            "protein-stability-baseline-benefit": self.baseline_benefit_kcal_per_mol,
            "protein-stability-interval-coverage": self.interval_coverage,
            "protein-stability-mean-interval-width": self.mean_interval_width_kcal_per_mol,
        }
        values.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return values


@dataclass(frozen=True, slots=True)
class ProteinStabilityQualificationResult:
    claim_profile: ScientificClaimProfile
    evidence: QualificationEvidence
    stage_evidence: tuple[QualificationEvidence, ...]
    predictions: tuple[GroupedStabilityPrediction, ...]
    metrics: GroupedStabilityMetrics
    selected_model_fit_id: str
    model_selection_id: str
    baseline_fit_id: str
    conformal_radius_kcal_per_mol: float | None
    result_id: str


def protein_stability_claim_profile(
    cohort: ProteinStabilityCohort,
    support: SupportTuple,
    thresholds: ProteinStabilityThresholds,
    /,
) -> ScientificClaimProfile:
    """Build the assay-bounded empirical claim; this is not a force-field claim."""
    if not isinstance(cohort, ProteinStabilityCohort):
        raise TypeError("cohort must be ProteinStabilityCohort.")
    if not isinstance(support, SupportTuple):
        raise TypeError("support must be SupportTuple.")
    capability = "protein.mutation-stability-prediction"
    if support.capability != capability:
        raise ValueError(f"support capability must be {capability!r}.")
    criteria = (
        ScientificMetricCriterion(
            "protein-stability-family-macro-mae",
            "at_most",
            None,
            thresholds.maximum_family_macro_mae_kcal_per_mol,
            "kcal/mol",
            "independent_unit_macro",
        ),
        ScientificMetricCriterion(
            "protein-stability-interval-coverage",
            "between",
            thresholds.minimum_interval_coverage,
            1.0,
            "fraction",
            "independent_unit_macro",
        ),
        ScientificMetricCriterion(
            "protein-stability-mean-interval-width",
            "at_most",
            None,
            thresholds.maximum_mean_interval_width_kcal_per_mol,
            "kcal/mol",
            "independent_unit_macro",
        ),
        ScientificMetricCriterion(
            "protein-stability-baseline-benefit",
            "at_least",
            thresholds.minimum_baseline_benefit_kcal_per_mol,
            None,
            "kcal/mol",
            "independent_unit_macro",
        ),
        ScientificMetricCriterion(
            "protein-stability-prediction-coverage",
            "at_least",
            thresholds.minimum_prediction_coverage,
            None,
            "fraction",
            "independent_unit_macro",
        ),
    )
    return ScientificClaimProfile(
        capability,
        support,
        tuple(sorted({item.observable for item in cohort.measurements})),
        tuple(sorted({item.condition_id for item in cohort.measurements})),
        cohort.campaign.campaign_id,
        (
            "source-admission",
            "measurement-calibration",
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        ),
        criteria,
        "abstain-on-feature-or-condition-domain-mismatch",
        (
            "source-artifact-or-lineage-change",
            "campaign-membership-change",
            "feature-definition-change",
            "assay-condition-domain-change",
            "model-or-uncertainty-calibration-change",
        ),
        frozen_criteria_ids=cohort.campaign.criteria_ids,
    )


def _stability_raw_artifact_ids(
    cohort: ProteinStabilityCohort,
    features: Sequence[ProteinMutationFeatures],
    /,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                artifact_id
                for item in cohort.measurements
                for artifact_id in (
                    item.source_manifest_id,
                    item.uncertainty_source_manifest_id,
                )
                if artifact_id is not None
            }
            | {
                manifest.manifest_id
                for feature in features
                for manifest in feature.source_manifests
            }
        )
    )


def source_admission_evidence(
    cohort: ProteinStabilityCohort,
    features: Sequence[ProteinMutationFeatures],
    /,
    **execution,
) -> QualificationEvidence:
    """Record measurement and feature artifacts admitted with reversible lineage."""
    manifests = _stability_raw_artifact_ids(cohort, features)
    measurement_by_id = {
        measurement.measurement_id: measurement for measurement in cohort.measurements
    }
    fields_retained = all(bool(item.source_fields) for item in cohort.measurements)
    feature_sources_retained = all(
        feature.source_manifests
        and set(feature.preprocessing_source_ids)
        == {manifest.manifest_id for manifest in feature.source_manifests}
        | {measurement_by_id[feature.measurement_id].source_manifest_id}
        for feature in features
    )
    complete = fields_retained and feature_sources_retained
    outcome = "passed" if complete else "failed"
    reason = (
        "measurement-and-feature-source-artifacts-admitted"
        if complete
        else "source-admission-lacks-reversible-measurement-or-feature-artifacts"
    )
    return _evidence(
        cohort.campaign.campaign_id,
        "source-admission",
        outcome,
        reason,
        raw_artifact_ids=manifests,
        **execution,
    )


def _measurement_calibration_evidence(
    cohort: ProteinStabilityCohort,
    raw_artifact_ids: Sequence[str],
    /,
    **execution,
) -> QualificationEvidence:
    blocks: dict[tuple[str, str], int] = {}
    for measurement in cohort.measurements:
        block = (measurement.shared_wt_id, measurement.assay_channel)
        blocks[block] = blocks.get(block, 0) + 1
    if any(
        measurement.standard_error_kcal_per_mol is None
        or measurement.uncertainty_source_manifest_id is None
        for measurement in cohort.measurements
    ):
        outcome = "inconclusive"
        reason = "per-measurement-uncertainty-unquantified"
    elif any(count > 1 for count in blocks.values()):
        outcome = "inconclusive"
        reason = "shared-wt-block-covariance-unquantified"
    else:
        outcome = "passed"
        reason = "independent-per-measurement-standard-errors-complete"
    return _evidence(
        cohort.campaign.campaign_id,
        "measurement-calibration",
        outcome,
        reason,
        raw_artifact_ids=raw_artifact_ids,
        **execution,
    )


def _model_fit_stage_evidence(
    cohort: ProteinStabilityCohort,
    model_fit: ProteinStabilityModelFit,
    raw_artifact_ids: Sequence[str],
    /,
    **execution,
) -> QualificationEvidence:
    if model_fit.successful:
        outcome = "passed"
        reason = "native-regularized-fit-and-posterior-completed"
    elif any("native" in value for value in model_fit.reasons):
        outcome = "failed"
        reason = "native-model-fit-failed:" + ",".join(model_fit.reasons)
    else:
        outcome = "inconclusive"
        reason = "model-fit-scientifically-inconclusive:" + ",".join(model_fit.reasons)
    return _evidence(
        cohort.campaign.campaign_id,
        "numerical-validity",
        outcome,
        reason,
        raw_artifact_ids=raw_artifact_ids,
        **execution,
    )


def _calibrate_group_conformal(
    cohort: ProteinStabilityCohort,
    feature_by_id: Mapping[str, ProteinMutationFeatures],
    predictor: AbstractProteinStabilityPredictor,
    alpha: float,
    /,
) -> tuple[float | None, str]:
    calibration_ids = _role_case_ids(cohort.campaign, "interval_calibration")
    measurement_by_id = {item.measurement_id: item for item in cohort.measurements}
    quantitative_ids = tuple(
        case_id
        for case_id in calibration_ids
        if measurement_by_id[case_id].mutation_order == 1
        and measurement_by_id[case_id].censoring == "none"
    )
    if not quantitative_ids or any(
        case_id not in feature_by_id for case_id in quantitative_ids
    ):
        return None, "interval-calibration-single-mutant-features-missing"
    eligible_ids = quantitative_ids
    predictions = predictor.predict(
        tuple(feature_by_id[case_id] for case_id in eligible_ids)
    )
    if not bool(np.all(np.asarray(predictions.valid))):
        return None, "interval-calibration-prediction-abstention"
    group_scores: dict[str, list[float]] = {}
    for index, case_id in enumerate(eligible_ids):
        measurement = measurement_by_id[case_id]
        score = abs(float(predictions.mean[index]) - measurement.value_kcal_per_mol)
        group_scores.setdefault(measurement.independent_group_id, []).append(score)
    minimum_groups = math.ceil(1.0 / alpha) - 1
    if len(group_scores) < max(minimum_groups, 2):
        return None, "insufficient-independent-groups-for-conformal-quantile"
    scores = np.asarray([max(values) for values in group_scores.values()])
    conformal = SplitConformal.calibrate(
        scores,
        np.zeros_like(scores),
        alpha=alpha,
    )
    return float(conformal.radius), "group-max-split-conformal-calibrated"


def qualify_protein_stability(
    cohort: ProteinStabilityCohort,
    features: Sequence[ProteinMutationFeatures],
    baseline_fit: ProteinStabilityModelFit,
    model_selection: ProteinStabilityModelSelectionRecord,
    support: SupportTuple,
    thresholds: ProteinStabilityThresholds,
    /,
    *,
    stage_evidence: Sequence[QualificationEvidence] = (),
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
) -> ProteinStabilityQualificationResult:
    """Evaluate only the fit frozen by exact held-out model-selection evidence."""
    if not isinstance(cohort, ProteinStabilityCohort):
        raise TypeError("cohort must be ProteinStabilityCohort.")
    if not isinstance(baseline_fit, ProteinStabilityModelFit):
        raise TypeError("baseline_fit must be a ProteinStabilityModelFit.")
    if not isinstance(model_selection, ProteinStabilityModelSelectionRecord):
        raise TypeError("model_selection must be a ProteinStabilityModelSelectionRecord.")
    values = tuple(features)
    baseline_fit.validate(cohort, values)
    model_selection.validate(cohort, values)
    if (
        baseline_fit.fit_id != model_selection.chosen_baseline_fit_id
        or baseline_fit.predictor is None
        or not isinstance(baseline_fit.predictor, GlobalSubstitutionBaseline)
    ):
        raise ValueError(
            "Baseline comparison must use the exact strongest prespecified baseline "
            "selected on model-selection cases."
        )
    baseline_fit = model_selection.chosen_baseline_fit
    selected_fit = model_selection.chosen_fit
    if any(not isinstance(item, ProteinMutationFeatures) for item in values):
        raise TypeError("features must contain ProteinMutationFeatures.")
    feature_by_id = {item.measurement_id: item for item in values}
    measurement_by_id = {item.measurement_id: item for item in cohort.measurements}
    for feature in values:
        if feature.measurement_id not in measurement_by_id:
            raise ValueError("Every feature must identify a cohort measurement.")
        measurement = measurement_by_id[feature.measurement_id]
        if (
            feature.source_measurement_record_id != measurement.record_id
            or feature.domain_id != measurement.domain_id
            or feature.domain_family_id != measurement.domain_family_id
            or feature.background_id != measurement.background_id
            or feature.wt_sequence != measurement.sequence
            or feature.mutation_code != measurement.mutation_code
            or feature.assay_channel != measurement.assay_channel
            or feature.condition_id != measurement.condition_id
            or feature.observable != measurement.observable
            or feature.sign_convention != measurement.sign_convention
            or feature.shared_wt_id != measurement.shared_wt_id
        ):
            raise ValueError(
                "Feature ABI does not identify its exact cohort measurement."
            )
    if len(feature_by_id) != len(values):
        raise ValueError("Feature case IDs must be unique.")
    profile = protein_stability_claim_profile(cohort, support, thresholds)
    if {criterion.criterion_id for criterion in profile.criteria} != set(
        cohort.campaign.criteria_ids
    ):
        raise ValueError(
            "Protein stability campaign criteria must exactly match the claim."
        )
    execution = dict(
        build_id=_identifier(build_id, "build_id"),
        environment_id=_identifier(environment_id, "environment_id"),
        backend=_identifier(backend, "backend"),
        topology=_identifier(topology, "topology"),
        precision=_identifier(precision, "precision"),
        reduction=_identifier(reduction, "reduction"),
        replay_id=_identifier(replay_id, "replay_id"),
        reviewer_id=_identifier(reviewer_id, "reviewer_id"),
        issued_at=issued_at,
        expires_at=expires_at,
    )
    raw_artifact_ids = _stability_raw_artifact_ids(cohort, values)
    generated: list[QualificationEvidence] = [
        source_admission_evidence(cohort, values, **execution),
        _measurement_calibration_evidence(cohort, raw_artifact_ids, **execution),
        _model_fit_stage_evidence(cohort, selected_fit, raw_artifact_ids, **execution),
    ]

    measurement_by_id = {item.measurement_id: item for item in cohort.measurements}
    locked_ids = _role_case_ids(cohort.campaign, "locked_evaluation")
    claim_ids = tuple(
        case_id
        for case_id in locked_ids
        if measurement_by_id[case_id].mutation_order == 1
    )
    quantitative_ids = tuple(
        case_id
        for case_id in claim_ids
        if measurement_by_id[case_id].censoring == "none" and case_id in feature_by_id
    )
    leakage = bool(
        set(claim_ids)
        & (
            set(selected_fit.calibration_case_ids)
            | set(baseline_fit.calibration_case_ids)
        )
    )
    locked_family_ids = {
        measurement_by_id[case_id].domain_family_id for case_id in claim_ids
    }
    for model_fit in (baseline_fit, selected_fit):
        if model_fit.predictor is not None and locked_family_ids.intersection(
            model_fit.predictor.training_family_ids
        ):
            raise ValueError(
                "Protein stability training families cannot enter locked evaluation."
            )
    conformal_radius: float | None = None
    conformal_reason = "selected-model-fit-unavailable"
    predictions: tuple[GroupedStabilityPrediction, ...] = ()
    metrics = GroupedStabilityMetrics(None, None, None, None, None, None, 0.0, 0, 0)

    if selected_fit.successful and selected_fit.predictor is not None:
        conformal_radius, conformal_reason = _calibrate_group_conformal(
            cohort,
            feature_by_id,
            selected_fit.predictor,
            thresholds.interval_miscoverage,
        )
    calibration_outcome = "passed" if conformal_radius is not None else "inconclusive"
    generated.append(
        _evidence(
            cohort.campaign.campaign_id,
            "predictive-calibration",
            calibration_outcome,
            conformal_reason,
            raw_artifact_ids=raw_artifact_ids,
            **execution,
        )
    )

    locked_reason = "locked-evaluation-not-executable"
    locked_outcome = "inconclusive"
    output: list[GroupedStabilityPrediction] = []
    selected_by_id: dict[str, tuple[float, bool, str | None]] = {}
    baseline_by_id: dict[str, tuple[float, bool, str | None]] = {}
    if leakage:
        locked_outcome = "failed"
        locked_reason = "locked-evaluation-cases-influenced-model-parameters"
    elif not claim_ids:
        locked_reason = "locked-evaluation-has-no-single-mutant-cases"
    elif (
        selected_fit.successful
        and baseline_fit.successful
        and selected_fit.predictor is not None
        and baseline_fit.predictor is not None
        and quantitative_ids
    ):
        locked_features = tuple(feature_by_id[case_id] for case_id in quantitative_ids)
        selected_prediction = selected_fit.predictor.predict(locked_features)
        baseline_prediction = baseline_fit.predictor.predict(locked_features)
        for index, case_id in enumerate(quantitative_ids):
            selected_by_id[case_id] = (
                float(selected_prediction.mean[index]),
                bool(selected_prediction.valid[index]),
                selected_prediction.abstention_reasons[index],
            )
            baseline_by_id[case_id] = (
                float(baseline_prediction.mean[index]),
                bool(baseline_prediction.valid[index]),
                baseline_prediction.abstention_reasons[index],
            )

    valid_case_ids: list[str] = []
    for case_id in claim_ids:
        measurement = measurement_by_id[case_id]
        selected_value = selected_by_id.get(case_id)
        baseline_value = baseline_by_id.get(case_id)
        if measurement.censoring != "none":
            valid = False
            reason = f"unsupported-{measurement.censoring}-censored-endpoint"
        elif case_id not in feature_by_id:
            valid = False
            reason = "mutation-features-missing"
        elif selected_value is None or baseline_value is None:
            valid = False
            reason = "model-fit-unavailable"
        else:
            valid = selected_value[1] and baseline_value[1]
            reason = (
                None
                if valid
                else selected_value[2] or baseline_value[2] or "model-abstention"
            )
        predicted_value = None if selected_value is None else selected_value[0]
        baseline_prediction_value = None if baseline_value is None else baseline_value[0]
        radius = conformal_radius if valid else None
        if valid:
            valid_case_ids.append(case_id)
        output.append(
            GroupedStabilityPrediction(
                case_id,
                measurement.independent_group_id,
                measurement.domain_family_id,
                measurement.background_id,
                measurement.value_kcal_per_mol,
                predicted_value,
                baseline_prediction_value,
                None if radius is None else predicted_value - radius,
                None if radius is None else predicted_value + radius,
                valid,
                reason,
            )
        )
    predictions = tuple(output)
    locked_group_ids = tuple(
        sorted({measurement_by_id[case_id].independent_group_id for case_id in claim_ids})
    )
    valid_case_id_set = set(valid_case_ids)
    prediction_coverage = (
        float(
            np.mean(
                [
                    sum(
                        case_id in valid_case_id_set
                        for case_id in claim_ids
                        if measurement_by_id[case_id].independent_group_id == group_id
                    )
                    / sum(
                        measurement_by_id[case_id].independent_group_id == group_id
                        for case_id in claim_ids
                    )
                    for group_id in locked_group_ids
                ]
            )
        )
        if locked_group_ids
        else 0.0
    )

    if valid_case_ids:
        group_ids = tuple(
            sorted(
                {
                    measurement_by_id[case_id].independent_group_id
                    for case_id in valid_case_ids
                }
            )
        )
        group_errors: list[float] = []
        baseline_group_errors: list[float] = []
        group_coverage: list[float] = []
        group_width: list[float] = []
        pooled_errors: list[float] = []
        for group_id in group_ids:
            members = tuple(
                case_id
                for case_id in valid_case_ids
                if measurement_by_id[case_id].independent_group_id == group_id
            )
            selected_errors = tuple(
                abs(
                    selected_by_id[case_id][0]
                    - measurement_by_id[case_id].value_kcal_per_mol
                )
                for case_id in members
            )
            baseline_errors = tuple(
                abs(
                    baseline_by_id[case_id][0]
                    - measurement_by_id[case_id].value_kcal_per_mol
                )
                for case_id in members
            )
            pooled_errors.extend(selected_errors)
            group_errors.append(float(np.mean(selected_errors)))
            baseline_group_errors.append(float(np.mean(baseline_errors)))
            if conformal_radius is not None:
                group_coverage.append(
                    float(np.mean(np.asarray(selected_errors) <= conformal_radius))
                )
                group_width.append(2.0 * conformal_radius)
        selected_macro = float(np.mean(group_errors))
        baseline_macro = float(np.mean(baseline_group_errors))
        metrics = GroupedStabilityMetrics(
            selected_macro,
            float(np.mean(pooled_errors)),
            baseline_macro,
            baseline_macro - selected_macro,
            float(np.mean(group_coverage)) if group_coverage else None,
            float(np.mean(group_width)) if group_width else None,
            prediction_coverage,
            len(locked_group_ids),
            len(valid_case_ids),
        )
    elif claim_ids:
        metrics = GroupedStabilityMetrics(
            None,
            None,
            None,
            None,
            None,
            None,
            prediction_coverage,
            len(locked_group_ids),
            0,
        )
    if not leakage and claim_ids:
        locked_outcome = (
            "passed" if len(valid_case_ids) == len(claim_ids) else "inconclusive"
        )
        locked_reason = (
            "locked-independent-family-predictions-completed"
            if locked_outcome == "passed"
            else "locked-predictions-contain-declared-abstentions"
        )
    generated.append(
        _evidence(
            cohort.campaign.campaign_id,
            "locked-prediction",
            locked_outcome,
            locked_reason,
            raw_artifact_ids=raw_artifact_ids,
            **execution,
        )
    )
    required_identifiability_subject_ids = {
        cohort.campaign.campaign_id,
        selected_fit.predictor.model_id
        if selected_fit.predictor is not None
        else selected_fit.fit_id,
        selected_fit.fit_id,
        model_selection.selection_id,
    }
    applicable_stage_evidence = tuple(
        item
        for item in stage_evidence
        if "parameter-identifiability" not in item.criteria_ids
        or required_identifiability_subject_ids.issubset(item.subject_ids)
    )
    all_stage_evidence = (*applicable_stage_evidence, *generated)
    metric_values = metrics.criterion_values()
    metric_units = {
        "protein-stability-family-macro-mae": "kcal/mol",
        "protein-stability-interval-coverage": "fraction",
        "protein-stability-mean-interval-width": "kcal/mol",
        "protein-stability-baseline-benefit": "kcal/mol",
        "protein-stability-prediction-coverage": "fraction",
    }
    metric_aggregations = {key: "independent_unit_macro" for key in metric_units}
    evidence = profile.evaluate(
        metric_values,
        all_stage_evidence,
        metric_units=metric_units,
        metric_aggregations=metric_aggregations,
        raw_artifact_ids=raw_artifact_ids,
        **execution,
    )
    result_id = canonical_fingerprint(
        {
            "kind": "protein-stability-qualification-result",
            "claim_id": profile.claim_id,
            "evidence_id": evidence.evidence_id,
            "selected_fit_id": selected_fit.fit_id,
            "baseline_fit_id": baseline_fit.fit_id,
            "model_selection_id": model_selection.selection_id,
            "prediction_case_ids": [item.case_id for item in predictions],
            "conformal_radius": conformal_radius,
        }
    )
    return ProteinStabilityQualificationResult(
        profile,
        evidence,
        tuple(all_stage_evidence),
        predictions,
        metrics,
        selected_fit.fit_id,
        model_selection.selection_id,
        baseline_fit.fit_id,
        conformal_radius,
        result_id,
    )


def prepare_double_mutant_campaign(
    cases: Sequence[DoubleMutantCase],
    role_by_pair_unit: Mapping[str, str],
    /,
    *,
    preparation_id_by_case: Mapping[str, str],
    batch_id_by_case: Mapping[str, str],
    preprocessing_source_ids: Sequence[str] = (),
    criteria_ids: Sequence[str] = (),
) -> ScientificCampaign:
    """Freeze every provided substitution at a residue pair into one campaign role."""
    values = tuple(cases)
    if not values or any(not isinstance(item, DoubleMutantCase) for item in values):
        raise TypeError("cases must contain DoubleMutantCase values.")
    case_ids = tuple(item.double_measurement.measurement_id for item in values)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Double-mutant campaign case IDs must be unique.")
    pair_ids = {item.pair_features.pair_unit_id for item in values}
    if set(role_by_pair_unit) != pair_ids:
        raise ValueError("Role mapping must cover residue-pair units exactly.")
    if set(preparation_id_by_case) != set(case_ids) or set(batch_id_by_case) != set(
        case_ids
    ):
        raise ValueError(
            "Preparation and batch mappings must cover double-mutant cases exactly."
        )
    role_names = (
        "calibration",
        "model_selection",
        "interval_calibration",
        "locked_evaluation",
        "prospective",
    )
    unknown_roles = set(role_by_pair_unit.values()) - set(role_names)
    if unknown_roles:
        raise ValueError(f"Unknown scientific campaign roles: {sorted(unknown_roles)!r}.")
    scientific_cases = tuple(
        ScientificCase(
            item.double_measurement.measurement_id,
            item.pair_features.pair_unit_id,
            ProteinConstruct(("A",), (item.double_measurement.sequence,)).fingerprint(),
            item.double_measurement.condition_id,
            _identifier(
                preparation_id_by_case[item.double_measurement.measurement_id],
                "preparation_id",
            ),
            _identifier(
                batch_id_by_case[item.double_measurement.measurement_id],
                "batch_id",
            ),
            tuple(
                sorted(
                    {
                        artifact_id
                        for measurement in (
                            item.double_measurement,
                            item.first_single,
                            item.second_single,
                        )
                        for artifact_id in (
                            measurement.source_manifest_id,
                            measurement.uncertainty_source_manifest_id,
                        )
                        if artifact_id is not None
                    }
                )
            ),
        )
        for item in values
    )
    roles = tuple(
        CampaignRole(
            role,
            tuple(
                item.double_measurement.measurement_id
                for item in values
                if role_by_pair_unit[item.pair_features.pair_unit_id] == role
            ),
        )
        for role in role_names
    )
    return ScientificCampaign(
        scientific_cases,
        roles,
        preprocessing_source_ids=tuple(preprocessing_source_ids),
        criteria_ids=tuple(criteria_ids),
    )


@dataclass(frozen=True, slots=True)
class DoubleMutantUncertainty:
    """Externally justified covariance-aware uncertainty for one pair challenge row."""

    case_id: str
    measurement_record_id: str
    additive_observation_variance_kcal2_per_mol2: float
    coupling_observation_variance_kcal2_per_mol2: float
    source_id: str
    additive_coupling_covariance_kcal2_per_mol2: float = 0.0

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case_id")
        _identifier(self.source_id, "source_id")
        additive_variance = self.additive_observation_variance_kcal2_per_mol2
        coupling_variance = self.coupling_observation_variance_kcal2_per_mol2
        covariance = self.additive_coupling_covariance_kcal2_per_mol2
        if (
            any(
                not math.isfinite(value)
                for value in (additive_variance, coupling_variance, covariance)
            )
            or additive_variance < 0.0
            or coupling_variance < 0.0
        ):
            raise ValueError(
                "Double-mutant variances/covariance must be finite and variances "
                "non-negative."
            )
        if covariance * covariance > additive_variance * coupling_variance:
            raise ValueError("Double-mutant observation covariance must be PSD.")
        _identifier(self.measurement_record_id, "measurement_record_id")


@dataclass(frozen=True, slots=True)
class DoubleMutantChallengeResult:
    claim_profile: ScientificClaimProfile
    evidence: QualificationEvidence
    case_ids: tuple[str, ...]
    pair_unit_ids: tuple[str, ...]
    additive_predictions_kcal_per_mol: tuple[float, ...]
    corrected_predictions_kcal_per_mol: tuple[float, ...]
    observed_kcal_per_mol: tuple[float, ...]
    additive_pair_macro_mae_kcal_per_mol: float | None
    corrected_pair_macro_mae_kcal_per_mol: float | None
    benefit_kcal_per_mol: float | None
    benefit_lower_95_kcal_per_mol: float | None

    result_id: str


def _validate_pair_model_lineage(
    cases: Sequence[DoubleMutantCase],
    campaign: ScientificCampaign,
    pair_model: RegularizedPairInteractionModel,
    /,
) -> None:
    by_id = {item.double_measurement.measurement_id: item for item in cases}
    if set(by_id) != set(campaign.case_ids):
        raise ValueError("Double-mutant cases must equal the pair-model campaign.")
    calibration_ids = _role_case_ids(campaign, "calibration")
    selected = tuple(by_id[case_id] for case_id in calibration_ids)
    manifests = {}
    for item in selected:
        for measurement in (
            item.double_measurement,
            item.first_single,
            item.second_single,
        ):
            if measurement.source_manifest is None:
                raise ValueError(
                    "Pair-model training measurement lacks its admitted source manifest."
                )
            manifests[measurement.source_manifest.manifest_id] = (
                measurement.source_manifest
            )
            if measurement.standard_error_kcal_per_mol is not None:
                if measurement.uncertainty_source_manifest is None:
                    raise ValueError(
                        "Pair-model uncertainty lacks its admitted source manifest."
                    )
                manifests[measurement.uncertainty_source_manifest.manifest_id] = (
                    measurement.uncertainty_source_manifest
                )
        for manifest in item.pair_features.source_manifests:
            manifests[manifest.manifest_id] = manifest
    for manifest in manifests.values():
        manifest.require_rights(training_use=True)
    matrix = np.stack([np.asarray(item.pair_features.values) for item in selected])
    expected_mean = np.mean(matrix, axis=0)
    empirical_scale = np.std(matrix, axis=0, ddof=0)
    expected_scale = np.where(empirical_scale == 0.0, 1.0, empirical_scale)
    expected_transform_id = canonical_fingerprint(
        {
            "kind": "protein-pair-feature-transform",
            "campaign_id": campaign.campaign_id,
            "feature_definition_id": selected[0].pair_features.feature_definition_id,
            "training_case_ids": list(calibration_ids),
            "mean": [float(value).hex() for value in expected_mean],
            "scale": [float(value).hex() for value in expected_scale],
        }
    )
    if (
        pair_model.campaign_id != campaign.campaign_id
        or pair_model.training_case_ids != calibration_ids
        or pair_model.training_pair_unit_ids
        != tuple(sorted(item.pair_features.pair_unit_id for item in selected))
        or pair_model.training_source_manifest_ids != tuple(sorted(manifests))
        or pair_model.transform_id != expected_transform_id
        or pair_model.feature_names != selected[0].pair_features.feature_names
        or pair_model.feature_definition_id
        != selected[0].pair_features.feature_definition_id
        or not np.array_equal(np.asarray(pair_model.mean), expected_mean)
        or not np.array_equal(np.asarray(pair_model.scale), expected_scale)
    ):
        raise ValueError(
            "Pair correction must bind the exact campaign calibration cases, "
            "sources, feature transform, and training rights."
        )


def qualify_double_mutant_challenge(
    cases: Sequence[DoubleMutantCase],
    first_features: Mapping[str, ProteinMutationFeatures],
    second_features: Mapping[str, ProteinMutationFeatures],
    campaign: ScientificCampaign,
    single_predictor: AbstractProteinStabilityPredictor,
    pair_model: RegularizedPairInteractionModel,
    uncertainties: Mapping[str, DoubleMutantUncertainty],
    support: SupportTuple,
    /,
    *,
    maximum_corrected_pair_macro_mae_kcal_per_mol: float,
    minimum_benefit_lower_95_kcal_per_mol: float,
    stage_evidence: Sequence[QualificationEvidence] = (),
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
) -> DoubleMutantChallengeResult:
    """Challenge frozen singles on held-out residue-pair units, not fitted rows."""
    values = tuple(cases)
    if not values or any(not isinstance(item, DoubleMutantCase) for item in values):
        raise TypeError("cases must contain DoubleMutantCase values.")
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    if not isinstance(single_predictor, AbstractProteinStabilityPredictor):
        raise TypeError("single_predictor must be an AbstractProteinStabilityPredictor.")
    if not isinstance(pair_model, RegularizedPairInteractionModel):
        raise TypeError("pair_model must be a RegularizedPairInteractionModel.")
    if not all(
        isinstance(value, Mapping)
        for value in (first_features, second_features, uncertainties)
    ):
        raise TypeError("Double-mutant feature and uncertainty inputs must be mappings.")
    _validate_pair_model_lineage(values, campaign, pair_model)
    double_measurement_by_case = {
        item.double_measurement.measurement_id: item.double_measurement for item in values
    }
    for key, uncertainty in uncertainties.items():
        if (
            key != uncertainty.case_id
            or key not in double_measurement_by_case
            or uncertainty.measurement_record_id
            != double_measurement_by_case[key].record_id
        ):
            raise ValueError(
                "Double-mutant uncertainty must bind its mapping key and exact "
                "measurement record."
            )
    if pair_model.single_model_id != single_predictor.model_id:
        raise ValueError(
            "Pair correction must name the exact frozen single-mutant model."
        )
    capability = "protein.double-mutant-interaction-prediction"
    if support.capability != capability:
        raise ValueError(f"support capability must be {capability!r}.")
    maximum_error = float(maximum_corrected_pair_macro_mae_kcal_per_mol)
    minimum_benefit = float(minimum_benefit_lower_95_kcal_per_mol)
    if not math.isfinite(maximum_error) or maximum_error <= 0.0:
        raise ValueError("maximum corrected pair MAE must be finite and positive.")
    if not math.isfinite(minimum_benefit) or minimum_benefit < 0.0:
        raise ValueError("minimum benefit lower bound must be finite and non-negative.")
    criteria = (
        ScientificMetricCriterion(
            "protein-double-mutant-corrected-pair-macro-mae",
            "at_most",
            None,
            maximum_error,
            "kcal/mol",
            "independent_unit_macro",
        ),
        ScientificMetricCriterion(
            "protein-double-mutant-benefit-lower-95",
            "at_least",
            minimum_benefit,
            None,
            "kcal/mol",
            "independent_unit_macro",
        ),
    )
    conditions = tuple(sorted({item.double_measurement.condition_id for item in values}))
    profile = ScientificClaimProfile(
        capability,
        support,
        ("delta_delta_g", "thermodynamic_coupling"),
        conditions,
        campaign.campaign_id,
        (
            "source-admission",
            "measurement-calibration",
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        ),
        criteria,
        "abstain-without-shared-wt-covariance-or-held-out-pair-support",
        (
            "single-mutant-model-change",
            "pair-feature-or-membership-change",
            "shared-wt-covariance-change",
            "assay-condition-domain-change",
        ),
        frozen_criteria_ids=campaign.criteria_ids,
    )
    if {criterion.criterion_id for criterion in criteria} != set(campaign.criteria_ids):
        raise ValueError("Double-mutant campaign criteria must exactly match the claim.")
    role_by_case = {
        case_id: role.name for role in campaign.roles for case_id in role.case_ids
    }
    pair_roles: dict[str, set[str]] = {}
    for item in values:
        case_id = item.double_measurement.measurement_id
        if case_id not in role_by_case:
            raise ValueError("Double-mutant case is outside the scientific campaign.")
        pair_roles.setdefault(item.pair_features.pair_unit_id, set()).add(
            role_by_case[case_id]
        )
    if any(len(roles) != 1 for roles in pair_roles.values()):
        raise ValueError(
            "All substitutions at one residue pair must share one campaign role."
        )
    locked = tuple(
        item
        for item in values
        if role_by_case[item.double_measurement.measurement_id] == "locked_evaluation"
    )
    locked_ids = tuple(item.double_measurement.measurement_id for item in locked)
    if set(first_features) != set(locked_ids) or set(second_features) != set(locked_ids):
        raise ValueError(
            "First and second feature mappings must cover locked double cases exactly."
        )
    for case_id, item in zip(locked_ids, locked, strict=True):
        first = first_features[case_id]
        second = second_features[case_id]
        if (
            not isinstance(first, ProteinMutationFeatures)
            or not isinstance(second, ProteinMutationFeatures)
            or not _feature_matches_measurement(first, item.first_single)
            or not _feature_matches_measurement(second, item.second_single)
            or {
                first.feature_id,
                second.feature_id,
            }
            != {
                item.pair_features.first_feature_id,
                item.pair_features.second_feature_id,
            }
        ):
            raise ValueError(
                "Double-mutant feature mappings must bind each key to the exact "
                "ordered component measurements and pair features."
            )
    leaked_pairs = {item.pair_features.pair_unit_id for item in locked} & set(
        pair_model.training_pair_unit_ids
    )
    if leaked_pairs:
        raise ValueError("Pair correction training pairs cannot enter locked evaluation.")
    locked_component_case_ids = {
        measurement.measurement_id
        for item in locked
        for measurement in (item.first_single, item.second_single)
    }
    if locked_component_case_ids.intersection(single_predictor.training_case_ids):
        raise ValueError(
            "Frozen single-mutant training cases cannot include locked components."
        )
    locked_families = {item.double_measurement.domain_family_id for item in locked}
    if locked_families.intersection(single_predictor.training_family_ids):
        raise ValueError(
            "Frozen single-mutant training families cannot enter locked double mutants."
        )
    pair_model_domain_supported = all(
        item.double_measurement.assay_channel == pair_model.assay_channel
        and item.double_measurement.condition_id == pair_model.condition_id
        and item.double_measurement.sign_convention == pair_model.sign_convention
        for item in locked
    )
    execution = dict(
        build_id=_identifier(build_id, "build_id"),
        environment_id=_identifier(environment_id, "environment_id"),
        backend=_identifier(backend, "backend"),
        topology=_identifier(topology, "topology"),
        precision=_identifier(precision, "precision"),
        reduction=_identifier(reduction, "reduction"),
        replay_id=_identifier(replay_id, "replay_id"),
        reviewer_id=_identifier(reviewer_id, "reviewer_id"),
        issued_at=issued_at,
        expires_at=expires_at,
    )
    raw_ids = tuple(
        sorted(
            {
                artifact_id
                for item in values
                for measurement in (
                    item.double_measurement,
                    item.first_single,
                    item.second_single,
                )
                for artifact_id in (
                    measurement.source_manifest_id,
                    measurement.uncertainty_source_manifest_id,
                )
                if artifact_id is not None
            }
            | {
                manifest.manifest_id
                for feature in (*first_features.values(), *second_features.values())
                for manifest in feature.source_manifests
            }
            | {value.source_id for value in uncertainties.values()}
        )
    )
    source = _evidence(
        campaign.campaign_id,
        "source-admission",
        "passed",
        "double-and-component-mutations-exactly-compose-with-grouped-lineage",
        raw_artifact_ids=raw_ids,
        **execution,
    )
    metric_values: dict[str, float] = {}
    case_ids: tuple[str, ...] = ()
    pair_ids: tuple[str, ...] = ()
    additive_values: tuple[float, ...] = ()
    corrected_values: tuple[float, ...] = ()
    observed_values: tuple[float, ...] = ()
    additive_macro = corrected_macro = benefit = benefit_lower = None
    outcome = "inconclusive"
    reason = "locked-double-mutant-cases-or-covariance-evidence-missing"
    if locked:
        locked_ids = tuple(item.double_measurement.measurement_id for item in locked)
        feature_complete = all(
            case_id in first_features
            and case_id in second_features
            and {
                first_features[case_id].feature_id,
                second_features[case_id].feature_id,
            }
            == {
                item.pair_features.first_feature_id,
                item.pair_features.second_feature_id,
            }
            for case_id, item in zip(locked_ids, locked, strict=True)
        )
        uncertainty_complete = set(locked_ids) == set(uncertainties)
        if feature_complete and uncertainty_complete and pair_model_domain_supported:
            additive_prediction = single_predictor.predict_additive(
                tuple(first_features[case_id] for case_id in locked_ids),
                tuple(second_features[case_id] for case_id in locked_ids),
            )
            correction_prediction = pair_model.predict(
                tuple(item.pair_features for item in locked)
            )
            valid = np.asarray(additive_prediction.valid) & np.asarray(
                correction_prediction.valid
            )
            if bool(np.all(valid)):
                additive_array = np.asarray(additive_prediction.mean)
                corrected_array = additive_array + np.asarray(correction_prediction.mean)
                observed_array = np.asarray(
                    [item.double_measurement.value_kcal_per_mol for item in locked]
                )
                pair_by_case = {
                    item.double_measurement.measurement_id: item.pair_features.pair_unit_id
                    for item in locked
                }
                pair_errors: list[tuple[float, float]] = []
                pair_contrast_variances: list[float] = []
                for pair_id in sorted(set(pair_by_case.values())):
                    indices = np.asarray(
                        [pair_by_case[case_id] == pair_id for case_id in locked_ids]
                    )
                    member_ids = tuple(
                        case_id
                        for case_id in locked_ids
                        if pair_by_case[case_id] == pair_id
                    )
                    pair_errors.append(
                        (
                            float(
                                np.mean(
                                    np.abs(
                                        additive_array[indices] - observed_array[indices]
                                    )
                                )
                            ),
                            float(
                                np.mean(
                                    np.abs(
                                        corrected_array[indices] - observed_array[indices]
                                    )
                                )
                            ),
                        )
                    )
                    contrast_variances = tuple(
                        uncertainties[
                            case_id
                        ].additive_observation_variance_kcal2_per_mol2
                        + uncertainties[
                            case_id
                        ].coupling_observation_variance_kcal2_per_mol2
                        + 2.0
                        * abs(
                            uncertainties[
                                case_id
                            ].additive_coupling_covariance_kcal2_per_mol2
                        )
                        for case_id in member_ids
                    )
                    pair_contrast_variances.append(
                        float(sum(contrast_variances) / len(member_ids) ** 2)
                    )
                additive_macro = float(np.mean([value[0] for value in pair_errors]))
                corrected_macro = float(np.mean([value[1] for value in pair_errors]))
                improvements = np.asarray([left - right for left, right in pair_errors])
                benefit = float(np.mean(improvements))
                if len(pair_errors) >= 2:
                    sampling_variance = float(np.var(improvements, ddof=1)) / len(
                        pair_errors
                    )
                    observation_variance = (
                        sum(pair_contrast_variances) / len(pair_errors) ** 2
                    )
                    benefit_lower = float(
                        benefit
                        - 1.96 * math.sqrt(sampling_variance + observation_variance)
                    )
                    metric_values = {
                        "protein-double-mutant-corrected-pair-macro-mae": corrected_macro,
                        "protein-double-mutant-benefit-lower-95": benefit_lower,
                    }
                    outcome = "passed"
                    reason = "held-out-pair-predictions-and-covariance-evidence-complete"
                else:
                    reason = "fewer-than-two-independent-held-out-residue-pair-units"
                case_ids = locked_ids
                pair_ids = tuple(pair_by_case[case_id] for case_id in locked_ids)
                additive_values = tuple(float(value) for value in additive_array)
                corrected_values = tuple(float(value) for value in corrected_array)
                observed_values = tuple(float(value) for value in observed_array)
            else:
                reason = "single-or-pair-model-abstained-on-locked-double-mutant"
    locked_evidence = _evidence(
        campaign.campaign_id,
        "locked-prediction",
        outcome,
        reason,
        raw_artifact_ids=raw_ids,
        **execution,
    )
    model_bound_stage_ids = frozenset(
        (
            "measurement-calibration",
            "parameter-identifiability",
            "predictive-calibration",
        )
    )
    required_model_ids = {single_predictor.model_id, pair_model.model_id}
    applicable_stage_evidence = tuple(
        item
        for item in stage_evidence
        if not model_bound_stage_ids.intersection(item.criteria_ids)
        or required_model_ids.issubset(item.subject_ids)
    )
    all_stage_evidence = (*applicable_stage_evidence, source, locked_evidence)
    evidence = profile.evaluate(
        metric_values,
        all_stage_evidence,
        metric_units={
            "protein-double-mutant-corrected-pair-macro-mae": "kcal/mol",
            "protein-double-mutant-benefit-lower-95": "kcal/mol",
        },
        metric_aggregations={
            "protein-double-mutant-corrected-pair-macro-mae": "independent_unit_macro",
            "protein-double-mutant-benefit-lower-95": "independent_unit_macro",
        },
        raw_artifact_ids=raw_ids,
        **execution,
    )
    result_id = canonical_fingerprint(
        {
            "kind": "protein-double-mutant-challenge",
            "claim_id": profile.claim_id,
            "evidence_id": evidence.evidence_id,
            "single_model_id": single_predictor.model_id,
            "pair_model_id": pair_model.model_id,
            "case_ids": list(case_ids),
            "pair_unit_ids": list(pair_ids),
        }
    )
    return DoubleMutantChallengeResult(
        profile,
        evidence,
        case_ids,
        pair_ids,
        additive_values,
        corrected_values,
        observed_values,
        additive_macro,
        corrected_macro,
        benefit,
        benefit_lower,
        result_id,
    )


__all__ = [
    "prepare_double_mutant_campaign",
    "DoubleMutantChallengeResult",
    "DoubleMutantUncertainty",
    "GroupedStabilityMetrics",
    "GroupedStabilityPrediction",
    "ProteinStabilityQualificationResult",
    "ProteinStabilityThresholds",
    "protein_stability_claim_profile",
    "qualify_double_mutant_challenge",
    "qualify_protein_stability",
    "source_admission_evidence",
]
