#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Held-out pulse/chase observation and identifiable-rate claim evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax._fingerprint import canonical_fingerprint
from phydrax.qualification import (
    QualificationEvidence,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificClaimProfile,
)

from ._labeled_assay import LabeledTranscriptAssay, LabeledTranscriptCounts
from ._pulse_chase import (
    PulseChaseIdentifiability,
    PulseChasePrediction,
    PulseChaseSchedule,
)


_REQUIRED_CLAIM_STAGES = frozenset(
    (
        "source-admission",
        "measurement-calibration",
        "parameter-identifiability",
        "locked-prediction",
        "external-transfer",
    )
)

_CAPABILITY_NAME = "single-cell.pulse-chase-rates"
_OBSERVABLE_IDS = ("labeled-unlabeled-U-S",)
_CONDITION_DOMAIN_IDS = ("declared-pulse-chase-schedule",)
_SUPPORT_ATTRIBUTES = (("cell-line", "RPE1"),)


def _require_exact_claim_scope(claim: ScientificClaimProfile, /) -> None:
    if (
        claim.capability_name != _CAPABILITY_NAME
        or claim.observable_ids != _OBSERVABLE_IDS
        or claim.condition_domain_ids != _CONDITION_DOMAIN_IDS
        or claim.support.capability != _CAPABILITY_NAME
        or claim.support.attributes != _SUPPORT_ATTRIBUTES
    ):
        raise ValueError(
            "Pulse/chase claim capability, observables, conditions, and support "
            "must equal the exact assessed scope."
        )


@dataclass(frozen=True, slots=True)
class PulseChaseQualificationAssessment:
    predicted_observed_means: Array
    standardized_residuals: Array
    culture_macro_standardized_rms: float
    identified_rate_combinations: Array
    model_id: str
    schedule_id: str
    assay_id: str
    prediction_id: str
    identifiability_id: str
    calibration_observation_id: str
    locked_observation_id: str
    campaign_id: str
    campaign_criteria_ids: tuple[str, ...]
    missing_prerequisites: tuple[str, ...]
    failed_checks: tuple[str, ...]
    assessment_id: str

    @property
    def status(self) -> str:
        if self.failed_checks:
            return "failed"
        if self.missing_prerequisites:
            return "inconclusive"
        return "ready-for-claim-evaluation"

    def evaluate_claim(
        self,
        claim: ScientificClaimProfile,
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
        raw_artifact_ids: Sequence[str],
        reviewer_id: str,
        issued_at: int,
        expires_at: int,
    ) -> QualificationEvidence:
        if not isinstance(claim, ScientificClaimProfile):
            raise TypeError("claim must be ScientificClaimProfile.")
        _require_exact_claim_scope(claim)
        if (
            claim.campaign_id != self.campaign_id
            or claim.frozen_criteria_ids != self.campaign_criteria_ids
        ):
            raise ValueError(
                "Claim and frozen criteria do not match the assessed campaign."
            )
        issues = self.failed_checks or self.missing_prerequisites
        if issues:
            return QualificationEvidence(
                "scientific",
                "failed" if self.failed_checks else "inconclusive",
                (claim.claim_id, claim.campaign_id, claim.support.support_tuple_id),
                build_id=build_id,
                environment_id=environment_id,
                backend=backend,
                topology=topology,
                precision=precision,
                reduction=reduction,
                replay_id=replay_id,
                criteria_ids=("single-cell-pulse-chase-domain-readiness", *issues),
                raw_artifact_ids=raw_artifact_ids,
                reviewer_id=reviewer_id,
                issued_at=issued_at,
                expires_at=expires_at,
                reason=";".join(issues),
                requalification_triggers=claim.invalidation_triggers,
                campaign_start_record_ids=(),
                campaign_observation_record_ids=(),
            )
        bound_subjects = {
            "measurement-calibration": frozenset((claim.campaign_id, self.assay_id)),
            "parameter-identifiability": frozenset(
                (
                    claim.campaign_id,
                    self.model_id,
                    self.schedule_id,
                    self.assay_id,
                    self.identifiability_id,
                )
            ),
            "locked-prediction": frozenset(
                (
                    claim.campaign_id,
                    self.model_id,
                    self.schedule_id,
                    self.assay_id,
                    self.prediction_id,
                    self.locked_observation_id,
                )
            ),
        }
        admissible_evidence = tuple(
            item
            for item in stage_evidence
            if all(
                stage not in item.criteria_ids
                or claim.campaign_id not in item.subject_ids
                or required <= frozenset(item.subject_ids)
                for stage, required in bound_subjects.items()
            )
        )
        return claim.evaluate(
            {
                "pulse-chase-culture-macro-standardized-rms": self.culture_macro_standardized_rms
            },
            admissible_evidence,
            metric_units={"pulse-chase-culture-macro-standardized-rms": "1"},
            metric_aggregations={
                "pulse-chase-culture-macro-standardized-rms": "independent_unit_macro"
            },
            build_id=build_id,
            environment_id=environment_id,
            backend=backend,
            topology=topology,
            precision=precision,
            reduction=reduction,
            replay_id=replay_id,
            raw_artifact_ids=raw_artifact_ids,
            reviewer_id=reviewer_id,
            issued_at=issued_at,
            expires_at=expires_at,
        )


def assess_pulse_chase_prediction(
    assay: LabeledTranscriptAssay,
    schedule: PulseChaseSchedule,
    calibration_observations: LabeledTranscriptCounts,
    locked_observations: LabeledTranscriptCounts,
    prediction: PulseChasePrediction,
    identifiability: PulseChaseIdentifiability,
    /,
    *,
    campaign: ScientificCampaign,
    claim: ScientificClaimProfile,
    claimed_parameter_names: tuple[str, ...],
    timing_reference: ReferenceArtifactManifest | None,
    maximum_standardized_rms: float,
) -> PulseChaseQualificationAssessment:
    """Assess one frozen fit against disjoint locked cultures, plates, and times."""

    if not isinstance(assay, LabeledTranscriptAssay):
        raise TypeError("assay must be LabeledTranscriptAssay.")
    if not isinstance(schedule, PulseChaseSchedule):
        raise TypeError("schedule must be PulseChaseSchedule.")
    if not isinstance(
        calibration_observations, LabeledTranscriptCounts
    ) or not isinstance(locked_observations, LabeledTranscriptCounts):
        raise TypeError(
            "Pulse/chase assessment requires labeled transcript observations."
        )
    if not isinstance(prediction, PulseChasePrediction):
        raise TypeError("prediction must be a frozen PulseChasePrediction.")
    if not isinstance(identifiability, PulseChaseIdentifiability):
        raise TypeError("identifiability must be PulseChaseIdentifiability.")
    if not isinstance(claim, ScientificClaimProfile):
        raise TypeError("claim must be ScientificClaimProfile.")
    _require_exact_claim_scope(claim)
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be ScientificCampaign.")
    if (
        claim.campaign_id != campaign.campaign_id
        or claim.frozen_criteria_ids != campaign.criteria_ids
    ):
        raise ValueError(
            "Claim campaign and frozen criteria must match the exact campaign."
        )
    case_by_id = {case.case_id: case for case in campaign.cases}
    calibration_cases = tuple(
        case_by_id[case_id]
        for role in campaign.roles
        if role.name == "calibration"
        for case_id in role.case_ids
    )
    locked_cases = tuple(
        case_by_id[case_id]
        for role in campaign.roles
        if role.name == "locked_evaluation"
        for case_id in role.case_ids
    )
    if (
        set(prediction.fit_culture_ids)
        != {case.independent_unit_id for case in calibration_cases}
        or set(prediction.fit_plate_ids)
        != {case.preparation_id for case in calibration_cases}
        or not set(calibration_observations.source_parent_ids)
        <= {
            source_id
            for case in calibration_cases
            for source_id in case.source_manifest_ids
        }
    ):
        raise ValueError(
            "Prediction fit lineage does not match campaign calibration cases."
        )
    if (
        not set(locked_observations.culture_ids)
        <= {case.independent_unit_id for case in locked_cases}
        or not set(locked_observations.plate_ids)
        <= {case.preparation_id for case in locked_cases}
        or not set(locked_observations.source_parent_ids)
        <= {source_id for case in locked_cases for source_id in case.source_manifest_ids}
    ):
        raise ValueError("Locked observations do not match campaign locked cases.")
    if calibration_observations.gene != locked_observations.gene:
        raise ValueError(
            "Calibration and locked observations must address the same gene."
        )
    if (
        calibration_observations.assay_id != assay.assay_id
        or locked_observations.assay_id != assay.assay_id
    ):
        raise ValueError("Calibration and locked counts must use the assessed assay.")
    if calibration_observations.preprocessing_id != locked_observations.preprocessing_id:
        raise ValueError(
            "Calibration and locked counts must use the same preprocessing record."
        )
    if (
        calibration_observations.time_unit.unit_id
        != locked_observations.time_unit.unit_id
        or schedule.time_unit.unit_id != locked_observations.time_unit.unit_id
    ):
        raise ValueError(
            "Schedule, calibration, and locked physical times must use the same unit."
        )
    expected_fit_observations = (calibration_observations.observation_id,)
    expected_fit_cultures = frozenset(calibration_observations.culture_ids)
    expected_fit_plates = frozenset(calibration_observations.plate_ids)
    if (
        prediction.schedule_id != schedule.schedule_id
        or prediction.assay_id != assay.assay_id
        or prediction.preprocessing_id != calibration_observations.preprocessing_id
        or prediction.fit_observation_ids != expected_fit_observations
        or frozenset(prediction.fit_culture_ids) != expected_fit_cultures
        or frozenset(prediction.fit_plate_ids) != expected_fit_plates
    ):
        raise ValueError(
            "Prediction lineage does not match the exact schedule, assay, "
            "preprocessing, and calibration observations."
        )
    if (
        identifiability.model_id != prediction.model_id
        or identifiability.schedule_id != prediction.schedule_id
        or identifiability.assay_id != prediction.assay_id
        or identifiability.preprocessing_id != prediction.preprocessing_id
        or identifiability.fit_observation_ids != prediction.fit_observation_ids
        or identifiability.fit_culture_ids != prediction.fit_culture_ids
        or identifiability.fit_plate_ids != prediction.fit_plate_ids
    ):
        raise ValueError(
            "Identifiability evidence does not belong to the assessed prediction fit."
        )
    latent = np.asarray(prediction.latent_means, dtype=float)
    latent_covariance = np.asarray(prediction.latent_covariance, dtype=float)
    expected_shape = locked_observations.counts.shape
    if latent.shape != expected_shape:
        raise ValueError("Frozen latent prediction must align with locked counts.")
    if not math.isfinite(maximum_standardized_rms) or maximum_standardized_rms <= 0.0:
        raise ValueError("maximum_standardized_rms must be finite and positive.")
    predicted, observation_covariance = assay.conditional_moments(latent)
    propagated_latent_covariance = np.einsum(
        "ot,...tu,pu->...op",
        np.asarray(assay.observation_probabilities),
        latent_covariance,
        np.asarray(assay.observation_probabilities),
    )
    combined_covariance = (
        np.asarray(observation_covariance) + propagated_latent_covariance
    )
    mask = np.asarray(locked_observations.valid)
    raw_residuals = np.asarray(predicted) - np.asarray(locked_observations.counts)
    residuals = np.zeros_like(raw_residuals, dtype=float)
    whitening_valid = np.ones((raw_residuals.shape[0],), dtype=bool)
    covariance_missing: list[str] = []
    covariance_failed: list[str] = []
    for row in range(raw_residuals.shape[0]):
        active_indices = np.flatnonzero(mask[row])
        if not active_indices.size:
            continue
        covariance_block = combined_covariance[row][
            np.ix_(active_indices, active_indices)
        ]
        if not np.all(np.isfinite(covariance_block)) or not np.allclose(
            covariance_block,
            covariance_block.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            whitening_valid[row] = False
            covariance_failed.append(f"invalid-observation-covariance:cell-{row}")
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_block)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        tolerance = 1e-12 * scale
        if float(eigenvalues[0]) < -tolerance:
            whitening_valid[row] = False
            covariance_failed.append(f"invalid-observation-covariance:cell-{row}")
            continue
        if float(eigenvalues[0]) <= tolerance:
            whitening_valid[row] = False
            covariance_missing.append(f"singular-observation-covariance:cell-{row}")
            continue
        residuals[row, active_indices] = (
            eigenvectors.T @ raw_residuals[row, active_indices]
        ) / np.sqrt(eigenvalues)
    cultures = np.asarray(locked_observations.culture_ids)
    scores = []
    missing_cultures = []
    for culture in dict.fromkeys(locked_observations.culture_ids):
        selected = cultures == culture
        active = residuals[selected][mask[selected]]
        if not active.size or not np.all(whitening_valid[selected]):
            missing_cultures.append(f"locked-culture-whitened-observations:{culture}")
        else:
            scores.append(float(np.sqrt(np.mean(np.square(active)))))
    metric = float(np.mean(scores)) if scores else float("inf")
    missing = [*missing_cultures, *covariance_missing]
    if assay.calibration_covariance is None:
        missing.append("assay-calibration-covariance")
    if timing_reference is None:
        missing.append("physical-schedule-timing-calibration")
    elif not isinstance(timing_reference, ReferenceArtifactManifest):
        raise TypeError("timing_reference must be a manifest or None.")
    elif timing_reference.uncertainty is None:
        missing.append("physical-schedule-timing-calibration:unquantified-uncertainty")
    else:
        timing_reference.require_rights()
    claimed = tuple(claimed_parameter_names)
    if (
        not claimed
        or len(set(claimed)) != len(claimed)
        or any(name not in identifiability.parameter_names for name in claimed)
    ):
        raise ValueError(
            "Claimed rate names must be nonempty unique members of the sensitivity layout."
        )
    null = np.asarray(identifiability.unidentifiable_combinations)
    for name in claimed:
        index = identifiability.parameter_names.index(name)
        if (
            null.shape[0]
            and np.linalg.norm(null[:, index]) > identifiability.relative_tolerance
        ):
            missing.append(f"individually-identifiable-rate:{name}")
    missing.extend(
        f"claim-required-stage:{stage}"
        for stage in sorted(_REQUIRED_CLAIM_STAGES - set(claim.required_stage_ids))
    )
    failed = list(covariance_failed)
    if set(calibration_observations.cell_ids) & set(locked_observations.cell_ids):
        failed.append("cell-leakage")
    if calibration_observations.observation_id == locked_observations.observation_id:
        failed.append("observation-leakage")
    if calibration_observations.source_id == locked_observations.source_id:
        failed.append("source-leakage")
    if set(calibration_observations.source_parent_ids) & set(
        locked_observations.source_parent_ids
    ):
        failed.append("source-parent-leakage")
    if set(calibration_observations.preprocessing_parent_ids) & set(
        locked_observations.preprocessing_parent_ids
    ):
        failed.append("preprocessing-parent-leakage")
    locked_cultures = set(locked_observations.culture_ids)
    locked_plates = set(locked_observations.plate_ids)
    if expected_fit_cultures & locked_cultures:
        failed.append("culture-leakage")
    if expected_fit_plates & locked_plates:
        failed.append("plate-leakage")
    if set(prediction.fit_culture_ids) & locked_cultures:
        failed.append("prediction-fit-culture-leakage")
    if set(prediction.fit_plate_ids) & locked_plates:
        failed.append("prediction-fit-plate-leakage")
    calibration_times = set(np.asarray(calibration_observations.times).tolist())
    locked_times = set(np.asarray(locked_observations.times).tolist())
    if not locked_times - calibration_times:
        failed.append("no-heldout-physical-time-point")
    if not np.all(np.isfinite(np.asarray(predicted))):
        failed.append("nonfinite-observed-mean-prediction")
    if not np.all(np.isfinite(residuals[mask & whitening_valid[:, None]])):
        failed.append("nonfinite-whitened-residuals")
    if (
        not covariance_missing
        and not covariance_failed
        and metric > maximum_standardized_rms
    ):
        failed.append("pulse-chase-culture-macro-standardized-rms")
    missing_tuple = tuple(sorted(set(missing)))
    failed_tuple = tuple(sorted(set(failed)))
    assessment_id = canonical_fingerprint(
        {
            "kind": "single-cell-pulse-chase-assessment",
            "model": prediction.model_id,
            "schedule": schedule.schedule_id,
            "assay": assay.assay_id,
            "calibration": calibration_observations.observation_id,
            "locked": locked_observations.observation_id,
            "prediction": prediction.prediction_id,
            "identifiability": identifiability.evidence_id,
            "campaign": campaign.campaign_id,
            "campaign_criteria": campaign.criteria_ids,
            "claimed_parameters": claimed,
            "metric": float(metric).hex(),
            "missing": missing_tuple,
            "failed": failed_tuple,
            "claim": claim.claim_id,
        }
    )
    return PulseChaseQualificationAssessment(
        predicted,
        jnp.asarray(residuals),
        metric,
        identifiability.identified_combinations,
        prediction.model_id,
        schedule.schedule_id,
        assay.assay_id,
        prediction.prediction_id,
        identifiability.evidence_id,
        calibration_observations.observation_id,
        locked_observations.observation_id,
        campaign.campaign_id,
        campaign.criteria_ids,
        missing_tuple,
        failed_tuple,
        assessment_id,
    )


__all__ = ["PulseChaseQualificationAssessment", "assess_pulse_chase_prediction"]
