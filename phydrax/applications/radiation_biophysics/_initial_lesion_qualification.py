#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed staged qualification for external initial-lesion predictions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from phydrax._fingerprint import canonical_fingerprint
from phydrax.qualification import (
    QualificationEvidence,
    ScientificCampaign,
    ScientificClaimProfile,
)

from ._plasmid_gel import PlasmidGelEvaluation, PlasmidGelObservations
from ._qualification import RadiationStageEvidence
from .interchange._history_profile import (
    RadiationHistoryCoverage,
    TimedRadiationHistoryProfile,
)


_REQUIRED_DOMAIN_STAGES = (
    "dosimetry",
    "transport",
    "chemical-G",
    "target-reactions",
    "lesion-yields",
)
_REQUIRED_CLAIM_STAGES = frozenset(
    (
        "source-admission",
        "measurement-calibration",
        "numerical-validity",
        "chemical-validity",
        "locked-prediction",
        "external-transfer",
    )
)

_CAPABILITY_NAME = "radiation.initial-lesion-gel"
_OBSERVABLE_IDS = ("raw-plasmid-gel-bands",)
_CONDITION_DOMAIN_IDS = ("declared-physical-tuples",)
_SUPPORT_ATTRIBUTES = (("target", "plasmid"),)


def _require_exact_claim_scope(claim: ScientificClaimProfile, /) -> None:
    if (
        claim.capability_name != _CAPABILITY_NAME
        or claim.observable_ids != _OBSERVABLE_IDS
        or claim.condition_domain_ids != _CONDITION_DOMAIN_IDS
        or claim.support.capability != _CAPABILITY_NAME
        or claim.support.attributes != _SUPPORT_ATTRIBUTES
    ):
        raise ValueError(
            "Radiation gel claim capability, observables, conditions, and support "
            "must equal the exact assessed scope."
        )


@dataclass(frozen=True, slots=True)
class RadiationInitialLesionAssessment:
    """Domain readiness before an explicit ScientificClaimProfile evaluation."""

    history_coverage: RadiationHistoryCoverage
    gel_evaluation: PlasmidGelEvaluation
    campaign_id: str
    campaign_criteria_ids: tuple[str, ...]
    missing_prerequisites: tuple[str, ...]
    failed_checks: tuple[str, ...]
    metric_values: tuple[tuple[str, float], ...]
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
        """Return failed/inconclusive evidence or delegate a ready exact claim."""

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
                criteria_ids=("radiation-initial-lesion-domain-readiness", *issues),
                raw_artifact_ids=raw_artifact_ids,
                reviewer_id=reviewer_id,
                issued_at=issued_at,
                expires_at=expires_at,
                reason=(";".join(issues)),
                requalification_triggers=claim.invalidation_triggers,
                campaign_start_record_ids=(),
                campaign_observation_record_ids=(),
            )
        bound_subjects = {
            "measurement-calibration": frozenset(
                (claim.campaign_id, self.gel_evaluation.assay_id)
            ),
            "chemical-validity": frozenset(
                (claim.campaign_id, self.gel_evaluation.history_profile_id)
            ),
            "numerical-validity": frozenset(
                (
                    claim.campaign_id,
                    self.gel_evaluation.model_id,
                    self.gel_evaluation.fit_id,
                    self.gel_evaluation.prediction_id,
                )
            ),
            "locked-prediction": frozenset(
                (
                    claim.campaign_id,
                    self.gel_evaluation.model_id,
                    self.gel_evaluation.fit_id,
                    self.gel_evaluation.prediction_id,
                    self.gel_evaluation.prediction_source_artifact_id,
                    self.gel_evaluation.assay_id,
                    self.gel_evaluation.history_profile_id,
                    self.gel_evaluation.observation_id,
                    self.gel_evaluation.evaluation_id,
                    *self.gel_evaluation.physical_tuple_ids,
                )
            ),
            "external-transfer": frozenset(
                (
                    claim.campaign_id,
                    self.gel_evaluation.model_id,
                    self.gel_evaluation.fit_id,
                    self.gel_evaluation.prediction_id,
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
        metrics = dict(self.metric_values)
        return claim.evaluate(
            metrics,
            admissible_evidence,
            metric_units={name: "1" for name in metrics},
            metric_aggregations={name: "independent_unit_macro" for name in metrics},
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


def assess_radiation_initial_lesions(
    history_profile: TimedRadiationHistoryProfile,
    calibration_observations: PlasmidGelObservations,
    locked_observations: PlasmidGelObservations,
    locked_evaluation: PlasmidGelEvaluation,
    domain_stage_evidence: tuple[RadiationStageEvidence, ...],
    /,
    *,
    required_species_ids: tuple[str, ...],
    required_sample_times: tuple[float, ...],
    campaign: ScientificCampaign,
    claim: ScientificClaimProfile,
    maximum_day_macro_standardized_rms: float,
    commercial_use: bool = False,
) -> RadiationInitialLesionAssessment:
    """Assess stages and held-out grouping without coupling to transcript biology."""

    if not isinstance(history_profile, TimedRadiationHistoryProfile):
        raise TypeError("history_profile must be TimedRadiationHistoryProfile.")
    if not isinstance(calibration_observations, PlasmidGelObservations) or not isinstance(
        locked_observations, PlasmidGelObservations
    ):
        raise TypeError("Calibration and locked data must be plasmid-gel observations.")
    if not isinstance(locked_evaluation, PlasmidGelEvaluation):
        raise TypeError("locked_evaluation must be PlasmidGelEvaluation.")
    if locked_evaluation.observation_id != locked_observations.observation_id:
        raise ValueError(
            "Locked gel evaluation does not belong to the locked observations."
        )
    if locked_evaluation.history_profile_id != history_profile.profile_id:
        raise ValueError(
            "Locked gel evaluation does not belong to the assessed history profile."
        )
    if locked_evaluation.physical_tuple_ids != locked_observations.physical_tuple_ids:
        raise ValueError(
            "Locked gel evaluation physical tuples do not align with locked observations."
        )
    profile_tuples = frozenset(history_profile.physical_tuple_ids)
    if (
        not set(
            (
                *calibration_observations.physical_tuple_ids,
                *locked_observations.physical_tuple_ids,
            )
        )
        <= profile_tuples
    ):
        raise ValueError(
            "Calibration and locked gel tuples must belong to the assessed history profile."
        )
    if not isinstance(claim, ScientificClaimProfile):
        raise TypeError("claim must be ScientificClaimProfile.")
    _require_exact_claim_scope(claim)
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be ScientificCampaign.")
    if (
        locked_evaluation.campaign_id != campaign.campaign_id
        or claim.campaign_id != campaign.campaign_id
        or claim.frozen_criteria_ids != campaign.criteria_ids
    ):
        raise ValueError(
            "Gel prediction, claim, and frozen criteria must match the exact campaign."
        )
    if (
        not isinstance(maximum_day_macro_standardized_rms, (int, float))
        or maximum_day_macro_standardized_rms <= 0.0
    ):
        raise ValueError("The gel acceptance threshold must be positive.")
    coverage = history_profile.coverage(required_species_ids, required_sample_times)
    missing = [
        *(f"history-reference:{name}" for name in coverage.missing_references),
        *(f"chemical-species:{name}" for name in coverage.missing_species),
        *(
            f"chemical-history-grid:{species}@{time}"
            for species, time in coverage.missing_time_species_rows
        ),
    ]
    for name in locked_evaluation.uncertainty_limitations:
        stage = (
            "predictive-calibration"
            if name == "radiation-form-prediction-covariance"
            else "measurement-calibration"
        )
        missing.append(f"{stage}:{name}")
    for name, reference in (
        ("dosimetry", history_profile.dosimetry_reference),
        ("transport", history_profile.transport_reference),
        ("chemical-G", history_profile.chemical_reference),
    ):
        if reference is not None:
            if reference.uncertainty is None:
                missing.append(f"unquantified-stage-uncertainty:{name}")
            missing.extend(
                f"stage-rights:{name}:{reason}"
                for reason in reference.rights_refusal_reasons(
                    commercial_use=commercial_use
                )
            )
    for item in domain_stage_evidence:
        missing.extend(
            f"stage-rights:{item.stage}:{reason}"
            for reason in item.reference.rights_refusal_reasons(
                commercial_use=commercial_use
            )
        )
    missing.extend(
        f"claim-required-stage:{stage}"
        for stage in sorted(_REQUIRED_CLAIM_STAGES - set(claim.required_stage_ids))
    )
    failed: list[str] = []
    if set(calibration_observations.irradiation_day_ids) & set(
        locked_observations.irradiation_day_ids
    ):
        failed.append("irradiation-day-leakage")
    if set(calibration_observations.physical_tuple_ids) & set(
        locked_observations.physical_tuple_ids
    ):
        failed.append("physical-tuple-leakage")
    for stage in _REQUIRED_DOMAIN_STAGES:
        stage_records = tuple(
            item for item in domain_stage_evidence if item.stage == stage
        )
        matching = tuple(
            item
            for item in stage_records
            if history_profile.profile_id in item.upstream_artifact_ids
        )
        if not matching or not any(item.source_kind != "synthetic" for item in matching):
            missing.append(f"independent-stage:{stage}")
        if stage_records and not matching:
            missing.append(f"history-profile-stage-binding:{stage}")
        if any(not item.accepted for item in matching):
            failed.append(f"stage-criterion:{stage}")
    if not locked_evaluation.finite:
        failed.append("nonfinite-gel-evaluation")
    elif (
        locked_evaluation.day_macro_standardized_rms > maximum_day_macro_standardized_rms
    ):
        failed.append("gel-day-macro-standardized-rms")
    metrics = (
        (
            "gel-day-macro-standardized-rms",
            float(locked_evaluation.day_macro_standardized_rms),
        ),
    )
    missing_tuple = tuple(sorted(set(missing)))
    failed_tuple = tuple(sorted(set(failed)))
    assessment_id = canonical_fingerprint(
        {
            "kind": "radiation-initial-lesion-assessment",
            "history_profile": history_profile.profile_id,
            "calibration": calibration_observations.observation_id,
            "locked": locked_observations.observation_id,
            "evaluation": locked_evaluation.evaluation_id,
            "campaign": campaign.campaign_id,
            "campaign_criteria": campaign.criteria_ids,
            "domain_stages": tuple(
                (item.stage, item.accepted, item.reference.manifest_id)
                for item in domain_stage_evidence
            ),
            "missing": missing_tuple,
            "failed": failed_tuple,
            "metrics": metrics,
            "threshold": float(maximum_day_macro_standardized_rms),
            "claim": claim.claim_id,
        }
    )
    return RadiationInitialLesionAssessment(
        coverage,
        locked_evaluation,
        campaign.campaign_id,
        campaign.criteria_ids,
        missing_tuple,
        failed_tuple,
        metrics,
        assessment_id,
    )


__all__ = ["RadiationInitialLesionAssessment", "assess_radiation_initial_lesions"]
