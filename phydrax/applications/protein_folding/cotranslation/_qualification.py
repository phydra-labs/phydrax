# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Length-resolved cotranslational observation laws and qualification evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....qualification import (
    QualificationEvidence,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificClaimProfile,
)
from ....units import conversion_factor, derived_unit, METER, SECOND, UnitDefinition


_PER_SECOND = derived_unit("1/s", ((SECOND, -1),))
_MANDATORY_CLAIM_STAGES = frozenset(
    (
        "source-admission",
        "measurement-calibration",
        "parameter-identifiability",
        "predictive-calibration",
        "locked-prediction",
    )
)
_CLAIM_CAPABILITY = "protein-cotranslation"


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return value


def _role_case_ids(campaign: ScientificCampaign, role_name: str, /) -> tuple[str, ...]:
    return next(role.case_ids for role in campaign.roles if role.name == role_name)


def _validate_locked_campaign(
    campaign: ScientificCampaign,
    observations: LengthResolvedCotranslationObservations,
    /,
) -> None:
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    if set(_role_case_ids(campaign, "locked_evaluation")) != set(observations.case_ids):
        raise ValueError(
            "Campaign locked-evaluation cases must exactly match cotranslation rows."
        )
    cases = {case.case_id: case for case in campaign.cases}
    for case_id, independent_unit_id, construct_id, preparation_id, condition_id in zip(
        observations.case_ids,
        observations.independent_unit_ids,
        observations.construct_ids,
        observations.preparation_ids,
        observations.condition_ids,
        strict=True,
    ):
        case = cases[case_id]
        if (
            case.independent_unit_id != independent_unit_id
            or case.construct_id != construct_id
            or case.preparation_id != preparation_id
            or case.condition_id != condition_id
            or observations.source.manifest_id not in case.source_manifest_ids
        ):
            raise ValueError(
                "Campaign case identity does not match cotranslation observations."
            )


@dataclass(frozen=True, slots=True, init=False)
class CotranslationObservationLaw:
    """Calibrated FRET-distance or arrest-release measurement transformation."""

    observable_kind: Literal["length-resolved-fret", "calibrated-arrest-release"]
    forster_radius: float | None
    forster_radius_standard_error: float | None
    length_unit: UnitDefinition | None
    calibration: ReferenceArtifactManifest
    law_id: str

    def __init__(
        self,
        observable_kind: Literal["length-resolved-fret", "calibrated-arrest-release"],
        calibration: ReferenceArtifactManifest,
        /,
        *,
        forster_radius: float | None = None,
        forster_radius_standard_error: float | None = None,
        length_unit: UnitDefinition | None = None,
    ):
        if observable_kind not in (
            "length-resolved-fret",
            "calibrated-arrest-release",
        ):
            raise ValueError("Unsupported cotranslational observation law.")
        if observable_kind == "length-resolved-fret":
            if (
                forster_radius is None
                or not math.isfinite(forster_radius)
                or forster_radius <= 0.0
                or forster_radius_standard_error is None
                or not math.isfinite(forster_radius_standard_error)
                or forster_radius_standard_error <= 0.0
                or not isinstance(length_unit, UnitDefinition)
            ):
                raise ValueError(
                    "FRET observations require a positive radius, uncertainty, and length unit."
                )
            conversion_factor(length_unit, METER)
            radius = float(forster_radius)
            radius_error = float(forster_radius_standard_error)
            unit = length_unit
        elif any(
            value is not None
            for value in (
                forster_radius,
                forster_radius_standard_error,
                length_unit,
            )
        ):
            raise ValueError(
                "Arrest-release observations do not accept hidden FRET calibration."
            )
        else:
            radius = None
            radius_error = None
            unit = None
        if not isinstance(calibration, ReferenceArtifactManifest):
            raise TypeError(
                "Observation calibration must be a ReferenceArtifactManifest."
            )
        calibration.require_rights()
        calibration.require_uncertainty()
        object.__setattr__(self, "observable_kind", observable_kind)
        object.__setattr__(self, "forster_radius", radius)
        object.__setattr__(self, "forster_radius_standard_error", radius_error)
        object.__setattr__(self, "length_unit", unit)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(
            self,
            "law_id",
            canonical_fingerprint(
                {
                    "kind": "cotranslation-observation-law",
                    "observable_kind": observable_kind,
                    "forster_radius": radius,
                    "forster_radius_standard_error": radius_error,
                    "length_unit": None if unit is None else unit.unit_id,
                    "calibration": calibration.manifest_id,
                }
            ),
        )

    def predict(
        self,
        latent: ArrayLike,
        latent_standard_errors: ArrayLike,
        latent_unit: UnitDefinition,
        measured_dwell_times: ArrayLike,
        dwell_time_standard_errors: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Map typed latent values and their uncertainty to response moments."""

        values = jnp.asarray(latent, dtype=float)
        latent_errors = jnp.asarray(latent_standard_errors, dtype=values.dtype)
        dwell = jnp.asarray(measured_dwell_times, dtype=values.dtype)
        dwell_errors = jnp.asarray(dwell_time_standard_errors, dtype=values.dtype)
        if (
            latent_errors.shape != values.shape
            or dwell.shape != values.shape
            or dwell_errors.shape != values.shape
        ):
            raise ValueError(
                "Latent and measured dwell values and uncertainties must align."
            )
        if self.observable_kind == "length-resolved-fret":
            factor = float(conversion_factor(latent_unit, self.length_unit))
            distance = values * factor
            distance_error = latent_errors * factor
            ratio = distance / self.forster_radius
            denominator = 1.0 + ratio**6
            prediction = 1.0 / denominator
            distance_derivative = (
                6.0 * jnp.abs(ratio) ** 5 / (self.forster_radius * denominator**2)
            )
            radius_derivative = 6.0 * ratio**6 / (self.forster_radius * denominator**2)
            variance = (distance_derivative * distance_error) ** 2 + (
                radius_derivative * self.forster_radius_standard_error
            ) ** 2
            return prediction, jnp.sqrt(jnp.maximum(variance, 0.0))
        factor = float(conversion_factor(latent_unit, _PER_SECOND))
        rate = values * factor
        rate_error = latent_errors * factor
        survival = jnp.exp(-rate * dwell)
        prediction = -jnp.expm1(-rate * dwell)
        variance = survival**2 * ((dwell * rate_error) ** 2 + (rate * dwell_errors) ** 2)
        return prediction, jnp.sqrt(jnp.maximum(variance, 0.0))


@dataclass(frozen=True, slots=True, init=False)
class LengthResolvedCotranslationObservations:
    """Measured physical timing plus FRET or arrest-peptide response by chain length."""

    case_ids: tuple[str, ...]
    independent_unit_ids: tuple[str, ...]
    construct_ids: tuple[str, ...]
    preparation_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    nascent_lengths: Array
    measured_dwell_times: Array
    dwell_time_standard_errors: Array
    values: Array
    standard_errors: Array
    time_unit: UnitDefinition
    source: ReferenceArtifactManifest
    timing_reference: ReferenceArtifactManifest | None
    observation_id: str

    def __init__(
        self,
        case_ids: tuple[str, ...],
        independent_unit_ids: tuple[str, ...],
        preparation_ids: tuple[str, ...],
        condition_ids: tuple[str, ...],
        /,
        *,
        construct_ids: tuple[str, ...],
        nascent_lengths: ArrayLike,
        measured_dwell_times: ArrayLike,
        dwell_time_standard_errors: ArrayLike,
        values: ArrayLike,
        standard_errors: ArrayLike,
        time_unit: UnitDefinition,
        timing_semantics: Literal["measured-dwell-time"],
        source: ReferenceArtifactManifest,
        timing_reference: ReferenceArtifactManifest | None,
    ):
        cases = tuple(case_ids)
        independent = tuple(independent_unit_ids)
        preparations = tuple(preparation_ids)
        constructs = tuple(construct_ids)
        conditions = tuple(condition_ids)
        n = len(cases)
        if (
            not n
            or len(set(cases)) != n
            or any(
                len(items) != n
                for items in (independent, constructs, preparations, conditions)
            )
            or any(
                not value or value != value.strip()
                for value in (
                    *cases,
                    *independent,
                    *constructs,
                    *preparations,
                    *conditions,
                )
            )
        ):
            raise ValueError(
                "Cotranslation rows require exact case and preparation grouping."
            )
        if timing_semantics != "measured-dwell-time":
            raise ValueError(
                "Ribosome density or profiling position is not a measured dwell time."
            )
        lengths = np.asarray(nascent_lengths)
        time_factor = float(conversion_factor(time_unit, SECOND))
        dwell = np.asarray(measured_dwell_times, dtype=float) * time_factor
        dwell_errors = np.asarray(dwell_time_standard_errors, dtype=float) * time_factor
        observed = np.asarray(values, dtype=float)
        errors = np.asarray(standard_errors, dtype=float)
        if (
            lengths.shape != (n,)
            or lengths.dtype.kind not in "iu"
            or np.any(lengths <= 0)
            or dwell.shape != (n,)
            or dwell_errors.shape != (n,)
            or observed.shape != (n,)
            or errors.shape != (n,)
            or not all(
                np.all(np.isfinite(array))
                for array in (dwell, dwell_errors, observed, errors)
            )
            or np.any(dwell <= 0.0)
            or np.any(dwell_errors <= 0.0)
            or np.any(errors <= 0.0)
            or np.any((observed < 0.0) | (observed > 1.0))
        ):
            raise ValueError(
                "Length-resolved observations need positive lengths/timing/errors and responses in [0, 1]."
            )
        if not isinstance(source, ReferenceArtifactManifest):
            raise TypeError("Cotranslation observations require a source manifest.")
        if timing_reference is not None and not isinstance(
            timing_reference, ReferenceArtifactManifest
        ):
            raise TypeError("timing_reference must be a manifest or None.")
        source.require_rights()
        source.require_uncertainty()
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "independent_unit_ids", independent)
        object.__setattr__(self, "construct_ids", constructs)
        object.__setattr__(self, "preparation_ids", preparations)
        object.__setattr__(self, "condition_ids", conditions)
        object.__setattr__(self, "nascent_lengths", jnp.asarray(lengths))
        object.__setattr__(self, "measured_dwell_times", jnp.asarray(dwell))
        object.__setattr__(self, "dwell_time_standard_errors", jnp.asarray(dwell_errors))
        object.__setattr__(self, "values", jnp.asarray(observed))
        object.__setattr__(self, "standard_errors", jnp.asarray(errors))
        object.__setattr__(self, "time_unit", SECOND)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "timing_reference", timing_reference)
        object.__setattr__(
            self,
            "observation_id",
            canonical_fingerprint(
                {
                    "kind": "length-resolved-cotranslation-observations",
                    "cases": cases,
                    "independent_units": independent,
                    "constructs": constructs,
                    "preparations": preparations,
                    "conditions": conditions,
                    "nascent_lengths": lengths.tolist(),
                    "measured_dwell_times": dwell.tolist(),
                    "dwell_time_standard_errors": dwell_errors.tolist(),
                    "values": observed.tolist(),
                    "standard_errors": errors.tolist(),
                    "time_unit": SECOND.unit_id,
                    "source": source.manifest_id,
                    "timing_reference": (
                        None if timing_reference is None else timing_reference.manifest_id
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class CotranslationModelFit:
    """Content-bound fitted parameters, artifacts, execution, and campaign roles."""

    campaign: ScientificCampaign
    model_id: str
    calibration_case_ids: tuple[str, ...]
    model_selection_case_ids: tuple[str, ...]
    source_manifest_ids: tuple[str, ...]
    independent_unit_ids: tuple[str, ...]
    preparation_ids: tuple[str, ...]
    construct_ids: tuple[str, ...]
    parameter_artifact_ids: tuple[str, ...]
    fitted_parameter_id: str
    prediction_code_manifest_id: str
    fit_execution_evidence_id: str
    fit_integrity_ids: tuple[str, ...]
    fit_id: str

    def __init__(
        self,
        model_id: str,
        campaign: ScientificCampaign,
        fitted_parameters: ArrayLike,
        parameter_artifacts: Sequence[ReferenceArtifactManifest],
        prediction_code: ReferenceArtifactManifest,
        fit_execution_evidence: QualificationEvidence,
        /,
    ):
        if not isinstance(campaign, ScientificCampaign):
            raise TypeError("campaign must be a ScientificCampaign.")
        model = _identifier(model_id, "model_id")
        calibration = _role_case_ids(campaign, "calibration")
        selection = _role_case_ids(campaign, "model_selection")
        cases = {case.case_id: case for case in campaign.cases}
        lineage = tuple(cases[case_id] for case_id in (*calibration, *selection))
        sources = tuple(
            sorted(
                {source_id for case in lineage for source_id in case.source_manifest_ids}
            )
        )
        units = tuple(sorted({case.independent_unit_id for case in lineage}))
        preparations = tuple(sorted({case.preparation_id for case in lineage}))
        constructs = tuple(sorted({case.construct_id for case in lineage}))
        parameters = np.asarray(fitted_parameters, dtype=float)
        if not parameters.size or not np.all(np.isfinite(parameters)):
            raise ValueError(
                "Fitted cotranslation parameters must be finite and nonempty."
            )
        parameter_id = canonical_fingerprint(
            {
                "kind": "cotranslation-fitted-parameters",
                "values": array_tree_fingerprint(parameters),
            }
        )
        artifacts = tuple(parameter_artifacts)
        if not artifacts or any(
            not isinstance(item, ReferenceArtifactManifest) for item in artifacts
        ):
            raise TypeError("parameter_artifacts must contain admitted manifests.")
        artifact_ids = tuple(sorted(item.manifest_id for item in artifacts))
        if not set(sources) <= set(artifact_ids):
            raise ValueError(
                "Parameter/source artifacts must cover every fit campaign source."
            )
        for artifact in artifacts:
            artifact.require_rights(training_use=True)
        if not isinstance(prediction_code, ReferenceArtifactManifest):
            raise TypeError("prediction_code must be a ReferenceArtifactManifest.")
        prediction_code.require_rights()
        if not isinstance(fit_execution_evidence, QualificationEvidence):
            raise TypeError("fit_execution_evidence must be QualificationEvidence.")
        required_subjects = {
            campaign.campaign_id,
            model,
            parameter_id,
            prediction_code.manifest_id,
            *sources,
        }
        if (
            fit_execution_evidence.outcome != "passed"
            or "fit-execution" not in fit_execution_evidence.criteria_ids
            or not required_subjects <= set(fit_execution_evidence.subject_ids)
        ):
            raise ValueError(
                "Fit execution evidence must bind campaign, model, parameters, code, and sources."
            )
        integrity = (
            fit_execution_evidence.evidence_id,
            parameter_id,
            prediction_code.manifest_id,
            *artifact_ids,
        )
        for name, value in (
            ("campaign", campaign),
            ("model_id", model),
            ("calibration_case_ids", calibration),
            ("model_selection_case_ids", selection),
            ("source_manifest_ids", sources),
            ("independent_unit_ids", units),
            ("preparation_ids", preparations),
            ("construct_ids", constructs),
            ("parameter_artifact_ids", artifact_ids),
            ("fitted_parameter_id", parameter_id),
            ("prediction_code_manifest_id", prediction_code.manifest_id),
            ("fit_execution_evidence_id", fit_execution_evidence.evidence_id),
            ("fit_integrity_ids", tuple(sorted(set(integrity)))),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "fit_id",
            canonical_fingerprint(
                {
                    "kind": "cotranslation-model-fit",
                    "campaign": campaign.campaign_id,
                    "model": model,
                    "calibration_cases": calibration,
                    "model_selection_cases": selection,
                    "sources": sources,
                    "parameter_artifacts": artifact_ids,
                    "fitted_parameters": parameter_id,
                    "prediction_code": prediction_code.manifest_id,
                    "fit_execution_evidence": fit_execution_evidence.evidence_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class CotranslationModelPrediction:
    """Frozen latent prediction with explicit content-bound fit lineage."""

    latent_values: Array
    latent_standard_errors: Array
    latent_unit: UnitDefinition
    model_id: str
    fit_id: str
    campaign_id: str
    observation_id: str
    campaign_criteria_ids: tuple[str, ...]
    fit_integrity_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    fit_case_ids: tuple[str, ...]
    model_selection_case_ids: tuple[str, ...]
    fit_independent_unit_ids: tuple[str, ...]
    fit_preparation_ids: tuple[str, ...]
    fit_construct_ids: tuple[str, ...]
    prediction_id: str

    def __init__(
        self,
        observations: LengthResolvedCotranslationObservations,
        latent_values: ArrayLike,
        latent_standard_errors: ArrayLike,
        fit: CotranslationModelFit,
        /,
        *,
        latent_unit: UnitDefinition,
    ):
        if not isinstance(observations, LengthResolvedCotranslationObservations):
            raise TypeError(
                "observations must be LengthResolvedCotranslationObservations."
            )
        if not isinstance(fit, CotranslationModelFit):
            raise TypeError("fit must be a CotranslationModelFit.")
        _validate_locked_campaign(fit.campaign, observations)
        if not isinstance(latent_unit, UnitDefinition):
            raise TypeError("latent_unit must be a UnitDefinition.")
        values = np.asarray(latent_values, dtype=float)
        errors = np.asarray(latent_standard_errors, dtype=float)
        if (
            values.shape != observations.values.shape
            or errors.shape != values.shape
            or not np.all(np.isfinite(errors))
            or np.any(errors <= 0.0)
        ):
            raise ValueError(
                "Latent prediction and positive uncertainty must match every row."
            )
        for name, value in (
            ("latent_values", jnp.asarray(values)),
            ("latent_standard_errors", jnp.asarray(errors)),
            ("latent_unit", latent_unit),
            ("model_id", fit.model_id),
            ("fit_id", fit.fit_id),
            ("campaign_id", fit.campaign.campaign_id),
            ("campaign_criteria_ids", fit.campaign.criteria_ids),
            ("fit_integrity_ids", fit.fit_integrity_ids),
            ("observation_id", observations.observation_id),
            ("case_ids", observations.case_ids),
            ("fit_case_ids", fit.calibration_case_ids),
            ("model_selection_case_ids", fit.model_selection_case_ids),
            ("fit_independent_unit_ids", fit.independent_unit_ids),
            ("fit_preparation_ids", fit.preparation_ids),
            ("fit_construct_ids", fit.construct_ids),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "prediction_id",
            canonical_fingerprint(
                {
                    "kind": "locked-cotranslation-model-prediction",
                    "latent_values": array_tree_fingerprint(values),
                    "latent_standard_errors": array_tree_fingerprint(errors),
                    "latent_unit": latent_unit.unit_id,
                    "model": fit.model_id,
                    "fit": fit.fit_id,
                    "campaign": fit.campaign.campaign_id,
                    "fit_integrity_ids": fit.fit_integrity_ids,
                    "observation": observations.observation_id,
                    "cases": observations.case_ids,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CotranslationQualificationAssessment:
    predicted_observations: Array
    predictive_standard_errors: Array
    standardized_residuals: Array
    independent_unit_macro_standardized_rms: float
    campaign_id: str
    model_id: str
    fit_id: str
    prediction_id: str
    qualification_evidence_ids: tuple[str, ...]
    campaign_criteria_ids: tuple[str, ...]
    capability_name: str
    observable_ids: tuple[str, ...]
    condition_domain_ids: tuple[str, ...]
    support_attributes: tuple[tuple[str, str], ...]
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
        if (
            claim.capability_name != self.capability_name
            or claim.observable_ids != self.observable_ids
            or claim.condition_domain_ids != self.condition_domain_ids
            or claim.support.attributes != self.support_attributes
        ):
            raise ValueError(
                "Scientific claim scope must exactly match the cotranslation assessment."
            )
        if claim.campaign_id != self.campaign_id:
            raise ValueError(
                "Scientific claim profile must bind the prediction campaign."
            )
        if tuple(claim.frozen_criteria_ids) != self.campaign_criteria_ids:
            raise ValueError(
                "Scientific claim criteria must exactly match the prediction campaign."
            )
        if not _MANDATORY_CLAIM_STAGES <= set(claim.required_stage_ids):
            raise ValueError(
                "Cotranslation claim profile omits mandatory scientific stages."
            )
        supplied_evidence_ids = {
            evidence.evidence_id
            for evidence in stage_evidence
            if isinstance(evidence, QualificationEvidence)
        }
        if not set(self.qualification_evidence_ids) <= supplied_evidence_ids:
            raise ValueError(
                "Claim evaluation must retain prediction qualification evidence."
            )
        issues = self.failed_checks or self.missing_prerequisites
        if issues:
            return QualificationEvidence(
                "scientific",
                "failed" if self.failed_checks else "inconclusive",
                (
                    claim.claim_id,
                    claim.campaign_id,
                    claim.support.support_tuple_id,
                    *self.qualification_evidence_ids,
                ),
                build_id=build_id,
                environment_id=environment_id,
                backend=backend,
                topology=topology,
                precision=precision,
                reduction=reduction,
                replay_id=replay_id,
                criteria_ids=("cotranslation-domain-readiness", *issues),
                raw_artifact_ids=raw_artifact_ids,
                reviewer_id=reviewer_id,
                issued_at=issued_at,
                expires_at=expires_at,
                reason=";".join(issues),
                requalification_triggers=claim.invalidation_triggers,
                campaign_start_record_ids=(),
                campaign_observation_record_ids=(),
            )
        metrics = {
            "cotranslation-macro-standardized-rms": self.independent_unit_macro_standardized_rms
        }
        model_stages = {
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        }
        bound_stage_evidence = tuple(
            evidence
            for evidence in stage_evidence
            if not model_stages.intersection(evidence.criteria_ids)
            or evidence.evidence_id in self.qualification_evidence_ids
        )
        return claim.evaluate(
            metrics,
            bound_stage_evidence,
            metric_units={"cotranslation-macro-standardized-rms": "1"},
            metric_aggregations={
                "cotranslation-macro-standardized-rms": "independent_unit_macro"
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


def _prediction_evidence_status(
    evidence: Sequence[QualificationEvidence],
    prediction: CotranslationModelPrediction,
    /,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if any(not isinstance(item, QualificationEvidence) for item in evidence):
        raise TypeError("prediction_evidence must contain QualificationEvidence.")
    missing = []
    failed = []
    accepted = []
    for stage_id, include_prediction in (
        ("parameter-identifiability", False),
        ("predictive-calibration", False),
        ("locked-prediction", True),
    ):
        subjects = {
            prediction.campaign_id,
            prediction.model_id,
            prediction.fit_id,
            *prediction.fit_integrity_ids,
        }
        if include_prediction:
            subjects.add(prediction.prediction_id)
        matching = tuple(
            item
            for item in evidence
            if item.evidence_kind == "scientific"
            and stage_id in item.criteria_ids
            and subjects <= set(item.subject_ids)
        )
        accepted.extend(item.evidence_id for item in matching)
        if any(item.outcome == "failed" for item in matching):
            failed.append(stage_id)
        elif not matching or any(item.outcome == "inconclusive" for item in matching):
            missing.append(stage_id)
    return tuple(missing), tuple(failed), tuple(sorted(set(accepted)))


def assess_cotranslation_prediction(
    law: CotranslationObservationLaw,
    observations: LengthResolvedCotranslationObservations,
    latent_prediction: CotranslationModelPrediction,
    /,
    *,
    maximum_standardized_rms: float,
    prediction_evidence: Sequence[QualificationEvidence],
) -> CotranslationQualificationAssessment:
    """Evaluate a frozen held-out prediction through the calibrated observation law."""

    if not isinstance(law, CotranslationObservationLaw) or not isinstance(
        observations, LengthResolvedCotranslationObservations
    ):
        raise TypeError("Cotranslation assessment needs a law and observations.")
    if not isinstance(latent_prediction, CotranslationModelPrediction):
        raise TypeError(
            "latent_prediction must be a CotranslationModelPrediction record."
        )
    if (
        latent_prediction.observation_id != observations.observation_id
        or latent_prediction.case_ids != observations.case_ids
    ):
        raise ValueError("Cotranslation prediction targets do not match observations.")
    if not math.isfinite(maximum_standardized_rms) or maximum_standardized_rms <= 0.0:
        raise ValueError("maximum_standardized_rms must be finite and positive.")
    latent = np.asarray(latent_prediction.latent_values)
    prediction, predictive_errors = law.predict(
        latent_prediction.latent_values,
        latent_prediction.latent_standard_errors,
        latent_prediction.latent_unit,
        observations.measured_dwell_times,
        observations.dwell_time_standard_errors,
    )
    total_errors = jnp.sqrt(observations.standard_errors**2 + predictive_errors**2)
    residuals = (prediction - observations.values) / total_errors
    host = np.asarray(residuals)
    identities = np.asarray(observations.independent_unit_ids)
    scores = [
        float(np.sqrt(np.mean(np.square(host[identities == identity]))))
        for identity in dict.fromkeys(observations.independent_unit_ids)
    ]
    metric = float(np.mean(scores))
    missing = []
    timing = observations.timing_reference
    if timing is None:
        missing.append("measured-timing-calibration")
    elif timing.uncertainty is None:
        missing.append("measured-timing-calibration:unquantified-uncertainty")
    else:
        timing.require_rights()
    failed = []
    evidence_missing, evidence_failed, evidence_ids = _prediction_evidence_status(
        prediction_evidence, latent_prediction
    )
    missing.extend(evidence_missing)
    failed.extend(evidence_failed)
    if not np.all(np.isfinite(np.asarray(prediction))) or not np.all(
        np.isfinite(np.asarray(predictive_errors))
    ):
        failed.append("nonfinite-observation-prediction")
    if law.observable_kind == "length-resolved-fret" and np.any(latent < 0.0):
        failed.append("negative-fret-distance")
    if law.observable_kind == "calibrated-arrest-release" and np.any(latent < 0.0):
        failed.append("negative-arrest-release-rate")
    if set(latent_prediction.fit_independent_unit_ids) & set(
        observations.independent_unit_ids
    ):
        failed.append("independent-unit-leakage")
    if set(latent_prediction.fit_preparation_ids) & set(observations.preparation_ids):
        failed.append("preparation-leakage")
    if set(latent_prediction.fit_construct_ids) & set(observations.construct_ids):
        failed.append("construct-leakage")
    if metric > maximum_standardized_rms:
        failed.append("cotranslation-macro-standardized-rms")
    missing_tuple = tuple(sorted(set(missing)))
    failed_tuple = tuple(sorted(set(failed)))
    capability_name = _CLAIM_CAPABILITY
    observable_ids = (observations.observation_id,)
    condition_domain_ids = tuple(sorted(set(observations.condition_ids)))
    support_attributes = tuple(
        sorted(
            {
                "construct_scope_id": canonical_fingerprint(
                    tuple(sorted(set(observations.construct_ids)))
                ),
                "fit_id": latent_prediction.fit_id,
                "law_id": law.law_id,
                "model_id": latent_prediction.model_id,
                "observable_kind": law.observable_kind,
                "observation_id": observations.observation_id,
                "prediction_id": latent_prediction.prediction_id,
                "time_unit_id": observations.time_unit.unit_id,
            }.items()
        )
    )
    assessment_id = canonical_fingerprint(
        {
            "kind": "cotranslation-qualification-assessment",
            "law": law.law_id,
            "observations": observations.observation_id,
            "prediction": latent_prediction.prediction_id,
            "capability": capability_name,
            "observable_ids": observable_ids,
            "condition_domain_ids": condition_domain_ids,
            "support_attributes": support_attributes,
            "predictive_standard_errors": array_tree_fingerprint(
                np.asarray(predictive_errors)
            ),
            "metric": metric,
            "missing": missing_tuple,
            "campaign_criteria_ids": latent_prediction.campaign_criteria_ids,
            "failed": failed_tuple,
            "qualification_evidence": evidence_ids,
        }
    )
    return CotranslationQualificationAssessment(
        prediction,
        predictive_errors,
        residuals,
        metric,
        latent_prediction.campaign_id,
        latent_prediction.model_id,
        latent_prediction.fit_id,
        latent_prediction.prediction_id,
        evidence_ids,
        latent_prediction.campaign_criteria_ids,
        capability_name,
        observable_ids,
        condition_domain_ids,
        support_attributes,
        missing_tuple,
        failed_tuple,
        assessment_id,
    )


__all__ = [
    "CotranslationModelFit",
    "CotranslationModelPrediction",
    "CotranslationObservationLaw",
    "CotranslationQualificationAssessment",
    "LengthResolvedCotranslationObservations",
    "assess_cotranslation_prediction",
]
