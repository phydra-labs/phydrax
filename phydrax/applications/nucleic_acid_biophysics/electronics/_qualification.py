# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Time-resolved nucleotide charge-transfer qualification against a kinetic baseline."""

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
from ....units import conversion_factor, ONE, SECOND, UnitDefinition


_PROBABILITY_NORMALIZATION_TOLERANCE = 1.0e-8
_MANDATORY_CLAIM_STAGES = frozenset(
    (
        "source-admission",
        "measurement-calibration",
        "parameter-identifiability",
        "predictive-calibration",
        "locked-prediction",
    )
)
_CLAIM_CAPABILITY = "nucleotide-electronics"


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
    observations: ChargeTransferObservationSeries,
    /,
) -> None:
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    if set(_role_case_ids(campaign, "locked_evaluation")) != set(observations.series_ids):
        raise ValueError(
            "Campaign locked-evaluation cases must exactly match the scored series."
        )
    cases = {case.case_id: case for case in campaign.cases}
    for case_id, independent_unit_id, condition_id in zip(
        observations.series_ids,
        observations.independent_unit_ids,
        observations.condition_ids,
        strict=True,
    ):
        case = cases[case_id]
        if (
            case.independent_unit_id != independent_unit_id
            or case.condition_id != condition_id
            or case.construct_id != observations.sequence_id
            or observations.source.manifest_id not in case.source_manifest_ids
        ):
            raise ValueError(
                "Campaign case identity does not match the scored electronic series."
            )


@dataclass(frozen=True, slots=True, init=False)
class ChargeTransferObservationSeries:
    """One sequence/environment series of time-resolved experimental observables."""

    series_ids: tuple[str, ...]
    independent_unit_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    sequence_id: str
    environment_id: str
    observable_kind: Literal["charge-transfer-population", "spectroscopic-signal"]
    times: Array
    values: Array
    standard_errors: Array
    valid: Array
    time_unit: UnitDefinition
    observable_unit: UnitDefinition
    source: ReferenceArtifactManifest
    observation_id: str

    def __init__(
        self,
        series_ids: tuple[str, ...],
        independent_unit_ids: tuple[str, ...],
        condition_ids: tuple[str, ...],
        /,
        *,
        sequence_id: str,
        environment_id: str,
        observable_kind: Literal["charge-transfer-population", "spectroscopic-signal"],
        times: ArrayLike,
        values: ArrayLike,
        standard_errors: ArrayLike,
        valid: ArrayLike | None,
        time_unit: UnitDefinition,
        observable_unit: UnitDefinition,
        source: ReferenceArtifactManifest,
    ):
        series = tuple(series_ids)
        units = tuple(independent_unit_ids)
        conditions = tuple(condition_ids)
        n = len(series)
        if (
            not n
            or len(set(series)) != n
            or len(units) != n
            or len(conditions) != n
            or any(
                not value or value != value.strip()
                for value in (*series, *units, *conditions)
            )
        ):
            raise ValueError(
                "Charge-transfer series need unique IDs and aligned independent units."
            )
        for value, name in (
            (sequence_id, "sequence_id"),
            (environment_id, "environment_id"),
        ):
            if not value or value != value.strip():
                raise ValueError(f"{name} must be a nonempty canonical string.")
        if observable_kind not in ("charge-transfer-population", "spectroscopic-signal"):
            raise ValueError(
                "Electronics qualification is limited to population transfer or spectroscopic signal."
            )
        coordinates = np.asarray(times, dtype=float)
        observations = np.asarray(values, dtype=float)
        errors = np.asarray(standard_errors, dtype=float)
        mask = (
            np.ones(observations.shape, dtype=bool)
            if valid is None
            else np.asarray(valid)
        )
        if (
            coordinates.ndim != 1
            or coordinates.size < 2
            or not np.all(np.isfinite(coordinates))
            or np.any(coordinates < 0.0)
            or np.any(np.diff(coordinates) <= 0.0)
        ):
            raise ValueError(
                "Time coordinates must be finite, nonnegative, increasing, and resolved."
            )
        if observations.ndim != 3 or observations.shape[:2] != (n, len(coordinates)):
            raise ValueError("Observations must have shape (series, time, channel).")
        if (
            errors.shape != observations.shape
            or mask.shape != observations.shape
            or mask.dtype != bool
            or not np.all(np.isfinite(observations[mask]))
            or not np.all(np.isfinite(errors[mask]))
            or np.any(errors[mask] <= 0.0)
        ):
            raise ValueError(
                "Active observations require finite values and positive standard errors."
            )
        if observable_kind == "charge-transfer-population":
            if observable_unit.unit_id != ONE.unit_id:
                raise ValueError(
                    "Charge-transfer populations require the canonical dimensionless unit."
                )
            if np.any((observations[mask] < 0.0) | (observations[mask] > 1.0)):
                raise ValueError(
                    "Observed charge-transfer populations must lie in [0, 1]."
                )
            complete_rows = np.all(mask, axis=-1)
            row_sums = np.sum(observations, axis=-1)
            if np.any(
                ~np.isclose(
                    row_sums[complete_rows],
                    1.0,
                    rtol=0.0,
                    atol=_PROBABILITY_NORMALIZATION_TOLERANCE,
                )
            ):
                raise ValueError(
                    "Fully observed charge-transfer population rows must sum to one."
                )
        if any(np.count_nonzero(np.any(mask[row], axis=-1)) < 2 for row in range(n)):
            raise ValueError(
                "Each independent series needs at least two observed time points."
            )
        conversion_factor(time_unit, SECOND)
        if not isinstance(source, ReferenceArtifactManifest):
            raise TypeError(
                "Charge-transfer observations require a ReferenceArtifactManifest."
            )
        source.require_rights()
        source.require_uncertainty()
        object.__setattr__(self, "series_ids", series)
        object.__setattr__(self, "independent_unit_ids", units)
        object.__setattr__(self, "condition_ids", conditions)
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "environment_id", environment_id)
        object.__setattr__(self, "observable_kind", observable_kind)
        object.__setattr__(self, "times", jnp.asarray(coordinates))
        object.__setattr__(self, "values", jnp.asarray(np.where(mask, observations, 0.0)))
        object.__setattr__(
            self, "standard_errors", jnp.asarray(np.where(mask, errors, 1.0))
        )
        object.__setattr__(self, "valid", jnp.asarray(mask))
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(self, "observable_unit", observable_unit)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "observation_id",
            canonical_fingerprint(
                {
                    "kind": "charge-transfer-observation-series",
                    "series": series,
                    "independent_units": units,
                    "conditions": conditions,
                    "sequence": sequence_id,
                    "environment": environment_id,
                    "observable_kind": observable_kind,
                    "times": coordinates.tolist(),
                    "values": array_tree_fingerprint(np.where(mask, observations, 0.0)),
                    "standard_errors": array_tree_fingerprint(
                        np.where(mask, errors, 1.0)
                    ),
                    "valid": array_tree_fingerprint(mask),
                    "units": (time_unit.unit_id, observable_unit.unit_id),
                    "source": source.manifest_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class ElectronicModelFit:
    """Content-bound fitted parameters, artifacts, execution, and campaign roles."""

    campaign: ScientificCampaign
    model_id: str
    calibration_case_ids: tuple[str, ...]
    model_selection_case_ids: tuple[str, ...]
    source_manifest_ids: tuple[str, ...]
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
        sources = tuple(
            sorted(
                {
                    source_id
                    for case_id in (*calibration, *selection)
                    for source_id in cases[case_id].source_manifest_ids
                }
            )
        )
        parameters = np.asarray(fitted_parameters, dtype=float)
        if not parameters.size or not np.all(np.isfinite(parameters)):
            raise ValueError("Fitted electronic parameters must be finite and nonempty.")
        parameter_id = canonical_fingerprint(
            {
                "kind": "electronic-fitted-parameters",
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
                "Fit execution evidence must pass and bind campaign, model, parameters, code, and sources."
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
                    "kind": "electronic-model-fit",
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
class ElectronicModelPrediction:
    """Frozen model output bound to one content-addressed campaign fit."""

    values: Array
    model_id: str
    fit_id: str
    campaign_id: str
    campaign_criteria_ids: tuple[str, ...]
    fit_integrity_ids: tuple[str, ...]
    observation_id: str
    case_ids: tuple[str, ...]
    observable_unit: UnitDefinition
    prediction_id: str

    def __init__(
        self,
        observations: ChargeTransferObservationSeries,
        values: ArrayLike,
        fit: ElectronicModelFit,
        /,
        *,
        observable_unit: UnitDefinition,
    ):
        if not isinstance(observations, ChargeTransferObservationSeries):
            raise TypeError("observations must be ChargeTransferObservationSeries.")
        if not isinstance(fit, ElectronicModelFit):
            raise TypeError("fit must be an ElectronicModelFit.")
        _validate_locked_campaign(fit.campaign, observations)
        converted = np.asarray(values, dtype=float)
        if converted.shape != observations.values.shape:
            raise ValueError("Electronic prediction must match the observation tensor.")
        factor = float(conversion_factor(observable_unit, observations.observable_unit))
        converted = converted * factor
        for name, value in (
            ("campaign_criteria_ids", fit.campaign.criteria_ids),
            ("values", jnp.asarray(converted)),
            ("model_id", fit.model_id),
            ("fit_id", fit.fit_id),
            ("fit_integrity_ids", fit.fit_integrity_ids),
            ("campaign_id", fit.campaign.campaign_id),
            ("observation_id", observations.observation_id),
            ("case_ids", observations.series_ids),
            ("observable_unit", observations.observable_unit),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "prediction_id",
            canonical_fingerprint(
                {
                    "kind": "locked-electronic-model-prediction",
                    "values": array_tree_fingerprint(converted),
                    "model": fit.model_id,
                    "fit": fit.fit_id,
                    "campaign": fit.campaign.campaign_id,
                    "fit_integrity_ids": fit.fit_integrity_ids,
                    "observation": observations.observation_id,
                    "cases": observations.series_ids,
                    "unit": observations.observable_unit.unit_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ElectronicModelComparison:
    quantum_macro_standardized_rms: float
    kinetic_macro_standardized_rms: float
    quantum_improvement: float
    quantum_standardized_residuals: Array
    kinetic_standardized_residuals: Array
    campaign_id: str
    campaign_criteria_ids: tuple[str, ...]
    quantum_model_id: str
    quantum_fit_id: str
    kinetic_model_id: str
    kinetic_fit_id: str
    qualification_evidence_ids: tuple[str, ...]
    capability_name: str
    observable_ids: tuple[str, ...]
    condition_domain_ids: tuple[str, ...]
    support_attributes: tuple[tuple[str, str], ...]
    missing_prerequisites: tuple[str, ...]
    failed_checks: tuple[str, ...]
    comparison_id: str

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
                "Scientific claim scope must exactly match the electronic assessment."
            )
        if claim.campaign_id != self.campaign_id:
            raise ValueError(
                "Scientific claim profile must bind the prediction campaign."
            )
        if tuple(claim.frozen_criteria_ids) != self.campaign_criteria_ids:
            raise ValueError(
                "Scientific claim criteria must exactly match the prediction campaign."
            )
        supplied_evidence_ids = {
            evidence.evidence_id
            for evidence in stage_evidence
            if isinstance(evidence, QualificationEvidence)
        }
        if not set(self.qualification_evidence_ids) <= supplied_evidence_ids:
            raise ValueError(
                "Claim evaluation must retain the prediction qualification evidence."
            )
        if not _MANDATORY_CLAIM_STAGES <= set(claim.required_stage_ids):
            raise ValueError(
                "Electronic claim profile omits mandatory scientific stages."
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
                criteria_ids=("nucleotide-electronics-domain-readiness", *issues),
                raw_artifact_ids=raw_artifact_ids,
                reviewer_id=reviewer_id,
                issued_at=issued_at,
                expires_at=expires_at,
                reason=";".join(issues),
                requalification_triggers=claim.invalidation_triggers,
            )
        metrics = {
            "quantum-macro-standardized-rms": self.quantum_macro_standardized_rms,
            "kinetic-macro-standardized-rms": self.kinetic_macro_standardized_rms,
            "quantum-improvement": self.quantum_improvement,
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


def _macro_standardized_rms(
    residuals: np.ndarray, mask: np.ndarray, independent_unit_ids: tuple[str, ...]
) -> float:
    scores = []
    identities = np.asarray(independent_unit_ids)
    for independent_unit in dict.fromkeys(independent_unit_ids):
        selected = identities == independent_unit
        active = residuals[selected][mask[selected]]
        scores.append(float(np.sqrt(np.mean(np.square(active)))))
    return float(np.mean(scores))


def _probability_support_reasons(
    values: np.ndarray, mask: np.ndarray, prefix: str, /
) -> tuple[str, ...]:
    del mask
    reasons = []
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        reasons.append(f"{prefix}-population-outside-probability-support")
    row_sums = np.sum(values, axis=-1)
    if np.any(
        ~np.isclose(
            row_sums,
            1.0,
            rtol=0.0,
            atol=_PROBABILITY_NORMALIZATION_TOLERANCE,
        )
    ):
        reasons.append(f"{prefix}-population-row-not-normalized")
    return tuple(reasons)


def _prediction_evidence_status(
    evidence: Sequence[QualificationEvidence],
    prediction: ElectronicModelPrediction,
    prefix: str,
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
        required_subjects = {
            prediction.campaign_id,
            prediction.model_id,
            prediction.fit_id,
            *prediction.fit_integrity_ids,
        }
        if include_prediction:
            required_subjects.add(prediction.prediction_id)
        matching = tuple(
            item
            for item in evidence
            if item.evidence_kind == "scientific"
            and stage_id in item.criteria_ids
            and required_subjects <= set(item.subject_ids)
        )
        accepted.extend(item.evidence_id for item in matching)
        if any(item.outcome == "failed" for item in matching):
            failed.append(f"{prefix}:{stage_id}")
        elif not matching or any(item.outcome == "inconclusive" for item in matching):
            missing.append(f"{prefix}:{stage_id}")
    return tuple(missing), tuple(failed), tuple(sorted(set(accepted)))


def compare_electronic_models(
    observations: ChargeTransferObservationSeries,
    quantum_prediction: ElectronicModelPrediction,
    kinetic_prediction: ElectronicModelPrediction,
    /,
    *,
    instrument_calibration: ReferenceArtifactManifest | None,
    environment_characterization: ReferenceArtifactManifest | None,
    maximum_quantum_standardized_rms: float,
    minimum_quantum_improvement: float,
    prediction_evidence: Sequence[QualificationEvidence],
) -> ElectronicModelComparison:
    """Compare frozen quantum and kinetic predictions on one locked campaign role."""

    if not isinstance(observations, ChargeTransferObservationSeries):
        raise TypeError("observations must be ChargeTransferObservationSeries.")
    if not isinstance(quantum_prediction, ElectronicModelPrediction) or not isinstance(
        kinetic_prediction, ElectronicModelPrediction
    ):
        raise TypeError("Both model outputs must be ElectronicModelPrediction records.")
    for prediction in (quantum_prediction, kinetic_prediction):
        if (
            prediction.observation_id != observations.observation_id
            or prediction.case_ids != observations.series_ids
        ):
            raise ValueError("Electronic prediction targets do not match observations.")
    if quantum_prediction.campaign_id != kinetic_prediction.campaign_id:
        raise ValueError("Electronic predictions must use one locked campaign.")
    if (
        quantum_prediction.model_id == kinetic_prediction.model_id
        or quantum_prediction.fit_id == kinetic_prediction.fit_id
    ):
        raise ValueError("Compared electronic models and fits must be distinct.")
    if (
        quantum_prediction.campaign_criteria_ids
        != kinetic_prediction.campaign_criteria_ids
    ):
        raise ValueError("Electronic predictions use different campaign criteria.")
    quantum = np.asarray(quantum_prediction.values)
    kinetic = np.asarray(kinetic_prediction.values)
    if (
        not math.isfinite(maximum_quantum_standardized_rms)
        or maximum_quantum_standardized_rms <= 0.0
    ):
        raise ValueError("maximum_quantum_standardized_rms must be finite and positive.")
    if (
        not math.isfinite(minimum_quantum_improvement)
        or minimum_quantum_improvement < 0.0
    ):
        raise ValueError("minimum_quantum_improvement must be finite and nonnegative.")
    mask = np.asarray(observations.valid)
    observed = np.asarray(observations.values)
    errors = np.asarray(observations.standard_errors)
    quantum_residual = np.where(mask, (quantum - observed) / errors, 0.0)
    kinetic_residual = np.where(mask, (kinetic - observed) / errors, 0.0)
    quantum_rms = _macro_standardized_rms(
        quantum_residual, mask, observations.independent_unit_ids
    )
    kinetic_rms = _macro_standardized_rms(
        kinetic_residual, mask, observations.independent_unit_ids
    )
    improvement = kinetic_rms - quantum_rms
    missing = []
    failed = []
    for name, reference in (
        ("instrument-calibration", instrument_calibration),
        ("environment-characterization", environment_characterization),
    ):
        if reference is None:
            missing.append(name)
        elif not isinstance(reference, ReferenceArtifactManifest):
            raise TypeError("Electronic prerequisites must be manifests or None.")
        elif reference.uncertainty is None:
            missing.append(f"{name}:unquantified-uncertainty")
        else:
            reference.require_rights()
    evidence_ids = []
    for prefix, prediction in (
        ("quantum", quantum_prediction),
        ("kinetic", kinetic_prediction),
    ):
        evidence_missing, evidence_failed, accepted = _prediction_evidence_status(
            prediction_evidence, prediction, prefix
        )
        missing.extend(evidence_missing)
        failed.extend(evidence_failed)
        evidence_ids.extend(accepted)
    if not np.all(np.isfinite(quantum[mask])):
        failed.append("nonfinite-quantum-prediction")
    if not np.all(np.isfinite(kinetic[mask])):
        failed.append("nonfinite-kinetic-prediction")
    if observations.observable_kind == "charge-transfer-population":
        failed.extend(_probability_support_reasons(quantum, mask, "quantum"))
        failed.extend(_probability_support_reasons(kinetic, mask, "kinetic"))
    if quantum_rms > maximum_quantum_standardized_rms:
        failed.append("quantum-macro-standardized-rms")
    if improvement < minimum_quantum_improvement:
        failed.append("quantum-does-not-beat-kinetic-baseline")
    missing_tuple = tuple(sorted(set(missing)))
    failed_tuple = tuple(sorted(set(failed)))
    capability_name = _CLAIM_CAPABILITY
    observable_ids = (observations.observation_id,)
    condition_domain_ids = tuple(sorted(set(observations.condition_ids)))
    support_attributes = tuple(
        sorted(
            {
                "environment_id": observations.environment_id,
                "kinetic_model_id": kinetic_prediction.model_id,
                "observable_kind": observations.observable_kind,
                "observable_unit_id": observations.observable_unit.unit_id,
                "observation_id": observations.observation_id,
                "quantum_model_id": quantum_prediction.model_id,
                "sequence_id": observations.sequence_id,
                "time_unit_id": observations.time_unit.unit_id,
            }.items()
        )
    )
    comparison_id = canonical_fingerprint(
        {
            "kind": "electronic-model-comparison",
            "observations": observations.observation_id,
            "campaign": quantum_prediction.campaign_id,
            "campaign_criteria_ids": quantum_prediction.campaign_criteria_ids,
            "capability": capability_name,
            "observable_ids": observable_ids,
            "condition_domain_ids": condition_domain_ids,
            "support_attributes": support_attributes,
            "quantum_prediction": quantum_prediction.prediction_id,
            "kinetic_prediction": kinetic_prediction.prediction_id,
            "quantum_rms": quantum_rms,
            "kinetic_rms": kinetic_rms,
            "improvement": improvement,
            "missing": missing_tuple,
            "failed": failed_tuple,
            "qualification_evidence": tuple(sorted(set(evidence_ids))),
            "thresholds": (
                maximum_quantum_standardized_rms,
                minimum_quantum_improvement,
            ),
        }
    )
    return ElectronicModelComparison(
        quantum_rms,
        kinetic_rms,
        improvement,
        jnp.asarray(quantum_residual),
        jnp.asarray(kinetic_residual),
        quantum_prediction.campaign_id,
        quantum_prediction.campaign_criteria_ids,
        quantum_prediction.model_id,
        quantum_prediction.fit_id,
        kinetic_prediction.model_id,
        kinetic_prediction.fit_id,
        tuple(sorted(set(evidence_ids))),
        capability_name,
        observable_ids,
        condition_domain_ids,
        support_attributes,
        missing_tuple,
        failed_tuple,
        comparison_id,
    )


__all__ = [
    "ChargeTransferObservationSeries",
    "ElectronicModelComparison",
    "ElectronicModelFit",
    "ElectronicModelPrediction",
    "compare_electronic_models",
]
