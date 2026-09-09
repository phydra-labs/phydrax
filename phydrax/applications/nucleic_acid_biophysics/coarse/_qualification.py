# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Restricted nucleotide-mechanics calibration and held-out response evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

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
from ....units import UnitDefinition
from ._mechanics import PreparedNucleotideModel


_MANDATORY_CLAIM_STAGES = frozenset(
    (
        "source-admission",
        "measurement-calibration",
        "parameter-identifiability",
        "predictive-calibration",
        "locked-prediction",
    )
)


@dataclass(frozen=True, slots=True, init=False)
class NucleotideMechanicalResponseData:
    """Caller-computed local response surface and measured force/twist rows."""

    case_ids: tuple[str, ...]
    independent_unit_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    source_row_ids: tuple[str, ...]
    parent_case_ids: tuple[tuple[str, ...], ...]
    baseline_response: Array
    parameter_sensitivities: Array
    observed_response: Array
    standard_errors: Array
    force_unit: UnitDefinition
    twist_unit: UnitDefinition
    source: ReferenceArtifactManifest
    model_id: str
    construct_id: str
    parameter_manifest_id: str
    response_id: str

    def __init__(
        self,
        case_ids: tuple[str, ...],
        independent_unit_ids: tuple[str, ...],
        condition_ids: tuple[str, ...],
        /,
        *,
        source_row_ids: tuple[str, ...],
        parent_case_ids: tuple[tuple[str, ...], ...],
        baseline_response: ArrayLike,
        parameter_sensitivities: ArrayLike,
        observed_response: ArrayLike,
        standard_errors: ArrayLike,
        force_unit: UnitDefinition,
        twist_unit: UnitDefinition,
        source: ReferenceArtifactManifest,
        model: PreparedNucleotideModel,
    ):
        cases = tuple(case_ids)
        units = tuple(independent_unit_ids)
        conditions = tuple(condition_ids)
        source_rows = tuple(source_row_ids)
        parents = tuple(tuple(values) for values in parent_case_ids)
        n = len(cases)
        if (
            not n
            or len(set(cases)) != n
            or len(source_rows) != n
            or len(set(source_rows)) != n
            or len(parents) != n
            or len(units) != n
            or len(conditions) != n
            or any(
                not value or value != value.strip()
                for value in (
                    *cases,
                    *units,
                    *conditions,
                    *source_rows,
                    *(parent for values in parents for parent in values),
                )
            )
        ):
            raise ValueError(
                "Mechanical rows require unique cases and aligned unit/condition IDs."
            )
        baseline = np.asarray(baseline_response, dtype=float)
        sensitivities = np.asarray(parameter_sensitivities, dtype=float)
        observed = np.asarray(observed_response, dtype=float)
        errors = np.asarray(standard_errors, dtype=float)
        if (
            baseline.shape != (n, 2)
            or observed.shape != baseline.shape
            or errors.shape != baseline.shape
            or sensitivities.ndim != 3
            or sensitivities.shape[:2] != baseline.shape
            or sensitivities.shape[2] < 2
            or not np.all(np.isfinite(baseline))
            or not np.all(np.isfinite(sensitivities))
            or not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(errors))
            or np.any(errors <= 0.0)
        ):
            raise ValueError(
                "Mechanical response needs finite (case, force/twist) values, positive errors, and sensitivities."
            )
        if not isinstance(source, ReferenceArtifactManifest):
            raise TypeError(
                "Mechanical observations require a ReferenceArtifactManifest."
            )
        if not isinstance(model, PreparedNucleotideModel):
            raise TypeError("model must be a PreparedNucleotideModel.")
        source.require_rights()
        source.require_uncertainty()
        for name, value in (
            ("case_ids", cases),
            ("independent_unit_ids", units),
            ("condition_ids", conditions),
            ("source_row_ids", source_rows),
            ("parent_case_ids", parents),
            ("baseline_response", jnp.asarray(baseline)),
            ("parameter_sensitivities", jnp.asarray(sensitivities)),
            ("observed_response", jnp.asarray(observed)),
            ("standard_errors", jnp.asarray(errors)),
            ("force_unit", force_unit),
            ("twist_unit", twist_unit),
            ("source", source),
            ("model_id", model.prepared_id),
            ("construct_id", model.construct_id),
            ("parameter_manifest_id", model.parameter_manifest_id),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "response_id",
            canonical_fingerprint(
                {
                    "kind": "nucleotide-force-twist-response",
                    "cases": cases,
                    "independent_units": units,
                    "conditions": conditions,
                    "source_rows": source_rows,
                    "parent_cases": parents,
                    "baseline": array_tree_fingerprint(baseline),
                    "sensitivities": array_tree_fingerprint(sensitivities),
                    "observed": array_tree_fingerprint(observed),
                    "standard_errors": array_tree_fingerprint(errors),
                    "units": (force_unit.unit_id, twist_unit.unit_id),
                    "source": source.manifest_id,
                    "model": model.prepared_id,
                    "construct": model.construct_id,
                    "parameters": model.parameter_manifest_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class NucleotideMechanicsAssessment:
    fitted_parameter_names: tuple[str, ...]
    fitted_parameter_offsets: Array
    parameter_covariance: Array | None
    predicted_locked_response: Array
    locked_prediction_standard_errors: Array | None
    force_macro_standardized_rms: float
    twist_macro_standardized_rms: float
    singular_values: Array
    sensitivity_rank: int
    condition_number: float
    relative_rank_tolerance: float
    maximum_condition_number: float
    campaign_id: str
    campaign_criteria_ids: tuple[str, ...]
    model_id: str
    fit_id: str
    prediction_id: str
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
                "Scientific claim scope must exactly match the mechanics assessment."
            )
        if claim.campaign_id != self.campaign_id:
            raise ValueError("Scientific claim profile must bind the mechanics campaign.")
        if tuple(claim.frozen_criteria_ids) != self.campaign_criteria_ids:
            raise ValueError(
                "Scientific claim criteria must exactly match the mechanics campaign."
            )
        if not _MANDATORY_CLAIM_STAGES <= set(claim.required_stage_ids):
            raise ValueError(
                "Nucleotide mechanics claim profile omits mandatory scientific stages."
            )
        for stage_id, include_prediction in (
            ("parameter-identifiability", False),
            ("predictive-calibration", False),
            ("locked-prediction", True),
        ):
            subjects = {claim.campaign_id, self.model_id, self.fit_id}
            if include_prediction:
                subjects.add(self.prediction_id)
            if not any(
                isinstance(evidence, QualificationEvidence)
                and evidence.outcome == "passed"
                and stage_id in evidence.criteria_ids
                and subjects <= set(evidence.subject_ids)
                for evidence in stage_evidence
            ):
                raise ValueError(
                    f"{stage_id} evidence is not bound to this mechanics fit."
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
                criteria_ids=("nucleotide-mechanics-domain-readiness", *issues),
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
            "force-macro-standardized-rms": self.force_macro_standardized_rms,
            "twist-macro-standardized-rms": self.twist_macro_standardized_rms,
        }
        return claim.evaluate(
            metrics,
            stage_evidence,
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


def _macro_rms(values: np.ndarray, unit_ids: tuple[str, ...], column: int) -> float:
    scores = []
    units = np.asarray(unit_ids)
    for independent_unit in dict.fromkeys(unit_ids):
        selected = units == independent_unit
        scores.append(float(np.sqrt(np.mean(np.square(values[selected, column])))))
    return float(np.mean(scores))


def fit_restricted_nucleotide_mechanics(
    calibration: NucleotideMechanicalResponseData,
    locked: NucleotideMechanicalResponseData,
    parameter_names: tuple[str, ...],
    fitted_parameter_indices: tuple[int, ...],
    /,
    *,
    thermal_reference: ReferenceArtifactManifest | None,
    structural_reference: ReferenceArtifactManifest | None,
    maximum_standardized_rms: float,
    campaign: ScientificCampaign,
    relative_rank_tolerance: float,
    maximum_condition_number: float,
) -> NucleotideMechanicsAssessment:
    """Fit conditioned local offsets and propagate their covariance to locked rows."""

    if not isinstance(calibration, NucleotideMechanicalResponseData) or not isinstance(
        locked, NucleotideMechanicalResponseData
    ):
        raise TypeError("calibration and locked must be mechanical response data.")
    names = tuple(parameter_names)
    count = calibration.parameter_sensitivities.shape[2]
    selected = tuple(fitted_parameter_indices)
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    role_cases = {role.name: role.case_ids for role in campaign.roles}
    if set(calibration.case_ids) != set(role_cases["calibration"]):
        raise ValueError("Calibration rows must exactly equal the campaign role.")
    if set(locked.case_ids) != set(role_cases["locked_evaluation"]):
        raise ValueError("Locked rows must exactly equal the campaign role.")
    campaign_cases = {case.case_id: case for case in campaign.cases}
    for response in (calibration, locked):
        for case_id, unit_id, condition_id in zip(
            response.case_ids,
            response.independent_unit_ids,
            response.condition_ids,
            strict=True,
        ):
            case = campaign_cases[case_id]
            if (
                case.independent_unit_id != unit_id
                or case.condition_id != condition_id
                or case.construct_id != response.construct_id
                or response.source.manifest_id not in case.source_manifest_ids
            ):
                raise ValueError(
                    "Mechanical response identity does not match the campaign."
                )
    if (
        len(names) != count
        or len(set(names)) != count
        or any(not name or name != name.strip() for name in names)
    ):
        raise ValueError(
            "Parameter names must exactly identify every sensitivity column."
        )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < count
            for index in selected
        )
        or len(selected) >= count
    ):
        raise ValueError(
            "Fitted parameters must be an explicit proper subset of columns."
        )
    if locked.parameter_sensitivities.shape[2] != count:
        raise ValueError("Calibration and locked sensitivity layouts differ.")
    if (
        calibration.model_id != locked.model_id
        or calibration.construct_id != locked.construct_id
        or calibration.parameter_manifest_id != locked.parameter_manifest_id
        or calibration.force_unit.unit_id != locked.force_unit.unit_id
        or calibration.twist_unit.unit_id != locked.twist_unit.unit_id
    ):
        raise ValueError(
            "Calibration and locked response surfaces use different model support."
        )
    if not math.isfinite(maximum_standardized_rms) or maximum_standardized_rms <= 0.0:
        raise ValueError("maximum_standardized_rms must be finite and positive.")
    if (
        not math.isfinite(relative_rank_tolerance)
        or not 0.0 < relative_rank_tolerance < 1.0
    ):
        raise ValueError("relative_rank_tolerance must be finite and lie in (0, 1).")
    if not math.isfinite(maximum_condition_number) or maximum_condition_number < 1.0:
        raise ValueError("maximum_condition_number must be finite and at least one.")
    missing = [
        f"calibration-source-rights:{reason}"
        for reason in calibration.source.rights_refusal_reasons(training_use=True)
    ]
    for label, reference in (
        ("thermal-evidence", thermal_reference),
        ("structural-evidence", structural_reference),
    ):
        if reference is None:
            missing.append(label)
        elif not isinstance(reference, ReferenceArtifactManifest):
            raise TypeError("Thermal/structural evidence must be manifests or None.")
        elif reference.uncertainty is None:
            missing.append(f"{label}:unquantified-uncertainty")
        else:
            missing.extend(
                f"{label}:rights:{reason}"
                for reason in reference.rights_refusal_reasons(training_use=True)
            )
    calibration_sensitivity = np.asarray(calibration.parameter_sensitivities)[
        ..., selected
    ]
    calibration_error = np.asarray(calibration.standard_errors)
    design = (calibration_sensitivity / calibration_error[..., None]).reshape(
        -1, len(selected)
    )
    target = (
        (
            np.asarray(calibration.observed_response)
            - np.asarray(calibration.baseline_response)
        )
        / calibration_error
    ).reshape(-1)
    singular_values = np.linalg.svd(design, compute_uv=False)
    singular_floor = relative_rank_tolerance * singular_values[0]
    rank = int(np.count_nonzero(singular_values > singular_floor))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else math.inf
    )
    identified = (
        rank == len(selected)
        and math.isfinite(condition_number)
        and condition_number <= maximum_condition_number
    )
    offsets, _, _, _ = np.linalg.lstsq(design, target, rcond=relative_rank_tolerance)
    locked_sensitivity = np.asarray(locked.parameter_sensitivities)[..., selected]
    locked_prediction = np.asarray(locked.baseline_response) + np.einsum(
        "rop,p->ro", locked_sensitivity, offsets
    )
    parameter_covariance = None
    prediction_standard_errors = None
    total_locked_errors = np.asarray(locked.standard_errors)
    if identified:
        information = design.T @ design
        covariance = np.linalg.solve(
            information, np.eye(len(selected), dtype=information.dtype)
        )
        if np.all(np.isfinite(covariance)):
            prediction_variance = np.einsum(
                "rop,pq,roq->ro",
                locked_sensitivity,
                covariance,
                locked_sensitivity,
            )
            prediction_variance = np.maximum(prediction_variance, 0.0)
            prediction_standard_errors = np.sqrt(prediction_variance)
            total_locked_errors = np.sqrt(total_locked_errors**2 + prediction_variance)
            parameter_covariance = covariance
        else:
            missing.append("parameter-covariance")
    else:
        missing.append("parameter-identifiability")
    residuals = (
        locked_prediction - np.asarray(locked.observed_response)
    ) / total_locked_errors
    force_rms = _macro_rms(residuals, locked.independent_unit_ids, 0)
    twist_rms = _macro_rms(residuals, locked.independent_unit_ids, 1)
    failed = []
    if set(calibration.case_ids) & set(locked.case_ids):
        failed.append("case-id-leakage")
    calibration_lineage = {
        *calibration.source_row_ids,
        *(parent for values in calibration.parent_case_ids for parent in values),
    }
    locked_lineage = {
        *locked.source_row_ids,
        *(parent for values in locked.parent_case_ids for parent in values),
    }
    if calibration_lineage & locked_lineage:
        failed.append("source-row-ancestry-leakage")
    if set(calibration.independent_unit_ids) & set(locked.independent_unit_ids):
        failed.append("independent-unit-leakage")
    if not np.all(np.isfinite(locked_prediction)):
        failed.append("nonfinite-locked-prediction")
    if force_rms > maximum_standardized_rms:
        failed.append("force-macro-standardized-rms")
    if twist_rms > maximum_standardized_rms:
        failed.append("twist-macro-standardized-rms")
    missing_tuple = tuple(sorted(set(missing)))
    failed_tuple = tuple(sorted(set(failed)))
    fit_id = canonical_fingerprint(
        {
            "kind": "restricted-nucleotide-mechanics-fit",
            "model": calibration.model_id,
            "campaign": campaign.campaign_id,
            "calibration": calibration.response_id,
            "parameter_names": names,
            "fitted_indices": selected,
            "offsets": array_tree_fingerprint(offsets),
            "parameter_covariance": (
                None
                if parameter_covariance is None
                else array_tree_fingerprint(parameter_covariance)
            ),
            "relative_rank_tolerance": relative_rank_tolerance,
            "maximum_condition_number": maximum_condition_number,
        }
    )
    prediction_id = canonical_fingerprint(
        {
            "kind": "locked-nucleotide-mechanics-prediction",
            "fit": fit_id,
            "locked": locked.response_id,
            "values": array_tree_fingerprint(locked_prediction),
            "standard_errors": (
                None
                if prediction_standard_errors is None
                else array_tree_fingerprint(prediction_standard_errors)
            ),
        }
    )
    capability_name = "nucleotide-mechanics"
    observable_ids = (locked.response_id,)
    condition_domain_ids = tuple(sorted(set(locked.condition_ids)))
    support_attributes = tuple(
        sorted(
            {
                "construct_id": locked.construct_id,
                "fit_id": fit_id,
                "force_unit_id": locked.force_unit.unit_id,
                "model_id": locked.model_id,
                "parameter_manifest_id": locked.parameter_manifest_id,
                "prediction_id": prediction_id,
                "response_id": locked.response_id,
                "twist_unit_id": locked.twist_unit.unit_id,
            }.items()
        )
    )
    assessment_id = canonical_fingerprint(
        {
            "kind": "restricted-nucleotide-mechanics-assessment",
            "campaign": campaign.campaign_id,
            "campaign_criteria_ids": campaign.criteria_ids,
            "calibration": calibration.response_id,
            "locked": locked.response_id,
            "model": calibration.model_id,
            "fit": fit_id,
            "prediction": prediction_id,
            "capability": capability_name,
            "observable_ids": observable_ids,
            "condition_domain_ids": condition_domain_ids,
            "support_attributes": support_attributes,
            "parameter_names": names,
            "fitted_indices": selected,
            "offsets": array_tree_fingerprint(offsets),
            "parameter_covariance": (
                None
                if parameter_covariance is None
                else array_tree_fingerprint(parameter_covariance)
            ),
            "locked_prediction_standard_errors": (
                None
                if prediction_standard_errors is None
                else array_tree_fingerprint(prediction_standard_errors)
            ),
            "singular_values": array_tree_fingerprint(singular_values),
            "rank": rank,
            "condition_number": condition_number,
            "relative_rank_tolerance": relative_rank_tolerance,
            "maximum_condition_number": maximum_condition_number,
            "force_rms": force_rms,
            "twist_rms": twist_rms,
            "missing": missing_tuple,
            "failed": failed_tuple,
            "threshold": maximum_standardized_rms,
        }
    )
    return NucleotideMechanicsAssessment(
        tuple(names[index] for index in selected),
        jnp.asarray(offsets),
        None if parameter_covariance is None else jnp.asarray(parameter_covariance),
        jnp.asarray(locked_prediction),
        (
            None
            if prediction_standard_errors is None
            else jnp.asarray(prediction_standard_errors)
        ),
        force_rms,
        twist_rms,
        jnp.asarray(singular_values),
        rank,
        condition_number,
        relative_rank_tolerance,
        maximum_condition_number,
        campaign.campaign_id,
        campaign.criteria_ids,
        calibration.model_id,
        fit_id,
        prediction_id,
        capability_name,
        observable_ids,
        condition_domain_ids,
        support_attributes,
        missing_tuple,
        failed_tuple,
        assessment_id,
    )


__all__ = [
    "NucleotideMechanicalResponseData",
    "NucleotideMechanicsAssessment",
    "fit_restricted_nucleotide_mechanics",
]
