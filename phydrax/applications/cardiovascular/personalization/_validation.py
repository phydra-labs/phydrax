#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only governance records for investigational cardiovascular research."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ...._fingerprint import canonical_fingerprint, canonical_mapping


class ClinicalResearchUse(Enum):
    """The only use represented by this module."""

    INVESTIGATIONAL_RESEARCH = "investigational_research"


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if (
        not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be non-empty canonical text.")
    return value


def _unique_text(
    values: Sequence[str], name: str, /, *, required: bool = True
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings.")
    resolved = tuple(_text(value, name) for value in values)
    if required and not resolved:
        raise ValueError(f"{name} must be non-empty.")
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"{name} entries must be unique.")
    return resolved


def _immutable_mapping(value: Mapping[str, Any], name: str, /) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    normalized = canonical_mapping(value)
    _finite_numbers(normalized, name)
    return _freeze_json(normalized)


def _freeze_json(value: Any, /) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any, /) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _finite_numbers(value: Any, path: str, /) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings.")
            _finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_numbers(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number.")


@dataclass(frozen=True, slots=True, init=False)
class ClinicalResearchContext:
    """Research question, oversight, and data-governance boundary.

    Clinical decision support, regulated claims, and protected health information
    are deliberately unrepresentable as accepted contexts.
    """

    research_question: str
    study_context: str
    investigational_use: str
    protocol_id: str
    irb_id: str | None
    waiver_id: str | None
    deidentification_ids: tuple[str, ...]
    data_rights_ids: tuple[str, ...]
    intended_use: ClinicalResearchUse = field(init=False)
    context_id: str = field(init=False)

    def __init__(
        self,
        research_question: str,
        study_context: str,
        investigational_use: str,
        protocol_id: str,
        /,
        *,
        irb_id: str | None = None,
        waiver_id: str | None = None,
        deidentification_ids: Sequence[str],
        data_rights_ids: Sequence[str],
        contains_phi: bool = False,
        clinical_decision_use: bool = False,
        regulated_claim: bool = False,
    ):
        if bool(contains_phi):
            raise ValueError(
                "ClinicalResearchContext refuses protected health information."
            )
        if bool(clinical_decision_use):
            raise ValueError("ClinicalResearchContext refuses clinical decision use.")
        if bool(regulated_claim):
            raise ValueError("ClinicalResearchContext refuses regulated claims.")
        question = _text(research_question, "research_question")
        study = _text(study_context, "study_context")
        use = _text(investigational_use, "investigational_use")
        protocol = _text(protocol_id, "protocol_id")
        irb = None if irb_id is None else _text(irb_id, "irb_id")
        waiver = None if waiver_id is None else _text(waiver_id, "waiver_id")
        if (irb is None) == (waiver is None):
            raise ValueError("Provide exactly one of irb_id or waiver_id.")
        deidentification = _unique_text(deidentification_ids, "deidentification_ids")
        rights = _unique_text(data_rights_ids, "data_rights_ids")
        identity = canonical_fingerprint(
            {
                "kind": "cardiovascular-clinical-research-context",
                "research_question": question,
                "study_context": study,
                "investigational_use": use,
                "intended_use": ClinicalResearchUse.INVESTIGATIONAL_RESEARCH.value,
                "protocol_id": protocol,
                "irb_id": irb,
                "waiver_id": waiver,
                "deidentification_ids": list(deidentification),
                "data_rights_ids": list(rights),
            }
        )
        object.__setattr__(self, "research_question", question)
        object.__setattr__(self, "study_context", study)
        object.__setattr__(self, "investigational_use", use)
        object.__setattr__(self, "protocol_id", protocol)
        object.__setattr__(self, "irb_id", irb)
        object.__setattr__(self, "waiver_id", waiver)
        object.__setattr__(self, "deidentification_ids", deidentification)
        object.__setattr__(self, "data_rights_ids", rights)
        object.__setattr__(
            self, "intended_use", ClinicalResearchUse.INVESTIGATIONAL_RESEARCH
        )
        object.__setattr__(self, "context_id", identity)


@dataclass(frozen=True, slots=True, init=False)
class ClinicalResearchValidationPlan:
    """Prospective research validation design, without a certification outcome."""

    context: ClinicalResearchContext
    training_cohort_id: str
    calibration_cohort_id: str
    validation_cohort_id: str
    site_holdout_ids: tuple[str, ...]
    temporal_holdout_id: str
    endpoint_definition: str
    comparator_definition: str
    subgroup_definitions: tuple[str, ...]
    ood_definition: str
    failure_analysis_plan: str
    acceptance_criteria: tuple[str, ...]
    plan_id: str = field(init=False)

    def __init__(
        self,
        context: ClinicalResearchContext,
        training_cohort_id: str,
        calibration_cohort_id: str,
        validation_cohort_id: str,
        /,
        *,
        site_holdout_ids: Sequence[str],
        temporal_holdout_id: str,
        endpoint_definition: str,
        comparator_definition: str,
        subgroup_definitions: Sequence[str],
        ood_definition: str,
        failure_analysis_plan: str,
        acceptance_criteria: Sequence[str],
    ):
        if not isinstance(context, ClinicalResearchContext):
            raise TypeError("context must be a ClinicalResearchContext.")
        training = _text(training_cohort_id, "training_cohort_id")
        calibration = _text(calibration_cohort_id, "calibration_cohort_id")
        validation = _text(validation_cohort_id, "validation_cohort_id")
        if len({training, calibration, validation}) != 3:
            raise ValueError(
                "Training, calibration, and validation cohorts must be separate."
            )
        sites = _unique_text(site_holdout_ids, "site_holdout_ids")
        temporal = _text(temporal_holdout_id, "temporal_holdout_id")
        if temporal in {training, calibration, validation} or temporal in sites:
            raise ValueError("Temporal holdout must be distinct from all other cohorts.")
        if set(sites) & {training, calibration, validation}:
            raise ValueError("Site holdouts must be distinct from development cohorts.")
        endpoint = _text(endpoint_definition, "endpoint_definition")
        comparator = _text(comparator_definition, "comparator_definition")
        subgroups = _unique_text(subgroup_definitions, "subgroup_definitions")
        ood = _text(ood_definition, "ood_definition")
        failures = _text(failure_analysis_plan, "failure_analysis_plan")
        criteria = _unique_text(acceptance_criteria, "acceptance_criteria")
        identity = canonical_fingerprint(
            {
                "kind": "cardiovascular-clinical-research-validation-plan",
                "context": context.context_id,
                "training_cohort_id": training,
                "calibration_cohort_id": calibration,
                "validation_cohort_id": validation,
                "site_holdout_ids": list(sites),
                "temporal_holdout_id": temporal,
                "endpoint_definition": endpoint,
                "comparator_definition": comparator,
                "subgroup_definitions": list(subgroups),
                "ood_definition": ood,
                "failure_analysis_plan": failures,
                "acceptance_criteria": list(criteria),
            }
        )
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "training_cohort_id", training)
        object.__setattr__(self, "calibration_cohort_id", calibration)
        object.__setattr__(self, "validation_cohort_id", validation)
        object.__setattr__(self, "site_holdout_ids", sites)
        object.__setattr__(self, "temporal_holdout_id", temporal)
        object.__setattr__(self, "endpoint_definition", endpoint)
        object.__setattr__(self, "comparator_definition", comparator)
        object.__setattr__(self, "subgroup_definitions", subgroups)
        object.__setattr__(self, "ood_definition", ood)
        object.__setattr__(self, "failure_analysis_plan", failures)
        object.__setattr__(self, "acceptance_criteria", criteria)
        object.__setattr__(self, "plan_id", identity)


@dataclass(frozen=True, slots=True, init=False)
class ClinicalResearchValidationRecord:
    """Immutable execution record whose completeness is evaluated fail closed."""

    record_id: str
    plan: ClinicalResearchValidationPlan
    execution_id: str
    calibration_results: Mapping[str, Any]
    validation_results: Mapping[str, Any]
    site_holdout_results: Mapping[str, Any]
    temporal_holdout_results: Mapping[str, Any]
    subgroup_results: Mapping[str, Any]
    ood_results: Mapping[str, Any]
    failure_analysis_results: Mapping[str, Any]
    record_content_id: str = field(init=False)

    def __init__(
        self,
        record_id: str,
        plan: ClinicalResearchValidationPlan,
        execution_id: str,
        /,
        *,
        calibration_results: Mapping[str, Any],
        validation_results: Mapping[str, Any],
        site_holdout_results: Mapping[str, Any],
        temporal_holdout_results: Mapping[str, Any],
        subgroup_results: Mapping[str, Any],
        ood_results: Mapping[str, Any],
        failure_analysis_results: Mapping[str, Any],
        contains_phi: bool = False,
        clinical_decision_claim: bool = False,
        regulated_claim: bool = False,
    ):
        if not isinstance(plan, ClinicalResearchValidationPlan):
            raise TypeError("plan must be a ClinicalResearchValidationPlan.")
        if bool(contains_phi):
            raise ValueError(
                "ClinicalResearchValidationRecord refuses protected health information."
            )
        if bool(clinical_decision_claim):
            raise ValueError("Validation records cannot make clinical decision claims.")
        if bool(regulated_claim):
            raise ValueError("Validation records cannot make regulated claims.")
        identifier = _text(record_id, "record_id")
        execution = _text(execution_id, "execution_id")
        calibration = _immutable_mapping(calibration_results, "calibration_results")
        validation = _immutable_mapping(validation_results, "validation_results")
        site = _immutable_mapping(site_holdout_results, "site_holdout_results")
        temporal = _immutable_mapping(
            temporal_holdout_results, "temporal_holdout_results"
        )
        subgroup = _immutable_mapping(subgroup_results, "subgroup_results")
        ood = _immutable_mapping(ood_results, "ood_results")
        failures = _immutable_mapping(
            failure_analysis_results, "failure_analysis_results"
        )
        identity = canonical_fingerprint(
            {
                "kind": "cardiovascular-clinical-research-validation-record",
                "record_id": identifier,
                "plan": plan.plan_id,
                "execution_id": execution,
                "calibration_results": _thaw_json(calibration),
                "validation_results": _thaw_json(validation),
                "site_holdout_results": _thaw_json(site),
                "temporal_holdout_results": _thaw_json(temporal),
                "subgroup_results": _thaw_json(subgroup),
                "ood_results": _thaw_json(ood),
                "failure_analysis_results": _thaw_json(failures),
            }
        )
        object.__setattr__(self, "record_id", identifier)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "execution_id", execution)
        object.__setattr__(self, "calibration_results", calibration)
        object.__setattr__(self, "validation_results", validation)
        object.__setattr__(self, "site_holdout_results", site)
        object.__setattr__(self, "temporal_holdout_results", temporal)
        object.__setattr__(self, "subgroup_results", subgroup)
        object.__setattr__(self, "ood_results", ood)
        object.__setattr__(self, "failure_analysis_results", failures)
        object.__setattr__(self, "record_content_id", identity)

    def evaluate(self, /) -> "ClinicalResearchValidationEvidence":
        return ClinicalResearchValidationEvidence.from_record(self)


@dataclass(frozen=True, slots=True)
class ClinicalResearchValidationEvidence:
    """Coverage evidence only; it is not a clinical or regulatory certificate."""

    plan_id: str
    record_content_id: str
    question_and_context_bound: bool
    investigational_use_only: bool
    oversight_bound: bool
    deidentification_bound: bool
    data_rights_bound: bool
    calibration_validation_separate: bool
    site_holdout_complete: bool
    temporal_holdout_complete: bool
    endpoint_and_comparator_bound: bool
    subgroup_analysis_complete: bool
    ood_analysis_complete: bool
    failure_analysis_complete: bool
    record_complete: bool

    @classmethod
    def from_record(
        cls, record: ClinicalResearchValidationRecord, /
    ) -> "ClinicalResearchValidationEvidence":
        if not isinstance(record, ClinicalResearchValidationRecord):
            raise TypeError("record must be a ClinicalResearchValidationRecord.")
        plan = record.plan
        context = plan.context
        question = bool(context.research_question and context.study_context)
        investigational = (
            context.intended_use is ClinicalResearchUse.INVESTIGATIONAL_RESEARCH
        )
        oversight = bool(context.protocol_id and (context.irb_id or context.waiver_id))
        deidentified = bool(context.deidentification_ids)
        rights = bool(context.data_rights_ids)
        separated = (
            len(
                {
                    plan.training_cohort_id,
                    plan.calibration_cohort_id,
                    plan.validation_cohort_id,
                }
            )
            == 3
            and bool(record.calibration_results)
            and bool(record.validation_results)
        )
        site = set(record.site_holdout_results) == set(plan.site_holdout_ids) and all(
            bool(record.site_holdout_results[site_id])
            for site_id in plan.site_holdout_ids
        )
        temporal = set(record.temporal_holdout_results) == {
            plan.temporal_holdout_id
        } and bool(record.temporal_holdout_results[plan.temporal_holdout_id])
        endpoint = bool(plan.endpoint_definition and plan.comparator_definition)
        subgroup = set(record.subgroup_results) == set(plan.subgroup_definitions) and all(
            bool(record.subgroup_results[name]) for name in plan.subgroup_definitions
        )
        ood = bool(record.ood_results)
        failures = bool(record.failure_analysis_results)
        complete = all(
            (
                question,
                investigational,
                oversight,
                deidentified,
                rights,
                separated,
                site,
                temporal,
                endpoint,
                subgroup,
                ood,
                failures,
            )
        )
        return cls(
            plan_id=plan.plan_id,
            record_content_id=record.record_content_id,
            question_and_context_bound=question,
            investigational_use_only=investigational,
            oversight_bound=oversight,
            deidentification_bound=deidentified,
            data_rights_bound=rights,
            calibration_validation_separate=separated,
            site_holdout_complete=site,
            temporal_holdout_complete=temporal,
            endpoint_and_comparator_bound=endpoint,
            subgroup_analysis_complete=subgroup,
            ood_analysis_complete=ood,
            failure_analysis_complete=failures,
            record_complete=complete,
        )


__all__ = [
    "ClinicalResearchContext",
    "ClinicalResearchUse",
    "ClinicalResearchValidationEvidence",
    "ClinicalResearchValidationPlan",
    "ClinicalResearchValidationRecord",
]
