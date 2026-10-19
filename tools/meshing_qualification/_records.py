# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, final

import equinox as eqx

import phydrax as phx
from phydrax import canonical_fingerprint
from phydrax._meshcore import MeshcoreError
from phydrax.meshing import BisectionCompatibility, NativeMeshingPhaseMeasurement
from tools._meshing_cases import (
    CORPUS,
    corpus_identity,
    load_cad_fixture,
    load_corpus_registration,
    load_curved_periodic_source_registration,
    meshing_case,
    POLYHEDRAL_CASE_NAMES,
    polyhedral_identity,
)
from tools.meshing_qualification._controls import (
    _cad_qualification_requirements,
    _corpus_contract_choices,
    _family_qualification_requirements,
    _hybrid_qualification_requirements,
    _periodic_qualification_requirements,
    _surface_qualification_requirements,
    CORPUS_SCENARIO_WORKFLOWS,
    QualificationControls,
    SCENARIO_WORKFLOWS,
)
from tools.meshing_qualification._runtime import qualification_failure


@final
class FailureRecord(phx.StrictModule):
    """Host failure identity and native evidence, serialized only at the JSON edge."""

    failure_phase: str = eqx.field(static=True)
    failure_type: str = eqx.field(static=True)
    failure: str = eqx.field(static=True)
    evidence: phx.meshing.MeshingFailureEvidence | None = None
    native_status: str | None = eqx.field(static=True, default=None)
    elapsed_seconds: float | None = eqx.field(static=True, default=None)
    maximum_seconds: float | None = eqx.field(static=True, default=None)

    @classmethod
    @phx.typing.checked
    def from_exception(cls, failure: Exception, phase: str) -> FailureRecord:
        match failure:
            case phx.meshing.MeshingFailure():
                return cls(phase, type(failure).__name__, str(failure), failure.evidence)
            case MeshcoreError():
                return cls(
                    phase,
                    type(failure).__name__,
                    str(failure),
                    native_status=failure.status.name,
                    elapsed_seconds=failure.elapsed_seconds,
                    maximum_seconds=failure.maximum_seconds,
                )
            case _:
                return cls(phase, type(failure).__name__, str(failure))

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "failure_phase": self.failure_phase,
            "failure_type": self.failure_type,
            "failure": self.failure,
        }
        if self.evidence is not None:
            evidence = self.evidence
            record.update(
                {
                    "failure_category": evidence.category.value,
                    "failure_evidence_id": evidence.evidence_id,
                    "native_stage": evidence.stage,
                    "provider_code": evidence.provider_code,
                    "failing_entities": list(evidence.entity_ids),
                    "requested": dict(evidence.requested),
                    "achieved": dict(evidence.achieved),
                    "locations": [list(point) for point in evidence.locations],
                    "checkpoint_id": evidence.checkpoint_id,
                }
            )
        if self.native_status is not None:
            record["native_status"] = self.native_status
            if self.elapsed_seconds is not None:
                record["elapsed_seconds"] = self.elapsed_seconds
            if self.maximum_seconds is not None:
                record["maximum_seconds"] = self.maximum_seconds
        return record


def record_failure(record: dict[str, Any], failure: Exception, phase: str) -> None:
    """Attach the shared exception identity and native failure evidence."""
    record.update(FailureRecord.from_exception(failure, phase).to_record())


def _record_mapping(value: object, name: str) -> dict[str, object]:
    """Validate nested serialized evidence before consumer access."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(
            f"_record_mapping.value must be a string-keyed evidence mapping; got {type(value).__name__!r}."
        )
    return value


def _phase_recordings(records: list[NativeMeshingPhaseMeasurement]) -> dict[str, object]:
    """Serialize only intervals emitted by the execution path."""
    totals: dict[str, float] = {}
    invocations: dict[str, int] = {}
    for record in records:
        totals[record.phase] = totals.get(record.phase, 0.0) + record.elapsed_seconds
        invocations[record.phase] = invocations.get(record.phase, 0) + record.invocations
    return {
        "status": "recorded" if records else "unmeasured",
        "measurements": [
            {
                "phase": record.phase,
                "elapsed_seconds": record.elapsed_seconds,
                "work_units": record.work_units,
                "invocations": record.invocations,
            }
            for record in records
        ],
        "total_seconds_by_phase": dict(sorted(totals.items())),
        "invocations_by_phase": dict(sorted(invocations.items())),
        "missing_phases_are_unmeasured": True,
        "interval_scope": "Some construction intervals include subphases; do not sum across phases.",
    }


def _quality_summary(report: phx.meshing.CellQualityReport, /) -> dict[str, object]:
    return {
        "minimum_mean_ratio": report.minimum_mean_ratio,
        "maximum_aspect_ratio": report.maximum_aspect_ratio,
        "minimum_scaled_jacobian": report.minimum_scaled_jacobian,
        "minimum_angle": report.minimum_angle,
        "maximum_angle": report.maximum_angle,
        "sampled_invalid_count": report.sampled_invalid_count,
    }


def _result_identity(result: phx.meshing.CellMeshingResult, /) -> dict[str, object]:
    return {
        "result_id": result.result_id,
        "topology_id": result.mesh.topology_id,
        "vertices": result.mesh.coordinates.shape[0],
        "cells": sum(block.cell_count for block in result.mesh.blocks),
        "audit_passed": result.audit.passed,
        "resource_guarantees": {
            "enforced": list(result.runtime.enforced_limits),
            "unenforced": list(result.runtime.unenforced_limits),
        },
        "execution_evidence": (
            None
            if result.execution_evidence is None
            else result.execution_evidence.to_record()
        ),
    }


def _mandatory_result(
    name: str,
    profile: dict[str, Any],
    assessment: dict[str, Any],
) -> dict[str, object]:
    """Build the result schema; certified refusals never satisfy positive workflows."""

    expected = (
        "refusal" if profile["expected_outcome"] == "expected-refusal" else "positive"
    )
    positive = assessment["positive_completion"] is True and expected == "positive"
    case_certified = assessment.get("case_certified") is True and expected == "positive"
    refusal = assessment["refusal_certified"] is True and expected == "refusal"
    status = (
        "positive-complete"
        if positive
        else "case-certified"
        if case_certified
        else "refusal-certified"
        if refusal
        else "not-run"
        if assessment["execution"] == "not-run"
        else "blocked"
    )
    body: dict[str, object] = {
        "kind": "mandatory-meshing-workflow-result",
        "profile": name,
        "workflow": profile["workflow"],
        "scenario": profile["scenario"],
        "expected_admission": expected,
        "observed_status": assessment["execution"],
        "status": status,
        "positive_completion": positive,
        "case_certified": case_certified,
        "refusal_certified": refusal,
        "scientific_certification": assessment["scientific_certification"],
        "phase_measurement_gaps": assessment.get("phase_measurement_gaps", []),
        "resource_evidence_gaps": assessment.get("resource_evidence_gaps", []),
        "blocking_status": assessment["blocking_status"],
        "release": assessment["release"],
        "leadership": assessment["leadership"],
        "source_reference_id": canonical_fingerprint(profile["source_reference"]),
    }
    if positive and refusal:
        raise qualification_failure(
            "_mandatory_result",
            ("A mandatory result cannot be both positive and a refusal."),
            {"positive": positive, "refusal": refusal},
        )
    if expected == "refusal" and positive:
        raise qualification_failure(
            "_mandatory_result",
            ("A refusal profile cannot satisfy a mandatory positive workflow."),
            {"expected": expected, "positive": positive},
        )
    return {**body, "result_id": canonical_fingerprint(body)}


_MANDATORY_WORKFLOW_RESULT_FIELDS = frozenset(
    {
        "kind",
        "workflow",
        "scenario",
        "expected_admission",
        "observed_status",
        "status",
        "positive_completion",
        "refusal_certified",
        "refusals_count_as_positive_completion",
        "scientific_certification",
        "mandatory_workflow_completion",
        "release",
        "leadership",
        "result_id",
    }
)


def _validate_mandatory_workflow_result(value: object, /) -> dict[str, object]:
    """Validate the fail-closed result emitted by every mandatory driver."""
    result = _record_mapping(value, "Mandatory meshing workflow result")
    missing = sorted(_MANDATORY_WORKFLOW_RESULT_FIELDS - set(result))
    unknown = sorted(set(result) - _MANDATORY_WORKFLOW_RESULT_FIELDS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(
            f"_validate_mandatory_workflow_result.fields must equal the mandatory workflow field set; got {(missing, unknown)!r}."
        )
    for field in (
        "kind",
        "workflow",
        "scenario",
        "expected_admission",
        "observed_status",
        "status",
        "scientific_certification",
        "mandatory_workflow_completion",
        "release",
        "leadership",
        "result_id",
    ):
        if not isinstance(result[field], str) or not result[field]:
            raise TypeError(
                f"_validate_mandatory_workflow_result.{field} must be a nonempty string; got {result[field]!r}."
            )
    for field in (
        "positive_completion",
        "refusal_certified",
        "refusals_count_as_positive_completion",
    ):
        if type(result[field]) is not bool:
            raise TypeError(
                f"_validate_mandatory_workflow_result.{field} must be a boolean; got {result[field]!r}."
            )
    if result["kind"] != "mandatory-meshing-workflow-result":
        raise ValueError(
            f"_validate_mandatory_workflow_result.kind must be 'mandatory-meshing-workflow-result'; got {result['kind']!r}."
        )
    if result["expected_admission"] not in ("positive", "refusal"):
        raise ValueError(
            f"_validate_mandatory_workflow_result.expected_admission must be 'positive' or 'refusal'; got {result['expected_admission']!r}."
        )
    if result["status"] not in ("positive-complete", "refusal-certified", "blocked"):
        raise ValueError(
            f"_validate_mandatory_workflow_result.status must represent a positive workflow, a certified refusal or a blocked workflow; got {result['status']!r}."
        )
    if result["refusals_count_as_positive_completion"] is not False:
        raise ValueError(
            f"_validate_mandatory_workflow_result.refusals_count_as_positive_completion must be False; got {result['refusals_count_as_positive_completion']!r}."
        )
    positive = (
        result["expected_admission"] == "positive"
        and result["observed_status"] == "passed"
        and result["scientific_certification"] == "passed"
        and result["mandatory_workflow_completion"] == "complete"
    )
    refusal = (
        result["expected_admission"] == "refusal"
        and result["observed_status"] == "expected-refusal"
    )
    expected_status = (
        "positive-complete" if positive else "refusal-certified" if refusal else "blocked"
    )
    if (
        result["positive_completion"] is not positive
        or result["refusal_certified"] is not refusal
        or result["status"] != expected_status
    ):
        raise ValueError(
            f"_validate_mandatory_workflow_result.completion must agree with admission, outcome and expected status; got {result!r}."
        )
    body = {name: field for name, field in result.items() if name != "result_id"}
    if result["result_id"] != canonical_fingerprint(body):
        raise ValueError(
            f"_validate_mandatory_workflow_result.result_id must equal the workflow body fingerprint; got {result['result_id']!r}."
        )
    return result


def _finalize_mandatory_cli_record(
    args: QualificationControls,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Attach the fail-closed outcome schema at the scenario JSON boundary."""
    if args.scenario == "native-corpus":
        return record
    workflow = SCENARIO_WORKFLOWS.get(args.scenario)
    if args.scenario == "native-lifecycle":
        workflow = meshing_case(args.corpus_case).workflow
    if workflow is None:
        return record
    expected_value = record.get("expected_admission", "positive")
    if expected_value not in ("positive", "refusal"):
        raise ValueError(
            f"_finalize_mandatory_cli_record.expected_admission must be 'positive' or 'refusal'; got {expected_value!r}."
        )
    observed = record.get("status", "failed")
    scientific = record.get("independent_scientific_certification", "unassessed")
    refusal = expected_value == "refusal" and observed == "expected-refusal"
    positive = (
        expected_value == "positive"
        and observed == "passed"
        and scientific == "passed"
        and record.get("mandatory_workflow_completion") == "complete"
    )
    result_body: dict[str, object] = {
        "kind": "mandatory-meshing-workflow-result",
        "workflow": workflow,
        "scenario": args.scenario,
        "expected_admission": expected_value,
        "observed_status": observed,
        "status": (
            "positive-complete"
            if positive
            else "refusal-certified"
            if refusal
            else "blocked"
        ),
        "positive_completion": positive,
        "refusal_certified": refusal,
        "refusals_count_as_positive_completion": False,
        "scientific_certification": scientific,
        "mandatory_workflow_completion": record.get(
            "mandatory_workflow_completion", "incomplete"
        ),
        "release": record.get("release_evidence", "unassessed"),
        "leadership": record.get("leadership_evidence", "unassessed"),
    }
    workflow_result = {
        **result_body,
        "result_id": canonical_fingerprint(result_body),
    }
    return {
        **record,
        "kind": "native-meshing-qualification-result",
        "positive_completion": positive,
        "refusal_certified": refusal,
        "workflow_result": workflow_result,
    }


def _field_template_identity(
    role: str, field: phx.discretization.FiniteElementFieldSpec
) -> dict[str, Any]:
    """Describe a mesh-free finite-element declaration and its physical role."""
    (element,) = field.elements
    return {
        "name": field.name,
        "role": role,
        "cell_kind": element.cell_kind,
        "space_family": element.family,
        "degree": element.degree,
        "continuity": element.continuity,
        "component_shape": field.component_shape,
        "representation": element.representation,
        "value_spec_id": element.value_spec.value_spec_id,
        "field_spec_id": field.field_spec_id,
    }


def _family_field_template_identities(
    controls: dict[str, str | int | float | tuple[float, ...]],
) -> list[dict[str, Any]]:
    requirements = _family_qualification_requirements()
    name = controls.get("family_case", requirements["parser_default"])
    if not isinstance(name, str):
        raise TypeError(
            f"qualification.family_case must name an authored profile; got {name!r}"
        )
    profile = requirements["profiles"][name]
    kinds = (
        profile["family"],
        *profile.get("family_policy", {}).get("allowed_transitions", ()),
    )
    templates = []
    for kind in sorted(set(kinds)):
        templates.extend(phx.meshing.qualification.family_solver_field_templates(kind))
        if profile["dimension"] == 3:
            for field_kind in phx.meshing.qualification.LAYER_CORE_FIELD_KINDS:
                templates.extend(
                    phx.meshing.qualification.family_transport_field_templates(
                        kind, kind=field_kind
                    )
                )
    identities = [_field_template_identity(role, field) for role, field in templates]
    scalar_quadrature = name == "curved-quarter-extrusion-quad"
    if (
        scalar_quadrature
        or profile["dimension"] == 3
        or profile["route"] in ("sweep", "structured_transfinite")
    ):
        identities.extend(
            dict(template)
            for template in phx.meshing.qualification.family_finite_volume_field_templates(
                profile.get("ambient_dimension", profile["dimension"]),
                scalar_quadrature=scalar_quadrature,
            )
        )
    return identities


def _hybrid_field_template_identities(
    controls: dict[str, str | int | float | tuple[float, ...]],
) -> list[dict[str, Any]]:
    from tools.meshing_qualification.scenarios.hybrid import (
        hybrid_solver_field_templates,
    )

    name = controls.get("hybrid_profile", QualificationControls().hybrid_profile)
    templates = []
    if name == "unit-box":
        for kind in ("prism", "tetrahedron"):
            templates.extend(hybrid_solver_field_templates(kind))
    elif name in ("concave-box-core-lifecycle", "archived-layer-core"):
        kinds = (
            ("prism", "tetrahedron")
            if name == "concave-box-core-lifecycle"
            else ("hexahedron", "prism", "pyramid", "tetrahedron")
        )
        for kind in kinds:
            templates.extend(phx.meshing.qualification.layer_core_field_templates(kind))
            templates.extend(
                phx.meshing.qualification.layer_core_history_field_templates(kind)
            )
    identities = [_field_template_identity(role, field) for role, field in templates]
    if name in ("concave-box-core-lifecycle", "archived-layer-core"):
        identities.extend(
            dict(template)
            for template in phx.meshing.qualification.layer_core_finite_volume_field_templates()
        )
    return identities


def _scenario_field_template_identities(
    scenario: str,
    controls: dict[str, str | int | float | tuple[float, ...]],
) -> list[dict[str, Any]]:
    """Collect declarations from the same factories used by scenario execution."""
    templates: tuple[tuple[str, phx.discretization.FiniteElementFieldSpec], ...] = ()
    identities: list[dict[str, Any]] = []
    match scenario:
        case "native-surface-pde":
            templates = phx.meshing.qualification.surface_field_templates()
        case "native-design-lifecycle":
            from tools.meshing_qualification.scenarios.design import (
                design_field_templates,
            )

            templates = design_field_templates()
        case "native-distributed-lifecycle":
            templates = phx.meshing.qualification.distributed_field_templates()
        case "native-periodic-lifecycle":
            templates = phx.meshing.qualification.periodic_field_templates()
        case "native-family-lifecycle":
            identities = _family_field_template_identities(controls)
        case "native-hybrid-lifecycle":
            identities = _hybrid_field_template_identities(controls)
        case "native-envelope-lifecycle":
            templates = (
                *phx.meshing.qualification.envelope_solver_field_templates(),
                *phx.meshing.qualification.envelope_transport_field_templates(
                    "tetrahedron"
                ),
            )
        case "native-overset-lifecycle":
            templates = phx.meshing.qualification.overset_field_templates()
        case "native-implicit-lifecycle":
            from tools.meshing_qualification.scenarios.implicit import (
                implicit_field_templates,
            )

            templates = implicit_field_templates()
        case "native-image-lifecycle":
            from tools.neurofluid_qualification import native_transport_field_templates

            identities = [
                dict(template) for template in native_transport_field_templates()
            ]
        case _:
            identities = []
    if templates:
        identities = [_field_template_identity(role, field) for role, field in templates]
    return sorted(
        identities,
        key=lambda field: (
            field["name"],
            field.get("cell_kind", ""),
            field["role"],
            field.get("field_spec_id", ""),
        ),
    )


def _scenario_source_reference(
    scenario: str,
    controls: dict[str, str | int | float | tuple[float, ...]],
) -> dict[str, object]:
    """Identify the generated source descriptor rather tha generated mesh output."""
    if scenario == "native-cad-lifecycle":
        fixture = load_cad_fixture()
        return {
            "kind": "reviewed-cad-corpus-source",
            "source_digest": fixture["file_sha256"],
            "source_revision": fixture["record_id"],
            "source_artifact": {
                "locator": "tests/data/cad/native_authored_boxes.json",
                "media_type": "application/json",
                "sha256": fixture["file_sha256"],
                "record_id": fixture["record_id"],
                "license": fixture["license"],
            },
            "source_provenance": fixture["provenance"],
            "independent_reference": fixture["independent_reference"],
        }
    if scenario == "native-hybrid-lifecycle":
        profile = controls.get("hybrid_profile")
        roles = {
            "curved-periodic-layer-core": "positive",
            "original-curved-periodic-refusal": "negative-refusal",
        }
        if isinstance(profile, str) and profile in roles:
            registration = load_curved_periodic_source_registration()
            sources = registration["sources"]
            if not isinstance(sources, list):
                raise TypeError(
                    f"_scenario_source_reference.sources must be a list; got {type(sources).__name__!r}."
                )
            matches = [
                source
                for source in sources
                if isinstance(source, dict) and source.get("role") == roles[profile]
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"_scenario_source_reference.matches must contain one source registration; got {len(matches)!r}."
                )
            source = matches[0]
            return {
                "kind": "reviewed-curved-periodic-layer-source",
                "source_id": source["source_id"],
                "source_digest": source["source_digest"],
                "source_revision": source["source_revision"],
                "source_artifact": source["source_artifact"],
                "source_provenance": {
                    "author": "PHYDRA, Inc.",
                    "license": "LicenseRef-PHYDRA-Proprietary",
                    "origin": source["upper_construction"],
                    "redistribution": "repository-internal",
                },
                "independent_reference": {
                    "kind": "exact-periodic-source-expression-trace",
                    "periodic_trace": source["periodic_trace"],
                    "orbit_ids": source["orbit_ids"],
                },
                "registration_id": registration["registration_id"],
            }
    descriptor = {
        "kind": "owning-scenario-source-descriptor",
        "scenario": scenario,
        "controls": [[name, controls[name]] for name in sorted(controls)],
        "driver": f"tools/meshing_qualification.py::{scenario}",
    }
    field_templates = _scenario_field_template_identities(scenario, controls)
    if field_templates:
        descriptor["field_templates"] = field_templates
    digest = canonical_fingerprint(descriptor)
    return {
        **descriptor,
        "source_digest": digest,
        "source_revision": digest,
        "source_artifact": {
            "locator": f"tools/meshing_qualification.py::{scenario}",
            "media_type": "text/x-python",
            "descriptor_sha256": digest,
            "license": "LicenseRef-PHYDRA-Proprietary",
        },
        "source_provenance": {
            "author": "PHYDRA, Inc.",
            "license": "LicenseRef-PHYDRA-Proprietary",
            "origin": "Owning deterministic qualification source factory",
            "redistribution": "repository-internal",
        },
        "independent_reference": {
            "kind": "scenario-owned-independent-acceptance",
            "owner": f"tools/meshing_qualification.py::{scenario}",
        },
    }


def _corpus_profiles() -> dict[str, dict[str, Any]]:
    """Enumerate real routes; selection never removes mandatory obligations."""

    load_corpus_registration()
    profiles: dict[str, dict[str, Any]] = {
        case.name: {
            "case_name": case.name,
            "workflow": case.workflow,
            "scenario": None,
            "controls": {},
            "full_workflow": False,
            "expected_outcome": "passed"
            if case.expected_admission == "positive"
            else "expected-refusal",
            "source_reference": corpus_identity(case),
        }
        for case in CORPUS
    }
    profiles["sphere-feature-surface"] = {
        "case_name": "sphere-feature-surface",
        "workflow": "planar-features",
        "scenario": None,
        "controls": {},
        "expected_outcome": "passed",
        "full_workflow": False,
        "source_reference": corpus_identity(meshing_case("sphere-feature-surface")),
    }
    for name in POLYHEDRAL_CASE_NAMES:
        profiles[name] = {
            "case_name": name,
            "workflow": "polyhedral-system",
            "scenario": None,
            "controls": {},
            "expected_outcome": "passed",
            "full_workflow": True,
            "source_reference": polyhedral_identity(name),
        }
    workflows = CORPUS_SCENARIO_WORKFLOWS
    surface_contract = _surface_qualification_requirements()
    hybrid_contract = _hybrid_qualification_requirements()
    choices: dict[str, tuple[str, tuple[str, ...]]] = {
        "native-cad-lifecycle": (
            "cad_format",
            _corpus_contract_choices(_cad_qualification_requirements(), "cad_format"),
        ),
        "native-periodic-lifecycle": (
            "periodic_case",
            _corpus_contract_choices(
                _periodic_qualification_requirements(), "periodic_case"
            ),
        ),
        "native-family-lifecycle": (
            "family_case",
            _corpus_contract_choices(_family_qualification_requirements(), None),
        ),
        "native-surface-pde": (
            "surface_case",
            _corpus_contract_choices(surface_contract, "surface_case"),
        ),
        "native-hybrid-lifecycle": (
            "hybrid_profile",
            _corpus_contract_choices(hybrid_contract, "hybrid_profile"),
        ),
    }
    for workflow, scenario in workflows.items():
        variants = choices.get(scenario)
        selections: tuple[dict[str, str | int | float | tuple[float, ...]], ...] = (
            ({},)
            if variants is None
            else tuple({variants[0]: value} for value in variants[1])
        )
        for controls in selections:
            name = (
                scenario
                if not controls
                else f"{scenario}:{next(iter(controls.values()))}"
            )
            if scenario == "native-surface-pde" and controls["surface_case"] == "sphere":
                name = "sphere-feature-surface"
            contract: Mapping[str, object] = {}
            if scenario == "native-hybrid-lifecycle":
                profile_name = controls.get("hybrid_profile")
                if not isinstance(profile_name, str):
                    raise TypeError(
                        f"_corpus_profiles.profile_name must be a string; got {type(profile_name).__name__!r}."
                    )
                contract = hybrid_contract["profile_contracts"][profile_name]
                if profile_name == "curved-periodic-layer-core":
                    controls = {
                        **controls,
                        "layer_count": 2,
                        "first_thickness": 0.01,
                        "layer_growth": 1.0,
                        "geometry_order": 2,
                    }
                elif profile_name == "concave-box-core-lifecycle":
                    # The closed container authors one fluid material across
                    # both the boundary-layer cells and the remaining core.
                    controls = {
                        **controls,
                        "hybrid_region_densities": (2.0,),
                        "hybrid_material_site": "fluid-density",
                    }
            elif scenario == "native-distributed-lifecycle":
                controls = {
                    **controls,
                    "distributed_bisection_compatibility": BisectionCompatibility.UNIFORM_REFINEMENT.value,
                }
            source_reference = (
                profiles[name]["source_reference"]
                if name in profiles
                else _scenario_source_reference(scenario, controls)
            )
            profiles[name] = {
                "case_name": scenario,
                "workflow": workflow,
                "scenario": scenario,
                "controls": controls,
                "expected_outcome": "passed",
                "source_reference": source_reference,
                **contract,
            }
    return profiles
