# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from subprocess import TimeoutExpired
from typing import Any, assert_never, final

import equinox as eqx

import phydrax as phx
from phydrax import canonical_fingerprint
from phydrax._meshcore import MeshcoreError
from phydrax.meshing import (
    BisectionCompatibility,
    MeshcoreStatus,
    NativeMeshingPhaseMeasurement,
)
from phydrax.meshing._measurements import (
    native_publication_resource_binding,
    NATIVE_RUNTIME_MEASUREMENT_REQUIREMENTS,
    validate_native_execution_interchange,
)
from phydrax.typing import parse
from tools._meshing_cases import (
    CORPUS,
    corpus_identity,
    frozen_request_record,
    load_cad_fixture,
    load_corpus_registration,
    load_curved_periodic_source_registration,
    meshing_case,
    POLYHEDRAL_CASE_NAMES,
    polyhedral_identity,
    prepare_polyhedral_case,
)
from tools.meshing_qualification._controls import (
    _cad_qualification_requirements,
    _corpus_contract_choices,
    _family_qualification_requirements,
    _hybrid_qualification_requirements,
    _periodic_qualification_profile_contracts,
    _periodic_qualification_requirements,
    _surface_qualification_requirements,
    authored_resource_envelope,
    CAD_CASES,
    CadCase,
    CORPUS_SCENARIOS,
    DESIGN_CASES,
    IMAGE_CASES,
    IMPLICIT_CASES,
    meshing_limit_values,
    MeshingLimitControls,
    OVERSET_PROFILES,
    QualificationControls,
    RECONSTRUCTION_CASES,
    SCENARIO_WORKFLOWS,
    with_meshing_limits,
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


def record_failure(
    record: dict[str, Any], failure: Exception, phase: str
) -> dict[str, object]:
    """Attach and return the classified exception identity and native evidence."""
    detail = FailureRecord.from_exception(failure, phase).to_record()
    match failure:
        case (
            TimeoutError()
            | TimeoutExpired()
            | MeshcoreError(status=MeshcoreStatus.TIMEOUT)
            | phx.meshing.MeshingFailure(
                category=phx.meshing.MeshingFailureCategory.TIMED_OUT
            )
        ):
            detail["status"] = "timeout"
        case _:
            detail["status"] = "failed"
    record.update(detail)
    return detail


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
        "resource_binding": native_publication_resource_binding(
            result.result_id, result.runtime.runtime_id, result.execution_evidence,
            enforced_limits=result.runtime.enforced_limits,
            unenforced_limits=result.runtime.unenforced_limits,
        ),
        "execution_evidence": (
            None
            if result.execution_evidence is None
            else result.execution_evidence.to_record()
        ),
    }


def _publication_receipt_gaps(
    publication: dict[str, Any], prefix: str, runtime: object
) -> list[str]:
    try:
        observation = validate_native_execution_interchange(
            publication["execution_evidence"]
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return [prefix + "invalid-native-receipt"]
    gaps = []
    phases = publication["execution_evidence"]["phases"]
    if any(
        phases[index]["status_name"] != "OK"
        for index in observation["current_phase_indexes"]
    ):
        gaps.append(prefix + "native-phase-status")
    if not observation["preparation_complete"]:
        gaps.append(prefix + "incomplete-native-preparation-receipt")
    if observation["managed_admission_peak_bytes_upper"] is None:
        gaps.append(prefix + "unobserved-managed-admission-envelope")
    binding = publication.get("resource_binding")
    if not isinstance(binding, dict) or set(binding) != {
        "kind",
        "result_id",
        "request_id",
        "runtime_id",
        "native_build",
        "root_receipt_id",
        "binding_id",
        "native_managed_claim",
        "outside_admission_measurement_requirements",
    }:
        return gaps + [prefix + "actual-publication-resource-binding"]
    body = {name: value for name, value in binding.items() if name != "binding_id"}
    if (
        binding["kind"] != "native-publication-resource-binding"
        or binding["result_id"] != publication.get("result_id")
        or any(
            not isinstance(binding[name], str) or not binding[name]
            for name in ("result_id", "request_id", "runtime_id")
        )
        or binding["request_id"] != observation["owner_id"]
        or binding["root_receipt_id"] != observation["root_receipt_id"]
        or binding["binding_id"] != canonical_fingerprint(body)
    ):
        gaps.append(prefix + "actual-publication-resource-binding")
    guarantees = publication.get("resource_guarantees")
    if (
        not isinstance(guarantees, dict)
        or binding["native_managed_claim"]
        != {
            "scope": "native-managed-request",
            "enforced": guarantees.get("enforced"),
            "unenforced": guarantees.get("unenforced"),
        }
        or binding["outside_admission_measurement_requirements"]
        != list(NATIVE_RUNTIME_MEASUREMENT_REQUIREMENTS)
    ):
        gaps.append(prefix + "publication-resource-scope-binding")
    native = binding["native_build"]
    measured_native = runtime.get("meshcore_build") if isinstance(runtime, dict) else None
    if (
        not isinstance(native, dict)
        or not isinstance(measured_native, dict)
        or any(
            not isinstance(native.get(name), str)
            or not native[name]
            or native[name] != measured_native.get(name)
            for name in ("release", "source_digest", "binary_digest", "configuration")
        )
    ):
        gaps.append(prefix + "publication-native-build-binding")
    return gaps


def _source_prefix_boundary_limits(
    limits: phx.meshing.MeshingLimits,
    control: str,
    observed: int | float,
) -> phx.meshing.MeshingLimits:
    """Own the original integer R−1 and elapsed-time half-boundary probe policy."""
    declared = meshing_limit_values(limits)
    if control not in declared:
        raise ValueError(f"Unknown source-prefix resource control {control!r}.")
    if (
        isinstance(observed, bool)
        or not math.isfinite(observed)
        or observed > declared[control]
    ):
        raise ValueError(
            "Source-prefix boundary observation must fit its original limit."
        )
    if control == "maximum_wall_seconds":
        if observed <= 0:
            raise ValueError("Source-prefix elapsed boundary must be positive.")
        reduced = observed * 0.5
    else:
        if not isinstance(observed, int) or observed <= 1:
            raise ValueError("Source-prefix integer boundary must exceed one.")
        reduced = observed - 1
    return replace(
        MeshingLimitControls.from_limits(limits),
        **{control: reduced},
    ).to_limits()


def _require_source_prefix_request(
    value: object,
    expected: dict[str, object],
) -> None:
    actual = _record_mapping(value, "source-prefix frozen request")
    if canonical_fingerprint(actual) != canonical_fingerprint(expected):
        raise ValueError(
            "Source-prefix trial must retain its actual frozen source and controls."
        )


def _source_prefix_observations(
    positive: dict[str, Any],
    limits: phx.meshing.MeshingLimits,
) -> dict[str, int | float]:
    declared = meshing_limit_values(limits)
    bounds = _record_mapping(positive.get("bounds"), "source-prefix bounds")
    if set(bounds) != set(declared):
        raise ValueError(
            "Source-prefix evidence must retain all original resource bounds."
        )
    observed = {}
    for control, limit in declared.items():
        bound = _record_mapping(bounds[control], f"source-prefix {control}")
        value = bound.get("observed")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= limit
            or control != "maximum_wall_seconds"
            and not isinstance(value, int)
            or bound.get("original_limit") != limit
            or bound.get("runtime_enforced") is not True
            or bound.get("within_original_bound") is not True
        ):
            raise ValueError(
                f"Source-prefix {control} must have an actual enforced observation."
            )
        observed[control] = value
    return observed


def _source_prefix_quantities(
    value: object,
    name: str,
) -> tuple[tuple[str, int | float], ...]:
    record = _record_mapping(value, name)
    quantities = []
    for key, quantity in sorted(record.items()):
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
            raise TypeError(f"{name}.{key} must be a serialized numeric quantity.")
        quantities.append((key, quantity))
    return tuple(quantities)


def _require_source_prefix_refusal(
    refusal: dict[str, Any],
    control: str,
) -> None:
    expected = (
        phx.meshing.MeshingFailureCategory.TIMED_OUT
        if control == "maximum_wall_seconds"
        else phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    )
    if (
        refusal.get("failure_type") != "MeshingFailure"
        or refusal.get("failure_category") != expected.value
        or type(refusal.get("logical_findings_count")) is not int
        or refusal["logical_findings_count"] != 0
    ):
        raise ValueError(
            "Source-prefix refusal must retain complete typed failure evidence."
        )
    requested = _source_prefix_quantities(
        refusal.get("requested"),
        "source-prefix refusal requested",
    )
    achieved = _source_prefix_quantities(
        refusal.get("achieved"),
        "source-prefix refusal achieved",
    )
    if not requested or not achieved:
        raise ValueError(
            "Source-prefix refusal must retain requested and achieved quantities."
        )
    evidence = phx.meshing.MeshingFailureEvidence(
        expected,
        refusal["failure"],
        provider_code=refusal["provider_code"],
        stage=refusal["native_stage"],
        entity_ids=tuple(refusal["failing_entities"]),
        locations=tuple(tuple(point) for point in refusal["locations"]),
        requested=requested,
        achieved=achieved,
        checkpoint_id=refusal["checkpoint_id"],
    )
    if evidence.evidence_id != refusal.get("failure_evidence_id"):
        raise ValueError(
            "Source-prefix refusal must match its owning failure evidence identity."
        )


def _source_prefix_work_graph(publication: dict[str, Any]) -> str:
    """Compare charged work and graph edges, not freshly earned clocks/receipt IDs."""
    receipt = publication["execution_evidence"]
    validate_native_execution_interchange(receipt)
    return canonical_fingerprint([
        {
            name: phase[name]
            for name in (
                "owner_id", "status_name", "work_counters",
                "source_preparation_work_units", "source_preparation_geometry_queries",
                "total_work_units", "total_geometry_queries",
                "preparation_evidence", "consumer_evidence", "history_evidence",
            )
        }
        for phase in receipt["phases"]
    ])


def _require_source_prefix_trial(
    refusal: dict[str, Any],
    observed: int | float,
    source: Any,
    request: phx.meshing.VolumeMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    positive: dict[str, Any],
    runtime: object,
) -> None:
    control = refusal["control"]
    reduced = _source_prefix_boundary_limits(request.limits, control, observed)
    reduced_values = meshing_limit_values(reduced)
    if (
        refusal.get("original_limit") != meshing_limit_values(request.limits)[control]
        or refusal.get("reduced_limit") != reduced_values[control]
        or refusal.get("limits_id") != reduced.limits_id
    ):
        raise ValueError(
            "Source-prefix refusal must use the owning reduced resource boundary."
        )
    _require_source_prefix_request(
        refusal.get("frozen_request"),
        frozen_request_record(
            source, request if control == "maximum_work_units"
            else with_meshing_limits(request, reduced), options,
        ),
    )
    _require_source_prefix_refusal(refusal, control)
    if control == "maximum_work_units":
        _require_source_prefix_work_trial(
            refusal, observed, source, request, options, positive, reduced
        )
        return
    sharp = refusal.get("measured_R_positive")
    if control != "maximum_scratch_bytes":
        if sharp is not None:
            raise ValueError(
                "Only the managed-storage probe declares a sharp R-positive replay."
            )
        return
    sharp = _record_mapping(sharp, "source-prefix sharp R")
    sharp_limits = replace(
        MeshingLimitControls.from_limits(request.limits),
        maximum_scratch_bytes=observed,
    ).to_limits()
    _require_source_prefix_request(
        sharp.get("frozen_request"),
        frozen_request_record(
            source, with_meshing_limits(request, sharp_limits), options
        ),
    )
    publication: dict[str, Any] = _record_mapping(
        sharp.get("result"), "source-prefix R result"
    )
    gaps = _publication_receipt_gaps(publication, "source-prefix-R:", runtime)
    if gaps:
        raise ValueError(f"Source-prefix R publication is incomplete: {gaps}.")
    observation = validate_native_execution_interchange(publication["execution_evidence"])
    if (
        sharp.get("limit") != observed
        or sharp.get("limits_id") != sharp_limits.limits_id
        or sharp.get("observed_managed_admission_envelope") != observed
        or observation["managed_admission_peak_bytes_upper"] != observed
        or sharp.get("mesh_and_geometry_bitwise_preserved") is not True
    ):
        raise ValueError(
            "Source-prefix R must admit its actually measured managed envelope."
        )
    for name in ("mesh_fingerprint", "geometry_fingerprint"):
        original = _record_mapping(positive.get(name), f"source-prefix original {name}")
        current = _record_mapping(sharp.get(name), f"source-prefix R {name}")
        if not original or canonical_fingerprint(original) != canonical_fingerprint(
            current
        ):
            raise ValueError(
                "Source-prefix R must preserve the original numerical mesh and geometry."
            )

def _require_source_prefix_work_trial(
    refusal: dict[str, Any],
    observed: int | float,
    source: Any,
    request: phx.meshing.VolumeMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    positive: dict[str, Any],
    reduced: phx.meshing.MeshingLimits,
) -> None:
    sharp = _record_mapping(refusal.get("measured_R_positive"), "source-prefix work R")
    sharp_limits = replace(
        MeshingLimitControls.from_limits(request.limits), maximum_work_units=int(observed)
    ).to_limits()
    _require_source_prefix_request(
        sharp.get("frozen_request"), frozen_request_record(source, request, options)
    )
    original: dict[str, Any] = _record_mapping(positive.get("result"), "original work result")
    replay: dict[str, Any] = _record_mapping(sharp.get("result"), "replayed work result")
    original_receipt = original["execution_evidence"]
    replay_receipt = replay["execution_evidence"]
    original_observation = validate_native_execution_interchange(original_receipt)
    replay_observation = validate_native_execution_interchange(replay_receipt)
    if (
        original_observation["managed_admission_peak_bytes_upper"] is None
        or replay_observation["managed_admission_peak_bytes_upper"] is None
    ):
        raise ValueError("Work boundary requires a genuine managed admission envelope for original and replay.")
    original_root = original_receipt["phases"][0]
    replay_root = replay_receipt["phases"][0]
    preparation_index = original_root["preparation_evidence"]
    if preparation_index is None:
        raise ValueError("Work boundary requires the original preparation receipt.")
    preparation = original_receipt["phases"][preparation_index]["receipt_id"]
    for trial, limits in ((refusal, reduced), (sharp, sharp_limits)):
        if (
            trial.get("mechanical_admission") != meshing_limit_values(limits)
            or trial.get("plan_id") != original_root["owner_id"]
            or trial.get("preparation_receipt_id") != preparation
        ):
            raise ValueError("Work boundary must retain its original plan and preparation, separate from mechanical admission.")
    if (
        sharp.get("limit") != observed
        or sharp.get("limits_id") != sharp_limits.limits_id
        or sharp.get("observed_work_units") != observed
        or replay_root["total_work_units"] != observed
        or original_root["total_work_units"] != observed
        or replay_root["receipt_id"] == original_root["receipt_id"]
        or replay_root["status_name"] != "OK"
        or replay["result_id"] != original["result_id"]
        or replay["topology_id"] != original["topology_id"]
        or replay.get("audit_passed") is not True
        or _source_prefix_work_graph(replay) != _source_prefix_work_graph(original)
        or replay_receipt["phases"][replay_root["preparation_evidence"]]["receipt_id"] != preparation
        or sharp.get("mesh_and_geometry_bitwise_preserved") is not True
    ):
        raise ValueError("Work R must be a fresh replay of the original publication and phase graph, not copied or replanned evidence.")
    for name in ("mesh_fingerprint", "geometry_fingerprint"):
        if not positive.get(name) or sharp.get(name) != positive[name]:
            raise ValueError("Work R must preserve the original mesh and geometry bytes.")



def _bound_source_prefix_resources(
    record: Mapping[str, Any],
    warm_publication: dict[str, Any],
    /,
) -> dict[str, Any]:
    """Validate original-prefix trials, never substitute them for a complete request."""
    prefix: dict[str, Any] = _record_mapping(
        record.get("source_prefix_resources"),
        "source-prefix resources",
    )
    disposition = {
        "status": "passed",
        "mandatory_capacity_gate": "complete",
        "admitted_scope": "SOURCEPREFIX",
        "complete_request_R_claimed": False,
        "original_source_and_controls_preserved": True,
        "unobserved_enforcement_claimed": False,
    }
    if (
        any(
            type(prefix.get(name)) is not type(value) or prefix.get(name) != value
            for name, value in disposition.items()
        )
        or prefix.get("unobserved_or_unenforced_original_bounds") != []
    ):
        raise ValueError(
            "Source-prefix disposition must preserve its original limited scope."
        )
    controls: dict[str, Any] = _record_mapping(
        record.get("qualification_controls"),
        "source-prefix qualification controls",
    )
    source, request, options = prepare_polyhedral_case(
        controls["polyhedral_source"],
        controls["resolution"],
        controls["capacity"],
        controls["timeout"],
        source=True,
    )
    _require_source_prefix_request(
        prefix.get("frozen_request"),
        frozen_request_record(source, request, options),
    )
    positive: dict[str, Any] = _record_mapping(
        prefix.get("positive"), "source-prefix positive"
    )
    publication: dict[str, Any] = _record_mapping(
        positive.get("result"), "source-prefix result"
    )
    for label, value in (
        ("source-prefix:", publication),
        ("warm-source:", warm_publication),
    ):
        gaps = _publication_receipt_gaps(value, label, record.get("runtime"))
        if gaps:
            raise ValueError(f"Source-prefix publication binding is incomplete: {gaps}.")
    original = validate_native_execution_interchange(publication["execution_evidence"])
    warm = validate_native_execution_interchange(warm_publication["execution_evidence"])
    original_phases = publication["execution_evidence"]["phases"]
    warm_phases = warm_publication["execution_evidence"]["phases"]
    original_ids = {
        original_phases[index]["receipt_id"]
        for index in original["current_phase_indexes"]
    }
    warm_ids = {
        warm_phases[index]["receipt_id"] for index in warm["current_phase_indexes"]
    }
    if (
        not original_ids <= warm_ids
        or publication["result_id"] != warm_publication["result_id"]
        or publication["topology_id"] != warm_publication["topology_id"]
        or publication.get("audit_passed") is not True
    ):
        raise ValueError(
            "Original source-prefix work must be paid in the actual warm source request."
        )
    observed = _source_prefix_observations(positive, request.limits)
    root = original_phases[0]
    recorded = {
        "maximum_work_units": root["total_work_units"],
        "maximum_geometry_queries": root["total_geometry_queries"],
        "maximum_wall_seconds": root["total_elapsed_seconds"],
        "maximum_scratch_bytes": original["managed_admission_peak_bytes_upper"],
        "maximum_cavity_cells": max(
            original_phases[index]["work_counters"]["peak_cavity_cells"]
            for index in original["current_phase_indexes"]
        ),
        "maximum_cells": publication["cells"],
        "maximum_vertices": publication["vertices"],
    }
    if any(observed[name] != value for name, value in recorded.items()):
        raise ValueError(
            "Source-prefix observations must match their actual publication and receipt."
        )
    refusals = prefix.get("refusals")
    if not isinstance(refusals, list) or any(
        not isinstance(item, dict) for item in refusals
    ):
        raise TypeError("Source-prefix refusals must be actual trial records.")
    expected = {
        name
        for name, value in observed.items()
        if value > (0 if name == "maximum_wall_seconds" else 1)
    }
    names = [item.get("control") for item in refusals]
    if (
        any(not isinstance(name, str) for name in names)
        or len(names) != len(set(names))
        or set(names) != expected
    ):
        raise ValueError(
            "Source-prefix trials must cover every observed reducible original limit once."
        )
    for refusal in refusals:
        _require_source_prefix_trial(
            refusal,
            observed[refusal["control"]],
            source,
            request,
            options,
            positive,
            record.get("runtime"),
        )
    return prefix


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
        "qualification_limitations": assessment.get("qualification_limitations", []),
        "blocking_status": assessment["blocking_status"],
        "release": assessment["release"],
        "leadership": assessment["leadership"],
        "source_reference_id": canonical_fingerprint(profile["source_reference"]),
        "profile_controls_id": canonical_fingerprint(
            {
                "scenario_controls": profile["controls"],
                "resolution_scale": profile["resolution_scale"],
                "minimum_adaptation_rounds": profile["minimum_adaptation_rounds"],
            }
        ),
        "sample_binding": assessment["sample_binding"],
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


def _record_evidence_id(record: Mapping[str, object], /) -> str:
    """Fingerprint a scientific record exactly as its JSON consumer decodes it.

    The child emits JSON and the campaign consumer parses it, so both sides hash
    the same decoded value. Non-finite values cannot reach a consumer (the parser
    rejects them), so they are hashed by their JSON tokens only for the producer.
    """
    decoded = json.loads(json.dumps(dict(record)))
    canonical = json.dumps(
        decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scenario_workflow(controls: Mapping[str, Any], /) -> str | None:
    """Return the mandatory workflow a child scenario's verdict certifies."""
    scenario = controls["scenario"]
    if scenario == "native-corpus":
        return None
    if scenario == "native-lifecycle":
        return meshing_case(controls["corpus_case"]).workflow
    return SCENARIO_WORKFLOWS.get(scenario)


def _workflow_outcome(record: Mapping[str, Any], /) -> dict[str, object]:
    """Derive admission, case and mandatory completion from one scientific record."""
    expected = record.get("expected_admission", "positive")
    if expected not in ("positive", "refusal"):
        raise ValueError(
            f"_workflow_outcome.expected_admission must be 'positive' or 'refusal'; got {expected!r}."
        )
    observed = record.get("status", "failed")
    scientific = record.get("independent_scientific_certification", "unassessed")
    mandatory = record.get("mandatory_workflow_completion", "incomplete")
    case_contract = record.get("case_contract_completion", mandatory)
    scientific_case = (
        expected == "positive" and observed == "passed" and scientific == "passed"
    )
    positive = scientific_case and mandatory == "complete"
    refusal = expected == "refusal" and observed == "expected-refusal"
    return {
        "expected_admission": expected,
        "observed_status": observed,
        "status": "positive-complete"
        if positive
        else "refusal-certified"
        if refusal
        else "blocked",
        "positive_completion": positive,
        "case_accepted": positive or scientific_case and case_contract == "complete",
        "refusal_certified": refusal,
        "refusals_count_as_positive_completion": False,
        "scientific_certification": scientific,
        "mandatory_workflow_completion": mandatory,
        "case_contract_completion": case_contract,
        "release": record.get("release_evidence", "unassessed"),
        "leadership": record.get("leadership_evidence", "unassessed"),
    }


def _workflow_verdict(
    controls: Mapping[str, Any], evidence: Mapping[str, Any], /
) -> dict[str, object] | None:
    """Derive the verdict bound to its controls, request, runtime and record.

    ``evidence`` is the full emitted record without ``workflow_result``. The
    producer and every consumer call this function; a verdict is accepted only
    when it equals the one derived from its enclosing record.
    """
    workflow = _scenario_workflow(controls)
    if workflow is None:
        return None
    request = evidence.get("request_id")
    runtime = evidence.get("runtime")
    body: dict[str, object] = {
        "kind": "mandatory-meshing-workflow-result",
        "workflow": workflow,
        "scenario": controls["scenario"],
        **_workflow_outcome(evidence),
        "controls_id": canonical_fingerprint(dict(controls)),
        "request_id": request if isinstance(request, str) else None,
        "runtime_id": None if runtime is None else canonical_fingerprint(runtime),
        "evidence_id": _record_evidence_id(evidence),
    }
    return {**body, "result_id": canonical_fingerprint(body)}


def _finalize_mandatory_cli_record(
    args: QualificationControls,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Attach the fail-closed verdict bound to this specific scenario record."""
    controls = args.to_record()
    if _scenario_workflow(controls) is None:
        return record
    outcome = _workflow_outcome(record)
    evidence = {
        **record,
        "kind": "native-meshing-qualification-result",
        "qualification_controls": controls,
        "positive_completion": outcome["positive_completion"],
        "refusal_certified": outcome["refusal_certified"],
    }
    return {**evidence, "workflow_result": _workflow_verdict(controls, evidence)}


def _bound_workflow_verdict(
    record: Mapping[str, Any],
    expected_controls: Mapping[str, object],
    expected_build_id: str,
    /,
) -> dict[str, object]:
    """Accept a child verdict only if it is the one its record and launch derive."""
    controls = _record_mapping(record.get("qualification_controls"), "controls")
    mismatched = sorted(
        name
        for name, value in expected_controls.items()
        if name not in controls
        or canonical_fingerprint(controls[name]) != canonical_fingerprint(value)
    )
    if mismatched:
        raise ValueError(
            f"_bound_workflow_verdict.qualification_controls must equal the launched controls; got mismatched {mismatched!r}."
        )
    runtime = _record_mapping(record.get("runtime"), "runtime")
    if runtime.get("build_id") != expected_build_id:
        raise ValueError(
            f"_bound_workflow_verdict.runtime.build_id must equal the launcher's loaded build; got {runtime.get('build_id')!r}."
        )
    evidence = {
        name: value for name, value in record.items() if name != "workflow_result"
    }
    verdict = _workflow_verdict(controls, evidence)
    if verdict is None or record.get("workflow_result") != verdict:
        raise ValueError(
            f"_bound_workflow_verdict.workflow_result must equal the verdict derived from its enclosing record; got {record.get('workflow_result')!r}."
        )
    if (
        record.get("positive_completion") is not verdict["positive_completion"]
        or record.get("refusal_certified") is not verdict["refusal_certified"]
    ):
        raise ValueError(
            f"_bound_workflow_verdict.completion must equal the bound verdict; got positive={record.get('positive_completion')!r}, refusal={record.get('refusal_certified')!r}."
        )
    return verdict


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
        for field_kind in phx.meshing.qualification.LAYER_CORE_FIELD_KINDS:
            templates.extend(
                phx.meshing.qualification.family_transport_field_templates(
                    kind, kind=field_kind
                )
            )
    identities = [_field_template_identity(role, field) for role, field in templates]
    scalar_quadrature = name == "curved-quarter-extrusion-quad"
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

    name = controls.get("hybrid_profile", QualificationControls().hybrid_profile)
    templates = []
    if name in (
        "unit-box",
        "concave-box-core-lifecycle",
        "archived-layer-core",
        "curved-periodic-layer-core",
    ):
        kinds = (
            ("hexahedron", "prism", "pyramid", "tetrahedron")
            if name == "archived-layer-core"
            else ("prism", "tetrahedron")
        )
        for kind in kinds:
            templates.extend(phx.meshing.qualification.layer_core_field_templates(kind))
            templates.extend(
                phx.meshing.qualification.layer_core_history_field_templates(kind)
            )
            if name == "unit-box":
                from phydrax.meshing.qualification._layer_continuation import (
                    layer_core_original_scalar_field_template,
                )
                templates.append(
                    ("solve:intensive", layer_core_original_scalar_field_template(kind))
                )
    identities = [_field_template_identity(role, field) for role, field in templates]
    if name in (
        "unit-box",
        "concave-box-core-lifecycle",
        "archived-layer-core",
        "curved-periodic-layer-core",
    ):
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


def _cad_source_reference(
    controls: Mapping[str, str | int | float | tuple[float, ...]],
) -> dict[str, object]:
    """Bind each CAD operation to its authored data and independent reference."""
    case = parse(controls.get("cad_case", "planar-box"), CadCase, "cad_case")
    external_artifact: dict[str, object] | None = None
    match case:
        case "planar-box":
            fixture = load_cad_fixture()
            locator = "tests/data/cad/native_authored_boxes.json"
            operation = None
        case "quadric-routine" | "quadric-larger":
            from tools.meshing_qualification.scenarios.cad import load_cad_quadric_fixture

            fixture = load_cad_quadric_fixture()
            locator = "tests/data/cad/native_authored_quadrics.json"
            operation = "intersection"
        case "external-routine" | "external-combined":
            from tools.meshing_qualification.scenarios.cad import (
                load_cad_external_fixture,
            )

            fixture = load_cad_external_fixture()
            locator = "tests/data/cad/nist_ftc_11_external.json"
            operation = (
                "import"
                if case == "external-routine"
                else "subtract-authored-radial-hole"
            )
            external_artifact = {
                "locator": f"tests/data/cad/{fixture['source_file']}",
                "media_type": "model/step",
                "sha256": fixture["source_sha256"],
            }
        case _:
            assert_never(case)
    digest = fixture["file_sha256"]
    revision = fixture["record_id"]
    if operation is not None:
        descriptor = {
            "fixture_digest": digest,
            "fixture_id": revision,
            "operation": operation,
        }
        digest = canonical_fingerprint(descriptor)
        revision = digest
    record: dict[str, object] = {
        "kind": "reviewed-cad-corpus-source",
        "source_digest": digest,
        "source_revision": revision,
        "source_artifact": {
            "locator": locator,
            "media_type": "application/json",
            "sha256": fixture["file_sha256"],
            "record_id": fixture["record_id"],
            "license": fixture["license"],
        },
        "source_provenance": fixture["provenance"],
        "independent_reference": fixture["independent_reference"],
    }
    if operation is not None:
        record["operation"] = operation
    if external_artifact is not None:
        record["external_source_artifact"] = external_artifact
    return record


def _scenario_source_reference(
    scenario: str,
    controls: dict[str, str | int | float | tuple[float, ...]],
    *,
    resolution: int | None = None,
) -> dict[str, object]:
    """Identify the authored source descriptor rather than generated mesh output."""
    if scenario == "native-cad-lifecycle":
        return _cad_source_reference(controls)
    if scenario == "native-periodic-lifecycle":
        from tools.meshing_qualification.scenarios.periodic import (
            _periodic_qualification_source_reference,
        )

        defaults = QualificationControls()
        profile = controls.get("periodic_case", defaults.periodic_case)
        source_resolution = (
            controls.get("resolution", defaults.resolution)
            if resolution is None
            else resolution
        )
        if not isinstance(profile, str) or not isinstance(source_resolution, int):
            raise TypeError(
                "periodic source registration requires a profile and integer resolution"
            )
        return _periodic_qualification_source_reference(profile, source_resolution)
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


def _corpus_profiles(resolution: int | None = None) -> dict[str, dict[str, Any]]:
    """Enumerate real routes; selection never removes mandatory obligations."""

    load_corpus_registration()
    profiles: dict[str, dict[str, Any]] = {
        case.name: {
            "case_name": case.name,
            "workflow": case.workflow,
            "scenario": None,
            "controls": {},
            "full_workflow": case.expected_admission == "positive",
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
    surface_contract = _surface_qualification_requirements()
    hybrid_contract = _hybrid_qualification_requirements()
    periodic_contracts = _periodic_qualification_profile_contracts()
    defaults = QualificationControls()
    routine_selectors = {
        "native-implicit-lifecycle": ("implicit_case", defaults.implicit_case),
        "native-image-lifecycle": ("image_case", defaults.image_case),
        "native-design-lifecycle": ("design_case", defaults.design_case),
        "native-overset-lifecycle": ("overset_profile", defaults.overset_profile),
    }
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
        "native-implicit-lifecycle": ("implicit_case", IMPLICIT_CASES),
        "native-image-lifecycle": ("image_case", IMAGE_CASES),
        "native-design-lifecycle": ("design_case", DESIGN_CASES),
        "native-overset-lifecycle": ("overset_profile", OVERSET_PROFILES),
        "native-reconstruction-lifecycle": ("reconstruction_case", RECONSTRUCTION_CASES),
    }
    for entry in CORPUS_SCENARIOS:
        workflow, scenario = entry.workflow, entry.name
        variants = choices.get(scenario)
        selections: tuple[dict[str, str | int | float | tuple[float, ...]], ...] = (
            ({},)
            if variants is None
            else tuple({variants[0]: value} for value in variants[1])
        )
        if scenario == "native-cad-lifecycle":
            selections = tuple(
                {**selection, "cad_case": case}
                for selection in selections
                for case in CAD_CASES
            )
        for controls in selections:
            if scenario == "native-cad-lifecycle":
                name = f"{scenario}:{controls['cad_format']}"
                if controls["cad_case"] != defaults.cad_case:
                    name += f":{controls['cad_case']}"
            elif scenario in routine_selectors:
                selector, default = routine_selectors[scenario]
                name = (
                    scenario
                    if controls[selector] == default
                    else f"{scenario}:{controls[selector]}"
                )
            else:
                name = (
                    scenario
                    if not controls
                    else f"{scenario}:{next(iter(controls.values()))}"
                )
            if scenario == "native-surface-pde" and controls["surface_case"] == "sphere":
                name = "sphere-feature-surface"
            contract: Mapping[str, object] = {}
            if scenario == "native-periodic-lifecycle":
                profile_name = controls["periodic_case"]
                if not isinstance(profile_name, str):
                    raise TypeError(
                        "periodic profile must be a declared string identifier"
                    )
                contract = periodic_contracts[profile_name]
            elif scenario == "native-hybrid-lifecycle":
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
                elif profile_name in ("unit-box", "concave-box-core-lifecycle"):
                    # The closed container authors one fluid material across
                    # both the boundary-layer cells and the remaining core.
                    controls = {
                        **controls,
                        "hybrid_region_densities": (2.0,),
                        "hybrid_material_site": "fluid-density",
                    }
            elif (
                scenario == "native-image-lifecycle"
                and controls["image_case"] == "oblique-junction-transport"
            ):
                controls = {**controls, "image_size": 6}
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
    return _corpus_request_profiles(
        profiles, defaults.resolution if resolution is None else resolution
    )


def _corpus_request_profiles(
    profiles: dict[str, dict[str, Any]],
    base_resolution: int,
) -> dict[str, dict[str, Any]]:
    """Register distinct routine/larger requests, independent of one global size."""
    expanded: dict[str, dict[str, Any]] = {}
    larger_cases = {
        "planar-hole-feature": 8,
        "plc-material-cavity": 2,
        **{name: 2 for name in POLYHEDRAL_CASE_NAMES},
    }
    for name, original in sorted(profiles.items()):
        profile = {
            **original,
            "controls": dict(original["controls"]),
            "resolution_scale": 1,
            "minimum_adaptation_rounds": 0,
        }
        scenario = profile["scenario"]
        controls = profile["controls"]
        if (
            scenario == "native-implicit-lifecycle"
            and controls["implicit_case"] != "sphere"
        ):
            profile["minimum_adaptation_rounds"] = 2
        if scenario == "native-distributed-lifecycle":
            controls.update(distributed_parts=4, restart_parts=2, distributed_processes=4)
        expanded[name] = profile
        if scenario == "native-periodic-lifecycle":
            profile["source_reference"] = _scenario_source_reference(
                scenario, controls, resolution=base_resolution
            )
        if (
            profile["expected_outcome"] != "passed"
            and scenario != "native-periodic-lifecycle"
        ):
            continue
        scale = larger_cases.get(name)
        if scenario in (
            "native-surface-pde",
            "native-reconstruction-lifecycle",
            "native-periodic-lifecycle",
            "native-distributed-lifecycle",
        ):
            scale = 2
        elif scenario == "native-family-lifecycle" and controls["family_case"] in (
            "curved-cavity-hex",
            "curved-material-hex",
        ):
            scale = 2
        elif scenario == "native-hybrid-lifecycle" and controls["hybrid_profile"] in (
            "concave-box-core-lifecycle",
            "curved-periodic-layer-core",
        ):
            scale = 2
        if scale is None:
            continue
        larger_controls = dict(controls)
        if scenario == "native-distributed-lifecycle":
            larger_controls.update(
                distributed_parts=6, restart_parts=3, distributed_processes=6
            )
        expanded[f"{name}:larger"] = {
            **profile,
            "controls": larger_controls,
            "resolution_scale": scale,
        }
        if scenario == "native-hybrid-lifecycle":
            expanded[f"{name}:larger"]["minimum_adaptation_rounds"] = 2
        if scenario == "native-periodic-lifecycle":
            expanded[f"{name}:larger"]["source_reference"] = _scenario_source_reference(
                scenario, larger_controls, resolution=base_resolution * scale
            )
    for name, profile in expanded.items():
        envelope = authored_resource_envelope(
            profile["scenario"],
            profile["case_name"],
            profile["controls"],
            larger=name.endswith(":larger"),
        )
        if envelope is not None:
            profile["controls"].update(envelope["effective"])
            profile["resource_envelope"] = envelope
    return dict(sorted(expanded.items()))


def _corpus_request_dimensions(
    profile: Mapping[str, Any], resolution: int, adaptation_rounds: int
) -> tuple[int, int]:
    """Apply the same registered size and lifecycle depth in both launchers."""
    scale = profile["resolution_scale"]
    minimum_rounds = profile["minimum_adaptation_rounds"]
    if (
        type(scale) is not int
        or scale < 1
        or type(minimum_rounds) is not int
        or minimum_rounds < 0
    ):
        raise ValueError(
            "corpus.profile must declare positive size scaling and nonnegative lifecycle depth"
        )
    case_name = profile["case_name"]
    rounds = (
        0
        if case_name in POLYHEDRAL_CASE_NAMES and case_name != "polyhedral-periodic-cube"
        else max(adaptation_rounds, minimum_rounds)
    )
    return resolution * scale, rounds
