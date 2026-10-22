# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from contextlib import nullcontext
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, final, Literal, TypedDict

import phydrax as phx
from phydrax import canonical_fingerprint
from tools._meshing_cases import (
    load_corpus_registration,
    MANDATORY_WORKFLOWS,
)
from tools.meshing_qualification._campaign import (
    _launcher_build_id,
    _missing_profile_inputs,
    _native_campaign_sample,
    _native_driver_controls,
    CampaignSample,
    corpus_phase_evidence,
)
from tools.meshing_qualification._controls import (
    qualification_timeout,
    QualificationControls,
)
from tools.meshing_qualification._records import (
    _bound_source_prefix_resources,
    _bound_workflow_verdict,
    _corpus_profiles,
    _corpus_request_dimensions,
    _mandatory_result,
    _publication_receipt_gaps,
)
from tools.meshing_qualification._runtime import (
    _cpu_device_peak_capability,
    _runtime_identity,
)


def _corpus_compiler_phases(
    sample: CampaignSample,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Consume the canonical adapter, bound back to its actual owning payload."""
    evidence = sample.phase_evidence
    capacity = evidence.get("controlling_entity_capacity")
    records = evidence.get("compiler_epochs")
    if type(capacity) is not int or capacity <= 0:
        return [], ["compiler-phase-controlling-capacity"]
    if not isinstance(records, list) or any(
        not isinstance(item, dict) for item in records
    ):
        return [], ["actual-compiler-phase-evidence"]
    raw = sample.to_record()
    child = raw.get("child_evidence")
    declared = (
        child.get("qualification_controls")
        if isinstance(child, dict)
        else raw.get("qualification_controls")
    )
    if isinstance(declared, dict) and (
        type(declared.get("capacity")) is not int or declared["capacity"] != capacity
    ):
        return records, ["compiler-phase-controlling-capacity-binding"]
    if raw.get("scenario") == "native-polyhedral-lifecycle":
        case = raw.get("case")
        controls = case.get("controls") if isinstance(case, dict) else None
        accepted = raw.get("accepted_epochs", ())
        if (
            not isinstance(controls, (list, tuple))
            or any(not isinstance(item, str) for item in controls)
            or not isinstance(accepted, (list, tuple))
            or any(not isinstance(item, str) or not item for item in accepted)
        ):
            return records, ["polyhedral-compiler-phase-source-shape"]
    try:
        expected = corpus_phase_evidence(raw, capacity)["compiler_epochs"]
        matches_source = canonical_fingerprint(records) == canonical_fingerprint(expected)
    except (KeyError, TypeError, ValueError, OverflowError):
        return records, ["compiler-phase-source-shape"]
    gaps = []
    if not matches_source:
        gaps.append("compiler-phase-source-binding")
        actual_paths = {
            item.get("evidence_path")
            for item in records
            if isinstance(item.get("evidence_path"), str)
        }
        gaps.extend(
            f"missing-compiler-phase:{item['evidence_path']}"
            for item in expected
            if item["evidence_path"] not in actual_paths
        )
    return records, gaps


def _measured_seconds(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and value >= 0
    )


def _corpus_phase_gaps(
    sample: CampaignSample, *, full_workflow: bool, refusal: bool
) -> list[str]:
    records, phase_missing = _corpus_compiler_phases(sample)
    for index, record in enumerate(records):
        path = record.get("evidence_path")
        label = path if isinstance(path, str) and path else f"compiler-phase-{index}"
        solver = record.get("solver")
        if (
            not isinstance(path, str)
            or not path
            or record.get("phase") not in ("warm", "cold")
            or type(record.get("epoch")) is not int
            or record["epoch"] < 0
            or type(record.get("controlling_entity_capacity")) is not int
            or record.get("controlling_entity_capacity")
            != sample.phase_evidence["controlling_entity_capacity"]
            or not isinstance(solver, str)
            or solver not in ("epoch", "fe", "vem", "fv", "fv-transferred-update")
            and not (
                solver.startswith("fv-trace:")
                and solver not in ("fv-trace:", "fv-trace:None")
            )
            or record.get("epoch_id") is not None
            and (not isinstance(record["epoch_id"], str) or not record["epoch_id"])
        ):
            phase_missing.append(f"{label}:compiler-phase-identity")
        for field in (
            "temporary_bytes",
            "output_bytes",
            "generated_code_bytes",
            "logical_retained_bytes",
        ):
            if not _measured_bytes(record.get(field)):
                phase_missing.append(f"{label}:{field}")
        warm = record.get("warm_samples_seconds")
        if (
            not isinstance(warm, (list, tuple))
            or not warm
            or any(not _measured_seconds(value) for value in warm)
        ):
            phase_missing.append(f"{label}:warm_samples_seconds")
        if solver != "epoch":
            for field in ("kernel", "memory_scope"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    phase_missing.append(f"{label}:{field}")
            if record.get("memory_status") != "measured":
                phase_missing.append(f"{label}:memory_status")
            for field in (
                "lowering_seconds",
                "compile_seconds",
                "first_execution_seconds",
            ):
                if not _measured_seconds(record.get(field)):
                    phase_missing.append(f"{label}:{field}")
    if not records and full_workflow and not refusal:
        phase_missing.append("accepted-epoch-phase-measurements")
    return phase_missing


_REQUEST_LIMITS = frozenset(
    (
        "vertices",
        "edges",
        "faces",
        "cells",
        "connectivity_entries",
        "data_bytes",
        "scratch_bytes",
        "work_units",
        "wall_seconds",
        "cavity_cells",
        "geometry_queries",
    )
)
# Canonical certification guarantees connectivity normalization, not any resource cap.
_ROUTE_CAPABILITIES = frozenset(
    ("grid_capacity", "surface_capacity", "projection", "canonical_connectivity")
)
# These are declarations of partial coverage, not synonyms for request-wide limits.
_PARTIAL_NATIVE_SCOPES = frozenset(
    (
        "work_units:native_and_metered_host",
        "work_units:native_and_coordinate_batches",
        "wall_seconds:native_and_host_boundaries",
        "cavity_cells:native",
        "geometry_queries:native_and_source_batches",
        "scratch_bytes:managed_native_and_bounded_host",
    )
)
# The policy owner explicitly places unmanaged scratch outside admission.
# In contrast, "unmetered_host" and "nonpreemptible_host" do not establish
# that the excluded host work is outside the native/managed request. Those
# labels still fail closed, even if another declaration claims a bare limit.
_OUTSIDE_ADMISSION_SCOPES = frozenset(("scratch_bytes:unmanaged_host_device_compiler",))


class _CorpusResourceAssessment(TypedDict):
    admission_coverage: Literal["proved", "unproved", "not-assessed"]
    runtime_measurement_availability: Literal[
        "complete",
        "complete-with-cpu-device-peak-limitation",
        "incomplete",
        "not-assessed",
    ]
    admission_gaps: list[str]
    runtime_measurement_gaps: list[str]
    limitations: list[str]
    approved_measurement_limitations: list[dict[str, Any]]
    publications: list[dict[str, Any]]
    publication_evidence_paths: list[str]
    source_prefix_resource_boundary: Literal[
        "not-assessed", "unproved", "proved-source-prefix"
    ]
    source_prefix_resource_evidence: dict[str, Any] | None


def _corpus_publications(
    sample: CampaignSample, *, full_workflow: bool, refusal: bool
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Retain actual publication endpoints, never resource summaries or history."""
    publications = []
    paths = []
    gaps = []
    raw = sample.to_record()
    polyhedral = raw.get("scenario") == "native-polyhedral-lifecycle"

    def publish(value: object, path: str) -> None:
        if not isinstance(value, dict) or not value:
            gaps.append(f"publication:{path}:actual-native-publication")
            return
        publications.append(value)
        paths.append(path)

    generation = raw.get("generation")
    if generation is not None:
        if polyhedral and isinstance(generation, dict) and "result_id" not in generation:
            for role, result in generation.items():
                if role not in ("source", "target"):
                    gaps.append(f"publication:generation:unknown-result-role-{role}")
                publish(result, f"generation.{role}")
            if full_workflow and not refusal:
                gaps.extend(
                    f"publication:generation.{role}:actual-native-publication"
                    for role in ("source", "target")
                    if role not in generation
                )
        else:
            publish(generation, "generation")
    for index, epoch in enumerate(sample.epochs):
        if "result" in epoch:
            publish(epoch["result"], f"epochs[{index}].result")
    checkpoint = sample.checkpoint
    if isinstance(checkpoint, dict) and "continued_result" in checkpoint:
        publish(checkpoint["continued_result"], "checkpoint.continued_result")
    if not polyhedral:
        return publications, paths, gaps

    if "independent_target_generation" in raw:
        publish(raw["independent_target_generation"], "independent_target_generation")
    warm = raw.get("warm_publications")
    if warm is None:
        if full_workflow and not refusal:
            gaps.append("publication:warm_publications:actual-native-publications")
    elif not isinstance(warm, (list, tuple)) or any(
        not isinstance(item, dict) for item in warm
    ):
        gaps.append("publication:warm_publications:invalid-publication-sequence")
    else:
        for index, entry in enumerate(warm):
            if (
                type(entry.get("epoch")) is not int
                or entry["epoch"] != index
                or not isinstance(entry.get("epoch_id"), str)
                or not entry["epoch_id"]
            ):
                gaps.append(
                    f"publication:warm_publications[{index}]:actual-epoch-binding"
                )
            publish(entry.get("result"), f"warm_publications[{index}].result")
        if full_workflow and not refusal:
            case = raw.get("case")
            controls = case.get("controls") if isinstance(case, dict) else None
            expected = None
            if not isinstance(controls, (list, tuple)) or any(
                not isinstance(value, str) for value in controls
            ):
                gaps.append("publication:warm_publications:unknown-case-controls")
            elif "periodic-quotient" in controls:
                requested = raw.get("qualification_controls")
                rounds = (
                    requested.get("adaptation_rounds")
                    if isinstance(requested, dict)
                    else None
                )
                if type(rounds) is not int or rounds < 0:
                    gaps.append("publication:warm_publications:unknown-requested-rounds")
                else:
                    expected = 1 + 2 * rounds
            else:
                expected = 2
            if expected is not None and len(warm) != expected:
                gaps.append(
                    f"publication:warm_publications:expected-{expected}-actual-{len(warm)}"
                )
            accepted = raw.get("accepted_epochs")
            if accepted is not None and (
                not isinstance(accepted, (list, tuple))
                or list(accepted) != [entry.get("epoch_id") for entry in warm]
            ):
                gaps.append("publication:warm_publications:accepted-epoch-binding")

    cold = raw.get("cold")
    fresh = cold.get("fresh_interpreter") if isinstance(cold, dict) else None
    if isinstance(fresh, dict) and "cold_request_result" in fresh:
        publish(
            fresh["cold_request_result"], "cold.fresh_interpreter.cold_request_result"
        )
    elif full_workflow and not refusal:
        gaps.append(
            "publication:cold.fresh_interpreter.cold_request_result:actual-native-publication"
        )
    return publications, paths, gaps


def _resource_sample_binding_gaps(sample: CampaignSample) -> list[str]:
    """Revalidate the actual emitted child, not a status-only sample surrogate."""
    record = sample.to_record()
    child = record.get("child_evidence")
    runtime = record.get("runtime")
    controls = record.get("qualification_controls")
    if (
        not isinstance(child, dict)
        or not isinstance(runtime, dict)
        or not isinstance(controls, dict)
    ):
        return ["actual-child-resource-binding"]
    build_id = runtime.get("build_id")
    if not isinstance(build_id, str) or not build_id:
        return ["actual-child-build-binding"]
    if build_id != _launcher_build_id():
        return ["actual-child-build-binding"]
    try:
        verdict = _bound_workflow_verdict(child, controls, build_id)
    except (KeyError, TypeError, ValueError):
        return ["actual-child-resource-binding"]
    if (
        any(
            sample.binding.get(name) != verdict[name]
            for name in ("request_id", "runtime_id", "controls_id", "evidence_id")
        )
        or sample.binding.get("workflow_result_id") != verdict["result_id"]
    ):
        return ["actual-child-resource-binding"]
    fields = (
        "scenario",
        "generation",
        "epochs",
        "checkpoint",
        "runtime",
        "peak_process_resident_bytes",
        "process_peak_scope",
        "memory_scope",
        "device_memory",
    )
    if (
        record.get("scenario") == "native-polyhedral-lifecycle"
        or child.get("scenario") == "native-polyhedral-lifecycle"
    ):
        fields += (
            "case",
            "source_physics",
            "physics",
            "accepted_epochs",
            "cold",
            "independent_target_generation",
            "warm_publications",
            "source_prefix_resources",
        )
    if any(record.get(name) != child.get(name) for name in fields):
        return ["actual-child-resource-binding"]
    if (
        sample.generation != child.get("generation")
        or list(sample.epochs) != child.get("epochs", [])
        or sample.checkpoint != child.get("checkpoint")
        or sample.resource_evidence.get("process_peak_resident_bytes")
        != child.get("peak_process_resident_bytes")
        or sample.resource_evidence.get("process_peak_scope")
        != child.get("process_peak_scope")
        or sample.resource_evidence.get("device_memory") != child.get("device_memory")
    ):
        return ["actual-child-resource-binding"]
    return []


def _measured_bytes(value: object) -> bool:
    return type(value) is int and value >= 0


def _verified_cpu_peak_disposition(
    sample: CampaignSample, device_record: dict[str, Any], index: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Separate the actual unsupported capability from its unavailable observation."""
    standard = "device_id" in device_record
    transport = isinstance(device_record.get("device"), str)
    if standard == transport:
        return None, None
    runtime = sample.to_record().get("runtime")
    if not isinstance(runtime, dict) or runtime.get("backend") != "cpu":
        return None, None
    environment = runtime.get("environment")
    if not isinstance(environment, dict) or runtime.get(
        "environment_id"
    ) != canonical_fingerprint(environment):
        return None, None
    devices = environment.get("devices")
    if (
        not isinstance(devices, list)
        or not devices
        or any(
            not isinstance(item, dict) or item.get("platform") != "cpu"
            for item in devices
        )
    ):
        return None, None
    import jax

    for actual in jax.local_devices():
        if (
            standard
            and (
                type(device_record.get("device_id")) is not int
                or device_record["device_id"] != actual.id
            )
            or transport
            and device_record["device"] != str(actual)
        ):
            continue
        if "platform" in device_record and device_record["platform"] != "cpu":
            continue
        if (
            "device_kind" in device_record
            and device_record["device_kind"] != actual.device_kind
        ):
            continue
        capability = _cpu_device_peak_capability(actual)
        if (
            capability is None
            or environment.get("jax_release") != capability["jax_release"]
        ):
            continue
        identity = capability["device"]
        if not isinstance(identity, dict):
            continue
        matched = [
            item
            for item in devices
            if all(item.get(name) == value for name, value in identity.items())
        ]
        if (
            len(matched) != 1
            or matched[0].get("device_allocator_peak_capability") != capability
        ):
            continue
        explicit_unavailable = (
            standard
            and device_record.get("status") == "unavailable"
            and (
                "peak_bytes" in device_record
                and device_record["peak_bytes"] is None
                and "memory_statistics" in device_record
                and device_record["memory_statistics"] is None
                or "peak_bytes" not in device_record
                and "statistics" in device_record
                and device_record["statistics"] is None
            )
            or transport
            and "status" not in device_record
            and "peak_bytes" in device_record
            and device_record["peak_bytes"] is None
            and isinstance(device_record.get("unavailable_reason"), str)
            and bool(device_record["unavailable_reason"].strip())
        )
        if not explicit_unavailable:
            return capability, None
        return capability, {
            "kind": "approved-cpu-device-allocator-peak-limitation",
            "measurement": "device_allocation_peak_bytes",
            "availability": "unavailable",
            "peak_bytes": None,
            "device_index": index,
            "device_measurement": device_record,
            "reason": device_record.get("unavailable_reason", capability["reason"]),
            "verified_capability": capability,
            "runtime_id": sample.binding["runtime_id"],
            "build_id": runtime["build_id"],
            "environment_id": runtime["environment_id"],
            "sample_binding": sample.binding,
            "qualification_scope": (
                "CPU-only qualification retains this device-allocator-peak limitation; "
                "all other measurements, native/managed admission, scientific and cold gates remain required."
            ),
        }
    return None, None


def _corpus_runtime_measurement_gaps(
    sample: CampaignSample, *, allow_verified_cpu_limitation: bool
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Keep measured availability distinct from the explicit CPU-only limitation."""
    gaps = []
    limitations = [
        "Native/managed request admission does not hard-cap compiler, device or process memory."
    ]
    approved = []
    resources = sample.resource_evidence
    if not _measured_bytes(resources.get("process_peak_resident_bytes")):
        gaps.append("runtime:process-peak-resident-bytes")
    scope = resources.get("process_peak_scope")
    # These two owners publish the process interval under memory_scope; retain
    # that actual scope, already bound to the emitted child, without renaming it.
    if scope is None and sample.to_record().get("scenario") in (
        "native-design-lifecycle",
        "native-implicit-lifecycle",
    ):
        scope = sample.to_record().get("memory_scope")
    if not isinstance(scope, str) or not scope.strip():
        gaps.append("runtime:process-peak-scope")
    else:
        limitations.append(f"process:{scope}")
    devices = resources.get("device_memory")
    if not isinstance(devices, list) or not devices:
        gaps.append("runtime:device-memory")
    else:
        for index, device in enumerate(devices):
            if not isinstance(device, dict):
                gaps.append(f"runtime:device-{index}:memory")
                continue
            peak = device.get("peak_bytes")
            status = device.get("status")
            if "peak_bytes" not in device and "statistics" in device:
                # Design/implicit publish the actual backend statistics bank.
                statistics = device["statistics"]
                peak = (
                    statistics.get("peak_bytes_in_use")
                    if isinstance(statistics, dict)
                    else None
                )
            if (
                "status" not in device
                and "unavailable_reason" in device
                and isinstance(device.get("device"), str)
            ):
                # The image/transport owner reports explicit unavailable_reason,
                # not the generic lifecycle status field. None never invents bytes.
                status = (
                    "backend-reported"
                    if device["unavailable_reason"] is None
                    else "unavailable"
                )
            cpu_capability, approved_cpu = (
                _verified_cpu_peak_disposition(sample, device, index)
                if allow_verified_cpu_limitation
                else (None, None)
            )
            if cpu_capability is not None and peak is not None:
                gaps.append(
                    f"runtime:device-{index}:contradicts-verified-unsupported-cpu-capability"
                )
            if approved_cpu is not None:
                approved.append(approved_cpu)
                limitations.append(
                    f"device-{index}:APPROVED CPU-only device allocator peak UNAVAILABLE; "
                    "not zero, RSS, a snapshot or static compiler memory; qualification retains this limitation"
                )
            else:
                if not _measured_bytes(peak):
                    gaps.append(f"runtime:device-{index}:peak-bytes")
                if status != "backend-reported":
                    gaps.append(f"runtime:device-{index}:measurement-status")
            limitations.append(
                f"device-{index}:{device.get('platform')}:{status}; backend peak, not native admission"
            )
            if device.get("platform") == "cpu" and peak is None:
                limitations.append(
                    f"device-{index}:CPU shared host RAM; allocator high-water unavailable; "
                    "process RSS and live-array/heap snapshots are not a measured device peak"
                )
    if approved:
        import jax

        expected = {
            (device.process_index, device.id)
            for device in jax.local_devices()
            if _cpu_device_peak_capability(device) is not None
        }
        observed = [
            (
                item["verified_capability"]["device"]["process_index"],
                item["verified_capability"]["device"]["id"],
            )
            for item in approved
        ]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            gaps.append("runtime:incomplete-cpu-device-capability-observations")
    compiler_phases, compiler_binding_gaps = _corpus_compiler_phases(sample)
    gaps.extend(f"runtime:{gap}" for gap in compiler_binding_gaps)
    for index, compiler in enumerate(compiler_phases):
        path = compiler.get("evidence_path", f"compiler-phase-{index}")
        for field in ("temporary_bytes", "output_bytes", "generated_code_bytes"):
            if not _measured_bytes(compiler.get(field)):
                gaps.append(f"{path}:compiler-{field}")
        if compiler.get("solver") != "epoch":
            if compiler.get("memory_status") != "measured":
                gaps.append(f"{path}:compiler-memory-status")
            if (
                not isinstance(compiler.get("memory_scope"), str)
                or not compiler["memory_scope"].strip()
            ):
                gaps.append(f"{path}:compiler-memory-scope")
    return gaps, limitations, approved


def _corpus_resource_assessment(
    sample: CampaignSample, *, full_workflow: bool, refusal: bool
) -> _CorpusResourceAssessment:
    publications, publication_paths, admission_gaps = _corpus_publications(
        sample, full_workflow=full_workflow, refusal=refusal
    )
    raw = sample.to_record()
    prefix_required = raw.get("scenario") == "native-polyhedral-lifecycle" and (
        full_workflow and not refusal or "source_prefix_resources" in raw
    )
    prefix_scope: Literal["not-assessed", "unproved", "proved-source-prefix"] = (
        "not-assessed"
    )
    prefix_evidence: dict[str, Any] | None = None
    binding_gaps = []
    runtime_gaps = []
    limitations = []
    approved_limitations = []
    assessed = bool(publications) or full_workflow and not refusal or prefix_required
    if assessed:
        binding_gaps = _resource_sample_binding_gaps(sample)
        admission_gaps.extend(binding_gaps)
        runtime_gaps, limitations, approved_limitations = (
            _corpus_runtime_measurement_gaps(
                sample, allow_verified_cpu_limitation=not binding_gaps
            )
        )
    if prefix_required:
        prefix_scope = "unproved"
        child = raw.get("child_evidence")
        warm = child.get("warm_publications") if isinstance(child, dict) else None
        if binding_gaps or not isinstance(child, dict):
            admission_gaps.append("source-prefix-resource-boundary:actual-source-binding")
        elif (
            not isinstance(warm, (list, tuple))
            or not warm
            or not isinstance(warm[0], dict)
            or not isinstance(warm[0].get("result"), dict)
        ):
            admission_gaps.append(
                "source-prefix-resource-boundary:actual-warm-source-publication"
            )
        else:
            try:
                prefix_evidence = _bound_source_prefix_resources(child, warm[0]["result"])
            except (KeyError, TypeError, ValueError, OverflowError) as failure:
                admission_gaps.append(
                    f"source-prefix-resource-boundary:invalid-evidence:{failure}"
                )
            else:
                prefix_scope = "proved-source-prefix"
                limitations.append(
                    "SOURCEPREFIX R/R-1 proves only the original generation prefix; "
                    "complete-request R is not claimed and full warm/cold receipts remain required"
                )
    for index, publication in enumerate(publications):
        prefix = f"publication-{index}:"
        guarantees = publication.get("resource_guarantees", {})
        if not isinstance(guarantees, dict):
            guarantees = {}
        enforced = guarantees.get("enforced", ())
        unenforced = guarantees.get("unenforced", ())
        if not isinstance(enforced, (list, tuple)) or any(
            not isinstance(name, str) for name in enforced
        ):
            enforced = ()
            admission_gaps.append(prefix + "invalid-enforced-limits")
        if not isinstance(unenforced, (list, tuple)) or any(
            not isinstance(name, str) for name in unenforced
        ):
            unenforced = ()
            admission_gaps.append(prefix + "invalid-unenforced-limits")
        admission_gaps.extend(
            prefix + f"missing-{name}" for name in sorted(_REQUEST_LIMITS - set(enforced))
        )
        for name in enforced:
            if name in _PARTIAL_NATIVE_SCOPES:
                admission_gaps.append(prefix + f"partial-{name}")
            elif name not in _REQUEST_LIMITS | _ROUTE_CAPABILITIES:
                admission_gaps.append(prefix + f"unknown-enforced-scope-{name}")
        for name in unenforced:
            if name in _OUTSIDE_ADMISSION_SCOPES:
                limitations.append(prefix + f"outside-admission:{name}")
            else:
                admission_gaps.append(prefix + f"unenforced-{name}")
        receipt = publication.get("execution_evidence")
        if not isinstance(receipt, dict) or not receipt.get("phases"):
            admission_gaps.append(prefix + "actual-ended-native-receipt")
        else:
            admission_gaps.extend(
                _publication_receipt_gaps(
                    publication, prefix, sample.to_record().get("runtime")
                )
            )
    if not publications and full_workflow and not refusal:
        admission_gaps.append("accepted-publication-runtime-and-receipt")
    return {
        "admission_coverage": (
            "unproved" if admission_gaps else "proved" if assessed else "not-assessed"
        ),
        "runtime_measurement_availability": (
            "incomplete"
            if runtime_gaps
            else "complete-with-cpu-device-peak-limitation"
            if approved_limitations
            else "complete"
            if assessed
            else "not-assessed"
        ),
        "admission_gaps": admission_gaps,
        "runtime_measurement_gaps": runtime_gaps,
        "limitations": limitations,
        "approved_measurement_limitations": approved_limitations,
        # Keep every original scoped label and ended receipt, unchanged.
        "publications": publications,
        "publication_evidence_paths": publication_paths,
        "source_prefix_resource_boundary": prefix_scope,
        "source_prefix_resource_evidence": prefix_evidence,
    }


def _corpus_resource_gaps(
    sample: CampaignSample, *, full_workflow: bool, refusal: bool
) -> list[str]:
    assessment = _corpus_resource_assessment(
        sample, full_workflow=full_workflow, refusal=refusal
    )
    return assessment["admission_gaps"] + assessment["runtime_measurement_gaps"]


def _corpus_profile_assessment(
    profile: dict[str, Any],
    sample: CampaignSample | None,
) -> dict[str, Any]:
    """Keep execution, scientific, workflow and release facts distinct."""
    if sample is None:
        return {
            "execution": "not-run",
            "scientific_certification": "unassessed",
            "mandatory_positive_completion": "incomplete",
            "positive_completion": False,
            "refusal_certified": False,
            "qualification_eligible": False,
            "release": "unassessed",
            "leadership": "unassessed",
            "blocking_status": "not-run",
            "sample_binding": None,
            "qualification_limitations": [],
        }
    matched = sample.status == profile["expected_outcome"]
    refusal = profile["expected_outcome"] == "expected-refusal"
    refusal_evidence = (
        matched
        and refusal
        and (
            sample.failure_evidence_id is not None
            and bool(sample.failure_evidence_id)
            and sample.requested is not None
            and sample.achieved is not None
            and sample.source_bits_preserved is True
        )
    )
    full_workflow = profile.get("full_workflow", True) is True
    phase_missing = _corpus_phase_gaps(
        sample, full_workflow=full_workflow, refusal=refusal
    )
    resources = _corpus_resource_assessment(
        sample, full_workflow=full_workflow, refusal=refusal
    )
    resource_missing = resources["admission_gaps"] + resources["runtime_measurement_gaps"]
    acceptance = sample.independent_acceptance
    scientific = sample.independent_scientific_certification == "passed" or (
        acceptance is not None and acceptance.get("passed") is True
    )
    completion = sample.mandatory_workflow_completion
    case_completion = sample.case_contract_completion
    positive_case = (
        not refusal
        and matched
        and scientific
        and case_completion == "complete"
        and not phase_missing
        and not resource_missing
    )
    eligible = refusal_evidence or positive_case
    mandatory_positive = positive_case and full_workflow and completion == "complete"
    return {
        "deliberate_refusal_certification": (
            "passed" if refusal_evidence else "not-established"
        ),
        "refusal_certified": refusal_evidence,
        "phase_measurement_gaps": phase_missing,
        "resource_evidence_gaps": resource_missing,
        "resource_assessment": resources,
        "qualification_limitations": resources["approved_measurement_limitations"],
        "positive_case_completion": case_completion,
        "case_certified": positive_case,
        "positive_completion": mandatory_positive,
        "execution": sample.status,
        "expected_outcome_matched": matched,
        "scientific_certification": "passed" if scientific else "unassessed",
        "mandatory_positive_completion": completion,
        "qualification_eligible": eligible,
        "release": sample.release_evidence,
        "leadership": sample.leadership_evidence,
        "blocking_status": None if eligible else sample.status,
        "sample_binding": sample.binding,
    }


@final
class _CorpusCampaign(phx.StrictModule):
    selected: tuple[str, ...]
    samples: list[dict[str, Any]]
    by_profile: dict[str, CampaignSample]
    missing_inputs: dict[str, list[str]]


def _execute_corpus_profiles(
    args: QualificationControls, profiles: dict[str, dict[str, Any]]
) -> _CorpusCampaign:
    arguments = args.to_record()
    requested_profiles = args.corpus_profile
    selected = (
        tuple(profiles) if requested_profiles is None else tuple(requested_profiles)
    )
    unknown = set(selected) - profiles.keys()
    if unknown:
        raise ValueError(
            "native_corpus.corpus_profile must name declared profiles; "
            f"got unknown profiles {sorted(unknown)}"
        )
    if len(selected) != len(set(selected)):
        raise ValueError(
            f"native_corpus.corpus_profile must contain unique profiles; got {selected!r}"
        )
    samples = []
    by_profile = {}
    missing_inputs = {}
    for name in selected:
        profile = profiles[name]
        missing = _missing_profile_inputs(profile, arguments)
        if missing:
            missing_inputs[name] = missing
            continue
        scenario = profile["scenario"]
        controls = {} if scenario is None else _native_driver_controls(scenario, args)
        controls.update(profile["controls"])
        resolution, rounds = _corpus_request_dimensions(
            profile, args.resolution, args.adaptation_rounds
        )
        with (
            TemporaryDirectory(prefix="phydrax-corpus-distributed-")
            if scenario == "native-distributed-lifecycle"
            and "distributed_checkpoint_root" not in controls
            else nullcontext()
        ) as checkpoint_directory:
            if checkpoint_directory is not None:
                controls["distributed_checkpoint_root"] = str(
                    Path(checkpoint_directory) / "checkpoint"
                )
            sample = _native_campaign_sample(
                profile["case_name"],
                resolution,
                args.capacity,
                qualification_timeout(controls, args.timeout),
                args.repeats,
                args.target_error,
                rounds,
                "native",
                scenario_override=scenario,
                controls=controls,
                resource_envelope=profile.get("resource_envelope"),
            )
        sample_record = {
            **sample.to_record(),
            "corpus_profile": name,
            "mandatory_workflow": profile["workflow"],
            **(
                {"resource_envelope": profile["resource_envelope"]}
                if "resource_envelope" in profile else {}
            ),
        }
        samples.append(sample_record)
        by_profile[name] = sample
    return _CorpusCampaign(
        selected=selected,
        samples=samples,
        by_profile=by_profile,
        missing_inputs=missing_inputs,
    )


def _corpus_workflow_results(
    profiles: dict[str, dict[str, Any]], mandatory_results: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    workflow_results = {}
    for workflow in MANDATORY_WORKFLOWS:
        names = sorted(
            name for name, profile in profiles.items() if profile["workflow"] == workflow
        )
        positive_names = [
            name
            for name in names
            if profiles[name]["expected_outcome"] == "passed"
            and profiles[name].get("full_workflow", True) is True
        ]
        case_names = [
            name
            for name in names
            if profiles[name]["expected_outcome"] == "passed"
            and profiles[name].get("full_workflow", True) is not True
        ]
        completed = [
            name
            for name in positive_names
            if mandatory_results[name]["positive_completion"] is True
        ]
        cases_certified = [
            name
            for name in case_names
            if mandatory_results[name]["case_certified"] is True
        ]
        refusals = [
            name for name in names if mandatory_results[name]["refusal_certified"] is True
        ]
        workflow_results[workflow] = {
            "completion": (
                "complete"
                if positive_names
                and len(completed) == len(positive_names)
                and len(cases_certified) == len(case_names)
                else "incomplete"
            ),
            "positive_profiles": positive_names,
            "positive_profiles_completed": completed,
            "routine_case_profiles": case_names,
            "routine_case_profiles_certified": cases_certified,
            "certified_refusal_profiles": refusals,
            "refusals_count_as_positive_completion": False,
            "qualification_limitations": {
                name: mandatory_results[name]["qualification_limitations"]
                for name in names
                if mandatory_results[name].get("qualification_limitations")
            },
        }
    return workflow_results


def _corpus_outcome_counts(
    statuses: list[str], mandatory_results: dict[str, dict[str, Any]]
) -> dict[str, int]:
    outcome_names = (
        "passed",
        "expected-refusal",
        "failed",
        "timeout",
        "missing-dependency",
        "ineligible-comparison",
    )
    counts = {status: statuses.count(status) for status in outcome_names}
    counts.update(
        {
            status: statuses.count(status)
            for status in sorted(set(statuses) - set(outcome_names))
        }
    )
    counts["positive_completed"] = sum(
        result["positive_completion"] is True for result in mandatory_results.values()
    )
    counts["case_certified"] = sum(
        result["case_certified"] is True for result in mandatory_results.values()
    )
    counts["refusal_certified"] = sum(
        result["refusal_certified"] is True for result in mandatory_results.values()
    )
    return counts


def qualify_native_corpus(args: QualificationControls) -> dict[str, object]:
    profiles = _corpus_profiles(args.resolution)
    campaign = _execute_corpus_profiles(args, profiles)
    selected, samples, by_profile, missing_inputs = (
        campaign.selected,
        campaign.samples,
        campaign.by_profile,
        campaign.missing_inputs,
    )
    statuses = [sample.status for sample in by_profile.values()]
    assessments = {
        name: _corpus_profile_assessment(profile, by_profile.get(name))
        for name, profile in profiles.items()
    }
    mandatory_results = {
        name: _mandatory_result(name, profiles[name], assessments[name])
        for name in sorted(profiles)
    }
    workflow_results = _corpus_workflow_results(profiles, mandatory_results)
    counts = _corpus_outcome_counts(statuses, mandatory_results)
    registration = load_corpus_registration()
    record: dict[str, object] = {
        "kind": "native-meshing-corpus-result",
        "scenario": "native-corpus",
        "runtime": _runtime_identity(),
        "corpus_registration_id": registration["registration_id"],
        "samples": samples,
        "counts": counts,
        "mandatory_workflow_results": mandatory_results,
        "mandatory_workflows": workflow_results,
        "mandatory_workflows_incomplete": [
            workflow
            for workflow, result in workflow_results.items()
            if result["completion"] != "complete"
        ],
        "selected_profiles": list(selected),
        "selected_profiles_missing_inputs": missing_inputs,
        "unqualified_selected_profiles": [
            name for name in selected if not assessments[name]["qualification_eligible"]
        ],
        "qualification_limitations": {
            name: assessment["qualification_limitations"]
            for name, assessment in assessments.items()
            if assessment["qualification_limitations"]
        },
        "qualification_dependencies": {
            name: {
                **profile,
                **assessments[name],
                "missing_original_source_inputs": missing_inputs.get(name, []),
                "prerequisites": [
                    "current native source/binary/configuration identity",
                    "all original root resource guarantees",
                    "independent geometry/semantics/physical/state acceptance",
                    "original capacity phase/scaling campaign",
                ],
            }
            for name, profile in profiles.items()
        },
        "reviewed_hybrid_layer_sources": registration["hybrid_layer_sources"],
        "preserved_source_incompatibilities": {
            "original-curved-layer-exact-period-x1": {
                "disposition": "immutable-negative-regression",
                "source_id": "curved-periodic-narrow-gap",
                "source_revision": (
                    "fadef789744332ba906c2bf1b82231d331e7cc7f4128997191b2d97cb9301e4f"
                ),
                "required_period_x": 1,
                "original_top_trace_x_residuals": (
                    "-101121/2361183241434822606848",
                    "-6620711477/77371252455336267181195264",
                ),
                "proof_origin": "tests/data/meshing/curved_periodic_sources.json",
                "reviewed_record": "tests/data/meshing/curved_periodic_sources.json",
                "evidence_scope": (
                    "Original mathematical source incompatibility; the corrected "
                    "positive owns a different source ID, digest, and revision."
                ),
                "rerun_profile": "original-curved-periodic-refusal",
            },
        },
        "research_gates": {
            "general-all-hex": "incomplete",
            "singular-cad-arrangements": "incomplete",
            "underspecified-defective-surface-reconstruction": "incomplete",
        },
        "release": "unassessed",
        "leadership": "unassessed",
    }
    return {**record, "result_id": canonical_fingerprint(record)}


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_native_corpus(args)
