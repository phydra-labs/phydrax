# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import resource
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, final, NoReturn

import equinox as eqx
import numpy as np

import phydrax as phx
from phydrax import canonical_fingerprint
from tools._meshing_cases import (
    corpus_identity,
    load_corpus_registration,
    MANDATORY_WORKFLOWS,
    meshing_case,
    POLYHEDRAL_CASE_NAMES,
    polyhedral_identity,
)
from tools.meshing_qualification._controls import (
    _NATIVE_DRIVER_ARGUMENTS,
    BenchmarkControls,
    QualificationControls,
    SCENARIO_WORKFLOWS,
)
from tools.meshing_qualification._records import (
    _corpus_profiles,
    _scenario_source_reference,
    _validate_mandatory_workflow_result,
    FailureRecord,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _runtime_identity,
    peak_resident_bytes,
    run_campaign_process,
)


@final
class CampaignSample(phx.StrictModule):
    """A decoded attempt; the interchange payload is retained only for serialization."""

    payload: dict[str, Any]
    status: str = eqx.field(static=True)
    positive_completion: bool = eqx.field(static=True)
    request_id: str | None = eqx.field(static=True)
    time_to_target_error_seconds: int | float | None = eqx.field(static=True)
    peak_process_resident_bytes: int | None = eqx.field(static=True)
    physical_error: int | float | None = eqx.field(static=True)
    failure_evidence_id: str | None = eqx.field(static=True)
    requested: dict[str, Any] | None
    achieved: dict[str, Any] | None
    source_bits_preserved: bool = eqx.field(static=True)
    epochs: tuple[dict[str, Any], ...]
    generation: dict[str, Any] | None
    checkpoint: dict[str, Any] | None
    independent_acceptance: dict[str, Any] | None
    independent_scientific_certification: str = eqx.field(static=True)
    mandatory_workflow_completion: str = eqx.field(static=True)
    case_contract_completion: str = eqx.field(static=True)
    release_evidence: str = eqx.field(static=True)
    leadership_evidence: str = eqx.field(static=True)
    phase_evidence: dict[str, Any]
    resource_evidence: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        """Return the attempt's unchanged JSON-edge representation."""
        return self.payload


def _optional_number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) else None


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _decode_campaign_sample(record: dict[str, Any]) -> CampaignSample:
    """Narrow dynamic child evidence once, without inventing absent measurements."""
    epochs = record.get("epochs", [])
    if not isinstance(epochs, (tuple, list)) or any(
        not isinstance(epoch, dict) for epoch in epochs
    ):
        raise TypeError(
            f"_decode_campaign_sample.epochs must be a sequence of mappings; got {epochs!r}."
        )
    completion = record["mandatory_workflow_completion"]
    peak = record.get("peak_process_resident_bytes")
    return CampaignSample(
        payload=record,
        status=record.get("status", "unmeasured"),
        positive_completion=record["positive_completion"],
        request_id=_optional_string(record.get("request_id")),
        time_to_target_error_seconds=_optional_number(
            record.get("time_to_target_error_seconds")
        ),
        peak_process_resident_bytes=peak if isinstance(peak, int) else None,
        physical_error=_optional_number(record.get("physical_error")),
        failure_evidence_id=_optional_string(record.get("failure_evidence_id")),
        requested=_optional_mapping(record.get("requested")),
        achieved=_optional_mapping(record.get("achieved")),
        source_bits_preserved=record.get("source_bits_preserved") is True,
        epochs=tuple(epochs),
        generation=_optional_mapping(record.get("generation")),
        checkpoint=_optional_mapping(record.get("checkpoint")),
        independent_acceptance=_optional_mapping(record["independent_acceptance"]),
        independent_scientific_certification=record.get(
            "independent_scientific_certification", "unassessed"
        ),
        mandatory_workflow_completion=completion,
        case_contract_completion=record.get("case_contract_completion", completion),
        release_evidence=record.get("release_evidence", "unassessed"),
        leadership_evidence=record.get("leadership_evidence", "unassessed"),
        phase_evidence=record["phase_evidence"],
        resource_evidence=record["resource_evidence"],
    )


def _reject_nonfinite_json(value: str, /) -> NoReturn:
    raise ValueError(f"_reject_nonfinite_json.value must be finite JSON; got {value!r}.")


def _unique_json_object(pairs: list[tuple[str, object]], /) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(
                f"_unique_json_object.name must occur once in the JSON object; got {name!r}."
            )
        result[name] = value
    return result


def _captured_text(value: object, /) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, str):
        return value
    raise TypeError(
        f"_captured_text.value must be text, bytes or None; got {type(value).__name__!r}."
    )


def _launcher_evidence(
    started: float,
    timeout: float,
    /,
    *,
    terminated: bool,
    returncode: int | None,
) -> dict[str, object]:

    peak = peak_resident_bytes(resource.RUSAGE_CHILDREN)
    return {
        "hard_timeout": {
            "limit_seconds": timeout,
            "mechanism": "session-process-group-kill",
            "completed_before_deadline": not terminated,
            "terminated_by_launcher": terminated,
            "returncode": returncode,
        },
        "resource_measurement": {
            "child_process_peak_resident_bytes": peak,
            "unit": "bytes",
            "scope": "launcher-lifetime-maximum-over-completed-children",
            "per_attempt_peak_available": False,
        },
        "process_total_seconds": perf_counter() - started,
    }


def _complete_campaign_sample(
    record: dict[str, Any],
    capacity: int,
    /,
    *,
    positive: bool,
) -> CampaignSample:
    """Attach the phase/resource evidence required on every attempted sample."""
    record.setdefault("runtime", None)
    record.setdefault("hard_timeout", None)
    record.setdefault("independent_acceptance", None)
    record.setdefault("mandatory_workflow_completion", "incomplete")
    record["positive_completion"] = positive
    phase_evidence = corpus_phase_evidence(record, capacity)
    record["phase_evidence"] = phase_evidence
    record["resource_evidence"] = phase_evidence["resources"]
    return _decode_campaign_sample(record)


def _qualification_positive_completion(record: dict[str, Any], engine: str, /) -> bool:
    """Accept only the child result, never a bare successful status."""
    if engine != "native":
        acceptance = record.get("independent_acceptance")
        return (
            record.get("status") == "passed"
            and record.get("case_contract_completion") == "complete"
            and record.get("independent_scientific_certification") == "passed"
            and isinstance(acceptance, dict)
            and acceptance.get("passed") is True
        )

    workflow = _validate_mandatory_workflow_result(record.get("workflow_result"))
    if workflow["observed_status"] != record.get("status"):
        raise ValueError(
            f"_qualification_positive_completion.observed_status must equal the qualification record status; got {workflow['observed_status']!r}."
        )
    if record.get("positive_completion") is not workflow["positive_completion"]:
        raise ValueError(
            f"_qualification_positive_completion.positive_completion must equal the workflow result completion; got {record.get('positive_completion')!r}."
        )
    return workflow["positive_completion"] is True


@final
class CampaignAttempt(phx.StrictModule):
    """Frozen process admission and launch controls for one isolated attempt."""

    command: list[str]
    frozen_controls: dict[str, Any]
    capacity: int = eqx.field(static=True)
    timeout: float = eqx.field(static=True)
    engine: str = eqx.field(static=True)
    ineligible: str | None = eqx.field(static=True)


def _prepare_campaign_attempt(
    case_name: str,
    resolution: int,
    capacity: int,
    timeout: float,
    repeats: int,
    target_error: float,
    adaptation_rounds: int,
    engine: str,
    *,
    scenario_override: str | None = None,
    controls: dict[str, str | int | float | tuple[float, ...]] | None = None,
    attempt_index: int = 0,
) -> CampaignAttempt:
    """Freeze source controls, attempt-local checkpoint destination, and eligibility."""
    if attempt_index < 0:
        raise ValueError(
            f"_prepare_campaign_attempt.attempt_index must be nonnegative; got {attempt_index!r}."
        )
    generic = scenario_override is not None
    surface = case_name == "sphere-feature-surface"
    polyhedral = case_name in POLYHEDRAL_CASE_NAMES
    scenario = (
        scenario_override
        if generic
        else "native-polyhedral-lifecycle"
        if polyhedral
        else "native-surface-pde"
        if surface
        else "native-lifecycle"
        if engine == "native"
        else "comparison-triangle-lifecycle"
    )
    effective_controls = dict(controls or {})
    checkpoint_root = effective_controls.get("distributed_checkpoint_root")
    if scenario == "native-distributed-lifecycle" and isinstance(checkpoint_root, str):
        effective_controls["distributed_checkpoint_root"] = (
            f"{checkpoint_root}-resolution-{resolution}-capacity-{capacity}"
            f"-attempt-{attempt_index}"
        )
    command = [
        sys.executable,
        "-m",
        "tools.meshing_qualification",
        "--scenario",
        scenario,
        "--corpus-case",
        "planar-hole-feature" if surface or polyhedral or generic else case_name,
        "--resolution",
        str(resolution),
        "--capacity",
        str(capacity),
        "--timeout",
        str(timeout),
        "--repeats",
        str(repeats),
        "--target-error",
        str(target_error),
        "--adaptation-rounds",
        str(adaptation_rounds),
    ]
    if polyhedral:
        command.extend(("--polyhedral-source", case_name))
    for name, value in sorted(effective_controls.items()):
        command.append(f"--{name.replace('_', '-')}")
        command.extend(
            str(item) for item in (value if isinstance(value, tuple) else (value,))
        )
    frozen_controls = {
        "case_name": case_name,
        "scenario": scenario,
        "engine": engine,
        "resolution": resolution,
        "entity_capacity": capacity,
        "wall_seconds": timeout,
        "solver_warm_repeats": repeats,
        "target_error": target_error,
        "adaptation_rounds": adaptation_rounds,
        "scenario_controls": {
            name: value for name, value in sorted(effective_controls.items())
        },
    }
    ineligible = (
        "No explicit baseline adapter matches this complete source/control/state contract."
        if generic and engine != "native"
        else "Triangle does not admit the nonconvex material polyhedral VEM/FV contract."
        if polyhedral and engine != "native"
        else "Triangle does not admit the source sphere/seam/pole/PDE contract."
        if surface and engine != "native"
        else None
    )
    return CampaignAttempt(
        command=command,
        frozen_controls=frozen_controls,
        capacity=capacity,
        timeout=timeout,
        engine=engine,
        ineligible=ineligible,
    )


def _execute_campaign_attempt(attempt: CampaignAttempt) -> CampaignSample:
    """Launch an admitted child and decode completion or cancellation evidence."""
    command = attempt.command
    frozen_controls = attempt.frozen_controls
    capacity = attempt.capacity
    timeout = attempt.timeout
    engine = attempt.engine
    ineligible = attempt.ineligible
    if ineligible is not None:
        return _complete_campaign_sample(
            {
                "status": "ineligible-comparison",
                "engine": engine,
                "command": command,
                "frozen_controls": frozen_controls,
                "frozen_controls_id": canonical_fingerprint(frozen_controls),
                "process_started": False,
                "failure": ineligible,
            },
            capacity,
            positive=False,
        )
    started = perf_counter()
    try:
        completed = run_campaign_process(
            command, cwd=Path(__file__).resolve().parents[2], timeout=timeout
        )
    except subprocess.TimeoutExpired as failure:
        launcher = _launcher_evidence(started, timeout, terminated=True, returncode=None)
        return _complete_campaign_sample(
            {
                "status": "timeout",
                "engine": engine,
                "command": command,
                "frozen_controls": frozen_controls,
                "frozen_controls_id": canonical_fingerprint(frozen_controls),
                "process_started": True,
                **FailureRecord.from_exception(failure, "campaign-launch").to_record(),
                "stdout": _captured_text(failure.stdout),
                "stderr": _captured_text(failure.stderr),
                **launcher,
            },
            capacity,
            positive=False,
        )
    launcher = _launcher_evidence(
        started, timeout, terminated=False, returncode=completed.returncode
    )
    try:
        record = json.loads(
            completed.stdout,
            parse_constant=_reject_nonfinite_json,
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, ValueError) as failure:
        return _complete_campaign_sample(
            {
                "status": "failed",
                "engine": engine,
                "command": command,
                "frozen_controls": frozen_controls,
                "frozen_controls_id": canonical_fingerprint(frozen_controls),
                "process_started": True,
                **FailureRecord.from_exception(
                    failure, "qualification-result-json"
                ).to_record(),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
                **launcher,
            },
            capacity,
            positive=False,
        )
    if not isinstance(record, dict):
        return _complete_campaign_sample(
            {
                "status": "failed",
                "engine": engine,
                "command": command,
                "frozen_controls": frozen_controls,
                "frozen_controls_id": canonical_fingerprint(frozen_controls),
                "process_started": True,
                **FailureRecord(
                    failure_phase="qualification-result-schema",
                    failure_type="TypeError",
                    failure="Qualification output is not an evidence object.",
                ).to_record(),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
                **launcher,
            },
            capacity,
            positive=False,
        )
    record.update(
        {
            "command": command,
            "returncode": completed.returncode,
            "stderr": completed.stderr,
            "frozen_controls": frozen_controls,
            "frozen_controls_id": canonical_fingerprint(frozen_controls),
            "process_started": True,
            **launcher,
        }
    )
    if completed.returncode != 0 and record.get("status") not in ("failed", "timeout"):
        record["status"] = "failed"
    try:
        positive = _qualification_positive_completion(record, engine)
    except (TypeError, ValueError) as failure:
        reported = record.get("status", "missing")
        record_failure(record, failure, "qualification-result-schema")
        record.update(
            {
                "status": "failed",
                "reported_status": reported,
                "failure": f"{type(failure).__name__}: {failure}",
            }
        )
        positive = False
    return _complete_campaign_sample(record, capacity, positive=positive)


def _native_campaign_sample(
    case_name: str,
    resolution: int,
    capacity: int,
    timeout: float,
    repeats: int,
    target_error: float,
    adaptation_rounds: int,
    engine: str,
    *,
    scenario_override: str | None = None,
    controls: dict[str, str | int | float | tuple[float, ...]] | None = None,
    attempt_index: int = 0,
) -> CampaignSample:
    """Measure one frozen attempt, including hard cancellation and failures."""
    attempt = _prepare_campaign_attempt(
        case_name,
        resolution,
        capacity,
        timeout,
        repeats,
        target_error,
        adaptation_rounds,
        engine,
        scenario_override=scenario_override,
        controls=controls,
        attempt_index=attempt_index,
    )
    return _execute_campaign_attempt(attempt)


def corpus_phase_evidence(sample: dict[str, Any], capacity: int) -> dict[str, Any]:
    """Expose phase measurements without deriving missing work or bytes."""
    epochs = sample.get("epochs", [])
    compiler_records = []
    stages = sample.get("stages_seconds", {})
    if not isinstance(stages, dict):
        raise TypeError(
            f"corpus_phase_evidence.stages must be a mapping; got {type(stages).__name__!r}."
        )
    for epoch in epochs:
        compiler = epoch.get("compiler", {})
        index = epoch["epoch"]
        compiler_records.append(
            {
                "epoch": index,
                "controlling_entity_capacity": capacity,
                "lowering_seconds": stages.get(f"solver_lowering_epoch_{index}"),
                "compile_seconds": stages.get(f"solver_compile_epoch_{index}"),
                "first_execution_seconds": stages.get(
                    f"solver_first_execution_epoch_{index}"
                ),
                "warm_samples_seconds": compiler.get("warm_samples_seconds"),
                "execution_scope": compiler.get("execution_scope"),
                "assembly_seconds": compiler.get("assembly_seconds"),
                "assembly_invocations": compiler.get("assembly_invocations"),
                "assembly_compilation_count": compiler.get("assembly_compilation_count"),
                "assembly_compile_seconds": compiler.get("assembly_compile_seconds"),
                "assembly_binding_id": compiler.get("assembly_binding_id"),
                "temporary_bytes": compiler.get("temporary_bytes"),
                "output_bytes": compiler.get("output_bytes"),
                "argument_bytes": compiler.get("argument_bytes"),
                "generated_code_bytes": compiler.get("generated_code_bytes"),
                "logical_retained_bytes": epoch.get("retained_bytes"),
                "memory_status": compiler.get("memory_status", "unmeasured"),
            }
        )
    measured_stages = [
        {
            "name": name,
            "seconds": seconds,
            "measurement": "wall-clock",
            "accounting": "owned-interval-not-assumed-exclusive",
        }
        for name, seconds in sorted(stages.items())
    ]
    native_execution = sample.get("native_phases")
    resources = {
        "hard_timeout": sample.get("hard_timeout"),
        "launcher_resource_measurement": sample.get("resource_measurement"),
        "process_peak_resident_bytes": sample.get("peak_process_resident_bytes"),
        "process_peak_scope": sample.get("process_peak_scope"),
        "device_memory": sample.get("device_memory"),
    }
    recorded = bool(
        measured_stages
        or compiler_records
        or native_execution
        or any(value is not None for value in resources.values())
    )
    return {
        "controlling_entity_capacity": capacity,
        "measured_stages": measured_stages,
        "stage_compilation_and_accounting": sample.get("stage_measurements"),
        "import_seconds": sample.get("import_seconds"),
        "compiler_epochs": compiler_records,
        "native_execution": native_execution,
        "resources": resources,
        "scope": (
            "Measured stage intervals, compiler analysis, logical retention, process peak, "
            "and backend device reports remain separate; inclusive intervals are not summed."
        ),
        "missing_measurements_are_zero": False,
        "status": "recorded" if recorded else "unmeasured",
    }


def _campaign_candidate_summary(
    samples: list[CampaignSample],
    outcome_names: tuple[str, ...],
    *,
    physics_error: bool,
) -> dict[str, object]:
    """Aggregate admitted completions without discarding failed attempts."""
    candidate_identities = [
        sample.request_id for sample in samples if sample.request_id is not None
    ]
    statuses = [sample.status for sample in samples]
    accepted = [sample for sample in samples if sample.positive_completion]
    times = [
        sample.time_to_target_error_seconds
        for sample in accepted
        if sample.time_to_target_error_seconds is not None
    ]
    peaks = [
        sample.peak_process_resident_bytes
        for sample in accepted
        if sample.peak_process_resident_bytes is not None
    ]
    counts = {status: statuses.count(status) for status in outcome_names}
    counts.update(
        {
            status: statuses.count(status)
            for status in sorted(set(statuses) - set(outcome_names))
        }
    )
    errors = []
    for sample in accepted:
        if sample.physical_error is None:
            raise KeyError("physical_error")
        errors.append(sample.physical_error)
    return {
        "samples": [sample.to_record() for sample in samples],
        "attempts": len(samples),
        "counts": counts,
        "positive_completions": len(accepted),
        "certified_refusals": statuses.count("expected-refusal"),
        "refusals_count_as_positive_completion": False,
        "success_rate": len(accepted) / len(samples),
        "request_identities": sorted(set(candidate_identities)),
        "request_identity_complete": len(candidate_identities) == len(samples),
        "time_to_target_error_seconds": {
            "median": None if not times else float(np.median(times)),
            "minimum": None if not times else float(np.min(times)),
            "maximum": None if not times else float(np.max(times)),
        },
        "peak_process_resident_bytes": None if not peaks else int(np.max(peaks)),
        (
            "worst_accepted_physics_error" if physics_error else "worst_accepted_l2_error"
        ): None if not accepted else max(errors),
        "leadership_eligible": False,
        "nonclaim": (
            "No leadership claim until the complete mandatory corpus and all "
            "equal construction-work/scratch/query guarantees are qualified."
        ),
    }


@final
class CampaignSource(phx.StrictModule):
    """Frozen source identity and its independently authored site count."""

    identity: dict[str, object]
    site_count: int = eqx.field(static=True)


def _campaign_source(
    case_name: str,
    resolution: int,
    scenario_override: str | None,
    controls: dict[str, str | int | float | tuple[float, ...]],
    source_reference: dict[str, object] | None,
) -> CampaignSource:
    """Resolve the source contract without using generated mesh output as identity."""
    polyhedral = case_name in POLYHEDRAL_CASE_NAMES
    if source_reference is not None:
        identity = source_reference
    else:
        identity = (
            {
                **_scenario_source_reference(scenario_override, controls),
                "scenario": scenario_override,
                "frozen_scenario_controls": controls,
            }
            if scenario_override is not None
            else polyhedral_identity(case_name)
            if polyhedral
            else corpus_identity(meshing_case(case_name))
        )
    site_count = resolution
    if polyhedral and "sites" in identity:
        sites = identity["sites"]
        if not isinstance(sites, list):
            raise TypeError(
                f"_campaign_source.sites must be a frozen list; got {type(sites).__name__!r}."
            )
        site_count = len(sites)
    return CampaignSource(identity=identity, site_count=site_count)


def _optional_campaign_baselines(
    polyhedral: bool, outcome_names: tuple[str, ...]
) -> dict[str, object]:
    """Record baseline admission separately from attempted campaign samples."""
    if not polyhedral:
        return {}
    available = shutil.which("vc_mesh") is not None
    status = "ineligible-comparison" if available else "missing-dependency"
    return {
        "vorocrust": {
            "status": status,
            "attempts": 0,
            "counts": {name: int(name == status) for name in outcome_names},
            "dependency": "vc_mesh",
            "detail": (
                "Installed VoroCrust has no qualified adapter matching prescribed sites, "
                "two-material organization, VEM/FV and consumed conservative state."
                if available
                else "Optional VoroCrust executable is not installed."
            ),
            "leadership_eligible": False,
        }
    }


def benchmark_native_lifecycle(
    case_name: str,
    resolution: int,
    capacity: int,
    timeout: float,
    repeats: int,
    target_error: float,
    adaptation_rounds: int,
    comparison: str | None,
    *,
    benchmark_warm_repeats: int = 5,
    scenario_override: str | None = None,
    controls: dict[str, str | int | float | tuple[float, ...]] | None = None,
    source_reference: dict[str, object] | None = None,
) -> dict[str, object]:
    """Freeze a common physical contract and retain every candidate's sample."""

    polyhedral = case_name in POLYHEDRAL_CASE_NAMES
    engines = ("native",) if comparison is None else ("native", comparison)
    candidates = {}
    identities: set[str] = set()
    request_identity_complete = True
    outcome_names = (
        "passed",
        "expected-refusal",
        "failed",
        "timeout",
        "missing-dependency",
        "ineligible-comparison",
    )
    for engine in engines:
        samples = [
            _native_campaign_sample(
                case_name,
                resolution,
                capacity,
                timeout,
                benchmark_warm_repeats,
                target_error,
                adaptation_rounds,
                engine,
                scenario_override=scenario_override,
                controls=controls,
                attempt_index=attempt,
            )
            for attempt in range(repeats)
        ]
        for sample in samples:
            if sample.request_id is not None:
                identities.add(sample.request_id)
            else:
                request_identity_complete = False
        candidates[engine] = _campaign_candidate_summary(
            samples,
            outcome_names,
            physics_error=polyhedral or scenario_override is not None,
        )
    optional_baselines = _optional_campaign_baselines(polyhedral, outcome_names)
    normalized_controls = {
        name: value for name, value in sorted((controls or {}).items())
    }
    source = _campaign_source(
        case_name, resolution, scenario_override, normalized_controls, source_reference
    )
    frozen_controls: dict[str, object] = {
        "resolution": resolution,
        "entity_capacity": capacity,
        "campaign_repeats": repeats,
        "qualification_solver_warm_repeats": benchmark_warm_repeats,
        "adaptation_rounds": adaptation_rounds,
        "timeout_seconds": timeout,
        "target_error": target_error,
        "comparison": comparison,
        "scenario_controls": normalized_controls,
    }
    target_name = (
        "physical_error"
        if scenario_override is not None
        else "maximum_physics_error"
        if polyhedral
        else "l2_error"
    )
    body: dict[str, object] = {
        "kind": "native-meshing-benchmark-campaign",
        "case": source.identity,
        "runtime": _runtime_identity(),
        "frozen_controls": frozen_controls,
        "frozen_controls_id": canonical_fingerprint(frozen_controls),
        "control_capacities": (
            {
                "requested_resolution": resolution,
                "site_count": source.site_count,
                "entity_capacity": capacity,
                "adaptation_claimed": False,
            }
            if polyhedral
            else {
                "resolution": resolution,
                "entity_capacity": capacity,
                "adaptation_rounds": adaptation_rounds,
            }
        ),
        "targets": {target_name: target_error, "wall_seconds": timeout},
        "request_identities": sorted(identities),
        "request_identity_complete": request_identity_complete,
        "matching_request_identity": request_identity_complete and len(identities) == 1,
        "candidates": candidates,
        "comparison_scope": (
            "Frozen native source/control/state/solver lifecycle; no exact baseline adapter admitted."
            if scenario_override is not None
            else "Restricted power generation, VEM/FV, conservative material remap and consumed FV update."
            if polyhedral
            else "Generation engines; identical native bisection/transfer/solver consumer."
        ),
        "optional_baselines": optional_baselines,
        "mandatory_workflow_completion": "incomplete",
        "hard_size_claim": False,
        "release": "unassessed",
        "leadership": "unassessed",
    }
    return {**body, "campaign_id": canonical_fingerprint(body)}


_CAMPAIGN_RESULT_FIELDS = (
    "runtime",
    "frozen_controls",
    "hard_timeout",
    "resource_evidence",
    "phase_evidence",
    "independent_acceptance",
    "positive_completion",
    "mandatory_workflow_completion",
)


_CAMPAIGN_PHASE_SECTIONS = (
    "source_decode_and_geometry_preparation",
    "constraint_metric_and_reusable_plan_preparation",
    "generation_classification_refinement_improvement",
    "projection_curving_and_independent_certification",
    "publication_and_state_transfer",
    "lowering_compilation_first_and_warmed_execution",
    "compiler_process_device_and_logical_memory",
    "partition_communication_checkpoint_restart",
    "solver_time_and_memory_to_physical_target",
)


NATIVE_CAMPAIGN_REGISTRATION_PATH = (
    Path(__file__).resolve().parents[2] / "tests/data/meshing/campaigns.json"
)


def campaign_registration_record() -> dict[str, object]:
    """Return the deterministic mandatory campaign matrix from qualification routes."""

    profiles = _corpus_profiles()
    workflows = []
    for workflow in MANDATORY_WORKFLOWS:
        names = sorted(
            name
            for name, profile in profiles.items()
            if profile["workflow"] == workflow and profile["expected_outcome"] == "passed"
        )
        scenarios = {
            scenario
            for scenario, owner in SCENARIO_WORKFLOWS.items()
            if owner == workflow
        }
        for name in names:
            profile = profiles[name]
            scenarios.add(
                profile["scenario"]
                if profile["scenario"] is not None
                else "native-polyhedral-lifecycle"
                if name in POLYHEDRAL_CASE_NAMES
                else "native-lifecycle"
            )
        workflows.append(
            {
                "workflow": workflow,
                "expected_admission": "positive",
                "refusal_counts_as_positive_completion": False,
                "profiles": names,
                "scenarios": sorted(scenarios),
            }
        )
    body: dict[str, object] = {
        "kind": "native-meshing-campaign-registration",
        "license": "LicenseRef-PHYDRA-Proprietary",
        "provenance": {
            "owner": "tools/meshing_qualification/_campaign.py::campaign_registration_record",
            "origin": "Deterministic qualification workflow, scenario, and profile tables",
        },
        "contains_benchmark_results": False,
        "expected_artifacts": [
            "benchmarks/native_meshing_qualification.json",
            "benchmarks/native_meshing_benchmarks.json",
        ],
        "required_result_fields": list(_CAMPAIGN_RESULT_FIELDS),
        "phase_sections": list(_CAMPAIGN_PHASE_SECTIONS),
        "source_corpus_registration_id": load_corpus_registration()["registration_id"],
        "workflows": workflows,
    }
    return {**body, "registration_id": canonical_fingerprint(body)}


def load_native_campaign_registration(
    path: Path = NATIVE_CAMPAIGN_REGISTRATION_PATH,
) -> dict[str, object]:
    """Load the reviewed mandatory campaign matrix and verify its identity."""
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(value, dict):
        raise TypeError(
            f"load_native_campaign_registration.value must be a JSON object; got {type(value).__name__!r}."
        )
    if value != campaign_registration_record():
        raise ValueError(
            f"load_native_campaign_registration.value must equal campaign_registration_record(); got {value!r}."
        )
    return value


def _native_driver_controls(
    name: str,
    args: BenchmarkControls | QualificationControls,
) -> dict[str, str | int | float | tuple[float, ...]]:

    values: dict[str, str | int | float | tuple[float, ...] | None] = {
        "image_size": args.image_size,
        "transport_steps": args.transport_steps,
        "transport_dt": args.transport_dt,
        "balance_tolerance": args.balance_tolerance,
        "oracle_capacity": args.oracle_capacity,
        "cad_format": args.cad_format,
        "periodic_case": args.periodic_case,
        "maximum_memory_bytes": args.maximum_memory_bytes,
        "maximum_condition": args.maximum_condition,
        "family_case": args.family_case,
        "surface_case": args.surface_case,
        "layer_count": args.layer_count,
        "first_thickness": args.first_thickness,
        "layer_growth": args.layer_growth,
        "geometry_order": args.geometry_order,
        "hybrid_profile": args.hybrid_profile,
        "hybrid_archive": args.hybrid_archive,
        "hybrid_content_id": args.hybrid_content_id,
        "hybrid_core_size": args.hybrid_core_size,
        "hybrid_region_densities": args.hybrid_region_densities,
        "hybrid_material_site": args.hybrid_material_site,
        "implicit_radius": args.implicit_radius,
        "implicit_fidelity": args.implicit_fidelity,
        "implicit_maximum_level": args.implicit_maximum_level,
        "implicit_edge_isolation_depth": args.implicit_edge_isolation_depth,
        "implicit_root_tolerance": args.implicit_root_tolerance,
        "implicit_refinement_rounds": args.implicit_refinement_rounds,
        "implicit_improvement_passes": args.implicit_improvement_passes,
        "implicit_metric_passes": args.implicit_metric_passes,
        "distributed_parts": args.distributed_parts,
        "restart_parts": args.restart_parts,
        "distributed_processes": args.distributed_processes,
        "distributed_source": args.distributed_source,
        "distributed_halo_width": args.distributed_halo_width,
        "distributed_halo_capacity": args.distributed_halo_capacity,
        "distributed_cavity_capacity": args.distributed_cavity_capacity,
        "distributed_slack": args.distributed_slack,
        "distributed_checkpoint_root": args.distributed_checkpoint_root,
        "distributed_bisection_compatibility": args.distributed_bisection_compatibility.value,
        "distributed_host_packet_bytes": args.distributed_host_packet_bytes,
        "overset_motion_steps": args.overset_motion_steps,
        "overset_wall_pairs": args.overset_wall_pairs,
        "overset_donor_pairs": args.overset_donor_pairs,
        "overset_overlap_pairs": args.overset_overlap_pairs,
    }
    result: dict[str, str | int | float | tuple[float, ...]] = {}
    for key in _NATIVE_DRIVER_ARGUMENTS[name]:
        value = values[key]
        if value is not None:
            result[key] = value
    return result


def _benchmark_registered_native_corpus(
    args: BenchmarkControls,
    profiles: dict[str, dict[str, Any]],
    /,
) -> dict[str, object]:
    """Execute the deterministic registered profile matrix without omitting gaps."""
    inputs = {
        "hybrid_archive": args.hybrid_archive,
        "hybrid_content_id": args.hybrid_content_id,
        "hybrid_core_size": args.hybrid_core_size,
        "hybrid_region_densities": args.hybrid_region_densities,
        "hybrid_material_site": args.hybrid_material_site,
        "distributed_checkpoint_root": args.distributed_checkpoint_root,
    }
    requested = args.corpus_profile
    selected = tuple(profiles) if requested is None else tuple(requested)
    unknown = sorted(set(selected) - set(profiles))
    if unknown or len(selected) != len(set(selected)):
        raise ValueError(
            f"_benchmark_registered_native_corpus.selected must contain unique registered benchmark profiles; got {selected!r}."
        )
    records: dict[str, object] = {}
    missing_inputs: dict[str, list[str]] = {}
    for name in selected:
        profile = profiles[name]
        scenario = profile["scenario"]
        case_name = profile["case_name"]
        workflow = profile["workflow"]
        source_reference = profile["source_reference"]
        profile_controls = profile["controls"]
        prerequisites = profile.get("prerequisites", ())
        required = list(prerequisites)
        if scenario == "native-distributed-lifecycle":
            required.append("distributed_checkpoint_root")
        missing = sorted({key for key in required if inputs.get(key) is None})
        if missing:
            missing_inputs[name] = missing
            body: dict[str, object] = {
                "kind": "native-meshing-benchmark-profile",
                "profile": name,
                "workflow": workflow,
                "scenario": scenario,
                "status": "blocked-missing-input",
                "missing_inputs": missing,
                "source_reference_id": canonical_fingerprint(source_reference),
                "campaigns": [],
            }
            records[name] = {**body, "profile_result_id": canonical_fingerprint(body)}
            continue
        controls = {} if scenario is None else _native_driver_controls(scenario, args)
        controls.update(profile_controls)
        rounds = (
            0
            if case_name in POLYHEDRAL_CASE_NAMES
            and case_name != "polyhedral-periodic-cube"
            else args.adaptation_rounds
        )
        campaigns = [
            benchmark_native_lifecycle(
                case_name,
                resolution,
                capacity,
                args.timeout,
                args.repeats,
                args.target_error,
                rounds,
                args.comparison,
                benchmark_warm_repeats=args.benchmark_warm_repeats,
                scenario_override=scenario,
                controls=controls,
                source_reference=source_reference,
            )
            for resolution in args.resolution
            for capacity in args.capacity
        ]
        body = {
            "kind": "native-meshing-benchmark-profile",
            "profile": name,
            "workflow": workflow,
            "scenario": scenario,
            "status": "attempted",
            "missing_inputs": [],
            "source_reference_id": canonical_fingerprint(source_reference),
            "campaigns": campaigns,
        }
        records[name] = {**body, "profile_result_id": canonical_fingerprint(body)}
    body = {
        "kind": "registered-native-meshing-benchmark-corpus",
        "selected_profiles": list(selected),
        "missing_profile_inputs": missing_inputs,
        "profiles": records,
    }
    return {**body, "corpus_benchmark_id": canonical_fingerprint(body)}
