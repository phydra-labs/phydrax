# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import replace
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter
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
    _hybrid_material_controls,
    _NATIVE_DRIVER_ARGUMENTS,
    BenchmarkControls,
    qualification_timeout,
    QualificationControls,
    SCENARIO_WORKFLOWS,
)
from tools.meshing_qualification._records import (
    _bound_workflow_verdict,
    _corpus_profiles,
    _corpus_request_dimensions,
    _scenario_source_reference,
    FailureRecord,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _runtime_identity,
    CampaignProcessTimeout,
    Deadline,
    epoch_stage_name,
    PEAK_RESIDENT_SCOPE,
    REPORTING_GRACE_SECONDS,
    run_campaign_process,
)


@final
class CampaignSample(phx.StrictModule):
    """A decoded attempt; the interchange payload is retained only for serialization."""

    payload: dict[str, Any]
    status: str = eqx.field(static=True)
    positive_completion: bool = eqx.field(static=True)
    accepted_case: bool = eqx.field(static=True)
    request_id: str | None = eqx.field(static=True)
    time_to_target_error_seconds: int | float | None = eqx.field(static=True)
    end_to_end_seconds: int | float | None = eqx.field(static=True)
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
    binding: dict[str, str | None]

    def to_record(self) -> dict[str, Any]:
        """Return the attempt's unchanged JSON-edge representation."""
        return self.payload


def _optional_number(value: Any) -> int | float | None:
    return (
        value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    )


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
        accepted_case=record["accepted_case"],
        request_id=_optional_string(record.get("request_id")),
        time_to_target_error_seconds=_optional_number(
            record.get("time_to_target_error_seconds")
        ),
        end_to_end_seconds=_optional_number(record.get("end_to_end_seconds")),
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
        binding=record["sample_binding"],
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


def _launcher_evidence(
    started: float,
    timeout: float,
    /,
    *,
    exited_after_deadline: bool,
    terminated: bool,
    returncode: int | None,
) -> dict[str, object]:
    total = perf_counter() - started
    return {
        "hard_timeout": {
            "limit_seconds": timeout,
            "deadline_origin": "launcher-process-launch",
            "child_deadline": "inherited-launch-expiry-enforced-in-process",
            "reporting_grace_seconds": REPORTING_GRACE_SECONDS,
            "watchdog_seconds": timeout + REPORTING_GRACE_SECONDS,
            "mechanism": "session-process-group-kill",
            "completed_before_deadline": not terminated and not exited_after_deadline,
            "terminated_by_launcher": terminated,
            "returncode": returncode,
        },
        "process_total_seconds": total,
        "reporting_overhead_seconds": None,
        "reporting_grace_seconds_used": None if terminated else max(0.0, total - timeout),
    }


def _sample_binding(
    record: dict[str, Any], verdict: dict[str, object] | None, /
) -> dict[str, str | None]:
    """Identify the launched controls and, when validated, the bound child verdict."""

    def identifier(name: str) -> str | None:
        value = None if verdict is None else verdict[name]
        return value if isinstance(value, str) else None

    return {
        "frozen_controls_id": record.get("frozen_controls_id"),
        "workflow_result_id": identifier("result_id"),
        "evidence_id": identifier("evidence_id"),
        "controls_id": identifier("controls_id"),
        "request_id": identifier("request_id"),
        "runtime_id": identifier("runtime_id"),
    }


def _complete_campaign_sample(
    record: dict[str, Any],
    capacity: int,
    /,
    *,
    positive: bool,
    accepted: bool,
    verdict: dict[str, object] | None = None,
) -> CampaignSample:
    """Attach the phase/resource evidence required on every attempted sample."""
    record.setdefault("runtime", None)
    record.setdefault("hard_timeout", None)
    record.setdefault("independent_acceptance", None)
    record.setdefault("mandatory_workflow_completion", "incomplete")
    record["positive_completion"] = positive
    record["accepted_case"] = accepted
    record["sample_binding"] = _sample_binding(record, verdict)
    phase_evidence = corpus_phase_evidence(record, capacity)
    record["phase_evidence"] = phase_evidence
    record["resource_evidence"] = phase_evidence["resources"]
    return _decode_campaign_sample(record)


@cache
def _launcher_build_id() -> str:
    """The build this launcher loaded; every bound child verdict must share it."""
    build_id = _runtime_identity()["build_id"]
    if not isinstance(build_id, str):
        raise TypeError(
            f"_launcher_build_id.build_id must be a string; got {type(build_id).__name__!r}."
        )
    return build_id


def _accepted_comparison_case(record: dict[str, Any], /) -> bool:
    """Accept a comparison-engine case; baselines own no mandatory native workflow."""
    acceptance = record.get("independent_acceptance")
    return (
        record.get("status") == "passed"
        and record.get("case_contract_completion") == "complete"
        and record.get("independent_scientific_certification") == "passed"
        and isinstance(acceptance, dict)
        and acceptance.get("passed") is True
    )


@final
class CampaignAttempt(phx.StrictModule):
    """Frozen process admission and launch controls for one isolated attempt."""

    command: list[str]
    frozen_controls: dict[str, Any]
    expected_controls: dict[str, Any]
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
    resource_envelope: dict[str, Any] | None = None,
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
    timeout = qualification_timeout(effective_controls, timeout)
    if scenario == "native-hybrid-lifecycle":
        defaults = QualificationControls()
        profile = effective_controls.get("hybrid_profile", defaults.hybrid_profile)
        densities = effective_controls.get("hybrid_region_densities")
        material_site = effective_controls.get("hybrid_material_site")
        if (
            not isinstance(profile, str)
            or (densities is not None and not isinstance(densities, tuple))
            or (material_site is not None and not isinstance(material_site, str))
        ):
            raise TypeError(
                "Hybrid campaign material controls must retain their declared kinds."
            )
        resolved_densities, resolved_site = _hybrid_material_controls(
            profile, densities, material_site
        )
        effective_controls.update(
            hybrid_profile=profile,
            hybrid_region_densities=resolved_densities,
            hybrid_material_site=resolved_site,
        )
    checkpoint_root = effective_controls.get("distributed_checkpoint_root")
    if scenario == "native-distributed-lifecycle" and isinstance(checkpoint_root, str):
        effective_controls["distributed_checkpoint_root"] = (
            f"{checkpoint_root}-resolution-{resolution}-capacity-{capacity}"
            f"-attempt-{attempt_index}"
        )
    corpus_case = "planar-hole-feature" if surface or polyhedral or generic else case_name
    command = [
        sys.executable,
        "-m",
        "tools.meshing_qualification",
        "--scenario",
        scenario,
        "--corpus-case",
        corpus_case,
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
        if name == "timeout":
            continue
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
    if resource_envelope is not None:
        frozen_controls["resource_envelope"] = resource_envelope
    # The child echoes its parsed controls; its verdict binds them to these values.
    expected_controls: dict[str, Any] = {
        "scenario": scenario,
        "corpus_case": corpus_case,
        "resolution": resolution,
        "capacity": capacity,
        "timeout": timeout,
        "repeats": repeats,
        "target_error": target_error,
        "adaptation_rounds": adaptation_rounds,
        **({"polyhedral_source": case_name} if polyhedral else {}),
        **effective_controls,
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
        expected_controls=expected_controls,
        capacity=capacity,
        timeout=timeout,
        engine=engine,
        ineligible=ineligible,
    )


def _execute_campaign_attempt(
    attempt: CampaignAttempt, *, deadline: Deadline | None = None
) -> CampaignSample:
    """Launch an admitted child and decode completion or cancellation evidence."""
    command = attempt.command
    frozen_controls = attempt.frozen_controls
    capacity = attempt.capacity
    timeout = attempt.timeout
    engine = attempt.engine
    ineligible = attempt.ineligible
    launch = {
        "engine": engine,
        "command": command,
        "frozen_controls": frozen_controls,
        "frozen_controls_id": canonical_fingerprint(frozen_controls),
    }
    if ineligible is not None:
        return _complete_campaign_sample(
            {
                "status": "ineligible-comparison",
                **launch,
                "process_started": False,
                "failure": ineligible,
            },
            capacity,
            positive=False,
            accepted=False,
        )
    started = perf_counter()
    launch_timeout = timeout if deadline is None else min(timeout, deadline.remaining())
    try:
        completed = run_campaign_process(
            command, cwd=Path(__file__).resolve().parents[2], timeout=launch_timeout
        )
    except CampaignProcessTimeout as failure:
        return _complete_campaign_sample(
            {
                "status": "timeout",
                **launch,
                "process_started": True,
                **FailureRecord.from_exception(failure, "campaign-launch").to_record(),
                "stdout": failure.stdout,
                "stderr": failure.stderr,
                "peak_process_resident_bytes": failure.peak_resident_bytes,
                "process_peak_scope": (
                    f"launcher-read-before-kill;{PEAK_RESIDENT_SCOPE}"
                    if failure.peak_resident_bytes is not None
                    else "unavailable:child-high-water-unreadable-before-kill"
                ),
                **_launcher_evidence(
                    started,
                    launch_timeout,
                    exited_after_deadline=True,
                    terminated=True,
                    returncode=None,
                ),
            },
            capacity,
            positive=False,
            accepted=False,
        )
    exited_at = monotonic()
    launcher = _launcher_evidence(
        started,
        launch_timeout,
        exited_after_deadline=completed.deadline_expired,
        terminated=False,
        returncode=completed.returncode,
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
                "status": "timeout" if completed.deadline_expired else "failed",
                **launch,
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
            accepted=False,
        )
    if not isinstance(record, dict):
        return _complete_campaign_sample(
            {
                "status": "timeout" if completed.deadline_expired else "failed",
                **launch,
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
            accepted=False,
        )
    # Bind the verdict to the child's own emitted record before launcher fields join it.
    verdict: dict[str, object] | None = None
    try:
        if engine == "native":
            verdict = _bound_workflow_verdict(
                record, attempt.expected_controls, _launcher_build_id()
            )
            positive = verdict["positive_completion"] is True
            accepted = verdict["case_accepted"] is True
        else:
            positive, accepted = False, _accepted_comparison_case(record)
        schema_failure = None
    except (KeyError, TypeError, ValueError) as failure:
        verdict, positive, accepted, schema_failure = None, False, False, failure
    policy = record.get("deadline_policy")
    science_end = _optional_number(
        policy.get("science_returned_monotonic_seconds")
        if isinstance(policy, dict)
        else None
    )
    if (
        science_end is not None
        and math.isfinite(science_end)
        and science_end <= exited_at
    ):
        launcher["reporting_overhead_seconds"] = exited_at - science_end
    # Keep the exact emitted evidence available for independent verdict revalidation.
    # Launcher and prerequisite fields belong to the campaign, not that child verdict.
    record = {**record, "child_evidence": record}
    record.pop("workflow_result", None)
    record.update(
        {
            "command": command,
            "returncode": completed.returncode,
            "stderr": completed.stderr,
            "frozen_controls": frozen_controls,
            "frozen_controls_id": launch["frozen_controls_id"],
            "process_started": True,
            **launcher,
        }
    )
    if schema_failure is not None:
        reported = record.get("status", "missing")
        record_failure(record, schema_failure, "qualification-result-schema")
        record.update(
            {
                "status": "failed",
                "reported_status": reported,
                "failure": f"{type(schema_failure).__name__}: {schema_failure}",
            }
        )
    elif completed.returncode != 0 and record.get("status") not in ("failed", "timeout"):
        record["status"] = "failed"
        positive = accepted = False
    policy = record.get("deadline_policy")
    science_in_time = (
        isinstance(policy, dict)
        and policy.get("origin") == "campaign-launch"
        and policy.get("science_returned_before_deadline") is True
    )
    if completed.deadline_expired and not science_in_time:
        # The child exited inside the reporting grace without proving its science
        # ended before the scientific deadline; the grace never adds science time.
        record["launcher_deadline_classification"] = {
            "exited_after_scientific_deadline": True,
            "status_before_classification": record.get("status"),
        }
        record["status"] = "timeout"
        positive = accepted = False
    return _complete_campaign_sample(
        record, capacity, positive=positive, accepted=accepted, verdict=verdict
    )


def _missing_profile_inputs(
    profile: Mapping[str, Any], inputs: Mapping[str, object], /
) -> list[str]:
    """Distinguish absent automatic artifacts from partial caller-owned inputs."""
    missing = {
        name for name in profile.get("prerequisites", ()) if inputs.get(name) is None
    }
    if (
        profile["scenario"] == "native-hybrid-lifecycle"
        and profile["controls"].get("hybrid_profile") == "archived-layer-core"
        and inputs.get("hybrid_archive") is None
        and inputs.get("hybrid_content_id") is None
    ):
        missing.difference_update(("hybrid_archive", "hybrid_content_id"))
    if profile["scenario"] == "native-distributed-lifecycle":
        missing.discard("distributed_checkpoint_root")
    return sorted(missing)


def _bind_generated_controls(
    attempt: CampaignAttempt, controls: Mapping[str, str], /
) -> CampaignAttempt:
    """Bind generated artifact handles to both the command and its frozen admission."""
    existing = attempt.frozen_controls["scenario_controls"]
    if set(controls) & existing.keys():
        raise ValueError(
            "Generated campaign controls cannot replace caller-owned inputs."
        )
    command = list(attempt.command)
    for name, value in sorted(controls.items()):
        command.extend((f"--{name.replace('_', '-')}", value))
    return replace(
        attempt,
        command=command,
        frozen_controls={
            **attempt.frozen_controls,
            "scenario_controls": {**existing, **controls},
        },
        expected_controls={**attempt.expected_controls, **controls},
    )


def _produce_hybrid_prerequisite(
    destination: Path, attempt: CampaignAttempt, deadline: Deadline, /
) -> dict[str, Any]:
    """Run the real native producer and retain its failure or authenticated receipt."""
    timeout = deadline.remaining()
    controls = attempt.frozen_controls["scenario_controls"]
    original = QualificationControls()
    producer_controls = {
        "capacity": attempt.capacity,
        "resolution": attempt.expected_controls["resolution"],
        "rounds": attempt.frozen_controls["adaptation_rounds"],
        "solver_repeats": attempt.frozen_controls["solver_warm_repeats"],
        "hybrid_region_densities": list(controls["hybrid_region_densities"]),
        "hybrid_material_site": controls["hybrid_material_site"],
        "hybrid_core_size": controls.get("hybrid_core_size", original.hybrid_core_size),
    }
    command = [
        sys.executable,
        "-m",
        "tools.layer_core_lifecycle_production",
        str(destination),
        "--timeout",
        str(attempt.timeout),
    ]
    for name, value in sorted(producer_controls.items()):
        command.append(f"--{name.replace('_', '-')}")
        command.extend(
            str(item) for item in (value if isinstance(value, list) else (value,))
        )
    started = perf_counter()
    evidence: dict[str, Any] = {"command": command, "process_started": True}
    try:
        completed = run_campaign_process(
            command, cwd=Path(__file__).resolve().parents[2], timeout=timeout
        )
    except CampaignProcessTimeout as failure:
        return {
            **evidence,
            "status": "timeout",
            **FailureRecord.from_exception(failure, "hybrid-native-producer").to_record(),
            "stdout": failure.stdout,
            "stderr": failure.stderr,
            "peak_process_resident_bytes": failure.peak_resident_bytes,
            "process_total_seconds": perf_counter() - started,
        }
    evidence.update(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        deadline_expired=completed.deadline_expired,
        process_total_seconds=perf_counter() - started,
    )
    if completed.returncode:
        return {
            **evidence,
            "status": "timeout" if completed.deadline_expired else "failed",
        }
    try:
        receipt = json.loads(
            completed.stdout,
            parse_constant=_reject_nonfinite_json,
            object_pairs_hook=_unique_json_object,
        )
        if not isinstance(receipt, dict):
            raise TypeError("The native hybrid producer must emit a receipt mapping.")
        archive = receipt.get("archive")
        content_id = receipt.get("content_id")
        producer = receipt.get("producer")
        if (
            not isinstance(archive, str)
            or Path(archive).resolve() != (destination / "lifecycle").resolve()
            or not isinstance(content_id, str)
            or not content_id
            or not isinstance(producer, dict)
        ):
            raise ValueError(
                "The native hybrid receipt must bind its published artifact."
            )
        for name in (
            "source_id",
            "source_revision",
            "source_binding_id",
            "specification_id",
            "limits_id",
        ):
            if type(receipt.get(name)) is not str or not receipt[name]:
                raise ValueError(
                    f"The native hybrid receipt must retain its authored {name}."
                )
        process_id = producer.get("process_id")
        runtime = producer.get("runtime")
        if (
            type(process_id) is not int
            or process_id <= 0
            or process_id == os.getpid()
            or not isinstance(runtime, dict)
            or runtime.get("build_id") != _launcher_build_id()
            or producer.get("science_returned_before_deadline") is not True
            or producer.get("controls") != producer_controls
        ):
            raise ValueError(
                "The native hybrid producer must bind this build and deadline."
            )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as failure:
        return {
            **evidence,
            "status": "failed",
            **FailureRecord.from_exception(
                failure, "hybrid-producer-receipt"
            ).to_record(),
        }
    return {**evidence, "status": "passed", "receipt": receipt}


def _execute_generated_hybrid_attempt(attempt: CampaignAttempt) -> CampaignSample:
    """Generate and cold-consume the original source under one workflow deadline."""
    started = perf_counter()
    clock_started = monotonic()
    deadline = Deadline(started + attempt.timeout)
    with TemporaryDirectory(prefix="phydrax-campaign-layer-core-") as directory:
        prerequisite = _produce_hybrid_prerequisite(Path(directory), attempt, deadline)
        if prerequisite["status"] != "passed":
            return _complete_campaign_sample(
                {
                    "status": prerequisite["status"],
                    "engine": attempt.engine,
                    "frozen_controls": attempt.frozen_controls,
                    "frozen_controls_id": canonical_fingerprint(attempt.frozen_controls),
                    "prerequisite": prerequisite,
                    "failure_phase": "hybrid-native-producer",
                    "end_to_end_seconds": perf_counter() - started,
                },
                attempt.capacity,
                positive=False,
                accepted=False,
            )
        receipt = prerequisite["receipt"]
        consumer_attempt = _bind_generated_controls(
            attempt,
            {
                "hybrid_archive": receipt["archive"],
                "hybrid_content_id": receipt["content_id"],
            },
        )
        try:
            sample = _execute_campaign_attempt(consumer_attempt, deadline=deadline)
        except TimeoutError as failure:
            return _complete_campaign_sample(
                {
                    "status": "timeout",
                    "engine": attempt.engine,
                    "frozen_controls": attempt.frozen_controls,
                    "frozen_controls_id": canonical_fingerprint(attempt.frozen_controls),
                    "prerequisite": prerequisite,
                    **FailureRecord.from_exception(
                        failure, "hybrid-consumer-launch"
                    ).to_record(),
                    "end_to_end_seconds": perf_counter() - started,
                },
                attempt.capacity,
                positive=False,
                accepted=False,
            )
        record = dict(sample.to_record())
        policy = record.get("deadline_policy")
        science_end = _optional_number(
            policy.get("science_returned_monotonic_seconds")
            if isinstance(policy, dict)
            else None
        )
        science_seconds = (
            science_end - clock_started
            if science_end is not None and math.isfinite(science_end)
            else None
        )
        in_time = (
            science_seconds is not None and 0.0 <= science_seconds <= attempt.timeout
        )
        positive = sample.positive_completion and in_time
        accepted = sample.accepted_case and in_time
        cold = record.get("cold")
        cold_evidence = cold.get("evidence") if isinstance(cold, dict) else None
        restored = (
            cold_evidence[0]
            if isinstance(cold_evidence, list)
            and cold_evidence
            and isinstance(cold_evidence[0], dict)
            else {}
        )
        epoch_ids = receipt.get("accepted_epoch_ids")
        continued_from = restored.get("continued_from")
        epoch_binding = (
            isinstance(epoch_ids, list)
            and len(epoch_ids) >= 3
            and len(epoch_ids) == 2 * attempt.frozen_controls["adaptation_rounds"] + 1
            and restored.get("restored_epoch_ids") == epoch_ids
            and isinstance(continued_from, dict)
            and continued_from.get("epoch_id") == epoch_ids[-2]
        )
        positive = positive and epoch_binding
        accepted = accepted and epoch_binding
        producer_peak = receipt["producer"].get("peak_process_resident_bytes")
        consumer_peak = sample.peak_process_resident_bytes
        cold_process = record.get("cold_process")
        cold_peak = (
            cold_process.get("peak_resident_bytes")
            if isinstance(cold_process, dict)
            else None
        )
        workflow_peak = (
            max(producer_peak, consumer_peak, cold_peak)
            if type(producer_peak) is int
            and type(consumer_peak) is int
            and type(cold_peak) is int
            and min(producer_peak, consumer_peak, cold_peak) >= 0
            else None
        )
        request = {
            "kind": "native-hybrid-producer-consumer-request",
            "controls": attempt.frozen_controls,
            "source_reference": _scenario_source_reference(
                "native-hybrid-lifecycle",
                attempt.frozen_controls["scenario_controls"],
                resolution=attempt.expected_controls["resolution"],
            ),
            "source_id": receipt["source_id"],
            "source_revision": receipt["source_revision"],
            "source_binding_id": receipt["source_binding_id"],
            "specification_id": receipt["specification_id"],
            "limits_id": receipt["limits_id"],
        }
        request_id = canonical_fingerprint(request)
        pipeline = {
            "kind": "native-hybrid-producer-consumer-workflow",
            "request": request,
            "request_id": request_id,
            "frozen_controls": attempt.frozen_controls,
            "prerequisite": prerequisite,
            "consumer_binding": sample.binding,
            "scientific_seconds": science_seconds,
            "process_total_seconds": perf_counter() - started,
            "limit_seconds": attempt.timeout,
            "completed_before_deadline": in_time,
            "accepted_epoch_inventory_restored": epoch_binding,
            "process_peaks": {
                "producer": producer_peak,
                "consumer": consumer_peak,
                "cold_worker": cold_peak,
            },
        }
        record.update(
            prerequisite_workflow=pipeline,
            prerequisite_workflow_id=canonical_fingerprint(pipeline),
            request_id=request_id,
            consumer_request_id=sample.request_id,
            consumer_end_to_end_seconds=sample.end_to_end_seconds,
            end_to_end_seconds=pipeline["process_total_seconds"],
            time_to_target_error_seconds=science_seconds if positive else None,
            positive_completion=positive,
            accepted_case=accepted,
            peak_process_resident_bytes=workflow_peak,
            process_peak_scope=(
                "maximum-individual-producer-consumer-cold-worker-post-exec-high-water;"
                "not-simultaneous-aggregate-memory"
            ),
            sample_binding={
                **sample.binding,
                "prerequisite_workflow_id": canonical_fingerprint(pipeline),
                "prerequisite_request_id": request_id,
            },
        )
        if not in_time:
            if sample.status != "timeout":
                record["status"] = "timeout" if science_seconds is not None else "failed"
            record["mandatory_workflow_completion"] = "incomplete"
            record["case_contract_completion"] = "incomplete"
        if not epoch_binding:
            if sample.positive_completion or sample.accepted_case:
                record["status"] = "failed"
                record["failure_phase"] = "hybrid-restored-epoch-inventory"
            record["mandatory_workflow_completion"] = "incomplete"
            record["case_contract_completion"] = "incomplete"
        record["phase_evidence"] = corpus_phase_evidence(record, attempt.capacity)
        record["resource_evidence"] = record["phase_evidence"]["resources"]
        return _decode_campaign_sample(record)


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
    resource_envelope: dict[str, Any] | None = None,
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
        resource_envelope=resource_envelope,
    )
    scenario = attempt.frozen_controls["scenario"]
    scenario_controls = attempt.frozen_controls["scenario_controls"]
    if attempt.ineligible is None and (
        scenario == "native-hybrid-lifecycle"
        and scenario_controls.get("hybrid_profile") == "archived-layer-core"
        and scenario_controls.get("hybrid_archive") is None
        and scenario_controls.get("hybrid_content_id") is None
    ):
        return _execute_generated_hybrid_attempt(attempt)
    if (
        attempt.ineligible is None
        and scenario == "native-distributed-lifecycle"
        and scenario_controls.get("distributed_checkpoint_root") is None
    ):
        with TemporaryDirectory(prefix="phydrax-campaign-distributed-") as directory:
            return _execute_campaign_attempt(
                _bind_generated_controls(
                    attempt,
                    {"distributed_checkpoint_root": str(Path(directory) / "checkpoint")},
                )
            )
    return _execute_campaign_attempt(attempt)


def _phase_mapping(value: Any, path: str) -> dict[str, Any]:
    """Distinguish an unreached phase from malformed emitted evidence."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            f"Compiler phase {path} must be a mapping; got {type(value).__name__}."
        )
    return value


def _phase_sequence(value: Any, path: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, dict) for item in value
    ):
        raise TypeError(f"Compiler phase {path} must be a sequence of mappings.")
    return tuple(value)


def _compiler_phase_record(
    compiler: Any,
    capacity: int,
    *,
    index: int,
    phase: str,
    solver: str,
    evidence_path: str,
    epoch_id: Any = None,
) -> dict[str, Any]:
    compiler = _phase_mapping(compiler, evidence_path)
    return {
        "epoch": index,
        "epoch_id": epoch_id,
        "phase": phase,
        "solver": solver,
        "evidence_path": evidence_path,
        "controlling_entity_capacity": capacity,
        **{
            name: compiler.get(name)
            for name in (
                "kernel",
                "lowering_seconds",
                "compile_seconds",
                "first_execution_seconds",
                "warm_samples_seconds",
                "execution_scope",
                "assembly_seconds",
                "assembly_invocations",
                "assembly_compilation_count",
                "assembly_compile_seconds",
                "assembly_binding_id",
                "temporary_bytes",
                "output_bytes",
                "argument_bytes",
                "generated_code_bytes",
                "logical_retained_bytes",
                "memory_scope",
                "compilation_count",
                "compilation_count_semantics",
            )
        },
        "memory_status": compiler.get("memory_status", "unmeasured"),
    }


def _physical_compiler_phases(
    physical: Any,
    path: str,
    index: int,
    phase: str,
    capacity: int,
    *,
    periodic: bool,
    epoch_id: Any,
) -> list[dict[str, Any]]:
    physical = _phase_mapping(physical, path)
    records = []
    for solver in ("fe", "vem", "fv"):
        owner = _phase_mapping(physical.get(solver), f"{path}.{solver}")
        records.append(
            _compiler_phase_record(
                owner.get("compiler"),
                capacity,
                index=index,
                phase=phase,
                solver=solver,
                evidence_path=f"{path}.{solver}.compiler",
                epoch_id=epoch_id,
            )
        )
    fv = _phase_mapping(physical.get("fv"), f"{path}.fv")
    if not periodic:
        records.append(
            _compiler_phase_record(
                fv.get("transferred_update_compiler"),
                capacity,
                index=index,
                phase=phase,
                solver="fv-transferred-update",
                epoch_id=epoch_id,
                evidence_path=f"{path}.fv.transferred_update_compiler",
            )
        )
    trace_path = f"{path}.fv.trace_repreparation"
    trace = _phase_mapping(fv.get("trace_repreparation"), trace_path)
    for trace_index, entry in enumerate(
        _phase_sequence(
            trace.get("current_and_history"),
            f"{trace_path}.current_and_history",
        )
    ):
        records.append(
            _compiler_phase_record(
                entry.get("compiler"),
                capacity,
                index=index,
                phase=phase,
                solver=f"fv-trace:{entry.get('state')}",
                epoch_id=epoch_id,
                evidence_path=f"{trace_path}.current_and_history[{trace_index}].compiler",
            )
        )
    return records


def _polyhedral_compiler_phases(
    sample: dict[str, Any],
    capacity: int,
) -> list[dict[str, Any]]:
    case = _phase_mapping(sample.get("case"), "case")
    periodic = "periodic-quotient" in case.get("controls", ())
    accepted = sample.get("accepted_epochs", ())
    if not isinstance(accepted, (tuple, list)):
        raise TypeError("Compiler phase accepted_epochs must be a sequence.")
    if periodic:
        physical = _phase_sequence(sample.get("physics"), "physics")
        count = max(len(accepted), len(physical), 1)
        warm = [
            (physical[index] if index < len(physical) else None, f"physics[{index}]")
            for index in range(count)
        ]
    else:
        warm = [(sample.get(name), name) for name in ("source_physics", "physics")]
    records = []
    for index, (physical, path) in enumerate(warm):
        records.extend(
            _physical_compiler_phases(
                physical,
                path,
                index,
                "warm",
                capacity,
                periodic=periodic,
                epoch_id=accepted[index] if index < len(accepted) else None,
            )
        )
    cold = _phase_mapping(sample.get("cold"), "cold")
    fresh = _phase_mapping(cold.get("fresh_interpreter"), "cold.fresh_interpreter")
    path = "cold.fresh_interpreter.continued_epochs"
    continued = _phase_sequence(fresh.get("continued_epochs"), path)
    for index in range(max(len(accepted), len(warm), len(continued))):
        epoch = continued[index] if index < len(continued) else {}
        records.extend(
            _physical_compiler_phases(
                epoch.get("physics"),
                f"{path}[{index}].physics",
                index,
                "cold",
                capacity,
                periodic=periodic,
                epoch_id=epoch.get("epoch_id"),
            )
        )
    return records


def corpus_phase_evidence(sample: dict[str, Any], capacity: int) -> dict[str, Any]:
    """Expose phase measurements without deriving missing work or bytes."""
    epochs = _phase_sequence(sample.get("epochs"), "epochs")
    compiler_records = []
    stages = sample.get("stages_seconds", {})
    if not isinstance(stages, dict):
        raise TypeError(
            f"corpus_phase_evidence.stages must be a mapping; got {type(stages).__name__!r}."
        )
    for position, epoch in enumerate(epochs):
        index = epoch["epoch"]
        measured = _compiler_phase_record(
            epoch.get("compiler"),
            capacity,
            index=index,
            phase="warm",
            solver="epoch",
            evidence_path=f"epochs[{position}].compiler",
            epoch_id=epoch.get("epoch_id"),
        )
        for field, stage in (
            ("lowering_seconds", "solver_lowering"),
            ("compile_seconds", "solver_compile"),
            ("first_execution_seconds", "solver_first_execution"),
        ):
            if measured[field] is None:
                measured[field] = stages.get(epoch_stage_name(stage, index))
        measured["logical_retained_bytes"] = epoch.get("retained_bytes")
        compiler_records.append(measured)
    if sample.get("scenario") == "native-polyhedral-lifecycle":
        compiler_records.extend(_polyhedral_compiler_phases(sample, capacity))
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
        "reporting_overhead_seconds": sample.get("reporting_overhead_seconds"),
        "process_peak_resident_bytes": sample.get("peak_process_resident_bytes"),
        "process_peak_scope": sample.get("process_peak_scope"),
        "device_memory": sample.get("device_memory"),
    }
    recorded = bool(
        measured_stages
        or any(
            phase.get(field) is not None
            for phase in compiler_records
            for field in (
                "lowering_seconds",
                "compile_seconds",
                "first_execution_seconds",
                "warm_samples_seconds",
                "temporary_bytes",
                "output_bytes",
                "generated_code_bytes",
                "logical_retained_bytes",
            )
        )
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
        "prerequisite_workflow": sample.get("prerequisite_workflow"),
        "scope": (
            "Measured stage intervals, compiler analysis, logical retention, process peak, "
            "and backend device reports remain separate; inclusive intervals are not summed."
        ),
        "missing_measurements_are_zero": False,
        "status": "recorded" if recorded else "unmeasured",
    }


def _seconds_summary(values: list[int | float], /) -> dict[str, float | None]:
    return {
        "median": None if not values else float(np.median(values)),
        "minimum": None if not values else float(np.min(values)),
        "maximum": None if not values else float(np.max(values)),
    }


_ACCEPTED_CASE_PREDICATE = (
    "Positive admission, child status passed, independent scientific certification "
    "passed and case contract complete, with the child verdict re-derived from its own "
    "record and launched controls. Mandatory workflow completion is counted separately "
    "and is never implied by an accepted case."
)


def _campaign_candidate_summary(
    samples: list[CampaignSample],
    outcome_names: tuple[str, ...],
    *,
    physics_error: bool,
) -> dict[str, object]:
    """Aggregate accepted-case timings without discarding failed attempts."""
    candidate_identities = [
        sample.request_id for sample in samples if sample.request_id is not None
    ]
    statuses = [sample.status for sample in samples]
    accepted = [sample for sample in samples if sample.accepted_case]
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
    # Source-contract profiles (e.g. hybrid full_workflow=False) are accepted on
    # their own contract without a physical target; their error stays unavailable.
    errors = [
        sample.physical_error for sample in accepted if sample.physical_error is not None
    ]
    unmeasured = len(accepted) - len(errors)
    return {
        "samples": [sample.to_record() for sample in samples],
        "attempts": len(samples),
        "counts": counts,
        "mandatory_positive_completions": sum(
            sample.positive_completion for sample in samples
        ),
        "accepted_cases": len(accepted),
        "accepted_case_rate": len(accepted) / len(samples),
        "accepted_case_predicate": _ACCEPTED_CASE_PREDICATE,
        "certified_refusals": statuses.count("expected-refusal"),
        "refusals_count_as_positive_completion": False,
        "request_identities": sorted(set(candidate_identities)),
        "request_identity_complete": len(candidate_identities) == len(samples),
        "runtime_identities": sorted(
            {
                identity
                for sample in samples
                if (identity := sample.binding["runtime_id"]) is not None
            }
        ),
        "time_to_target_error_seconds": _seconds_summary(
            [
                sample.time_to_target_error_seconds
                for sample in accepted
                if sample.time_to_target_error_seconds is not None
            ]
        ),
        "end_to_end_seconds": _seconds_summary(
            [
                sample.end_to_end_seconds
                for sample in accepted
                if sample.end_to_end_seconds is not None
            ]
        ),
        "timing_semantics": {
            "time_to_target_error_seconds": "scenario start to the first accepted solver epoch meeting the target error",
            "end_to_end_seconds": "scenario start to completion of the whole lifecycle, including restart and certification",
            "population": "accepted cases only",
        },
        "peak_process_resident_bytes": None if not peaks else int(np.max(peaks)),
        "peak_process_resident_scope": "child-reported post-exec high water of accepted cases",
        (
            "worst_accepted_physics_error" if physics_error else "worst_accepted_l2_error"
        ): None if not errors else max(errors),
        "accepted_cases_without_physical_error": unmeasured,
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
                **_scenario_source_reference(
                    scenario_override, controls, resolution=resolution
                ),
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
    resource_envelope: dict[str, Any] | None = None,
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
                resource_envelope=resource_envelope,
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
                if profile["case_name"] in POLYHEDRAL_CASE_NAMES
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
        "profile_requests": {
            name: {
                "scenario_controls": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in profile["controls"].items()
                },
                "resolution_scale": profile["resolution_scale"],
                "minimum_adaptation_rounds": profile["minimum_adaptation_rounds"],
                **(
                    {"resource_envelope": profile["resource_envelope"]}
                    if "resource_envelope" in profile else {}
                ),
            }
            for name, profile in profiles.items()
        },
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
        "maximum_work_units": args.maximum_work_units,
        "maximum_geometry_queries": args.maximum_geometry_queries,
        "maximum_scratch_bytes": args.maximum_scratch_bytes,
        "maximum_data_bytes": args.maximum_data_bytes,
        "maximum_recipe_bytes": args.maximum_recipe_bytes,
        "transport_steps": args.transport_steps,
        "transport_dt": args.transport_dt,
        "balance_tolerance": args.balance_tolerance,
        "oracle_capacity": args.oracle_capacity,
        "cad_format": args.cad_format,
        "cad_case": args.cad_case,
        "implicit_case": args.implicit_case,
        "image_case": args.image_case,
        "design_case": args.design_case,
        "overset_profile": args.overset_profile,
        "reconstruction_case": args.reconstruction_case,
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
    for key in (
        *_NATIVE_DRIVER_ARGUMENTS[name],
        "maximum_work_units",
        "maximum_geometry_queries",
        *(("maximum_scratch_bytes",) if name in (
            "native-overset-lifecycle", "native-family-lifecycle",
            "native-surface-pde", "native-periodic-lifecycle",
        ) else ()),
        *(("maximum_data_bytes",) if name == "native-family-lifecycle" else ()),
        *(("maximum_recipe_bytes",) if name == "native-periodic-lifecycle" else ()),
        *(("maximum_memory_bytes",) if name in (
            "native-distributed-lifecycle", "native-family-lifecycle",
        ) else ()),
    ):
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
        missing = _missing_profile_inputs(profile, inputs)
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
        campaigns = [
            benchmark_native_lifecycle(
                case_name,
                resolution,
                capacity,
                qualification_timeout(controls, args.timeout),
                args.repeats,
                args.target_error,
                rounds,
                args.comparison,
                benchmark_warm_repeats=args.benchmark_warm_repeats,
                scenario_override=scenario,
                controls=controls,
                resource_envelope=profile.get("resource_envelope"),
                source_reference=(
                    _scenario_source_reference(scenario, controls, resolution=resolution)
                    if scenario == "native-periodic-lifecycle"
                    else source_reference
                ),
            )
            for resolution, rounds in (
                _corpus_request_dimensions(profile, value, args.adaptation_rounds)
                for value in args.resolution
            )
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
            **(
                {"resource_envelope": profile["resource_envelope"]}
                if "resource_envelope" in profile else {}
            ),
        }
        records[name] = {**body, "profile_result_id": canonical_fingerprint(body)}
    body = {
        "kind": "registered-native-meshing-benchmark-corpus",
        "selected_profiles": list(selected),
        "missing_profile_inputs": missing_inputs,
        "profiles": records,
    }
    return {**body, "corpus_benchmark_id": canonical_fingerprint(body)}
