# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Synthetic evidence-parser fixtures do not constitute physical GPU evidence."""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from phydrax.linalg.krylov import KrylovBreakdownStatus
from tests._support.cuda_execution_worker import _copy_evidence, pattern, SIZES, STEPS
from tools import qualify_cuda_execution as qualification
from tools._accelerator_process import CommandEvidence


DEVICES = [f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(1, 9)]


def _command(*, timeout: bool = False, status: int = 0) -> CommandEvidence:
    return {
        "command": ["synthetic-parser-fixture"],
        "cwd": "/fixture",
        "returncode": status,
        "timed_out": timeout,
        "stdout": "",
        "stderr": "",
        "stdout_path": "/fixture/out",
        "stderr_path": "/fixture/err",
        "stdout_sha256": "0" * 64,
        "stderr_sha256": "0" * 64,
    }


def _ownership(devices: list[str], processes: int) -> list[dict[str, object]]:
    local = len(devices) // processes
    return [
        {
            "rank": index // local,
            "local_ordinal": index,
            "gpu_uuid": device,
            "platform": "gpu",
        }
        for index, device in enumerate(devices)
    ]


def _peers(devices: list[str] = DEVICES) -> dict[str, Any]:
    rows = []
    for source_index, source in enumerate(devices):
        for destination_index, destination in enumerate(devices):
            if source == destination:
                continue
            for size in SIZES:
                for step in STEPS:
                    values = pattern(source_index, destination_index, size, step)
                    digest = hashlib.sha256(values.tobytes()).hexdigest()
                    prefix = [int(value) for value in values[:8]]
                    for route in ("cuda", "jax"):
                        rows.append(
                            {
                                "route": route,
                                "source_uuid": source,
                                "destination_uuid": destination,
                                "bytes": size,
                                "step": step,
                                "expected_sha256": digest,
                                "observed_sha256": digest,
                                "mismatch_count": 0,
                                "passed": True,
                                "ingress_sha256": digest,
                                "ingress_mismatch_count": 0,
                                "expected_prefix": prefix,
                                "ingress_prefix": prefix,
                                "observed_prefix": prefix,
                                "failure_stage": "none",
                            }
                        )
    return {
        "passed": True,
        "errors": [],
        "ownership": _ownership(devices, 1),
        "peer_matrix_status": "not-applicable" if len(devices) == 1 else "required",
        "peer_connectivity": [
            {
                "source_uuid": source,
                "destination_uuid": destination,
                "capability_reported": True,
            }
            for source in devices
            for destination in devices
            if source != destination
        ],
        "rows": rows,
        "coverage": {
            "expected_copy_observations": len(rows),
            "recorded_copy_observations": len(rows),
            "recorded_ingress_observations": len(rows),
            "passed_ingress_observations": len(rows),
            "passed_copy_observations": len(rows),
        },
    }


def _collectives(scenario: str) -> dict[str, Any]:
    processes, local = (int(value) for value in scenario.split("x"))
    devices = DEVICES[-processes * local :]
    owners = _ownership(devices, processes)
    ranks = []
    for rank in range(processes):
        ranks.append(
            {
                "passed": True,
                "errors": [],
                "ownership": owners[rank * local : (rank + 1) * local],
                "runtime": {
                    "process_index": rank,
                    "process_count": processes,
                    "global_device_count": processes * local,
                    "local_device_count": local,
                    "execution_group_id": "observed-fixture-group",
                    "process_host_ids": [
                        [index, "same-host"] for index in range(processes)
                    ],
                },
                "problem": {
                    "logical_size": 17,
                    "iterations": 2,
                    "group_identity_consistent": True,
                },
                "weighted_mean": {
                    "absolute_error": 0.0,
                    "observed": 0.5,
                    "expected": 0.5,
                },
                "distributed_pcg": {
                    "maximum_solution_error": 0.0,
                    "converged": True,
                    "breakdown": int(KrylovBreakdownStatus.HAPPY),
                    "residual_norm": 0.0,
                    "residual_absolute_tolerance": 1e-12,
                },
            }
        )
    return {"passed": True, "errors": [], "ranks": ranks} if processes > 1 else ranks[0]


def test_explicit_eight_gpu_selection_requires_complete_scenarios() -> None:
    qualification.validate_selection(8, DEVICES, qualification.SCENARIOS)
    for scenarios in (
        qualification.SCENARIOS[:-1],
        (*qualification.SCENARIOS, "8x1"),
        ("peer-copies",),
    ):
        with pytest.raises(ValueError, match="Missing, duplicate, or narrowed"):
            qualification.validate_selection(8, DEVICES, scenarios)
    for devices in (DEVICES[:-1], [DEVICES[0]] * 8, ["0"] * 8):
        with pytest.raises(ValueError):
            qualification.validate_selection(8, devices, qualification.SCENARIOS)


def test_peer_fixture_covers_all_56_directed_pairs_on_both_routes() -> None:
    record = _peers()
    assert qualification.scenario_errors(_command(), record, "peer-copies", DEVICES) == []
    pairs = {(row["source_uuid"], row["destination_uuid"]) for row in record["rows"]}
    assert len(pairs) == 56


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_peer_qualification_requires_complete_direct_matrix(count: int) -> None:
    devices = DEVICES[:count]
    record = _peers(devices)
    assert len(record["peer_connectivity"]) == count * (count - 1)
    assert qualification.scenario_errors(_command(), record, "peer-copies", devices) == []


@pytest.mark.parametrize(
    "fault",
    [
        "unavailable",
        "missing-matrix",
        "missing-pair",
        "extra-pair",
        "duplicate",
        "self-pair",
        "foreign-source",
        "foreign-destination",
        "malformed-matrix",
        "malformed-row",
        "malformed-source",
        "malformed-destination",
        "missing-capability",
        "integer-capability",
        "string-capability",
    ],
)
def test_correct_copy_contents_cannot_replace_direct_peer_evidence(fault: str) -> None:
    devices = DEVICES[:2]
    record = _peers(devices)
    rows = record["peer_connectivity"]
    if fault == "unavailable":
        rows[0]["capability_reported"] = False
    elif fault == "missing-matrix":
        del record["peer_connectivity"]
    elif fault == "missing-pair":
        rows.pop()
    elif fault == "extra-pair":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif fault == "self-pair":
        rows[0]["destination_uuid"] = rows[0]["source_uuid"]
    elif fault == "foreign-source":
        rows[0]["source_uuid"] = DEVICES[2]
    elif fault == "foreign-destination":
        rows[0]["destination_uuid"] = DEVICES[2]
    elif fault == "malformed-matrix":
        record["peer_connectivity"] = {}
    elif fault == "malformed-row":
        rows[0] = None
    elif fault == "malformed-source":
        rows[0]["source_uuid"] = []
    elif fault == "malformed-destination":
        rows[0]["destination_uuid"] = None
    elif fault == "missing-capability":
        del rows[0]["capability_reported"]
    elif fault == "integer-capability":
        rows[0]["capability_reported"] = 1
    else:
        rows[0]["capability_reported"] = "true"
    assert all(row["passed"] is True for row in record["rows"])
    assert qualification.scenario_errors(_command(), record, "peer-copies", devices)


@pytest.mark.parametrize(
    "fault",
    ["missing-pair", "duplicate", "skip", "contents", "stale", "sizes", "cpu", "uuid"],
)
def test_peer_qualification_refuses_false_success(fault: str) -> None:
    record = _peers()
    if fault == "missing-pair":
        record["rows"] = [
            row
            for row in record["rows"]
            if not (
                row["source_uuid"] == DEVICES[0]
                and row["destination_uuid"] == DEVICES[-1]
            )
        ]
    elif fault == "duplicate":
        record["rows"].append(copy.deepcopy(record["rows"][0]))
    elif fault == "skip":
        record["rows"][0]["passed"] = "skipped"
    elif fault == "contents":
        record["rows"][0]["observed_sha256"] = "0" * 64
    elif fault == "stale":
        for row in record["rows"]:
            row["expected_sha256"] = row["observed_sha256"] = "0" * 64
    elif fault == "sizes":
        record["rows"] = [row for row in record["rows"] if row["bytes"] == SIZES[0]]
    elif fault == "cpu":
        record["ownership"][0]["platform"] = "cpu"
    else:
        record["ownership"][1]["gpu_uuid"] = DEVICES[0]
    assert qualification.scenario_errors(_command(), record, "peer-copies", DEVICES)


@pytest.mark.parametrize(
    "field", ["bytes", "step", "ingress_mismatch_count", "mismatch_count"]
)
def test_peer_qualification_rejects_boolean_identity_and_counts(field: str) -> None:
    record = _peers()
    record["rows"][0][field] = False
    assert qualification.scenario_errors(_command(), record, "peer-copies", DEVICES)


@pytest.mark.parametrize("field", ["rank", "local_ordinal"])
def test_peer_qualification_rejects_boolean_ownership(field: str) -> None:
    record = _peers()
    record["ownership"][0][field] = False
    assert qualification.scenario_errors(_command(), record, "peer-copies", DEVICES)


@pytest.mark.parametrize(
    "field",
    [
        "expected_copy_observations",
        "recorded_copy_observations",
        "recorded_ingress_observations",
        "passed_ingress_observations",
        "passed_copy_observations",
    ],
)
def test_peer_qualification_rejects_boolean_zero_coverage(field: str) -> None:
    devices = DEVICES[:1]
    record = _peers(devices)
    record["coverage"][field] = False
    assert qualification.scenario_errors(_command(), record, "peer-copies", devices)


@pytest.mark.parametrize("scenario", ["1x1", "1x2", "1x4", "1x8", "2x4", "4x2", "8x1"])
def test_collective_parser_admits_exact_topology_and_current_numerics(
    scenario: str,
) -> None:
    count = 1
    for value in scenario.split("x"):
        count *= int(value)
    assert (
        qualification.scenario_errors(
            _command(), _collectives(scenario), scenario, DEVICES[-count:]
        )
        == []
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing-rank",
        "duplicate-rank",
        "group",
        "uuid",
        "cpu",
        "nan",
        "convergence",
        "narrowed",
        "skip",
    ],
)
def test_collective_qualification_refuses_false_success(fault: str) -> None:
    record = _collectives("4x2")
    ranks = record["ranks"]
    if fault == "missing-rank":
        ranks.pop()
    elif fault == "duplicate-rank":
        ranks[-1] = ranks[0]
    elif fault == "group":
        ranks[-1]["runtime"]["execution_group_id"] = "another-group"
    elif fault == "uuid":
        ranks[-1]["ownership"][0]["gpu_uuid"] = DEVICES[0]
    elif fault == "cpu":
        ranks[-1]["ownership"][0]["platform"] = "cpu"
    elif fault == "nan":
        ranks[-1]["weighted_mean"]["absolute_error"] = float("nan")
    elif fault == "convergence":
        ranks[-1]["distributed_pcg"]["converged"] = False
    elif fault == "narrowed":
        ranks[-1]["problem"]["logical_size"] = 1
    else:
        ranks[-1]["passed"] = "skipped"
    assert qualification.scenario_errors(_command(), record, "4x2", DEVICES)


def test_timed_out_or_failed_command_cannot_qualify_complete_evidence() -> None:
    record = _collectives("8x1")
    for command in (_command(timeout=True), _command(status=77), _command(status=1)):
        assert qualification.scenario_errors(command, record, "8x1", DEVICES)


@pytest.mark.parametrize(
    "environment",
    [
        {"JAX_PLATFORMS": "cpu"},
        {"XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
        {"PHYDRAX_NUM_PROCESSES": "8"},
        {"JAX_NUM_CPU_DEVICES": "8"},
        {"JAX_PLATFORM_NAME": "cpu"},
    ],
)
def test_forced_or_inherited_topology_refuses_qualification(
    environment: dict[str, str],
) -> None:
    assert qualification.environment_errors(environment)


def test_cli_retains_failed_evidence_for_incomplete_scenario_selection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "failed.json"
    python = tmp_path / "python.exe"
    python.touch()
    with pytest.raises(SystemExit) as error:
        qualification.main(
            [
                "--required-gpus",
                "8",
                "--devices",
                *DEVICES,
                "--scenarios",
                "peer-copies",
                "--output",
                str(output),
                "--python",
                str(python),
                "--work-root",
                str(tmp_path),
            ]
        )
    assert error.value.code == 1
    record = json.loads(output.read_text())
    assert record["passed"] is False
    assert record["errors"]
    assert record["scenarios"] == []


def test_missing_result_retains_command_and_fails_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(command: list[str], **kwargs: Any) -> CommandEvidence:
        completed = _command(timeout=True, status=-9)
        completed["command"] = command
        return completed

    monkeypatch.setattr(qualification, "run", timeout)
    result = qualification._scenario(
        "1x2",
        DEVICES[-2:],
        tmp_path,
        {},
        0.1,
        python=tmp_path / "python.exe",
        harness=tmp_path,
    )
    assert result["passed"] is False
    assert result["command"]["timed_out"]
    assert result["devices"] == DEVICES[-2:]
    assert result["result"] == {}
    assert result["errors"]


def test_complete_cli_selection_still_refuses_forced_cpu_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    output = tmp_path / "cpu-refused.json"
    python = tmp_path / "python.exe"
    python.touch()
    with pytest.raises(SystemExit) as error:
        qualification.main(
            [
                "--required-gpus",
                "8",
                "--devices",
                *DEVICES,
                "--scenarios",
                *qualification.SCENARIOS,
                "--output",
                str(output),
                "--python",
                str(python),
                "--work-root",
                str(tmp_path),
            ]
        )
    assert error.value.code == 1
    record = json.loads(output.read_text())
    assert record["passed"] is False
    assert record["errors"]
    assert record["scenarios"] == []


@pytest.mark.parametrize("status", list(KrylovBreakdownStatus))
def test_collective_qualification_preserves_canonical_breakdown_status(
    status: KrylovBreakdownStatus,
) -> None:
    record = _collectives("8x1")
    record["ranks"][-1]["distributed_pcg"]["breakdown"] = int(status)
    errors = qualification.scenario_errors(_command(), record, "8x1", DEVICES)
    assert bool(errors) is (
        status not in (KrylovBreakdownStatus.NONE, KrylovBreakdownStatus.HAPPY)
    )


@pytest.mark.parametrize("status", [False, True])
def test_boolean_breakdown_evidence_is_not_a_canonical_status(status: bool) -> None:
    record = _collectives("8x1")
    record["ranks"][-1]["distributed_pcg"]["breakdown"] = status
    assert qualification.scenario_errors(_command(), record, "8x1", DEVICES)


@pytest.mark.parametrize("fault", ["ingress", "unobserved-counts", "missing-ingress"])
def test_peer_qualification_refuses_uncertified_ingress_and_invented_counts(
    fault: str,
) -> None:
    record = _peers()
    if fault == "ingress":
        record["rows"][0]["ingress_sha256"] = "0" * 64
        record["rows"][0]["ingress_mismatch_count"] = 1
    elif fault == "unobserved-counts":
        record["coverage"]["passed_ingress_observations"] += 1
    else:
        del record["rows"][0]["ingress_sha256"]
    assert qualification.scenario_errors(_command(), record, "peer-copies", DEVICES)


@pytest.mark.parametrize(
    ("ingress_correct", "peer_correct", "stage"),
    [
        (True, True, "none"),
        (False, True, "source-ingress"),
        (False, False, "source-ingress"),
        (True, False, "peer-transfer"),
    ],
)
def test_copy_evidence_distinguishes_ingress_and_peer_failures(
    ingress_correct: bool, peer_correct: bool, stage: str
) -> None:
    expected = np.arange(17, dtype=np.uint32)
    ingress = expected if ingress_correct else np.full_like(expected, 99)
    observed = expected if peer_correct else np.full_like(expected, 101)
    row = _copy_evidence(
        expected,
        ingress,
        observed,
        route="cuda",
        source_uuid=DEVICES[0],
        destination_uuid=DEVICES[1],
        step=0,
        expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
    )
    assert row["failure_stage"] == stage
    assert row["passed"] is (ingress_correct and peer_correct)
    assert row["ingress_mismatch_count"] == (0 if ingress_correct else expected.size)
    assert row["mismatch_count"] == (0 if peer_correct else expected.size)
    assert (
        len(row["expected_prefix"])
        == len(row["ingress_prefix"])
        == len(row["observed_prefix"])
        == 8
    )


def _consumers(devices: list[str]) -> dict[str, Any]:
    from tests._support.cuda_execution_worker import (
        consumer_evidence,
        consumer_oracle,
        CONSUMER_ROUTES,
        CONSUMER_SIZES,
    )

    rows = []
    for source, source_uuid in enumerate(devices):
        for destination, destination_uuid in enumerate(devices):
            if source == destination:
                continue
            for route in CONSUMER_ROUTES:
                for size in CONSUMER_SIZES:
                    for step in STEPS:
                        values = consumer_oracle(source, destination, size, step)
                        rows.append(
                            dict(
                                route=route,
                                source_uuid=source_uuid,
                                destination_uuid=destination_uuid,
                                bytes=size,
                                step=step,
                                **consumer_evidence(
                                    values, values, values, values * 3 + 7, 0, 0
                                ),
                            )
                        )
    return dict(
        passed=True,
        errors=[],
        ownership=_ownership(devices, 1),
        peer_matrix_status="not-applicable" if len(devices) == 1 else "required",
        rows=rows,
        sizes_bytes=list(CONSUMER_SIZES),
        steps=list(STEPS),
        routes=list(CONSUMER_ROUTES),
        coverage=dict(
            expected_consumer_observations=len(rows),
            recorded_consumer_observations=len(rows),
            passed_consumer_observations=len(rows),
        ),
    )


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_consumer_coverage_is_exact_for_every_directed_pair(count: int) -> None:
    record = _consumers(DEVICES[:count])
    assert len(record["rows"]) == count * (count - 1) * 3 * 5 * 3
    assert {row["bytes"] for row in record["rows"]} == (
        {4, 12, 16, 40, 1024} if count > 1 else set()
    )
    assert (
        qualification.scenario_errors(
            _command(), record, "jax-consumers", DEVICES[:count]
        )
        == []
    )
    assert "jax-consumers" in qualification.required_scenarios(count)


def test_consumer_controls_do_not_qualify_unavailable_direct_peers() -> None:
    devices = DEVICES[:2]
    peers = _peers(devices)
    peers["peer_connectivity"][0]["capability_reported"] = False
    consumers = _consumers(devices)
    consumers["peer_connectivity"] = copy.deepcopy(peers["peer_connectivity"])
    assert (
        qualification.scenario_errors(_command(), consumers, "jax-consumers", devices)
        == []
    )
    assert qualification.scenario_errors(_command(), peers, "peer-copies", devices)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "consumer",
        "transfer",
        "ingress",
        "readback",
        "stale",
        "prefix",
        "count",
        "boolean-count",
        "missing-field",
        "failed",
        "route",
        "size",
        "step",
        "malformed",
    ],
)
def test_consumer_gate_refuses_incomplete_or_incorrect_evidence(fault: str) -> None:
    record = _consumers(DEVICES[:2])
    row = record["rows"][0]
    if fault == "missing":
        record["rows"].pop()
    elif fault == "duplicate":
        record["rows"][-1] = copy.deepcopy(row)
    elif fault == "consumer":
        row["device_consumer_mismatch_count"] = 1
    elif fault == "transfer":
        row["device_transfer_mismatch_count"] = 1
    elif fault == "ingress":
        row["ingress_mismatch_count"] = 1
    elif fault == "readback":
        row["consumer_sha256"] = "0" * 64
    elif fault == "stale":
        row["expected_sha256"] = "0" * 64
    elif fault == "prefix":
        row["consumer_prefix"] = [0]
    elif fault == "count":
        record["coverage"]["expected_consumer_observations"] -= 1
    elif fault == "boolean-count":
        row["device_transfer_mismatch_count"] = False
    elif fault == "missing-field":
        del row["consumer_sha256"]
    elif fault == "failed":
        row["passed"] = False
    elif fault == "route":
        row["route"] = "host-fallback"
    elif fault == "size":
        row["bytes"] = 8
    elif fault == "step":
        row["step"] = 99
    else:
        record["rows"][0] = None
    assert qualification.scenario_errors(_command(), record, "jax-consumers", DEVICES[:2])


@pytest.mark.parametrize(
    "stage",
    [
        "source-ingress",
        "peer-transfer",
        "destination-consumer",
        "host-readback",
    ],
)
def test_consumer_evidence_distinguishes_failure_boundaries(stage: str) -> None:
    from tests._support.cuda_execution_worker import consumer_evidence, consumer_oracle

    values = consumer_oracle(0, 1, 40, 1)
    ingress, transferred, consumed = values.copy(), values.copy(), values * 3 + 7
    transfer_bad = consumer_bad = 0
    if stage == "source-ingress":
        ingress[0] += 1
    elif stage == "peer-transfer":
        transfer_bad = 1
        transferred[0] += 1
    elif stage == "destination-consumer":
        consumer_bad = 1
        consumed[0] += 1
    else:
        consumed[0] += 1
    row = consumer_evidence(
        values, ingress, transferred, consumed, transfer_bad, consumer_bad
    )
    assert row["passed"] is False
    assert row["failure_stage"] == stage
    assert len(row["expected_prefix"]) == 8


@pytest.mark.parametrize(
    "scenario,factory",
    [
        ("peer-copies", _peers),
        ("jax-consumers", _consumers),
    ],
)
def test_single_device_empty_matrix_is_not_peer_qualification(
    scenario: str, factory: Callable[[list[str]], dict[str, Any]]
) -> None:
    record = factory(DEVICES[:1])
    assert qualification.scenario_errors(_command(), record, scenario, DEVICES[:1]) == []
    del record["peer_matrix_status"]
    assert qualification.scenario_errors(_command(), record, scenario, DEVICES[:1])


def test_one_process_scientific_smoke_never_launches_coordinator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tools import distributed_substrate_benchmark as substrate

    monkeypatch.setattr(
        substrate, "_launch_ranks", lambda *_: pytest.fail("one-process coordinator")
    )
    monkeypatch.setattr(substrate.jax, "local_devices", lambda: [])
    monkeypatch.setattr(substrate, "_local_record", lambda _: ({"passed": True}, []))
    substrate.main(
        [
            "--local-processes",
            "1",
            "--devices-per-process",
            "1",
            "--required-gpus",
            "1",
            "--size",
            "17",
            "--iterations",
            "2",
            "--output",
            str(tmp_path / "smoke.json"),
        ]
    )
    assert json.loads((tmp_path / "smoke.json").read_text())["passed"]


@pytest.mark.parametrize(
    "identity_fault",
    [None, "missing", "duplicate", "malformed", "short", "nonzero", "timeout"],
)
def test_windows_qualification_requires_installed_identity_before_gpu_scenarios(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, identity_fault: str | None
) -> None:
    from benchmarks import _runtime
    from tools import qualify_cuda_wheels

    monkeypatch.setattr(qualification.sys, "platform", "win32")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_NUM_CPU_DEVICES", raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setattr(
        _runtime,
        "source_build_fingerprint",
        lambda *_: pytest.fail(
            "installed qualification must not require checkout identity"
        ),
    )
    monkeypatch.setattr(_runtime, "benchmark_driver_fingerprint", lambda *_: "synthetic")
    monkeypatch.setattr(qualify_cuda_wheels, "copy_harness", lambda *_: {})
    scenarios: list[str] = []

    def scenario(name: str, *args: object, **kwargs: object) -> dict[str, object]:
        scenarios.append(name)
        return {"errors": [], "passed": True}

    monkeypatch.setattr(qualification, "_scenario", scenario)
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> CommandEvidence:
        commands.append(command)
        completed = _command()
        completed["command"] = command
        if command[0] == str(python):
            completed["stdout"] = "INSTALLED_BUILD_FINGERPRINT " + "a" * 64 + "\n"
            if identity_fault == "missing":
                completed["stdout"] = ""
            elif identity_fault == "duplicate":
                completed["stdout"] *= 2
            elif identity_fault == "malformed":
                completed["stdout"] = "INSTALLED_BUILD_FINGERPRINT " + "z" * 64
            elif identity_fault == "short":
                completed["stdout"] = "INSTALLED_BUILD_FINGERPRINT abc"
            elif identity_fault == "nonzero":
                completed["returncode"] = 1
            elif identity_fault == "timeout":
                completed["timed_out"] = True
        return completed

    monkeypatch.setattr(qualification, "run", run)
    python = tmp_path / "python.exe"
    python.touch()
    arguments = [
        "--python",
        str(python),
        "--work-root",
        str(tmp_path),
        "--required-gpus",
        "1",
        "--devices",
        DEVICES[0],
        "--scenarios",
        *qualification.required_scenarios(1),
        "--output",
        str(tmp_path / "inventory.json"),
    ]
    if identity_fault is None:
        qualification.main(arguments)
    else:
        with pytest.raises(SystemExit) as error:
            qualification.main(arguments)
        assert error.value.code == 1
    record = json.loads((tmp_path / "inventory.json").read_text())
    assert record["passed"] is (identity_fault is None)
    assert "source_build_fingerprint" not in record["identity"]
    assert record["identity"]["qualification_driver_fingerprint"] == "synthetic"
    assert record["identity"]["substrate_driver_fingerprint"] == "synthetic"
    assert record["installed_build_identity_command"]["command"][0] == str(python)
    assert "-I" in record["installed_build_identity_command"]["command"]
    if identity_fault is None:
        assert record["identity"]["installed_build_fingerprint"] == "a" * 64
        assert scenarios == list(qualification.required_scenarios(1))
        assert any(command[0] == "nvidia-smi" for command in commands)
    else:
        assert "installed_build_fingerprint" not in record["identity"]
        assert record["errors"]
        assert not scenarios
        assert not any(command[0] == "nvidia-smi" for command in commands)
    assert not any(command[0] == "lscpu" or "topo" in command for command in commands)


def _worker_windows_runtime(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> ModuleType:
    from tests._support import cuda_execution_worker as worker

    monkeypatch.setattr(worker, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setitem(
        sys.modules, "phydrax_cuda13_pjrt", ModuleType("phydrax_cuda13_pjrt")
    )
    monkeypatch.setitem(sys.modules, "phydrax_cuda13_pjrt.runtime", module)
    return worker


def test_windows_worker_driver_lookup_uses_public_api_and_windows_convention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker_windows_runtime(
        monkeypatch, ModuleType("phydrax_cuda13_pjrt.runtime")
    )
    convention = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
    callback = convention(ctypes.c_int, ctypes.c_int)(lambda value: value + 7)
    address = ctypes.cast(callback, ctypes.c_void_p).value
    signatures: list[tuple[Any, ...]] = []

    def windows_signature(*signature: Any) -> Any:
        signatures.append(signature)
        return convention(*signature)

    def entrypoint(
        symbol: bytes, output: Any, version: int, flags: int, query: Any
    ) -> int:
        assert symbol == b"cuDeviceGet"
        assert version == 13000 and flags == 0
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = address
        ctypes.cast(query, ctypes.POINTER(ctypes.c_int))[0] = 0
        return 0

    monkeypatch.setattr(ctypes, "WINFUNCTYPE", windows_signature, raising=False)
    runtime = SimpleNamespace(cudaGetDriverEntryPointByVersion=entrypoint)
    function = worker._driver_function(runtime, b"cuDeviceGet", (ctypes.c_int,))
    assert function(3) == 10
    assert signatures == [(ctypes.c_int, ctypes.c_int)]


@pytest.mark.parametrize("version", (13010, 14010))
def test_windows_inventory_records_hybrid_not_a_fictitious_shared_runtime(
    monkeypatch: pytest.MonkeyPatch,
    version: int,
) -> None:
    import importlib.metadata

    from benchmarks import _runtime
    from phydrax import _cuda_native

    identity = {
        "linkage": "driver-hybrid",
        "owner": "display-driver",
        "runtime_path": "System32/DriverStore/FileRepository/test/nvcudart_hybrid64.dll",
        "runtime_sha256": "a" * 64,
        "driver_path": "System32/DriverStore/FileRepository/test/nvcuda64.dll",
        "driver_sha256": "b" * 64,
        "driver_dispatch_path": "System32/nvcuda.dll",
        "driver_dispatch_sha256": "c" * 64,
    }
    module = ModuleType("phydrax_cuda13_pjrt.runtime")
    monkeypatch.setattr(
        module,
        "runtime_identity",
        lambda: SimpleNamespace(
            runtime_path=identity["runtime_path"], _asdict=lambda: identity
        ),
        raising=False,
    )
    worker = _worker_windows_runtime(monkeypatch, module)
    monkeypatch.setattr(worker, "ownership", lambda _: [])
    monkeypatch.setattr(
        _cuda_native,
        "load_cuda_component",
        lambda _: SimpleNamespace(
            name="cudss",
            path="admitted.dll",
            version="0.2.2",
            build_hash="d" * 64,
            major=13,
        ),
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "synthetic")
    monkeypatch.setattr(
        _runtime, "capture_environment", lambda: SimpleNamespace(to_dict=lambda: {})
    )

    def observed(output: Any) -> int:
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = version
        return 0

    # Deliberately no _name: a bound API is not a shared-library handle.
    api = SimpleNamespace(
        owner_path="admitted.dll",
        cudaDriverGetVersion=observed,
        cudaRuntimeGetVersion=observed,
    )
    record = worker.inventory(api)
    assert record["runtime_identity"] == identity
    assert record["runtime_path"] == identity["runtime_path"]
    assert record["runtime_api_owner_path"] == "admitted.dll"
    assert record["compiled_cuda_major"] == 13
    assert record["driver_compatibility_version"] == version
    assert record["runtime_version"] == version
    assert "driver_version" not in record
