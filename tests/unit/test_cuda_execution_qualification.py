# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Synthetic evidence-parser fixtures do not constitute physical GPU evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
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


def _peers() -> dict[str, Any]:
    rows = []
    for source_index, source in enumerate(DEVICES):
        for destination_index, destination in enumerate(DEVICES):
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
        "ownership": _ownership(DEVICES, 1),
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
    result = qualification._scenario("1x2", DEVICES[-2:], tmp_path, {}, 0.1)
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
