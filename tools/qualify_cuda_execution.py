# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fail-closed, single-host physical CUDA peer and native-collective campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from tests._support.cuda_execution_worker import SIZES, STEPS
from tools._accelerator_process import (
    CommandEvidence,
    read_json_evidence,
    run,
    write_json_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("peer-copies", "1x1", "1x2", "1x4", "1x8", "2x4", "4x2", "8x1")


class ScenarioEvidence(TypedDict):
    scenario: str
    devices: list[str]
    command: CommandEvidence
    result: dict[str, Any]
    errors: list[str]
    passed: bool


def required_scenarios(required_gpus: int) -> tuple[str, ...]:
    if required_gpus not in (1, 2, 4, 8):
        raise ValueError("required-gpus must be 1, 2, 4, or 8")
    return tuple(
        case
        for case in SCENARIOS
        if case == "peer-copies"
        or math.prod(int(value) for value in case.split("x")) <= required_gpus
    )


def validate_selection(
    required_gpus: int, devices: Sequence[str], scenarios: Sequence[str]
) -> None:
    if len(devices) != required_gpus or len(set(devices)) != len(devices):
        raise ValueError("Every required GPU must have one distinct explicit UUID")
    for device in devices:
        if not device.startswith("GPU-") or "GPU-" + str(uuid.UUID(device[4:])) != device:
            raise ValueError("devices must contain complete canonical physical GPU UUIDs")
    expected = required_scenarios(required_gpus)
    if len(set(scenarios)) != len(scenarios) or set(scenarios) != set(expected):
        raise ValueError(
            f"Missing, duplicate, or narrowed scenarios; required={expected}"
        )


def environment_errors(environment: Mapping[str, str]) -> list[str]:
    errors = []
    if "--xla_force_host_platform_device_count" in environment.get("XLA_FLAGS", ""):
        errors.append("Forced CPU topology is not physical CUDA qualification")
    if environment.get("JAX_NUM_CPU_DEVICES", "-1") != "-1":
        errors.append("Forced CPU device counts are not physical CUDA qualification")
    if environment.get("JAX_PLATFORM_NAME", "gpu") not in ("cuda", "gpu"):
        errors.append("A CPU platform override is not physical CUDA qualification")
    if environment.get("JAX_PLATFORMS", "cuda") not in ("cuda", "gpu"):
        errors.append("JAX_PLATFORMS must select only CUDA")
    for key in (
        "PHYDRAX_COORDINATOR_ADDRESS",
        "PHYDRAX_NUM_PROCESSES",
        "PHYDRAX_PROCESS_ID",
        "PHYDRAX_LOCAL_DEVICE_IDS",
    ):
        if key in environment:
            errors.append(f"Inherited launch topology is not admitted: {key}")
    return errors


def _ownership_errors(
    rows: list[dict[str, Any]], devices: Sequence[str], processes: int, local: int
) -> list[str]:
    if len(rows) != len(devices):
        return ["Missing or duplicate physical device ownership"]
    errors = []
    expected = [
        (rank, ordinal)
        for rank in range(processes)
        for ordinal in range(rank * local, (rank + 1) * local)
    ]
    actual = [(row["rank"], row["local_ordinal"]) for row in rows]
    if actual != expected or [row["gpu_uuid"] for row in rows] != list(devices):
        errors.append(
            "Rank/ordinal/UUID ownership differs from the explicit ordered GPU selection"
        )
    if any(row["platform"] != "gpu" for row in rows):
        errors.append("CPU topology cannot qualify physical CUDA execution")
    return errors


def peer_errors(record: dict[str, Any], devices: Sequence[str]) -> list[str]:
    errors = _ownership_errors(record["ownership"], devices, 1, len(devices))
    expected = {
        (route, source, destination, size, step)
        for route in ("cuda", "jax")
        for source in devices
        for destination in devices
        if source != destination
        for size in SIZES
        for step in STEPS
    }
    rows = record["rows"]
    actual = [
        (
            row["route"],
            row["source_uuid"],
            row["destination_uuid"],
            row["bytes"],
            row["step"],
        )
        for row in rows
    ]
    if set(actual) != expected or len(actual) != len(expected):
        errors.append("Missing, duplicate, or narrowed directed peer-copy points")
    coverage = record["coverage"]
    required_counts = {
        "expected_copy_observations": len(expected),
        "recorded_copy_observations": len(rows),
        "recorded_ingress_observations": len(rows),
        "passed_ingress_observations": sum(
            row["ingress_mismatch_count"] == 0 for row in rows
        ),
        "passed_copy_observations": sum(row["passed"] is True for row in rows),
    }
    if any(coverage[key] != count for key, count in required_counts.items()):
        errors.append(
            "Reported ingress/copy counts differ from actual recorded observations"
        )
    if any(
        row["ingress_mismatch_count"] != 0
        or row["ingress_sha256"] != row["expected_sha256"]
        for row in rows
    ):
        errors.append(
            "Incorrect source-local ingress contents; peer integrity cannot be certified"
        )
    for row in rows:
        if (
            row["passed"] is not True
            or row["mismatch_count"] != 0
            or row["expected_sha256"] != row["observed_sha256"]
            or row["failure_stage"] != "none"
        ):
            errors.append("Incorrect peer-copy contents")
            break
    # Distinct current inputs must be evidenced, not a repeated constant or stale buffer.
    for source_index, source in enumerate(devices):
        for destination_index, destination in enumerate(devices):
            if source == destination:
                continue
            for size in SIZES:
                from tests._support.cuda_execution_worker import pattern

                expected_hashes = {
                    step: hashlib.sha256(
                        pattern(source_index, destination_index, size, step).tobytes()
                    ).hexdigest()
                    for step in STEPS
                }
                selected = [
                    row
                    for row in rows
                    if row["source_uuid"] == source
                    and row["destination_uuid"] == destination
                    and row["bytes"] == size
                ]
                if any(
                    row["expected_sha256"] != expected_hashes[row["step"]]
                    for row in selected
                ):
                    errors.append(
                        "Peer-copy oracle does not match the requested current inputs"
                    )
    return errors


def collective_errors(
    record: dict[str, Any], scenario: str, devices: Sequence[str]
) -> list[str]:
    from phydrax.linalg.krylov import KrylovBreakdownStatus

    processes, local = (int(value) for value in scenario.split("x"))
    ranks = record["ranks"] if processes > 1 else [record]
    if len(ranks) != processes or [
        rank["runtime"]["process_index"] for rank in ranks
    ] != list(range(processes)):
        return ["Missing or duplicate terminal ranks"]
    errors = []
    group_ids = set()
    ownership = []
    for rank in ranks:
        runtime = rank["runtime"]
        group_ids.add(runtime["execution_group_id"])
        ownership.extend(rank["ownership"])
        if (
            runtime["process_count"] != processes
            or runtime["local_device_count"] != local
            or runtime["global_device_count"] != len(devices)
        ):
            errors.append(
                "Observed process/device topology differs from the requested scenario"
            )
        if rank["passed"] is not True or rank["errors"]:
            errors.append("A required rank did not reach successful terminal agreement")
        if rank["problem"]["logical_size"] != 17 or rank["problem"]["iterations"] != 2:
            errors.append("Narrowed collective workload")
        if rank["problem"]["group_identity_consistent"] is not True:
            errors.append("Collective group identity differs")
        for name, metric in (
            ("weighted_mean", "absolute_error"),
            ("distributed_pcg", "maximum_solution_error"),
        ):
            value = rank[name][metric]
            if (
                not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 1e-12
            ):
                errors.append(f"{name} failed its finite numerical tolerance")
        if rank["distributed_pcg"]["converged"] is not True:
            errors.append("Distributed PCG did not converge")
        pcg = rank["distributed_pcg"]
        residual = pcg["residual_norm"]
        bound = pcg["residual_absolute_tolerance"]
        if (
            type(pcg["breakdown"]) is not int
            or pcg["breakdown"]
            not in (int(KrylovBreakdownStatus.NONE), int(KrylovBreakdownStatus.HAPPY))
            or not isinstance(residual, (float, int))
            or not isinstance(bound, (float, int))
            or not math.isfinite(residual)
            or not math.isfinite(bound)
            or not 0 <= residual <= bound
        ):
            errors.append("Distributed PCG residual/breakdown gate failed")
        mean = rank["weighted_mean"]
        if not all(
            isinstance(mean[key], (float, int)) and math.isfinite(mean[key])
            for key in ("observed", "expected")
        ):
            errors.append("Weighted-mean values are not finite")
        if processes > 1:
            hosts = runtime["process_host_ids"]
            if [row[0] for row in hosts] != list(range(processes)) or len(
                {row[1] for row in hosts}
            ) != 1:
                errors.append("Ranks do not own one observed shared host")
    if len(group_ids) != 1:
        errors.append("Terminal rank group IDs differ")
    errors.extend(_ownership_errors(ownership, devices, processes, local))
    return errors


def scenario_errors(
    command: CommandEvidence,
    record: dict[str, Any],
    scenario: str,
    devices: Sequence[str],
) -> list[str]:
    errors = []
    if command["timed_out"] or command["returncode"] != 0:
        errors.append("Required command timed out or exited unsuccessfully")
    try:
        if record["passed"] is not True or record["errors"]:
            errors.append("Required scenario failed or was skipped")
        errors.extend(
            peer_errors(record, devices)
            if scenario == "peer-copies"
            else collective_errors(record, scenario, devices)
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"Incomplete or malformed scenario evidence: {error}")
    return errors


def _scenario(
    scenario: str,
    devices: list[str],
    directory: Path,
    environment: dict[str, str],
    timeout: float,
) -> ScenarioEvidence:
    output = directory / f"{scenario}.json"
    selected = (
        devices
        if scenario == "peer-copies"
        else devices[-math.prod(int(value) for value in scenario.split("x")) :]
    )
    child_environment = dict(environment, CUDA_VISIBLE_DEVICES=",".join(selected))
    if scenario == "peer-copies":
        command = [
            sys.executable,
            "-m",
            "tests._support.cuda_execution_worker",
            "--case",
            scenario,
            "--required-gpus",
            str(len(selected)),
            "--output",
            str(output),
        ]
    else:
        processes, local = scenario.split("x")
        command = [
            sys.executable,
            "-m",
            "tools.distributed_substrate_benchmark",
            "--local-processes",
            processes,
            "--devices-per-process",
            local,
            "--required-gpus",
            str(len(selected)),
            "--size",
            "17",
            "--iterations",
            "2",
            "--output",
            str(output),
        ]
    completed = run(command, cwd=ROOT, environment=child_environment, timeout=timeout)
    result: dict[str, Any] = {}
    errors = []
    try:
        result = read_json_evidence(output)
        json.dumps(result, allow_nan=False)
    except (OSError, ValueError) as error:
        result = {}
        errors.append(f"Missing authoritative result: {error}")
    errors.extend(scenario_errors(completed, result, scenario, selected))
    return {
        "scenario": scenario,
        "devices": selected,
        "command": completed,
        "result": result,
        "errors": errors,
        "passed": not errors,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-gpus", choices=(1, 2, 4, 8), type=int, required=True)
    parser.add_argument(
        "--devices", nargs="+", required=True, help="Ordered complete GPU UUIDs"
    )
    parser.add_argument("--scenarios", choices=SCENARIOS, nargs="+", required=True)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Whole-process-group seconds per scenario",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    record: dict[str, Any] = {
        "passed": False,
        "errors": [],
        "scenarios": [],
        "scope": "single-host physical CUDA correctness; not performance or RDMA qualification",
        "configuration": {
            "required_gpus": args.required_gpus,
            "devices": args.devices,
            "scenarios": args.scenarios,
            "sizes_bytes": list(SIZES),
            "steps": list(STEPS),
        },
        "environment": {
            key: value
            for key, value in sorted(environment.items())
            if key.startswith(("JAX_", "XLA_", "NCCL_", "UCX_", "CUDA_", "PHYDRAX_"))
        },
    }
    try:
        validate_selection(args.required_gpus, args.devices, args.scenarios)
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        errors = environment_errors(environment)
        if errors:
            raise ValueError("; ".join(errors))
        from benchmarks._runtime import (
            benchmark_driver_fingerprint,
            source_build_fingerprint,
        )

        record["identity"] = {
            "source_build_fingerprint": source_build_fingerprint(ROOT),
            "qualification_driver_fingerprint": benchmark_driver_fingerprint(
                ROOT, Path(__file__)
            ),
            "substrate_driver_fingerprint": benchmark_driver_fingerprint(
                ROOT, ROOT / "tools/distributed_substrate_benchmark.py"
            ),
        }
        environment.update(
            JAX_PLATFORMS="cuda",
            JAX_ENABLE_X64="true",
            PYTHONPATH=str(ROOT),
            PYTHONNOUSERSITE="1",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        record["effective_environment"] = {
            key: value
            for key, value in sorted(environment.items())
            if key.startswith(
                ("JAX_", "XLA_", "NCCL_", "UCX_", "CUDA_", "PHYDRAX_", "PYTHON")
            )
            or key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
        directory = Path(
            tempfile.mkdtemp(prefix=f"{args.output.stem}-", dir=args.output.parent)
        )
        record["host_commands"] = [
            run(command, cwd=ROOT, environment=environment, timeout=args.timeout)
            for command in (
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,pci.bus_id,name,driver_version,memory.total,memory.used",
                    "--format=csv",
                ],
                ["nvidia-smi", "topo", "-m"],
                ["lscpu", "--json"],
            )
        ]
        if any(
            command["returncode"] or command["timed_out"]
            for command in record["host_commands"]
        ):
            record["errors"].append("Required physical host inventory command failed")
        for scenario in required_scenarios(args.required_gpus):
            result = _scenario(
                scenario, args.devices, directory, environment, args.timeout
            )
            record["scenarios"].append(result)
            record["errors"].extend(f"{scenario}: {error}" for error in result["errors"])
        record["passed"] = not record["errors"]
    except Exception as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        write_json_evidence(args.output, record)
    print(
        json.dumps(
            {
                "passed": record["passed"],
                "output": str(args.output),
                "errors": record["errors"],
            }
        ),
        flush=True,
    )
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
