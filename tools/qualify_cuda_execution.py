# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fail-closed, single-host physical CUDA peer and native-collective campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from tests._support.cuda_execution_worker import (
    consumer_evidence,
    consumer_oracle,
    CONSUMER_ROUTES,
    CONSUMER_SIZES,
    SIZES,
    STEPS,
)
from tools._accelerator_process import (
    CommandEvidence,
    read_json_evidence,
    run,
    write_json_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    "peer-copies",
    "jax-consumers",
    "1x1",
    "1x2",
    "1x4",
    "1x8",
    "2x4",
    "4x2",
    "8x1",
)


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
        if case in ("peer-copies", "jax-consumers")
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
    if not isinstance(rows, list) or len(rows) != len(devices):
        return ["Missing or duplicate physical device ownership"]
    if any(
        not isinstance(row, dict)
        or type(row.get("rank")) is not int
        or type(row.get("local_ordinal")) is not int
        or not isinstance(row.get("gpu_uuid"), str)
        or not isinstance(row.get("platform"), str)
        for row in rows
    ):
        return ["Malformed physical device ownership evidence"]
    errors: list[str] = []
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


def _peer_connectivity_errors(
    record: dict[str, Any], devices: Sequence[str]
) -> list[str]:
    expected = {
        (source, destination)
        for source in devices
        for destination in devices
        if source != destination
    }
    rows = record.get("peer_connectivity")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return ["Missing or duplicate directed peer connectivity evidence"]
    observed = set()
    for row in rows:
        if not isinstance(row, dict):
            return ["Malformed directed peer connectivity evidence"]
        source = row.get("source_uuid")
        destination = row.get("destination_uuid")
        if not isinstance(source, str) or not isinstance(destination, str):
            return ["Malformed directed peer connectivity identities"]
        pair = (source, destination)
        if pair not in expected or pair in observed:
            return ["Unexpected or duplicate directed peer connectivity pair"]
        observed.add(pair)
        if row.get("capability_reported") is not True:
            return ["Direct peer capability is unavailable or unproven"]
    return []


def peer_errors(record: dict[str, Any], devices: Sequence[str]) -> list[str]:
    errors = _ownership_errors(record["ownership"], devices, 1, len(devices))
    errors.extend(_peer_connectivity_errors(record, devices))
    if len(devices) == 1 and record.get("peer_matrix_status") != "not-applicable":
        errors.append("Single-device peer matrix must be explicitly not-applicable")
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
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("route"), str)
        or not isinstance(row.get("source_uuid"), str)
        or not isinstance(row.get("destination_uuid"), str)
        or type(row.get("bytes")) is not int
        or type(row.get("step")) is not int
        or type(row.get("ingress_mismatch_count")) is not int
        or type(row.get("mismatch_count")) is not int
        for row in rows
    ):
        errors.append("Malformed directed peer-copy identity or count evidence")
        return errors
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
    if not isinstance(coverage, dict) or any(
        type(coverage.get(key)) is not int for key in required_counts
    ):
        errors.append("Malformed peer-copy coverage counts")
    elif any(coverage[key] != count for key, count in required_counts.items()):
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


def consumer_errors(record: dict[str, Any], devices: Sequence[str]) -> list[str]:
    errors = _ownership_errors(record["ownership"], devices, 1, len(devices))
    if len(devices) == 1 and record.get("peer_matrix_status") != "not-applicable":
        errors.append("Single-device consumer matrix must be explicitly not-applicable")
    expected = {
        (route, source, destination, size, step)
        for route in CONSUMER_ROUTES
        for source in devices
        for destination in devices
        if source != destination
        for size in CONSUMER_SIZES
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
        errors.append("Missing, duplicate, or narrowed directed consumer points")
    if (
        record["sizes_bytes"] != list(CONSUMER_SIZES)
        or record["steps"] != list(STEPS)
        or record["routes"] != list(CONSUMER_ROUTES)
    ):
        errors.append("Consumer route/size/step contract differs")
    if any(type(row["bytes"]) is not int or type(row["step"]) is not int for row in rows):
        errors.append("Consumer size and step must be integer identities")
    counts = {
        "expected_consumer_observations": len(expected),
        "recorded_consumer_observations": len(rows),
        "passed_consumer_observations": sum(row["passed"] is True for row in rows),
    }
    if any(
        type(record["coverage"][key]) is not int or record["coverage"][key] != value
        for key, value in counts.items()
    ):
        errors.append("Consumer coverage counts differ from authoritative observations")
    for identity, row in zip(actual, rows, strict=True):
        if identity not in expected:
            continue
        _, source, destination, size, step = identity
        values = consumer_oracle(
            devices.index(source), devices.index(destination), size, step
        )
        oracle = consumer_evidence(values, values, values, values * 3 + 7, 0, 0)
        # Validate every diagnostic, not merely a producer's success flag.
        if any(
            type(row[key]) is not type(value) or row[key] != value
            for key, value in oracle.items()
        ):
            errors.append(f"Incorrect or malformed consumer evidence: {identity}")
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
            consumer_errors(record, devices)
            if scenario == "jax-consumers"
            else peer_errors(record, devices)
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
    *,
    python: Path,
    harness: Path,
) -> ScenarioEvidence:
    output = directory / f"{scenario}.json"
    from tools.qualify_cuda_wheels import installed_command

    selected = (
        devices
        if scenario in ("peer-copies", "jax-consumers")
        else devices[-math.prod(int(value) for value in scenario.split("x")) :]
    )
    child_environment = dict(environment, CUDA_VISIBLE_DEVICES=",".join(selected))
    if scenario in ("peer-copies", "jax-consumers"):
        command = [
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
    completed = run(
        installed_command(python, harness, command[0], command[1:]),
        cwd=harness,
        environment=child_environment,
        timeout=timeout,
    )
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
    parser.add_argument(
        "--python", type=Path, required=True, help="Installed interpreter"
    )
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args(argv)
    from tools.qualify_cuda_wheels import validate_work_path

    args.work_root = args.work_root.absolute()
    if sys.platform == "win32":
        args.work_root = validate_work_path(args.work_root, Path.home())
    args.python = validate_work_path(args.python, args.work_root)
    args.output = validate_work_path(args.output, args.work_root)
    if not args.python.is_file():
        parser.error("installed interpreter does not exist")
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
            "consumer_sizes_bytes": list(CONSUMER_SIZES),
            "consumer_routes": list(CONSUMER_ROUTES),
        },
        "environment": {
            key: value
            for key, value in sorted(environment.items())
            if key.startswith(("JAX_", "XLA_", "NCCL_", "UCX_", "CUDA_", "PHYDRAX_"))
        },
    }
    try:
        validate_selection(args.required_gpus, args.devices, args.scenarios)
        if sys.platform == "win32" and args.required_gpus != 1:
            raise ValueError("Windows CUDA qualification admits only one physical GPU")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        errors = environment_errors(environment)
        if errors:
            raise ValueError("; ".join(errors))
        from benchmarks._runtime import benchmark_driver_fingerprint

        record["identity"] = {
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
            PYTHONSAFEPATH="1",
            PYTHONNOUSERSITE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        environment.pop("PYTHONPATH", None)
        environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        for key, child in (
            ("TEMP", "tmp"),
            ("TMP", "tmp"),
            ("CUDA_CACHE_PATH", "cuda-cache"),
            ("JAX_COMPILATION_CACHE_DIR", "jax-cache"),
            ("QUDA_RESOURCE_PATH", "quda-cache"),
        ):
            path = validate_work_path(args.work_root / child, args.work_root)
            path.mkdir(exist_ok=True)
            environment[key] = str(path)
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
        from tools.qualify_cuda_wheels import copy_harness, INSTALLED_BOOTSTRAP

        harness = directory / "harness"
        record["harness_sha256"] = copy_harness(harness)
        record["installed_python"] = str(args.python)
        environment["PHYDRAX_QUALIFICATION_HARNESS"] = str(harness)
        identity_command = run(
            [
                str(args.python),
                "-I",
                "-c",
                INSTALLED_BOOTSTRAP
                + "\nfrom benchmarks._runtime import installed_build_fingerprint\n"
                + 'print("INSTALLED_BUILD_FINGERPRINT " + installed_build_fingerprint(), flush=True)\n',
                str(harness),
            ],
            cwd=harness,
            environment=environment,
            timeout=args.timeout,
        )
        record["installed_build_identity_command"] = identity_command
        fingerprints = [
            line.removeprefix("INSTALLED_BUILD_FINGERPRINT ")
            for line in identity_command["stdout"].splitlines()
            if line.startswith("INSTALLED_BUILD_FINGERPRINT ")
        ]
        if (
            identity_command["returncode"] != 0
            or identity_command["timed_out"]
            or len(fingerprints) != 1
            or len(fingerprints[0]) != 64
            or any(character not in "0123456789abcdef" for character in fingerprints[0])
        ):
            raise ValueError("Missing or invalid installed build identity evidence")
        record["identity"]["installed_build_fingerprint"] = fingerprints[0]
        record["host"] = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        }
        record["host_commands"] = [
            run(command, cwd=harness, environment=environment, timeout=args.timeout)
            for command in [
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,pci.bus_id,name,driver_version,memory.total,memory.used",
                    "--format=csv",
                ],
                *(
                    [["nvidia-smi", "topo", "-m"], ["lscpu", "--json"]]
                    if sys.platform != "win32"
                    else []
                ),
            ]
        ]
        if any(
            command["returncode"] or command["timed_out"]
            for command in record["host_commands"]
        ):
            record["errors"].append("Required physical host inventory command failed")
        for scenario in required_scenarios(args.required_gpus):
            result = _scenario(
                scenario,
                args.devices,
                directory,
                environment,
                args.timeout,
                python=args.python,
                harness=harness,
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
