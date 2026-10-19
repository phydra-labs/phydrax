#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Fail-closed qualification of one caller-provided, single-host CUDA MPI stack."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks._io import write_json_atomic
from benchmarks._runtime import DurationDistribution
from benchmarks.cuda_mpi import DEFAULT_SIZES, Operation, Precision, Route
from phydrax.interchange import (
    ExternalExecutionPolicy,
    ExternalRuntimeError,
    pin_executable,
    PinnedExecutable,
    PinnedFileOutputs,
    PinnedFileRequest,
    PinnedRunResult,
    run_pinned_command,
)


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/integration/test_cuda_direct_mpi.py"


def coverage_errors(data: bytes, rank_count: int) -> list[str]:
    """An optional-suite skip is a qualification failure, not successful coverage."""
    if rank_count < 2:
        raise ValueError("Qualification requires at least two ranks")
    cases = tuple(ET.fromstring(data).iter("testcase"))
    expected = {
        f"test_cuda_mpi_collectives[ranks-{rank_count}-{placement}-{route.value}-{precision.value}]"
        for placement in ("distinct-gpus", "shared-gpu")
        for route in Route
        for precision in Precision
    }
    observed = [case.attrib["name"] for case in cases]
    errors = []
    if set(observed) != expected or len(observed) != len(expected):
        errors.append(
            f"Required MPI cases differ: missing={sorted(expected.difference(observed))}, observed={sorted(observed)}"
        )
    for case in cases:
        for state in ("skipped", "failure", "error"):
            child = case.find(state)
            if child is not None:
                errors.append(
                    f"{case.attrib['name']}: {state}: {child.attrib.get('message', '')}"
                )
    return errors


def validate_record(
    record: dict[str, Any], rank_count: int, *, profile_capture: bool
) -> None:
    """Reject partial topology, ownership, point coverage, and timing evidence."""
    configuration = record["configuration"]
    scope = record["scope"]
    ranks = sorted(record["ranks"], key=lambda item: item["rank"])
    expected_ranks = list(range(rank_count))
    if (
        rank_count < 2
        or configuration["ranks"] != rank_count
        or scope["mpi_ranks"] != rank_count
        or [rank["rank"] for rank in ranks] != expected_ranks
        or record["passed"] is not True
    ):
        raise ValueError("Measurement must contain every requested successful rank")
    ownership = sorted(record["ownership"], key=lambda item: item["rank"])
    if (
        [item["rank"] for item in ownership] != expected_ranks
        or len({item["gpu_uuid"] for item in ownership}) != rank_count
        or any(not item["gpu_uuid"].startswith("GPU-") for item in ownership)
        or len({item["host"] for item in ownership}) != 1
        or scope["hosts"] != 1
        or scope["shared_memory_ranks"] != rank_count
        or scope["distinct_gpus"] != rank_count
    ):
        raise ValueError(
            "Measurement requires distinct physical GPUs in one shared-memory domain"
        )
    if configuration["profile_capture"] is not profile_capture:
        raise ValueError("Profiler-instrumented timings are not performance evidence")
    if configuration["repeats"] < 1:
        raise ValueError("Measurement requires a positive repeat count")
    expected_points = {
        (operation, precision, size)
        for operation in configuration["operation"]
        for precision in configuration["precision"]
        for size in configuration["sizes"]
    }
    if not expected_points:
        raise ValueError("Measurement must contain requested points")
    for rank in ranks:
        rows = {
            (row["operation"], row["precision"], row["message_bytes_per_rank"]): row
            for row in rank["rows"]
        }
        if len(rows) != len(rank["rows"]) or any(
            not row["passed"] for row in rows.values()
        ):
            raise ValueError("Duplicate or unsuccessful MPI measurement point")
        if rows.keys() != expected_points:
            raise ValueError("Ranks did not measure the complete requested MPI point set")
        for row in rows.values():
            samples = row["warm"]["samples_seconds"]
            if len(samples) != configuration["repeats"] or any(
                not math.isfinite(sample) or sample <= 0 for sample in samples
            ):
                raise ValueError("Rank samples must be positive, finite, and complete")


def aggregate_rows(record: dict[str, Any], rank_count: int) -> list[dict[str, Any]]:
    """Aggregate matching synchronized samples, not separately computed medians."""
    validate_record(record, rank_count, profile_capture=False)
    ranks = sorted(record["ranks"], key=lambda item: item["rank"])
    indexed = [
        {
            (row["operation"], row["precision"], row["message_bytes_per_rank"]): row
            for row in rank["rows"]
        }
        for rank in ranks
    ]
    result = []
    for key in sorted(indexed[0]):
        samples = [index[key]["warm"]["samples_seconds"] for index in indexed]
        distribution = DurationDistribution(
            tuple(max(pair) for pair in zip(*samples, strict=True))
        )
        median = distribution.median_seconds
        if median is None or median <= 0:
            raise ValueError("A positive synchronized median is required")
        result.append(
            {
                "operation": key[0],
                "precision": key[1],
                "message_bytes_per_rank": key[2],
                "rank_maximum_warm": distribution.to_dict(),
                "effective_payload_bytes_per_second": key[2] / median,
            }
        )
    return result


def _environment(args: argparse.Namespace) -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in (
            "HOME",
            "USER",
            "LOGNAME",
            "LANG",
            "TMPDIR",
            "NCCL_NVLS_ENABLE",
            "NCCL_LAUNCH_ORDER_IMPLICIT",
            "NCCL_LAUNCH_RACE_FATAL",
            "UCX_MEMTYPE_CACHE",
        )
        if key in os.environ
    }
    mpi = args.mpi_prefix.resolve()
    ucx = args.ucx_prefix.resolve()
    environment.update(
        {
            "PATH": os.pathsep.join((str(mpi / "bin"), str(ucx / "bin"), os.defpath)),
            "LD_LIBRARY_PATH": os.pathsep.join((str(mpi / "lib"), str(ucx / "lib"))),
            "PYTHONPATH": str(ROOT),
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": args.visible_devices,
            "PHYDRAX_MPI_RANKS": str(args.ranks),
            "JAX_ENABLE_X64": "true",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.25",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMPI_MCA_pml": "ucx",
            "UCX_TLS": "self,sm,cuda_copy,cuda_ipc",
            "UCX_LOG_LEVEL": "info",
            "UCX_PROTO_INFO": "y",
            "MPI4JAX_USE_CUDA_MPI": "1",
        }
    )
    launcher = _launcher(args, timeout=min(args.timeout, 240))
    environment["PHYDRAX_MPI_LAUNCHER"] = shlex.join(launcher)
    return environment


def _launcher(args: argparse.Namespace, *, timeout: int) -> list[str]:
    return [
        str(args.mpi_prefix.resolve() / "bin/mpirun"),
        "--allow-run-as-root",
        "--bind-to",
        "none",
        "--timeout",
        str(timeout),
    ]


def _run(
    launcher: PinnedExecutable,
    command: Sequence[str],
    environment: dict[str, str],
    evidence: dict[str, Any],
    output: Path,
    *,
    timeout: int,
    outputs: tuple[str, ...] = (),
    artifacts: PinnedFileOutputs | None = None,
) -> PinnedRunResult:
    policy = ExternalExecutionPolicy(
        inherit_environment=False,
        allowed_environment_variables=tuple(sorted(environment)),
    )
    try:
        result = run_pinned_command(
            launcher,
            command,
            inputs={},
            outputs=outputs,
            timeout=timeout,
            max_output_bytes=33554432,
            environment=environment,
            execution_policy=policy,
            artifacts=artifacts,
        )
    except ExternalRuntimeError as error:
        if error.result is None:
            raise
        result = error.result
    evidence["commands"].append(
        {
            "command": list(result.command),
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "elapsed_seconds": result.elapsed_seconds,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "error": result.error,
            "execution_policy_id": result.execution_policy_id,
            "execution_artifact_id": result.artifact.artifact_id,
            "files": [
                {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
                for item in result.file_artifacts
            ],
        }
    )
    if result.returncode != 0 or result.timed_out or result.error:
        evidence["errors"].append(f"Command failed: {shlex.join(command)}")
    write_json_atomic(output, evidence)
    return result


def _measure(
    args: argparse.Namespace,
    launcher: PinnedExecutable,
    environment: dict[str, str],
    evidence: dict[str, Any],
    route: Route,
) -> None:
    environment = {
        **environment,
        "MPI4JAX_USE_CUDA_MPI": "1" if route is Route.CUDA_DIRECT else "0",
    }
    command = [
        *_launcher(args, timeout=args.timeout),
        "-np",
        str(args.ranks),
        str(args.python.absolute()),
        "-m",
        "mpi4py",
        "-m",
        "benchmarks.cuda_mpi",
        "--ranks",
        str(args.ranks),
        "--route",
        route.value,
        "--sizes",
        *map(str, args.sizes),
        "--repeats",
        str(args.repeats),
        "--maximum-buffer-bytes",
        str(args.maximum_buffer_bytes),
        "--output",
        "measurement.json",
    ]
    result = _run(
        launcher,
        command,
        environment,
        evidence,
        args.output,
        timeout=args.timeout + 30,
        outputs=("measurement.json",),
    )
    if evidence["errors"]:
        return
    record = json.loads(result.output("measurement.json"))
    try:
        _validate_workload(record, args, route, profile_capture=False)
        aggregate = aggregate_rows(record, args.ranks)
    except ValueError as error:
        evidence["errors"].append(str(error))
    else:
        evidence["measurements"][route.value] = {
            "record": record,
            "aggregate": aggregate,
        }
    write_json_atomic(args.output, evidence)


def _validate_workload(
    record: dict[str, Any],
    args: argparse.Namespace,
    route: Route,
    *,
    profile_capture: bool,
    operation: Operation | None = None,
) -> None:
    configuration = record["configuration"]
    expected = {
        "route": route.value,
        "sizes": [max(args.sizes)] if profile_capture else sorted(args.sizes),
        "precision": ["float64"]
        if profile_capture
        else [value.value for value in Precision],
        "operation": [operation.value]
        if operation is not None
        else [value.value for value in Operation],
        "repeats": 3 if profile_capture else args.repeats,
        "maximum_buffer_bytes": args.maximum_buffer_bytes,
    }
    if any(configuration[key] != value for key, value in expected.items()):
        raise ValueError("Measurement configuration differs from the requested workload")
    validate_record(record, args.ranks, profile_capture=profile_capture)


def _profile(
    args: argparse.Namespace,
    launcher: PinnedExecutable,
    environment: dict[str, str],
    evidence: dict[str, Any],
    route: Route,
    operation: Operation,
) -> None:
    environment = {
        **environment,
        "MPI4JAX_USE_CUDA_MPI": "1" if route is Route.CUDA_DIRECT else "0",
    }
    destination = (
        args.output.absolute().parent
        / (args.output.stem + "-traces")
        / route.value
        / operation.value
    )
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = PinnedFileOutputs(
        str(destination),
        tuple(
            PinnedFileRequest(f"rank-{rank}.nsys-rep", 536870912)
            for rank in range(args.ranks)
        ),
        maximum_total_bytes=536870912 * args.ranks,
    )
    command = [
        *_launcher(args, timeout=args.timeout),
        "-np",
        str(args.ranks),
        str(args.nsys.resolve()),
        "profile",
        "--trace=cuda,mpi",
        "--mpi-impl=openmpi",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--output=rank-%q{OMPI_COMM_WORLD_RANK}",
        str(args.python.absolute()),
        "-m",
        "mpi4py",
        "-m",
        "benchmarks.cuda_mpi",
        "--ranks",
        str(args.ranks),
        "--route",
        route.value,
        "--sizes",
        str(max(args.sizes)),
        "--precision",
        "float64",
        "--operation",
        operation.value,
        "--repeats",
        "3",
        "--warmup",
        "2",
        "--capture",
        "--maximum-buffer-bytes",
        str(args.maximum_buffer_bytes),
        "--output",
        "profile-workload.json",
    ]
    result = _run(
        launcher,
        command,
        environment,
        evidence,
        args.output,
        timeout=args.timeout + 30,
        outputs=("profile-workload.json",),
        artifacts=artifacts,
    )
    if evidence["errors"]:
        return
    record = json.loads(result.output("profile-workload.json"))
    try:
        _validate_workload(record, args, route, profile_capture=True, operation=operation)
    except ValueError as error:
        evidence["errors"].append(str(error))
        write_json_atomic(args.output, evidence)
        return
    evidence["profiles"].append(
        {
            "route": route.value,
            "operation": operation.value,
            "directory": str(destination),
            "record": record,
        }
    )
    write_json_atomic(args.output, evidence)


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.timeout < 1 or args.repeats < 1:
        parser.error("timeout and repeats must be positive")
    if not 1 <= args.maximum_buffer_bytes <= 2147483647 or any(
        size < 8 or size % 8 or size > args.maximum_buffer_bytes for size in args.sizes
    ):
        parser.error("sizes must be multiples of eight within the explicit buffer budget")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("message sizes must not repeat")
    devices = [device.strip() for device in args.visible_devices.split(",")]
    if (
        len(devices) < 2
        or any(not device for device in devices)
        or len(set(devices)) != len(devices)
    ):
        parser.error("declare at least two distinct CUDA-visible devices")
    args.visible_devices = ",".join(devices)
    args.ranks = len(devices)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--mpi-prefix", type=Path, required=True)
    parser.add_argument("--ucx-prefix", type=Path, required=True)
    parser.add_argument("--visible-devices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nsys", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--maximum-buffer-bytes", type=int, default=67108864)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    env_path = shutil.which("env")
    if env_path is None:
        parser.error("GNU env is required as the pinned process launcher")
    # Immutable host preparation only. Pin env rather than copying a path-sensitive
    # virtual-environment interpreter or MPI launcher into a private directory.
    version = subprocess.run(
        [env_path, "--version"], capture_output=True, text=True, timeout=10, check=True
    ).stdout.splitlines()[0]
    launcher = pin_executable(
        env_path,
        version=version,
        license_id="GPL-3.0-or-later",
        source_url="https://www.gnu.org/software/coreutils/",
    )
    environment = _environment(args)
    evidence: dict[str, Any] = {
        "scope": f"single-host, {args.ranks} distinct GPUs; CUDA awareness does not imply zero host copies or network RDMA",
        "requested_ranks": args.ranks,
        "environment": environment,
        "commands": [],
        "errors": [],
        "measurements": {},
        "profiles": [],
        "successful": False,
    }
    write_json_atomic(args.output, evidence)
    for command in (
        [str(args.mpi_prefix.resolve() / "bin/mpirun"), "--version"],
        [str(args.ucx_prefix.resolve() / "bin/ucx_info"), "-v"],
        [str(args.ucx_prefix.resolve() / "bin/ucx_info"), "-d"],
    ):
        _run(launcher, command, environment, evidence, args.output, timeout=30)
        if evidence["errors"]:
            return 1
    result = _run(
        launcher,
        [
            str(args.python.absolute()),
            "-m",
            "pytest",
            "-n0",
            "-q",
            "-rs",
            str(TEST) + "::test_cuda_mpi_collectives",
            "--junitxml=tests.xml",
        ],
        environment,
        evidence,
        args.output,
        timeout=args.timeout,
        outputs=("tests.xml",),
    )
    if evidence["errors"]:
        return 1
    evidence["errors"].extend(coverage_errors(result.output("tests.xml"), args.ranks))
    write_json_atomic(args.output, evidence)
    if evidence["errors"]:
        return 1
    for route in Route:
        _measure(args, launcher, environment, evidence, route)
        if evidence["errors"]:
            return 1
    if args.nsys is not None:
        _run(
            launcher,
            [str(args.nsys.resolve()), "--version"],
            environment,
            evidence,
            args.output,
            timeout=30,
        )
        if evidence["errors"]:
            return 1
        for route in Route:
            for operation in Operation:
                _profile(args, launcher, environment, evidence, route, operation)
                if evidence["errors"]:
                    return 1
    evidence["successful"] = not evidence["errors"]
    write_json_atomic(args.output, evidence)
    return 0 if evidence["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
