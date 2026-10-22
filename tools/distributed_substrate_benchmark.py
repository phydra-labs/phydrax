#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Smoke-test native JAX bootstrap, collectives, and PCG without MPI.

Four-GPU process ownership campaigns use ``--local-processes`` and
``--devices-per-process`` as 1x4, 2x2, or 4x1, not spectral mesh shapes.
For AMD hardware proof also require ``--expected-vendor amd``,
``--expected-global-devices 4``, and the corresponding
``--expected-local-devices``. A compatible ROCm runtime must be provisioned
before launch; this tool never substitutes CPU evidence for requested hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from benchmarks._runtime import capture_environment
from phydrax import config as pxconfig
from phydrax._execution_device import device_vendor
from phydrax.execution import (
    DistributionMode,
    ExecutionCandidate,
    ExecutionPolicy,
    ExecutionRequirements,
    ExecutionRuntime,
    global_weighted_mean,
    launch_local_processes,
    LocalProcessLaunchError,
    LocalProcessLaunchPlan,
    LocalProcessResult,
    resolve_execution_plan,
    ResourceRequest,
    runtime_info,
    shard_array_axis,
)
from phydrax.linalg import (
    DistributedKrylovPolicy,
    DistributedLinearOperator,
    DistributedPairing,
    solve_distributed_pcg,
)
from phydrax.linalg.krylov import KrylovBreakdownStatus
from tools._accelerator_process import write_json_evidence


RANK_RESULT_PREFIX = b"SUBSTRATE_RESULT "


def _terminal_record(stdout: bytes) -> dict[str, Any]:
    records = [
        line[len(RANK_RESULT_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(RANK_RESULT_PREFIX)
    ]
    if len(records) != 1:
        raise ValueError("each rank must emit exactly one terminal result")
    record = json.loads(records[0])
    if not isinstance(record, dict):
        raise ValueError("rank record must be an object")
    return record


def qualification_errors(record: dict[str, Any]) -> list[str]:
    """Scientific success is independent of whether an executable returned."""
    errors = []
    for name, metric, tolerance in (
        ("weighted_mean", "absolute_error", 1.0e-12),
        ("distributed_pcg", "maximum_solution_error", 1.0e-12),
    ):
        value = record[name][metric]
        seconds = record[name]["seconds"]
        if not math.isfinite(value) or value > tolerance or value < 0:
            errors.append(f"{name}.{metric} exceeds finite tolerance {tolerance}")
        if not math.isfinite(seconds) or seconds < 0:
            errors.append(f"{name}.seconds must be finite and nonnegative")
    if record["distributed_pcg"]["converged"] is not True:
        errors.append("distributed_pcg did not converge")
    pcg = record["distributed_pcg"]
    if (
        type(pcg["breakdown"]) is not int
        or pcg["breakdown"]
        not in (int(KrylovBreakdownStatus.NONE), int(KrylovBreakdownStatus.HAPPY))
        or not math.isfinite(pcg["residual_norm"])
        or not math.isfinite(pcg["residual_absolute_tolerance"])
        or pcg["residual_absolute_tolerance"] < 0
        or not 0 <= pcg["residual_norm"] <= pcg["residual_absolute_tolerance"]
    ):
        errors.append("distributed_pcg residual or breakdown gate failed")
    if not all(
        math.isfinite(record["weighted_mean"][key]) for key in ("observed", "expected")
    ):
        errors.append("weighted_mean values must be finite")
    if record["problem"]["group_identity_consistent"] is not True:
        errors.append("execution-group identity differs across ranks")
    return errors


def _json_safe(value: Any) -> Any:
    """Keep failed numerical evidence serializable without nonstandard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _time(operation: Any, iterations: int) -> tuple[object, float]:
    value = operation()
    jax.block_until_ready(value)
    started = time.perf_counter()
    for _ in range(iterations):
        value = operation()
    jax.block_until_ready(value)
    elapsed = time.perf_counter() - started
    return value, elapsed / iterations


def _validate_runtime(
    devices: Sequence[jax.Device],
    local_devices: Sequence[jax.Device],
    *,
    expected_vendor: str | None,
    expected_global_devices: int | None,
    expected_local_devices: int | None,
) -> None:
    for name, expected, observed in (
        ("global", expected_global_devices, len(devices)),
        ("local", expected_local_devices, len(local_devices)),
    ):
        if expected is not None:
            if expected <= 0:
                raise ValueError(f"expected {name} device count must be positive")
            if observed != expected:
                raise RuntimeError(
                    f"expected {name} device count {expected}, observed {observed}"
                )
    if expected_vendor is not None:
        if not expected_vendor.strip():
            raise ValueError("expected vendor must be non-empty")
        vendors = tuple(device_vendor(device) for device in devices)
        if not vendors or any(vendor != expected_vendor for vendor in vendors):
            raise RuntimeError(
                f"expected accelerator vendor {expected_vendor!r}, observed {vendors}"
            )


def run(
    size: int,
    iterations: int,
    *,
    expected_vendor: str | None = None,
    expected_global_devices: int | None = None,
    expected_local_devices: int | None = None,
) -> dict[str, object]:
    if size <= 0 or iterations <= 0:
        raise ValueError("size and iterations must be positive")
    info = runtime_info()
    _validate_runtime(
        jax.devices(),
        jax.local_devices(),
        expected_vendor=expected_vendor,
        expected_global_devices=expected_global_devices,
        expected_local_devices=expected_local_devices,
    )
    runtime = ExecutionRuntime.current()
    device_count = runtime.root_group.spec.device_count
    group_token = np.frombuffer(
        bytes.fromhex(runtime.root_group.spec.group_id), dtype=np.uint8
    )
    group_tokens = np.asarray(multihost_utils.process_allgather(group_token, tiled=False))
    group_identity_consistent = bool(np.all(group_tokens == group_tokens[0]))
    if not group_identity_consistent:
        raise RuntimeError(
            "execution-group identity differs across JAX processes: "
            f"local={runtime.root_group.spec.group_id}; "
            f"tokens={group_tokens.tolist()}; "
            f"inventory={runtime.inventory.to_payload()}"
        )
    local_hardware_ids = np.asarray(
        [device.local_hardware_id for device in jax.local_devices()], dtype=np.int64
    )
    hardware_ids = np.asarray(
        multihost_utils.process_allgather(local_hardware_ids, tiled=False)
    ).reshape(info.process_count, -1)
    padded_size = ((size + device_count - 1) // device_count) * device_count
    requirements = ExecutionRequirements("substrate-smoke")
    request = ResourceRequest(
        cpu_cores=1,
        memory_bytes=16 * padded_size * np.dtype(np.float64).itemsize,
        accelerator_count=sum(
            device.platform == "gpu" for device in runtime.inventory.devices
        ),
        host_count=1,
        process_count=info.process_count,
        accelerator_vendor=expected_vendor,
    )
    admitted = resolve_execution_plan(
        ExecutionPolicy(DistributionMode.AUTO, resources=request),
        runtime.inventory,
        (
            ExecutionCandidate(
                "substrate-smoke",
                requirements.requirements_id,
                runtime.root_group.spec,
                estimated_memory_bytes=request.memory_bytes,
            ),
        ),
        precision_policy_id="substrate-smoke-float64",
        solver_policy_id="substrate-smoke-diagonal-pcg",
    )
    if admitted.group is None:
        raise RuntimeError("resource admission omitted the execution group")
    host = np.linspace(0.0, 1.0, padded_size, dtype=np.float64)
    valid = np.arange(padded_size) < size
    values = shard_array_axis(host, runtime.root_group)
    mask = shard_array_axis(valid, runtime.root_group)

    mean, mean_seconds = _time(
        lambda: global_weighted_mean(values, mask=mask),
        iterations,
    )
    expected_mean = np.mean(host[:size])

    operator = DistributedLinearOperator(
        lambda value: 2.0 * value,
        lambda value: 2.0 * value,
        (padded_size,),
        (padded_size,),
        operator_id="benchmark-diagonal-two",
    )
    pairing = DistributedPairing(mask, pairing_id="benchmark-owned-mask")
    right_hand_side = shard_array_axis(
        np.where(valid, 2.0 * host, 0.0),
        runtime.root_group,
    )
    policy = DistributedKrylovPolicy(8, relative_tolerance=1.0e-12)
    result, pcg_seconds = _time(
        lambda: solve_distributed_pcg(
            operator,
            right_hand_side,
            pairing,
            policy,
        ),
        iterations,
    )
    # ty: ignore[unresolved-attribute]
    solution_error = jnp.max(jnp.where(mask, jnp.abs(result.value - values), 0.0))
    jax.block_until_ready(solution_error)
    return {
        "environment": capture_environment().to_dict(),
        "runtime": {
            "process_count": info.process_count,
            "process_index": info.process_index,
            "global_device_count": info.global_device_count,
            "local_device_count": info.local_device_count,
            "platforms": list(info.platforms),
            "execution_group_id": runtime.root_group.spec.group_id,
            "local_hardware_ids_by_process": hardware_ids.tolist(),
            "nccl_nvls_enable": os.environ.get("NCCL_NVLS_ENABLE"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "nccl_launch_order_implicit": os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT"),
            "nccl_launch_race_fatal": os.environ.get("NCCL_LAUNCH_RACE_FATAL"),
            "inventory_id": runtime.inventory.inventory_id,
            "process_host_ids": [
                list(record) for record in runtime.inventory.process_host_ids
            ],
        },
        "problem": {
            "logical_size": size,
            "group_identity_consistent": group_identity_consistent,
            "padded_size": padded_size,
            "iterations": iterations,
            "qualification": "bootstrap-and-collective-smoke",
        },
        "resource_admission": {
            "admitted": True,
            "request": request.to_payload(),
            "execution_plan_id": admitted.execution_plan_id,
            "group_id": admitted.group.group_id,
            "memory_envelope": request.memory_envelope,
        },
        "weighted_mean": {
            "seconds": mean_seconds,
            "observed": float(np.asarray(mean)),
            "expected": float(expected_mean),
            "absolute_error": abs(float(np.asarray(mean)) - float(expected_mean)),
        },
        "distributed_pcg": {
            "seconds": pcg_seconds,
            # ty: ignore[unresolved-attribute]
            "iterations": int(np.asarray(result.iterations)),
            # ty: ignore[unresolved-attribute]
            "converged": bool(np.asarray(result.converged)),
            # ty: ignore[unresolved-attribute]
            "breakdown": int(np.asarray(result.breakdown)),
            # ty: ignore[unresolved-attribute]
            "residual_norm": float(np.asarray(result.residual_norm)),
            "residual_absolute_tolerance": 1e-12 * float(np.linalg.norm(2 * host[:size])),
            "maximum_solution_error": float(np.asarray(solution_error)),
        },
    }


def _launch_ranks(
    arguments: argparse.Namespace,
) -> tuple[tuple[LocalProcessResult, ...], str | None]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "tools.distributed_substrate_benchmark",
        "--size",
        str(arguments.size),
        "--iterations",
        str(arguments.iterations),
        "--local-processes",
        str(arguments.local_processes),
        "--devices-per-process",
        str(arguments.devices_per_process),
        "--worker",
    ]
    if arguments.required_gpus is not None:
        command.extend(("--required-gpus", str(arguments.required_gpus)))
    for option, value in (
        ("--expected-vendor", arguments.expected_vendor),
        ("--expected-global-devices", arguments.expected_global_devices),
        ("--expected-local-devices", arguments.expected_local_devices),
    ):
        if value is not None:
            command.extend((option, str(value)))
    try:
        results = launch_local_processes(
            command,
            LocalProcessLaunchPlan(
                arguments.local_processes,
                f"127.0.0.1:{port}",
                devices_per_process=arguments.devices_per_process,
            ),
            environment=pxconfig.environment(
                pxconfig.RuntimeConfig(cpu_collectives_implementation="gloo")
            ),
        )
    except LocalProcessLaunchError as error:
        return error.results, str(error)
    return results, None


def _rank_record(
    results: Sequence[LocalProcessResult],
    process_count: int,
    launch_error: str | None,
) -> tuple[dict[str, Any], list[str]]:
    ranks = []
    group_ids = []
    errors = [] if launch_error is None else [launch_error]
    if sorted(result.process_id for result in results) != list(range(process_count)):
        errors.append("supervised rank identities differ from requested topology")
    for result in results:
        try:
            rank = _terminal_record(result.stdout)
            runtime = rank["runtime"]
            if (
                type(runtime["process_index"]) is not int
                or runtime["process_index"] != result.process_id
            ):
                errors.append(f"rank {result.process_id} identity differs")
            if runtime["process_count"] != process_count:
                errors.append(f"rank {result.process_id} process count differs")
            group_id = runtime["execution_group_id"]
            if not isinstance(group_id, str) or not group_id:
                raise ValueError("execution group identity must be nonempty")
            group_ids.append(group_id)
            ranks.append(rank)
            if rank["passed"] is not True or rank["errors"] != []:
                errors.append(f"rank {result.process_id} scientific gate failed")
            errors.extend(
                f"rank {result.process_id}: {error}"
                for error in qualification_errors(rank)
            )
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            errors.append(f"rank {result.process_id} terminal record: {error}")
        if result.returncode != 0:
            errors.append(f"rank {result.process_id} exited {result.returncode}")
        if result.output_truncated:
            errors.append(f"rank {result.process_id} output was truncated")
    if len(ranks) != process_count:
        errors.append("missing terminal ranks")
    if len(set(group_ids)) != 1:
        errors.append("terminal rank groups differ")
    record = dict(ranks[0]) if ranks else {}
    record.update(ranks=ranks, passed=not errors, errors=errors)
    record["rank_outputs"] = [
        {
            "rank": result.process_id,
            "returncode": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "output_truncated": result.output_truncated,
        }
        for result in results
    ]
    return record, errors


def _local_record(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    record = run(
        arguments.size,
        arguments.iterations,
        expected_vendor=arguments.expected_vendor,
        expected_global_devices=arguments.expected_global_devices,
        expected_local_devices=arguments.expected_local_devices,
    )
    errors = qualification_errors(record)
    if jax.process_count() != arguments.local_processes:
        errors.append("observed process count differs from requested topology")
    if jax.local_device_count() != arguments.devices_per_process:
        errors.append("observed local devices differ from requested topology")
    if arguments.required_gpus is not None:
        from tests._support.cuda_execution_worker import cuda_runtime, ownership

        if any(device_vendor(device) != "nvidia" for device in jax.devices()):
            errors.append("physical CUDA qualification requires NVIDIA devices")
        else:
            record["ownership"] = ownership(cuda_runtime())
        if jax.device_count() != arguments.required_gpus:
            errors.append("observed GPUs differ from required-gpus")
    terminal = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(not errors, dtype=np.bool_), tiled=False
        )
    )
    if not np.all(terminal):
        errors.append("not every rank reached a successful scientific terminal state")
    record.update(passed=not errors, errors=errors)
    return record, errors


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1 << 18)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--local-processes", type=int, default=1)
    parser.add_argument(
        "--devices-per-process",
        type=int,
        default=1,
        help="Disjoint GPUs per local worker; forced CPU device counts are unchanged.",
    )
    parser.add_argument("--expected-vendor")
    parser.add_argument("--expected-global-devices", type=int)
    parser.add_argument("--expected-local-devices", type=int)
    parser.add_argument(
        "--required-gpus",
        type=int,
        help="Require NVIDIA CUDA GPU count and physical ownership evidence.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.local_processes <= 0 or arguments.devices_per_process <= 0:
        parser.error("local-processes and devices-per-process must be positive")
    for option, value in (
        ("--expected-vendor", arguments.expected_vendor),
        ("--expected-global-devices", arguments.expected_global_devices),
        ("--expected-local-devices", arguments.expected_local_devices),
    ):
        if value is not None:
            if isinstance(value, int) and value <= 0:
                parser.error(f"{option} must be positive")
            if isinstance(value, str) and not value.strip():
                parser.error(f"{option} must be non-empty")
    if arguments.required_gpus is not None and (
        arguments.required_gpus <= 0
        or arguments.required_gpus
        != arguments.local_processes * arguments.devices_per_process
    ):
        parser.error("required-gpus must equal the requested process/device product")
    if arguments.required_gpus is not None and arguments.expected_vendor not in (
        None,
        "nvidia",
    ):
        parser.error(
            "required-gpus requires expected-vendor nvidia, not AMD qualification"
        )
    if arguments.local_processes > 1 and not arguments.worker:
        results, launch_error = _launch_ranks(arguments)
        record, errors = _rank_record(results, arguments.local_processes, launch_error)
    else:
        local_devices = jax.local_devices()
        if arguments.worker and any(device.platform == "gpu" for device in local_devices):
            start = jax.process_index() * arguments.devices_per_process
            expected = list(range(start, start + arguments.devices_per_process))
            actual = [device.local_hardware_id for device in local_devices]
            if actual != expected:
                raise RuntimeError(
                    f"local worker GPU ownership differs: expected={expected}, actual={actual}"
                )
        record, errors = _local_record(arguments)
    safe_record = _json_safe(record)
    if arguments.output is not None and not arguments.worker:
        write_json_evidence(arguments.output, safe_record)
    encoded = json.dumps(safe_record, allow_nan=False, sort_keys=True)
    print(
        (RANK_RESULT_PREFIX.decode("ascii") if arguments.worker else "") + encoded,
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
