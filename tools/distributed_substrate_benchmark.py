#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run synchronized execution-substrate benchmarks on the active JAX runtime."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from phydrax.execution import (
    ExecutionRuntime,
    global_weighted_mean,
    initialize_from_environment,
    launch_local_processes,
    LocalProcessLaunchPlan,
    shard_array_axis,
)
from phydrax.linalg import (
    DistributedKrylovPolicy,
    DistributedLinearOperator,
    DistributedPairing,
    solve_distributed_pcg,
)


def _time(operation, iterations: int) -> tuple[object, float]:
    value = operation()
    jax.block_until_ready(value)
    started = time.perf_counter()
    for _ in range(iterations):
        value = operation()
    jax.block_until_ready(value)
    elapsed = time.perf_counter() - started
    return value, elapsed / iterations


def run(size: int, iterations: int) -> dict[str, object]:
    if size <= 0 or iterations <= 0:
        raise ValueError("size and iterations must be positive")
    info = initialize_from_environment()
    runtime = ExecutionRuntime.current()
    device_count = runtime.root_group.spec.device_count
    group_token = np.asarray(
        int(runtime.root_group.spec.group_id[:16], 16),
        dtype=np.uint64,
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
    padded_size = ((size + device_count - 1) // device_count) * device_count
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
    solution_error = jnp.max(jnp.where(mask, jnp.abs(result.value - values), 0.0))
    jax.block_until_ready(solution_error)
    return {
        "runtime": {
            "process_count": info.process_count,
            "process_index": info.process_index,
            "global_device_count": info.global_device_count,
            "local_device_count": info.local_device_count,
            "platforms": list(info.platforms),
            "execution_group_id": runtime.root_group.spec.group_id,
        },
        "problem": {
            "logical_size": size,
            "group_identity_consistent": group_identity_consistent,
            "padded_size": padded_size,
            "iterations": iterations,
        },
        "weighted_mean": {
            "seconds": mean_seconds,
            "observed": float(np.asarray(mean)),
            "expected": float(expected_mean),
            "absolute_error": abs(float(np.asarray(mean)) - float(expected_mean)),
        },
        "distributed_pcg": {
            "seconds": pcg_seconds,
            "iterations": int(np.asarray(result.iterations)),
            "converged": bool(np.asarray(result.converged)),
            "maximum_solution_error": float(np.asarray(solution_error)),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1 << 18)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--local-processes", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.local_processes > 1 and not arguments.worker:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        environment = dict(os.environ)
        environment["PHYDRAX_CPU_COLLECTIVES"] = "gloo"
        results = launch_local_processes(
            (
                sys.executable,
                "-m",
                "tools.distributed_substrate_benchmark",
                "--size",
                str(arguments.size),
                "--iterations",
                str(arguments.iterations),
                "--worker",
            ),
            LocalProcessLaunchPlan(
                arguments.local_processes,
                f"127.0.0.1:{port}",
            ),
            environment=environment,
        )
        print(results[0].stdout.decode("utf-8").strip())
        return
    result = run(arguments.size, arguments.iterations)
    if jax.process_index() == 0:
        print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
