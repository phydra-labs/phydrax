#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark deterministic skeletal-muscle execution workset packing and vmap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax._fingerprint import array_tree_fingerprint
from phydrax.execution import (
    evaluate_execution_worksets_serial,
    evaluate_execution_worksets_vmap,
    ExecutionWorksetPlan,
    PoolExecutionSignature,
)


def _operation(signature, item, key, semantic_index):
    gain = 1.01 if signature.topology_id == "fast-motor-unit" else 0.99
    perturbation = 1.0e-4 * jax.random.normal(key, item.shape, dtype=item.dtype)
    return gain * item + perturbation + 0.0 * semantic_index.astype(item.dtype)


def _case(item_count: int, state_size: int, capacity: int, warmup: int, repeats: int):
    fast = PoolExecutionSignature(
        topology_id="fast-motor-unit",
        method_id="qualified-affine-update",
        precision_id="float32",
        backend_id=jax.default_backend(),
    )
    slow = PoolExecutionSignature(
        topology_id="slow-motor-unit",
        method_id="qualified-affine-update",
        precision_id="float32",
        backend_id=jax.default_backend(),
    )
    plan = ExecutionWorksetPlan(
        tuple(f"motor-unit-{index:06d}" for index in reversed(range(item_count))),
        tuple(fast if index % 2 else slow for index in reversed(range(item_count))),
        bucket_capacity=capacity,
    )
    prepared = plan.prepare()
    values = jnp.sin(
        jnp.arange(item_count * state_size, dtype=jnp.float32).reshape(
            (item_count, state_size)
        )
        / 31.0
    )
    counters = jnp.arange(item_count, dtype=jnp.uint32) % 17
    root_key = jax.random.key(1729)
    function = eqx.filter_jit(
        lambda state, key, count: evaluate_execution_worksets_vmap(
            prepared, _operation, state, key, count
        ).values
    )
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(values, root_key, counters),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(values, root_key, counters),
        warmup=warmup,
        repeats=repeats,
    )
    serial = evaluate_execution_worksets_serial(
        prepared, _operation, values, root_key, counters
    )
    compiler = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    serial_vmap_error = jnp.max(jnp.abs(serial.values - result))
    serial_vmap_tolerance = (
        8.0
        * jnp.finfo(result.dtype).eps
        * jnp.maximum(1.0, jnp.max(jnp.abs(serial.values)))
    )
    return {
        "dimensions": {
            "items": item_count,
            "state_size": state_size,
            "bucket_capacity": capacity,
            "bucket_count": prepared.bucket_count,
            "padded_lanes": prepared.bucket_count * capacity - item_count,
        },
        "identity": {
            "plan_id": plan.plan_id,
            "prepared_id": prepared.prepared_id,
        },
        "lower": {"seconds": compilation.lowering_seconds},
        "compile": {"seconds": compilation.compilation_seconds},
        "run": execution.to_milliseconds_dict(),
        "memory": {
            "logical_input_bytes": logical_array_bytes((values, counters)),
            "logical_output_bytes": logical_array_bytes(result),
            "compiler_argument_bytes": compiler.argument_bytes,
            "compiler_output_bytes": compiler.output_bytes,
            "compiler_temporary_bytes": compiler.temporary_bytes,
            "compiler_generated_code_bytes": compiler.generated_code_bytes,
        },
        "work": {
            "compiler_flops": compiler.flops,
            "compiler_bytes_accessed": compiler.bytes_accessed,
        },
        "certificate": {
            "serial_vmap_maximum_absolute_error": float(serial_vmap_error),
            "serial_vmap_tolerance": float(serial_vmap_tolerance),
            "serial_vmap_equivalent": bool(
                serial_vmap_error <= serial_vmap_tolerance
            ),
            "finite": bool(jnp.all(jnp.isfinite(result))),
            "exact_coverage": bool(serial.evidence.exact_coverage),
            "input_fingerprint": array_tree_fingerprint(values),
            "output_fingerprint": array_tree_fingerprint(result),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=1024)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--bucket-capacity", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.items < 1
        or arguments.state_size < 1
        or not 1 <= arguments.bucket_capacity <= 64
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError("Benchmark dimensions, capacity, warmup, and repeats are invalid.")
    case = _case(
        arguments.items,
        arguments.state_size,
        arguments.bucket_capacity,
        arguments.warmup,
        arguments.repeats,
    )
    devices = tuple(str(device) for device in jax.local_devices())
    payload = {
        "benchmark": "skeletal-muscle-execution-worksets",
        "environment": capture_environment().to_dict(),
        "distributed_gate": {
            "released": False,
            "reason": (
                "No distributed API is released: qualification requires at least two "
                "local JAX devices and no emulated fallback is permitted."
            ),
            "local_device_count": jax.local_device_count(),
            "local_devices": devices,
        },
        "case": case,
        "all_valid": bool(
            case["certificate"]["finite"]
            and case["certificate"]["exact_coverage"]
            and case["certificate"]["serial_vmap_equivalent"]
        ),
    }
    if arguments.output is None:
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    else:
        write_json_atomic(arguments.output, payload)
    if not payload["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
