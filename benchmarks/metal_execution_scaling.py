"""Bounded, phase-separated native reduction, indexing, and contraction workloads.

Run unchanged through each installed wheel. Host references use the identical
rounded inputs; their preparation and all readback remain outside warm timings.
The ordinary-sum accuracy gate deliberately exposes the older serial reducer.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import config, ein


_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    capacity: int
    function: Callable[..., Array]
    inputs: tuple[NDArray[Any], ...]
    reference: NDArray[Any]
    absolute_tolerance: float
    relative_tolerance: float


def _sum(values: Array) -> Array:
    return jnp.sum(values, dtype=jnp.float32)


def _scatter(values: Array, index: Array, update: Array) -> Array:
    return values.at[index].set(update, mode="drop")


def _gather(values: Array, indices: Array) -> Array:
    return jax.lax.map(lambda index: values[index], indices)


def _contraction(left: Array, coefficients: Array, right: Array, weights: Array) -> Array:
    first = ein.contract("...ui,oij->...uoj", left, coefficients)
    second = ein.contract("...uoj,...vj->...uvo", first, right)
    return ein.contract("...uvo,...uv->...uo", second, weights)


def _cases() -> list[Case]:
    cases = []
    for size in (1024, 65536, 1_000_001):
        values = np.full(size, np.float32(1e-8), dtype=np.float32)
        values[0] = np.float32(1)
        reference = np.asarray(values.sum(dtype=np.float64))
        cases.append(Case("ordinary-sum", size, _sum, (values,), reference, 2e-6, 0.0))
    for size in (32, 128, 512):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(size, 8)).astype(np.float32)
        update = rng.normal(size=8).astype(np.float32)
        index = np.asarray(size // 2, dtype=np.int64)
        expected = values.copy()
        expected[size // 2] = update
        cases.append(
            Case(
                "single-window-scatter",
                size,
                _scatter,
                (values, index, update),
                expected,
                0.0,
                0.0,
            )
        )
        indices = rng.integers(0, size, size=size, dtype=np.int64)
        cases.append(
            Case(
                "ordered-map-gather",
                size,
                _gather,
                (values, indices),
                values[indices],
                0.0,
                0.0,
            )
        )
    for lanes in (8, 64, 256):
        rng = np.random.default_rng(42)
        shapes = ((lanes, 8, 3), (3, 3, 3), (lanes, 1, 3), (lanes, 8, 1))
        inputs = tuple(rng.normal(size=shape).astype(np.float32) for shape in shapes)
        left, coefficients, right, weights = (
            value.astype(np.float64) for value in inputs
        )
        reference = np.empty((lanes, 8, 3), dtype=np.float64)
        # Independent host formula: three fixed output-irrep components.
        for component in range(3):
            first = left @ coefficients[component]
            paired = np.sum(
                first[:, :, None, :] * right[:, None, :, :], axis=-1, dtype=np.float64
            )
            reference[:, :, component] = np.sum(
                paired * weights, axis=-1, dtype=np.float64
            )
        cases.append(
            Case("weighted-uvu", lanes, _contraction, inputs, reference, 2e-6, 2e-5)
        )
    return cases


def _measure(case: Case, repeats: int) -> dict[str, Any]:
    inputs, transfer = measure_synchronized(
        lambda: tuple(jax.device_put(value) for value in case.inputs)
    )
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(case.function).lower(*inputs), lambda lowered: lowered.compile()
    )
    first, first_seconds = measure_synchronized(lambda: compiled(*inputs))
    output, warm = measure_repeated(lambda: compiled(*inputs), warmup=1, repeats=repeats)
    host, readback = measure_host(lambda: np.asarray(output))
    _, ready_wait = measure_synchronized(lambda: output)
    error = float(np.max(np.abs(host.astype(np.float64) - case.reference), initial=0.0))
    bound = case.absolute_tolerance + case.relative_tolerance * float(
        np.max(np.abs(case.reference), initial=0.0)
    )
    devices = output.devices()
    if len(devices) != 1 or next(iter(devices)).platform != jax.default_backend():
        raise RuntimeError("Scientific output escaped the selected backend.")
    if output.dtype != jnp.float32:
        raise RuntimeError("Scientific output changed the declared float32 precision.")
    digest = hashlib.sha256()
    for value in case.inputs:
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes(order="C"))
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="native-compiled-executable",
        unavailable_reason="Backend did not expose compiler cost or memory analysis.",
    )
    return {
        "case": case.name,
        "capacity": case.capacity,
        "input_sha256": digest.hexdigest(),
        "dtype": str(output.dtype),
        "devices": sorted(str(device) for device in devices),
        "passed": bool(np.all(np.isfinite(host))) and error <= bound,
        "accuracy": {
            "maximum_absolute_error": error,
            "absolute_error_bound": bound,
            "reference": "independent NumPy host calculation on identical rounded inputs",
        },
        "timing": {
            "input_transfer_and_sync_seconds": transfer,
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "first_execution_and_sync_seconds": first_seconds,
            "warm_synchronized": warm.to_seconds_dict(),
            "output_readback_seconds": readback,
            "ready_result_wait_seconds": ready_wait,
        },
        "compiler": asdict(evidence),
        "logical_bytes": {
            "inputs": logical_array_bytes(inputs),
            "outputs": logical_array_bytes(output),
            "retained": logical_array_bytes(
                (inputs, first, output, case.inputs, case.reference, host)
            ),
        },
        "values": host.tolist(),
    }


def run(repeats: int) -> dict[str, Any]:
    if repeats < 1 or repeats > 100:
        raise ValueError("repeats must be between 1 and 100.")
    if not config.x64_enabled():
        raise ValueError("Exact integer indexing requires x64 enabled.")
    cases, preparation = measure_host(_cases)
    records = [_measure(case, repeats) for case in cases]
    return {
        "identity": capture_benchmark_identity(
            _ROOT, Path(__file__), records[0].keys()
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "host_input_and_reference_preparation_seconds": preparation,
        "scope": "Fixed bounded capacities; no CPU numerical fallback. Unknown compiler memory remains unknown.",
        "records": records,
        "passed": all(record["passed"] for record in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.repeats)
    write_json_atomic(arguments.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
