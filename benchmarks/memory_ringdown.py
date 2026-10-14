"""Phase-separated ringdown time-axis workspace and compiler adoption evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from phydrax import config as pxconfig
from phydrax.applications.compact_objects import QuadraticRingdownPlan

from ._io import write_json_atomic
from ._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
    process_peak_rss_bytes,
)
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def _forward(
    plan: QuadraticRingdownPlan, times: Array
) -> tuple[Array, Array, Array, Array]:
    result = plan.evaluate(times)
    return (
        result.quadratic_modes,
        result.total_signal,
        result.pair_detuning,
        result.status,
    )


class _Compiled(Protocol):
    compiled: jax.stages.Compiled

    def __call__(
        self, plan: QuadraticRingdownPlan, times: Array
    ) -> tuple[Array, Array, Array, Array]: ...


class _Lowered(Protocol):
    def compile(self) -> _Compiled: ...


@runtime_checkable
class _Entry(Protocol):
    def lower(self, plan: QuadraticRingdownPlan, times: Array) -> _Lowered: ...


def _require_entry(entry: object) -> _Entry:
    if not isinstance(entry, _Entry):
        raise TypeError("Ringdown compilation requires a lowering entry.")
    return entry


_FORWARD = _require_entry(eqx.filter_jit(_forward))


def _reference(
    poles: NDArray[np.complex128],
    amplitudes: NDArray[np.complex128],
    coupling: NDArray[np.complex128],
    times: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Direct integrated exponential differences, independent of sinc evaluation."""
    result = np.zeros((times.size, poles.size), dtype=np.complex128)
    for k in range(poles.size):
        delta = poles[:, None] + poles[None, :] - poles[k]
        nonzero = delta != 0.0
        for index, sample in enumerate(times):
            integral = np.full(delta.shape, sample, dtype=np.complex128)
            integral[nonzero] = np.expm1(-1.0j * sample * delta[nonzero]) / (
                -1.0j * delta[nonzero]
            )
            result[index, k] = np.exp(-1.0j * sample * poles[k]) * np.sum(
                coupling[k] * amplitudes[:, None] * amplitudes[None, :] * integral
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--compile-only-times", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--response-workspace-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
    runtime_sizes = [8, 16] if args.smoke else args.times
    compile_only_sizes = [] if args.smoke else args.compile_only_times
    sizes = runtime_sizes + compile_only_sizes
    modes = 4 if args.smoke else args.modes
    if len(sizes) < 2 or any(t <= 0 for t in sizes) or sizes != sorted(set(sizes)):
        raise ValueError("Time capacities must be positive, increasing, and distinct.")
    poles = np.linspace(0.4, 2.0, modes) - 1.0j * np.linspace(0.03, 0.1, modes)
    amplitudes = np.linspace(0.1, 0.5, modes).astype(np.complex128)
    coupling = np.full((modes, modes, modes), 0.001 + 0.002j, dtype=np.complex128)

    def prepare() -> QuadraticRingdownPlan:
        return QuadraticRingdownPlan(
            poles,
            amplitudes,
            coupling,
            response_workspace_bytes=args.response_workspace_bytes,
            time_chunk_size=args.chunk,
        )

    preparation_start = time.perf_counter()
    plan, preparation_peak = host_allocation_peak(prepare)
    preparation_seconds = time.perf_counter() - preparation_start
    full_plan = QuadraticRingdownPlan(
        poles,
        amplitudes,
        coupling,
        response_workspace_bytes=args.response_workspace_bytes,
        time_chunk_size=max(sizes),
    )
    bytes_per_time = plan.pair_detuning.dtype.itemsize * modes**3
    cases: list[dict[str, Any]] = []
    for size in sizes:
        times = jnp.linspace(0.0, 5.0, size)
        compile_only = size in compile_only_sizes
        chunk = min(
            size,
            args.chunk
            if args.chunk is not None
            else max(1, plan.response_workspace_bytes // bytes_per_time),
        )
        route_records: dict[str, Any] = {}
        for route, route_plan in (("budget", plan), ("full", full_plan)):
            executable, timing = measure_lower_and_compile(
                lambda: _FORWARD.lower(route_plan, times),
                lambda lowered: lowered.compile(),
            )
            compiler = compiler_evidence(
                executable.compiled.cost_analysis(),
                executable.compiled.memory_analysis(),
                source="jax-compiled-quadratic-ringdown",
                unavailable_reason="Compiler memory analysis unavailable.",
            )
            route_record: dict[str, Any] = {
                "plan_id": route_plan.plan_id,
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "compiler": asdict(compiler),
                "retained_plan_bytes": logical_array_bytes(route_plan),
                "logical_response_bytes": bytes_per_time
                * (chunk if route == "budget" else size),
                "time_chunk_size": chunk if route == "budget" else size,
                "response_workspace_bytes": route_plan.response_workspace_bytes,
                "execution": "vmap" if route == "full" or chunk == size else "lax.map",
                "warm": None,
                "retained_output_bytes": None,
                "status": None,
                "reference_max_absolute_defect": None,
            }
            if not compile_only:
                result, warm = measure_repeated(
                    lambda: executable(route_plan, times),
                    warmup=1,
                    repeats=args.repeats,
                )
                reference = _reference(
                    poles, amplitudes, coupling, np.asarray(times, dtype=np.float64)
                )
                route_record.update(
                    {
                        "warm": warm.to_seconds_dict(),
                        "retained_output_bytes": logical_array_bytes(result),
                        "status": int(result[3]),
                        "reference_max_absolute_defect": float(
                            np.max(np.abs(np.asarray(result[0]) - reference))
                        ),
                    }
                )
            route_records[route] = route_record
        cases.append(
            {
                "time_count": size,
                "mode_count": modes,
                "compile_only": compile_only,
                "memory_comparison": "tiled" if chunk < size else "single-tile",
                **route_records["budget"],
                "full_response": route_records["full"],
            }
        )
    temporary = [case["compiler"]["temporary_bytes"] for case in cases]
    workspace = [case["logical_response_bytes"] for case in cases]
    tiled = [case for case in cases if case["memory_comparison"] == "tiled"]
    record: dict[str, Any] = {
        "identity": asdict(
            capture_benchmark_identity(
                Path.cwd(),
                Path(__file__),
                ("environment", "configuration", "cases", "gate"),
            )
        ),
        "environment": asdict(capture_environment()),
        "command": sys.argv,
        "configuration": {
            "response_workspace_bytes": plan.response_workspace_bytes,
            "time_chunk_size": plan.time_chunk_size,
            "mode_count": modes,
            "runtime_times": runtime_sizes,
            "compile_only_times": compile_only_sizes,
        },
        "gate_basis": "same-run current-code budget route versus explicit full-tile vmap",
        "single_tile_comparison": "Equal memory below the budget is expected, not a gate failure.",
        "preparation_host_peak_bytes": preparation_peak,
        "preparation_seconds": preparation_seconds,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "cases": cases,
        "logical_workspace_exponent": fitted_scaling_exponent(sizes, workspace),
        "compiler_temporary_exponent": fitted_scaling_exponent(sizes, temporary)
        if all(temporary)
        else None,
        "doubled_time_workspace_ratios": [
            b / a for a, b in zip(workspace, workspace[1:])
        ],
        "gate": None,
    }
    if tiled:
        record["gate"] = evaluate_memory_gate(
            [case["time_count"] for case in tiled],
            [case["full_response"]["compiler"]["temporary_bytes"] for case in tiled],
            [case["compiler"]["temporary_bytes"] for case in tiled],
            MemoryGatePolicy(),
        ).to_dict()
    else:
        record["gate_unavailable_reason"] = "No capacity exceeds the selected time tile."
    record["scaling_acceptance"] = all(
        ratio <= 2.25 for ratio in record["doubled_time_workspace_ratios"]
    )
    record["doubled_time_compiler_temporary_ratios"] = (
        [b / a for a, b in zip(temporary, temporary[1:])] if all(temporary) else []
    )
    record["compiler_scaling_acceptance"] = (
        all(ratio <= 2.25 for ratio in record["doubled_time_compiler_temporary_ratios"])
        if all(temporary)
        else None
    )
    output = args.output or Path(__file__).with_name("memory_ringdown.json")
    write_json_atomic(output, record)
    print(
        json.dumps(
            {
                "output": str(output),
                "temporary_bytes": temporary,
                "gate": record["gate"],
                "rss": record["process_peak_rss_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
