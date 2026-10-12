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
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).with_name("memory_ringdown_baseline.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    sizes = [8, 16] if args.smoke else args.times
    modes = 4 if args.smoke else args.modes
    if len(sizes) < 2 or any(t <= 0 for t in sizes) or sizes != sorted(set(sizes)):
        raise ValueError("Time capacities must be positive, increasing, and distinct.")
    poles = np.linspace(0.4, 2.0, modes) - 1.0j * np.linspace(0.03, 0.1, modes)
    amplitudes = np.linspace(0.1, 0.5, modes).astype(np.complex128)
    coupling = np.full((modes, modes, modes), 0.001 + 0.002j, dtype=np.complex128)

    def prepare() -> QuadraticRingdownPlan:
        if args.candidate:
            return QuadraticRingdownPlan(
                poles, amplitudes, coupling, time_chunk_size=args.chunk
            )
        return QuadraticRingdownPlan(poles, amplitudes, coupling)

    preparation_start = time.perf_counter()
    plan, preparation_peak = host_allocation_peak(prepare)
    preparation_seconds = time.perf_counter() - preparation_start
    cases: list[dict[str, Any]] = []
    for size in sizes:
        times = jnp.linspace(0.0, 5.0, size)
        executable, timing = measure_lower_and_compile(
            lambda: _FORWARD.lower(plan, times), lambda lowered: lowered.compile()
        )
        result, warm = measure_repeated(
            lambda: executable(plan, times), warmup=1, repeats=args.repeats
        )
        compiler = compiler_evidence(
            executable.compiled.cost_analysis(),
            executable.compiled.memory_analysis(),
            source="jax-compiled-quadratic-ringdown",
            unavailable_reason="Compiler memory analysis unavailable.",
        )
        reference = _reference(
            poles, amplitudes, coupling, np.asarray(times, dtype=np.float64)
        )
        cases.append(
            {
                "time_count": size,
                "mode_count": modes,
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "warm": warm.to_seconds_dict(),
                "compiler": asdict(compiler),
                "retained_plan_bytes": logical_array_bytes(plan),
                "retained_output_bytes": logical_array_bytes(result),
                "logical_response_bytes": 16
                * modes**3
                * (min(size, args.chunk) if args.candidate else size),
                "status": int(result[3]),
                "reference_max_absolute_defect": float(
                    np.max(np.abs(np.asarray(result[0]) - reference))
                ),
            }
        )
    temporary = [case["compiler"]["temporary_bytes"] for case in cases]
    workspace = [case["logical_response_bytes"] for case in cases]
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
            "candidate": args.candidate,
            "time_chunk_size": args.chunk,
            "mode_count": modes,
        },
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
    }
    if args.candidate and not args.smoke:
        baseline = json.loads(args.baseline.read_text())
        if [case["time_count"] for case in baseline["cases"]] != sizes or baseline[
            "configuration"
        ]["mode_count"] != modes:
            raise ValueError("Baseline capacities must match candidate capacities.")
        record["gate"] = evaluate_memory_gate(
            sizes,
            [case["compiler"]["temporary_bytes"] for case in baseline["cases"]],
            temporary,
            MemoryGatePolicy(),
        ).to_dict()
        record["scaling_acceptance"] = all(
            ratio <= 2.25 for ratio in record["doubled_time_workspace_ratios"]
        )
        record["doubled_time_compiler_temporary_ratios"] = [
            b / a for a, b in zip(temporary, temporary[1:])
        ]
        record["compiler_scaling_acceptance"] = all(
            ratio <= 2.25 for ratio in record["doubled_time_compiler_temporary_ratios"]
        )
    output = args.output or Path(__file__).with_name(
        "memory_ringdown.json" if args.candidate else "memory_ringdown_baseline.json"
    )
    write_json_atomic(output, record)
    print(
        json.dumps(
            {
                "output": str(output),
                "temporary_bytes": temporary,
                "gate": record.get("gate"),
                "rss": record["process_peak_rss_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
