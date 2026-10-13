"""Phase-separated graph curvature and KKT sensitivity memory scaling."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from phydrax import optim
from phydrax.linalg import (
    AbstractLinearOperator,
    GeneralizedLSMR,
    inverse_residual_certificate,
    InverseResidualCertificate,
    JacobianLinearOperator,
    LeastSquaresProblem,
    LinearSolvePolicy,
    MaterializationPolicy,
    materialize,
    prepare,
    prepare_linearization,
    solve_many,
    TolerancePolicy,
)

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


def graph_problem(size: int) -> optim.ResidualGraphProblem:
    def parameter(index: int) -> optim.ParameterBlock:
        def extract(value: Array) -> Array:
            return value[index : index + 1]

        def replace(value: Array, replacement: Array) -> Array:
            return value.at[index : index + 1].set(replacement)

        return optim.ParameterBlock(extract, replace, block_id=f"x-{index}")

    def residual(values: tuple[Array, ...], _: Any) -> Array:
        return values[0] + 2.0 * values[1] - 1.0

    return optim.ResidualGraphProblem(
        tuple(parameter(i) for i in range(size)),
        tuple(
            optim.ResidualBlock(residual, (f"x-{i}", f"x-{i + 1}"), block_id=f"r-{i}")
            for i in range(size - 1)
        ),
        problem_id=f"memory-chain-{size}",
    )


def _graph(graph: optim.ResidualGraphProblem, point: Array) -> AbstractLinearOperator:
    return optim.linearize_residual_graph(graph, point).curvature


def _kkt(point: Array) -> Array:
    n = point.size // 2
    return jnp.concatenate((2.0 * point[:n] + point[n:], point[:n]))


def _sensitivity(point: Array, baseline: bool) -> Any:
    if baseline:
        return jax.jacfwd(_kkt)(point)
    return JacobianLinearOperator(prepare_linearization(_kkt, point))


def _certification(point: Array) -> InverseResidualCertificate:
    operator = JacobianLinearOperator(prepare_linearization(_kkt, point))
    prepared = prepare(
        LeastSquaresProblem(operator),
        LinearSolvePolicy(
            GeneralizedLSMR(),
            tolerance=TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=100),
        ),
    )

    def inverse_action(columns: Array) -> tuple[Array, Array]:
        result = solve_many(prepared, columns)
        return result.value, (
            result.successful & result.diagnostics.converged & result.diagnostics.finite
        )

    return inverse_residual_certificate(operator, inverse_action)


class _Compiled(Protocol):
    compiled: jax.stages.Compiled

    def __call__(
        self, graph: optim.ResidualGraphProblem, point: Array, baseline: bool, name: str
    ) -> Any: ...


class _Lowered(Protocol):
    def compile(self) -> _Compiled: ...


@runtime_checkable
class _Entry(Protocol):
    def __call__(
        self, graph: optim.ResidualGraphProblem, point: Array, baseline: bool, name: str
    ) -> Any: ...

    def lower(
        self, graph: optim.ResidualGraphProblem, point: Array, baseline: bool, name: str
    ) -> _Lowered: ...


def _entry(
    graph: optim.ResidualGraphProblem, point: Array, baseline: bool, name: str
) -> Any:
    if name == "curvature":
        curvature = _graph(graph, point)
        return materialize(curvature, MaterializationPolicy()) if baseline else curvature
    if name == "sensitivity-certification":
        return _certification(point)
    return _sensitivity(point, baseline)


def _require_entry(
    entry: Callable[[optim.ResidualGraphProblem, Array, bool, str], Any],
) -> _Entry:
    if not isinstance(entry, _Entry):
        raise TypeError("Benchmark requires a lowerable compiled entry.")
    return entry


_ENTRY = _require_entry(eqx.filter_jit(_entry))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    cases: list[dict[str, Any]] = []
    names = (
        ("curvature", "sensitivity")
        if options.baseline
        else ("curvature", "sensitivity", "sensitivity-certification")
    )
    for size in options.sizes:
        graph = graph_problem(size)
        point = jnp.zeros((size,), dtype=jnp.float64)
        for name in names:
            entry = _ENTRY
            args = (
                graph,
                point
                if name == "curvature"
                else jnp.zeros((2 * size,), dtype=jnp.float64),
                options.baseline,
                name,
            )
            executable, timing = measure_lower_and_compile(
                lambda: entry.lower(*args), lambda lowered: lowered.compile()
            )
            result, warm = measure_repeated(
                lambda: executable(*args), warmup=1, repeats=1
            )
            _, host_peak = host_allocation_peak(lambda: entry(*args))
            evidence = compiler_evidence(
                executable.compiled.cost_analysis(),
                executable.compiled.memory_analysis(),
                source=name,
                unavailable_reason="Compiler memory unavailable",
            )
            cases.append(
                {
                    "name": name,
                    "capacity": size,
                    "logical_bytes": logical_array_bytes(result),
                    "host_peak_bytes": host_peak,
                    "lowering_seconds": timing.lowering_seconds,
                    "compilation_seconds": timing.compilation_seconds,
                    "warm": warm.to_seconds_dict(),
                    "compiler": asdict(evidence),
                    "status": int(jax.device_get(result.status))
                    if name == "sensitivity-certification"
                    else None,
                    "rank": int(jax.device_get(result.rank))
                    if name == "sensitivity-certification"
                    else None,
                    "condition_upper_bound": float(
                        jax.device_get(result.condition_upper_bound)
                    )
                    if name == "sensitivity-certification"
                    else None,
                }
            )
    exponents = {
        name: fitted_scaling_exponent(
            options.sizes,
            [case["logical_bytes"] for case in cases if case["name"] == name],
        )
        for name in names
    }
    temporary_exponents: dict[str, float | None] = {}
    for name in names:
        values = [
            case["compiler"]["temporary_bytes"] for case in cases if case["name"] == name
        ]
        temporary_exponents[name] = (
            fitted_scaling_exponent(options.sizes, values)
            if all(isinstance(value, int) and value > 0 for value in values)
            else None
        )
    gates = {}
    if not options.baseline:
        baseline_record = json.loads(
            Path(__file__).with_name("memory_optim_curvature_baseline.json").read_text()
        )
        for name in ("curvature", "sensitivity"):
            previous = [
                case
                for case in baseline_record["cases"]
                if case["name"] == name and case["capacity"] in options.sizes
            ]
            candidate = [case for case in cases if case["name"] == name]
            gates[name] = evaluate_memory_gate(
                options.sizes,
                [case["compiler"]["temporary_bytes"] for case in previous],
                [case["compiler"]["temporary_bytes"] for case in candidate],
                MemoryGatePolicy(),
            ).to_dict()
    output = options.output or Path(__file__).with_name(
        "memory_optim_curvature_baseline.json"
        if options.baseline
        else "memory_optim_curvature.json"
    )
    write_json_atomic(
        output,
        {
            "identity": asdict(
                capture_benchmark_identity(
                    Path(__file__).resolve().parents[1],
                    Path(__file__),
                    ("environment", "cases"),
                )
            ),
            "environment": asdict(capture_environment()),
            "baseline": options.baseline,
            "cases": cases,
            "logical_scaling_exponents": exponents,
            "temporary_scaling_exponents": temporary_exponents,
            "compiler_gates": gates,
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        },
    )


if __name__ == "__main__":
    main()
