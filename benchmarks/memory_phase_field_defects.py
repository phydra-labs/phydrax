"""Phase-separated Chebyshev action memory scaling (default smoke sizes)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ._io import write_json_atomic
from ._memory import fitted_scaling_exponent, host_allocation_peak, process_peak_rss_bytes
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def _dense_reference(count: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.cos(np.pi * np.arange(count) / (count - 1))[::-1]
    coefficients = np.asarray((-1.0) ** np.arange(count), dtype=np.float64)
    coefficients[[0, -1]] *= 2.0
    difference = nodes[:, None] - nodes[None, :]
    matrix = (coefficients[:, None] / coefficients[None, :]) / (
        difference + np.eye(count)
    )
    matrix -= np.diag(np.sum(matrix, axis=1))
    factor = (1.0 - nodes**2) ** 2 / (1.0 + nodes**2)
    first = -factor[:, None] * matrix
    return first, first @ first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[33, 65, 129, 257])
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    options = parser.parse_args()
    sizes = [17, 33] if options.smoke else options.sizes
    cases: list[dict[str, Any]] = []
    for count in sizes:
        values = jnp.asarray(np.sin(np.linspace(-1.0, 1.0, count)), dtype=jnp.float64)
        if options.baseline:
            operators, peak = host_allocation_peak(lambda: _dense_reference(count))
            first, second = jnp.asarray(operators[0]), jnp.asarray(operators[1])

            def action(value: Array) -> tuple[Array, Array]:
                return first @ value, second @ value

            retained = logical_array_bytes((first, second))
            workspace = 16 * count * count
        else:
            from phydrax.discretization.collocation import ChebyshevDifferentiationAction

            plan, peak = host_allocation_peak(
                lambda: ChebyshevDifferentiationAction(count)
            )
            factor = (1.0 - plan.nodes**2) ** 2 / (1.0 + plan.nodes**2)

            def action(value: Array) -> tuple[Array, Array]:
                first_value = -factor * plan.derivative(value)
                return first_value, -factor * plan.derivative(first_value)

            retained = logical_array_bytes(plan)
            workspace = 8 * 8 * count
        executable, timing = measure_lower_and_compile(
            lambda: jax.jit(action).lower(values), lambda lowered: lowered.compile()
        )
        result, warm = measure_repeated(lambda: executable(values), warmup=1, repeats=3)
        compiler = compiler_evidence(
            executable.cost_analysis(),
            executable.memory_analysis(),
            source="phase-field-chebyshev-action",
            unavailable_reason="Compiler memory evidence unavailable.",
        )
        dense_first, dense_second = _dense_reference(count)
        first_defect = float(
            np.max(np.abs(np.asarray(result[0]) - dense_first @ np.asarray(values)))
        )
        second_defect = float(
            np.max(np.abs(np.asarray(result[1]) - dense_second @ np.asarray(values)))
        )
        if first_defect > 1e-10 or second_defect > 1e-8:
            raise ValueError(
                "Chebyshev action does not match the independent dense reference."
            )
        cases.append(
            {
                "count": count,
                "preparation_host_peak_bytes": peak,
                "retained_bytes": retained,
                "logical_workspace_bytes": workspace,
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "warm": warm.to_seconds_dict(),
                "compiler": asdict(compiler),
                "output_bytes": logical_array_bytes(result),
                "finite": bool(jnp.all(jnp.isfinite(result[1]))),
                "first_derivative_reference_defect": first_defect,
                "second_derivative_reference_defect": second_defect,
            }
        )
    output = options.output or Path(__file__).with_name(
        "memory_phase_field_defects_baseline.json"
        if options.baseline
        else "memory_phase_field_defects.json"
    )
    write_json_atomic(
        output,
        {
            "identity": capture_benchmark_identity(
                Path(__file__).resolve().parents[1],
                Path(__file__).resolve(),
                ("environment", "cases", "workspace_exponent"),
            ).to_dict(),
            "environment": asdict(capture_environment()),
            "cases": cases,
            "workspace_exponent": fitted_scaling_exponent(
                sizes, [case["logical_workspace_bytes"] for case in cases]
            ),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        },
    )


if __name__ == "__main__":
    main()
