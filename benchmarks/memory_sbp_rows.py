"""Host preparation and compiled action evidence for fixed-width SBP rows."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from phydrax.discretization import SBPDerivativePlan, TensorGridPlan, UniformAxisSpec

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    args = parser.parse_args()
    sizes = [32, 64] if args.smoke else args.sizes
    cases: list[dict[str, Any]] = []
    for periodic in (False, True):
        for order in (2, 4, 6, 8):
            for count in sizes:
                grid = TensorGridPlan(
                    (UniformAxisSpec(count, periodic=periodic, endpoint=not periodic),),
                    axis_names=("x",),
                ).prepare(jnp.asarray([[0.0], [1.0]], dtype=jnp.float64))
                plan = SBPDerivativePlan(grid, "x", interior_order=order)
                started = time.perf_counter()
                prepared, peak = host_allocation_peak(plan.prepare)
                seconds = time.perf_counter() - started
                state = jnp.sin(jnp.arange(count, dtype=jnp.float64))
                entry = jax.jit(lambda value: prepared.operator.mv(value))
                executable, timing = measure_lower_and_compile(
                    lambda: entry.lower(state), lambda lowered: lowered.compile()
                )
                _, warm = measure_repeated(lambda: executable(state), warmup=1, repeats=2)
                evidence = compiler_evidence(
                    executable.cost_analysis(),
                    executable.memory_analysis(),
                    source="jax.stages.Compiled",
                )
                cases.append(
                    {
                        "count": count,
                        "order": order,
                        "periodic": periodic,
                        "host_peak_bytes": peak,
                        "preparation_seconds": seconds,
                        "retained_bytes": logical_array_bytes(prepared),
                        "stencil_bytes": logical_array_bytes(
                            prepared.operator.stencil_set.stencil
                        ),
                        "residual": prepared.stability_report.residual,
                        "compilation": asdict(timing),
                        "warm": asdict(warm),
                        "compiler": asdict(evidence),
                    }
                )
    scaling = [
        {
            "order": order,
            "periodic": periodic,
            "host_exponent": fitted_scaling_exponent(
                sizes,
                [
                    case["host_peak_bytes"]
                    for case in cases
                    if case["order"] == order and case["periodic"] == periodic
                ],
            ),
        }
        for periodic in (False, True)
        for order in (2, 4, 6, 8)
    ]
    write_json_atomic(
        args.output,
        {
            "identity": asdict(
                capture_benchmark_identity(
                    Path(__file__).resolve().parents[1],
                    Path(__file__).resolve(),
                    ("cases", "scaling", "environment"),
                )
            ),
            "environment": asdict(capture_environment()),
            "cases": cases,
            "scaling": scaling,
            "rss_bytes": process_peak_rss_bytes(),
            "adoption": "unconditional exact host allocation remediation",
        },
    )


if __name__ == "__main__":
    main()
