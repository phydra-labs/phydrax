#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""B-spline field-view preparation and multi-RHS solve memory scaling."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.interpolate import BSpline

from phydrax.discretization.finite_difference._field_view import (
    BSplineGridInterpolation,
    FiniteDifferenceFieldReconstructionKernel,
)

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


def _solve(kernel: FiniteDifferenceFieldReconstructionKernel, values: Array) -> Array:
    return kernel._solve_axes(values, transpose=False)


class _Compiled(Protocol):
    compiled: jax.stages.Compiled

    def __call__(
        self, kernel: FiniteDifferenceFieldReconstructionKernel, values: Array, /
    ) -> Array: ...


class _Lowered(Protocol):
    def compile(self) -> _Compiled: ...


@runtime_checkable
class _Entry(Protocol):
    def lower(
        self, kernel: FiniteDifferenceFieldReconstructionKernel, values: Array, /
    ) -> _Lowered: ...


def _require_entry(
    entry: Callable[[FiniteDifferenceFieldReconstructionKernel, Array], Array], /
) -> _Entry:
    if not isinstance(entry, _Entry):
        raise TypeError("Field-view benchmark requires a callable lowering entry.")
    return entry


_SOLVE = _require_entry(eqx.filter_jit(_solve))


def _case(size: int, degree: int, repeats: int) -> dict[str, Any]:
    nodes = np.linspace(0.0, 1.0, size, dtype=np.float64)

    # One untimed preparation separates first-use imports/compilation from traced peaks.
    def prepare() -> FiniteDifferenceFieldReconstructionKernel:
        return FiniteDifferenceFieldReconstructionKernel(
            BSplineGridInterpolation(degree),
            (nodes,),
            nodes[:1],
            nodes[-1:],
            (False,),
            field_space_id="memory-field-view",
        )

    prepare()
    start = time.perf_counter()
    kernel, peak = host_allocation_peak(prepare)
    preparation_seconds = time.perf_counter() - start
    values = jnp.asarray(np.stack((np.sin(nodes), nodes**degree, np.cos(nodes)), axis=1))
    compiled, timing = measure_lower_and_compile(
        lambda: _SOLVE.lower(kernel, values), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(
        lambda: compiled(kernel, values), warmup=1, repeats=repeats
    )
    compiler = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-field-view",
        unavailable_reason="Compiler evidence unavailable.",
    )
    reference = BSpline.design_matrix(nodes, np.asarray(kernel.knots[0]), degree)
    residual = reference @ np.asarray(result) - np.asarray(values)
    return {
        "size": size,
        "degree": degree,
        "host_peak_bytes": peak,
        "preparation_seconds": preparation_seconds,
        "retained_bytes": logical_array_bytes(kernel),
        "condition_evidence": kernel.condition,
        "native_condition_brackets": [
            {
                "status": int(bracket.status),
                "steps": int(bracket.steps),
                "minimum_lower": float(bracket.minimum_lower),
                "minimum_upper": float(bracket.minimum_upper),
                "maximum_lower": float(bracket.maximum_lower),
                "maximum_upper": float(bracket.maximum_upper),
                "minimum_width": float(bracket.minimum_width),
                "maximum_width": float(bracket.maximum_width),
                "minimum_rounding_floor": float(bracket.minimum_rounding_floor),
                "maximum_rounding_floor": float(bracket.maximum_rounding_floor),
            }
            for bracket in kernel.condition_evidence
        ],
        "solve_residual_max": float(np.max(np.abs(residual))),
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(compiler),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    cases = [_case(size, args.degree, args.repeats) for size in args.sizes]
    record = {
        "environment": asdict(capture_environment()),
        "command": sys.argv,
        "identity": capture_benchmark_identity(
            Path(__file__).resolve().parents[1], Path(__file__), ("cases", "scaling")
        ).to_dict(),
        "cases": cases,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "scaling": {
            "retained_exponent": fitted_scaling_exponent(
                args.sizes, [case["retained_bytes"] for case in cases]
            ),
            "host_peak_exponent": fitted_scaling_exponent(
                args.sizes, [case["host_peak_bytes"] for case in cases]
            ),
        },
        "gate": "unconditional-host-dense-remediation",
    }
    write_json_atomic(args.output, record)
    print(record["scaling"])


if __name__ == "__main__":
    main()
