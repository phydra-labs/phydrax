"""Phase-separated exact FEL wake memory scaling; run with --smoke."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax import Array
from scipy import constants

from phydrax import ElectromagneticScaleContract
from phydrax.applications.accelerator import InsertionDeviceField, WakeFunctionPlan
from phydrax.applications.accelerator.fel import (
    FELBeamSlices,
    FELLoading,
    FELPlan,
    FELUndulatorLattice,
    FELUndulatorSegment,
    FELWakeLoss,
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


def prepare(count: int) -> tuple[FELPlan, FELBeamSlices]:
    lattice = FELUndulatorLattice(
        ElectromagneticScaleContract.si(),
        (
            FELUndulatorSegment(
                InsertionDeviceField(
                    1.0, 0.03, 1, polarization="planar", center=0.0, aperture=(0.01, 0.01)
                )
            ),
        ),
        step_length=0.01,
    )
    spacing = 1.0e-6
    slices = FELBeamSlices(
        np.arange(count) * spacing,
        np.linspace(1000.0, 3000.0, count),
        np.full(count, 1.0e4),
        np.zeros(count),
        np.full((count, 2), 1.0e-6),
        np.full((count, 2), 10.0),
        np.zeros((count, 2)),
    )
    wake = WakeFunctionPlan(
        "longitudinal",
        np.array([0.0, count * spacing]),
        np.array([1.0e13, 2.0e12]),
        units="V/C",
        causality_convention="behind-positive",
    )
    return FELPlan(
        lattice,
        1.0e-9,
        loading=FELLoading(8, 2, shot_noise="quiet"),
        wake=FELWakeLoss(wake, structure_length=2.0),
    ), slices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/memory_fel_wake.json")
    )
    args = parser.parse_args()
    sizes = [32, 64] if args.smoke else args.sizes
    cases: list[dict[str, Any]] = []
    for count in sizes:
        (plan, slices), peak = host_allocation_peak(lambda: prepare(count))

        def rates(currents: Array) -> Array:
            import equinox as eqx

            dynamic = eqx.tree_at(lambda beam: beam.currents, slices, currents)
            return plan._wake_rates(dynamic)

        entry = jax.jit(rates)
        executable, timing = measure_lower_and_compile(
            lambda: entry.lower(slices.currents), lambda lowered: lowered.compile()
        )
        result, warm = measure_repeated(
            lambda: executable(slices.currents), warmup=1, repeats=3
        )
        memory = executable.memory_analysis()
        if memory is None:
            raise RuntimeError("The FEL memory gate requires compiler memory analysis.")
        compiler = compiler_evidence(
            executable.cost_analysis(),
            memory,
            source="jax-fel-wake",
            unavailable_reason="Compiler estimates unavailable",
        )
        positions = np.asarray(slices.positions)
        lag = positions[:, None] - positions[None, :]
        values = np.where(
            lag >= 0.0, np.interp(lag, [0.0, count * 1.0e-6], [1.0e13, 2.0e12]), 0.0
        )
        values[np.diag_indices(count)] *= 0.5
        expected = (
            constants.e
            * (values @ (np.asarray(slices.currents) * 1.0e-6 / constants.c))
            / (constants.m_e * constants.c**2 * 2.0)
        )
        np.testing.assert_allclose(result, expected, rtol=2.0e-11)
        cases.append(
            {
                "capacity": count,
                "host_peak_bytes": peak,
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "warm": warm.to_seconds_dict(),
                "compiler": asdict(compiler),
                "temporary_bytes": memory.temp_size_in_bytes,
                "retained_bytes": logical_array_bytes((plan, slices)),
                "reference_relative_defect": float(
                    np.max(np.abs(np.asarray(result) / expected - 1.0))
                ),
            }
        )
    record: dict[str, Any] = {
        "identity": asdict(
            capture_benchmark_identity(
                Path.cwd(), Path(__file__), ("cases", "environment")
            )
        ),
        "environment": asdict(capture_environment()),
        "cases": cases,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        record["gate"] = evaluate_memory_gate(
            sizes,
            [case["temporary_bytes"] for case in baseline["cases"]],
            [case["temporary_bytes"] for case in cases],
            MemoryGatePolicy(),
        ).to_dict()
        record["temporary_scaling_exponent"] = fitted_scaling_exponent(
            sizes, [case["temporary_bytes"] for case in cases]
        )
        record["doubling_ratios"] = [
            later["temporary_bytes"] / earlier["temporary_bytes"]
            for earlier, later in zip(cases, cases[1:])
        ]
    write_json_atomic(args.output, record)


if __name__ == "__main__":
    main()
