"""Host allocation scaling for subband form factors and scenario reduction.

Run with --smoke for quick evidence. Capture --baseline before editing the owners;
then rerun the same sizes without it. Compiler phases are inapplicable: both
public operations perform immutable NumPy host work, not a compiled JAX kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from phydrax import config as pxconfig
from phydrax.applications.quantum_hall import (
    evaluate_subband_coulomb_form_factor,
    SubbandCoulombFormFactorPlan,
)
from phydrax.finance.core import FinancialScenarioSet, PhysicalLaw
from phydrax.finance.risk import reduce_scenarios

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
    logical_array_bytes,
    measure_host,
    measure_repeated,
    synchronize,
)


def _measure[T](operation: Callable[[], T], repeats: int) -> tuple[T, dict[str, Any]]:
    # Keep one-time JAX result-construction compilation out of host-kernel workspace.
    synchronize(operation())
    result, peak = host_allocation_peak(operation)
    synchronize(result)
    result, warm = measure_repeated(operation, warmup=0, repeats=repeats)
    return result, {
        "host_allocation_peak_bytes": peak,
        "peak_preceded_by_warmup": True,
        "warm": warm.to_seconds_dict(),
        "retained_output_bytes": logical_array_bytes(result),
        "lowering_seconds": None,
        "compilation_seconds": None,
        "compiler": None,
        "compiler_unavailable_reason": "Immutable NumPy host operation; no compiled entry point.",
    }


def _quantum_hall(size: int, repeats: int) -> dict[str, Any]:
    positions = 1.0e-8 * np.linspace(0.0, 1.0, size, dtype=np.float64) ** 1.2
    density = 1.0 + 0.3 * np.cos(positions / 1.0e-8 * np.pi)
    density /= np.trapezoid(density, positions)
    plan, preparation = measure_host(
        lambda: SubbandCoulombFormFactorPlan(positions, density)
    )
    wave_numbers = np.asarray((0.0, 1.0e8, 3.0e8), dtype=np.float64)
    result, evidence = _measure(
        lambda: evaluate_subband_coulomb_form_factor(plan, wave_numbers), repeats
    )
    defect = None
    if size <= 101:
        weights = np.empty_like(positions)
        weights[0] = (positions[1] - positions[0]) / 2.0
        weights[-1] = (positions[-1] - positions[-2]) / 2.0
        weights[1:-1] = (positions[2:] - positions[:-2]) / 2.0
        p = weights * density
        reference = np.asarray(
            [
                p @ np.exp(-q * np.abs(positions[:, None] - positions[None, :])) @ p
                for q in wave_numbers
            ],
            dtype=np.float64,
        )
        defect = float(np.max(np.abs(np.asarray(result.form_factors) - reference)))
        if defect > 1.0e-12:
            raise RuntimeError(f"Subband dense quadrature mismatch: {defect}.")
    return {
        "capacity": size,
        "wave_number_count": wave_numbers.size,
        "preparation_seconds": preparation,
        "retained_input_bytes": logical_array_bytes(plan),
        "form_factors": np.asarray(result.form_factors).tolist(),
        "plan_id": plan.plan_id,
        "result_id": result.result_id,
        "successful": bool(result.successful),
        "zero_momentum_residual": float(result.zero_momentum_residual),
        "dense_reference_maximum_defect": defect,
        **evidence,
    }


def _scenarios(size: int, representatives: int, repeats: int) -> dict[str, Any]:
    index = np.arange(size, dtype=np.float64)[:, None]
    factors = np.arange(1, 9, dtype=np.float64)[None, :]
    values = np.sin(index * factors * 0.013)[:, None, :]
    weights = np.arange(1, size + 1, dtype=np.float64)
    weights /= np.sum(weights)
    law = PhysicalLaw("memory-physical", "deterministic", "eight-factors", "instant")
    scenarios, preparation = measure_host(
        lambda: FinancialScenarioSet(
            values,
            weights,
            np.asarray((0.0,), dtype=np.float64),
            np.ones((size, 1), dtype=np.bool_),
            law,
            "memory-factors",
            "memory-source",
        )
    )
    result, evidence = _measure(
        lambda: reduce_scenarios(scenarios, representatives, numeric_id="memory-reduced"),
        repeats,
    )
    selected = np.asarray(result.representative_indices)
    distances = np.sum(
        (values[:, None, :, :] - values[selected][None, :, :, :]) ** 2, axis=(2, 3)
    )
    reference_assignment = np.argmin(distances, axis=1)
    reference_distortion = np.sum(weights * np.min(distances, axis=1))
    return {
        "capacity": size,
        "representative_count": representatives,
        "flattened_factor_count": 8,
        "preparation_seconds": preparation,
        "retained_input_bytes": logical_array_bytes(scenarios),
        "representative_indices": selected.tolist(),
        "assignment_matches_dense": bool(
            np.array_equal(result.assignment, reference_assignment)
        ),
        "distortion": float(result.distortion),
        "dense_distortion_defect": float(
            abs(float(result.distortion) - reference_distortion)
        ),
        "source_numeric_id": result.source_numeric_id,
        "numeric_id": result.scenarios.numeric_id,
        **evidence,
    }


def _tie_contract() -> bool:
    values = np.asarray(
        (((0.0, 0.0),), ((2.0, 0.0),), ((1.0, 0.0),), ((0.0, 0.0),), ((2.0, 0.0),)),
        dtype=np.float64,
    )
    scenarios = FinancialScenarioSet(
        values,
        np.asarray((0.4, 0.15, 0.2, 0.1, 0.15), dtype=np.float64),
        np.asarray((0.0,), dtype=np.float64),
        np.ones((5, 1), dtype=np.bool_),
        PhysicalLaw("tie-law", "deterministic", "two-factors", "instant"),
        "tie-factors",
        "tie-source",
    )
    result = reduce_scenarios(scenarios, 2, numeric_id="tie-reduced")
    if not np.array_equal(result.assignment, (0, 1, 0, 0, 1)):
        raise RuntimeError("Scenario reduction overwrote a first-representative tie.")
    if not np.isclose(float(result.distortion), 0.2, rtol=1.0e-13, atol=1.0e-14):
        raise RuntimeError("Scenario reduction changed tie-fixture distortion.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--qh-sizes", type=int, nargs="+", default=[101, 201, 401, 801, 1601, 3201, 6401]
    )
    parser.add_argument(
        "--scenario-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--representatives", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
    root = Path(__file__).resolve().parents[1]
    qh_sizes = [5, 101] if args.smoke else args.qh_sizes
    scenario_sizes = [64, 128] if args.smoke else args.scenario_sizes
    representatives = 4 if args.smoke else args.representatives
    fields = (
        "environment",
        "command",
        "configuration",
        "quantum_hall",
        "scenarios",
        "scaling",
        "host_gates",
    )
    record: dict[str, Any] = {
        "identity": capture_benchmark_identity(root, Path(__file__), fields).to_dict(),
        "environment": capture_environment().to_dict(),
        "command": [sys.executable, "-m", "benchmarks.memory_host_exact", *sys.argv[1:]],
        "configuration": {
            "qh_sizes": qh_sizes,
            "scenario_sizes": scenario_sizes,
            "representatives": representatives,
            "repeats": args.repeats,
            "baseline": args.baseline,
        },
        "quantum_hall": [_quantum_hall(n, args.repeats) for n in qh_sizes],
        "scenarios": [
            _scenarios(n, representatives, args.repeats) for n in scenario_sizes
        ],
        "scaling": {},
        "host_gates": {},
    }
    for name, sizes in (("quantum_hall", qh_sizes), ("scenarios", scenario_sizes)):
        peaks = [case["host_allocation_peak_bytes"] for case in record[name]]
        record["scaling"][name] = {
            "host_peak_exponent": fitted_scaling_exponent(sizes, peaks),
            "largest_four_host_peak_exponent": fitted_scaling_exponent(
                sizes[-4:], peaks[-4:]
            ),
            "adjacent_peak_ratios": [
                b / a for a, b in zip(peaks[:-1], peaks[1:], strict=True)
            ],
            "workspace_formula": (
                ("O(N * N + W)" if args.baseline else "O(N + W)")
                if name == "quantum_hall"
                else ("O(N * R * D)" if args.baseline else "O(N * D + N + R * D)")
            ),
        }
    baseline_path = root / "benchmarks/memory_host_exact_baseline.json"
    if not args.baseline and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        for name, sizes in (("quantum_hall", qh_sizes), ("scenarios", scenario_sizes)):
            if [case["capacity"] for case in baseline[name]] == sizes:
                record["host_gates"][name] = evaluate_memory_gate(
                    sizes,
                    [case["host_allocation_peak_bytes"] for case in baseline[name]],
                    [case["host_allocation_peak_bytes"] for case in record[name]],
                    MemoryGatePolicy(),
                ).to_dict()
    record["process_peak_rss_bytes"] = process_peak_rss_bytes()
    record["first_tie_contract"] = _tie_contract()
    output = args.output or root / (
        "benchmarks/memory_host_exact_baseline.json"
        if args.baseline
        else "benchmarks/memory_host_exact.json"
    )
    write_json_atomic(output, record)
    print(
        json.dumps(
            {
                "output": str(output),
                "scaling": record["scaling"],
                "host_gates": record["host_gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
