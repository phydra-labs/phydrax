#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated A14 Kubo memory gate; baseline must precede the production edit.

Run --phase baseline, then --phase candidate after applying the proposal. Frequency
capacities vary at fixed k/band/Cartesian sizes. Timings under contention are evidence,
not acceptance criteria. Full raw transitions and response evidence remain outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import numpy as np
from numpy.typing import NDArray

from phydrax.chemistry.periodic._kubo import (
    finite_frequency_kubo_response,
    KuboDiamagneticSumRule,
    KuboLinewidth,
    PeriodicKuboPlan,
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
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)


_E = 1.602176634e-19
_HBAR = 1.0545718176461565e-34


@runtime_checkable
class _KuboEntry(Protocol):
    def __call__(self, value: PeriodicKuboPlan, /) -> Any: ...

    def lower(self, value: PeriodicKuboPlan, /) -> Any: ...


_FIELDS = ("environment", "command", "configuration", "cases", "gate", "scaling")


def _plan(k_count: int, bands: int, dimension: int) -> PeriodicKuboPlan:
    energies = np.broadcast_to(np.linspace(-0.8, 0.8, bands), (k_count, bands)) * _E
    rng = np.random.default_rng(814)
    values = rng.normal(size=(k_count, bands, bands, dimension))
    velocities = 1.0e5 * (values + values.swapaxes(1, 2))
    weights = np.full(k_count, 1.0 / k_count, dtype=np.float64)
    occupations = 1.0 / (1.0 + np.exp(energies / (1.380649e-23 * 300.0)))
    rule = np.zeros((dimension, dimension), dtype=np.float64)
    for k in range(k_count):
        for n in range(bands):
            derivative = (
                occupations[k, n] * (1.0 - occupations[k, n]) / (1.380649e-23 * 300.0)
            )
            rule += (
                0.5
                * weights[k]
                * derivative
                * np.outer(velocities[k, n, n], velocities[k, n, n])
            )
            for m in range(n + 1, bands):
                factor = (
                    weights[k]
                    * (occupations[k, n] - occupations[k, m])
                    / (energies[k, m] - energies[k, n])
                )
                rule += factor * np.outer(velocities[k, n, m], velocities[k, n, m])
    rule *= np.pi * _E**2 / 1.0e-28
    return PeriodicKuboPlan(
        energies,
        velocities,
        weights,
        KuboDiamagneticSumRule(rule, source_id="independent-loop-memory-fixture"),
        chemical_potential_joule=0.0,
        temperature_kelvin=300.0,
        cell_volume_m3=1.0e-28,
    )


def _reference(
    plan: PeriodicKuboPlan,
    frequencies: NDArray[np.float64],
    width: float,
    *,
    include_antiresonant: bool = True,
) -> NDArray[np.complex128]:
    energies = np.asarray(plan.energies_joule)
    velocities = np.asarray(plan.velocity_matrices_m_per_s)
    weights = np.asarray(plan.k_weights)
    occupations = 1.0 / (1.0 + np.exp(energies / (1.380649e-23 * 300.0)))
    result = np.zeros(
        (frequencies.size, plan.dimension, plan.dimension), dtype=np.complex128
    )
    for k in range(energies.shape[0]):
        for n in range(energies.shape[1]):
            for m in range(n + 1, energies.shape[1]):
                delta = energies[k, m] - energies[k, n]
                factor = weights[k] * (occupations[k, n] - occupations[k, m]) / delta
                product = np.outer(velocities[k, n, m], velocities[k, n, m].conjugate())
                resonant = 1.0 / (frequencies - delta / _HBAR + 1j * width / _HBAR)
                poles = resonant[:, None, None] * product[None]
                if include_antiresonant:
                    antiresonant = 1.0 / (
                        frequencies + delta / _HBAR + 1j * width / _HBAR
                    )
                    poles += antiresonant[:, None, None] * product.T[None]
                result += 1j * _E**2 / plan.cell_volume_m3 * factor * poles
    return result


def _case(capacity: int, args: argparse.Namespace) -> dict[str, Any]:
    (plan, host_peak), preparation = measure_host(
        lambda: host_allocation_peak(
            lambda: _plan(args.k_count, args.bands, args.dimension)
        )
    )
    frequencies = np.linspace(0.1, 2.0, capacity) * _E / _HBAR
    linewidth = KuboLinewidth(0.03 * _E, mechanism_id="memory-fixture-interband")

    def response(value: PeriodicKuboPlan) -> Any:
        return finite_frequency_kubo_response(value, frequencies, linewidth)

    entry = eqx.filter_jit(response)
    if not isinstance(entry, _KuboEntry):
        raise TypeError("Kubo benchmark entry must support lowering.")
    executable, timing = measure_lower_and_compile(
        lambda: entry.lower(plan), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(
        lambda: executable(plan), warmup=1, repeats=args.repeats
    )
    compiler = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="jax-compiled-full-periodic-kubo-response",
        unavailable_reason="Compiler did not supply complete analysis.",
    )
    chunk = capacity if args.phase == "baseline" else min(capacity, 8)
    row = {
        "capacity": capacity,
        "preparation_seconds": preparation,
        "host_allocation_peak_bytes": host_peak,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(compiler),
        "retained_plan_bytes": logical_array_bytes(plan),
        "retained_output_bytes": logical_array_bytes(result),
        "logical_frequency_transition_workspace_bytes": 16
        * chunk
        * args.k_count
        * args.bands**2
        * args.dimension**2,
        "frequency_chunk_size": chunk,
        "successful": bool(result.successful),
        "passive": bool(result.evidence.passive),
        "f_sum_relative_residual": float(result.evidence.f_sum_relative_residual),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "timing_acceptance": "deferred-to-quiet-serialized-wave",
    }
    reference = _reference(plan, frequencies, linewidth.energy_width_joule)
    observed = np.asarray(result.regular_conductivity_siemens_per_m)
    np.testing.assert_allclose(observed, reference, rtol=2.0e-11, atol=2.0e-13)
    scale = float(np.max(np.abs(reference)))
    row["maximum_relative_reference_defect"] = (
        float(np.max(np.abs(observed - reference))) / scale
    )
    mutant = _reference(
        plan, frequencies, linewidth.energy_width_joule, include_antiresonant=False
    )
    mutant_defect = float(np.max(np.abs(mutant - reference))) / scale
    if mutant_defect <= 2.0e-11:
        raise ValueError(
            "Independent reference did not detect the dropped antiresonant pole."
        )
    row["dropped_antiresonant_relative_defect"] = mutant_defect
    row["dropped_antiresonant_fault_detected"] = True
    if args.phase == "candidate":

        def parallel_response(value: PeriodicKuboPlan) -> Any:
            return finite_frequency_kubo_response(
                value, frequencies, linewidth, frequency_policy="parallel"
            )

        parallel_entry = eqx.filter_jit(parallel_response)
        if not isinstance(parallel_entry, _KuboEntry):
            raise TypeError("Kubo benchmark entry must support lowering.")
        parallel_executable, parallel_timing = measure_lower_and_compile(
            lambda: parallel_entry.lower(plan), lambda lowered: lowered.compile()
        )
        parallel_result, parallel_warm = measure_repeated(
            lambda: parallel_executable(plan), warmup=1, repeats=args.repeats
        )
        np.testing.assert_allclose(
            parallel_result.regular_conductivity_siemens_per_m,
            reference,
            rtol=2.0e-11,
            atol=2.0e-13,
        )
        parallel_compiler = compiler_evidence(
            parallel_executable.compiled.cost_analysis(),
            parallel_executable.compiled.memory_analysis(),
            source="jax-compiled-parallel-periodic-kubo-response",
            unavailable_reason="Compiler did not supply complete analysis.",
        )
        row["parallel"] = {
            "lowering_seconds": parallel_timing.lowering_seconds,
            "compilation_seconds": parallel_timing.compilation_seconds,
            "compiler": asdict(parallel_compiler),
            "warm": parallel_warm.to_seconds_dict(),
            "successful": bool(parallel_result.successful),
            "frequency_chunk_size": capacity,
            "maximum_relative_reference_defect": float(
                np.max(
                    np.abs(
                        np.asarray(parallel_result.regular_conductivity_siemens_per_m)
                        - reference
                    )
                )
            )
            / scale,
        }
    jax.clear_caches()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 32, 128, 512])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--k-count", type=int, default=16)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.sizes = [8, 32]
    if len(args.sizes) < 2 or any(a >= b for a, b in zip(args.sizes, args.sizes[1:])):
        parser.error("Sizes must contain at least two strictly increasing capacities.")
    root = Path(__file__).resolve().parents[1]
    baseline_path = root / "benchmarks/memory_periodic_kubo_baseline.json"
    output = args.output or (
        baseline_path
        if args.phase == "baseline"
        else root / "benchmarks/memory_periodic_kubo.json"
    )
    cases = [_case(size, args) for size in args.sizes]
    temporary = [case["compiler"]["temporary_bytes"] for case in cases]
    logical = [case["logical_frequency_transition_workspace_bytes"] for case in cases]
    scaling = {
        "controlling_capacity": "frequency_count",
        "declared_workspace_formula": "16 * min(F, frequency_chunk_size) * K * bands**2 * dimension**2",
        "declared_logical_exponent": 1.0 if args.phase == "baseline" else 0.0,
        "logical_fitted_exponent": fitted_scaling_exponent(args.sizes, logical),
        "declared_compiler_exponent_upper_bound": 1.1,
        "compiler_fitted_exponent": fitted_scaling_exponent(args.sizes, temporary),
        "compiler_formula": "constant transition workspace plus O(F * dimension**2) passivity/output scratch",
    }
    gate = None
    if args.phase == "candidate":
        baseline = json.loads(baseline_path.read_text())
        if any(
            size not in baseline["configuration"]["sizes"] for size in args.sizes
        ) or any(
            baseline["configuration"][name] != vars(args)[name]
            for name in ("k_count", "bands", "dimension")
        ):
            raise ValueError("Baseline and candidate capacities must match.")
        gate = evaluate_memory_gate(
            args.sizes,
            [
                case["compiler"]["temporary_bytes"]
                for case in baseline["cases"]
                if case["capacity"] in args.sizes
            ],
            temporary,
            MemoryGatePolicy(),
        ).to_dict()
        gate["logical_scaling_matches"] = (
            abs(scaling["logical_fitted_exponent"] - scaling["declared_logical_exponent"])
            <= 0.1
        )
        gate["compiler_scaling_matches"] = (
            scaling["compiler_fitted_exponent"]
            <= scaling["declared_compiler_exponent_upper_bound"]
        )
        if not gate["logical_scaling_matches"] or not gate["compiler_scaling_matches"]:
            gate["decision"] = "retain-baseline"
    payload = {
        "identity": capture_benchmark_identity(root, Path(__file__), _FIELDS).to_dict(),
        "environment": capture_environment().to_dict(),
        "command": [
            sys.executable,
            "-m",
            "benchmarks.memory_periodic_kubo",
            *sys.argv[1:],
        ],
        "configuration": {
            "phase": args.phase,
            "sizes": args.sizes,
            "k_count": args.k_count,
            "bands": args.bands,
            "dimension": args.dimension,
            "repeats": args.repeats,
        },
        "cases": cases,
        "gate": gate,
        "scaling": scaling,
    }
    write_json_atomic(output, payload)
    print(json.dumps({"output": str(output), "gate": gate, "scaling": scaling}, indent=2))


if __name__ == "__main__":
    main()
