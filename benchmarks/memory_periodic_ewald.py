"""Exact Ewald real-space workspace, including force and strain reverse mode.

Capture --baseline on the untouched dense implementation before remediation.
Defaults are a small smoke sweep; --sizes 16 32 64 128 --shells 1 2 is the gate.
Compiler memory is not process RSS. Warm timings under contention are evidence,
not a timing adoption decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.atomistic import AtomisticUnitSystem
from phydrax.chemistry.periodic import PeriodicEwaldPlan, PreparedPeriodicEwald
from phydrax.discretization import PeriodicCell

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


def _dense_minimum(positions: np.ndarray, cell: np.ndarray) -> float:
    fractional = positions @ np.linalg.inv(cell)
    displacement = fractional[:, None, :] - fractional[None, :, :]
    displacement -= np.rint(displacement)
    distances = np.linalg.norm(displacement @ cell, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances, initial=np.inf))


def _derivatives(
    prepared: PreparedPeriodicEwald, positions: Array, charges: Array, cell: Array
) -> tuple[Array, tuple[Array, Array, Array, Array], Array, Array]:
    (energy, components), gradient = jax.value_and_grad(
        prepared._energy, argnums=0, has_aux=True
    )(positions, charges, cell)

    def strained(strain: Array) -> Array:
        deformation = jnp.eye(3, dtype=cell.dtype) + strain
        return prepared._energy(positions @ deformation.T, charges, cell @ deformation.T)[
            0
        ]

    strain_gradient = jax.grad(strained)(jnp.zeros_like(cell))
    stress = (strain_gradient + strain_gradient.T) / (2.0 * jnp.abs(jnp.linalg.det(cell)))
    return energy, components, -gradient, stress


def _case(size: int, shell: int, args: argparse.Namespace) -> dict[str, Any]:
    cell_host = np.asarray(
        [[5.0, 0.0, 0.0], [0.6, 4.8, 0.0], [0.3, 0.2, 5.2]], dtype=np.float64
    )
    # A deterministic separated lattice avoids geometry generation noise.
    side = int(np.ceil(size ** (1.0 / 3.0)))
    indices = np.arange(size, dtype=np.int64)
    fractions = (
        np.stack(
            (indices % side, (indices // side) % side, indices // (side * side)), axis=1
        )
        / side
        + 0.05
    )
    positions_host = fractions @ cell_host
    charges_host = np.where(indices % 2 == 0, 1.0, -1.0).astype(np.float64)
    charges_host -= np.mean(charges_host)
    kwargs: dict[str, Any] = {}
    if not args.baseline:
        kwargs = {
            "pair_tile_size": args.pair_tile_size,
            "translation_tile_size": args.translation_tile_size,
        }
    started = perf_counter()
    prepared, preparation_peak = host_allocation_peak(
        lambda: PeriodicEwaldPlan(
            PeriodicCell(cell_host, periodic_axes=(True, True, True)),
            AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
            0.8,
            real_shell=shell,
            reciprocal_shell=shell,
            **kwargs,
        ).prepare()
    )
    preparation_seconds = perf_counter() - started
    positions = jnp.asarray(positions_host, dtype=jnp.float64)
    charges = jnp.asarray(charges_host, dtype=jnp.float64)
    cell = jnp.asarray(cell_host, dtype=jnp.float64)
    minimum, minimum_peak = host_allocation_peak(
        lambda: (
            _dense_minimum(positions_host, cell_host)
            if args.baseline
            else prepared._minimum_pair_distance(positions_host, cell_host)
        )
    )
    records: dict[str, Any] = {}
    for name in ("forward", "force-and-strain"):
        if name == "forward":

            def operation(p: Array, q: Array, c: Array) -> Any:
                return prepared._energy(p, q, c)
        else:

            def operation(p: Array, q: Array, c: Array) -> Any:
                return _derivatives(prepared, p, q, c)

        entry = jax.jit(operation)
        executable, timing = measure_lower_and_compile(
            partial(entry.lower, positions, charges, cell),
            lambda lowered: lowered.compile(),
        )
        result, warm = measure_repeated(
            partial(executable, positions, charges, cell), warmup=1, repeats=args.repeats
        )
        compiler = compiler_evidence(
            executable.cost_analysis(),
            executable.memory_analysis(),
            source="jax-periodic-ewald",
        )
        records[name] = {
            "lowering_seconds": timing.lowering_seconds,
            "compilation_seconds": timing.compilation_seconds,
            "warm": warm.to_seconds_dict(),
            "compiler": asdict(compiler),
            "output_bytes": logical_array_bytes(result),
            "energy": float(result[0]),
            "components": [float(value) for value in result[1]],
            "energy_closure_residual": abs(
                float(result[0]) - sum(float(value) for value in result[1])
            ),
            "finite": bool(
                jnp.isfinite(result[0]) & jnp.all(jnp.isfinite(jnp.asarray(result[1])))
            ),
        }
        if name == "force-and-strain":
            records[name]["force_balance_residual"] = float(
                jnp.max(jnp.abs(jnp.sum(result[2], axis=0)))
            )
            records[name]["stress_symmetry_residual"] = float(
                jnp.max(jnp.abs(result[3] - result[3].T))
            )
            records[name]["successful"] = (
                records[name]["finite"]
                and bool(
                    jnp.all(jnp.isfinite(result[2])) & jnp.all(jnp.isfinite(result[3]))
                )
                and records[name]["force_balance_residual"] <= 1.0e-9
                and records[name]["stress_symmetry_residual"] <= 1.0e-9
                and records[name]["energy_closure_residual"] <= 1.0e-9
                and minimum > prepared.plan.minimum_distance
            )
        del executable, entry
        jax.clear_caches()
    pair = size if args.baseline else min(size, args.pair_tile_size)
    translation = (
        (2 * shell + 1) ** 3
        if args.baseline
        else min((2 * shell + 1) ** 3, args.translation_tile_size)
    )
    return {
        "size": size,
        "shell": shell,
        "pair_tile_size": pair,
        "translation_tile_size": translation,
        "logical_displacement_workspace_bytes": 3 * 8 * pair * pair * translation,
        "logical_host_pair_workspace_bytes": 3 * 8 * pair * pair,
        "logical_retained_bytes": logical_array_bytes(prepared),
        "preparation_seconds": preparation_seconds,
        "preparation_host_peak_bytes": preparation_peak,
        "minimum_distance_host_peak_bytes": minimum_peak,
        "minimum_pair_distance": minimum,
        "plan_id": prepared.plan.plan_id,
        "prepared_id": prepared.prepared_id,
        "phases": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--shells", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--pair-tile-size", type=int, default=16)
    parser.add_argument("--translation-tile-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.baseline and "pair_tile_size" in PeriodicEwaldPlan.__annotations__:
        raise ValueError("Capture the baseline on the untouched dense implementation.")
    if (
        len(args.sizes) < 2
        or args.sizes != sorted(set(args.sizes))
        or min(args.sizes) < 2
    ):
        raise ValueError(
            "Provide at least two increasing distinct sizes of at least two."
        )
    cases = [_case(size, shell, args) for shell in args.shells for size in args.sizes]
    fields = (
        "environment",
        "command",
        "configuration",
        "cases",
        "scaling",
        "gates",
        "process_peak_rss_bytes",
    )
    record: dict[str, Any] = {
        "identity": capture_benchmark_identity(
            Path.cwd(), Path(__file__), fields
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "command": sys.argv,
        "configuration": vars(args)
        | {"output": str(args.output) if args.output else None},
        "cases": cases,
        "scaling": [],
        "gates": [],
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }
    baseline_path = Path(__file__).with_name("memory_periodic_ewald_baseline.json")
    baseline = None if args.baseline else json.loads(baseline_path.read_text())
    for shell in args.shells:
        selected = [case for case in cases if case["shell"] == shell]
        record["scaling"].append(
            {
                "shell": shell,
                "logical_displacement_exponent": fitted_scaling_exponent(
                    args.sizes,
                    [case["logical_displacement_workspace_bytes"] for case in selected],
                ),
                "compiler_derivative_exponent": fitted_scaling_exponent(
                    args.sizes,
                    [
                        case["phases"]["force-and-strain"]["compiler"]["temporary_bytes"]
                        for case in selected
                    ],
                ),
                "host_minimum_distance_exponent": fitted_scaling_exponent(
                    args.sizes,
                    [case["minimum_distance_host_peak_bytes"] for case in selected],
                ),
                "logical_host_pair_exponent": fitted_scaling_exponent(
                    args.sizes,
                    [case["logical_host_pair_workspace_bytes"] for case in selected],
                ),
                "maximum_doubled_derivative_workspace_ratio": max(
                    (
                        later["phases"]["force-and-strain"]["compiler"]["temporary_bytes"]
                        / earlier["phases"]["force-and-strain"]["compiler"][
                            "temporary_bytes"
                        ]
                        for earlier, later in zip(selected, selected[1:])
                        if later["size"] == 2 * earlier["size"]
                    ),
                    default=None,
                ),
                "maximum_doubled_logical_workspace_ratio": max(
                    (
                        later["logical_displacement_workspace_bytes"]
                        / earlier["logical_displacement_workspace_bytes"]
                        for earlier, later in zip(selected, selected[1:])
                        if later["size"] == 2 * earlier["size"]
                    ),
                    default=None,
                ),
            }
        )
        if baseline is not None:
            old = [
                case
                for case in baseline["cases"]
                if case["shell"] == shell and case["size"] in args.sizes
            ]
            if [case["size"] for case in old] != args.sizes:
                raise ValueError("Baseline capacities must match the candidate sweep.")
            for phase in ("forward", "force-and-strain"):
                gate = evaluate_memory_gate(
                    args.sizes,
                    [
                        case["phases"][phase]["compiler"]["temporary_bytes"]
                        for case in old
                    ],
                    [
                        case["phases"][phase]["compiler"]["temporary_bytes"]
                        for case in selected
                    ],
                    MemoryGatePolicy(),
                )
                record["gates"].append({"shell": shell, "phase": phase, **gate.to_dict()})
                for current, previous in zip(selected, old):
                    current["phases"][phase]["baseline_energy_defect"] = abs(
                        current["phases"][phase]["energy"]
                        - previous["phases"][phase]["energy"]
                    )
    output = args.output or Path(__file__).with_name(
        "memory_periodic_ewald_baseline.json"
        if args.baseline
        else "memory_periodic_ewald.json"
    )
    write_json_atomic(output, record)
    print(
        json.dumps(
            {
                "output": str(output),
                "scaling": record["scaling"],
                "gates": record["gates"],
                "process_peak_rss_bytes": record["process_peak_rss_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
