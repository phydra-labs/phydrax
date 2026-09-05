# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Run actual native energy solves; report time, status, and independent replay.

Usage: PYTHONPATH=. JAX_ENABLE_X64=1 python tools/energy_planning_benchmarks.py
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import equinox as eqx
import jax
import numpy as np

from phydrax.applications import energy_planning as ep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("heat", "hydrogen"), default="heat")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    example = (
        ep.electricity_heat_storage_example
        if args.example == "heat"
        else ep.electricity_hydrogen_example
    )
    spec = example(exact=args.exact)
    started = perf_counter()
    compiled = ep.compile_energy_system(spec, exact=args.exact)
    compile_seconds = perf_counter() - started
    rows = []
    successful = True
    for repeat in range(args.repeats):
        refresh_seconds = 0.0
        if repeat:
            costs = np.asarray(spec.sources[0].marginal_cost) * (1 + 0.01 * repeat)
            refreshed_spec = eqx.tree_at(
                lambda system: system.sources[0].marginal_cost, spec, costs
            )
            started = perf_counter()
            compiled = ep.refresh_energy_system(compiled, refreshed_spec)
            refresh_seconds = perf_counter() - started
        started = perf_counter()
        solution = ep.solve_energy_system(compiled)
        jax.block_until_ready(solution.native_result.primal)
        seconds = perf_counter() - started
        successful = successful and solution.successful
        rows.append(
            {
                "repeat": repeat,
                "refresh_seconds": refresh_seconds,
                "solve_and_replay_seconds": seconds,
                "native_status": int(solution.native_result.status),
                "successful": solution.successful,
                "cost": float(solution.replay.cost),
                "emissions": float(solution.replay.emissions),
                "objective_error": float(solution.replay.objective_error),
                "maximum_physical_violation": float(
                    solution.replay.maximum_physical_violation
                ),
                "physical_failures": list(solution.replay.failures),
            }
        )
    print(
        json.dumps(
            {
                "example": args.example,
                "exact": args.exact,
                "backend": "phydrax",
                "compile_seconds": compile_seconds,
                "variables": compiled.program.num_variables,
                "semantic_rows": len(compiled.rows),
                "binary_variables": len(compiled.binary_indices),
                "executions": rows,
            },
            indent=2,
        )
    )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
