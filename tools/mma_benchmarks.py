#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def run_case(size: int, /) -> dict[str, object]:
    coefficients = jnp.linspace(0.5, 4.0, size)
    volume = 0.5 * size
    initial = jnp.full((size,), 0.5)
    problem = phx.optim.MinimizationProblem(
        lambda value, _: jnp.sum(coefficients / value),
        bounds=phx.optim.Bounds(0.05, 2.0),
        constraints=(
            phx.optim.NonlinearConstraint(
                lambda value, _: jnp.sum(value),
                upper=volume,
                constraint_id="volume",
            ),
        ),
        problem_id=f"mma-reciprocal-{size}",
    )
    started = time.perf_counter()
    result = phx.optim.minimize(
        problem,
        initial,
        method=phx.optim.MethodOfMovingAsymptotes(),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=2.0e-6,
            relative_optimality=0.0,
            maximum_steps=180,
        ),
    )
    jax.block_until_ready(result.objective)
    wall_seconds = time.perf_counter() - started
    expected = jnp.sqrt(coefficients)
    expected = expected / jnp.sum(expected) * volume
    relative_point_error = jnp.linalg.norm(
        result.parameters - expected
    ) / jnp.linalg.norm(expected)
    return {
        "size": size,
        "status": int(result.status),
        "objective": float(result.objective),
        "relative_point_error": float(relative_point_error),
        "volume_violation": float(jnp.maximum(jnp.sum(result.parameters) - volume, 0.0)),
        "optimality": float(result.diagnostics.final_optimality_norm),
        "iterations": int(result.diagnostics.iterations),
        "objective_evaluations": int(result.diagnostics.objective_evaluations),
        "constraint_evaluations": int(result.diagnostics.constraint_evaluations),
        "wall_seconds": wall_seconds,
        "finite": bool(np.isfinite(np.asarray(result.objective))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/mma.json"))
    parser.add_argument("--sizes", type=int, nargs="+", default=(16, 128))
    args = parser.parse_args()
    payload = {"cases": [run_case(int(size)) for size in args.sizes]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
