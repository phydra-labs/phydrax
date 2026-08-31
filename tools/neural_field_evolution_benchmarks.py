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
import jax.random as jr

import phydrax as phx
from examples.neural_galerkin_advection import FourierMode


def _run(formulation: str, samples: int, seed: int) -> dict[str, object]:
    speed = 0.5
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Function("x")(FourierMode(jnp.asarray([1.0, 0.0])))
    component = domain.component()
    batch = component.sample(
        phx.domain.PointSampling(
            samples,
            layout=phx.domain.SampleLayout((("x",),)),
            design="sobol_scrambled",
        ),
        key=jr.key(seed),
    )
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component), batch
    )
    problem = phx.solver.NeuralGalerkinProblem(
        {"u": field},
        lambda _time, functions, _args: {
            "u": -speed * phx.operators.partial_n(functions["u"], var="x", order=1)
        },
        (phx.solver.FieldProjectionMetric("u", realization),),
        problem_id=f"advection:{formulation}:{samples}:{seed}",
    )
    damping = 1e-8
    preconditioner = (
        phx.linalg.RandomizedNystromPreconditionerBuilder(
            2,
            oversampling=0,
            shift=damping,
            seed=seed,
        )
        if formulation == "gram-nystrom"
        else None
    )
    tangent = phx.solver.NeuralTangentSolvePolicy(
        "gram" if formulation.startswith("gram") else "rectangular",
        damping=damping,
        preconditioner=preconditioner,
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 1.0, 11),
        time_id="advection-benchmark-grid",
    )
    started = time.perf_counter()
    result = phx.solver.solve_neural_galerkin(
        problem,
        grid,
        tangent=tangent,
        rtol=1e-7,
        atol=1e-9,
    )
    jax.block_until_ready(result.parameter_solution.states)
    elapsed = time.perf_counter() - started
    prediction = result.field_at(grid.num_times - 1, "u")(batch)
    points = jnp.asarray(batch.points["x"].data)
    exact = jnp.sin(2.0 * jnp.pi * (points[..., 0] - speed))
    error_norm = jnp.sqrt(jnp.sum(jnp.abs(prediction.data - exact) ** 2))
    target_norm = jnp.sqrt(jnp.sum(jnp.abs(exact) ** 2))
    return {
        "formulation": formulation,
        "samples": samples,
        "seed": seed,
        "successful": bool(result.successful),
        "relative_l2_error": float(error_norm / target_norm),
        "max_projection_defect": float(jnp.max(result.audit.relative_projection_defect)),
        "accepted_steps": int(result.parameter_solution.stats["num_accepted_steps"]),
        "rejected_steps": int(result.parameter_solution.stats["num_rejected_steps"]),
        "audit_matvecs": int(jnp.sum(result.audit.matvec_count)),
        "wall_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--formulations",
        nargs="+",
        choices=("rectangular", "gram", "gram-nystrom"),
        default=("rectangular", "gram", "gram-nystrom"),
    )
    args = parser.parse_args()
    rows = [
        _run(formulation, args.samples, seed)
        for formulation in args.formulations
        for seed in args.seeds
    ]
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
