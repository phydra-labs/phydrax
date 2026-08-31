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
import optax

import phydrax as phx


def _solver(method: str, seed: int, points: int):
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        width_size=16,
        depth=2,
        key=jr.key(seed),
    )
    field = domain.Model("x")(model)

    @domain.Function("x")
    def target(x):
        return jnp.exp(-200.0 * (x[0] - 0.73) ** 2)

    condition = phx.conditions.Residual("u", component, lambda value: value - target)
    sampling = phx.domain.PointSampling(
        points,
        layout=phx.domain.SampleLayout((("x",),)),
        design="sobol_scrambled",
    )
    if method == "attention":
        source = phx.integration.adaptive(
            phx.integration.mean_over(component),
            sampling,
            phx.sampling.collocation.ResidualAttentionCollocation(
                refresh_every=5,
                decay=0.9,
                minimum_ess_fraction=0.35,
            ),
        )
    else:
        batch = component.sample(sampling, key=jr.fold_in(jr.key(seed), 1))
        source = phx.integration.fixed(
            phx.integration.from_samples(phx.integration.mean_over(component), batch)
        )
    train_term = phx.terms.ResidualPenalty(condition, source, label="train")
    monitor_batch = component.sample(
        phx.domain.PointSampling(
            4 * points,
            layout=phx.domain.SampleLayout((("x",),)),
            design="halton_scrambled",
        ),
        key=jr.fold_in(jr.key(seed), 2),
    )
    monitor = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(
            phx.integration.from_samples(
                phx.integration.mean_over(component), monitor_batch
            )
        ),
        label="monitor",
    )
    return phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(train_term,),
        evaluation_terms=(monitor,),
    ), monitor


def _run(method: str, seed: int, points: int, iterations: int):
    solver, monitor = _solver(method, seed, points)
    started = time.perf_counter()
    trained = solver.solve(
        num_iter=iterations,
        optim=optax.adam(2e-3),
        seed=seed + 100,
        jit=True,
        keep_best=False,
        log_every=0,
    )
    jax.block_until_ready(trained.functions)
    elapsed = time.perf_counter() - started
    monitor_loss = monitor.loss(trained.ansatz_functions(), key=jr.key(seed + 200))
    return {
        "method": method,
        "seed": seed,
        "points": points,
        "iterations": iterations,
        "monitor_loss": float(monitor_loss),
        "wall_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    args = parser.parse_args()
    rows = [
        _run(method, seed, args.points, args.iterations)
        for method in ("fixed", "attention")
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
