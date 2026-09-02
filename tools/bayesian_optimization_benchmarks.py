# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Focused finite-candidate q/fantasy Bayesian-optimization benchmark."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def run(*, budget: int, batch_size: int, candidates: int, fantasies: int):
    domain = phx.uq.BayesianOptimizationDomain(
        jnp.asarray([0.5, 0.5]),
        lower_bounds=jnp.zeros((2,)),
        upper_bounds=jnp.ones((2,)),
    )
    problem = phx.uq.BayesianOptimizationProblem(
        lambda point: jnp.sum((point.continuous - jnp.asarray([0.2, 0.7])) ** 2),
        domain,
    )
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.Matern52Kernel(length_scale=0.3),
        noise_scale=0.0,
    )
    plan = phx.uq.GaussianProcessBayesianOptimization(
        budget,
        objective_surrogate=state,
        initial_evaluations=max(2, min(8, budget)),
        batch_size=batch_size,
        candidate_tuple_count=candidates,
        fantasy_count=fantasies,
    )
    started = perf_counter()
    result = phx.uq.bayesian_optimize(problem, plan, jr.key(0))
    elapsed = perf_counter() - started
    return {
        "budget": budget,
        "batch_size": batch_size,
        "candidate_tuple_count": candidates,
        "fantasy_count": fantasies,
        "seconds": elapsed,
        "best_objective": float(result.best_objective),
        "evaluation_count": result.evaluation_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--fantasies", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                budget=args.budget,
                batch_size=args.batch_size,
                candidates=args.candidates,
                fantasies=args.fantasies,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
