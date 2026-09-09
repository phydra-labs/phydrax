# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Exact-geometry and seeded mixed/constrained/noisy qHVI benchmarks.

Run from the repository root with PYTHONPATH=. The stochastic comparison uses
true feasible objectives for held-out reporting, not noisy observed scores.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.optim._pareto import hypervolume
from phydrax.uq._multiobjective_bayesian_optimization import (
    GaussianProcessMultiObjectiveBayesianOptimization,
    multiobjective_bayesian_optimize,
    MultiObjectiveBayesianOptimizationProblem,
)


def exact_geometry(*, repetitions: int):
    front = jnp.asarray([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
    reference = jnp.asarray([5.0, 5.0])
    improved = jnp.concatenate((front, jnp.asarray([[1.0, 1.0]])))
    exact = jax.jit(hypervolume)
    value = exact(front, reference).block_until_ready()
    improvement = (exact(improved, reference) - value).block_until_ready()
    if float(value) != 11.0 or float(improvement) != 5.0:
        raise AssertionError("Exact 2D front must have volume 11 and improvement 5.")
    three_dimensional = jnp.concatenate((front, jnp.ones((3, 1))), axis=1)
    reference_3d = jnp.asarray([5.0, 5.0, 4.0])
    volume_3d = exact(three_dimensional, reference_3d).block_until_ready()
    if float(volume_3d) != 33.0:
        raise AssertionError("Extruded exact 3D front must have volume 33.")
    started = perf_counter()
    for _ in range(repetitions):
        exact(three_dimensional, reference_3d).block_until_ready()
    return {
        "hypervolume_2d": float(value),
        "improvement_2d": float(improvement),
        "hypervolume_3d": float(volume_3d),
        "repetitions": repetitions,
        "seconds": perf_counter() - started,
    }


def stochastic_search(
    *,
    seed: int,
    budget: int,
    batch_size: int,
    candidates: int,
    fantasies: int,
    objectives: int,
):
    names = ("cost", "loss") if objectives == 2 else ("cost", "loss", "burden")
    categories = phx.optim.FiniteProductSpace(
        {"material": phx.optim.FiniteAxis(jnp.asarray([0.0, 1.0]))}
    )
    domain = phx.uq.BayesianOptimizationDomain(
        jnp.asarray([0.5]),
        lower_bounds=jnp.asarray([0.0]),
        upper_bounds=jnp.asarray([1.0]),
        categorical=categories,
    )

    def truth(point):
        x, material = point.continuous[0], point.categorical["material"]
        values = ((x - 0.1) ** 2 + 0.07 * material, (x - 0.9) ** 2 - 0.04 * material)
        if objectives == 3:
            values += ((x - 0.45) ** 2 + 0.03 * material,)
        return jnp.stack(values)

    def objective(point, key):
        return truth(point) + 0.025 * jr.normal(key, (objectives,))

    def constraint(point, key):
        return point.continuous[0] - 0.85 + 0.01 * jr.normal(key)

    reference = jnp.full((objectives,), 1.5)
    problem = MultiObjectiveBayesianOptimizationProblem(
        objective,
        domain,
        objective_names=names,
        directions=("min",) * objectives,
        scales=jnp.ones((objectives,)),
        reference=reference,
        constraints=(constraint,),
        pending=(domain.decode(jnp.asarray([0.75]), jnp.asarray(1)),),
    )
    state = phx.uq.MultiOutputGaussianProcessLikelihoodState(
        kernel=phx.uq.IntrinsicCoregionalizationKernel(
            phx.kernels.Matern52Kernel(length_scale=0.3),
            phx.uq.Coregionalization(
                jnp.full((objectives, 1), 0.5),
                jnp.full((objectives,), 0.5),
                output_names=names,
            ),
        ),
        noise_scale=jnp.full((objectives,), 0.025),
        jitter=1e-6,
    )
    plan = GaussianProcessMultiObjectiveBayesianOptimization(
        budget,
        objective_surrogate=state,
        constraint_surrogates=(
            phx.uq.GaussianProcessLikelihoodState(noise_scale=0.01, jitter=1e-6),
        ),
        initial_evaluations=min(4, budget),
        batch_size=batch_size,
        candidate_tuple_count=candidates,
        fantasy_count=fantasies,
    )
    started = perf_counter()
    result = multiobjective_bayesian_optimize(problem, plan, jr.key(seed))
    result.observed_hypervolume.block_until_ready()
    elapsed = perf_counter() - started
    true_values = jnp.stack(tuple(truth(record.point) for record in result.observations))
    true_feasible = jnp.stack(
        tuple(record.point.continuous[0] <= 0.85 for record in result.observations)
    )
    initial_count = plan.initial_evaluations
    initial_hv = hypervolume(
        true_values[:initial_count], reference, valid=true_feasible[:initial_count]
    )
    true_hv = hypervolume(true_values, reference, valid=true_feasible)
    random_key, category_key = jr.split(jr.fold_in(jr.key(seed), 991))
    random_units = jr.uniform(random_key, (budget, 1))
    random_categories = jr.randint(category_key, (budget,), 0, categories.size)
    random_points = tuple(
        domain.decode(unit, category)
        for unit, category in zip(random_units, random_categories, strict=True)
    )
    random_values = jnp.stack(tuple(truth(point) for point in random_points))
    random_feasible = jnp.stack(
        tuple(point.continuous[0] <= 0.85 for point in random_points)
    )
    random_hv = hypervolume(random_values, reference, valid=random_feasible)
    return {
        "seed": seed,
        "objective_count": objectives,
        "seconds": elapsed,
        "evaluation_count": result.evaluation_count,
        "batch_size": batch_size,
        "candidate_tuple_count": candidates,
        "fantasy_count": fantasies,
        "observed_hypervolume": float(result.observed_hypervolume),
        "initial_true_hypervolume": float(initial_hv),
        "final_true_hypervolume": float(true_hv),
        "random_true_hypervolume": float(random_hv),
        "acquisition_estimates": result.acquisition_estimates.tolist(),
        "acquisition_standard_errors": result.acquisition_standard_errors.tolist(),
        "observed_pareto_objectives": result.observed_pareto_objectives.tolist(),
        "scored_tuple_count": result.scored_tuple_count,
        "estimated_peak_bytes": result.estimated_peak_bytes,
        "hypervolume_work_bound": result.hypervolume_work_bound,
        "pending_id": result.pending_id,
        "work_id": result.work_id,
        "termination_reason": result.termination_reason,
        "globally_optimal": result.globally_optimal,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("exact", "search", "all"), default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--fantasies", type=int, default=32)
    parser.add_argument("--objectives", type=int, choices=(2, 3), default=2)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    results = {}
    if args.case in ("exact", "all"):
        results["exact"] = exact_geometry(repetitions=args.repetitions)
    if args.case in ("search", "all"):
        results["search"] = stochastic_search(
            seed=args.seed,
            budget=args.budget,
            batch_size=args.batch_size,
            candidates=args.candidates,
            fantasies=args.fantasies,
            objectives=args.objectives,
        )
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
