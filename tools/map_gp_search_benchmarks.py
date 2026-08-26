#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark the private sequential GP initializer for bounded MAP search."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.optim._differential_evolution import _bounded_differential_evolution
from phydrax.optim._pytree import _PyTreeVectorizer
from phydrax.uq._map_gp_search import (
    _bounded_gaussian_process_map_search,
    GaussianProcessMAPSearch,
)


@dataclass(frozen=True)
class _Problem:
    name: str
    posterior: phx.uq.PosteriorProblem
    vectorizer: _PyTreeVectorizer
    objective: Callable[[jax.Array], jax.Array]
    initial: jax.Array
    lower: jax.Array
    upper: jax.Array
    target: float
    target_basis: str
    overhead_objective: Callable[[jax.Array], jax.Array] | None = None


@dataclass(frozen=True)
class _Trace:
    method: str
    best_vector: np.ndarray
    best_history: np.ndarray
    evaluation_counts: np.ndarray
    objective_evaluations: int
    proposal_seconds: float | None
    objective_seconds: float | None
    metadata: dict[str, object]


class _FlatPosteriorObjective(eqx.Module):
    posterior: phx.uq.PosteriorProblem
    vectorizer: _PyTreeVectorizer

    def __call__(self, vector):
        return self.posterior.negative_log_density(self.vectorizer.unravel(vector))


def _flat_problem(name, posterior, initial, lower, upper, target, target_basis):
    vectorizer = _PyTreeVectorizer(posterior.initial_position)
    objective = eqx.filter_jit(_FlatPosteriorObjective(posterior, vectorizer))
    return _Problem(
        name=name,
        posterior=posterior,
        vectorizer=vectorizer,
        objective=objective,
        initial=vectorizer.ravel(initial, name="initial_position"),
        lower=vectorizer.ravel_bound(lower, side="Lower"),
        upper=vectorizer.ravel_bound(upper, side="Upper"),
        target=float(target),
        target_basis=target_basis,
    )


def _mixture_problem() -> _Problem:
    global_mode = jnp.asarray([1.25, -0.75])
    local_mode = jnp.asarray([-1.0, 1.0])
    space = phx.uq.ParameterSpace(
        jnp.zeros((2,)),
        log_prior=lambda value: jnp.asarray(0.0),
    )

    def log_likelihood(value):
        global_well = -jnp.sum((value - global_mode) ** 2) / 0.08
        local_well = -0.3 - jnp.sum((value - local_mode) ** 2) / 0.16
        return jnp.logaddexp(global_well, local_well)

    posterior = phx.uq.PosteriorProblem(space, log_likelihood)
    optimum = float(posterior.negative_log_density(global_mode))
    return _flat_problem(
        "analytic-mixture-map",
        posterior,
        jnp.zeros((2,)),
        -3.0 * jnp.ones((2,)),
        3.0 * jnp.ones((2,)),
        optimum + 0.4,
        "negative log density within 0.4 of the analytic global mode",
    )


def _state_space_problem() -> _Problem:
    state_dimension = 72
    steps = 32
    cases = 2
    times = jnp.linspace(0.05, 1.6, steps)
    channels = jnp.linspace(-0.3, 0.3, state_dimension)
    phase = times[:, None] * jnp.linspace(0.4, 1.2, state_dimension)[None, :]
    true_offset = 0.55
    identity = jnp.eye(state_dimension)

    def make_template(index):
        shifted_phase = phase + 0.17 * index
        first = (
            true_offset
            + channels
            + 0.15 * jnp.sin(shifted_phase)
            + 0.01 * (index + 1) * jnp.cos(0.5 * shifted_phase)
        )
        second = (
            true_offset
            + channels
            + 0.15 * jnp.sin(shifted_phase + 0.09)
            + 0.04 * jnp.cos((index + 1) * shifted_phase)
        )
        observations = phx.stochastic.ObservationSequence(
            times,
            jnp.stack((first, second), axis=0),
            case_axes=("replicate",),
            case_shape=(cases,),
            case_ids=("replicate-a", "replicate-b"),
            sequence_id=f"map-gp-state-space-sequence-{index}",
        )
        prior = phx.stochastic.GaussianStatePrior(
            jnp.zeros((cases, state_dimension)),
            jnp.broadcast_to(identity, (cases, state_dimension, state_dimension)),
            state_shape=(state_dimension,),
            prior_id=f"map-gp-state-space-prior-{index}",
        )
        transition = phx.stochastic.LinearGaussianTransitionKernel(
            (1.0 - 0.002 * index) * identity,
            (0.04 + 0.005 * index) * identity,
            state_shape=(state_dimension,),
            process_id=f"map-gp-state-space-transition-{index}",
        )
        observation = phx.stochastic.LinearGaussianObservationModel(
            identity,
            0.09 * identity,
            state_shape=(state_dimension,),
            observation_shape=(state_dimension,),
            offset=jnp.zeros((state_dimension,)),
            observation_id=f"map-gp-state-space-observation-{index}",
        )
        model = phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id=f"map-gp-state-space-model-{index}",
        )
        return phx.stochastic.StateSpaceProblem(
            model,
            observations,
            initial_time=jnp.zeros((cases,)),
            problem_id=f"map-gp-state-space-problem-{index}",
        )

    templates = tuple(make_template(index) for index in range(6))

    def parameterizer(template):
        def parameterized_problem(parameters):
            offset = jnp.full((state_dimension,), parameters["offset"])
            covariance = jnp.exp(2.0 * parameters["log_scale"]) * identity
            return eqx.tree_at(
                lambda value: (
                    value.model.observation.offset,
                    value.model.observation.covariance,
                ),
                template,
                (offset, covariance),
            )

        return parameterized_problem

    experiments = tuple(
        phx.uq.StateSpaceExperiment(
            parameterizer(template),
            experiment_id=f"map-gp-state-space-experiment-{index}",
            case_axes=("replicate",),
            case_shape=(cases,),
            case_ids=("replicate-a", "replicate-b"),
        )
        for index, template in enumerate(templates)
    )
    initial = {
        "offset": jnp.asarray(-0.6),
        "log_scale": jnp.asarray(-0.1),
    }
    parameter_space = phx.uq.ParameterSpace(
        initial,
        log_prior=lambda parameters: (
            -0.5 * ((parameters["offset"] / 2.0) ** 2 + parameters["log_scale"] ** 2)
        ),
    )
    estimation = phx.uq.StateSpaceEstimation(parameter_space, (experiments[0],))
    overhead_estimation = phx.uq.StateSpaceEstimation(parameter_space, experiments)
    posterior = estimation.posterior
    vectorizer = _PyTreeVectorizer(posterior.initial_position)
    objective = eqx.filter_jit(_FlatPosteriorObjective(posterior, vectorizer))
    overhead_objective = eqx.filter_jit(
        _FlatPosteriorObjective(overhead_estimation.posterior, vectorizer)
    )
    lower_tree = {"offset": -1.5, "log_scale": -2.0}
    upper_tree = {"offset": 1.5, "log_scale": 0.5}
    lower = vectorizer.ravel_bound(lower_tree, side="Lower")
    upper = vectorizer.ravel_bound(upper_tree, side="Upper")
    offset_grid = jnp.linspace(lower[0], upper[0], 9)
    scale_grid = jnp.linspace(lower[1], upper[1], 9)
    first, second = jnp.meshgrid(offset_grid, scale_grid, indexing="ij")
    grid = jnp.stack((first.ravel(), second.ravel()), axis=1)
    grid_objectives = jax.block_until_ready(jax.jit(jax.vmap(objective))(grid))
    oracle = float(jnp.min(grid_objectives))
    return _Problem(
        name="state-space-map",
        posterior=posterior,
        vectorizer=vectorizer,
        objective=objective,
        initial=vectorizer.ravel(initial, name="initial_position"),
        lower=lower,
        upper=upper,
        target=oracle + 0.01 * abs(oracle),
        target_basis=(
            "negative log density within one percent of an independent 9x9 "
            "dense-grid reference"
        ),
        overhead_objective=overhead_objective,
    )


def _problems() -> tuple[_Problem, ...]:
    return (_mixture_problem(), _state_space_problem())


def _unit_to_physical(points, problem):
    return problem.lower + points * (problem.upper - problem.lower)


def _design_trace(problem, *, budget, key, design, method):
    started = perf_counter()
    initial_unit = ((problem.initial - problem.lower) / (problem.upper - problem.lower))[
        None, :
    ]
    sampled = phx.sampling.materialize_design(
        design,
        count=budget - 1,
        dimension=int(problem.initial.size),
        key=key,
    ).astype(initial_unit.dtype)
    unit_points = jnp.concatenate((initial_unit, sampled), axis=0)
    proposal_seconds = perf_counter() - started
    objective_started = perf_counter()
    values = jax.block_until_ready(
        jax.vmap(problem.objective)(_unit_to_physical(unit_points, problem))
    )
    objective_seconds = perf_counter() - objective_started
    masked = jnp.where(jnp.isfinite(values), values, jnp.inf)
    best_index = int(jnp.argmin(masked))
    return _Trace(
        method=method,
        best_vector=np.asarray(_unit_to_physical(unit_points[best_index], problem)),
        best_history=np.asarray(jnp.minimum.accumulate(masked)),
        evaluation_counts=np.arange(1, budget + 1),
        objective_evaluations=budget,
        proposal_seconds=proposal_seconds,
        objective_seconds=objective_seconds,
        metadata={"proposal_evaluations": budget - 1},
    )


def _de_trace(problem, *, budget, key):
    population_size = 8
    generations = budget // population_size - 1
    search = phx.optim.DifferentialEvolutionSearch(
        population_size,
        generations,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        design=phx.sampling.SobolDesign(scrambled=True),
    )
    started = perf_counter()
    result = _bounded_differential_evolution(
        problem.objective,
        problem.initial,
        problem.lower,
        problem.upper,
        search,
        key=key,
    )
    elapsed = perf_counter() - started
    actual_generations = int(result.generations)
    return _Trace(
        method="differential-evolution",
        best_vector=np.asarray(result.best_vector),
        best_history=np.asarray(result.best_objective_history[: actual_generations + 1]),
        evaluation_counts=population_size * np.arange(1, actual_generations + 2),
        objective_evaluations=int(result.objective_evaluations),
        proposal_seconds=None,
        objective_seconds=None,
        metadata={
            "elapsed_seconds": elapsed,
            "population_size": population_size,
            "configured_generations": generations,
            "actual_generations": actual_generations,
            "termination_reason": result.termination_reason,
        },
    )


def _gp_trace(problem, *, budget, key):
    search = GaussianProcessMAPSearch(
        budget,
        surrogate=phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.Matern52Kernel(length_scale=0.25),
            noise_scale=0.0,
            jitter=1e-8,
        ),
        initial_evaluations=8,
        candidate_count=512,
        design=phx.sampling.SobolDesign(scrambled=True),
        improvement_margin=0.01,
        minimum_separation=1e-6,
    )
    evidence = _bounded_gaussian_process_map_search(
        problem.objective,
        problem.initial,
        problem.lower,
        problem.upper,
        search,
        key=key,
    )
    return _Trace(
        method="gaussian-process-map",
        best_vector=np.asarray(evidence.best_vector),
        best_history=np.asarray(evidence.best_objective_history),
        evaluation_counts=np.arange(1, budget + 1),
        objective_evaluations=evidence.objective_evaluations,
        proposal_seconds=evidence.proposal_seconds,
        objective_seconds=evidence.objective_seconds,
        metadata={
            "fallback_count": evidence.fallback_count,
            "surrogate_failure_count": evidence.surrogate_failure_count,
            "proposal_evaluations": budget - 1,
        },
    )


def _local_refine(problem, vector):
    result = phx.uq.find_map(
        problem.posterior,
        problem.vectorizer.unravel(jnp.asarray(vector)),
        max_steps=20,
        gradient_tolerance=1e-4,
        raise_on_failure=False,
    )
    return bool(result.converged), float(result.objective)


def _record(problem, trace, *, budget, seed, elapsed, perform_local):
    hits = np.flatnonzero(trace.best_history <= problem.target)
    evaluations_to_target = (
        int(trace.evaluation_counts[int(hits[0])]) if hits.size else None
    )
    if perform_local:
        local_valid, local_objective = _local_refine(problem, trace.best_vector)
    else:
        local_valid, local_objective = None, None
    proposal_count = int(
        trace.metadata.get(
            "proposal_evaluations",
            max(trace.objective_evaluations - 1, 1),
        )
    )
    proposal_per_evaluation = (
        None
        if trace.proposal_seconds is None
        else trace.proposal_seconds / max(1, proposal_count)
    )
    objective_per_evaluation = (
        None
        if trace.objective_seconds is None
        else trace.objective_seconds / max(1, trace.objective_evaluations)
    )
    return {
        "problem": problem.name,
        "method": trace.method,
        "budget": budget,
        "seed": seed,
        "target": problem.target,
        "target_basis": problem.target_basis,
        "target_hit": bool(hits.size),
        "evaluations_to_target": evaluations_to_target,
        "final_objective": float(trace.best_history[-1]),
        "best_vector": trace.best_vector.tolist(),
        "objective_evaluations": trace.objective_evaluations,
        "evaluation_counts": trace.evaluation_counts.tolist(),
        "best_history": trace.best_history.tolist(),
        "local_valid": local_valid,
        "local_objective": local_objective,
        "proposal_seconds_per_evaluation": proposal_per_evaluation,
        "objective_seconds_per_evaluation": objective_per_evaluation,
        "elapsed_seconds": elapsed,
        **trace.metadata,
    }


def _summaries(records):
    summaries = []
    groups = sorted({(row["problem"], row["method"], row["budget"]) for row in records})
    for problem, method, budget in groups:
        rows = [
            row
            for row in records
            if row["problem"] == problem
            and row["method"] == method
            and row["budget"] == budget
        ]
        censored = [
            row["evaluations_to_target"] if row["target_hit"] else budget + 1
            for row in rows
        ]
        proposal = [
            row["proposal_seconds_per_evaluation"]
            for row in rows
            if row["proposal_seconds_per_evaluation"] is not None
        ]
        objective = [
            row["objective_seconds_per_evaluation"]
            for row in rows
            if row["objective_seconds_per_evaluation"] is not None
        ]
        valid_local = [
            row["local_objective"] for row in rows if row["local_valid"] is True
        ]
        summaries.append(
            {
                "problem": problem,
                "method": method,
                "budget": budget,
                "target": rows[0]["target"],
                "target_hit_rate": float(np.mean([row["target_hit"] for row in rows])),
                "median_censored_evaluations_to_target": float(np.median(censored)),
                "median_final_objective": float(
                    np.median([row["final_objective"] for row in rows])
                ),
                "local_success_rate": (
                    float(
                        np.mean(
                            [
                                row["local_valid"]
                                for row in rows
                                if row["local_valid"] is not None
                            ]
                        )
                    )
                    if any(row["local_valid"] is not None for row in rows)
                    else None
                ),
                "median_local_objective": (
                    float(np.median(valid_local)) if valid_local else None
                ),
                "median_proposal_seconds_per_evaluation": (
                    float(np.median(proposal)) if proposal else None
                ),
                "median_objective_seconds_per_evaluation": (
                    float(np.median(objective)) if objective else None
                ),
                "actual_objective_evaluations": sorted(
                    {int(row["objective_evaluations"]) for row in rows}
                ),
            }
        )
    return summaries


def _attach_local_results(records, problems, budget):
    preliminary = _summaries(records)
    problem_by_name = {problem.name: problem for problem in problems}
    for problem_name, problem in problem_by_name.items():
        relevant = [
            row
            for row in preliminary
            if row["problem"] == problem_name and row["budget"] == budget
        ]
        baseline = min(
            (row for row in relevant if row["method"] != "gaussian-process-map"),
            key=lambda row: (
                -float(row["target_hit_rate"]),
                float(row["median_censored_evaluations_to_target"]),
                float(row["median_final_objective"]),
            ),
        )
        retained_methods = {"gaussian-process-map", baseline["method"]}
        for method in retained_methods:
            selected = sorted(
                (
                    row
                    for row in records
                    if row["problem"] == problem_name and row["method"] == method
                ),
                key=lambda row: int(row["seed"]),
            )[:5]
            for row in selected:
                valid, objective = _local_refine(problem, row["best_vector"])
                row["local_valid"] = valid
                row["local_objective"] = objective


def _measure_overhead(problem):
    if problem.overhead_objective is None:
        raise ValueError("State-space overhead workload is unavailable.")
    jax.block_until_ready(problem.overhead_objective(problem.initial))
    samples = []
    for _ in range(5):
        started = perf_counter()
        jax.block_until_ready(problem.overhead_objective(problem.initial))
        samples.append(perf_counter() - started)
    return {
        "workload": (
            "six genuinely distinct fixed state-space experiments, two replicates each"
        ),
        "repeats": len(samples),
        "steady_seconds": samples,
        "median_steady_seconds": float(np.median(samples)),
    }


def _bootstrap(records, *, problem, budget, primary, baseline, seed):
    relevant = [
        row for row in records if row["problem"] == problem and row["budget"] == budget
    ]
    primary_rows = {int(row["seed"]): row for row in relevant if row["method"] == primary}
    baseline_rows = {
        int(row["seed"]): row for row in relevant if row["method"] == baseline
    }
    seeds = sorted(set(primary_rows) & set(baseline_rows))
    differences = np.asarray(
        [
            (
                primary_rows[item]["evaluations_to_target"]
                if primary_rows[item]["target_hit"]
                else budget + 1
            )
            - (
                baseline_rows[item]["evaluations_to_target"]
                if baseline_rows[item]["target_hit"]
                else budget + 1
            )
            for item in seeds
        ],
        dtype=float,
    )
    generator = np.random.default_rng(seed)
    samples = generator.choice(
        differences,
        size=(10_000, differences.size),
        replace=True,
    )
    medians = np.median(samples, axis=1)
    lower, upper = np.quantile(medians, (0.025, 0.975))
    return {
        "paired_seeds": seeds,
        "paired_median_difference": float(np.median(differences)),
        "bootstrap_samples": 10_000,
        "confidence_interval": [float(lower), float(upper)],
        "excludes_zero_in_favor_of_gp": bool(upper < 0.0),
        "excludes_zero_against_gp": bool(lower > 0.0),
    }


def _gate(records, summaries, budget, overhead_benchmark):
    primary_method = "gaussian-process-map"
    checks = []
    for problem_index, problem in enumerate(("analytic-mixture-map", "state-space-map")):
        relevant = [
            row
            for row in summaries
            if row["problem"] == problem and row["budget"] == budget
        ]
        primary = next(row for row in relevant if row["method"] == primary_method)
        baselines = [row for row in relevant if row["method"] != primary_method]
        baseline = min(
            baselines,
            key=lambda row: (
                -float(row["target_hit_rate"]),
                float(row["median_censored_evaluations_to_target"]),
                float(row["median_final_objective"]),
            ),
        )
        bootstrap = _bootstrap(
            records,
            problem=problem,
            budget=budget,
            primary=primary_method,
            baseline=baseline["method"],
            seed=20260824 + problem_index,
        )
        local_objective_tolerance = 1e-8 * max(
            1.0,
            abs(float(baseline["median_local_objective"])),
        )
        passed = (
            primary["target_hit_rate"] >= baseline["target_hit_rate"]
            and primary["median_censored_evaluations_to_target"]
            <= 0.8 * baseline["median_censored_evaluations_to_target"]
            and primary["median_final_objective"] <= baseline["median_final_objective"]
            and primary["local_success_rate"] >= baseline["local_success_rate"]
            and primary["median_local_objective"]
            <= baseline["median_local_objective"] + local_objective_tolerance
        )
        checks.append(
            {
                "problem": problem,
                "primary": primary,
                "baseline": baseline,
                "paired_bootstrap": bootstrap,
                "local_objective_tolerance": local_objective_tolerance,
                "passed": bool(passed),
            }
        )
    state_space = next(
        row
        for row in summaries
        if row["problem"] == "state-space-map"
        and row["method"] == primary_method
        and row["budget"] == budget
    )
    overhead_ratio = (
        state_space["median_proposal_seconds_per_evaluation"]
        / overhead_benchmark["median_steady_seconds"]
    )
    confidence_positive = any(
        check["paired_bootstrap"]["excludes_zero_in_favor_of_gp"] for check in checks
    )
    confidence_not_against = all(
        not check["paired_bootstrap"]["excludes_zero_against_gp"] for check in checks
    )
    return {
        "checks": checks,
        "paired_confidence_positive": confidence_positive,
        "paired_confidence_not_against": confidence_not_against,
        "state_space_overhead": {
            "objective_workload": overhead_benchmark,
            "proposal_seconds": state_space["median_proposal_seconds_per_evaluation"],
            "ratio": overhead_ratio,
            "maximum_ratio": 0.1,
            "passed": overhead_ratio <= 0.1,
        },
        "passed": bool(
            all(check["passed"] for check in checks)
            and confidence_positive
            and confidence_not_against
            and overhead_ratio <= 0.1
        ),
    }


def _source_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "git_revision": revision,
        "working_tree_clean": not bool(diff),
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def run_benchmark(arguments):
    if arguments.seeds < 1:
        raise ValueError("--seeds must be positive.")
    if arguments.phase == "admission" and (
        arguments.seed_start != 800 or arguments.seeds != 20
    ):
        raise ValueError("Admission is frozen to seeds 800 through 819.")
    budget = 32
    records = []
    problems = _problems()
    for problem in problems:
        for seed_offset in range(arguments.seeds):
            seed = arguments.seed_start + seed_offset
            key = jr.fold_in(jr.key(20260824), seed)
            methods = (
                lambda: _design_trace(
                    problem,
                    budget=budget,
                    key=key,
                    design=phx.sampling.IIDDesign(),
                    method="iid",
                ),
                lambda: _design_trace(
                    problem,
                    budget=budget,
                    key=key,
                    design=phx.sampling.SobolDesign(scrambled=True),
                    method="sobol",
                ),
                lambda: _de_trace(problem, budget=budget, key=key),
                lambda: _gp_trace(problem, budget=budget, key=key),
            )
            for method in methods:
                started = perf_counter()
                trace = method()
                elapsed = perf_counter() - started
                records.append(
                    _record(
                        problem,
                        trace,
                        budget=budget,
                        seed=seed,
                        elapsed=elapsed,
                        perform_local=False,
                    )
                )
    if not arguments.skip_local:
        _attach_local_results(records, problems, budget)
    summaries = _summaries(records)
    state_space_problem = next(
        problem for problem in problems if problem.name == "state-space-map"
    )
    overhead_benchmark = _measure_overhead(state_space_problem)
    gate = (
        _gate(records, summaries, budget, overhead_benchmark)
        if arguments.phase == "admission"
        else None
    )
    return {
        "phase": arguments.phase,
        "passed": None if gate is None else gate["passed"],
        "configuration": {
            "budget": budget,
            "candidate_count": 512,
            "initial_evaluations": 8,
            "seed_start": arguments.seed_start,
            "seed_stop_exclusive": arguments.seed_start + arguments.seeds,
            "policy": "sequential expected improvement",
            "kernel": "Matern52(length_scale=0.25) on unit positions",
            "noise_scale_units": "raw negative-log-density units",
            "noise_standardization": "noise_scale / objective_scale",
            "jitter_units": "standardized covariance units",
            "state_space_search_experiments": 1,
            "state_space_overhead_experiments": 6,
            "state_space_experiments_are_distinct": True,
            "local_refinement_seed_count_per_selected_method": (
                0 if arguments.skip_local else 5
            ),
            "de_budget_contract": {
                "population_size": 8,
                "maximum_generations": 3,
                "configured_evaluations": 32,
                "comparison": "actual evaluation counts; DE may terminate early",
            },
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "device": jax.devices()[0].device_kind,
            "x64": bool(jax.config.x64_enabled),
        },
        "source_provenance": _source_provenance(),
        "overhead_benchmark": overhead_benchmark,
        "gate": gate,
        "summaries": summaries,
        "records": records,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "admission"), default="pilot")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--skip-local", action="store_true")
    return parser


def main():
    arguments = _parser().parse_args()
    report = run_benchmark(arguments)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
