#!/usr/bin/env python3
"""Deterministic native causal-inference software benchmarks."""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _assumptions() -> phx.causal.AssumptionLedger:
    return phx.causal.AssumptionLedger(
        phx.causal.CausalAssumption(
            kind=kind,
            disposition=phx.causal.AssumptionDisposition.DECLARED,
            statement=f"Synthetic benchmark declaration for {kind.value}.",
        )
        for kind in (
            phx.causal.AssumptionKind.CONSISTENCY,
            phx.causal.AssumptionKind.POSITIVITY,
            phx.causal.AssumptionKind.NO_INTERFERENCE,
            phx.causal.AssumptionKind.CAUSAL_MARKOV,
        )
    )


def _estimation(samples: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=samples)
    propensity = np.where(z == 0, 0.2, 0.8)
    treatment = rng.binomial(1, propensity)
    outcome = 1.5 * treatment + 0.5 * z + rng.normal(scale=0.2, size=samples)
    schema = phx.causal.CausalSchema(
        (
            phx.causal.CausalVariable(name="z", scale="binary"),
            phx.causal.CausalVariable(name="t", scale="binary"),
            phx.causal.CausalVariable(name="y"),
        )
    )
    dataset = phx.causal.CausalDataset(
        schema=schema,
        values=(jnp.asarray(z), jnp.asarray(treatment), jnp.asarray(outcome)),
    )
    design = phx.causal.CausalStudyDesign(
        schema=schema,
        assignment_kind="observational",
        assignment_variable="t",
        exposure_variable="t",
        source_population_id="benchmark-population",
        assumptions=_assumptions(),
        no_interference=True,
        known_assignment_probability=jnp.asarray(propensity),
    )
    query = phx.causal.CausalQuery(
        schema=schema,
        outcome_variable="y",
        contrast=phx.causal.TreatmentContrast(
            active=phx.causal.TreatmentRegime(exposure_variable="t", value=1),
            reference=phx.causal.TreatmentRegime(exposure_variable="t", value=0),
        ),
        population=phx.causal.TargetPopulation(
            source_population_id="benchmark-population"
        ),
    )
    problem = phx.causal.CausalProblem(dataset=dataset, design=design, query=query)
    graph = phx.causal.CausalDAG(
        schema=schema,
        directed_edges=(("z", "t"), ("z", "y"), ("t", "y")),
    )
    certificate = phx.causal.issue_identification_certificate(
        phx.causal.identify_causal_effect(problem, graph, adjustment_set=("z",)),
        problem,
    )
    plan = phx.causal.NuisancePlan(
        outcome_recipe=phx.ml.linear.OLSRecipe(),
        split_plan=phx.ml.model_selection.KFoldPlan(5),
    )
    started = time.perf_counter()
    nuisance = phx.causal.fit_cross_fitted_nuisance(
        problem,
        certificate,
        plan,
        key=jax.random.key(seed),
    )
    overlap = phx.causal.evaluate_overlap(
        problem,
        nuisance,
        phx.causal.OverlapPolicy(minimum_effective_sample_size=10),
    )
    estimate = phx.causal.estimate_aipw(problem, certificate, nuisance, overlap)
    estimate.effect.block_until_ready()
    elapsed = time.perf_counter() - started
    return {
        "workflow": "estimation",
        "samples": samples,
        "seconds": elapsed,
        "effect": float(estimate.effect),
        "absolute_error": abs(float(estimate.effect) - 1.5),
        "standard_error": float(estimate.standard_error),
        "status": int(estimate.status),
        "certificate_id": certificate.certificate_id,
        "result_id": estimate.result_id,
        "active_effective_sample_size": float(overlap.active_effective_sample_size),
        "reference_effective_sample_size": float(overlap.reference_effective_sample_size),
    }


def _scm(samples: int, seed: int) -> dict[str, object]:
    schema = phx.causal.CausalSchema(
        (phx.causal.CausalVariable(name="x"), phx.causal.CausalVariable(name="y"))
    )
    graph = phx.causal.CausalDAG(schema=schema, directed_edges=(("x", "y"),))
    root = phx.causal.InvertibleNoiseMechanism(
        output="x",
        parents=(),
        noise_sampler=lambda key, n: jax.random.normal(key, (n,)),
        forward=lambda parents, noise: noise,
        inverse=lambda parents, value: value,
        semantic_ids=(
            "benchmark-root-noise",
            "benchmark-root-forward",
            "benchmark-root-inverse",
        ),
        numeric_ids=(
            "benchmark-root-noise-r0",
            "benchmark-root-forward-r0",
            "benchmark-root-inverse-r0",
        ),
    )
    child = phx.causal.InvertibleNoiseMechanism(
        output="y",
        parents=("x",),
        noise_sampler=lambda key, n: jax.random.normal(key, (n,)),
        forward=lambda parents, noise: 2.0 * parents[0] + noise,
        inverse=lambda parents, value: value - 2.0 * parents[0],
        semantic_ids=(
            "benchmark-child-noise",
            "benchmark-child-forward",
            "benchmark-child-inverse",
        ),
        numeric_ids=(
            "benchmark-child-noise-r0",
            "benchmark-child-forward-r0",
            "benchmark-child-inverse-r0",
        ),
    )
    plan = phx.causal.SCMPlan(
        graph=graph, mechanisms=(root, child), exogenous_independence=True
    )
    regime = phx.causal.build_intervention_regime(
        plan,
        (phx.causal.PerfectIntervention(variable="x", value=1.0),),
    )
    started = time.perf_counter()
    result = phx.causal.sample_scm(regime, jax.random.key(seed), samples)
    result.value("y").block_until_ready()
    elapsed = time.perf_counter() - started
    return {
        "workflow": "scm",
        "samples": samples,
        "seconds": elapsed,
        "interventional_mean": float(jnp.mean(result.value("y"))),
        "status": int(result.status),
        "scm_plan_id": plan.plan_id,
        "regime_id": regime.regime_id,
    }


def _discovery(samples: int, variables: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    columns = [rng.normal(size=samples)]
    for _ in range(1, variables):
        columns.append(columns[-1] + rng.normal(scale=0.8, size=samples))
    schema = phx.causal.CausalSchema(
        tuple(phx.causal.CausalVariable(name=f"x{index}") for index in range(variables))
    )
    dataset = phx.causal.CausalDataset(
        schema=schema,
        values=tuple(jnp.asarray(column) for column in columns),
    )
    plan = phx.causal.PCStablePlan(
        ci_test=phx.causal.FisherZTest(alpha=0.001),
        knowledge=phx.causal.DiscoveryBackgroundKnowledge(schema=schema),
        resources=phx.causal.DiscoveryResourcePolicy(
            maximum_ci_tests=50_000,
            maximum_conditioning_depth=2,
            maximum_extensions=65_536,
        ),
    )
    started = time.perf_counter()
    result = phx.causal.discover_pc_stable(dataset, plan)
    elapsed = time.perf_counter() - started
    return {
        "workflow": "discovery",
        "samples": samples,
        "variables": variables,
        "seconds": elapsed,
        "status": result.status.value,
        "graph_kind": type(result.graph).__name__,
        "ci_tests": result.ci_tests,
        "graph_id": result.graph.graph_id,
        "result_id": result.result_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("estimation", "scm", "discovery", "all"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    samples = 120 if args.smoke else 2000
    variables = 4 if args.smoke else 8
    records = []
    if args.mode in {"estimation", "all"}:
        records.append(_estimation(samples, args.seed))
    if args.mode in {"scm", "all"}:
        records.append(_scm(samples, args.seed))
    if args.mode in {"discovery", "all"}:
        records.append(_discovery(samples, variables, args.seed))
    print(
        json.dumps(
            {
                "benchmark": "phydrax-causal-native",
                "scientific_claim": "none; deterministic synthetic software benchmark",
                "smoke": bool(args.smoke),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
