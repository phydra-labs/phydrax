#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

import phydrax as phx


def gaussian_benchmark(*, dimension: int, num_live: int):
    prior_scale = 2.0
    observation_scale = 0.5
    space = phx.uq.ParameterSpace(
        jnp.zeros((dimension,)),
        priors=phx.uq.Normal(0.0, prior_scale),
    )

    def log_likelihood(value):
        standardized = value / observation_scale
        return jnp.sum(
            -0.5 * standardized**2
            - jnp.log(observation_scale)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )

    problem = phx.uq.PosteriorProblem(space, log_likelihood)
    plan = phx.uq.NestedSamplingPlan(
        phx.uq.NestedSamplingCapacity(
            max_live=num_live,
            max_dead_points=20_000,
            max_likelihood_evaluations=1_000_000,
            max_dynamic_batches=1,
            max_clusters=4,
            max_phantoms=num_live,
        ),
        phx.uq.NestedPriorPlan(continuous_paths=("<root>",)),
        phx.uq.NestedProposalPlan(
            "hit-and-run",
            ellipsoid=True,
            maximum_attempts=max(5, 2 * dimension),
        ),
        initial_live=num_live,
    )
    result = phx.uq.sample_nested(
        problem,
        key=jr.key(10_000 + dimension),
        plan=plan,
        remaining_evidence_tolerance=0.05,
    )
    evidence_truth = dimension * jsp.stats.norm.logpdf(
        0.0,
        loc=0.0,
        scale=jnp.sqrt(prior_scale**2 + observation_scale**2),
    )
    return {
        "case": "gaussian",
        "dimension": dimension,
        "num_live": num_live,
        "num_inner_steps": result.num_inner_steps,
        "status": phx.uq.nested_sampling_status_name(int(result.status)),
        "valid": bool(result.valid),
        "duration_seconds": result.duration_seconds,
        "num_dead": result.num_dead,
        "num_likelihood_evaluations": result.num_likelihood_evaluations,
        "log_evidence": float(result.log_evidence),
        "log_evidence_truth": float(evidence_truth),
        "log_evidence_error": float(result.log_evidence - evidence_truth),
        "log_evidence_shrinkage_std": float(result.log_evidence_shrinkage_std),
        "posterior_effective_sample_size": float(result.posterior_effective_sample_size),
        "insertion_rank_pvalue": float(result.diagnostics.insertion_rank_pvalue),
        "zero_movement_fraction": float(result.diagnostics.zero_movement_fraction),
        "sample_memory_bytes": result.sample_memory_bytes,
    }


def bimodal_benchmark(*, num_live: int, method: str):
    prior_scale = 3.0
    likelihood_scale = 0.3
    mode = 2.0
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        priors=phx.uq.Normal(0.0, prior_scale),
    )

    def log_likelihood(value):
        components = jnp.stack(
            (
                jsp.stats.norm.logpdf(value, loc=-mode, scale=likelihood_scale),
                jsp.stats.norm.logpdf(value, loc=mode, scale=likelihood_scale),
            )
        )
        return jsp.special.logsumexp(components) - jnp.log(2.0)

    problem = phx.uq.PosteriorProblem(space, log_likelihood)
    plan = phx.uq.NestedSamplingPlan(
        phx.uq.NestedSamplingCapacity(
            max_live=num_live,
            max_dead_points=20_000,
            max_likelihood_evaluations=1_000_000,
            max_dynamic_batches=1,
            max_clusters=4,
            max_phantoms=num_live,
        ),
        phx.uq.NestedPriorPlan(continuous_paths=("<root>",)),
        phx.uq.NestedProposalPlan(
            method,
            ellipsoid=True,
            phantom_recycling=True,
            maximum_attempts=8,
        ),
        initial_live=num_live,
    )
    result = phx.uq.sample_nested(
        problem,
        key=jr.key(20_000 + num_live),
        plan=plan,
        remaining_evidence_tolerance=0.05,
    )
    weights = jnp.exp(result.posterior_log_weights)
    positive_mass = jnp.sum(jnp.where(result.samples > 0.0, weights, 0.0))
    evidence_truth = jsp.stats.norm.logpdf(
        mode,
        loc=0.0,
        scale=jnp.sqrt(prior_scale**2 + likelihood_scale**2),
    )
    return {
        "case": "bimodal",
        "method": method,
        "num_live": num_live,
        "status": phx.uq.nested_sampling_status_name(int(result.status)),
        "valid": bool(result.valid),
        "duration_seconds": result.duration_seconds,
        "num_dead": result.num_dead,
        "num_likelihood_evaluations": result.num_likelihood_evaluations,
        "log_evidence": float(result.log_evidence),
        "log_evidence_truth": float(evidence_truth),
        "log_evidence_error": float(result.log_evidence - evidence_truth),
        "positive_mode_mass": float(positive_mass),
        "posterior_effective_sample_size": float(result.posterior_effective_sample_size),
        "insertion_rank_pvalue": float(result.diagnostics.insertion_rank_pvalue),
        "effective_lineage_count": float(result.diagnostics.effective_lineage_count),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("all", "gaussian", "bimodal"),
        default="all",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = []
    if args.case in ("all", "gaussian"):
        configurations = ((2, 40),) if args.quick else ((2, 100), (10, 200), (30, 500))
        records.extend(
            gaussian_benchmark(
                dimension=dimension,
                num_live=num_live,
            )
            for dimension, num_live in configurations
        )
    if args.case in ("all", "bimodal"):
        configurations = (
            ((80, "hit-and-run"),)
            if args.quick
            else (
                (250, "hit-and-run"),
                (250, "slice-within-gibbs"),
                (500, "hit-and-run"),
            )
        )
        records.extend(
            bimodal_benchmark(num_live=num_live, method=method)
            for num_live, method in configurations
        )

    payload = json.dumps(records, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
