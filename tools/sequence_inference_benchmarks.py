#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic variational and particle sequence-inference benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from benchmarks._runtime import measure_synchronized


def _state_problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.linspace(0.25, 2.0, 8),
        jnp.sin(jnp.linspace(0.25, 2.0, 8))[:, None],
        case_ids=("only",),
        sequence_id="sequence-benchmark",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[0.9]]),
        jnp.asarray([[0.15]]),
        state_shape=(1,),
        process_id="latent",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="sequence-benchmark-model",
        ),
        observations,
        initial_time=0.0,
        problem_id="sequence-benchmark-problem",
    )


def _timed(function):
    return measure_synchronized(function)


def run_sequence_inference_benchmarks(*, quick: bool = False):
    optimization_steps = 20 if quick else 200
    posterior = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.asarray(0.0),
            priors=phx.uq.Normal(0.0, 1.0),
        ),
        lambda value: -0.5 * jnp.square((value - 1.2) / 0.5),
    )
    variational, variational_seconds = _timed(
        lambda: phx.uq.fit_variational(
            posterior,
            key=jr.key(1),
            config=phx.uq.VariationalConfig(
                num_steps=optimization_steps,
                samples_per_step=8,
                learning_rate=0.02,
                record_every=max(1, optimization_steps // 4),
            ),
            num_samples=64,
        )
    )
    problem = _state_problem()
    state_variational, state_variational_seconds = _timed(
        lambda: phx.uq.fit_state_space_variational(
            problem,
            key=jr.key(2),
            config=phx.uq.StateSpaceVariationalConfig(
                optimization=phx.uq.VariationalConfig(
                    num_steps=optimization_steps,
                    samples_per_step=8,
                    learning_rate=0.02,
                    record_every=max(1, optimization_steps // 4),
                )
            ),
            num_samples=32,
        )
    )
    buffered, buffered_seconds = _timed(
        lambda: phx.uq.fit_buffered_state_space_variational(
            problem,
            key=jr.key(3),
            config=phx.uq.BufferedStateSpaceVariationalConfig(
                target_length=4,
                left_buffer=2,
                right_buffer=2,
                hidden_size=8,
                optimization=phx.uq.VariationalConfig(
                    num_steps=optimization_steps,
                    samples_per_step=4,
                    learning_rate=0.01,
                    record_every=max(1, optimization_steps // 4),
                ),
            ),
            num_samples=16,
        )
    )
    particle_cases = []
    for count in (8, 16) if quick else (8, 16, 32, 64):
        filtered, filter_seconds = _timed(
            lambda count=count: phx.uq.bootstrap_particle_filter(
                jr.key(100 + count),
                problem,
                num_particles=count,
                resampling_policy="always",
            )
        )
        score, score_seconds = _timed(
            lambda filtered=filtered: phx.uq.particle_genealogical_score(filtered)
        )
        particle_cases.append(
            {
                "num_particles": count,
                "filter_seconds": filter_seconds,
                "score_seconds": score_seconds,
                "parameter_size": score.parameter_size,
                "score_norm": float(jnp.linalg.norm(score.flat_score)),
                "valid": bool(score.valid),
            }
        )
    return {
        "schema": "phydrax-sequence-inference-benchmark-v1",
        "backend": jax.default_backend(),
        "quick": bool(quick),
        "variational": {
            "seconds": variational_seconds,
            "final_elbo": float(variational.diagnostics.elbo[-1]),
            "mean": float(jnp.mean(variational.samples)),
            "scale": float(jnp.std(variational.samples)),
        },
        "state_space_variational": {
            "seconds": state_variational_seconds,
            "final_elbo": float(state_variational.diagnostics.elbo[-1]),
            "finite": bool(jnp.all(jnp.isfinite(state_variational.log_model))),
        },
        "buffered_state_space_variational": {
            "seconds": buffered_seconds,
            "final_elbo": float(buffered.diagnostics.elbo[-1]),
            "finite": bool(jnp.all(jnp.isfinite(buffered.log_model))),
            "inclusion_probability": [
                float(value) for value in buffered.window_plan.inclusion_probability
            ],
        },
        "particle_genealogical": particle_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    report = run_sequence_inference_benchmarks(quick=arguments.quick)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n")
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
