#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark prepared-noise full-state policy evaluation and holdout evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_repeated,
)
from phydrax._fingerprint import array_tree_fingerprint


def _transition(context, state, action, noise, args):
    return state + context.duration * (args["decay"] * state + action) + noise


def _stage_cost(context, state, action, args):
    del args
    return context.duration * jnp.mean(state * state + 0.2 * action * action)


def _terminal_cost(time, state, args):
    del time, args
    return jnp.mean(state * state)


def _full_state_policy(context, state, args):
    del context
    # Deliberately no noise argument and no random key: this is causal state feedback.
    return args["gain"] * state


def _noise(
    role: str,
    path_count: int,
    horizon: int,
    case_count: int,
    cluster_count: int,
    /,
):
    path = jnp.arange(path_count, dtype=jnp.float32)[:, None, None]
    step = jnp.arange(horizon, dtype=jnp.float32)[None, :, None]
    coordinate = jnp.arange(case_count, dtype=jnp.float32)[None, None, :]
    role_offset = 1.0 if role == "training" else 101.0
    increments = (
        0.02
        * jnp.cos((path + role_offset) * (step + 1.0) * (coordinate + 1.0) * 0.017)
        / jnp.sqrt(jnp.asarray(horizon, dtype=jnp.float32))
    )
    return phx.control.stochastic.PreparedControlledNoise(
        increments,
        valid=jnp.ones((path_count,), dtype=bool),
        realization_ids=tuple(
            f"benchmark-feedback:{role}:path-{index}" for index in range(path_count)
        ),
        coupling_id=f"benchmark-feedback:{role}:independent-bundle",
        independence_labels=(jnp.arange(path_count, dtype=jnp.int32) % cluster_count),
        noise_shape=(case_count,),
    )


def _problem(horizon: int, case_count: int, /):
    time_grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 1.0, horizon + 1, dtype=jnp.float32),
        time_id=f"benchmark-feedback:T{horizon}:coordinates{case_count}",
    )
    return phx.control.stochastic.ControlledTransitionProblem(
        _transition,
        time_grid,
        jnp.linspace(0.1, 0.3, case_count, dtype=jnp.float32),
        state_shape=(case_count,),
        action_shape=(case_count,),
        noise_shape=(case_count,),
        stage_cost=_stage_cost,
        terminal_cost=_terminal_cost,
        args={"decay": jnp.asarray(-0.15), "gain": jnp.asarray(-0.45)},
        problem_id=f"benchmark-feedback:T{horizon}:coordinates{case_count}",
    )


def _evaluate(problem, training_noise, holdout_noise):
    training = phx.control.stochastic.evaluate_feedback_policy(
        problem,
        _full_state_policy,
        training_noise,
        policy_id="benchmark-full-state-linear-feedback",
        method="asymptotic-normal",
        sample_role="training",
    )
    holdout = phx.control.stochastic.evaluate_feedback_policy(
        problem,
        _full_state_policy,
        holdout_noise,
        policy_id="benchmark-full-state-linear-feedback",
        method="asymptotic-normal",
        sample_role="holdout",
    )
    return training, holdout


def _certificates(training, holdout, training_noise, holdout_noise) -> dict[str, Any]:
    disjoint_ids = set(training_noise.realization_ids).isdisjoint(
        holdout_noise.realization_ids
    )
    distinct_coupling = training_noise.coupling_id != holdout_noise.coupling_id
    return {
        "claim": "fixed-policy-prepared-noise-holdout-expectation-evidence",
        "policy_information": "current full state and step context only; no noise and no random key",
        "all_training_paths_valid": bool(jnp.all(training.paths.valid)),
        "all_holdout_paths_valid": bool(jnp.all(holdout.paths.valid)),
        "training_status": int(training.status),
        "holdout_status": int(holdout.status),
        "training_holdout_realization_ids_disjoint": disjoint_ids,
        "training_holdout_coupling_ids_distinct": distinct_coupling,
        "training_coverage": training.evidence.coverage,
        "training_has_coverage_claim": training.evidence.has_coverage_claim,
        "holdout_coverage": holdout.evidence.coverage,
        "holdout_has_coverage_claim": holdout.evidence.has_coverage_claim,
        "holdout_valid_path_count": int(holdout.evidence.valid_path_count),
        "holdout_independent_cluster_count": int(
            holdout.evidence.independent_cluster_count
        ),
        "holdout_estimate": float(holdout.evidence.estimate),
        "holdout_standard_error": float(holdout.evidence.standard_error),
        "holdout_interval": [
            float(holdout.evidence.lower),
            float(holdout.evidence.upper),
        ],
        "holdout_confidence": holdout.evidence.confidence,
        "coverage_assumptions": list(holdout.evidence.coverage_assumptions),
        "output_fingerprint": array_tree_fingerprint((training, holdout)),
        "not_claimed": [
            "finite-sample distribution-free coverage",
            "policy optimality",
            "population optimality",
        ],
    }


def _case(
    name: str,
    path_count: int,
    horizon: int,
    case_count: int,
    cluster_count: int,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if cluster_count < 2 or cluster_count > path_count:
        raise ValueError("cluster_count must be between two and path_count")
    problem = _problem(horizon, case_count)
    training_noise = _noise("training", path_count, horizon, case_count, cluster_count)
    holdout_noise = _noise("holdout", path_count, horizon, case_count, cluster_count)
    fingerprint_inputs = (
        problem.time_grid.times,
        problem.initial_state,
        problem.args,
        training_noise.increments,
        training_noise.valid,
        training_noise.independence_labels,
        holdout_noise.increments,
        holdout_noise.valid,
        holdout_noise.independence_labels,
    )
    result, execution = measure_repeated(
        lambda: _evaluate(problem, training_noise, holdout_noise),
        warmup=warmup,
        repeats=repeats,
    )
    training, holdout = result
    unavailable = compiler_evidence(
        None,
        None,
        source="host-orchestrated-feedback-evaluation",
        unavailable_reason=(
            "coverage and sample-role decisions are host logic; this benchmark "
            "does not represent them as one lowered executable"
        ),
    )
    return {
        "name": name,
        "dimensions": {
            "paths_per_bundle": path_count,
            "horizon": horizon,
            "case_count": case_count,
            "case_scope": "independent scalar state coordinates aggregated into each path return",
            "independent_clusters_per_bundle": cluster_count,
        },
        "dtype": str(training_noise.increments.dtype),
        "input_fingerprint": array_tree_fingerprint(fingerprint_inputs),
        "lower": {
            "seconds": None,
            "scope": "not applicable to host-orchestrated coverage evaluation",
        },
        "compile": {
            "seconds": None,
            "scope": "not applicable to host-orchestrated coverage evaluation",
        },
        "run": {
            **execution.to_milliseconds_dict(),
            "scope": "synchronized training-role and holdout-role evaluations",
        },
        "memory": {
            "logical_input_bytes": logical_array_bytes(fingerprint_inputs),
            "logical_output_bytes": logical_array_bytes(result),
            "compiler_argument_bytes": unavailable.argument_bytes,
            "compiler_output_bytes": unavailable.output_bytes,
            "compiler_temporary_bytes": unavailable.temporary_bytes,
            "compiler_generated_code_bytes": unavailable.generated_code_bytes,
            "source": unavailable.source,
            "unavailable_reason": unavailable.unavailable_reason,
        },
        "work": {
            "compiler_flops": unavailable.flops,
            "compiler_bytes_accessed": unavailable.bytes_accessed,
            "prepared_noise_values": 2 * path_count * horizon * case_count,
            "policy_decisions": 2 * path_count * horizon,
            "state_coordinate_updates": 2 * path_count * horizon * case_count,
            "independent_cluster_summaries": 2 * cluster_count,
        },
        "certificate": _certificates(training, holdout, training_noise, holdout_noise),
    }


def _specifications():
    return (
        ("baseline", 64, 16, 4, 64),
        ("paths-16", 16, 16, 4, 16),
        ("paths-256", 256, 16, 4, 256),
        ("horizon-4", 64, 4, 4, 64),
        ("horizon-64", 64, 64, 4, 64),
        ("cases-1", 64, 16, 1, 64),
        ("cases-32", 64, 16, 32, 64),
        ("clusters-4", 64, 16, 4, 4),
        ("clusters-16", 64, 16, 4, 16),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    cases = [
        _case(
            name,
            path_count,
            horizon,
            case_count,
            cluster_count,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for name, path_count, horizon, case_count, cluster_count in _specifications()
    ]
    payload = {
        "benchmark": "stochastic-feedback-evaluation",
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_valid": all(
            case["certificate"]["all_training_paths_valid"]
            and case["certificate"]["all_holdout_paths_valid"]
            and case["certificate"]["training_holdout_realization_ids_disjoint"]
            and case["certificate"]["training_holdout_coupling_ids_distinct"]
            and not case["certificate"]["training_has_coverage_claim"]
            and case["certificate"]["holdout_has_coverage_claim"]
            for case in cases
        ),
    }
    if arguments.output is None:
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    else:
        write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
