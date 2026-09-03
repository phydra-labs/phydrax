#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark frozen-path finite-dimensional stochastic policy-game SAA solves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.nonlinear import NonlinearTermination


def _quadratic_path_costs(parameters, noise, args):
    """Return every complete frozen-path cost before SAA aggregation."""
    signal = jnp.mean(noise.increments, axis=tuple(range(1, noise.increments.ndim)))
    signal_shape = (1,) * (parameters.ndim - 1) + (noise.num_paths, 1)
    target = args["offset"][..., None, :] + args["scale"][..., None, :] * signal.reshape(
        signal_shape
    )
    difference = parameters[..., None, :] - target
    return phx.ein.contract(
        "...rk,pk->...rp",
        0.5 * difference * difference,
        args["ownership"],
    )


def _noise(
    role: str,
    path_count: int,
    horizon: int,
    noise_size: int,
    cluster_count: int,
    /,
):
    path = jnp.arange(path_count, dtype=jnp.float32)[:, None, None]
    step = jnp.arange(horizon, dtype=jnp.float32)[None, :, None]
    component = jnp.arange(noise_size, dtype=jnp.float32)[None, None, :]
    role_offset = 3.0 if role == "training" else 211.0
    increments = 0.3 * jnp.sin(
        0.031 * (path + role_offset) * (step + 1.0) * (component + 1.0)
    )
    return phx.control.stochastic.PreparedControlledNoise(
        increments,
        valid=jnp.ones((path_count,), dtype=bool),
        realization_ids=tuple(
            f"benchmark-policy-game:{role}:path-{index}" for index in range(path_count)
        ),
        coupling_id=f"benchmark-policy-game:{role}:independent-bundle",
        independence_labels=(jnp.arange(path_count, dtype=jnp.int32) % cluster_count),
        noise_shape=(noise_size,),
    )


def _prepared_problem(
    path_count: int,
    horizon: int,
    noise_size: int,
    control_sizes: tuple[int, ...],
    case_shape: tuple[int, ...],
    cluster_count: int,
    /,
):
    partition = phx.control.games.PlayerControlPartition(
        tuple(f"player-{index}" for index in range(len(control_sizes))),
        control_sizes,
    )
    parameter_size = partition.joint_control_size
    parameter = jnp.arange(parameter_size, dtype=jnp.float32)
    base_offset = 0.08 * jnp.sin(parameter + 1.0)
    base_scale = 0.15 + 0.02 * jnp.cos(parameter + 1.0)
    if case_shape:
        case_index = jnp.arange(math.prod(case_shape), dtype=jnp.float32).reshape(
            case_shape + (1,)
        )
        offset = jnp.broadcast_to(base_offset, case_shape + (parameter_size,))
        offset = offset + 0.01 * case_index
        scale = jnp.broadcast_to(base_scale, case_shape + (parameter_size,))
    else:
        offset = base_offset
        scale = base_scale
    ownership = jax.nn.one_hot(
        jnp.asarray(partition.control_owner, dtype=jnp.int32),
        partition.num_players,
        dtype=jnp.float32,
    ).T
    problem = phx.control.games.StochasticPolicyGameProblem(
        _quadratic_path_costs,
        partition,
        case_shape=case_shape,
        args={"offset": offset, "scale": scale, "ownership": ownership},
        callback_id="benchmark-complete-frozen-path-quadratic-costs",
        feasible_set_id="unconstrained-finite-policy-parameter-space",
        problem_id=(
            f"benchmark-policy-game:r{path_count}:T{horizon}:w{noise_size}:"
            f"parameters{parameter_size}:players{partition.num_players}:"
            f"cases{'x'.join(map(str, case_shape)) or '1'}:clusters{cluster_count}"
        ),
    )
    training_noise = _noise("training", path_count, horizon, noise_size, cluster_count)
    holdout_noise = _noise("holdout", path_count, horizon, noise_size, cluster_count)
    initial_parameters = jnp.zeros(case_shape + (parameter_size,), dtype=jnp.float32)
    termination = NonlinearTermination(
        absolute_residual=1.0e-6,
        relative_residual=1.0e-6,
        absolute_step=1.0e-7,
        relative_step=1.0e-6,
        maximum_steps=20,
    )
    plan = phx.control.games.plan_stochastic_policy_game(
        problem,
        termination=termination,
    )
    prepared = phx.control.games.prepare_stochastic_policy_game(
        plan,
        problem,
        initial_parameters,
        training_noise,
        holdout_noise,
    )
    return prepared


def _certificates(prepared, result) -> dict[str, Any]:
    training_ids = set(prepared.training_realization_ids)
    holdout_ids = set(prepared.holdout_realization_ids)
    training_signal_mean = jnp.mean(prepared.training_noise.increments)
    expected_parameters = (
        prepared.problem.args["offset"]
        + prepared.problem.args["scale"] * training_signal_mean
    )
    cluster_count = int(result.holdout_cluster_costs.shape[-2])
    centered_cluster_costs = result.holdout_cluster_costs - jnp.mean(
        result.holdout_cluster_costs, axis=-2, keepdims=True
    )
    cluster_standard_error = jnp.sqrt(
        jnp.sum(centered_cluster_costs * centered_cluster_costs, axis=-2)
        / (cluster_count * (cluster_count - 1))
    )
    return {
        "claim": result.certification_claim,
        "claim_scope": "local stationarity of the declared frozen training-sample policy game",
        "successful": bool(jnp.all(result.successful)),
        "status": jnp.asarray(result.status).tolist(),
        "root_status": jnp.asarray(result.root_status).tolist(),
        "maximum_stationarity_residual": float(
            jnp.max(jnp.abs(result.original_residual))
        ),
        "maximum_analytic_parameter_defect": float(
            jnp.max(jnp.abs(result.parameters - expected_parameters))
        ),
        "training_holdout_realization_ids_disjoint": training_ids.isdisjoint(holdout_ids),
        "training_holdout_coupling_ids_distinct": (
            result.training_coupling_id != result.holdout_coupling_id
        ),
        "training_holdout_bundle_ids_distinct": (
            result.training_bundle_id != result.holdout_bundle_id
        ),
        "frozen_training_ids_reproduced": (
            result.training_realization_ids == prepared.training_realization_ids
        ),
        "frozen_holdout_ids_reproduced": (
            result.holdout_realization_ids == prepared.holdout_realization_ids
        ),
        "all_holdout_clusters_valid": bool(jnp.all(result.holdout_cluster_valid)),
        "holdout_cluster_counts": jnp.asarray(result.holdout_cluster_counts).tolist(),
        "holdout_cluster_weights": jnp.asarray(result.holdout_cluster_weights).tolist(),
        "holdout_saa_costs": jnp.asarray(result.holdout_saa_costs).tolist(),
        "maximum_descriptive_cluster_standard_error": float(
            jnp.max(cluster_standard_error)
        ),
        "statistical_scope": (
            "descriptive independent-cluster holdout summaries; no population "
            "coverage interval is asserted"
        ),
        "output_fingerprint": array_tree_fingerprint(result),
        "not_claimed": [
            "population Nash equilibrium",
            "feedback Nash equilibrium",
            "global policy-game stationarity",
            "holdout population coverage",
        ],
    }


def _case(
    name: str,
    path_count: int,
    horizon: int,
    noise_size: int,
    control_sizes: tuple[int, ...],
    case_shape: tuple[int, ...],
    cluster_count: int,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if cluster_count < 2 or path_count % cluster_count != 0:
        raise ValueError("cluster_count must divide path_count and be at least two")
    prepared = _prepared_problem(
        path_count,
        horizon,
        noise_size,
        control_sizes,
        case_shape,
        cluster_count,
    )
    function = eqx.filter_jit(phx.control.games.solve_prepared_stochastic_policy_game)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(prepared),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(prepared),
        warmup=warmup,
        repeats=repeats,
    )
    compiler = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    numeric_fingerprint_inputs = (
        prepared.initial_parameters,
        prepared.problem.args,
        prepared.training_noise.increments,
        prepared.training_noise.valid,
        prepared.training_noise.independence_labels,
        prepared.holdout_noise.increments,
        prepared.holdout_noise.valid,
        prepared.holdout_noise.independence_labels,
        prepared.training_weights,
        prepared.holdout_weights,
    )
    provenance_fingerprint = canonical_fingerprint(
        {
            "problem_id": prepared.problem.problem_id,
            "callback_id": prepared.problem.callback_id,
            "feasible_set_id": prepared.problem.feasible_set_id,
            "training_ids": list(prepared.training_realization_ids),
            "holdout_ids": list(prepared.holdout_realization_ids),
            "training_coupling_id": prepared.training_noise.coupling_id,
            "holdout_coupling_id": prepared.holdout_noise.coupling_id,
        }
    )
    case_count = math.prod(case_shape) if case_shape else 1
    parameter_size = sum(control_sizes)
    return {
        "name": name,
        "dimensions": {
            "training_paths": path_count,
            "holdout_paths": path_count,
            "horizon": horizon,
            "noise_size": noise_size,
            "control_parameter_sizes": list(control_sizes),
            "joint_parameter_size": parameter_size,
            "players": len(control_sizes),
            "case_shape": list(case_shape),
            "holdout_independent_clusters": cluster_count,
        },
        "dtype": str(prepared.initial_parameters.dtype),
        "input_fingerprint": array_tree_fingerprint(numeric_fingerprint_inputs),
        "provenance_fingerprint": provenance_fingerprint,
        "lower": {
            "seconds": compilation.lowering_seconds,
            "scope": "eqx-filter-jit lowering of the frozen prepared solve",
        },
        "compile": {
            "seconds": compilation.compilation_seconds,
            "scope": "lowered JAX executable compilation",
        },
        "run": {
            **execution.to_milliseconds_dict(),
            "scope": "synchronized root solve plus untouched holdout evaluation",
        },
        "memory": {
            "logical_input_bytes": logical_array_bytes(prepared),
            "logical_output_bytes": logical_array_bytes(result),
            "compiler_argument_bytes": compiler.argument_bytes,
            "compiler_output_bytes": compiler.output_bytes,
            "compiler_temporary_bytes": compiler.temporary_bytes,
            "compiler_generated_code_bytes": compiler.generated_code_bytes,
            "source": compiler.source,
            "unavailable_reason": compiler.unavailable_reason,
        },
        "work": {
            "compiler_flops": compiler.flops,
            "compiler_bytes_accessed": compiler.bytes_accessed,
            "case_count": case_count,
            "training_path_parameters": case_count * path_count * parameter_size,
            "complete_path_gradient_entries": (
                case_count * path_count * len(control_sizes) * parameter_size
            ),
            "holdout_path_player_costs": (case_count * path_count * len(control_sizes)),
            "holdout_cluster_player_costs": (
                case_count * cluster_count * len(control_sizes)
            ),
        },
        "certificate": _certificates(prepared, result),
    }


def _specifications():
    return (
        ("baseline", 32, 8, 2, (2, 2), (), 8),
        ("paths-8", 8, 8, 2, (2, 2), (), 4),
        ("paths-128", 128, 8, 2, (2, 2), (), 32),
        ("horizon-2", 32, 2, 2, (2, 2), (), 8),
        ("horizon-32", 32, 32, 2, (2, 2), (), 8),
        ("parameters-1-1", 32, 8, 2, (1, 1), (), 8),
        ("parameters-8-8", 32, 8, 2, (8, 8), (), 8),
        ("players-4", 32, 8, 2, (1, 1, 1, 1), (), 8),
        ("cases-8", 32, 8, 2, (2, 2), (8,), 8),
        ("clusters-2", 32, 8, 2, (2, 2), (), 2),
        ("clusters-32", 32, 8, 2, (2, 2), (), 32),
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
            noise_size,
            control_sizes,
            case_shape,
            cluster_count,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for (
            name,
            path_count,
            horizon,
            noise_size,
            control_sizes,
            case_shape,
            cluster_count,
        ) in _specifications()
    ]
    payload = {
        "benchmark": "control-stochastic-policy-game-saa",
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_valid": all(
            case["certificate"]["successful"]
            and case["certificate"]["training_holdout_realization_ids_disjoint"]
            and case["certificate"]["training_holdout_coupling_ids_distinct"]
            and case["certificate"]["training_holdout_bundle_ids_distinct"]
            and case["certificate"]["all_holdout_clusters_valid"]
            for case in cases
        ),
    }
    if arguments.output is None:
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    else:
        write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
