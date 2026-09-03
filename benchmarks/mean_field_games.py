#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_repeated,
)
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.control.games._mean_field_common_noise import (
    CommonNoiseMeanFieldPlan,
    CommonNoiseMeanFieldProblem,
    solve_common_noise_mean_field_fixed_point,
)


_HOST_COMPILATION_REASON = (
    "the fixed-capacity solver is a host-controlled evidence driver and exposes no "
    "JAX lowering or executable-compilation boundary"
)


def _law_means(flow):
    return jax.vmap(lambda time: flow.snapshot(time).mean)(flow.times)


def _centered_offsets(particle_count: int, weights: jax.Array) -> jax.Array:
    offsets = jnp.linspace(-0.25, 0.25, particle_count)
    return offsets - jnp.sum(offsets * weights) / jnp.sum(weights)


def _flow(
    case_name: str,
    scenario_index: int,
    particle_count: int,
    time_count: int,
):
    times = jnp.linspace(0.0, 1.0, time_count)
    particle_weights = jnp.linspace(1.0, 2.0, particle_count)
    offsets = _centered_offsets(particle_count, particle_weights)
    scenario_center = float(scenario_index)
    particles = jnp.broadcast_to(
        scenario_center + offsets[:, None, None],
        (particle_count, time_count, 1),
    )
    weights = jnp.broadcast_to(
        particle_weights[:, None],
        (particle_count, time_count),
    )
    return phx.stochastic.EmpiricalMeanField(
        times,
        particles,
        sample_shape=(particle_count,),
        state_shape=(1,),
        mean_field_id=f"{case_name}:scenario:{scenario_index}:initial",
        weights=weights,
        source_path_id=f"{case_name}:scenario:{scenario_index}:initial-paths",
    )


def _frozen_response(flow, args):
    del args
    paths = phx.stochastic.BSDEPathBatch(
        flow.times,
        flow.particles,
        jnp.zeros(flow.sample_shape + (flow.times.shape[0] - 1, 1)),
        sample_shape=flow.sample_shape,
        state_shape=flow.state_shape,
        noise_shape=(1,),
        path_id=f"candidate-paths:{flow.mean_field_id}",
        process_id="benchmark-frozen-response-process",
    )
    adapter = phx.stochastic.MeanFieldBSDEControlAdapter(
        lambda time, state, law, value, z, adapter_args: -z.reshape((1,)),
        lambda time, state, law, action, adapter_args: 0.5 * action**2,
        lambda time, state, law, action, adapter_args: action,
        control_shape=(1,),
        output_shape=(1,),
        noise_shape=(1,),
        adapter_id="benchmark-frozen-response-adapter",
    )
    base = phx.stochastic.adapt_mean_field_control_bsde(
        lambda key: paths,
        flow,
        lambda time, state, law, base_args: jnp.zeros((1,)),
        lambda time, state, law, base_args: jnp.ones((1, 1)),
        lambda state, law, base_args: jnp.zeros((1,)),
        adapter,
        state_shape=(1,),
        problem_id=f"benchmark-base:{flow.mean_field_id}",
        process_id=paths.process_id,
    )
    problem = phx.control.games.FrozenLawBestResponseProblem(
        base,
        adapter,
        supplied_law_id=f"supplied:{flow.mean_field_id}",
        problem_id=f"benchmark-frozen:{flow.mean_field_id}",
    )
    return phx.control.games.solve_frozen_law_best_response(
        problem,
        paths,
        lambda time, state: jnp.zeros((1,)),
        control_predictor=lambda time, state: jnp.zeros((1, 1)),
        key=jr.key(0),
        minimum_effective_sample_size=2.0,
    )


def _induce_toward(response, target: float):
    current = response.mean_field
    induced_means = target + 0.5 * (_law_means(current) - target)
    particle_weights = current.weights[:, 0]
    offsets = _centered_offsets(current.num_particles, particle_weights)
    particles = induced_means[None, ...] + offsets[:, None, None]
    return phx.stochastic.EmpiricalMeanField(
        current.times,
        particles,
        sample_shape=current.sample_shape,
        state_shape=current.state_shape,
        mean_field_id=f"induced:{response.flow_id}",
        weights=current.weights,
        source_path_id=f"independent-forward-paths:{response.flow_id}",
    )


def _induced_flow(response, target):
    return _induce_toward(response, float(target))


def _conditional_frozen_response(flow, history, args):
    del history
    return _frozen_response(flow, args)


def _conditional_induced_flow(response, history, args):
    del args
    return _induce_toward(response, float(history))


def _law_distance(current, induced, args):
    del args
    return jnp.max(jnp.abs(_law_means(current) - _law_means(induced)))


def _conditional_law_distance(current, induced, history, args):
    del history
    return _law_distance(current, induced, args)


def _build_unconditional(
    name: str,
    particle_count: int,
    time_count: int,
    outer_iterations: int,
):
    flow = _flow(name, 0, particle_count, time_count)
    problem_id = f"{name}:unconditional-fixed-point"
    problem = phx.control.games.MeanFieldGameFixedPointProblem(
        flow,
        _frozen_response,
        _induced_flow,
        _law_distance,
        best_response_id="benchmark-frozen-law-response",
        induced_flow_id="benchmark-independent-forward-law",
        law_distance_id="maximum-time-node-mean-distance",
        problem_id=problem_id,
    )
    plan = phx.control.games.MeanFieldGameFixedPointPlan(
        maximum_iterations=outer_iterations,
        consistency_tolerance=0.2501 * 0.5 ** (outer_iterations - 1),
        damping=1.0,
        minimum_effective_sample_size=2.0,
        problem_id=problem_id,
    )
    return (
        problem,
        plan,
        lambda: phx.control.games.solve_mean_field_game_fixed_point(
            problem,
            plan,
            args=0.5,
        ),
    )


def _build_conditional(
    name: str,
    particle_count: int,
    time_count: int,
    outer_iterations: int,
    common_scenario_count: int,
):
    flows = tuple(
        _flow(name, scenario, particle_count, time_count)
        for scenario in range(common_scenario_count)
    )
    histories = tuple(float(scenario) + 0.5 for scenario in range(common_scenario_count))
    probability = 1.0 / common_scenario_count
    problem_id = f"{name}:conditional-fixed-point"
    problem = CommonNoiseMeanFieldProblem(
        flows,
        histories,
        jnp.full((common_scenario_count,), probability),
        tuple(tuple(range(particle_count)) for _ in flows),
        _conditional_frozen_response,
        _conditional_induced_flow,
        _conditional_law_distance,
        scenario_ids=tuple(
            f"{name}:common-scenario:{scenario}"
            for scenario in range(common_scenario_count)
        ),
        common_history_ids=tuple(
            f"{name}:common-history:{scenario}"
            for scenario in range(common_scenario_count)
        ),
        best_response_id="benchmark-conditional-frozen-law-response",
        induced_flow_id="benchmark-independent-conditional-forward-law",
        law_distance_id="maximum-conditional-time-node-mean-distance",
        problem_id=problem_id,
    )
    plan = CommonNoiseMeanFieldPlan(
        maximum_iterations=outer_iterations,
        consistency_tolerance=0.2501 * 0.5 ** (outer_iterations - 1),
        damping=1.0,
        minimum_effective_sample_size=2.0,
        minimum_independent_clusters=2,
        problem_id=problem_id,
    )
    return (
        problem,
        plan,
        lambda: solve_common_noise_mean_field_fixed_point(
            problem,
            plan,
        ),
    )


def _compiler_record() -> dict[str, Any]:
    evidence = compiler_evidence(
        None,
        None,
        source="host-controlled-fixed-point-driver",
        unavailable_reason=_HOST_COMPILATION_REASON,
    )
    return {
        "flops": evidence.flops,
        "bytes_accessed": evidence.bytes_accessed,
        "argument_bytes": evidence.argument_bytes,
        "output_bytes": evidence.output_bytes,
        "temporary_bytes": evidence.temporary_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
        "source": evidence.source,
        "unavailable_reason": evidence.unavailable_reason,
    }


def _array_inputs(problem, plan):
    if isinstance(problem, CommonNoiseMeanFieldProblem):
        flows = problem.initial_conditional_flows
        return (
            tuple((flow.times, flow.particles, flow.weights) for flow in flows),
            problem.scenario_probabilities,
            jnp.asarray(plan.consistency_tolerance),
        )
    flow = problem.initial_flow
    return (
        flow.times,
        flow.particles,
        flow.weights,
        jnp.asarray(plan.consistency_tolerance),
    )


def _result_evidence(result, conditional: bool) -> dict[str, Any]:
    iterations = int(result.iterations)
    if conditional:
        distance_history = result.distance_history[:iterations]
        aggregate_history = result.aggregate_distance_history[:iterations]
        maximum_history = result.maximum_conditional_distance_history[:iterations]
        current_ess = result.current_effective_sample_size_history[:iterations]
        induced_ess = result.induced_effective_sample_size_history[:iterations]
        final_distances = result.final_distances
        claims = {
            "candidate_evaluation_only": result.candidate_evaluation_only,
            "conditional_law_consistency_evaluated": (
                result.conditional_law_consistency_evaluated
            ),
            "unconditional_law_consistency_evaluated": (
                result.unconditional_law_consistency_evaluated
            ),
            "best_response_optimality_evaluated": (
                result.best_response_optimality_evaluated
            ),
            "mean_field_game_equilibrium_claimed": (
                result.mean_field_game_equilibrium_claimed
            ),
            "common_noise_equilibrium_claimed": (result.common_noise_equilibrium_claimed),
            "unconditional_mean_field_equilibrium_claimed": (
                result.unconditional_mean_field_equilibrium_claimed
            ),
            "mean_field_control_optimum_claimed": (
                result.mean_field_control_optimum_claimed
            ),
            "finite_population_game_claimed": result.finite_population_game_claimed,
        }
    else:
        distance_history = result.distance_history[:iterations, None]
        aggregate_history = result.distance_history[:iterations]
        maximum_history = result.distance_history[:iterations]
        current_ess = result.current_effective_sample_size_history[:iterations, None]
        induced_ess = result.induced_effective_sample_size_history[:iterations, None]
        final_distances = result.final_distance[None]
        claims = {
            "candidate_evaluation_only": result.candidate_evaluation_only,
            "law_consistency_evaluated": result.law_consistency_evaluated,
            "best_response_optimality_evaluated": (
                result.best_response_optimality_evaluated
            ),
            "mean_field_game_equilibrium_claimed": (
                result.mean_field_game_equilibrium_claimed
            ),
            "common_noise_equilibrium_claimed": (result.common_noise_equilibrium_claimed),
            "mean_field_control_optimum_claimed": (
                result.mean_field_control_optimum_claimed
            ),
            "finite_population_game_claimed": result.finite_population_game_claimed,
        }
    return {
        "law_distance": {
            "conditional_history": distance_history.tolist(),
            "aggregate_history": aggregate_history.tolist(),
            "maximum_history": maximum_history.tolist(),
            "final_by_scenario": final_distances.tolist(),
            "maximum_final": float(jnp.max(final_distances)),
        },
        "effective_sample_size": {
            "current_history": current_ess.tolist(),
            "induced_history": induced_ess.tolist(),
            "minimum_current": float(jnp.min(current_ess)),
            "minimum_induced": float(jnp.min(induced_ess)),
        },
        "certificate": {
            "valid": bool(result.valid),
            "status": int(result.status),
            "label": result.certificate_label,
            "iterations": iterations,
            "accepted_iterations": int(result.accepted_iterations),
            "accepted_iteration": int(result.accepted_iteration),
            "claims": claims,
        },
    }


def _case(
    name: str,
    particle_count: int,
    time_count: int,
    outer_iterations: int,
    common_scenario_count: int,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    conditional = common_scenario_count > 1
    if conditional:
        problem, plan, operation = _build_conditional(
            name,
            particle_count,
            time_count,
            outer_iterations,
            common_scenario_count,
        )
    else:
        problem, plan, operation = _build_unconditional(
            name,
            particle_count,
            time_count,
            outer_iterations,
        )
    result, execution = measure_repeated(
        operation,
        warmup=warmup,
        repeats=repeats,
    )
    evidence = _result_evidence(result, conditional)
    configuration = {
        "name": name,
        "particle_count_per_scenario": particle_count,
        "time_count": time_count,
        "outer_iteration_capacity": outer_iterations,
        "common_scenario_count": common_scenario_count,
        "conditional_common_noise": conditional,
    }
    inputs = _array_inputs(problem, plan)
    return {
        **configuration,
        "input_fingerprint": {
            "configuration_sha256": canonical_fingerprint(configuration),
            "arrays": array_tree_fingerprint(inputs),
        },
        "lower": {
            "seconds": None,
            "source": "host-controlled-fixed-point-driver",
            "unavailable_reason": _HOST_COMPILATION_REASON,
        },
        "compile": _compiler_record(),
        "run": execution.to_milliseconds_dict(),
        "memory": {
            "logical_input_bytes": logical_array_bytes((problem, plan)),
            "logical_output_bytes": logical_array_bytes(result),
            "compiler_estimated_device_bytes": None,
            "compiler_estimate_unavailable_reason": _HOST_COMPILATION_REASON,
        },
        "work": {
            "particle_time_nodes": (particle_count * time_count * common_scenario_count),
            "time_intervals_per_path": time_count - 1,
            "frozen_law_evaluation_capacity": (outer_iterations * common_scenario_count),
            "induced_law_evaluation_capacity": (outer_iterations * common_scenario_count),
            "law_distance_evaluation_capacity": (
                outer_iterations * common_scenario_count
            ),
        },
        "law_distance_evidence": evidence["law_distance"],
        "effective_sample_size_evidence": evidence["effective_sample_size"],
        "certificate": evidence["certificate"],
    }


def _specifications():
    return (
        ("baseline", 16, 8, 4, 1),
        ("particles-64", 64, 8, 4, 1),
        ("time-nodes-32", 16, 32, 4, 1),
        ("outer-iterations-8", 16, 8, 8, 1),
        ("common-scenarios-4", 16, 8, 4, 4),
        ("common-scenarios-16", 16, 8, 4, 16),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive.")

    cases = [
        _case(
            name,
            particle_count,
            time_count,
            outer_iterations,
            common_scenario_count,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for (
            name,
            particle_count,
            time_count,
            outer_iterations,
            common_scenario_count,
        ) in _specifications()
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_valid": all(case["certificate"]["valid"] for case in cases),
    }
    if arguments.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
