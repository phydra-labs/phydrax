#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import equinox as eqx
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
from phydrax._fingerprint import array_tree_fingerprint


def _problem(
    horizon: int,
    state_size: int,
    control_sizes: tuple[int, ...],
    case_count: int,
    /,
):
    control_size = sum(control_sizes)
    players = len(control_sizes)
    partition = phx.control.games.PlayerControlPartition(
        tuple(f"player-{index}" for index in range(players)),
        control_sizes,
    )
    time_grid = phx.dynamics.TimeGrid(
        jnp.arange(horizon + 1, dtype=float),
        time_id=(
            f"benchmark-ilq-feedback:T{horizon}:n{state_size}:"
            f"m{control_size}:p{players}:c{case_count}"
        ),
    )
    state_index = jnp.arange(state_size, dtype=float)
    control_index = jnp.arange(control_size, dtype=float)
    dynamics_matrix = 0.82 * jnp.eye(state_size)
    control_matrix = 0.08 * jnp.cos(
        (state_index[:, None] + 1.0) * (control_index[None, :] + 1.0)
    )

    def transition(context, state, control, args):
        del context
        return args["A"] @ state + args["B"] @ control + 0.015 * jnp.sin(state)

    def stage_cost(player, start, stop):
        def cost(context, state, control, args):
            del context, state, args
            target = 0.18 + 0.02 * player
            error = control[start:stop] - target
            return 0.5 * jnp.sum(error**2) + 0.005 * jnp.sum(error**4)

        return cost

    def terminal_cost(time, state, args):
        del time, state, args
        return jnp.asarray(0.0)

    stage_costs = tuple(
        stage_cost(player, start, stop)
        for player, (start, stop) in enumerate(partition.control_slices)
    )
    terminal_costs = (terminal_cost,) * players
    initial = jnp.linspace(-0.65, 0.65, state_size)
    case_shape = () if case_count == 1 else (case_count,)
    if case_shape:
        offsets = jnp.linspace(-0.1, 0.1, case_count)[:, None]
        initial = jnp.broadcast_to(initial, case_shape + (state_size,)) + offsets
    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((state_size,)),
        input_layout=phx.dynamics.InputLayout((control_size,), roles="control"),
        system_id=f"{time_grid.time_id}:system",
    )
    problem = phx.control.games.DeterministicFeedbackGameProblem(
        phx.control.DiscreteControlDynamics(system),
        time_grid,
        initial,
        partition,
        stage_costs=stage_costs,
        terminal_costs=terminal_costs,
        args={"A": dynamics_matrix, "B": control_matrix},
        problem_id=f"{time_grid.time_id}:problem",
    )
    initial_policy = phx.control.AffineFeedbackPolicy(
        jnp.zeros(case_shape + (horizon, control_size, state_size)),
        jnp.zeros(case_shape + (horizon, control_size)),
        time_grid=time_grid,
        state_size=state_size,
        case_shape=case_shape,
        policy_id=f"{time_grid.time_id}:initial-policy",
    )
    scaling = phx.control.games.ILQGameScaling(
        jnp.ones(state_size),
        jnp.ones(control_size),
        jnp.ones(players),
        scaling_id=f"{time_grid.time_id}:scaling",
    )
    plan = phx.control.games.plan_ilq_feedback_game(
        problem,
        scaling,
        maximum_iterations=10,
        maximum_line_search_steps=8,
        residual_tolerance=5.0e-5,
        step_tolerance=5.0e-5,
        dynamics_tolerance=5.0e-6,
    )
    return phx.control.games.prepare_ilq_feedback_game(plan, problem, initial_policy)


def _case(
    name: str,
    horizon: int,
    state_size: int,
    control_sizes: tuple[int, ...],
    case_count: int,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    prepared = _problem(horizon, state_size, control_sizes, case_count)
    function = eqx.filter_jit(phx.control.games.solve_prepared_ilq_feedback_game)
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
    diagnostics = result.diagnostics
    certificate = {
        "label": result.certificate_label,
        "valid": bool(jnp.all(result.valid)),
        "status": jnp.asarray(result.status).tolist(),
        "stationarity_rms_max": float(jnp.max(result.residual.stationarity_rms_norm)),
        "stationarity_infinity_max": float(
            jnp.max(result.residual.stationarity_infinity_norm)
        ),
        "dynamics_rms_max": float(jnp.max(result.residual.dynamics_defect_rms_norm)),
        "dynamics_infinity_max": float(
            jnp.max(result.residual.dynamics_defect_infinity_norm)
        ),
        "final_unregularized_local_valid": bool(
            jnp.all(diagnostics.final_unregularized_local_valid)
        ),
        "feedback_nash_claimed": diagnostics.feedback_nash_claimed,
        "global_convergence_claimed": diagnostics.global_convergence_claimed,
    }
    return {
        "name": name,
        "horizon": horizon,
        "state_size": state_size,
        "control_sizes": list(control_sizes),
        "joint_control_size": sum(control_sizes),
        "player_count": len(control_sizes),
        "case_count": case_count,
        "dtype": str(prepared.problem.initial_state.dtype),
        "input_fingerprint": array_tree_fingerprint(prepared),
        "lower": {"seconds": compilation.lowering_seconds},
        "compile": {"seconds": compilation.compilation_seconds},
        "run": execution.to_milliseconds_dict(),
        "memory": {
            "logical_input_bytes": logical_array_bytes(prepared),
            "logical_output_bytes": logical_array_bytes(result),
            "compiler_argument_bytes": compiler.argument_bytes,
            "compiler_output_bytes": compiler.output_bytes,
            "compiler_temporary_bytes": compiler.temporary_bytes,
            "compiler_generated_code_bytes": compiler.generated_code_bytes,
            "compiler_source": compiler.source,
            "compiler_unavailable_reason": compiler.unavailable_reason,
        },
        "work": {
            "compiler_flops": compiler.flops,
            "compiler_bytes_accessed": compiler.bytes_accessed,
            "iterations_max": int(jnp.max(diagnostics.iterations)),
            "accepted_iterations_sum": int(jnp.sum(diagnostics.accepted_iterations)),
            "line_search_evaluations_sum": int(
                jnp.sum(diagnostics.line_search_evaluations_history)
            ),
            "evaluated_trials_sum": int(jnp.sum(diagnostics.trial_history_valid)),
        },
        "certificate": certificate,
    }


def _specifications():
    return (
        ("baseline", 6, 4, (1, 1), 1),
        ("horizon-2", 2, 4, (1, 1), 1),
        ("horizon-12", 12, 4, (1, 1), 1),
        ("state-2", 6, 2, (1, 1), 1),
        ("state-8", 6, 8, (1, 1), 1),
        ("controls-2-2", 6, 4, (2, 2), 1),
        ("players-1", 6, 4, (2,), 1),
        ("players-4", 6, 4, (1, 1, 1, 1), 1),
        ("cases-8", 6, 4, (1, 1), 8),
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
            horizon,
            state_size,
            control_sizes,
            case_count,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for name, horizon, state_size, control_sizes, case_count in _specifications()
    ]
    residuals = [
        value
        for case in cases
        for key, value in case["certificate"].items()
        if key.endswith("_max")
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_valid": all(case["certificate"]["valid"] for case in cases),
        "all_residuals_finite": all(math.isfinite(value) for value in residuals),
    }
    if arguments.output is None:
        print(json.dumps(payload, indent=2))
    else:
        write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
