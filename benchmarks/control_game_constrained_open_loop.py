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
from phydrax.control.games._open_loop_kkt import (
    NonlinearOpenLoopGameProblem,
    plan_open_loop_game_kkt,
    prepare_open_loop_game_kkt,
    solve_prepared_open_loop_game_kkt,
)
from phydrax.nonlinear import NonlinearTermination


def _constraint_block(
    function,
    constraint_id: str,
    *,
    scope,
    participants: tuple[str, ...],
    owner: str,
    control_dependencies: tuple[str, ...],
):
    return phx.control.games.GameConstraintBlock(
        phx.control.BoundedPathConstraint(
            function,
            lower=-jnp.inf,
            upper=0.0,
            constraint_id=constraint_id,
        ),
        scope=scope,
        participants=participants,
        owner=owner,
        site=phx.control.games.GameConstraintSite.PATH,
        equality=False,
        residual_shape=(),
        time_dependent=False,
        state_dependent=False,
        control_dependencies=control_dependencies,
    )


def _problem(
    horizon: int,
    state_size: int,
    control_sizes: tuple[int, ...],
    case_count: int,
    constraint_count: int,
    /,
):
    players = len(control_sizes)
    control_size = sum(control_sizes)
    player_ids = tuple(f"player-{index}" for index in range(players))
    partition = phx.control.games.PlayerControlPartition(player_ids, control_sizes)
    blocks = []
    for constraint in range(constraint_count):
        owner_index = constraint % players
        start, stop = partition.control_slices[owner_index]
        owner = player_ids[owner_index]
        repetition = constraint // players
        capacity = 0.38 * (stop - start) + 0.02 * control_size * 0.38**2
        capacity += 0.15 * repetition
        if players == 1:
            scope = phx.control.games.GameConstraintScope.PLAYER_LOCAL
            participants = (owner,)
            dependencies = (owner,)
        else:
            scope = phx.control.games.GameConstraintScope.PLAYER_OWNED_COUPLED
            participants = player_ids
            dependencies = player_ids
        blocks.append(
            _constraint_block(
                lambda time, state, control, args, start=start, stop=stop, capacity=capacity: (
                    jnp.sum(control[start:stop]) + 0.02 * jnp.sum(control**2) - capacity
                ),
                f"private-capacity-{constraint}",
                scope=scope,
                participants=participants,
                owner=owner,
                control_dependencies=dependencies,
            )
        )
    constraints = phx.control.games.OpenLoopGameConstraints(partition, tuple(blocks))

    state_index = jnp.arange(state_size, dtype=float)
    control_index = jnp.arange(control_size, dtype=float)
    a = 0.83 * jnp.eye(state_size)
    b = 0.04 * jnp.sin((state_index[:, None] + 1.0) * (control_index[None, :] + 1.0))

    def transition(context, state, control, args):
        del context
        return args["A"] @ state + args["B"] @ control + 0.01 * jnp.sin(state)

    def stage_cost(player, start, stop):
        def cost(context, state, control, args):
            del context, args
            owned = control[start:stop]
            target = 0.62 + 0.02 * player
            coordinate = player % state_size
            return (
                0.5 * jnp.sum((owned - target) ** 2)
                + 0.003 * jnp.sum(owned**4)
                + 0.005 * state[coordinate] ** 2
            )

        return cost

    def terminal_cost(player):
        def cost(time, state, args):
            del time, args
            coordinate = player % state_size
            return 0.02 * state[coordinate] ** 2

        return cost

    stage_costs = tuple(
        stage_cost(player, start, stop)
        for player, (start, stop) in enumerate(partition.control_slices)
    )
    terminal_costs = tuple(terminal_cost(player) for player in range(players))
    initial = jnp.linspace(-0.15, 0.15, state_size)
    case_shape = () if case_count == 1 else (case_count,)
    if case_shape:
        initial = jnp.broadcast_to(initial, case_shape + (state_size,))
        initial = initial + jnp.linspace(-0.04, 0.04, case_count)[:, None]
    problem_id = (
        f"benchmark-private-open-loop-kkt:T{horizon}:n{state_size}:"
        f"m{control_size}:p{players}:c{case_count}:g{constraint_count}"
    )
    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((state_size,)),
        input_layout=phx.dynamics.InputLayout((control_size,), roles="control"),
        system_id=f"{problem_id}:system",
    )
    problem = NonlinearOpenLoopGameProblem(
        phx.control.DiscreteControlDynamics(system),
        phx.dynamics.TimeGrid(
            jnp.arange(horizon + 1, dtype=float),
            time_id=f"{problem_id}:time",
        ),
        initial,
        partition,
        stage_costs=stage_costs,
        terminal_costs=terminal_costs,
        constraints=constraints,
        args={"A": a, "B": b},
        problem_id=problem_id,
    )
    plan = plan_open_loop_game_kkt(
        problem,
        termination=NonlinearTermination(maximum_steps=32),
        feasibility_tolerance=2.0e-6,
        kkt_tolerance=5.0e-6,
    )
    controls = jnp.zeros(
        case_shape + (horizon, control_size), dtype=problem.initial_state.dtype
    )
    return prepare_open_loop_game_kkt(plan, problem, controls)


def _case(
    name: str,
    horizon: int,
    state_size: int,
    control_sizes: tuple[int, ...],
    case_count: int,
    constraint_count: int,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    prepared = _problem(
        horizon,
        state_size,
        control_sizes,
        case_count,
        constraint_count,
    )
    function = eqx.filter_jit(solve_prepared_open_loop_game_kkt)
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
    diagnostics = result.vi_result.diagnostics
    certificate = {
        "label": result.certificate_label,
        "claim": result.certification_claim,
        "constraint_scope": result.constraint_scope,
        "claim_boundary": "local nominal open-loop KKT, not feedback or global GNE",
        "valid": bool(jnp.all(result.valid)),
        "status": jnp.asarray(result.status).tolist(),
        "finite": bool(jnp.all(result.finite)),
        "feasible": bool(jnp.all(result.feasible)),
        "dynamics_valid": bool(jnp.all(result.dynamics_valid)),
        "constraint_qualification_satisfied": bool(
            jnp.all(result.constraint_qualification_satisfied)
        ),
        "feedback_claim": result.feedback_claim,
        "global_equilibrium_claim": result.global_equilibrium_claim,
        "original_stationarity_residual_max": float(
            jnp.max(result.original_stationarity_residual)
        ),
        "original_equality_residual_max": float(
            jnp.max(result.original_equality_residual)
        ),
        "original_inequality_violation_max": float(
            jnp.max(result.original_inequality_violation)
        ),
        "original_primal_residual_max": float(jnp.max(result.original_primal_residual)),
        "original_dual_violation_max": float(jnp.max(result.original_dual_violation)),
        "original_ncp_residual_max": float(jnp.max(result.original_ncp_residual)),
        "original_complementarity_residual_max": float(
            jnp.max(result.original_complementarity_residual)
        ),
        "original_kkt_residual_max": float(jnp.max(result.original_kkt_residual)),
        "active_constraint_count_max": int(jnp.max(result.active_constraint_count)),
        "private_multiplier_counts": [
            int(value.shape[-1]) for value in result.private_multipliers
        ],
    }
    return {
        "name": name,
        "horizon": horizon,
        "state_size": state_size,
        "control_sizes": list(control_sizes),
        "joint_control_size": sum(control_sizes),
        "player_count": len(control_sizes),
        "case_count": case_count,
        "constraint_block_count": constraint_count,
        "physical_constraint_rows": prepared.plan.constraint_layout.num_residuals,
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
            "vi_iterations_max": int(jnp.max(diagnostics.iterations)),
            "vi_residual_evaluations_sum": int(jnp.sum(diagnostics.residual_evaluations)),
            "vi_jacobian_preparations_sum": int(
                jnp.sum(diagnostics.jacobian_preparations)
            ),
            "vi_linear_solves_sum": int(jnp.sum(diagnostics.linear_solves)),
            "vi_linear_iterations_sum": int(jnp.sum(diagnostics.linear_iterations)),
            "accepted_steps_sum": int(jnp.sum(diagnostics.accepted_steps)),
            "rejected_steps_sum": int(jnp.sum(diagnostics.rejected_steps)),
            "counts_complete": diagnostics.counts_complete,
        },
        "certificate": certificate,
    }


def _specifications():
    return (
        ("baseline", 4, 4, (1, 1), 1, 2),
        ("horizon-1", 1, 4, (1, 1), 1, 2),
        ("horizon-8", 8, 4, (1, 1), 1, 2),
        ("state-1", 4, 1, (1, 1), 1, 2),
        ("state-8", 4, 8, (1, 1), 1, 2),
        ("controls-2-2", 4, 4, (2, 2), 1, 2),
        ("players-1", 4, 4, (2,), 1, 1),
        ("players-4", 4, 4, (1, 1, 1, 1), 1, 4),
        ("cases-8", 4, 4, (1, 1), 8, 2),
        ("constraints-6", 4, 4, (1, 1), 1, 6),
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
            constraint_count,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for (
            name,
            horizon,
            state_size,
            control_sizes,
            case_count,
            constraint_count,
        ) in _specifications()
    ]
    residuals = [
        value
        for case in cases
        for key, value in case["certificate"].items()
        if key.endswith("_max") and isinstance(value, float)
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
