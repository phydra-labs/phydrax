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
from phydrax.nonlinear import NonlinearTermination


def _path_block(
    function,
    constraint_id: str,
    *,
    scope,
    participants: tuple[str, ...],
    owner: str | None,
    residual_shape: tuple[int, ...],
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
        residual_shape=residual_shape,
        time_dependent=False,
        state_dependent=False,
        control_dependencies=control_dependencies,
    )


def _problem(
    horizon: int,
    state_size: int,
    control_sizes: tuple[int, ...],
    case_count: int,
    shared_constraint_count: int,
    /,
):
    players = len(control_sizes)
    control_size = sum(control_sizes)
    player_ids = tuple(f"player-{index}" for index in range(players))
    partition = phx.control.games.PlayerControlPartition(player_ids, control_sizes)
    blocks = []
    for player, ((start, stop), player_id) in enumerate(
        zip(partition.control_slices, player_ids, strict=True)
    ):
        blocks.append(
            _path_block(
                lambda time, state, control, args, start=start, stop=stop: (
                    -control[start:stop]
                ),
                f"{player_id}-nonnegative",
                scope=phx.control.games.GameConstraintScope.PLAYER_LOCAL,
                participants=(player_id,),
                owner=player_id,
                residual_shape=(stop - start,),
                control_dependencies=(player_id,),
            )
        )
    for constraint in range(shared_constraint_count):
        capacity = control_size * (0.4 + 0.15 * constraint)
        blocks.append(
            _path_block(
                lambda time, state, control, args, capacity=capacity: (
                    jnp.sum(control) - capacity
                ),
                f"shared-resource-{constraint}",
                scope=phx.control.games.GameConstraintScope.SHARED,
                participants=player_ids,
                owner=None,
                residual_shape=(),
                control_dependencies=player_ids,
            )
        )
    constraints = phx.control.games.OpenLoopGameConstraints(partition, tuple(blocks))

    state_index = jnp.arange(state_size, dtype=float)
    control_index = jnp.arange(control_size, dtype=float)
    a = jnp.broadcast_to(0.84 * jnp.eye(state_size), (horizon, state_size, state_size))
    b_base = 0.025 * jnp.cos(
        (state_index[:, None] + 1.0) * (control_index[None, :] + 1.0)
    )
    b = jnp.broadcast_to(b_base, (horizon, state_size, control_size))
    initial = jnp.linspace(-0.2, 0.2, state_size)
    q = jnp.zeros((players, horizon, state_size, state_size))
    r = jnp.zeros((players, horizon, control_size, control_size))
    linear = jnp.zeros((players, horizon, control_size))
    for player, (start, stop) in enumerate(partition.control_slices):
        owned = jnp.arange(start, stop)
        r = r.at[player, :, owned, owned].set(1.0 + 0.05 * player)
        linear = linear.at[player, :, start:stop].set(-(0.72 + 0.03 * player))
    terminal_q = jnp.zeros((players, state_size, state_size))

    case_shape = () if case_count == 1 else (case_count,)
    if case_shape:
        initial = jnp.broadcast_to(initial, case_shape + (state_size,))
        initial = initial + jnp.linspace(-0.05, 0.05, case_count)[:, None]
        arrays = (a, b, q, r, terminal_q, linear)
        a, b, q, r, terminal_q, linear = tuple(
            jnp.broadcast_to(value, case_shape + value.shape) for value in arrays
        )
    problem_id = (
        f"benchmark-polyhedral-ve:T{horizon}:n{state_size}:m{control_size}:"
        f"p{players}:c{case_count}:g{shared_constraint_count}"
    )
    problem = phx.control.games.FiniteHorizonLQOpenLoopVEProblem(
        a,
        b,
        initial,
        q,
        r,
        terminal_q,
        partition,
        constraints=constraints,
        control_linear=linear,
        problem_id=problem_id,
    )
    plan = phx.control.games.plan_open_loop_ve(
        problem,
        termination=NonlinearTermination(maximum_steps=32),
        kkt_tolerance=2.0e-6,
        natural_residual_tolerance=2.0e-6,
    )
    initial_controls = jnp.zeros(
        case_shape + (horizon, control_size), dtype=problem.dynamics_matrices.dtype
    )
    return phx.control.games.prepare_open_loop_ve(plan, problem, initial_controls)


def _case(
    name: str,
    horizon: int,
    state_size: int,
    control_sizes: tuple[int, ...],
    case_count: int,
    shared_constraint_count: int,
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
        shared_constraint_count,
    )
    function = eqx.filter_jit(phx.control.games.solve_prepared_open_loop_ve)
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
    vi_diagnostics = result.vi_result.diagnostics
    certificate = {
        "label": result.certificate_label,
        "claim": result.certification_claim,
        "claim_boundary": "convex open-loop variational equilibrium, not feedback Nash",
        "valid": bool(jnp.all(result.valid)),
        "status": jnp.asarray(result.status).tolist(),
        "convexity_certified": bool(jnp.all(result.convexity_certified)),
        "monotone": bool(jnp.all(result.monotone)),
        "original_stationarity_residual_max": float(
            jnp.max(result.original_stationarity_residual)
        ),
        "original_equality_residual_max": float(
            jnp.max(result.original_equality_residual)
        ),
        "original_inequality_violation_max": float(
            jnp.max(result.original_inequality_violation)
        ),
        "original_dual_violation_max": float(jnp.max(result.original_dual_violation)),
        "original_complementarity_residual_max": float(
            jnp.max(result.original_complementarity_residual)
        ),
        "original_kkt_residual_max": float(jnp.max(result.original_kkt_residual)),
        "natural_residual_max": float(jnp.max(result.natural_residual)),
        "common_shared_multiplier_count": int(result.shared_multipliers.shape[-1]),
    }
    return {
        "name": name,
        "horizon": horizon,
        "state_size": state_size,
        "control_sizes": list(control_sizes),
        "joint_control_size": sum(control_sizes),
        "player_count": len(control_sizes),
        "case_count": case_count,
        "shared_constraint_count": shared_constraint_count,
        "physical_constraint_rows": prepared.plan.constraint_layout.num_residuals,
        "dtype": str(prepared.problem.dynamics_matrices.dtype),
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
            "vi_iterations_max": int(jnp.max(vi_diagnostics.iterations)),
            "vi_residual_evaluations_sum": int(
                jnp.sum(vi_diagnostics.residual_evaluations)
            ),
            "vi_jacobian_preparations_sum": int(
                jnp.sum(vi_diagnostics.jacobian_preparations)
            ),
            "vi_linear_solves_sum": int(jnp.sum(vi_diagnostics.linear_solves)),
            "vi_linear_iterations_sum": int(jnp.sum(vi_diagnostics.linear_iterations)),
            "phase_one_iterations_max": int(jnp.max(result.phase_one_result.iterations)),
            "projection_iterations_max": int(
                jnp.max(result.projection_result.iterations)
            ),
            "counts_complete": vi_diagnostics.counts_complete,
        },
        "certificate": certificate,
    }


def _specifications():
    return (
        ("baseline", 4, 4, (1, 1), 1, 1),
        ("horizon-1", 1, 4, (1, 1), 1, 1),
        ("horizon-12", 12, 4, (1, 1), 1, 1),
        ("state-1", 4, 1, (1, 1), 1, 1),
        ("state-8", 4, 8, (1, 1), 1, 1),
        ("controls-2-2", 4, 4, (2, 2), 1, 1),
        ("players-4", 4, 4, (1, 1, 1, 1), 1, 1),
        ("cases-8", 4, 4, (1, 1), 8, 1),
        ("shared-constraints-4", 4, 4, (1, 1), 1, 4),
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
            shared_constraints,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for (
            name,
            horizon,
            state_size,
            control_sizes,
            case_count,
            shared_constraints,
        ) in _specifications()
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
