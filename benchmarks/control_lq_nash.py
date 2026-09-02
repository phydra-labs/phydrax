#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
import phydrax.ein as ein
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax._fingerprint import array_tree_fingerprint


BenchmarkArguments = tuple[jax.Array, ...]


def _broadcast_case(value: jax.Array, case_shape: tuple[int, ...], /) -> jax.Array:
    return jnp.broadcast_to(value, case_shape + value.shape)


def _problem(
    horizon: int,
    state_size: int,
    control_sizes: tuple[int, ...],
    case_shape: tuple[int, ...],
    /,
) -> tuple[
    BenchmarkArguments, phx.control.games.PlayerControlPartition, phx.dynamics.TimeGrid
]:
    control_size = sum(control_sizes)
    player_count = len(control_sizes)
    time = jnp.arange(horizon, dtype=jnp.float64)
    state_index = jnp.arange(state_size, dtype=jnp.float64)
    control_index = jnp.arange(control_size, dtype=jnp.float64)
    identity = jnp.eye(state_size, dtype=jnp.float64)
    dynamics = jnp.stack(
        [(0.82 + 0.02 * jnp.cos(0.2 * step)) * identity for step in range(horizon)]
    )
    control_base = 0.02 * jnp.cos(
        (state_index[:, None] + 1.0) * (control_index[None, :] + 1.0)
    )
    controls = jnp.stack([control_base * (1.0 + 0.01 * step) for step in range(horizon)])
    bias = 0.005 * jnp.sin(time[:, None] + state_index[None, :])

    state_costs = jnp.stack(
        [
            jnp.broadcast_to(
                (0.5 + 0.1 * player) * identity,
                (horizon, state_size, state_size),
            )
            for player in range(player_count)
        ]
    )
    control_costs = []
    cross = []
    state_linear = []
    control_linear = []
    stage_constants = []
    for player in range(player_count):
        diagonal = 1.5 + 0.1 * player + 0.01 * control_index
        control_cost = jnp.diag(diagonal) + 0.002 * (
            jnp.ones((control_size, control_size), dtype=jnp.float64)
            - jnp.eye(control_size, dtype=jnp.float64)
        )
        control_costs.append(
            jnp.broadcast_to(control_cost, (horizon, control_size, control_size))
        )
        cross_base = (
            0.001
            * (player + 1)
            * jnp.sin((state_index[:, None] + 1.0) + (control_index[None, :] + 1.0))
        )
        cross.append(jnp.broadcast_to(cross_base, (horizon,) + cross_base.shape))
        state_linear.append(
            0.002 * (player + 1) * jnp.cos(time[:, None] + state_index[None, :])
        )
        control_linear.append(
            0.002 * (player + 1) * jnp.sin(time[:, None] + control_index[None, :])
        )
        stage_constants.append(0.001 * (player + 1) * (time + 1.0))
    control_costs_array = jnp.stack(control_costs)
    cross_array = jnp.stack(cross)
    state_linear_array = jnp.stack(state_linear)
    control_linear_array = jnp.stack(control_linear)
    stage_constants_array = jnp.stack(stage_constants)
    terminal_state_costs = jnp.stack(
        [(0.8 + 0.1 * player) * identity for player in range(player_count)]
    )
    terminal_linear = jnp.stack(
        [
            0.003 * (player + 1) * jnp.cos(state_index + 1.0)
            for player in range(player_count)
        ]
    )
    terminal_constants = 0.01 * (jnp.arange(player_count, dtype=jnp.float64) + 1.0)
    arguments = tuple(
        _broadcast_case(value, case_shape)
        for value in (
            dynamics,
            controls,
            state_costs,
            control_costs_array,
            terminal_state_costs,
            bias,
            cross_array,
            state_linear_array,
            control_linear_array,
            stage_constants_array,
            terminal_linear,
            terminal_constants,
        )
    )
    partition = phx.control.games.PlayerControlPartition(
        tuple(f"player-{player}" for player in range(player_count)),
        control_sizes,
    )
    time_grid = phx.dynamics.TimeGrid(
        jnp.arange(horizon + 1, dtype=jnp.float64),
        time_id=(
            f"benchmark-lq-nash:T{horizon}:n{state_size}:m{control_size}:p{player_count}"
        ),
    )
    return arguments, partition, time_grid


def _solve_function(partition, time_grid):
    def solve_game(arguments):
        (
            dynamics,
            controls,
            state_costs,
            control_costs,
            terminal_state_costs,
            bias,
            cross,
            state_linear,
            control_linear,
            stage_constants,
            terminal_linear,
            terminal_constants,
        ) = arguments
        return phx.control.games.finite_horizon_lq_feedback_nash(
            dynamics,
            controls,
            state_costs,
            control_costs,
            terminal_state_costs,
            partition,
            dynamics_bias=bias,
            state_control_cross=cross,
            state_linear=state_linear,
            control_linear=control_linear,
            stage_constants=stage_constants,
            terminal_linear=terminal_linear,
            terminal_constants=terminal_constants,
            time_grid=time_grid,
            policy_id=f"{time_grid.time_id}:policy",
        )

    return eqx.filter_jit(solve_game)


def _certificates(arguments, partition, result) -> dict[str, float]:
    (
        dynamics,
        controls,
        state_costs,
        control_costs,
        terminal_state_costs,
        bias,
        cross,
        state_linear,
        control_linear,
        stage_constants,
        terminal_linear,
        terminal_constants,
    ) = arguments
    del stage_constants, terminal_constants
    case_shape = dynamics.shape[:-3]
    player_axis = len(case_shape)
    value_matrices = jnp.stack(
        [value.matrices for value in result.values],
        axis=player_axis,
    )
    value_linear = jnp.stack(
        [value.linear for value in result.values],
        axis=player_axis,
    )
    maximum_stationarity = jnp.asarray(0.0, dtype=dynamics.dtype)
    maximum_bellman = jnp.asarray(0.0, dtype=dynamics.dtype)
    for step in range(dynamics.shape[-3]):
        a = dynamics[..., step, :, :]
        b = controls[..., step, :, :]
        c = bias[..., step, :]
        k_matrix = result.feedback_gain[..., step, :, :]
        k_vector = result.feedforward[..., step, :]
        z_next = value_matrices[..., :, step + 1, :, :]
        z_linear_next = value_linear[..., :, step + 1, :]
        h = (
            control_costs[..., :, step, :, :]
            + jnp.swapaxes(b, -1, -2)[..., None, :, :] @ z_next @ b[..., None, :, :]
        )
        w = jnp.swapaxes(b, -1, -2)[..., None, :, :] @ z_next @ a[
            ..., None, :, :
        ] + jnp.swapaxes(cross[..., :, step, :, :], -1, -2)
        affine_next = ein.contract("...pij,...j->...pi", z_next, c) + z_linear_next
        g = control_linear[..., :, step, :] + ein.contract(
            "...ji,...pj->...pi",
            b,
            affine_next,
        )
        stationarity_matrix_blocks = []
        stationarity_vector_blocks = []
        for player, (start, stop) in enumerate(partition.control_slices):
            stationarity_matrix_blocks.append(
                h[..., player, start:stop, :] @ k_matrix + w[..., player, start:stop, :]
            )
            stationarity_vector_blocks.append(
                ein.contract(
                    "...ij,...j->...i",
                    h[..., player, start:stop, :],
                    k_vector,
                )
                + g[..., player, start:stop]
            )
        stationarity_matrix = jnp.concatenate(stationarity_matrix_blocks, axis=-2)
        stationarity_vector = jnp.concatenate(stationarity_vector_blocks, axis=-1)
        stationarity_scale = jnp.maximum(
            jnp.maximum(jnp.max(jnp.abs(w)), jnp.max(jnp.abs(g))),
            1.0,
        )
        maximum_stationarity = jnp.maximum(
            maximum_stationarity,
            jnp.maximum(
                jnp.max(jnp.abs(stationarity_matrix)),
                jnp.max(jnp.abs(stationarity_vector)),
            )
            / stationarity_scale,
        )

        closed_loop = a + b @ k_matrix
        closed_bias = c + ein.contract("...ij,...j->...i", b, k_vector)
        direct_z = (
            state_costs[..., :, step, :, :]
            + cross[..., :, step, :, :] @ k_matrix[..., None, :, :]
            + jnp.swapaxes(k_matrix, -1, -2)[..., None, :, :]
            @ jnp.swapaxes(cross[..., :, step, :, :], -1, -2)
            + jnp.swapaxes(k_matrix, -1, -2)[..., None, :, :]
            @ control_costs[..., :, step, :, :]
            @ k_matrix[..., None, :, :]
            + jnp.swapaxes(closed_loop, -1, -2)[..., None, :, :]
            @ z_next
            @ closed_loop[..., None, :, :]
        )
        direct_linear = (
            state_linear[..., :, step, :]
            + ein.contract(
                "...pij,...j->...pi",
                cross[..., :, step, :, :],
                k_vector,
            )
            + ein.contract(
                "...ji,...pj->...pi",
                k_matrix,
                ein.contract(
                    "...pij,...j->...pi",
                    control_costs[..., :, step, :, :],
                    k_vector,
                )
                + control_linear[..., :, step, :],
            )
            + ein.contract(
                "...ji,...pj->...pi",
                closed_loop,
                ein.contract("...pij,...j->...pi", z_next, closed_bias) + z_linear_next,
            )
        )
        z_current = value_matrices[..., :, step, :, :]
        linear_current = value_linear[..., :, step, :]
        bellman_numerator = jnp.maximum(
            jnp.max(jnp.abs(z_current - direct_z)),
            jnp.max(jnp.abs(linear_current - direct_linear)),
        )
        bellman_scale = jnp.maximum(
            jnp.maximum(jnp.max(jnp.abs(direct_z)), jnp.max(jnp.abs(direct_linear))),
            1.0,
        )
        maximum_bellman = jnp.maximum(
            maximum_bellman,
            bellman_numerator / bellman_scale,
        )
    terminal_defect = jnp.maximum(
        jnp.max(jnp.abs(value_matrices[..., :, -1, :, :] - terminal_state_costs)),
        jnp.max(jnp.abs(value_linear[..., :, -1, :] - terminal_linear)),
    )

    state_size = dynamics.shape[-1]
    initial_state = jnp.linspace(-0.25, 0.25, state_size, dtype=dynamics.dtype)
    states = jnp.broadcast_to(initial_state, case_shape + initial_state.shape)
    maximum_rollout = jnp.asarray(0.0, dtype=dynamics.dtype)
    for step in range(dynamics.shape[-3]):
        action = (result.feedback_gain[..., step, :, :] @ states[..., None])[
            ..., 0
        ] + result.feedforward[..., step, :]
        following = (
            (dynamics[..., step, :, :] @ states[..., None])[..., 0]
            + (controls[..., step, :, :] @ action[..., None])[..., 0]
            + bias[..., step, :]
        )
        defect = (
            following
            - ein.contract("...ij,...j->...i", dynamics[..., step, :, :], states)
            - ein.contract("...ij,...j->...i", controls[..., step, :, :], action)
            - bias[..., step, :]
        )
        maximum_rollout = jnp.maximum(maximum_rollout, jnp.max(jnp.abs(defect)))
        states = following
    return {
        "normalized_stationarity": float(maximum_stationarity),
        "normalized_bellman": float(maximum_bellman),
        "terminal_defect": float(terminal_defect),
        "rollout_defect": float(maximum_rollout),
    }


def _case(
    name: str,
    horizon: int,
    state_size: int,
    control_sizes: tuple[int, ...],
    case_shape: tuple[int, ...],
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    arguments, partition, time_grid = _problem(
        horizon,
        state_size,
        control_sizes,
        case_shape,
    )
    function = _solve_function(partition, time_grid)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(arguments),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(arguments),
        warmup=warmup,
        repeats=repeats,
    )
    evidence = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    certificates = _certificates(arguments, partition, result)
    return {
        "name": name,
        "horizon": horizon,
        "state_size": state_size,
        "control_sizes": list(control_sizes),
        "player_count": len(control_sizes),
        "joint_control_size": sum(control_sizes),
        "case_shape": list(case_shape),
        "dtype": str(arguments[0].dtype),
        "input_fingerprint": array_tree_fingerprint(arguments),
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "logical_input_bytes": logical_array_bytes(arguments),
        "logical_output_bytes": logical_array_bytes(result),
        "compiler": {
            "flops": evidence.flops,
            "bytes_accessed": evidence.bytes_accessed,
            "argument_bytes": evidence.argument_bytes,
            "output_bytes": evidence.output_bytes,
            "temporary_bytes": evidence.temporary_bytes,
            "generated_code_bytes": evidence.generated_code_bytes,
            "source": evidence.source,
            "unavailable_reason": evidence.unavailable_reason,
        },
        "valid": bool(jnp.all(result.valid)),
        "status": jnp.asarray(result.status).tolist(),
        "minimum_rank": int(jnp.min(result.diagnostics.coupled_ranks)),
        "maximum_condition": float(jnp.max(result.diagnostics.coupled_condition_numbers)),
        "certificates": certificates,
    }


def _specifications():
    return (
        ("baseline", 16, 8, (2, 2), ()),
        ("horizon-4", 4, 8, (2, 2), ()),
        ("horizon-64", 64, 8, (2, 2), ()),
        ("state-4", 16, 4, (2, 2), ()),
        ("state-32", 16, 32, (2, 2), ()),
        ("controls-1-1", 16, 8, (1, 1), ()),
        ("controls-8-8", 16, 8, (8, 8), ()),
        ("players-1", 16, 8, (8,), ()),
        ("players-2", 16, 8, (4, 4), ()),
        ("players-4", 16, 8, (2, 2, 2, 2), ()),
        ("players-8", 16, 8, (1, 1, 1, 1, 1, 1, 1, 1), ()),
        ("cases-8", 16, 8, (2, 2), (8,)),
        ("cases-64", 16, 8, (2, 2), (64,)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
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
            case_shape,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for name, horizon, state_size, control_sizes, case_shape in _specifications()
    ]
    certificate_values = [
        value for case in cases for value in case["certificates"].values()
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_valid": all(case["valid"] for case in cases),
        "all_finite": all(math.isfinite(value) for value in certificate_values),
        "maximum_certificate": max(certificate_values),
    }
    if arguments.output is None:
        import json

        print(json.dumps(payload, indent=2))
    else:
        write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
