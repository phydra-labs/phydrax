#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark exact additive-noise finite-horizon LQG feedback Nash games."""

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


def _broadcast(value: jax.Array, case_shape: tuple[int, ...], /) -> jax.Array:
    return jnp.broadcast_to(value, case_shape + value.shape)


def _problem(
    horizon: int,
    state_size: int,
    control_sizes: tuple[int, ...],
    noise_size: int,
    case_shape: tuple[int, ...],
    /,
) -> tuple[
    BenchmarkArguments, phx.control.games.PlayerControlPartition, phx.dynamics.TimeGrid
]:
    dtype = jnp.float64
    player_count = len(control_sizes)
    control_size = sum(control_sizes)
    steps = jnp.arange(horizon, dtype=dtype)
    states = jnp.arange(state_size, dtype=dtype)
    controls = jnp.arange(control_size, dtype=dtype)
    noises = jnp.arange(noise_size, dtype=dtype)
    identity = jnp.eye(state_size, dtype=dtype)

    dynamics_base = 0.80 * identity + 0.003 * jnp.cos(
        (states[:, None] + 1.0) * (states[None, :] + 1.0)
    )
    dynamics = jnp.stack(
        [dynamics_base + 0.002 * jnp.sin(step + 1.0) * identity for step in steps]
    )
    control_base = 0.025 * jnp.sin((states[:, None] + 1.0) * (controls[None, :] + 1.0))
    control = jnp.stack([control_base * (1.0 + 0.004 * step) for step in steps])
    state_costs = jnp.stack(
        [
            (0.35 + 0.05 * player + 0.002 * steps)[:, None, None] * identity
            for player in range(player_count)
        ]
    )
    control_costs = jnp.stack(
        [
            jnp.broadcast_to(
                (1.5 + 0.1 * player) * jnp.eye(control_size, dtype=dtype),
                (horizon, control_size, control_size),
            )
            for player in range(player_count)
        ]
    )
    terminal_state_costs = jnp.stack(
        [(0.9 + 0.1 * player) * identity for player in range(player_count)]
    )
    noise_factor_base = 0.035 * jnp.cos((states[:, None] + 1.0) * (noises[None, :] + 1.0))
    noise_factors = jnp.stack(
        [noise_factor_base * (1.0 + 0.008 * step) for step in steps]
    )
    covariance_diagonal = 0.12 + 0.008 * noises[None, :] + 0.001 * steps[:, None]
    noise_covariances = jax.vmap(jnp.diag)(covariance_diagonal)
    initial_mean = 0.04 * jnp.cos(states + 1.0)
    initial_covariance = 0.15 * identity

    arguments = tuple(
        _broadcast(value, case_shape)
        for value in (
            dynamics,
            control,
            state_costs,
            control_costs,
            terminal_state_costs,
            noise_factors,
            noise_covariances,
            initial_mean,
            initial_covariance,
        )
    )
    partition = phx.control.games.PlayerControlPartition(
        tuple(f"player-{index}" for index in range(player_count)),
        control_sizes,
    )
    time_grid = phx.dynamics.TimeGrid(
        jnp.arange(horizon + 1, dtype=dtype),
        time_id=(
            f"benchmark-lqg-nash:T{horizon}:n{state_size}:m{control_size}:"
            f"p{player_count}:w{noise_size}:cases"
            f"{'x'.join(map(str, case_shape)) or '1'}"
        ),
    )
    return arguments, partition, time_grid


def _solve_function(partition, time_grid):
    def solve(arguments):
        (
            dynamics,
            controls,
            state_costs,
            control_costs,
            terminal_state_costs,
            noise_factors,
            noise_covariances,
            initial_mean,
            initial_covariance,
        ) = arguments
        return phx.control.games.finite_horizon_lqg_feedback_nash(
            dynamics,
            controls,
            state_costs,
            control_costs,
            terminal_state_costs,
            partition,
            process_noise_factors=noise_factors,
            process_noise_covariances=noise_covariances,
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            time_grid=time_grid,
            policy_id=f"{time_grid.time_id}:policy",
        )

    return eqx.filter_jit(solve)


def _implied_process_covariances(arguments: BenchmarkArguments) -> jax.Array:
    noise_factors = arguments[5]
    noise_covariances = arguments[6]
    return ein.contract(
        "...tia,...tab,...tjb->...tij",
        noise_factors,
        noise_covariances,
        noise_factors,
    )


def _cubature_expected_cost(
    result,
    arguments: BenchmarkArguments,
) -> tuple[jax.Array, int]:
    (
        dynamics,
        controls,
        state_costs,
        control_costs,
        terminal_state_costs,
        noise_factors,
        noise_covariances,
        initial_mean,
        initial_covariance,
    ) = arguments
    horizon = dynamics.shape[-3]
    state_size = dynamics.shape[-1]
    noise_size = noise_covariances.shape[-1]
    latent_size = state_size + horizon * noise_size
    standard_directions = jnp.concatenate(
        (jnp.eye(latent_size), -jnp.eye(latent_size)),
        axis=0,
    ).astype(dynamics.dtype)
    standard_directions = (
        jnp.sqrt(jnp.asarray(latent_size, dtype=dynamics.dtype)) * standard_directions
    )

    initial_factor = phx.uq.gaussian_factor_from_covariance(
        initial_covariance,
        factor_id="benchmark-lqg-nash-cubature-initial",
    ).factor
    driving_factors = phx.uq.gaussian_factor_from_covariance(
        noise_covariances,
        factor_id="benchmark-lqg-nash-cubature-driving",
    ).factor
    state = initial_mean[..., None, :] + ein.contract(
        "...ir,qr->...qi",
        initial_factor,
        standard_directions[:, :state_size],
    )
    cost = jnp.zeros(
        state.shape[:-2] + (state_costs.shape[-4], state.shape[-2]),
        dtype=state.dtype,
    )
    for index in range(horizon):
        control = (
            ein.contract(
                "...ij,...qj->...qi",
                result.feedback_gain[..., index, :, :],
                state,
            )
            + result.feedforward[..., index, None, :]
        )
        cost = cost + 0.5 * (
            ein.contract(
                "...qi,...pij,...qj->...pq",
                state,
                state_costs[..., :, index, :, :],
                state,
            )
            + ein.contract(
                "...qi,...pij,...qj->...pq",
                control,
                control_costs[..., :, index, :, :],
                control,
            )
        )
        start = state_size + index * noise_size
        stop = start + noise_size
        driving_noise = ein.contract(
            "...ir,qr->...qi",
            driving_factors[..., index, :, :],
            standard_directions[:, start:stop],
        )
        state = (
            ein.contract(
                "...ij,...qj->...qi",
                dynamics[..., index, :, :],
                state,
            )
            + ein.contract(
                "...ij,...qj->...qi",
                controls[..., index, :, :],
                control,
            )
            + ein.contract(
                "...ij,...qj->...qi",
                noise_factors[..., index, :, :],
                driving_noise,
            )
        )
    cost = cost + 0.5 * ein.contract(
        "...qi,...pij,...qj->...pq",
        state,
        terminal_state_costs,
        state,
    )
    return jnp.mean(cost, axis=-1), 2 * latent_size


def _expected_cost_oracle(
    result,
    arguments: BenchmarkArguments,
    /,
    *,
    available: bool,
) -> dict[str, Any]:
    if not available:
        return {
            "available": False,
            "method": None,
            "point_count": None,
            "expected_costs": None,
            "maximum_defect": None,
            "unavailable_reason": (
                "cubature is reserved for the designated horizon-4 case"
            ),
        }
    expected_costs, point_count = _cubature_expected_cost(result, arguments)
    return {
        "available": True,
        "method": "degree-2-spherical-radial-cubature-rollout",
        "point_count": point_count,
        "expected_costs": jnp.asarray(expected_costs).tolist(),
        "maximum_defect": float(
            jnp.max(jnp.abs(result.initial_expected_cost - expected_costs))
        ),
        "unavailable_reason": None,
    }


def _certificates(
    result,
    arguments: BenchmarkArguments,
    case_shape: tuple[int, ...],
    /,
    *,
    expected_cost_oracle_available: bool,
) -> dict[str, Any]:
    player_axis = len(case_shape)
    deterministic_values = result.deterministic_result.values
    matrices = jnp.stack(
        tuple(value.matrices for value in deterministic_values),
        axis=player_axis,
    )
    linear = jnp.stack(
        tuple(value.linear for value in deterministic_values),
        axis=player_axis,
    )
    deterministic_constants = jnp.stack(
        tuple(value.constants for value in deterministic_values),
        axis=player_axis,
    )
    corrected_constants = jnp.stack(
        tuple(value.constants for value in result.values),
        axis=player_axis,
    )
    expected_process_covariances = _implied_process_covariances(arguments)
    expected_trace = 0.5 * ein.contract(
        "...ptij,...tji->...pt",
        matrices[..., 1:, :, :],
        expected_process_covariances,
    )
    reverse_trace = jnp.flip(
        jnp.cumsum(jnp.flip(expected_trace, axis=-1), axis=-1),
        axis=-1,
    )
    expected_corrections = jnp.concatenate(
        (
            reverse_trace,
            jnp.zeros(reverse_trace.shape[:-1] + (1,), dtype=reverse_trace.dtype),
        ),
        axis=-1,
    )
    expected_constants = deterministic_constants + expected_corrections
    initial_mean = arguments[7]
    initial_covariance = arguments[8]
    initial_matrix = matrices[..., 0, :, :]
    initial_linear = linear[..., 0, :]
    expected_initial_cost = (
        0.5
        * ein.contract(
            "...i,...pij,...j->...p",
            initial_mean,
            initial_matrix,
            initial_mean,
        )
        + ein.contract("...pi,...i->...p", initial_linear, initial_mean)
        + expected_constants[..., 0]
        + 0.5
        * ein.contract(
            "...pij,...ji->...p",
            initial_matrix,
            initial_covariance,
        )
    )
    process_covariance_defect = float(
        jnp.max(jnp.abs(result.process_covariances - expected_process_covariances))
    )
    trace_identity_defect = float(
        jnp.max(jnp.abs(result.trace_increments - expected_trace))
    )
    reverse_accumulation_defect = float(
        jnp.max(jnp.abs(result.value_constant_corrections - expected_corrections))
    )
    value_constant_defect = float(
        jnp.max(jnp.abs(corrected_constants - expected_constants))
    )
    initial_expected_cost_defect = float(
        jnp.max(jnp.abs(result.initial_expected_cost - expected_initial_cost))
    )
    expected_cost_oracle = _expected_cost_oracle(
        result,
        arguments,
        available=expected_cost_oracle_available,
    )
    certificate_tolerance = 1.0e-10
    independent_defects = (
        process_covariance_defect,
        trace_identity_defect,
        reverse_accumulation_defect,
        value_constant_defect,
        initial_expected_cost_defect,
    )
    independent_valid = all(
        defect <= certificate_tolerance for defect in independent_defects
    ) and (
        not expected_cost_oracle["available"]
        or expected_cost_oracle["maximum_defect"] <= certificate_tolerance
    )
    diagnostics = result.deterministic_result.diagnostics
    return {
        "claim": "exact-additive-zero-mean-full-state-lqg-feedback-nash-for-declared-arrays",
        "method": result.method,
        "all_valid": bool(jnp.all(result.valid)) and independent_valid,
        "certificate_tolerance": certificate_tolerance,
        "status": jnp.asarray(result.status).tolist(),
        "maximum_process_covariance_defect": process_covariance_defect,
        "maximum_trace_identity_defect": trace_identity_defect,
        "maximum_reverse_accumulation_defect": reverse_accumulation_defect,
        "maximum_value_constant_defect": value_constant_defect,
        "maximum_initial_expected_cost_defect": initial_expected_cost_defect,
        "expected_cost_oracle": expected_cost_oracle,
        "maximum_covariance_symmetry_residual": float(
            jnp.max(result.covariance_symmetry_residuals)
        ),
        "minimum_covariance_eigenvalue": float(
            jnp.min(result.covariance_minimum_eigenvalues)
        ),
        "maximum_stationarity_residual": float(
            jnp.max(diagnostics.maximum_stationarity_residual)
        ),
        "maximum_bellman_residual": float(jnp.max(diagnostics.maximum_bellman_residual)),
        "minimum_coupled_rank": int(jnp.min(diagnostics.coupled_ranks)),
        "maximum_coupled_condition_number": float(
            jnp.max(diagnostics.maximum_coupled_condition_number)
        ),
        "output_fingerprint": array_tree_fingerprint(result),
        "not_claimed": [
            "state-dependent-noise feedback Nash",
            "action-dependent-noise feedback Nash",
            "nonlinear-game equilibrium",
        ],
    }


def _case(
    name: str,
    horizon: int,
    state_size: int,
    control_sizes: tuple[int, ...],
    noise_size: int,
    case_shape: tuple[int, ...],
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    inputs, partition, time_grid = _problem(
        horizon, state_size, control_sizes, noise_size, case_shape
    )
    function = _solve_function(partition, time_grid)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(inputs),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(inputs),
        warmup=warmup,
        repeats=repeats,
    )
    compiler = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return {
        "name": name,
        "dimensions": {
            "horizon": horizon,
            "state_size": state_size,
            "control_sizes": list(control_sizes),
            "joint_control_size": sum(control_sizes),
            "player_count": len(control_sizes),
            "noise_size": noise_size,
            "case_shape": list(case_shape),
        },
        "dtype": str(inputs[0].dtype),
        "input_fingerprint": array_tree_fingerprint(inputs),
        "lower": {
            "seconds": compilation.lowering_seconds,
            "scope": "eqx-filter-jit host lowering",
        },
        "compile": {
            "seconds": compilation.compilation_seconds,
            "scope": "lowered JAX executable compilation",
        },
        "run": {
            **execution.to_milliseconds_dict(),
            "scope": "synchronized compiled feedback-Nash solve",
        },
        "memory": {
            "logical_input_bytes": logical_array_bytes(inputs),
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
            "riccati_stages": horizon,
            "batched_cases": math.prod(case_shape) if case_shape else 1,
            "players": len(control_sizes),
            "player_trace_contractions": horizon * len(control_sizes),
        },
        "certificate": _certificates(
            result,
            inputs,
            case_shape,
            expected_cost_oracle_available=name == "horizon-4",
        ),
    }


def _specifications():
    return (
        ("baseline", 16, 8, (2, 2), 3, ()),
        ("horizon-4", 4, 8, (2, 2), 3, ()),
        ("horizon-64", 64, 8, (2, 2), 3, ()),
        ("state-4", 16, 4, (2, 2), 3, ()),
        ("state-16", 16, 16, (2, 2), 3, ()),
        ("controls-1-1", 16, 8, (1, 1), 3, ()),
        ("controls-4-4", 16, 8, (4, 4), 3, ()),
        ("players-1", 16, 8, (4,), 3, ()),
        ("players-4", 16, 8, (1, 1, 1, 1), 3, ()),
        ("noise-1", 16, 8, (2, 2), 1, ()),
        ("noise-6", 16, 8, (2, 2), 6, ()),
        ("cases-8", 16, 8, (2, 2), 3, (8,)),
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
            horizon,
            state_size,
            control_sizes,
            noise_size,
            case_shape,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for name, horizon, state_size, control_sizes, noise_size, case_shape in _specifications()
    ]
    payload = {
        "benchmark": "control-lqg-feedback-nash",
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_valid": all(case["certificate"]["all_valid"] for case in cases),
    }
    if arguments.output is None:
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    else:
        write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
