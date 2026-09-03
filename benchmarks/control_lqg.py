#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark exact additive-noise finite-horizon LQG state feedback."""

from __future__ import annotations

import argparse
import json
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
    control_size: int,
    noise_size: int,
    case_shape: tuple[int, ...],
    /,
) -> tuple[BenchmarkArguments, phx.dynamics.TimeGrid]:
    dtype = jnp.float64
    steps = jnp.arange(horizon, dtype=dtype)
    states = jnp.arange(state_size, dtype=dtype)
    controls = jnp.arange(control_size, dtype=dtype)
    noises = jnp.arange(noise_size, dtype=dtype)
    identity = jnp.eye(state_size, dtype=dtype)

    dynamics_base = 0.82 * identity + 0.004 * jnp.sin(
        (states[:, None] + 1.0) * (states[None, :] + 1.0)
    )
    dynamics = jnp.stack(
        [dynamics_base + 0.002 * jnp.cos(step + 1.0) * identity for step in steps]
    )
    control_base = 0.03 * jnp.cos((states[:, None] + 1.0) * (controls[None, :] + 1.0))
    control = jnp.stack([control_base * (1.0 + 0.005 * step) for step in steps])
    state_cost = (0.5 + 0.005 * steps)[:, None, None] * identity
    control_cost = jnp.broadcast_to(
        1.5 * jnp.eye(control_size, dtype=dtype),
        (horizon, control_size, control_size),
    )
    terminal_state_cost = 1.25 * identity
    noise_factor_base = 0.04 * jnp.sin((states[:, None] + 1.0) * (noises[None, :] + 1.0))
    noise_factors = jnp.stack([noise_factor_base * (1.0 + 0.01 * step) for step in steps])
    covariance_diagonal = 0.15 + 0.01 * noises[None, :] + 0.002 * steps[:, None]
    noise_covariances = jax.vmap(jnp.diag)(covariance_diagonal)
    initial_mean = 0.05 * jnp.sin(states + 1.0)
    initial_covariance = 0.2 * identity

    arguments = tuple(
        _broadcast(value, case_shape)
        for value in (
            dynamics,
            control,
            state_cost,
            control_cost,
            terminal_state_cost,
            noise_factors,
            noise_covariances,
            initial_mean,
            initial_covariance,
        )
    )
    time_grid = phx.dynamics.TimeGrid(
        jnp.arange(horizon + 1, dtype=dtype),
        time_id=(
            f"benchmark-lqg:T{horizon}:n{state_size}:m{control_size}:"
            f"w{noise_size}:cases{'x'.join(map(str, case_shape)) or '1'}"
        ),
    )
    return arguments, time_grid


def _solve_function(time_grid):
    def solve(arguments):
        (
            dynamics,
            controls,
            state_costs,
            control_costs,
            terminal_state_cost,
            noise_factors,
            noise_covariances,
            initial_mean,
            initial_covariance,
        ) = arguments
        return phx.control.stochastic.finite_horizon_lqg_state_feedback(
            dynamics,
            controls,
            state_costs,
            control_costs,
            terminal_state_cost,
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
        terminal_state_cost,
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
        factor_id="benchmark-lqg-cubature-initial",
    ).factor
    driving_factors = phx.uq.gaussian_factor_from_covariance(
        noise_covariances,
        factor_id="benchmark-lqg-cubature-driving",
    ).factor
    state = initial_mean[..., None, :] + ein.contract(
        "...ir,qr->...qi",
        initial_factor,
        standard_directions[:, :state_size],
    )
    cost = jnp.zeros(state.shape[:-1], dtype=state.dtype)
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
                "...qi,...ij,...qj->...q",
                state,
                state_costs[..., index, :, :],
                state,
            )
            + ein.contract(
                "...qi,...ij,...qj->...q",
                control,
                control_costs[..., index, :, :],
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
        "...qi,...ij,...qj->...q",
        state,
        terminal_state_cost,
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
            "expected_cost": None,
            "maximum_defect": None,
            "unavailable_reason": (
                "cubature is reserved for the designated horizon-4 case"
            ),
        }
    expected_cost, point_count = _cubature_expected_cost(result, arguments)
    return {
        "available": True,
        "method": "degree-2-spherical-radial-cubature-rollout",
        "point_count": point_count,
        "expected_cost": jnp.asarray(expected_cost).tolist(),
        "maximum_defect": float(
            jnp.max(jnp.abs(result.initial_expected_cost - expected_cost))
        ),
        "unavailable_reason": None,
    }


def _certificates(
    result,
    arguments: BenchmarkArguments,
    /,
    *,
    expected_cost_oracle_available: bool,
) -> dict[str, Any]:
    expected_process_covariances = _implied_process_covariances(arguments)
    expected_trace = 0.5 * ein.contract(
        "...tij,...tji->...t",
        result.deterministic_result.value.matrices[..., 1:, :, :],
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
    deterministic_value = result.deterministic_result.value
    expected_constants = deterministic_value.constants + expected_corrections
    initial_mean = arguments[7]
    initial_covariance = arguments[8]
    matrix = deterministic_value.matrices[..., 0, :, :]
    linear = deterministic_value.linear[..., 0, :]
    expected_initial_cost = (
        0.5
        * ein.contract(
            "...i,...ij,...j->...",
            initial_mean,
            matrix,
            initial_mean,
        )
        + ein.contract("...i,...i->...", linear, initial_mean)
        + expected_constants[..., 0]
        + 0.5 * ein.contract("...ij,...ji->...", matrix, initial_covariance)
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
        jnp.max(jnp.abs(result.value.constants - expected_constants))
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
    return {
        "claim": "exact-additive-zero-mean-lqg-state-feedback-for-declared-arrays",
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
        "output_fingerprint": array_tree_fingerprint(result),
        "not_claimed": [
            "state-dependent-noise optimality",
            "action-dependent-noise optimality",
            "non-Gaussian distributional guarantees beyond second-moment value identities",
        ],
    }


def _case(
    name: str,
    horizon: int,
    state_size: int,
    control_size: int,
    noise_size: int,
    case_shape: tuple[int, ...],
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    inputs, time_grid = _problem(
        horizon, state_size, control_size, noise_size, case_shape
    )
    function = _solve_function(time_grid)
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
            "control_size": control_size,
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
            "scope": "synchronized compiled solve",
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
            "batched_cases": int(jnp.prod(jnp.asarray(case_shape))) if case_shape else 1,
            "trace_contractions": horizon,
        },
        "certificate": _certificates(
            result,
            inputs,
            expected_cost_oracle_available=name == "horizon-4",
        ),
    }


def _specifications():
    return (
        ("baseline", 16, 8, 4, 3, ()),
        ("horizon-4", 4, 8, 4, 3, ()),
        ("horizon-64", 64, 8, 4, 3, ()),
        ("state-4", 16, 4, 4, 3, ()),
        ("state-16", 16, 16, 4, 3, ()),
        ("control-2", 16, 8, 2, 3, ()),
        ("control-8", 16, 8, 8, 3, ()),
        ("noise-1", 16, 8, 4, 1, ()),
        ("noise-8", 16, 8, 4, 8, ()),
        ("cases-8", 16, 8, 4, 3, (8,)),
        ("cases-32", 16, 8, 4, 3, (32,)),
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
            control_size,
            noise_size,
            case_shape,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for name, horizon, state_size, control_size, noise_size, case_shape in _specifications()
    ]
    payload = {
        "benchmark": "control-lqg-state-feedback",
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
