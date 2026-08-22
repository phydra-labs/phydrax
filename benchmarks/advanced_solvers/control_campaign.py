#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _problem(horizon: int, seed: int, /):
    generator = np.random.Generator(np.random.PCG64(seed))
    dynamics = 0.95 + 0.02 * generator.standard_normal((horizon, 1, 1))
    controls = np.ones((horizon, 1, 1))
    return phx.control.LinearQuadraticControlProblem(
        jnp.asarray(dynamics),
        jnp.asarray(controls),
        jnp.asarray([1.0]),
        jnp.ones((horizon, 1, 1)),
        0.1 * jnp.ones((horizon, 1, 1)),
        jnp.ones((1, 1)),
        control_lower_bounds=-jnp.ones((horizon, 1)),
        control_upper_bounds=jnp.ones((horizon, 1)),
        problem_id=f"benchmark-mpc:{horizon}:{seed}",
    )


def _measure(operation, warmup: int, repeats: int, /):
    for _ in range(warmup):
        jax.block_until_ready(operation())
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        jax.block_until_ready(result)
        samples.append(1_000.0 * (time.perf_counter() - started))
    return result, {
        "samples_ms": samples,
        "median_ms": float(np.median(samples)),
        "minimum_ms": float(np.min(samples)),
        "maximum_ms": float(np.max(samples)),
    }


def _certificate(problem, result, /):
    states = np.asarray(result.states)
    controls = np.asarray(result.controls)
    predicted = (
        np.asarray(problem.dynamics_matrices) @ states[:-1, :, None]
        + np.asarray(problem.control_matrices) @ controls[:, :, None]
        + np.asarray(problem.dynamics_bias)[..., None]
    )[..., 0]
    dynamics_residual = float(np.max(np.abs(states[1:] - predicted), initial=0.0))
    lower_violation = float(
        np.max(
            np.maximum(np.asarray(problem.control_lower_bounds) - controls, 0.0),
            initial=0.0,
        )
    )
    upper_violation = float(
        np.max(
            np.maximum(controls - np.asarray(problem.control_upper_bounds), 0.0),
            initial=0.0,
        )
    )
    return {
        "successful": bool(np.asarray(result.successful)),
        "status": int(np.asarray(result.status)),
        "objective": float(np.asarray(result.objective)),
        "dynamics_residual": dynamics_residual,
        "bound_violation": max(lower_violation, upper_violation),
        "maximum_kkt_residual": float(
            max(np.asarray(item.kkt_residual_norm) for item in result.qp_results)
        ),
        "iterations": [int(np.asarray(item.iterations)) for item in result.qp_results],
    }


def run_control_horizon_campaign(
    horizons: Sequence[int] = (8, 32, 128),
    /,
    *,
    seed: int = 20260816,
    warmup: int = 1,
    repeats: int = 5,
) -> dict[str, Any]:
    """Compare cold and explicitly shifted warm MPC across declared horizons."""

    values = tuple(int(value) for value in horizons)
    if not values or any(value < 1 for value in values):
        raise ValueError("horizons must contain positive integers.")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive.")
    rows = []
    for index, horizon in enumerate(values):
        problem = _problem(horizon, seed + index)
        prediction = min(16, horizon)
        policy = phx.optim.ConvexSolvePolicy(
            phx.optim.DensePrimalDualQP(max_kkt_dimension=max(512, 8 * horizon)),
            termination=phx.optim.ConvexTermination(
                absolute=1e-7,
                maximum_steps=100,
            ),
        )
        dense_compilation = phx.control.compile_linear_quadratic_control(problem)
        sparse_compilation = phx.control.compile_linear_quadratic_control(
            problem,
            compilation_policy=phx.control.LinearControlCompilationPolicy("sparse"),
        )
        cold_operation = lambda: phx.control.solve_receding_horizon_mpc(
            problem,
            prediction_horizon=prediction,
            terminal_policy="none",
            policy=policy,
        )
        warm_operation = lambda: phx.control.solve_receding_horizon_mpc(
            problem,
            prediction_horizon=prediction,
            terminal_policy="none",
            policy=policy,
            warm_start_policy=phx.control.MPCWarmStartPolicy(),
        )
        cold, cold_timing = _measure(cold_operation, warmup, repeats)
        warm, warm_timing = _measure(warm_operation, warmup, repeats)
        sparse_quadratic = sparse_compilation.sparse_quadratic
        sparse_equality = sparse_compilation.sparse_equality
        sparse_inequality = sparse_compilation.sparse_inequality
        if (
            sparse_quadratic is None
            or sparse_equality is None
            or sparse_inequality is None
        ):
            raise RuntimeError("Sparse control compilation did not produce operators.")
        rows.append(
            {
                "horizon": horizon,
                "prediction_horizon": prediction,
                "dense_matrix_bytes": int(
                    dense_compilation.qp.quadratic.nbytes
                    + dense_compilation.qp.equality_matrix.nbytes
                    + dense_compilation.qp.inequality_matrix.nbytes
                ),
                "sparse_value_bytes": int(
                    sparse_quadratic.coefficients.nbytes
                    + sparse_equality.coefficients.nbytes
                    + sparse_inequality.coefficients.nbytes
                ),
                "cold": {
                    "timing": cold_timing,
                    "certificate": _certificate(problem, cold),
                },
                "warm": {
                    "timing": warm_timing,
                    "certificate": _certificate(problem, warm),
                },
            }
        )
    return {
        "schema_version": 1,
        "campaign": "control-horizon-warm-start",
        "seed": seed,
        "warmup": warmup,
        "repeats": repeats,
        "rows": rows,
    }


__all__ = ["run_control_horizon_campaign"]
