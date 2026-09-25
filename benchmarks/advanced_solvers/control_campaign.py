#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    DurationDistribution,
    measure_repeated,
    measure_synchronized,
)


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
    warmup_timing = DurationDistribution(()).to_milliseconds_dict()
    if warmup:
        _, warmup_distribution = measure_repeated(
            operation,
            warmup=0,
            repeats=warmup,
        )
        warmup_timing = warmup_distribution.to_milliseconds_dict()
    result, steady_distribution = measure_repeated(
        operation,
        warmup=0,
        repeats=repeats,
    )
    return result, {
        "warmup": warmup_timing,
        "steady": steady_distribution.to_milliseconds_dict(),
    }


def _certificate(problem, result, qp_results, /):
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
    backend_successful = bool(np.asarray(result.successful))
    objective = float(np.asarray(result.objective))
    tolerance = 1e-7
    certified = (
        backend_successful
        and np.isfinite(objective)
        and dynamics_residual <= tolerance
        and max(lower_violation, upper_violation) <= tolerance
    )
    return {
        "kind": "independent-trajectory-feasibility",
        "independently_computed": True,
        "successful": certified,
        "backend_successful": backend_successful,
        "status": int(np.asarray(result.status)),
        "objective": objective,
        "dynamics_residual": dynamics_residual,
        "bound_violation": max(lower_violation, upper_violation),
        "tolerance": tolerance,
        "provider_maximum_kkt_residual": float(
            max(np.asarray(item.kkt_residual_norm) for item in qp_results)
        ),
        "iterations": [int(np.asarray(item.iterations)) for item in qp_results],
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

    values = tuple(horizons)
    if not values or any(value < 1 for value in values):
        raise ValueError("horizons must contain positive integers.")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive.")
    if len(set(values)) != len(values):
        raise ValueError("horizons must not contain duplicates.")
    rows = []
    for horizon in values:
        row_seed = seed + horizon
        phase_timings: dict[str, Any] = {}
        problem, sample = measure_synchronized(lambda: _problem(horizon, row_seed))
        phase_timings["setup"] = DurationDistribution((sample,)).to_milliseconds_dict()
        prediction = min(16, horizon)
        policy = phx.optim.ConvexSolvePolicy(
            phx.optim.DensePrimalDualQP(max_kkt_dimension=max(512, 8 * horizon)),
            termination=phx.optim.ConvexTermination(
                absolute=1e-7,
                maximum_steps=100,
            ),
        )
        dense_compilation, dense_compile_sample = measure_synchronized(
            lambda: phx.control.compile_linear_quadratic_control(problem)
        )
        phase_timings["dense_compilation"] = DurationDistribution(
            (dense_compile_sample,)
        ).to_milliseconds_dict()
        sparse_compilation, sparse_compile_sample = measure_synchronized(
            lambda: phx.control.compile_linear_quadratic_control(
                problem,
                compilation_policy=phx.control.LinearControlCompilationPolicy("sparse"),
            )
        )
        phase_timings["sparse_compilation"] = DurationDistribution(
            (sparse_compile_sample,)
        ).to_milliseconds_dict()
        sparse_prepared, sparse_prepare_sample = measure_synchronized(
            lambda: phx.control.prepare_linear_quadratic_control(
                problem,
                compilation_policy=phx.control.LinearControlCompilationPolicy("sparse"),
            )
        )
        phase_timings["sparse_preparation"] = DurationDistribution(
            (sparse_prepare_sample,)
        ).to_milliseconds_dict()
        sparse_operation = lambda: phx.control.solve_prepared_linear_quadratic_control(
            sparse_prepared
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
        sparse_solution, sparse_timing = _measure(sparse_operation, warmup, repeats)
        cold_certificate, cold_verification = measure_synchronized(
            lambda: _certificate(problem, cold, cold.qp_results)
        )
        warm_certificate, warm_verification = measure_synchronized(
            lambda: _certificate(problem, warm, warm.qp_results)
        )
        sparse_certificate, sparse_verification = measure_synchronized(
            lambda: _certificate(problem, sparse_solution, (sparse_solution.qp_result,))
        )
        phase_timings["verification"] = DurationDistribution(
            (cold_verification, warm_verification, sparse_verification)
        ).to_milliseconds_dict()
        dense_program = dense_compilation.program
        sparse_program = sparse_compilation.program
        sparse_quadratic = sparse_program.quadratic
        sparse_constraints = sparse_program.constraint_matrix
        if not isinstance(
            sparse_quadratic, phx.linalg.AbstractSparseLinearOperator
        ) or not isinstance(
            sparse_constraints,
            phx.linalg.AbstractSparseLinearOperator,
        ):
            raise RuntimeError("Sparse control compilation did not produce operators.")
        rows.append(
            {
                "horizon": horizon,
                "seed": row_seed,
                "problem_fingerprint": _problem_fingerprint(problem, row_seed),
                "phase_timings": phase_timings,
                "prediction_horizon": prediction,
                "dense_matrix_bytes": int(
                    dense_program.quadratic.nbytes
                    + dense_program.equality_matrix.nbytes
                    + dense_program.inequality_matrix.nbytes
                ),
                "sparse_value_bytes": int(
                    sparse_quadratic.sparse_storage().values.nbytes
                    + sparse_constraints.sparse_storage().values.nbytes
                ),
                "sparse": {
                    "timing": sparse_timing,
                    "certificate": sparse_certificate,
                },
                "cold": {
                    "timing": cold_timing,
                    "certificate": cold_certificate,
                },
                "warm": {
                    "timing": warm_timing,
                    "certificate": warm_certificate,
                },
            }
        )
    passed = all(
        path["certificate"]["successful"]
        for row in rows
        for path in (row["sparse"], row["cold"], row["warm"])
    )
    return {
        "campaign": "control-horizon-warm-start",
        "seed": seed,
        "warmup": warmup,
        "repeats": repeats,
        "environment": capture_environment().to_dict(),
        "source_fingerprint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": passed,
        "rows": rows,
    }


def _problem_fingerprint(problem: Any, seed: int, /) -> str:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(str(problem.problem_id).encode("utf-8"))
    for value in (
        problem.dynamics_matrices,
        problem.control_matrices,
        problem.dynamics_bias,
        problem.control_lower_bounds,
        problem.control_upper_bounds,
    ):
        array = np.asarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = ["run_control_horizon_campaign"]
