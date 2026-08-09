#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _block(tree: Any, /) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(
    operation: Callable[[], Any],
    /,
    *,
    repeats: int,
) -> tuple[Any, float, float]:
    result = operation()
    _block(result)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        _block(result)
        samples.append(1e3 * (time.perf_counter() - started))
    return result, float(np.mean(samples)), float(np.std(samples))


def _array_bytes(tree: Any, /) -> int:
    return sum(
        int(leaf.nbytes)
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _lorenz_derivative(states: jax.Array, /) -> jax.Array:
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    return jnp.stack(
        (
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ),
        axis=-1,
    )


def _lorenz_coefficients(feature_names: tuple[str, ...], /) -> jax.Array:
    index = {name: position for position, name in enumerate(feature_names)}
    coefficients = jnp.zeros((3, len(feature_names)))
    coefficients = coefficients.at[0, index["state:x"]].set(-10.0)
    coefficients = coefficients.at[0, index["state:y"]].set(10.0)
    coefficients = coefficients.at[1, index["state:x"]].set(28.0)
    coefficients = coefficients.at[1, index["state:y"]].set(-1.0)
    coefficients = coefficients.at[1, index["state:x * state:z"]].set(-1.0)
    coefficients = coefficients.at[2, index["state:z"]].set(-8.0 / 3.0)
    coefficients = coefficients.at[2, index["state:x * state:y"]].set(1.0)
    return coefficients


def _sparse_recovery_benchmark(
    *,
    samples: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    states = jnp.asarray(
        generator.uniform(
            low=np.asarray((-20.0, -30.0, 0.0)),
            high=np.asarray((20.0, 30.0, 50.0)),
            size=(samples, 3),
        )
    )
    layout = phx.dynamics.StateLayout(
        (3,),
        component_names=("x", "y", "z"),
    )
    data = phx.dynamics.TrajectoryData(
        jnp.arange(samples, dtype=states.dtype),
        states,
        state_layout=layout,
        derivatives=_lorenz_derivative(states),
        coordinate_id="sample",
        source_id="benchmark:lorenz-exact",
    )
    library = phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=2)
    problem = phx.dynamics.identification.SINDyProblem(
        data=data,
        library=library,
        formulation=phx.dynamics.identification.StrongSINDyFormulation(),
    )
    regressor = phx.dynamics.identification.SequentialThresholdedLeastSquares(
        1e-8,
        scale_features=True,
        threshold_space="physical",
        unbiased_refit=True,
    )
    result, mean_ms, standard_deviation_ms = _measure(
        lambda: phx.dynamics.identification.fit_sindy(problem, regressor),
        repeats=repeats,
    )
    expected = _lorenz_coefficients(result.design.feature_names)
    expected_support = expected != 0.0
    recovered_support = result.support
    true_positive = int(jnp.sum(expected_support & recovered_support))
    false_positive = int(jnp.sum(~expected_support & recovered_support))
    false_negative = int(jnp.sum(expected_support & ~recovered_support))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    coefficient_error = float(jnp.max(jnp.abs(result.coefficients - expected)))
    target = result.design.target
    prediction = result.predict_design()
    relative_residual = float(
        jnp.linalg.norm(prediction - target) / jnp.maximum(jnp.linalg.norm(target), 1.0)
    )
    passed = bool(result.valid) and precision == 1.0 and recall == 1.0
    passed = passed and coefficient_error <= 1e-8 and relative_residual <= 1e-10
    return {
        "problem_id": "lorenz-exact-strong-sindy",
        "samples": samples,
        "features": result.design.num_features,
        "outputs": result.design.output_size,
        "mean_ms": mean_ms,
        "standard_deviation_ms": standard_deviation_ms,
        "working_set_bytes": _array_bytes((data, result)),
        "valid": bool(result.valid),
        "status": np.asarray(result.status).tolist(),
        "support_size": int(jnp.sum(recovered_support)),
        "support_precision": precision,
        "support_recall": recall,
        "maximum_coefficient_error": coefficient_error,
        "relative_residual": relative_residual,
        "passed": passed,
    }


def _diagonal_map(coordinate, state, rates):
    del coordinate
    return jnp.exp(rates) * state


def _matrix_free_benchmark(
    *,
    dimension: int,
    leading_k: int,
    num_steps: int,
    repeats: int,
) -> dict[str, Any]:
    layout = phx.dynamics.StateLayout((dimension,))
    system = phx.dynamics.DiscreteSystem(
        _diagonal_map,
        state_layout=layout,
        system_id=f"benchmark:diagonal-map:{dimension}",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    grid = phx.dynamics.IterationGrid.from_steps(
        num_steps,
        iteration_id=f"benchmark:diagonal-map:{dimension}:{num_steps}",
    )
    rates = jnp.linspace(0.5, -3.0, leading_k)
    if leading_k < dimension:
        rates = jnp.concatenate((rates, jnp.linspace(-3.5, -4.0, dimension - leading_k)))
    initial_state = jnp.ones((dimension,))
    initial_basis = jnp.eye(dimension, leading_k)
    cadence = max(1, min(4, num_steps))
    burn_in = cadence * max(1, num_steps // (4 * cadence))
    accumulation_interval = cadence * max(1, num_steps // (8 * cadence))
    num_intervals = (num_steps + cadence - 1) // cadence
    backward_discard = max(1, num_intervals // 2)

    spectrum, spectrum_mean_ms, spectrum_standard_deviation_ms = _measure(
        lambda: phx.dynamics.analysis.finite_time_lyapunov_spectrum(
            evolution,
            initial_state,
            grid,
            args=rates,
            leading_k=leading_k,
            qr_interval=cadence,
            burn_in=burn_in,
            accumulation_interval=accumulation_interval,
            initial_basis=initial_basis,
        ),
        repeats=repeats,
    )
    directions, directions_mean_ms, directions_standard_deviation_ms = _measure(
        lambda: phx.dynamics.analysis.covariant_directions(
            evolution,
            initial_state,
            grid,
            args=rates,
            leading_k=leading_k,
            initial_basis=initial_basis,
            memory_mode="store",
            qr_interval=cadence,
            save_every=2,
            backward_discard=backward_discard,
            convergence_tolerance=1e-6,
        ),
        repeats=repeats,
    )
    expected = rates[:leading_k]
    exponent_error = float(jnp.max(jnp.abs(spectrum.exponents - expected)))
    valid_covariance = jnp.where(
        directions.direction_valid[..., None],
        directions.covariance_error,
        jnp.nan,
    )
    maximum_covariance_error = float(jnp.nanmax(valid_covariance))
    valid_drift = jnp.where(
        directions.direction_valid[..., None],
        directions.backward_convergence_drift,
        jnp.nan,
    )
    maximum_convergence_drift = float(jnp.nanmax(valid_drift))
    passed = bool(spectrum.valid) and exponent_error <= 1e-10
    passed = passed and bool(directions.valid) and bool(directions.converged)
    passed = passed and maximum_covariance_error <= 1e-10
    return {
        "problem_id": "high-dimensional-diagonal-map",
        "dimension": dimension,
        "leading_k": leading_k,
        "num_steps": num_steps,
        "dense_jacobian_materialized": False,
        "tangent_method": spectrum.tangent_method,
        "spectrum": {
            "mean_ms": spectrum_mean_ms,
            "standard_deviation_ms": spectrum_standard_deviation_ms,
            "working_set_bytes": _array_bytes(spectrum),
            "valid": bool(spectrum.valid),
            "status": int(spectrum.status),
            "maximum_exponent_error": exponent_error,
        },
        "covariant_directions": {
            "mean_ms": directions_mean_ms,
            "standard_deviation_ms": directions_standard_deviation_ms,
            "working_set_bytes": _array_bytes(directions),
            "valid": bool(directions.valid),
            "converged": bool(directions.converged),
            "maximum_backward_convergence_drift": maximum_convergence_drift,
            "status": int(directions.status),
            "maximum_covariance_error": maximum_covariance_error,
            "stored_frame_count": directions.stored_frame_count,
            "tangent_evaluations": directions.tangent_evaluations,
        },
        "passed": passed,
    }


def run_benchmarks(
    *,
    sparse_samples: int,
    dimension: int,
    leading_k: int,
    num_steps: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Benchmark exact sparse recovery and matrix-free tangent analysis."""
    sparse = _sparse_recovery_benchmark(
        samples=sparse_samples,
        repeats=repeats,
        seed=seed,
    )
    matrix_free = _matrix_free_benchmark(
        dimension=dimension,
        leading_k=leading_k,
        num_steps=num_steps,
        repeats=repeats,
    )
    device = jax.devices()[0]
    return {
        "configuration": {
            "sparse_samples": sparse_samples,
            "dimension": dimension,
            "leading_k": leading_k,
            "num_steps": num_steps,
            "repeats": repeats,
            "seed": seed,
        },
        "environment": {
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "device_kind": device.device_kind,
            "machine": platform.machine(),
            "system": platform.system(),
            "system_release": platform.release(),
            "x64_enabled": bool(jax.config.x64_enabled),
        },
        "sparse_recovery": sparse,
        "matrix_free_analysis": matrix_free,
        "passed": sparse["passed"] and matrix_free["passed"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the Phydrax nonlinear-dynamics substrate."
    )
    parser.add_argument("--sparse-samples", type=int, default=2_048)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--leading-k", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if (
        min(
            args.sparse_samples,
            args.dimension,
            args.leading_k,
            args.num_steps,
            args.repeats,
        )
        <= 0
    ):
        parser.error("benchmark sizes and repeats must be positive")
    if args.leading_k > args.dimension:
        parser.error("leading-k cannot exceed dimension")
    print(
        json.dumps(
            run_benchmarks(
                sparse_samples=args.sparse_samples,
                dimension=args.dimension,
                leading_k=args.leading_k,
                num_steps=args.num_steps,
                repeats=args.repeats,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
