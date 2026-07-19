#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median
from typing import Any, Literal

import jax
import jax.numpy as jnp

import phydrax as phx

from .report import (
    BenchmarkReport,
    collect_environment,
    metric,
    ScenarioResult,
    utc_now_iso,
)


ProfileName = Literal["smoke", "standard"]
_STANDARD_CASES = ((24, 9), (64, 16), (128, 24), (256, 32), (512, 48))
_SMOKE_CASES = ((24, 9), (256, 32))
_MIN_EXACT_FIXED_FACTOR_SPEEDUP = 2.0
_MIN_FITC_FIXED_FACTOR_SPEEDUP = 1.2
_FITC_FIXED_FACTOR_GATE_MIN_OBSERVATIONS = 256
_MAX_FITC_EXACT_MEAN_RMSE = 0.01
_MAX_FITC_EXACT_VARIANCE_RMSE = 0.003


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _timed_call(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = function()
    _block_until_ready(value)
    return value, time.perf_counter() - started


def _jit_timings(
    function: Callable[..., Any],
    *args: Any,
    repetitions: int,
) -> tuple[float, float, float]:
    compiled = jax.jit(function)
    started = time.perf_counter()
    _block_until_ready(compiled(*args))
    cold_seconds = time.perf_counter() - started
    warm_seconds = []
    for _ in range(repetitions):
        started = time.perf_counter()
        _block_until_ready(compiled(*args))
        warm_seconds.append(time.perf_counter() - started)
    execution_seconds = median(warm_seconds)
    return (
        cold_seconds,
        max(0.0, cold_seconds - execution_seconds),
        execution_seconds,
    )


def _case(
    *,
    num_observations: int,
    num_inducing: int,
    repetitions: int,
    seed: int,
) -> ScenarioResult:
    points = jnp.linspace(0.0, 1.0, num_observations)
    query = jnp.linspace(0.0, 1.0, 65)
    observations = (
        0.8 * points
        + 0.25 * jnp.sin(2.0 * jnp.pi * points)
        + 0.01 * jax.random.normal(jax.random.key(seed), points.shape)
    )
    physical_mean = 0.75 * points
    inducing_indices = jnp.round(
        jnp.linspace(0, num_observations - 1, num_inducing)
    ).astype(jnp.int32)
    inducing_points = points[inducing_indices]
    amplitude = jnp.asarray(0.25)
    length_scale = jnp.asarray(0.18)
    noise_scale = jnp.asarray(0.01)

    exact_model = phx.uq.ExactGaussianProcessDiscrepancy(
        points,
        observations,
        kernel="matern32",
    )
    sparse_model = phx.uq.SparseGaussianProcessDiscrepancy(
        points,
        observations,
        inducing_points,
        kernel="matern32",
    )
    exact_factor, exact_factor_build_seconds = _timed_call(
        lambda: exact_model.factor(
            amplitude=amplitude,
            length_scale=length_scale,
            noise_scale=noise_scale,
        )
    )
    sparse_factor, fitc_factor_build_seconds = _timed_call(
        lambda: sparse_model.factor(
            amplitude=amplitude,
            length_scale=length_scale,
            noise_scale=noise_scale,
        )
    )
    exact_conditioner, exact_conditioner_build_seconds = _timed_call(
        lambda: exact_factor.conditioner(query)
    )
    sparse_conditioner, fitc_conditioner_build_seconds = _timed_call(
        lambda: sparse_factor.conditioner(query)
    )

    exact_rebuild = lambda mean: exact_model.log_marginal_likelihood(
        mean,
        amplitude=amplitude,
        length_scale=length_scale,
        noise_scale=noise_scale,
    )
    exact_reuse = lambda mean: exact_factor.log_probability(exact_model.residual(mean))
    fitc_rebuild = lambda mean: sparse_model.log_marginal_likelihood(
        mean,
        amplitude=amplitude,
        length_scale=length_scale,
        noise_scale=noise_scale,
    )
    fitc_reuse = lambda mean: sparse_factor.log_probability(sparse_model.residual(mean))
    exact_rebuild_timing = _jit_timings(
        exact_rebuild, physical_mean, repetitions=repetitions
    )
    exact_reuse_timing = _jit_timings(exact_reuse, physical_mean, repetitions=repetitions)
    fitc_rebuild_timing = _jit_timings(
        fitc_rebuild, physical_mean, repetitions=repetitions
    )
    fitc_reuse_timing = _jit_timings(fitc_reuse, physical_mean, repetitions=repetitions)

    def fixed_exact_objective(parameter):
        return exact_factor.log_probability(exact_model.residual(parameter * points))

    def fixed_fitc_objective(parameter):
        return sparse_factor.log_probability(sparse_model.residual(parameter * points))

    def inferred_exact_objective(unconstrained):
        parameter, log_amplitude, log_length_scale, log_noise_scale = unconstrained
        return exact_model.log_marginal_likelihood(
            parameter * points,
            amplitude=jnp.exp(log_amplitude),
            length_scale=jnp.exp(log_length_scale),
            noise_scale=jnp.exp(log_noise_scale),
        )

    def inferred_fitc_objective(unconstrained):
        parameter, log_amplitude, log_length_scale, log_noise_scale = unconstrained
        return sparse_model.log_marginal_likelihood(
            parameter * points,
            amplitude=jnp.exp(log_amplitude),
            length_scale=jnp.exp(log_length_scale),
            noise_scale=jnp.exp(log_noise_scale),
        )

    fixed_parameter = jnp.asarray(0.75)
    inferred_parameters = jnp.asarray(
        [
            0.75,
            jnp.log(amplitude),
            jnp.log(length_scale),
            jnp.log(noise_scale),
        ]
    )
    exact_fixed_gradient_timing = _jit_timings(
        jax.value_and_grad(fixed_exact_objective),
        fixed_parameter,
        repetitions=repetitions,
    )
    fitc_fixed_gradient_timing = _jit_timings(
        jax.value_and_grad(fixed_fitc_objective),
        fixed_parameter,
        repetitions=repetitions,
    )
    exact_inferred_gradient_timing = _jit_timings(
        jax.value_and_grad(inferred_exact_objective),
        inferred_parameters,
        repetitions=repetitions,
    )
    fitc_inferred_gradient_timing = _jit_timings(
        jax.value_and_grad(inferred_fitc_objective),
        inferred_parameters,
        repetitions=repetitions,
    )

    residual = exact_model.residual(physical_mean)
    exact_condition_apply_timing = _jit_timings(
        lambda value: exact_conditioner.condition(value),
        residual,
        repetitions=repetitions,
    )
    fitc_condition_apply_timing = _jit_timings(
        lambda value: sparse_conditioner.condition(value),
        residual,
        repetitions=repetitions,
    )
    exact_condition = exact_conditioner.condition(residual)
    fitc_condition = sparse_conditioner.condition(residual)
    _block_until_ready((exact_condition, fitc_condition))
    mean_rmse = jnp.sqrt(jnp.mean((fitc_condition.mean - exact_condition.mean) ** 2))
    variance_rmse = jnp.sqrt(
        jnp.mean((fitc_condition.variance - exact_condition.variance) ** 2)
    )
    dtype_bytes = int(observations.dtype.itemsize)
    exact_factor_bytes = dtype_bytes * exact_factor.factor_storage_elements
    fitc_factor_bytes = dtype_bytes * sparse_factor.factor_storage_elements

    metrics = {
        "exact_factor_build_seconds": metric(
            exact_factor_build_seconds, "performance", unit="s"
        ),
        "fitc_factor_build_seconds": metric(
            fitc_factor_build_seconds, "performance", unit="s"
        ),
        "exact_conditioner_build_seconds": metric(
            exact_conditioner_build_seconds, "performance", unit="s"
        ),
        "fitc_conditioner_build_seconds": metric(
            fitc_conditioner_build_seconds, "performance", unit="s"
        ),
        "exact_factor_bytes": metric(exact_factor_bytes, "performance", unit="bytes"),
        "fitc_factor_bytes": metric(fitc_factor_bytes, "performance", unit="bytes"),
        "fitc_exact_storage_ratio": metric(
            fitc_factor_bytes / exact_factor_bytes,
            "performance",
            maximum=0.75,
            description="Reusable FITC factor storage relative to dense Cholesky.",
        ),
        "exact_rebuild_compile_seconds": metric(
            exact_rebuild_timing[1], "performance", unit="s"
        ),
        "exact_rebuild_warm_seconds": metric(
            exact_rebuild_timing[2], "performance", unit="s"
        ),
        "exact_reuse_compile_seconds": metric(
            exact_reuse_timing[1], "performance", unit="s"
        ),
        "exact_reuse_warm_seconds": metric(
            exact_reuse_timing[2], "performance", unit="s"
        ),
        "exact_fixed_factor_speedup": metric(
            exact_rebuild_timing[2] / exact_reuse_timing[2],
            "performance",
            minimum=_MIN_EXACT_FIXED_FACTOR_SPEEDUP,
            description="Warm fixed-factor likelihood speedup over refactorization.",
        ),
        "fitc_rebuild_compile_seconds": metric(
            fitc_rebuild_timing[1], "performance", unit="s"
        ),
        "fitc_rebuild_warm_seconds": metric(
            fitc_rebuild_timing[2], "performance", unit="s"
        ),
        "fitc_reuse_compile_seconds": metric(
            fitc_reuse_timing[1], "performance", unit="s"
        ),
        "fitc_reuse_warm_seconds": metric(fitc_reuse_timing[2], "performance", unit="s"),
        "fitc_fixed_factor_speedup": metric(
            fitc_rebuild_timing[2] / fitc_reuse_timing[2],
            "performance",
            minimum=(
                _MIN_FITC_FIXED_FACTOR_SPEEDUP
                if num_observations >= _FITC_FIXED_FACTOR_GATE_MIN_OBSERVATIONS
                else None
            ),
            description="Warm reusable-FITC likelihood speedup over rebuilding factors.",
        ),
        "exact_fixed_gradient_warm_seconds": metric(
            exact_fixed_gradient_timing[2], "performance", unit="s"
        ),
        "fitc_fixed_gradient_warm_seconds": metric(
            fitc_fixed_gradient_timing[2], "performance", unit="s"
        ),
        "exact_inferred_gradient_compile_seconds": metric(
            exact_inferred_gradient_timing[1], "performance", unit="s"
        ),
        "exact_inferred_gradient_warm_seconds": metric(
            exact_inferred_gradient_timing[2], "performance", unit="s"
        ),
        "fitc_inferred_gradient_compile_seconds": metric(
            fitc_inferred_gradient_timing[1], "performance", unit="s"
        ),
        "fitc_inferred_gradient_warm_seconds": metric(
            fitc_inferred_gradient_timing[2], "performance", unit="s"
        ),
        "exact_condition_apply_warm_seconds": metric(
            exact_condition_apply_timing[2], "performance", unit="s"
        ),
        "fitc_condition_apply_warm_seconds": metric(
            fitc_condition_apply_timing[2], "performance", unit="s"
        ),
        "fitc_exact_mean_rmse": metric(
            mean_rmse,
            "accuracy",
            maximum=_MAX_FITC_EXACT_MEAN_RMSE,
        ),
        "fitc_exact_variance_rmse": metric(
            variance_rmse,
            "accuracy",
            maximum=_MAX_FITC_EXACT_VARIANCE_RMSE,
        ),
    }
    return ScenarioResult(
        name=f"gp_scaling_n{num_observations}_m{num_inducing}",
        description="Compare exact and FITC fixed/inferred GP costs at one design size.",
        seed=seed,
        metrics=metrics,
        metadata={
            "num_observations": num_observations,
            "num_inducing": num_inducing,
            "timing_repetitions": repetitions,
        },
    )


def _sustained_crossover(
    scenarios: tuple[ScenarioResult, ...],
    *,
    exact_metric: str,
    fitc_metric: str,
) -> int | None:
    """Return the first size after which every measured FITC case is faster."""
    for index, scenario in enumerate(scenarios):
        if all(
            later.metrics[fitc_metric].value <= later.metrics[exact_metric].value
            for later in scenarios[index:]
        ):
            return int(scenario.metadata["num_observations"])
    return None


def run_gp_scaling_benchmark(
    *,
    profile: ProfileName = "smoke",
    root_seed: int = 20260718,
) -> BenchmarkReport:
    """Run deterministic exact-versus-FITC scaling cases."""
    if profile == "smoke":
        cases = _SMOKE_CASES
        repetitions = 3
    elif profile == "standard":
        cases = _STANDARD_CASES
        repetitions = 7
    else:
        raise ValueError("profile must be 'smoke' or 'standard'.")
    jax.config.update("jax_enable_x64", True)
    started_at = utc_now_iso()
    started = time.perf_counter()
    scenarios = tuple(
        _case(
            num_observations=num_observations,
            num_inducing=num_inducing,
            repetitions=repetitions,
            seed=root_seed + index,
        )
        for index, (num_observations, num_inducing) in enumerate(cases)
    )
    return BenchmarkReport(
        profile=profile,
        root_seed=root_seed,
        started_at_utc=started_at,
        duration_seconds=time.perf_counter() - started,
        configuration={
            "profile": profile,
            "cases": [
                {"num_observations": observations, "num_inducing": inducing}
                for observations, inducing in cases
            ],
            "timing_repetitions": repetitions,
            "observed_sustained_crossovers": {
                "fixed_likelihood_rebuild": _sustained_crossover(
                    scenarios,
                    exact_metric="exact_rebuild_warm_seconds",
                    fitc_metric="fitc_rebuild_warm_seconds",
                ),
                "fixed_likelihood_reuse": _sustained_crossover(
                    scenarios,
                    exact_metric="exact_reuse_warm_seconds",
                    fitc_metric="fitc_reuse_warm_seconds",
                ),
                "inferred_hyperparameter_gradient": _sustained_crossover(
                    scenarios,
                    exact_metric="exact_inferred_gradient_warm_seconds",
                    fitc_metric="fitc_inferred_gradient_warm_seconds",
                ),
            },
            "regression_gates": {
                "minimum_exact_fixed_factor_speedup": (_MIN_EXACT_FIXED_FACTOR_SPEEDUP),
                "minimum_fitc_fixed_factor_speedup": (_MIN_FITC_FIXED_FACTOR_SPEEDUP),
                "fitc_fixed_factor_gate_minimum_observations": (
                    _FITC_FIXED_FACTOR_GATE_MIN_OBSERVATIONS
                ),
                "maximum_fitc_exact_mean_rmse": _MAX_FITC_EXACT_MEAN_RMSE,
                "maximum_fitc_exact_variance_rmse": (_MAX_FITC_EXACT_VARIANCE_RMSE),
            },
        },
        environment=collect_environment(),
        scenarios=scenarios,
        suite="phydrax-uq-gp-scaling",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PhydraX exact-versus-FITC GP scaling benchmark."
    )
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--root-seed", type=int, default=20260718)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-fail", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_gp_scaling_benchmark(profile=args.profile, root_seed=args.root_seed)
    output = args.output or Path(f".tmp/uq-gp-scaling-{args.profile}.json")
    report.write_json(output)
    for scenario in report.scenarios:
        status = "PASS" if scenario.passed else "FAIL"
        print(f"{status} {scenario.name}")
    print(
        f"{report.summary['scenarios_passed']}/{len(report.scenarios)} cases passed "
        f"in {report.duration_seconds:.3f}s"
    )
    print(output)
    return 0 if report.passed or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_gp_scaling_benchmark"]
