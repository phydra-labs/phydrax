#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import (
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    synchronize,
)

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
_FIXED_FACTOR_GATE_MIN_OBSERVATIONS = 256
_MAX_FITC_EXACT_MEAN_RMSE = 0.01
_MAX_FITC_EXACT_VARIANCE_RMSE = 0.003
_MAX_CAGP_EXACT_STORAGE_RATIO = 0.75
_MAX_CAGP_CONSERVATIVE_VIOLATION = 1e-8


def _timed_call(function: Callable[[], Any]) -> tuple[Any, float]:
    return measure_synchronized(function)


def _jit_timings(
    function: Callable[..., Any],
    *args: Any,
    repetitions: int,
) -> tuple[float, float, float]:
    jitted = jax.jit(function)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(*args),
        lambda lowered: lowered.compile(),
    )
    _, first_execution_seconds = measure_synchronized(lambda: compiled(*args))
    _, steady = measure_repeated(
        lambda: compiled(*args),
        warmup=0,
        repeats=repetitions,
    )
    compile_seconds = compilation.lowering_seconds + compilation.compilation_seconds
    return (
        compile_seconds + first_execution_seconds,
        compile_seconds,
        float(steady.median_seconds),
    )


def _matern32_state(
    amplitude: Any,
    length_scale: Any,
    noise_scale: Any,
) -> phx.uq.GaussianProcessLikelihoodState:
    return phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(
            phx.kernels.Matern32Kernel(length_scale=length_scale),
            amplitude,
        ),
        noise_scale=noise_scale,
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
    state = _matern32_state(amplitude, length_scale, noise_scale)

    exact_model = phx.uq.ExactGaussianProcessDiscrepancy(
        points,
        observations,
    )
    sparse_model = phx.uq.SparseGaussianProcessDiscrepancy(
        points,
        observations,
        inducing_points,
    )
    computation_aware_model = phx.uq.ComputationAwareGaussianProcessDiscrepancy(
        points,
        observations,
    )
    action_values = jax.random.normal(
        jax.random.key(seed + 10_000),
        (num_observations,),
    )
    computation_aware_actions = phx.uq.BlockSparseGaussianProcessActionPolicy(
        action_values,
        num_inducing,
    )
    exact_factor, exact_factor_build_seconds = _timed_call(
        lambda: exact_model.factor(state=state)
    )
    sparse_factor, fitc_factor_build_seconds = _timed_call(
        lambda: sparse_model.factor(state=state)
    )
    computation_aware_factor, cagp_factor_build_seconds = _timed_call(
        lambda: computation_aware_model.factor(
            state=state,
            actions=computation_aware_actions,
        )
    )
    exact_conditioner, exact_conditioner_build_seconds = _timed_call(
        lambda: exact_factor.conditioner(query)
    )
    sparse_conditioner, fitc_conditioner_build_seconds = _timed_call(
        lambda: sparse_factor.conditioner(query)
    )
    computation_aware_conditioner, cagp_conditioner_build_seconds = _timed_call(
        lambda: computation_aware_factor.conditioner(query)
    )

    exact_rebuild = lambda mean, factor_state: exact_model.log_marginal_likelihood(
        mean,
        state=factor_state,
    )
    exact_reuse = lambda mean, factor: factor.log_probability(exact_model.residual(mean))
    fitc_rebuild = lambda mean, factor_state: sparse_model.log_marginal_likelihood(
        mean,
        state=factor_state,
    )
    fitc_reuse = lambda mean, factor: factor.log_probability(sparse_model.residual(mean))
    cagp_rebuild = lambda mean, factor_state, action_policy: computation_aware_model.elbo(
        mean,
        state=factor_state,
        actions=action_policy,
    )
    cagp_reuse = lambda mean, factor: factor.elbo(computation_aware_model.residual(mean))
    exact_rebuild_timing = _jit_timings(
        exact_rebuild, physical_mean, state, repetitions=repetitions
    )
    exact_reuse_timing = _jit_timings(
        exact_reuse, physical_mean, exact_factor, repetitions=repetitions
    )
    fitc_rebuild_timing = _jit_timings(
        fitc_rebuild, physical_mean, state, repetitions=repetitions
    )
    fitc_reuse_timing = _jit_timings(
        fitc_reuse, physical_mean, sparse_factor, repetitions=repetitions
    )
    cagp_rebuild_timing = _jit_timings(
        cagp_rebuild,
        physical_mean,
        state,
        computation_aware_actions,
        repetitions=repetitions,
    )
    cagp_reuse_timing = _jit_timings(
        cagp_reuse,
        physical_mean,
        computation_aware_factor,
        repetitions=repetitions,
    )

    def fixed_exact_objective(parameter):
        return exact_factor.log_probability(exact_model.residual(parameter * points))

    def fixed_fitc_objective(parameter):
        return sparse_factor.log_probability(sparse_model.residual(parameter * points))

    def fixed_cagp_objective(parameter):
        return computation_aware_factor.elbo(
            computation_aware_model.residual(parameter * points)
        )

    def inferred_exact_objective(unconstrained):
        parameter, log_amplitude, log_length_scale, log_noise_scale = unconstrained
        return exact_model.log_marginal_likelihood(
            parameter * points,
            state=_matern32_state(
                jnp.exp(log_amplitude),
                jnp.exp(log_length_scale),
                jnp.exp(log_noise_scale),
            ),
        )

    def inferred_fitc_objective(unconstrained):
        parameter, log_amplitude, log_length_scale, log_noise_scale = unconstrained
        return sparse_model.log_marginal_likelihood(
            parameter * points,
            state=_matern32_state(
                jnp.exp(log_amplitude),
                jnp.exp(log_length_scale),
                jnp.exp(log_noise_scale),
            ),
        )

    def inferred_cagp_objective(unconstrained):
        parameter, log_amplitude, log_length_scale, log_noise_scale = unconstrained
        return computation_aware_model.elbo(
            parameter * points,
            state=_matern32_state(
                jnp.exp(log_amplitude),
                jnp.exp(log_length_scale),
                jnp.exp(log_noise_scale),
            ),
            actions=computation_aware_actions,
        )

    def inferred_cagp_action_objective(values):
        return computation_aware_model.elbo(
            physical_mean,
            state=state,
            actions=phx.uq.BlockSparseGaussianProcessActionPolicy(
                values,
                num_inducing,
            ),
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
    cagp_fixed_gradient_timing = _jit_timings(
        jax.value_and_grad(fixed_cagp_objective),
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
    cagp_inferred_gradient_timing = _jit_timings(
        jax.value_and_grad(inferred_cagp_objective),
        inferred_parameters,
        repetitions=repetitions,
    )
    cagp_action_gradient_timing = _jit_timings(
        jax.value_and_grad(inferred_cagp_action_objective),
        action_values,
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
    cagp_condition_apply_timing = _jit_timings(
        lambda value: computation_aware_conditioner.condition(value),
        residual,
        repetitions=repetitions,
    )
    exact_condition = exact_conditioner.condition(residual)
    fitc_condition = sparse_conditioner.condition(residual)
    synchronize((exact_condition, fitc_condition))
    cagp_condition = computation_aware_conditioner.condition(residual)
    synchronize(cagp_condition)
    mean_rmse = jnp.sqrt(jnp.mean((fitc_condition.mean - exact_condition.mean) ** 2))
    variance_rmse = jnp.sqrt(
        jnp.mean((fitc_condition.variance - exact_condition.variance) ** 2)
    )
    cagp_mean_rmse = jnp.sqrt(jnp.mean((cagp_condition.mean - exact_condition.mean) ** 2))
    cagp_variance_rmse = jnp.sqrt(
        jnp.mean((cagp_condition.variance - exact_condition.variance) ** 2)
    )
    conservative_eigenvalue = jnp.linalg.eigvalsh(
        cagp_condition.covariance - exact_condition.covariance
    ).min()
    conservative_violation = jnp.maximum(-conservative_eigenvalue, 0.0)
    exact_log_probability = exact_factor.log_probability(residual)
    cagp_elbo = computation_aware_factor.elbo(residual)
    cagp_elbo_gap = exact_log_probability - cagp_elbo
    dtype_bytes = int(observations.dtype.itemsize)
    exact_factor_bytes = dtype_bytes * exact_factor.factor_storage_elements
    fitc_factor_bytes = dtype_bytes * sparse_factor.factor_storage_elements
    cagp_factor_bytes = dtype_bytes * computation_aware_factor.factor_storage_elements

    metrics = {
        "exact_factor_build_seconds": metric(
            exact_factor_build_seconds, "performance", unit="s"
        ),
        "fitc_factor_build_seconds": metric(
            fitc_factor_build_seconds, "performance", unit="s"
        ),
        "cagp_factor_build_seconds": metric(
            cagp_factor_build_seconds, "performance", unit="s"
        ),
        "exact_conditioner_build_seconds": metric(
            exact_conditioner_build_seconds, "performance", unit="s"
        ),
        "fitc_conditioner_build_seconds": metric(
            fitc_conditioner_build_seconds, "performance", unit="s"
        ),
        "cagp_conditioner_build_seconds": metric(
            cagp_conditioner_build_seconds, "performance", unit="s"
        ),
        "exact_factor_bytes": metric(exact_factor_bytes, "performance", unit="bytes"),
        "fitc_factor_bytes": metric(fitc_factor_bytes, "performance", unit="bytes"),
        "cagp_factor_bytes": metric(cagp_factor_bytes, "performance", unit="bytes"),
        "fitc_exact_storage_ratio": metric(
            fitc_factor_bytes / exact_factor_bytes,
            "performance",
            maximum=0.75,
            description="Reusable FITC factor storage relative to dense Cholesky.",
        ),
        "cagp_exact_storage_ratio": metric(
            cagp_factor_bytes / exact_factor_bytes,
            "performance",
            maximum=(
                _MAX_CAGP_EXACT_STORAGE_RATIO
                if num_observations >= _FIXED_FACTOR_GATE_MIN_OBSERVATIONS
                else None
            ),
            description="Reusable computation-aware factor storage relative to exact.",
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
            minimum=(
                _MIN_EXACT_FIXED_FACTOR_SPEEDUP
                if num_observations >= _FIXED_FACTOR_GATE_MIN_OBSERVATIONS
                else None
            ),
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
                if num_observations >= _FIXED_FACTOR_GATE_MIN_OBSERVATIONS
                else None
            ),
            description="Warm reusable-FITC likelihood speedup over rebuilding factors.",
        ),
        "cagp_rebuild_compile_seconds": metric(
            cagp_rebuild_timing[1], "performance", unit="s"
        ),
        "cagp_rebuild_warm_seconds": metric(
            cagp_rebuild_timing[2], "performance", unit="s"
        ),
        "cagp_reuse_compile_seconds": metric(
            cagp_reuse_timing[1], "performance", unit="s"
        ),
        "cagp_reuse_warm_seconds": metric(cagp_reuse_timing[2], "performance", unit="s"),
        "exact_fixed_gradient_warm_seconds": metric(
            exact_fixed_gradient_timing[2], "performance", unit="s"
        ),
        "fitc_fixed_gradient_warm_seconds": metric(
            fitc_fixed_gradient_timing[2], "performance", unit="s"
        ),
        "cagp_fixed_gradient_warm_seconds": metric(
            cagp_fixed_gradient_timing[2], "performance", unit="s"
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
        "cagp_inferred_gradient_compile_seconds": metric(
            cagp_inferred_gradient_timing[1], "performance", unit="s"
        ),
        "cagp_inferred_gradient_warm_seconds": metric(
            cagp_inferred_gradient_timing[2], "performance", unit="s"
        ),
        "cagp_action_gradient_compile_seconds": metric(
            cagp_action_gradient_timing[1], "performance", unit="s"
        ),
        "cagp_action_gradient_warm_seconds": metric(
            cagp_action_gradient_timing[2], "performance", unit="s"
        ),
        "exact_condition_apply_warm_seconds": metric(
            exact_condition_apply_timing[2], "performance", unit="s"
        ),
        "fitc_condition_apply_warm_seconds": metric(
            fitc_condition_apply_timing[2], "performance", unit="s"
        ),
        "cagp_condition_apply_warm_seconds": metric(
            cagp_condition_apply_timing[2], "performance", unit="s"
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
        "cagp_exact_mean_rmse": metric(cagp_mean_rmse, "accuracy"),
        "cagp_exact_variance_rmse": metric(cagp_variance_rmse, "accuracy"),
        "cagp_conservative_covariance_violation": metric(
            conservative_violation,
            "accuracy",
            maximum=_MAX_CAGP_CONSERVATIVE_VIOLATION,
        ),
        "cagp_elbo_gap": metric(
            cagp_elbo_gap,
            "accuracy",
            minimum=-_MAX_CAGP_CONSERVATIVE_VIOLATION,
        ),
    }
    return ScenarioResult(
        name=f"gp_scaling_n{num_observations}_m{num_inducing}",
        description="Compare exact, FITC, and computation-aware GP costs.",
        seed=seed,
        metrics=metrics,
        metadata={
            "num_observations": num_observations,
            "num_inducing": num_inducing,
            "num_actions": num_inducing,
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
    """Run deterministic exact, FITC, and computation-aware GP scaling cases."""
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
                {
                    "num_observations": observations,
                    "num_inducing": inducing,
                    "num_actions": inducing,
                }
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
                "computation_aware_elbo_rebuild": _sustained_crossover(
                    scenarios,
                    exact_metric="exact_rebuild_warm_seconds",
                    fitc_metric="cagp_rebuild_warm_seconds",
                ),
                "computation_aware_elbo_reuse": _sustained_crossover(
                    scenarios,
                    exact_metric="exact_reuse_warm_seconds",
                    fitc_metric="cagp_reuse_warm_seconds",
                ),
                "computation_aware_hyperparameter_gradient": _sustained_crossover(
                    scenarios,
                    exact_metric="exact_inferred_gradient_warm_seconds",
                    fitc_metric="cagp_inferred_gradient_warm_seconds",
                ),
            },
            "regression_gates": {
                "minimum_exact_fixed_factor_speedup": (_MIN_EXACT_FIXED_FACTOR_SPEEDUP),
                "minimum_fitc_fixed_factor_speedup": (_MIN_FITC_FIXED_FACTOR_SPEEDUP),
                "fitc_fixed_factor_gate_minimum_observations": (
                    _FIXED_FACTOR_GATE_MIN_OBSERVATIONS
                ),
                "maximum_fitc_exact_mean_rmse": _MAX_FITC_EXACT_MEAN_RMSE,
                "maximum_fitc_exact_variance_rmse": (_MAX_FITC_EXACT_VARIANCE_RMSE),
                "maximum_cagp_exact_storage_ratio": (_MAX_CAGP_EXACT_STORAGE_RATIO),
                "maximum_cagp_conservative_violation": (_MAX_CAGP_CONSERVATIVE_VIOLATION),
            },
        },
        environment=collect_environment(),
        scenarios=scenarios,
        suite="phydrax-uq-gp-scaling",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact, FITC, and computation-aware GP scaling benchmarks."
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
