#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import time
from collections.abc import Callable
from statistics import median
from typing import Any

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

import phydrax as phx

from .configuration import BenchmarkConfiguration
from .report import Metric, metric, ScenarioResult


Scenario = Callable[[BenchmarkConfiguration, int], ScenarioResult]


def _block_until_ready(value: Any) -> Any:
    return jax.block_until_ready(value)


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
    warm_seconds: list[float] = []
    for _ in range(int(repetitions)):
        started = time.perf_counter()
        _block_until_ready(compiled(*args))
        warm_seconds.append(time.perf_counter() - started)
    execution_seconds = median(warm_seconds)
    return cold_seconds, max(0.0, cold_seconds - execution_seconds), execution_seconds


def _performance_metrics(
    *,
    wall_seconds: float,
    cold_seconds: float,
    compile_seconds: float,
    execution_seconds: float,
    sample_memory_bytes: int | float = 0,
) -> dict[str, Metric]:
    return {
        "wall_seconds": metric(wall_seconds, "performance", unit="s"),
        "forward_cold_seconds": metric(cold_seconds, "performance", unit="s"),
        "forward_compile_seconds": metric(compile_seconds, "performance", unit="s"),
        "forward_execute_seconds": metric(execution_seconds, "performance", unit="s"),
        "sample_memory_bytes": metric(sample_memory_bytes, "performance", unit="byte"),
    }


def _convergence_limits(configuration: BenchmarkConfiguration) -> tuple[float, float]:
    if configuration.profile == "smoke":
        return 1.08, 30.0
    return 1.03, 400.0


def _binomial_bounds(nominal: float, count: int, *, standard_errors: float = 4.0):
    standard_error = math.sqrt(nominal * (1.0 - nominal) / float(count))
    return (
        max(0.0, nominal - standard_errors * standard_error),
        min(1.0, nominal + standard_errors * standard_error),
    )


def _calibration_statistics(
    *,
    mean: Any,
    scale: Any,
    truth: Any,
    observation_scale: float,
    key: Any,
    num_cases: int,
) -> dict[str, float]:
    mean_array = jnp.asarray(mean, dtype=float)
    scale_array = jnp.maximum(jnp.asarray(scale, dtype=float), jnp.finfo(float).eps)
    truth_array = jnp.asarray(truth, dtype=float)
    targets = truth_array + observation_scale * jr.normal(
        key,
        (int(num_cases), *truth_array.shape),
    )
    standardized = (targets - mean_array) / scale_array
    coverage_90 = jnp.mean(jnp.abs(standardized) <= 1.6448536269514722)
    coverage_95 = jnp.mean(jnp.abs(standardized) <= 1.959963984540054)
    nll = jnp.mean(-phx.uq.GaussianLikelihood(scale_array).log_prob(mean_array, targets))
    crps = jnp.mean(phx.uq.gaussian_crps(mean_array, scale_array, targets))
    return {
        "coverage_90": float(coverage_90),
        "coverage_95": float(coverage_95),
        "nll": float(nll),
        "crps": float(crps),
    }


def _calibration_metrics(
    statistics: dict[str, float],
    *,
    num_cases: int,
) -> dict[str, Metric]:
    lower_90, upper_90 = _binomial_bounds(0.90, num_cases)
    lower_95, upper_95 = _binomial_bounds(0.95, num_cases)
    return {
        "predictive_coverage_90": metric(
            statistics["coverage_90"],
            "calibration",
            minimum=lower_90,
            maximum=upper_90,
        ),
        "predictive_coverage_95": metric(
            statistics["coverage_95"],
            "calibration",
            minimum=lower_95,
            maximum=upper_95,
        ),
        "predictive_nll": metric(statistics["nll"], "calibration"),
        "predictive_crps": metric(statistics["crps"], "calibration"),
    }


def _merge_chain_draws(samples: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value: value.reshape((-1, *value.shape[2:])),
        samples,
    )


def _tree_nbytes(tree: Any) -> int:
    return sum(int(jnp.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(tree))


def _elliptic_solution(parameters: Any, *, num_interior: int = 31):
    theta = jnp.asarray(parameters)
    spacing = 1.0 / float(num_interior + 1)
    interfaces = jnp.linspace(0.0, 1.0, num_interior + 1)
    conductivity = jnp.exp(theta[0] + theta[1] * interfaces)
    diagonal = (conductivity[:-1] + conductivity[1:]) / spacing**2
    upper = -conductivity[1:-1] / spacing**2
    matrix = jnp.diag(diagonal) + jnp.diag(upper, 1) + jnp.diag(upper, -1)
    return jnp.linalg.solve(matrix, jnp.ones(num_interior))


def elliptic_coefficient_inverse(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Recover a spatial log-conductivity from a differentiable elliptic solve."""
    started = time.perf_counter()
    root_key = jr.key(seed)
    data_key, calibration_key = jr.split(root_key)
    true_parameters = jnp.asarray([0.2, -0.55])
    true_solution = _elliptic_solution(true_parameters)
    sensor_indices = jnp.arange(1, true_solution.size, 3)
    observation_scale = 0.0025
    observations = true_solution[sensor_indices] + observation_scale * jr.normal(
        data_key,
        sensor_indices.shape,
    )
    likelihood = phx.uq.GaussianLikelihood(observation_scale)
    space = phx.uq.ParameterSpace(
        jnp.asarray([0.0, 0.0]),
        priors=phx.uq.Normal(0.0, 1.0),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda parameters: jnp.sum(
            likelihood.log_prob(
                _elliptic_solution(parameters)[sensor_indices],
                observations,
            )
        ),
    )
    cold, compile_seconds, execute = _jit_timings(
        _elliptic_solution,
        true_parameters,
        repetitions=configuration.jit_warm_repetitions,
    )
    mode, map_seconds = _timed_call(
        lambda: phx.uq.find_map(problem, gradient_tolerance=1e-6)
    )
    nuts, nuts_seconds = _timed_call(
        lambda: phx.uq.sample_nuts(
            problem,
            key=jr.fold_in(root_key, 1),
            num_chains=configuration.num_chains,
            num_warmup=configuration.num_warmup,
            num_samples=configuration.num_draws,
            initial_step_size=0.08,
            target_acceptance_rate=0.9,
            max_num_doublings=8,
            chain_method="vectorized",
        )
    )
    max_rhat, min_ess = _convergence_limits(configuration)
    convergence = nuts.convergence_report(
        max_rhat=max_rhat,
        min_bulk_ess=min_ess,
        min_tail_ess=min_ess,
    )
    flat_parameters = _merge_chain_draws(nuts.samples)
    solution_samples = jax.vmap(_elliptic_solution)(flat_parameters)
    prediction_mean = jnp.mean(solution_samples, axis=0)
    prediction_scale = jnp.sqrt(
        jnp.var(solution_samples, axis=0, ddof=1) + observation_scale**2
    )
    calibration = _calibration_statistics(
        mean=prediction_mean,
        scale=prediction_scale,
        truth=true_solution,
        observation_scale=observation_scale,
        key=calibration_key,
        num_cases=configuration.calibration_cases,
    )
    lower_90 = jnp.quantile(flat_parameters, 0.05, axis=0)
    upper_90 = jnp.quantile(flat_parameters, 0.95, axis=0)
    lower_95 = jnp.quantile(flat_parameters, 0.025, axis=0)
    upper_95 = jnp.quantile(flat_parameters, 0.975, axis=0)
    metrics = {
        "map_parameter_rmse": metric(
            jnp.sqrt(jnp.mean((mode.parameters - true_parameters) ** 2)),
            "accuracy",
            maximum=0.12,
        ),
        "posterior_parameter_rmse": metric(
            jnp.sqrt(
                jnp.mean((jnp.mean(flat_parameters, axis=0) - true_parameters) ** 2)
            ),
            "accuracy",
            maximum=0.12,
        ),
        "field_rmse": metric(
            jnp.sqrt(jnp.mean((prediction_mean - true_solution) ** 2)),
            "accuracy",
            maximum=0.006,
        ),
        "parameter_coverage_90": metric(
            jnp.mean((true_parameters >= lower_90) & (true_parameters <= upper_90)),
            "calibration",
            minimum=0.5,
        ),
        "parameter_coverage_95": metric(
            jnp.mean((true_parameters >= lower_95) & (true_parameters <= upper_95)),
            "calibration",
            minimum=0.5,
        ),
        "max_rhat": metric(nuts.diagnostics.max_rhat, "convergence", maximum=max_rhat),
        "min_bulk_ess": metric(
            nuts.diagnostics.min_bulk_ess,
            "convergence",
            minimum=min_ess,
        ),
        "min_tail_ess": metric(
            nuts.diagnostics.min_tail_ess,
            "convergence",
            minimum=min_ess,
        ),
        "divergence_count": metric(
            nuts.diagnostics.divergence_count,
            "convergence",
            maximum=0.0,
        ),
        "map_seconds": metric(map_seconds, "performance", unit="s"),
        "nuts_seconds": metric(nuts_seconds, "performance", unit="s"),
        "samples_per_second": metric(
            nuts.samples_per_second,
            "performance",
            unit="sample/s",
        ),
        **_calibration_metrics(
            calibration,
            num_cases=configuration.calibration_cases,
        ),
        **_performance_metrics(
            wall_seconds=time.perf_counter() - started,
            cold_seconds=cold,
            compile_seconds=compile_seconds,
            execution_seconds=execute,
            sample_memory_bytes=nuts.sample_memory_bytes,
        ),
    }
    return ScenarioResult(
        name="elliptic_coefficient_inverse",
        description=elliptic_coefficient_inverse.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "equation": "-d/dx(k(x) du/dx) = 1",
            "conductivity": "k(x)=exp(theta_0 + theta_1 x)",
            "num_interior": int(true_solution.size),
            "num_sensors": int(sensor_indices.size),
            "calibration_cases": configuration.calibration_cases,
            "convergence_report_passed": convergence.passed,
        },
    )


def nonlinear_transformed_ode(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Compare NUTS, Laplace, and Pathfinder on a positive-rate decay inverse."""
    started = time.perf_counter()
    root_key = jr.key(seed)
    data_key, calibration_key = jr.split(root_key)
    sensor_time = jnp.linspace(0.0, 2.0, 30)
    query_time = jnp.linspace(0.0, 2.5, 41)
    true_amplitude = 1.7
    true_rate = 0.8
    observation_scale = 0.03

    def forward(parameters, locations):
        return parameters["amplitude"] * jnp.exp(-parameters["rate"] * locations)

    observations = forward(
        {"amplitude": true_amplitude, "rate": true_rate},
        sensor_time,
    ) + observation_scale * jr.normal(data_key, sensor_time.shape)
    space = phx.uq.ParameterSpace(
        {"amplitude": jnp.asarray(1.4), "rate": jnp.log(jnp.asarray(0.6))},
        priors={
            "amplitude": phx.uq.Normal(0.0, 3.0),
            "rate": phx.uq.LogNormal(jnp.log(0.8), 0.5),
        },
        bijectors={
            "amplitude": phx.uq.IdentityBijector(),
            "rate": phx.uq.ExpBijector(),
        },
    )
    likelihood = phx.uq.GaussianLikelihood(observation_scale)
    problem = phx.uq.PosteriorProblem(
        space,
        lambda parameters: jnp.sum(
            likelihood.log_prob(forward(parameters, sensor_time), observations)
        ),
        predict=lambda parameters, locations: cx.Field(
            forward(parameters, locations), dims=("time",)
        ),
    )
    timing_parameters = {"amplitude": jnp.asarray(1.7), "rate": jnp.asarray(0.8)}
    cold, compile_seconds, execute = _jit_timings(
        forward,
        timing_parameters,
        query_time,
        repetitions=configuration.jit_warm_repetitions,
    )
    mode, map_seconds = _timed_call(
        lambda: phx.uq.find_map(problem, gradient_tolerance=1e-7)
    )
    laplace, laplace_seconds = _timed_call(
        lambda: phx.uq.fit_laplace(problem, mode.position)
    )
    pathfinder, pathfinder_seconds = _timed_call(
        lambda: phx.uq.fit_pathfinder(
            problem,
            key=jr.fold_in(root_key, 2),
            num_samples=configuration.pathfinder_samples,
            num_elbo_samples=min(200, configuration.pathfinder_samples),
            max_steps=60,
        )
    )
    nuts, nuts_seconds = _timed_call(
        lambda: phx.uq.sample_nuts(
            problem,
            key=jr.fold_in(root_key, 3),
            num_chains=configuration.num_chains,
            num_warmup=configuration.num_warmup,
            num_samples=configuration.num_draws,
            initial_step_size=0.05,
            target_acceptance_rate=0.9,
            max_num_doublings=8,
            chain_method="vectorized",
        )
    )
    max_rhat, min_ess = _convergence_limits(configuration)
    flat_amplitude = nuts.samples["amplitude"].reshape((-1,))
    flat_rate = nuts.samples["rate"].reshape((-1,))
    prediction_samples = flat_amplitude[:, None] * jnp.exp(
        -flat_rate[:, None] * query_time[None, :]
    )
    prediction_mean = jnp.mean(prediction_samples, axis=0)
    prediction_scale = jnp.sqrt(
        jnp.var(prediction_samples, axis=0, ddof=1) + observation_scale**2
    )
    truth = true_amplitude * jnp.exp(-true_rate * query_time)
    calibration = _calibration_statistics(
        mean=prediction_mean,
        scale=prediction_scale,
        truth=truth,
        observation_scale=observation_scale,
        key=calibration_key,
        num_cases=configuration.calibration_cases,
    )
    laplace_samples = laplace.sample(
        jr.fold_in(root_key, 4),
        num_samples=configuration.posterior_prediction_samples,
    )
    metrics = {
        "map_amplitude_error": metric(
            jnp.abs(mode.parameters["amplitude"] - true_amplitude),
            "accuracy",
            maximum=0.03,
        ),
        "map_rate_error": metric(
            jnp.abs(mode.parameters["rate"] - true_rate),
            "accuracy",
            maximum=0.03,
        ),
        "posterior_field_rmse": metric(
            jnp.sqrt(jnp.mean((prediction_mean - truth) ** 2)),
            "accuracy",
            maximum=0.01,
        ),
        "laplace_nuts_amplitude_delta": metric(
            jnp.abs(jnp.mean(laplace_samples["amplitude"]) - jnp.mean(flat_amplitude)),
            "accuracy",
            maximum=0.015,
        ),
        "laplace_nuts_rate_delta": metric(
            jnp.abs(jnp.mean(laplace_samples["rate"]) - jnp.mean(flat_rate)),
            "accuracy",
            maximum=0.015,
        ),
        "pathfinder_nuts_amplitude_delta": metric(
            jnp.abs(jnp.mean(pathfinder.samples["amplitude"]) - jnp.mean(flat_amplitude)),
            "accuracy",
            maximum=0.015,
        ),
        "pathfinder_nuts_rate_delta": metric(
            jnp.abs(jnp.mean(pathfinder.samples["rate"]) - jnp.mean(flat_rate)),
            "accuracy",
            maximum=0.015,
        ),
        "max_rhat": metric(nuts.diagnostics.max_rhat, "convergence", maximum=max_rhat),
        "min_bulk_ess": metric(
            nuts.diagnostics.min_bulk_ess, "convergence", minimum=min_ess
        ),
        "min_tail_ess": metric(
            nuts.diagnostics.min_tail_ess, "convergence", minimum=min_ess
        ),
        "divergence_count": metric(
            nuts.diagnostics.divergence_count, "convergence", maximum=0.0
        ),
        "map_seconds": metric(map_seconds, "performance", unit="s"),
        "laplace_seconds": metric(laplace_seconds, "performance", unit="s"),
        "pathfinder_seconds": metric(pathfinder_seconds, "performance", unit="s"),
        "nuts_seconds": metric(nuts_seconds, "performance", unit="s"),
        **_calibration_metrics(calibration, num_cases=configuration.calibration_cases),
        **_performance_metrics(
            wall_seconds=time.perf_counter() - started,
            cold_seconds=cold,
            compile_seconds=compile_seconds,
            execution_seconds=execute,
            sample_memory_bytes=(
                nuts.sample_memory_bytes
                + _tree_nbytes(laplace_samples)
                + pathfinder.sample_memory_bytes
            ),
        ),
    }
    return ScenarioResult(
        name="nonlinear_transformed_ode",
        description=nonlinear_transformed_ode.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "model": "amplitude * exp(-rate * time)",
            "positive_parameter_bijector": "ExpBijector",
            "calibration_cases": configuration.calibration_cases,
        },
    )


def neural_selected_subspace(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Infer an explicitly selected neural last layer with structured GGN Laplace."""
    started = time.perf_counter()
    root_key = jr.key(seed)
    data_key, calibration_key = jr.split(root_key)
    sensor_x = jnp.linspace(-1.0, 1.0, 36)
    query_x = jnp.linspace(-1.2, 1.2, 49)
    observation_scale = 0.04
    model = {
        "features": {
            "bias": jnp.asarray([-0.3, 0.1, 0.25, -0.15]),
            "weight": jnp.asarray([-2.0, -0.7, 0.8, 1.9]),
        },
        "head": {
            "bias": jnp.asarray(0.0),
            "weight": jnp.zeros(4),
        },
    }
    true_model = {
        **model,
        "head": {
            "bias": jnp.asarray(0.12),
            "weight": jnp.asarray([0.7, -1.0, 0.45, 0.8]),
        },
    }
    subspace = phx.uq.ParameterSubspace.last_layer(model, num_leaves=2)

    def full_forward(full_model, locations):
        hidden = jnp.tanh(
            locations[:, None] * full_model["features"]["weight"][None, :]
            + full_model["features"]["bias"][None, :]
        )
        return hidden @ full_model["head"]["weight"] + full_model["head"]["bias"]

    def selected_forward(selected, locations):
        return full_forward(subspace.reconstruct(selected), locations)

    sensor_truth = full_forward(true_model, sensor_x)
    observations = sensor_truth + observation_scale * jr.normal(data_key, sensor_x.shape)
    priors = jax.tree_util.tree_map(
        lambda _: phx.uq.Normal(0.0, 2.0),
        subspace.initial,
    )
    space = phx.uq.ParameterSpace(subspace.initial, priors=priors)

    def normalized_residual(selected):
        return (selected_forward(selected, sensor_x) - observations) / observation_scale

    problem = phx.uq.PosteriorProblem(
        space,
        lambda selected: -0.5 * jnp.sum(normalized_residual(selected) ** 2),
        gauss_newton_residual=normalized_residual,
    )
    cold, compile_seconds, execute = _jit_timings(
        selected_forward,
        subspace.initial,
        query_x,
        repetitions=configuration.jit_warm_repetitions,
    )
    mode, map_seconds = _timed_call(
        lambda: phx.uq.find_map(problem, gradient_tolerance=1e-6)
    )
    structured, laplace_seconds = _timed_call(
        lambda: phx.uq.fit_laplace(
            problem,
            mode.position,
            curvature="full",
            likelihood_curvature="ggn",
        )
    )
    nuts, nuts_seconds = _timed_call(
        lambda: phx.uq.sample_nuts(
            problem,
            key=jr.fold_in(root_key, 1),
            num_chains=configuration.num_chains,
            num_warmup=configuration.num_warmup,
            num_samples=configuration.num_draws,
            initial_step_size=0.08,
            target_acceptance_rate=0.9,
            max_num_doublings=8,
            chain_method="vectorized",
        )
    )
    flat_selected = _merge_chain_draws(nuts.samples)
    prediction_samples = jax.vmap(lambda selected: selected_forward(selected, query_x))(
        flat_selected
    )
    prediction_mean = jnp.mean(prediction_samples, axis=0)
    prediction_scale = jnp.sqrt(
        jnp.var(prediction_samples, axis=0, ddof=1) + observation_scale**2
    )
    truth = full_forward(true_model, query_x)
    calibration = _calibration_statistics(
        mean=prediction_mean,
        scale=prediction_scale,
        truth=truth,
        observation_scale=observation_scale,
        key=calibration_key,
        num_cases=configuration.calibration_cases,
    )
    laplace_draws = structured.sample(
        jr.fold_in(root_key, 2),
        num_samples=configuration.posterior_prediction_samples,
    )
    laplace_prediction = jax.vmap(lambda selected: selected_forward(selected, query_x))(
        laplace_draws
    )
    posterior_selected_mean = jax.tree_util.tree_map(
        lambda value: jnp.mean(value, axis=0),
        flat_selected,
    )
    inferred_model = subspace.reconstruct(posterior_selected_mean)
    max_rhat, min_ess = _convergence_limits(configuration)
    metrics = {
        "selected_dimension": metric(
            subspace.total_dimension,
            "diagnostic",
            minimum=5.0,
            maximum=5.0,
        ),
        "last_layer_parameter_rmse": metric(
            jnp.sqrt(
                (
                    jnp.sum(
                        (inferred_model["head"]["weight"] - true_model["head"]["weight"])
                        ** 2
                    )
                    + (inferred_model["head"]["bias"] - true_model["head"]["bias"]) ** 2
                )
                / 5.0
            ),
            "diagnostic",
        ),
        "posterior_field_rmse": metric(
            jnp.sqrt(jnp.mean((prediction_mean - truth) ** 2)),
            "accuracy",
            maximum=0.035,
        ),
        "structured_nuts_field_delta": metric(
            jnp.sqrt(
                jnp.mean((jnp.mean(laplace_prediction, axis=0) - prediction_mean) ** 2)
            ),
            "accuracy",
            maximum=0.012,
        ),
        "max_rhat": metric(nuts.diagnostics.max_rhat, "convergence", maximum=max_rhat),
        "min_bulk_ess": metric(
            nuts.diagnostics.min_bulk_ess, "convergence", minimum=min_ess
        ),
        "min_tail_ess": metric(
            nuts.diagnostics.min_tail_ess, "convergence", minimum=min_ess
        ),
        "divergence_count": metric(
            nuts.diagnostics.divergence_count, "convergence", maximum=0.0
        ),
        "map_seconds": metric(map_seconds, "performance", unit="s"),
        "structured_laplace_seconds": metric(laplace_seconds, "performance", unit="s"),
        "nuts_seconds": metric(nuts_seconds, "performance", unit="s"),
        **_calibration_metrics(calibration, num_cases=configuration.calibration_cases),
        **_performance_metrics(
            wall_seconds=time.perf_counter() - started,
            cold_seconds=cold,
            compile_seconds=compile_seconds,
            execution_seconds=execute,
            sample_memory_bytes=(
                nuts.sample_memory_bytes + structured.approximate_memory_bytes
            ),
        ),
    }
    return ScenarioResult(
        name="neural_selected_subspace",
        description=neural_selected_subspace.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "subspace_leaf_paths": list(subspace.leaf_paths),
            "curvature": structured.curvature,
            "likelihood_curvature": structured.likelihood_curvature,
            "calibration_cases": configuration.calibration_cases,
        },
    )


def multimodal_tempered_inference(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Require tempered SMC to preserve both modes that local Pathfinder misses."""
    started = time.perf_counter()
    root_key = jr.key(seed)
    prior_scale = 3.0
    likelihood_scale = 0.3
    mode_location = 2.0
    component_variance = 1.0 / (1.0 / likelihood_scale**2 + 1.0 / prior_scale**2)
    component_mean = component_variance * mode_location / likelihood_scale**2
    expected_variance = component_variance + component_mean**2

    def log_likelihood(value):
        return jsp.special.logsumexp(
            jnp.stack(
                [
                    -0.5 * ((value - mode_location) / likelihood_scale) ** 2,
                    -0.5 * ((value + mode_location) / likelihood_scale) ** 2,
                ]
            )
        )

    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.asarray(1.8),
            priors=phx.uq.Normal(0.0, prior_scale),
        ),
        log_likelihood,
    )
    cold, compile_seconds, execute = _jit_timings(
        log_likelihood,
        jnp.asarray(1.0),
        repetitions=configuration.jit_warm_repetitions,
    )
    smc, smc_seconds = _timed_call(
        lambda: phx.uq.sample_tempered_smc(
            problem,
            key=jr.fold_in(root_key, 1),
            num_particles=configuration.smc_particles,
            target_ess=0.8,
            num_mcmc_steps=5,
            step_size=0.15,
            num_integration_steps=8,
        )
    )
    pathfinder, pathfinder_seconds = _timed_call(
        lambda: phx.uq.fit_pathfinder(
            problem,
            key=jr.fold_in(root_key, 2),
            num_samples=configuration.pathfinder_samples,
            num_elbo_samples=min(200, configuration.pathfinder_samples),
            max_steps=50,
        )
    )
    positive_mass = jnp.mean(smc.samples > 0.0)
    local_positive_mass = jnp.mean(pathfinder.samples > 0.0)
    metrics = {
        "positive_mode_mass": metric(
            positive_mass,
            "accuracy",
            minimum=0.40,
            maximum=0.60,
        ),
        "posterior_mean_error": metric(
            jnp.abs(jnp.mean(smc.samples)),
            "accuracy",
            maximum=0.22,
        ),
        "posterior_variance_relative_error": metric(
            jnp.abs(jnp.var(smc.samples) - expected_variance) / expected_variance,
            "accuracy",
            maximum=0.12,
        ),
        "pathfinder_single_mode_mass": metric(
            jnp.maximum(local_positive_mass, 1.0 - local_positive_mass),
            "diagnostic",
            minimum=0.95,
        ),
        "final_temperature": metric(
            smc.temperatures[-1],
            "convergence",
            minimum=1.0,
            maximum=1.0,
        ),
        "max_divergence_rate": metric(
            jnp.max(smc.divergence_rates),
            "convergence",
            maximum=0.01,
        ),
        "minimum_relative_ess": metric(
            jnp.min(smc.effective_sample_sizes[1:]) / configuration.smc_particles,
            "convergence",
            minimum=0.35,
        ),
        "unique_initial_particle_fraction": metric(
            smc.num_unique_initial_particles / configuration.smc_particles,
            "convergence",
            minimum=0.15,
        ),
        "tempering_steps": metric(
            smc.num_tempering_steps,
            "diagnostic",
            minimum=2.0,
        ),
        "smc_seconds": metric(smc_seconds, "performance", unit="s"),
        "pathfinder_seconds": metric(pathfinder_seconds, "performance", unit="s"),
        **_performance_metrics(
            wall_seconds=time.perf_counter() - started,
            cold_seconds=cold,
            compile_seconds=compile_seconds,
            execution_seconds=execute,
            sample_memory_bytes=(
                smc.sample_memory_bytes + pathfinder.sample_memory_bytes
            ),
        ),
    }
    return ScenarioResult(
        name="multimodal_tempered_inference",
        description=multimodal_tempered_inference.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "expected_modes": [-float(component_mean), float(component_mean)],
            "smc_particles": configuration.smc_particles,
            "resampling_method": smc.resampling_method,
        },
    )


def flow_assisted_multimodal(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Measure exact flow-assisted transport across represented posterior modes."""
    started = time.perf_counter()
    root_key = jr.key(seed)
    dimension = 4
    prior_scale = 4.0
    positive_weight = 0.7
    component_location = jnp.asarray([2.2, -1.8, 1.5, 2.0])
    likelihood_covariance = jnp.asarray(
        [
            [0.20, 0.04, 0.00, 0.02],
            [0.04, 0.16, 0.03, 0.00],
            [0.00, 0.03, 0.18, -0.02],
            [0.02, 0.00, -0.02, 0.22],
        ]
    )
    likelihood_precision = jnp.linalg.inv(likelihood_covariance)
    component_covariance = jnp.linalg.inv(
        likelihood_precision + jnp.eye(dimension) / prior_scale**2
    )
    component_mean = component_covariance @ likelihood_precision @ component_location
    exact_mean = (2.0 * positive_weight - 1.0) * component_mean
    exact_covariance = component_covariance + 4.0 * positive_weight * (
        1.0 - positive_weight
    ) * jnp.outer(component_mean, component_mean)

    def component_log_density(value, location):
        difference = value - location
        return -0.5 * difference @ likelihood_precision @ difference

    def log_likelihood(value):
        return jsp.special.logsumexp(
            jnp.stack(
                (
                    jnp.log(1.0 - positive_weight)
                    + component_log_density(value, -component_location),
                    jnp.log(positive_weight)
                    + component_log_density(value, component_location),
                )
            )
        )

    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.zeros((dimension,)),
            priors=phx.uq.Normal(0.0, prior_scale),
        ),
        log_likelihood,
    )
    initial_signs = jnp.where(
        jnp.arange(configuration.num_chains) % 2 == 0,
        -1.0,
        1.0,
    )
    initial_positions = initial_signs[:, None] * component_mean[None, :]
    cold, compile_seconds, execute = _jit_timings(
        log_likelihood,
        component_mean,
        repetitions=configuration.jit_warm_repetitions,
    )
    nuts, nuts_seconds = _timed_call(
        lambda: phx.uq.sample_nuts(
            problem,
            key=jr.fold_in(root_key, 1),
            num_chains=configuration.num_chains,
            num_warmup=configuration.num_warmup,
            num_samples=configuration.num_draws,
            initial_positions=initial_positions,
            initial_step_size=0.12,
            target_acceptance_rate=0.9,
            max_num_doublings=8,
            chain_method="vectorized",
        )
    )
    flow_config = phx.uq.FlowNUTSConfig(
        num_adaptation_rounds=configuration.flow_adaptation_rounds,
        num_local_adaptation_steps=configuration.flow_local_adaptation_steps,
        num_global_adaptation_steps=configuration.flow_global_adaptation_steps,
        num_stabilization_steps=20,
        num_local_steps=configuration.flow_local_steps,
        num_global_steps=configuration.flow_global_steps,
        history_capacity_per_chain=configuration.flow_history_capacity,
        history_thinning=1,
        flow_layers=3,
        num_knots=8,
        nn_width=32,
        nn_depth=2,
        learning_rate=1e-3,
        max_epochs=configuration.flow_epochs,
        max_patience=configuration.flow_epochs,
        batch_size=64,
        validation_fraction=0.2,
    )
    flow, flow_seconds = _timed_call(
        lambda: phx.uq.sample_flow_nuts(
            problem,
            key=jr.fold_in(root_key, 2),
            num_chains=configuration.num_chains,
            num_warmup=configuration.num_warmup,
            num_samples=configuration.num_draws,
            initial_positions=initial_positions,
            initial_step_size=0.12,
            target_acceptance_rate=0.9,
            max_num_doublings=8,
            config=flow_config,
            chain_method="vectorized",
        )
    )
    flat_flow = flow.samples.reshape((-1, dimension))
    flat_nuts = nuts.samples.reshape((-1, dimension))
    flow_positive = flat_flow @ component_mean > 0.0
    nuts_positive = flat_nuts @ component_mean > 0.0
    chain_modes = flow.samples @ component_mean > 0.0
    nuts_chain_modes = nuts.samples @ component_mean > 0.0
    flow_transitions = jnp.sum(
        chain_modes[:, 1:] != chain_modes[:, :-1],
        axis=1,
    )
    nuts_transitions = jnp.sum(
        nuts_chain_modes[:, 1:] != nuts_chain_modes[:, :-1],
        axis=1,
    )
    estimated_covariance = jnp.cov(flat_flow, rowvar=False)
    proposal_ess_fraction = flow.adaptation_proposal_ess[-1] / (
        configuration.num_chains * configuration.flow_global_adaptation_steps
    )
    max_rhat = 1.15 if configuration.profile == "smoke" else 1.05
    min_ess = 20.0 if configuration.profile == "smoke" else 200.0
    mode_mass_error = 0.15 if configuration.profile == "smoke" else 0.06
    mean_rmse = 0.30 if configuration.profile == "smoke" else 0.12
    covariance_error = 0.30 if configuration.profile == "smoke" else 0.15
    minimum_global_acceptance = 0.03 if configuration.profile == "smoke" else 0.05
    minimum_proposal_ess_fraction = 0.05 if configuration.profile == "smoke" else 0.10
    sample_count = configuration.num_chains * configuration.num_draws
    metrics = {
        "flow_mode_mass_error": metric(
            jnp.abs(jnp.mean(flow_positive) - positive_weight),
            "accuracy",
            maximum=mode_mass_error,
        ),
        "flow_posterior_mean_rmse": metric(
            jnp.sqrt(jnp.mean((jnp.mean(flat_flow, axis=0) - exact_mean) ** 2)),
            "accuracy",
            maximum=mean_rmse,
        ),
        "flow_covariance_relative_frobenius_error": metric(
            jnp.linalg.norm(estimated_covariance - exact_covariance)
            / jnp.linalg.norm(exact_covariance),
            "accuracy",
            maximum=covariance_error,
        ),
        "ordinary_nuts_mode_mass_error": metric(
            jnp.abs(jnp.mean(nuts_positive) - positive_weight),
            "diagnostic",
        ),
        "minimum_flow_mode_transitions_per_chain": metric(
            jnp.min(flow_transitions),
            "convergence",
            minimum=1.0,
        ),
        "minimum_nuts_mode_transitions_per_chain": metric(
            jnp.min(nuts_transitions),
            "diagnostic",
        ),
        "max_rhat": metric(
            flow.diagnostics.max_rhat,
            "convergence",
            maximum=max_rhat,
        ),
        "min_bulk_ess": metric(
            flow.diagnostics.min_bulk_ess,
            "convergence",
            minimum=min_ess,
        ),
        "divergence_count": metric(
            flow.diagnostics.divergence_count,
            "convergence",
            maximum=0.0,
        ),
        "global_acceptance_rate": metric(
            jnp.mean(flow.global_acceptance_rate),
            "convergence",
            minimum=minimum_global_acceptance,
        ),
        "proposal_ess_fraction": metric(
            proposal_ess_fraction,
            "diagnostic",
            minimum=minimum_proposal_ess_fraction,
        ),
        "nuts_seconds": metric(nuts_seconds, "performance", unit="s"),
        "flow_seconds": metric(flow_seconds, "performance", unit="s"),
        "flow_adaptation_seconds": metric(
            flow.adaptation_duration_seconds,
            "performance",
            unit="s",
        ),
        "flow_training_seconds": metric(
            sum(flow.flow_training_duration_seconds),
            "performance",
            unit="s",
        ),
        "flow_production_seconds": metric(
            flow.sampling_duration_seconds,
            "performance",
            unit="s",
        ),
        "flow_samples_per_second": metric(
            sample_count / flow.sampling_duration_seconds,
            "performance",
            unit="sample/s",
        ),
        "flow_min_bulk_ess_per_second": metric(
            flow.diagnostics.min_bulk_ess / flow_seconds,
            "performance",
            unit="1/s",
        ),
        "flow_parameter_memory_bytes": metric(
            flow.flow_parameter_memory_bytes,
            "performance",
            unit="byte",
        ),
        "flow_history_memory_bytes": metric(
            flow.history_memory_bytes,
            "performance",
            unit="byte",
        ),
        "production_global_target_evaluations": metric(
            sample_count * configuration.flow_global_steps,
            "performance",
        ),
        "production_local_integration_steps": metric(
            jnp.sum(flow.num_integration_steps),
            "performance",
        ),
        **_performance_metrics(
            wall_seconds=time.perf_counter() - started,
            cold_seconds=cold,
            compile_seconds=compile_seconds,
            execution_seconds=execute,
            sample_memory_bytes=(
                nuts.sample_memory_bytes
                + flow.sample_memory_bytes
                + flow.flow_parameter_memory_bytes
                + flow.history_memory_bytes
            ),
        ),
    }
    return ScenarioResult(
        name="flow_assisted_multimodal",
        description=flow_assisted_multimodal.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "dimension": dimension,
            "positive_mode_weight": positive_weight,
            "component_means": [
                (-component_mean).tolist(),
                component_mean.tolist(),
            ],
            "component_covariance": component_covariance.tolist(),
            "exact_mean": exact_mean.tolist(),
            "exact_covariance": exact_covariance.tolist(),
            "initial_positions": initial_positions.tolist(),
            "flow_config": flow_config.as_dict(),
        },
    )


def _poisson_basis(x):
    return 0.5 * x * (1.0 - x)


def misspecified_pde_discrepancy(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Gate exact, sparse, and inferred GP corrections for an omitted PDE source."""
    started = time.perf_counter()
    root_key = jr.key(seed)
    observation_x = jnp.linspace(0.0, 1.0, 24)
    test_x = jnp.linspace(0.02, 0.98, 37)
    true_parameter = 1.2
    observation_scale = 0.03
    fixed_amplitude = 0.25
    fixed_length_scale = 0.22

    def physical(parameter, locations):
        return parameter * locations

    def truth(locations):
        return physical(true_parameter, locations) + 0.3 * jnp.sin(jnp.pi * locations)

    def joint_physical_mean(parameters):
        return physical(parameters["parameter"], observation_x)

    def joint_hyperparameters(parameters):
        return {
            "amplitude": parameters["amplitude"],
            "length_scale": parameters["length_scale"],
            "noise_scale": parameters["noise_scale"],
        }

    cold, compile_seconds, execute = _jit_timings(
        truth,
        test_x,
        repetitions=configuration.jit_warm_repetitions,
    )
    baseline_parameters = []
    fixed_parameters = []
    joint_parameters = []
    baseline_rmse = []
    fixed_rmse = []
    sparse_rmse = []
    baseline_nll = []
    fixed_nll = []
    baseline_crps = []
    fixed_crps = []
    fixed_coverage = []
    joint_correlations = []
    sparse_exact_mean_delta = []
    sparse_exact_variance_delta = []
    inducing_indices = jnp.round(jnp.linspace(0, int(observation_x.size) - 1, 9)).astype(
        jnp.int32
    )
    inducing_x = observation_x[inducing_indices]
    baseline_map_stage_seconds = []
    fixed_exact_map_stage_seconds = []
    sparse_fitc_map_stage_seconds = []
    joint_hyperparameter_map_stage_seconds = []
    joint_map_compilation_seconds = []
    joint_map_execution_seconds = []
    joint_map_steps = []
    joint_map_objective_evaluations = []
    joint_laplace_stage_seconds = []
    exact_condition_stage_seconds = []
    fitc_condition_stage_seconds = []
    likelihood = phx.uq.GaussianLikelihood(observation_scale)
    exact_factor, exact_factor_build_seconds = _timed_call(
        lambda: phx.uq.ExactGaussianProcessFactor(
            observation_x,
            amplitude=fixed_amplitude,
            length_scale=fixed_length_scale,
            noise_scale=observation_scale,
            kernel="matern32",
        )
    )
    sparse_factor, fitc_factor_build_seconds = _timed_call(
        lambda: phx.uq.SparseGaussianProcessFactor(
            observation_x,
            inducing_x,
            amplitude=fixed_amplitude,
            length_scale=fixed_length_scale,
            noise_scale=observation_scale,
            kernel="matern32",
        )
    )
    exact_conditioner, exact_conditioner_build_seconds = _timed_call(
        lambda: exact_factor.conditioner(test_x)
    )
    sparse_conditioner, fitc_conditioner_build_seconds = _timed_call(
        lambda: sparse_factor.conditioner(test_x)
    )
    exact_storage = exact_factor.factor_storage_elements
    sparse_storage = sparse_factor.factor_storage_elements

    for repeat in range(configuration.gp_repeats):
        observations = truth(observation_x) + observation_scale * jr.normal(
            jr.fold_in(root_key, repeat), observation_x.shape
        )
        exact_gp = phx.uq.ExactGaussianProcessDiscrepancy(
            observation_x,
            observations,
            kernel="matern32",
        )
        sparse_gp = phx.uq.SparseGaussianProcessDiscrepancy(
            observation_x,
            observations,
            inducing_x,
            kernel="matern32",
        )
        physical_space = phx.uq.ParameterSpace(
            {"parameter": jnp.asarray(1.0)},
            priors={"parameter": phx.uq.Normal(0.0, 3.0)},
        )
        baseline_problem = phx.uq.PosteriorProblem(
            physical_space,
            lambda parameters: jnp.sum(
                likelihood.log_prob(
                    physical(parameters["parameter"], observation_x), observations
                )
            ),
        )
        fixed_problem = phx.uq.PosteriorProblem(
            physical_space,
            lambda parameters: exact_factor.log_probability(
                exact_gp.residual(physical(parameters["parameter"], observation_x))
            ),
        )
        sparse_problem = phx.uq.PosteriorProblem(
            physical_space,
            lambda parameters: sparse_factor.log_probability(
                sparse_gp.residual(physical(parameters["parameter"], observation_x))
            ),
        )
        joint_space = phx.uq.ParameterSpace(
            {
                "parameter": jnp.asarray(1.0),
                "amplitude": jnp.log(jnp.asarray(fixed_amplitude)),
                "length_scale": jnp.log(jnp.asarray(fixed_length_scale)),
                "noise_scale": jnp.log(jnp.asarray(observation_scale)),
            },
            priors={
                "parameter": phx.uq.Normal(0.0, 3.0),
                "amplitude": phx.uq.LogNormal(jnp.log(fixed_amplitude), 0.5),
                "length_scale": phx.uq.LogNormal(jnp.log(fixed_length_scale), 0.5),
                "noise_scale": phx.uq.LogNormal(jnp.log(observation_scale), 0.3),
            },
            bijectors={
                "parameter": phx.uq.IdentityBijector(),
                "amplitude": phx.uq.ExpBijector(),
                "length_scale": phx.uq.ExpBijector(),
                "noise_scale": phx.uq.ExpBijector(),
            },
        )
        joint_problem = phx.uq.PosteriorProblem.from_terms(
            joint_space,
            [
                phx.uq.GaussianProcessMarginalLikelihood(
                    exact_gp,
                    joint_physical_mean,
                    hyperparameters=joint_hyperparameters,
                    label="joint_gp_discrepancy",
                )
            ],
        )
        baseline_mode, baseline_map_seconds = _timed_call(
            lambda: phx.uq.find_map(baseline_problem)
        )
        fixed_mode, fixed_map_seconds = _timed_call(
            lambda: phx.uq.find_map(fixed_problem)
        )
        sparse_mode, sparse_map_seconds = _timed_call(
            lambda: phx.uq.find_map(sparse_problem)
        )
        joint_mode, joint_map_seconds = _timed_call(
            lambda: phx.uq.find_map(joint_problem, gradient_tolerance=1e-5)
        )
        joint_laplace, joint_laplace_seconds = _timed_call(
            lambda: phx.uq.fit_laplace(
                joint_problem,
                joint_mode.position,
                damping=1e-6,
                stationarity_tolerance=1e-4,
            )
        )
        baseline_map_stage_seconds.append(baseline_map_seconds)
        fixed_exact_map_stage_seconds.append(fixed_map_seconds)
        sparse_fitc_map_stage_seconds.append(sparse_map_seconds)
        joint_hyperparameter_map_stage_seconds.append(joint_map_seconds)
        joint_laplace_stage_seconds.append(joint_laplace_seconds)
        joint_map_compilation_seconds.append(joint_mode.compilation_seconds)
        joint_map_execution_seconds.append(joint_mode.execution_seconds)
        joint_map_steps.append(joint_mode.num_steps)
        joint_map_objective_evaluations.append(joint_mode.objective_evaluations)
        if not isinstance(joint_laplace, phx.uq.LaplaceResult):
            raise RuntimeError("Joint discrepancy benchmark requires dense Laplace.")
        baseline_parameter = baseline_mode.parameters["parameter"]
        fixed_parameter = fixed_mode.parameters["parameter"]
        sparse_parameter = sparse_mode.parameters["parameter"]
        joint_parameter = joint_mode.parameters["parameter"]
        exact_condition, exact_condition_seconds = _timed_call(
            lambda: exact_conditioner.condition(
                exact_gp.residual(physical(fixed_parameter, observation_x))
            )
        )
        sparse_condition, sparse_condition_seconds = _timed_call(
            lambda: sparse_conditioner.condition(
                sparse_gp.residual(physical(sparse_parameter, observation_x))
            )
        )
        exact_condition_stage_seconds.append(exact_condition_seconds)
        fitc_condition_stage_seconds.append(sparse_condition_seconds)
        target = truth(test_x)
        baseline_mean = physical(baseline_parameter, test_x)
        fixed_mean = physical(fixed_parameter, test_x) + exact_condition.mean
        sparse_mean = physical(sparse_parameter, test_x) + sparse_condition.mean
        fixed_scale = jnp.sqrt(exact_condition.variance + observation_scale**2)
        interval_radius = 1.6448536269514722 * jnp.sqrt(exact_condition.variance)
        baseline_parameters.append(baseline_parameter)
        fixed_parameters.append(fixed_parameter)
        joint_parameters.append(joint_parameter)
        baseline_rmse.append(jnp.sqrt(jnp.mean((baseline_mean - target) ** 2)))
        fixed_rmse.append(jnp.sqrt(jnp.mean((fixed_mean - target) ** 2)))
        sparse_rmse.append(jnp.sqrt(jnp.mean((sparse_mean - target) ** 2)))
        baseline_nll.append(jnp.mean(-likelihood.log_prob(baseline_mean, target)))
        fixed_nll.append(
            jnp.mean(-phx.uq.GaussianLikelihood(fixed_scale).log_prob(fixed_mean, target))
        )
        baseline_crps.append(
            jnp.mean(phx.uq.gaussian_crps(baseline_mean, observation_scale, target))
        )
        fixed_crps.append(jnp.mean(phx.uq.gaussian_crps(fixed_mean, fixed_scale, target)))
        fixed_coverage.append(
            jnp.mean(
                (target >= fixed_mean - interval_radius)
                & (target <= fixed_mean + interval_radius)
            )
        )
        paths = phx.uq.ParameterSubspace.array_leaf_paths(joint_mode.parameters)
        parameter_index = paths.index("['parameter']")
        gp_indices = jnp.asarray(
            [index for index, path in enumerate(paths) if path != "['parameter']"]
        )
        joint_correlations.append(
            joint_laplace.physical_correlation()[parameter_index, gp_indices]
        )
        sparse_exact_mean_delta.append(
            jnp.sqrt(jnp.mean((sparse_mean - fixed_mean) ** 2))
        )
        sparse_exact_variance_delta.append(
            jnp.sqrt(
                jnp.mean((sparse_condition.variance - exact_condition.variance) ** 2)
            )
        )

    baseline_parameters_array = jnp.stack(baseline_parameters)
    fixed_parameters_array = jnp.stack(fixed_parameters)
    joint_parameters_array = jnp.stack(joint_parameters)
    identifiability = phx.uq.discrepancy_identifiability_report(
        true_parameters=true_parameter,
        baseline_parameter_estimates=baseline_parameters_array,
        fixed_gp_parameter_estimates=fixed_parameters_array,
        joint_gp_parameter_estimates=joint_parameters_array,
        baseline_nll=jnp.stack(baseline_nll),
        fixed_gp_nll=jnp.stack(fixed_nll),
        baseline_crps=jnp.stack(baseline_crps),
        fixed_gp_crps=jnp.stack(fixed_crps),
        fixed_gp_coverage=jnp.stack(fixed_coverage),
        joint_parameter_gp_correlations=jnp.stack(joint_correlations),
        thresholds=phx.uq.DiscrepancyIdentifiabilityThresholds(
            min_repeats=configuration.gp_repeats,
        ),
    )
    baseline_bias = jnp.abs(jnp.mean(baseline_parameters_array) - true_parameter)
    fixed_bias = jnp.abs(jnp.mean(fixed_parameters_array) - true_parameter)
    joint_bias = jnp.abs(jnp.mean(joint_parameters_array) - true_parameter)
    mean_baseline_rmse = jnp.mean(jnp.stack(baseline_rmse))
    mean_fixed_rmse = jnp.mean(jnp.stack(fixed_rmse))
    mean_sparse_rmse = jnp.mean(jnp.stack(sparse_rmse))
    mean_reused_compilation_seconds = sum(joint_map_compilation_seconds[1:]) / (
        len(joint_map_compilation_seconds) - 1
    )
    compilation_reuse_ratio = mean_reused_compilation_seconds / max(
        joint_map_compilation_seconds[0], 1e-12
    )
    metrics = {
        "baseline_parameter_bias": metric(baseline_bias, "accuracy"),
        "fixed_gp_parameter_bias": metric(
            fixed_bias, "accuracy", maximum=0.75 * float(baseline_bias)
        ),
        "joint_gp_parameter_bias": metric(
            joint_bias, "accuracy", maximum=0.75 * float(baseline_bias)
        ),
        "baseline_field_rmse": metric(mean_baseline_rmse, "accuracy"),
        "fixed_gp_field_rmse": metric(
            mean_fixed_rmse, "accuracy", maximum=0.35 * float(mean_baseline_rmse)
        ),
        "sparse_gp_field_rmse": metric(
            mean_sparse_rmse, "accuracy", maximum=0.45 * float(mean_baseline_rmse)
        ),
        "sparse_exact_mean_rmse": metric(
            jnp.mean(jnp.stack(sparse_exact_mean_delta)),
            "accuracy",
            maximum=0.02,
        ),
        "sparse_exact_variance_rmse": metric(
            jnp.mean(jnp.stack(sparse_exact_variance_delta)),
            "accuracy",
            maximum=0.004,
        ),
        "nll_improvement": metric(
            identifiability.nll_improvement, "calibration", minimum=0.0
        ),
        "crps_improvement": metric(
            identifiability.crps_improvement, "calibration", minimum=0.0
        ),
        "latent_coverage_90": metric(
            identifiability.mean_coverage, "calibration", minimum=0.85
        ),
        "max_parameter_gp_correlation": metric(
            identifiability.max_abs_parameter_gp_correlation,
            "diagnostic",
            maximum=0.95,
        ),
        "identifiability_passed": metric(
            float(identifiability.passed),
            "diagnostic",
            minimum=1.0,
            maximum=1.0,
        ),
        "sparse_storage_ratio": metric(
            sparse_storage / exact_storage,
            "performance",
            maximum=0.75,
        ),
        "exact_factor_build_seconds": metric(exact_factor_build_seconds, "performance"),
        "fitc_factor_build_seconds": metric(fitc_factor_build_seconds, "performance"),
        "exact_conditioner_build_seconds": metric(
            exact_conditioner_build_seconds, "performance"
        ),
        "fitc_conditioner_build_seconds": metric(
            fitc_conditioner_build_seconds, "performance"
        ),
        "baseline_map_stage_seconds": metric(
            sum(baseline_map_stage_seconds), "performance"
        ),
        "fixed_exact_map_stage_seconds": metric(
            sum(fixed_exact_map_stage_seconds), "performance"
        ),
        "sparse_fitc_map_stage_seconds": metric(
            sum(sparse_fitc_map_stage_seconds), "performance"
        ),
        "joint_hyperparameter_map_stage_seconds": metric(
            sum(joint_hyperparameter_map_stage_seconds), "performance"
        ),
        "joint_map_first_compilation_seconds": metric(
            joint_map_compilation_seconds[0], "performance"
        ),
        "joint_map_reused_compilation_seconds": metric(
            sum(joint_map_compilation_seconds[1:]), "performance"
        ),
        "joint_map_mean_reused_compilation_seconds": metric(
            mean_reused_compilation_seconds, "performance"
        ),
        "joint_map_compilation_reuse_ratio": metric(
            compilation_reuse_ratio,
            "performance",
            maximum=0.1,
        ),
        "joint_map_execution_seconds": metric(
            sum(joint_map_execution_seconds), "performance"
        ),
        "joint_map_max_steps": metric(max(joint_map_steps), "performance"),
        "joint_map_total_objective_evaluations": metric(
            sum(joint_map_objective_evaluations), "performance"
        ),
        "joint_map_converged_count": metric(
            len(joint_map_steps),
            "diagnostic",
            minimum=configuration.gp_repeats,
            maximum=configuration.gp_repeats,
        ),
        "joint_laplace_stage_seconds": metric(
            sum(joint_laplace_stage_seconds), "performance"
        ),
        "exact_condition_stage_seconds": metric(
            sum(exact_condition_stage_seconds), "performance"
        ),
        "fitc_condition_stage_seconds": metric(
            sum(fitc_condition_stage_seconds), "performance"
        ),
        **_performance_metrics(
            wall_seconds=time.perf_counter() - started,
            cold_seconds=cold,
            compile_seconds=compile_seconds,
            execution_seconds=execute,
            sample_memory_bytes=8 * (exact_storage + sparse_storage),
        ),
    }
    return ScenarioResult(
        name="misspecified_pde_discrepancy",
        description=misspecified_pde_discrepancy.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "equation": "-u'' = omitted source with u(0)=0 and u(1)=parameter",
            "gp_repeats": configuration.gp_repeats,
            "num_observations": int(observation_x.size),
            "num_inducing": 9,
            "identifiability_failures": list(identifiability.failures),
        },
    )


def correlated_vector_discrepancy(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Validate coherent correlated-output GP discrepancy and multivariate scoring."""
    started = time.perf_counter()
    root_key = jr.key(seed)
    observation_x = jnp.linspace(0.0, 1.0, 18)
    query_x = jnp.linspace(0.0, 1.0, 31)
    output_covariance = jnp.asarray([[1.0, -0.75], [-0.75, 1.0]])
    amplitude = 0.15
    length_scale = 0.25
    noise_scale = jnp.asarray([0.01, 0.015])

    def base(locations):
        return jnp.stack([locations, 2.0 * locations], axis=1)

    def discrepancy(locations):
        latent = amplitude * jnp.sin(2.0 * jnp.pi * locations)
        return jnp.stack([latent, -0.6 * latent], axis=1)

    observation_base = base(observation_x)
    observations = observation_base + discrepancy(observation_x)
    model = phx.uq.MultiOutputGaussianProcessDiscrepancy(
        observation_x,
        observations,
        output_covariance=output_covariance,
        output_names=("velocity", "pressure"),
        kernel="exp_squared",
    )
    cold, compile_seconds, execute = _jit_timings(
        base,
        query_x,
        repetitions=configuration.jit_warm_repetitions,
    )
    condition, condition_seconds = _timed_call(
        lambda: model.condition(
            observation_base,
            query_x,
            amplitude=amplitude,
            length_scale=length_scale,
            noise_scale=noise_scale,
            point_dim="x",
            output_dim="field",
        )
    )
    sample_count = min(configuration.posterior_prediction_samples, 1_024)
    discrepancy_samples, sampling_seconds = _timed_call(
        lambda: condition.sample(
            jr.fold_in(root_key, 1),
            num_samples=sample_count,
        )
    )
    query_base = base(query_x)
    target = query_base + discrepancy(query_x)
    predictive_samples = query_base + discrepancy_samples
    predictive_mean = query_base + condition.mean
    mean_rmse = jnp.sqrt(jnp.mean((predictive_mean - target) ** 2))
    midpoint = query_x.size // 2 + 1
    first_index = 2 * midpoint
    second_index = first_index + 1
    posterior_cross_correlation = condition.covariance[
        first_index, second_index
    ] / jnp.sqrt(
        condition.covariance[first_index, first_index]
        * condition.covariance[second_index, second_index]
    )
    empirical_cross_correlation = jnp.corrcoef(
        discrepancy_samples[:, midpoint, 0],
        discrepancy_samples[:, midpoint, 1],
    )[0, 1]
    lower_90 = jnp.quantile(predictive_samples, 0.05, axis=0)
    upper_90 = jnp.quantile(predictive_samples, 0.95, axis=0)
    lower_95 = jnp.quantile(predictive_samples, 0.025, axis=0)
    upper_95 = jnp.quantile(predictive_samples, 0.975, axis=0)
    predictive_scale = jnp.sqrt(condition.variance + noise_scale[None, :] ** 2)
    calibration = _calibration_statistics(
        mean=predictive_mean,
        scale=predictive_scale,
        truth=target,
        observation_scale=float(jnp.mean(noise_scale)),
        key=jr.fold_in(root_key, 2),
        num_cases=configuration.calibration_cases,
    )
    metrics = {
        "field_rmse": metric(mean_rmse, "accuracy", maximum=0.012),
        "latent_coverage_90": metric(
            jnp.mean((target >= lower_90) & (target <= upper_90)),
            "calibration",
            minimum=0.80,
        ),
        "latent_coverage_95": metric(
            jnp.mean((target >= lower_95) & (target <= upper_95)),
            "calibration",
            minimum=0.85,
        ),
        "posterior_cross_correlation": metric(
            posterior_cross_correlation,
            "diagnostic",
            maximum=-0.05,
        ),
        "empirical_correlation_error": metric(
            jnp.abs(empirical_cross_correlation - posterior_cross_correlation),
            "diagnostic",
            maximum=0.10,
        ),
        "energy_score": metric(
            phx.uq.energy_score(
                predictive_samples,
                target,
                chunk_size=128,
            ),
            "calibration",
            minimum=0.0,
        ),
        "condition_seconds": metric(condition_seconds, "performance", unit="s"),
        "sampling_seconds": metric(sampling_seconds, "performance", unit="s"),
        **_calibration_metrics(calibration, num_cases=configuration.calibration_cases),
        **_performance_metrics(
            wall_seconds=time.perf_counter() - started,
            cold_seconds=cold,
            compile_seconds=compile_seconds,
            execution_seconds=execute,
            sample_memory_bytes=(predictive_samples.nbytes + condition.covariance.nbytes),
        ),
    }
    return ScenarioResult(
        name="correlated_vector_discrepancy",
        description=correlated_vector_discrepancy.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "output_names": list(condition.output_names),
            "declared_output_covariance": output_covariance.tolist(),
            "num_prediction_samples": sample_count,
            "calibration_cases": configuration.calibration_cases,
        },
    )


SCENARIOS: dict[str, Scenario] = {
    "elliptic_coefficient_inverse": elliptic_coefficient_inverse,
    "nonlinear_transformed_ode": nonlinear_transformed_ode,
    "neural_selected_subspace": neural_selected_subspace,
    "multimodal_tempered_inference": multimodal_tempered_inference,
    "misspecified_pde_discrepancy": misspecified_pde_discrepancy,
    "correlated_vector_discrepancy": correlated_vector_discrepancy,
    "flow_assisted_multimodal": flow_assisted_multimodal,
}


__all__ = ["SCENARIOS"]
