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
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import optax

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

    def gp_state(amplitude, length_scale, noise_scale):
        return phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.AmplitudeKernel(
                phx.kernels.Matern32Kernel(length_scale=length_scale),
                amplitude,
            ),
            noise_scale=noise_scale,
        )

    def joint_state(parameters):
        return gp_state(
            parameters["amplitude"],
            parameters["length_scale"],
            parameters["noise_scale"],
        )

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
    fixed_state = gp_state(
        fixed_amplitude,
        fixed_length_scale,
        observation_scale,
    )
    exact_factor, exact_factor_build_seconds = _timed_call(
        lambda: phx.uq.ExactGaussianProcessFactor(
            observation_x,
            state=fixed_state,
        )
    )
    sparse_factor, fitc_factor_build_seconds = _timed_call(
        lambda: phx.uq.SparseGaussianProcessFactor(
            observation_x,
            inducing_x,
            state=fixed_state,
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
        )
        sparse_gp = phx.uq.SparseGaussianProcessDiscrepancy(
            observation_x,
            observations,
            inducing_x,
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
                    state=joint_state,
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
    """Validate heterotopic correlated-output GP discrepancy and scoring."""
    started = time.perf_counter()
    root_key = jr.key(seed)
    observation_x = jnp.linspace(0.0, 1.0, 18)
    query_x = jnp.linspace(0.0, 1.0, 31)
    output_names = ("velocity", "pressure")
    output_covariance = jnp.asarray([[1.0, -0.75], [-0.75, 1.0]])
    amplitude = 0.15
    length_scale = 0.25
    noise_scale = jnp.asarray([0.01, 0.015])
    observation_mask = jnp.ones((observation_x.size, 2), dtype=bool)
    observation_mask = observation_mask.at[1::3, 1].set(False).at[2::4, 0].set(False)

    def base(locations):
        return jnp.stack([locations, 2.0 * locations], axis=1)

    def discrepancy(locations):
        latent = amplitude * jnp.sin(2.0 * jnp.pi * locations)
        return jnp.stack([latent, -0.6 * latent], axis=1)

    observation_base = base(observation_x)
    observations = observation_base + discrepancy(observation_x)
    model = phx.uq.MultiOutputGaussianProcessDiscrepancy.from_dense(
        observation_x,
        observations,
        output_names=output_names,
        mask=observation_mask,
    )
    coregionalization = phx.uq.Coregionalization(
        jnp.linalg.cholesky(output_covariance),
        jnp.zeros((2,)),
        output_names=output_names,
    )
    state = phx.uq.MultiOutputGaussianProcessLikelihoodState(
        kernel=phx.uq.IntrinsicCoregionalizationKernel(
            phx.kernels.AmplitudeKernel(
                phx.kernels.SquaredExponentialKernel(length_scale=length_scale),
                amplitude,
            ),
            coregionalization,
        ),
        noise_scale=noise_scale,
    )
    query_design = phx.uq.MultiOutputDesign.from_dense(
        query_x,
        output_names=output_names,
    )
    cold, compile_seconds, execute = _jit_timings(
        base,
        query_x,
        repetitions=configuration.jit_warm_repetitions,
    )
    condition, condition_seconds = _timed_call(
        lambda: model.condition(
            observation_base,
            query_design,
            state=state,
        )
    )
    sample_count = min(configuration.posterior_prediction_samples, 1_024)
    discrepancy_samples, sampling_seconds = _timed_call(
        lambda: condition.sample(
            jr.fold_in(root_key, 1),
            num_samples=sample_count,
        ).reshape((sample_count, query_x.size, 2))
    )
    query_base = base(query_x)
    target = query_base + discrepancy(query_x)
    predictive_samples = query_base + discrepancy_samples
    predictive_mean = query_base + condition.dense_mean()
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
    predictive_scale = jnp.sqrt(condition.dense_variance() + noise_scale[None, :] ** 2)
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
        "heterotopic_observation_fraction": metric(
            jnp.mean(observation_mask),
            "diagnostic",
            minimum=0.5,
            maximum=0.95,
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
            "observation_mask": observation_mask.tolist(),
            "calibration_cases": configuration.calibration_cases,
        },
    )


def operator_conditioned_inverse_pde(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Infer an elliptic coefficient through value and operator observations."""
    started = time.perf_counter()
    value_points = jnp.linspace(0.0, 1.0, 8)
    operator_points = jnp.linspace(0.05, 0.95, 10)
    query_points = jnp.linspace(0.0, 1.0, 41)
    true_diffusion = 1.7
    wrong_diffusion = jnp.asarray(1.0)

    def field(points):
        return jnp.sin(jnp.pi * points)

    forcing = true_diffusion * jnp.pi**2 * field(operator_points)
    value = phx.uq.value_functional(1)
    laplacian = phx.uq.laplacian_functional(1)
    state = phx.uq.FunctionalGaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(
            phx.kernels.SquaredExponentialKernel(length_scale=0.25),
            1.0,
        ),
        noise_scale=jnp.asarray([0.005, 0.02]),
    )

    def discrepancy(diffusion):
        return phx.uq.FunctionalGaussianProcessDiscrepancy(
            (
                phx.uq.FunctionalObservationBlock(
                    value_points,
                    value,
                    name="field-values",
                ),
                phx.uq.FunctionalObservationBlock(
                    operator_points,
                    -diffusion * laplacian,
                    name="elliptic-operator",
                ),
            ),
            (field(value_points), forcing),
        )

    zero_mean = (
        jnp.zeros_like(value_points),
        jnp.zeros_like(operator_points),
    )

    def log_likelihood(diffusion):
        return discrepancy(diffusion).log_marginal_likelihood(
            zero_mean,
            state=state,
        )

    cold, compile_seconds, execute = _jit_timings(
        log_likelihood,
        jnp.asarray(true_diffusion),
        repetitions=configuration.jit_warm_repetitions,
    )
    candidates = jnp.linspace(0.7, 2.7, 41)
    scores = jax.vmap(log_likelihood)(candidates)
    selected_diffusion = candidates[jnp.argmax(scores)]
    query_design = phx.uq.FunctionalDesign.from_points(
        query_points,
        value,
        name="query-values",
    )
    condition, condition_seconds = _timed_call(
        lambda: discrepancy(selected_diffusion).condition(
            zero_mean,
            query_design,
            state=state,
        )
    )
    field_rmse = jnp.sqrt(jnp.mean((condition.mean - field(query_points)) ** 2))
    score_margin = log_likelihood(true_diffusion) - log_likelihood(wrong_diffusion)
    wrong_gradient = jax.grad(log_likelihood)(wrong_diffusion)
    metrics = {
        "diffusion_absolute_error": metric(
            jnp.abs(selected_diffusion - true_diffusion),
            "accuracy",
            maximum=0.051,
        ),
        "field_reconstruction_rmse": metric(
            field_rmse,
            "accuracy",
            maximum=1.0e-3,
        ),
        "true_wrong_log_likelihood_margin": metric(
            score_margin,
            "diagnostic",
            minimum=1.0,
        ),
        "wrong_diffusion_gradient": metric(
            wrong_gradient,
            "diagnostic",
            minimum=0.0,
        ),
        "minimum_query_variance": metric(
            jnp.min(condition.variance),
            "diagnostic",
            minimum=0.0,
        ),
        "condition_seconds": metric(condition_seconds, "performance", unit="s"),
        **_performance_metrics(
            wall_seconds=time.perf_counter() - started,
            cold_seconds=cold,
            compile_seconds=compile_seconds,
            execution_seconds=execute,
            sample_memory_bytes=condition.covariance.nbytes,
        ),
    }
    return ScenarioResult(
        name="operator_conditioned_inverse_pde",
        description=operator_conditioned_inverse_pde.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "true_diffusion": true_diffusion,
            "selected_diffusion": float(selected_diffusion),
            "num_value_observations": int(value_points.size),
            "num_operator_observations": int(operator_points.size),
            "operator": "-diffusion * Laplacian",
        },
    )


class _BenchmarkFeatureMap(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __call__(self, point):
        return jnp.tanh(self.weight @ point + self.bias)


def deep_kernel_likelihood_timing(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Measure learned-feature gradients and fixed-factor likelihood reuse."""
    started = time.perf_counter()
    num_observations = 48 if configuration.profile == "smoke" else 128
    coordinate = jnp.linspace(-1.0, 1.0, num_observations)
    points = jnp.stack((coordinate, coordinate**2), axis=1)
    observations = 0.7 * coordinate + 0.12 * jnp.sin(3.0 * coordinate)
    physical_mean = 0.7 * coordinate
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    weight_key, bias_key = jr.split(jr.key(seed))
    feature_map = _BenchmarkFeatureMap(
        0.5 * jr.normal(weight_key, (4, 2)),
        0.1 * jr.normal(bias_key, (4,)),
    )

    def deep_state(candidate):
        return phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.AmplitudeKernel(
                phx.kernels.InputTransformedKernel(
                    phx.kernels.SquaredExponentialKernel(length_scale=jnp.ones((4,))),
                    candidate,
                    transform_id="benchmark-learned-features",
                    max_derivative_order=None,
                ),
                0.2,
            ),
            noise_scale=0.02,
        )

    def deep_objective(candidate):
        return model.log_marginal_likelihood(
            physical_mean,
            state=deep_state(candidate),
        )

    def stationary_objective(log_length_scale):
        state = phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.AmplitudeKernel(
                phx.kernels.SquaredExponentialKernel(
                    length_scale=jnp.exp(log_length_scale)
                ),
                0.2,
            ),
            noise_scale=0.02,
        )
        return model.log_marginal_likelihood(physical_mean, state=state)

    fixed_state = deep_state(feature_map)
    factor, factor_build_seconds = _timed_call(lambda: model.factor(state=fixed_state))
    deep_rebuild_timing = _jit_timings(
        lambda mean: model.log_marginal_likelihood(mean, state=fixed_state),
        physical_mean,
        repetitions=configuration.jit_warm_repetitions,
    )
    deep_reuse_timing = _jit_timings(
        lambda mean: factor.log_probability(model.residual(mean)),
        physical_mean,
        repetitions=configuration.jit_warm_repetitions,
    )
    deep_gradient_timing = _jit_timings(
        jax.value_and_grad(deep_objective),
        feature_map,
        repetitions=configuration.jit_warm_repetitions,
    )
    stationary_gradient_timing = _jit_timings(
        jax.value_and_grad(stationary_objective),
        jnp.log(jnp.asarray([0.4, 0.4])),
        repetitions=configuration.jit_warm_repetitions,
    )
    deep_value, deep_gradient = jax.value_and_grad(deep_objective)(feature_map)
    gradient_leaves = jax.tree.leaves(deep_gradient)
    gradient_norm = jnp.sqrt(sum(jnp.vdot(leaf, leaf) for leaf in gradient_leaves))
    query_coordinate = jnp.linspace(-0.95, 0.95, 33)
    query = jnp.stack((query_coordinate, query_coordinate**2), axis=1)
    condition, condition_seconds = _timed_call(
        lambda: model.condition(
            physical_mean,
            query,
            state=fixed_state,
            output_dim="query",
        )
    )
    condition_finite = jnp.all(
        jnp.isfinite(condition.mean)
        & jnp.isfinite(condition.variance)
        & (condition.variance >= 0.0)
    )
    metrics = {
        "wall_seconds": metric(
            time.perf_counter() - started,
            "performance",
            unit="s",
        ),
        "deep_kernel_factor_build_seconds": metric(
            factor_build_seconds,
            "performance",
            unit="s",
        ),
        "deep_kernel_rebuild_compile_seconds": metric(
            deep_rebuild_timing[1],
            "performance",
            unit="s",
        ),
        "deep_kernel_rebuild_warm_seconds": metric(
            deep_rebuild_timing[2],
            "performance",
            unit="s",
        ),
        "deep_kernel_reuse_warm_seconds": metric(
            deep_reuse_timing[2],
            "performance",
            unit="s",
        ),
        "deep_kernel_factor_reuse_speedup": metric(
            deep_rebuild_timing[2] / deep_reuse_timing[2],
            "performance",
            description="Warm learned-feature likelihood speedup from factor reuse.",
        ),
        "deep_kernel_gradient_compile_seconds": metric(
            deep_gradient_timing[1],
            "performance",
            unit="s",
        ),
        "deep_kernel_gradient_warm_seconds": metric(
            deep_gradient_timing[2],
            "performance",
            unit="s",
        ),
        "stationary_gradient_warm_seconds": metric(
            stationary_gradient_timing[2],
            "performance",
            unit="s",
        ),
        "deep_stationary_gradient_time_ratio": metric(
            deep_gradient_timing[2] / stationary_gradient_timing[2],
            "performance",
        ),
        "condition_seconds": metric(condition_seconds, "performance", unit="s"),
        "deep_log_likelihood": metric(deep_value, "diagnostic"),
        "feature_gradient_norm": metric(
            gradient_norm,
            "diagnostic",
            minimum=1.0e-12,
        ),
        "finite_condition": metric(
            condition_finite,
            "diagnostic",
            minimum=1.0,
            maximum=1.0,
        ),
        "minimum_query_variance": metric(
            jnp.min(condition.variance),
            "diagnostic",
            minimum=0.0,
        ),
    }
    return ScenarioResult(
        name="deep_kernel_likelihood_timing",
        description=deep_kernel_likelihood_timing.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "num_observations": num_observations,
            "input_dimension": 2,
            "feature_dimension": 4,
            "feature_parameter_count": sum(
                int(leaf.size) for leaf in jax.tree.leaves(feature_map)
            ),
            "factor_storage_elements": factor.factor_storage_elements,
        },
    )


def _small_deep_ensemble_predictions(
    train_inputs: jax.Array,
    train_targets: jax.Array,
    prediction_inputs: jax.Array,
    /,
    *,
    key: jax.Array,
    num_members: int,
    num_steps: int,
) -> jax.Array:
    """Fit independent bootstrapped nonlinear regressors for a robust baseline."""
    optimizer = optax.adam(1.0e-2)

    def predict(parameters, inputs):
        hidden = jnp.tanh(
            inputs[:, None] * parameters["input_weight"][None, :]
            + parameters["hidden_bias"][None, :]
        )
        return hidden @ parameters["output_weight"] + parameters["output_bias"]

    def loss(parameters, inputs, targets):
        residual = predict(parameters, inputs) - targets
        return jnp.mean(residual**2)

    @jax.jit
    def train_step(parameters, optimizer_state, inputs, targets):
        gradients = jax.grad(loss)(parameters, inputs, targets)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            parameters,
        )
        return (
            optax.apply_updates(parameters, updates),
            next_optimizer_state,
        )

    predictions = []
    member_keys = jr.split(key, int(num_members))
    for member_key in member_keys:
        bootstrap_key, input_key, output_key = jr.split(member_key, 3)
        bootstrap = jr.randint(
            bootstrap_key,
            train_inputs.shape,
            0,
            train_inputs.shape[0],
        )
        member_inputs = train_inputs[bootstrap]
        member_targets = train_targets[bootstrap]
        parameters = {
            "input_weight": 0.25 * jr.normal(input_key, (8,)),
            "hidden_bias": jnp.zeros((8,)),
            "output_weight": 0.25 * jr.normal(output_key, (8,)),
            "output_bias": jnp.asarray(0.0),
        }
        optimizer_state = optimizer.init(parameters)
        for _ in range(int(num_steps)):
            parameters, optimizer_state = train_step(
                parameters,
                optimizer_state,
                member_inputs,
                member_targets,
            )
        predictions.append(predict(parameters, prediction_inputs))
    return jnp.stack(predictions)


def stochastic_gradient_regression(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Benchmark fixed-step SG-MCMC against exact Gaussian and ensemble references."""
    started = time.perf_counter()
    root_key = jr.key(seed)
    observation_key = jr.fold_in(root_key, 1)
    num_factors = 128
    inputs = jnp.linspace(-1.5, 1.5, num_factors)
    design = jnp.stack((jnp.ones_like(inputs), inputs), axis=1)
    true_position = jnp.asarray([0.4, -0.7])
    observation_scale = 0.6
    observations = design @ true_position + observation_scale * jr.normal(
        observation_key, (num_factors,)
    )
    prior_scale = 2.0
    precision = jnp.eye(2) / prior_scale**2 + design.T @ design / observation_scale**2
    analytic_covariance = jnp.linalg.inv(precision)
    analytic_mean = analytic_covariance @ (design.T @ observations / observation_scale**2)
    analytic_scale = jnp.sqrt(jnp.diag(analytic_covariance))

    def position_from_vector(vector):
        return {
            "offset": vector[0],
            "nested": {"positive_rate": vector[1]},
        }

    def vector_from_physical(parameters):
        return jnp.stack(
            (
                parameters["offset"],
                jnp.log(parameters["nested"]["positive_rate"]),
            )
        )

    parameter_space = phx.uq.ParameterSpace(
        position_from_vector(jnp.zeros((2,))),
        priors={
            "offset": phx.uq.Normal(0.0, prior_scale),
            "nested": {"positive_rate": phx.uq.LogNormal(0.0, prior_scale)},
        },
        bijectors={
            "offset": phx.uq.IdentityBijector(),
            "nested": {"positive_rate": phx.uq.ExpBijector()},
        },
    )
    source = phx.uq.ArrayMinibatchSource(
        {"input": inputs, "target": observations},
        batch_size=configuration.sgmcmc_batch_size,
        seed=seed + 1,
    )

    def likelihood_factors(parameters, batch):
        vector = vector_from_physical(parameters)
        prediction = vector[0] + vector[1] * batch.data["input"]
        return -0.5 * ((batch.data["target"] - prediction) / observation_scale) ** 2

    def full_log_likelihood(parameters):
        vector = vector_from_physical(parameters)
        return jnp.sum(-0.5 * ((observations - design @ vector) / observation_scale) ** 2)

    problem = phx.uq.MinibatchPosteriorProblem(
        parameter_space,
        likelihood_factors,
        num_factors=num_factors,
        full_log_likelihood=full_log_likelihood,
    )
    exact_problem = phx.uq.PosteriorProblem(
        parameter_space,
        full_log_likelihood,
    )
    diagnostics = phx.uq.diagnose_minibatch_posterior(problem, source)
    center = position_from_vector(analytic_mean)
    control, control_seconds = _timed_call(
        lambda: phx.uq.build_sgmcmc_control_variate(
            problem,
            source,
            center,
        )
    )
    first_batch = next(source.epoch(0))
    cold, compile_seconds, execute = _jit_timings(
        lambda position, batch: jax.value_and_grad(problem.log_density_estimate)(
            position, batch
        ),
        problem.initial_position,
        first_batch,
        repetitions=configuration.jit_warm_repetitions,
    )

    step_size = 5.0e-4
    sgld, sgld_seconds = _timed_call(
        lambda: phx.uq.sample_sgld(
            problem,
            source,
            key=jr.fold_in(root_key, 2),
            step_size=step_size,
            num_chains=configuration.num_chains,
            num_burnin=configuration.sgmcmc_burnin,
            num_samples=configuration.sgmcmc_draws,
            steps_per_sample=configuration.sgmcmc_steps_per_sample,
            chain_method="vectorized",
        )
    )
    refined, refined_seconds = _timed_call(
        lambda: phx.uq.sample_sgld(
            problem,
            source,
            key=jr.fold_in(root_key, 3),
            step_size=step_size / 2.0,
            num_chains=configuration.num_chains,
            num_burnin=configuration.sgmcmc_burnin,
            num_samples=configuration.sgmcmc_draws,
            steps_per_sample=configuration.sgmcmc_steps_per_sample,
            chain_method="vectorized",
        )
    )
    controlled, controlled_seconds = _timed_call(
        lambda: phx.uq.sample_sgld(
            problem,
            source,
            key=jr.fold_in(root_key, 4),
            step_size=step_size / 2.0,
            control_variate=control,
            num_chains=configuration.num_chains,
            num_burnin=configuration.sgmcmc_burnin,
            num_samples=configuration.sgmcmc_draws,
            steps_per_sample=configuration.sgmcmc_steps_per_sample,
            chain_method="vectorized",
        )
    )
    sgnht, sgnht_seconds = _timed_call(
        lambda: phx.uq.sample_sgnht(
            problem,
            source,
            key=jr.fold_in(root_key, 5),
            step_size=5.0e-3,
            diffusion=0.1,
            control_variate=control,
            num_chains=configuration.num_chains,
            num_burnin=configuration.sgmcmc_burnin,
            num_samples=configuration.sgmcmc_draws,
            steps_per_sample=configuration.sgmcmc_steps_per_sample,
            chain_method="vectorized",
        )
    )
    nuts, nuts_seconds = _timed_call(
        lambda: phx.uq.sample_nuts(
            exact_problem,
            key=jr.fold_in(root_key, 6),
            num_chains=configuration.num_chains,
            num_warmup=configuration.num_warmup,
            num_samples=configuration.num_draws,
            initial_step_size=0.05,
            target_acceptance_rate=0.9,
            max_num_doublings=7,
            chain_method="vectorized",
        )
    )
    laplace, laplace_seconds = _timed_call(
        lambda: phx.uq.fit_laplace(
            exact_problem,
            center,
            stationarity_tolerance=1.0e-6,
        )
    )

    prediction_inputs = jnp.linspace(-1.75, 1.75, 81)
    prediction_design = jnp.stack(
        (jnp.ones_like(prediction_inputs), prediction_inputs),
        axis=1,
    )
    analytic_prediction_mean = prediction_design @ analytic_mean
    analytic_prediction_variance = jnp.einsum(
        "ni,ij,nj->n",
        prediction_design,
        analytic_covariance,
        prediction_design,
    )
    ensemble_members = 4 if configuration.profile == "smoke" else 8
    ensemble_steps = 300 if configuration.profile == "smoke" else 800
    ensemble_predictions, ensemble_seconds = _timed_call(
        lambda: _small_deep_ensemble_predictions(
            inputs,
            observations,
            prediction_inputs,
            key=jr.fold_in(root_key, 7),
            num_members=ensemble_members,
            num_steps=ensemble_steps,
        )
    )

    def sample_matrix(result):
        return jnp.stack(
            (
                result.unconstrained_samples["offset"].reshape(-1),
                result.unconstrained_samples["nested"]["positive_rate"].reshape(-1),
            ),
            axis=1,
        )

    matrices = {
        "sgld": sample_matrix(sgld),
        "sgld_refined": sample_matrix(refined),
        "sgld_control_variate": sample_matrix(controlled),
        "sgnht": sample_matrix(sgnht),
        "nuts": sample_matrix(nuts),
    }
    sample_means = {name: jnp.mean(values, axis=0) for name, values in matrices.items()}
    sample_covariances = {
        name: jnp.cov(values, rowvar=False) for name, values in matrices.items()
    }

    max_rhat, min_ess = _convergence_limits(configuration)
    metrics: dict[str, Metric] = {
        "full_density_reconstruction": metric(
            int(
                diagnostics.full_log_density_matches is True
                and diagnostics.full_gradient_matches is True
            ),
            "diagnostic",
            minimum=1.0,
        ),
        "source_epoch_factor_fraction": metric(
            diagnostics.epoch_active_factor_count / num_factors,
            "diagnostic",
            minimum=1.0,
            maximum=1.0,
        ),
        "control_variate_construction_seconds": metric(
            control_seconds,
            "performance",
            unit="s",
        ),
        "control_variate_gradient_evaluations": metric(
            control.construction_gradient_evaluations,
            "performance",
        ),
        "sgld_seconds": metric(sgld_seconds, "performance", unit="s"),
        "sgld_refined_seconds": metric(
            refined_seconds,
            "performance",
            unit="s",
        ),
        "sgld_control_variate_seconds": metric(
            controlled_seconds,
            "performance",
            unit="s",
        ),
        "sgnht_seconds": metric(sgnht_seconds, "performance", unit="s"),
        "nuts_seconds": metric(nuts_seconds, "performance", unit="s"),
        "laplace_seconds": metric(laplace_seconds, "performance", unit="s"),
        "deep_ensemble_seconds": metric(
            ensemble_seconds,
            "performance",
            unit="s",
        ),
        "sgld_samples_per_second": metric(
            sgld.samples_per_second,
            "performance",
            unit="sample/s",
        ),
        "sgld_updates_per_second": metric(
            sgld.updates_per_second,
            "performance",
            unit="update/s",
        ),
        "sgnht_samples_per_second": metric(
            sgnht.samples_per_second,
            "performance",
            unit="sample/s",
        ),
        "sgld_gradient_evaluations_per_second": metric(
            sgld.gradient_evaluations_per_second,
            "performance",
            unit="gradient/s",
        ),
        "sgld_sample_memory_bytes": metric(
            sgld.sample_memory_bytes,
            "performance",
            unit="byte",
        ),
        "sgnht_sample_memory_bytes": metric(
            sgnht.sample_memory_bytes,
            "performance",
            unit="byte",
        ),
        "batch_memory_fraction": metric(
            sum(
                int(jnp.asarray(leaf).nbytes)
                for leaf in jax.tree_util.tree_leaves(first_batch.data)
            )
            / sum(
                int(jnp.asarray(leaf).nbytes)
                for leaf in jax.tree_util.tree_leaves(source.data)
            ),
            "performance",
        ),
    }

    accuracy_gates = {
        "sgld": (0.75, 0.75),
        "sgld_refined": (0.65, 0.65),
        "sgld_control_variate": (0.65, 0.65),
        "sgnht": (0.65, 0.75),
        "nuts": (0.55, 0.55),
    }
    chain_results = {
        "sgld": sgld,
        "sgld_refined": refined,
        "sgld_control_variate": controlled,
        "sgnht": sgnht,
    }
    covariance_norm = jnp.linalg.norm(analytic_covariance)
    for name, values in matrices.items():
        mean_error = jnp.sqrt(
            jnp.mean(((sample_means[name] - analytic_mean) / analytic_scale) ** 2)
        )
        covariance_error = (
            jnp.linalg.norm(sample_covariances[name] - analytic_covariance)
            / covariance_norm
        )
        prediction_mean = prediction_design @ sample_means[name]
        prediction_variance = jnp.einsum(
            "ni,ij,nj->n",
            prediction_design,
            sample_covariances[name],
            prediction_design,
        )
        mean_gate, covariance_gate = accuracy_gates[name]
        metrics[f"{name}_posterior_mean_standardized_rmse"] = metric(
            mean_error,
            "accuracy",
            maximum=mean_gate,
        )
        metrics[f"{name}_covariance_relative_error"] = metric(
            covariance_error,
            "accuracy",
            maximum=covariance_gate,
        )
        metrics[f"{name}_predictive_mean_rmse"] = metric(
            jnp.sqrt(jnp.mean((prediction_mean - analytic_prediction_mean) ** 2)),
            "accuracy",
            maximum=0.08,
        )
        metrics[f"{name}_predictive_variance_relative_rmse"] = metric(
            jnp.sqrt(jnp.mean((prediction_variance - analytic_prediction_variance) ** 2))
            / jnp.mean(analytic_prediction_variance),
            "accuracy",
            maximum=0.8,
        )
        if name in chain_results:
            result = chain_results[name]
            metrics[f"{name}_max_rhat"] = metric(
                result.diagnostics.max_rhat,
                "convergence",
                maximum=(
                    1.12
                    if name == "sgnht" and configuration.profile == "smoke"
                    else (1.05 if name == "sgnht" else max_rhat)
                ),
            )
            metrics[f"{name}_min_ess"] = metric(
                jnp.minimum(
                    result.diagnostics.min_bulk_ess,
                    result.diagnostics.min_tail_ess,
                ),
                "convergence",
                minimum=min_ess,
            )
            metrics[f"{name}_ess_per_second"] = metric(
                jnp.minimum(
                    result.diagnostics.min_bulk_ess,
                    result.diagnostics.min_tail_ess,
                )
                / max(result.sampling_duration_seconds, 1.0e-12),
                "performance",
                unit="ESS/s",
            )

    ordinary_gradient = jax.grad(problem.log_density_estimate)
    probe = position_from_vector(
        analytic_mean + jnp.asarray([0.2, -0.2]) * analytic_scale
    )
    epoch_batches = tuple(source.epoch(0))

    def gradient_vector(gradient):
        return jnp.stack(
            (
                gradient["offset"],
                gradient["nested"]["positive_rate"],
            )
        )

    ordinary_gradients = jnp.stack(
        [gradient_vector(ordinary_gradient(probe, batch)) for batch in epoch_batches]
    )
    controlled_gradients = jnp.stack(
        [
            gradient_vector(
                jax.tree_util.tree_map(
                    lambda full, current, at_center: full + current - at_center,
                    control.full_gradient,
                    ordinary_gradient(probe, batch),
                    ordinary_gradient(control.center, batch),
                )
            )
            for batch in epoch_batches
        ]
    )
    ordinary_variance = jnp.sum(jnp.var(ordinary_gradients, axis=0))
    controlled_variance = jnp.sum(jnp.var(controlled_gradients, axis=0))
    metrics["control_variate_gradient_variance_reduction"] = metric(
        ordinary_variance / jnp.maximum(controlled_variance, jnp.finfo(float).eps),
        "diagnostic",
        minimum=10.0,
    )
    metrics["step_halving_mean_discrepancy"] = metric(
        jnp.linalg.norm(sample_means["sgld"] - sample_means["sgld_refined"])
        / jnp.linalg.norm(analytic_scale),
        "diagnostic",
    )
    metrics["step_halving_covariance_discrepancy"] = metric(
        jnp.linalg.norm(sample_covariances["sgld"] - sample_covariances["sgld_refined"])
        / covariance_norm,
        "diagnostic",
    )
    metrics["laplace_mean_error"] = metric(
        jnp.linalg.norm(
            jnp.asarray(
                [
                    laplace.map_position["offset"],
                    laplace.map_position["nested"]["positive_rate"],
                ]
            )
            - analytic_mean
        ),
        "accuracy",
        maximum=1.0e-8,
    )
    laplace_order = jnp.asarray([1, 0])
    laplace_covariance = laplace.covariance[jnp.ix_(laplace_order, laplace_order)]
    metrics["laplace_covariance_relative_error"] = metric(
        jnp.linalg.norm(laplace_covariance - analytic_covariance) / covariance_norm,
        "accuracy",
        maximum=1.0e-8,
    )
    ensemble_mean = jnp.mean(ensemble_predictions, axis=0)
    ensemble_variance = jnp.var(ensemble_predictions, axis=0)
    metrics["deep_ensemble_predictive_mean_rmse"] = metric(
        jnp.sqrt(jnp.mean((ensemble_mean - analytic_prediction_mean) ** 2)),
        "accuracy",
        maximum=0.2,
    )
    metrics["deep_ensemble_predictive_variance_relative_rmse"] = metric(
        jnp.sqrt(jnp.mean((ensemble_variance - analytic_prediction_variance) ** 2))
        / jnp.mean(analytic_prediction_variance),
        "accuracy",
    )
    metrics.update(
        _performance_metrics(
            wall_seconds=time.perf_counter() - started,
            cold_seconds=cold,
            compile_seconds=compile_seconds,
            execution_seconds=execute,
            sample_memory_bytes=sum(
                result.sample_memory_bytes
                for result in (sgld, refined, controlled, sgnht)
            ),
        )
    )
    return ScenarioResult(
        name="stochastic_gradient_regression",
        description=stochastic_gradient_regression.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "num_factors": num_factors,
            "batch_size": configuration.sgmcmc_batch_size,
            "num_chains": configuration.num_chains,
            "num_burnin": configuration.sgmcmc_burnin,
            "num_draws": configuration.sgmcmc_draws,
            "steps_per_sample": configuration.sgmcmc_steps_per_sample,
            "sgld_step_size": step_size,
            "sgnht_step_size": 5.0e-3,
            "sgnht_diffusion": 0.1,
            "approximation": "unadjusted_fixed_step",
            "source_fingerprint": source.fingerprint,
            "analytic_mean": analytic_mean.tolist(),
            "analytic_covariance": analytic_covariance.tolist(),
            "parameter_structure": "nested_with_positive_log_coordinate",
            "deep_ensemble_members": ensemble_members,
            "deep_ensemble_steps": ensemble_steps,
        },
    )


def linearized_uncertainty_propagation(
    configuration: BenchmarkConfiguration,
    seed: int,
) -> ScenarioResult:
    """Validate matrix-free local covariance propagation and normalized EIV inference."""
    started = time.perf_counter()
    root_key = jr.key(seed)
    (
        affine_key,
        qmc_key,
        data_key,
        low_rank_key,
        output_key,
        probe_key,
    ) = jr.split(root_key, 6)

    affine_input_dimension = 6
    affine_output_dimension = 4
    affine_matrix = jr.normal(
        affine_key,
        (affine_output_dimension, affine_input_dimension),
    ) / jnp.sqrt(affine_input_dimension)
    affine_center = jnp.linspace(-0.4, 0.6, affine_input_dimension)
    affine_variance = jnp.linspace(0.2, 1.1, affine_input_dimension)
    affine_covariance = jnp.diag(affine_variance)
    affine_expected = affine_matrix @ affine_covariance @ affine_matrix.T
    affine_representations = (
        phx.uq.DiagonalCovariance(affine_variance),
        phx.uq.DenseCovariance(affine_covariance),
        phx.uq.FactorCovariance(jnp.diag(jnp.sqrt(affine_variance))),
        phx.uq.CovarianceOperator(lambda vector: affine_covariance @ vector),
    )
    affine_results = tuple(
        phx.uq.propagate_linearized(
            lambda value: affine_matrix @ value,
            affine_center,
            covariance,
        )
        for covariance in affine_representations
    )
    affine_errors = tuple(
        jnp.linalg.norm(result.materialize_covariance().matrix - affine_expected)
        / jnp.linalg.norm(affine_expected)
        for result in affine_results
    )
    affine_hutchinson = affine_results[-1].estimate_variance(
        jr.fold_in(root_key, 1),
        num_probes=configuration.linearized_hutchinson_probes,
        batch_size=min(configuration.linearized_hutchinson_probes, 128),
    )
    affine_hutchinson_error = jnp.linalg.norm(
        affine_hutchinson.variance - jnp.diag(affine_expected)
    ) / jnp.linalg.norm(jnp.diag(affine_expected))

    common_effect = phx.uq.propagate_linearized(
        lambda value: jnp.asarray([value[0] + value[2], value[1] + value[2]]),
        jnp.zeros(3),
        phx.uq.DiagonalCovariance(jnp.asarray([1.0, 1.0, 9.0])),
    )
    common_effect_expected = jnp.asarray([[10.0, 9.0], [9.0, 10.0]])
    common_effect_error = jnp.linalg.norm(
        common_effect.materialize_covariance().matrix - common_effect_expected
    ) / jnp.linalg.norm(common_effect_expected)

    nonlinear_center = jnp.asarray([0.6, -0.4])
    nonlinear_base_covariance = jnp.asarray([[1.0, 0.35], [0.35, 0.8]])

    def nonlinear_map(value):
        return jnp.asarray(
            [
                jnp.sin(value[0]) + 0.1 * value[1] ** 2,
                jnp.exp(0.2 * value[0] - 0.1 * value[1]),
                value[0] * value[1],
            ]
        )

    unit_design = phx.sampling.materialize_design(
        phx.sampling.SobolDesign(scrambled=True),
        count=configuration.linearized_qmc_samples,
        dimension=2,
        key=qmc_key,
    )
    epsilon = jnp.finfo(unit_design.dtype).eps
    normal_design = jsp.special.ndtri(jnp.clip(unit_design, epsilon, 1.0 - epsilon))
    nonlinear_scales = (0.25, 0.125, 0.0625)
    nonlinear_covariance_errors: list[float] = []
    nonlinear_mean_errors: list[float] = []
    for scale in nonlinear_scales:
        covariance = scale**2 * nonlinear_base_covariance
        linearized = phx.uq.propagate_linearized(
            nonlinear_map,
            nonlinear_center,
            phx.uq.DenseCovariance(covariance),
        )
        samples = nonlinear_center + normal_design @ jnp.linalg.cholesky(covariance).T
        outputs = jax.vmap(nonlinear_map)(samples)
        qmc_mean = jnp.mean(outputs, axis=0)
        qmc_covariance = jnp.cov(outputs, rowvar=False)
        linearized_covariance = linearized.materialize_covariance().matrix
        nonlinear_covariance_errors.append(
            float(
                jnp.linalg.norm(linearized_covariance - qmc_covariance)
                / jnp.linalg.norm(qmc_covariance)
            )
        )
        nonlinear_mean_errors.append(float(jnp.linalg.norm(linearized.mean - qmc_mean)))

    true_slope = 2.0
    input_scale = 0.5
    observation_scale = 0.12
    latent_inputs = jnp.linspace(-2.0, 2.0, 12)
    input_noise_key, observation_noise_key = jr.split(data_key)
    measured_inputs = latent_inputs + input_scale * jr.normal(
        input_noise_key,
        latent_inputs.shape,
    )
    measured_targets = true_slope * latent_inputs + observation_scale * jr.normal(
        observation_noise_key,
        latent_inputs.shape,
    )
    measurement_term = phx.uq.LinearizedGaussianMeasurementLikelihood(
        lambda slope, value: slope * value[0],
        measured_inputs[:, None],
        measured_targets,
        input_covariance=jnp.asarray([[input_scale**2]]),
        observation_covariance=jnp.asarray([[observation_scale**2]]),
    )
    slope_grid = jnp.linspace(0.5, 3.0, 1_024)
    eiv_log_likelihood = jax.vmap(measurement_term.log_prob)(slope_grid)

    def latent_log_likelihood(slope):
        effective_scale = jnp.sqrt(observation_scale**2 + slope**2 * input_scale**2)
        residual = measured_targets - slope * measured_inputs
        return jnp.sum(
            -0.5 * (residual / effective_scale) ** 2
            - jnp.log(effective_scale * jnp.sqrt(2.0 * jnp.pi))
        )

    latent_reference_log_likelihood = jax.vmap(latent_log_likelihood)(slope_grid)
    prior_log_density = -0.5 * (slope_grid / 3.0) ** 2
    eiv_weights = jax.nn.softmax(eiv_log_likelihood + prior_log_density)
    latent_weights = jax.nn.softmax(latent_reference_log_likelihood + prior_log_density)
    ordinary_log_likelihood = jnp.sum(
        -0.5
        * (
            (measured_targets[None, :] - slope_grid[:, None] * measured_inputs[None, :])
            / observation_scale
        )
        ** 2,
        axis=1,
    )
    ordinary_weights = jax.nn.softmax(ordinary_log_likelihood + prior_log_density)
    eiv_mean = jnp.sum(eiv_weights * slope_grid)
    latent_mean = jnp.sum(latent_weights * slope_grid)
    ordinary_mean = jnp.sum(ordinary_weights * slope_grid)
    eiv_variance = jnp.sum(eiv_weights * (slope_grid - eiv_mean) ** 2)
    latent_variance = jnp.sum(latent_weights * (slope_grid - latent_mean) ** 2)

    input_dimension = configuration.linearized_input_dimension
    output_dimension = configuration.linearized_output_dimension
    factor_rank = min(configuration.linearized_factor_rank, input_dimension)
    factors = jr.normal(low_rank_key, (factor_rank, input_dimension)) / jnp.sqrt(
        factor_rank
    )
    output_matrix = jr.normal(
        output_key,
        (output_dimension, input_dimension),
    ) / jnp.sqrt(input_dimension)
    low_rank = phx.uq.propagate_linearized(
        lambda value: output_matrix @ value,
        jnp.zeros(input_dimension),
        phx.uq.FactorCovariance(factors),
    )
    propagated_factors = output_matrix @ factors.T
    expected_variance = jnp.sum(propagated_factors**2, axis=1)
    observed_variance = low_rank.exact_variance(batch_size=max(1, factor_rank // 2))
    operator_low_rank = phx.uq.propagate_linearized(
        lambda value: output_matrix @ value,
        jnp.zeros(input_dimension),
        phx.uq.CovarianceOperator(lambda vector: factors.T @ (factors @ vector)),
    )
    output_probe = jr.normal(probe_key, (output_dimension,))
    expected_action = propagated_factors @ (propagated_factors.T @ output_probe)
    observed_action = operator_low_rank.covariance_vector_product(output_probe)
    low_rank_variance_error = jnp.linalg.norm(
        observed_variance - expected_variance
    ) / jnp.linalg.norm(expected_variance)
    low_rank_action_error = jnp.linalg.norm(
        observed_action - expected_action
    ) / jnp.linalg.norm(expected_action)
    hutchinson = operator_low_rank.estimate_variance(
        jr.fold_in(root_key, 2),
        num_probes=configuration.linearized_hutchinson_probes,
        batch_size=min(configuration.linearized_hutchinson_probes, 64),
    )
    low_rank_hutchinson_error = jnp.linalg.norm(
        hutchinson.variance - expected_variance
    ) / jnp.linalg.norm(expected_variance)
    low_rank_hutchinson_normalized_error = jnp.linalg.norm(
        hutchinson.variance - expected_variance
    ) / jnp.maximum(
        jnp.linalg.norm(hutchinson.standard_error),
        jnp.finfo(float).eps,
    )
    cold, compile_seconds, execute = _jit_timings(
        lambda vector: operator_low_rank.covariance_vector_product(vector),
        output_probe,
        repetitions=configuration.jit_warm_repetitions,
    )
    represented_memory = int(factors.nbytes)
    dense_output_memory = output_dimension * output_dimension * jnp.dtype(float).itemsize

    metrics = {
        "affine_representation_relative_error": metric(
            max(float(value) for value in affine_errors),
            "accuracy",
            maximum=1.0e-11,
        ),
        "affine_hutchinson_relative_error": metric(
            affine_hutchinson_error,
            "accuracy",
            maximum=0.15,
        ),
        "jcgm_common_effect_relative_error": metric(
            common_effect_error,
            "accuracy",
            maximum=1.0e-12,
        ),
        "nonlinear_qmc_covariance_relative_error": metric(
            nonlinear_covariance_errors[-1],
            "accuracy",
            maximum=0.08,
        ),
        "nonlinear_qmc_error_contraction": metric(
            nonlinear_covariance_errors[-1] / nonlinear_covariance_errors[0],
            "diagnostic",
        ),
        "eiv_latent_log_likelihood_max_error": metric(
            jnp.max(jnp.abs(eiv_log_likelihood - latent_reference_log_likelihood)),
            "accuracy",
            maximum=1.0e-8,
        ),
        "eiv_latent_posterior_mean_error": metric(
            jnp.abs(eiv_mean - latent_mean),
            "accuracy",
            maximum=1.0e-8,
        ),
        "eiv_latent_posterior_variance_relative_error": metric(
            jnp.abs(eiv_variance - latent_variance) / latent_variance,
            "accuracy",
            maximum=1.0e-7,
        ),
        "ordinary_eiv_posterior_mean_discrepancy": metric(
            jnp.abs(ordinary_mean - latent_mean),
            "diagnostic",
        ),
        "low_rank_variance_relative_error": metric(
            low_rank_variance_error,
            "accuracy",
            maximum=1.0e-11,
        ),
        "low_rank_operator_relative_error": metric(
            low_rank_action_error,
            "accuracy",
            maximum=1.0e-11,
        ),
        "low_rank_hutchinson_relative_error": metric(
            low_rank_hutchinson_error,
            "diagnostic",
        ),
        "low_rank_hutchinson_normalized_error": metric(
            low_rank_hutchinson_normalized_error,
            "accuracy",
            maximum=3.0,
        ),
        "matrix_free_memory_reduction": metric(
            dense_output_memory / represented_memory,
            "diagnostic",
            minimum=100.0,
        ),
        "input_dimension": metric(input_dimension, "diagnostic"),
        "output_dimension": metric(output_dimension, "diagnostic"),
        "factor_rank": metric(factor_rank, "diagnostic"),
        "represented_covariance_memory_bytes": metric(
            represented_memory,
            "performance",
            unit="byte",
        ),
        "dense_output_covariance_memory_bytes": metric(
            dense_output_memory,
            "diagnostic",
            unit="byte",
        ),
    }
    metrics.update(
        _performance_metrics(
            wall_seconds=time.perf_counter() - started,
            cold_seconds=cold,
            compile_seconds=compile_seconds,
            execution_seconds=execute,
            sample_memory_bytes=represented_memory,
        )
    )
    return ScenarioResult(
        name="linearized_uncertainty_propagation",
        description=linearized_uncertainty_propagation.__doc__ or "",
        seed=seed,
        metrics=metrics,
        metadata={
            "profile": configuration.profile,
            "qmc_design": phx.sampling.design_signature("sobol_scrambled"),
            "qmc_samples": configuration.linearized_qmc_samples,
            "nonlinear_scales": list(nonlinear_scales),
            "nonlinear_covariance_relative_errors": nonlinear_covariance_errors,
            "nonlinear_mean_errors": nonlinear_mean_errors,
            "input_dimension": input_dimension,
            "output_dimension": output_dimension,
            "factor_rank": factor_rank,
            "hutchinson_probes": configuration.linearized_hutchinson_probes,
            "represented_covariance_memory_bytes": represented_memory,
            "dense_output_covariance_memory_bytes": dense_output_memory,
            "eiv_posterior_mean": float(eiv_mean),
            "latent_reference_posterior_mean": float(latent_mean),
            "ordinary_posterior_mean": float(ordinary_mean),
        },
    )


SCENARIOS: dict[str, Scenario] = {
    "elliptic_coefficient_inverse": elliptic_coefficient_inverse,
    "nonlinear_transformed_ode": nonlinear_transformed_ode,
    "neural_selected_subspace": neural_selected_subspace,
    "multimodal_tempered_inference": multimodal_tempered_inference,
    "misspecified_pde_discrepancy": misspecified_pde_discrepancy,
    "operator_conditioned_inverse_pde": operator_conditioned_inverse_pde,
    "correlated_vector_discrepancy": correlated_vector_discrepancy,
    "deep_kernel_likelihood_timing": deep_kernel_likelihood_timing,
    "flow_assisted_multimodal": flow_assisted_multimodal,
    "stochastic_gradient_regression": stochastic_gradient_regression,
    "linearized_uncertainty_propagation": linearized_uncertainty_propagation,
}


__all__ = ["SCENARIOS"]
