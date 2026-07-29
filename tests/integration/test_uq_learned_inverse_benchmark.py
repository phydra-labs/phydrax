#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


_TRUE_SOURCE = 4.0
_OBSERVATION_SCALE = 0.01
_NUM_ENSEMBLE_MEMBERS = 3


class _OutputComponent(eqx.Module):
    field: Any
    index: int = eqx.field(static=True)

    def __call__(self, *args, key=None, **kwargs):
        return self.field.func(*args, key=key, **kwargs)[self.index]


class _StagedTrainer:
    def __init__(
        self,
        solver,
        *,
        first_iterations: int = 60,
        second_iterations: int = 40,
    ):
        self.solver = solver
        self.first_iterations = int(first_iterations)
        self.second_iterations = int(second_iterations)

    def solve(self, *, seed: int, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected staged-solver arguments: {tuple(kwargs)!r}.")
        first = self.solver.solve(
            num_iter=self.first_iterations,
            optim=optax.adam(3e-2),
            seed=seed,
            jit=True,
            keep_best=True,
            log_every=0,
        )
        return first.solve(
            num_iter=self.second_iterations,
            optim=optax.adam(1.5e-2),
            seed=seed + 1,
            jit=True,
            keep_best=True,
            log_every=0,
        )


def _project_output(field, index: int):
    return phx.domain.DomainFunction(
        domain=field.domain,
        deps=field.deps,
        func=_OutputComponent(field, index),
    )


def _derived_fields(state) -> Mapping[str, Any]:
    geometry = state.domain

    @geometry.Function("x")
    def boundary_factor(x):
        return x[0] * (1.0 - x[0])

    solution = boundary_factor * _project_output(state, 0)
    source = _project_output(state, 1)
    residual = -phx.operators.laplacian(solution, var="x") - source
    return {"u": solution, "source": source, "residual": residual}


def _make_inverse_solver(
    key,
    /,
    *,
    randomized_prior: bool = False,
    sensor_x=None,
    sensor_targets=None,
    observation_key=None,
):
    geometry = phx.domain.Interval1d(0.0, 1.0)
    structure = phx.domain.ProductStructure((("x",),))

    def model_factory(model_key):
        return phx.nn.MLP(
            in_size=1,
            out_size=2,
            width_size=6,
            depth=1,
            key=model_key,
        )

    if randomized_prior:
        learned_key, prior_key = jr.split(key)
        model = phx.uq.RandomizedPriorModel(
            model_factory(learned_key),
            model_factory(prior_key),
            beta=0.05,
        )
    else:
        model = model_factory(key)
    state = geometry.Model("x")(model)

    def poisson_residual(state_field):
        fields = _derived_fields(state_field)
        return -phx.operators.laplacian(fields["u"], var="x") - fields["source"]

    def constant_source_residual(state_field):
        return phx.operators.grad(_derived_fields(state_field)["source"], var="x")

    poisson = phx.constraints.ContinuousPointwiseInteriorConstraint(
        "state",
        geometry,
        operator=poisson_residual,
        num_points=16,
        structure=structure,
        sampling_mode="fixed",
        fixed_batch_key=jr.key(90),
        weight=2.0,
    )
    constant_source = phx.constraints.ContinuousPointwiseInteriorConstraint(
        "state",
        geometry,
        operator=constant_source_residual,
        num_points=16,
        structure=structure,
        sampling_mode="fixed",
        fixed_batch_key=jr.key(92),
    )

    if sensor_x is None:
        sensor_x = jnp.linspace(0.08, 0.92, 12)
    else:
        sensor_x = jnp.asarray(sensor_x)
    if observation_key is None:
        observation_key = jr.key(91)
    if sensor_targets is None:
        exact_sensor_values = 0.5 * _TRUE_SOURCE * sensor_x * (1.0 - sensor_x)
    else:
        exact_sensor_values = jnp.asarray(sensor_targets)
        if exact_sensor_values.shape != sensor_x.shape:
            raise ValueError("sensor_targets must have the same shape as sensor_x.")
    raw_noise = jr.normal(observation_key, sensor_x.shape)
    observations = exact_sensor_values + _OBSERVATION_SCALE * (
        raw_noise - jnp.mean(raw_noise)
    )

    @geometry.Function("x")
    def observed_field(x):
        return jnp.interp(x[0], sensor_x, observations)

    data = phx.constraints.PointSetConstraint.from_points(
        component=geometry.component(),
        points={"x": sensor_x[:, None]},
        residual=lambda functions: (
            _derived_fields(functions["state"])["u"] - observed_field
        ),
        constraint_vars=("state",),
        weight=20.0,
    )
    return phx.solver.FunctionalSolver(
        functions={"state": state},
        constraints=(poisson, constant_source, data),
    )


def _fit_staged(key, *, seed: int):
    return _StagedTrainer(_make_inverse_solver(key)).solve(seed=seed)


def _fit_ensemble(key, *, randomized_prior: bool):
    return phx.uq.fit_ensemble(
        lambda member_key: _StagedTrainer(
            _make_inverse_solver(member_key, randomized_prior=randomized_prior)
        ),
        num_members=_NUM_ENSEMBLE_MEMBERS,
        key=key,
        homogeneous=False,
        return_diagnostics=True,
    )


def _ensemble_fields(fit_result):
    members = tuple(
        _derived_fields(member["state"]) for member in fit_result.ensemble.members
    )
    return phx.uq.HeterogeneousFunctionEnsemble(members)


def _field_rmse(center, exact):
    return jnp.sqrt(jnp.mean((jnp.asarray(center) - jnp.asarray(exact)) ** 2))


def _calibration_metrics(
    center,
    epistemic_scale,
    observation_scale,
    trajectories,
    calibration_indices,
    test_indices,
):
    center = jnp.asarray(center)
    total_scale = jnp.sqrt(jnp.asarray(epistemic_scale) ** 2 + observation_scale**2)
    calibration_center = jnp.broadcast_to(center, (calibration_indices.size, center.size))
    calibration_scale = jnp.broadcast_to(
        total_scale, (calibration_indices.size, center.size)
    )
    calibrator = phx.uq.FunctionalConformal.calibrate(
        cx.Field(calibration_center, dims=("case", "x")),
        cx.Field(trajectories[calibration_indices], dims=("case", "x")),
        alpha=0.1,
        case_dim="case",
        scale=cx.Field(calibration_scale, dims=("case", "x")),
    )

    test_center = cx.Field(
        jnp.broadcast_to(center, (test_indices.size, center.size)),
        dims=("case", "x"),
    )
    test_scale = cx.Field(
        jnp.broadcast_to(total_scale, (test_indices.size, center.size)),
        dims=("case", "x"),
    )
    test_target = trajectories[test_indices]
    pre_width = 1.6448536269514722 * test_scale.data
    pre_coordinate_coverage = (test_target >= test_center.data - pre_width) & (
        test_target <= test_center.data + pre_width
    )
    interval = calibrator.interval(test_center, test_scale)
    post_coordinate_coverage = (test_target >= interval.lower.data) & (
        test_target <= interval.upper.data
    )
    likelihood = phx.uq.GaussianLikelihood(total_scale)
    return {
        "nll": phx.uq.negative_log_likelihood(
            likelihood,
            test_center.data,
            test_target,
        ),
        "crps": jnp.mean(
            phx.uq.gaussian_crps(
                test_center.data,
                test_scale.data,
                test_target,
            )
        ),
        "pre_pointwise_coverage": jnp.mean(pre_coordinate_coverage),
        "pre_simultaneous_coverage": jnp.mean(jnp.all(pre_coordinate_coverage, axis=1)),
        "post_simultaneous_coverage": jnp.mean(jnp.all(post_coordinate_coverage, axis=1)),
        "post_width": phx.uq.interval_width(
            interval.lower.data,
            interval.upper.data,
        ),
    }


@pytest.mark.skipif(
    os.environ.get("PHYDRAX_RUN_SLOW_BENCHMARKS") != "1",
    reason="set PHYDRAX_RUN_SLOW_BENCHMARKS=1 to run learned UQ training",
)
def test_learned_inverse_poisson_uq_comparison_benchmark(record_property):
    query_x = jnp.linspace(0.0, 1.0, 65)
    points = {"x": cx.Field(query_x[:, None], dims=("x", None))}
    exact = 0.5 * _TRUE_SOURCE * query_x * (1.0 - query_x)

    deterministic_started = time.perf_counter()
    deterministic_solver = _fit_staged(jr.key(100), seed=100)
    jax.block_until_ready(deterministic_solver)
    deterministic_duration = time.perf_counter() - deterministic_started

    deep_fit = _fit_ensemble(jr.key(102), randomized_prior=False)
    prior_fit = _fit_ensemble(jr.key(103), randomized_prior=True)

    deterministic_fields = _derived_fields(
        phx.nn.inference_mode(deterministic_solver)["state"]
    )
    deterministic_center = deterministic_fields["u"](points).data
    deterministic_source = deterministic_fields["source"](points).data
    deterministic_residual = deterministic_fields["residual"](points).data

    deep_ensemble = _ensemble_fields(deep_fit)
    deep_started = time.perf_counter()
    deep_prediction = deep_ensemble.predict_many(
        ("u", "source", "residual"),
        points,
        key=jr.key(105),
        valid_policy="raise",
    )
    jax.block_until_ready(deep_prediction)
    deep_evaluation_duration = time.perf_counter() - deep_started

    prior_ensemble = _ensemble_fields(prior_fit)
    prior_started = time.perf_counter()
    prior_prediction = prior_ensemble.predict_many(
        ("u", "source", "residual"),
        points,
        key=jr.key(106),
        valid_policy="raise",
    )
    jax.block_until_ready(prior_prediction)
    prior_evaluation_duration = time.perf_counter() - prior_started

    centers = {
        "deterministic": deterministic_center,
        "ensemble": deep_prediction["u"].mean().data,
        "randomized_prior": prior_prediction["u"].mean().data,
    }
    epistemic_scales = {
        "deterministic": jnp.zeros_like(deterministic_center),
        "ensemble": deep_prediction["u"].std().data,
        "randomized_prior": prior_prediction["u"].std().data,
    }
    source_estimates = {
        "deterministic": jnp.mean(deterministic_source),
        "ensemble": jnp.mean(deep_prediction["source"].mean().data),
        "randomized_prior": jnp.mean(prior_prediction["source"].mean().data),
    }
    residual_rms = {
        "deterministic": jnp.sqrt(jnp.mean(deterministic_residual[2:-2] ** 2)),
        "ensemble": jnp.sqrt(
            jnp.mean(deep_prediction["residual"].samples.data[:, 2:-2] ** 2)
        ),
        "randomized_prior": jnp.sqrt(
            jnp.mean(prior_prediction["residual"].samples.data[:, 2:-2] ** 2)
        ),
    }

    num_cases = 6000
    coefficients = jr.normal(jr.key(107), (num_cases, 4))
    observation_shape = 0.012 + 0.008 * query_x
    standardized_noise = (
        0.70 * coefficients[:, 0, None]
        + 0.45 * coefficients[:, 1, None] * jnp.cos(2.0 * jnp.pi * query_x)[None, :]
        + 0.30 * coefficients[:, 2, None] * jnp.sin(3.0 * jnp.pi * query_x)[None, :]
        + 0.20 * coefficients[:, 3, None] * (2.0 * query_x - 1.0)[None, :]
    )
    trajectories = exact[None, :] + observation_shape[None, :] * standardized_noise
    train_indices, calibration_indices, test_indices = (
        phx.data_utils.train_calibration_test_split_indices(
            num_cases,
            calibration_fraction=0.2,
            test_fraction=0.2,
            key=jr.key(108),
        )
    )
    estimated_observation_scale = jnp.std(
        trajectories[train_indices] - exact[None, :],
        axis=0,
        ddof=1,
    )
    metrics = {
        name: _calibration_metrics(
            center,
            epistemic_scales[name],
            estimated_observation_scale,
            trajectories,
            calibration_indices,
            test_indices,
        )
        for name, center in centers.items()
    }

    stochastic_samples = {
        "ensemble": deep_prediction["u"].samples.data,
        "randomized_prior": prior_prediction["u"].samples.data,
    }
    for name, center in centers.items():
        assert _field_rmse(center, exact) < 0.03
        assert jnp.abs(source_estimates[name] - _TRUE_SOURCE) < 0.4
        assert residual_rms[name] < 0.25
        assert 0.86 <= metrics[name]["post_simultaneous_coverage"] <= 0.94
        assert jnp.abs(metrics[name]["post_simultaneous_coverage"] - 0.9) < jnp.abs(
            metrics[name]["pre_simultaneous_coverage"] - 0.9
        )
        assert metrics[name]["post_width"] > 0.0
    boundary_indices = jnp.asarray([0, -1])
    assert jnp.allclose(deterministic_center[boundary_indices], 0.0, atol=1e-10)
    for samples in stochastic_samples.values():
        assert jnp.allclose(samples[:, boundary_indices], 0.0, atol=1e-10)
    stochastic_names = ("ensemble", "randomized_prior")
    best_stochastic_nll = jnp.min(
        jnp.asarray([metrics[name]["nll"] for name in stochastic_names])
    )
    best_stochastic_crps = jnp.min(
        jnp.asarray([metrics[name]["crps"] for name in stochastic_names])
    )
    assert best_stochastic_nll <= metrics["deterministic"]["nll"] + 0.05
    assert best_stochastic_crps <= metrics["deterministic"]["crps"] + 0.005

    durations = {
        "deterministic_fit_seconds": deterministic_duration,
        "deep_fit_seconds": deep_fit.total_duration_seconds,
        "randomized_prior_fit_seconds": prior_fit.total_duration_seconds,
        "deep_evaluation_seconds": deep_evaluation_duration,
        "randomized_prior_evaluation_seconds": prior_evaluation_duration,
    }
    for name, value in durations.items():
        record_property(name, value)
    for name, center in centers.items():
        record_property(f"{name}_field_rmse", float(_field_rmse(center, exact)))
        record_property(f"{name}_source", float(source_estimates[name]))
        record_property(f"{name}_pde_residual_rms", float(residual_rms[name]))
        for metric_name, value in metrics[name].items():
            record_property(f"{name}_{metric_name}", float(value))
    for name, samples in stochastic_samples.items():
        record_property(f"{name}_sample_bytes", samples.nbytes)


_STRESS_METHODS = ("deterministic", "ensemble", "randomized_prior")
_STRESS_TRIAL_COUNT = 3
_STRESS_FIRST_ITERATIONS = 50
_STRESS_SECOND_ITERATIONS = 30


def _fit_stress_solver(
    model_key,
    observation_key,
    *,
    seed: int,
):
    trainer = _StagedTrainer(
        _make_inverse_solver(
            model_key,
            sensor_x=jnp.linspace(0.05, 0.65, 6),
            observation_key=observation_key,
        ),
        first_iterations=_STRESS_FIRST_ITERATIONS,
        second_iterations=_STRESS_SECOND_ITERATIONS,
    )
    return trainer.solve(seed=seed)


def _fit_stress_ensemble(
    key,
    observation_key,
    *,
    randomized_prior: bool,
):
    return phx.uq.fit_ensemble(
        lambda member_key: _StagedTrainer(
            _make_inverse_solver(
                member_key,
                randomized_prior=randomized_prior,
                sensor_x=jnp.linspace(0.05, 0.65, 6),
                observation_key=observation_key,
            ),
            first_iterations=_STRESS_FIRST_ITERATIONS,
            second_iterations=_STRESS_SECOND_ITERATIONS,
        ),
        num_members=_NUM_ENSEMBLE_MEMBERS,
        key=key,
        homogeneous=False,
        return_diagnostics=True,
    )


def _rank_correlation(left, right) -> float:
    left = jnp.asarray(left)
    right = jnp.asarray(right)
    if float(jnp.std(left)) <= 1e-12 or float(jnp.std(right)) <= 1e-12:
        return 0.0

    def ranks(values):
        order = jnp.argsort(values)
        return (
            jnp.zeros(values.shape, dtype=float)
            .at[order]
            .set(jnp.arange(values.size, dtype=float))
        )

    left_rank = ranks(left)
    right_rank = ranks(right)
    left_centered = left_rank - jnp.mean(left_rank)
    right_centered = right_rank - jnp.mean(right_rank)
    denominator = jnp.sqrt(jnp.sum(left_centered**2) * jnp.sum(right_centered**2))
    return float(jnp.sum(left_centered * right_centered) / denominator)


def _stress_trajectories(key, query_x, exact, *, num_cases: int = 3000):
    coefficients = jr.normal(key, (num_cases, 4))
    observation_shape = 0.012 + 0.008 * query_x
    standardized_noise = (
        0.70 * coefficients[:, 0, None]
        + 0.45 * coefficients[:, 1, None] * jnp.cos(2.0 * jnp.pi * query_x)[None, :]
        + 0.30 * coefficients[:, 2, None] * jnp.sin(3.0 * jnp.pi * query_x)[None, :]
        + 0.20 * coefficients[:, 3, None] * (2.0 * query_x - 1.0)[None, :]
    )
    return exact[None, :] + observation_shape[None, :] * standardized_noise


def _evaluate_stress_trial(trial_index: int):
    query_x = jnp.linspace(0.0, 1.0, 65)
    points = {"x": cx.Field(query_x[:, None], dims=("x", None))}
    exact = 0.5 * _TRUE_SOURCE * query_x * (1.0 - query_x)
    observed_region = query_x <= 0.65
    extrapolation_region = query_x >= 0.70
    observation_key = jr.key(2000 + trial_index)
    deterministic_key, ensemble_key, prior_key = jr.split(
        jr.key(3000 + trial_index),
        3,
    )

    started = time.perf_counter()
    deterministic_solver = _fit_stress_solver(
        deterministic_key,
        observation_key,
        seed=4000 + 10 * trial_index,
    )
    jax.block_until_ready(deterministic_solver)
    deterministic_fit_seconds = time.perf_counter() - started

    ensemble_fit = _fit_stress_ensemble(
        ensemble_key,
        observation_key,
        randomized_prior=False,
    )
    prior_fit = _fit_stress_ensemble(
        prior_key,
        observation_key,
        randomized_prior=True,
    )

    deterministic_fields = _derived_fields(
        phx.nn.inference_mode(deterministic_solver)["state"]
    )
    deterministic_center = deterministic_fields["u"](points).data
    deterministic_source = deterministic_fields["source"](points).data
    deterministic_residual = deterministic_fields["residual"](points).data

    started = time.perf_counter()
    ensemble_prediction = _ensemble_fields(ensemble_fit).predict_many(
        ("u", "source", "residual"),
        points,
        key=jr.key(6000 + trial_index),
        valid_policy="raise",
    )
    jax.block_until_ready(ensemble_prediction)
    ensemble_evaluation_seconds = time.perf_counter() - started

    started = time.perf_counter()
    prior_prediction = _ensemble_fields(prior_fit).predict_many(
        ("u", "source", "residual"),
        points,
        key=jr.key(7000 + trial_index),
        valid_policy="raise",
    )
    jax.block_until_ready(prior_prediction)
    prior_evaluation_seconds = time.perf_counter() - started

    centers = {
        "deterministic": deterministic_center,
        "ensemble": ensemble_prediction["u"].mean().data,
        "randomized_prior": prior_prediction["u"].mean().data,
    }
    epistemic_scales = {
        "deterministic": jnp.zeros_like(deterministic_center),
        "ensemble": ensemble_prediction["u"].std().data,
        "randomized_prior": prior_prediction["u"].std().data,
    }
    source_estimates = {
        "deterministic": jnp.mean(deterministic_source),
        "ensemble": jnp.mean(ensemble_prediction["source"].mean().data),
        "randomized_prior": jnp.mean(prior_prediction["source"].mean().data),
    }
    residual_rms = {
        "deterministic": jnp.sqrt(jnp.mean(deterministic_residual[2:-2] ** 2)),
        "ensemble": jnp.sqrt(
            jnp.mean(ensemble_prediction["residual"].samples.data[:, 2:-2] ** 2)
        ),
        "randomized_prior": jnp.sqrt(
            jnp.mean(prior_prediction["residual"].samples.data[:, 2:-2] ** 2)
        ),
    }
    stochastic_samples = {
        "ensemble": ensemble_prediction["u"].samples.data,
        "randomized_prior": prior_prediction["u"].samples.data,
    }

    trajectories = _stress_trajectories(
        jr.key(8000 + trial_index),
        query_x,
        exact,
    )
    train_indices, calibration_indices, test_indices = (
        phx.data_utils.train_calibration_test_split_indices(
            trajectories.shape[0],
            calibration_fraction=0.2,
            test_fraction=0.2,
            key=jr.key(9000 + trial_index),
        )
    )
    observation_scale = jnp.std(
        trajectories[train_indices] - exact[None, :],
        axis=0,
        ddof=1,
    )
    calibrated = {
        name: _calibration_metrics(
            center,
            epistemic_scales[name],
            observation_scale,
            trajectories,
            calibration_indices,
            test_indices,
        )
        for name, center in centers.items()
    }
    fit_seconds = {
        "deterministic": deterministic_fit_seconds,
        "ensemble": ensemble_fit.total_duration_seconds,
        "randomized_prior": prior_fit.total_duration_seconds,
    }
    evaluation_seconds = {
        "deterministic": 0.0,
        "ensemble": ensemble_evaluation_seconds,
        "randomized_prior": prior_evaluation_seconds,
    }

    boundary_indices = jnp.asarray([0, -1])
    assert jnp.allclose(
        deterministic_center[boundary_indices],
        0.0,
        atol=1e-10,
    )
    for samples in stochastic_samples.values():
        assert jnp.allclose(samples[:, boundary_indices], 0.0, atol=1e-10)

    method_metrics = {}
    for name in _STRESS_METHODS:
        absolute_error = jnp.abs(centers[name] - exact)
        observed_scale = jnp.mean(epistemic_scales[name][observed_region])
        extrapolation_scale = jnp.mean(epistemic_scales[name][extrapolation_region])
        scale_ratio = jnp.where(
            observed_scale > 1e-12,
            extrapolation_scale / observed_scale,
            0.0,
        )
        method_metrics[name] = {
            "full_rmse": float(_field_rmse(centers[name], exact)),
            "observed_rmse": float(
                _field_rmse(
                    centers[name][observed_region],
                    exact[observed_region],
                )
            ),
            "extrapolation_rmse": float(
                _field_rmse(
                    centers[name][extrapolation_region],
                    exact[extrapolation_region],
                )
            ),
            "source_estimate": float(source_estimates[name]),
            "source_absolute_error": float(
                jnp.abs(source_estimates[name] - _TRUE_SOURCE)
            ),
            "pde_residual_rms": float(residual_rms[name]),
            "epistemic_error_spearman": _rank_correlation(
                epistemic_scales[name],
                absolute_error,
            ),
            "extrapolation_scale_ratio": float(scale_ratio),
            "fit_seconds": float(fit_seconds[name]),
            "evaluation_seconds": float(evaluation_seconds[name]),
            **{
                metric_name: float(value)
                for metric_name, value in calibrated[name].items()
            },
        }
    return method_metrics


def _stress_retention_summary(trials):
    metric_names = tuple(trials[0]["deterministic"])
    aggregate = {
        method: {
            metric: sum(trial[method][metric] for trial in trials) / len(trials)
            for metric in metric_names
        }
        for method in _STRESS_METHODS
    }
    baseline = aggregate["deterministic"]
    decisions = {"deterministic": {"decision": "reference"}}
    for method in _STRESS_METHODS[1:]:
        values = aggregate[method]
        nll_deltas = [
            trial[method]["nll"] - trial["deterministic"]["nll"] for trial in trials
        ]
        crps_deltas = [
            trial[method]["crps"] - trial["deterministic"]["crps"] for trial in trials
        ]
        nll_delta = values["nll"] - baseline["nll"]
        crps_delta = values["crps"] - baseline["crps"]
        median_nll_delta = float(jnp.median(jnp.asarray(nll_deltas)))
        median_crps_delta = float(jnp.median(jnp.asarray(crps_deltas)))
        nll_win_rate = sum(delta < 0.0 for delta in nll_deltas) / len(trials)
        crps_win_rate = sum(delta < 0.0 for delta in crps_deltas) / len(trials)
        extrapolation_win_rate = sum(
            trial[method]["extrapolation_rmse"]
            < trial["deterministic"]["extrapolation_rmse"]
            for trial in trials
        ) / len(trials)
        proper_score_improvement = max(nll_win_rate, crps_win_rate) >= 2.0 / 3.0
        proper_score_competitive = (
            median_nll_delta <= 0.05 and median_crps_delta <= 0.0015
        )
        uncertainty_signal = (
            values["epistemic_error_spearman"] >= 0.25
            and values["extrapolation_scale_ratio"] >= 1.05
        )
        coverage_efficient = (
            0.86 <= values["post_simultaneous_coverage"] <= 0.94
            and values["post_width"] <= 1.25 * baseline["post_width"]
        )
        extrapolation_advantage = extrapolation_win_rate >= 2.0 / 3.0
        accuracy_stable = values["full_rmse"] <= baseline["full_rmse"]
        if (
            proper_score_improvement
            and accuracy_stable
            and (extrapolation_advantage or uncertainty_signal)
        ):
            decision = "promote"
        elif coverage_efficient and (
            proper_score_competitive or extrapolation_advantage or uncertainty_signal
        ):
            decision = "keep_experimental"
        else:
            decision = "remove_candidate"
        decisions[method] = {
            "decision": decision,
            "nll_delta": nll_delta,
            "crps_delta": crps_delta,
            "median_paired_nll_delta": median_nll_delta,
            "median_paired_crps_delta": median_crps_delta,
            "nll_win_rate": nll_win_rate,
            "crps_win_rate": crps_win_rate,
            "extrapolation_win_rate": extrapolation_win_rate,
            "proper_score_competitive": proper_score_competitive,
            "uncertainty_signal": uncertainty_signal,
            "coverage_efficient": coverage_efficient,
            "accuracy_stable": accuracy_stable,
        }
    return {
        "trial_count": len(trials),
        "sensor_count": 6,
        "sensor_range": [0.05, 0.65],
        "extrapolation_range": [0.70, 1.0],
        "methods": aggregate,
        "retention": decisions,
        "trials": trials,
    }


def test_stress_retention_gates_use_paired_scores_and_stability():
    def metrics(nll, crps, extrapolation_rmse, full_rmse):
        return {
            "nll": nll,
            "crps": crps,
            "extrapolation_rmse": extrapolation_rmse,
            "full_rmse": full_rmse,
            "epistemic_error_spearman": 0.9,
            "extrapolation_scale_ratio": 0.65,
            "post_simultaneous_coverage": 0.9,
            "post_width": 0.09,
        }

    trials = [
        {
            "deterministic": metrics(-2.927, 0.0074, 0.0019, 0.0023),
            "ensemble": metrics(-2.929, 0.0073, 0.0018, 0.0020),
            "randomized_prior": metrics(-2.334, 0.0141, 0.0164, 0.0217),
        },
        {
            "deterministic": metrics(3.486, 0.0354, 0.0335, 0.0447),
            "ensemble": metrics(-2.969, 0.0071, 0.0011, 0.0009),
            "randomized_prior": metrics(-1.977, 0.0199, 0.0243, 0.0326),
        },
        {
            "deterministic": metrics(-2.951, 0.0072, 0.0014, 0.0026),
            "ensemble": metrics(-2.628, 0.0089, 0.0060, 0.0075),
            "randomized_prior": metrics(-2.975, 0.0070, 0.0002, 0.0004),
        },
    ]

    decisions = _stress_retention_summary(trials)["retention"]
    assert decisions["ensemble"]["decision"] == "promote"
    assert decisions["randomized_prior"]["decision"] == "keep_experimental"


@pytest.mark.skipif(
    os.environ.get("PHYDRAX_RUN_UQ_STRESS_BENCHMARKS") != "1",
    reason="set PHYDRAX_RUN_UQ_STRESS_BENCHMARKS=1 to run repeated UQ stress fits",
)
def test_sparse_sensor_extrapolation_retention_benchmark(record_property):
    trials = [
        _evaluate_stress_trial(trial_index) for trial_index in range(_STRESS_TRIAL_COUNT)
    ]
    for trial in trials:
        for method in _STRESS_METHODS:
            metrics = trial[method]
            assert all(jnp.isfinite(value) for value in metrics.values())
            assert metrics["full_rmse"] < 0.12
            assert metrics["source_absolute_error"] < 1.5
            assert metrics["pde_residual_rms"] < 0.3
            assert 0.84 <= metrics["post_simultaneous_coverage"] <= 0.96
            assert metrics["post_width"] > 0.0

    summary = _stress_retention_summary(trials)
    report = json.dumps(summary, sort_keys=True)
    record_property("retention_summary_json", report)
    report_path = os.environ.get("PHYDRAX_UQ_STRESS_REPORT")
    if report_path:
        Path(report_path).write_text(report + "\n", encoding="utf-8")


_MISSPECIFICATION_AMPLITUDE = 0.03
_MISSPECIFICATION_METHODS = ("deterministic", "ensemble")
_MISSPECIFICATION_TRIAL_COUNT = 3


def _misspecified_truth(x):
    x = jnp.asarray(x)
    baseline = 0.5 * _TRUE_SOURCE * x * (1.0 - x)
    return baseline + _MISSPECIFICATION_AMPLITUDE * jnp.sin(2.0 * jnp.pi * x)


def _misspecified_fields(state) -> Mapping[str, Any]:
    fields = _derived_fields(state)
    geometry = state.domain

    @geometry.Function("x")
    def true_forcing(x):
        return _TRUE_SOURCE + (
            _MISSPECIFICATION_AMPLITUDE
            * (2.0 * jnp.pi) ** 2
            * jnp.sin(2.0 * jnp.pi * x[0])
        )

    true_residual = -phx.operators.laplacian(fields["u"], var="x") - true_forcing
    return {
        **fields,
        "true_residual": true_residual,
    }


def _fit_misspecification_solver(
    model_key,
    observation_key,
    *,
    seed: int,
):
    sensor_x = jnp.linspace(0.05, 0.95, 12)
    trainer = _StagedTrainer(
        _make_inverse_solver(
            model_key,
            sensor_x=sensor_x,
            sensor_targets=_misspecified_truth(sensor_x),
            observation_key=observation_key,
        ),
        first_iterations=_STRESS_FIRST_ITERATIONS,
        second_iterations=_STRESS_SECOND_ITERATIONS,
    )
    return trainer.solve(seed=seed)


def _fit_misspecification_ensemble(key, observation_key):
    sensor_x = jnp.linspace(0.05, 0.95, 12)
    sensor_targets = _misspecified_truth(sensor_x)
    return phx.uq.fit_ensemble(
        lambda member_key: _StagedTrainer(
            _make_inverse_solver(
                member_key,
                sensor_x=sensor_x,
                sensor_targets=sensor_targets,
                observation_key=observation_key,
            ),
            first_iterations=_STRESS_FIRST_ITERATIONS,
            second_iterations=_STRESS_SECOND_ITERATIONS,
        ),
        num_members=_NUM_ENSEMBLE_MEMBERS,
        key=key,
        homogeneous=False,
        return_diagnostics=True,
    )


def _misspecification_ensemble_fields(fit_result):
    members = tuple(
        _misspecified_fields(member["state"]) for member in fit_result.ensemble.members
    )
    return phx.uq.HeterogeneousFunctionEnsemble(members)


def _evaluate_misspecification_trial(trial_index: int):
    query_x = jnp.linspace(0.0, 1.0, 65)
    points = {"x": cx.Field(query_x[:, None], dims=("x", None))}
    exact = _misspecified_truth(query_x)
    discrepancy = jnp.abs(_MISSPECIFICATION_AMPLITUDE * jnp.sin(2.0 * jnp.pi * query_x))
    interior = (query_x >= 0.05) & (query_x <= 0.95)
    high_discrepancy = discrepancy >= 0.8 * _MISSPECIFICATION_AMPLITUDE
    low_discrepancy = interior & (discrepancy <= 0.2 * _MISSPECIFICATION_AMPLITUDE)
    observation_key = jr.key(10000 + trial_index)
    deterministic_key, ensemble_key = jr.split(
        jr.key(11000 + trial_index),
        2,
    )

    started = time.perf_counter()
    deterministic_solver = _fit_misspecification_solver(
        deterministic_key,
        observation_key,
        seed=12000 + 10 * trial_index,
    )
    jax.block_until_ready(deterministic_solver)
    deterministic_fit_seconds = time.perf_counter() - started

    ensemble_fit = _fit_misspecification_ensemble(
        ensemble_key,
        observation_key,
    )

    deterministic_fields = _misspecified_fields(
        phx.nn.inference_mode(deterministic_solver)["state"]
    )
    deterministic_center = deterministic_fields["u"](points).data
    deterministic_source = deterministic_fields["source"](points).data
    deterministic_assumed_residual = deterministic_fields["residual"](points).data
    deterministic_true_residual = deterministic_fields["true_residual"](points).data

    started = time.perf_counter()
    ensemble_prediction = _misspecification_ensemble_fields(ensemble_fit).predict_many(
        ("u", "source", "residual", "true_residual"),
        points,
        key=jr.key(14000 + trial_index),
        valid_policy="raise",
    )
    jax.block_until_ready(ensemble_prediction)
    ensemble_evaluation_seconds = time.perf_counter() - started

    centers = {
        "deterministic": deterministic_center,
        "ensemble": ensemble_prediction["u"].mean().data,
    }
    epistemic_scales = {
        "deterministic": jnp.zeros_like(deterministic_center),
        "ensemble": ensemble_prediction["u"].std().data,
    }
    source_estimates = {
        "deterministic": jnp.mean(deterministic_source),
        "ensemble": jnp.mean(ensemble_prediction["source"].mean().data),
    }
    assumed_residual_rms = {
        "deterministic": jnp.sqrt(jnp.mean(deterministic_assumed_residual[2:-2] ** 2)),
        "ensemble": jnp.sqrt(
            jnp.mean(ensemble_prediction["residual"].samples.data[:, 2:-2] ** 2)
        ),
    }
    true_residual_rms = {
        "deterministic": jnp.sqrt(jnp.mean(deterministic_true_residual[2:-2] ** 2)),
        "ensemble": jnp.sqrt(
            jnp.mean(ensemble_prediction["true_residual"].samples.data[:, 2:-2] ** 2)
        ),
    }
    stochastic_samples = {
        "ensemble": ensemble_prediction["u"].samples.data,
    }

    trajectories = _stress_trajectories(
        jr.key(15000 + trial_index),
        query_x,
        exact,
    )
    train_indices, calibration_indices, test_indices = (
        phx.data_utils.train_calibration_test_split_indices(
            trajectories.shape[0],
            calibration_fraction=0.2,
            test_fraction=0.2,
            key=jr.key(16000 + trial_index),
        )
    )
    observation_scale = jnp.std(
        trajectories[train_indices] - exact[None, :],
        axis=0,
        ddof=1,
    )
    calibrated = {
        name: _calibration_metrics(
            center,
            epistemic_scales[name],
            observation_scale,
            trajectories,
            calibration_indices,
            test_indices,
        )
        for name, center in centers.items()
    }
    fit_seconds = {
        "deterministic": deterministic_fit_seconds,
        "ensemble": ensemble_fit.total_duration_seconds,
    }
    evaluation_seconds = {
        "deterministic": 0.0,
        "ensemble": ensemble_evaluation_seconds,
    }

    boundary_indices = jnp.asarray([0, -1])
    assert jnp.allclose(
        deterministic_center[boundary_indices],
        0.0,
        atol=1e-10,
    )
    for samples in stochastic_samples.values():
        assert jnp.allclose(samples[:, boundary_indices], 0.0, atol=1e-10)

    method_metrics = {}
    for name in _MISSPECIFICATION_METHODS:
        low_scale = jnp.mean(epistemic_scales[name][low_discrepancy])
        high_scale = jnp.mean(epistemic_scales[name][high_discrepancy])
        scale_ratio = jnp.where(
            low_scale > 1e-12,
            high_scale / low_scale,
            0.0,
        )
        method_metrics[name] = {
            "full_rmse": float(_field_rmse(centers[name], exact)),
            "source_estimate": float(source_estimates[name]),
            "source_absolute_error": float(
                jnp.abs(source_estimates[name] - _TRUE_SOURCE)
            ),
            "assumed_pde_residual_rms": float(assumed_residual_rms[name]),
            "true_pde_residual_rms": float(true_residual_rms[name]),
            "misspecification_scale_spearman": _rank_correlation(
                epistemic_scales[name],
                discrepancy,
            ),
            "misspecification_scale_ratio": float(scale_ratio),
            "fit_seconds": float(fit_seconds[name]),
            "evaluation_seconds": float(evaluation_seconds[name]),
            **{
                metric_name: float(value)
                for metric_name, value in calibrated[name].items()
            },
        }
    return method_metrics


def _misspecification_retention_summary(trials):
    metric_names = tuple(trials[0]["deterministic"])
    aggregate = {
        method: {
            metric: sum(trial[method][metric] for trial in trials) / len(trials)
            for metric in metric_names
        }
        for method in _MISSPECIFICATION_METHODS
    }
    baseline = aggregate["deterministic"]
    decisions = {"deterministic": {"decision": "reference"}}
    for method in _MISSPECIFICATION_METHODS[1:]:
        values = aggregate[method]
        nll_deltas = [
            trial[method]["nll"] - trial["deterministic"]["nll"] for trial in trials
        ]
        crps_deltas = [
            trial[method]["crps"] - trial["deterministic"]["crps"] for trial in trials
        ]
        nll_win_rate = sum(delta < 0.0 for delta in nll_deltas) / len(trials)
        crps_win_rate = sum(delta < 0.0 for delta in crps_deltas) / len(trials)
        true_residual_win_rate = sum(
            trial[method]["true_pde_residual_rms"]
            < trial["deterministic"]["true_pde_residual_rms"]
            for trial in trials
        ) / len(trials)
        median_nll_delta = float(jnp.median(jnp.asarray(nll_deltas)))
        median_crps_delta = float(jnp.median(jnp.asarray(crps_deltas)))
        proper_score_improvement = max(nll_win_rate, crps_win_rate) >= 2.0 / 3.0
        proper_score_competitive = (
            median_nll_delta <= 0.05 and median_crps_delta <= 0.0015
        )
        uncertainty_signal = (
            values["misspecification_scale_spearman"] >= 0.25
            and values["misspecification_scale_ratio"] >= 1.25
        )
        coverage_efficient = (
            0.86 <= values["post_simultaneous_coverage"] <= 0.94
            and values["post_width"] <= 1.25 * baseline["post_width"]
        )
        true_physics_advantage = true_residual_win_rate >= 2.0 / 3.0
        accuracy_stable = values["full_rmse"] <= baseline["full_rmse"]
        if (
            proper_score_improvement
            and accuracy_stable
            and (true_physics_advantage or uncertainty_signal)
        ):
            decision = "promote"
        elif coverage_efficient and (
            proper_score_competitive or true_physics_advantage or uncertainty_signal
        ):
            decision = "keep_experimental"
        else:
            decision = "remove_candidate"
        decisions[method] = {
            "decision": decision,
            "median_paired_nll_delta": median_nll_delta,
            "median_paired_crps_delta": median_crps_delta,
            "nll_win_rate": nll_win_rate,
            "crps_win_rate": crps_win_rate,
            "true_residual_win_rate": true_residual_win_rate,
            "proper_score_competitive": proper_score_competitive,
            "uncertainty_signal": uncertainty_signal,
            "coverage_efficient": coverage_efficient,
            "accuracy_stable": accuracy_stable,
        }
    return {
        "trial_count": len(trials),
        "truth": {
            "baseline_source": _TRUE_SOURCE,
            "unmodeled_solution_amplitude": _MISSPECIFICATION_AMPLITUDE,
            "unmodeled_mode": "sin(2*pi*x)",
        },
        "methods": aggregate,
        "retention": decisions,
        "trials": trials,
    }


@pytest.mark.skipif(
    os.environ.get("PHYDRAX_RUN_UQ_MISSPEC_BENCHMARKS") != "1",
    reason="set PHYDRAX_RUN_UQ_MISSPEC_BENCHMARKS=1 to run model-form UQ fits",
)
def test_model_misspecification_retention_benchmark(record_property):
    replay_path = os.environ.get("PHYDRAX_UQ_MISSPEC_REPLAY")
    if replay_path:
        replay = json.loads(Path(replay_path).read_text(encoding="utf-8"))
        trials = replay["trials"]
    else:
        trials = [
            _evaluate_misspecification_trial(trial_index)
            for trial_index in range(_MISSPECIFICATION_TRIAL_COUNT)
        ]
    summary = _misspecification_retention_summary(trials)
    report = json.dumps(summary, sort_keys=True)
    record_property("misspecification_summary_json", report)
    report_path = os.environ.get("PHYDRAX_UQ_MISSPEC_REPORT")
    if report_path:
        Path(report_path).write_text(report + "\n", encoding="utf-8")

    for trial in trials:
        for method in _MISSPECIFICATION_METHODS:
            metrics = trial[method]
            assert all(jnp.isfinite(value) for value in metrics.values())
            assert metrics["full_rmse"] < 0.15
            assert metrics["source_absolute_error"] < 2.0
            assert metrics["assumed_pde_residual_rms"] < 0.35
            assert metrics["true_pde_residual_rms"] < 2.5
            assert 0.84 <= metrics["post_simultaneous_coverage"] <= 0.96
            assert metrics["post_width"] > 0.0
