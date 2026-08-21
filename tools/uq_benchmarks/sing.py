#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.special import gammaln

import phydrax as phx


Scenario = Literal["linear-gaussian", "nonlinear-gaussian", "nonlinear-poisson"]


def _latent_model(
    state_size: int,
    *,
    nonlinear: bool,
) -> tuple[
    phx.stochastic.GaussianStatePrior,
    phx.stochastic.EulerMaruyamaTransitionKernel,
]:
    diagonal = jnp.linspace(0.12, 0.3, state_size)
    coupling = 0.04 * jnp.diag(jnp.ones((max(state_size - 1, 0),)), 1)
    drift_matrix = -jnp.diag(diagonal) + coupling
    dispersion = 0.24 * jnp.eye(state_size)

    def drift(time, state, args):
        del time, args
        linear = drift_matrix @ state
        return linear - 0.04 * state**3 if nonlinear else linear

    system = phx.dynamics.ContinuousSystem(
        drift,
        state_layout=phx.dynamics.StateLayout((state_size,)),
        system_id=f"sing-benchmark-{'nonlinear' if nonlinear else 'linear'}-system",
    )
    noise = phx.solver.WienerTerm(
        "sing-benchmark-noise",
        lambda time, state, args: dispersion,
        (state_size,),
        structure="additive",
        basis_id="sing-benchmark-basis",
    )
    transition = phx.stochastic.EulerMaruyamaTransitionKernel(
        system,
        (noise,),
        state_shape=(state_size,),
        noise_shape=(state_size,),
        process_id="sing-benchmark-euler",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((state_size,)),
        0.7 * jnp.eye(state_size),
        state_shape=(state_size,),
        prior_id="sing-benchmark-prior",
    )
    return prior, transition


def _linear_reference(
    prior,
    observation,
    observations,
    state_size: int,
) -> phx.stochastic.StateSpaceProblem:
    diagonal = jnp.linspace(0.12, 0.3, state_size)
    drift = -jnp.diag(diagonal) + 0.04 * jnp.diag(jnp.ones((max(state_size - 1, 0),)), 1)
    identity = jnp.eye(state_size)
    process_covariance = 0.24**2 * identity
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        lambda start, end, context: identity + (end - start) * drift,
        lambda start, end, context: (end - start) * process_covariance,
        state_shape=(state_size,),
        process_id="sing-benchmark-linear-reference",
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="sing-benchmark-linear-reference-model",
        ),
        observations,
        initial_time=0.0,
        problem_id="sing-benchmark-linear-reference-problem",
    )


def _problem(
    scenario: Scenario,
    num_steps: int,
    state_size: int,
) -> tuple[
    phx.stochastic.StateSpaceProblem,
    phx.stochastic.StateSpaceProblem | None,
]:
    times = jnp.linspace(0.02, 2.0, num_steps)
    projection = jnp.linspace(0.5, 1.0, state_size)
    projection = projection / jnp.linalg.norm(projection)
    nonlinear = scenario != "linear-gaussian"
    prior, transition = _latent_model(state_size, nonlinear=nonlinear)

    if scenario == "linear-gaussian":
        values = jnp.sin(1.7 * times)[:, None]
        observation = phx.stochastic.LinearGaussianObservationModel(
            projection[None, :],
            jnp.asarray([[0.12]]),
            state_shape=(state_size,),
            observation_shape=(1,),
            observation_id="sing-benchmark-linear-observation",
        )
    elif scenario == "nonlinear-gaussian":
        values = (0.7 * jnp.sin(1.7 * times) + 0.1 * jnp.cos(4.0 * times))[:, None]
        observation = phx.stochastic.GaussianObservationModel(
            lambda state, time, context: jnp.asarray(
                [projection @ state + 0.12 * (projection @ state) ** 3]
            ),
            jnp.asarray([[0.12]]),
            state_shape=(state_size,),
            observation_shape=(1,),
            observation_id="sing-benchmark-nonlinear-observation",
        )
    else:
        values = jnp.round(2.0 + 1.3 * (1.0 + jnp.sin(1.7 * times)))[:, None]

        def rate(state, time, context):
            del time, context
            return jnp.asarray([jnp.exp(jnp.clip(projection @ state, -12.0, 12.0))])

        def log_prob(value, state, time, mask, context):
            del time, context
            log_rate = jnp.clip(projection @ state, -12.0, 12.0)
            count = value.reshape(())
            term = count * log_rate - jnp.exp(log_rate) - gammaln(count + 1.0)
            return jnp.where(mask.reshape(()), term, 0.0)

        def sample(key, state, time, sample_shape, context):
            del time, context
            value = jr.poisson(
                key,
                jnp.exp(jnp.clip(projection @ state, -12.0, 12.0)),
                shape=sample_shape,
            )
            return value[..., None]

        observation = phx.stochastic.CallableObservationModel(
            rate,
            log_prob,
            sample,
            state_shape=(state_size,),
            observation_shape=(1,),
            observation_id="sing-benchmark-poisson-observation",
        )

    observations = phx.stochastic.ObservationSequence(
        times,
        values,
        case_ids=("benchmark",),
        sequence_id=f"sing-benchmark-{scenario}-{num_steps}-{state_size}",
    )
    problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id=f"sing-benchmark-{scenario}-model",
        ),
        observations,
        initial_time=0.0,
        problem_id=f"sing-benchmark-{scenario}-problem",
    )
    reference = (
        _linear_reference(prior, observation, observations, state_size)
        if scenario == "linear-gaussian"
        else None
    )
    return problem, reference


def _time_call(function: Callable[[], object], repeats: int):
    started = time.perf_counter()
    result = function()
    jax.block_until_ready(result.elbo.total_elbo)
    first_seconds = time.perf_counter() - started
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        jax.block_until_ready(result.elbo.total_elbo)
        durations.append(time.perf_counter() - started)
    return result, first_seconds, durations


def _information_storage_bytes(result: phx.uq.SINGResult) -> int:
    information = result.state.information
    return sum(
        np.asarray(value).nbytes
        for value in (
            information.diagonal_precision,
            information.transition_precision,
            information.information_vector,
        )
    )


def _conversion_timing(
    result: phx.uq.SINGResult,
    method: Literal["sequential", "parallel"],
) -> dict[str, float | str]:
    compiled = jax.jit(
        lambda: phx.uq.gaussian_markov_moments(result.state.information, method=method)
    )
    started = time.perf_counter()
    moments = compiled()
    jax.block_until_ready(moments.means)
    first = time.perf_counter() - started
    started = time.perf_counter()
    moments = compiled()
    jax.block_until_ready(moments.means)
    warm = time.perf_counter() - started
    return {
        "first_seconds": first,
        "warm_seconds": warm,
        "resolved_method": moments.execution_method,
    }


def _measure(
    scenario: Scenario,
    *,
    num_steps: int,
    state_size: int,
    iterations: int,
    repeats: int,
    expectation_method: phx.uq.SINGExpectationMethod,
    execution_method: phx.uq.SINGExecutionMethod,
    num_samples: int,
    order: int,
    max_backtracks: int,
) -> dict[str, object]:
    problem, reference_problem = _problem(scenario, num_steps, state_size)
    key = jr.key(918) if expectation_method == "monte-carlo" else None
    compiled = jax.jit(
        lambda: phx.uq.sing_smoother(
            problem,
            key=key,
            expectation_method=expectation_method,
            method=execution_method,
            num_samples=num_samples,
            order=order,
            max_iterations=iterations,
            max_backtracks=max_backtracks,
        )
    )
    result, first_seconds, durations = _time_call(compiled, repeats)
    report: dict[str, object] = {
        "scenario": scenario,
        "num_steps": num_steps,
        "state_size": state_size,
        "iterations": iterations,
        "expectation_method": expectation_method,
        "requested_execution_method": execution_method,
        "resolved_execution_method": result.execution_method,
        "num_samples": num_samples,
        "gauss_hermite_order": order,
        "first_execution_seconds": first_seconds,
        "minimum_warm_seconds": min(durations),
        "mean_warm_seconds": sum(durations) / len(durations),
        "mean_warm_seconds_per_variational_iteration": (
            sum(durations) / len(durations) / iterations
        ),
        "information_storage_bytes": _information_storage_bytes(result),
        "elbo": float(result.elbo.total_elbo),
        "minimum_elbo_increment": float(
            jnp.min(jnp.diff(result.elbo_history)) if iterations > 1 else 0.0
        ),
        "maximum_natural_residual": float(jnp.max(result.natural_residual_history)),
        "minimum_accepted_step_size": float(jnp.min(result.step_size_history)),
        "valid": bool(jnp.all(result.valid)),
        "converged": bool(jnp.all(result.converged)),
        "status": np.asarray(result.status).tolist(),
        "sequential_conversion": _conversion_timing(result, "sequential"),
        "parallel_conversion": _conversion_timing(result, "parallel"),
    }
    if reference_problem is not None:
        filtered = phx.uq.kalman_filter(reference_problem, method="sequential")
        smoothed = phx.uq.rts_smoother(filtered, method="sequential")
        exact_log_evidence = jnp.sum(filtered.incremental_log_likelihood)
        report.update(
            {
                "exact_log_evidence": float(exact_log_evidence),
                "absolute_elbo_gap": float(
                    jnp.abs(result.elbo.total_elbo - exact_log_evidence)
                ),
                "maximum_mean_error": float(
                    jnp.max(jnp.abs(result.observation_means - smoothed.means))
                ),
                "maximum_covariance_error": float(
                    jnp.max(
                        jnp.abs(result.observation_covariances - smoothed.covariances)
                    )
                ),
            }
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark native SING inference, accuracy, and temporal execution."
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "linear-gaussian",
            "nonlinear-gaussian",
            "nonlinear-poisson",
            "all",
        ),
        default="linear-gaussian",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--state-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--expectation-method",
        choices=("cubature", "unscented", "gauss-hermite", "monte-carlo"),
        default="cubature",
    )
    parser.add_argument(
        "--execution-method",
        choices=("sequential", "parallel", "auto"),
        default="auto",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--max-backtracks", type=int, default=4)
    args = parser.parse_args(argv)
    positive = (
        args.steps,
        args.state_size,
        args.iterations,
        args.repeats,
        args.samples,
        args.order,
    )
    if any(value <= 0 for value in positive) or args.max_backtracks < 0:
        parser.error(
            "steps, state-size, iterations, repeats, samples, and order must be "
            "positive; max-backtracks must be nonnegative"
        )
    scenarios: tuple[Scenario, ...] = (
        ("linear-gaussian", "nonlinear-gaussian", "nonlinear-poisson")
        if args.scenario == "all"
        else (args.scenario,)
    )
    reports = [
        _measure(
            scenario,
            num_steps=args.steps,
            state_size=args.state_size,
            iterations=args.iterations,
            repeats=args.repeats,
            expectation_method=args.expectation_method,
            execution_method=args.execution_method,
            num_samples=args.samples,
            order=args.order,
            max_backtracks=args.max_backtracks,
        )
        for scenario in scenarios
    ]
    print(
        json.dumps(
            {"benchmark_id": "sing-native", "reports": reports},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
