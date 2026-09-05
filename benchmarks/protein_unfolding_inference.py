# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real joint temperature/denaturant inference and held-out signal benchmark.

Run as ``python -m benchmarks.protein_unfolding_inference --repeats 3``.
Synthetic data qualify numerical inference only, not experimental accuracy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.applications.protein_folding.experiments import (
    ExperimentConditions,
    ExperimentParameter,
    fit_protein_experiments,
    FluorescenceExperiment,
    prepare_protein_experiments,
    protein_experiment_identifiability,
    sample_protein_experiments,
    TwoStateUnfolding,
)
from phydrax.optim import OptimizationTermination


def _signal(temperature, denaturant, channels):
    dg = (
        180 * (1 - temperature / 298.15)
        + 15 * temperature / 298.15
        + 1.2 * (temperature - 298.15 - temperature * np.log(temperature / 298.15))
        - 0.004 * denaturant
    )
    folded = 1 / (1 + np.exp(-dg / (0.00831446261815324 * temperature)))
    return np.where(
        channels == 0,
        1.2 * folded + 0.2 * (1 - folded),
        0.3 * folded + 1.4 * (1 - folded),
    )


def _problem(temperature_count, denaturant_count, *, isotherm=False):
    temperatures = [298.15] if isotherm else np.linspace(289, 358, temperature_count)
    t, d = np.meshgrid(
        temperatures, np.linspace(0, 6000, denaturant_count), indexing="ij"
    )
    t, d = np.repeat(t.ravel(), 2), np.repeat(d.ravel(), 2)
    channel = np.arange(t.size) % 2
    groups = tuple(
        "channel-a:replicate-1" if value == 0 else "channel-b:replicate-1"
        for value in channel
    )
    observed = _signal(t, d, channel)
    # A deterministic, declared perturbation avoids treating a perfect solver
    # interpolation as the only inference case. It is not measured uncertainty.
    observed = observed + 0.0005 * np.sin(np.arange(t.size) * np.sqrt(2.0))
    model = TwoStateUnfolding()
    plan = FluorescenceExperiment(
        "joint",
        model,
        ExperimentConditions(t, d),
        groups,
        observed,
        np.full(t.size, 0.02),
        "synthetic:protein-unfolding-benchmark",
        "Equilibrium-generating formula, no ramp or aggregation",
        True,
        baseline_terms=("intercept",),
    )
    values = [13.0, 170.0, 1.2, 0.0035, 0.0, 1.1, 0.25, 0.35, 1.3]
    scales = [10.0, 100.0, 1.0, 0.004, 1e-5, 1.0, 1.0, 1.0, 1.0]
    parameters = tuple(
        ExperimentParameter(name, value, unit, scale, free=i not in (2, 4))
        for i, ((name, unit), value, scale) in enumerate(
            zip(plan.parameter_slots(), values, scales, strict=True)
        )
    )
    return prepare_protein_experiments((plan,), parameters)


def run_case(*, temperature_count, denaturant_count, repeats, posterior_samples):
    problem, preparation_seconds = measure_synchronized(
        lambda: _problem(temperature_count, denaturant_count)
    )
    residual = jax.jit(lambda coordinates: problem.residual(coordinates))
    compiled, compilation = measure_lower_and_compile(
        lambda: residual.lower(problem.initial_coordinates),
        lambda lowered: lowered.compile(),
    )
    _, residual_time = measure_repeated(
        lambda: compiled(problem.initial_coordinates), warmup=1, repeats=repeats
    )
    native_compiler = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-residual-executable",
    )
    termination = OptimizationTermination(maximum_steps=100)
    fit, cold_fit_seconds = measure_synchronized(
        lambda: fit_protein_experiments(problem, termination=termination)
    )
    fit, fit_timings = measure_repeated(
        lambda: fit_protein_experiments(problem, termination=termination),
        warmup=0,
        repeats=repeats,
    )

    t = np.repeat(np.linspace(293, 353, 23), 2)
    d = np.repeat(np.linspace(125, 5875, 23), 2)
    channel = np.arange(t.size) % 2
    groups = tuple(
        "channel-a:replicate-1" if value == 0 else "channel-b:replicate-1"
        for value in channel
    )
    heldout = problem.observations[0].prepare_prediction(
        ExperimentConditions(t, d), groups=groups
    )
    predicted = heldout(problem.parameters.decode(fit.coordinates))
    heldout_error = np.asarray(predicted) - _signal(t, d, channel)
    deficient = protein_experiment_identifiability(
        _problem(temperature_count, denaturant_count, isotherm=True)
    )
    report = {
        "qualification": "synthetic numerical inference, not experimental validation",
        "model": "reversible two-state, constant dCp, linear denaturant; two channel baselines",
        "temperature_count": temperature_count,
        "denaturant_count": denaturant_count,
        "channel_replicate_count": 2,
        "active_observations": int(problem.residual(problem.initial_coordinates).size),
        "free_parameter_names": problem.parameters.free_names,
        "preparation_seconds": preparation_seconds,
        "residual_compilation": asdict(compilation),
        "residual_execution": residual_time.to_dict(),
        "residual_compiler": asdict(native_compiler),
        "cold_complete_fit_seconds": cold_fit_seconds,
        "repeated_complete_fit_including_host_and_internal_compilation": fit_timings.to_dict(),
        "logical_problem_bytes": logical_array_bytes(problem),
        "fit_solver_status": int(fit.optimization.status),
        "fit_solver_successful": bool(fit.optimization.successful),
        "fit_objective": float(fit.optimization.objective),
        "local_likelihood_rank": fit.identifiability.rank,
        "local_condition_number": fit.identifiability.condition_number,
        "fit_covariance_available": fit.covariance is not None,
        "heldout_rmse": float(np.sqrt(np.mean(heldout_error**2))),
        "heldout_max_absolute_error": float(np.max(np.abs(heldout_error))),
        "isotherm_rank": deficient.rank,
        "isotherm_free_parameters": len(deficient.free_names),
        "isotherm_remains_nonidentifiable": not deficient.locally_identifiable,
        "posterior": None,
    }
    if posterior_samples:
        posterior, posterior_seconds = measure_synchronized(
            lambda: sample_protein_experiments(
                problem,
                key=jax.random.key(9021),
                prior_mean=np.zeros(len(problem.parameters.free_names)),
                prior_standard_deviation=np.full(len(problem.parameters.free_names), 3.0),
                initial_coordinates=fit.coordinates,
                num_chains=2,
                num_warmup=150,
                num_samples=posterior_samples,
            )
        )
        mean_draws = posterior.predictive_samples()[0]
        report["posterior"] = {
            "seconds": posterior_seconds,
            "chain_draw_shape": posterior.mcmc.log_density.shape,
            "divergences": int(jnp.sum(posterior.mcmc.divergent)),
            "mean_acceptance": float(jnp.mean(posterior.mcmc.acceptance_rate)),
            "conditional_mean_interval_average_width": float(
                jnp.mean(
                    jnp.quantile(mean_draws, 0.975, axis=(0, 1))
                    - jnp.quantile(mean_draws, 0.025, axis=(0, 1))
                )
            ),
            "interval_kind": "posterior conditional-mean draws, not fit covariance or observation noise",
        }
    report["successful"] = (
        report["fit_solver_successful"]
        and report["heldout_rmse"] < 0.002
        and fit.identifiability.locally_identifiable
        and report["isotherm_remains_nonidentifiable"]
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperatures", type=int, default=9)
    parser.add_argument("--denaturants", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--posterior-samples", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.temperatures < 3
        or args.denaturants < 4
        or args.repeats < 1
        or args.posterior_samples < 0
    ):
        parser.error(
            "Need >=3 temperatures, >=4 denaturants, >=1 repeat and nonnegative posterior draws."
        )
    report = {
        "environment": capture_environment().to_dict(),
        "case": run_case(
            temperature_count=args.temperatures,
            denaturant_count=args.denaturants,
            repeats=args.repeats,
            posterior_samples=args.posterior_samples,
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)
    if not report["case"]["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
