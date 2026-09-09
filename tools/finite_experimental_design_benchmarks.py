#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-channel reference and chunk-scaled photon-count design benchmark.

Run from the repository root, for example::

    python tools/finite_experimental_design_benchmarks.py --parameters 256 --designs 96 --shots 128

The scaling channel is a binomial photon counter: each of N pulses detects at
least one photon with probability 1-exp(-exposure*(cross_section+dark_rate)).
Outcomes include EVERY count 0,...,N, rather than a truncated Poisson tail. The
sequential example explicitly simulates a measured count at a declared true
parameter and conditions on that observation, never on an expected outcome.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _space(values):
    return phx.optim.FiniteProductSpace(phx.optim.FiniteAxis(values))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=int, default=256)
    parser.add_argument("--designs", type=int, default=96)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--candidate-batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--outcome-chunks", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--maximum-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--maximum-oracle-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def _reference_channel():
    def channel(parameters, design, outcomes, context):
        answer = parameters == design
        return jnp.where(answer[:, None] == outcomes[None, :], 0.0, -jnp.inf)

    problem = phx.uq.FiniteExperimentalDesignProblem(
        _space(jnp.arange(3)),
        _space(jnp.arange(3)),
        _space(jnp.arange(2)),
        channel,
        likelihood_id="three-hypothesis-query",
        context={"sensor_id": 7},
    )
    belief = phx.uq.FiniteDesignBelief(
        problem.parameters, jnp.log(jnp.asarray([0.6, 0.3, 0.1]))
    )
    policy = phx.uq.ExpectedInformationGain(candidate_batch_size=2, outcome_batch_size=1)
    first = phx.uq.select_finite_experimental_design(problem, belief, policy=policy)
    expected = -0.6 * math.log(0.6) - 0.4 * math.log(0.4)
    np.testing.assert_allclose(
        first.expected_information_gain, expected, atol=1e-12, rtol=1e-12
    )
    assert int(first.design_flat_index) == 0
    observation = phx.uq.bind_finite_design_experiment(
        problem, belief, first, 0, experiment_id="reference-no"
    )
    update = phx.uq.update_finite_design_belief(
        problem, belief, observation, policy=policy
    )
    assert bool(update.accepted)
    np.testing.assert_allclose(
        jnp.exp(update.belief.log_masses), [0.0, 0.75, 0.25], atol=1e-12
    )
    second = phx.uq.select_finite_experimental_design(
        problem, update.belief, policy=policy
    )
    assert int(second.design_flat_index) == 1
    return {
        "first_design_flat_index": int(first.design_flat_index),
        "eig_nats": float(first.expected_information_gain),
        "analytic_eig_nats": expected,
        "observed_outcome_flat_index": 0,
        "posterior_masses": np.exp(np.asarray(update.belief.log_masses)).tolist(),
        "next_design_flat_index": int(second.design_flat_index),
    }


def _photon_problem(arguments):
    parameters = _space(jnp.geomspace(0.05, 5.0, arguments.parameters))
    designs = _space({"exposure": jnp.geomspace(0.005, 10.0, arguments.designs)})
    outcomes = _space(jnp.arange(arguments.shots + 1))

    def channel(cross_section, design, counts, context):
        n = context["shots"]
        dose = design["exposure"] * (cross_section + context["dark_rate"])
        log_success = jnp.log(-jnp.expm1(-dose))
        log_failure = -dose
        combinations = (
            jax.scipy.special.gammaln(n + 1)
            - jax.scipy.special.gammaln(counts + 1)
            - jax.scipy.special.gammaln(n - counts + 1)
        )
        return (
            combinations[None, :]
            + counts[None, :] * log_success[:, None]
            + (n - counts)[None, :] * log_failure[:, None]
        )

    problem = phx.uq.FiniteExperimentalDesignProblem(
        parameters,
        designs,
        outcomes,
        channel,
        likelihood_id="binomial-photon-counter",
        context={"shots": jnp.asarray(arguments.shots, dtype=float), "dark_rate": 0.01},
    )
    rates = parameters.take(jnp.arange(parameters.size))
    log_masses = -0.5 * ((jnp.log(rates) - jnp.log(0.4)) / 0.7) ** 2
    return problem, phx.uq.FiniteDesignBelief(parameters, log_masses)


def _dense_one_design_oracle(problem, belief, design_index, maximum_bytes):
    p, o = problem.parameters.size, problem.outcomes.size
    needed = 8 * p * o * np.dtype(belief.log_masses.dtype).itemsize
    if needed > maximum_bytes:
        raise MemoryError(
            f"One-design dense oracle requires {needed} reserved bytes; increase --maximum-oracle-bytes."
        )
    rates = np.asarray(problem.parameters.take(jnp.arange(p)))
    exposure = float(problem.designs.take(design_index)["exposure"])
    n = o - 1
    counts = np.arange(o)
    dose = exposure * (rates + float(problem.context["dark_rate"]))
    combinations = np.asarray(
        [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in counts]
    )
    logs = (
        combinations[None, :]
        + counts[None, :] * np.log(-np.expm1(-dose[:, None]))
        - (n - counts)[None, :] * dose[:, None]
    )
    joint = np.asarray(belief.log_masses)[:, None] + logs
    maximum = joint.max(axis=0)
    predictive = maximum + np.log(np.exp(joint - maximum).sum(axis=0))
    return float(np.sum(np.exp(joint) * (logs - predictive)))


def _memory_report(executable):
    memory = executable.memory_analysis()
    return {
        "argument_bytes": int(memory.argument_size_in_bytes),
        "output_bytes": int(memory.output_size_in_bytes),
        "temporary_bytes": int(memory.temp_size_in_bytes),
        "alias_bytes": int(memory.alias_size_in_bytes),
        "generated_code_bytes": int(memory.generated_code_size_in_bytes),
    }


def _benchmark_chunks(problem, belief, arguments, batch_size, outcome_chunk, oracle):
    policy = phx.uq.ExpectedInformationGain(
        candidate_batch_size=batch_size,
        outcome_batch_size=outcome_chunk,
        maximum_bytes=arguments.maximum_bytes,
    )
    resources = policy.preflight(problem, belief, selection=True)
    indices = jnp.arange(resources.candidate_batch_size, dtype=jnp.int64)
    # Explicit outer vmap matches the preflight candidate batch. The public scalar
    # scorer does not claim to budget arbitrary external batching.
    score_batch = jax.jit(
        jax.vmap(
            lambda index: phx.uq.evaluate_finite_experimental_design(
                problem, belief, index, policy=policy
            )
        )
    )
    started = perf_counter()
    executable = score_batch.lower(indices).compile()
    compilation_seconds = perf_counter() - started
    result = jax.block_until_ready(executable(indices))
    assert bool(jnp.all(result.valid))
    np.testing.assert_allclose(
        result.expected_information_gain[0], oracle, atol=2e-11, rtol=2e-11
    )
    samples = []
    for _ in range(arguments.repeat):
        started = perf_counter()
        result = jax.block_until_ready(executable(indices))
        samples.append(perf_counter() - started)
    started = perf_counter()
    selected = phx.uq.select_finite_experimental_design(problem, belief, policy=policy)
    jax.block_until_ready(selected)
    selection_seconds = perf_counter() - started
    assert bool(selected.valid)
    assert selected.search_result.landscape_scores is None
    median = float(np.median(samples))
    report = {
        "candidate_batch_size": resources.candidate_batch_size,
        "outcome_chunk_size": resources.outcome_batch_size,
        "preflight_working_bytes": resources.estimated_working_bytes,
        "unallocated_dense_likelihood_bytes": resources.dense_log_likelihood_bytes,
        "compiled_memory": _memory_report(executable),
        "batch_compilation_seconds": compilation_seconds,
        "batch_median_seconds": median,
        "candidate_scores_per_second": resources.candidate_batch_size / median,
        "selection_seconds_including_compilation": selection_seconds,
        "selected_design_flat_index": int(selected.design_flat_index),
        "selected_exposure": float(selected.design["exposure"]),
        "selected_eig_nats": float(selected.expected_information_gain),
        "evaluated_designs": int(selected.search_result.attempted_evaluations),
        "invalid_designs": int(selected.search_result.invalid_evaluations),
    }
    return report, selected, policy


def run_benchmark(arguments):
    for name in (
        "parameters",
        "designs",
        "shots",
        "maximum_bytes",
        "maximum_oracle_bytes",
        "repeat",
    ):
        if vars(arguments)[name] <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if any(
        value <= 0 for value in (*arguments.candidate_batches, *arguments.outcome_chunks)
    ):
        raise ValueError("Candidate batches and outcome chunks must be positive.")
    reference = _reference_channel()
    problem, belief = _photon_problem(arguments)
    oracle = _dense_one_design_oracle(problem, belief, 0, arguments.maximum_oracle_bytes)
    scaling = []
    selected = None
    policy = None
    for batch_size in arguments.candidate_batches:
        for outcome_chunk in arguments.outcome_chunks:
            report, selected, policy = _benchmark_chunks(
                problem, belief, arguments, batch_size, outcome_chunk, oracle
            )
            if scaling:
                assert (
                    report["selected_design_flat_index"]
                    == scaling[0]["selected_design_flat_index"]
                )
                np.testing.assert_allclose(
                    report["selected_eig_nats"],
                    scaling[0]["selected_eig_nats"],
                    atol=2e-11,
                    rtol=2e-11,
                )
            scaling.append(report)
    assert selected is not None and policy is not None
    true_parameter_index = problem.parameters.size // 2
    true_parameter = problem.parameters.take(jnp.asarray([true_parameter_index]))
    likelihood = problem.log_conditional_probability(
        true_parameter,
        selected.design,
        problem.outcomes.take(jnp.arange(problem.outcomes.size)),
        problem.context,
    )[0]
    observed_index = int(
        jax.random.categorical(jax.random.PRNGKey(arguments.seed), likelihood)
    )
    experiment = phx.uq.bind_finite_design_experiment(
        problem,
        belief,
        selected,
        observed_index,
        experiment_id="simulated-photon-measurement",
    )
    updated = phx.uq.update_finite_design_belief(
        problem, belief, experiment, policy=policy
    )
    assert bool(updated.accepted)
    following = phx.uq.select_finite_experimental_design(
        problem, updated.belief, policy=policy
    )
    assert bool(following.valid)
    return {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "reference": reference,
        "photon_channel": {
            "parameters": problem.parameters.size,
            "designs": problem.designs.size,
            "complete_outcomes": problem.outcomes.size,
            "one_design_dense_oracle_nats": oracle,
        },
        "chunk_scaling": scaling,
        "sequential_simulated_measurement": {
            "seed": arguments.seed,
            "true_cross_section": float(true_parameter[0]),
            "observed_photon_count": int(experiment.observations),
            "log_predictive_probability": float(updated.log_predictive_probability),
            "posterior_log_masses": np.asarray(updated.belief.log_masses).tolist(),
            "next_design_flat_index": int(following.design_flat_index),
            "next_exposure": float(following.design["exposure"]),
            "next_eig_nats": float(following.expected_information_gain),
        },
    }


def main():
    print(json.dumps(run_benchmark(_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
