#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.solver._variational_monte_carlo import (
    _energy_force,
    _estimate_from_samples,
    _model_log_target,
    _score_geometry,
)


def _seconds(callable_, /):
    started = perf_counter()
    value = callable_()
    jax.block_until_ready(value)
    return value, perf_counter() - started


def markov_benchmark(*, chains: int, draws: int, dimension: int):
    proposal = phx.sampling.GaussianRandomWalkProposal(0.2)
    kernel = phx.sampling.MetropolisHastings(proposal)
    initial = jnp.zeros((chains, dimension))

    def log_target(value):
        return -0.5 * jnp.sum(value**2)

    state = kernel.initialize(log_target, initial)

    @eqx.filter_jit
    def run(current, key):
        return phx.sampling.sample_markov(
            log_target,
            kernel,
            current,
            key=key,
            num_draws=draws,
            steps_per_draw=2,
            warmup_steps=16,
        )

    first, compile_seconds = _seconds(lambda: run(state, jr.key(0)))
    second, steady_seconds = _seconds(lambda: run(state, jr.key(1)))
    return {
        "case": "markov",
        "chains": chains,
        "draws": draws,
        "dimension": dimension,
        "transitions": chains * (16 + 2 * draws),
        "compile_and_first_seconds": compile_seconds,
        "steady_seconds": steady_seconds,
        "steady_transitions_per_second": chains * (16 + 2 * draws) / steady_seconds,
        "mean_acceptance_rate": float(jnp.mean(second.acceptance_rate)),
        "sample_bytes": sum(
            leaf.nbytes for leaf in jax.tree_util.tree_leaves(first.samples)
        ),
    }


def gram_benchmark(*, samples: int, parameters: int):
    matrix = jr.normal(jr.key(2), (samples, parameters))
    weights = jax.nn.softmax(jr.normal(jr.key(3), (samples,)))
    direction = jr.normal(jr.key(4), (parameters,))
    operator = phx.linalg.EmpiricalGramLinearOperator(
        phx.linalg.DenseLinearOperator(matrix),
        weights,
        centered=True,
        damping=1e-3,
    )
    apply = eqx.filter_jit(operator.mv)
    first, compile_seconds = _seconds(lambda: apply(direction))
    second, steady_seconds = _seconds(lambda: apply(direction))
    dense, dense_seconds = _seconds(
        lambda: (
            (matrix - jnp.sum(weights[:, None] * matrix, axis=0)).T
            @ (weights[:, None] * (matrix - jnp.sum(weights[:, None] * matrix, axis=0)))
            + 1e-3 * jnp.eye(parameters)
        )
    )
    residual = jnp.linalg.norm(second - dense @ direction)
    return {
        "case": "empirical-gram",
        "samples": samples,
        "parameters": parameters,
        "rank_upper_bound": operator.rank_upper_bound,
        "weight_ess": float(operator.weight_ess),
        "compile_and_first_seconds": compile_seconds,
        "steady_action_seconds": steady_seconds,
        "dense_materialization_seconds": dense_seconds,
        "matrix_free_storage_bytes": int(matrix.nbytes + weights.nbytes),
        "dense_storage_bytes": int(dense.nbytes),
        "dense_action_residual": float(residual),
        "output_norm": float(jnp.linalg.norm(first)),
    }


class _TableModel(eqx.Module):
    parameters: jax.Array

    def __call__(self, configuration):
        bits = (configuration > 0).astype(jnp.int32)
        index = 2 * bits[0] + bits[1]
        return phx.operators.LogAmplitude(self.parameters[index], 1.0 + 0.0j)


def _ising_operator():
    def diagonal(configurations):
        return -configurations[..., 0] * configurations[..., 1]

    def connections(configurations):
        first = configurations.at[..., 0].multiply(-1)
        second = configurations.at[..., 1].multiply(-1)
        values = jnp.stack((first, second), axis=-2)
        shape = configurations.shape[:-1] + (2,)
        return phx.operators.ConnectedConfigurations(
            values,
            -0.5 * jnp.ones(shape),
            jnp.ones(shape, dtype=bool),
            configuration_shape=(2,),
        )

    return phx.operators.CallableDiscreteQuantumOperator(
        diagonal,
        connections,
        configuration_shape=(2,),
        operator_id="benchmark-ising",
    )


def _spin_kernel():
    def sample(key, current):
        index = jr.randint(key, (), 0, current.shape[0])
        return current.at[index].multiply(-1)

    def log_prob(_proposed, current):
        return -jnp.log(float(current.shape[0]))

    return phx.sampling.MetropolisHastings(
        phx.sampling.CallableProposal(
            sample,
            log_prob,
            proposal_id="benchmark-spin-flip",
        )
    )


def vmc_benchmark(*, iterations: int, draws: int):
    problem = phx.solver.VariationalMonteCarloProblem(
        _TableModel(jnp.asarray([0.2, -0.1, 0.1, -0.2])),
        _ising_operator(),
        _spin_kernel(),
        jnp.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=jnp.int32),
    )
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=iterations,
        draws_per_iteration=draws,
        steps_per_draw=2,
        warmup_steps=8,
        final_evaluation_draws=draws,
        learning_rate=0.03,
        damping=0.1,
    )
    result, elapsed = _seconds(
        lambda: phx.solver.solve_variational_monte_carlo(
            problem,
            policy,
            key=jr.key(5),
        )
    )
    return {
        "case": "vmc",
        "iterations": iterations,
        "draws_per_iteration": draws,
        "seconds": elapsed,
        "initial_energy": float(jnp.real(result.energy_history[0]))
        if result.energy_history.size
        else None,
        "final_energy": float(result.final_estimate.physical_energy),
        "final_variance": float(result.final_estimate.variance),
        "mean_acceptance_rate": float(jnp.mean(result.acceptance_history))
        if result.acceptance_history.size
        else float(result.final_estimate.acceptance_rate),
        "successful": bool(result.successful),
    }


class _JastrowChain(eqx.Module):
    fields: jax.Array
    bonds: jax.Array

    def __call__(self, configuration):
        log_abs = jnp.vdot(self.fields, configuration) + jnp.vdot(
            self.bonds, configuration * jnp.roll(configuration, -1)
        )
        return phx.operators.LogAmplitude(jnp.real(log_abs), 1.0 + 0.0j)


def _spin_chain_operator(num_sites: int, *, coupling: float, field: float):
    def diagonal(configurations):
        return -coupling * jnp.sum(
            configurations * jnp.roll(configurations, -1, axis=-1), axis=-1
        )

    def connections(configurations):
        values = jnp.stack(
            [configurations.at[..., site].multiply(-1) for site in range(num_sites)],
            axis=-2,
        )
        shape = configurations.shape[:-1] + (num_sites,)
        return phx.operators.ConnectedConfigurations(
            values,
            -field * jnp.ones(shape),
            jnp.ones(shape, dtype=bool),
            configuration_shape=(num_sites,),
        )

    return phx.operators.CallableDiscreteQuantumOperator(
        diagonal,
        connections,
        configuration_shape=(num_sites,),
        operator_id=f"periodic-tfim-{num_sites}",
    )


def spin_chain_benchmark(
    *,
    num_sites: int,
    num_chains: int,
    draws: int,
    iterations: int,
):
    parameter_key, chain_key, run_key = jr.split(jr.key(num_sites), 3)
    fields, bonds = jr.normal(parameter_key, (2, num_sites)) * 0.02
    initial = (
        2 * jr.bernoulli(chain_key, shape=(num_chains, num_sites)).astype(jnp.int32) - 1
    )
    problem = phx.solver.VariationalMonteCarloProblem(
        _JastrowChain(fields, bonds),
        _spin_chain_operator(num_sites, coupling=1.0, field=0.8),
        _spin_kernel(),
        initial,
        problem_id=f"benchmark-periodic-tfim-{num_sites}",
    )
    state = problem.initial_state(key=run_key)
    target = _model_log_target(state.model)
    refreshed = problem.kernel.refresh(target, state.markov_state)
    sampled, sampler_seconds = _seconds(
        lambda: phx.sampling.sample_markov(
            target,
            problem.kernel,
            refreshed,
            key=run_key,
            num_draws=draws,
            steps_per_draw=2,
            warmup_steps=16,
        )
    )
    estimate, local_energy_seconds = _seconds(
        lambda: _estimate_from_samples(
            problem,
            state.model,
            sampled,
            energy_imag_tolerance=1e-8,
            compute_chain_diagnostics=False,
        )
    )

    geometry_started = perf_counter()
    score, metric = _score_geometry(
        problem,
        state.parameter_coordinates,
        sampled.samples,
        damping=0.05,
    )
    force = _energy_force(
        score,
        estimate.local.value,
        estimate.energy,
        problem.complex_parameter_mode,
    )
    action = metric.mv(force)
    jax.block_until_ready(action)
    geometry_seconds = perf_counter() - geometry_started

    linear, solve_seconds = _seconds(
        lambda: phx.linalg.solve(phx.linalg.LinearSystem(metric), force)
    )
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=iterations,
        draws_per_iteration=draws,
        steps_per_draw=2,
        warmup_steps=16,
        final_evaluation_draws=draws,
        learning_rate=0.02,
        damping=0.05,
        final_chain_diagnostics=False,
    )
    result, end_to_end_seconds = _seconds(
        lambda: phx.solver.solve_variational_monte_carlo(
            problem,
            policy,
            key=run_key,
        )
    )
    connection_bytes = num_chains * draws * num_sites * num_sites * initial.dtype.itemsize
    return {
        "case": "spin-chain-vmc",
        "num_sites": num_sites,
        "num_chains": num_chains,
        "draws": draws,
        "iterations": iterations,
        "parameters": int(state.parameter_coordinates.size),
        "sampler_seconds": sampler_seconds,
        "local_energy_seconds": local_energy_seconds,
        "geometry_action_seconds": geometry_seconds,
        "linear_solve_seconds": solve_seconds,
        "linear_iterations": int(jnp.max(linear.diagnostics.iterations)),
        "estimated_connected_configuration_bytes": connection_bytes,
        "end_to_end_seconds": end_to_end_seconds,
        "initial_energy": float(jnp.real(result.energy_history[0])),
        "final_energy": float(result.final_estimate.physical_energy),
        "final_variance": float(result.final_estimate.variance),
        "mean_acceptance_rate": float(jnp.mean(result.acceptance_history)),
        "successful": bool(result.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("all", "markov", "gram", "vmc", "spin-chain"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    if args.case in ("all", "markov"):
        records.append(markov_benchmark(chains=32, draws=128, dimension=8))
    if args.case in ("all", "gram"):
        records.extend(
            [
                gram_benchmark(samples=64, parameters=512),
                gram_benchmark(samples=256, parameters=256),
                gram_benchmark(samples=512, parameters=64),
            ]
        )
    if args.case in ("all", "vmc"):
        records.append(vmc_benchmark(iterations=8, draws=64))
    if args.case in ("all", "spin-chain"):
        records.extend(
            [
                spin_chain_benchmark(
                    num_sites=num_sites,
                    num_chains=16,
                    draws=64,
                    iterations=2,
                )
                for num_sites in (8, 12, 16)
            ]
        )
    payload = json.dumps(records, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
