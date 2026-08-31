#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from benchmarks._runtime import measure_repeated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark exact finite-state Schrödinger bridge solves and sampling."
    )
    parser.add_argument("--states", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _target(states, probabilities, provenance):
    return phx.integration.discrete(
        states,
        cx.Field(probabilities, dims=("state",)),
        axes="state",
        normalized=True,
        provenance=provenance,
    )


def _kernel(matrix):
    def sample(key, state, _t0, _t1, _context):
        probability = matrix[jnp.asarray(state, dtype=jnp.int32)]
        return jr.categorical(key, jnp.log(probability)).astype(float)

    def log_prob(next_state, state, _t0, _t1, _context):
        probability = matrix[
            jnp.asarray(state, dtype=jnp.int32),
            jnp.asarray(next_state, dtype=jnp.int32),
        ]
        return jnp.where(probability > 0.0, jnp.log(probability), -jnp.inf)

    return phx.stochastic.CallableTransitionKernel(
        sample,
        state_shape=(),
        process_id="benchmark-local-chain-with-global-mixing",
        approximation_id="exact-finite-matrix",
        log_prob_fn=log_prob,
    )


def _problem(num_states, num_steps):
    states = jnp.arange(num_states, dtype=float)
    indices = jnp.arange(num_states)
    distance = jnp.abs(indices[:, None] - indices[None, :])
    matrix = jnp.where(distance == 0, 0.7, jnp.where(distance == 1, 0.15, 0.0))
    matrix = matrix / jnp.sum(matrix, axis=-1, keepdims=True)
    matrix = 0.95 * matrix + 0.05 / num_states
    coordinate = jnp.linspace(-1.0, 1.0, num_states)
    initial = jax.nn.softmax(-12.0 * (coordinate + 0.45) ** 2)
    terminal = jax.nn.softmax(-12.0 * (coordinate - 0.45) ** 2)
    return phx.transport.dynamic.SchrodingerBridgeProblem(
        _target(states, initial, "benchmark-initial"),
        _target(states, terminal, "benchmark-terminal"),
        jnp.linspace(0.0, 1.0, num_steps + 1),
        _kernel(matrix),
        phx.stochastic.StateSpaceStepContext.empty(),
    )


def _timed(operation, ready, repeats):
    result, distribution = measure_repeated(
        operation,
        warmup=1,
        repeats=repeats,
        synchronizer=lambda value: jax.block_until_ready(ready(value)),
    )
    return result, 1_000.0 * float(distribution.mean_seconds)


def main() -> None:
    arguments = _parser().parse_args()
    states = 8 if arguments.smoke else int(arguments.states)
    steps = 3 if arguments.smoke else int(arguments.steps)
    samples = 128 if arguments.smoke else int(arguments.samples)
    repeats = 1 if arguments.smoke else int(arguments.repeats)
    problem = _problem(states, steps)
    solver = phx.transport.dynamic.SchrodingerBridgeSolver(
        max_iterations=500, tolerance=1e-9
    )
    compiled = eqx.filter_jit(solver.solve)
    result, solve_ms = _timed(
        lambda: compiled(problem),
        lambda value: value.diagnostics.endpoint_residual,
        repeats,
    )
    path_sample, sample_ms = _timed(
        lambda: phx.transport.dynamic.sample_bridge(
            jr.key(2026), result, sample_shape=(samples,)
        ),
        lambda value: value.values,
        repeats,
    )
    diagnostics = phx.transport.dynamic.bridge_path_law_diagnostics(result, path_sample)
    record = {
        "scenario": "exact-finite-state-schrodinger-bridge",
        "states": states,
        "steps": steps,
        "samples": samples,
        "solve_ms": solve_ms,
        "sample_ms": sample_ms,
        "converged": bool(result.converged),
        "iterations": int(result.diagnostics.num_iterations),
        "endpoint_residual": float(result.diagnostics.endpoint_residual),
        "path_kl": float(result.diagnostics.path_kl),
        "empirical_marginal_residual": float(diagnostics.empirical_marginal_residual),
        "reference_process": result.provenance.reference_process,
        "approximation": result.provenance.approximation,
    }
    print(json.dumps({"records": [record]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
