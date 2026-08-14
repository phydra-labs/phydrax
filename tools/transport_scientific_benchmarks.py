#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import coordax as cx
import jax
import jax.numpy as jnp

import phydrax as phx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark scientific native-transport integration paths."
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _timed(operation, ready, repeats):
    result = operation()
    jax.block_until_ready(ready(result))
    started = time.perf_counter()
    for _ in range(repeats):
        result = operation()
        jax.block_until_ready(ready(result))
    return result, 1e3 * (time.perf_counter() - started) / repeats


def _spatial_density_record(size: int, repeats: int):
    parameter = jnp.linspace(0.0, 1.0, size)
    points = (parameter**2)[:, None]
    widths = jnp.diff(
        jnp.concatenate(
            [
                jnp.asarray([0.0]),
                0.5 * (points[1:, 0] + points[:-1, 0]),
                jnp.asarray([1.0]),
            ]
        )
    )
    source_weights = widths * jnp.exp(-8.0 * (points[:, 0] - 0.35) ** 2)
    target_weights = widths * jnp.exp(-8.0 * (points[:, 0] - 0.65) ** 2)
    source = phx.integration.discrete(
        points,
        cx.Field(source_weights, dims=("atom",)),
        axes="atom",
        normalized=True,
    )
    target = phx.integration.discrete(
        points,
        cx.Field(target_weights, dims=("atom",)),
        axes="atom",
        normalized=True,
    )
    problem = phx.transport.discrete_problem(
        source,
        target,
        cost=phx.transport.SquaredEuclideanCost(),
    )
    solver = phx.transport.Sinkhorn(
        0.2,
        max_iterations=300,
        tolerance=1e-6,
        check_every=5,
        block_size=32,
    )
    result, steady_ms = _timed(
        lambda: phx.transport.sinkhorn_divergence(problem, solver),
        lambda value: value.value,
        repeats,
    )
    return {
        "scenario": "translated-nonuniform-spatial-density",
        "size": size,
        "steady_ms": steady_ms,
        "value": float(result.value),
        "converged": bool(result.converged),
        "cross_residual": float(result.cross.diagnostics.normalized_marginal_residual),
    }


def _operator_record(size: int, samples: int, repeats: int):
    nodes = jnp.linspace(0.0, 1.0, size)
    coordinates = jnp.broadcast_to(nodes[None, :, None], (2, size, 1))
    quadrature = jnp.linspace(1.0, 2.0, size)
    quadrature = quadrature / jnp.sum(quadrature)
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=jnp.broadcast_to(quadrature, (2, size)),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.zeros((2, size)),
        coordinates=coordinates,
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"state": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(2,),
    )
    indices = jnp.arange(samples, dtype=float)[:, None, None]
    base = jnp.sin(2.0 * jnp.pi * nodes)[None, None, :] + 0.03 * indices
    left_values = jnp.broadcast_to(base, (samples, 2, size))
    right_values = left_values + jnp.asarray([0.0, 0.4])[None, :, None]

    def predictive(values, dim):
        return phx.uq.operator_predictive_from_samples(
            values,
            batch,
            phx.nn.operator.OperatorOutputSpec("scalar"),
            sample_axes=(phx.uq.SampleAxis(dim, "process"),),
            field_name="output",
            query_name="query",
        )

    left = predictive(left_values, "left")
    right = predictive(right_values, "right")
    result, steady_ms = _timed(
        lambda: phx.uq.operator_ensemble_sinkhorn_divergence(
            left,
            right,
            epsilon=1.0,
            reduction="none",
        ),
        lambda value: value.value,
        repeats,
    )
    return {
        "scenario": "whole-field-operator-ensemble",
        "size": size,
        "samples": samples,
        "steady_ms": steady_ms,
        "per_case": [float(value) for value in result.per_case],
        "converged": [bool(value) for value in result.transport.converged],
    }


def _particle_record(size: int, repeats: int):
    particles = jnp.stack(
        [jnp.linspace(-1.0, 1.0, size), jnp.linspace(0.0, 2.0, size)],
        axis=1,
    )
    weights = jnp.exp(jnp.linspace(-4.0, 0.0, size))
    result, steady_ms = _timed(
        lambda: phx.uq.optimal_transport_ensemble_transform(
            particles,
            weights,
            epsilon=1.0,
        ),
        lambda value: value.particles,
        repeats,
    )
    return {
        "scenario": "deterministic-particle-transform",
        "size": size,
        "steady_ms": steady_ms,
        "converged": bool(result.transport.converged),
        "marginal_residual": float(
            result.transport.diagnostics.normalized_marginal_residual
        ),
        "mean_error": float(jnp.linalg.norm(result.mean_error)),
    }


def main() -> None:
    arguments = _parser().parse_args()
    size = 8 if arguments.smoke else int(arguments.size)
    samples = 6 if arguments.smoke else int(arguments.samples)
    repeats = 1 if arguments.smoke else int(arguments.repeats)
    records = [
        _spatial_density_record(size, repeats),
        _operator_record(size, samples, repeats),
        _particle_record(size, repeats),
    ]
    print(json.dumps({"records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
