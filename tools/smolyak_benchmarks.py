#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import logical_array_bytes
from phydrax._numerics import (
    smolyak_terms,
    SmolyakAxisRule,
    weighted_total_degree_indices,
)
from phydrax.integration._sparse_grid import _smolyak_rule
from phydrax.operators.interpolation._smolyak import _build_topology


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Phydrax Smolyak construction, fitting, and evaluation."
    )
    parser.add_argument("--dimensions", type=int, nargs="+", default=(4, 8, 16, 32))
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument(
        "--rules",
        nargs="+",
        choices=("leja", "gauss-hermite"),
        default=("leja", "gauss-hermite"),
    )
    parser.add_argument("--query-sizes", type=int, nargs="+", default=(1, 64))
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def _domain(dimension: int, rule: SmolyakAxisRule):
    if rule == "gauss-hermite":
        factors = tuple(
            phx.domain.ProbabilityDomain(
                phx.uq.Normal(0.0, 1.0),
                label=f"x{axis}",
            )
            for axis in range(dimension)
        )
    else:
        factors = tuple(
            phx.domain.ScalarInterval(-1.0, 1.0, label=f"x{axis}")
            for axis in range(dimension)
        )
    return phx.domain.ProductDomain(*factors)


def _target(domain, dimension: int, output_size: int):
    labels = tuple(f"x{axis}" for axis in range(dimension))
    frequencies = jnp.arange(1, output_size + 1, dtype=float)

    def observable(*coordinates):
        signal = sum(
            (axis + 1.0) / dimension * (coordinate + 0.1 * coordinate**2)
            for axis, coordinate in enumerate(coordinates)
        )
        if output_size == 1:
            return signal
        return jnp.sin(frequencies * signal)

    return domain.Function(*labels)(observable)


def _interpolation_record(
    dimension: int,
    level: int,
    rule: SmolyakAxisRule,
    output_size: int,
    query_sizes: Sequence[int],
    repeats: int,
):
    anisotropy = tuple(
        1.0 + 2.0 * axis / max(dimension - 1, 1) for axis in range(dimension)
    )

    started = time.perf_counter()
    indices = weighted_total_degree_indices(dimension, level, anisotropy)
    terms = smolyak_terms(dimension, level, anisotropy)
    index_ms = 1e3 * (time.perf_counter() - started)

    rules: tuple[SmolyakAxisRule, ...] = (rule,) * dimension
    started = time.perf_counter()
    canonical_points, topology = _build_topology(
        dimension,
        level,
        anisotropy,
        rules,
    )
    topology_ms = 1e3 * (time.perf_counter() - started)
    tensor_entries = sum(int(term.gather_indices.size) for term in topology)

    domain = _domain(dimension, rule)
    function = _target(domain, dimension, output_size)
    plan = phx.operators.SmolyakInterpolationPlan(
        dimension,
        level,
        anisotropy=anisotropy,
        axis_rules=rule,
    )
    started = time.perf_counter()
    approximation = phx.operators.interpolate_smolyak(function, plan)
    interpolant = approximation.func
    if not isinstance(interpolant, phx.operators.SmolyakInterpolant):
        raise RuntimeError("Smolyak interpolation did not return a SmolyakInterpolant.")
    jax.block_until_ready(interpolant.blocks[-1].values)
    fit_ms = 1e3 * (time.perf_counter() - started)

    query_records = []
    for query_size in query_sizes:
        reference = jnp.linspace(-0.7, 0.7, query_size * dimension).reshape(
            (query_size, dimension)
        )

        def evaluate(rows):
            return jax.vmap(
                lambda row: interpolant(*tuple(row[axis] for axis in range(dimension)))
            )(rows)

        compiled = jax.jit(evaluate)
        started = time.perf_counter()
        observed = compiled(reference)
        jax.block_until_ready(observed)
        compile_ms = 1e3 * (time.perf_counter() - started)

        started = time.perf_counter()
        for _ in range(repeats):
            observed = compiled(reference)
            jax.block_until_ready(observed)
        steady_ms = 1e3 * (time.perf_counter() - started) / repeats

        coordinates = tuple(reference[:, axis] for axis in range(dimension))
        signal = jnp.zeros_like(coordinates[0])
        for axis, coordinate in enumerate(coordinates):
            signal = signal + (axis + 1.0) / dimension * (
                coordinate + 0.1 * coordinate**2
            )
        expected = (
            signal
            if output_size == 1
            else jnp.sin(jnp.arange(1, output_size + 1) * signal[:, None])
        )
        error = observed - expected
        query_records.append(
            {
                "query_size": query_size,
                "compile_ms": compile_ms,
                "steady_ms": steady_ms,
                "max_abs_error": float(jnp.max(jnp.abs(error))),
                "rms_error": float(jnp.sqrt(jnp.mean(jnp.abs(error) ** 2))),
            }
        )

    return {
        "kind": "interpolation",
        "dimension": dimension,
        "level": level,
        "rule": rule,
        "anisotropy": anisotropy,
        "output_size": output_size,
        "num_indices": len(indices),
        "num_terms": len(terms),
        "num_unique_nodes": int(canonical_points.shape[0]),
        "tensor_entries": tensor_entries,
        "num_blocks": interpolant.num_blocks,
        "maximum_active_dimension": interpolant.maximum_active_dimension,
        "fitted_bytes": logical_array_bytes(interpolant),
        "index_ms": index_ms,
        "topology_ms": topology_ms,
        "fit_ms": fit_ms,
        "queries": query_records,
    }


def _integration_record(dimension: int, level: int):
    started = time.perf_counter()
    nodes, weights = _smolyak_rule(dimension, level, None)
    construction_ms = 1e3 * (time.perf_counter() - started)
    return {
        "kind": "integration",
        "dimension": dimension,
        "level": level,
        "rule": "clenshaw-curtis",
        "num_unique_nodes": int(nodes.shape[0]),
        "weight_sum": float(jnp.asarray(weights).sum()),
        "expected_reference_mass": float(2.0**dimension),
        "mass_error": abs(float(weights.sum()) - math.pow(2.0, dimension)),
        "construction_ms": construction_ms,
    }


def main() -> None:
    arguments = _parser().parse_args()
    if any(dimension < 1 for dimension in arguments.dimensions):
        raise ValueError("dimensions must be positive.")
    if arguments.level < 1:
        raise ValueError("level must be positive.")
    if arguments.output_size < 1 or arguments.repeats < 1:
        raise ValueError("output-size and repeats must be positive.")
    if any(query_size < 1 for query_size in arguments.query_sizes):
        raise ValueError("query sizes must be positive.")

    records = [
        _interpolation_record(
            dimension,
            arguments.level,
            rule,
            arguments.output_size,
            arguments.query_sizes,
            arguments.repeats,
        )
        for rule in arguments.rules
        for dimension in arguments.dimensions
    ]
    records.extend(
        _integration_record(dimension, arguments.level)
        for dimension in arguments.dimensions
    )
    print(json.dumps({"records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
