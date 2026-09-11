#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized


def _problem(source_count: int, target_count: int):
    rng = np.random.default_rng(390174)
    sources = jnp.asarray(rng.uniform(-0.9, 0.9, size=(source_count, 3)))
    targets = jnp.asarray(rng.uniform(-0.9, 0.9, size=(target_count, 3)))
    strengths = jnp.asarray(rng.normal(size=(source_count,)))
    return sources, targets, strengths


def _reference(kernel, sources, strengths, targets, parameter):
    radii = jnp.linalg.norm(targets[:, None, :] - sources[None, :, :], axis=-1)
    if kernel == "laplace":
        numerator = jnp.ones_like(radii)
    elif kernel == "helmholtz":
        numerator = jnp.exp(1j * parameter * radii)
    else:
        numerator = jnp.exp(-parameter * radii)
    return jnp.sum(numerator * strengths[None, :] / (4.0 * jnp.pi * radii), axis=1)


def _case(kernel, sources, targets, strengths, order, depth, repeats, parameter):
    if kernel == "laplace":
        plan_type, keyword = phx.operators.LaplaceMultipolePlan3D, {}
    elif kernel == "helmholtz":
        plan_type, keyword = (
            phx.operators.HelmholtzMultipolePlan3D,
            {"wavenumber": parameter},
        )
    else:
        plan_type, keyword = (
            phx.operators.ModifiedHelmholtzMultipolePlan3D,
            {"decay": parameter},
        )
    started = time.perf_counter_ns()
    prepared = plan_type(
        sources,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        reference_targets=targets,
        depth=depth,
        expansion_order=order,
        **keyword,
    ).prepare()
    preparation_ms = (time.perf_counter_ns() - started) / 1.0e6
    compiled = eqx.filter_jit(
        lambda source, weight, target: prepared.evaluate(source, weight, target).values
    )
    first, first_seconds = measure_synchronized(
        lambda: compiled(sources, strengths, targets)
    )
    result, distribution = measure_repeated(
        lambda: compiled(sources, strengths, targets),
        warmup=1,
        repeats=repeats,
    )
    expected = _reference(kernel, sources, strengths, targets, parameter)
    resources = prepared.resources
    return {
        "preparation_ms": preparation_ms,
        "first_jit_ms": 1.0e3 * first_seconds,
        "steady": distribution.to_milliseconds_dict(),
        "maximum_absolute_error": float(jnp.max(jnp.abs(result - expected))),
        "checksum": float(jnp.sum(jnp.abs(first))),
        "logical_modes": resources.logical_mode_count,
        "padded_modes": resources.padded_mode_count,
        "required_coefficient_bytes": resources.required_coefficient_bytes,
        "maximum_coefficient_bytes": resources.maximum_coefficient_bytes,
        "quadrature_nodes": resources.quadrature_node_count,
        "far_interaction_capacity": resources.far_interaction_capacity,
        "near_interaction_capacity": resources.near_interaction_capacity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark complete 3-D multipole passes."
    )
    parser.add_argument("--source-count", type=int, default=16)
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parameter", type=float, default=0.7)
    arguments = parser.parse_args()
    sources, targets, strengths = _problem(arguments.source_count, arguments.target_count)
    output = {
        "source_count": arguments.source_count,
        "target_count": arguments.target_count,
        "expansion_order": arguments.order,
        "depth": arguments.depth,
        "parameter": arguments.parameter,
        "kernels": {
            kernel: _case(
                kernel,
                sources,
                targets,
                strengths,
                arguments.order,
                arguments.depth,
                arguments.repeats,
                arguments.parameter,
            )
            for kernel in ("laplace", "helmholtz", "modified-helmholtz")
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
