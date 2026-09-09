#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _timed(function, arguments, iterations):
    start = time.perf_counter()
    value = function(*arguments)
    jax.block_until_ready(value)
    compile_seconds = time.perf_counter() - start
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        value = function(*arguments)
        jax.block_until_ready(value)
        samples.append(time.perf_counter() - start)
    return compile_seconds, float(np.median(samples))


def benchmark(*, size: int, iterations: int) -> dict[str, object]:
    sst = phx.equations.SSTTurbulencePlan("sst-2003-m")
    evaluate_sst = eqx.filter_jit(sst.evaluate)
    density = jnp.ones((size,))
    viscosity = jnp.full((size,), 1.8e-5)
    kinetic = jnp.full((size,), 0.1)
    omega = jnp.full((size,), 10.0)
    velocity_gradient = jnp.zeros((size, 2, 2)).at[..., 0, 1].set(20.0)
    scalar_gradient = jnp.full((size, 2), 0.01)
    wall_distance = jnp.full((size,), 0.01)
    sst_compile, sst_median = _timed(
        evaluate_sst,
        (
            density,
            viscosity,
            kinetic,
            omega,
            velocity_gradient,
            scalar_gradient,
            scalar_gradient,
            wall_distance,
        ),
        iterations,
    )

    reservoir = phx.solver.MaxwellianReservoirPlan(size, 3)
    sample = eqx.filter_jit(reservoir.sample)
    reservoir_compile, reservoir_median = _timed(
        sample,
        (
            jax.random.PRNGKey(0),
            jnp.asarray((100.0, 0.0, 0.0)),
            jnp.asarray(1000.0),
            jnp.asarray(4.65e-26),
        ),
        iterations,
    )
    values = (sst_compile, sst_median, reservoir_compile, reservoir_median)
    return {
        "benchmark": "aerothermodynamic-production-kernels",
        "size": size,
        "iterations": iterations,
        "sst": {
            "compile_seconds": sst_compile,
            "median_seconds": sst_median,
            "evaluations_per_second": size / sst_median,
        },
        "maxwellian_reservoir": {
            "compile_seconds": reservoir_compile,
            "median_seconds": reservoir_median,
            "particles_per_second": size / reservoir_median,
        },
        "finite": all(np.isfinite(value) for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.iterations <= 0:
        raise ValueError("iterations must be positive.")
    report = benchmark(
        size=128 if arguments.smoke else 16384,
        iterations=arguments.iterations,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    if not report["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
