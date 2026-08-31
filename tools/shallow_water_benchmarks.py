#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _case(count, *, muscl, dry):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    system = phx.equations.ShallowWaterSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    reconstruction = (
        phx.discretization.MUSCLReconstruction()
        if muscl
        else phx.discretization.PiecewiseConstantReconstruction()
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        reconstruction,
        phx.discretization.ShallowWaterHydrostaticHLLPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "shallow-water-benchmark",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    x = (jnp.arange(count) + 0.5) / count
    bed = (
        0.1 + 1.1 * jnp.exp(-100.0 * (x - 0.5) ** 2)
        if dry
        else 0.05 * jnp.sin(2.0 * jnp.pi * x)
    )
    compiled = phx.equations.compile_conservation_problem(
        problem, discretization, method, bathymetry=bed
    )
    surface = 1.0 + 0.02 * jnp.sin(4.0 * jnp.pi * x)
    depth = jnp.maximum(surface - bed, 0.0)
    state = jnp.stack((depth, jnp.zeros_like(depth)), axis=-1)
    return compiled, state


def _measure(compiled, state, repeats):
    residual = eqx.filter_jit(compiled.dynamics)
    started = time.perf_counter()
    first = residual(jnp.asarray(0.0), state)
    jax.block_until_ready(first)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        value = residual(jnp.asarray(0.0), state)
    jax.block_until_ready(value)
    execution_seconds = (time.perf_counter() - started) / repeats
    return {
        "compile_seconds": compile_seconds,
        "execution_seconds": execution_seconds,
        "cell_residuals_per_second": state.shape[0] / execution_seconds,
        "maximum_residual": float(jnp.max(jnp.abs(value))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.count < 32 or arguments.repeats <= 0:
        raise ValueError("Benchmark count and repeats must be positive and nontrivial.")
    reports = []
    for muscl in (False, True):
        for dry in (False, True):
            compiled, state = _case(arguments.count, muscl=muscl, dry=dry)
            measurement = _measure(compiled, state, arguments.repeats)
            reports.append(
                {
                    "reconstruction": "muscl" if muscl else "piecewise-constant",
                    "regime": "wet-dry" if dry else "fully-wet",
                    "count": arguments.count,
                    "repeats": arguments.repeats,
                    **measurement,
                }
            )
    print(json.dumps({"benchmarks": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
