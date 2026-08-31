#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import platform
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_synchronized


def _timed(function, *args):
    return measure_synchronized(lambda: function(*args))


def main():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(4) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    runtime = phx.solver.CompatibleMaxwellPlan(bridge).prepare()
    state = runtime.initialize()
    step = jax.jit(
        lambda value: runtime.leapfrog_step(0.0, value, 0.05 * runtime.stable_dt)
    )
    state, compile_and_step = _timed(step, state)
    start = time.perf_counter()
    for _ in range(20):
        state = step(state)
    jax.block_until_ready(state.primary.electric_displacement)
    maxwell_steps = time.perf_counter() - start

    axis = jnp.linspace(-1.0, 1.0, 10)
    x, y = jnp.meshgrid(axis, axis, indexing="ij")
    points = jnp.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    point = phx.discretization.PointCloudPlan(
        points,
        jnp.ones((points.shape[0],)) / points.shape[0],
        degree=2,
        neighbor_count=12,
    ).prepare()
    values = points[:, 0] ** 2 + points[:, 1] ** 2
    _, point_time = _timed(
        jax.jit(lambda value: point.laplacian(value)),
        values,
    )

    teno = phx.discretization.HighResolutionReconstructionPlan("teno", order=8)
    signal = jnp.sin(jnp.linspace(0.0, 4.0 * jnp.pi, 256))
    _, teno_time = _timed(eqx.filter_jit(teno.reconstruct), signal)

    report = runtime.diagnostics(20 * 0.05 * runtime.stable_dt, state)
    print(
        json.dumps(
            {
                "kind": "maxwell-amoeba-benchmark-v1",
                "python": platform.python_version(),
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "maxwell": {
                    "degree_counts": bridge.cochain.cell_counts,
                    "compile_and_first_step_seconds": compile_and_step,
                    "twenty_cached_steps_seconds": maxwell_steps,
                    "electric_constraint_linf": float(report.electric_constraint_linf),
                    "magnetic_constraint_linf": float(report.magnetic_constraint_linf),
                },
                "point_cloud": {
                    "points": int(points.shape[0]),
                    "laplacian_compile_and_run_seconds": point_time,
                    "maximum_condition_number": point.report.maximum_condition_number,
                    "maximum_moment_residual": point.report.maximum_moment_residual,
                },
                "teno8": {
                    "cells": int(signal.shape[0]),
                    "compile_and_run_seconds": teno_time,
                    "qualification_passed": bool(teno.qualification.passed),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
