#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=262144)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.cells < 16 or arguments.devices <= 0 or arguments.repeats <= 0:
        raise ValueError("Scaling benchmark controls are invalid.")
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(arguments.cells, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="scaling-advection",
    )
    problem = phx.equations.ConservationProblemIR(
        "scaling-advection",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(
        problem, discretization, method
    )
    decomposition = phx.discretization.FiniteVolumeDecompositionPlan(
        (arguments.cells,),
        (arguments.devices,),
        ("x",),
        halo_width=1,
    ).prepare(jax.devices()[: arguments.devices])
    x = grid.structured_axes[0].interval_centers
    state = decomposition.shard_state(jnp.sin(2.0 * jnp.pi * x)[:, None])
    action = decomposition.compile_residual(compiled.dynamics, 0.0)
    started = time.perf_counter()
    first = action(state)
    first.block_until_ready()
    compile_and_first = time.perf_counter() - started
    samples = []
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        value = action(state)
        value.block_until_ready()
        samples.append(time.perf_counter() - started)
    steady = float(jnp.median(jnp.asarray(samples)))
    report = {
        "cells": arguments.cells,
        "devices": arguments.devices,
        "local_cells": decomposition.local_shape[0],
        "compile_and_first_seconds": compile_and_first,
        "steady_median_seconds": steady,
        "cells_per_second": arguments.cells / steady,
        "nanoseconds_per_cell": 1e9 * steady / arguments.cells,
        "halo_width": decomposition.plan.halo_width,
        "dtype": str(state.dtype),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "decomposition_id": decomposition.prepared_id,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
