#!/usr/bin/env python3
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


def _mesh(width):
    coordinates = np.linspace(0.0, 1.0, width + 1)
    vertices = np.asarray([(x, y) for y in coordinates for x in coordinates])
    triangles = []
    for j in range(width):
        for i in range(width):
            lower_left = j * (width + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + width + 1
            upper_right = upper_left + 1
            triangles.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return vertices, np.asarray(triangles, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.width < 2 or arguments.repeats <= 0:
        raise ValueError("Triangle benchmark controls are invalid.")
    vertices, triangles = _mesh(arguments.width)
    started = time.perf_counter()
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices, triangles
    ).prepare()
    system = phx.equations.ScalarConservationSystem(
        2,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="triangle-benchmark-advection",
    )
    boundaries = phx.discretization.TriangleFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    problem = phx.equations.ConservationProblemIR(
        "triangle-benchmark", "state", system, boundaries
    )
    method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    preparation = time.perf_counter() - started
    state = jnp.sin(2.0 * jnp.pi * discretization.cell_centers[:, :1])
    action = eqx.filter_jit(lambda value: compiled(jnp.asarray(0.0), value))
    started = time.perf_counter()
    action(state).block_until_ready()
    compile_and_first = time.perf_counter() - started
    samples = []
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        result = action(state)
        result.block_until_ready()
        samples.append(time.perf_counter() - started)
    steady = float(np.median(samples))
    report = {
        "vertices": int(vertices.shape[0]),
        "cells": discretization.cell_count,
        "faces": int(discretization.face_measures.size),
        "preparation_seconds": preparation,
        "compile_and_first_seconds": compile_and_first,
        "steady_median_seconds": steady,
        "cells_per_second": discretization.cell_count / steady,
        "faces_per_second": discretization.face_measures.size / steady,
        "nanoseconds_per_cell": 1e9 * steady / discretization.cell_count,
        "mesh_id": discretization.prepared_id,
        "maximum_nonorthogonality_degrees": float(
            discretization.quality.maximum_nonorthogonality_degrees
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
