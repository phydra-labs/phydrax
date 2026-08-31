#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp

import phydrax as phx


def benchmark(degree: int) -> dict[str, float | int | bool]:
    coordinates = jnp.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.2, 0.8), (0.5, 1.3), (-0.2, 0.8))
    )
    mesh = phx.discretization.CellMesh.from_polygons(coordinates, ((0, 1, 2, 3, 4),))
    field = phx.discretization.VirtualElementFieldSpec(
        "u", phx.discretization.conforming_h1_virtual_element(degree)
    )
    start = perf_counter()
    space = phx.discretization.VirtualElementPlan(mesh, field).prepare()
    preparation_seconds = perf_counter() - start
    projection = space.default_runtime.projections[0]
    form = phx.equations.VirtualElementForm(
        f"degree-{degree}-diffusion",
        "u",
        (phx.equations.DiffusionAction("u", 1.0),),
    )
    matrix_free = phx.equations.compile_virtual_element_problem(
        form,
        space,
        execution_policy=phx.equations.VirtualElementExecutionPolicy(
            realization="matrix_free"
        ),
    ).affine_operator()
    sparse = phx.equations.compile_virtual_element_problem(
        form,
        space,
        execution_policy=phx.equations.VirtualElementExecutionPolicy(
            realization="sparse"
        ),
    ).affine_operator()
    state = jnp.arange(space.dof_map.global_dof_count, dtype=float)
    matrix_free_value = matrix_free.mv(state)
    sparse_value = sparse.mv(state)
    parity = float(jnp.max(jnp.abs(matrix_free_value - sparse_value)))
    h1 = float(jnp.max(projection.evidence.h1_reproduction_error))
    l2 = float(jnp.max(projection.evidence.l2_reproduction_error))
    leakage = phx.discretization.stabilize_virtual_element_tensor(
        projection,
        jnp.zeros(
            (
                1,
                projection.dof_matrix.shape[1],
                projection.dof_matrix.shape[1],
            )
        ),
        phx.discretization.VirtualElementStabilizationPolicy(),
        projector="h1",
    ).evidence.polynomial_leakage
    leakage_value = float(jnp.max(leakage))
    passed = bool(
        h1 <= 1.0e-8
        and l2 <= 1.0e-8
        and parity <= 1.0e-10
        and leakage_value <= 1.0e-9
        and bool(jnp.all(projection.evidence.factorization_valid))
    )
    return {
        "degree": degree,
        "cells": 1,
        "maximum_arity": 5,
        "global_dofs": space.dof_map.global_dof_count,
        "preparation_seconds": preparation_seconds,
        "h1_reproduction_error": h1,
        "l2_reproduction_error": l2,
        "stabilization_polynomial_leakage": leakage_value,
        "matrix_free_sparse_error": parity,
        "maximum_g_condition": float(jnp.max(projection.evidence.maximum_g_condition)),
        "maximum_h_condition": float(jnp.max(projection.evidence.maximum_h_condition)),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/virtual_element.json"),
    )
    args = parser.parse_args()
    results = [benchmark(degree) for degree in (1, 2, 3)]
    if not all(bool(result["passed"]) for result in results):
        raise RuntimeError("Virtual-element qualification failed.")
    payload = {"polygon": results}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
