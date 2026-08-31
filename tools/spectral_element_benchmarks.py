#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import argparse
import json
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


def _quadrilateral(order: int) -> dict[str, float | int | bool]:
    coordinates = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    block = phx.discretization.CellBlock(
        "quadrilaterals",
        "quadrilateral",
        jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32),
    )
    mesh = phx.discretization.CellMesh(coordinates, (block,))
    element = phx.discretization.fem.ReferenceNodalFamily(
        "quadrilateral", order
    ).finite_element()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("u", element),
    ).prepare()
    form = phx.equations.FiniteElementForm(
        f"spectral-element-benchmark-p{order}",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.MassAction("u", 0.1),
        ),
    )
    dense = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free", local_kernel="dense"
        ),
    )
    factorized = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free", local_kernel="sum_factorized"
        ),
    )
    state = discretization.project(
        "u",
        lambda points, args: (
            jnp.sin(jnp.pi * points[..., 0]) * jnp.sin(jnp.pi * points[..., 1])
        ),
    )
    dense_action = eqx.filter_jit(dense.weak_residual)
    factorized_action = eqx.filter_jit(factorized.weak_residual)

    start = perf_counter()
    dense_value = dense_action(state)
    dense_value.block_until_ready()
    dense_first = perf_counter() - start
    start = perf_counter()
    factorized_value = factorized_action(state)
    factorized_value.block_until_ready()
    factorized_first = perf_counter() - start

    iterations = 20
    start = perf_counter()
    for _ in range(iterations):
        dense_action(state).block_until_ready()
    dense_steady = (perf_counter() - start) / iterations
    start = perf_counter()
    for _ in range(iterations):
        factorized_action(state).block_until_ready()
    factorized_steady = (perf_counter() - start) / iterations

    defect = jnp.max(jnp.abs(dense_value - factorized_value))
    finite = jnp.all(jnp.isfinite(factorized_value))
    passed = bool(finite) and float(defect) <= 5.0e-10
    return {
        "order": order,
        "global_dofs": discretization.dof_maps[0].global_dof_count,
        "dense_first_seconds": dense_first,
        "factorized_first_seconds": factorized_first,
        "dense_steady_seconds": dense_steady,
        "factorized_steady_seconds": factorized_steady,
        "dense_factorized_defect": float(defect),
        "finite": bool(finite),
        "passed": passed,
    }


def run() -> dict[str, object]:
    quadrilateral = [_quadrilateral(order) for order in (2, 3, 5)]
    return {
        "quadrilateral": quadrilateral,
        "passed": all(bool(result["passed"]) for result in quadrilateral),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run()
    if arguments.output is not None:
        arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not bool(result["passed"]):
        raise RuntimeError("Spectral-element qualification benchmark failed.")


if __name__ == "__main__":
    main()
