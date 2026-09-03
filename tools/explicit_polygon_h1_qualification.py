#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp

import phydrax as phx


def _mesh(count: int):
    coordinates = jnp.asarray(
        tuple((i / count, j / count) for j in range(count + 1) for i in range(count + 1))
    )
    cells = tuple(
        (
            j * (count + 1) + i,
            j * (count + 1) + i + 1,
            (j + 1) * (count + 1) + i + 1,
            (j + 1) * (count + 1) + i,
        )
        for j in range(count)
        for i in range(count)
    )
    return phx.discretization.CellMesh.from_polygons(coordinates, cells)


def _timed(action):
    start = perf_counter()
    value = action()
    jax.block_until_ready(value)
    return value, perf_counter() - start


def qualify(count: int) -> dict[str, float | int | bool]:
    mesh = _mesh(count)
    explicit_field = phx.discretization.ExplicitPolygonH1FieldSpec("u")
    start = perf_counter()
    explicit = phx.discretization.ExplicitPolygonH1Plan(mesh, explicit_field).prepare()
    jax.block_until_ready(explicit.default_runtime.coordinates)
    explicit_prepare = perf_counter() - start
    explicit_form = phx.equations.FiniteElementForm(
        "explicit-polygon-diffusion",
        "u",
        (phx.equations.DiffusionAction("u", 1.0),),
    )
    explicit_compiled = phx.equations.compile_finite_element_problem(
        explicit_form, explicit
    )
    explicit_operator = explicit_compiled.affine_operator()
    state = jnp.linspace(-0.5, 0.5, explicit.dof_map.global_dof_count)
    jax.block_until_ready(explicit_operator.mv(state))
    _, explicit_apply = _timed(lambda: explicit_operator.mv(state))

    vem_field = phx.discretization.VirtualElementFieldSpec(
        "u", phx.discretization.conforming_h1_virtual_element(1)
    )
    start = perf_counter()
    vem = phx.discretization.VirtualElementPlan(mesh, vem_field).prepare()
    jax.block_until_ready(vem.default_runtime.coordinates)
    vem_prepare = perf_counter() - start
    vem_form = phx.equations.VirtualElementForm(
        "vem-diffusion", "u", (phx.equations.DiffusionAction("u", 1.0),)
    )
    vem_operator = phx.equations.compile_virtual_element_problem(
        vem_form,
        vem,
        execution_policy=phx.equations.VirtualElementExecutionPolicy(
            realization="matrix_free"
        ),
    ).affine_operator()
    jax.block_until_ready(vem_operator.mv(state))
    _, vem_apply = _timed(lambda: vem_operator.mv(state))

    evidence = explicit.default_runtime.bases[0].evidence
    passed = bool(jnp.all(evidence.passed))
    return {
        "cells": count * count,
        "global_dofs": explicit.dof_map.global_dof_count,
        "explicit_prepare_seconds": explicit_prepare,
        "explicit_apply_seconds": explicit_apply,
        "vem_prepare_seconds": vem_prepare,
        "vem_apply_seconds": vem_apply,
        "maximum_partition_error": float(jnp.max(evidence.partition_error)),
        "maximum_affine_value_error": float(jnp.max(evidence.affine_value_error)),
        "maximum_affine_gradient_error": float(jnp.max(evidence.affine_gradient_error)),
        "minimum_fan_measure": float(jnp.min(evidence.minimum_fan_measure)),
        "maximum_stiffness_condition": float(jnp.max(evidence.stiffness_condition)),
        "minimum_mass_eigenvalue": float(jnp.min(evidence.mass_minimum_eigenvalue)),
        "retained_bytes": dict(explicit.preparation.resource_counts)["retained_bytes"],
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/explicit_polygon_h1.json"),
    )
    args = parser.parse_args()
    results = [qualify(count) for count in (2, 4, 8)]
    if not all(bool(result["passed"]) for result in results):
        raise RuntimeError("Explicit polygon H1 qualification failed.")
    payload = {"structured_quadrilateral": results}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
