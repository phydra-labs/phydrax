"""Sparse FE assembly and prepared Hodge solve capacity campaign.

Run with ``python -m benchmarks.fem_de_rham --smoke``. Numerical complex leaves
remain executable arguments, so prepared reuse does not hide frozen matrices.
The dense comparison is bounded to small admitted systems and is not a second FE
implementation. Every record is gated by solve residual and exact d-squared.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.discretization import CellMesh
from phydrax.discretization.fem._de_rham import FiniteElementDeRhamComplex


OUTPUT = Path(__file__).with_suffix(".json")


def _mesh(subdivisions: int) -> CellMesh:
    axis = np.linspace(0.0, 1.0, subdivisions + 1, dtype=np.float64)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    coordinates = np.column_stack((first.reshape(-1), second.reshape(-1)))
    cells: list[tuple[int, int, int]] = []
    width = subdivisions + 1
    for row in range(subdivisions):
        for column in range(subdivisions):
            lower = row * width + column
            cells.extend(
                (
                    (lower, lower + width, lower + width + 1),
                    (lower, lower + width + 1, lower + 1),
                )
            )
    return CellMesh.from_triangles(coordinates, np.asarray(cells, dtype=np.int32))


def _case(subdivisions: int, order: int, repeats: int) -> dict[str, object]:
    mesh = _mesh(subdivisions)
    complex_, assembly_seconds = measure_synchronized(
        lambda: FiniteElementDeRhamComplex(mesh, family="trimmed", order=order)
    )
    size = complex_.hilbert_complex().space(1).size
    initial = jnp.linspace(-0.7, 1.2, size, dtype=jnp.float64)
    numerical, static = eqx.partition(complex_, eqx.is_array)

    def solve(dynamic: FiniteElementDeRhamComplex, rhs: Array) -> Array:
        return eqx.combine(dynamic, static).inverse_hodge_star(1, rhs)

    executable, compilation = measure_lower_and_compile(
        lambda: jax.jit(solve).lower(numerical, initial),
        lambda lowered: lowered.compile(),
    )
    first, first_seconds = measure_synchronized(lambda: executable(numerical, initial))
    samples = tuple(
        (1.0 + 0.05 * index) * initial + 0.1 * index for index in range(repeats)
    )
    stream = iter(samples)
    final, steady = measure_repeated(
        lambda: executable(numerical, next(stream)), warmup=0, repeats=repeats
    )
    initial_error = jnp.linalg.norm(
        complex_.hodge_star(1, first) - initial
    ) / jnp.linalg.norm(initial)
    final_error = jnp.linalg.norm(
        complex_.hodge_star(1, final) - samples[-1]
    ) / jnp.linalg.norm(samples[-1])
    scalar_count = complex_.hilbert_complex().space(0).size
    scalar = jnp.linspace(-0.4, 0.8, scalar_count, dtype=jnp.float64)
    nilpotency = jnp.max(
        jnp.abs(complex_.exterior_derivative(1, complex_.exterior_derivative(0, scalar)))
    )
    dense_reference: dict[str, object] = {
        "maximum_admitted_dofs": 512,
        "evaluated": False,
    }
    if size <= 512:
        dense, dense_assembly_seconds = measure_synchronized(
            lambda: (
                jax.vmap(lambda values: complex_.hodge_star(1, values))(
                    jnp.eye(size, dtype=jnp.float64)
                ).T
            )
        )
        reference, dense_solve_seconds = measure_synchronized(
            lambda: jnp.linalg.solve(dense, initial)
        )
        dense_reference = {
            "maximum_admitted_dofs": 512,
            "evaluated": True,
            "assembly_seconds": dense_assembly_seconds,
            "solve_seconds": dense_solve_seconds,
            "retained_bytes": logical_array_bytes(dense),
            "solution_relative_error": float(
                jnp.linalg.norm(first - reference) / jnp.linalg.norm(reference)
            ),
        }
    evidence = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Selected backend does not expose compiler estimates.",
    )
    errors = {
        "initial_solve_relative_residual": float(initial_error),
        "updated_solve_relative_residual": float(final_error),
        "nilpotency_absolute_error": float(nilpotency),
    }
    successful = all(np.isfinite(value) and value <= 1e-8 for value in errors.values())
    return {
        "mesh_subdivisions": subdivisions,
        "cell_capacity": 2 * subdivisions**2,
        "polynomial_order": order,
        "degree_one_dofs": size,
        "assembly_seconds": assembly_seconds,
        "compilation": asdict(compilation),
        "first_execution_seconds": first_seconds,
        "prepared_execution": steady.to_seconds_dict(),
        "compiler": asdict(evidence),
        "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
        "retained_complex_bytes": logical_array_bytes(complex_),
        "dense_reference": dense_reference,
        "errors": errors,
        "successful": successful,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    capacities = (1, 2) if arguments.smoke else (2, 4, 8, 16)
    orders = (1,) if arguments.smoke else (1, 2)
    records = [
        _case(size, order, 2 if arguments.smoke else 5)
        for order in orders
        for size in capacities
    ]
    successful = all(record["successful"] for record in records)
    driver = Path(__file__).resolve()
    identity = capture_benchmark_identity(driver.parents[1], driver, records[0])
    write_json_atomic(
        arguments.output,
        {
            "benchmark": "fem-de-rham",
            "identity": identity.to_dict(),
            "environment": capture_environment().to_dict(),
            "cases": records,
            "successful": successful,
        },
    )
    if not successful:
        raise SystemExit(
            "FE Hodge solve or exact complex benchmark failed its numerical gate."
        )


if __name__ == "__main__":
    main()
