"""Sparse FE assembly and prepared Hodge solve capacity campaign.

Run with ``python -m benchmarks.fem_de_rham --smoke``. Numerical complex leaves
remain executable arguments, so prepared reuse does not hide frozen matrices.
The dense comparison is bounded to small admitted systems and is not a second FE
implementation. Every record is gated by solve residual and exact d-squared.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import assert_never, get_args, Literal, TYPE_CHECKING, TypeAlias

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
from phydrax.typing import parse


if TYPE_CHECKING:
    from phydrax.discretization.fem._exact_form_moments import MomentIntegralMatrix


OUTPUT = Path(__file__).with_suffix(".json")
BenchmarkMode: TypeAlias = Literal["hodge", "partition-moments"]


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


def _same_moment_bits(first: MomentIntegralMatrix, second: MomentIntegralMatrix) -> bool:
    return all(
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
        for left, right in (
            (first.values, second.values),
            (first.errors, second.errors),
            (first.exact, second.exact),
            (first.enclosed, second.enclosed),
        )
    )


def _partition_moment_case(
    capacity: int, *, repeated_geometry: bool
) -> dict[str, object]:
    """Time host-only local moments for declared exact prism partitions."""
    from phydrax.discretization._nested_reference_types import (
        ReferenceCellMapWitnesses,
        ReferenceCellPartition,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.discretization.fem._exact_form_moments import (
        _ReferenceMomentCache,
        exact_partitioned_reference_form_moments,
    )
    from phydrax.discretization.fem._form_elements import form_element
    from phydrax.discretization.fem._partition_form_moments import (
        prepare_partitioned_reference_entities,
    )

    started = perf_counter()
    element = form_element(
        "prism", 1, 1, family="prism-trimmed", twist="untwisted", proxy="components"
    )
    if element.form_basis is None:
        raise RuntimeError("Partition benchmark requires a canonical form basis")
    vertices = tuple(
        tuple(Fraction(int(value)) for value in vertex)
        for vertex in reference_cell_topology("prism").vertices
    )
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(element.local_dof_count))
        for row in range(element.local_dof_count)
    )
    records = []
    for index in range(capacity):
        cut = Fraction(1, 2) if repeated_geometry else Fraction(index + 1, capacity + 1)
        children = (
            tuple((x, y, cut * z) for x, y, z in vertices),
            tuple((x, y, cut + (1 - cut) * z) for x, y, z in vertices),
        )
        coarse_id = 3 * index + 2
        witnesses = ReferenceCellMapWitnesses(
            np.asarray((3 * index, 3 * index + 1), dtype=np.int64),
            np.asarray((coarse_id, coarse_id), dtype=np.int64),
            "prism",
            "prism",
            np.asarray(children, dtype=np.float64),
            children,
        )
        records.extend(
            prepare_partitioned_reference_entities(
                ReferenceCellPartition((witnesses,)), coarse_id, element.form_basis
            )
        )
    preparation_seconds = perf_counter() - started
    started = perf_counter()
    reference = tuple(
        exact_partitioned_reference_form_moments(
            element, element, record, identity, identity, None
        )
        for record in records
    )
    reference_seconds = perf_counter() - started
    cache = _ReferenceMomentCache()

    def run() -> tuple[MomentIntegralMatrix, ...]:
        return tuple(
            cache.partitioned(element, element, record, identity, identity, None)
            for record in records
        )

    started = perf_counter()
    cold = run()
    cold_seconds = perf_counter() - started
    cold_info = cache.table.cache_info()._asdict()
    started = perf_counter()
    steady = run()
    steady_seconds = perf_counter() - started
    steady_info = cache.table.cache_info()._asdict()

    exact_equality = all(
        _same_moment_bits(expected, first) and _same_moment_bits(expected, again)
        for expected, first, again in zip(reference, cold, steady, strict=True)
    )
    bounded = (
        steady_info["currsize"] <= steady_info["max_entries"]
        and steady_info["current_bytes"] <= steady_info["max_bytes"]
    )
    return {
        "geometry_scenario": "repeated" if repeated_geometry else "distinct",
        "declared_partitions": capacity,
        "declared_local_records": len(records),
        "immutable_preparation_seconds": preparation_seconds,
        "uncached_reference_seconds": reference_seconds,
        "uncached_reference_scope": (
            "measured separately before cached phases; other host preparation caches may be warm"
        ),
        "cold_seconds": cold_seconds,
        "steady_seconds": steady_seconds,
        "cold_cache": cold_info,
        "steady_cache": steady_info,
        "logical_retained_bytes_scope": "cache key and output payload, not process RSS",
        "process_peak_rss": {
            "available": False,
            "reason": "process peak RSS is not measured by this portable host benchmark",
        },
        "compilation": {
            "applicable": False,
            "reason": "host-only exact moment preparation",
        },
        "evidence": [
            {
                "fine_cell_id": record.fine_cell_id,
                "coarse_cell_id": record.coarse_cell_id,
                "values": result.values.tolist(),
                "errors": result.errors.tolist(),
                "exact": result.exact.tolist(),
                "enclosed": result.enclosed.tolist(),
            }
            for record, result in zip(records, cold, strict=True)
        ],
        "all_evidence_bitwise_equal_to_uncached_and_steady": exact_equality,
        "cache_within_bounds": bounded,
        "successful": exact_equality and bounded,
    }


def _partition_moment_campaign(output: Path, capacities: tuple[int, ...]) -> None:
    if output.exists():
        raise SystemExit("Partition-moment mode requires a fresh --output path.")
    if not capacities or any(capacity < 1 for capacity in capacities):
        raise SystemExit("Partition capacities must be positive.")
    records = [
        _partition_moment_case(capacity, repeated_geometry=repeated_geometry)
        for capacity in capacities
        for repeated_geometry in (True, False)
    ]
    successful = all(record["successful"] for record in records)
    driver = Path(__file__).resolve()
    identity = capture_benchmark_identity(driver.parents[1], driver, records[0])
    write_json_atomic(
        output,
        {
            "benchmark": "fem-de-rham-partition-moments",
            "scope": "host-local reference partition moments; not a physical transfer",
            "identity": identity.to_dict(),
            "environment": capture_environment().to_dict(),
            "cases": records,
            "successful": successful,
        },
    )
    if not successful:
        raise SystemExit("Partition moment evidence or cache bounds differ.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mode", choices=get_args(BenchmarkMode), default="hodge")
    parser.add_argument("--capacities", type=int, nargs="+")
    arguments = parser.parse_args()
    mode = parse(arguments.mode, BenchmarkMode, "--mode")
    match mode:
        case "partition-moments":
            if arguments.output == OUTPUT:
                parser.error("partition-moments requires an explicit fresh --output")
            capacities = tuple(
                arguments.capacities or ((1, 2) if arguments.smoke else (1, 4, 16))
            )
            _partition_moment_campaign(arguments.output, capacities)
            return
        case "hodge":
            if arguments.capacities is not None:
                parser.error("--capacities is only available in partition-moments mode")
            capacities = (1, 2) if arguments.smoke else (2, 4, 8, 16)
            orders = (1,) if arguments.smoke else (1, 2)
        case _:
            assert_never(mode)
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
