"""Prepared sparse unstructured Maxwell scan and bounded dense-reference campaign.

Run ``python -m benchmarks.unstructured_maxwell --smoke`` after integration.
Cold FE assembly, spectral/runtime admission, lowering, compilation, first run,
and prepared reuse are distinct. Numerical runtime leaves remain executable
arguments; no host factorization occurs in the time scan. Dense materialization
is admitted only for the small comparison envelope, not a second FE assembly.
"""

from __future__ import annotations

import argparse
import itertools
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
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization.fem import FiniteElementDeRhamComplex
from phydrax.meshing import summarize_cell_quality
from phydrax.solver.maxwell import (
    CompatibleMaxwellState,
    DiagonalMaxwellConstitutivePlan,
    PreparedUnstructuredMaxwell,
    UnstructuredMaxwellPlan,
)


OUTPUT = Path(__file__).with_suffix(".json")


def _mesh(subdivisions: int) -> CellMesh:
    axis = np.linspace(0.0, 1.0, subdivisions + 1, dtype=np.float64)
    lattice = np.meshgrid(axis, axis, axis, indexing="ij")
    coordinates = np.stack(lattice, axis=-1).reshape((-1, 3))
    width = subdivisions + 1
    cells: list[tuple[int, ...]] = []
    for i, j, k in itertools.product(range(subdivisions), repeat=3):
        for order in itertools.permutations(range(3)):
            index = np.asarray((i, j, k), dtype=np.int32)
            row = [int((index[0] * width + index[1]) * width + index[2])]
            for direction in order:
                index[direction] += 1
                row.append(int((index[0] * width + index[1]) * width + index[2]))
            points = coordinates[row]
            if np.linalg.det((points[1:] - points[0]).T) < 0.0:
                row[0], row[1] = row[1], row[0]
            cells.append(tuple(row))
    return CellMesh(
        coordinates,
        (CellBlock("tetrahedra", "tetrahedron", np.asarray(cells, dtype=np.int32)),),
    )


def _initial(runtime: PreparedUnstructuredMaxwell) -> CompatibleMaxwellState:
    complex_ = runtime.plan.cochain
    electric = jnp.sin(jnp.arange(complex_.cell_counts[1], dtype=jnp.float64) + 0.2)
    displacement = runtime.constitutive.electric_displacement(electric, None)
    flux = complex_.exterior_derivative(1, electric)
    charge = -complex_.codifferential(1, displacement)
    return eqx.tree_at(
        lambda state: (
            state.primary.electric_displacement,
            state.primary.magnetic_flux,
            state.primary.charge,
        ),
        runtime.initialize(),
        (displacement, flux, charge),
    )


def _dense_reference(
    runtime: PreparedUnstructuredMaxwell,
    initial: CompatibleMaxwellState,
    dt: Array,
    current: Array,
    steps: int,
) -> tuple[dict[str, object], tuple[Array, Array, Array] | None]:
    complex_ = runtime.plan.cochain
    if max(complex_.cell_counts) > 512:
        return {"maximum_admitted_coordinates": 512, "evaluated": False}, None
    masses = tuple(
        jax.vmap(lambda values: complex_.hodge_star(degree, values))(
            jnp.eye(count, dtype=jnp.float64)
        ).T
        for degree, count in enumerate(complex_.cell_counts[:3])
    )
    gradient = jax.vmap(lambda values: complex_.exterior_derivative(0, values))(
        jnp.eye(complex_.cell_counts[0], dtype=jnp.float64)
    ).T
    curl = jax.vmap(lambda values: complex_.exterior_derivative(1, values))(
        jnp.eye(complex_.cell_counts[1], dtype=jnp.float64)
    ).T

    def run(
        mass_zero: Array,
        mass_one: Array,
        mass_two: Array,
        d_zero: Array,
        d_one: Array,
        displacement: Array,
        flux: Array,
        charge: Array,
        step: Array,
        source: Array,
    ) -> tuple[Array, Array, Array]:
        def advance(
            index: Array, state: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            del index
            electric_displacement, magnetic_flux, density = state
            half_flux = magnetic_flux - 0.5 * step * (d_one @ electric_displacement)
            next_displacement = electric_displacement + step * (
                jnp.linalg.solve(mass_one, d_one.T @ (mass_two @ half_flux)) - source
            )
            next_flux = half_flux - 0.5 * step * (d_one @ next_displacement)
            next_charge = density + step * jnp.linalg.solve(
                mass_zero, d_zero.T @ (mass_one @ source)
            )
            return next_displacement, next_flux, next_charge

        return jax.lax.fori_loop(0, steps, advance, (displacement, flux, charge))

    arguments = (
        *masses,
        gradient,
        curl,
        initial.primary.electric_displacement,
        initial.primary.magnetic_flux,
        initial.primary.charge,
        dt,
        current,
    )
    executable, compilation = measure_lower_and_compile(
        lambda: jax.jit(run).lower(*arguments), lambda lowered: lowered.compile()
    )
    final, timing = measure_repeated(lambda: executable(*arguments), warmup=1, repeats=3)
    evidence = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Selected backend does not expose compiler estimates.",
    )
    return {
        "maximum_admitted_coordinates": 512,
        "evaluated": True,
        "compilation": asdict(compilation),
        "prepared_execution": timing.to_seconds_dict(),
        "compiler": asdict(evidence),
        "retained_bytes": logical_array_bytes((masses, gradient, curl)),
    }, final


def _case(subdivisions: int, repeats: int, steps: int) -> dict[str, object]:
    mesh = _mesh(subdivisions)
    complex_, assembly_seconds = measure_synchronized(
        lambda: FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
    )
    # Exercise actual automatic admission in the smallest envelope. Larger
    # capacity rows deliberately supply a conservative external CFL bound.
    supplied_bound = None if subdivisions == 1 else 1.0e6 * subdivisions**2
    runtime, preparation_seconds = measure_synchronized(
        lambda: UnstructuredMaxwellPlan(
            complex_,
            DiagonalMaxwellConstitutivePlan(),
            spectral_upper_bound=supplied_bound,
            courant_factor=0.9,
        ).prepare()
    )
    if runtime.mesh_quality is None:
        raise ValueError(
            "FE Maxwell admission must retain canonical meshing quality evidence."
        )
    quality = summarize_cell_quality(runtime.mesh_quality)
    initial = _initial(runtime)
    current = 0.01 * jnp.cos(jnp.arange(complex_.cell_counts[1], dtype=jnp.float64))
    dt = 0.05 * runtime.stable_dt
    numerical, static = eqx.partition(runtime, eqx.is_array)

    def run(
        dynamic: PreparedUnstructuredMaxwell,
        state: CompatibleMaxwellState,
        step: Array,
        source: Array,
    ) -> CompatibleMaxwellState:
        prepared = eqx.combine(dynamic, static)

        def advance(
            index: Array, carry: CompatibleMaxwellState
        ) -> CompatibleMaxwellState:
            return prepared.step(index * step, carry, step, electric_current=source)

        return jax.lax.fori_loop(0, steps, advance, state)

    executable, compilation = measure_lower_and_compile(
        lambda: jax.jit(run).lower(numerical, initial, dt, current),
        lambda lowered: lowered.compile(),
    )
    first, first_seconds = measure_synchronized(
        lambda: executable(numerical, initial, dt, current)
    )
    source_samples = tuple((1.0 + 0.05 * index) * current for index in range(repeats))
    stream = iter(source_samples)
    final, timing = measure_repeated(
        lambda: executable(numerical, initial, dt, next(stream)),
        warmup=0,
        repeats=repeats,
    )
    gauss, closedness = runtime.constraints(final)
    errors = {
        "gauss_absolute_error": float(jnp.max(jnp.abs(gauss), initial=0.0)),
        "closedness_absolute_error": float(jnp.max(jnp.abs(closedness), initial=0.0)),
    }
    (dense, dense_final), dense_preparation_seconds = measure_synchronized(
        lambda: _dense_reference(runtime, initial, dt, current, steps)
    )
    if dense_final is not None:
        sparse_final = (
            first.primary.electric_displacement,
            first.primary.magnetic_flux,
            first.primary.charge,
        )
        errors["dense_step_absolute_error"] = max(
            float(jnp.max(jnp.abs(sparse - reference), initial=0.0))
            for sparse, reference in zip(sparse_final, dense_final, strict=True)
        )
    evidence = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Selected backend does not expose compiler estimates.",
    )
    return {
        "mesh_subdivisions": subdivisions,
        "cell_capacity": 6 * subdivisions**3,
        "coordinate_counts": complex_.cell_counts,
        "step_capacity": steps,
        "assembly_seconds": assembly_seconds,
        "runtime_preparation_seconds": preparation_seconds,
        "spectral_admission": "trace-certified"
        if supplied_bound is None
        else "caller-bound",
        "mesh_quality": {
            "report_id": quality.report_id,
            "minimum_measure": quality.minimum_measure,
            "maximum_aspect_ratio": quality.maximum_aspect_ratio,
            "minimum_scaled_jacobian": quality.minimum_scaled_jacobian,
            "sampled_invalid_count": quality.sampled_invalid_count,
        },
        "compilation": asdict(compilation),
        "first_execution_seconds": first_seconds,
        "prepared_execution": timing.to_seconds_dict(),
        "compiler": asdict(evidence),
        "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
        "retained_runtime_bytes": logical_array_bytes(runtime),
        "dense_reference": dense,
        "dense_preparation_seconds": dense_preparation_seconds,
        "errors": errors,
        "successful": quality.sampled_invalid_count == 0
        and all(np.isfinite(error) and error < 1e-8 for error in errors.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    capacities = (1, 2) if arguments.smoke else (1, 2, 4, 8)
    records = [
        _case(size, 2 if arguments.smoke else 5, 4 if arguments.smoke else 32)
        for size in capacities
    ]
    successful = all(record["successful"] for record in records)
    driver = Path(__file__).resolve()
    identity = capture_benchmark_identity(driver.parents[1], driver, records[0])
    write_json_atomic(
        arguments.output,
        {
            "benchmark": "unstructured-maxwell",
            "identity": identity.to_dict(),
            "environment": capture_environment().to_dict(),
            "cases": records,
            "successful": successful,
        },
    )
    if not successful:
        raise SystemExit(
            "Unstructured Maxwell conservation or dense-reference gate failed."
        )


if __name__ == "__main__":
    main()
