#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class ParticleBackendBenchmark:
    realization: str
    pair_capacity: int
    candidate_pair_count: int
    physical_pair_count: int
    maximum_cell_occupancy: int
    candidate_slot_count: int
    relation_bytes: int
    neighborhood_compile_seconds: float
    neighborhood_seconds: float
    force_compile_seconds: float
    force_seconds: float
    energy_gradient_error: float
    momentum_defect: float
    trajectory_energy_defect: float
    successful: bool


@dataclass(frozen=True)
class ParticleParityBenchmark:
    density_error: float
    force_error: float
    energy_error: float
    jvp_error: float
    trajectory_error: float


@dataclass(frozen=True)
class ParticleSPHBenchmarkRecord:
    schema_version: int
    particle_count: int
    state_bytes: int
    density_spread: float
    dense: ParticleBackendBenchmark
    cell: ParticleBackendBenchmark
    parity: ParticleParityBenchmark
    overflow_detected: bool
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.dense.successful
            and self.cell.successful
            and self.overflow_detected
            and self.density_spread <= 0.1
            and self.dense.energy_gradient_error <= 1.0e-10
            and self.cell.energy_gradient_error <= 1.0e-10
            and self.dense.momentum_defect <= 1.0e-11
            and self.cell.momentum_defect <= 1.0e-11
            and self.dense.trajectory_energy_defect <= 1.0e-6
            and self.cell.trajectory_energy_defect <= 1.0e-6
            and self.parity.density_error <= 1.0e-12
            and self.parity.force_error <= 1.0e-11
            and self.parity.energy_error <= 1.0e-12
            and self.parity.jvp_error <= 2.0e-8
            and self.parity.trajectory_error <= 1.0e-10
        )


def _problem_components(count: int):
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.full((count,), spacing),
        ambient_dimension=1,
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    method = phx.discretization.BarotropicSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(1),
        1.25 * spacing,
    )
    problem = phx.equations.BarotropicFluidProblemIR(
        "particle-sph-benchmark",
        phx.equations.TaitBarotropicMaterial(1.0, 1.0),
    )
    lattice = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    position = lattice + 0.001 * jnp.sin(2.0 * jnp.pi * lattice)
    return particles, box, method, problem, position


def _compile_backends(count: int):
    particles, box, method, problem, position = _problem_components(count)
    dense = phx.equations.compile_barotropic_sph_problem(
        problem,
        particles,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2,
            box=box,
        ),
    )
    cell = phx.equations.compile_barotropic_sph_problem(
        problem,
        particles,
        method,
        neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(
            method.kernel.support_factor * method.smoothing_length,
            8,
            4 * count,
            box,
        ),
    )
    return dense, cell, position


def _relation_bytes(state: phx.discretization.ParticleNeighborhoodState) -> int:
    relation = state.pair_relation
    arrays = (
        relation.relation.source_indices,
        relation.relation.target_indices,
        relation.relation.valid,
        relation.left_particle_ids,
        relation.right_particle_ids,
        state.storage_to_logical,
        state.logical_to_storage,
        state.cell_ids,
        state.cell_counts,
        state.cell_offsets,
    )
    return sum(int(value.nbytes) for value in arrays)


def _solve(compiled, position, final_time):
    velocity = jnp.zeros_like(position)
    return phx.solver.solve_diffrax(
        compiled.as_differential_problem(
            position,
            velocity,
            t0=0.0,
            t1=final_time,
        ),
        save_times=jnp.asarray([final_time]),
        solver=phx.solver.StormerVerlet(1),
        dt0=2.0e-4,
        max_steps=64,
    )


def _backend_benchmark(compiled, position) -> tuple[ParticleBackendBenchmark, object]:
    neighborhood = eqx.filter_jit(compiled.dynamics.neighborhood.build)
    start = perf_counter()
    state = neighborhood(position)
    state.pair_count.block_until_ready()
    neighborhood_compile_seconds = perf_counter() - start
    start = perf_counter()
    repeated_state = neighborhood(position)
    repeated_state.pair_count.block_until_ready()
    neighborhood_seconds = perf_counter() - start
    force = jax.jit(lambda value: compiled.dynamics.force(0.0, value, None))
    start = perf_counter()
    compiled_force = force(position).block_until_ready()
    force_compile_seconds = perf_counter() - start
    start = perf_counter()
    repeated_force = force(position).block_until_ready()
    force_seconds = perf_counter() - start
    reference_gradient = jax.grad(
        lambda value: compiled.dynamics.potential_energy(0.0, value, None)
    )(position)
    energy_gradient_error = jnp.max(jnp.abs(-compiled_force - reference_gradient))
    momentum_defect = jnp.max(jnp.abs(jnp.sum(repeated_force, axis=0)))
    zero_momentum = jnp.zeros_like(position)
    initial = compiled.dynamics.diagnostics(0.0, position, zero_momentum, None)
    final_time = 0.005
    solution = _solve(compiled, position, final_time)
    final_position, final_momentum, _ = compiled.dynamics.unpack_phase_state(
        solution.states[-1]
    )
    final = compiled.dynamics.diagnostics(
        final_time,
        final_position,
        final_momentum,
        None,
    )
    energy_scale = jnp.maximum(jnp.abs(initial.total_energy), jnp.finfo(float).tiny)
    trajectory_energy_defect = (
        jnp.abs(final.total_energy - initial.total_energy) / energy_scale
    )
    candidate_slot_count = 0
    if isinstance(
        compiled.dynamics.neighborhood,
        phx.discretization.PreparedCellListParticleNeighborhood,
    ):
        candidate_slot_count = compiled.dynamics.neighborhood.candidate_slot_count
    record = ParticleBackendBenchmark(
        realization=compiled.dynamics.neighborhood.backend,
        pair_capacity=compiled.dynamics.neighborhood.pair_capacity,
        candidate_pair_count=int(repeated_state.candidate_pair_count),
        physical_pair_count=int(final.active_pairs),
        maximum_cell_occupancy=int(repeated_state.maximum_cell_occupancy),
        candidate_slot_count=candidate_slot_count,
        relation_bytes=_relation_bytes(repeated_state),
        neighborhood_compile_seconds=neighborhood_compile_seconds,
        neighborhood_seconds=neighborhood_seconds,
        force_compile_seconds=force_compile_seconds,
        force_seconds=force_seconds,
        energy_gradient_error=float(energy_gradient_error),
        momentum_defect=float(momentum_defect),
        trajectory_energy_defect=float(trajectory_energy_defect),
        successful=bool(repeated_state.successful & solution.backend_successful),
    )
    return record, solution


def run_particle_sph_benchmark(count: int = 32, /) -> ParticleSPHBenchmarkRecord:
    if int(count) < 4:
        raise ValueError("Particle SPH benchmark requires at least four particles.")
    dense, cell, position = _compile_backends(int(count))
    dense_record, dense_solution = _backend_benchmark(dense, position)
    cell_record, cell_solution = _backend_benchmark(cell, position)
    dense_density = dense.dynamics.density(position)
    cell_density = cell.dynamics.density(position)
    dense_force = dense.dynamics.force(0.0, position, None)
    cell_force = cell.dynamics.force(0.0, position, None)
    dense_energy = dense.dynamics.potential_energy(0.0, position, None)
    cell_energy = cell.dynamics.potential_energy(0.0, position, None)
    direction = jnp.cos(3.0 * jnp.pi * position)
    direction = direction / jnp.sqrt(jnp.sum(direction * direction))
    _, dense_jvp = jax.jvp(
        lambda value: dense.dynamics.potential_gradient(0.0, value, None),
        (position,),
        (direction,),
    )
    _, cell_jvp = jax.jvp(
        lambda value: cell.dynamics.potential_gradient(0.0, value, None),
        (position,),
        (direction,),
    )
    parity = ParticleParityBenchmark(
        density_error=float(jnp.max(jnp.abs(cell_density - dense_density))),
        force_error=float(jnp.max(jnp.abs(cell_force - dense_force))),
        energy_error=float(jnp.abs(cell_energy - dense_energy)),
        jvp_error=float(jnp.max(jnp.abs(cell_jvp - dense_jvp))),
        trajectory_error=float(
            jnp.max(jnp.abs(cell_solution.states - dense_solution.states))
        ),
    )
    overflow_plan = phx.discretization.CellListParticleNeighborhoodPlan(
        cell.dynamics.method.kernel.support_factor
        * cell.dynamics.method.smoothing_length,
        1,
        1,
        cell.dynamics.neighborhood.box,
    ).prepare(cell.dynamics.particles)
    clustered = jnp.linspace(0.1, 0.11, int(count))[:, None]
    overflow_state = overflow_plan.build(clustered)
    values = jnp.asarray(
        (
            parity.density_error,
            parity.force_error,
            parity.energy_error,
            parity.jvp_error,
            parity.trajectory_error,
            dense_record.energy_gradient_error,
            cell_record.energy_gradient_error,
        )
    )
    return ParticleSPHBenchmarkRecord(
        schema_version=2,
        particle_count=int(count),
        state_bytes=int(2 * position.nbytes),
        density_spread=float(jnp.max(dense_density) - jnp.min(dense_density)),
        dense=dense_record,
        cell=cell_record,
        parity=parity,
        overflow_detected=bool(~overflow_state.successful),
        finite=bool(jnp.all(jnp.isfinite(values))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify dense and cell-list conservative barotropic SPH."
    )
    parser.add_argument("--particle-count", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    count = 12 if arguments.smoke else arguments.particle_count
    record = run_particle_sph_benchmark(count)
    payload = json.dumps({**asdict(record), "passed": record.passed}, indent=2)
    if arguments.output is not None:
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, arguments.output)
    print(payload)
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
