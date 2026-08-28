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
class NeighborScalingCase:
    dimension: int
    resolution: int
    particle_count: int
    dense_pair_capacity: int
    cell_pair_capacity: int
    cell_candidate_slots: int
    dense_relation_bytes: int
    cell_relation_bytes: int
    dense_neighborhood_seconds: float
    cell_neighborhood_seconds: float
    dense_force_seconds: float
    cell_force_seconds: float
    force_parity_error: float
    successful: bool


@dataclass(frozen=True)
class NeighborScalingReport:
    schema_version: int
    cases: tuple[NeighborScalingCase, ...]
    finite: bool

    @property
    def passed(self):
        return self.finite and all(
            case.successful and case.force_parity_error <= 1.0e-11 for case in self.cases
        )


def _relation_bytes(state):
    relation = state.pair_relation
    return sum(
        int(value.nbytes)
        for value in (
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
    )


def _case(dimension, resolution):
    axes = [
        (jnp.arange(resolution, dtype=float) + 0.5) / resolution for _ in range(dimension)
    ]
    grids = jnp.meshgrid(*axes, indexing="ij")
    position = jnp.stack(tuple(grid.reshape(-1) for grid in grids), axis=-1)
    count = int(position.shape[0])
    spacing = 1.0 / resolution
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.full((count,), spacing**dimension),
        ambient_dimension=dimension,
    ).prepare()
    box = phx.discretization.ParticleBox(jnp.zeros((dimension,)), jnp.ones((dimension,)))
    method = phx.discretization.BarotropicSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(dimension), 1.25 * spacing
    )
    problem = phx.equations.BarotropicFluidProblemIR(
        "scaling-fluid", phx.equations.TaitBarotropicMaterial(1.0, 1.0)
    )
    dense = phx.equations.compile_barotropic_sph_problem(
        problem,
        particles,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2, box=box
        ),
    )
    cell_capacity = {1: 8, 2: 16, 3: 64}[dimension]
    pair_multiplier = {1: 4, 2: 16, 3: 48}[dimension]
    cell = phx.equations.compile_barotropic_sph_problem(
        problem,
        particles,
        method,
        neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(
            method.kernel.support_factor * method.smoothing_length,
            cell_capacity,
            max(pair_multiplier * count, 1),
            box,
        ),
    )
    dense_build = eqx.filter_jit(dense.dynamics.neighborhood.build)
    cell_build = eqx.filter_jit(cell.dynamics.neighborhood.build)
    dense_build(position).pair_count.block_until_ready()
    cell_build(position).pair_count.block_until_ready()
    start = perf_counter()
    dense_state = dense_build(position)
    dense_state.pair_count.block_until_ready()
    dense_neighborhood_seconds = perf_counter() - start
    start = perf_counter()
    cell_state = cell_build(position)
    cell_state.pair_count.block_until_ready()
    cell_neighborhood_seconds = perf_counter() - start
    dense_force_fn = jax.jit(lambda value: dense.dynamics.force(0.0, value, None))
    cell_force_fn = jax.jit(lambda value: cell.dynamics.force(0.0, value, None))
    dense_force_fn(position).block_until_ready()
    cell_force_fn(position).block_until_ready()
    start = perf_counter()
    dense_force = dense_force_fn(position).block_until_ready()
    dense_force_seconds = perf_counter() - start
    start = perf_counter()
    cell_force = cell_force_fn(position).block_until_ready()
    cell_force_seconds = perf_counter() - start
    return NeighborScalingCase(
        dimension=int(dimension),
        resolution=int(resolution),
        particle_count=count,
        dense_pair_capacity=dense.dynamics.neighborhood.pair_capacity,
        cell_pair_capacity=cell.dynamics.neighborhood.pair_capacity,
        cell_candidate_slots=cell.dynamics.neighborhood.candidate_slot_count,
        dense_relation_bytes=_relation_bytes(dense_state),
        cell_relation_bytes=_relation_bytes(cell_state),
        dense_neighborhood_seconds=dense_neighborhood_seconds,
        cell_neighborhood_seconds=cell_neighborhood_seconds,
        dense_force_seconds=dense_force_seconds,
        cell_force_seconds=cell_force_seconds,
        force_parity_error=float(jnp.max(jnp.abs(cell_force - dense_force))),
        successful=bool(dense_state.successful & cell_state.successful),
    )


def run_scaling_sweep(*, smoke=False):
    configurations = (
        ((1, 16), (2, 4), (3, 3))
        if smoke
        else (
            (1, 32),
            (1, 64),
            (1, 128),
            (2, 8),
            (2, 12),
            (2, 16),
            (3, 4),
            (3, 5),
            (3, 6),
        )
    )
    cases = tuple(
        _case(dimension, resolution) for dimension, resolution in configurations
    )
    values = jnp.asarray(
        tuple(
            value
            for case in cases
            for value in (
                case.dense_neighborhood_seconds,
                case.cell_neighborhood_seconds,
                case.dense_force_seconds,
                case.cell_force_seconds,
                case.force_parity_error,
            )
        )
    )
    return NeighborScalingReport(
        schema_version=1,
        cases=cases,
        finite=bool(jnp.all(jnp.isfinite(values))),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Measure dense and cell-list particle scaling."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_scaling_sweep(smoke=arguments.smoke)
    payload = json.dumps({**asdict(report), "passed": report.passed}, indent=2)
    if arguments.output is not None:
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, arguments.output)
    print(payload)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
