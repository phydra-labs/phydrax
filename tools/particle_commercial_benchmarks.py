#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class CommercialParticleBenchmark:
    schema_version: int
    maturity: str
    native_pair_count: int
    native_data_plane_successful: bool
    small_solve_residual: float
    adaptive_h_residual: float
    iisph_action_error: float
    iisph_production_qualified: bool
    dfsph_production_qualified: bool
    distributed_migration_count: int
    mixed_precision_error: float
    finite: bool

    @property
    def passed(self):
        return (
            self.maturity == "experimental"
            and self.native_data_plane_successful
            and self.small_solve_residual <= 1e-10
            and self.adaptive_h_residual <= 1e-8
            and self.iisph_action_error <= 1e-10
            and not self.iisph_production_qualified
            and not self.dfsph_production_qualified
            and self.distributed_migration_count >= 1
            and self.mixed_precision_error <= 1e-6
            and self.finite
        )


def run_commercial_particle_benchmark():
    count = 6
    spacing = 1.0 / count
    box = phx.discretization.ParticleBox([0.0], [1.0])
    first_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=1, name="first"
    ).prepare()
    second_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=1, name="second"
    ).prepare()
    first = phx.discretization.ParticlePopulation(
        "first", first_particles, role="material-phase", state_shape=(count, 3)
    )
    second = phx.discretization.ParticlePopulation(
        "second", second_particles, role="material-phase", state_shape=(count, 3)
    )
    prepared_cells = phx.discretization.MultiPopulationCellPlan(
        box, 0.25, (4, 4)
    ).prepare((first, second))
    first_position = (jnp.arange(count, dtype=float) + 0.25)[:, None] * spacing
    second_position = (jnp.arange(count, dtype=float) + 0.75)[:, None] * spacing
    cell_state = prepared_cells.build((first_position, second_position))
    relation = prepared_cells.bipartite_relation(
        cell_state,
        (first_position, second_position),
        phx.discretization.ParticleSearchKey(
            first.population_id, second.population_id, 0.3
        ),
        36,
    )
    matrix = jnp.asarray([[[2.0, 0.5], [0.5, 1.5]]])
    rhs = jnp.asarray([[1.0, 2.0]])
    small = phx.linalg.solve_small_linear(phx.linalg.SmallLinearSolvePlan(2), matrix, rhs)
    root = phx.discretization.solve_adaptive_h_root(
        phx.discretization.AdaptiveHRootPlan(1.2, 1, 0.1, 1.0),
        jnp.asarray([1.0]),
        lambda h: jnp.asarray([4.0]),
        jnp.asarray([0.8]),
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2, box=box
    ).prepare(particles)
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    velocity = jnp.zeros_like(position)
    kernel = phx.discretization.WendlandC2SPHKernel(1)
    iisph = phx.discretization.PreparedIISPH(
        particles,
        neighborhood,
        kernel,
        1.25 * spacing,
        phx.discretization.IISPHMethodPlan(1.0, maximum_iterations=2, tolerance=1.0),
    )
    iisph_result = iisph.step_detailed(
        0.0, iisph.initialize_state(position, velocity), 0.001
    )
    oracle = phx.discretization.assemble_iisph_operator(iisph, position, 0.001)
    dfsph = phx.discretization.PreparedDFSPH(
        particles,
        neighborhood,
        kernel,
        1.25 * spacing,
        phx.discretization.DFSPHMethodPlan(
            1.0,
            divergence_iterations=2,
            density_iterations=2,
            divergence_tolerance=1.0,
            density_tolerance=1.0,
        ),
    )
    dfsph_result = dfsph.step_detailed(
        0.0, dfsph.initialize_state(position, velocity), 0.001
    )
    decomposition = phx.discretization.ParticleDomainDecompositionPlan(2, 0.15, box)
    halo = phx.discretization.prepare_particle_halos(
        decomposition, position, particles.active_mask
    )
    migrated = phx.discretization.migrate_particle_halos(
        decomposition,
        halo,
        position + jnp.where(jnp.arange(count)[:, None] == 0, 0.6, 0.0),
        particles.active_mask,
    )
    precision = phx.discretization.certify_particle_precision(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([1.0, 2.0 + 1e-8]),
        phx.discretization.ParticlePrecisionPolicy(),
        tolerance=1e-6,
    )
    values = jnp.asarray(
        (
            small.residual_norm[0],
            root.residual,
            oracle.action_error,
            precision.relative_error,
        )
    )
    return CommercialParticleBenchmark(
        schema_version=1,
        maturity="experimental",
        native_pair_count=int(relation.pair_count),
        native_data_plane_successful=bool(relation.successful),
        small_solve_residual=float(small.residual_norm[0]),
        adaptive_h_residual=float(root.residual),
        iisph_action_error=float(oracle.action_error),
        iisph_production_qualified=bool(iisph_result.production_qualified),
        dfsph_production_qualified=bool(dfsph_result.production_qualified),
        distributed_migration_count=int(migrated.migration_count),
        mixed_precision_error=float(precision.relative_error),
        finite=bool(jnp.all(jnp.isfinite(values))),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Qualify commercial particle-hardening infrastructure."
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    record = run_commercial_particle_benchmark()
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
