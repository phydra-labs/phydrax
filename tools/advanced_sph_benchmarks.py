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


class IntervalGeometry:
    bounds = jnp.asarray([[0.0], [1.0]])

    @staticmethod
    def signed_distance(points):
        x = points[:, 0]
        return jnp.minimum(x, 1.0 - x)

    @staticmethod
    def boundary_normal(points):
        return jnp.where(points[:, :1] < 0.5, -1.0, 1.0)


@dataclass(frozen=True)
class AdvancedSPHBenchmark:
    schema_version: int
    maturity: str
    execution_successful: bool
    numerical_constraints_satisfied: bool
    production_gate_satisfied: bool
    stabilization_finite: bool
    viscous_dissipation: float
    artificial_dissipation: float
    density_variance_rate: float
    free_surface_count: int
    wall_action_reaction_defect: float
    wall_minimum_volume: float
    detected_open_surface_particles: int
    adaptive_omega_minimum: float
    multiphase_action_reaction_defect: float
    iisph_residual: float
    iisph_successful: bool
    iisph_production_qualified: bool
    dfsph_divergence_residual: float
    dfsph_density_residual: float
    dfsph_successful: bool
    dfsph_production_qualified: bool

    @property
    def passed(self):
        return (
            self.maturity == "experimental"
            and self.execution_successful
            and not self.numerical_constraints_satisfied
            and not self.production_gate_satisfied
            and not self.iisph_production_qualified
            and not self.dfsph_production_qualified
            and self.stabilization_finite
            and self.viscous_dissipation >= 0.0
            and self.artificial_dissipation >= 0.0
            and self.adaptive_omega_minimum > 0.0
            and self.wall_action_reaction_defect <= 1e-11
            and self.wall_minimum_volume > 0.0
            and self.detected_open_surface_particles >= 1
            and self.multiphase_action_reaction_defect <= 1e-11
            and self.iisph_successful
            and self.dfsph_successful
        )


def _phase(name, count=6):
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1, name=name
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    kernel = phx.discretization.WendlandC2SPHKernel(1)
    method = phx.discretization.WeaklyCompressibleSPHMethodPlan(
        kernel,
        1.25 * spacing,
        density=phx.discretization.ContinuityDensityPlan(),
        physical_viscosity=phx.discretization.MorrisViscosityPlan(0.01),
        artificial_viscosity=phx.discretization.MonaghanArtificialViscosityPlan(0.1),
        density_diffusion=phx.discretization.MolteniColagrossiDensityDiffusionPlan(0.05),
        free_surface_detection=phx.discretization.FreeSurfaceDetectionPlan(
            completeness_threshold=0.7, normal_threshold=0.01
        ),
    )
    compiled = phx.equations.compile_weakly_compressible_sph_problem(
        phx.equations.WeaklyCompressibleFluidProblemIR(
            name, phx.equations.TaitBarotropicMaterial(1.0, 2.0)
        ),
        particles,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2, box=box
        ),
    )
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    state = compiled.initialize_state(position, -0.02 * (position - 0.5))
    return compiled, position, state, box


def run_advanced_sph_benchmark():
    compiled, position, state, box = _phase("phase-a")
    diagnostics = compiled.dynamics.diagnostics(0.0, state, None)
    neighborhood = compiled.dynamics.neighborhood.build(position)
    geometry = phx.discretization.particle_pair_geometry(
        position, neighborhood.pair_relation, box=box
    )
    adaptive = phx.discretization.adaptive_smoothing_state(
        phx.discretization.AlgebraicSmoothingLengthPlan(1.2, 0.1, 0.4),
        compiled.dynamics.particles,
        neighborhood.pair_relation,
        geometry,
        compiled.dynamics.method.kernel,
        compiled.dynamics.execution,
        density=compiled.dynamics.state_layout.density(state),
    )
    second, second_position, second_state, _ = _phase("phase-b")
    phase_a = phx.discretization.PhaseDefinition("phase-a", compiled.dynamics)
    phase_b = phx.discretization.PhaseDefinition("phase-b", second.dynamics)
    relation = phx.discretization.DenseBipartiteParticleNeighborhoodPlan(36).prepare(
        compiled.dynamics.particles,
        second.dynamics.particles,
        target_population_id=phase_a.phase_id,
        source_population_id=phase_b.phase_id,
    )
    multiphase = phx.discretization.PreparedMultiphaseWCSPHDynamics(
        phase_a,
        phase_b,
        phx.discretization.MultiphaseWCSPHPlan(surface_tension=0.01),
        relation,
        box=box,
    )
    multiphase_diagnostics = multiphase.diagnostics(
        0.0, multiphase.pack(state, second_state), None
    )
    kernel = compiled.dynamics.method.kernel
    prepared_neighborhood = compiled.dynamics.neighborhood
    particles = compiled.dynamics.particles
    zero_velocity = jnp.zeros_like(position)
    iisph = phx.discretization.PreparedIISPH(
        particles,
        prepared_neighborhood,
        kernel,
        compiled.dynamics.method.smoothing_length,
        phx.discretization.IISPHMethodPlan(1.0, maximum_iterations=3, tolerance=1.0),
    )
    iisph_result = iisph.step_detailed(
        0.0, iisph.initialize_state(position, zero_velocity), 0.001
    )
    dfsph = phx.discretization.PreparedDFSPH(
        particles,
        prepared_neighborhood,
        kernel,
        compiled.dynamics.method.smoothing_length,
        phx.discretization.DFSPHMethodPlan(
            1.0,
            divergence_iterations=3,
            density_iterations=3,
            divergence_tolerance=1.0,
            density_tolerance=1.0,
        ),
    )
    dfsph_result = dfsph.step_detailed(
        0.0, dfsph.initialize_state(position, zero_velocity), 0.001
    )
    kernel = compiled.dynamics.method.kernel
    wall = phx.discretization.WallParticleGenerationPlan(
        IntervalGeometry(), kernel, 0.25, 0.3, layers=1
    ).prepare()
    wall_relation = (
        phx.discretization.DenseBipartiteParticleNeighborhoodPlan(
            compiled.dynamics.particles.capacity * wall.particles.capacity
        )
        .prepare(
            compiled.dynamics.particles,
            wall.particles,
            target_population_id="fluid",
            source_population_id="wall",
        )
        .build(position, wall.positions)
    )
    _, velocity, density = compiled.dynamics.state_layout.unpack(state)
    wall_result = phx.discretization.evaluate_wall_interaction(
        phx.discretization.AdamiWallBoundaryPlan(compiled.problem.material),
        wall,
        wall_relation,
        position,
        velocity,
        density,
        compiled.problem.material.pressure(density),
        compiled.dynamics.particles.safe_masses,
        kernel,
        0.3,
    )
    open_prepared = phx.discretization.DenseParticleNeighborhoodPlan(
        compiled.dynamics.particles.capacity
        * (compiled.dynamics.particles.capacity - 1)
        // 2
    ).prepare(compiled.dynamics.particles)
    open_state = open_prepared.build(position)
    open_geometry = phx.discretization.particle_pair_geometry(
        position, open_state.pair_relation
    )
    open_physical = open_geometry.valid & (open_geometry.distance < 0.4)
    free_surface = phx.discretization.detect_free_surface(
        phx.discretization.FreeSurfaceDetectionPlan(
            completeness_threshold=0.95,
            normal_threshold=0.01,
            cone_angle=1.2,
        ),
        compiled.dynamics.particles,
        density,
        open_state.pair_relation,
        open_geometry,
        open_physical,
        kernel,
        0.2,
        compiled.dynamics.execution,
    )
    finite_values = jnp.asarray(
        (
            diagnostics.viscous_dissipation_rate,
            diagnostics.artificial_viscosity_dissipation,
            diagnostics.density_variance_rate,
            jnp.min(adaptive.omega),
            jnp.max(jnp.abs(multiphase_diagnostics.total_momentum_rate)),
            iisph_result.residual,
            dfsph_result.divergence_residual,
            dfsph_result.density_residual,
            jnp.max(jnp.abs(wall_result.ledger.action_reaction_defect)),
            jnp.min(wall.volumes),
            jnp.sum(free_surface.hard_mask),
        )
    )
    return AdvancedSPHBenchmark(
        schema_version=2,
        maturity="experimental",
        execution_successful=bool(iisph_result.successful & dfsph_result.successful),
        numerical_constraints_satisfied=bool(
            iisph_result.numerical_constraints_satisfied
            & dfsph_result.numerical_constraints_satisfied
        ),
        production_gate_satisfied=bool(
            iisph_result.production_qualified & dfsph_result.production_qualified
        ),
        stabilization_finite=bool(jnp.all(jnp.isfinite(finite_values))),
        viscous_dissipation=float(diagnostics.viscous_dissipation_rate),
        artificial_dissipation=float(diagnostics.artificial_viscosity_dissipation),
        density_variance_rate=float(diagnostics.density_variance_rate),
        free_surface_count=int(diagnostics.free_surface_count),
        adaptive_omega_minimum=float(jnp.min(adaptive.omega)),
        multiphase_action_reaction_defect=float(
            jnp.max(jnp.abs(multiphase_diagnostics.total_momentum_rate))
        ),
        wall_action_reaction_defect=float(
            jnp.max(jnp.abs(wall_result.ledger.action_reaction_defect))
        ),
        wall_minimum_volume=float(jnp.min(wall.volumes)),
        detected_open_surface_particles=int(jnp.sum(free_surface.hard_mask)),
        iisph_residual=float(iisph_result.residual),
        iisph_successful=bool(iisph_result.successful),
        iisph_production_qualified=bool(iisph_result.production_qualified),
        dfsph_divergence_residual=float(dfsph_result.divergence_residual),
        dfsph_density_residual=float(dfsph_result.density_residual),
        dfsph_successful=bool(dfsph_result.successful),
        dfsph_production_qualified=bool(dfsph_result.production_qualified),
    )


def main():
    parser = argparse.ArgumentParser(description="Qualify advanced particle methods.")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    record = run_advanced_sph_benchmark()
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
