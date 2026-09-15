#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from benchmarks._runtime import capture_environment


cosmology = phx.applications.cosmology


def _particle_mesh(particles, grid_count):
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(grid_count, periodic=True)
        for _ in range(3)
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y", "z")).prepare(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    system = phx.equations.EulerSystem(3)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "dark-matter-sidm-benchmark",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLCFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    gravity = phx.solver.ParticleMeshGravityPlan(
        phx.solver.NewtonianSelfGravityPlan(0.01).prepare(
            phx.solver.prepare_balance_law_transport(runtime)
        ),
        phx.discretization.ParticleGridSplatPlan(grid).prepare(particles),
    )
    kdk = cosmology.CosmologicalKDKPlan(particles, (1.0, 1.0, 1.0))
    return kdk, cosmology.CosmologicalParticleMeshPlan(kdk, gravity, (0.5, 0.51))


def _case(particle_count: int, grid_count: int, repetitions: int):
    indices = jnp.arange(particle_count, dtype=jnp.float64)
    positions = jnp.stack(
        (
            jnp.mod(0.5 + indices * 0.754877666, 1.0),
            jnp.mod(0.5 + indices * 0.569840291, 1.0),
            jnp.mod(0.5 + indices * 0.438579021, 1.0),
        ),
        axis=-1,
    )
    mass = jnp.full((particle_count,), 1.0 / particle_count)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count), mass, ambient_dimension=3
    ).prepare()
    kdk, particle_mesh = _particle_mesh(particles, grid_count)
    box = phx.discretization.ParticleBox(
        jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
    )
    pair_capacity = particle_count * (particle_count - 1) // 2
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        pair_capacity, box=box
    ).prepare(particles)
    sidm = cosmology.CosmologicalSIDMPlan(
        particle_mesh,
        neighborhood,
        phx.discretization.CoupledSummationSmoothingLengthPlan(
            1.0,
            1.0e-4,
            1.0,
            maximum_iterations=80,
            tolerance=1.0e-6,
            relaxation=0.7,
        ),
        phx.discretization.WendlandC2SPHKernel(3),
        cosmology.SIDMCrossSectionPlan(0.01),
        cosmology.SIDMCollisionPolicy(
            maximum_pair_probability=0.1,
            maximum_particle_probability=0.25,
            minimum_knudsen_number=1.0e-3,
            maximum_events_per_half_step=particle_count // 2,
        ),
    )
    velocity = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * positions[:, 0]),
            jnp.cos(2.0 * jnp.pi * positions[:, 1]),
            jnp.sin(2.0 * jnp.pi * positions[:, 2]),
        ),
        axis=-1,
    )
    state = kdk.initialize(positions, mass[:, None] * 0.5 * velocity, 0.5)
    collision = eqx.filter_jit(sidm.collide)

    started = time.perf_counter()
    first = collision(state, jr.key(0), 0, 0.25)
    jax.block_until_ready(first.accepted_state.canonical_momenta)
    compile_and_first_ms = 1000.0 * (time.perf_counter() - started)

    started = time.perf_counter()
    result = first
    for epoch in range(repetitions):
        result = collision(state, jr.key(epoch + 1), epoch + 1, 0.25)
    jax.block_until_ready(result.accepted_state.canonical_momenta)
    execution_ms = 1000.0 * (time.perf_counter() - started) / repetitions
    expected_events = float(jnp.sum(result.diagnostics.pair_probability))
    return {
        "particle_count": particle_count,
        "pair_capacity": pair_capacity,
        "grid_count_per_axis": grid_count,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "pairs_per_second": 1000.0 * pair_capacity / execution_ms,
        "expected_rare_events": expected_events,
        "accepted_events": int(result.diagnostics.event_count),
        "maximum_particle_probability": float(
            jnp.max(result.diagnostics.particle_aggregate_probability)
        ),
        "maximum_pair_probability": float(jnp.max(result.diagnostics.pair_probability)),
        "momentum_defect_norm": float(
            jnp.sqrt(jnp.sum(result.diagnostics.total_momentum_defect**2))
        ),
        "kinetic_energy_defect": float(result.diagnostics.total_kinetic_energy_defect),
        "successful": bool(result.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particle-counts", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--grid-count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/dark_matter_sidm.json"),
    )
    arguments = parser.parse_args()
    cases = [
        _case(count, arguments.grid_count, arguments.repeats)
        for count in arguments.particle_counts
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_successful": all(case["successful"] for case in cases),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
