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
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm import SIDMCollisionPolicy
from phydrax.applications.cosmology._sidm_kernels import TwoBodyDifferentialKernelPlan
from phydrax.applications.cosmology._sidm_weighted import WeightedSIDMPlan


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
    finite_volume = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "weighted-sidm-benchmark",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        finite_volume,
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
    kinematics = cosmology.CosmologicalKDKPlan(particles, (1.0, 1.0, 1.0))
    return cosmology.CosmologicalParticleMeshPlan(
        kinematics, gravity, jnp.asarray((0.5, 0.51))
    )


def _case(root_count: int, grid_count: int, repetitions: int):
    capacity = 2 * root_count
    indices = jnp.arange(capacity, dtype=jnp.float64)
    positions = jnp.stack(
        (
            jnp.mod(0.5 + indices * 0.754877666, 1.0),
            jnp.mod(0.5 + indices * 0.569840291, 1.0),
            jnp.mod(0.5 + indices * 0.438579021, 1.0),
        ),
        axis=-1,
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(capacity, dtype=jnp.int64),
        jnp.ones((capacity,)),
        ambient_dimension=3,
    ).prepare()
    box = phx.discretization.ParticleBox(
        jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        capacity * (capacity - 1) // 2, box=box
    ).prepare(particles)
    species = DarkSectorSpeciesPlan("chi", 1.0)
    plan = WeightedSIDMPlan(
        _particle_mesh(particles, grid_count),
        neighborhood,
        phx.discretization.WendlandC2SPHKernel(3),
        TwoBodyDifferentialKernelPlan.constant_isotropic(species, 1.0e-4),
        SIDMCollisionPolicy(
            maximum_pair_probability=0.1,
            maximum_particle_probability=0.25,
            minimum_knudsen_number=1.0e-6,
            maximum_events_per_half_step=root_count // 2,
        ),
        smoothing_length_comoving=0.2,
    )
    active = jnp.arange(capacity) < root_count
    weights = jnp.where(active, 1.0 + (jnp.arange(capacity) % 3), 0.0)
    microscopic = jnp.where(active, 1.0, 0.0)
    velocity = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * positions[:, 0]),
            jnp.cos(2.0 * jnp.pi * positions[:, 1]),
            jnp.sin(2.0 * jnp.pi * positions[:, 2]),
        ),
        axis=-1,
    )
    state = plan.initialize(
        positions,
        microscopic,
        weights,
        weights[:, None] * 0.5 * velocity,
        0.5,
        active_mask=active,
    )
    collide = eqx.filter_jit(plan.collide)
    started = time.perf_counter()
    root_key = jr.key(0)
    first = collide(state, jr.fold_in(root_key, 0), 0, 1.0e-3)
    jax.block_until_ready(first)
    compile_and_first_ms = 1000.0 * (time.perf_counter() - started)

    started = time.perf_counter()
    result = first
    for epoch in range(repetitions):
        result = collide(state, jr.fold_in(root_key, epoch + 1), epoch + 1, 1.0e-3)
        jax.block_until_ready(result)
    execution_ms = 1000.0 * (time.perf_counter() - started) / repetitions
    return {
        "root_packet_count": root_count,
        "packet_capacity": capacity,
        "pair_capacity": neighborhood.pair_capacity,
        "grid_count_per_axis": grid_count,
        "compile_and_first_ms": compile_and_first_ms,
        "rng": {"root_seed": 0, "stream": "fold_in(epoch)"},
        "execution_ms": execution_ms,
        "pairs_per_second": 1000.0 * neighborhood.pair_capacity / execution_ms,
        "accepted_events": int(result.diagnostics.event_count),
        "children_required": int(result.diagnostics.child_required),
        "children_created": int(result.diagnostics.child_slots_used),
        "mass_defect": float(result.diagnostics.mass_defect),
        "momentum_defect_norm": float(
            jnp.sqrt(jnp.sum(result.diagnostics.momentum_defect**2))
        ),
        "kinetic_energy_defect": float(result.diagnostics.kinetic_energy_defect),
        "successful": bool(result.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-counts", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--grid-count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/dark_matter_sidm_weighted.json"),
    )
    arguments = parser.parse_args()
    cases = [
        _case(count, arguments.grid_count, arguments.repeats)
        for count in arguments.root_counts
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
