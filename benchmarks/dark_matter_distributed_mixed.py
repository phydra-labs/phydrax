from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment, measure_repeated, measure_synchronized

import phydrax as phx
from phydrax.applications.cosmology._coupled import ComovingEulerPlan
from phydrax.applications.cosmology._distributed_mixed import (
    DistributedMixedExecutionPlan,
)
from phydrax.applications.cosmology._force_scalability import (
    DistributedPMFeasibilityEvidence,
)
from phydrax.applications.cosmology._mixed_matter import (
    WaveParticleGasCosmologyPlan,
    WaveParticleGasCosmologyState,
)
from phydrax.applications.cosmology._particles import CosmologicalKDKPlan
from phydrax.applications.cosmology._wave_dark_matter import (
    WaveDarkMatterPlan,
    WaveDarkMatterStepPolicy,
)
from phydrax.solver._particle_gravity import DistributedParticleLayout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--maximum-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.size < 4
        or arguments.parts < 1
        or arguments.steps < 1
        or arguments.warmup < 0
        or arguments.repeats < 1
        or arguments.maximum_bytes <= 0
    ):
        raise ValueError(
            "size >= 4 and positive parts/steps/repeats/resources are required"
        )
    devices = tuple(jax.devices()[: arguments.parts])
    if len(devices) != arguments.parts:
        raise RuntimeError(
            f"requested {arguments.parts} real devices but only {len(jax.devices())} are available"
        )
    count = arguments.size
    particle_count = count**2
    if count % arguments.parts or particle_count % arguments.parts:
        raise ValueError("grid and particle capacities must divide exactly across parts")

    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    system = phx.equations.EulerSystem(2)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "distributed-mixed-benchmark",
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
    gravity_owner = phx.solver.NewtonianSelfGravityPlan(0.02).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )
    coordinates = jnp.meshgrid(
        *(grid.structured_axes[axis].interval_centers for axis in range(2)),
        indexing="ij",
    )
    positions = jnp.stack(tuple(value.reshape((-1,)) for value in coordinates), axis=-1)
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count, dtype=jnp.int64),
        jnp.full((particle_count,), 1.0 / particle_count),
        ambient_dimension=2,
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(support)
    particle_gravity = phx.solver.ParticleMeshGravityPlan(gravity_owner, transfer)
    particles = CosmologicalKDKPlan(support, (1.0, 1.0))
    half_cell = 0.5 / count
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(2)),
        axis_names=("x", "y"),
        field_name="psi",
    ).prepare(
        tuple(
            phx.discretization.AxisDomain.periodic(half_cell, 1.0 + half_cell)
            for _ in range(2)
        )
    )
    background = phx.applications.cosmology.FLRWBackground(1.0, 1.0)
    schedule = 1.0 + 1.0e-6 * jnp.arange(arguments.steps + 1)
    wave = WaveDarkMatterPlan(
        1.0,
        schedule,
        gravitational_constant=0.02,
        reduced_planck_constant=0.03,
        step_policy=WaveDarkMatterStepPolicy(
            maximum_phase_radians=2.0,
            minimum_de_broglie_cells=2.0,
            norm_relative_tolerance=1.0e-7,
            poisson_relative_tolerance=1.0e-7,
            zero_mode_absolute_tolerance=1.0e-8,
        ),
    ).prepare(space, background)
    gas = ComovingEulerPlan(
        dynamics,
        adiabatic_index=5.0 / 3.0,
        expansion_dimension=2,
        substeps=2,
    )
    mixed = WaveParticleGasCosmologyPlan(wave, particles, gas, particle_gravity).prepare()
    perturbation = (
        1.0e-4
        * (
            jnp.cos(2.0 * jnp.pi * coordinates[0])
            + jnp.cos(2.0 * jnp.pi * coordinates[1])
        )
        / 2.0
    )
    gas_average = jnp.zeros((count, count, 4), dtype=jnp.float64)
    gas_average = gas_average.at[..., 0].set(1.0 + perturbation)
    gas_average = gas_average.at[..., -1].set(1.0)
    initial = WaveParticleGasCosmologyState(
        wave.initialize(jnp.sqrt(1.0 + perturbation).astype(jnp.complex128)),
        particles.initialize(positions, jnp.zeros_like(positions), schedule[0]),
        gas.initialize(gas_average, schedule[0]),
    )
    topology = phx.discretization.SpectralMeshTopology(
        (arguments.parts,), devices=devices, axis_names=("mixed",)
    )
    spectral = phx.discretization.DistributedSpectralExecutionPlan.from_discretization(
        topology,
        space,
        checkpoint_count=1,
        maximum_bytes=arguments.maximum_bytes,
    )
    group = phx.execution.ExecutionGroupSpec(
        "distributed-mixed-benchmark-group",
        sorted({key[0] for key in topology.device_keys}),
        topology.device_keys,
        mesh_axes=tuple(zip(topology.mesh_axis_names, topology.mesh_shape, strict=True)),
    )
    placements = tuple(
        phx.execution.ValuePlacement(name, phx.execution.PlacementKind.PARTITIONED)
        for name in (
            "wave",
            "particle_positions",
            "particle_momenta",
            "potential",
            "gas",
        )
    )
    execution_plan = phx.execution.ExecutionPlan(
        "distributed-mixed-benchmark-execution",
        "jax-distributed",
        "float64",
        "mixed-kdk",
        device_mesh_id=topology.topology_id,
        group=group,
        value_placements=placements,
    )
    capacity_per_device = particle_count // arguments.parts
    particle_layout = DistributedParticleLayout(
        arguments.parts,
        capacity_per_device,
        jnp.linspace(
            0,
            np.iinfo(np.uint32).max,
            arguments.parts + 1,
            dtype=jnp.uint32,
        ),
    )
    feasibility = DistributedPMFeasibilityEvidence(
        space.modal_shape,
        (arguments.parts, 1),
        capacity_per_device,
        byte_budget_per_device=arguments.maximum_bytes,
    )
    preparation = DistributedMixedExecutionPlan(
        mixed,
        spectral,
        execution_plan,
        particle_layout,
        feasibility,
        maximum_checkpoint_bytes=arguments.maximum_bytes,
        particle_send_capacity=capacity_per_device,
        particle_receive_capacity=capacity_per_device,
        particle_ghost_capacity=capacity_per_device,
        particle_ghost_width=half_cell,
    ).prepare()
    if not bool(preparation.successful) or preparation.executable is None:
        raise RuntimeError(
            f"distributed mixed preparation refused: {preparation.evidence.reasons}"
        )
    distributed = preparation.executable
    state = distributed.initialize(initial)
    first, first_seconds = measure_synchronized(lambda: distributed.rollout(state))
    result, execution = measure_repeated(
        lambda: distributed.rollout(state),
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    if not bool(first.successful) or not bool(result.successful):
        raise RuntimeError("distributed mixed benchmark rollout failed atomically")
    median_seconds = execution.median_seconds
    assert median_seconds is not None
    updates = particle_count * arguments.steps
    checkpoint = preparation.evidence.checkpoint
    resource = spectral.report.resource
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "global_shape": (count, count),
            "parts": arguments.parts,
            "particles": particle_count,
            "particle_capacity_per_device": capacity_per_device,
            "steps": arguments.steps,
            "gas_substeps": gas.substeps,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "identity": {
            "execution": distributed.execution_id,
            "mesh": topology.topology_id,
            "particle_layout": particle_layout.layout_id,
            "checkpoint_shard_plan": checkpoint.shard_plan_id,
            "checkpoint_physics": distributed.checkpoint_physics_id,
            "checkpoint_schema": distributed.checkpoint_schema_id,
            "checkpoint_numeric": distributed.checkpoint_numeric_id,
            "checkpoint_execution": distributed.checkpoint_execution_id,
        },
        "preflight": {
            "status": preparation.evidence.status,
            "required_checkpoint_bytes": checkpoint.required_bytes,
            "maximum_checkpoint_bytes": checkpoint.maximum_bytes,
            "checkpoint_unpadded_bytes": distributed.checkpoint_unpadded_bytes,
            "checkpoint_payload_bytes": int(
                distributed.checkpoint_tree(state)["payload"].size
            ),
            "checkpoint_padding_bytes": (
                distributed.checkpoint_payload_bytes
                - distributed.checkpoint_unpadded_bytes
            ),
            "checkpoint_payload_alignment": distributed.checkpoint_payload_alignment,
            "spectral_state_bytes": resource.state_bytes,
            "spectral_collective_bytes": resource.collective_bytes,
            "estimated_mesh_bytes_per_device": feasibility.estimated_mesh_bytes_per_device,
            "estimated_transpose_bytes": feasibility.estimated_transpose_bytes,
            "particle_exchange_bytes_per_device": (
                preparation.evidence.collective.particle_exchange_bytes_per_device
            ),
            "host_gather_fallback": preparation.evidence.collective.host_gather_fallback,
        },
        "compilation": {"first_compile_and_execution_seconds": first_seconds},
        "execution": execution.to_seconds_dict(),
        "scaling": {
            "particle_updates_per_second": updates / median_seconds,
            "cell_updates_per_second": count**2 * arguments.steps / median_seconds,
            "bytes_per_device": (
                feasibility.estimated_mesh_bytes_per_device
                + feasibility.estimated_transpose_bytes
                + preparation.evidence.collective.particle_exchange_bytes_per_device
            ),
        },
        "physics": {
            "accepted_steps": int(result.accepted_steps),
            "initial_mass": float(result.initial_mass),
            "final_mass": float(result.final_mass),
            "maximum_mass_balance_defect": float(result.maximum_mass_balance_defect),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
