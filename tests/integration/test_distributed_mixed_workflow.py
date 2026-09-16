from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
from phydrax.lifecycle import (
    assemble_distributed_checkpoint_from_repository,
    CheckpointManifest,
    publish_process_checkpoint,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.solver._particle_gravity import DistributedParticleLayout


def _workflow():
    count = 4
    particle_count = count**2
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
        "distributed-mixed-integration",
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
    position = jnp.stack(tuple(value.reshape((-1,)) for value in coordinates), axis=-1)
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count),
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
    wave = WaveDarkMatterPlan(
        1.0,
        jnp.asarray((1.0, 1.000001)),
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
    state = WaveParticleGasCosmologyState(
        wave.initialize(jnp.sqrt(1.0 + perturbation).astype(jnp.complex128)),
        particles.initialize(position, jnp.zeros_like(position), 1.0),
        gas.initialize(gas_average, 1.0),
    )
    topology = phx.discretization.SpectralMeshTopology.one_device()
    spectral = phx.discretization.DistributedSpectralExecutionPlan.from_discretization(
        topology, space, checkpoint_count=1, maximum_bytes=2**28
    )
    group = phx.execution.ExecutionGroupSpec(
        "distributed-mixed-integration-group",
        sorted({key[0] for key in topology.device_keys}),
        topology.device_keys,
        mesh_axes=tuple(zip(topology.mesh_axis_names, topology.mesh_shape, strict=True)),
    )
    placements = tuple(
        phx.execution.ValuePlacement(name, phx.execution.PlacementKind.PARTITIONED)
        for name in ("wave", "particle_positions", "particle_momenta", "potential", "gas")
    )
    execution = phx.execution.ExecutionPlan(
        "distributed-mixed-integration-execution",
        "jax-distributed",
        "float64",
        "mixed-kdk-gas",
        device_mesh_id=topology.topology_id,
        group=group,
        value_placements=placements,
    )
    particle_layout = DistributedParticleLayout(
        1,
        particle_count,
        jnp.asarray((0, np.iinfo(np.uint32).max), dtype=jnp.uint32),
    )
    feasibility = DistributedPMFeasibilityEvidence(
        space.modal_shape, (1, 1), particle_count, byte_budget_per_device=2**28
    )
    preparation = DistributedMixedExecutionPlan(
        mixed,
        spectral,
        execution,
        particle_layout,
        feasibility,
        maximum_checkpoint_bytes=2**28,
        particle_ghost_capacity=particle_count,
        particle_ghost_width=half_cell,
    ).prepare()
    assert preparation.executable is not None
    return mixed, state, preparation.executable


def _repository(path: Path) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        "posix.distributed-mixed",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXArtifactRepository(
        path,
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=2**20,
            maximum_metadata_bytes=2**20,
        ),
    )


def test_distributed_wave_particle_gas_rollout_is_atomic_and_matches_single_part():
    mixed, state, execution = _workflow()
    local = mixed.rollout(state)
    initialized = execution.initialize(state)
    step = execution.advance(initialized, mixed.plan.wave.scale_factors[1])
    distributed = execution.rollout(initialized)
    logical = execution.particle_runtime.logical_arrays(distributed.state.particles)

    assert bool(local.successful)
    assert bool(distributed.successful)
    assert bool(step.homogeneous_gas_successful)
    assert bool(step.phase_resolved)
    np.testing.assert_allclose(
        distributed.state.wave.psi, local.state.wave.psi, rtol=3e-10
    )
    np.testing.assert_allclose(
        logical["positions"], local.state.particles.positions, rtol=3e-10
    )
    np.testing.assert_allclose(
        logical["momenta"], local.state.particles.canonical_momenta, rtol=3e-10
    )
    assert distributed.state.gas is not None
    np.testing.assert_allclose(
        distributed.state.gas.cell_average, local.state.gas.cell_average, rtol=3e-9
    )
    np.testing.assert_allclose(
        distributed.initial_mass, distributed.final_mass, rtol=2e-10
    )


def test_distributed_checkpoint_has_exact_coverage_and_restores_into_destination_sharding(
    tmp_path: Path,
):
    _, state, execution = _workflow()
    distributed = execution.initialize(
        state, rng_counters=jnp.arange(16, dtype=jnp.uint64)
    )
    repository = _repository(tmp_path / "distributed-mixed")
    publication = publish_process_checkpoint(
        repository,
        "distributed-mixed-checkpoint",
        execution.checkpoint_execution_id,
        execution.checkpoint_tree(distributed),
        writer_id="distributed-mixed-writer",
        topology_epoch=execution.plan.execution.topology_epoch,
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "distributed-mixed-checkpoint",
        execution.checkpoint_schema_id,
        execution.checkpoint_numeric_id,
        execution.checkpoint_execution_id,
        expected_process_count=jax.process_count(),
        diagnostic_ids=(execution.checkpoint_physics_id,),
    )
    restored = execution.restore_checkpoint(repository, manifest, distributed)

    assert publication.shards
    assert manifest.complete
    np.testing.assert_allclose(restored.wave.psi, distributed.wave.psi)
    np.testing.assert_array_equal(
        restored.particles.stable_ids, distributed.particles.stable_ids
    )
    np.testing.assert_array_equal(
        restored.particles.rng_counters, distributed.particles.rng_counters
    )
    assert restored.wave.psi.sharding == execution.field_sharding
    assert (
        restored.particles.positions.sharding
        == execution.particle_runtime.vector_sharding
    )


def test_checkpoint_restore_rejects_incomplete_and_identity_substitution(tmp_path: Path):
    _, state, execution = _workflow()
    distributed = execution.initialize(state)
    repository = _repository(tmp_path / "distributed-mixed-rejection")
    publish_process_checkpoint(
        repository,
        "distributed-mixed-rejection",
        execution.checkpoint_execution_id,
        execution.checkpoint_tree(distributed),
        writer_id="distributed-mixed-rejection-writer",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "distributed-mixed-rejection",
        execution.checkpoint_schema_id,
        execution.checkpoint_numeric_id,
        execution.checkpoint_execution_id,
        expected_process_count=jax.process_count(),
        diagnostic_ids=(execution.checkpoint_physics_id,),
    )
    incomplete = CheckpointManifest(
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
        manifest.shards,
        complete=False,
        diagnostic_ids=manifest.diagnostic_ids,
    )
    substituted = CheckpointManifest(
        manifest.checkpoint_id,
        f"{manifest.analysis_plan_id}-substituted",
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
        manifest.shards,
        complete=True,
        diagnostic_ids=manifest.diagnostic_ids,
    )
    physics_substituted = CheckpointManifest(
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
        manifest.shards,
        complete=True,
        diagnostic_ids=(f"{execution.checkpoint_physics_id}-substituted",),
    )
    numeric_substituted = CheckpointManifest(
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        f"{manifest.numeric_revision_id}-substituted",
        manifest.execution_plan_id,
        manifest.shards,
        complete=True,
        diagnostic_ids=manifest.diagnostic_ids,
    )

    with pytest.raises(ValueError, match="incomplete|identity"):
        execution.restore_checkpoint(repository, incomplete, distributed)
    with pytest.raises(ValueError, match="incomplete|identity"):
        execution.restore_checkpoint(repository, substituted, distributed)
    with pytest.raises(ValueError, match="incomplete|identity"):
        execution.restore_checkpoint(repository, physics_substituted, distributed)
    with pytest.raises(ValueError, match="incomplete|identity"):
        execution.restore_checkpoint(repository, numeric_substituted, distributed)
