from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.cosmology._distributed_mixed import (
    DistributedMixedExecutionPlan,
)
from phydrax.applications.cosmology._force_scalability import (
    DistributedPMFeasibilityEvidence,
)
from phydrax.applications.cosmology._mixed_matter import (
    WaveParticleCosmologyPlan,
    WaveParticleCosmologyState,
)
from phydrax.applications.cosmology._particles import CosmologicalKDKPlan
from phydrax.applications.cosmology._wave_dark_matter import (
    WaveDarkMatterPlan,
    WaveDarkMatterStepPolicy,
)
from phydrax.lifecycle import (
    assemble_distributed_checkpoint_from_repository,
    publish_process_checkpoint,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.solver._particle_gravity import DistributedParticleLayout


def _prepared(
    parts: int = 1,
    *,
    count: int = 4,
    send_capacity: int | None = None,
    maximum_phase_radians: float = 2.0,
):
    particle_count = count**2
    devices = tuple(jax.devices()[:parts])
    if len(devices) != parts:
        pytest.skip(f"test requires {parts} real JAX devices")
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
        "distributed-mixed-unit",
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
    gravity_owner = phx.solver.NewtonianSelfGravityPlan(0.05).prepare(
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
    wave = WaveDarkMatterPlan(
        1.0,
        jnp.asarray((1.0, 1.00001)),
        gravitational_constant=0.05,
        reduced_planck_constant=0.05,
        step_policy=WaveDarkMatterStepPolicy(
            maximum_phase_radians=maximum_phase_radians,
            minimum_de_broglie_cells=2.0,
            poisson_relative_tolerance=1.0e-8,
            zero_mode_absolute_tolerance=1.0e-8,
            norm_relative_tolerance=1.0e-8,
        ),
    ).prepare(space, background)
    mixed = WaveParticleCosmologyPlan(wave, particles, particle_gravity).prepare()
    perturbation = (
        1.0e-3
        * (
            jnp.cos(2.0 * jnp.pi * coordinates[0])
            + jnp.cos(2.0 * jnp.pi * coordinates[1])
        )
        / 2.0
    )
    state = WaveParticleCosmologyState(
        wave.initialize(jnp.sqrt(1.0 + perturbation).astype(jnp.complex128)),
        particles.initialize(positions, jnp.zeros_like(positions), 1.0),
    )
    topology = phx.discretization.SpectralMeshTopology(
        (parts,), devices=devices, axis_names=("mixed",)
    )
    spectral = phx.discretization.DistributedSpectralExecutionPlan.from_discretization(
        topology,
        space,
        checkpoint_count=1,
        maximum_bytes=2**28,
    )
    group = phx.execution.ExecutionGroupSpec(
        "distributed-mixed-unit-group",
        sorted({key[0] for key in topology.device_keys}),
        topology.device_keys,
        mesh_axes=tuple(zip(topology.mesh_axis_names, topology.mesh_shape, strict=True)),
    )
    placements = tuple(
        phx.execution.ValuePlacement(name, phx.execution.PlacementKind.PARTITIONED)
        for name in ("wave", "particle_positions", "particle_momenta", "potential")
    )
    execution = phx.execution.ExecutionPlan(
        "distributed-mixed-unit-execution",
        "jax-distributed",
        "float64",
        "mixed-kdk",
        device_mesh_id=topology.topology_id,
        group=group,
        value_placements=placements,
    )
    capacity = particle_count // parts
    layout = DistributedParticleLayout(
        parts,
        capacity,
        jnp.linspace(0, np.iinfo(np.uint32).max, parts + 1, dtype=jnp.uint32),
    )
    feasibility = DistributedPMFeasibilityEvidence(
        space.modal_shape,
        (parts, 1),
        capacity,
        byte_budget_per_device=2**28,
    )
    plan = DistributedMixedExecutionPlan(
        mixed,
        spectral,
        execution,
        layout,
        feasibility,
        maximum_checkpoint_bytes=2**28,
        particle_send_capacity=send_capacity,
        particle_ghost_capacity=capacity,
        particle_ghost_width=0.5 / count,
    )
    preparation = plan.prepare()
    assert bool(preparation.successful)
    assert preparation.executable is not None
    return mixed, state, preparation.executable


def test_single_part_density_poisson_force_and_mass_match_local_authority():
    mixed, state, distributed = _prepared()
    local_density = mixed.density.assemble(state)
    local_gravity = mixed.gravity.solve(local_density)
    sharded = distributed.initialize(state)
    density = distributed.assemble_density(sharded)
    gravity = distributed.solve_gravity(sharded)

    np.testing.assert_allclose(
        density.total_density, local_density.total_density, rtol=2e-12
    )
    np.testing.assert_allclose(
        density.component_mass, local_density.component_mass, rtol=2e-12
    )
    np.testing.assert_allclose(
        gravity.potential, local_gravity.potential, rtol=2e-10, atol=2e-12
    )
    np.testing.assert_allclose(
        gravity.particle_acceleration,
        local_gravity.particle_acceleration,
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(gravity.total_force, local_gravity.total_force, atol=2e-11)
    assert bool(gravity.successful)


def test_phase_gate_matches_local_authority_and_rolls_back_atomically():
    mixed, state, distributed = _prepared(maximum_phase_radians=1.0e-20)
    local = mixed.rollout(state)
    sharded = distributed.initialize(state)
    step = distributed.advance(sharded, mixed.plan.wave.scale_factors[1])

    assert not bool(local.successful)
    assert not bool(step.successful)
    assert not bool(step.phase_resolved)
    assert float(step.kinetic_phase) > 0.0 or float(step.potential_phase) > 0.0
    np.testing.assert_array_equal(step.state.wave.psi, sharded.wave.psi)
    np.testing.assert_array_equal(
        step.state.particles.positions, sharded.particles.positions
    )


@pytest.mark.parametrize(
    ("wave_scale", "particle_scale"),
    (
        (np.nan, 1.0),
        (1.0, 1.000001),
        (1.000001, 1.000001),
    ),
)
def test_initialize_rejects_nonfinite_misaligned_or_noninitial_scales(
    wave_scale, particle_scale
):
    _, state, distributed = _prepared()
    invalid = WaveParticleCosmologyState(
        type(state.wave)(state.wave.psi, jnp.asarray(wave_scale)),
        type(state.particles)(
            state.particles.positions,
            state.particles.canonical_momenta,
            jnp.asarray(particle_scale),
        ),
    )

    with pytest.raises(Exception, match="finite, positive, aligned|first scheduled"):
        initialized = distributed.initialize(invalid)
        jax.block_until_ready(initialized.wave.scale_factor)


def test_initialize_rejects_runtime_component_precision_substitution():
    _, state, distributed = _prepared()
    particles = type(state.particles)(
        state.particles.positions.astype(jnp.float32),
        state.particles.canonical_momenta.astype(jnp.float32),
        state.particles.scale_factor.astype(jnp.float32),
    )
    substituted = WaveParticleCosmologyState(state.wave, particles)

    with pytest.raises(TypeError, match="precision ABI"):
        distributed.initialize(substituted)


def test_spectral_state_and_precision_abi_mismatch_refuses_preparation():
    _, _, distributed = _prepared()
    admitted = distributed.plan
    wave = admitted.mixed.plan.wave
    topology = admitted.spectral.topology
    invalid_spectral = (
        phx.discretization.DistributedSpectralExecutionPlan.from_discretization(
            topology,
            wave.discretization,
            state_shape=(1,),
            checkpoint_count=1,
            maximum_bytes=2**28,
        ),
        phx.discretization.DistributedSpectralExecutionPlan(
            topology,
            admitted.spectral.spatial_shape,
            schedule="slab",
            domain_lengths=admitted.spectral.domain_lengths,
            coefficient_dtype=jnp.complex64,
            accumulation_dtype=jnp.float64,
            checkpoint_count=1,
            maximum_bytes=2**28,
        ),
    )
    for spectral in invalid_spectral:
        result = DistributedMixedExecutionPlan(
            admitted.mixed,
            spectral,
            admitted.execution,
            admitted.particles,
            admitted.feasibility,
            maximum_checkpoint_bytes=2**28,
            particle_ghost_capacity=admitted.particle_ghost_capacity,
            particle_ghost_width=admitted.particle_ghost_width,
        ).prepare()
        assert not bool(result.successful)
        assert result.executable is None
        assert "spectral-abi" in result.evidence.collective.missing_primitives


def test_particle_migration_preserves_stable_ids_and_rng_at_periodic_boundary():
    _, state, distributed = _prepared(parts=2)
    sharded = distributed.initialize(state, rng_counters=jnp.arange(16, dtype=jnp.uint64))
    logical = distributed.particle_runtime.logical_arrays(sharded.particles)
    proposed = logical["positions"].at[3, 0].set(0.51).at[15, 0].set(0.01)
    owner_proposed = distributed.particle_runtime.owner_order(proposed, sharded.particles)
    owner_momenta = sharded.particles.momenta
    owner_rng = sharded.particles.rng_counters + sharded.particles.active_mask.astype(
        jnp.uint64
    )
    migrated = distributed.particle_runtime.migrate(
        sharded.particles,
        owner_proposed,
        owner_momenta,
        proposed_rng_counters=owner_rng,
    )
    restored = distributed.particle_runtime.logical_arrays(migrated.state)

    assert bool(migrated.evidence.successful)
    assert int(migrated.evidence.migration_count) == 2
    np.testing.assert_array_equal(restored["stable_ids"], jnp.arange(16))
    np.testing.assert_array_equal(restored["rng_counters"], jnp.arange(16) + 1)
    np.testing.assert_allclose(restored["positions"], proposed)


def test_particle_migration_detects_duplicate_stable_ids_across_owners():
    _, state, distributed = _prepared(parts=2)
    sharded = distributed.initialize(state)
    particles = sharded.particles
    duplicated = type(particles)(
        particles.positions,
        particles.momenta,
        particles.masses,
        particles.stable_ids.at[8].set(particles.stable_ids[0]),
        particles.logical_slots,
        particles.active_mask,
        particles.rng_counters,
        particles.scale_factor,
        particles.owner,
        particles.runtime_id,
    )
    attempted = distributed.particle_runtime.migrate(
        duplicated, duplicated.positions, duplicated.momenta
    )

    assert not bool(attempted.evidence.successful)
    assert not bool(attempted.evidence.ids_unique)


def test_periodic_particle_ghost_exchange_is_bounded_and_owner_identified():
    _, state, distributed = _prepared(parts=2)
    sharded = distributed.initialize(state)
    ghosts = distributed.particle_runtime.exchange_ghosts(sharded.particles)

    assert bool(ghosts.successful)
    assert ghosts.positions.shape == (32, 2)
    assert ghosts.valid.shape == (32,)
    assert ghosts.left_count.shape == (2,)
    assert ghosts.right_count.shape == (2,)
    assert bool(jnp.all(ghosts.left_count <= ghosts.capacity_per_side))
    assert bool(jnp.all(ghosts.right_count <= ghosts.capacity_per_side))
    assert bool(jnp.all((ghosts.source_owner >= 0) & (ghosts.source_owner < 2)))
    valid = ghosts.valid.reshape((2, 2, ghosts.capacity_per_side))
    np.testing.assert_array_equal(ghosts.left_count, jnp.sum(valid[:, 0], axis=1))
    np.testing.assert_array_equal(ghosts.right_count, jnp.sum(valid[:, 1], axis=1))


def test_receive_capacity_failure_rolls_back_every_particle_field():
    _, state, distributed = _prepared(parts=2)
    sharded = distributed.initialize(state, rng_counters=jnp.arange(16, dtype=jnp.uint64))
    proposed = jnp.full_like(sharded.particles.positions, 0.25)
    attempted = distributed.particle_runtime.migrate(
        sharded.particles,
        proposed,
        sharded.particles.momenta + 1.0,
        proposed_rng_counters=sharded.particles.rng_counters + 1,
    )

    assert not bool(attempted.evidence.successful)
    for incoming, retained in zip(
        jax.tree.leaves(attempted.state),
        jax.tree.leaves(sharded.particles),
        strict=True,
    ):
        if isinstance(incoming, jax.Array):
            np.testing.assert_array_equal(incoming, retained)


def test_multi_part_deposit_has_unique_ownership_and_additive_mass():
    mixed, state, distributed = _prepared(parts=2)
    sharded = distributed.initialize(state)
    density = distributed.assemble_density(sharded)
    gravity = distributed.solve_gravity(sharded)
    local = mixed.density.assemble(state)
    local_gravity = mixed.gravity.solve(local)

    assert bool(density.successful)
    np.testing.assert_allclose(
        density.particle_density, local.particle_density, rtol=2e-12
    )
    np.testing.assert_allclose(density.particle_source_mass, 1.0, rtol=2e-12)
    np.testing.assert_allclose(density.particle_deposited_mass, 1.0, rtol=2e-12)
    np.testing.assert_allclose(gravity.potential, local_gravity.potential, rtol=2e-10)
    np.testing.assert_allclose(
        gravity.particle_acceleration,
        local_gravity.particle_acceleration,
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(gravity.total_force, local_gravity.total_force, atol=2e-11)
    assert gravity.potential.sharding == distributed.field_sharding
    assert (
        sharded.particles.positions.sharding
        == distributed.particle_runtime.vector_sharding
    )
    logical = distributed.particle_runtime.logical_arrays(sharded.particles)
    np.testing.assert_array_equal(jnp.sort(logical["stable_ids"]), jnp.arange(16))


def test_checkpoint_evidence_binds_complete_coverage_and_changed_sharding_restore():
    _, _, distributed = _prepared()
    checkpoint = distributed.evidence.checkpoint

    assert checkpoint.successful
    assert checkpoint.distributed_checkpoint_executable
    assert checkpoint.restart_identity_executable
    assert checkpoint.changed_sharding_restore_executable
    assert "particle_rng_counters" in checkpoint.covered_values
    assert "wave" in checkpoint.covered_values
    assert checkpoint.shard_plan_id
    assert (
        checkpoint.unpadded_bytes_per_checkpoint == distributed.checkpoint_unpadded_bytes
    )
    assert checkpoint.payload_bytes_per_checkpoint == distributed.checkpoint_payload_bytes
    assert checkpoint.payload_alignment == distributed.checkpoint_payload_alignment
    assert checkpoint.required_bytes == (
        checkpoint.payload_bytes_per_checkpoint * distributed.plan.checkpoint_count
    )
    collective = distributed.evidence.collective
    assert collective.particle_send_capacity == 16
    assert collective.particle_receive_capacity == 16
    assert collective.particle_ghost_capacity == 16
    assert collective.particle_exchange_bytes_per_device > 0
    assert not collective.host_gather_fallback


def test_checkpoint_resource_preflight_refuses_before_execution():
    _, _, distributed = _prepared()
    admitted = distributed.plan
    refused = DistributedMixedExecutionPlan(
        admitted.mixed,
        admitted.spectral,
        admitted.execution,
        admitted.particles,
        admitted.feasibility,
        maximum_checkpoint_bytes=admitted.required_checkpoint_bytes - 1,
        particle_send_capacity=admitted.particle_send_capacity,
        particle_receive_capacity=admitted.particle_receive_capacity,
        particle_ghost_capacity=admitted.particle_ghost_capacity,
        particle_ghost_width=admitted.particle_ghost_width,
    ).prepare()

    assert not bool(refused.successful)
    assert refused.executable is None
    assert refused.evidence.status == "checkpoint-incomplete"
    assert not refused.evidence.checkpoint.capacity_sufficient


def test_ghost_capacity_cannot_exceed_local_particle_capacity():
    _, _, distributed = _prepared()
    admitted = distributed.plan
    with pytest.raises(ValueError, match="exchange capacities"):
        DistributedMixedExecutionPlan(
            admitted.mixed,
            admitted.spectral,
            admitted.execution,
            admitted.particles,
            admitted.feasibility,
            maximum_checkpoint_bytes=2**28,
            particle_ghost_capacity=admitted.particles.capacity_per_device + 1,
        )


def test_checkpoint_payload_padding_is_mesh_divisible_and_schema_unpadded():
    _, state, distributed = _prepared(parts=3, count=6)
    payload = distributed.checkpoint_tree(distributed.initialize(state))["payload"]
    expected_padding = (
        distributed.checkpoint_payload_bytes - distributed.checkpoint_unpadded_bytes
    )

    assert payload.size == distributed.checkpoint_unpadded_bytes + expected_padding
    assert payload.size % 3 == 0
    assert expected_padding > 0
    assert distributed.checkpoint_payload_alignment == 6
    assert payload.size % distributed.checkpoint_payload_alignment == 0
    np.testing.assert_array_equal(
        payload[distributed.checkpoint_unpadded_bytes :],
        jnp.zeros((expected_padding,), dtype=jnp.uint8),
    )


@pytest.mark.parametrize(
    ("source_parts", "destination_parts", "count"),
    ((2, 1, 4), (2, 3, 6), (3, 2, 6)),
)
def test_checkpoint_restore_changes_particle_and_field_sharding(
    tmp_path: Path, source_parts: int, destination_parts: int, count: int
):
    _, state, source = _prepared(parts=source_parts, count=count)
    _, _, destination = _prepared(parts=destination_parts, count=count)
    source_state = source.initialize(
        state, rng_counters=jnp.arange(count**2, dtype=jnp.uint64)
    )
    profile = HPCFilesystemProfile(
        "posix.distributed-mixed-reshard",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    repository = POSIXArtifactRepository(
        tmp_path / "reshard",
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=2**20,
            maximum_metadata_bytes=2**20,
        ),
    )
    publish_process_checkpoint(
        repository,
        "distributed-mixed-reshard",
        source.checkpoint_execution_id,
        source.checkpoint_tree(source_state),
        writer_id="distributed-mixed-reshard-writer",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "distributed-mixed-reshard",
        source.checkpoint_schema_id,
        source.checkpoint_numeric_id,
        source.checkpoint_execution_id,
        expected_process_count=jax.process_count(),
        diagnostic_ids=(source.checkpoint_physics_id,),
    )
    destination_prototype = destination.initialize(state)
    restored = destination.restore_checkpoint(repository, manifest, destination_prototype)

    np.testing.assert_allclose(restored.wave.psi, source_state.wave.psi)
    source_logical = source.particle_runtime.logical_arrays(source_state.particles)
    restored_logical = destination.particle_runtime.logical_arrays(restored.particles)
    np.testing.assert_allclose(restored_logical["positions"], source_logical["positions"])
    np.testing.assert_array_equal(
        restored_logical["rng_counters"], source_logical["rng_counters"]
    )
    assert restored.wave.psi.sharding == destination.field_sharding
    assert (
        restored.particles.positions.sharding
        == destination.particle_runtime.vector_sharding
    )
