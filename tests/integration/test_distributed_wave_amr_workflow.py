from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec

import phydrax as phx
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax.applications.cosmology._wave_amr import (
    WaveAMRAdaptivityPlan,
    WaveAMRDiscretizationPlan,
    WaveAMRPhysicsPlan,
)
from phydrax.lifecycle import assemble_distributed_checkpoint_from_repository
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.solver._distributed_wave_amr import (
    DistributedWaveAMRResult,
    PreparedDistributedWaveAMRTopologyTransition,
)


def _execution_group(devices):
    values = tuple(devices)
    return ExecutionGroup(
        ExecutionGroupSpec(
            "distributed-wave-amr-integration",
            tuple(sorted({device.process_index for device in values})),
            tuple((device.process_index, device.id) for device in values),
            mesh_axes=(("block_parts", len(values)),),
        ),
        values,
    )


def _problem(*, refinement_ratio: int = 2, coarsen: bool = False):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(
                0,
                (4,),
                4,
                halo_width=1,
                refinement_ratio=refinement_ratio,
            ),
            phx.discretization.BlockLevelPlan(
                1,
                (4,),
                max(8, 4 * refinement_ratio),
                halo_width=1,
            ),
        ),
    )
    fd = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()
    initial = fd.initial_topology()
    tags = jnp.zeros((4, 4), dtype=bool).at[2, 1:3].set(True)
    compilation = fd.compile_topology(initial, (tags,))
    assert compilation.status.successful
    topology = compilation.topology
    prepared = WaveAMRDiscretizationPlan(
        fd,
        adaptivity=(
            WaveAMRAdaptivityPlan(
                maximum_phase_change=3.0,
                maximum_density_contrast=100.0,
                maximum_quantum_potential_indicator=100.0,
                current_relative_tolerance=0.5,
                phase_defect_tolerance=1.0,
            )
            if coarsen
            else WaveAMRAdaptivityPlan(
                maximum_phase_change=0.01,
                current_relative_tolerance=0.5,
                phase_defect_tolerance=1.0,
            )
        ),
        norm_relative_tolerance=2.0e-8,
        maximum_phase_radians=2.0,
    ).prepare(
        WaveAMRPhysicsPlan(
            1.0,
            gravitational_constant=0.02,
            reduced_planck_constant=0.05,
        ),
        topology,
        phx.applications.cosmology.FLRWBackground(1.0, 1.0),
    )
    values = []
    for level_plan, metadata, spacing in zip(
        topology.plan.levels,
        topology.levels,
        topology.plan.level_spacings,
        strict=True,
    ):
        level = jnp.zeros(
            (level_plan.maximum_blocks, *level_plan.block_shape),
            dtype=jnp.complex128,
        )
        logical = np.asarray(metadata.logical_indices)
        for slot in np.flatnonzero(np.asarray(metadata.active)):
            origin = logical[slot, 0] * level_plan.block_shape[0]
            coordinate = (origin + jnp.arange(4) + 0.5) * spacing[0]
            amplitude = 0.3 + jnp.exp(-(((coordinate - 0.45) / 0.18) ** 2))
            level = level.at[slot].set(amplitude * jnp.exp(2j * jnp.pi * coordinate))
        values.append(level)
    return hierarchy, prepared, prepared.initialize(tuple(values), 1.0)


def _repository(tmp_path):
    profile = HPCFilesystemProfile(
        "posix.distributed-wave-amr",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXArtifactRepository(
        tmp_path / "repository",
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=256,
            maximum_metadata_bytes=64 * 1024,
        ),
    )


@pytest.mark.skipif(len(jax.devices()) < 2, reason="requires at least two JAX devices")
def test_two_part_wave_amr_step_checkpoint_and_changed_partition_restart(
    tmp_path,
    monkeypatch,
):
    hierarchy, prepared, state = _problem()
    group = _execution_group(jax.devices()[:2])
    partition = phx.discretization.BlockAMRPartitionPlan(hierarchy, 2)
    source = prepared.prepare_distributed(
        partition,
        maximum_bytes=100_000_000,
        execution_group=group,
    )
    assert source.executable and source.execution is not None
    assert source.hierarchy is not None
    assert source.execution.route_crossing_count > 0
    with pytest.raises(ValueError, match="finite and greater"):
        prepared.distributed_step(source, state, jnp.nan)
    with pytest.raises(ValueError, match="finite and greater"):
        prepared.distributed_step(source, state, state.scale_factor)

    authority = prepared.step(state, 1.00001)
    result = prepared.distributed_step(source, state, 1.00001)
    jax.block_until_ready(result.successful)

    assert isinstance(result, DistributedWaveAMRResult)
    assert bool(authority.successful) and bool(result.successful)
    assert result.state.scale_factor.sharding.spec == PartitionSpec()
    assert result.state.accepted_boundary.sharding.spec == PartitionSpec()
    assert bool(result.diagnostics.halo_complete)
    assert bool(result.diagnostics.rank_agreement)
    assert bool(result.diagnostics.poisson_closed)
    assert bool(result.diagnostics.kinetic_closed)
    assert float(result.diagnostics.probability_relative_error) < 2.0e-8
    assert float(result.diagnostics.self_adjoint_residual) < 1.0e-10
    assert abs(float(result.gravity.source_integral)) < 1.0e-12
    assert float(result.gravity.gauge_defect) < 1.0e-12
    assert float(result.gravity.interface_flux_conservation_defect) == 0.0
    assert int(result.kinetic_solve.iterations) <= prepared.plan.maximum_solve_steps
    restored_authority = source.hierarchy.unpack(result.state.psi)
    for actual, expected in zip(
        restored_authority.levels, authority.state.psi.levels, strict=True
    ):
        np.testing.assert_allclose(
            actual.values, expected.values, rtol=2.0e-8, atol=2.0e-10
        )
    rejected = prepared.distributed_step(source, state, 1.1)
    jax.block_until_ready(rejected.successful)
    assert not bool(rejected.successful)
    assert bool(rejected.state.accepted_boundary) == bool(state.accepted_boundary)
    initial_packed = source.execution.pack_canonical_values(
        prepared.layout.bind_state(state.psi)
    )
    for actual, expected in zip(rejected.state.psi, initial_packed, strict=True):
        np.testing.assert_array_equal(actual, expected)

    repository = _repository(tmp_path)
    publication, checkpoint = source.execution.publish_checkpoint(
        repository,
        "distributed-wave-amr-checkpoint",
        result.state,
        writer_id="wave-amr-writer",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        checkpoint.checkpoint_id,
        prepared.prepared_id,
        prepared.physics.plan_id,
        source.execution.execution_id,
        expected_process_count=jax.process_count(),
        diagnostic_ids=(checkpoint.evidence_id,),
    )
    assert publication.shards and manifest.complete
    foreign = prepared.plan.prepare(
        WaveAMRPhysicsPlan(
            1.1,
            gravitational_constant=prepared.physics.gravitational_constant,
            reduced_planck_constant=prepared.physics.reduced_planck_constant,
        ),
        prepared.topology,
        prepared.background,
    )
    foreign_distribution = foreign.prepare_distributed(
        partition,
        maximum_bytes=100_000_000,
        execution_group=group,
    )
    with pytest.raises(ValueError, match="change only distributed partition"):
        foreign_distribution.execution.restore_checkpoint(
            repository,
            manifest,
            source.execution,
        )

    target = prepared.prepare_distributed(
        partition,
        maximum_bytes=100_000_000,
        execution_group=group,
        costs=(jnp.asarray([100.0, 1.0, 1.0, 1.0]), None),
    )
    assert target.execution is not None and target.hierarchy is not None
    migration = source.hierarchy.migration_to(target.hierarchy)
    assert sum(migration.moved_block_counts) > 0
    expected = source.execution.migrate_accepted_state(result.state, target.execution)
    restarted, restart = target.execution.restore_checkpoint(
        repository, manifest, source.execution
    )

    assert restart.exact_coverage
    assert restart.changed_partition
    assert restart.migration_id == migration.migration_id
    assert restarted.execution_id == target.execution.execution_id
    for actual, expected_level in zip(restarted.psi, expected.psi, strict=True):
        np.testing.assert_array_equal(actual, expected_level)

    proposal = prepared.propose_topology(state)
    assert proposal.successful and proposal.compilation.status.changed
    successor = prepared.plan.prepare(
        prepared.physics,
        proposal.compilation.topology,
        prepared.background,
    )
    successor_distribution = successor.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, 2),
        maximum_bytes=100_000_000,
        execution_group=group,
    )
    topology_state = source.execution.bind_packed_state(
        initial_packed,
        state.scale_factor,
    )
    authority_transition = prepared.transition(state, proposal)
    transitioned = prepared.transition_distributed(
        source,
        topology_state,
        proposal,
        successor_distribution,
    )
    jax.block_until_ready(transitioned.successful)
    assert bool(transitioned.successful)
    assert transitioned.state.topology_id == successor.topology.topology_id
    assert bool(transitioned.evidence.probability_preserved)
    assert bool(transitioned.evidence.current_preserved)
    assert bool(transitioned.evidence.winding_preserved)
    assert transitioned.transfer.route_complete
    distributed_candidate = successor_distribution.hierarchy.unpack(
        transitioned.candidate_state.psi
    )
    for actual, expected_level in zip(
        distributed_candidate.levels,
        authority_transition.candidate_state.psi.levels,
        strict=True,
    ):
        np.testing.assert_allclose(
            actual.values,
            expected_level.values,
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def forbidden_transition_routes(*args, **kwargs):
        del args, kwargs
        raise AssertionError("topology routes constructed before resource preflight")

    with monkeypatch.context() as guard:
        guard.setattr(
            PreparedDistributedWaveAMRTopologyTransition,
            "__init__",
            forbidden_transition_routes,
        )
        rolled_back = prepared.transition_distributed(
            source,
            topology_state,
            proposal,
            successor_distribution,
            maximum_bytes=0,
        )
    assert not bool(rolled_back.successful)
    assert bool(rolled_back.rolled_back)
    assert rolled_back.prepared.prepared_id == prepared.prepared_id
    for actual, expected_level in zip(
        rolled_back.state.psi, topology_state.psi, strict=True
    ):
        np.testing.assert_array_equal(actual, expected_level)

    successor_repartition = successor.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, 2),
        maximum_bytes=100_000_000,
        execution_group=group,
        costs=(jnp.asarray([100.0, 1.0, 1.0, 1.0]), None),
    )
    migrated_successor = successor.migrate_distributed_state(
        successor_distribution,
        transitioned.state,
        successor_repartition,
    )
    assert migrated_successor.execution_id == (
        successor_repartition.execution.execution_id
    )

    successor_publication, successor_checkpoint = (
        successor_repartition.execution.publish_checkpoint(
            repository,
            "distributed-wave-amr-successor-checkpoint",
            migrated_successor,
            writer_id="wave-amr-successor-writer",
        )
    )
    successor_manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        successor_checkpoint.checkpoint_id,
        successor.prepared_id,
        successor.physics.plan_id,
        successor_repartition.execution.execution_id,
        expected_process_count=jax.process_count(),
        diagnostic_ids=(
            successor_checkpoint.evidence_id,
            transitioned.evidence.transition_id,
        ),
    )
    assert successor_publication.shards and successor_manifest.complete
    restored_successor, successor_restart = (
        successor_distribution.execution.restore_checkpoint(
            repository,
            successor_manifest,
            successor_repartition.execution,
        )
    )
    assert successor_restart.changed_partition
    assert successor_restart.exact_coverage
    for actual, expected_level in zip(
        restored_successor.psi,
        transitioned.state.psi,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected_level)


@pytest.mark.skipif(len(jax.devices()) < 2, reason="requires at least two JAX devices")
def test_high_ratio_topology_preflight_rejects_before_padded_route_allocation(
    monkeypatch,
):
    hierarchy, prepared, state = _problem(refinement_ratio=4, coarsen=True)
    group = _execution_group(jax.devices()[:2])
    partition = phx.discretization.BlockAMRPartitionPlan(hierarchy, 2)
    source = prepared.prepare_distributed(
        partition,
        maximum_bytes=100_000_000,
        execution_group=group,
    )
    zero_tags = (
        jnp.zeros(
            (
                hierarchy.levels[0].maximum_blocks,
                *hierarchy.levels[0].block_shape,
            ),
            dtype=bool,
        ),
    )
    compilation = prepared.fd_hierarchy.compile_topology(
        prepared.topology,
        zero_tags,
    )
    assert compilation.status.successful and compilation.status.changed
    indicator_proposal = prepared.propose_topology(state)
    proposal = type(indicator_proposal)(
        zero_tags,
        indicator_proposal.indicators,
        compilation,
        prepared.prepared_id,
        prepared.topology.epoch.epoch_id,
        prepared.topology.topology_id,
        True,
        True,
        "high-ratio-coarsening-proposal",
    )
    successor = prepared.plan.prepare(
        prepared.physics,
        proposal.compilation.topology,
        prepared.background,
    )
    target = successor.prepare_distributed(
        partition,
        maximum_bytes=100_000_000,
        execution_group=group,
    )
    density_transition = prepared.fd_hierarchy.field_transition(
        prepared.topology,
        successor.topology,
        "wave-probability-density",
        dtype=prepared.real_dtype,
    )
    estimate = PreparedDistributedWaveAMRTopologyTransition.estimate_required_bytes(
        source.execution,
        target.execution,
        density_transition,
    )
    target_owners = np.asarray(target.execution.routes.halo.entity_owner)
    local_target_capacity = int(np.max(np.bincount(target_owners, minlength=2)))
    padded_overlap_entries = (
        2
        * local_target_capacity
        * density_transition.leaf_routes.maximum_target_row_width
    )
    assert padded_overlap_entries > int(
        density_transition.leaf_routes.relation.source_indices.size
    )
    insufficient_budget = source.required_bytes + target.required_bytes + estimate - 1
    packed = source.execution.bind_packed_state(
        source.execution.pack_canonical_values(prepared.layout.bind_state(state.psi)),
        state.scale_factor,
    )

    def forbidden_owner(*args, **kwargs):
        del args, kwargs
        raise AssertionError("padded topology packets allocated before preflight")

    with monkeypatch.context() as guard:
        guard.setattr(
            PreparedDistributedWaveAMRTopologyTransition,
            "__init__",
            forbidden_owner,
        )
        rejected = prepared.transition_distributed(
            source,
            packed,
            proposal,
            target,
            maximum_bytes=insufficient_budget,
        )
    assert not bool(rejected.successful)
    assert rejected.candidate_state.execution_id == source.execution.execution_id
    assert rejected.candidate_prepared.prepared_id == prepared.prepared_id
    assert (
        rejected.candidate_distributed_preparation.preparation_id == source.preparation_id
    )
    for actual, expected in zip(rejected.candidate_state.psi, packed.psi, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    jax.process_count() < 2,
    reason="requires a real multi-process JAX runtime",
)
def test_multiprocess_checkpoint_local_publications_form_exact_global_inventory(
    tmp_path,
):
    hierarchy, prepared, state = _problem()
    devices_by_process = {}
    for device in sorted(
        jax.devices(), key=lambda value: (value.process_index, value.id)
    ):
        devices_by_process.setdefault(device.process_index, device)
    devices = tuple(devices_by_process[index] for index in range(jax.process_count()))
    group = _execution_group(devices)
    distribution = prepared.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, len(devices)),
        maximum_bytes=100_000_000,
        execution_group=group,
    )
    execution = distribution.execution
    assert distribution.executable and execution is not None
    packed = execution.bind_packed_state(
        execution.pack_canonical_values(prepared.layout.bind_state(state.psi)),
        state.scale_factor,
    )
    repository = _repository(tmp_path)
    publication, checkpoint = execution.publish_checkpoint(
        repository,
        "multiprocess-wave-amr-checkpoint",
        packed,
        writer_id=f"wave-process-{jax.process_index()}",
    )
    local_paths = {dict(shard.metadata)["array_path"] for shard in publication.shards}
    assert local_paths.issubset(set(checkpoint.array_paths))
    multihost_utils.sync_global_devices("wave-amr-checkpoint-published")
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        checkpoint.checkpoint_id,
        prepared.prepared_id,
        prepared.physics.plan_id,
        execution.execution_id,
        expected_process_count=jax.process_count(),
        diagnostic_ids=(checkpoint.evidence_id,),
    )
    execution._validate_checkpoint_manifest(manifest, execution)
    assert {dict(shard.metadata)["array_path"] for shard in manifest.shards} == set(
        checkpoint.array_paths
    )
