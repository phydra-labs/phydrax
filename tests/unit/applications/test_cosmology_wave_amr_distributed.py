from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import phydrax.solver._distributed_wave_amr as distributed_wave
from phydrax.applications.cosmology._wave_amr import (
    WaveAMRAdaptivityPlan,
    WaveAMRDiscretizationPlan,
    WaveAMRPhysicsPlan,
)
from phydrax.lifecycle._distributed_checkpoint import ProcessCheckpointPublication
from phydrax.lifecycle._models import CheckpointManifest, CheckpointShard
from phydrax.solver._distributed_wave_amr import (
    DistributedWaveAMRState,
    PreparedDistributedWaveAMRTopologyTransition,
)


def _prepared(*, cells: int = 8, adaptive: bool = False):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    base_blocks = cells // 4
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(
                0,
                (4,),
                base_blocks,
                halo_width=1,
                refinement_ratio=2,
            ),
            phx.discretization.BlockLevelPlan(
                1,
                (4,),
                max(4, 2 * base_blocks),
                halo_width=1,
            ),
        ),
    )
    fd = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()
    initial = fd.initial_topology()
    tags = jnp.zeros((base_blocks, 4), dtype="bool").at[base_blocks // 2, 1:3].set(True)
    compilation = fd.compile_topology(initial, (tags,))
    assert compilation.status.successful
    topology = compilation.topology
    prepared = WaveAMRDiscretizationPlan(
        fd,
        adaptivity=(
            WaveAMRAdaptivityPlan(
                maximum_phase_change=0.01,
                current_relative_tolerance=0.5,
                phase_defect_tolerance=1.0,
            )
            if adaptive
            else None
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
    lower = float(np.asarray(grid.structured_axes[0].bounds[0]))
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
            coordinate = (
                lower
                + (origin + jnp.arange(level_plan.block_shape[0]) + 0.5) * spacing[0]
            )
            profile = 0.3 + jnp.exp(-(((coordinate - 0.45) / 0.18) ** 2))
            level = level.at[slot].set(profile * jnp.exp(2j * jnp.pi * coordinate))
        values.append(level)
    return hierarchy, prepared, prepared.initialize(tuple(values), 1.0)


def test_single_part_distributed_entry_is_exact_local_authority():
    hierarchy, prepared, state = _prepared()
    distributed = prepared.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, 1),
        maximum_bytes=10_000_000,
    )

    authority = prepared.step(state, 1.00001)
    result = prepared.distributed_step(distributed, state, 1.00001)

    assert distributed.admitted and distributed.executable
    assert distributed.execution is not None
    assert distributed.operator_id == distributed.execution.operator_id
    assert distributed.physics_id == prepared.physics.plan_id
    assert result.prepared_id == authority.prepared_id
    np.testing.assert_array_equal(result.successful, authority.successful)
    for actual, expected in zip(
        result.state.psi.levels, authority.state.psi.levels, strict=True
    ):
        np.testing.assert_array_equal(actual.values, expected.values)


def test_two_part_metadata_prepares_exact_composite_and_fillpatch_routes_without_devices():
    hierarchy, prepared, _ = _prepared()
    distributed = prepared.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, 2),
        maximum_bytes=10_000_000,
    )

    assert distributed.admitted
    assert not distributed.executable
    assert distributed.execution is not None
    assert distributed.hierarchy is not None
    assert distributed.execution.route_crossing_count > 0
    assert distributed.execution.routes.halo.permutations
    assert sum(distributed.hierarchy.resources.same_level_routes) > 0
    assert sum(distributed.hierarchy.resources.interface_routes) > 0
    assert distributed.required_bytes == distributed.execution.required_bytes
    assert distributed.required_bytes <= distributed.maximum_bytes
    assert distributed.mesh_id is None
    assert "ExecutionGroup mesh" in distributed.reason


def test_resource_preflight_rejects_before_any_route_or_operator_allocation(monkeypatch):
    hierarchy, prepared, _ = _prepared()
    partition = phx.discretization.BlockAMRPartitionPlan(hierarchy, 2)

    def forbidden_prepare(*args, **kwargs):
        del args, kwargs
        raise AssertionError("route construction happened before resource admission")

    monkeypatch.setattr(type(partition), "prepare", forbidden_prepare)
    rejected = prepared.prepare_distributed(partition, maximum_bytes=0)

    assert not rejected.admitted
    assert not rejected.executable
    assert rejected.hierarchy is None
    assert rejected.execution is None
    assert rejected.required_bytes == rejected.preflight_required_bytes
    assert rejected.required_bytes > rejected.maximum_bytes


def test_accepted_boundary_repartition_uses_stable_block_identity_routes():
    hierarchy, prepared, state = _prepared(cells=16)
    partition = phx.discretization.BlockAMRPartitionPlan(hierarchy, 2)
    source = prepared.prepare_distributed(partition, maximum_bytes=20_000_000)
    target = prepared.prepare_distributed(
        partition,
        maximum_bytes=20_000_000,
        costs=(jnp.asarray([100.0, 1.0, 1.0, 1.0]), None),
    )
    assert source.execution is not None and target.execution is not None
    assert source.hierarchy is not None and target.hierarchy is not None
    migration = source.hierarchy.migration_to(target.hierarchy)
    assert sum(migration.moved_block_counts) > 0
    packed = source.execution.bind_packed_state(
        source.execution.pack_canonical_values(prepared.layout.bind_state(state.psi)),
        state.scale_factor,
    )

    migrated = prepared.migrate_distributed_state(source, packed, target)

    assert isinstance(migrated, DistributedWaveAMRState)
    assert migrated.execution_id == target.execution.execution_id
    source_values = source.hierarchy.unpack(packed.psi)
    target_values = target.hierarchy.unpack(migrated.psi)
    for expected, actual in zip(source_values.levels, target_values.levels, strict=True):
        np.testing.assert_allclose(actual.values, expected.values)

    rejected = DistributedWaveAMRState(
        packed.psi,
        packed.scale_factor,
        jnp.asarray(False),
        packed.execution_id,
        packed.topology_id,
        packed.partition_id,
        packed.operator_id,
        packed.physics_id,
    )
    with pytest.raises(ValueError, match="accepted boundary"):
        prepared.migrate_distributed_state(source, rejected, target)


def test_checkpoint_contract_binds_every_continuation_array_and_partition_identity():
    hierarchy, prepared, _ = _prepared()
    distributed = prepared.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, 2),
        maximum_bytes=10_000_000,
    )
    assert distributed.execution is not None

    evidence = distributed.execution.checkpoint_evidence("wave-amr-checkpoint")

    assert evidence.exact_coverage_required
    assert evidence.execution_id == distributed.execution.execution_id
    assert evidence.topology_id == prepared.topology.topology_id
    assert evidence.partition_id == distributed.partition_plan_id
    assert evidence.array_paths == (
        "['psi'][0]",
        "['psi'][1]",
        "['scale_factor']",
        "['accepted_boundary']",
    )


def test_topology_successor_route_algebra_is_available_without_two_devices():
    hierarchy, prepared, state = _prepared(cells=16, adaptive=True)
    proposal = prepared.propose_topology(state)
    assert proposal.successful and proposal.compilation.status.changed
    source = prepared.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, 2),
        maximum_bytes=20_000_000,
    )
    successor = prepared.plan.prepare(
        prepared.physics,
        proposal.compilation.topology,
        prepared.background,
    )
    target = successor.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, 2),
        maximum_bytes=20_000_000,
    )
    assert source.execution is not None and target.execution is not None
    density_transition = prepared.fd_hierarchy.field_transition(
        prepared.topology,
        successor.topology,
        "wave-probability-density",
        dtype=prepared.real_dtype,
    )

    owner = PreparedDistributedWaveAMRTopologyTransition(
        source.execution,
        target.execution,
        density_transition,
    )
    estimated_bytes = (
        PreparedDistributedWaveAMRTopologyTransition.estimate_required_bytes(
            source.execution,
            target.execution,
            density_transition,
        )
    )

    assert owner.transition_id
    assert owner.required_bytes > 0
    assert owner.required_bytes == estimated_bytes
    assert owner.halo.entity_count == (
        source.execution.routes.halo.entity_count
        + target.execution.routes.halo.entity_count
    )
    assert any(owner.common_stable_block_ids)
    assert owner.overlap_capacity >= 1
    assert owner.halo.permutations


def test_packed_state_validation_rejects_invalid_physics_and_storage():
    hierarchy, prepared, state = _prepared()
    distributed = prepared.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, 2),
        maximum_bytes=10_000_000,
    )
    execution = distributed.execution
    assert execution is not None
    packed = execution.pack_canonical_values(prepared.layout.bind_state(state.psi))
    bound = execution.bind_packed_state(packed, state.scale_factor)
    assert isinstance(bound, DistributedWaveAMRState)

    with pytest.raises(ValueError, match="positive in scale/global probability"):
        execution.bind_packed_state(
            tuple(jnp.zeros_like(value) for value in packed),
            state.scale_factor,
        )
    with pytest.raises(ValueError, match="positive in scale/global probability"):
        execution.bind_packed_state(packed, 0.0)
    nonfinite = list(packed)
    nonfinite[0] = nonfinite[0].at[0, 0, 0].set(jnp.nan + 0j)
    with pytest.raises(ValueError, match="must be finite"):
        execution.bind_packed_state(tuple(nonfinite), state.scale_factor)

    unmasked = list(packed)
    found = False
    for level, mask in enumerate(execution.routes.level_leaf_masks):
        indices = np.argwhere(~np.asarray(mask))
        if indices.size:
            unmasked[level] = unmasked[level].at[tuple(indices[0])].set(1.0 + 0j)
            found = True
            break
    assert found
    with pytest.raises(ValueError, match="inactive or covered"):
        execution.bind_packed_state(tuple(unmasked), state.scale_factor)


def test_process_checkpoint_accepts_local_path_subset_but_manifest_requires_inventory(
    monkeypatch,
):
    hierarchy, prepared, state = _prepared()
    distributed = prepared.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, 2),
        maximum_bytes=10_000_000,
    )
    execution = distributed.execution
    assert execution is not None
    packed = execution.bind_packed_state(
        execution.pack_canonical_values(prepared.layout.bind_state(state.psi)),
        state.scale_factor,
    )
    shard = CheckpointShard(
        "local-wave-shard",
        "0" * 64,
        1,
        ("local-wave-layout",),
        metadata={"array_path": "['psi'][0]"},
    )
    publication = ProcessCheckpointPublication(0, None, (shard,))
    monkeypatch.setattr(
        distributed_wave,
        "publish_process_checkpoint",
        lambda *args, **kwargs: publication,
    )

    observed, _ = execution.publish_checkpoint(
        None,
        "local-process-checkpoint",
        packed,
        writer_id="writer",
    )
    assert observed is publication

    incomplete = CheckpointManifest(
        "local-process-checkpoint",
        prepared.prepared_id,
        prepared.physics.plan_id,
        execution.execution_id,
        (shard,),
        complete=True,
    )
    with pytest.raises(ValueError, match="inventory is incomplete"):
        execution._validate_checkpoint_manifest(incomplete, execution)


def test_single_part_topology_transition_matches_local_authority_at_wave_node():
    hierarchy, prepared, state = _prepared(cells=16, adaptive=True)
    values = list(prepared.layout.bind_state(state.psi))
    for level, mask in enumerate(prepared.layout.leaf_mask):
        indices = np.argwhere(np.asarray(mask))
        if indices.size:
            values[level] = values[level].at[tuple(indices[0])].set(0.0 + 0.0j)
            break
    nodal_state = prepared.initialize(tuple(values), state.scale_factor)
    proposal = prepared.propose_topology(nodal_state)
    assert proposal.successful and proposal.compilation.status.changed
    successor = prepared.plan.prepare(
        prepared.physics,
        proposal.compilation.topology,
        prepared.background,
    )
    partition = phx.discretization.BlockAMRPartitionPlan(hierarchy, 1)
    source = prepared.prepare_distributed(
        partition,
        maximum_bytes=20_000_000,
    )
    target = successor.prepare_distributed(
        partition,
        maximum_bytes=20_000_000,
    )
    packed = source.execution.bind_packed_state(
        source.execution.pack_canonical_values(
            prepared.layout.bind_state(nodal_state.psi)
        ),
        nodal_state.scale_factor,
    )
    authority = prepared.transition(nodal_state, proposal)

    result = prepared.transition_distributed(
        source,
        packed,
        proposal,
        target,
    )

    assert bool(result.successful) == bool(authority.successful)
    np.testing.assert_allclose(
        result.evidence.probability_relative_defect,
        authority.evidence.probability_relative_defect,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.evidence.phase_defect,
        authority.evidence.phase_defect,
        rtol=0.0,
        atol=0.0,
    )
    candidate = target.hierarchy.unpack(result.candidate_state.psi)
    for actual, expected in zip(
        candidate.levels,
        authority.candidate_state.psi.levels,
        strict=True,
    ):
        np.testing.assert_array_equal(actual.values, expected.values)
    assert result.candidate_state.execution_id == target.execution.execution_id
    assert result.candidate_prepared.prepared_id == successor.prepared_id

    rejected = prepared.transition_distributed(
        source,
        packed,
        proposal,
        target,
        maximum_bytes=0,
    )
    assert not bool(rejected.successful)
    assert rejected.candidate_state.execution_id == source.execution.execution_id
    assert rejected.candidate_prepared.prepared_id == prepared.prepared_id
    assert (
        rejected.candidate_distributed_preparation.preparation_id == source.preparation_id
    )
