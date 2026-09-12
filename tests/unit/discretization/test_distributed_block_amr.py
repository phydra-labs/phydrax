#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax.discretization.amr._distributed import (
    BlockAMRPartitionPlan,
    BlockAMRStableIDMigrationPlan,
)


def _hierarchy(*, cells=8, periodic=True, fine_capacity=8):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    return phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(0, (4,), cells // 4, halo_width=1),
            phx.discretization.BlockLevelPlan(1, (2,), fine_capacity, halo_width=1),
        ),
    )


def _compiled(hierarchy, *, coarse_slot=1, coarse_cell=2):
    compiler = phx.discretization.BlockTopologyCompiler(hierarchy)
    initial = compiler.initialize()
    tags = (
        jnp.zeros(
            (
                hierarchy.levels[0].maximum_blocks,
                *hierarchy.levels[0].block_shape,
            ),
            dtype=bool,
        )
        .at[coarse_slot, coarse_cell]
        .set(True)
    )
    return compiler, compiler.compile(initial.topology, (tags,))


def _state(topology, fd_hierarchy, *, inactive_nonfinite=True):
    dtype = fd_hierarchy.plan.precision.field_dtype
    levels = []
    for level, (plan, metadata) in enumerate(
        zip(topology.plan.levels, topology.levels, strict=True)
    ):
        count = plan.maximum_blocks * prod(plan.block_shape)
        values = jnp.arange(count, dtype=dtype).reshape(
            (plan.maximum_blocks, *plan.block_shape)
        )
        values = values + jnp.asarray(10 * level, dtype=dtype)
        if inactive_nonfinite:
            active = metadata.active.reshape(
                (plan.maximum_blocks,) + (1,) * len(plan.block_shape)
            )
            values = jnp.where(active, values, jnp.asarray(jnp.nan, dtype=dtype))
        levels.append(phx.discretization.BlockLevelState(plan, metadata, values))
    return phx.discretization.BlockHierarchyState(topology, tuple(levels))


def _prepare(hierarchy, compiled, part_count, *, costs=None, group=None):
    fd = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()
    distributed = BlockAMRPartitionPlan(hierarchy, part_count).prepare(
        compiled,
        fd,
        costs=costs,
        execution_group=group,
    )
    return fd, distributed


def _execution_group(devices, axis_name="block_parts"):
    devices = tuple(devices)
    processes = tuple(sorted({device.process_index for device in devices}))
    specification = ExecutionGroupSpec(
        "test-distributed-block-amr",
        processes,
        tuple((device.process_index, device.id) for device in devices),
        mesh_axes=((axis_name, len(devices)),),
    )
    return ExecutionGroup(specification, devices)


def test_partition_ownership_is_local_and_independent_of_compilation_history():
    hierarchy = _hierarchy()
    compiler, direct = _compiled(hierarchy)
    initial = compiler.initialize().topology
    other_tags = jnp.zeros((2, 4), dtype=bool).at[0, 0].set(True)
    other = compiler.compile(initial, (other_tags,)).topology
    target_tags = jnp.zeros((2, 4), dtype=bool).at[1, 2].set(True)
    by_other_history = compiler.compile(other, (target_tags,))

    _, first = _prepare(hierarchy, direct, 3)
    _, second = _prepare(hierarchy, by_other_history, 3)

    for level, (first_layout, second_layout) in enumerate(
        zip(first.layouts, second.layouts, strict=True)
    ):
        np.testing.assert_array_equal(first_layout.block_owner, second_layout.block_owner)
        assert first_layout.layout_id == second_layout.layout_id
        active_owners = np.asarray(first_layout.block_owner)[
            np.asarray(first.topology.levels[level].active)
        ]
        assert np.all(active_owners[:-1] <= active_owners[1:])


def test_canonical_pack_unpack_masks_inactive_nonfinite_payloads_and_allows_empty_parts():
    hierarchy = _hierarchy()
    _, compiled = _compiled(hierarchy)
    fd, prepared = _prepare(hierarchy, compiled, 3)
    state = _state(compiled.topology, fd)

    packed = prepared.pack(state)
    restored = prepared.unpack(packed)

    for level, (layout, packed_level, restored_level) in enumerate(
        zip(prepared.layouts, packed, restored.levels, strict=True)
    ):
        assert packed_level.shape[:2] == (
            3,
            layout.local_block_capacity,
        )
        assert bool(jnp.all(jnp.isfinite(packed_level)))
        active = np.asarray(compiled.topology.levels[level].active)
        np.testing.assert_allclose(
            restored_level.values[active], state.levels[level].values[active]
        )
        np.testing.assert_array_equal(restored_level.values[~active], 0.0)

    fine_valid_by_part = np.sum(np.asarray(prepared.layouts[1].local_block_valid), axis=1)
    assert np.count_nonzero(fine_valid_by_part == 0) == 2


def test_route_phases_are_symmetric_include_zero_payload_and_have_exact_reverse():
    hierarchy = _hierarchy()
    _, compiled = _compiled(hierarchy)
    fd, prepared = _prepare(hierarchy, compiled, 3)
    state = _state(compiled.topology, fd)

    for routes in (
        prepared.same_level_routes,
        prepared.coarse_fine_routes,
        prepared.interface_routes,
    ):
        for route in routes:
            assert route.phase_count >= 1
            assert route.send_local_indices.shape[:2] == (
                prepared.partition.part_count,
                route.phase_count,
            )
            assert route.receive_local_indices.shape == route.send_local_indices.shape
            for phase, reverse in zip(
                route.permutations, route.reverse_permutations, strict=True
            ):
                assert len({source for source, _ in phase}) == len(phase)
                assert len({target for _, target in phase}) == len(phase)
                assert reverse == tuple((target, source) for source, target in phase)

    zero_route = prepared.coarse_fine_routes[0]
    assert zero_route.route_count == 0
    assert zero_route.permutations == ((),)
    coarse_packed = prepared.pack(state)[0]
    np.testing.assert_array_equal(zero_route.serial_exchange(coarse_packed), 0.0)

    route = prepared.same_level_routes[0]
    received = route.serial_exchange(coarse_packed)
    received_cotangent = jnp.arange(received.size, dtype=coarse_packed.dtype).reshape(
        received.shape
    )
    source_cotangent = route.serial_accumulate(
        jnp.zeros_like(coarse_packed), received_cotangent
    )
    np.testing.assert_allclose(
        jnp.vdot(received, received_cotangent),
        jnp.vdot(coarse_packed, source_cotangent),
        rtol=0.0,
        atol=0.0,
    )


def test_repartition_migrates_packed_values_by_stable_block_id():
    hierarchy = _hierarchy(cells=16, fine_capacity=16)
    _, compiled = _compiled(hierarchy, coarse_slot=3, coarse_cell=1)
    fd, source = _prepare(hierarchy, compiled, 2)
    _, target = _prepare(
        hierarchy,
        compiled,
        2,
        costs=(jnp.asarray([100.0, 1.0, 1.0, 1.0]), None),
    )
    state = _state(compiled.topology, fd)

    migration = source.migration_to(target)
    assert isinstance(migration, BlockAMRStableIDMigrationPlan)
    assert sum(migration.moved_block_counts) > 0
    migrated = migration.migrate(source.pack(state))
    restored = target.unpack(migrated)

    for expected, actual in zip(state.levels, restored.levels, strict=True):
        np.testing.assert_allclose(actual.values, expected.safe_values())


def test_resource_evidence_counts_the_real_allocations_exactly_and_manifest_is_canonical():
    hierarchy = _hierarchy()
    _, compiled = _compiled(hierarchy)
    _, prepared = _prepare(hierarchy, compiled, 3)
    schedules = (
        *prepared.same_level_routes,
        *prepared.coarse_fine_routes,
        *prepared.interface_routes,
    )
    dynamic_arrays = tuple(
        array for route in schedules for array in route.dynamic_arrays
    ) + tuple(array for route in prepared.fill_routes for array in route.dynamic_arrays)

    evidence = prepared.resources
    assert evidence.active_blocks == tuple(
        int(np.count_nonzero(np.asarray(level.active)))
        for level in compiled.topology.levels
    )
    assert evidence.allocated_block_slots == tuple(
        prepared.partition.part_count * value for value in prepared.local_block_capacities
    )
    assert evidence.dynamic_route_array_entries == sum(
        int(array.size) for array in dynamic_arrays
    )
    assert evidence.dynamic_route_array_bytes == sum(
        int(array.size) * int(array.dtype.itemsize) for array in dynamic_arrays
    )
    assert evidence.static_permutation_pairs == sum(
        len(phase) for route in schedules for phase in route.permutations
    )

    manifest = prepared.manifest_compatibility_data()
    assert json.loads(json.dumps(manifest, allow_nan=False)) == manifest
    assert manifest["topology_epoch_id"] == compiled.topology.epoch.epoch_id
    assert manifest["active_stable_block_ids"] == [
        np.asarray(level.block_ids)[np.asarray(level.active)].tolist()
        for level in compiled.topology.levels
    ]
    assert "shard_payload" not in manifest


def test_serial_packed_fill_patch_and_reverse_match_canonical_foundation():
    hierarchy = _hierarchy()
    _, compiled = _compiled(hierarchy)
    fd, prepared = _prepare(hierarchy, compiled, 3)
    state = _state(compiled.topology, fd)

    reference = fd.fill_patch(state)
    packed_result = prepared.serial_fill_patch(state)

    assert bool(reference.complete) and bool(packed_result.complete)
    for expected, actual in zip(
        reference.workspaces, packed_result.workspaces, strict=True
    ):
        np.testing.assert_allclose(actual.values, expected.values)
        np.testing.assert_array_equal(actual.valid, expected.valid)
        np.testing.assert_array_equal(actual.source_class, expected.source_class)

    cotangents = tuple(jnp.ones_like(value.values) for value in reference.workspaces)
    packed_reverse = prepared.serial_fill_patch_reverse(cotangents, state)

    def canonical_values(values):
        hierarchy_state = phx.discretization.BlockHierarchyState(
            compiled.topology,
            tuple(
                phx.discretization.BlockLevelState(plan, metadata, value)
                for plan, metadata, value in zip(
                    hierarchy.levels,
                    compiled.topology.levels,
                    values,
                    strict=True,
                )
            ),
        )
        result = fd.fill_patch(hierarchy_state)
        return tuple(workspace.values for workspace in result.workspaces)

    values = tuple(level.values for level in state.levels)
    _, pullback = jax.vjp(canonical_values, values)
    canonical_reverse = pullback(cotangents)[0]
    combined_packed_reverse = tuple(
        current.values + old.values + new.values
        for current, old, new in zip(
            packed_reverse[0].levels,
            packed_reverse[1].levels,
            packed_reverse[2].levels,
            strict=True,
        )
    )
    for expected, actual in zip(canonical_reverse, combined_packed_reverse, strict=True):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("old_time", "new_time", "fill_time"),
    (
        (jnp.nan, 1.0, 0.5),
        (1.0, 0.0, 0.5),
    ),
)
def test_serial_fill_patch_rejects_nonfinite_or_reversed_time_intervals(
    old_time,
    new_time,
    fill_time,
):
    hierarchy = _hierarchy()
    _, compiled = _compiled(hierarchy)
    fd, prepared = _prepare(hierarchy, compiled, 3)
    state = _state(compiled.topology, fd)

    with pytest.raises(Exception, match="FillPatch time"):
        result = prepared.serial_fill_patch(
            state,
            coarse_old_time=old_time,
            coarse_new_time=new_time,
            fill_time=fill_time,
        )
        jax.block_until_ready(result.workspaces[1].values)


@pytest.mark.skipif(
    len(jax.devices()) < 2,
    reason="requires at least two JAX devices",
)
def test_real_two_device_fill_patch_and_reverse_match_serial_routes():
    hierarchy = _hierarchy()
    _, compiled = _compiled(hierarchy)
    group = _execution_group(jax.devices()[:2])
    fd, prepared = _prepare(hierarchy, compiled, 2, group=group)
    state = _state(compiled.topology, fd)

    serial = prepared.serial_fill_patch(state)
    distributed = prepared.distributed_fill_patch(state)
    for expected, actual in zip(serial.workspaces, distributed.workspaces, strict=True):
        np.testing.assert_allclose(actual.values, expected.values)
        np.testing.assert_array_equal(actual.valid, expected.valid)

    for route in (
        *prepared.same_level_routes,
        *prepared.coarse_fine_routes,
        *prepared.interface_routes,
    ):
        for array in route.dynamic_arrays:
            assert isinstance(array.sharding, jax.sharding.NamedSharding)
            assert array.sharding.spec[0] == prepared.partition.axis_name

    cotangents = tuple(jnp.ones_like(value.values) for value in serial.workspaces)
    serial_reverse = prepared.serial_fill_patch_reverse(cotangents, state)
    distributed_reverse = prepared.distributed_fill_patch_reverse(cotangents, state)
    for serial_state, distributed_state in zip(
        serial_reverse[:3], distributed_reverse[:3], strict=True
    ):
        for expected, actual in zip(
            serial_state.levels, distributed_state.levels, strict=True
        ):
            np.testing.assert_allclose(actual.values, expected.values)
