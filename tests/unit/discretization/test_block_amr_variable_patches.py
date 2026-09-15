#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _grid(*, periodic=False):
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _plan(*, periodic=False):
    wide = phx.discretization.PatchShapeSignature((4,), halo_width=1)
    narrow = phx.discretization.PatchShapeSignature((2,), halo_width=1)
    return phx.discretization.VariablePatchHierarchyPlan(
        _grid(periodic=periodic),
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (
                    phx.discretization.PatchBucketPlan(wide, 2),
                    phx.discretization.PatchBucketPlan(narrow, 4),
                ),
            ),
            phx.discretization.VariablePatchLevelPlan(
                1,
                (
                    phx.discretization.PatchBucketPlan(wide, 2),
                    phx.discretization.PatchBucketPlan(narrow, 8),
                ),
            ),
        ),
        (
            phx.discretization.LogicalPatchBox(0, (0,), (4,)),
            phx.discretization.LogicalPatchBox(0, (4,), (8,)),
        ),
    )


def _tags(plan, *, first=(), second=()):
    level = plan.levels[0]
    wide = jnp.zeros(
        (level.buckets[0].lane_capacity,) + level.buckets[0].signature.envelope_shape,
        dtype=bool,
    )
    narrow = jnp.zeros(
        (level.buckets[1].lane_capacity,) + level.buckets[1].signature.envelope_shape,
        dtype=bool,
    )
    for lane, local in first:
        wide = wide.at[lane, local].set(True)
    for lane, local in second:
        narrow = narrow.at[lane, local].set(True)
    return ((wide, narrow),)


def test_variable_patch_compiler_preserves_logical_identity_across_bucket_lanes():
    plan = _plan()
    compiler = phx.discretization.VariablePatchTopologyCompiler(plan)
    initial = compiler.initial_topology()

    result = compiler.compile(initial, _tags(plan, first=((0, 1),)))

    assert result.status.successful
    assert result.status.changed
    fine = result.topology.levels[1]
    active_boxes = fine.active_boxes()
    assert len(active_boxes) == 1
    bucket, lane, box = active_boxes[0]
    assert (bucket, lane) == (1, 0)
    assert box.lower == (2,)
    assert box.upper == (4,)
    assert result.topology.epoch.index == initial.epoch.index + 1
    assert result.topology.locate(1, (2,)) is not None
    assert result.topology.locate(1, (4,)) is None


def test_variable_patch_compiler_rejects_inactive_and_envelope_padding_tags():
    plan = _plan()
    compiler = phx.discretization.VariablePatchTopologyCompiler(plan)
    initial = compiler.initial_topology()
    with pytest.raises(ValueError, match="inactive or envelope-padding"):
        compiler.compile(initial, _tags(plan, second=((0, 0),)))


def test_variable_patch_compiler_checks_nonzero_parent_nesting_coordinates():
    plan = _plan()
    compiler = phx.discretization.VariablePatchTopologyCompiler(
        plan,
        proper_nesting=1,
    )
    initial = compiler.initial_topology()

    result = compiler.compile(initial, _tags(plan, first=((1, 3),)))

    assert not result.status.successful
    assert result.status.code == "proper_nesting_failed"
    assert result.topology.epoch.epoch_id == initial.epoch.epoch_id


def test_variable_patch_field_masks_inactive_lanes_and_padding():
    plan = _plan()
    topology = phx.discretization.VariablePatchTopologyCompiler(plan).initial_topology()
    metadata = topology.levels[0]
    state = phx.discretization.VariablePatchFieldState(
        metadata,
        (
            jnp.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            jnp.full((4, 2), jnp.nan),
        ),
    )
    wide, narrow = state.safe_values()
    assert jnp.all(jnp.isfinite(wide))
    assert jnp.all(narrow == 0.0)


def _state(topology, *, coarse_offset=0.0):
    level_values = []
    for level, metadata in enumerate(topology.levels):
        bucket_values = []
        for bucket_index, bucket in enumerate(metadata.plan.buckets):
            shape = (bucket.lane_capacity,) + bucket.signature.envelope_shape
            values = jnp.full(shape, jnp.nan)
            if level == 0 and bucket_index == 0:
                values = values.at[0].set(
                    jnp.asarray([1.0, 2.0, 3.0, 4.0]) + coarse_offset
                )
                values = values.at[1].set(
                    jnp.asarray([5.0, 6.0, 7.0, 8.0]) + coarse_offset
                )
            if level == 1 and bucket_index == 1 and bool(metadata.active[1][0]):
                values = values.at[0].set(jnp.asarray([10.0, 20.0]))
            bucket_values.append(values)
        level_values.append(
            phx.discretization.VariablePatchFieldState(metadata, bucket_values)
        )
    return phx.discretization.VariablePatchHierarchyState(topology, level_values)


def test_variable_patch_fill_patch_reads_same_level_across_unequal_lanes():
    topology = phx.discretization.VariablePatchTopologyCompiler(
        _plan()
    ).initial_topology()
    state = _state(topology)
    fill = phx.discretization.VariablePatchFillPatchPlan(topology, 0)

    workspace, request = fill.execute(state)

    assert workspace.values[0][0, 5] == 5.0
    assert workspace.source_class[0][0, 5] == int(
        phx.discretization.VariablePatchFillSource.SAME_LEVEL
    )
    assert request.masks[0][0, 0]


def test_variable_patch_fill_patch_interpolates_coarse_bucket_values():
    plan = _plan()
    compiler = phx.discretization.VariablePatchTopologyCompiler(plan)
    initial = compiler.initial_topology()
    topology = compiler.compile(initial, _tags(plan, first=((0, 1),))).topology
    old = _state(topology, coarse_offset=0.0)
    new = _state(topology, coarse_offset=10.0)
    fill = phx.discretization.VariablePatchFillPatchPlan(topology, 1)

    workspace, _ = fill.execute(
        new,
        old,
        new,
        0.0,
        1.0,
        0.5,
    )

    assert workspace.values[1][0, 0] == 6.0
    assert workspace.values[1][0, 3] == 8.0


def test_variable_patch_topology_rejects_direct_unaligned_fine_metadata():
    plan = _plan()
    initial = phx.discretization.VariablePatchTopologyCompiler(plan).initial_topology()
    invalid_fine = phx.discretization.VariablePatchLevelMetadata(
        plan.levels[1],
        (
            (phx.discretization.LogicalPatchBox(1, (1,), (3,)),),
            (),
        ),
    )

    with pytest.raises(ValueError, match="align to complete coarse"):
        phx.discretization.VariablePatchHierarchyTopology(
            plan,
            (initial.levels[0], invalid_fine),
        )


def test_variable_patch_fill_patch_keeps_envelope_padding_inert():
    wide = phx.discretization.PatchShapeSignature((4,), halo_width=1)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        _grid(),
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(wide, 2),)
            ),
            phx.discretization.VariablePatchLevelPlan(
                1, (phx.discretization.PatchBucketPlan(wide, 4),)
            ),
        ),
        (
            phx.discretization.LogicalPatchBox(0, (0,), (4,)),
            phx.discretization.LogicalPatchBox(0, (4,), (8,)),
        ),
    )
    compiler = phx.discretization.VariablePatchTopologyCompiler(plan)
    initial = compiler.initial_topology()
    tags = ((jnp.asarray([[False, True, False, False], [False, False, False, False]]),),)
    topology = compiler.compile(initial, tags).topology
    fill = phx.discretization.VariablePatchFillPatchPlan(topology, 1)

    assert not bool(fill.active_padding[0][0, 4])
    assert fill.source_class[0][0, 4] == int(
        phx.discretization.VariablePatchFillSource.INACTIVE
    )


def test_variable_patch_fill_patch_preserves_tensor_component_shape():
    topology = phx.discretization.VariablePatchTopologyCompiler(
        _plan()
    ).initial_topology()
    fields = []
    for metadata in topology.levels:
        arrays = []
        for bucket_index, bucket in enumerate(metadata.plan.buckets):
            shape = (bucket.lane_capacity,) + bucket.signature.envelope_shape + (2, 2)
            value = jnp.full(shape, jnp.nan)
            if bucket_index == 0:
                value = value.at[0].set(jnp.arange(16.0).reshape((4, 2, 2)))
                value = value.at[1].set(jnp.arange(16.0, 32.0).reshape((4, 2, 2)))
            arrays.append(value)
        fields.append(phx.discretization.VariablePatchFieldState(metadata, arrays))
    state = phx.discretization.VariablePatchHierarchyState(topology, fields)
    workspace, _ = phx.discretization.VariablePatchFillPatchPlan(
        topology,
        0,
        component_shape=(2, 2),
    ).execute(state)

    assert workspace.values[0].shape == (2, 6, 2, 2)
    assert jnp.array_equal(workspace.values[0][0, 5], fields[0].values[0][1, 0])


def test_variable_patch_topology_id_ignores_bucket_lane_placement():
    four = phx.discretization.PatchShapeSignature((4,), halo_width=1)
    six = phx.discretization.PatchShapeSignature((6,), halo_width=1)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        _grid(),
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (
                    phx.discretization.PatchBucketPlan(four, 2),
                    phx.discretization.PatchBucketPlan(six, 2),
                ),
            ),
        ),
        (
            phx.discretization.LogicalPatchBox(0, (0,), (4,)),
            phx.discretization.LogicalPatchBox(0, (4,), (8,)),
        ),
    )
    boxes = (
        phx.discretization.LogicalPatchBox(0, (0,), (4,)),
        phx.discretization.LogicalPatchBox(0, (4,), (8,)),
    )
    first = phx.discretization.VariablePatchHierarchyTopology(
        plan,
        (
            phx.discretization.VariablePatchLevelMetadata(
                plan.levels[0],
                (boxes, ()),
            ),
        ),
    )
    second = phx.discretization.VariablePatchHierarchyTopology(
        plan,
        (
            phx.discretization.VariablePatchLevelMetadata(
                plan.levels[0],
                ((), boxes),
            ),
        ),
    )

    assert first.topology_id == second.topology_id
    assert first.layout_id != second.layout_id


def test_variable_patch_compiler_ignores_stale_deeper_tags_during_coarsening():
    signature = phx.discretization.PatchShapeSignature((4,), halo_width=1)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        _grid(),
        tuple(
            phx.discretization.VariablePatchLevelPlan(
                level,
                (phx.discretization.PatchBucketPlan(signature, 8),),
            )
            for level in range(3)
        ),
        (
            phx.discretization.LogicalPatchBox(0, (0,), (4,)),
            phx.discretization.LogicalPatchBox(0, (4,), (8,)),
        ),
    )
    compiler = phx.discretization.VariablePatchTopologyCompiler(plan)
    initial = compiler.initial_topology()
    first = compiler.compile(
        initial,
        (
            (
                jnp.asarray(
                    [[False, True, False, False], [False] * 4] + [[False] * 4] * 6
                ),
            ),
            (jnp.zeros((8, 4), dtype=bool),),
        ),
    ).topology
    refine_second = jnp.zeros((8, 4), dtype=bool).at[0, 1].set(True)
    second = compiler.compile(
        first,
        (
            (
                jnp.asarray(
                    [[False, True, False, False], [False] * 4] + [[False] * 4] * 6
                ),
            ),
            (refine_second,),
        ),
    ).topology
    stale_deep = jnp.zeros((8, 4), dtype=bool).at[0, 1].set(True)
    result = compiler.compile(
        second,
        (
            (jnp.zeros((8, 4), dtype=bool),),
            (stale_deep,),
        ),
    )

    assert result.status.successful
    assert not result.topology.levels[1].active_boxes()
    assert not result.topology.levels[2].active_boxes()


def test_clustering_policy_splits_sparse_connected_component():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    base_signature = phx.discretization.PatchShapeSignature((4, 4), halo_width=1)
    wide = phx.discretization.PatchShapeSignature((8, 8), halo_width=1, alignment=2)
    narrow = phx.discretization.PatchShapeSignature((2, 2), halo_width=1, alignment=2)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (phx.discretization.PatchBucketPlan(base_signature, 1),),
            ),
            phx.discretization.VariablePatchLevelPlan(
                1,
                (
                    phx.discretization.PatchBucketPlan(wide, 64),
                    phx.discretization.PatchBucketPlan(narrow, 64),
                ),
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0), (4, 4)),),
    )
    compiler = phx.discretization.VariablePatchTopologyCompiler(
        plan,
        clustering=phx.discretization.PatchClusteringPolicy(
            minimum_fill_ratio=0.7,
            maximum_aspect_ratio=8.0,
        ),
    )
    source = compiler.initial_topology()
    tags = jnp.zeros((1, 4, 4), dtype=bool)
    tags = tags.at[0, 0, :].set(True)
    tags = tags.at[0, :, 0].set(True)

    result = compiler.compile(source, ((tags,),))

    assert result.status.successful
    assert len(result.topology.logical_boxes[1]) > 1
    refined_tags = {
        (2 * i + di, 2 * j + dj)
        for i in range(4)
        for j in range(4)
        if i == 0 or j == 0
        for di in range(2)
        for dj in range(2)
    }
    assert all(
        sum(box.contains_cell(cell) for cell in refined_tags) / box.cell_count >= 0.7
        for box in result.topology.logical_boxes[1]
    )
