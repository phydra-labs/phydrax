#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(cells: Any = 8, *, periodic: Any = False) -> Any:
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _hierarchy(
    *, fine_capacity: Any = 8, fine_shape: Any = 2, periodic: Any = False
) -> Any:
    return phx.discretization.BlockHierarchyPlan(
        _grid(periodic=periodic),
        (
            phx.discretization.BlockLevelPlan(0, (4,), 2, halo_width=1),
            phx.discretization.BlockLevelPlan(
                1, (fine_shape,), fine_capacity, halo_width=1
            ),
        ),
    )


def test_amr_scenario_1() -> None:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8),
            phx.discretization.UniformCellAxisSpec(8),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, -2.0], [2.0, 2.0]]))
    plan = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(0, (4, 4), 4),
            phx.discretization.BlockLevelPlan(1, (2, 4), 32),
        ),
    )

    assert plan.global_cell_shapes == ((8, 8), (16, 16))
    assert plan.block_lattice_shapes == ((2, 2), (8, 4))
    assert plan.children_per_parent == ((4, 2),)
    assert plan.level_spacings == ((0.25, 0.5), (0.125, 0.25))
    point_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(9),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="interval-primary"):
        phx.discretization.BlockHierarchyPlan(
            point_grid, (phx.discretization.BlockLevelPlan(0, (3,), 3),)
        )

    with pytest.raises(ValueError, match="exact parent/child"):
        phx.discretization.BlockHierarchyPlan(
            _grid(),
            (
                phx.discretization.BlockLevelPlan(0, (2,), 4),
                phx.discretization.BlockLevelPlan(1, (8,), 2),
            ),
        )
    hierarchy = _hierarchy()
    topology = phx.discretization.BlockTopologyCompiler(hierarchy).initial_topology()
    base = topology.levels[0]

    assert base.block_ids.dtype == jnp.int32
    assert base.parent_ids.dtype == jnp.int32
    np.testing.assert_array_equal(base.block_ids, jnp.asarray([0, 1], dtype=jnp.int32))
    np.testing.assert_array_equal(base.logical_indices, jnp.asarray([[0], [1]]))
    assert len(topology.patch_boxes(0)) == 2
    assert not bool(jnp.any(topology.covered_cells[0]))
    assert not bool(jnp.any(topology.covered_cells[1]))
    assert topology.epoch.index == 0
    assert topology.epoch.geometry_id == hierarchy.geometry_id
    hierarchy = _hierarchy()
    compiler = phx.discretization.BlockTopologyCompiler(hierarchy)
    initial = compiler.initial_topology()
    tags = jnp.zeros((2, 4), dtype="bool").at[0, 1].set(True)

    first = compiler.compile(initial, (tags,))
    unchanged = compiler.compile(first.topology, (tags,))
    other_tags = jnp.zeros((2, 4), dtype="bool").at[1, 2].set(True)
    other = compiler.compile(initial, (other_tags,))
    same_by_other_path = compiler.compile(other.topology, (tags,))

    assert first.status.successful and first.status.changed
    assert first.evidence.requested_blocks == (2, 1)
    np.testing.assert_array_equal(
        first.topology.levels[1].logical_indices[0], jnp.asarray([1])
    )
    assert int(first.topology.levels[1].parent_ids[0]) == 0
    assert unchanged.topology.epoch.epoch_id == first.topology.epoch.epoch_id
    assert unchanged.status.code == "unchanged"
    np.testing.assert_array_equal(
        same_by_other_path.topology.levels[1].block_ids,
        first.topology.levels[1].block_ids,
    )
    np.testing.assert_array_equal(
        same_by_other_path.topology.levels[1].logical_indices,
        first.topology.levels[1].logical_indices,
    )
    hierarchy = _hierarchy()
    compiler = phx.discretization.BlockTopologyCompiler(hierarchy)
    source = compiler.initial_topology()

    with pytest.raises(TypeError, match="exact Boolean dtype"):
        compiler.compile(source, (jnp.zeros((2, 4), dtype=jnp.int32),))


def test_amr_scenario_2() -> None:
    hierarchy = phx.discretization.BlockHierarchyPlan(
        _grid(),
        (
            phx.discretization.BlockLevelPlan(0, (4,), 2),
            phx.discretization.BlockLevelPlan(1, (2,), 8),
            phx.discretization.BlockLevelPlan(2, (2,), 16),
        ),
    )
    permissive = phx.discretization.BlockTopologyCompiler(hierarchy)
    initial = permissive.initial_topology()
    coarse_tags = jnp.zeros((2, 4), dtype="bool").at[0, 1].set(True)
    middle = permissive.compile(
        initial, (coarse_tags, jnp.zeros((8, 2), dtype="bool"))
    ).topology
    fine_tags = jnp.zeros((8, 2), dtype="bool").at[0, 0].set(True)

    result = phx.discretization.BlockTopologyCompiler(
        hierarchy, proper_nesting=1
    ).compile(middle, (coarse_tags, fine_tags))

    assert not result.status.successful
    assert result.status.code == "proper_nesting_failed"
    assert result.evidence.proper_nesting_rejections == (0, 1)
    assert result.topology.epoch.epoch_id == middle.epoch.epoch_id
    np.testing.assert_array_equal(
        result.routes.old_to_new_slots[1],
        jnp.arange(8, dtype=jnp.int32).at[1:].set(-1),
    )
    hierarchy = phx.discretization.BlockHierarchyPlan(
        _grid(),
        (
            phx.discretization.BlockLevelPlan(0, (4,), 2),
            phx.discretization.BlockLevelPlan(1, (2,), 8),
            phx.discretization.BlockLevelPlan(2, (2,), 16),
        ),
    )
    compiler = phx.discretization.BlockTopologyCompiler(
        hierarchy,
        proper_nesting=1,
        balance=phx.discretization.AMRBalanceStencil.FACE,
    )
    coarse_tags = jnp.zeros((2, 4), dtype="bool").at[0, 1].set(True)
    middle = compiler.compile(
        compiler.initial_topology(), (coarse_tags, jnp.zeros((8, 2), dtype="bool"))
    ).topology
    fine_tags = jnp.zeros((8, 2), dtype="bool").at[0, 0].set(True)

    result = compiler.compile(middle, (coarse_tags, fine_tags))

    assert result.status.code == "success"
    assert result.evidence.proper_nesting_rejections == (0, 0)
    assert result.evidence.balance_additions == (0, 1)
    finest = np.zeros((32,), dtype=np.int32)
    for level, boxes in enumerate(result.topology.logical_boxes):
        scale = 2 ** (2 - level)
        for box in boxes:
            finest[box.lower[0] * scale : box.upper[0] * scale] = level
    assert finest.max() == 2
    assert np.all(np.abs(np.diff(finest)) <= 1)
    hierarchy = _hierarchy(fine_capacity=1)
    compiler = phx.discretization.BlockTopologyCompiler(hierarchy)
    source = compiler.initial_topology()
    tags = jnp.zeros((2, 4), dtype="bool").at[0, 0].set(True).at[0, 2].set(True)

    result = compiler.compile(source, (tags,))

    assert not result.status.successful
    assert result.status.code == "capacity_exceeded"
    assert result.evidence.overflow_level == 1
    assert result.topology.epoch.epoch_id == source.epoch.epoch_id
    np.testing.assert_array_equal(
        result.topology.levels[1].block_ids, source.levels[1].block_ids
    )
    np.testing.assert_array_equal(
        result.routes.old_to_new_slots[0], jnp.asarray([0, 1], dtype=jnp.int32)
    )


def test_inactive_payload_is_inert_and_state_binds_realized_topology() -> None:
    hierarchy = _hierarchy()
    topology = phx.discretization.BlockTopologyCompiler(hierarchy).initial_topology()
    coarse = phx.discretization.BlockLevelState(
        hierarchy.levels[0], topology.levels[0], jnp.arange(8.0).reshape((2, 4))
    )
    fine = phx.discretization.BlockLevelState(
        hierarchy.levels[1],
        topology.levels[1],
        jnp.full((8, 2), jnp.nan),
    )
    state = phx.discretization.BlockHierarchyState(topology, (coarse, fine))

    assert jnp.all(jnp.isfinite(state.levels[1].safe_values()))
    assert state.topology.epoch.epoch_id == topology.epoch.epoch_id
    assert state.plan.plan_id == hierarchy.plan_id
