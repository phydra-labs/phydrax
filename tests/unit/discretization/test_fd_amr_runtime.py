#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _prepared(*, periodic=False, levels=2, halo=1):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    level_plans = [phx.discretization.BlockLevelPlan(0, (4,), 2, halo_width=halo)]
    if levels == 2:
        level_plans.append(phx.discretization.BlockLevelPlan(1, (2,), 8, halo_width=halo))
    hierarchy = phx.discretization.BlockHierarchyPlan(grid, level_plans)
    return phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()


def _state(topology, level_values):
    levels = tuple(
        phx.discretization.BlockLevelState(plan, metadata, values)
        for plan, metadata, values in zip(
            topology.plan.levels, topology.levels, level_values, strict=True
        )
    )
    return phx.discretization.BlockHierarchyState(topology, levels)


def _refined_topology(prepared, tagged_cell):
    initial = prepared.initial_topology()
    tags = jnp.zeros((2, 4), dtype="bool").at[tagged_cell // 4, tagged_cell % 4].set(True)
    result = prepared.compile_topology(initial, (tags,))
    assert result.status.successful
    return result.topology


def test_entity_transfer_seam_preserves_declared_cell_and_noncell_invariants():
    cell = phx.discretization.AMREntityTransferPlan.cells(2)
    node = phx.discretization.AMREntityTransferPlan.nodes(1)
    coarse_cell = jnp.arange(16.0).reshape((4, 4))
    coarse_node = 2.0 * jnp.linspace(0.0, 1.0, 5) - 0.4

    fine_cell = cell.prolong(coarse_cell)
    fine_node = node.prolong(coarse_node)

    np.testing.assert_allclose(cell.restrict(fine_cell), coarse_cell, atol=1e-14)
    np.testing.assert_allclose(
        fine_node, 2.0 * jnp.linspace(0.0, 1.0, 9) - 0.4, atol=2e-14
    )
    assert cell.report.passed and node.report.passed


def test_fill_patch_classifies_same_level_and_periodic_before_other_sources():
    prepared = _prepared(periodic=True, levels=1)
    topology = prepared.initial_topology()
    state = _state(
        topology,
        (jnp.asarray([[1.0] * 4, [2.0] * 4], dtype=jnp.float64),),
    )

    result = prepared.fill_patch(state)
    workspace = result.require_complete()[0]

    np.testing.assert_allclose(workspace.values[0], [2.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    assert int(workspace.source_class[0, 0]) == int(
        phx.discretization.FillPatchSource.PERIODIC
    )
    assert int(workspace.source_class[0, -1]) == int(
        phx.discretization.FillPatchSource.SAME_LEVEL
    )


def test_fill_patch_uses_multiple_coarse_blocks_and_old_new_time_interpolation():
    prepared = _prepared(halo=2)
    topology = _refined_topology(prepared, 3)
    fine_fill_plan = prepared.prepare_fill_patch(topology)[1]
    routed_donors = np.asarray(fine_fill_plan.coarse_donor_slots)[
        np.asarray(fine_fill_plan.coarse_donor_valid)
    ]
    np.testing.assert_array_equal(np.unique(routed_donors), [0, 1])
    fine = jnp.zeros((8, 2), dtype=jnp.float64).at[0].set(100.0)
    current = _state(
        topology,
        (
            jnp.asarray([[0.0] * 4, [10.0] * 4], dtype=jnp.float64),
            fine,
        ),
    )
    old = _state(
        topology,
        (
            jnp.asarray([[0.0] * 4, [10.0] * 4], dtype=jnp.float64),
            fine,
        ),
    )
    new = _state(
        topology,
        (
            jnp.asarray([[2.0] * 4, [14.0] * 4], dtype=jnp.float64),
            fine,
        ),
    )
    boundary_values = (
        jnp.zeros((2, 8), dtype=jnp.float64),
        jnp.zeros((8, 6), dtype=jnp.float64),
    )

    result = prepared.fill_patch(
        current,
        coarse_old=old,
        coarse_new=new,
        coarse_old_time=0.0,
        coarse_new_time=2.0,
        fill_time=1.0,
        physical_boundary_values=boundary_values,
    )
    fine_workspace = result.require_complete()[1]

    np.testing.assert_allclose(fine_workspace.values[0, :2], 1.0)
    np.testing.assert_allclose(fine_workspace.values[0, 2:4], 100.0)
    np.testing.assert_allclose(fine_workspace.values[0, 4:], 12.0)
    assert jnp.all(
        fine_workspace.source_class[0, jnp.asarray([0, 1, 4, 5])]
        == int(phx.discretization.FillPatchSource.COARSE_TIME_INTERPOLATED)
    )


def test_fill_patch_same_level_data_precedes_available_coarse_data():
    prepared = _prepared(halo=2)
    initial = prepared.initial_topology()
    tags = jnp.zeros((2, 4), dtype="bool").at[0, 2].set(True).at[0, 3].set(True)
    topology = prepared.compile_topology(initial, (tags,)).topology
    fine = jnp.zeros((8, 2), dtype=jnp.float64)
    fine = fine.at[0].set(7.0).at[1].set(9.0)
    state = _state(
        topology,
        (jnp.ones((2, 4), dtype=jnp.float64), fine),
    )
    boundaries = (
        jnp.zeros((2, 8), dtype=jnp.float64),
        jnp.zeros((8, 6), dtype=jnp.float64),
    )

    workspace = prepared.fill_patch(
        state, physical_boundary_values=boundaries
    ).require_complete()[1]

    np.testing.assert_allclose(workspace.values[1, :2], 7.0)
    assert jnp.all(
        workspace.source_class[1, :2]
        == int(phx.discretization.FillPatchSource.SAME_LEVEL)
    )


def test_physical_boundary_values_remain_caller_owned_and_incomplete_is_rejected():
    prepared = _prepared(levels=1)
    topology = prepared.initial_topology()
    state = _state(
        topology,
        (jnp.ones((2, 4), dtype=jnp.float64),),
    )

    request = prepared.fill_patch(state)
    assert not bool(request.complete)
    assert request.physical_boundary_requests[0].required
    with pytest.raises(ValueError, match="unresolved cells"):
        request.require_complete()

    supplied = prepared.fill_patch(
        state,
        physical_boundary_values=(jnp.full((2, 6), 5.0, dtype=jnp.float64),),
    )
    workspace = supplied.require_complete()[0]
    assert workspace.values[0, 0] == 5.0
    assert workspace.values[1, -1] == 5.0


def test_componentwise_topology_transition_is_conservative_and_zeroes_inactive_slots():
    prepared = _prepared()
    prepared.initial_topology()
    source = _refined_topology(prepared, 1)
    target_tags = jnp.zeros((2, 4), dtype="bool").at[1, 1].set(True)
    target = prepared.compile_topology(source, (target_tags,)).topology
    coarse = jnp.stack(
        (
            jnp.arange(8.0, dtype=jnp.float64).reshape((2, 4)),
            2.0 * jnp.arange(8.0, dtype=jnp.float64).reshape((2, 4)) + 1.0,
        ),
        axis=-1,
    )
    fine = (
        jnp.zeros((8, 2, 2), dtype=jnp.float64)
        .at[0]
        .set(jnp.asarray([[20.0, 3.0], [24.0, 5.0]]))
    )
    state = _state(source, (coarse, fine))
    transition = prepared.field_transition(
        source, target, "conserved", component_shape=(2,)
    )

    result = transition.apply(state)

    assert bool(result.successful)
    np.testing.assert_allclose(result.conservation_residual, 0.0, atol=1e-12)
    assert result.state.topology.epoch.epoch_id == target.epoch.epoch_id
    assert jnp.all(result.state.levels[1].values[1:] == 0.0)


def test_fill_patch_preparation_rejects_unresolved_coarse_routes():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(0, (4,), 2, halo_width=2),
            phx.discretization.BlockLevelPlan(1, (2,), 8, halo_width=2),
            phx.discretization.BlockLevelPlan(2, (2,), 16, halo_width=2),
        ),
    )
    prepared = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()
    initial = prepared.initial_topology()
    coarse_tags = jnp.zeros((2, 4), dtype="bool").at[0, 1].set(True)
    level_one_empty = jnp.zeros((8, 2), dtype="bool")
    middle = prepared.compile_topology(initial, (coarse_tags, level_one_empty)).topology
    level_one_tags = jnp.zeros((8, 2), dtype="bool").at[0, 0].set(True)
    target = prepared.compile_topology(middle, (coarse_tags, level_one_tags)).topology

    with pytest.raises(ValueError, match="unresolved"):
        prepared.prepare_fill_patch(target)


def test_prepared_fill_patch_explicitly_refuses_noncell_entity_routes():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(0, (4,), 2),
            phx.discretization.BlockLevelPlan(1, (2,), 8),
        ),
    )

    with pytest.raises(NotImplementedError, match="cell-centered"):
        phx.discretization.FDAMRHierarchyPlan(
            hierarchy, (phx.discretization.AMREntityTransferPlan.nodes(1),)
        )


def test_block_stencil_execution_accepts_only_complete_fill_patch_workspace():
    prepared = _prepared(periodic=True, levels=1)
    topology = prepared.initial_topology()
    values = jnp.arange(8.0, dtype=jnp.float64).reshape((2, 4))
    state = _state(topology, (values,))
    workspace = prepared.fill_patch(state).require_complete()[0]
    footprint = phx.discretization.StencilFootprint(("x",), (1,), (1,))
    execution = phx.discretization.BlockLocalStencilExecutionPlan(
        topology.plan.levels[0], footprint
    )

    result = execution.apply(workspace, lambda block: block[1:-1])

    assert bool(result.successful)
    np.testing.assert_allclose(result.values, values)
