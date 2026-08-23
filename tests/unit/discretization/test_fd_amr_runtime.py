#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _plan_2d():
    return phx.discretization.BlockLevelPlan(
        0,
        (4, 4),
        4,
        halo_width=(1, 1),
        refinement_ratio=2,
        spacing=(1.0, 1.0),
    )


def _metadata_2d(plan):
    return phx.discretization.BlockMetadata(
        plan,
        active=[True, True, True, True],
        block_ids=[0, 1, 2, 3],
        parent_ids=[-1, -1, -1, -1],
        logical_indices=[[0, 0], [1, 0], [0, 1], [1, 1]],
        neighbor_slots=[
            [[-1, 1], [-1, 2]],
            [[0, -1], [-1, 3]],
            [[-1, 3], [0, -1]],
            [[2, -1], [1, -1]],
        ],
    )


def test_multidimensional_same_level_halos_fill_faces_edges_and_corners():
    plan = _plan_2d()
    metadata = _metadata_2d(plan)
    values = jnp.stack(
        tuple(jnp.full(plan.block_shape, float(slot)) for slot in range(4))
    )
    state = phx.discretization.BlockLevelState(plan, metadata, values)

    workspace = phx.discretization.FDAMRHaloPlan(plan).fill_same_level(state)

    assert workspace.values.shape == (4, 6, 6)
    np.testing.assert_allclose(workspace.values[0, -1, 2], 1.0)
    np.testing.assert_allclose(workspace.values[0, 2, -1], 2.0)
    np.testing.assert_allclose(workspace.values[0, -1, -1], 3.0)
    np.testing.assert_allclose(workspace.values[3, 0, 0], 0.0)


def test_entity_specific_transfers_preserve_declared_invariants():
    cell = phx.discretization.AMREntityTransferPlan.cells(2)
    node = phx.discretization.AMREntityTransferPlan.nodes(1)
    face = phx.discretization.AMREntityTransferPlan.faces(2, 0)
    edge = phx.discretization.AMREntityTransferPlan.edges(2, 1)
    coarse_cell = jnp.arange(16.0).reshape((4, 4))
    coarse_node = 2.0 * jnp.linspace(0.0, 1.0, 5) - 0.4

    fine_cell = cell.prolong(coarse_cell)
    fine_node = node.prolong(coarse_node)

    assert cell.report.passed and node.report.passed
    assert face.report.passed and edge.report.passed
    np.testing.assert_allclose(
        cell.restrict(fine_cell), coarse_cell, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        fine_node,
        2.0 * jnp.linspace(0.0, 1.0, 9) - 0.4,
        rtol=0.0,
        atol=2e-14,
    )
    assert face.fine_shape((5, 4)) == (9, 8)
    assert edge.fine_shape((4, 5)) == (7, 10)


def _one_dimensional_state(plan, active, values, *, block_ids, parent_ids):
    capacity = plan.maximum_blocks
    metadata = phx.discretization.BlockMetadata(
        plan,
        active=active,
        block_ids=block_ids,
        parent_ids=parent_ids,
        logical_indices=[[index] for index in range(capacity)],
        neighbor_slots=[[[-1, -1]] for _ in range(capacity)],
    )
    return phx.discretization.BlockLevelState(plan, metadata, values)


def test_coarse_fine_halos_use_parent_prolongation_and_child_offsets():
    coarse_plan = phx.discretization.BlockLevelPlan(
        0,
        (4,),
        1,
        halo_width=1,
        refinement_ratio=2,
        spacing=(1.0,),
    )
    fine_plan = phx.discretization.BlockLevelPlan(
        1,
        (4,),
        2,
        halo_width=1,
        refinement_ratio=2,
        spacing=(0.5,),
    )
    coarse = _one_dimensional_state(
        coarse_plan,
        [True],
        jnp.asarray([[1.0, 2.0, 3.0, 4.0]]),
        block_ids=[0],
        parent_ids=[-1],
    )
    fine = _one_dimensional_state(
        fine_plan,
        [True, True],
        jnp.zeros((2, 4)),
        block_ids=[10, 11],
        parent_ids=[0, 0],
    )
    transfer = phx.discretization.AMREntityTransferPlan.cells(1)

    workspace = phx.discretization.FDAMRHaloPlan(fine_plan).fill_coarse_fine(
        fine,
        coarse,
        jnp.asarray([0, 0]),
        jnp.asarray([[0], [1]]),
        transfer,
    )

    assert workspace.values.shape == (2, 6)
    np.testing.assert_allclose(workspace.values[:, 1:-1], 0.0)
    assert workspace.values[0, 0] != workspace.values[1, 0]


def test_subcycling_accumulates_time_integrated_flux_and_refluxes_coarse_state():
    plan = phx.discretization.ConservativeAMRSubcyclingPlan(2)
    coarse = jnp.asarray([10.0, 20.0])
    fine = jnp.asarray([1.0, 2.0, 3.0, 4.0])

    result = plan.advance(
        0.0,
        coarse,
        fine,
        0.2,
        lambda time, state, dt, args: state,
        lambda time, state, dt, args: state,
        lambda state, args: jnp.asarray([1.0, 0.0]),
        lambda state, args: jnp.asarray([2.0, 0.0]),
        lambda flux: flux,
        jnp.asarray([True, False]),
        jnp.asarray([0.5, 0.5]),
    )

    np.testing.assert_allclose(result.flux_register.mismatch(), [0.2, 0.0])
    np.testing.assert_allclose(result.coarse_state, [10.4, 20.0])
    assert result.substeps == 2
    assert result.temporal_method_id == "temporal:caller-supplied"


def test_regridding_populates_children_deterministically_and_masks_inactive_payload():
    parent_plan = phx.discretization.BlockLevelPlan(
        0,
        (4,),
        2,
        halo_width=1,
        refinement_ratio=2,
        spacing=(1.0,),
    )
    child_plan = phx.discretization.BlockLevelPlan(
        1,
        (4,),
        4,
        halo_width=1,
        refinement_ratio=2,
        spacing=(0.5,),
    )
    parent = _one_dimensional_state(
        parent_plan,
        [True, False],
        jnp.asarray([[1.0, 2.0, 3.0, 4.0], [jnp.nan] * 4]),
        block_ids=[0, -1],
        parent_ids=[-1, -1],
    )
    child = _one_dimensional_state(
        child_plan,
        [False] * 4,
        jnp.full((4, 4), jnp.nan),
        block_ids=[-1] * 4,
        parent_ids=[0, 0, 1, 1],
    )
    refinement = phx.discretization.FixedCapacityRefinementPlan([[0, 1], [2, 3]])
    regrid = phx.discretization.FDRegridPlan(
        refinement,
        phx.discretization.AMREntityTransferPlan.cells(1),
        jnp.asarray([[[0], [1]], [[0], [1]]]),
    )
    apply = eqx.filter_jit(regrid.apply)

    first = apply(parent, child, jnp.asarray([2.0, 0.0]), 1.0)
    second = apply(parent, child, jnp.asarray([2.0, 0.0]), 1.0)

    assert jnp.array_equal(first.decision.child_active, [True, True, False, False])
    assert jnp.all(jnp.isfinite(first.child_values))
    np.testing.assert_allclose(first.child_values, second.child_values)
    assert first.regrid_trace_id == second.regrid_trace_id


def test_amr_migration_reorders_active_blocks_and_preserves_inactive_sentinels():
    plan = phx.discretization.BlockLevelPlan(
        0,
        (4,),
        3,
        halo_width=1,
        refinement_ratio=2,
        spacing=(1.0,),
    )
    state = _one_dimensional_state(
        plan,
        [True, True, False],
        jnp.asarray([[1.0] * 4, [2.0] * 4, [jnp.nan] * 4]),
        block_ids=[10, 11, -1],
        parent_ids=[-1, -1, -1],
    )
    migration = phx.discretization.AMRMigrationPlan([2, 0, -1], 4)

    result = eqx.filter_jit(migration.migrate)(state)

    assert jnp.array_equal(result.active, [True, False, True, False])
    assert jnp.array_equal(result.block_ids, [11, -1, 10, -1])
    np.testing.assert_allclose(result.values[0], 2.0)
    np.testing.assert_allclose(result.values[2], 1.0)
    np.testing.assert_allclose(result.values[jnp.asarray([1, 3])], 0.0)
