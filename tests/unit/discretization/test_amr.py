#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _level_plan(level=0, maximum_blocks=3, spacing=1.0):
    return phx.discretization.BlockLevelPlan(
        level,
        (4,),
        maximum_blocks,
        halo_width=1,
        refinement_ratio=2,
        spacing=(spacing,),
    )


def _metadata(plan):
    return phx.discretization.BlockMetadata(
        plan,
        active=[True, True, False],
        block_ids=[10, 11, -1],
        parent_ids=[-1, -1, -1],
        logical_indices=[[0], [1], [0]],
        neighbor_slots=[[[-1, 1]], [[0, -1]], [[-1, -1]]],
    )


def test_inactive_amr_payload_is_inert_even_when_nonfinite():
    plan = _level_plan()
    metadata = _metadata(plan)
    values = jnp.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [jnp.nan, jnp.inf, -jnp.inf, jnp.nan],
        ]
    )
    state = phx.discretization.BlockLevelState(plan, metadata, values)

    safe = state.safe_values()
    halos = state.fill_same_level_halo_1d()

    assert jnp.all(jnp.isfinite(safe))
    assert jnp.allclose(safe[2], 0.0)
    assert jnp.allclose(halos[0], jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert jnp.allclose(halos[1], jnp.asarray([4.0, 5.0, 6.0, 7.0, 8.0, 0.0]))


def test_conservative_block_transfer_preserves_constants_and_averages():
    transfer = phx.discretization.ConservativeBlockTransfer(2, 2)
    coarse = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    fine = transfer.prolong(coarse)
    restricted = transfer.restrict(fine)

    assert fine.shape == (4, 4)
    assert jnp.allclose(restricted, coarse)
    assert jnp.allclose(transfer.conservation_residual(coarse), 0.0)


def test_flux_register_applies_only_interface_mismatch():
    register = phx.discretization.FluxRegister(
        coarse_flux=jnp.asarray([1.0, 2.0, 3.0]),
        fine_flux=jnp.asarray([1.5, 9.0, 2.0]),
        interface_mask=jnp.asarray([True, False, True]),
    )
    state = jnp.asarray([10.0, 20.0, 30.0])

    corrected = register.apply(state, jnp.asarray([0.5, 0.5, 0.25]))

    assert jnp.allclose(corrected, jnp.asarray([11.0, 20.0, 26.0]))


def test_fixed_capacity_refinement_reports_missing_child_slots():
    parent_plan = _level_plan()
    child_plan = _level_plan(level=1, spacing=0.5)
    parent = _metadata(parent_plan)
    child = phx.discretization.BlockMetadata(
        child_plan,
        active=[False, False, False],
        block_ids=[-1, -1, -1],
        parent_ids=[10, 10, 11],
        logical_indices=[[0], [1], [2]],
        neighbor_slots=[[[-1, -1]], [[-1, -1]], [[-1, -1]]],
    )
    refinement = phx.discretization.FixedCapacityRefinementPlan(
        [[0, 1], [2, -1], [-1, -1]]
    )

    decision = refinement.decide(
        parent,
        child,
        indicators=jnp.asarray([2.0, 3.0, 0.0]),
        threshold=1.0,
    )

    assert jnp.array_equal(decision.child_active, jnp.asarray([True, True, True]))
    assert decision.overflow


def test_hierarchy_state_records_exact_realized_block_trace():
    coarse_plan = _level_plan()
    fine_plan = _level_plan(level=1, spacing=0.5)
    hierarchy = phx.discretization.BlockHierarchyPlan((coarse_plan, fine_plan))
    coarse = phx.discretization.BlockLevelState(
        coarse_plan,
        _metadata(coarse_plan),
        jnp.ones((3, 4)),
    )
    fine = phx.discretization.BlockLevelState(
        fine_plan,
        _metadata(fine_plan),
        jnp.ones((3, 4)),
    )

    state = phx.discretization.BlockHierarchyState(hierarchy, (coarse, fine))

    assert state.refinement_trace_id
    assert state.plan.plan_id == hierarchy.plan_id
