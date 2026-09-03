from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _plan(capacity: int = 64):
    return phx.discretization.AdaptiveDyadicGridPlan(
        phx.discretization.MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 4),
        cell_capacity=capacity,
        enforce_two_to_one_balance=True,
    )


def _refine(plan, topology, slot):
    mask = jnp.zeros((plan.cell_capacity,), dtype=bool).at[slot].set(True)
    return plan.adapt(topology, refine_mask=mask)


def test_dyadic_refinement_is_covering_balanced_and_stably_identified() -> None:
    plan = _plan()
    root = plan.prepare()
    first = _refine(plan, root, int(root.root_slot))
    assert bool(first.accepted_candidate)
    topology = first.accepted
    assert int(topology.evidence.active_leaves) == 4
    assert bool(topology.evidence.covering)
    assert bool(topology.evidence.two_to_one_balanced)
    assert bool(topology.evidence.antichain)
    support = phx.discretization.DiscreteSupport(topology, 2, "dyadic-square")
    assert support.topology.topology_id == topology.topology_id

    selected = int(jnp.nonzero(topology.leaf_active, size=1)[0][0])
    second = _refine(plan, topology, selected)
    assert bool(second.accepted_candidate)
    assert int(second.accepted.evidence.active_leaves) == 7
    retained_keys = {
        (int(topology.levels[index]), int(topology.prefixes[index]))
        for index in np.flatnonzero(np.asarray(topology.leaf_active))
        if index != selected
    }
    next_keys = {
        (int(second.accepted.levels[index]), int(second.accepted.prefixes[index]))
        for index in np.flatnonzero(np.asarray(second.accepted.leaf_active))
    }
    assert retained_keys <= next_keys


def test_dyadic_coarsening_requires_complete_requested_siblings() -> None:
    plan = _plan()
    refined = _refine(plan, plan.prepare(), 0).accepted
    leaves = np.flatnonzero(np.asarray(refined.leaf_active))
    partial_mask = jnp.zeros((plan.cell_capacity,), dtype=bool).at[leaves[0]].set(True)
    partial = plan.adapt(refined, coarsen_mask=partial_mask)
    assert bool(partial.accepted_candidate)
    assert int(partial.evidence.accepted_coarsenings) == 0
    assert int(partial.accepted.evidence.active_leaves) == 4

    complete_mask = jnp.zeros((plan.cell_capacity,), dtype=bool).at[leaves].set(True)
    complete = plan.adapt(refined, coarsen_mask=complete_mask)
    assert bool(complete.accepted_candidate)
    assert int(complete.evidence.accepted_coarsenings) == 1
    assert int(complete.accepted.evidence.active_leaves) == 1


def test_dyadic_capacity_failure_preserves_previous_topology() -> None:
    plan = _plan(capacity=1)
    root = plan.prepare()
    transition = _refine(plan, root, 0)
    assert not bool(transition.accepted_candidate)
    assert int(transition.evidence.required_capacity) > plan.cell_capacity
    assert transition.accepted.topology_id == root.topology_id
    np.testing.assert_array_equal(transition.accepted.leaf_active, root.leaf_active)


def test_dyadic_refinement_closes_two_to_one_face_balance() -> None:
    plan = _plan()
    level_one = _refine(plan, plan.prepare(), 0).accepted
    level_one_slot = next(
        int(slot)
        for slot in np.flatnonzero(np.asarray(level_one.leaf_active))
        if int(level_one.levels[slot]) == 1 and int(level_one.prefixes[slot]) == 0
    )
    level_two = _refine(plan, level_one, level_one_slot).accepted
    boundary_grandchild = next(
        int(slot)
        for slot in np.flatnonzero(np.asarray(level_two.leaf_active))
        if int(level_two.levels[slot]) == 2 and int(level_two.prefixes[slot]) == 3
    )
    transition = _refine(plan, level_two, boundary_grandchild)
    assert bool(transition.accepted_candidate)
    assert int(transition.evidence.balance_refinements) > 0
    assert bool(transition.accepted.evidence.two_to_one_balanced)
