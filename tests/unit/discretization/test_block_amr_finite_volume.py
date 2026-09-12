#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization.finite_volume._amr import BlockAMRConservationPlan
from phydrax.discretization.finite_volume._block_amr import (
    BlockAMRFiniteVolumePlan,
)
from phydrax.discretization.finite_volume._dynamics import PreparedFiniteVolumeDynamics


def _prepared(*, periodic, capacity=3, levels=1, halo=1):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    level_plans = [phx.discretization.BlockLevelPlan(0, (4,), capacity, halo_width=halo)]
    if levels == 2:
        level_plans.append(phx.discretization.BlockLevelPlan(1, (2,), 8, halo_width=halo))
    hierarchy = phx.discretization.BlockHierarchyPlan(grid, level_plans)
    return phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()


def _system():
    return phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="block-amr-unit-advection",
    )


def _method():
    return phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )


def _state(topology, values):
    levels = tuple(
        phx.discretization.BlockLevelState(plan, metadata, value)
        for plan, metadata, value in zip(
            topology.plan.levels, topology.levels, values, strict=True
        )
    )
    return phx.discretization.BlockHierarchyState(topology, levels)


def test_one_level_periodic_blocks_match_structured_finite_volume_residual():
    prepared = _prepared(periodic=True)
    topology = prepared.initial_topology()
    global_state = jnp.asarray(
        [[0.2], [0.6], [-0.1], [0.3], [0.8], [0.4], [-0.2], [0.5]],
        dtype=jnp.float64,
    )
    block_values = jnp.full((3, 4, 1), jnp.nan, dtype=jnp.float64)
    block_values = block_values.at[:2].set(global_state.reshape((2, 4, 1)))
    state = _state(topology, (block_values,))
    boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x",))
    dynamics = BlockAMRFiniteVolumePlan(
        prepared, _system(), _method(), boundaries
    ).prepare(topology)

    fill = dynamics.fill_patch(0.0, state)
    result = dynamics.evaluate(0.0, state, fill)
    structured_geometry = phx.discretization.FiniteVolumePlan(
        topology.plan.grid
    ).prepare()
    structured = PreparedFiniteVolumeDynamics(
        _system(), structured_geometry, _method(), boundaries
    )
    expected = structured(jnp.asarray(0.0), global_state)

    np.testing.assert_allclose(
        result.residuals[0][:2].reshape(global_state.shape), expected
    )
    np.testing.assert_allclose(
        result.ledger.scatter_content_rate(),
        jnp.concatenate((expected * 0.125, jnp.zeros((4, 1))), axis=0),
    )
    assert result.ledger.topology_epoch_id == topology.epoch.epoch_id
    assert result.ledger.evidence_policy_id == dynamics.plan.precision.policy_id


def test_constant_periodic_state_is_zero_and_shared_faces_cancel_once():
    prepared = _prepared(periodic=True)
    topology = prepared.initial_topology()
    values = jnp.full((3, 4, 1), jnp.nan, dtype=jnp.float64)
    values = values.at[:2].set(2.5)
    state = _state(topology, (values,))
    dynamics = BlockAMRFiniteVolumePlan(
        prepared,
        _system(),
        _method(),
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    ).prepare(topology)

    result = dynamics.evaluate(0.0, state, dynamics.fill_patch(0.0, state))

    np.testing.assert_array_equal(result.residuals[0], np.zeros((3, 4, 1)))
    same_level = [
        block for block in result.ledger.blocks if block.block_kind == "same-level"
    ]
    assert len(same_level) == 1
    assert same_level[0].flux_rate.shape[0] == 8
    np.testing.assert_allclose(jnp.sum(result.ledger.scatter_content_rate(), axis=0), 0.0)


def test_physical_boundary_callbacks_are_not_used_on_interblock_faces():
    calls = []

    def target(time, interior, coordinates, outward_normal, args):
        del time, args
        calls.append((np.asarray(coordinates), np.asarray(outward_normal)))
        return interior

    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.PrescribedStateBoundary(target, boundary_id="left-count"),
        phx.discretization.PrescribedStateBoundary(target, boundary_id="right-count"),
    )
    prepared = _prepared(periodic=False)
    topology = prepared.initial_topology()
    values = jnp.full((3, 4, 1), jnp.nan, dtype=jnp.float64).at[:2].set(1.0)
    state = _state(topology, (values,))
    dynamics = BlockAMRFiniteVolumePlan(
        prepared,
        _system(),
        _method(),
        phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,)),
    ).prepare(topology)

    fill = dynamics.fill_patch(0.0, state)
    result = dynamics.evaluate(0.0, state, fill)

    assert len(calls) == 2
    assert all(points.shape == (1, 1) for points, _ in calls)
    assert sum(block.flux_rate.shape[0] for block in result.ledger.blocks) == 9
    assert (
        sum(
            block.flux_rate.shape[0]
            for block in result.ledger.blocks
            if block.block_kind == "physical"
        )
        == 2
    )


def test_inactive_nonfinite_payload_is_inert_and_fine_interfaces_are_distinct():
    prepared = _prepared(periodic=False, levels=2)
    initial = prepared.initial_topology()
    tags = jnp.zeros((3, 4), dtype=bool).at[0, 1].set(True)
    topology = prepared.compile_topology(initial, (tags,)).topology
    coarse = jnp.full((3, 4, 1), jnp.nan, dtype=jnp.float64).at[:2].set(1.0)
    fine = jnp.full((8, 2, 1), jnp.nan, dtype=jnp.float64)
    fine_count = int(jnp.sum(topology.levels[1].active))
    fine = fine.at[:fine_count].set(1.0)
    state = _state(topology, (coarse, fine))
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    dynamics = BlockAMRFiniteVolumePlan(
        prepared,
        _system(),
        _method(),
        phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,)),
    ).prepare(topology)

    result = dynamics.evaluate(0.0, state, dynamics.fill_patch(0.0, state))

    assert dynamics.coarse_fine_route_pairs == (
        (
            "block-amr:transition-0-1:axis-0:coarse",
            "block-amr:transition-0-1:axis-0:fine",
        ),
    )
    coarse_route, fine_route = dynamics.coarse_fine_route_pairs[0]
    routed = {block.block_id: block for block in result.ledger.blocks}
    assert routed[coarse_route].block_kind == "coarse-fine"
    assert routed[fine_route].block_kind == "coarse-fine"
    assert np.all(np.asarray(routed[coarse_route].neighbour_cells) >= 0)
    assert np.all(np.asarray(routed[fine_route].neighbour_cells) == -1)
    np.testing.assert_array_equal(result.residuals[0][2], np.zeros((4, 1)))
    np.testing.assert_array_equal(
        result.residuals[1][fine_count:], np.zeros((8 - fine_count, 2, 1))
    )
    assert np.all(np.isfinite(np.asarray(result.ledger.scatter_content_rate())))


def test_covered_cell_restriction_is_volume_weighted_and_leaves_uncovered_cells():
    prepared = _prepared(periodic=False, levels=2)
    initial = prepared.initial_topology()
    tags = jnp.zeros((3, 4), dtype=bool).at[0, 1].set(True)
    topology = prepared.compile_topology(initial, (tags,)).topology
    synchronization = BlockAMRConservationPlan(prepared, topology)
    coarse = jnp.ones((3, 4, 1), dtype=jnp.float64)
    fine = jnp.zeros((8, 2, 1), dtype=jnp.float64)
    fine_count = int(jnp.sum(topology.levels[1].active))
    fine = fine.at[:fine_count].set(7.0)

    restricted = synchronization.restrict_covered(coarse, fine, 0)
    covered = np.asarray(synchronization.covered_cell_mask(0))

    np.testing.assert_allclose(np.asarray(restricted)[covered], 7.0)
    active_uncovered = (
        np.broadcast_to(np.asarray(topology.levels[0].active)[:, None], covered.shape)
        & ~covered
    )
    np.testing.assert_allclose(np.asarray(restricted)[active_uncovered], 1.0)
    np.testing.assert_array_equal(restricted[2], np.zeros((4, 1)))
