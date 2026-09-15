#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.discretization import polygonal_cell_complex, prepare_cell_boundary_paths
from phydrax.graph import MatrixGaugeLinkSpace
from phydrax.graph._gauge_transport import GaugeStaplePlan
from phydrax.metrix import SpecialUnitaryGroup, UnitaryGroup
from phydrax.sampling._gauge_updates import (
    gauge_replica_exchange,
    gauge_update_sweeps,
    GaugeReplicaExchangePlan,
    GaugeUpdatePlan,
    initialize_gauge_replica_state,
    initialize_gauge_update_state,
    prepare_gauge_update,
)


def _prepared(group, kind="heatbath", attempts=128):
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    boundaries = prepare_cell_boundary_paths(topology)
    space = MatrixGaugeLinkSpace(topology, group)
    staples = GaugeStaplePlan(space, boundaries)
    colors = jnp.arange(space.num_edges)
    conflicts = ~jnp.eye(space.num_edges, dtype=bool)
    name = "u1" if group.dimension == 1 else f"su{group.dimension}"
    plan = GaugeUpdatePlan(
        name,
        coupling=0.8,
        update=kind,
        rejection_attempts=attempts,
    )
    return space, prepare_gauge_update(plan, staples, colors, conflicts)


def test_u1_exact_conditional_preserves_haar_support_and_detailed_balance():
    space, prepared = _prepared(UnitaryGroup(1), attempts=256)
    state = initialize_gauge_update_state(prepared, space.identity())
    result = gauge_update_sweeps(prepared, state, key=jax.random.key(11))

    evidence = result.evidence
    assert space.contains(result.state.links)
    assert jnp.all(evidence.membership_preserved)
    assert jnp.allclose(
        evidence.log_forward_reverse_ratio + evidence.log_target_ratio,
        0.0,
        atol=2e-6,
    )
    assert jnp.all(evidence.rejection_attempts <= prepared.rejection_attempts)
    assert jnp.all((evidence.status == 0) | (evidence.status == 1))
    assert result.reference_measure == "product-haar"


def test_su2_heatbath_and_overrelaxation_preserve_measure_support():
    space, heatbath = _prepared(SpecialUnitaryGroup(2), attempts=256)
    initial = initialize_gauge_update_state(heatbath, space.identity())
    sampled = gauge_update_sweeps(heatbath, initial, key=jax.random.key(12))
    assert space.contains(sampled.state.links)
    assert jnp.all(sampled.evidence.exact_target_correction)

    _, reflection = _prepared(SpecialUnitaryGroup(2), kind="overrelaxation")
    reflected_state = initialize_gauge_update_state(reflection, sampled.state.links)
    reflected = gauge_update_sweeps(reflection, reflected_state, key=jax.random.key(13))
    assert space.contains(reflected.state.links)
    assert jnp.allclose(reflected.evidence.log_target_ratio, 0.0, atol=3e-5)
    assert jnp.all(reflected.evidence.exact_target_correction)


def test_su3_cabibbo_marinari_subgroups_remain_special_unitary():
    space, prepared = _prepared(SpecialUnitaryGroup(3), attempts=256)
    state = initialize_gauge_update_state(prepared, space.identity())
    result = gauge_update_sweeps(prepared, state, key=jax.random.key(14))

    assert space.contains(result.state.links)
    assert result.evidence.link_index.shape == (3 * space.num_edges,)
    assert jnp.all(result.evidence.membership_preserved)
    assert jnp.allclose(
        result.evidence.log_forward_reverse_ratio + result.evidence.log_target_ratio,
        0.0,
        atol=3e-5,
    )


def test_invalid_coloring_and_group_combination_fail_during_prepare():
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    boundaries = prepare_cell_boundary_paths(topology)
    space = MatrixGaugeLinkSpace(topology, SpecialUnitaryGroup(2))
    staples = GaugeStaplePlan(space, boundaries)
    conflicts = ~jnp.eye(space.num_edges, dtype=bool)

    with pytest.raises(ValueError, match="sharing a sweep color conflict"):
        prepare_gauge_update(
            GaugeUpdatePlan("su2", coupling=1.0),
            staples,
            jnp.zeros(space.num_edges, dtype=int),
            conflicts,
        )
    with pytest.raises(ValueError, match="link count, coloring"):
        prepare_gauge_update(
            GaugeUpdatePlan("su3", coupling=1.0),
            staples,
            jnp.arange(space.num_edges),
            conflicts,
        )


def test_replica_exchange_reports_exact_balance_ratio_and_alternates_pairs():
    plan = GaugeReplicaExchangePlan(jnp.asarray([1.0, 0.7, 0.4]))
    configurations = jnp.arange(6.0).reshape((3, 2))
    reduced = jnp.asarray(
        [
            [0.0, 4.0, 8.0],
            [8.0, 0.0, 4.0],
            [4.0, 8.0, 0.0],
        ]
    )
    state = initialize_gauge_replica_state(plan, configurations, reduced)
    first = gauge_replica_exchange(plan, state, key=jax.random.key(15))
    second = gauge_replica_exchange(plan, first.state, key=jax.random.key(15))

    assert jnp.array_equal(first.attempted, jnp.asarray([True, False]))
    assert jnp.array_equal(second.attempted, jnp.asarray([False, True]))
    assert first.log_target_ratio[0] == -12.0
    assert jnp.allclose(first.detailed_balance_residual, 0.0)
    assert jnp.allclose(first.log_forward_reverse_ratio, 0.0)
    assert jnp.array_equal(first.status, jnp.asarray([0, 4], dtype=jnp.int32))
    assert first.state.attempted_swaps == 1
    assert second.state.attempted_swaps == 2
