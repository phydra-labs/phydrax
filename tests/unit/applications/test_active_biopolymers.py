#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.applications.cellular_mechanics._active_polymers import (
    ActinNetworkPlan,
    ChromatinDynamicsPlan,
    FocalAdhesionPlan,
    MotorCrosslinkerPlan,
)


def test_chromatin_joint_occupancy_and_extrusion_collision_evidence():
    positions = np.arange(8, dtype=float)[:, None]
    runtime = ChromatinDynamicsPlan(
        positions,
        2,
        roadblocks=np.asarray([False, False, False, False, False, False, False, True]),
        spring_rest_length=1.0,
    ).prepare()
    state = runtime.initialize(
        left=np.asarray([3, 6], dtype=np.int32),
        right=np.asarray([1, 4], dtype=np.int32),
    )
    observables = runtime.observables(state)
    assert int(observables.loop_count) == 2
    np.testing.assert_array_equal(
        observables.occupied_sites,
        np.asarray([False, True, False, True, True, False, True, False]),
    )
    assert bool(observables.roadblock_occupancy[7])
    np.testing.assert_array_equal(state.relations.left, np.asarray([1, 4]))
    np.testing.assert_array_equal(state.relations.right, np.asarray([3, 6]))

    result = runtime.extrude(state)
    assert bool(result.successful)
    assert int(result.evidence.collision_count) == 2
    assert int(result.evidence.extruded_count) == 0
    np.testing.assert_array_equal(
        result.accepted_state.relations.left, state.relations.left
    )
    np.testing.assert_array_equal(
        result.accepted_state.relations.right, state.relations.right
    )


def test_chromatin_direct_capture_canonicalizes_reversed_feet():
    runtime = ChromatinDynamicsPlan(
        np.arange(8, dtype=float)[:, None],
        1,
        capture_distance=8.0,
    ).prepare()
    state = runtime.initialize()
    captured = runtime.bind(state, 6, 2, event_id=8)
    assert bool(captured.successful)
    assert int(captured.accepted_state.left[0]) == 2
    assert int(captured.accepted_state.right[0]) == 6


def test_chromatin_addressed_step_replays_identically():
    runtime = ChromatinDynamicsPlan(
        np.arange(12, dtype=float)[:, None],
        4,
        binding_rate=4.0,
        unbinding_rate=0.2,
        extrusion_rate=0.5,
        capture_distance=4.0,
    ).prepare()
    state = runtime.initialize()
    key = jr.key(91)
    left = runtime.step(state, key, 0.25)
    with pytest.raises(ValueError, match="dt must be scalar"):
        runtime.step(state, key, jnp.ones((1,)))
    right = runtime.step(state, key, 0.25)
    np.testing.assert_array_equal(
        left.accepted_state.relations.left, right.accepted_state.relations.left
    )
    np.testing.assert_array_equal(
        left.accepted_state.relations.incarnations,
        right.accepted_state.relations.incarnations,
    )
    np.testing.assert_array_equal(
        left.evidence.relation.event_status, right.evidence.relation.event_status
    )


def test_actin_growth_turnover_and_severing_conserve_mass_and_lineage():
    runtime = ActinNetworkPlan(
        6,
        6,
        ambient_dimension=2,
        initial_monomer_pool=10.0,
        monomer_mass=1.0,
        segment_length=1.0,
    ).prepare()
    initial = runtime.initialize(np.asarray([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="dt must be scalar"):
        runtime.step(initial, jr.key(0), jnp.ones((1,)))
    total = runtime.total_mass(initial)

    grown = runtime.polymerize(initial, 0, jnp.asarray([1.0, 0.0]), event_id=1)
    assert bool(grown.successful)
    branched = runtime.branch(
        grown.accepted_state, 0, jnp.asarray([0.0, 1.0]), event_id=2
    )
    assert bool(branched.successful)
    np.testing.assert_allclose(runtime.total_mass(branched.accepted_state), total)
    assert int(jnp.sum(branched.accepted_state.node_active)) == 3
    assert int(branched.accepted_state.lineage_id[1]) == 0
    assert int(branched.accepted_state.lineage_id[2]) == 0

    relation = branched.accepted_state.relations
    severed = runtime.sever(
        branched.accepted_state,
        relation.relation_ids[0],
        relation.incarnations[0],
        event_id=3,
    )
    assert bool(severed.successful)
    assert int(severed.accepted_state.lineage_id[1]) != 0
    assert int(severed.accepted_state.lineage_id[2]) == 0
    np.testing.assert_allclose(runtime.total_mass(severed.accepted_state), total)
    assert bool(severed.evidence.lineage_valid)

    depolymerized = runtime.depolymerize(severed.accepted_state, 2, event_id=4)
    assert bool(depolymerized.successful)
    np.testing.assert_allclose(runtime.total_mass(depolymerized.accepted_state), total)
    assert int(jnp.sum(depolymerized.accepted_state.node_active)) == 2
    np.testing.assert_allclose(depolymerized.evidence.mass_residual, 0.0, atol=1.0e-6)


def test_motor_endpoint_step_preserves_left_foot_and_moves_right_foot():
    runtime = MotorCrosslinkerPlan(
        4,
        1,
        ambient_dimension=1,
        stepping_rate=100.0,
        stall_force=100.0,
    ).prepare()
    state = runtime.initialize(
        np.float32,
        left=np.asarray([0]),
        right=np.asarray([2]),
        motor=np.asarray([True]),
    )
    result = runtime.step(
        state,
        jr.key(4),
        1.0,
        jnp.arange(4, dtype=jnp.float32)[:, None],
        jnp.asarray([1, 2, 3, -1]),
    )
    assert bool(result.successful)
    assert int(result.evidence.stepped_count) == 1
    assert int(result.accepted_state.relations.left[0]) == 0
    assert int(result.accepted_state.relations.right[0]) == 3
    blocked = runtime.step(
        state,
        jr.key(4),
        1.0,
        jnp.arange(4, dtype=jnp.float32)[:, None],
        jnp.asarray([1, 2, 4, -1]),
    )
    assert bool(blocked.successful)
    assert int(blocked.evidence.stepped_count) == 0
    assert int(blocked.evidence.endpoint_blocked_count) == 1
    assert int(blocked.accepted_state.relations.right[0]) == 2
    with pytest.raises(ValueError, match="dt must be scalar"):
        runtime.step(
            state,
            jr.key(4),
            jnp.ones((1,)),
            jnp.arange(4, dtype=jnp.float32)[:, None],
            jnp.asarray([1, 2, 3, -1]),
        )


def test_focal_adhesion_traction_is_energy_derived_and_balanced():
    runtime = FocalAdhesionPlan(
        2,
        2,
        1,
        ambient_dimension=2,
        spring_stiffness=2.0,
        rest_length=1.0,
    ).prepare()
    state = runtime.initialize(
        np.float32,
        cell_endpoints=np.asarray([0]),
        substrate_endpoints=np.asarray([0]),
    )
    cell_positions = jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    substrate_positions = jnp.asarray([[2.0, 0.0], [2.0, 1.0]], dtype=jnp.float32)
    evidence = runtime.traction(state, cell_positions, substrate_positions)
    assert bool(evidence.successful)
    np.testing.assert_allclose(evidence.springs.energy, 1.0, rtol=1.0e-6)
    np.testing.assert_allclose(
        evidence.cell_traction,
        jnp.asarray([[2.0, 0.0], [0.0, 0.0]]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(evidence.total_traction, jnp.asarray([2.0, 0.0]))
    np.testing.assert_allclose(jnp.sum(evidence.springs.forces, axis=0), 0.0, atol=1.0e-6)
    with pytest.raises(ValueError, match="dt must be scalar"):
        runtime.step(
            state,
            jr.key(0),
            jnp.ones((1,)),
            cell_positions,
            substrate_positions,
        )
