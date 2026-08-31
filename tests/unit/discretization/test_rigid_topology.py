#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._rigid_body import RigidBodySetPlan
from phydrax.discretization.particle._rigid_joints import (
    FixedJointSetPlan,
    RigidJointGraphPlan,
)
from phydrax.discretization.particle._rigid_topology import (
    apply_rigid_topology_transactions,
    BreakableRigidJointLawPlan,
    RigidTopologyEventKind,
    RigidTopologyFailure,
    RigidTopologyPlan,
    update_breakable_rigid_joints,
)


def _bodies(*, active_mask=(True, True, True)):
    count = len(active_mask)
    identifiers = jnp.arange(100, 100 + count, dtype=jnp.int64)
    particles = ParticleSetPlan(
        identifiers,
        jnp.ones((count,)),
        ambient_dimension=3,
        active_mask=jnp.asarray(active_mask),
    ).prepare()
    bodies = RigidBodySetPlan(
        jnp.zeros((count,), dtype=jnp.int32),
        jnp.broadcast_to(jnp.eye(3), (count, 3, 3)),
        fixed_mask=jnp.asarray([True] + [False] * (count - 1)),
    ).prepare(particles)
    orientation = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (count, 4))
    reference = bodies.kinematics(
        jnp.stack(
            (
                jnp.arange(count, dtype=jnp.float64),
                jnp.zeros((count,)),
                jnp.zeros((count,)),
            ),
            axis=-1,
        ),
        jnp.zeros((count, 3)),
        orientation,
        jnp.zeros((count, 3)),
    )
    return identifiers, bodies, reference


def _two_joint_topology(*, event_capacity=8, initial_active=(True, True), plan_id=None):
    identifiers, bodies, reference = _bodies()
    graph = RigidJointGraphPlan(
        fixed=FixedJointSetPlan(
            jnp.asarray([20, 10], dtype=jnp.int64),
            identifiers[jnp.asarray([0, 0])],
            identifiers[jnp.asarray([1, 2])],
        )
    ).prepare(bodies, reference)
    law = BreakableRigidJointLawPlan(
        jnp.asarray([20, 10], dtype=jnp.int64),
        jnp.ones((2,)),
        2.0 * jnp.ones((2,)),
        jnp.asarray([3.0, 5.0]),
        arming_loading=0.5,
        minimum_loading_rate=0.1,
        initial_active_mask=jnp.asarray(initial_active),
    )
    plan = RigidTopologyPlan(
        law,
        jnp.empty((0,), dtype=jnp.int64),
        event_capacity=event_capacity,
        plan_id=plan_id,
    )
    return plan.prepare(bodies, graph)


def _no_transactions(prepared, state):
    return prepared.proposal(
        jnp.zeros((prepared.plan.transaction_capacity,), dtype=bool),
        state.replay_digest,
    )


def test_break_is_one_time_irreversible_and_dissipation_is_monotone():
    law = BreakableRigidJointLawPlan(
        jnp.asarray([7]),
        jnp.asarray([1.0]),
        jnp.asarray([2.0]),
        jnp.asarray([4.0]),
        arming_loading=jnp.asarray([0.5]),
        minimum_loading_rate=jnp.asarray([0.1]),
    )
    initial = law.initialize_state()
    broken = update_breakable_rigid_joints(
        law,
        initial,
        jnp.asarray([2.1]),
        jnp.asarray([1.0]),
        jnp.asarray(4),
    )
    assert broken.successful
    assert broken.newly_broken_mask.tolist() == [True]
    assert broken.accepted_state.active_mask.tolist() == [False]
    assert broken.accepted_state.damage.tolist() == [1.0]
    assert broken.accepted_state.cumulative_fracture_dissipation.tolist() == [4.0]
    assert broken.accepted_state.break_step.tolist() == [4]
    assert broken.accepted_state.break_event_id.tolist() == [0]

    repeated = update_breakable_rigid_joints(
        law,
        broken.accepted_state,
        jnp.asarray([5.0]),
        jnp.asarray([2.0]),
        jnp.asarray(5),
    )
    assert repeated.successful
    assert repeated.newly_broken_mask.tolist() == [False]
    assert repeated.accepted_state.damage.tolist() == [1.0]
    assert repeated.accepted_state.cumulative_fracture_dissipation.tolist() == [4.0]
    assert repeated.accepted_state.break_step.tolist() == [4]
    assert repeated.accepted_state.break_event_id.tolist() == [0]
    assert repeated.accepted_state.next_event_id == 1


def test_unload_arms_high_initial_load_and_reload_crossing_breaks():
    law = BreakableRigidJointLawPlan(
        jnp.asarray([8]),
        1.0,
        2.0,
        3.0,
        arming_loading=0.5,
        minimum_loading_rate=0.1,
    )
    initial = law.initialize_state(initial_loading=jnp.asarray([1.5]))
    assert initial.armed.tolist() == [False]
    unloaded = update_breakable_rigid_joints(
        law,
        initial,
        jnp.asarray([0.25]),
        jnp.asarray([-1.0]),
        jnp.asarray(1),
    )
    assert unloaded.successful
    assert unloaded.accepted_state.armed.tolist() == [True]
    assert unloaded.accepted_state.active_mask.tolist() == [True]
    assert unloaded.accepted_state.damage.tolist() == [0.5]
    reloaded = update_breakable_rigid_joints(
        law,
        unloaded.accepted_state,
        jnp.asarray([2.25]),
        jnp.asarray([0.5]),
        jnp.asarray(2),
    )
    assert reloaded.successful
    assert reloaded.newly_broken_mask.tolist() == [True]
    assert reloaded.accepted_state.damage.tolist() == [1.0]


def test_simultaneous_breaks_are_journaled_in_stable_id_order():
    prepared = _two_joint_topology()
    state = prepared.initialize_state()
    result = apply_rigid_topology_transactions(
        prepared,
        state,
        _no_transactions(prepared, state),
        jnp.asarray([2.5, 2.5]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray(6),
    )
    assert result.successful
    assert result.proposed_events.entity_ids[result.proposed_events.valid].tolist() == [
        10,
        20,
    ]
    journal = result.accepted_state.journal
    assert journal.event_ids[journal.valid].tolist() == [0, 1]
    assert journal.entity_ids[journal.valid].tolist() == [10, 20]
    assert journal.event_kinds[journal.valid].tolist() == [
        int(RigidTopologyEventKind.JOINT_BREAK),
        int(RigidTopologyEventKind.JOINT_BREAK),
    ]
    assert result.accepted_state.joint_state.break_event_id.tolist() == [1, 0]
    assert result.accepted_state.replay_digest != state.replay_digest


def test_event_capacity_overflow_rolls_back_entire_composite():
    prepared = _two_joint_topology(event_capacity=1)
    state = prepared.initialize_state()
    result = apply_rigid_topology_transactions(
        prepared,
        state,
        _no_transactions(prepared, state),
        jnp.asarray([2.5, 2.5]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray(3),
    )
    assert not result.successful
    assert result.rejection.event_capacity_overflow
    assert result.rejection.failure_reasons & int(
        RigidTopologyFailure.EVENT_CAPACITY_OVERFLOW
    )
    assert jnp.array_equal(
        result.accepted_state.joint_state.active_mask, state.joint_state.active_mask
    )
    assert jnp.array_equal(result.accepted_state.journal.valid, state.journal.valid)
    assert result.accepted_state.replay_digest == state.replay_digest
    assert not jnp.any(result.multiplier_reset_joint_mask)


def test_inactive_dual_gauge_and_multiplier_reset_follow_foundation_layout():
    prepared = _two_joint_topology()
    state = prepared.initialize_state()
    result = apply_rigid_topology_transactions(
        prepared,
        state,
        _no_transactions(prepared, state),
        jnp.asarray([2.5, 0.0]),
        jnp.asarray([1.0, 0.0]),
        jnp.asarray(1),
    )
    assert result.successful
    assert result.accepted_state.joint_state.active_mask.tolist() == [False, True]
    assert result.dual_gauge.row_layout.layout_id == prepared.joints.row_layout.layout_id
    expected_reset = prepared.joints.row_layout.row_active(jnp.asarray([True, False]))
    assert jnp.array_equal(result.multiplier_reset_row_mask, expected_reset)
    assert jnp.array_equal(result.dual_gauge.inactive_row_mask, expected_reset)
    assert jnp.array_equal(
        result.dual_gauge.gauge_diagonal, expected_reset.astype(jnp.float64)
    )
    assert jnp.all(result.dual_gauge.gauge_rhs == 0.0)
    assert result.dual_gauge.finite_evidence


def test_predeclared_joint_activation_and_body_successor_transaction():
    prepared = _two_joint_topology(initial_active=(False, True))
    activation_plan = RigidTopologyPlan(
        prepared.plan.breakable_joints,
        jnp.asarray([500]),
        activated_joint_ids=jnp.asarray([[20]]),
        event_capacity=4,
    ).prepare(prepared.bodies, prepared.joints)
    state = activation_plan.initialize_state()
    result = apply_rigid_topology_transactions(
        activation_plan,
        state,
        activation_plan.proposal(jnp.asarray([True]), state.replay_digest),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(2),
    )
    assert result.successful
    assert result.accepted_state.joint_state.active_mask.tolist() == [True, True]
    assert result.accepted_state.contact_cache_epoch == 1
    assert result.multiplier_reset_joint_mask.tolist() == [True, False]
    assert result.accepted_state.journal.entity_ids[0] == 20
    assert result.accepted_state.journal.event_kinds[0] == int(
        RigidTopologyEventKind.JOINT_ACTIVATION
    )

    body_ids, bodies, reference = _bodies(active_mask=(True, True, False))
    graph = RigidJointGraphPlan(
        fixed=FixedJointSetPlan(
            jnp.asarray([30]), jnp.asarray([body_ids[0]]), jnp.asarray([body_ids[1]])
        )
    ).prepare(bodies, reference)
    law = BreakableRigidJointLawPlan(
        jnp.asarray([30]), 1.0, 2.0, 1.0, minimum_loading_rate=0.1
    )
    body_plan = RigidTopologyPlan(
        law,
        jnp.asarray([600]),
        predecessor_body_ids=jnp.asarray([[101]]),
        successor_body_ids=jnp.asarray([[102]]),
        deactivated_joint_ids=jnp.asarray([[30]]),
        event_capacity=4,
    ).prepare(bodies, graph)
    body_state = body_plan.initialize_state()
    assert body_plan.predecessor_body_indices.tolist() == [[1]]
    assert body_plan.successor_body_indices.tolist() == [[2]]
    body_result = apply_rigid_topology_transactions(
        body_plan,
        body_state,
        body_plan.proposal(jnp.asarray([True]), body_state.replay_digest),
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        jnp.asarray(3),
    )
    assert body_result.successful
    assert body_result.accepted_state.body_active_mask.tolist() == [True, False, True]
    assert body_result.accepted_state.joint_state.active_mask.tolist() == [False]
    assert body_result.accepted_state.contact_cache_epoch == 1


def test_prepared_identity_and_replay_digest_are_enforced():
    prepared = _two_joint_topology()
    alternate = _two_joint_topology(plan_id="different-topology")
    state = prepared.initialize_state()
    alternate_state = alternate.initialize_state()
    with pytest.raises(ValueError, match="another prepared topology"):
        apply_rigid_topology_transactions(
            alternate,
            state,
            _no_transactions(alternate, alternate_state),
            jnp.zeros((2,)),
            jnp.zeros((2,)),
            jnp.asarray(0),
        )
    mismatch = prepared.proposal(
        jnp.zeros((prepared.plan.transaction_capacity,), dtype=bool),
        state.replay_digest + 1,
    )
    rejected = apply_rigid_topology_transactions(
        prepared,
        state,
        mismatch,
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.asarray(0),
    )
    assert not rejected.successful
    assert rejected.rejection.replay_digest_mismatch
    assert rejected.rejection.failure_reasons & int(
        RigidTopologyFailure.REPLAY_DIGEST_MISMATCH
    )
    assert rejected.accepted_state.replay_digest == state.replay_digest
    assert jnp.array_equal(
        rejected.accepted_state.joint_state.damage, state.joint_state.damage
    )


def test_jit_scan_preserves_all_fixed_capacities():
    prepared = _two_joint_topology(event_capacity=4)
    initial = prepared.initialize_state()
    loading = jnp.asarray([[2.5, 0.0], [3.0, 0.0], [4.0, 0.0]])
    derivative = jnp.ones_like(loading)

    def run(state):
        def step(carry, inputs):
            load, rate, index = inputs
            transition = apply_rigid_topology_transactions(
                prepared,
                carry,
                _no_transactions(prepared, carry),
                load,
                rate,
                index,
            )
            return transition.accepted_state, transition

        return jax.lax.scan(
            step,
            state,
            (loading, derivative, jnp.arange(loading.shape[0], dtype=jnp.int32)),
        )

    final, transitions = jax.jit(run)(initial)
    assert final.joint_state.active_mask.tolist() == [False, True]
    assert final.journal.valid.shape == (4,)
    assert transitions.proposed_events.valid.shape == (
        loading.shape[0],
        prepared.maximum_proposal_count,
    )
    assert transitions.multiplier_reset_row_mask.shape == (
        loading.shape[0],
        prepared.constraint_row_capacity,
    )
    assert jnp.sum(final.journal.valid) == 1


def test_invalid_derivative_margins_and_runtime_derivatives_are_rejected():
    with pytest.raises(ValueError, match="derivative margins"):
        BreakableRigidJointLawPlan(
            jnp.asarray([1]), 1.0, 2.0, 1.0, minimum_loading_rate=0.0
        )
    with pytest.raises(ValueError, match="derivative margins"):
        BreakableRigidJointLawPlan(
            jnp.asarray([1]), 1.0, 2.0, 1.0, minimum_loading_rate=jnp.nan
        )
    law = BreakableRigidJointLawPlan(
        jnp.asarray([1]), 1.0, 2.0, 1.0, minimum_loading_rate=0.1
    )
    state = law.initialize_state()
    rejected = update_breakable_rigid_joints(
        law,
        state,
        jnp.asarray([2.5]),
        jnp.asarray([jnp.nan]),
        jnp.asarray(1),
    )
    assert not rejected.successful
    assert not rejected.finite_evidence
    assert rejected.failure_reasons & int(RigidTopologyFailure.INVALID_DERIVATIVE)
    assert jnp.array_equal(rejected.accepted_state.damage, state.damage)
    assert jnp.array_equal(rejected.accepted_state.active_mask, state.active_mask)
