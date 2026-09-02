#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.particle._relations import (
    DynamicPairRelationPlan,
    make_pair_relation_events,
    PairRelationEventKind,
    PairRelationStatus,
    PairSpringPlan,
)


def _runtime(*, relation_capacity: int = 2, event_capacity: int = 1):
    plan = DynamicPairRelationPlan(
        np.zeros((4,), dtype=np.int32),
        relation_capacity,
        2,
        symmetric_kinds=np.ones((1,), dtype=bool),
        event_capacity=event_capacity,
    )
    return plan.prepare()


def _bind(runtime, state, left, right, *, event_id, parameters=(2.0, 1.0)):
    events = make_pair_relation_events(
        runtime.event_capacity,
        runtime.parameter_width,
        event_ids=[event_id],
        event_kind=[PairRelationEventKind.BIND],
        left=[left],
        right=[right],
        relation_kind=[0],
        parameters=[parameters],
    )
    return runtime.apply(state, events)


def test_relation_ids_are_stable_and_old_incarnations_are_stale():
    runtime = _runtime(relation_capacity=1)
    empty = runtime.initialize()
    first = _bind(runtime, empty, 0, 1, event_id=10)
    assert bool(first.successful)
    relation_id = int(first.accepted_state.relation_ids[0])
    incarnation = int(first.accepted_state.incarnations[0])

    unbind = make_pair_relation_events(
        1,
        2,
        event_ids=[11],
        event_kind=[PairRelationEventKind.UNBIND],
        relation_ids=[relation_id],
        relation_incarnations=[incarnation],
    )
    pending = runtime.evaluate(first.accepted_state, unbind)
    removed = runtime.apply(first.accepted_state, unbind)
    assert bool(removed.successful)
    second = _bind(runtime, removed.accepted_state, 2, 3, event_id=12)
    assert bool(second.successful)
    assert int(second.accepted_state.relation_ids[0]) == relation_id
    assert int(second.accepted_state.incarnations[0]) == incarnation + 1
    obsolete = runtime.commit(second.accepted_state, pending)
    assert not bool(obsolete.successful)
    assert not bool(obsolete.evidence.source_state_match)
    assert int(obsolete.evidence.stale_identity_count) == 1
    np.testing.assert_array_equal(
        obsolete.accepted_state.left, second.accepted_state.left
    )

    stale = runtime.apply(second.accepted_state, unbind)
    assert not bool(stale.successful)
    assert int(stale.evidence.event_status[0]) == PairRelationStatus.STALE_IDENTITY
    assert int(stale.evidence.stale_identity_count) == 1
    np.testing.assert_array_equal(
        stale.accepted_state.occupied, second.accepted_state.occupied
    )


def test_capacity_and_duplicate_fail_closed_atomically():
    runtime = _runtime(relation_capacity=1)
    first = _bind(runtime, runtime.initialize(), 0, 1, event_id=1)
    overflow = _bind(runtime, first.accepted_state, 2, 3, event_id=2)
    assert not bool(overflow.successful)
    assert int(overflow.evidence.overflow_count) == 1
    assert int(overflow.evidence.event_status[0]) == PairRelationStatus.CAPACITY_EXCEEDED
    np.testing.assert_array_equal(overflow.accepted_state.left, first.accepted_state.left)

    pair_runtime = _runtime(relation_capacity=2, event_capacity=2)
    duplicate_events = make_pair_relation_events(
        2,
        2,
        event_ids=[20, 21],
        event_kind=[PairRelationEventKind.BIND, PairRelationEventKind.BIND],
        left=[0, 1],
        right=[1, 0],
        relation_kind=[0, 0],
        parameters=[[1.0, 0.5], [1.0, 0.5]],
    )
    duplicate = pair_runtime.apply(pair_runtime.initialize(), duplicate_events)
    assert not bool(duplicate.successful)
    assert int(duplicate.evidence.duplicate_count) == 1
    assert not bool(jnp.any(duplicate.accepted_state.occupied))
    assert bool(duplicate.candidate_state.occupied[0])


def test_endpoint_compatibility_and_exclusion_have_distinct_evidence():
    compatibility = np.zeros((1, 2, 2), dtype=bool)
    compatibility[0, 0, 1] = True
    runtime = DynamicPairRelationPlan(
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        2,
        2,
        compatibility=compatibility,
        exclusion=np.ones((1, 1), dtype=bool),
        event_capacity=1,
    ).prepare()
    invalid = _bind(runtime, runtime.initialize(), 0, 1, event_id=1)
    assert not bool(invalid.successful)
    assert int(invalid.evidence.invalid_endpoint_count) == 1

    first = _bind(runtime, runtime.initialize(), 0, 2, event_id=2)
    excluded = _bind(runtime, first.accepted_state, 0, 3, event_id=3)
    assert not bool(excluded.successful)
    assert int(excluded.evidence.exclusion_count) == 1


def test_deactivate_move_and_reactivate_preserve_identity():
    runtime = _runtime(relation_capacity=1)
    bound = _bind(runtime, runtime.initialize(), 0, 1, event_id=1)
    relation_id = bound.accepted_state.relation_ids[0]
    incarnation = bound.accepted_state.incarnations[0]

    deactivate = make_pair_relation_events(
        1,
        2,
        event_ids=[2],
        event_kind=[PairRelationEventKind.DEACTIVATE],
        relation_ids=[relation_id],
        relation_incarnations=[incarnation],
    )
    dormant = runtime.apply(bound.accepted_state, deactivate)
    assert bool(dormant.successful)
    assert bool(dormant.accepted_state.occupied[0])
    assert not bool(dormant.accepted_state.active[0])

    move = make_pair_relation_events(
        1,
        2,
        event_ids=[3],
        event_kind=[PairRelationEventKind.MOVE],
        relation_ids=[relation_id],
        relation_incarnations=[incarnation],
        left=[2],
        right=[3],
    )
    moved = runtime.apply(dormant.accepted_state, move)
    activate = make_pair_relation_events(
        1,
        2,
        event_ids=[4],
        event_kind=[PairRelationEventKind.ACTIVATE],
        relation_ids=[relation_id],
        relation_incarnations=[incarnation],
    )
    active = runtime.apply(moved.accepted_state, activate)
    assert bool(active.successful)
    assert bool(active.accepted_state.active[0])
    assert int(active.accepted_state.left[0]) == 2
    assert int(active.accepted_state.right[0]) == 3
    assert int(active.accepted_state.incarnations[0]) == int(incarnation)


def test_structural_failure_is_not_misreported_as_nonfinite():
    runtime = _runtime(relation_capacity=1)
    bound = _bind(runtime, runtime.initialize(), 0, 1, event_id=1)
    empty = make_pair_relation_events(1, 2)
    evaluation = runtime.evaluate(
        bound.accepted_state,
        empty,
        endpoint_active=jnp.asarray([False, True, True, True]),
    )
    assert not bool(evaluation.evidence.successful)
    assert bool(evaluation.evidence.finite)
    assert int(evaluation.evidence.nonfinite_count) == 0
    assert int(evaluation.evidence.invalid_state_count) == 1


def test_relation_age_requires_a_scalar_time_step():
    runtime = _runtime(relation_capacity=1)
    with pytest.raises(ValueError, match="dt must be scalar"):
        runtime.advance_age(runtime.initialize(), jnp.ones((1,)))


def test_pair_spring_force_is_negative_energy_gradient():
    runtime = _runtime(relation_capacity=1)
    bound = _bind(runtime, runtime.initialize(), 0, 1, event_id=1, parameters=(3.0, 1.0))
    spring = PairSpringPlan().prepare(runtime, ambient_dimension=2)
    positions = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    evaluation = spring.evaluate(bound.accepted_state, positions)
    gradient = jax.grad(lambda value: spring.energy(bound.accepted_state, value))(
        positions
    )
    assert bool(evaluation.successful)
    np.testing.assert_allclose(evaluation.forces, -gradient, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(evaluation.energy, 1.5, rtol=1.0e-6)
    np.testing.assert_allclose(jnp.sum(evaluation.forces, axis=0), 0.0, atol=1.0e-6)
