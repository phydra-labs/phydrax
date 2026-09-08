#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications import electrophysiology as ep


def _runtime(*, weight=0.4, delay=0.2, execution="clock", two_relations=False):
    connections = [
        ep.SynapseConnection(
            "distal",
            0,
            0,
            1,
            1,
            ep.ConductanceSynapse(5.0, 0.2, -10.0),
            weight=weight,
            delay_ms=delay,
        )
    ]
    if two_relations:
        connections.append(
            ep.SynapseConnection(
                "proximal",
                0,
                0,
                1,
                0,
                ep.CurrentSynapse(5.0, -0.3),
                weight=weight,
                delay_ms=delay,
            )
        )
    return ep.SynapseNetworkPlan(
        (1, 2), 2, 0.5, 0.1, connections=connections, execution=execution
    ).prepare()


def _pair(**kwargs):
    return ep.PairSTDPPlan(20.0, 25.0, 0.1, 0.05, 0.0, 1.0, **kwargs)


def _relation_event(kind, *, slot=0, pre_compartment=0, delay=0.2):
    return ep.SynapseRelationEvent(
        int(kind),
        slot,
        0,
        pre_compartment,
        1,
        1,
        int(ep.SynapseKind.CONDUCTANCE),
        0.4,
        0.2,
        -10.0,
        5.0,
        delay,
    )


def _assert_same_tree(actual, expected):
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_physical_delays_reject_clock_quantization_but_allow_event_execution():
    with pytest.raises(ValueError, match="multiple"):
        _runtime(delay=0.15)
    event_runtime = _runtime(delay=0.15, execution="event")
    event_state = ep.initialize_synapse_network(event_runtime)
    event = _relation_event(ep.SynapseRelationEventKind.ACTIVATE, slot=1, delay=0.15)
    accepted = ep.evaluate_synapse_relation_event(
        event_runtime, event_state.relations, event
    )
    assert bool(accepted.successful)
    np.testing.assert_allclose(accepted.proposed.delay_ms[1], 0.15)

    clock_runtime = _runtime()
    clock_state = ep.initialize_synapse_network(clock_runtime)
    rejected = ep.evaluate_synapse_relation_event(
        clock_runtime, clock_state.relations, event
    )
    assert not bool(rejected.successful)
    assert int(rejected.status) == int(ep.SynapseStatus.INVALID_PARAMETER)
    _assert_same_tree(
        ep.commit_synapse_network_relation_event(rejected, clock_state), clock_state
    )
    # A compartment exists on cell 1 but not on cell 0; rectangular padding is invalid.
    invalid_endpoint = _relation_event(
        ep.SynapseRelationEventKind.ACTIVATE, slot=1, pre_compartment=1
    )
    rejected = ep.evaluate_synapse_relation_event(
        clock_runtime, clock_state.relations, invalid_endpoint
    )
    assert int(rejected.status) == int(ep.SynapseStatus.INVALID_ENDPOINT)


def test_clock_arrival_uses_emission_weight_despite_intervening_learning():
    runtime = _runtime()
    state = ep.initialize_synapse_network(runtime)
    pre = jnp.asarray([1.0, 0.0, 0.0])
    zero = jnp.zeros(3)
    emitted = ep.evaluate_synapse_network_transition(runtime, state, pre)
    state = ep.commit_synapse_network_transition(emitted, state)
    traces = ep.initialize_pair_stdp(runtime)
    pair = _pair()
    first = ep.evaluate_pair_stdp(
        runtime, pair, state.relations, traces, pre, zero, elapsed_ms=0.0
    )
    relations, traces = ep.commit_pair_stdp(first, state.relations, traces)
    second = ep.evaluate_pair_stdp(
        runtime,
        pair,
        relations,
        traces,
        zero,
        jnp.asarray([0.0, 0.0, 1.0]),
        elapsed_ms=0.1,
    )
    updated, _ = ep.commit_pair_stdp(second, relations, traces)
    assert float(updated.weight[0]) > 0.4
    np.testing.assert_array_equal(updated.generation, relations.generation)
    assert int(updated.relation_version[0]) == int(relations.relation_version[0]) + 1
    state = eqx.tree_at(lambda value: value.relations, state, updated)
    intermediate = ep.evaluate_synapse_network_transition(runtime, state, zero)
    state = ep.commit_synapse_network_transition(intermediate, state)
    arrived = ep.evaluate_synapse_network_transition(runtime, state, zero)
    np.testing.assert_allclose(arrived.proposed.relations.activation[0], 0.4)
    np.testing.assert_allclose(arrived.evidence.arrival_counts, [1.0, 0.0])
    np.testing.assert_allclose(arrived.evidence.conductance_uS, [0.0, 0.0, 0.08])
    np.testing.assert_allclose(arrived.evidence.current_offset_nA, [0.0, 0.0, 0.8])


def test_zero_weight_arrival_counts_still_form_learning_pairs():
    runtime = _runtime(weight=0.0)
    state = ep.initialize_synapse_network(runtime)
    pre = jnp.asarray([1.0, 0.0, 0.0])
    zero = jnp.zeros(3)
    traces = ep.initialize_pair_stdp(runtime)
    plan = _pair(pairing="arrival")
    for spikes in (pre, zero, zero):
        transition = ep.evaluate_synapse_network_transition(runtime, state, spikes)
        state = ep.commit_synapse_network_transition(transition, state)
        learning = ep.evaluate_pair_stdp(
            runtime,
            plan,
            state.relations,
            traces,
            spikes,
            zero,
            presynaptic_arrivals=transition.evidence.arrival_counts,
        )
        relations, traces = ep.commit_pair_stdp(learning, state.relations, traces)
        state = eqx.tree_at(lambda value: value.relations, state, relations)
    np.testing.assert_array_equal(state.relations.activation, [0.0, 0.0])
    rewarded = ep.evaluate_pair_stdp(
        runtime,
        plan,
        state.relations,
        traces,
        zero,
        jnp.asarray([0.0, 0.0, 1.0]),
        elapsed_ms=2.0,
        presynaptic_arrivals=jnp.zeros(2),
    )
    np.testing.assert_allclose(rewarded.relations.weight[0], 0.1 * np.exp(-2.0 / 20.0))


def test_arrival_pairing_reverses_causality_when_delay_crosses_post_spike():
    runtime = _runtime(execution="event")
    original = ep.initialize_synapse_network(runtime).relations
    pre = jnp.asarray([1.0, 0.0, 0.0])
    post = jnp.asarray([0.0, 0.0, 1.0])
    zero = jnp.zeros(3)
    weights = {}
    for pairing in ("emission", "arrival"):
        plan = _pair(pairing=pairing)
        relations = original
        traces = ep.initialize_pair_stdp(runtime)
        for pre_spikes, post_spikes, elapsed, arrivals in (
            (pre, zero, 0.0, jnp.zeros(2)),
            (zero, post, 5.0, jnp.zeros(2)),
            (zero, zero, 5.0, jnp.asarray([1.0, 0.0])),
        ):
            candidate = ep.evaluate_pair_stdp(
                runtime,
                plan,
                relations,
                traces,
                pre_spikes,
                post_spikes,
                elapsed_ms=elapsed,
                presynaptic_arrivals=arrivals,
            )
            relations, traces = ep.commit_pair_stdp(candidate, relations, traces)
        weights[pairing] = relations.weight[0]
    np.testing.assert_allclose(weights["emission"], 0.4 + 0.1 * np.exp(-5.0 / 20.0))
    np.testing.assert_allclose(weights["arrival"], 0.4 - 0.05 * np.exp(-5.0 / 25.0))


def test_soft_bound_pair_rule_uses_actual_elapsed_time_and_normalized_weight_distance():
    runtime = _runtime(weight=0.8, execution="event")
    relations = ep.initialize_synapse_network(runtime).relations
    traces = ep.initialize_pair_stdp(runtime)
    plan = _pair(weight_dependence="soft-bound", weight_exponent=2.0)
    zero = jnp.zeros(3)
    first = ep.evaluate_pair_stdp(
        runtime,
        plan,
        relations,
        traces,
        jnp.asarray([1.0, 0.0, 0.0]),
        zero,
        elapsed_ms=0.0,
    )
    relations, traces = ep.commit_pair_stdp(first, relations, traces)
    candidate = ep.evaluate_pair_stdp(
        runtime,
        plan,
        relations,
        traces,
        zero,
        jnp.asarray([0.0, 0.0, 1.0]),
        elapsed_ms=7.0,
    )
    expected = 0.8 + 0.1 * np.exp(-7.0 / 20.0) * 0.2**2
    np.testing.assert_allclose(candidate.relations.weight[0], expected)
    np.testing.assert_allclose(candidate.plasticity.pre_trace[0], np.exp(-7.0 / 20.0))


@pytest.mark.parametrize(
    ("scope", "modulation", "expected_factors"),
    [
        ("global", 2.0, [2.0, 2.0]),
        ("post", [0.0, -1.0, 2.0], [2.0, -1.0]),
        ("relation", [-1.0, 2.0], [-1.0, 2.0]),
    ],
)
def test_delayed_reward_routes_signed_credit_by_selected_modulation_scope(
    scope, modulation, expected_factors
):
    runtime = _runtime(two_relations=True, execution="event")
    relations = ep.initialize_synapse_network(runtime).relations
    traces = ep.initialize_eligibility_stdp(runtime)
    plan = ep.EligibilitySTDPPlan(_pair(), 50.0)
    zero = jnp.zeros(3)
    first = ep.evaluate_eligibility_stdp(
        runtime,
        plan,
        relations,
        traces,
        jnp.asarray([1.0, 0.0, 0.0]),
        zero,
        elapsed_ms=0.0,
    )
    relations, traces = ep.commit_eligibility_stdp(first, relations, traces)
    paired = ep.evaluate_eligibility_stdp(
        runtime,
        plan,
        relations,
        traces,
        zero,
        jnp.asarray([0.0, 1.0, 1.0]),
        elapsed_ms=5.0,
    )
    relations, traces = ep.commit_eligibility_stdp(paired, relations, traces)
    np.testing.assert_allclose(relations.weight, [0.4, 0.4])
    np.testing.assert_allclose(traces.eligibility, 0.1 * np.exp(-5.0 / 20.0))
    reward = ep.evaluate_eligibility_stdp(
        runtime,
        plan,
        relations,
        traces,
        zero,
        zero,
        elapsed_ms=30.0,
        modulation=jnp.asarray(modulation),
        modulation_scope=scope,
    )
    committed, _ = ep.commit_eligibility_stdp(reward, relations, traces)
    credit = 0.1 * np.exp(-5.0 / 20.0) * np.exp(-30.0 / 50.0)
    np.testing.assert_allclose(
        committed.weight, 0.4 + credit * np.asarray(expected_factors)
    )
    np.testing.assert_array_equal(committed.generation, relations.generation)


def test_rejected_reward_atomically_retains_weights_pair_traces_and_eligibility():
    runtime = _runtime(execution="event")
    relations = ep.initialize_synapse_network(runtime).relations
    traces = ep.initialize_eligibility_stdp(runtime)
    plan = ep.EligibilitySTDPPlan(_pair(), 50.0)
    zero = jnp.zeros(3)
    first = ep.evaluate_eligibility_stdp(
        runtime,
        plan,
        relations,
        traces,
        jnp.asarray([1.0, 0.0, 0.0]),
        zero,
        elapsed_ms=0.0,
    )
    relations, traces = ep.commit_eligibility_stdp(first, relations, traces)
    evaluate = jax.jit(
        lambda relation, state, reward: ep.evaluate_eligibility_stdp(
            runtime,
            plan,
            relation,
            state,
            zero,
            jnp.asarray([0.0, 0.0, 1.0]),
            elapsed_ms=2.0,
            modulation=reward,
        )
    )
    rejected = evaluate(relations, traces, jnp.asarray(jnp.nan))
    assert not bool(rejected.successful)
    assert int(rejected.evidence.status) == int(ep.SynapseStatus.NONFINITE)
    _assert_same_tree(
        ep.commit_eligibility_stdp(rejected, relations, traces), (relations, traces)
    )
    invalid_time = ep.evaluate_pair_stdp(
        runtime, _pair(), relations, traces.pair_state, zero, zero, elapsed_ms=-1.0
    )
    assert not bool(invalid_time.successful)
    _assert_same_tree(
        ep.commit_pair_stdp(invalid_time, relations, traces.pair_state),
        (relations, traces.pair_state),
    )


def test_delete_and_reuse_cancel_pending_deliveries_and_all_learning_credit():
    runtime = _runtime(two_relations=True)
    state = ep.initialize_synapse_network(runtime)
    traces = ep.initialize_eligibility_stdp(runtime)
    plan = ep.EligibilitySTDPPlan(_pair(), 50.0)
    zero = jnp.zeros(3)
    pre = jnp.asarray([1.0, 0.0, 0.0])
    emission = ep.evaluate_synapse_network_transition(runtime, state, pre)
    state = ep.commit_synapse_network_transition(emission, state)
    first = ep.evaluate_eligibility_stdp(
        runtime, plan, state.relations, traces, pre, zero, elapsed_ms=0.0
    )
    relations, traces = ep.commit_eligibility_stdp(first, state.relations, traces)
    paired = ep.evaluate_eligibility_stdp(
        runtime,
        plan,
        relations,
        traces,
        zero,
        jnp.asarray([0.0, 1.0, 1.0]),
        elapsed_ms=0.1,
    )
    relations, traces = ep.commit_eligibility_stdp(paired, relations, traces)
    state = eqx.tree_at(lambda value: value.relations, state, relations)
    delete = ep.evaluate_synapse_relation_event(
        runtime, state.relations, _relation_event(ep.SynapseRelationEventKind.DEACTIVATE)
    )
    deleted, cleared = ep.commit_synapse_network_relation_event_with_plasticity(
        delete, state, traces
    )
    np.testing.assert_array_equal(cleared.pair_state.pre_trace[0], 0.0)
    np.testing.assert_array_equal(cleared.pair_state.post_trace[0], 0.0)
    np.testing.assert_array_equal(cleared.eligibility[0], 0.0)
    np.testing.assert_array_equal(cleared.eligibility[1], traces.eligibility[1])
    reuse = ep.evaluate_synapse_relation_event(
        runtime, deleted.relations, _relation_event(ep.SynapseRelationEventKind.ACTIVATE)
    )
    reused, cleared = ep.commit_synapse_network_relation_event_with_plasticity(
        reuse, deleted, cleared
    )
    assert int(reused.relations.generation[0]) == int(state.relations.generation[0]) + 2
    for _ in range(2):
        transition = ep.evaluate_synapse_network_transition(runtime, reused, zero)
        reused = ep.commit_synapse_network_transition(transition, reused)
    np.testing.assert_allclose(reused.relations.activation, [0.0, 0.4])
    reward = ep.evaluate_eligibility_stdp(
        runtime,
        plan,
        reused.relations,
        cleared,
        zero,
        zero,
        elapsed_ms=0.2,
        modulation=1.0,
    )
    np.testing.assert_allclose(reward.relations.weight[0], 0.4)
    assert float(reward.relations.weight[1]) > 0.4

    # A separately retained old learning snapshot cannot credit a reused lifetime.
    stale = ep.evaluate_eligibility_stdp(
        runtime,
        plan,
        reused.relations,
        traces,
        zero,
        zero,
        elapsed_ms=0.2,
        modulation=1.0,
    )
    np.testing.assert_allclose(stale.relations.weight[0], 0.4)
