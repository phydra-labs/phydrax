#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.applications.skeletal_muscle.motor_units import (
    commit_fuglevand_winter_patla_1993,
    fuglevand_force_variability_evidence,
    FuglevandWinterPatla1993Plan,
    FuglevandWinterPatla1993RandomInput,
    FuglevandWinterPatla1993Status,
)


def _prepared(*, capacity: int = 8):
    return FuglevandWinterPatla1993Plan(
        32,
        event_capacity_per_unit=capacity,
        random_stream_id="unit-test-discharge",
    ).prepare()


def _random(state, key=jr.key(17)):
    return FuglevandWinterPatla1993RandomInput(
        key,
        state.random_step,
        stream_id="unit-test-discharge",
    )


def test_source_distributions_and_truncated_normal_statistics():
    prepared = _prepared(capacity=32)
    state = prepared.initialize()
    candidate = prepared.evaluate(
        state,
        prepared.maximum_excitation,
        5.0,
        _random(state),
    )
    scores = np.asarray(candidate.evidence.normal_scores).reshape(-1)

    assert np.isclose(
        float(prepared.recruitment_threshold_excitation[-1]), 30.0, rtol=1.0e-6
    )
    assert np.isclose(
        float(prepared.peak_twitch_force_arbitrary[-1]), 100.0, rtol=1.0e-6
    )
    assert np.isclose(float(prepared.contraction_time_ms[-1]), 30.0, rtol=1.0e-6)
    assert scores.min() >= -3.9
    assert scores.max() <= 3.9
    assert abs(scores.mean()) < 0.1
    assert abs(scores.std(ddof=1) - 1.0) < 0.1


def test_stochastic_step_replays_exactly_and_commits_once():
    prepared = _prepared()
    state = prepared.initialize()
    random_input = _random(state)
    first = prepared.evaluate(state, prepared.maximum_excitation, 10.0, random_input)
    replay = prepared.evaluate(state, prepared.maximum_excitation, 10.0, random_input)

    assert bool(first.evidence.successful)
    np.testing.assert_array_equal(first.evidence.event_mask, replay.evidence.event_mask)
    np.testing.assert_array_equal(
        first.evidence.event_times_ms, replay.evidence.event_times_ms
    )
    np.testing.assert_array_equal(
        first.proposed.motor_unit_force, replay.proposed.motor_unit_force
    )
    committed = commit_fuglevand_winter_patla_1993(first, state)
    assert int(committed.random_step) == 1
    stale = commit_fuglevand_winter_patla_1993(replay, committed)
    np.testing.assert_array_equal(stale.motor_unit_force, committed.motor_unit_force)
    assert int(stale.random_step) == 1


def test_event_overflow_rolls_back_whole_state_including_rng_counter():
    prepared = _prepared(capacity=1)
    state = prepared.initialize()
    candidate = prepared.evaluate(
        state,
        prepared.maximum_excitation,
        1000.0,
        _random(state),
    )

    assert not bool(candidate.evidence.successful)
    assert int(candidate.evidence.status) & int(
        FuglevandWinterPatla1993Status.EVENT_CAPACITY_OVERFLOW
    )
    rolled_back = commit_fuglevand_winter_patla_1993(candidate, state)
    assert int(rolled_back.random_step) == 0
    assert float(rolled_back.time_ms) == 0.0
    np.testing.assert_array_equal(rolled_back.motor_unit_force, state.motor_unit_force)


def test_force_trace_has_finite_nonzero_variability():
    prepared = _prepared(capacity=4)
    state = prepared.initialize()
    key = jr.key(101)
    values = []
    for _ in range(500):
        candidate = prepared.evaluate(
            state,
            0.65 * prepared.maximum_excitation,
            5.0,
            _random(state, key),
        )
        assert bool(candidate.evidence.successful)
        state = commit_fuglevand_winter_patla_1993(candidate, state)
        values.append(prepared.force(state).total_force_arbitrary)
    evidence = fuglevand_force_variability_evidence(jnp.stack(values[100:]))

    assert bool(evidence.finite)
    assert float(evidence.mean_force_arbitrary) > 0.0
    assert float(evidence.standard_deviation_force_arbitrary) > 0.0
    assert float(evidence.coefficient_of_variation) > 0.0


def test_event_topology_and_times_are_excluded_from_ad():
    prepared = _prepared()
    state = prepared.initialize()
    random_input = _random(state)

    def endpoint_force(excitation):
        candidate = prepared.evaluate(state, excitation, 10.0, random_input)
        return candidate.evidence.total_force_arbitrary

    derivative = jax.grad(endpoint_force)(prepared.maximum_excitation)
    assert float(derivative) == 0.0
    assert prepared.evaluate(
        state, prepared.maximum_excitation, 10.0, random_input
    ).evidence.topology_gradient_supported is False
