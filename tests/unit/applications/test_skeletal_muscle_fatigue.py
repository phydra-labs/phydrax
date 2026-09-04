#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from phydrax.applications.skeletal_muscle.fatigue import (
    commit_liu_brown_yue_2002,
    LiuBrownYue2002Parameters,
    LiuBrownYue2002Plan,
    LiuBrownYue2002Status,
)


def _prepared():
    return LiuBrownYue2002Plan(
        LiuBrownYue2002Parameters(
            fatigue_rate_per_s=0.0206,
            recovery_rate_per_s=0.0084,
        ),
        muscle_id="test-handgrip",
        protocol_id="sustained-then-zero-effort",
    ).prepare()


def _fractions(state):
    return np.asarray(
        [state.uncommitted_fraction, state.active_fraction, state.fatigued_fraction]
    )


def test_exact_steps_conserve_all_motor_unit_compartments():
    prepared = _prepared()
    state = prepared.initialize()
    for _ in range(240):
        candidate = prepared.evaluate(state, 1.0, 0.5)
        assert bool(candidate.evidence.successful)
        state = commit_liu_brown_yue_2002(candidate, state)
        np.testing.assert_allclose(_fractions(state).sum(), 1.0, atol=2.0e-7)
        assert _fractions(state).min() >= -2.0e-7

    capacity = prepared.capacity(state)
    assert 0.0 < float(capacity.active_relative_force) < 1.0
    assert 0.0 < float(capacity.fatigued_fraction) < 1.0


def test_zero_effort_recovery_transfers_fatigued_units_to_active_units():
    prepared = _prepared()
    state = prepared.initialize(
        uncommitted_fraction=0.0,
        active_fraction=0.2,
        fatigued_fraction=0.8,
    )
    before = _fractions(state)
    candidate = prepared.evaluate(state, 0.0, 5.0)
    after = _fractions(candidate.proposed)

    assert bool(candidate.evidence.successful)
    assert after[2] < before[2]
    assert after[1] > before[1]
    np.testing.assert_allclose(after[1] + after[2], before[1] + before[2], atol=2e-7)
    np.testing.assert_allclose(after.sum(), before.sum(), atol=2e-7)


def test_invalid_brain_effort_rolls_back_every_compartment():
    prepared = _prepared()
    state = prepared.initialize(
        uncommitted_fraction=0.5,
        active_fraction=0.3,
        fatigued_fraction=0.2,
    )
    candidate = prepared.evaluate(state, -1.0, 1.0)

    assert not bool(candidate.evidence.successful)
    assert int(candidate.evidence.status) & int(
        LiuBrownYue2002Status.NEGATIVE_BRAIN_EFFORT
    )
    rolled_back = commit_liu_brown_yue_2002(candidate, state)
    np.testing.assert_array_equal(_fractions(rolled_back), _fractions(state))
    assert float(rolled_back.time_s) == float(state.time_s)
    assert int(rolled_back.step_index) == int(state.step_index)


def test_piecewise_constant_update_agrees_with_small_step_limit():
    prepared = _prepared()
    single = prepared.initialize()
    single_candidate = prepared.evaluate(single, 0.7, 4.0)
    single = commit_liu_brown_yue_2002(single_candidate, single)

    refined = prepared.initialize()
    for _ in range(400):
        candidate = prepared.evaluate(refined, 0.7, 0.01)
        refined = commit_liu_brown_yue_2002(candidate, refined)

    np.testing.assert_allclose(_fractions(single), _fractions(refined), rtol=2e-5, atol=2e-6)
