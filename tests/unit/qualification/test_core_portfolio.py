#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.qualification import (
    core_candidate_profiles,
    core_portfolio_observation,
    CoreQualificationObservation,
)


def _passing(profile):
    return CoreQualificationObservation(
        profile,
        {gate: True for gate in profile.required_gates},
        {"bounded_metric": 0.0},
    )


def test_core_portfolio_covers_each_exact_family_once() -> None:
    profiles = core_candidate_profiles()
    record = core_portfolio_observation(tuple(_passing(profile) for profile in profiles))

    assert len(profiles) == 10
    assert record["passed"] is True
    assert len(record["observations"]) == len(profiles)
    assert record["portfolio_id"]


def test_core_observation_requires_every_gate_and_finite_metric() -> None:
    profile = core_candidate_profiles()[0]

    with pytest.raises(ValueError, match="every required gate"):
        CoreQualificationObservation(
            profile,
            {profile.required_gates[0]: True},
            {"bounded_metric": 0.0},
        )
    with pytest.raises(ValueError, match="finite"):
        CoreQualificationObservation(
            profile,
            {gate: True for gate in profile.required_gates},
            {"bounded_metric": float("nan")},
        )


def test_core_portfolio_rejects_partial_or_duplicate_coverage() -> None:
    profile = core_candidate_profiles()[0]
    observation = _passing(profile)

    with pytest.raises(ValueError, match="every exact profile once"):
        core_portfolio_observation((observation, observation))
