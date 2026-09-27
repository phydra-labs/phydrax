#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import pytest

from phydrax.qualification import (
    core_candidate_profiles,
    core_portfolio_observation,
    CoreQualificationObservation,
)


def _passing(profile: Any) -> Any:
    return CoreQualificationObservation(
        profile,
        {gate: True for gate in profile.required_gates},
        {"bounded_metric": 0.0},
    )


def test_core_contracts() -> None:
    profiles = core_candidate_profiles()
    record = core_portfolio_observation(tuple(_passing(profile) for profile in profiles))

    assert len(profiles) == 10
    assert record["passed"] is True
    # ty: ignore[invalid-argument-type]
    assert len(record["observations"]) == len(profiles)
    assert record["portfolio_id"]
    profile = core_candidate_profiles()[0]
    observation = _passing(profile)

    with pytest.raises(ValueError, match="every exact profile once"):
        core_portfolio_observation((observation, observation))
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
