#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from enum import IntEnum


class AstrophysicsObservationStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    INVALID_GEOMETRY = 2
    OUTSIDE_SUPPORT = 3
    NONPHYSICAL_MODEL = 4
    INCOMPATIBLE_CONTEXT = 5


_MESSAGES = {
    AstrophysicsObservationStatus.SUCCESS: "successful",
    AstrophysicsObservationStatus.NONFINITE_INPUT: "input contains non-finite values",
    AstrophysicsObservationStatus.INVALID_GEOMETRY: "observation geometry is invalid",
    AstrophysicsObservationStatus.OUTSIDE_SUPPORT: "requested coordinates are outside support",
    AstrophysicsObservationStatus.NONPHYSICAL_MODEL: "the observation model is nonphysical",
    AstrophysicsObservationStatus.INCOMPATIBLE_CONTEXT: "observation contexts are incompatible",
}


def astrophysics_observation_status_message(
    status: int | AstrophysicsObservationStatus, /
) -> str:
    return _MESSAGES[AstrophysicsObservationStatus(int(status))]


__all__ = [
    "AstrophysicsObservationStatus",
    "astrophysics_observation_status_message",
]
