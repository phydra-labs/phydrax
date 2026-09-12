#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from enum import IntEnum


class GravitationalWaveStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_WAVEFORM = 1
    OUTSIDE_WAVEFORM_SUPPORT = 2
    INVALID_RESPONSE = 3
    MARGINALIZATION_FAILURE = 4
    APPROXIMATION_FAILURE = 5


_MESSAGES = {
    GravitationalWaveStatus.SUCCESS: "successful",
    GravitationalWaveStatus.NONFINITE_WAVEFORM: "waveform contains non-finite values",
    GravitationalWaveStatus.OUTSIDE_WAVEFORM_SUPPORT: "parameters are outside waveform support",
    GravitationalWaveStatus.INVALID_RESPONSE: "detector response is invalid",
    GravitationalWaveStatus.MARGINALIZATION_FAILURE: "nuisance marginalization failed",
    GravitationalWaveStatus.APPROXIMATION_FAILURE: "likelihood approximation failed",
}


def gravitational_wave_status_message(status: int | GravitationalWaveStatus, /) -> str:
    return _MESSAGES[GravitationalWaveStatus(int(status))]


__all__ = ["GravitationalWaveStatus", "gravitational_wave_status_message"]
