#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from enum import IntEnum


class AstrodynamicsStatus(IntEnum):
    """JAX-compatible status codes for astrodynamics kernels."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    INVALID_DOMAIN = 2
    NONCONVERGED = 3
    SINGULAR_GEOMETRY = 4
    COLLISION = 5
    NO_SOLUTION = 6
    CAPACITY_EXCEEDED = 7
    INCOMPATIBLE_CONTEXT = 8
    UNSUPPORTED_REGIME = 9


_STATUS_MESSAGES = {
    AstrodynamicsStatus.SUCCESS: "successful",
    AstrodynamicsStatus.NONFINITE_INPUT: "input contains non-finite values",
    AstrodynamicsStatus.INVALID_DOMAIN: "input is outside the physical domain",
    AstrodynamicsStatus.NONCONVERGED: "bounded numerical iteration did not converge",
    AstrodynamicsStatus.SINGULAR_GEOMETRY: "geometry is singular or ambiguous",
    AstrodynamicsStatus.COLLISION: "a collision singularity was encountered",
    AstrodynamicsStatus.NO_SOLUTION: "no requested solution exists",
    AstrodynamicsStatus.CAPACITY_EXCEEDED: "static result capacity was exceeded",
    AstrodynamicsStatus.INCOMPATIBLE_CONTEXT: "physical contexts are incompatible",
    AstrodynamicsStatus.UNSUPPORTED_REGIME: "the requested physical regime is unsupported",
}


def astrodynamics_status_message(status: int | AstrodynamicsStatus, /) -> str:
    """Return a stable host-readable status description."""

    return _STATUS_MESSAGES[AstrodynamicsStatus(int(status))]


__all__ = ["AstrodynamicsStatus", "astrodynamics_status_message"]
