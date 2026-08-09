#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from enum import IntEnum


class TransportStatus(IntEnum):
    """JAX-compatible terminal status codes for transport algorithms."""

    CONVERGED = 0
    MAXIMUM_ITERATIONS_REACHED = 1
    MARGINAL_STAGNATION = 2
    NONFINITE_ITERATE = 3
    NONFINITE_OBJECTIVE = 4


_STATUS_MESSAGES = {
    TransportStatus.CONVERGED: "converged",
    TransportStatus.MAXIMUM_ITERATIONS_REACHED: "maximum iterations reached",
    TransportStatus.MARGINAL_STAGNATION: "marginal residual stagnated",
    TransportStatus.NONFINITE_ITERATE: "transport iteration produced non-finite values",
    TransportStatus.NONFINITE_OBJECTIVE: "transport objective is non-finite",
}


def status_message(status: int | TransportStatus, /) -> str:
    """Return a stable human-readable transport status description."""
    return _STATUS_MESSAGES[TransportStatus(int(status))]


__all__ = ["TransportStatus", "status_message"]
