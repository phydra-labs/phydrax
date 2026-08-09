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
    APPROXIMATION_FAILED = 5
    ZERO_KERNEL_ROW = 6
    INFEASIBLE_SUPPORT = 7
    INTEGRATION_FAILURE = 8
    MASS_MISMATCH = 9
    SUPPORT_COLLAPSE = 10
    TRANSPORT_MASS_COLLAPSED = 11


_STATUS_MESSAGES = {
    TransportStatus.CONVERGED: "converged",
    TransportStatus.MAXIMUM_ITERATIONS_REACHED: "maximum iterations reached",
    TransportStatus.MARGINAL_STAGNATION: "marginal residual stagnated",
    TransportStatus.NONFINITE_ITERATE: "transport iteration produced non-finite values",
    TransportStatus.NONFINITE_OBJECTIVE: "transport objective is non-finite",
    TransportStatus.APPROXIMATION_FAILED: "kernel approximation failed validation",
    TransportStatus.ZERO_KERNEL_ROW: "kernel approximation has an active zero row",
    TransportStatus.INFEASIBLE_SUPPORT: "endpoint support is unreachable under the reference process",
    TransportStatus.INTEGRATION_FAILURE: "continuous integration failed",
    TransportStatus.MASS_MISMATCH: "continuous and finite physical masses differ",
    TransportStatus.SUPPORT_COLLAPSE: "free barycenter support collapsed",
    TransportStatus.TRANSPORT_MASS_COLLAPSED: "transported mass collapsed below the declared threshold",
}


def status_message(status: int | TransportStatus, /) -> str:
    """Return a stable human-readable transport status description."""
    return _STATUS_MESSAGES[TransportStatus(int(status))]


__all__ = ["TransportStatus", "status_message"]
