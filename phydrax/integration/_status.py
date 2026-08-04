#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from enum import IntEnum


class IntegrationStatus(IntEnum):
    """JAX-compatible terminal status codes for integration algorithms."""

    CONVERGED = 0
    MAXIMUM_INTERVALS_REACHED = 1
    MAXIMUM_EVALUATIONS_REACHED = 2
    NONFINITE_INTEGRAND = 3
    REFINEMENT_STAGNATION = 4
    INVALID_BOUNDS = 5
    INVALID_NORMALIZATION_MASS = 6
    PROPOSAL_SUPPORT_FAILURE = 7
    UNSAMPLED_STRATUM = 8
    INVALID_WEIGHTS = 9
    NO_VALID_SAMPLES = 10


_STATUS_MESSAGES = {
    IntegrationStatus.CONVERGED: "converged",
    IntegrationStatus.MAXIMUM_INTERVALS_REACHED: "maximum intervals reached",
    IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED: "maximum evaluations reached",
    IntegrationStatus.NONFINITE_INTEGRAND: "integrand produced non-finite values",
    IntegrationStatus.REFINEMENT_STAGNATION: "adaptive refinement stagnated",
    IntegrationStatus.INVALID_BOUNDS: "integration bounds are invalid",
    IntegrationStatus.INVALID_NORMALIZATION_MASS: "normalization mass is zero or invalid",
    IntegrationStatus.PROPOSAL_SUPPORT_FAILURE: "proposal support does not cover target mass",
    IntegrationStatus.UNSAMPLED_STRATUM: "a positive-measure stratum received no samples",
    IntegrationStatus.INVALID_WEIGHTS: "integration weights are invalid",
    IntegrationStatus.NO_VALID_SAMPLES: "no valid samples remain after masking",
}


def status_message(status: int | IntegrationStatus, /) -> str:
    """Return a stable human-readable status description."""
    return _STATUS_MESSAGES[IntegrationStatus(int(status))]


__all__ = ["IntegrationStatus", "status_message"]
