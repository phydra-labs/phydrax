#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Portable step status for hard-label threshold dynamics."""

from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
from jax import Array


class ThresholdDynamicsStatus(IntEnum):
    """Outcome of one threshold-dynamics step, ordered by increasing severity.

    ``SUCCESS`` and ``UNDER_RESOLVED`` commit the candidate labels; every other
    status rolls the step back and returns the input state unchanged.
    """

    SUCCESS = 0
    UNDER_RESOLVED = 1
    ENERGY_INCREASE = 2
    VOLUME_CONSTRAINT_FAILED = 3
    NONFINITE = 4
    HEAT_ACTION_FAILED = 5
    INADMISSIBLE_PARAMETERS = 6
    CANDIDATE_OVERFLOW = 7


_STATUS_MESSAGES = {
    ThresholdDynamicsStatus.SUCCESS: "threshold step committed",
    ThresholdDynamicsStatus.UNDER_RESOLVED: (
        "threshold step committed with a kernel width below the declared grid "
        "resolution ratio; interfaces may pin"
    ),
    ThresholdDynamicsStatus.ENERGY_INCREASE: (
        "discrete energy increased although dissipation is admitted; step rolled back"
    ),
    ThresholdDynamicsStatus.VOLUME_CONSTRAINT_FAILED: (
        "capacitated label assignment failed; step rolled back"
    ),
    ThresholdDynamicsStatus.INADMISSIBLE_PARAMETERS: (
        "tension or mobility values are not admissible for the kernel; step rolled back"
    ),
    ThresholdDynamicsStatus.CANDIDATE_OVERFLOW: (
        "sparse candidate-label or site capacity overflowed; step rolled back"
    ),
    ThresholdDynamicsStatus.HEAT_ACTION_FAILED: (
        "heat-kernel action failed its native status; step rolled back"
    ),
    ThresholdDynamicsStatus.NONFINITE: "non-finite kernel values; step rolled back",
}


def threshold_dynamics_status_message(status: int | ThresholdDynamicsStatus, /) -> str:
    """Return the stable message for one threshold-dynamics status."""
    return _STATUS_MESSAGES[ThresholdDynamicsStatus(int(status))]


def _combine_status(
    *,
    nonfinite: Array,
    heat_failed: Array,
    overflow: Array,
    inadmissible: Array,
    volume_failed: Array,
    energy_increase: Array,
    under_resolved: Array,
) -> Array:
    """Most severe status among device failure flags (fixed precedence)."""
    # Root causes (capacity overflow, inadmissible parameters, a failed native heat
    # action) invalidate every downstream potential, so they outrank the
    # non-finite values they cause.
    ordered = (
        (overflow, ThresholdDynamicsStatus.CANDIDATE_OVERFLOW),
        (inadmissible, ThresholdDynamicsStatus.INADMISSIBLE_PARAMETERS),
        (heat_failed, ThresholdDynamicsStatus.HEAT_ACTION_FAILED),
        (nonfinite, ThresholdDynamicsStatus.NONFINITE),
        (volume_failed, ThresholdDynamicsStatus.VOLUME_CONSTRAINT_FAILED),
        (energy_increase, ThresholdDynamicsStatus.ENERGY_INCREASE),
        (under_resolved, ThresholdDynamicsStatus.UNDER_RESOLVED),
    )
    status = jnp.asarray(int(ThresholdDynamicsStatus.SUCCESS), dtype=jnp.int32)
    for flag, value in reversed(ordered):
        status = jnp.where(flag, jnp.asarray(int(value), dtype=jnp.int32), status)
    return status


def _committed(status: Array, /) -> Array:
    return status <= int(ThresholdDynamicsStatus.UNDER_RESOLVED)


__all__ = ["ThresholdDynamicsStatus", "threshold_dynamics_status_message"]
