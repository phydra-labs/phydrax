#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Terminal statuses and event kinds shared by radial bubble workflows."""

from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class BubbleDynamicsStatus(IntEnum):
    """Mutually exclusive terminal status of one bubble-dynamics solve.

    `SUCCESS`, `MINIMUM_RADIUS` and `DISSOLVED` are physical endpoints: the
    solve reached the requested final time, the requested minimum-radius
    stopping event, or complete dissolution. Every other member is a refusal
    or failure and the reported terminal state is the last accepted state.
    """

    SUCCESS = 0
    SOLVER_FAILURE = 1
    MAX_STEPS = 2
    MINIMUM_RADIUS = 3
    HARD_CORE = 4
    MACH_LIMIT = 5
    INVALID_STATE = 6
    SUPPORT_EXIT = 7
    REGIME_CAPACITY = 8
    INVALID_EQUILIBRIUM = 9
    VALIDITY_EXCEEDED = 10
    OVERLAP = 11
    DISSOLVED = 12
    COUPLING_FAILURE = 13
    ILL_CONDITIONED = 14
    HISTORY_CAPACITY = 15
    NEUTRAL_UNSTABLE = 16


class BubbleEventKind(IntEnum):
    """Event that terminated one integration segment."""

    NONE = 0
    MINIMUM_RADIUS = 1
    HARD_CORE = 2
    MACH_LIMIT = 3
    INVALID_STATE = 4
    SUPPORT_EXIT = 5
    REGIME_TRANSITION = 6
    DISSOLUTION = 7
    OVERLAP = 8
    COUPLING_FAILURE = 9
    ILL_CONDITIONED = 10
    SUPPORT_GROWTH = 11
    NEUTRAL_UNSTABLE = 12


def bubble_status_successful(status: ArrayLike, /) -> Array:
    """Return whether a status is a physical endpoint rather than a refusal."""
    status_ = jnp.asarray(status, dtype=jnp.int32)
    return (
        (status_ == int(BubbleDynamicsStatus.SUCCESS))
        | (status_ == int(BubbleDynamicsStatus.MINIMUM_RADIUS))
        | (status_ == int(BubbleDynamicsStatus.DISSOLVED))
    )


__all__ = ["BubbleDynamicsStatus", "BubbleEventKind", "bubble_status_successful"]
