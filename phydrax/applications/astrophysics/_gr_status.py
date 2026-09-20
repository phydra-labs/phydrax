#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class GRRayStatus(IntEnum):
    """JAX-compatible terminal statuses for relativistic trajectories."""

    SUCCESS = 0
    CAPTURED = 1
    ESCAPED = 2
    DOMAIN_EXIT = 3
    WORK_EXHAUSTED = 4
    NONFINITE = 5
    INVALID_INITIAL_STATE = 6
    CONSTRAINT_VIOLATION = 7
    NUMERICAL_FAILURE = 8
    INACTIVE = 9


_STATUS_MESSAGES = {
    GRRayStatus.SUCCESS: "trajectory reached its requested endpoint",
    GRRayStatus.CAPTURED: "trajectory crossed the ordered capture surface",
    GRRayStatus.ESCAPED: "trajectory crossed the ordered escape surface",
    GRRayStatus.DOMAIN_EXIT: "trajectory left the metric chart domain",
    GRRayStatus.WORK_EXHAUSTED: "trajectory exhausted its fixed affine or solver budget",
    GRRayStatus.NONFINITE: "trajectory produced non-finite numerical state",
    GRRayStatus.INVALID_INITIAL_STATE: "initial trajectory state is not physically admissible",
    GRRayStatus.CONSTRAINT_VIOLATION: "trajectory mass-shell residual exceeds tolerance",
    GRRayStatus.NUMERICAL_FAILURE: "differential solver failed before a terminal event",
    GRRayStatus.INACTIVE: "fixed-capacity trajectory lane is inactive",
}


def gr_ray_status_message(status: int | GRRayStatus, /) -> str:
    """Return the stable host-readable description of one ray status."""

    return _STATUS_MESSAGES[GRRayStatus(int(status))]


class GRRayStatusEvidence(StrictModule):
    """Independent scientific gates for a scalar or batched ray result."""

    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array

    def __init__(
        self,
        finite: ArrayLike,
        converged: ArrayLike,
        physically_valid: ArrayLike,
        qualified: ArrayLike,
        derivative_valid: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(value, dtype=jnp.bool_)
            for value in (
                finite,
                converged,
                physically_valid,
                qualified,
                derivative_valid,
            )
        )
        shape = values[0].shape
        if any(value.shape != shape for value in values[1:]):
            raise ValueError("GR ray scientific status gates must have equal shapes.")
        (
            self.finite,
            self.converged,
            self.physically_valid,
            self.qualified,
            self.derivative_valid,
        ) = values


__all__ = [
    "GRRayStatus",
    "GRRayStatusEvidence",
    "gr_ray_status_message",
]
