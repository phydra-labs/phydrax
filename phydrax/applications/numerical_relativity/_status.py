#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class NumericalRelativityStatus(IntFlag):
    """Composable fail-closed status bits for fixed-grid spacetime evolution."""

    SUCCESS = 0
    NONFINITE_STATE = 1
    NONPOSITIVE_LAPSE = 2
    NONPOSITIVE_CONFORMAL_FACTOR = 4
    SINGULAR_CONFORMAL_METRIC = 8
    CONSTRAINT_TOLERANCE_EXCEEDED = 16
    BOUNDARY_FAILURE = 32
    ENFORCEMENT_FAILURE = 64
    DERIVATIVE_INVALID = 128
    TIME_GRID_MISMATCH = 256
    SOURCE_INVALID = 512
    STEP_REJECTED = 1024


_STATUS_MESSAGES = {
    NumericalRelativityStatus.NONFINITE_STATE: "state contains non-finite values",
    NumericalRelativityStatus.NONPOSITIVE_LAPSE: "lapse is not everywhere positive",
    NumericalRelativityStatus.NONPOSITIVE_CONFORMAL_FACTOR: (
        "conformal factor is not everywhere positive"
    ),
    NumericalRelativityStatus.SINGULAR_CONFORMAL_METRIC: (
        "conformal metric is singular or orientation reversing"
    ),
    NumericalRelativityStatus.CONSTRAINT_TOLERANCE_EXCEEDED: (
        "constraint tolerance was exceeded"
    ),
    NumericalRelativityStatus.BOUNDARY_FAILURE: "boundary treatment failed",
    NumericalRelativityStatus.ENFORCEMENT_FAILURE: (
        "accepted-step algebraic enforcement failed"
    ),
    NumericalRelativityStatus.DERIVATIVE_INVALID: (
        "the requested derivative does not have valid numerical evidence"
    ),
    NumericalRelativityStatus.TIME_GRID_MISMATCH: (
        "logical time does not match the fixed temporal grid"
    ),
    NumericalRelativityStatus.SOURCE_INVALID: "stress-energy projection is invalid",
    NumericalRelativityStatus.STEP_REJECTED: "candidate step was rejected",
}


def numerical_relativity_status_message(
    status: int | NumericalRelativityStatus, /
) -> str:
    """Return a stable host-readable message for one or several status bits."""

    value = NumericalRelativityStatus(int(status))
    if value == NumericalRelativityStatus.SUCCESS:
        return "successful"
    return "; ".join(
        message for flag, message in _STATUS_MESSAGES.items() if value & flag
    )


class ScientificStatus(StrictModule):
    """Orthogonal numerical and scientific predicates for one result."""

    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array

    def __init__(
        self,
        status: ArrayLike,
        finite: ArrayLike,
        converged: ArrayLike,
        physically_valid: ArrayLike,
        qualified: ArrayLike,
        derivative_valid: ArrayLike,
        /,
    ):
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())
        self.finite = jnp.asarray(finite, dtype=jnp.bool_).reshape(())
        self.converged = jnp.asarray(converged, dtype=jnp.bool_).reshape(())
        self.physically_valid = jnp.asarray(physically_valid, dtype=jnp.bool_).reshape(())
        self.qualified = jnp.asarray(qualified, dtype=jnp.bool_).reshape(())
        self.derivative_valid = jnp.asarray(derivative_valid, dtype=jnp.bool_).reshape(())

    @property
    def successful(self) -> Array:
        return (
            (self.status == int(NumericalRelativityStatus.SUCCESS))
            & self.finite
            & self.converged
            & self.physically_valid
            & self.qualified
            & self.derivative_valid
        )


__all__ = [
    "NumericalRelativityStatus",
    "ScientificStatus",
    "numerical_relativity_status_message",
]
