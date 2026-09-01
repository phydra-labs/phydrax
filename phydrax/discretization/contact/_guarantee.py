#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class ContactGuaranteeLevel(IntEnum):
    """Ordered strength of a contact-geometry or trajectory guarantee."""

    UNAVAILABLE = 0
    HEURISTIC = 1
    PRACTICAL_CONSERVATIVE = 2
    ENCLOSURE_CONSERVATIVE = 3
    ANALYTIC_CONSERVATIVE = 4
    ROUNDING_CERTIFIED = 5


class ContactCapability(IntFlag):
    NONE = 0
    STATIC_DISTANCE = 1 << 0
    LINEAR_TRAJECTORY = 1 << 1
    NONLINEAR_TRAJECTORY = 1 << 2
    DIFFERENTIABLE_KINEMATICS = 1 << 3
    FORCE_PULLBACK = 1 << 4
    REMESH_TRANSFER = 1 << 5
    DISTRIBUTED = 1 << 6
    GPU_COMPILED = 1 << 7


class ContactGuaranteeEvidence(StrictModule):
    """Machine-checkable guarantee strength and completion evidence."""

    level: Array
    required_level: Array
    finite: Array
    work_complete: Array
    successful: Array
    failure_code: Array
    margin: Array
    backend_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: ContactGuaranteeLevel | int,
        /,
        *,
        required_level: ContactGuaranteeLevel | int = ContactGuaranteeLevel.UNAVAILABLE,
        finite: ArrayLike = True,
        work_complete: ArrayLike = True,
        failure_code: ArrayLike = 0,
        margin: ArrayLike = jnp.inf,
        backend_id: str,
    ):
        level_ = jnp.asarray(int(level), dtype=jnp.int32)
        required = jnp.asarray(int(required_level), dtype=jnp.int32)
        finite_ = jnp.asarray(finite, dtype=bool)
        complete = jnp.asarray(work_complete, dtype=bool)
        failure = jnp.asarray(failure_code, dtype=jnp.int32)
        margin_ = jnp.asarray(margin)
        if any(
            value.shape != ()
            for value in (level_, required, finite_, complete, failure, margin_)
        ):
            raise ValueError("Contact guarantee evidence fields must be scalar.")
        identifier = str(backend_id)
        if not identifier:
            raise ValueError("backend_id must be nonempty.")
        successful = finite_ & complete & (failure == 0) & (level_ >= required)
        self.level = level_
        self.required_level = required
        self.finite = finite_
        self.work_complete = complete
        self.successful = successful
        self.failure_code = failure
        self.margin = margin_
        self.backend_id = identifier

    def meets(self, required: ContactGuaranteeLevel | int, /) -> Array:
        return self.successful & (self.level >= int(required))


__all__ = [
    "ContactCapability",
    "ContactGuaranteeEvidence",
    "ContactGuaranteeLevel",
]
