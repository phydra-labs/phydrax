#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..graph._harmonic_classes import HarmonicClassFrame


class HarmonicConstraint(StrictModule, NonTrainableState):
    """Solver-owned policy and target periods for one harmonic class frame."""

    frame: HarmonicClassFrame
    target_periods: Array
    policy: Literal["prescribed", "free", "gauge", "deflated"] = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        frame: HarmonicClassFrame,
        target_periods: ArrayLike,
        /,
        *,
        policy: Literal["prescribed", "free", "gauge", "deflated"] = "prescribed",
    ):
        if policy not in ("prescribed", "free", "gauge", "deflated"):
            raise ValueError("Unknown harmonic constraint policy.")
        periods = jnp.asarray(target_periods)
        if periods.shape != (frame.exact_basis.generator_count,):
            raise ValueError("Harmonic target periods do not match the class frame.")
        if not bool(jnp.all(jnp.isfinite(periods))):
            raise ValueError("Harmonic target periods must be finite.")
        self.frame = frame
        self.target_periods = periods
        self.policy = policy
        self.constraint_id = canonical_fingerprint(
            {
                "kind": "harmonic-constraint",
                "frame": frame.frame_id,
                "policy": policy,
                "period_count": int(periods.shape[0]),
            }
        )

    def apply(self, cochain: ArrayLike, /) -> Array:
        values = jnp.asarray(cochain)
        if self.policy == "free":
            return values
        target = (
            jnp.zeros_like(self.target_periods)
            if self.policy in ("gauge", "deflated")
            else self.target_periods
        )
        return self.frame.with_periods(values, target)

    def residual(self, cochain: ArrayLike, /) -> Array:
        values = jnp.asarray(cochain)
        target = (
            self.frame.periods(values)
            if self.policy == "free"
            else (
                jnp.zeros_like(self.target_periods)
                if self.policy in ("gauge", "deflated")
                else self.target_periods
            )
        )
        return jnp.linalg.norm(self.frame.periods(values) - target)


def preserve_magnetic_periods(
    magnetic_cochain: ArrayLike,
    constraint: HarmonicConstraint,
    /,
) -> Array:
    """Restore solver-declared magnetic periods after a local closedness projection."""
    if constraint.frame.degree != 2:
        raise ValueError("Magnetic harmonic constraints require degree-two cochains.")
    return constraint.apply(magnetic_cochain)


__all__ = ["HarmonicConstraint", "preserve_magnetic_periods"]
