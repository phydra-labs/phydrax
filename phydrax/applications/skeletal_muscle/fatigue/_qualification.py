#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualification evidence for Liu--Brown--Yue (2002) trajectories."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._liu_brown_yue_2002 import PreparedLiuBrownYue2002


class LiuBrownYue2002QualificationEvidence(StrictModule, NonTrainableState):
    """Conservation, nonnegativity, fatigue, and recovery evidence."""

    maximum_conservation_error: Array
    minimum_compartment_fraction: Array
    sustained_effort_fatigue_increases: Array
    zero_effort_fatigued_nonincreasing: Array
    zero_effort_active_non_decreasing: Array
    zero_effort_transfer_conserved: Array
    finite: Array
    valid: Array
    claim_scope: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)


class LiuBrownYue2002QualificationPlan(StrictModule, NonTrainableState):
    """Fixed trajectory tolerances for independently generated source trials."""

    conservation_tolerance: float = eqx.field(static=True)
    monotonic_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        conservation_tolerance: float = 2.0e-6,
        monotonic_tolerance: float = 2.0e-7,
    ):
        conservation = float(conservation_tolerance)
        monotonic = float(monotonic_tolerance)
        if not isfinite(conservation) or conservation <= 0.0:
            raise ValueError("conservation_tolerance must be positive and finite.")
        if not isfinite(monotonic) or monotonic < 0.0:
            raise ValueError("monotonic_tolerance must be nonnegative and finite.")
        self.conservation_tolerance = conservation
        self.monotonic_tolerance = monotonic
        self.plan_id = canonical_fingerprint(
            {
                "kind": "liu-brown-yue-2002-qualification",
                "conservation_tolerance": conservation.hex(),
                "monotonic_tolerance": monotonic.hex(),
            }
        )

    def evaluate(
        self,
        prepared: PreparedLiuBrownYue2002,
        sustained_compartments: ArrayLike,
        recovery_compartments: ArrayLike,
        /,
    ) -> LiuBrownYue2002QualificationEvidence:
        """Qualify one sustained-effort trace and one zero-effort recovery trace."""

        if not isinstance(prepared, PreparedLiuBrownYue2002):
            raise TypeError("prepared must be PreparedLiuBrownYue2002.")
        sustained = jnp.asarray(sustained_compartments)
        recovery = jnp.asarray(recovery_compartments)
        if sustained.ndim != 2 or sustained.shape[1] != 3 or sustained.shape[0] < 2:
            raise ValueError("sustained_compartments must have shape (steps>=2, 3).")
        if recovery.ndim != 2 or recovery.shape[1] != 3 or recovery.shape[0] < 2:
            raise ValueError("recovery_compartments must have shape (steps>=2, 3).")
        dtype = jnp.result_type(sustained, recovery, float)
        sustained = sustained.astype(dtype)
        recovery = recovery.astype(dtype)
        sustained_total = jnp.sum(sustained, axis=1)
        recovery_total = jnp.sum(recovery, axis=1)
        reference_sustained = sustained_total[0]
        reference_recovery = recovery_total[0]
        maximum_error = jnp.maximum(
            jnp.max(jnp.abs(sustained_total - reference_sustained)),
            jnp.max(jnp.abs(recovery_total - reference_recovery)),
        )
        minimum = jnp.minimum(jnp.min(sustained), jnp.min(recovery))
        fatigued_sustained = sustained[:, 2]
        fatigued_recovery = recovery[:, 2]
        active_recovery = recovery[:, 1]
        tolerance = jnp.asarray(self.monotonic_tolerance, dtype=dtype)
        sustained_increases = fatigued_sustained[-1] > fatigued_sustained[0] + tolerance
        recovery_nonincreasing = jnp.all(jnp.diff(fatigued_recovery) <= tolerance)
        active_non_decreasing = jnp.all(jnp.diff(active_recovery) >= -tolerance)
        transfer_conserved = jnp.max(
            jnp.abs(
                (active_recovery - active_recovery[0])
                + (fatigued_recovery - fatigued_recovery[0])
            )
        ) <= self.conservation_tolerance
        finite = jnp.all(jnp.isfinite(sustained)) & jnp.all(jnp.isfinite(recovery))
        valid = (
            finite
            & (maximum_error <= self.conservation_tolerance)
            & (minimum >= -self.conservation_tolerance)
            & sustained_increases
            & recovery_nonincreasing
            & active_non_decreasing
            & transfer_conserved
        )
        return LiuBrownYue2002QualificationEvidence(
            maximum_error,
            minimum,
            sustained_increases,
            recovery_nonincreasing,
            active_non_decreasing,
            transfer_conserved,
            finite,
            valid,
            (
                "piecewise-constant brain-effort trajectories under "
                "Liu--Brown--Yue 2002 Eqs. 1a--1b; no intermittent recovery multiplier"
            ),
            canonical_fingerprint(
                {
                    "kind": "liu-brown-yue-2002-qualified-trajectories",
                    "plan": self.plan_id,
                    "model": prepared.plan.model_id,
                }
            ),
        )


__all__ = [
    "LiuBrownYue2002QualificationEvidence",
    "LiuBrownYue2002QualificationPlan",
]
