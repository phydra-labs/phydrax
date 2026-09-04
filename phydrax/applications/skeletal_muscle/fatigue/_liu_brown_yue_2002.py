#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Liu--Brown--Yue (2002) activation, fatigue, and recovery model.

This implements Eqs. 1a--1b of Liu, Brown & Yue, Biophysical Journal 82
(2002) 2344--2359, https://doi.org/10.1016/S0006-3495(02)75580-X:

``M_uc -> M_A -> M_F -> M_A`` at rates ``B``, ``F``, and ``R``.

The citation supports this macroscopic motor-unit-compartment model and its
constant-parameter interpretation.  It does not support a target-load
controller, an enhanced intermittent-rest recovery multiplier, or attachment
to D1.  Those later variants are intentionally absent.
"""

from __future__ import annotations

from enum import IntFlag
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


_SOURCE_DOI = "10.1016/S0006-3495(02)75580-X"


class LiuBrownYue2002Status(IntFlag):
    """Fail-closed status for one exact piecewise-constant-brain-effort step."""

    SUCCESS = 0
    NONFINITE_BRAIN_EFFORT = 1
    NEGATIVE_BRAIN_EFFORT = 2
    NONFINITE_STEP = 4
    NONPOSITIVE_STEP = 8
    NONFINITE_STATE = 16
    NEGATIVE_COMPARTMENT = 32
    CONSERVATION_FAILURE = 64


class LiuBrownYue2002Parameters(StrictModule):
    """Dynamic phenomenological rate leaves in inverse seconds."""

    fatigue_rate_per_s: Array
    recovery_rate_per_s: Array

    def __init__(self, *, fatigue_rate_per_s: float, recovery_rate_per_s: float):
        fatigue = float(fatigue_rate_per_s)
        recovery = float(recovery_rate_per_s)
        if not isfinite(fatigue) or fatigue <= 0.0:
            raise ValueError("fatigue_rate_per_s must be positive and finite.")
        if not isfinite(recovery) or recovery <= 0.0:
            raise ValueError("recovery_rate_per_s must be positive and finite.")
        self.fatigue_rate_per_s = jnp.asarray(fatigue, dtype=float)
        self.recovery_rate_per_s = jnp.asarray(recovery, dtype=float)


class LiuBrownYue2002Plan(StrictModule):
    """Source identity and dynamic rate parameters for one macroscopic muscle."""

    parameters: LiuBrownYue2002Parameters
    muscle_id: str = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: LiuBrownYue2002Parameters,
        /,
        *,
        muscle_id: str,
        protocol_id: str,
    ):
        if not isinstance(parameters, LiuBrownYue2002Parameters):
            raise TypeError("parameters must be LiuBrownYue2002Parameters.")
        if not isinstance(muscle_id, str) or not muscle_id:
            raise ValueError("muscle_id must be a nonempty string.")
        if not isinstance(protocol_id, str) or not protocol_id:
            raise ValueError("protocol_id must be a nonempty string.")
        self.parameters = parameters
        self.muscle_id = muscle_id
        self.protocol_id = protocol_id
        self.model_id = canonical_fingerprint(
            {
                "kind": "liu-brown-yue-2002-macroscopic-fatigue-recovery",
                "source_doi": _SOURCE_DOI,
                "muscle_id": muscle_id,
                "protocol_id": protocol_id,
                "compartments": ["uncommitted", "active", "fatigued"],
                "flows": ["B:uncommitted->active", "F:active->fatigued", "R:fatigued->active"],
            }
        )

    def prepare(self, /) -> "PreparedLiuBrownYue2002":
        """Prepare the exact fixed-compartment update."""

        return PreparedLiuBrownYue2002(
            self,
            canonical_fingerprint(
                {
                    "kind": "prepared-liu-brown-yue-2002",
                    "model": self.model_id,
                    "update": "exact-piecewise-constant-brain-effort",
                }
            ),
        )


class LiuBrownYue2002State(StrictModule, NonTrainableState):
    """Committed compartment fractions and physical time."""

    time_s: Array
    uncommitted_fraction: Array
    active_fraction: Array
    fatigued_fraction: Array
    total_fraction: Array
    step_index: Array
    model_id: str = eqx.field(static=True)


class LiuBrownYue2002Evidence(StrictModule, NonTrainableState):
    """Conservation, positivity, force-capacity, and failure evidence."""

    brain_effort_rate_per_s: Array
    duration_s: Array
    conservation_error: Array
    minimum_compartment_fraction: Array
    active_relative_force: Array
    available_fraction: Array
    brain_effort_finite: Array
    step_finite: Array
    state_finite: Array
    compartments_nonnegative: Array
    conserved: Array
    status: Array
    successful: Array
    claim_scope: str = eqx.field(static=True)


class LiuBrownYue2002Candidate(StrictModule):
    """Whole-compartment candidate retaining its exact rollback state."""

    source: LiuBrownYue2002State
    proposed: LiuBrownYue2002State
    evidence: LiuBrownYue2002Evidence


class LiuBrownYue2002Capacity(StrictModule, NonTrainableState):
    """Observable macroscopic active force and nonfatigued capacity fractions."""

    active_relative_force: Array
    available_fraction: Array
    fatigued_fraction: Array
    time_s: Array
    model_id: str = eqx.field(static=True)


class PreparedLiuBrownYue2002(StrictModule):
    """Prepared exact runtime for the source's three-group dynamics."""

    plan: LiuBrownYue2002Plan
    prepared_id: str = eqx.field(static=True)

    def initialize(
        self,
        /,
        *,
        uncommitted_fraction: float = 1.0,
        active_fraction: float = 0.0,
        fatigued_fraction: float = 0.0,
        time_s: float = 0.0,
    ) -> LiuBrownYue2002State:
        """Initialize nonnegative compartment fractions with their conserved total."""

        values = tuple(
            float(value)
            for value in (
                uncommitted_fraction,
                active_fraction,
                fatigued_fraction,
                time_s,
            )
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("Initial state values must be finite.")
        uncommitted, active, fatigued, time = values
        if min(uncommitted, active, fatigued) < 0.0:
            raise ValueError("Initial compartment fractions must be nonnegative.")
        total = uncommitted + active + fatigued
        if abs(total - 1.0) > 1.0e-10:
            raise ValueError("Initial compartment fractions must sum to one.")
        dtype = self.plan.parameters.fatigue_rate_per_s.dtype
        return LiuBrownYue2002State(
            jnp.asarray(time, dtype=dtype),
            jnp.asarray(uncommitted, dtype=dtype),
            jnp.asarray(active, dtype=dtype),
            jnp.asarray(fatigued, dtype=dtype),
            jnp.asarray(total, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan.model_id,
        )

    def evaluate(
        self,
        state: LiuBrownYue2002State,
        brain_effort_rate_per_s: ArrayLike,
        duration_s: ArrayLike,
        /,
    ) -> LiuBrownYue2002Candidate:
        """Propose the exact source solution for constant ``B`` over one step."""

        if not isinstance(state, LiuBrownYue2002State):
            raise TypeError("state must be LiuBrownYue2002State.")
        if state.model_id != self.plan.model_id:
            raise ValueError("state does not belong to this prepared model.")
        dtype = state.active_fraction.dtype
        brain_effort = jnp.asarray(brain_effort_rate_per_s, dtype=dtype)
        duration = jnp.asarray(duration_s, dtype=dtype)
        if brain_effort.shape != () or duration.shape != ():
            raise ValueError("brain_effort_rate_per_s and duration_s must be scalar.")
        brain_effort_finite = jnp.isfinite(brain_effort)
        brain_effort_nonnegative = brain_effort >= 0.0
        step_finite = jnp.isfinite(duration)
        step_positive = duration > 0.0
        safe_brain_effort = jnp.where(
            brain_effort_finite & brain_effort_nonnegative, brain_effort, 0.0
        )
        safe_duration = jnp.where(step_finite & step_positive, duration, 0.0)

        fatigue = self.plan.parameters.fatigue_rate_per_s.astype(dtype)
        recovery = self.plan.parameters.recovery_rate_per_s.astype(dtype)
        exchange_rate = fatigue + recovery
        uncommitted_decay = jnp.exp(-safe_brain_effort * safe_duration)
        exchange_decay = jnp.exp(-exchange_rate * safe_duration)
        new_uncommitted = state.uncommitted_fraction * uncommitted_decay

        half_difference_time = (
            0.5 * (exchange_rate - safe_brain_effort) * safe_duration
        )
        safe_denominator = jnp.where(
            jnp.abs(half_difference_time) > jnp.sqrt(jnp.finfo(dtype).eps),
            half_difference_time,
            1.0,
        )
        sinhc_direct = jnp.sinh(half_difference_time) / safe_denominator
        x2 = half_difference_time**2
        sinhc_series = 1.0 + x2 / 6.0 + x2**2 / 120.0
        sinhc = jnp.where(
            jnp.abs(half_difference_time) > jnp.sqrt(jnp.finfo(dtype).eps),
            sinhc_direct,
            sinhc_series,
        )
        uncommitted_convolution = (
            safe_duration
            * jnp.exp(
                -0.5 * (safe_brain_effort + exchange_rate) * safe_duration
            )
            * sinhc
        )
        recovered_equilibrium = (
            fatigue
            * state.total_fraction
            * (-jnp.expm1(-exchange_rate * safe_duration))
            / exchange_rate
        )
        new_fatigued = (
            state.fatigued_fraction * exchange_decay
            + recovered_equilibrium
            - fatigue * state.uncommitted_fraction * uncommitted_convolution
        )
        new_active = state.total_fraction - new_uncommitted - new_fatigued
        proposed = LiuBrownYue2002State(
            state.time_s + safe_duration,
            new_uncommitted,
            new_active,
            new_fatigued,
            state.total_fraction,
            state.step_index + jnp.asarray(1, dtype=jnp.int32),
            state.model_id,
        )
        compartments = jnp.stack(
            (proposed.uncommitted_fraction, proposed.active_fraction, proposed.fatigued_fraction)
        )
        state_finite = jnp.all(jnp.isfinite(compartments)) & jnp.isfinite(
            proposed.time_s
        )
        conservation_error = jnp.abs(jnp.sum(compartments) - state.total_fraction)
        tolerance = 128.0 * jnp.finfo(dtype).eps * jnp.maximum(
            1.0, jnp.abs(state.total_fraction)
        )
        compartments_nonnegative = jnp.min(compartments) >= -tolerance
        conserved = conservation_error <= tolerance
        active_relative_force = proposed.active_fraction / state.total_fraction
        available_fraction = (
            proposed.uncommitted_fraction + proposed.active_fraction
        ) / state.total_fraction

        status = jnp.asarray(int(LiuBrownYue2002Status.SUCCESS), dtype=jnp.int32)
        status = jnp.where(
            brain_effort_finite,
            status,
            jnp.bitwise_or(status, int(LiuBrownYue2002Status.NONFINITE_BRAIN_EFFORT)),
        )
        status = jnp.where(
            brain_effort_nonnegative,
            status,
            jnp.bitwise_or(status, int(LiuBrownYue2002Status.NEGATIVE_BRAIN_EFFORT)),
        )
        status = jnp.where(
            step_finite,
            status,
            jnp.bitwise_or(status, int(LiuBrownYue2002Status.NONFINITE_STEP)),
        )
        status = jnp.where(
            step_positive,
            status,
            jnp.bitwise_or(status, int(LiuBrownYue2002Status.NONPOSITIVE_STEP)),
        )
        status = jnp.where(
            state_finite,
            status,
            jnp.bitwise_or(status, int(LiuBrownYue2002Status.NONFINITE_STATE)),
        )
        status = jnp.where(
            compartments_nonnegative,
            status,
            jnp.bitwise_or(status, int(LiuBrownYue2002Status.NEGATIVE_COMPARTMENT)),
        )
        status = jnp.where(
            conserved,
            status,
            jnp.bitwise_or(status, int(LiuBrownYue2002Status.CONSERVATION_FAILURE)),
        )
        successful = status == int(LiuBrownYue2002Status.SUCCESS)
        return LiuBrownYue2002Candidate(
            state,
            proposed,
            LiuBrownYue2002Evidence(
                brain_effort,
                duration,
                conservation_error,
                jnp.min(compartments),
                active_relative_force,
                available_fraction,
                brain_effort_finite,
                step_finite,
                state_finite,
                compartments_nonnegative,
                conserved,
                status,
                successful,
                (
                    "Liu--Brown--Yue 2002 macroscopic compartment fractions only; "
                    "active fraction is relative isometric force for this route and "
                    "is never composed with D1"
                ),
            ),
        )

    def capacity(self, state: LiuBrownYue2002State, /) -> LiuBrownYue2002Capacity:
        """Observe active force and available nonfatigued fractions."""

        if not isinstance(state, LiuBrownYue2002State):
            raise TypeError("state must be LiuBrownYue2002State.")
        if state.model_id != self.plan.model_id:
            raise ValueError("state does not belong to this prepared model.")
        return LiuBrownYue2002Capacity(
            state.active_fraction / state.total_fraction,
            (state.uncommitted_fraction + state.active_fraction)
            / state.total_fraction,
            state.fatigued_fraction / state.total_fraction,
            state.time_s,
            state.model_id,
        )


def commit_liu_brown_yue_2002(
    candidate: LiuBrownYue2002Candidate,
    current: LiuBrownYue2002State,
    /,
) -> LiuBrownYue2002State:
    """Atomically commit a successful exact update or preserve all source state."""

    if not isinstance(candidate, LiuBrownYue2002Candidate):
        raise TypeError("candidate must be LiuBrownYue2002Candidate.")
    if not isinstance(current, LiuBrownYue2002State):
        raise TypeError("current must be LiuBrownYue2002State.")
    source = candidate.source
    source_matches = (
        (source.model_id == current.model_id)
        & (source.time_s == current.time_s)
        & (source.uncommitted_fraction == current.uncommitted_fraction)
        & (source.active_fraction == current.active_fraction)
        & (source.fatigued_fraction == current.fatigued_fraction)
        & (source.total_fraction == current.total_fraction)
        & (source.step_index == current.step_index)
    )
    return jax.lax.cond(
        candidate.evidence.successful & source_matches,
        lambda _: candidate.proposed,
        lambda _: current,
        operand=None,
    )


__all__ = [
    "LiuBrownYue2002Candidate",
    "LiuBrownYue2002Capacity",
    "LiuBrownYue2002Evidence",
    "LiuBrownYue2002Parameters",
    "LiuBrownYue2002Plan",
    "LiuBrownYue2002State",
    "LiuBrownYue2002Status",
    "PreparedLiuBrownYue2002",
    "commit_liu_brown_yue_2002",
]
