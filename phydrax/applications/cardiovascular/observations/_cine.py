#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Cardiac-cycle timing for normalized cine observations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._metadata import TimeBase


class CineTimingEvidence(StrictModule):
    """Temporal coverage and validity evidence for one cine cycle."""

    largest_phase_gap: Array
    phase_coverage: Array
    unique_phases: Array
    finite: Array
    successful: Array


class CineTimingResult(StrictModule):
    """Phase and circular frame-duration coordinates in acquisition order."""

    sample_times_ms: Array
    phase: Array
    frame_duration_ms: Array
    evidence: CineTimingEvidence
    prepared_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class CineTimingPlan:
    """Host plan for one cine series spanning no more than one cardiac cycle.

    ``end_diastolic_time_ms`` defines phase zero.  The source timebase remains in
    acquisition order; frame integration durations are circular Voronoi widths
    whose sum is exactly one declared cycle.
    """

    timebase: TimeBase
    cycle_length_ms: float
    end_diastolic_time_ms: float
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.timebase, TimeBase):
            raise TypeError("timebase must be a TimeBase.")
        cycle = float(self.cycle_length_ms)
        reference = float(self.end_diastolic_time_ms)
        if not math.isfinite(cycle) or cycle <= 0.0:
            raise ValueError("cycle_length_ms must be finite and positive.")
        if not math.isfinite(reference):
            raise ValueError("end_diastolic_time_ms must be finite.")
        tolerance = (
            64.0 * np.finfo(self.timebase.sample_times_ms.dtype).eps * max(1.0, cycle)
        )
        if self.timebase.duration_ms >= cycle - tolerance:
            raise ValueError(
                "A cine timing plan must contain one cycle without duplicating its periodic endpoint."
            )
        object.__setattr__(self, "cycle_length_ms", cycle)
        object.__setattr__(self, "end_diastolic_time_ms", reference)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-cine-timing-plan",
                    "timebase_id": self.timebase.timebase_id,
                    "sample_times_ms": array_tree_fingerprint(
                        self.timebase.sample_times_ms
                    ),
                    "cycle_length_ms": cycle,
                    "end_diastolic_time_ms": reference,
                }
            ),
        )

    def prepare(self) -> "PreparedCineTiming":
        times = self.timebase.sample_times_ms
        phases = np.mod((times - self.end_diastolic_time_ms) / self.cycle_length_ms, 1.0)
        order = np.argsort(phases, kind="stable")
        sorted_phase = phases[order]
        gaps = np.diff(np.concatenate((sorted_phase, sorted_phase[:1] + 1.0)))
        previous = np.roll(gaps, 1)
        sorted_durations = 0.5 * (previous + gaps) * self.cycle_length_ms
        durations = np.empty_like(sorted_durations)
        durations[order] = sorted_durations
        tolerance = 64.0 * np.finfo(phases.dtype).eps
        unique = bool(np.all(gaps > tolerance))
        return PreparedCineTiming(
            times,
            phases,
            durations,
            float(np.max(gaps)),
            unique,
            self.cycle_length_ms,
            self.end_diastolic_time_ms,
            self.timebase.timebase_id,
            self.plan_id,
        )


class PreparedCineTiming(StrictModule, NonTrainableState):
    """Prepared fixed-shape cine phase coordinates."""

    sample_times_ms: Array
    phase: Array
    frame_duration_ms: Array
    largest_phase_gap: Array
    cycle_length_ms: float = eqx.field(static=True)
    end_diastolic_time_ms: float = eqx.field(static=True)
    timebase_id: str = eqx.field(static=True)
    unique_phases: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_times_ms: ArrayLike,
        phase: ArrayLike,
        frame_duration_ms: ArrayLike,
        largest_phase_gap: float,
        unique_phases: bool,
        cycle_length_ms: float,
        end_diastolic_time_ms: float,
        timebase_id: str,
        plan_id: str,
        /,
    ):
        times = jax.lax.stop_gradient(jnp.asarray(sample_times_ms))
        phase_ = jax.lax.stop_gradient(jnp.asarray(phase, dtype=times.dtype))
        duration = jax.lax.stop_gradient(
            jnp.asarray(frame_duration_ms, dtype=times.dtype)
        )
        if (
            times.ndim != 1
            or phase_.shape != times.shape
            or duration.shape != times.shape
        ):
            raise ValueError(
                "Prepared cine timing arrays must be matching rank-one arrays."
            )
        self.sample_times_ms = times
        self.phase = phase_
        self.frame_duration_ms = duration
        self.largest_phase_gap = jnp.asarray(largest_phase_gap, dtype=times.dtype)
        self.cycle_length_ms = float(cycle_length_ms)
        self.end_diastolic_time_ms = float(end_diastolic_time_ms)
        self.timebase_id = str(timebase_id)
        self.unique_phases = bool(unique_phases)
        self.plan_id = str(plan_id)
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-cardiovascular-cine-timing", "plan_id": self.plan_id}
        )

    def phase_at(self, times_ms: ArrayLike, /) -> Array:
        """Map dynamic millisecond times to periodic phase in ``[0, 1)``."""
        times = jnp.asarray(times_ms, dtype=self.sample_times_ms.dtype)
        return jnp.mod(
            (times - self.end_diastolic_time_ms) / self.cycle_length_ms,
            1.0,
        )

    def evaluate(self) -> CineTimingResult:
        finite = (
            jnp.all(jnp.isfinite(self.sample_times_ms))
            & jnp.all(jnp.isfinite(self.phase))
            & jnp.all(jnp.isfinite(self.frame_duration_ms))
            & jnp.isfinite(self.largest_phase_gap)
        )
        unique = jnp.asarray(self.unique_phases)
        successful = finite & unique & jnp.all(self.frame_duration_ms > 0.0)
        evidence = CineTimingEvidence(
            self.largest_phase_gap,
            1.0 - self.largest_phase_gap,
            unique,
            finite,
            successful,
        )
        return CineTimingResult(
            self.sample_times_ms,
            self.phase,
            self.frame_duration_ms,
            evidence,
            self.prepared_id,
        )


__all__ = [
    "CineTimingEvidence",
    "CineTimingPlan",
    "CineTimingResult",
    "PreparedCineTiming",
]
