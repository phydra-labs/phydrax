#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fuglevand--Winter--Patla (1993) stochastic isometric motor-unit pool.

The equations implemented here are Eqs. 1--20 of Fuglevand, Winter & Patla,
J. Neurophysiol. 70 (1993) 2470--2488,
https://doi.org/10.1152/jn.1993.70.6.2470.  The citation supports only the
recruitment/rate-coding renewal process, critically damped twitch, nonlinear
instantaneous force gain, and independent summation used below.  It does not
support dynamic contractions, fatigue, synchronization, or gradients through
recruitment and discharge topology; none of those claims are made here.
"""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


_SOURCE_DOI = "10.1152/jn.1993.70.6.2470"


class FuglevandWinterPatla1993Status(IntFlag):
    """Fail-closed status for one stochastic motor-unit-pool step."""

    SUCCESS = 0
    NONFINITE_EXCITATION = 1
    NEGATIVE_EXCITATION = 2
    NONFINITE_STEP = 4
    NONPOSITIVE_STEP = 8
    RANDOM_STEP_MISMATCH = 16
    EVENT_CAPACITY_OVERFLOW = 32
    NONFINITE_STATE = 64
    NEGATIVE_FORCE_STATE = 128


class FuglevandWinterPatla1993Parameters(StrictModule):
    """Dynamic numeric leaves for the source equations.

    Excitation and recruitment thresholds use the paper's arbitrary excitation
    unit.  Twitch force uses its arbitrary force unit; no relation to D1 force
    or to newtons is implied.  Time kernels use milliseconds and rates use Hz.
    """

    recruitment_threshold_range: Array
    twitch_force_range: Array
    contraction_time_range: Array
    longest_contraction_time_ms: Array
    minimum_firing_rate_hz: Array
    peak_firing_rate_first_hz: Array
    peak_firing_rate_difference_hz: Array
    firing_rate_gain_hz_per_excitation: Array
    interspike_interval_cv: Array

    def __init__(
        self,
        *,
        recruitment_threshold_range: float = 30.0,
        twitch_force_range: float = 100.0,
        contraction_time_range: float = 3.0,
        longest_contraction_time_ms: float = 90.0,
        minimum_firing_rate_hz: float = 8.0,
        peak_firing_rate_first_hz: float = 45.0,
        peak_firing_rate_difference_hz: float = 10.0,
        firing_rate_gain_hz_per_excitation: float = 1.0,
        interspike_interval_cv: float = 0.2,
    ):
        values = {
            "recruitment_threshold_range": float(recruitment_threshold_range),
            "twitch_force_range": float(twitch_force_range),
            "contraction_time_range": float(contraction_time_range),
            "longest_contraction_time_ms": float(longest_contraction_time_ms),
            "minimum_firing_rate_hz": float(minimum_firing_rate_hz),
            "peak_firing_rate_first_hz": float(peak_firing_rate_first_hz),
            "peak_firing_rate_difference_hz": float(peak_firing_rate_difference_hz),
            "firing_rate_gain_hz_per_excitation": float(
                firing_rate_gain_hz_per_excitation
            ),
            "interspike_interval_cv": float(interspike_interval_cv),
        }
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("Every Fuglevand parameter must be finite.")
        if values["recruitment_threshold_range"] <= 1.0:
            raise ValueError("recruitment_threshold_range must exceed one.")
        if values["twitch_force_range"] <= 1.0:
            raise ValueError("twitch_force_range must exceed one.")
        if values["contraction_time_range"] <= 1.0:
            raise ValueError("contraction_time_range must exceed one.")
        if values["longest_contraction_time_ms"] <= 0.0:
            raise ValueError("longest_contraction_time_ms must be positive.")
        if values["minimum_firing_rate_hz"] <= 0.0:
            raise ValueError("minimum_firing_rate_hz must be positive.")
        if values["peak_firing_rate_difference_hz"] < 0.0:
            raise ValueError("peak_firing_rate_difference_hz must be nonnegative.")
        if (
            values["peak_firing_rate_first_hz"]
            - values["peak_firing_rate_difference_hz"]
            <= values["minimum_firing_rate_hz"]
        ):
            raise ValueError("Every peak firing rate must exceed the minimum rate.")
        if values["firing_rate_gain_hz_per_excitation"] <= 0.0:
            raise ValueError("firing_rate_gain_hz_per_excitation must be positive.")
        cv = values["interspike_interval_cv"]
        if cv <= 0.0 or cv >= 1.0 / 3.9:
            raise ValueError(
                "interspike_interval_cv must be in (0, 1/3.9) so every "
                "source-truncated interval remains positive."
            )
        for name, value in values.items():
            setattr(self, name, jnp.asarray(value, dtype=float))


class FuglevandWinterPatla1993Plan(StrictModule):
    """Static pool/event topology and dynamic source parameters."""

    parameters: FuglevandWinterPatla1993Parameters
    unit_count: int = eqx.field(static=True)
    event_capacity_per_unit: int = eqx.field(static=True)
    random_stream_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        unit_count: int = 120,
        /,
        *,
        event_capacity_per_unit: int = 16,
        random_stream_id: str,
        parameters: FuglevandWinterPatla1993Parameters | None = None,
    ):
        if isinstance(unit_count, bool) or not isinstance(unit_count, int):
            raise TypeError("unit_count must be an integer.")
        if unit_count <= 0:
            raise ValueError("unit_count must be positive.")
        if (
            isinstance(event_capacity_per_unit, bool)
            or not isinstance(event_capacity_per_unit, int)
        ):
            raise TypeError("event_capacity_per_unit must be an integer.")
        if event_capacity_per_unit <= 0:
            raise ValueError("event_capacity_per_unit must be positive.")
        if not isinstance(random_stream_id, str) or not random_stream_id:
            raise ValueError("random_stream_id must be a nonempty string.")
        parameters_ = (
            FuglevandWinterPatla1993Parameters()
            if parameters is None
            else parameters
        )
        if not isinstance(parameters_, FuglevandWinterPatla1993Parameters):
            raise TypeError(
                "parameters must be FuglevandWinterPatla1993Parameters or None."
            )
        self.parameters = parameters_
        self.unit_count = unit_count
        self.event_capacity_per_unit = event_capacity_per_unit
        self.random_stream_id = random_stream_id
        self.model_id = canonical_fingerprint(
            {
                "kind": "fuglevand-winter-patla-1993-isometric-motor-unit-pool",
                "source_doi": _SOURCE_DOI,
                "unit_count": unit_count,
                "event_capacity_per_unit": event_capacity_per_unit,
                "random_stream_id": random_stream_id,
            }
        )

    def prepare(self, /) -> "PreparedFuglevandWinterPatla1993":
        """Materialize the fixed-rank source distributions."""

        rank = jnp.arange(1, self.unit_count + 1, dtype=float)
        fraction = rank / float(self.unit_count)
        parameters = self.parameters
        thresholds = jnp.exp(
            jnp.log(parameters.recruitment_threshold_range) * fraction
        )
        peak_twitch = jnp.exp(jnp.log(parameters.twitch_force_range) * fraction)
        contraction_power = jnp.log(parameters.contraction_time_range) / jnp.log(
            parameters.twitch_force_range
        )
        contraction_time = parameters.longest_contraction_time_ms / (
            peak_twitch**contraction_power
        )
        peak_firing = parameters.peak_firing_rate_first_hz - (
            parameters.peak_firing_rate_difference_hz
            * thresholds
            / thresholds[-1]
        )
        maximum_excitation = thresholds[-1] + (
            peak_firing[-1] - parameters.minimum_firing_rate_hz
        ) / parameters.firing_rate_gain_hz_per_excitation
        return PreparedFuglevandWinterPatla1993(
            self,
            thresholds,
            peak_twitch,
            contraction_time,
            peak_firing,
            maximum_excitation,
            canonical_fingerprint(
                {
                    "kind": "prepared-fuglevand-winter-patla-1993",
                    "model": self.model_id,
                    "rank_indexing": "i=1,...,n",
                    "event_layout": "motor-unit,event-slot",
                }
            ),
        )


class FuglevandWinterPatla1993RandomInput(StrictModule, NonTrainableState):
    """One explicitly named semantic PRNG input.

    Reusing the same key, stream identity, semantic step, and source state gives
    bitwise replay.  A committed state advances ``random_step`` exactly once;
    failed candidates roll that counter back with the rest of the state.
    """

    key: Array
    semantic_step: Array
    stream_id: str = eqx.field(static=True)

    def __init__(self, key: Any, semantic_step: ArrayLike, /, *, stream_id: str):
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("stream_id must be a nonempty string.")
        key_data = jr.key_data(key)
        if key_data.shape != (2,):
            raise ValueError("key must contain one JAX PRNG key.")
        step = jnp.asarray(semantic_step, dtype=jnp.int32)
        if step.shape != ():
            raise ValueError("semantic_step must be scalar.")
        self.key = jr.wrap_key_data(key_data)
        self.semantic_step = step
        self.stream_id = stream_id


class FuglevandWinterPatla1993State(StrictModule, NonTrainableState):
    """Committed renewal and exact critically damped twitch state."""

    time_ms: Array
    last_discharge_ms: Array
    next_discharge_ms: Array
    twitch_driver: Array
    motor_unit_force: Array
    random_step: Array
    step_index: Array
    model_id: str = eqx.field(static=True)


class FuglevandWinterPatla1993Evidence(StrictModule, NonTrainableState):
    """Fixed-capacity event, force, RNG, and fail-closed evidence."""

    event_times_ms: Array
    event_mask: Array
    event_gain: Array
    normal_scores: Array
    event_count: Array
    overflow_by_unit: Array
    firing_rate_hz: Array
    total_force_arbitrary: Array
    excitation_finite: Array
    step_finite: Array
    random_step_matches: Array
    state_finite: Array
    force_nonnegative: Array
    status: Array
    successful: Array
    topology_gradient_supported: bool = eqx.field(static=True)
    differentiation_scope: str = eqx.field(static=True)


class FuglevandWinterPatla1993Candidate(StrictModule):
    """Whole-state stochastic step before atomic commit."""

    source: FuglevandWinterPatla1993State
    proposed: FuglevandWinterPatla1993State
    evidence: FuglevandWinterPatla1993Evidence


class FuglevandWinterPatla1993Force(StrictModule, NonTrainableState):
    """Terminal isometric force of this source route in arbitrary force units."""

    motor_unit_force_arbitrary: Array
    total_force_arbitrary: Array
    time_ms: Array
    model_id: str = eqx.field(static=True)


class PreparedFuglevandWinterPatla1993(StrictModule):
    """Prepared fixed-capacity stochastic isometric motor-unit runtime."""

    plan: FuglevandWinterPatla1993Plan
    recruitment_threshold_excitation: Array
    peak_twitch_force_arbitrary: Array
    contraction_time_ms: Array
    peak_firing_rate_hz: Array
    maximum_excitation: Array
    prepared_id: str = eqx.field(static=True)

    def firing_rate(self, excitation: ArrayLike, /) -> Array:
        """Evaluate source Eq. 2 with recruitment and peak-rate saturation."""

        value = jnp.asarray(excitation, dtype=self.recruitment_threshold_excitation.dtype)
        if value.shape != ():
            raise ValueError("excitation must be scalar.")
        parameters = self.plan.parameters
        raw = parameters.minimum_firing_rate_hz + (
            parameters.firing_rate_gain_hz_per_excitation
            * (value - self.recruitment_threshold_excitation)
        )
        return jnp.where(
            value >= self.recruitment_threshold_excitation,
            jnp.minimum(raw, self.peak_firing_rate_hz),
            0.0,
        )

    def initialize(self, /, *, time_ms: float = 0.0) -> FuglevandWinterPatla1993State:
        """Create an unfired, zero-force committed pool state."""

        time = float(time_ms)
        if not isfinite(time):
            raise ValueError("time_ms must be finite.")
        shape = (self.plan.unit_count,)
        dtype = self.recruitment_threshold_excitation.dtype
        return FuglevandWinterPatla1993State(
            jnp.asarray(time, dtype=dtype),
            jnp.full(shape, -jnp.inf, dtype=dtype),
            jnp.full(shape, jnp.inf, dtype=dtype),
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan.model_id,
        )

    def evaluate(
        self,
        state: FuglevandWinterPatla1993State,
        excitation: ArrayLike,
        duration_ms: ArrayLike,
        random_input: FuglevandWinterPatla1993RandomInput,
        /,
    ) -> FuglevandWinterPatla1993Candidate:
        """Propose one piecewise-constant-excitation stochastic renewal step.

        Recruitment, event counts, event times, and gain branch selection are
        explicitly stop-gradient.  Derivatives of smooth twitch decay with a
        fixed realized event schedule remain local derivatives only.
        """

        if not isinstance(state, FuglevandWinterPatla1993State):
            raise TypeError("state must be FuglevandWinterPatla1993State.")
        if state.model_id != self.plan.model_id:
            raise ValueError("state does not belong to this prepared model.")
        if not isinstance(random_input, FuglevandWinterPatla1993RandomInput):
            raise TypeError(
                "random_input must be FuglevandWinterPatla1993RandomInput."
            )
        if random_input.stream_id != self.plan.random_stream_id:
            raise ValueError("random_input stream_id does not match the plan.")
        n = self.plan.unit_count
        expected = (n,)
        for name, value in (
            ("last_discharge_ms", state.last_discharge_ms),
            ("next_discharge_ms", state.next_discharge_ms),
            ("twitch_driver", state.twitch_driver),
            ("motor_unit_force", state.motor_unit_force),
        ):
            if value.shape != expected:
                raise ValueError(f"state.{name} must have shape {expected}.")
        excitation_ = jnp.asarray(
            excitation, dtype=self.recruitment_threshold_excitation.dtype
        )
        duration = jnp.asarray(duration_ms, dtype=excitation_.dtype)
        if excitation_.shape != () or duration.shape != ():
            raise ValueError("excitation and duration_ms must be scalar.")

        excitation_finite = jnp.isfinite(excitation_)
        excitation_nonnegative = excitation_ >= 0.0
        step_finite = jnp.isfinite(duration)
        step_positive = duration > 0.0
        safe_excitation = jnp.where(
            excitation_finite & excitation_nonnegative, excitation_, 0.0
        )
        safe_duration = jnp.where(step_finite & step_positive, duration, 0.0)
        topology_excitation = jax.lax.stop_gradient(safe_excitation)
        topology_duration = jax.lax.stop_gradient(safe_duration)
        topology_end = jax.lax.stop_gradient(state.time_ms) + topology_duration
        eligible = topology_excitation >= self.recruitment_threshold_excitation
        firing_rate = jax.lax.stop_gradient(self.firing_rate(topology_excitation))
        safe_rate = jnp.where(eligible, firing_rate, 1.0)
        mean_interval_ms = 1000.0 / safe_rate

        capacity = self.plan.event_capacity_per_unit
        folded_key = jr.fold_in(
            random_input.key, random_input.semantic_step.astype(jnp.uint32)
        )
        normal_scores = jr.truncated_normal(
            folded_key,
            lower=-3.9,
            upper=3.9,
            shape=(n, capacity),
            dtype=excitation_.dtype,
        )
        normal_scores = jax.lax.stop_gradient(normal_scores)
        event_times = jnp.full((n, capacity), jnp.inf, dtype=excitation_.dtype)
        event_mask = jnp.zeros((n, capacity), dtype=bool)
        event_gain = jnp.zeros((n, capacity), dtype=excitation_.dtype)
        last = state.last_discharge_ms
        next_time = jnp.where(
            eligible,
            jnp.where(jnp.isfinite(state.next_discharge_ms), state.next_discharge_ms, state.time_ms),
            jnp.inf,
        )
        gain_boundary = jnp.asarray(0.4, dtype=excitation_.dtype)
        sigmoid_boundary = -jnp.expm1(-2.0 * gain_boundary**3) / gain_boundary
        parameters = self.plan.parameters

        for slot in range(capacity):
            fires = eligible & (next_time <= topology_end)
            time = jax.lax.stop_gradient(next_time)
            interval = time - last
            has_predecessor = fires & jnp.isfinite(last) & (interval > 0.0)
            safe_interval = jnp.where(has_predecessor, interval, 1.0)
            normalized_rate = self.contraction_time_ms / safe_interval
            safe_normalized_rate = jnp.maximum(normalized_rate, jnp.finfo(excitation_.dtype).tiny)
            high_gain = (
                -jnp.expm1(-2.0 * safe_normalized_rate**3)
                / safe_normalized_rate
                / sigmoid_boundary
            )
            gain = jnp.where(
                has_predecessor & (normalized_rate > gain_boundary), high_gain, 1.0
            )
            gain = jax.lax.stop_gradient(gain)
            event_times = event_times.at[:, slot].set(jnp.where(fires, time, jnp.inf))
            event_mask = event_mask.at[:, slot].set(fires)
            event_gain = event_gain.at[:, slot].set(jnp.where(fires, gain, 0.0))
            last = jnp.where(fires, time, last)
            interval_multiplier = 1.0 + (
                parameters.interspike_interval_cv * normal_scores[:, slot]
            )
            following = time + mean_interval_ms * interval_multiplier
            next_time = jnp.where(fires, following, next_time)

        event_times = jax.lax.stop_gradient(event_times)
        event_mask = jax.lax.stop_gradient(event_mask)
        event_gain = jax.lax.stop_gradient(event_gain)
        overflow = eligible & (next_time <= topology_end)
        end_time = state.time_ms + safe_duration
        rates_per_ms = 1.0 / self.contraction_time_ms
        decay = jnp.exp(-rates_per_ms * safe_duration)
        base_driver = state.twitch_driver * decay
        base_force = (
            state.motor_unit_force
            + rates_per_ms * state.twitch_driver * safe_duration
        ) * decay
        age = jnp.maximum(end_time - event_times, 0.0)
        impulse_decay = jnp.exp(-rates_per_ms[:, None] * age)
        amplitude = (
            jnp.e
            * self.peak_twitch_force_arbitrary[:, None]
            * event_gain
            * event_mask
        )
        impulse_driver = jnp.sum(amplitude * impulse_decay, axis=1)
        impulse_force = jnp.sum(
            amplitude * rates_per_ms[:, None] * age * impulse_decay,
            axis=1,
        )
        proposed_driver = base_driver + impulse_driver
        proposed_force = base_force + impulse_force
        proposed = FuglevandWinterPatla1993State(
            end_time,
            last,
            next_time,
            proposed_driver,
            proposed_force,
            state.random_step + jnp.asarray(1, dtype=jnp.int32),
            state.step_index + jnp.asarray(1, dtype=jnp.int32),
            state.model_id,
        )
        state_finite = (
            jnp.isfinite(proposed.time_ms)
            & jnp.all(jnp.isfinite(proposed.twitch_driver))
            & jnp.all(jnp.isfinite(proposed.motor_unit_force))
            & jnp.all(jnp.isfinite(jnp.where(jnp.isinf(proposed.next_discharge_ms), 0.0, proposed.next_discharge_ms)))
        )
        force_nonnegative = jnp.all(proposed.motor_unit_force >= 0.0)
        random_matches = random_input.semantic_step == state.random_step
        status = jnp.asarray(int(FuglevandWinterPatla1993Status.SUCCESS), dtype=jnp.int32)
        status = jnp.where(
            excitation_finite,
            status,
            jnp.bitwise_or(status, int(FuglevandWinterPatla1993Status.NONFINITE_EXCITATION)),
        )
        status = jnp.where(
            excitation_nonnegative,
            status,
            jnp.bitwise_or(status, int(FuglevandWinterPatla1993Status.NEGATIVE_EXCITATION)),
        )
        status = jnp.where(
            step_finite,
            status,
            jnp.bitwise_or(status, int(FuglevandWinterPatla1993Status.NONFINITE_STEP)),
        )
        status = jnp.where(
            step_positive,
            status,
            jnp.bitwise_or(status, int(FuglevandWinterPatla1993Status.NONPOSITIVE_STEP)),
        )
        status = jnp.where(
            random_matches,
            status,
            jnp.bitwise_or(status, int(FuglevandWinterPatla1993Status.RANDOM_STEP_MISMATCH)),
        )
        status = jnp.where(
            ~jnp.any(overflow),
            status,
            jnp.bitwise_or(status, int(FuglevandWinterPatla1993Status.EVENT_CAPACITY_OVERFLOW)),
        )
        status = jnp.where(
            state_finite,
            status,
            jnp.bitwise_or(status, int(FuglevandWinterPatla1993Status.NONFINITE_STATE)),
        )
        status = jnp.where(
            force_nonnegative,
            status,
            jnp.bitwise_or(status, int(FuglevandWinterPatla1993Status.NEGATIVE_FORCE_STATE)),
        )
        successful = status == int(FuglevandWinterPatla1993Status.SUCCESS)
        evidence = FuglevandWinterPatla1993Evidence(
            event_times,
            event_mask,
            event_gain,
            normal_scores,
            jnp.sum(event_mask, axis=1).astype(jnp.int32),
            overflow,
            firing_rate,
            jnp.sum(proposed_force),
            excitation_finite,
            step_finite,
            random_matches,
            state_finite,
            force_nonnegative,
            status,
            successful,
            False,
            (
                "smooth twitch response conditional on fixed realized events; no "
                "derivatives through recruitment, event count/time, or force-gain branch"
            ),
        )
        return FuglevandWinterPatla1993Candidate(state, proposed, evidence)

    def force(
        self, state: FuglevandWinterPatla1993State, /
    ) -> FuglevandWinterPatla1993Force:
        """Read this route's terminal isometric force without rescaling."""

        if not isinstance(state, FuglevandWinterPatla1993State):
            raise TypeError("state must be FuglevandWinterPatla1993State.")
        if state.model_id != self.plan.model_id:
            raise ValueError("state does not belong to this prepared model.")
        return FuglevandWinterPatla1993Force(
            state.motor_unit_force,
            jnp.sum(state.motor_unit_force),
            state.time_ms,
            state.model_id,
        )


def commit_fuglevand_winter_patla_1993(
    candidate: FuglevandWinterPatla1993Candidate,
    current: FuglevandWinterPatla1993State,
    /,
) -> FuglevandWinterPatla1993State:
    """Atomically commit a successful candidate from exactly ``current``."""

    if not isinstance(candidate, FuglevandWinterPatla1993Candidate):
        raise TypeError("candidate must be FuglevandWinterPatla1993Candidate.")
    if not isinstance(current, FuglevandWinterPatla1993State):
        raise TypeError("current must be FuglevandWinterPatla1993State.")
    source = candidate.source
    source_matches = (
        (source.model_id == current.model_id)
        & (source.time_ms == current.time_ms)
        & (source.random_step == current.random_step)
        & (source.step_index == current.step_index)
        & jnp.array_equal(source.last_discharge_ms, current.last_discharge_ms)
        & jnp.array_equal(source.next_discharge_ms, current.next_discharge_ms)
        & jnp.array_equal(source.twitch_driver, current.twitch_driver)
        & jnp.array_equal(source.motor_unit_force, current.motor_unit_force)
    )
    return jax.lax.cond(
        candidate.evidence.successful & source_matches,
        lambda _: candidate.proposed,
        lambda _: current,
        operand=None,
    )


def fuglevand_force_variability_evidence(
    force_samples_arbitrary: ArrayLike, /
) -> "FuglevandForceVariabilityEvidence":
    """Summarize realized force variability without claiming a population law."""

    samples = jnp.asarray(force_samples_arbitrary)
    if samples.ndim != 1 or samples.shape[0] < 2:
        raise ValueError("force_samples_arbitrary must be a vector with at least two samples.")
    if not jnp.issubdtype(samples.dtype, jnp.inexact):
        samples = samples.astype(float)
    mean = jnp.mean(samples)
    standard_deviation = jnp.std(samples, ddof=1)
    coefficient = standard_deviation / jnp.maximum(
        jnp.abs(mean), jnp.finfo(samples.dtype).tiny
    )
    finite = jnp.all(jnp.isfinite(samples)) & jnp.isfinite(coefficient)
    return FuglevandForceVariabilityEvidence(
        mean,
        standard_deviation,
        coefficient,
        jnp.asarray(samples.shape[0], dtype=jnp.int32),
        finite,
        "descriptive statistics of this realized force trace; not a universal physiological distribution",
    )


class FuglevandForceVariabilityEvidence(StrictModule, NonTrainableState):
    """Descriptive force statistics for one realized stochastic trace."""

    mean_force_arbitrary: Array
    standard_deviation_force_arbitrary: Array
    coefficient_of_variation: Array
    sample_count: Array
    finite: Array
    claim_scope: str = eqx.field(static=True)


__all__ = [
    "FuglevandForceVariabilityEvidence",
    "FuglevandWinterPatla1993Candidate",
    "FuglevandWinterPatla1993Evidence",
    "FuglevandWinterPatla1993Force",
    "FuglevandWinterPatla1993Parameters",
    "FuglevandWinterPatla1993Plan",
    "FuglevandWinterPatla1993RandomInput",
    "FuglevandWinterPatla1993State",
    "FuglevandWinterPatla1993Status",
    "PreparedFuglevandWinterPatla1993",
    "commit_fuglevand_winter_patla_1993",
    "fuglevand_force_variability_evidence",
]
