#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from .._strict import StrictModule


HawkesTiePolicy: TypeAlias = Literal["simultaneous", "ordered"]
HawkesLikelihoodStatus: TypeAlias = Literal[0, 1, 2, 3]

HAWKES_SUCCESS = 0
HAWKES_UNSTABLE = 1
HAWKES_NONPOSITIVE_INTENSITY = 2
HAWKES_NONFINITE = 3


class PointProcessObservation(StrictModule):
    """A fixed-capacity marked point-process observation window."""

    times: Array
    channels: Array
    valid: Array
    start_time: Array
    end_time: Array
    capacity: int = eqx.field(static=True)
    channel_count: int = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        channels: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        start_time: ArrayLike,
        end_time: ArrayLike,
        channel_count: int,
    ):
        times_ = jnp.asarray(times)
        channels_ = jnp.asarray(channels, dtype=jnp.int32)
        valid_ = jnp.asarray(valid, dtype=bool)
        if times_.ndim != 1:
            raise ValueError("times must have shape (capacity,).")
        if channels_.shape != times_.shape or valid_.shape != times_.shape:
            raise ValueError("channels and valid must have the same shape as times.")
        if not jnp.issubdtype(times_.dtype, jnp.inexact):
            times_ = times_.astype(float)
        channels_count = int(channel_count)
        if channels_count < 1:
            raise ValueError("channel_count must be positive.")
        start = jnp.asarray(start_time, dtype=times_.dtype)
        end = jnp.asarray(end_time, dtype=times_.dtype)
        if start.ndim != 0 or end.ndim != 0:
            raise ValueError("start_time and end_time must be scalars.")
        times_ = eqx.error_if(
            times_,
            ~jnp.isfinite(start) | ~jnp.isfinite(end) | (end <= start),
            "observation bounds must be finite and end_time must exceed start_time.",
        )
        invalid_prefix = jnp.any(valid_[1:] & ~valid_[:-1])
        invalid_time = jnp.any(
            valid_ & (~jnp.isfinite(times_) | (times_ < start) | (times_ >= end))
        )
        invalid_channel = jnp.any(
            valid_ & ((channels_ < 0) | (channels_ >= channels_count))
        )
        safe_times = jnp.where(valid_, times_, end)
        out_of_order = jnp.any(valid_[1:] & (safe_times[1:] < safe_times[:-1]))
        times_ = eqx.error_if(
            times_,
            invalid_prefix | invalid_time | invalid_channel | out_of_order,
            "active events must form a finite, ordered prefix inside the window with valid channels.",
        )
        self.times = times_
        self.channels = channels_
        self.valid = valid_
        self.start_time = start
        self.end_time = end
        self.capacity = int(times_.shape[0])
        self.channel_count = channels_count

    @property
    def event_count(self) -> Array:
        return jnp.sum(self.valid).astype(jnp.int32)


class ExponentialHawkesProcess(StrictModule):
    """Linear multivariate Hawkes intensity with exponential kernels.

    ``excitation[target, source]`` is the instantaneous intensity jump caused by
    one source event. ``decay[target, source]`` is its exponential decay rate.
    """

    baseline: Array
    excitation: Array
    decay: Array
    branching_matrix: Array
    spectral_radius: Array
    stable: Array
    channel_count: int = eqx.field(static=True)

    def __init__(
        self,
        baseline: ArrayLike,
        excitation: ArrayLike,
        decay: ArrayLike,
        /,
    ):
        baseline_ = jnp.asarray(baseline)
        excitation_ = jnp.asarray(excitation)
        decay_ = jnp.asarray(decay)
        if baseline_.ndim != 1 or baseline_.shape[0] < 1:
            raise ValueError("baseline must have shape (channels,).")
        channels = int(baseline_.shape[0])
        if excitation_.shape != (channels, channels):
            raise ValueError("excitation must have shape (channels, channels).")
        if decay_.shape not in ((), (channels, channels)):
            raise ValueError("decay must be scalar or have shape (channels, channels).")
        dtype = jnp.result_type(baseline_, excitation_, decay_, float)
        baseline_ = baseline_.astype(dtype)
        excitation_ = excitation_.astype(dtype)
        decay_ = jnp.broadcast_to(decay_.astype(dtype), (channels, channels))
        invalid = (
            jnp.any(~jnp.isfinite(baseline_) | (baseline_ <= 0.0))
            | jnp.any(~jnp.isfinite(excitation_) | (excitation_ < 0.0))
            | jnp.any(~jnp.isfinite(decay_) | (decay_ <= 0.0))
        )
        baseline_ = eqx.error_if(
            baseline_,
            invalid,
            "baseline and decay must be finite and positive; excitation must be finite and nonnegative.",
        )
        branching = excitation_ / decay_
        radius = jnp.max(jnp.abs(jnp.linalg.eigvals(branching))).real
        self.baseline = baseline_
        self.excitation = excitation_
        self.decay = decay_
        self.branching_matrix = branching
        self.spectral_radius = radius
        self.stable = jnp.isfinite(radius) & (radius < 1.0)
        self.channel_count = channels


class HawkesLikelihoodPlan(StrictModule):
    """Prepared shape and same-time convention for Hawkes likelihood evaluation."""

    capacity: int = eqx.field(static=True)
    channel_count: int = eqx.field(static=True)
    tie_policy: HawkesTiePolicy = eqx.field(static=True)
    require_stable: bool = eqx.field(static=True)

    def __init__(
        self,
        capacity: int,
        channel_count: int,
        /,
        *,
        tie_policy: HawkesTiePolicy = "simultaneous",
        require_stable: bool = True,
    ):
        capacity_ = int(capacity)
        channels = int(channel_count)
        if capacity_ < 1 or channels < 1:
            raise ValueError("capacity and channel_count must be positive.")
        if tie_policy not in ("simultaneous", "ordered"):
            raise ValueError("tie_policy must be 'simultaneous' or 'ordered'.")
        self.capacity = capacity_
        self.channel_count = channels
        self.tie_policy = tie_policy
        self.require_stable = bool(require_stable)


class HawkesLikelihoodResult(StrictModule):
    """Exact exponential-Hawkes log likelihood and compensator evidence."""

    log_likelihood: Array
    event_log_likelihood: Array
    compensator: Array
    event_intensity: Array
    channel_intensity: Array
    event_mask: Array
    event_count: Array
    tied_event_count: Array
    spectral_radius: Array
    stable: Array
    finite: Array
    positive_intensity: Array
    status: Array
    plan: HawkesLikelihoodPlan = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == HAWKES_SUCCESS


class HawkesProcessResult(StrictModule):
    """Fixed-capacity simulation with total-count and overflow evidence."""

    observation: PointProcessObservation
    generated_event_count: Array
    stored_event_count: Array
    proposal_count: Array
    overflow: Array
    stable: Array
    max_proposals: int = eqx.field(static=True)


def prepare_hawkes_likelihood(
    observation: PointProcessObservation,
    process: ExponentialHawkesProcess,
    /,
    *,
    tie_policy: HawkesTiePolicy = "simultaneous",
    require_stable: bool = True,
) -> HawkesLikelihoodPlan:
    """Bind observation and channel shapes before numerical likelihood evaluation."""

    if not isinstance(observation, PointProcessObservation):
        raise TypeError("observation must be a PointProcessObservation.")
    if not isinstance(process, ExponentialHawkesProcess):
        raise TypeError("process must be an ExponentialHawkesProcess.")
    if observation.channel_count != process.channel_count:
        raise ValueError("observation and process channel counts must match.")
    return HawkesLikelihoodPlan(
        observation.capacity,
        observation.channel_count,
        tie_policy=tie_policy,
        require_stable=require_stable,
    )


def evaluate_hawkes_likelihood(
    observation: PointProcessObservation,
    process: ExponentialHawkesProcess,
    /,
    *,
    plan: HawkesLikelihoodPlan | None = None,
) -> HawkesLikelihoodResult:
    """Evaluate the exact finite-window likelihood of an exponential Hawkes model."""

    if not isinstance(observation, PointProcessObservation):
        raise TypeError("observation must be a PointProcessObservation.")
    if not isinstance(process, ExponentialHawkesProcess):
        raise TypeError("process must be an ExponentialHawkesProcess.")
    plan_ = prepare_hawkes_likelihood(observation, process) if plan is None else plan
    if not isinstance(plan_, HawkesLikelihoodPlan):
        raise TypeError("plan must be a HawkesLikelihoodPlan or None.")
    if (
        plan_.capacity != observation.capacity
        or plan_.channel_count != observation.channel_count
        or plan_.channel_count != process.channel_count
    ):
        raise ValueError("plan shapes must match observation and process.")

    times = observation.times
    channels = jnp.where(observation.valid, observation.channels, 0)
    initial_history = jnp.zeros_like(process.excitation)
    initial_pending = jnp.zeros_like(process.excitation)

    def likelihood_step(carry, inputs):
        history, pending, previous_time = carry
        time, channel, valid = inputs
        elapsed = jnp.maximum(time - previous_time, 0.0)
        if plan_.tie_policy == "simultaneous":
            next_group = elapsed > 0.0
            advanced = (history + pending) * jnp.exp(-process.decay * elapsed)
            evaluated_history = jnp.where(next_group, advanced, history)
            pending = jnp.where(next_group, 0.0, pending)
            intensity = process.baseline + jnp.sum(evaluated_history, axis=-1)
            increment = (
                jnp.zeros_like(pending).at[:, channel].set(process.excitation[:, channel])
            )
            next_history = evaluated_history
            next_pending = pending + increment
        else:
            evaluated_history = history * jnp.exp(-process.decay * elapsed)
            intensity = process.baseline + jnp.sum(evaluated_history, axis=-1)
            increment = (
                jnp.zeros_like(history).at[:, channel].set(process.excitation[:, channel])
            )
            next_history = evaluated_history + increment
            next_pending = pending
        return (
            (
                jnp.where(valid, next_history, history),
                jnp.where(valid, next_pending, pending),
                jnp.where(valid, time, previous_time),
            ),
            jnp.where(valid, intensity, process.baseline),
        )

    _, channel_intensity = jax.lax.scan(
        likelihood_step,
        (initial_history, initial_pending, observation.start_time),
        (times, channels, observation.valid),
    )
    event_intensity = jnp.take_along_axis(channel_intensity, channels[:, None], axis=-1)[
        :, 0
    ]
    positive = jnp.all(jnp.where(observation.valid, event_intensity > 0.0, True))
    event_log = jnp.where(
        observation.valid,
        jnp.log(jnp.maximum(event_intensity, jnp.finfo(times.dtype).tiny)),
        0.0,
    )

    remaining = observation.end_time - times
    integrated_event = (
        process.excitation[..., None]
        / process.decay[..., None]
        * (1.0 - jnp.exp(-process.decay[..., None] * remaining[None, None, :]))
    )
    integrated_by_source = jnp.sum(integrated_event, axis=0)
    event_compensator = integrated_by_source[channels, jnp.arange(observation.capacity)]
    compensator = jnp.sum(process.baseline) * (
        observation.end_time - observation.start_time
    ) + jnp.sum(jnp.where(observation.valid, event_compensator, 0.0))
    likelihood = jnp.sum(event_log) - compensator
    finite = (
        jnp.isfinite(likelihood)
        & jnp.isfinite(compensator)
        & jnp.all(jnp.where(observation.valid, jnp.isfinite(event_intensity), True))
    )
    stable = process.stable
    status = jnp.where(
        ~finite,
        HAWKES_NONFINITE,
        jnp.where(
            ~positive,
            HAWKES_NONPOSITIVE_INTENSITY,
            jnp.where(plan_.require_stable & ~stable, HAWKES_UNSTABLE, HAWKES_SUCCESS),
        ),
    ).astype(jnp.int32)
    tied = jnp.sum(
        observation.valid[1:] & observation.valid[:-1] & (times[1:] == times[:-1])
    ).astype(jnp.int32)
    return HawkesLikelihoodResult(
        log_likelihood=likelihood,
        event_log_likelihood=event_log,
        compensator=compensator,
        event_intensity=event_intensity,
        channel_intensity=channel_intensity,
        event_mask=observation.valid,
        event_count=observation.event_count,
        tied_event_count=tied,
        spectral_radius=process.spectral_radius,
        stable=stable,
        finite=finite,
        positive_intensity=positive,
        status=status,
        plan=plan_,
    )


def simulate_hawkes(
    process: ExponentialHawkesProcess,
    key: PRNGKeyArray,
    /,
    *,
    start_time: float,
    end_time: float,
    capacity: int,
    max_proposals: int = 4096,
    require_stable: bool = True,
) -> HawkesProcessResult:
    """Simulate by Ogata thinning while retaining capacity-overflow evidence."""

    if not isinstance(process, ExponentialHawkesProcess):
        raise TypeError("process must be an ExponentialHawkesProcess.")
    start = float(start_time)
    end = float(end_time)
    capacity_ = int(capacity)
    proposals = int(max_proposals)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("finite end_time must exceed start_time.")
    if capacity_ < 1 or proposals < 1:
        raise ValueError("capacity and max_proposals must be positive.")
    baseline = process.baseline
    if require_stable:
        baseline = eqx.error_if(
            baseline,
            ~process.stable,
            "stable Hawkes simulation requires spectral radius below one.",
        )
    keys = jr.split(key, proposals * 2).reshape((proposals, 2) + key.shape)
    dtype = baseline.dtype
    initial_times = jnp.zeros((capacity_,), dtype=dtype)
    initial_channels = jnp.zeros((capacity_,), dtype=jnp.int32)
    initial_excitation = jnp.zeros_like(process.excitation)

    def body(index, state):
        (
            current_time,
            excitation_state,
            times,
            channels,
            generated,
            proposal_count,
            active,
        ) = state
        intensity_upper = baseline + jnp.sum(excitation_state, axis=-1)
        rate_upper = jnp.sum(intensity_upper)
        wait = jr.exponential(keys[index, 0], dtype=dtype) / rate_upper
        candidate = current_time + wait
        inside = active & (candidate < end)
        proposal_count = proposal_count + active.astype(jnp.int32)
        elapsed = jnp.where(inside, wait, 0.0)
        decayed = excitation_state * jnp.exp(-process.decay * elapsed)
        intensity = baseline + jnp.sum(decayed, axis=-1)
        rate = jnp.sum(intensity)
        accept_probability = jnp.minimum(rate / rate_upper, 1.0)
        uniform = jr.uniform(keys[index, 1], shape=(2,), dtype=dtype)
        accepted = inside & (uniform[0] < accept_probability)
        cumulative = jnp.cumsum(intensity) / rate
        channel = jnp.searchsorted(cumulative, uniform[1], side="right")
        channel = jnp.minimum(channel, process.channel_count - 1).astype(jnp.int32)
        store = accepted & (generated < capacity_)
        slot = jnp.minimum(generated, capacity_ - 1)
        times = times.at[slot].set(jnp.where(store, candidate, times[slot]))
        channels = channels.at[slot].set(jnp.where(store, channel, channels[slot]))
        column = jax.nn.one_hot(channel, process.channel_count, dtype=dtype)
        updated_excitation = decayed + process.excitation * column[None, :]
        excitation_state = jnp.where(accepted, updated_excitation, decayed)
        generated = generated + accepted.astype(jnp.int32)
        current_time = jnp.where(inside, candidate, current_time)
        active = active & inside
        return (
            current_time,
            excitation_state,
            times,
            channels,
            generated,
            proposal_count,
            active,
        )

    initial = (
        jnp.asarray(start, dtype=dtype),
        initial_excitation,
        initial_times,
        initial_channels,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(True),
    )
    _, _, times, channels, generated, proposal_count, active = jax.lax.fori_loop(
        0, proposals, body, initial
    )
    stored = jnp.minimum(generated, capacity_)
    valid = jnp.arange(capacity_) < stored
    observation = PointProcessObservation(
        times,
        channels,
        valid,
        start_time=jnp.asarray(start, dtype=dtype),
        end_time=jnp.asarray(end, dtype=dtype),
        channel_count=process.channel_count,
    )
    return HawkesProcessResult(
        observation=observation,
        generated_event_count=generated,
        stored_event_count=stored,
        proposal_count=proposal_count,
        overflow=generated > capacity_,
        stable=process.stable,
        max_proposals=proposals,
    )


__all__ = [
    "HAWKES_NONFINITE",
    "HAWKES_NONPOSITIVE_INTENSITY",
    "HAWKES_SUCCESS",
    "HAWKES_UNSTABLE",
    "ExponentialHawkesProcess",
    "HawkesLikelihoodPlan",
    "HawkesLikelihoodResult",
    "HawkesProcessResult",
    "HawkesTiePolicy",
    "PointProcessObservation",
    "evaluate_hawkes_likelihood",
    "prepare_hawkes_likelihood",
    "simulate_hawkes",
]
