#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...stochastic import (
    GaussianStatePrior,
    LinearGaussianObservationModel,
    LinearGaussianTransitionKernel,
    ObservationSequence,
    StateSpaceModel,
    StateSpaceProblem,
)
from ...uq import kalman_filter, rts_smoother
from ._types import TrackResult, TrackSmoothingResult


class TrackSmoothingPlan(StrictModule, NonTrainableState):
    """Frozen-association constant-velocity Kalman/RTS policy."""

    process_acceleration_variance: float = eqx.field(static=True)
    initial_velocity_variance: float = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)
    execution_method: Literal["sequential", "parallel", "auto"] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        process_acceleration_variance: float = 1e-3,
        initial_velocity_variance: float = 1.0,
        covariance_regularization: float = 0.0,
        execution_method: Literal["sequential", "parallel", "auto"] = "auto",
    ):
        values = jnp.asarray(
            (
                process_acceleration_variance,
                initial_velocity_variance,
                covariance_regularization,
            )
        )
        if not bool(jnp.all(jnp.isfinite(values))) or bool(jnp.any(values < 0.0)):
            raise ValueError(
                "Smoothing variances and regularization must be finite and nonnegative."
            )
        if execution_method not in ("sequential", "parallel", "auto"):
            raise ValueError(
                "execution_method must be 'sequential', 'parallel', or 'auto'."
            )
        self.process_acceleration_variance = float(process_acceleration_variance)
        self.initial_velocity_variance = float(initial_velocity_variance)
        self.covariance_regularization = float(covariance_regularization)
        self.execution_method = execution_method
        self.plan_id = canonical_fingerprint(
            {
                "kind": "frozen-particle-track-rts",
                "process_acceleration_variance": self.process_acceleration_variance,
                "initial_velocity_variance": self.initial_velocity_variance,
                "covariance_regularization": self.covariance_regularization,
                "execution_method": self.execution_method,
            }
        )


def _transition_matrix(start, end, context, /):
    del context
    dt = end - start
    identity = jnp.eye(3, dtype=jnp.result_type(start, end, float))
    return jnp.eye(6, dtype=identity.dtype).at[:3, 3:].set(dt * identity)


def _process_covariance(acceleration_variance: float):
    def covariance(start, end, context, /):
        del context
        dt = end - start
        identity = jnp.eye(3, dtype=jnp.result_type(start, end, float))
        value = jnp.zeros((6, 6), dtype=identity.dtype)
        value = value.at[:3, :3].set(0.25 * dt**4 * identity)
        value = value.at[:3, 3:].set(0.5 * dt**3 * identity)
        value = value.at[3:, :3].set(0.5 * dt**3 * identity)
        value = value.at[3:, 3:].set(dt**2 * identity)
        return acceleration_variance * value

    return covariance


def _observation_covariance(values):
    def covariance(time, context, /):
        del time
        return values[context.step_index]

    return covariance


def _segments(result: TrackResult, /) -> list[tuple[int, int, int]]:
    active = np.asarray(jax.device_get(result.active), dtype=bool)
    identifiers = np.asarray(jax.device_get(result.track_ids), dtype=np.int64)
    segments: list[tuple[int, int, int]] = []
    for slot in range(result.track_capacity):
        start = 0
        while start < active.shape[1]:
            while start < active.shape[1] and not active[slot, start]:
                start += 1
            if start == active.shape[1]:
                break
            identifier = identifiers[slot, start]
            end = start + 1
            while (
                end < active.shape[1]
                and active[slot, end]
                and identifiers[slot, end] == identifier
            ):
                end += 1
            segments.append((slot, start, end))
            start = end
    return segments


def smooth_tracks(
    result: TrackResult,
    plan: TrackSmoothingPlan,
    /,
) -> TrackSmoothingResult:
    """Smooth each frozen identity segment through Phydrax's Kalman substrate."""
    if not isinstance(result, TrackResult):
        raise TypeError("result must be a TrackResult.")
    if not isinstance(plan, TrackSmoothingPlan):
        raise TypeError("plan must be a TrackSmoothingPlan.")
    smoothed_states = jnp.zeros_like(result.states)
    smoothed_covariances = jnp.zeros_like(result.covariances)
    smoothed_valid = jnp.zeros_like(result.active)
    innovations = jnp.zeros(result.states.shape[:2] + (3,), dtype=result.states.dtype)
    innovation_covariances = jnp.zeros(
        result.states.shape[:2] + (3, 3), dtype=result.states.dtype
    )
    segment_status = jnp.full(result.states.shape[:2], -1, dtype=jnp.int32)
    backend_results = []
    observation_matrix = (
        jnp.zeros((3, 6), dtype=result.states.dtype)
        .at[:, :3]
        .set(jnp.eye(3, dtype=result.states.dtype))
    )
    for segment_index, (slot, start, end) in enumerate(_segments(result)):
        times = result.times[start:end]
        observed = result.observed[slot, start:end]
        values = jnp.where(
            observed[:, None],
            result.observations[slot, start:end],
            result.states[slot, start:end, :3],
        )
        measurement_covariance = jnp.where(
            observed[:, None, None],
            result.observation_covariances[slot, start:end],
            0.0,
        )
        sequence = ObservationSequence(
            times,
            values,
            observation_mask=jnp.broadcast_to(observed[:, None], values.shape),
            sequence_id=f"track-observations:{result.result_id}:{slot}:{segment_index}",
            sensor_id="multiview-particle-reconstruction",
        )
        initial_mean = result.states[slot, start]
        initial_covariance = result.covariances[slot, start]
        initial_covariance = initial_covariance.at[3:, 3:].add(
            plan.initial_velocity_variance * jnp.eye(3, dtype=result.states.dtype)
        )
        prior = GaussianStatePrior(
            initial_mean,
            initial_covariance,
            state_shape=(6,),
            prior_id=f"track-prior:{result.result_id}:{slot}:{segment_index}",
        )
        transition = LinearGaussianTransitionKernel(
            _transition_matrix,
            _process_covariance(plan.process_acceleration_variance),
            state_shape=(6,),
            process_id="constant-velocity-particle",
            approximation_id="exact-discrete-white-acceleration",
        )
        observation = LinearGaussianObservationModel(
            observation_matrix,
            _observation_covariance(measurement_covariance),
            state_shape=(6,),
            observation_shape=(3,),
            observation_id="particle-position",
        )
        model = StateSpaceModel(
            prior,
            transition,
            observation,
            model_id=f"frozen-track-model:{plan.plan_id}:{slot}:{segment_index}",
        )
        problem = StateSpaceProblem(
            model,
            sequence,
            initial_time=times[0],
            problem_id=f"frozen-track-smoothing:{result.result_id}:{plan.plan_id}:{slot}:{segment_index}",
        )
        filtered = kalman_filter(
            problem,
            method=plan.execution_method,
            covariance_regularization=plan.covariance_regularization,
            raise_on_failure=False,
        )
        smoothed = rts_smoother(filtered, method=plan.execution_method)
        smoothed_states = smoothed_states.at[slot, start:end].set(smoothed.means)
        smoothed_covariances = smoothed_covariances.at[slot, start:end].set(
            smoothed.covariances
        )
        smoothed_valid = smoothed_valid.at[slot, start:end].set(smoothed.valid)
        innovations = innovations.at[slot, start:end].set(filtered.innovations)
        innovation_covariances = innovation_covariances.at[slot, start:end].set(
            filtered.innovation_covariances
        )
        segment_status = segment_status.at[slot, start:end].set(filtered.status)
        backend_results.append(smoothed)
    smoothing_id = "track-smoothing:" + canonical_fingerprint(
        {
            "track_result": result.result_id,
            "plan": plan.plan_id,
            "segments": tuple(_segments(result)),
        }
    )
    return TrackSmoothingResult(
        smoothed_states,
        smoothed_covariances,
        smoothed_valid,
        innovations,
        innovation_covariances,
        segment_status,
        tuple(backend_results),
        smoothing_id,
        result.result_id,
    )


__all__ = ["TrackSmoothingPlan", "smooth_tracks"]
