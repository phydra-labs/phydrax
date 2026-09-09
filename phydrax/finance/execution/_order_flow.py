#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Queue-reactive and exponential Hawkes order-flow references."""

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...stochastic._jump import JumpProcess


HawkesHistorySide: TypeAlias = Literal["left", "ordered"]


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _finite_vector(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{owner} must be a nonempty rank-one vector.")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must be real-valued.")
    array = array.astype(jnp.result_type(array, float))
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{owner} must be finite.")
    return array


def _finite_matrix(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{owner} must be a nonempty rank-two matrix.")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must be real-valued.")
    array = array.astype(jnp.result_type(array, float))
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{owner} must be finite.")
    return array


class QueueReactiveModel(StrictModule):
    """Exponential-link channel intensities driven by normalized queue depth."""

    baseline_intensity: Array
    queue_loading: Array
    action_loading: Array
    channel_queue_deltas: Array
    num_channels: int = eqx.field(static=True)
    num_queues: int = eqx.field(static=True)
    action_size: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        baseline_intensity: ArrayLike,
        queue_loading: ArrayLike,
        action_loading: ArrayLike,
        channel_queue_deltas: ArrayLike,
        /,
        *,
        model_id: str,
    ):
        baseline = _finite_vector(baseline_intensity, "baseline_intensity")
        queue_coefficients = _finite_matrix(queue_loading, "queue_loading")
        action_coefficients = _finite_matrix(action_loading, "action_loading")
        deltas = _finite_matrix(channel_queue_deltas, "channel_queue_deltas")
        channels = int(baseline.size)
        if bool(jnp.any(baseline < 0.0)):
            raise ValueError("baseline_intensity must be nonnegative.")
        if queue_coefficients.shape[0] != channels:
            raise ValueError("queue_loading must have one row per channel.")
        if action_coefficients.shape[0] != channels:
            raise ValueError("action_loading must have one row per channel.")
        if deltas.shape != queue_coefficients.shape:
            raise ValueError(
                "channel_queue_deltas must match the channel-by-queue layout."
            )
        self.baseline_intensity = baseline
        self.queue_loading = queue_coefficients
        self.action_loading = action_coefficients
        self.channel_queue_deltas = deltas
        self.num_channels = channels
        self.num_queues = int(queue_coefficients.shape[1])
        self.action_size = int(action_coefficients.shape[1])
        self.model_id = _identifier(model_id, "model_id")

    def intensities(self, queue_depth: ArrayLike, action: ArrayLike, /) -> Array:
        """Evaluate nonnegative intensities for one declared queue/action state."""

        queues = jnp.asarray(queue_depth)
        controls = jnp.asarray(action)
        if queues.shape != (self.num_queues,):
            raise ValueError(f"queue_depth must have shape ({self.num_queues},).")
        if controls.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},).")
        if not bool(jnp.all(jnp.isfinite(queues))) or bool(jnp.any(queues < 0.0)):
            raise ValueError("queue_depth must be finite and nonnegative.")
        if not bool(jnp.all(jnp.isfinite(controls))):
            raise ValueError("action must be finite.")
        total = jnp.sum(queues)
        normalized = jnp.where(total > 0.0, queues / total, jnp.zeros_like(queues))
        log_multiplier = self.queue_loading @ normalized + self.action_loading @ controls
        values = self.baseline_intensity * jnp.exp(log_multiplier)
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("Queue-reactive intensities overflowed or became nonfinite.")
        return values

    def apply_channel(self, queue_depth: ArrayLike, channel: ArrayLike, /) -> Array:
        """Apply one queue delta, masking decrements that would cross zero."""

        queues = jnp.asarray(queue_depth)
        channel_value = jnp.asarray(channel)
        if queues.shape != (self.num_queues,):
            raise ValueError(f"queue_depth must have shape ({self.num_queues},).")
        if channel_value.shape != () or not jnp.issubdtype(
            channel_value.dtype, jnp.integer
        ):
            raise TypeError("channel must be an integer scalar.")
        if not bool((channel_value >= 0) & (channel_value < self.num_channels)):
            raise ValueError("channel lies outside the declared channel layout.")
        delta = self.channel_queue_deltas[channel_value]
        candidate = queues + delta
        return jnp.where(candidate < 0.0, queues, candidate)


class QueueIntensityEvidence(StrictModule):
    """Finite/nonnegative diagnostics over caller-supplied queue/action probes."""

    minimum_intensity: Array
    maximum_intensity: Array
    finite: Array
    nonnegative: Array
    passed: Array
    model_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def diagnose_queue_intensities(
    model: QueueReactiveModel,
    queue_depths: ArrayLike,
    actions: ArrayLike,
    /,
) -> QueueIntensityEvidence:
    """Evaluate paired queue/action probes without extrapolating beyond them."""

    if not isinstance(model, QueueReactiveModel):
        raise TypeError("model must be a QueueReactiveModel.")
    queues = jnp.asarray(queue_depths)
    controls = jnp.asarray(actions)
    if queues.ndim != 2 or queues.shape[1] != model.num_queues:
        raise ValueError(f"queue_depths must have shape (probes, {model.num_queues}).")
    if controls.shape != (queues.shape[0], model.action_size):
        raise ValueError(
            f"actions must have shape ({queues.shape[0]}, {model.action_size})."
        )
    values = jnp.stack(
        tuple(model.intensities(queue, action) for queue, action in zip(queues, controls))
    )
    finite = jnp.all(jnp.isfinite(values))
    nonnegative = jnp.all(values >= 0.0)
    return QueueIntensityEvidence(
        minimum_intensity=jnp.min(values),
        maximum_intensity=jnp.max(values),
        finite=finite,
        nonnegative=nonnegative,
        passed=finite & nonnegative,
        model_id=model.model_id,
        scope="caller-supplied-queue-action-probes-only",
    )


def queue_reactive_jump_process(
    model: QueueReactiveModel,
    /,
    *,
    process_id: str,
) -> JumpProcess:
    """Adapt queue dynamics to the generic jump substrate used by control rollouts."""

    if not isinstance(model, QueueReactiveModel):
        raise TypeError("model must be a QueueReactiveModel.")

    def intensity(time, state, controlled_args):
        del time
        action, _ = controlled_args
        return model.intensities(state, action)

    def jump(state, channel, mark, controlled_args):
        del mark, controlled_args
        return model.apply_channel(state, channel)

    return JumpProcess(
        intensity,
        jump,
        state_shape=(model.num_queues,),
        num_channels=model.num_channels,
        process_id=_identifier(process_id, "process_id"),
    )


class HawkesOrderFlowModel(StrictModule):
    """Multichannel exponential Hawkes law with explicit branching diagnostics."""

    baseline_intensity: Array
    excitation: Array
    decay_rates: Array
    num_channels: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        baseline_intensity: ArrayLike,
        excitation: ArrayLike,
        decay_rates: ArrayLike,
        /,
        *,
        model_id: str,
    ):
        baseline = _finite_vector(baseline_intensity, "baseline_intensity")
        excitation_matrix = _finite_matrix(excitation, "excitation")
        decay = _finite_vector(decay_rates, "decay_rates")
        channels = int(baseline.size)
        if excitation_matrix.shape != (channels, channels):
            raise ValueError(
                "excitation must have shape (num_channels, num_channels), with "
                "rows as affected channels and columns as event channels."
            )
        if decay.shape != (channels,):
            raise ValueError("decay_rates must contain one rate per affected channel.")
        if bool(jnp.any(baseline < 0.0)) or bool(jnp.any(excitation_matrix < 0.0)):
            raise ValueError("Hawkes baseline and excitation must be nonnegative.")
        if bool(jnp.any(decay <= 0.0)):
            raise ValueError("Hawkes decay_rates must be strictly positive.")
        self.baseline_intensity = baseline
        self.excitation = excitation_matrix
        self.decay_rates = decay
        self.num_channels = channels
        self.model_id = _identifier(model_id, "model_id")

    @property
    def branching_matrix(self) -> Array:
        return self.excitation / self.decay_rates[:, None]

    def decay_excitation(
        self, excitation_state: ArrayLike, elapsed: ArrayLike, /
    ) -> Array:
        state = jnp.asarray(excitation_state)
        duration = jnp.asarray(elapsed)
        if state.shape != (self.num_channels,):
            raise ValueError(f"excitation_state must have shape ({self.num_channels},).")
        if (
            duration.shape != ()
            or not bool(jnp.isfinite(duration))
            or bool(duration < 0.0)
        ):
            raise ValueError("elapsed must be a finite nonnegative scalar.")
        if not bool(jnp.all(jnp.isfinite(state))) or bool(jnp.any(state < 0.0)):
            raise ValueError("excitation_state must be finite and nonnegative.")
        return state * jnp.exp(-self.decay_rates * duration)

    def apply_event(self, excitation_state: ArrayLike, channel: ArrayLike, /) -> Array:
        state = jnp.asarray(excitation_state)
        channel_value = jnp.asarray(channel)
        if state.shape != (self.num_channels,):
            raise ValueError(f"excitation_state must have shape ({self.num_channels},).")
        if channel_value.shape != () or not jnp.issubdtype(
            channel_value.dtype, jnp.integer
        ):
            raise TypeError("channel must be an integer scalar.")
        if not bool((channel_value >= 0) & (channel_value < self.num_channels)):
            raise ValueError("channel lies outside the declared channel layout.")
        return state + self.excitation[:, channel_value]


class HawkesStabilityEvidence(StrictModule):
    """Spectral-radius evidence for the declared linear exponential Hawkes law."""

    branching_matrix: Array
    spectral_radius: Array
    stability_margin: Array
    tolerance: Array
    finite: Array
    stable: Array
    model_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def diagnose_hawkes_stability(
    model: HawkesOrderFlowModel,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> HawkesStabilityEvidence:
    """Check subcriticality of the integrated excitation matrix."""

    if not isinstance(model, HawkesOrderFlowModel):
        raise TypeError("model must be a HawkesOrderFlowModel.")
    threshold = float(tolerance)
    if not isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    branching = np.asarray(model.branching_matrix, dtype=float)
    eigenvalues = np.linalg.eigvals(branching)
    radius = float(np.max(np.abs(eigenvalues)))
    finite = bool(np.isfinite(radius))
    margin = 1.0 - radius
    stable = finite and radius < 1.0 - threshold
    return HawkesStabilityEvidence(
        branching_matrix=jnp.asarray(branching),
        spectral_radius=jnp.asarray(radius),
        stability_margin=jnp.asarray(margin),
        tolerance=jnp.asarray(threshold),
        finite=jnp.asarray(finite),
        stable=jnp.asarray(stable),
        model_id=model.model_id,
        scope="linear-exponential-integrated-kernel-spectral-radius",
    )


class HawkesIntensityPath(StrictModule):
    """Causal intensities at supplied queries with an explicit history-side rule."""

    query_times: Array
    intensities: Array
    side: HawkesHistorySide = eqx.field(static=True)
    model_id: str = eqx.field(static=True)


def hawkes_intensity_path(
    model: HawkesOrderFlowModel,
    event_times: ArrayLike,
    event_channels: ArrayLike,
    query_times: ArrayLike,
    /,
    *,
    side: HawkesHistorySide = "left",
) -> HawkesIntensityPath:
    """Evaluate intensity from strictly prior or supplied-order prior events.

    ``left`` excludes events with time equal to a query. ``ordered`` includes all
    events at equal time, so it is intended only after their sequence has already
    been resolved by the caller.
    """

    if not isinstance(model, HawkesOrderFlowModel):
        raise TypeError("model must be a HawkesOrderFlowModel.")
    if side not in ("left", "ordered"):
        raise ValueError("side must be 'left' or 'ordered'.")
    times = jnp.asarray(event_times)
    if times.ndim != 1 or jnp.issubdtype(times.dtype, jnp.complexfloating):
        raise TypeError("event_times must be a real rank-one vector.")
    times = times.astype(jnp.result_type(times, float))
    if not bool(jnp.all(jnp.isfinite(times))):
        raise ValueError("event_times must be finite.")
    channels = jnp.asarray(event_channels)
    queries = _finite_vector(query_times, "query_times")
    if channels.shape != times.shape or not jnp.issubdtype(channels.dtype, jnp.integer):
        raise TypeError(
            "event_channels must be an integer vector aligned with event_times."
        )
    if bool(jnp.any(jnp.diff(times) < 0.0)):
        raise ValueError("event_times must be nondecreasing in resolved event order.")
    if bool(jnp.any((channels < 0) | (channels >= model.num_channels))):
        raise ValueError("event_channels contain an undeclared channel.")
    if bool(jnp.any(jnp.diff(queries) < 0.0)):
        raise ValueError("query_times must be nondecreasing.")
    values: list[Array] = []
    for query in queries:
        elapsed = query - times
        include = elapsed > 0.0 if side == "left" else elapsed >= 0.0
        contributions = model.excitation[:, channels] * jnp.exp(
            -model.decay_rates[:, None] * jnp.maximum(elapsed, 0.0)[None, :]
        )
        values.append(
            model.baseline_intensity
            + jnp.sum(jnp.where(include[None, :], contributions, 0.0), axis=-1)
        )
    return HawkesIntensityPath(
        query_times=queries,
        intensities=jnp.stack(values),
        side=side,
        model_id=model.model_id,
    )


__all__ = [
    "HawkesHistorySide",
    "HawkesIntensityPath",
    "HawkesOrderFlowModel",
    "HawkesStabilityEvidence",
    "QueueIntensityEvidence",
    "QueueReactiveModel",
    "diagnose_hawkes_stability",
    "diagnose_queue_intensities",
    "hawkes_intensity_path",
    "queue_reactive_jump_process",
]
