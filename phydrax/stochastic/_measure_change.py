#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._jump import JumpEventBatch
from ._wiener import WienerRealization


MeasureChangeKind: TypeAlias = Literal["diffusion", "jump"]


def _model_id(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be a non-empty string.")
    return identifier


def _broadcast_steps(
    value: ArrayLike,
    shape: tuple[int, ...],
    /,
    *,
    name: str,
) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape == (shape[-1],):
        return jnp.broadcast_to(array, shape)
    try_shape = jnp.broadcast_shapes(array.shape, shape)
    if try_shape != shape:
        raise ValueError(f"{name} must broadcast to {shape}; got {array.shape}.")
    return jnp.broadcast_to(array, shape)


class DiffusionMeasureChange(StrictModule):
    """Pathwise Girsanov likelihood ratio for a Brownian diffusion proposal.

    ``control`` follows the convention
    ``target_drift = proposal_drift + diffusion @ control``. Consequently,
    ``log_likelihood_ratio`` is the log density of the target path law relative to
    the proposal path law.
    """

    log_likelihood_ratio: Array
    stochastic_integral: Array
    quadratic_variation: Array
    valid: Array
    proposal_model_id: str = eqx.field(static=True)
    target_model_id: str = eqx.field(static=True)
    kind: MeasureChangeKind = eqx.field(static=True)

    def __init__(
        self,
        log_likelihood_ratio: ArrayLike,
        stochastic_integral: ArrayLike,
        quadratic_variation: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        proposal_model_id: str,
        target_model_id: str,
    ):
        log_ratio = jnp.asarray(log_likelihood_ratio, dtype=float)
        stochastic = jnp.asarray(stochastic_integral, dtype=float)
        quadratic = jnp.asarray(quadratic_variation, dtype=float)
        validity = jnp.asarray(valid, dtype=bool)
        if any(
            value.shape != log_ratio.shape for value in (stochastic, quadratic, validity)
        ):
            raise ValueError("Diffusion measure-change arrays must have equal shapes.")
        self.log_likelihood_ratio = log_ratio
        self.stochastic_integral = stochastic
        self.quadratic_variation = quadratic
        self.valid = validity
        self.proposal_model_id = _model_id(proposal_model_id, "proposal_model_id")
        self.target_model_id = _model_id(target_model_id, "target_model_id")
        self.kind = "diffusion"

    @property
    def likelihood_ratio(self) -> Array:
        return jnp.exp(self.log_likelihood_ratio)

    @property
    def support_valid(self) -> Array:
        return self.valid


class JumpMeasureChange(StrictModule):
    """Likelihood ratio for a piecewise-constant marked jump-process proposal."""

    log_likelihood_ratio: Array
    event_log_ratio: Array
    mark_log_ratio: Array
    compensator: Array
    valid: Array
    support_valid: Array
    proposal_model_id: str = eqx.field(static=True)
    target_model_id: str = eqx.field(static=True)
    kind: MeasureChangeKind = eqx.field(static=True)

    def __init__(
        self,
        log_likelihood_ratio: ArrayLike,
        event_log_ratio: ArrayLike,
        mark_log_ratio: ArrayLike,
        compensator: ArrayLike,
        valid: ArrayLike,
        support_valid: ArrayLike,
        /,
        *,
        proposal_model_id: str,
        target_model_id: str,
    ):
        log_ratio = jnp.asarray(log_likelihood_ratio, dtype=float)
        event_term = jnp.asarray(event_log_ratio, dtype=float)
        mark_term = jnp.asarray(mark_log_ratio, dtype=float)
        compensator_term = jnp.asarray(compensator, dtype=float)
        validity = jnp.asarray(valid, dtype=bool)
        support = jnp.asarray(support_valid, dtype=bool)
        if any(
            value.shape != log_ratio.shape
            for value in (
                event_term,
                mark_term,
                compensator_term,
                validity,
                support,
            )
        ):
            raise ValueError("Jump measure-change arrays must have equal shapes.")
        self.log_likelihood_ratio = log_ratio
        self.event_log_ratio = event_term
        self.mark_log_ratio = mark_term
        self.compensator = compensator_term
        self.valid = validity
        self.support_valid = support
        self.proposal_model_id = _model_id(proposal_model_id, "proposal_model_id")
        self.target_model_id = _model_id(target_model_id, "target_model_id")
        self.kind = "jump"

    @property
    def likelihood_ratio(self) -> Array:
        return jnp.exp(self.log_likelihood_ratio)


PathMeasureChange: TypeAlias = DiffusionMeasureChange | JumpMeasureChange


def diffusion_measure_change(
    control: ArrayLike,
    driver_increments: ArrayLike,
    durations: ArrayLike,
    *,
    valid: ArrayLike | None = None,
    proposal_model_id: str = "proposal",
    target_model_id: str = "target",
) -> DiffusionMeasureChange:
    """Evaluate a discrete Girsanov density from left-point controls.

    Controls and increments have shape ``path_shape + (step, noise)``. Durations have
    shape ``(step,)`` or ``path_shape + (step,)``. Every noise dimension is
    contracted before time accumulation. Invalid intervals invalidate the complete
    path; partial likelihood ratios are never returned as valid samples.
    """

    controls = jnp.asarray(control, dtype=float)
    increments = jnp.asarray(driver_increments, dtype=controls.dtype)
    if controls.shape != increments.shape or controls.ndim < 2:
        raise ValueError(
            "control and driver_increments must have equal shapes with step and noise axes."
        )
    step_axis = controls.ndim - 2
    path_shape = controls.shape[:step_axis]
    num_steps = controls.shape[step_axis]
    duration_shape = path_shape + (num_steps,)
    steps = _broadcast_steps(durations, duration_shape, name="durations")
    interval_valid = (
        jnp.ones(duration_shape, dtype=bool)
        if valid is None
        else jnp.broadcast_to(jnp.asarray(valid, dtype=bool), duration_shape)
    )
    finite_noise = jnp.all(
        jnp.isfinite(controls) & jnp.isfinite(increments),
        axis=tuple(range(step_axis + 1, controls.ndim)),
    )
    interval_valid = interval_valid & finite_noise & jnp.isfinite(steps) & (steps > 0.0)
    contraction_axes = tuple(range(step_axis + 1, controls.ndim))
    stochastic_steps = jnp.sum(controls * increments, axis=contraction_axes)
    quadratic_steps = 0.5 * jnp.sum(controls * controls, axis=contraction_axes) * steps
    stochastic = jnp.sum(jnp.where(interval_valid, stochastic_steps, 0.0), axis=-1)
    quadratic = jnp.sum(jnp.where(interval_valid, quadratic_steps, 0.0), axis=-1)
    path_valid = jnp.all(interval_valid, axis=-1)
    log_ratio = jnp.where(path_valid, stochastic - quadratic, -jnp.inf)
    return DiffusionMeasureChange(
        log_ratio,
        stochastic,
        quadratic,
        path_valid,
        proposal_model_id=proposal_model_id,
        target_model_id=target_model_id,
    )


def wiener_measure_change(
    realization: WienerRealization,
    times: ArrayLike,
    control: ArrayLike,
    /,
    *,
    valid: ArrayLike | None = None,
    proposal_model_id: str = "proposal",
    target_model_id: str = "target",
) -> DiffusionMeasureChange:
    """Evaluate a diffusion measure change on one explicit Wiener realization."""

    if not isinstance(realization, WienerRealization):
        raise TypeError("realization must be a WienerRealization.")
    nodes = jnp.asarray(times, dtype=float)
    if nodes.ndim != 1 or nodes.shape[0] < 2:
        raise ValueError("times must be a one-dimensional array with at least two nodes.")
    increments = realization.increments(nodes[:-1], nodes[1:])
    return diffusion_measure_change(
        control,
        increments,
        jnp.diff(nodes),
        valid=valid,
        proposal_model_id=proposal_model_id,
        target_model_id=target_model_id,
    )


def jump_measure_change(
    events: JumpEventBatch,
    interval_edges: ArrayLike,
    proposal_intensities: ArrayLike,
    target_intensities: ArrayLike,
    /,
    *,
    mark_log_ratios: ArrayLike | None = None,
    proposal_model_id: str = "proposal",
    target_model_id: str = "target",
) -> JumpMeasureChange:
    """Evaluate a marked jump-process density on a piecewise-constant partition.

    Intensity arrays have shape ``batch_shape + (interval, channel)``. ``interval_edges``
    may be shared or batch-shaped. Marks contribute an optional event-wise log density
    ratio. The returned density is target relative to proposal.
    """

    if not isinstance(events, JumpEventBatch):
        raise TypeError("events must be a JumpEventBatch.")
    proposal = jnp.asarray(proposal_intensities, dtype=float)
    target = jnp.asarray(target_intensities, dtype=proposal.dtype)
    if proposal.shape != target.shape or proposal.ndim < 2:
        raise ValueError(
            "proposal_intensities and target_intensities must have equal interval-channel shapes."
        )
    if proposal.shape[:-2] != events.batch_shape:
        raise ValueError("Intensity batch dimensions must match events.batch_shape.")
    num_intervals, num_channels = proposal.shape[-2:]
    edges = jnp.asarray(interval_edges, dtype=proposal.dtype)
    expected_edges = events.batch_shape + (num_intervals + 1,)
    if edges.shape == (num_intervals + 1,):
        edges = jnp.broadcast_to(edges, expected_edges)
    elif edges.shape != expected_edges:
        raise ValueError(
            f"interval_edges must have shape {(num_intervals + 1,)} or {expected_edges}."
        )
    durations = jnp.diff(edges, axis=-1)
    finite_intensities = (
        jnp.isfinite(proposal)
        & jnp.isfinite(target)
        & (proposal >= 0.0)
        & (target >= 0.0)
    )
    support = jnp.all(
        finite_intensities & ~((target > 0.0) & (proposal <= 0.0)),
        axis=(-2, -1),
    )
    partition_valid = jnp.all(jnp.isfinite(edges), axis=-1) & jnp.all(
        durations > 0.0,
        axis=-1,
    )
    support = support & partition_valid

    interval_indices = jnp.sum(
        events.times[..., :, None] >= edges[..., None, 1:],
        axis=-1,
    )
    interval_indices = jnp.clip(interval_indices, 0, num_intervals - 1)
    channel_indices = jnp.clip(events.channels, 0, num_channels - 1)
    flat_size = 1
    for size in events.batch_shape:
        flat_size *= size
    flat_proposal = proposal.reshape((flat_size, num_intervals, num_channels))
    flat_target = target.reshape((flat_size, num_intervals, num_channels))
    flat_intervals = interval_indices.reshape((flat_size, events.max_events))
    flat_channels = channel_indices.reshape((flat_size, events.max_events))

    def gather_path(
        intensities: Array,
        intervals: Array,
        channels: Array,
    ) -> Array:
        return intensities[intervals, channels]

    proposal_events = jax.vmap(gather_path)(
        flat_proposal,
        flat_intervals,
        flat_channels,
    ).reshape(events.times.shape)
    target_events = jax.vmap(gather_path)(
        flat_target,
        flat_intervals,
        flat_channels,
    ).reshape(events.times.shape)
    safe_proposal = jnp.where(proposal_events > 0.0, proposal_events, 1.0)
    safe_target = jnp.where(target_events > 0.0, target_events, 1.0)
    event_terms = jnp.where(
        events.valid,
        jnp.where(
            target_events > 0.0,
            jnp.log(safe_target) - jnp.log(safe_proposal),
            -jnp.inf,
        ),
        0.0,
    )
    event_log_ratio = jnp.sum(event_terms, axis=-1)
    marks = (
        jnp.zeros(events.times.shape, dtype=proposal.dtype)
        if mark_log_ratios is None
        else jnp.broadcast_to(
            jnp.asarray(mark_log_ratios, dtype=proposal.dtype),
            events.times.shape,
        )
    )
    mark_finite = jnp.all(
        jnp.where(
            events.valid,
            ~jnp.isnan(marks) & ~jnp.isposinf(marks),
            True,
        ),
        axis=-1,
    )
    mark_term = jnp.sum(jnp.where(events.valid, marks, 0.0), axis=-1)
    compensator = jnp.sum((target - proposal) * durations[..., :, None], axis=(-2, -1))
    event_proposal_valid = jnp.all(
        jnp.where(events.valid, proposal_events > 0.0, True),
        axis=-1,
    )
    event_channels_valid = jnp.all(
        jnp.where(
            events.valid,
            (events.channels >= 0) & (events.channels < num_channels),
            True,
        ),
        axis=-1,
    )
    valid = (
        events.successful
        & support
        & event_proposal_valid
        & event_channels_valid
        & mark_finite
        & jnp.isfinite(compensator)
    )
    log_ratio = jnp.where(
        valid,
        event_log_ratio + mark_term - compensator,
        -jnp.inf,
    )
    return JumpMeasureChange(
        log_ratio,
        event_log_ratio,
        mark_term,
        compensator,
        valid,
        support,
        proposal_model_id=proposal_model_id,
        target_model_id=target_model_id,
    )


def measure_changed_target(
    samples: Any,
    change: PathMeasureChange,
    /,
    *,
    sample_axes: int | tuple[int, ...] = 0,
    independent: bool = False,
):
    """Expose a path-law change as Phydrax's canonical weighted empirical measure."""

    from ..integration import weighted

    if not isinstance(change, (DiffusionMeasureChange, JumpMeasureChange)):
        raise TypeError("change must be a diffusion or jump measure-change result.")
    axes = (sample_axes,) if isinstance(sample_axes, int) else tuple(sample_axes)
    resolved_axes = tuple(
        axis + change.log_likelihood_ratio.ndim if axis < 0 else axis for axis in axes
    )
    support_valid = jnp.all(change.support_valid, axis=resolved_axes)
    return weighted(
        samples,
        change.log_likelihood_ratio,
        normalized=False,
        independent=independent,
        support_valid=support_valid,
        mask=change.valid,
        sample_axes=sample_axes,
        provenance=(
            f"measure-change:{change.kind}:"
            f"{change.proposal_model_id}->{change.target_model_id}"
        ),
    )


__all__ = [
    "diffusion_measure_change",
    "DiffusionMeasureChange",
    "jump_measure_change",
    "JumpMeasureChange",
    "measure_changed_target",
    "MeasureChangeKind",
    "PathMeasureChange",
    "wiener_measure_change",
]
