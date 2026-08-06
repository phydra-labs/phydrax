#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..integration import WeightedSampleTarget
from ._integration import _trajectory_arrays
from ._trajectory import StochasticTrajectory


TrajectoryStateTimeMode: TypeAlias = Literal["global", "per_time"]


def _independence_indices(
    labels: Sequence[str | None],
    /,
) -> tuple[int, ...]:
    resolved: dict[str, int] = {}
    out: list[int] = []
    for label in labels:
        if label is None:
            out.append(-1)
            continue
        if label not in resolved:
            resolved[label] = len(resolved)
        out.append(resolved[label])
    return tuple(out)


class TrajectoryStateTimeSamples(StrictModule):
    """Axis-preserving state-time particles with path dependence provenance."""

    states: cx.Field
    times: cx.Field
    valid: cx.Field
    log_weights: cx.Field
    path_indices: cx.Field
    independence_indices: cx.Field
    time_indices: cx.Field
    leading_axes: tuple[str, ...] = eqx.field(static=True)
    realization_axes: tuple[str, ...] = eqx.field(static=True)
    time_axis: str = eqx.field(static=True)
    state_axes: tuple[str, ...] = eqx.field(static=True)
    state_label: str = eqx.field(static=True)
    time_label: str = eqx.field(static=True)
    mode: TrajectoryStateTimeMode = eqx.field(static=True)
    path_labels: tuple[str, ...] = eqx.field(static=True)
    independence_labels: tuple[str | None, ...] = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.log_weights.shape)

    @property
    def num_nodes(self) -> int:
        count = 1
        for size in self.sample_shape:
            count *= size
        return count

    @property
    def num_paths(self) -> int:
        return len(self.path_labels)

    @property
    def num_times(self) -> int:
        return int(self.log_weights.named_shape[self.time_axis])

    @property
    def sample_axes(self) -> tuple[str, ...]:
        if self.mode == "global":
            return self.leading_axes + (self.time_axis,)
        return self.realization_axes

    @property
    def samples(self):
        return frozendict(
            {
                self.state_label: self.states,
                self.time_label: self.times,
                "path_index": self.path_indices,
                "independence_index": self.independence_indices,
                "time_index": self.time_indices,
            }
        )

    def target(self) -> WeightedSampleTarget:
        """Expose the structured particles through the generic weighted-target API."""
        return WeightedSampleTarget(
            self.samples,
            self.log_weights,
            normalized=True,
            independent=False,
            ancestry=self.path_indices,
            mask=self.valid,
            sample_axes=self.sample_axes,
            provenance=self.provenance,
        )


def trajectory_state_time_samples(
    trajectory: StochasticTrajectory,
    /,
    *,
    mode: TrajectoryStateTimeMode = "global",
    log_weights: ArrayLike | cx.Field | None = None,
    state_label: str = "x",
    time_label: str = "t",
) -> TrajectoryStateTimeSamples:
    """Adapt every valid trajectory node to a weighted state-time particle batch."""
    if not isinstance(trajectory, StochasticTrajectory):
        raise TypeError("trajectory must be a StochasticTrajectory.")
    if mode not in ("global", "per_time"):
        raise ValueError("mode must be 'global' or 'per_time'.")
    if not state_label or not time_label or state_label == time_label:
        raise ValueError("state_label and time_label must be distinct non-empty strings.")
    states, valid, leading_axes, realization_axes = _trajectory_arrays(trajectory)
    if trajectory.realization_axes:
        times_array = trajectory.times
    else:
        times_array = jnp.expand_dims(trajectory.times, axis=len(trajectory.case_shape))
    weight_dims = leading_axes + (trajectory.time_axis,)
    times = cx.Field(times_array, dims=weight_dims)
    if log_weights is None:
        weights = cx.Field(jnp.zeros(times.shape, dtype=float), dims=weight_dims)
    elif isinstance(log_weights, cx.Field):
        if log_weights.dims != weight_dims or log_weights.shape != times.shape:
            raise ValueError("log_weights field must match trajectory state-time axes.")
        weights = log_weights
    else:
        array = jnp.asarray(log_weights, dtype=float)
        if array.shape != times.shape:
            raise ValueError("log_weights must match the trajectory leading/time shape.")
        weights = cx.Field(array, dims=weight_dims)
    weight_values = jnp.asarray(weights.data, dtype=float)
    if bool(jnp.any(~jnp.isfinite(weight_values))):
        raise ValueError("log_weights must be finite.")

    leading_shape = tuple(int(size) for size in times.shape[:-1])
    path_count = 1
    for size in leading_shape:
        path_count *= size
    path_base = jnp.arange(path_count, dtype=jnp.int32).reshape(leading_shape + (1,))
    path_values = jnp.broadcast_to(path_base, times.shape)
    path_indices = cx.Field(path_values, dims=weight_dims)

    independence_labels = trajectory.independence_ids
    independence_base = jnp.asarray(
        _independence_indices(independence_labels),
        dtype=jnp.int32,
    ).reshape(leading_shape + (1,))
    independence_values = jnp.broadcast_to(independence_base, times.shape)
    independence_indices = cx.Field(independence_values, dims=weight_dims)
    time_values = jnp.broadcast_to(
        jnp.arange(trajectory.num_times, dtype=jnp.int32),
        times.shape,
    )
    time_indices = cx.Field(time_values, dims=weight_dims)

    return TrajectoryStateTimeSamples(
        states=states,
        times=times,
        valid=valid,
        log_weights=weights,
        path_indices=path_indices,
        independence_indices=independence_indices,
        time_indices=time_indices,
        leading_axes=leading_axes,
        realization_axes=realization_axes,
        time_axis=trajectory.time_axis,
        state_axes=trajectory.state_axes,
        state_label=str(state_label),
        time_label=str(time_label),
        mode=mode,
        path_labels=trajectory.trajectory_ids,
        independence_labels=independence_labels,
        provenance=f"stochastic-trajectory:state-time:{mode}",
    )


def trajectory_state_time_measure(
    trajectory: StochasticTrajectory,
    /,
    **kwargs,
) -> WeightedSampleTarget:
    """Return the generic weighted-target view of state-time trajectory nodes."""
    return trajectory_state_time_samples(trajectory, **kwargs).target()


__all__ = [
    "trajectory_state_time_measure",
    "trajectory_state_time_samples",
    "TrajectoryStateTimeMode",
    "TrajectoryStateTimeSamples",
]
