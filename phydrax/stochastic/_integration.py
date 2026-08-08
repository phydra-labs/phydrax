#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import coordax as cx
import jax.numpy as jnp

from ..integration._targets import DiscreteMeasureTarget, WeightedSampleTarget
from ._trajectory import StochasticTrajectory


TrajectoryMeasureMode = Literal["marginal", "path"]
TrajectoryTimeRule = Literal["left", "trapezoid"]


def _trajectory_arrays(
    trajectory: StochasticTrajectory, /
) -> tuple[
    cx.Field,
    cx.Field,
    tuple[str, ...],
    tuple[str, ...],
]:
    case_dims = trajectory.case_axes
    if trajectory.realization_axes:
        realization_dims = trajectory.realization_axes
        states = trajectory.states
        valid = trajectory.valid
    else:
        existing = set(case_dims + (trajectory.time_axis,) + trajectory.state_axes)
        sample_dim = "trajectory_sample"
        while sample_dim in existing:
            sample_dim = f"_{sample_dim}"
        realization_dims = (sample_dim,)
        position = len(trajectory.case_shape)
        states = jnp.expand_dims(trajectory.states, axis=position)
        valid = jnp.expand_dims(trajectory.valid, axis=position)
    leading_dims = case_dims + realization_dims
    state_field = cx.Field(
        states,
        dims=leading_dims + (trajectory.time_axis,) + trajectory.state_axes,
    )
    valid_field = cx.Field(
        valid,
        dims=leading_dims + (trajectory.time_axis,),
    )
    return state_field, valid_field, leading_dims, realization_dims


def _independent_paths(trajectory: StochasticTrajectory, /) -> bool:
    if not trajectory.realization_axes:
        return False
    labels = trajectory.independence_ids
    count = trajectory.num_realizations
    for case_index in range(trajectory.num_cases):
        start = case_index * count
        case_labels = labels[start : start + count]
        if any(label is None for label in case_labels):
            return False
        if len(set(case_labels)) != len(case_labels):
            return False
    return True


def trajectory_measure(
    trajectory: StochasticTrajectory,
    /,
    *,
    mode: TrajectoryMeasureMode = "marginal",
) -> WeightedSampleTarget:
    """Expose trajectory realizations as a masked empirical measure.

    ``mode="marginal"`` retains saved time and excludes invalid states independently.
    ``mode="path"`` treats each complete trajectory as one sample and excludes every
    path containing an invalid saved state.
    """
    if not isinstance(trajectory, StochasticTrajectory):
        raise TypeError("trajectory must be a StochasticTrajectory.")
    if mode not in ("marginal", "path"):
        raise ValueError("mode must be 'marginal' or 'path'.")
    states, marginal_valid, leading_dims, sample_dims = _trajectory_arrays(trajectory)
    if mode == "marginal":
        mask = marginal_valid
        weight_dims = leading_dims + (trajectory.time_axis,)
        provenance = "stochastic-trajectory:marginal"
    else:
        path_valid = jnp.all(jnp.asarray(marginal_valid.data, dtype=bool), axis=-1)
        mask = cx.Field(path_valid, dims=leading_dims)
        weight_dims = leading_dims
        provenance = "stochastic-trajectory:path"
    log_weights = cx.Field(jnp.zeros(mask.shape), dims=weight_dims)
    return WeightedSampleTarget(
        states,
        log_weights,
        normalized=True,
        independent=_independent_paths(trajectory),
        mask=mask,
        sample_axes=sample_dims,
        provenance=provenance,
    )


def time_measure(
    trajectory: StochasticTrajectory,
    /,
    *,
    rule: TrajectoryTimeRule = "trapezoid",
    normalized: bool = False,
) -> DiscreteMeasureTarget:
    """Expose each trajectory's irregular saved-time grid as a fixed measure."""
    if not isinstance(trajectory, StochasticTrajectory):
        raise TypeError("trajectory must be a StochasticTrajectory.")
    if rule not in ("left", "trapezoid"):
        raise ValueError("rule must be 'left' or 'trapezoid'.")
    valid = jnp.asarray(trajectory.marginal_valid, dtype=bool)
    invalid_prefix = jnp.cumsum(~valid, axis=-1) > 0
    if bool(jnp.any(valid & invalid_prefix)):
        raise ValueError("Trajectory validity masks must be contiguous prefixes.")
    times = jnp.asarray(trajectory.times, dtype=float)
    weights = jnp.zeros_like(times)
    if trajectory.num_times > 1:
        intervals = jnp.diff(times, axis=-1)
        active_intervals = valid[..., :-1] & valid[..., 1:]
        interval_weights = jnp.where(active_intervals, intervals, 0.0)
        if rule == "left":
            weights = weights.at[..., :-1].set(interval_weights)
        else:
            half = 0.5 * interval_weights
            weights = weights.at[..., :-1].add(half)
            weights = weights.at[..., 1:].add(half)
    dims = trajectory.case_axes + trajectory.realization_axes + (trajectory.time_axis,)
    points = cx.Field(times, dims=dims)
    weight_field = cx.Field(weights, dims=dims)
    mask = cx.Field(weights > 0.0, dims=dims)
    return DiscreteMeasureTarget(
        points,
        weight_field,
        axes=trajectory.time_axis,
        mask=mask,
        normalized=normalized,
        provenance=f"stochastic-trajectory-time:{rule}",
    )


__all__ = [
    "time_measure",
    "trajectory_measure",
    "TrajectoryMeasureMode",
    "TrajectoryTimeRule",
]
