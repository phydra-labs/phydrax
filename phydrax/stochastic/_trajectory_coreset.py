#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lazy fixed-length stochastic trajectory blocks and native coreset selection."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Integral
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..coresets import (
    CoresetSelection,
    kernel_herd,
    KernelHerding,
    moment_recombine,
    MomentRecombination,
)
from ._trajectory import StochasticDriverSegmentReference, StochasticTrajectory


TrajectoryBlockWeighting = Literal["trajectory", "block", "duration"]
TrajectoryCoresetMethod = MomentRecombination | KernelHerding


class StochasticTrajectoryBlockView(StrictModule):
    """Lazy path/start index view over fixed-length saved-state blocks."""

    trajectory: StochasticTrajectory
    path_indices: Array
    start_indices: Array
    valid: Array
    block_length: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    blocks_per_path: int = eqx.field(static=True)

    def __init__(
        self,
        trajectory: StochasticTrajectory,
        /,
        *,
        block_length: int,
        stride: int = 1,
    ):
        if not isinstance(trajectory, StochasticTrajectory):
            raise TypeError("trajectory must be a StochasticTrajectory.")
        length = _positive_integer(block_length, name="block_length")
        step = _positive_integer(stride, name="stride")
        if length > trajectory.num_times:
            raise ValueError("block_length cannot exceed saved trajectory times.")
        starts = jnp.arange(0, trajectory.num_times - length + 1, step, dtype=jnp.int32)
        if starts.size == 0:
            raise ValueError("No trajectory block fits the requested length and stride.")
        path_count = prod(trajectory.leading_shape) if trajectory.leading_shape else 1
        path_indices = jnp.repeat(jnp.arange(path_count, dtype=jnp.int32), starts.size)
        start_indices = jnp.tile(starts, path_count)
        flat_valid = trajectory.valid.reshape((path_count, trajectory.num_times))
        offsets = jnp.arange(length, dtype=jnp.int32)
        block_valid = jnp.all(
            flat_valid[path_indices[:, None], start_indices[:, None] + offsets], axis=1
        )
        self.trajectory = trajectory
        self.path_indices = path_indices
        self.start_indices = start_indices
        self.valid = block_valid
        self.block_length = length
        self.stride = step
        self.blocks_per_path = int(starts.size)

    @property
    def count(self) -> int:
        return int(self.path_indices.shape[0])

    @property
    def states(self) -> Array:
        path_count = (
            prod(self.trajectory.leading_shape) if self.trajectory.leading_shape else 1
        )
        flat = self.trajectory.states.reshape(
            (path_count, self.trajectory.num_times) + self.trajectory.state_shape
        )
        offsets = jnp.arange(self.block_length, dtype=jnp.int32)
        return flat[
            self.path_indices[:, None],
            self.start_indices[:, None] + offsets,
        ]

    @property
    def times(self) -> Array:
        path_count = (
            prod(self.trajectory.leading_shape) if self.trajectory.leading_shape else 1
        )
        flat = self.trajectory.times.reshape((path_count, self.trajectory.num_times))
        offsets = jnp.arange(self.block_length, dtype=jnp.int32)
        return flat[
            self.path_indices[:, None],
            self.start_indices[:, None] + offsets,
        ]

    @property
    def durations(self) -> Array:
        values = self.times
        return values[:, -1] - values[:, 0]


@dataclass(frozen=True)
class TrajectoryBlockCoreset:
    """Selected lazy blocks with positive weights and driver/coupling provenance."""

    view: StochasticTrajectoryBlockView
    selection: CoresetSelection
    path_indices: Array
    start_indices: Array
    states: Array
    times: Array
    weights: Array
    mask: Array
    references: tuple[StochasticDriverSegmentReference, ...]
    weighting: TrajectoryBlockWeighting
    objective: str

    def to_operator_dataset(self):
        """Lower selected blocks to canonical weighted operator cases."""
        return trajectory_block_coreset_to_operator_dataset(self)


def trajectory_blocks(
    trajectory: StochasticTrajectory,
    /,
    *,
    block_length: int,
    stride: int = 1,
) -> StochasticTrajectoryBlockView:
    """Construct a lazy candidate view without crossing path boundaries."""
    return StochasticTrajectoryBlockView(
        trajectory,
        block_length=block_length,
        stride=stride,
    )


def compress_trajectory_blocks(
    view: StochasticTrajectoryBlockView,
    method: TrajectoryCoresetMethod,
    /,
    *,
    features: ArrayLike,
    weighting: TrajectoryBlockWeighting = "trajectory",
) -> TrajectoryBlockCoreset:
    """Select valid blocks under one explicit trajectory/block/duration measure."""
    if not isinstance(view, StochasticTrajectoryBlockView):
        raise TypeError("view must be a StochasticTrajectoryBlockView.")
    if weighting not in ("trajectory", "block", "duration"):
        raise ValueError("weighting must be 'trajectory', 'block', or 'duration'.")
    feature_values = jnp.asarray(features)
    if feature_values.ndim != 2 or feature_values.shape[0] != view.count:
        raise ValueError("features must have shape (candidate_block, feature).")
    weights = _source_weights(view, weighting)
    log_weights = jnp.where(weights > 0.0, jnp.log(weights), -jnp.inf)
    if isinstance(method, MomentRecombination):
        selection = moment_recombine(
            feature_values,
            method,
            log_weights=log_weights,
            mask=view.valid & (weights > 0.0),
        )
    elif isinstance(method, KernelHerding):
        selection = kernel_herd(
            feature_values,
            method,
            log_weights=log_weights,
            mask=view.valid & (weights > 0.0),
        )
    else:
        raise TypeError("method must be MomentRecombination or KernelHerding.")
    indices = selection.indices
    path_indices = view.path_indices[indices]
    start_indices = view.start_indices[indices]
    references = _references(view, path_indices, start_indices)
    return TrajectoryBlockCoreset(
        view=view,
        selection=selection,
        path_indices=path_indices,
        start_indices=start_indices,
        states=view.states[indices],
        times=view.times[indices],
        weights=selection.weights,
        mask=selection.mask,
        references=references,
        weighting=weighting,
        objective=(
            "supplied-block-feature-moments"
            if isinstance(method, MomentRecombination)
            else "declared-block-kernel-MMD"
        ),
    )


def _source_weights(
    view: StochasticTrajectoryBlockView,
    weighting: TrajectoryBlockWeighting,
    /,
) -> Array:
    valid = view.valid
    if weighting == "block":
        raw = valid.astype(float)
    else:
        path_count = int(jnp.max(view.path_indices)) + 1
        if weighting == "trajectory":
            within = valid.astype(float)
        else:
            duration = view.durations
            within = jnp.where(valid & (duration > 0.0), duration, 0.0)
        path_mass = (
            jnp.zeros((path_count,), dtype=within.dtype).at[view.path_indices].add(within)
        )
        active_paths = path_mass > 0.0
        path_count_active = jnp.sum(active_paths)
        raw = jnp.where(
            valid & active_paths[view.path_indices],
            within
            / jnp.where(
                path_mass[view.path_indices] > 0.0, path_mass[view.path_indices], 1.0
            )
            / jnp.where(path_count_active > 0, path_count_active, 1),
            0.0,
        )
    total = jnp.sum(raw)
    return eqx.error_if(
        jnp.where(total > 0.0, raw / total, raw),
        ~jnp.isfinite(total) | (total <= 0.0),
        "Trajectory block weighting has no valid positive mass.",
    )


def _references(
    view: StochasticTrajectoryBlockView,
    path_indices: Array,
    start_indices: Array,
    /,
) -> tuple[StochasticDriverSegmentReference, ...]:
    trajectory = view.trajectory
    times = np.asarray(jax.device_get(trajectory.times)).reshape(
        (-1, trajectory.num_times)
    )
    paths = np.asarray(jax.device_get(path_indices))
    starts = np.asarray(jax.device_get(start_indices))
    realization_count = trajectory.num_realizations
    references = []
    for path, start in zip(paths, starts, strict=True):
        case_index = int(path) // realization_count
        realization = trajectory.realizations[case_index]
        target = int(start) + view.block_length - 1
        references.append(
            StochasticDriverSegmentReference(
                trajectory.trajectory_ids[int(path)],
                trajectory.case_ids[case_index],
                trajectory.parameter_ids[case_index],
                None if realization is None else realization.realization_id,
                None if realization is None else realization.coupling_id,
                int(start),
                target,
                float(times[int(path), int(start)]),
                float(times[int(path), target]),
            )
        )
    return tuple(references)


def trajectory_block_coreset_to_operator_dataset(
    coreset: TrajectoryBlockCoreset,
    /,
):
    """Lower block endpoints to one weighted canonical OperatorDataset."""
    if not isinstance(coreset, TrajectoryBlockCoreset):
        raise TypeError("coreset must be a TrajectoryBlockCoreset.")
    from ..nn.operator import (
        FunctionSamples,
        OperatorBatch,
        OperatorCaseProvenance,
        OperatorTargetBatch,
    )
    from ..nn.operator.training import OperatorDataset

    count = coreset.selection.capacity
    source_coordinates = coreset.times[:, :1, None]
    target_coordinates = coreset.times[:, -1:, None]
    source_values = coreset.states[:, :1]
    target_values = coreset.states[:, -1:]
    batch = OperatorBatch(
        inputs={
            "state": FunctionSamples(
                values=source_values,
                coordinates=source_coordinates,
                support_id="trajectory-block-source",
            )
        },
        queries={
            "target": FunctionSamples(
                values=None,
                coordinates=target_coordinates,
                support_id="trajectory-block-target",
            )
        },
        case_axes=("block",),
        case_shape=(count,),
    )
    targets = OperatorTargetBatch.from_arrays(
        {"state": target_values},
        batch,
        query_names={"state": "target"},
    )
    provenance = tuple(
        OperatorCaseProvenance(
            f"{reference.trajectory_id}:block:{reference.source_index}:"
            f"{reference.target_index}:{position}",
            identities={
                "physical_case": reference.physical_case_id,
                "trajectory": reference.trajectory_id,
                **(
                    {}
                    if reference.parameter_id is None
                    else {"parameters": reference.parameter_id}
                ),
                **(
                    {}
                    if reference.realization_id is None
                    else {"realization": reference.realization_id}
                ),
                **(
                    {}
                    if reference.coupling_id is None
                    else {"coupling": reference.coupling_id}
                ),
            },
            order={
                "source_time": reference.source_time,
                "target_time": reference.target_time,
            },
        )
        for position, reference in enumerate(coreset.references)
    )
    return OperatorDataset(
        batch,
        targets,
        provenance,
        case_log_weights=coreset.selection.log_weights,
        case_mask=coreset.mask,
    )


def _positive_integer(value: int, /, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


__all__ = [
    "trajectory_block_coreset_to_operator_dataset",
    "StochasticTrajectoryBlockView",
    "TrajectoryBlockCoreset",
    "TrajectoryBlockWeighting",
    "compress_trajectory_blocks",
    "trajectory_blocks",
]
