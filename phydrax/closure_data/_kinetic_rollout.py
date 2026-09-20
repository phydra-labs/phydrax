#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._dataset import (
    ClosureSample,
    ClosureSampleKey,
    DatasetSplit,
    LeakageSafePartition,
    LeakageSafePartitionPlan,
)


_D2V17_POPULATION_COUNT = 17
_CONSERVED_COMPONENT_NAMES = (
    "density",
    "momentum_x",
    "momentum_y",
    "total_energy",
)


def _identifier(value: str, role: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{role} must be non-empty.")
    return identifier


def _optional_identifier(value: str | None, role: str, /) -> str | None:
    if value is None:
        return None
    return _identifier(value, role)


def _positive_integer(value: int, role: str, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{role} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{role} must be positive.")
    return result


def _spatial_shape(value: tuple[int, ...], /) -> tuple[int, ...]:
    shape = tuple(value)
    if not shape:
        raise ValueError("spatial_shape must contain at least one dimension.")
    return tuple(_positive_integer(size, "spatial_shape dimension") for size in shape)


class SmoothCompressibleRolloutSchema(StrictModule, NonTrainableState):
    """Static native D2V17 f/g/U rollout representation and provenance."""

    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    f_shape: tuple[int, ...] = eqx.field(static=True)
    g_shape: tuple[int, ...] = eqx.field(static=True)
    U_shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    population_count: int = eqx.field(static=True)
    conserved_component_names: tuple[str, ...] = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    stage_one_artifact_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    oracle_id: str = eqx.field(static=True)
    boundary_schedule_id: str | None = eqx.field(static=True)
    force_schedule_id: str | None = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_shape: tuple[int, ...],
        dtype: Any,
        /,
        *,
        runtime_id: str,
        topology_id: str,
        method_id: str,
        quadrature_id: str,
        material_id: str,
        stage_one_artifact_id: str,
        support_id: str,
        oracle_id: str,
        dt: float,
        boundary_schedule_id: str | None = None,
        force_schedule_id: str | None = None,
    ):
        shape = _spatial_shape(spatial_shape)
        dtype_ = jnp.dtype(dtype)
        if not jnp.issubdtype(dtype_, jnp.floating):
            raise TypeError("Smooth-compressible rollout dtype must be real floating.")
        timestep = float(dt)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError(
                "Smooth-compressible rollout dt must be finite and positive."
            )
        identifiers = tuple(
            _identifier(value, role)
            for value, role in (
                (runtime_id, "runtime_id"),
                (topology_id, "topology_id"),
                (method_id, "method_id"),
                (quadrature_id, "quadrature_id"),
                (material_id, "material_id"),
                (stage_one_artifact_id, "stage_one_artifact_id"),
                (support_id, "support_id"),
                (oracle_id, "oracle_id"),
            )
        )
        boundary = _optional_identifier(boundary_schedule_id, "boundary_schedule_id")
        force = _optional_identifier(force_schedule_id, "force_schedule_id")
        self.spatial_shape = shape
        self.f_shape = (*shape, _D2V17_POPULATION_COUNT)
        self.g_shape = (*shape, _D2V17_POPULATION_COUNT)
        self.U_shape = (*shape, len(_CONSERVED_COMPONENT_NAMES))
        self.dtype = dtype_.name
        self.population_count = _D2V17_POPULATION_COUNT
        self.conserved_component_names = _CONSERVED_COMPONENT_NAMES
        (
            self.runtime_id,
            self.topology_id,
            self.method_id,
            self.quadrature_id,
            self.material_id,
            self.stage_one_artifact_id,
            self.support_id,
            self.oracle_id,
        ) = identifiers
        self.boundary_schedule_id = boundary
        self.force_schedule_id = force
        self.dt = timestep
        self.schema_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v17-rollout-schema",
                "spatial_shape": list(shape),
                "f_shape": list(self.f_shape),
                "g_shape": list(self.g_shape),
                "U_shape": list(self.U_shape),
                "dtype": self.dtype,
                "population_count": _D2V17_POPULATION_COUNT,
                "conserved_components": list(_CONSERVED_COMPONENT_NAMES),
                "runtime": self.runtime_id,
                "topology": self.topology_id,
                "method": self.method_id,
                "quadrature": self.quadrature_id,
                "material": self.material_id,
                "stage_one_artifact": self.stage_one_artifact_id,
                "support": self.support_id,
                "oracle": self.oracle_id,
                "boundary_schedule": boundary,
                "force_schedule": force,
                "dt": timestep,
            }
        )


class SmoothCompressibleRolloutTrajectory(StrictModule, NonTrainableState):
    """One complete parent trajectory in native f/g/U ordering."""

    f: Array
    g: Array
    U: Array
    valid: Array
    schema: SmoothCompressibleRolloutSchema
    case_id: str = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    time_block_id: str = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    parent_id: str = eqx.field(static=True)

    def __init__(
        self,
        f: ArrayLike,
        g: ArrayLike,
        U: ArrayLike,
        valid: ArrayLike,
        schema: SmoothCompressibleRolloutSchema,
        /,
        *,
        case_id: str,
        trajectory_id: str,
        realization_id: str,
        time_block_id: str,
    ):
        if not isinstance(schema, SmoothCompressibleRolloutSchema):
            raise TypeError("schema must be a SmoothCompressibleRolloutSchema.")
        f_ = jnp.asarray(f)
        g_ = jnp.asarray(g)
        U_ = jnp.asarray(U)
        valid_ = jnp.asarray(valid)
        if f_.ndim < 2:
            raise ValueError("Rollout populations require a leading time dimension.")
        count = f_.shape[0]
        if count <= 0:
            raise ValueError("A rollout trajectory must contain at least one state.")
        if f_.shape != (count, *schema.f_shape):
            raise ValueError("f history does not match the exact schema shape.")
        if g_.shape != (count, *schema.g_shape):
            raise ValueError("g history does not match the exact schema shape.")
        if U_.shape != (count, *schema.U_shape):
            raise ValueError(
                "U history does not match the exact schema shape and ordering."
            )
        if valid_.dtype != jnp.dtype(jnp.bool_) or valid_.shape != (count,):
            raise ValueError("Trajectory validity must be one boolean per time state.")
        expected_dtype = jnp.dtype(schema.dtype)
        if any(value.dtype != expected_dtype for value in (f_, g_, U_)):
            raise TypeError(
                "Trajectory f/g/U dtypes must exactly match the schema dtype."
            )
        if any(np.any(~np.isfinite(np.asarray(value))) for value in (f_, g_, U_)):
            raise ValueError("Trajectory f/g/U histories must be finite.")
        identifiers = tuple(
            _identifier(value, role)
            for value, role in (
                (case_id, "case_id"),
                (trajectory_id, "trajectory_id"),
                (realization_id, "realization_id"),
                (time_block_id, "time_block_id"),
            )
        )
        self.f = f_
        self.g = g_
        self.U = U_
        self.valid = valid_
        self.schema = schema
        self.case_id, self.trajectory_id, self.realization_id, self.time_block_id = (
            identifiers
        )
        self.sample_count = count
        self.parent_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v17-rollout-trajectory",
                "schema": schema.schema_id,
                "identity": list(identifiers),
                "sample_count": count,
                "arrays": array_tree_fingerprint((f_, g_, U_, valid_)),
            }
        )

    def sample_key(self, /) -> ClosureSampleKey:
        """Return the existing leakage infrastructure's complete-parent key."""

        return ClosureSampleKey(
            case_id=self.case_id,
            trajectory_id=self.trajectory_id,
            realization_id=self.realization_id,
            time_block_id=self.time_block_id,
            time_index=0,
        )


class SmoothCompressibleRolloutWindowPlan(StrictModule, NonTrainableState):
    """Deterministic contiguous history/rollout window selection."""

    history_steps: int = eqx.field(static=True)
    horizon_steps: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    maximum_windows_per_trajectory: int | None = eqx.field(static=True)
    selection_salt: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        history_steps: int,
        horizon_steps: int,
        /,
        *,
        stride: int = 1,
        maximum_windows_per_trajectory: int | None = None,
        selection_salt: str = "smooth-compressible-rollout",
    ):
        history = _positive_integer(history_steps, "history_steps")
        horizon = _positive_integer(horizon_steps, "horizon_steps")
        stride_ = _positive_integer(stride, "stride")
        maximum = (
            None
            if maximum_windows_per_trajectory is None
            else _positive_integer(
                maximum_windows_per_trajectory, "maximum_windows_per_trajectory"
            )
        )
        salt = _identifier(selection_salt, "selection_salt")
        self.history_steps = history
        self.horizon_steps = horizon
        self.stride = stride_
        self.maximum_windows_per_trajectory = maximum
        self.selection_salt = salt
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-rollout-window-plan",
                "history_steps": history,
                "horizon_steps": horizon,
                "stride": stride_,
                "maximum_windows_per_trajectory": maximum,
                "selection_salt": salt,
            }
        )

    def anchors(
        self, trajectory: SmoothCompressibleRolloutTrajectory, /
    ) -> tuple[int, ...]:
        """Select valid anchors reproducibly, returning them in time order."""

        if not isinstance(trajectory, SmoothCompressibleRolloutTrajectory):
            raise TypeError("trajectory must be a SmoothCompressibleRolloutTrajectory.")
        last = trajectory.sample_count - self.horizon_steps
        candidates = tuple(range(self.history_steps, last + 1, self.stride))
        if not candidates:
            raise ValueError(
                "Trajectory is too short for the requested history and horizon."
            )
        if (
            self.maximum_windows_per_trajectory is None
            or len(candidates) <= self.maximum_windows_per_trajectory
        ):
            return candidates
        ranked = sorted(
            candidates,
            key=lambda anchor: canonical_fingerprint(
                {
                    "kind": "smooth-compressible-rollout-window-selection",
                    "plan": self.plan_id,
                    "parent": trajectory.parent_id,
                    "anchor": anchor,
                }
            ),
        )
        return tuple(sorted(ranked[: self.maximum_windows_per_trajectory]))

    def validate_anchor(
        self, trajectory: SmoothCompressibleRolloutTrajectory, anchor: int, /
    ) -> int:
        if isinstance(anchor, (bool, np.bool_)) or not isinstance(
            anchor, (int, np.integer)
        ):
            raise TypeError("Window anchor must be an integer.")
        anchor_ = int(anchor)
        if (
            anchor_ < self.history_steps
            or anchor_ + self.horizon_steps > trajectory.sample_count
        ):
            raise ValueError(
                "Window anchor does not admit the requested history and horizon."
            )
        if (anchor_ - self.history_steps) % self.stride != 0:
            raise ValueError("Window anchor is not on the plan stride.")
        return anchor_


class SmoothCompressibleRolloutWindow(StrictModule, NonTrainableState):
    """One contiguous coupled f/g/U history and oracle rollout target."""

    f_history: Array
    g_history: Array
    U_history: Array
    f_targets: Array
    g_targets: Array
    U_targets: Array
    parent_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    split: DatasetSplit = eqx.field(static=True)
    anchor: int = eqx.field(static=True)
    history_indices: tuple[int, ...] = eqx.field(static=True)
    target_indices: tuple[int, ...] = eqx.field(static=True)
    time_indices: tuple[int, ...] = eqx.field(static=True)
    window_plan_id: str = eqx.field(static=True)
    window_id: str = eqx.field(static=True)

    def __init__(
        self,
        trajectory: SmoothCompressibleRolloutTrajectory,
        plan: SmoothCompressibleRolloutWindowPlan,
        anchor: int,
        /,
        *,
        split: DatasetSplit,
    ):
        if not isinstance(trajectory, SmoothCompressibleRolloutTrajectory):
            raise TypeError("trajectory must be a SmoothCompressibleRolloutTrajectory.")
        if not isinstance(plan, SmoothCompressibleRolloutWindowPlan):
            raise TypeError("plan must be a SmoothCompressibleRolloutWindowPlan.")
        split_ = str(split).strip()
        if split_ not in ("train", "validation", "test"):
            raise ValueError("Unknown rollout dataset split.")
        anchor_ = plan.validate_anchor(trajectory, anchor)
        history_start = anchor_ - plan.history_steps
        target_stop = anchor_ + plan.horizon_steps
        history_indices = tuple(range(history_start, anchor_))
        target_indices = tuple(range(anchor_, target_stop))
        if not bool(np.all(np.asarray(trajectory.valid[history_start:target_stop]))):
            raise ValueError("A rollout window cannot include an invalid oracle state.")
        self.f_history = trajectory.f[history_start:anchor_]
        self.g_history = trajectory.g[history_start:anchor_]
        self.U_history = trajectory.U[history_start:anchor_]
        self.f_targets = trajectory.f[anchor_:target_stop]
        self.g_targets = trajectory.g[anchor_:target_stop]
        self.U_targets = trajectory.U[anchor_:target_stop]
        self.parent_id = trajectory.parent_id
        self.schema_id = trajectory.schema.schema_id
        self.split = split_
        self.anchor = anchor_
        self.history_indices = history_indices
        self.target_indices = target_indices
        self.time_indices = (*history_indices, *target_indices)
        self.window_plan_id = plan.plan_id
        self.window_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v17-rollout-window",
                "parent": trajectory.parent_id,
                "schema": self.schema_id,
                "window_plan": plan.plan_id,
                "split": split_,
                "anchor": anchor_,
                "history_indices": list(history_indices),
                "target_indices": list(target_indices),
            }
        )


class SmoothCompressibleRolloutStatistics(StrictModule, NonTrainableState):
    """Train-parent statistics in native f/g/U channel order, counted once."""

    f_mean: Array
    f_scale: Array
    g_mean: Array
    g_scale: Array
    U_mean: Array
    U_scale: Array
    training_parent_ids: tuple[str, ...] = eqx.field(static=True)
    training_sample_ids: tuple[str, ...] = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    statistics_id: str = eqx.field(static=True)

    def __init__(
        self,
        f_mean: ArrayLike,
        f_scale: ArrayLike,
        g_mean: ArrayLike,
        g_scale: ArrayLike,
        U_mean: ArrayLike,
        U_scale: ArrayLike,
        /,
        *,
        training_parent_ids: tuple[str, ...],
        training_sample_ids: tuple[str, ...],
        partition_id: str,
        schema_id: str,
        epsilon: float,
    ):
        arrays = tuple(
            jnp.asarray(value)
            for value in (f_mean, f_scale, g_mean, g_scale, U_mean, U_scale)
        )
        expected_shapes = ((17,), (17,), (17,), (17,), (4,), (4,))
        if tuple(value.shape for value in arrays) != expected_shapes:
            raise ValueError("Rollout statistics do not match native f/g/U channels.")
        if any(np.any(~np.isfinite(np.asarray(value))) for value in arrays):
            raise ValueError("Rollout statistics must be finite.")
        epsilon_ = float(epsilon)
        if not np.isfinite(epsilon_) or epsilon_ <= 0.0:
            raise ValueError("Statistics epsilon must be finite and positive.")
        if any(np.any(np.asarray(value) < epsilon_) for value in arrays[1::2]):
            raise ValueError("Rollout statistic scales must be at least epsilon.")
        parents = tuple(
            _identifier(value, "training parent id") for value in training_parent_ids
        )
        samples = tuple(
            _identifier(value, "training sample id") for value in training_sample_ids
        )
        if not parents or len(parents) != len(samples):
            raise ValueError(
                "Statistics require aligned training parent and sample identities."
            )
        if len(set(parents)) != len(parents) or len(set(samples)) != len(samples):
            raise ValueError("Statistics training identities must be unique.")
        partition = _identifier(partition_id, "partition_id")
        schema = _identifier(schema_id, "schema_id")
        (
            self.f_mean,
            self.f_scale,
            self.g_mean,
            self.g_scale,
            self.U_mean,
            self.U_scale,
        ) = arrays
        self.training_parent_ids = parents
        self.training_sample_ids = samples
        self.partition_id = partition
        self.schema_id = schema
        self.epsilon = epsilon_
        self.statistics_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v17-rollout-statistics",
                "training_parents": list(parents),
                "training_samples": list(samples),
                "partition": partition,
                "schema": schema,
                "epsilon": epsilon_,
                "arrays": array_tree_fingerprint(arrays),
            }
        )


def _channel_moments(
    arrays: tuple[Array, ...],
    channels: int,
    epsilon: float,
    /,
) -> tuple[Array, Array]:
    count = 0
    mean: Array | None = None
    second: Array | None = None
    for array in arrays:
        values = array.reshape((-1, channels))
        local_count = values.shape[0]
        local_mean = jnp.mean(values, axis=0)
        local_second = jnp.sum(jnp.square(values - local_mean), axis=0)
        if mean is None:
            count = local_count
            mean = local_mean
            second = local_second
            continue
        assert second is not None
        total = count + local_count
        delta = local_mean - mean
        second = (
            second
            + local_second
            + jnp.square(delta) * (float(count * local_count) / float(total))
        )
        mean = mean + delta * (float(local_count) / float(total))
        count = total
    assert mean is not None and second is not None and count > 0
    scale = jnp.maximum(jnp.sqrt(second / float(count)), float(epsilon))
    return mean, scale


def _fit_statistics(
    trajectories: tuple[SmoothCompressibleRolloutTrajectory, ...],
    samples: tuple[ClosureSample, ...],
    partition: LeakageSafePartition,
    epsilon: float,
    /,
) -> SmoothCompressibleRolloutStatistics:
    training_pairs = tuple(
        (trajectory, sample)
        for trajectory, sample in zip(trajectories, samples, strict=True)
        if partition.assignment_for(sample.sample_id).split == "train"
    )
    if not training_pairs:
        raise ValueError("Rollout preparation requires a nonempty training parent split.")
    training = tuple(value[0] for value in training_pairs)
    f_mean, f_scale = _channel_moments(tuple(value.f for value in training), 17, epsilon)
    g_mean, g_scale = _channel_moments(tuple(value.g for value in training), 17, epsilon)
    U_mean, U_scale = _channel_moments(tuple(value.U for value in training), 4, epsilon)
    return SmoothCompressibleRolloutStatistics(
        f_mean,
        f_scale,
        g_mean,
        g_scale,
        U_mean,
        U_scale,
        training_parent_ids=tuple(value[0].parent_id for value in training_pairs),
        training_sample_ids=tuple(value[1].sample_id for value in training_pairs),
        partition_id=partition.partition_id,
        schema_id=training[0].schema.schema_id,
        epsilon=epsilon,
    )


class PreparedSmoothCompressibleRolloutDataset(StrictModule, NonTrainableState):
    """Complete-parent partition, train-only statistics, then derived windows."""

    trajectories: tuple[SmoothCompressibleRolloutTrajectory, ...]
    parent_samples: tuple[ClosureSample, ...]
    windows: tuple[SmoothCompressibleRolloutWindow, ...]
    train_windows: tuple[SmoothCompressibleRolloutWindow, ...]
    validation_windows: tuple[SmoothCompressibleRolloutWindow, ...]
    test_windows: tuple[SmoothCompressibleRolloutWindow, ...]
    partition: LeakageSafePartition
    window_plan: SmoothCompressibleRolloutWindowPlan
    statistics: SmoothCompressibleRolloutStatistics
    schema_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    def __init__(
        self,
        trajectories: tuple[SmoothCompressibleRolloutTrajectory, ...],
        parent_samples: tuple[ClosureSample, ...],
        windows: tuple[SmoothCompressibleRolloutWindow, ...],
        partition: LeakageSafePartition,
        window_plan: SmoothCompressibleRolloutWindowPlan,
        statistics: SmoothCompressibleRolloutStatistics,
        /,
    ):
        parents = tuple(trajectories)
        samples = tuple(parent_samples)
        windows_ = tuple(windows)
        if not parents or any(
            not isinstance(value, SmoothCompressibleRolloutTrajectory)
            for value in parents
        ):
            raise ValueError("Prepared rollout datasets require parent trajectories.")
        if len(samples) != len(parents) or any(
            not isinstance(value, ClosureSample) for value in samples
        ):
            raise ValueError("Prepared rollout parent samples are incomplete.")
        if not windows_ or any(
            not isinstance(value, SmoothCompressibleRolloutWindow) for value in windows_
        ):
            raise ValueError("Prepared rollout datasets require derived windows.")
        if not isinstance(partition, LeakageSafePartition):
            raise TypeError("partition must be a LeakageSafePartition.")
        if not isinstance(window_plan, SmoothCompressibleRolloutWindowPlan):
            raise TypeError("window_plan must be a SmoothCompressibleRolloutWindowPlan.")
        if not isinstance(statistics, SmoothCompressibleRolloutStatistics):
            raise TypeError("statistics must be SmoothCompressibleRolloutStatistics.")
        if len({value.parent_id for value in parents}) != len(parents):
            raise ValueError("Prepared rollout parent identities must be unique.")
        if len({value.sample_id for value in samples}) != len(samples):
            raise ValueError("Prepared rollout parent samples must be unique.")
        if len({value.window_id for value in windows_}) != len(windows_):
            raise ValueError("Prepared rollout window identities must be unique.")
        schema_ids = {value.schema.schema_id for value in parents}
        if len(schema_ids) != 1:
            raise ValueError("Prepared rollout parent schemas must match exactly.")
        schema_id = next(iter(schema_ids))
        if any(value.schema_id != schema_id for value in samples):
            raise ValueError("Parent sample and rollout schema identities do not match.")
        for parent, sample in zip(parents, samples, strict=True):
            expected_sample = ClosureSample(
                jnp.zeros((1,), dtype=jnp.dtype(parent.schema.dtype)),
                parent.sample_key(),
                schema_id=parent.schema.schema_id,
            )
            if sample.sample_id != expected_sample.sample_id:
                raise ValueError(
                    "Parent samples must bind each complete trajectory exactly."
                )
        if any(value.schema_id != schema_id for value in windows_):
            raise ValueError("Window and rollout schema identities do not match.")
        if (
            statistics.schema_id != schema_id
            or statistics.partition_id != partition.partition_id
        ):
            raise ValueError("Statistics identities do not match the prepared dataset.")
        assignment_ids = {value.sample_id for value in partition.assignments}
        if assignment_ids != {value.sample_id for value in samples}:
            raise ValueError("Partition assignments must exactly cover complete parents.")
        if any(not bool(np.all(np.asarray(value.valid))) for value in parents):
            raise ValueError("Failed oracle parent trajectories are rejected in full.")
        split_by_parent: dict[str, DatasetSplit] = {}
        for parent, sample in zip(parents, samples, strict=True):
            assignment = partition.assignment_for(sample.sample_id)
            if assignment.group_key != sample.key.group_key(partition.plan.level):
                raise ValueError(
                    "Partition group does not match its complete parent key."
                )
            split_by_parent[parent.parent_id] = assignment.split
        parent_ids = {value.parent_id for value in parents}
        if any(value.parent_id not in parent_ids for value in windows_):
            raise ValueError("A rollout window refers to an undeclared parent.")
        if any(split_by_parent[value.parent_id] != value.split for value in windows_):
            raise ValueError("Every window must retain its complete parent's split.")
        expected_selection = tuple(
            (
                parent.parent_id,
                anchor,
                partition.assignment_for(sample.sample_id).split,
            )
            for parent, sample in zip(parents, samples, strict=True)
            for anchor in window_plan.anchors(parent)
        )
        observed_selection = tuple(
            (value.parent_id, value.anchor, value.split) for value in windows_
        )
        if observed_selection != expected_selection or any(
            value.window_plan_id != window_plan.plan_id for value in windows_
        ):
            raise ValueError(
                "Windows must exactly match deterministic complete-parent selection."
            )
        split_windows = {
            split: tuple(value for value in windows_ if value.split == split)
            for split in ("train", "validation", "test")
        }
        if not split_windows["train"]:
            raise ValueError("Prepared rollout datasets require training windows.")
        training_pairs = tuple(
            (parent, sample)
            for parent, sample in zip(parents, samples, strict=True)
            if partition.assignment_for(sample.sample_id).split == "train"
        )
        if statistics.training_parent_ids != tuple(
            value[0].parent_id for value in training_pairs
        ) or statistics.training_sample_ids != tuple(
            value[1].sample_id for value in training_pairs
        ):
            raise ValueError(
                "Statistics must identify complete training parents exactly."
            )
        authoritative_statistics = _fit_statistics(
            parents, samples, partition, statistics.epsilon
        )
        if statistics.statistics_id != authoritative_statistics.statistics_id:
            raise ValueError(
                "Statistics must be fit once over complete training parents."
            )
        self.trajectories = parents
        self.parent_samples = samples
        self.windows = windows_
        self.train_windows = split_windows["train"]
        self.validation_windows = split_windows["validation"]
        self.test_windows = split_windows["test"]
        self.partition = partition
        self.window_plan = window_plan
        self.statistics = statistics
        self.schema_id = schema_id
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "prepared-smooth-compressible-d2v17-rollout-dataset",
                "parents": [value.parent_id for value in parents],
                "parent_samples": [value.sample_id for value in samples],
                "partition": partition.partition_id,
                "window_plan": window_plan.plan_id,
                "windows": [value.window_id for value in windows_],
                "statistics": statistics.statistics_id,
                "schema": schema_id,
                "splits": {
                    split: [value.window_id for value in split_windows[split]]
                    for split in ("train", "validation", "test")
                },
            }
        )


def prepare_smooth_compressible_rollout_dataset(
    trajectories: tuple[SmoothCompressibleRolloutTrajectory, ...],
    partition_plan: LeakageSafePartitionPlan,
    window_plan: SmoothCompressibleRolloutWindowPlan,
    /,
    *,
    epsilon: float = 1e-8,
) -> PreparedSmoothCompressibleRolloutDataset:
    """Partition complete oracle parents, fit them once, then derive windows."""

    parents = tuple(trajectories)
    if not parents or any(
        not isinstance(value, SmoothCompressibleRolloutTrajectory) for value in parents
    ):
        raise ValueError("Rollout preparation requires parent trajectories.")
    if not isinstance(partition_plan, LeakageSafePartitionPlan):
        raise TypeError("partition_plan must be a LeakageSafePartitionPlan.")
    if not isinstance(window_plan, SmoothCompressibleRolloutWindowPlan):
        raise TypeError("window_plan must be a SmoothCompressibleRolloutWindowPlan.")
    epsilon_ = float(epsilon)
    if not np.isfinite(epsilon_) or epsilon_ <= 0.0:
        raise ValueError("Preparation epsilon must be finite and positive.")
    if len({value.parent_id for value in parents}) != len(parents):
        raise ValueError("Rollout parent identities must be unique.")
    coordinates = {
        (
            value.case_id,
            value.trajectory_id,
            value.realization_id,
            value.time_block_id,
        )
        for value in parents
    }
    if len(coordinates) != len(parents):
        raise ValueError("Rollout parent coordinates must be unique.")
    if len({value.schema.schema_id for value in parents}) != 1:
        raise ValueError("Rollout parent schemas must match exactly.")
    if any(not bool(np.all(np.asarray(value.valid))) for value in parents):
        raise ValueError("Failed oracle parent trajectories are rejected in full.")
    samples = tuple(
        ClosureSample(
            jnp.zeros((1,), dtype=jnp.dtype(parent.schema.dtype)),
            parent.sample_key(),
            schema_id=parent.schema.schema_id,
        )
        for parent in parents
    )
    partition = partition_plan.assign(samples)
    statistics = _fit_statistics(parents, samples, partition, epsilon_)
    windows: list[SmoothCompressibleRolloutWindow] = []
    for parent, sample in zip(parents, samples, strict=True):
        split = partition.assignment_for(sample.sample_id).split
        windows.extend(
            SmoothCompressibleRolloutWindow(parent, window_plan, anchor, split=split)
            for anchor in window_plan.anchors(parent)
        )
    return PreparedSmoothCompressibleRolloutDataset(
        parents,
        samples,
        tuple(windows),
        partition,
        window_plan,
        statistics,
    )


__all__ = [
    "PreparedSmoothCompressibleRolloutDataset",
    "SmoothCompressibleRolloutSchema",
    "SmoothCompressibleRolloutStatistics",
    "SmoothCompressibleRolloutTrajectory",
    "SmoothCompressibleRolloutWindow",
    "SmoothCompressibleRolloutWindowPlan",
    "prepare_smooth_compressible_rollout_dataset",
]
