#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic group-local batches and global-array construction.

Every distributed data-plane view binds one `ExecutionGroupSpec` through
`ExecutionGroupRank`: the group's named data axis, its data-shard count and the
contiguous data shards addressed by one member process. Group-local ranks never
assume that the group spans the whole initialized process world, so a child
group of `partition_execution_group_specs` (or any explicit subgroup) receives
its own shard positions while global epoch/draw identities stay immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import jax
import numpy as np
from jax.sharding import Sharding

from .._execution_resources import ExecutionGroupSpec
from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint
from ..typing import checked
from ._epoch import IndexEpochPlan


@dataclass(frozen=True, slots=True)
class ProcessLocalBatch:
    """Fixed-capacity local slice of one globally ordered logical batch."""

    batch_index: int
    indices: tuple[int, ...]
    valid: tuple[bool, ...]
    global_valid_count: int

    def __post_init__(self) -> None:
        if self.batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        if not self.indices or len(self.indices) != len(self.valid):
            raise ValueError("indices and valid must have matching positive capacity")
        if self.global_valid_count < 0:
            raise ValueError("global_valid_count must be non-negative")
        if any(index < 0 for index in self.indices):
            raise ValueError("indices must be non-negative")

    @property
    def local_valid_count(self) -> int:
        return sum(self.valid)

    @property
    def mask(self) -> np.ndarray:
        return np.asarray(self.valid, dtype=np.bool_)


def _group_axes(spec: ExecutionGroupSpec, /) -> tuple[tuple[str, int], ...]:
    # A group without declared mesh axes binds one ``device`` axis, exactly as
    # `ExecutionGroup` builds its mesh.
    return spec.mesh_axes or (("device", spec.device_count),)


def _shard_coordinates(spec: ExecutionGroupSpec, axis: str, /) -> np.ndarray:
    """Data-axis coordinate of every group device in mesh (device-key) order."""

    axes = _group_axes(spec)
    names = tuple(name for name, _ in axes)
    if axis not in names:
        raise ValueError(
            f"execution group {spec.group_id!r} has no mesh axis {axis!r}; "
            f"its axes are {names}"
        )
    shape = tuple(size for _, size in axes)
    positions = np.arange(spec.device_count, dtype=np.int64)
    return np.unravel_index(positions, shape)[names.index(axis)].astype(np.int64)


@dataclass(frozen=True, slots=True)
class ExecutionGroupRank:
    """Group-local data-shard rank of one member process of an execution group.

    The data axis is one named mesh axis of the group (the only axis when the
    group declares one). Data shard ``s`` is coordinate ``s`` along that axis;
    devices that differ only along other axes (for example a domain owner axis
    of a data×owner mesh) address the same data shard and hold replicated data
    rows. ``rank`` is the member's position among the group's processes in
    ascending global process order, never the global ``jax.process_index``.
    Every member process must address one contiguous range of the same number
    of data shards so process-local arrays assemble into the group's global
    arrays without renumbering. ``layout_id`` is rank-independent: every member
    of one group/axis/shard layout shares it.
    """

    group_id: str
    axis: str
    process_index: int
    rank: int
    rank_count: int
    shard_count: int
    shards: tuple[int, ...]
    layout_id: str

    @checked
    def __init__(
        self,
        spec: ExecutionGroupSpec,
        process_index: int,
        /,
        *,
        axis: str | None = None,
    ) -> None:
        process = int(process_index)
        members = tuple(sorted(spec.process_indices))
        if process not in members:
            raise ValueError(
                f"process {process} is not a member of execution group {spec.group_id!r}"
            )
        axes = _group_axes(spec)
        if axis is None:
            if len(axes) != 1:
                raise ValueError(
                    "a multi-axis execution group requires an explicit data axis"
                )
            axis_ = axes[0][0]
        else:
            axis_ = str(axis)
        coordinates = _shard_coordinates(spec, axis_)
        shard_count = int(dict(axes)[axis_])
        owners = np.asarray([owner for owner, _ in spec.device_keys], dtype=np.int64)
        addressed: dict[int, tuple[int, ...]] = {}
        for member in members:
            rows = np.unique(coordinates[owners == member])
            if rows.size == 0:
                raise ValueError(
                    f"group process {member} owns no device of execution group "
                    f"{spec.group_id!r}"
                )
            if int(rows[-1] - rows[0]) + 1 != rows.size:
                raise ValueError(
                    f"group process {member} addresses non-contiguous data shards "
                    f"{rows.tolist()} along {axis_!r}; order group devices so each "
                    "process owns one contiguous data range"
                )
            addressed[member] = tuple(int(row) for row in rows)
        counts = {len(rows) for rows in addressed.values()}
        if len(counts) != 1:
            raise ValueError(
                "every group process must address the same number of data shards "
                f"along {axis_!r}"
            )
        object.__setattr__(self, "group_id", spec.group_id)
        object.__setattr__(self, "axis", axis_)
        object.__setattr__(self, "process_index", process)
        object.__setattr__(self, "rank", members.index(process))
        object.__setattr__(self, "rank_count", len(members))
        object.__setattr__(self, "shard_count", shard_count)
        object.__setattr__(self, "shards", addressed[process])
        object.__setattr__(
            self,
            "layout_id",
            canonical_fingerprint(
                {
                    "kind": "execution-group-data-layout",
                    "group_id": spec.group_id,
                    "axis": axis_,
                    "shard_count": shard_count,
                    "local_shards": len(addressed[process]),
                    "addressed": [
                        [member, list(addressed[member])] for member in members
                    ],
                }
            ),
        )

    @classmethod
    def current(
        cls,
        group: ExecutionGroup,
        /,
        *,
        axis: str | None = None,
    ) -> ExecutionGroupRank:
        """Rank of the calling process in a live execution group."""

        if not isinstance(group, ExecutionGroup):
            raise TypeError("group must be a live ExecutionGroup")
        return cls(group.spec, jax.process_index(), axis=axis)

    @property
    def local_shard_count(self) -> int:
        return len(self.shards)

    @property
    def shard_offset(self) -> int:
        """First data shard addressed by this process."""

        return self.shards[0]


def _shard_batch_size(global_plan: IndexEpochPlan, shard_count: int, /) -> int:
    return ceil(int(global_plan.batch_size) / shard_count)


def _epoch_slice(
    global_plan: IndexEpochPlan,
    shard_count: int,
    batch_index: int,
    start: int,
    stop: int,
    /,
) -> ProcessLocalBatch:
    global_indices = global_plan.batch(batch_index)
    if not global_indices:
        raise ValueError("global epoch batches must not be empty")
    padding = _shard_batch_size(global_plan, shard_count) * shard_count - len(
        global_indices
    )
    indices = global_indices + (global_indices[0],) * padding
    valid = (True,) * len(global_indices) + (False,) * padding
    return ProcessLocalBatch(
        batch_index,
        indices[start:stop],
        valid[start:stop],
        len(global_indices),
    )


def epoch_shard(
    global_plan: IndexEpochPlan,
    shard_count: int,
    batch_index: int,
    shard: int,
    /,
) -> ProcessLocalBatch:
    """Fixed-capacity slice of data shard ``shard`` of one global batch.

    The global batch is padded to ``ceil(batch_size / shard_count) *
    shard_count`` entries and cut into equal shards in shard order. Padding
    duplicates a valid logical index only as storage; its validity flag gives
    it zero scientific mass. This is the single shard-slicing rule of every
    group-local view (`DistributedIndexEpochPlan`) and of lane-reference data
    shards, so a lane oracle and a device group of the same shard count see
    identical global indices.
    """

    count = int(shard_count)
    position = int(shard)
    if count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= position < count:
        raise IndexError(f"data shard {position} is outside [0, {count})")
    width = _shard_batch_size(global_plan, count)
    start = position * width
    return _epoch_slice(global_plan, count, batch_index, start, start + width)


@dataclass(frozen=True, slots=True)
class DistributedIndexEpochPlan:
    """Group-local views of one deterministic global epoch plan.

    Shards follow `epoch_shard` over the group's data shards. Indices are
    global epoch identities and never depend on the rank.
    """

    global_plan: IndexEpochPlan
    rank: ExecutionGroupRank
    shard_batch_size: int

    @checked
    def __init__(self, global_plan: IndexEpochPlan, rank: ExecutionGroupRank) -> None:
        object.__setattr__(self, "global_plan", global_plan)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(
            self, "shard_batch_size", _shard_batch_size(global_plan, rank.shard_count)
        )

    @classmethod
    def current(
        cls,
        global_plan: IndexEpochPlan,
        group: ExecutionGroup,
        /,
        *,
        axis: str | None = None,
    ) -> DistributedIndexEpochPlan:
        """View of the calling process in a live execution group."""

        return cls(global_plan, ExecutionGroupRank.current(group, axis=axis))

    @property
    def batch_count(self) -> int:
        return self.global_plan.batch_count

    @property
    def local_batch_size(self) -> int:
        return self.shard_batch_size * self.rank.local_shard_count

    @property
    def global_batch_capacity(self) -> int:
        return self.shard_batch_size * self.rank.shard_count

    def shard(self, batch_index: int, shard: int, /) -> ProcessLocalBatch:
        """Fixed-capacity slice of any data shard of one global batch.

        Planners that pack every shard symmetrically (each process derives the
        same per-shard plan from global metadata) read other ranks' shards here
        without assuming any process layout.
        """

        return epoch_shard(self.global_plan, self.rank.shard_count, batch_index, shard)

    def batch(self, batch_index: int, /) -> ProcessLocalBatch:
        """Contiguous slice of every data shard addressed by this process."""

        start = self.rank.shard_offset * self.shard_batch_size
        return _epoch_slice(
            self.global_plan,
            self.rank.shard_count,
            batch_index,
            start,
            start + self.local_batch_size,
        )


def make_global_array_from_process_local_data(
    sharding: Sharding,
    local_data: Any,
    /,
    *,
    global_shape: Any | None = None,
) -> Any:
    """Construct a global JAX array tree without global host materialization."""

    return jax.make_array_from_process_local_data(
        sharding,
        local_data,
        global_shape,
    )


__all__ = (
    "DistributedIndexEpochPlan",
    "epoch_shard",
    "ExecutionGroupRank",
    "ProcessLocalBatch",
    "make_global_array_from_process_local_data",
)
