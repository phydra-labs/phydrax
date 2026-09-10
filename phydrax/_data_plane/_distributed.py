#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic process-local batches and global-array construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import jax
import numpy as np
from jax.sharding import Sharding

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
        return np.asarray(self.valid, dtype=bool)


@dataclass(frozen=True, slots=True)
class DistributedIndexEpochPlan:
    """Process-symmetric local views of a deterministic global epoch plan."""

    global_plan: IndexEpochPlan
    process_count: int
    process_index: int
    devices_per_process: int
    global_device_count: int
    local_batch_size: int

    def __init__(
        self,
        global_plan: IndexEpochPlan,
        process_count: int,
        process_index: int,
        devices_per_process: int = 1,
    ) -> None:
        if not isinstance(global_plan, IndexEpochPlan):
            raise TypeError("global_plan must be an IndexEpochPlan")
        processes = int(process_count)
        process = int(process_index)
        local_devices = int(devices_per_process)
        if processes <= 0 or local_devices <= 0:
            raise ValueError("process_count and devices_per_process must be positive")
        if not 0 <= process < processes:
            raise ValueError("process_index must be in [0, process_count)")
        object.__setattr__(self, "global_plan", global_plan)
        object.__setattr__(self, "process_count", processes)
        object.__setattr__(self, "process_index", process)
        object.__setattr__(self, "devices_per_process", local_devices)
        object.__setattr__(
            self,
            "global_device_count",
            processes * local_devices,
        )
        per_device = ceil(global_plan.batch_size / (processes * local_devices))
        object.__setattr__(
            self,
            "local_batch_size",
            per_device * local_devices,
        )

    @classmethod
    def current(cls, global_plan: IndexEpochPlan) -> DistributedIndexEpochPlan:
        local_devices = jax.local_device_count()
        if jax.device_count() != jax.process_count() * local_devices:
            raise ValueError(
                "process-local epoch batching requires equal device counts per process"
            )
        return cls(
            global_plan,
            jax.process_count(),
            jax.process_index(),
            local_devices,
        )

    @property
    def batch_count(self) -> int:
        return self.global_plan.batch_count

    @property
    def global_batch_capacity(self) -> int:
        return self.local_batch_size * self.process_count

    def batch(self, batch_index: int, /) -> ProcessLocalBatch:
        global_indices = self.global_plan.batch(batch_index)
        capacity = self.global_batch_capacity
        if not global_indices:
            raise ValueError("global epoch batches must not be empty")
        padded = global_indices + (global_indices[0],) * (capacity - len(global_indices))
        valid = (True,) * len(global_indices) + (False,) * (
            capacity - len(global_indices)
        )
        start = self.process_index * self.local_batch_size
        stop = start + self.local_batch_size
        return ProcessLocalBatch(
            batch_index,
            padded[start:stop],
            valid[start:stop],
            len(global_indices),
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
    "ProcessLocalBatch",
    "make_global_array_from_process_local_data",
)
