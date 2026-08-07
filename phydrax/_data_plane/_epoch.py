#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil

from ._ordering import EPOCH_ORDER_ALGORITHM, StatelessIndexPermutation


@dataclass(frozen=True, slots=True)
class IndexEpochPlan:
    """O(1)-memory deterministic index batches for one finite logical epoch."""

    source_size: int
    batch_size: int
    shuffle: bool
    seed: int
    epoch: int
    drop_last: bool

    def __post_init__(self):
        if int(self.source_size) <= 0:
            raise ValueError("source_size must be positive.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(self.seed) < 0:
            raise ValueError("seed must be nonnegative.")
        if int(self.epoch) < 0:
            raise ValueError("epoch must be nonnegative.")

    @property
    def batch_count(self) -> int:
        if self.drop_last:
            return int(self.source_size) // int(self.batch_size)
        return ceil(int(self.source_size) / int(self.batch_size))

    @property
    def ordering(self) -> str:
        return EPOCH_ORDER_ALGORITHM

    def batch(self, batch_index: int, /) -> tuple[int, ...]:
        index = int(batch_index)
        if index < 0 or index >= self.batch_count:
            raise IndexError("Epoch batch index is out of range.")
        start = index * int(self.batch_size)
        stop = min(start + int(self.batch_size), int(self.source_size))
        if not self.shuffle:
            return tuple(range(start, stop))
        permutation = StatelessIndexPermutation(
            int(self.source_size),
            int(self.seed),
            int(self.epoch),
        )
        return tuple(permutation(position) for position in range(start, stop))

    def iter_batches(
        self,
        *,
        start_batch: int = 0,
    ) -> Iterator[tuple[int, tuple[int, ...]]]:
        start = int(start_batch)
        if start < 0 or start > self.batch_count:
            raise ValueError("start_batch must lie within the epoch plan.")
        for batch_index in range(start, self.batch_count):
            yield batch_index, self.batch(batch_index)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        for _, indices in self.iter_batches():
            yield indices


__all__ = ["IndexEpochPlan"]
