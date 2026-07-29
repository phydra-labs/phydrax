#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Any, Iterator, Sequence

import jax
import numpy as np

from ..models.core._operator import OperatorBatch, OperatorTargetBatch
from ..models.core._operator_sharding import (
    OperatorShardingPolicy,
    shard_operator_batch,
    shard_operator_targets,
)
from ._dataset import OperatorCaseProvenance, OperatorDataset
from ._dtype import OperatorDTypePolicy
from ._normalization import OperatorNormalizationPolicy
from ._sampling import (
    AnchorQuerySamplingPolicy,
    InMemoryOperatorCaseSource,
    OperatorCaseSource,
    read_operator_case_batch,
    take_function_samples,
)


@dataclass(frozen=True)
class OperatorTrainingBatch:
    """One case mini-batch after optional normalization and device placement."""

    batch: OperatorBatch
    targets: OperatorTargetBatch
    indices: tuple[int, ...]
    epoch: int = 0
    batch_index: int = 0
    microstep: int = 0
    provenance: tuple[OperatorCaseProvenance, ...] = ()
    physical_batch: OperatorBatch | None = None
    physical_targets: OperatorTargetBatch | None = None


class OperatorBatchLoader:
    """Deterministic epoch loader with asynchronous device-put prefetching."""

    def __init__(
        self,
        dataset: OperatorDataset | OperatorCaseSource,
        /,
        *,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        prefetch: int = 2,
        normalization: OperatorNormalizationPolicy | None = None,
        dtype_policy: OperatorDTypePolicy | None = None,
        sharding_policy: OperatorShardingPolicy | None = None,
        sampling: AnchorQuerySamplingPolicy | None = None,
        split: str = "train",
    ):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(prefetch) <= 0:
            raise ValueError("prefetch must be positive.")
        self.dataset = dataset
        self.source = (
            InMemoryOperatorCaseSource(dataset)
            if isinstance(dataset, OperatorDataset)
            else dataset
        )
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.prefetch = int(prefetch)
        self.normalization = normalization
        self.dtype_policy = dtype_policy
        self.sharding_policy = sharding_policy
        self.sampling = sampling
        self.split = str(split)

    @property
    def batches_per_epoch(self) -> int:
        if self.drop_last:
            return self.source.size // self.batch_size
        return ceil(self.source.size / self.batch_size)

    def configuration(self) -> dict[str, Any]:
        """Return stable loader semantics used by exact-resume checkpoints."""
        sampling = self.sampling
        sampling_contract = (
            None
            if sampling is None
            else {
                "anchor_counts": [list(item) for item in sampling.anchor_counts],
                "query_counts": [list(item) for item in sampling.query_counts],
                "strategy": sampling.strategy,
                "query_strategy": sampling.query_strategy,
                "seed": sampling.seed,
                "fixed_anchor_indices": [
                    [name, list(indices)]
                    for name, indices in sampling.fixed_anchor_indices
                ],
                "fixed_query_indices": [
                    [name, list(indices)]
                    for name, indices in sampling.fixed_query_indices
                ],
            }
        )
        return {
            "source_size": self.source.size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "split": self.split,
            "sampling": sampling_contract,
        }

    def fixed_query_fingerprints(
        self,
        query_names: Sequence[str],
        /,
    ) -> dict[str, str]:
        """Validate and fingerprint one fixed physical query across every case."""
        names = tuple(str(name) for name in query_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Fixed query names must be non-empty and unique.")
        sampling = self.sampling
        if sampling is not None:
            sampled = set(dict(sampling.query_counts)) & set(names)
            fixed = set(dict(sampling.fixed_query_indices))
            if sampled and (
                sampling.query_strategy != "fixed_indices"
                or not sampled.issubset(fixed)
            ):
                raise ValueError(
                    "Fixed queries only support explicit fixed-index query sampling."
                )

        reference: dict[str, str] = {}
        for index in range(self.source.size):
            metadata = self.source.case_metadata(index)
            queries = dict(metadata.queries)
            if sampling is not None:
                request = sampling.request(
                    metadata,
                    split=self.split,
                    epoch=0,
                    case_index=index,
                )
                for name, selection in request.query_selections.items():
                    queries[name] = take_function_samples(queries[name], selection)
            for name in names:
                if name not in queries:
                    raise KeyError(f"Fixed query {name!r} is absent from case metadata.")
                fingerprint = queries[name].geometry_fingerprint()
                previous = reference.setdefault(name, fingerprint)
                if fingerprint != previous:
                    raise ValueError(
                        f"Fixed query {name!r} changes physical geometry across cases."
                    )
        return reference


    def _indices(self, epoch: int) -> tuple[tuple[int, ...], ...]:
        indices = np.arange(self.source.size)
        if self.shuffle:
            indices = np.random.default_rng(self.seed + int(epoch)).permutation(indices)
        chunks = []
        for start in range(0, self.source.size, self.batch_size):
            chunk = indices[start : start + self.batch_size]
            if self.drop_last and int(chunk.size) < self.batch_size:
                continue
            chunks.append(tuple(int(value) for value in chunk))
        return tuple(chunks)

    def _prepare(
        self,
        indices: tuple[int, ...],
        epoch: int,
        batch_index: int,
    ) -> OperatorTrainingBatch:
        selected = read_operator_case_batch(
            self.source,
            indices,
            sampling=self.sampling,
            split=self.split,
            epoch=epoch,
        )
        batch = selected.batch
        targets = selected.targets
        physical_batch = batch
        physical_targets = targets
        if self.normalization is not None:
            batch = self.normalization.normalize_batch(batch)
            targets = self.normalization.normalize_targets(targets)
        if self.dtype_policy is not None:
            batch = self.dtype_policy.cast_batch(batch)
            targets = self.dtype_policy.cast_targets(targets)
        if self.sharding_policy is not None:
            batch = shard_operator_batch(batch, self.sharding_policy)
            targets = shard_operator_targets(targets, self.sharding_policy)
            if physical_batch is not batch:
                physical_batch = shard_operator_batch(
                    physical_batch,
                    self.sharding_policy,
                )
                physical_targets = shard_operator_targets(
                    physical_targets,
                    self.sharding_policy,
                )
        else:
            batch = jax.tree_util.tree_map(
                lambda leaf: (
                    jax.device_put(leaf) if isinstance(leaf, jax.Array) else leaf
                ),
                batch,
            )
            if physical_batch is not batch:
                physical_batch = jax.tree_util.tree_map(
                    lambda leaf: (
                        jax.device_put(leaf) if isinstance(leaf, jax.Array) else leaf
                    ),
                    physical_batch,
                )
                physical_targets = physical_targets.map_values(jax.device_put)
            targets = targets.map_values(jax.device_put)
        assert selected.provenance is not None
        return OperatorTrainingBatch(
            batch=batch,
            targets=targets,
            indices=indices,
            epoch=int(epoch),
            batch_index=int(batch_index),
            microstep=int(epoch) * self.batches_per_epoch + int(batch_index),
            provenance=tuple(selected.provenance),
            physical_batch=physical_batch,
            physical_targets=physical_targets,
        )
    def epoch(self, epoch: int = 0, /) -> Iterator[OperatorTrainingBatch]:
        """Yield one reproducible epoch while keeping future transfers in flight."""
        chunks = iter(enumerate(self._indices(epoch)))
        queue: deque[OperatorTrainingBatch] = deque()
        for _ in range(self.prefetch):
            item = next(chunks, None)
            if item is None:
                break
            batch_index, indices = item
            queue.append(self._prepare(indices, epoch, batch_index))
        while queue:
            current = queue.popleft()
            item = next(chunks, None)
            if item is not None:
                batch_index, indices = item
                queue.append(self._prepare(indices, epoch, batch_index))
            yield current


__all__ = ["OperatorBatchLoader", "OperatorTrainingBatch"]
