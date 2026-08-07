#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import jax

from ..._data_plane import (
    BoundedPrefetchIterator,
    EPOCH_ORDER_ALGORITHM,
    IndexEpochPlan,
)
from ..._fingerprint import canonical_fingerprint, canonical_mapping
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


_LOADER_FINGERPRINT_FORMAT = "phydrax-operator-loader-v2"


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


class OperatorEpochPlan(IndexEpochPlan):
    """Operator-facing deterministic case-index plan for one logical epoch."""

    __slots__ = ()


class OperatorBatchEpoch(Iterator[OperatorTrainingBatch]):
    """Closable operator epoch backed by shared bounded preparation mechanics."""

    def __init__(
        self,
        loader: OperatorBatchLoader,
        plan: OperatorEpochPlan,
        /,
        *,
        start_batch: int,
    ):
        self._loader = loader
        self._plan = plan
        self._next_batch_index = int(start_batch)
        if self._next_batch_index < 0 or self._next_batch_index > plan.batch_count:
            raise ValueError("start_batch must lie within the epoch plan.")
        capacity = (
            loader.effective_prefetch if self._next_batch_index < plan.batch_count else 0
        )
        self._iterator = BoundedPrefetchIterator(
            plan.iter_batches(start_batch=self._next_batch_index),
            self._prepare_item,
            capacity=capacity,
            thread_name=f"phydrax-operator-epoch-{plan.epoch}",
        )

    @property
    def closed(self) -> bool:
        return self._iterator.closed

    def __iter__(self) -> OperatorBatchEpoch:
        return self

    def __next__(self) -> OperatorTrainingBatch:
        item = next(self._iterator)
        if not isinstance(item, OperatorTrainingBatch):
            self.close()
            raise RuntimeError("Operator epoch producer emitted an invalid item.")
        self._next_batch_index = item.batch_index + 1
        return item

    def close(self) -> None:
        self._iterator.close()

    def __enter__(self) -> OperatorBatchEpoch:
        self._iterator.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._iterator.__exit__(exc_type, exc_value, traceback)

    def _prepare_item(
        self,
        item: tuple[int, tuple[int, ...]],
        /,
    ) -> OperatorTrainingBatch:
        batch_index, indices = item
        return self._loader._prepare(
            indices,
            self._plan.epoch,
            batch_index,
        )


class OperatorBatchLoader:
    """Deterministic operator batches with bounded ordered host prefetching."""

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
        if int(seed) < 0:
            raise ValueError("seed must be nonnegative.")
        if int(prefetch) < 0:
            raise ValueError("prefetch must be nonnegative.")
        source = (
            InMemoryOperatorCaseSource(dataset)
            if isinstance(dataset, OperatorDataset)
            else dataset
        )
        if not isinstance(source, OperatorCaseSource):
            raise TypeError("dataset must be an OperatorDataset or OperatorCaseSource.")
        self.source = source
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
        return self.epoch_plan().batch_count

    @property
    def effective_prefetch(self) -> int:
        return self.prefetch if self.source.background_read_safe else 0

    @property
    def fingerprint(self) -> str:
        """Hash source content and every loader setting that changes logical batches."""
        payload = {
            "format": _LOADER_FINGERPRINT_FORMAT,
            "source": {
                "configuration": canonical_mapping(self.source.configuration()),
                "content_fingerprint": self.source.content_fingerprint,
            },
            "loader": self.configuration(),
        }
        return canonical_fingerprint(payload)

    def configuration(self) -> dict[str, Any]:
        """Return stable batch semantics used by exact-resume checkpoints."""
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
            "ordering": EPOCH_ORDER_ALGORITHM,
            "sampling": sampling_contract,
            "normalization": (
                None if self.normalization is None else self.normalization.to_dict()
            ),
            "dtype_policy": (
                None if self.dtype_policy is None else self.dtype_policy.to_dict()
            ),
            "sharding": (
                None
                if self.sharding_policy is None
                else {
                    "mesh_axis": self.sharding_policy.mesh_axis,
                    "case_axis": self.sharding_policy.case_axis,
                    "mesh_shape": list(self.sharding_policy.mesh.devices.shape),
                    "device_count": int(self.sharding_policy.mesh.devices.size),
                }
            ),
        }

    def epoch_plan(self, epoch: int = 0, /) -> OperatorEpochPlan:
        """Return the stateless case-index plan for one epoch."""
        return OperatorEpochPlan(
            source_size=self.source.size,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed,
            epoch=int(epoch),
            drop_last=self.drop_last,
        )

    def prepare_indices(
        self,
        indices: Sequence[int],
        /,
        *,
        epoch: int,
        batch_index: int,
    ) -> OperatorTrainingBatch:
        """Prepare explicit case indices through the configured loader pipeline."""
        selected = tuple(int(index) for index in indices)
        if not selected:
            raise ValueError("indices must contain at least one case.")
        if any(index < 0 or index >= self.source.size for index in selected):
            raise ValueError("indices contain a case outside the source.")
        return self._prepare(selected, int(epoch), int(batch_index))

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
                sampling.query_strategy != "fixed_indices" or not sampled.issubset(fixed)
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

    def epoch(
        self,
        epoch: int = 0,
        /,
        *,
        start_batch: int = 0,
    ) -> OperatorBatchEpoch:
        """Return one reproducible epoch beginning at an absolute batch cursor."""
        plan = self.epoch_plan(epoch)
        return OperatorBatchEpoch(self, plan, start_batch=start_batch)


__all__ = [
    "OperatorBatchEpoch",
    "OperatorBatchLoader",
    "OperatorEpochPlan",
    "OperatorTrainingBatch",
]
