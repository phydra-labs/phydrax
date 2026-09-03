#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


CovarianceStorage: TypeAlias = Literal["dense", "factor"]


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _balanced_ranges(size: int, shards: int, /) -> tuple[tuple[int, int], ...]:
    quotient, remainder = divmod(size, shards)
    starts: list[tuple[int, int]] = []
    start = 0
    for shard in range(shards):
        count = quotient + (1 if shard < remainder else 0)
        starts.append((start, start + count))
        start += count
    return tuple(starts)


class DistributedShard(StrictModule, NonTrainableState):
    process_index: int = eqx.field(static=True)
    start: int = eqx.field(static=True)
    stop: int = eqx.field(static=True)
    local_count: int = eqx.field(static=True)
    local_bytes: int = eqx.field(static=True)
    shard_id: str = eqx.field(static=True)

    def __init__(
        self,
        process_index: int,
        start: int,
        stop: int,
        local_bytes: int,
        /,
        *,
        owner_id: str,
    ):
        process = int(process_index)
        start_ = int(start)
        stop_ = int(stop)
        bytes_ = int(local_bytes)
        if process < 0 or start_ < 0 or stop_ < start_ or bytes_ < 0 or not owner_id:
            raise ValueError("Distributed shard metadata is invalid.")
        self.process_index = process
        self.start = start_
        self.stop = stop_
        self.local_count = stop_ - start_
        self.local_bytes = bytes_
        self.shard_id = canonical_fingerprint(
            {
                "kind": "statistical-dynamics-shard",
                "owner": owner_id,
                "process": process,
                "range": [start_, stop_],
                "bytes": bytes_,
            }
        )


class DistributedBatchLayout(StrictModule, NonTrainableState):
    shards: tuple[DistributedShard, ...]
    global_batch_size: int = eqx.field(static=True)
    process_count: int = eqx.field(static=True)
    item_bytes: int = eqx.field(static=True)
    maximum_local_bytes: int = eqx.field(static=True)
    semantic_layout_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        global_batch_size: int,
        process_count: int,
        /,
        *,
        item_bytes: int,
        maximum_local_bytes: int,
    ):
        size = _positive_integer(global_batch_size, "global_batch_size")
        processes = _positive_integer(process_count, "process_count")
        item = _positive_integer(item_bytes, "item_bytes")
        maximum = _positive_integer(maximum_local_bytes, "maximum_local_bytes")
        semantic_id = canonical_fingerprint(
            {
                "kind": "statistical-dynamics-batch-layout",
                "global_batch_size": size,
                "item_bytes": item,
            }
        )
        topology_id = canonical_fingerprint(
            {
                "kind": "distributed-statistical-batch-topology",
                "semantic_layout": semantic_id,
                "process_count": processes,
            }
        )
        ranges = _balanced_ranges(size, processes)
        shards = tuple(
            DistributedShard(
                process,
                start,
                stop,
                (stop - start) * item,
                owner_id=topology_id,
            )
            for process, (start, stop) in enumerate(ranges)
        )
        if max(shard.local_bytes for shard in shards) > maximum:
            raise MemoryError("Distributed batch shard exceeds maximum_local_bytes.")
        self.shards = shards
        self.global_batch_size = size
        self.process_count = processes
        self.item_bytes = item
        self.maximum_local_bytes = maximum
        self.semantic_layout_id = semantic_id
        self.topology_id = topology_id

    def shard(self, batch: ArrayLike, /) -> tuple[Array, ...]:
        value = jnp.asarray(batch)
        if value.ndim < 1 or value.shape[0] != self.global_batch_size:
            raise ValueError("Batch leading dimension does not match distributed layout.")
        return tuple(value[item.start : item.stop] for item in self.shards)

    def assemble(self, shards: Sequence[ArrayLike], /) -> Array:
        values = tuple(jnp.asarray(value) for value in shards)
        if len(values) != self.process_count or any(
            value.shape[0] != shard.local_count
            for value, shard in zip(values, self.shards, strict=True)
        ):
            raise ValueError("Batch shards do not match the distributed layout.")
        trailing = values[0].shape[1:]
        if any(value.shape[1:] != trailing for value in values):
            raise ValueError("Batch shards must have identical trailing shapes.")
        return jnp.concatenate(values, axis=0)


class DistributedCovarianceLayout(StrictModule, NonTrainableState):
    shards: tuple[DistributedShard, ...]
    covariance_dimension: int = eqx.field(static=True)
    process_count: int = eqx.field(static=True)
    factor_rank: int | None = eqx.field(static=True)
    storage: CovarianceStorage = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    maximum_local_bytes: int = eqx.field(static=True)
    semantic_layout_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        covariance_dimension: int,
        process_count: int,
        /,
        *,
        storage: CovarianceStorage = "dense",
        factor_rank: int | None = None,
        dtype: Any = float,
        maximum_local_bytes: int = 512 * 1024 * 1024,
    ):
        dimension = _positive_integer(covariance_dimension, "covariance_dimension")
        processes = _positive_integer(process_count, "process_count")
        maximum = _positive_integer(maximum_local_bytes, "maximum_local_bytes")
        dtype_ = np.dtype(dtype)
        if not np.issubdtype(dtype_, np.inexact):
            raise TypeError("Distributed covariance dtype must be inexact.")
        if storage not in ("dense", "factor"):
            raise ValueError("storage must be 'dense' or 'factor'.")
        rank = None if factor_rank is None else int(factor_rank)
        if storage == "dense" and rank is not None:
            raise ValueError("factor_rank is only valid for factor storage.")
        if storage == "factor" and (rank is None or rank < 0 or rank > dimension):
            raise ValueError("Factor storage requires rank in [0, covariance_dimension].")
        columns = dimension if storage == "dense" else int(rank)
        semantic_id = canonical_fingerprint(
            {
                "kind": "statistical-dynamics-covariance-layout",
                "dimension": dimension,
                "storage": storage,
                "factor_rank": rank,
                "dtype": dtype_.str,
            }
        )
        topology_id = canonical_fingerprint(
            {
                "kind": "distributed-statistical-covariance-topology",
                "semantic_layout": semantic_id,
                "process_count": processes,
            }
        )
        ranges = _balanced_ranges(dimension, processes)
        shards = tuple(
            DistributedShard(
                process,
                start,
                stop,
                (stop - start) * columns * dtype_.itemsize,
                owner_id=topology_id,
            )
            for process, (start, stop) in enumerate(ranges)
        )
        if max(shard.local_bytes for shard in shards) > maximum:
            raise MemoryError("Distributed covariance shard exceeds maximum_local_bytes.")
        self.shards = shards
        self.covariance_dimension = dimension
        self.process_count = processes
        self.factor_rank = rank
        self.storage = storage
        self.dtype = dtype_.str
        self.maximum_local_bytes = maximum
        self.semantic_layout_id = semantic_id
        self.topology_id = topology_id

    @property
    def global_shape(self) -> tuple[int, int]:
        columns = (
            self.covariance_dimension
            if self.storage == "dense"
            else int(self.factor_rank)
        )
        return self.covariance_dimension, columns

    def shard(self, covariance: ArrayLike, /) -> tuple[Array, ...]:
        value = jnp.asarray(covariance)
        if value.shape != self.global_shape:
            raise ValueError(
                f"Covariance representation must have shape {self.global_shape}; got {value.shape}."
            )
        return tuple(value[item.start : item.stop, :] for item in self.shards)

    def assemble(self, shards: Sequence[ArrayLike], /) -> Array:
        values = tuple(jnp.asarray(value) for value in shards)
        columns = self.global_shape[1]
        if len(values) != self.process_count or any(
            value.shape != (shard.local_count, columns)
            for value, shard in zip(values, self.shards, strict=True)
        ):
            raise ValueError("Covariance shards do not match the distributed layout.")
        return jnp.concatenate(values, axis=0)


class DistributedStatisticalLayout(StrictModule, NonTrainableState):
    batch: DistributedBatchLayout
    covariance: DistributedCovarianceLayout
    topology_id: str = eqx.field(static=True)
    semantic_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        batch: DistributedBatchLayout,
        covariance: DistributedCovarianceLayout,
        /,
    ):
        if not isinstance(batch, DistributedBatchLayout) or not isinstance(
            covariance, DistributedCovarianceLayout
        ):
            raise TypeError(
                "Distributed statistical layout requires batch and covariance layouts."
            )
        if batch.process_count != covariance.process_count:
            raise ValueError(
                "Batch and covariance layouts must use one process topology."
            )
        self.batch = batch
        self.covariance = covariance
        self.semantic_layout_id = canonical_fingerprint(
            {
                "kind": "distributed-statistical-semantic-layout",
                "batch": batch.semantic_layout_id,
                "covariance": covariance.semantic_layout_id,
            }
        )
        self.topology_id = canonical_fingerprint(
            {
                "kind": "distributed-statistical-topology",
                "semantic_layout": self.semantic_layout_id,
                "batch_topology": batch.topology_id,
                "covariance_topology": covariance.topology_id,
            }
        )


class DistributedRestartRelation(StrictModule, NonTrainableState):
    source: DistributedStatisticalLayout
    target: DistributedStatisticalLayout
    source_semantic_layout_id: str = eqx.field(static=True)
    target_semantic_layout_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    topology_changed: bool = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: DistributedStatisticalLayout,
        target: DistributedStatisticalLayout,
        /,
    ):
        if not isinstance(source, DistributedStatisticalLayout) or not isinstance(
            target, DistributedStatisticalLayout
        ):
            raise TypeError("Restart relation requires distributed statistical layouts.")
        accepted = source.semantic_layout_id == target.semantic_layout_id
        changed = source.topology_id != target.topology_id
        self.source = source
        self.target = target
        self.source_semantic_layout_id = source.semantic_layout_id
        self.target_semantic_layout_id = target.semantic_layout_id
        self.source_topology_id = source.topology_id
        self.target_topology_id = target.topology_id
        self.topology_changed = changed
        self.accepted = accepted
        self.relation_id = canonical_fingerprint(
            {
                "kind": "distributed-statistical-restart-relation",
                "source_semantic": source.semantic_layout_id,
                "target_semantic": target.semantic_layout_id,
                "source_topology": source.topology_id,
                "target_topology": target.topology_id,
                "accepted": accepted,
            }
        )

    def require(self, /) -> None:
        if not self.accepted:
            raise ValueError(
                "Distributed restart changes global batch or covariance semantics."
            )

    def redistribute_batch(self, shards: Sequence[ArrayLike], /) -> tuple[Array, ...]:
        self.require()
        return self.target.batch.shard(self.source.batch.assemble(shards))

    def redistribute_covariance(
        self, shards: Sequence[ArrayLike], /
    ) -> tuple[Array, ...]:
        self.require()
        return self.target.covariance.shard(self.source.covariance.assemble(shards))


__all__ = [
    "CovarianceStorage",
    "DistributedBatchLayout",
    "DistributedCovarianceLayout",
    "DistributedRestartRelation",
    "DistributedShard",
    "DistributedStatisticalLayout",
]
