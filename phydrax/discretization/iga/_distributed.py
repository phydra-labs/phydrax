#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical-ID distributed views for isogeometric control variables.

The objects in this module describe communication; they deliberately do not own an
MPI communicator.  This keeps a one-rank execution a real partition (rather than a
``COMM_SELF`` special case) and makes the same schedule replayable by every backend.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _integer_vector(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be a one-dimensional integer array.")
    return array.astype(np.int64, copy=False)


class IsogeometricPartition(StrictModule, NonTrainableState):
    """One rank's owned and ghost control-variable replicas.

    ``canonical_ids`` identify variables independently of local ordering.  A ghost is
    simply a local replica whose owner is another rank.
    """

    canonical_ids: Array
    owner_ranks: Array
    owned_mask: Array
    ghost_mask: Array
    rank: int = eqx.field(static=True)
    rank_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        canonical_ids: ArrayLike,
        owner_ranks: ArrayLike,
        /,
        *,
        rank: int,
        rank_count: int,
    ):
        ids = _integer_vector(canonical_ids, "canonical_ids")
        owners = _integer_vector(owner_ranks, "owner_ranks")
        rank_ = int(rank)
        ranks = int(rank_count)
        if (
            ids.size == 0
            or owners.shape != ids.shape
            or np.any(ids < 0)
            or np.unique(ids).size != ids.size
            or ranks <= 0
            or rank_ < 0
            or rank_ >= ranks
            or np.any(owners < 0)
            or np.any(owners >= ranks)
        ):
            raise ValueError("IGA partition IDs, owners, or rank metadata are invalid.")
        owned = owners == rank_
        self.canonical_ids = jnp.asarray(ids)
        self.owner_ranks = jnp.asarray(owners, dtype=jnp.int32)
        self.owned_mask = jnp.asarray(owned)
        self.ghost_mask = jnp.asarray(~owned)
        self.rank = rank_
        self.rank_count = ranks
        self.partition_id = canonical_fingerprint(
            {
                "kind": "isogeometric-partition",
                "canonical_ids": array_tree_fingerprint(ids),
                "owner_ranks": array_tree_fingerprint(owners),
                "rank": rank_,
                "rank_count": ranks,
            }
        )

    @property
    def local_size(self) -> int:
        return int(self.canonical_ids.size)

    def owned(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim == 0 or value.shape[0] != self.local_size:
            raise ValueError("Partition values must have the local partition size.")
        mask = self.owned_mask.reshape(self.owned_mask.shape + (1,) * (value.ndim - 1))
        return jnp.where(mask, value, jnp.zeros((), dtype=value.dtype))


class IsogeometricHaloPlan(StrictModule, NonTrainableState):
    """Fixed canonical-ID routes for primal exchange and its exact transpose."""

    canonical_ids: Array
    source_indices: Array
    target_indices: Array
    source_ranks: Array
    target_ranks: Array
    source_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        canonical_ids: ArrayLike,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        /,
        *,
        source_size: int,
        target_size: int,
        source_ranks: ArrayLike | None = None,
        target_ranks: ArrayLike | None = None,
    ):
        ids = _integer_vector(canonical_ids, "canonical_ids")
        source = _integer_vector(source_indices, "source_indices")
        target = _integer_vector(target_indices, "target_indices")
        source_n, target_n = int(source_size), int(target_size)
        if (
            source.shape != ids.shape
            or target.shape != ids.shape
            or source_n <= 0
            or target_n <= 0
            or np.any(ids < 0)
            or np.unique(ids).size != ids.size
            or np.any(source < 0)
            or np.any(source >= source_n)
            or np.any(target < 0)
            or np.any(target >= target_n)
            or np.unique(target).size != target.size
        ):
            raise ValueError("IGA halo routes must be canonical and non-overlapping.")
        source_rank = (
            np.zeros(ids.shape, dtype=np.int32)
            if source_ranks is None
            else _integer_vector(source_ranks, "source_ranks")
        )
        target_rank = (
            np.zeros(ids.shape, dtype=np.int32)
            if target_ranks is None
            else _integer_vector(target_ranks, "target_ranks")
        )
        if (
            source_rank.shape != ids.shape
            or target_rank.shape != ids.shape
            or np.any(source_rank < 0)
            or np.any(target_rank < 0)
        ):
            raise ValueError("IGA halo route ranks are invalid.")
        self.canonical_ids = jnp.asarray(ids)
        self.source_indices = jnp.asarray(source, dtype=jnp.int32)
        self.target_indices = jnp.asarray(target, dtype=jnp.int32)
        self.source_ranks = jnp.asarray(source_rank, dtype=jnp.int32)
        self.target_ranks = jnp.asarray(target_rank, dtype=jnp.int32)
        self.source_size, self.target_size = source_n, target_n
        self.plan_id = canonical_fingerprint(
            {
                "kind": "isogeometric-halo-plan",
                "canonical_ids": array_tree_fingerprint(ids),
                "source_indices": array_tree_fingerprint(source),
                "target_indices": array_tree_fingerprint(target),
                "source_ranks": array_tree_fingerprint(source_rank),
                "target_ranks": array_tree_fingerprint(target_rank),
                "source_size": source_n,
                "target_size": target_n,
            }
        )

    def exchange(self, source_values: ArrayLike, target_values: ArrayLike, /) -> Array:
        """Copy owner values into ghost slots without requiring a communicator."""
        source = jnp.asarray(source_values)
        target = jnp.asarray(target_values)
        if (
            source.ndim == 0
            or target.ndim == 0
            or source.shape[0] != self.source_size
            or target.shape[0] != self.target_size
            or source.shape[1:] != target.shape[1:]
        ):
            raise ValueError("Halo exchange values have incompatible shapes.")
        return target.at[self.target_indices].set(source[self.source_indices])

    def transpose(self, target_cotangent: ArrayLike, /) -> Array:
        """Accumulate ghost adjoints back at their owner slots in route order."""
        cotangent = jnp.asarray(target_cotangent)
        if cotangent.ndim == 0 or cotangent.shape[0] != self.target_size:
            raise ValueError("Halo cotangent must have the target partition size.")
        result = jnp.zeros((self.source_size,) + cotangent.shape[1:], cotangent.dtype)
        # ``.add`` has deterministic route order because the schedule is canonical-ID sorted.
        order = jnp.argsort(self.canonical_ids, stable=True)
        return result.at[self.source_indices[order]].add(
            cotangent[self.target_indices[order]]
        )


class IsogeometricRepartitionRecord(StrictModule, NonTrainableState):
    """A deterministic canonical-ID migration between two partition epochs."""

    canonical_ids: Array
    previous_owners: Array
    next_owners: Array
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        canonical_ids: ArrayLike,
        previous_owners: ArrayLike,
        next_owners: ArrayLike,
        /,
    ):
        ids = _integer_vector(canonical_ids, "canonical_ids")
        before = _integer_vector(previous_owners, "previous_owners")
        after = _integer_vector(next_owners, "next_owners")
        if (
            ids.size == 0
            or before.shape != ids.shape
            or after.shape != ids.shape
            or np.any(ids < 0)
            or np.unique(ids).size != ids.size
            or np.any(before < 0)
            or np.any(after < 0)
        ):
            raise ValueError("IGA repartition record is invalid.")
        order = np.argsort(ids, kind="stable")
        self.canonical_ids = jnp.asarray(ids[order])
        self.previous_owners = jnp.asarray(before[order], dtype=jnp.int32)
        self.next_owners = jnp.asarray(after[order], dtype=jnp.int32)
        self.record_id = canonical_fingerprint(
            {
                "kind": "isogeometric-repartition-record",
                "canonical_ids": array_tree_fingerprint(ids[order]),
                "previous_owners": array_tree_fingerprint(before[order]),
                "next_owners": array_tree_fingerprint(after[order]),
            }
        )

    @property
    def moved_mask(self) -> Array:
        return self.previous_owners != self.next_owners


def deterministic_reduce(
    canonical_ids: ArrayLike,
    values: ArrayLike,
    /,
    *,
    reduction: str = "sum",
) -> tuple[Array, Array]:
    """Reduce replicas in canonical-ID order, independent of input ordering."""
    ids = _integer_vector(canonical_ids, "canonical_ids")
    value = jnp.asarray(values)
    if ids.size == 0 or value.ndim == 0 or value.shape[0] != ids.size:
        raise ValueError(
            "Reduction IDs and values must have matching non-empty leading axes."
        )
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'.")
    order = np.argsort(ids, kind="stable")
    ordered_ids = ids[order]
    starts = np.r_[True, ordered_ids[1:] != ordered_ids[:-1]]
    unique_ids = ordered_ids[starts]
    groups = np.cumsum(starts, dtype=np.int32) - 1
    ordered_value = value[jnp.asarray(order)]
    result = (
        jnp.zeros((unique_ids.size,) + value.shape[1:], value.dtype)
        .at[jnp.asarray(groups)]
        .add(ordered_value)
    )
    if reduction == "mean":
        counts = np.bincount(groups, minlength=unique_ids.size)
        result = result / jnp.asarray(counts).reshape(
            (unique_ids.size,) + (1,) * (value.ndim - 1)
        )
    return jnp.asarray(unique_ids), result


__all__ = [
    "IsogeometricHaloPlan",
    "IsogeometricPartition",
    "IsogeometricRepartitionRecord",
    "deterministic_reduce",
]
