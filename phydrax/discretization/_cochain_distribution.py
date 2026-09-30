#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cell_de_rham import AbstractCellDeRhamComplex


@final
class CochainPartition(StrictModule, NonTrainableState):
    """Immutable ownership of the coordinates of a cell de Rham complex."""

    owners: tuple[Array, ...]
    partition_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(self, owners: Sequence[ArrayLike], partition_count: int, /) -> None:
        count = int(partition_count)
        host = tuple(np.asarray(value) for value in owners)
        if count <= 0 or not host:
            raise ValueError("Cochain partition inputs are invalid.")
        if any(value.ndim != 1 for value in host):
            raise ValueError("Cochain owner arrays must be vectors.")
        if any(not np.issubdtype(value.dtype, np.integer) for value in host):
            raise TypeError("Cochain owner indices must be integers.")
        if any(np.any((value < 0) | (value >= count)) for value in host):
            raise ValueError("Cochain owner indices are outside partition_count.")
        arrays = tuple(jnp.asarray(value, dtype=jnp.int32) for value in host)
        identifier = canonical_fingerprint(
            {
                "kind": "cochain-partition",
                "owners": [array_tree_fingerprint(value) for value in arrays],
                "partition_count": count,
            }
        )
        self.owners = arrays
        self.partition_count = count
        self.partition_id = identifier


@final
class CochainHaloExchange(StrictModule, NonTrainableState):
    """Sparse interpartition payload routes induced by an oriented incidence."""

    source_degree: int = eqx.field(static=True)
    source_size: int = eqx.field(static=True)
    source_indices: Array
    target_partitions: Array
    exchange_id: str = eqx.field(static=True)

    def __init__(
        self,
        cochain: AbstractCellDeRhamComplex,
        partition: CochainPartition,
        degree: int,
        /,
    ) -> None:
        if not isinstance(cochain, AbstractCellDeRhamComplex):
            raise TypeError("cochain must be an AbstractCellDeRhamComplex.")
        if not isinstance(partition, CochainPartition):
            raise TypeError("partition must be a CochainPartition.")
        degree_ = int(degree)
        counts = tuple(entity.count for entity in cochain.topology.entity_sets)
        if len(partition.owners) != len(counts) or any(
            owner.shape != (size,)
            for owner, size in zip(partition.owners, counts, strict=True)
        ):
            raise ValueError(
                "Partition must cover every cell coordinate of every degree."
            )
        if degree_ < 0 or degree_ >= cochain.dimension:
            raise ValueError("Halo exchange degree must have a following incidence.")
        incidence = cochain.topology.incidences[degree_]
        valid = np.asarray(incidence.relation.valid, dtype=np.bool_)
        sources = np.asarray(incidence.relation.source_indices)[valid]
        targets = np.asarray(incidence.relation.target_indices)[valid]
        source_owner = np.asarray(partition.owners[degree_])[sources]
        target_owner = np.asarray(partition.owners[degree_ + 1])[targets]
        remote = source_owner != target_owner
        identifier = canonical_fingerprint(
            {
                "kind": "cochain-halo-exchange",
                "complex": cochain.realization_id,
                "partition": partition.partition_id,
                "degree": degree_,
                "source": array_tree_fingerprint(sources[remote]),
                "target_partition": array_tree_fingerprint(target_owner[remote]),
            }
        )
        self.source_degree = degree_
        self.source_size = counts[degree_]
        self.source_indices = jnp.asarray(sources[remote], dtype=jnp.int32)
        self.target_partitions = jnp.asarray(target_owner[remote], dtype=jnp.int32)
        self.exchange_id = identifier

    def payload(self, values: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(values)
        if value.ndim < 1 or value.shape[0] != self.source_size:
            raise ValueError("Halo payload must have the source cell coordinate extent.")
        return self.target_partitions, value[self.source_indices]


__all__ = ["CochainHaloExchange", "CochainPartition"]
