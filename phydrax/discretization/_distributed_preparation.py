#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..typing import checked, ConvertibleToArray
from ._cell_mesh import CellMesh, CellMeshStorage


def owner_local_host_array(
    value: ConvertibleToArray | None, /, *, dtype: DTypeLike | None = None
) -> np.ndarray:
    """Snapshot addressable preparation data using NumPy scalar conversion.

    ``None`` follows NumPy's dtype-dependent conversion; caller admission owns
    the resulting rank and numerical-kind contract.
    """
    match value:
        case Array():
            if not value.is_fully_addressable:
                raise ValueError(
                    "owner_local_host_array.value must be process-local addressable data; "
                    "got a nonaddressable global array"
                )
            return np.asarray(jax.device_get(value), dtype=dtype)
        case _:
            return np.asarray(value, dtype=dtype)


def replicated_host_array(value: Array, /) -> np.ndarray:
    """Snapshot a replicated collective result from one addressable shard."""
    return np.asarray(jax.device_get(value.addressable_shards[0].data))


@checked
def require_owner_local_storage(mesh: CellMesh, /) -> CellMeshStorage:
    """Return storage after admitting one owner-local mesh closure."""
    storage = mesh.storage
    if storage is None:
        raise ValueError(
            "owner_local.mesh.storage must describe an owner-local closure; got None"
        )
    return storage


class FieldQuerySources(StrictModule):
    storages: tuple[CellMeshStorage, ...]
    values: tuple[Array, ...]
    query_ids: tuple[np.ndarray, ...]
    owners: tuple[np.ndarray, ...]
    assigned: tuple[jax.Device, ...]
    ranks: tuple[int, ...]
    degree: int
    bound: int
    axis: str


def prepare_field_query_sources(
    source_meshes: Sequence[CellMesh],
    source_values: Sequence[ArrayLike],
    query_global_ids: Sequence[ArrayLike],
    query_owners: Sequence[ArrayLike],
    entity_degree: int,
    source_topology_id: str,
    devices: Sequence[jax.Device],
    axis_name: str,
    message_capacity: int,
    /,
) -> FieldQuerySources:
    meshes = tuple(source_meshes)
    values = tuple(jnp.asarray(value) for value in source_values)
    if any(not value.is_fully_addressable for value in values) or any(
        isinstance(value, Array) and not value.is_fully_addressable
        for group in (query_global_ids, query_owners)
        for value in group
    ):
        raise ValueError(
            "field_queries.sources must contain process-local addressable query IDs, owners, "
            "and values; got a nonaddressable global array"
        )
    query_ids = tuple(
        owner_local_host_array(identifiers) for identifiers in query_global_ids
    )
    owners = tuple(owner_local_host_array(placement) for placement in query_owners)
    assigned = tuple(devices)
    degree = int(entity_degree)
    bound = int(message_capacity)
    axis = str(axis_name).strip()
    if (
        not meshes
        or bound < 1
        or not axis
        or len(values) != len(meshes)
        or len(query_ids) != len(meshes)
        or len(owners) != len(meshes)
    ):
        raise ValueError(
            "field_queries.sources must align with local values, queries, and a positive message bound; got incompatible declarations"
        )
    storages = tuple(require_owner_local_storage(mesh) for mesh in meshes)
    if any(
        not 0 <= degree < len(storage.global_entity_counts)
        or storage.partition_count != len(assigned)
        or storage.logical_topology_id != source_topology_id
        for storage in storages
    ):
        raise ValueError(
            "field_queries.sources must bind the accepted predecessor topology and placement; got incompatible storage"
        )
    ranks = tuple(storage.partition_index for storage in storages)
    values = tuple(
        jax.device_put(value, assigned[rank])
        for rank, value in zip(ranks, values, strict=True)
    )
    return FieldQuerySources(
        storages, values, query_ids, owners, assigned, ranks, degree, bound, axis
    )


class FieldQueryRows(StrictModule):
    lookup: dict[int, int]
    phase_rows: tuple[np.ndarray, ...]
    own: np.ndarray
    source_rows: np.ndarray
    maximum: Array
    metadata: Array
    missing: bool


@checked
def prepare_field_query_rows(
    storage: CellMeshStorage,
    field: Array,
    ids: np.ndarray,
    placement: np.ndarray,
    degree: int,
    component_shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
    parts: int,
    source_topology_id: str,
    bound: int,
    /,
) -> FieldQueryRows:
    source_ids = owner_local_host_array(storage.entity_global_ids[degree], dtype=np.int64)
    owned = owner_local_host_array(storage.entity_owned[degree], dtype=np.bool_)
    if (
        not field.is_fully_addressable
        or field.shape != (source_ids.size,) + component_shape
        or field.dtype != dtype
        or ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids > np.iinfo(np.int64).max)
        or placement.shape != ids.shape
        or placement.dtype.kind not in "iu"
        or np.any(placement < 0)
        or np.any(placement >= parts)
    ):
        raise ValueError(
            "field_queries.query_global_ids must align with field components and valid query owners; got incompatible shapes, dtypes, or IDs"
        )
    lookup = {
        int(identifier): row for row, identifier in enumerate(source_ids) if owned[row]
    }
    phase_rows = tuple(
        np.flatnonzero(placement == (storage.partition_index + offset) % parts).astype(
            np.int32
        )
        for offset in range(1, parts)
    )
    own = np.flatnonzero(placement == storage.partition_index).astype(np.int32)
    source_rows = np.asarray(
        [lookup.get(int(ids[row]), -1) for row in own], dtype=np.int32
    )
    maximum = jnp.asarray(
        max((row.size for row in phase_rows), default=0), dtype=jnp.int32
    )
    identity = canonical_fingerprint(
        {
            "kind": "accepted-owner-local-field-query",
            "topology": source_topology_id,
            "degree": degree,
            "count": storage.global_entity_counts[degree],
            "evidence": storage.evidence_id,
            "parts": parts,
            "component_shape": component_shape,
            "dtype": str(dtype),
            "message_bound": bound,
        }
    )
    metadata = jnp.asarray(
        [int(identity[offset : offset + 7], 16) for offset in range(0, 64, 7)],
        dtype=jnp.int32,
    )
    return FieldQueryRows(
        lookup,
        phase_rows,
        own,
        source_rows,
        maximum,
        metadata,
        bool(np.any(source_rows < 0)),
    )
