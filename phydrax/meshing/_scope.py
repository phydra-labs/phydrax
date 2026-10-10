#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from math import prod
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from numpy.typing import ArrayLike, NDArray

from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, EntitySelection, EntitySet
from ..typing import checked


if TYPE_CHECKING:
    from ._initial_certification import InitialCollectiveMeshEvidence
    from ._publication_lowering import PublicationProjection
    from ._result import CollectiveMeshEvidence


class MeshingEntityKind(StrEnum):
    GEOMETRY = "geometry"
    MESH = "mesh"
    PART = "part"
    LABEL = "label"
    ZONE = "zone"


@final
class MeshScopeProjection(StrictModule, NonTrainableState):
    """An all-owner scientific scope projection consumed without local collectives."""

    publication: PublicationProjection
    global_ids: Array
    members: Array
    exclusive_scopes: tuple[MeshScopeProjection, ...]
    packet_ids: Array
    packet_owners: Array
    packet_valid: Array
    packet_membership: Array
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    membership_name: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    logical_entity_set_id: str = eqx.field(static=True)
    scope_id: str = eqx.field(static=True)

    def __init__(
        self,
        evidence: CollectiveMeshEvidence | InitialCollectiveMeshEvidence,
        publication: PublicationProjection,
        source_revision: str,
        dimension: int,
        membership_name: str,
        /,
        *,
        exclusive_scopes: tuple[MeshScopeProjection, ...] = (),
    ) -> None:
        from ._initial_certification import InitialCollectiveMeshEvidence
        from ._publication_lowering import PublicationProjection
        from ._result import CollectiveMeshEvidence

        if not isinstance(
            evidence, (CollectiveMeshEvidence, InitialCollectiveMeshEvidence)
        ):
            raise TypeError(
                "Scientific scope projection requires its actual accepted theorem."
            )
        if not isinstance(publication, PublicationProjection):
            raise TypeError(
                "Scientific scope projection requires canonical publication receipts."
            )
        evidence.require_passed()
        publication.require_source(evidence.logical_arrays)
        if (
            publication.topology_id != evidence.topology_id
            or publication.organization_id != evidence.global_organization_id
            or publication.partition_count != evidence.partition_count
            or dimension < 0
            or dimension >= len(evidence.global_entity_counts)
            or not source_revision.strip()
        ):
            raise ValueError(
                "Scientific scope projection has a different accepted source or placement."
            )
        arrays = dict(evidence.logical_arrays)
        member = arrays[membership_name]
        count = evidence.global_entity_counts[dimension]
        ids = publication.entity_ids[dimension]
        if member.dtype != jnp.bool_ or member.shape != ids.shape:
            raise ValueError(
                "Scientific scope membership has an incompatible accepted entity axis."
            )
        if bool(jax.device_get(jnp.any(member[count:]))):
            raise ValueError("Scientific scope membership selects padded rows.")
        packets = dict(publication.projected_arrays)
        packet_ids = packets[f"entity/{dimension}/ids"]
        packet_owners = packets[f"entity/{dimension}/owners"]
        packet_valid = packets[f"entity/{dimension}/valid"]
        packet_membership = packets[membership_name]
        if (
            packet_ids.ndim != 2
            or packet_ids.shape[0] != publication.partition_count
            or packet_ids.dtype != jnp.int64
            or packet_owners.dtype != jnp.int32
            or packet_valid.dtype != jnp.bool_
            or packet_membership.dtype != jnp.bool_
            or any(
                value.shape != packet_ids.shape
                for value in (packet_owners, packet_valid, packet_membership)
            )
        ):
            raise ValueError(
                "Scientific scope receipts require exact fixed-capacity part-leading arrays."
            )
        source_order = jnp.argsort(ids[:count], stable=True)
        ordered_ids = ids[:count][source_order]
        rows = jnp.minimum(jnp.searchsorted(ordered_ids, packet_ids), count - 1)
        source_rows = source_order[rows]
        accepted = ~packet_valid | (
            (ordered_ids[rows] == packet_ids)
            & (publication.entity_owners[dimension][source_rows] == packet_owners)
            & (member[source_rows] == packet_membership)
        )
        accepted &= packet_valid | ~packet_membership
        if not bool(jax.device_get(jnp.all(accepted))):
            raise ValueError(
                "Scientific scope receipt differs from actual accepted membership or ownership."
            )
        global_ids = jnp.sort(ids[:count], stable=True)
        members = jnp.sort(_selected_ids(ids[:count], member[:count]), stable=True)
        inventory, identity = _mesh_scope_inventory_ids(
            evidence.mesh_id,
            source_revision,
            evidence.topology_id,
            dimension,
            global_ids,
            members,
        )
        for other in exclusive_scopes:
            if (
                not isinstance(other, MeshScopeProjection)
                or other.publication is not publication
                or other.source_id != evidence.mesh_id
                or other.source_revision != source_revision
                or other.dimension != dimension
                or other.evidence_id != evidence.evidence_id
            ):
                raise ValueError(
                    "Exclusive scientific scopes require one actual accepted entity inventory."
                )
            if bool(jax.device_get(jnp.any(_contains_ids(other.members, members)))):
                raise ValueError(
                    "Exclusive scientific scopes overlap in the accepted global inventory."
                )
        self.publication = publication
        self.global_ids = global_ids
        self.members = members
        self.exclusive_scopes = exclusive_scopes
        self.packet_ids = packet_ids
        self.packet_owners = packet_owners
        self.packet_valid = packet_valid
        self.packet_membership = packet_membership
        self.source_id = evidence.mesh_id
        self.source_revision = source_revision
        self.dimension = dimension
        self.membership_name = membership_name
        self.evidence_id = evidence.evidence_id
        self.logical_entity_set_id = inventory
        self.scope_id = identity

    def local_mask(self, mesh: CellMesh, /) -> NDArray[np.bool_]:
        """Check exact received entity IDs, owners and scope membership locally."""
        storage = mesh.storage
        if (
            storage is None
            or mesh.mesh_id != self.source_id
            or mesh.numeric_version != self.source_revision
            or mesh.topology_id != self.publication.topology_id
            or storage.evidence_id != self.evidence_id
            or storage.partition_count != self.publication.partition_count
        ):
            raise ValueError(
                "Scientific scope receipt does not belong to this owner-local mesh."
            )
        banks = dict(self.publication.projected_arrays)
        if (
            banks[f"entity/{self.dimension}/ids"] is not self.packet_ids
            or banks[f"entity/{self.dimension}/owners"] is not self.packet_owners
            or banks[f"entity/{self.dimension}/valid"] is not self.packet_valid
            or banks[self.membership_name] is not self.packet_membership
        ):
            raise ValueError(
                "Scientific scope receipt changed after its actual numerical proof."
            )
        packets = dict(self.publication.addressable_arrays(storage.partition_index))
        degree = self.dimension
        valid = np.asarray(packets[f"entity/{degree}/valid"], dtype=np.bool_)
        ids = np.asarray(packets[f"entity/{degree}/ids"], dtype=np.int64)[valid]
        owners = np.asarray(packets[f"entity/{degree}/owners"], dtype=np.int32)[valid]
        membership = np.asarray(packets[self.membership_name], dtype=np.bool_)[valid]
        order = np.argsort(ids, stable=True)
        ids, owners, membership = ids[order], owners[order], membership[order]
        entities = mesh.entity_set(degree)
        local_ids = np.asarray(entities.entity_ids, dtype=np.int64)
        rows = np.searchsorted(ids, local_ids)
        if (
            ids.size != local_ids.size
            or np.any(rows >= ids.size)
            or not np.array_equal(ids[rows], local_ids)
            or not np.array_equal(
                owners[rows], np.asarray(storage.entity_owner[degree], dtype=np.int32)
            )
        ):
            raise ValueError(
                "Scientific scope receipt differs from the exact local entity or owner table."
            )
        return membership[rows] & np.asarray(entities.active_mask, dtype=np.bool_)


class MeshingScope(StrictModule, NonTrainableState):
    """Exact entity scope bound to one immutable source revision."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    entity_kind: MeshingEntityKind = eqx.field(static=True)
    entity_dimension: int = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    logical_entity_set_id: str = eqx.field(static=True)
    global_entity_ids: Array
    entity_ids: Array
    local_coverage_id: str | None = eqx.field(static=True)
    local_partition_index: int | None = eqx.field(static=True)
    local_partition_count: int | None = eqx.field(static=True)
    entity_owner: Array
    _local_entity_universe: Array | None
    _local_owner_universe: Array | None
    _global_entity_universe: Array | None
    _scope_projection: MeshScopeProjection | None
    scope_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        entity_kind: MeshingEntityKind,
        entity_dimension: int,
        entity_set_id: str,
        entity_ids: ArrayLike,
        /,
        *,
        local_mesh: CellMesh | None = None,
        _local_basis: MeshingScope | None = None,
        _projection: MeshScopeProjection | None = None,
    ) -> None:
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        entity_set = str(entity_set_id).strip()
        if not source or not revision or not entity_set:
            raise ValueError("Meshing scope identities must be non-empty.")
        if not isinstance(entity_kind, MeshingEntityKind):
            raise TypeError("entity_kind must be MeshingEntityKind.")
        dimension = int(entity_dimension)
        if dimension < 0:
            raise ValueError("entity_dimension must be non-negative.")
        if _projection is None:
            identifiers = jnp.asarray(entity_ids)
            if identifiers.ndim != 1 or not jnp.issubdtype(
                identifiers.dtype, jnp.integer
            ):
                raise TypeError("Meshing scope entity_ids must be one integer vector.")
            identifiers = jnp.sort(identifiers.astype(jnp.int64), stable=True)
            if identifiers.size == 0:
                raise ValueError("Meshing scopes must contain at least one entity.")
            invalid = jnp.any(identifiers < 0) | jnp.any(
                identifiers[1:] == identifiers[:-1]
            )
            if bool(jax.device_get(invalid)):
                raise ValueError(
                    "Meshing scope entity_ids must be unique and non-negative."
                )
        else:
            if (
                not isinstance(_projection, MeshScopeProjection)
                or entity_ids is not _projection.members
            ):
                raise ValueError(
                    "A prepared scope must consume its exact globally accepted member bank."
                )
            if (
                local_mesh is None
                or _local_basis is not None
                or (source, revision, dimension, entity_kind)
                != (
                    _projection.source_id,
                    _projection.source_revision,
                    _projection.dimension,
                    MeshingEntityKind.MESH,
                )
            ):
                raise ValueError(
                    "A prepared scope requires its exact owner-local mesh binding."
                )
            identifiers = _projection.members
        local_ids = identifiers
        local_coverage_id = None
        partition_index = None
        partition_count = None
        owners = jnp.zeros(identifiers.shape, dtype=jnp.int32)
        local_universe = None
        owner_universe = None
        global_universe = None
        logical_entity_set = entity_set
        if local_mesh is not None and _local_basis is not None:
            raise ValueError("Local scope coverage must have exactly one owner.")
        if local_mesh is not None:
            if not isinstance(local_mesh, CellMesh):
                raise TypeError("local_mesh must be CellMesh or None.")
            if (
                entity_kind is not MeshingEntityKind.MESH
                or source != local_mesh.mesh_id
                or revision != local_mesh.numeric_version
                or dimension > local_mesh.topological_dimension
                or entity_set != local_mesh.entity_set(dimension).entity_set_id
            ):
                raise ValueError("A local scope view requires the exact mesh binding.")
            entities = local_mesh.entity_set(dimension)
            local_universe = entities.entity_ids
            global_universe = (
                _mesh_global_ids(local_mesh, dimension)
                if _projection is None
                else _projection.global_ids
            )
            mask = (
                _local_logical_lookup(identifiers, None, entities.entity_ids)[1]
                if _projection is None
                else _projection.local_mask(local_mesh)
            )
            mask = mask & np.asarray(entities.active_mask, dtype=np.bool_)
            present = jnp.asarray(mask, dtype=jnp.bool_)
            local_ids = jnp.asarray(
                np.asarray(entities.entity_ids)[mask], dtype=jnp.int64
            )
            order = jnp.argsort(local_ids, stable=True)
            local_ids = local_ids[order]
            if local_mesh.storage is None:
                complete = jnp.sum(present, dtype=jnp.int64) == identifiers.size
                owners = jnp.zeros(local_ids.shape, dtype=jnp.int32)
                owner_universe = jnp.zeros(local_universe.shape, dtype=jnp.int32)
            else:
                storage = local_mesh.storage
                complete = (
                    jnp.all(_contains_ids(global_universe, identifiers))
                    if _projection is None
                    else jnp.asarray(True, dtype=jnp.bool_)
                )
                owner_universe = storage.entity_owner[dimension]
                owners = owner_universe[mask][order]
                local_coverage_id = storage.evidence_id
                partition_index = storage.partition_index
                partition_count = storage.partition_count
            if not bool(jax.device_get(complete)):
                raise ValueError(
                    "Meshing scope contains undeclared global mesh entity IDs."
                )
            logical_entity_set = (
                _logical_mesh_entity_set_id(
                    local_mesh.topology_id, dimension, global_universe
                )
                if _projection is None
                else _projection.logical_entity_set_id
            )
        if _local_basis is not None:
            if not isinstance(_local_basis, MeshingScope):
                raise TypeError("_local_basis must be MeshingScope.")
            basis = _local_basis
            if (
                (source, revision, entity_kind, dimension, entity_set)
                != (
                    basis.source_id,
                    basis.source_revision,
                    basis.entity_kind,
                    basis.entity_dimension,
                    basis.entity_set_id,
                )
                or basis._local_entity_universe is None
                or basis._local_owner_universe is None
                or basis._global_entity_universe is None
            ):
                raise ValueError(
                    "Scope set operations require the exact established local coverage."
                )
            local_universe = basis._local_entity_universe
            owner_universe = basis._local_owner_universe
            global_universe = basis._global_entity_universe
            if not bool(
                jax.device_get(jnp.all(_contains_ids(global_universe, identifiers)))
            ):
                raise ValueError(
                    "Meshing scope contains undeclared global mesh entity IDs."
                )
            mask = _local_logical_lookup(identifiers, None, local_universe)[1]
            local_ids = local_universe[mask]
            order = jnp.argsort(local_ids, stable=True)
            local_ids, owners = local_ids[order], owner_universe[mask][order]
            local_coverage_id = basis.local_coverage_id
            partition_index, partition_count = (
                basis.local_partition_index,
                basis.local_partition_count,
            )
            logical_entity_set = basis.logical_entity_set_id
        self.source_id = source
        self.source_revision = revision
        self.entity_kind = entity_kind
        self.entity_dimension = dimension
        self.entity_set_id = entity_set
        self.logical_entity_set_id = logical_entity_set
        self.global_entity_ids = identifiers
        self.entity_ids = local_ids
        self.local_coverage_id = local_coverage_id
        self.local_partition_index = partition_index
        self.local_partition_count = partition_count
        self.entity_owner = owners
        self._local_entity_universe = local_universe
        self._local_owner_universe = owner_universe
        self._global_entity_universe = global_universe
        self._scope_projection = _projection
        self.scope_id = (
            _scope_inventory_id(
                source, revision, entity_kind, dimension, logical_entity_set, identifiers
            )
            if _projection is None
            else _projection.scope_id
        )

    def lower(self, mesh: CellMesh, /) -> MeshingScope:
        """Certify the exact locally present view without changing logical membership."""
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        entity_set = (
            mesh.entity_set(self.entity_dimension).entity_set_id
            if self._local_entity_universe is not None
            else self.entity_set_id
        )
        lowered = MeshingScope(
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            entity_set,
            self.global_entity_ids,
            local_mesh=mesh,
        )
        if (
            self._local_entity_universe is not None
            and lowered.logical_entity_set_id != self.logical_entity_set_id
        ):
            raise ValueError(
                "Scope rehosting requires the same complete logical entity inventory."
            )
        return lowered

    @classmethod
    @checked
    def from_selection(
        cls,
        source_id: str,
        source_revision: str,
        entities: EntitySet,
        selection: EntitySelection,
        /,
    ) -> MeshingScope:
        """Convert a positional selection using its exact owning entity set."""
        if selection.entity_set_id != entities.entity_set_id:
            raise ValueError("Selection must belong to the supplied entity set.")
        mask = np.asarray(selection.mask, dtype=np.bool_)
        if mask.shape != (entities.count,) or not np.array_equal(
            selection.active_mask, entities.active_mask
        ):
            raise ValueError(
                "Selection must match the entity set's capacity and active mask."
            )
        return cls(
            source_id,
            source_revision,
            MeshingEntityKind.MESH,
            entities.intrinsic_dimension,
            entities.entity_set_id,
            np.asarray(entities.entity_ids)[mask],
        )

    @checked
    def _check_compatible(self, other: MeshingScope, /) -> None:
        binding = (
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
        )
        other_binding = (
            other.source_id,
            other.source_revision,
            other.entity_kind,
            other.entity_dimension,
            other.entity_set_id,
        )
        if binding != other_binding:
            raise ValueError("Meshing scope set operations require one exact binding.")
        if (
            self._local_entity_universe is not None
            and other._local_entity_universe is not None
            and (
                self.local_coverage_id,
                self.local_partition_index,
                self.local_partition_count,
            )
            != (
                other.local_coverage_id,
                other.local_partition_index,
                other.local_partition_count,
            )
        ):
            raise ValueError(
                "Scope set operations require one exact local coverage placement."
            )

    def union(self, other: MeshingScope, /) -> MeshingScope:
        self._check_compatible(other)
        return MeshingScope(
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
            _unique_ids(
                jnp.concatenate((self.global_entity_ids, other.global_entity_ids))
            ),
            _local_basis=self
            if self._local_entity_universe is not None
            else (other if other._local_entity_universe is not None else None),
        )

    def intersection(self, other: MeshingScope, /) -> MeshingScope:
        self._check_compatible(other)
        values = _selected_ids(
            self.global_entity_ids,
            _contains_ids(other.global_entity_ids, self.global_entity_ids),
        )
        if values.size == 0:
            raise ValueError("Meshing scope intersection is empty.")
        return MeshingScope(
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
            values,
            _local_basis=self
            if self._local_entity_universe is not None
            else (other if other._local_entity_universe is not None else None),
        )

    def difference(self, other: MeshingScope, /) -> MeshingScope:
        self._check_compatible(other)
        values = _selected_ids(
            self.global_entity_ids,
            ~_contains_ids(other.global_entity_ids, self.global_entity_ids),
        )
        if values.size == 0:
            raise ValueError("Meshing scope difference is empty.")
        return MeshingScope(
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
            values,
            _local_basis=self
            if self._local_entity_universe is not None
            else (other if other._local_entity_universe is not None else None),
        )

    def __or__(self, other: MeshingScope) -> MeshingScope:
        return self.union(other)

    def __and__(self, other: MeshingScope) -> MeshingScope:
        return self.intersection(other)

    def __sub__(self, other: MeshingScope) -> MeshingScope:
        return self.difference(other)


def _selected_ids(identifiers: Array, mask: Array, /) -> Array:
    if (
        mask.dtype == jnp.bool_
        and not isinstance(identifiers, Tracer)
        and not isinstance(mask, Tracer)
        and identifiers.is_fully_addressable
        and mask.is_fully_addressable
        and isinstance(identifiers.sharding, jax.sharding.SingleDeviceSharding)
        and isinstance(mask.sharding, jax.sharding.SingleDeviceSharding)
    ):
        values, selected = jax.device_get((identifiers, mask))
        return jnp.asarray(
            np.asarray(values)[np.nonzero(np.asarray(selected))[0]],
            dtype=identifiers.dtype,
        )
    count = int(jax.device_get(jnp.sum(mask, dtype=jnp.int64)))
    return identifiers[jnp.nonzero(mask, size=count)[0]]


def _unique_ids(identifiers: Array, /) -> Array:
    ordered = jnp.sort(identifiers, stable=True)
    fresh = jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), ordered[1:] != ordered[:-1])
    )
    return _selected_ids(ordered, fresh)


def _contains_ids(sorted_ids: Array, requested: Array, /) -> Array:
    positions = jnp.minimum(jnp.searchsorted(sorted_ids, requested), sorted_ids.size - 1)
    return sorted_ids[positions] == requested


def _logical_mesh_entity_set_id(
    topology_id: str, degree: int, identifiers: Array, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "logical-mesh-entity-inventory",
            "topology": topology_id,
            "dimension": degree,
            "entity_ids": logical_array_value_collection_digest(
                {"entity_ids": identifiers}
            ),
        }
    )


def _scope_inventory_id(
    source_id: str,
    source_revision: str,
    kind: MeshingEntityKind,
    degree: int,
    entity_set_id: str,
    members: Array,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "meshing-scope",
            "source_id": source_id,
            "source_revision": source_revision,
            "entity_kind": kind.value,
            "entity_dimension": degree,
            "entity_set_id": entity_set_id,
            "entity_ids": logical_array_value_collection_digest({"entity_ids": members}),
        }
    )


def _mesh_scope_inventory_ids(
    source_id: str,
    source_revision: str,
    topology_id: str,
    degree: int,
    global_ids: Array,
    members: Array,
    /,
) -> tuple[str, str]:
    """Hash canonically sorted complete inventory and membership in the global phase."""
    if (
        global_ids.ndim != 1
        or members.ndim != 1
        or global_ids.dtype != jnp.int64
        or members.dtype != jnp.int64
    ):
        raise ValueError(
            "Canonical mesh inventory hashes require exact int64 ID vectors."
        )
    if global_ids.size == 0 or members.size == 0:
        raise ValueError(
            "Logical mesh scopes require a nonempty complete inventory and membership."
        )
    inventory = _logical_mesh_entity_set_id(topology_id, degree, global_ids)
    return inventory, _scope_inventory_id(
        source_id, source_revision, MeshingEntityKind.MESH, degree, inventory, members
    )


def _logical_query_positions(
    identifiers: Array, queries: Array, /
) -> tuple[Array, Array]:
    if identifiers.ndim == 1:
        positions = jnp.minimum(
            jnp.searchsorted(identifiers, queries), identifiers.shape[0] - 1
        )
        return positions, identifiers[positions] == queries
    from ._device_adaptation import _key_positions

    positions = _key_positions(identifiers, queries)
    return jnp.maximum(positions, 0), positions >= 0


def _logical_query_values(
    identifiers: Array, values: Array, queries: Array, /
) -> tuple[Array, Array]:
    positions, valid = _logical_query_positions(identifiers, queries)
    return values[positions], valid


def _logical_query_membership(identifiers: Array, queries: Array, /) -> Array:
    return _logical_query_positions(identifiers, queries)[1]


_compiled_logical_query_values = jax.jit(_logical_query_values)
_compiled_logical_membership = jax.jit(_logical_query_membership)


def _local_logical_lookup(
    identifiers: Array,
    values: Array | None,
    queries: Array | np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Route bounded semantic-ID worksets; never materialize a global shard bank."""
    if not isinstance(queries, (Array, np.ndarray)):
        raise TypeError("Local semantic queries must be canonical JAX or NumPy arrays.")
    if isinstance(queries, Array) and not queries.is_fully_addressable:
        raise ValueError("Logical inventory queries must be locally addressable.")
    queries = jnp.asarray(queries)
    if (
        identifiers.ndim not in (1, 2)
        or identifiers.shape[0] == 0
        or queries.ndim != identifiers.ndim
        or queries.shape[1:] != identifiers.shape[1:]
        or identifiers.dtype != jnp.int64
        or queries.dtype != jnp.int64
    ):
        raise ValueError(
            "Logical inventory queries require exact locally addressable int64 scalar or fixed-width semantic IDs."
        )
    if values is not None and (
        values.ndim == 0 or values.shape[0] != identifiers.shape[0]
    ):
        raise ValueError("Logical inventory values must match their exact ID axis.")
    if identifiers.is_fully_addressable and (
        values is None or values.is_fully_addressable
    ):
        if values is None:
            valid = np.asarray(
                jax.device_get(_logical_query_membership(identifiers, queries)),
                dtype=np.bool_,
            )
            return valid, valid
        selected, valid = _logical_query_values(identifiers, values, queries)
        return np.asarray(jax.device_get(selected)), np.asarray(
            jax.device_get(valid), dtype=np.bool_
        )
    from jax.experimental.multihost_utils import broadcast_one_to_all
    from jax.sharding import NamedSharding, PartitionSpec

    if not isinstance(identifiers.sharding, NamedSharding):
        raise ValueError(
            "Distributed inventory queries require the accepted named-device mesh."
        )
    replicated = NamedSharding(identifiers.sharding.mesh, PartitionSpec())
    host_queries = np.asarray(queries, dtype=np.int64)
    trailing = () if values is None else values.shape[1:]
    dtype = np.dtype(np.bool_) if values is None else values.dtype
    row_bytes = (
        identifiers.dtype.itemsize * prod(identifiers.shape[1:])
        + prod(trailing) * dtype.itemsize
        + np.dtype(np.bool_).itemsize
    )
    capacity = (1 << 20) // row_bytes
    if capacity == 0:
        raise ValueError(
            "A scientific inventory row exceeds the bounded semantic packet capacity."
        )
    selected_host = np.empty((host_queries.shape[0], *trailing), dtype=dtype)
    valid_host = np.empty((host_queries.shape[0],), dtype=np.bool_)
    for process in range(jax.process_count()):
        source = process == jax.process_index()
        count = int(
            np.asarray(
                broadcast_one_to_all(
                    np.asarray(host_queries.shape[0] if source else 0, dtype=np.int64),
                    is_source=source,
                )
            )
        )
        for first in range(0, count, capacity):
            stop = min(first + capacity, count)
            packet = np.zeros((capacity, *identifiers.shape[1:]), dtype=np.int64)
            if source:
                packet[: stop - first] = host_queries[first:stop]
            packet = broadcast_one_to_all(packet, is_source=source)
            logical_queries = jax.device_put(packet, replicated)
            if values is None:
                valid = _compiled_logical_membership(identifiers, logical_queries)
                valid = jax.device_put(valid, replicated)
                answer = valid
            else:
                answer, valid = _compiled_logical_query_values(
                    identifiers, values, logical_queries
                )
                answer, valid = (
                    jax.device_put(answer, replicated),
                    jax.device_put(valid, replicated),
                )
            if source:
                selected_host[first:stop] = np.asarray(answer.addressable_shards[0].data)[
                    : stop - first
                ]
                valid_host[first:stop] = np.asarray(
                    valid.addressable_shards[0].data, dtype=np.bool_
                )[: stop - first]
    return selected_host, valid_host


def _mesh_global_ids(mesh: CellMesh, dimension: int, /) -> Array:
    if mesh.storage is None:
        return jnp.sort(mesh.entity_set(dimension).entity_ids)
    name = (
        "vertex_global_ids"
        if dimension == 0
        else "cell_global_ids"
        if dimension == mesh.topological_dimension
        else f"entity_global_ids_{dimension}"
    )
    return jnp.sort(
        dict(mesh.storage.logical_arrays)[name][
            : mesh.storage.global_entity_counts[dimension]
        ]
    )


def resolve_mesh_scope(mesh: CellMesh, scope: MeshingScope, /) -> EntitySelection:
    """Resolve persistent mesh-entity IDs onto one exact mesh topology."""

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(scope, MeshingScope):
        raise TypeError("scope must be MeshingScope.")
    if scope.source_id != mesh.mesh_id:
        raise ValueError("Meshing scope belongs to another mesh source.")
    if scope.source_revision != mesh.numeric_version:
        raise ValueError("Meshing scope belongs to another mesh revision.")
    if scope.entity_kind is not MeshingEntityKind.MESH:
        raise ValueError("Meshing scope must select mesh entities.")
    if scope.entity_dimension < 0 or scope.entity_dimension > mesh.topological_dimension:
        raise ValueError("Meshing scope entity dimension is not present on the mesh.")
    entities = mesh.entity_set(scope.entity_dimension)
    if scope.entity_set_id != entities.entity_set_id:
        raise ValueError("Meshing scope does not match the mesh entity set.")
    if scope._scope_projection is not None:
        projection = scope._scope_projection
        if (
            scope.global_entity_ids is not projection.members
            or scope.logical_entity_set_id != projection.logical_entity_set_id
            or scope.scope_id != projection.scope_id
        ):
            raise ValueError(
                "Prepared mesh scope lost its exact scientific member receipt."
            )
        mask = jnp.asarray(projection.local_mask(mesh), dtype=jnp.bool_)
        if not np.array_equal(
            np.sort(
                np.asarray(entities.entity_ids, dtype=np.int64)[
                    np.asarray(mask, dtype=np.bool_)
                ]
            ),
            np.asarray(scope.entity_ids, dtype=np.int64),
        ):
            raise ValueError(
                "Prepared mesh scope local membership differs from its actual entity receipt."
            )
        return EntitySelection(entities, mask)
    requested = scope.global_entity_ids
    local_ids = jnp.asarray(entities.entity_ids, dtype=jnp.int64)
    mask = entities.active_mask & jnp.asarray(
        _local_logical_lookup(requested, None, local_ids)[1], dtype=jnp.bool_
    )
    if mesh.storage is None:
        complete = jnp.sum(mask, dtype=jnp.int64) == requested.size
    else:
        complete = jnp.all(
            _contains_ids(_mesh_global_ids(mesh, scope.entity_dimension), requested)
        )
    if not bool(jax.device_get(complete)):
        raise ValueError("Meshing scope contains undeclared mesh entity IDs.")
    return EntitySelection(entities, mask)


class ScopeResolutionReport(StrictModule, NonTrainableState):
    query: str = eqx.field(static=True)
    matched_names: tuple[str, ...] = eqx.field(static=True)
    unmatched_names: tuple[str, ...] = eqx.field(static=True)
    scope: MeshingScope
    report_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        query: str,
        scope: MeshingScope,
        /,
        *,
        matched_names: tuple[str, ...] = (),
        unmatched_names: tuple[str, ...] = (),
    ) -> None:
        expression = str(query).strip()
        if not expression:
            raise ValueError("Scope resolution query must be non-empty.")
        matched = tuple(str(value).strip() for value in matched_names)
        unmatched = tuple(str(value).strip() for value in unmatched_names)
        if any(not value for value in (*matched, *unmatched)):
            raise ValueError("Scope resolution names must be non-empty.")
        self.query = expression
        self.matched_names = matched
        self.unmatched_names = unmatched
        self.scope = scope
        self.report_id = canonical_fingerprint(
            {
                "kind": "scope-resolution-report",
                "query": expression,
                "matched_names": matched,
                "unmatched_names": unmatched,
                "scope": scope.scope_id,
            }
        )


__all__ = [
    "MeshingEntityKind",
    "MeshingScope",
    "ScopeResolutionReport",
    "resolve_mesh_scope",
]
