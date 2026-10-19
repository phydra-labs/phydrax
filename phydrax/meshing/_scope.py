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
from jax.experimental.multihost_utils import broadcast_one_to_all
from jax.sharding import NamedSharding, PartitionSpec, Sharding
from numpy.typing import ArrayLike, NDArray

import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._topology as topology_owner

from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked, parse
from ._entity_keys import key_positions


if TYPE_CHECKING:
    from . import (
        _initial_certification as initial_certification_owner,
        _publication_lowering as publication_lowering_owner,
        _result as result_owner,
    )
    from ._publication_lowering import PublicationProjection


@checked
def mesh_entity_scope(
    mesh: cell_mesh_owner.CellMesh, dimension: int, identifiers: np.ndarray, /
) -> MeshingScope:
    """Bind selected entity IDs to the mesh's dimension-specific inventory."""
    entities = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        identifiers,
    )


class MeshingEntityKind(StrEnum):
    GEOMETRY = "geometry"
    MESH = "mesh"
    PART = "part"
    LABEL = "label"
    ZONE = "zone"


class LocalScopeMembership(StrictModule, NonTrainableState):
    """Host-local membership with the first failed publication condition."""

    mask: NDArray[np.bool_] | None
    issue: str | None = eqx.field(static=True)


@jax.jit
def _scope_packet_values(
    ids: Array,
    owners: Array,
    membership: Array,
    packet_ids: Array,
    packet_owners: Array,
    packet_valid: Array,
    packet_membership: Array,
) -> tuple[Array, Array]:
    source_order = jnp.argsort(ids, stable=True)
    ordered_ids = ids[source_order]
    rows = jnp.minimum(jnp.searchsorted(ordered_ids, packet_ids), ids.shape[0] - 1)
    source_rows = source_order[rows]
    accepted = ~packet_valid | (
        (ordered_ids[rows] == packet_ids)
        & (owners[source_rows] == packet_owners)
        & (membership[source_rows] == packet_membership)
    )
    accepted &= packet_valid | ~packet_membership
    return ordered_ids, jnp.all(accepted)


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
        evidence: result_owner.CollectiveMeshEvidence
        | initial_certification_owner.InitialCollectiveMeshEvidence,
        publication: publication_lowering_owner.PublicationProjection,
        source_revision: str,
        dimension: int,
        membership_name: str,
        /,
        *,
        exclusive_scopes: tuple[MeshScopeProjection, ...] = (),
    ) -> None:
        # _scope ↔ publication/evidence: resolve the accepted global owner after scope definitions.
        from . import (
            _initial_certification as initial_certification_owner,
            _publication_lowering as publication_lowering_owner,
            _result as result_owner,
        )

        if not isinstance(
            evidence,
            (
                result_owner.CollectiveMeshEvidence,
                initial_certification_owner.InitialCollectiveMeshEvidence,
            ),
        ):
            raise TypeError(
                f"MeshScopeProjection.evidence must be collective mesh evidence; got {type(evidence).__name__}"
            )
        if not isinstance(publication, publication_lowering_owner.PublicationProjection):
            raise TypeError(
                f"MeshScopeProjection.publication must be PublicationProjection; got {type(publication).__name__}"
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
                f"MeshScopeProjection.publication must match evidence topology, organization, partition count and dimension; got {(publication.topology_id, publication.organization_id, publication.partition_count, dimension, source_revision)!r}"
            )
        arrays = dict(evidence.logical_arrays)
        member = arrays[membership_name]
        count = evidence.global_entity_counts[dimension]
        ids = publication.entity_ids[dimension]
        if member.dtype != jnp.bool_ or member.shape != ids.shape:
            raise ValueError(
                f"MeshScopeProjection.members must be bool values on the publication entity axis; got {member.dtype} {member.shape} for {ids.shape}"
            )
        if bool(jax.device_get(jnp.any(member[count:]))):
            raise ValueError(
                f"MeshScopeProjection.members must exclude padded rows; got selected rows beyond count {count}"
            )
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
                f"MeshScopeProjection.packets must have aligned partition-leading ID, owner, validity and membership arrays; got {packet_ids.dtype} {packet_ids.shape}"
            )
        global_ids, accepted = _scope_packet_values(
            ids[:count],
            publication.entity_owners[dimension],
            member,
            packet_ids,
            packet_owners,
            packet_valid,
            packet_membership,
        )
        if not bool(jax.device_get(accepted)):
            raise ValueError(
                "MeshScopeProjection.packets must agree with publication membership and ownership; got a differing packet entry"
            )
        members = jnp.sort(selected_ids(ids[:count], member[:count]), stable=True)
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
                    f"MeshScopeProjection.exclusive_scopes must share publication, source, dimension and evidence; got {type(other).__name__}"
                )
            if bool(jax.device_get(jnp.any(contains_ids(other.members, members)))):
                raise ValueError(
                    f"MeshScopeProjection.exclusive_scopes must be disjoint; got overlap with scope {other.scope_id!r}"
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

    @checked
    def local_mask(self, mesh: cell_mesh_owner.CellMesh, /) -> NDArray[np.bool_]:
        """Check received entity IDs, owners, and scope membership locally."""
        receipt = self._membership_receipt(mesh)
        if receipt.issue is not None:
            raise ValueError(receipt.issue)
        if receipt.mask is None:
            raise ValueError(
                "MeshScopeProjection.local_mask must contain membership when no issue is present; got None"
            )
        return receipt.mask

    @checked
    def matches_local_mesh(self, mesh: cell_mesh_owner.CellMesh, /) -> bool:
        """Admit a local constructor context without exception-driven probing."""
        return self._membership_receipt(mesh).issue is None

    def _membership_receipt(
        self, mesh: cell_mesh_owner.CellMesh, /
    ) -> LocalScopeMembership:
        storage = mesh.storage
        if (
            storage is None
            or mesh.mesh_id != self.source_id
            or mesh.numeric_version != self.source_revision
            or mesh.topology_id != self.publication.topology_id
            or storage.evidence_id != self.evidence_id
            or storage.partition_count != self.publication.partition_count
        ):
            return LocalScopeMembership(
                None,
                f"MeshScopeProjection.local_mask must match mesh source, revision and storage placement; got {(mesh.mesh_id, mesh.numeric_version, mesh.topology_id)!r}",
            )
        banks = dict(self.publication.projected_arrays)
        if (
            banks[f"entity/{self.dimension}/ids"] is not self.packet_ids
            or banks[f"entity/{self.dimension}/owners"] is not self.packet_owners
            or banks[f"entity/{self.dimension}/valid"] is not self.packet_valid
            or banks[self.membership_name] is not self.packet_membership
        ):
            return LocalScopeMembership(
                None,
                "MeshScopeProjection.packets must retain publication array identity; got replaced packet arrays",
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
            return LocalScopeMembership(
                None,
                f"MeshScopeProjection.local_mask must match local entity IDs and owners; got packet/local counts {ids.size}/{local_ids.size}",
            )
        return LocalScopeMembership(
            membership[rows] & np.asarray(entities.active_mask, dtype=np.bool_), None
        )


def _host_member_ids(entity_ids: ArrayLike, /) -> NDArray[np.int64]:
    identifiers = np.asarray(
        jax.device_get(entity_ids) if isinstance(entity_ids, Array) else entity_ids
    )
    if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
        raise TypeError(
            f"MeshingScope.entity_ids must be an integer vector; got {identifiers.dtype} {identifiers.shape}"
        )
    identifiers = np.sort(identifiers.astype(np.int64), stable=True)
    if identifiers.size == 0:
        raise ValueError(
            "MeshingScope.entity_ids must contain at least one entity; got an empty vector"
        )
    if np.any(identifiers < 0) or np.any(identifiers[1:] == identifiers[:-1]):
        raise ValueError(
            f"MeshingScope.entity_ids must be unique and nonnegative; got an invalid vector of size {identifiers.size}"
        )
    return identifiers


def _prepare_scope_binding(
    source_id: str,
    source_revision: str,
    entity_kind: MeshingEntityKind,
    entity_dimension: int,
    entity_set_id: str,
    entity_ids: ArrayLike,
    local_mesh: cell_mesh_owner.CellMesh | None,
    local_basis: MeshingScope | None,
    projection: MeshScopeProjection | None,
    /,
) -> tuple[str, str, str, int, NDArray[np.int64] | Array]:
    source = str(source_id).strip()
    revision = str(source_revision).strip()
    entity_set = str(entity_set_id).strip()
    if not source or not revision or not entity_set:
        raise ValueError(
            f"MeshingScope.identities must be nonempty; got {(source, revision, entity_set)!r}"
        )
    parse(entity_kind, MeshingEntityKind, "MeshingScope.entity_kind")
    dimension = int(entity_dimension)
    if dimension < 0:
        raise ValueError(
            f"MeshingScope.entity_dimension must be nonnegative; got {dimension}"
        )
    if projection is None:
        identifiers = _host_member_ids(entity_ids)
    else:
        if entity_ids is not projection.members:
            raise ValueError(
                "MeshingScope.entity_ids must retain the projection members array identity; got a different bank"
            )
        if (
            local_mesh is None
            or local_basis is not None
            or (source, revision, dimension, entity_kind)
            != (
                projection.source_id,
                projection.source_revision,
                projection.dimension,
                MeshingEntityKind.MESH,
            )
        ):
            raise ValueError(
                f"MeshingScope.projection must match the local mesh binding; got source/revision/dimension {(source, revision, dimension)!r}"
            )
        identifiers = projection.members
    return source, revision, entity_set, dimension, identifiers


class MeshingScope(StrictModule, NonTrainableState):
    """Entity scope bound to one immutable source revision."""

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

    @checked
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
        local_mesh: cell_mesh_owner.CellMesh | None = None,
        _local_basis: MeshingScope | None = None,
        _projection: MeshScopeProjection | None = None,
    ) -> None:
        source, revision, entity_set, dimension, identifiers = _prepare_scope_binding(
            source_id,
            source_revision,
            entity_kind,
            entity_dimension,
            entity_set_id,
            entity_ids,
            local_mesh,
            _local_basis,
            _projection,
        )
        local_ids = identifiers
        local_coverage_id = None
        partition_index = None
        partition_count = None
        owners = np.zeros(identifiers.shape, dtype=np.int32)
        local_universe = None
        owner_universe = None
        global_universe = None
        logical_entity_set = entity_set
        if local_mesh is not None and _local_basis is not None:
            raise ValueError(
                "MeshingScope.local_mesh must be exclusive with _local_basis; got both owners"
            )
        if local_mesh is not None:
            if (
                entity_kind is not MeshingEntityKind.MESH
                or source != local_mesh.mesh_id
                or revision != local_mesh.numeric_version
                or dimension > local_mesh.topological_dimension
                or entity_set != local_mesh.entity_set(dimension).entity_set_id
            ):
                raise ValueError(
                    f"MeshingScope.local_mesh must match source, revision, dimension and entity set; got {(source, revision, dimension, entity_set)!r}"
                )
            entities = local_mesh.entity_set(dimension)
            local_universe = entities.entity_ids
            global_universe = (
                mesh_global_ids(local_mesh, dimension)
                if _projection is None
                else _projection.global_ids
            )
            mask = (
                np.isin(np.asarray(entities.entity_ids), identifiers)
                if _projection is None
                else _projection.local_mask(local_mesh)
            )
            mask = mask & np.asarray(entities.active_mask, dtype=np.bool_)
            local_ids = np.asarray(entities.entity_ids, dtype=np.int64)[mask]
            order = np.argsort(local_ids, stable=True)
            local_ids = local_ids[order]
            if local_mesh.storage is None:
                complete = np.count_nonzero(mask) == identifiers.size
                owners = np.zeros(local_ids.shape, dtype=np.int32)
                owner_universe = np.zeros(local_universe.shape, dtype=np.int32)
            else:
                storage = local_mesh.storage
                complete = (
                    np.all(local_logical_lookup(global_universe, None, identifiers)[1])
                    if _projection is None
                    else True
                )
                owner_universe = storage.entity_owner[dimension]
                owners = np.asarray(owner_universe)[mask][order]
                local_coverage_id = storage.evidence_id
                partition_index = storage.partition_index
                partition_count = storage.partition_count
            if not complete:
                raise ValueError(
                    "MeshingScope.entity_ids must belong to the global mesh inventory; got undeclared IDs"
                )
            logical_entity_set = (
                _logical_mesh_entity_set_id(
                    local_mesh.topology_id, dimension, global_universe
                )
                if _projection is None
                else _projection.logical_entity_set_id
            )
        if _local_basis is not None:
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
                    f"MeshingScope.local_basis must match the binding and contain placement inventories; got {basis.scope_id!r}"
                )
            local_universe = basis._local_entity_universe
            owner_universe = basis._local_owner_universe
            global_universe = basis._global_entity_universe
            if not np.all(local_logical_lookup(global_universe, None, identifiers)[1]):
                raise ValueError(
                    "MeshingScope.entity_ids must belong to the global mesh inventory; got undeclared IDs"
                )
            mask = np.isin(np.asarray(local_universe), identifiers)
            local_ids = np.asarray(local_universe)[mask]
            order = np.argsort(local_ids, stable=True)
            local_ids, owners = local_ids[order], np.asarray(owner_universe)[mask][order]
            local_coverage_id = basis.local_coverage_id
            partition_index, partition_count = (
                basis.local_partition_index,
                basis.local_partition_count,
            )
            logical_entity_set = basis.logical_entity_set_id
        if _projection is not None:
            identity = _projection.scope_id
        elif _local_basis is not None and np.array_equal(
            identifiers, np.asarray(_local_basis.global_entity_ids)
        ):
            identity = _local_basis.scope_id
        else:
            identity = _scope_inventory_id(
                source, revision, entity_kind, dimension, logical_entity_set, identifiers
            )
        shares_identifiers = local_ids is identifiers
        host_banks = tuple(
            value
            for value in (
                identifiers,
                None if shares_identifiers else local_ids,
                owners,
                owner_universe,
            )
            if value is not None and not isinstance(value, Array)
        )
        device_banks = iter(jax.device_put(host_banks))
        identifiers = (
            identifiers if isinstance(identifiers, Array) else next(device_banks)
        )
        local_ids = (
            local_ids
            if isinstance(local_ids, Array)
            else identifiers
            if shares_identifiers
            else next(device_banks)
        )
        owners = owners if isinstance(owners, Array) else next(device_banks)
        owner_universe = (
            owner_universe
            if owner_universe is None or isinstance(owner_universe, Array)
            else next(device_banks)
        )
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
        self.scope_id = identity

    @property
    def local_entity_universe(self) -> Array | None:
        """Return the entity inventory of this local placement."""
        return self._local_entity_universe

    @property
    def local_owner_universe(self) -> Array | None:
        """Return owners aligned with the local entity inventory."""
        return self._local_owner_universe

    @property
    def global_entity_universe(self) -> Array | None:
        """Return the accepted global inventory without gathering device leaves."""
        return self._global_entity_universe

    @property
    def projection(self) -> MeshScopeProjection | None:
        """Return the globally accepted membership projection, when prepared."""
        return self._scope_projection

    @checked
    def matches_local_mesh(
        self,
        mesh: cell_mesh_owner.CellMesh,
        /,
        *,
        projection: MeshScopeProjection | None = None,
    ) -> bool:
        """Admit a matching mesh binding and inventory for reconstruction."""
        if (
            self.entity_kind is not MeshingEntityKind.MESH
            or self.source_id != mesh.mesh_id
            or self.source_revision != mesh.numeric_version
            or self.entity_dimension < 0
            or self.entity_dimension > mesh.topological_dimension
            or self._local_entity_universe is None
        ):
            return False
        entities = mesh.entity_set(self.entity_dimension)
        if (
            self.entity_set_id != entities.entity_set_id
            or self._local_entity_universe.shape != entities.entity_ids.shape
            or self._local_entity_universe.dtype != entities.entity_ids.dtype
            or not bool(
                jax.device_get(
                    jnp.all(self._local_entity_universe == entities.entity_ids)
                )
            )
        ):
            return False
        if projection is not None:
            return (
                projection.source_id == self.source_id
                and projection.source_revision == self.source_revision
                and projection.dimension == self.entity_dimension
                and projection.logical_entity_set_id == self.logical_entity_set_id
                and projection.scope_id == self.scope_id
                and projection.matches_local_mesh(mesh)
            )
        if mesh.storage is not None:
            inventory = mesh_global_ids(mesh, self.entity_dimension)
            return bool(
                np.all(local_logical_lookup(inventory, None, self.global_entity_ids)[1])
            )
        selected = np.isin(
            np.asarray(jax.device_get(entities.entity_ids)),
            np.asarray(jax.device_get(self.global_entity_ids)),
        ) & np.asarray(jax.device_get(entities.active_mask), dtype=np.bool_)
        return np.count_nonzero(selected) == self.global_entity_ids.size

    @classmethod
    @checked
    def from_stored_fields(
        cls,
        value: MeshingScope,
        /,
        *,
        local_mesh: cell_mesh_owner.CellMesh | None = None,
        projection: MeshScopeProjection | None = None,
    ) -> MeshingScope:
        """Revalidate a stored scope using its retained mesh and membership projection."""
        return cls(
            value.source_id,
            value.source_revision,
            value.entity_kind,
            value.entity_dimension,
            value.entity_set_id,
            value.global_entity_ids if projection is None else projection.members,
            local_mesh=local_mesh,
            _projection=projection,
        )

    @checked
    def require_same_local_placement(
        self,
        expected: MeshingScope,
        /,
        *,
        preserve_global_identity: bool = False,
    ) -> None:
        """Require matching local coverage, membership, and ownership receipts."""
        if (
            self.scope_id != expected.scope_id
            or self.logical_entity_set_id != expected.logical_entity_set_id
            or (
                self.source_id,
                self.source_revision,
                self.entity_kind,
                self.entity_dimension,
                self.entity_set_id,
            )
            != (
                expected.source_id,
                expected.source_revision,
                expected.entity_kind,
                expected.entity_dimension,
                expected.entity_set_id,
            )
            or self.local_coverage_id != expected.local_coverage_id
            or self.local_partition_index != expected.local_partition_index
            or self.local_partition_count != expected.local_partition_count
        ):
            raise ValueError(
                f"MeshingScope.local_placement must match the expected coverage binding; got scope IDs {self.scope_id!r}/{expected.scope_id!r}"
            )
        for values, expected_values in (
            (self.global_entity_ids, expected.global_entity_ids),
            (self.entity_ids, expected.entity_ids),
            (self.entity_owner, expected.entity_owner),
            (self._local_entity_universe, expected._local_entity_universe),
            (self._local_owner_universe, expected._local_owner_universe),
            (self._global_entity_universe, expected._global_entity_universe),
        ):
            if values is None or expected_values is None:
                raise ValueError(
                    "MeshingScope.local_placement must contain both inventory universes; got None"
                )
            if preserve_global_identity and (
                values is self.global_entity_ids or values is self._global_entity_universe
            ):
                if values is not expected_values:
                    raise ValueError(
                        "MeshingScope.local_placement must retain global array identity; got a replaced member or inventory bank"
                    )
                continue
            if (
                values.dtype != expected_values.dtype
                or values.shape != expected_values.shape
                or not bool(jax.device_get(jnp.all(values == expected_values)))
            ):
                raise ValueError(
                    f"MeshingScope.local_placement must match membership and owners; got differing arrays with shape {values.shape}/{expected_values.shape}"
                )

    @checked
    def lower(self, mesh: cell_mesh_owner.CellMesh, /) -> MeshingScope:
        """Certify the locally present view without changing logical membership."""
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
                f"MeshingScope.lower must retain the logical entity inventory; got {lowered.logical_entity_set_id!r} instead of {self.logical_entity_set_id!r}"
            )
        return lowered

    @classmethod
    @checked
    def from_selection(
        cls,
        source_id: str,
        source_revision: str,
        entities: topology_owner.EntitySet,
        selection: topology_owner.EntitySelection,
        /,
    ) -> MeshingScope:
        """Convert a positional selection using its bound entity set."""
        if selection.entity_set_id != entities.entity_set_id:
            raise ValueError(
                f"MeshingScope.selection must belong to the supplied entity set; got {selection.entity_set_id!r} instead of {entities.entity_set_id!r}"
            )
        mask = np.asarray(selection.mask, dtype=np.bool_)
        if mask.shape != (entities.count,) or not np.array_equal(
            selection.active_mask, entities.active_mask
        ):
            raise ValueError(
                f"MeshingScope.selection must match entity capacity and active mask; got mask shape {mask.shape} for capacity {entities.count}"
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
            raise ValueError(
                f"MeshingScope.set_operation must use one source binding; got {binding!r} and {other_binding!r}"
            )
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
                f"MeshingScope.set_operation must use one local coverage placement; got coverage IDs {self.local_coverage_id!r}/{other.local_coverage_id!r}"
            )

    def union(self, other: MeshingScope, /) -> MeshingScope:
        self._check_compatible(other)
        return MeshingScope(
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
            unique_ids(
                jnp.concatenate((self.global_entity_ids, other.global_entity_ids))
            ),
            _local_basis=self
            if self._local_entity_universe is not None
            else (other if other._local_entity_universe is not None else None),
        )

    def intersection(self, other: MeshingScope, /) -> MeshingScope:
        self._check_compatible(other)
        values = selected_ids(
            self.global_entity_ids,
            contains_ids(other.global_entity_ids, self.global_entity_ids),
        )
        if values.size == 0:
            raise ValueError(
                "MeshingScope.intersection must contain at least one entity; got empty membership"
            )
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
        values = selected_ids(
            self.global_entity_ids,
            ~contains_ids(other.global_entity_ids, self.global_entity_ids),
        )
        if values.size == 0:
            raise ValueError(
                "MeshingScope.difference must contain at least one entity; got empty membership"
            )
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


def _host_scope_selection_layout(identifiers: Sharding, mask: Sharding) -> bool:
    """Recognize host selection without assembling a partitioned global array."""
    match identifiers, mask:
        case jax.sharding.SingleDeviceSharding(), jax.sharding.SingleDeviceSharding():
            return True
        case NamedSharding(), _:
            return (
                identifiers.num_devices == mask.num_devices == 1
                and identifiers.device_set == mask.device_set
                and all(
                    device.platform == "cpu"
                    for placement in (identifiers, mask)
                    for device in placement.device_set
                )
            )
        case _:
            return False


def selected_ids(identifiers: Array, mask: Array, /) -> Array:
    if (
        mask.dtype == jnp.bool_
        and not isinstance(identifiers, Tracer)
        and not isinstance(mask, Tracer)
        and identifiers.is_fully_addressable
        and mask.is_fully_addressable
        and _host_scope_selection_layout(identifiers.sharding, mask.sharding)
    ):
        values, selected = jax.device_get((identifiers, mask))
        values = np.asarray(values)[np.nonzero(np.asarray(selected))[0]]
        if isinstance(identifiers.sharding, NamedSharding):
            return jax.device_put(values, identifiers.sharding)
        return jnp.asarray(values, dtype=identifiers.dtype)
    count = int(jax.device_get(jnp.sum(mask, dtype=jnp.int64)))
    return identifiers[jnp.nonzero(mask, size=count)[0]]


def unique_ids(identifiers: Array, /) -> Array:
    if (
        not isinstance(identifiers, Tracer)
        and identifiers.is_fully_addressable
        and isinstance(identifiers.sharding, jax.sharding.SingleDeviceSharding)
    ):
        return jax.device_put(np.unique(np.asarray(identifiers)))
    ordered = jnp.sort(identifiers, stable=True)
    fresh = jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), ordered[1:] != ordered[:-1])
    )
    return selected_ids(ordered, fresh)


@jax.jit
def contains_ids(sorted_ids: Array, requested: Array, /) -> Array:
    positions = jnp.minimum(jnp.searchsorted(sorted_ids, requested), sorted_ids.size - 1)
    return sorted_ids[positions] == requested


@jax.jit
def _overlapping_id_kernel(sorted_ids: Array, requested: Array, /) -> Array:
    return jnp.any(contains_ids(sorted_ids, requested))


def overlapping_ids(sorted_ids: Array, requested: Array, /) -> bool:
    """Decide static scope overlap on the host or a nonresident device bank."""
    if all(
        identifiers.is_fully_addressable
        and isinstance(identifiers.sharding, jax.sharding.SingleDeviceSharding)
        for identifiers in (sorted_ids, requested)
    ):
        return bool(np.any(np.isin(np.asarray(requested), np.asarray(sorted_ids))))
    return bool(_overlapping_id_kernel(sorted_ids, requested))


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
    members: Array | NDArray[np.int64],
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
    """Hash sorted inventory and membership in the global phase."""
    if (
        global_ids.ndim != 1
        or members.ndim != 1
        or global_ids.dtype != jnp.int64
        or members.dtype != jnp.int64
    ):
        raise ValueError(
            f"MeshingScope.inventory must contain int64 ID vectors; got {global_ids.dtype} {global_ids.shape} and {members.dtype} {members.shape}"
        )
    if global_ids.size == 0 or members.size == 0:
        raise ValueError(
            f"MeshingScope.inventory must contain nonempty inventory and membership; got sizes {global_ids.size}/{members.size}"
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

    positions = key_positions(identifiers, queries)
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


def local_logical_lookup(
    identifiers: Array,
    values: Array | None,
    queries: Array | np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Route bounded semantic-ID worksets; never materialize a global shard bank."""
    if not isinstance(queries, (Array, np.ndarray)):
        raise TypeError(
            f"MeshingScope.queries must be JAX or NumPy arrays; got {type(queries).__name__}"
        )
    if isinstance(queries, Array) and not queries.is_fully_addressable:
        raise ValueError(
            "MeshingScope.queries must be locally addressable; got a nonaddressable array"
        )
    if (
        identifiers.ndim not in (1, 2)
        or identifiers.shape[0] == 0
        or queries.ndim != identifiers.ndim
        or queries.shape[1:] != identifiers.shape[1:]
        or identifiers.dtype != np.int64
        or queries.dtype != np.int64
    ):
        raise ValueError(
            f"MeshingScope.queries must be int64 vectors or fixed-width keys on the inventory axis; got {queries.dtype} {queries.shape} for {identifiers.dtype} {identifiers.shape}"
        )
    if values is not None and (
        values.ndim == 0 or values.shape[0] != identifiers.shape[0]
    ):
        raise ValueError(
            f"MeshingScope.values must match the inventory axis; got {values.shape} for {identifiers.shape}"
        )
    if identifiers.is_fully_addressable and (
        values is None or values.is_fully_addressable
    ):
        source_values, query_values, field_values = jax.device_get(
            (identifiers, queries, values)
        )
        source = np.asarray(source_values)
        requested = np.asarray(query_values)
        if source.ndim == 1:
            positions = np.minimum(
                np.searchsorted(source, requested), source.shape[0] - 1
            )
            valid = source[positions] == requested
        else:
            key_dtype = np.dtype(
                [(f"key_{column}", np.int64) for column in range(source.shape[1])]
            )
            keys = np.ascontiguousarray(source).view(key_dtype).reshape(-1)
            query_keys = np.ascontiguousarray(requested).view(key_dtype).reshape(-1)
            positions = np.minimum(
                np.searchsorted(keys, query_keys, side="left"), source.shape[0] - 1
            )
            valid = np.all(source[positions] == requested, axis=1)
            positions = np.where(valid, positions, 0)
        if values is None:
            return valid, valid
        return np.asarray(field_values)[positions], valid

    if not isinstance(identifiers.sharding, NamedSharding):
        raise ValueError(
            f"MeshingScope.inventory must use NamedSharding for distributed queries; got {type(identifiers.sharding).__name__}"
        )
    replicated = NamedSharding(identifiers.sharding.mesh, PartitionSpec())
    host_queries = np.asarray(jax.device_get(queries), dtype=np.int64)
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
            f"MeshingScope.inventory must fit one semantic row in the packet allowance; got {row_bytes} bytes per row"
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


def mesh_global_ids(mesh: cell_mesh_owner.CellMesh, dimension: int, /) -> Array:
    if mesh.storage is None:
        identifiers = jax.device_get(mesh.entity_set(dimension).entity_ids)
        return jax.device_put(np.sort(np.asarray(identifiers)))
    name = (
        "vertex_global_ids"
        if dimension == 0
        else "cell_global_ids"
        if dimension == mesh.topological_dimension
        else f"entity_global_ids_{dimension}"
    )
    bank = dict(mesh.storage.logical_arrays)[name]
    count = mesh.storage.global_entity_counts[dimension]
    if bank.is_fully_addressable:
        return jax.device_put(np.sort(np.asarray(jax.device_get(bank))[:count]))
    return jnp.sort(bank[:count])


@checked
def resolve_mesh_scope(
    mesh: cell_mesh_owner.CellMesh, scope: MeshingScope, /
) -> topology_owner.EntitySelection:
    """Resolve persistent mesh-entity IDs onto one bound mesh topology."""

    if scope.source_id != mesh.mesh_id:
        raise ValueError(
            f"MeshingScope.source_id must match the mesh source; got {scope.source_id!r} instead of {mesh.mesh_id!r}"
        )
    if scope.source_revision != mesh.numeric_version:
        raise ValueError(
            f"MeshingScope.source_revision must match the mesh revision; got {scope.source_revision!r} instead of {mesh.numeric_version!r}"
        )
    if scope.entity_kind is not MeshingEntityKind.MESH:
        raise ValueError(
            f"MeshingScope.entity_kind must select mesh entities; got {scope.entity_kind!r}"
        )
    if scope.entity_dimension < 0 or scope.entity_dimension > mesh.topological_dimension:
        raise ValueError(
            f"MeshingScope.entity_dimension must be present on the mesh; got {scope.entity_dimension} for dimension {mesh.topological_dimension}"
        )
    entities = mesh.entity_set(scope.entity_dimension)
    if scope.entity_set_id != entities.entity_set_id:
        raise ValueError(
            f"MeshingScope.entity_set_id must match the mesh entity set; got {scope.entity_set_id!r} instead of {entities.entity_set_id!r}"
        )
    if scope._scope_projection is not None:
        projection = scope._scope_projection
        if (
            scope.global_entity_ids is not projection.members
            or scope.logical_entity_set_id != projection.logical_entity_set_id
            or scope.scope_id != projection.scope_id
        ):
            raise ValueError(
                f"MeshingScope.projection must retain member bank and logical identifiers; got scope ID {scope.scope_id!r}"
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
                f"MeshingScope.entity_ids must match the projection local membership; got {scope.entity_ids.size} members"
            )
        return topology_owner.EntitySelection(entities, mask)
    requested = scope.global_entity_ids
    local_ids = jnp.asarray(entities.entity_ids, dtype=jnp.int64)
    if mesh.storage is None:
        host_mask = (
            np.asarray(entities.active_mask)
            & local_logical_lookup(requested, None, local_ids)[1]
        )
        if np.count_nonzero(host_mask) != requested.size:
            raise ValueError(
                "MeshingScope.entity_ids must belong to the mesh inventory; got undeclared IDs"
            )
        return topology_owner.EntitySelection(entities, jax.device_put(host_mask))
    mask = entities.active_mask & jnp.asarray(
        local_logical_lookup(requested, None, local_ids)[1], dtype=jnp.bool_
    )
    complete = jnp.all(
        contains_ids(mesh_global_ids(mesh, scope.entity_dimension), requested)
    )
    if not bool(jax.device_get(complete)):
        raise ValueError(
            "MeshingScope.entity_ids must belong to the mesh inventory; got undeclared IDs"
        )
    return topology_owner.EntitySelection(entities, mask)


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
            raise ValueError(
                f"ScopeResolutionReport.query must be nonempty; got {expression!r}"
            )
        matched = tuple(str(value).strip() for value in matched_names)
        unmatched = tuple(str(value).strip() for value in unmatched_names)
        if any(not value for value in (*matched, *unmatched)):
            raise ValueError(
                f"ScopeResolutionReport.names must be nonempty; got {(*matched, *unmatched)!r}"
            )
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
