#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax

from .._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification._registry import SupportTuple
from ..typing import checked, parse
from ._chunk_repository import _digest, _identifier, RepositoryCorruptionError


RestartClass: TypeAlias = Literal["bitwise", "tolerance", "unsupported"]
PayloadClass: TypeAlias = Literal["restart-state", "execution-cache"]
ChunkRangeReader: TypeAlias = Callable[["CanonicalRestartChunk", int, int], bytes]
DestinationShardWriter: TypeAlias = Callable[["DestinationShard", int, bytes], None]


MeshingArrayRole: TypeAlias = Literal[
    "entity-ids",
    "geometry",
    "topology",
    "lineage",
    "epoch",
    "evidence",
    "state",
    "constraints",
    "allocator",
    "source-record",
    "pending-work",
]
_MESHING_REFERENCE_NAMES = (
    "source_topology_id",
    "source_geometry_id",
    "topology_id",
    "geometry_id",
    "result_id",
    "source_revision_id",
    "coordinate_contract_id",
    "lineage_id",
    "epoch_id",
    "evidence_id",
    "transfer_id",
    "request_id",
)
_OPTIONAL_MESHING_REFERENCE_NAMES = frozenset(
    ("source_topology_id", "source_geometry_id", "lineage_id", "transfer_id")
)
_REQUIRED_MESHING_ROLES = frozenset(
    ("entity-ids", "geometry", "topology", "epoch", "evidence")
)


class MeshingCheckpointState(StrictModule, NonTrainableState):
    """Scientific checkpoint binding, not a mesh or an execution-state carrier.

    Array names identify decomposition-neutral logical arrays supplied separately
    to the addressable-shard publisher. Incidences and lineage refer to stable
    scientific IDs; neither former rank numbers nor prepared local slots define
    their identity. ``source_closure`` retains actual owning certification inputs,
    the accepted report, associations and native authoring records admitted by the
    owning source-closure validator. Its canonical recipe and complete numerical
    content digest bind every authored field to the same accepted state. Source
    numerical roles are declared automatically; scientific archive limits remain
    explicit admission controls, not checkpoint identity fields.
    """

    source_topology_id: str | None = eqx.field(static=True)
    source_geometry_id: str | None = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    source_revision_id: str = eqx.field(static=True)
    coordinate_contract_id: str = eqx.field(static=True)
    lineage_id: str | None = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    transfer_id: str | None = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    mesh_epoch: int = eqx.field(static=True)
    array_roles: tuple[tuple[str, MeshingArrayRole], ...] = eqx.field(static=True)
    checkpoint_state_id: str = eqx.field(static=True)
    source_recipe_json: str | None = eqx.field(static=True)
    source_content_digest: str | None = eqx.field(static=True)
    source_closure_id: str | None = eqx.field(static=True)
    source_closure: Mapping[str, Any] | None
    source_owner_index: int = eqx.field(static=True)
    source_array_bindings: tuple[tuple[str, str], ...] = eqx.field(static=True)
    source_logical_array_names: tuple[str, ...] = eqx.field(static=True)
    source_logical_content_digest: str | None = eqx.field(static=True)
    source_logical_arrays: Mapping[str, jax.Array] | None
    source_owner_states: tuple[MeshingCheckpointState, ...]

    def __init__(
        self,
        *,
        source_topology_id: str | None,
        source_geometry_id: str | None,
        topology_id: str,
        geometry_id: str,
        result_id: str,
        source_revision_id: str,
        coordinate_contract_id: str,
        lineage_id: str | None,
        epoch_id: str,
        evidence_id: str,
        transfer_id: str | None,
        request_id: str,
        mesh_epoch: int,
        array_roles: Sequence[tuple[str, MeshingArrayRole]],
        source_closure: Mapping[str, Any] | None = None,
        source_recipe_json: str | None = None,
        source_content_digest: str | None = None,
        source_owner_index: int | None = None,
        source_array_bindings: Mapping[str, str] | None = None,
        source_logical_arrays: Mapping[str, jax.Array] | None = None,
        source_logical_array_names: Sequence[str] = (),
        source_logical_content_digest: str | None = None,
        source_owner_states: Sequence[MeshingCheckpointState] = (),
        limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
        maximum_source_chunk_bytes: int = 1 << 20,
    ) -> None:
        required_references = {
            "topology_id": topology_id,
            "geometry_id": geometry_id,
            "result_id": result_id,
            "source_revision_id": source_revision_id,
            "coordinate_contract_id": coordinate_contract_id,
            "epoch_id": epoch_id,
            "evidence_id": evidence_id,
            "request_id": request_id,
        }
        required_references = {
            name: _identifier(value, name) for name, value in required_references.items()
        }
        optional_references = {
            "source_topology_id": source_topology_id,
            "source_geometry_id": source_geometry_id,
            "lineage_id": lineage_id,
            "transfer_id": transfer_id,
        }
        optional_references = {
            name: None if value is None else _identifier(value, name)
            for name, value in optional_references.items()
        }
        references = {**required_references, **optional_references}
        if type(mesh_epoch) is not int or mesh_epoch < 0:
            raise ValueError("mesh_epoch must be a non-negative integer.")
        from ._distributed_checkpoint import (
            _admit_meshing_source_logical_arrays,
            _prepare_meshing_source_binding,
            _validate_meshing_source_references,
        )

        owner_index = (
            (
                jax.process_index()
                if source_closure is not None or source_recipe_json is not None
                else 0
            )
            if source_owner_index is None
            else source_owner_index
        )
        if type(owner_index) is not int or owner_index < 0:
            raise ValueError("source_owner_index must be a non-negative integer.")
        source_logical_arrays = _admit_meshing_source_logical_arrays(
            source_logical_arrays, maximum_chunk_bytes=maximum_source_chunk_bytes
        )
        bindings = {} if source_array_bindings is None else dict(source_array_bindings)
        (
            source_recipe_json,
            source_content_digest,
            source_roles,
            logical_names,
            logical_digest,
            source_logical_arrays,
        ) = _prepare_meshing_source_binding(
            source_closure,
            source_recipe_json,
            source_content_digest,
            owner_index=owner_index,
            array_bindings=bindings,
            logical_arrays=source_logical_arrays,
            logical_array_names=source_logical_array_names,
            logical_content_digest=source_logical_content_digest,
            limits=limits,
            maximum_chunk_bytes=maximum_source_chunk_bytes,
        )
        owner_states = tuple(source_owner_states)
        if any(type(state) is not MeshingCheckpointState for state in owner_states):
            raise TypeError(
                "source_owner_states must contain exact meshing checkpoint states."
            )
        if owner_states and (
            tuple(state.source_owner_index for state in owner_states)
            != tuple(range(len(owner_states)))
            or any(state.source_owner_states for state in owner_states)
        ):
            raise ValueError(
                "Source owner records must cover original owners exactly once."
            )
        if source_closure is not None:
            _validate_meshing_source_references(source_closure, references)
        source_closure_id = (
            None
            if source_recipe_json is None
            else canonical_fingerprint(
                {
                    "recipe": source_recipe_json,
                    "array_digest": source_content_digest,
                }
            )
        )
        supplied_roles = tuple(array_roles)
        if any(role == "source-record" for _, role in supplied_roles):
            if {item for item in supplied_roles if item[1] == "source-record"} != set(
                source_roles
            ):
                raise ValueError("Source record roles must exactly match their recipe.")
        else:
            supplied_roles += source_roles
        roles = tuple(
            sorted(
                (
                    _identifier(name, "array name"),
                    parse(role, MeshingArrayRole, "array role"),
                )
                for name, role in supplied_roles
            )
        )
        if len({name for name, _ in roles}) != len(roles):
            raise ValueError("Meshing checkpoint array names must be unique.")
        required_roles = _REQUIRED_MESHING_ROLES | (
            {"lineage"} if optional_references["lineage_id"] is not None else set()
        )
        has_declared_state = (source_closure is None and source_recipe_json is None) or (
            source_closure is not None
            and bool(source_closure.get("accepted_data", {}).get("fields", {}))
        )
        if has_declared_state:
            required_roles |= {"state"}
        if not required_roles.issubset(role for _, role in roles):
            raise ValueError("Meshing checkpoint lacks required scientific array roles.")
        self.source_topology_id = optional_references["source_topology_id"]
        self.source_geometry_id = optional_references["source_geometry_id"]
        self.topology_id = required_references["topology_id"]
        self.geometry_id = required_references["geometry_id"]
        self.result_id = required_references["result_id"]
        self.source_revision_id = required_references["source_revision_id"]
        self.coordinate_contract_id = required_references["coordinate_contract_id"]
        self.lineage_id = optional_references["lineage_id"]
        self.epoch_id = required_references["epoch_id"]
        self.evidence_id = required_references["evidence_id"]
        self.transfer_id = optional_references["transfer_id"]
        self.request_id = required_references["request_id"]
        self.mesh_epoch = mesh_epoch
        self.array_roles = roles
        self.source_recipe_json = source_recipe_json
        self.source_content_digest = source_content_digest
        self.source_closure_id = source_closure_id
        self.source_closure = source_closure
        self.source_owner_index = owner_index
        self.source_array_bindings = tuple(sorted(bindings.items()))
        self.source_logical_array_names = logical_names
        self.source_logical_content_digest = logical_digest
        self.source_logical_arrays = source_logical_arrays
        self.source_owner_states = owner_states
        self.checkpoint_state_id = canonical_fingerprint(
            {
                **references,
                "mesh_epoch": mesh_epoch,
                "array_roles": roles,
                "source_recipe_json": source_recipe_json,
                "source_content_digest": source_content_digest,
                "source_closure_id": source_closure_id,
                "source_owner_index": owner_index,
                "source_array_bindings": self.source_array_bindings,
                "source_logical_array_names": logical_names,
                "source_logical_content_digest": logical_digest,
            }
        )

    def to_payload(self) -> dict[str, Any]:
        """Return bounded immutable metadata; numerical arrays are separate."""
        return {
            "source_topology_id": self.source_topology_id,
            "source_geometry_id": self.source_geometry_id,
            "topology_id": self.topology_id,
            "geometry_id": self.geometry_id,
            "result_id": self.result_id,
            "source_revision_id": self.source_revision_id,
            "coordinate_contract_id": self.coordinate_contract_id,
            "lineage_id": self.lineage_id,
            "epoch_id": self.epoch_id,
            "evidence_id": self.evidence_id,
            "transfer_id": self.transfer_id,
            "request_id": self.request_id,
            "mesh_epoch": self.mesh_epoch,
            "array_roles": [list(item) for item in self.array_roles],
            "source_recipe_json": self.source_recipe_json,
            "source_content_digest": self.source_content_digest,
            "source_owner_index": self.source_owner_index,
            "source_array_bindings": [list(item) for item in self.source_array_bindings],
            "source_logical_array_names": list(self.source_logical_array_names),
            "source_logical_content_digest": self.source_logical_content_digest,
        }

    def source_array_name(self, recipe_name: str, /) -> str:
        """Return the exact checkpoint path binding for one canonical source leaf."""
        from ._distributed_checkpoint import _source_array_name

        return _source_array_name(
            recipe_name, self.source_owner_index, dict(self.source_array_bindings)
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        /,
        *,
        source_closure: Mapping[str, Any] | None = None,
        source_logical_arrays: Mapping[str, jax.Array] | None = None,
        source_owner_states: Sequence[MeshingCheckpointState] = (),
        limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
        maximum_source_chunk_bytes: int = 1 << 20,
    ) -> MeshingCheckpointState:
        """Restore through the normal validated construction boundary."""
        if set(payload) != {
            *_MESHING_REFERENCE_NAMES,
            "mesh_epoch",
            "array_roles",
            "source_recipe_json",
            "source_content_digest",
            "source_owner_index",
            "source_array_bindings",
            "source_logical_array_names",
            "source_logical_content_digest",
        }:
            raise ValueError("Meshing checkpoint metadata fields are invalid.")
        if any(
            not isinstance(payload[name], str)
            and not (name in _OPTIONAL_MESHING_REFERENCE_NAMES and payload[name] is None)
            for name in _MESHING_REFERENCE_NAMES
        ):
            raise TypeError(
                "Meshing checkpoint references require strings or owning typed absence."
            )
        roles = payload["array_roles"]
        if not isinstance(roles, list) or any(
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            for item in roles
        ):
            raise TypeError("Meshing checkpoint array roles are invalid.")
        bindings = payload["source_array_bindings"]
        if (
            not isinstance(bindings, list)
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
                for item in bindings
            )
            or len({item[0] for item in bindings}) != len(bindings)
        ):
            raise ValueError("Meshing checkpoint source array bindings are invalid.")
        logical_names = payload["source_logical_array_names"]
        if not isinstance(logical_names, list) or any(
            not isinstance(name, str) for name in logical_names
        ):
            raise ValueError("Meshing checkpoint source logical names are invalid.")
        return cls(
            source_topology_id=payload["source_topology_id"],
            source_geometry_id=payload["source_geometry_id"],
            topology_id=payload["topology_id"],
            geometry_id=payload["geometry_id"],
            result_id=payload["result_id"],
            source_revision_id=payload["source_revision_id"],
            coordinate_contract_id=payload["coordinate_contract_id"],
            lineage_id=payload["lineage_id"],
            epoch_id=payload["epoch_id"],
            evidence_id=payload["evidence_id"],
            transfer_id=payload["transfer_id"],
            request_id=payload["request_id"],
            mesh_epoch=payload["mesh_epoch"],
            source_closure=source_closure,
            source_recipe_json=payload["source_recipe_json"],
            source_content_digest=payload["source_content_digest"],
            source_owner_index=payload["source_owner_index"],
            source_array_bindings=dict(bindings),
            source_logical_arrays=source_logical_arrays,
            source_logical_array_names=logical_names,
            source_logical_content_digest=payload["source_logical_content_digest"],
            source_owner_states=source_owner_states,
            limits=limits,
            maximum_source_chunk_bytes=maximum_source_chunk_bytes,
            array_roles=tuple(
                (name, parse(role, MeshingArrayRole, "array role"))
                for name, role in roles
            ),
        )


class TopologyRestartRelation(StrictModule, NonTrainableState):
    """Exact source/target support relation and restart-equivalence class."""

    source_support_tuple_id: str = eqx.field(static=True)
    target_support_tuple_id: str = eqx.field(static=True)
    restart_class: RestartClass = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: SupportTuple | str,
        target: SupportTuple | str,
        restart_class: RestartClass,
        /,
        *,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
        reason: str = "",
    ) -> None:
        source_id = _support_id(source, "source")
        target_id = _support_id(target, "target")
        restart_class = parse(restart_class, RestartClass, "restart_class")
        absolute = _finite_nonnegative(absolute_tolerance, "absolute_tolerance")
        relative = _finite_nonnegative(relative_tolerance, "relative_tolerance")
        reason_ = str(reason).strip()
        if restart_class == "bitwise" and (absolute != 0.0 or relative != 0.0):
            raise ValueError("Bitwise restart relations cannot declare tolerances.")
        if restart_class == "tolerance" and absolute == 0.0 and relative == 0.0:
            raise ValueError("Tolerance restart relations require a positive tolerance.")
        if restart_class == "unsupported" and not reason_:
            raise ValueError("Unsupported restart relations require an explicit reason.")
        if len(reason_.encode("utf-8")) > 4096:
            raise ValueError("Restart relation reasons must be at most 4096 bytes.")
        self.source_support_tuple_id = source_id
        self.target_support_tuple_id = target_id
        self.restart_class = restart_class
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.reason = reason_
        self.relation_id = canonical_fingerprint(
            {
                "kind": "topology-restart-relation",
                "source_support_tuple_id": source_id,
                "target_support_tuple_id": target_id,
                "restart_class": restart_class,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "reason": reason_,
            }
        )


class TopologyRestartPolicy(StrictModule, NonTrainableState):
    """Fail-closed admission policy for exact and topology-changing restarts."""

    allow_topology_change: bool = eqx.field(static=True)
    allow_tolerance_restart: bool = eqx.field(static=True)
    maximum_absolute_tolerance: float = eqx.field(static=True)
    maximum_relative_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        allow_topology_change: bool,
        allow_tolerance_restart: bool = False,
        maximum_absolute_tolerance: float = 0.0,
        maximum_relative_tolerance: float = 0.0,
    ) -> None:
        absolute = _finite_nonnegative(
            maximum_absolute_tolerance, "maximum_absolute_tolerance"
        )
        relative = _finite_nonnegative(
            maximum_relative_tolerance, "maximum_relative_tolerance"
        )
        if not allow_tolerance_restart and (absolute != 0.0 or relative != 0.0):
            raise ValueError("A bitwise-only restart policy cannot declare tolerances.")
        self.allow_topology_change = bool(allow_topology_change)
        self.allow_tolerance_restart = bool(allow_tolerance_restart)
        self.maximum_absolute_tolerance = absolute
        self.maximum_relative_tolerance = relative
        self.policy_id = canonical_fingerprint(
            {
                "kind": "topology-restart-policy",
                "allow_topology_change": self.allow_topology_change,
                "allow_tolerance_restart": self.allow_tolerance_restart,
                "maximum_absolute_tolerance": absolute,
                "maximum_relative_tolerance": relative,
            }
        )


class RestartAdmission(StrictModule, NonTrainableState):
    """Immutable fail-closed decision for one relation under one policy."""

    relation_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    admitted: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation_id: str,
        policy_id: str,
        admitted: bool,
        reason: str,
        /,
    ) -> None:
        relation = _digest(relation_id, "relation_id")
        policy = _digest(policy_id, "policy_id")
        reason_ = str(reason).strip()
        if not reason_:
            raise ValueError("Restart admission requires an explicit reason.")
        self.relation_id = relation
        self.policy_id = policy
        self.admitted = bool(admitted)
        self.reason = reason_
        self.admission_id = canonical_fingerprint(
            {
                "kind": "topology-restart-admission",
                "relation_id": relation,
                "policy_id": policy,
                "admitted": self.admitted,
                "reason": reason_,
            }
        )


class CanonicalRestartChunk(StrictModule, NonTrainableState):
    """One decomposition-neutral canonical byte interval."""

    logical_name: str = eqx.field(static=True)
    canonical_offset: int = eqx.field(static=True)
    byte_count: int = eqx.field(static=True)
    payload_sha256: str = eqx.field(static=True)
    payload_class: PayloadClass = eqx.field(static=True)
    chunk_id: str = eqx.field(static=True)

    def __init__(
        self,
        logical_name: str,
        canonical_offset: int,
        byte_count: int,
        payload_sha256: str,
        /,
        *,
        payload_class: PayloadClass = "restart-state",
    ) -> None:
        logical = _identifier(logical_name, "logical_name")
        offset = _nonnegative(canonical_offset, "canonical_offset")
        count = _positive(byte_count, "byte_count")
        digest = _digest(payload_sha256, "payload_sha256")
        payload_class = parse(payload_class, PayloadClass, "payload_class")
        self.logical_name = logical
        self.canonical_offset = offset
        self.byte_count = count
        self.payload_sha256 = digest
        self.payload_class = payload_class
        self.chunk_id = canonical_fingerprint(
            {
                "kind": "canonical-restart-chunk",
                "logical_name": logical,
                "canonical_offset": offset,
                "byte_count": count,
                "payload_sha256": digest,
                "payload_class": payload_class,
            }
        )


class DestinationShard(StrictModule, NonTrainableState):
    """One destination-owned canonical interval restored without global gather."""

    shard_id: str = eqx.field(static=True)
    logical_name: str = eqx.field(static=True)
    canonical_offset: int = eqx.field(static=True)
    byte_count: int = eqx.field(static=True)
    shard_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        shard_id: str,
        logical_name: str,
        canonical_offset: int,
        byte_count: int,
        /,
    ) -> None:
        shard = _identifier(shard_id, "shard_id")
        logical = _identifier(logical_name, "logical_name")
        offset = _nonnegative(canonical_offset, "canonical_offset")
        count = _positive(byte_count, "byte_count")
        self.shard_id = shard
        self.logical_name = logical
        self.canonical_offset = offset
        self.byte_count = count
        self.shard_fingerprint = canonical_fingerprint(
            {
                "kind": "restart-destination-shard",
                "shard_id": shard,
                "logical_name": logical,
                "canonical_offset": offset,
                "byte_count": count,
            }
        )


class RestartChunkMapping(StrictModule, NonTrainableState):
    """Direct range transfer from one canonical source chunk to a target shard."""

    source_chunk_id: str = eqx.field(static=True)
    source_offset: int = eqx.field(static=True)
    target_shard_id: str = eqx.field(static=True)
    target_offset: int = eqx.field(static=True)
    byte_count: int = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_chunk_id: str,
        source_offset: int,
        target_shard_id: str,
        target_offset: int,
        byte_count: int,
        /,
    ) -> None:
        source = _digest(source_chunk_id, "source_chunk_id")
        source_offset_ = _nonnegative(source_offset, "source_offset")
        target = _identifier(target_shard_id, "target_shard_id")
        target_offset_ = _nonnegative(target_offset, "target_offset")
        count = _positive(byte_count, "byte_count")
        self.source_chunk_id = source
        self.source_offset = source_offset_
        self.target_shard_id = target
        self.target_offset = target_offset_
        self.byte_count = count
        self.mapping_id = canonical_fingerprint(
            {
                "kind": "restart-chunk-mapping",
                "source_chunk_id": source,
                "source_offset": source_offset_,
                "target_shard_id": target,
                "target_offset": target_offset_,
                "byte_count": count,
            }
        )


class DirectRestorePlan(StrictModule, NonTrainableState):
    """Validated destination-shard-direct restore with no execution-cache inputs."""

    relation: TopologyRestartRelation
    admission: RestartAdmission
    source_chunks: tuple[CanonicalRestartChunk, ...]
    destination_shards: tuple[DestinationShard, ...]
    mappings: tuple[RestartChunkMapping, ...]
    excludes_execution_cache: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        relation: TopologyRestartRelation,
        admission: RestartAdmission,
        source_chunks: Sequence[CanonicalRestartChunk],
        destination_shards: Sequence[DestinationShard],
        mappings: Sequence[RestartChunkMapping],
        /,
    ) -> None:
        if admission.relation_id != relation.relation_id or not admission.admitted:
            raise ValueError("Direct restore requires an admitted matching relation.")
        sources = tuple(
            sorted(
                tuple(source_chunks),
                key=lambda item: (
                    item.logical_name,
                    item.canonical_offset,
                    item.chunk_id,
                ),
            )
        )
        destinations = tuple(
            sorted(
                tuple(destination_shards),
                key=lambda item: (
                    item.logical_name,
                    item.canonical_offset,
                    item.shard_id,
                ),
            )
        )
        mappings_ = tuple(
            sorted(
                tuple(mappings),
                key=lambda item: (
                    item.target_shard_id,
                    item.target_offset,
                    item.source_chunk_id,
                    item.source_offset,
                ),
            )
        )
        if not sources or any(
            not isinstance(item, CanonicalRestartChunk) for item in sources
        ):
            raise TypeError("Direct restore requires typed source chunks.")
        if any(item.payload_class != "restart-state" for item in sources):
            raise ValueError("Execution-cache chunks cannot enter a restore plan.")
        if not destinations or any(
            not isinstance(item, DestinationShard) for item in destinations
        ):
            raise TypeError("Direct restore requires typed destination shards.")
        if not mappings_ or any(
            not isinstance(item, RestartChunkMapping) for item in mappings_
        ):
            raise TypeError("Direct restore requires typed chunk mappings.")
        if len({item.chunk_id for item in sources}) != len(sources):
            raise ValueError("Source chunk IDs must be unique.")
        if len({item.shard_id for item in destinations}) != len(destinations):
            raise ValueError("Destination shard IDs must be unique.")
        _validate_canonical_partition(sources, "source chunks")
        _validate_canonical_partition(destinations, "destination shards")
        _validate_mappings(sources, destinations, mappings_)
        self.relation = relation
        self.admission = admission
        self.source_chunks = sources
        self.destination_shards = destinations
        self.mappings = mappings_
        self.excludes_execution_cache = True
        self.plan_id = canonical_fingerprint(
            {
                "kind": "direct-topology-restore-plan",
                "relation_id": relation.relation_id,
                "admission_id": admission.admission_id,
                "source_chunk_ids": [item.chunk_id for item in sources],
                "destination_shards": [item.shard_fingerprint for item in destinations],
                "mapping_ids": [item.mapping_id for item in mappings_],
                "excludes_execution_cache": True,
            }
        )


class RestartExecutionReport(StrictModule, NonTrainableState):
    """Immutable account of a completed direct restore execution."""

    plan_id: str = eqx.field(static=True)
    restored_shard_ids: tuple[str, ...] = eqx.field(static=True)
    transferred_bytes: int = eqx.field(static=True)
    transfer_count: int = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan_id: str,
        restored_shard_ids: Sequence[str],
        transferred_bytes: int,
        transfer_count: int,
        /,
    ) -> None:
        plan = _digest(plan_id, "plan_id")
        shards = tuple(
            _identifier(item, "restored shard ID") for item in restored_shard_ids
        )
        if len(set(shards)) != len(shards):
            raise ValueError("Restored shard IDs must be unique.")
        transferred = _nonnegative(transferred_bytes, "transferred_bytes")
        count = _nonnegative(transfer_count, "transfer_count")
        self.plan_id = plan
        self.restored_shard_ids = shards
        self.transferred_bytes = transferred
        self.transfer_count = count
        self.report_id = canonical_fingerprint(
            {
                "kind": "restart-execution-report",
                "plan_id": plan,
                "restored_shard_ids": list(shards),
                "transferred_bytes": transferred,
                "transfer_count": count,
            }
        )


def admit_topology_restart(
    relation: TopologyRestartRelation,
    policy: TopologyRestartPolicy,
    /,
) -> RestartAdmission:
    """Apply exact fail-closed same/topology-change restart admission."""
    if not isinstance(relation, TopologyRestartRelation) or not isinstance(
        policy, TopologyRestartPolicy
    ):
        raise TypeError("Restart admission requires a relation and policy.")
    same = relation.source_support_tuple_id == relation.target_support_tuple_id
    admitted: bool
    reason: str
    if relation.restart_class == "unsupported":
        admitted = False
        reason = relation.reason
    elif not same and not policy.allow_topology_change:
        admitted = False
        reason = "Topology-changing restart is disabled by policy."
    elif relation.restart_class == "tolerance" and not policy.allow_tolerance_restart:
        admitted = False
        reason = "Tolerance-class restart is disabled by policy."
    elif (
        relation.absolute_tolerance > policy.maximum_absolute_tolerance
        or relation.relative_tolerance > policy.maximum_relative_tolerance
    ):
        admitted = False
        reason = "Restart relation exceeds the admitted tolerance envelope."
    else:
        admitted = True
        reason = (
            "Exact support tuple restart admitted."
            if same
            else "Qualified topology-change restart admitted."
        )
    return RestartAdmission(relation.relation_id, policy.policy_id, admitted, reason)


def canonical_chunk_mapping(
    source_chunks: Sequence[CanonicalRestartChunk],
    destination_shards: Sequence[DestinationShard],
    /,
) -> tuple[RestartChunkMapping, ...]:
    """Map intersections of canonical intervals directly into destination shards."""
    if not all(isinstance(item, CanonicalRestartChunk) for item in source_chunks):
        raise TypeError("source_chunks must contain CanonicalRestartChunk values.")
    if not all(isinstance(item, DestinationShard) for item in destination_shards):
        raise TypeError("destination_shards must contain DestinationShard values.")
    sources = tuple(
        sorted(
            (item for item in source_chunks if item.payload_class == "restart-state"),
            key=lambda item: (item.logical_name, item.canonical_offset, item.chunk_id),
        )
    )
    destinations = tuple(
        sorted(
            destination_shards,
            key=lambda item: (item.logical_name, item.canonical_offset, item.shard_id),
        )
    )
    _validate_canonical_partition(sources, "source chunks")
    _validate_canonical_partition(destinations, "destination shards")
    mappings: list[RestartChunkMapping] = []
    for destination in destinations:
        destination_end = destination.canonical_offset + destination.byte_count
        for source in sources:
            if source.logical_name != destination.logical_name:
                continue
            source_end = source.canonical_offset + source.byte_count
            start = max(source.canonical_offset, destination.canonical_offset)
            stop = min(source_end, destination_end)
            if start < stop:
                mappings.append(
                    RestartChunkMapping(
                        source.chunk_id,
                        start - source.canonical_offset,
                        destination.shard_id,
                        start - destination.canonical_offset,
                        stop - start,
                    )
                )
    _validate_mappings(sources, destinations, mappings)
    return tuple(mappings)


def prepare_direct_restore(
    relation: TopologyRestartRelation,
    policy: TopologyRestartPolicy,
    source_chunks: Sequence[CanonicalRestartChunk],
    destination_shards: Sequence[DestinationShard],
    /,
) -> DirectRestorePlan:
    """Admit and prepare a canonical, cache-excluding direct restore plan."""
    admission = admit_topology_restart(relation, policy)
    if not admission.admitted:
        raise ValueError(f"Topology restart is not admitted: {admission.reason}")
    state_chunks = tuple(
        item for item in source_chunks if item.payload_class == "restart-state"
    )
    mappings = canonical_chunk_mapping(state_chunks, destination_shards)
    return DirectRestorePlan(
        relation, admission, state_chunks, destination_shards, mappings
    )


def execute_direct_restore(
    plan: DirectRestorePlan,
    read_range: ChunkRangeReader,
    write_range: DestinationShardWriter,
    /,
) -> RestartExecutionReport:
    """Execute bounded segment transfers directly into each destination shard."""
    if not isinstance(plan, DirectRestorePlan):
        raise TypeError("plan must be DirectRestorePlan.")
    mappings_by_source: dict[str, list[RestartChunkMapping]] = {}
    for mapping in plan.mappings:
        mappings_by_source.setdefault(mapping.source_chunk_id, []).append(mapping)
    destinations = {item.shard_id: item for item in plan.destination_shards}
    transferred = 0
    for source in plan.source_chunks:
        digest = hashlib.sha256()
        for mapping in sorted(
            mappings_by_source[source.chunk_id],
            key=lambda item: item.source_offset,
        ):
            destination = destinations[mapping.target_shard_id]
            payload = read_range(source, mapping.source_offset, mapping.byte_count)
            if not isinstance(payload, bytes) or len(payload) != mapping.byte_count:
                raise RepositoryCorruptionError(
                    "Restart range reader returned an invalid bounded payload."
                )
            digest.update(payload)
            write_range(destination, mapping.target_offset, payload)
            transferred += mapping.byte_count
        if digest.hexdigest() != source.payload_sha256:
            raise RepositoryCorruptionError(
                f"Restart chunk {source.chunk_id!r} checksum failed."
            )
    return RestartExecutionReport(
        plan.plan_id,
        tuple(item.shard_id for item in plan.destination_shards),
        transferred,
        len(plan.mappings),
    )


def _validate_canonical_partition(
    records: Sequence[CanonicalRestartChunk] | Sequence[DestinationShard],
    label: str,
    /,
) -> None:
    if not records:
        raise ValueError(f"Canonical {label} cannot be empty.")
    logical_names = sorted({item.logical_name for item in records})
    for logical_name in logical_names:
        matching = tuple(item for item in records if item.logical_name == logical_name)
        expected = 0
        for item in matching:
            if item.canonical_offset != expected:
                relation = "overlap" if item.canonical_offset < expected else "hole"
                raise ValueError(
                    f"Canonical {label} for {logical_name!r} contain a {relation}."
                )
            expected += item.byte_count


def _validate_mappings(
    sources: Sequence[CanonicalRestartChunk],
    destinations: Sequence[DestinationShard],
    mappings: Sequence[RestartChunkMapping],
    /,
) -> None:
    source_by_id = {item.chunk_id: item for item in sources}
    destination_by_id = {item.shard_id: item for item in destinations}
    if len({item.mapping_id for item in mappings}) != len(mappings):
        raise ValueError("Restart mapping IDs must be unique.")
    for mapping in mappings:
        source = source_by_id.get(mapping.source_chunk_id)
        destination = destination_by_id.get(mapping.target_shard_id)
        if source is None or destination is None:
            raise ValueError("Restart mapping references an unknown chunk or shard.")
        if source.payload_class != "restart-state":
            raise ValueError("Restart mapping references execution-cache content.")
        if mapping.source_offset + mapping.byte_count > source.byte_count:
            raise ValueError("Restart mapping exceeds its source chunk.")
        if mapping.target_offset + mapping.byte_count > destination.byte_count:
            raise ValueError("Restart mapping exceeds its destination shard.")
        source_canonical = source.canonical_offset + mapping.source_offset
        target_canonical = destination.canonical_offset + mapping.target_offset
        if (
            source.logical_name != destination.logical_name
            or source_canonical != target_canonical
        ):
            raise ValueError("Restart mapping changes canonical payload coordinates.")
    for destination in destinations:
        matching = tuple(
            sorted(
                (
                    item
                    for item in mappings
                    if item.target_shard_id == destination.shard_id
                ),
                key=lambda item: item.target_offset,
            )
        )
        expected = 0
        for mapping in matching:
            if mapping.target_offset != expected:
                relation = "overlap" if mapping.target_offset < expected else "hole"
                raise ValueError(
                    f"Restart mappings for shard {destination.shard_id!r} contain a {relation}."
                )
            expected += mapping.byte_count
        if expected != destination.byte_count:
            raise ValueError(
                f"Restart mappings for shard {destination.shard_id!r} contain a hole."
            )
    for source in sources:
        matching = tuple(
            sorted(
                (item for item in mappings if item.source_chunk_id == source.chunk_id),
                key=lambda item: item.source_offset,
            )
        )
        expected = 0
        for mapping in matching:
            if mapping.source_offset != expected:
                relation = "overlap" if mapping.source_offset < expected else "hole"
                raise ValueError(
                    f"Restart mappings for chunk {source.chunk_id!r} contain a {relation}."
                )
            expected += mapping.byte_count
        if expected != source.byte_count:
            raise ValueError(
                f"Restart mappings for chunk {source.chunk_id!r} contain a hole."
            )


def _support_id(value: SupportTuple | str, name: str, /) -> str:
    if isinstance(value, SupportTuple):
        return value.support_tuple_id
    if isinstance(value, str):
        return _digest(value, f"{name}_support_tuple_id")
    raise TypeError(f"{name} must be SupportTuple or a support-tuple ID.")


def _finite_nonnegative(value: float, name: str, /) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return normalized


def _positive(value: int, name: str, /) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _nonnegative(value: int, name: str, /) -> int:
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return normalized


__all__ = [
    "CanonicalRestartChunk",
    "ChunkRangeReader",
    "DestinationShard",
    "DestinationShardWriter",
    "DirectRestorePlan",
    "PayloadClass",
    "RestartAdmission",
    "RestartChunkMapping",
    "RestartClass",
    "RestartExecutionReport",
    "MeshingArrayRole",
    "MeshingCheckpointState",
    "TopologyRestartPolicy",
    "TopologyRestartRelation",
    "admit_topology_restart",
    "canonical_chunk_mapping",
    "execute_direct_restore",
    "prepare_direct_restore",
]
