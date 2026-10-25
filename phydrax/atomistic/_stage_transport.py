#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Root-group fixed-capacity transport of typed stage packets.

Stage groups are explicit `ExecutionGroup` children of one root group. A
prepared `AtomisticStageTransportPlan` owns the only cross-stage communication:
a fixed root-group schedule of three collectives per round, always executed in
the same order by every root process, even when nothing moves.

1. **Control all-gather.** Every process contributes one fixed-width row: its
   cancellation request, its driver's ``drained`` flag and, for every stage it
   belongs to, the stage's free inbox slots (credits) and its outbox headers.
   All members of a stage must report identical stage rows (else the job fails
   with `AtomisticTransportDivergenceError`). Every process derives the same
   grants from the gathered rows: per edge in plan order, outbox packets in
   sequence order are granted while the destination has credit and every
   fragment route has a free slot of the fixed per-lane-pair capacity. A packet
   that cannot be granted stops its edge (FIFO, no overtaking) and stays parked
   in its outbox, bitwise unchanged.
2. **Payload all-to-all.** Granted fragments travel in fixed slots
   ``(lane_slots, slot_bytes)`` per ordered lane pair; each slot carries its
   grant, fragment, kind, byte count and SHA-256 digest.
3. **Receipt all-gather.** Destination processes verify and decode every
   expected fragment (sink hosts also publish to their sink and obtain a
   durable receipt) and report one acknowledgement per grant. A grant commits
   only if every process acknowledged it: destination members then append the
   packet to their inbox and source members release it. Otherwise nothing
   changes anywhere (atomic group transfer); the packet is retried later.

Cancellation and termination are decided from the same control round: any
cancellation request cancels the schedule for every process; termination
requires every process drained and every outbox and inbox empty. A process
that disappears leaves its peers blocked inside a collective; the launcher /
supervisor fails the coupled job within its deadline and the run restarts from
its last durable global commit. No collective rollback after rank loss is
promised.

Packets are `AtomisticTaskPacket` values encoded by a per-edge
`AtomisticPacketCodec` (fixed byte schema of a template packet; a domain
continuation's task identity travels in the packet header). ``replicated``
edges send the source stage coordinator's packet to every destination member;
``process-local`` edges send one fragment per source member process (for
example an `AtomisticDomainPacket` continuation of that process's owner
blocks) to destination member ``k mod |P_d|``; destination members without a
row fragment receive the header-only fragment
(`AtomisticDomainPacket.without_rows`). Only endpoint processes hold an edge's
template; preparation agrees every edge's schema and slot size over the root
group. Child stage collectives are entered only through
`PreparedAtomisticStageTransport.stage_group`, which refuses non-members.

Stage×domain campaigns (`AtomisticDomainCampaignPlan`) execute native
distributed domain dynamics on each stage's child group between rounds:
the entry stage ingresses tasks process-locally, every stage advances its one
resident task by a jitted segment of `PreparedAtomisticDomainDynamics`
transactions, a finished visit leaves as the members' `to_packet` fragments
(one atomic group transfer, re-ingressed with `from_packet` and primed by the
next stage) or as a replicated report published to an
`AbstractAtomisticCampaignSink` by semantic output key, and a full outbox
parks the finished resident without new physics. `checkpoint` commits the
transport state, campaign controller, identities and every resident's
complete domain state (plus its packet) with one lifecycle global commit,
leaving the live state unchanged; `restore` resumes bitwise on the same
topology and `migrate` re-fragments every pending continuation onto a
changed topology. Rank loss is fail-coupled by the launcher.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import signal
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import final, Literal, Protocol, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from .._execution_resources import ExecutionGroupSpec
from .._execution_runtime import bind_execution_group, ExecutionGroup
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint, canonical_json
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..backends.distributed import JaxCollectiveProvider
from ..discretization.spatial._distributed_relations import DistributedOwnershipPlan
from ..lifecycle._chunk_repository import ArtifactRepository
from ..lifecycle._distributed_checkpoint import (
    assemble_distributed_checkpoint_from_repository,
    commit_distributed_checkpoint,
    load_distributed_checkpoint_commit,
    publish_process_checkpoint,
    read_checkpoint_region,
    restore_global_array_from_checkpoint,
)
from ..lifecycle._models import CheckpointManifest
from ..typing import parse
from ._campaign import (
    _conditions_id,
    _task_output,
    AbstractAtomisticCampaignSink,
    AtomisticCampaignOutput,
    AtomisticCampaignTask,
    AtomisticEvaluationKey,
    AtomisticEvaluationRecord,
    AtomisticStageOutcome,
    AtomisticTaskPacket,
)
from ._domain_checkpoint import _flat_state, _leaf_data, _structure_id
from ._domain_dynamics import (
    AtomisticDomainConstraintRows,
    PreparedAtomisticDomainDynamics,
)
from ._domain_runtime import AtomisticDomainPacket
from ._domain_state import AtomisticDomainState
from ._dynamics import CAPACITY_REJECTIONS


TransportMode = Literal["replicated", "process-local"]

_PACKET_FIELDS = (
    "positions",
    "cell_vectors",
    "momenta",
    "species",
    "atom_type_ids",
    "atom_mask",
    "path",
    "path_energies",
    "time",
    "evaluation",
    "continuation",
)
_STATIC_HEADER_BYTES = 512
_ROWS = 0
_HEADER_ONLY = 1
# Slot/outbox header: present, grant-or-edge, sequence, fragment, kind,
# byte_count, then the SHA-256 digest as four little-endian int64 words.
_HEADER_WIDTH = 10


class AtomisticTransportCapacityError(ValueError):
    """A fixed transport capacity (outbox, slots, rounds) refused the request.

    Capacity refusal is distinct from scientific failure: nothing was lost or
    overwritten, and the refused packet remains with its producer.
    """


class AtomisticTransportSchemaError(ValueError):
    """A packet does not match its edge's fixed byte schema."""


class AtomisticTransportDivergenceError(RuntimeError):
    """Members of one stage reported different transport state; fail the job."""


class AtomisticTransportClosedError(RuntimeError):
    """The schedule was cancelled or terminated by consensus."""


class AtomisticTransportSink(Protocol):
    """Durable, idempotent publication target hosted by the root coordinator.

    ``publish`` returns a durable receipt identifier or ``None`` for
    backpressure (nothing durable). ``receipt`` looks up an earlier durable
    publication of the same semantic key for idempotent replay.
    """

    @property
    def sink_id(self) -> str: ...

    def publish(self, key: str, packet: AtomisticTaskPacket, /) -> str | None: ...

    def receipt(self, key: str, /) -> str | None: ...


def _digest_words(payload: np.ndarray, /) -> np.ndarray:
    digest = hashlib.sha256(payload.tobytes()).digest()
    return np.frombuffer(digest, dtype="<i8").astype(np.int64)


def _identity_words(identity: str, /) -> np.ndarray:
    """Four int64 words of a schema/identity string (consensus rows)."""
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return np.frombuffer(digest, dtype="<i8").astype(np.int64)


def _stage_schema(schema_id: str, /) -> str:
    """A stage's root-agreed continuation schema label (`_agree_stages`).

    Campaign checkpoints hold ``_identity_words`` of this label, not of the
    codec's ``schema_id`` itself.
    """
    return _identity_words(schema_id).tobytes().hex()


def _domain_packet(
    packet: AtomisticDomainPacket,
    /,
    *,
    header_id: str,
    run_id: str,
    logical_count: int,
) -> AtomisticDomainPacket:
    return AtomisticDomainPacket(
        rows=packet.rows,
        atom_payload=packet.atom_payload,
        cell_vectors=packet.cell_vectors,
        replicated_payload=packet.replicated_payload,
        controllers=packet.controllers,
        ledger=packet.ledger,
        conditioning=packet.conditioning,
        constraints=packet.constraints,
        rng_key_data=packet.rng_key_data,
        case_identity_words=packet.case_identity_words,
        step_index=packet.step_index,
        time=packet.time,
        force_record=packet.force_record,
        key_implementation=packet.key_implementation,
        logical_count=logical_count,
        role_ids=packet.role_ids,
        model_revision_id=packet.model_revision_id,
        run_id=run_id,
        header_id=header_id,
    )


def _packet_body(
    packet: AtomisticTaskPacket, /
) -> tuple[tuple[object, ...], dict[str, str | int] | None]:
    """Packet leaves plus the per-packet identity of a domain continuation.

    A domain continuation's ``header_id``, ``run_id`` and ``logical_count``
    identify one task's state rather than the edge schema; they travel in the
    canonical JSON header and are canonicalized out of the tree structure.
    """
    body = {name: getattr(packet, name) for name in _PACKET_FIELDS}
    continuation = body["continuation"]
    identity: dict[str, str | int] | None = None
    if isinstance(continuation, AtomisticDomainPacket):
        identity = {
            "header_id": continuation.header_id,
            "run_id": continuation.run_id,
            "logical_count": continuation.logical_count,
        }
        body["continuation"] = _domain_packet(
            continuation, header_id="", run_id="", logical_count=0
        )
    return tuple(body[name] for name in _PACKET_FIELDS), identity


@final
class AtomisticPacketCodec(StrictModule, NonTrainableState):
    """Fixed byte schema of `AtomisticTaskPacket` values shaped like a template.

    Array leaves (including the evaluation record and the continuation) must
    match the template's tree structure, shapes and dtypes exactly, so static
    metadata of ``evaluation`` and ``continuation`` is an edge constant. The
    per-packet statics ``child_id``/``family`` and, for an
    `AtomisticDomainPacket` continuation, its task identity (``header_id``,
    ``run_id``, ``logical_count``) travel in a canonical JSON header.
    Decoding reconstructs the packet through its constructor.
    """

    treedef: jax.tree_util.PyTreeDef = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)
    byte_count: int = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(self, template: AtomisticTaskPacket, /) -> None:
        if not isinstance(template, AtomisticTaskPacket):
            raise TypeError("template must be an AtomisticTaskPacket.")
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(
            _packet_body(template)[0]
        )
        paths: list[str] = []
        shapes: list[tuple[int, ...]] = []
        dtypes: list[str] = []
        body = 0
        for path, leaf in path_leaves:
            if isinstance(leaf, jax.Array):
                raise TypeError("Transport packets hold host NumPy leaves only.")
            array = np.asarray(leaf)
            if array.dtype.hasobject or array.dtype.kind not in "biuf":
                raise TypeError(
                    "Transport packet leaves must be boolean, integer or floating."
                )
            paths.append(jax.tree_util.keystr(path))
            shapes.append(tuple(int(extent) for extent in array.shape))
            dtypes.append(array.dtype.str)
            body += array.nbytes
        self.treedef = treedef
        self.paths = tuple(paths)
        self.shapes = tuple(shapes)
        self.dtypes = tuple(dtypes)
        self.byte_count = _STATIC_HEADER_BYTES + body
        self.schema_id = canonical_fingerprint(
            {
                "kind": "atomistic-transport-packet-schema",
                "fields": list(_PACKET_FIELDS),
                "paths": list(self.paths),
                "shapes": [list(shape) for shape in self.shapes],
                "dtypes": list(self.dtypes),
                "static_header_bytes": _STATIC_HEADER_BYTES,
            }
        )

    def encode(self, packet: AtomisticTaskPacket, /) -> np.ndarray:
        """Canonical ``uint8[byte_count]`` encoding of one packet."""
        if not isinstance(packet, AtomisticTaskPacket):
            raise TypeError("packet must be an AtomisticTaskPacket.")
        body, identity = _packet_body(packet)
        leaves, treedef = jax.tree_util.tree_flatten(body)
        if treedef != self.treedef:
            raise AtomisticTransportSchemaError(
                "The packet structure differs from its edge schema."
            )
        header = canonical_json(
            {
                "child_id": packet.child_id,
                "family": packet.family,
                "continuation": identity,
            }
        ).encode("utf-8")
        if len(header) > _STATIC_HEADER_BYTES - 4:
            raise AtomisticTransportSchemaError(
                "Packet identity metadata exceeds the static header capacity."
            )
        encoded = np.zeros((self.byte_count,), dtype=np.uint8)
        encoded[:4] = np.asarray([len(header)], dtype="<u4").view(np.uint8)
        encoded[4 : 4 + len(header)] = np.frombuffer(header, dtype=np.uint8)
        offset = _STATIC_HEADER_BYTES
        for leaf, shape, dtype in zip(leaves, self.shapes, self.dtypes, strict=True):
            array = np.asarray(leaf)
            if array.shape != shape or array.dtype.str != dtype:
                raise AtomisticTransportSchemaError(
                    "A packet leaf shape or dtype differs from its edge schema."
                )
            raw = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
            encoded[offset : offset + raw.size] = raw
            offset += raw.size
        return encoded

    def decode(self, encoded: np.ndarray, /) -> AtomisticTaskPacket:
        """Reconstruct and validate one packet from its canonical encoding."""
        data = np.asarray(encoded)
        if data.dtype != np.uint8 or data.shape != (self.byte_count,):
            raise AtomisticTransportSchemaError("Encoded packet has the wrong size.")
        length = int(data[:4].view("<u4")[0])
        if length > _STATIC_HEADER_BYTES - 4:
            raise AtomisticTransportSchemaError("Encoded packet header is invalid.")
        try:
            statics = json.loads(bytes(data[4 : 4 + length]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AtomisticTransportSchemaError(
                "Encoded packet header is invalid."
            ) from error
        if (
            not isinstance(statics, dict)
            or set(statics) != {"child_id", "family", "continuation"}
            or any(
                statics[name] is not None and not isinstance(statics[name], str)
                for name in ("child_id", "family")
            )
        ):
            raise AtomisticTransportSchemaError("Encoded packet header is invalid.")
        identity = statics["continuation"]
        if identity is not None and (
            not isinstance(identity, dict)
            or set(identity) != {"header_id", "run_id", "logical_count"}
            or not isinstance(identity["header_id"], str)
            or not isinstance(identity["run_id"], str)
            or type(identity["logical_count"]) is not int
            or identity["logical_count"] < 0
        ):
            raise AtomisticTransportSchemaError(
                "Encoded continuation identity is invalid."
            )
        leaves = []
        offset = _STATIC_HEADER_BYTES
        for shape, dtype_text in zip(self.shapes, self.dtypes, strict=True):
            dtype = np.dtype(dtype_text)
            count = dtype.itemsize * int(np.prod(shape, dtype=np.int64))
            leaves.append(data[offset : offset + count].copy().view(dtype).reshape(shape))
            offset += count
        body = jax.tree_util.tree_unflatten(self.treedef, leaves)
        fields = dict(zip(_PACKET_FIELDS, body, strict=True))
        evaluation = fields["evaluation"]
        if evaluation is not None and not isinstance(
            evaluation, AtomisticEvaluationRecord
        ):
            raise AtomisticTransportSchemaError("Decoded evaluation is invalid.")
        continuation = fields["continuation"]
        if isinstance(continuation, AtomisticDomainPacket) != (identity is not None):
            raise AtomisticTransportSchemaError(
                "A continuation identity accompanies exactly a domain continuation."
            )
        if isinstance(continuation, AtomisticDomainPacket) and identity is not None:
            fields["continuation"] = _domain_packet(
                continuation,
                header_id=identity["header_id"],
                run_id=identity["run_id"],
                logical_count=identity["logical_count"],
            )
        return AtomisticTaskPacket(
            **fields, child_id=statics["child_id"], family=statics["family"]
        )


def _header_only(packet: AtomisticTaskPacket, /) -> AtomisticTaskPacket:
    continuation = packet.continuation
    if not isinstance(continuation, AtomisticDomainPacket):
        raise AtomisticTransportSchemaError(
            "Process-local edges carry AtomisticDomainPacket continuations."
        )
    return AtomisticTaskPacket(
        positions=packet.positions,
        cell_vectors=packet.cell_vectors,
        momenta=packet.momenta,
        species=packet.species,
        atom_type_ids=packet.atom_type_ids,
        atom_mask=packet.atom_mask,
        path=packet.path,
        path_energies=packet.path_energies,
        time=packet.time,
        evaluation=packet.evaluation,
        continuation=continuation.without_rows(),
        child_id=packet.child_id,
        family=packet.family,
    )


@final
class AtomisticTransportStage(StrictModule, NonTrainableState):
    """One named stage bound to an explicit child of the root execution group."""

    name: str = eqx.field(static=True)
    group: ExecutionGroupSpec = eqx.field(static=True)

    def __init__(self, name: str, group: ExecutionGroupSpec, /) -> None:
        identifier = str(name).strip()
        if not identifier:
            raise ValueError("Stage names must be non-empty.")
        if not isinstance(group, ExecutionGroupSpec):
            raise TypeError("group must be an ExecutionGroupSpec.")
        if group.parent_group_id is None:
            raise ValueError("Stage groups must be explicit children of a root group.")
        self.name = identifier
        self.group = group


@final
class AtomisticTransportEdge(StrictModule, NonTrainableState):
    """Directed transfer of one packet schema from a stage to a stage or sink.

    ``template`` fixes the packet schema on processes that encode or decode
    the edge (source members, destination members or the sink host). A
    process that is neither passes ``template=None`` with the edge's declared
    ``layout_id``; the prepared transport agrees the schema and slot size of
    every edge across the root group. Without ``layout_id`` the layout is the
    template's schema (every process then supplies the template).
    """

    source: str = eqx.field(static=True)
    destination: str = eqx.field(static=True)
    mode: TransportMode = eqx.field(static=True)
    codec: AtomisticPacketCodec | None
    header_codec: AtomisticPacketCodec | None
    layout_id: str = eqx.field(static=True)
    edge_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: str,
        destination: str,
        template: AtomisticTaskPacket | None,
        /,
        *,
        mode: TransportMode = "replicated",
        layout_id: str | None = None,
    ) -> None:
        mode = parse(mode, TransportMode, "mode")
        codec = None if template is None else AtomisticPacketCodec(template)
        header = (
            AtomisticPacketCodec(_header_only(template))
            if template is not None and mode == "process-local"
            else None
        )
        if layout_id is None:
            if codec is None:
                raise ValueError("An edge without a local template declares layout_id.")
            layout = canonical_fingerprint(
                {
                    "kind": "atomistic-transport-template-layout",
                    "schema": codec.schema_id,
                    "header_schema": None if header is None else header.schema_id,
                }
            )
        else:
            layout = str(layout_id).strip()
            if not layout:
                raise ValueError("layout_id must be non-empty.")
        self.source = str(source)
        self.destination = str(destination)
        self.mode = mode
        self.codec = codec
        self.header_codec = header
        self.layout_id = layout
        self.edge_id = canonical_fingerprint(
            {
                "kind": "atomistic-transport-edge",
                "source": self.source,
                "destination": self.destination,
                "mode": mode,
                "layout": layout,
            }
        )


def _positive(name: str, value: int, /) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@final
class AtomisticStageTransportPlan(StrictModule, NonTrainableState):
    """Stages, sinks, edges and the fixed capacities of one transport schedule."""

    stages: tuple[AtomisticTransportStage, ...]
    edges: tuple[AtomisticTransportEdge, ...]
    sinks: tuple[str, ...] = eqx.field(static=True)
    outbox_capacity: int = eqx.field(static=True)
    inbox_capacity: int = eqx.field(static=True)
    lane_slots: int = eqx.field(static=True)
    maximum_rounds: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stages: Sequence[AtomisticTransportStage],
        edges: Sequence[AtomisticTransportEdge],
        /,
        *,
        sinks: Sequence[str] = (),
        outbox_capacity: int,
        inbox_capacity: int,
        lane_slots: int,
        maximum_rounds: int,
    ) -> None:
        stages_ = tuple(stages)
        edges_ = tuple(edges)
        sinks_ = tuple(str(name) for name in sinks)
        if not stages_ or not all(
            isinstance(stage, AtomisticTransportStage) for stage in stages_
        ):
            raise TypeError("stages must be non-empty AtomisticTransportStage values.")
        if not all(isinstance(edge, AtomisticTransportEdge) for edge in edges_):
            raise TypeError("edges must be AtomisticTransportEdge values.")
        names = tuple(stage.name for stage in stages_)
        if len(set(names + sinks_)) != len(names) + len(sinks_):
            raise ValueError("Stage and sink names must be distinct.")
        parents = {stage.group.parent_group_id for stage in stages_}
        if len(parents) != 1:
            raise ValueError("Every stage group must be a child of the same root.")
        for edge in edges_:
            if edge.source not in names:
                raise ValueError(f"Edge source {edge.source!r} is not a stage.")
            if edge.destination not in names and edge.destination not in sinks_:
                raise ValueError(
                    f"Edge destination {edge.destination!r} is not a stage or sink."
                )
            if edge.destination in sinks_ and edge.mode != "replicated":
                raise ValueError("Sink edges transfer replicated packets.")
        if len({(edge.source, edge.destination) for edge in edges_}) != len(edges_):
            raise ValueError("Each (source, destination) pair has one edge.")
        self.stages = stages_
        self.edges = edges_
        self.sinks = sinks_
        self.outbox_capacity = _positive("outbox_capacity", outbox_capacity)
        self.inbox_capacity = _positive("inbox_capacity", inbox_capacity)
        self.lane_slots = _positive("lane_slots", lane_slots)
        self.maximum_rounds = _positive("maximum_rounds", maximum_rounds)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-stage-transport-plan",
                "stages": [[stage.name, stage.group.to_payload()] for stage in stages_],
                "edges": [edge.edge_id for edge in edges_],
                "sinks": list(sinks_),
                "outbox_capacity": self.outbox_capacity,
                "inbox_capacity": self.inbox_capacity,
                "lane_slots": self.lane_slots,
                "maximum_rounds": self.maximum_rounds,
            }
        )

    def prepare(
        self, root: ExecutionGroup, /, *, maximum_round_bytes: int
    ) -> PreparedAtomisticStageTransport:
        """Bind the schedule to the live root group and admit its round memory."""
        return PreparedAtomisticStageTransport(
            self, root, maximum_round_bytes=maximum_round_bytes
        )


def process_subset_group_spec(
    parent: ExecutionGroupSpec,
    processes: Sequence[int],
    /,
    *,
    mesh_axis: str = "device",
) -> ExecutionGroupSpec:
    """Explicit child of ``parent`` holding every device of ``processes``.

    The complement of `partition_execution_group_specs` (process-symmetric
    splits): a stage group on a process subset, so non-member processes hold
    no device of the child mesh and never enter its collectives.
    """
    members = tuple(sorted({int(process) for process in processes}))
    if not members or not set(members) <= set(parent.process_indices):
        raise ValueError("Child processes must be a non-empty subset of the parent.")
    device_keys = tuple(key for key in parent.device_keys if key[0] in members)
    hosts = tuple(
        (process, host) for process, host in parent.process_host_ids if process in members
    )
    group_id = canonical_fingerprint(
        {
            "kind": "process-subset-execution-group",
            "parent_group_id": parent.group_id,
            "processes": list(members),
            "device_keys": [list(key) for key in device_keys],
            "mesh_axis": mesh_axis,
        }
    )
    return ExecutionGroupSpec(
        group_id,
        members,
        device_keys,
        mesh_axes=((mesh_axis, len(device_keys)),),
        parent_group_id=parent.group_id,
        process_host_ids=hosts,
    )


@final
class AtomisticStageTransportState(StrictModule):
    """Per-process host transport state of fixed capacity.

    ``outbox_*`` hold this process's packet or fragment of each parked
    outbound packet of every stage it belongs to; ``inbox_*`` hold committed
    inbound packets (up to ``fragment_capacity`` fragments per process) with
    their commit order. All members of a stage hold identical headers.
    ``transfer_digest`` chains every committed transfer and is identical on
    every process.
    """

    outbox_header: np.ndarray
    outbox_bytes: np.ndarray
    inbox_header: np.ndarray
    inbox_bytes: np.ndarray
    inbox_order: np.ndarray
    next_sequence: np.ndarray
    next_order: np.ndarray
    published: np.ndarray
    round_index: np.ndarray
    flags: np.ndarray
    transfer_digest: np.ndarray
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        outbox_header: np.ndarray,
        outbox_bytes: np.ndarray,
        inbox_header: np.ndarray,
        inbox_bytes: np.ndarray,
        inbox_order: np.ndarray,
        next_sequence: np.ndarray,
        next_order: np.ndarray,
        published: np.ndarray,
        round_index: np.ndarray,
        flags: np.ndarray,
        transfer_digest: np.ndarray,
        prepared_id: str,
    ) -> None:
        arrays = {
            "outbox_header": (outbox_header, np.int64, 3),
            "outbox_bytes": (outbox_bytes, np.uint8, 3),
            "inbox_header": (inbox_header, np.int64, 4),
            "inbox_bytes": (inbox_bytes, np.uint8, 4),
            "inbox_order": (inbox_order, np.int64, 2),
            "next_sequence": (next_sequence, np.int64, 1),
            "next_order": (next_order, np.int64, 1),
            "published": (published, np.int64, 1),
            "round_index": (round_index, np.int64, 0),
            "flags": (flags, np.bool_, 1),
            "transfer_digest": (transfer_digest, np.uint8, 1),
        }
        validated = {}
        for name, (value, dtype, rank) in arrays.items():
            array = np.asarray(value)
            if array.dtype != dtype or array.ndim != rank:
                raise TypeError(f"{name} must be a rank-{rank} {np.dtype(dtype)} array.")
            validated[name] = array
        if validated["flags"].shape != (2,) or validated["transfer_digest"].shape != (
            32,
        ):
            raise ValueError("Transport flags or digest have the wrong shape.")
        self.outbox_header = validated["outbox_header"]
        self.outbox_bytes = validated["outbox_bytes"]
        self.inbox_header = validated["inbox_header"]
        self.inbox_bytes = validated["inbox_bytes"]
        self.inbox_order = validated["inbox_order"]
        self.next_sequence = validated["next_sequence"]
        self.next_order = validated["next_order"]
        self.published = validated["published"]
        self.round_index = validated["round_index"]
        self.flags = validated["flags"]
        self.transfer_digest = validated["transfer_digest"]
        self.prepared_id = str(prepared_id)

    @property
    def cancelled(self) -> bool:
        return bool(self.flags[0])

    @property
    def terminated(self) -> bool:
        return bool(self.flags[1])

    def replace(self, **changes: np.ndarray) -> AtomisticStageTransportState:
        """A new state with selected arrays replaced (validated again)."""
        fields = {
            name: getattr(self, name)
            for name in (
                "outbox_header",
                "outbox_bytes",
                "inbox_header",
                "inbox_bytes",
                "inbox_order",
                "next_sequence",
                "next_order",
                "published",
                "round_index",
                "flags",
                "transfer_digest",
            )
        }
        unknown = set(changes) - set(fields)
        if unknown:
            raise KeyError(f"Unknown transport state fields: {sorted(unknown)}.")
        fields.update(changes)
        return AtomisticStageTransportState(**fields, prepared_id=self.prepared_id)


_STATE_FIELDS = (
    "outbox_header",
    "outbox_bytes",
    "inbox_header",
    "inbox_bytes",
    "inbox_order",
    "next_sequence",
    "next_order",
    "published",
    "round_index",
    "flags",
    "transfer_digest",
)


@final
class AtomisticTransportDelivery(StrictModule):
    """One committed inbound packet as delivered to this process.

    ``fragments`` are the packets routed to this process in fragment order;
    ``header_only`` marks header-only domain fragments.
    """

    fragments: tuple[AtomisticTaskPacket, ...]
    header_only: tuple[bool, ...] = eqx.field(static=True)
    stage: str = eqx.field(static=True)
    source: str = eqx.field(static=True)
    sequence: int = eqx.field(static=True)
    order: int = eqx.field(static=True)


@final
class AtomisticTransportRoundEvidence(StrictModule, NonTrainableState):
    """Consensus outcome of one scheduled round (identical on every process)."""

    round_index: int = eqx.field(static=True)
    offered: int = eqx.field(static=True)
    granted: int = eqx.field(static=True)
    committed: int = eqx.field(static=True)
    parked: int = eqx.field(static=True)
    refused: int = eqx.field(static=True)
    credits: tuple[int, ...] = eqx.field(static=True)
    cancelled: bool = eqx.field(static=True)
    terminated: bool = eqx.field(static=True)
    receipts: tuple[str, ...] = eqx.field(static=True)
    transfer_digest: str = eqx.field(static=True)


@final
class AtomisticTransportRound(StrictModule):
    state: AtomisticStageTransportState
    evidence: AtomisticTransportRoundEvidence


type _Grant = tuple[int, int, int, int]
"""(edge index, source stage index, outbox slot, sequence)."""

type _Assignment = tuple[int, tuple[int, int, int, int], int]
"""(grant position, route (source, destination, fragment, kind), lane slot)."""

type _Staged = dict[int, list[tuple[int, int, np.ndarray, np.ndarray]]]
"""Received fragments by grant position: (fragment, kind, header, payload)."""


@functools.partial(jax.jit, static_argnums=(0, 1))
def _all_gather_rows(mesh: Mesh, axis: str, rows: jax.Array) -> jax.Array:
    def body(local: jax.Array) -> jax.Array:
        return JaxCollectiveProvider(axis).all_gather(local, axis=0, tiled=True)

    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=PartitionSpec(axis),
        out_specs=PartitionSpec(),
        check_vma=False,
    )(rows)


def _root_gather(
    root: ExecutionGroup,
    axis: str,
    processes: tuple[int, ...],
    lanes: tuple[int, ...],
    row: np.ndarray,
    /,
) -> np.ndarray:
    """All-gather one int64 row per process over the root group (lane devices).

    A root-group collective: every root process enters it in the same order.
    """
    width = row.shape[0]
    me = jax.process_index()
    lane = lanes[processes.index(me)]
    blocks = [
        jax.device_put((row if index == lane else np.zeros_like(row))[None, :], device)
        for index, device in enumerate(root.devices)
        if device.process_index == me
    ]
    rows = jax.make_array_from_single_device_arrays(
        (len(root.devices), width),
        NamedSharding(root.mesh, PartitionSpec(axis)),
        blocks,
    )
    gathered = np.asarray(_all_gather_rows(root.mesh, axis, rows).addressable_data(0))
    return gathered[list(lanes)]


_SCHEMA_WIDTH = 11


def _agree_schemas(
    plan: AtomisticStageTransportPlan,
    root: ExecutionGroup,
    axis: str,
    processes: tuple[int, ...],
    lanes: tuple[int, ...],
    destinations: tuple[tuple[int, ...], ...],
    sources: tuple[tuple[int, ...], ...],
    /,
) -> tuple[
    tuple[AtomisticPacketCodec | None, ...],
    tuple[AtomisticPacketCodec | None, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[str, ...],
]:
    """Root-group consensus of every edge's packet schema and slot size.

    Each process reports, per edge, whether it holds the schema, its byte
    counts and schema digests. Every endpoint (source member, destination
    member or sink host) must hold the schema and all holders must agree;
    otherwise every process raises the same `AtomisticTransportSchemaError`.
    Non-endpoints keep no codec and adopt the agreed byte counts.
    """
    row = np.zeros((_SCHEMA_WIDTH * len(plan.edges),), np.int64)
    for index, edge in enumerate(plan.edges):
        if edge.codec is None:
            continue
        base = _SCHEMA_WIDTH * index
        header = edge.header_codec
        row[base : base + 3] = (
            1,
            edge.codec.byte_count,
            0 if header is None else header.byte_count,
        )
        row[base + 3 : base + 7] = _identity_words(edge.codec.schema_id)
        if header is not None:
            row[base + 7 : base + 11] = _identity_words(header.schema_id)
    rows = _root_gather(root, axis, processes, lanes, row)
    byte_counts: list[int] = []
    header_counts: list[int] = []
    schema_ids: list[str] = []
    for index, edge in enumerate(plan.edges):
        segment = rows[:, _SCHEMA_WIDTH * index : _SCHEMA_WIDTH * (index + 1)]
        holders = [
            position for position in range(len(processes)) if segment[position, 0] == 1
        ]
        endpoints = set(sources[index]) | set(destinations[index])
        missing = sorted(
            process for process in endpoints if processes.index(process) not in holders
        )
        if missing:
            raise AtomisticTransportSchemaError(
                f"Endpoint processes {missing} of edge {edge.source!r} -> "
                f"{edge.destination!r} hold no packet schema."
            )
        reference = segment[holders[0]]
        if any(not np.array_equal(segment[p], reference) for p in holders):
            raise AtomisticTransportSchemaError(
                f"Processes disagree on the packet schema of edge {edge.source!r} -> "
                f"{edge.destination!r}."
            )
        byte_counts.append(int(reference[1]))
        header_counts.append(int(reference[2]))
        schema_ids.append(np.ascontiguousarray(reference[3:]).tobytes().hex())
    return (
        tuple(edge.codec for edge in plan.edges),
        tuple(edge.header_codec for edge in plan.edges),
        tuple(byte_counts),
        tuple(header_counts),
        tuple(schema_ids),
    )


@eqx.filter_jit
def _exchange_slots(
    prepared: PreparedAtomisticStageTransport,
    payload: jax.Array,
    headers: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    axis = prepared.axis_name

    def body(
        local_payload: jax.Array, local_headers: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        provider = JaxCollectiveProvider(axis)
        return (
            provider.all_to_all(local_payload, split_axis=0, concat_axis=0),
            provider.all_to_all(local_headers, split_axis=0, concat_axis=0),
        )

    return jax.shard_map(
        body,
        mesh=prepared.root.mesh,
        in_specs=(PartitionSpec(axis), PartitionSpec(axis)),
        out_specs=(PartitionSpec(axis), PartitionSpec(axis)),
        check_vma=False,
    )(payload, headers)


@final
class PreparedAtomisticStageTransport(StrictModule, NonTrainableState):
    """A transport plan bound to the live root group.

    One transport lane per process (its first root device) carries every
    packet; other root devices participate in every collective with empty
    blocks. ``lane_mesh`` (one lane device per process) also hosts the
    process-blocked checkpoint form of the per-process state.
    """

    plan: AtomisticStageTransportPlan
    root: ExecutionGroup = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    processes: tuple[int, ...] = eqx.field(static=True)
    lanes: tuple[int, ...] = eqx.field(static=True)
    lane_mesh: Mesh = eqx.field(static=True)
    stage_processes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    routes: tuple[tuple[tuple[int, int, int, int], ...], ...] = eqx.field(static=True)
    sink_host: int = eqx.field(static=True)
    codecs: tuple[AtomisticPacketCodec | None, ...]
    header_codecs: tuple[AtomisticPacketCodec | None, ...]
    schema_ids: tuple[str, ...] = eqx.field(static=True)
    fragment_capacity: int = eqx.field(static=True)
    slot_bytes: int = eqx.field(static=True)
    grant_capacity: int = eqx.field(static=True)
    control_width: int = eqx.field(static=True)
    round_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: AtomisticStageTransportPlan,
        root: ExecutionGroup,
        /,
        *,
        maximum_round_bytes: int,
    ) -> None:
        if not isinstance(plan, AtomisticStageTransportPlan):
            raise TypeError("plan must be an AtomisticStageTransportPlan.")
        if not isinstance(root, ExecutionGroup):
            raise TypeError("root must be a live ExecutionGroup.")
        axes = tuple(root.mesh.axis_names)
        if len(axes) != 1:
            raise ValueError("The transport root group requires a one-dimensional mesh.")
        root_keys = set(root.spec.device_keys)
        for stage in plan.stages:
            if stage.group.parent_group_id != root.spec.group_id:
                raise ValueError(f"Stage {stage.name!r} is not a child of this root.")
            if not set(stage.group.device_keys) <= root_keys:
                raise ValueError(f"Stage {stage.name!r} uses devices outside the root.")
        processes = tuple(sorted(root.spec.process_indices))
        lanes = tuple(
            min(
                index
                for index, device in enumerate(root.devices)
                if device.process_index == process
            )
            for process in processes
        )
        stage_processes = tuple(
            tuple(sorted(stage.group.process_indices)) for stage in plan.stages
        )
        names = tuple(stage.name for stage in plan.stages)
        sink_host = processes[0]
        routes: list[tuple[tuple[int, int, int, int], ...]] = []
        for edge in plan.edges:
            source = stage_processes[names.index(edge.source)]
            destination = (
                stage_processes[names.index(edge.destination)]
                if edge.destination in names
                else (sink_host,)
            )
            if edge.mode == "replicated":
                edge_routes = tuple(
                    (source[0], process, 0, _ROWS) for process in destination
                )
            else:
                edge_routes = tuple(
                    (process, destination[index % len(destination)], index, _ROWS)
                    for index, process in enumerate(source)
                ) + tuple(
                    (source[index % len(source)], destination[index], index, _HEADER_ONLY)
                    for index in range(len(source), len(destination))
                )
            routes.append(edge_routes)
        fragment_capacity = max(
            [1]
            + [
                sum(1 for route in edge_routes if route[1] == process)
                for edge_routes in routes
                for process in processes
            ]
        )
        codecs, header_codecs, byte_counts, header_counts, schema_ids = _agree_schemas(
            plan,
            root,
            str(axes[0]),
            processes,
            lanes,
            tuple(
                stage_processes[names.index(edge.destination)]
                if edge.destination in names
                else (sink_host,)
                for edge in plan.edges
            ),
            tuple(stage_processes[names.index(edge.source)] for edge in plan.edges),
        )
        slot_bytes = max([1, *byte_counts, *header_counts])
        device_count = len(root.devices)
        round_bytes = (
            device_count
            * device_count
            * plan.lane_slots
            * (slot_bytes + 8 * _HEADER_WIDTH)
        )
        if type(maximum_round_bytes) is not int or maximum_round_bytes <= 0:
            raise ValueError("maximum_round_bytes must be a positive integer.")
        if round_bytes > maximum_round_bytes:
            raise AtomisticTransportCapacityError(
                f"One transport round needs {round_bytes} bytes, above the admitted "
                f"{maximum_round_bytes}."
            )
        grant_capacity = len(plan.stages) * plan.outbox_capacity
        self.plan = plan
        self.root = root
        self.axis_name = str(axes[0])
        self.processes = processes
        self.lanes = lanes
        self.lane_mesh = Mesh(
            np.asarray([root.devices[lane] for lane in lanes], dtype=object),
            ("process",),
        )
        self.stage_processes = stage_processes
        self.routes = tuple(routes)
        self.sink_host = sink_host
        self.codecs = codecs
        self.header_codecs = header_codecs
        self.schema_ids = schema_ids
        self.fragment_capacity = fragment_capacity
        self.slot_bytes = slot_bytes
        self.grant_capacity = grant_capacity
        self.control_width = 3 + len(plan.stages) * (2 + 3 * plan.outbox_capacity)
        self.round_bytes = round_bytes
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-stage-transport",
                "plan_id": plan.plan_id,
                "root_group_id": root.spec.group_id,
                "lanes": list(lanes),
                "slot_bytes": slot_bytes,
                "fragment_capacity": fragment_capacity,
                "schemas": list(schema_ids),
            }
        )

    # -- codecs -------------------------------------------------------------

    def codec(self, edge_index: int, kind: int = _ROWS, /) -> AtomisticPacketCodec:
        """This process's agreed codec of an edge it encodes or decodes."""
        codec = (
            self.codecs[edge_index] if kind == _ROWS else self.header_codecs[edge_index]
        )
        if codec is None:
            raise AtomisticTransportSchemaError(
                f"Process {jax.process_index()} holds no schema of edge "
                f"{self.plan.edges[edge_index].edge_id}."
            )
        return codec

    def edge_index(self, source: str, destination: str, /) -> int:
        for index, edge in enumerate(self.plan.edges):
            if edge.source == source and edge.destination == destination:
                return index
        raise KeyError(f"No transport edge {source!r} -> {destination!r}.")

    # -- membership ---------------------------------------------------------

    def _stage_index(self, name: str, /) -> int:
        for index, stage in enumerate(self.plan.stages):
            if stage.name == name:
                return index
        raise KeyError(f"Unknown transport stage {name!r}.")

    def is_member(self, stage: str, /) -> bool:
        return jax.process_index() in self.stage_processes[self._stage_index(stage)]

    def stage_group(self, stage: str, /) -> ExecutionGroup:
        """Bind a stage's child group; non-members are refused.

        Child collectives are entered only through this binding, so a
        non-participant can never enter an incompatible child collective.
        """
        index = self._stage_index(stage)
        if jax.process_index() not in self.stage_processes[index]:
            raise ValueError(
                f"Process {jax.process_index()} is not a member of {stage!r}."
            )
        return bind_execution_group(self.plan.stages[index].group)

    # -- state --------------------------------------------------------------

    def _state_shapes(self) -> dict[str, tuple[int, ...]]:
        stages = len(self.plan.stages)
        outbox = self.plan.outbox_capacity
        inbox = self.plan.inbox_capacity
        fragments = self.fragment_capacity
        return {
            "outbox_header": (stages, outbox, _HEADER_WIDTH),
            "outbox_bytes": (stages, outbox, self.slot_bytes),
            "inbox_header": (stages, inbox, fragments, _HEADER_WIDTH),
            "inbox_bytes": (stages, inbox, fragments, self.slot_bytes),
            "inbox_order": (stages, inbox),
            "next_sequence": (stages,),
            "next_order": (stages,),
            "published": (len(self.plan.sinks),),
            "round_index": (),
            "flags": (2,),
            "transfer_digest": (32,),
        }

    def initial_state(self) -> AtomisticStageTransportState:
        shapes = self._state_shapes()
        return AtomisticStageTransportState(
            outbox_header=np.zeros(shapes["outbox_header"], np.int64),
            outbox_bytes=np.zeros(shapes["outbox_bytes"], np.uint8),
            inbox_header=np.zeros(shapes["inbox_header"], np.int64),
            inbox_bytes=np.zeros(shapes["inbox_bytes"], np.uint8),
            inbox_order=np.full(shapes["inbox_order"], -1, np.int64),
            next_sequence=np.zeros(shapes["next_sequence"], np.int64),
            next_order=np.zeros(shapes["next_order"], np.int64),
            published=np.zeros(shapes["published"], np.int64),
            round_index=np.zeros((), np.int64),
            flags=np.zeros((2,), np.bool_),
            transfer_digest=np.frombuffer(
                hashlib.sha256(self.prepared_id.encode("utf-8")).digest(), np.uint8
            ).copy(),
            prepared_id=self.prepared_id,
        )

    def _require_state(self, state: AtomisticStageTransportState, /) -> None:
        if not isinstance(state, AtomisticStageTransportState):
            raise TypeError("state must be an AtomisticStageTransportState.")
        shapes = self._state_shapes()
        if state.prepared_id != self.prepared_id or any(
            getattr(state, name).shape != shape for name, shape in shapes.items()
        ):
            raise ValueError("The transport state belongs to another schedule.")

    def outbox_free(self, state: AtomisticStageTransportState, stage: str, /) -> int:
        """Free outbox slots of ``stage``; zero parks further completed tasks."""
        self._require_state(state)
        index = self._stage_index(stage)
        return int(np.count_nonzero(state.outbox_header[index, :, 0] == 0))

    def inbox_count(self, state: AtomisticStageTransportState, stage: str, /) -> int:
        self._require_state(state)
        return int(np.count_nonzero(state.inbox_order[self._stage_index(stage)] >= 0))

    def deposit(
        self,
        state: AtomisticStageTransportState,
        stage: str,
        packet: AtomisticTaskPacket,
        /,
        *,
        destination: str | None = None,
    ) -> AtomisticStageTransportState:
        """Park one completed packet (or this process's fragment) in an outbox.

        Every member of the stage deposits in the same order: the coordinator's
        copy is transmitted on ``replicated`` edges and every member's fragment
        on ``process-local`` edges. A full outbox refuses with
        `AtomisticTransportCapacityError` and never overwrites a parked packet;
        drivers check `outbox_free` before producing new physics.
        """
        self._require_state(state)
        if state.cancelled or state.terminated:
            raise AtomisticTransportClosedError("The transport schedule is closed.")
        index = self._stage_index(stage)
        if jax.process_index() not in self.stage_processes[index]:
            raise ValueError(
                f"Process {jax.process_index()} is not a member of {stage!r}."
            )
        candidates = [
            edge_index
            for edge_index, edge in enumerate(self.plan.edges)
            if edge.source == stage
            and (destination is None or edge.destination == destination)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Stage {stage!r} needs an explicit, existing destination edge."
            )
        edge_index = candidates[0]
        encoded = self.codec(edge_index).encode(packet)
        free = np.flatnonzero(state.outbox_header[index, :, 0] == 0)
        if free.size == 0:
            raise AtomisticTransportCapacityError(
                f"The outbox of stage {stage!r} is full; the completed task stays "
                "parked with its producer."
            )
        slot = int(free[0])
        header = np.array(state.outbox_header, copy=True)
        payload = np.array(state.outbox_bytes, copy=True)
        sequences = np.array(state.next_sequence, copy=True)
        header[index, slot, :6] = (
            1,
            edge_index,
            sequences[index],
            0,
            _ROWS,
            encoded.size,
        )
        header[index, slot, 6:] = _digest_words(encoded)
        payload[index, slot] = 0
        payload[index, slot, : encoded.size] = encoded
        sequences[index] += 1
        return state.replace(
            outbox_header=header, outbox_bytes=payload, next_sequence=sequences
        )

    def recommit(
        self,
        state: AtomisticStageTransportState,
        stage: str,
        packet: AtomisticTaskPacket,
        /,
        *,
        source: str,
        sequence: int,
    ) -> AtomisticStageTransportState:
        """Place one already committed inbound packet in ``stage``'s inbox.

        Changed-topology continuation only: a transfer that a checkpoint holds
        committed in this stage's inbox stays committed (no credit, grant or
        exchange) and keeps its source ``sequence``. Every member of the stage
        recommits in the source commit order, each with its own complete
        fragment on ``process-local`` edges (the replicated packet otherwise).
        A full inbox refuses with `AtomisticTransportCapacityError`.
        """
        self._require_state(state)
        if state.cancelled or state.terminated:
            raise AtomisticTransportClosedError("The transport schedule is closed.")
        if type(sequence) is not int or sequence < 0:
            raise ValueError("sequence must be a non-negative integer.")
        index = self._stage_index(stage)
        if jax.process_index() not in self.stage_processes[index]:
            raise ValueError(
                f"Process {jax.process_index()} is not a member of {stage!r}."
            )
        candidates = [
            edge_index
            for edge_index, edge in enumerate(self.plan.edges)
            if edge.source == source and edge.destination == stage
        ]
        if len(candidates) != 1:
            raise ValueError(f"No transport edge {source!r} -> {stage!r} exists.")
        edge_index = candidates[0]
        encoded = self.codec(edge_index).encode(packet)
        free = np.flatnonzero(state.inbox_order[index] < 0)
        if free.size == 0:
            raise AtomisticTransportCapacityError(
                f"The inbox of stage {stage!r} is full."
            )
        slot = int(free[0])
        header = np.array(state.inbox_header, copy=True)
        payload = np.array(state.inbox_bytes, copy=True)
        orders = np.array(state.inbox_order, copy=True)
        next_order = np.array(state.next_order, copy=True)
        header[index, slot] = 0
        payload[index, slot] = 0
        header[index, slot, 0, :6] = (1, edge_index, sequence, 0, _ROWS, encoded.size)
        header[index, slot, 0, 6:] = _digest_words(encoded)
        payload[index, slot, 0, : encoded.size] = encoded
        orders[index, slot] = next_order[index]
        next_order[index] += 1
        return state.replace(
            inbox_header=header,
            inbox_bytes=payload,
            inbox_order=orders,
            next_order=next_order,
        )

    def receive(
        self, state: AtomisticStageTransportState, stage: str, /
    ) -> tuple[AtomisticStageTransportState, AtomisticTransportDelivery | None]:
        """Take the oldest committed inbound packet of ``stage`` (members only)."""
        self._require_state(state)
        index = self._stage_index(stage)
        if jax.process_index() not in self.stage_processes[index]:
            raise ValueError(
                f"Process {jax.process_index()} is not a member of {stage!r}."
            )
        orders = state.inbox_order[index]
        occupied = np.flatnonzero(orders >= 0)
        if occupied.size == 0:
            return state, None
        slot = int(occupied[np.argmin(orders[occupied])])
        fragments: list[AtomisticTaskPacket] = []
        header_only: list[bool] = []
        edge_index = -1
        sequence = -1
        for position in range(self.fragment_capacity):
            row = state.inbox_header[index, slot, position]
            if row[0] == 0:
                continue
            edge_index = int(row[1])
            sequence = int(row[2])
            codec = self.codec(edge_index, int(row[4]))
            fragments.append(
                codec.decode(state.inbox_bytes[index, slot, position, : codec.byte_count])
            )
            header_only.append(int(row[4]) == _HEADER_ONLY)
        if not fragments:
            raise AtomisticTransportSchemaError("A committed inbox entry is empty.")
        delivery = AtomisticTransportDelivery(
            fragments=tuple(fragments),
            header_only=tuple(header_only),
            stage=stage,
            source=self.plan.edges[edge_index].source,
            sequence=sequence,
            order=int(orders[slot]),
        )
        inbox_header = np.array(state.inbox_header, copy=True)
        inbox_bytes = np.array(state.inbox_bytes, copy=True)
        inbox_order = np.array(state.inbox_order, copy=True)
        inbox_header[index, slot] = 0
        inbox_bytes[index, slot] = 0
        inbox_order[index, slot] = -1
        return (
            state.replace(
                inbox_header=inbox_header,
                inbox_bytes=inbox_bytes,
                inbox_order=inbox_order,
            ),
            delivery,
        )

    # -- collectives --------------------------------------------------------

    def _local_root_devices(self) -> tuple[tuple[int, jax.Device], ...]:
        me = jax.process_index()
        return tuple(
            (index, device)
            for index, device in enumerate(self.root.devices)
            if device.process_index == me
        )

    def _gather(self, row: np.ndarray, /) -> np.ndarray:
        """All-gather one row per process over the root group (lane devices)."""
        return _root_gather(self.root, self.axis_name, self.processes, self.lanes, row)

    def _exchange(
        self, payload: np.ndarray, headers: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """All-to-all of this lane's ``(devices * lane_slots, ...)`` slot blocks."""
        devices = len(self.root.devices)
        slots = self.plan.lane_slots
        lane = self.lanes[self.processes.index(jax.process_index())]
        sharding = NamedSharding(self.root.mesh, PartitionSpec(self.axis_name))
        payload_blocks = []
        header_blocks = []
        for index, device in self._local_root_devices():
            payload_blocks.append(
                jax.device_put(
                    payload if index == lane else np.zeros_like(payload), device
                )
            )
            header_blocks.append(
                jax.device_put(
                    headers if index == lane else np.zeros_like(headers), device
                )
            )
        global_payload = jax.make_array_from_single_device_arrays(
            (devices * devices * slots, self.slot_bytes), sharding, payload_blocks
        )
        global_headers = jax.make_array_from_single_device_arrays(
            (devices * devices * slots, _HEADER_WIDTH), sharding, header_blocks
        )
        received_payload, received_headers = _exchange_slots(
            self, global_payload, global_headers
        )
        lane_device = self.root.devices[lane]

        def local(array: jax.Array) -> np.ndarray:
            for shard in array.addressable_shards:
                if shard.device == lane_device:
                    return np.asarray(shard.data)
            raise RuntimeError("The transport lane device holds no received block.")

        return local(received_payload), local(received_headers)

    # -- round --------------------------------------------------------------

    def _control_row(
        self, state: AtomisticStageTransportState, drained: bool, cancel: bool, /
    ) -> np.ndarray:
        outbox = self.plan.outbox_capacity
        row = np.zeros((self.control_width,), np.int64)
        row[0] = 1
        row[1] = int(bool(cancel))
        row[2] = int(bool(drained))
        me = jax.process_index()
        for index, members in enumerate(self.stage_processes):
            if me not in members:
                continue
            base = 3 + index * (2 + 3 * outbox)
            row[base] = 1
            row[base + 1] = int(np.count_nonzero(state.inbox_order[index] < 0))
            row[base + 2 : base + 2 + 3 * outbox] = state.outbox_header[
                index, :, :3
            ].reshape(-1)
        return row

    def _decide(
        self, rows: np.ndarray, /
    ) -> tuple[bool, bool, tuple[_Grant, ...], tuple[int, ...], int]:
        outbox = self.plan.outbox_capacity
        if not np.all(rows[:, 0] == 1):
            raise AtomisticTransportDivergenceError("A transport lane is missing.")
        cancelled = bool(np.any(rows[:, 1] != 0))
        credits: list[int] = []
        offers: list[list[tuple[int, int, int]]] = []
        empty = True
        for index, members in enumerate(self.stage_processes):
            base = 3 + index * (2 + 3 * outbox)
            segment = rows[:, base : base + 2 + 3 * outbox]
            positions = [self.processes.index(process) for process in members]
            others = [
                position
                for position in range(len(self.processes))
                if position not in positions
            ]
            reference = segment[positions[0]]
            if (
                reference[0] != 1
                or any(not np.array_equal(segment[p], reference) for p in positions)
                or any(np.any(segment[p] != 0) for p in others)
            ):
                raise AtomisticTransportDivergenceError(
                    f"Members of stage {self.plan.stages[index].name!r} disagree."
                )
            credits.append(int(reference[1]))
            entries = reference[2:].reshape(outbox, 3)
            stage_offers = sorted(
                (int(entry[2]), int(entry[1]), slot)
                for slot, entry in enumerate(entries)
                if entry[0] == 1
            )
            offers.append(stage_offers)
            if stage_offers or credits[-1] != self.plan.inbox_capacity:
                empty = False
        offered = sum(len(stage_offers) for stage_offers in offers)
        terminated = (not cancelled) and empty and bool(np.all(rows[:, 2] != 0))
        if cancelled or terminated:
            return cancelled, terminated, (), tuple(credits), offered
        names = tuple(stage.name for stage in self.plan.stages)
        remaining = list(credits)
        usage: dict[tuple[int, int], int] = {}
        grants: list[_Grant] = []
        for edge_index, edge in enumerate(self.plan.edges):
            source = names.index(edge.source)
            destination = (
                names.index(edge.destination) if edge.destination in names else None
            )
            for sequence, offer_edge, slot in offers[source]:
                if offer_edge != edge_index:
                    continue
                if destination is not None and remaining[destination] <= 0:
                    break
                demand: dict[tuple[int, int], int] = {}
                for route in self.routes[edge_index]:
                    pair = (route[0], route[1])
                    demand[pair] = demand.get(pair, 0) + 1
                if any(
                    usage.get(pair, 0) + count > self.plan.lane_slots
                    for pair, count in demand.items()
                ):
                    break
                for pair, count in demand.items():
                    usage[pair] = usage.get(pair, 0) + count
                if destination is not None:
                    remaining[destination] -= 1
                grants.append((edge_index, source, slot, sequence))
        return cancelled, terminated, tuple(grants), tuple(credits), offered

    def _slot_plan(self, grants: Sequence[_Grant], /) -> list[_Assignment]:
        """Deterministic (grant, route, slot) assignment shared by every process."""
        usage: dict[tuple[int, int], int] = {}
        assignments = []
        for position, grant in enumerate(grants):
            for route in self.routes[grant[0]]:
                pair = (route[0], route[1])
                slot = usage.get(pair, 0)
                usage[pair] = slot + 1
                assignments.append((position, route, slot))
        return assignments

    def _send_blocks(
        self,
        state: AtomisticStageTransportState,
        grants: Sequence[_Grant],
        assignments: Sequence[_Assignment],
        lane_of: Mapping[int, int],
        /,
    ) -> tuple[np.ndarray, np.ndarray]:
        """This lane's exchange blocks: every granted fragment it sources.

        Always the full exchange shape (all zero when this process sends none).
        """
        me = jax.process_index()
        devices = len(self.root.devices)
        slots = self.plan.lane_slots
        send_payload = np.zeros((devices * slots, self.slot_bytes), np.uint8)
        send_headers = np.zeros((devices * slots, _HEADER_WIDTH), np.int64)
        for position, route, slot in assignments:
            source, destination, fragment, kind = route
            if source != me:
                continue
            edge_index, stage_index, outbox_slot, sequence = grants[position]
            stored = state.outbox_header[stage_index, outbox_slot]
            encoded = state.outbox_bytes[stage_index, outbox_slot, : int(stored[5])]
            if kind == _HEADER_ONLY:
                encoded = self.codec(edge_index, _HEADER_ONLY).encode(
                    _header_only(self.codec(edge_index).decode(encoded))
                )
            row = lane_of[destination] * slots + slot
            send_payload[row, : encoded.size] = encoded
            send_headers[row, :6] = (1, position, sequence, fragment, kind, encoded.size)
            send_headers[row, 6:] = _digest_words(encoded)
        return send_payload, send_headers

    def _stage_in(
        self,
        grants: Sequence[_Grant],
        assignments: Sequence[_Assignment],
        received_payload: np.ndarray,
        received_headers: np.ndarray,
        lane_of: Mapping[int, int],
        sinks: Mapping[str, AtomisticTransportSink],
        /,
    ) -> tuple[_Staged, np.ndarray, list[str]]:
        """Verify, decode and stage the fragments this process receives.

        Sink-bound packets are published here (an existing receipt first).
        Grants of one edge are sequence ordered; once one of them fails here,
        later grants of that edge are neither staged nor published (FIFO).
        Returns the staged fragments, this process's acknowledgement row and
        the receipts in publication order.
        """
        me = jax.process_index()
        slots = self.plan.lane_slots
        staged: _Staged = {}
        acknowledged = np.ones((1 + self.grant_capacity,), np.int64)
        receipts: list[str] = []
        failed_edges: set[int] = set()
        for position, route, slot in assignments:
            source, destination, fragment, kind = route
            if destination != me:
                continue
            edge_index, _, _, sequence = grants[position]
            if edge_index in failed_edges:
                acknowledged[1 + position] = 0
                continue
            edge = self.plan.edges[edge_index]
            row = lane_of[source] * slots + slot
            header = received_headers[row]
            codec = self.codec(edge_index, kind)
            ok = _verify_fragment(
                header,
                received_payload[row],
                (position, sequence, fragment, kind, codec.byte_count),
            )
            packet = None
            if ok:
                try:
                    packet = codec.decode(received_payload[row, : codec.byte_count])
                except AtomisticTransportSchemaError:
                    ok = False
            if ok and packet is not None and edge.destination in self.plan.sinks:
                sink = sinks[edge.destination]
                key = f"{self.plan.plan_id}:{edge.edge_id}:{sequence}"
                receipt = sink.receipt(key)
                if receipt is None:
                    receipt = sink.publish(key, packet)
                ok = receipt is not None
                if receipt is not None:
                    receipts.append(str(receipt))
            if not ok:
                acknowledged[1 + position] = 0
                failed_edges.add(edge_index)
                continue
            staged.setdefault(position, []).append(
                (fragment, kind, header, received_payload[row])
            )
        return staged, acknowledged, receipts

    def _commit(
        self,
        state: AtomisticStageTransportState,
        grants: Sequence[_Grant],
        committed: Sequence[bool],
        staged: _Staged,
        cancelled: bool,
        terminated: bool,
        /,
    ) -> AtomisticStageTransportState:
        """Apply the committed grants: identical decisions on every process.

        Source-stage members free the outbox slot, destination-stage members
        fill the oldest free inbox slot in grant order, sink edges count a
        publication; the transfer digest chains every committed grant onto
        the previous digest and round index.
        """
        me = jax.process_index()
        names = tuple(stage.name for stage in self.plan.stages)
        outbox_header = np.array(state.outbox_header, copy=True)
        outbox_bytes = np.array(state.outbox_bytes, copy=True)
        inbox_header = np.array(state.inbox_header, copy=True)
        inbox_bytes = np.array(state.inbox_bytes, copy=True)
        inbox_order = np.array(state.inbox_order, copy=True)
        next_order = np.array(state.next_order, copy=True)
        published = np.array(state.published, copy=True)
        chain = hashlib.sha256(state.transfer_digest.tobytes())
        chain.update(int(state.round_index).to_bytes(8, "little"))
        for position, grant in enumerate(grants):
            if not committed[position]:
                continue
            edge_index, stage_index, outbox_slot, sequence = grant
            edge = self.plan.edges[edge_index]
            chain.update(np.asarray([edge_index, stage_index, sequence], "<i8").tobytes())
            if me in self.stage_processes[stage_index]:
                outbox_header[stage_index, outbox_slot] = 0
                outbox_bytes[stage_index, outbox_slot] = 0
            if edge.destination in self.plan.sinks:
                published[self.plan.sinks.index(edge.destination)] += 1
                continue
            destination = names.index(edge.destination)
            if me not in self.stage_processes[destination]:
                continue
            free = np.flatnonzero(inbox_order[destination] < 0)
            if free.size == 0:
                raise AtomisticTransportDivergenceError(
                    "A granted transfer found no inbox slot."
                )
            inbox_slot = int(free[0])
            fragments = sorted(staged.get(position, ()), key=lambda item: item[0])
            if not fragments or len(fragments) > self.fragment_capacity:
                raise AtomisticTransportDivergenceError(
                    "A committed transfer has no fragment for this member."
                )
            for local_index, (fragment, kind, header, payload) in enumerate(fragments):
                inbox_header[destination, inbox_slot, local_index, :6] = (
                    1,
                    edge_index,
                    sequence,
                    fragment,
                    kind,
                    header[5],
                )
                inbox_header[destination, inbox_slot, local_index, 6:] = header[6:]
                inbox_bytes[destination, inbox_slot, local_index] = payload
            inbox_order[destination, inbox_slot] = next_order[destination]
            next_order[destination] += 1
        digest = np.frombuffer(chain.digest(), np.uint8).copy()
        flags = np.asarray([cancelled, terminated], np.bool_)
        return state.replace(
            outbox_header=outbox_header,
            outbox_bytes=outbox_bytes,
            inbox_header=inbox_header,
            inbox_bytes=inbox_bytes,
            inbox_order=inbox_order,
            next_order=next_order,
            published=published,
            round_index=np.asarray(int(state.round_index) + 1, np.int64),
            flags=flags,
            transfer_digest=digest,
        )

    def round(
        self,
        state: AtomisticStageTransportState,
        /,
        *,
        drained: bool,
        cancel: bool = False,
        sinks: Mapping[str, AtomisticTransportSink] | None = None,
    ) -> AtomisticTransportRound:
        """Execute one scheduled round (a collective of every root process).

        ``drained`` declares that this process's stage drivers hold no active
        task and will produce no more packets. ``sinks`` are required on the
        root coordinator (sink host) for every declared sink.
        """
        self._require_state(state)
        if state.cancelled or state.terminated:
            raise AtomisticTransportClosedError("The transport schedule is closed.")
        if int(state.round_index) >= self.plan.maximum_rounds:
            raise AtomisticTransportCapacityError(
                f"The transport schedule reached its {self.plan.maximum_rounds}-round "
                "bound."
            )
        me = jax.process_index()
        sinks_ = dict(sinks or {})
        if me == self.sink_host and set(sinks_) != set(self.plan.sinks):
            raise ValueError("The sink host must supply every declared sink.")
        lane_of = dict(zip(self.processes, self.lanes, strict=True))

        # Phase 1: control consensus.
        rows = self._gather(self._control_row(state, drained, cancel))
        cancelled, terminated, grants, credits, offered = self._decide(rows)

        # Phase 2: payload exchange (always executed; empty when no grants).
        assignments = self._slot_plan(grants)
        send_payload, send_headers = self._send_blocks(
            state, grants, assignments, lane_of
        )
        received_payload, received_headers = self._exchange(send_payload, send_headers)

        # Phase 3: stage-in, sink publication and receipt consensus.
        staged, acknowledged, receipts = self._stage_in(
            grants, assignments, received_payload, received_headers, lane_of, sinks_
        )
        receipt_rows = self._gather(acknowledged)
        # Commit only the acknowledged sequence prefix of every edge, so a
        # refused packet is never overtaken by a later one of the same edge.
        committed: list[bool] = []
        blocked_edges: set[int] = set()
        for position, grant in enumerate(grants):
            acknowledged_everywhere = bool(np.all(receipt_rows[:, 1 + position] == 1))
            if grant[0] in blocked_edges or not acknowledged_everywhere:
                blocked_edges.add(grant[0])
                committed.append(False)
            else:
                committed.append(True)

        new_state = self._commit(state, grants, committed, staged, cancelled, terminated)
        committed_count = sum(committed)
        evidence = AtomisticTransportRoundEvidence(
            round_index=int(state.round_index),
            offered=offered,
            granted=len(grants),
            committed=committed_count,
            parked=offered - committed_count,
            refused=len(grants) - committed_count,
            credits=credits,
            cancelled=cancelled,
            terminated=terminated,
            receipts=tuple(receipts),
            transfer_digest=new_state.transfer_digest.tobytes().hex(),
        )
        return AtomisticTransportRound(new_state, evidence)

    # -- process-blocked checkpoint form -----------------------------------

    def blocked(self, value: np.ndarray, /) -> jax.Array:
        """Global ``(processes, ...)`` array holding this process's host block.

        Each process contributes its own block on its lane device, so the
        lifecycle publication writes one canonical shard per process.
        """
        block = np.asarray(value)
        device = self.lane_mesh.devices[self.processes.index(jax.process_index())]
        return jax.make_array_from_single_device_arrays(
            (len(self.processes),) + block.shape,
            self.block_sharding(block.ndim),
            [jax.device_put(block[None], device)],
        )

    def block_sharding(self, rank: int, /) -> NamedSharding:
        """Process-blocked sharding of a block of ``rank`` trailing axes."""
        return NamedSharding(self.lane_mesh, PartitionSpec("process", *([None] * rank)))

    def own_block(self, array: jax.Array, /) -> np.ndarray:
        """This process's host block of a process-blocked global array."""
        lane_device = self.lane_mesh.devices[self.processes.index(jax.process_index())]
        for shard in array.addressable_shards:
            if shard.device == lane_device:
                return np.array(np.asarray(shard.data)[0], copy=True)
        raise ValueError("This process holds no process block.")

    def process_blocked(
        self, state: AtomisticStageTransportState, /
    ) -> dict[str, jax.Array]:
        """Global ``(processes, ...)`` arrays of every process's transport state."""
        self._require_state(state)
        return {name: self.blocked(getattr(state, name)) for name in _STATE_FIELDS}

    def process_block_sharding(self, name: str, /) -> NamedSharding:
        return self.block_sharding(len(self._state_shapes()[name]))

    def from_process_blocked(
        self, arrays: Mapping[str, jax.Array], /
    ) -> AtomisticStageTransportState:
        """This process's transport state from its block of the global arrays."""
        if set(arrays) != set(_STATE_FIELDS):
            raise ValueError("Process-blocked transport state is incomplete.")
        fields = {name: self.own_block(arrays[name]) for name in _STATE_FIELDS}
        state = AtomisticStageTransportState(**fields, prepared_id=self.prepared_id)
        self._require_state(state)
        return state


def _verify_fragment(
    header: np.ndarray,
    payload: np.ndarray,
    expected: tuple[int, int, int, int, int],
    /,
) -> bool:
    """Whether one received slot is exactly the expected authenticated fragment."""
    position, sequence, fragment, kind, byte_count = expected
    if (
        int(header[0]) != 1
        or int(header[1]) != position
        or int(header[2]) != sequence
        or int(header[3]) != fragment
        or int(header[4]) != kind
        or int(header[5]) != byte_count
    ):
        return False
    return bool(np.array_equal(_digest_words(payload[:byte_count]), header[6:]))


# ---------------------------------------------------------------------------
# Stage×domain campaigns
# ---------------------------------------------------------------------------


class AtomisticDomainCampaignCheckpointError(ValueError):
    """A campaign checkpoint does not match this campaign, topology or program."""


class AtomisticDomainStageProgram(Protocol):
    """Member-side preparation of one stage's native distributed physics.

    Called only on the stage's member processes with its live child group;
    returns the prepared domain dynamics whose ownership is exactly that group.
    """

    def __call__(self, group: ExecutionGroup, /) -> PreparedAtomisticDomainDynamics: ...


def _label(name: str, value: object, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


@final
@dataclass(frozen=True, eq=False)
class AtomisticDomainCampaignTask:
    """One task of a stage×domain campaign.

    ``task.packet`` holds unwrapped Cartesian ``positions`` ``(N, 3)``, atomic
    numbers ``species``, the model's native species indices
    ``atom_type_ids``, optional ``momenta`` and the optional current
    ``cell_vectors`` (default: the entry stage's plan cell). ``masses`` are
    the ``N`` atomic masses; ``mobile``/``molecules`` are the logical-order
    labels a method's ensemble declares (`AtomisticDomainDynamicsPlan`
    ``atom_payload``).
    """

    task: AtomisticCampaignTask
    masses: np.ndarray
    mobile: np.ndarray | None = None
    molecules: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, AtomisticCampaignTask):
            raise TypeError("task must be an AtomisticCampaignTask.")
        packet = self.task.packet
        positions = packet.positions
        if positions is None or not np.all(np.isfinite(positions)):
            raise ValueError("A domain task packet carries finite positions.")
        count = positions.shape[0]
        for name in ("species", "atom_type_ids"):
            value = getattr(packet, name)
            if value is None or value.shape != (count,):
                raise ValueError(
                    f"A domain task packet carries {name} with one entry per atom."
                )
        if packet.cell_vectors is not None and packet.cell_vectors.shape != (3, 3):
            raise ValueError("A domain task cell must have shape (3, 3).")
        masses = self.masses
        if (
            not isinstance(masses, np.ndarray)
            or masses.dtype != np.float64
            or masses.shape != (count,)
            or not np.all(np.isfinite(masses))
            or np.any(masses <= 0.0)
        ):
            raise ValueError("masses must hold one finite positive float64 per atom.")

    @property
    def atom_count(self) -> int:
        positions = self.task.packet.positions
        if positions is None:
            raise ValueError("A domain task packet carries positions.")
        return int(positions.shape[0])


@final
@dataclass(frozen=True, eq=False)
class AtomisticDomainStage:
    """One campaign stage: a child group, its native domain program and budget.

    ``program_id`` is the declared topology-neutral label of the program
    (model, Hamiltonian, ensemble and method); it enters the campaign's
    analysis identity on every process but proves nothing by itself. Members
    record and verify the prepared runtime, revision and dynamics identities
    (layout-bound) and the program's derived layout-independent
    `PreparedAtomisticDomainDynamics.scientific_identity`. ``budget``
    is the accepted steps of one visit; ``segment`` the steps one scheduled
    round may execute.
    """

    name: str
    group: ExecutionGroupSpec
    program: AtomisticDomainStageProgram
    program_id: str
    budget: int
    segment: int

    def __post_init__(self) -> None:
        _label("stage name", self.name)
        _label("program_id", self.program_id)
        if not isinstance(self.group, ExecutionGroupSpec):
            raise TypeError("group must be an ExecutionGroupSpec.")
        if self.group.parent_group_id is None:
            raise ValueError("Stage groups must be explicit children of a root group.")
        if not callable(self.program):
            raise TypeError("program must be a callable stage program.")
        _positive("budget", self.budget)
        _positive("segment", self.segment)
        if self.segment > self.budget:
            raise ValueError("segment must not exceed the visit budget.")


@final
@dataclass(frozen=True, eq=False)
class AtomisticDomainCampaignPlan:
    """Linear stage×domain campaign: entry stage, downstream stages, one sink.

    Tasks enter the first stage in plan order. A visit that completes its
    budget continues to the next stage over a process-local edge carrying
    the `AtomisticDomainPacket` fragments of every member (one atomic group
    transfer); completed visits of the last stage and every failed or
    capacity-refused visit are published through the stage's replicated
    report edge to the campaign sink. A stage holds at most one resident task
    (one periodic system per domain plan); a finished resident whose outbox
    is full stays parked without new physics. Task RNG keys are
    ``fold_in(key(rng_root), task.ordinal)``: semantic, never rank-derived.
    """

    campaign_id: str
    stages: tuple[AtomisticDomainStage, ...]
    tasks: tuple[AtomisticDomainCampaignTask, ...]
    rng_root: int = 0
    sink: str = "campaign-outputs"
    outbox_capacity: int = 1
    inbox_capacity: int = 1
    lane_slots: int = 1
    maximum_rounds: int = 4096
    maximum_round_bytes: int = 1 << 26

    def __post_init__(self) -> None:
        _label("campaign_id", self.campaign_id)
        _label("sink", self.sink)
        if (
            not isinstance(self.stages, tuple)
            or not self.stages
            or not all(isinstance(stage, AtomisticDomainStage) for stage in self.stages)
        ):
            raise TypeError("stages must be a non-empty tuple of AtomisticDomainStage.")
        if (
            not isinstance(self.tasks, tuple)
            or not self.tasks
            or not all(
                isinstance(task, AtomisticDomainCampaignTask) for task in self.tasks
            )
        ):
            raise TypeError(
                "tasks must be a non-empty tuple of AtomisticDomainCampaignTask."
            )
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names) or self.sink in names:
            raise ValueError("Stage names and the sink name must be distinct.")
        if len({stage.group.parent_group_id for stage in self.stages}) != 1:
            raise ValueError("Every stage group must be a child of the same root.")
        identifiers = [entry.task.task_id for entry in self.tasks]
        ordinals = [entry.task.ordinal for entry in self.tasks]
        if len(set(identifiers)) != len(identifiers) or len(set(ordinals)) != len(
            ordinals
        ):
            raise ValueError("Task identities and RNG ordinals must be unique.")
        if type(self.rng_root) is not int or not 0 <= self.rng_root < 2**32:
            raise ValueError("rng_root must be a uint32-range int.")
        for name in (
            "outbox_capacity",
            "inbox_capacity",
            "lane_slots",
            "maximum_rounds",
            "maximum_round_bytes",
        ):
            _positive(name, getattr(self, name))

    @property
    def analysis_id(self) -> str:
        """Topology-neutral scientific identity (stages, programs, tasks, RNG)."""
        return canonical_fingerprint(
            {
                "kind": "atomistic-domain-campaign",
                "campaign": self.campaign_id,
                "stages": [
                    [stage.name, stage.program_id, stage.budget, stage.segment]
                    for stage in self.stages
                ],
                "tasks": [
                    {
                        **entry.task.record(),
                        "masses": array_tree_fingerprint(entry.masses),
                        "mobile": None
                        if entry.mobile is None
                        else array_tree_fingerprint(np.asarray(entry.mobile)),
                        "molecules": None
                        if entry.molecules is None
                        else array_tree_fingerprint(np.asarray(entry.molecules)),
                    }
                    for entry in self.tasks
                ],
                "rng_root": self.rng_root,
                "sink": self.sink,
            }
        )

    @property
    def revision_id(self) -> str:
        """Declared program revision identity of every stage."""
        return canonical_fingerprint(
            {
                "kind": "atomistic-domain-campaign-programs",
                "programs": [stage.program_id for stage in self.stages],
            }
        )

    @property
    def atom_capacity(self) -> int:
        return max(entry.atom_count for entry in self.tasks)

    def run_id(self, index: int, /) -> str:
        """Canonical run digest of one task (its domain continuation identity)."""
        return canonical_fingerprint(
            {
                "kind": "atomistic-domain-campaign-run",
                "campaign": self.campaign_id,
                "task": self.tasks[index].task.task_id,
            }
        )

    def prepare(self, root: ExecutionGroup, /) -> PreparedAtomisticDomainCampaign:
        """Bind every member stage and the transport (a root-group collective)."""
        return PreparedAtomisticDomainCampaign(self, root)


@final
class AtomisticDomainStageReport(StrictModule, NonTrainableState):
    """Replicated sink record of one task visit (host NumPy leaves).

    ``forces`` are logical-order rows padded to the campaign atom capacity.
    The evaluation identity words (exact digests as `IdentityWords` words) are
    all layout-independent and become the output's `AtomisticEvaluationKey`:
    ``hamiltonian_words`` the domain Hamiltonian
    (`PreparedAtomisticDomainRuntime.scientific_identity`),
    ``revision_words`` its numeric revision
    (``hamiltonian_revision_identity``) and ``conditions_words`` the evaluated
    conditions (no domain controls, the state's evaluated conditioning
    ``conditioning_id`` and the stage ensemble's thermodynamic ``plan_id``).
    A report parked in a checkpoint keeps the words it was written with.
    """

    task_index: np.ndarray
    stage_index: np.ndarray
    outcome: np.ndarray
    reasons: np.ndarray
    accepted_steps: np.ndarray
    step_index: np.ndarray
    energy: np.ndarray
    forces: np.ndarray
    stress: np.ndarray
    stress_available: np.ndarray
    force_valid: np.ndarray
    hamiltonian_words: np.ndarray
    revision_words: np.ndarray
    conditions_words: np.ndarray
    rng_words: np.ndarray


def _key_words() -> int:
    return int(np.asarray(jax.random.key_data(jax.random.key(0))).size)


def _report_template(capacity: int, /) -> AtomisticTaskPacket:
    zeros = np.zeros((capacity, 3), np.float64)
    return AtomisticTaskPacket(
        positions=zeros,
        cell_vectors=np.zeros((3, 3), np.float64),
        momenta=zeros,
        species=np.zeros((capacity,), np.int32),
        atom_type_ids=np.zeros((capacity,), np.int32),
        atom_mask=np.zeros((capacity,), np.bool_),
        continuation=AtomisticDomainStageReport(
            task_index=np.zeros((), np.int64),
            stage_index=np.zeros((), np.int64),
            outcome=np.zeros((), np.int32),
            reasons=np.zeros((), np.int32),
            accepted_steps=np.zeros((), np.int64),
            step_index=np.zeros((), np.int64),
            energy=np.zeros((), np.float64),
            forces=zeros,
            stress=np.zeros((3, 3), np.float64),
            stress_available=np.zeros((), np.bool_),
            force_valid=np.zeros((), np.bool_),
            hamiltonian_words=np.zeros((8,), np.uint32),
            revision_words=np.zeros((8,), np.uint32),
            conditions_words=np.zeros((8,), np.uint32),
            rng_words=np.zeros((_key_words(),), np.uint32),
        ),
    )


def _host_replica(value: jax.Array, /) -> np.ndarray:
    """This process's replica of a replicated (possibly multi-process) value."""
    return np.array(np.asarray(value.addressable_data(0)), copy=True)


def _rejection_outcome(reasons: int, /) -> AtomisticStageOutcome:
    """Capacity-only rejections are capacity refusals, never scientific failure."""
    if reasons and not reasons & ~CAPACITY_REJECTIONS:
        return AtomisticStageOutcome.CAPACITY_REFUSED
    return AtomisticStageOutcome.FAILED


@eqx.filter_jit
def _domain_segment(
    dynamics: PreparedAtomisticDomainDynamics,
    state: AtomisticDomainState,
    remaining: jax.Array,
    length: int,
) -> tuple[AtomisticDomainState, jax.Array, jax.Array, jax.Array]:
    """Up to ``remaining`` accepted steps of a fixed-length segment, one program.

    The first rejected transaction stops the segment: its accepted state is
    the previous state (P5 transaction semantics) and its reasons are kept.
    """

    def body(
        index: jax.Array,
        carry: tuple[AtomisticDomainState, jax.Array, jax.Array, jax.Array],
    ) -> tuple[AtomisticDomainState, jax.Array, jax.Array, jax.Array]:
        current, steps, failed, reasons = carry
        evaluation = dynamics.step_detailed(current)
        live = (index < remaining) & ~failed
        accept = live & evaluation.successful
        rejected = live & ~evaluation.successful
        current = jax.lax.cond(accept, lambda: evaluation.accepted_state, lambda: current)
        return (
            current,
            steps + accept.astype(jnp.int32),
            failed | rejected,
            jnp.where(rejected, evaluation.rejection_reasons, reasons),
        )

    return jax.lax.fori_loop(
        0,
        length,
        body,
        (
            state,
            jnp.zeros((), jnp.int32),
            jnp.zeros((), jnp.bool_),
            jnp.zeros((), jnp.int32),
        ),
    )


@eqx.filter_jit
def _gathered_rows(
    ownership: DistributedOwnershipPlan, rows: tuple[jax.Array, ...]
) -> tuple[jax.Array, ...]:
    """Child-group all-gather of owner-blocked rows (members only)."""
    axis = ownership.axis_name

    def body(*blocks: jax.Array) -> tuple[jax.Array, ...]:
        provider = JaxCollectiveProvider(axis)
        return tuple(provider.all_gather(block, axis=0, tiled=True) for block in blocks)

    return ownership.map(
        body,
        tuple(PartitionSpec(axis, *([None] * (row.ndim - 1))) for row in rows),
        tuple(PartitionSpec() for _ in rows),
    )(*rows)


def _resized(
    packet: AtomisticTaskPacket, rows: int, constraint_rows: int, /
) -> AtomisticTaskPacket:
    """A domain-continuation template with another per-process fragment layout."""
    continuation = packet.continuation
    if not isinstance(continuation, AtomisticDomainPacket):
        raise AtomisticTransportSchemaError("A stage template carries a domain packet.")

    def resize(value: np.ndarray, count: int) -> np.ndarray:
        return np.zeros((count,) + value.shape[1:], dtype=value.dtype)

    constraints = continuation.constraints
    if constraints is not None:
        if constraint_rows < 0:
            raise AtomisticDomainCampaignCheckpointError(
                "The source continuation carries no constraint rows."
            )
        constraints = AtomisticDomainConstraintRows(
            row_ids=resize(constraints.row_ids, constraint_rows),
            atoms=resize(constraints.atoms, constraint_rows),
            components=resize(constraints.components, constraint_rows),
            targets=resize(constraints.targets, constraint_rows),
            lagrange=resize(constraints.lagrange, constraint_rows),
            active=resize(constraints.active, constraint_rows),
            position_residual=constraints.position_residual,
            velocity_residual=constraints.velocity_residual,
            topology_epoch=constraints.topology_epoch,
            row_capacity=constraints.row_capacity,
            row_count=constraints.row_count,
            halo_capacity=constraints.halo_capacity,
            directory_capacity=constraints.directory_capacity,
            migration_capacity=constraints.migration_capacity,
            plan_id=constraints.plan_id,
        )
    resized = AtomisticDomainPacket(
        rows={name: resize(value, rows) for name, value in continuation.rows.items()},
        atom_payload={
            name: resize(value, rows) for name, value in continuation.atom_payload.items()
        },
        cell_vectors=continuation.cell_vectors,
        replicated_payload=continuation.replicated_payload,
        controllers=continuation.controllers,
        ledger=continuation.ledger,
        conditioning=continuation.conditioning,
        constraints=constraints,
        rng_key_data=continuation.rng_key_data,
        case_identity_words=continuation.case_identity_words,
        step_index=continuation.step_index,
        time=continuation.time,
        force_record=continuation.force_record,
        key_implementation=continuation.key_implementation,
        logical_count=continuation.logical_count,
        role_ids=continuation.role_ids,
        model_revision_id=continuation.model_revision_id,
        run_id=continuation.run_id,
        header_id=continuation.header_id,
    )
    return AtomisticTaskPacket(continuation=resized, family=packet.family)


def _routed(
    fragments: Sequence[AtomisticDomainPacket], member: int, members: int, /
) -> tuple[AtomisticDomainPacket, ...]:
    """The process-local edge rule: fragment ``j`` goes to member ``j mod n``."""
    mine = tuple(
        fragment for index, fragment in enumerate(fragments) if index % members == member
    )
    return mine if mine else (fragments[0].without_rows(),)


@final
@dataclass(frozen=True, eq=False)
class _StageBinding:
    group: ExecutionGroup
    dynamics: PreparedAtomisticDomainDynamics
    template: AtomisticTaskPacket
    codec: AtomisticPacketCodec
    member_index: int
    member_count: int
    fragment_rows: int
    constraint_rows: int

    @property
    def identity(self) -> np.ndarray:
        """Rows: prepared runtime, model revision, dynamics plan (all bound to
        this execution layout) and the layout-independent scientific identity
        (`PreparedAtomisticDomainDynamics.scientific_identity`)."""
        runtime = self.dynamics.runtime
        return np.stack(
            [
                np.asarray(runtime.runtime_identity.words, np.uint32),
                np.asarray(runtime.model_revision_identity.words, np.uint32),
                np.asarray(self.dynamics.plan.plan_identity.words, np.uint32),
                np.asarray(self.dynamics.scientific_identity.words, np.uint32),
            ]
        )


@final
@dataclass(frozen=True, eq=False)
class AtomisticDomainResident:
    """The task a stage's members hold and its visit progress (identical on
    every member; ``state`` is the stage group's distributed domain state)."""

    task_index: int
    accepted_steps: int
    outcome: AtomisticStageOutcome
    reasons: int
    state: AtomisticDomainState


@final
@dataclass(frozen=True, eq=False)
class AtomisticDomainCampaignState:
    """This process's complete continuation of a stage×domain campaign.

    ``cursor`` is the next entry task (entry-stage members); ``residents``
    hold one optional resident per stage this process belongs to; ``epoch``
    counts committed restarts/migrations (it names later checkpoints).
    """

    transport: AtomisticStageTransportState
    cursor: int
    residents: tuple[AtomisticDomainResident | None, ...]
    epoch: int

    @property
    def round_index(self) -> int:
        return int(self.transport.round_index)

    @property
    def terminated(self) -> bool:
        return self.transport.terminated


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticDomainCampaignRound:
    """One scheduled round: this process's stage work and the transport consensus.

    ``executed_steps`` are accepted steps this process's member stages
    executed (zero on non-members, which never enter a stage collective);
    ``parked`` marks finished residents held because their outbox was full.
    """

    state: AtomisticDomainCampaignState
    transport: AtomisticTransportRoundEvidence
    executed_steps: tuple[int, ...]
    admitted: tuple[str | None, ...]
    exported: tuple[str | None, ...]
    parked: tuple[bool, ...]


@final
@dataclass(frozen=True, slots=True)
class AtomisticDomainCampaignPublication:
    """One globally committed campaign checkpoint."""

    checkpoint_id: str
    manifest_id: str
    execution_id: str
    analysis_id: str
    round_index: int


@final
@dataclass(frozen=True, slots=True)
class AtomisticDomainCampaignMigration:
    """Changed-topology continuation evidence of a committed campaign checkpoint.

    Per stage this process belongs to: ``residents`` is the resumed task and
    ``reissued`` counts committed inbox transfers recommitted into the stage's
    new inbox plus parked outbox packets re-issued from its new outbox.
    """

    source_checkpoint_id: str
    source_manifest_id: str
    source_execution_id: str
    target_execution_id: str
    source_process_count: int
    target_process_count: int
    source_round_index: int
    source_transfer_digest: str
    residents: tuple[str | None, ...]
    reissued: tuple[int, ...]
    read_bytes: int


@final
@dataclass(frozen=True, slots=True)
class AtomisticDomainCampaignCheckpointPolicy:
    """Durable cadence of a campaign run: every ``every`` rounds, one commit."""

    repository: ArtifactRepository
    every: int
    prefix: str
    writer_id: str
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS

    def __post_init__(self) -> None:
        _positive("every", self.every)
        _label("prefix", self.prefix)
        _label("writer_id", self.writer_id)
        if not isinstance(self.limits, ArrayArchiveLimits):
            raise TypeError("limits must be ArrayArchiveLimits.")

    def checkpoint_id(self, epoch: int, round_index: int, /) -> str:
        return f"{self.prefix}-e{int(epoch):04d}-r{int(round_index):06d}"


class _DomainCampaignSink:
    """Transport sink of a campaign: report packets become campaign outputs.

    Publication is idempotent by the campaign's semantic output key; a
    replayed transfer republishes the identical payload (a different payload
    is refused by the campaign sink). ``receipt`` has no semantic key for a
    transport key and defers to that idempotent publication.
    """

    __slots__ = ("_campaign", "_round", "_sink")

    def __init__(
        self,
        campaign: PreparedAtomisticDomainCampaign,
        sink: AbstractAtomisticCampaignSink,
        round_index: int,
        /,
    ) -> None:
        self._campaign = campaign
        self._sink = sink
        self._round = round_index

    @property
    def sink_id(self) -> str:
        return self._sink.sink_id

    def receipt(self, key: str, /) -> str | None:
        del key
        return None

    def publish(self, key: str, packet: AtomisticTaskPacket, /) -> str | None:
        del key
        output = self._campaign.output(packet)
        receipt = self._sink.publish(output, round_index=self._round)
        if receipt is None:
            return None
        return f"{receipt.sink_id}:{receipt.semantic_key}:{receipt.payload_id}"


_CONTROL_HEAD = 2
_CONTROL_STAGE = 11
# Stage identity rows (`_StageBinding.identity`); the last is layout-independent.
_IDENTITY_ROWS = 4
_SCIENCE_ROW = 3


@final
class _MigrationSource:
    """Authenticated region reads of one committed campaign checkpoint.

    Holds the small per-process control blocks (controller, stage identity,
    schemas) and every source stage's members in member order; each region
    read counts its bytes (`AtomisticDomainCampaignMigration.read_bytes`).
    """

    __slots__ = (
        "controller",
        "identity",
        "limits",
        "manifest",
        "members",
        "processes",
        "read_bytes",
        "repository",
        "schemas",
    )

    def __init__(
        self,
        repository: ArtifactRepository,
        manifest: CheckpointManifest,
        limits: ArrayArchiveLimits,
        stages: int,
        /,
    ) -> None:
        self.repository = repository
        self.manifest = manifest
        self.limits = limits
        self.read_bytes = 0
        everything = (slice(None),)
        self.controller = self.region("['campaign']['controller']", everything * 2)
        self.identity = self.region("['campaign']['identity']", everything * 4)
        self.schemas = self.region("['campaign']['schemas']", everything * 3)
        self.processes = int(self.controller.shape[0])
        if self.controller.shape[1] != _CONTROL_HEAD + _CONTROL_STAGE * stages:
            raise AtomisticDomainCampaignCheckpointError(
                "The checkpoint holds another number of stages."
            )
        if self.identity.shape[1:] != (stages, _IDENTITY_ROWS, 8):
            raise AtomisticDomainCampaignCheckpointError(
                "The checkpoint records no layout-independent stage scientific "
                "identity; publish a new checkpoint from the source campaign."
            )
        self.members = tuple(self._members(index) for index in range(stages))

    def region(self, path: str, region: tuple[slice, ...], /) -> np.ndarray:
        block = read_checkpoint_region(
            self.repository, self.manifest, path, region, limits=self.limits
        )
        self.read_bytes += block.nbytes
        return block

    def fields(self, process: int, index: int, /) -> np.ndarray:
        """Controller fields of stage ``index`` recorded by source ``process``."""
        base = _CONTROL_HEAD + _CONTROL_STAGE * index
        return self.controller[process, base : base + _CONTROL_STAGE]

    def _members(self, index: int, /) -> list[int]:
        holders = [
            process
            for process in range(self.processes)
            if self.fields(process, index)[0] == 1
        ]
        if not holders:
            raise AtomisticDomainCampaignCheckpointError("A source stage has no member.")
        return sorted(holders, key=lambda process: int(self.fields(process, index)[1]))

    def one(self, path: str, process: int, index: int, rank: int, /) -> np.ndarray:
        """Source ``process``'s block of stage ``index`` of a rank-``rank`` array."""
        return self.region(
            path,
            (slice(process, process + 1), slice(index, index + 1))
            + (slice(None),) * (rank - 2),
        )[0, 0]


def _verified_packet(header: np.ndarray, payload: np.ndarray, /) -> np.ndarray:
    """The encoded bytes of one checkpointed transport packet, digest-checked."""
    encoded = np.array(payload[: int(header[5])])
    if not np.array_equal(_digest_words(encoded), header[6:]):
        raise AtomisticDomainCampaignCheckpointError(
            "A checkpointed transport packet failed its integrity digest."
        )
    return encoded


def _domain_continuation(
    codec: AtomisticPacketCodec, encoded: np.ndarray, /
) -> tuple[AtomisticDomainPacket, str | None]:
    packet = codec.decode(encoded)
    if not isinstance(packet.continuation, AtomisticDomainPacket):
        raise AtomisticTransportSchemaError("A stage packet is a domain packet.")
    return packet.continuation, packet.family


def _reissued(
    binding: _StageBinding,
    fragments: list[AtomisticDomainPacket],
    family: str | None,
    /,
) -> AtomisticTaskPacket:
    """A pending continuation re-fragmented onto this member's new stage."""
    mine = _routed(fragments, binding.member_index, binding.member_count)
    return AtomisticTaskPacket(
        continuation=binding.dynamics.runtime.merge_packets(mine),
        family=family,
    )


@final
class PreparedAtomisticDomainCampaign:
    """A stage×domain campaign bound to the live root group.

    Preparation is a root-group collective: members of each stage build its
    native domain program on the live child group, ingress the first task to
    fix the continuation schema, and every process agrees the per-stage
    schemas and the transport edges. The campaign's own scheduling is the
    transport schedule; stage work between rounds runs only on members.
    """

    __slots__ = (
        "bindings",
        "execution_id",
        "plan",
        "resident_bytes",
        "root",
        "stage_schemas",
        "transport",
        "_runs",
    )

    def __init__(
        self, plan: AtomisticDomainCampaignPlan, root: ExecutionGroup, /
    ) -> None:
        if not isinstance(plan, AtomisticDomainCampaignPlan):
            raise TypeError("plan must be an AtomisticDomainCampaignPlan.")
        if not isinstance(root, ExecutionGroup):
            raise TypeError("root must be a live ExecutionGroup.")
        if set(root.spec.process_indices) != set(range(jax.process_count())):
            raise ValueError("The campaign root group spans every process of the job.")
        axes = tuple(root.mesh.axis_names)
        if len(axes) != 1:
            raise ValueError("The campaign root group requires a one-dimensional mesh.")
        me = jax.process_index()
        runs = {plan.run_id(index): index for index in range(len(plan.tasks))}
        self.plan = plan
        self.root = root
        self._runs = runs
        bindings: list[_StageBinding | None] = []
        for stage in plan.stages:
            if stage.group.parent_group_id != root.spec.group_id:
                raise ValueError(f"Stage {stage.name!r} is not a child of this root.")
            members = tuple(sorted(stage.group.process_indices))
            if me not in members:
                bindings.append(None)
                continue
            group = bind_execution_group(stage.group)
            dynamics = stage.program(group)
            if not isinstance(dynamics, PreparedAtomisticDomainDynamics):
                raise TypeError(
                    "A stage program returns PreparedAtomisticDomainDynamics."
                )
            ownership = dynamics.runtime.plan.ownership
            if (
                ownership.lane_reference
                or ownership.execution_group.spec.group_id != stage.group.group_id
            ):
                raise ValueError(
                    f"Stage {stage.name!r}: the domain must own exactly its child group "
                    "with device owners."
                )
            rows = ownership.local_capacity * sum(
                1 for process in ownership.owner_processes if process == me
            )
            entry = self._ingress(dynamics, members.index(me), len(members), rows, 0)
            template = AtomisticTaskPacket(
                continuation=dynamics.runtime.to_packet(entry),
                family=plan.tasks[0].task.family,
            )
            continuation = template.continuation
            if not isinstance(continuation, AtomisticDomainPacket):
                raise TypeError("A stage continuation is an AtomisticDomainPacket.")
            bindings.append(
                _StageBinding(
                    group=group,
                    dynamics=dynamics,
                    template=template,
                    codec=AtomisticPacketCodec(template),
                    member_index=members.index(me),
                    member_count=len(members),
                    fragment_rows=rows,
                    constraint_rows=-1
                    if continuation.constraints is None
                    else int(continuation.constraints.row_ids.shape[0]),
                )
            )
        self.bindings = tuple(bindings)
        processes = tuple(sorted(root.spec.process_indices))
        lanes = tuple(
            min(
                index
                for index, device in enumerate(root.devices)
                if device.process_index == process
            )
            for process in processes
        )
        schemas, sizes = self._agree_stages(root, str(axes[0]), processes, lanes)
        for index in range(len(plan.stages) - 1):
            if schemas[index] != schemas[index + 1]:
                raise AtomisticTransportSchemaError(
                    f"Stages {plan.stages[index].name!r} and "
                    f"{plan.stages[index + 1].name!r} hold different continuation "
                    "schemas (model revision, roles, payload, controllers or "
                    "per-process fragment rows)."
                )
        self.stage_schemas = schemas
        self.resident_bytes = max(sizes)
        names = tuple(stage.name for stage in plan.stages)
        edges = []
        for index in range(len(plan.stages) - 1):
            local = bindings[index] or bindings[index + 1]
            edges.append(
                AtomisticTransportEdge(
                    names[index],
                    names[index + 1],
                    None if local is None else local.template,
                    mode="process-local",
                    layout_id=canonical_fingerprint(
                        {
                            "kind": "atomistic-domain-stage-continuation",
                            "analysis": plan.analysis_id,
                            "stage": index,
                            "schema": schemas[index],
                        }
                    ),
                )
            )
        report = _report_template(plan.atom_capacity)
        edges.extend(AtomisticTransportEdge(name, plan.sink, report) for name in names)
        self.transport = AtomisticStageTransportPlan(
            tuple(
                AtomisticTransportStage(stage.name, stage.group) for stage in plan.stages
            ),
            tuple(edges),
            sinks=(plan.sink,),
            outbox_capacity=plan.outbox_capacity,
            inbox_capacity=plan.inbox_capacity,
            lane_slots=plan.lane_slots,
            maximum_rounds=plan.maximum_rounds,
        ).prepare(root, maximum_round_bytes=plan.maximum_round_bytes)
        self.execution_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-domain-campaign",
                "analysis": plan.analysis_id,
                "transport": self.transport.prepared_id,
                "schemas": list(schemas),
                "resident_bytes": self.resident_bytes,
            }
        )

    # -- preparation ----------------------------------------------------------

    def _agree_stages(
        self,
        root: ExecutionGroup,
        axis: str,
        processes: tuple[int, ...],
        lanes: tuple[int, ...],
        /,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Root-group consensus of every stage's continuation schema."""
        width = 6
        row = np.zeros((width * len(self.bindings),), np.int64)
        for index, binding in enumerate(self.bindings):
            if binding is None:
                continue
            row[width * index : width * index + 2] = (1, binding.codec.byte_count)
            row[width * index + 2 : width * (index + 1)] = _identity_words(
                binding.codec.schema_id
            )
        rows = _root_gather(root, axis, processes, lanes, row)
        schemas: list[str] = []
        sizes: list[int] = []
        for index, stage in enumerate(self.plan.stages):
            segment = rows[:, width * index : width * (index + 1)]
            holders = [
                processes.index(process)
                for process in sorted(stage.group.process_indices)
            ]
            reference = segment[holders[0]]
            if any(
                segment[position, 0] != 1
                or not np.array_equal(segment[position], reference)
                for position in holders
            ):
                raise AtomisticTransportSchemaError(
                    f"Members of stage {stage.name!r} disagree on its continuation "
                    "schema."
                )
            schemas.append(np.ascontiguousarray(reference[2:]).tobytes().hex())
            sizes.append(int(reference[1]))
        return tuple(schemas), tuple(sizes)

    def _ingress(
        self,
        dynamics: PreparedAtomisticDomainDynamics,
        member: int,
        members: int,
        rows: int,
        index: int,
        /,
    ) -> AtomisticDomainState:
        """Process-local entry ingress of task ``index`` and its initialization.

        Member ``k`` supplies atoms ``i`` with ``i mod n == k`` as its rows;
        nothing is gathered on a host. Padding rows carry stable IDs disjoint
        from every atom and every other member.
        """
        entry = self.plan.tasks[index]
        task = entry.task
        packet = task.packet
        positions = packet.positions
        native = packet.atom_type_ids
        if positions is None or native is None:
            raise ValueError("A domain task packet carries positions and type ids.")
        count = positions.shape[0]
        mine = np.flatnonzero(np.arange(count) % members == member)
        if mine.size > rows:
            raise ValueError(
                f"Task {task.task_id!r}: {mine.size} atoms exceed this member's "
                f"{rows} owner rows (capacity refusal before any physics)."
            )

        def local(values: np.ndarray, fill: object = 0) -> np.ndarray:
            output = np.full((rows,) + values.shape[1:], fill, dtype=values.dtype)
            output[: mine.size] = values[mine]
            return output

        active = np.zeros((rows,), np.bool_)
        active[: mine.size] = True
        stable = count + member * rows + np.arange(rows, dtype=np.int64)
        stable[: mine.size] = mine
        momenta = np.zeros_like(positions) if packet.momenta is None else packet.momenta
        payload = dynamics.plan.atom_payload(
            mobile=entry.mobile, molecules=entry.molecules, dtype=positions.dtype
        )
        key = jax.random.fold_in(jax.random.key(self.plan.rng_root), task.ordinal)
        state = dynamics.runtime.ingress_process_local(
            local(positions),
            local(native.astype(np.int32)),
            rng_key=key,
            active=active,
            stable_ids=stable,
            logical_indices=local(np.arange(count, dtype=np.int32)),
            logical_count=count,
            momenta=local(momenta),
            masses=local(entry.masses, 1.0),
            cell_vectors=packet.cell_vectors,
            atom_payload={name: local(value) for name, value in payload.items()},
            case_identity=task.material_id,
            run_id=self.plan.run_id(index),
        )
        return dynamics.initialize(state)

    # -- stage work -----------------------------------------------------------

    def _binding(self, index: int, /) -> _StageBinding:
        binding = self.bindings[index]
        if binding is None:
            raise ValueError(
                f"Process {jax.process_index()} is not a member of stage "
                f"{self.plan.stages[index].name!r}."
            )
        return binding

    def _task_of(self, run_id: str, /) -> int:
        index = self._runs.get(run_id)
        if index is None:
            raise AtomisticTransportSchemaError(
                "A continuation belongs to no task of this campaign."
            )
        return index

    def _admit_entry(self, index: int, task_index: int, /) -> AtomisticDomainResident:
        binding = self._binding(index)
        state = self._ingress(
            binding.dynamics,
            binding.member_index,
            binding.member_count,
            binding.fragment_rows,
            task_index,
        )
        return AtomisticDomainResident(
            task_index, 0, AtomisticStageOutcome.ACTIVE, 0, state
        )

    def _admit_delivery(
        self, index: int, delivery: AtomisticTransportDelivery, /
    ) -> AtomisticDomainResident:
        """Ingress a transferred continuation; the stage evaluates its entry.

        Entry priming is the stage's own Hamiltonian evaluation of the
        transferred configuration (no step, time or random draw consumed).
        """
        binding = self._binding(index)
        parts = []
        for fragment in delivery.fragments:
            continuation = fragment.continuation
            if not isinstance(continuation, AtomisticDomainPacket):
                raise AtomisticTransportSchemaError(
                    "Stage transfers carry AtomisticDomainPacket continuations."
                )
            parts.append(continuation)
        task_index = self._task_of(parts[0].run_id)
        state = binding.dynamics.runtime.from_packet(parts)
        primed = binding.dynamics.prime(state)
        successful = bool(_host_replica(primed.successful))
        reasons = int(_host_replica(primed.rejection_reasons))
        return AtomisticDomainResident(
            task_index,
            0,
            AtomisticStageOutcome.ACTIVE if successful else _rejection_outcome(reasons),
            reasons,
            primed.accepted_state,
        )

    def _segment(
        self, index: int, resident: AtomisticDomainResident, /
    ) -> tuple[AtomisticDomainResident, int]:
        binding = self._binding(index)
        stage = self.plan.stages[index]
        state, steps_, failed_, reasons_ = _domain_segment(
            binding.dynamics,
            resident.state,
            jnp.asarray(stage.budget - resident.accepted_steps, jnp.int32),
            stage.segment,
        )
        steps = int(_host_replica(steps_))
        reasons = int(_host_replica(reasons_))
        accepted = resident.accepted_steps + steps
        if bool(_host_replica(failed_)):
            outcome = _rejection_outcome(reasons)
        elif accepted >= stage.budget:
            outcome = AtomisticStageOutcome.COMPLETED
        else:
            outcome = AtomisticStageOutcome.ACTIVE
        return (
            AtomisticDomainResident(
                resident.task_index, accepted, outcome, reasons, state
            ),
            steps,
        )

    def _report(
        self, index: int, resident: AtomisticDomainResident, /
    ) -> AtomisticTaskPacket:
        """Replicated report of a finished visit (child-group gather, members)."""
        binding = self._binding(index)
        runtime = binding.dynamics.runtime
        state = resident.state
        layout = state.layout
        positions, images, momenta, forces, logical, active = (
            _host_replica(value)
            for value in _gathered_rows(
                runtime.plan.ownership,
                (
                    state.positions,
                    state.image_counts,
                    state.momenta,
                    state.force.forces,
                    layout.logical_indices,
                    layout.active,
                ),
            )
        )
        cell = _host_replica(state.cell_vectors).astype(np.float64)
        entry = self.plan.tasks[resident.task_index]
        packet = entry.task.packet
        capacity = self.plan.atom_capacity
        count = entry.atom_count
        logical_rows = logical[active].astype(np.int64)

        def ordered(values: np.ndarray) -> np.ndarray:
            output = np.zeros((capacity, 3), np.float64)
            output[logical_rows] = values[active]
            return output

        def padded(values: np.ndarray | None) -> np.ndarray:
            output = np.zeros((capacity,), np.int32)
            if values is not None:
                output[:count] = values
            return output

        mask = np.zeros((capacity,), np.bool_)
        mask[:count] = True
        force = state.force
        force_step = int(_host_replica(force.step))
        step = int(_host_replica(state.step_index))
        # The evaluated conditions: the domain evaluation's controls (a domain
        # Hamiltonian declares none), the state's evaluated conditioning and the
        # stage's declared thermodynamic state, each through its owner.
        conditions = canonical_fingerprint(
            {
                "kind": "atomistic-domain-evaluation-conditions",
                "controls": _conditions_id(np.zeros((0,)), ()),
                "conditioning": runtime.conditioning_id(state),
                "thermodynamic": binding.dynamics.plan.ensemble.plan_id,
            }
        )
        report = AtomisticDomainStageReport(
            task_index=np.asarray(resident.task_index, np.int64),
            stage_index=np.asarray(index, np.int64),
            outcome=np.asarray(int(resident.outcome), np.int32),
            reasons=np.asarray(resident.reasons, np.int32),
            accepted_steps=np.asarray(resident.accepted_steps, np.int64),
            step_index=np.asarray(step, np.int64),
            energy=_host_replica(force.energy).astype(np.float64).reshape(()),
            forces=ordered(forces.astype(np.float64)),
            stress=_host_replica(force.stress).astype(np.float64),
            stress_available=_host_replica(force.strain_available).astype(np.bool_),
            force_valid=np.asarray(
                bool(_host_replica(force.successful)) and force_step == step, np.bool_
            ),
            hamiltonian_words=np.asarray(runtime.scientific_identity.words, np.uint32),
            revision_words=np.asarray(
                runtime.hamiltonian_revision_identity.words, np.uint32
            ),
            conditions_words=np.asarray(
                IdentityWords.from_fingerprint(conditions).words, np.uint32
            ),
            rng_words=_host_replica(jax.random.key_data(state.rng_key))
            .astype(np.uint32)
            .reshape(-1),
        )
        unwrapped = positions.astype(np.float64) + images.astype(np.float64) @ cell
        return AtomisticTaskPacket(
            positions=ordered(unwrapped),
            cell_vectors=cell,
            momenta=ordered(momenta.astype(np.float64)),
            species=padded(packet.species),
            atom_type_ids=padded(packet.atom_type_ids),
            atom_mask=mask,
            time=float(_host_replica(state.time)),
            continuation=report,
            family=entry.task.family,
        )

    def output(self, packet: AtomisticTaskPacket, /) -> AtomisticCampaignOutput:
        """The campaign output of one report packet (sink host)."""
        report = packet.continuation
        if not isinstance(report, AtomisticDomainStageReport):
            raise AtomisticTransportSchemaError("Sink packets carry a stage report.")
        entry = self.plan.tasks[int(report.task_index)]
        stage = self.plan.stages[int(report.stage_index)]
        outcome = AtomisticStageOutcome(int(report.outcome))
        count = entry.atom_count
        revision = IdentityWords(jnp.asarray(report.revision_words)).fingerprint()
        evaluation = None
        stress = bool(report.stress_available) and bool(
            np.all(np.isfinite(report.stress))
        )
        # Only an accepted, current and finite force cache becomes an evaluation
        # record; a failed visit publishes its configuration and reasons only.
        if (
            outcome is AtomisticStageOutcome.COMPLETED
            and bool(report.force_valid)
            and bool(np.isfinite(report.energy))
            and bool(np.all(np.isfinite(report.forces[:count])))
        ):
            evaluation = AtomisticEvaluationRecord(
                report.energy,
                report.forces[:count],
                report.stress if stress else None,
                AtomisticEvaluationKey(
                    IdentityWords(jnp.asarray(report.hamiltonian_words)).fingerprint(),
                    numeric_revision_ids=(revision,),
                    conditions_id=IdentityWords(
                        jnp.asarray(report.conditions_words)
                    ).fingerprint(),
                    quantities=("energy", "forces", "stress")
                    if stress
                    else ("energy", "forces"),
                ),
            )
        positions = packet.positions
        momenta = packet.momenta
        species = packet.species
        types = packet.atom_type_ids
        if positions is None or momenta is None or species is None or types is None:
            raise AtomisticTransportSchemaError("A report packet is incomplete.")
        configuration = AtomisticTaskPacket(
            positions=positions[:count],
            cell_vectors=packet.cell_vectors,
            momenta=momenta[:count],
            species=species[:count],
            atom_type_ids=types[:count],
            time=packet.time,
            evaluation=evaluation,
        )
        return _task_output(
            self.plan.campaign_id,
            entry.task,
            None,
            (),
            stage.name,
            0,
            int(outcome),
            configuration,
            (revision,),
            report.rng_words,
            {
                "accepted_steps": int(report.accepted_steps),
                "step_index": int(report.step_index),
            },
            int(report.reasons),
            0,
        )

    # -- rounds ---------------------------------------------------------------

    def start(self) -> AtomisticDomainCampaignState:
        """Fresh continuation: empty transport, entry cursor at the first task."""
        return AtomisticDomainCampaignState(
            self.transport.initial_state(),
            0,
            (None,) * len(self.plan.stages),
            0,
        )

    def run_round(
        self,
        state: AtomisticDomainCampaignState,
        /,
        *,
        sink: AbstractAtomisticCampaignSink | None = None,
        cancel: bool = False,
    ) -> AtomisticDomainCampaignRound:
        """Member stage work, then one scheduled transport round (collective).

        Each member stage admits a task when it holds none (the entry stage
        from the task list, others from their inbox), executes one segment of
        its resident's native domain dynamics, and deposits a finished
        resident to its outbox, or parks it when the outbox is full. ``sink``
        is the campaign sink on the transport's sink host.
        """
        if not isinstance(state, AtomisticDomainCampaignState):
            raise TypeError("state must be an AtomisticDomainCampaignState.")
        transport = self.transport
        plan = self.plan
        names = tuple(stage.name for stage in plan.stages)
        me = jax.process_index()
        if me == transport.sink_host and sink is None:
            raise ValueError("The transport sink host supplies the campaign sink.")
        current = state.transport
        cursor = state.cursor
        residents = list(state.residents)
        count = len(names)
        executed = [0] * count
        admitted: list[str | None] = [None] * count
        exported: list[str | None] = [None] * count
        parked = [False] * count
        for index, binding in enumerate(self.bindings):
            if binding is None:
                continue
            resident = residents[index]
            if resident is None:
                if index == 0:
                    if cursor < len(plan.tasks):
                        resident = self._admit_entry(index, cursor)
                        cursor += 1
                else:
                    current, delivery = transport.receive(current, names[index])
                    if delivery is not None:
                        resident = self._admit_delivery(index, delivery)
                if resident is not None:
                    admitted[index] = plan.tasks[resident.task_index].task.task_id
            if resident is not None and resident.outcome is AtomisticStageOutcome.ACTIVE:
                resident, executed[index] = self._segment(index, resident)
            if resident is not None and resident.outcome is not (
                AtomisticStageOutcome.ACTIVE
            ):
                downstream = (
                    resident.outcome is AtomisticStageOutcome.COMPLETED
                    and index + 1 < count
                )
                if transport.outbox_free(current, names[index]) > 0:
                    task = plan.tasks[resident.task_index].task
                    packet = (
                        AtomisticTaskPacket(
                            continuation=binding.dynamics.runtime.to_packet(
                                resident.state
                            ),
                            family=task.family,
                        )
                        if downstream
                        else self._report(index, resident)
                    )
                    current = transport.deposit(
                        current,
                        names[index],
                        packet,
                        destination=names[index + 1] if downstream else plan.sink,
                    )
                    exported[index] = task.task_id
                    resident = None
                else:
                    parked[index] = True
            residents[index] = resident
        drained = all(
            residents[index] is None
            for index, binding in enumerate(self.bindings)
            if binding is not None
        ) and (self.bindings[0] is None or cursor >= len(plan.tasks))
        sinks = (
            {plan.sink: _DomainCampaignSink(self, sink, int(current.round_index))}
            if me == transport.sink_host and sink is not None
            else None
        )
        result = transport.round(current, drained=drained, cancel=cancel, sinks=sinks)
        return AtomisticDomainCampaignRound(
            AtomisticDomainCampaignState(
                result.state, cursor, tuple(residents), state.epoch
            ),
            result.evidence,
            tuple(executed),
            tuple(admitted),
            tuple(exported),
            tuple(parked),
        )

    def run(
        self,
        state: AtomisticDomainCampaignState,
        /,
        *,
        sink: AbstractAtomisticCampaignSink | None = None,
        checkpoint: AtomisticDomainCampaignCheckpointPolicy | None = None,
        stop_after_checkpoints: int | None = None,
        after_publish: Callable[[str], None] | None = None,
    ) -> tuple[
        AtomisticDomainCampaignState,
        tuple[AtomisticDomainCampaignRound, ...],
        tuple[AtomisticDomainCampaignPublication, ...],
    ]:
        """Rounds until consensus termination (bounded by the round schedule).

        With ``checkpoint``, every ``every``-th round ends with one global
        commit; ``stop_after_checkpoints`` returns after that many commits.
        """
        rounds: list[AtomisticDomainCampaignRound] = []
        publications: list[AtomisticDomainCampaignPublication] = []
        current = state
        while not current.terminated:
            result = self.run_round(current, sink=sink)
            rounds.append(result)
            current = result.state
            if (
                checkpoint is not None
                and not current.terminated
                and current.round_index % checkpoint.every == 0
            ):
                publication = self.checkpoint(
                    current,
                    checkpoint.repository,
                    checkpoint.checkpoint_id(current.epoch, current.round_index),
                    writer_id=checkpoint.writer_id,
                    after_publish=after_publish,
                    limits=checkpoint.limits,
                )
                publications.append(publication)
                if (
                    stop_after_checkpoints is not None
                    and len(publications) >= stop_after_checkpoints
                ):
                    break
        return current, tuple(rounds), tuple(publications)

    # -- durability -----------------------------------------------------------

    def _encoded_resident(
        self, index: int, resident: AtomisticDomainResident, /
    ) -> np.ndarray:
        binding = self._binding(index)
        task = self.plan.tasks[resident.task_index].task
        return binding.codec.encode(
            AtomisticTaskPacket(
                continuation=binding.dynamics.runtime.to_packet(resident.state),
                family=task.family,
            )
        )

    def _blocks(self, state: AtomisticDomainCampaignState, /) -> dict[str, np.ndarray]:
        count = len(self.plan.stages)
        controller = np.zeros((_CONTROL_HEAD + _CONTROL_STAGE * count,), np.int64)
        controller[:_CONTROL_HEAD] = (state.cursor, state.epoch)
        identity = np.zeros((count, _IDENTITY_ROWS, 8), np.uint32)
        residents = np.zeros((count, self.resident_bytes), np.uint8)
        digests = np.zeros((count, 4), np.int64)
        structures = np.zeros((count, 4), np.int64)
        for index, binding in enumerate(self.bindings):
            if binding is None:
                continue
            base = _CONTROL_HEAD + _CONTROL_STAGE * index
            controller[base : base + 5] = (
                1,
                binding.member_index,
                binding.member_count,
                binding.fragment_rows,
                binding.constraint_rows,
            )
            identity[index] = binding.identity
            resident = state.residents[index]
            if resident is None:
                continue
            encoded = self._encoded_resident(index, resident)
            controller[base + 5 : base + 11] = (
                1,
                resident.task_index,
                resident.accepted_steps,
                int(resident.outcome),
                resident.reasons,
                encoded.size,
            )
            residents[index, : encoded.size] = encoded
            digests[index] = _digest_words(encoded)
            paths, leaves, _ = _flat_state(resident.state)
            structures[index] = _identity_words(
                _structure_id(resident.state, paths, leaves)
            )
        schemas = np.stack([_identity_words(schema) for schema in self.stage_schemas])
        return {
            "controller": controller,
            "digests": digests,
            "identity": identity,
            "residents": residents,
            "schemas": schemas,
            "structures": structures,
        }

    def _resident_states(
        self, state: AtomisticDomainCampaignState, /
    ) -> dict[str, dict[str, jax.Array]]:
        """Every leaf of each member stage's resident domain state, unchanged.

        The owning domain-continuation leaf contract (`_domain_checkpoint`):
        owner-blocked rows, replicated controllers/ledger, force cache with
        owner energies, every role topology epoch and relation leaves, typed
        RNG keys as key data. Each leaf is a global array of the stage group;
        the lifecycle publication writes its canonical addressable shards from
        the members only.
        """
        trees: dict[str, dict[str, jax.Array]] = {}
        for index, binding in enumerate(self.bindings):
            resident = state.residents[index]
            if binding is None or resident is None:
                continue
            _, leaves, _ = _flat_state(resident.state)
            trees[f"{index:02d}"] = {
                f"{position:06d}": _leaf_data(leaf)[0]
                for position, leaf in enumerate(leaves)
            }
        return trees

    def checkpoint(
        self,
        state: AtomisticDomainCampaignState,
        repository: ArtifactRepository,
        checkpoint_id: str,
        /,
        *,
        writer_id: str,
        after_publish: Callable[[str], None] | None = None,
        limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ) -> AtomisticDomainCampaignPublication:
        """Collective durable commit of the complete campaign continuation.

        Every process publishes its transport state (credits, parked and
        committed packets, sequences, receipts counter, transfer digest), its
        campaign controller (entry cursor, epoch, per-stage membership,
        resident task, visit progress, outcome and reasons), the prepared
        runtime/revision/dynamics identities, and for each resident of its
        stages both the complete distributed domain state (every leaf,
        including the force cache, role topologies, controllers, ledger and
        RNG key) and the encoded domain packet (the changed-topology
        continuation). The coordinator then writes the single global commit;
        partial coverage never restores. Publication reads the accepted state
        and changes nothing: the caller continues with the same ``state``.
        ``after_publish`` runs after this process's shard is durable and
        before the commit barrier.
        """
        _label("checkpoint_id", checkpoint_id)
        transport = self.transport
        tree = {
            "transport": transport.process_blocked(state.transport),
            "campaign": {
                name: transport.blocked(value)
                for name, value in self._blocks(state).items()
            },
            "residents": self._resident_states(state),
        }
        publish_process_checkpoint(
            repository,
            checkpoint_id,
            self.execution_id,
            tree,
            analysis_plan_id=self.plan.analysis_id,
            numeric_revision_id=self.plan.revision_id,
            writer_id=writer_id,
            limits=limits,
        )
        if after_publish is not None:
            after_publish(checkpoint_id)
        _global_barrier(f"atomistic-domain-campaign:{checkpoint_id}:published")
        if jax.process_index() == transport.processes[0]:
            manifest = assemble_distributed_checkpoint_from_repository(
                repository,
                checkpoint_id,
                self.plan.analysis_id,
                self.plan.revision_id,
                self.execution_id,
                expected_process_count=len(transport.processes),
                limits=limits,
            )
            commit_distributed_checkpoint(
                repository,
                manifest,
                expected_process_count=len(transport.processes),
                writer_id=writer_id,
                limits=limits,
            )
        _global_barrier(f"atomistic-domain-campaign:{checkpoint_id}:committed")
        committed = load_distributed_checkpoint_commit(
            repository, checkpoint_id, limits=limits
        )
        return AtomisticDomainCampaignPublication(
            checkpoint_id=checkpoint_id,
            manifest_id=committed.manifest_id,
            execution_id=self.execution_id,
            analysis_id=self.plan.analysis_id,
            round_index=state.round_index,
        )

    def _committed(
        self,
        repository: ArtifactRepository,
        checkpoint_id: str,
        limits: ArrayArchiveLimits,
        /,
    ) -> CheckpointManifest:
        manifest = load_distributed_checkpoint_commit(
            repository, checkpoint_id, limits=limits
        )
        if manifest.analysis_plan_id != self.plan.analysis_id:
            raise AtomisticDomainCampaignCheckpointError(
                "The checkpoint belongs to another campaign (tasks, stages, "
                "programs or RNG root)."
            )
        if manifest.numeric_revision_id != self.plan.revision_id:
            raise AtomisticDomainCampaignCheckpointError(
                "The checkpoint was written by other stage programs."
            )
        return manifest

    def restore(
        self,
        repository: ArtifactRepository,
        checkpoint_id: str,
        /,
        *,
        limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ) -> AtomisticDomainCampaignState:
        """Same-topology restart from a global commit (collective, bitwise).

        Each process restores only its own blocks. Every resident's complete
        domain state is restored leaf for leaf into the shardings of a
        structure template (its own packet's ingress, admitted only when the
        `_domain_checkpoint` structure identity matches the checkpointed
        state's), so force cache, topologies, controllers, ledger and RNG key
        are exactly the accepted ones. A changed topology is refused
        (`migrate`).
        """
        manifest = self._committed(repository, checkpoint_id, limits)
        if manifest.execution_plan_id != self.execution_id:
            raise AtomisticDomainCampaignCheckpointError(
                "The checkpoint was written by another execution topology; "
                "continue it with migrate()."
            )
        transport = self.transport
        state = transport.from_process_blocked(
            {
                name: restore_global_array_from_checkpoint(
                    repository,
                    manifest,
                    f"['transport']['{name}']",
                    transport.process_block_sharding(name),
                    limits=limits,
                )
                for name in _STATE_FIELDS
            }
        )
        template = self._blocks(self.start())
        blocks = {
            name: transport.own_block(
                restore_global_array_from_checkpoint(
                    repository,
                    manifest,
                    f"['campaign']['{name}']",
                    transport.block_sharding(value.ndim),
                    limits=limits,
                )
            )
            for name, value in template.items()
        }
        controller = blocks["controller"]
        if blocks["identity"].shape != template["identity"].shape:
            raise AtomisticDomainCampaignCheckpointError(
                "The checkpoint records no layout-independent stage scientific "
                "identity; publish a new checkpoint from this campaign."
            )
        if not np.array_equal(blocks["schemas"], template["schemas"]):
            raise AtomisticDomainCampaignCheckpointError(
                "The checkpoint's continuation schemas differ from this campaign."
            )
        residents: list[AtomisticDomainResident | None] = []
        for index, binding in enumerate(self.bindings):
            base = _CONTROL_HEAD + _CONTROL_STAGE * index
            fields = controller[base : base + _CONTROL_STAGE]
            if binding is None:
                if fields[0] != 0:
                    raise AtomisticDomainCampaignCheckpointError(
                        "Checkpoint stage membership differs from this process's."
                    )
                residents.append(None)
                continue
            if tuple(int(value) for value in fields[:5]) != (
                1,
                binding.member_index,
                binding.member_count,
                binding.fragment_rows,
                binding.constraint_rows,
            ) or not np.array_equal(blocks["identity"][index], binding.identity):
                raise AtomisticDomainCampaignCheckpointError(
                    f"Stage {self.plan.stages[index].name!r} was checkpointed with "
                    "another membership, runtime, revision, dynamics plan or "
                    "scientific program."
                )
            if fields[5] == 0:
                residents.append(None)
                continue
            encoded = blocks["residents"][index, : int(fields[10])]
            if not np.array_equal(_digest_words(encoded), blocks["digests"][index]):
                raise AtomisticDomainCampaignCheckpointError(
                    "A resident continuation failed its integrity digest."
                )
            continuation = binding.codec.decode(encoded).continuation
            if not isinstance(continuation, AtomisticDomainPacket):
                raise AtomisticTransportSchemaError("A resident is a domain packet.")
            template_state = binding.dynamics.runtime.from_packet((continuation,))
            paths, leaves, treedef = _flat_state(template_state)
            if not np.array_equal(
                _identity_words(_structure_id(template_state, paths, leaves)),
                blocks["structures"][index],
            ):
                raise AtomisticDomainCampaignCheckpointError(
                    f"Stage {self.plan.stages[index].name!r}: the resident's "
                    "checkpointed state structure differs from its continuation."
                )
            restored = []
            for position, leaf in enumerate(leaves):
                data = restore_global_array_from_checkpoint(
                    repository,
                    manifest,
                    f"['residents']['{index:02d}']['{position:06d}']",
                    leaf.sharding,
                    limits=limits,
                    committed=leaf.committed,
                )
                restored.append(
                    jax.random.wrap_key_data(data, impl=jax.random.key_impl(leaf))
                    if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
                    else data
                )
            residents.append(
                AtomisticDomainResident(
                    int(fields[6]),
                    int(fields[7]),
                    AtomisticStageOutcome(int(fields[8])),
                    int(fields[9]),
                    jax.tree_util.tree_unflatten(treedef, restored),
                )
            )
        return AtomisticDomainCampaignState(
            state, int(controller[0]), tuple(residents), int(controller[1]) + 1
        )

    def _admit_source_stages(
        self, source: _MigrationSource, /
    ) -> list[AtomisticPacketCodec | None]:
        """Scientific admission of every member stage against its source members.

        Returns each member stage's codec of the source fragment layout (None
        for stages this process does not belong to).
        """
        source_codecs: list[AtomisticPacketCodec | None] = []
        for index, binding in enumerate(self.bindings):
            if binding is None:
                source_codecs.append(None)
                continue
            head = source.members[index][0]
            # Rows 0-2 bind the source execution layout; only the
            # layout-independent scientific row is comparable across topologies
            # (it subsumes the layered model revision).
            if any(
                not np.array_equal(
                    source.identity[process, index, _SCIENCE_ROW],
                    binding.identity[_SCIENCE_ROW],
                )
                for process in source.members[index]
            ):
                raise AtomisticDomainCampaignCheckpointError(
                    f"Stage {self.plan.stages[index].name!r} was checkpointed with "
                    "another scientific program (model, terms, roles, cell, "
                    "integrator, ensemble, barostat, schedule or constraints)."
                )
            codec = AtomisticPacketCodec(
                _resized(
                    binding.template,
                    int(source.fields(head, index)[3]),
                    int(source.fields(head, index)[4]),
                )
            )
            # The checkpoint holds the words of the agreed stage label
            # (`_blocks`); the codec reconstructs the source fragment layout.
            if not np.array_equal(
                _identity_words(_stage_schema(codec.schema_id)),
                source.schemas[head, index],
            ):
                raise AtomisticDomainCampaignCheckpointError(
                    f"Stage {self.plan.stages[index].name!r} continuations of the "
                    "checkpoint do not match this program's schema."
                )
            source_codecs.append(codec)
        return source_codecs

    def _migrated_resident(
        self,
        source: _MigrationSource,
        index: int,
        binding: _StageBinding,
        codec: AtomisticPacketCodec,
        /,
    ) -> AtomisticDomainResident | None:
        """The resident of stage ``index``, resumed on this member's rows."""
        runtime = binding.dynamics.runtime
        head = source.fields(source.members[index][0], index)
        if head[5] != 1:
            return None
        fragments = []
        for process in source.members[index]:
            size = int(source.fields(process, index)[10])
            encoded = np.array(
                source.one("['campaign']['residents']", process, index, 3)[:size]
            )
            if not np.array_equal(
                _digest_words(encoded),
                source.one("['campaign']['digests']", process, index, 3),
            ):
                raise AtomisticDomainCampaignCheckpointError(
                    "A resident continuation failed its integrity digest."
                )
            fragments.append(_domain_continuation(codec, encoded)[0])
        return AtomisticDomainResident(
            int(head[6]),
            int(head[7]),
            AtomisticStageOutcome(int(head[8])),
            int(head[9]),
            runtime.from_packet(
                _routed(fragments, binding.member_index, binding.member_count)
            ),
        )

    def _committed_inbox(
        self,
        source: _MigrationSource,
        index: int,
        codec: AtomisticPacketCodec,
        binding: _StageBinding,
        /,
    ) -> list[tuple[int, AtomisticTaskPacket]]:
        """Committed, unreceived transfers into stage ``index``, oldest first,
        with their source sequences (``binding`` is the receiving stage's)."""
        holders = source.members[index]
        orders = source.one("['transport']['inbox_order']", holders[0], index, 3)
        headers = {
            process: source.one("['transport']['inbox_header']", process, index, 5)
            for process in holders
        }
        payloads = {
            process: source.one("['transport']['inbox_bytes']", process, index, 5)
            for process in holders
        }
        packets = []
        for slot in sorted(np.flatnonzero(orders >= 0), key=lambda s: orders[s]):
            rows: dict[int, AtomisticDomainPacket] = {}
            family: str | None = None
            sequences: set[int] = set()
            for process in holders:
                for position, header in enumerate(headers[process][slot]):
                    if header[0] == 0:
                        continue
                    sequences.add(int(header[2]))
                    if int(header[4]) != _ROWS:
                        continue
                    rows[int(header[3])], family = _domain_continuation(
                        codec,
                        _verified_packet(header, payloads[process][slot, position]),
                    )
            if not rows:
                raise AtomisticDomainCampaignCheckpointError(
                    "A committed transfer has no row fragment in the checkpoint."
                )
            if len(sequences) != 1:
                raise AtomisticDomainCampaignCheckpointError(
                    "A committed transfer's fragments carry different sequences."
                )
            packets.append(
                (
                    sequences.pop(),
                    _reissued(binding, [rows[key] for key in sorted(rows)], family),
                )
            )
        return packets

    def _parked_outbox(
        self,
        source: _MigrationSource,
        index: int,
        codec: AtomisticPacketCodec,
        binding: _StageBinding,
        /,
    ) -> list[AtomisticTaskPacket]:
        """Parked outbound packets of stage ``index`` in sequence order."""
        stages = len(self.plan.stages)
        holders = source.members[index]
        headers = {
            process: source.one("['transport']['outbox_header']", process, index, 4)
            for process in holders
        }
        payloads = {
            process: source.one("['transport']['outbox_bytes']", process, index, 4)
            for process in holders
        }
        head = headers[holders[0]]
        packets = []
        for slot in sorted(np.flatnonzero(head[:, 0] == 1), key=lambda s: head[s, 2]):
            edge = int(head[slot, 1])
            if edge < stages - 1:
                fragments = []
                family: str | None = None
                for process in holders:
                    fragment, family = _domain_continuation(
                        codec,
                        _verified_packet(headers[process][slot], payloads[process][slot]),
                    )
                    fragments.append(fragment)
                packets.append(_reissued(binding, fragments, family))
            else:
                packets.append(
                    self.transport.codec(stages - 1 + index).decode(
                        _verified_packet(head[slot], payloads[holders[0]][slot])
                    )
                )
        return packets

    def migrate(
        self,
        repository: ArtifactRepository,
        checkpoint_id: str,
        /,
        *,
        limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ) -> tuple[AtomisticDomainCampaignState, AtomisticDomainCampaignMigration]:
        """Changed-topology continuation of a committed checkpoint (collective).

        Admitted when the campaign is the same and every stage's derived
        layout-independent scientific identity
        (`PreparedAtomisticDomainDynamics.scientific_identity`: model, terms,
        roles, cell, integrator, ensemble, barostat, schedule, constraints)
        equals the one each source member recorded, while the execution
        topology differs (process count, stage groups, owners or
        capacities). Every process reads only the authenticated regions of
        the source processes it needs: controllers and identities (small), the
        resident, inbox and outbox regions of its own stages' source members.
        Pending continuations are re-fragmented by the process-local edge
        rule (`PreparedAtomisticDomainRuntime.merge_packets`) and keep their
        FIFO order: a resident resumes its visit on the new stage group;
        continuations committed into a stage's inbox stay committed and are
        recommitted, in commit order, into that stage's new inbox
        (`PreparedAtomisticStageTransport.recommit`); continuations and reports
        parked in a stage's outbox are re-issued, oldest first, from its new
        outbox (idempotent sink publication by semantic key). Scientific
        state, visit progress and RNG lineage are preserved; the trajectory
        continues within roundoff, not bitwise.
        """
        manifest = self._committed(repository, checkpoint_id, limits)
        if manifest.execution_plan_id == self.execution_id:
            raise AtomisticDomainCampaignCheckpointError(
                "Same execution topology: restore() continues it bitwise."
            )
        stages = len(self.plan.stages)
        source = _MigrationSource(repository, manifest, limits, stages)
        anchor = source.members[0][0]
        entry_head = int(source.controller[anchor, 0])
        epoch = int(source.controller[anchor, 1])
        source_codecs = self._admit_source_stages(source)

        transport = self.transport
        names = tuple(stage.name for stage in self.plan.stages)
        current = transport.initial_state()
        residents: list[AtomisticDomainResident | None] = []
        reissued = [0] * stages
        for index, binding in enumerate(self.bindings):
            codec = source_codecs[index]
            if binding is None or codec is None:
                residents.append(None)
                continue
            residents.append(self._migrated_resident(source, index, binding, codec))
            if index > 0:
                # Committed into this stage's inbox: stays committed, in order.
                for sequence, packet in self._committed_inbox(
                    source, index, codec, binding
                ):
                    current = transport.recommit(
                        current,
                        names[index],
                        packet,
                        source=names[index - 1],
                        sequence=sequence,
                    )
                    reissued[index] += 1
            for packet in self._parked_outbox(source, index, codec, binding):
                downstream = isinstance(packet.continuation, AtomisticDomainPacket)
                current = transport.deposit(
                    current,
                    names[index],
                    packet,
                    destination=names[index + 1] if downstream else self.plan.sink,
                )
                reissued[index] += 1
        source_digest = source.region(
            "['transport']['transfer_digest']", (slice(anchor, anchor + 1), slice(None))
        )[0]
        source_round = source.region(
            "['transport']['round_index']", (slice(anchor, anchor + 1),)
        )[0]
        state = AtomisticDomainCampaignState(
            current, entry_head, tuple(residents), epoch + 1
        )
        return state, AtomisticDomainCampaignMigration(
            source_checkpoint_id=checkpoint_id,
            source_manifest_id=manifest.manifest_id,
            source_execution_id=manifest.execution_plan_id,
            target_execution_id=self.execution_id,
            source_process_count=source.processes,
            target_process_count=len(transport.processes),
            source_round_index=int(source_round),
            source_transfer_digest=np.asarray(source_digest, np.uint8).tobytes().hex(),
            residents=tuple(
                None
                if resident is None
                else self.plan.tasks[resident.task_index].task.task_id
                for resident in residents
            ),
            reissued=tuple(reissued),
            read_bytes=source.read_bytes,
        )


def _global_barrier(name: str, /) -> None:
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(name)


# ---------------------------------------------------------------------------
# Multi-process probe
# ---------------------------------------------------------------------------


def _probe_packet(value: float, /) -> AtomisticTaskPacket:
    positions = np.arange(12, dtype=np.float64).reshape(4, 3) * value
    return AtomisticTaskPacket(
        positions=positions,
        species=np.ones((4,), np.int32),
        atom_mask=np.ones((4,), np.bool_),
        time=value,
    )


class _ProbeRound(TypedDict):
    """One transport round of the two-process probe report."""

    granted: int
    committed: int
    parked: int
    credits: list[int]
    digest: str


class _ProbeReport(TypedDict):
    """JSON report of one rank of `stage_transport_probe_rank`."""

    rank: int
    physics: list[float]
    received: list[float]
    rounds: list[_ProbeRound]
    manifest_id: str
    transfer_digest: str
    restored_bitwise: bool


def stage_transport_probe_rank(directory: str, scenario: str, /) -> _ProbeReport:
    """Tiny real two-process stage-transport probe (one rank of the job).

    Launch with `phydrax._execution_launch.launch_local_processes` on two
    processes. Stage ``produce`` is the process-subset child group of process
    0 and ``consume`` that of process 1; each stage's physics is a child-group
    collective (a psum over the stage mesh) entered only by its member. Three
    packets cross the root group through a one-slot consumer inbox, so
    credits saturate and packets park upstream. The transport state is then
    published per process and globally committed by the coordinator.
    ``scenario="kill"`` makes process 1 die (SIGKILL) after publishing its
    shard but before the global commit: the coupled job must fail and no
    commit may exist. Returns (and prints) a JSON-ready report.
    """
    from jax.experimental import multihost_utils

    from .._execution_runtime import ExecutionRuntime
    from ..lifecycle._repository import (
        HPCFilesystemProfile,
        POSIXArtifactRepository,
        POSIXRepositoryPolicy,
    )

    if scenario not in ("complete", "kill"):
        raise ValueError("scenario must be 'complete' or 'kill'.")
    runtime = ExecutionRuntime.current()
    if jax.process_count() != 2:
        raise RuntimeError("The stage-transport probe runs on exactly two processes.")
    root = runtime.root_group
    produce = process_subset_group_spec(root.spec, (0,))
    consume = process_subset_group_spec(root.spec, (1,))
    template = _probe_packet(1.0)
    plan = AtomisticStageTransportPlan(
        (
            AtomisticTransportStage("produce", produce),
            AtomisticTransportStage("consume", consume),
        ),
        (AtomisticTransportEdge("produce", "consume", template),),
        outbox_capacity=3,
        inbox_capacity=1,
        lane_slots=1,
        maximum_rounds=16,
    )
    prepared = plan.prepare(root, maximum_round_bytes=1 << 24)
    state = prepared.initial_state()
    me = jax.process_index()
    physics: list[float] = []

    def stage_physics(stage: str, packet: AtomisticTaskPacket) -> float:
        positions = packet.positions
        if positions is None:
            raise AtomisticTransportSchemaError("Probe packets carry positions.")
        group = prepared.stage_group(stage)
        axis = group.mesh.axis_names[0]
        count = len(group.devices)
        sharding = NamedSharding(group.mesh, PartitionSpec(axis))
        rows = np.broadcast_to(positions.reshape(1, -1), (count, 12)).copy()
        local = [
            jax.device_put(rows[index : index + 1], device)
            for index, device in enumerate(group.devices)
            if device.process_index == me
        ]
        values = jax.make_array_from_single_device_arrays((count, 12), sharding, local)
        total = jax.shard_map(
            lambda block: JaxCollectiveProvider(axis).sum(block.sum()),
            mesh=group.mesh,
            in_specs=PartitionSpec(axis),
            out_specs=PartitionSpec(),
            check_vma=False,
        )(values)
        return float(np.asarray(total.addressable_data(0)))

    if prepared.is_member("produce"):
        for value in (1.0, 2.0, 3.0):
            packet = _probe_packet(value)
            physics.append(stage_physics("produce", packet))
            state = prepared.deposit(state, "produce", packet)
    rounds: list[_ProbeRound] = []
    received: list[float] = []
    while True:
        drained = (
            prepared.is_member("produce") and prepared.outbox_free(state, "produce") == 3
        ) or (prepared.is_member("consume") and len(received) == 3)
        result = prepared.round(state, drained=drained)
        state = result.state
        rounds.append(
            {
                "granted": result.evidence.granted,
                "committed": result.evidence.committed,
                "parked": result.evidence.parked,
                "credits": list(result.evidence.credits),
                "digest": result.evidence.transfer_digest,
            }
        )
        if result.evidence.terminated:
            break
        if prepared.is_member("consume"):
            state, delivery = prepared.receive(state, "consume")
            if delivery is not None:
                received.append(stage_physics("consume", delivery.fragments[0]))

    profile = HPCFilesystemProfile(
        "posix.stage-transport-probe",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    repository = POSIXArtifactRepository(
        os.path.join(directory, "repository"),
        POSIXRepositoryPolicy(
            profile, maximum_chunk_bytes=1 << 16, maximum_metadata_bytes=1 << 20
        ),
    )
    checkpoint_id = f"stage-transport-probe-{scenario}"
    publish_process_checkpoint(
        repository,
        checkpoint_id,
        prepared.prepared_id,
        {"transport": prepared.process_blocked(state)},
        analysis_plan_id=plan.plan_id,
        numeric_revision_id="stage-transport-probe",
        writer_id=f"rank-{me}",
    )
    if scenario == "kill" and me == 1:
        os.kill(os.getpid(), signal.SIGKILL)
    multihost_utils.sync_global_devices("stage-transport-probe-published")
    if me == 0:
        manifest = assemble_distributed_checkpoint_from_repository(
            repository,
            checkpoint_id,
            plan.plan_id,
            "stage-transport-probe",
            prepared.prepared_id,
            expected_process_count=2,
        )
        commit_distributed_checkpoint(
            repository, manifest, expected_process_count=2, writer_id="rank-0"
        )
    multihost_utils.sync_global_devices("stage-transport-probe-committed")
    manifest = load_distributed_checkpoint_commit(repository, checkpoint_id)
    restored = prepared.from_process_blocked(
        {
            name: restore_global_array_from_checkpoint(
                repository,
                manifest,
                f"['transport']['{name}']",
                prepared.process_block_sharding(name),
            )
            for name in _STATE_FIELDS
        }
    )
    report: _ProbeReport = {
        "rank": me,
        "physics": physics,
        "received": received,
        "rounds": rounds,
        "manifest_id": manifest.manifest_id,
        "transfer_digest": state.transfer_digest.tobytes().hex(),
        "restored_bitwise": all(
            np.array_equal(getattr(restored, name), getattr(state, name))
            for name in _STATE_FIELDS
        ),
    }
    print(json.dumps(report, sort_keys=True))
    return report


__all__ = (
    "AtomisticDomainCampaignCheckpointError",
    "AtomisticDomainCampaignCheckpointPolicy",
    "AtomisticDomainCampaignMigration",
    "AtomisticDomainCampaignPlan",
    "AtomisticDomainCampaignPublication",
    "AtomisticDomainCampaignRound",
    "AtomisticDomainCampaignState",
    "AtomisticDomainCampaignTask",
    "AtomisticDomainResident",
    "AtomisticDomainStage",
    "AtomisticDomainStageProgram",
    "AtomisticDomainStageReport",
    "AtomisticPacketCodec",
    "AtomisticStageTransportPlan",
    "AtomisticStageTransportState",
    "AtomisticTransportCapacityError",
    "AtomisticTransportClosedError",
    "AtomisticTransportDelivery",
    "AtomisticTransportDivergenceError",
    "AtomisticTransportEdge",
    "AtomisticTransportRound",
    "AtomisticTransportRoundEvidence",
    "AtomisticTransportSchemaError",
    "AtomisticTransportSink",
    "AtomisticTransportStage",
    "PreparedAtomisticDomainCampaign",
    "PreparedAtomisticStageTransport",
    "process_subset_group_spec",
    "stage_transport_probe_rank",
)
