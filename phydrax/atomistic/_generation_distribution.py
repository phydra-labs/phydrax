#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit execution-group placement of atomistic generation streams.

`AtomisticGenerationPlacement` binds one generation stream
(`AtomisticGenerationStreamPlan.prepare(placement=...)`) to a live
`ExecutionGroup`. Every member process is one participant; the stream keeps
one global ordinal window, accepted quota and attempt budget, and every
participant holds the same replicated cursor, pending bank and receipt chain
plus its own sampler lanes.

- **Participants.** Group member processes in ascending process order; a
  participant's position is its `ExecutionGroupRank` (group-local, never the
  global process index). Placement construction on a nonparticipant process
  is refused, so a nonparticipant never enters a stream collective.
- **Exchange.** One fixed-shape all-gather per stream transaction over a mesh
  of one exchange device per participant (`JaxCollectiveProvider`), entered
  by every participant in the same order even when nothing moves, through one
  stable compiled entrypoint per block layout. Boolean blocks travel as
  ``uint8``.
- **Capacities.** ``maximum_round_bytes`` bounds the gathered bytes of one
  exchange and ``maximum_replicated_bytes`` the replicated continuation
  (cursor, pending bank, root key and receipt chain) resident on every
  participant; a prepared stream that exceeds either is refused before any
  work.
- **Publication authority.** The first participant hosts the sink; the other
  participants learn its receipt through one exchange and commit
  identically.
- **Persistence.** `publish_placement_checkpoint` publishes participant
  blocks (participant-blocked arrays) and replicated arrays through the
  lifecycle owner with the group-local rank as writer index, then the
  authority writes the single global commit; `restore_placement_checkpoint`
  restores them on the same placement.

Participant positions order free lanes for placement only; they never enter
proposal identities, keys or publication order. A participant that
disappears leaves its peers blocked inside an exchange; the launcher fails the
coupled job within its deadline and the run restarts from its last global
commit. No collective rollback after the loss of a participant is promised.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from .._data_plane._distributed import ExecutionGroupRank
from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..backends.distributed import JaxCollectiveProvider
from ..lifecycle._chunk_repository import ArtifactRepository, ChunkEncoding
from ..lifecycle._distributed_checkpoint import (
    assemble_distributed_checkpoint_from_repository,
    commit_distributed_checkpoint,
    load_distributed_checkpoint_commit,
    publish_process_checkpoint,
    restore_global_array_from_checkpoint,
)
from ..lifecycle._models import CheckpointManifest


_AXIS = "participant"


class AtomisticGenerationCapacityError(ValueError):
    """A declared generation capacity refused the request.

    Capacity refusal is distinct from scientific rejection: nothing was drawn,
    advanced or lost.
    """


class AtomisticGenerationDivergenceError(RuntimeError):
    """Participants of one placed stream disagree on its replicated state."""


class AtomisticGenerationCheckpointError(ValueError):
    """A committed generation checkpoint does not match the stream or placement."""


def _positive_bytes(value: int, name: str, /) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@final
class AtomisticGenerationPlacement(StrictModule, NonTrainableState):
    """One generation stream placed on a live execution group (this process).

    ``rank`` is this participant's `ExecutionGroupRank` along the group's
    first mesh axis; ``processes`` are all participants in rank order, each
    exchanging through its first group device. The first participant is the
    publication authority. ``placement_id`` binds the group and its exchange
    devices; the byte bounds are execution capacities and stay outside it.
    """

    group: ExecutionGroup = eqx.field(static=True)
    rank: ExecutionGroupRank = eqx.field(static=True)
    processes: tuple[int, ...] = eqx.field(static=True)
    mesh: Mesh = eqx.field(static=True)
    maximum_round_bytes: int = eqx.field(static=True)
    maximum_replicated_bytes: int = eqx.field(static=True)
    placement_id: str = eqx.field(static=True)

    def __init__(
        self,
        group: ExecutionGroup,
        /,
        *,
        maximum_round_bytes: int,
        maximum_replicated_bytes: int,
    ) -> None:
        if not isinstance(group, ExecutionGroup):
            raise TypeError("group must be a live ExecutionGroup.")
        round_bytes = _positive_bytes(maximum_round_bytes, "maximum_round_bytes")
        replicated_bytes = _positive_bytes(
            maximum_replicated_bytes, "maximum_replicated_bytes"
        )
        # Refuses a nonparticipant process: it never binds a stream collective.
        rank = ExecutionGroupRank.current(group, axis=str(group.mesh.axis_names[0]))
        processes = tuple(sorted(group.spec.process_indices))
        exchange: list[jax.Device] = []
        for process in processes:
            owned = [
                device for device in group.devices if device.process_index == process
            ]
            if not owned:
                raise ValueError(f"Group process {process} holds no group device.")
            exchange.append(owned[0])
        self.group = group
        self.rank = rank
        self.processes = processes
        self.mesh = Mesh(np.asarray(exchange, dtype=object), (_AXIS,))
        self.maximum_round_bytes = round_bytes
        self.maximum_replicated_bytes = replicated_bytes
        self.placement_id = canonical_fingerprint(
            {
                "kind": "atomistic-generation-placement",
                "group_id": group.spec.group_id,
                "processes": list(processes),
                "exchange_devices": [
                    [device.process_index, device.id] for device in exchange
                ],
            }
        )

    @property
    def participants(self) -> int:
        return len(self.processes)

    @property
    def position(self) -> int:
        """This participant's group-local rank (placement order only)."""
        return self.rank.rank

    @property
    def is_authority(self) -> bool:
        return self.rank.rank == 0

    @property
    def device(self) -> jax.Device:
        """This participant's exchange device."""
        return self.mesh.devices[self.rank.rank]

    def admit(self, *, round_bytes: int, replicated_bytes: int) -> None:
        """Refuse a stream whose exchange or replicated continuation exceeds a bound."""
        if round_bytes > self.maximum_round_bytes:
            raise AtomisticGenerationCapacityError(
                f"One generation exchange needs {round_bytes} bytes, above the "
                f"admitted {self.maximum_round_bytes}."
            )
        if replicated_bytes > self.maximum_replicated_bytes:
            raise AtomisticGenerationCapacityError(
                f"The replicated generation continuation needs {replicated_bytes} "
                f"bytes on every participant, above the admitted "
                f"{self.maximum_replicated_bytes}."
            )

    def _block_sharding(self, rank: int, /) -> NamedSharding:
        return NamedSharding(self.mesh, PartitionSpec(_AXIS, *([None] * rank)))

    def _replicated_sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, PartitionSpec())

    def _blocked(self, block: np.ndarray, /) -> jax.Array:
        """Global ``(participants, ...)`` array holding this participant's block."""
        return jax.make_array_from_single_device_arrays(
            (self.participants, *block.shape),
            self._block_sharding(block.ndim),
            [jax.device_put(block[None], self.device)],
        )

    def _local(self, array: jax.Array, /) -> np.ndarray:
        """This participant's block of a participant-blocked array."""
        for shard in array.addressable_shards:
            if shard.device == self.device:
                return np.array(np.asarray(shard.data)[0], copy=True)
        raise RuntimeError("The exchange device holds no participant block.")

    def gather(self, blocks: Sequence[np.ndarray], /) -> tuple[np.ndarray, ...]:
        """Every participant's block of each leaf on a leading participant axis.

        One collective program; every participant enters it with identically
        shaped and typed blocks in the same transaction order. Streams charge
        their exchange against ``maximum_round_bytes`` when prepared.
        """
        local = tuple(np.asarray(block, order="C") for block in blocks)
        if not local:
            raise ValueError("An exchange carries at least one block.")
        wire = tuple(
            block.view(np.uint8) if block.dtype == np.bool_ else block for block in local
        )
        gathered = _gather_blocks(self, tuple(self._blocked(block) for block in wire))
        return tuple(
            np.asarray(array.addressable_data(0)).astype(np.bool_)
            if block.dtype == np.bool_
            else np.array(np.asarray(array.addressable_data(0)), copy=True)
            for array, block in zip(gathered, local, strict=True)
        )

    def barrier(self) -> None:
        """Return only after every participant entered the same barrier."""
        self.gather((np.zeros((1,), dtype=np.int64),))


@eqx.filter_jit
def _gather_blocks(
    placement: AtomisticGenerationPlacement, blocks: tuple[jax.Array, ...]
) -> tuple[jax.Array, ...]:
    def body(*local: jax.Array) -> tuple[jax.Array, ...]:
        provider = JaxCollectiveProvider(_AXIS)
        return tuple(provider.all_gather(block, axis=0, tiled=True) for block in local)

    return jax.shard_map(
        body,
        mesh=placement.mesh,
        in_specs=tuple(PartitionSpec(_AXIS) for _ in blocks),
        out_specs=tuple(PartitionSpec() for _ in blocks),
        check_vma=False,
    )(*blocks)


def _replicated(
    placement: AtomisticGenerationPlacement, leaf: np.ndarray, /
) -> jax.Array:
    value = np.asarray(leaf, order="C")
    return jax.make_array_from_callback(
        value.shape,
        placement._replicated_sharding(),
        lambda index: np.asarray(value[index]),
    )


def _host_leaves(leaves: Sequence[np.ndarray], label: str, /) -> tuple[np.ndarray, ...]:
    result = tuple(np.asarray(leaf) for leaf in leaves)
    if any(leaf.dtype.kind not in "biuf" for leaf in result):
        raise TypeError(f"{label} leaves must be boolean, integer or floating arrays.")
    return result


def publish_placement_checkpoint(
    repository: ArtifactRepository,
    placement: AtomisticGenerationPlacement,
    checkpoint_id: str,
    /,
    *,
    analysis_id: str,
    revision_id: str,
    execution_id: str,
    replicated: Sequence[np.ndarray],
    participant: Sequence[np.ndarray],
    writer_id: str,
    attempt_id: str | None = None,
    parent_manifest: CheckpointManifest | None = None,
    encoding: ChunkEncoding = "identity",
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> CheckpointManifest:
    """Collective publication of one placed continuation and its global commit.

    Every participant calls this at the same transaction boundary with its
    identical ``replicated`` leaves and its own ``participant`` leaves. Each
    participant publishes its canonical shards under its group-local rank;
    after a group barrier the authority re-derives the complete manifest from
    the durable artifacts and writes the single global commit; after a second
    barrier every participant loads it. A participant lost before the commit
    leaves only uncommitted artifacts, which `load_distributed_checkpoint_commit`
    refuses.
    """
    if not isinstance(placement, AtomisticGenerationPlacement):
        raise TypeError("placement must be an AtomisticGenerationPlacement.")
    tree = {
        "replicated": {
            f"{index:06d}": _replicated(placement, leaf)
            for index, leaf in enumerate(_host_leaves(replicated, "Replicated"))
        },
        "participant": {
            f"{index:06d}": placement._blocked(np.asarray(leaf, order="C"))
            for index, leaf in enumerate(_host_leaves(participant, "Participant"))
        },
    }
    publish_process_checkpoint(
        repository,
        checkpoint_id,
        execution_id,
        tree,
        analysis_plan_id=analysis_id,
        numeric_revision_id=revision_id,
        writer_id=writer_id,
        writer_index=placement.position,
        attempt_id=attempt_id,
        encoding=encoding,
        limits=limits,
        parent_manifest=parent_manifest,
    )
    placement.barrier()
    if placement.is_authority:
        manifest = assemble_distributed_checkpoint_from_repository(
            repository,
            checkpoint_id,
            analysis_id,
            revision_id,
            execution_id,
            expected_process_count=placement.participants,
            parent_manifest=parent_manifest,
            limits=limits,
        )
        commit_distributed_checkpoint(
            repository,
            manifest,
            expected_process_count=placement.participants,
            writer_id=writer_id,
            attempt_id=attempt_id,
            parent_manifest=parent_manifest,
            limits=limits,
        )
    placement.barrier()
    return load_distributed_checkpoint_commit(
        repository, checkpoint_id, parent_manifest=parent_manifest, limits=limits
    )


def _restored(value: np.ndarray, template: np.ndarray, path: str, /) -> np.ndarray:
    if value.shape != template.shape or value.dtype != template.dtype:
        raise AtomisticGenerationCheckpointError(
            f"Checkpoint leaf {path} does not match the stream's continuation layout."
        )
    return value


def restore_placement_checkpoint(
    repository: ArtifactRepository,
    placement: AtomisticGenerationPlacement,
    checkpoint_id: str,
    /,
    *,
    analysis_id: str,
    revision_id: str,
    execution_id: str,
    replicated: Sequence[np.ndarray],
    participant: Sequence[np.ndarray],
    parent_manifest: CheckpointManifest | None = None,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], CheckpointManifest]:
    """Restore a globally committed placed continuation on the same placement.

    ``replicated`` and ``participant`` are host templates of the leaves. The
    committed analysis, revision and execution identities must equal the
    given ones; a changed stream, sampler revision or placement is refused.
    Each participant reads the replicated leaves and its own block only.
    """
    if not isinstance(placement, AtomisticGenerationPlacement):
        raise TypeError("placement must be an AtomisticGenerationPlacement.")
    manifest = load_distributed_checkpoint_commit(
        repository, checkpoint_id, parent_manifest=parent_manifest, limits=limits
    )
    if (
        manifest.analysis_plan_id != analysis_id
        or manifest.numeric_revision_id != revision_id
        or manifest.execution_plan_id != execution_id
    ):
        raise AtomisticGenerationCheckpointError(
            "The committed generation checkpoint belongs to another stream, sampler "
            "revision or placement."
        )
    shared = []
    for index, template in enumerate(_host_leaves(replicated, "Replicated")):
        path = f"['replicated']['{index:06d}']"
        array = restore_global_array_from_checkpoint(
            repository,
            manifest,
            path,
            placement._replicated_sharding(),
            limits=limits,
            parent_manifest=parent_manifest,
        )
        value = np.array(np.asarray(array.addressable_data(0)), copy=True)
        shared.append(_restored(value, template, path))
    owned = []
    for index, template in enumerate(_host_leaves(participant, "Participant")):
        path = f"['participant']['{index:06d}']"
        array = restore_global_array_from_checkpoint(
            repository,
            manifest,
            path,
            placement._block_sharding(template.ndim),
            limits=limits,
            parent_manifest=parent_manifest,
        )
        owned.append(_restored(placement._local(array), template, path))
    return tuple(shared), tuple(owned), manifest


__all__ = [
    "AtomisticGenerationCapacityError",
    "AtomisticGenerationCheckpointError",
    "AtomisticGenerationDivergenceError",
    "AtomisticGenerationPlacement",
    "publish_placement_checkpoint",
    "restore_placement_checkpoint",
]
