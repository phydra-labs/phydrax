#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Stage and atomically accept owner-local simplex forest and field migration."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from math import prod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_array_hex_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._distribution_forest import SimplexForestRecords, validate_simplex_forest


type NamedArrays = tuple[tuple[str, Array], ...]


def _payload_bank(
    values: Mapping[str, ArrayLike], row_count: int, side: str, /
) -> NamedArrays:
    if not isinstance(values, Mapping):
        raise TypeError(f"{side}_payloads must be a name-to-array mapping.")
    entries: list[tuple[str, Array]] = []
    for name, value in sorted(values.items()):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{side} payload names must be non-empty strings.")
        array = np.asarray(value)
        if (
            array.ndim == 0
            or array.shape[0] != row_count
            or array.dtype.hasobject
            or array.dtype.kind not in "biufc"
            or not array.dtype.isnative
        ):
            raise ValueError(
                f"{side} payload {name!r} is not a native row-aligned array."
            )
        entries.append((name, jnp.asarray(array)))
    return tuple(entries)


def _bank_schema(bank: NamedArrays, /) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    return tuple(
        (name, np.dtype(value.dtype).str, tuple(value.shape[1:])) for name, value in bank
    )


def _bank_identity(bank: NamedArrays, /) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, canonical_array_hex_fingerprint(np.asarray(value))) for name, value in bank
    )


@final
class OwnerLocalMigrationState(StrictModule, NonTrainableState):
    """One rank's complete stable-ID forest and explicitly named field rows."""

    forest: SimplexForestRecords
    cell_payloads: NamedArrays
    vertex_payloads: NamedArrays
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        forest: SimplexForestRecords,
        /,
        *,
        cell_payloads: Mapping[str, ArrayLike] | None = None,
        vertex_payloads: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        if not isinstance(forest, SimplexForestRecords):
            raise TypeError("forest must be SimplexForestRecords.")
        validate_simplex_forest(forest)
        cell_bank = _payload_bank(
            {} if cell_payloads is None else cell_payloads,
            forest.cell_ids.shape[0],
            "cell",
        )
        vertex_bank = _payload_bank(
            {} if vertex_payloads is None else vertex_payloads,
            forest.vertex_ids.shape[0],
            "vertex",
        )
        forest_cell_names = {name for name, _ in forest.cell_fields}
        forest_vertex_names = {name for name, _ in forest.vertex_fields}
        if any(name in forest_cell_names for name, _ in cell_bank):
            raise ValueError("Cell payload names must not duplicate forest record names.")
        if any(name in forest_vertex_names for name, _ in vertex_bank):
            raise ValueError(
                "Vertex payload names must not duplicate forest record names."
            )
        self.forest = forest
        self.cell_payloads = cell_bank
        self.vertex_payloads = vertex_bank
        self.state_id = canonical_fingerprint(
            {
                "forest": forest.forest_id,
                "cell_payloads": _bank_identity(cell_bank),
                "vertex_payloads": _bank_identity(vertex_bank),
            }
        )


@final
class OwnerLocalMigrationEvidence(StrictModule, NonTrainableState):
    """Collectively validated stable-ID source/target migration evidence."""

    migration_id: str = eqx.field(static=True)
    source_identity: str = eqx.field(static=True)
    target_identity: str = eqx.field(static=True)
    group_id: str = eqx.field(static=True)
    local_rank: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    sent_cells: int = eqx.field(static=True)
    received_cells: int = eqx.field(static=True)
    sent_vertices: int = eqx.field(static=True)
    received_vertices: int = eqx.field(static=True)

    def __init__(
        self,
        migration_id: str,
        source_identity: str,
        target_identity: str,
        group_id: str,
        local_rank: int,
        part_count: int,
        sent_cells: int,
        received_cells: int,
        sent_vertices: int,
        received_vertices: int,
        /,
    ) -> None:
        if (
            any(
                not isinstance(value, str) or not value.strip()
                for value in (migration_id, source_identity, target_identity, group_id)
            )
            or isinstance(local_rank, bool)
            or not isinstance(local_rank, (int, np.integer))
            or isinstance(part_count, bool)
            or not isinstance(part_count, (int, np.integer))
            or not 0 <= int(local_rank) < int(part_count)
            or int(part_count) < 1
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
                for value in (
                    sent_cells,
                    received_cells,
                    sent_vertices,
                    received_vertices,
                )
            )
        ):
            raise ValueError("Owner-local simplex migration evidence is malformed.")
        self.migration_id = migration_id
        self.source_identity = source_identity
        self.target_identity = target_identity
        self.group_id = group_id
        self.local_rank = int(local_rank)
        self.part_count = int(part_count)
        self.sent_cells = int(sent_cells)
        self.received_cells = int(received_cells)
        self.sent_vertices = int(sent_vertices)
        self.received_vertices = int(received_vertices)


@final
class PreparedOwnerLocalMigration(StrictModule, NonTrainableState):
    """Unpublished candidate state; the source remains immutable until acceptance."""

    original: OwnerLocalMigrationState
    candidate: OwnerLocalMigrationState
    evidence: OwnerLocalMigrationEvidence

    def __init__(
        self,
        original: OwnerLocalMigrationState,
        candidate: OwnerLocalMigrationState,
        evidence: OwnerLocalMigrationEvidence,
        /,
    ) -> None:
        if (
            not isinstance(original, OwnerLocalMigrationState)
            or not isinstance(candidate, OwnerLocalMigrationState)
            or not isinstance(evidence, OwnerLocalMigrationEvidence)
        ):
            raise TypeError("Prepared migration requires canonical state and evidence.")
        self.original = original
        self.candidate = candidate
        self.evidence = evidence


def _rank(execution_group: ExecutionGroup, /) -> int:
    members = execution_group.spec.process_indices
    try:
        return members.index(jax.process_index())
    except ValueError as error:
        raise ValueError(
            "The process is outside the owner-local migration group."
        ) from error


def _schema_identity(
    state: OwnerLocalMigrationState,
    /,
) -> str:
    forest = state.forest
    return canonical_fingerprint(
        {
            "dimension": forest.dimension,
            "ambient_dimension": forest.ambient_dimension,
            "cell_fields": tuple(
                (name, np.dtype(value.dtype).str, tuple(value.shape[1:]))
                for name, value in forest.cell_fields
            ),
            "vertex_fields": tuple(
                (name, np.dtype(value.dtype).str, tuple(value.shape[1:]))
                for name, value in forest.vertex_fields
            ),
            "cell_payloads": _bank_schema(state.cell_payloads),
            "vertex_payloads": _bank_schema(state.vertex_payloads),
            "protected_edge_row_shape": tuple(forest.protected_edge_ids.shape[1:]),
        }
    )


def _collective_header(
    state: OwnerLocalMigrationState,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[int, int, str]:
    from ._distribution import _owner_group_allgather, _owner_group_fail_if_any

    forest = state.forest
    schema_id = _schema_identity(state)
    header = np.asarray(
        (
            forest.dimension,
            forest.ambient_dimension,
            forest.next_vertex_id,
            forest.next_cell_id,
        ),
        dtype=np.int64,
    )
    local_error = None
    try:
        if (
            not execution_group.is_member
            or execution_group.spec.group_id == ""
            or len(execution_group.spec.process_indices) != len(execution_group.devices)
            or not axis_name.strip()
        ):
            raise ValueError("Execution group or collective axis is malformed.")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "simplex migration input group"
    )
    gathered = _owner_group_allgather(header, execution_group, f"{axis_name}_header")
    schema_rows = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(schema_id), dtype=np.uint8),
        execution_group,
        f"{axis_name}_schema",
    )
    schema_text = tuple(
        np.asarray(row, dtype=np.uint8).tobytes().hex() for row in schema_rows
    )
    local_error = (
        None
        if len(set(schema_text)) == 1 and gathered.shape[1] == 4
        else "Simplex forest schemas differ across migration owners."
    )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_schema_agreement",
        "simplex migration schema",
    )
    return (
        int(np.max(gathered[:, 2])),
        int(np.max(gathered[:, 3])),
        canonical_fingerprint(
            {
                "schema": schema_id,
                "dimension": int(gathered[0, 0]),
                "ambient_dimension": int(gathered[0, 1]),
                "next_vertex_id": int(np.max(gathered[:, 2])),
                "next_cell_id": int(np.max(gathered[:, 3])),
            }
        ),
    )


def _pack_records(
    identifiers: np.ndarray,
    fields: NamedArrays,
    /,
) -> tuple[np.ndarray, tuple[tuple[str, np.dtype, tuple[int, ...], int], ...], int]:
    schema: list[tuple[str, np.dtype, tuple[int, ...], int]] = []
    width = 8
    for name, value in fields:
        host = np.asarray(value)
        tail = tuple(host.shape[1:])
        if (
            host.dtype.hasobject
            or host.dtype.kind not in "biufc"
            or not host.dtype.isnative
        ):
            raise ValueError(f"Migration field {name!r} is not a native numeric array.")
        byte_count = prod(tail) * host.dtype.itemsize if tail else host.dtype.itemsize
        schema.append((name, host.dtype, tail, byte_count))
        width += byte_count
    messages = np.zeros((identifiers.size, width), dtype=np.uint8)
    for row, identifier in enumerate(identifiers):
        messages[row, :8] = np.frombuffer(
            np.asarray(identifier, dtype="<i8").tobytes(), dtype=np.uint8
        )
        offset = 8
        for (_, value), (_, _, _, byte_count) in zip(fields, schema, strict=True):
            raw = np.ascontiguousarray(np.asarray(value)[row]).view(np.uint8).reshape(-1)
            messages[row, offset : offset + byte_count] = raw
            offset += byte_count
    return messages, tuple(schema), width


def _unpack_records(
    identifiers: np.ndarray,
    rows_by_id: Mapping[int, bytes],
    schema: tuple[tuple[str, np.dtype, tuple[int, ...], int], ...],
    /,
) -> NamedArrays:
    values = {
        name: np.empty((identifiers.size, *tail), dtype=dtype)
        for name, dtype, tail, _ in schema
    }
    for row, identifier in enumerate(identifiers):
        payload = rows_by_id[int(identifier)]
        offset = 0
        for name, dtype, tail, byte_count in schema:
            flat = np.frombuffer(
                payload,
                dtype=dtype,
                count=prod(tail) if tail else 1,
                offset=offset,
            )
            values[name][row] = flat.reshape(tail) if tail else flat[0]
            offset += byte_count
    return tuple((name, jnp.asarray(values[name])) for name, *_ in schema)


def _migrate_records(
    identifiers: ArrayLike,
    target_owner_ranks: ArrayLike,
    target_ids: ArrayLike,
    fields: NamedArrays,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
    *,
    side: str,
) -> tuple[NamedArrays, int, int]:
    from ._distribution import _owner_exchange_message_banks, _owner_group_fail_if_any

    rank = _rank(execution_group)
    part_count = len(execution_group.spec.process_indices)
    source_ids = np.asarray(identifiers, dtype=np.int64)
    target_owners = np.asarray(target_owner_ranks)
    expected_ids = np.asarray(target_ids)
    local_error = None
    try:
        messages, schema, width = _pack_records(source_ids, fields)
        if (
            source_ids.ndim != 1
            or np.any(source_ids < 0)
            or np.unique(source_ids).size != source_ids.size
            or target_owners.shape != source_ids.shape
            or target_owners.dtype.kind not in "iu"
            or np.any(target_owners < 0)
            or np.any(target_owners >= part_count)
            or expected_ids.ndim != 1
            or expected_ids.dtype.kind not in "iu"
            or np.any(expected_ids < 0)
            or (expected_ids.size > 1 and np.any(np.diff(expected_ids) <= 0))
        ):
            raise ValueError(
                f"Local {side} migration IDs or target owners are malformed."
            )
        expected_ids = expected_ids.astype(np.int64, copy=False)
        outgoing_rows: list[list[np.ndarray]] = [[] for _ in range(part_count)]
        rows_by_id: dict[int, bytes] = {}
        sent = 0
        for row, identifier_value in enumerate(source_ids):
            identifier = int(identifier_value)
            destination = int(target_owners[row])
            payload = messages[row].copy()
            if destination == rank:
                if identifier in rows_by_id:
                    raise ValueError(f"Local {side} migration duplicates a stable ID.")
                rows_by_id[identifier] = payload[8:].tobytes()
            else:
                outgoing_rows[destination].append(payload)
                sent += 1
        outgoing = tuple(
            np.stack(rows, axis=0) if rows else np.zeros((0, width), dtype=np.uint8)
            for rows in outgoing_rows
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        schema = ()
        width = 8
        sent = 0
        rows_by_id = {}
        outgoing = tuple(np.zeros((0, width), dtype=np.uint8) for _ in range(part_count))
        expected_ids = np.zeros((0,), dtype=np.int64)
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_{side}_encode",
        f"{side} migration packet preparation",
    )
    incoming = _owner_exchange_message_banks(
        outgoing, rank, execution_group, f"{axis_name}_{side}_exchange"
    )
    local_error = None
    received = 0
    try:
        for sender, bank in incoming.items():
            packet = np.asarray(bank, dtype=np.uint8)
            if packet.ndim != 2 or packet.shape[1] != width:
                raise ValueError(f"Incoming {side} migration packet width is invalid.")
            for row in packet:
                identifier = int(np.frombuffer(row[:8].tobytes(), dtype="<i8")[0])
                if identifier in rows_by_id:
                    raise ValueError(f"Received duplicate {side} stable ID {identifier}.")
                rows_by_id[identifier] = row[8:].tobytes()
                received += 1
        expected_set = {int(identifier) for identifier in expected_ids}
        if len(rows_by_id) != len(expected_set) or set(rows_by_id) != expected_set:
            raise ValueError(
                f"Received {side} IDs differ from the accepted local target rows."
            )
        unpacked = _unpack_records(expected_ids, rows_by_id, schema)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        unpacked = ()
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_{side}_validate",
        f"{side} migration receive validation",
    )
    return unpacked, sent, received


def _migrate_protected_edges(
    edge_ids: ArrayLike,
    cell_rows: ArrayLike,
    target_cell_owner_ranks: ArrayLike,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> np.ndarray:
    from ._distribution import _owner_exchange_message_banks, _owner_group_fail_if_any

    rank = _rank(execution_group)
    part_count = len(execution_group.spec.process_indices)
    edges = np.asarray(edge_ids, dtype=np.int64)
    cells = np.asarray(cell_rows, dtype=np.int64)
    owners = np.asarray(target_cell_owner_ranks)
    local_error = None
    try:
        if (
            edges.ndim != 2
            or edges.shape[1] != 2
            or np.any(edges < 0)
            or np.any(edges[:, 0] >= edges[:, 1])
            or len({tuple(row) for row in edges.tolist()}) != edges.shape[0]
            or cells.ndim != 2
            or owners.shape != (cells.shape[0],)
            or owners.dtype.kind not in "iu"
            or np.any(owners < 0)
            or np.any(owners >= part_count)
        ):
            raise ValueError(
                "Protected edge migration rows or target owners are malformed."
            )
        outgoing_rows: list[list[np.ndarray]] = [[] for _ in range(part_count)]
        local_edges: set[tuple[int, int]] = set()
        for edge in edges:
            incident = np.any(cells == edge[0], axis=1) & np.any(cells == edge[1], axis=1)
            if not np.any(incident):
                raise ValueError("A protected edge has no incident simplex record.")
            destination = int(np.min(owners[incident]))
            key = (int(edge[0]), int(edge[1]))
            if destination == rank:
                local_edges.add(key)
            else:
                outgoing_rows[destination].append(
                    np.asarray(edge, dtype="<i8").view(np.uint8)
                )
        outgoing = tuple(
            np.stack(rows, axis=0) if rows else np.zeros((0, 16), dtype=np.uint8)
            for rows in outgoing_rows
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        outgoing = tuple(np.zeros((0, 16), dtype=np.uint8) for _ in range(part_count))
        local_edges = set()
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_protected_edge_encode",
        "protected edge migration packet preparation",
    )
    incoming = _owner_exchange_message_banks(
        outgoing, rank, execution_group, f"{axis_name}_protected_edge_exchange"
    )
    local_error = None
    try:
        for bank in incoming.values():
            packets = np.asarray(bank, dtype=np.uint8)
            if packets.ndim != 2 or packets.shape[1] != 16:
                raise ValueError("Incoming protected-edge packet width is invalid.")
            for packet in packets:
                edge = np.frombuffer(packet.tobytes(), dtype="<i8")
                key = (int(edge[0]), int(edge[1]))
                if key[0] < 0 or key[0] >= key[1] or key in local_edges:
                    raise ValueError(
                        "Incoming protected-edge identity is invalid or duplicated."
                    )
                local_edges.add(key)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_protected_edge_validate",
        "protected edge migration receive validation",
    )
    return np.asarray(sorted(local_edges), dtype=np.int64).reshape((-1, 2))


def _record_digest(
    kind: bytes,
    identifier: int,
    fields: NamedArrays,
    row: int,
    /,
) -> bytes:
    digest = sha256(kind + identifier.to_bytes(8, "little", signed=False))
    for name, values in fields:
        host = np.ascontiguousarray(np.asarray(values)[row])
        digest.update(name.encode("utf-8"))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.view(np.uint8).tobytes())
    return digest.digest()


def _collective_identity(
    state: OwnerLocalMigrationState,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
    *,
    next_vertex_id: int,
    next_cell_id: int,
) -> str:
    from ._distribution import _owner_group_allgather

    forest = state.forest
    cell_fields = (*forest.cell_fields, *state.cell_payloads)
    vertex_fields = (*forest.vertex_fields, *state.vertex_payloads)
    local_xor = np.zeros((32,), dtype=np.uint8)
    local_sum = 0
    local_count = 0
    for kind, ids, fields in (
        (b"cell", np.asarray(forest.cell_ids, dtype=np.int64), cell_fields),
        (b"vertex", np.asarray(forest.vertex_ids, dtype=np.int64), vertex_fields),
    ):
        for row, identifier in enumerate(ids):
            record = _record_digest(kind, int(identifier), fields, row)
            local_xor ^= np.frombuffer(record, dtype=np.uint8)
            local_sum = (local_sum + int.from_bytes(record, "big")) % (1 << 256)
            local_count += 1
    for edge in np.asarray(forest.protected_edge_ids, dtype=np.int64):
        record = sha256(
            b"protected-edge" + np.asarray(edge, dtype="<i8").tobytes()
        ).digest()
        local_xor ^= np.frombuffer(record, dtype=np.uint8)
        local_sum = (local_sum + int.from_bytes(record, "big")) % (1 << 256)
        local_count += 1
    local = np.concatenate(
        (
            local_xor,
            np.frombuffer(local_sum.to_bytes(32, "big"), dtype=np.uint8),
            np.asarray((local_count,), dtype=np.int64).view(np.uint8),
        )
    )
    gathered = _owner_group_allgather(local, execution_group, f"{axis_name}_identity")
    xor_digest = np.bitwise_xor.reduce(gathered[:, :32], axis=0)
    sum_digest = sum(
        int.from_bytes(np.asarray(row[32:64], dtype=np.uint8).tobytes(), "big")
        for row in gathered
    ) % (1 << 256)
    count = int(np.sum(gathered[:, 64:].view(np.int64).reshape((-1,)), dtype=np.int64))
    return canonical_fingerprint(
        {
            "schema": _schema_identity(state),
            "xor": xor_digest,
            "sum": sum_digest.to_bytes(32, "big").hex(),
            "count": count,
            "next_vertex_id": next_vertex_id,
            "next_cell_id": next_cell_id,
        }
    )


def prepare_ownerlocal_migration(
    state: OwnerLocalMigrationState,
    target_cell_owner_ranks: ArrayLike,
    target_vertex_owner_ranks: ArrayLike,
    target_cell_ids: ArrayLike,
    target_vertex_ids: ArrayLike,
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str = "owner_local_simplex_migration",
) -> PreparedOwnerLocalMigration:
    """Stage cell, vertex, forest-history, and field rows by stable scientific ID.

    Every transfer is privately staged. No caller-visible state is changed until
    ``accept_ownerlocal_migration`` completes its collective identity agreement.
    """
    from ._distribution import _owner_group_fail_if_any

    if not isinstance(state, OwnerLocalMigrationState):
        raise TypeError("state must be OwnerLocalMigrationState.")
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    if not isinstance(axis_name, str) or not axis_name.strip():
        raise ValueError("axis_name must be a non-empty string.")
    axis = axis_name.strip()
    rank = _rank(execution_group)
    part_count = len(execution_group.spec.process_indices)
    next_vertex_id, next_cell_id, _ = _collective_header(state, execution_group, axis)
    source_identity = _collective_identity(
        state,
        execution_group,
        axis,
        next_vertex_id=next_vertex_id,
        next_cell_id=next_cell_id,
    )
    forest = state.forest
    cell_owner_values = np.asarray(target_cell_owner_ranks)
    vertex_owner_values = np.asarray(target_vertex_owner_ranks)
    local_error = None
    try:
        if (
            cell_owner_values.shape != forest.cell_ids.shape
            or vertex_owner_values.shape != forest.vertex_ids.shape
            or cell_owner_values.dtype.kind not in "iu"
            or vertex_owner_values.dtype.kind not in "iu"
            or np.any(cell_owner_values < 0)
            or np.any(cell_owner_values >= part_count)
            or np.any(vertex_owner_values < 0)
            or np.any(vertex_owner_values >= part_count)
        ):
            raise ValueError("Target owner maps must align with stable forest IDs.")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, f"{axis}_owner_map", "simplex migration owner map"
    )
    cell_fields = (*forest.cell_fields, *state.cell_payloads)
    vertex_fields = (*forest.vertex_fields, *state.vertex_payloads)
    migrated_cells, sent_cells, received_cells = _migrate_records(
        forest.cell_ids,
        cell_owner_values,
        target_cell_ids,
        cell_fields,
        execution_group,
        axis,
        side="cells",
    )
    migrated_vertices, sent_vertices, received_vertices = _migrate_records(
        forest.vertex_ids,
        vertex_owner_values,
        target_vertex_ids,
        vertex_fields,
        execution_group,
        axis,
        side="vertices",
    )
    cell_field_map = forest.cell_field_map()
    migrated_edges = _migrate_protected_edges(
        forest.protected_edge_ids,
        cell_field_map["cells"],
        cell_owner_values,
        execution_group,
        axis,
    )
    migrated_cell_map = dict(migrated_cells)
    migrated_vertex_map = dict(migrated_vertices)
    cell_names = tuple(name for name, _ in forest.cell_fields)
    vertex_names = tuple(name for name, _ in forest.vertex_fields)
    candidate_state: OwnerLocalMigrationState | None = None
    local_error = None
    try:
        candidate_forest = SimplexForestRecords(
            np.asarray(target_cell_ids, dtype=np.int64),
            np.full(np.asarray(target_cell_ids).shape, rank, dtype=np.int32),
            {name: migrated_cell_map[name] for name in cell_names},
            np.asarray(target_vertex_ids, dtype=np.int64),
            np.full(np.asarray(target_vertex_ids).shape, rank, dtype=np.int32),
            {name: migrated_vertex_map[name] for name in vertex_names},
            migrated_edges,
            forest.dimension,
            forest.ambient_dimension,
            next_vertex_id,
            next_cell_id,
        )
        candidate_state = OwnerLocalMigrationState(
            candidate_forest,
            cell_payloads={
                name: migrated_cell_map[name] for name, _ in state.cell_payloads
            },
            vertex_payloads={
                name: migrated_vertex_map[name] for name, _ in state.vertex_payloads
            },
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis}_candidate",
        "simplex migration candidate validation",
    )
    if candidate_state is None:
        raise ValueError(
            "Owner-local migration candidate validation failed collectively."
        )
    target_identity = _collective_identity(
        candidate_state,
        execution_group,
        axis,
        next_vertex_id=next_vertex_id,
        next_cell_id=next_cell_id,
    )
    local_error = (
        None
        if source_identity == target_identity
        else "Stable-ID forest or field content changed during owner migration."
    )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis}_content",
        "simplex migration content preservation",
    )
    migration_id = canonical_fingerprint(
        {
            "kind": "owner-local-simplex-migration",
            "source": source_identity,
            "target": target_identity,
            "group": execution_group.spec.group_id,
            "part_count": part_count,
            "axis": axis,
        }
    )
    evidence = OwnerLocalMigrationEvidence(
        migration_id,
        source_identity,
        target_identity,
        execution_group.spec.group_id,
        rank,
        part_count,
        sent_cells,
        received_cells,
        sent_vertices,
        received_vertices,
    )
    return PreparedOwnerLocalMigration(state, candidate_state, evidence)


def accept_ownerlocal_migration(
    prepared: PreparedOwnerLocalMigration,
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str = "owner_local_simplex_migration_accept",
) -> OwnerLocalMigrationState:
    """Collectively publish a staged migration; failure leaves source objects intact."""
    from ._distribution import _owner_group_allgather, _owner_group_fail_if_any

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    if not isinstance(axis_name, str) or not axis_name.strip():
        raise ValueError("axis_name must be a non-empty string.")
    local_error = None
    if not isinstance(prepared, PreparedOwnerLocalMigration):
        local_error = "prepared must be PreparedOwnerLocalMigration."
    elif (
        execution_group.spec.group_id != prepared.evidence.group_id
        or len(execution_group.spec.process_indices) != prepared.evidence.part_count
        or _rank(execution_group) != prepared.evidence.local_rank
    ):
        local_error = "Prepared migration does not belong to this execution group."
    axis = axis_name.strip()
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        axis,
        "simplex migration acceptance input",
    )
    if not isinstance(prepared, PreparedOwnerLocalMigration):
        raise TypeError("prepared must be PreparedOwnerLocalMigration.")
    digest = np.frombuffer(bytes.fromhex(prepared.evidence.migration_id), dtype=np.uint8)
    copies = _owner_group_allgather(digest, execution_group, axis)
    agreed = all(
        np.asarray(row, dtype=np.uint8).tobytes() == digest.tobytes() for row in copies
    )
    _owner_group_fail_if_any(
        None if agreed else "Owner-local migration identities differ across ranks.",
        execution_group,
        f"{axis}_identity",
        "simplex migration acceptance",
    )
    return prepared.candidate


__all__ = [
    "OwnerLocalMigrationEvidence",
    "OwnerLocalMigrationState",
    "PreparedOwnerLocalMigration",
    "accept_ownerlocal_migration",
    "prepare_ownerlocal_migration",
]
