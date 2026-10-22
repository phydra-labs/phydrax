# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization._adaptive_simplex import (
    AdaptiveSimplexParts,
    AdaptiveSimplexStatus,
)
from phydrax.meshing._distribution import SimplexNeighborhoodWorkset
from phydrax.meshing._distribution_migration import (
    accept_ownerlocal_migration,
    OwnerLocalMigrationState,
    prepare_ownerlocal_migration,
)
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
)


_CONTRACT_KEYS = (
    "migration_59",
    "migration_60",
    "migration_61",
    "migration_62",
    "migration_63",
    "migration_64",
    "migration_65",
    "migration_66",
    "migration_67",
    "migration_68",
    "migration_69",
    "migration_70",
    "migration_71",
    "migration_73",
    "migration_74",
    "migration_75",
    "migration_76",
    "migration_77",
    "migration_78",
    "migration_80",
    "migration_82",
    "migration_84",
)


def _migration(scenario: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    capacity = 1 if scenario in ("overflow", "cell_boundary") else 2
    scientific = 2**54
    ids = np.full((2, capacity), np.iinfo(np.int64).max, dtype=np.int64)
    ids[:, 0] = [scientific + 11, scientific + 29]
    vertices = np.zeros((2, capacity, 3), dtype=np.int64)
    vertices[:, 0] = [
        [scientific + 1, scientific + 2, scientific + 3],
        [scientific + 2, scientific + 4, scientific + 3],
    ]
    coordinates = np.zeros((2, capacity, 3, 2), dtype=np.float64)
    coordinates[:, 0] = [
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ]
    valid = np.zeros((2, capacity), dtype=np.bool_)
    valid[:, 0] = True
    owner = np.broadcast_to(np.arange(2, dtype=np.int32)[:, None], (2, capacity)).copy()
    exterior = np.zeros((2, capacity, 3), dtype=np.bool_)
    exterior[:, 0] = [[False, True, True], [True, False, True]]
    ancestry = np.full((2, capacity, 3, 3), -1, dtype=np.int64)
    weights = np.zeros((2, capacity, 3, 3), dtype=np.float64)
    for p in range(2):
        for v in range(3):
            ancestry[p, 0, v, v] = vertices[p, 0, v]
            weights[p, 0, v, v] = 1.0
    work = SimplexNeighborhoodWorkset(
        *map(
            jnp.asarray,
            (
                ids,
                vertices,
                coordinates,
                owner,
                np.zeros((2, capacity), np.int32),
                exterior,
                valid,
                ids.copy(),
                ids[..., None].copy(),
                ancestry,
                weights,
                np.zeros(2, np.int32),
            ),
        )
    )
    fields = np.zeros((2, capacity), np.float64)
    fields[:, 0] = [3.0, 7.0]
    vertex_fields = coordinates.sum(axis=-1)
    if scenario == "shared_field_conflict":
        vertex_fields[1, 0, 0] += 1.0
    epochs = np.zeros((2, capacity), np.int64)
    epochs[:, 0] = [41, 42]
    vertex_owner = np.zeros((2, capacity, 3), np.int32)
    vertex_owner[1, 0] = [0, 1, 0]
    original = OwnerLocalMigrationState(
        work,
        jnp.asarray(fields),
        jnp.asarray(vertex_fields),
        jnp.asarray(epochs),
        jnp.asarray(vertex_owner),
    )
    parts = AdaptiveSimplexParts(jax.devices(), neighbor_pairs=((0, 1), (1, 0)))
    targets = np.zeros((2, capacity), np.int32)
    if scenario != "overflow":
        targets[0, 0] = 1
    prepared = prepare_ownerlocal_migration(
        parts,
        original,
        jnp.asarray(targets),
        halo_width=0 if scenario in ("overflow", "cell_boundary") else 1,
        vertex_capacity=3 if scenario in ("cell_boundary", "vertex_overflow") else 4,
    )
    target = accept_ownerlocal_migration(prepared)
    if scenario == "positive":
        _record_array(checks, "migration_59", prepared.evidence.accepted, [True, True])
        _record_array(
            checks,
            "migration_60",
            target.workset.cell_ids,
            [[scientific + 11, scientific + 29]] * 2,
        )
        _record_array(checks, "migration_61", target.workset.cell_owner, [[1, 0]] * 2)
        _record_array(checks, "migration_62", target.cell_fields, [[3.0, 7.0]] * 2)
        _record_array(checks, "migration_63", target.cell_epochs, [[41, 42]] * 2)
        _record_array(
            checks, "migration_64", target.workset.cell_vertices, [vertices[:, 0]] * 2
        )
        _record_array(
            checks,
            "migration_65",
            target.workset.cell_coordinates,
            [coordinates[:, 0]] * 2,
        )
        _record_array(
            checks, "migration_66", target.workset.cell_exterior, [exterior[:, 0]] * 2
        )
        _record_array(
            checks, "migration_67", target.workset.source_vertex_ids, [ancestry[:, 0]] * 2
        )
        _record_array(
            checks, "migration_68", target.workset.source_weights, [weights[:, 0]] * 2
        )
        _record_array(
            checks, "migration_69", target.vertex_fields, [vertex_fields[:, 0]] * 2
        )
        _record_array(
            checks,
            "migration_70",
            target.workset.source_cell_ids,
            [[[scientific + 11], [scientific + 29]]] * 2,
        )
        _record_array(
            checks, "migration_71", target.vertex_owners[0], [[1, 0, 0], [0, 0, 0]]
        )
        owned = np.asarray(target.workset.cell_owner) == np.arange(2)[:, None]
        checks.setdefault("migration_73", []).append(
            bool(np.sum(np.asarray(target.cell_fields)[owned]) == 10.0)
        )
        _record_array(checks, "migration_74", prepared.evidence.sent_cells, [1, 1])
        _record_array(checks, "migration_75", prepared.evidence.received_cells, [1, 1])
        _record_array(checks, "migration_76", prepared.evidence.vertices_after, [4, 4])
        checks.setdefault("migration_77", []).append(
            bool(prepared.evidence.route_metadata_bytes == 4)
        )
        checks.setdefault("migration_78", []).append(
            bool(prepared.target_parts.neighbor_pairs == ((0, 1), (1, 0)))
        )
    elif scenario == "cell_boundary":
        # Exactly one incoming owned cell fills each one-cell bucket; the
        # adjacent `overflow` fixture sends two cells into the same bucket.
        _record_array(checks, "migration_59", prepared.evidence.accepted, [True, True])
        _record_array(
            checks, "migration_60", target.workset.cell_ids,
            [[scientific + 29], [scientific + 11]],
        )
        _record_array(checks, "migration_61", target.workset.cell_owner, [[0], [1]])
        _record_array(checks, "migration_62", target.cell_fields, [[7.0], [3.0]])
        _record_array(checks, "migration_63", target.cell_epochs, [[42], [41]])
        _record_array(checks, "migration_74", prepared.evidence.sent_cells, [1, 1])
        _record_array(checks, "migration_75", prepared.evidence.received_cells, [1, 1])
        _record_array(checks, "migration_76", prepared.evidence.vertices_after, [3, 3])
    else:
        _record_array(checks, "migration_80", prepared.evidence.accepted, [False, False])
        flag = (
            AdaptiveSimplexStatus.CAPACITY_EXCEEDED
            if scenario in ("overflow", "vertex_overflow")
            else AdaptiveSimplexStatus.INVALID_GEOMETRY
        )
        checks.setdefault("migration_82", []).append(
            bool(np.all(np.asarray(prepared.evidence.status) & int(flag)))
        )
        for old, new in zip(
            jax.tree_util.tree_leaves(original),
            jax.tree_util.tree_leaves(target),
            strict=True,
        ):
            _record_array(checks, "migration_84", old, new)
    return checks


def migration_positive() -> dict[str, Any]:
    return _migration("positive")


def migration_overflow() -> dict[str, Any]:
    return _migration("overflow")


def migration_shared_field_conflict() -> dict[str, Any]:
    return _migration("shared_field_conflict")


def migration_cell_boundary() -> dict[str, Any]:
    return _migration("cell_boundary")


def migration_vertex_overflow() -> dict[str, Any]:
    return _migration("vertex_overflow")
