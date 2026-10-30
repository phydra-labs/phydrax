#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Repack saved owner shards and simplex histories onto a changed device group."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
from jax.typing import ArrayLike

from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._distribution_migration import (
    accept_ownerlocal_migration,
    OwnerLocalMigrationEvidence,
    OwnerLocalMigrationState,
    prepare_ownerlocal_migration,
)


if TYPE_CHECKING:
    from ._distribution import ShardedOwnerDirectoryData, ShardedOwnerDirectoryWitness


@final
class SimplexRestartRepack(StrictModule, NonTrainableState):
    """Accepted stable-ID forest and directory rows on the current owner group."""

    state: OwnerLocalMigrationState
    owner_directory: ShardedOwnerDirectoryWitness
    migration_evidence: OwnerLocalMigrationEvidence
    source_directory_id: str = eqx.field(static=True)
    target_directory_id: str = eqx.field(static=True)
    repack_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: OwnerLocalMigrationState,
        owner_directory: ShardedOwnerDirectoryWitness,
        migration_evidence: OwnerLocalMigrationEvidence,
        source_directory_id: str,
        target_directory_id: str,
        /,
    ) -> None:
        from ._distribution import ShardedOwnerDirectoryWitness

        if (
            not isinstance(state, OwnerLocalMigrationState)
            or not isinstance(owner_directory, ShardedOwnerDirectoryWitness)
            or not isinstance(migration_evidence, OwnerLocalMigrationEvidence)
            or not isinstance(source_directory_id, str)
            or not source_directory_id
            or not isinstance(target_directory_id, str)
            or not target_directory_id
        ):
            raise ValueError("Simplex restart repack identity or state is malformed.")
        self.state = state
        self.owner_directory = owner_directory
        self.migration_evidence = migration_evidence
        self.source_directory_id = source_directory_id
        self.target_directory_id = target_directory_id
        self.repack_id = canonical_fingerprint(
            {
                "kind": "simplex-restart-repack",
                "source_directory": source_directory_id,
                "target_directory": target_directory_id,
                "forest_identity": migration_evidence.target_identity,
                "migration": migration_evidence.migration_id,
            }
        )


def _rank(execution_group: ExecutionGroup, /) -> int:
    members = execution_group.spec.process_indices
    try:
        return members.index(jax.process_index())
    except ValueError as error:
        raise ValueError("The process is outside the restart owner group.") from error


def _stored_directory_id(value: ShardedOwnerDirectoryData, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "sharded-mesh-owner-directory",
            "source_manifest": value.source_manifest_id,
            "group": value.group_id,
            "part_count": value.part_count,
            "source": value.source_directory_id,
            "cells": value.cell_directory_id,
            "vertices": value.vertex_directory_id,
        }
    )


def repack_simplex_restart(
    stored_state: OwnerLocalMigrationState,
    stored_directory_shards: Sequence[ShardedOwnerDirectoryData],
    target_cell_owner_ranks: ArrayLike,
    target_vertex_owner_ranks: ArrayLike,
    target_cell_ids: ArrayLike,
    target_vertex_ids: ArrayLike,
    execution_group: ExecutionGroup,
    local_source_point_ids: ArrayLike,
    local_source_owner_ranks: ArrayLike,
    local_source_owned: ArrayLike,
    local_cell_ids: ArrayLike,
    local_cell_owner_ranks: ArrayLike,
    local_vertex_ids: ArrayLike,
    local_vertex_owner_ranks: ArrayLike,
    /,
    *,
    directory_entry_id: str,
    source_map_id: str | None = None,
    source_context_id: str | None = None,
    axis_name: str = "simplex_restart_repack",
) -> SimplexRestartRepack:
    """Stage forest/field restoration, rebuild owner hashes, then accept atomically."""
    from ._distribution import repack_sharded_owner_directory

    if not isinstance(stored_state, OwnerLocalMigrationState):
        raise TypeError("stored_state must be OwnerLocalMigrationState.")
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    staged = prepare_ownerlocal_migration(
        stored_state,
        target_cell_owner_ranks,
        target_vertex_owner_ranks,
        target_cell_ids,
        target_vertex_ids,
        execution_group,
        axis_name=f"{axis_name}_forest",
    )
    migrated_state = accept_ownerlocal_migration(
        staged,
        execution_group,
        axis_name=f"{axis_name}_accept",
    )
    witness = repack_sharded_owner_directory(
        stored_directory_shards,
        execution_group,
        local_source_point_ids,
        local_source_owner_ranks,
        local_source_owned,
        local_cell_ids,
        local_cell_owner_ranks,
        local_vertex_ids,
        local_vertex_owner_ranks,
        directory_entry_id=directory_entry_id,
        source_map_id=source_map_id,
        source_context_id=source_context_id,
        axis_name=f"{axis_name}_directory",
    )
    return SimplexRestartRepack(
        migrated_state,
        witness,
        staged.evidence,
        _stored_directory_id(stored_directory_shards[0]),
        witness.directory_id,
    )


__all__ = [
    "SimplexRestartRepack",
    "repack_simplex_restart",
]
