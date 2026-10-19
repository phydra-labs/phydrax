"""Fresh-process replay of topology-neutral collective migration snapshots."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveLimits
from phydrax._fingerprint import tree_content_equal
from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
from phydrax.discretization._adaptive_simplex import refine_adaptive_simplex_parts
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    reconstruct_meshing_source,
    write_meshing_source_closure,
)
from phydrax.meshing._device_adaptation import (
    commit_partitioned_adaptive_simplex,
    validate_partitioned_mesh_evidence,
)
from phydrax.meshing._result import CollectiveMeshEvidence
from tests._support.distributed_meshing.device_adaptation import _public_preparation
from tests._support.distributed_meshing.native_distributed_records import (
    _require_instance,
    _require_present,
)


def _assert_device_free(value: Any) -> None:
    visited: set[int] = set()
    pending = [value]
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, (jax.Device, jax.sharding.Mesh)):
            raise AssertionError(
                "MigrationCheckpoint.payload must exclude device handles; got live execution object"
            )
        if isinstance(node, (jax.Array, np.ndarray)):
            continue
        if is_dataclass(node):
            pending.extend(getattr(node, field.name) for field in fields(node))
        elif isinstance(node, Mapping):
            pending.extend(node.values())
        elif isinstance(node, (tuple, list)):
            pending.extend(node)


def save_migration(path: str) -> dict[str, Any]:
    _, _, _, _, partitioned, first = _public_preparation(False)
    marks = np.asarray(first.state.mesh.cell_active).copy()
    marks[1:] = False
    second = refine_adaptive_simplex_parts(
        partitioned.layout, partitioned.parts, first.state, marks
    )
    target = (
        commit_partitioned_adaptive_simplex(partitioned, second.state)
        .local_results[0]
        .target
    )
    evidence = _require_instance(target.collective_evidence, CollectiveMeshEvidence)
    migration = _require_present(evidence.migration_record)
    _assert_device_free(migration)
    restored_migration = reconstruct_meshing_source(migration)
    if not tree_content_equal(restored_migration, migration):
        raise AssertionError(
            "MigrationCheckpoint.record must restore through validated standalone fields; got changed record"
        )
    altered = eqx.tree_at(
        lambda value: value.requested_target_owners,
        migration,
        migration.requested_target_owners + 1,
    )
    with pytest.raises(
        ValueError, match=r"CollectiveMigrationRecord\.migration_id must bind"
    ):
        reconstruct_meshing_source(altered)
    receipt = write_meshing_source_closure(
        Path(path), target, limits=ArrayArchiveLimits()
    )
    return {
        "content_id": receipt.content_id,
        "result_id": target.result_id,
        "evidence_id": evidence.evidence_id,
        "migration_id": migration.migration_id,
        "device_free": True,
        "standalone_validated": True,
        "tampered_record_refused": True,
    }


def replay_migration(path: str, content_id: str) -> dict[str, Any]:
    with NativeExecutionScope(NativeExecutionLimits(1, 0, 0, 1 << 20, 120.0)) as scope:
        with scope.batch(1) as ticket:
            ticket.consume(1)
            with ticket.native_target():
                target = read_meshing_source_closure(
                    Path(path),
                    expected_content_id=content_id,
                    limits=ArrayArchiveLimits(),
                )
            consumed_work = ticket.consumed_work_units
    restore_usage = _require_present(scope.evidence)
    evidence = _require_present(target.collective_evidence)
    migration = _require_present(evidence.migration_record)
    _assert_device_free(migration)
    validate_partitioned_mesh_evidence(_require_present(target.mesh.storage), evidence)
    return {
        "result_id": target.result_id,
        "evidence_id": evidence.evidence_id,
        "migration_id": migration.migration_id,
        "device_free": True,
        "replayed": True,
        "consumed_work": consumed_work,
        "restore_native_work": restore_usage.work_counters.construction_work
        - restore_usage.externally_charged_work,
        "restore_native_queries": restore_usage.native_primitive_queries,
        "restore_external_work": restore_usage.externally_charged_work,
    }


def refuse_topology_change(path: str, content_id: str) -> dict[str, Any]:
    target = read_meshing_source_closure(
        Path(path), expected_content_id=content_id, limits=ArrayArchiveLimits()
    )
    evidence = _require_present(target.collective_evidence)
    migration = _require_present(evidence.migration_record)
    try:
        migration.placement.rebind()
    except ValueError as failure:
        if "MigrationPlacement.topology" not in str(failure):
            raise
        return {"refused": True, "error": str(failure)}
    raise AssertionError(
        "MigrationCheckpoint.topology must refuse an unadmitted device-count change; got admission"
    )
