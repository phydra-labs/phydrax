# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import numpy as np
import pytest

from phydrax.discretization import CellMesh
from phydrax.meshing import EntityLineage, MeshLineage


def _mesh(offset: int) -> CellMesh:
    return CellMesh.from_simplices(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
        dimension=2,
        vertex_global_ids=np.arange(3, dtype=np.int64) + offset,
        cell_global_ids=np.asarray((offset,), dtype=np.int64),
    )


def _records(source: CellMesh, target: CellMesh) -> tuple[EntityLineage, ...]:
    records = []
    for dimension in range(3):
        before = source.entity_set(dimension)
        after = target.entity_set(dimension)
        records.append(
            EntityLineage(
                dimension,
                before.entity_set_id,
                after.entity_set_id,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int32),
                created_target_ids=np.setdiff1d(after.entity_ids, before.entity_ids),
                deleted_source_ids=np.setdiff1d(before.entity_ids, after.entity_ids),
            )
        )
    return tuple(records)


def test_complete_lineage_accounts_for_created_deleted_and_identity_entities() -> None:
    source, target = _mesh(0), _mesh(10)
    lineage = MeshLineage(
        source.topology_id, target.topology_id, _records(source, target)
    )
    previous_id = lineage.lineage_id
    accounting = lineage.validate_complete(source, target)
    assert lineage.lineage_id == previous_id
    assert accounting.lineage_id == previous_id
    assert tuple(item.dimension for item in accounting.dimensions) == (0, 1, 2)
    assert accounting.dimensions[0].created == 3
    assert accounting.dimensions[0].deleted == 3
    assert accounting.to_record()["complete"] is True
    assert accounting.to_record()["dimensions"] == {
        str(item.dimension): item.to_record() for item in accounting.dimensions
    }


def test_lineage_refuses_missing_dimensions_and_stale_topology() -> None:
    source, target = _mesh(0), _mesh(10)
    records = _records(source, target)
    incomplete = MeshLineage(source.topology_id, target.topology_id, records[:-1])
    with pytest.raises(ValueError, match="MeshLineage.dimensions"):
        incomplete.validate_complete(source, target)
    stale = MeshLineage("stale", target.topology_id, records)
    with pytest.raises(ValueError, match="MeshLineage.topology_ids"):
        stale.validate_complete(source, target)


def test_lineage_refuses_unaccounted_entities_and_wrong_entity_sets() -> None:
    source, target = _mesh(0), _mesh(10)
    records = _records(source, target)
    vertices = records[0]
    missing = EntityLineage(
        0,
        vertices.source_entity_set_id,
        vertices.target_entity_set_id,
        vertices.source_global_ids,
        vertices.target_global_ids,
        vertices.relation_kinds,
        created_target_ids=vertices.created_target_ids[:-1],
        deleted_source_ids=vertices.deleted_source_ids,
    )
    incomplete = MeshLineage(
        source.topology_id, target.topology_id, (missing, *records[1:])
    )
    with pytest.raises(ValueError, match="MeshLineage.entities"):
        incomplete.validate_complete(source, target)
    wrong_set = EntityLineage(
        0,
        "wrong-source-entity-set",
        vertices.target_entity_set_id,
        vertices.source_global_ids,
        vertices.target_global_ids,
        vertices.relation_kinds,
        created_target_ids=vertices.created_target_ids,
        deleted_source_ids=vertices.deleted_source_ids,
    )
    stale = MeshLineage(source.topology_id, target.topology_id, (wrong_set, *records[1:]))
    with pytest.raises(ValueError, match="MeshLineage.entity_set_ids"):
        stale.validate_complete(source, target)
