import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.geometry._volume_material_map import prepare_eulerian_volume_overlap
from phydrax.meshing._association_transfer import (
    MaterialAssociationTransferError,
    prepare_eulerian_material_source_association_transfer,
    propagate_material_association_rows,
)
from phydrax.meshing._lineage import (
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
)
from tests._support.volume_material_map_fixtures import (
    make_curved_material_refinement,
    make_material_coverage_pair,
    make_material_source,
)


def test_material_association_transfer_keeps_region_occurrence_and_overlap_measure() -> (
    None
):
    mesh = CellMesh(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        (
            CellBlock(
                "cells",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),)),
                global_ids=np.asarray((17,)),
            ),
        ),
    )
    geometry = CellGeometrySpec.affine(mesh)
    domain, source_coverage, target_coverage = make_material_coverage_pair(
        mesh,
        geometry,
        mesh,
        geometry,
        region_id="steel",
        occurrence_path=("assembly-0", "occurrence-3"),
    )
    source_map = make_material_source(
        domain,
        entity_id="solid-17",
        region_id="steel",
        occurrence_path=("assembly-0", "occurrence-3"),
    )
    eulerian_overlap = prepare_eulerian_volume_overlap(
        mesh,
        geometry,
        mesh,
        geometry,
        source_map,
        ("solid-17",),
        ("solid-17",),
        ("steel",),
        ("steel",),
        source_coverage=source_coverage,
        target_coverage=target_coverage,
    )

    transfer = prepare_eulerian_material_source_association_transfer(eulerian_overlap)
    (source_side,) = transfer.source_sides_for_target(0)
    target_side = transfer.target_side(0)
    assert (source_side.cell_global_id, source_side.entity_id) == (17, "solid-17")
    assert source_side.region_id == target_side.region_id == "steel"
    assert (
        source_side.occurrence_path
        == target_side.occurrence_path
        == (
            "assembly-0",
            "occurrence-3",
        )
    )
    assert source_side.overlap_measure == target_side.physical_measure == 1.0 / 6.0
    assert transfer.unmatched_source_rows.size == transfer.unmatched_target_rows.size == 0
    np.testing.assert_allclose(transfer.source_coverage_defects, (0.0,), atol=1e-14)
    np.testing.assert_allclose(transfer.target_coverage_defects, (0.0,), atol=1e-14)


def test_material_rows_follow_cell_lineage_only_when_all_parents_agree() -> None:
    source_mesh, _, _, target_mesh, _, _, _ = make_curved_material_refinement()
    source_cell_set = source_mesh.entity_set(3).entity_set_id
    target_cell_set = target_mesh.entity_set(3).entity_set_id
    lineage = MeshLineage(
        source_mesh.topology_id,
        target_mesh.topology_id,
        (
            EntityLineage(
                3,
                source_cell_set,
                target_cell_set,
                np.asarray((7, 7), dtype=np.int64),
                np.asarray((10, 11), dtype=np.int64),
                np.asarray(
                    (
                        int(EntityLineageKind.REFINED_FROM),
                        int(EntityLineageKind.REFINED_FROM),
                    ),
                    dtype=np.int32,
                ),
            ),
        ),
    )
    path = ("assembly-0", "occurrence-3")
    rows = propagate_material_association_rows(
        source_mesh,
        target_mesh,
        lineage,
        ("solid-17",),
        ("steel",),
        (path,),
        ("steel", "steel"),
    )
    assert rows.target_cell_global_ids == (10, 11)
    assert rows.cell_entity_ids == ("solid-17", "solid-17")
    assert rows.region_ids == ("steel", "steel")
    assert rows.occurrence_paths == (path, path)
    assert rows.lineage_id == lineage.lineage_id

    conflict = MeshLineage(
        target_mesh.topology_id,
        source_mesh.topology_id,
        (
            EntityLineage(
                3,
                target_cell_set,
                source_cell_set,
                np.asarray((10, 11), dtype=np.int64),
                np.asarray((7, 7), dtype=np.int64),
                np.asarray(
                    (
                        int(EntityLineageKind.COARSENED_INTO),
                        int(EntityLineageKind.COARSENED_INTO),
                    ),
                    dtype=np.int32,
                ),
            ),
        ),
    )
    with pytest.raises(MaterialAssociationTransferError, match="agreeing") as error:
        propagate_material_association_rows(
            target_mesh,
            source_mesh,
            conflict,
            ("solid-17", "other-solid"),
            ("steel", "steel"),
            (path, path),
            ("steel",),
        )
    np.testing.assert_array_equal(error.value.target_conflicting_rows, (0,))
