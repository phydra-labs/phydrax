#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax import config as pxconfig
from tests._support.forced_devices import run_on_forced_devices


@pytest.mark.parametrize("operation", ("refine", "coarsen"))
def test_logical_organization_preserves_remote_material_and_rejects_incomplete_views(
    operation: str,
) -> None:
    report = run_on_forced_devices(
        "tests._support.distributed_meshing.logical_organization",
        f"{operation}_organization",
        (),
        devices=2,
        env={},
    )
    expected = (1,) if operation == "coarsen" else (4, 5)
    for local in report["locals"]:
        np.testing.assert_array_equal(local["global_label_ids"], expected)
        np.testing.assert_array_equal(local["global_zone_ids"], expected)
        assert local["material_id"] == "water"
        assert local["region_role"] == phx.meshing.RegionRole.FLUID.value
        np.testing.assert_array_equal(local["selected_ids"], local["local_ids"])
        assert local["coverage_id"] == local["storage_id"]
        assert local["label_id"] == report["locals"][0]["label_id"]
        assert local["union_scope_id"] == local["scope_id"]
        np.testing.assert_array_equal(local["union_ids"], local["local_ids"])
    assert len(report["locals"][0]["local_ids"]) == 0
    np.testing.assert_array_equal(report["locals"][1]["local_ids"], expected)
    assert report["material_change_rejected"], "Changed scientific material was accepted"
    assert report["dropped_label_rejected"], "A globally defined label was dropped"
    assert report["wrong_owner_rejected"], "Disjoint owner placement was accepted"
    assert report["global_overlap_rejected"], (
        "Globally overlapping zones escaped through locally empty views"
    )


def test_scope_lineage_rejects_ambiguous_coarsening_and_global_removal() -> None:
    from phydrax.meshing._lineage import inherit_scope

    meshing = phx.meshing
    points = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    coarse = phx.discretization.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    refined = phx.discretization.CellMesh.from_triangles(
        np.concatenate((points, np.asarray(((0.5, 0.5),), dtype=np.float64))),
        np.asarray(((0, 1, 4), (1, 2, 4), (0, 4, 3), (4, 2, 3)), dtype=np.int32),
    )
    original = meshing.MeshingScope(
        coarse.mesh_id,
        coarse.numeric_version,
        meshing.MeshingEntityKind.MESH,
        2,
        coarse.entity_set(2).entity_set_id,
        np.asarray((1,), dtype=np.int64),
    )
    split_record = meshing.EntityLineage(
        2,
        coarse.entity_set(2).entity_set_id,
        refined.entity_set(2).entity_set_id,
        np.asarray((0, 0, 1, 1), dtype=np.int64),
        np.arange(4, dtype=np.int64),
        np.full((4,), meshing.EntityLineageKind.REFINED_FROM, dtype=np.int32),
    )
    split = meshing.MeshLineage(coarse.topology_id, refined.topology_id, (split_record,))
    members = inherit_scope(original, split, refined, "region")
    np.testing.assert_array_equal(members.global_entity_ids, (2, 3))
    merge_record = meshing.EntityLineage(
        2,
        refined.entity_set(2).entity_set_id,
        coarse.entity_set(2).entity_set_id,
        np.arange(4, dtype=np.int64),
        np.asarray((0, 0, 1, 1), dtype=np.int64),
        np.full((4,), meshing.EntityLineageKind.COARSENED_INTO, dtype=np.int32),
    )
    merge = meshing.MeshLineage(refined.topology_id, coarse.topology_id, (merge_record,))
    restored = inherit_scope(members, merge, coarse, "region")
    np.testing.assert_array_equal(restored.global_entity_ids, original.global_entity_ids)
    partial = meshing.MeshingScope(
        refined.mesh_id,
        refined.numeric_version,
        meshing.MeshingEntityKind.MESH,
        2,
        refined.entity_set(2).entity_set_id,
        np.asarray((2,), dtype=np.int64),
    )
    with pytest.raises(ValueError, match=r"MeshLineage\.organization_membership"):
        inherit_scope(partial, merge, coarse, "region")
    removed_record = meshing.EntityLineage(
        2,
        coarse.entity_set(2).entity_set_id,
        refined.entity_set(2).entity_set_id,
        np.asarray((0, 0), dtype=np.int64),
        np.asarray((0, 1), dtype=np.int64),
        np.full((2,), meshing.EntityLineageKind.REFINED_FROM, dtype=np.int32),
        deleted_source_ids=np.asarray((1,), dtype=np.int64),
    )
    removed = meshing.MeshLineage(
        coarse.topology_id, refined.topology_id, (removed_record,)
    )
    with pytest.raises(ValueError, match=r"MeshLineage\.organization_scope"):
        inherit_scope(original, removed, refined, "region")


def test_scope_lineage_collapsed_membership_uses_deciding_source_precedence() -> None:
    from phydrax.meshing._lineage import inherit_scope

    meshing = phx.meshing
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    source = phx.discretization.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    target = phx.discretization.CellMesh.from_triangles(
        points[:3], np.asarray(((0, 1, 2),), dtype=np.int32)
    )
    source_set, target_set = source.entity_set(2), target.entity_set(2)
    dominant = meshing.MeshingScope(
        source.mesh_id,
        source.numeric_version,
        meshing.MeshingEntityKind.MESH,
        2,
        source_set.entity_set_id,
        np.asarray((0,), dtype=np.int64),
    )
    collapsed = meshing.MeshingScope(
        source.mesh_id,
        source.numeric_version,
        meshing.MeshingEntityKind.MESH,
        2,
        source_set.entity_set_id,
        np.asarray((1,), dtype=np.int64),
    )
    record = meshing.EntityLineage(
        2,
        source_set.entity_set_id,
        target_set.entity_set_id,
        np.asarray((1, 0), dtype=np.int64),
        np.asarray((0, 0), dtype=np.int64),
        np.asarray(
            (
                meshing.EntityLineageKind.COLLAPSED_INTO,
                meshing.EntityLineageKind.PRESERVED,
            ),
            dtype=np.int32,
        ),
    )
    lineage = meshing.MeshLineage(source.topology_id, target.topology_id, (record,))
    inherited = inherit_scope(dominant, lineage, target, "material")
    np.testing.assert_array_equal(inherited.global_entity_ids, (0,))
    with pytest.raises(ValueError, match=r"MeshLineage\.organization_scope"):
        inherit_scope(collapsed, lineage, target, "material")
    only_collapsed = meshing.EntityLineage(
        2,
        source_set.entity_set_id,
        target_set.entity_set_id,
        np.asarray((1, 0), dtype=np.int64),
        np.asarray((0, 0), dtype=np.int64),
        np.full((2,), meshing.EntityLineageKind.COLLAPSED_INTO, dtype=np.int32),
    )
    collapsed_lineage = meshing.MeshLineage(
        source.topology_id, target.topology_id, (only_collapsed,)
    )
    with pytest.raises(ValueError, match=r"MeshLineage\.organization_membership"):
        inherit_scope(dominant, collapsed_lineage, target, "material")
    all_members = dominant.union(collapsed)
    np.testing.assert_array_equal(
        inherit_scope(
            all_members, collapsed_lineage, target, "material"
        ).global_entity_ids,
        (0,),
    )


def test_sparse_semantic_inventory_queries_handle_an_empty_real_rank() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"

    def run(rank: int) -> dict[str, Any]:
        return run_on_forced_devices(
            "tests._support.distributed_meshing.logical_organization",
            "sparse_inventory",
            (),
            devices=1,
            env=pxconfig.environment(
                pxconfig.RuntimeConfig(
                    bootstrap=pxconfig.RuntimeBootstrap(
                        coordinator_address=address, num_processes=2, process_id=rank
                    )
                ),
                base={},
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = list(executor.map(run, range(2)))
    empty, populated = reports
    np.testing.assert_array_equal(empty["selected"], np.empty((0,), dtype=np.int32))
    np.testing.assert_array_equal(empty["valid"], np.empty((0,), dtype=np.bool_))
    np.testing.assert_array_equal(empty["membership"], np.empty((0,), dtype=np.bool_))
    np.testing.assert_array_equal(empty["key_values"], np.empty((0,), dtype=np.int32))
    np.testing.assert_array_equal(empty["key_valid"], np.empty((0,), dtype=np.bool_))
    assert empty["selected_dtype"] == empty["key_values_dtype"] == "int32"
    assert (
        empty["valid_dtype"]
        == empty["membership_dtype"]
        == empty["key_valid_dtype"]
        == "bool"
    )
    np.testing.assert_array_equal(populated["valid"], (True, False, True))
    np.testing.assert_array_equal(populated["membership"], populated["valid"])
    np.testing.assert_array_equal(
        np.asarray(populated["selected"])[np.asarray(populated["valid"])], (200, 400)
    )
    np.testing.assert_array_equal(populated["key_valid"], (True, False, True))
    np.testing.assert_array_equal(
        np.asarray(populated["key_values"])[np.asarray(populated["key_valid"])],
        (200, 400),
    )
