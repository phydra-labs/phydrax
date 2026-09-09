import os
import shutil

import numpy as np
import pytest

import phydrax as phx


@pytest.mark.meshing_omega_h
def test_real_omega_h_refines_and_preserves_partition_evidence():
    requested = os.environ.get("PHYDRAX_OMEGA_H_EXECUTABLE", "phydrax_omega_h")
    executable = shutil.which(requested)
    if executable is None:
        pytest.skip("Real phydrax_omega_h native bridge is not installed")

    points = np.asarray(
        (
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.0, 0.5),
            (0.5, 0.5),
            (1.0, 0.5),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 1.0),
        )
    )
    cells = np.asarray(
        (
            (0, 1, 4),
            (0, 4, 3),
            (1, 2, 5),
            (1, 5, 4),
            (3, 4, 7),
            (3, 7, 6),
            (4, 5, 8),
            (4, 8, 7),
        )
    )
    mesh = phx.discretization.CellMesh.from_triangles(
        points,
        cells,
        vertex_global_ids=11 + 7 * np.arange(len(points), dtype=np.int64),
        cell_global_ids=17 + 13 * np.arange(len(cells), dtype=np.int64),
    )
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        mesh.entity_set(0).entity_set_id,
        mesh.vertex_global_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(np.eye(2) * 100.0, (len(points), 2, 2)),
        minimum_size=0.05,
        maximum_size=0.5,
        maximum_anisotropy=1.0,
    )

    result = phx.meshing.OmegaHProvider(executable).execute(
        mesh,
        metric,
        phx.SpatialCoordinateContract.si(),
    )

    target = result.target.mesh
    corners = np.asarray(target.coordinates)[np.asarray(target.blocks[0].vertices)]
    measures = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 2.0
    assert np.all(measures > 0.0)
    assert np.sum(measures) == pytest.approx(1.0, abs=1e-12)
    assert len(corners) > len(cells)
    assert result.target.audit.passed
    assert result.lineage_status == "unknown"
    assert len(result.partitions) == 1
    assert all(owner == 0 for owner in result.partitions[0].cell_owner_ranks)
    assert np.all(np.linalg.eigvalsh(np.asarray(result.metric.values)) > 0.0)
