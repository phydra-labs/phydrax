#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import phydrax as phx


def _cell_block(cell_type, data):
    return SimpleNamespace(type=cell_type, data=np.asarray(data, dtype=np.int32))


def test_mesh_import_preserves_volume_and_facet_groups_with_loss_evidence(
    monkeypatch, tmp_path
):
    cells = (
        _cell_block("triangle", [[0, 1, 2], [0, 2, 3]]),
        _cell_block("line", [[0, 1], [1, 2], [2, 3], [3, 0]]),
        _cell_block("vertex", [[0]]),
    )
    source = SimpleNamespace(
        points=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        cells=cells,
        cell_sets={
            "myocardium": (
                np.asarray([0, 1], dtype=np.int32),
                np.asarray([], dtype=np.int32),
                np.asarray([], dtype=np.int32),
            ),
            "base": (
                np.asarray([], dtype=np.int32),
                np.asarray([0], dtype=np.int32),
                np.asarray([], dtype=np.int32),
            ),
        },
        point_sets={},
        point_data={"temperature": np.arange(4.0)},
        cell_data={"material": (np.asarray([7, 7]), np.arange(4), np.asarray([0]))},
        field_data={"myocardium": np.asarray([7, 2], dtype=np.int32)},
        info={"generator": "deterministic-fixture", "revision": 3},
        gmsh_periodic=[
            (
                1,
                (10, 20),
                np.eye(4),
                np.asarray([[0, 1]], dtype=np.int32),
            )
        ],
    )
    monkeypatch.setitem(sys.modules, "meshio", SimpleNamespace(read=lambda path: source))

    imported = phx.discretization.read_finite_element_mesh(tmp_path / "case.msh")

    assert imported.volume_cells("myocardium") == (0, 1)
    assert len(imported.boundary_facets("base")) == 1
    assert imported.report.volume_names == ("myocardium",)
    assert imported.report.boundary_names == ("base",)
    assert imported.report.source_path.endswith("case.msh")
    assert imported.report.source_format == "msh"
    assert dict(imported.report.source_metadata)["info"] == (
        '{"generator":"deterministic-fixture","revision":3}'
    )
    assert dict(imported.report.source_entity_counts) == {
        "facet_cells": 4,
        "other_cells": 1,
        "points": 4,
        "volume_cells": 2,
    }
    assert dict(imported.report.imported_entity_counts)["facet_cells"] == 1
    assert dict(imported.report.dropped_entity_counts)["facet_cells"] == 3
    assert dict(imported.report.dropped_entity_counts)["other_cells"] == 1
    assert imported.report.lossy
    assert "dropped_ungrouped_facet_cells:3" in imported.report.losses
    assert "dropped_unsupported_cells:1" in imported.report.losses
    assert (
        imported.report.report_id
        == phx.discretization.read_finite_element_mesh(
            tmp_path / "case.msh"
        ).report.report_id
    )
    assert "gmsh_periodic_not_imported:1" in imported.report.losses
    assert "gmsh_periodic" in dict(imported.report.source_metadata)

    with pytest.raises(ValueError, match="Unknown imported volume group"):
        imported.volume_cells("missing")
