from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

import phydrax.discretization._cell_geometry_predecessor as predecessor_module
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_predecessor import (
    prepare_cell_geometry_predecessor,
    reuse_cell_geometry_validity,
)
from phydrax.discretization.fem import lagrange_element


def test_unchanged_cell_validity_predecessor_matches_fresh_certification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = lagrange_element("hexahedron", 1)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinates = np.concatenate(
        (reference_nodes, reference_nodes + np.asarray((2.0, 0.0, 0.0))), axis=0
    )
    vertices = np.asarray(
        (np.arange(8, dtype=np.int32), np.arange(8, 16, dtype=np.int32))
    )
    cell_ids = np.asarray((101, 202), dtype=np.int64)
    source_mesh = CellMesh(
        coordinates,
        (CellBlock("cells", "hexahedron", vertices, global_ids=cell_ids),),
    )
    target_coordinates = np.array(coordinates, copy=True)
    target_coordinates[8:, 0] = 2.0 + 1.5 * (target_coordinates[8:, 0] - 2.0)
    target_mesh = CellMesh(
        target_coordinates,
        (CellBlock("cells", "hexahedron", vertices, global_ids=cell_ids),),
        vertex_global_ids=np.asarray(source_mesh.vertex_global_ids),
    )
    source_geometry = CellGeometrySpec.affine(source_mesh)
    target_geometry = CellGeometrySpec.affine(target_mesh)
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source_mesh)
    fresh = certify_cell_geometry_validity(target_geometry, mesh=target_mesh)
    predecessor = prepare_cell_geometry_predecessor(
        source_mesh,
        source_geometry,
        source_validity,
        target_mesh,
        target_geometry,
    )

    proof_helper = Mock(wraps=predecessor_module._certify_source_block)
    monkeypatch.setattr(predecessor_module, "_certify_source_block", proof_helper)
    reused = reuse_cell_geometry_validity(predecessor)

    assert predecessor.reused_cell_ids == (101,)
    assert proof_helper.call_count > 0
    expected_changed_routes = np.asarray(target_geometry.geometry_dofs[0])[1:2]
    for proof_call in proof_helper.call_args_list:
        np.testing.assert_array_equal(proof_call.args[3], expected_changed_routes)
    assert reused.certificate_id == fresh.certificate_id
    np.testing.assert_array_equal(reused.status, fresh.status)
    np.testing.assert_array_equal(reused.execution_status, fresh.execution_status)
    np.testing.assert_array_equal(reused.determinant_lower, fresh.determinant_lower)
    np.testing.assert_array_equal(reused.determinant_upper, fresh.determinant_upper)
    np.testing.assert_array_equal(reused.depth, fresh.depth)
    np.testing.assert_array_equal(reused.source_status, fresh.source_status)
    np.testing.assert_array_equal(
        reused.source_determinant_lower, fresh.source_determinant_lower
    )
    np.testing.assert_array_equal(
        reused.source_determinant_upper, fresh.source_determinant_upper
    )
    np.testing.assert_array_equal(reused.source_depth, fresh.source_depth)
    np.testing.assert_array_equal(
        reused.maximum_coefficient_rounding_error,
        fresh.maximum_coefficient_rounding_error,
    )
    np.testing.assert_array_equal(
        reused.maximum_element_coefficient_rounding_error,
        fresh.maximum_element_coefficient_rounding_error,
    )
