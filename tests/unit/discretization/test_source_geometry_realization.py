from __future__ import annotations

import numpy as np

import phydrax as phx
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._source_geometry_realization import (
    prepare_source_geometry_realization,
)
from phydrax.discretization.fem import lagrange_element


def test_source_realization_binds_independent_maps_and_physical_measures() -> None:
    element = lagrange_element("hexahedron", 1)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    cell = np.arange(reference_nodes.shape[0], dtype=np.int32)[None, :]
    identifiers = np.asarray((701,), dtype=np.int64)
    source_mesh = CellMesh(
        reference_nodes,
        (CellBlock("volume", "hexahedron", cell, global_ids=identifiers),),
    )
    target_coordinates = reference_nodes * np.asarray((2.0, 1.0, 1.0))
    target_mesh = CellMesh(
        target_coordinates,
        (CellBlock("volume", "hexahedron", cell, global_ids=identifiers),),
        vertex_global_ids=np.asarray(source_mesh.vertex_global_ids),
    )
    source_geometry = CellGeometrySpec.affine(source_mesh)
    target_geometry = CellGeometrySpec.affine(target_mesh)
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source_mesh)
    target_validity = certify_cell_geometry_validity(target_geometry, mesh=target_mesh)
    source_embedding = phx.geometry.certify_global_embedding(
        source_mesh, source_geometry, source_validity
    )
    target_embedding = phx.geometry.certify_global_embedding(
        target_mesh, target_geometry, target_validity
    )

    realization = prepare_source_geometry_realization(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        source_embedding=source_embedding,
        target_embedding=target_embedding,
        maximum_displacement=1.0,
    )

    realization.require_current()
    assert realization.transition.evidence.kind == "source_realization"
    assert realization.maximum_displacement == 1.0
    np.testing.assert_allclose(realization.source_cell_measures, (1.0,), atol=1e-12)
    np.testing.assert_allclose(realization.target_cell_measures, (2.0,), atol=1e-12)
