from __future__ import annotations

from fractions import Fraction

import numpy as np

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from phydrax.meshing._dual_extraction import _RestrictedCoordinateMap
from phydrax.meshing._quad_generation import _compose_quad_source_geometry


def test_restricted_coordinate_map_replays_with_source_mesh_and_root_map() -> None:
    source_points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    source = CellMesh(
        source_points,
        (CellBlock("surface", "triangle", np.asarray(((0, 1, 2),), dtype=np.int32)),),
    )
    exact_source_map = ExactPlcCellGeometrySource(
        "restricted-quad-source",
        "restricted-quad-r1",
        np.asarray(source.vertex_global_ids, dtype=np.int64),
        tuple(
            tuple(Fraction(float(value)) for value in point) for point in source_points
        ),
        vertex_source_entity_ids=("source-v0", "source-v1", "source-v2"),
        vertex_occurrence_paths=((), (), ()),
    )
    affine = CellGeometrySpec.affine(source)
    source_geometry = CellGeometrySpec(
        dict(zip(affine.block_names, affine.elements, strict=True)),
        {
            name: np.asarray(route)
            for name, route in zip(affine.block_names, affine.geometry_dofs, strict=True)
        },
        source_points,
        source_map=exact_source_map,
        source_mesh=source,
    )
    target = CellMesh(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (1.0 / 3.0, 1.0 / 3.0, 0.0),
                (0.0, 0.5, 0.0),
            ),
            dtype=np.float64,
        ),
        (
            CellBlock(
                "quad", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    mapped_mesh, geometry = _compose_quad_source_geometry(
        source,
        target,
        np.asarray(
            ((0, -1, -1, -1), (0, 1, -1, -1), (0, 1, 2, -1), (0, 2, -1, -1)),
            dtype=np.int64,
        ),
        np.asarray((0, 1), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        source_geometry,
    )

    coordinate_map = geometry.source_map
    assert isinstance(coordinate_map, _RestrictedCoordinateMap)
    restored = _RestrictedCoordinateMap.from_stored_fields(
        coordinate_map,
        target_mesh=mapped_mesh,
    )

    assert restored.source_map_id == coordinate_map.source_map_id
    assert restored.source_mesh.mesh_id == source.mesh_id
    assert restored.source_geometry.source_map_id == exact_source_map.source_map_id
    assert np.array_equal(
        np.asarray(restored.rounded_vertices), np.asarray(mapped_mesh.coordinates)
    )
    assert restored.source_coordinates() == source_geometry.source_coordinates()
