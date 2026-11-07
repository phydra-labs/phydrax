#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep import brep_box, BRepModel, BRepTessellationPolicy


def _bottom_wall_control(
    model: BRepModel, thicknesses: tuple[float, ...]
) -> phx.meshing.BoundaryLayerControl:
    geometry = model.geometry
    points = np.asarray(geometry.vertex_points, dtype=np.float64)

    def face_rows(face: int) -> list[int]:
        return sorted(
            {
                int(vertex)
                for loop in geometry.face_loops[face]
                for coedge in loop
                for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
            }
        )

    walls = tuple(
        face
        for face in range(model.topology.num_faces)
        if np.all(points[face_rows(face), 2] == 0.0)
    )

    def scope(dimension: int, identifiers: tuple[int, ...]) -> phx.meshing.MeshingScope:
        return phx.meshing.MeshingScope(
            model.source_id,
            model.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            dimension,
            f"{model.source_revision}:brep:{dimension}",
            np.asarray(identifiers, dtype=np.int64),
        )

    return phx.meshing.BoundaryLayerControl(
        scope(2, walls),
        phx.meshing.LayerSchedule(thicknesses),
        route=phx.meshing.BoundaryLayerRoute.CAD_EXTRUSION,
        volume_scope=scope(3, (0,)),
    )


@pytest.mark.parametrize(
    ("thicknesses", "core_count"), (((0.25, 0.75), 0), ((0.05, 0.10), 1))
)
def test_native_box_extrusion_selects_the_source_solid_and_omits_a_filled_core(
    thicknesses: tuple[float, ...], core_count: int
) -> None:
    # Original CAD slab profiles: the unit-depth box is either filled by its
    # wall slab (no core region) or keeps one exact core above the cap.
    model = brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 1.0),
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.05, angular_deflection=0.2
        ),
    )

    extrusion = phx.meshing.prepare_boundary_layer_extrusion(
        model, _bottom_wall_control(model, thicknesses)
    )

    assert len(extrusion.layer_solid_ids) == 1
    assert len(extrusion.core_solid_ids) == core_count
    np.testing.assert_allclose(
        np.asarray(extrusion.slab_volumes), (6.0 * sum(thicknesses),), rtol=1e-12
    )
    assert extrusion.control.route is phx.meshing.BoundaryLayerRoute.EXACT_SWEEP
