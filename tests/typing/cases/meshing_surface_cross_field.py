"""Consumer-visible source-bound cross-field and quad-layout types."""

from typing import assert_type

from jax.typing import ArrayLike

import phydrax as phx


def surface_quad_field(
    source: phx.discretization.CellMesh,
    target: phx.meshing.CellMeshingTarget,
    scope: phx.meshing.MeshingScope,
    guidance: ArrayLike,
) -> None:
    field = phx.meshing.prepare_surface_cross_field(source, guide_angles=guidance)
    assert_type(field, phx.meshing.SurfaceCrossField)
    layout = phx.meshing.prepare_surface_quad_layout(source, field)
    assert_type(layout, phx.meshing.SurfaceQuadLayout)
    extraction = phx.meshing.extract_surface_quads(source, cross_field=field)
    assert_type(extraction, phx.meshing.DualExtraction)
    assert_type(extraction.field_event, phx.meshing.SurfaceQuadFieldEvent | None)
    request = phx.meshing.SurfaceMeshingSpec(
        target, scope, size_controls=(), cross_field=field
    )
    assert_type(request.cross_field, phx.meshing.SurfaceCrossField | None)
