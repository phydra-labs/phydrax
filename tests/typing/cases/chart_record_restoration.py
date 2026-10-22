"""Pin concrete chart restoration inputs and return types."""

from typing import assert_type

from phydrax.discretization import (
    PreparedSphereChartDeformation,
    PreparedSurfaceChartDeformation,
    SphereChartPieceBank,
    SphereGeometryReconstruction,
    SurfaceGeometryReconstruction,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import SphereMaterialCellAtlas


def chart_restoration_types(
    surface: PreparedSurfaceChartDeformation,
    sphere: PreparedSphereChartDeformation,
    reconstruction: SphereGeometryReconstruction,
    surface_reconstruction: SurfaceGeometryReconstruction,
    atlas: SphereMaterialCellAtlas,
    budget: CoordinateEnclosureBudget,
) -> None:
    assert_type(sphere.pieces, SphereChartPieceBank)
    assert_type(
        SphereChartPieceBank.from_pieces(tuple(sphere.pieces), budget=budget),
        SphereChartPieceBank,
    )
    assert_type(
        PreparedSurfaceChartDeformation.from_stored_fields(surface),
        PreparedSurfaceChartDeformation,
    )
    assert_type(
        PreparedSphereChartDeformation.from_stored_fields(sphere),
        PreparedSphereChartDeformation,
    )
    assert_type(
        SphereGeometryReconstruction.from_stored_fields(reconstruction),
        SphereGeometryReconstruction,
    )
    assert_type(
        PreparedSurfaceChartDeformation.from_stored_fields(surface, budget=budget),
        PreparedSurfaceChartDeformation,
    )
    assert_type(
        PreparedSphereChartDeformation.from_stored_fields(sphere, budget=budget),
        PreparedSphereChartDeformation,
    )
    assert_type(
        SphereGeometryReconstruction.from_stored_fields(reconstruction, budget=budget),
        SphereGeometryReconstruction,
    )
    assert_type(
        SurfaceGeometryReconstruction.from_stored_fields(surface_reconstruction),
        SurfaceGeometryReconstruction,
    )
    assert_type(
        SurfaceGeometryReconstruction.from_stored_fields(
            surface_reconstruction, budget=budget
        ),
        SurfaceGeometryReconstruction,
    )
    assert_type(
        SphereMaterialCellAtlas.from_stored_fields(atlas, budget=budget),
        SphereMaterialCellAtlas,
    )
    PreparedSurfaceChartDeformation.from_stored_fields(sphere)  # ty: ignore[invalid-argument-type]
    PreparedSphereChartDeformation.from_stored_fields(surface)  # ty: ignore[invalid-argument-type]
    SphereGeometryReconstruction.from_stored_fields(sphere)  # ty: ignore[invalid-argument-type]
    SurfaceGeometryReconstruction.from_stored_fields(sphere)  # ty: ignore[invalid-argument-type]
    SurfaceGeometryReconstruction.from_stored_fields(surface_reconstruction, budget=surface)  # ty: ignore[invalid-argument-type]
