"""Pin concrete chart restoration inputs and return types."""

from typing import assert_type

from phydrax.discretization import (
    PreparedSphereChartDeformation,
    PreparedSurfaceChartDeformation,
    SphereGeometryReconstruction,
)


def chart_restoration_types(
    surface: PreparedSurfaceChartDeformation,
    sphere: PreparedSphereChartDeformation,
    reconstruction: SphereGeometryReconstruction,
) -> None:
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
    PreparedSurfaceChartDeformation.from_stored_fields(sphere)  # ty: ignore[invalid-argument-type]
    PreparedSphereChartDeformation.from_stored_fields(surface)  # ty: ignore[invalid-argument-type]
    SphereGeometryReconstruction.from_stored_fields(sphere)  # ty: ignore[invalid-argument-type]
