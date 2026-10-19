"""Triangulation boundaries accept conversion inputs and retain concrete owners."""

from typing import assert_type

import numpy as np

from phydrax.discretization import PeriodicCell
from phydrax.geometry import (
    ConstrainedDelaunayTriangulation,
    DelaunayTriangulation,
    PowerDiagram,
    RestrictedPowerDiagram,
    SimplexQualitySubcomplex,
    VoronoiDiagram,
)
from phydrax.geometry._periodic_power_preparation import PeriodicPowerPreparation
from phydrax.typing import ConvertibleToArray


def planar_conversions(
    points: ConvertibleToArray, weights: ConvertibleToArray, indices: ConvertibleToArray
) -> None:
    assert_type(DelaunayTriangulation(points), DelaunayTriangulation)
    assert_type(
        ConstrainedDelaunayTriangulation(points, indices),
        ConstrainedDelaunayTriangulation,
    )
    assert_type(
        SimplexQualitySubcomplex(points, indices, minimum_quality=0.0),
        SimplexQualitySubcomplex,
    )
    assert_type(
        VoronoiDiagram(points, box_lower=(0.0, 0.0), box_upper=(1.0, 1.0)), VoronoiDiagram
    )
    assert_type(
        PowerDiagram(points, weights, box_lower=(0.0, 0.0), box_upper=(1.0, 1.0)),
        PowerDiagram,
    )
    assert_type(
        DelaunayTriangulation([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]).points, np.ndarray
    )
    DelaunayTriangulation(object())  # ty: ignore[invalid-argument-type]


def periodic_power_conversions(
    points: ConvertibleToArray,
    weights: ConvertibleToArray,
    carrier: ConvertibleToArray,
    tetrahedra: ConvertibleToArray,
    regions: ConvertibleToArray,
    facets: ConvertibleToArray,
    cell: PeriodicCell,
) -> None:
    preparation = PeriodicPowerPreparation(
        points, weights, carrier, cell, maximum_images=1000
    )
    assert_type(preparation, PeriodicPowerPreparation)
    diagram = RestrictedPowerDiagram(
        points,
        weights,
        carrier,
        tetrahedra,
        regions,
        facets,
        periodic_preparation=preparation,
    )
    assert_type(diagram, RestrictedPowerDiagram)
    assert_type(diagram.periodic_preparation, PeriodicPowerPreparation | None)
