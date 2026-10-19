"""White-box Gmsh wall-turn classification against polygon corner geometry."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from phydrax.meshing import (
    BoundaryLayerControl,
    BoundaryLayerCornerPolicy,
    BoundaryLayerRoute,
    LayerSchedule,
    MeshingEntityKind,
    MeshingScope,
)
from phydrax.meshing.providers._gmsh_layers import _wall_fan_points


pytestmark = pytest.mark.meshing_gmsh


@pytest.fixture
def gmsh_model() -> Iterator[Any]:
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("wall-corner-scale")
    try:
        yield gmsh
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("exponent", [-20, 0, 20], ids=["small", "unit", "large"])
@pytest.mark.parametrize(
    ("hole", "convex_indices"),
    [
        (((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)), (0, 1, 2, 3)),
        (
            ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.5, 0.5), (0.5, 0.0)),
            (0, 1, 2, 3, 5),
        ),
    ],
    ids=["convex-obstacle", "reentrant-obstacle"],
)
def test_wall_fans_classify_oriented_corners_independently_of_length_units(
    gmsh_model: Any,
    exponent: int,
    hole: tuple[tuple[float, float], ...],
    convex_indices: tuple[int, ...],
) -> None:
    scale = 2.0**exponent
    geo = gmsh_model.model.geo
    outer = [
        geo.addPoint(scale * x, scale * y, 0.0)
        for x, y in ((-2.0, -2.0), (3.0, -2.0), (3.0, 3.0), (-2.0, 3.0))
    ]
    wall = [geo.addPoint(scale * x, scale * y, 0.0) for x, y in hole]
    outer_curves = [
        geo.addLine(first, second) for first, second in zip(outer, outer[1:] + outer[:1])
    ]
    wall_curves = [
        geo.addLine(first, second) for first, second in zip(wall, wall[1:] + wall[:1])
    ]
    # Gmsh reverses inner wire senses when forming the surface boundary. Supply
    # that wire counterclockwise so its domain-boundary sense is clockwise.
    surface = geo.addPlaneSurface(
        [
            geo.addCurveLoop(outer_curves),
            geo.addCurveLoop([-curve for curve in reversed(wall_curves)]),
        ]
    )
    geo.synchronize()
    oriented = gmsh_model.model.getBoundary(
        [(2, surface)], combined=False, oriented=True, recursive=False
    )
    assert {
        int(tag)
        for dimension, tag in oriented
        if dimension == 1 and abs(tag) in wall_curves
    } == set(wall_curves)
    scope = MeshingScope(
        "polygon-wall",
        "fixed",
        MeshingEntityKind.GEOMETRY,
        1,
        "wall-curves",
        np.asarray(wall_curves, dtype=np.int64),
    )
    control = BoundaryLayerControl(
        scope,
        LayerSchedule.geometric(2, scale / 1024.0, growth_rate=1.25),
        route=BoundaryLayerRoute.PROVIDER,
        corner=BoundaryLayerCornerPolicy.FAN,
    )

    # Polygon orientation is clockwise around the excluded obstacle: all right
    # turns are convex wall corners; the L's sole left turn is reentrant.
    assert _wall_fan_points(gmsh_model, control, tuple(wall_curves)) == tuple(
        sorted(wall[index] for index in convex_indices)
    )
