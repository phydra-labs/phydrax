#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Build declared meshing-domain fixtures and native protected-corner scenarios."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.geometry import (
    BSplineCurve,
    BSplineSurfacePatch,
    LineCurve,
    MeshingDomain,
    MeshingDomainBoundarySource,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    PlanePatch,
)


M = phx.meshing

# Provenance: the analytic/source-query regressions formerly in test_meshing_domain.py.
# These binary64 reference-comparison allowances are retained unchanged; certificate
# enclosure bounds are still asserted independently rather than widened by them.
SOURCE_ROUNDOFF_ATOL = 1.0e-12
TORUS_PROJECTION_RTOL = 1.0e-9
SPHERE_PROJECTION_RTOL = 1.0e-8
PRINCIPAL_CURVATURE_RTOL = 1.0e-10
ROOT_POSITION_ATOL = 1.0e-10
TRIM_RADIUS_ATOL = 1.0e-14


def meshing_scope(
    domain: MeshingDomain,
    dimension: int,
    entities: ArrayLike | Sequence[int],
    revision: str = "r1",
) -> M.MeshingScope:
    return M.MeshingScope(
        domain.source_id,
        revision,
        M.MeshingEntityKind.GEOMETRY,
        dimension,
        domain.entity_set_id(dimension),
        np.asarray(entities, dtype=np.int64),
    )


def square_patch(
    curves: tuple[int, int, int, int], shift: float = 0.0
) -> MeshingSurfacePatch:
    """Unit square patch of the plane z = 0 whose right side may be shifted."""

    bottom, right, top, left = curves
    return MeshingSurfacePatch(
        PlanePatch((shift, 0, 0), (1, 0, 0), (0, 1, 0)),
        (
            (
                PatchCurveUse(bottom, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
                PatchCurveUse(right, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
                PatchCurveUse(top, LineCurve((0.0, 1.0), (1.0, 0.0)), 1.0, 0.0),
                PatchCurveUse(left, LineCurve((0.0, 0.0), (0.0, 1.0)), 1.0, 0.0),
            ),
        ),
    )


def polygon_boundary_domain(polygons: tuple[np.ndarray, ...]) -> MeshingDomain:
    loops: list[tuple[PatchCurveUse, ...]] = []
    curves: list[MeshingDomainCurve] = []
    corner = 0
    for polygon in polygons:
        uses = []
        for index, first in enumerate(polygon):
            last = polygon[(index + 1) % polygon.shape[0]]
            uses.append(
                PatchCurveUse(
                    corner + index,
                    LineCurve(first, last - first),
                    0.0,
                    1.0,
                )
            )
            curves.append(
                MeshingDomainCurve(
                    corner + index,
                    corner + (index + 1) % polygon.shape[0],
                )
            )
        loops.append(tuple(uses))
        corner += polygon.shape[0]
    return MeshingDomain(
        (MeshingSurfacePatch(PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)), tuple(loops)),),
        tuple(curves),
        corner,
        source_id="authored-trim",
        source_revision="trim-r1",
    )


def c0_tensor_spline_trim_domain() -> MeshingDomain:
    controls = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            [[2.0, 1.0, 0.0], [2.0, 1.0, 1.0]],
        ],
    )
    surface = BSplineSurfacePatch(
        controls,
        np.ones((3, 2)),
        [0.0, 0.0, 1.0, 2.0, 2.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
        1,
    )
    corners = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))
    loop = tuple(
        PatchCurveUse(
            row,
            BSplineCurve.bezier(
                (head, 0.5 * (head + corners[(row + 1) % 4]), corners[(row + 1) % 4]),
                [1.0, 1.0, 1.0],
            ),
            0.0,
            1.0,
        )
        for row, head in enumerate(corners)
    )
    return MeshingDomain(
        (MeshingSurfacePatch(surface, (loop,)),),
        tuple(MeshingDomainCurve(row, (row + 1) % 4) for row in range(4)),
        4,
        source_id="authored-c0-tensor-trim",
        source_revision="original-knot-bank",
    )


@dataclass(frozen=True, slots=True)
class PlanarProtectionScenario:
    source: M.NativePlanarSource
    specification: M.SurfaceMeshingSpec
    feature: M.ProtectedFeature


def planar_protection_scenario(
    kind: M.FeatureKind,
    feature_revision: str = "r1",
) -> PlanarProtectionScenario:
    """Bind a dimension-zero square corner to one declared protected feature kind."""
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ((0, 1, 2, 3),),
        feature_id="protected-square",
    )
    source = M.NativePlanarSource(region, "r1")
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "square-region",
        np.asarray((0,)),
    )
    corner = M.MeshingScope(
        source.source_id,
        feature_revision,
        M.MeshingEntityKind.GEOMETRY,
        0,
        "square-corners",
        np.asarray((0,)),
    )
    feature = M.ProtectedFeature(corner, kind, maximum_deviation=0.0)
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            M.UniformSizeControl(scope, 0.3, strength=M.SizeControlStrength.SOFT),
        ),
        protected_features=(feature,),
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
    )
    return PlanarProtectionScenario(source, specification, feature)


def square_chart_boundary_source() -> MeshingDomainBoundarySource:
    """Build the two-triangle chart chain of a four-coedge unit square."""
    curves = tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0)))
    domain = MeshingDomain(
        (square_patch((0, 1, 2, 3)),), curves, 4, source_id="square", source_revision="r1"
    )
    charts = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    points = domain.evaluate(np.zeros((4,), dtype=np.int64), charts)
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    boundary = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    provenance = np.asarray(
        [[0, 0, 0, 1], [0, 1, 0, 1], [0, 2, 1, 0], [0, 3, 1, 0]], dtype=np.float64
    )
    restrictions = (
        np.zeros((charts.shape[0],), dtype=np.bool_),
        np.empty((0,), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
    )
    return MeshingDomainBoundarySource(
        domain,
        (0,),
        chart_triangulations=(
            (0, charts, points, cells, boundary, provenance, *restrictions),
        ),
    )
