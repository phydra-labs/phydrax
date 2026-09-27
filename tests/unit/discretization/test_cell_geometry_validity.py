#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available


D = phx.discretization
G = phx.geometry
Status = D.CellValidityStatus


def _polygon_status(points: Any, policy: Any = None) -> Any:
    points = np.asarray(points, dtype=np.float64)
    loop = np.arange(points.shape[0], dtype=np.int32)[None]
    mesh = D.CellMesh(points, (D.CellBlock("cells", "polygon", loop),))
    certificate = D.certify_cell_geometry_validity(mesh, policy=policy)
    return Status(int(certificate.status[0]))


def _pentagram() -> Any:
    angles = np.pi / 2.0 + 4.0 * np.pi * np.arange(5) / 5.0
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _embedded(points: Any, *, bend: Any = 0.0) -> Any:
    planar = np.column_stack((np.asarray(points, dtype=np.float64), np.zeros(5)))
    planar[3, 2] = bend
    angle = 0.4
    rotation = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(angle), -np.sin(angle)),
            (0.0, np.sin(angle), np.cos(angle)),
        )
    )
    return planar @ rotation.T


CONVEX = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 1.0))
# Notched square whose vertex-zero fan leaves the cell.
CONCAVE = ((0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (1.5, 1.0), (0.0, 3.0))


def test_cell_geometry_validity_scenario_1() -> None:
    for points, expected in (
        (CONVEX, Status.CERTIFIED_VALID),
        (CONCAVE, Status.CERTIFIED_VALID),
        (CONVEX[::-1], Status.INVALID),
        # Bow tie with a crossing between non-adjacent edges.
        (((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (1.0, -1.0), (0.0, 2.0)), Status.INVALID),
        (_pentagram(), Status.INVALID),
        # Repeated vertex coordinates behind distinct vertex ids.
        (((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 0.0), (0.0, 2.0)), Status.INVALID),
        # A vertex touching a non-adjacent edge.
        (((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 0.0), (0.0, 4.0)), Status.INVALID),
        # Adjacent edges doubling back along one line.
        (((0.0, 0.0), (4.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)), Status.INVALID),
    ):
        assert _polygon_status(points) == expected
    points = np.asarray(((0.0, 0.0), (1.0e-10, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    policy = D.CellValidityPolicy(relative_determinant_floor=1.0e-8)

    assert _polygon_status(points, policy) == Status.INVALID
    result = G.polygon_simplicity_2d(
        np.asarray(CONVEX),
        mode=G.PredicateMode.EXACT,
        maximum_candidate_pairs=1,
    )

    assert int(result.status) == G.PolygonSimplicityStatus.UNCERTAIN
    assert result.candidate_pair_count == 1
    assert result.candidate_capacity_exceeded
    assert _polygon_status(_embedded(CONCAVE)) == Status.CERTIFIED_VALID
    bent = _embedded(CONCAVE, bend=0.1)
    loose = D.CellValidityPolicy(relative_planarity_tolerance=0.5)

    assert _polygon_status(bent) == Status.INVALID
    assert _polygon_status(bent, loose) == Status.CERTIFIED_VALID


def _spike(offset: Any) -> Any:
    # Vertex 3 approaches edge 0 (on the line y = x) from above.
    tip = (0.3, 0.3 if offset == 0 else np.nextafter(0.3, 1.0))
    return np.asarray(((0.1, 0.1), (0.7, 0.7), (0.1, 0.9), tip, (0.0, 0.5)))


@pytest.mark.skipif(not meshcore_available(), reason="phydrax-meshcore unavailable")
def test_filtered_simplicity_is_uncertain_where_exact_decides() -> None:
    for offset, exact in (
        (0, G.PolygonSimplicityStatus.SELF_INTERSECTING),
        (1, G.PolygonSimplicityStatus.SIMPLE),
    ):
        points = _spike(offset)
        filtered = G.polygon_simplicity_2d(points, mode=G.PredicateMode.FILTERED)
        resolved = G.polygon_simplicity_2d(points, mode=G.PredicateMode.EXACT)

        assert int(filtered.status) == G.PolygonSimplicityStatus.UNCERTAIN
        assert int(filtered.uncertain_pairs) > 0
        assert int(resolved.status) == exact
        assert int(resolved.uncertain_pairs) == 0
        expected = Status.INVALID if offset == 0 else Status.CERTIFIED_VALID
        assert _polygon_status(points) == expected


def test_cell_geometry_validity_scenario_2() -> None:
    a = np.asarray(((0.0, 0.0),) * 5)
    b = np.asarray(((2.0, 2.0), (1.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, 0.0)))
    c = np.asarray(((0.0, 2.0), (1.0, 0.0), (1.0, 0.0), (2.0, 2.0), (2.0, 0.0)))
    d = np.asarray(((2.0, 0.0), (1.0, 1.0), (3.0, 0.0), (3.0, 3.0), (3.0, 0.0)))
    S = G.SegmentIntersectionStatus

    result = G.segment_intersections_2d(a, b, c, d, mode=G.PredicateMode.FILTERED)

    assert tuple(int(value) for value in result.status) == (
        S.PROPER_CROSSING,
        S.ENDPOINT_CONTACT,
        S.COLLINEAR_OVERLAP,
        S.DISJOINT,
        S.DISJOINT,
    )
    with pytest.raises(ValueError):
        G.segment_intersections_2d(a, b, c, d, mode=G.PredicateMode.FILTERED_DEVICE)
    points = np.asarray(((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (1.0, -1.0), (0.0, 2.0)))
    mesh = D.CellMesh(
        points, (D.CellBlock("cells", "polygon", np.arange(5, dtype=np.int32)[None]),)
    )

    with pytest.raises(phx.meshing.MeshingFailure, match="self_intersection"):
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
