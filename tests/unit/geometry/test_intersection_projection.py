"""Exact elimination and source-branch projection contracts."""

import math
from fractions import Fraction as F

import numpy as np

from phydrax.geometry.brep._intersection import intersect_surface_regions
from phydrax.geometry.brep._intersection_curve import (
    orthogonal_cylinder_family,
    SurfaceRegion,
)
from phydrax.geometry.brep._intersection_projection import (
    _stationary_polynomials,
    project_intersection_curve,
)
from phydrax.geometry.brep._patches import CylinderPatch
from phydrax.geometry.brep._projection_contracts import (
    BRepProjectionPolicy,
    BRepProjectionStatus,
)


def _eval(poly: tuple[F, ...], t: F) -> F:
    return sum((coefficient * t**i for i, coefficient in enumerate(poly)), F(0))


def test_stationary_elimination_matches_unsquared_derivative() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 2.0)
    small = CylinderPatch((0, 0.125, -0.25), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -3.0), (math.tau, 3.0)))
    family = orthogonal_cylinder_family(SurfaceRegion(large, box), SurfaceRegion(small, box))
    assert family is not None
    for sign in (-1, 1):
        aa, bb, h, polynomial = _stationary_polynomials(
            family, sign, F(1, 4), F(2, 7), F(-3, 8)
        )
        for t in (F(-2), F(-1, 3), F(0), F(3, 2)):
            assert (
                _eval(polynomial, t)
                == _eval(aa, t) ** 2 * _eval(h, t) - _eval(bb, t) ** 2
            )
            # Independently differentiate the two rational terms before
            # squaring: A = D L' - 2 D' L, B = -s qx D(D H' - 2 D' H).
            d, dp = 1 + t * t, 2 * t
            a, b = family.small_normal_coefficients
            c, e = family.small_large_axis_coefficients
            n = family.delta_n * d + family.small_radius * (a * (1 - t * t) + 2 * b * t)
            np_ = family.delta_n * dp + family.small_radius * (-2 * a * t + 2 * b)
            z = family.delta_large_axis * d + family.small_radius * (
                c * (1 - t * t) + 2 * e * t
            )
            zp = family.delta_large_axis * dp + family.small_radius * (-2 * c * t + 2 * e)
            h0 = family.large_radius**2 * d * d - n * n
            hp = 2 * family.large_radius**2 * d * dp - 2 * n * np_
            w, wp = z - F(-3, 8) * d, zp - F(-3, 8) * dp
            ell = -2 * F(2, 7) * n * d + w * w
            ellp = -2 * F(2, 7) * (np_ * d + n * dp) + 2 * w * wp
            assert _eval(aa, t) == ellp * d - 2 * ell * dp
            assert _eval(bb, t) == -sign * F(1, 4) * d * (hp * d - 2 * h0 * dp)


def test_projection_on_certified_source_intersection() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    assert result.complete and len(result.curves) == 2
    curve = result.curves[0]
    parameter = 0.37
    query = np.asarray(curve.evaluate_coupled(parameter).point, dtype=np.float64)
    projection = project_intersection_curve(
        curve,
        (0.2, 0.6),
        query,
        BRepProjectionPolicy(),
        edge_range=(0.0, float(curve.num_charts)),
    )
    assert projection.status == int(BRepProjectionStatus.UNIQUE)
    assert 0.0 <= projection.lower < 1e-8
    assert projection.upper < 1e-8
    assert abs(projection.parameter - parameter) < 1e-7
    assert projection.root_id is not None


def test_stationary_root_at_exact_large_seam_node() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    curve = next(curve for curve in result.curves if float(curve.evaluate_coupled(0.0).point[0]) > 0)
    query = np.asarray((1.0, 0.0, 0.5))
    projection = project_intersection_curve(
        curve,
        (0.0, float(curve.num_charts)),
        query,
        BRepProjectionPolicy(),
        edge_range=(0.0, float(curve.num_charts)),
    )
    assert projection.status == int(BRepProjectionStatus.UNIQUE)
    assert projection.upper < 1e-8
    assert projection.lower <= 0.0 <= projection.upper
    np.testing.assert_allclose(projection.point, query, rtol=0.0, atol=1e-12)


def test_projection_of_singleton_endpoint() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    curve = result.curves[0]
    query = np.asarray(curve.evaluate_coupled(0.0).point, dtype=np.float64)
    projection = project_intersection_curve(
        curve,
        (0.0, 0.0),
        query,
        BRepProjectionPolicy(),
        edge_range=(0.0, float(curve.num_charts)),
    )
    assert projection.status == int(BRepProjectionStatus.SEAM)
    assert projection.parameter == 0.0
    assert projection.upper < 1e-8


def test_pi_chart_node_does_not_confuse_symbolic_closing_turn() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    curve = next(curve for curve in result.curves if float(curve.evaluate_coupled(0.0).point[0]) < 0)
    theta = np.asarray(curve.chart_start)[:, 2]
    node = int(np.argmin(abs(theta - math.pi)))
    assert node > 0 and theta[node] == math.pi
    query = np.asarray(curve.evaluate_coupled(float(node)).point, dtype=np.float64)
    projection = project_intersection_curve(
        curve,
        (0.0, float(curve.num_charts)),
        query,
        BRepProjectionPolicy(),
        edge_range=(0.0, float(curve.num_charts)),
    )
    assert projection.status == int(BRepProjectionStatus.UNIQUE)
    assert abs(projection.parameter - node) < 1e-7
    assert projection.upper < 1e-8


def test_projection_preserves_sheet_for_query_on_other_branch() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    positive = next(curve for curve in result.curves if float(curve.evaluate_coupled(0.0).point[0]) > 0)
    negative = next(curve for curve in result.curves if float(curve.evaluate_coupled(0.0).point[0]) < 0)
    query = np.asarray(negative.evaluate_coupled(0.37).point, dtype=np.float64)
    projection = project_intersection_curve(
        positive,
        (0.0, float(positive.num_charts)),
        query,
        BRepProjectionPolicy(),
        edge_range=(0.0, float(positive.num_charts)),
    )
    assert projection.status in (
        int(BRepProjectionStatus.UNIQUE),
        int(BRepProjectionStatus.AMBIGUOUS),
    )
    assert projection.point[0] > 0
    assert projection.lower > 1.7


def test_two_distinct_source_minima_are_ambiguous() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    curve = next(curve for curve in result.curves if float(curve.evaluate_coupled(0.0).point[0]) < 0)
    projection = project_intersection_curve(
        curve,
        (0.0, float(curve.num_charts)),
        np.zeros(3),
        BRepProjectionPolicy(),
        edge_range=(0.0, float(curve.num_charts)),
    )
    assert projection.status == int(BRepProjectionStatus.AMBIGUOUS)
    assert projection.lower <= 1.0 <= projection.upper
