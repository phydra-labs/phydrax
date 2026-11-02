#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction
from math import pi, sin

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry.brep._constructors import (
    ProfileCurveUse,
    ProfileLine,
    ProfileLoop,
    ProfilePlane,
    _arc_circle,
    _line_arc_contact,
    _profile_edge,
    _segment_curve,
)
from phydrax.geometry.brep._intersection import ConicParameterEndpoint
from phydrax.geometry.brep._patches import CircleCurve
from phydrax.geometry.design._sketch import FixedPoint, Sketch, SketchProfileResult


def _profile(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    ccw: bool,
    *,
    reverse: bool = False,
) -> tuple[Sketch, SketchProfileResult]:
    points = np.asarray((start, end, center), dtype=np.float64)
    sketch = Sketch(
        points,
        lines=np.array([[1, 0]], dtype=np.int64),
        arc_endpoints=np.array([[1, 0] if reverse else [0, 1]], dtype=np.int64),
        arc_centers=np.array([2], dtype=np.int64),
        arc_counterclockwise=np.array([not ccw if reverse else ccw], dtype=np.bool_),
        constraints=tuple(FixedPoint(i, point) for i, point in enumerate(points)),
        feature_id="arc-segment",
    )
    return sketch, sketch.to_profile(sketch.solve(), name="arc-segment")


def _integrated_area(loop: ProfileLoop) -> float:
    # Green's theorem on the actual native line and circle carriers, not samples.
    twice_area = 0.0
    for i, segment in enumerate(loop.segments):
        start = np.asarray(loop.vertices[i], dtype=np.float64)
        end = np.asarray(loop.vertices[(i + 1) % len(loop.vertices)], dtype=np.float64)
        if isinstance(segment, ProfileLine):
            twice_area += start[0] * end[1] - start[1] * end[0]
        else:
            curve, first, last = _segment_curve(loop, i)
            assert isinstance(curve, CircleCurve)
            center = np.asarray(curve.center, dtype=np.float64)
            a = np.asarray(curve.first_axis, dtype=np.float64)
            b = np.asarray(curve.second_axis, dtype=np.float64)
            orientation = a[0] * b[1] - a[1] * b[0]
            twice_area += (
                center[0] * (end[1] - start[1])
                - center[1] * (end[0] - start[0])
                + float(curve.radius) ** 2 * orientation * (last - first)
            )
    return float(twice_area / 2)


@pytest.mark.parametrize(
    "start,end,ccw,sweep",
    [
        ((1.0, 0.0), (0.0, -1.0), True, 3 * pi / 2),
        ((0.0, 1.0), (-1.0, 0.0), False, 3 * pi / 2),
        ((-1.0, 0.0), (0.0, 1.0), True, 3 * pi / 2),
        ((0.0, -1.0), (1.0, 0.0), False, 3 * pi / 2),
        ((1.0, 0.0), (0.0, 1.0), True, pi / 2),
        ((1.0, 0.0), (0.0, -1.0), False, pi / 2),
        ((1.0, 0.0), (-1.0, 0.0), True, pi),
        ((1.0, 0.0), (-1.0, 0.0), False, pi),
    ],
)
def test_directed_circular_segment_area_and_lineage(
    start: tuple[float, float], end: tuple[float, float], ccw: bool, sweep: float
) -> None:
    sketch, result = _profile(start, end, (0.0, 0.0), ccw)
    reversed_sketch, reversed_result = _profile(start, end, (0.0, 0.0), ccw, reverse=True)
    assert _integrated_area(result.profile.outer) == pytest.approx(
        (sweep - sin(sweep)) / 2
    )
    assert _integrated_area(reversed_result.profile.outer) == pytest.approx(
        (sweep - sin(sweep)) / 2
    )
    assert result.profile.endpoint_gap_bounds == (((0.0, 0.0), (0.0, 0.0)),)
    assert set(result.profile.outer.vertices) == {start, end}
    assert sketch._problem_id() != reversed_sketch._problem_id()
    assert result.lineage_id != reversed_result.lineage_id
    assert result.source_id != reversed_result.source_id
    if sweep > pi:
        use = next(
            s for s in result.profile.outer.segments if isinstance(s, ProfileCurveUse)
        )
        assert isinstance(use.curve, CircleCurve)
        assert use.last - use.first == pytest.approx(sweep)
        assert use.curve.radius_exact**2 == Fraction(1)
        assert use.curve.center_exact == (Fraction(0), Fraction(0))


def test_irrational_radius_keeps_exact_original_circle() -> None:
    _, result = _profile((1.0, 1.0), (-1.0, 1.0), (0.0, 0.0), False)
    loop = result.profile.outer
    use = next(s for s in loop.segments if isinstance(s, ProfileCurveUse))
    assert isinstance(use.curve, CircleCurve)
    assert use.curve.radius_exact**2 == Fraction(2)
    assert use.curve.center_exact == (Fraction(0), Fraction(0))
    assert result.profile.endpoint_gap_bounds == (((0.0, 0.0), (0.0, 0.0)),)
    assert _integrated_area(loop) == pytest.approx(2 * (3 * pi / 2 + 1) / 2)
    assert set(loop.vertices) == {(1.0, 1.0), (-1.0, 1.0)}


def test_major_arc_with_prime_squared_radius_retains_native_roots() -> None:
    start = (3995190446.0, 1576450879.0)
    end = (1576450879.0, -3995190446.0)
    sketch, result = _profile(start, end, (0.0, 0.0), True)
    use = next(
        segment
        for segment in result.profile.outer.segments
        if isinstance(segment, ProfileCurveUse)
    )
    assert isinstance(use.curve, CircleCurve)
    assert use.last - use.first == pytest.approx(3 * pi / 2)
    assert use.curve.radius_exact**2 == Fraction(18446744073709551557)
    assert result.profile.endpoint_gap_bounds == (((0.0, 0.0), (0.0, 0.0)),)
    model = sketch.to_model(
        result, coordinate_contract=SpatialCoordinateContract.si()
    ).model
    geometry = model.geometry
    roots = next(
        pair
        for index, pair in enumerate(geometry.edge_endpoint_roots)
        if isinstance(geometry.curves[geometry.edge_curves[index]], CircleCurve)
    )
    assert all(isinstance(root, ConicParameterEndpoint) for root in roots)
    assert {
        root.point_exact for root in roots if isinstance(root, ConicParameterEndpoint)
    } == {
        (Fraction(3995190446), Fraction(1576450879), Fraction(0)),
        (Fraction(1576450879), Fraction(-3995190446), Fraction(0)),
    }
    assert any(
        all(isinstance(root, ConicParameterEndpoint) for root in pair)
        for pair in geometry.coedge_endpoint_roots
    )


def test_native_contact_uses_exact_circle_and_detects_near_crossing() -> None:
    _, result = _profile((1.0, 1.0), (-1.0, 1.0), (0.0, 0.0), False)
    loop = result.profile.outer
    arc_index = next(
        i for i, s in enumerate(loop.segments) if isinstance(s, ProfileCurveUse)
    )
    arc = _profile_edge(
        loop.vertices[arc_index],
        loop.vertices[(arc_index + 1) % 2],
        loop.segments[arc_index],
    )
    line = _profile_edge((1.0, 1.0), (-1.0, 1.0), ProfileLine())
    allowed = frozenset(((Fraction(1), Fraction(1)), (Fraction(-1), Fraction(1))))
    assert not _line_arc_contact(line, arc, allowed)
    crossing = _profile_edge((-2.0, 1.0 - 2.0**-40), (2.0, 1.0 - 2.0**-40), ProfileLine())
    assert _line_arc_contact(crossing, arc, frozenset())


def test_contact_circle_keeps_algebraic_center_and_radius() -> None:
    root = RadicalElement.sqrt(Fraction(2))
    curve = CircleCurve(
        (float(root), 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        float(root),
        center_exact=(root, Fraction(0)),
        radius_exact=root,
    )
    use = ProfileCurveUse(curve, 0.0, pi)
    edge = _profile_edge((float(root) + float(root), 0.0), (0.0, 0.0), use)
    center, squared = _arc_circle(edge)
    assert center == (root, Fraction(0))
    assert squared == Fraction(2)


def test_native_planar_face_binds_exact_conic_endpoints() -> None:
    sketch, result = _profile((1.0, 1.0), (-1.0, 1.0), (0.0, 0.0), False)
    model = sketch.to_model(
        result, coordinate_contract=SpatialCoordinateContract.si()
    ).model
    geometry = model.geometry
    assert all(
        all(isinstance(root, ConicParameterEndpoint) for root in pair)
        for pair in geometry.edge_endpoint_roots
    )
    roots = next(
        pair
        for index, pair in enumerate(geometry.edge_endpoint_roots)
        if isinstance(geometry.curves[geometry.edge_curves[index]], CircleCurve)
    )
    assert {
        root.point_exact for root in roots if isinstance(root, ConicParameterEndpoint)
    } == {
        (Fraction(1), Fraction(1), Fraction(0)),
        (Fraction(-1), Fraction(1), Fraction(0)),
    }
    assert any(
        all(isinstance(root, ConicParameterEndpoint) for root in pair)
        for pair in geometry.coedge_endpoint_roots
    )


def test_planar_face_rejects_unproved_lifted_conic_endpoints() -> None:
    sketch, _ = _profile((1.0, 1.0), (-1.0, 1.0), (0.0, 0.0), False)
    # This frame passes the numerical plane contract, but its binary
    # coefficients are not an exactly orthonormal conic chart.
    plane = ProfilePlane(x_axis=(0.6, 0.8, 0.0), y_axis=(-0.8, 0.6, 0.0))
    source = sketch.to_profile(sketch.solve(), name="rotated", plane=plane)
    with pytest.raises(ValueError):
        sketch.to_model(source, coordinate_contract=SpatialCoordinateContract.si())


def test_major_arc_does_not_round_an_antipodal_split() -> None:
    a = 2.0**-54
    start, end, center = (1.0, a), (a, 1.0), (a, a)
    _, result = _profile(start, end, center, False)
    loop = result.profile.outer
    use = next(s for s in loop.segments if isinstance(s, ProfileCurveUse))
    assert isinstance(use.curve, CircleCurve)
    radius = Fraction(1) - Fraction.from_float(a)
    assert use.curve.radius_exact**2 == radius**2
    assert use.curve.center_exact == (Fraction.from_float(a), Fraction.from_float(a))
    assert set(loop.vertices) == {start, end}
    assert result.profile.endpoint_gap_bounds == (((0.0, 0.0), (0.0, 0.0)),)
    assert _integrated_area(loop) == pytest.approx(
        float(radius**2) * (3 * pi / 2 + 1) / 2
    )


def test_opposite_directions_select_different_regions_and_sources() -> None:
    _, major = _profile((1.0, 0.0), (0.0, -1.0), (0.0, 0.0), True)
    _, minor = _profile((1.0, 0.0), (0.0, -1.0), (0.0, 0.0), False)
    assert _integrated_area(major.profile.outer) == pytest.approx((3 * pi / 2 + 1) / 2)
    assert _integrated_area(minor.profile.outer) == pytest.approx((pi / 2 - 1) / 2)
    assert major.lineage_id != minor.lineage_id
    assert major.source_id != minor.source_id


def test_major_arc_does_not_accept_nearly_incident_endpoint() -> None:
    points = np.array(((1.0, 0.0), (0.0, -1.0 - 2.0**-40), (0.0, 0.0)), dtype=np.float64)
    sketch = Sketch(
        points,
        lines=np.array([[1, 0]], dtype=np.int64),
        arc_endpoints=np.array([[0, 1]], dtype=np.int64),
        arc_centers=np.array([2], dtype=np.int64),
        arc_counterclockwise=np.array([True], dtype=np.bool_),
        constraints=tuple(FixedPoint(i, point) for i, point in enumerate(points)),
    )
    solution = sketch.solve()
    assert solution.root_evidence.root_converged
    with pytest.raises(ValueError, match="represented coordinates"):
        sketch.to_profile(solution, name="inexact")
