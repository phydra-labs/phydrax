"""Consumer contracts of native bounded curve/surface intersection.

Independent references are closed-form: circle/line roots, the plane/sphere
circle radius, the plane/cylinder ellipse, the Steinmetz curves of orthogonal
cylinders, and the torus/plane circles ``R +- sqrt(r^2 - h^2)``.
"""

import math
from collections.abc import Sequence
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep._intersection import (
    BranchRootEndpoint,
    CurveSurfaceIntersectionRoot,
    intersect_trim_curves,
    NativePeriodEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from phydrax.geometry.brep._intersection_curve import (
    AffinePCurve,
    decode_geometry,
    encode_geometry,
)


geometry = phx.geometry
PI = math.pi


def _sphere(center: Sequence[float], radius: float = 1.0) -> geometry.SurfaceRegion:
    patch = geometry.SpherePatch(center, [1, 0, 0], [0, 1, 0], [0, 0, 1], radius)
    return geometry.SurfaceRegion(patch, [[0.0, -PI / 2], [2 * PI, PI / 2]])


def _plane(
    origin: Sequence[float], first: Sequence[float], second: Sequence[float]
) -> geometry.SurfaceRegion:
    return geometry.SurfaceRegion(
        geometry.PlanePatch(origin, first, second), [[0.0, 0.0], [1.0, 1.0]]
    )


def _cylinder(
    first: Sequence[float],
    second: Sequence[float],
    axis: Sequence[float],
    radius: float,
    half_height: float,
) -> geometry.SurfaceRegion:
    patch = geometry.CylinderPatch([0, 0, 0], first, second, axis, radius)
    return geometry.SurfaceRegion(patch, [[0.0, -half_height], [2 * PI, half_height]])


def _samples(
    curve: geometry.IntersectionCurve, count: int = 97
) -> geometry.IntersectionCurvePoint:
    return curve.evaluate(jnp.linspace(0.0, curve.num_charts, count))


def _assert_on_both(
    first: geometry.SurfaceRegion,
    second: geometry.SurfaceRegion,
    curve: geometry.IntersectionCurve,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Every sample lies on both generating surfaces within the reported bound."""
    sample = _samples(curve)
    on_first = np.asarray(first.patch.evaluate(sample.first_parameters))
    on_second = np.asarray(second.patch.evaluate(sample.second_parameters))
    np.testing.assert_allclose(np.asarray(sample.point), on_first, atol=1e-12)
    assert np.max(np.linalg.norm(on_first - on_second, axis=1)) <= tolerance
    assert np.all(np.isfinite(np.asarray(sample.parameter_bound)))
    return np.asarray(sample.point)


def test_line_circle_transversal_points_are_certified() -> None:
    line = geometry.LineCurve([-2.0, 0.3], [1.0, 0.0])
    circle = geometry.CircleCurve([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], 1.0)
    result = geometry.intersect_curve_ranges(
        geometry.CurveRange(line, 0.0, 4.0), geometry.CurveRange(circle)
    )
    assert result.complete
    assert [point.kind for point in result.points] == ["transversal"] * 2
    xs = sorted(point.point[0] for point in result.points)
    np.testing.assert_allclose(xs, [-math.sqrt(0.91), math.sqrt(0.91)], atol=1e-12)
    for point in result.points:
        assert np.all(point.parameter_lower <= point.parameters)
        assert np.all(point.parameters <= point.parameter_upper)


def test_trim_endpoint_intersection_retains_exact_source_root() -> None:
    first = geometry.CurveTrimSegment(
        geometry.LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0
    )
    second = geometry.CurveTrimSegment(
        geometry.LineCurve((1.0, -1.0), (0.0, 1.0)), 0.0, 2.0
    )
    result = intersect_trim_curves(first, second)
    assert result.complete
    assert [point.kind for point in result.points] == ["transversal"]
    point = result.points[0]
    root = TrimIntersectionRoot(
        first,
        second,
        parameter_lower=point.parameter_lower,
        parameter_upper=point.parameter_upper,
    )
    position, bound, certified = root.evaluate()
    assert certified and bound < 1.0e-10
    np.testing.assert_allclose(position, (1.0, 0.0), atol=bound, rtol=0.0)
    assert root.first is first and root.second is second


def test_curve_owned_phase_roundtrip_retains_exact_source_authority() -> None:
    circle = geometry.CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    transported = AffinePCurve(
        circle,
        ((Fraction(2), Fraction(0)), (Fraction(0), Fraction(3))),
        (Fraction(1, 2), Fraction(-1, 4)),
    )
    phase = NativePeriodEndpoint(transported, turns=Fraction(1, 4))
    restored = decode_geometry(encode_geometry(phase))
    assert isinstance(restored, NativePeriodEndpoint)
    assert restored.patch is None and restored.axis is None
    assert restored.root_id == phase.root_id
    assert restored.rational == Fraction(0) and restored.turns == Fraction(1, 4)
    lower, upper = restored.parameter_enclosure()
    assert lower <= math.pi / 2 <= upper
    np.testing.assert_allclose(
        np.asarray(restored.carrier.evaluate(np.asarray(restored.parameter))),
        (0.5, 2.75),
        atol=1.0e-15,
        rtol=0.0,
    )
    with pytest.raises(ValueError):
        NativePeriodEndpoint(circle, None, 0)
    with pytest.raises(ValueError):
        NativePeriodEndpoint(geometry.LineCurve((0.0, 0.0), (1.0, 0.0)))


def test_spatial_root_lift_retains_source_endpoint_and_refuses_unrelated_pcurve() -> None:
    source = geometry.SurfaceRegion(
        geometry.PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, -1.0), (1.0, 1.0)),
    )
    opposite = geometry.SurfaceRegion(
        geometry.PlanePatch((0.3, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((-1.0, -1.0), (1.0, 1.0)),
    )
    curve = geometry.LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    pcurve = geometry.LineCurve((0.0, 0.0), (1.0, 0.0))
    result = geometry.intersect_curve_region(
        geometry.CurveRange(curve, 0.0, 1.0), opposite
    )
    assert result.complete
    point = result.points[0]
    root = CurveSurfaceIntersectionRoot(
        curve,
        opposite,
        parameter_lower=point.parameter_lower,
        parameter_upper=point.parameter_upper,
    )
    endpoint = TrimRootEndpoint(
        root, "first", source_pcurve=pcurve, source_surface=source
    )
    segment = geometry.CurveTrimSegment(
        pcurve, endpoint.parameter, 1.0, first_root=endpoint
    )
    lower, upper = segment.enclosure(0.0, 0.0)
    assert np.all(lower <= (0.3, 0.0)) and np.all(np.asarray((0.3, 0.0)) <= upper)
    assert endpoint.root is root
    restored = decode_geometry(encode_geometry(endpoint))
    assert (
        isinstance(restored, TrimRootEndpoint) and restored.root.root_id == root.root_id
    )
    with pytest.raises(ValueError):
        TrimRootEndpoint(
            root,
            "first",
            source_pcurve=geometry.LineCurve((0.0, 0.1), (1.0, 0.0)),
            source_surface=source,
        )
    with pytest.raises(ValueError):
        TrimRootEndpoint(root, "first", source_pcurve=pcurve)

    from phydrax.geometry.brep import _intersection as root_owner

    with root_owner.original_curve_surface_root_preparation((endpoint, restored)):
        np.testing.assert_array_equal(segment.enclosure(0.0, 0.0), (lower, upper))
        restored_segment = geometry.CurveTrimSegment(
            pcurve,
            restored.parameter,
            1.0,
            first_root=restored,
        )
        restored_lower, restored_upper = restored_segment.enclosure(0.0, 0.0)
        assert np.all(restored_lower <= (0.3, 0.0))
        assert np.all(np.asarray((0.3, 0.0)) <= restored_upper)
    np.testing.assert_array_equal(segment.enclosure(0.0, 0.0), (lower, upper))


def test_rooted_branch_enclosure_authorizes_only_source_endpoint_uncertainty() -> None:
    first = geometry.SurfaceRegion(
        geometry.PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, -1.0), (1.0, 1.0)),
    )
    second = geometry.SurfaceRegion(
        geometry.PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, -1.0), (1.0, 1.0)),
    )
    discovery = geometry.intersect_surface_regions(first, second)
    assert discovery.complete
    branch = discovery.curves[0]
    source = geometry.LineCurve((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    pcurve = geometry.LineCurve((0.0, 0.0), (0.0, 1.0))
    intersection = geometry.intersect_curve_region(
        geometry.CurveRange(source, -1.0, 1.0), second
    )
    assert intersection.complete
    point = intersection.points[0]
    root = CurveSurfaceIntersectionRoot(
        source,
        second,
        parameter_lower=point.parameter_lower,
        parameter_upper=point.parameter_upper,
    )
    endpoint = BranchRootEndpoint(
        root,
        branch,
        0,
        source_pcurve=pcurve,
        source_side="first",
        source_first=-1.0,
        source_last=1.0,
    )
    outside = float(np.nextafter(0.0, -np.inf))
    box = branch.bounding_box(outside, 0.1, endpoint_roots=(endpoint, None))
    positions = np.asarray(branch.evaluate(np.asarray((0.0, 0.1))).point)
    assert np.all(box[0] <= positions) and np.all(positions <= box[1])
    coupled = branch.p_curve("first")
    segment = geometry.CurveTrimSegment(coupled, outside, 0.1, first_root=endpoint)
    lower, upper = segment.enclosure(0.0, 1.0)
    parameters = np.asarray(
        branch.evaluate(np.asarray((0.0, 0.1), dtype=np.float64)).first_parameters
    )
    assert np.all(lower <= parameters) and np.all(parameters <= upper)
    with pytest.raises(ValueError):
        geometry.CurveTrimSegment(coupled, outside, 0.1)
    with pytest.raises(ValueError):
        branch.bounding_box(outside, 0.1)
    with pytest.raises(ValueError):
        branch.bounding_box(-0.01, 0.1, endpoint_roots=(endpoint, None))


def test_tangent_line_circle_reports_tangent_contact() -> None:
    line = geometry.LineCurve([-2.0, 1.0], [1.0, 0.0])
    circle = geometry.CircleCurve([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], 1.0)
    result = geometry.intersect_curve_ranges(
        geometry.CurveRange(line, 0.0, 4.0), geometry.CurveRange(circle)
    )
    assert [point.kind for point in result.points] == ["tangent"]
    np.testing.assert_allclose(result.points[0].point, [0.0, 1.0], atol=1e-7)


def test_line_sphere_curve_region_points() -> None:
    line = geometry.LineCurve([-2.0, 0.2, 0.1], [1.0, 0.0, 0.0])
    result = geometry.intersect_curve_region(
        geometry.CurveRange(line, 0.0, 4.0), _sphere([0.0, 0.0, 0.0])
    )
    assert result.complete
    assert [point.kind for point in result.points] == ["transversal"] * 2
    xs = sorted(point.point[0] for point in result.points)
    root = math.sqrt(1.0 - 0.05)
    np.testing.assert_allclose(xs, [-root, root], atol=1e-12)


def test_plane_sphere_is_one_closed_certified_circle() -> None:
    plane = _plane([-2, -2, 0.5], [4, 0, 0], [0, 4, 0])
    sphere = _sphere([0, 0, 0])
    result = geometry.intersect_surface_regions(plane, sphere)
    assert result.complete
    assert len(result.curves) == 1
    curve = result.curves[0]
    assert curve.closed and curve.fully_certified
    points = _assert_on_both(plane, sphere, curve)
    np.testing.assert_allclose(points[:, 2], 0.5, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(points[:, :2], axis=1), math.sqrt(0.75), atol=1e-12
    )


def test_plane_cylinder_is_one_closed_ellipse() -> None:
    plane = _plane([-3, -3, -1.5], [6, 0, 3], [0, 6, 0])
    cylinder = _cylinder([1, 0, 0], [0, 1, 0], [0, 0, 1], 1.0, 3.0)
    result = geometry.intersect_surface_regions(plane, cylinder)
    assert result.complete
    assert len(result.curves) == 1 and result.curves[0].closed
    points = _assert_on_both(plane, cylinder, result.curves[0])
    np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(points[:, 2], 0.5 * points[:, 0], atol=1e-12)


def test_orthogonal_unequal_cylinders_give_two_closed_loops() -> None:
    large = _cylinder([1, 0, 0], [0, 1, 0], [0, 0, 1], 1.0, 2.0)
    small = _cylinder([0, 1, 0], [0, 0, 1], [1, 0, 0], 0.5, 2.0)
    result = geometry.intersect_surface_regions(large, small)
    assert result.complete
    assert len(result.curves) == 2
    for curve in result.curves:
        assert curve.closed
        points = _assert_on_both(large, small, curve)
        np.testing.assert_allclose(np.hypot(points[:, 0], points[:, 1]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.hypot(points[:, 1], points[:, 2]), 0.5, atol=1e-12)
    signs = sorted(np.sign(float(_samples(curve).point[0, 0])) for curve in result.curves)
    assert signs == [-1.0, 1.0]


def test_sphere_sphere_external_tangency_is_tangent_point() -> None:
    result = geometry.intersect_surface_regions(
        _sphere([0.0, 0.0, 0.0]), _sphere([0.0, 2.0, 0.0])
    )
    assert result.complete
    assert result.curves == ()
    assert [point.kind for point in result.points] == ["tangent"]
    np.testing.assert_allclose(result.points[0].point, [0.0, 1.0, 0.0], atol=1e-12)
    assert result.points[0].gap_bound <= 1e-12


def test_orthogonal_equal_cylinders_meet_at_singular_junctions() -> None:
    first = _cylinder([1, 0, 0], [0, 1, 0], [0, 0, 1], 1.0, 2.0)
    second = _cylinder([0, 1, 0], [0, 0, 1], [1, 0, 0], 1.0, 2.0)
    result = geometry.intersect_surface_regions(first, second)
    singular = sorted(
        (point for point in result.points if point.kind == "singular"),
        key=lambda point: point.point[1],
    )
    assert len(singular) == 2
    np.testing.assert_allclose(singular[0].point, [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(singular[1].point, [0.0, 1.0, 0.0], atol=1e-12)
    assert len(result.curves) == 4
    for curve in result.curves:
        assert curve.fully_certified
        assert not result.complete
        assert any(region.reason == "singular" for region in result.unresolved)
        assert (curve.start_kind, curve.end_kind) == ("singular", "singular")
        points = _assert_on_both(first, second, curve)
        # Steinmetz branches lie on the planes z = x or z = -x.
        assert np.all(
            np.minimum(
                np.abs(points[:, 2] - points[:, 0]), np.abs(points[:, 2] + points[:, 0])
            )
            <= 1e-10
        )


def test_coincident_planes_report_coincident_region() -> None:
    first = _plane([0, 0, 0], [1, 0, 0], [0, 1, 0])
    second = _plane([0.5, 0.25, 0], [0, 1, 0], [-1, 0, 0])
    result = geometry.intersect_surface_regions(
        first, second, policy=geometry.ParametricIntersectionPolicy(maximum_boxes=4000)
    )
    assert result.curves == ()
    assert result.coincident
    region = result.coincident[0]
    assert region.distance_bound <= 1e-10
    assert np.any(~region.partial)
    first_points = np.asarray(
        first.patch.evaluate(
            jnp.asarray(0.5 * (region.cell_lower[:, :2] + region.cell_upper[:, :2]))
        )
    )
    np.testing.assert_allclose(first_points[:, 2], 0.0, atol=1e-14)


def test_torus_plane_has_two_circle_branches() -> None:
    torus = geometry.SurfaceRegion(
        geometry.TorusPatch([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 2.0, 0.5),
        [[0.0, 0.0], [2 * PI, 2 * PI]],
    )
    plane = _plane([-3, -3, 0.3], [6, 0, 0], [0, 6, 0])
    result = geometry.intersect_surface_regions(torus, plane)
    assert result.complete
    assert len(result.curves) == 2
    radii = []
    for curve in result.curves:
        assert curve.closed
        points = _assert_on_both(torus, plane, curve)
        radius = np.linalg.norm(points[:, :2], axis=1)
        np.testing.assert_allclose(radius, radius[0], atol=1e-12)
        radii.append(radius[0])
    np.testing.assert_allclose(sorted(radii), [2.0 - 0.4, 2.0 + 0.4], atol=1e-12)


def test_disjoint_surfaces_are_certified_empty() -> None:
    result = geometry.intersect_surface_regions(
        _sphere([0, 0, 0]), _plane([-2, -2, 3], [4, 0, 0], [0, 4, 0])
    )
    assert result.complete
    assert result.curves == () and result.points == () and result.coincident == ()


def test_exhausted_budget_is_unresolved_not_empty() -> None:
    torus = geometry.SurfaceRegion(
        geometry.TorusPatch([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 2.0, 0.5),
        [[0.0, 0.0], [2 * PI, 2 * PI]],
    )
    plane = _plane([-3, -3, 0.3], [6, 0, 0], [0, 6, 0])
    result = geometry.intersect_surface_regions(
        torus, plane, policy=geometry.ParametricIntersectionPolicy(maximum_boxes=8)
    )
    assert not result.complete
    assert result.work.budget_exhausted
    assert result.unresolved
    assert {region.reason for region in result.unresolved} == {"budget"}


def test_intersection_curve_payload_round_trip_is_lossless() -> None:
    plane = _plane([-2, -2, 0.5], [4, 0, 0], [0, 4, 0])
    sphere = _sphere([0, 0, 0])
    (curve,) = geometry.intersect_surface_regions(plane, sphere).curves
    restored = geometry.IntersectionCurve.from_payload(curve.payload())
    assert restored.branch_id == curve.branch_id
    tau = jnp.linspace(0.0, curve.num_charts, 13)
    np.testing.assert_array_equal(
        np.asarray(restored.evaluate(tau).point), np.asarray(curve.evaluate(tau).point)
    )


def test_intersection_p_curve_trims_classify_exactly() -> None:
    plane = _plane([-2, -2, 0.5], [4, 0, 0], [0, 4, 0])
    sphere = _sphere([0, 0, 0])
    (curve,) = geometry.intersect_surface_regions(plane, sphere).curves
    loop = geometry.CurveTrimLoop([curve.p_curve("first")], tolerance=1e-2)
    trim = geometry.TrimDomain(loop)
    radius = math.sqrt(0.75) / 4.0
    queries = np.asarray(
        [[0.5, 0.5], [0.5 + 0.99 * radius, 0.5], [0.5 + 1.01 * radius, 0.5], [0.9, 0.9]]
    )
    classified = trim.classify(queries)
    assert classified.resolved.all()
    np.testing.assert_array_equal(classified.inside, [True, True, False, False])
    band = np.asarray(trim.in_band(jnp.asarray(queries)))
    outside_band = ~band
    np.testing.assert_array_equal(
        np.asarray(trim.contains(jnp.asarray(queries)))[outside_band],
        classified.inside[outside_band],
    )


def test_polygon_trim_domain_keeps_affine_semantics() -> None:
    trim = geometry.TrimDomain(
        [[0, 0], [1, 0], [0, 1]], holes=[[[0.1, 0.1], [0.2, 0.1], [0.1, 0.2]]]
    )
    assert isinstance(trim.outer, geometry.PolygonTrimLoop)
    queries = np.asarray([[0.3, 0.3], [0.12, 0.12], [0.8, 0.8], [0.5, 0.0]])
    classified = trim.classify(queries)
    np.testing.assert_array_equal(classified.inside, [True, False, False, False])
    np.testing.assert_array_equal(classified.boundary, [False, False, False, True])
    np.testing.assert_array_equal(
        np.asarray(trim.contains(jnp.asarray(queries[:3]))), [True, False, False]
    )


def test_curve_trim_loop_refuses_open_chain() -> None:
    circle = geometry.CircleCurve([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], 1.0)
    half = geometry.CurveTrimSegment(circle, 0.0, PI)
    with pytest.raises(ValueError, match="not closed"):
        geometry.CurveTrimLoop([half], tolerance=1e-3)


def test_rational_curve_roots_bind_to_original_source() -> None:
    curve = geometry.BSplineCurve.bezier([[0, 0], [0.5, 1], [1, 0]], [1, 2, 1])
    line = geometry.LineCurve([0, 0.5], [1, 0])
    result = geometry.intersect_curve_ranges(
        geometry.CurveRange(curve), geometry.CurveRange(line, 0, 1)
    )
    assert result.complete
    parameters = sorted(point.parameters[0] for point in result.points)
    expected = [(1 - math.sqrt(1 / 3)) / 2, (1 + math.sqrt(1 / 3)) / 2]
    np.testing.assert_allclose(parameters, expected, atol=1e-12)
    for point in result.points:
        original = np.asarray(curve.evaluate(jnp.asarray(point.parameters[0])))
        np.testing.assert_allclose(original, point.point, atol=point.gap_bound + 1e-14)


def test_rational_surface_intersection_binds_to_original_patch() -> None:
    patch = geometry.BSplineSurfacePatch(
        [[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]],
        [[1, 2], [3, 4]],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        1,
        1,
    )
    line = geometry.LineCurve([0.35, 0.45, -1], [0, 0, 1])
    result = geometry.intersect_curve_region(
        geometry.CurveRange(line, 0, 2), geometry.SurfaceRegion(patch, [[0, 0], [1, 1]])
    )
    assert result.complete
    assert len(result.points) == 1
    np.testing.assert_allclose(result.points[0].point, [0.35, 0.45, 0], atol=1e-12)
    np.testing.assert_allclose(
        patch.evaluate(jnp.asarray(result.points[0].parameters[1:])),
        result.points[0].point,
        atol=1e-12,
    )


def test_rational_spline_branch_differentiates_through_live_source_leaves() -> None:
    import equinox as eqx
    import jax

    # Biquadratic rational sheet with an interior u knot: the branch crosses
    # two Bernstein pieces, each pinned to its canonical knot span.
    heights = [[0.0, 0.1, 0.0], [0.2, 0.3, 0.1], [0.1, 0.4, 0.2], [0.0, 0.2, 0.1]]
    xs, ys = (0.0, 0.3, 0.7, 1.0), (0.0, 0.5, 1.0)
    points = [[[x, y, heights[i][j]] for j, y in enumerate(ys)] for i, x in enumerate(xs)]
    weights = [[1.0, 1.3, 1.0], [0.9, 1.2, 1.1], [1.1, 0.8, 1.0], [1.0, 1.1, 0.9]]
    patch = geometry.BSplineSurfacePatch(
        points,
        weights,
        [0, 0, 0, 0.5, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        2,
        2,
    )
    sheet = geometry.SurfaceRegion(patch, [[0, 0], [1, 1]])
    plane = geometry.SurfaceRegion(
        geometry.PlanePatch([-0.5, 0.4, -2.0], [2, 0, 0], [0, 0, 4]), [[0, 0], [1, 1]]
    )
    (curve,) = geometry.intersect_surface_regions(sheet, plane).curves
    assert curve.fully_certified
    parameters = jnp.linspace(0.0, float(curve.num_charts), 9)

    # Eager evaluation is certified against the host Bernstein enclosures and
    # lies on the original rational source and the plane.
    eager = curve.evaluate(parameters)
    np.testing.assert_allclose(
        patch.evaluate(eager.first_parameters), eager.point, rtol=0.0, atol=1e-11
    )
    np.testing.assert_allclose(np.asarray(eager.point)[:, 1], 0.4, rtol=0.0, atol=1e-11)
    assert np.all(np.isfinite(np.asarray(eager.parameter_bound)))
    jitted = eqx.filter_jit(lambda value: value.evaluate(parameters).point)(curve)
    np.testing.assert_allclose(jitted, eager.point, rtol=0.0, atol=1e-13)

    probe = jnp.linspace(1.0, -0.5, 27).reshape((9, 3))

    def functional(source: geometry.BSplineSurfacePatch) -> jax.Array:
        moved = eqx.tree_at(lambda value: value.first.patch, curve, source)
        return jnp.sum(probe * moved.evaluate(parameters).point)

    knot_direction = np.zeros(7)
    knot_direction[3] = 1.0
    direction = eqx.tree_at(
        lambda value: (value.control_points, value.weights, value.u_knots, value.v_knots),
        jax.tree_util.tree_map(jnp.zeros_like, patch),
        (
            jnp.linspace(-0.3, 0.4, 36).reshape((4, 3, 3)),
            jnp.linspace(0.5, -0.2, 12).reshape((4, 3)),
            jnp.asarray(knot_direction),
            jnp.zeros(6),
        ),
    )
    _, tangent = jax.jvp(functional, (patch,), (direction,))
    gradient = jax.grad(functional)(patch)
    pairing = sum(
        float(jnp.vdot(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(gradient),
            jax.tree_util.tree_leaves(direction),
            strict=True,
        )
    )
    step = 1e-6
    shifted = [
        jax.tree_util.tree_map(
            lambda value, delta, sign=sign: value + sign * step * delta, patch, direction
        )
        for sign in (1.0, -1.0)
    ]
    # Public compiled evaluation binds the perturbed source equations while
    # retaining the immutable atlas-box enclosure metadata.
    compiled_functional = eqx.filter_jit(functional)
    difference = (
        float(compiled_functional(shifted[0])) - float(compiled_functional(shifted[1]))
    ) / (2 * step)
    assert abs(float(tangent)) > 1e-3
    assert float(tangent) == pytest.approx(difference, rel=1e-6, abs=1e-6)
    assert pairing == pytest.approx(float(tangent), rel=1e-10, abs=1e-12)
    public = jax.grad(compiled_functional)(patch)
    for a, b in zip(
        jax.tree_util.tree_leaves(public),
        jax.tree_util.tree_leaves(gradient),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)
    assert float(jnp.max(jnp.abs(gradient.u_knots))) > 0.0
    # Unchanged input round-trips its identity; a deliberately revised source
    # carries no stale identity through the payload.
    assert (
        curve.branch_id
        == geometry.IntersectionCurve.from_payload(curve.payload()).branch_id
    )
    revised = eqx.tree_at(lambda value: value.first.patch, curve, shifted[0])
    with pytest.raises(ValueError):
        geometry.IntersectionCurve.from_payload(revised.payload())

    # Merging the interior knot into a domain end changes span topology: the
    # pinned addresses are refused on the host and inside traced evaluation.
    merged = eqx.tree_at(
        lambda value: value.first.patch.u_knots,
        curve,
        jnp.asarray([0, 0, 0, 0, 1, 1, 1.0]),
    )
    with pytest.raises(ValueError, match="knot-span topology"):
        merged.evaluate(parameters)
    with pytest.raises(eqx.EquinoxRuntimeError, match="knot-span topology"):
        jax.block_until_ready(
            eqx.filter_jit(lambda value: value.evaluate(parameters).point)(merged)
        )


def test_source_extraction_bounds_cover_exact_knot_insertions() -> None:
    from fractions import Fraction

    curve = geometry.BSplineCurve(
        [[0.1, 0.2], [0.7, 0.9], [1.2, -0.4], [2.3, 0.1]],
        [0.7, 1.3, 0.9, 1.1],
        [0, 0, 0, 0.37, 1, 1, 1],
        2,
    )
    for numerical, exact in zip(
        curve.bezier_pieces(), curve.exact_bezier_pieces(), strict=True
    ):
        for lower, value, upper in zip(
            numerical.homogeneous_lower.flat,
            exact.homogeneous_controls.flat,
            numerical.homogeneous_upper.flat,
            strict=True,
        ):
            assert Fraction(float(lower)) <= value <= Fraction(float(upper))
    # Adjacent span endpoints are the same source point exactly, even if their
    # rounded coefficient boxes overlap and cannot decide that equality.
    first, second = curve.exact_bezier_pieces()
    assert all(
        a == b
        for a, b in zip(
            first.homogeneous_controls[-1], second.homogeneous_controls[0], strict=True
        )
    )


def test_intersection_pcurve_subarcs_refine_and_reverse() -> None:
    plane, sphere = _plane([-2, -2, 0.5], [4, 0, 0], [0, 4, 0]), _sphere([0, 0, 0])
    (curve,) = geometry.intersect_surface_regions(plane, sphere).curves
    full = curve.p_curve("first")
    large, small = full.enclosure(0.2, 0.8), full.enclosure(0.49, 0.51)
    assert np.linalg.norm(small[1] - small[0]) < 0.1 * np.linalg.norm(large[1] - large[0])
    samples = np.asarray(full.evaluate(jnp.linspace(0.49, 0.51, 17)))
    assert np.all(samples >= small[0]) and np.all(samples <= small[1])
    reverse = geometry.IntersectionPCurve(
        curve, "first", first=0.2, last=0.8, reversed=True
    )
    np.testing.assert_allclose(
        reverse.evaluate(jnp.asarray([0.2, 0.8])),
        full.evaluate(jnp.asarray([0.8, 0.2])),
        atol=1e-12,
    )


def test_unclosed_exact_trim_is_not_closed_by_chord_tolerance() -> None:
    line = geometry.LineCurve([0, 0], [1, 0])
    segment = geometry.CurveTrimSegment(line, 0, 0.01)
    with pytest.raises(ValueError, match="not closed"):
        geometry.CurveTrimLoop([segment], tolerance=0.1)


def test_rational_denominator_failure_is_explicit() -> None:
    curve = geometry.BSplineCurve.bezier([[0, 0], [1, 1]], [1, -1])
    with pytest.raises(ValueError, match="denominator"):
        geometry.intersect_curve_ranges(
            geometry.CurveRange(curve),
            geometry.CurveRange(geometry.LineCurve([0, 0], [1, 0]), 0, 1),
        )


@pytest.mark.parametrize("mutation", ["dtype", "shape", "unknown", "missing", "nan"])
def test_exact_geometry_decoder_rejects_malformed_support(mutation: str) -> None:
    from phydrax.geometry.brep._intersection_curve import decode_geometry, encode_geometry

    payload = encode_geometry(geometry.LineCurve([0, 0], [1, 0]))
    if mutation == "dtype":
        payload["fields"]["origin"]["dtype"] = "float32"
    elif mutation == "shape":
        payload["fields"]["origin"]["shape"] = [1, 2]
    elif mutation == "unknown":
        payload["fields"]["_live_handle"] = 1
    elif mutation == "missing":
        del payload["fields"]["direction"]
    else:
        payload["fields"]["origin"]["hex"][0] = "nan"
    with pytest.raises(ValueError):
        decode_geometry(payload)


def test_near_tangent_circle_does_not_prune_two_real_roots() -> None:
    line = geometry.LineCurve([-2, 1 - 1e-10], [1, 0])
    circle = geometry.CircleCurve([0, 0], [1, 0], [0, 1], 1)
    result = geometry.intersect_curve_ranges(
        geometry.CurveRange(line, 0, 4), geometry.CurveRange(circle)
    )
    regular = [point for point in result.points if point.kind == "transversal"]
    assert len(regular) == 2
    assert regular[0].point[0] * regular[1].point[0] < 0


def test_c0_spline_hessian_does_not_claim_global_zero_curvature() -> None:
    curve = geometry.BSplineCurve(
        [[0, 0], [1, 1], [2, 0]], [1, 1, 1], [0, 0, 0.5, 1, 1], 1
    )
    lower, upper = curve.derivative_bounds(0.1, 0.9, order=2)
    assert not curve.is_c1_on(0.1, 0.9)
    assert np.all(np.isneginf(lower)) and np.all(np.isposinf(upper))


def test_offset_sphere_retains_exact_operation_and_source_jets() -> None:
    from phydrax.geometry.brep._patches import OffsetSurface

    sphere = _sphere([0, 0, 0]).patch
    offset = OffsetSurface(sphere, 0.25)
    queries = jnp.asarray([[0.0, 0.0], [0.5, 0.3]])
    np.testing.assert_allclose(
        np.linalg.norm(offset.evaluate(queries), axis=1), 1.25, atol=1e-12
    )
    lower, upper = offset.derivative_bounds([[0.2, -0.3], [0.4, 0.3]], order=2)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    lowered = offset.analytic_equivalent()
    if lowered is None:
        pytest.fail(
            "The representable offset sphere did not retain its analytic equivalent."
        )
    np.testing.assert_allclose(
        lowered.evaluate(queries), offset.evaluate(queries), atol=1e-12
    )


def test_nonclamped_source_bezier_bounds_preserve_original_domain() -> None:
    from fractions import Fraction

    curve = geometry.BSplineCurve(
        [[0.1, 0.2], [0.7, 0.9], [1.2, -0.4], [2.3, 0.1]],
        [0.7, 1.3, 0.9, 1.1],
        [-2, -1, 0, 1, 2, 3, 4],
        2,
    )
    original_knots = np.asarray(curve.knots).copy()
    pieces, exact = curve.bezier_pieces(), curve.exact_bezier_pieces()
    assert [piece.parameter_bounds for piece in pieces] == [((0.0, 1.0),), ((1.0, 2.0),)]
    for numerical, rational in zip(pieces, exact, strict=True):
        assert all(
            Fraction(float(lower)) <= coefficient <= Fraction(float(upper))
            for lower, coefficient, upper in zip(
                numerical.homogeneous_lower.flat,
                rational.homogeneous_controls.flat,
                numerical.homogeneous_upper.flat,
                strict=True,
            )
        )
    box = curve.bounding_box(0.3, 0.4)
    points = np.asarray(curve.evaluate(jnp.linspace(0.3, 0.4, 11)))
    assert np.all(points >= box[0]) and np.all(points <= box[1])
    np.testing.assert_array_equal(np.asarray(curve.knots), original_knots)


def test_trim_box_membership_never_uses_center_alone() -> None:
    trim = geometry.TrimDomain([[0, 0], [1, 0], [1, 1], [0, 1]])
    classified = trim.classify_boxes(
        np.asarray(
            [
                [[0.2, 0.2], [0.4, 0.4]],
                [[1.1, 0.2], [1.3, 0.4]],
                [[0.8, 0.2], [1.2, 0.4]],
            ]
        )
    )
    np.testing.assert_array_equal(classified.resolved, [True, True, False])
    np.testing.assert_array_equal(classified.inside, [True, False, False])


def test_native_period_endpoint_closes_seam_without_reinterpreting_binary_tau() -> None:
    from phydrax.geometry.brep._intersection import NativePeriodEndpoint
    from phydrax.geometry.brep._intersection_curve import (
        decode_geometry,
        encode_geometry,
        PeriodicPCurve,
    )

    patch = _sphere([0, 0, 0]).patch
    horizontal = geometry.LineCurve([0, 0], [1, 0])
    vertical = geometry.LineCurve([0, 0], [0, 1])
    right = geometry.CurveTrimSegment(PeriodicPCurve(vertical, patch, (1, 0)), 0, 1)
    literal = geometry.CurveTrimSegment(horizontal, 0, math.tau)
    endpoint = NativePeriodEndpoint(horizontal, patch, 0)
    exact = geometry.CurveTrimSegment(
        horizontal, 0, endpoint.parameter, last_root=endpoint
    )

    assert not literal.shares_endpoint(right)
    assert exact.shares_endpoint(right)
    assert not exact.shares_endpoint(geometry.CurveTrimSegment(vertical, 0, 1))
    restored = decode_geometry(encode_geometry(exact))
    assert restored.shares_endpoint(right)
    lower, upper = restored.carrier_parameter_enclosure(1, 1)
    from fractions import Fraction

    mathematical_tau = Fraction("6.28318530717958647692528676655900576839433879875021")
    assert Fraction(float(lower)) <= mathematical_tau <= Fraction(float(upper))
    other_source = geometry.LineCurve([0, 1], [1, 0])
    with pytest.raises(ValueError):
        geometry.CurveTrimSegment(other_source, 0, endpoint.parameter, last_root=endpoint)


def test_native_period_endpoint_reads_only_derived_isoline_period_as_tau() -> None:
    from phydrax.geometry.brep._intersection import NativePeriodEndpoint
    from phydrax.geometry.brep._patches import (
        OffsetSurface,
        SurfaceIsoparametricCurve,
        TorusPatch,
    )

    torus = OffsetSurface(
        TorusPatch([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 2.0, 0.5), 0.125
    )
    derived = SurfaceIsoparametricCurve(torus, 1, 0.0)
    for turns in (0, 1):
        assert NativePeriodEndpoint(derived, torus, 0, turns=turns).turns == turns
    with pytest.raises(ValueError):
        NativePeriodEndpoint(derived, torus, 0, turns=2)
    literal = SurfaceIsoparametricCurve(torus, 1, 0.0, parameter_range=(0.0, math.tau))
    NativePeriodEndpoint(literal, torus, 0, turns=0)
    with pytest.raises(ValueError):
        NativePeriodEndpoint(literal, torus, 0, turns=1)


def test_native_period_circle_endpoint_preserves_true_quarter_turns() -> None:
    from fractions import Fraction

    from phydrax.geometry.brep._intersection import NativePeriodEndpoint

    patch = _sphere([0, 0, 0]).patch
    circle = geometry.CircleCurve([0, 0], [1, 0], [0, 1], 1)
    endpoint = NativePeriodEndpoint(circle, patch, 0)
    closed = geometry.CurveTrimSegment(circle, 0, endpoint.parameter, last_root=endpoint)
    literal = geometry.CurveTrimSegment(circle, 0, math.tau)
    assert closed.shares_endpoint(closed)
    assert not literal.shares_endpoint(literal)
    phase = 0.37
    phase_endpoint = NativePeriodEndpoint(circle, patch, 0, rational=Fraction(phase))
    phase_loop = geometry.CurveTrimSegment(
        circle, phase, phase_endpoint.parameter, last_root=phase_endpoint
    )
    assert phase_loop.shares_endpoint(phase_loop)
    literal_phase_loop = geometry.CurveTrimSegment(circle, phase, phase + math.tau)
    assert not literal_phase_loop.shares_endpoint(literal_phase_loop)

    quarter = NativePeriodEndpoint(circle, patch, 0, turns=Fraction(1, 4))
    arc = geometry.CurveTrimSegment(circle, 0, quarter.parameter, last_root=quarter)
    tangent = geometry.CurveTrimSegment(geometry.LineCurve([0, 1], [-1, 0]), 0, 1)
    assert arc.shares_endpoint(tangent)
    with pytest.raises(ValueError):
        NativePeriodEndpoint(circle, patch, 1)


def test_unclamped_degree_multiplicity_trim_endpoint_retains_exact_source_join() -> None:
    controls = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 2.0],
            [-1.0, 2.0],
            [-2.0, 0.0],
            [-1.0, -2.0],
            [1.0, -2.0],
            [1.0, 0.0],
        ],
    )
    weights = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
    knots = [-1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    source = geometry.BSplineCurve(controls, weights, knots, 2)
    closed = geometry.CurveTrimSegment(source, 0.0, 3.0)
    reverse = geometry.CurveTrimSegment(source, 0.0, 3.0, reversed=True)
    assert closed.shares_endpoint(closed)
    assert reverse.shares_endpoint(reverse)
    tangent = geometry.CurveTrimSegment(
        geometry.LineCurve([1.0, 0.0], [0.0, 1.0]), 0.0, 1.0
    )
    assert closed.shares_endpoint(tangent)
    # A numerically tiny authored gap is still a distinct exact source point.
    separated = controls.copy()
    separated[-1, 0] = np.nextafter(separated[-1, 0], np.inf)
    open_source = geometry.BSplineCurve(separated, weights, knots, 2)
    open_trim = geometry.CurveTrimSegment(open_source, 0.0, 3.0)
    assert not open_trim.shares_endpoint(open_trim)
    assert not open_trim.shares_endpoint(tangent)


def test_lower_multiplicity_spline_trim_does_not_claim_boundary_control_identity() -> (
    None
):
    source = geometry.BSplineCurve(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [1.0] * 4,
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        2,
    )
    trim = geometry.CurveTrimSegment(source, 0.0, 2.0)
    assert not trim.shares_endpoint(trim)


@pytest.mark.parametrize(
    "multiplicity", (2, 3), ids=("degree-run", "degree-plus-one-run")
)
def test_active_trim_endpoint_uses_knot_run_not_exterior_control_bank(
    multiplicity: int,
) -> None:
    controls = [[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    knots = [-2.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0]
    if multiplicity == 3:
        controls.append([9.0, 9.0])
        knots = [-1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0]
    source = geometry.BSplineCurve(controls, [1.0] * len(controls), knots, 2)
    closed = geometry.CurveTrimSegment(source, 0.0, 1.0)
    tangent = geometry.CurveTrimSegment(
        geometry.LineCurve([1.0, 0.0], [0.0, 1.0]), 0.0, 1.0
    )
    assert closed.shares_endpoint(closed)
    assert closed.shares_endpoint(tangent)
    controls[3][0] = np.nextafter(1.0, np.inf)
    open_source = geometry.BSplineCurve(controls, [1.0] * len(controls), knots, 2)
    open_trim = geometry.CurveTrimSegment(open_source, 0.0, 1.0)
    assert not open_trim.shares_endpoint(open_trim)


def test_repeated_original_intersection_trim_bounds_preserve_affine_source_wrapper() -> (
    None
):
    from phydrax.geometry.brep._intersection_curve import (
        original_trim_intersection_preparation,
    )

    first = _plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    second = _plane((0.0, 0.5, -0.5), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    discovered = geometry.intersect_surface_regions(first, second)
    assert discovered.complete and len(discovered.curves) == 1
    branch = discovered.curves[0]
    carrier = AffinePCurve(
        branch.p_curve("first"), ((1.0, 0.0), (0.0, 1.0)), (0.125, 0.0)
    )
    trim = geometry.CurveTrimSegment(carrier, 0.0, float(branch.num_charts))
    interval = (0.1, 0.9)
    expected_box = trim.enclosure(*interval)
    expected_jet = trim.derivative_bounds(*interval)
    with original_trim_intersection_preparation((trim,)):
        for _ in range(3):
            np.testing.assert_array_equal(trim.enclosure(*interval), expected_box)
            actual = trim.derivative_bounds(*interval)
            np.testing.assert_array_equal(actual[0], expected_jet[0])
            np.testing.assert_array_equal(actual[1], expected_jet[1])
    np.testing.assert_array_equal(trim.enclosure(*interval), expected_box)
