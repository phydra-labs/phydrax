"""Focused native source-defined B-Rep intersection contracts."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry.brep._boolean_core import affine_root_endpoint, PeriodicPCurve
from phydrax.geometry.brep._correspondence import exact_source_lift
from phydrax.geometry.brep._intersection import (
    BranchRootEndpoint,
    CurveSurfaceIntersectionRoot,
    intersect_surface_regions,
    intersect_trim_curves,
    IntersectionCurvePointRoot,
    SurfaceIntersectionResult,
    TrimIntersectionRoot,
    TrimRootEndpoint,
    TripleSurfaceIntersectionRoot,
)
from phydrax.geometry.brep._intersection_curve import (
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    SurfaceRegion,
)
from phydrax.geometry.brep._patches import (
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    SpherePatch,
)
from phydrax.geometry.brep._root_bindings import BRepRootSupport, BRepVertexRoot


def _cylinder(axis: tuple[float, float, float], radius: float, /) -> CylinderPatch:
    if axis == (0.0, 0.0, 1.0):
        return CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), axis, radius)
    return CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), axis, radius)


def _cylinder_region(patch: CylinderPatch, /) -> SurfaceRegion:
    return SurfaceRegion(
        patch, np.asarray(((0.0, -2.0), (math.tau, 2.0)), dtype=np.float64)
    )


@pytest.mark.parametrize("radius", (1.0, math.nextafter(1.5, 0.0), 3.0))
def test_coaxial_cylinder_torus_radial_separation_is_complete(radius: float) -> None:
    from phydrax.geometry.brep._patches import TorusPatch

    cylinder = _cylinder_region(_cylinder((0.0, 0.0, 1.0), radius))
    torus = SurfaceRegion(
        TorusPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 2.0, 0.5),
        np.asarray(((0.0, 0.0), (math.tau, math.tau))),
    )
    for first, second in ((cylinder, torus), (torus, cylinder)):
        result = intersect_surface_regions(first, second)
        assert result.complete
        assert not result.curves and not result.points and not result.coincident


def test_radial_separation_retains_touching_and_noncoaxial_supports() -> None:
    from phydrax.geometry.brep._intersection import _coaxial_radial_separation
    from phydrax.geometry.brep._patches import TorusPatch

    torus = TorusPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 2.0, 0.5)
    assert not _coaxial_radial_separation(_cylinder((0.0, 0.0, 1.0), 1.5), torus)
    assert not _coaxial_radial_separation(_cylinder((1.0, 0.0, 0.0), 1.0), torus)
    shifted = CylinderPatch((2, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    assert not _coaxial_radial_separation(shifted, torus)
    outer = TorusPatch((0, 0, 3), (1, 0, 0), (0, -1, 0), (0, 0, -1), 4.0, 0.5)
    assert _coaxial_radial_separation(torus, outer)


@pytest.mark.parametrize("reverse", (False, True))
def test_mixed_cylinder_carriers_keep_source_charts_and_complete_branches(
    reverse: bool,
) -> None:
    from phydrax.geometry.brep._patches import RevolutionSurface

    analytic = _cylinder_region(_cylinder((0.0, 0.0, 1.0), 1.0))
    revolved = SurfaceRegion(
        RevolutionSurface(
            LineCurve((-2.0, 0.5, 0.0), (4.0, 0.0, 0.0)),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        np.asarray(((0.0, 0.0), (math.tau, 1.0))),
    )
    first, second = (revolved, analytic) if reverse else (analytic, revolved)
    result = intersect_surface_regions(first, second)
    assert result.complete and len(result.curves) == 2
    for branch in result.curves:
        assert branch.first is first and branch.second is second
        assert branch.fully_certified and branch.closed
        parameters = jnp.linspace(0.15, branch.num_charts - 0.15, 11)
        coupled = branch.evaluate_coupled(parameters)
        np.testing.assert_allclose(
            first.patch.evaluate(coupled.first_parameters),
            coupled.point,
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            second.patch.evaluate(coupled.second_parameters),
            coupled.point,
            rtol=0.0,
            atol=1e-10,
        )


@pytest.mark.parametrize("spline", (False, True))
def test_affine_edges_isolate_exact_analytic_cylinder_trim_roots(spline: bool) -> None:
    from phydrax.geometry.brep._intersection import CurveRange, intersect_curve_region
    from phydrax.geometry.brep._patches import BSplineCurve

    edge = (
        BSplineCurve(
            np.asarray(((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
            np.ones(2),
            np.asarray((0.0, 0.0, 1.0, 1.0)),
            1,
        )
        if spline
        else LineCurve((-2.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    )
    region = _cylinder_region(_cylinder((0.0, 0.0, 1.0), 1.0))
    result = intersect_curve_region(CurveRange(edge, 0.0, 1.0), region)
    assert result.complete and len(result.points) == 2
    roots = sorted(result.points, key=lambda point: float(point.parameters[0]))
    for root, parameter in zip(roots, (0.25, 0.75), strict=True):
        assert root.kind == "crossing" and root.certificate == "krawczyk"
        assert root.parameter_lower[0] <= parameter <= root.parameter_upper[0]
        np.testing.assert_allclose(
            edge.evaluate(jnp.asarray(parameter)), root.point, rtol=0.0, atol=1e-12
        )


def test_unequal_orthogonal_quadrics_have_complete_source_branch_atlas() -> None:
    first = _cylinder((0.0, 0.0, 1.0), 1.0)
    second = _cylinder((1.0, 0.0, 0.0), 0.5)
    first_before = tuple(
        np.asarray(value).copy()
        for value in (
            first.origin,
            first.first_axis,
            first.second_axis,
            first.axis,
            first.radius,
        )
    )
    second_before = tuple(
        np.asarray(value).copy()
        for value in (
            second.origin,
            second.first_axis,
            second.second_axis,
            second.axis,
            second.radius,
        )
    )

    result = intersect_surface_regions(_cylinder_region(first), _cylinder_region(second))

    assert isinstance(result, SurfaceIntersectionResult)
    assert result.complete
    assert result.points == () and result.coincident == ()
    assert len(result.curves) == 2
    parameters = jnp.linspace(0.0, result.curves[0].num_charts, 33, endpoint=False)
    for curve in result.curves:
        assert curve.closed and curve.fully_certified
        assert curve.node_lower.shape == (curve.num_charts + 1, 4)
        coupled = curve.evaluate_coupled(parameters)
        first_uv = curve.p_curve("first").evaluate(parameters)
        second_uv = curve.p_curve("second").evaluate(parameters)
        np.testing.assert_allclose(coupled.first_parameters, first_uv, atol=1e-12)
        np.testing.assert_allclose(coupled.second_parameters, second_uv, atol=1e-12)
        np.testing.assert_allclose(first.evaluate(first_uv), coupled.point, atol=2e-9)
        np.testing.assert_allclose(second.evaluate(second_uv), coupled.point, atol=2e-9)
        np.testing.assert_allclose(coupled.gap, 0.0, atol=2e-9)

    for before, after in zip(
        first_before,
        (first.origin, first.first_axis, first.second_axis, first.axis, first.radius),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(
        second_before,
        (
            second.origin,
            second.first_axis,
            second.second_axis,
            second.axis,
            second.radius,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)


def test_exact_sphere_sphere_tangency_has_separate_rational_root() -> None:
    first_patch = SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    second_patch = SpherePatch((0, 2, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    result = intersect_surface_regions(
        SurfaceRegion(
            first_patch,
            np.asarray(((0.0, -math.pi / 2), (math.tau, math.pi / 2)), dtype=np.float64),
        ),
        SurfaceRegion(
            second_patch,
            np.asarray(((0.0, -math.pi / 2), (math.tau, math.pi / 2)), dtype=np.float64),
        ),
    )

    assert result.complete
    assert result.curves == ()
    assert [point.kind for point in result.points] == ["tangent_point"]
    assert result.points[0].point_exact == (Fraction(0), Fraction(1), Fraction(0))
    np.testing.assert_allclose(result.points[0].point, (0, 1, 0), atol=0.0)


def test_coincident_plane_overlap_is_source_scoped() -> None:
    first = SurfaceRegion(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        np.asarray(((0, 0), (1, 1)), dtype=np.float64),
    )
    second = SurfaceRegion(
        PlanePatch((0.5, 0.25, 0), (0, 1, 0), (-1, 0, 0)),
        np.asarray(((0, 0), (1, 1)), dtype=np.float64),
    )

    result = intersect_surface_regions(first, second)

    assert result.curves == () and result.points == ()
    assert result.coincident
    overlap = result.coincident[0]
    assert overlap.distance_bound == 0.0
    assert not np.any(overlap.partial)
    np.testing.assert_allclose(overlap.cell_lower[0, :2], (0.0, 0.25), atol=0.0)
    np.testing.assert_allclose(overlap.cell_upper[0, :2], (0.5, 1.0), atol=0.0)


def test_equal_orthogonal_cylinders_report_singular_junctions() -> None:
    first = _cylinder_region(_cylinder((0.0, 0.0, 1.0), 1.0))
    second = _cylinder_region(_cylinder((1.0, 0.0, 0.0), 1.0))

    result = intersect_surface_regions(first, second)

    singular = sorted(
        (point for point in result.points if point.kind == "singular_point"),
        key=lambda point: float(point.point[1]),
    )
    assert len(singular) == 2
    assert [point.branch_count for point in singular] == [4, 4]
    np.testing.assert_array_equal(
        singular[0].point_exact, (Fraction(0), Fraction(-1), Fraction(0))
    )
    np.testing.assert_array_equal(
        singular[1].point_exact, (Fraction(0), Fraction(1), Fraction(0))
    )
    assert {region.reason for region in result.unresolved} == {"singular"}
    assert not result.complete


def test_native_period_endpoint_keeps_exact_turns_and_support_authority() -> None:
    from phydrax.geometry.brep._intersection import NativePeriodEndpoint
    from phydrax.geometry.brep._patches import CircleCurve, LineCurve

    circle = CircleCurve((0, 0), (1, 0), (0, 1), 1.0)
    exact = NativePeriodEndpoint(circle, rational=Fraction(1, 3), turns=Fraction(1, 4))
    assert exact.rational == Fraction(1, 3)
    assert exact.turns == Fraction(1, 4)
    cylinder = _cylinder((0.0, 0.0, 1.0), 1.0)
    pcurve = LineCurve((0.0, 0.5), (1.0, 0.0))
    bound = NativePeriodEndpoint(pcurve, cylinder, 0, rational=0, turns=1)
    assert bound.turns == Fraction(1, 1)
    with pytest.raises(ValueError, match="source-declared mathematical 2pi period"):
        NativePeriodEndpoint(LineCurve((0.0, 0.0), (1.0, 1.0)))


def test_transverse_plane_regions_share_exact_bounded_branch_and_endpoints() -> None:
    first = SurfaceRegion(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        np.asarray(((0, 0), (1, 1)), dtype=np.float64),
    )
    second = SurfaceRegion(
        PlanePatch((0.5, 0, -1), (0, 1, 0), (0, 0, 1)),
        np.asarray(((0, 0), (1, 2)), dtype=np.float64),
    )

    result = intersect_surface_regions(first, second)

    assert result.complete
    assert len(result.curves) == 1
    curve = result.curves[0]
    assert curve.start_kind == curve.end_kind == "boundary"
    endpoints = sorted(result.points, key=lambda point: float(point.point[1]))
    assert [point.point_exact for point in endpoints] == [
        (Fraction(1, 2), Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(1), Fraction(0)),
    ]
    parameter = jnp.linspace(0.0, curve.num_charts, 17)
    coupled = curve.evaluate_coupled(parameter)
    np.testing.assert_allclose(
        first.patch.evaluate(coupled.first_parameters), coupled.point, atol=1e-12
    )
    np.testing.assert_allclose(
        second.patch.evaluate(coupled.second_parameters), coupled.point, atol=1e-12
    )


def test_exact_bspline_endpoint_uses_homogeneous_source_arithmetic() -> None:
    from phydrax.geometry.brep._intersection_curve import exact_curve_parameter_point
    from phydrax.geometry.brep._patches import BSplineCurve

    curve = BSplineCurve(((0, 0), (1, 0)), (1, 1), (0, 0, 1, 1), 1)
    assert exact_curve_parameter_point(curve, Fraction(1, 2)) == (
        Fraction(1, 2),
        Fraction(0),
    )


def test_exact_line_curve_evaluates_algebraic_source_parameters() -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement
    from phydrax.geometry.brep._patches import ExactLineCurve

    parameter = RadicalElement.sqrt(Fraction(1, 2))
    curve = ExactLineCurve((0, 0, 0), (2, 4, 6))
    assert curve.evaluate_exact(parameter) == (
        2 * parameter,
        4 * parameter,
        6 * parameter,
    )
    assert curve.evaluate_exact(Fraction(1, 2)) == (
        Fraction(1),
        Fraction(2),
        Fraction(3),
    )


def test_trim_root_replay_preserves_its_authored_isolation_box() -> None:
    from phydrax.geometry.brep._intersection import TrimIntersectionRoot
    from phydrax.geometry.brep._patches import LineCurve

    first = LineCurve((0.0, 0.0), (1.0, 1.0))
    second = LineCurve((0.0, 1.0), (1.0, -1.0))
    lower = np.asarray((0.4, 0.4), dtype=np.float64)
    upper = np.asarray((0.6, 0.6), dtype=np.float64)

    root = TrimIntersectionRoot(
        first,
        second,
        parameter_lower=lower,
        parameter_upper=upper,
    )
    replayed = TrimIntersectionRoot(
        root.first,
        root.second,
        parameter_lower=root.parameter_lower,
        parameter_upper=root.parameter_upper,
    )

    np.testing.assert_array_equal(root.parameter_lower, lower)
    np.testing.assert_array_equal(root.parameter_upper, upper)
    assert replayed.root_id == root.root_id
    np.testing.assert_array_equal(replayed.parameter_lower, root.parameter_lower)
    np.testing.assert_array_equal(replayed.parameter_upper, root.parameter_upper)


def test_radical_conic_root_accepts_exact_unit_direction() -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement
    from phydrax.geometry.brep._patches import CircleCurve, ConicParameterRoot

    conic = CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0)
    root10 = RadicalElement.sqrt(Fraction(10))
    cosine = Fraction(3, 10) * root10
    sine = Fraction(1, 10) * root10
    point = (cosine, sine, Fraction(0))
    root = ConicParameterRoot.from_exact_point(conic, point, math.atan2(1, 3))

    assert root is not None
    assert root.cosine_exact == cosine and root.sine_exact == sine
    assert root.exact_point() == point
    assert (
        ConicParameterRoot.from_exact_point(
            conic, (Fraction(1), Fraction(1), Fraction(0)), math.atan2(1, 1)
        )
        is None
    )


def test_surface_mapped_revolution_lift_requires_identical_source_composition() -> None:
    from phydrax.geometry.brep._correspondence import (
        exact_source_lift,
        native_edge_pcurve,
    )
    from phydrax.geometry.brep._patches import (
        BSplineCurve,
        RevolutionSurface,
        SurfaceMappedCurve,
    )

    profile = BSplineCurve(((1, 0, 0), (1, 0, 1)), (1, 1), (0, 0, 1, 1), 1)
    surface = RevolutionSurface(profile, (0, 0, 0), (0, 0, 1))
    pcurve = BSplineCurve(((0, 0), (0, 1)), (1, 1), (0, 0, 1, 1), 1)
    mapped = SurfaceMappedCurve(surface, pcurve)
    region = SurfaceRegion(surface, np.asarray(((0, 0), (math.tau, 1)), dtype=np.float64))

    assert native_edge_pcurve(mapped, region.patch) is pcurve
    assert exact_source_lift(mapped, pcurve, region, 0.0, 1.0)
    different = BSplineCurve(((0, 0.25), (0, 1)), (1, 1), (0, 0, 1, 1), 1)
    assert not exact_source_lift(mapped, different, region, 0.0, 1.0)


def test_regular_plane_sphere_quadric_branch_has_coupled_certified_atlas() -> None:
    plane = SurfaceRegion(
        PlanePatch((0, 0, 0.5), (1, 0, 0), (0, 1, 0)),
        np.asarray(((-2, -2), (2, 2)), dtype=np.float64),
    )
    sphere = SurfaceRegion(
        SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0),
        np.asarray(((0, -math.pi / 2), (math.tau, math.pi / 2)), dtype=np.float64),
    )

    result = intersect_surface_regions(plane, sphere)

    assert result.complete
    assert len(result.curves) == 1
    curve = result.curves[0]
    assert curve.closed and curve.fully_certified
    parameter = jnp.linspace(0.0, curve.num_charts, 17, endpoint=False)
    coupled = curve.evaluate_coupled(parameter)
    np.testing.assert_allclose(
        plane.patch.evaluate(coupled.first_parameters), coupled.point, atol=2e-9
    )
    np.testing.assert_allclose(
        sphere.patch.evaluate(coupled.second_parameters), coupled.point, atol=2e-9
    )
    np.testing.assert_allclose(coupled.gap, 0.0, atol=2e-9)


@pytest.mark.parametrize(
    "axis_scale", (1.0, 4.0, -2.0), ids=("unit", "scaled", "reversed-scaled")
)
def test_plane_cylinder_cut_is_one_complete_closed_ellipse(axis_scale: float) -> None:
    plane = SurfaceRegion(
        PlanePatch((-3, -3, -1.5), (6, 0, 3), (0, 6, 0)),
        np.asarray(((0, 0), (1, 1)), dtype=np.float64),
    )
    cylinder = SurfaceRegion(
        CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, axis_scale), 1.0),
        np.asarray(
            ((0, -3 / abs(axis_scale)), (math.tau, 3 / abs(axis_scale))),
            dtype=np.float64,
        ),
    )

    result = intersect_surface_regions(plane, cylinder)

    assert result.complete
    assert len(result.curves) == 1
    curve = result.curves[0]
    assert curve.closed and curve.fully_certified
    parameter = jnp.linspace(0.0, curve.num_charts, 33, endpoint=False)
    coupled = curve.evaluate_coupled(parameter)
    np.testing.assert_allclose(
        curve.p_curve("first").evaluate(parameter), coupled.first_parameters, atol=1e-12
    )
    np.testing.assert_allclose(
        curve.p_curve("second").evaluate(parameter), coupled.second_parameters, atol=1e-12
    )
    points = np.asarray(coupled.point)
    np.testing.assert_allclose(np.hypot(points[:, 0], points[:, 1]), 1.0, atol=2e-9)
    np.testing.assert_allclose(points[:, 2], 0.5 * points[:, 0], atol=2e-9)
    np.testing.assert_allclose(
        np.asarray(coupled.second_parameters)[:, 1],
        points[:, 2] / axis_scale,
        atol=2e-9,
    )


def test_nonrational_cylinder_frame_has_partial_plane_branch_and_disjoint_refusal() -> (
    None
):
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    inverse_root17 = RadicalElement.sqrt(Fraction(1, 17))
    axis_exact = (inverse_root17, Fraction(0), 4 * inverse_root17)
    first_axis_exact = (4 * inverse_root17, Fraction(0), -inverse_root17)
    second_axis_exact = (Fraction(0), Fraction(1), Fraction(0))
    cylinder_patch = CylinderPatch(
        (0, 0, 0),
        tuple(float(value) for value in first_axis_exact),
        tuple(float(value) for value in second_axis_exact),
        tuple(float(value) for value in axis_exact),
        0.25,
        first_axis_exact=first_axis_exact,
        second_axis_exact=second_axis_exact,
        axis_exact=axis_exact,
        radius_exact=Fraction(1, 4),
    )
    cylinder = SurfaceRegion(
        cylinder_patch,
        np.asarray(((0.0, -10.0), (math.pi / 2, 10.0)), dtype=np.float64),
    )
    plane = SurfaceRegion(
        PlanePatch((0, 0, 3), (1, 0, 0), (0, 1, 0)),
        np.asarray(((-20, -20), (20, 20)), dtype=np.float64),
    )

    result = intersect_surface_regions(plane, cylinder)

    assert result.complete and len(result.curves) == 1
    curve = result.curves[0]
    assert curve.fully_certified and not curve.closed
    assert curve.start_kind == curve.end_kind == "boundary"
    endpoint_parameter = jnp.asarray((0.0, float(curve.num_charts)), dtype=jnp.float64)
    coupled = curve.evaluate_coupled(endpoint_parameter)
    np.testing.assert_allclose(
        curve.p_curve("second").evaluate(endpoint_parameter)[:, 0],
        (0.0, math.pi / 2),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        plane.patch.evaluate(coupled.first_parameters), coupled.point, atol=2e-9
    )
    np.testing.assert_allclose(
        cylinder_patch.evaluate(coupled.second_parameters), coupled.point, atol=2e-9
    )

    disjoint_plane = SurfaceRegion(
        PlanePatch((0, 0, 100), (1, 0, 0), (0, 1, 0)),
        np.asarray(((-20, -20), (20, 20)), dtype=np.float64),
    )
    disjoint = intersect_surface_regions(disjoint_plane, cylinder)
    assert disjoint.complete and disjoint.curves == () and disjoint.points == ()
    assert disjoint.unresolved == ()


@pytest.mark.parametrize("axis_scale", (1, 4), ids=("unit", "scaled"))
def test_plane_cylinder_conic_carriers_have_exact_coupled_source_lifts(
    axis_scale: int,
) -> None:
    from phydrax.discretization._exact_algebraic_geometry import (
        ExactScalar,
        RadicalElement,
    )
    from phydrax.geometry.brep._correspondence import exact_source_lift
    from phydrax.geometry.brep._intersection import (
        ConicParameterEndpoint,
        exact_plane_cylinder_conic_section,
    )
    from phydrax.geometry.brep._patches import (
        CircleCurve,
        ConicParameterRoot,
        ConicRootLinePCurve,
        EllipseCurve,
        LineCurve,
        PlaneCylinderSectionPCurve,
    )
    from phydrax.geometry.brep._root_bindings import BRepConicVertex, BRepVertexRoot

    plane_patch = PlanePatch((0, 0, 0), (1, 0, 0.5), (0, 1, 0.25))
    plane = SurfaceRegion(
        plane_patch,
        np.asarray(((-2, -2), (2, 2)), dtype=np.float64),
    )
    cylinder_patch = CylinderPatch(
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, axis_scale), 1.0
    )
    cylinder = SurfaceRegion(
        cylinder_patch,
        np.asarray(((0, -1), (math.pi / 2, 1)), dtype=np.float64),
    )
    angle_circle = CircleCurve(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        1.0,
        center_exact=(Fraction(0), Fraction(0), Fraction(0)),
        first_axis_exact=(Fraction(1), Fraction(0), Fraction(0)),
        second_axis_exact=(Fraction(0), Fraction(1), Fraction(0)),
        radius_exact=Fraction(1),
    )
    angle_roots = (
        ConicParameterRoot(angle_circle, Fraction(1), Fraction(0)),
        ConicParameterRoot(angle_circle, Fraction(0), Fraction(1)),
    )
    before = tuple(
        np.asarray(value).copy()
        for value in (
            plane_patch.origin,
            plane_patch.first_axis,
            plane_patch.second_axis,
            cylinder_patch.origin,
            cylinder_patch.first_axis,
            cylinder_patch.second_axis,
            cylinder_patch.axis,
            cylinder_patch.radius,
        )
    )

    result = intersect_surface_regions(plane, cylinder)
    section = exact_plane_cylinder_conic_section(plane, cylinder, angle_roots)

    assert result.complete and len(result.curves) == 1
    assert section is not None
    assert isinstance(section.edge_curve, EllipseCurve)
    assert isinstance(section.plane_pcurve, PlaneCylinderSectionPCurve)
    assert isinstance(section.cylinder_pcurve, PlaneCylinderSectionPCurve)
    assert section.edge_curve.first_radius_exact == RadicalElement.sqrt(Fraction(21, 16))
    assert section.edge_curve.second_radius_exact == Fraction(1)
    assert section.endpoint_roots[0].exact_point() == (
        Fraction(1),
        Fraction(0),
        Fraction(1, 2),
    )
    assert section.endpoint_roots[1].exact_point() == (
        Fraction(0),
        Fraction(1),
        Fraction(1, 4),
    )
    assert exact_source_lift(
        section.edge_curve,
        section.plane_pcurve,
        plane,
        *section.parameter_range,
    )
    assert exact_source_lift(
        section.edge_curve,
        section.cylinder_pcurve,
        cylinder,
        *section.parameter_range,
    )

    root = section.endpoint_roots[0]
    root_point = root.exact_point()
    point_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        root_point[0],
        root_point[1],
        root_point[2],
    )
    vertex_root = BRepVertexRoot(BRepConicVertex(root))
    plane_line = LineCurve(
        (1, 0),
        (0, 1),
        origin_exact=(Fraction(1), Fraction(0)),
        direction_exact=(Fraction(0), Fraction(1)),
    )
    section_plane_endpoint = ConicParameterEndpoint(
        root, section.plane_pcurve, None, point_exact
    )
    line_plane_endpoint = ConicParameterEndpoint(
        root, plane_line, Fraction(0), point_exact
    )
    assert vertex_root.same_uv_endpoint(
        plane.patch,
        section.plane_pcurve,
        section_plane_endpoint,
        plane_line,
        line_plane_endpoint,
    )

    circle_center = (
        Fraction(1) - root.cosine_exact,
        -root.sine_exact,
    )
    circle_plane = CircleCurve(
        tuple(float(value) for value in circle_center),
        (1, 0),
        (0, 1),
        1.0,
        center_exact=circle_center,
        first_axis_exact=(Fraction(1), Fraction(0)),
        second_axis_exact=(Fraction(0), Fraction(1)),
        radius_exact=Fraction(1),
    )
    circle_plane_endpoint = ConicParameterEndpoint(root, circle_plane, None, point_exact)
    assert vertex_root.same_uv_endpoint(
        plane.patch,
        section.plane_pcurve,
        section_plane_endpoint,
        circle_plane,
        circle_plane_endpoint,
    )
    wrong_plane_line = LineCurve(
        (2, 0),
        (0, 1),
        origin_exact=(Fraction(2), Fraction(0)),
        direction_exact=(Fraction(0), Fraction(1)),
    )
    wrong_plane_endpoint = ConicParameterEndpoint(
        root, wrong_plane_line, Fraction(0), point_exact
    )
    assert not vertex_root.same_uv_endpoint(
        plane.patch,
        section.plane_pcurve,
        section_plane_endpoint,
        wrong_plane_line,
        wrong_plane_endpoint,
    )

    contact_pcurve = ConicRootLinePCurve(
        angle_roots[0],
        Fraction(1, 2 * axis_scale),
        Fraction(1),
        parameter_range=(0.0, 1.0),
    )
    section_cylinder_endpoint = ConicParameterEndpoint(
        root, section.cylinder_pcurve, None, point_exact
    )
    contact_cylinder_endpoint = ConicParameterEndpoint(
        angle_roots[0], contact_pcurve, Fraction(0), point_exact
    )
    assert vertex_root.same_uv_endpoint(
        cylinder_patch,
        section.cylinder_pcurve,
        section_cylinder_endpoint,
        contact_pcurve,
        contact_cylinder_endpoint,
    )
    unrelated_angle = ConicParameterEndpoint(
        angle_roots[1], contact_pcurve, Fraction(0), point_exact
    )
    assert not vertex_root.same_uv_endpoint(
        cylinder_patch,
        section.cylinder_pcurve,
        section_cylinder_endpoint,
        contact_pcurve,
        unrelated_angle,
    )
    wrong_winding_root = ConicParameterRoot(
        angle_circle, Fraction(1), Fraction(0), turns=1
    )
    wrong_winding_pcurve = ConicRootLinePCurve(
        wrong_winding_root,
        Fraction(1, 2 * axis_scale),
        Fraction(1),
        parameter_range=(0.0, 1.0),
    )
    wrong_winding_endpoint = ConicParameterEndpoint(
        wrong_winding_root, wrong_winding_pcurve, Fraction(0), point_exact
    )
    assert not vertex_root.same_uv_endpoint(
        cylinder_patch,
        section.cylinder_pcurve,
        section_cylinder_endpoint,
        wrong_winding_pcurve,
        wrong_winding_endpoint,
    )

    parameters = jnp.linspace(*section.parameter_range, 17)
    points = section.edge_curve.evaluate(parameters)
    np.testing.assert_allclose(
        plane.patch.evaluate(section.plane_pcurve.evaluate(parameters)),
        points,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        cylinder_patch.evaluate(section.cylinder_pcurve.evaluate(parameters)),
        points,
        atol=3e-12,
    )
    for previous, current in zip(
        before,
        (
            plane_patch.origin,
            plane_patch.first_axis,
            plane_patch.second_axis,
            cylinder_patch.origin,
            cylinder_patch.first_axis,
            cylinder_patch.second_axis,
            cylinder_patch.axis,
            cylinder_patch.radius,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(previous, current)


@pytest.mark.parametrize("height", (0.3, -0.3))
def test_horizontal_torus_plane_has_two_complete_source_circle_branches(
    height: float,
) -> None:
    from phydrax.geometry.brep._patches import TorusPatch

    torus = SurfaceRegion(
        TorusPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 2.0, 0.5),
        np.asarray(((0, 0), (math.tau, math.tau)), dtype=np.float64),
    )
    plane = SurfaceRegion(
        PlanePatch((-3, -3, height), (6, 0, 0), (0, 6, 0)),
        np.asarray(((0, 0), (1, 1)), dtype=np.float64),
    )

    result = intersect_surface_regions(torus, plane)

    assert result.complete
    assert len(result.curves) == 2
    radii = []
    for curve in result.curves:
        assert curve.closed and curve.fully_certified
        parameter = jnp.linspace(0.0, curve.num_charts, 49, endpoint=False)
        coupled = curve.evaluate_coupled(parameter)
        np.testing.assert_allclose(
            curve.p_curve("first").evaluate(parameter),
            coupled.first_parameters,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            curve.p_curve("second").evaluate(parameter),
            coupled.second_parameters,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            torus.patch.evaluate(coupled.first_parameters), coupled.point, atol=2e-9
        )
        np.testing.assert_allclose(
            plane.patch.evaluate(coupled.second_parameters), coupled.point, atol=2e-9
        )
        points = np.asarray(coupled.point)
        radius = np.linalg.norm(points[:, :2], axis=1)
        np.testing.assert_allclose(radius, radius[0], atol=2e-9)
        radii.append(float(radius[0]))
    np.testing.assert_allclose(sorted(radii), (1.6, 2.4), atol=2e-9)


def test_public_revolution_sphere_and_torus_sections_keep_source_faces() -> None:
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.brep._constructors import brep_sphere, brep_torus
    from phydrax.geometry.brep._patches import RevolutionSurface

    contract = SpatialCoordinateContract.si()
    sphere_model = brep_sphere(1.0, coordinate_contract=contract)
    sphere_face = next(
        index
        for index, patch in enumerate(sphere_model.patches)
        if isinstance(patch, RevolutionSurface)
    )
    sphere_patch = sphere_model.patches[sphere_face]
    sphere = SurfaceRegion(
        sphere_patch,
        np.asarray(sphere_model.parameter_bounds[sphere_face], dtype=np.float64),
    )
    sphere_plane = SurfaceRegion(
        PlanePatch((-2, -2, 0.5), (4, 0, 0), (0, 4, 0)),
        np.asarray(((0, 0), (1, 1)), dtype=np.float64),
    )
    sphere_result = intersect_surface_regions(sphere, sphere_plane)
    assert sphere_result.complete and len(sphere_result.curves) == 1
    sphere_curve = sphere_result.curves[0]
    assert sphere_curve.closed and sphere_curve.first.patch is sphere_patch
    sphere_parameters = jnp.linspace(0.0, sphere_curve.num_charts, 17, endpoint=False)
    sphere_points = sphere_curve.evaluate_coupled(sphere_parameters)
    np.testing.assert_allclose(
        sphere_patch.evaluate(sphere_points.first_parameters),
        sphere_points.point,
        atol=2e-9,
    )

    torus_model = brep_torus(2.0, 0.5, coordinate_contract=contract)
    torus_face = next(
        index
        for index, patch in enumerate(torus_model.patches)
        if isinstance(patch, RevolutionSurface)
    )
    torus_patch = torus_model.patches[torus_face]
    torus = SurfaceRegion(
        torus_patch,
        np.asarray(torus_model.parameter_bounds[torus_face], dtype=np.float64),
    )
    torus_plane = SurfaceRegion(
        PlanePatch((-3, -3, 0.3), (6, 0, 0), (0, 6, 0)),
        np.asarray(((0, 0), (1, 1)), dtype=np.float64),
    )
    torus_result = intersect_surface_regions(torus, torus_plane)
    assert torus_result.complete and len(torus_result.curves) == 2
    radii = []
    for curve in torus_result.curves:
        assert curve.closed and curve.first.patch is torus_patch
        parameters = jnp.linspace(0.0, curve.num_charts, 17, endpoint=False)
        coupled = curve.evaluate_coupled(parameters)
        np.testing.assert_allclose(
            torus_patch.evaluate(coupled.first_parameters), coupled.point, atol=2e-9
        )
        radii.append(float(np.linalg.norm(np.asarray(coupled.point[0, :2]))))
    np.testing.assert_allclose(sorted(radii), (1.6, 2.4), atol=2e-9)


@pytest.fixture(scope="module")
def _orthogonal_branch_root() -> tuple[
    IntersectionCurve, IntersectionCurve, TripleSurfaceIntersectionRoot
]:
    result = intersect_surface_regions(
        _cylinder_region(_cylinder((0.0, 0.0, 1.0), 1.0)),
        _cylinder_region(_cylinder((1.0, 0.0, 0.0), 0.5)),
    )
    if not result.complete or len(result.curves) != 2:
        raise AssertionError("The fixture requires both certified cylinder components.")
    curve, sibling = result.curves
    coupled = curve.evaluate_coupled(jnp.asarray(0.5, dtype=jnp.float64))
    point = np.asarray(coupled.point, dtype=np.float64)
    plane = SurfaceRegion(
        PlanePatch((0.0, float(point[1]), 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        np.asarray(((-2.0, -2.0), (2.0, 2.0)), dtype=np.float64),
    )
    center = np.concatenate(
        (
            np.asarray(coupled.first_parameters),
            np.asarray(coupled.second_parameters),
            point[[0, 2]],
        )
    )
    root = TripleSurfaceIntersectionRoot(
        curve.first,
        curve.second,
        plane,
        parameter_lower=center - 1.0e-7,
        parameter_upper=center + 1.0e-7,
    )
    return curve, sibling, root


def test_branch_endpoint_refuses_another_component_with_overlapping_axis(
    _orthogonal_branch_root: tuple[
        IntersectionCurve, IntersectionCurve, TripleSurfaceIntersectionRoot
    ],
) -> None:
    curve, sibling, root = _orthogonal_branch_root
    endpoint = BranchRootEndpoint(root, curve, 0)
    lower, upper = endpoint.parameter_enclosure()
    assert lower <= 0.5 <= upper
    wrong = BranchRootEndpoint(root, sibling, 0)
    image = wrong.source_image()
    axis = int(np.asarray(sibling.chart_axes)[0])
    start = float(np.asarray(sibling.chart_start)[0, axis])
    end = float(np.asarray(sibling.chart_end)[0, axis])
    assert min(start, end) <= image[0, axis] <= image[1, axis] <= max(start, end)
    with pytest.raises(ValueError, match="outside its selected chart"):
        wrong.parameter_enclosure()


def test_branch_uv_coincidence_preserves_native_periodic_sheet(
    _orthogonal_branch_root: tuple[
        IntersectionCurve, IntersectionCurve, TripleSurfaceIntersectionRoot
    ],
) -> None:
    curve, _, root = _orthogonal_branch_root
    pcurve = curve.p_curve("first")
    endpoint = BranchRootEndpoint(
        root, curve, 0, source_pcurve=pcurve, source_side="first"
    )
    shifted = BranchRootEndpoint(
        root,
        curve,
        0,
        source_pcurve=pcurve,
        source_side="first",
        periodic_shifts=(1, 0, 0, 0),
    )
    binding = BRepVertexRoot(root)
    assert binding.same_uv_endpoint(curve.first.patch, pcurve, endpoint, pcurve, endpoint)
    assert not binding.same_uv_endpoint(
        curve.first.patch, pcurve, endpoint, pcurve, shifted
    )


@pytest.fixture
def _trim_uv_root() -> tuple[
    CylinderPatch, LineCurve, TrimIntersectionRoot, BRepVertexRoot, SurfaceRegion
]:
    patch = _cylinder((0.0, 0.0, 1.0), 1.0)
    curve = LineCurve((0.0, 0.5), (1.0, 0.0))
    transverse = LineCurve((0.25, 0.0), (0.0, 1.0))
    root = TrimIntersectionRoot(
        curve,
        transverse,
        parameter_lower=np.asarray((0.2, 0.4), dtype=np.float64),
        parameter_upper=np.asarray((0.3, 0.6), dtype=np.float64),
    )
    return (
        patch,
        curve,
        root,
        BRepVertexRoot(BRepRootSupport(patch, root)),
        SurfaceRegion(patch, np.asarray(((0.0, 0.0), (math.tau, 1.0)), dtype=np.float64)),
    )


def test_trim_uv_coincidence_preserves_authored_periodic_transport(
    _trim_uv_root: tuple[
        CylinderPatch, LineCurve, TrimIntersectionRoot, BRepVertexRoot, SurfaceRegion
    ],
) -> None:
    patch, curve, root, binding, region = _trim_uv_root
    shifted_curve = PeriodicPCurve(curve, patch, (1, 0))
    endpoint = TrimRootEndpoint(root, "first", source_pcurve=curve, source_surface=region)
    shifted = TrimRootEndpoint(
        root, "first", source_pcurve=shifted_curve, source_surface=region
    )
    assert binding.same_uv_endpoint(patch, shifted_curve, shifted, shifted_curve, shifted)
    assert not binding.same_uv_endpoint(patch, curve, endpoint, shifted_curve, shifted)


def test_trim_uv_coincidence_refuses_unproved_parameter_correspondence(
    _trim_uv_root: tuple[
        CylinderPatch, LineCurve, TrimIntersectionRoot, BRepVertexRoot, SurfaceRegion
    ],
) -> None:
    patch, curve, root, binding, region = _trim_uv_root
    changed = LineCurve((0.0, 0.5), (1.125, 0.0))
    endpoint = TrimRootEndpoint(root, "first", source_pcurve=curve, source_surface=region)
    unproved = TrimRootEndpoint(
        root, "first", source_pcurve=changed, source_surface=region
    )
    assert binding.same_uv_endpoint(patch, curve, endpoint, curve, endpoint)
    assert not binding.same_uv_endpoint(patch, curve, endpoint, changed, unproved)


def test_spatial_trim_uv_uses_source_parameter_not_local_arc_parameter() -> None:
    curve = LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    cutter = SurfaceRegion(
        PlanePatch((2.5, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        np.asarray(((-1.0, -1.0), (1.0, 1.0)), dtype=np.float64),
    )
    root = CurveSurfaceIntersectionRoot(
        curve,
        cutter,
        parameter_lower=np.asarray((2.4, -0.1, -0.1), dtype=np.float64),
        parameter_upper=np.asarray((2.6, 0.1, 0.1), dtype=np.float64),
    )
    patch = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    region = SurfaceRegion(patch, np.asarray(((0.0, -1.0), (4.0, 1.0)), dtype=np.float64))
    pcurve = LineCurve((0.0, 0.0), (1.0, 0.0))
    source = TrimRootEndpoint(root, "first", source_pcurve=pcurve, source_surface=region)
    local = affine_root_endpoint(source, 1.0, -2.0)
    assert local.parameter_enclosure()[1] < source.parameter_enclosure()[0]
    binding = BRepVertexRoot(root)
    assert binding.same_uv_endpoint(patch, pcurve, source, pcurve, local)


def test_coupled_pcurve_reversal_requires_reversed_trim_traversal(
    _orthogonal_branch_root: tuple[
        IntersectionCurve, IntersectionCurve, TripleSurfaceIntersectionRoot
    ],
) -> None:
    curve, _, _ = _orthogonal_branch_root
    forward = IntersectionPCurve(curve, "first", first=0.0, last=1.0)
    reversed_source = IntersectionPCurve(
        curve, "first", first=0.0, last=1.0, reversed=True
    )
    first = TrimRootEndpoint(IntersectionCurvePointRoot(curve, 0.25), "first")
    last = TrimRootEndpoint(IntersectionCurvePointRoot(curve, 0.75), "first")
    assert exact_source_lift(curve, forward, curve.first, 0.25, 0.75)
    assert not exact_source_lift(curve, reversed_source, curve.first, 0.25, 0.75)
    with pytest.raises(ValueError, match="source carrier definition"):
        CurveTrimSegment(reversed_source, 0.25, 0.75, first_root=first, last_root=last)
    traversal = CurveTrimSegment(
        forward, 0.25, 0.75, reversed=True, first_root=first, last_root=last
    )
    np.testing.assert_allclose(
        traversal.evaluate(jnp.asarray(0.0, dtype=jnp.float64)),
        forward.evaluate(jnp.asarray(0.75, dtype=jnp.float64)),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_trim_root_certifies_exact_phase_not_an_admissible_numeric_shadow() -> None:
    direction = RadicalElement.sqrt(Fraction(1, 2))
    root = ConicParameterRoot(
        CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0),
        direction,
        direction,
    )
    shift = Fraction(1, 7)
    source = ConicRootLinePCurve(
        root,
        Fraction(0),
        Fraction(1),
        phase_shift_exact=shift,
        parameter_range=(-1.0, 1.0),
    )
    box = source.bounding_box(0.0, 0.0)
    shadow = float(box[1, 0])
    false_lower = float(np.nextafter(shadow, -np.inf))
    false_upper = float(np.nextafter(shadow, np.inf))
    with localcontext() as context:
        context.prec = 80
        # Independent high-precision pi/4 + 1/7 reference.
        pi = Decimal(
            "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
        )
        exact_angle = pi / 4 + Decimal(1) / 7
        assert exact_angle < Decimal.from_float(false_lower)
    realized = ConicRootLinePCurve(
        root,
        Fraction(0),
        Fraction(1),
        angle_realization=shadow,
        phase_shift_exact=shift,
        parameter_range=(-1.0, 1.0),
    )
    horizontal = LineCurve((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError, match="unique regular root"):
        TrimIntersectionRoot(
            realized,
            horizontal,
            parameter_lower=np.asarray((-0.1, false_lower), dtype=np.float64),
            parameter_upper=np.asarray((0.1, false_upper), dtype=np.float64),
        )
    accepted = TrimIntersectionRoot(
        realized,
        horizontal,
        parameter_lower=np.asarray((-0.1, box[0, 0] - 0.01), dtype=np.float64),
        parameter_upper=np.asarray((0.1, box[1, 0] + 0.01), dtype=np.float64),
    )
    assert accepted.contraction < 1.0


def test_shared_lift_endpoint_refuses_another_root_in_overlapping_isolating_boxes() -> (
    None
):
    from phydrax.geometry.brep._patches import RuledSurface, SurfaceMappedCurve
    from phydrax.geometry.brep._root_bindings import BRepCurveSurfaceLift

    direction = RadicalElement.sqrt(Fraction(1, 2))
    conic_root = ConicParameterRoot(
        CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0),
        direction,
        direction,
    )
    chart = ConicRootLinePCurve(
        conic_root,
        Fraction(0),
        Fraction(1),
        phase_shift_exact=Fraction(1, 7),
        parameter_range=(-1.0, 1.0),
    )
    plane = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    translated = PlanePatch((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    rail = SurfaceMappedCurve(plane, chart)
    ruled = RuledSurface(rail, SurfaceMappedCurve(translated, chart))
    plane_region = SurfaceRegion(
        plane, np.asarray(((0.0, -1.0), (2.0, 1.0)), dtype=np.float64)
    )
    ruled_region = SurfaceRegion(
        ruled, np.asarray(((-1.0, 0.0), (1.0, 1.0)), dtype=np.float64)
    )
    rail_chart = LineCurve((0.0, 0.0), (1.0, 0.0))
    primary = BRepRootSupport(
        plane,
        TrimIntersectionRoot(
            chart,
            LineCurve((0.0, 0.25), (1.0, 0.0)),
            parameter_lower=np.asarray((-0.5, 0.5), dtype=np.float64),
            parameter_upper=np.asarray((0.5, 1.5), dtype=np.float64),
        ),
    )
    alias = BRepRootSupport(
        ruled,
        TrimIntersectionRoot(
            rail_chart,
            LineCurve((0.0, 0.0), (0.0, 1.0)),
            parameter_lower=np.asarray((-0.5, -0.5), dtype=np.float64),
            parameter_upper=np.asarray((0.5, 0.5), dtype=np.float64),
        ),
    )
    lifts = (
        BRepCurveSurfaceLift(rail, plane, chart, 0.0, 1.0, source_region=plane_region),
        BRepCurveSurfaceLift(
            rail, ruled, rail_chart, 0.0, 1.0, source_region=ruled_region
        ),
    )
    with pytest.raises(ValueError, match="exact shared-lift endpoint proof"):
        BRepVertexRoot(primary, aliases=(alias,), source_edge_lifts=lifts)


@pytest.mark.parametrize("turns", (0, 1), ids=("unshifted-wrapper", "native-period"))
def test_mapped_source_lift_preserves_periodic_chart_authority(turns: int) -> None:
    from phydrax.geometry.brep._patches import SurfaceMappedCurve

    patch = _cylinder((0.0, 0.0, 1.0), 1.0)
    chart = LineCurve((0.0, 0.25), (1.0, 0.0))
    curve = SurfaceMappedCurve(patch, chart)
    region = SurfaceRegion(
        patch, np.asarray(((0.0, 0.0), (math.tau, 1.0)), dtype=np.float64)
    )
    periodic = PeriodicPCurve(chart, patch, (turns, 0))
    assert exact_source_lift(curve, periodic, region, 0.0, math.tau)
    shifted = LineCurve(
        (1.0 / 7.0, 0.25),
        (1.0, 0.0),
        origin_exact=(Fraction(1, 7), Fraction(1, 4)),
        direction_exact=(Fraction(1), Fraction(0)),
    )
    assert not exact_source_lift(
        curve, PeriodicPCurve(shifted, patch, (turns, 0)), region, 0.0, math.tau
    )
    other_patch = _cylinder((0.0, 0.0, 1.0), 2.0)
    assert not exact_source_lift(
        curve, PeriodicPCurve(chart, other_patch, (turns, 0)), region, 0.0, math.tau
    )


@pytest.mark.parametrize(
    ("turns", "last", "accepted"),
    ((0, math.pi, True), (1, math.pi, False), (0, math.tau, False)),
    ids=("within-partial-region", "wrong-periodic-sheet", "outside-region"),
)
def test_mapped_source_lift_respects_partial_region(
    turns: int, last: float, accepted: bool
) -> None:
    from phydrax.geometry.brep._patches import SurfaceMappedCurve

    patch = _cylinder((0.0, 0.0, 1.0), 1.0)
    chart = LineCurve((0.0, 0.25), (1.0, 0.0))
    curve = SurfaceMappedCurve(patch, chart)
    region = SurfaceRegion(
        patch, np.asarray(((0.0, 0.0), (math.pi, 1.0)), dtype=np.float64)
    )
    periodic = PeriodicPCurve(chart, patch, (turns, 0))
    assert exact_source_lift(curve, periodic, region, 0.0, last) is accepted


def test_mapped_source_lift_does_not_promote_near_period_bounds_to_full_scope() -> None:
    from phydrax.geometry.brep._patches import SurfaceMappedCurve

    patch = _cylinder((0.0, 0.0, 1.0), 1.0)
    chart = LineCurve((0.0, 0.25), (1.0, 0.0))
    curve = SurfaceMappedCurve(patch, chart)
    upper = np.nextafter(math.tau, -np.inf)
    region = SurfaceRegion(
        patch, np.asarray(((0.0, 0.0), (upper, 1.0)), dtype=np.float64)
    )
    assert exact_source_lift(curve, chart, region, 0.0, 1.0)
    assert not exact_source_lift(
        curve, PeriodicPCurve(chart, patch, (1, 0)), region, 0.0, 1.0
    )


@pytest.mark.parametrize("index", (0, -1))
def test_terminal_trim_subdivision_replaces_the_selected_piece(index: int) -> None:
    from phydrax.geometry._atlas import _SegmentEnclosures, _split_terminal_piece

    segment = CurveTrimSegment(LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0)
    pieces = [(0.0, 1.0, np.asarray(((0.0, 0.0), (1.0, 0.0))))]
    query = np.asarray(((2.0, 2.0), (3.0, 3.0)))
    assert _split_terminal_piece(_SegmentEnclosures(segment), pieces, query, index)
    assert [(first, last) for first, last, _ in pieces] == [(0.0, 0.5), (0.5, 1.0)]


def _planar_crossing_pcurves() -> tuple[
    SurfaceRegion, IntersectionPCurve, IntersectionPCurve
]:
    domain = np.asarray(((-2.0, -2.0), (2.0, 2.0)), dtype=np.float64)
    common = SurfaceRegion(PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)), domain)
    first_cut = SurfaceRegion(PlanePatch((0, 0, 0), (0, 1, 0), (0, 0, 1)), domain)
    second_cut = SurfaceRegion(PlanePatch((0, 0, 0), (1, 0, 0), (0, 0, 1)), domain)
    first = intersect_surface_regions(common, first_cut)
    second = intersect_surface_regions(common, second_cut)
    assert first.complete and second.complete
    assert len(first.curves) == len(second.curves) == 1
    return common, first.curves[0].p_curve("first"), second.curves[0].p_curve("first")


def test_implicit_pcurve_crossing_uses_certified_source_enclosures() -> None:
    common, first, second = _planar_crossing_pcurves()

    result = intersect_trim_curves(
        first,
        second,
        source_patch=common.patch,
    )

    assert result.complete and len(result.points) == 1
    assert result.coincident == ()
    point = result.points[0]
    assert point.kind == "crossing"
    np.testing.assert_allclose(point.point, (0.0, 0.0), atol=1e-12)
    assert point.gap_bound < 1e-12
    assert np.all(np.asarray(point.parameter_lower) <= 0.5)
    assert np.all(np.asarray(point.parameter_upper) >= 0.5)


def test_implicit_pcurve_and_exact_ellipse_have_two_regular_intersections() -> None:
    from phydrax.geometry.brep._intersection import CurveRange
    from phydrax.geometry.brep._patches import EllipseCurve

    common, first, _ = _planar_crossing_pcurves()
    ellipse = EllipseCurve((0.25, 0.125), (1, 0), (0, 1), 1.0, 0.5)
    result = intersect_trim_curves(
        first, CurveRange(ellipse, 0.0, math.tau), source_patch=common.patch
    )
    assert result.complete and len(result.points) == 2
    assert all(point.kind == "crossing" for point in result.points)
    points = np.stack([np.asarray(point.point) for point in result.points])
    np.testing.assert_allclose(points[:, 0], 0.0, atol=1e-10)
    ordered = sorted(result.points, key=lambda point: float(point.point[1]))
    for point, sign in zip(ordered, (-1, 1), strict=True):
        reference = Fraction(1, 8) + sign * RadicalElement.sqrt(Fraction(15)) / 8
        bounds = first.bounding_box(
            float(point.parameter_lower[0]), float(point.parameter_upper[0])
        )
        assert Fraction(float(bounds[0, 1])) <= reference
        assert reference <= Fraction(float(bounds[1, 1]))


@pytest.mark.parametrize("reversed", (False, True), ids=("forward", "reversed"))
def test_nonbinary_trim_interval_retains_exact_endpoint_transport(reversed: bool) -> None:
    common, first, second = _planar_crossing_pcurves()
    view = IntersectionPCurve(first.curve, "first", first=0.1, last=0.7)
    wrapped = CurveTrimSegment(view, 0.1, 0.7, reversed=reversed)
    result = intersect_trim_curves(wrapped, second, source_patch=common.patch)
    assert result.complete and len(result.points) == 1
    point = result.points[0]
    np.testing.assert_allclose(point.point, (0.0, 0.0), atol=1e-12)
    parameter = (Fraction(0.5) - Fraction(0.1)) / (Fraction(0.7) - Fraction(0.1))
    if reversed:
        parameter = 1 - parameter
    assert Fraction(float(point.parameter_lower[0])) <= parameter
    assert parameter <= Fraction(float(point.parameter_upper[0]))


def test_fixed_chart_point_enclosures_keep_the_declared_periodic_sheet(
    _orthogonal_branch_root: tuple[
        IntersectionCurve, IntersectionCurve, TripleSurfaceIntersectionRoot
    ],
) -> None:
    from phydrax.geometry._interval_enclosure import interval_add, interval_subtract

    candidates = tuple(
        (curve, node)
        for curve in _orthogonal_branch_root[:2]
        for node in range(1, curve.num_charts)
        if np.any(
            np.asarray(curve.node_period_shifts[node, 0, :2])
            != np.asarray(curve.node_period_shifts[node, 1, :2])
        )
    )
    assert candidates, (
        "The source pair must contain a declared periodic chart transition."
    )
    curve, node = candidates[0]
    pcurve = curve.p_curve("first")
    boxes = tuple(
        pcurve.bounding_box(float(node), float(node), chart_index=chart)
        for chart in (node - 1, node)
    )
    gauges = np.asarray(curve.node_period_shifts[node], dtype=np.int64)
    for side, box in enumerate(boxes):
        shift = curve.shift_parameter_bounds(gauges[side])
        node_box = np.stack(
            interval_add(
                (np.asarray(curve.node_lower[node]), np.asarray(curve.node_upper[node])),
                shift,
            )
        )
        assert np.all(box[0] <= node_box[1][:2])
        assert np.all(node_box[0][:2] <= box[1])
    delta = gauges[1] - gauges[0]
    axis = int(np.flatnonzero(delta[:2])[0])
    difference = np.stack(
        interval_subtract((boxes[1][0], boxes[1][1]), (boxes[0][0], boxes[0][1]))
    )
    period = curve.shift_parameter_bounds(delta)
    assert difference[0][axis] <= period[0][axis]
    assert period[1][axis] <= difference[1][axis]
    assert boxes[0][1, axis] < boxes[1][0, axis] or boxes[1][1, axis] < boxes[0][0, axis]


def test_shared_edge_coverage_refuses_unproved_rooted_parameter_limits() -> None:
    from phydrax.geometry.brep._boolean_full_overlay import _covered_affine_edge_range
    from phydrax.geometry.brep._model import BRepGeometry

    line = LineCurve((0, 0, 0), (3, 0, 0))
    plane = SurfaceRegion(
        PlanePatch((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        np.asarray(((-1.0, -1.0), (1.0, 1.0)), dtype=np.float64),
    )
    root = CurveSurfaceIntersectionRoot(
        line,
        plane,
        parameter_lower=np.asarray((0.3, -0.1, -0.1), dtype=np.float64),
        parameter_upper=np.asarray((0.4, 0.1, 0.1), dtype=np.float64),
    )
    geometry = BRepGeometry(
        vertex_points=np.asarray(((1.0, 0.0, 0.0), (3.0, 0.0, 0.0)), dtype=np.float64),
        vertex_exact_points=(None, (Fraction(3), Fraction(0), Fraction(0))),
        vertex_roots=(BRepVertexRoot(root), None),
        curves=(line,),
        edge_curves=(0,),
        edge_ranges=np.asarray(((float(Fraction(1, 3)), 1.0),), dtype=np.float64),
        edge_vertices=((0, 1),),
        edge_endpoint_roots=((TrimRootEndpoint(root, "first"), None),),
        pcurves=(),
        coedge_edges=(),
        coedge_senses=(),
        face_loops=(),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    # The native root is at 1/3, strictly after its rounded representative.
    assert Fraction(float(geometry.edge_ranges[0, 0])) < Fraction(1, 3)
    assert _covered_affine_edge_range(geometry, 0, line) is None


@pytest.mark.parametrize("node", (False, True), ids=("pair", "fixed-node"))
@pytest.mark.parametrize("sign", (-1, 1), ids=("negative", "positive"))
def test_source_residual_encloses_underflowed_exact_origin(node: bool, sign: int) -> None:
    from phydrax.geometry.brep._intersection_curve import (
        _NodeSystem,
        _PreparedNode,
        _PreparedPair,
        SurfacePairSystem,
    )

    offset = Fraction(sign, 2**1076)
    cylinder = CylinderPatch(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        1,
        origin_exact=(Fraction(0), Fraction(0), offset),
    )
    plane = PlanePatch((1, 0, 0), (0, 1, 0), (0, 0, 1))
    system = SurfacePairSystem(cylinder, plane)
    prepared = (
        _PreparedNode(_NodeSystem(system, 0, 0.0)) if node else _PreparedPair(system)
    )
    point = np.zeros((1, 4), dtype=np.float64)
    lower, upper = prepared.value.evaluate(point, point)

    # The execution shadow loses this displacement; the source equation does not.
    assert Fraction.from_float(float(lower[0, 2])) <= offset
    assert offset <= Fraction.from_float(float(upper[0, 2]))


def test_tspline_analytic_counterpart_retains_exact_source_origin() -> None:
    from phydrax.geometry.brep._tspline_intersection import _AnalyticSource

    offset = Fraction(1, 2**1076)
    cylinder = CylinderPatch(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        1,
        origin_exact=(Fraction(0), Fraction(0), offset),
    )
    point = np.zeros((2,), dtype=np.float64)
    lower, upper = _AnalyticSource(cylinder).values(point, point)

    assert Fraction.from_float(float(lower[2])) <= offset
    assert offset <= Fraction.from_float(float(upper[2]))


@pytest.mark.parametrize("reverse", (False, True), ids=("forward", "reversed"))
def test_affine_trim_incidence_keeps_boundary_unresolved_without_false_exclusions(
    reverse: bool,
) -> None:
    from phydrax.geometry._atlas import TrimDomain
    from phydrax.geometry.brep._patches import PlaneCylinderSectionPCurve

    corners = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(1)),
    )
    segments = tuple(
        CurveTrimSegment(
            PlaneCylinderSectionPCurve(
                first,
                (last[0] - first[0], last[1] - first[1]),
                (Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0)),
                parameter_range=(0.0, 1.0),
            ),
            0.0,
            1.0,
            reversed=reverse,
        )
        for first, last in zip(corners, (*corners[1:], corners[0]), strict=True)
    )
    domain = TrimDomain(curve_loops=(segments[::-1] if reverse else segments,))
    result = domain.classify(
        np.asarray(
            (
                (0.5, 0.0),
                (0.0, 0.0),
                (0.5, 0.5),
                (2.0, 0.0),
                (0.5, 2.0**-40),
                (0.5, -(2.0**-40)),
            ),
            dtype=np.float64,
        )
    )
    np.testing.assert_array_equal(result.resolved, (False, False, True, True, True, True))
    np.testing.assert_array_equal(result.inside, (False, False, True, False, True, False))
    np.testing.assert_array_equal(result.boundary, (False,) * 6)
