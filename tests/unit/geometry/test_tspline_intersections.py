from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from phydrax._interpolation import TMesh2D
from phydrax.geometry._rational import fraction_down, fraction_up
from phydrax.geometry.brep._intersection import (
    intersect_surface_regions,
    ParametricIntersectionPolicy,
)
from phydrax.geometry.brep._intersection_curve import (
    IntersectionCurve,
    SurfaceRegion,
)
from phydrax.geometry.brep._patches import (
    BSplineSurfacePatch,
    CylinderPatch,
    PlanePatch,
    SpherePatch,
)
from phydrax.geometry.brep._tspline import TSplineSurfacePatch
from phydrax.geometry.brep._tspline_intersection import _radius_squared_interval


type _PointMap = Callable[[Fraction, Fraction], tuple[Fraction, Fraction, Fraction]]
type _WeightMap = Callable[[Fraction, Fraction], Fraction]


def _tspline_patch(
    point_map: _PointMap,
    weight_map: _WeightMap,
    /,
) -> TSplineSurfacePatch:
    knots = tuple(Fraction(index, 4) for index in range(5))
    mesh = TMesh2D.tensor_product(knots, knots)
    parameters = tuple(anchor.local_knots.greville_parameter for anchor in mesh.anchors)
    points = tuple(point_map(u, v) for u, v in parameters)
    weights = tuple(weight_map(u, v) for u, v in parameters)
    return TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in point] for point in points]),
        np.asarray([float(value) for value in weights]),
        exact_control_points=points,
        exact_weights=weights,
    )


def _linear_nurbs_crossing_patch() -> BSplineSurfacePatch:
    knots = (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(1))
    values = tuple(Fraction(index, 2) for index in range(3))
    points = tuple(
        tuple((2 * u - 1, 2 * v - 1, 2 * v - 1) for v in values) for u in values
    )
    weights = tuple(tuple(Fraction(1) for _ in values) for _ in values)
    return BSplineSurfacePatch(
        np.asarray(
            [[[float(value) for value in point] for point in row] for row in points],
            dtype=np.float64,
        ),
        np.ones((3, 3), dtype=np.float64),
        np.asarray([float(value) for value in knots], dtype=np.float64),
        np.asarray([float(value) for value in knots], dtype=np.float64),
        2,
        2,
        exact_control_points=points,
        exact_weights=weights,
        exact_u_knots=knots,
        exact_v_knots=knots,
    )


def _curve_point_at_first_parameter(
    curve: IntersectionCurve, first_parameter: float, /
) -> np.ndarray:
    for chart in range(curve.num_charts):
        chart_start = float(np.asarray(curve.chart_start[chart, 0]))
        chart_end = float(np.asarray(curve.chart_end[chart, 0]))
        lower = min(chart_start, chart_end)
        upper = max(chart_start, chart_end)
        if lower - 1.0e-12 <= first_parameter <= upper + 1.0e-12:
            fraction = (first_parameter - chart_start) / (chart_end - chart_start)
            return np.asarray(
                curve.evaluate(jnp.asarray(chart + fraction, dtype=jnp.float64)),
                dtype=np.float64,
            )
    raise ValueError(
        "The certified branch atlas does not cover the requested source parameter."
    )


def _assert_branch_certificates(curve: IntersectionCurve, /) -> None:
    assert curve.fully_certified
    assert np.all(np.asarray(curve.contractions, dtype=np.float64) < 1.0)
    lower = np.asarray(curve.chart_lower, dtype=np.float64)
    upper = np.asarray(curve.chart_upper, dtype=np.float64)
    assert lower.shape == upper.shape == (curve.num_charts, 4)
    assert np.all(lower < upper)
    enclosures = np.asarray(
        curve.parameter_enclosures(0.0, float(curve.num_charts)),
        dtype=np.float64,
    )
    assert enclosures.shape == (curve.num_charts, 2, 4)
    assert np.all(enclosures[:, 0] <= enclosures[:, 1])


def test_t_spline_plane_section_with_nonuniform_weights_matches_root_oracle() -> None:
    patch = _tspline_patch(
        lambda u, v: (2 * u - 1, 2 * v - 1, v),
        lambda u, _v: Fraction(1) + u,
    )
    plane = PlanePatch(
        (0.0, 0.0, 0.5),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    result = intersect_surface_regions(
        SurfaceRegion(patch, np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64)),
        SurfaceRegion(plane, np.asarray(((-1.0, -1.0), (1.0, 1.0)), dtype=np.float64)),
        policy=ParametricIntersectionPolicy(),
    )

    assert result.complete
    assert len(result.curves) == 1
    curve = result.curves[0]
    _assert_branch_certificates(curve)
    assert np.ptp(np.asarray(patch.weights, dtype=np.float64)) > 0.0
    assert np.all(np.asarray(patch.weights, dtype=np.float64) > 0.0)
    for u in np.linspace(0.025, 0.975, 39):

        def signed_distance(v: float, /) -> float:
            point = np.asarray(
                patch.evaluate(jnp.asarray((u, v), dtype=jnp.float64)),
                dtype=np.float64,
            )
            return float(point[2] - 0.5)

        v = brentq(signed_distance, 0.0, 1.0, xtol=1.0e-14)
        expected = np.asarray(
            patch.evaluate(jnp.asarray((u, v), dtype=jnp.float64)),
            dtype=np.float64,
        )
        actual = _curve_point_at_first_parameter(curve, float(u))
        np.testing.assert_allclose(actual, expected, atol=2.0e-8, rtol=0.0)


@pytest.mark.parametrize(
    "radius", (Fraction(1), Fraction(99, 100)), ids=("binary", "exact-nonbinary")
)
def test_t_spline_cylinder_intersection_matches_independent_root_oracle(
    radius: Fraction,
) -> None:
    patch = _tspline_patch(
        lambda u, v: (2 * u - 1, Fraction(1, 2) + Fraction(3, 4) * v, v),
        lambda _u, _v: Fraction(1),
    )
    cylinder = CylinderPatch(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        float(radius),
        radius_exact=radius,
    )
    spline_region = SurfaceRegion(
        patch, np.asarray(((0.25, 0.4), (0.75, 0.7)), dtype=np.float64)
    )
    cylinder_region = SurfaceRegion(
        cylinder,
        np.asarray(((0.0, 0.4), (np.pi, 0.7)), dtype=np.float64),
    )

    result = intersect_surface_regions(
        spline_region, cylinder_region, policy=ParametricIntersectionPolicy()
    )

    assert result.complete
    assert len(result.curves) == 1
    curve = result.curves[0]
    _assert_branch_certificates(curve)
    assert not curve.closed
    assert curve.start_kind == curve.end_kind == "boundary"
    node_lower = np.asarray(curve.node_lower)
    node_upper = np.asarray(curve.node_upper)
    node_values = np.asarray(curve.node_values)
    chart_lower = np.asarray(curve.chart_lower)
    chart_upper = np.asarray(curve.chart_upper)
    assert node_lower.shape == node_upper.shape == (curve.num_charts + 1, 4)
    assert np.all(np.asarray(curve.node_axes) == 0)
    assert np.all(node_lower < node_upper)
    assert np.all(node_lower[:, 0] <= node_values)
    assert np.all(node_values <= node_upper[:, 0])
    assert np.array_equal(node_values[:-1], np.asarray(curve.chart_start)[:, 0])
    assert np.array_equal(node_values[1:], np.asarray(curve.chart_end)[:, 0])
    for chart in range(curve.num_charts):
        assert np.all(node_lower[chart : chart + 2] <= chart_upper[chart])
        assert np.all(node_upper[chart : chart + 2] >= chart_lower[chart])

    radius_lower, radius_upper = _radius_squared_interval(cylinder)
    assert float(radius_lower) == fraction_down(radius * radius)
    assert float(radius_upper) == fraction_up(radius * radius)

    for u in np.linspace(0.27, 0.73, 19):

        def residual(value: float, /) -> float:
            point = np.asarray(
                patch.evaluate(jnp.asarray((u, value), dtype=jnp.float64)),
                dtype=np.float64,
            )
            return float(point[0] ** 2 + point[1] ** 2 - float(radius * radius))

        v = brentq(residual, 0.4, 0.7, xtol=1.0e-14)
        expected = np.asarray(
            patch.evaluate(jnp.asarray((u, v), dtype=jnp.float64)), dtype=np.float64
        )
        actual = _curve_point_at_first_parameter(curve, float(u))
        np.testing.assert_allclose(actual, expected, atol=2.0e-8, rtol=0.0)
        chart = int(
            np.argmin(
                np.maximum(
                    np.asarray(curve.chart_lower)[:, 0] - u,
                    u - np.asarray(curve.chart_upper)[:, 0],
                )
            )
        )
        assert (
            float(curve.chart_lower[chart, 1]) <= v <= float(curve.chart_upper[chart, 1])
        )


def test_t_spline_quadric_tangency_returns_singular_unresolved_evidence() -> None:
    patch = _tspline_patch(
        lambda u, v: (2 * u - 1, 2 * v - 1, Fraction(0)),
        lambda u, _v: Fraction(1) + u,
    )
    sphere = SpherePatch(
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        1.0,
    )
    spline_region = SurfaceRegion(
        patch, np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64)
    )
    sphere_region = SurfaceRegion(
        sphere,
        np.asarray(((-np.pi, -0.5 * np.pi), (np.pi, 0.5 * np.pi)), dtype=np.float64),
    )

    result = intersect_surface_regions(
        spline_region, sphere_region, policy=ParametricIntersectionPolicy()
    )

    assert not result.complete
    assert not result.curves
    assert len(result.unresolved) == 1
    evidence = result.unresolved[0]
    assert evidence.reason == "singular"
    assert np.asarray(evidence.lower).shape == (4,)
    assert np.all(np.asarray(evidence.lower) <= np.asarray(evidence.upper))


@pytest.mark.parametrize("second_family", ("nurbs", "tspline"))
def test_t_spline_spline_pair_produces_certified_intersection_graph(
    second_family: Literal["nurbs", "tspline"],
) -> None:
    first_patch = _tspline_patch(
        lambda u, v: (2 * u - 1, 2 * v - 1, Fraction(0)),
        lambda _u, _v: Fraction(1),
    )
    if second_family == "nurbs":
        second_patch = _linear_nurbs_crossing_patch()
    else:
        second_patch = _tspline_patch(
            lambda u, v: (2 * u - 1, 2 * v - 1, 2 * v - 1),
            lambda _u, _v: Fraction(1),
        )
    first_region = SurfaceRegion(
        first_patch, np.asarray(((0.1, 0.0), (0.9, 1.0)), dtype=np.float64)
    )
    second_region = SurfaceRegion(
        second_patch, np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64)
    )

    result = intersect_surface_regions(
        first_region, second_region, policy=ParametricIntersectionPolicy()
    )

    assert result.complete
    assert len(result.curves) == 1
    curve = result.curves[0]
    _assert_branch_certificates(curve)
    for u in np.linspace(0.12, 0.88, 19):
        actual = _curve_point_at_first_parameter(curve, float(u))
        np.testing.assert_allclose(
            actual, (2.0 * u - 1.0, 0.0, 0.0), atol=2.0e-8, rtol=0.0
        )


def test_t_spline_spline_pair_refuses_closed_loop_between_slices() -> None:
    flat = _tspline_patch(
        lambda u, v: (2 * u - 1, 2 * v - 1, Fraction(0)),
        lambda _u, _v: Fraction(1),
    )
    bowl = _tspline_patch(
        lambda u, v: (
            2 * u - 1,
            2 * v - 1,
            (2 * u - 1) ** 2 + (2 * v - 1) ** 2 - Fraction(9, 25),
        ),
        lambda _u, _v: Fraction(1),
    )
    region = np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64)

    result = intersect_surface_regions(
        SurfaceRegion(flat, region),
        SurfaceRegion(bowl, region),
        policy=ParametricIntersectionPolicy(),
    )

    # The section is one closed loop that meets no region face and no first
    # slice; the transversality proof must localize its turning point.
    assert not result.complete
    assert not result.curves
    assert len(result.unresolved) == 1
    evidence = result.unresolved[0]
    assert evidence.reason == "singular"

    def height(u: float, /) -> float:
        point = bowl.evaluate(jnp.asarray((u, 0.5), dtype=jnp.float64))
        return float(np.asarray(point, dtype=np.float64)[2])

    turning = (
        brentq(height, 0.0, 0.5, xtol=1.0e-15),
        brentq(height, 0.5, 1.0, xtol=1.0e-15),
    )
    lower = np.asarray(evidence.lower, dtype=np.float64)
    upper = np.asarray(evidence.upper, dtype=np.float64)
    assert min(abs(0.5 * (lower[0] + upper[0]) - value) for value in turning) < 1.0e-9
    np.testing.assert_allclose(0.5 * (lower + upper)[[1, 3]], (0.5, 0.5), atol=1.0e-9)
