from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from phydrax.geometry.brep._intersection_curve import SurfaceRegion
from phydrax.geometry.brep._patches import BSplineCurve, BSplineSurfacePatch, LineCurve
from phydrax.geometry.brep._spline_face_gap import rational_spline_face_gap_squared


_KNOTS = tuple(Fraction(value) for value in (0, 0, 0, 1, 1, 1))
_SURFACE_WEIGHTS = tuple(tuple(Fraction(i + 1) for _ in range(3)) for i in range(3))
_SURFACE_POINTS = tuple(
    tuple(
        (Fraction(i, 2), Fraction(j, 2), Fraction(int(i == 1 and j == 1)))
        for j in range(3)
    )
    for i in range(3)
)


def _surface() -> SurfaceRegion:
    patch = BSplineSurfacePatch(
        np.asarray(_SURFACE_POINTS, dtype=np.float64),
        np.asarray(_SURFACE_WEIGHTS, dtype=np.float64),
        np.asarray(_KNOTS, dtype=np.float64),
        np.asarray(_KNOTS, dtype=np.float64),
        2,
        2,
        exact_control_points=_SURFACE_POINTS,
        exact_weights=_SURFACE_WEIGHTS,
        exact_u_knots=_KNOTS,
        exact_v_knots=_KNOTS,
    )
    return SurfaceRegion(patch, np.asarray(((0.0, 0.0), (1.0, 1.0))))


def _curve(
    points: tuple[tuple[Fraction, ...], ...], weights: tuple[Fraction, ...]
) -> BSplineCurve:
    return BSplineCurve(
        np.asarray(points, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(_KNOTS, dtype=np.float64),
        2,
        exact_control_points=points,
        exact_weights=weights,
        exact_knots=_KNOTS,
    )


def test_iso_line_lift_has_exactly_zero_gap_and_ulp_pole_shift_is_resolved() -> None:
    region = _surface()
    iso_points = tuple(row[0] for row in _SURFACE_POINTS)
    iso_weights = tuple(row[0] for row in _SURFACE_WEIGHTS)
    chart = LineCurve((0.0, 0.0), (1.0, 0.0))

    exact = rational_spline_face_gap_squared(
        _curve(iso_points, iso_weights), chart, region, 0.0, 1.0
    )
    assert exact == 0

    shift = Fraction(1, 2**50)
    shifted_points = (
        iso_points[0],
        (iso_points[1][0], iso_points[1][1] + shift, iso_points[1][2]),
        iso_points[2],
    )
    shifted = rational_spline_face_gap_squared(
        _curve(shifted_points, iso_weights), chart, region, 0.0, 1.0
    )
    # The shifted pole moves the curve by at most ``shift`` (convex hull) and by
    # w1 B1 / W at t = 1/2, i.e. shift / 2; the bound sits within a factor four.
    assert shifted is not None
    assert shift * shift / 4 <= shifted <= shift * shift


def test_curved_spline_chart_gap_is_sound_and_tight() -> None:
    region = _surface()
    chart_points = (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(1)),
        (Fraction(1), Fraction(0)),
    )
    chart_weights = (Fraction(1), Fraction(1), Fraction(1))
    chart = _curve(chart_points, chart_weights)
    surface = region.patch
    start = np.asarray(surface.evaluate(jnp.asarray([[0.0, 0.0]])))[0]
    end = np.asarray(surface.evaluate(jnp.asarray([[1.0, 0.0]])))[0]
    chord = (
        tuple(Fraction(float(value)) for value in start),
        tuple((Fraction(float(a)) + Fraction(float(b))) / 2 for a, b in zip(start, end)),
        tuple(Fraction(float(value)) for value in end),
    )
    edge = _curve(chord, chart_weights)

    bound = rational_spline_face_gap_squared(edge, chart, region, 0.0, 1.0)

    parameters = jnp.linspace(0.0, 1.0, 2049)
    lifted = np.asarray(surface.evaluate(chart.evaluate(parameters)))
    sampled = np.asarray(edge.evaluate(parameters))
    sampled_squared = float(np.max(np.sum((lifted - sampled) ** 2, axis=-1)))
    assert bound is not None
    assert float(bound) >= sampled_squared * (1.0 - 1.0e-9)
    assert float(bound) <= 4.0 * sampled_squared * (1.0 + 1.0e-9)
