"""Curved planar B-spline profile certification by exact hull subdivision."""

import numpy as np
import pytest

from phydrax.geometry.brep._constructors import (
    PlanarProfile,
    ProfileLine,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._patches import BSplineCurve


def _quadratic(controls: list[tuple[float, float]]) -> BSplineCurve:
    spans = len(controls) - 2
    knots = [0.0] * 3 + [index / spans for index in range(1, spans)] + [1.0] * 3
    return BSplineCurve(controls, [1.0] * len(controls), knots, 2)


def _profile(spline: BSplineCurve) -> PlanarProfile:
    # The spline returns from (2, 0) to the origin over a straight base.
    return PlanarProfile(
        ProfilePlane(), ProfileLoop(((0.0, 0.0), (2.0, 0.0)), (ProfileLine(), spline))
    )


def test_curved_bspline_loop_certifies_with_material_on_the_left() -> None:
    spline = _quadratic([(2.0, 0.0), (1.0, 2.0), (0.0, 0.0)])
    profile = _profile(spline)

    assert profile.outer.vertices == ((0.0, 0.0), (2.0, 0.0))
    assert isinstance(profile.outer.segments[1], BSplineCurve)
    np.testing.assert_array_equal(
        np.asarray(profile.outer.segments[1].control_points),
        np.asarray(spline.control_points),
    )
    assert profile.endpoint_gap_bounds is None


def test_bspline_end_off_its_vertex_keeps_a_certified_gap() -> None:
    # The spline ends 2^-60 above the declared vertex; nothing is moved.
    offset = 2.0**-60
    spline = _quadratic([(2.0, 0.0), (1.0, 2.0), (0.0, offset)])
    profile = _profile(spline)

    rows = profile.endpoint_gap_bounds
    assert rows is not None
    assert rows[0][0] == (0.0, 0.0)
    assert rows[0][1][0] == 0.0 and offset <= rows[0][1][1] <= 2 * offset
    kept = profile.outer.segments[1]
    assert isinstance(kept, BSplineCurve)
    assert tuple(np.asarray(kept.control_points)[-1]) == (0.0, offset)


def test_reversed_curved_bspline_loop_is_reoriented() -> None:
    spline = _quadratic([(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)])
    profile = PlanarProfile(
        ProfilePlane(), ProfileLoop(((2.0, 0.0), (0.0, 0.0)), (ProfileLine(), spline))
    )

    # Clockwise input is reversed so the bulge is traversed from (2, 0).
    reversed_spline = profile.outer.segments[1]
    assert isinstance(reversed_spline, BSplineCurve)
    assert tuple(np.asarray(reversed_spline.control_points)[0]) == (2.0, 0.0)


def test_nonconvex_simple_bspline_loop_certifies() -> None:
    spline = _quadratic([(2.0, 0.0), (2.0, 2.0), (1.0, 0.5), (0.0, 2.0), (0.0, 0.0)])
    profile = _profile(spline)

    assert len(profile.outer.segments) == 2


def test_self_intersecting_bspline_loop_is_refused() -> None:
    # The spline dips below the base line between its endpoints.
    spline = _quadratic([(2.0, 0.0), (3.0, 2.0), (1.0, -1.0), (-1.0, 2.0), (0.0, 0.0)])

    with pytest.raises(ValueError, match="must not touch, cross or overlap"):
        _profile(spline)
