"""Consumer-visible turning certificates for curved profile constructors."""

from math import pi

import numpy as np
import pytest

from phydrax.geometry.brep._constructors import (
    PlanarProfile,
    ProfileArc,
    ProfileCurveUse,
    ProfileLine,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._patches import BSplineCurve, CircleCurve, EllipseCurve


def _spline(controls: list[tuple[float, float]]) -> BSplineCurve:
    spans = len(controls) - 2
    knots = [0.0] * 3 + [i / spans for i in range(1, spans)] + [1.0] * 3
    return BSplineCurve(controls, [1.0] * len(controls), knots, 2)


def _profile(controls: list[tuple[float, float]]) -> PlanarProfile:
    return PlanarProfile(
        ProfilePlane(),
        ProfileLoop((controls[-1], controls[0]), (ProfileLine(), _spline(controls))),
    )


@pytest.mark.parametrize(
    "vertices",
    (
        ((0, 0), (3, 0), (3, 2), (0, 2)),
        ((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)),
        ((0, 0), (4, 0), (4, 4), (2 + 2**-40, 4), (2, 2**-40), (2 - 2**-40, 4), (0, 4)),
    ),
    ids=("convex", "concave", "near-half-turn-reflex"),
)
def test_rational_polygon_profiles_keep_one_turn(
    vertices: tuple[tuple[float, float], ...],
) -> None:
    profile = PlanarProfile(ProfilePlane(), ProfileLoop.polygon(vertices))
    assert len(profile.outer.vertices) == len(vertices)


@pytest.mark.parametrize("exponent", (-540, 540))
def test_polygon_turns_with_extreme_binary64_coordinates(exponent: int) -> None:
    scale = 2.0**exponent
    vertices = ((0.0, 0.0), (2 * scale, 0.0), (2 * scale, scale), (0.0, scale))
    profile = PlanarProfile(ProfilePlane(), ProfileLoop.polygon(vertices))
    assert profile.outer.vertices == vertices


def test_figure_eight_polygon_is_refused() -> None:
    loop = ProfileLoop.polygon(((0, 0), (2, 2), (0, 2), (2, 0)))
    with pytest.raises(ValueError):
        PlanarProfile(ProfilePlane(), loop)


def test_inward_turn_enclosure_cannot_claim_an_integer() -> None:
    from phydrax.geometry.brep._constructors import _isolated_profile_turn
    from phydrax.geometry.brep._intersection_curve import interval_context

    ctx = interval_context(80)
    upper = np.nextafter(np.nextafter(1.0, np.inf), np.inf)
    valid = ctx.mpf((1.0, upper))
    inward = ctx.mpf((np.nextafter(1.0, np.inf), upper))
    assert _isolated_profile_turn(valid) == 1
    assert _isolated_profile_turn(inward) is None
    assert _isolated_profile_turn(ctx.mpf((1.0, 2.0))) is None
    assert _isolated_profile_turn(ctx.mpf((1.25, 1.75))) is None


@pytest.mark.parametrize("exponent", (-540, 0, 540))
def test_convex_spline_turns_at_extreme_scales(exponent: int) -> None:
    scale = 2.0**exponent
    controls = [(2 * scale, 0.0), (scale, 2 * scale), (0.0, 0.0)]
    profile = _profile(controls)
    assert profile.outer.vertices == (controls[-1], controls[0])
    assert isinstance(profile.outer.segments[1], BSplineCurve)
    np.testing.assert_array_equal(profile.outer.segments[1].control_points, controls)


def test_nonconvex_spline_has_a_simple_certified_loop() -> None:
    controls = [(2.0, 0.0), (2.0, 2.0), (1.0, 0.5), (0.0, 2.0), (0.0, 0.0)]
    profile = _profile(controls)
    assert profile.outer.vertices == (controls[-1], controls[0])


def test_corner_arbitrarily_close_to_half_turn_is_not_a_cusp() -> None:
    controls = [(2.0, 0.0), (2.0, 2.0**-80), (0.0, 0.0)]
    profile = _profile(controls)
    assert profile.outer.vertices == (controls[-1], controls[0])


def test_full_circle_preserves_arc() -> None:
    loop = ProfileLoop.circle((0.0, 0.0), 2.0)
    profile = PlanarProfile(ProfilePlane(), loop)
    assert profile.outer == loop
    assert isinstance(profile.outer.segments[0], ProfileArc)


def test_half_circle_with_exact_rooted_endpoints() -> None:
    loop = ProfileLoop(
        ((1.0, 0.0), (-1.0, 0.0)),
        (ProfileArc((0.0, 0.0)), ProfileLine()),
    )
    profile = PlanarProfile(ProfilePlane(), loop)
    assert profile.outer.vertices == loop.vertices


@pytest.mark.parametrize("ellipse", (False, True))
def test_full_conic_use_keeps_its_carrier(ellipse: bool) -> None:
    if ellipse:
        curve = EllipseCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 2.0, 1.0)
        vertex = (2.0, 0.0)
    else:
        curve = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
        vertex = (1.0, 0.0)
    loop = ProfileLoop((vertex,), (ProfileCurveUse(curve, 0.0, 2 * pi),))
    profile = PlanarProfile(ProfilePlane(), loop)
    segment = profile.outer.segments[0]
    assert isinstance(segment, ProfileCurveUse)
    assert segment.curve is curve


def test_crossing_spline_loop_is_refused() -> None:
    controls = [(2.0, 0.0), (3.0, 2.0), (1.0, -1.0), (-1.0, 2.0), (0.0, 0.0)]
    with pytest.raises((ValueError, NotImplementedError)):
        _profile(controls)


def test_reversing_spline_tangent_is_refused() -> None:
    # A degree-one spline reverses at its interior knot.
    curve = BSplineCurve(
        [(2.0, 0.0), (1.0, 1.0), (2.0, 0.0), (0.0, 0.0)],
        [1.0] * 4,
        [0.0, 0.0, 0.25, 0.5, 1.0, 1.0],
        1,
    )
    loop = ProfileLoop(((0.0, 0.0), (2.0, 0.0)), (ProfileLine(), curve))
    with pytest.raises((ValueError, NotImplementedError)):
        PlanarProfile(ProfilePlane(), loop)
