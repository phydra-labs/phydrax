"""Pose binding and source-coordinate enclosure contracts."""

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    CircleCurve,
    ExtrusionSurface,
    LineCurve,
    PlanePatch,
)
from phydrax.geometry.brep._placed import PlacedCurve, PlacedSurface, same_source_pose


# Signed permutations and dyadic translation are exactly represented in binary64.
_ROTATION = np.asarray(
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)), dtype=np.float64
)
_TRANSLATION = np.asarray((2.0, -4.0, 8.0), dtype=np.float64)


def test_surface_pose_preserves_authored_coefficients_and_encloses_evaluation() -> None:
    source = PlanePatch((1.0, 2.0, 3.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    placed = PlacedSurface(source, _ROTATION, _TRANSLATION)
    uv = jnp.asarray((0.25, 0.5), dtype=jnp.float64)
    expected = np.asarray((-(2.0 + 0.5) + 2.0, 1.0 + 0.25 - 4.0, -3.0 + 8.0))

    np.testing.assert_array_equal(placed.rotation, _ROTATION)
    np.testing.assert_array_equal(placed.translation, _TRANSLATION)
    np.testing.assert_array_equal(placed.evaluate(uv), expected)
    box = placed.bounding_box(np.asarray(((0.25, 0.5), (0.25, 0.5)), dtype=np.float64))
    assert np.all(box[0] <= expected)
    assert np.all(expected <= box[1])


def test_curve_pose_shares_pose_identity_without_merging_definitions() -> None:
    curve = PlacedCurve(
        LineCurve((1.0, 2.0, 3.0), (1.0, 0.0, 0.0)), _ROTATION, _TRANSLATION
    )
    surface = PlacedSurface(
        PlanePatch((1.0, 2.0, 3.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        _ROTATION,
        _TRANSLATION,
    )

    assert same_source_pose(curve, surface)
    parameter = jnp.asarray(0.5, dtype=jnp.float64)
    expected = np.asarray((0.0, -2.5, 5.0), dtype=np.float64)
    np.testing.assert_array_equal(curve.evaluate(parameter), expected)
    box = curve.bounding_box(0.5, 0.5)
    assert np.all(box[0] <= expected)
    assert np.all(expected <= box[1])


@pytest.mark.parametrize(
    ("rotation", "field"),
    (
        (np.eye(2, dtype=np.float64), r"placement\.pose"),
        (np.full((3, 3), np.nan, dtype=np.float64), r"placement\.pose"),
        (np.diag(np.asarray((1.0, 1.0, 2.0), dtype=np.float64)), r"placement\.rotation"),
    ),
    ids=("wrong-shape", "nonfinite", "nonorthogonal"),
)
def test_placed_surface_refuses_invalid_pose(rotation: np.ndarray, field: str) -> None:
    source = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match=field):
        PlacedSurface(source, rotation, _TRANSLATION)


def test_placed_curve_refuses_nonspatial_definition_before_invalid_pose() -> None:
    curve = LineCurve((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError, match="placed_curve.definition"):
        PlacedCurve(curve, np.eye(2, dtype=np.float64), _TRANSLATION)


@pytest.fixture
def split_spline() -> BSplineCurve:
    """Two dyadic degree-one spans with a corner at parameter one half."""
    return BSplineCurve(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.5, 1.0, 1.0),
        1,
    )


def test_placed_spline_pieces_preserve_partition_and_coefficient_bounds(
    split_spline: BSplineCurve,
) -> None:
    curve = PlacedCurve(split_spline, _ROTATION, _TRANSLATION)
    with coordinate_source_request((curve, 0.0, 1.0)) as budget:
        pieces = curve.source_pieces(0.0, 1.0, coordinate_budget=budget)

    assert tuple((piece.lower, piece.upper, piece.index) for piece in pieces) == (
        (0.0, 0.5, 0),
        (0.5, 1.0, 1),
    )
    # The native degree-one spline values at quarter parameters are dyadic;
    # signed permutation and translation coefficients are binary64-representable.
    expected = ((2.0, -3.5, 8.0), (1.5, -3.0, 8.0))
    for piece, point in zip(pieces, expected, strict=True):
        parameter = jnp.asarray(0.5 * (piece.lower + piece.upper), dtype=jnp.float64)
        np.testing.assert_array_equal(piece.evaluator.evaluate(parameter), point)
        bounds = piece.evaluator.coefficient_enclosures()
        assert bounds
        for nominal, lower, upper in bounds:
            assert np.all(lower <= nominal)
            assert np.all(nominal <= upper)


def test_placed_extrusion_restricts_nested_placed_spline_spans(
    split_spline: BSplineCurve,
) -> None:
    curve = PlacedCurve(split_spline, _ROTATION, _TRANSLATION)
    surface = PlacedSurface(
        ExtrusionSurface(curve, (0.0, 0.0, 1.0)), _ROTATION, _TRANSLATION
    )
    box = np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64)
    with coordinate_source_request((surface, box)) as budget:
        pieces = surface.source_pieces(box, budget=budget)

    assert tuple(piece.index for piece in pieces) == (0, 1)
    np.testing.assert_array_equal(
        np.asarray([piece.lower for piece in pieces]), ((0.0, 0.0), (0.5, 0.0))
    )
    np.testing.assert_array_equal(
        np.asarray([piece.upper for piece in pieces]), ((0.5, 1.0), (1.0, 1.0))
    )
    expected = ((5.5, -2.0, -0.5), (5.0, -2.5, -0.5))
    for piece, point in zip(pieces, expected, strict=True):
        parameters = jnp.asarray(0.5 * (piece.lower + piece.upper), dtype=jnp.float64)
        np.testing.assert_array_equal(piece.evaluator.evaluate(parameters), point)
        assert piece.evaluator.coefficient_enclosures()


def test_placed_sources_preserve_native_period_symbols() -> None:
    curve = PlacedCurve(
        CircleCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0),
        _ROTATION,
        _TRANSLATION,
    )
    surface = PlacedSurface(
        ExtrusionSurface(curve, (0.0, 0.0, 1.0)), _ROTATION, _TRANSLATION
    )

    assert curve.native_period_symbol() == "two_pi"
    assert surface.native_period_symbols() == ("two_pi", None)
