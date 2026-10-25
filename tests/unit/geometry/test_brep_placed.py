"""Pose binding and source-coordinate enclosure contracts."""

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    BSplineCurve,
    CircleCurve,
    CylinderPatch,
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


def _turn(first: float, second: float) -> np.ndarray:
    """A rotation with inexact binary64 coefficients, about z and then x."""
    c1, s1, c2, s2 = np.cos(first), np.sin(first), np.cos(second), np.sin(second)
    about_z = np.asarray(((c1, -s1, 0.0), (s1, c1, 0.0), (0.0, 0.0, 1.0)))
    about_x = np.asarray(((1.0, 0.0, 0.0), (0.0, c2, -s2), (0.0, s2, c2)))
    return about_x @ about_z


def _parameter_boxes(count: int) -> np.ndarray:
    rows = np.arange(count, dtype=np.float64)
    lower = np.stack((rows / (2 * count + 2), rows[::-1] / (3 * count + 3)), axis=1)
    return np.stack((lower, lower + (0.25, 0.375)), axis=1)


def _box_surfaces(split_spline: BSplineCurve) -> dict[str, AbstractSurfacePatch]:
    cylinder = CylinderPatch(
        (1.0, -2.0, 0.5), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 0.75
    )
    placed_curve = PlacedCurve(split_spline, _turn(0.4, -1.1), _TRANSLATION)
    return {
        "plane": PlanePatch((1.0, 2.0, 3.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "rotated": PlacedSurface(cylinder, _turn(0.3, 0.7), (0.125, -3.0, 1e-3)),
        "nested": PlacedSurface(
            PlacedSurface(cylinder, _turn(-0.9, 0.2), (1e5, 0.0, -7.0)),
            _ROTATION,
            _TRANSLATION,
        ),
        "spline": PlacedSurface(
            ExtrusionSurface(placed_curve, (0.0, 0.25, 1.0)), _turn(1.3, 2.1), (0.0, 0.0, 0.0)
        ),
    }


@pytest.mark.parametrize("count", (0, 1, 3, 65))
@pytest.mark.parametrize("name", ("plane", "rotated", "nested", "spline"))
def test_surface_box_banks_equal_scalar_rows(
    split_spline: BSplineCurve, name: str, count: int
) -> None:
    surface = _box_surfaces(split_spline)[name]
    boxes = _parameter_boxes(count)
    with coordinate_source_request((surface, boxes), rows=max(count, 1)) as budget:
        expected = np.empty((count, 2, 3), dtype=np.float64)
        for row, box in enumerate(boxes):
            expected[row] = surface.bounding_box(box, budget=budget)
    with coordinate_source_request((surface, boxes), rows=max(count, 1)) as budget:
        bank = surface.bounding_boxes(boxes, budget=budget)

    assert bank.shape == (count, 2, 3)
    assert bank.dtype == np.float64
    assert bank.tobytes() == expected.tobytes()


_HUGE = 1.5e308
_FINITE_MAX = float(np.finfo(np.float64).max)


@pytest.mark.parametrize("pose", ("definition", "translation", "rotation", "nested"))
def test_surface_box_banks_stop_after_first_nonfinite_row(pose: str) -> None:
    plane = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    finite = ((-2e300, 0.0), (-1e300, 1.0))
    overflow = ((_HUGE, _HUGE), (_HUGE, _HUGE))
    if pose == "definition":
        surface: AbstractSurfacePatch = PlanePatch(
            (_HUGE, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        )
    elif pose == "translation":
        # Every placed row is finite until a finite definition box overflows.
        surface = PlacedSurface(plane, np.eye(3), (_FINITE_MAX, 0.0, 0.0))
        overflow = ((0.0, 0.0), (1.0, 1.0))
    elif pose == "rotation":
        surface = PlacedSurface(plane, _turn(np.pi / 4, 0.0), (0.0, 0.0, 0.0))
    else:
        surface = PlacedSurface(
            PlacedSurface(plane, _turn(np.pi / 4, 0.0), (0.0, 0.0, 0.0)),
            _ROTATION,
            _TRANSLATION,
        )
    # Evaluating the reversed last row would raise; the bank must stop first.
    boxes = np.asarray((finite, finite, overflow, ((1.0, 1.0), (0.0, 0.0))))
    with pytest.raises(ValueError, match="lower <= upper"):
        surface.bounding_box(boxes[3])

    with coordinate_source_request((surface, boxes), rows=4) as budget:
        bank = surface.bounding_boxes(boxes, budget=budget)
    expected = np.stack([surface.bounding_box(box) for box in boxes[:3]])

    assert bank.shape == (3, 2, 3)
    assert np.all(np.isfinite(bank[:2])) and not np.all(np.isfinite(bank[2]))
    assert bank.tobytes() == expected.tobytes()


def test_surface_box_banks_keep_scalar_row_failure_order() -> None:
    surface = PlacedSurface(
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        _turn(0.3, 0.7),
        _TRANSLATION,
    )
    boxes = np.asarray(
        (
            ((0.0, 0.0), (1.0, 1.0)),
            ((1.0, 1.0), (0.0, 0.0)),
            ((0.0, 0.0), (np.inf, 1.0)),
        )
    )
    with pytest.raises(ValueError, match="must be finite"):
        surface.bounding_box(boxes[2])
    # The second row's scalar refusal precedes any later row's refusal.
    with pytest.raises(ValueError, match="lower <= upper"):
        surface.bounding_boxes(boxes)
    with pytest.raises(ValueError, match="num_boxes, 2, 2"):
        surface.bounding_boxes(boxes[0])


def test_surface_box_banks_query_the_current_pose() -> None:
    surface = PlacedSurface(
        PlanePatch((1.0, 2.0, 3.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        _turn(0.3, 0.7),
        _TRANSLATION,
    )
    boxes = _parameter_boxes(3)
    first = surface.bounding_boxes(boxes)
    retained = first.copy()
    boxes[...] = 0.0
    np.testing.assert_array_equal(first, retained)
    original = surface.translation
    object.__setattr__(surface, "translation", original + jnp.asarray((0.0, 0.0, 0.25)))
    try:
        moved = surface.bounding_boxes(_parameter_boxes(3))
        expected = np.stack([surface.bounding_box(box) for box in _parameter_boxes(3)])
    finally:
        object.__setattr__(surface, "translation", original)

    assert moved.tobytes() == expected.tobytes()
    assert not np.array_equal(moved, first)
    np.testing.assert_array_equal(moved[:, :, :2], first[:, :, :2])
