"""Exact dyadic witnesses for angular-window projection certificates."""

import math
from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry.brep._assembly import BRepAffinePlacement
from phydrax.geometry.brep._patches import (
    CylinderPatch,
    LineCurve,
    RevolutionSurface,
    TorusPatch,
)
from phydrax.geometry.brep._projection import (
    _angular_window,
    _prepare_torus_window_filter,
    _radical_difference_sign,
    _torus_window_exact,
    _torus_window_filter,
    _window_frame,
    NativeBRepProjection,
    prepare_brep_projection,
)
from phydrax.geometry.brep._projection_contracts import (
    BRepProjectionResult,
    BRepProjectionStatus,
)
from tests.unit.geometry.test_brep_projection import _instanced_box, _rectangle_face_model


def _trig(value: Fraction, /, *, cosine: bool) -> tuple[Fraction, Fraction]:
    first, degree = (0, 60) if cosine else (1, 61)
    total = sum(
        (
            Fraction((-1) ** ((power - first) // 2), math.factorial(power)) * value**power
            for power in range(first, degree + 1, 2)
        ),
        Fraction(0),
    )
    error = abs(value) ** (degree + 1) / math.factorial(degree + 1)
    return total - error, total + error


def _projection(kind: str, lower: float, upper: float, /) -> NativeBRepProjection:
    origin, first, second, axis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
    if kind == "cylinder":
        patch = CylinderPatch(origin, first, second, axis, 1)
        bounds = ((lower, -1.0), (upper, 1.0))
    else:
        patch = RevolutionSurface(LineCurve((1, 0, -1), (0, 0, 2)), origin, axis)
        bounds = ((lower, 0.0), (upper, 1.0))
    return prepare_brep_projection(_rectangle_face_model(patch, bounds))


def _query(
    projection: NativeBRepProjection, x: Fraction, y: Fraction, /
) -> BRepProjectionResult:
    return projection.project(
        np.asarray(((float(x), float(y), 0.25),), dtype=np.float64),
        np.asarray((2,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
    )


@pytest.mark.parametrize("kind", ("cylinder", "revolution"))
@pytest.mark.parametrize(
    ("x", "y", "edge_hex", "side"),
    (
        (Fraction(1, 8), Fraction(3, 16), "0x1.f730bd281f69bp-1", "upper"),
        (Fraction(3, 16), Fraction(1, 4), "0x1.dac670561bb4fp-1", "upper"),
        (Fraction(1, 8), Fraction(3, 16), "0x1.f730bd281f69cp-1", "lower"),
        (Fraction(3, 16), Fraction(1, 4), "0x1.dac670561bb50p-1", "lower"),
        (Fraction(3, 16), Fraction(-1, 4), "-0x1.dac670561bb50p-1", "upper"),
    ),
)
def test_outside_angular_foot_never_claims_carrier_upper(
    kind: str, x: Fraction, y: Fraction, edge_hex: str, side: str
) -> None:
    edge = float.fromhex(edge_hex)
    sine = _trig(Fraction(edge), cosine=False)
    cosine = _trig(Fraction(edge), cosine=True)
    products = tuple(y * c - x * s for c in cosine for s in sine)
    if side == "upper":
        assert min(products) > 0
        lower, upper = (-2.0, edge) if y < 0 else (0.0, edge)
    else:
        assert max(products) < 0
        lower, upper = edge, 2.0
    result = _query(_projection(kind, lower, upper), x, y)
    residual = float(result.residuals[0])
    if abs(x) == Fraction(3, 16) and abs(y) == Fraction(1, 4):
        # The exact carrier distance is 11/16, but strict outside membership
        # makes the true distance to this face strictly larger.
        assert residual > 11 / 16
    assert int(result.status[0]) != BRepProjectionStatus.UNIQUE


@pytest.mark.parametrize("kind", ("cylinder", "revolution"))
def test_proven_interior_boundary_and_full_turn_cases(kind: str) -> None:
    inside = _query(_projection(kind, 0.0, 2.0), Fraction(1), Fraction(1))
    assert int(inside.status[0]) == BRepProjectionStatus.UNIQUE
    edge = _query(_projection(kind, 0.0, 2.0), Fraction(1), Fraction(0))
    # Exact edge equality is not strictly inside; it cannot be interior-unique.
    assert int(edge.status[0]) != BRepProjectionStatus.UNIQUE
    full = _query(_projection(kind, -math.pi, math.pi), Fraction(1), Fraction(1))
    assert int(full.status[0]) == BRepProjectionStatus.UNIQUE


@pytest.mark.parametrize("kind", ("cylinder", "revolution"))
def test_wide_and_wrapped_windows_use_exact_direction_signs(kind: str) -> None:
    wide = _projection(kind, -2.0, 2.0)
    assert int(_query(wide, Fraction(0), Fraction(1)).status[0]) == (
        BRepProjectionStatus.UNIQUE
    )
    outside = _query(wide, Fraction(-1), Fraction(0))
    assert int(outside.status[0]) != BRepProjectionStatus.UNIQUE
    assert float(outside.residuals[0]) > 0.0
    wrapped = _projection(kind, 2.0, 4.0)
    assert int(_query(wrapped, Fraction(-1), Fraction(0)).status[0]) == (
        BRepProjectionStatus.UNIQUE
    )


def test_exact_nonbinary_frame_keeps_edge_equality_conservative() -> None:
    patch = CylinderPatch(
        (0, 0, 0),
        (0.6, 0.8, 0),
        (-0.8, 0.6, 0),
        (0, 0, 1),
        1,
        first_axis_exact=(Fraction(3, 5), Fraction(4, 5), Fraction(0)),
        second_axis_exact=(Fraction(-4, 5), Fraction(3, 5), Fraction(0)),
    )
    projection = prepare_brep_projection(
        _rectangle_face_model(patch, ((0.0, -1.0), (2.0, 1.0)))
    )
    result = _query(projection, Fraction(3), Fraction(4))
    # The exact cross product against (3/5, 4/5) is zero.
    assert Fraction(3) * Fraction(4, 5) - Fraction(4) * Fraction(3, 5) == 0
    assert int(result.status[0]) != BRepProjectionStatus.UNIQUE
    assert Fraction(float(result.residuals[0])) ** 2 >= 16


def test_placed_nondeferred_upper_covers_world_point_and_exact_plane() -> None:
    placement = BRepAffinePlacement.from_matrix(
        (
            (Fraction(3, 5), Fraction(-4, 5), 0, Fraction(1, 3)),
            (Fraction(4, 5), Fraction(3, 5), 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    projection = prepare_brep_projection(_instanced_box(placement))
    point = np.asarray(((float(Fraction(17, 60)), float(Fraction(7, 20)), -0.125),))
    result = projection.project(
        point,
        np.asarray((2,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        occurrence_paths=(("pair", "right"),),
    )
    assert int(result.status[0]) == BRepProjectionStatus.UNIQUE
    lower = Fraction(float(result.distance_lower_bounds[0]))
    upper = Fraction(float(result.residuals[0]))
    assert lower <= Fraction(1, 8) <= upper
    squared = sum(
        (Fraction(float(a)) - Fraction(float(b))) ** 2
        for a, b in zip(point[0], np.asarray(result.points)[0], strict=True)
    )
    assert upper**2 >= squared


def test_torus_partial_windows_prove_inside_and_reject_rounded_boundary() -> None:
    patch = TorusPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 2, 0.5)
    inside = prepare_brep_projection(
        _rectangle_face_model(patch, ((-1.0, -1.0), (1.0, 1.0)))
    )
    points = np.asarray(((2.5, 0.0, 0.25), (2.25, 0.25, 0.25)))
    result = inside.project(points, np.asarray((2, 2)), np.asarray((0, 0)))
    assert np.all(result.status == BRepProjectionStatus.UNIQUE)
    for index in range(2):
        alone = inside.project(
            points[index : index + 1], np.asarray((2,)), np.asarray((0,))
        )
        assert np.array_equal(alone.parameters[0], result.parameters[index])
        assert np.array_equal(alone.points[0], result.points[index])

    edge = float.fromhex("0x1.dac670561bb4fp-2")
    cosine = _trig(Fraction(edge), cosine=True)
    sine = _trig(Fraction(edge), cosine=False)
    cross = (cosine[0] / 4 - sine[1] / 2, cosine[1] / 4 - sine[0] / 2)
    assert cross[0] > 0 or cross[1] < 0
    v_bounds = (-1.0, edge) if cross[0] > 0 else (edge, 1.0)
    bounds = np.asarray(((-1.0, v_bounds[0]), (1.0, v_bounds[1])))
    first, second = _angular_window(bounds, 0, False), _angular_window(bounds, 1, False)
    frame = _window_frame(patch)
    assert frame is not None
    filt = _prepare_torus_window_filter(patch, frame, first, second)
    assert filt is not None
    assert _torus_window_filter(filt, points[0], first, second) is not True
    assert not _torus_window_exact(patch, frame, points[0], first, second)


def test_exact_radical_difference_sign_uses_square_comparisons() -> None:
    for a in range(-3, 4):
        for b in range(-3, 4):
            expected = a - 2 * b
            assert _radical_difference_sign(Fraction(a), Fraction(b), Fraction(4)) == (
                (expected > 0) - (expected < 0)
            )
    assert _radical_difference_sign(Fraction(7, 5), Fraction(1), Fraction(2)) == -1
    assert _radical_difference_sign(Fraction(3, 2), Fraction(1), Fraction(2)) == 1
    assert _radical_difference_sign(Fraction(-7, 5), Fraction(-1), Fraction(2)) == 1
    assert _radical_difference_sign(Fraction(-3, 2), Fraction(-1), Fraction(2)) == -1
