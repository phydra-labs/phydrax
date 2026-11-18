"""Exact dyadic witnesses for periodic face projection certificates."""

import math
from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry.brep._assembly import BRepAffinePlacement
from phydrax.geometry.brep._patches import CylinderPatch, LineCurve, RevolutionSurface
from phydrax.geometry.brep._projection import NativeBRepProjection, prepare_brep_projection
from phydrax.geometry.brep._projection_contracts import BRepProjectionStatus
from tests.unit.geometry.test_brep_projection import _instanced_box, _rectangle_face_model


def _trig(value: Fraction, *, cosine: bool) -> tuple[Fraction, Fraction]:
    first, degree = (0, 60) if cosine else (1, 61)
    total = sum(
        (Fraction((-1) ** ((n - first) // 2), math.factorial(n)) * value**n
         for n in range(first, degree + 1, 2)),
        Fraction(0),
    )
    error = abs(value) ** (degree + 1) / math.factorial(degree + 1)
    return total - error, total + error


def _projection(kind: str, lower: float, upper: float) -> NativeBRepProjection:
    origin, axis = (0, 0, 0), (0, 0, 1)
    if kind == "cylinder":
        patch = CylinderPatch(origin, (1, 0, 0), (0, 1, 0), axis, 1)
        bounds = ((lower, -1.0), (upper, 1.0))
    else:
        patch = RevolutionSurface(LineCurve((1, 0, -1), (0, 0, 2)), origin, axis)
        bounds = ((lower, 0.0), (upper, 1.0))
    return prepare_brep_projection(_rectangle_face_model(patch, bounds))


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
def test_outside_carrier_foot_is_not_certified(
    kind: str, x: Fraction, y: Fraction, edge_hex: str, side: str
) -> None:
    edge = float.fromhex(edge_hex)
    sine, cosine = _trig(Fraction(edge), cosine=False), _trig(Fraction(edge), cosine=True)
    products = tuple(y * c - x * s for c in cosine for s in sine)
    if side == "upper":
        assert min(products) > 0
        lower, upper = (-2.0, edge) if y < 0 else (0.0, edge)
    else:
        assert max(products) < 0
        lower, upper = edge, 2.0
    result = _projection(kind, lower, upper).project(
        np.asarray(((float(x), float(y), 0.25),), dtype=np.float64),
        np.asarray((2,), dtype=np.int64), np.asarray((0,), dtype=np.int64),
    )
    face_squared = tuple((x - c) ** 2 + (y - s) ** 2 for c in cosine for s in sine)
    assert Fraction(float(result.distance_lower_bounds[0])) ** 2 <= min(face_squared)
    upper_distance = float(result.residuals[0])
    assert math.isinf(upper_distance) or Fraction(upper_distance) ** 2 >= max(face_squared)
    parameter = float(result.parameters[0, 0])
    if math.isnan(parameter):
        assert int(result.status[0]) == BRepProjectionStatus.UNRESOLVED
    else:
        assert lower <= parameter <= upper

@pytest.mark.parametrize("kind", ("cylinder", "revolution"))
def test_two_equidistant_edges_are_not_unique(kind: str) -> None:
    result = _projection(kind, -1.0, 1.0).project(
        np.asarray(((-2.0, 0.0, 0.25),)), np.asarray((2,)), np.asarray((0,))
    )
    assert int(result.status[0]) != BRepProjectionStatus.UNIQUE


@pytest.mark.parametrize("kind", ("cylinder", "revolution"))
def test_interior_foot_remains_unique(kind: str) -> None:
    result = _projection(kind, 0.0, 2.0).project(
        np.asarray(((1.0, 1.0, 0.25),)), np.asarray((2,)), np.asarray((0,))
    )
    assert int(result.status[0]) == BRepProjectionStatus.UNIQUE


@pytest.mark.parametrize("kind", ("cylinder", "revolution"))
def test_full_turn_retains_interior_and_seam(kind: str) -> None:
    projection = _projection(kind, -math.pi, math.pi)
    interior = projection.project(
        np.asarray(((1.0, 1.0, 0.25),)), np.asarray((2,)), np.asarray((0,))
    )
    assert int(interior.status[0]) == BRepProjectionStatus.UNIQUE
    seam = projection.project(
        np.asarray(((-1.0, 0.0, 0.25),)), np.asarray((2,)), np.asarray((0,))
    )
    assert int(seam.status[0]) == BRepProjectionStatus.SEAM


def test_placed_point_residual_covers_exact_distance() -> None:
    placement = BRepAffinePlacement.from_matrix(
        ((Fraction(3, 5), Fraction(-4, 5), 0, Fraction(1, 3)),
         (Fraction(4, 5), Fraction(3, 5), 0, 0),
         (0, 0, 1, 0), (0, 0, 0, 1))
    )
    projection = prepare_brep_projection(_instanced_box(placement))
    point = np.asarray(((float(Fraction(17, 60)), float(Fraction(7, 20)), -0.125),))
    result = projection.project(point, np.asarray((2,)), np.asarray((0,)),
                                occurrence_paths=(("pair", "right"),))
    assert int(result.status[0]) == BRepProjectionStatus.UNIQUE
    lower = Fraction(float(result.distance_lower_bounds[0]))
    upper = Fraction(float(result.residuals[0]))
    assert lower <= Fraction(1, 8) <= upper
    squared = sum((Fraction(float(a)) - Fraction(float(b))) ** 2
                  for a, b in zip(point[0], np.asarray(result.points)[0], strict=True))
    assert upper**2 >= squared
