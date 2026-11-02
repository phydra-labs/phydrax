from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from phydrax.geometry.brep._assembly import BRepAffinePlacement
from phydrax.geometry.brep._boolean_core import AffinePCurve, PeriodicPCurve
from phydrax.geometry.brep._patches import (
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    ExactLineCurve,
    ExtrusionSurface,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
)
from phydrax.geometry.brep._placed import PlacedCurve, PlacedSurface
from phydrax.geometry.brep._root_bindings import (
    _patch_box,
    _source_coefficients_are_binary,
)


def _contains(box: NDArray[np.float64], point: tuple[Fraction, ...]) -> None:
    assert all(
        Fraction.from_float(float(lo)) <= value <= Fraction.from_float(float(hi))
        for lo, value, hi in zip(box[0], point, box[1], strict=True)
    )


def test_nested_exact_line_rails_value_and_product_rule() -> None:
    q = Fraction(1, 3)
    first = LineCurve(
        (float(q), 0, 0),
        (float(q), 1, 0),
        origin_exact=(q, Fraction(0), Fraction(0)),
        direction_exact=(q, Fraction(1), Fraction(0)),
    )
    second = LineCurve((1, 0, 1), (1, 2, 0))
    ruled = RuledSurface(first, second)
    nested = ExtrusionSurface(SurfaceIsoparametricCurve(ruled, 1, 0.5), (0, 0, 1))
    singleton = np.array((0.0, 0.0), dtype=np.float64)
    _contains(_patch_box(ruled, singleton, singleton), (q, Fraction(0), Fraction(0)))
    assert not _source_coefficients_are_binary(nested)
    box = ruled.bounding_box(np.array(((0.0, 0.25), (1.0, 0.75)), dtype=np.float64))
    _contains(box, (Fraction(1, 2), Fraction(0), Fraction(1, 4)))
    lower, upper = ruled.derivative_bounds(
        np.array(((0.0, 0.25), (1.0, 0.75)), dtype=np.float64)
    )
    for u in (Fraction(0), Fraction(1)):
        for v in (Fraction(1, 4), Fraction(3, 4)):
            du = ((1 - v) * q + v, (1 - v) + 2 * v, Fraction(0))
            dv = (1 + u - q * (1 + u), u, Fraction(1))
            for axis in range(3):
                assert (
                    Fraction.from_float(float(lower[axis, 0]))
                    <= du[axis]
                    <= Fraction.from_float(float(upper[axis, 0]))
                )
                assert (
                    Fraction.from_float(float(lower[axis, 1]))
                    <= dv[axis]
                    <= Fraction.from_float(float(upper[axis, 1]))
                )


def test_mapped_and_isoparametric_fixed_exact_source() -> None:
    q = Fraction(1, 3)
    plane = PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))
    iso = SurfaceIsoparametricCurve(plane, 0, float(q), fixed_value_exact=q)
    _contains(iso.bounding_box(0.0, 1.0), (q, Fraction(1, 2), Fraction(0)))
    pcurve = LineCurve((float(q), 0), (0, 1), origin_exact=(q, Fraction(0)))
    mapped = SurfaceMappedCurve(plane, pcurve)
    _contains(mapped.bounding_box(0.0, 1.0), (q, Fraction(1, 2), Fraction(0)))
    assert not _source_coefficients_are_binary(mapped)
    assert _source_coefficients_are_binary(
        SurfaceMappedCurve(plane, LineCurve((0, 0), (0, 1)))
    )


def test_wrapper_and_root_realizations_are_not_binary() -> None:
    curve = LineCurve((0, 0), (1, 0))
    affine = AffinePCurve(
        curve, np.asarray(((Fraction(1, 3), 0), (0, 1)), dtype=np.object_), (0, 0)
    )
    cylinder = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1)
    periodic = PeriodicPCurve(curve, cylinder, (1, 0))
    circle = CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 1)
    root = ConicParameterRoot(circle, Fraction(0), Fraction(1))
    root_line = ConicRootLinePCurve(root, 0, 1)
    assert not _source_coefficients_are_binary(affine)
    assert not _source_coefficients_are_binary(periodic)
    assert not _source_coefficients_are_binary(root_line)


def test_exact_placement_preserves_singleton_and_rail_authority() -> None:
    q = Fraction(1, 3)
    affine = BRepAffinePlacement(
        tuple(
            tuple(Fraction(value) for value in row)
            for row in ((1, 0, 0, q), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        )
    )
    rotation = np.eye(3, dtype=np.float64)
    translation = np.array((float(q), 0, 0), dtype=np.float64)
    plane = PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))
    surface = PlacedSurface(plane, rotation, translation, affine=affine)
    point = np.array((0.0, 0.0), dtype=np.float64)
    _contains(_patch_box(surface, point, point), (q, Fraction(0), Fraction(0)))
    curve = PlacedCurve(
        LineCurve((0, 0, 0), (0, 1, 0)), rotation, translation, affine=affine
    )
    _contains(curve.bounding_box(0.0, 1.0), (q, Fraction(1, 2), Fraction(0)))
    assert not _source_coefficients_are_binary(surface)
    assert not _source_coefficients_are_binary(curve)


def test_revolution_mapped_isoline_derivative_uses_nondegenerate_chart() -> None:
    profile = ExactLineCurve((1, 0, 0), (1, 0, 1))
    surface = RevolutionSurface(profile, (0, 0, 0), (0, 0, 1))
    pcurve = LineCurve((0, 0.5), (1, 0))
    curve = SurfaceMappedCurve(surface, pcurve)
    lower, upper = curve.derivative_bounds(0.0, 1.0)
    assert np.all(lower <= np.array((0.0, 1.0, 0.0), dtype=np.float64))
    assert np.all(upper >= np.array((0.0, 1.0, 0.0), dtype=np.float64))
