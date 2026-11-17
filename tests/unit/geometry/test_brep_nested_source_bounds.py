from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.geometry.brep import _patches
from phydrax.geometry.brep._assembly import BRepAffinePlacement
from phydrax.geometry.brep._boolean_core import AffinePCurve, PeriodicPCurve
from phydrax.geometry.brep._patches import (
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
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


from phydrax.geometry._interval_enclosure import (
    Interval,
    interval_add as add,
    interval_cos as cos,
    interval_divide as divide,
    interval_multiply as mul,
    interval_sin as sin,
    interval_subtract as sub,
)
from phydrax.geometry._rational import fraction_down, fraction_up


def _point(value: object) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    return array, array


def _fraction(value: Fraction) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(fraction_down(value), dtype=np.float64),
        np.asarray(fraction_up(value), dtype=np.float64),
    )


def _neg(value: Interval) -> tuple[np.ndarray, np.ndarray]:
    return -np.asarray(value[1]), -np.asarray(value[0])


def _stack_interval(
    *values: Interval,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([value[0] for value in values], axis=-1),
        np.stack([value[1] for value in values], axis=-1),
    )


def _parity(actual: object, expected: object) -> None:
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float64).view(np.uint64),
        np.asarray(expected, dtype=np.float64).view(np.uint64),
    )


def _blend(
    v: Interval,
    first: Interval,
    second: Interval,
) -> Interval:
    return add(mul(sub(_point(1), v), first), mul(v, second))


def test_ruled_bounds_randomized_endpoint_parity() -> None:
    rng = np.random.default_rng(73439)
    for _ in range(32):
        vectors = rng.integers(-8, 9, size=(4, 3))
        vectors[1, 0], vectors[3, 0] = 1, -1
        rails = (LineCurve(vectors[0], vectors[1]), LineCurve(vectors[2], vectors[3]))
        surface = RuledSurface(*rails)
        u, v = np.sort(rng.uniform(-2, 2, size=2)), np.sort(rng.uniform(0, 1, size=2))
        box = np.array(((u[0], v[0]), (u[1], v[1])), dtype=np.float64)
        rail_boxes = [rail.bounding_box(*u) for rail in rails]
        value = _blend((v[0], v[1]), tuple(rail_boxes[0]), tuple(rail_boxes[1]))
        expected = np.stack(
            (
                np.maximum(value[0], np.minimum(rail_boxes[0][0], rail_boxes[1][0])),
                np.minimum(value[1], np.maximum(rail_boxes[0][1], rail_boxes[1][1])),
            )
        )
        _parity(surface.bounding_box(box), expected)
        along = _blend(
            (v[0], v[1]), rails[0].derivative_bounds(*u), rails[1].derivative_bounds(*u)
        )
        across = sub(tuple(rail_boxes[1]), tuple(rail_boxes[0]))
        _parity(surface.derivative_bounds(box), _stack_interval(along, across))


def _witness_fault(bounds: np.ndarray, index: tuple[int, ...], exact: Fraction) -> None:
    lower = Fraction.from_float(float(bounds[(0, *index)]))
    upper = Fraction.from_float(float(bounds[(1, *index)]))
    assert lower <= exact <= upper
    faulty = bounds.copy()
    faulty[(1, *index)] = fraction_down(exact) - 1.0
    assert not (
        Fraction.from_float(float(faulty[(0, *index)]))
        <= exact
        <= Fraction.from_float(float(faulty[(1, *index)]))
    )
    with pytest.raises(AssertionError):
        _parity(faulty, bounds)


def test_ruled_bounds_reject_inward_fault_by_exact_witness() -> None:
    surface = RuledSurface(
        LineCurve((0, 0, 0), (1, 0, 0)), LineCurve((0, 1, 0), (2, 0, 0))
    )
    box = np.array(((0.0, 0.25), (1.0, 0.75)), dtype=np.float64)
    _witness_fault(surface.bounding_box(box), (0,), Fraction(7, 4))
    _witness_fault(np.asarray(surface.derivative_bounds(box)), (0, 0), Fraction(7, 4))


def _circle_derivative(first: float, last: float) -> tuple[np.ndarray, np.ndarray]:
    angle = (np.asarray(first, dtype=np.float64), np.asarray(last, dtype=np.float64))
    negative = (_fraction(Fraction(-1, 3)), _point(0), _point(0))
    positive = (_point(0), _fraction(Fraction(1, 3)), _point(0))
    components = [
        add(mul(a, sin(angle)), mul(b, cos(angle)))
        for a, b in zip(negative, positive, strict=True)
    ]
    return _stack_interval(*components)


def test_circle_derivative_randomized_endpoint_parity() -> None:
    circle = CircleCurve(
        (0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0 / 3.0, radius_exact=Fraction(1, 3)
    )
    rng = np.random.default_rng(4821)
    for _ in range(64):
        samples = rng.uniform(-8, 8, size=2)
        first, last = sorted((float(samples[0]), float(samples[1])))
        _parity(circle.derivative_bounds(first, last), _circle_derivative(first, last))
    reference = np.asarray(_circle_derivative(0.0, 0.0))
    _parity(circle.derivative_bounds(0.0, 0.0), reference)
    _witness_fault(reference, (1,), Fraction(1, 3))


def _cross(axis: tuple, vector: tuple) -> tuple:
    return tuple(
        sub(
            mul(axis[(i + 1) % 3], vector[(i + 2) % 3]),
            mul(axis[(i + 2) % 3], vector[(i + 1) % 3]),
        )
        for i in range(3)
    )


def _revolution_reference(box: np.ndarray, circle: bool) -> tuple[np.ndarray, tuple]:
    axis = tuple(_point(x) for x in (0, 0, 1))
    origin = tuple(_point(0) for _ in range(3))
    angle = (box[0, 0], box[1, 0])
    c, s = cos(angle), sin(angle)
    if circle:
        pc, ps = cos((box[0, 1], box[1, 1])), sin((box[0, 1], box[1, 1]))
        center = tuple(_point(x) for x in (1, 0, 0))
        a = (_fraction(Fraction(1, 3)), _point(0), _point(0))
        b = (_point(0), _point(0), _fraction(Fraction(1, 3)))
        na = (_fraction(Fraction(-1, 3)), _point(0), _point(0))
        relative = tuple(
            add(add(center[i], mul(a[i], pc)), mul(b[i], ps)) for i in range(3)
        )
        tangent = tuple(add(mul(na[i], ps), mul(b[i], pc)) for i in range(3))

        def dot(vector: tuple) -> tuple:
            result = _point(0)
            for i in range(3):
                result = add(result, mul(axis[i], vector[i]))
            return result

        axial, axial_tangent = dot(relative), dot(tangent)
        cross, cross_tangent = _cross(axis, relative), _cross(axis, tangent)
        one_minus_c = sub(_point(1), c)
        position = tuple(
            add(
                origin[i],
                add(
                    add(mul(relative[i], c), mul(cross[i], s)),
                    mul(mul(axis[i], axial), one_minus_c),
                ),
            )
            for i in range(3)
        )
        along = tuple(
            add(
                add(mul(relative[i], _neg(s)), mul(cross[i], c)),
                mul(mul(axis[i], axial), s),
            )
            for i in range(3)
        )
        profile = tuple(
            add(
                add(mul(tangent[i], c), mul(cross_tangent[i], s)),
                mul(mul(axis[i], axial_tangent), one_minus_c),
            )
            for i in range(3)
        )
    else:
        radial = tuple(_point(x) for x in (1, 0, 0))
        tangent = tuple(_point(0) for _ in range(3))
        axial = (np.asarray(box[0, 1]), np.asarray(box[1, 1]))
        cross, cross_tangent = _cross(axis, radial), _cross(axis, tangent)
        position = tuple(
            add(
                origin[i],
                add(add(mul(axis[i], axial), mul(radial[i], c)), mul(cross[i], s)),
            )
            for i in range(3)
        )
        along = tuple(add(mul(radial[i], _neg(s)), mul(cross[i], c)) for i in range(3))
        profile = tuple(
            add(
                add(mul(axis[i], _point(1)), mul(tangent[i], c)), mul(cross_tangent[i], s)
            )
            for i in range(3)
        )
    value = _stack_interval(*position)
    derivative = _stack_interval(_stack_interval(*along), _stack_interval(*profile))
    return np.asarray(value), derivative


@pytest.mark.parametrize("circle_profile", (False, True))
def test_revolution_randomized_endpoint_parity(circle_profile: bool) -> None:
    profile = (
        CircleCurve(
            (1, 0, 0), (1, 0, 0), (0, 0, 1), 1.0 / 3.0, radius_exact=Fraction(1, 3)
        )
        if circle_profile
        else LineCurve((1, 0, 0), (0, 0, 1))
    )
    surface = RevolutionSurface(profile, (0, 0, 0), (0, 0, 1))
    rng = np.random.default_rng(963)
    for _ in range(32):
        lower = rng.uniform(-1, 1, size=2)
        box = np.stack((lower, lower + rng.uniform(0.01, 1, size=2)))
        value, derivative = _revolution_reference(box, circle_profile)
        _parity(surface.bounding_box(box), value)
        _parity(surface.derivative_bounds(box), derivative)
    box = np.array(((0.0, 0.0), (0.25, 0.25)), dtype=np.float64)
    exact = Fraction(4, 3) if circle_profile else Fraction(1)
    value, derivative = _revolution_reference(box, circle_profile)
    _parity(surface.bounding_box(box), value)
    _parity(surface.derivative_bounds(box), derivative)
    _witness_fault(value, (0,), exact)
    _witness_fault(np.asarray(derivative), (1, 0), exact)


def test_revolution_unsupported_ellipse_derivative() -> None:
    profile = EllipseCurve((1, 0, 0), (1, 0, 0), (0, 0, 1), 1, 2)
    surface = RevolutionSurface(profile, (0, 0, 0), (0, 0, 1))
    box = np.array(((0.0, 0.0), (0.25, 0.25)), dtype=np.float64)
    with pytest.raises(NotImplementedError, match="line profile"):
        surface.derivative_bounds(box)


def _rooted_cylinder_ruled_surface() -> tuple[RuledSurface, ConicRootLinePCurve]:
    conic = CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 1)
    root = ConicParameterRoot(conic, Fraction(3, 5), Fraction(4, 5))
    first = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1)
    second = CylinderPatch((2, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1)
    pcurve = ConicRootLinePCurve(root, Fraction(1, 3), Fraction(2, 3))
    return RuledSurface(
        SurfaceMappedCurve(first, pcurve), SurfaceMappedCurve(second, pcurve)
    ), pcurve


def test_rooted_cylinder_ruled_derivative_randomized_endpoint_parity() -> None:
    surface, pcurve = _rooted_cylinder_ruled_surface()
    rng = np.random.default_rng(75841)
    for _ in range(32):
        box = np.stack((rng.uniform(0.05, 0.4, size=2), rng.uniform(0.6, 0.95, size=2)))
        chart = pcurve.bounding_box(box[0, 0], box[1, 0])
        c, s = cos((chart[0, 0], chart[1, 0])), sin((chart[0, 0], chart[1, 0]))
        height = (chart[0, 1], chart[1, 1])
        rails = []
        for x in (0, 2):
            coords = []
            for origin, ca, sa, axis in zip(
                (x, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), strict=True
            ):
                radial = add(mul(_point(ca), c), mul(_point(sa), s))
                coords.append(add(add(_point(origin), radial), mul(_point(axis), height)))
            rails.append(_stack_interval(*coords))
        v = (box[0, 1], box[1, 1])
        velocity = (_point(0), _point(0), _fraction(Fraction(2, 3)))
        along = _stack_interval(*(_blend(v, item, item) for item in velocity))
        across = sub(rails[1], rails[0])
        _parity(surface.derivative_bounds(box), _stack_interval(along, across))
    _witness_fault(
        np.asarray(
            surface.derivative_bounds(
                np.array(((0.25, 0.25), (0.75, 0.75)), dtype=np.float64)
            )
        ),
        (0, 1),
        Fraction(2),
    )


@pytest.mark.parametrize("ellipse", (False, True))
def test_conic_bounds_randomized_endpoint_parity(ellipse: bool) -> None:
    curve = (
        EllipseCurve(
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            1 / 3,
            2 / 3,
            first_radius_exact=Fraction(1, 3),
            second_radius_exact=Fraction(2, 3),
        )
        if ellipse
        else CircleCurve(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), 1 / 3, radius_exact=Fraction(1, 3)
        )
    )
    rng = np.random.default_rng(98413)
    for _ in range(48):
        first = float(rng.uniform(-8, 8))
        last = first + float(rng.uniform(0, 5))
        angle = (np.asarray(first, dtype=np.float64), np.asarray(last, dtype=np.float64))
        c, s = cos(angle), sin(angle)
        radii = (Fraction(1, 3), Fraction(2 if ellipse else 1, 3))
        expected = np.array(
            [
                [
                    fraction_down(radii[0] * Fraction.from_float(float(c[0]))),
                    fraction_down(radii[1] * Fraction.from_float(float(s[0]))),
                    0,
                ],
                [
                    fraction_up(radii[0] * Fraction.from_float(float(c[1]))),
                    fraction_up(radii[1] * Fraction.from_float(float(s[1]))),
                    0,
                ],
            ],
            dtype=np.float64,
        )
        _parity(curve.bounding_box(first, last), expected)
    _witness_fault(curve.bounding_box(0.0, 0.0), (0,), Fraction(1, 3))


def _analytic_surface(kind: str) -> _patches.AbstractSurfacePatch:
    origin, first, second, axis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
    if kind == "cylinder":
        return CylinderPatch(
            origin, first, second, axis, 1 / 3, radius_exact=Fraction(1, 3)
        )
    if kind == "cone":
        return _patches.ConePatch(origin, first, second, axis, 1 / 3, 0.25)
    if kind == "sphere":
        return _patches.SpherePatch(origin, first, second, axis, 1 / 3)
    return _patches.TorusPatch(
        origin, first, second, axis, 2, 1 / 3, minor_radius_exact=Fraction(1, 3)
    )


def _analytic_reference(kind: str, box: np.ndarray) -> tuple:
    first, second, axis = _point((1, 0, 0)), _point((0, 1, 0)), _point((0, 0, 1))
    su, cu = sin((box[0, 0], box[1, 0])), cos((box[0, 0], box[1, 0]))
    radius = _fraction(Fraction(1, 3)) if kind in ("cylinder", "torus") else _point(1 / 3)
    if kind == "cylinder":
        du = add(mul(mul(_neg(radius), su), first), mul(mul(radius, cu), second))
        return _stack_interval(du, axis)
    radial = add(mul(cu, first), mul(su, second))
    tangent = add(mul(_neg(su), first), mul(cu, second))
    if kind == "cone":
        angle = _point(0.25)
        tan = divide(sin(angle), cos(angle))
        du = mul(add(radius, mul((box[0, 1], box[1, 1]), tan)), tangent)
        dv = add(axis, mul(tan, radial))
    else:
        sv, cv = sin((box[0, 1], box[1, 1])), cos((box[0, 1], box[1, 1]))
        tv = add(mul(_neg(sv), radial), mul(cv, axis))
        du = (
            mul(radius, mul(cv, tangent))
            if kind == "sphere"
            else mul(add(_point(2), mul(radius, cv)), tangent)
        )
        dv = mul(radius, tv)
    return _stack_interval(du, dv)


@pytest.mark.parametrize("kind", ("cylinder", "cone", "sphere", "torus"))
def test_analytic_surface_derivative_randomized_endpoint_parity(kind: str) -> None:
    surface = _analytic_surface(kind)
    rng = np.random.default_rng(92741)
    for _ in range(48):
        lower = rng.uniform(-1, 0, size=2)
        box = np.stack((lower, lower + rng.uniform(0.01, 1, size=2)))
        _parity(surface.derivative_bounds(box), _analytic_reference(kind, box))
    box = np.array(((-0.25, -0.25), (0.25, 0.25)), dtype=np.float64)
    exact = (
        Fraction(7, 3)
        if kind == "torus"
        else Fraction(1, 3)
        if kind == "cylinder"
        else Fraction.from_float(1 / 3)
    )
    reference = np.asarray(_analytic_reference(kind, box))
    _parity(surface.derivative_bounds(box), reference)
    _witness_fault(reference, (1, 0), exact)
