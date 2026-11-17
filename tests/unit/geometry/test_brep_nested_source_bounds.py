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


def test_ruled_bounds_randomized_native_host_endpoint_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _patches._ruled_blend_program() is not None
    rng = np.random.default_rng(73439)
    for _ in range(32):
        vectors = rng.integers(-8, 9, size=(4, 3))
        vectors[1, 0] = 1
        vectors[3, 0] = -1
        surface = RuledSurface(
            LineCurve(vectors[0], vectors[1]), LineCurve(vectors[2], vectors[3])
        )
        u = np.sort(rng.uniform(-2.0, 2.0, size=2))
        v = np.sort(rng.uniform(0.0, 1.0, size=2))
        box = np.array(((u[0], v[0]), (u[1], v[1])), dtype=np.float64)
        actual_box = surface.bounding_box(box)
        actual_derivative = surface.derivative_bounds(box)
        with monkeypatch.context() as context:
            context.setattr(_patches, "_ruled_blend_program", lambda: None)
            expected_box = surface.bounding_box(box)
            expected_derivative = surface.derivative_bounds(box)
        np.testing.assert_array_equal(
            actual_box.view(np.uint64), expected_box.view(np.uint64)
        )
        for actual, expected in zip(actual_derivative, expected_derivative, strict=True):
            np.testing.assert_array_equal(
                actual.view(np.uint64), expected.view(np.uint64)
            )


def test_ruled_bounds_reject_inward_fault_by_exact_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = RuledSurface(
        LineCurve((0, 0, 0), (1, 0, 0)),
        LineCurve((0, 1, 0), (2, 0, 0)),
    )
    box = np.array(((0.0, 0.25), (1.0, 0.75)), dtype=np.float64)

    def contains_witness() -> tuple[bool, bool]:
        bounds = surface.bounding_box(box)
        lower, upper = surface.derivative_bounds(box)
        # At (u, v) = (1, 3/4), x = 7/4 and dx/du = 7/4.
        exact = Fraction(7, 4)
        return (
            Fraction.from_float(bounds[0, 0])
            <= exact
            <= Fraction.from_float(bounds[1, 0]),
            Fraction.from_float(lower[0, 0]) <= exact <= Fraction.from_float(upper[0, 0]),
        )

    assert contains_witness() == (True, True)
    original = _patches._ruled_blend

    def inward(
        v: _patches.Interval, first: _patches.Interval, second: _patches.Interval, /
    ) -> _patches.Interval:
        lower, upper = original(v, first, second)
        upper_array = np.array(upper, dtype=np.float64, copy=True).reshape(3)
        upper_array[0] = 0.0
        return lower, upper_array

    monkeypatch.setattr(_patches, "_ruled_blend", inward)
    assert contains_witness() == (False, False)


def test_circle_derivative_randomized_native_host_endpoint_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _patches._circle_derivative_program(3) is not None
    circle = CircleCurve(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        1.0 / 3.0,
        radius_exact=Fraction(1, 3),
    )
    rng = np.random.default_rng(4821)
    for _ in range(64):
        samples = rng.uniform(-8.0, 8.0, size=2)
        first, last = sorted((float(samples[0]), float(samples[1])))
        actual = circle.derivative_bounds(first, last)
        with monkeypatch.context() as context:
            context.setattr(
                _patches, "_circle_derivative_program", lambda dimension: None
            )
            expected = circle.derivative_bounds(first, last)
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(
                value.view(np.uint64), reference.view(np.uint64)
            )


def test_circle_derivative_exact_witness_detects_inward_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circle = CircleCurve(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        1.0 / 3.0,
        radius_exact=Fraction(1, 3),
    )
    program = _patches._circle_derivative_program(3)
    assert program is not None

    def contains_witness() -> bool:
        lower, upper = circle.derivative_bounds(0.0, 0.0)
        return (
            Fraction.from_float(lower[1])
            <= Fraction(1, 3)
            <= Fraction.from_float(upper[1])
        )

    assert contains_witness()

    class InwardProgram:
        def evaluate(
            self, lower: NDArray[np.float64], upper: NDArray[np.float64], /
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            result_lower, result_upper = program.evaluate(lower, upper)
            result_upper[0, 1] = 0.0
            return result_lower, result_upper

    monkeypatch.setattr(
        _patches, "_circle_derivative_program", lambda dimension: InwardProgram()
    )
    assert not contains_witness()


@pytest.mark.parametrize("circle_profile", (False, True))
def test_revolution_randomized_native_host_endpoint_parity(
    circle_profile: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = (
        CircleCurve(
            (1, 0, 0), (1, 0, 0), (0, 0, 1), 1.0 / 3.0, radius_exact=Fraction(1, 3)
        )
        if circle_profile
        else LineCurve((1, 0, 0), (0, 0, 1))
    )
    surface = RevolutionSurface(profile, (0, 0, 0), (0, 0, 1))
    name = "_revolution_circle_program" if circle_profile else "_revolution_line_program"
    program = getattr(_patches, name)
    assert program(False) is not None and program(True) is not None
    rng = np.random.default_rng(963)
    for _ in range(32):
        lower = rng.uniform(-1.0, 1.0, size=2)
        upper = lower + rng.uniform(0.01, 1.0, size=2)
        box = np.stack((lower, upper))
        actual_box = surface.bounding_box(box)
        actual_derivative = surface.derivative_bounds(box)
        with monkeypatch.context() as context:
            context.setattr(_patches, name, lambda derivative: None)
            expected_box = surface.bounding_box(box)
            expected_derivative = surface.derivative_bounds(box)
        np.testing.assert_array_equal(
            actual_box.view(np.uint64), expected_box.view(np.uint64)
        )
        for actual, expected in zip(actual_derivative, expected_derivative, strict=True):
            np.testing.assert_array_equal(
                actual.view(np.uint64), expected.view(np.uint64)
            )


@pytest.mark.parametrize("circle_profile", (False, True))
def test_revolution_inward_fault_breaks_exact_witness(
    circle_profile: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = (
        CircleCurve((1, 0, 0), (1, 0, 0), (0, 0, 1), 1.0)
        if circle_profile
        else LineCurve((1, 0, 0), (0, 0, 1))
    )
    surface = RevolutionSurface(profile, (0, 0, 0), (0, 0, 1))
    box = np.zeros((2, 2), dtype=np.float64)
    expected = Fraction(2 if circle_profile else 1)
    name = "_revolution_circle_program" if circle_profile else "_revolution_line_program"
    prepare = getattr(_patches, name)

    def witnesses() -> tuple[bool, bool]:
        bounds = surface.bounding_box(box)
        lower, upper = surface.derivative_bounds(box)
        return (
            Fraction.from_float(bounds[0, 0])
            <= expected
            <= Fraction.from_float(bounds[1, 0]),
            Fraction.from_float(lower[1, 0])
            <= expected
            <= Fraction.from_float(upper[1, 0]),
        )

    assert witnesses() == (True, True)

    class InwardProgram:
        def __init__(self, derivative: bool) -> None:
            self.derivative = derivative

        def evaluate(
            self, lower: NDArray[np.float64], upper: NDArray[np.float64], /
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            program = prepare(self.derivative)
            assert program is not None
            result_lower, result_upper = program.evaluate(lower, upper)
            if self.derivative:
                result_upper[0, 1, 0] = 0.0
            else:
                result_upper[0, 0] = 0.0
            return result_lower, result_upper

    monkeypatch.setattr(_patches, name, lambda derivative: InwardProgram(derivative))
    assert witnesses() == (False, False)


def _rooted_cylinder_ruled_surface() -> RuledSurface:
    conic = CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 1)
    root = ConicParameterRoot(conic, Fraction(3, 5), Fraction(4, 5))
    first = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1)
    second = CylinderPatch((2, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1)
    pcurve = ConicRootLinePCurve(root, Fraction(1, 3), Fraction(2, 3))
    return RuledSurface(
        SurfaceMappedCurve(first, pcurve), SurfaceMappedCurve(second, pcurve)
    )


def test_rooted_cylinder_ruled_derivative_randomized_native_host_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _patches._cylinder_rail_program() is not None
    surface = _rooted_cylinder_ruled_surface()
    rng = np.random.default_rng(75841)
    for _ in range(32):
        lower = rng.uniform(0.05, 0.4, size=2)
        upper = rng.uniform(0.6, 0.95, size=2)
        box = np.stack((lower, upper))
        actual = surface.derivative_bounds(box)
        with monkeypatch.context() as context:
            context.setattr(_patches, "_cylinder_rail_program", lambda: None)
            expected = surface.derivative_bounds(box)
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(
                value.view(np.uint64), reference.view(np.uint64)
            )


def test_rooted_cylinder_ruled_derivative_detects_inward_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = _rooted_cylinder_ruled_surface()
    box = np.asarray(((0.25, 0.25), (0.75, 0.75)), dtype=np.float64)
    program = _patches._cylinder_rail_program()
    assert program is not None

    def contains_witness() -> bool:
        lower, upper = surface.derivative_bounds(box)
        return Fraction.from_float(lower[0, 1]) <= 2 <= Fraction.from_float(upper[0, 1])

    assert contains_witness()

    class InwardProgram:
        def evaluate(
            self, lower: NDArray[np.float64], upper: NDArray[np.float64], /
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            result_lower, result_upper = program.evaluate(lower, upper)
            result_lower[0, 0] = 100.0
            return result_lower, result_upper

    monkeypatch.setattr(_patches, "_cylinder_rail_program", lambda: InwardProgram())
    assert not contains_witness()


@pytest.mark.parametrize("ellipse", (False, True))
def test_conic_bounds_randomized_native_host_and_batch_parity(
    ellipse: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = (
        EllipseCurve(
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            1.0 / 3.0,
            2.0 / 3.0,
            first_radius_exact=Fraction(1, 3),
            second_radius_exact=Fraction(2, 3),
        )
        if ellipse
        else CircleCurve(
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            1.0 / 3.0,
            radius_exact=Fraction(1, 3),
        )
    )
    program = _patches._conic_trig_program()
    assert program is not None
    rng = np.random.default_rng(98413)
    starts = rng.uniform(-8.0, 8.0, size=48)
    ends = starts + rng.uniform(0.0, 5.0, size=48)
    boxes = [
        (float(first), float(last)) for first, last in zip(starts, ends, strict=True)
    ]
    single = [curve.bounding_box(first, last) for first, last in boxes]
    with monkeypatch.context() as context:
        context.setattr(_patches, "_conic_trig_program", lambda: None)
        reference = [curve.bounding_box(first, last) for first, last in boxes]
    for value, expected in zip(single, reference, strict=True):
        np.testing.assert_array_equal(value.view(np.uint64), expected.view(np.uint64))

    lower, upper = program.evaluate(starts[:, None], ends[:, None])
    rows = {box: index for index, box in enumerate(boxes)}

    class BatchedProgram:
        def evaluate(
            self, first: NDArray[np.float64], last: NDArray[np.float64], /
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            index = rows[(float(first[0, 0]), float(last[0, 0]))]
            return lower[index : index + 1], upper[index : index + 1]

    with monkeypatch.context() as context:
        context.setattr(_patches, "_conic_trig_program", lambda: BatchedProgram())
        batched = [curve.bounding_box(first, last) for first, last in boxes]
    for value, expected in zip(batched, single, strict=True):
        np.testing.assert_array_equal(value.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("ellipse", (False, True))
def test_conic_bounds_trig_inward_fault_breaks_exact_witness(
    ellipse: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = (
        EllipseCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 1, 2)
        if ellipse
        else CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 1)
    )
    program = _patches._conic_trig_program()
    assert program is not None

    def contains_witness() -> bool:
        box = curve.bounding_box(0.0, 0.0)
        return Fraction.from_float(box[0, 0]) <= 1 <= Fraction.from_float(box[1, 0])

    assert contains_witness()

    class InwardProgram:
        def evaluate(
            self, lower: NDArray[np.float64], upper: NDArray[np.float64], /
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            result_lower, result_upper = program.evaluate(lower, upper)
            result_lower[0, 0] = 0.0
            result_upper[0, 0] = 0.0
            return result_lower, result_upper

    monkeypatch.setattr(_patches, "_conic_trig_program", lambda: InwardProgram())
    assert not contains_witness()


def _analytic_surface(
    kind: _patches._AnalyticSurfaceKind,
) -> _patches.AbstractSurfacePatch:
    origin = (0, 0, 0)
    first = (1, 0, 0)
    second = (0, 1, 0)
    axis = (0, 0, 1)
    if kind == "cylinder":
        return CylinderPatch(
            origin, first, second, axis, 1.0 / 3.0, radius_exact=Fraction(1, 3)
        )
    if kind == "cone":
        return _patches.ConePatch(origin, first, second, axis, 1.0 / 3.0, 0.25)
    if kind == "sphere":
        return _patches.SpherePatch(origin, first, second, axis, 1.0 / 3.0)
    return _patches.TorusPatch(
        origin, first, second, axis, 2, 1.0 / 3.0, minor_radius_exact=Fraction(1, 3)
    )


@pytest.mark.parametrize("kind", ("cylinder", "cone", "sphere", "torus"))
def test_analytic_surface_derivative_randomized_host_parity(
    kind: _patches._AnalyticSurfaceKind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = _analytic_surface(kind)
    assert _patches._analytic_surface_program(kind) is not None
    rng = np.random.default_rng(92741)
    for _ in range(48):
        lower = rng.uniform(-1.0, 0.0, size=2)
        upper = lower + rng.uniform(0.01, 1.0, size=2)
        box = np.stack((lower, upper))
        actual = surface.derivative_bounds(box)
        with monkeypatch.context() as context:
            context.setattr(_patches, "_analytic_surface_program", lambda selected: None)
            expected = surface.derivative_bounds(box)
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(
                value.view(np.uint64), reference.view(np.uint64)
            )


@pytest.mark.parametrize("kind", ("cylinder", "cone", "sphere", "torus"))
def test_analytic_surface_derivative_inward_fault_breaks_exact_witness(
    kind: _patches._AnalyticSurfaceKind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = _analytic_surface(kind)
    program = _patches._analytic_surface_program(kind)
    assert program is not None
    box = np.asarray(((-0.25, -0.25), (0.25, 0.25)), dtype=np.float64)
    if kind == "torus":
        exact = Fraction(7, 3)
    elif kind == "cylinder":
        exact = Fraction(1, 3)
    else:
        exact = Fraction.from_float(1.0 / 3.0)

    def contains_witness() -> bool:
        lower, upper = surface.derivative_bounds(box)
        return (
            Fraction.from_float(lower[1, 0]) <= exact <= Fraction.from_float(upper[1, 0])
        )

    assert contains_witness()

    class InwardProgram:
        def evaluate(
            self, lower: NDArray[np.float64], upper: NDArray[np.float64], /
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            result_lower, result_upper = program.evaluate(lower, upper)
            result_upper[0, 1, 0] = 0.0
            return result_lower, result_upper

    monkeypatch.setattr(
        _patches, "_analytic_surface_program", lambda selected: InwardProgram()
    )
    assert not contains_witness()
