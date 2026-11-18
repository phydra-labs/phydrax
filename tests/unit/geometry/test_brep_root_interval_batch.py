from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry._interval_enclosure import (
    Interval,
    interval_add,
    interval_cos,
    interval_multiply,
    interval_sin,
)
from phydrax.geometry.brep._intersection_curve import (
    _PreparedPairValue,
    SurfacePairSystem,
)
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    CylinderPatch,
    PlanePatch,
    TorusPatch,
)


def _exact_interval(value: Fraction) -> tuple[Fraction, Fraction]:
    return value, value


def _fraction_interval(
    interval: Interval,
) -> tuple[Fraction, Fraction]:
    return Fraction.from_float(float(interval[0])), Fraction.from_float(
        float(interval[1])
    )


def _add(
    a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    return a[0] + b[0], a[1] + b[1]


def _multiply(
    a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    products = [x * y for x in a for y in b]
    return min(products), max(products)


def _rounded_box(intervals: list[tuple[Fraction, Fraction]]) -> np.ndarray:
    result = np.array(
        [[float(bounds[side]) for bounds in intervals] for side in range(2)],
        dtype=np.float64,
    )
    for side in range(2):
        for axis, bounds in enumerate(intervals):
            rounded = Fraction.from_float(float(result[side, axis]))
            if (side == 0 and rounded > bounds[side]) or (
                side == 1 and rounded < bounds[side]
            ):
                result[side, axis] = np.nextafter(
                    result[side, axis], -np.inf if side == 0 else np.inf
                )
    return result


def _trig(
    box: np.ndarray, axis: int
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    parameter = (box[0, axis], box[1, axis])
    return _fraction_interval(interval_cos(parameter)), _fraction_interval(
        interval_sin(parameter)
    )


def _torus_reference(box: np.ndarray) -> np.ndarray:
    cosine_u, sine_u = _trig(box, 0)
    cosine_v, sine_v = _trig(box, 1)
    major = _exact_interval(Fraction(7, 3))
    minor = _exact_interval(Fraction(2, 7))
    ring = _add(major, _multiply(minor, cosine_v))
    zero, one = _exact_interval(Fraction(0)), _exact_interval(Fraction(1))
    intervals = []
    for first, second, axis in ((one, zero, zero), (zero, one, zero), (zero, zero, one)):
        radial = _add(_multiply(cosine_u, first), _multiply(sine_u, second))
        height = _multiply(minor, _multiply(sine_v, axis))
        intervals.append(_add(_add(zero, _multiply(ring, radial)), height))
    return _rounded_box(intervals)


def _cylinder_reference(box: np.ndarray) -> np.ndarray:
    cosine, sine = _trig(box, 0)
    height = (
        Fraction.from_float(float(box[0, 1])),
        Fraction.from_float(float(box[1, 1])),
    )
    zero = _exact_interval(Fraction(0))
    radius = _exact_interval(Fraction(1, 3))
    one = _exact_interval(Fraction(1))
    intervals = []
    for first, second, axis in (
        (radius, zero, zero),
        (zero, radius, zero),
        (zero, zero, one),
    ):
        radial = _add(_multiply(first, cosine), _multiply(second, sine))
        intervals.append(_add(_add(zero, radial), _multiply(axis, height)))
    return _rounded_box(intervals)


def _plane_reference(box: np.ndarray) -> np.ndarray:
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    first = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    second = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = (box[0, :1], box[1, :1])
    v = (box[0, 1:], box[1, 1:])
    result = interval_add(
        interval_add((origin, origin), interval_multiply(u, (first, first))),
        interval_multiply(v, (second, second)),
    )
    return np.stack(result)


def _patches_and_references() -> tuple[
    tuple[AbstractSurfacePatch, Callable[[np.ndarray], np.ndarray]], ...
]:
    return (
        (PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)), _plane_reference),
        (
            CylinderPatch(
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                float(Fraction(1, 3)),
                radius_exact=Fraction(1, 3),
            ),
            _cylinder_reference,
        ),
        (
            TorusPatch(
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                float(Fraction(7, 3)),
                float(Fraction(2, 7)),
                major_radius_exact=Fraction(7, 3),
                minor_radius_exact=Fraction(2, 7),
            ),
            _torus_reference,
        ),
    )


def test_randomized_patch_boxes_match_independent_references_in_mixed_order() -> None:
    rng = np.random.default_rng(8729)
    lower = rng.uniform(-4.0, 4.0, size=(32, 2))
    boxes = np.stack((lower, lower + rng.uniform(0.0, 0.75, size=(32, 2))), axis=1)
    cases = [
        (patch, reference, box)
        for patch, reference in _patches_and_references()
        for box in boxes
    ]
    rng.shuffle(cases)
    expected = np.stack([reference(box) for _, reference, box in cases])

    def evaluate(
        case: tuple[AbstractSurfacePatch, Callable[[np.ndarray], np.ndarray], np.ndarray],
    ) -> np.ndarray:
        patch, _, box = case
        return patch.bounding_box(box)

    sequential = np.stack([evaluate(case) for case in cases])
    with ThreadPoolExecutor(max_workers=3) as pool:
        parallel = np.stack(list(pool.map(evaluate, reversed(cases))))[::-1]
    np.testing.assert_array_equal(sequential.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(parallel.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize(
    "index,witness", [(0, Fraction(0)), (1, Fraction(1, 3)), (2, Fraction(55, 21))]
)
def test_inward_reference_endpoint_is_rejected(index: int, witness: Fraction) -> None:
    patch, reference = _patches_and_references()[index]
    box = np.array([[0.0, 0.0], [1e-10, 1e-10]], dtype=np.float64)
    expected = reference(box)
    actual = patch.bounding_box(box)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert Fraction.from_float(float(expected[0, 0])) <= witness
    assert Fraction.from_float(float(expected[1, 0])) >= witness

    inward = expected.copy()
    inward[0, 0] = np.nextafter(float(witness), np.inf)
    assert Fraction.from_float(float(inward[0, 0])) > witness
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(actual.view(np.uint64), inward.view(np.uint64))


def test_surface_pair_batch_matches_per_item_enclosures() -> None:
    torus = _patches_and_references()[2][0]
    plane = _patches_and_references()[0][0]
    prepared = _PreparedPairValue(SurfacePairSystem(torus, plane))
    lower = np.array(
        [[0.0, 0.0, 0.0, 0.0], [-1.2, 0.3, -0.5, 0.2], [2.0, -2.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    upper = lower + np.array(
        [[1e-10, 1e-10, 0.0, 0.0], [0.4, 0.6, 0.3, 0.7], [0.2, 0.9, 0.5, 0.8]],
        dtype=np.float64,
    )
    batch = prepared.evaluate(lower, upper)
    singles = [prepared.evaluate(lower[i : i + 1], upper[i : i + 1]) for i in range(3)]
    for side in range(2):
        expected = np.concatenate([result[side] for result in singles])
        np.testing.assert_array_equal(
            batch[side].view(np.uint64), expected.view(np.uint64)
        )


def test_surface_pair_rejects_inward_torus_endpoint_fault() -> None:
    torus = _patches_and_references()[2][0]
    plane = _patches_and_references()[0][0]
    prepared = _PreparedPairValue(SurfacePairSystem(torus, plane))
    lower = np.zeros((2, 4), dtype=np.float64)
    upper = lower.copy()
    upper[:, :2] = 1e-10
    witness = Fraction(55, 21)

    def contains_witness(low: np.ndarray, high: np.ndarray) -> None:
        for index in range(2):
            assert Fraction.from_float(float(low[index, 0])) <= witness
            assert Fraction.from_float(float(high[index, 0])) >= witness

    low, high = prepared.evaluate(lower, upper)
    contains_witness(low, high)
    inward = low.copy()
    inward[:, 0] = np.nextafter(float(witness), np.inf)
    with pytest.raises(AssertionError):
        contains_witness(inward, high)
