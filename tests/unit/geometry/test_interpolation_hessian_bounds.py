"""Independent arithmetic checks for batched interpolation certificates."""

from fractions import Fraction

import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry import _meshing_domain
from phydrax.geometry._meshing_domain import MeshingDomain, SourcePreparation
from phydrax.geometry.brep import brep_sphere
from phydrax.linalg._hermitian_spectral import fraction_sqrt_interval


def _exact_squared(lower: np.ndarray, upper: np.ndarray, box: np.ndarray) -> Fraction:
    widths = tuple(
        Fraction(float(box[1, axis])) - Fraction(float(box[0, axis])) for axis in range(2)
    )
    components = []
    for physical in range(lower.shape[0]):
        component = (
            sum(
                (
                    max(
                        abs(Fraction(float(lower[physical, i, j]))),
                        abs(Fraction(float(upper[physical, i, j]))),
                    )
                    * widths[i]
                    * widths[j]
                    for i in range(2)
                    for j in range(2)
                ),
                Fraction(0),
            )
            / 2
        )
        components.append(component)
    return sum((component * component for component in components), Fraction(0))


def _contains_squared(lower: float, upper: float, exact_squared: Fraction) -> bool:
    return (
        0 <= lower <= upper
        and Fraction(lower) ** 2 <= exact_squared <= Fraction(upper) ** 2
    )


def test_interpolation_hessian_encloses_exact_fraction_reference() -> None:
    rng = np.random.default_rng(620431)
    count = 128
    values = np.ldexp(
        rng.uniform(-1.0, 1.0, (count, 3, 2, 2)),
        rng.integers(-60, 60, (count, 3, 2, 2)),
    )
    others = np.ldexp(
        rng.uniform(-1.0, 1.0, (count, 3, 2, 2)),
        rng.integers(-60, 60, (count, 3, 2, 2)),
    )
    lower, upper = np.minimum(values, others), np.maximum(values, others)
    origins = rng.uniform(-5.0, 5.0, (count, 2))
    widths = np.ldexp(
        rng.uniform(0.5, 1.0, (count, 2)), rng.integers(-20, 20, (count, 2))
    )
    boxes = np.stack((origins, origins + widths), axis=1)
    boxes[0, 1] = boxes[0, 0]
    lower[1] = 0.0
    upper[1] = 0.0
    bounds_lower, bounds_upper = _meshing_domain._interpolation_hessian_bounds(
        lower, upper, boxes
    )
    assert bounds_lower.shape == bounds_upper.shape == (count,)
    for row in range(count):
        exact = _exact_squared(lower[row], upper[row], boxes[row])
        assert _contains_squared(
            float(bounds_lower[row]), float(bounds_upper[row]), exact
        )
        if exact:
            exact_upper = fraction_sqrt_interval(exact)[1]
            assert Fraction(float(bounds_upper[row])) >= exact_upper
            assert Fraction(float(bounds_upper[row])) <= exact_upper * Fraction(
                1_000_000_000_001, 1_000_000_000_000
            )


def test_interpolation_reference_detects_wrong_input_and_omitted_mixed_term() -> None:
    lower = np.zeros((1, 3, 2, 2), dtype=np.float64)
    upper = lower.copy()
    upper[0, 2, 0, 1] = 7.0
    boxes = np.asarray([[[0.0, 0.0], [2.0, 3.0]]], dtype=np.float64)
    exact = _exact_squared(lower[0], upper[0], boxes[0])
    assert exact == 441
    correct = _meshing_domain._interpolation_hessian_bounds(lower, upper, boxes)
    assert _contains_squared(float(correct[0][0]), float(correct[1][0]), exact)
    # Omitting the one-half factor doubles the bound and violates tightness.
    assert Fraction(float(correct[1][0])) * 2 > fraction_sqrt_interval(exact)[
        1
    ] * Fraction(1_000_000_000_001, 1_000_000_000_000)

    omitted = upper.copy()
    omitted[0, 2, 0, 1] = 0.0
    wrong_box = boxes.copy()
    wrong_box[0, 1, 1] = 1.0
    for candidate_upper, candidate_box in ((omitted, boxes), (upper, wrong_box)):
        candidate = _meshing_domain._interpolation_hessian_bounds(
            lower, candidate_upper, candidate_box
        )
        assert not _contains_squared(
            float(candidate[0][0]), float(candidate[1][0]), exact
        )


def test_interpolation_hessian_reports_natural_overflow() -> None:
    lower = np.ones((1, 3, 2, 2), dtype=np.float64)
    upper = lower.copy()
    boxes = np.asarray([[[0.0, 0.0], [1e200, 1e200]]], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        _, result = _meshing_domain._interpolation_hessian_bounds(lower, upper, boxes)
    assert np.isposinf(result[0])
    assert _exact_squared(lower[0], upper[0], boxes[0]) > 0


def test_interpolation_hessian_batch_is_row_independent() -> None:
    rng = np.random.default_rng(245)
    lower = rng.uniform(-10.0, 0.0, (24, 3, 2, 2))
    upper = rng.uniform(0.0, 10.0, (24, 3, 2, 2))
    origins = rng.uniform(-2.0, 2.0, (24, 2))
    boxes = np.stack((origins, origins + rng.uniform(0.01, 4.0, (24, 2))), axis=1)
    lower[0] = -0.0
    upper[0] = 0.0
    boxes[0] = np.asarray([[-0.0, 0.0], [0.0, -0.0]], dtype=np.float64)
    batch = _meshing_domain._interpolation_hessian_bounds(lower, upper, boxes)
    for row in range(len(boxes)):
        singleton = _meshing_domain._interpolation_hessian_bounds(
            lower[row : row + 1], upper[row : row + 1], boxes[row : row + 1]
        )
        np.testing.assert_array_equal(
            batch[0][row : row + 1].view(np.uint64), singleton[0].view(np.uint64)
        )
        np.testing.assert_array_equal(
            batch[1][row : row + 1].view(np.uint64), singleton[1].view(np.uint64)
        )
    order = rng.permutation(len(boxes))
    permuted = _meshing_domain._interpolation_hessian_bounds(
        lower[order], upper[order], boxes[order]
    )
    np.testing.assert_array_equal(permuted[0], batch[0][order])
    np.testing.assert_array_equal(permuted[1], batch[1][order])


def test_adaptive_interpolation_decides_naturally_ambiguous_bound() -> None:
    model = brep_sphere(
        1.0,
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="interpolation-exact-fallback",
    )
    domain = MeshingDomain.from_brep(model)
    surface = domain.patches[0].surface
    preparation = SourcePreparation(domain)
    chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 4), Fraction(0)),
        (Fraction(0), Fraction(1, 4)),
    )
    points = np.asarray(chart, dtype=np.float64)
    boxes = _meshing_domain._outward_surface_parameter_box(
        surface, np.min(points, axis=0)[None], np.max(points, axis=0)[None]
    )
    lower, upper = preparation.derivative_program(surface, 2).evaluate(
        boxes[:, 0], boxes[:, 1]
    )
    exact_squared = _exact_squared(lower[0], upper[0], boxes[0])
    expected = fraction_sqrt_interval(exact_squared)[1]
    assert expected * expected >= exact_squared
    _, outward_upper = _meshing_domain._interpolation_hessian_bounds(
        np.asarray(lower), np.asarray(upper), boxes
    )
    assert Fraction(float(outward_upper[0])) > expected
    (natural_result,) = domain.adaptive_interpolation_bounds(
        0, (chart,), (expected,), preparation=preparation
    )
    assert natural_result == expected

    smaller = (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 8), Fraction(0)),
        (Fraction(0), Fraction(1, 8)),
    )
    large_budget = expected + 1
    batch = domain.adaptive_interpolation_bounds(
        0, (chart, smaller), (expected, large_budget), preparation=preparation
    )
    singleton = domain.adaptive_interpolation_bounds(
        0, (smaller,), (large_budget,), preparation=preparation
    )
    assert batch == (natural_result, singleton[0])


def test_adaptive_vertex_enclosures_cover_exact_midpoint_intervals() -> None:
    model = brep_sphere(
        1.0,
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="interpolation-vertex-intervals",
    )
    domain = MeshingDomain.from_brep(model)
    surface = domain.patches[0].surface
    preparation = SourcePreparation(domain)
    rng = np.random.default_rng(6142)
    charts = tuple(
        (
            (Fraction(int(u), 32), Fraction(int(v), 32)),
            (Fraction(int(u + du), 32), Fraction(int(v), 32)),
            (Fraction(int(u), 32), Fraction(int(v + dv), 32)),
        )
        for u, v, du, dv in rng.integers((1, -8, 2, 2), (16, 8, 8, 8), (6, 4))
    )
    budgets = (Fraction(0),) * len(charts)
    batch = domain.adaptive_interpolation_bounds(
        0, charts, budgets, preparation=preparation
    )
    for row, chart in enumerate(charts):
        points = list(chart)
        points.extend(
            ((chart[i][0] + chart[j][0]) / 2, (chart[i][1] + chart[j][1]) / 2)
            for i, j in ((0, 1), (1, 2), (2, 0))
        )
        values = np.asarray(points, dtype=np.float64)
        boxes = _meshing_domain._outward_surface_parameter_box(surface, values, values)
        lower, upper = preparation.derivative_program(surface, 0).evaluate(
            boxes[:, 0], boxes[:, 1]
        )
        assert batch[row] is not None
        bound = batch[row]
        assert bound is not None
        for index, (i, j) in enumerate(((0, 1), (1, 2), (2, 0)), start=3):
            squared = Fraction(0)
            for axis in range(lower.shape[1]):
                interpolated_lower = (
                    Fraction(float(lower[i, axis])) + Fraction(float(lower[j, axis]))
                ) / 2
                interpolated_upper = (
                    Fraction(float(upper[i, axis])) + Fraction(float(upper[j, axis]))
                ) / 2
                magnitude = max(
                    abs(Fraction(float(lower[index, axis])) - interpolated_upper),
                    abs(Fraction(float(upper[index, axis])) - interpolated_lower),
                )
                squared += magnitude * magnitude
            assert bound * bound >= squared
        singleton = domain.adaptive_interpolation_bounds(
            0, (chart,), (Fraction(0),), preparation=preparation
        )
        assert singleton == (bound,)


def test_exact_chart_coordinates_have_tight_binary64_enclosures() -> None:
    from phydrax.geometry._source_interpolation import _fraction_float_interval

    rng = np.random.default_rng(8430)
    values = [
        Fraction(int(numerator), int(denominator))
        for numerator, denominator in zip(
            rng.integers(-(10**12), 10**12, 256),
            rng.integers(1, 10**12, 256),
            strict=True,
        )
    ]
    values.extend((Fraction(0), Fraction(1, 2**1100), Fraction(-1, 2**1100)))
    for value in values:
        lower, upper = _fraction_float_interval(value)
        assert Fraction(lower) <= value <= Fraction(upper)
        if lower != upper:
            assert np.nextafter(lower, np.inf) == upper


def test_chart_norm_interval_encloses_exact_upper_and_detects_bad_component() -> None:
    from phydrax.geometry._source_interpolation import _float_component_norm_interval

    rng = np.random.default_rng(315)
    for _ in range(128):
        components = rng.uniform(0.01, 100.0, 3)
        squared = sum((Fraction(float(value)) ** 2 for value in components), Fraction(0))
        exact_upper = fraction_sqrt_interval(squared)[1]
        enclosure = _float_component_norm_interval(components, components)
        assert enclosure is not None
        assert Fraction(enclosure[0]) <= exact_upper <= Fraction(enclosure[1])
        broken = components.copy()
        broken[0] = 0.0
        wrong = _float_component_norm_interval(broken, broken)
        assert wrong is not None
        assert Fraction(wrong[1]) < exact_upper
