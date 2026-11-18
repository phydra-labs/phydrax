from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from phydrax._polynomial._cubature import cubature_rule_data
from phydrax.discretization._coordinate_enclosure import Expression
from phydrax.geometry._common_root_volume_overlap import (
    _chart_image_box_disjoint,
    CommonRootOverlapError,
    CurvedOverlapIntegrationPolicy,
    prepare_common_root_volume_refinement,
)
from phydrax.geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
)
from tests._support.volume_material_map_fixtures import make_curved_material_refinement


def _nearest(value: Fraction) -> float:
    return float(value)


def _down(value: float) -> float:
    return max(0.0, float(np.nextafter(value, -np.inf)))


def _reference_diameter(bounds: np.ndarray) -> float:
    components = []
    for lower, upper in zip(bounds[0], bounds[1], strict=True):
        difference = abs(
            Fraction.from_float(float(upper)) - Fraction.from_float(float(lower))
        )
        components.append(_down(_nearest(difference)) if difference else 0.0)
    total = 0.0
    for component in components:
        square = (
            _down(_nearest(Fraction.from_float(component) ** 2)) if component else 0.0
        )
        rounded = _nearest(Fraction.from_float(total) + Fraction.from_float(square))
        total = _down(rounded) if total and square else rounded
    with localcontext() as context:
        context.prec = 200
        fraction = Fraction.from_float(total)
        root = (Decimal(fraction.numerator) / Decimal(fraction.denominator)).sqrt()
        return _down(float(root)) if total else 0.0


def _reference_tolerance(
    bounds: np.ndarray,
    measure: float,
    cell_error: float,
    overlap_errors: list[float],
    tolerance: float,
) -> float:
    diameter = _reference_diameter(bounds)
    squared_length = sum(
        (Fraction.from_float(float(upper)) - Fraction.from_float(float(lower))) ** 2
        for lower, upper in zip(bounds[0], bounds[1], strict=True)
    )
    slack = 256.0 * np.finfo(np.float64).eps
    for _ in range(3):
        slack = _down(
            _nearest(Fraction.from_float(slack) * Fraction.from_float(diameter))
        )
    assert Fraction.from_float(slack) ** 2 <= (
        Fraction.from_float(256.0 * np.finfo(np.float64).eps) ** 2 * squared_length**3
    )
    overlap_error = 0.0
    for error in overlap_errors:
        overlap_error = _down(
            _nearest(Fraction.from_float(overlap_error) + Fraction.from_float(error))
        )
    value = _down(_nearest(Fraction.from_float(tolerance) * Fraction.from_float(measure)))
    for term in (cell_error, overlap_error, slack):
        value = _down(_nearest(Fraction.from_float(value) + Fraction.from_float(term)))
    return value


@pytest.mark.parametrize("gap", [False, True])
def test_curved_overlap_stored_tolerances_and_coverage_status(gap: bool) -> None:
    root, _, source_geometry, target, target_geometry, _, witness = (
        make_curved_material_refinement(gap=gap, exact_chart_arguments=True)
    )
    assert witness is not None
    policy = CommonRefinementPolicy(
        coverage=CommonRefinementCoverage.TARGET
        if gap
        else CommonRefinementCoverage.COMPLETE,
        coverage_tolerance=0.0,
    )
    result, integration = prepare_common_root_volume_refinement(
        root,
        source_geometry,
        target,
        target_geometry,
        witness,
        policy=policy,
        integration_policy=CurvedOverlapIntegrationPolicy(),
    )
    assert result.status is CommonRefinementStatus.SUCCESS
    for raw_bounds, raw_errors, raw_measures, raw_cells, raw_stored in (
        (
            integration.source_coordinate_bounds,
            integration.source_errors,
            result.source_measures,
            result.source_cells,
            result.evidence.source_coverage_tolerances,
        ),
        (
            integration.target_coordinate_bounds,
            integration.target_errors,
            result.target_measures,
            result.target_cells,
            result.evidence.target_coverage_tolerances,
        ),
    ):
        bounds = np.asarray(raw_bounds)
        errors = np.asarray(raw_errors)[:, 0]
        measures = np.asarray(raw_measures)
        cells = np.asarray(raw_cells)
        overlap_errors = np.asarray(integration.overlap_errors)[:, 0]
        stored = np.asarray(raw_stored)
        expected = [
            _reference_tolerance(
                bounds[row],
                float(measures[row]),
                float(errors[row]),
                [
                    float(error)
                    for cell, error in zip(cells, overlap_errors, strict=True)
                    if cell == row
                ],
                policy.coverage_tolerance,
            )
            for row in range(len(measures))
        ]
        np.testing.assert_array_equal(stored, expected)
    if gap:
        with pytest.raises(CommonRootOverlapError) as failure:
            prepare_common_root_volume_refinement(
                root,
                source_geometry,
                target,
                target_geometry,
                witness,
                policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.COMPLETE),
                integration_policy=CurvedOverlapIntegrationPolicy(),
            )
        assert failure.value.status is CommonRefinementStatus.COVERAGE_GAP


def test_chart_box_scale_accepts_unrepresentable_exact_edge_lengths() -> None:
    huge = Fraction(1 << 1100)
    zero = Fraction(0)
    vertices = (
        (zero, zero, zero),
        (huge, zero, zero),
        (zero, huge, zero),
        (zero, zero, huge),
    )
    chart: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    points = np.asarray(cubature_rule_data("tetrahedron", 8).points, dtype=np.float64)
    assert not _chart_image_box_disjoint(((0.0, 1.0),) * 3, chart, vertices, points)


@pytest.mark.parametrize(
    ("scale", "outside"),
    [
        (Fraction(1, 8), True),
        (Fraction(2**55 - 1, 2**55), True),
        (Fraction(2), False),
    ],
)
def test_chart_box_subdivision_uses_exact_squared_edge_threshold(
    scale: Fraction, outside: bool
) -> None:
    zero = Fraction(0)
    vertices = (
        (zero, zero, zero),
        (scale, zero, zero),
        (zero, scale, zero),
        (zero, zero, scale),
    )
    # The coarse Bernstein enclosure contains zero, but after bisection the
    # exact polynomial (x - 1/2)^2 + 1/16 is strictly positive.
    coordinate: Expression = {
        (0, 0, 0): Fraction(5, 16),
        (1, 0, 0): Fraction(-1),
        (2, 0, 0): Fraction(1),
    }
    points = np.asarray(cubature_rule_data("tetrahedron", 8).points, dtype=np.float64)
    assert (
        _chart_image_box_disjoint(
            ((-0.125, 0.0),) * 3, (coordinate,) * 3, vertices, points
        )
        is outside
    )
