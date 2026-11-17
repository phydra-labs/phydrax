#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization import CellMesh
from phydrax.geometry import (
    CommonRefinementPolicy,
    CommonRefinementStatus,
    prepare_common_refinement,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable"),
]


def _check_tolerance(
    vertices: np.ndarray, measure: float, tolerance: float, relative: float
) -> None:
    # Compare the reported slack with the exact irrational box-diameter allowance
    # by squaring nonnegative rationals; no floating-point norm is the oracle.
    widths = [
        Fraction(float(vertices[:, axis].max()))
        - Fraction(float(vertices[:, axis].min()))
        for axis in range(vertices.shape[1])
    ]
    components = [
        math.nextafter(float(width), -math.inf) if width else 0.0 for width in widths
    ]
    total = 0.0
    for component in components:
        square = max(0.0, math.nextafter(float(Fraction(component) ** 2), -math.inf))
        rounded = float(Fraction(total) + Fraction(square))
        total = math.nextafter(rounded, -math.inf) if total and square else rounded
    with localcontext() as context:
        context.prec = 200
        exact_total = Fraction(total)
        root = (Decimal(exact_total.numerator) / Decimal(exact_total.denominator)).sqrt()
        diameter = math.nextafter(float(root), -math.inf) if total else 0.0
    floor = float(Fraction(256, 2**52))
    for _ in range(vertices.shape[1]):
        floor = max(
            0.0, math.nextafter(float(Fraction(floor) * Fraction(diameter)), -math.inf)
        )
    relative_rounded = max(
        0.0, math.nextafter(float(Fraction(relative) * Fraction(measure)), -math.inf)
    )
    expected = max(
        0.0,
        math.nextafter(float(Fraction(relative_rounded) + Fraction(floor)), -math.inf),
    )
    assert tolerance == expected
    squared_diameter = sum((width * width for width in widths), Fraction(0))
    exact_relative = Fraction(relative) * Fraction(measure)
    reported = Fraction(tolerance)
    coefficient = Fraction(256, 2**52)
    exact_floor_squared = coefficient**2 * squared_diameter ** vertices.shape[1]
    assert (
        reported <= exact_relative
        or (reported - exact_relative) ** 2 <= exact_floor_squared
    )
    # Bound the rounding loss in the floor, relative product and final sum.
    allowance = reported - exact_relative
    assert allowance > 0
    assert allowance**2 >= exact_floor_squared * Fraction(1 - 2**-24) ** 2


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("relative", [0.0, 2.0**-20])
@pytest.mark.parametrize("scenario", ["identical", "gap", "double"])
def test_stored_coverage_tolerances_and_status(
    dimension: int, relative: float, scenario: str
) -> None:
    vertices = np.vstack((np.zeros((1, dimension)), np.diag(np.arange(1, dimension + 1))))
    cells = np.arange(dimension + 1, dtype=np.int32)[None, :]
    construct = CellMesh.from_triangles if dimension == 2 else CellMesh.from_tetrahedra
    source_vertices = vertices.copy()
    target_vertices = vertices.copy()
    source_cells = cells
    if scenario == "gap":
        target_vertices[:, 0] += 0.5
    elif scenario == "double":
        source_vertices = np.concatenate((vertices, vertices))
        source_cells = np.concatenate((cells, cells + len(vertices)))
    source = construct(source_vertices, source_cells)
    target = construct(target_vertices, cells)
    refinement = prepare_common_refinement(
        source, target, policy=CommonRefinementPolicy(coverage_tolerance=relative)
    )
    expected = {
        "identical": CommonRefinementStatus.SUCCESS,
        "gap": CommonRefinementStatus.COVERAGE_GAP,
        "double": CommonRefinementStatus.DOUBLE_COVERAGE,
    }
    assert refinement.status is expected[scenario]
    for points, measures, tolerances in (
        (
            source_vertices,
            refinement.source_measures,
            refinement.evidence.source_coverage_tolerances,
        ),
        (
            target_vertices,
            refinement.target_measures,
            refinement.evidence.target_coverage_tolerances,
        ),
    ):
        for measure, tolerance in zip(
            np.asarray(measures), np.asarray(tolerances), strict=True
        ):
            _check_tolerance(points, float(measure), float(tolerance), relative)
