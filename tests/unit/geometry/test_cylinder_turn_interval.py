"""Consumer checks for orthogonal-cylinder intersection chart sheets."""

from __future__ import annotations

import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.brep._intersection import intersect_surface_regions
from phydrax.geometry.brep._intersection_curve import (
    _unique_nearest_interval_integer,
    IntersectionPCurve,
    interval_context,
    SurfaceRegion,
)
from phydrax.geometry.brep._patches import CylinderPatch


@pytest.mark.parametrize("swap", (False, True))
def test_cylinder_chart_evaluation_stays_on_source_sheet(
    swap: bool,
) -> None:
    radius = Fraction(1, 2)
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch(
        (0, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 0),
        float(radius),
        radius_exact=radius,
    )
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)), dtype=np.float64)
    large_region, small_region = SurfaceRegion(large, box), SurfaceRegion(small, box)
    first, second = (small_region, large_region) if swap else (large_region, small_region)
    result = intersect_surface_regions(first, second)
    assert result.complete and len(result.curves) == 2
    for curve in result.curves:
        assert curve.fully_certified
        for chart in range(curve.num_charts):
            for side, region in (("first", first), ("second", second)):
                pcurve = IntersectionPCurve(
                    curve, side, first=float(chart), last=float(chart + 1)
                )
                parameter = jnp.asarray(chart + 0.5)
                actual = pcurve.evaluate(parameter)
                point = curve.evaluate(parameter)
                np.testing.assert_allclose(
                    region.patch.evaluate(actual), point, rtol=0.0, atol=1e-9
                )
                coupled = curve.evaluate_coupled(parameter)
                expected = np.asarray(
                    coupled.first_parameters
                    if side == "first"
                    else coupled.second_parameters
                )
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9)


def test_half_turn_interval_is_refused_and_adjacent_sheets_are_unique() -> None:
    context = interval_context(80)
    half = context.mpf(1) / 2
    delta = context.mpf(1) / 10**70
    assert _unique_nearest_interval_integer(half) is None
    assert _unique_nearest_interval_integer(half + context.mpf((-delta, delta))) is None
    assert _unique_nearest_interval_integer(half - 3 * delta) == 0
    assert _unique_nearest_interval_integer(half + 3 * delta) == 1
