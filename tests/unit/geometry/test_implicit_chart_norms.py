#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import numpy as np

from phydrax.geometry._rational import binary64_norm_upper
from phydrax.geometry.implicit._source_charts import (
    _interval_vector_norm_bounds,
    _normal_bracket_start_steps,
    _source_cell_extent_upper,
)


def _exact_squared(values: np.ndarray) -> Fraction:
    return sum((Fraction.from_float(float(value)) ** 2 for value in values), Fraction(0))


def test_normal_bracket_spacing_uses_certified_point_norm() -> None:
    below_two = float(np.nextafter(2.0, 0.0))
    points = np.asarray(
        ((below_two, 0.0, 0.0), (1.9, 0.125, 0.0), (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    steps = _normal_bracket_start_steps(points)
    upper = binary64_norm_upper(points)
    for point, bound, step in zip(points, upper, steps, strict=True):
        assert Fraction.from_float(float(bound)) ** 2 >= _exact_squared(point)
        assert step == np.spacing(max(1.0, float(bound)))
    # The represented point is strictly inside the radius-two sphere, but
    # its directed norm enclosure reaches the next binade.
    assert _exact_squared(points[0]) < 4
    assert upper[0] >= 2.0
    assert steps[0] == np.spacing(np.float64(2.0))
    assert np.spacing(np.float64(below_two)) < steps[0]
    np.testing.assert_array_equal(
        steps, np.concatenate([_normal_bracket_start_steps(row[None]) for row in points])
    )


def test_source_cell_extent_encloses_exact_endpoint_difference() -> None:
    tiny = float(np.nextafter(0.0, 1.0))
    domains = (
        (np.asarray(((-1.0, -0.25, 0.0), (2.0**54, 0.5, 1.0))), 3),
        (np.asarray(((0.0, 0.0, 0.0), (tiny, tiny, tiny))), 1),
        (np.asarray(((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))), 7),
    )
    for domain, level in domains:
        upper = _source_cell_extent_upper(domain, level)
        squared = sum(
            (
                (Fraction.from_float(float(high)) - Fraction.from_float(float(low))) ** 2
                for low, high in zip(domain[0], domain[1], strict=True)
            ),
            Fraction(0),
        ) / (1 << (2 * level))
        assert Fraction.from_float(upper) ** 2 >= squared


def test_interval_vector_norm_refuses_inward_square_rounding() -> None:
    components = tuple(
        float.fromhex(value)
        for value in (
            "0x1.6bfbbf7d8ed35p+0",
            "0x1.5030474f27fcep-1",
            "0x1.0b91f0133360ap-2",
        )
    )
    exact_squared = sum(
        (Fraction.from_float(value) ** 2 for value in components), Fraction(0)
    )
    old_upper = float.fromhex("0x1.96772b896d262p+0")
    assert Fraction.from_float(old_upper) ** 2 < exact_squared
    lower, upper = _interval_vector_norm_bounds(
        tuple((value, value) for value in components)
    )
    assert Fraction.from_float(lower) ** 2 <= exact_squared
    assert Fraction.from_float(upper) ** 2 >= exact_squared


def test_interval_vector_norm_encloses_box_extrema() -> None:
    intervals = ((-3.0, -2.0), (-1.0, 1.0), (4.0, 5.0))
    lower, upper = _interval_vector_norm_bounds(intervals)
    assert Fraction.from_float(lower) ** 2 <= 20
    assert Fraction.from_float(upper) ** 2 >= 35
