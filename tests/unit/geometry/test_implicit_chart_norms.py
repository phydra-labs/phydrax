#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import numpy as np

from phydrax.geometry._rational import binary64_norm_upper
from phydrax.geometry.implicit._source_charts import (
    _normal_bracket_start_steps,
    _source_cell_extent_upper,
)


def _exact_squared(values: np.ndarray) -> Fraction:
    return sum(
        (Fraction.from_float(float(value)) ** 2 for value in values), Fraction(0)
    )


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
                (Fraction.from_float(float(high)) - Fraction.from_float(float(low)))
                ** 2
                for low, high in zip(domain[0], domain[1], strict=True)
            ),
            Fraction(0),
        ) / (1 << (2 * level))
        assert Fraction.from_float(upper) ** 2 >= squared
