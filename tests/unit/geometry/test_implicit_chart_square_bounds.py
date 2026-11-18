#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import numpy as np

from phydrax.geometry.implicit._source_charts import _offset_square


def test_offset_square_encloses_exact_binary64_squares() -> None:
    tiny = float(np.nextafter(0.0, 1.0))
    values = (
        -1.0,
        -0.75,
        -0.1,
        -tiny,
        -0.0,
        0.0,
        tiny,
        0.1,
        0.25,
        float(np.nextafter(0.5, np.inf)),
        0.75,
        1.0,
    )
    for left in values:
        for right in values:
            if left > right:
                continue
            low, high = _offset_square((left, right))
            a, b = Fraction.from_float(left), Fraction.from_float(right)
            expected_low = Fraction(0) if a <= 0 <= b else min(a * a, b * b)
            expected_high = max(a * a, b * b)
            assert Fraction.from_float(low) <= expected_low
            assert Fraction.from_float(high) >= expected_high
