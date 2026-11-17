from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization._coordinate_enclosure import Polynomial
from phydrax.geometry._source_fidelity import _exact_surface_jacobian_bounds


@pytest.mark.meshcore
@pytest.mark.skipif(not meshcore_available(), reason="native meshcore is unavailable")
def test_zero_high_degree_tangent_matches_exact_jacobian_bounds() -> None:
    expressions: tuple[Polynomial, ...] = (
        {(1, 0): Fraction(1)},
        {(0, 1): Fraction(1)},
        {(2, 0): Fraction(0)},
    )
    triangle = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    bounds = _exact_surface_jacobian_bounds(expressions, triangle)
    exact = _exact_surface_jacobian_bounds(expressions, triangle, composed=True)
    assert bounds is not None and exact is not None
    np.testing.assert_array_equal(bounds[0], exact[0])
    np.testing.assert_array_equal(bounds[1], exact[1])
