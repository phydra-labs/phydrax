from fractions import Fraction

from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry.brep._patches import CircleCurve, ConicParameterRoot
from phydrax.geometry.brep._projection import _upper_half_profile_admitted


def test_torus_profile_rejects_long_wrapped_upper_endpoints() -> None:
    circle = CircleCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)
    radical = RadicalElement.sqrt(3) / 2
    thirty = ConicParameterRoot(circle, radical, Fraction(1, 2))
    sixty = ConicParameterRoot(circle, Fraction(1, 2), radical)
    next_thirty = ConicParameterRoot(circle, radical, Fraction(1, 2), turns=1)
    assert _upper_half_profile_admitted(thirty, sixty)
    assert not _upper_half_profile_admitted(sixty, next_thirty)
    assert not _upper_half_profile_admitted(sixty, thirty)
    next_sixty = ConicParameterRoot(circle, Fraction(1, 2), radical, turns=1)
    assert _upper_half_profile_admitted(next_thirty, next_sixty)
