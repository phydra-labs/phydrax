import math
from fractions import Fraction

import numpy as np

from phydrax.geometry.brep._boolean_arrangement import _spatial_root_vertex
from phydrax.geometry.brep._intersection import (
    _cardinal_conic_angle_bounds,
    CurveSurfaceIntersectionRoot,
)
from phydrax.geometry.brep._intersection_curve import SurfaceRegion
from phydrax.geometry.brep._patches import (
    CircleCurve,
    ConicParameterRoot,
    LineCurve,
    PlanePatch,
)
from tools.meshing_qualification.scenarios.native_operation_sequences import (
    _blend_cut_sequence,
    _pi_bounds,
)


def test_cardinal_bounds_preserve_distinct_node_and_signed_gauges() -> None:
    conic = CircleCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0)
    root = ConicParameterRoot(conic, Fraction(-1), Fraction(0))
    bounds = _cardinal_conic_angle_bounds(root)
    assert bounds is not None
    lower, upper = bounds
    pi_lower, pi_upper = _pi_bounds()
    assert lower <= pi_lower < pi_upper <= upper
    assert Fraction(math.nextafter(math.nextafter(math.pi, -math.inf), -math.inf)) < lower
    shifted = _cardinal_conic_angle_bounds(root, Fraction(1, 3), -1)
    assert shifted is not None
    low, high = shifted
    assert low <= Fraction(1, 3) - pi_upper < Fraction(1, 3) - pi_lower <= high
    noncardinal = ConicParameterRoot(conic, Fraction(3, 5), Fraction(4, 5))
    assert _cardinal_conic_angle_bounds(noncardinal) is None


def test_spatial_root_vertex_retains_source_ownership() -> None:
    line = LineCurve((0, 0, 0), (3, 0, 0))
    plane = SurfaceRegion(
        PlanePatch((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        np.asarray(((-1.0, -1.0), (1.0, 1.0)), dtype=np.float64),
    )
    root = CurveSurfaceIntersectionRoot(
        line,
        plane,
        parameter_lower=np.asarray((0.3, -0.1, -0.1), dtype=np.float64),
        parameter_upper=np.asarray((0.4, 0.1, 0.1), dtype=np.float64),
    )
    original = _spatial_root_vertex((0, 1), 2, (1, 3), root)
    assert original == _spatial_root_vertex((0, 7), 2, (1, 3), root)
    assert original != _spatial_root_vertex((0, 1), 9, (1, 3), root)
    assert original != _spatial_root_vertex((1, 1), 2, (1, 3), root)
    assert original != _spatial_root_vertex((0, 1), 2, (1, 4), root)


def test_blended_corner_split_retains_two_certified_solids() -> None:
    result = _blend_cut_sequence()
    cut = result["mixed_plane_cylinder_cut"]
    assert isinstance(cut, dict)
    assert cut["regions"] == 2
