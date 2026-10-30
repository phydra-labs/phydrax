from math import atan, pi

import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._constructors import brep_cone, brep_cylinder, brep_sphere
from phydrax.geometry.brep._normal_range import brep_normal_ranges
from phydrax.geometry.brep._patches import CylinderPatch, PlanePatch


_CONTRACT = SpatialCoordinateContract.si()


def _face_range(model: BRepModel, face: int) -> np.ndarray:
    result = brep_normal_ranges(
        model,
        (0.0, 0.0, 1.0),
        draft_angle=0.0,
        angular_tolerance=2.0e-2,
    )
    return np.asarray(result[face].angle_enclosure, dtype=np.float64)


def test_cylinder_normal_range_matches_analytic_vertical_cone() -> None:
    model = brep_cylinder(1.0, 2.0, coordinate_contract=_CONTRACT)
    side = next(
        index for index, patch in enumerate(model.patches) if isinstance(patch, CylinderPatch)
    )

    enclosure = _face_range(model, side)

    assert enclosure[0] <= pi / 2.0 <= enclosure[1]
    assert enclosure[1] - enclosure[0] < 1.0e-10


def test_cone_normal_range_contains_analytic_draft_angle() -> None:
    model = brep_cone(1.0, 2.0, 3.0, coordinate_contract=_CONTRACT)
    side = next(
        index for index, patch in enumerate(model.patches) if not isinstance(patch, PlanePatch)
    )
    expected = pi / 2.0 + atan(1.0 / 3.0)

    enclosure = _face_range(model, side)

    assert enclosure[0] <= expected <= enclosure[1]
    assert enclosure[1] - enclosure[0] <= 4.0e-2


def test_sphere_normal_range_contains_full_analytic_normal_cone() -> None:
    model = brep_sphere(1.0, coordinate_contract=_CONTRACT)
    face = next(
        index for index, patch in enumerate(model.patches) if not isinstance(patch, PlanePatch)
    )

    enclosure = _face_range(model, face)

    assert enclosure[0] <= 0.0
    assert enclosure[1] >= pi
    assert enclosure[1] <= np.nextafter(pi, np.inf)
