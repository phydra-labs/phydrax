from math import atan, pi

import jax.numpy as jnp
import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._constructors import brep_cone, brep_cylinder, brep_sphere
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._normal_range import brep_normal_ranges, BRepNormalRange


_CONTRACT = SpatialCoordinateContract.si()
_PULL = (0.0, 0.0, 1.0)


def _ranges(model: BRepModel) -> tuple[BRepNormalRange, ...]:
    return brep_normal_ranges(model, _PULL, draft_angle=0.0, angular_tolerance=2.0e-2)


def _face_at_height(model: BRepModel, height: float) -> int:
    """Select a face by an evaluated interior point, independent of carrier type."""
    for face, patch in enumerate(model.patches):
        bounds = np.asarray(model.parameter_bounds, dtype=np.float64)[face]
        point = np.asarray(patch.evaluate(jnp.asarray(0.5 * (bounds[0] + bounds[1]))))
        if abs(point[2] - height) <= 1.0e-12:
            return face
    raise AssertionError(f"No face has an interior point at height {height}.")


def test_cylinder_normal_ranges_match_analytic_side_and_cap_cones() -> None:
    model = brep_cylinder(1.0, 2.0, coordinate_contract=_CONTRACT)
    ranges = _ranges(model)
    side = ranges[_face_at_height(model, 1.0)]
    bottom = ranges[_face_at_height(model, 0.0)]
    top = ranges[_face_at_height(model, 2.0)]
    side_enclosure = np.asarray(side.angle_enclosure)
    bottom_enclosure = np.asarray(bottom.angle_enclosure)
    top_enclosure = np.asarray(top.angle_enclosure)

    # Outward side normals are horizontal; disc caps point along -z and +z,
    # including at the profile point where each cap meets the revolution axis.
    assert side.status == "vertical"
    assert side_enclosure[0] <= pi / 2.0 <= side_enclosure[1]
    assert side_enclosure[1] - side_enclosure[0] < 1.0e-10
    assert bottom.status == "undercut"
    assert bottom_enclosure[0] <= pi <= bottom_enclosure[1]
    assert bottom_enclosure[1] - bottom_enclosure[0] < 1.0e-6
    assert top.status == "positive"
    assert top_enclosure[0] <= 0.0 <= top_enclosure[1]
    assert top_enclosure[1] - top_enclosure[0] < 1.0e-6
    for face in (side, bottom, top):
        assert np.asarray(face.unresolved_cells).shape == (0, 2, 2)
        assert np.asarray(face.singular_points).shape == (0, 2, 3)


def test_cone_normal_range_contains_analytic_draft_angle() -> None:
    model = brep_cone(1.0, 2.0, 3.0, coordinate_contract=_CONTRACT)
    side = _ranges(model)[_face_at_height(model, 1.5)]
    # Radius grows by 1 over height 3: the outward normal is (3 e_r - e_z) / sqrt(10).
    expected = pi / 2.0 + atan(1.0 / 3.0)
    enclosure = np.asarray(side.angle_enclosure)

    assert side.status == "undercut"
    assert enclosure[0] <= expected <= enclosure[1]
    assert enclosure[1] - enclosure[0] <= 4.0e-2


def test_sphere_normal_range_contains_full_analytic_normal_cone() -> None:
    model = brep_sphere(1.0, coordinate_contract=_CONTRACT)
    (face,) = _ranges(model)
    enclosure = np.asarray(face.angle_enclosure)
    singular = np.asarray(face.singular_points)
    poles = np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)), dtype=np.float64)

    assert enclosure[0] <= 0.0
    assert enclosure[1] >= pi
    assert enclosure[1] <= np.nextafter(pi, np.inf)
    # The authored latitude range reaches the poles, where the profile meets the
    # axis; each reported singular region must enclose one of them.
    assert face.status == "unresolved"
    assert singular.shape[0] >= 1
    for lower, upper in singular:
        assert any(np.all(lower <= pole) and np.all(pole <= upper) for pole in poles)
        assert np.all(upper - lower <= 0.25)
