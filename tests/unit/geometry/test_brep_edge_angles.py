from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from phydrax._exact_kernels import float64_interval_elementary
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._interval_enclosure import interval_acos
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_extrusion,
    brep_revolution,
    PlanarProfile,
    ProfileArc,
    ProfileLine,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._edge_classification import brep_edge_classification


def _encloses(bounds: np.ndarray, angle: mp.mpf) -> bool:
    return mp.mpf(float(bounds[0])) <= angle <= mp.mpf(float(bounds[1]))


def _check_angle(bounds: np.ndarray, angle: mp.mpf) -> None:
    assert _encloses(bounds, angle)
    # Bracket the exact witness by adjacent binary64 values, then verify that
    # a single inward step on either side is rejected by the same check.
    rounded = float(angle)
    below = rounded if mp.mpf(rounded) <= angle else np.nextafter(rounded, -np.inf)
    above = rounded if mp.mpf(rounded) >= angle else np.nextafter(rounded, np.inf)
    tight = np.asarray((below, above), dtype=np.float64)
    assert _encloses(tight, angle)
    lower_fault, upper_fault = tight.copy(), tight.copy()
    lower_fault[0] = np.nextafter(lower_fault[0], np.inf)
    upper_fault[1] = np.nextafter(upper_fault[1], -np.inf)
    assert not _encloses(lower_fault, angle)
    assert not _encloses(upper_fault, angle)


def test_box_edge_dihedral_encloses_zero_cosine() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 4.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    results = brep_edge_classification(model, model.geometry)
    with mp.workdps(100):
        for result in results:
            assert result.status in ("convex", "concave")
            _check_angle(np.asarray(result.dihedral_range), mp.pi / 2)


def test_inverse_trigonometric_endpoints_are_outward_and_batch_independent() -> None:
    cosines = np.asarray((-1.0, 0.0, 1.0), dtype=np.float64)
    lower, upper = interval_acos((cosines, cosines))
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    with mp.workdps(100):
        for index, witness in enumerate((mp.pi, mp.pi / 2, mp.mpf(0))):
            bounds = np.asarray(
                (lower_array[index], upper_array[index]), dtype=np.float64
            )
            _check_angle(bounds, witness)
            single = interval_acos(
                (cosines[index : index + 1], cosines[index : index + 1])
            )
            np.testing.assert_array_equal(
                bounds.view(np.uint64), np.asarray(single).reshape(2).view(np.uint64)
            )
        zero = np.asarray((0.0,), dtype=np.float64)
        atan = float64_interval_elementary("atan", zero, zero)
        _check_angle(np.asarray((atan[0][0], atan[1][0]), dtype=np.float64), mp.mpf(0))


def test_coplanar_split_edge_encloses_zero_angle() -> None:
    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0))),
    )
    model = brep_extrusion(
        profile, (0.0, 0.0, 4.0), coordinate_contract=SpatialCoordinateContract.si()
    )
    points = np.asarray(model.geometry.vertex_points)
    edge = next(
        index
        for index, (first, last) in enumerate(model.geometry.edge_vertices)
        if all(points[v, 0] == 1.0 and points[v, 1] == 0.0 for v in (first, last))
        and abs(points[first, 2] - points[last, 2]) == 4.0
    )
    result = brep_edge_classification(model, model.geometry)[edge]
    assert result.status == "unresolved"
    with mp.workdps(100):
        _check_angle(np.asarray(result.dihedral_range), mp.mpf(0))


@pytest.mark.parametrize("top", [False, True])
def test_oblique_planar_edge_dihedral_encloses_exact_normal_angle(top: bool) -> None:
    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop.polygon(((0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0))),
    )
    model = brep_extrusion(
        profile, (1.0, 0.0, 4.0), coordinate_contract=SpatialCoordinateContract.si()
    )
    points = np.asarray(model.geometry.vertex_points)
    x, z = (1.0, 4.0) if top else (0.0, 0.0)
    edge = next(
        index
        for index, (first, last) in enumerate(model.geometry.edge_vertices)
        if all(points[v, 0] == x and points[v, 2] == z for v in (first, last))
        and abs(points[first, 1] - points[last, 1]) == 3.0
    )
    result = brep_edge_classification(model, model.geometry)[edge]
    assert result.status in ("convex", "concave")
    with mp.workdps(100):
        cosine = (1 if top else -1) / mp.sqrt(17)
        _check_angle(np.asarray(result.dihedral_range), mp.acos(cosine))


def test_revolution_seam_dihedral_encloses_exact_sphere_plane_angle() -> None:
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ProfileLoop(
            ((0.0, -5.0), (4.0, 3.0), (0.0, 3.0)),
            (ProfileArc((0.0, 0.0)), ProfileLine(), ProfileLine()),
        ),
    )
    model = brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    points = np.asarray(model.geometry.vertex_points)
    edge = next(
        index
        for index, (first, last) in enumerate(model.geometry.edge_vertices)
        if first == last and points[first, 2] == 3.0
    )
    result = brep_edge_classification(model, model.geometry)[edge]
    assert result.status in ("convex", "concave")
    with mp.workdps(100):
        _check_angle(np.asarray(result.dihedral_range), mp.acos(mp.mpf(3) / 5))
