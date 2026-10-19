"""Independent closed-form oracles for certified sections and clearance."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_planar_face,
    brep_sphere,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._contact_indicatrix import contact_indicatrix
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._intersection_curve import SurfaceRegion
from phydrax.geometry.brep._patches import (
    BSplineSurfacePatch,
    CylinderPatch,
    PlanePatch,
    SpherePatch,
    TorusPatch,
)
from phydrax.geometry.brep._section_clearance import (
    brep_clearance,
    brep_section,
    BRepClearanceResult,
)


# Empirical `test_sphere_clearance_closed_form`: matched max |delta|=4.44089e-16; joint allowance fraction=2.22045e-09, margin=100%.
_SPHERE_CLEARANCE_WITNESS_RADIUS_ATOL = 1e-14
# Empirical `test_sphere_clearance_closed_form`: matched max |delta|=2.22045e-16; joint allowance fraction=2.22045e-09, margin=100%.
_SPHERE_CLEARANCE_WITNESS_DISTANCE_ATOL = 1e-14
# Empirical `test_sphere_plane_section_circle`: matched max |delta|=3.33067e-16; joint allowance fraction=3.33064e-09, margin=100%.
_SPHERE_SECTION_PLANE_RESIDUAL_ATOL = 1e-12
# Empirical `test_sphere_plane_section_circle`: matched max |delta|=1.33227e-15; joint allowance fraction=4.44074e-09, margin=100%.
_SPHERE_SECTION_RADIUS_SQUARED_ATOL = 1e-11
# Empirical `test_cylinder_plane_section_ellipse`: matched max |delta|=1.38223e-14; joint allowance fraction=4.72277e-07, margin=100%.
_CYLINDER_SECTION_PLANE_RESIDUAL_ATOL = 1e-12
# Empirical `test_cylinder_plane_section_ellipse`: matched max |delta|=1.11022e-16; joint allowance fraction=1.11011e-09, margin=100%.
_CYLINDER_SECTION_RADIUS_SQUARED_ATOL = 1e-11
# Empirical `_attains_upper`: matched max |delta|=3.55271e-15; joint allowance fraction=0.00142109, margin=99.8579%.
_CLEARANCE_UPPER_WITNESS_DISTANCE_ATOL = 1e-12
# Empirical `_attains_upper`: matched max |delta|=3.55271e-15; joint allowance fraction=0.00142109, margin=99.8579%.
_CLEARANCE_UPPER_WITNESS_DISTANCE_RTOL = 1e-12
# Empirical `test_parallel_cylinder_clearance_closed_form`: matched max |delta|=4.44089e-16; joint allowance fraction=2.22044e-09, margin=100%.
_CYLINDER_CLEARANCE_WITNESS_RADIUS_ATOL = 1e-13
# Empirical `test_parallel_cylinder_nonvertical_circle_intersection`: matched max |delta|=1.11022e-16; joint allowance fraction=1.11022e-09, margin=100%.
_CYLINDER_INTERSECTION_RADIUS_ATOL = 1e-13


def _sphere(center: list[float], radius: float = 1.0) -> SurfaceRegion:
    return SurfaceRegion(
        SpherePatch(center, [1, 0, 0], [0, 1, 0], [0, 0, 1], radius),
        [[0, -math.pi / 2], [math.tau, math.pi / 2]],
    )


@pytest.mark.parametrize(
    "distance,radius,expected",
    [(5.0, 2.0, 2.0), (1.0, 3.0, 1.0), (2.0, 2.0, 0.0), (0.0, 2.0, 1.0)],
)
def test_sphere_clearance_closed_form(
    distance: float, radius: float, expected: float
) -> None:
    first, second = _sphere([0, 0, 0]), _sphere([distance, 0, 0], radius)
    result = brep_clearance(first, second)
    assert result.status == "certified"
    assert result.distance_lower <= expected <= result.distance_upper
    witnesses = (result.witness_lower + result.witness_upper) / 2
    np.testing.assert_allclose(
        np.linalg.norm(witnesses[0]), 1, atol=_SPHERE_CLEARANCE_WITNESS_RADIUS_ATOL
    )
    np.testing.assert_allclose(
        np.linalg.norm(witnesses[1] - np.asarray([distance, 0, 0])),
        radius,
        atol=_SPHERE_CLEARANCE_WITNESS_RADIUS_ATOL,
    )
    np.testing.assert_allclose(
        np.linalg.norm(witnesses[0] - witnesses[1]),
        expected,
        atol=_SPHERE_CLEARANCE_WITNESS_DISTANCE_ATOL,
    )


def test_sphere_plane_section_circle() -> None:
    result = brep_section(
        _sphere([0, 0, 0], 2), PlanePatch([0, 0, 1], [1, 0, 0], [0, 1, 0])
    )
    assert result.status == "certified"
    curves = tuple(curve for pair in result.intersections for curve in pair.curves)
    assert len(curves) == 1
    sample = curves[0].evaluate(jnp.linspace(0, curves[0].num_charts, 33))
    points = np.asarray(sample.point)
    np.testing.assert_allclose(points[:, 2], 1, atol=_SPHERE_SECTION_PLANE_RESIDUAL_ATOL)
    np.testing.assert_allclose(
        np.sum(points[:, :2] ** 2, axis=1), 3, atol=_SPHERE_SECTION_RADIUS_SQUARED_ATOL
    )
    latitude = np.asarray(sample.first_parameters)[:, 1]
    bound = np.asarray(sample.parameter_bound)
    assert np.all(latitude - bound <= math.asin(0.5))
    assert np.all(math.asin(0.5) <= latitude + bound)
    assert curves[0].fully_certified


def test_cylinder_plane_section_ellipse() -> None:
    cylinder = SurfaceRegion(
        CylinderPatch([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 1),
        [[0, -2], [math.tau, 2]],
    )
    result = brep_section(cylinder, PlanePatch([0, 0, 0], [1, 0, 1], [0, 1, 0]))
    assert result.status == "certified"
    curves = tuple(curve for pair in result.intersections for curve in pair.curves)
    assert len(curves) == 1
    sample = curves[0].evaluate(jnp.linspace(0, curves[0].num_charts, 33))
    points = np.asarray(sample.point)
    np.testing.assert_allclose(
        points[:, 2], points[:, 0], atol=_CYLINDER_SECTION_PLANE_RESIDUAL_ATOL
    )
    np.testing.assert_allclose(
        points[:, 0] ** 2 + points[:, 1] ** 2,
        1,
        atol=_CYLINDER_SECTION_RADIUS_SQUARED_ATOL,
    )
    parameters = np.asarray(sample.first_parameters)
    height = np.cos(parameters[:, 0])
    bound = np.asarray(sample.parameter_bound)
    assert np.all(parameters[:, 1] - bound <= height)
    assert np.all(height <= parameters[:, 1] + bound)
    assert curves[0].fully_certified


def test_sphere_plane_tangent_has_no_branches() -> None:
    sphere = _sphere([0, 0, 0])
    result = brep_section(sphere, PlanePatch([0, 1, 0], [1, 0, 0], [0, 0, 1]))
    assert result.status == "certified"
    assert result.intersections[0].curves == ()
    assert result.intersections[0].points[0].kind == "tangent_point"
    assert result.intersections[0].points[0].branch_count == 0


def test_perpendicular_cylinder_indicatrix_four_branches() -> None:
    first = CylinderPatch([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 1)
    second = CylinderPatch([0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], 1)
    center = np.asarray([math.pi / 2, 0, 0, 0], dtype=np.float64)
    with coordinate_source_request((first, second)) as budget:
        result = contact_indicatrix(
            first, second, center - 1e-10, center + 1e-10, budget=budget
        )
    assert result.kind == "singular_point"
    assert result.branches == 4
    assert result.determinant_upper < 0


def test_degenerate_contact_indicatrix_is_unresolved() -> None:
    plane = PlanePatch([0, 0, 0], [1, 0, 0], [0, 1, 0])
    center = np.zeros(4, dtype=np.float64)
    with coordinate_source_request((plane,)) as budget:
        result = contact_indicatrix(plane, plane, center, center, budget=budget)
    assert result.kind == "unresolved"
    assert result.branches == -1


def test_clearance_general_faces_report_unresolved_pairs() -> None:
    plane = SurfaceRegion(PlanePatch([0, 0, 0], [1, 0, 0], [0, 1, 0]), [[-1, -1], [1, 1]])
    result = brep_clearance(plane, _sphere([0, 0, 0]), maximum_boxes=0)
    assert result.status == "unresolved"
    assert result.distance_lower == 0 and result.distance_upper == np.inf
    assert result.witness_lower.shape == (0, 3)
    assert result.unresolved_pairs == ((0, 0),)
    assert result.boxes_processed == 0


def test_saddle_indicatrix_matches_hyperbolic_closed_form() -> None:
    # Bilinear Bernstein controls represent (u, v, u*v) exactly.
    saddle = BSplineSurfacePatch(
        [[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 1]]],
        [[1, 1], [1, 1]],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        1,
        1,
    )
    plane = PlanePatch([0, 0, 0], [1, 0, 0], [0, 1, 0])
    center = np.zeros(4, dtype=np.float64)
    with coordinate_source_request((saddle, plane)) as budget:
        result = contact_indicatrix(saddle, plane, center, center, budget=budget)
    assert result.kind == "singular_point"
    assert result.branches == 4


def test_clearance_native_brep_models_retains_identity() -> None:
    coordinates = SpatialCoordinateContract.si()
    tessellation = BRepTessellationPolicy(realize=False)
    first = brep_sphere(1, coordinate_contract=coordinates, tessellation=tessellation)
    second = brep_sphere(
        2, center=(5, 0, 0), coordinate_contract=coordinates, tessellation=tessellation
    )
    identities = first.model_id, second.model_id
    result = brep_clearance(first, second)
    assert result.status == "certified"
    assert result.distance_lower <= 2 <= result.distance_upper
    assert (first.model_id, second.model_id) == identities


def _cylinder(
    center: list[float], radius: float = 1.0, heights: tuple[float, float] = (-1, 1)
) -> SurfaceRegion:
    return SurfaceRegion(
        CylinderPatch(center, [1, 0, 0], [0, 1, 0], [0, 0, 1], radius),
        [[0, heights[0]], [math.tau, heights[1]]],
    )


def _attains_upper(result: BRepClearanceResult) -> None:
    points = (result.witness_lower + result.witness_upper) / 2
    np.testing.assert_allclose(
        np.linalg.norm(points[0] - points[1]),
        result.distance_upper,
        atol=_CLEARANCE_UPPER_WITNESS_DISTANCE_ATOL,
        rtol=_CLEARANCE_UPPER_WITNESS_DISTANCE_RTOL,
    )


@pytest.mark.parametrize(
    "center,radius,heights,expected",
    [
        ([5, 0, 0], 2, (-1, 1), 2),
        ([1, 0, 0], 3, (-1, 1), 1),
        ([0, 0, 0], 2, (-1, 1), 1),
        ([2, 0, 0], 1, (-1, 1), 0),
        ([1, 0, 0], 1, (-1, 1), 0),
        ([5, 0, 0], 2, (4, 5), math.sqrt(13)),
    ],
)
def test_parallel_cylinder_clearance_closed_form(
    center: list[float], radius: float, heights: tuple[float, float], expected: float
) -> None:
    result = brep_clearance(_cylinder([0, 0, 0]), _cylinder(center, radius, heights))
    assert result.status == "certified"
    assert result.distance_lower <= expected <= result.distance_upper
    _attains_upper(result)
    points = (result.witness_lower + result.witness_upper) / 2
    np.testing.assert_allclose(
        np.linalg.norm(points[0, :2]), 1, atol=_CYLINDER_CLEARANCE_WITNESS_RADIUS_ATOL
    )
    np.testing.assert_allclose(
        np.linalg.norm(points[1, :2] - np.asarray(center[:2])),
        radius,
        atol=_CYLINDER_CLEARANCE_WITNESS_RADIUS_ATOL,
    )


def test_cylinder_plane_clearance_closed_form() -> None:
    plane = SurfaceRegion(PlanePatch([4, 0, 0], [0, 1, 0], [0, 0, 1]), [[-2, -2], [2, 2]])
    result = brep_clearance(_cylinder([0, 0, 0]), plane, maximum_boxes=128)
    # The infinite-plane distance is 4 - radius; its closest point is in this patch.
    assert result.distance_lower <= 3 <= result.distance_upper
    assert result.status == "unresolved"
    assert result.unresolved_pairs == ((0, 0),)
    _attains_upper(result)


def test_torus_sphere_clearance_closed_form() -> None:
    torus = SurfaceRegion(
        TorusPatch([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 3, 1),
        [[0, 0], [math.tau, math.tau]],
    )
    result = brep_clearance(torus, _sphere([0, 0, 0], 0.5), maximum_boxes=128)
    # Concentric torus inner radius minus the sphere radius.
    assert result.distance_lower <= 1.5 <= result.distance_upper
    _attains_upper(result)


def test_touching_general_faces_have_zero_lower_bound() -> None:
    first = SurfaceRegion(PlanePatch([0, 0, 0], [1, 0, 0], [0, 1, 0]), [[-1, -1], [1, 1]])
    second = SurfaceRegion(
        PlanePatch([0, 0, 0], [1, 0, 0], [0, 0, 1]), [[-1, -1], [1, 1]]
    )
    result = brep_clearance(first, second, maximum_boxes=16)
    assert result.distance_lower == 0
    _attains_upper(result)


def test_clearance_trim_hole_does_not_supply_witness() -> None:
    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop.polygon([[-2, -2], [2, -2], [2, 2], [-2, 2]]),
        (ProfileLoop.polygon([[-1, -1], [-1, 1], [1, 1], [1, -1]]),),
    )
    model = brep_planar_face(
        profile,
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    other = SurfaceRegion(
        PlanePatch([0, 0, 1], [1, 0, 0], [0, 1, 0]), [[-0.25, -0.25], [0.25, 0.25]]
    )
    identity = model.model_id
    result = brep_clearance(model, other, maximum_boxes=128)
    assert result.distance_lower <= 1.25 <= result.distance_upper
    points = (result.witness_lower + result.witness_upper) / 2
    assert max(abs(points[0, 0]), abs(points[0, 1])) >= 1
    assert model.model_id == identity
    _attains_upper(result)


def test_clearance_face_hierarchy_global_minimum() -> None:
    coordinates = SpatialCoordinateContract.si()
    policy = BRepTessellationPolicy(realize=False)
    first = brep_box(
        [0, 0, 0], [1, 1, 1], coordinate_contract=coordinates, tessellation=policy
    )
    second = brep_box(
        [4, 0, 0], [5, 1, 1], coordinate_contract=coordinates, tessellation=policy
    )
    before = first.model_id, second.model_id
    result = brep_clearance(first, second, maximum_boxes=128)
    assert result.distance_lower <= 3 <= result.distance_upper
    assert result.unresolved_pairs
    assert result.boxes_processed == 128
    assert (first.model_id, second.model_id) == before
    _attains_upper(result)


def test_clearance_resource_refusal_retains_all_pairs() -> None:
    first = SurfaceRegion(PlanePatch([0, 0, 0], [1, 0, 0], [0, 1, 0]), [[-1, -1], [1, 1]])
    second = SurfaceRegion(
        PlanePatch([0, 0, 3], [1, 0, 0], [0, 1, 0]), [[-1, -1], [1, 1]]
    )
    with HostBatchTicket(1000, bytes_upper=4096) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=0, maximum_memory_bytes=0
        )
        with budget.activate():
            result = brep_clearance(
                first, second, maximum_boxes=8, coordinate_budget=budget
            )
    assert result.status == "unresolved"
    assert result.unresolved_pairs == ((0, 0),)
    assert result.distance_lower == 0 and result.distance_upper == np.inf


@pytest.mark.parametrize(
    "pairs,processed", [(((0, 0), (0, 0)), 0), (((-1, 0),), 0), ((), -1)]
)
def test_clearance_result_rejects_invalid_work_metadata(
    pairs: tuple[tuple[int, int], ...], processed: int
) -> None:
    empty = np.empty((0, 3), dtype=np.float64)
    with pytest.raises(
        ValueError, match=r"BRepClearanceResult\.(unresolved_pairs|boxes_processed)"
    ):
        BRepClearanceResult(
            "unresolved",
            0,
            np.inf,
            empty,
            empty,
            unresolved_pairs=pairs,
            boxes_processed=processed,
        )


def test_parallel_cylinder_antiparallel_axis() -> None:
    other = SurfaceRegion(
        CylinderPatch([5, 0, 5], [1, 0, 0], [0, -1, 0], [0, 0, -1], 2),
        [[0, -1], [math.tau, 1]],
    )
    result = brep_clearance(_cylinder([0, 0, 0]), other)
    assert result.status == "certified"
    assert result.distance_lower <= math.sqrt(13) <= result.distance_upper
    _attains_upper(result)


def test_parallel_cylinder_nonvertical_circle_intersection() -> None:
    regions = tuple(
        SurfaceRegion(
            CylinderPatch(center, [0, 1, 0], [0, 0, 1], [1, 0, 0], 1),
            [[0, -1], [math.tau, 1]],
        )
        for center in ([0, 0, 0], [0, 1, 0])
    )
    result = brep_clearance(*regions)
    assert result.status == "certified"
    assert result.distance_lower == result.distance_upper == 0
    points = (result.witness_lower + result.witness_upper) / 2
    np.testing.assert_allclose(
        np.linalg.norm(points[0, 1:]), 1, atol=_CYLINDER_INTERSECTION_RADIUS_ATOL
    )
    np.testing.assert_allclose(
        np.linalg.norm(points[1, 1:] - [1, 0]), 1, atol=_CYLINDER_INTERSECTION_RADIUS_ATOL
    )
    _attains_upper(result)
