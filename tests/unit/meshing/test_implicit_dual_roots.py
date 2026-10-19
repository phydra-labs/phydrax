"""Dual-root coverage, unresolved evidence and native endpoint enclosures."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import phydrax as phx
from phydrax._meshcore import (
    MeshcoreStatus,
    NativeExecutionScope,
    restricted_centers_3d,
    restricted_dual_3d,
    restricted_rays_3d,
)
from phydrax.meshing._implicit_volume import (
    ImplicitDualIntervalClass,
    isolate_implicit_dual_roots,
    prepare_implicit_restricted_queries,
    prepare_implicit_volume_domain,
)
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.units import METER
from tests._support.implicit_meshing import (
    assert_box_coverage,
    assert_terminal_cover,
    execution_scope as execution_scope,
    IMPLICIT_DOMAIN,
    implicit_query,
    implicit_specification,
    NATIVE,
)


# Unchanged empirical binary64 allowances from the right-tetrahedron center,
# Fraction(33, 10) center enclosure and unit-box hull-exit reference fixtures.
CIRCUMCENTER_COMPARISON_ATOL = 1.0e-15
CIRCUMCENTER_ENCLOSURE_WIDTH_BOUND = 1.0e-12
HULL_EXIT_ROUNDOFF_ATOL = 1.0e-12


def test_dual_interval_isolation_proves_both_sphere_crossings() -> None:
    radius = 0.371
    query = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), radius).compile(coordinate_unit=METER)
    )
    endpoints = np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64)
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.01,
        maximum_depth=14,
        maximum_geometry_queries=3000,
    )
    assert_terminal_cover(work)
    roots = work.parameter_intervals[
        work.classes == int(ImplicitDualIntervalClass.UNIQUE_ROOT)
    ]
    expected = np.asarray(((1.0 - radius) / 2.0, (1.0 + radius) / 2.0))
    assert roots.shape == (2, 2)
    assert np.all(roots[:, 0] < expected)
    assert np.all(expected < roots[:, 1])
    assert np.all(2.0 * (roots[:, 1] - roots[:, 0]) <= 0.01)
    assert not np.any(work.classes == int(ImplicitDualIntervalClass.UNRESOLVED))
    assert not work.query_budget_exhausted


def test_tangential_dual_zero_is_pending_not_certified_empty() -> None:
    radius = 0.371
    query = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), radius).compile(coordinate_unit=METER)
    )
    endpoints = np.asarray((((-1.0, radius, 0.0), (1.0, radius, 0.0)),), dtype=np.float64)
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.01,
        maximum_depth=14,
        maximum_geometry_queries=3000,
    )
    assert_terminal_cover(work)
    touches = (work.parameter_intervals[:, 0] <= 0.5) & (
        work.parameter_intervals[:, 1] >= 0.5
    )
    assert np.all(work.classes[touches] == int(ImplicitDualIntervalClass.UNRESOLVED))
    assert not np.any(work.classes == int(ImplicitDualIntervalClass.UNIQUE_ROOT))


def test_dual_query_exhaustion_preserves_full_parameter_cover() -> None:
    query = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER)
    )
    endpoints = np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64)
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.01,
        maximum_depth=14,
        maximum_geometry_queries=3,
    )
    assert_terminal_cover(work)
    assert work.query_budget_exhausted
    assert work.geometry_queries == 3
    assert np.all(work.classes == int(ImplicitDualIntervalClass.UNRESOLVED))
    assert np.all(np.isneginf(work.value_lower))
    assert np.all(np.isposinf(work.value_upper))


def test_dual_interval_isolation_proves_four_torus_crossings() -> None:
    major = 0.613
    minor = 0.087
    query = implicit_query(
        phx.geometry.Torus((0.0, 0.0, 0.0), major - minor, major + minor).compile(
            coordinate_unit=METER
        )
    )
    endpoints = np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64)
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.01,
        maximum_depth=14,
        maximum_geometry_queries=3000,
    )
    assert_terminal_cover(work)
    roots = work.parameter_intervals[
        work.classes == int(ImplicitDualIntervalClass.UNIQUE_ROOT)
    ]
    expected = 0.5 * (
        1.0 + np.asarray((-major - minor, -major + minor, major - minor, major + minor))
    )
    assert roots.shape == (4, 2)
    assert np.all(roots[:, 0] < expected)
    assert np.all(expected < roots[:, 1])
    assert not np.any(work.classes == int(ImplicitDualIntervalClass.UNRESOLVED))


def test_endpoint_uncertainty_cannot_be_replaced_with_numerical_line_roots() -> None:
    query = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER)
    )
    endpoints = np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64)
    bounds = np.stack((endpoints, endpoints), axis=2)
    # The tube includes the nominal two-crossing line and lines entirely
    # outside the sphere, so a uniform UNIQUE_ROOT conclusion is not possible.
    bounds[:, :, 1, 1] = 0.8
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        endpoint_bounds=bounds,
        spatial_tolerance=0.01,
        maximum_depth=8,
        maximum_geometry_queries=6000,
    )
    assert_terminal_cover(work)
    assert np.any(work.classes == int(ImplicitDualIntervalClass.UNRESOLVED))
    assert not np.any(work.classes == int(ImplicitDualIntervalClass.UNIQUE_ROOT))


@NATIVE
def test_native_dual_adjacency_and_exact_circumcenter_enclosures() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int32)
    bounds, status = restricted_centers_3d(points, cells)
    # Independent rational centers of these two right tetrahedra.
    expected = np.asarray(((0.5, 0.5, 0.5), (0.5, 0.5, -0.5)), dtype=np.float64)
    assert np.all(status == int(MeshcoreStatus.OK))
    assert np.all(bounds[:, 0] <= expected)
    assert np.all(expected <= bounds[:, 1])
    assert np.max(bounds[:, 1] - bounds[:, 0]) < CIRCUMCENTER_ENCLOSURE_WIDTH_BOUND
    dual = restricted_dual_3d(
        points,
        cells,
        2.0 * IMPLICIT_DOMAIN,
        8,
        100,
    )
    facets = dual.facets
    incident = dual.incident
    endpoints = dual.endpoints
    kinds = dual.kinds
    dual_status = dual.item_status
    counters = dual.counters
    finite = incident[:, 1] >= 0
    np.testing.assert_array_equal(facets[finite], ((0, 1, 2),))
    np.testing.assert_array_equal(incident[finite], ((0, 1),))
    np.testing.assert_allclose(
        endpoints[finite][0], expected, atol=CIRCUMCENTER_COMPARISON_ATOL
    )
    assert np.all(dual_status == int(MeshcoreStatus.OK))
    assert kinds[finite][0] == 0
    assert counters.interior_facets == 1 and counters.hull_facets == 6


@NATIVE
def test_restricted_preparation_preserves_actual_pending_query_obligations(
    execution_scope: NativeExecutionScope,
) -> None:
    query = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER)
    )
    domain = prepare_implicit_volume_domain(
        query,
        implicit_specification(),
        NativeVolumeSchedule(refinement_rounds=1),
        scope=execution_scope,
    )
    work = prepare_implicit_restricted_queries(
        domain,
        spatial_tolerance=0.02,
        maximum_depth=12,
        scope=execution_scope,
    )
    pending = work.roots.classes == int(ImplicitDualIntervalClass.UNRESOLVED)
    assert np.all(
        np.isin(work.root_facets[work.roots.segment_ids[pending]], work.unresolved_facets)
    )
    failed_rays = work.ray_facets[work.ray_status != int(MeshcoreStatus.OK)]
    assert np.all(np.isin(failed_rays, work.unresolved_facets))
    assert_box_coverage(work.domain)
    assert work.geometry_queries <= domain.specification.limits.maximum_geometry_queries
    assert work.work_units <= domain.specification.limits.maximum_work_units


@NATIVE
def test_exact_hull_ray_cover_contains_domain_exit() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    centers, center_status = restricted_centers_3d(points, cells)
    assert center_status[0] == int(MeshcoreStatus.OK)
    bounds, kinds, status = restricted_rays_3d(
        points,
        cells,
        centers,
        np.asarray(((0, 1, 2),), dtype=np.int32),
        np.asarray((0,), dtype=np.int32),
        2.0 * IMPLICIT_DOMAIN,
    )
    assert status[0] == int(MeshcoreStatus.OK) and kinds[0] == 1
    start = np.asarray((0.5, 0.5, 0.5), dtype=np.float64)
    assert np.all(bounds[0, 0, 0] <= start)
    assert np.all(start <= bounds[0, 0, 1])
    # This hull face's Voronoi ray is vertical in the negative z
    # direction. The covered range must reach/past the box plane z=-2.
    assert bounds[0, 1, 1, 2] <= -2.0 + HULL_EXIT_ROUNDOFF_ATOL
    assert bounds[0, 1, 0, 2] <= -2.0
    assert bounds[0, 1, 0, 0] <= 0.5 <= bounds[0, 1, 1, 0]
    assert bounds[0, 1, 0, 1] <= 0.5 <= bounds[0, 1, 1, 1]


@NATIVE
def test_circumcenter_bounds_enclose_nonrepresentable_exact_rational_center() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 3.0, 5.0)),
        dtype=np.float64,
    )
    bounds, status = restricted_centers_3d(
        points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    assert status[0] == int(MeshcoreStatus.OK)
    # Equal squared distances to the unit-axis sites imply x=y=1/2.
    # The final site gives 2*x+3*y+5*z=19, hence z=33/10 exactly.
    exact_z = Fraction(33, 10)
    assert Fraction(float(bounds[0, 0, 2])) <= exact_z
    assert exact_z <= Fraction(float(bounds[0, 1, 2]))
    assert bounds[0, 1, 2] - bounds[0, 0, 2] < CIRCUMCENTER_ENCLOSURE_WIDTH_BOUND


def test_exact_zero_subdivision_endpoints_are_isolated_by_joint_interval_proof() -> None:
    query = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.5).compile(coordinate_unit=METER)
    )
    endpoints = np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64)
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.01,
        maximum_depth=14,
        maximum_geometry_queries=3000,
    )
    assert_terminal_cover(work)
    roots = work.parameter_intervals[
        work.classes == int(ImplicitDualIntervalClass.UNIQUE_ROOT)
    ]
    expected = np.asarray((0.25, 0.75), dtype=np.float64)
    assert roots.shape == (2, 2)
    assert np.all(roots[:, 0] < expected)
    assert np.all(expected < roots[:, 1])
    assert not np.any(work.classes == int(ImplicitDualIntervalClass.UNRESOLVED))


def test_oblique_monotone_dual_excludes_false_axis_box_zero() -> None:
    query = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile(coordinate_unit=METER)
    )
    endpoints = np.asarray((((0.999, 0.0, 0.0), (0.998, 0.06, 0.0)),), dtype=np.float64)
    # Both endpoints lie inside the convex ball, so their full segment
    # does too. Its axis box nevertheless includes points outside the ball.
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.04,
        maximum_depth=32,
        maximum_geometry_queries=10000,
    )
    assert_terminal_cover(work)
    assert np.all(work.classes == int(ImplicitDualIntervalClass.EXCLUDED))
    assert np.all(work.value_upper < 0.0)
