#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    exact_orient3d,
    meshcore_available,
    MeshcoreStatus,
    restricted_centers_3d,
    restricted_dual_3d,
    restricted_rays_3d,
)
from phydrax.geometry._contracts import CompiledGeometry
from phydrax.geometry.design._schema import ParameterId
from phydrax.geometry.implicit._adaptive_discovery import (
    ImplicitVolumeClass,
    ImplicitVolumeQuery,
)
from phydrax.geometry.implicit._analytic_profile import (
    AnalyticBoundaryCoverCapacityError,
    AnalyticImplicitProfile,
)
from phydrax.geometry.implicit._enclosure import _field_bounds, _INTERVAL_PROGRAMS
from phydrax.geometry.implicit._policy import ImplicitDiscoveryEnclosure
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._implicit_volume import (
    _classify_adaptive_volume_seeds,
    ImplicitDualIntervalClass,
    ImplicitDualRootWorkset,
    ImplicitVolumeDomainWorkset,
    ImplicitVolumeWorkStop,
    isolate_implicit_dual_roots,
    prepare_adaptive_implicit_volume,
    prepare_implicit_restricted_queries,
    prepare_implicit_volume_domain,
)
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing._volume_generation import (
    native_volume_execution_budget,
    NativeVolumeSchedule,
)
from phydrax.units import CENTIMETER


_DOMAIN = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=np.float64)
_NATIVE = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)


def _query(
    geometry: CompiledGeometry,
    enclosure: ImplicitDiscoveryEnclosure = "interval",
) -> ImplicitVolumeQuery:
    # A complete one-box enclosure is sufficient for the volume query contract;
    # no extracted surface or fixed-route geometry realization is involved.
    bounds = _field_bounds(geometry, enclosure)
    values = bounds.boxes(_DOMAIN[:1], _DOMAIN[1:])
    return ImplicitVolumeQuery(
        bounds,
        _DOMAIN[:1],
        _DOMAIN[1:],
        values.value_lower,
        values.value_upper,
        np.asarray((0,), dtype=np.int64),
        _DOMAIN,
        maximum_level=1,
    )


def _specification(limits: MeshingLimits | None = None) -> VolumeMeshingSpec:
    scope = MeshingScope(
        "implicit",
        "state",
        MeshingEntityKind.GEOMETRY,
        2,
        "boundary",
        np.asarray((0,), dtype=np.int64),
    )
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(scope, 0.2),),
        limits=limits,
    )


def _assert_box_coverage(work: ImplicitVolumeDomainWorkset) -> None:
    corners = work.points[work.tetrahedra]
    assert np.all(
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]) > 0
    )
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    assert np.sum(volumes) == pytest.approx(8.0, abs=1.0e-12)
    assert np.all(work.points >= _DOMAIN[0])
    assert np.all(work.points <= _DOMAIN[1])


def _assert_terminal_cover(work: ImplicitDualRootWorkset, segment: int = 0) -> None:
    intervals = work.parameter_intervals[work.segment_ids == segment]
    assert intervals[0, 0] == 0.0
    assert intervals[-1, 1] == 1.0
    np.testing.assert_array_equal(intervals[:-1, 1], intervals[1:, 0])
    assert np.all(intervals[:, 0] < intervals[:, 1])


@_NATIVE
def test_certified_inside_box_tetrahedra_cover_declared_domain() -> None:
    query = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), 3.0).compile())
    work = prepare_implicit_volume_domain(
        query, _specification(), NativeVolumeSchedule(refinement_rounds=2)
    )
    _assert_box_coverage(work)
    assert work.stop is ImplicitVolumeWorkStop.ALL_CLASSIFIED
    assert np.all(work.classes == int(ImplicitVolumeClass.INSIDE))
    assert np.all(work.value_upper < 0.0)
    assert work.unresolved_tetrahedra.size == 0


@_NATIVE
def test_tiny_bounded_component_cannot_be_discarded_by_coarse_samples() -> None:
    center = np.asarray((0.37, 0.13, -0.36), dtype=np.float64)
    query = _query(phx.geometry.Sphere(tuple(center), 0.04).compile())
    work = prepare_implicit_volume_domain(
        query, _specification(), NativeVolumeSchedule(refinement_rounds=2)
    )
    _assert_box_coverage(work)
    # Independent barycentric location of the required component center.
    corners = work.points[work.tetrahedra]
    matrix = np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
    barycentric = np.linalg.solve(matrix, (center - corners[:, 0])[..., None])[..., 0]
    contains = np.all(barycentric >= -1.0e-12, axis=1) & (
        np.sum(barycentric, axis=1) <= 1.0 + 1.0e-12
    )
    assert np.any(contains)
    assert np.all(work.classes[contains] != int(ImplicitVolumeClass.OUTSIDE))
    assert work.unresolved_tetrahedra.size > 0
    assert work.refinement_rounds == 2
    # No whole-cell inside verdict may contradict the analytic ball enclosure.
    inside_corners = corners[work.inside_tetrahedra]
    assert np.all(np.linalg.norm(inside_corners - center, axis=2) < 0.04)


@_NATIVE
def test_geometry_budget_retains_unqueried_domain_cells_with_unbounded_intervals() -> (
    None
):
    query = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile())
    work = prepare_implicit_volume_domain(
        query,
        _specification(MeshingLimits(maximum_geometry_queries=1)),
        NativeVolumeSchedule(refinement_rounds=2),
    )
    _assert_box_coverage(work)
    assert work.stop is ImplicitVolumeWorkStop.GEOMETRY_QUERY_LIMIT
    assert work.geometry_queries == 1
    assert np.all(work.classes[~work.queried] == int(ImplicitVolumeClass.UNKNOWN))
    assert np.all(np.isneginf(work.value_lower[~work.queried]))
    assert np.all(np.isposinf(work.value_upper[~work.queried]))
    assert "maximum_scratch_bytes" in work.unenforced_limits


@_NATIVE
def test_workset_refuses_changed_source_state_even_with_same_query_cover() -> None:
    first = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), 3.0).compile())
    second = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), 4.0).compile())
    specification = _specification()
    work = prepare_implicit_volume_domain(first, specification, NativeVolumeSchedule())
    work.require_current(first, specification)
    with pytest.raises(ValueError, match="stale"):
        work.require_current(second, specification)


def test_sampled_queries_cannot_supply_certified_domain_work() -> None:
    query = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(), "sampled")
    with pytest.raises(ValueError, match="rigorous"):
        prepare_implicit_volume_domain(query, _specification(), NativeVolumeSchedule())


@_NATIVE
@pytest.mark.parametrize(
    ("enclosure", "query_count"),
    (("sampled", 18), ("lipschitz", 20), ("interval", 4)),
)
def test_enclosure_batches_consume_the_actual_original_query_allowance(
    enclosure: ImplicitDiscoveryEnclosure,
    query_count: int,
) -> None:
    geometry = phx.geometry.Sphere((0.013, -0.021, 0.007), 0.75).compile()
    bounds = _field_bounds(geometry, enclosure)
    lower = np.asarray(((-0.2, -0.2, -0.2), (0.1, 0.1, 0.1)))
    upper = lower + 0.1
    with native_volume_execution_budget(
        MeshingLimits(maximum_geometry_queries=query_count),
    ) as execution:
        result = bounds.boxes(lower, upper)
    assert execution.evidence is not None
    assert execution.evidence.externally_charged_geometry_queries == query_count
    assert np.all(result.value_lower <= result.value_upper)
    with pytest.raises(MeshingFailure) as failure:
        with native_volume_execution_budget(
            MeshingLimits(maximum_geometry_queries=query_count - 1),
        ):
            bounds.boxes(lower, upper)
    assert failure.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED


@_NATIVE
def test_adaptive_seed_queries_reserve_the_remaining_source_allowance() -> None:
    geometry = phx.geometry.Sphere((0.013, -0.021, 0.007), 0.75).compile()
    policy = phx.geometry.AdaptiveImplicitSurfacePolicy(
        maximum_level=7,
        flatness_tolerance=0.02,
    )
    schedule = NativeVolumeSchedule()
    contract = phx.SpatialCoordinateContract.si()
    original = _specification()
    prepared = prepare_adaptive_implicit_volume(
        geometry,
        1.4 * _DOMAIN,
        original,
        schedule,
        policy,
        "state",
        contract,
    )
    specification = VolumeMeshingSpec(
        original.target,
        original.boundary_scope,
        original.fill_strategy,
        size_controls=original.size_controls,
        region_seeds=(
            phx.meshing.RegionSeed(
                np.asarray((0.013, -0.021, 0.007), dtype=np.float64),
                "body",
                "material",
                phx.meshing.RegionRole.SOLID,
            ),
        ),
        limits=MeshingLimits(maximum_geometry_queries=prepared.geometry_queries + 1),
    )
    with pytest.raises(MeshingFailure) as failure:
        _classify_adaptive_volume_seeds(
            prepared.surface,
            specification,
            prepared.geometry_queries,
        )
    assert failure.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.value.evidence.achieved) == {
        "geometry_queries": prepared.geometry_queries,
        "requested_seed_queries": 1,
    }
    admitted = VolumeMeshingSpec(
        original.target,
        original.boundary_scope,
        original.fill_strategy,
        size_controls=original.size_controls,
        region_seeds=specification.region_seeds,
        limits=MeshingLimits(maximum_geometry_queries=prepared.geometry_queries + 2),
    )
    assert (
        _classify_adaptive_volume_seeds(
            prepared.surface,
            admitted,
            prepared.geometry_queries,
        )
        == prepared.geometry_queries + 1
    )


def test_dual_interval_isolation_proves_both_sphere_crossings() -> None:
    radius = 0.371
    query = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), radius).compile())
    endpoints = np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64)
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.01,
        maximum_depth=14,
        maximum_geometry_queries=3000,
    )
    _assert_terminal_cover(work)
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
    query = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), radius).compile())
    endpoints = np.asarray((((-1.0, radius, 0.0), (1.0, radius, 0.0)),), dtype=np.float64)
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.01,
        maximum_depth=14,
        maximum_geometry_queries=3000,
    )
    _assert_terminal_cover(work)
    touches = (work.parameter_intervals[:, 0] <= 0.5) & (
        work.parameter_intervals[:, 1] >= 0.5
    )
    assert np.all(work.classes[touches] == int(ImplicitDualIntervalClass.UNRESOLVED))
    assert not np.any(work.classes == int(ImplicitDualIntervalClass.UNIQUE_ROOT))


def test_dual_query_exhaustion_preserves_full_parameter_cover() -> None:
    query = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile())
    endpoints = np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64)
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.01,
        maximum_depth=14,
        maximum_geometry_queries=3,
    )
    _assert_terminal_cover(work)
    assert work.query_budget_exhausted
    assert work.geometry_queries == 3
    assert np.all(work.classes == int(ImplicitDualIntervalClass.UNRESOLVED))
    assert np.all(np.isneginf(work.value_lower))
    assert np.all(np.isposinf(work.value_upper))


def test_dual_interval_isolation_proves_four_torus_crossings() -> None:
    major = 0.613
    minor = 0.087
    query = _query(
        phx.geometry.Torus((0.0, 0.0, 0.0), major - minor, major + minor).compile()
    )
    endpoints = np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64)
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.01,
        maximum_depth=14,
        maximum_geometry_queries=3000,
    )
    _assert_terminal_cover(work)
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
    query = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile())
    endpoints = np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64)
    bounds = np.stack((endpoints, endpoints), axis=2)
    # The tube includes the nominal two-crossing line and lines entirely
    # outside the sphere; no uniform UNIQUE_ROOT witness is possible.
    bounds[:, :, 1, 1] = 0.8
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        endpoint_bounds=bounds,
        spatial_tolerance=0.01,
        maximum_depth=8,
        maximum_geometry_queries=6000,
    )
    _assert_terminal_cover(work)
    assert np.any(work.classes == int(ImplicitDualIntervalClass.UNRESOLVED))
    assert not np.any(work.classes == int(ImplicitDualIntervalClass.UNIQUE_ROOT))


@_NATIVE
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
    # Independent exact rational centers of these two right tetrahedra.
    expected = np.asarray(((0.5, 0.5, 0.5), (0.5, 0.5, -0.5)), dtype=np.float64)
    assert np.all(status == int(MeshcoreStatus.OK))
    assert np.all(bounds[:, 0] <= expected)
    assert np.all(expected <= bounds[:, 1])
    assert np.max(bounds[:, 1] - bounds[:, 0]) < 1.0e-12
    facets, incident, endpoints, kinds, dual_status, counters = restricted_dual_3d(
        points,
        cells,
        2.0 * _DOMAIN,
        8,
        100,
    )
    finite = incident[:, 1] >= 0
    np.testing.assert_array_equal(facets[finite], ((0, 1, 2),))
    np.testing.assert_array_equal(incident[finite], ((0, 1),))
    np.testing.assert_allclose(endpoints[finite][0], expected, atol=1.0e-15)
    assert np.all(dual_status == int(MeshcoreStatus.OK))
    assert kinds[finite][0] == 0
    assert counters[1] == 1 and counters[2] == 6


@_NATIVE
def test_restricted_preparation_preserves_actual_pending_query_obligations() -> None:
    query = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile())
    domain = prepare_implicit_volume_domain(
        query, _specification(), NativeVolumeSchedule(refinement_rounds=1)
    )
    work = prepare_implicit_restricted_queries(
        domain,
        spatial_tolerance=0.02,
        maximum_depth=12,
    )
    pending = work.roots.classes == int(ImplicitDualIntervalClass.UNRESOLVED)
    assert np.all(
        np.isin(work.root_facets[work.roots.segment_ids[pending]], work.unresolved_facets)
    )
    failed_rays = work.ray_facets[work.ray_status != int(MeshcoreStatus.OK)]
    assert np.all(np.isin(failed_rays, work.unresolved_facets))
    _assert_box_coverage(work.domain)
    assert work.geometry_queries <= domain.specification.limits.maximum_geometry_queries
    assert work.work_units <= domain.specification.limits.maximum_work_units


@_NATIVE
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
        2.0 * _DOMAIN,
    )
    assert status[0] == int(MeshcoreStatus.OK) and kinds[0] == 1
    start = np.asarray((0.5, 0.5, 0.5), dtype=np.float64)
    assert np.all(bounds[0, 0, 0] <= start)
    assert np.all(start <= bounds[0, 0, 1])
    # This hull face's exact Voronoi ray is vertical in the negative z
    # direction. The covered range must reach/past the box plane z=-2.
    assert bounds[0, 1, 1, 2] <= -2.0 + 1.0e-12
    assert bounds[0, 1, 0, 2] <= -2.0
    assert bounds[0, 1, 0, 0] <= 0.5 <= bounds[0, 1, 1, 0]
    assert bounds[0, 1, 0, 1] <= 0.5 <= bounds[0, 1, 1, 1]


@_NATIVE
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
    assert bounds[0, 1, 2] - bounds[0, 0, 2] < 1.0e-12


def test_exact_zero_subdivision_endpoints_are_isolated_by_joint_interval_proof() -> None:
    query = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), 0.5).compile())
    endpoints = np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64)
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.01,
        maximum_depth=14,
        maximum_geometry_queries=3000,
    )
    _assert_terminal_cover(work)
    roots = work.parameter_intervals[
        work.classes == int(ImplicitDualIntervalClass.UNIQUE_ROOT)
    ]
    expected = np.asarray((0.25, 0.75), dtype=np.float64)
    assert roots.shape == (2, 2)
    assert np.all(roots[:, 0] < expected)
    assert np.all(expected < roots[:, 1])
    assert not np.any(work.classes == int(ImplicitDualIntervalClass.UNRESOLVED))


@pytest.mark.parametrize("family", ("sphere", "ring-torus"))
def test_nominal_profiles_establish_actual_reach_and_distance_brackets(
    family: str,
) -> None:
    if family == "sphere":
        geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile()
        points = np.asarray(
            ((0.0, 0.0, 0.0), (0.371, 0.0, 0.0), (0.571, 0.0, 0.0)), dtype=np.float64
        )
        reach, genus = 0.371, 0
    else:
        geometry = phx.geometry.Torus((0.0, 0.0, 0.0), 0.6, 1.0).compile()
        points = np.asarray(
            ((0.8, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.0, 0.0)), dtype=np.float64
        )
        reach, genus = 0.2, 1
    profile = AnalyticImplicitProfile(geometry, phx.SpatialCoordinateContract.si())
    assert 0.999999999999 * reach <= profile.reach_lower <= reach
    assert profile.genus == genus and profile.component_count == 1
    distances = profile.boundary_distance(points)
    with localcontext() as context:
        context.prec = 80
        for index, point in enumerate(points):
            x, y, z = (Decimal.from_float(float(value)) for value in point)
            radius = Decimal.from_float(reach)
            if family == "sphere":
                exact = abs((x * x + y * y + z * z).sqrt() - radius)
            else:
                major = Decimal.from_float(0.8)
                exact = abs(
                    (((x * x + y * y).sqrt() - major) ** 2 + z * z).sqrt() - radius
                )
            assert Decimal.from_float(float(distances.lower[index])) <= exact
            assert exact <= Decimal.from_float(float(distances.upper[index]))


@pytest.mark.parametrize("family", ("sphere", "ring-torus"))
def test_analytic_atlas_cover_contains_independent_dense_source_probes(
    family: str,
) -> None:
    if family == "sphere":
        geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile()
        first = np.linspace(0.0, 2.0 * np.pi, 37, endpoint=False, dtype=np.float64)
        second = np.linspace(0.0, np.pi, 19, dtype=np.float64)
        u, v = np.meshgrid(first, second, indexing="ij")
        probes = 0.371 * np.stack(
            (np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)), axis=-1
        ).reshape((-1, 3))
    else:
        geometry = phx.geometry.Torus((0.0, 0.0, 0.0), 0.6, 1.0).compile()
        axis = np.linspace(0.0, 2.0 * np.pi, 37, endpoint=False, dtype=np.float64)
        u, v = np.meshgrid(axis, axis, indexing="ij")
        probes = np.stack(
            (
                (0.8 + 0.2 * np.cos(v)) * np.cos(u),
                (0.8 + 0.2 * np.cos(v)) * np.sin(u),
                0.2 * np.sin(v),
            ),
            axis=-1,
        ).reshape((-1, 3))
    profile = AnalyticImplicitProfile(geometry, phx.SpatialCoordinateContract.si())
    cover = profile.boundary_cover(0.05, 5000)
    assert cover.complete and cover.semantics == "certified"
    assert np.max(cover.covering_radius) <= 0.05
    for chunk in np.array_split(probes, 8):
        separation = np.linalg.norm(chunk[:, None] - cover.points[None], axis=2)
        assert np.all(np.min(separation - cover.covering_radius[None], axis=1) <= 0.0)


def test_profile_refuses_changed_radius_with_unchanged_source_metadata() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.371, feature_id="profile-sphere"
    ).compile()
    contract = phx.SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        geometry, contract, source_id="external", source_revision="stable"
    )
    changed = geometry.with_parameters({ParameterId("profile-sphere", "radius"): 0.4})
    with pytest.raises(ValueError):
        profile.require_bound(changed, contract)


@pytest.mark.parametrize("change", ("frame", "unit"))
def test_profile_refuses_changed_physical_coordinate_contract(change: str) -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile()
    contract = phx.SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(geometry, contract)
    changed = (
        phx.SpatialCoordinateContract(contract.length_unit, reference_frame="scanner")
        if change == "frame"
        else phx.SpatialCoordinateContract(CENTIMETER)
    )
    with pytest.raises(ValueError):
        profile.require_bound(geometry, changed)


@pytest.mark.parametrize("invalidity", ("changed-angle", "horn"))
def test_profile_refuses_invalid_full_torus_state(invalidity: str) -> None:
    geometry = phx.geometry.Torus(
        (0.0, 0.0, 0.0), 0.6, 1.0, feature_id="profile-torus"
    ).compile()
    changed = geometry.with_parameters(
        {ParameterId("profile-torus", "angle"): np.pi}
        if invalidity == "changed-angle"
        else {ParameterId("profile-torus", "minor_radius"): 0.8}
    )
    with pytest.raises(ValueError):
        AnalyticImplicitProfile(changed, phx.SpatialCoordinateContract.si())


def test_analytic_cover_capacity_refusal_preserves_achieved_bound() -> None:
    profile = AnalyticImplicitProfile(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(),
        phx.SpatialCoordinateContract.si(),
    )
    with pytest.raises(AnalyticBoundaryCoverCapacityError) as failure:
        profile.boundary_cover(1.0e-4, 8)
    assert failure.value.requested_radius == 1.0e-4
    assert failure.value.achieved_radius > failure.value.requested_radius
    assert failure.value.maximum_samples == 8


def test_nonanalytic_same_zero_set_cannot_inherit_nominal_source_facts() -> None:
    sphere = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371)
    equivalent_field = (sphere | sphere).compile()
    with pytest.raises(ValueError):
        AnalyticImplicitProfile(equivalent_field, phx.SpatialCoordinateContract.si())


@_NATIVE
@pytest.mark.parametrize("family", ("sphere", "ring-torus"))
def test_native_provider_publishes_source_certified_restricted_volume(
    family: str,
) -> None:
    meshing = phx.meshing
    if family == "sphere":
        geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile()
        radius, major = 0.371, 0.0
    else:
        geometry = phx.geometry.Torus((0.0, 0.0, 0.0), 0.6, 1.0).compile()
        radius, major = 0.2, 0.8
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(1.2 * _DOMAIN))
    source = meshing.NativeImplicitSource(geometry, grid, "implicit", "state")
    scope = _specification().boundary_scope
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(scope, 0.2, strength=SizeControlStrength.SOFT),
        ),
    )
    result = (
        meshing.NativeMeshingProvider(
            meshing.NativeMeshingOptions("implicit_restricted_delaunay")
        )
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    assert isinstance(result, meshing.CellMeshingResult)
    certificate = result.certification
    assert certificate is not None and certificate.passed
    assert certificate.coverage is not None and certificate.coverage.status == "certified"
    fidelity = certificate.fidelity
    assert fidelity is not None and fidelity.status == "certified"
    projection = fidelity.projection_coverage
    assert projection is not None and projection.status == "certified"
    assert projection.fiber_crossing_classes.count("interior") == 1
    points = np.asarray(result.mesh.coordinates)
    cells = np.asarray(result.mesh.blocks[0].vertices)
    corners = points[cells]
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    assert np.all(volumes > 0.0)
    deviation = projection.forward_upper
    volume = float(np.sum(volumes))
    if family == "sphere":
        lower = 4.0 * np.pi / 3.0 * (radius - deviation) ** 3
        upper = 4.0 * np.pi / 3.0 * (radius + deviation) ** 3
    else:
        lower = 2.0 * np.pi**2 * major * (radius - deviation) ** 2
        upper = 2.0 * np.pi**2 * major * (radius + deviation) ** 2
    assert lower <= volume <= upper


@pytest.mark.parametrize("family", ("sphere", "ring-torus"))
def test_analytic_source_archive_rebuilds_cold_queries(
    tmp_path: Path, family: str
) -> None:
    if family == "sphere":
        geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile()
        points = np.asarray(
            ((0.0, 0.0, 0.0), (0.371, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64
        )
        expected = np.asarray((0.371, 0.0, 0.629), dtype=np.float64)
    else:
        geometry = phx.geometry.Torus((0.0, 0.0, 0.0), 0.6, 1.0).compile()
        points = np.asarray(
            ((0.8, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)), dtype=np.float64
        )
        expected = np.asarray((0.2, 0.0, 0.6), dtype=np.float64)
    profile = AnalyticImplicitProfile(
        geometry, phx.SpatialCoordinateContract.si(), cover_radius=0.05
    )
    path = tmp_path / "analytic-source.zip"
    receipt = write_meshing_source_closure(path, profile)
    _INTERVAL_PROGRAMS.clear()
    restored = read_meshing_source_closure(path, expected_content_id=receipt.content_id)
    assert isinstance(restored, AnalyticImplicitProfile)
    restored.validate_source_integrity()
    distances = restored.boundary_distance(points)
    assert np.all(distances.lower <= expected + 1e-15)
    assert np.all(distances.upper >= expected - 1e-15)
    np.testing.assert_allclose(distances.upper, expected, rtol=0.0, atol=1e-14)
    cover = restored.boundary_cover(0.05, 5000)
    assert cover.complete and cover.semantics == "certified"
    assert np.max(cover.covering_radius) <= 0.05


def test_interval_execution_cache_binds_actual_state_not_equal_shape() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.371, feature_id="cached-sphere"
    ).compile()
    first = AnalyticImplicitProfile(geometry, phx.SpatialCoordinateContract.si())
    changed = geometry.with_parameters(
        {
            phx.geometry.ParameterId("cached-sphere", "radius"): 0.613,
        }
    )
    second = AnalyticImplicitProfile(changed, phx.SpatialCoordinateContract.si())
    point = np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64)
    initial = first.boundary_distance(point)
    updated = second.boundary_distance(point)
    np.testing.assert_allclose(initial.upper, (0.629,), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(updated.upper, (0.387,), rtol=0.0, atol=1e-14)


def test_restored_profile_rejects_divergent_bound_state_before_queries() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.371, feature_id="bound-sphere"
    ).compile()
    profile = AnalyticImplicitProfile(geometry, phx.SpatialCoordinateContract.si())
    radius = geometry.schema.index(phx.geometry.ParameterId("bound-sphere", "radius"))
    divergent = eqx.tree_at(
        lambda value: value.field_bounds.state.values[radius],
        profile,
        jnp.asarray(0.613, dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="inconsistent scientific facts"):
        divergent.boundary_distance(np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64))


def test_oblique_monotone_dual_excludes_false_axis_box_zero() -> None:
    query = _query(phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile())
    endpoints = np.asarray((((0.999, 0.0, 0.0), (0.998, 0.06, 0.0)),), dtype=np.float64)
    # Both exact endpoints lie inside the convex ball, so their full segment
    # does too. Its axis box nevertheless includes points outside the ball.
    work = isolate_implicit_dual_roots(
        query,
        endpoints,
        spatial_tolerance=0.04,
        maximum_depth=32,
        maximum_geometry_queries=10000,
    )
    _assert_terminal_cover(work)
    assert np.all(work.classes == int(ImplicitDualIntervalClass.EXCLUDED))
    assert np.all(work.value_upper < 0.0)


@_NATIVE
@pytest.mark.parametrize("source_kind", ("non-distance", "internal-tangency"))
def test_general_enclosed_scalar_source_publishes_and_rejects_stale_state(
    source_kind: str,
    tmp_path: Path,
) -> None:
    from phydrax.geometry.implicit._adaptive_discovery import (
        AdaptiveImplicitBoundarySource,
    )
    from phydrax.meshing._implicit_volume import (
        execute_adaptive_implicit_volume,
        PreparedAdaptiveImplicitVolume,
    )

    if source_kind == "non-distance":
        primitive = phx.geometry.Ellipsoid((0.0123, 0.0037, -0.0102), (0.35, 0.28, 0.23))
        geometry = (primitive | primitive).compile()
        expected_volume = 4.0 * np.pi / 3.0 * 0.35 * 0.28 * 0.23
        components = 1
    else:
        geometry = (
            phx.geometry.Sphere((-0.37, 0.013, -0.017), 0.19)
            | phx.geometry.Sphere((0.36, -0.019, 0.021), 0.14)
        ).compile()
        expected_volume = 4.0 * np.pi / 3.0 * (0.19**3 + 0.14**3)
        components = 2
    meshing = phx.meshing
    domain = jnp.asarray(((-0.8, -0.6, -0.6), (0.8, 0.6, 0.6)), dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(domain)
    source = meshing.NativeImplicitSource(
        geometry, grid, "general-source", "scientific-revision"
    )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        "zero-set",
        np.asarray((0,), dtype=np.int64),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(scope, 0.2, strength=SizeControlStrength.SOFT),
        ),
    )
    provider = meshing.NativeMeshingProvider(
        meshing.NativeMeshingOptions(
            "implicit_adaptive_tetrahedral",
            implicit_policy=phx.geometry.AdaptiveImplicitSurfacePolicy(
                initial_level=2,
                minimum_surface_level=3,
                maximum_level=9,
            ),
            volume_schedule=NativeVolumeSchedule(
                refinement_rounds=1,
                improvement_passes=1,
                metric_optimization_passes=1,
            ),
        )
    )
    contract = phx.SpatialCoordinateContract.si()
    plan = provider.plan(source, specification, coordinate_contract=contract)
    result = plan.execute()
    assert isinstance(plan.prepared, PreparedAdaptiveImplicitVolume)
    prepared = plan.prepared
    assert prepared.surface.mesh is not None
    assert prepared.surface.mesh.topology.num_face_components == components
    assert prepared.surface.evidence.unresolved_count == 0
    certificate = result.certification
    assert certificate is not None and certificate.passed
    assert certificate.coverage is not None and certificate.coverage.status == "certified"
    assert certificate.fidelity is not None and certificate.fidelity.status == "certified"
    corners = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    volume = np.sum(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0
    assert abs(volume - expected_volume) < 0.1 * expected_volume
    if source_kind == "non-distance":
        boundary = AdaptiveImplicitBoundarySource(
            geometry, prepared.surface, source.source_revision
        )
        distance = boundary.boundary_distance(
            np.asarray(((0.4123, 0.0037, -0.0102),), dtype=np.float64)
        )
        assert distance.lower[0] <= 0.05 <= distance.upper[0]
    path = tmp_path / "general-source.zip"
    receipt = write_meshing_source_closure(path, prepared)
    restored = read_meshing_source_closure(path, expected_content_id=receipt.content_id)
    assert isinstance(restored, PreparedAdaptiveImplicitVolume)
    restored_boundary = AdaptiveImplicitBoundarySource(
        restored.geometry, restored.surface, restored.source_revision
    )
    point, exact_distance = (
        ((0.4123, 0.0037, -0.0102), 0.05)
        if source_kind == "non-distance"
        else ((0.5, -0.019, 0.021), 0.0)
    )
    distance = restored_boundary.boundary_distance(np.asarray((point,), dtype=np.float64))
    assert distance.lower[0] <= exact_distance <= distance.upper[0]
    changed = phx.geometry.Ellipsoid((0.02, 0.03, 0.04), (0.35, 0.28, 0.23)).compile()
    with pytest.raises(ValueError, match="stale"):
        execute_adaptive_implicit_volume(
            changed,
            specification,
            prepared,
            contract,
            provider.info(),
            plan.plan_id,
        )
    assert result.certification is certificate


@_NATIVE
def test_analytic_cover_refuses_cumulative_queries_before_returning_samples() -> None:
    from phydrax._meshcore import MeshcoreError, NativeExecutionBudget

    profile = AnalyticImplicitProfile(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(),
        phx.SpatialCoordinateContract.si(),
    )
    published: list[str] = []
    with pytest.raises(MeshcoreError) as caught:
        with NativeExecutionBudget(
            max_work=1_000_000,
            max_geometry_queries=2,
            max_cavity_cells=100,
            max_scratch_bytes=8 * 1024**2,
            max_wall_seconds=120.0,
        ) as execution:
            profile.boundary_samples(1)
            published.append("inadmissible")
    assert not published
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert execution.evidence is not None
    assert int(execution.evidence.work_evidence[1]) == 2


@_NATIVE
def test_implicit_boundary_tags_follow_sci_through_native_coefficient_regrouping() -> (
    None
):
    from tools.meshing_design_source import original_controls
    from tools.meshing_qualification import (
        _design_qualification_native_event,
        _design_qualification_source,
    )

    controls = original_controls()
    geometry, source, request, options = _design_qualification_source(controls)
    plan = phx.meshing.NativeMeshingProvider(options).plan(
        source,
        request,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    initial = plan.execute(geometry.state)
    assert initial.boundary is not None
    source_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in initial.boundary.mesh.blocks]
    )
    source_tags = dict(
        zip(source_ids.tolist(), initial.boundary.metadata.cell_tags, strict=True)
    )
    adapted, _, _, _, _ = _design_qualification_native_event(
        initial,
        "uniform",
        request.limits,
        {},
    )
    boundary = adapted.target.boundary
    assert boundary is not None and adapted.lineage is not None
    record = adapted.lineage.entity_lineage(2)
    parents: dict[int, set[str]] = {}
    for parent, child in zip(
        np.asarray(record.source_global_ids),
        np.asarray(record.target_global_ids),
        strict=True,
    ):
        parents.setdefault(int(child), set()).add(source_tags[int(parent)])
    target_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in boundary.mesh.blocks]
    )
    assert set(target_ids.tolist()) == set(parents)
    for identifier, tag in zip(target_ids, boundary.metadata.cell_tags, strict=True):
        assert parents[int(identifier)] == {tag}
    assert boundary.metadata.source_id == initial.boundary.metadata.source_id
    assert boundary.metadata.source_revision == initial.boundary.metadata.source_revision
    assert tuple(value.name for value in boundary.selections) == tuple(
        value.name for value in initial.boundary.selections
    )
    assert tuple(value.name for value in boundary.interfaces) == tuple(
        value.name for value in initial.boundary.interfaces
    )
