#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from itertools import product
from typing import NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest
from jax import Array
from scipy.spatial import cKDTree

import phydrax.meshing.providers._native_implicit as implicit_provider
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import (
    CellMesh,
    SimplicialConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._tensor_support import PreparedTensorGrid
from phydrax.geometry import Sphere, Torus
from phydrax.geometry._approximate_source_coverage import (
    _single_sheet_preimages,
    SourceCoverageUnavailable,
)
from phydrax.geometry._capabilities import GeometryCapability
from phydrax.geometry._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from phydrax.geometry._contracts import CompiledGeometry, GeometryKernel, GeometryKind
from phydrax.geometry._interval_enclosure import (
    prepare_interval_function,
    PreparedIntervalFunction,
)
from phydrax.geometry._sampling import RejectionSamplingPlan
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryDistanceWitness,
)
from phydrax.geometry._validity import GeometryValidityEvidence, unrestricted_validity
from phydrax.geometry.analytic._primitives import _BallKernel
from phydrax.geometry.design._schema import DesignState
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfacePolicy,
    AdaptiveImplicitSurfaceStatus,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.geometry.implicit._restricted_delaunay_coverage import (
    _prove_degree_one_normal_fiber,
    _prove_face_projection_orientation,
    _revalidate_oriented_surface_topology,
    RestrictedDelaunayBoundaryWitness,
    RestrictedDelaunayCoverageUnavailable,
    validate_restricted_delaunay_boundary_witness,
)
from phydrax.geometry.implicit._source_charts import (
    prepare_adaptive_implicit_root_charts,
)
from phydrax.meshing._association import GeometryAssociationKind
from phydrax.meshing._audit import _boundary_issues
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingFailure,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._implicit_volume import (
    _inside_seed_rows,
    _root_edge_bank,
    _strict_region_site_mask,
    _tetra_quality,
    _topology_matches,
    _triangle_diameter_upper,
)
from phydrax.meshing._native_construction import NativeMeshConstruction
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native import NativeMeshingProvider
from phydrax.meshing.providers._native_implicit import (
    _carrier_face_bounds,
    _carrier_forward_bound,
    _carrier_interval_programs,
    _NativeImplicitBoundaryQuery,
    _PreparedImplicitRoute,
    normalize_analytic_implicit_source,
    prepare_native_implicit,
)
from phydrax.meshing.providers._native_options import (
    NativeMeshingOptions,
    NativeMeshingRoute,
)
from phydrax.meshing.providers._native_plc_geometry import volume_cell_entity_rows
from phydrax.meshing.providers._native_sources import NativeImplicitSource
from phydrax.typing import PRNGKey


type _Point3 = tuple[Fraction, Fraction, Fraction]
type _Triangle3 = tuple[_Point3, _Point3, _Point3]


class _PolynomialBallKernel(GeometryKernel):
    ball: _BallKernel
    mode: str = eqx.field(static=True)

    def __init__(self, ball: _BallKernel, mode: str, /) -> None:
        self.ball = ball
        self.mode = mode

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 3

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return frozenset({GeometryCapability.REGION_QUERY})

    @property
    def field_certificate(self) -> FieldCertificate:
        return FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.EXACT,
            sign_reliability=SignReliability.RELIABLE,
            distance_semantics=DistanceSemantics.LEVEL_SET,
            regularity=FieldRegularity.SMOOTH,
            safe_step_factor=None,
            validity_region="all_space",
            parameter_differentiable=True,
            provenance=("polynomial_ball_test",),
        )

    def geometry_validity(self, state: DesignState, /) -> GeometryValidityEvidence:
        del state
        return unrestricted_validity(contract_id="polynomial-ball")

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        center, radius = self.ball._parameters(state)
        if self.mode == "ellipsoid":
            axes = radius * jnp.asarray((1.0, 0.8, 0.6), dtype=jnp.float64)
            return jnp.sum(((points - center) / axes) ** 2, axis=-1) - 1.0
        if self.mode == "scaled_distance":
            distance = jnp.linalg.norm(points - center, axis=-1) - radius
            return distance * (2.0 + 0.1 * points[..., 0] ** 2)
        value = jnp.sum((points - center) ** 2, axis=-1) - radius**2
        if self.mode == "scaled":
            return value * (2.0 + 0.1 * points[..., 0] ** 2)
        if self.mode == "tanh":
            return value * (2.0 + 0.1 * jnp.tanh(points[..., 0]))
        if self.mode == "singular":
            return value**3
        return value

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        return self.ball.boundary_normal(state, points)

    def bounds(self, state: DesignState, /) -> Array:
        return self.ball.bounds(state)

    def measure(self, state: DesignState, /) -> Array:
        return self.ball.measure(state)

    def boundary_measure(self, state: DesignState, /) -> Array:
        return self.ball.boundary_measure(state)

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: PRNGKey,
        plan: RejectionSamplingPlan | None = None,
    ) -> NoReturn:
        raise NotImplementedError

    def sample_boundary(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey
    ) -> NoReturn:
        raise NotImplementedError

    def boundary_atlas(self, state: DesignState, /) -> NoReturn:
        raise NotImplementedError


def _geometry_and_grid() -> tuple[CompiledGeometry, PreparedTensorGrid]:
    geometry = Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id="adaptive-route-sphere",
    ).compile()
    grid = TensorGridPlan(
        tuple(UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))
    return geometry, grid


def _grid_domain(grid: PreparedTensorGrid, /) -> np.ndarray:
    lower: list[Array] = []
    upper: list[Array] = []
    for axis in grid.axes:
        bounds = axis.bounds
        if bounds is None:
            raise AssertionError("Prepared implicit grid axis must retain its bounds.")
        lower.append(bounds[0])
        upper.append(bounds[1])
    return np.asarray((lower, upper), dtype=np.float64)


def _request(
    source: NativeImplicitSource | AnalyticImplicitProfile,
    /,
    *,
    maximum_deviation: float | None,
) -> tuple[SurfaceMeshingSpec, NativeMeshingOptions, SpatialCoordinateContract]:
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            3,
            CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=(),
        maximum_deviation=maximum_deviation,
    )
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08,
        root_tolerance=1.0e-8,
    )
    options = NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=policy,
    )
    contract = SpatialCoordinateContract.si()
    return specification, options, contract


def _assert_consumer_metadata(
    source: NativeImplicitSource | AnalyticImplicitProfile,
    specification: SurfaceMeshingSpec,
    construction: NativeMeshConstruction,
    contract: SpatialCoordinateContract,
) -> None:
    mesh = construction.mesh
    boundary = construction.organization.boundary
    assert boundary is not None
    assert boundary.mesh is mesh
    assert boundary.metadata.source_id == source.source_id
    assert boundary.metadata.source_revision == source.source_revision
    assert boundary.metadata.coordinate_contract.spatial_id == contract.spatial_id
    assert (
        boundary.metadata.cell_tags
        == ("implicit-zero-set",) * (mesh.blocks[0].vertices.shape[0])
    )

    source_scope_id = int(source.scope_entity_ids(2)[0])
    source_entity_id, occurrence = source.source_entity_key_for_scope(2, source_scope_id)
    cell_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id == mesh.entity_set(2).entity_set_id
    )
    assert cell_association.association_kind is GeometryAssociationKind.IMPLICIT
    assert cell_association.source_id == source.source_id
    assert cell_association.source_revision == source.source_revision
    assert cell_association.source_entity_ids == (source_entity_id,) * len(
        cell_association.source_entity_ids
    )
    assert cell_association.source_occurrence_paths == (occurrence,) * len(
        cell_association.source_occurrence_paths
    )
    assert cell_association.complete

    labels = construction.organization.labels
    assert len(labels) == 1
    assert labels[0].name == "implicit-zero-set"
    np.testing.assert_array_equal(
        labels[0].scope.entity_ids,
        mesh.entity_set(2).entity_ids,
    )
    patches = construction.organization.patches
    assert len(patches) == 1
    assert patches[0].name == "implicit-zero-set"
    np.testing.assert_array_equal(
        patches[0].scope.entity_ids,
        mesh.entity_set(2).entity_ids,
    )

    certification = construction.certification
    assert certification.schedule.route == "surface"
    assert certification.source_id == source.source_id
    assert certification.source_revision == source.source_revision
    assert certification.fidelity_source is not None
    assert certification.fidelity_source.source_id == source.source_id
    assert certification.fidelity_source.source_revision == source.source_revision
    if specification.maximum_deviation is None:
        assert certification.fidelity_tolerance is None
    else:
        assert certification.fidelity_tolerance is not None
        assert certification.fidelity_tolerance <= specification.maximum_deviation
    witness = certification.fidelity_witness
    if not isinstance(witness, SourceBoundaryChartWitness):
        raise AssertionError("Implicit fidelity must retain its source chart witness.")
    assert witness.source_id == source.source_id
    assert witness.source_revision == source.source_revision
    assert witness.source_query_id
    assert np.asarray(witness.implicit_cell_keys).shape == (mesh.blocks[0].cell_count, 4)
    assert np.asarray(witness.implicit_root_edge_keys).shape == (
        mesh.blocks[0].cell_count,
        3,
        2,
        3,
    )
    witness.require_current(mesh, construction.geometry, certification.fidelity_source)
    evidence = certification.fidelity_source.boundary_distance(
        np.asarray(mesh.coordinates, dtype=np.float64)[:4]
    )
    assert evidence.semantics == "certified"
    assert construction.compliance.passed


def test_native_analytic_implicit_profile_normalizes_inside_plan_preparation() -> None:
    geometry, _ = _geometry_and_grid()
    contract = SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        geometry,
        contract,
        source_id="analytic-profile-sphere",
        source_revision="analytic-profile-revision-4",
    )
    specification, options, _ = _request(
        profile,
        maximum_deviation=0.8,
    )

    assert profile.scope_entity_ids(0).size == 0
    assert profile.scope_entity_ids(1).size == 0
    assert profile.scope_entity_ids(2).shape == (1,)
    assert profile.scope_entity_ids(3).shape == (1,)
    assert profile.source_id == "analytic-profile-sphere"
    assert profile.source_revision == "analytic-profile-revision-4"

    construction = prepare_native_implicit(
        profile,
        specification,
        options,
        contract,
    ).execute()

    _assert_consumer_metadata(profile, specification, construction, contract)


@pytest.mark.parametrize("shape", ["sphere", "torus"])
@pytest.mark.parametrize("cap", [0.125, 0.0625])
def test_carrier_forward_bound_covers_analytic_triangle_samples(
    shape: str, cap: float
) -> None:
    contract = SpatialCoordinateContract.si()
    model = (
        Sphere((0.0, 0.0, 0.0), 1.0)
        if shape == "sphere"
        else Torus((0.0, 0.0, 0.0), 0.5, 1.5)
    )
    profile = AnalyticImplicitProfile(
        model.compile(), contract, source_id=f"bound-{shape}"
    )
    specification, _, _ = _request(profile, maximum_deviation=0.2)
    options = NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=cap, root_tolerance=1.0e-8
        ),
    )
    prepared = prepare_native_implicit(profile, specification, options, contract)
    construction = prepared.execute()
    bound = dict(construction.compliance.achieved)["source_fidelity_bound"]
    mesh = construction.mesh
    points = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    weights = np.asarray(
        [(i / 6.0, j / 6.0, (6 - i - j) / 6.0) for i in range(7) for j in range(7 - i)],
        dtype=np.float64,
    )
    samples = np.einsum("ij,njk->nik", weights, points)
    if shape == "sphere":
        distances = np.abs(np.linalg.norm(samples, axis=-1) - 1.0)
    else:
        radial = np.linalg.norm(samples[..., :2], axis=-1)
        distances = np.abs(np.hypot(radial - 1.0, samples[..., 2]) - 0.5)
    assert float(np.max(distances)) <= bound
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    phi = np.linspace(
        0.0, np.pi if shape == "sphere" else 2.0 * np.pi, 48, endpoint=False
    )
    t, p = np.meshgrid(theta, phi)
    if shape == "sphere":
        roots = np.stack(
            (np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)), axis=-1
        )
    else:
        roots = np.stack(
            (
                (1.0 + 0.5 * np.cos(p)) * np.cos(t),
                (1.0 + 0.5 * np.cos(p)) * np.sin(t),
                0.5 * np.sin(p),
            ),
            axis=-1,
        )
    roots = roots.reshape(-1, 3)
    candidate_rows = cKDTree(np.mean(points, axis=1)).query(roots, k=8)[1]
    candidate = points[candidate_rows]
    query = roots[:, None, :]
    a, b, c = candidate[:, :, 0], candidate[:, :, 1], candidate[:, :, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    normal_squared = np.sum(normal * normal, axis=-1)
    signed = np.sum((query - a) * normal, axis=-1) / normal_squared
    projected = query - signed[..., None] * normal
    ap = projected - a
    d00 = np.sum(ab * ab, axis=-1)
    d01 = np.sum(ab * ac, axis=-1)
    d11 = np.sum(ac * ac, axis=-1)
    d20 = np.sum(ap * ab, axis=-1)
    d21 = np.sum(ap * ac, axis=-1)
    denominator = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    inside = (v >= 0.0) & (w >= 0.0) & (v + w <= 1.0)
    distance_squared = np.where(inside, signed * signed * normal_squared, np.inf)
    for first, second in ((a, b), (b, c), (c, a)):
        edge = second - first
        fraction = np.clip(
            np.sum((query - first) * edge, axis=-1) / np.sum(edge * edge, axis=-1),
            0.0,
            1.0,
        )
        delta = query - (first + fraction[..., None] * edge)
        distance_squared = np.minimum(distance_squared, np.sum(delta * delta, axis=-1))
    assert float(np.max(np.sqrt(np.min(distance_squared, axis=1)))) <= bound


def test_implicit_resolution_is_coarsest_certified_level() -> None:
    contract = SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        Sphere((0.013, -0.017, 0.019), 1.0).compile(),
        contract,
        source_id="coarsest-certified-ball",
    )
    specification, _, _ = _request(profile, maximum_deviation=0.2)
    policy = AdaptiveImplicitSurfacePolicy(root_tolerance=1.0e-8)
    source = normalize_analytic_implicit_source(
        profile, policy, contract, specification=specification
    )
    selected = source.discovery
    assert selected is not None
    programs = _carrier_interval_programs(profile.geometry)
    assert programs is not None
    assert _carrier_forward_bound(selected, programs, 0.1) <= 0.1
    for level in range(selected.scaffold.target_level):
        trial = discover_adaptive_implicit_surface(
            profile.geometry,
            domain=selected.evidence.declared_domain,
            policy=AdaptiveImplicitSurfacePolicy(
                relative_cell_diameter=1.01 * 2.0**-level,
                root_tolerance=1.0e-8,
            ),
            source_id=profile.source_id,
        )
        if (
            trial.evidence.status_flags == AdaptiveImplicitSurfaceStatus.SUCCESS
            and trial.mesh is not None
            and trial.scaffold.target_level < selected.scaffold.target_level
        ):
            assert _carrier_forward_bound(trial, programs, 0.1) > 0.1


def test_carrier_bound_includes_nonzero_root_residual() -> None:
    contract = SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        Sphere((0.0, 0.0, 0.0), 1.0).compile(),
        contract,
        source_id="root-residual-ball",
    )
    specification, _, _ = _request(profile, maximum_deviation=0.8)
    options = NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.125, root_tolerance=1.0e-8
        ),
    )
    construction = prepare_native_implicit(
        profile, specification, options, contract
    ).execute()
    bound = dict(construction.compliance.achieved)["source_fidelity_bound"]
    distances = np.abs(
        np.linalg.norm(np.asarray(construction.mesh.coordinates), axis=1) - 1.0
    )
    assert 0.0 < float(np.max(distances)) <= bound


@pytest.mark.parametrize("scaled", [False, True])
def test_carrier_bound_for_non_distance_sphere_field(scaled: bool) -> None:
    original, grid = _geometry_and_grid()
    ball = original.kernel
    if not isinstance(ball, _BallKernel):
        raise AssertionError("The analytic sphere must compile to a ball kernel.")
    kernel = _PolynomialBallKernel(ball, "scaled" if scaled else "polynomial")
    geometry = CompiledGeometry(kernel, original.state)
    source = NativeImplicitSource(geometry, f"polynomial-ball-{scaled}", "r1", grid=grid)
    specification, options, contract = _request(source, maximum_deviation=0.2)
    construction = prepare_native_implicit(
        source, specification, options, contract
    ).execute()
    bound = dict(construction.compliance.achieved)["source_fidelity_bound"]
    mesh = construction.mesh
    triangles = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    samples = np.mean(triangles, axis=1)
    true_distances = np.abs(np.linalg.norm(samples, axis=1) - 0.75)
    assert float(np.max(true_distances)) <= bound


def _ellipsoid_distances(points: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Independent Lagrange-multiplier closest-point solve, including pole cases."""
    squared = axes * axes
    inside = np.sum((points / axes) ** 2, axis=1) < 1.0
    low = np.where(inside, np.nextafter(-squared[-1], 0.0), 0.0)
    high = np.where(inside, 0.0, np.maximum(1.0, np.linalg.norm(points, axis=1)))
    # At a zero shortest-axis component the multiplier can equal the pole;
    # then the missing coordinate follows from the ellipsoid constraint.
    pole_coordinates = squared[:2] * points[:, :2] / (squared[:2] - squared[-1])
    pole_remainder = 1.0 - np.sum((pole_coordinates / axes[:2]) ** 2, axis=1)
    pole = inside & (np.abs(points[:, 2]) < 1.0e-14) & (pole_remainder >= 0.0)
    for _ in range(80):
        mid = 0.5 * (low + high)
        level = np.sum(squared * points**2 / (squared + mid[:, None]) ** 2, axis=1)
        low = np.where(level > 1.0, mid, low)
        high = np.where(level > 1.0, high, mid)
    multiplier = 0.5 * (low + high)
    with np.errstate(divide="ignore", invalid="ignore"):
        closest = squared * points / (squared + multiplier[:, None])
    closest[pole, :2] = pole_coordinates[pole]
    closest[pole, 2] = axes[2] * np.sqrt(pole_remainder[pole])
    return np.linalg.norm(points - closest, axis=1)


@pytest.mark.parametrize("mode", ["ellipsoid", "scaled_distance"])
@pytest.mark.parametrize("cap", [0.125, 0.08])
def test_carrier_bound_against_dense_independent_closest_points(
    mode: str, cap: float
) -> None:
    original, grid = _geometry_and_grid()
    ball = original.kernel
    if not isinstance(ball, _BallKernel):
        raise AssertionError("The analytic sphere must compile to a ball kernel.")
    geometry = CompiledGeometry(_PolynomialBallKernel(ball, mode), original.state)
    discovery = discover_adaptive_implicit_surface(
        geometry,
        domain=_grid_domain(grid),
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=cap, root_tolerance=1.0e-8
        ),
        source_id=f"dense-{mode}-{cap}",
    )
    if discovery.mesh is None:
        raise AssertionError("The reference geometry must have a discovery carrier.")
    programs = _carrier_interval_programs(geometry)
    if programs is None:
        raise AssertionError("The reference field must admit interval programs.")
    bound = _carrier_forward_bound(discovery, programs, 0.1)
    triangles = np.asarray(discovery.mesh.vertices)[np.asarray(discovery.mesh.faces)]
    weights = np.asarray(
        [(i / 6.0, j / 6.0, (6 - i - j) / 6.0) for i in range(7) for j in range(7 - i)],
        dtype=np.float64,
    )
    samples = np.einsum("ij,njk->nik", weights, triangles).reshape(-1, 3)
    if mode == "ellipsoid":
        distances = _ellipsoid_distances(
            samples, np.asarray((0.75, 0.6, 0.45), dtype=np.float64)
        )
    else:
        distances = np.abs(np.linalg.norm(samples, axis=1) - 0.75)
    assert np.all(np.isfinite(distances))
    assert float(np.max(distances)) <= bound


def test_singular_gradient_retains_first_order_carrier_bound() -> None:
    original, grid = _geometry_and_grid()
    ball = original.kernel
    if not isinstance(ball, _BallKernel):
        raise AssertionError("The analytic sphere must compile to a ball kernel.")
    discovery = discover_adaptive_implicit_surface(
        original,
        domain=_grid_domain(grid),
        policy=AdaptiveImplicitSurfacePolicy(relative_cell_diameter=0.08),
        source_id="singular-polynomial-ball",
    )
    kernel = _PolynomialBallKernel(ball, "singular")
    programs = _carrier_interval_programs(CompiledGeometry(kernel, original.state))
    assert programs is not None
    bound = _carrier_forward_bound(discovery, programs, 0.2)
    first_order = float(
        np.nextafter(
            discovery.evidence.maximum_surface_box_diagonal
            + discovery.policy.root_tolerance,
            np.inf,
        )
    )
    assert bound == first_order


def test_native_route_retains_first_order_when_native_program_is_refused() -> None:
    original, grid = _geometry_and_grid()
    ball = original.kernel
    if not isinstance(ball, _BallKernel):
        raise AssertionError("The analytic sphere must compile to a ball kernel.")
    geometry = CompiledGeometry(_PolynomialBallKernel(ball, "tanh"), original.state)
    assert _carrier_interval_programs(geometry) is None
    source = NativeImplicitSource(geometry, "tanh-ball", "r1", grid=grid)
    specification, options, contract = _request(source, maximum_deviation=0.8)
    prepared = prepare_native_implicit(source, specification, options, contract)
    if not isinstance(prepared, _PreparedImplicitRoute):
        raise AssertionError("Expected the adaptive implicit surface route.")
    construction = prepared.execute()
    first = np.nextafter(
        prepared.discovery.evidence.maximum_surface_box_diagonal
        + prepared.discovery.policy.root_tolerance,
        np.inf,
    )
    assert dict(construction.compliance.achieved)["source_fidelity_bound"] == first


def _face_programs(
    field: Callable[[Array], Array],
) -> tuple[
    PreparedIntervalFunction,
    PreparedIntervalFunction,
    PreparedIntervalFunction,
    PreparedIntervalFunction,
]:
    return (
        prepare_interval_function(field, 3, batch_capacity=1, execution="batched"),
        prepare_interval_function(
            jax.grad(field), 3, batch_capacity=1, execution="batched"
        ),
        prepare_interval_function(
            jax.hessian(field), 3, batch_capacity=1, execution="batched"
        ),
        prepare_interval_function(field, 3, batch_capacity=1024, execution="native"),
    )


def _test_face_bound(
    field: Callable[[Array], Array], triangle: np.ndarray, radius: float
) -> float:
    domain = np.asarray(((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0)), dtype=np.float64)
    return float(
        _carrier_face_bounds(triangle[None], _face_programs(field), radius, domain)[0]
    )


def test_face_bound_includes_vertex_residual() -> None:
    def field(point: Array) -> Array:
        return jnp.sum(point * point) - 1.0

    r = 0.001
    x = np.sqrt(1.01**2 - r**2)
    triangle = np.asarray(
        ((x, r, 0.0), (x, -r / 2, r), (x, -r / 2, -r)), dtype=np.float64
    )
    assert _test_face_bound(field, triangle, 0.03) >= 0.01


@pytest.mark.parametrize(
    ("length", "residual", "radius"), [(1.0, 0.3, 0.5), (0.4, 0.25, 0.1)]
)
def test_face_bound_refuses_flow_beyond_tube(
    length: float, residual: float, radius: float
) -> None:
    def field(point: Array) -> Array:
        z = point[2]
        return residual - (z - z * z / (2.0 * length))

    triangle = np.asarray(
        ((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.0, 0.001, 0.0)), dtype=np.float64
    )
    assert np.isinf(_test_face_bound(field, triangle, radius))


def test_face_bound_covers_equilateral_sphere_sagitta() -> None:
    def field(point: Array) -> Array:
        return jnp.sum(point * point) - 1.0

    edge = 0.15
    r = edge / np.sqrt(3.0)
    x = np.sqrt(1.0 - r * r)
    triangle = np.asarray(
        ((x, r, 0.0), (x, -r / 2.0, edge / 2.0), (x, -r / 2.0, -edge / 2.0)),
        dtype=np.float64,
    )
    assert _test_face_bound(field, triangle, 0.01) >= 1.0 - x


def test_face_bound_refuses_crease() -> None:
    def field(point: Array) -> Array:
        return point[2] - jnp.abs(point[0])

    triangle = np.asarray(
        ((-0.05, 0.0, 0.05), (0.05, 0.0, 0.05), (0.05, 0.1, 0.05)),
        dtype=np.float64,
    )
    assert _test_face_bound(field, triangle, 0.1) >= 0.05 / np.sqrt(2.0)


def test_face_bound_restricts_only_faces_near_domain_boundary() -> None:
    def field(point: Array) -> Array:
        return point[2]

    triangles = np.asarray(
        (
            ((0.0, 0.0, 0.01), (0.01, 0.0, 0.01), (0.0, 0.01, 0.01)),
            ((0.95, 0.0, 0.01), (0.96, 0.0, 0.01), (0.95, 0.01, 0.01)),
        ),
        dtype=np.float64,
    )
    domain = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=np.float64)
    bounds = _carrier_face_bounds(triangles, _face_programs(field), 0.1, domain)
    assert 0.01 <= bounds[0] < np.inf
    assert np.isinf(bounds[1])


@pytest.mark.parametrize("refused_program", [0, 1, 2])
def test_carrier_interval_evaluation_refusal_retains_first_order(
    refused_program: int,
) -> None:
    geometry, grid = _geometry_and_grid()
    discovery = discover_adaptive_implicit_surface(
        geometry,
        domain=_grid_domain(grid),
        policy=AdaptiveImplicitSurfacePolicy(relative_cell_diameter=0.08),
        source_id="interval-evaluation-refusal",
    )
    programs = _carrier_interval_programs(geometry)
    assert programs is not None

    def negative_base(point: Array) -> Array:
        return (point[0] - 10.0) ** 0.5

    refused = prepare_interval_function(
        negative_base, 3, batch_capacity=1, execution="batched"
    )
    selected = list(programs)
    selected[refused_program] = refused
    bound = _carrier_forward_bound(
        discovery, (selected[0], selected[1], selected[2], selected[3]), 0.2
    )
    first = np.nextafter(
        discovery.evidence.maximum_surface_box_diagonal + discovery.policy.root_tolerance,
        np.inf,
    )
    assert bound == first


def test_carrier_does_not_follow_flow_outside_declared_domain() -> None:
    geometry, grid = _geometry_and_grid()
    discovery = discover_adaptive_implicit_surface(
        geometry,
        domain=_grid_domain(grid),
        policy=AdaptiveImplicitSurfacePolicy(relative_cell_diameter=0.08),
        source_id="domain-restricted-flow",
    )
    programs = _carrier_interval_programs(geometry)
    assert programs is not None
    first = np.nextafter(
        discovery.evidence.maximum_surface_box_diagonal + discovery.policy.root_tolerance,
        np.inf,
    )
    assert _carrier_forward_bound(discovery, programs, 0.8) == first


def test_ladder_preserves_discovery_failure_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, grid = _geometry_and_grid()
    source = NativeImplicitSource(geometry, "failing-discovery", "r1", grid=grid)
    specification, options, contract = _request(source, maximum_deviation=0.8)
    policy = options.implicit_policy
    if not isinstance(policy, AdaptiveImplicitSurfacePolicy):
        raise AssertionError("The request must have an adaptive implicit policy.")

    def refuse(
        geometry: CompiledGeometry,
        *,
        domain: np.ndarray,
        policy: AdaptiveImplicitSurfacePolicy,
        source_id: str,
    ) -> NoReturn:
        raise ValueError("discovery refusal")

    monkeypatch.setattr(implicit_provider, "discover_adaptive_implicit_surface", refuse)
    with pytest.raises(MeshingFailure) as old:
        implicit_provider._discover(source, _grid_domain(grid), policy)
    with pytest.raises(MeshingFailure) as new:
        prepare_native_implicit(source, specification, options, contract)
    assert (
        new.value.category,
        new.value.provider_code,
        new.value.stage,
        str(new.value),
    ) == (
        old.value.category,
        old.value.provider_code,
        old.value.stage,
        str(old.value),
    )


def test_ladder_continues_after_coarse_discovery_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, grid = _geometry_and_grid()
    source = NativeImplicitSource(geometry, "coarse-refusal", "r1", grid=grid)
    specification, _, contract = _request(source, maximum_deviation=0.8)
    policy = AdaptiveImplicitSurfacePolicy(root_tolerance=1.0e-8)
    options = NativeMeshingOptions("implicit_surface", implicit_policy=policy)
    original = implicit_provider.discover_adaptive_implicit_surface
    refused: list[float] = []

    def refuse_coarse(
        geometry: CompiledGeometry,
        *,
        domain: np.ndarray,
        policy: AdaptiveImplicitSurfacePolicy,
        source_id: str,
    ) -> AdaptiveImplicitSurface:
        resolution = policy.relative_cell_diameter
        if resolution is not None and resolution > 0.1:
            refused.append(resolution)
            raise ValueError("coarse discovery refusal")
        return original(geometry, domain=domain, policy=policy, source_id=source_id)

    monkeypatch.setattr(
        implicit_provider, "discover_adaptive_implicit_surface", refuse_coarse
    )
    prepared = prepare_native_implicit(source, specification, options, contract)
    if not isinstance(prepared, _PreparedImplicitRoute):
        raise AssertionError("Expected the adaptive implicit surface route.")
    assert refused
    resolution = prepared.discovery.policy.relative_cell_diameter
    if resolution is None:
        raise AssertionError("The effective discovery policy must have a resolution.")
    assert resolution <= 0.1
    assert prepared.execute().compliance.passed


def test_native_implicit_grid_source_builds_certified_surface_construction() -> None:
    geometry, grid = _geometry_and_grid()
    source = NativeImplicitSource(
        geometry,
        "implicit-grid-sphere",
        "grid-revision-3",
        grid=grid,
    )
    assert source.scope_entity_ids(0).size == 0
    assert source.scope_entity_ids(1).size == 0
    assert source.scope_entity_ids(2).shape == (1,)
    assert source.scope_entity_ids(3).shape == (1,)
    specification, options, contract = _request(
        source,
        maximum_deviation=0.8,
    )
    construction = prepare_native_implicit(
        source,
        specification,
        options,
        contract,
    ).execute()

    assert construction.mesh.topological_dimension == 2
    assert construction.mesh.ambient_dimension == 3
    assert all(block.cell_kind == "triangle" for block in construction.mesh.blocks)
    np.testing.assert_array_equal(
        construction.geometry.coordinates,
        construction.mesh.coordinates,
    )
    _assert_consumer_metadata(source, specification, construction, contract)


def test_native_implicit_retained_discovery_preserves_zero_set_identity() -> None:
    geometry, grid = _geometry_and_grid()
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08,
        root_tolerance=1.0e-8,
    )
    source_id = "implicit-retained-sphere"
    discovery = discover_adaptive_implicit_surface(
        geometry,
        domain=_grid_domain(grid),
        policy=policy,
        source_id=source_id,
    )
    source = NativeImplicitSource(
        geometry,
        source_id,
        "retained-revision-8",
        discovery=discovery,
    )
    specification, options, contract = _request(
        source,
        maximum_deviation=None,
    )
    construction = prepare_native_implicit(
        source,
        specification,
        options,
        contract,
    ).execute()

    assert discovery.mesh is not None
    np.testing.assert_array_equal(
        np.asarray(construction.mesh.coordinates),
        np.asarray(discovery.mesh.vertices),
    )
    _assert_consumer_metadata(source, specification, construction, contract)


def test_native_implicit_route_rejects_sampled_discovery() -> None:
    geometry, grid = _geometry_and_grid()
    source_id = "implicit-sampled-sphere"
    policy = AdaptiveImplicitSurfacePolicy(
        enclosure="sampled",
        relative_cell_diameter=0.08,
        root_tolerance=1.0e-8,
    )
    discovery = discover_adaptive_implicit_surface(
        geometry,
        domain=_grid_domain(grid),
        policy=policy,
        source_id=source_id,
    )
    source = NativeImplicitSource(
        geometry,
        source_id,
        "sampled-revision-1",
        grid=grid,
        discovery=discovery,
    )
    specification, _, contract = _request(source, maximum_deviation=None)
    options = NativeMeshingOptions("implicit_surface", implicit_policy=policy)

    with pytest.raises(MeshingFailure, match="complete rigorous cover"):
        prepare_native_implicit(source, specification, options, contract)


@pytest.mark.parametrize(
    "route",
    (
        "implicit_restricted_delaunay",
        "implicit_adaptive_tetrahedral",
    ),
)
def test_native_implicit_volume_route_builds_region_mesh(
    route: NativeMeshingRoute,
) -> None:
    geometry, grid = _geometry_and_grid()
    source_id = "implicit-volume-sphere"
    revision = "implicit-volume-revision-4"
    contract = SpatialCoordinateContract.si()
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08,
        root_tolerance=1.0e-8,
    )
    discovery: AdaptiveImplicitSurface | None = None
    if route == "implicit_restricted_delaunay":
        source: NativeImplicitSource | AnalyticImplicitProfile = AnalyticImplicitProfile(
            geometry,
            contract,
            tube_radius=0.5,
            source_id=source_id,
            source_revision=revision,
        )
    else:
        discovery = discover_adaptive_implicit_surface(
            geometry,
            domain=_grid_domain(grid),
            policy=policy,
            source_id=source_id,
        )
        source = NativeImplicitSource(
            geometry,
            source_id,
            revision,
            discovery=discovery,
        )
    boundary_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=0.8,
    )
    options = NativeMeshingOptions(
        route,
        implicit_policy=policy if route == "implicit_adaptive_tetrahedral" else None,
        volume_schedule=NativeVolumeSchedule(),
    )
    construction = prepare_native_implicit(
        source,
        specification,
        options,
        contract,
    ).execute()

    assert construction.mesh.topological_dimension == 3
    assert construction.mesh.ambient_dimension == 3
    assert all(
        reference_cell_topology(block.cell_kind).family == "tetrahedron"
        for block in construction.mesh.blocks
    )
    assert construction.certification.schedule.route == "volume_implicit"
    assert construction.certification.fidelity_source is not None
    assert construction.certification.fidelity_source.source_id == source.source_id
    assert (
        construction.certification.fidelity_source.source_revision
        == source.source_revision
    )
    assert construction.certification.fidelity_tolerance is not None
    assert specification.maximum_deviation is not None
    assert (
        construction.certification.fidelity_tolerance <= specification.maximum_deviation
    )
    assert construction.certification.cell_regions is not None
    assert np.asarray(construction.certification.cell_regions).shape == (
        construction.mesh.blocks[0].vertices.shape[0],
    )
    assert np.all(np.asarray(construction.certification.cell_regions) == 0)
    boundary = construction.organization.boundary
    assert boundary is not None
    assert boundary.metadata.source_id == source.source_id
    assert boundary.metadata.source_revision == source.source_revision
    assert all(tag == "implicit-zero-set" for tag in boundary.metadata.cell_tags)
    # Acceptance audits the boundary model against the volume by vertex identity.
    connectivity = construction.mesh.connectivity
    if isinstance(connectivity, TetrahedralConnectivity):
        face_rows = np.asarray(connectivity.faces)
    elif isinstance(connectivity, SimplicialConnectivity):
        face_rows = np.asarray(connectivity.entities[2])
    else:
        raise AssertionError("Implicit volume meshes must carry simplex connectivity.")
    boundary_rows = np.unique(face_rows[np.asarray(connectivity.boundary_faces)])
    volume_ids = np.asarray(construction.mesh.vertex_global_ids, dtype=np.int64)
    np.testing.assert_array_equal(
        np.sort(np.asarray(boundary.mesh.vertex_global_ids, dtype=np.int64)),
        np.sort(volume_ids[boundary_rows]),
    )
    assert _boundary_issues(construction.mesh, boundary) == ()
    assert len(construction.organization.zones) == 1
    if route == "implicit_restricted_delaunay":
        assert isinstance(source, AnalyticImplicitProfile)
        assert construction.certification.domain is source
        assert construction.certification.fidelity_source is source
        assert construction.organization.zones[0].name == source.region_ids[0]
        distance_witness = construction.certification.fidelity_witness
        coverage_witness = construction.certification.source_coverage_witness
        assert isinstance(distance_witness, SourceBoundaryDistanceWitness)
        assert isinstance(coverage_witness, RestrictedDelaunayBoundaryWitness)
        assert distance_witness.source_id == source.source_id
        assert distance_witness.source_revision == source.source_revision
        assert distance_witness.cover_radius < source.tube_radius
        proof = validate_restricted_delaunay_boundary_witness(
            construction.mesh,
            construction.geometry,
            source,
            construction.certification.cell_regions,
            distance_witness,
            coverage_witness,
            embedding=construction.validated_global_embedding,
        )
        assert proof.region_ids == source.region_ids
        assert proof.target_facet_count > 0
        assert proof.source_facet_count == proof.covered_source_facet_count == 0
        assert coverage_witness.distance_witness_id == distance_witness.witness_id
        validated_embedding = construction.validated_global_embedding
        assert validated_embedding is not None
        embedding = validated_embedding.certificate
        forged_embedding = eqx.tree_at(
            lambda value: value.validity.determinant_lower,
            embedding,
            embedding.validity.determinant_lower * 0.5,
        )
        with pytest.raises(RestrictedDelaunayCoverageUnavailable):
            validate_restricted_delaunay_boundary_witness(
                construction.mesh,
                construction.geometry,
                source,
                construction.certification.cell_regions,
                distance_witness,
                coverage_witness,
                embedding=forged_embedding,
            )
        # Witness identities are static fields; forge one through the constructor.
        forged_binding = RestrictedDelaunayBoundaryWitness(
            mesh_id="forged-restricted-mesh",
            geometry_id=coverage_witness.geometry_id,
            topology_id=coverage_witness.topology_id,
            profile_id=coverage_witness.profile_id,
            source_id=coverage_witness.source_id,
            source_revision=coverage_witness.source_revision,
            domain_id=coverage_witness.domain_id,
            distance_witness_id=coverage_witness.distance_witness_id,
        )
        with pytest.raises(RestrictedDelaunayCoverageUnavailable) as forged_witness:
            validate_restricted_delaunay_boundary_witness(
                construction.mesh,
                construction.geometry,
                source,
                construction.certification.cell_regions,
                distance_witness,
                forged_binding,
            )
        assert forged_witness.value.check == "restricted_witness_binding"
        forged_vertex_ids = np.asarray(
            construction.mesh.vertex_global_ids, dtype=np.int64
        ).copy()
        forged_vertex_ids[0] = int(np.max(forged_vertex_ids)) + 1
        forged_mesh = eqx.tree_at(
            lambda value: value.vertex_global_ids,
            construction.mesh,
            jnp.asarray(forged_vertex_ids),
        )
        with pytest.raises(RestrictedDelaunayCoverageUnavailable) as forged_target:
            validate_restricted_delaunay_boundary_witness(
                forged_mesh,
                construction.geometry,
                source,
                construction.certification.cell_regions,
                distance_witness,
                coverage_witness,
            )
        assert forged_target.value.check == "restricted_mesh_identity"
        from phydrax.discretization._cell_geometry import CellGeometrySpec
        from phydrax.discretization._cell_geometry_validity import cell_geometry_id
        from phydrax.discretization._cell_mesh import CellMesh

        source_center = np.asarray(source.center, dtype=np.float64)
        target_coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
        off_source_mesh = CellMesh.from_tetrahedra(
            source_center + 3.0 * (target_coordinates - source_center),
            np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int32),
        )
        off_source_geometry = CellGeometrySpec.affine(off_source_mesh)
        off_source_connectivity = off_source_mesh.connectivity
        if not isinstance(off_source_connectivity, TetrahedralConnectivity):
            raise AssertionError("Off-source tetrahedral mesh lost its connectivity.")
        off_source_faces = off_source_mesh.entity_set(2)
        off_source_boundary_ids = np.asarray(off_source_faces.entity_ids, dtype=np.int64)[
            np.asarray(off_source_connectivity.boundary_faces, dtype=np.bool_)
        ]
        off_source_distance = SourceBoundaryDistanceWitness(
            mesh_id=off_source_mesh.mesh_id,
            geometry_id=cell_geometry_id(off_source_geometry),
            topology_id=off_source_mesh.topology_id,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_query_id=source.profile_id,
            sample_count=distance_witness.sample_count,
            cover_radius=distance_witness.cover_radius,
            target_facet_global_ids=off_source_boundary_ids,
        )
        off_source_coverage = RestrictedDelaunayBoundaryWitness(
            mesh_id=off_source_mesh.mesh_id,
            geometry_id=cell_geometry_id(off_source_geometry),
            topology_id=off_source_mesh.topology_id,
            profile_id=source.profile_id,
            source_id=source.source_id,
            source_revision=source.source_revision,
            domain_id=source.domain_id,
            distance_witness_id=off_source_distance.witness_id,
        )
        with pytest.raises(RestrictedDelaunayCoverageUnavailable) as off_source:
            validate_restricted_delaunay_boundary_witness(
                off_source_mesh,
                off_source_geometry,
                source,
                construction.certification.cell_regions,
                off_source_distance,
                off_source_coverage,
            )
        assert off_source.value.check == "analytic_projection_tube"
    else:
        assert isinstance(source, NativeImplicitSource)
        assert discovery is not None
        assert construction.certification.domain is discovery
        assert construction.organization.zones[0].name == discovery.region_ids[0]
        witness = construction.certification.fidelity_witness
        assert witness is not None
        fidelity_source = construction.certification.fidelity_source
        assert fidelity_source is not None
        assert isinstance(witness, SourceBoundaryChartWitness)
        assert witness.source_query_id == discovery.boundary_source.source_revision
        source_surface_key = source.source_entity_key_for_scope(
            2, int(source.scope_entity_ids(2)[0])
        )
        assert witness.source_entity_ids == (source_surface_key[0],) * len(
            witness.source_chart_ids
        )
        assert witness.occurrence_paths == (source_surface_key[1],) * len(
            witness.source_chart_ids
        )
        assert np.asarray(witness.implicit_cell_keys).shape == (
            boundary.mesh.blocks[0].cell_count,
            4,
        )
        witness.require_current(
            construction.mesh,
            construction.geometry,
            fidelity_source,
        )
    assert any(
        patch.name == "implicit-zero-set" for patch in construction.organization.patches
    )
    source_surface_id = int(source.scope_entity_ids(2)[0])
    source_surface_key = source.source_entity_key_for_scope(2, source_surface_id)
    boundary_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id
        == construction.mesh.entity_set(2).entity_set_id
    )
    assert source_surface_key[0] in boundary_association.source_entity_ids
    assert source_surface_key[1] in boundary_association.source_occurrence_paths
    assert boundary_association.complete
    assert all(
        association.source_id == source.source_id
        and association.source_revision == source.source_revision
        for association in construction.organization.associations
    )
    assert construction.compliance.passed


def _adaptive_volume_construction(
    schedule: NativeVolumeSchedule, /
) -> tuple[NativeMeshConstruction, AdaptiveImplicitSurface]:
    geometry, grid = _geometry_and_grid()
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08, root_tolerance=1.0e-8
    )
    discovery = discover_adaptive_implicit_surface(
        geometry, domain=_grid_domain(grid), policy=policy, source_id="sheet-sphere"
    )
    # The sphere distance field is singular at its centre; the exact profile
    # proves that singular set lies away from the zero-set cover.
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="sheet-sphere",
        source_revision="r1",
    )
    source = NativeImplicitSource(
        geometry, "sheet-sphere", "r1", discovery=discovery, authoring_profile=profile
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        MeshingScope(
            source.source_id,
            source.source_revision,
            MeshingEntityKind.GEOMETRY,
            2,
            source.scope_entity_set_id(2),
            source.scope_entity_ids(2),
        ),
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=0.8,
    )
    options = NativeMeshingOptions(
        "implicit_adaptive_tetrahedral", implicit_policy=policy, volume_schedule=schedule
    )
    construction = prepare_native_implicit(
        source, specification, options, SpatialCoordinateContract.si()
    ).execute()
    return construction, discovery


def test_native_implicit_carrier_remesh_reports_fixed_fill_quality() -> None:
    construction, _ = _adaptive_volume_construction(NativeVolumeSchedule())
    achieved = dict(construction.compliance.achieved)

    # The scaffold carrier cannot meet the FIXED-fill bound; the remesh can.
    assert achieved["carrier_remeshed"] == 1.0
    assert achieved["carrier_maximum_radius_edge_ratio_before"] > 2.0
    assert achieved["carrier_maximum_radius_edge_ratio_after"] <= 2.0
    assert (
        achieved["carrier_minimum_angle_degrees_after"]
        > achieved["carrier_minimum_angle_degrees_before"]
    )
    assert achieved["source_fidelity_bound"] <= 0.8
    assert construction.compliance.passed


def _duplicated_sheet(
    construction: NativeMeshConstruction, /
) -> tuple[CellMesh, CellGeometrySpec, SourceBoundaryChartWitness]:
    """The volume twice over (disjoint coincident copies) and its doubled witness."""
    mesh = construction.mesh
    witness = construction.certification.fidelity_witness
    assert isinstance(witness, SourceBoundaryChartWitness)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    block = mesh.blocks[0]
    cells = np.asarray(block.vertices, dtype=np.int64)
    doubled = CellMesh.from_tetrahedra(
        np.concatenate((coordinates, coordinates)),
        np.concatenate((cells, cells + coordinates.shape[0])),
    )
    doubled_geometry = CellGeometrySpec.affine(doubled)
    reference = reference_cell_topology(block.cell_kind)
    target = reference_cell_topology(doubled.blocks[0].cell_kind)
    target_vertices = np.asarray(target.vertices, dtype=np.float64)
    target_faces = tuple(tuple(int(v) for v in face) for face in target.entities[2])
    row_of_cell = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    doubled_ids = np.asarray(doubled.blocks[0].global_ids, dtype=np.int64)
    _, cell_faces = volume_cell_entity_rows(doubled, doubled_ids)
    face_ids = np.asarray(doubled.entity_set(2).entity_ids, dtype=np.int64)
    columns: dict[str, list[object]] = {
        "cells": [],
        "entities": [],
        "locals": [],
        "references": [],
    }
    for copy in range(2):
        for row in range(witness.cell_global_ids.size):
            cell_row = row_of_cell[int(witness.cell_global_ids[row])]
            # Cell-local vertex slots of the witnessed face corners, in order.
            slots = tuple(
                int(
                    np.flatnonzero(
                        np.all(
                            np.asarray(reference.vertices, dtype=np.float64)
                            == corner[None, :],
                            axis=1,
                        )
                    )[0]
                )
                for corner in np.asarray(witness.target_reference_vertices[row])
            )
            local = next(
                index
                for index, face in enumerate(target_faces)
                if set(face) == set(slots)
            )
            doubled_row = cell_row + copy * cells.shape[0]
            columns["cells"].append(int(doubled_ids[doubled_row]))
            columns["entities"].append(int(face_ids[cell_faces[doubled_row, local]]))
            columns["locals"].append(local)
            columns["references"].append(target_vertices[list(slots)])
    doubled_witness = SourceBoundaryChartWitness(
        mesh_id=doubled.mesh_id,
        geometry_id=cell_geometry_id(doubled_geometry),
        topology_id=doubled.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(columns["cells"], dtype=np.int64),
        entity_global_ids=np.asarray(columns["entities"], dtype=np.int64),
        local_entity_indices=np.asarray(columns["locals"], dtype=np.int32),
        target_reference_vertices=np.asarray(columns["references"], dtype=np.float64),
        source_chart_ids=witness.source_chart_ids * 2,
        source_entity_ids=witness.source_entity_ids * 2,
        occurrence_paths=witness.occurrence_paths * 2,
        source_parameters=np.concatenate((np.asarray(witness.source_parameters),) * 2),
        orientations=np.concatenate((np.asarray(witness.orientations),) * 2),
        implicit_cell_keys=np.concatenate((np.asarray(witness.implicit_cell_keys),) * 2),
        implicit_root_edge_keys=np.concatenate(
            (np.asarray(witness.implicit_root_edge_keys),) * 2
        ),
    )
    return doubled, doubled_geometry, doubled_witness


def test_implicit_boundary_degree_refuses_a_duplicated_sheet() -> None:
    # A scaffold carrier within the bound keeps its one-to-one charts; its
    # coincident copy turns every witnessed face into a second sheet.
    construction, _ = _adaptive_volume_construction(
        NativeVolumeSchedule(radius_edge_bound=8.0, minimum_dihedral_degrees=0.0)
    )
    source = construction.certification.fidelity_source
    assert isinstance(source, _NativeImplicitBoundaryQuery)

    doubled, doubled_geometry, doubled_witness = _duplicated_sheet(construction)
    with pytest.raises(ValueError, match="duplicates an exact scaffold triangle"):
        prepare_adaptive_implicit_root_charts(
            doubled,
            doubled_geometry,
            source,
            doubled_witness,
            tolerance=0.8,
            maximum_normal_deviation=np.pi,
        )


def test_remeshed_implicit_carrier_certifies_charts_and_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Remeshed vertices are generally not their witnessed scaffold roots, so
    # those chart corners lift from their points and coverage proves degree
    # one from one certified preimage per component.
    geometry, grid = _geometry_and_grid()
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08, root_tolerance=1.0e-8
    )
    discovery = discover_adaptive_implicit_surface(
        geometry, domain=_grid_domain(grid), policy=policy, source_id="sheet-sphere"
    )
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="sheet-sphere",
        source_revision="r1",
    )
    source = NativeImplicitSource(
        geometry, "sheet-sphere", "r1", discovery=discovery, authoring_profile=profile
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=0.8,
    )
    result = (
        NativeMeshingProvider(
            NativeMeshingOptions(
                "implicit_adaptive_tetrahedral",
                implicit_policy=policy,
                volume_schedule=NativeVolumeSchedule(radius_edge_bound=2.0),
            )
        )
        .plan(source, specification, coordinate_contract=SpatialCoordinateContract.si())
        .execute()
    )
    achieved = dict(result.compliance.achieved)

    assert achieved["carrier_remeshed"] == 1.0
    assert achieved["carrier_maximum_radius_edge_ratio_after"] <= 2.0
    certification = result.certification
    assert certification is not None and certification.passed
    assert certification.fidelity is not None
    assert certification.fidelity.status == "certified"
    assert certification.coverage is not None
    from phydrax.geometry._domain_certification import certify_domain_coverage
    from phydrax.geometry.implicit import _source_charts as charts

    prepare = charts.prepare_adaptive_implicit_root_charts

    def weaker_proof(
        mesh: CellMesh,
        coordinates: CellGeometrySpec,
        query: charts.AdaptiveImplicitSourceQuery,
        witness: SourceBoundaryChartWitness,
        /,
        *,
        frame_id: str | None = None,
        tolerance: float | None = None,
        maximum_normal_deviation: float | None = None,
    ) -> charts.AdaptiveImplicitRootChartMap:
        actual = prepare(
            mesh,
            coordinates,
            query,
            witness,
            frame_id=frame_id,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
        )
        faces = tuple(
            replace(
                face,
                evidence=replace(
                    face.evidence,
                    chart_jacobian_lower=face.evidence.chart_jacobian_lower / 2.0,
                    chart_injectivity_margin=face.evidence.chart_injectivity_margin / 2.0,
                ),
            )
            for face in actual._faces
        )
        weaker = charts.AdaptiveImplicitRootChartMap(
            mesh,
            coordinates,
            query,
            witness,
            actual.frame_id,
            faces,
            actual._bounds,
            actual.tolerance,
            actual.maximum_normal_deviation,
            actual._surface,
            actual._hessian_program,
            actual._branches,
        )
        project = weaker.project_target

        def wider_point(
            row: int, reference: tuple[Fraction, ...]
        ) -> charts.ImplicitRootChartPoint:
            point = project(row, reference)
            return replace(
                point, root_error=float(np.nextafter(point.root_error, np.inf))
            )

        # The real single-preimage proof must still exclude every other face
        # with this wider valid root enclosure; no admission is mocked.
        monkeypatch.setattr(weaker, "project_target", wider_point)
        return weaker

    monkeypatch.setattr(charts, "prepare_adaptive_implicit_root_charts", weaker_proof)
    coverage = certification.coverage
    recomputed = certify_domain_coverage(
        result.mesh,
        result.geometry,
        coverage.domain,
        coverage.cell_regions,
        embedding=coverage.embedding,
        source=coverage.source,
        fidelity=coverage.fidelity,
        source_coverage_witness=coverage.source_coverage_witness,
    )
    assert recomputed.status == "certified"
    assert recomputed.source_region_degree_ids == coverage.source_region_degree_ids
    assert recomputed.certificate_id == coverage.certificate_id


def test_remeshed_implicit_duplicated_sheet_is_refused_by_single_preimage_proof() -> None:
    construction, _ = _adaptive_volume_construction(NativeVolumeSchedule())
    assert dict(construction.compliance.achieved)["carrier_remeshed"] == 1.0
    source = construction.certification.fidelity_source
    assert isinstance(source, _NativeImplicitBoundaryQuery)
    witness = construction.certification.fidelity_witness
    assert isinstance(witness, SourceBoundaryChartWitness)
    mesh = construction.mesh
    geometry = construction.geometry
    single = prepare_adaptive_implicit_root_charts(
        mesh, geometry, source, witness, tolerance=0.8, maximum_normal_deviation=np.pi
    )
    assert single.status == "certified"
    preimages = _single_sheet_preimages(
        mesh, geometry, np.zeros(mesh.blocks[0].cell_count, np.int64), single, witness
    )
    assert len(preimages) == 1

    # Remeshed rows may repeat their seeding scaffold triangle, so the doubled
    # witness is admitted and every chart certifies; only the preimage count
    # sees the coincident second sheet.
    doubled, doubled_geometry, doubled_witness = _duplicated_sheet(construction)
    root_map = prepare_adaptive_implicit_root_charts(
        doubled,
        doubled_geometry,
        source,
        doubled_witness,
        tolerance=0.8,
        maximum_normal_deviation=np.pi,
    )
    assert root_map.status == "certified"
    with pytest.raises(SourceCoverageUnavailable, match="second implicit target sheet"):
        _single_sheet_preimages(
            doubled,
            doubled_geometry,
            np.zeros(doubled.blocks[0].cell_count, np.int64),
            root_map,
            doubled_witness,
        )


def _octahedral_surface(
    center: _Point3,
    radius: Fraction,
    /,
    *,
    vertex_offset: int = 0,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[_Triangle3, ...]]:
    directions: tuple[tuple[int, int, int], ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    rows = {
        (axis, sign): 2 * axis + (0 if sign > 0 else 1)
        for axis in range(3)
        for sign in (-1, 1)
    }
    faces: list[tuple[int, int, int]] = []
    triangles: list[_Triangle3] = []

    def point_for(vertex: int) -> _Point3:
        return (
            center[0] + radius * directions[vertex][0],
            center[1] + radius * directions[vertex][1],
            center[2] + radius * directions[vertex][2],
        )

    for sx, sy, sz in product((-1, 1), repeat=3):
        ids = (rows[(0, sx)], rows[(1, sy)], rows[(2, sz)])
        points = (point_for(ids[0]), point_for(ids[1]), point_for(ids[2]))
        first, second, third = (
            np.asarray(directions[vertex], dtype=np.int64) for vertex in ids
        )
        normal = np.cross(second - first, third - first)
        centroid = first + second + third
        if int(np.dot(normal, centroid)) < 0:
            ids = (ids[0], ids[2], ids[1])
            points = (points[0], points[2], points[1])
        faces.append(
            (
                vertex_offset + ids[0],
                vertex_offset + ids[1],
                vertex_offset + ids[2],
            )
        )
        triangles.append(points)
    return tuple(faces), tuple(triangles)


def test_projection_refuses_whole_face_normal_tangency() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(geometry, SpatialCoordinateContract.si())
    center: _Point3 = (
        Fraction.from_float(float(profile.center[0])),
        Fraction.from_float(float(profile.center[1])),
        Fraction.from_float(float(profile.center[2])),
    )
    radius = Fraction.from_float(profile.radius)
    step = Fraction(1, 100)
    triangle = (
        (center[0] + radius, center[1], center[2]),
        (center[0] + radius + step, center[1], center[2]),
        (center[0] + radius, center[1] + step, center[2]),
    )
    with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="tangent"):
        _prove_face_projection_orientation(
            profile,
            triangle,
            (Fraction(0), Fraction(0), Fraction(1)),
            91,
        )


def test_projection_refuses_missing_band_and_hidden_shell() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(geometry, SpatialCoordinateContract.si())
    center: _Point3 = (
        Fraction.from_float(float(profile.center[0])),
        Fraction.from_float(float(profile.center[1])),
        Fraction.from_float(float(profile.center[2])),
    )
    source_radius = Fraction.from_float(profile.radius)
    complete_faces, _complete_triangles = _octahedral_surface(center, source_radius)
    _revalidate_oriented_surface_topology(complete_faces, 2)

    with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="edges"):
        _revalidate_oriented_surface_topology(complete_faces[:-1], 2)

    inner_radius = source_radius - Fraction(1, 20)
    outer_radius = source_radius + Fraction(1, 20)
    inner_faces, inner_triangles = _octahedral_surface(center, inner_radius)
    outer_faces, outer_triangles = _octahedral_surface(
        center, outer_radius, vertex_offset=6
    )
    with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="components"):
        _revalidate_oriented_surface_topology(inner_faces + outer_faces, 2)
    with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="preimages"):
        _prove_degree_one_normal_fiber(
            profile,
            inner_triangles + outer_triangles,
            Fraction.from_float(profile.reach_lower),
        )


def test_restricted_root_bank_batches_and_retains_strict_evidence() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="restricted-root-bank",
        source_revision="restricted-root-bank-r1",
    )
    edge_points = (
        ((0.5, 0.0, 0.0), (2.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        ((0.55, -0.6, 0.0), (1.0, -0.2, 0.0)),
        ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    endpoint_bounds = np.asarray(
        [((first, first), (second, second)) for first, second in edge_points],
        dtype=np.float64,
    )
    roots_by_edge = _root_edge_bank(profile, endpoint_bounds, 1.0e-3)
    assert tuple(len(roots) for roots in roots_by_edge) == (1, 0, 0, 2)
    for roots in roots_by_edge:
        for (
            parameter_bounds,
            coordinate_bounds,
            value_bounds,
            derivative_bounds,
            orientation,
        ) in roots:
            assert np.all(np.isfinite(parameter_bounds))
            assert np.all(np.isfinite(coordinate_bounds))
            assert np.all(np.isfinite(value_bounds))
            assert np.all(np.isfinite(derivative_bounds))
            assert derivative_bounds[0] > 0.0 or derivative_bounds[1] < 0.0
            assert orientation in (-1, 1)

    tangent_bounds = np.asarray(
        (
            ((-1.0, 0.0, 0.75), (-1.0, 0.0, 0.75)),
            ((1.0, 0.0, 0.75), (1.0, 0.0, 0.75)),
        ),
        dtype=np.float64,
    )[None, ...]
    with pytest.raises(MeshingFailure):
        _root_edge_bank(profile, tangent_bounds, 1.0e-3)


def test_restricted_surface_topology_ignores_unreferenced_volume_vertices() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="restricted-topology-offset-vertices",
        source_revision="restricted-topology-offset-vertices-r1",
    )
    center: _Point3 = (
        Fraction(float(profile.center[0])),
        Fraction(float(profile.center[1])),
        Fraction(float(profile.center[2])),
    )
    faces, _triangles = _octahedral_surface(
        center,
        Fraction.from_float(profile.radius),
        vertex_offset=12,
    )
    assert _topology_matches(profile, np.asarray(faces, dtype=np.int64))


def test_restricted_region_sites_require_strict_side_but_not_cover_clearance() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="restricted-region-site-signs",
        source_revision="restricted-region-site-signs-r1",
    )
    candidates = np.asarray(
        (
            (0.75 - 1.0e-8, 0.0, 0.0),
            (0.75, 0.0, 0.0),
            (0.75 + 1.0e-8, 0.0, 0.0),
            (1.6, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    carrier = np.asarray(
        ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        _strict_region_site_mask(profile, candidates, carrier),
        np.asarray((True, False, True, False), dtype=np.bool_),
    )


def test_restricted_inside_witness_uses_a_certified_cell_star() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="restricted-witness-star",
        source_revision="restricted-witness-star-r1",
    )
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.25, 0.0, 0.0),
            (0.0, 0.25, 0.0),
            (0.0, 0.0, 0.25),
            (0.0, 0.0, -0.25),
        ),
        dtype=np.float64,
    )
    tetrahedra = np.asarray(
        ((0, 1, 2, 3), (0, 1, 2, 4)),
        dtype=np.int64,
    )
    assert _inside_seed_rows(profile, points, tetrahedra) == (0, 1)
    touching_source = points.copy()
    touching_source[3] = (0.75, 0.0, 0.0)
    with pytest.raises(MeshingFailure, match="boundary-free ball"):
        _inside_seed_rows(
            profile,
            touching_source,
            np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        )


def test_restricted_projection_degree_refuses_an_off_source_boundary() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="restricted-off-source",
        source_revision="restricted-off-source-r1",
    )
    center: _Point3 = (
        Fraction(float(profile.center[0])),
        Fraction(float(profile.center[1])),
        Fraction(float(profile.center[2])),
    )
    radius = Fraction.from_float(profile.radius)
    tube = Fraction.from_float(profile.reach_lower)
    _faces, triangles = _octahedral_surface(center, radius + 3 * tube)
    with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="preimages"):
        _prove_degree_one_normal_fiber(profile, triangles, tube)


def test_native_restricted_torus_projection_degree_accepts_complete_volume() -> None:
    contract = SpatialCoordinateContract.si()
    geometry = Torus(
        (0.0, 0.0, 0.0),
        0.6,
        1.2,
        feature_id="restricted-projection-degree-torus",
    ).compile()
    profile = AnalyticImplicitProfile(
        geometry,
        contract,
        source_id="restricted-projection-degree-torus",
        source_revision="restricted-projection-degree-torus-revision",
    )
    scope = MeshingScope(
        profile.source_id,
        profile.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        profile.scope_entity_set_id(2),
        profile.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=0.8,
    )
    construction = prepare_native_implicit(
        profile,
        specification,
        NativeMeshingOptions(
            "implicit_restricted_delaunay",
            volume_schedule=NativeVolumeSchedule(),
        ),
        contract,
    ).execute()
    distance_witness = construction.certification.fidelity_witness
    coverage_witness = construction.certification.source_coverage_witness
    assert isinstance(distance_witness, SourceBoundaryDistanceWitness)
    assert isinstance(coverage_witness, RestrictedDelaunayBoundaryWitness)
    regions = construction.certification.cell_regions
    assert regions is not None
    proof = validate_restricted_delaunay_boundary_witness(
        construction.mesh,
        construction.geometry,
        profile,
        regions,
        distance_witness,
        coverage_witness,
        embedding=construction.validated_global_embedding,
    )
    assert proof.region_ids == ("interior",)
    assert proof.target_facet_count > 0


def test_qualification_implicit_volume_preserves_hard_controls() -> None:
    from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
    from tools.meshing_qualification.scenarios.native_implicit import _volume_spec

    profile = AnalyticImplicitProfile(
        Sphere((0.0, 0.0, 0.0), 0.371, feature_id="qualification-control-test").compile(),
        SpatialCoordinateContract.si(),
        source_id="qualification-control-test",
    )
    specification = _volume_spec(profile, 0.371 / 2.0, 0.02)
    control = specification.size_controls[0]
    assert isinstance(control, UniformSizeControl)
    assert control.target_size == 0.371 / 2.0
    assert control.maximum_size == 1.5 * control.target_size
    assert control.strength is SizeControlStrength.HARD
    assert specification.size_compliance.relative_tolerance == 0.5
    assert specification.maximum_deviation == 0.02
    assert specification.protected_features[0].maximum_deviation == 0.02
    assert specification.protected_features[0].hard


def test_restricted_implicit_volume_refuses_soft_shape_strength() -> None:
    from phydrax.meshing._sizing import SizeControlStrength

    contract = SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        Sphere((0.0, 0.0, 0.0), 0.5, feature_id="restricted-soft-shape").compile(),
        contract,
        source_id="restricted-soft-shape",
    )
    scope = MeshingScope(
        profile.source_id,
        profile.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        profile.scope_entity_set_id(2),
        profile.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=0.25,
    )
    soft = NativeVolumeSchedule(shape_strength=SizeControlStrength.SOFT)
    with pytest.raises(MeshingFailure, match="hard NativeVolumeSchedule shape strength"):
        prepare_native_implicit(
            profile,
            specification,
            NativeMeshingOptions("implicit_restricted_delaunay", volume_schedule=soft),
            contract,
        )


def _binary64_rows(rows: tuple[tuple[str, str, str], ...], /) -> np.ndarray:
    return np.asarray(
        [[float.fromhex(value) for value in row] for row in rows], dtype=np.float64
    )


def _exact_circumcenter(vertices: np.ndarray, /) -> tuple[Fraction, ...]:
    points = [[Fraction(float(value)) for value in row] for row in vertices]
    matrix = [[2 * (points[k][j] - points[0][j]) for j in range(3)] for k in (1, 2, 3)]
    rhs = [
        sum(
            (points[k][j] ** 2 - points[0][j] ** 2 for j in range(3)),
            Fraction(0),
        )
        for k in (1, 2, 3)
    ]

    def determinant(rows: list[list[Fraction]], /) -> Fraction:
        return (
            rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
        )

    denominator = determinant(matrix)
    return tuple(
        determinant(
            [
                [rhs[k] if j == column else matrix[k][j] for j in range(3)]
                for k in range(3)
            ]
        )
        / denominator
        for column in range(3)
    )


def test_restricted_tetra_radius_edge_bounds_the_exact_circumradius_ratio() -> None:
    # LIN counterexample: per-axis maxima against the box midpoint missed the
    # farthest box corner and gave a radius-edge ratio below the exact one.
    vertices = _binary64_rows(
        (
            ("-0x1.0f6f9200fbda4p-2", "-0x1.769daa60f5baap-1", "0x1.512387e8bf764p-2"),
            ("0x1.090416b2e0a50p-3", "-0x1.ccecfbf14a9a8p-3", "-0x1.cc2746e2a2d2cp-1"),
            ("-0x1.7eb7f48721164p-2", "-0x1.fc48c50dd92f2p-1", "0x1.718abda9319ecp-1"),
            ("-0x1.55b66c1792d00p-3", "-0x1.01fb751def2d6p-1", "-0x1.d54fb09bdf930p-1"),
        )
    )
    center_box = _binary64_rows(
        (
            ("0x1.d4f3f03538956p+1", "-0x1.14ff391547e52p+2", "-0x1.597bc8a8cfebap-1"),
            ("0x1.d7f258166139ep+1", "-0x1.11148a13b501ep+2", "-0x1.37c6292aec438p-1"),
        )
    )
    center = _exact_circumcenter(vertices)
    assert all(
        Fraction(float(center_box[0, axis]))
        <= center[axis]
        <= Fraction(float(center_box[1, axis]))
        for axis in range(3)
    )
    radius_edge, _ = _tetra_quality(
        vertices, np.asarray(((0, 1, 2, 3),), dtype=np.int64), center_box[None]
    )
    exact_points = [[Fraction(float(value)) for value in row] for row in vertices]
    radius_squared = sum(
        ((center[axis] - exact_points[0][axis]) ** 2 for axis in range(3)), Fraction(0)
    )
    edge_squared = min(
        sum(
            ((exact_points[i][axis] - exact_points[j][axis]) ** 2 for axis in range(3)),
            Fraction(0),
        )
        for i in range(4)
        for j in range(i + 1, 4)
    )
    assert Fraction(float(radius_edge[0])) ** 2 * edge_squared >= radius_squared


def test_restricted_tetra_minimum_dihedral_is_a_lower_bound() -> None:
    # LIN counterexample: the round-to-nearest arccos estimate published
    # 9.858257410430951 degrees above the exact minimum 9.85825741043092873...
    vertices = _binary64_rows(
        (
            ("-0x1.3a60b06fe380cp-2", "0x1.7ed014d8791d4p-1", "-0x1.ba55e34ca6b00p-4"),
            ("0x1.1fff288168c58p-3", "-0x1.a550c4951489cp-2", "0x1.147b417c09fdep-1"),
            ("-0x1.6afb014635b1cp-1", "-0x1.6f6c6135b2280p-1", "0x1.49f89d844745cp-1"),
            ("0x1.877272ae95a10p-2", "-0x1.a552c30cd75bcp-1", "0x1.21b0e9e3d49e0p-1"),
        )
    )
    center_box = _binary64_rows(
        (
            ("-0x1.41aa192e6874dp-1", "-0x1.8eaf3facb09c6p+0", "-0x1.25a6a5154d26cp+1"),
            ("-0x1.42277a889fc3fp-8", "-0x1.c92cf1fa291cap-1", "-0x1.9cf5b5a42c34cp+0"),
        )
    )
    _, minimum_dihedral = _tetra_quality(
        vertices, np.asarray(((0, 1, 2, 3),), dtype=np.int64), center_box[None]
    )
    with mpmath.workprec(256):
        points = [[mpmath.mpf(float(value)) for value in row] for row in vertices]
        normals = []
        for face in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
            origin, first, second = (points[index] for index in face)
            u = [first[axis] - origin[axis] for axis in range(3)]
            v = [second[axis] - origin[axis] for axis in range(3)]
            normals.append(
                (
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                )
            )
        exact = min(
            180
            - mpmath.degrees(
                mpmath.acos(
                    mpmath.fdot(normals[i], normals[j])
                    / mpmath.sqrt(
                        mpmath.fdot(normals[i], normals[i])
                        * mpmath.fdot(normals[j], normals[j])
                    )
                )
            )
            for i in range(4)
            for j in range(i + 1, 4)
        )
        assert 0.0 < minimum_dihedral
        assert mpmath.mpf(minimum_dihedral) <= exact
        assert exact - mpmath.mpf(minimum_dihedral) < mpmath.mpf(2) ** -30


def test_restricted_facet_diameter_bound_encloses_negative_differences() -> None:
    # WIN-5 C6: ``nextafter(p - q, +inf)`` shrinks negative components, so the
    # edge-length bound 0x1.81cde9998a48ap-9 was below the exact length. The
    # third vertex lies near the midpoint, so ``p - q`` is the longest edge.
    first, second = _binary64_rows(
        (
            ("-0x1.48456c21b0cf2p-1", "0x1.9b4987011ef3ep-1", "-0x1.2773cb11b4cdep-1"),
            ("-0x1.4770d3afb54ebp-1", "0x1.9b32525efa419p-1", "-0x1.2632afb2e198ap-1"),
        )
    )
    triangle = np.stack((first, second, first + 0.5 * (second - first)))
    diameter = _triangle_diameter_upper(triangle[None])
    exact = max(
        sum(
            (
                (Fraction(float(a)) - Fraction(float(b))) ** 2
                for a, b in zip(triangle[row], triangle[(row + 1) % 3], strict=True)
            ),
            Fraction(0),
        )
        for row in range(3)
    )
    bound = Fraction(float(diameter[0]))
    assert exact <= bound * bound <= exact * (1 + Fraction(1, 2**47))
