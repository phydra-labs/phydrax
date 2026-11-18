# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Strict root and box-cover checks at the implicit discovery boundary."""

from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry import Sphere
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingFailure,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import FeatureKind, ProtectedFeature
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import (
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing.providers._native_implicit import (
    _direct_discovery_cover_matches,
    _discovery_roots_match,
    _effective_policy,
    _nearest_scaffold_rows,
    _validate_discovery,
    normalize_analytic_implicit_source,
)
from phydrax.meshing.providers._native_sources import NativeImplicitSource


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_explicit_discovery_resolution_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(
        ValueError, match="relative_cell_diameter must be finite and positive"
    ):
        AdaptiveImplicitSurfacePolicy(relative_cell_diameter=value)


def test_unset_discovery_resolution_is_valid() -> None:
    assert AdaptiveImplicitSurfacePolicy().relative_cell_diameter is None


@pytest.mark.parametrize("cap", [None, 0.06])
def test_analytic_source_discovery_observes_requested_resolution(
    cap: float | None,
) -> None:
    contract = SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        Sphere((0.0, 0.0, 0.0), 0.371).compile(),
        contract,
        source_id="resolution-contract",
    )
    scope = MeshingScope.whole(profile, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.6, strength=SizeControlStrength.HARD),
        ),
        maximum_deviation=0.4,
    )
    source = normalize_analytic_implicit_source(
        profile,
        AdaptiveImplicitSurfacePolicy(relative_cell_diameter=cap),
        contract,
        specification=specification,
    )
    assert source.discovery is not None
    resolution = source.discovery.policy.relative_cell_diameter
    assert resolution is not None
    assert resolution > 0.0
    bounds = np.asarray(profile.boundary_bounds, dtype=np.float64)
    padding = np.max(bounds[1] - bounds[0]) / 8.0
    lower = np.nextafter(bounds[0] - padding, -np.inf)
    upper = np.nextafter(bounds[1] + padding, np.inf)
    old_resolution = 0.2 / float(np.linalg.norm(upper - lower))
    if cap is None:
        assert resolution >= old_resolution
    else:
        assert resolution >= min(old_resolution, cap)
        assert resolution <= cap
    assert source.discovery.policy.root_tolerance <= 0.1
    assert source.discovery.evidence.unresolved_count == 0


@pytest.mark.parametrize("supplied", [False, True])
def test_unspecified_discovery_resolution_uses_specification_candidates(
    supplied: bool,
) -> None:
    scope = MeshingScope(
        "source",
        "revision",
        MeshingEntityKind.GEOMETRY,
        2,
        "faces",
        np.asarray([0], dtype=np.int64),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 1.0, strength=SizeControlStrength.HARD),
            UniformSizeControl(scope, 0.4, strength=SizeControlStrength.SOFT),
        ),
        size_compliance=SizeCompliancePolicy(
            absolute_tolerance=0.1, relative_tolerance=0.1
        ),
        maximum_deviation=1.0,
    )
    # The 3-4-12 box has diagonal 13. The candidates are 0.5/13,
    # 1.2/13 and 0.4/13; neither the legacy 0.001 nor their maximum wins.
    domain = np.asarray(((0.0, 0.0, 0.0), (3.0, 4.0, 12.0)))
    policy = AdaptiveImplicitSurfacePolicy() if supplied else None
    resolved = _effective_policy(policy, specification, domain)
    assert resolved.relative_cell_diameter == pytest.approx(0.4 / 13.0)


def test_explicit_discovery_resolution_is_a_cap() -> None:
    scope = MeshingScope(
        "source",
        "revision",
        MeshingEntityKind.GEOMETRY,
        2,
        "faces",
        np.asarray([0], dtype=np.int64),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(UniformSizeControl(scope, 1.0),),
        maximum_deviation=1.0,
    )
    domain = np.asarray(((0.0, 0.0, 0.0), (3.0, 4.0, 12.0)))
    policy = AdaptiveImplicitSurfacePolicy(relative_cell_diameter=0.01)
    assert _effective_policy(policy, specification, domain).relative_cell_diameter == 0.01

    tighter = AdaptiveImplicitSurfacePolicy(relative_cell_diameter=0.1)
    assert _effective_policy(tighter, specification, domain).relative_cell_diameter == (
        pytest.approx(0.5 / 13.0)
    )


def test_unspecified_discovery_resolution_without_candidates_uses_legacy_default() -> (
    None
):
    scope = MeshingScope(
        "source",
        "revision",
        MeshingEntityKind.GEOMETRY,
        2,
        "faces",
        np.asarray([0], dtype=np.int64),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(),
    )
    domain = np.asarray(((0.0, 0.0, 0.0), (3.0, 4.0, 12.0)))
    assert (
        _effective_policy(
            AdaptiveImplicitSurfacePolicy(), specification, domain
        ).relative_cell_diameter
        == 0.001
    )


def test_root_membership_preserves_bytes_duplicates_and_error_equality() -> None:
    roots = np.array(
        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [-0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    )
    errors = np.array([1.0, 2.0, -0.0, np.inf])
    # Reordering forces the sorted path, including duplicate points and signed zero.
    points = roots[[2, 1, 0, 3]]
    assert _discovery_roots_match(
        roots, errors, points, np.array([0.0, 2.0, 1.0, np.inf])
    )
    assert not _discovery_roots_match(
        roots, errors, points, np.array([0.0, 3.0, 1.0, np.inf])
    )
    altered = points.copy()
    altered[0, 0] = 0.0
    assert not _discovery_roots_match(
        roots, errors, altered, np.array([0.0, 2.0, 1.0, np.inf])
    )
    assert not _discovery_roots_match(
        roots, errors, points, np.array([np.nan, 2.0, 1.0, np.inf])
    )


def test_root_membership_across_batches_with_noncontiguous_points() -> None:
    roots = np.arange(3 * 16386, dtype=np.float64).reshape(-1, 3)
    errors = np.arange(len(roots), dtype=np.float64)
    points = roots[::-1]
    assert _discovery_roots_match(roots, errors, points, errors[::-1])
    bad_errors = errors[::-1].copy()
    bad_errors[16384] = -1.0
    assert not _discovery_roots_match(roots, errors, points, bad_errors)


def test_direct_cover_fast_path_requires_unique_bijective_box_roots() -> None:
    n = 16385
    lower = np.column_stack((np.arange(n, dtype=np.float64), np.zeros((n, 2))))
    boxes = np.stack((lower, lower + 0.25), axis=1)
    points = lower + 0.125
    errors = np.zeros(n)
    cover = SimpleNamespace(
        boxes=boxes[::-1], value_lower=np.full(n, -1.0), value_upper=np.ones(n)
    )
    assert _direct_discovery_cover_matches(cover, points, errors, boxes, 1.0)
    repeated = cover.boxes.copy()
    repeated[0] = repeated[1]
    cover.boxes = repeated
    assert not _direct_discovery_cover_matches(cover, points, errors, boxes, 1.0)
    cover.boxes = boxes[::-1]
    displaced = points.copy()
    displaced[0] += 2.0
    assert not _direct_discovery_cover_matches(cover, displaced, errors, boxes, 1.0)


@pytest.fixture(scope="module")
def implicit_case() -> tuple[NativeImplicitSource, AdaptiveImplicitSurface]:
    geometry = Sphere((0.0, 0.0, 0.0), 0.75).compile()
    source_id, revision = "discovery-fixture", "r1"
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08, root_tolerance=1.0e-8
    )
    surface = discover_adaptive_implicit_surface(
        geometry,
        domain=np.asarray(((-1.4, -1.4, -1.4), (1.4, 1.4, 1.4))),
        policy=policy,
        source_id=source_id,
    )
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        tube_radius=0.03,
        source_id=source_id,
        source_revision=revision,
    )
    return NativeImplicitSource(
        geometry, source_id, revision, discovery=surface, authoring_profile=profile
    ), surface


@pytest.mark.parametrize("forgery", ["source", "cover"])
def test_discovery_refuses_forged_cover(
    implicit_case: tuple[NativeImplicitSource, AdaptiveImplicitSurface], forgery: str
) -> None:
    source, surface = implicit_case
    _validate_discovery(source, surface)
    altered = AdaptiveImplicitSurface(
        surface.mesh,
        surface.evidence,
        None if forgery == "cover" else surface.cover,
        surface.topology,
        surface.volume,
        surface.scaffold,
        surface.feature_edges,
        surface.boundary_witness_boxes,
        surface.boundary_witnesses,
        surface.boundary_errors,
        surface.geometry,
        surface.policy,
        "foreign-source" if forgery == "source" else surface.source_id,
    )
    with pytest.raises(MeshingFailure):
        _validate_discovery(source, altered)


@pytest.mark.parametrize("scaffold_count,carrier_count", [(16389, 7), (7, 4105)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nearest_scaffold_centroids_across_batches_preserve_rows(
    scaffold_count: int, carrier_count: int, dtype: type[np.float32] | type[np.float64]
) -> None:
    rng = np.random.default_rng(4387)
    vertices = rng.normal(size=(3 * max(scaffold_count, carrier_count), 3)).astype(dtype)
    triangles = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
    scaffold_faces = triangles[:scaffold_count].copy()
    scaffold_faces[1] = scaffold_faces[0]
    if scaffold_count > 16384:
        scaffold_faces[-1] = scaffold_faces[0]
    carrier_faces = triangles[:carrier_count]
    source_centroids = np.mean(vertices[scaffold_faces], axis=1)
    carrier_centroids = np.mean(vertices[carrier_faces], axis=1)
    expected = np.argmin(
        np.sum(
            (carrier_centroids[:, None, :] - source_centroids[None, :, :]) ** 2, axis=2
        ),
        axis=1,
    )
    rows = _nearest_scaffold_rows(
        vertices,
        scaffold_faces,
        vertices,
        carrier_faces,
    )
    np.testing.assert_array_equal(rows, expected)
    assert rows[0] == 0


def test_nearest_scaffold_nonfinite_centroids_keep_exhaustive_semantics() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (np.nan, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
        )
    )
    scaffold = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
    carrier = np.asarray(((0, 1, 2),), dtype=np.int32)
    source_centroids = np.mean(vertices[scaffold], axis=1)
    carrier_centroids = np.mean(vertices[carrier], axis=1)
    expected = np.argmin(
        np.sum(
            (carrier_centroids[:, None, :] - source_centroids[None, :, :]) ** 2, axis=2
        ),
        axis=1,
    )
    np.testing.assert_array_equal(
        _nearest_scaffold_rows(vertices, scaffold, vertices, carrier), expected
    )


def test_nearest_scaffold_nan_in_later_shard_and_empty_inputs() -> None:
    vertices = np.zeros((16389 * 3, 3), dtype=np.float64)
    triangles = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    vertices[triangles[16384, 0], 0] = np.nan
    rows = _nearest_scaffold_rows(vertices, triangles, vertices, triangles[:1])
    np.testing.assert_array_equal(rows, np.asarray((16384,), dtype=np.int64))
    vertices[triangles[4096, 0], 0] = np.nan
    vertices[triangles[-1, 0], 0] = np.nan
    rows = _nearest_scaffold_rows(vertices, triangles, vertices, triangles[:1])
    np.testing.assert_array_equal(rows, np.asarray((4096,), dtype=np.int64))
    assert _nearest_scaffold_rows(vertices, triangles, vertices, triangles[:0]).size == 0
    with pytest.raises(ValueError):
        _nearest_scaffold_rows(vertices, triangles[:0], vertices, triangles[:1])


def test_nearest_scaffold_preserves_underflow_ties_before_sampled_row() -> None:
    vertices = np.ones((16385, 3), dtype=np.float64)
    vertices[2] = (1e-200, 0.0, 0.0)
    vertices[-1] = 0.0
    scaffold = np.repeat(np.arange(len(vertices), dtype=np.int64)[:, None], 3, axis=1)
    carrier_vertices = np.zeros((1, 3), dtype=np.float64)
    carrier = np.zeros((1, 3), dtype=np.int64)
    np.testing.assert_array_equal(
        _nearest_scaffold_rows(vertices, scaffold, carrier_vertices, carrier),
        np.asarray((2,), dtype=np.int64),
    )


def test_nearest_scaffold_reduces_unsorted_overlap_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing.providers import _native_implicit as owner

    vertices = np.ones((100, 3), dtype=np.float64)
    vertices[2] = vertices[4] = 0.0
    scaffold = np.repeat(np.arange(len(vertices), dtype=np.int64)[:, None], 3, axis=1)
    carrier_vertices = np.zeros((1, 3), dtype=np.float64)
    carrier = np.zeros((1, 3), dtype=np.int64)

    def blocks(
        *args: object, **kwargs: object
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        assert kwargs["item_block"] == 4096
        for row in (0, 4, 2):
            yield np.asarray((0,), dtype=np.int64), np.asarray((row,), dtype=np.int64)

    monkeypatch.setattr(owner, "bvh_overlap_pair_blocks", blocks)
    np.testing.assert_array_equal(
        owner._nearest_scaffold_rows(vertices, scaffold, carrier_vertices, carrier),
        np.asarray((2,), dtype=np.int64),
    )


def _surface_scope(source: NativeImplicitSource) -> MeshingScope:
    return MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )


def _volume_spec(
    source: NativeImplicitSource,
    maximum_deviation: float | None,
    features: tuple[ProtectedFeature, ...] = (),
) -> VolumeMeshingSpec:
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        _surface_scope(source),
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        protected_features=features,
        maximum_deviation=maximum_deviation,
    )


def test_carrier_band_respects_declared_feature_and_source_bounds(
    implicit_case: tuple[NativeImplicitSource, AdaptiveImplicitSurface],
) -> None:
    from phydrax.meshing.providers import _native_implicit as owner

    source, _ = implicit_case
    assert (
        owner._implicit_carrier_admission_tolerance(source, _volume_spec(source, 0.02))
        == 0.02
    )
    assert (
        owner._implicit_carrier_admission_tolerance(source, _volume_spec(source, 0.05))
        == 0.03
    )
    scope = _surface_scope(source)
    hard = ProtectedFeature(scope, FeatureKind.SURFACE, maximum_deviation=0.005)
    soft = ProtectedFeature(
        scope, FeatureKind.SURFACE, maximum_deviation=0.001, hard=False
    )
    assert (
        owner._implicit_carrier_admission_tolerance(
            source, _volume_spec(source, 0.02, (hard, soft))
        )
        == 0.005
    )
    assert (
        owner._implicit_carrier_admission_tolerance(
            source, _volume_spec(source, None, (hard,))
        )
        == 0.005
    )
    exact = ProtectedFeature(scope, FeatureKind.SURFACE, maximum_deviation=0.0)
    for specification in (
        _volume_spec(source, None, (exact,)),
        _volume_spec(source, None),
        _volume_spec(source, 0.0),
    ):
        with pytest.raises(MeshingFailure):
            owner._implicit_carrier_admission_tolerance(source, specification)


def test_carrier_band_rejects_foreign_feature_scope(
    implicit_case: tuple[NativeImplicitSource, AdaptiveImplicitSurface],
) -> None:
    from phydrax.meshing.providers import _native_implicit as owner

    source, _ = implicit_case
    foreign = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        "foreign-surface-set",
        source.scope_entity_ids(2),
    )
    feature = ProtectedFeature(foreign, FeatureKind.SURFACE, maximum_deviation=0.005)
    with pytest.raises(MeshingFailure):
        owner._implicit_carrier_admission_tolerance(
            source, _volume_spec(source, 0.02, (feature,))
        )


def test_carrier_identity_binds_new_and_untouched_face_bounds(
    implicit_case: tuple[NativeImplicitSource, AdaptiveImplicitSurface],
) -> None:
    from phydrax.meshing.providers import _native_implicit as owner

    source, discovery = implicit_case
    identity = owner._implicit_remeshed_carrier_identity(
        source, discovery, 0.1855, 0.02, 0.0015
    )
    assert identity != owner._implicit_remeshed_carrier_identity(
        source, discovery, 0.1855, 0.01, 0.0015
    )
    assert identity != owner._implicit_remeshed_carrier_identity(
        source, discovery, 0.1855, 0.02, 0.001
    )
