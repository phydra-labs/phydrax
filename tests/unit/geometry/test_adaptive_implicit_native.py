#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.geometry._contracts import CompiledGeometry
from phydrax.geometry._mesh_certificates import MeshCertificateBinding
from phydrax.geometry.implicit._adaptive_discovery import (
    AdaptiveImplicitBoxIssue,
    AdaptiveImplicitCover,
    AdaptiveImplicitScaffold,
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfaceArchiveRecipe,
    AdaptiveImplicitSurfaceStatus,
    discover_adaptive_implicit_surface,
    ImplicitVolumeClass,
    restore_adaptive_implicit_surface,
)
from phydrax.geometry.implicit._policy import AdaptiveImplicitSurfacePolicy


_G = phx.geometry


def _discover(
    geometry: CompiledGeometry,
    domain: np.ndarray,
    /,
    *,
    relative_cell_diameter: float,
    allow_nonsmooth_field: bool = False,
) -> AdaptiveImplicitSurface:
    return discover_adaptive_implicit_surface(
        geometry,
        domain=domain,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=relative_cell_diameter,
            allow_nonsmooth_field=allow_nonsmooth_field,
        ),
        source_id="adaptive-implicit-test",
    )


def _contains(boxes: np.ndarray, point: np.ndarray) -> bool:
    return bool(np.any(np.all((boxes[:, 0] <= point) & (point <= boxes[:, 1]), axis=1)))


@pytest.fixture(scope="module")
def cover_surface() -> AdaptiveImplicitSurface:
    return _discover(
        _G.Sphere((0.031, -0.019, 0.017), 0.3).compile(),
        np.asarray(((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)), dtype=np.float64),
        relative_cell_diameter=0.25,
    )


def _sampling_cover(
    surface: AdaptiveImplicitSurface,
    boxes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> AdaptiveImplicitCover:
    """Stage numerical cover proposals; no full source certification is claimed."""
    from phydrax.geometry.implicit._adaptive_discovery import _cover_identity

    assert surface.cover is not None
    domain = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
    return AdaptiveImplicitCover(
        boxes,
        lower,
        upper,
        np.full((len(boxes), 3), -np.inf, dtype=np.float64),
        np.full((len(boxes), 3), np.inf, dtype=np.float64),
        domain,
        surface.source_id,
        surface.cover.state_id,
        "interval",
        True,
        True,
        _cover_identity(
            boxes,
            lower,
            upper,
            domain,
            surface.source_id,
            surface.cover.state_id,
            "interval",
            True,
        ),
    )


@pytest.mark.parametrize(
    "geometry",
    (
        _G.Sphere((0.031, -0.019, 0.017), 0.3).compile(),
        _G.Superquadric((0.0, 0.0, 0.0), (0.45, 0.50, 0.55)).compile(),
    ),
)
def test_root_point_schedule_preserves_intervals_and_box_programs(
    geometry: CompiledGeometry,
) -> None:
    from phydrax.geometry.implicit._adaptive_discovery import _FieldEnclosures

    bounds = _FieldEnclosures(geometry, "interval")
    points = np.random.default_rng(141).uniform(-0.7, 0.7, size=(2053, 3))
    points[0] = (-0.0, 0.0, -0.0)
    original_points = bounds.point_values(points)
    original_boxes = bounds.boxes(points[:9] - 0.01, points[:9] + 0.01)
    roots = bounds._root_point_enclosures(1024)
    for original, scheduled in zip(
        original_points, roots.point_values(points), strict=True
    ):
        np.testing.assert_array_equal(original.view(np.uint64), scheduled.view(np.uint64))
    replayed_boxes = bounds.boxes(points[:9] - 0.01, points[:9] + 0.01)
    for name in ("value_lower", "value_upper", "gradient_lower", "gradient_upper"):
        np.testing.assert_array_equal(
            getattr(original_boxes, name).view(np.uint64),
            getattr(replayed_boxes, name).view(np.uint64),
        )
    assert roots.point_values(np.empty((0, 3), dtype=np.float64))[0].shape == (0,)
    with pytest.raises(ValueError):
        roots.point_values(np.asarray(((np.nan, 0.0, 0.0),)))


def test_root_point_schedule_preserves_complete_root_records() -> None:
    from phydrax.geometry.implicit._adaptive_discovery import (
        _FieldEnclosures,
        _root_on_edges,
    )

    center = np.asarray((0.031, -0.019, 0.017), dtype=np.float64)
    geometry = _G.Sphere(center, 0.3).compile()
    bounds = _FieldEnclosures(geometry, "interval")
    directions = np.random.default_rng(142).normal(size=(1031, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    endpoints = np.stack((center + 0.24 * directions, center + 0.36 * directions), axis=1)
    signs = np.broadcast_to(np.asarray((-1, 1), dtype=np.int8), (len(endpoints), 2))
    original = _root_on_edges(bounds, endpoints, signs, 1e-10)
    scheduled = _root_on_edges(
        bounds._root_point_enclosures(1024), endpoints, signs, 1e-10
    )
    for field in range(7):
        first = np.asarray([row[field] for row in original])
        second = np.asarray([row[field] for row in scheduled])
        assert first.dtype == second.dtype and first.shape == second.shape
        if first.dtype == np.dtype(np.float64):
            first, second = first.view(np.uint64), second.view(np.uint64)
        np.testing.assert_array_equal(first, second)


def test_adaptive_enclosure_finds_hidden_component_and_certifies_source_cover() -> None:
    center = np.asarray((0.37, 0.13, -0.36))
    radius = 0.04
    geometry = _G.Sphere(center, radius).compile()
    domain = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))

    coarse = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(domain))
    with pytest.raises(ValueError, match="no surface crossings"):
        _G.discover_implicit_surface(geometry, coarse, source_id="fixed-grid-test")

    result = _discover(geometry, domain, relative_cell_diameter=0.005)
    scaffold = result.scaffold
    diagnostic = {
        "status_flags": result.evidence.status_flags,
        "unresolved_count": result.evidence.unresolved_count,
        "unresolved_issue_counts": np.unique(
            result.evidence.unresolved_issues, return_counts=True
        ),
        "cell_issue_counts": np.unique(scaffold.cell_issues, return_counts=True),
        "root_issue_counts": np.unique(scaffold.root_issues, return_counts=True),
    }
    mesh = result.mesh
    assert mesh is not None, diagnostic
    assert mesh.topology.num_face_components == 1, diagnostic
    assert result.region_ids == ("interior",), diagnostic
    assert result.domain_id

    assert result.cover is not None and result.cover.complete and result.cover.certified
    assert result.evidence.accuracy == "enclosed"
    assert result.topology is None
    assert result.policy.relative_cell_diameter == 0.005
    assert isinstance(result.scaffold, AdaptiveImplicitScaffold)
    assert scaffold.corner_coordinates.shape == scaffold.corner_ids.shape
    assert scaffold.corner_value_lower.shape == scaffold.corner_signs.shape
    assert scaffold.tetrahedra.shape[0] == 6 * scaffold.cell_indices.shape[0]
    assert scaffold.triangle_cell_rows.shape == (mesh.faces.shape[0],)
    assert scaffold.triangle_root_edge_rows.shape == mesh.faces.shape
    assert np.all(scaffold.corner_signs != 0)
    endpoint_negative_positive = (
        (scaffold.root_endpoint_value_upper[:, 0] < 0.0)
        & (scaffold.root_endpoint_value_lower[:, 1] > 0.0)
    ) | (
        (scaffold.root_endpoint_value_upper[:, 1] < 0.0)
        & (scaffold.root_endpoint_value_lower[:, 0] > 0.0)
    )
    assert np.all(endpoint_negative_positive)
    assert np.all(scaffold.root_issues == 0)
    assert any(
        leaf is scaffold.root_edge_ids for leaf in jax.tree_util.tree_leaves(scaffold)
    )
    recipe = result.archive_recipe
    assert isinstance(recipe, AdaptiveImplicitSurfaceArchiveRecipe)
    assert recipe.geometry is result.geometry
    assert np.array_equal(recipe.domain, result.volume.domain)
    assert recipe.policy is result.policy
    assert recipe.source_id == result.source_id
    assert np.array_equal(recipe.mesh_arrays[0], mesh.vertices)
    assert np.array_equal(recipe.mesh_arrays[1], mesh.faces)
    assert np.array_equal(recipe.scaffold_arrays[-1], scaffold.triangle_root_edge_rows)
    source = result.boundary_source
    binding_mesh = CellMesh.from_triangles(mesh.vertices, mesh.faces.astype(np.int32))
    binding_geometry = CellGeometrySpec.affine(binding_mesh)
    binding = MeshCertificateBinding(
        binding_mesh,
        binding_geometry,
        "affine",
        source=source,
    )
    binding.require(binding_mesh, binding_geometry, source=source)
    samples = source.boundary_cover(0.02)
    assert samples.semantics == "certified" and samples.complete
    assert samples.points.shape[0] > 0
    distance = np.abs(np.linalg.norm(samples.points - center, axis=1) - radius)
    assert np.max(distance - samples.point_error) <= 1.0e-12
    assert np.all(np.isfinite(samples.covering_radius))


def test_adaptive_archive_retains_and_reproves_stored_arrays() -> None:
    geometry = _G.Sphere((0.031, -0.019, 0.017), 0.3).compile()
    domain = np.asarray(((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)))
    original = _discover(geometry, domain, relative_cell_diameter=0.08)
    recipe = original.archive_recipe
    from phydrax.meshing._model_codec import restore_model_archive_value

    codec_restored = restore_model_archive_value(recipe)
    assert codec_restored.mesh is not None and original.mesh is not None
    np.testing.assert_array_equal(codec_restored.mesh.faces, original.mesh.faces)
    np.testing.assert_array_equal(
        codec_restored.scaffold.root_points, original.scaffold.root_points
    )
    restored = restore_adaptive_implicit_surface(recipe, geometry, original.policy)
    assert restored is not original
    assert restored.mesh is not None and original.mesh is not None
    assert np.array_equal(restored.mesh.faces, original.mesh.faces)
    assert np.array_equal(restored.scaffold.root_points, original.scaffold.root_points)
    changed = np.array(recipe.scaffold_arrays[-1], copy=True)
    changed[0, 0] = changed[0, 0] + 1
    bad = replace(recipe, scaffold_arrays=(*recipe.scaffold_arrays[:-1], changed))
    with pytest.raises(ValueError):
        restore_model_archive_value(bad)
    with pytest.raises(ValueError):
        restore_adaptive_implicit_surface(bad, geometry, original.policy)


def test_adaptive_archive_replays_scientific_state_without_authoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from phydrax.geometry.implicit import _adaptive_discovery as owner

    geometry = _G.Sphere((0.031, -0.019, 0.017), 0.3).compile()
    domain = np.asarray(((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)))
    original = _discover(geometry, domain, relative_cell_diameter=0.08)
    recipe = original.archive_recipe
    monkeypatch.setattr(
        owner,
        "discover_adaptive_implicit_surface",
        lambda *args, **kwargs: pytest.fail("archive re-authored discovery"),
    )
    leaves = jax.tree_util.tree_leaves(recipe)
    assert all(leaf is not recipe.policy for leaf in leaves)
    assert any(leaf is recipe.scaffold_arrays[-1] for leaf in leaves)
    restored = restore_adaptive_implicit_surface(recipe, geometry, original.policy)
    for group in (
        "mesh_arrays",
        "cover_arrays",
        "evidence_arrays",
        "scaffold_arrays",
        "boundary_arrays",
    ):
        for actual, expected in zip(
            getattr(restored.archive_recipe, group), getattr(recipe, group), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
    points = np.asarray(((0.031, -0.019, 0.017), (0.49, 0.49, 0.49)))
    np.testing.assert_array_equal(
        restored.volume.classify_points(points).classes,
        original.volume.classify_points(points).classes,
    )
    assert restored.source_id == original.source_id
    assert restored.volume.query_id == original.volume.query_id
    assert restored.cover is not None and original.cover is not None
    assert restored.cover.cover_id == original.cover.cover_id
    assert restored.evidence.evidence_id == original.evidence.evidence_id

    names = tuple(
        name
        for name in AdaptiveImplicitScaffold.__annotations__
        if name != "target_level"
    )

    def changed_scaffold(
        name: str,
        update: Callable[[np.ndarray], None],
    ) -> AdaptiveImplicitSurfaceArchiveRecipe:
        index = names.index(name)
        values = np.array(recipe.scaffold_arrays[index], copy=True)
        update(values)
        arrays = list(recipe.scaffold_arrays)
        arrays[index] = values
        return replace(recipe, scaffold_arrays=tuple(arrays))

    sealed = np.array(recipe.scaffold_arrays[names.index("root_points")], copy=True)
    sealed.setflags(write=False)
    sealed_arrays = list(recipe.scaffold_arrays)
    sealed_arrays[names.index("root_points")] = sealed
    sealed_recipe = replace(recipe, scaffold_arrays=tuple(sealed_arrays))
    isolated = restore_adaptive_implicit_surface(sealed_recipe, geometry, original.policy)
    first_root = isolated.scaffold.root_points[0, 0]
    sealed.setflags(write=True)
    sealed[0, 0] += 1e-6
    assert isolated.scaffold.root_points[0, 0] == first_root

    immutable_arrays = tuple(
        np.frombuffer(
            np.asarray(values).tobytes(), dtype=np.asarray(values).dtype
        ).reshape(np.shape(values))
        for values in recipe.scaffold_arrays
    )
    immutable_recipe = replace(recipe, scaffold_arrays=immutable_arrays)
    immutable = restore_adaptive_implicit_surface(
        immutable_recipe, geometry, original.policy
    )
    assert np.shares_memory(
        immutable.scaffold.root_points, immutable_arrays[names.index("root_points")]
    )

    root = changed_scaffold(
        "root_points", lambda a: a.__setitem__((0, 0), a[0, 0] + 1e-6)
    )
    sign = changed_scaffold("corner_signs", lambda a: a.__setitem__(0, -a[0]))
    ancestry = changed_scaffold(
        "triangle_root_edge_rows", lambda a: a.__setitem__((0, 0), a[0, 1])
    )
    faces = np.array(recipe.mesh_arrays[1], copy=True)
    faces[0, [1, 2]] = faces[0, [2, 1]]
    contour = replace(recipe, mesh_arrays=(recipe.mesh_arrays[0], faces))
    metadata = recipe.evidence_metadata
    wrong_metadata = replace(
        recipe,
        evidence_metadata=(*metadata[:4], metadata[4] + 1, *metadata[5:]),
    )
    history = list(recipe.evidence_metadata)
    # A later lattice than the authored one, which the history replay refutes.
    history[7] = original.evidence.lattice_anchor + 1
    history[-1] = canonical_fingerprint(
        {
            "kind": "adaptive-implicit-evidence",
            "accuracy": "enclosed",
            "enclosure": recipe.policy.enclosure,
            "status": history[2],
            "unresolved_boxes": array_tree_fingerprint(recipe.evidence_arrays[0]),
            "unresolved_issues": array_tree_fingerprint(recipe.evidence_arrays[1]),
            "cover": recipe.cover_metadata[-1],
            "leaf_count": history[3],
            "lattice_anchor": history[7],
            "declared_domain": array_tree_fingerprint(recipe.evidence_arrays[2]),
        }
    )
    wrong_history = replace(recipe, evidence_metadata=tuple(history))

    # Authenticate a changed cover with freshly computed IDs: hashes are not proof.
    cover = list(recipe.cover_arrays)
    cover[1] = np.array(cover[1], copy=True)
    cover[1][0] = np.nextafter(cover[1][0], -np.inf)
    cover_meta = list(recipe.cover_metadata)
    cover_meta[-1] = owner._cover_identity(
        cover[0],
        cover[1],
        cover[2],
        cover[5],
        recipe.source_id,
        recipe.cover_metadata[1],
        recipe.cover_metadata[2],
        True,
    )
    evidence_meta = list(recipe.evidence_metadata)
    evidence_meta[-1] = canonical_fingerprint(
        {
            "kind": "adaptive-implicit-evidence",
            "accuracy": "enclosed",
            "enclosure": recipe.policy.enclosure,
            "status": evidence_meta[2],
            "unresolved_boxes": array_tree_fingerprint(recipe.evidence_arrays[0]),
            "unresolved_issues": array_tree_fingerprint(recipe.evidence_arrays[1]),
            "cover": cover_meta[-1],
            "leaf_count": evidence_meta[3],
        }
    )
    rehashed = replace(
        recipe,
        cover_arrays=tuple(cover),
        cover_metadata=tuple(cover_meta),
        evidence_metadata=tuple(evidence_meta),
    )
    for tampered in (
        root,
        sign,
        ancestry,
        contour,
        wrong_metadata,
        wrong_history,
        rehashed,
    ):
        with pytest.raises(ValueError):
            restore_adaptive_implicit_surface(tampered, geometry, original.policy)


def test_adaptive_archive_replays_reanchored_discovery_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _G.Superquadric((0.0, 0.0, 0.0), (0.45, 0.50, 0.55)).compile()
    domain = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.12, root_tolerance=1e-8
    )
    original = discover_adaptive_implicit_surface(
        geometry, domain=domain, policy=policy, source_id="archive-reanchored"
    )
    assert original.evidence.lattice_anchor > 0
    from phydrax.geometry.implicit import _adaptive_discovery as owner

    monkeypatch.setattr(
        owner,
        "_discover_on_lattice",
        lambda *args, **kwargs: pytest.fail("re-executed discovery"),
    )
    monkeypatch.setattr(
        owner,
        "_discover_adaptive_implicit_surface_once",
        lambda *args, **kwargs: pytest.fail("re-executed discovery"),
    )
    restored = restore_adaptive_implicit_surface(
        original.archive_recipe, geometry, policy
    )
    assert restored.evidence.lattice_anchor == original.evidence.lattice_anchor
    assert restored.mesh is not None and original.mesh is not None
    np.testing.assert_array_equal(restored.mesh.faces, original.mesh.faces)


@pytest.mark.parametrize(
    "geometry",
    (
        _G.Sphere((0.031, -0.019, 0.017), 0.3).compile(),
        _G.Superquadric((0.0, 0.0, 0.0), (0.45, 0.50, 0.55)).compile(),
    ),
)
def test_streamed_history_attempt_matches_authoring_predicates(
    geometry: CompiledGeometry,
) -> None:
    from phydrax.geometry.implicit import _adaptive_discovery as owner
    from phydrax.geometry.implicit._adaptive_polygon_replay import replay_history_attempt

    domain = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.12, root_tolerance=1e-8
    )
    previous = owner._discover_adaptive_implicit_surface_once(
        geometry,
        domain=domain,
        policy=policy,
        source_id="history-parity",
        declared_domain=domain,
        lattice_anchor=0,
    )
    expected_retry = owner._next_discovery_resolution(previous, policy) is not None
    expected_coincidence = owner._vertex_roots_are_lattice_coincidences(
        geometry, previous, policy.enclosure
    )
    retry, coincidence, tangencies = replay_history_attempt(geometry, domain, policy)
    assert (retry, coincidence) == (expected_retry, expected_coincidence)
    np.testing.assert_array_equal(tangencies, owner._tangency_boxes(previous))


def test_streamed_history_accepts_real_edge_refinement_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry.implicit import _adaptive_discovery as owner
    from phydrax.geometry.implicit._adaptive_polygon_replay import replay_history_attempt

    geometry = _G.Sphere((0.103, 0.17, -0.13), 0.21).compile()
    domain = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.23, root_tolerance=1e-8
    )
    previous = owner._discover_adaptive_implicit_surface_once(
        geometry,
        domain=domain,
        policy=policy,
        source_id="history-retry",
        declared_domain=domain,
        lattice_anchor=0,
    )
    assert owner._next_discovery_resolution(previous, policy) is not None
    assert replay_history_attempt(geometry, domain, policy)[0]
    original = discover_adaptive_implicit_surface(
        geometry, domain=domain, policy=policy, source_id="history-retry"
    )
    assert original.scaffold.target_level > previous.scaffold.target_level
    monkeypatch.setattr(
        owner,
        "_discover_adaptive_implicit_surface_once",
        lambda *args, **kwargs: pytest.fail("re-executed discovery"),
    )
    monkeypatch.setattr(
        owner,
        "_discover_on_lattice",
        lambda *args, **kwargs: pytest.fail("re-executed discovery"),
    )
    restored = restore_adaptive_implicit_surface(
        original.archive_recipe, geometry, policy
    )
    assert restored.mesh is not None and original.mesh is not None
    np.testing.assert_array_equal(restored.mesh.faces, original.mesh.faces)


def test_streamed_history_closes_scratch_and_propagates_io_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    from phydrax.geometry.implicit import _adaptive_polygon_replay as replay

    connect = sqlite3.connect
    closed = []

    class ReadOnlyScratch:
        def __init__(self) -> None:
            self.connection = connect("")
            self.connection.execute("PRAGMA query_only=ON")

        def execute(
            self,
            sql: str,
            parameters: tuple[object, ...] = (),
        ) -> sqlite3.Cursor:
            return self.connection.execute(sql, parameters)

        def close(self) -> None:
            self.connection.close()
            closed.append(True)

    monkeypatch.setattr(replay.sqlite3, "connect", lambda *args: ReadOnlyScratch())
    geometry = _G.Sphere((0.031, -0.019, 0.017), 0.3).compile()
    domain = np.asarray(((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)))
    with pytest.raises(sqlite3.OperationalError):
        replay.replay_history_attempt(geometry, domain, AdaptiveImplicitSurfacePolicy())
    assert closed == [True]


def test_truncated_discovery_domain_does_not_expose_a_volume_region() -> None:
    geometry = _G.Sphere((0.0, 0.0, 0.0), 0.8).compile()
    domain = np.asarray(((-0.4, -0.4, -0.4), (0.4, 0.4, 0.4)))

    result = _discover(geometry, domain, relative_cell_diameter=0.02)

    assert (
        AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING
        in result.evidence.status_flags
    )
    assert result.region_ids == ()


def test_adaptive_tangent_singularity_remains_an_explicit_unresolved_box() -> None:
    geometry = (
        _G.Sphere((-0.3, 0.0, 0.0), 0.3) & _G.Sphere((0.3, 0.0, 0.0), 0.3)
    ).compile()
    domain = np.asarray(((-1.013, -1.013, -1.013), (1.013, 1.013, 1.013)))

    result = _discover(
        geometry,
        domain,
        relative_cell_diameter=0.015,
        allow_nonsmooth_field=True,
    )

    assert result.mesh is None
    assert result.region_ids == ()
    assert result.cover is not None and result.cover.certified
    assert AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES in result.evidence.status_flags
    assert np.any(
        result.evidence.unresolved_issues & int(AdaptiveImplicitBoxIssue.SINGULAR)
    )
    assert _contains(result.evidence.unresolved_boxes, np.zeros(3))


def test_adaptive_resolution_separates_a_thin_gap_between_components() -> None:
    geometry = (
        _G.Sphere((-0.41, 0.013, 0.007), 0.4) | _G.Sphere((0.41, 0.013, 0.007), 0.4)
    ).compile()
    domain = np.asarray(((-1.1, -1.1, -1.1), (1.1, 1.1, 1.1)))

    result = _discover(
        geometry,
        domain,
        relative_cell_diameter=0.004,
        allow_nonsmooth_field=True,
    )

    diagnostic = {
        "status_flags": result.evidence.status_flags,
        "unresolved_issue_counts": np.unique(
            result.evidence.unresolved_issues, return_counts=True
        ),
        "unresolved_boxes": result.evidence.unresolved_boxes,
        "cell_issue_counts": np.unique(result.scaffold.cell_issues, return_counts=True),
        "root_issue_counts": np.unique(result.scaffold.root_issues, return_counts=True),
    }
    assert result.cover is not None and result.cover.complete and result.cover.certified
    assert result.policy.relative_cell_diameter == 0.004
    assert result.evidence.unresolved_count == 0, diagnostic
    assert result.mesh is not None, diagnostic
    gap_midpoint = np.asarray((0.0, 0.013, 0.007))
    assert result.volume.classify_points(gap_midpoint[None, :]).classes.tolist() == [
        int(ImplicitVolumeClass.OUTSIDE)
    ]
    samples = result.boundary_source.boundary_cover(0.01)
    assert samples.complete and samples.semantics == "certified"


# The sphere's x poles (0.25 and 0.5, both exact in binary64) lie on lattice
# planes of [-1, 1]^3 at every level >= 3, and on no lattice vertex: the zero
# set touches a lattice face at each pole, at every refinement.
_POLE_SPHERE = ((0.375, 0.013, -0.021), 0.125)
_POLE_DOMAIN = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
_POLE_POLICY = AdaptiveImplicitSurfacePolicy(
    relative_cell_diameter=0.05, root_tolerance=1.0e-8
)
# Level-5 cells of width 2**-4: each pole lies in the interior of one face of
# exactly one zero-free-cornered cell beyond it.
_POLE_BOXES = np.asarray(
    [
        [[0.1875, 0.0, -0.0625], [0.25, 0.0625, 0.0]],
        [[0.5, 0.0, -0.0625], [0.5625, 0.0625, 0.0]],
    ]
)


def _refuse_hugs(monkeypatch: pytest.MonkeyPatch, /) -> None:
    from phydrax.geometry.implicit import _adaptive_discovery as owner

    monkeypatch.setattr(owner, "_hug_certificate", lambda *args: None)


def test_lattice_face_tangency_is_proved_on_the_touched_face() -> None:
    from phydrax.geometry.implicit import _adaptive_discovery as owner

    geometry = _G.Sphere(*_POLE_SPHERE).compile()
    bounds = owner._FieldEnclosures(geometry, _POLE_POLICY.enclosure)
    # Only the touched faces (upper x of the first box, lower x of the second)
    # confine the zeros; their neighbourhoods are [p - 2**-12, p + 2**-12],
    # widened by one rounding step.
    width = 2.0**-12
    assert owner._tangency_slabs(bounds, _POLE_BOXES) == tuple(
        (0, np.nextafter(plane - width, -np.inf), np.nextafter(plane + width, np.inf))
        for plane in (0.25, 0.5)
    )


def test_lattice_face_tangency_resolves_on_its_cell_pair() -> None:
    from phydrax.geometry.implicit import _adaptive_discovery as owner
    from phydrax.geometry.implicit._adaptive_polygon_replay import (
        replay_history_attempt,
        replay_polygon,
    )

    geometry = _G.Sphere(*_POLE_SPHERE).compile()
    result = discover_adaptive_implicit_surface(
        geometry, domain=_POLE_DOMAIN, policy=_POLE_POLICY, source_id="pole-tangency"
    )
    # Both pole cells are decided on the first lattice, each on the pair of
    # cells across its touched x face, with the x gradient strict on the pair.
    assert result.evidence.lattice_anchor == 0
    assert result.scaffold.target_level == 5
    assert result.evidence.unresolved_count == 0
    resolutions = np.asarray(result.scaffold.tangency_resolutions)
    np.testing.assert_array_equal(
        resolutions[:, 1:],
        [[19, 16, 15, 21, 17, 16, 0], [23, 16, 15, 25, 17, 16, 0]],
    )
    np.testing.assert_array_equal(
        result.scaffold.cell_boxes[resolutions[:, 0]], _POLE_BOXES
    )
    assert np.all(result.scaffold.cell_issues[resolutions[:, 0]] == 0)
    assert not np.any(np.isin(result.scaffold.triangle_cell_rows, resolutions[:, 0]))
    mesh = result.mesh
    assert mesh is not None and mesh.topology.watertight
    faces = np.asarray(mesh.faces)
    edges = {
        tuple(sorted(edge))
        for face in faces.tolist()
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    }
    assert len(mesh.vertices) - len(edges) + len(faces) == 2
    center, radius = _POLE_SPHERE
    distance = np.abs(np.linalg.norm(np.asarray(mesh.vertices) - center, axis=1) - radius)
    assert np.all(distance <= np.max(result.scaffold.root_errors))
    # Each resolved cell keeps a witness like every other band cell: a point
    # of the cell (here next to the pole on its touched face) within its
    # certified bound of the sphere.
    for box in _POLE_BOXES:
        rows = np.flatnonzero(np.all(result.boundary_witness_boxes == box, axis=(1, 2)))
        assert rows.shape == (1,)
        point = result.boundary_witnesses[rows[0]]
        assert np.all((box[0] <= point) & (point <= box[1]))
        bound = result.boundary_errors[rows[0]]
        assert abs(np.linalg.norm(point - center) - radius) <= bound
    bounds = owner._FieldEnclosures(geometry, _POLE_POLICY.enclosure)
    assert replay_polygon(result, bounds)
    assert owner.verify_discovery_history(result)
    retry, coincidence, tangencies = replay_history_attempt(
        geometry, _POLE_DOMAIN, _POLE_POLICY
    )
    assert (retry, coincidence, tangencies.shape) == (False, False, (0, 2, 3))
    # The record is bound by replay: a scaffold without it is refused.
    recipe = result.archive_recipe
    names = [
        name
        for name in AdaptiveImplicitScaffold.__annotations__
        if name != "target_level"
    ]
    arrays = list(recipe.scaffold_arrays)
    arrays[names.index("tangency_resolutions")] = np.empty((0, 8), dtype=np.int64)
    with pytest.raises(ValueError):
        restore_adaptive_implicit_surface(
            replace(recipe, scaffold_arrays=tuple(arrays)), geometry, _POLE_POLICY
        )


def test_refused_lattice_face_tangency_reanchors_without_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry.implicit import _adaptive_discovery as owner
    from phydrax.geometry.implicit._adaptive_polygon_replay import replay_history_attempt

    _refuse_hugs(monkeypatch)
    geometry = _G.Sphere(*_POLE_SPHERE).compile()
    first = owner._discover_adaptive_implicit_surface_once(
        geometry,
        domain=_POLE_DOMAIN,
        policy=_POLE_POLICY,
        source_id="pole-tangency",
        declared_domain=_POLE_DOMAIN,
        lattice_anchor=0,
    )
    np.testing.assert_array_equal(first.evidence.unresolved_boxes, _POLE_BOXES)
    assert (
        first.evidence.unresolved_issues.tolist()
        == [int(AdaptiveImplicitBoxIssue.TANGENCY)] * 2
    )
    assert first.scaffold.tangency_resolutions.shape == (0, 8)
    assert owner._next_discovery_resolution(first, _POLE_POLICY) is None
    retry, coincidence, tangencies = replay_history_attempt(
        geometry, _POLE_DOMAIN, _POLE_POLICY
    )
    assert (retry, coincidence) == (False, False)
    np.testing.assert_array_equal(tangencies, _POLE_BOXES)

    result = discover_adaptive_implicit_surface(
        geometry, domain=_POLE_DOMAIN, policy=_POLE_POLICY, source_id="pole-tangency"
    )
    assert result.evidence.lattice_anchor == 1
    assert result.evidence.unresolved_count == 0
    assert result.mesh is not None and result.mesh.topology.watertight
    # The tangency was not refined: the re-anchored lattice keeps the
    # declared resolution, and none of its planes meets either touched face.
    domain = np.asarray(result.scaffold.domain)
    assert result.scaffold.target_level == 5
    assert owner._lattice_separates(
        domain,
        5,
        tuple(
            (
                0,
                np.nextafter(plane - 2.0**-12, -np.inf),
                np.nextafter(plane + 2.0**-12, np.inf),
            )
            for plane in (0.25, 0.5)
        ),
    )
    assert owner.verify_discovery_history(result)
    monkeypatch.setattr(
        owner,
        "_discover_adaptive_implicit_surface_once",
        lambda *args, **kwargs: pytest.fail("re-executed discovery"),
    )
    restored = restore_adaptive_implicit_surface(
        result.archive_recipe, geometry, _POLE_POLICY
    )
    assert restored.evidence.lattice_anchor == 1
    assert restored.mesh is not None
    np.testing.assert_array_equal(restored.mesh.faces, result.mesh.faces)


def test_straddling_hidden_ball_tangency_is_an_unresolved_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry.implicit import _adaptive_discovery as owner

    # A ball of radius 1e-4 centred 5e-5 beyond the lattice plane x = 0.25
    # pokes 5e-5 into the cell below it, inside that face's slab (2**-12):
    # the cell is a tangency, but every cell group around it holds the
    # ball's centre, where no gradient component is strict, so none is proved.
    monkeypatch.setattr(owner, "_LATTICE_ANCHORS", ())
    geometry = _G.Sphere((0.25 + 5.0e-5, 0.013, -0.021), 1.0e-4).compile()
    result = discover_adaptive_implicit_surface(
        geometry, domain=_POLE_DOMAIN, policy=_POLE_POLICY, source_id="hidden-ball"
    )
    assert result.evidence.lattice_anchor == 0
    assert result.scaffold.target_level == 5
    assert result.mesh is None
    assert result.scaffold.tangency_resolutions.shape == (0, 8)
    issues = result.evidence.unresolved_issues.tolist()
    assert int(AdaptiveImplicitBoxIssue.TANGENCY) in issues
    tangent = result.evidence.unresolved_boxes[
        result.evidence.unresolved_issues == int(AdaptiveImplicitBoxIssue.TANGENCY)
    ]
    np.testing.assert_array_equal(
        tangent, [[[0.1875, 0.0, -0.0625], [0.25, 0.0625, 0.0]]]
    )
    assert owner.verify_discovery_history(result)


def test_discovery_buffer_adoption_requires_sealed_owning_storage() -> None:
    from phydrax.geometry.implicit._adaptive_discovery import (
        _discovery_array,
        _PACKED_OWNERSHIP,
    )

    source = np.arange(6, dtype=np.float64)
    copied = _discovery_array(source, np.float64, (-1, 3), None)
    source[0] = 99.0
    assert copied[0, 0] == 0.0
    with pytest.raises(ValueError, match="sealed owning"):
        _discovery_array(source, np.float64, (-1, 3), _PACKED_OWNERSHIP)
    source.setflags(write=False)
    with pytest.raises(ValueError, match="sealed owning"):
        _discovery_array(source.reshape(2, 3), np.float64, (-1, 3), _PACKED_OWNERSHIP)
    adopted = _discovery_array(source, np.float64, (-1, 3), _PACKED_OWNERSHIP)
    assert np.shares_memory(adopted, source)
    assert not adopted.flags.writeable


def test_axis_aligned_directed_norm_matches_scalar_reference() -> None:
    from phydrax.geometry.implicit._adaptive_discovery import (
        _norm_upper,
        _norm_upper_rows,
    )

    magnitudes = np.exp2(np.arange(-1022, 1023, 53, dtype=np.float64))
    rows = np.zeros((3 * magnitudes.size + 1, 3), dtype=np.float64)
    for axis in range(3):
        rows[axis * magnitudes.size : (axis + 1) * magnitudes.size, axis] = (
            magnitudes if axis != 1 else -magnitudes
        )
    assert np.array_equal(
        _norm_upper_rows(rows), np.asarray([_norm_upper(row) for row in rows])
    )
    rng = np.random.default_rng(70471)
    random_rows = np.zeros((4096, 3), dtype=np.float64)
    exponents = rng.integers(-1022, 1020, size=len(random_rows))
    random_rows[
        np.arange(len(random_rows)), rng.integers(0, 3, size=len(random_rows))
    ] = rng.uniform(1.0, 2.0, len(random_rows)) * np.exp2(exponents)
    assert np.array_equal(
        _norm_upper_rows(random_rows),
        np.asarray([_norm_upper(row) for row in random_rows]),
    )

    exceptional = np.asarray(
        [
            [np.nextafter(0.0, 1.0), 0.0, 0.0],
            [1.0, -2.0, 3.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert np.array_equal(
        _norm_upper_rows(exceptional),
        np.asarray([_norm_upper(row) for row in exceptional]),
    )


def test_boundary_cover_matches_permuted_witnesses_and_nearest_fallback(
    cover_surface: AdaptiveImplicitSurface,
) -> None:
    from phydrax.geometry.implicit._adaptive_discovery import (
        AdaptiveImplicitBoundarySource,
    )

    first = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    second = np.asarray(((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)))
    near = np.asarray(((-0.1, 0.0, 0.0), (0.9, 1.0, 1.0)))
    far = np.asarray(((10.0, 0.0, 0.0), (11.0, 1.0, 1.0)))
    cover = _sampling_cover(
        cover_surface,
        np.asarray((first, far, second, near, far)),
        np.asarray((-1.0, 1.0, -1.0, -1.0, -1.0)),
        np.asarray((1.0, 2.0, 1.0, 1.0, 1.0)),
    )
    surface = replace(
        cover_surface,
        cover=cover,
        boundary_witness_boxes=np.asarray((second, first)),
        boundary_witnesses=np.asarray(((2.5, 0.5, 0.5), (0.5, 0.5, 0.5))),
        boundary_errors=np.asarray((0.02, 0.01)),
    )
    samples = AdaptiveImplicitBoundarySource._cover_from(surface, 1.8)
    assert not samples.complete and samples.semantics == "sampled"
    assert np.array_equal(samples.points, surface.boundary_witnesses[[1, 0, 1]])
    assert np.array_equal(samples.point_error, np.asarray((0.01, 0.02, 0.01)))
    extent = np.nextafter(first[1] - first[0], np.inf)
    direct = np.nextafter(
        np.sqrt(np.nextafter(np.sum(np.nextafter(extent * extent, np.inf)), np.inf)),
        np.inf,
    )
    point = surface.boundary_witnesses[1]
    delta = np.maximum(
        np.nextafter(np.abs(point - near[0]), np.inf),
        np.nextafter(np.abs(point - near[1]), np.inf),
    )
    squares = np.nextafter(delta * delta, np.inf)
    total = 0.0
    for value in squares:
        total = np.nextafter(total + value, np.inf)
    fallback = np.nextafter(np.sqrt(np.nextafter(total, np.inf)), np.inf)
    assert np.array_equal(samples.covering_radius, np.asarray((direct, direct, fallback)))


def test_boundary_cover_rejects_duplicate_signed_zero_boxes(
    cover_surface: AdaptiveImplicitSurface,
) -> None:
    from phydrax.geometry.implicit._adaptive_discovery import (
        AdaptiveImplicitBoundarySource,
    )

    boxes = np.asarray(
        (((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), ((-0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    surface = replace(
        cover_surface,
        cover=None,
        boundary_witness_boxes=boxes,
        boundary_witnesses=np.asarray(((0.5, 0.5, 0.5), (0.6, 0.5, 0.5))),
        boundary_errors=np.asarray((0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="uniquely identify"):
        AdaptiveImplicitBoundarySource._cover_from(surface, 2.0)


def test_boundary_cover_preserves_order_across_chunks(
    cover_surface: AdaptiveImplicitSurface,
) -> None:
    from phydrax.geometry.implicit._adaptive_discovery import (
        AdaptiveImplicitBoundarySource,
    )

    n = 16385
    lower = np.column_stack((np.arange(n, dtype=np.float64), np.zeros((n, 2))))
    boxes = np.stack((lower, lower + 0.25), axis=1)
    points = lower + 0.125
    errors = np.arange(n, dtype=np.float64) * 1.0e-9
    surface = replace(
        cover_surface,
        cover=_sampling_cover(cover_surface, boxes[::-1], np.full(n, -1.0), np.ones(n)),
        boundary_witness_boxes=boxes,
        boundary_witnesses=points,
        boundary_errors=errors,
    )
    samples = AdaptiveImplicitBoundarySource._cover_from(surface, 1.0)
    assert samples.complete and samples.semantics == "certified"
    assert np.array_equal(samples.points, points[::-1])
    assert np.array_equal(samples.point_error, errors[::-1])
