#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import weakref
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

from phydrax.geometry.implicit._adaptive_discovery import (
    _BoxBounds,
    _Cell,
    _CORNER_OFFSETS,
    _CUBE_EDGES,
    _CUBE_FACES,
    _FieldEnclosures,
    _TETRAHEDRA,
    _triangulate_cell_band,
    AdaptiveImplicitBoxIssue,
)
from phydrax.geometry.implicit._adaptive_native_polygon import contours as native_contours
from phydrax.geometry.implicit._adaptive_packed_frontier import (
    PackedLeaves,
    PackedPolygonLeaves,
)
from phydrax.geometry.implicit._adaptive_packed_polygon import (
    _CHUNK,
    _Chunks,
    _sort_unique_inplace,
    PackedPolygon,
    polygonize,
)
from phydrax.geometry.implicit._adaptive_polygon_replay import (
    _same_float_bits,
    replay_polygon,
)
from phydrax.geometry.implicit._policy import AdaptiveImplicitSurfacePolicy


class _Plane(_FieldEnclosures):
    value_rigorous = True

    def __init__(self, intercept: float) -> None:
        self.intercept = intercept

    def field_values(self, points: np.ndarray) -> np.ndarray:
        return points[:, 0] - self.intercept

    def point_values(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = self.field_values(points)
        return value, value.copy()

    def boxes(self, lower: np.ndarray, upper: np.ndarray) -> _BoxBounds:
        count = len(lower)
        return _BoxBounds(
            lower[:, 0] - self.intercept,
            upper[:, 0] - self.intercept,
            np.tile((1.0, 0.0, 0.0), (count, 1)),
            np.tile((1.0, 0.0, 0.0), (count, 1)),
        )


class _Saddle(_Plane):
    def field_values(self, points: np.ndarray) -> np.ndarray:
        return (points[:, 0] - 0.3125) * (points[:, 1] - 0.3125)

    def boxes(self, lower: np.ndarray, upper: np.ndarray) -> _BoxBounds:
        count = len(lower)
        unbounded = np.full(count, np.inf)
        return _BoxBounds(
            -unbounded,
            unbounded,
            np.full((count, 3), -np.inf),
            np.full((count, 3), np.inf),
        )


def _leaves(
    indices: npt.ArrayLike, issues: npt.ArrayLike | None = None
) -> PackedPolygonLeaves:
    indices_array = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    count = len(indices_array)
    boxes = np.stack((indices_array / 8, (indices_array + 1) / 8), axis=1)
    source = PackedLeaves(
        level=np.full(count, 3, dtype=np.int64),
        index=indices_array,
        boxes=boxes,
        value_lower=np.full(count, -1.0),
        value_upper=np.full(count, 1.0),
        gradient_lower=np.tile((1.0, 0.0, 0.0), (count, 1)),
        gradient_upper=np.tile((1.0, 0.0, 0.0), (count, 1)),
        issue=np.zeros(count, dtype=np.int64)
        if issues is None
        else np.asarray(issues, dtype=np.int64),
        surface_rows=np.arange(count, dtype=np.int64),
        deepest_level=3,
        status=0,
    )
    return PackedPolygonLeaves.from_leaves(source)


def _polygon(leaves: PackedPolygonLeaves, intercept: float) -> PackedPolygon:
    return polygonize(
        _Plane(intercept),
        np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
        3,
        _policy(root_tolerance=0.001),
        leaves,
        "packed-plane-test",
        False,
    )


def _policy(root_tolerance: float = 1.0e-10) -> AdaptiveImplicitSurfacePolicy:
    return AdaptiveImplicitSurfacePolicy(root_tolerance=root_tolerance)


def test_packed_two_cell_topology_has_exact_shared_ancestry() -> None:
    leaves = _leaves(((2, 2, 2), (2, 3, 2)))
    expected_corners = leaves.index[:, None, :] + _CORNER_OFFSETS
    result = _polygon(leaves, 0.3125)
    s = result.scaffold_arrays
    np.testing.assert_array_equal(s["cell_corner_ids"], expected_corners)
    with pytest.raises(RuntimeError, match="consumed"):
        _ = leaves.index
    expected_ids = np.asarray(
        sorted(set(map(tuple, expected_corners.reshape(-1, 3)))), dtype=np.int64
    )
    np.testing.assert_array_equal(s["corner_ids"], expected_ids)
    np.testing.assert_array_equal(s["corner_coordinates"], expected_ids / 8)
    np.testing.assert_array_equal(
        s["corner_sampled_values"], expected_ids[:, 0] / 8 - 0.3125
    )
    np.testing.assert_array_equal(
        s["corner_signs"], np.where(expected_ids[:, 0] == 2, -1, 1)
    )
    corner_lookup = {tuple(index): row for row, index in enumerate(expected_ids)}
    corner_rows = np.asarray(
        [[corner_lookup[tuple(index)] for index in cell] for cell in expected_corners]
    )
    np.testing.assert_array_equal(
        s["tetrahedra"], corner_rows[:, _TETRAHEDRA].reshape(-1, 4)
    )
    np.testing.assert_array_equal(s["tetrahedron_cell_rows"], np.repeat((0, 1), 6))
    expected_edges = np.asarray(
        [((2, y, z), (3, y, z)) for y in (2, 3, 4) for z in (2, 3)]
    )
    np.testing.assert_array_equal(s["root_edge_ids"], expected_edges)
    assert s["triangle_cell_rows"].shape == (4,)
    assert s["triangle_root_edge_rows"].shape == (4, 3)
    for triangle, cell in zip(
        s["triangle_root_edge_rows"], s["triangle_cell_rows"], strict=True
    ):
        edges = s["root_edge_ids"][triangle]
        assert all(
            tuple(endpoint) in set(map(tuple, expected_corners[cell]))
            for edge in edges
            for endpoint in edge
        )
        a, b, c = s["root_points"][triangle]
        assert np.cross(b - a, c - a)[0] > 0
    assert not len(result.unresolved_issues)
    np.testing.assert_array_equal(result.witness_boxes, leaves.boxes)
    np.testing.assert_array_equal(result.witness_values, s["root_points"][[0, 2]])
    assert np.all(s["root_brackets"][:, 0, 0] <= 0.3125)
    assert np.all(s["root_brackets"][:, 1, 0] >= 0.3125)
    assert np.all(s["root_errors"] <= 0.001)
    assert np.all(s["root_issues"] == 0)


def test_polygon_replay_detects_root_ancestry_corruption() -> None:
    leaves = _leaves(((2, 2, 2), (2, 3, 2)))
    result = _polygon(leaves, 0.3125)
    scaffold = SimpleNamespace(
        **result.scaffold_arrays, domain=np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    policy = SimpleNamespace(root_tolerance=0.001)
    surface = SimpleNamespace(
        scaffold=scaffold,
        mesh=None,
        policy=policy,
        feature_edges=np.empty((0, 2), dtype=np.int64),
        boundary_witness_boxes=result.witness_boxes,
        boundary_witnesses=result.witness_values,
        boundary_errors=result.witness_errors,
    )
    assert replay_polygon(surface, _Plane(0.3125))
    altered = scaffold.triangle_root_edge_rows.copy()
    altered[0, 0] = (altered[0, 0] + 1) % len(scaffold.root_edge_ids)
    scaffold.triangle_root_edge_rows = altered
    assert not replay_polygon(surface, _Plane(0.3125))
    scaffold.triangle_root_edge_rows = result.scaffold_arrays["triangle_root_edge_rows"]
    surface.boundary_errors = result.witness_errors.copy()
    surface.boundary_errors[0] += 1.0
    assert not replay_polygon(surface, _Plane(0.3125))


def test_packed_refusal_preserves_surface_order_and_singular_evidence() -> None:
    represented = int(AdaptiveImplicitBoxIssue.REPRESENTABILITY)
    leaves = _leaves(((2, 2, 2), (4, 2, 2), (5, 2, 2)), (0, represented, 0))
    boxes = leaves.boxes.copy()
    result = _polygon(leaves, 0.25)
    # The third leaf's corners share a sign and the plane is monotone in x, so
    # its minimum face x = 5/8 encloses positive values: proved free of the
    # zero set, it is neither refused nor contoured.
    np.testing.assert_array_equal(result.unresolved_boxes, boxes[:2])
    np.testing.assert_array_equal(
        result.unresolved_issues, (int(AdaptiveImplicitBoxIssue.SINGULAR), represented)
    )
    assert not len(result.witness_values)
    assert not len(result.scaffold_arrays["root_edge_ids"])
    assert result.mesh is None


def test_ambiguous_face_retains_refusal_and_root_witness() -> None:
    leaves = _leaves(((2, 2, 2),))
    domain = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    policy = _policy(root_tolerance=0.001)
    bounds = _Saddle(0.0)
    result = polygonize(bounds, domain, 3, policy, leaves, "ambiguous", False)
    assert result.unresolved_issues[0] & int(AdaptiveImplicitBoxIssue.EDGE)
    assert len(result.scaffold_arrays["root_edge_ids"]) == 8
    assert len(result.witness_values) == 1
    scaffold = SimpleNamespace(**result.scaffold_arrays, domain=domain)
    surface = SimpleNamespace(
        scaffold=scaffold,
        mesh=None,
        policy=policy,
        feature_edges=np.empty((0, 2), dtype=np.int64),
        boundary_witness_boxes=result.witness_boxes,
        boundary_witnesses=result.witness_values,
        boundary_errors=result.witness_errors,
    )
    assert replay_polygon(surface, bounds)


def test_inplace_lexical_unique_across_chunk_boundary() -> None:
    rows = np.repeat(np.arange(_CHUNK + 3, dtype=np.int64), 2)
    data = np.column_stack((rows[::-1], rows[::-1] % 5, rows[::-1] % 7)).copy()
    actual = _sort_unique_inplace(data)
    # Construct the reference from the original key sequence, not compacted storage.
    expected = np.asarray([(i, i % 5, i % 7) for i in range(_CHUNK + 3)], dtype=np.int64)
    np.testing.assert_array_equal(actual, expected)


def test_polygon_consumes_private_leaf_auxiliary_without_retaining_arrays() -> None:
    leaves = _leaves(((2, 2, 2),))
    refs = tuple(
        weakref.ref(value)
        for value in (leaves.level, leaves.index, leaves.issue, leaves.surface_rows)
    )
    _polygon(leaves, 0.3125)
    assert all(ref() is None for ref in refs)
    with pytest.raises(RuntimeError, match="consumed"):
        _ = leaves.surface_rows
    assert leaves.boxes.shape == (1, 2, 3)


def test_chunk_take_array_releases_blocks_and_preserves_order() -> None:
    chunks = _Chunks((2,), np.int64)
    values = np.arange((_CHUNK + 3) * 2, dtype=np.int64).reshape(-1, 2)
    chunks.extend(values)
    result = chunks.take_array()
    np.testing.assert_array_equal(result, values)
    assert result.flags.owndata and result.flags.c_contiguous
    assert len(chunks) == 0 and not chunks.blocks


def test_replay_float_identity_distinguishes_signed_zero() -> None:
    assert _same_float_bits(np.asarray([0.0]), np.asarray([0.0]))
    assert not _same_float_bits(np.asarray([0.0]), np.asarray([-0.0]))


def test_native_simple_cases_match_scalar_contour_kernel() -> None:
    rng = np.random.default_rng(4102)
    corners = tuple(map(tuple, _CORNER_OFFSETS))
    handled_count = 0
    for pattern in range(1, 255):
        signs = np.asarray(
            [1 if pattern & (1 << i) else -1 for i in range(8)], dtype=np.int8
        )
        values = signs * rng.uniform(0.3, 2.0, size=8)
        edge_keys = [tuple(sorted((corners[a], corners[b]))) for a, b in _CUBE_EDGES]
        crossing = [i for i, (a, b) in enumerate(_CUBE_EDGES) if signs[a] != signs[b]]
        if len(crossing) not in (3, 4):
            continue
        if any(
            sum(signs[face[i]] != signs[face[(i + 1) % 4]] for i in range(4)) == 4
            for face in _CUBE_FACES
        ):
            continue
        sorted_edges = sorted(edge_keys[i] for i in crossing)
        lookup = {key: row for row, key in enumerate(sorted_edges)}
        root_points = np.asarray(
            [
                np.asarray(a) + rng.uniform(0.2, 0.8) * (np.asarray(b) - a)
                for a, b in sorted_edges
            ],
            dtype=np.float64,
        )
        edge_roots = np.full((1, 12), -1, dtype=np.int64)
        for edge in crossing:
            edge_roots[0, edge] = lookup[edge_keys[edge]]
        root_cache = {
            key: (root_points[row], 0.0, 0, root_points[row], root_points[row], -1, 1)
            for row, key in enumerate(sorted_edges)
        }
        vertices, errors, faces, vertex_edges, triangle_cells = [], [], [], [], []
        cell = _Cell(
            0, (0, 0, 0), np.zeros(3), np.ones(3), -1.0, 1.0, np.zeros(3), np.ones(3)
        )
        policy = _policy()
        issue, _, _ = _triangulate_cell_band(
            cell,
            corners,
            dict(zip(corners, map(int, signs), strict=True)),
            dict(zip(corners, map(float, values), strict=True)),
            {},
            root_cache,
            vertices,
            errors,
            {},
            faces,
            vertex_edges,
            triangle_cells,
            0,
            policy,
        )
        vertex_by_root = np.full(len(sorted_edges), -1, dtype=np.int64)
        for vertex, key in enumerate(vertex_edges):
            vertex_by_root[lookup[key]] = vertex
        handled, counts, triangles, issues = native_contours(
            signs[None],
            values[None],
            np.asarray([[np.zeros(3), np.ones(3)]]),
            edge_roots,
            vertex_by_root,
            root_points,
            np.zeros(len(sorted_edges), dtype=np.int64),
            np.zeros(len(sorted_edges)),
        )
        if handled[0]:
            handled_count += 1
            assert issues[0] == issue
            assert counts[0] == len(faces)
            np.testing.assert_array_equal(triangles[0, : counts[0]], faces)
    assert handled_count > 0


def test_native_five_and_six_edge_contours_match_scalar() -> None:
    rng = np.random.default_rng(5329)
    corners = tuple(map(tuple, _CORNER_OFFSETS))
    handled_by_size = {5: 0, 6: 0}
    for _ in range(8):
        for pattern in range(1, 255):
            signs = np.asarray(
                [1 if pattern & (1 << i) else -1 for i in range(8)], dtype=np.int8
            )
            crossing = [i for i, (a, b) in enumerate(_CUBE_EDGES) if signs[a] != signs[b]]
            if len(crossing) not in handled_by_size:
                continue
            if any(
                sum(int(signs[face[i]] != signs[face[(i + 1) % 4]]) for i in range(4))
                == 4
                for face in _CUBE_FACES
            ):
                continue
            values = signs * rng.uniform(0.3, 2.0, size=8)
            edge_keys = [tuple(sorted((corners[a], corners[b]))) for a, b in _CUBE_EDGES]
            sorted_edges = sorted(edge_keys[i] for i in crossing)
            lookup = {key: row for row, key in enumerate(sorted_edges)}
            points = np.asarray(
                [
                    np.asarray(a) + rng.uniform(0.15, 0.85) * (np.asarray(b) - a)
                    for a, b in sorted_edges
                ]
            )
            edge_roots = np.full((1, 12), -1, dtype=np.int64)
            for edge in crossing:
                edge_roots[0, edge] = lookup[edge_keys[edge]]
            root_cache = {
                key: (points[row], 0.0, 0, points[row], points[row], -1, 1)
                for row, key in enumerate(sorted_edges)
            }
            vertices, errors, faces, vertex_edges, triangle_cells = [], [], [], [], []
            cell = _Cell(
                0, (0, 0, 0), np.zeros(3), np.ones(3), -1.0, 1.0, np.zeros(3), np.ones(3)
            )
            issue, _, _ = _triangulate_cell_band(
                cell,
                corners,
                dict(zip(corners, map(int, signs), strict=True)),
                dict(zip(corners, map(float, values), strict=True)),
                {},
                root_cache,
                vertices,
                errors,
                {},
                faces,
                vertex_edges,
                triangle_cells,
                0,
                _policy(),
            )
            vertex_by_root = np.full(len(sorted_edges), -1, dtype=np.int64)
            for vertex, key in enumerate(vertex_edges):
                vertex_by_root[lookup[key]] = vertex
            handled, counts, triangles, issues = native_contours(
                signs[None],
                values[None],
                np.asarray([[np.zeros(3), np.ones(3)]]),
                edge_roots,
                vertex_by_root,
                points,
                np.zeros(len(sorted_edges), dtype=np.int64),
                np.zeros(len(sorted_edges)),
            )
            if handled[0]:
                handled_by_size[len(crossing)] += 1
                assert int(issues[0]) == issue
                assert counts[0] == len(faces)
                np.testing.assert_array_equal(triangles[0, : counts[0]], faces)
    assert all(handled_by_size.values()), handled_by_size


@pytest.mark.parametrize("displacement", (0.0, 1.0e-12, 0.05))
def test_translated_tiny_quad_diagonal_matches_scalar(displacement: float) -> None:
    base = np.asarray((0.417, -0.319, 0.173))
    step = 0.001
    corners = tuple(map(tuple, _CORNER_OFFSETS))
    signs = np.where(_CORNER_OFFSETS[:, 0], 1, -1).astype(np.int8)
    values = signs.astype(np.float64)
    edge_keys = [tuple(sorted((corners[a], corners[b]))) for a, b in _CUBE_EDGES]
    crossing = [i for i, (a, b) in enumerate(_CUBE_EDGES) if signs[a] != signs[b]]
    sorted_edges = sorted(edge_keys[i] for i in crossing)
    lookup = {key: row for row, key in enumerate(sorted_edges)}
    points = np.asarray(
        [
            base + step * (np.asarray(a) + 0.37 * (np.asarray(b) - a))
            for a, b in sorted_edges
        ]
    )
    points[0, 0] += displacement * step
    edge_roots = np.full((1, 12), -1, dtype=np.int64)
    for edge in crossing:
        edge_roots[0, edge] = lookup[edge_keys[edge]]
    root_cache = {
        key: (points[row], 0.0, 0, points[row], points[row], -1, 1)
        for row, key in enumerate(sorted_edges)
    }
    vertices, errors, faces, vertex_edges, triangle_cells = [], [], [], [], []
    cell = _Cell(0, (0, 0, 0), base, base + step, -1.0, 1.0, np.zeros(3), np.ones(3))
    issue, _, _ = _triangulate_cell_band(
        cell,
        corners,
        dict(zip(corners, map(int, signs), strict=True)),
        dict(zip(corners, map(float, values), strict=True)),
        {},
        root_cache,
        vertices,
        errors,
        {},
        faces,
        vertex_edges,
        triangle_cells,
        0,
        _policy(),
    )
    vertex_by_root = np.full(len(sorted_edges), -1, dtype=np.int64)
    for vertex, key in enumerate(vertex_edges):
        vertex_by_root[lookup[key]] = vertex
    handled, counts, triangles, issues = native_contours(
        signs[None],
        values[None],
        np.asarray([[base, base + step]]),
        edge_roots,
        vertex_by_root,
        points,
        np.zeros(len(sorted_edges), dtype=np.int64),
        np.zeros(len(sorted_edges)),
    )
    if displacement == 0.0:
        assert not handled[0]
    if displacement == 0.05:
        assert handled[0]
    if handled[0]:
        assert issues[0] == issue
        np.testing.assert_array_equal(triangles[0, : counts[0]], faces)


def test_native_sphere_polygon_matches_legacy_identity_and_replay() -> None:
    import phydrax as phx
    from phydrax.geometry.implicit._adaptive_discovery import (
        _FieldEnclosures,
        _target_level,
    )
    from phydrax.geometry.implicit._adaptive_packed_frontier import discover_leaves
    from phydrax.geometry.implicit._policy import AdaptiveImplicitSurfacePolicy

    geometry = phx.geometry.Sphere((0.031, -0.017, 0.043), 0.371).compile()
    domain = np.asarray(((-0.7, -0.7, -0.7), (0.7, 0.7, 0.7)))
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.125, root_tolerance=1.0e-10
    )
    bounds = _FieldEnclosures(geometry, policy.enclosure)
    level = _target_level(domain, policy.relative_cell_diameter)
    immutable_leaves = discover_leaves(bounds, domain, level)
    leaves = PackedPolygonLeaves.from_leaves(immutable_leaves)
    del immutable_leaves
    candidate = polygonize(
        bounds, domain, level, policy, leaves, "candidate-differential", True
    )
    assert candidate.mesh is not None
    assert candidate.status == 0
    # Captured before the packed cutover from the original scalar implementation.
    arrays = dict(candidate.scaffold_arrays)
    arrays.update(
        mesh_faces=np.asarray(candidate.mesh.faces),
        mesh_vertices=np.asarray(candidate.mesh.vertices),
        unresolved_boxes=candidate.unresolved_boxes,
        unresolved_issues=candidate.unresolved_issues,
        witness_boxes=candidate.witness_boxes,
        witness_values=candidate.witness_values,
        witness_errors=candidate.witness_errors,
    )
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.asarray(value)
        digest.update(name.encode() + b"\0")
        digest.update(array.dtype.str.encode() + b"\0")
        digest.update(repr(array.shape).encode() + b"\0")
        digest.update(np.ascontiguousarray(array).tobytes())
    assert (
        digest.hexdigest()
        == "1b62bc44858571f296afc6fd8263e886afd1b576b3c3a78fd7d565c7a8736c69"
    )
    scaffold = SimpleNamespace(**candidate.scaffold_arrays, domain=domain)
    surface = SimpleNamespace(
        scaffold=scaffold,
        mesh=candidate.mesh,
        policy=policy,
        feature_edges=np.empty((0, 2), dtype=np.int64),
        boundary_witness_boxes=candidate.witness_boxes,
        boundary_witnesses=candidate.witness_values,
        boundary_errors=candidate.witness_errors,
    )
    assert replay_polygon(surface, bounds)


def test_polygon_batch_interrupt_propagates_without_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry.implicit import _adaptive_native_polygon as native

    def interrupt(*args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(native, "contours", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _polygon(_leaves(((2, 2, 2),)), 0.3125)


@pytest.mark.parametrize("failing_sort", (1, 2))
def test_polygon_mapped_sort_failure_preserves_original_exception(
    monkeypatch: pytest.MonkeyPatch, failing_sort: int
) -> None:
    from phydrax.geometry.implicit import _adaptive_packed_polygon as owner

    original = owner._sort_unique_inplace
    calls = 0

    def fail_with_live_view(rows: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == failing_sort:
            view = rows.view()
            assert view.shape == rows.shape
            raise RuntimeError("mapped sort failure")
        return original(rows)

    monkeypatch.setattr(owner, "_sort_unique_inplace", fail_with_live_view)
    with pytest.raises(RuntimeError, match="mapped sort failure"):
        _polygon(_leaves(((2, 2, 2),)), 0.3125)


def test_triangle_mesh_chunked_validation_preserves_identity_and_error_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry.simplicial import _mesh

    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=np.int32)
    monkeypatch.setattr(_mesh, "_VALIDATION_BATCH_SIZE", 100)
    reference = _mesh.TriangleMesh(vertices, faces, source_id="chunk-identity")
    monkeypatch.setattr(_mesh, "_VALIDATION_BATCH_SIZE", 2)
    chunked = _mesh.TriangleMesh(vertices, faces, source_id="chunk-identity")
    np.testing.assert_array_equal(chunked.vertices, reference.vertices)
    np.testing.assert_array_equal(chunked.faces, reference.faces)
    for name in (
        "edges",
        "halfedge_origin",
        "halfedge_destination",
        "halfedge_face",
        "halfedge_next",
        "halfedge_previous",
        "halfedge_twin",
        "halfedge_edge",
    ):
        np.testing.assert_array_equal(
            getattr(chunked.topology, name), getattr(reference.topology, name)
        )
    assert chunked.source_id == reference.source_id

    collinear = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 0.0, 0.0))
    )
    with pytest.raises(ValueError, match="three distinct"):
        _mesh.TriangleMesh(collinear, ((0, 1, 3), (0, 1, 2), (0, 1, 1)))
    with pytest.raises(ValueError, match="out-of-range"):
        _mesh.TriangleMesh(collinear, ((0, 1, 1), (0, 1, 2), (0, 1, 4)))
    with pytest.raises(ValueError, match="degenerate face"):
        _mesh.TriangleMesh(collinear, ((0, 1, 2), (1, 2, 3), (0, 1, 3)))


def test_triangle_mesh_chunked_validation_preserves_area_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry.simplicial import _mesh

    monkeypatch.setattr(_mesh, "_VALIDATION_BATCH_SIZE", 1)
    threshold = np.finfo(np.float64).eps * 64.0
    for height, accepted in ((threshold, False), (np.nextafter(threshold, np.inf), True)):
        vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, height, 0.0)))
        if accepted:
            assert _mesh.TriangleMesh(vertices, ((0, 1, 2),)).faces.shape == (1, 3)
        else:
            with pytest.raises(ValueError, match="degenerate face"):
                _mesh.TriangleMesh(vertices, ((0, 1, 2),))


def _scalar_and_native_cell(
    values: np.ndarray, scale: float, root_error: float
) -> tuple[int, list[tuple[int, int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Contour one cell of the affine field on both paths with exact roots."""
    corners = tuple(map(tuple, _CORNER_OFFSETS))
    signs = np.where(values > 0.0, 1, -1).astype(np.int8)
    crossing = [i for i, (a, b) in enumerate(_CUBE_EDGES) if signs[a] != signs[b]]
    edge_keys = [tuple(sorted((corners[a], corners[b]))) for a, b in _CUBE_EDGES]
    sorted_edges = sorted(edge_keys[i] for i in crossing)
    lookup = {key: row for row, key in enumerate(sorted_edges)}
    by_corner = dict(zip(corners, values, strict=True))
    points = []
    for a, b in sorted_edges:
        first, second = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        fa, fb = by_corner[a], by_corner[b]
        points.append(scale * (first + (-fa / (fb - fa)) * (second - first)))
    points = np.asarray(points)
    roots = np.full((1, 12), -1, dtype=np.int64)
    for edge in crossing:
        roots[0, edge] = lookup[edge_keys[edge]]
    root_cache = {
        key: (points[row], root_error, 0, points[row], points[row], -1, 1)
        for row, key in enumerate(sorted_edges)
    }
    vertices, errors, faces, vertex_edges, cell_rows = [], [], [], [], []
    cell = _Cell(
        0, (0, 0, 0), np.zeros(3), np.full(3, scale), -1.0, 1.0, np.zeros(3), np.ones(3)
    )
    issue, _, _ = _triangulate_cell_band(
        cell,
        corners,
        dict(zip(corners, map(int, signs), strict=True)),
        dict(zip(corners, map(float, values), strict=True)),
        {},
        root_cache,
        vertices,
        errors,
        {},
        faces,
        vertex_edges,
        cell_rows,
        0,
        _policy(),
    )
    vertex_by_root = np.full(len(sorted_edges), -1, dtype=np.int64)
    for vertex, key in enumerate(vertex_edges):
        vertex_by_root[lookup[key]] = vertex
    handled, counts, triangles, _ = native_contours(
        signs[None],
        values[None],
        np.asarray([[np.zeros(3), np.full(3, scale)]]),
        roots,
        vertex_by_root,
        points,
        np.zeros(len(sorted_edges), dtype=np.int64),
        np.full(len(sorted_edges), root_error),
    )
    return issue, faces, handled, counts, triangles


@pytest.mark.parametrize(
    "scale,root_error,certified",
    (
        (1.0e-80, 0.0, True),
        (1.0, 0.0, True),
        (1.0, 1.0, False),
    ),
)
def test_extended_cycle_orientation_is_certified_against_root_errors(
    scale: float, root_error: float, certified: bool
) -> None:
    from phydrax.geometry.implicit._policy import AdaptiveImplicitBoxIssue

    values = np.sum(_CORNER_OFFSETS, axis=1).astype(np.float64) - 1.2
    issue, faces, handled, counts, triangles = _scalar_and_native_cell(
        values, scale, root_error
    )
    assert len(faces) == 4 * int(certified)
    assert bool(issue & int(AdaptiveImplicitBoxIssue.FACE)) is not certified
    if scale != 1.0 or not certified:
        # Subnormal-scale areas and uncertified triangles are the scalar path's.
        assert not handled[0], (scale, root_error, counts, triangles)
    if handled[0]:
        np.testing.assert_array_equal(triangles[0, : counts[0]], faces)


@pytest.mark.parametrize("root_error,certified", ((1.0e-10, True), (1.0e-6, False)))
def test_lattice_near_coincident_corner_cut_is_decided_by_root_errors(
    root_error: float, certified: bool
) -> None:
    """A lattice corner 1e-7 m across the zero set yields a valid tiny corner cut.

    Its area is about 1e-14 m², below the former absolute floor that no lattice
    at this resolution avoids (the original torus refusal); certified root
    brackets admit it, and only root uncertainty comparable to the cut refuses it.
    """
    from phydrax.geometry.implicit._policy import AdaptiveImplicitBoxIssue

    scale = 0.0027
    values = np.full(8, 1.0e-3)
    values[6] = -4.0e-8
    issue, faces, handled, counts, triangles = _scalar_and_native_cell(
        values, scale, root_error
    )
    assert bool(issue & int(AdaptiveImplicitBoxIssue.FACE)) is not certified
    assert len(faces) == int(certified)
    if handled[0]:
        assert certified
        np.testing.assert_array_equal(triangles[0, : counts[0]], faces)
