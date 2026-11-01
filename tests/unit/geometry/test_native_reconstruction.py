#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import builtins
from types import ModuleType
from typing import Any

import numpy as np
import pytest


def _sphere_samples(seed: int, count: int) -> np.ndarray:
    random = np.random.default_rng(seed)
    points = random.normal(size=(count, 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def _block_external_reconstruction_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> ModuleType:
        if (
            name.split(".", 1)[0] in {"pyvista", "open3d"}
            or name.startswith("scipy.spatial")
            or (name == "scipy" and fromlist is not None and "spatial" in fromlist)
        ):
            raise AssertionError(
                f"native reconstruction attempted forbidden import {name!r}"
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_planar_and_point_routes_need_no_qhull_or_external_surface_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_external_reconstruction_imports(monkeypatch)
    from phydrax.geometry.reconstruction import (
        reconstruct_planar_region,
        reconstruct_point_region,
    )

    planar = reconstruct_planar_region(
        np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)))
    )
    assert planar.report.algorithm == "meshcore_delaunay_2d_native_boundary"

    points = _sphere_samples(3, 160)
    reconstructed = reconstruct_point_region(
        points,
        neighborhood_size=8,
        sample_spacing=0.5,
    )
    report = reconstructed.report
    assert report.algorithm == "native_screened_poisson"
    assert report.watertight and report.winding_consistent
    assert report.connected_components == 1
    assert report.euler_characteristic == 2
    assert report.normal_evidence is not None
    assert report.normal_evidence.route.startswith("bvh-knn-pca")
    assert report.coverage_evidence is not None
    assert report.deviation_evidence is not None
    assert report.poisson_evidence is not None
    assert report.poisson_evidence.discretization == "octree"
    assert report.poisson_evidence.solver_converged


def test_pca_normals_are_oriented_and_reported_on_a_planar_cloud() -> None:
    from phydrax.geometry.reconstruction import estimate_point_normals

    axis = np.linspace(-1.0, 1.0, 7)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((x.reshape(-1), y.reshape(-1), np.zeros(x.size)))
    normals = estimate_point_normals(points, neighborhood_size=8)

    assert normals.evidence.route.startswith("bvh-knn-pca")
    assert normals.evidence.graph_components == 1
    assert normals.evidence.ambiguous_direction_count == 0
    np.testing.assert_allclose(np.abs(normals.normals[:, 2]), 1.0, atol=1.0e-12)
    assert not normals.normals.flags.writeable


def test_terrain_of_three_samples_extrudes_one_closed_prism() -> None:
    from phydrax.geometry.reconstruction import reconstruct_dem_region
    from phydrax.geometry.simplicial._regions import MeshRegion

    samples = np.asarray(((0.02, 0.04, 0.03), (0.05, 0.09, 0.06), (0.01, 0.03, 0.02)))
    terrain = reconstruct_dem_region(
        samples, recenter=False, tolerance=0.0, extrude_depth=0.75
    )
    assert isinstance(terrain.source, MeshRegion)
    faces = np.asarray(terrain.source.faces)
    assert faces.shape == (8, 3)
    assert terrain.report.watertight and terrain.report.euler_characteristic == 2

    with pytest.raises(ValueError, match="at least three samples"):
        reconstruct_dem_region(samples[:2], recenter=False, extrude_depth=0.75)


def test_open_cloud_completion_is_reported_as_inferred_not_observed() -> None:
    from phydrax.geometry.reconstruction import (
        reconstruct_surface_region,
        ReconstructionRobustness,
    )
    from phydrax.geometry.simplicial import MeshRegion

    candidates = _sphere_samples(11, 1200)
    points = candidates[candidates[:, 2] < 0.55]
    assert points.shape[0] > 400
    reconstruction = reconstruct_surface_region(
        points,
        normals=points,
        normal_orientation="supplied",
        neighborhood_size=10,
        sample_spacing=0.35,
        robustness=ReconstructionRobustness(coverage_distance=0.05),
    )

    report = reconstruction.report
    assert report.algorithm == "native_screened_poisson"
    assert report.watertight and report.winding_consistent
    assert report.euler_characteristic is not None
    assert report.poisson_evidence is not None
    assert report.poisson_evidence.discretization == "octree"
    assert report.poisson_evidence.solver_converged
    assert report.normal_evidence is not None
    assert report.coverage_evidence is not None
    assert report.coverage_evidence.unsupported_triangles > 0
    assert report.repair_interpretation == "poisson_inferred_closure"
    assert any("not observed surface" in warning for warning in report.warnings)
    assert report.deviation_evidence is not None
    if not isinstance(reconstruction.source, MeshRegion):
        raise AssertionError("Surface reconstruction must retain its MeshRegion source.")
    assert np.asarray(reconstruction.source.vertices).shape[0] == report.output_vertices


def test_triangle_charts_bind_shared_corners_exactly() -> None:
    import jax.numpy as jnp

    from phydrax.geometry.reconstruction._core import (
        _ReconstructedTrianglePatch,
        _triangle_complex_meshing_domain,
    )

    # 0.7 + (-0.06075031604080055 - 0.7) rounds away from -0.06075031604080055,
    # so an edge-vector chart misses this shared corner by one unit in the last place.
    vertices = np.asarray(
        (
            (0.1, -0.17805949718576636, 0.7),
            (0.3, 0.2, -0.06075031604080055),
            (-0.2, 0.4, 0.1),
            (0.05, 0.1, 0.9),
        ),
        dtype=np.float64,
    )
    assert vertices[0, 2] + (vertices[1, 2] - vertices[0, 2]) != vertices[1, 2]
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)), dtype=np.int64)
    chart_corners = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    for face in faces:
        images = _ReconstructedTrianglePatch(vertices[face]).evaluate(chart_corners)
        np.testing.assert_array_equal(np.asarray(images), vertices[face])
    edges = np.asarray(
        sorted(
            {
                (
                    min(int(face[local]), int(face[local - 1])),
                    max(int(face[local]), int(face[local - 1])),
                )
                for face in faces
                for local in range(3)
            }
        ),
        dtype=np.int64,
    )
    entity_ids = tuple(
        np.arange(count, dtype=np.int64) for count in (4, edges.shape[0], 4)
    )

    arguments = (
        vertices,
        faces,
        edges,
        "exact-corner-tetrahedron",
        "revision-1",
        "binding-1",
        entity_ids,
        0,
        ("reconstructed_vertex", "reconstructed_edge", "reconstructed_triangle"),
    )
    domain = _triangle_complex_meshing_domain(*arguments, None)

    assert domain.accuracy == "exact_represented"
    assert domain.tolerance == 0.0
    np.testing.assert_array_equal(np.asarray(domain.corner_points), vertices)

    # A retained domain is reused only for exactly its construction inputs.
    assert _triangle_complex_meshing_domain(*arguments, domain) is domain
    moved = vertices.copy()
    moved[3, 2] = np.nextafter(moved[3, 2], 1.0)
    rebuilt = _triangle_complex_meshing_domain(moved, *arguments[1:], domain)
    assert rebuilt is not domain
    np.testing.assert_array_equal(np.asarray(rebuilt.corner_points), moved)
    # Same oriented triangle, rotated chart corners: a different construction.
    rotated = faces.copy()
    rotated[0] = np.roll(rotated[0], 1)
    rotated_domain = _triangle_complex_meshing_domain(
        vertices, rotated, *arguments[2:], domain
    )
    assert rotated_domain is not domain
    assert rotated_domain.domain_id != domain.domain_id
    revised = _triangle_complex_meshing_domain(
        *arguments[:4], "revision-2", *arguments[5:], domain
    )
    assert revised is not domain
    assert revised.source_revision == "revision-2"


def test_triangle_complex_domain_host_evaluation_and_currentness() -> None:
    import copy

    import equinox as eqx
    import jax.numpy as jnp

    from phydrax.geometry.reconstruction._core import (
        _ReconstructedTrianglePatch,
        _triangle_complex_meshing_domain,
    )

    vertices = np.asarray(
        (
            (0.1, -0.17805949718576636, 0.7),
            (0.3, 0.2, -0.06075031604080055),
            (-0.2, 0.4, 0.1),
            (0.05, 0.1, 0.9),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)), dtype=np.int64)
    edges = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64)
    domain = _triangle_complex_meshing_domain(
        vertices,
        faces,
        edges,
        "host-evaluation-tetrahedron",
        "revision-1",
        "binding-1",
        tuple(np.arange(count, dtype=np.int64) for count in (4, 6, 4)),
        0,
        ("reconstructed_vertex", "reconstructed_edge", "reconstructed_triangle"),
        None,
    )

    # Host (stacked) chart evaluation is bit-identical to each chart's evaluate.
    generator = np.random.default_rng(7)
    rows = generator.integers(0, faces.shape[0], 64)
    charts = generator.random((64, 2)) * 0.5
    expected = np.stack(
        [
            np.asarray(domain.patches[int(row)].surface.evaluate(jnp.asarray(chart)))
            for row, chart in zip(rows, charts, strict=True)
        ]
    )
    np.testing.assert_array_equal(domain.evaluate(rows, charts), expected)
    # Host trim endpoints are bit-identical to the p-curve's own evaluation.
    for patch in domain.patches:
        for use in patch.loops[0]:
            parameters = (use.first, use.last) if use.sense > 0 else (use.last, use.first)
            np.testing.assert_array_equal(
                use.traversal_endpoints(),
                np.asarray(use.pcurve.evaluate(jnp.asarray(parameters))),
            )

    domain.require_current("host-evaluation-tetrahedron", "revision-1")
    copy.deepcopy(domain).require_current()
    # A pytree rebuild is a different object: its content is fingerprinted again.
    surface = domain.patches[1].surface
    assert isinstance(surface, _ReconstructedTrianglePatch)
    stale = eqx.tree_at(
        lambda value: value.patches[1].surface.corners, domain, surface.corners + 1.0e-3
    )
    with pytest.raises(ValueError, match="content changed"):
        stale.require_current()


def test_triangle_trim_partition_matches_constrained_triangulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractions import Fraction

    from phydrax.geometry import _domain_boundary as boundary

    zero, one = Fraction(0), Fraction(1)
    triangles = (
        ((zero, zero), (one, zero), (zero, one)),
        ((one, zero), (zero, one), (zero, zero)),
        (
            (Fraction(3, 4), Fraction(-1, 8)),
            (Fraction(5, 2), one),
            (-one, Fraction(9, 16)),
        ),
    )
    segments = ((0, 1), (1, 2), (2, 0))

    def partitions() -> list[object]:
        return [
            (
                boundary._exact_trim_triangles(points, segments, (points,)),
                boundary._cover_chord_triangles(0, points, segments, (points,)),
            )
            for points in triangles
        ]

    screened = partitions()
    # Without the triangle screen every trim goes through the native
    # constrained triangulation; the exact partitions are the same tuples.
    monkeypatch.setattr(boundary, "_is_positive_trim_triangle", lambda *_: False)
    assert partitions() == screened == [((points,), (points,)) for points in triangles]
