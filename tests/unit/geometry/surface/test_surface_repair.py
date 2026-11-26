#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unit and contract tests for certified native surface mesh repair."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phydrax.geometry.surface import (
    export_surface,
    import_surface,
    repair_surface_arrays,
    repair_surface_mesh,
    SurfaceExportPolicy,
    SurfaceFileFormat,
    SurfaceImportPolicy,
    SurfaceRepairArrays,
    SurfaceRepairError,
    SurfaceRepairPolicy,
)
from phydrax.units import METER


def _box_mesh(
    lo: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hi: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    invert: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    vertices = np.asarray(
        (
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        ),
        dtype=np.int64,
    )
    if invert:
        faces = faces[:, (0, 2, 1)]
    return vertices, faces


def _concat(
    *meshes: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    verts_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    offset = 0
    for verts, faces in meshes:
        verts_list.append(verts)
        faces_list.append(faces + offset)
        offset += verts.shape[0]
    return np.vstack(verts_list), np.vstack(faces_list)


def _icosphere(
    subdivisions: int = 2, radius: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.asarray(
        (
            (-1.0, phi, 0.0),
            (1.0, phi, 0.0),
            (-1.0, -phi, 0.0),
            (1.0, -phi, 0.0),
            (0.0, -1.0, phi),
            (0.0, 1.0, phi),
            (0.0, -1.0, -phi),
            (0.0, 1.0, -phi),
            (phi, 0.0, -1.0),
            (phi, 0.0, 1.0),
            (-phi, 0.0, -1.0),
            (-phi, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ),
        dtype=np.int64,
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    for _ in range(subdivisions):
        edge_mid: dict[tuple[int, int], int] = {}
        new_verts = verts.tolist()
        new_faces: list[tuple[int, int, int]] = []

        def _mid(u: int, v: int) -> int:
            key = (min(u, v), max(u, v))
            existing = edge_mid.get(key)
            if existing is not None:
                return existing
            pt = 0.5 * (verts[u] + verts[v])
            pt = pt / np.linalg.norm(pt)
            idx = len(new_verts)
            new_verts.append(pt.tolist())
            edge_mid[key] = idx
            return idx

        for a, b, c in faces:
            ia, ib, ic = int(a), int(b), int(c)
            ab = _mid(ia, ib)
            bc = _mid(ib, ic)
            ca = _mid(ic, ia)
            new_faces.extend(((ia, ab, ca), (ib, bc, ab), (ic, ca, bc), (ab, bc, ca)))
        verts = np.asarray(new_verts, dtype=np.float64)
        faces = np.asarray(new_faces, dtype=np.int64)
    return verts * radius, faces


def test_default_policy_is_zero_displacement_and_refuses_dirty_soup() -> None:
    policy = SurfaceRepairPolicy()
    assert policy.weld_tolerance == 0.0
    assert not policy.drop_degenerate_triangles
    assert not policy.resolve_self_intersections
    assert not policy.fill_holes
    assert not policy.extract_solid
    assert not policy.orient_shells
    assert not policy.prune_small_components

    v1, f1 = _box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    v2, f2 = _box_mesh((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
    dirty = _concat((v1, f1), (v2, f2))
    with pytest.raises(SurfaceRepairError) as exc_info:
        repair_surface_mesh(dirty, policy=policy)
    assert exc_info.value.status == "embedding_failed"


def test_watertight_solid_repairs_self_intersections_baffles_and_coplanar_duplicates() -> (
    None
):
    v1, f1 = _box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    v2, f2 = _box_mesh((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
    # First check pure closed self-intersecting union + canceling coplanar duplicates: exact Kirchhoff.
    closed_v, closed_f = _concat((v1, f1), (v2, f2))
    closed_f = np.vstack((closed_f, closed_f[0:1], closed_f[0:1, (0, 2, 1)]))
    policy = SurfaceRepairPolicy.watertight_solid()
    exact_result = repair_surface_mesh((closed_v, closed_f), policy=policy)
    assert exact_result.evidence.embedding_verified
    assert exact_result.evidence.closure.boundary_edges == 0
    assert exact_result.evidence.closure.nonmanifold_edges == 0
    assert exact_result.evidence.closure.nonmanifold_vertices == 0
    assert not exact_result.evidence.used_generalized_winding
    assert abs(exact_result.evidence.closure.signed_volume - 1.875) < 1.0e-12

    # Now add a non-cycle internal baffle wall cutting straight across box 1.
    baffle_v = np.asarray(
        ((0.25, 0.0, 0.0), (0.25, 1.0, 0.0), (0.25, 1.0, 1.0), (0.25, 0.0, 1.0)),
        dtype=np.float64,
    )
    baffle_f = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    soup_v, soup_f = _concat((v1, f1), (v2, f2), (baffle_v, baffle_f))
    soup_f = np.vstack((soup_f, soup_f[0:1], soup_f[0:1, (0, 2, 1)]))
    result = repair_surface_mesh((soup_v, soup_f), policy=policy)
    assert result.evidence.embedding_verified
    assert result.evidence.closure.boundary_edges == 0
    assert result.evidence.closure.nonmanifold_edges == 0
    assert result.evidence.closure.nonmanifold_vertices == 0
    assert result.evidence.arrangement_fragments > 0
    assert abs(result.evidence.closure.signed_volume - 1.875) < 1.0e-12
    # Deterministic canonical fingerprints on repeated execution.
    repeat = repair_surface_mesh((soup_v, soup_f), policy=policy)
    assert repeat.result_id == result.result_id
    assert repeat.evidence.evidence_id == result.evidence.evidence_id

    # Host-array entry point matches SurfaceModel output and evidence identically.
    arrays = repair_surface_arrays((soup_v, soup_f), policy=policy)
    assert isinstance(arrays, SurfaceRepairArrays)
    assert arrays.evidence.evidence_id == result.evidence.evidence_id
    assert np.array_equal(
        arrays.vertices, np.asarray(result.surface.mesh.coordinates, dtype=np.float64)
    )
    assert np.array_equal(
        arrays.triangles,
        np.asarray(result.surface.mesh.connectivity.cell_vertices, dtype=np.int64)[:, :3],
    )
    arrays_repeat = repair_surface_arrays((soup_v, soup_f), policy=policy)
    assert arrays_repeat.result_id == arrays.result_id

    # Self-intersecting soup WITH an open hole still runs Stage 4 hole fill + Stage 5a extraction.
    open_v1, open_f1 = _box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    open_f1 = open_f1[2:]  # remove -z face to create a 4-edge open hole
    open_soup = _concat((open_v1, open_f1), (v2, f2))
    open_repaired = repair_surface_arrays(open_soup, policy=policy)
    assert open_repaired.evidence.filled_hole_count >= 1
    assert open_repaired.evidence.embedding_verified
    assert open_repaired.evidence.closure.boundary_edges == 0
    assert abs(open_repaired.evidence.closure.signed_volume - 1.875) < 1.0e-12


def test_displacement_bounded_welding_t_junction_and_degenerate_cleanup() -> None:
    verts, faces = _box_mesh()
    # Crack face 0 by 2e-4 and add a collinear degenerate triangle.
    crack_id = verts.shape[0]
    mid_id = verts.shape[0] + 1
    verts = np.vstack((verts, ((2.0e-4, 0.0, 0.0), (0.5, 0.0, 0.0))))
    faces = faces.copy()
    faces[0, 0] = crack_id
    faces = np.vstack((faces, ((0, mid_id, 1),)))

    policy = SurfaceRepairPolicy(
        weld_tolerance=5.0e-4,
        drop_degenerate_triangles=True,
        require_watertight=True,
        require_manifold=True,
        verify_embedding=True,
    )
    result = repair_surface_mesh((verts, faces), policy=policy)
    assert result.evidence.welded_vertex_count >= 1
    assert result.evidence.dropped_degenerate_faces >= 1
    assert result.evidence.max_vertex_displacement <= 5.0e-4 + 1.0e-12
    assert result.evidence.closure.boundary_edges == 0
    assert result.evidence.embedding_verified
    # Welding a corner by <= 2e-4 changes unit-cube volume by at most 2e-4 / 3.
    assert abs(result.evidence.closure.signed_volume - 1.0) < 1.0e-3


def test_t_junction_stitching_closes_cracked_face() -> None:
    # Two coplanar squares meeting along x=1 where the right side has a midpoint vertex at (1, 0.5, 0).
    verts = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.0, 0.5, 0.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 1, 2),
            (0, 2, 3),
            (1, 4, 6),
            (4, 5, 6),
            (6, 5, 2),
        ),
        dtype=np.int64,
    )
    policy = SurfaceRepairPolicy(
        t_junction_tolerance=1.0e-6,
        verify_embedding=True,
    )
    result = repair_surface_mesh((verts, faces), policy=policy)
    assert result.evidence.t_junction_splits == 1
    # Interior seam between the two squares is now conforming (only the 6 outer rim edges remain open).
    assert result.evidence.closure.boundary_edges == 6
    assert result.evidence.closure.nonmanifold_edges == 0


def test_liepa_hole_filling_respects_size_bounds_and_fairs_curved_caps() -> None:
    verts, faces = _icosphere(subdivisions=2, radius=1.0)
    # Remove a patch of faces around the north pole (+z) to create a non-planar 3D hole.
    centroids = verts[faces].mean(axis=1)
    keep = centroids[:, 2] < 0.82
    open_faces = faces[keep]

    # Size bound too tight: hole is skipped and require_watertight fails closed.
    strict_skip = SurfaceRepairPolicy(
        fill_holes=True,
        max_hole_edges=3,
        require_watertight=True,
    )
    with pytest.raises(SurfaceRepairError) as exc_info:
        repair_surface_mesh((verts, open_faces), policy=strict_skip)
    assert exc_info.value.status == "open_boundary"

    # Full Liepa DP + refinement + harmonic fairing closes the curved sphere hole cleanly.
    fill_policy = SurfaceRepairPolicy(
        fill_holes=True,
        fair_hole_caps=True,
        refine_hole_caps=True,
        require_watertight=True,
        require_manifold=True,
        verify_embedding=True,
    )
    repaired = repair_surface_mesh((verts, open_faces), policy=fill_policy)
    assert repaired.evidence.filled_hole_count == 1
    assert repaired.evidence.skipped_large_holes == 0
    assert repaired.evidence.added_cap_triangles >= 3
    assert repaired.evidence.closure.boundary_edges == 0
    assert repaired.evidence.embedding_verified
    assert repaired.evidence.closure.signed_volume > 3.8


def test_nested_cavity_orientation_and_debris_pruning() -> None:
    outer_v, outer_f = _box_mesh((0.0, 0.0, 0.0), (4.0, 4.0, 4.0), invert=True)
    inner_v, inner_f = _box_mesh((1.0, 1.0, 1.0), (3.0, 3.0, 3.0), invert=False)
    floater_v = np.asarray(
        ((10.0, 10.0, 10.0), (10.02, 10.0, 10.0), (10.0, 10.02, 10.0)),
        dtype=np.float64,
    )
    floater_f = np.asarray(((0, 1, 2),), dtype=np.int64)
    soup = _concat((outer_v, outer_f), (inner_v, inner_f), (floater_v, floater_f))

    policy = SurfaceRepairPolicy(
        orient_shells=True,
        prune_small_components=True,
        min_component_faces=4,
        require_watertight=True,
        require_manifold=True,
        verify_embedding=True,
    )
    result = repair_surface_mesh(soup, policy=policy)
    assert result.evidence.cavity_component_count == 1
    assert result.evidence.pruned_component_count == 1
    assert result.evidence.pruned_face_count == 1
    assert result.evidence.flipped_face_count == 24
    assert result.evidence.embedding_verified
    assert abs(result.evidence.closure.signed_volume - 56.0) < 1.0e-9


def test_touching_solids_pinch_split_policy() -> None:
    soup = _concat(
        _box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        _box_mesh((1.0, 1.0, 0.0), (2.0, 2.0, 1.0)),
    )
    # Without split_nonmanifold_pinches, require_manifold fails closed.
    with pytest.raises(SurfaceRepairError) as exc_info:
        repair_surface_mesh(soup, policy=SurfaceRepairPolicy.watertight_solid())
    assert exc_info.value.status == "nonmanifold_output"

    # With split_nonmanifold_pinches=True, touching solids separate with 0 drift.
    split_result = repair_surface_mesh(
        soup,
        policy=SurfaceRepairPolicy.watertight_solid(split_nonmanifold_pinches=True),
    )
    assert split_result.evidence.split_pinch_vertices >= 2
    assert split_result.evidence.closure.nonmanifold_edges == 0
    assert split_result.evidence.closure.nonmanifold_vertices == 0
    assert split_result.evidence.max_vertex_displacement == 0.0
    assert abs(split_result.evidence.closure.signed_volume - 2.0) < 1.0e-12


def test_certified_polish_remesh_produces_hausdorff_bound() -> None:
    v1, f1 = _icosphere(subdivisions=2, radius=1.0)
    v2, f2 = _icosphere(subdivisions=2, radius=1.0)
    v2 = v2 + np.asarray((0.8, 0.0, 0.0), dtype=np.float64)
    soup = _concat((v1, f1), (v2, f2))
    policy = SurfaceRepairPolicy.watertight_solid(
        polish_edge_length=0.25,
        polish_tolerance=0.05,
        polish_minimum_angle_degrees=25.0,
    )
    result = repair_surface_mesh(soup, policy=policy)
    assert result.evidence.embedding_verified
    assert result.evidence.hausdorff_bound is not None
    assert result.evidence.hausdorff_bound <= 0.05
    assert result.evidence.closure.boundary_edges == 0
    assert result.evidence.closure.nonmanifold_edges == 0


def test_import_surface_repair_policy_hook_repairs_dirty_stl(tmp_path: Path) -> None:
    pytest.importorskip("meshio")
    # Build a valid clean mesh first just to export an STL, then overwrite with a dirty cracked STL.
    v1, f1 = _box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    v2, f2 = _box_mesh((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
    dirty_v, dirty_f = _concat((v1, f1), (v2, f2))
    stl_path = tmp_path / "dirty_overlap.stl"
    import meshio

    meshio.Mesh(dirty_v, [("triangle", dirty_f)]).write(stl_path)

    imported = import_surface(
        stl_path,
        SurfaceImportPolicy(
            source_length_unit=METER,
            allow_lossy=True,
            repair_policy=SurfaceRepairPolicy.watertight_solid(),
        ),
        file_format=SurfaceFileFormat.STL,
    )
    assert imported.repair_evidence is not None
    assert imported.repair_evidence.embedding_verified
    assert imported.repair_evidence.closure.boundary_edges == 0
    assert abs(imported.repair_evidence.closure.signed_volume - 1.875) < 1.0e-12
    # Cleanly exportable back as a certified solid STL.
    out_stl = tmp_path / "repaired_solid.stl"
    exported = export_surface(
        out_stl,
        imported.model,
        SurfaceExportPolicy(target_length_unit=METER, allow_lossy=True),
        file_format=SurfaceFileFormat.STL,
    )
    assert exported.path.is_file()
