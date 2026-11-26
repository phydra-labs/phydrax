#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Head-to-head benchmark: Phydrax native certified surface repair vs. trimesh.repair.

Evaluates both wall-clock runtime and independent exact mathematical validity
(watertightness, 2-manifoldness, exact self-intersection / embedding findings via
``verify_surface_embedding``, signed volume accuracy, and minimum triangle angle)
across six dirty-mesh torture suites.
"""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import numpy as np
import trimesh

from phydrax._meshcore import verify_surface_embedding
from phydrax.geometry.surface import repair_surface_mesh, SurfaceRepairPolicy


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


def _icosphere(
    subdivisions: int = 3, radius: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return (
        np.ascontiguousarray(sphere.vertices, dtype=np.float64),
        np.ascontiguousarray(sphere.faces, dtype=np.int64),
    )


def _concat(*meshes: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    verts_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    offset = 0
    for verts, faces in meshes:
        verts_list.append(verts)
        faces_list.append(faces + offset)
        offset += verts.shape[0]
    return np.vstack(verts_list), np.vstack(faces_list)


def _audit_output(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    pts = np.ascontiguousarray(vertices, dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int64)
    if tris.shape[0] == 0:
        return {
            "triangles": 0,
            "watertight": False,
            "embedding_findings": -1,
            "signed_volume": 0.0,
            "min_angle_deg": 0.0,
        }
    p0, p1, p2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    volume = float(np.sum(p0 * np.cross(p1, p2)) / 6.0)
    e01 = p1 - p0
    e12 = p2 - p1
    e20 = p0 - p2
    l01 = np.linalg.norm(e01, axis=1)
    l12 = np.linalg.norm(e12, axis=1)
    l20 = np.linalg.norm(e20, axis=1)
    valid = (l01 > 0.0) & (l12 > 0.0) & (l20 > 0.0)
    if np.all(valid):
        c0 = np.clip(np.sum(e01 * (-e20), axis=1) / (l01 * l20), -1.0, 1.0)
        c1 = np.clip(np.sum(e12 * (-e01), axis=1) / (l12 * l01), -1.0, 1.0)
        c2 = np.clip(np.sum(e20 * (-e12), axis=1) / (l20 * l12), -1.0, 1.0)
        min_ang = float(np.degrees(np.min(np.arccos(np.concatenate((c0, c1, c2))))))
    else:
        min_ang = 0.0
    verdict = verify_surface_embedding(pts, tris)
    findings = int(verdict.findings.shape[0])
    tm = trimesh.Trimesh(vertices=pts, faces=tris, process=False)
    return {
        "triangles": int(tris.shape[0]),
        "watertight": bool(tm.is_watertight),
        "embedding_findings": findings,
        "signed_volume": volume,
        "min_angle_deg": min_ang,
    }


def _run_trimesh_repair(
    vertices: np.ndarray, faces: np.ndarray, *, weld_tol: float = 0.0
) -> tuple[np.ndarray, np.ndarray, float]:
    start = perf_counter()
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
    if weld_tol > 0.0:
        digits = max(1, int(round(-np.log10(weld_tol))))
        mesh.merge_vertices(merge_tex=False, merge_norm=False, digits_vertex=digits)
    else:
        mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh)
    elapsed = perf_counter() - start
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        elapsed,
    )


def _cases() -> list[
    tuple[str, tuple[np.ndarray, np.ndarray], SurfaceRepairPolicy, float, float]
]:
    cases: list[
        tuple[str, tuple[np.ndarray, np.ndarray], SurfaceRepairPolicy, float, float]
    ] = []

    # 1. Self-intersecting overlapping boxes + internal baffle + canceling duplicate.
    v1, f1 = _box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    v2, f2 = _box_mesh((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
    bv = np.asarray(
        ((0.25, 0.0, 0.0), (0.25, 1.0, 0.0), (0.25, 1.0, 1.0), (0.25, 0.0, 1.0)),
        dtype=np.float64,
    )
    bf = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    sv, sf = _concat((v1, f1), (v2, f2), (bv, bf))
    sf = np.vstack((sf, sf[0:1], sf[0:1, (0, 2, 1)]))
    cases.append(
        (
            "self-intersecting-boxes-baffle",
            (sv, sf),
            SurfaceRepairPolicy.watertight_solid(),
            0.0,
            1.875,
        )
    )

    # 2. Overlapping high-res icospheres (subdivision 3, 2560 input tris).
    s1v, s1f = _icosphere(3, 1.0)
    s2v, s2f = _icosphere(3, 1.0)
    s2v = s2v + np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    # Exact analytical union volume of two unit spheres with center distance d=1:
    # 2*(4/3 pi) - (pi/12)*(4*1 + 1)*(2 - 1)^2 = (8/3 - 5/12) pi = (9/4) pi.
    cases.append(
        (
            "self-intersecting-spheres-2560",
            _concat((s1v, s1f), (s2v, s2f)),
            SurfaceRepairPolicy.watertight_solid(),
            0.0,
            float(2.25 * np.pi),
        )
    )
    cases.append(
        (
            "self-intersecting-spheres-polished",
            _concat((s1v, s1f), (s2v, s2f)),
            SurfaceRepairPolicy.watertight_solid(
                polish_edge_length=0.18,
                polish_tolerance=0.08,
                polish_minimum_angle_degrees=25.0,
            ),
            0.0,
            float(2.25 * np.pi),
        )
    )

    # 3. Curved 3D sphere with 24-edge polar hole (non-planar loop).
    hv, hf = _icosphere(3, 1.0)
    keep = hv[hf].mean(axis=1)[:, 2] < 0.85
    cases.append(
        (
            "curved-sphere-polar-hole",
            (hv, hf[keep]),
            SurfaceRepairPolicy(
                fill_holes=True,
                fair_hole_caps=True,
                refine_hole_caps=True,
                require_watertight=True,
                require_manifold=True,
                verify_embedding=True,
            ),
            0.0,
            float((4.0 / 3.0) * np.pi),
        )
    )

    # 4. Inverted outer shell + wrongly-outward nested cavity + stray debris floaters.
    ov, of = _box_mesh((0.0, 0.0, 0.0), (4.0, 4.0, 4.0), invert=True)
    iv, inf = _box_mesh((1.0, 1.0, 1.0), (3.0, 3.0, 3.0), invert=False)
    dv = np.asarray(
        ((10.0, 10.0, 10.0), (10.02, 10.0, 10.0), (10.0, 10.02, 10.0)),
        dtype=np.float64,
    )
    df = np.asarray(((0, 1, 2),), dtype=np.int64)
    cases.append(
        (
            "inverted-nested-cavity-debris",
            _concat((ov, of), (iv, inf), (dv, df)),
            SurfaceRepairPolicy(
                orient_shells=True,
                prune_small_components=True,
                min_component_faces=4,
                require_watertight=True,
                require_manifold=True,
                verify_embedding=True,
            ),
            0.0,
            56.0,
        )
    )

    # 5. Cracked float32-style sphere soup (unshared vertices perturbed by 1e-5) + collinear slivers.
    rng = np.random.default_rng(42)
    cv, cf = _icosphere(3, 1.0)
    soup_verts = cv[cf.reshape(-1)] + rng.uniform(-1.0e-5, 1.0e-5, size=(cf.size, 3))
    soup_faces = np.arange(cf.size, dtype=np.int64).reshape(-1, 3)
    # Append 5 exact collinear slivers.
    sliver_pts = np.asarray(
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64
    )
    base_idx = soup_verts.shape[0]
    soup_verts = np.vstack((soup_verts, sliver_pts))
    sliver_faces = np.tile(
        np.asarray(((base_idx, base_idx + 1, base_idx + 2),), dtype=np.int64), (5, 1)
    )
    soup_faces = np.vstack((soup_faces, sliver_faces))
    cases.append(
        (
            "cracked-stl-soup-slivers",
            (soup_verts, soup_faces),
            SurfaceRepairPolicy(
                weld_tolerance=1.0e-4,
                drop_degenerate_triangles=True,
                orient_shells=True,
                require_watertight=True,
                require_manifold=True,
                verify_embedding=True,
            ),
            1.0e-4,
            float((4.0 / 3.0) * np.pi),
        )
    )

    return cases


def main() -> None:
    rows: list[dict[str, Any]] = []
    for name, (verts, faces), policy, weld_tol, ref_vol in _cases():
        # Warm once, then min-of-3 timing for both tools.
        phx_times: list[float] = []
        tri_times: list[float] = []
        phx_result = None
        tri_out: tuple[np.ndarray, np.ndarray] = (verts, faces)
        for _ in range(3):
            t0 = perf_counter()
            phx_result = repair_surface_mesh((verts, faces), policy=policy)
            phx_times.append(perf_counter() - t0)
            tv, tf, dt = _run_trimesh_repair(verts, faces, weld_tol=weld_tol)
            tri_times.append(dt)
            tri_out = (tv, tf)
        assert phx_result is not None
        phx_pts = np.asarray(phx_result.surface.mesh.coordinates, dtype=np.float64)
        phx_tris = np.asarray(
            phx_result.surface.mesh.connectivity.cell_vertices, dtype=np.int64
        )[:, :3]
        phx_audit = _audit_output(phx_pts, phx_tris)
        tri_audit = _audit_output(tri_out[0], tri_out[1])
        rows.append(
            {
                "case": name,
                "reference_volume": round(ref_vol, 6),
                "phydrax_ms": round(min(phx_times) * 1000.0, 3),
                "trimesh_ms": round(min(tri_times) * 1000.0, 3),
                "phydrax": {
                    **phx_audit,
                    "embedding_verified": phx_result.evidence.embedding_verified,
                    "max_displacement": phx_result.evidence.max_vertex_displacement,
                },
                "trimesh": tri_audit,
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
