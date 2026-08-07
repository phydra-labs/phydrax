#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import meshio
import numpy as np
import pyvista as pv
import trimesh
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ._regions import MeshRegion, PlanarMeshRegion


def _canonical_faces(faces: np.ndarray) -> np.ndarray:
    minimum_position = np.argmin(faces, axis=1)
    offsets = (
        minimum_position[:, None] + np.arange(faces.shape[1], dtype=np.int32)
    ) % faces.shape[1]
    rotated = np.take_along_axis(faces, offsets, axis=1)
    order = np.lexsort(tuple(rotated[:, axis] for axis in reversed(range(rotated.shape[1]))))
    return rotated[order]


def _canonical_triangle_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices_ = np.asarray(vertices, dtype=float)
    faces_ = np.asarray(faces, dtype=np.int32)
    if vertices_.ndim != 2 or faces_.ndim != 2 or faces_.shape[1] != 3:
        return vertices_, faces_
    if (
        vertices_.shape[0] == 0
        or faces_.size == 0
        or np.any(faces_ < 0)
        or np.any(faces_ >= vertices_.shape[0])
    ):
        return vertices_, faces_

    referenced = np.unique(faces_.reshape((-1,)))
    old_to_referenced = np.full(vertices_.shape[0], -1, dtype=np.int32)
    old_to_referenced[referenced] = np.arange(referenced.shape[0], dtype=np.int32)
    vertices_ = vertices_[referenced]
    faces_ = old_to_referenced[faces_]

    vertices_, referenced_to_unique = np.unique(
        vertices_,
        axis=0,
        return_inverse=True,
    )
    faces_ = referenced_to_unique[faces_].astype(np.int32)
    faces_ = _canonical_faces(faces_)

    if vertices_.shape[1] >= 3:
        triangles = vertices_[faces_, :3]
        signed_volume = np.sum(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            )
        )
        if signed_volume < 0.0:
            faces_ = _canonical_faces(faces_[:, [0, 2, 1]])
    return vertices_, faces_


def _canonical_feature_id(
    prefix: str,
    vertices: np.ndarray,
    connectivity: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for values, dtype in (
        (vertices, np.float64),
        (connectivity, np.int64),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return f"{prefix}-{digest.hexdigest()[:24]}"


def _canonical_ring(points: np.ndarray) -> np.ndarray:
    points_ = np.asarray(points, dtype=float)
    start = int(np.lexsort((points_[:, 1], points_[:, 0]))[0])
    return np.roll(points_, -start, axis=0)


def _meshio_triangles(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray]:
    blocks = [
        np.asarray(block.data, dtype=np.int32)
        for block in mesh.cells
        if block.type == "triangle"
    ]
    if not blocks:
        raise ValueError("Mesh input contains no triangle cells.")
    return np.asarray(mesh.points, dtype=float), np.concatenate(blocks, axis=0)


def _pyvista_triangles(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    triangulated = mesh.extract_surface().triangulate()
    packed = np.asarray(triangulated.faces, dtype=np.int64)
    if packed.size == 0 or packed.size % 4 != 0:
        raise ValueError("PolyData contains no triangle faces.")
    records = packed.reshape((-1, 4))
    if np.any(records[:, 0] != 3):
        raise ValueError("Triangulated PolyData contains a non-triangle cell.")
    return (
        np.asarray(triangulated.points, dtype=float),
        records[:, 1:].astype(np.int32),
    )


def _trimesh_arrays(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh vertices must contain only finite values.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must have shape (num_faces, 3).")
    mesh_ = mesh.copy()
    mesh_.update_faces(mesh_.unique_faces() & mesh_.nondegenerate_faces())
    mesh_.remove_unreferenced_vertices()
    mesh_.merge_vertices()
    mesh_.fix_normals(multibody=True)
    return np.asarray(mesh_.vertices, dtype=float), np.asarray(
        mesh_.faces, dtype=np.int32
    )


def triangle_arrays(source: Any, /) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize a mesh object or mesh file into triangle arrays."""

    if isinstance(source, meshio.Mesh):
        return _canonical_triangle_arrays(*_meshio_triangles(source))
    if isinstance(source, trimesh.Trimesh):
        return _canonical_triangle_arrays(*_trimesh_arrays(source))
    if isinstance(source, pv.PolyData):
        return _canonical_triangle_arrays(*_pyvista_triangles(source))
    if isinstance(source, (str, Path)):
        loaded = trimesh.load_mesh(Path(source).expanduser(), process=True)
        if isinstance(loaded, trimesh.Scene):
            geometries = tuple(loaded.geometry.values())
            if not geometries:
                raise ValueError("Mesh scene contains no geometry.")
            loaded = trimesh.util.concatenate(geometries)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError("Mesh file did not resolve to triangular surface geometry.")
        return _canonical_triangle_arrays(*_trimesh_arrays(loaded))
    raise TypeError(
        "Mesh input must be a path, meshio.Mesh, trimesh.Trimesh, or pyvista.PolyData."
    )


def mesh_region_from_source(
    source: Any,
    /,
    *,
    recenter: bool = True,
    feature_id: str | None = None,
) -> MeshRegion:
    """Build one watertight 3D simplicial region from a mesh source."""

    vertices, faces = triangle_arrays(source)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError("A 3D mesh must provide three-dimensional vertices.")
    vertices = vertices[:, :3]
    if recenter:
        vertices = vertices - 0.5 * (np.min(vertices, axis=0) + np.max(vertices, axis=0))
    feature_id_ = feature_id or _canonical_feature_id("mesh-region", vertices, faces)
    return MeshRegion(vertices, faces, feature_id=feature_id_)


def planar_region_from_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    /,
    *,
    recenter: bool = True,
    feature_id: str | None = None,
) -> PlanarMeshRegion:
    """Recover oriented polygon loops from one triangulated planar region."""

    vertices_ = np.asarray(vertices, dtype=float)
    faces_ = np.asarray(faces, dtype=np.int32)
    if vertices_.ndim != 2 or vertices_.shape[1] < 2:
        raise ValueError("Planar vertices must have at least two coordinates.")
    if faces_.ndim != 2 or faces_.shape[1] != 3:
        raise ValueError("faces must have shape (num_triangles, 3).")
    coordinates = vertices_[:, :2]
    polygons: list[ShapelyPolygon] = []
    for face in faces_:
        polygon = ShapelyPolygon(coordinates[face])
        if polygon.area > 0.0:
            polygons.append(polygon)
    region = unary_union(polygons)
    if region.geom_type != "Polygon":
        raise ValueError("Planar mesh must represent one connected polygonal region.")
    region = orient(region, sign=1.0)
    exterior = _canonical_ring(np.asarray(region.exterior.coords[:-1], dtype=float))
    interiors = sorted(
        (
            _canonical_ring(np.asarray(interior.coords[:-1], dtype=float))
            for interior in region.interiors
        ),
        key=lambda points: tuple(points.reshape((-1,))),
    )
    loop_points = [exterior, *interiors]
    compact_vertices = np.concatenate(loop_points, axis=0)
    if recenter:
        compact_vertices = compact_vertices - 0.5 * (
            np.min(compact_vertices, axis=0) + np.max(compact_vertices, axis=0)
        )
    loops: list[np.ndarray] = []
    cursor = 0
    for points in loop_points:
        loops.append(np.arange(cursor, cursor + points.shape[0], dtype=np.int32))
        cursor += points.shape[0]
    loop_offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int32),
            np.cumsum(
                np.asarray([loop.shape[0] for loop in loops], dtype=np.int32)
            ),
        )
    )
    feature_id_ = feature_id or _canonical_feature_id(
        "planar-region",
        compact_vertices,
        loop_offsets,
    )
    return PlanarMeshRegion(compact_vertices, loops, feature_id=feature_id_)


def planar_region_from_source(
    source: Any,
    /,
    *,
    recenter: bool = True,
    feature_id: str | None = None,
) -> PlanarMeshRegion:
    """Build one 2D simplicial region from a triangulated mesh source."""

    vertices, faces = triangle_arrays(source)
    if vertices.shape[1] >= 3 and not np.allclose(vertices[:, 2], vertices[0, 2]):
        raise ValueError("A planar mesh must lie in one constant-z plane.")
    return planar_region_from_triangles(
        vertices,
        faces,
        recenter=recenter,
        feature_id=feature_id,
    )


__all__ = [
    "mesh_region_from_source",
    "planar_region_from_source",
    "planar_region_from_triangles",
    "triangle_arrays",
]
