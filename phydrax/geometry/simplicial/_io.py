#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import meshio
import numpy as np

import phydrax.ein as ein

from ..._external_resource import read_bounded_resource, ResourceLimits
from ._mesh import TriangleMesh
from ._regions import MeshRegion, PlanarMeshRegion
from ._topology import TriangleTopology


def _canonical_faces(faces: np.ndarray) -> np.ndarray:
    minimum_position = np.argmin(faces, axis=1)
    offsets = (
        minimum_position[:, None] + np.arange(faces.shape[1], dtype=np.int32)
    ) % faces.shape[1]
    rotated = np.take_along_axis(faces, offsets, axis=1)
    order = np.lexsort(
        tuple(rotated[:, axis] for axis in reversed(range(rotated.shape[1])))
    )
    return rotated[order]


def _canonical_triangle_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices_ = np.asarray(vertices, dtype=np.float64)
    faces_ = np.asarray(faces, dtype=np.int32)
    if vertices_.ndim != 2 or vertices_.shape[0] == 0 or vertices_.shape[1] < 2:
        raise ValueError(
            "Mesh vertices must have shape (num_vertices > 0, dimension >= 2)."
        )
    if not np.all(np.isfinite(vertices_)):
        raise ValueError("Mesh vertices must contain only finite values.")
    if faces_.ndim != 2 or faces_.shape[0] == 0 or faces_.shape[1] != 3:
        raise ValueError("Mesh faces must have shape (num_faces > 0, 3).")
    if np.any(faces_ < 0) or np.any(faces_ >= vertices_.shape[0]):
        raise ValueError("Mesh faces contain an out-of-range vertex index.")

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
    if np.any(
        (faces_[:, 0] == faces_[:, 1])
        | (faces_[:, 1] == faces_[:, 2])
        | (faces_[:, 2] == faces_[:, 0])
    ):
        raise ValueError("Vertex canonicalization produced a degenerate triangle.")
    if np.unique(np.sort(faces_, axis=1), axis=0).shape[0] != faces_.shape[0]:
        raise ValueError("Mesh input contains duplicate triangle faces.")
    triangles = vertices_[faces_]
    if vertices_.shape[1] == 2:
        doubled_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
            triangles[:, 2, 1] - triangles[:, 0, 1]
        ) - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (
            triangles[:, 2, 0] - triangles[:, 0, 0]
        )
        scale = max(float(np.max(np.abs(vertices_))), 1.0)
        if np.any(np.abs(doubled_area) <= 128.0 * np.finfo(np.float64).eps * scale**2):
            raise ValueError("Mesh input contains a zero-area triangle.")
    else:
        cross = np.cross(
            triangles[:, 1, :3] - triangles[:, 0, :3],
            triangles[:, 2, :3] - triangles[:, 0, :3],
        )
        scale = max(float(np.max(np.abs(vertices_[:, :3]))), 1.0)
        if np.any(
            np.sum(cross * cross, axis=1)
            <= (128.0 * np.finfo(np.float64).eps * scale**2) ** 2
        ):
            raise ValueError("Mesh input contains a zero-area triangle.")

    faces_ = _canonical_faces(faces_)
    if vertices_.shape[1] >= 3:
        triangles = vertices_[faces_, :3]
        signed_volume = np.sum(
            ein.contract(
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


def _signed_area2(points: np.ndarray, /) -> float:
    return float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _canonical_ring(points: np.ndarray, /, *, counter_clockwise: bool) -> np.ndarray:
    points_ = np.asarray(points, dtype=np.float64)
    if (_signed_area2(points_) > 0.0) != counter_clockwise:
        points_ = points_[::-1]
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
    return np.asarray(mesh.points, dtype=np.float64), np.concatenate(blocks, axis=0)


def triangle_arrays(source: Any, /) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize native arrays, a TriangleMesh, Meshio data, or a mesh file."""

    if isinstance(source, TriangleMesh):
        return _canonical_triangle_arrays(source.vertices, source.faces)
    if isinstance(source, meshio.Mesh):
        return _canonical_triangle_arrays(*_meshio_triangles(source))
    if isinstance(source, tuple) and len(source) == 2:
        return _canonical_triangle_arrays(source[0], source[1])
    if isinstance(source, (str, Path)):
        source_path = Path(source).expanduser().absolute()
        resource = read_bounded_resource(
            source_path.name,
            trusted_root=source_path.parent,
            limits=ResourceLimits(
                1_000_000_000,
                64,
                50_000_000,
                1024,
                1024,
            ),
        )
        with TemporaryDirectory(prefix="phydrax-triangle-read-") as temporary:
            staged = Path(temporary) / source_path.name
            staged.write_bytes(resource.data)
            return _canonical_triangle_arrays(*_meshio_triangles(meshio.read(staged)))
    raise TypeError(
        "Mesh input must be a path, meshio.Mesh, TriangleMesh, or (vertices, faces) pair."
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
    if vertices.shape[1] < 3:
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
    """Recover oriented boundary loops from one connected triangle complex."""

    vertices_, faces_ = _canonical_triangle_arrays(vertices, faces)
    coordinates = vertices_[:, :2]
    topology = TriangleTopology(faces_, num_vertices=coordinates.shape[0])
    if topology.num_face_components != 1:
        raise ValueError("Planar mesh must represent one connected polygonal region.")
    offsets = np.asarray(topology.boundary_loop_offsets, dtype=np.int32)
    loop_vertices = np.asarray(topology.boundary_loop_vertices, dtype=np.int32)
    if offsets.shape[0] <= 1:
        raise ValueError("Planar mesh has no boundary loops.")
    raw_loops = [
        coordinates[loop_vertices[offsets[index] : offsets[index + 1]]]
        for index in range(offsets.shape[0] - 1)
    ]
    exterior_index = int(np.argmax([abs(_signed_area2(loop)) for loop in raw_loops]))
    exterior = _canonical_ring(raw_loops[exterior_index], counter_clockwise=True)
    interiors = sorted(
        (
            _canonical_ring(loop, counter_clockwise=False)
            for index, loop in enumerate(raw_loops)
            if index != exterior_index
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
            np.cumsum(np.asarray([loop.shape[0] for loop in loops], dtype=np.int32)),
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
