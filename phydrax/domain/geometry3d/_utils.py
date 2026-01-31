#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path
from typing import Literal, Sequence

import meshio
import numpy as np
import trimesh


_MESH_EQ_DECIMALS = 8


def _canonicalize_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    decimals: int = _MESH_EQ_DECIMALS,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2:
        v = v.reshape((-1, v.shape[-1]))
    if f.ndim != 2:
        f = f.reshape((-1, f.shape[-1]))

    v_round = np.round(v, decimals=decimals)
    v_keys = tuple(v_round[:, i] for i in range(v_round.shape[1] - 1, -1, -1))
    v_order = np.lexsort(v_keys)
    v_sorted = v_round[v_order]

    inv = np.empty_like(v_order)
    inv[v_order] = np.arange(v_order.shape[0])
    f_remap = inv[f]
    f_sorted = np.sort(f_remap, axis=1)
    f_keys = tuple(f_sorted[:, i] for i in range(f_sorted.shape[1] - 1, -1, -1))
    f_order = np.lexsort(f_keys)
    f_sorted = f_sorted[f_order]

    return v_sorted, f_sorted


def _sanitize_meshio_mesh(
    meshio_mesh: meshio.Mesh,
    *,
    output_type: Literal["trimesh", "meshio"],
    recenter: bool = True,
) -> trimesh.Trimesh | meshio.Mesh:
    def ensure_counterclockwise(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
        v0, v1, v2 = vertices[face]
        if np.cross(v1 - v0, v2 - v0)[2] < 0:  # If clockwise
            return face[::-1]  # Reverse the order
        return face

    vertices = meshio_mesh.points

    if recenter:
        mesh_vertices = vertices
        mesh_faces = np.vstack(
            [
                cell_block.data
                for cell_block in meshio_mesh.cells
                if cell_block.type == "triangle"
            ]
        )

        triangles = mesh_vertices[mesh_faces]  # shape: (num_triangles, 3, 3)
        triangle_centroids = np.mean(triangles, axis=1)  # shape: (num_triangles, 3)

        trimesh_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
        triangle_areas = trimesh_mesh.area_faces  # shape: (num_triangles,)

        weighted_centroids = triangle_centroids * triangle_areas[:, None]
        center_of_mass = np.sum(weighted_centroids, axis=0) / np.sum(triangle_areas)

        vertices -= center_of_mass

    new_faces = []

    for cell_block in meshio_mesh.cells:
        if cell_block.type == "triangle":
            for face in cell_block.data:
                new_faces.append(ensure_counterclockwise(vertices, face))
        elif cell_block.type == "quad":
            quads = cell_block.data
            for quad in quads:
                triangle1 = ensure_counterclockwise(vertices, quad[[0, 1, 2]])
                triangle2 = ensure_counterclockwise(vertices, quad[[0, 2, 3]])
                new_faces.extend([triangle1, triangle2])

    faces = np.array(new_faces)

    if output_type == "trimesh":
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    elif output_type == "meshio":
        mesh = meshio.Mesh(points=vertices, cells=[("triangle", faces)])

    return mesh


def _sanitize_mesh(
    mesh: trimesh.Trimesh | meshio.Mesh | Path | str,
    *,
    recenter: bool = True,
) -> trimesh.Trimesh:
    if isinstance(mesh, Path | str):
        mesh_path = Path(mesh).resolve(strict=True)
        mesh = trimesh.load_mesh(mesh_path)

    if isinstance(mesh, meshio.Mesh):
        mesh = _sanitize_meshio_mesh(mesh, output_type="trimesh", recenter=recenter)

    assert isinstance(mesh, trimesh.Trimesh)

    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_infinite_values()
    mesh.fill_holes()
    mesh.fix_normals()

    if recenter:
        mesh.apply_translation(-mesh.center_mass)

    return mesh


def _boolean_mesh(
    meshes: Sequence[trimesh.Trimesh],
    *,
    operation: Literal["union", "difference", "intersection"],
    engine: Literal["manifold", "blender"] | None = "manifold",
    check_volume: bool = True,
) -> trimesh.Trimesh:
    """Perform a boolean operation on a sequence of meshes.

    Prefers the Manifold backend when available, with Blender as a fallback.
    Returns a single `trimesh.Trimesh`.
    """
    if len(meshes) == 0:
        raise ValueError("_boolean_mesh requires at least one mesh")

    op = operation.lower().strip()
    if op not in {"union", "difference", "intersection"}:
        raise ValueError(f"Unsupported boolean operation: {operation!r}")

    eng = engine or "manifold"
    if op == "union":
        res = trimesh.boolean.union(meshes, engine=eng, check_volume=check_volume)
    elif op == "difference":
        res = trimesh.boolean.difference(meshes, engine=eng, check_volume=check_volume)
    else:
        res = trimesh.boolean.intersection(meshes, engine=eng, check_volume=check_volume)
    assert isinstance(res, trimesh.Trimesh)
    assert res.faces is not None and res.faces.shape[0] > 0
    return res


def _z0_faces_to_meshio(
    mesh: trimesh.Trimesh,
    *,
    tol: float = 1e-9,
) -> meshio.Mesh:
    """Extract faces lying on the z=0 plane to a 2D triangle mesh (z=0).

    - Select faces whose all three vertices have |z| <= tol.
    - Deduplicate vertices and remap faces.
    - Returns a `meshio.Mesh` with points shaped (N, 3) and z=0.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("_z0_faces_to_meshio expects a trimesh.Trimesh")

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("Input mesh vertices must be (N,3)")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("Input mesh faces must be (M,3)")

    z = np.abs(V[:, 2])
    on0 = z <= float(tol)
    # keep faces where all three vertices are on z=0 plane
    face_mask = np.all(on0[F], axis=1)
    F_sel = F[face_mask]
    if F_sel.size == 0:
        # return empty meshio mesh
        return meshio.Mesh(
            points=np.zeros((0, 3), dtype=float),
            cells=[("triangle", np.zeros((0, 3), dtype=np.int64))],
        )

    used = np.unique(F_sel.ravel())
    # build new vertex array (XY) with z=0
    V_used = V[used]
    V2 = np.column_stack(
        [V_used[:, 0], V_used[:, 1], np.zeros((V_used.shape[0],), dtype=V_used.dtype)]
    )
    # mapping old->new indices
    mapping = -np.ones((V.shape[0],), dtype=np.int64)
    mapping[used] = np.arange(used.shape[0], dtype=np.int64)
    F2 = mapping[F_sel]
    return meshio.Mesh(points=V2.astype(float), cells=[("triangle", F2.astype(np.int64))])
