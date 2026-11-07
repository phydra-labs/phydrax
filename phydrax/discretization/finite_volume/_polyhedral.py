#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_complex import (
    PolyhedralConnectivity,
    prepare_polyhedral_worksets,
)
from .._cell_mesh import CellMesh


class PreparedPolyhedralFiniteVolumeGeometry(StrictModule, NonTrainableState):
    """Certified polyhedral FV geometry with planar and warped faces."""

    cell_volumes: Array
    cell_centers: Array
    face_centers: Array
    face_area_vectors: Array
    face_measures: Array
    face_quadrature_points: Array
    face_quadrature_weights: Array
    face_quadrature_valid: Array
    cell_quadrature_points: Array
    cell_quadrature_weights: Array
    cell_quadrature_valid: Array
    owner_cells: Array
    neighbor_cells: Array
    closure_residual: Array
    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


def _face_triangles(
    polygon: np.ndarray, planarity: float
) -> tuple[list[tuple[int, np.ndarray, float]], np.ndarray, float]:
    """Positive sub-triangles ``(slot, corners, weight)``, area vector, measure.

    A face within ``planarity`` (relative to its diameter) of its Newell plane
    fans from its first vertex. A warped face is decomposed about its vertex
    centroid and its area vector is the sum of those triangle area vectors.
    """
    relative = polygon - polygon[0]
    newell = 0.5 * np.sum(np.cross(relative, np.roll(relative, -1, axis=0)), axis=0)
    measure = float(np.linalg.norm(newell))
    if not np.isfinite(measure) or measure <= 0:
        raise ValueError("Polyhedral faces require positive finite measure.")
    unit = newell / measure
    distance = (polygon - polygon[0]) @ unit
    scale = max(1.0, float(np.max(np.linalg.norm(polygon - polygon[0], axis=1))))
    triangles: list[tuple[int, np.ndarray, float]] = []
    if np.max(np.abs(distance)) <= planarity * scale:
        # Conforming faces may carry collinear vertices on their edges; the
        # zero-area fan triangles they produce carry no measure and are skipped.
        degenerate_area = planarity * scale * scale
        for local in range(1, polygon.shape[0] - 1):
            triangle = polygon[[0, local, local + 1]]
            vector = 0.5 * np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            weight = float(np.dot(vector, unit))
            if not np.isfinite(weight) or weight < -degenerate_area:
                raise ValueError(
                    "Polyhedral face loop is not consistently oriented/convex."
                )
            if weight > degenerate_area:
                triangles.append((local - 1, triangle, weight))
        return triangles, newell, measure
    apex = np.mean(polygon, axis=0)
    following = np.roll(polygon, -1, axis=0)
    vectors = 0.5 * np.cross(polygon - apex, following - apex)
    area_vector = np.sum(vectors, axis=0)
    measure = float(np.linalg.norm(area_vector))
    if not np.isfinite(measure) or measure <= 0:
        raise ValueError("Polyhedral faces require positive finite measure.")
    unit = area_vector / measure
    for local in range(polygon.shape[0]):
        weight = float(np.dot(vectors[local], unit))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(
                "Warped polyhedral face is not star-shaped about its vertex centroid."
            )
        triangles.append(
            (local, np.stack((apex, polygon[local], following[local])), weight)
        )
    return triangles, area_vector, measure


def prepare_polyhedral_finite_volume_geometry(
    mesh: CellMesh,
    /,
    *,
    planarity_tolerance: float = 1.0e-10,
    closure_tolerance: float = 1.0e-10,
) -> PreparedPolyhedralFiniteVolumeGeometry:
    """Prepare Newell/divergence geometry from canonical polyhedral connectivity.

    Faces within ``planarity_tolerance`` of their Newell plane fan from their
    first vertex. Warped faces are decomposed into triangles about their vertex
    centroid; the face area vector is the sum of those triangle area vectors and
    both adjacent cells integrate volume and centroid over the same triangles,
    so the shared triangulated surface keeps closure and the divergence theorem
    exact.
    """
    if not isinstance(mesh, CellMesh) or not isinstance(
        mesh.connectivity, PolyhedralConnectivity
    ):
        raise TypeError(
            "Polyhedral FV geometry requires a canonical polyhedral CellMesh."
        )
    planarity = float(planarity_tolerance)
    closure = float(closure_tolerance)
    if (
        not np.isfinite(planarity)
        or planarity <= 0
        or not np.isfinite(closure)
        or closure <= 0
    ):
        raise ValueError("Polyhedral geometry tolerances must be positive and finite.")
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    connectivity = mesh.connectivity
    worksets = prepare_polyhedral_worksets(connectivity)
    face_vertices = np.asarray(worksets.face_vertices, dtype=np.int32)
    face_valid = np.asarray(worksets.face_vertex_valid, dtype=np.bool_)
    face_count, max_vertices = face_vertices.shape
    face_centers = np.zeros((face_count, points.shape[1]), dtype=np.float64)
    area_vectors = np.zeros_like(face_centers)
    face_measures = np.zeros((face_count,), dtype=np.float64)
    face_triangles: list[list[tuple[int, np.ndarray, float]]] = []
    for face in range(face_count):
        polygon = points[face_vertices[face, face_valid[face]]]
        if polygon.shape[0] < 3:
            raise ValueError("Every polyhedral face requires at least three vertices.")
        triangles, area_vector, measure = _face_triangles(polygon, planarity)
        if not triangles:
            raise ValueError("Polyhedral faces require positive finite measure.")
        anchor = polygon[0]
        face_centers[face] = anchor + np.average(
            np.asarray([np.mean(triangle - anchor, axis=0) for _, triangle, _ in triangles]),
            axis=0,
            weights=np.asarray([weight for _, _, weight in triangles]),
        )
        area_vectors[face], face_measures[face] = area_vector, measure
        face_triangles.append(triangles)
    # Planar n-gons fan into n - 2 triangles; warped n-gons need n slots.
    face_capacity = max(
        max_vertices - 2,
        max(
            (slot + 1 for triangles in face_triangles for slot, _, _ in triangles),
            default=0,
        ),
    )
    face_q_points = np.zeros(
        (face_count, face_capacity, points.shape[1]), dtype=np.float64
    )
    face_q_weights = np.zeros((face_count, face_capacity), dtype=np.float64)
    face_q_valid = np.zeros((face_count, face_capacity), dtype=np.bool_)
    for face, triangles in enumerate(face_triangles):
        for slot, triangle, weight in triangles:
            face_q_points[face, slot] = triangle[0] + np.mean(triangle - triangle[0], axis=0)
            face_q_weights[face, slot] = weight
            face_q_valid[face, slot] = True
    cell_faces = np.asarray(worksets.cell_faces, dtype=np.int32)
    cell_face_signs = np.asarray(worksets.cell_face_signs, dtype=np.int8)
    cell_face_valid = np.asarray(worksets.cell_face_valid, dtype=np.bool_)
    cell_vertices = np.asarray(worksets.cell_vertices, dtype=np.int32)
    cell_vertex_valid = np.asarray(worksets.cell_vertex_valid, dtype=np.bool_)
    cell_count, max_faces = cell_faces.shape
    cell_capacity = max_faces * face_capacity
    cell_centers = np.zeros((cell_count, points.shape[1]), dtype=np.float64)
    cell_volumes = np.zeros((cell_count,), dtype=np.float64)
    cell_q_points = np.zeros(
        (cell_count, cell_capacity, points.shape[1]), dtype=np.float64
    )
    cell_q_weights = np.zeros((cell_count, cell_capacity), dtype=np.float64)
    cell_q_valid = np.zeros((cell_count, cell_capacity), dtype=np.bool_)
    closure_residual = np.zeros((cell_count,), dtype=np.float64)
    for cell in range(cell_count):
        volume_sum = 0.0
        first_moment = np.zeros((points.shape[1],), dtype=np.float64)
        closure_vector = np.zeros_like(first_moment)
        star = np.mean(points[cell_vertices[cell, cell_vertex_valid[cell]]], axis=0)
        slot = 0
        for face, sign in zip(
            cell_faces[cell, cell_face_valid[cell]],
            cell_face_signs[cell, cell_face_valid[cell]],
            strict=True,
        ):
            closure_vector += float(sign) * area_vectors[face]
            for _, triangle, _ in face_triangles[face]:
                volume = (
                    float(sign)
                    * np.dot(
                        triangle[0] - star,
                        np.cross(triangle[1] - star, triangle[2] - star),
                    )
                    / 6.0
                )
                if not np.isfinite(volume) or volume <= 0:
                    raise ValueError(
                        "Polyhedral cell is not star-shaped about its certified center."
                    )
                centroid_offset = np.sum(triangle - star, axis=0) / 4.0
                centroid = star + centroid_offset
                volume_sum += volume
                first_moment += volume * centroid_offset
                cell_q_points[cell, slot] = centroid
                cell_q_weights[cell, slot] = volume
                cell_q_valid[cell, slot] = True
                slot += 1
        if not np.isfinite(volume_sum) or volume_sum <= 0:
            raise ValueError("Polyhedral cells require positive oriented volume.")
        cell_volumes[cell] = volume_sum
        cell_centers[cell] = star + first_moment / volume_sum
        closure_residual[cell] = np.linalg.norm(closure_vector) / max(
            np.sum(face_measures[cell_faces[cell, cell_face_valid[cell]]]), 1.0
        )
        if closure_residual[cell] > closure:
            raise ValueError("Polyhedral cell fails its oriented closure certificate.")
        weight_sum = np.sum(cell_q_weights[cell])
        if not np.isclose(
            weight_sum,
            volume_sum,
            atol=closure * max(1.0, volume_sum),
            rtol=0,
        ):
            raise ValueError(
                "Polyhedral positive tetrahedral quadrature misses cell volume."
            )
    owner = np.asarray(connectivity.face_owner, dtype=np.int32)
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int32)
    geometry_id = canonical_fingerprint(
        {
            "kind": "prepared-polyhedral-finite-volume-geometry",
            "mesh": mesh.mesh_id,
            "planarity_tolerance": planarity,
            "closure_tolerance": closure,
        }
    )
    return PreparedPolyhedralFiniteVolumeGeometry(
        jnp.asarray(cell_volumes),
        jnp.asarray(cell_centers),
        jnp.asarray(face_centers),
        jnp.asarray(area_vectors),
        jnp.asarray(face_measures),
        jnp.asarray(face_q_points),
        jnp.asarray(face_q_weights),
        jnp.asarray(face_q_valid),
        jnp.asarray(cell_q_points),
        jnp.asarray(cell_q_weights),
        jnp.asarray(cell_q_valid),
        jnp.asarray(owner),
        jnp.asarray(neighbor),
        jnp.asarray(closure_residual),
        mesh.mesh_id,
        geometry_id,
    )


__all__ = [
    "PreparedPolyhedralFiniteVolumeGeometry",
    "prepare_polyhedral_finite_volume_geometry",
]
