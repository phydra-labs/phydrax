#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_complex import (
    PolyhedralConnectivity,
    prepare_polyhedral_worksets,
)
from .._cell_mesh import CellMesh


class PreparedPolyhedralFiniteVolumeGeometry(StrictModule, NonTrainableState):
    """Certified fixed-capacity planar-face polyhedral FV geometry."""

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
    neighbour_cells: Array
    closure_residual: Array
    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


def prepare_polyhedral_finite_volume_geometry(
    mesh: CellMesh,
    /,
    *,
    planarity_tolerance: float = 1.0e-10,
    closure_tolerance: float = 1.0e-10,
    maximum_workset_entries: int = 100_000_000,
) -> PreparedPolyhedralFiniteVolumeGeometry:
    """Prepare Newell/divergence geometry from canonical polyhedral connectivity."""
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
    points = np.asarray(mesh.coordinates, dtype=float)
    connectivity = mesh.connectivity
    worksets = prepare_polyhedral_worksets(
        connectivity,
        maximum_entries=maximum_workset_entries,
    )
    face_vertices = np.asarray(worksets.face_vertices, dtype=np.int32)
    face_valid = np.asarray(worksets.face_vertex_valid, dtype=bool)
    face_count, max_vertices = face_vertices.shape
    face_centers = np.zeros((face_count, points.shape[1]), dtype=float)
    area_vectors = np.zeros_like(face_centers)
    face_measures = np.zeros((face_count,), dtype=float)
    face_q_points = np.zeros((face_count, max_vertices - 2, points.shape[1]), dtype=float)
    face_q_weights = np.zeros((face_count, max_vertices - 2), dtype=float)
    face_q_valid = np.zeros((face_count, max_vertices - 2), dtype=bool)
    for face in range(face_count):
        polygon = points[face_vertices[face, face_valid[face]]]
        if polygon.shape[0] < 3:
            raise ValueError("Every polyhedral face requires at least three vertices.")
        newell = 0.5 * np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0)
        measure = float(np.linalg.norm(newell))
        if not np.isfinite(measure) or measure <= 0:
            raise ValueError("Polyhedral faces require positive finite measure.")
        unit = newell / measure
        distance = (polygon - polygon[0]) @ unit
        scale = max(1.0, float(np.max(np.linalg.norm(polygon - polygon[0], axis=1))))
        if np.max(np.abs(distance)) > planarity * scale:
            raise ValueError("Polyhedral face fails its planarity certificate.")
        triangle_areas, triangle_centers = [], []
        for local in range(1, polygon.shape[0] - 1):
            triangle = polygon[[0, local, local + 1]]
            vector = 0.5 * np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            weight = float(np.dot(vector, unit))
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError(
                    "Polyhedral face loop is not consistently oriented/convex."
                )
            center = np.mean(triangle, axis=0)
            triangle_areas.append(weight)
            triangle_centers.append(center)
            face_q_points[face, local - 1] = center
            face_q_weights[face, local - 1] = weight
            face_q_valid[face, local - 1] = True
        face_centers[face] = np.average(
            np.asarray(triangle_centers), axis=0, weights=np.asarray(triangle_areas)
        )
        area_vectors[face], face_measures[face] = newell, measure
    cell_faces = np.asarray(worksets.cell_faces, dtype=np.int32)
    cell_face_signs = np.asarray(worksets.cell_face_signs, dtype=np.int8)
    cell_face_valid = np.asarray(worksets.cell_face_valid, dtype=bool)
    cell_vertices = np.asarray(worksets.cell_vertices, dtype=np.int32)
    cell_vertex_valid = np.asarray(worksets.cell_vertex_valid, dtype=bool)
    cell_count, max_faces = cell_faces.shape
    cell_capacity = max_faces * (max_vertices - 2)
    cell_centers = np.zeros((cell_count, points.shape[1]), dtype=float)
    cell_volumes = np.zeros((cell_count,), dtype=float)
    cell_q_points = np.zeros((cell_count, cell_capacity, points.shape[1]), dtype=float)
    cell_q_weights = np.zeros((cell_count, cell_capacity), dtype=float)
    cell_q_valid = np.zeros((cell_count, cell_capacity), dtype=bool)
    closure_residual = np.zeros((cell_count,), dtype=float)
    for cell in range(cell_count):
        volume_sum = 0.0
        first_moment = np.zeros((points.shape[1],), dtype=float)
        closure_vector = np.zeros_like(first_moment)
        star = np.mean(points[cell_vertices[cell, cell_vertex_valid[cell]]], axis=0)
        slot = 0
        for face, sign in zip(
            cell_faces[cell, cell_face_valid[cell]],
            cell_face_signs[cell, cell_face_valid[cell]],
            strict=True,
        ):
            polygon = points[face_vertices[face, face_valid[face]]]
            closure_vector += float(sign) * area_vectors[face]
            for local in range(1, polygon.shape[0] - 1):
                triangle = polygon[[0, local, local + 1]]
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
                        "Polyhedral cell is not star-shaped about its certified centre."
                    )
                centroid = (star + np.sum(triangle, axis=0)) / 4.0
                volume_sum += volume
                first_moment += volume * centroid
                cell_q_points[cell, slot] = centroid
                cell_q_weights[cell, slot] = volume
                cell_q_valid[cell, slot] = True
                slot += 1
        if not np.isfinite(volume_sum) or volume_sum <= 0:
            raise ValueError("Polyhedral cells require positive oriented volume.")
        cell_volumes[cell] = volume_sum
        cell_centers[cell] = first_moment / volume_sum
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
    neighbour = np.asarray(connectivity.face_neighbour, dtype=np.int32)
    geometry_id = canonical_fingerprint(
        {
            "kind": "prepared-polyhedral-finite-volume-geometry",
            "mesh": mesh.mesh_id,
            "planarity_tolerance": planarity,
            "closure_tolerance": closure,
            "maximum_workset_entries": int(maximum_workset_entries),
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
        jnp.asarray(neighbour),
        jnp.asarray(closure_residual),
        mesh.mesh_id,
        geometry_id,
    )


__all__ = [
    "PreparedPolyhedralFiniteVolumeGeometry",
    "prepare_polyhedral_finite_volume_geometry",
]
