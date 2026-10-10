#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import exact_orient3d
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._mesh_certificates import GlobalEmbeddingCertificate
from .._cell_complex import (
    PolyhedralConnectivity,
    prepare_polyhedral_worksets,
)
from .._cell_geometry import CellGeometrySpec
from .._cell_mesh import CellMesh
from .._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from .._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from .._polygon_geometry import prepare_polyhedral_face_triangulation


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
    neighbor_cells: Array
    closure_residual: Array
    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    face_triangulation_id: str = eqx.field(static=True)


def _periodic_polyhedral_faces(
    mesh: CellMesh,
    /,
    *,
    actual_geometry: CellGeometrySpec | None = None,
    embedding: GlobalEmbeddingCertificate | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair SCI quotient facets once, retaining the neighbor's physical lift.

    ``actual_geometry`` is the accepted coordinate owner when the periodic
    topology deliberately carries only global embedding identity. A certified
    ``embedding`` may provide independent authority for mapped or restricted
    coordinates that intentionally retain no exact coordinate-source object.
    """
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Periodic polyhedral faces require polyhedral connectivity.")
    owner = np.asarray(connectivity.face_owner, dtype=np.int32)
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int32).copy()
    active = np.ones(owner.size, dtype=np.bool_)
    dimension = mesh.ambient_dimension
    maps = np.broadcast_to(
        np.eye(dimension + 1),
        (owner.size, dimension + 1, dimension + 1),
    ).copy()
    periodic = mesh.periodic_topology
    if periodic is None:
        if actual_geometry is not None:
            raise ValueError("Explicit periodic geometry requires a periodic mesh.")
        return neighbor, active, maps
    retained = periodic.actual_geometry
    geometry = retained if actual_geometry is None else actual_geometry
    if geometry is None:
        raise ValueError(
            "Periodic polyhedral FV requires exact source geometry authority."
        )
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("actual_geometry must be a CellGeometrySpec.")
    if retained is not None:
        from .._cell_geometry_validity import cell_geometry_id

        if cell_geometry_id(retained) != cell_geometry_id(geometry):
            raise ValueError(
                "Explicit periodic geometry differs from retained authority."
            )
    if embedding is not None:
        if not isinstance(embedding, GlobalEmbeddingCertificate):
            raise TypeError("embedding must be a GlobalEmbeddingCertificate.")
        if embedding.status != "certified":
            raise ValueError("Periodic FV requires a certified global embedding.")
        embedding.binding.require(mesh, geometry)
    if (
        geometry.exact_source is None
        and geometry.periodic_source is None
        and embedding is None
    ):
        raise ValueError(
            "Periodic polyhedral FV requires exact source or embedding authority."
        )
    _, routes, _ = geometry.resolve(mesh)
    if any(
        route.ndim != 2 or route.shape[0] != block.cell_count
        for block, route in zip(mesh.blocks, routes, strict=True)
    ):
        raise ValueError("Periodic FV geometry does not cover the mesh cells.")
    owner_sign = np.zeros(owner.size, dtype=np.int32)
    offsets = np.asarray(connectivity.cell_face_offsets)
    face_rows = np.asarray(connectivity.cell_face_values)
    signs = np.asarray(connectivity.cell_face_sign_values)
    for cell in range(connectivity.cell_count):
        rows = slice(offsets[cell], offsets[cell + 1])
        incident = face_rows[rows]
        owned = owner[incident] == cell
        owner_sign[incident[owned]] = signs[rows][owned]
    from .._periodic_topology import _exact_periodic_element, _exact_periodic_generators

    degree = mesh.topological_dimension - 1
    start, stop = np.asarray(periodic.lifted_offsets)[degree : degree + 2]
    orbits = np.asarray(periodic.orbit_indices)[start:stop]
    orientations = np.asarray(periodic.orbit_orientations)[start:stop]
    shifts = np.asarray(periodic.orbit_shifts)[start:stop]
    generators, orders = _exact_periodic_generators(periodic.cell)
    members: dict[int, list[int]] = {}
    for face in np.flatnonzero(neighbor < 0):
        members.setdefault(int(orbits[face]), []).append(int(face))
    for faces in members.values():
        if len(faces) == 1:
            continue
        if len(faces) != 2:
            raise ValueError(
                "Periodic FV facet orbit must have two reciprocal physical sides."
            )
        left, right = faces
        if (
            orientations[left] * owner_sign[left]
            == orientations[right] * owner_sign[right]
        ):
            raise ValueError("Periodic FV reciprocal facet orientations must oppose.")
        exponents = tuple(int(value) for value in shifts[right] - shifts[left])
        maps[left] = np.asarray(
            _exact_periodic_element(generators, orders, exponents), dtype=np.float64
        )
        neighbor[left] = owner[right]
        neighbor[right] = owner[left]
        active[right] = False
    return neighbor, active, maps


def prepare_polyhedral_finite_volume_geometry(
    mesh: CellMesh,
    /,
    *,
    planarity_tolerance: float = 1.0e-10,
    closure_tolerance: float = 1.0e-10,
    maximum_workset_entries: int = 100_000_000,
    cell_geometry: CellGeometrySpec | None = None,
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
    if cell_geometry is None and mesh.periodic_topology is not None:
        cell_geometry = mesh.periodic_topology.actual_geometry
        if cell_geometry is None:
            raise ValueError(
                "Periodic polyhedral geometry cannot use rounded carrier coordinates."
            )
    if cell_geometry is not None:
        from .._exact_power_consumers import exact_power_finite_volume_geometry

        match cell_geometry.exact_source:
            case (
                ExactPowerCellGeometrySource()
                | ExactPowerCellGeometryRestrictionSource()
                | ExactPowerCellGeometryLinearActionSource()
            ):
                return exact_power_finite_volume_geometry(
                    mesh, cell_geometry, maximum_entries=maximum_workset_entries
                )
            case None | ExactPlcCellGeometrySource() | ExactPlcCellGeometryConvexSource():
                raise ValueError(
                    "Polyhedral FV source geometry requires exact power construction."
                )
            case invalid:
                assert_never(invalid)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    connectivity = mesh.connectivity
    worksets = prepare_polyhedral_worksets(
        connectivity,
        maximum_entries=maximum_workset_entries,
    )
    face_count, max_vertices = worksets.face_vertices.shape
    prepared_faces = prepare_polyhedral_face_triangulation(
        mesh, planarity_tolerance=planarity, maximum_entries=maximum_workset_entries
    )
    triangle_offsets = prepared_faces.triangle_offsets
    triangle_vertices = prepared_faces.triangle_vertices
    face_centers = prepared_faces.face_centroids
    area_vectors = prepared_faces.face_area_vectors
    face_measures = prepared_faces.face_measures
    face_q_points = np.zeros(
        (face_count, max_vertices - 2, points.shape[1]), dtype=np.float64
    )
    face_q_weights = np.zeros((face_count, max_vertices - 2), dtype=np.float64)
    face_q_valid = np.zeros((face_count, max_vertices - 2), dtype=np.bool_)
    for face in range(face_count):
        start, stop = triangle_offsets[face : face + 2]
        triangles = points[triangle_vertices[start:stop]]
        vectors = (
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            / 2
        )
        unit = area_vectors[face] / np.linalg.norm(area_vectors[face])
        count = int(stop - start)
        face_q_points[face, :count] = np.mean(triangles, axis=1)
        face_q_weights[face, :count] = vectors @ unit
        face_q_valid[face, :count] = True
    cell_faces = np.asarray(worksets.cell_faces, dtype=np.int32)
    cell_face_signs = np.asarray(worksets.cell_face_signs, dtype=np.int8)
    cell_face_valid = np.asarray(worksets.cell_face_valid, dtype=np.bool_)
    cell_vertices = np.asarray(worksets.cell_vertices, dtype=np.int32)
    cell_vertex_valid = np.asarray(worksets.cell_vertex_valid, dtype=np.bool_)
    cell_count, max_faces = cell_faces.shape
    cell_capacity = max_faces * (max_vertices - 2)
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
            start, stop = triangle_offsets[face : face + 2]
            closure_vector += float(sign) * area_vectors[face]
            face_triangles = points[triangle_vertices[start:stop]]
            signs = exact_orient3d(
                np.broadcast_to(star, face_triangles[:, 0].shape),
                face_triangles[:, 0],
                face_triangles[:, 1],
                face_triangles[:, 2],
            )
            if np.any(signs * int(sign) <= 0):
                raise ValueError(
                    "Polyhedral cell has unresolved/nonpositive exact star visibility."
                )
            for triangle in face_triangles:
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
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int32)
    geometry_id = canonical_fingerprint(
        {
            "kind": "prepared-polyhedral-finite-volume-geometry",
            "mesh": mesh.mesh_id,
            "planarity_tolerance": planarity,
            "closure_tolerance": closure,
            "maximum_workset_entries": int(maximum_workset_entries),
            "face_triangulation": prepared_faces.triangulation_id,
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
        prepared_faces.triangulation_id,
    )


__all__ = [
    "PreparedPolyhedralFiniteVolumeGeometry",
    "prepare_polyhedral_finite_volume_geometry",
]
