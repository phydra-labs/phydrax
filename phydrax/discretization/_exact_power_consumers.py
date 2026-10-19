#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import math
from fractions import Fraction

import jax
import numpy as np

from phydrax.discretization._polygon_types import PolyhedralFaceTriangulation

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..geometry._exact_polyhedral_geometry import (
    determinant3,
    exact_vertices,
    star_tetrahedra,
    triangulate_loop,
    triangulation_bounds,
)
from ..typing import checked
from ._cell_complex import PolyhedralConnectivity, prepare_polyhedral_worksets
from ._cell_geometry import (
    cell_geometry_preparation_bounds,
    cell_geometry_preparation_query_upper,
    CellGeometrySpec,
)
from ._cell_geometry_validity import cell_geometry_id
from ._cell_mesh import CellMesh
from ._coordinate_budget import CoordinateEnclosureBudget
from .finite_volume._polyhedral_types import PreparedPolyhedralFiniteVolumeGeometry


@checked
def exact_power_finite_volume_preparation_bounds(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[int, int, int]:
    """Bound the source, face proofs, star determinants, and FV arrays."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "exact_power_finite_volume.connectivity must be polyhedral; got another connectivity kind"
        )
    source = geometry.exact_source
    if source is None:
        raise ValueError(
            "exact_power_finite_volume.geometry.exact_source must retain the coordinate source; got None"
        )
    bits = max(source.maximum_bits, 1075)
    source_work, source_memory = cell_geometry_preparation_bounds(geometry)
    work, memory = 2 * source_work, 2 * source_memory
    offsets = np.asarray(connectivity.face_vertex_offsets)
    sizes = np.diff(offsets)
    faces = np.asarray(connectivity.cell_face_values)
    pieces = 0
    for count in sizes:
        size = int(count)
        upper, storage = triangulation_bounds(size, bits)
        work += upper + 3 * size
        memory += storage
    for face in faces:
        size = int(sizes[face])
        upper, storage = triangulation_bounds(size, bits)
        work += upper + 3 * size
        memory += storage
        pieces += max(size - 2, 0)
    # Star positivity and physical cell integration each form a determinant.
    work += 2 * pieces * (4 * 3**3)
    face_capacity = max((max(int(size) - 2, 0) for size in sizes), default=0)
    # A cell cannot retain more fan entries than all admitted face occurrences.
    cell_entries = connectivity.cell_count * pieces
    face_entries = connectivity.face_count * face_capacity
    memory += (cell_entries + face_entries) * (
        3 * np.dtype(np.float64).itemsize
        + np.dtype(np.float64).itemsize
        + np.dtype(np.bool_).itemsize
    )
    queries = 4 * cell_geometry_preparation_query_upper(geometry)
    return work, memory, queries


@checked
def exact_power_cell_measures(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    maximum_work: int,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, bool]:

    ledger = budget
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "exact_power_cell_measures.connectivity must be polyhedral; got another connectivity kind"
        )
    face_sizes = np.diff(np.asarray(connectivity.face_vertex_offsets))
    ledger.reserve(
        sum(int(size) ** 3 for size in face_sizes)
        + 40 * connectivity.cell_face_values.size
    )
    stars = star_tetrahedra(
        mesh, exact_vertices(mesh, geometry, budget=budget), budget=budget
    )
    if sum(len(star) * 40 for star in stars) > maximum_work:
        raise ValueError(
            "exact_power_cell_measures.maximum_work must admit cell integration; got an exhausted work budget"
        )
    integrals = tuple(
        sum(
            (
                determinant3(
                    *(
                        tuple(
                            value - base
                            for value, base in zip(point, tet[0], strict=True)
                        )
                        for point in tet[1:]
                    ),
                    budget=budget,
                )
                / 6
                for tet in star
            ),
            Fraction(0),
        )
        for star in stars
    )
    values = np.asarray(integrals, dtype=np.float64)
    errors = np.asarray(
        tuple(
            float(np.nextafter(float(abs(value - Fraction(float(rounded)))), math.inf))
            if value != Fraction(float(rounded))
            else 0.0
            for value, rounded in zip(integrals, values, strict=True)
        ),
        dtype=np.float64,
    )
    return values, errors, True


@checked
def exact_power_face_triangulation(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    maximum_entries: int,
    budget: CoordinateEnclosureBudget,
) -> PolyhedralFaceTriangulation:
    c = mesh.connectivity
    if not isinstance(c, PolyhedralConnectivity):
        raise TypeError(
            "exact_power_face_triangulation.connectivity must be polyhedral; got another connectivity kind"
        )
    offsets, values = np.asarray(c.face_vertex_offsets), np.asarray(c.face_vertex_values)
    if sum(int(size) ** 3 for size in np.diff(offsets)) > maximum_entries:
        raise ValueError(
            "exact_power_face_triangulation.maximum_entries must admit face preparation work; got exhausted work capacity"
        )
    points = exact_vertices(mesh, geometry, budget=budget)
    triangles, triangle_offsets, vectors, measures, centers = [], [0], [], [], []
    work = 0
    for face in range(c.face_count):
        row = tuple(values[offsets[face] : offsets[face + 1]].tolist())
        work += len(row) ** 3
        if work > maximum_entries:
            raise ValueError(
                "exact_power_face_triangulation.maximum_entries must admit face preparation work; got exhausted work capacity"
            )
        pieces = triangulate_loop(points, row, budget=budget)
        triangles.extend(pieces)
        if 3 * len(triangles) > maximum_entries:
            raise ValueError(
                "exact_power_face_triangulation.maximum_entries must admit retained triangles; got exhausted retained capacity"
            )
        triangle_offsets.append(len(triangles))
        area = [Fraction(0)] * 3
        moment = [Fraction(0)] * 3
        pivot = None
        signed_area = Fraction(0)
        for piece in pieces:
            first, second, third = (tuple(points[index]) for index in piece)
            a, b = (
                tuple(y - x for x, y in zip(first, second, strict=True)),
                tuple(y - x for x, y in zip(first, third, strict=True)),
            )
            vector = (
                (a[1] * b[2] - a[2] * b[1]) / 2,
                (a[2] * b[0] - a[0] * b[2]) / 2,
                (a[0] * b[1] - a[1] * b[0]) / 2,
            )
            if pivot is None:
                pivot = next(axis for axis in range(3) if vector[axis])
            signed_area += vector[pivot]
            for axis in range(3):
                area[axis] += vector[axis]
                moment[axis] += (
                    vector[pivot] * (first[axis] + second[axis] + third[axis]) / 3
                )
        vectors.append(tuple(float(value) for value in area))
        measures.append(
            math.sqrt(float(sum((value * value for value in area), Fraction(0))))
        )
        centers.append(tuple(float(value / signed_area) for value in moment))
    identity = canonical_fingerprint(
        {
            "kind": "exact-power-source-face-triangulation",
            "geometry": cell_geometry_id(geometry),
            "triangles": array_tree_fingerprint(np.asarray(triangles, dtype=np.int32)),
            "maximum_entries": maximum_entries,
        }
    )
    # This route did not execute a native triangulator; the empty planar work
    # bank and allocator-record tuple bind no fabricated native measurements.
    return PolyhedralFaceTriangulation(
        np.asarray(triangle_offsets, dtype=np.int64),
        np.asarray(triangles, dtype=np.int32),
        np.asarray(vectors, dtype=np.float64),
        np.asarray(measures, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
        np.empty((0,), dtype=np.int64),
        (),
        (identity,),
        identity,
    )


@checked
def exact_power_finite_volume_geometry(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    maximum_entries: int,
    budget: CoordinateEnclosureBudget,
) -> PreparedPolyhedralFiniteVolumeGeometry:

    c = mesh.connectivity
    if not isinstance(c, PolyhedralConnectivity):
        raise TypeError(
            "exact_power_finite_volume.connectivity must be polyhedral; got another connectivity kind"
        )
    worksets = prepare_polyhedral_worksets(c, maximum_entries=maximum_entries)
    points = exact_vertices(mesh, geometry, budget=budget)
    faces = exact_power_face_triangulation(
        mesh,
        geometry,
        maximum_entries=maximum_entries,
        budget=budget,
    )
    stars = star_tetrahedra(mesh, points, budget=budget)
    face_capacity = worksets.face_vertices.shape[1] - 2
    cell_capacity = max(len(star) for star in stars)
    retained_entries = 5 * (len(stars) * cell_capacity + c.face_count * face_capacity)
    if retained_entries > maximum_entries:
        raise ValueError(
            "exact_power_finite_volume.maximum_entries must admit retained quadrature; got exhausted retained capacity"
        )
    cell_points, cell_weights, cell_valid = (
        np.zeros((len(stars), cell_capacity, 3), dtype=np.float64),
        np.zeros((len(stars), cell_capacity), dtype=np.float64),
        np.zeros((len(stars), cell_capacity), dtype=np.bool_),
    )
    volumes, centers = [], []
    for cell, star in enumerate(stars):
        volume, moment = Fraction(0), [Fraction(0)] * 3
        for slot, tet in enumerate(star):
            measure = (
                determinant3(
                    *(
                        tuple(
                            value - base
                            for value, base in zip(point, tet[0], strict=True)
                        )
                        for point in tet[1:]
                    ),
                    budget=budget,
                )
                / 6
            )
            center = tuple(
                sum((point[axis] for point in tet), Fraction(0)) / 4 for axis in range(3)
            )
            volume += measure
            for axis in range(3):
                moment[axis] += measure * center[axis]
            cell_points[cell, slot], cell_weights[cell, slot], cell_valid[cell, slot] = (
                center,
                measure,
                True,
            )
        volumes.append(float(volume))
        centers.append(tuple(float(value / volume) for value in moment))
    face_points, face_weights, face_valid = (
        np.zeros((c.face_count, face_capacity, 3), dtype=np.float64),
        np.zeros((c.face_count, face_capacity), dtype=np.float64),
        np.zeros((c.face_count, face_capacity), dtype=np.bool_),
    )
    for face in range(c.face_count):
        start, stop = faces.triangle_offsets[face : face + 2]
        for slot, row in enumerate(faces.triangle_vertices[start:stop]):
            first, second, third = points[row]
            vector = np.cross(second - first, third - first) / 2
            face_points[face, slot] = (first + second + third) / 3
            face_weights[face, slot] = math.sqrt(
                float(sum(value * value for value in vector))
            )
            face_valid[face, slot] = True
    signs, cell_faces, valid = (
        np.asarray(worksets.cell_face_signs),
        np.asarray(worksets.cell_faces),
        np.asarray(worksets.cell_face_valid),
    )
    # Topological oriented closure plus face-plane proof establishes zero
    # ideal closure; numerical residual is separately exposed to FV consumers.
    residual = np.linalg.norm(
        np.sum(
            np.where(
                valid[..., None],
                signs[..., None] * faces.face_area_vectors[cell_faces],
                0.0,
            ),
            axis=1,
        ),
        axis=1,
    )
    identity = canonical_fingerprint(
        {
            "kind": "exact-power-finite-volume-geometry",
            "geometry": cell_geometry_id(geometry),
            "faces": faces.triangulation_id,
        }
    )
    return PreparedPolyhedralFiniteVolumeGeometry(
        cell_volumes=jax.device_put(np.asarray(volumes, dtype=np.float64)),
        cell_centers=jax.device_put(np.asarray(centers, dtype=np.float64)),
        face_centers=jax.device_put(faces.face_centroids),
        face_area_vectors=jax.device_put(faces.face_area_vectors),
        face_measures=jax.device_put(faces.face_measures),
        face_quadrature_points=jax.device_put(face_points),
        face_quadrature_weights=jax.device_put(face_weights),
        face_quadrature_valid=jax.device_put(face_valid),
        cell_quadrature_points=jax.device_put(cell_points),
        cell_quadrature_weights=jax.device_put(cell_weights),
        cell_quadrature_valid=jax.device_put(cell_valid),
        owner_cells=c.face_owner,
        neighbor_cells=c.face_neighbor,
        closure_residual=jax.device_put(residual),
        mesh_id=mesh.mesh_id,
        geometry_id=identity,
        face_triangulation_id=faces.triangulation_id,
    )
