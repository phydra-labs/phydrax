#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from contextlib import nullcontext
from typing import assert_never, NamedTuple

import jax.numpy as jnp
import numpy as np

import phydrax.discretization._periodic_exact as periodic_exact_owner

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import exact_orient3d
from ...geometry._global_embedding import GlobalEmbeddingCertificate
from ...typing import checked
from .. import (
    _cell_geometry_validity as imported_cell_geometry_validity,
    _exact_power_consumers as imported_exact_power_consumers,
    _polygon_geometry as imported_polygon_geometry,
)
from .._cell_complex import (
    PolyhedralConnectivity,
    prepare_polyhedral_worksets,
)
from .._cell_geometry import CellGeometrySpec, owned_coordinate_preparation
from .._cell_mesh import CellMesh
from .._coordinate_budget import CoordinateEnclosureBudget
from .._exact_algebraic_geometry import ExactAlgebraicCellGeometrySource
from .._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from .._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from .._exact_rational_geometry import ExactRationalCellGeometrySource
from .._polygon_geometry import prepare_polyhedral_face_triangulation
from . import _polyhedral_types as geometry_types_owner


class _PeriodicPolyhedralFaces(NamedTuple):
    neighbor_cells: np.ndarray
    active: np.ndarray
    frame_maps: np.ndarray


@checked
def periodic_polyhedral_faces(
    mesh: CellMesh,
    /,
    *,
    actual_geometry: CellGeometrySpec | None = None,
    embedding: GlobalEmbeddingCertificate | None = None,
    budget: CoordinateEnclosureBudget,
) -> _PeriodicPolyhedralFaces:
    """Pair SCI quotient facets once, retaining the neighbor's physical lift.

    ``actual_geometry`` is the accepted coordinate owner when the periodic
    topology deliberately carries only global embedding identity. A certified
    ``embedding`` may provide independent coordinate-source evidence for mapped or restricted
    coordinates that intentionally retain no certified coordinate-source object.
    """
    match mesh.connectivity:
        case PolyhedralConnectivity() as connectivity:
            pass
        case _:
            raise TypeError(
                "periodic_polyhedral_faces.mesh.connectivity must be polyhedral; got another connectivity kind."
            )
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
            raise ValueError(
                "periodic_polyhedral_faces.actual_geometry must accompany periodic mesh topology; got a nonperiodic mesh."
            )
        return _PeriodicPolyhedralFaces(neighbor, active, maps)
    retained = periodic.actual_geometry
    geometry = retained if actual_geometry is None else actual_geometry
    if geometry is None:
        raise ValueError(
            "finite_volume_geometry.cell_geometry must bind the periodic source map; got no source geometry."
        )
    if retained is not None:
        if imported_cell_geometry_validity.cell_geometry_id(
            retained
        ) != imported_cell_geometry_validity.cell_geometry_id(geometry):
            raise ValueError(
                "periodic_polyhedral_faces.actual_geometry must match the retained coordinate source; got a different geometry ID."
            )
    if embedding is not None:
        if embedding.status != "certified":
            raise ValueError(
                "periodic_polyhedral_faces.embedding.status must be certified; got another embedding status."
            )
        embedding.binding.require(mesh, geometry)
    if (
        geometry.exact_source is None
        and geometry.periodic_source is None
        and embedding is None
    ):
        raise ValueError(
            "periodic_polyhedral_faces.cell_geometry must retain a coordinate source or certified embedding; got neither."
        )
    _, routes, _ = geometry.resolve(mesh, budget=budget)
    if any(
        route.ndim != 2 or route.shape[0] != block.cell_count
        for block, route in zip(mesh.blocks, routes, strict=True)
    ):
        raise ValueError(
            "periodic_polyhedral_faces.geometry_dofs must cover every mesh cell; got incompatible block routes."
        )
    owner_sign = np.zeros(owner.size, dtype=np.int32)
    offsets = np.asarray(connectivity.cell_face_offsets)
    face_rows = np.asarray(connectivity.cell_face_values)
    signs = np.asarray(connectivity.cell_face_sign_values)
    for cell in range(connectivity.cell_count):
        rows = slice(offsets[cell], offsets[cell + 1])
        incident = face_rows[rows]
        owned = owner[incident] == cell
        owner_sign[incident[owned]] = signs[rows][owned]

    degree = mesh.topological_dimension - 1
    start, stop = np.asarray(periodic.lifted_offsets)[degree : degree + 2]
    orbits = np.asarray(periodic.orbit_indices)[start:stop]
    orientations = np.asarray(periodic.orbit_orientations)[start:stop]
    shifts = np.asarray(periodic.orbit_shifts)[start:stop]
    generators, orders = periodic_exact_owner.exact_periodic_generators(
        periodic.cell, budget=budget
    )
    members: dict[int, list[int]] = {}
    for face in np.flatnonzero(neighbor < 0):
        members.setdefault(int(orbits[face]), []).append(int(face))
    for faces in members.values():
        if len(faces) == 1:
            continue
        if len(faces) != 2:
            raise ValueError(
                "periodic_polyhedral_faces.facet_orbit must have two reciprocal physical sides; got a different orbit cardinality."
            )
        left, right = faces
        if (
            orientations[left] * owner_sign[left]
            == orientations[right] * owner_sign[right]
        ):
            raise ValueError(
                "periodic_polyhedral_faces.facet_orientations must oppose on reciprocal sides; got matching oriented signs."
            )
        exponents = tuple(int(value) for value in shifts[right] - shifts[left])
        maps[left] = np.asarray(
            periodic_exact_owner.exact_periodic_element(
                generators, orders, exponents, budget=budget
            ),
            dtype=np.float64,
        )
        neighbor[left] = owner[right]
        neighbor[right] = owner[left]
        active[right] = False
    return _PeriodicPolyhedralFaces(neighbor, active, maps)


@checked
def prepare_polyhedral_finite_volume_geometry(
    mesh: CellMesh,
    /,
    *,
    planarity_tolerance: float = 1.0e-10,
    closure_tolerance: float = 1.0e-10,
    maximum_workset_entries: int = 100_000_000,
    cell_geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> geometry_types_owner.PreparedPolyhedralFiniteVolumeGeometry:
    """Prepare Newell/divergence geometry from prepared polyhedral connectivity.

    ``budget`` shares certified face/source preparation with the enclosing request.
    Standalone host bounds come from that recipe and numeric buffer shapes;
    native proof requests retain their separately declared native limits.
    """
    match mesh.connectivity:
        case PolyhedralConnectivity() as connectivity:
            pass
        case _:
            raise TypeError(
                "prepare_polyhedral_finite_volume_geometry.mesh.connectivity must be polyhedral; got another connectivity kind."
            )
    planarity = float(planarity_tolerance)
    closure = float(closure_tolerance)
    if (
        not np.isfinite(planarity)
        or planarity <= 0
        or not np.isfinite(closure)
        or closure <= 0
    ):
        raise ValueError(
            "prepare_polyhedral_finite_volume_geometry.tolerances must be positive and finite; got an invalid planarity or closure tolerance."
        )
    if cell_geometry is None and mesh.periodic_topology is not None:
        cell_geometry = mesh.periodic_topology.actual_geometry
        if cell_geometry is None:
            raise ValueError(
                "prepare_polyhedral_finite_volume_geometry.cell_geometry must retain the periodic coordinate source; got None."
            )
    if cell_geometry is not None:
        match cell_geometry.exact_source:
            case (
                ExactPowerCellGeometrySource()
                | ExactPowerCellGeometryRestrictionSource()
                | ExactPowerCellGeometryLinearActionSource()
            ):
                if budget is not None:
                    return (
                        imported_exact_power_consumers.exact_power_finite_volume_geometry(
                            mesh,
                            cell_geometry,
                            maximum_entries=maximum_workset_entries,
                            budget=budget,
                        )
                    )
                work, memory, queries = (
                    imported_exact_power_consumers.exact_power_finite_volume_preparation_bounds(
                        mesh, cell_geometry
                    )
                )
                with owned_coordinate_preparation(
                    work, memory, maximum_geometry_queries=queries
                ) as owner_budget:
                    return (
                        imported_exact_power_consumers.exact_power_finite_volume_geometry(
                            mesh,
                            cell_geometry,
                            maximum_entries=maximum_workset_entries,
                            budget=owner_budget,
                        )
                    )
            case ExactAlgebraicCellGeometrySource():
                raise ValueError(
                    "prepare_polyhedral_finite_volume_geometry.cell_geometry must use a power-polyhedron source; got an algebraic nodal source"
                )
            case (
                None
                | ExactPlcCellGeometrySource()
                | ExactPlcCellGeometryConvexSource()
                | ExactRationalCellGeometrySource()
            ):
                raise ValueError(
                    "prepare_polyhedral_finite_volume_geometry.cell_geometry must use an ExactPowerCellGeometrySource construction; got another source kind."
                )
            case invalid:
                assert_never(invalid)
    numeric_memory = _polyhedral_numeric_preparation_bytes(mesh, connectivity)
    if budget is None:
        work, memory, queries = polyhedral_finite_volume_preparation_bounds(
            mesh, maximum_workset_entries=maximum_workset_entries
        )
        context = owned_coordinate_preparation(
            work, memory, maximum_geometry_queries=queries
        )
    else:
        context = nullcontext(budget)
    with context as owner_budget, owner_budget.temporary_scope():
        owner_budget.reserve(0, numeric_memory)
        return _prepare_raw_polyhedral_finite_volume_geometry(
            mesh,
            connectivity,
            planarity=planarity,
            closure=closure,
            maximum_workset_entries=maximum_workset_entries,
            budget=owner_budget,
        )


def _polyhedral_numeric_preparation_bytes(
    mesh: CellMesh, connectivity: PolyhedralConnectivity, /
) -> int:
    faces, cells = connectivity.face_count, connectivity.cell_count
    face_width = connectivity.maximum_face_arity
    cell_width = connectivity.maximum_cell_faces
    vertex_width = connectivity.maximum_cell_vertices
    face_capacity = max(face_width - 2, 0)
    cell_capacity = cell_width * face_capacity
    dimension = mesh.ambient_dimension
    index_bytes = np.dtype(np.int32).itemsize
    mask_bytes = np.dtype(np.bool_).itemsize
    scalar_bytes = np.dtype(np.float64).itemsize
    # Dense worksets and their host views, with every index/sign/mask shape.
    worksets = 2 * (
        faces * face_width * (index_bytes + mask_bytes)
        + faces
        * face_width
        * (index_bytes + connectivity.face_edge_sign_values.dtype.itemsize + mask_bytes)
        + cells
        * cell_width
        * (index_bytes + connectivity.cell_face_sign_values.dtype.itemsize + mask_bytes)
        + cells * vertex_width * (index_bytes + mask_bytes)
    )
    quadrature = (faces * face_capacity + cells * cell_capacity) * (
        (dimension + 1) * scalar_bytes + mask_bytes
    )
    cell_arrays = cells * (dimension + 2) * scalar_bytes
    # Both face-loop triangle banks, vectors, centroids, weights and four
    # contiguous predicate inputs; ten small vector-operation temporaries.
    temporary_scalars = (
        2 * 3 * dimension * face_capacity
        + 2 * dimension * face_capacity
        + face_capacity
        + 4 * dimension * face_capacity
        + 10 * dimension
        + vertex_width * dimension
        + cell_width
    )
    temporary = (
        temporary_scalars * scalar_bytes + face_capacity * np.dtype(np.int8).itemsize
    )
    points = mesh.coordinates.size * scalar_bytes
    routing = 2 * faces * index_bytes
    return points + worksets + quadrature + cell_arrays + temporary + routing


@checked
def polyhedral_finite_volume_preparation_bounds(
    mesh: CellMesh,
    /,
    *,
    maximum_workset_entries: int,
    cell_geometry: CellGeometrySpec | None = None,
) -> tuple[int, int, int]:
    """Bound the source-aware FV preparation route from its recipe and shapes."""

    match mesh.connectivity:
        case PolyhedralConnectivity() as connectivity:
            pass
        case _:
            raise TypeError(
                "prepare_polyhedral_finite_volume_geometry.mesh.connectivity must be polyhedral; got another connectivity kind."
            )
    if cell_geometry is None and mesh.periodic_topology is not None:
        cell_geometry = mesh.periodic_topology.actual_geometry
        if cell_geometry is None:
            raise ValueError(
                "prepare_polyhedral_finite_volume_geometry.cell_geometry must retain the periodic coordinate source; got None."
            )
    if cell_geometry is not None:
        match cell_geometry.exact_source:
            case (
                ExactPowerCellGeometrySource()
                | ExactPowerCellGeometryRestrictionSource()
                | ExactPowerCellGeometryLinearActionSource()
            ):
                return imported_exact_power_consumers.exact_power_finite_volume_preparation_bounds(
                    mesh, cell_geometry
                )
            case ExactAlgebraicCellGeometrySource():
                raise ValueError(
                    "polyhedral_finite_volume_preparation_bounds.cell_geometry must use a power-polyhedron source; got an algebraic nodal source"
                )
            case (
                None
                | ExactPlcCellGeometrySource()
                | ExactPlcCellGeometryConvexSource()
                | ExactRationalCellGeometrySource()
            ):
                raise ValueError(
                    "prepare_polyhedral_finite_volume_geometry.cell_geometry must use an ExactPowerCellGeometrySource construction; got another source kind."
                )
            case invalid:
                assert_never(invalid)

    face_work, face_memory, face_queries = (
        imported_polygon_geometry.polyhedral_face_preparation_bounds(
            mesh, maximum_entries=maximum_workset_entries
        )
    )
    numeric_memory = _polyhedral_numeric_preparation_bytes(mesh, connectivity)
    # Native predicate work and queries are telemetry, not mechanical debits.
    return face_work, face_memory + numeric_memory, face_queries


def _prepare_raw_polyhedral_finite_volume_geometry(
    mesh: CellMesh,
    connectivity: PolyhedralConnectivity,
    /,
    *,
    planarity: float,
    closure: float,
    maximum_workset_entries: int,
    budget: CoordinateEnclosureBudget,
) -> geometry_types_owner.PreparedPolyhedralFiniteVolumeGeometry:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    worksets = prepare_polyhedral_worksets(
        connectivity,
        maximum_entries=maximum_workset_entries,
    )
    face_count, max_vertices = worksets.face_vertices.shape
    prepared_faces = prepare_polyhedral_face_triangulation(
        mesh,
        planarity_tolerance=planarity,
        maximum_entries=maximum_workset_entries,
        budget=budget,
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
                    "prepare_polyhedral_finite_volume_geometry.cell_centers must have positive facet visibility; got an unresolved or nonpositive orientation."
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
                        "prepare_polyhedral_finite_volume_geometry.cell_centers must lie strictly inside facet halfspaces; got a nonpositive star visibility."
                    )
                centroid = (star + np.sum(triangle, axis=0)) / 4.0
                volume_sum += volume
                first_moment += volume * centroid
                cell_q_points[cell, slot] = centroid
                cell_q_weights[cell, slot] = volume
                cell_q_valid[cell, slot] = True
                slot += 1
        if not np.isfinite(volume_sum) or volume_sum <= 0:
            raise ValueError(
                "prepare_polyhedral_finite_volume_geometry.cell_volumes must be positive; got nonpositive oriented volume."
            )
        cell_volumes[cell] = volume_sum
        cell_centers[cell] = first_moment / volume_sum
        closure_residual[cell] = np.linalg.norm(closure_vector) / max(
            np.sum(face_measures[cell_faces[cell, cell_face_valid[cell]]]), 1.0
        )
        if closure_residual[cell] > closure:
            raise ValueError(
                "prepare_polyhedral_finite_volume_geometry.area_vectors must close within tolerance; got an excessive oriented closure residual."
            )
        weight_sum = np.sum(cell_q_weights[cell])
        if not np.isclose(
            weight_sum,
            volume_sum,
            atol=closure * max(1.0, volume_sum),
            rtol=0,
        ):
            raise ValueError(
                "prepare_polyhedral_finite_volume_geometry.cell_quadrature_weights must reproduce cell volume; got a quadrature mass defect above tolerance."
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
    return geometry_types_owner.PreparedPolyhedralFiniteVolumeGeometry(
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
    "prepare_polyhedral_finite_volume_geometry",
    "polyhedral_finite_volume_preparation_bounds",
]
