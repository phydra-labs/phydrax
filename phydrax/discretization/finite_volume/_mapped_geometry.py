#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._cell_complex import PolyhedralConnectivity
from .._cell_geometry import (
    CellGeometrySpec,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from .._cell_mesh import CellBlock, CellMesh, PolyhedralBlock
from .._mapped_geometry_measure import MappedCellMoments
from .._reference_cell import reference_cell_topology
from ..fem._generic import (
    _degree_aware_reference_rule,
    _GEOMETRY_WORKSET_BYTES,
    _reference_rule_exactness,
)


class MappedFiniteVolumeGeometry(NamedTuple):
    cell_points: Array
    cell_weights: Array
    cell_valid: Array
    face_centers: Array
    area_vectors: Array
    face_measures: Array
    face_points: Array
    face_weights: Array
    closure: Array


def mapped_constant_quadrature_exact(
    mesh: CellMesh, moments: MappedCellMoments, /
) -> bool:
    """Compare certified Jacobian degrees against the owning FE reference rule."""
    offset = 0
    for block in mesh.blocks:
        exactness = _reference_rule_exactness(block.cell_kind, 3)
        for degree in moments.jacobian_degrees[offset : offset + block.cell_count]:
            if degree is None:
                return False
            # The pyramid collapse introduces the chart substitution and its
            # two-power Jacobian; this bound is deliberately conservative.
            bound = 2 * degree + 2 if block.cell_kind == "pyramid" else degree
            if bound > exactness:
                return False
        offset += block.cell_count
    if offset != len(moments.jacobian_degrees):
        raise ValueError("Mapped Jacobian degree evidence does not cover the mesh.")
    return True


def _require_shared_face_traces(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    connectivity: PolyhedralConnectivity,
    owner: np.ndarray,
    neighbor: np.ndarray,
    /,
) -> None:
    """Prove whole exact facet equality when no embedding proof is supplied."""
    from ...geometry._embedding_cells import _iter_embedding_cells
    from ...geometry._global_embedding import _periodic_face_trace
    from .._coordinate_enclosure import expression_equal

    offsets = np.asarray(connectivity.face_vertex_offsets)
    vertices = np.asarray(connectivity.face_vertex_values)
    cell_ids = np.asarray(connectivity.cell_global_ids)
    exact_cells = {cell.global_id: cell for cell in _iter_embedding_cells(mesh, geometry)}
    for face in np.flatnonzero(neighbor >= 0):
        face_vertices = tuple(int(v) for v in vertices[offsets[face] : offsets[face + 1]])
        traces = []
        for cell_row in (int(owner[face]), int(neighbor[face])):
            cell = exact_cells[int(cell_ids[cell_row])]
            local_by_vertex = {int(v): i for i, v in enumerate(cell.vertex_indices)}
            try:
                local = tuple(local_by_vertex[v] for v in face_vertices)
            except KeyError as error:
                raise ValueError(
                    "Mapped FV face is not owned by both incident cells."
                ) from error
            trace = _periodic_face_trace(cell, local)
            if trace is None:
                raise ValueError("Mapped FV cannot prove the exact shared-face chart.")
            traces.append(trace)
        if not all(expression_equal(a, b) for a, b in zip(*traces, strict=True)):
            raise ValueError(
                "Mapped FV incident cells have different exact face geometry."
            )


def _evaluate_chunks(
    element: ScalarCoordinateElement,
    route: ArrayLike,
    rows: Sequence[int] | np.ndarray,
    points: ArrayLike,
    coordinates: np.ndarray,
    /,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate points and Jacobians with the FE geometry workset bound."""
    rows = np.asarray(rows, dtype=np.int32)
    route = np.asarray(route, dtype=np.int32)
    points = np.asarray(points)
    q, dimension = points.shape
    local = route.shape[1]
    per_point = (
        2 * local * (1 + dimension)
        + 3 * (1 + 2 * dimension)
        + 2 * dimension * dimension
        + 3
    )
    cell_bytes = coordinates.dtype.itemsize * (q * per_point + local * 3)
    chunk = max(1, _GEOMETRY_WORKSET_BYTES // cell_bytes)
    per_cell = isinstance(element, PerCellRestrictedCellGeometryElement)
    table = None if per_cell else element.tabulate(points)
    for start in range(0, len(rows), chunk):
        selected_rows = rows[start : start + chunk]
        nodes = coordinates[route[selected_rows]]
        if isinstance(element, PerCellRestrictedCellGeometryElement):
            values, gradients = element.tabulate_cells(selected_rows, points)
        else:
            assert table is not None
            values, gradients = table
        values = np.asarray(values, dtype=coordinates.dtype)
        gradients = np.asarray(gradients, dtype=coordinates.dtype)
        if per_cell:
            mapped = ein.contract("nqi,nid->nqd", values, nodes)
            jacobian = ein.contract("nqir,nid->nqdr", gradients, nodes)
        else:
            mapped = ein.contract("qi,nid->nqd", values, nodes)
            jacobian = ein.contract("qir,nid->nqdr", gradients, nodes)
        yield start, selected_rows, np.asarray(mapped), np.asarray(jacobian)


@dataclass(slots=True)
class _GeometryBuffers:
    cell_points: np.ndarray
    cell_weights: np.ndarray
    cell_valid: np.ndarray
    face_centers: np.ndarray
    area_vectors: np.ndarray
    face_measures: np.ndarray
    face_points: np.ndarray
    face_weights: np.ndarray
    assigned: np.ndarray

    @classmethod
    def allocate(
        cls, cells: int, faces: int, cell_q: int, face_q: int, dtype: np.dtype
    ) -> _GeometryBuffers:
        return cls(
            np.zeros((cells, cell_q, 3), dtype=dtype),
            np.zeros((cells, cell_q), dtype=dtype),
            np.zeros((cells, cell_q), dtype=np.bool_),
            np.zeros((faces, 3), dtype=dtype),
            np.zeros((faces, 3), dtype=dtype),
            np.zeros((faces,), dtype=dtype),
            np.zeros((faces, face_q, 3), dtype=dtype),
            np.zeros((faces, face_q), dtype=dtype),
            np.zeros((faces,), dtype=np.bool_),
        )


def _prepare_cells(
    block: CellBlock | PolyhedralBlock,
    element: ScalarCoordinateElement,
    route: ArrayLike,
    coordinates: np.ndarray,
    indices: np.ndarray,
    rule: tuple[Array, Array],
    buffers: _GeometryBuffers,
    /,
) -> None:
    points, weights = rule
    rows = np.arange(block.cell_count, dtype=np.int32)
    nq = points.shape[0]
    for _, selected_rows, mapped, jacobian in _evaluate_chunks(
        element, route, rows, points, coordinates
    ):
        determinant = np.linalg.det(jacobian)
        if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
            raise ValueError("Mapped FV quadrature lost its certified positive Jacobian.")
        selected = indices[selected_rows]
        buffers.cell_points[selected, :nq] = mapped
        buffers.cell_weights[selected, :nq] = determinant * np.asarray(weights)[None, :]
        buffers.cell_valid[selected, :nq] = True


def _facet_chart(
    vertices: np.ndarray, facet: tuple[int, ...], dtype: np.dtype, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if len(facet) == 3:
        uv, weights = _degree_aware_reference_rule("triangle", 3)
        axis2 = vertices[facet[2]] - vertices[facet[0]]
    elif len(facet) == 4:
        uv, weights = _degree_aware_reference_rule("quadrilateral", 3)
        axis2 = vertices[facet[3]] - vertices[facet[0]]
    else:
        raise ValueError("Mapped FV supports triangular and quadrilateral facets.")
    axis1 = vertices[facet[1]] - vertices[facet[0]]
    if len(facet) == 4 and not np.array_equal(
        vertices[facet[2]], vertices[facet[0]] + axis1 + axis2
    ):
        raise ValueError(
            "Mapped FV quadrilateral facet is not an affine reference chart."
        )
    uv = np.asarray(uv, dtype=dtype)
    reference = (
        np.asarray(vertices[facet[0]], dtype=dtype)
        + uv[:, :1] * axis1
        + uv[:, 1:] * axis2
    )
    normal = np.cross(axis1, axis2)
    outward = np.dot(
        normal, np.mean(vertices[list(facet)], axis=0) - np.mean(vertices, axis=0)
    )
    return reference, np.asarray(weights, dtype=dtype), axis1, axis2, float(outward)


def _prepare_face_group(
    element: ScalarCoordinateElement,
    route: ArrayLike,
    coordinates: np.ndarray,
    rows: list[int],
    faces: list[int],
    chart: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    buffers: _GeometryBuffers,
    /,
) -> None:
    reference, weights, axis1, axis2, outward = chart
    nq = len(weights)
    for start, selected_rows, mapped, jacobian in _evaluate_chunks(
        element, route, rows, reference, coordinates
    ):
        selected = np.asarray(faces[start : start + len(selected_rows)])
        tangent1 = ein.contract("nqij,j->nqi", jacobian, axis1)
        tangent2 = ein.contract("nqij,j->nqi", jacobian, axis2)
        normals = np.cross(tangent1, tangent2)
        normals *= 1.0 if outward > 0 else -1.0
        physical_weights = np.linalg.norm(normals, axis=-1) * weights[None, :]
        if np.any(~np.isfinite(physical_weights)) or np.any(physical_weights <= 0.0):
            raise ValueError(
                "Mapped FV face quadrature requires finite positive measures."
            )
        measures = np.sum(physical_weights, axis=1)
        buffers.face_points[selected, :nq] = mapped
        buffers.face_weights[selected, :nq] = physical_weights
        buffers.area_vectors[selected] = np.sum(normals * weights[None, :, None], axis=1)
        buffers.face_measures[selected] = measures
        buffers.face_centers[selected] = (
            np.sum(mapped * physical_weights[..., None], axis=1) / measures[:, None]
        )


def _prepare_block_faces(
    block: CellBlock | PolyhedralBlock,
    element: ScalarCoordinateElement,
    route: ArrayLike,
    coordinates: np.ndarray,
    indices: np.ndarray,
    owner: np.ndarray,
    faces: dict[tuple[int, ...], int],
    buffers: _GeometryBuffers,
    /,
) -> None:
    topology = reference_cell_topology(block.cell_kind)
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    for facet in topology.entities[2]:
        rows: list[int] = []
        face_rows: list[int] = []
        for row, cell in enumerate(np.asarray(block.vertices)):
            key = tuple(sorted(int(cell[i]) for i in facet))
            if key not in faces:
                raise ValueError("Mapped FV facet does not match the carrier incidence.")
            face = faces[key]
            if int(owner[face]) == int(indices[row]):
                rows.append(row)
                face_rows.append(face)
        if not rows:
            continue
        if np.any(buffers.assigned[face_rows]):
            raise ValueError("Mapped FV face has multiple owners.")
        buffers.assigned[face_rows] = True
        chart = _facet_chart(vertices, facet, coordinates.dtype)
        _prepare_face_group(element, route, coordinates, rows, face_rows, chart, buffers)


def _finish_geometry(
    buffers: _GeometryBuffers, owner: np.ndarray, neighbor: np.ndarray, /
) -> MappedFiniteVolumeGeometry:
    if not np.all(buffers.assigned):
        raise ValueError(
            "Mapped FV requires every carrier face to have a standard-cell owner."
        )
    if (
        np.any(~np.isfinite(buffers.cell_weights))
        or np.any(buffers.cell_weights[buffers.cell_valid] <= 0)
        or np.any(~np.isfinite(buffers.face_measures))
        or np.any(buffers.face_measures <= 0)
        or np.any(~np.isfinite(buffers.area_vectors))
        or np.any(~np.isfinite(buffers.face_centers))
    ):
        raise ValueError(
            "Mapped FV quadrature requires finite positive cell and face measures."
        )
    closure = np.zeros_like(buffers.cell_points[:, 0, :])
    np.add.at(closure, owner, buffers.area_vectors)
    interior = np.flatnonzero(neighbor >= 0)
    np.add.at(closure, neighbor[interior], -buffers.area_vectors[interior])
    return MappedFiniteVolumeGeometry(
        jnp.asarray(buffers.cell_points),
        jnp.asarray(buffers.cell_weights),
        jnp.asarray(buffers.cell_valid),
        jnp.asarray(buffers.face_centers),
        jnp.asarray(buffers.area_vectors),
        jnp.asarray(buffers.face_measures),
        jnp.asarray(buffers.face_points),
        jnp.asarray(buffers.face_weights),
        jnp.asarray(closure),
    )


def prepare_mapped_fv_geometry(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    owner: np.ndarray,
    neighbor: np.ndarray,
    /,
    *,
    embedding_certified: bool = False,
) -> MappedFiniteVolumeGeometry:
    """Prepare actual-map quadrature and integrated owner-oriented face normals.

    FV flux unknowns remain scalar face averages; these arrays do not claim a
    representation of general vector fluxes against pointwise curved normals.
    """
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Mapped FV requires polyhedral connectivity.")
    elements, routes, coordinates = geometry.resolve(mesh)
    if not embedding_certified:
        _require_shared_face_traces(mesh, geometry, connectivity, owner, neighbor)
    rules = tuple(
        _degree_aware_reference_rule(block.cell_kind, 3) for block in mesh.blocks
    )
    cell_q = max(int(points.shape[0]) for points, _ in rules)
    face_q = max(
        int(_degree_aware_reference_rule(kind, 3)[0].shape[0])
        for kind in ("triangle", "quadrilateral")
    )
    coordinates_host = np.asarray(coordinates)
    buffers = _GeometryBuffers.allocate(
        connectivity.cell_count,
        connectivity.face_count,
        cell_q,
        face_q,
        coordinates_host.dtype,
    )
    offsets = np.asarray(connectivity.face_vertex_offsets)
    vertices = np.asarray(connectivity.face_vertex_values)
    faces = {
        tuple(sorted(int(v) for v in vertices[offsets[i] : offsets[i + 1]])): i
        for i in range(connectivity.face_count)
    }
    cell_index = {
        int(value): i for i, value in enumerate(np.asarray(connectivity.cell_global_ids))
    }
    for block, element, route, rule in zip(
        mesh.blocks, elements, routes, rules, strict=True
    ):
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError("Mapped FV requires standard scalar coordinate elements.")
        indices = np.asarray(
            [cell_index[int(i)] for i in np.asarray(block.global_ids)], dtype=np.int32
        )
        _prepare_cells(block, element, route, coordinates_host, indices, rule, buffers)
        _prepare_block_faces(
            block, element, route, coordinates_host, indices, owner, faces, buffers
        )
    return _finish_geometry(buffers, owner, neighbor)
