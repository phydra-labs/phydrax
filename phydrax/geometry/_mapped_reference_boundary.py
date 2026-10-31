#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact source queries over mapped reference-root boundary charts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import overload, Protocol, runtime_checkable

import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization._cell_complex import (
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import ScalarCoordinateElement
from ..discretization._cell_mesh import CellMesh, SimplicialConnectivity
from ..discretization._coordinate_enclosure import (
    compose_expression,
    coordinate_map_expressions,
    evaluate_expression,
    ExactScalar,
    Polynomial,
    polynomial_bounds,
    RationalPolynomial,
    subtract,
)
from ..discretization._exact_algebraic_geometry import (
    coefficient_endpoint,
    RadicalElement,
)
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import reference_cell_topology
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..typing import ConvertibleToArray
from ._mapped_reference_domain import MappedReferenceDomain
from ._planar_primitives import area2 as _area2, intersection as _intersection
from ._rational import fraction_endpoint
from ._source_boundary_records import (
    SourceBoundaryChartCover,
    SourceBoundaryChartWitness,
    SourceBoundaryDistance,
)


type _ExactPoint3 = tuple[ExactScalar, ExactScalar, ExactScalar]
type _FloatPoint3 = tuple[float, float, float]
type _ChartBounds = tuple[tuple[float, float], ...]
type _ChartRecord = tuple[_ChartBounds, tuple[_ExactPoint3, ...]]
type _Point2 = tuple[Fraction, Fraction]
type _Triangle2 = tuple[_Point2, _Point2, _Point2]
type _ExactPoint2 = tuple[ExactScalar, ExactScalar]


def _exact_scalar(value: object, /) -> ExactScalar:
    if isinstance(value, (Fraction, RadicalElement)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if math.isfinite(scalar):
            return Fraction.from_float(scalar)
        raise ValueError("Mapped root reference points must be finite.")
    raise TypeError(
        "Mapped root reference points require exact scalars or authored floats."
    )


def _triangle2(points: tuple[tuple[Fraction, ...], ...], /) -> _Triangle2:
    if len(points) != 3 or any(len(point) != 2 for point in points):
        raise ValueError("Mapped root face chart parameters must be exact 2-D triangles.")
    return (
        (points[0][0], points[0][1]),
        (points[1][0], points[1][1]),
        (points[2][0], points[2][1]),
    )


@runtime_checkable
class MappedPlcRootRelation(Protocol):
    """Checked PLC-to-root entity relation retained by a mapped source authority."""

    vertex_root_global_ids: np.ndarray
    relation_id: str
    polygon_source_entity_ids: np.ndarray
    polygon_source_facet_rows: np.ndarray
    polygon_root_face_global_ids: np.ndarray
    polygon_root_cell_global_ids: np.ndarray
    polygon_root_local_face_indices: np.ndarray
    polygon_root_reference_vertices: np.ndarray
    polygon_orientation_signs: np.ndarray
    polygon_source_entity_keys: tuple[str, ...]
    polygon_source_occurrence_paths: tuple[tuple[str, ...], ...]
    segment_root_dimensions: np.ndarray
    segment_root_entity_ids: np.ndarray
    segment_root_cell_global_ids: np.ndarray
    segment_root_local_entity_indices: np.ndarray
    segment_root_reference_vertices: np.ndarray
    segment_orientation_signs: np.ndarray
    segment_source_entity_ids: np.ndarray
    segment_source_entity_keys: tuple[str, ...]
    segment_source_occurrence_paths: tuple[tuple[str, ...], ...]


@runtime_checkable
class _MappedPlcComplexInventory(Protocol):
    """Actual authored PLC vertex inventory retained by the native source."""

    vertex_entity_ids: tuple[int, ...]
    vertex_occurrence_paths: tuple[tuple[str, ...], ...]


@runtime_checkable
class MappedPlcRootAuthority(Protocol):
    """Current source owner that can reconstruct its checked root relation."""

    domain: MappedReferenceDomain
    complex: _MappedPlcComplexInventory
    source_id: str
    source_revision: str
    root_relation: MappedPlcRootRelation
    binding_id: str

    def validate_restored(self, /) -> None: ...
    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]: ...
    def scope_entity_ids(self, dimension: int, /) -> np.ndarray: ...


def _mapped_root_entity_banks(
    mesh: CellMesh, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return face vertices, reference-ordered cell faces, and edge vertices."""
    connectivity = mesh.connectivity
    if isinstance(connectivity, (HexahedralConnectivity, TetrahedralConnectivity)):
        return (
            np.asarray(connectivity.faces, dtype=np.int64),
            np.asarray(connectivity.cell_faces, dtype=np.int64),
            np.asarray(connectivity.edges, dtype=np.int64),
        )
    if isinstance(connectivity, SimplicialConnectivity):
        if connectivity.dimension != 3:
            raise TypeError("Mapped root simplex connectivity must have dimension three.")
        faces = np.asarray(connectivity.entities[2], dtype=np.int64)
        cells = np.asarray(connectivity.cells, dtype=np.int64)
        edges = np.asarray(connectivity.entities[1], dtype=np.int64)
        topology = reference_cell_topology("simplex:3")
    elif isinstance(connectivity, PolyhedralConnectivity):
        # Multi-block homogeneous roots use packed incidence; their faces and
        # cells are re-read in the blocks' reference vertex order.
        kinds = {
            "tetrahedron" if block.cell_kind == "simplex:3" else block.cell_kind
            for block in mesh.blocks
        }
        if len(kinds) != 1 or next(iter(kinds)) not in {"hexahedron", "tetrahedron"}:
            raise TypeError("Mapped root polyhedral connectivity must be homogeneous.")
        topology = reference_cell_topology(next(iter(kinds)))
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        arity = len(topology.entities[2][0])
        if np.any(np.diff(offsets) != arity):
            raise TypeError("Mapped root polyhedral faces must have one reference arity.")
        faces = values.reshape((-1, arity))
        rows_by_id = {
            int(identifier): vertices
            for block in mesh.blocks
            for identifier, vertices in zip(
                np.asarray(block.global_ids, dtype=np.int64).tolist(),
                np.asarray(block.vertices, dtype=np.int64).tolist(),
                strict=True,
            )
        }
        cells = np.asarray(
            [
                rows_by_id[int(identifier)]
                for identifier in np.asarray(
                    connectivity.cell_global_ids, dtype=np.int64
                ).tolist()
            ],
            dtype=np.int64,
        )
        edges = np.asarray(connectivity.edges, dtype=np.int64)
    else:
        raise TypeError(
            "Mapped root charts require hexahedral, tetrahedral, simplex, or "
            "homogeneous polyhedral connectivity."
        )
    # Local face indices follow the reference cell (simplex face i opposite
    # vertex i), as every mapped-root chart producer names them.
    face_rows = {
        tuple(sorted(int(value) for value in face)): row
        for row, face in enumerate(faces.tolist())
    }
    local_faces = topology.entities[2]
    cell_faces = np.asarray(
        [
            [
                face_rows[tuple(sorted(int(cell[local]) for local in face))]
                for face in local_faces
            ]
            for cell in cells.tolist()
        ],
        dtype=np.int64,
    ).reshape((-1, len(local_faces)))
    return faces, cell_faces, edges


def _affine_chart_arguments(
    vertices: tuple[tuple[Fraction, ...], ...], dimension: int, /
) -> tuple[Polynomial, ...]:
    if dimension not in (1, 2) or len(vertices) != dimension + 1:
        raise ValueError(
            "Mapped root chart arguments require exact edge or triangle corners."
        )
    arguments: list[Polynomial] = []
    for axis in range(3):
        polynomial: Polynomial = {}
        polynomial[(0,) * dimension] = vertices[0][axis]
        for parameter_axis in range(dimension):
            exponent = tuple(
                1 if coordinate == parameter_axis else 0
                for coordinate in range(dimension)
            )
            polynomial[exponent] = vertices[parameter_axis + 1][axis] - vertices[0][axis]
        arguments.append(polynomial)
    return tuple(arguments)


def _mapped_boundary_query_identity(
    authority_id: str,
    domain: MappedReferenceDomain,
    relation_id: str,
    root_vertex_global_ids: ArrayLike,
    face_arrays: tuple[np.ndarray, ...],
    face_keys: tuple[tuple[str, tuple[str, ...]], ...],
    edge_arrays: tuple[np.ndarray, ...],
    edge_keys: tuple[tuple[str, tuple[str, ...]], ...],
    face_scope_keys: tuple[tuple[str, tuple[str, ...]], ...],
    edge_scope_keys: tuple[tuple[str, tuple[str, ...]], ...],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "authority": authority_id,
            "kind": "mapped-reference-boundary-source-query",
            "domain": domain.domain_id,
            "spatial_id": domain.coordinate_contract.spatial_id,
            "reference_geometry": domain.reference_geometry_id,
            "relation": relation_id,
            "root_vertex_bindings": array_tree_fingerprint(
                np.asarray(root_vertex_global_ids, dtype=np.int64)
            ),
            "faces": array_tree_fingerprint(face_arrays),
            "face_keys": face_keys,
            "edges": array_tree_fingerprint(edge_arrays),
            "edge_keys": edge_keys,
            "scopes": (face_scope_keys, edge_scope_keys),
        }
    )


class MappedReferenceBoundarySource(StrictModule, NonTrainableState):
    """Original mapped-root boundary query with explicit facet/edge ancestry."""

    authority: MappedPlcRootAuthority
    authority_id: str
    domain: MappedReferenceDomain
    relation_id: str
    polygon_source_facet_rows: np.ndarray
    polygon_source_entity_ids: np.ndarray
    polygon_root_face_global_ids: np.ndarray
    polygon_root_cell_global_ids: np.ndarray
    polygon_root_local_face_indices: np.ndarray
    polygon_root_reference_vertices: np.ndarray
    polygon_orientation_signs: np.ndarray
    polygon_source_entity_keys: tuple[str, ...]
    polygon_source_occurrence_paths: tuple[tuple[str, ...], ...]
    segment_root_dimensions: np.ndarray
    segment_root_entity_global_ids: np.ndarray
    segment_root_cell_global_ids: np.ndarray
    segment_root_local_entity_indices: np.ndarray
    segment_root_reference_vertices: np.ndarray
    segment_orientations: np.ndarray
    segment_source_entity_ids: np.ndarray
    segment_source_entity_keys: tuple[str, ...]
    segment_source_occurrence_paths: tuple[tuple[str, ...], ...]
    face_chart_ids: tuple[str, ...]
    edge_chart_ids: tuple[str, ...]
    face_scope_keys: tuple[tuple[str, tuple[str, ...]], ...]
    edge_scope_keys: tuple[tuple[str, tuple[str, ...]], ...]
    _face_selected_rows: tuple[int, ...]
    _edge_selected_rows: tuple[int, ...]
    source_query_id: str

    def __init__(
        self,
        authority: MappedPlcRootAuthority,
        /,
        *,
        face_scope_keys: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
        edge_scope_keys: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    ) -> None:
        if not isinstance(authority, MappedPlcRootAuthority):
            raise TypeError(
                "authority must retain the native PLC root relation and source binding."
            )
        domain = authority.domain
        relation = authority.root_relation
        if not isinstance(domain, MappedReferenceDomain):
            raise TypeError("authority.domain must be MappedReferenceDomain.")
        if not isinstance(relation, MappedPlcRootRelation):
            raise TypeError(
                "authority.root_relation must retain the canonical mapped PLC/root relation."
            )
        complex_ = authority.complex
        if not isinstance(complex_, _MappedPlcComplexInventory):
            raise TypeError("Mapped source authority must retain its direct PLC complex.")
        source_vertex_ids = np.asarray(complex_.vertex_entity_ids)
        source_vertex_paths = tuple(
            tuple(path) for path in complex_.vertex_occurrence_paths
        )
        root_vertex_ids = np.asarray(relation.vertex_root_global_ids)
        if (
            source_vertex_ids.ndim != 1
            or source_vertex_ids.dtype.kind not in "iu"
            or source_vertex_ids.size != len(source_vertex_paths)
            or root_vertex_ids.shape != source_vertex_ids.shape
            or root_vertex_ids.dtype.kind not in "iu"
            or (
                source_vertex_ids.dtype.kind == "u"
                and np.any(source_vertex_ids > np.iinfo(np.int64).max)
            )
            or (
                root_vertex_ids.dtype.kind == "u"
                and np.any(root_vertex_ids > np.iinfo(np.int64).max)
            )
            or np.any(source_vertex_ids < 0)
            or np.any(root_vertex_ids < 0)
            or any(
                not isinstance(path, tuple)
                or any(not isinstance(part, str) or not part for part in path)
                for path in source_vertex_paths
            )
        ):
            raise ValueError(
                "Mapped PLC vertex keys and root global IDs are malformed or misaligned."
            )
        root_mesh_vertex_ids = np.asarray(
            domain.reference_mesh.vertex_global_ids, dtype=np.int64
        )
        if not np.all(np.isin(root_vertex_ids.astype(np.int64), root_mesh_vertex_ids)):
            raise ValueError(
                "Mapped PLC vertex ancestry names an absent root mesh vertex."
            )
        if (authority.source_id, authority.source_revision) != (
            domain.source_id,
            domain.source_revision,
        ):
            raise ValueError(
                "Mapped source authority and root domain have different source identities."
            )
        authority_id = canonical_identifier(
            authority.binding_id, "mapped_source_binding_id"
        )
        relation_id = canonical_identifier(relation.relation_id, "relation_id")

        polygon_entity_ids = np.asarray(relation.polygon_source_entity_ids)
        polygon_facets = np.asarray(relation.polygon_source_facet_rows)
        polygon_faces = np.asarray(relation.polygon_root_face_global_ids)
        polygon_cells_all = np.asarray(relation.polygon_root_cell_global_ids)
        polygon_local_all = np.asarray(relation.polygon_root_local_face_indices)
        polygon_refs_all = np.asarray(
            relation.polygon_root_reference_vertices, dtype=np.float64
        )
        polygon_orientations_all = np.asarray(relation.polygon_orientation_signs)
        polygon_keys = tuple(
            canonical_identifier(value, "source_entity_id")
            for value in relation.polygon_source_entity_keys
        )
        polygon_paths = tuple(
            tuple(canonical_identifier(part, "occurrence_path") for part in path)
            for path in relation.polygon_source_occurrence_paths
        )
        polygon_count = polygon_facets.size
        if (
            polygon_facets.ndim != 1
            or polygon_count == 0
            or polygon_entity_ids.shape != (polygon_count,)
            or polygon_entity_ids.dtype.kind not in "iu"
            or polygon_faces.shape != (polygon_count,)
            or polygon_cells_all.shape != (polygon_count, 2)
            or polygon_local_all.shape != (polygon_count, 2)
            or polygon_refs_all.shape != (polygon_count, 2, 3, 3)
            or polygon_orientations_all.shape != (polygon_count, 2)
            or len(polygon_keys) != polygon_count
            or len(polygon_paths) != polygon_count
            or polygon_facets.dtype.kind not in "iu"
            or polygon_faces.dtype.kind not in "iu"
            or polygon_cells_all.dtype.kind not in "iu"
            or polygon_local_all.dtype.kind not in "iu"
            or polygon_orientations_all.dtype.kind not in "iu"
            or np.any(~np.isfinite(polygon_refs_all))
        ):
            raise ValueError("Mapped PLC polygon/root-face relation banks are malformed.")
        if polygon_entity_ids.dtype.kind == "u" and np.any(
            polygon_entity_ids > np.iinfo(np.int64).max
        ):
            raise ValueError("Mapped PLC source facet identities must fit int64.")
        polygon_entity_ids = np.asarray(polygon_entity_ids, dtype=np.int64)
        if np.any(polygon_entity_ids < 0):
            raise ValueError("Mapped PLC source facet identities must be nonnegative.")
        polygon_facets = np.asarray(polygon_facets, dtype=np.int64)
        polygon_faces = np.asarray(polygon_faces, dtype=np.int64)
        polygon_cells_all = np.asarray(polygon_cells_all, dtype=np.int64)
        polygon_local_all = np.asarray(polygon_local_all, dtype=np.int64)
        polygon_orientations_all = np.asarray(polygon_orientations_all, dtype=np.int8)
        selected_sides = np.empty((polygon_count,), dtype=np.int8)
        side_by_face: dict[int, int] = {}
        for row, face_id in enumerate(polygon_faces.tolist()):
            side = side_by_face.get(int(face_id))
            if side is None:
                side = 0 if int(polygon_cells_all[row, 0]) >= 0 else 1
                side_by_face[int(face_id)] = side
            if (
                int(polygon_cells_all[row, side]) < 0
                or int(polygon_local_all[row, side]) < 0
                or int(polygon_orientations_all[row, side]) == 0
            ):
                raise ValueError(
                    "Mapped PLC triangles disagree on their selected root-face side."
                )
            selected_sides[row] = side
        rows = np.arange(polygon_count, dtype=np.int64)
        polygon_cells = polygon_cells_all[rows, selected_sides]
        polygon_local = polygon_local_all[rows, selected_sides]
        polygon_refs = polygon_refs_all[rows, selected_sides]
        polygon_orientations = polygon_orientations_all[rows, selected_sides]
        if (
            np.any(polygon_cells < 0)
            or np.any(polygon_local < 0)
            or np.any(~np.isin(polygon_orientations, (-1, 1)))
            or np.any(polygon_faces < 0)
            or np.any(polygon_facets < 0)
        ):
            raise ValueError(
                "Mapped PLC polygon relation omits a declared root face side."
            )

        segment_dims_all = np.asarray(relation.segment_root_dimensions)
        segment_ids_all = np.asarray(relation.segment_root_entity_ids)
        segment_cells_all = np.asarray(relation.segment_root_cell_global_ids)
        segment_local_all = np.asarray(relation.segment_root_local_entity_indices)
        segment_refs_all = np.asarray(
            relation.segment_root_reference_vertices, dtype=np.float64
        )
        segment_orientations_all = np.asarray(relation.segment_orientation_signs)
        segment_source_entity_ids_all = np.asarray(relation.segment_source_entity_ids)
        segment_keys_all = tuple(
            canonical_identifier(value, "source_entity_id")
            for value in relation.segment_source_entity_keys
        )
        segment_paths_all = tuple(
            tuple(canonical_identifier(part, "occurrence_path") for part in path)
            for path in relation.segment_source_occurrence_paths
        )
        segment_count = segment_dims_all.size
        if (
            segment_dims_all.ndim != 1
            or segment_ids_all.shape != (segment_count,)
            or segment_cells_all.shape != (segment_count,)
            or segment_local_all.shape != (segment_count,)
            or segment_refs_all.shape != (segment_count, 2, 3)
            or segment_orientations_all.shape != (segment_count,)
            or len(segment_keys_all) != segment_count
            or len(segment_paths_all) != segment_count
            or segment_dims_all.dtype.kind not in "iu"
            or segment_ids_all.dtype.kind not in "iu"
            or segment_cells_all.dtype.kind not in "iu"
            or segment_local_all.dtype.kind not in "iu"
            or segment_orientations_all.dtype.kind not in "iu"
            or np.any(~np.isfinite(segment_refs_all))
        ):
            raise ValueError(
                "Mapped PLC segment/root-entity relation banks are malformed."
            )
        segment_dims_all = np.asarray(segment_dims_all, dtype=np.int8)
        if (
            segment_source_entity_ids_all.shape != (segment_count,)
            or segment_source_entity_ids_all.dtype.kind not in "iu"
        ):
            raise ValueError("Mapped PLC segment source identities are malformed.")
        if segment_source_entity_ids_all.dtype.kind == "u" and np.any(
            segment_source_entity_ids_all > np.iinfo(np.int64).max
        ):
            raise ValueError("Mapped PLC source edge identities must fit int64.")
        segment_source_entity_ids_all = np.asarray(
            segment_source_entity_ids_all, dtype=np.int64
        )
        if np.any(segment_source_entity_ids_all < 0):
            raise ValueError("Mapped PLC source edge identities must be nonnegative.")
        segment_ids_all = np.asarray(segment_ids_all, dtype=np.int64)
        segment_cells_all = np.asarray(segment_cells_all, dtype=np.int64)
        segment_local_all = np.asarray(segment_local_all, dtype=np.int32)
        segment_orientations_all = np.asarray(segment_orientations_all, dtype=np.int8)
        supported_segment_rows = np.flatnonzero(
            np.isin(segment_dims_all, (1, 2, 3)) & (segment_cells_all >= 0)
        )
        segment_dims = segment_dims_all[supported_segment_rows]
        segment_ids = segment_ids_all[supported_segment_rows]
        segment_cells = segment_cells_all[supported_segment_rows]
        segment_local = segment_local_all[supported_segment_rows]
        segment_refs = segment_refs_all[supported_segment_rows]
        segment_orientations = segment_orientations_all[supported_segment_rows]
        segment_source_entity_ids = segment_source_entity_ids_all[supported_segment_rows]
        segment_keys = tuple(segment_keys_all[int(row)] for row in supported_segment_rows)
        segment_paths = tuple(
            segment_paths_all[int(row)] for row in supported_segment_rows
        )

        face_keys = tuple(zip(polygon_keys, polygon_paths, strict=True))
        edge_keys = tuple(zip(segment_keys, segment_paths, strict=True))
        face_scope = _select_scope_keys(face_scope_keys, face_keys, "face")
        edge_scope = _select_scope_keys(edge_scope_keys, edge_keys, "edge")
        face_rows = np.asarray(
            [row for row, key in enumerate(face_keys) if key in set(face_scope)],
            dtype=np.int64,
        )
        edge_rows = np.asarray(
            [row for row, key in enumerate(edge_keys) if key in set(edge_scope)],
            dtype=np.int64,
        )
        if face_rows.size == 0 and edge_rows.size == 0:
            raise ValueError(
                "Mapped reference boundary query scope must contain a chart."
            )

        polygon_source_facet_rows = _readonly(polygon_facets)
        polygon_source_entity_ids = _readonly(polygon_entity_ids)
        polygon_root_face_global_ids = _readonly(polygon_faces)
        polygon_root_cell_global_ids = _readonly(polygon_cells)
        polygon_root_local_face_indices = _readonly(polygon_local)
        polygon_root_reference_vertices = _readonly(polygon_refs)
        polygon_orientation_signs = _readonly(polygon_orientations)
        segment_root_dimensions = _readonly(segment_dims)
        segment_root_entity_global_ids = _readonly(segment_ids)
        segment_root_cell_global_ids = _readonly(segment_cells)
        segment_root_local_entity_indices = _readonly(segment_local)
        segment_root_reference_vertices = _readonly(segment_refs)
        segment_source_entity_ids = _readonly(segment_source_entity_ids)
        segment_orientations = _readonly(segment_orientations)
        face_chart_ids = tuple(
            _chart_id(
                "mapped-root-face-triangle",
                domain,
                relation_id,
                int(polygon_faces[row]),
                int(polygon_cells[row]),
                int(polygon_local[row]),
                int(polygon_entity_ids[row]),
                polygon_keys[row],
                polygon_paths[row],
                polygon_refs[row],
                int(polygon_facets[row]),
                int(polygon_orientations[row]),
            )
            for row in range(polygon_count)
        )
        edge_chart_ids = tuple(
            _chart_id(
                "mapped-root-edge-segment",
                domain,
                relation_id,
                int(segment_dims[row]),
                int(segment_ids[row]),
                int(segment_cells[row]),
                int(segment_local[row]),
                int(segment_source_entity_ids[row]),
                segment_keys[row],
                segment_paths[row],
                segment_refs[row],
                int(supported_segment_rows[row]),
                int(segment_orientations[row]),
            )
            for row in range(segment_dims.size)
        )
        face_selected_rows = tuple(int(face_rows[row]) for row in range(face_rows.size))
        edge_selected_rows = tuple(int(edge_rows[row]) for row in range(edge_rows.size))
        MappedReferenceBoundarySource._validate_root_relation(
            domain,
            polygon_root_face_global_ids,
            polygon_root_cell_global_ids,
            polygon_root_local_face_indices,
            polygon_root_reference_vertices,
            polygon_orientation_signs,
            segment_root_dimensions,
            segment_root_entity_global_ids,
            segment_root_cell_global_ids,
            segment_root_local_entity_indices,
            segment_root_reference_vertices,
            segment_orientations,
        )
        face_arrays = (
            polygon_source_facet_rows,
            polygon_source_entity_ids,
            polygon_root_face_global_ids,
            polygon_root_cell_global_ids,
            polygon_root_local_face_indices,
            polygon_root_reference_vertices,
            polygon_orientation_signs,
        )
        edge_arrays = (
            segment_root_dimensions,
            segment_source_entity_ids,
            segment_root_entity_global_ids,
            segment_root_cell_global_ids,
            segment_root_local_entity_indices,
            segment_root_reference_vertices,
            segment_orientations,
        )
        source_query_id = _mapped_boundary_query_identity(
            authority_id,
            domain,
            relation_id,
            root_vertex_ids,
            face_arrays,
            face_keys,
            edge_arrays,
            edge_keys,
            face_scope,
            edge_scope,
        )
        self.authority = authority
        self.authority_id = authority_id
        self.domain = domain
        self.relation_id = relation_id
        self.polygon_source_facet_rows = polygon_source_facet_rows
        self.polygon_source_entity_ids = polygon_source_entity_ids
        self.polygon_root_face_global_ids = polygon_root_face_global_ids
        self.polygon_root_cell_global_ids = polygon_root_cell_global_ids
        self.polygon_root_local_face_indices = polygon_root_local_face_indices
        self.polygon_root_reference_vertices = polygon_root_reference_vertices
        self.polygon_orientation_signs = polygon_orientation_signs
        self.polygon_source_entity_keys = polygon_keys
        self.polygon_source_occurrence_paths = polygon_paths
        self.segment_root_dimensions = segment_root_dimensions
        self.segment_root_entity_global_ids = segment_root_entity_global_ids
        self.segment_root_cell_global_ids = segment_root_cell_global_ids
        self.segment_root_local_entity_indices = segment_root_local_entity_indices
        self.segment_root_reference_vertices = segment_root_reference_vertices
        self.segment_source_entity_ids = segment_source_entity_ids
        self.segment_orientations = segment_orientations
        self.segment_source_entity_keys = segment_keys
        self.segment_source_occurrence_paths = segment_paths
        self.face_scope_keys = face_scope
        self.edge_scope_keys = edge_scope
        self.face_chart_ids = face_chart_ids
        self.edge_chart_ids = edge_chart_ids
        self._face_selected_rows = face_selected_rows
        self._edge_selected_rows = edge_selected_rows
        self.source_query_id = source_query_id

    @property
    def coordinate_contract(self) -> SpatialCoordinateContract:
        """Return the exact physical coordinate contract retained by the domain."""
        return self.domain.coordinate_contract

    @property
    def source_id(self) -> str:
        return self.domain.source_id

    @property
    def source_revision(self) -> str:
        return self.domain.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.domain.ambient_dimension

    @property
    def query_id(self) -> str:
        return self.source_query_id

    @property
    def is_full_surface_scope(self) -> bool:
        """Whether selected source charts exhaust every declared root face chart."""
        return len(self._face_selected_rows) == len(self.face_chart_ids)

    def selected_source_keys(
        self, chart_dimension: int, /
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        if chart_dimension == 2:
            rows = self._face_selected_rows
            keys = zip(
                self.polygon_source_entity_keys,
                self.polygon_source_occurrence_paths,
                strict=True,
            )
        elif chart_dimension == 1:
            rows = self._edge_selected_rows
            keys = zip(
                self.segment_source_entity_keys,
                self.segment_source_occurrence_paths,
                strict=True,
            )
        else:
            raise ValueError(
                "Mapped root source keys support only curve and surface chart dimensions."
            )
        bank = tuple(keys)
        return tuple(sorted({bank[row] for row in rows}))

    def _identity(self) -> str:
        return _mapped_boundary_query_identity(
            self.authority_id,
            self.domain,
            self.relation_id,
            self.authority.root_relation.vertex_root_global_ids,
            (
                self.polygon_source_facet_rows,
                self.polygon_source_entity_ids,
                self.polygon_root_face_global_ids,
                self.polygon_root_cell_global_ids,
                self.polygon_root_local_face_indices,
                self.polygon_root_reference_vertices,
                self.polygon_orientation_signs,
            ),
            tuple(
                zip(
                    self.polygon_source_entity_keys,
                    self.polygon_source_occurrence_paths,
                    strict=True,
                )
            ),
            (
                self.segment_root_dimensions,
                self.segment_source_entity_ids,
                self.segment_root_entity_global_ids,
                self.segment_root_cell_global_ids,
                self.segment_root_local_entity_indices,
                self.segment_root_reference_vertices,
                self.segment_orientations,
            ),
            tuple(
                zip(
                    self.segment_source_entity_keys,
                    self.segment_source_occurrence_paths,
                    strict=True,
                )
            ),
            self.face_scope_keys,
            self.edge_scope_keys,
        )

    def _chart_expressions(
        self, chart_dimension: int, row: int, /
    ) -> tuple[Polynomial | RationalPolynomial, ...]:
        mesh = self.domain.reference_mesh
        cell_id = (
            int(self.polygon_root_cell_global_ids[row])
            if chart_dimension == 2
            else int(self.segment_root_cell_global_ids[row])
        )
        reference_vertices = (
            np.asarray(self.polygon_root_reference_vertices[row], dtype=np.float64)
            if chart_dimension == 2
            else np.asarray(self.segment_root_reference_vertices[row], dtype=np.float64)
        )
        cell_rows = {
            int(identifier): (block_index, local_row)
            for block_index, block in enumerate(mesh.blocks)
            for local_row, identifier in enumerate(
                np.asarray(block.global_ids, dtype=np.int64)
            )
        }
        found = cell_rows.get(cell_id)
        if found is None:
            raise ValueError("Mapped root chart names an absent source cell.")
        block_index, cell_row = found
        elements, routes, _ = self.domain.source_geometry.resolve(mesh)
        coefficients = self.domain.source_geometry.source_coordinates()
        local = tuple(
            coefficients[int(index)]
            for index in np.asarray(routes[block_index][cell_row], dtype=np.int64)
        )
        element = elements[block_index]
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError(
                "Mapped root chart geometry requires a scalar coordinate element."
            )
        physical = coordinate_map_expressions(element, local, cell_row=cell_row)
        exact_vertices = tuple(
            tuple(Fraction.from_float(float(value)) for value in point)
            for point in reference_vertices
        )
        arguments = _affine_chart_arguments(exact_vertices, chart_dimension)
        return tuple(compose_expression(value, arguments) for value in physical)

    def _selected_rows(self, chart_dimension: int, /) -> tuple[int, ...]:
        if chart_dimension == 2:
            return self._face_selected_rows
        if chart_dimension == 1:
            return self._edge_selected_rows
        raise ValueError("Mapped root charts support only curves and surfaces.")

    def chart_row(self, chart_dimension: int, chart_id: str, /) -> int:
        if chart_dimension == 2:
            ids = self.face_chart_ids
        elif chart_dimension == 1:
            ids = self.edge_chart_ids
        else:
            raise ValueError("Mapped root charts support only curves and surfaces.")
        wanted = canonical_identifier(chart_id, "source_chart_id")
        try:
            row = ids.index(wanted)
        except ValueError as error:
            raise ValueError(
                "Mapped source chart is absent from the root relation."
            ) from error
        if row not in self._selected_rows(chart_dimension):
            raise ValueError("Mapped source chart is outside the declared query scope.")
        return row

    def chart_source_key(
        self, chart_dimension: int, chart_id: str, /
    ) -> tuple[str, tuple[str, ...]]:
        row = self.chart_row(chart_dimension, chart_id)
        if chart_dimension == 2:
            return (
                self.polygon_source_entity_keys[row],
                self.polygon_source_occurrence_paths[row],
            )
        return self.segment_source_entity_keys[row], self.segment_source_occurrence_paths[
            row
        ]

    @overload
    def root_face_uv(
        self,
        root_face_global_id: int,
        root_cell_global_id: int,
        local_face_index: int,
        root_reference_points: Sequence[Sequence[Fraction]],
        /,
    ) -> tuple[_Point2, ...]: ...

    @overload
    def root_face_uv(
        self,
        root_face_global_id: int,
        root_cell_global_id: int,
        local_face_index: int,
        root_reference_points: ConvertibleToArray | Sequence[Sequence[ExactScalar]],
        /,
    ) -> tuple[_ExactPoint2, ...]: ...

    def root_face_uv(
        self,
        root_face_global_id: int,
        root_cell_global_id: int,
        local_face_index: int,
        root_reference_points: ConvertibleToArray | Sequence[Sequence[ExactScalar]],
        /,
    ) -> tuple[_ExactPoint2, ...]:
        return self._root_face_uv_for_domain(
            self.domain,
            root_face_global_id,
            root_cell_global_id,
            local_face_index,
            root_reference_points,
        )

    @staticmethod
    def _root_face_uv_for_domain(
        domain: MappedReferenceDomain,
        root_face_global_id: int,
        root_cell_global_id: int,
        local_face_index: int,
        root_reference_points: ConvertibleToArray | Sequence[Sequence[ExactScalar]],
        /,
    ) -> tuple[_ExactPoint2, ...]:
        from ..linalg._exact import factor_exact_rational_algebraic

        mesh = domain.reference_mesh
        try:
            face_vertex_rows, face_rows, _ = _mapped_root_entity_banks(mesh)
        except TypeError as error:
            raise TypeError(
                "Mapped root face chart has unsupported connectivity."
            ) from error
        face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
        cell_connectivity_rows = {
            int(identifier): row
            for row, identifier in enumerate(
                np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
            )
        }
        face_row_lookup = {
            int(identifier): row for row, identifier in enumerate(face_ids)
        }
        wanted_face = int(root_face_global_id)
        face_row = face_row_lookup.get(wanted_face)
        if face_row is None:
            raise ValueError(
                "root_face_global_id is absent from the mapped reference mesh."
            )
        cell_lookup = {
            int(identifier): (block_index, local_row)
            for block_index, block in enumerate(mesh.blocks)
            for local_row, identifier in enumerate(
                np.asarray(block.global_ids, dtype=np.int64)
            )
        }
        owner = cell_lookup.get(int(root_cell_global_id))
        if owner is None:
            raise ValueError(
                "root_cell_global_id is absent from the mapped reference mesh."
            )
        block_index, cell_row = owner
        connectivity_row = cell_connectivity_rows.get(int(root_cell_global_id))
        if connectivity_row is None:
            raise ValueError("Root cell is absent from canonical cell connectivity.")
        local_faces = face_rows[connectivity_row]
        local_face = int(local_face_index)
        if (
            local_face < 0
            or local_face >= local_faces.size
            or int(face_ids[int(local_faces[local_face])]) != wanted_face
        ):
            raise ValueError(
                "Root face ID does not match its declared root cell/local-face incidence."
            )
        topology = reference_cell_topology(mesh.blocks[block_index].cell_kind)
        face_loop = np.asarray(face_vertex_rows[face_row], dtype=np.int64)
        cell_vertices = np.asarray(mesh.blocks[block_index].vertices, dtype=np.int64)[
            cell_row
        ]
        reference_vertices = tuple(
            tuple(Fraction.from_float(float(value)) for value in point)
            for point in np.asarray(topology.vertices, dtype=np.float64)
        )
        reference_by_mesh_row = {
            int(vertex_row): reference_vertices[local_vertex]
            for local_vertex, vertex_row in enumerate(cell_vertices)
        }
        try:
            face_reference_loop = tuple(
                reference_by_mesh_row[int(vertex_row)] for vertex_row in face_loop
            )
        except KeyError as error:
            raise ValueError(
                "Canonical root face loop is not incident to its owner cell."
            ) from error
        if len(face_reference_loop) == 4:
            origin = face_reference_loop[0]
            first = tuple(
                face_reference_loop[1][axis] - origin[axis] for axis in range(3)
            )
            second = tuple(
                face_reference_loop[3][axis] - origin[axis] for axis in range(3)
            )
            if any(
                face_reference_loop[2][axis] != origin[axis] + first[axis] + second[axis]
                for axis in range(3)
            ):
                raise ValueError(
                    "Root hexahedral face loop is not an exact affine chart."
                )
        elif len(face_reference_loop) == 3:
            origin = face_reference_loop[0]
            first = tuple(
                face_reference_loop[1][axis] - origin[axis] for axis in range(3)
            )
            second = tuple(
                face_reference_loop[2][axis] - origin[axis] for axis in range(3)
            )
        else:
            raise ValueError("Mapped root face loop must be a triangle or quadrilateral.")
        coordinate_pair = next(
            (
                (left, right)
                for left in range(3)
                for right in range(left + 1, 3)
                if first[left] * second[right] - first[right] * second[left] != 0
            ),
            None,
        )
        if coordinate_pair is None:
            raise ValueError("Mapped root face chart has a singular affine basis.")
        left, right = coordinate_pair
        matrix = ((first[left], second[left]), (first[right], second[right]))
        factorization = factor_exact_rational_algebraic(matrix, RadicalElement)
        raw_points = np.asarray(root_reference_points, dtype=object)
        if raw_points.ndim != 2 or raw_points.shape[1] != 3:
            raise ValueError("root_reference_points must have shape (N, 3).")
        points = tuple(
            tuple(_exact_scalar(value) for value in raw_point) for raw_point in raw_points
        )
        if not points:
            return ()
        right_hand_side = (
            tuple(point[left] - origin[left] for point in points),
            tuple(point[right] - origin[right] for point in points),
        )
        solved = factorization.solve(right_hand_side)
        output: list[_ExactPoint2] = []
        for index, point in enumerate(points):
            chart_point = (solved[0][index], solved[1][index])
            if any(
                origin[axis]
                + chart_point[0] * first[axis]
                + chart_point[1] * second[axis]
                != point[axis]
                for axis in range(3)
            ):
                raise ValueError("Root reference point leaves the actual mapped face.")
            if len(face_reference_loop) == 4:
                inside = 0 <= chart_point[0] <= 1 and 0 <= chart_point[1] <= 1
            else:
                inside = (
                    chart_point[0] >= 0
                    and chart_point[1] >= 0
                    and chart_point[0] + chart_point[1] <= 1
                )
            if not inside:
                raise ValueError("Root reference point lies outside its face-loop chart.")
            output.append(chart_point)
        return tuple(output)

    def chart_face_uv(self, chart_id: str, /) -> tuple[tuple[Fraction, Fraction], ...]:
        """Return the source PLC triangle in its root global-face-loop chart."""
        row = self.chart_row(2, chart_id)
        reference_vertices = tuple(
            tuple(Fraction.from_float(float(value)) for value in point)
            for point in np.asarray(
                self.polygon_root_reference_vertices[row], dtype=np.float64
            )
        )
        return self.root_face_uv(
            int(self.polygon_root_face_global_ids[row]),
            int(self.polygon_root_cell_global_ids[row]),
            int(self.polygon_root_local_face_indices[row]),
            reference_vertices,
        )

    def chart_coordinate_expressions(
        self, chart_dimension: int, chart_id: str, /
    ) -> tuple[Polynomial | RationalPolynomial, ...]:
        row = self.chart_row(chart_dimension, chart_id)
        return self._chart_expressions(chart_dimension, row)

    def validate_chart_witness(
        self, witness: SourceBoundaryChartWitness, /
    ) -> tuple[int, ...]:
        self.require_current()
        if (
            witness.source_id != self.source_id
            or witness.source_revision != self.source_revision
            or witness.source_query_id != self.source_query_id
        ):
            raise ValueError("Mapped root witness changes its source/query identity.")
        if witness.chart_dimension not in (1, 2):
            raise ValueError("Mapped root chart witnesses must be curves or surfaces.")
        selected = set(self._selected_rows(witness.chart_dimension))
        chart_ids = (
            self.face_chart_ids if witness.chart_dimension == 2 else self.edge_chart_ids
        )
        expected_chart_ids = {chart_ids[row] for row in selected}
        witness_chart_ids = set(witness.source_chart_ids)
        if witness_chart_ids != expected_chart_ids:
            raise ValueError(
                "Mapped root chart witness omits or adds a source chart key."
            )
        chart_rows: list[int] = []
        surface_groups: dict[int, list[_Triangle2]] = {}
        curve_groups: dict[int, list[tuple[Fraction, Fraction]]] = {}
        for row, chart_id in enumerate(witness.source_chart_ids):
            chart_row = self.chart_row(witness.chart_dimension, chart_id)
            if chart_row not in selected or self.chart_source_key(
                witness.chart_dimension, chart_id
            ) != (witness.source_entity_ids[row], witness.occurrence_paths[row]):
                raise ValueError(
                    "Mapped root chart witness changes an authored source key."
                )
            chart_rows.append(chart_row)
            parameters = witness.source_parameter_chart(row)
            if witness.chart_dimension == 2:
                surface_groups.setdefault(chart_row, []).append(_triangle2(parameters))
            else:
                if len(parameters) != 2 or any(len(point) != 1 for point in parameters):
                    raise ValueError(
                        "Mapped root curve parameters must form an exact interval."
                    )
                curve_groups.setdefault(chart_row, []).append(
                    (parameters[0][0], parameters[1][0])
                )
        if witness.chart_dimension == 2:
            from ._source_fidelity import _require_reference_chart_partition

            unit_triangle = (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
            for rows in surface_groups.values():
                _require_reference_chart_partition(tuple(rows), (unit_triangle,))
        else:
            for intervals in curve_groups.values():
                ordered = sorted(
                    (min(first, last), max(first, last)) for first, last in intervals
                )
                cursor = Fraction(0)
                for first, last in ordered:
                    if first != cursor or last <= first:
                        raise ValueError(
                            "Mapped root curve chart intervals have a gap or overlap."
                        )
                    cursor = last
                if cursor != Fraction(1):
                    raise ValueError(
                        "Mapped root curve chart omits part of its authored interval."
                    )
        return tuple(chart_rows)

    def boundary_chart_cover(self, /) -> SourceBoundaryChartCover:
        self.require_current()
        rows = self._selected_rows(2)
        if not rows:
            raise ValueError(
                "Mapped root query has no surface charts for a surface cover."
            )
        standard = (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
        simplices: list[tuple[_FloatPoint3, _FloatPoint3, _FloatPoint3]] = []
        bounds: list[float] = []
        for row in rows:
            expressions = self._chart_expressions(2, row)
            corner_rows: list[_ExactPoint3] = []
            for point in standard:
                image = tuple(
                    evaluate_expression(expression, point) for expression in expressions
                )
                if len(image) != 3:
                    raise ValueError(
                        "Mapped surface chart must have three physical coordinates."
                    )
                corner_rows.append((image[0], image[1], image[2]))
            corners = (corner_rows[0], corner_rows[1], corner_rows[2])
            simplex: tuple[_FloatPoint3, _FloatPoint3, _FloatPoint3] = (
                (
                    float(corners[0][0]),
                    float(corners[0][1]),
                    float(corners[0][2]),
                ),
                (
                    float(corners[1][0]),
                    float(corners[1][1]),
                    float(corners[1][2]),
                ),
                (
                    float(corners[2][0]),
                    float(corners[2][1]),
                    float(corners[2][2]),
                ),
            )
            residuals: list[Fraction] = []
            for axis, expression in enumerate(expressions):
                affine: Polynomial = {
                    (0, 0): Fraction.from_float(simplex[0][axis]),
                    (1, 0): Fraction.from_float(simplex[1][axis])
                    - Fraction.from_float(simplex[0][axis]),
                    (0, 1): Fraction.from_float(simplex[2][axis])
                    - Fraction.from_float(simplex[0][axis]),
                }
                residual = subtract(expression, affine)
                lower, upper = polynomial_bounds(residual, "simplex", 2)
                if not math.isfinite(lower) or not math.isfinite(upper):
                    raise ValueError(
                        "Mapped source chart has no finite continuous image enclosure."
                    )
                residuals.append(
                    max(abs(Fraction.from_float(lower)), abs(Fraction.from_float(upper)))
                )
            simplices.append(simplex)
            bounds.append(
                fraction_endpoint(
                    fraction_sqrt_interval(
                        sum((value * value for value in residuals), Fraction(0))
                    )[1],
                    upper=True,
                    saturate=True,
                )
            )
        return SourceBoundaryChartCover(
            np.asarray(simplices, dtype=np.float64),
            np.asarray(bounds, dtype=np.float64),
            "certified",
            True,
            self.source_id,
            self.source_revision,
            restriction_ids=(self.source_query_id,),
        )

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or np.any(~np.isfinite(values)):
            raise ValueError(
                "Mapped boundary distance points must be finite three-vectors."
            )
        self.require_current()
        chart_records: list[_ChartRecord] = []
        for dimension in (2, 1):
            for row in self._selected_rows(dimension):
                expressions = self._chart_expressions(dimension, row)
                domain_name = "simplex" if dimension == 2 else "box"
                bounds = tuple(
                    polynomial_bounds(expression, domain_name, dimension)
                    for expression in expressions
                )
                if any(
                    not math.isfinite(low) or not math.isfinite(high)
                    for low, high in bounds
                ):
                    raise ValueError(
                        "Mapped boundary chart has no finite continuous distance enclosure."
                    )
                corners = (
                    (
                        (Fraction(0), Fraction(0)),
                        (Fraction(1), Fraction(0)),
                        (Fraction(0), Fraction(1)),
                    )
                    if dimension == 2
                    else ((Fraction(0),), (Fraction(1),))
                )
                sample_rows: list[_ExactPoint3] = []
                for point in corners:
                    image = tuple(
                        evaluate_expression(expression, point)
                        for expression in expressions
                    )
                    if len(image) != 3:
                        raise ValueError(
                            "Mapped distance chart must have three physical coordinates."
                        )
                    sample_rows.append((image[0], image[1], image[2]))
                chart_records.append((bounds, tuple(sample_rows)))
        if not chart_records:
            raise ValueError("Mapped boundary distance query has no root chart rows.")
        lower_results = np.empty((values.shape[0],), dtype=np.float64)
        upper_results = np.empty_like(lower_results)
        for point_row in range(values.shape[0]):
            point: _ExactPoint3 = (
                Fraction.from_float(float(values[point_row, 0])),
                Fraction.from_float(float(values[point_row, 1])),
                Fraction.from_float(float(values[point_row, 2])),
            )
            chart_lower_bounds: list[Fraction] = []
            for bounds, _ in chart_records:
                gaps = tuple(
                    Fraction(0)
                    if low <= float(point[axis]) <= high
                    else Fraction.from_float(low) - point[axis]
                    if float(point[axis]) < low
                    else point[axis] - Fraction.from_float(high)
                    for axis, (low, high) in enumerate(bounds)
                )
                chart_lower_bounds.append(
                    fraction_sqrt_interval(
                        sum((max(gap, Fraction(0)) ** 2 for gap in gaps), Fraction(0))
                    )[0]
                )
            lower_results[point_row] = fraction_endpoint(
                min(chart_lower_bounds), upper=False, saturate=True
            )
            sample_upper_bounds: list[Fraction] = []
            for _, samples in chart_records:
                for sample in samples:
                    errors = []
                    for axis, exact_value in enumerate(sample):
                        low = Fraction(coefficient_endpoint(exact_value, upper=False))
                        high = Fraction(coefficient_endpoint(exact_value, upper=True))
                        errors.append(
                            max(abs(point[axis] - low), abs(point[axis] - high))
                        )
                    sample_upper_bounds.append(
                        fraction_sqrt_interval(
                            sum((error * error for error in errors), Fraction(0))
                        )[1]
                    )
            upper_results[point_row] = fraction_endpoint(
                min(sample_upper_bounds), upper=True, saturate=True
            )
        return SourceBoundaryDistance(lower_results, upper_results, "certified")

    def require_current(self, /) -> None:
        self.authority.validate_restored()
        renewed_domain = MappedReferenceDomain(
            self.domain.reference_domain,
            self.domain.reference_mesh,
            self.domain.source_geometry,
            np.asarray(self.domain.cell_regions, dtype=np.int64),
            source_id=self.domain.source_id,
            source_revision=self.domain.source_revision,
            coordinate_contract=self.coordinate_contract,
            root_vertex_occurrence_paths=self.domain.root_vertex_occurrence_paths,
            root_cell_occurrence_paths=self.domain.root_cell_occurrence_paths,
            reference_domain_vertex_root_ids=(
                None
                if self.domain.reference_domain_vertex_root_ids is None
                else np.asarray(
                    self.domain.reference_domain_vertex_root_ids, dtype=np.int64
                )
            ),
        )
        renewed = type(self)(
            self.authority,
            face_scope_keys=self.face_scope_keys,
            edge_scope_keys=self.edge_scope_keys,
        )
        if (
            renewed_domain.domain_id != self.domain.domain_id
            or renewed.source_query_id != self.source_query_id
            or self._identity() != self.source_query_id
        ):
            raise ValueError(
                "Mapped boundary query no longer matches its exact source/root relation."
            )

    @staticmethod
    def _validate_root_relation(
        domain: MappedReferenceDomain,
        polygon_root_face_global_ids: np.ndarray,
        polygon_root_cell_global_ids: np.ndarray,
        polygon_root_local_face_indices: np.ndarray,
        polygon_root_reference_vertices: np.ndarray,
        polygon_orientation_signs: np.ndarray,
        segment_root_dimensions: np.ndarray,
        segment_root_entity_global_ids: np.ndarray,
        segment_root_cell_global_ids: np.ndarray,
        segment_root_local_entity_indices: np.ndarray,
        segment_root_reference_vertices: np.ndarray,
        segment_orientations: np.ndarray,
        /,
    ) -> None:
        root_mesh = domain.reference_mesh
        cell_kinds = {
            "tetrahedron" if block.cell_kind == "simplex:3" else block.cell_kind
            for block in root_mesh.blocks
        }
        if (
            root_mesh.topological_dimension != 3
            or root_mesh.ambient_dimension != 3
            or len(cell_kinds) != 1
            or next(iter(cell_kinds), None) not in {"hexahedron", "tetrahedron"}
        ):
            raise ValueError(
                "Mapped root surface query requires homogeneous hexahedral or three-dimensional simplex charts."
            )
        try:
            face_vertex_rows, face_rows, edge_vertices = _mapped_root_entity_banks(
                root_mesh
            )
        except TypeError as error:
            raise ValueError(
                "Mapped root surface query requires homogeneous hexahedral or three-dimensional simplex charts."
            ) from error
        root_cell_kind = next(iter(cell_kinds))
        topology = reference_cell_topology(root_cell_kind)
        root_cell_rows = {
            int(identifier): (block_index, local_row)
            for block_index, block in enumerate(root_mesh.blocks)
            for local_row, identifier in enumerate(
                np.asarray(block.global_ids, dtype=np.int64)
            )
        }
        face_ids = np.asarray(root_mesh.entity_set(2).entity_ids, dtype=np.int64)
        face_row_by_id = {int(identifier): row for row, identifier in enumerate(face_ids)}
        cell_ids = np.asarray(root_mesh.entity_set(3).entity_ids, dtype=np.int64)
        cell_connectivity_rows = {
            int(identifier): row for row, identifier in enumerate(cell_ids)
        }
        cell_regions = np.asarray(domain.cell_regions, dtype=np.int64)
        if face_rows.shape[0] != cell_ids.size or cell_regions.shape != cell_ids.shape:
            raise ValueError(
                "Mapped root region and face incidence banks are misaligned."
            )
        face_region_sets: list[set[int]] = [set() for _ in range(face_ids.size)]
        for cell_row, local_faces in enumerate(face_rows):
            for face_row in local_faces:
                face_region_sets[int(face_row)].add(int(cell_regions[cell_row]))
        boundary = np.asarray(root_mesh.boundary_masks[2], dtype=np.bool_)
        expected_faces = {
            int(face_ids[row])
            for row in range(face_ids.size)
            if boundary[row] or len(face_region_sets[row]) > 1
        }
        if set(int(value) for value in polygon_root_face_global_ids) != expected_faces:
            raise ValueError(
                "Mapped root boundary chart rows do not cover every source boundary/interface face."
            )
        vertex_ids = np.asarray(root_mesh.vertex_global_ids, dtype=np.int64)
        if root_cell_kind == "hexahedron":
            face_chart_corners = (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(1), Fraction(1)),
                (Fraction(0), Fraction(1)),
            )
            face_domain = (
                (
                    (Fraction(0), Fraction(0)),
                    (Fraction(1), Fraction(0)),
                    (Fraction(1), Fraction(1)),
                ),
                (
                    (Fraction(0), Fraction(0)),
                    (Fraction(1), Fraction(1)),
                    (Fraction(0), Fraction(1)),
                ),
            )
        else:
            face_chart_corners = (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
            face_domain = (face_chart_corners,)
        row_charts: dict[int, list[tuple[tuple[Fraction, Fraction], ...]]] = {}
        row_sides: dict[int, tuple[int, int]] = {}
        for row, face_id in enumerate(polygon_root_face_global_ids.tolist()):
            cell_id = int(polygon_root_cell_global_ids[row])
            local_face = int(polygon_root_local_face_indices[row])
            cell_owner = root_cell_rows.get(cell_id)
            face_row = face_row_by_id.get(int(face_id))
            if cell_owner is None or face_row is None:
                raise ValueError(
                    "Mapped source chart references an absent root cell or face."
                )
            block_index, cell_row = cell_owner
            local_faces = face_rows[cell_connectivity_rows[cell_id]]
            if local_face < 0 or local_face >= local_faces.size:
                raise ValueError("Mapped source chart local root face index is invalid.")
            if int(face_ids[int(local_faces[local_face])]) != int(face_id):
                raise ValueError(
                    "Mapped source chart root face differs from its actual cell incidence."
                )
            side = (cell_id, local_face)
            previous_side = row_sides.setdefault(int(face_id), side)
            if previous_side != side:
                raise ValueError(
                    "Mapped root face triangles select inconsistent incident cell sides."
                )
            cell_vertices = np.asarray(
                root_mesh.blocks[block_index].vertices, dtype=np.int64
            )[cell_row]
            local_face_vertices = tuple(
                int(value) for value in topology.entities[2][local_face]
            )
            face_vertex_global_ids = tuple(
                int(vertex_ids[int(cell_vertices[local])])
                for local in local_face_vertices
            )
            root_face_vertex_rows = tuple(
                int(value) for value in face_vertex_rows[face_row]
            )
            canonical_face_ids = tuple(
                int(vertex_ids[value]) for value in root_face_vertex_rows
            )
            if set(face_vertex_global_ids) != set(canonical_face_ids):
                raise ValueError(
                    "Mapped root local face changes canonical global face vertices."
                )
            face_chart_by_vertex = {
                identifier: face_chart_corners[index]
                for index, identifier in enumerate(canonical_face_ids)
            }
            local_reference_vertices = tuple(
                tuple(Fraction.from_float(float(value)) for value in point)
                for point in np.asarray(topology.vertices, dtype=np.float64)
            )
            local_vertex_by_reference = {
                point: int(vertex_ids[int(cell_vertices[local])])
                for local, point in enumerate(local_reference_vertices)
            }
            chart_triangle: list[tuple[Fraction, Fraction]] = []
            for point in np.asarray(
                polygon_root_reference_vertices[row], dtype=np.float64
            ):
                exact_point = tuple(Fraction.from_float(float(value)) for value in point)
                vertex_id = local_vertex_by_reference.get(exact_point)
                if vertex_id is None or vertex_id not in face_chart_by_vertex:
                    raise ValueError(
                        "Mapped root source triangle corner is not a root face vertex."
                    )
                chart_triangle.append(face_chart_by_vertex[vertex_id])
            row_charts.setdefault(int(face_id), []).append(tuple(chart_triangle))
            face_reference = tuple(
                local_reference_vertices[int(index)]
                for index in topology.entities[2][local_face]
            )
            triangle_reference = tuple(
                tuple(Fraction.from_float(float(value)) for value in point)
                for point in np.asarray(polygon_root_reference_vertices[row])
            )
            face_first = tuple(
                face_reference[1][axis] - face_reference[0][axis] for axis in range(3)
            )
            face_second = tuple(
                face_reference[-1][axis] - face_reference[0][axis] for axis in range(3)
            )
            triangle_first = tuple(
                triangle_reference[1][axis] - triangle_reference[0][axis]
                for axis in range(3)
            )
            triangle_second = tuple(
                triangle_reference[2][axis] - triangle_reference[0][axis]
                for axis in range(3)
            )
            face_normal = (
                face_first[1] * face_second[2] - face_first[2] * face_second[1],
                face_first[2] * face_second[0] - face_first[0] * face_second[2],
                face_first[0] * face_second[1] - face_first[1] * face_second[0],
            )
            triangle_normal = (
                triangle_first[1] * triangle_second[2]
                - triangle_first[2] * triangle_second[1],
                triangle_first[2] * triangle_second[0]
                - triangle_first[0] * triangle_second[2],
                triangle_first[0] * triangle_second[1]
                - triangle_first[1] * triangle_second[0],
            )
            orientation_value = sum(
                (
                    left * right
                    for left, right in zip(face_normal, triangle_normal, strict=True)
                ),
                Fraction(0),
            )
            orientation = (
                1 if orientation_value > 0 else -1 if orientation_value < 0 else 0
            )
            if orientation != int(polygon_orientation_signs[row]):
                raise ValueError(
                    "Mapped PLC root-face orientation does not reproduce its exact chart order."
                )
        for face_id in expected_faces:
            triangles = row_charts.get(face_id, ())
            if not triangles:
                raise ValueError("Mapped root face has no declared PLC source triangles.")
            for triangle in triangles:
                if _area2(triangle) == 0:
                    raise ValueError(
                        "Mapped root-face chart contains a degenerate PLC triangle."
                    )
                covered = sum(
                    (
                        _area2(_intersection(triangle, domain_triangle))
                        for domain_triangle in face_domain
                    ),
                    Fraction(0),
                )
                if covered != _area2(triangle):
                    raise ValueError(
                        "Mapped PLC root-face triangle leaves its exact face chart."
                    )
            if any(
                _area2(_intersection(first, second)) > 0
                for index, first in enumerate(triangles)
                for second in triangles[index + 1 :]
            ):
                raise ValueError(
                    "Mapped PLC root-face triangles overlap with positive area."
                )
            if sum((_area2(triangle) for triangle in triangles), Fraction(0)) != sum(
                (_area2(domain_triangle) for domain_triangle in face_domain),
                Fraction(0),
            ):
                raise ValueError(
                    "Mapped PLC root-face triangles omit part of the exact face chart."
                )
        root_edge_ids = np.asarray(root_mesh.entity_set(1).entity_ids, dtype=np.int64)
        edge_row_by_vertex_key = {
            tuple(sorted(int(vertex_ids[int(vertex)]) for vertex in edge)): row
            for row, edge in enumerate(edge_vertices)
        }
        cell_edge_rows: dict[int, tuple[int, ...]] = {}
        for cell_id, (block_index, cell_row) in root_cell_rows.items():
            cell_vertices = np.asarray(
                root_mesh.blocks[block_index].vertices, dtype=np.int64
            )[cell_row]
            rows: list[int] = []
            for local_edge in topology.entities[1]:
                key = tuple(
                    sorted(
                        int(vertex_ids[int(cell_vertices[int(local_vertex)])])
                        for local_vertex in local_edge
                    )
                )
                edge_row = edge_row_by_vertex_key.get(key)
                if edge_row is None:
                    raise ValueError(
                        "Mapped root local edge lacks actual global-edge incidence."
                    )
                rows.append(edge_row)
            cell_edge_rows[int(cell_id)] = tuple(rows)
        for row, (dimension, entity_id, cell_id, local_entity) in enumerate(
            zip(
                segment_root_dimensions,
                segment_root_entity_global_ids,
                segment_root_cell_global_ids,
                segment_root_local_entity_indices,
                strict=True,
            )
        ):
            dimension = int(dimension)
            entity_id = int(entity_id)
            cell_id = int(cell_id)
            local_entity = int(local_entity)
            owner = root_cell_rows.get(cell_id)
            if owner is None:
                raise ValueError("Mapped source segment references an absent root cell.")
            block_index, cell_row = owner
            refs = tuple(
                tuple(Fraction.from_float(float(value)) for value in point)
                for point in np.asarray(segment_root_reference_vertices[row])
            )
            if refs[0] == refs[1]:
                raise ValueError(
                    "Mapped source segment collapses in its exact root chart."
                )
            if dimension == 1:
                local_edges = topology.entities[1]
                if local_entity < 0 or local_entity >= len(local_edges):
                    raise ValueError(
                        "Mapped root edge segment names an absent local root edge."
                    )
                edge_row = cell_edge_rows[cell_id][local_entity]
                if (
                    edge_row < 0
                    or edge_row >= edge_vertices.shape[0]
                    or int(root_edge_ids[edge_row]) != entity_id
                ):
                    raise ValueError(
                        "Mapped source segment changes its actual root edge identity."
                    )
                endpoints = tuple(
                    tuple(Fraction.from_float(float(value)) for value in point)
                    for point in np.asarray(topology.vertices, dtype=np.float64)[
                        np.asarray(local_edges[local_entity], dtype=np.int64)
                    ]
                )
                if refs == endpoints:
                    expected_orientation = 1
                elif refs == endpoints[::-1]:
                    expected_orientation = -1
                else:
                    raise ValueError(
                        "Mapped root-edge segment endpoints change its exact edge chart."
                    )
            elif dimension == 2:
                if local_entity < 0 or local_entity >= face_rows.shape[1]:
                    raise ValueError(
                        "Mapped root-face segment names an absent local root face."
                    )
                face_row = int(face_rows[cell_connectivity_rows[cell_id], local_entity])
                if int(face_ids[face_row]) != entity_id:
                    raise ValueError(
                        "Mapped source segment changes its actual root face identity."
                    )
                MappedReferenceBoundarySource._root_face_uv_for_domain(
                    domain,
                    entity_id,
                    cell_id,
                    local_entity,
                    np.asarray(refs, dtype=object),
                )
                delta = tuple(last - first for first, last in zip(*refs, strict=True))
                expected_orientation = (
                    1 if next(value for value in delta if value) > 0 else -1
                )
            else:
                if local_entity != -1 or entity_id != cell_id:
                    raise ValueError(
                        "Mapped root-cell segment changes its exact cell ancestry."
                    )
                local_vertices = tuple(
                    tuple(Fraction.from_float(float(value)) for value in point)
                    for point in np.asarray(topology.vertices, dtype=np.float64)
                )
                if any(point not in local_vertices for point in refs):
                    raise ValueError(
                        "Mapped root-cell segment endpoints are not actual root vertices."
                    )
                delta = tuple(last - first for first, last in zip(*refs, strict=True))
                expected_orientation = (
                    1 if next(value for value in delta if value) > 0 else -1
                )
            if int(segment_orientations[row]) != expected_orientation:
                raise ValueError(
                    "Mapped source segment changes its retained root-chart orientation."
                )


def _readonly(values: ArrayLike) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _select_scope_keys(
    requested: tuple[tuple[str, tuple[str, ...]], ...] | None,
    available: tuple[tuple[str, tuple[str, ...]], ...],
    name: str,
    /,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    all_keys = tuple(sorted(set(available)))
    if requested is None:
        return all_keys
    keys = tuple(
        (
            canonical_identifier(entity_id, f"{name}_source_entity_id"),
            tuple(canonical_identifier(part, f"{name}_occurrence_id") for part in path),
        )
        for entity_id, path in requested
    )
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= set(all_keys):
        raise ValueError(
            f"{name}_scope_keys must be a nonempty subset of actual root relation keys."
        )
    return tuple(sorted(keys))


def _chart_id(
    kind: str,
    domain: MappedReferenceDomain,
    relation_id: str,
    *parts: object,
) -> str:
    return canonical_fingerprint(
        {
            "kind": kind,
            "domain": domain.domain_id,
            "source": (domain.source_id, domain.source_revision),
            "root_geometry": domain.reference_geometry_id,
            "relation": relation_id,
            "ancestry": parts,
        }
    )


__all__ = ["MappedPlcRootRelation", "MappedReferenceBoundarySource"]
