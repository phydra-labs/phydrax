#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-coordinate realization for constrained PLC constructions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from ..._strict import StrictModule
from ..._trainable import NonTrainableState

from ...discretization import (
    _cell_geometry as cell_geometry_owner,
    CellGeometrySpec,
    CellMesh,
)
from ...discretization._cell_complex import TetrahedralConnectivity
from ...discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    PerCellRestrictedCellGeometryElement,
)
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_mesh import SimplicialConnectivity
from ...discretization._coordinate_enclosure import compose_expression, RationalPolynomial
from ...discretization._nested_reference_types import (
    ExactReferenceChartExpression,
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
)
from ...discretization._reference_cell import reference_cell_topology
from ...geometry._affine_plc_domain import AffinePlcDomain
from ...geometry._domain_coverage import PiecewiseLinearDomain
from ...geometry._piecewise_linear_boundary import PiecewiseLinearBoundarySource
from ...geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ...geometry._planar_primitives import intersection as planar_intersection
from ...geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from ...typing import checked
from .._volume_generation import VolumeConstruction
from ._native_sources import NativePlcSource


class _PlcCurveAuthority(StrictModule, NonTrainableState):
    """Exact source-edge scope backed by the original PLC and its domain."""

    source: NativePlcSource
    domain: AffinePlcDomain

    def __init__(self, source: NativePlcSource, domain: PiecewiseLinearDomain, /) -> None:
        self.source = source
        self.domain = AffinePlcDomain(
            domain, np.eye(3), np.zeros(3), source_revision=source.source_revision
        )
        self.require_current()

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def source_revision(self) -> str:
        return self.source.source_revision

    @property
    def envelope_id(self) -> str:
        return self.source.binding_id

    @property
    def edges(self) -> np.ndarray:
        return np.asarray(
            self.source.complex.plc_edge_vertices, dtype=np.int64
        ).reshape((-1, 2))

    @property
    def scope_entity_ids_by_mesh_row(self) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(self.source.complex.vertex_entity_ids, dtype=np.int64),
            np.asarray(self.source.complex.plc_edge_entity_ids, dtype=np.int64),
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        return self.source.scope_entity_ids(dimension)

    def scope_entity_set_id(self, dimension: int, /) -> str:
        return self.source.scope_entity_set_id(dimension)

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        return self.source.source_entity_key_for_scope(dimension, entity_id)

    def require_current(self, /) -> None:
        self.source.validate_restored()
        self.domain.require_current()
        if (
            self.domain.source_id != self.source.source_id
            or self.domain.source_revision != self.source.source_revision
            or not np.array_equal(
                self.domain.reference_domain.vertices, self.source.complex.vertices
            )
        ):
            raise ValueError("PLC curve authority changed its original source vertices.")


def simplex_volume_entities(mesh: CellMesh, /) -> tuple[np.ndarray, np.ndarray]:
    connectivity = mesh.connectivity
    if mesh.topological_dimension != 3:
        raise ValueError(
            "Native PLC output requires three-dimensional simplex incidence."
        )
    if isinstance(connectivity, SimplicialConnectivity):
        edges = np.asarray(connectivity.entities[1], dtype=np.int64)
        faces = np.asarray(connectivity.entities[2], dtype=np.int64)
        cell_faces = np.asarray(connectivity.cell_entities[2], dtype=np.int64)
    elif isinstance(connectivity, TetrahedralConnectivity):
        edges = np.asarray(connectivity.edges, dtype=np.int64)
        faces = np.asarray(connectivity.faces, dtype=np.int64)
        cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int64)
    else:
        raise ValueError("Native PLC output requires canonical tetrahedral connectivity.")
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or faces.ndim != 2
        or faces.shape[1] != 3
        or cell_faces.shape != (sum(block.cell_count for block in mesh.blocks), 4)
    ):
        raise ValueError(
            "Native PLC volume incidence must retain tetrahedral cells, faces, and edges."
        )
    return edges, faces


def volume_cell_entity_rows(
    mesh: CellMesh, cell_global_ids: np.ndarray | None = None, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-local edge/face rows aligned to stable cell global IDs."""
    connectivity = mesh.connectivity
    if mesh.topological_dimension != 3:
        raise ValueError("Volume cell incidence requires a three-dimensional mesh.")
    if isinstance(connectivity, SimplicialConnectivity):
        edge_vertices = np.asarray(connectivity.entities[1], dtype=np.int64)
        face_vertices = np.asarray(connectivity.entities[2], dtype=np.int64)
    elif isinstance(connectivity, TetrahedralConnectivity):
        edge_vertices = np.asarray(connectivity.edges, dtype=np.int64)
        face_vertices = np.asarray(connectivity.faces, dtype=np.int64)
    else:
        from ...discretization._hexahedral import HexahedralConnectivity

        if not isinstance(connectivity, HexahedralConnectivity):
            raise ValueError(
                "Volume cell incidence requires simplex, tetrahedron, or hexahedron cells."
            )
        edge_vertices = np.asarray(connectivity.edges, dtype=np.int64)
        face_vertices = np.asarray(connectivity.faces, dtype=np.int64)
    if edge_vertices.ndim != 2 or face_vertices.ndim != 2:
        raise ValueError("Canonical volume edge/face incidence must be two-dimensional.")
    edge_rows = {
        tuple(sorted(int(value) for value in vertices)): row
        for row, vertices in enumerate(edge_vertices.tolist())
    }
    face_rows = {
        tuple(sorted(int(value) for value in vertices)): row
        for row, vertices in enumerate(face_vertices.tolist())
    }
    if (
        len(edge_rows) != edge_vertices.shape[0]
        or len(face_rows) != face_vertices.shape[0]
    ):
        raise ValueError("Canonical volume edge/face incidence is not unique.")
    rows_by_cell_id: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        if topology.dimension != 3:
            raise ValueError("Volume entity incidence found a non-volume cell block.")
        cells = np.asarray(block.vertices, dtype=np.int64)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        local_edges = tuple(
            tuple(int(value) for value in row) for row in topology.entities[1]
        )
        local_faces = tuple(
            tuple(int(value) for value in row) for row in topology.entities[2]
        )
        for identifier_raw, cell_values in zip(
            identifiers.tolist(), cells.tolist(), strict=True
        ):
            cell = tuple(int(value) for value in cell_values)
            identifier = int(identifier_raw)
            if identifier in rows_by_cell_id:
                raise ValueError("Volume cell global IDs must be unique across blocks.")
            try:
                cell_edges = tuple(
                    edge_rows[tuple(sorted(int(cell[local]) for local in edge))]
                    for edge in local_edges
                )
                cell_faces = tuple(
                    face_rows[tuple(sorted(int(cell[local]) for local in face))]
                    for face in local_faces
                )
            except KeyError as error:
                raise ValueError(
                    "A volume cell entity is absent from canonical incidence."
                ) from error
            rows_by_cell_id[identifier] = (cell_edges, cell_faces)
    raw_ids = (
        np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
        if cell_global_ids is None
        else np.asarray(cell_global_ids)
    )
    if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
        raise TypeError("cell_global_ids must be an integer vector.")
    if (
        raw_ids.dtype.kind == "u"
        and raw_ids.size
        and np.any(raw_ids > np.iinfo(np.int64).max)
    ):
        raise ValueError("cell_global_ids must fit int64.")
    identifiers = np.asarray(raw_ids, dtype=np.int64)
    if (
        identifiers.size != len(rows_by_cell_id)
        or np.unique(identifiers).size != identifiers.size
    ):
        raise ValueError(
            "Volume cell identity bank does not cover every cell exactly once."
        )
    try:
        ordered = tuple(
            rows_by_cell_id[int(identifier)] for identifier in identifiers.tolist()
        )
    except KeyError as error:
        raise ValueError(
            "A requested volume cell ID is absent from connectivity."
        ) from error
    if not ordered:
        return np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=np.int64)
    edge_counts = {len(edges) for edges, _ in ordered}
    face_counts = {len(faces) for _, faces in ordered}
    if len(edge_counts) != 1 or len(face_counts) != 1:
        raise ValueError(
            "Volume local entity arity must be uniform for this incidence query."
        )
    return np.asarray([edges for edges, _ in ordered], dtype=np.int64), np.asarray(
        [faces for _, faces in ordered], dtype=np.int64
    )


def _mapped_plc_geometry(
    source: NativePlcSource, construction: VolumeConstruction, /
) -> CellGeometrySpec:
    domain = source.mapped_domain
    if domain is None or source.complex.geometry is None:
        raise ValueError(
            "Mapped PLC realization requires both its exact geometry and MappedReferenceDomain."
        )
    if (
        not isinstance(construction.domain, type(domain))
        or construction.domain.domain_id != domain.domain_id
        or (construction.source_id, construction.source_revision)
        != (source.source_id, source.source_revision)
    ):
        raise ValueError(
            "PLC construction changed its mapped source or root-domain identity."
        )
    controls = construction.reference_chart_controls
    mesh = construction.mesh
    if len(mesh.blocks) != 1 or not controls:
        raise ValueError(
            "Mapped PLC geometry requires exact root charts for one generated tetrahedron block."
        )
    target_block = mesh.blocks[0]
    target_ids = np.asarray(target_block.global_ids, dtype=np.int64)
    if (
        target_block.cell_kind not in {"simplex:3", "tetrahedron"}
        or len(controls) != target_block.cell_count
        or tuple(control.target_cell_id for control in controls)
        != tuple(int(value) for value in target_ids.tolist())
    ):
        raise ValueError(
            "Mapped PLC target cells and exact root-chart records are not aligned."
        )

    root_mesh = domain.reference_mesh
    root_elements, root_routes, _ = domain.source_geometry.resolve(root_mesh)
    coordinates = np.asarray(domain.source_geometry.coordinates, dtype=np.float64)
    root_cell_rows = {
        int(identifier): row for row, identifier in enumerate(domain.root_cell_ids)
    }
    root_vertex_ids = np.asarray(root_mesh.vertex_global_ids, dtype=np.int64)
    source_elements = []
    charts: list[tuple[ExactReferenceChartExpression, ...]] = []
    source_routes = []
    parent_vertex_ids = []
    source_cell_ids = []
    exact_vertices = []
    for target_row, control in enumerate(controls):
        if (
            control.target_block_index != 0
            or control.target_cell_row != target_row
            or control.target_cell_kind != target_block.cell_kind
            or control.source_id != domain.source_id
            or control.source_revision != domain.source_revision
            or control.mapped_domain_id != domain.domain_id
            or control.source_geometry_id != domain.reference_geometry_id
            or control.source_topology_id != root_mesh.topology_id
        ):
            raise ValueError(
                "Mapped PLC chart control changed its source, target, or domain binding."
            )
        source_block = root_mesh.blocks[control.source_block_index]
        source_cells = np.asarray(source_block.vertices, dtype=np.int64)
        source_ids = np.asarray(source_block.global_ids, dtype=np.int64)
        if (
            source_block.cell_kind != control.source_cell_kind
            or control.source_cell_row < 0
            or control.source_cell_row >= source_block.cell_count
            or int(source_ids[control.source_cell_row]) != control.source_cell_id
        ):
            raise ValueError("Mapped PLC chart control names an absent root cell.")
        domain_row = root_cell_rows.get(control.source_cell_id)
        if (
            domain_row is None
            or domain.root_cell_occurrence_paths[domain_row]
            != control.source_occurrence_path
            or domain.region_ids[int(domain.cell_regions[domain_row])]
            != control.source_region_id
        ):
            raise ValueError(
                "Mapped PLC chart control changed its root occurrence or material region."
            )
        root_element = root_elements[control.source_block_index]
        if not isinstance(root_element, cell_geometry_owner.ScalarCoordinateElement):
            raise ValueError(
                "Mapped PLC roots require scalar source coordinate elements."
            )
        base_element = cell_geometry_owner.coordinate_reference_chain(root_element)[0]
        root_chart = cell_geometry_owner.restriction_chart_arguments(
            root_element,
            ancestor=base_element,
            cell_row=control.source_cell_row,
        )
        chart: tuple[ExactReferenceChartExpression, ...] = (
            control.arguments
            if root_chart is None
            else tuple(
                compose_expression(argument, control.arguments) for argument in root_chart
            )
        )
        if any(
            not isinstance(argument, (Mapping, RationalPolynomial)) for argument in chart
        ):
            raise ValueError(
                "Mapped PLC tetrahedra require exact polynomial or rational root charts."
            )
        source_elements.append(base_element)
        charts.append(tuple(chart))
        source_routes.append(
            np.asarray(root_routes[control.source_block_index], dtype=np.int64)[
                control.source_cell_row
            ]
        )
        source_cell_ids.append(control.source_cell_id)
        exact_vertices.append(control.source_reference_vertices)

        parent_vertex_ids.append(
            root_vertex_ids[source_cells[control.source_cell_row]].tolist()
        )

    element_ids = {element.element_id for element in source_elements}
    if len(element_ids) != 1:
        raise ValueError("Mapped PLC cells span incompatible root coordinate bases.")
    root_element = source_elements[0]
    local_rows = np.arange(target_block.cell_count, dtype=np.int64)
    root_ids = np.asarray(source_cell_ids, dtype=np.int64)
    exact_reference_vertices = tuple(exact_vertices)
    floating_reference_vertices = np.asarray(
        [
            [[float(value) for value in point] for point in cell]
            for cell in exact_reference_vertices
        ],
        dtype=np.float64,
    )
    reference_map = ReferenceCellMapWitnesses(
        local_rows,
        root_ids,
        target_block.cell_kind,
        controls[0].source_cell_kind,
        floating_reference_vertices,
        exact_reference_vertices,
        source_reference_chart_arguments=freeze_reference_chart_arguments(
            tuple(control.arguments for control in controls)
        ),
    )
    restriction = CellGeometryRestrictionSource(
        domain.reference_geometry_id,
        root_mesh.topology_id,
        {target_block.name: root_ids},
        {target_block.name: np.asarray(parent_vertex_ids, dtype=np.int64)},
        block_parent_reference_maps={target_block.name: reference_map},
    )
    element = PerCellRestrictedCellGeometryElement(
        root_element,
        target_block.cell_kind,
        tuple(int(value) for value in target_ids.tolist()),
        tuple(int(value) for value in root_ids.tolist()),
        tuple(charts),
    )
    return CellGeometrySpec(
        {target_block.name: element},
        {target_block.name: np.stack(source_routes)},
        coordinates,
        restriction_source=restriction,
    )


@checked
def plc_cell_geometry(
    source: NativePlcSource, construction: VolumeConstruction, /
) -> CellGeometrySpec:
    """Bind the volume mesh to its exact PLC or mapped-root source geometry."""
    if source.mapped_domain is not None:
        return _mapped_plc_geometry(source, construction)
    if source.complex.geometry is not None:
        raise ValueError(
            "Mapped PLC geometry requires its original MappedReferenceDomain."
        )
    return CellGeometrySpec.affine(construction.mesh)


@dataclass(frozen=True, slots=True, eq=False)
class PhysicalPlcTetrahedra:
    """Physical evaluation of a PLC mesh's actual coordinate map."""

    geometry: CellGeometrySpec
    points: np.ndarray
    tetrahedra: np.ndarray


@checked
def physical_plc_tetrahedra(
    source: NativePlcSource, construction: VolumeConstruction, /
) -> PhysicalPlcTetrahedra:
    """Evaluate every PLC tetrahedron in physical coordinates for spatial kernels."""
    geometry = plc_cell_geometry(source, construction)
    mesh = construction.mesh
    block = mesh.blocks[0]
    cells = np.asarray(block.vertices, dtype=np.int64)
    if block.cell_kind not in {"tetrahedron", "simplex:3"}:
        raise ValueError(
            "A PLC carrier for physical power construction must be tetrahedral."
        )
    reference_vertices = tuple(
        tuple(Fraction.from_float(float(value)) for value in vertex)
        for vertex in reference_cell_topology(block.cell_kind).vertices
    )
    references = (reference_vertices,) * block.cell_count
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    physical_corners = geometry.evaluate_exact(mesh, cell_ids, references)
    flat_vertices = cells.reshape(-1)
    flat_points = tuple(
        point for cell_points in physical_corners for point in cell_points
    )
    order = np.argsort(flat_vertices, kind="stable")
    sorted_vertices = flat_vertices[order]
    sorted_points = tuple(flat_points[int(row)] for row in order)
    unique_vertices, starts, counts = np.unique(
        sorted_vertices, return_index=True, return_counts=True
    )
    group_rows = np.repeat(np.arange(unique_vertices.size, dtype=np.int64), counts)
    if any(
        sorted_points[row] != sorted_points[int(starts[group])]
        for row, group in enumerate(group_rows.tolist())
    ):
        raise ValueError(
            "PLC source geometry is discontinuous across shared tetrahedron vertices."
        )
    points = np.empty((mesh.coordinates.shape[0], 3), dtype=np.float64)
    points[unique_vertices] = np.asarray(
        [
            [float(component) for component in sorted_points[int(start)]]
            for start in starts
        ],
        dtype=np.float64,
    )
    if unique_vertices.size != points.shape[0] or not np.all(np.isfinite(points)):
        raise ValueError(
            "Every PLC mesh vertex must have a finite physical source-map value."
        )
    points.setflags(write=False)
    cells.setflags(write=False)
    return PhysicalPlcTetrahedra(geometry, points, cells)


@checked
def mapped_plc_boundary_chart_witness(
    source: NativePlcSource,
    construction: VolumeConstruction,
    geometry: CellGeometrySpec,
    query: MappedReferenceBoundarySource,
    /,
) -> SourceBoundaryChartWitness:
    """Prove exact output-face ancestry in each selected mapped source chart."""
    domain = source.mapped_domain
    if (
        domain is None
        or source.root_relation is None
        or query.authority is not source
        or query.domain.domain_id != domain.domain_id
        or construction.domain.domain_id != domain.domain_id
    ):
        raise ValueError(
            "Mapped PLC chart witness requires the original source, root, and query authority."
        )
    query.require_current()
    mesh = construction.mesh
    if len(mesh.blocks) != 1:
        raise ValueError(
            "Mapped PLC chart witnesses require one exact simplex target block."
        )
    block = mesh.blocks[0]
    if block.cell_kind not in {"simplex:3", "tetrahedron"}:
        raise ValueError("Mapped PLC chart witnesses require tetrahedral targets.")
    controls_by_cell = {
        control.target_cell_id: control
        for control in construction.reference_chart_controls
    }
    cell_ids = tuple(int(value) for value in np.asarray(block.global_ids, dtype=np.int64))
    if len(controls_by_cell) != len(cell_ids) or set(controls_by_cell) != set(cell_ids):
        raise ValueError(
            "Every mapped PLC tetrahedron needs one exact target-to-root chart."
        )

    def cross2(
        first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction], /
    ) -> Fraction:
        return first[0] * second[1] - first[1] * second[0]

    def signed_double_area(points: tuple[tuple[Fraction, Fraction], ...], /) -> Fraction:
        return sum(
            cross2(points[row], points[(row + 1) % len(points)])
            for row in range(len(points))
        )

    def triangle_parameters(
        point: tuple[Fraction, Fraction],
        triangle: tuple[tuple[Fraction, Fraction], ...],
        /,
    ) -> tuple[Fraction, Fraction]:
        first = (triangle[1][0] - triangle[0][0], triangle[1][1] - triangle[0][1])
        second = (triangle[2][0] - triangle[0][0], triangle[2][1] - triangle[0][1])
        delta = (point[0] - triangle[0][0], point[1] - triangle[0][1])
        determinant = cross2(first, second)
        if determinant == 0:
            raise ValueError("Mapped root-face triangle chart is singular.")
        return cross2(delta, second) / determinant, cross2(first, delta) / determinant

    # Use the same exact convex restriction as the global-affine PLC charts.
    clip_triangle = planar_intersection

    target_topology = reference_cell_topology(block.cell_kind)
    target_local_faces = tuple(
        tuple(int(value) for value in face) for face in target_topology.entities[2]
    )
    target_reference_vertices = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in target_topology.vertices
    )
    _, target_face_vertices = simplex_volume_entities(mesh)
    target_face_rows = {
        frozenset(int(value) for value in face): row
        for row, face in enumerate(target_face_vertices)
    }
    target_face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    _, target_cell_face_rows = volume_cell_entity_rows(
        mesh, np.asarray(cell_ids, dtype=np.int64)
    )
    target_cell_row_by_id = {identifier: row for row, identifier in enumerate(cell_ids)}
    target_face_owners: dict[int, list[tuple[int, int]]] = {}
    for target_cell_id, target_row in target_cell_row_by_id.items():
        for local_face, face_row in enumerate(target_cell_face_rows[target_row].tolist()):
            target_face_owners.setdefault(int(face_row), []).append(
                (target_cell_id, local_face)
            )

    source_keys = {
        source.source_entity_key_for_scope(2, int(entity_id))
        for entity_id in source.scope_entity_ids(2).tolist()
    }
    selected_keys = set(query.face_scope_keys)
    charts_by_face_key: dict[tuple[int, tuple[str, tuple[str, ...]]], list[str]] = {}
    for chart_id in query.face_chart_ids:
        key = query.chart_source_key(2, chart_id)
        if key not in source_keys:
            raise ValueError(
                "Mapped PLC query chart is not bound to its source entity bank."
            )
        chart_row = query.chart_row(2, chart_id)
        root_face_id = int(query.polygon_root_face_global_ids[chart_row])
        charts_by_face_key.setdefault((root_face_id, key), []).append(chart_id)

    root_mesh = domain.reference_mesh
    root_topology_by_kind = {
        kind: reference_cell_topology(kind)
        for kind in {
            control.source_cell_kind for control in construction.reference_chart_controls
        }
    }
    root_cell_rows = {
        int(identifier): row for row, identifier in enumerate(domain.root_cell_ids)
    }
    _, root_cell_face_rows = volume_cell_entity_rows(
        root_mesh, np.asarray(domain.root_cell_ids, dtype=np.int64)
    )
    root_face_ids = np.asarray(root_mesh.entity_set(2).entity_ids, dtype=np.int64)

    def local_root_face(
        cell_kind: str,
        reference_points: tuple[tuple[Fraction, ...], ...],
        /,
    ) -> int:
        topology = root_topology_by_kind[cell_kind]
        candidates: list[int] = []
        for local_face, local_vertices in enumerate(topology.entities[2]):
            if cell_kind in {"tetrahedron", "simplex:3"}:
                opposite = next(
                    index
                    for index in range(len(topology.vertices))
                    if index not in local_vertices
                )
                for point in reference_points:
                    barycentric = (Fraction(1) - sum(point), *point)
                    if barycentric[opposite] != 0:
                        break
                else:
                    candidates.append(local_face)
            elif cell_kind == "hexahedron":
                face = tuple(
                    tuple(
                        Fraction.from_float(float(value))
                        for value in topology.vertices[int(index)]
                    )
                    for index in local_vertices
                )
                constant_axes = tuple(
                    axis
                    for axis in range(3)
                    if all(point[axis] == face[0][axis] for point in face)
                )
                if len(constant_axes) != 1:
                    continue
                axis = constant_axes[0]
                if all(point[axis] == face[0][axis] for point in reference_points):
                    candidates.append(local_face)
            else:
                raise ValueError(
                    "Mapped PLC source chart witness encountered an unsupported root cell kind."
                )
        if len(candidates) != 1:
            raise ValueError(
                "A mapped PLC output face must lie on one exact root-cell face."
            )
        return candidates[0]

    chart_cell_ids: list[int] = []
    chart_face_ids: list[int] = []
    chart_local_faces: list[int] = []
    target_reference_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    chart_ids: list[str] = []
    source_entity_ids: list[str] = []
    occurrence_paths: list[tuple[str, ...]] = []
    source_parameter_rows: list[tuple[tuple[Fraction, Fraction], ...]] = []
    orientations: list[int] = []
    face_sources = tuple(construction.face_source_ids)
    face_occurrences = tuple(construction.face_source_occurrence_paths)
    if len(construction.boundary_faces) != len(face_sources) or len(face_sources) != len(
        face_occurrences
    ):
        raise ValueError(
            "Mapped PLC boundary source keys must align with actual recovered facets."
        )
    for output_face_row, face_vertices in enumerate(
        np.asarray(construction.boundary_faces, dtype=np.int64)
    ):
        source_key = (face_sources[output_face_row], face_occurrences[output_face_row])
        if source_key not in source_keys:
            raise ValueError(
                "Recovered PLC boundary facet names an absent source entity key."
            )
        if source_key not in selected_keys:
            continue
        face_row = target_face_rows.get(frozenset(int(value) for value in face_vertices))
        if face_row is None:
            raise ValueError(
                "Recovered PLC boundary facet is absent from target connectivity."
            )
        owners = target_face_owners.get(face_row, ())
        if not 1 <= len(owners) <= 2:
            raise ValueError(
                "A mapped PLC boundary facet must have one or two target cell/local-face owners."
            )
        # The query retains one root-face side per material interface; witness
        # the target owner whose exact root cell is that selected side.
        selected_root_cells = {
            int(query.polygon_root_cell_global_ids[query.chart_row(2, chart_id)])
            for (_, key), chart_ids_for_face in charts_by_face_key.items()
            if key == source_key
            for chart_id in chart_ids_for_face
        }
        side_owners = tuple(
            owner
            for owner in owners
            if owner[0] in controls_by_cell
            and controls_by_cell[owner[0]].source_cell_id in selected_root_cells
        )
        if len(side_owners) != 1:
            raise ValueError(
                "A mapped PLC boundary facet must have one target owner on its selected root side."
            )
        target_cell_id, target_local_face = side_owners[0]
        control = controls_by_cell[target_cell_id]
        target_local_vertices = target_local_faces[target_local_face]
        target_reference_corners = tuple(
            target_reference_vertices[index] for index in target_local_vertices
        )
        root_reference_corners = tuple(
            control.source_reference_vertices[index] for index in target_local_vertices
        )
        root_local_face = local_root_face(
            control.source_cell_kind,
            root_reference_corners,
        )
        root_domain_row = root_cell_rows.get(control.source_cell_id)
        if root_domain_row is None:
            raise ValueError("Mapped PLC chart control names an absent root-domain cell.")
        root_face_row = int(root_cell_face_rows[root_domain_row, root_local_face])
        root_face_id = int(root_face_ids[root_face_row])
        target_uv = tuple(
            query.root_face_uv(
                root_face_id,
                control.source_cell_id,
                root_local_face,
                np.asarray(root_reference_corners, dtype=object),
            )
        )
        target_area = abs(signed_double_area(target_uv))
        if target_area == 0:
            raise ValueError(
                "Mapped PLC target boundary facet is degenerate in its exact root-face chart."
            )
        candidates = charts_by_face_key.get((root_face_id, source_key), ())
        if not candidates:
            raise ValueError(
                "Mapped PLC target facet has no matching exact source chart."
            )
        covered_area = Fraction(0)
        for chart_id in candidates:
            chart_row = query.chart_row(2, chart_id)
            if (
                int(query.polygon_root_cell_global_ids[chart_row])
                != control.source_cell_id
                or int(query.polygon_root_local_face_indices[chart_row])
                != root_local_face
            ):
                continue
            source_uv = tuple(query.chart_face_uv(chart_id))
            clipped = clip_triangle(target_uv, source_uv)
            if len(clipped) < 3:
                continue
            area = abs(signed_double_area(clipped))
            if area == 0:
                continue
            covered_area += area
            source_entity_id, occurrence = query.chart_source_key(2, chart_id)
            for triangle_row in range(1, len(clipped) - 1):
                uv_triangle = (
                    clipped[0],
                    clipped[triangle_row],
                    clipped[triangle_row + 1],
                )
                target_parameters = tuple(
                    triangle_parameters(point, target_uv) for point in uv_triangle
                )
                source_parameters = tuple(
                    triangle_parameters(point, source_uv) for point in uv_triangle
                )
                target_reference_triangle = tuple(
                    tuple(
                        target_reference_corners[0][axis]
                        + target_parameters[point_row][0]
                        * (
                            target_reference_corners[1][axis]
                            - target_reference_corners[0][axis]
                        )
                        + target_parameters[point_row][1]
                        * (
                            target_reference_corners[2][axis]
                            - target_reference_corners[0][axis]
                        )
                        for axis in range(3)
                    )
                    for point_row in range(3)
                )
                target_orientation = cross2(
                    (
                        target_parameters[1][0] - target_parameters[0][0],
                        target_parameters[1][1] - target_parameters[0][1],
                    ),
                    (
                        target_parameters[2][0] - target_parameters[0][0],
                        target_parameters[2][1] - target_parameters[0][1],
                    ),
                )
                source_orientation = cross2(
                    (
                        source_parameters[1][0] - source_parameters[0][0],
                        source_parameters[1][1] - source_parameters[0][1],
                    ),
                    (
                        source_parameters[2][0] - source_parameters[0][0],
                        source_parameters[2][1] - source_parameters[0][1],
                    ),
                )
                if target_orientation == 0 or source_orientation == 0:
                    raise ValueError(
                        "Mapped PLC source/target chart intersection is degenerate."
                    )
                chart_cell_ids.append(target_cell_id)
                chart_face_ids.append(int(target_face_ids[face_row]))
                chart_local_faces.append(target_local_face)
                target_reference_rows.append(target_reference_triangle)
                chart_ids.append(chart_id)
                source_entity_ids.append(source_entity_id)
                occurrence_paths.append(occurrence)
                source_parameter_rows.append(source_parameters)
                orientations.append(
                    1 if target_orientation * source_orientation > 0 else -1
                )
        if covered_area != target_area:
            raise ValueError(
                "Mapped PLC source charts do not exactly partition a recovered target facet."
            )
    if not chart_cell_ids:
        raise ValueError("Mapped PLC boundary query produced no target chart witnesses.")
    target_reference_values = np.asarray(
        [
            [[float(value) for value in point] for point in row]
            for row in target_reference_rows
        ],
        dtype=np.float64,
    )
    source_parameter_values = np.asarray(
        [
            [[float(value) for value in point] for point in row]
            for row in source_parameter_rows
        ],
        dtype=np.float64,
    )
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=query.source_query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(chart_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(chart_face_ids, dtype=np.int64),
        local_entity_indices=np.asarray(chart_local_faces, dtype=np.int32),
        target_reference_vertices=target_reference_values,
        target_reference_rational_tokens=rational_chart_tokens(target_reference_rows),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrence_paths),
        source_parameters=source_parameter_values,
        source_parameter_rational_tokens=rational_chart_tokens(source_parameter_rows),
        orientations=np.asarray(orientations, dtype=np.int8),
    )


@checked
def mapped_plc_point_witness(
    source: NativePlcSource,
    construction: VolumeConstruction,
    geometry: CellGeometrySpec,
    query: MappedReferenceBoundarySource,
    source_vertex_entity_ids: np.ndarray,
    /,
) -> SourceBoundaryPointWitness:
    """Bind preserved PLC corners to their exact mapped-root source keys."""
    if source.mapped_domain is None or query.authority is not source:
        raise ValueError(
            "Mapped PLC point witnesses require their original mapped source authority."
        )
    query.require_current()
    entity_ids = np.asarray(source_vertex_entity_ids, dtype=np.int64)
    if (
        entity_ids.ndim != 1
        or entity_ids.size == 0
        or np.unique(entity_ids).size != entity_ids.size
    ):
        raise ValueError(
            "Mapped PLC point witness requires unique authored source vertex IDs."
        )
    source_rows_by_key = {
        source.source_entity_key_for_scope(0, int(identifier)): row
        for row, identifier in enumerate(
            np.asarray(source.complex.vertex_entity_ids, dtype=np.int64)
        )
    }
    input_vertices = np.asarray(construction.input_vertices, dtype=np.int64)
    target_vertex_ids = np.asarray(construction.mesh.vertex_global_ids, dtype=np.int64)
    source_keys = tuple(
        source.source_entity_key_for_scope(0, int(identifier))
        for identifier in entity_ids.tolist()
    )
    target_ids: list[int] = []
    for key in source_keys:
        source_row = source_rows_by_key.get(key)
        if source_row is None:
            raise ValueError(
                "Mapped PLC corner key is absent from its authored vertex bank."
            )
        matches = np.flatnonzero(input_vertices == source_row)
        if matches.size != 1:
            raise ValueError(
                "Every mapped PLC source corner must remain one target vertex."
            )
        target_ids.append(int(target_vertex_ids[int(matches[0])]))
    if len(set(target_ids)) != len(target_ids):
        raise ValueError(
            "Distinct mapped PLC source corners cannot share a target vertex ID."
        )
    return SourceBoundaryPointWitness(
        mesh_id=construction.mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=construction.mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=query.query_id,
        vertex_global_ids=np.asarray(target_ids, dtype=np.int64),
        source_entity_ids=tuple(key for key, _ in source_keys),
        occurrence_paths=tuple(path for _, path in source_keys),
    )


@checked
def mapped_plc_edge_chart_witness(
    source: NativePlcSource,
    construction: VolumeConstruction,
    geometry: CellGeometrySpec,
    query: MappedReferenceBoundarySource,
    /,
) -> SourceBoundaryChartWitness:
    """Bind recovered PLC segments to exact mapped source-curve intervals."""
    domain = source.mapped_domain
    if domain is None or source.root_relation is None or query.authority is not source:
        raise ValueError(
            "Mapped PLC edge witnesses require their original root-bound source authority."
        )
    query.require_current()
    mesh = construction.mesh
    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind not in {
        "simplex:3",
        "tetrahedron",
    }:
        raise ValueError(
            "Mapped PLC edge witnesses require one tetrahedral target block."
        )
    block = mesh.blocks[0]
    controls_by_cell = {
        control.target_cell_id: control
        for control in construction.reference_chart_controls
    }
    target_cell_ids = tuple(
        int(value) for value in np.asarray(block.global_ids, dtype=np.int64)
    )
    if len(controls_by_cell) != len(target_cell_ids) or set(controls_by_cell) != set(
        target_cell_ids
    ):
        raise ValueError(
            "Mapped PLC edge witnesses need exact root controls for every target cell."
        )
    target_topology = reference_cell_topology(block.cell_kind)
    target_local_edges = tuple(
        tuple(int(value) for value in edge) for edge in target_topology.entities[1]
    )
    target_edge_vertices, _ = simplex_volume_entities(mesh)
    target_edge_rows = {
        frozenset(int(value) for value in edge): row
        for row, edge in enumerate(target_edge_vertices)
    }
    target_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    target_cell_edge_rows, _ = volume_cell_entity_rows(
        mesh, np.asarray(target_cell_ids, dtype=np.int64)
    )
    target_cell_row_by_id = {
        identifier: row for row, identifier in enumerate(target_cell_ids)
    }
    target_edge_owners: dict[int, list[tuple[int, int]]] = {}
    for target_cell_id, target_row in target_cell_row_by_id.items():
        for local_edge, edge_row in enumerate(target_cell_edge_rows[target_row].tolist()):
            target_edge_owners.setdefault(int(edge_row), []).append(
                (target_cell_id, local_edge)
            )

    source_keys = {
        source.source_entity_key_for_scope(1, int(entity_id))
        for entity_id in source.scope_entity_ids(1).tolist()
    }
    selected_keys = set(query.edge_scope_keys)
    charts_by_key: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for chart_id in query.edge_chart_ids:
        key = query.chart_source_key(1, chart_id)
        if key not in source_keys:
            raise ValueError(
                "Mapped PLC edge query chart is not bound to its source entity bank."
            )
        charts_by_key.setdefault(key, []).append(chart_id)

    root_mesh = domain.reference_mesh
    root_reference_geometry = CellGeometrySpec.affine(root_mesh)
    root_cell_rows = {
        int(identifier): row for row, identifier in enumerate(domain.root_cell_ids)
    }
    root_entity_ids = {
        dimension: np.asarray(root_mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        for dimension in (1, 2)
    }
    root_cell_edge_rows, root_cell_face_rows = volume_cell_entity_rows(
        root_mesh, np.asarray(domain.root_cell_ids, dtype=np.int64)
    )
    root_cell_entity_rows = {1: root_cell_edge_rows, 2: root_cell_face_rows}

    def root_global_points(
        cell_id: int,
        points: tuple[tuple[Fraction, ...], ...],
        /,
    ) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(
                value if isinstance(value, Fraction) else Fraction(value)
                for value in point
            )
            for point in root_reference_geometry.evaluate_exact(
                root_mesh,
                np.asarray((cell_id,), dtype=np.int64),
                (points,),
            )[0]
        )

    def belongs_to_root_stratum(cell_id: int, dimension: int, entity_id: int, /) -> bool:
        if dimension == 3:
            return cell_id == entity_id
        root_row = root_cell_rows.get(cell_id)
        if root_row is None:
            return False
        entity_rows = root_cell_entity_rows[dimension][root_row]
        actual_ids = root_entity_ids[dimension][entity_rows]
        return entity_id in set(int(value) for value in actual_ids.tolist())

    def line_parameters(
        points: tuple[tuple[Fraction, ...], ...],
        first: tuple[Fraction, ...],
        last: tuple[Fraction, ...],
        /,
    ) -> tuple[Fraction, ...] | None:
        delta = tuple(right - left for left, right in zip(first, last, strict=True))
        varying = tuple(axis for axis, value in enumerate(delta) if value != 0)
        if not varying:
            raise ValueError("Mapped source segment has a degenerate exact root chart.")
        axis = varying[0]
        result = []
        for point in points:
            parameter = (point[axis] - first[axis]) / delta[axis]
            if any(
                first[other] + parameter * delta[other] != point[other]
                for other in range(len(delta))
            ):
                return None
            result.append(parameter)
        return tuple(result)

    chart_cell_ids: list[int] = []
    chart_edge_ids: list[int] = []
    chart_local_edges: list[int] = []
    target_reference_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    chart_ids: list[str] = []
    source_entity_ids: list[str] = []
    occurrence_paths: list[tuple[str, ...]] = []
    source_parameter_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    orientations: list[int] = []
    intervals_by_chart: dict[str, list[tuple[Fraction, Fraction]]] = {}
    if len(construction.segments) != len(construction.segment_source_ids) or len(
        construction.segment_source_ids
    ) != len(construction.segment_source_occurrence_paths):
        raise ValueError(
            "Recovered PLC segment source keys must align with actual target edges."
        )
    for output_edge_row, edge_vertices in enumerate(
        np.asarray(construction.segments, dtype=np.int64)
    ):
        source_key = (
            construction.segment_source_ids[output_edge_row],
            construction.segment_source_occurrence_paths[output_edge_row],
        )
        if source_key not in source_keys:
            raise ValueError(
                "Recovered PLC segment names an absent authored source edge key."
            )
        if source_key not in selected_keys:
            continue
        target_edge_row = target_edge_rows.get(
            frozenset(int(value) for value in edge_vertices)
        )
        if target_edge_row is None:
            raise ValueError(
                "Recovered PLC segment is absent from target edge connectivity."
            )
        owners = target_edge_owners.get(target_edge_row, ())
        if not owners:
            raise ValueError(
                "Recovered PLC segment has no target cell/local-edge incidence."
            )
        candidates: list[
            tuple[
                int,
                int,
                str,
                tuple[Fraction, Fraction],
                tuple[tuple[Fraction, ...], ...],
                tuple[Fraction, Fraction],
            ]
        ] = []
        for target_cell_id, target_local_edge in owners:
            control = controls_by_cell[target_cell_id]
            target_local_vertices = target_local_edges[target_local_edge]
            target_reference_points = tuple(
                control.source_reference_vertices[index]
                for index in target_local_vertices
            )
            target_global_points = root_global_points(
                control.source_cell_id, target_reference_points
            )
            for chart_id in charts_by_key.get(source_key, ()):
                chart_row = query.chart_row(1, chart_id)
                root_dimension = int(query.segment_root_dimensions[chart_row])
                root_entity_id = int(query.segment_root_entity_global_ids[chart_row])
                root_cell_id = int(query.segment_root_cell_global_ids[chart_row])
                if root_dimension not in (1, 2, 3):
                    continue
                if not belongs_to_root_stratum(
                    control.source_cell_id, root_dimension, root_entity_id
                ):
                    continue
                if not belongs_to_root_stratum(
                    root_cell_id, root_dimension, root_entity_id
                ):
                    continue
                source_reference_points = tuple(
                    tuple(Fraction(*value) for value in point)
                    for point in query.segment_root_reference_rational_tokens[chart_row]
                )
                source_global_points = root_global_points(
                    root_cell_id, source_reference_points
                )
                parameters = line_parameters(
                    target_global_points,
                    source_global_points[0],
                    source_global_points[1],
                )
                if parameters is not None and parameters[0] != parameters[1]:
                    first = max(Fraction(0), min(parameters))
                    last = min(Fraction(1), max(parameters))
                    if first >= last:
                        continue
                    source_parameters = (
                        (first, last) if parameters[1] > parameters[0] else (last, first)
                    )
                    target_interval = tuple(
                        (value - parameters[0]) / (parameters[1] - parameters[0])
                        for value in source_parameters
                    )
                    references = tuple(
                        tuple(
                            target_reference_points[0][axis]
                            + t
                            * (
                                target_reference_points[1][axis]
                                - target_reference_points[0][axis]
                            )
                            for axis in range(3)
                        )
                        for t in target_interval
                    )
                    candidates.append(
                        (
                            target_cell_id,
                            target_local_edge,
                            chart_id,
                            source_parameters,
                            references,
                            target_interval,
                        )
                    )
        if not candidates:
            raise ValueError(
                "Mapped PLC output edge has no exact authored root-curve chart."
            )
        owner = min(candidate[0] for candidate in candidates)
        selected = sorted(
            (candidate for candidate in candidates if candidate[0] == owner),
            key=lambda candidate: min(candidate[5]),
        )
        intervals = tuple(
            (min(candidate[5]), max(candidate[5])) for candidate in selected
        )
        if (
            intervals[0][0] != 0
            or intervals[-1][1] != 1
            or any(left[1] != right[0] for left, right in zip(intervals, intervals[1:]))
        ):
            raise ValueError(
                "Mapped source curve pieces do not partition the target edge."
            )
        for (
            target_cell_id,
            target_local_edge,
            chart_id,
            parameters,
            target_references,
            _,
        ) in selected:
            source_key = query.chart_source_key(1, chart_id)
            target_reference_rows.append(target_references)
            source_parameter_rows.append(((parameters[0],), (parameters[1],)))
            chart_cell_ids.append(target_cell_id)
            chart_edge_ids.append(int(target_edge_ids[target_edge_row]))
            chart_local_edges.append(target_local_edge)
            chart_ids.append(chart_id)
            source_entity_ids.append(source_key[0])
            occurrence_paths.append(source_key[1])
            orientations.append(1 if parameters[1] > parameters[0] else -1)
            intervals_by_chart.setdefault(chart_id, []).append(
                (min(parameters), max(parameters))
            )
    for chart_id in query.edge_chart_ids:
        if query.chart_source_key(1, chart_id) not in selected_keys:
            continue
        ordered = sorted(intervals_by_chart.get(chart_id, ()))
        if (
            not ordered
            or ordered[0][0] != 0
            or ordered[-1][1] != 1
            or any(left[1] != right[0] for left, right in zip(ordered, ordered[1:]))
        ):
            raise ValueError(
                f"Mapped output edges do not exhaust source curve chart {chart_id}."
            )
    if not chart_cell_ids:
        raise ValueError("Mapped PLC edge query produced no target chart witnesses.")
    target_reference_values = np.asarray(
        [
            [[float(value) for value in point] for point in row]
            for row in target_reference_rows
        ],
        dtype=np.float64,
    )
    source_parameter_values = np.asarray(
        [
            [[float(value) for value in point] for point in row]
            for row in source_parameter_rows
        ],
        dtype=np.float64,
    )
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=query.source_query_id,
        chart_dimension=1,
        reference_dimension=3,
        cell_global_ids=np.asarray(chart_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(chart_edge_ids, dtype=np.int64),
        local_entity_indices=np.asarray(chart_local_edges, dtype=np.int32),
        target_reference_vertices=target_reference_values,
        target_reference_rational_tokens=rational_chart_tokens(target_reference_rows),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrence_paths),
        source_parameters=source_parameter_values,
        source_parameter_rational_tokens=rational_chart_tokens(source_parameter_rows),
        orientations=np.asarray(orientations, dtype=np.int8),
    )


def plc_curve_chart_witness(
    source: NativePlcSource,
    construction: VolumeConstruction,
    geometry: CellGeometrySpec,
    entity_ids: np.ndarray,
    /,
) -> tuple[PiecewiseLinearBoundarySource, SourceBoundaryChartWitness]:
    """Bind exact recovered edge chains to their complete authored PLC intervals."""
    if not isinstance(construction.domain, PiecewiseLinearDomain):
        raise TypeError("Ordinary PLC curve charts require the original PWL domain.")
    query = PiecewiseLinearBoundarySource(
        _PlcCurveAuthority(source, construction.domain), 1, entity_ids=entity_ids
    )
    mesh = construction.mesh
    edges, _ = simplex_volume_entities(mesh)
    edge_row = {tuple(sorted(map(int, edge))): row for row, edge in enumerate(edges)}
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    cell_edges, _ = volume_cell_entity_rows(mesh, cell_ids)
    cells_by_id = {int(identifier): row for row, identifier in enumerate(cell_ids)}
    owners: dict[int, tuple[int, int, tuple[int, int], np.ndarray]] = {}
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        vertices = np.asarray(block.vertices, dtype=np.int64)
        for block_row, raw_cell_id in enumerate(np.asarray(block.global_ids)):
            cell_id = int(raw_cell_id)
            for local_edge, local_vertices in enumerate(topology.entities[1]):
                row = int(cell_edges[cells_by_id[cell_id], local_edge])
                first, second = local_vertices
                pair = (
                    int(vertices[block_row, first]),
                    int(vertices[block_row, second]),
                )
                candidate = (
                    cell_id,
                    local_edge,
                    pair,
                    np.asarray(topology.vertices, dtype=np.float64)[list(local_vertices)],
                )
                if row not in owners or candidate[:2] < owners[row][:2]:
                    owners[row] = candidate
    selected = {int(value) for value in query.entity_ids}
    intervals: dict[int, list[tuple[Fraction, Fraction]]] = {
        identifier: [] for identifier in selected
    }
    target_cells, target_edges, local_indices = [], [], []
    target_references, parameters = [], []
    chart_ids, entity_keys, occurrences, orientations = [], [], [], []
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    source_points = np.asarray(source.complex.vertices, dtype=np.float64)
    for segment, source_row in zip(
        construction.segments, construction.segment_source_edges, strict=True
    ):
        source_row = int(source_row)
        complex_row = int(construction.source_segment_complex_rows[source_row])
        identifier = int(source.complex.plc_edge_entity_ids[complex_row])
        if identifier not in selected:
            continue
        row = edge_row.get(tuple(sorted(map(int, segment))))
        if row is None or row not in owners:
            raise ValueError("A recovered PLC curve segment has no target cell edge.")
        cell_id, local_index, pair, reference = owners[row]
        endpoints = np.asarray(source.complex.plc_edge_vertices[complex_row], dtype=np.int64)
        first, last = (
            tuple(Fraction.from_float(float(value)) for value in source_points[index])
            for index in endpoints
        )
        direction = tuple(b - a for a, b in zip(first, last, strict=True))
        axis = next((axis for axis, value in enumerate(direction) if value), None)
        if axis is None:
            raise ValueError("A protected PLC curve has coincident endpoints.")
        source_parameters = []
        for vertex in pair:
            point = tuple(Fraction.from_float(float(value)) for value in points[vertex])
            parameter = (point[axis] - first[axis]) / direction[axis]
            if not 0 <= parameter <= 1 or any(
                point[i] != first[i] + parameter * direction[i] for i in range(3)
            ):
                raise ValueError("A recovered PLC curve vertex is not exactly on its source.")
            source_parameters.append(parameter)
        a, b = source_parameters
        if a == b:
            raise ValueError("A recovered PLC curve edge has zero source interval.")
        intervals[identifier].append((min(a, b), max(a, b)))
        chart = query.chart_id_for_entity_id(identifier)
        key, path = query.chart_source_key(1, chart)
        target_cells.append(cell_id)
        target_edges.append(int(edge_ids[row]))
        local_indices.append(local_index)
        target_references.append(
            tuple(tuple(Fraction.from_float(float(value)) for value in point) for point in reference)
        )
        parameters.append(((a,), (b,)))
        chart_ids.append(chart)
        entity_keys.append(key)
        occurrences.append(path)
        orientations.append(1 if b > a else -1)
    for identifier, pieces in intervals.items():
        ordered = sorted(pieces)
        if (
            not ordered
            or ordered[0][0] != 0
            or ordered[-1][1] != 1
            or any(left[1] != right[0] for left, right in zip(ordered, ordered[1:]))
        ):
            raise ValueError(f"Recovered PLC curve {identifier} has a gap or overlap.")
    target_values = np.asarray(target_references, dtype=np.float64)
    parameter_values = np.asarray(parameters, dtype=np.float64)
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=query.source_query_id,
        chart_dimension=1,
        reference_dimension=3,
        cell_global_ids=np.asarray(target_cells, dtype=np.int64),
        entity_global_ids=np.asarray(target_edges, dtype=np.int64),
        local_entity_indices=np.asarray(local_indices, dtype=np.int32),
        target_reference_vertices=target_values,
        target_reference_rational_tokens=rational_chart_tokens(tuple(target_references)),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(entity_keys),
        occurrence_paths=tuple(occurrences),
        source_parameters=parameter_values,
        source_parameter_rational_tokens=rational_chart_tokens(tuple(parameters)),
        orientations=np.asarray(orientations, dtype=np.int8),
    )
    return query, witness


@checked
def mapped_plc_reference_map(
    source: NativePlcSource, construction: VolumeConstruction, /
) -> ReferenceCellMapWitnesses | None:
    """Retain every generated tetrahedron's exact source-root chart map."""
    domain = source.mapped_domain
    if domain is None:
        return None
    if (
        not isinstance(construction.domain, type(domain))
        or construction.domain.domain_id != domain.domain_id
        or not construction.reference_chart_controls
        or len(construction.mesh.blocks) != 1
    ):
        raise ValueError(
            "Mapped PLC reference maps require their complete current root-chart controls."
        )
    block = construction.mesh.blocks[0]
    controls = tuple(construction.reference_chart_controls)
    target_ids = np.asarray(block.global_ids, dtype=np.int64)
    if (
        len(controls) != target_ids.size
        or tuple(control.target_cell_id for control in controls)
        != tuple(int(value) for value in target_ids.tolist())
        or any(
            control.target_cell_row != row
            or control.target_cell_kind != block.cell_kind
            or control.source_id != domain.source_id
            or control.source_revision != domain.source_revision
            or control.mapped_domain_id != domain.domain_id
            for row, control in enumerate(controls)
        )
        or len({control.source_cell_kind for control in controls}) != 1
    ):
        raise ValueError(
            "Mapped PLC reference-map rows changed target, source, root, or occurrence identity."
        )
    exact_vertices = tuple(control.source_reference_vertices for control in controls)
    numeric_vertices = np.asarray(
        [[[float(value) for value in point] for point in row] for row in exact_vertices],
        dtype=np.float64,
    )
    return ReferenceCellMapWitnesses(
        target_ids,
        np.asarray([control.source_cell_id for control in controls], dtype=np.int64),
        block.cell_kind,
        controls[0].source_cell_kind,
        numeric_vertices,
        exact_vertices,
        source_reference_chart_arguments=freeze_reference_chart_arguments(
            tuple(control.arguments for control in controls)
        ),
    )


__all__ = [
    "PhysicalPlcTetrahedra",
    "mapped_plc_boundary_chart_witness",
    "mapped_plc_edge_chart_witness",
    "mapped_plc_point_witness",
    "mapped_plc_reference_map",
    "simplex_volume_entities",
    "physical_plc_tetrahedra",
    "plc_cell_geometry",
]
