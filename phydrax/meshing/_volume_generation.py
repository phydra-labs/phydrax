#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native constrained tetrahedral construction from oriented PLC sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    PlcBoundaryPolicy as NativePlcBoundaryPolicy,
    PlcRecovery3D,
    recover_plc_3d,
    TetMesh3D,
    TetMeshArrays,
    TetMeshEditStatus,
    TetMeshSourceComplex,
    TetMeshSourceEvidence,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import CellValidityPolicy
from ..discretization._nested_reference import (
    exact_reference_chart_controls,
    ExactReferenceChartControl,
)
from ..geometry._domain_coverage import PiecewiseLinearDomain
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ._contracts import VolumeMeshingSpec
from ._plc_source import PiecewiseLinearComplex, PlcBoundaryPolicy
from ._sizing import (
    resolve_size_controls,
    SizeCombinationPolicy,
    SizeControlStrength,
    SizeFieldDomain,
    UniformSizeControl,
)
from ._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus


_VOLUME_VALIDITY_POLICY = CellValidityPolicy()


@final
class NativeVolumeSchedule(StrictModule, NonTrainableState):
    """Numerical quality aims for native PLC tetrahedralization.

    These are physical/geometric aims, not work or storage allowances. Native
    refinement and improvement return their actual unmet numerical criteria.
    Native improvement receives the CellValidityPolicy floor for its affine
    tetrahedron metric. High-order Bernstein validity is checked independently.
    """

    radius_edge_bound: float = eqx.field(static=True)
    minimum_dihedral_degrees: float = eqx.field(static=True)
    minimum_relative_determinant: float = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        radius_edge_bound: float = 2.0,
        minimum_dihedral_degrees: float = 10.0,
    ) -> None:
        radius_edge = finite_real_scalar(
            float(radius_edge_bound), "NativeVolumeSchedule.radius_edge_bound"
        )
        dihedral = finite_real_scalar(
            float(minimum_dihedral_degrees),
            "NativeVolumeSchedule.minimum_dihedral_degrees",
        )
        if radius_edge < 1.0:
            raise ValueError("radius_edge_bound must be at least one.")
        if not 0.0 <= dihedral < 70.0:
            raise ValueError("minimum_dihedral_degrees must lie in [0, 70).")
        self.radius_edge_bound = radius_edge
        self.minimum_dihedral_degrees = dihedral
        self.minimum_relative_determinant = (
            _VOLUME_VALIDITY_POLICY.relative_determinant_floor
        )
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-volume-schedule",
                "radius_edge_bound": radius_edge,
                "minimum_dihedral_degrees": dihedral,
                "minimum_relative_determinant": self.minimum_relative_determinant,
            }
        )


@dataclass(frozen=True, slots=True, eq=False)
class VolumeConstruction:
    """Unaccepted constrained-volume construction and source ancestry.

    The mesh and ancestry tables are construction data only. The central native
    acceptance owner independently audits geometry, source fidelity, region
    coverage, associations and compliance before making any accepted result.
    """

    mesh: CellMesh
    domain: PiecewiseLinearDomain | MappedReferenceDomain
    cell_regions: np.ndarray
    source_id: str
    source_revision: str
    complex_id: str
    vertex_ids: tuple[str, ...]
    vertex_occurrence_paths: tuple[tuple[str, ...], ...]
    facet_ids: tuple[str, ...]
    facet_occurrence_paths: tuple[tuple[str, ...], ...]
    segment_ids: tuple[str, ...]
    segment_occurrence_paths: tuple[tuple[str, ...], ...]
    plc_edge_ids: tuple[str, ...]
    plc_edge_occurrence_paths: tuple[tuple[str, ...], ...]
    source_face_ids: tuple[str, ...]
    source_face_occurrence_paths: tuple[tuple[str, ...], ...]
    region_ids: tuple[str, ...]
    region_occurrence_paths: tuple[tuple[str, ...], ...]
    boundary_policy: PlcBoundaryPolicy
    input_vertices: np.ndarray
    vertex_source_ids: tuple[str, ...]
    vertex_source_occurrence_paths: tuple[tuple[str, ...], ...]
    vertex_dimension: np.ndarray
    witness_strata: np.ndarray
    witness_entities: np.ndarray
    witness_parameters: np.ndarray
    witness_deviations: np.ndarray
    source_points: np.ndarray
    source_faces: np.ndarray
    source_face_polygons: np.ndarray
    source_face_facets: np.ndarray
    source_face_tolerances: np.ndarray
    plc_edges: np.ndarray
    plc_edge_source_rows: np.ndarray
    plc_edge_facet_offsets: np.ndarray
    plc_edge_facets: np.ndarray
    source_facet_region_sides: np.ndarray
    source_segment_vertices: np.ndarray
    source_segment_complex_rows: np.ndarray
    source_segment_labels: np.ndarray
    source_segment_tolerances: np.ndarray
    boundary_faces: np.ndarray
    face_sources: np.ndarray
    face_source_facets: np.ndarray
    face_source_ids: tuple[str, ...]
    face_source_occurrence_paths: tuple[tuple[str, ...], ...]
    face_orientations: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    segment_source_edges: np.ndarray
    segment_source_ids: tuple[str, ...]
    segment_source_occurrence_paths: tuple[tuple[str, ...], ...]
    source_evidence: TetMeshSourceEvidence
    refinement_status: MeshcoreStatus
    improvement_status: MeshcoreStatus
    refinement_unmet: tuple[tuple[str, str, int, float], ...]
    improvement_unmet: tuple[tuple[str, str, int, float], ...]
    stages: tuple[MeshingStageReport, ...]
    reference_chart_controls: tuple[ExactReferenceChartControl, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class _SourceTables:
    points: np.ndarray
    faces: np.ndarray
    face_polygons: np.ndarray
    face_facets: np.ndarray
    face_tolerances: np.ndarray
    plc_edges: np.ndarray
    plc_edge_source_rows: np.ndarray
    edge_facet_offsets: np.ndarray
    edge_facets: np.ndarray
    source_facet_region_sides: np.ndarray
    source_segment_vertices: np.ndarray
    source_segment_complex_rows: np.ndarray
    facet_labels: np.ndarray
    edge_labels: np.ndarray
    face_ids: np.ndarray
    face_group_rows: np.ndarray
    segment_ids: np.ndarray
    segment_group_rows: np.ndarray
    segment_tolerances: np.ndarray
    source_segment_labels: np.ndarray
    vertex_ids: tuple[str, ...]
    vertex_occurrence_paths: tuple[tuple[str, ...], ...]
    facet_ids: tuple[str, ...]
    facet_occurrence_paths: tuple[tuple[str, ...], ...]
    segment_entity_ids: tuple[str, ...]
    segment_occurrence_paths: tuple[tuple[str, ...], ...]
    plc_edge_ids: tuple[str, ...]
    plc_edge_occurrence_paths: tuple[tuple[str, ...], ...]
    source_face_ids: tuple[str, ...]
    source_face_occurrence_paths: tuple[tuple[str, ...], ...]
    source: TetMeshSourceComplex


def _source_face_ids(
    faces: np.ndarray,
    face_facets: np.ndarray,
    vertex_ids: tuple[str, ...],
    vertex_occurrence_paths: tuple[tuple[str, ...], ...],
    facet_ids: tuple[str, ...],
    facet_occurrence_paths: tuple[tuple[str, ...], ...],
    /,
) -> tuple[str, ...]:
    identifiers: list[str] = []
    for face, facet in zip(faces.tolist(), face_facets.tolist(), strict=True):
        vertices = tuple(
            (vertex_ids[int(vertex)], vertex_occurrence_paths[int(vertex)])
            for vertex in face
        )
        rotations = tuple(vertices[offset:] + vertices[:offset] for offset in range(3))
        identifiers.append(
            canonical_fingerprint(
                {
                    "kind": "plc-source-triangle",
                    "facet": (facet_ids[facet], facet_occurrence_paths[facet]),
                    "oriented_vertices": min(rotations),
                }
            )
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "PLC source triangulation contains ambiguous duplicate triangles."
        )
    return tuple(identifiers)


def _source_rows(
    entity_ids: np.ndarray,
    source_entity_ids: tuple[int, ...],
    name: str,
    /,
) -> np.ndarray:
    rows = {int(identifier): row for row, identifier in enumerate(source_entity_ids)}
    values = np.asarray(entity_ids, dtype=np.int64)
    if any(int(identifier) not in rows for identifier in values):
        raise ValueError(f"{name} scope contains an undeclared stable PLC entity ID.")
    return np.asarray([rows[int(identifier)] for identifier in values], dtype=np.int64)


def _source_tables(
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    recovery: PlcRecovery3D,
    /,
) -> _SourceTables:
    """Resolve raw PLC carrier witnesses to canonical identities and active source banks."""
    canonical_source_edges = np.asarray(
        complex_.plc_edge_vertices, dtype=np.int64
    ).reshape((-1, 2))
    vertex_identity_rows = tuple(
        (identifier, tuple(path))
        for identifier, path in zip(
            complex_.vertex_ids, complex_.vertex_occurrence_paths, strict=True
        )
    )
    source_edge_row_by_endpoints: dict[tuple[tuple[str, tuple[str, ...]], ...], int] = {}
    source_edge_identity_rows = tuple(
        (
            identifier,
            tuple(path),
        )
        for identifier, path in zip(
            complex_.plc_edge_ids,
            complex_.plc_edge_occurrence_paths,
            strict=True,
        )
    )
    if len(set(source_edge_identity_rows)) != len(source_edge_identity_rows):
        raise ValueError("Canonical PLC source-edge ID/occurrence rows are ambiguous.")
    for complex_row, edge in enumerate(canonical_source_edges.tolist()):
        key = tuple(sorted(vertex_identity_rows[int(vertex)] for vertex in edge))
        if key in source_edge_row_by_endpoints:
            raise ValueError(
                "Coincident PLC source edges lack a unique recovery identity mapping."
            )
        source_edge_row_by_endpoints[key] = complex_row

    facet_bounds = np.full((complex_.facet_count,), np.inf, dtype=np.float64)
    segment_bounds = np.full((canonical_source_edges.shape[0],), np.inf, dtype=np.float64)
    requested_segment_complex_rows: set[int] = set()
    for feature in specification.protected_features:
        identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
        match feature.scope.entity_dimension, feature.feature_kind.value:
            case 2, "surface" | "material_interface":
                rows = _source_rows(identifiers, complex_.facet_entity_ids, "facet")
                facet_bounds[rows] = np.minimum(
                    facet_bounds[rows], feature.maximum_deviation
                )
            case 1, "curve":
                rows = _source_rows(identifiers, complex_.plc_edge_entity_ids, "PLC edge")
                requested_segment_complex_rows.update(int(row) for row in rows.tolist())
                segment_bounds[rows] = np.minimum(
                    segment_bounds[rows], feature.maximum_deviation
                )
            case _:
                continue
    facet_bounds[~np.isfinite(facet_bounds)] = 0.0
    segment_bounds[~np.isfinite(segment_bounds)] = 0.0

    faces = np.asarray(recovery.input_triangles, dtype=np.int64)
    face_polygons = np.asarray(recovery.input_polygons, dtype=np.int64)
    face_facets = np.asarray(complex_.polygon_facets[face_polygons], dtype=np.int64)
    face_tolerances = facet_bounds[face_facets]
    face_occurrence_paths = tuple(
        complex_.facet_occurrence_paths[int(facet)] for facet in face_facets
    )
    source_face_ids = _source_face_ids(
        faces,
        face_facets,
        complex_.vertex_ids,
        complex_.vertex_occurrence_paths,
        complex_.facet_ids,
        complex_.facet_occurrence_paths,
    )
    source_face_row_by_identity: dict[tuple[str, tuple[str, ...]], int] = {}
    for row, (identifier, path) in enumerate(
        zip(source_face_ids, face_occurrence_paths, strict=True)
    ):
        key = (identifier, path)
        if key in source_face_row_by_identity:
            raise ValueError(
                "Canonical PLC source-face ID/occurrence rows are ambiguous."
            )
        source_face_row_by_identity[key] = row
    # Native kSegment rows are a subset of the canonical carrier bank. Resolve
    # them by stable endpoint identity; never equate row order or bank lengths.
    plc_edges = np.asarray(recovery.plc_edges, dtype=np.int64).reshape((-1, 2))
    carrier_edge_complex_rows = np.asarray(
        [
            source_edge_row_by_endpoints.get(
                tuple(sorted(vertex_identity_rows[int(vertex)] for vertex in edge)),
                -1,
            )
            for edge in plc_edges.tolist()
        ],
        dtype=np.int64,
    )
    if np.any(carrier_edge_complex_rows < 0):
        raise ValueError(
            "Every raw PLC edge must resolve to a canonical endpoint/ID/occurrence row."
        )

    source_edge_row_by_identity = {
        identity: row for row, identity in enumerate(source_edge_identity_rows)
    }
    explicit_segment_identity_rows = tuple(
        (identifier, tuple(path))
        for identifier, path in zip(
            complex_.segment_ids, complex_.segment_occurrence_paths, strict=True
        )
    )
    try:
        explicit_segment_complex_rows = np.asarray(
            [
                source_edge_row_by_identity[identity]
                for identity in explicit_segment_identity_rows
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "Every authored PLC segment must retain its canonical edge identity."
        ) from error
    explicit_segment_vertices = np.asarray(complex_.segments, dtype=np.int64).reshape(
        (-1, 2)
    )
    if explicit_segment_vertices.shape != (
        explicit_segment_complex_rows.size,
        2,
    ) or not np.array_equal(
        canonical_source_edges[explicit_segment_complex_rows],
        explicit_segment_vertices,
    ):
        raise ValueError("Authored PLC segment identities changed their endpoint rows.")
    required_active_rows = requested_segment_complex_rows | set(
        int(row) for row in explicit_segment_complex_rows.tolist()
    )
    if not required_active_rows.issubset(set(carrier_edge_complex_rows.tolist())):
        raise ValueError(
            "Native PLC recovery omitted an authored or requested source edge."
        )

    active_source_complex_rows = np.unique(carrier_edge_complex_rows)
    active_source_row_by_complex = {
        int(complex_row): row
        for row, complex_row in enumerate(active_source_complex_rows.tolist())
    }
    carrier_edge_source_rows = np.asarray(
        [
            active_source_row_by_complex[int(complex_row)]
            for complex_row in carrier_edge_complex_rows.tolist()
        ],
        dtype=np.int64,
    )
    source_segment_vertices = canonical_source_edges[active_source_complex_rows]
    source_segment_labels = np.asarray(complex_.plc_edge_entity_ids, dtype=np.int64)[
        active_source_complex_rows
    ]
    carrier_edge_labels = source_segment_labels[carrier_edge_source_rows]
    active_source_identity_rows = tuple(
        source_edge_identity_rows[int(complex_row)]
        for complex_row in active_source_complex_rows.tolist()
    )
    source_segment_entity_ids = tuple(
        identifier for identifier, _ in active_source_identity_rows
    )
    source_segment_occurrence_paths = tuple(
        path for _, path in active_source_identity_rows
    )

    face_rows_by_edge: dict[tuple[int, int], set[int]] = {}
    for face_row, face in enumerate(faces.tolist()):
        for offset, first in enumerate(face):
            second = int(face[(offset + 1) % 3])
            key = (min(int(first), second), max(int(first), second))
            face_rows_by_edge.setdefault(key, set()).add(face_row)

    edge_facet_rows: list[int] = []
    edge_offsets = np.empty((plc_edges.shape[0] + 1,), dtype=np.int64)
    edge_offsets[0] = 0
    carrier_edge_ids: list[str] = []
    carrier_edge_paths: list[tuple[str, ...]] = []
    for row, edge in enumerate(plc_edges.tolist()):
        key = tuple(sorted(edge))
        source_row = int(carrier_edge_source_rows[row])
        parent_face_rows = tuple(sorted(face_rows_by_edge.get(key, ())))
        parent_facets = tuple(
            sorted({int(face_facets[face_row]) for face_row in parent_face_rows})
        )
        complex_row = int(active_source_complex_rows[source_row])
        declared_facets = tuple(
            sorted(int(facet) for facet in complex_.plc_edge_facets[complex_row])
        )
        if parent_facets != declared_facets:
            raise ValueError(
                "Recovered source edges must preserve their exact original facet incidence."
            )
        carrier_edge_ids.append(source_segment_entity_ids[source_row])
        carrier_edge_paths.append(source_segment_occurrence_paths[source_row])
        edge_facet_rows.extend(parent_facets)
        edge_offsets[row + 1] = len(edge_facet_rows)

    source_segment_tolerances = segment_bounds[active_source_complex_rows].copy()
    for local_row, complex_row in enumerate(active_source_complex_rows.tolist()):
        facets = complex_.plc_edge_facets[int(complex_row)]
        if facets:
            source_segment_tolerances[local_row] = min(
                float(source_segment_tolerances[local_row]),
                float(np.min(facet_bounds[np.asarray(facets, dtype=np.int64)])),
            )
    source_facet_region_sides = np.asarray(complex_.facet_regions, dtype=np.int64).copy()
    facet_labels = np.asarray(complex_.facet_entity_ids, dtype=np.int64)
    face_ids = np.unique(facet_labels)
    face_group_rows = np.searchsorted(face_ids, facet_labels[face_facets]).astype(
        np.int32
    )
    segment_ids = np.unique(source_segment_labels)
    segment_group_rows = np.searchsorted(segment_ids, source_segment_labels).astype(
        np.int32
    )
    plc_edge_facets = np.asarray(edge_facet_rows, dtype=np.int64)
    witness_strata = np.asarray(recovery.witness_strata, dtype=np.int8)
    witness_entities = np.asarray(recovery.witness_entities, dtype=np.int32).copy()
    witness_parameters = np.asarray(recovery.witness_parameters, dtype=np.float64).copy()
    if (
        witness_strata.shape != (recovery.points.shape[0],)
        or witness_entities.shape != (recovery.points.shape[0],)
        or witness_parameters.shape != (recovery.points.shape[0], 2)
    ):
        raise ValueError("Native PLC recovery witness banks must align with its points.")
    for point, stratum in enumerate(witness_strata.tolist()):
        entity = int(witness_entities[point])
        if stratum == 1:
            if not 0 <= entity < carrier_edge_source_rows.size:
                raise ValueError(
                    "Recovered source-edge witness names an absent carrier edge."
                )
            source_row = int(carrier_edge_source_rows[entity])
            if source_row < 0:
                raise ValueError(
                    "Recovered 1D witness lacks a canonical original PLC-edge identity."
                )
            carrier_endpoints = tuple(int(value) for value in plc_edges[entity])
            source_endpoints = tuple(
                int(value) for value in source_segment_vertices[source_row]
            )
            if carrier_endpoints == source_endpoints:
                pass
            elif carrier_endpoints == (source_endpoints[1], source_endpoints[0]):
                witness_parameters[point, 0] = 1.0 - witness_parameters[point, 0]
            else:
                raise ValueError(
                    "Recovered source-edge witness changed canonical endpoints."
                )
            witness_entities[point] = source_row
        elif stratum == 2:
            if not 0 <= entity < faces.shape[0]:
                raise ValueError(
                    "Recovered facet witness must name an input source-triangle row."
                )
            identity = (source_face_ids[entity], face_occurrence_paths[entity])
            source_row = source_face_row_by_identity.get(identity)
            if source_row is None:
                raise ValueError(
                    "Recovered facet witness lost its canonical source identity."
                )
            witness_entities[point] = source_row
        elif stratum not in (0, 1, 2):
            raise ValueError(
                "Native PLC recovery returned an unknown source witness stratum."
            )

    source = TetMeshSourceComplex(
        np.asarray(complex_.vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
        face_group_rows,
        np.asarray(face_tolerances, dtype=np.float64),
        np.asarray(source_segment_vertices, dtype=np.int32),
        segment_group_rows,
        np.asarray(source_segment_tolerances, dtype=np.float64),
        witness_strata,
        witness_entities,
        witness_parameters,
        face_ids=face_ids,
        segment_ids=segment_ids,
    )
    return _SourceTables(
        points=np.asarray(complex_.vertices, dtype=np.float64),
        faces=faces,
        face_polygons=face_polygons,
        face_facets=face_facets,
        face_tolerances=face_tolerances,
        plc_edges=plc_edges,
        plc_edge_source_rows=carrier_edge_source_rows,
        edge_facet_offsets=edge_offsets,
        edge_facets=plc_edge_facets,
        source_facet_region_sides=source_facet_region_sides,
        source_segment_vertices=source_segment_vertices,
        source_segment_complex_rows=active_source_complex_rows,
        facet_labels=facet_labels,
        edge_labels=carrier_edge_labels,
        face_ids=face_ids,
        face_group_rows=face_group_rows,
        segment_ids=segment_ids,
        segment_group_rows=segment_group_rows,
        segment_tolerances=source_segment_tolerances,
        source_segment_labels=source_segment_labels,
        vertex_ids=complex_.vertex_ids,
        vertex_occurrence_paths=complex_.vertex_occurrence_paths,
        facet_ids=complex_.facet_ids,
        facet_occurrence_paths=complex_.facet_occurrence_paths,
        segment_entity_ids=source_segment_entity_ids,
        segment_occurrence_paths=source_segment_occurrence_paths,
        plc_edge_ids=tuple(carrier_edge_ids),
        plc_edge_occurrence_paths=tuple(carrier_edge_paths),
        source_face_ids=source_face_ids,
        source_face_occurrence_paths=face_occurrence_paths,
        source=source,
    )


def _region_seed_points(
    complex_: PiecewiseLinearComplex, specification: VolumeMeshingSpec, /
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    regions: list[int] = []
    for seed in specification.region_seeds:
        matches = [
            row
            for row, region_name in enumerate(complex_.region_ids)
            if region_name == seed.region_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"PLC region seed {seed.region_name!r} must resolve one declared region."
            )
        points.append((float(seed.point[0]), float(seed.point[1]), float(seed.point[2])))
        regions.append(matches[0])
    for seed in specification.hole_seeds:
        points.append((float(seed.point[0]), float(seed.point[1]), float(seed.point[2])))
        regions.append(-1)
    return (
        np.asarray(points, dtype=np.float64).reshape((-1, 3)),
        np.asarray(regions, dtype=np.int32),
    )


def declared_plc_domain(
    complex_: PiecewiseLinearComplex,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    source_id: str,
    /,
) -> PiecewiseLinearDomain:
    """Build the independent oriented source-domain partition from the PLC."""
    polygons = np.asarray(input_polygons, dtype=np.int64)
    triangles = np.asarray(input_triangles, dtype=np.int64)
    triangle_facets = np.asarray(complex_.polygon_facets[polygons], dtype=np.int64)
    incidence = np.asarray(complex_.facet_regions[triangle_facets], dtype=np.int64)
    exterior = incidence[:, 0] != incidence[:, 1]
    return PiecewiseLinearDomain(
        np.asarray(complex_.vertices, dtype=np.float64),
        triangles[exterior],
        incidence[exterior][:, ::-1],
        complex_.region_ids,
        region_occurrence_paths=complex_.region_occurrence_paths,
        source_id=source_id,
    )


def _vertex_facets(
    recovery: PlcRecovery3D, tables: _SourceTables, /
) -> tuple[tuple[int, ...], ...]:
    incident: list[set[int]] = [set() for _ in range(recovery.points.shape[0])]
    for triangle, facet in zip(
        tables.faces.tolist(), tables.face_facets.tolist(), strict=True
    ):
        for vertex in triangle:
            incident[vertex].add(facet)
    for point in range(tables.points.shape[0], recovery.points.shape[0]):
        stratum = int(recovery.witness_strata[point])
        entity = int(recovery.witness_entities[point])
        if stratum == 1 and 0 <= entity < tables.plc_edges.shape[0]:
            start, stop = tables.edge_facet_offsets[entity : entity + 2]
            incident[point].update(tables.edge_facets[start:stop].tolist())
        elif stratum == 2 and 0 <= entity < tables.face_facets.size:
            incident[point].add(int(tables.face_facets[entity]))
    return tuple(tuple(sorted(values)) for values in incident)


def _size_field(
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    recovery: PlcRecovery3D,
    tables: _SourceTables,
    /,
) -> np.ndarray | None:
    requested_controls = tuple(specification.size_controls)
    controls = tuple(
        control
        for control in requested_controls
        if isinstance(control, UniformSizeControl)
    )
    if len(controls) != len(requested_controls):
        raise ValueError("Native PLC size fields require uniform facet-scoped controls.")
    if not controls:
        return None
    controlled_facets = {
        row
        for control in controls
        for row in _source_rows(
            np.asarray(control.scope.entity_ids, dtype=np.int64),
            complex_.facet_entity_ids,
            "facet",
        )
    }
    sample_keys = tuple(
        sorted(
            {
                (int(vertex), int(facet))
                for triangle, facet in zip(
                    tables.faces.tolist(), tables.face_facets.tolist(), strict=True
                )
                if facet in controlled_facets
                for vertex in triangle
            }
        )
    )
    if not sample_keys:
        raise ValueError("Volume size controls must cover at least one PLC facet.")
    sample_rows = {key: row for row, key in enumerate(sample_keys)}
    sample_vertices = np.asarray([key[0] for key in sample_keys], dtype=np.int64)
    sample_facets = np.asarray(
        [complex_.facet_entity_ids[key[1]] for key in sample_keys],
        dtype=np.int64,
    )
    adjacency: set[tuple[int, int]] = set()
    for triangle, facet in zip(
        tables.faces.tolist(), tables.face_facets.tolist(), strict=True
    ):
        if facet not in controlled_facets:
            continue
        for offset in range(3):
            left = sample_rows[(triangle[offset], facet)]
            right = sample_rows[(triangle[(offset + 1) % 3], facet)]
            adjacency.add((min(left, right), max(left, right)))
    edges = np.asarray(sorted(adjacency), dtype=np.int64).reshape((-1, 2))
    field, report = resolve_size_controls(
        controls,
        tables.points[sample_vertices],
        sample_facets,
        SizeFieldDomain.EUCLIDEAN_VOLUME,
        adjacency=edges,
        combination=specification.size_combination,
    )
    values_by_sample = np.asarray(field.values, dtype=np.float64)
    winner_by_sample = report.winning_control_ids
    control_by_id = {control.control_id: control for control in controls}
    vertex_facets = _vertex_facets(recovery, tables)
    source_supports: list[list[tuple[int, float, str]]] = [
        [] for _ in range(recovery.points.shape[0])
    ]
    original_count = tables.points.shape[0]
    for vertex in range(original_count):
        for facet in vertex_facets[vertex]:
            row = sample_rows.get((vertex, facet))
            if row is not None:
                source_supports[vertex].append(
                    (facet, float(values_by_sample[row]), winner_by_sample[row])
                )
    for vertex in range(original_count, recovery.points.shape[0]):
        stratum = int(recovery.witness_strata[vertex])
        entity = int(recovery.witness_entities[vertex])
        parameters = np.asarray(recovery.witness_parameters[vertex], dtype=np.float64)
        if stratum == 1 and 0 <= entity < tables.plc_edges.shape[0]:
            start, stop = tables.edge_facet_offsets[entity : entity + 2]
            endpoints = tables.plc_edges[entity]
            parameter = float(parameters[0])
            for facet in tables.edge_facets[start:stop].tolist():
                left = sample_rows.get((int(endpoints[0]), int(facet)))
                right = sample_rows.get((int(endpoints[1]), int(facet)))
                if left is not None and right is not None:
                    value = (1.0 - parameter) * values_by_sample[
                        left
                    ] + parameter * values_by_sample[right]
                    winner = winner_by_sample[left]
                    source_supports[vertex].append((int(facet), float(value), winner))
        elif stratum == 2 and 0 <= entity < tables.faces.shape[0]:
            triangle = tables.faces[entity]
            facet = int(tables.face_facets[entity])
            weights = (1.0 - parameters[0] - parameters[1], parameters[0], parameters[1])
            rows = (
                sample_rows.get((int(tables.faces[entity, 0]), facet)),
                sample_rows.get((int(tables.faces[entity, 1]), facet)),
                sample_rows.get((int(tables.faces[entity, 2]), facet)),
            )
            if rows[0] is None or rows[1] is None or rows[2] is None:
                continue
            triangle_rows = (int(rows[0]), int(rows[1]), int(rows[2]))
            values = tuple(values_by_sample[row] for row in triangle_rows)
            value = sum(
                weight * item for weight, item in zip(weights, values, strict=True)
            )
            winner = winner_by_sample[triangle_rows[0]]
            source_supports[vertex].append((facet, float(value), winner))

    radius_factor = np.sqrt(6.0) / 4.0
    radii = np.zeros((recovery.points.shape[0],), dtype=np.float64)
    for vertex, supports in enumerate(source_supports):
        if not supports:
            continue
        active_facets = {facet for facet, _, _ in supports}
        active_controls = tuple(
            control
            for control in controls
            if any(
                complex_.facet_entity_ids[facet] in control.scope.entity_ids
                for facet in active_facets
            )
        )
        hard = tuple(
            control
            for control in active_controls
            if control.strength is SizeControlStrength.HARD
        )
        lower = max(
            (
                0.0 if control.minimum_size is None else control.minimum_size
                for control in hard
            ),
            default=0.0,
        )
        upper = min(
            (
                np.inf if control.maximum_size is None else control.maximum_size
                for control in hard
            ),
            default=np.inf,
        )
        if lower > upper:
            raise ValueError(
                "Hard PLC size intervals conflict at a shared source vertex."
            )
        candidates = [
            (value, control_by_id[control_id])
            for _, value, control_id in supports
            if not hard or control_by_id[control_id].strength is SizeControlStrength.HARD
        ]
        if not candidates:
            continue
        if specification.size_combination is SizeCombinationPolicy.EXPLICIT_PRIORITY:
            priority = max(control.priority for _, control in candidates)
            candidates = [
                (value, control)
                for value, control in candidates
                if control.priority == priority
            ]
        candidate_radii = [
            value
            * (radius_factor if control.strength is SizeControlStrength.HARD else 1.0)
            for value, control in candidates
        ]
        radii[vertex] = min(
            max(min(candidate_radii), lower * radius_factor),
            0.5 * upper,
        )
    return radii


def _unmet_rows(state: TetMesh3D, /) -> tuple[tuple[str, str, int, float], ...]:
    unmet = state.unmet()
    return tuple(
        (
            str(int(criterion)),
            str(int(reason)),
            int(cell),
            float(value),
        )
        for cell, criterion, reason, value in zip(
            np.asarray(unmet.cell_ids),
            np.asarray(unmet.criteria),
            np.asarray(unmet.reasons),
            np.asarray(unmet.values),
            strict=True,
        )
    )


def _make_construction(
    complex_: PiecewiseLinearComplex,
    source_id: str,
    source_revision: str,
    domain: PiecewiseLinearDomain | MappedReferenceDomain,
    tables: _SourceTables,
    arrays: TetMeshArrays,
    evidence: TetMeshSourceEvidence,
    refinement_status: MeshcoreStatus,
    improvement_status: MeshcoreStatus,
    refinement_unmet: tuple[tuple[str, str, int, float], ...],
    improvement_unmet: tuple[tuple[str, str, int, float], ...],
) -> VolumeConstruction:
    cells = np.asarray(arrays.tetrahedra, dtype=np.int64)
    used = np.unique(
        np.concatenate(
            (
                cells.reshape(-1),
                np.asarray(arrays.faces, dtype=np.int64).reshape(-1),
                np.asarray(arrays.segments, dtype=np.int64).reshape(-1),
            )
        )
    )
    remap = np.full((arrays.points.shape[0],), -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    tetrahedra = remap[cells]
    points = np.asarray(arrays.points, dtype=np.float64)[used]
    boundary_faces = remap[np.asarray(arrays.faces, dtype=np.int64)]
    segments = remap[np.asarray(arrays.segments, dtype=np.int64)]
    if np.any(boundary_faces < 0) or np.any(segments < 0):
        raise ValueError(
            "Native constrained PLC entities must remain incident to the filled mesh."
        )
    cell_regions = np.asarray(arrays.tetrahedron_regions, dtype=np.int64).copy()
    mesh = CellMesh.from_simplices(
        points,
        tetrahedra,
        dimension=3,
        block_name="native-plc-tetrahedra",
        vertex_global_ids=used,
    )
    reference_chart_controls: tuple[ExactReferenceChartControl, ...] = ()
    if isinstance(domain, MappedReferenceDomain):
        if (
            complex_.region_ids != domain.reference_domain.region_ids
            or complex_.region_occurrence_paths
            != domain.reference_domain.region_occurrence_paths
        ):
            raise ValueError(
                "Mapped PLC regions must retain their declared root identities and occurrences."
            )
        target_geometry = CellGeometrySpec.affine(mesh)
        reference_chart_controls = exact_reference_chart_controls(
            domain, mesh, target_geometry, cell_regions
        )
        if len(reference_chart_controls) != cell_regions.size:
            raise ValueError(
                "Every generated mapped PLC tetrahedron needs one exact root chart."
            )
        target_cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
        for row, control in enumerate(reference_chart_controls):
            if (
                control.target_block_index != 0
                or control.target_cell_row != row
                or control.target_cell_id != int(target_cell_ids[row])
                or control.source_id != source_id
                or control.source_revision != source_revision
                or control.mapped_domain_id != domain.domain_id
                or control.source_region_id != complex_.region_ids[int(cell_regions[row])]
                or control.source_occurrence_path
                != complex_.region_occurrence_paths[int(cell_regions[row])]
            ):
                raise ValueError(
                    "Exact generated-tet chart controls changed source, cell, region, or occurrence identity."
                )
    input_vertices = np.where(used < tables.points.shape[0], used, -1).astype(np.int64)
    vertex_dimension = np.asarray(arrays.vertex_dimensions, dtype=np.int64)[used].copy()
    vertex_dimension[input_vertices >= 0] = 0
    witness_strata = np.asarray(evidence.witness_strata, dtype=np.int8)[used].copy()
    witness_entities = np.asarray(evidence.witness_entities, dtype=np.int32)[used].copy()
    witness_parameters = np.asarray(evidence.witness_parameters, dtype=np.float64)[
        used
    ].copy()
    witness_deviations = np.asarray(evidence.witness_deviations, dtype=np.float64)[
        used
    ].copy()
    vertex_regions: list[set[int]] = [set() for _ in range(points.shape[0])]
    for cell, region in zip(tetrahedra.tolist(), cell_regions.tolist(), strict=True):
        for vertex in cell:
            vertex_regions[vertex].add(region)
    vertex_source_ids: list[str] = []
    vertex_source_occurrence_paths: list[tuple[str, ...]] = []
    for vertex, (source_vertex, stratum, entity) in enumerate(
        zip(
            input_vertices.tolist(),
            witness_strata.tolist(),
            witness_entities.tolist(),
            strict=True,
        )
    ):
        if source_vertex >= 0:
            vertex_source_ids.append(tables.vertex_ids[source_vertex])
            vertex_source_occurrence_paths.append(
                tables.vertex_occurrence_paths[source_vertex]
            )
        elif stratum == 1 and 0 <= entity < len(tables.segment_entity_ids):
            vertex_source_ids.append(tables.segment_entity_ids[entity])
            vertex_source_occurrence_paths.append(tables.segment_occurrence_paths[entity])
        elif stratum == 2 and 0 <= entity < tables.face_facets.size:
            facet = int(tables.face_facets[entity])
            vertex_source_ids.append(tables.facet_ids[facet])
            vertex_source_occurrence_paths.append(tables.facet_occurrence_paths[facet])
        elif stratum == 0 and len(vertex_regions[vertex]) == 1:
            region = next(iter(vertex_regions[vertex]))
            vertex_source_ids.append(complex_.region_ids[region])
            vertex_source_occurrence_paths.append(
                complex_.region_occurrence_paths[region]
            )
        else:
            raise ValueError(
                "Native vertex source witnesses must identify one PLC stratum or region."
            )
    facet_row_by_label = {
        int(label): row for row, label in enumerate(tables.facet_labels.tolist())
    }
    source_segment_row_by_label = {
        int(label): row for row, label in enumerate(tables.source_segment_labels.tolist())
    }
    face_sources = np.asarray(arrays.face_sources, dtype=np.int64).copy()
    face_source_facets = np.asarray(
        [facet_row_by_label[int(label)] for label in face_sources], dtype=np.int64
    )
    face_source_ids = tuple(tables.facet_ids[row] for row in face_source_facets.tolist())
    face_source_occurrence_paths = tuple(
        tables.facet_occurrence_paths[row] for row in face_source_facets.tolist()
    )
    segment_sources = np.asarray(arrays.segment_sources, dtype=np.int64).copy()
    try:
        segment_source_edges = np.asarray(
            [source_segment_row_by_label[int(label)] for label in segment_sources],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "TetMesh returned an undeclared canonical source segment."
        ) from error
    segment_source_ids = tuple(
        tables.segment_entity_ids[row] for row in segment_source_edges.tolist()
    )
    segment_source_occurrence_paths = tuple(
        tables.segment_occurrence_paths[row] for row in segment_source_edges.tolist()
    )
    face_orientations = np.asarray(arrays.facet_orientations, dtype=np.int8).copy()
    arrays_out = (
        cell_regions,
        input_vertices,
        vertex_dimension,
        witness_strata,
        witness_entities,
        witness_parameters,
        witness_deviations,
        boundary_faces,
        face_sources,
        face_source_facets,
        face_orientations,
        segments,
        segment_sources,
        segment_source_edges,
        tables.points,
        tables.faces,
        tables.face_polygons,
        tables.face_facets,
        tables.face_tolerances,
        tables.plc_edges,
        tables.plc_edge_source_rows,
        tables.edge_facet_offsets,
        tables.edge_facets,
        tables.source_facet_region_sides,
        tables.source_segment_vertices,
        tables.source_segment_complex_rows,
        tables.source_segment_labels,
        tables.segment_tolerances,
    )
    for value in arrays_out:
        value.setflags(write=False)
    stages = (
        MeshingStageReport(
            MeshingStageKind.VOLUME_FILL,
            MeshingStageStatus.PASSED,
            input_ids=(complex_.complex_id,),
            output_ids=(mesh.mesh_id,),
            created_count=tetrahedra.shape[0],
        ),
        MeshingStageReport(
            MeshingStageKind.OPTIMIZATION,
            MeshingStageStatus.WARNING
            if refinement_status is not MeshcoreStatus.OK
            or improvement_status is not MeshcoreStatus.OK
            or refinement_unmet
            or improvement_unmet
            else MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=(mesh.mesh_id,),
        ),
    )
    return VolumeConstruction(
        mesh=mesh,
        domain=domain,
        cell_regions=cell_regions,
        source_id=source_id,
        source_revision=source_revision,
        complex_id=complex_.complex_id,
        vertex_ids=tables.vertex_ids,
        facet_ids=tables.facet_ids,
        vertex_occurrence_paths=tables.vertex_occurrence_paths,
        facet_occurrence_paths=tables.facet_occurrence_paths,
        segment_occurrence_paths=tables.segment_occurrence_paths,
        segment_ids=tables.segment_entity_ids,
        plc_edge_ids=tables.plc_edge_ids,
        source_face_ids=tables.source_face_ids,
        plc_edge_occurrence_paths=tables.plc_edge_occurrence_paths,
        source_face_occurrence_paths=tables.source_face_occurrence_paths,
        region_ids=complex_.region_ids,
        region_occurrence_paths=complex_.region_occurrence_paths,
        boundary_policy=complex_.boundary,
        input_vertices=input_vertices,
        vertex_source_ids=tuple(vertex_source_ids),
        vertex_source_occurrence_paths=tuple(vertex_source_occurrence_paths),
        vertex_dimension=vertex_dimension,
        witness_strata=witness_strata,
        witness_entities=witness_entities,
        witness_parameters=witness_parameters,
        witness_deviations=witness_deviations,
        source_points=tables.points,
        source_faces=tables.faces,
        source_face_polygons=tables.face_polygons,
        source_face_facets=tables.face_facets,
        source_face_tolerances=tables.face_tolerances,
        plc_edges=tables.plc_edges,
        plc_edge_source_rows=tables.plc_edge_source_rows,
        source_facet_region_sides=tables.source_facet_region_sides,
        source_segment_vertices=tables.source_segment_vertices,
        source_segment_complex_rows=tables.source_segment_complex_rows,
        plc_edge_facet_offsets=tables.edge_facet_offsets,
        plc_edge_facets=tables.edge_facets,
        source_segment_labels=tables.source_segment_labels,
        source_segment_tolerances=tables.segment_tolerances,
        boundary_faces=boundary_faces,
        face_sources=face_sources,
        face_source_facets=face_source_facets,
        face_source_ids=face_source_ids,
        face_orientations=face_orientations,
        face_source_occurrence_paths=face_source_occurrence_paths,
        segments=segments,
        segment_sources=segment_sources,
        segment_source_edges=segment_source_edges,
        segment_source_ids=segment_source_ids,
        source_evidence=evidence,
        segment_source_occurrence_paths=segment_source_occurrence_paths,
        refinement_status=refinement_status,
        improvement_status=improvement_status,
        refinement_unmet=refinement_unmet,
        improvement_unmet=improvement_unmet,
        stages=stages,
        reference_chart_controls=reference_chart_controls,
    )


def generate_plc_volume(
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    source_id: str,
    source_revision: str,
    source_map_id: str,
    frame_id: str,
    source_domain: MappedReferenceDomain | None = None,
) -> VolumeConstruction:
    """Recover constraints, classify every region, and refine the PLC fill."""
    canonical_source_edges = np.asarray(
        complex_.plc_edge_vertices, dtype=np.int64
    ).reshape((-1, 2))
    explicit_segment_vertices = np.asarray(complex_.segments, dtype=np.int64).reshape(
        (-1, 2)
    )
    explicit_segment_identity_rows = tuple(
        (identifier, tuple(path))
        for identifier, path in zip(
            complex_.segment_ids, complex_.segment_occurrence_paths, strict=True
        )
    )
    if explicit_segment_vertices.shape != (len(explicit_segment_identity_rows), 2):
        raise ValueError(
            "Explicit PLC segment identities and endpoint rows are misaligned."
        )
    canonical_edge_row_by_identity = {
        (identifier, tuple(path)): row
        for row, (identifier, path) in enumerate(
            zip(
                complex_.plc_edge_ids,
                complex_.plc_edge_occurrence_paths,
                strict=True,
            )
        )
    }
    try:
        explicit_segment_rows = np.asarray(
            [
                canonical_edge_row_by_identity[identity]
                for identity in explicit_segment_identity_rows
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "Every explicit PLC segment must retain its canonical edge identity."
        ) from error
    if not np.array_equal(
        canonical_source_edges[explicit_segment_rows], explicit_segment_vertices
    ):
        raise ValueError("Explicit PLC segment identities changed their endpoint rows.")

    seeds, seed_regions = _region_seed_points(complex_, specification)
    facet_tolerances = np.full((complex_.facet_count,), np.inf, dtype=np.float64)
    segment_tolerances = np.full(
        (canonical_source_edges.shape[0],), np.inf, dtype=np.float64
    )
    active_segment_rows = {int(row) for row in explicit_segment_rows.tolist()}
    for feature in specification.protected_features:
        identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
        match feature.scope.entity_dimension, feature.feature_kind.value:
            case 2, "surface" | "material_interface":
                rows = _source_rows(identifiers, complex_.facet_entity_ids, "facet")
                facet_tolerances[rows] = np.minimum(
                    facet_tolerances[rows], feature.maximum_deviation
                )
            case 1, "curve":
                rows = _source_rows(identifiers, complex_.plc_edge_entity_ids, "PLC edge")
                active_segment_rows.update(int(row) for row in rows.tolist())
                segment_tolerances[rows] = np.minimum(
                    segment_tolerances[rows], feature.maximum_deviation
                )
            case _:
                continue
    facet_tolerances[~np.isfinite(facet_tolerances)] = 0.0
    segment_tolerances[~np.isfinite(segment_tolerances)] = 0.0
    active_segment_complex_rows = np.asarray(sorted(active_segment_rows), dtype=np.int64)
    active_segment_vertices = canonical_source_edges[active_segment_complex_rows]
    active_segment_tolerances = segment_tolerances[active_segment_complex_rows]
    native_boundary_policy = (
        NativePlcBoundaryPolicy.FIXED
        if complex_.boundary == "fixed"
        else NativePlcBoundaryPolicy.CONFORMING
    )
    # Native source preparation owns polygon loops and omits grouped coplanar
    # internal diagonals. Only authored segments and requested curve features
    # are passed as explicit 1D constraints.
    recovery = recover_plc_3d(
        complex_.vertices,
        complex_.polygon_offsets,
        complex_.polygon_vertices,
        complex_.polygon_facets,
        complex_.facet_regions,
        segments=active_segment_vertices,
        seeds=seeds,
        seed_regions=seed_regions,
        facet_tolerances=facet_tolerances,
        segment_tolerances=active_segment_tolerances,
        boundary_policy=native_boundary_policy,
    )
    tables = _source_tables(complex_, specification, recovery)
    if complex_.geometry is not None and source_domain is None:
        raise ValueError(
            "Mapped PLC geometry requires its independently declared MappedReferenceDomain."
        )
    if source_domain is None:
        domain: PiecewiseLinearDomain | MappedReferenceDomain = declared_plc_domain(
            complex_, tables.faces, tables.face_polygons, source_id
        )
    else:
        if (
            source_domain.source_id != source_id
            or source_domain.source_revision != source_revision
            or source_domain.reference_domain.region_ids != complex_.region_ids
            or source_domain.reference_domain.region_occurrence_paths
            != complex_.region_occurrence_paths
        ):
            raise ValueError(
                "Mapped PLC domain must bind the source identity, revision and declared regions."
            )
        domain = source_domain
    face_rows = np.asarray(recovery.face_sources, dtype=np.int64)
    carrier_rows = np.asarray(recovery.segment_sources, dtype=np.int64)
    if np.any((face_rows < 0) | (face_rows >= tables.facet_labels.size)):
        raise ValueError(
            "Recovered constrained facets must retain declared PLC ancestry."
        )
    if np.any((carrier_rows < 0) | (carrier_rows >= tables.plc_edge_source_rows.size)):
        raise ValueError(
            "Recovered constrained segments must retain carrier-edge ancestry."
        )
    carrier_source_rows = tables.plc_edge_source_rows[carrier_rows]
    if np.any(carrier_source_rows < 0):
        raise ValueError(
            "Recovered constrained segments must retain source-edge ancestry."
        )
    recovered_source_rows = np.unique(carrier_source_rows)
    if not np.array_equal(
        recovered_source_rows,
        np.arange(tables.source_segment_vertices.shape[0], dtype=np.int64),
    ):
        raise ValueError("Native PLC recovery dropped an active source edge chain.")
    face_sources = tables.facet_labels[face_rows]
    segments = np.asarray(recovery.segments, dtype=np.int64)
    segment_sources = tables.source_segment_labels[carrier_source_rows]
    sizes = _size_field(complex_, specification, recovery, tables)
    state = TetMesh3D(
        recovery.points,
        recovery.tetrahedra,
        recovery.tetrahedron_regions,
        recovery.faces,
        face_sources,
        segments,
        segment_sources,
        boundary_policy=native_boundary_policy,
        face_orientations=np.asarray(recovery.face_orientations, dtype=np.int8),
        protection_radii=recovery.protection_radii,
        source=tables.source,
        source_id=source_id,
        source_revision=source_revision,
        source_map_id=source_map_id,
        frame_id=frame_id,
    )
    try:
        refinement = state.refine(
            radius_edge_bound=schedule.radius_edge_bound,
            sizes=sizes,
        )
        if refinement.edit_status is not TetMeshEditStatus.APPLIED:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"tet_mesh_refine edit status was {refinement.edit_status.name}.",
            )
        if refinement.status not in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED):
            raise MeshcoreError(
                refinement.status,
                "tet_mesh_refine could not complete its native pass.",
            )
        refinement_unmet = _unmet_rows(state)
        improvement = state.improve(
            radius_edge_bound=schedule.radius_edge_bound,
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=schedule.minimum_relative_determinant,
        )
        if improvement.edit_status is not TetMeshEditStatus.APPLIED:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"tet_mesh_improve_quality edit status was {improvement.edit_status.name}.",
            )
        if improvement.status not in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED):
            raise MeshcoreError(
                improvement.status,
                "tet_mesh_improve_quality could not complete its native pass.",
            )
        improvement_unmet = _unmet_rows(state)
        arrays = state.arrays()
        evidence = state.source_evidence()
    finally:
        state.close()
    return _make_construction(
        complex_,
        source_id,
        source_revision,
        domain,
        tables,
        arrays,
        evidence,
        refinement.status,
        improvement.status,
        refinement_unmet,
        improvement_unmet,
    )


__all__ = [
    "NativeVolumeSchedule",
    "VolumeConstruction",
    "declared_plc_domain",
    "generate_plc_volume",
]
