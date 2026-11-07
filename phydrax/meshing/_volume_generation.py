#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native constrained tetrahedral construction from oriented PLC sources."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from typing import final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._meshcore import (
    exact_orient3d,
    MeshcoreError,
    MeshcoreStatus,
    PlcBoundaryPolicy as NativePlcBoundaryPolicy,
    PlcRecovery3D,
    PlcSuppressionSearch,
    recover_plc_3d,
    TetMesh3D,
    TetMeshArrays,
    TetMeshCandidateQuality,
    TetMeshEditStatus,
    TetMeshIdentity,
    TetMeshQuality,
    TetMeshSourceComplex,
    TetMeshSourceEvidence,
    TetMeshUnmet,
    TetMeshUnmetReason,
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
from ..geometry._affine_plc_domain import AffinePlcDomain
from ..geometry._compartments import CompartmentMeshingSource
from ..geometry._domain_coverage import PiecewiseLinearDomain
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..linalg._exact import solve_exact
from ..typing import parse
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
    ``radius_edge_bound=None`` omits only that shape target; zero is not a
    valid explicit bound. A zero dihedral aim similarly requests no angle
    threshold. Both retain numerical validity and representability checks.

    ``shape_strength`` selects how acceptance treats the two shape targets
    (radius-edge bound and minimum dihedral). ``HARD`` gates every unmet
    shape row except the approved input-forced exemptions. ``SOFT`` refines
    and improves toward the identical targets but publishes each unmet shape
    row (criterion, achieved value, native reason, cell) as compliance
    evidence instead of an issue. Validity, determinant, orientation,
    representability, size, fidelity and coverage stay hard for both. Soft
    strength enters ``schedule_id``; hard schedules keep their identity.
    """

    radius_edge_bound: float | None = eqx.field(static=True)
    minimum_dihedral_degrees: float = eqx.field(static=True)
    minimum_relative_determinant: float = eqx.field(static=True)
    shape_strength: SizeControlStrength = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        radius_edge_bound: float | None = 2.0,
        minimum_dihedral_degrees: float = 10.0,
        shape_strength: SizeControlStrength = SizeControlStrength.HARD,
    ) -> None:
        strength = parse(
            shape_strength, SizeControlStrength, "NativeVolumeSchedule.shape_strength"
        )
        radius_edge = (
            None
            if radius_edge_bound is None
            else finite_real_scalar(
                float(radius_edge_bound), "NativeVolumeSchedule.radius_edge_bound"
            )
        )
        dihedral = finite_real_scalar(
            float(minimum_dihedral_degrees),
            "NativeVolumeSchedule.minimum_dihedral_degrees",
        )
        if radius_edge is not None and radius_edge < 1.0:
            raise ValueError("radius_edge_bound must be at least one.")
        if not 0.0 <= dihedral < 70.0:
            raise ValueError("minimum_dihedral_degrees must lie in [0, 70).")
        self.radius_edge_bound = radius_edge
        self.minimum_dihedral_degrees = dihedral
        self.minimum_relative_determinant = (
            _VOLUME_VALIDITY_POLICY.relative_determinant_floor
        )
        self.shape_strength = strength
        identity: dict[str, object] = {
            "kind": "native-volume-schedule",
            "radius_edge_bound": radius_edge,
            "minimum_dihedral_degrees": dihedral,
            "minimum_relative_determinant": self.minimum_relative_determinant,
        }
        if not _shape_targets_gated(self):
            identity["shape_strength"] = strength.value
        self.schedule_id = canonical_fingerprint(identity)


def _shape_targets_gated(schedule: NativeVolumeSchedule, /) -> bool:
    """Whether unmet radius-edge and dihedral rows are acceptance issues."""
    match schedule.shape_strength:
        case SizeControlStrength.HARD:
            return True
        case SizeControlStrength.SOFT:
            return False
        case invalid:
            raise ValueError(f"Unknown native volume shape strength {invalid!r}.")


@dataclass(frozen=True, slots=True, eq=False)
class VolumeConstruction:
    """Unaccepted constrained-volume construction and source ancestry.

    The mesh and ancestry tables are construction data only. The central native
    acceptance owner independently audits geometry, source fidelity, region
    coverage, associations and compliance before making any accepted result.
    """

    mesh: CellMesh
    domain: PiecewiseLinearDomain | MappedReferenceDomain | AffinePlcDomain
    cell_regions: np.ndarray
    # Native TetMesh cell IDs aligned with mesh rows; quality rows name these.
    native_cell_ids: np.ndarray
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
    refinement_inserted_vertices: int
    improvement_inserted_vertices: int
    improvement_collapsed_vertices: int
    improvement_relocated_vertices: int
    final_quality_identity: TetMeshIdentity | None
    final_quality_unmet: tuple[tuple[str, str, int, float], ...]
    stages: tuple[MeshingStageReport, ...]
    # With root fields, witness_* and source_evidence describe historical
    # execution coordinates, not the normalized candidate.
    root_vertex_stencils: tuple[tuple[tuple[int, Fraction], ...] | None, ...] = ()
    root_exact_points: tuple[tuple[Fraction, Fraction, Fraction] | None, ...] = ()
    candidate_quality_id: str | None = None
    historical_quality_identity: TetMeshIdentity | None = None
    mapped_quality_predecessor_identity: TetMeshIdentity | None = None
    boundary_quality_repair_status: MeshcoreStatus | None = None
    boundary_quality_repair_unmet: tuple[tuple[str, str, int, float], ...] = ()
    boundary_quality_predecessor_identity: TetMeshIdentity | None = None
    reference_chart_controls: tuple[ExactReferenceChartControl, ...] = ()

    @property
    def final_quality_schedule_unmet(self) -> tuple[tuple[str, str, int, float], ...]:
        """Geometric residuals; native SIZE rows are sizing preferences.

        The retained field combines controls of both strengths. Its residuals
        cannot be attributed to individual hard controls; consumers evaluate
        those controls independently with their declared compliance policies.
        """
        return _schedule_quality_rows(self.final_quality_unmet)


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


def _authored_coplanar_edge_rows(complex_: PiecewiseLinearComplex, /) -> np.ndarray:
    """Canonical rows of shared edges between coplanar polygons of authored facets.

    Native source preparation merges exactly coplanar adjacent polygons of one
    facet and drops their shared edge. The polygon edges of a discrete curved
    surface are authored, so a coplanar pair inside it (a symmetric trapezoid
    on a surface of revolution) must keep its diagonal instead of letting
    recovery choose the other one. A complex declaring ``facet_planes`` lets
    only its declared-plane facets merge; without declarations a facet may
    merge iff all its adjacent polygon pairs are exactly coplanar.
    """
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    loops = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    polygon_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
    sizes = np.diff(offsets)
    positions = np.arange(loops.size, dtype=np.int64)
    polygons = np.repeat(np.arange(sizes.size, dtype=np.int64), sizes)
    following = positions + 1
    wrapped = following == offsets[polygons + 1]
    following[wrapped] = offsets[polygons[wrapped]]
    first, second = loops[positions], loops[following]
    keys = np.minimum(first, second) * loops.size + np.maximum(first, second)
    order = np.argsort(keys, kind="stable")
    _, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    shared = starts[counts == 2]
    use, other = order[shared], order[shared + 1]
    adjacent = (
        (first[use] == second[other])
        & (second[use] == first[other])
        & (polygon_facets[polygons[use]] == polygon_facets[polygons[other]])
    )
    use, other = use[adjacent], other[adjacent]
    if use.size == 0:
        return np.zeros((0,), dtype=np.int64)
    left, right = polygons[use], polygons[other]
    # Two polygons sharing edge ab are coplanar iff every vertex pair (c, d)
    # of them satisfies orient3d(a, b, c, d) == 0: a nondegenerate polygon
    # owns a vertex off line ab, which fixes the common plane.
    combinations = sizes[left] * sizes[right]
    pair_rows = np.repeat(np.arange(use.size, dtype=np.int64), combinations)
    local = np.arange(pair_rows.size, dtype=np.int64) - np.repeat(
        np.cumsum(combinations) - combinations, combinations
    )
    right_sizes = sizes[right][pair_rows]
    vertices = np.asarray(complex_.vertices, dtype=np.float64)
    signs = exact_orient3d(
        vertices[first[use][pair_rows]],
        vertices[second[use][pair_rows]],
        vertices[loops[offsets[left][pair_rows] + local // right_sizes]],
        vertices[loops[offsets[right][pair_rows] + local % right_sizes]],
    )
    coplanar = np.bincount(pair_rows, weights=signs != 0, minlength=use.size) == 0
    pair_facets = polygon_facets[left]
    if complex_.facet_planes is None:
        mergeable = ~np.isin(pair_facets, pair_facets[~coplanar])
    else:
        declared = ~np.all(np.isnan(complex_.facet_planes), axis=(1, 2))
        mergeable = declared[pair_facets]
    locked = coplanar & ~mergeable
    if not np.any(locked):
        return np.zeros((0,), dtype=np.int64)
    row_by_key = {
        (min(edge), max(edge)): row for row, edge in enumerate(complex_.plc_edge_vertices)
    }
    return np.asarray(
        sorted(
            row_by_key[(min(edge), max(edge))]
            for edge in zip(
                first[use][locked].tolist(), second[use][locked].tolist(), strict=True
            )
        ),
        dtype=np.int64,
    )


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

    # Each source face row carries its facet's declared exact plane (NaN rows
    # undeclared); the TetMesh witnesses facet interior points on it (stratum 3).
    if complex_.facet_planes is None:
        face_planes = np.full((faces.shape[0], 3, 3), np.nan, dtype=np.float64)
        face_plane_tolerances = np.zeros((faces.shape[0],), dtype=np.float64)
    else:
        face_planes = np.asarray(complex_.facet_planes, dtype=np.float64)[face_facets]
        face_plane_tolerances = np.asarray(
            complex_.facet_plane_tolerances, dtype=np.float64
        )[face_facets]
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
        face_planes=face_planes,
        face_plane_tolerances=face_plane_tolerances,
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
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    /,
    *,
    affine: np.ndarray | None = None,
    affine_source: CompartmentMeshingSource | None = None,
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
    result = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if affine is not None and result.size:
        matrix = tuple(
            tuple(Fraction.from_float(float(value)) for value in row)
            for row in affine[:3, :3]
        )
        offset = tuple(Fraction.from_float(float(value)) for value in affine[:3, 3])
        exact_points = tuple(
            tuple(
                row[0]
                for row in solve_exact(
                    matrix,
                    tuple(
                        (Fraction.from_float(float(value)) - shift,)
                        for value, shift in zip(point, offset, strict=True)
                    ),
                )
            )
            for point in result
        )
        result = np.asarray(
            [[float(value) for value in point] for point in exact_points],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(result)):
            raise ValueError("A world seed has no finite index-space carrier.")
        if affine_source is None:
            raise ValueError("Affine seed inversion requires its categorical authority.")
        from ._compartments import _compartment_codes

        codes, region_ids, _ = _compartment_codes(affine_source)
        if region_ids != complex_.region_ids:
            raise ValueError("Affine seed regions differ from the categorical authority.")

        def seed_regions(point: tuple[Fraction, ...]) -> set[int]:
            axes = []
            for value in point:
                shifted = value + Fraction(1, 2)
                lower = shifted.numerator // shifted.denominator
                axes.append((lower - 1, lower) if shifted.denominator == 1 else (lower,))
            return {
                int(codes[index])
                if all(0 <= index[axis] < codes.shape[axis] for axis in range(3))
                else -1
                for index in product(*axes)
            }

        for exact, carrier, region in zip(exact_points, result, regions, strict=True):
            rounded = tuple(Fraction.from_float(float(value)) for value in carrier)
            if seed_regions(exact) != {region} or seed_regions(rounded) != {region}:
                raise ValueError(
                    "A world seed and its index carrier must lie in the same declared material interior."
                )
    return result, np.asarray(regions, dtype=np.int32)


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
    *,
    affine: np.ndarray | None = None,
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
        tables.points[sample_vertices]
        if affine is None
        else tables.points[sample_vertices] @ affine[:3, :3].T + affine[:3, 3],
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


def _schedule_quality_rows(
    rows: tuple[tuple[str, str, int, float], ...], /
) -> tuple[tuple[str, str, int, float], ...]:
    # PHX_MC_TET_UNMET_SIZE = 1. Keep every raw row in final_quality_unmet.
    return tuple(row for row in rows if row[0] != "1")


# PHX_MC_TET_UNMET_RADIUS_EDGE = 0 and PHX_MC_TET_UNMET_DIHEDRAL = 2 are the
# schedule's shape targets, named as they appear in soft evidence keys.
_SHAPE_TARGET_CRITERIA = {"0": "radius_edge_ratio", "2": "dihedral_degrees"}


def _split_shape_target_rows(
    schedule: NativeVolumeSchedule,
    rows: tuple[tuple[str, str, int, float], ...],
    /,
) -> tuple[
    tuple[tuple[str, str, int, float], ...], tuple[tuple[str, str, int, float], ...]
]:
    """Partition schedule rows into gated rows and soft shape-target rows.

    A hard schedule gates every row. A soft schedule moves each radius-edge or
    dihedral row whose native reason is not NONREPRESENTABLE out of the gate;
    validity, determinant, construction and nonrepresentable rows stay gated.
    """
    if _shape_targets_gated(schedule):
        return rows, ()
    gated: list[tuple[str, str, int, float]] = []
    soft: list[tuple[str, str, int, float]] = []
    for row in rows:
        if (
            row[0] in _SHAPE_TARGET_CRITERIA
            and int(row[1]) != TetMeshUnmetReason.NONREPRESENTABLE
        ):
            soft.append(row)
        else:
            gated.append(row)
    return tuple(gated), tuple(soft)


def _soft_shape_target_evidence(
    construction: VolumeConstruction,
    rows: tuple[tuple[str, str, int, float], ...],
    /,
) -> tuple[tuple[str, float], ...]:
    """Publish soft shape rows by published cell ID, criterion and native reason.

    Each row becomes ``soft_shape_target:cell:<id>:<criterion>:<reason>`` with
    its achieved value, after ``soft_shape_target_unmet_count``.
    """
    native_ids = np.asarray(construction.native_cell_ids, dtype=np.int64)
    published = np.concatenate(
        [
            np.asarray(block.global_ids, dtype=np.int64)
            for block in construction.mesh.blocks
        ]
    )
    if published.shape != native_ids.shape:
        raise ValueError("Native cell IDs must align with the published mesh rows.")
    cell_by_native = dict(zip(native_ids.tolist(), published.tolist(), strict=True))
    evidence: list[tuple[str, float]] = [
        ("soft_shape_target_unmet_count", float(len(rows)))
    ]
    for criterion, reason, native_cell, value in rows:
        cell = cell_by_native.get(int(native_cell))
        if cell is None:
            raise ValueError(
                "A soft shape row names a cell absent from the construction."
            )
        evidence.append(
            (
                f"soft_shape_target:cell:{cell}:{_SHAPE_TARGET_CRITERIA[criterion]}:"
                f"{TetMeshUnmetReason(int(reason)).name.lower()}",
                value,
            )
        )
    return tuple(evidence)


def _unmet_rows(
    unmet: TetMeshUnmet | TetMeshQuality | TetMeshCandidateQuality, /
) -> tuple[tuple[str, str, int, float], ...]:
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


def compose_plc_root_point(
    root_points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    vertex_stencils: tuple[tuple[tuple[int, Fraction], ...], ...],
    vertices: tuple[int, ...],
    coefficients: tuple[Fraction, ...],
    /,
) -> tuple[tuple[tuple[int, Fraction], ...], tuple[Fraction, Fraction, Fraction]]:
    """Compose an execution-simplex point against the original PLC roots.

    Coefficients and source stencils are exact, so the returned point has not
    passed through the rounded execution vertex coordinates. Root indices name
    rows of the original PLC, not rows of the execution complex.
    """
    if len(vertices) != len(coefficients) or not vertices:
        raise ValueError(
            "A source point must have matching simplex vertices and weights."
        )
    if (
        any(not isinstance(weight, Fraction) or weight < 0 for weight in coefficients)
        or sum(coefficients, Fraction(0)) != 1
    ):
        raise ValueError(
            "Source simplex coefficients must be nonnegative exact weights summing to one."
        )
    weights: dict[int, Fraction] = {}
    for vertex, coefficient in zip(vertices, coefficients, strict=True):
        if not isinstance(vertex, int) or not 0 <= vertex < len(vertex_stencils):
            raise ValueError(
                "A source simplex vertex is absent from the execution stencils."
            )
        stencil = vertex_stencils[vertex]
        if (
            not stencil
            or any(
                not isinstance(root, int)
                or not 0 <= root < len(root_points)
                or not isinstance(weight, Fraction)
                or weight < 0
                for root, weight in stencil
            )
            or sum((weight for _, weight in stencil), Fraction(0)) != 1
            or len({root for root, _ in stencil}) != len(stencil)
        ):
            raise ValueError(
                "Execution stencils must be unique, nonnegative exact root weights summing to one."
            )
        for root, weight in stencil:
            weights[root] = weights.get(root, Fraction(0)) + coefficient * weight
    terms = tuple(sorted((root, weight) for root, weight in weights.items() if weight))
    point = tuple(
        sum((weight * root_points[root][axis] for root, weight in terms), Fraction(0))
        for axis in range(3)
    )
    return terms, (point[0], point[1], point[2])


def _root_normalized_candidate(
    arrays: TetMeshArrays,
    tables: _SourceTables,
    root_points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    vertex_stencils: tuple[tuple[tuple[int, Fraction], ...], ...],
    /,
) -> tuple[
    np.ndarray,
    tuple[tuple[tuple[int, Fraction], ...] | None, ...],
    tuple[tuple[Fraction, Fraction, Fraction] | None, ...],
]:
    """Construct the actual candidate directly from original-root stencils."""
    points = np.asarray(arrays.points, dtype=np.float64).copy()
    stencils: list[tuple[tuple[int, Fraction], ...] | None] = [None] * len(points)
    exact: list[tuple[Fraction, Fraction, Fraction] | None] = [None] * len(points)
    boundary = set(map(int, np.asarray(arrays.faces).reshape(-1)))
    boundary.update(map(int, np.asarray(arrays.segments).reshape(-1)))
    boundary.update(range(min(len(vertex_stencils), len(points))))
    for row in sorted(boundary):
        if row < len(vertex_stencils):
            vertices, coefficients = (row,), (Fraction(1),)
        else:
            kind = int(arrays.witness_strata[row])
            entity = int(arrays.witness_entities[row])
            if kind == 1 and 0 <= entity < len(tables.source_segment_vertices):
                t = Fraction(float(arrays.witness_parameters[row, 0]))
                vertices = tuple(map(int, tables.source_segment_vertices[entity]))
                coefficients = (1 - t, t)
            elif kind == 2 and 0 <= entity < len(tables.faces):
                u, v = (
                    Fraction(float(value)) for value in arrays.witness_parameters[row]
                )
                vertices = tuple(map(int, tables.faces[entity]))
                coefficients = (1 - u - v, u, v)
            else:
                raise ValueError(
                    "A generated boundary point lacks an exact execution-simplex witness."
                )
        terms, point = compose_plc_root_point(
            root_points, vertex_stencils, vertices, coefficients
        )
        rounded = np.asarray(tuple(float(value) for value in point), dtype=np.float64)
        if row < len(vertex_stencils):
            # A rational zero has no sign. Keep authored input bits, including
            # signed zero, after checking the correctly rounded numeric value.
            if not np.array_equal(rounded, points[row]):
                raise ValueError(
                    "An execution input vertex disagrees with its original root stencil."
                )
        else:
            points[row] = rounded
        stencils[row] = terms
        exact[row] = point
    return points, tuple(stencils), tuple(exact)


def _mapped_plc_region_rows(
    complex_: PiecewiseLinearComplex, domain: MappedReferenceDomain, /
) -> np.ndarray:
    """Resolve core-local materials against the complete mapped source bank."""
    declared = tuple(
        zip(
            domain.reference_domain.region_ids,
            domain.reference_domain.region_occurrence_paths,
            strict=True,
        )
    )
    rows: list[int] = []
    for key in zip(complex_.region_ids, complex_.region_occurrence_paths, strict=True):
        matches = [row for row, candidate in enumerate(declared) if candidate == key]
        if len(matches) != 1:
            raise ValueError(
                "Mapped PLC regions must retain unique declared root identities and occurrences."
            )
        rows.append(matches[0])
    return np.asarray(rows, dtype=np.int64)


def _make_construction(
    complex_: PiecewiseLinearComplex,
    source_id: str,
    source_revision: str,
    domain: PiecewiseLinearDomain | MappedReferenceDomain | AffinePlcDomain,
    tables: _SourceTables,
    arrays: TetMeshArrays,
    evidence: TetMeshSourceEvidence,
    refinement_status: MeshcoreStatus,
    improvement_status: MeshcoreStatus,
    refinement_unmet: tuple[tuple[str, str, int, float], ...],
    improvement_unmet: tuple[tuple[str, str, int, float], ...],
    refinement_inserted_vertices: int,
    improvement_inserted_vertices: int,
    improvement_collapsed_vertices: int,
    improvement_relocated_vertices: int,
    final_quality_identity: TetMeshIdentity | None,
    final_quality_unmet: tuple[tuple[str, str, int, float], ...],
    root_vertex_stencils: tuple[tuple[tuple[int, Fraction], ...] | None, ...] = (),
    root_exact_points: tuple[tuple[Fraction, Fraction, Fraction] | None, ...] = (),
    candidate_quality_id: str | None = None,
    historical_quality_identity: TetMeshIdentity | None = None,
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
        mapped_region_rows = _mapped_plc_region_rows(complex_, domain)
        target_geometry = CellGeometrySpec.affine(mesh)
        reference_chart_controls = exact_reference_chart_controls(
            domain, mesh, target_geometry, mapped_region_rows[cell_regions]
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
        elif stratum in (2, 3) and 0 <= entity < tables.face_facets.size:
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
            if _schedule_quality_rows(final_quality_unmet)
            else MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=(mesh.mesh_id,),
        ),
    )
    return VolumeConstruction(
        mesh=mesh,
        domain=domain,
        cell_regions=cell_regions,
        native_cell_ids=np.asarray(arrays.cell_ids, dtype=np.uint64).copy(),
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
        refinement_inserted_vertices=refinement_inserted_vertices,
        improvement_inserted_vertices=improvement_inserted_vertices,
        improvement_collapsed_vertices=improvement_collapsed_vertices,
        improvement_relocated_vertices=improvement_relocated_vertices,
        final_quality_identity=final_quality_identity,
        final_quality_unmet=final_quality_unmet,
        stages=stages,
        root_vertex_stencils=tuple(root_vertex_stencils[int(row)] for row in used)
        if root_vertex_stencils
        else (),
        root_exact_points=tuple(root_exact_points[int(row)] for row in used)
        if root_exact_points
        else (),
        candidate_quality_id=candidate_quality_id,
        historical_quality_identity=historical_quality_identity,
        reference_chart_controls=reference_chart_controls,
    )


def _rebind_mapped_quality_proposal(
    arrays: TetMeshArrays,
    normalized: np.ndarray,
    proposal: TetMeshArrays,
    represented: np.ndarray,
    size_targets: np.ndarray,
    tables: _SourceTables,
    protected_facets: np.ndarray,
    root_rows: tuple[tuple[tuple[int, Fraction], ...] | None, ...],
    root_exact: tuple[tuple[Fraction, Fraction, Fraction] | None, ...],
    schedule: NativeVolumeSchedule,
    /,
    *,
    source_id: str,
    source_revision: str,
    source_map_id: str,
    frame_id: str,
) -> tuple[
    TetMeshArrays,
    TetMeshSourceEvidence,
    TetMeshCandidateQuality,
    tuple[tuple[tuple[int, Fraction], ...] | None, ...],
    tuple[tuple[Fraction, Fraction, Fraction] | None, ...],
]:
    """Rebuild proposal topology against the original reference source banks.

    Native source evidence is recomputed on the original native boundary
    coordinates. The separately evaluated candidate uses the exact-root
    represented reference coordinates, as in the ordinary PLC path.
    """
    old_ids = np.asarray(arrays.vertex_ids, dtype=np.int64)
    new_ids = np.asarray(proposal.vertex_ids, dtype=np.int64)
    old_rows = {int(identifier): row for row, identifier in enumerate(old_ids)}
    new_rows = {int(identifier): row for row, identifier in enumerate(new_ids)}
    input_count = len(tables.points)
    if (
        len(old_rows) != len(old_ids)
        or len(new_rows) != len(new_ids)
        or input_count > len(old_ids)
        or len(root_rows) != len(old_ids)
        or len(root_exact) != len(old_ids)
        or np.asarray(represented).shape != proposal.points.shape
    ):
        raise ValueError("Mapped proposal has inconsistent source or vertex ancestry.")
    input_ids = tuple(int(identifier) for identifier in old_ids[:input_count])
    if any(identifier not in new_rows for identifier in input_ids):
        raise ValueError("Mapped proposal removed an authored execution vertex.")
    first = [new_rows[identifier] for identifier in input_ids]
    input_set = set(first)
    if np.asarray(size_targets).shape != new_ids.shape:
        raise ValueError("Mapped proposal size targets are not bound to its vertex rows.")
    order = np.asarray(
        first + [row for row in range(len(new_ids)) if row not in input_set],
        dtype=np.int64,
    )
    inverse = np.empty(order.shape, dtype=np.int64)
    inverse[order] = np.arange(order.size, dtype=np.int64)
    target = np.asarray(represented, dtype=np.float64)[order]
    native_points = target.copy()
    strata = np.zeros(order.size, dtype=np.int8)
    entities = np.full(order.size, -1, dtype=np.int32)
    parameters = np.zeros((order.size, 2), dtype=np.float64)
    radii = np.zeros(order.size, dtype=np.float64)
    protected_vertices = np.zeros(order.size, dtype=np.bool_)
    new_root_rows: list[tuple[tuple[int, Fraction], ...] | None] = []
    new_root_exact: list[tuple[Fraction, Fraction, Fraction] | None] = []
    for new_row, proposal_row in enumerate(order):
        identifier = int(new_ids[int(proposal_row)])
        old_row = old_rows.get(identifier)
        if old_row is None:
            new_root_rows.append(None)
            new_root_exact.append(None)
            continue
        source_stencil = root_rows[old_row]
        source_point = root_exact[old_row]
        if source_stencil is not None:
            if source_point is None or not np.array_equal(
                target[new_row].view(np.uint64),
                np.asarray(normalized[old_row], dtype=np.float64).view(np.uint64),
            ):
                raise ValueError(
                    "Mapped proposal changed a retained exact source vertex."
                )
            native_points[new_row] = arrays.points[old_row]
        elif int(arrays.witness_strata[old_row]) != 0:
            raise ValueError(
                "A constrained mapped proposal vertex lacks original-root ancestry."
            )
        strata[new_row] = arrays.witness_strata[old_row]
        entities[new_row] = arrays.witness_entities[old_row]
        parameters[new_row] = arrays.witness_parameters[old_row]
        radii[new_row] = arrays.protection_radii[old_row]
        protected_vertices[new_row] = arrays.protected_vertices[old_row]
        new_root_rows.append(source_stencil)
        new_root_exact.append(source_point)
    original_source = TetMeshSourceComplex(
        arrays.source_points,
        arrays.source_faces,
        arrays.source_face_group_rows,
        arrays.source_face_tolerances,
        arrays.source_segments,
        arrays.source_segment_group_rows,
        arrays.source_segment_tolerances,
        strata,
        entities,
        parameters,
        face_ids=arrays.source_face_ids,
        segment_ids=arrays.source_segment_ids,
        face_planes=arrays.source_face_planes,
        face_plane_tolerances=arrays.source_face_plane_tolerances,
    )
    state = TetMesh3D(
        native_points,
        inverse[np.asarray(proposal.tetrahedra, dtype=np.int64)].astype(np.int32),
        proposal.tetrahedron_regions,
        inverse[np.asarray(proposal.faces, dtype=np.int64)].astype(np.int32),
        proposal.face_sources,
        inverse[np.asarray(proposal.segments, dtype=np.int64)].astype(np.int32),
        proposal.segment_sources,
        boundary_policy=NativePlcBoundaryPolicy.CONFORMING,
        initial_vertex_sizes=np.asarray(size_targets, dtype=np.float64)[order],
        protection_radii=radii,
        protected_vertices=protected_vertices,
        vertex_ids=new_ids[order],
        cell_ids=proposal.cell_ids,
        facet_ids=proposal.facet_ids,
        face_orientations=proposal.facet_orientations,
        segment_ids=proposal.segment_ids,
        source=original_source,
        implicit_source=None
        if arrays.implicit_source is None
        else arrays.implicit_source.source,
        source_id=source_id,
        source_revision=source_revision,
        source_map_id=source_map_id,
        frame_id=frame_id,
    )
    try:
        if np.any(protected_facets):
            groups = np.searchsorted(
                tables.face_ids, tables.facet_labels[protected_facets]
            )
            state.protect_facet_group_rows(groups.astype(np.int32))
        current = state.arrays()
        if not np.array_equal(current.vertex_ids, new_ids[order]):
            raise ValueError("Mapped source rebind changed the represented vertex order.")
        if not np.array_equal(
            np.asarray(current.vertex_size_targets, dtype=np.float64).view(np.uint64),
            np.asarray(size_targets, dtype=np.float64)[order].view(np.uint64),
        ):
            raise ValueError(
                "Mapped source rebind changed the validated vertex size targets."
            )
        evidence = state.source_evidence()
        candidate_quality = state.evaluate_candidate_quality(
            target,
            vertex_ids=current.vertex_ids,
            radius_edge_bound=schedule.radius_edge_bound,
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=schedule.minimum_relative_determinant,
        )
        if np.any((candidate_quality.criteria == 4) & (candidate_quality.reasons == 2)):
            raise ValueError(
                "Mapped proposal has nonpositive represented reference orientation."
            )
        return (
            current._replace(points=target),
            evidence,
            candidate_quality,
            tuple(new_root_rows),
            tuple(new_root_exact),
        )
    finally:
        state.close()


def _kept_carrier_recovery(
    complex_: PiecewiseLinearComplex,
    carrier: tuple[np.ndarray, np.ndarray],
    segments: np.ndarray,
    /,
) -> PlcRecovery3D:
    """Present a kept conforming carrier as its Steiner-free PLC recovery.

    The carrier tetrahedralizes the one region of a closed triangle PLC: its
    vertex bank starts with the PLC vertices bitwise, its remaining vertices are
    region-interior points, and its exterior faces are exactly the PLC
    triangles. Nothing is recovered, so every triangle is one output face in
    its declared orientation and every active segment one output segment. The
    native state revalidates orientation, manifoldness and conformity.
    """
    points = np.asarray(carrier[0], dtype=np.float64)
    tetrahedra = np.asarray(carrier[1])
    source_points = np.asarray(complex_.vertices, dtype=np.float64)
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    sides = np.sort(np.asarray(complex_.facet_regions, dtype=np.int64), axis=1)
    count = source_points.shape[0]
    if (
        len(complex_.region_ids) != 1
        or np.any(np.diff(offsets) != 3)
        or not np.array_equal(sides, np.tile((-1, 0), (sides.shape[0], 1)))
        or points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < count
        or not np.array_equal(
            points[:count].view(np.uint64), source_points.view(np.uint64)
        )
        or tetrahedra.ndim != 2
        or tetrahedra.shape[1] != 4
        or tetrahedra.shape[0] == 0
        or tetrahedra.dtype.kind not in "iu"
    ):
        raise ValueError(
            "A kept carrier must tetrahedralize one closed triangle PLC region whose vertex bank it extends."
        )
    cells = np.asarray(tetrahedra, dtype=np.int64)
    if (
        np.any(cells < 0)
        or np.any(cells >= points.shape[0])
        or np.unique(cells).size != points.shape[0]
    ):
        raise ValueError("A kept carrier must use exactly its own vertex bank.")
    triangles = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape((-1, 3))
    cell_faces = np.sort(
        np.concatenate(
            [
                cells[:, (1, 2, 3)],
                cells[:, (0, 2, 3)],
                cells[:, (0, 1, 3)],
                cells[:, (0, 1, 2)],
            ]
        ),
        axis=1,
    )
    unique_faces, face_counts = np.unique(cell_faces, axis=0, return_counts=True)
    if np.any(face_counts > 2) or not np.array_equal(
        unique_faces[face_counts == 1], np.unique(np.sort(triangles, axis=1), axis=0)
    ):
        raise ValueError("A kept carrier exterior must be exactly the PLC triangles.")
    segment_rows = np.asarray(segments, dtype=np.int32).reshape((-1, 2))
    vertex_count = points.shape[0]
    vertex_dimensions = np.full((vertex_count,), 3, dtype=np.int8)
    vertex_dimensions[:count] = 0
    return PlcRecovery3D(
        MeshcoreStatus.OK,
        points,
        cells.astype(np.int32),
        np.zeros((cells.shape[0],), dtype=np.int32),
        triangles.astype(np.int32),
        np.asarray(complex_.polygon_facets, dtype=np.int32),
        np.ones((triangles.shape[0],), dtype=np.int8),
        segment_rows,
        np.arange(segment_rows.shape[0], dtype=np.int32),
        np.zeros((vertex_count,), dtype=np.float64),
        vertex_dimensions,
        segment_rows,
        triangles.astype(np.int32),
        np.arange(triangles.shape[0], dtype=np.int32),
        np.zeros((vertex_count,), dtype=np.int8),
        np.full((vertex_count,), -1, dtype=np.int32),
        np.zeros((vertex_count, 2), dtype=np.float64),
        np.zeros((vertex_count,), dtype=np.float64),
        PlcSuppressionSearch(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
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
    global_affine: np.ndarray | None = None,
    affine_source: CompartmentMeshingSource | None = None,
    protected_facets: np.ndarray | None = None,
    boundary_collapse_quality: bool = False,
    root_points: object | None = None,
    root_vertex_stencils: tuple[tuple[tuple[int, Fraction], ...], ...] | None = None,
    carrier: tuple[np.ndarray, np.ndarray] | None = None,
) -> VolumeConstruction:
    """Recover constraints, classify every region, and refine the PLC fill.

    ``carrier`` replaces constraint recovery by a kept conforming
    tetrahedralization of a one-region triangle PLC (see
    ``_kept_carrier_recovery``), refined and improved in place.
    """
    if carrier is not None and (
        source_domain is not None
        or global_affine is not None
        or protected_facets is not None
        or boundary_collapse_quality
        or root_points is not None
    ):
        raise ValueError("A kept carrier is refined only in its own Cartesian PLC frame.")
    if not isinstance(boundary_collapse_quality, bool):
        raise TypeError("boundary_collapse_quality must be bool.")
    if boundary_collapse_quality and (
        complex_.boundary != "conforming" or protected_facets is None
    ):
        raise ValueError(
            "Boundary quality repair requires an explicitly conforming protected-facet policy."
        )
    if (root_points is None) != (root_vertex_stencils is None):
        raise ValueError("Root points and execution stencils must be supplied together.")
    if root_points is not None and global_affine is not None:
        raise ValueError("Root-normalized execution cannot use a global affine carrier.")
    exact_roots = None
    if root_points is not None:
        root_array = np.asarray(root_points, dtype=np.float64)
        if (
            root_array.ndim != 2
            or root_array.shape[1] != 3
            or not np.all(np.isfinite(root_array))
        ):
            raise ValueError(
                "Original PLC root points must be finite three-dimensional coordinates."
            )
        if root_vertex_stencils is None or len(root_vertex_stencils) != len(
            complex_.vertices
        ):
            raise ValueError(
                "Execution root stencils must align with every PLC input vertex."
            )
        exact_roots = tuple(
            (
                Fraction(float(point[0])),
                Fraction(float(point[1])),
                Fraction(float(point[2])),
            )
            for point in root_array
        )
    if (global_affine is None) != (affine_source is None) or (
        global_affine is not None and source_domain is not None
    ):
        raise ValueError(
            "A global affine PLC requires its source authority and no rooted map."
        )
    if protected_facets is not None:
        protected_facets = np.asarray(protected_facets)
        if protected_facets.shape != (
            complex_.facet_count,
        ) or protected_facets.dtype != np.dtype(np.bool_):
            raise ValueError("protected_facets must contain one boolean per PLC facet.")
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

    facet_tolerances = np.full((complex_.facet_count,), np.inf, dtype=np.float64)
    segment_tolerances = np.full(
        (canonical_source_edges.shape[0],), np.inf, dtype=np.float64
    )
    active_segment_rows = {int(row) for row in explicit_segment_rows.tolist()}
    if protected_facets is None:
        active_segment_rows.update(_authored_coplanar_edge_rows(complex_).tolist())
    else:
        # The selective conforming layer route independently replays exact
        # source-facet overlap; its coplanar triangulation edges are not
        # necessarily physical or authored protected features.
        from ._layer_core_edge_policy import conforming_physical_edge_rows

        active_segment_rows.update(conforming_physical_edge_rows(complex_))
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
    # internal diagonals, and those of declared-plane facets whatever their
    # rounding. Authored segments, requested curve features and the
    # coplanar-pair edges of undeclared facets are passed as explicit 1D
    # constraints.
    if carrier is None:
        seeds, seed_regions = _region_seed_points(
            complex_, specification, affine=global_affine, affine_source=affine_source
        )
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
            facet_planes=complex_.facet_planes,
            affine_metric=None if global_affine is None else global_affine[:3, :3],
            protected_facets=protected_facets,
            facet_plane_tolerances=complex_.facet_plane_tolerances,
        )
    else:
        recovery = _kept_carrier_recovery(complex_, carrier, active_segment_vertices)
    tables = _source_tables(complex_, specification, recovery)
    if complex_.geometry is not None and source_domain is None:
        raise ValueError(
            "Mapped PLC geometry requires its independently declared MappedReferenceDomain."
        )
    if source_domain is None:
        reference_domain = declared_plc_domain(
            complex_, tables.faces, tables.face_polygons, source_id
        )
        domain = (
            reference_domain
            if global_affine is None
            else AffinePlcDomain(
                reference_domain,
                global_affine[:3, :3],
                global_affine[:3, 3],
                source_revision=source_revision,
                authority=affine_source,
            )
        )
    else:
        if (
            source_domain.source_id != source_id
            or source_domain.source_revision != source_revision
        ):
            raise ValueError(
                "Mapped PLC domain must bind the source identity, revision and declared regions."
            )
        _mapped_plc_region_rows(complex_, source_domain)
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
    sizes = _size_field(complex_, specification, recovery, tables, affine=global_affine)
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
        affine_metric=None if global_affine is None else global_affine[:3, :3],
    )
    mapped_proposal_run = None
    mapped_predecessor_identity = None
    mapped_predecessor_candidate_id = None
    mapped_physical_quality = None
    boundary_repair = None
    boundary_repair_unmet: tuple[tuple[str, str, int, float], ...] = ()
    boundary_predecessor = None
    try:
        original_vertex_ids = None
        if boundary_collapse_quality:
            initial = state.arrays()
            input_count = len(tables.points)
            if input_count > len(initial.points) or not np.array_equal(
                np.asarray(initial.points[:input_count], dtype=np.float64).view(
                    np.uint64
                ),
                np.asarray(tables.points, dtype=np.float64).view(np.uint64),
            ):
                raise ValueError(
                    "Native recovery changed the authored execution vertex bank."
                )
            original_vertex_ids = np.asarray(
                initial.vertex_ids[:input_count], dtype=np.uint64
            ).copy()
        if protected_facets is not None and np.any(protected_facets):
            groups = np.searchsorted(
                tables.face_ids, tables.facet_labels[protected_facets]
            )
            state.protect_facet_group_rows(groups.astype(np.int32))
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
        refinement_unmet = _unmet_rows(state.unmet())
        # Refinement retained `sizes` (interpolated to its inserted vertices) on
        # the mesh; improvement gates every edit by that field.
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
        improvement_unmet = _unmet_rows(state.unmet())
        # Query the retained field on the current mesh without modifying it.
        # Its source identity and generation travel with the final evidence.
        quality = state.evaluate_quality(
            radius_edge_bound=schedule.radius_edge_bound,
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=schedule.minimum_relative_determinant,
        )
        final_quality_unmet = _unmet_rows(quality)
        if boundary_collapse_quality and _schedule_quality_rows(final_quality_unmet):
            if original_vertex_ids is None:
                raise ValueError(
                    "Boundary quality repair lost its authored input vertex bank."
                )
            boundary_predecessor = quality.identity
            boundary_repair = state.improve(
                radius_edge_bound=schedule.radius_edge_bound,
                min_dihedral_degrees=schedule.minimum_dihedral_degrees,
                minimum_relative_determinant=schedule.minimum_relative_determinant,
                boundary_collapse_quality=True,
                original_vertex_ids=original_vertex_ids,
            )
            if boundary_repair.edit_status is not TetMeshEditStatus.APPLIED:
                if (
                    boundary_repair.edit_status is not TetMeshEditStatus.REJECTED_TOPOLOGY
                    or boundary_repair.status is not MeshcoreStatus.UNRESOLVED
                    or state.identity != boundary_predecessor
                ):
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        f"Boundary quality repair edit status was {boundary_repair.edit_status.name}.",
                    )
            if boundary_repair.status not in (
                MeshcoreStatus.OK,
                MeshcoreStatus.UNRESOLVED,
            ):
                raise MeshcoreError(
                    boundary_repair.status, "Boundary quality repair failed."
                )
            boundary_repair_unmet = _unmet_rows(state.unmet())
            quality = state.evaluate_quality(
                radius_edge_bound=schedule.radius_edge_bound,
                min_dihedral_degrees=schedule.minimum_dihedral_degrees,
                minimum_relative_determinant=schedule.minimum_relative_determinant,
            )
            final_quality_unmet = _unmet_rows(quality)
        arrays = state.arrays()
        evidence = state.source_evidence()
        if state.identity != quality.identity:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "Final quality evidence does not match the exported mesh generation.",
            )
        root_rows: tuple[tuple[tuple[int, Fraction], ...] | None, ...] = ()
        root_exact: tuple[tuple[Fraction, Fraction, Fraction] | None, ...] = ()
        candidate_quality_id = None
        historical_identity = None
        final_identity: TetMeshIdentity | None = quality.identity
        if exact_roots is not None:
            assert root_vertex_stencils is not None
            candidate, root_rows, root_exact = _root_normalized_candidate(
                arrays, tables, exact_roots, root_vertex_stencils
            )
            candidate_quality = state.evaluate_candidate_quality(
                candidate,
                vertex_ids=arrays.vertex_ids,
                radius_edge_bound=schedule.radius_edge_bound,
                min_dihedral_degrees=schedule.minimum_dihedral_degrees,
                minimum_relative_determinant=schedule.minimum_relative_determinant,
            )
            if candidate_quality.historical_identity != quality.identity:
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    "Candidate quality was not evaluated against the exported native history.",
                )
            # Native VALIDITY/NONREPRESENTABLE includes nonpositive exact
            # orient3d; such a candidate must never be published.
            if np.any(
                (candidate_quality.criteria == 4) & (candidate_quality.reasons == 2)
            ):
                raise ValueError(
                    "A root-normalized tetrahedron has nonpositive orientation."
                )
            final_quality_unmet = _unmet_rows(candidate_quality)
            historical_identity = quality.identity
            candidate_quality_id = candidate_quality.candidate_id
            final_identity = None
            if source_domain is not None:
                if protected_facets is None:
                    raise ValueError(
                        "Mapped conforming quality requires its protected facet policy."
                    )
                from ._layer_core_mapped import (
                    _propose_mapped_physical_improvement,
                    _validate_mapped_atomic_size_candidate,
                )

                proposed = _propose_mapped_physical_improvement(
                    source_domain,
                    arrays._replace(points=candidate),
                    _mapped_plc_region_rows(complex_, source_domain),
                    np.asarray(arrays.vertex_ids[: len(tables.points)], dtype=np.int64),
                    schedule,
                )
                if proposed is not None:
                    (
                        proposal_arrays,
                        represented,
                        mapped_proposal_run,
                        mapped_physical_quality,
                    ) = proposed
                    try:
                        proposal_sizes = _validate_mapped_atomic_size_candidate(
                            state,
                            quality.identity,
                            tables,
                            exact_roots,
                            root_vertex_stencils,
                            candidate,
                            proposal_arrays,
                            represented,
                            schedule,
                        )
                    except ValueError as error:
                        raise MeshcoreError(
                            MeshcoreStatus.UNRESOLVED,
                            f"Mapped quality proposal could not prove retained size constraints: {error}",
                        ) from error
                    mapped_predecessor_identity = quality.identity
                    mapped_predecessor_candidate_id = candidate_quality.candidate_id
                    arrays, evidence, candidate_quality, root_rows, root_exact = (
                        _rebind_mapped_quality_proposal(
                            arrays,
                            candidate,
                            proposal_arrays,
                            represented,
                            proposal_sizes,
                            tables,
                            protected_facets,
                            root_rows,
                            root_exact,
                            schedule,
                            source_id=source_id,
                            source_revision=source_revision,
                            source_map_id=source_map_id,
                            frame_id=frame_id,
                        )
                    )
                    final_quality_unmet = _unmet_rows(candidate_quality)
                    historical_identity = candidate_quality.historical_identity
                    candidate_quality_id = candidate_quality.candidate_id
                else:
                    arrays = arrays._replace(points=candidate)
            else:
                arrays = arrays._replace(points=candidate)
    finally:
        state.close()
    construction = _make_construction(
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
        refinement.inserted_vertices,
        improvement.inserted_vertices,
        improvement.collapsed_vertices,
        improvement.relocated_vertices,
        final_identity,
        final_quality_unmet,
        root_rows,
        root_exact,
        candidate_quality_id,
        historical_identity,
    )
    if boundary_repair is not None:
        if boundary_predecessor is None:
            raise ValueError(
                "Boundary quality repair lost its predecessor native identity."
            )
        repair_stage = MeshingStageReport(
            MeshingStageKind.OPTIMIZATION,
            MeshingStageStatus.WARNING
            if boundary_repair.status is MeshcoreStatus.UNRESOLVED
            else MeshingStageStatus.PASSED,
            input_ids=(complex_.complex_id,),
            output_ids=(construction.mesh.mesh_id,),
            created_count=boundary_repair.inserted_vertices,
            modified_count=boundary_repair.relocated_vertices,
            deleted_count=boundary_repair.collapsed_vertices,
        )
        construction = replace(
            construction,
            stages=(*construction.stages, repair_stage),
            boundary_quality_repair_status=boundary_repair.status,
            boundary_quality_repair_unmet=boundary_repair_unmet,
            boundary_quality_predecessor_identity=boundary_predecessor,
        )
    if mapped_proposal_run is None:
        return construction
    if (
        mapped_predecessor_identity is None
        or mapped_predecessor_candidate_id is None
        or mapped_physical_quality is None
        or candidate_quality_id is None
    ):
        raise ValueError(
            "Mapped quality proposal lost its native history or final evaluation."
        )
    proposal_stage = MeshingStageReport(
        MeshingStageKind.OPTIMIZATION,
        MeshingStageStatus.WARNING
        if mapped_proposal_run.status is MeshcoreStatus.UNRESOLVED
        or len(mapped_physical_quality.cell_ids)
        else MeshingStageStatus.PASSED,
        input_ids=(mapped_predecessor_candidate_id,),
        output_ids=(candidate_quality_id,),
        created_count=mapped_proposal_run.inserted_vertices,
        modified_count=mapped_proposal_run.relocated_vertices,
        deleted_count=mapped_proposal_run.collapsed_vertices,
    )
    return replace(
        construction,
        stages=(*construction.stages, proposal_stage),
        mapped_quality_predecessor_identity=mapped_predecessor_identity,
    )


__all__ = [
    "NativeVolumeSchedule",
    "VolumeConstruction",
    "declared_plc_domain",
    "generate_plc_volume",
]
