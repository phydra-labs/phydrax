#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scientific source bindings consumed by the native meshing routes."""

from __future__ import annotations
from dataclasses import dataclass

from fractions import Fraction
from typing import final, Literal, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import normalized_identifier
from ...discretization import _cell_geometry as cell_geometry_owner, CellMesh
from ...discretization._cell_complex import (
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_mesh import SimplicialConnectivity
from ...discretization._hexahedral import HexahedralConnectivity
from ...discretization._reference_cell import reference_cell_topology
from ...discretization._tensor_support import PreparedTensorGrid
from ...geometry._atlas import BoundaryAtlas
from ...geometry._contracts import CompiledGeometry, GeometryKind
from ...geometry._coplanar_overlap import CoplanarTrianglePiece
from ...geometry._curve_boundary import SegmentChartCurve
from ...geometry._domain_coverage import PiecewiseLinearDomain
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from ...geometry._meshing_domain import MeshingDomain
from ...geometry._source_boundary_records import (
    SourceBoundaryQuery,
    SourceBoundaryRationalChartBank,
)
from ...geometry._source_identity import source_entity_id
from ...geometry._surface_source_support import SurfaceSourceCharts
from ...geometry.surface import SurfaceModel
from ...geometry.brep._projection import NativeBRepProjection, prepare_brep_projection
from ...geometry.brep._projection_contracts import BRepProjectionPolicy
from ...geometry.implicit._adaptive_discovery import AdaptiveImplicitSurface
from ...geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ...geometry.simplicial._regions import PlanarMeshRegion, SegmentMesh
from ...typing import checked
from .._association import (
    AssociationPropagationPolicy,
    GeometryAssociation,
    GeometryAssociationKind,
)
from .._boundary_layer import BoundaryLayerMesh, BoundaryLayerPolicy
from .._controls import BlockInterfaceControl, BoundaryLayerControl
from .._layer_core_regions import prepare_layer_region_identity
from .._plc_source import _numeric_entity_ids, PiecewiseLinearComplex
from .._scope import _SourceEntityScopes, MeshingScope
from .._structured import TransfiniteBlock
from .._sweep import SweepControl
from ._native_schedules import NativeCurveSchedule, NativeSurfaceSchedule


if TYPE_CHECKING:
    from ...geometry._curve_boundary import ParametricCurveBoundarySource
    from ...geometry._mapped_reference_boundary import MappedReferenceBoundarySource


type _ScopeEntityIdentifier = str | int | tuple[str | int, ...]


def _scope_entity_identifier(value: _ScopeEntityIdentifier, /) -> str | int:
    if isinstance(value, bool):
        raise TypeError("Source scope entity identifiers cannot be booleans.")
    if isinstance(value, (str, int)):
        return value
    return canonical_fingerprint(
        {"kind": "native-composite-scope-entity", "components": value}
    )


def _entity_ids(
    values: ArrayLike | None,
    count: int,
    kind: str,
    /,
    *,
    allow_repeated: bool = False,
) -> np.ndarray:
    if values is None:
        raise ValueError(f"Native source {kind} requires an authored entity-ID bank.")
    raw = np.asarray(values)
    if raw.shape != (count,) or raw.dtype.kind not in "iu":
        raise TypeError(
            f"Native source {kind} entity IDs must be an integer vector of length {count}."
        )
    if np.any(raw < 0):
        raise ValueError(f"Native source {kind} entity IDs must be nonnegative.")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise OverflowError(f"Native source {kind} entity IDs must fit int64.")
    identifiers = np.array(raw, dtype=np.int64, copy=True)
    if not allow_repeated and np.unique(identifiers).size != count:
        raise ValueError(f"Native source {kind} entity IDs must be unique.")
    identifiers.setflags(write=False)
    return identifiers


def _plc_scope_bank(
    complex_: PiecewiseLinearComplex, dimension: int, /
) -> tuple[tuple[str, tuple[str, ...], int], ...]:
    fields = {
        0: ("vertex_ids", "vertex_entity_ids", "vertex_occurrence_paths"),
        1: ("plc_edge_ids", "plc_edge_entity_ids", "plc_edge_occurrence_paths"),
        2: ("facet_ids", "facet_entity_ids", "facet_occurrence_paths"),
        3: ("region_ids", "region_entity_ids", "region_occurrence_paths"),
    }
    try:
        names_field, numeric_field, paths_field = fields[dimension]
    except KeyError as error:
        raise ValueError(
            "PLC source dimensions must be between zero and three."
        ) from error
    names = tuple(getattr(complex_, names_field))
    numeric = np.asarray(getattr(complex_, numeric_field), dtype=np.int64)
    raw_paths = getattr(complex_, paths_field, None)
    paths = (
        tuple(() for _ in names)
        if raw_paths is None
        else tuple(tuple(str(part) for part in path) for path in raw_paths)
    )
    if (
        numeric.shape != (len(names),)
        or len(paths) != len(names)
        or np.unique(numeric).size != numeric.size
        or any(not name for name in names)
    ):
        raise ValueError(
            "PLC source entity IDs, occurrence paths, and scope IDs must align."
        )
    return tuple(
        sorted(
            zip(names, paths, (int(value) for value in numeric.tolist()), strict=True),
            key=lambda row: row[2],
        )
    )


def _plc_scope_set_id(
    source_id: str,
    source_revision: str,
    dimension: int,
    bank: tuple[tuple[str, tuple[str, ...], int], ...],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "plc-meshing-entity-set",
            "source_id": source_id,
            "source_revision": source_revision,
            "dimension": dimension,
            "entities": tuple((name, path) for name, path, _ in bank),
        }
    )


@final
class _MappedPlcRootRelation(StrictModule, NonTrainableState):
    """Exact mapped ancestry with static rational metadata and dynamic arrays."""

    vertex_root_global_ids: np.ndarray
    segment_root_dimensions: np.ndarray
    segment_root_entity_ids: np.ndarray
    segment_root_cell_global_ids: np.ndarray
    segment_root_local_entity_indices: np.ndarray
    segment_root_reference_rational_tokens: SourceBoundaryRationalChartBank = eqx.field(
        static=True
    )
    segment_orientation_signs: np.ndarray
    polygon_source_facet_rows: np.ndarray
    polygon_root_face_global_ids: np.ndarray
    polygon_root_cell_global_ids: np.ndarray
    polygon_root_local_face_indices: np.ndarray
    polygon_root_reference_rational_tokens: tuple[
        SourceBoundaryRationalChartBank, ...
    ] = eqx.field(static=True)
    polygon_orientation_signs: np.ndarray
    segment_source_entity_ids: np.ndarray
    segment_source_entity_keys: tuple[str, ...] = eqx.field(static=True)
    segment_source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    polygon_source_entity_ids: np.ndarray
    polygon_source_entity_keys: tuple[str, ...] = eqx.field(static=True)
    polygon_source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    segment_source_parameter_rational_tokens: tuple[
        tuple[tuple[int, int], tuple[int, int]], ...
    ] = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)


def _mapped_triangle_key(vertices: tuple[int, ...], /) -> tuple[int, int, int]:
    if len(vertices) != 3:
        raise ValueError("Mapped boundary ancestry requires triangular faces.")
    first, second, third = vertices
    return min(
        (first, second, third),
        (second, third, first),
        (third, first, second),
    )


def _mapped_plc_bind_source_vertices(
    domain: MappedReferenceDomain,
    vertex_root_global_ids: ArrayLike | None,
    reference: NativePlcSource,
    complex_: PiecewiseLinearComplex,
    reference_vertex_roots: tuple[int, ...],
    mesh: CellMesh,
    root_vertex_rows: dict[int, int],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    root_paths = {
        int(identifier): path
        for identifier, path in zip(
            domain.root_vertex_ids, domain.root_vertex_occurrence_paths, strict=True
        )
    }
    if vertex_root_global_ids is None:
        boundary_root_ids = domain.reference_domain_vertex_root_ids
        if boundary_root_ids is None:
            raise ValueError(
                "Mapped PLC vertices need a declared reference-domain/root vertex relation."
            )
        root_key_to_id: dict[tuple[str, tuple[str, ...]], int] = {}
        for raw_root_id in boundary_root_ids:
            root_id = int(raw_root_id)
            key = (str(root_id), tuple(root_paths[root_id]))
            if key in root_key_to_id:
                raise ValueError(
                    "Mapped reference-domain source vertex keys are not unique."
                )
            root_key_to_id[key] = root_id
        root_scope_ids = np.asarray(reference.scope_entity_ids(0), dtype=np.int64)
        root_scope_keys = tuple(
            reference.source_entity_key_for_scope(0, int(identifier))
            for identifier in root_scope_ids.tolist()
        )
        scope_id_by_key = dict(zip(root_scope_keys, root_scope_ids.tolist(), strict=True))
        if len(scope_id_by_key) != root_scope_ids.size or set(root_key_to_id) != set(
            scope_id_by_key
        ):
            raise ValueError(
                "Mapped root boundary keys must be an exact subset of the current source owner's entity set."
            )
        root_bank = tuple(
            (name, path, scope_id_by_key[(name, path)])
            for name, path in sorted(root_key_to_id, key=lambda key: scope_id_by_key[key])
        )
        if _plc_scope_set_id(
            reference.source_id, reference.source_revision, 0, root_bank
        ) != reference.scope_entity_set_id(0):
            raise ValueError("Mapped PLC vertex set no longer matches its source owner.")
        source_vertex_scope_ids = np.asarray(complex_.vertex_entity_ids, dtype=np.int64)
        if source_vertex_scope_ids.shape != (len(complex_.vertex_ids),):
            raise ValueError(
                "Mapped PLC vertex scope IDs do not align with their authored keys."
            )
        source_vertex_keys = tuple(
            reference.source_entity_key_for_scope(0, int(identifier))
            for identifier in source_vertex_scope_ids.tolist()
        )
        if len(source_vertex_keys) != len(root_key_to_id) or set(
            source_vertex_keys
        ) != set(root_key_to_id):
            raise ValueError(
                "Mapped PLC vertices can auto-bind only when the complete authored source key set and occurrence paths match the declared root boundary keys; otherwise supply the retained root relation."
            )
        plc_vertex_root_ids = np.asarray(
            [root_key_to_id[key] for key in source_vertex_keys], dtype=np.int64
        )
    else:
        raw_root_ids = np.asarray(vertex_root_global_ids)
        if raw_root_ids.dtype.kind not in "iu" or raw_root_ids.shape != (
            len(complex_.vertex_ids),
        ):
            raise TypeError(
                "plc_vertex_root_global_ids must contain one integer root ID per authored PLC vertex."
            )
        if raw_root_ids.dtype.kind == "u" and np.any(
            raw_root_ids > np.iinfo(np.int64).max
        ):
            raise ValueError("PLC-to-root vertex IDs must fit int64.")
        plc_vertex_root_ids = np.asarray(raw_root_ids, dtype=np.int64)
    if (
        plc_vertex_root_ids.shape != (len(complex_.vertex_ids),)
        or np.unique(plc_vertex_root_ids).size != plc_vertex_root_ids.size
    ):
        raise ValueError("PLC vertices must map bijectively to root global vertex IDs.")
    try:
        plc_root_rows = np.asarray(
            [root_vertex_rows[int(identifier)] for identifier in plc_vertex_root_ids],
            dtype=np.int64,
        )
        domain_root_rows = np.asarray(
            [root_vertex_rows[int(identifier)] for identifier in reference_vertex_roots],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("PLC/root relation names an absent root vertex ID.") from error
    if (
        len(set(int(value) for value in reference_vertex_roots))
        != len(reference_vertex_roots)
        or not np.array_equal(
            np.asarray(complex_.vertices, dtype=np.float64),
            np.asarray(mesh.coordinates, dtype=np.float64)[plc_root_rows],
        )
        or not np.array_equal(
            np.asarray(domain.reference_domain.vertices, dtype=np.float64),
            np.asarray(mesh.coordinates, dtype=np.float64)[domain_root_rows],
        )
    ):
        raise ValueError(
            "Declared PLC/domain root-vertex relations do not validate against exact reference coordinates."
        )
    for source_row, root_id in enumerate(plc_vertex_root_ids.tolist()):
        if (
            tuple(complex_.vertex_occurrence_paths[source_row])
            != root_paths[int(root_id)]
        ):
            raise ValueError(
                "PLC/root vertex relation changed an authored occurrence path."
            )
    return plc_vertex_root_ids, plc_root_rows


def _mapped_plc_validate_source_facets(
    complex_: PiecewiseLinearComplex,
    plc_vertex_root_ids: np.ndarray,
    domain: MappedReferenceDomain,
    reference_vertex_roots: tuple[int, ...],
    /,
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[int]]:
    source_facet_regions = np.asarray(complex_.facet_regions, dtype=np.int64)
    if source_facet_regions.ndim != 2 or source_facet_regions.shape[1] != 2:
        raise ValueError("Mapped PLC facets require exactly two oriented region sides.")

    def sides_from_source(facet_row: int, /) -> tuple[str | None, str | None]:
        first, second = (
            int(source_facet_regions[facet_row, 1]),
            int(source_facet_regions[facet_row, 0]),
        )
        return (
            None if first < 0 else complex_.region_ids[first],
            None if second < 0 else complex_.region_ids[second],
        )

    source_surface: dict[tuple[int, int, int], tuple[str | None, str | None]] = {}
    polygon_offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    polygon_vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    polygon_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
    polygon_root_vertices: list[tuple[int, int, int]] = []
    polygon_source_facets: list[int] = []
    for polygon, facet_row in enumerate(polygon_facets.tolist()):
        first, last = (int(value) for value in polygon_offsets[polygon : polygon + 2])
        source_vertices = tuple(int(value) for value in polygon_vertices[first:last])
        if len(source_vertices) != 3:
            raise ValueError(
                "Mapped PLC/root correspondence requires triangular source facets."
            )
        root_triangle = (
            int(plc_vertex_root_ids[source_vertices[0]]),
            int(plc_vertex_root_ids[source_vertices[1]]),
            int(plc_vertex_root_ids[source_vertices[2]]),
        )
        key = _mapped_triangle_key(root_triangle)
        if key in source_surface:
            raise ValueError(
                "Mapped PLC/root correspondence contains duplicate oriented source triangles."
            )
        source_surface[key] = sides_from_source(int(facet_row))
        polygon_root_vertices.append(root_triangle)
        polygon_source_facets.append(int(facet_row))

    domain_surface: dict[tuple[int, int, int], tuple[str | None, str | None]] = {}
    domain_facets = np.asarray(domain.reference_domain.facets, dtype=np.int64)
    domain_facet_regions = np.asarray(
        domain.reference_domain.facet_regions, dtype=np.int64
    )
    if (
        domain_facets.ndim != 2
        or domain_facets.shape[1] != 3
        or domain_facet_regions.shape != (domain_facets.shape[0], 2)
    ):
        raise ValueError(
            "Mapped reference domains require triangular facets and two-sided material incidence."
        )
    for row in range(domain_facets.shape[0]):
        triangle = domain_facets[row]
        incidence = domain_facet_regions[row]
        root_triangle = (
            int(reference_vertex_roots[int(triangle[0])]),
            int(reference_vertex_roots[int(triangle[1])]),
            int(reference_vertex_roots[int(triangle[2])]),
        )
        key = _mapped_triangle_key(root_triangle)
        first, second = int(incidence[0]), int(incidence[1])
        sides = (
            None if first < 0 else domain.region_ids[first],
            None if second < 0 else domain.region_ids[second],
        )
        if key in domain_surface:
            raise ValueError(
                "Mapped reference domain contains duplicate oriented triangles."
            )
        domain_surface[key] = sides
    if source_surface != domain_surface:
        raise ValueError(
            "PLC facets, source keys, orientations, and region sides do not match the declared mapped reference boundary."
        )
    return source_facet_regions, polygon_root_vertices, polygon_source_facets


@dataclass(frozen=True, slots=True)
class _MappedRootInventory:
    """Host-only source and root incidence used to derive exact overlay pieces."""

    reference: NativePlcSource
    domain: MappedReferenceDomain
    complex: PiecewiseLinearComplex
    root_faces: np.ndarray | tuple[np.ndarray, ...]
    root_edges: np.ndarray
    root_cells: np.ndarray
    root_face_local_vertices: tuple[tuple[int, ...], ...]
    root_edge_local_vertices: tuple[tuple[int, ...], ...]
    root_cell_ids: np.ndarray
    root_face_ids: np.ndarray
    root_edge_ids: np.ndarray
    root_cell_regions: np.ndarray
    root_vertex_ids: np.ndarray
    root_vertex_rows: dict[int, int]
    face_owners: dict[int, list[tuple[int, int]]]
    edge_owners: dict[int, list[tuple[int, int]]]
    coordinates: tuple[tuple[Fraction, ...], ...]
    reference_corners: tuple[tuple[Fraction, ...], ...]
    root_reference_vertices: np.ndarray
    plc_vertex_root_ids: np.ndarray
    plc_root_rows: np.ndarray
    polygon_root_vertices: list[tuple[int, int, int]]
    polygon_source_facets: list[int]
    source_facet_regions: np.ndarray


type _MappedFaceOverlay = tuple[
    list[int],
    list[int],
    list[tuple[int, int]],
    list[tuple[int, int]],
    tuple[SourceBoundaryRationalChartBank, ...],
    list[tuple[int, int]],
]
type _MappedEdgePartition = tuple[
    list[int],
    list[int],
    list[int],
    list[int],
    list[tuple[tuple[Fraction, ...], ...] | np.ndarray],
    list[int],
    list[int],
    list[tuple[Fraction, Fraction]],
]


def _mapped_root_face_piece(
    inventory: _MappedRootInventory,
    source_triangle: tuple[tuple[Fraction, ...], ...],
    vertices: tuple[int, int, int],
    target_chart: tuple[tuple[Fraction, Fraction], ...],
    order: tuple[int, int, int],
    cell: int,
    local: int,
    side: int,
    /,
) -> tuple[tuple[tuple[Fraction, ...], ...], int]:
    coordinates, root_cells = inventory.coordinates, inventory.root_cells
    reference_corners = inventory.reference_corners
    root_face_local_vertices = inventory.root_face_local_vertices
    source_a = tuple(
        source_triangle[1][axis] - source_triangle[0][axis] for axis in range(3)
    )
    source_b = tuple(
        source_triangle[2][axis] - source_triangle[0][axis] for axis in range(3)
    )
    source_normal = (
        source_a[1] * source_b[2] - source_a[2] * source_b[1],
        source_a[2] * source_b[0] - source_a[0] * source_b[2],
        source_a[0] * source_b[1] - source_a[1] * source_b[0],
    )
    center = tuple(
        sum(
            (coordinates[int(vertex)][axis] for vertex in root_cells[cell]),
            Fraction(0),
        )
        / len(root_cells[cell])
        for axis in range(3)
    )
    source_side = sum(
        (
            source_normal[axis] * (center[axis] - source_triangle[0][axis])
            for axis in range(3)
        ),
        Fraction(0),
    )
    if (side == 0 and source_side <= 0) or (side == 1 and source_side >= 0):
        raise ValueError("Mapped root cell lies on the wrong oriented source facet side.")
    local_vertices = tuple(int(value) for value in root_cells[cell])
    target_refs = tuple(
        reference_corners[local_vertices.index(vertex)] for vertex in vertices
    )
    points = tuple(
        tuple(
            target_refs[0][axis]
            + target_chart[index][0] * (target_refs[1][axis] - target_refs[0][axis])
            + target_chart[index][1] * (target_refs[2][axis] - target_refs[0][axis])
            for axis in range(3)
        )
        for index in order
    )
    face_refs = tuple(
        reference_corners[index] for index in root_face_local_vertices[local]
    )
    first = tuple(face_refs[1][axis] - face_refs[0][axis] for axis in range(3))
    second = tuple(face_refs[-1][axis] - face_refs[0][axis] for axis in range(3))
    a = tuple(points[1][axis] - points[0][axis] for axis in range(3))
    b = tuple(points[2][axis] - points[0][axis] for axis in range(3))
    normal = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    piece_normal = (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
    dot = sum(
        (x * y for x, y in zip(normal, piece_normal, strict=True)),
        Fraction(0),
    )
    if not dot:
        raise ValueError("Mapped source overlap piece is degenerate.")
    return points, 1 if dot > 0 else -1


def _mapped_overlay_face_piece(
    inventory: _MappedRootInventory,
    facet: int,
    face: int,
    vertices: tuple[int, int, int],
    source_triangle: tuple[tuple[Fraction, ...], ...],
    piece: CoplanarTrianglePiece,
    /,
) -> tuple[
    int,
    tuple[int, int],
    tuple[int, int],
    tuple[tuple[tuple[Fraction, ...], ...], ...],
    tuple[int, int],
]:
    from ...geometry._planar_primitives import signed_area2

    root_faces, coordinates, complex_ = (
        inventory.root_faces,
        inventory.coordinates,
        inventory.complex,
    )
    face_owners, domain = inventory.face_owners, inventory.domain
    source_facet_regions, root_cell_regions = (
        inventory.source_facet_regions,
        inventory.root_cell_regions,
    )
    root_cell_ids, root_face_ids = inventory.root_cell_ids, inventory.root_face_ids
    face_vertices = tuple(int(value) for value in root_faces[face])
    if len(face_vertices) == 4:
        corners = tuple(coordinates[value] for value in face_vertices)
        if any(
            corners[0][axis] + corners[2][axis] != corners[1][axis] + corners[3][axis]
            for axis in range(3)
        ):
            raise ValueError(
                "Exact rational source/root overlay requires an affine root-face chart."
            )
    owners = face_owners[face]
    side_regions = tuple(
        None if int(value) < 0 else complex_.region_ids[int(value)]
        for value in source_facet_regions[facet]
    )
    owner_by_region = {
        domain.region_ids[int(root_cell_regions[cell])]: (cell, local)
        for cell, local in owners
    }
    if set(owner_by_region) != {value for value in side_regions if value is not None}:
        raise ValueError(
            "Mapped root face material incidence differs from the source facet."
        )
    order = (0, 1, 2) if signed_area2(piece.source_chart) > 0 else (0, 2, 1)
    cell_ids = [-1, -1]
    local_faces = [-1, -1]
    orientations = [0, 0]
    reference_points = [
        ((Fraction(0),) * 3,) * 3,
        ((Fraction(0),) * 3,) * 3,
    ]
    for side, region in enumerate(side_regions):
        if region is None:
            continue
        cell, local = owner_by_region[region]
        points, orientation = _mapped_root_face_piece(
            inventory,
            source_triangle,
            vertices,
            piece.target_chart,
            order,
            cell,
            local,
            side,
        )
        cell_ids[side] = int(root_cell_ids[cell])
        local_faces[side] = local
        orientations[side] = orientation
        reference_points[side] = points
    return (
        int(root_face_ids[face]),
        (cell_ids[0], cell_ids[1]),
        (local_faces[0], local_faces[1]),
        tuple(reference_points),
        (orientations[0], orientations[1]),
    )


def _mapped_plc_face_overlay(inventory: _MappedRootInventory, /) -> _MappedFaceOverlay:
    from ...geometry._coplanar_overlap import exact_coplanar_triangle_partitions
    from ...geometry._source_boundary_records import rational_chart_tokens

    root_faces = inventory.root_faces
    polygon_source_facets = inventory.polygon_source_facets
    source_facet_regions = inventory.source_facet_regions
    face_owners = inventory.face_owners
    domain = inventory.domain
    root_cell_regions = inventory.root_cell_regions
    root_vertex_rows = inventory.root_vertex_rows
    coordinates = inventory.coordinates
    polygon_root_vertices = inventory.polygon_root_vertices
    complex_ = inventory.complex

    face_regions = {
        row: tuple(
            sorted(
                domain.region_ids[int(root_cell_regions[cell])]
                for cell, _ in face_owners.get(row, ())
            )
        )
        for row in range(len(root_faces))
    }
    required_face_rows = {
        row
        for row, regions in face_regions.items()
        if len(regions) == 1 or (len(regions) == 2 and regions[0] != regions[1])
    }
    source_groups: dict[tuple[str, ...], list[int]] = {}
    for polygon, facet in enumerate(polygon_source_facets):
        regions = tuple(
            sorted(
                complex_.region_ids[int(value)]
                for value in source_facet_regions[facet]
                if int(value) >= 0
            )
        )
        source_groups.setdefault(regions, []).append(polygon)
    target_groups: dict[tuple[str, ...], list[tuple[int, tuple[int, int, int]]]] = {}
    for face in sorted(required_face_rows):
        vertices = tuple(int(value) for value in root_faces[face])
        for index in range(1, len(vertices) - 1):
            target_groups.setdefault(face_regions[face], []).append(
                (face, (vertices[0], vertices[index], vertices[index + 1]))
            )
    if source_groups.keys() != target_groups.keys():
        raise ValueError("Mapped source and root material strata differ.")
    polygon_root_face_ids: list[int] = []
    polygon_root_cell_ids: list[tuple[int, int]] = []
    polygon_root_local_faces: list[tuple[int, int]] = []
    polygon_root_reference_points: list[tuple[tuple[tuple[Fraction, ...], ...], ...]] = []
    polygon_orientation_signs: list[tuple[int, int]] = []
    piece_facets: list[int] = []
    for regions in sorted(source_groups):
        source_rows = source_groups[regions]
        targets = target_groups[regions]
        source_triangles = tuple(
            tuple(
                coordinates[root_vertex_rows[vertex]]
                for vertex in polygon_root_vertices[row]
            )
            for row in source_rows
        )
        target_triangles = tuple(
            tuple(coordinates[vertex] for vertex in vertices) for _, vertices in targets
        )
        pieces = exact_coplanar_triangle_partitions(source_triangles, target_triangles)
        for piece in pieces:
            polygon = source_rows[piece.source_row]
            facet = polygon_source_facets[polygon]
            face, vertices = targets[piece.target_row]
            face_id, cells, locals_, refs, signs = _mapped_overlay_face_piece(
                inventory,
                facet,
                face,
                vertices,
                source_triangles[piece.source_row],
                piece,
            )
            piece_facets.append(facet)
            polygon_root_face_ids.append(face_id)
            polygon_root_cell_ids.append(cells)
            polygon_root_local_faces.append(locals_)
            polygon_root_reference_points.append(refs)
            polygon_orientation_signs.append(signs)
    polygon_root_reference_tokens = tuple(
        tuple(rational_chart_tokens((side,))[0] for side in row)
        for row in polygon_root_reference_points
    )
    return (
        piece_facets,
        polygon_root_face_ids,
        polygon_root_cell_ids,
        polygon_root_local_faces,
        polygon_root_reference_tokens,
        polygon_orientation_signs,
    )


def _mapped_root_face_interval(
    inventory: _MappedRootInventory,
    first: tuple[Fraction, ...],
    last: tuple[Fraction, ...],
    face: int,
    /,
) -> tuple[Fraction, Fraction] | None:
    from ...geometry._planar_primitives import (
        cross,
        plane_key,
        project_polygon,
        projection_axes,
        signed_area2,
    )

    coordinates, root_faces = inventory.coordinates, inventory.root_faces
    vertices = tuple(coordinates[int(row)] for row in root_faces[face])
    plane = plane_key(vertices[:3])
    if any(
        sum(plane[axis] * point[axis] for axis in range(3)) + plane[3] != 0
        for point in (*vertices, first, last)
    ):
        return None
    axes = projection_axes(plane)
    polygon = project_polygon(vertices, axes)
    start, end = project_polygon((first, last), axes)
    sign = 1 if signed_area2(polygon) > 0 else -1
    low, high = Fraction(0), Fraction(1)
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        left, right = sign * cross(a, b, start), sign * cross(a, b, end)
        if left < 0 and right < 0:
            return None
        if left < 0:
            low = max(low, -left / (right - left))
        elif right < 0:
            high = min(high, left / (left - right))
    return (low, high) if low < high else None


def _mapped_root_edge_parameter(
    inventory: _MappedRootInventory, point: tuple[Fraction, ...], edge: int, /
) -> Fraction | None:
    coordinates, root_edges = inventory.coordinates, inventory.root_edges
    first, last = (coordinates[int(row)] for row in root_edges[edge])
    direction = tuple(last[axis] - first[axis] for axis in range(3))
    axis = next((axis for axis, value in enumerate(direction) if value), None)
    if axis is None:
        raise ValueError("Mapped root edge is degenerate.")
    parameter = (point[axis] - first[axis]) / direction[axis]
    if not 0 <= parameter <= 1 or any(
        point[axis] != first[axis] + parameter * direction[axis] for axis in range(3)
    ):
        return None
    return parameter


def _mapped_root_face_reference_point(
    inventory: _MappedRootInventory, point: tuple[Fraction, ...], face: int, cell: int, /
) -> tuple[Fraction, ...]:
    from ...geometry._planar_primitives import plane_key, projection_axes

    root_faces, coordinates = inventory.root_faces, inventory.coordinates
    root_cells, reference_corners = inventory.root_cells, inventory.reference_corners
    vertices = tuple(int(value) for value in root_faces[face])
    physical = tuple(coordinates[vertex] for vertex in vertices)
    plane = plane_key(physical[:3])
    axes = projection_axes(plane)
    last = 3 if len(vertices) == 4 else 2
    origin, first, second = (physical[index] for index in (0, 1, last))
    x, y = axes
    a, b = first[x] - origin[x], second[x] - origin[x]
    c, d = first[y] - origin[y], second[y] - origin[y]
    determinant = a * d - b * c
    if not determinant:
        raise ValueError("Mapped root face has a singular reference chart.")
    u = ((point[x] - origin[x]) * d - (point[y] - origin[y]) * b) / determinant
    v = (a * (point[y] - origin[y]) - c * (point[x] - origin[x])) / determinant
    if any(
        point[axis]
        != origin[axis]
        + u * (first[axis] - origin[axis])
        + v * (second[axis] - origin[axis])
        + (
            u
            * v
            * (
                physical[0][axis]
                - physical[1][axis]
                + physical[2][axis]
                - physical[3][axis]
            )
            if len(vertices) == 4
            else 0
        )
        for axis in range(3)
    ):
        raise ValueError(
            "Mapped root face does not have the required exact affine inverse."
        )
    cell_vertices = tuple(int(value) for value in root_cells[cell])
    refs = tuple(reference_corners[cell_vertices.index(vertex)] for vertex in vertices)
    return tuple(
        refs[0][axis]
        + u * (refs[1][axis] - refs[0][axis])
        + v * (refs[last][axis] - refs[0][axis])
        for axis in range(3)
    )


def _mapped_unsplit_source_segment(
    inventory: _MappedRootInventory,
    segment: np.ndarray,
    root_edge_rows_by_vertices: dict[frozenset[int], int],
    /,
) -> tuple[int, int, int, int, np.ndarray, int]:
    plc_vertex_root_ids, root_vertex_ids = (
        inventory.plc_vertex_root_ids,
        inventory.root_vertex_ids,
    )
    root_faces, root_cells = inventory.root_faces, inventory.root_cells
    root_edge_ids, root_face_ids, root_cell_ids = (
        inventory.root_edge_ids,
        inventory.root_face_ids,
        inventory.root_cell_ids,
    )
    edge_owners, face_owners = inventory.edge_owners, inventory.face_owners
    root_reference_vertices = inventory.root_reference_vertices
    root_edge_local_vertices = inventory.root_edge_local_vertices
    source_endpoints = tuple(int(plc_vertex_root_ids[int(row)]) for row in segment)
    endpoint_set = frozenset(source_endpoints)
    edge_row = root_edge_rows_by_vertices.get(endpoint_set)
    selected_owner: tuple[int, int] | None = None
    dimension = -1
    entity_id = -1
    local_entity = -1
    if edge_row is not None:
        owners = edge_owners.get(edge_row, ())
        if owners:
            selected_owner = min(owners, key=lambda item: int(root_cell_ids[item[0]]))
            dimension = 1
            entity_id = int(root_edge_ids[edge_row])
            local_entity = selected_owner[1]
    if selected_owner is None:
        candidate_faces = tuple(
            row
            for row, face in enumerate(root_faces)
            if endpoint_set.issubset({int(root_vertex_ids[int(value)]) for value in face})
        )
        if len(candidate_faces) == 1:
            face_row = candidate_faces[0]
            owners = face_owners.get(face_row, ())
            if owners:
                selected_owner = min(owners, key=lambda item: int(root_cell_ids[item[0]]))
                dimension = 2
                entity_id = int(root_face_ids[face_row])
                local_entity = selected_owner[1]
    if selected_owner is None:
        cell_candidates = tuple(
            row
            for row, vertices in enumerate(root_cells.tolist())
            if endpoint_set.issubset(
                {int(root_vertex_ids[int(value)]) for value in vertices}
            )
        )
        if len(cell_candidates) == 1:
            selected_owner = (cell_candidates[0], -1)
            dimension = 3
            entity_id = int(root_cell_ids[cell_candidates[0]])
    reference_points = np.zeros((2, 3), dtype=np.float64)
    orientation = 0
    if selected_owner is not None:
        cell_row = selected_owner[0]
        cell_vertex_ids = tuple(
            int(root_vertex_ids[int(row)]) for row in root_cells[cell_row]
        )
        local_positions = tuple(
            cell_vertex_ids.index(value) for value in source_endpoints
        )
        reference_points = root_reference_vertices[
            np.asarray(local_positions, dtype=np.int64)
        ]
        if np.array_equal(reference_points[0], reference_points[1]):
            raise ValueError("Mapped PLC segment collapses in its root-cell chart.")
        if dimension == 1:
            local_edge = root_edge_local_vertices[local_entity]
            oriented_edge = tuple(cell_vertex_ids[position] for position in local_edge)
            if source_endpoints == oriented_edge:
                orientation = 1
            elif source_endpoints == tuple(reversed(oriented_edge)):
                orientation = -1
            else:
                raise ValueError(
                    "Mapped PLC segment endpoint order changed its root edge."
                )
        else:
            delta = reference_points[1] - reference_points[0]
            varying = np.flatnonzero(delta)
            orientation = 1 if delta[int(varying[0])] > 0.0 else -1
    return (
        dimension,
        entity_id,
        -1 if selected_owner is None else int(root_cell_ids[selected_owner[0]]),
        local_entity,
        reference_points,
        orientation,
    )


def _mapped_split_edge_piece(
    inventory: _MappedRootInventory,
    points: tuple[tuple[Fraction, ...], ...],
    intervals: tuple[tuple[int, tuple[Fraction, Fraction]], ...],
    a: Fraction,
    b: Fraction,
    /,
) -> tuple[int, int, int, int, tuple[tuple[Fraction, ...], ...], int]:
    root_edges, root_edge_ids = inventory.root_edges, inventory.root_edge_ids
    edge_owners, face_owners = inventory.edge_owners, inventory.face_owners
    root_cell_ids, root_cells = inventory.root_cell_ids, inventory.root_cells
    root_edge_local_vertices = inventory.root_edge_local_vertices
    coordinates, reference_corners = inventory.coordinates, inventory.reference_corners
    root_face_ids = inventory.root_face_ids
    matching_edges = tuple(
        edge
        for edge in range(len(root_edges))
        if all(
            _mapped_root_edge_parameter(inventory, point, edge) is not None
            for point in points
        )
    )
    if matching_edges:
        edge = min(matching_edges, key=lambda row: int(root_edge_ids[row]))
        cell, local = min(edge_owners[edge], key=lambda item: int(root_cell_ids[item[0]]))
        local_edge = root_edge_local_vertices[local]
        edge_start = coordinates[int(root_cells[cell, local_edge[0]])]
        edge_end = coordinates[int(root_cells[cell, local_edge[1]])]
        direction = tuple(edge_end[axis] - edge_start[axis] for axis in range(3))
        axis = next(axis for axis, value in enumerate(direction) if value)
        parameters = tuple(
            (point[axis] - edge_start[axis]) / direction[axis] for point in points
        )
        refs = tuple(
            tuple(
                reference_corners[local_edge[0]][axis]
                + parameter
                * (
                    reference_corners[local_edge[1]][axis]
                    - reference_corners[local_edge[0]][axis]
                )
                for axis in range(3)
            )
            for parameter in parameters
        )
        dimension, entity = 1, int(root_edge_ids[edge])
        orientation = 1 if parameters[1] > parameters[0] else -1
    else:
        face = min(
            (face for face, (low, high) in intervals if low <= a and b <= high),
            key=lambda row: int(root_face_ids[row]),
        )
        cell, local = min(face_owners[face], key=lambda item: int(root_cell_ids[item[0]]))
        refs = tuple(
            _mapped_root_face_reference_point(inventory, point, face, cell)
            for point in points
        )
        delta = tuple(refs[1][axis] - refs[0][axis] for axis in range(3))
        orientation = 1 if next(value for value in delta if value) > 0 else -1
        dimension, entity = 2, int(root_face_ids[face])
    return dimension, entity, cell, local, refs, orientation


def _mapped_plc_edge_partition(
    inventory: _MappedRootInventory, /
) -> _MappedEdgePartition:
    root_edges = inventory.root_edges
    complex_ = inventory.complex
    root_faces = inventory.root_faces
    coordinates = inventory.coordinates
    root_cell_ids = inventory.root_cell_ids
    root_vertex_ids = inventory.root_vertex_ids
    plc_root_rows = inventory.plc_root_rows
    # Explicit segment maps retain root-edge or root-face charts. Unmapped
    # interior segments remain present in source identity but cannot satisfy a
    # mapped feature request.
    root_edge_rows_by_vertices: dict[frozenset[int], int] = {}
    for edge_row, edge in enumerate(root_edges.tolist()):
        root_edge_rows_by_vertices[
            frozenset(int(root_vertex_ids[int(row)]) for row in edge)
        ] = edge_row
    segment_dimensions: list[int] = []
    segment_entity_ids: list[int] = []
    segment_cell_ids: list[int] = []
    segment_local_entities: list[int] = []
    segment_reference_points: list[tuple[tuple[Fraction, ...], ...] | np.ndarray] = []
    segment_orientations: list[int] = []
    segment_source_rows: list[int] = []
    segment_source_parameters: list[tuple[Fraction, Fraction]] = []
    for source_row, segment in enumerate(
        np.asarray(complex_.plc_edge_vertices, dtype=np.int64)
    ):
        first, last = (coordinates[int(plc_root_rows[int(row)])] for row in segment)
        intervals = tuple(
            (face, interval)
            for face in range(len(root_faces))
            if (interval := _mapped_root_face_interval(inventory, first, last, face))
            is not None
        )
        cuts = sorted(
            {
                Fraction(0),
                Fraction(1),
                *(value for _, interval in intervals for value in interval),
            }
        )
        covered = all(
            any(low <= a and b <= high for _, (low, high) in intervals)
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)
        )
        if covered and intervals:
            for a, b in zip(cuts[:-1], cuts[1:], strict=True):
                points = tuple(
                    tuple(
                        first[axis] + parameter * (last[axis] - first[axis])
                        for axis in range(3)
                    )
                    for parameter in (a, b)
                )
                dimension, entity, cell, local, refs, orientation = (
                    _mapped_split_edge_piece(
                        inventory,
                        points,
                        intervals,
                        a,
                        b,
                    )
                )
                segment_source_rows.append(source_row)
                segment_source_parameters.append((a, b))
                segment_dimensions.append(dimension)
                segment_entity_ids.append(entity)
                segment_cell_ids.append(int(root_cell_ids[cell]))
                segment_local_entities.append(local)
                segment_reference_points.append(refs)
                segment_orientations.append(orientation)
            continue
        dimension, entity_id, cell_id, local_entity, reference_points, orientation = (
            _mapped_unsplit_source_segment(inventory, segment, root_edge_rows_by_vertices)
        )
        segment_dimensions.append(dimension)
        segment_entity_ids.append(entity_id)
        segment_cell_ids.append(cell_id)
        segment_local_entities.append(local_entity)
        segment_reference_points.append(reference_points)
        segment_orientations.append(orientation)
        segment_source_rows.append(source_row)
        segment_source_parameters.append((Fraction(0), Fraction(1)))
    return (
        segment_dimensions,
        segment_entity_ids,
        segment_cell_ids,
        segment_local_entities,
        segment_reference_points,
        segment_orientations,
        segment_source_rows,
        segment_source_parameters,
    )


def _mapped_plc_assemble_relation(
    inventory: _MappedRootInventory,
    faces: _MappedFaceOverlay,
    edges: _MappedEdgePartition,
    /,
) -> _MappedPlcRootRelation:
    from ...geometry._source_boundary_records import rational_chart_tokens

    plc_vertex_root_ids = inventory.plc_vertex_root_ids
    complex_ = inventory.complex
    reference = inventory.reference
    domain = inventory.domain
    (
        segment_dimensions,
        segment_entity_ids,
        segment_cell_ids,
        segment_local_entities,
        segment_reference_points,
        segment_orientations,
        segment_source_rows,
        segment_source_parameters,
    ) = edges
    (
        polygon_source_facets,
        polygon_root_face_ids,
        polygon_root_cell_ids,
        polygon_root_local_faces,
        polygon_root_reference_tokens,
        polygon_orientation_signs,
    ) = faces

    arrays = (
        np.asarray(plc_vertex_root_ids, dtype=np.int64),
        np.asarray(segment_dimensions, dtype=np.int8),
        np.asarray(segment_entity_ids, dtype=np.int64),
        np.asarray(segment_cell_ids, dtype=np.int64),
        np.asarray(segment_local_entities, dtype=np.int32),
        rational_chart_tokens(
            tuple(
                tuple(
                    Fraction.from_float(float(value))
                    if isinstance(value, (float, np.floating))
                    else value
                    for value in point
                )
                for point in row
            )
            for row in segment_reference_points
        ),
        np.asarray(segment_orientations, dtype=np.int8),
        np.asarray(polygon_source_facets, dtype=np.int64),
        np.asarray(polygon_root_face_ids, dtype=np.int64),
        np.asarray(polygon_root_cell_ids, dtype=np.int64).reshape((-1, 2)),
        np.asarray(polygon_root_local_faces, dtype=np.int32).reshape((-1, 2)),
        polygon_root_reference_tokens,
        np.asarray(polygon_orientation_signs, dtype=np.int8).reshape((-1, 2)),
    )
    segment_source_entity_ids = np.asarray(complex_.plc_edge_entity_ids, dtype=np.int64)[
        np.asarray(segment_source_rows, dtype=np.int64)
    ]
    segment_source_keys = tuple(
        reference.source_entity_key_for_scope(1, int(identifier))
        for identifier in segment_source_entity_ids.tolist()
    )
    polygon_source_entity_ids = np.asarray(complex_.facet_entity_ids, dtype=np.int64)[
        np.asarray(polygon_source_facets, dtype=np.int64)
    ]
    polygon_source_keys = tuple(
        reference.source_entity_key_for_scope(2, int(identifier))
        for identifier in polygon_source_entity_ids.tolist()
    )
    segment_source_entity_ids.setflags(write=False)
    polygon_source_entity_ids.setflags(write=False)
    segment_source_entity_keys = tuple(key for key, _ in segment_source_keys)
    segment_source_occurrence_paths = tuple(path for _, path in segment_source_keys)
    polygon_source_entity_keys = tuple(key for key, _ in polygon_source_keys)
    polygon_source_occurrence_paths = tuple(path for _, path in polygon_source_keys)
    for array in arrays:
        if isinstance(array, np.ndarray):
            array.setflags(write=False)
    segment_source_parameter_rational_tokens = tuple(
        (row[0].as_integer_ratio(), row[1].as_integer_ratio())
        for row in segment_source_parameters
    )
    relation_id = canonical_fingerprint(
        {
            "kind": "mapped-plc-root-entity-relation",
            "reference_source": reference.binding_id,
            "mapped_domain": domain.domain_id,
            "plc_vertex_root_ids": arrays[0],
            "segment_root_dimensions": arrays[1],
            "segment_root_entity_ids": arrays[2],
            "segment_root_cell_ids": arrays[3],
            "segment_root_local_entity_indices": arrays[4],
            "segment_root_reference_rational_tokens": arrays[5],
            "segment_orientations": arrays[6],
            "polygon_source_facet_rows": arrays[7],
            "polygon_root_face_ids": arrays[8],
            "polygon_root_cell_ids": arrays[9],
            "polygon_root_local_face_indices": arrays[10],
            "polygon_root_reference_rational_tokens": arrays[11],
            "polygon_orientation_signs": arrays[12],
            "segment_source_entity_ids": segment_source_entity_ids,
            "segment_source_entity_keys": segment_source_entity_keys,
            "segment_source_occurrence_paths": segment_source_occurrence_paths,
            "segment_source_parameter_rational_tokens": segment_source_parameter_rational_tokens,
            "polygon_source_entity_ids": polygon_source_entity_ids,
            "polygon_source_entity_keys": polygon_source_entity_keys,
            "polygon_source_occurrence_paths": polygon_source_occurrence_paths,
        }
    )
    return _MappedPlcRootRelation(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        arrays[4],
        arrays[5],
        arrays[6],
        arrays[7],
        arrays[8],
        arrays[9],
        arrays[10],
        arrays[11],
        arrays[12],
        segment_source_entity_ids,
        segment_source_entity_keys,
        segment_source_occurrence_paths,
        polygon_source_entity_ids,
        polygon_source_entity_keys,
        polygon_source_occurrence_paths,
        segment_source_parameter_rational_tokens,
        relation_id,
    )


def _mapped_root_connectivity(
    mesh: CellMesh, /
) -> tuple[np.ndarray | tuple[np.ndarray, ...], np.ndarray]:
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        root_faces = np.asarray(connectivity.entities[2], dtype=np.int64)
        root_edges = np.asarray(connectivity.entities[1], dtype=np.int64)
    elif isinstance(connectivity, TetrahedralConnectivity):
        root_faces = np.asarray(connectivity.faces, dtype=np.int64)
        root_edges = np.asarray(connectivity.edges, dtype=np.int64)
    elif isinstance(connectivity, HexahedralConnectivity):
        root_faces = np.asarray(connectivity.faces, dtype=np.int64)
        root_edges = np.asarray(connectivity.edges, dtype=np.int64)
    elif isinstance(connectivity, PolyhedralConnectivity):
        face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        root_faces = tuple(
            face_values[int(first) : int(last)]
            for first, last in zip(face_offsets[:-1], face_offsets[1:], strict=True)
        )
        root_edges = np.asarray(connectivity.edges, dtype=np.int64)
    else:
        raise ValueError(
            "Mapped root cells require canonical simplex, tetrahedron, hexahedron, or packed polyhedral incidence."
        )
    return root_faces, root_edges


def _mapped_plc_root_relation(
    reference: NativePlcSource,
    domain: MappedReferenceDomain,
    vertex_root_global_ids: ArrayLike | None,
    /,
) -> _MappedPlcRootRelation:
    complex_ = reference.complex
    mesh = domain.reference_mesh
    root_kinds = tuple(block.cell_kind for block in mesh.blocks)
    if not root_kinds:
        raise ValueError("Mapped PLC/root correspondence requires declared root cells.")
    root_topology = reference_cell_topology(root_kinds[0])
    root_families = {reference_cell_topology(kind).family for kind in root_kinds}
    if (
        (reference.source_id, reference.source_revision)
        != (domain.source_id, domain.source_revision)
        or complex_.region_ids != domain.region_ids
        or len(root_families) != 1
        or root_topology.family not in {"hexahedron", "tetrahedron", "prism"}
    ):
        raise ValueError(
            "Mapped PLC/root correspondence requires one source revision, identical regions, and a homogeneous hexahedral, tetrahedral, or prism root family."
        )
    reference_vertex_roots = domain.reference_domain_vertex_root_ids
    if reference_vertex_roots is None:
        raise ValueError(
            "Mapped PLC/root correspondence requires declared reference-domain vertex/root IDs."
        )

    root_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    root_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(root_vertex_ids.tolist())
    }
    plc_vertex_root_ids, plc_root_rows = _mapped_plc_bind_source_vertices(
        domain,
        vertex_root_global_ids,
        reference,
        complex_,
        reference_vertex_roots,
        mesh,
        root_vertex_rows,
    )
    source_facet_regions, polygon_root_vertices, polygon_source_facets = (
        _mapped_plc_validate_source_facets(
            complex_, plc_vertex_root_ids, domain, reference_vertex_roots
        )
    )

    root_reference_vertices = np.asarray(root_topology.vertices, dtype=np.float64)
    root_face_local_vertices = tuple(
        tuple(int(value) for value in face) for face in root_topology.entities[2]
    )
    root_edge_local_vertices = tuple(
        tuple(int(value) for value in edge) for edge in root_topology.entities[1]
    )
    root_cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks], axis=0
    )
    root_cell_ids = np.asarray(domain.root_cell_ids, dtype=np.int64)
    root_cell_regions = np.asarray(domain.cell_regions, dtype=np.int64)
    if (
        root_cells.shape != (root_cell_ids.size, len(root_topology.vertices))
        or root_cell_regions.shape != root_cell_ids.shape
    ):
        raise ValueError("Mapped root cell identity and region banks are misaligned.")
    root_faces, root_edges = _mapped_root_connectivity(mesh)
    root_face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    root_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    if (
        root_edges.ndim != 2
        or root_face_ids.shape != (len(root_faces),)
        or root_edge_ids.shape != (root_edges.shape[0],)
    ):
        raise ValueError("Mapped root entity connectivity and global IDs are misaligned.")
    face_row_by_vertices = {
        tuple(sorted(int(value) for value in face)): row
        for row, face in enumerate(root_faces)
    }
    edge_row_by_vertices = {
        tuple(sorted(int(value) for value in edge)): row
        for row, edge in enumerate(root_edges.tolist())
    }
    if (
        len(face_row_by_vertices) != len(root_faces)
        or len(edge_row_by_vertices) != root_edges.shape[0]
    ):
        raise ValueError("Mapped root entity incidence is not unique.")
    try:
        cell_face_rows = np.asarray(
            [
                [
                    face_row_by_vertices[
                        tuple(sorted(int(root_cells[cell, local]) for local in face))
                    ]
                    for face in root_face_local_vertices
                ]
                for cell in range(root_cells.shape[0])
            ],
            dtype=np.int64,
        )
        cell_edge_rows = np.asarray(
            [
                [
                    edge_row_by_vertices[
                        tuple(sorted(int(root_cells[cell, local]) for local in edge))
                    ]
                    for edge in root_edge_local_vertices
                ]
                for cell in range(root_cells.shape[0])
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "Mapped root cell incidence is incomplete in its entity banks."
        ) from error
    if cell_face_rows.shape != (
        root_cell_ids.size,
        len(root_face_local_vertices),
    ) or cell_edge_rows.shape != (root_cell_ids.size, len(root_edge_local_vertices)):
        raise ValueError("Mapped root cell/entity relations are misaligned.")
    face_owners: dict[int, list[tuple[int, int]]] = {}
    edge_owners: dict[int, list[tuple[int, int]]] = {}
    for cell_row, (faces_for_cell, edges_for_cell) in enumerate(
        zip(cell_face_rows.tolist(), cell_edge_rows.tolist(), strict=True)
    ):
        for local_face, face_row in enumerate(faces_for_cell):
            face_owners.setdefault(int(face_row), []).append((cell_row, local_face))
        for local_edge, edge_row in enumerate(edges_for_cell):
            edge_owners.setdefault(int(edge_row), []).append((cell_row, local_edge))
    # Intersect in the authored reference geometry.  A source facet is not
    # required to fit one root face: preserve its occurrence and material
    # incidence on every exact overlay piece.

    coordinates = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in np.asarray(mesh.coordinates, dtype=np.float64)
    )
    reference_corners = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in root_reference_vertices
    )
    inventory = _MappedRootInventory(
        reference=reference,
        domain=domain,
        complex=complex_,
        root_faces=root_faces,
        root_edges=root_edges,
        root_cells=root_cells,
        root_face_local_vertices=root_face_local_vertices,
        root_edge_local_vertices=root_edge_local_vertices,
        root_cell_ids=root_cell_ids,
        root_face_ids=root_face_ids,
        root_edge_ids=root_edge_ids,
        root_cell_regions=root_cell_regions,
        root_vertex_ids=root_vertex_ids,
        root_vertex_rows=root_vertex_rows,
        face_owners=face_owners,
        edge_owners=edge_owners,
        coordinates=coordinates,
        reference_corners=reference_corners,
        root_reference_vertices=root_reference_vertices,
        plc_vertex_root_ids=plc_vertex_root_ids,
        plc_root_rows=plc_root_rows,
        polygon_root_vertices=polygon_root_vertices,
        polygon_source_facets=polygon_source_facets,
        source_facet_regions=source_facet_regions,
    )
    faces = _mapped_plc_face_overlay(inventory)
    edges = _mapped_plc_edge_partition(inventory)
    return _mapped_plc_assemble_relation(inventory, faces, edges)


def _prepare_source_scope_bank(
    source_id: str,
    source_revision: str,
    dimension: int,
    records: tuple[tuple[str, str | int, tuple[str, ...], int], ...],
    /,
) -> tuple[np.ndarray, tuple[tuple[str, tuple[str, ...]], ...], str]:
    bank = tuple(
        (
            source_entity_id(
                source_id,
                source_revision,
                kind,
                entity_id,
                occurrence_path=path,
            ),
            path,
            int(scope_id),
        )
        for kind, entity_id, path, scope_id in records
    )
    if (
        len({scope_id for _, _, scope_id in bank}) != len(bank)
        or len({(name, path) for name, path, _ in bank}) != len(bank)
        or any(
            scope_id < 0 or scope_id > np.iinfo(np.int64).max for _, _, scope_id in bank
        )
    ):
        raise ValueError(
            "Source scope IDs must bijectively identify declared entity keys."
        )
    ordered = tuple(sorted(bank, key=lambda row: row[2]))
    identifiers = np.asarray([scope_id for _, _, scope_id in ordered], dtype=np.int64)
    identifiers.setflags(write=False)
    keys = tuple((name, path) for name, path, _ in ordered)
    return (
        identifiers,
        keys,
        _plc_scope_set_id(source_id, source_revision, dimension, ordered),
    )


def _prepare_projected_source_scope_bank(
    source_id: str,
    source_revision: str,
    dimension: int,
    entities: tuple[tuple[str, _ScopeEntityIdentifier, tuple[str, ...]], ...],
    /,
) -> tuple[np.ndarray, tuple[tuple[str, tuple[str, ...]], ...], str]:
    scalar_entities = tuple(
        (kind, _scope_entity_identifier(entity_id), path)
        for kind, entity_id, path in entities
    )
    keys = tuple(
        source_entity_id(
            source_id,
            source_revision,
            kind,
            entity_id,
            occurrence_path=path,
        )
        for kind, entity_id, path in scalar_entities
    )
    paths = tuple(path for _, _, path in scalar_entities)
    numeric_ids = _numeric_entity_ids(f"native-source-scope-{dimension}", keys, paths)
    return _prepare_source_scope_bank(
        source_id,
        source_revision,
        dimension,
        tuple(
            (kind, entity_id, path, numeric_id)
            for (kind, entity_id, path), numeric_id in zip(
                scalar_entities, numeric_ids, strict=True
            )
        ),
    )


def _provided_source_scope_banks(
    owner: object, /
) -> (
    tuple[
        tuple[np.ndarray, ...],
        tuple[tuple[tuple[str, tuple[str, ...]], ...], ...],
        tuple[str, ...],
    ]
    | None
):
    if not isinstance(owner, _SourceEntityScopes):
        return None
    id_banks: list[np.ndarray] = []
    key_banks: list[tuple[tuple[str, tuple[str, ...]], ...]] = []
    set_ids: list[str] = []
    for dimension in range(4):
        raw_ids = owner.scope_entity_ids(dimension)
        identifiers = _entity_ids(
            raw_ids,
            np.asarray(raw_ids).size,
            f"source-scope-dimension-{dimension}",
        )
        order = np.argsort(identifiers, kind="stable")
        identifiers = np.array(identifiers[order], dtype=np.int64, copy=True)
        identifiers.setflags(write=False)
        keys = tuple(
            owner.source_entity_key_for_scope(dimension, int(identifier))
            for identifier in identifiers.tolist()
        )
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(path, tuple)
            or any(not isinstance(part, str) or not part for part in path)
            for name, path in keys
        ) or len(set(keys)) != len(keys):
            raise ValueError(
                "Source scope API must bijectively resolve numeric IDs to typed keys and occurrence paths."
            )
        id_banks.append(identifiers)
        key_banks.append(keys)
        set_ids.append(
            normalized_identifier(owner.scope_entity_set_id(dimension), "entity_set_id")
        )
    return tuple(id_banks), tuple(key_banks), tuple(set_ids)


def _prepare_plc_scope_banks(
    complex_: PiecewiseLinearComplex,
    source_id: str,
    source_revision: str,
    /,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[tuple[tuple[str, tuple[str, ...]], ...], ...],
    tuple[str, ...],
]:
    banks = tuple(_plc_scope_bank(complex_, dimension) for dimension in range(4))
    id_banks: list[np.ndarray] = []
    key_banks: list[tuple[tuple[str, tuple[str, ...]], ...]] = []
    set_ids: list[str] = []
    for dimension, bank in enumerate(banks):
        identifiers = np.asarray(
            [numeric_id for _, _, numeric_id in bank], dtype=np.int64
        )
        identifiers.setflags(write=False)
        id_banks.append(identifiers)
        key_banks.append(tuple((name, path) for name, path, _ in bank))
        set_ids.append(_plc_scope_set_id(source_id, source_revision, dimension, bank))
    return tuple(id_banks), tuple(key_banks), tuple(set_ids)


@final
class NativePlanarSource(StrictModule, NonTrainableState):
    """Planar polygonal source with stable vertex, edge, and region identities."""

    region: PlanarMeshRegion
    embedded: SegmentMesh | None
    vertex_entity_ids: np.ndarray
    edge_entity_ids: np.ndarray
    embedded_vertex_entity_ids: np.ndarray
    embedded_edge_entity_ids: np.ndarray
    region_entity_id: str = eqx.field(static=True)
    region_scope_entity_ids: np.ndarray
    region_scope_entity_set_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    declared_source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        region: PlanarMeshRegion,
        source_revision: str,
        /,
        *,
        embedded: SegmentMesh | None = None,
        region_scope_entity_ids: ArrayLike | None = None,
        vertex_entity_ids: ArrayLike | None = None,
        edge_entity_ids: ArrayLike | None = None,
        embedded_vertex_entity_ids: ArrayLike | None = None,
        embedded_edge_entity_ids: ArrayLike | None = None,
    ) -> None:
        if embedded is not None and not isinstance(embedded, SegmentMesh):
            raise TypeError("embedded must be a SegmentMesh or None.")
        source = normalized_identifier(region.feature_id, "source_id")
        declared_revision = normalized_identifier(source_revision, "source_revision")
        vertices = np.asarray(region.vertices, dtype=np.float64)
        edges = np.asarray(region.edges, dtype=np.int64)
        vertex_ids = _entity_ids(vertex_entity_ids, vertices.shape[0], "planar-vertex")
        edge_ids = _entity_ids(edge_entity_ids, edges.shape[0], "planar-edge")
        if embedded is None:
            if (
                embedded_vertex_entity_ids is not None
                or embedded_edge_entity_ids is not None
            ):
                raise ValueError("Embedded entity IDs require an embedded SegmentMesh.")
            embedded_vertex_ids = np.empty((0,), dtype=np.int64)
            embedded_edge_ids = np.empty((0,), dtype=np.int64)
        else:
            embedded_points = np.asarray(embedded.vertices, dtype=np.float64)
            embedded_edges = np.asarray(embedded.edges, dtype=np.int64)
            embedded_vertex_ids = _entity_ids(
                embedded_vertex_entity_ids,
                embedded_points.shape[0],
                f"embedded-vertex:{embedded.source_id}",
            )
            embedded_edge_ids = _entity_ids(
                embedded_edge_entity_ids,
                embedded_edges.shape[0],
                f"embedded-edge:{embedded.source_id}",
            )
        embedded_vertex_ids.setflags(write=False)
        embedded_edge_ids.setflags(write=False)
        embedded_vertex_scope_records: tuple[
            tuple[str, str | int, tuple[str, ...], int], ...
        ] = ()
        embedded_edge_scope_records: tuple[
            tuple[str, str | int, tuple[str, ...], int], ...
        ] = ()
        if embedded is not None:
            embedded_vertex_scope_records = tuple(
                ("embedded-vertex", int(identifier), (), int(identifier))
                for identifier in embedded_vertex_ids.tolist()
            )
            embedded_edge_scope_records = tuple(
                ("embedded-edge", int(identifier), (), int(identifier))
                for identifier in embedded_edge_ids.tolist()
            )
        revision = canonical_fingerprint(
            {
                "kind": "native-planar-source-revision",
                "source": source,
                "declared_revision": declared_revision,
                "region": array_tree_fingerprint(
                    (
                        vertices,
                        edges,
                        np.asarray(region.loop_offsets),
                        vertex_ids,
                        edge_ids,
                    )
                ),
                "embedded": None
                if embedded is None
                else (
                    embedded.source_id,
                    array_tree_fingerprint((embedded_points, embedded_edges)),
                    array_tree_fingerprint((embedded_vertex_ids, embedded_edge_ids)),
                ),
            }
        )
        region_id = source_entity_id(source, revision, "region", source)
        raw_region_scope_ids = (
            np.asarray(
                _numeric_entity_ids("native-planar-region", (region_id,), ((),)),
                dtype=np.int64,
            )
            if region_scope_entity_ids is None
            else np.asarray(region_scope_entity_ids)
        )
        if (
            raw_region_scope_ids.shape != (1,)
            or not np.issubdtype(raw_region_scope_ids.dtype, np.integer)
            or np.issubdtype(raw_region_scope_ids.dtype, np.bool_)
            or int(raw_region_scope_ids[0]) < 0
            or int(raw_region_scope_ids[0]) > np.iinfo(np.int64).max
        ):
            raise ValueError(
                "region_scope_entity_ids must be one source-owned integer ID."
            )
        region_scope_ids = np.array(raw_region_scope_ids, dtype=np.int64, copy=True)
        region_scope_ids.setflags(write=False)
        scope_records = (
            tuple(
                ("vertex", int(identifier), (), int(identifier))
                for identifier in vertex_ids.tolist()
            )
            + embedded_vertex_scope_records,
            tuple(
                ("edge", int(identifier), (), int(identifier))
                for identifier in edge_ids.tolist()
            )
            + embedded_edge_scope_records,
            (("region", source, (), int(region_scope_ids[0])),),
            (),
        )
        prepared_banks = tuple(
            _prepare_source_scope_bank(source, revision, dimension, records)
            for dimension, records in enumerate(scope_records)
        )
        scope_id_banks = tuple(bank[0] for bank in prepared_banks)
        scope_key_banks = tuple(bank[1] for bank in prepared_banks)
        scope_set_ids = tuple(bank[2] for bank in prepared_banks)
        region_scope_set_id = scope_set_ids[2]
        self.region = region
        self.embedded = embedded
        self.vertex_entity_ids = vertex_ids
        self.edge_entity_ids = edge_ids
        self.embedded_vertex_entity_ids = embedded_vertex_ids
        self.embedded_edge_entity_ids = embedded_edge_ids
        self.region_entity_id = region_id
        self.region_scope_entity_ids = region_scope_ids
        self.region_scope_entity_set_id = region_scope_set_id
        self.scope_entity_id_banks = scope_id_banks
        self.scope_entity_key_banks = scope_key_banks
        self.scope_entity_set_ids = scope_set_ids
        self.source_id = source
        self.source_revision = revision
        self.declared_source_revision = declared_revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-planar-source",
                "source": source,
                "revision": revision,
                "declared_revision": declared_revision,
                "vertices": array_tree_fingerprint(vertices),
                "edges": array_tree_fingerprint(edges),
                "loops": array_tree_fingerprint(np.asarray(region.loop_offsets)),
                "vertex_ids": array_tree_fingerprint(vertex_ids),
                "edge_ids": array_tree_fingerprint(edge_ids),
                "region_entity_id": region_id,
                "region_scope_entity_ids": array_tree_fingerprint(region_scope_ids),
                "region_scope_entity_set_id": region_scope_set_id,
                "scope_entity_key_banks": scope_key_banks,
                "scope_entity_set_ids": scope_set_ids,
                "embedded": None
                if embedded is None
                else [
                    embedded.source_id,
                    array_tree_fingerprint(np.asarray(embedded.vertices)),
                    array_tree_fingerprint(np.asarray(embedded.edges)),
                    array_tree_fingerprint(embedded_vertex_ids),
                    array_tree_fingerprint(embedded_edge_ids),
                ],
            }
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Planar source dimension must be in 0..3.")
        return self.scope_entity_id_banks[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Planar source dimension must be in 0..3.")
        return self.scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Planar source dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("entity_id must be one stable integer scope ID.")
        identifiers = self.scope_entity_id_banks[dimension]
        position = int(np.searchsorted(identifiers, entity_id))
        if position >= identifiers.size or int(identifiers[position]) != entity_id:
            raise ValueError("entity_id is absent from this planar source dimension.")
        return self.scope_entity_key_banks[dimension][position]

    def _curve_source_arrays(
        self,
        /,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(self.region.vertices, dtype=np.float64)
        edges = np.asarray(self.region.edges, dtype=np.int64)
        vertex_ids = np.asarray(self.vertex_entity_ids, dtype=np.int64)
        edge_ids = np.asarray(self.edge_entity_ids, dtype=np.int64)
        if self.embedded is not None:
            embedded_points = np.asarray(self.embedded.vertices, dtype=np.float64)
            embedded_edges = np.asarray(self.embedded.edges, dtype=np.int64)
            edges = np.concatenate((edges, embedded_edges + points.shape[0]), axis=0)
            points = np.concatenate((points, embedded_points), axis=0)
            vertex_ids = np.concatenate(
                (vertex_ids, np.asarray(self.embedded_vertex_entity_ids, dtype=np.int64))
            )
            edge_ids = np.concatenate(
                (edge_ids, np.asarray(self.embedded_edge_entity_ids, dtype=np.int64))
            )
        return points, edges, vertex_ids, edge_ids

    def validate_curve_boundary_query(
        self,
        query: ParametricCurveBoundarySource,
        requested_entity_ids: ArrayLike,
        /,
    ) -> None:
        from ...geometry._curve_boundary import ParametricCurveBoundarySource

        if not isinstance(query, ParametricCurveBoundarySource):
            raise TypeError(
                "Planar curve validation requires ParametricCurveBoundarySource."
            )
        if (query.source_id, query.source_revision) != (
            self.source_id,
            self.source_revision,
        ):
            raise ValueError("Planar curve query must bind this source revision.")
        query.require_current()
        raw_ids = np.asarray(requested_entity_ids)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
            raise TypeError("requested_entity_ids must be an integer vector.")
        identifiers = np.asarray(raw_ids, dtype=np.int64)
        if identifiers.size == 0 or np.unique(identifiers).size != identifiers.size:
            raise ValueError(
                "requested_entity_ids must be a nonempty set of unique planar edge IDs."
            )
        points, edges, vertex_ids, edge_ids = self._curve_source_arrays()
        if np.any(~np.isin(identifiers, self.scope_entity_id_banks[1])):
            raise ValueError("Requested planar edge IDs are absent from this source.")
        selected_rows = np.flatnonzero(np.isin(edge_ids, identifiers))
        expected: dict[str, tuple[int, tuple[str, tuple[str, ...]]]] = {}
        for row in selected_rows.tolist():
            identifier = int(edge_ids[row])
            key = self.source_entity_key_for_scope(1, identifier)
            if key[0] in expected:
                raise ValueError("Planar source edge chart identities must be unique.")
            expected[key[0]] = (row, key)
        actual = {
            chart_id: (entity_id, tuple(occurrence), tuple(parameter_range))
            for entity_id, occurrence, chart_id, parameter_range in zip(
                query.curve_entity_ids,
                query.occurrence_paths,
                query.chart_ids,
                query.parameter_ranges,
                strict=True,
            )
        }
        expected_records = {
            chart_id: (key[0], key[1], (0.0, 1.0))
            for chart_id, (_, key) in expected.items()
        }
        if actual != expected_records:
            raise ValueError(
                "Planar curve query must retain every selected authored edge and full range."
            )
        for query_row, chart_id in enumerate(query.chart_ids):
            edge_row, key = expected[chart_id]
            first, last = (int(value) for value in edges[edge_row])
            curve = query.curves[query_row]
            if not isinstance(curve, SegmentChartCurve):
                raise TypeError(
                    "Planar source curve charts must be authored segment charts."
                )
            first_exact = tuple(
                Fraction.from_float(float(value)) for value in points[first]
            )
            last_exact = tuple(
                Fraction.from_float(float(value)) for value in points[last]
            )
            if (
                curve.first != 0.0
                or curve.last != 1.0
                or not np.array_equal(np.asarray(curve.start), points[first])
                or not np.array_equal(np.asarray(curve.end), points[last])
                or curve.origin_exact != first_exact
                or curve.direction_exact
                != tuple(b - a for a, b in zip(first_exact, last_exact, strict=True))
            ):
                raise ValueError("Planar curve query changed its authored line geometry.")
            expected_endpoint_keys = (
                self.source_entity_key_for_scope(0, int(vertex_ids[first])),
                self.source_entity_key_for_scope(0, int(vertex_ids[last])),
            )
            expected_bounds = tuple(
                (tuple(float(value) for value in points[row]),) * 2
                for row in (first, last)
            )
            if (
                query.endpoint_source_keys[query_row] != expected_endpoint_keys
                or query.endpoint_point_bounds[query_row] != expected_bounds
            ):
                raise ValueError(
                    "Planar curve query changed its authored endpoint identities or points."
                )

    def validate_point_boundary_query(
        self,
        query: ParametricCurveBoundarySource,
        requested_point_ids: ArrayLike,
        /,
    ) -> None:
        from ...geometry._curve_boundary import ParametricCurveBoundarySource

        if not isinstance(query, ParametricCurveBoundarySource):
            raise TypeError(
                "Planar point validation requires ParametricCurveBoundarySource."
            )
        if (query.source_id, query.source_revision) != (
            self.source_id,
            self.source_revision,
        ):
            raise ValueError("Planar point query must bind this source revision.")
        query.require_current()
        points, edges, vertex_ids, edge_ids = self._curve_source_arrays()
        scope_edge_ids = np.asarray(self.scope_entity_id_banks[1], dtype=np.int64)
        scope_keys = tuple(
            self.source_entity_key_for_scope(1, int(identifier))
            for identifier in scope_edge_ids.tolist()
        )
        id_by_key = {
            key: int(identifier)
            for identifier, key in zip(scope_edge_ids.tolist(), scope_keys, strict=True)
        }
        try:
            requested_edge_ids = np.asarray(
                sorted(
                    {
                        id_by_key[key]
                        for key in zip(
                            query.curve_entity_ids, query.occurrence_paths, strict=True
                        )
                    }
                ),
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "Planar point query names an unauthored curve chart."
            ) from error
        self.validate_curve_boundary_query(query, requested_edge_ids)
        expected_point_bounds: dict[
            tuple[str, tuple[str, ...]], tuple[np.ndarray, np.ndarray]
        ] = {}
        edge_row_by_key = {
            self.source_entity_key_for_scope(1, int(identifier)): row
            for row, identifier in enumerate(edge_ids.tolist())
        }
        for query_row, (entity_id, occurrence, parameter_range) in enumerate(
            zip(
                query.curve_entity_ids,
                query.occurrence_paths,
                query.parameter_ranges,
                strict=True,
            )
        ):
            edge_row = edge_row_by_key[(entity_id, tuple(occurrence))]
            first, last = (int(value) for value in edges[edge_row])
            expected_keys = (
                self.source_entity_key_for_scope(0, int(vertex_ids[first])),
                self.source_entity_key_for_scope(0, int(vertex_ids[last])),
            )
            expected_bounds = tuple(
                (tuple(float(value) for value in points[row]),) * 2
                for row in (first, last)
            )
            if (
                query.endpoint_source_keys[query_row] != expected_keys
                or query.endpoint_point_bounds[query_row] != expected_bounds
            ):
                raise ValueError("Planar point query changed an authored edge endpoint.")
            for side, (key, parameter, point) in enumerate(
                zip(expected_keys, parameter_range, (first, last), strict=True)
            ):
                point_value = points[point]
                lower, upper = query.chart_point_bounds(
                    query.chart_ids[query_row], float(parameter)
                )
                if not np.array_equal(lower, point_value) or not np.array_equal(
                    upper, point_value
                ):
                    raise ValueError(
                        "Planar endpoint enclosure differs from its authored coordinate."
                    )
                previous = expected_point_bounds.get(key)
                if previous is not None and (
                    not np.array_equal(previous[0], lower)
                    or not np.array_equal(previous[1], upper)
                ):
                    raise ValueError(
                        "Repeated planar point identities have different coordinates."
                    )
                expected_point_bounds[key] = (lower, upper)
        expected_keys = tuple(sorted(expected_point_bounds))
        actual_keys = tuple(
            zip(query.point_entity_ids, query.point_occurrence_paths, strict=True)
        )
        if expected_keys != actual_keys:
            raise ValueError(
                "Planar point query point bank does not match its source edges."
            )
        raw_point_ids = np.asarray(requested_point_ids)
        if raw_point_ids.ndim != 1 or raw_point_ids.dtype.kind not in "iu":
            raise TypeError("requested_point_ids must be an integer vector.")
        if (
            raw_point_ids.dtype.kind == "u"
            and raw_point_ids.size
            and np.any(raw_point_ids > np.iinfo(np.int64).max)
        ):
            raise ValueError("requested_point_ids must fit int64.")
        point_ids = np.asarray(raw_point_ids, dtype=np.int64)
        if point_ids.size == 0 or np.unique(point_ids).size != point_ids.size:
            raise ValueError(
                "requested_point_ids must be a nonempty set of unique point IDs."
            )
        requested = tuple(
            self.source_entity_key_for_scope(0, int(identifier))
            for identifier in point_ids.tolist()
        )
        if not requested or any(key not in expected_point_bounds for key in requested):
            raise ValueError(
                "Requested planar point scope is not an authored edge endpoint."
            )
        for key, (lower, upper) in expected_point_bounds.items():
            actual_lower, actual_upper = query.source_point_bounds(*key)
            if not np.array_equal(lower, actual_lower) or not np.array_equal(
                upper, actual_upper
            ):
                raise ValueError("Planar point query changed its source point enclosure.")

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[NativePlanarSource], value: NativePlanarSource, /
    ) -> NativePlanarSource:
        if type(value) is not cls:
            raise TypeError("stored_fields must have the registered native planar type.")
        fresh = cls(
            value.region,
            value.declared_source_revision,
            embedded=value.embedded,
            region_scope_entity_ids=value.region_scope_entity_ids,
            vertex_entity_ids=value.vertex_entity_ids,
            edge_entity_ids=value.edge_entity_ids,
            embedded_vertex_entity_ids=value.embedded_vertex_entity_ids,
            embedded_edge_entity_ids=value.embedded_edge_entity_ids,
        )
        if (
            fresh.source_id,
            fresh.source_revision,
            fresh.binding_id,
        ) != (value.source_id, value.source_revision, value.binding_id):
            raise ValueError("binding_id does not match the stored planar source.")
        return fresh


@final
class NativeCurveSource(StrictModule, NonTrainableState):
    """Parametric curve charts or a source segment network.

    Repeated atlas entity IDs denote multiple charts of one semantic source
    curve. Such charts require explicit persistent chart IDs and parameter
    ranges unless a revision-bound MeshingDomain supplies their topology.
    SegmentMesh vertices require authored IDs; atlas endpoints retain their
    source keys or resolve through ``gap_domain``. Coordinates never define
    endpoint identity.
    """

    curves: BoundaryAtlas | SegmentMesh
    ambient_dimension: int = eqx.field(static=True)
    curve_entity_ids: np.ndarray
    vertex_entity_ids: np.ndarray
    vertex_entity_keys: tuple[str | int, ...] = eqx.field(static=True)
    vertex_source_entity_ids: tuple[str, ...] = eqx.field(static=True)
    vertex_scope_entity_set_id: str = eqx.field(static=True)
    curve_scope_entity_set_id: str = eqx.field(static=True)
    chart_ids: tuple[str, ...] = eqx.field(static=True)
    chart_parameter_ranges: np.ndarray | None
    chart_endpoint_ids: tuple[tuple[str | None, str | None], ...] = eqx.field(static=True)
    chart_scope_entity_ids: np.ndarray
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    declared_source_revision: str = eqx.field(static=True)
    gap_domain: MeshingDomain | None
    binding_id: str = eqx.field(static=True)
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curves: BoundaryAtlas | SegmentMesh,
        source_revision: str,
        /,
        *,
        gap_domain: MeshingDomain | None = None,
        curve_entity_ids: ArrayLike | None = None,
        vertex_entity_ids: ArrayLike | None = None,
        chart_ids: tuple[str, ...] | None = None,
        chart_parameter_ranges: ArrayLike | None = None,
        chart_endpoint_ids: tuple[tuple[str | None, str | None], ...] | None = None,
    ) -> None:
        declared_revision = normalized_identifier(source_revision, "source_revision")
        revision = declared_revision
        if isinstance(curves, BoundaryAtlas):
            if curves.reference_dimension != 1 or curves.num_charts == 0:
                raise ValueError("BoundaryAtlas curve sources need nonempty 1D charts.")
            ambient = curves.ambient_dimension
            source = normalized_identifier(curves.source_id, "source_id")
            entity_ids = _entity_ids(
                curves.source_entity_ids,
                curves.num_charts,
                "atlas-curve",
                allow_repeated=True,
            )
        elif isinstance(curves, SegmentMesh):
            ambient = 2
            source = normalized_identifier(curves.source_id, "source_id")
            entity_ids = _entity_ids(
                curve_entity_ids,
                curves.edges.shape[0],
                "segment-curve",
                allow_repeated=True,
            )
        else:
            raise TypeError("curves must be BoundaryAtlas or SegmentMesh.")
        if isinstance(curves, BoundaryAtlas) and curve_entity_ids is not None:
            entity_ids = _entity_ids(
                curve_entity_ids,
                curves.num_charts,
                "atlas-curve",
                allow_repeated=True,
            )
        repeated = np.unique(entity_ids).size != entity_ids.size
        ranges = None
        if chart_parameter_ranges is not None:
            ranges = np.asarray(chart_parameter_ranges, dtype=np.float64)
            if (
                ranges.shape != (entity_ids.size, 2)
                or not np.all(np.isfinite(ranges))
                or np.any(ranges[:, 0] == ranges[:, 1])
            ):
                raise ValueError(
                    "chart_parameter_ranges must contain two distinct finite source parameters per chart."
                )
            ranges = np.array(ranges, dtype=np.float64, copy=True)
            ranges.setflags(write=False)
        if chart_ids is None:
            if repeated:
                raise ValueError(
                    "Repeated curve entity IDs require persistent chart_ids and source parameter ranges."
                )
            chart_names = tuple(f"{source}:curve:{int(value)}" for value in entity_ids)
        else:
            chart_names = tuple(
                normalized_identifier(name, "chart_id") for name in chart_ids
            )
            if len(chart_names) != entity_ids.size or len(set(chart_names)) != len(
                chart_names
            ):
                raise ValueError(
                    "chart_ids must be unique, with one ID per source chart."
                )
        if repeated and ranges is None:
            raise ValueError(
                "Repeated curve entity IDs require source-defined chart_parameter_ranges."
            )
        raw_endpoint_ids: tuple[tuple[str | None, str | None], ...] = (
            () if chart_endpoint_ids is None else tuple(chart_endpoint_ids)
        )
        if raw_endpoint_ids and (
            len(raw_endpoint_ids) != entity_ids.size
            or any(len(pair) != 2 for pair in raw_endpoint_ids)
        ):
            raise ValueError(
                "chart_endpoint_ids must contain two source vertex IDs per chart."
            )
        endpoint_ids: tuple[tuple[str | None, str | None], ...] = tuple(
            (pair[0], pair[1]) for pair in raw_endpoint_ids
        )
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for pair in endpoint_ids
            for value in pair
        ):
            raise ValueError(
                "chart_endpoint_ids must contain nonempty source vertex IDs."
            )
        if gap_domain is None:
            revision = canonical_fingerprint(
                {
                    "kind": "native-curve-source-revision",
                    "source": source,
                    "declared_revision": declared_revision,
                    "curves": array_tree_fingerprint(curves),
                    "curve_entity_ids": array_tree_fingerprint(entity_ids),
                    "chart_ids": chart_names,
                    "chart_parameter_ranges": None
                    if ranges is None
                    else array_tree_fingerprint(ranges),
                    "chart_endpoint_ids": endpoint_ids,
                    "vertex_entity_ids": None
                    if vertex_entity_ids is None
                    else array_tree_fingerprint(np.asarray(vertex_entity_ids)),
                }
            )
        if gap_domain is not None:
            if not isinstance(gap_domain, MeshingDomain):
                raise TypeError("gap_domain must be a MeshingDomain or None.")
            if (gap_domain.source_id, gap_domain.source_revision) != (source, revision):
                raise ValueError("gap_domain must bind the curve source ID and revision.")
            if endpoint_ids:
                domain_vertex_slot_by_key = {
                    gap_domain.entity_id(0, slot): slot
                    for slot in range(len(gap_domain.source_indices[0]))
                }
                resolved_pairs: list[tuple[str | None, str | None]] = []
                for pair in endpoint_ids:
                    resolved: list[str | None] = []
                    for endpoint in pair:
                        if endpoint is None:
                            resolved.append(None)
                            continue
                        direct_slot = domain_vertex_slot_by_key.get(endpoint)
                        raw_identity = source_entity_id(
                            source,
                            revision,
                            gap_domain.source_kinds[0],
                            endpoint,
                        )
                        raw_slot = domain_vertex_slot_by_key.get(raw_identity)
                        if (
                            direct_slot is not None
                            and raw_slot is not None
                            and direct_slot != raw_slot
                        ):
                            raise ValueError(
                                "Curve endpoint key has conflicting canonical and authored-source resolutions."
                            )
                        slot = direct_slot if direct_slot is not None else raw_slot
                        if slot is None:
                            raise ValueError(
                                "Curve endpoint key is absent from gap_domain's declared source corners."
                            )
                        resolved.append(gap_domain.entity_id(0, slot))
                    resolved_pairs.append((resolved[0], resolved[1]))
                endpoint_ids = tuple(resolved_pairs)
            else:
                resolved_endpoints: list[tuple[str | None, str | None]] = []
                for curve_id in entity_ids.tolist():
                    curve_identity = source_entity_id(
                        source,
                        revision,
                        gap_domain.source_kinds[1],
                        int(curve_id),
                    )
                    curve_slot = gap_domain.resolve_entity_id(1, curve_identity)
                    curve = gap_domain.curves[curve_slot]
                    resolved_endpoints.append(
                        (
                            gap_domain.entity_id(0, curve.start),
                            gap_domain.entity_id(0, curve.end),
                        )
                    )
                endpoint_ids = tuple(resolved_endpoints)

            domain_ids = np.asarray(gap_domain.scope_entity_ids(0), dtype=np.int64)
            domain_keys = tuple(
                gap_domain.source_entity_key_for_scope(0, int(identifier))[0]
                for identifier in domain_ids.tolist()
            )
            if isinstance(curves, SegmentMesh):
                vertex_ids = _entity_ids(
                    vertex_entity_ids, curves.vertices.shape[0], "segment-vertex"
                )
                domain_key_by_id = {
                    int(identifier): key
                    for identifier, key in zip(
                        domain_ids.tolist(), domain_keys, strict=True
                    )
                }
                if any(
                    int(identifier) not in domain_key_by_id for identifier in vertex_ids
                ):
                    raise ValueError(
                        "SegmentMesh vertex IDs must resolve in gap_domain's declared corner bank."
                    )
                vertex_keys = tuple(
                    domain_key_by_id[int(identifier)]
                    for identifier in vertex_ids.tolist()
                )
            else:
                if vertex_entity_ids is not None:
                    explicit = _entity_ids(
                        vertex_entity_ids, len(domain_ids), "atlas-vertex"
                    )
                    if not np.array_equal(explicit, domain_ids):
                        raise ValueError(
                            "Explicit atlas vertex IDs must equal the bound gap_domain corner bank."
                        )
                vertex_ids = np.array(domain_ids, dtype=np.int64, copy=True)
                vertex_keys = domain_keys
            vertex_source_ids = tuple(
                key
                if isinstance(key, str)
                else source_entity_id(source, revision, "vertex", key)
                for key in vertex_keys
            )
        elif isinstance(curves, SegmentMesh):
            vertex_ids = _entity_ids(
                vertex_entity_ids, curves.vertices.shape[0], "segment-vertex"
            )
            vertex_keys = tuple(int(value) for value in vertex_ids.tolist())
            vertex_source_ids = tuple(
                source_entity_id(source, revision, "vertex", key) for key in vertex_keys
            )
        else:
            vertex_keys = tuple(
                sorted(
                    {
                        value
                        for pair in endpoint_ids
                        for value in pair
                        if value is not None
                    }
                )
            )
            vertex_source_ids = tuple(
                source_entity_id(source, revision, "vertex", key) for key in vertex_keys
            )
            if vertex_entity_ids is None:
                vertex_ids = np.asarray(
                    _numeric_entity_ids(
                        "native-curve-vertex",
                        vertex_source_ids,
                        tuple(() for _ in vertex_source_ids),
                    ),
                    dtype=np.int64,
                )
            else:
                vertex_ids = _entity_ids(
                    vertex_entity_ids, len(vertex_keys), "atlas-vertex"
                )
        vertex_ids = np.array(vertex_ids, dtype=np.int64, copy=True)
        vertex_ids.setflags(write=False)
        scope_id_banks: list[np.ndarray] = []
        scope_key_banks: list[tuple[tuple[str, tuple[str, ...]], ...]] = []
        scope_set_ids: list[str] = []
        if gap_domain is not None:
            for dimension in range(4):
                raw_ids = gap_domain.scope_entity_ids(dimension)
                identifiers = np.array(raw_ids, dtype=np.int64, copy=True)
                keys = tuple(
                    gap_domain.source_entity_key_for_scope(dimension, int(identifier))
                    for identifier in identifiers.tolist()
                )
                identifiers.setflags(write=False)
                scope_id_banks.append(identifiers)
                scope_key_banks.append(keys)
                scope_set_ids.append(gap_domain.scope_entity_set_id(dimension))

            domain_curve_scope_ids = np.asarray(
                gap_domain.scope_entity_ids(1), dtype=np.int64
            )
            domain_curve_scope_id_by_key = dict(
                zip(
                    scope_key_banks[1],
                    domain_curve_scope_ids.tolist(),
                    strict=True,
                )
            )

            chart_scope_ids = np.empty((entity_ids.size,), dtype=np.int64)
            for row, (curve_id, endpoints) in enumerate(
                zip(entity_ids.tolist(), endpoint_ids, strict=True)
            ):
                candidates = [
                    slot
                    for slot, source_curve_id in enumerate(gap_domain.source_indices[1])
                    if int(source_curve_id) == int(curve_id)
                ]
                declared = {value for value in endpoints if value is not None}
                if declared:
                    candidates = [
                        slot
                        for slot in candidates
                        if declared.issubset(
                            {
                                gap_domain.entity_id(0, gap_domain.curves[slot].start),
                                gap_domain.entity_id(0, gap_domain.curves[slot].end),
                            }
                        )
                    ]
                if len(candidates) != 1:
                    raise ValueError(
                        "Curve chart must resolve to exactly one source curve occurrence in gap_domain."
                    )
                candidate = candidates[0]
                candidate_key = (
                    gap_domain.entity_id(1, candidate),
                    gap_domain.source_occurrences[1][candidate],
                )
                try:
                    chart_scope_ids[row] = domain_curve_scope_id_by_key[candidate_key]
                except KeyError as error:
                    raise ValueError(
                        "Curve chart is absent from gap_domain's canonical scope bank."
                    ) from error
        else:
            vertex_bank = tuple(
                sorted(
                    zip(
                        vertex_source_ids,
                        (tuple() for _ in vertex_source_ids),
                        (int(value) for value in vertex_ids.tolist()),
                        strict=True,
                    ),
                    key=lambda bank_row: bank_row[2],
                )
            )
            curve_scope_values = tuple(
                int(value) for value in np.unique(entity_ids).tolist()
            )
            curve_bank = tuple(
                (
                    source_entity_id(source, revision, "curve", curve_id),
                    (),
                    curve_id,
                )
                for curve_id in curve_scope_values
            )
            banks = (
                vertex_bank,
                curve_bank,
                (),
                (),
            )
            for dimension, bank in enumerate(banks):
                identifiers = np.asarray([value for _, _, value in bank], dtype=np.int64)
                identifiers.setflags(write=False)
                scope_id_banks.append(identifiers)
                scope_key_banks.append(tuple((name, path) for name, path, _ in bank))
                scope_set_ids.append(_plc_scope_set_id(source, revision, dimension, bank))
            chart_scope_ids = np.array(entity_ids, dtype=np.int64, copy=True)
        chart_scope_ids.setflags(write=False)
        vertex_scope_set_id = scope_set_ids[0]
        curve_scope_set_id = scope_set_ids[1]
        if repeated and not endpoint_ids and gap_domain is None:
            raise ValueError(
                "Repeated curve charts require source endpoint IDs or a revision-bound MeshingDomain."
            )
        self.vertex_entity_ids = vertex_ids
        self.vertex_source_entity_ids = vertex_source_ids
        self.vertex_entity_keys = vertex_keys
        self.vertex_scope_entity_set_id = vertex_scope_set_id
        self.curve_scope_entity_set_id = curve_scope_set_id
        self.curves = curves
        self.ambient_dimension = ambient
        self.curve_entity_ids = entity_ids
        self.chart_ids = chart_names
        self.chart_parameter_ranges = ranges
        self.chart_endpoint_ids = endpoint_ids
        self.chart_scope_entity_ids = chart_scope_ids
        self.source_id = source
        self.source_revision = revision
        self.declared_source_revision = declared_revision
        self.gap_domain = gap_domain
        self.scope_entity_id_banks = tuple(scope_id_banks)
        self.scope_entity_key_banks = tuple(scope_key_banks)
        self.scope_entity_set_ids = tuple(scope_set_ids)
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-curve-source",
                "source": source,
                "revision": revision,
                "declared_revision": declared_revision,
                "curves": array_tree_fingerprint(curves),
                "curve_entity_ids": array_tree_fingerprint(entity_ids),
                "chart_ids": chart_names,
                "chart_parameter_ranges": None
                if ranges is None
                else array_tree_fingerprint(ranges),
                "chart_endpoint_ids": endpoint_ids,
                "vertex_entity_ids": array_tree_fingerprint(vertex_ids),
                "vertex_entity_keys": vertex_keys,
                "vertex_source_entity_ids": vertex_source_ids,
                "vertex_scope_entity_set_id": vertex_scope_set_id,
                "chart_scope_entity_ids": array_tree_fingerprint(chart_scope_ids),
                "curve_scope_entity_set_id": curve_scope_set_id,
                "scope_entity_key_banks": tuple(scope_key_banks),
                "scope_entity_set_ids": tuple(scope_set_ids),
                "gap_domain": None if gap_domain is None else gap_domain.domain_id,
            }
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Curve source dimension must be in 0..3.")
        return self.scope_entity_id_banks[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Curve source dimension must be in 0..3.")
        return self.scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Curve source dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("entity_id must be one stable integer scope ID.")
        identifiers = self.scope_entity_id_banks[dimension]
        position = int(np.searchsorted(identifiers, entity_id))
        if position >= identifiers.size or int(identifiers[position]) != entity_id:
            raise ValueError("entity_id is absent from this curve source dimension.")
        return self.scope_entity_key_banks[dimension][position]

    def validate_curve_boundary_query(
        self,
        query: ParametricCurveBoundarySource,
        requested_entity_ids: ArrayLike,
        /,
    ) -> None:
        """Require query charts to match every authored range of selected curves."""
        from ...geometry._curve_boundary import ParametricCurveBoundarySource

        if not isinstance(query, ParametricCurveBoundarySource):
            raise TypeError(
                "Curve source validation requires ParametricCurveBoundarySource."
            )
        if (query.source_id, query.source_revision) != (
            self.source_id,
            self.source_revision,
        ):
            raise ValueError("Curve boundary query must bind this source revision.")
        query.require_current()
        raw_ids = np.asarray(requested_entity_ids)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
            raise TypeError("requested_entity_ids must be an integer vector.")
        if (
            raw_ids.dtype.kind == "u"
            and raw_ids.size
            and np.any(raw_ids > np.iinfo(np.int64).max)
        ):
            raise ValueError("requested_entity_ids must fit int64.")
        selected_ids = np.asarray(raw_ids, dtype=np.int64)
        if selected_ids.size == 0 or np.unique(selected_ids).size != selected_ids.size:
            raise ValueError(
                "requested_entity_ids must be a nonempty set of unique curve IDs."
            )
        unknown = np.setdiff1d(selected_ids, self.scope_entity_id_banks[1])
        if unknown.size:
            raise ValueError(
                "requested_entity_ids contain curves absent from this source."
            )
        chart_scope_ids = np.asarray(self.chart_scope_entity_ids, dtype=np.int64)
        expected_rows = np.flatnonzero(np.isin(chart_scope_ids, selected_ids))
        expected: dict[str, tuple[str, tuple[str, ...], tuple[float, float]]] = {}
        for row in expected_rows.tolist():
            numeric_id = int(chart_scope_ids[row])
            entity_id, occurrence = self.source_entity_key_for_scope(1, numeric_id)
            parameter_range = (
                (0.0, 1.0)
                if self.chart_parameter_ranges is None
                else (
                    float(self.chart_parameter_ranges[row, 0]),
                    float(self.chart_parameter_ranges[row, 1]),
                )
            )
            chart_id = self.chart_ids[row]
            if chart_id in expected:
                raise ValueError(
                    "Source curve chart IDs must uniquely identify authored ranges."
                )
            expected[chart_id] = (entity_id, occurrence, parameter_range)

        query_entities = tuple(query.curve_entity_ids)
        query_occurrences = tuple(query.occurrence_paths)
        query_charts = tuple(query.chart_ids)
        query_ranges = tuple(query.parameter_ranges)
        if not (
            len(query_entities)
            == len(query_occurrences)
            == len(query_charts)
            == len(query_ranges)
        ):
            raise ValueError("Curve boundary query chart records must have equal extent.")
        selected_keys = {
            self.source_entity_key_for_scope(1, int(identifier))
            for identifier in selected_ids.tolist()
        }
        actual: dict[str, tuple[str, tuple[str, ...], tuple[float, float]]] = {}
        for row in range(len(query_entities)):
            entity_id = query_entities[row]
            occurrence = query_occurrences[row]
            chart_id = query_charts[row]
            parameter_range = query_ranges[row]
            key = (entity_id, tuple(occurrence))
            if key not in selected_keys:
                continue
            if chart_id in actual:
                raise ValueError(
                    "Curve boundary query repeats an authored chart identity."
                )
            actual[chart_id] = (
                entity_id,
                tuple(occurrence),
                (float(parameter_range[0]), float(parameter_range[1])),
            )
        if actual != expected:
            raise ValueError(
                "Curve boundary query must cover every requested authored chart and full parameter range."
            )
        query_row_by_chart = {chart_id: row for row, chart_id in enumerate(query_charts)}
        for chart_id in expected:
            query_row = query_row_by_chart[chart_id]
            source_row = self.chart_ids.index(chart_id)
            first, last = expected[chart_id][2]
            carrier = query.curves[query_row]
            if isinstance(self.curves, SegmentMesh):
                edge = np.asarray(self.curves.edges, dtype=np.int64)[source_row]
                points = np.asarray(self.curves.vertices, dtype=np.float64)[edge]
                if (
                    not isinstance(carrier, SegmentChartCurve)
                    or not np.array_equal(np.asarray(carrier.start), points[0])
                    or not np.array_equal(np.asarray(carrier.end), points[1])
                    or carrier.first != first
                    or carrier.last != last
                ):
                    raise ValueError(
                        "Curve boundary query changed its SegmentMesh source chart."
                    )
            elif (
                getattr(carrier, "mapping", None) is not self.curves.mapping
                or getattr(carrier, "chart_row", None) != source_row
                or getattr(carrier, "first", None) != first
                or getattr(carrier, "last", None) != last
            ):
                raise ValueError(
                    "Curve boundary query changed its BoundaryAtlas source chart."
                )
        if (
            self.gap_domain is None
            or not isinstance(self.curves, BoundaryAtlas)
            or self.ambient_dimension != 3
        ):
            return
        authority = self.gap_domain.brep_authority
        if authority is None:
            return
        from ...geometry.brep._model import BRepGeometry, BRepModel

        for query_row, (entity_id, occurrence, parameter_range) in enumerate(
            zip(
                query.curve_entity_ids,
                query.occurrence_paths,
                query.parameter_ranges,
                strict=True,
            )
        ):
            if (entity_id, tuple(occurrence)) not in selected_keys:
                continue
            curve_row = self.gap_domain.resolve_entity_id(
                1, entity_id, occurrence_path=tuple(occurrence)
            )
            edge_index = int(self.gap_domain.source_indices[1][curve_row])
            if isinstance(authority, BRepModel):
                native_range = authority.source_curve_parameter_domain(
                    edge_index, occurrence_path=tuple(occurrence)
                )
                lower, upper = authority.source_curve_point_bounds(
                    edge_index,
                    min(parameter_range),
                    max(parameter_range),
                    occurrence_path=tuple(occurrence),
                )
            elif isinstance(authority, BRepGeometry):
                if occurrence:
                    raise ValueError(
                        "Placed BRep curve bounds require their BRepModel owner."
                    )
                native_range = authority.source_curve_parameter_domain(edge_index)
                lower, upper = authority.source_curve_point_bounds(
                    edge_index, min(parameter_range), max(parameter_range)
                )
            else:
                raise TypeError(
                    "MeshingDomain.brep_authority has no exact BRep curve bounds."
                )
            if tuple(sorted(parameter_range)) != tuple(sorted(native_range)):
                raise ValueError(
                    "Curve chart range does not match its authored BRep edge interval."
                )
            source_lower = np.asarray(lower, dtype=np.float64)
            source_upper = np.asarray(upper, dtype=np.float64)
            query_lower = np.asarray(query.boxes, dtype=np.float64)[query_row, 0]
            query_upper = np.asarray(query.boxes, dtype=np.float64)[query_row, 1]
            if (
                source_lower.shape != (3,)
                or source_upper.shape != (3,)
                or np.any(query_lower > source_lower)
                or np.any(query_upper < source_upper)
            ):
                raise ValueError(
                    "Continuous curve query bounds do not enclose its exact BRep edge."
                )

    def validate_point_boundary_query(
        self,
        query: ParametricCurveBoundarySource,
        requested_point_ids: ArrayLike,
        /,
    ) -> None:
        """Rebuild authored endpoint maps and continuous bounds for protected points."""
        from ...geometry._curve_boundary import ParametricCurveBoundarySource

        if not isinstance(query, ParametricCurveBoundarySource):
            raise TypeError(
                "Curve point validation requires ParametricCurveBoundarySource."
            )
        if (query.source_id, query.source_revision) != (
            self.source_id,
            self.source_revision,
        ):
            raise ValueError("Curve point query must bind this source revision.")
        query.require_current()
        raw_ids = np.asarray(requested_point_ids)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
            raise TypeError("requested_point_ids must be an integer vector.")
        if (
            raw_ids.dtype.kind == "u"
            and raw_ids.size
            and np.any(raw_ids > np.iinfo(np.int64).max)
        ):
            raise ValueError("requested_point_ids must fit int64.")
        identifiers = np.asarray(raw_ids, dtype=np.int64)
        if identifiers.size == 0 or np.unique(identifiers).size != identifiers.size:
            raise ValueError(
                "requested_point_ids must be a nonempty set of unique point IDs."
            )

        chart_row_by_id = {
            identifier: row for row, identifier in enumerate(self.chart_ids)
        }
        source_chart_rows: list[int] = []
        curve_scope_ids: list[int] = []
        for row, (entity_id, occurrence, chart_id, parameter_range) in enumerate(
            zip(
                query.curve_entity_ids,
                query.occurrence_paths,
                query.chart_ids,
                query.parameter_ranges,
                strict=True,
            )
        ):
            chart_row = chart_row_by_id.get(chart_id)
            if chart_row is None:
                raise ValueError(
                    "Curve point query names a chart absent from the source owner."
                )
            source_chart_rows.append(chart_row)
            curve_scope_id = int(self.chart_scope_entity_ids[chart_row])
            curve_scope_ids.append(curve_scope_id)
            expected_key = self.source_entity_key_for_scope(1, curve_scope_id)
            expected_range = (
                (0.0, 1.0)
                if self.chart_parameter_ranges is None
                else tuple(
                    float(value) for value in self.chart_parameter_ranges[chart_row]
                )
            )
            if (entity_id, tuple(occurrence)) != expected_key or tuple(
                float(value) for value in parameter_range
            ) != expected_range:
                raise ValueError(
                    "Curve point query changed an authored chart identity or range."
                )
            carrier = query.curves[row]
            first, last = expected_range
            if isinstance(self.curves, SegmentMesh):
                edge = np.asarray(self.curves.edges, dtype=np.int64)[chart_row]
                source_points = np.asarray(self.curves.vertices, dtype=np.float64)[edge]
                if (
                    not isinstance(carrier, SegmentChartCurve)
                    or not np.array_equal(np.asarray(carrier.start), source_points[0])
                    or not np.array_equal(np.asarray(carrier.end), source_points[1])
                    or carrier.first != first
                    or carrier.last != last
                ):
                    raise ValueError(
                        "Curve point query changed its SegmentMesh source chart."
                    )
            else:
                if (
                    getattr(carrier, "mapping", None) is not self.curves.mapping
                    or getattr(carrier, "chart_row", None) != chart_row
                    or getattr(carrier, "first", None) != first
                    or getattr(carrier, "last", None) != last
                ):
                    raise ValueError(
                        "Curve point query changed its BoundaryAtlas source chart."
                    )
        self.validate_curve_boundary_query(
            query, np.asarray(sorted(set(curve_scope_ids)), dtype=np.int64)
        )

        expected_point_bounds: dict[
            tuple[str, tuple[str, ...]], tuple[np.ndarray, np.ndarray]
        ] = {}
        for query_row, source_row in enumerate(source_chart_rows):
            curve_id = query.chart_ids[query_row]
            parameter_range = query.parameter_ranges[query_row]
            if isinstance(self.curves, SegmentMesh):
                edge = np.asarray(self.curves.edges, dtype=np.int64)[source_row]
                vertex_rows = tuple(int(value) for value in edge)
                point_keys = tuple(
                    self.source_entity_key_for_scope(
                        0, int(self.vertex_entity_ids[vertex_row])
                    )
                    for vertex_row in vertex_rows
                )
                source_points = np.asarray(self.curves.vertices, dtype=np.float64)[
                    np.asarray(vertex_rows, dtype=np.int64)
                ]
                supplied_pairs = tuple(
                    (tuple(point), tuple(point)) for point in source_points
                )
            else:
                endpoints = (
                    (None, None)
                    if not self.chart_endpoint_ids
                    else self.chart_endpoint_ids[source_row]
                )
                point_keys_list = []
                supplied_pairs_list = []
                for endpoint in endpoints:
                    if endpoint is None:
                        point_keys_list.append(None)
                        supplied_pairs_list.append(None)
                        continue
                    vertex_row = (
                        self.vertex_entity_keys.index(endpoint)
                        if self.gap_domain is None
                        else self.gap_domain.resolve_entity_id(0, endpoint)
                    )
                    point_id = int(self.vertex_entity_ids[vertex_row])
                    point_keys_list.append(self.source_entity_key_for_scope(0, point_id))
                    if self.gap_domain is None:
                        supplied_pairs_list.append(None)
                        continue
                    source_lower, source_upper = (
                        self.gap_domain.source_corner_point_bounds(vertex_row)
                    )
                    supplied = query.endpoint_point_bounds[query_row][
                        len(point_keys_list) - 1
                    ]
                    if supplied is None:
                        raise ValueError(
                            "Curve point query omits its domain-bound endpoint enclosure."
                        )
                    for value, low, high in zip(
                        supplied[0], source_lower, source_upper, strict=True
                    ):
                        if Fraction.from_float(float(value)) > low:
                            raise ValueError(
                                "Curve endpoint lower bound excludes its source corner."
                            )
                    for value, low, high in zip(
                        supplied[1], source_lower, source_upper, strict=True
                    ):
                        if Fraction.from_float(float(value)) < high:
                            raise ValueError(
                                "Curve endpoint upper bound excludes its source corner."
                            )
                    supplied_pairs_list.append(supplied)
                point_keys = tuple(point_keys_list)
                supplied_pairs = tuple(supplied_pairs_list)
            actual_keys = query.endpoint_source_keys[query_row]
            if point_keys != actual_keys:
                raise ValueError(
                    "Curve point query endpoint identities differ from the source owner."
                )
            for side, (key, parameter, supplied) in enumerate(
                zip(point_keys, parameter_range, supplied_pairs, strict=True)
            ):
                if key is None:
                    if supplied is not None:
                        raise ValueError(
                            "An unidentified curve endpoint cannot carry source point bounds."
                        )
                    continue
                if supplied is not None:
                    if query.endpoint_point_bounds[query_row][side] != supplied:
                        raise ValueError(
                            "Curve point query changed a source endpoint enclosure."
                        )
                lower, upper = query.chart_point_bounds(curve_id, parameter)
                if supplied is not None:
                    lower = np.maximum(lower, np.asarray(supplied[0], dtype=np.float64))
                    upper = np.minimum(upper, np.asarray(supplied[1], dtype=np.float64))
                if np.any(lower > upper):
                    raise ValueError(
                        "Curve endpoint point enclosures have empty intersection."
                    )
                previous = expected_point_bounds.get(key)
                if previous is not None:
                    lower = np.maximum(lower, previous[0])
                    upper = np.minimum(upper, previous[1])
                    if np.any(lower > upper):
                        raise ValueError(
                            "Repeated curve endpoint identities have disjoint enclosures."
                        )
                expected_point_bounds[key] = (np.asarray(lower), np.asarray(upper))

        expected_keys = tuple(sorted(expected_point_bounds))
        actual_keys = tuple(
            zip(query.point_entity_ids, query.point_occurrence_paths, strict=True)
        )
        if actual_keys != expected_keys:
            raise ValueError(
                "Curve point query point bank does not match its authored chart endpoints."
            )
        requested_keys = tuple(
            self.source_entity_key_for_scope(0, int(identifier))
            for identifier in identifiers.tolist()
        )
        if any(key not in expected_point_bounds for key in requested_keys):
            raise ValueError("Requested point scope is not an authored curve endpoint.")
        for key, (expected_lower, expected_upper) in expected_point_bounds.items():
            actual_lower, actual_upper = query.source_point_bounds(*key)
            if not np.array_equal(actual_lower, expected_lower) or not np.array_equal(
                actual_upper, expected_upper
            ):
                raise ValueError(
                    "Curve point query changed its continuous endpoint enclosure."
                )


@final
class NativePlcSource(StrictModule, NonTrainableState):
    """Oriented piecewise-linear complex bound to one source revision."""

    complex: PiecewiseLinearComplex
    mapped_domain: MappedReferenceDomain | None
    fidelity_source: SourceBoundaryQuery | None
    maximum_deviation: float | None = eqx.field(static=True)
    root_relation: _MappedPlcRootRelation | None
    plc_vertex_root_global_ids: np.ndarray | None
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        complex_: PiecewiseLinearComplex,
        source_id: str,
        source_revision: str,
        /,
        *,
        mapped_domain: MappedReferenceDomain | None = None,
        fidelity_source: SourceBoundaryQuery | None = None,
        maximum_deviation: float | None = None,
        plc_vertex_root_global_ids: ArrayLike | None = None,
    ) -> None:
        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        complex_id = complex_.complex_id
        scope_id_banks, scope_key_banks, scope_set_ids = _prepare_plc_scope_banks(
            complex_, source, revision
        )
        if mapped_domain is not None and (
            mapped_domain.source_id != source or mapped_domain.source_revision != revision
        ):
            raise ValueError("mapped_domain must bind source_id and source_revision.")
        if complex_.geometry is not None:
            if mapped_domain is None:
                raise ValueError(
                    "A mapped PLC geometry requires its original MappedReferenceDomain."
                )
            if cell_geometry_id(complex_.geometry) != mapped_domain.reference_geometry_id:
                raise ValueError(
                    "PLC geometry and mapped-domain source geometry must be identical exact maps."
                )
        if mapped_domain is not None and fidelity_source is not None:
            raise ValueError(
                "Mapped PLC sources derive their exact boundary query from the retained root relation."
            )
        if fidelity_source is not None:
            _bound_source_query(fidelity_source, source, revision)
        if mapped_domain is None and plc_vertex_root_global_ids is not None:
            raise ValueError(
                "PLC-to-root vertex IDs require a declared MappedReferenceDomain."
            )
        explicit_root_ids: np.ndarray | None = None
        if plc_vertex_root_global_ids is not None:
            raw_root_ids = np.asarray(plc_vertex_root_global_ids)
            if raw_root_ids.dtype.kind not in "iu" or raw_root_ids.shape != (
                len(complex_.vertex_ids),
            ):
                raise TypeError(
                    "plc_vertex_root_global_ids must contain one integer root ID per PLC vertex."
                )
            if raw_root_ids.dtype.kind == "u" and np.any(
                raw_root_ids > np.iinfo(np.int64).max
            ):
                raise ValueError("PLC-to-root vertex IDs must fit int64.")
            explicit_root_ids = np.array(raw_root_ids, dtype=np.int64, copy=True)
            explicit_root_ids.setflags(write=False)
        deviation = _source_deviation(maximum_deviation)
        self.complex = complex_
        self.mapped_domain = mapped_domain
        self.fidelity_source = fidelity_source
        self.maximum_deviation = deviation
        self.plc_vertex_root_global_ids = explicit_root_ids
        self.scope_entity_id_banks = scope_id_banks
        self.scope_entity_key_banks = scope_key_banks
        self.scope_entity_set_ids = scope_set_ids
        self.source_id, self.source_revision = source, revision
        base_binding_id = canonical_fingerprint(
            {
                "kind": "native-plc-source",
                "source": source,
                "revision": revision,
                "complex": complex_id,
                "mapped_domain": None
                if mapped_domain is None
                else mapped_domain.domain_id,
                "maximum_deviation": deviation,
                "fidelity_source": None
                if fidelity_source is None
                else (fidelity_source.source_id, fidelity_source.source_revision),
            }
        )
        self.binding_id = base_binding_id
        if mapped_domain is None:
            self.root_relation = None
        else:
            relation = _mapped_plc_root_relation(self, mapped_domain, explicit_root_ids)
            self.root_relation = relation
            self.binding_id = canonical_fingerprint(
                {
                    "kind": "native-plc-mapped-root-source",
                    "base_binding": base_binding_id,
                    "relation": relation.relation_id,
                }
            )

    @property
    def domain(self) -> MappedReferenceDomain:
        if self.mapped_domain is None:
            raise ValueError("This PLC source has no mapped-reference root domain.")
        return self.mapped_domain

    def boundary_query(
        self,
        /,
        *,
        face_scope_keys: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
        edge_scope_keys: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    ) -> SourceBoundaryQuery:
        if self.root_relation is not None:
            from ...geometry._mapped_reference_boundary import (
                MappedPlcRootAuthority,
                MappedReferenceBoundarySource,
            )

            if self.mapped_domain is None or not isinstance(self, MappedPlcRootAuthority):
                raise TypeError("Mapped PLC query requires a complete root authority.")
            return MappedReferenceBoundarySource(
                self,
                face_scope_keys=face_scope_keys,
                edge_scope_keys=edge_scope_keys,
            )
        if face_scope_keys is not None or edge_scope_keys is not None:
            raise ValueError(
                "Scoped PLC boundary queries require the exact mapped-root relation."
            )
        if self.fidelity_source is None:
            raise ValueError("This PLC source has no original boundary query.")
        return self.fidelity_source

    def validate_restored(self, /) -> None:
        from ...geometry._mesh_certificates import _same_tree_content

        fresh = type(self)(
            self.complex,
            self.source_id,
            self.source_revision,
            mapped_domain=self.mapped_domain,
            fidelity_source=self.fidelity_source,
            maximum_deviation=self.maximum_deviation,
            plc_vertex_root_global_ids=self.plc_vertex_root_global_ids,
        )
        if fresh.binding_id != self.binding_id or not _same_tree_content(
            fresh.root_relation, self.root_relation
        ):
            raise ValueError(
                "Native PLC source no longer matches its exact mapped root relation."
            )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("PLC source dimension must be in 0..3.")
        return self.scope_entity_id_banks[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("PLC source dimension must be in 0..3.")
        return self.scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("PLC source dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("entity_id must be one stable integer scope ID.")
        ids = self.scope_entity_id_banks[dimension]
        position = int(np.searchsorted(ids, entity_id))
        if position >= ids.size or int(ids[position]) != entity_id:
            raise ValueError("entity_id is absent from this PLC source dimension.")
        return self.scope_entity_key_banks[dimension][position]


@final
class NativePolyhedralSource(StrictModule, NonTrainableState):
    """PLC plus represented physical power sites and weights."""

    complex: PiecewiseLinearComplex
    sites: np.ndarray
    weights: np.ndarray
    site_ids: tuple[str, ...] = eqx.field(static=True)
    mapped_domain: MappedReferenceDomain | None
    fidelity_source: SourceBoundaryQuery | None
    maximum_deviation: float | None = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        complex_: PiecewiseLinearComplex,
        source_id: str,
        source_revision: str,
        /,
        *,
        sites: ArrayLike,
        weights: ArrayLike | None = None,
        site_ids: tuple[str, ...] | None = None,
        mapped_domain: MappedReferenceDomain | None = None,
        fidelity_source: SourceBoundaryQuery | None = None,
        maximum_deviation: float | None = None,
    ) -> None:
        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        complex_id = complex_.complex_id
        scope_id_banks, scope_key_banks, scope_set_ids = _prepare_plc_scope_banks(
            complex_, source, revision
        )
        points = np.asarray(sites, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("sites must have nonempty shape (N, 3).")
        if not np.all(np.isfinite(points)):
            raise ValueError("sites must be finite.")
        values = (
            np.zeros(points.shape[0], dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if values.shape != (points.shape[0],) or not np.all(np.isfinite(values)):
            raise ValueError("weights must be finite with one value per site.")
        if site_ids is None:
            raise ValueError("site_ids must be authored and stable for every power site.")
        identifiers = tuple(
            normalized_identifier(identifier, "site_id") for identifier in site_ids
        )
        if len(identifiers) != points.shape[0] or len(set(identifiers)) != len(
            identifiers
        ):
            raise ValueError("site_ids must contain one unique stable ID per power site.")
        if mapped_domain is not None and (
            mapped_domain.source_id != source or mapped_domain.source_revision != revision
        ):
            raise ValueError("mapped_domain must bind source_id and source_revision.")
        if fidelity_source is not None:
            _bound_source_query(fidelity_source, source, revision)
        deviation = _source_deviation(maximum_deviation)
        points = np.array(points, dtype=np.float64, copy=True)
        values = np.array(values, dtype=np.float64, copy=True)
        points.setflags(write=False)
        values.setflags(write=False)
        self.complex = complex_
        self.sites = points
        self.weights = values
        self.site_ids = identifiers
        self.mapped_domain = mapped_domain
        self.fidelity_source = fidelity_source
        self.maximum_deviation = deviation
        self.scope_entity_id_banks = scope_id_banks
        self.scope_entity_key_banks = scope_key_banks
        self.scope_entity_set_ids = scope_set_ids
        self.source_id, self.source_revision = source, revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-polyhedral-source",
                "source": source,
                "revision": revision,
                "complex": complex_id,
                "site_ids": self.site_ids,
                "power_sites": array_tree_fingerprint((points, values)),
                "mapped_domain": None
                if mapped_domain is None
                else mapped_domain.domain_id,
                "maximum_deviation": deviation,
                "fidelity_source": None
                if fidelity_source is None
                else (fidelity_source.source_id, fidelity_source.source_revision),
            }
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("PLC source dimension must be in 0..3.")
        return self.scope_entity_id_banks[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("PLC source dimension must be in 0..3.")
        return self.scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("PLC source dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("entity_id must be one stable integer scope ID.")
        ids = self.scope_entity_id_banks[dimension]
        position = int(np.searchsorted(ids, entity_id))
        if position >= ids.size or int(ids[position]) != entity_id:
            raise ValueError("entity_id is absent from this PLC source dimension.")
        return self.scope_entity_key_banks[dimension][position]


@final
class NativeMappedHexSource(StrictModule, NonTrainableState):
    """Bind a source PLC to exact root-cell maps and its declared entity relation."""

    reference: NativePlcSource
    domain: MappedReferenceDomain
    root_relation: _MappedPlcRootRelation
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        reference: NativePlcSource,
        domain: MappedReferenceDomain,
        /,
        *,
        plc_vertex_root_global_ids: ArrayLike | None = None,
    ) -> None:
        if (reference.source_id, reference.source_revision) != (
            domain.source_id,
            domain.source_revision,
        ):
            raise ValueError(
                "Reference PLC and mapped root domain must share source identity/revision."
            )
        if tuple(reference.complex.region_ids) != tuple(domain.region_ids):
            raise ValueError(
                "Reference PLC and mapped root domain must share ordered material identities."
            )
        relation = _mapped_plc_root_relation(
            reference, domain, plc_vertex_root_global_ids
        )
        source = normalized_identifier(domain.source_id, "source_id")
        revision = normalized_identifier(domain.source_revision, "source_revision")
        banks = tuple(
            _plc_scope_bank(reference.complex, dimension) for dimension in range(4)
        )
        id_banks: list[np.ndarray] = []
        key_banks: list[tuple[tuple[str, tuple[str, ...]], ...]] = []
        set_ids: list[str] = []
        for dimension, bank in enumerate(banks):
            identifiers = np.asarray([row[2] for row in bank], dtype=np.int64)
            identifiers.setflags(write=False)
            keys = tuple((name, occurrence) for name, occurrence, _ in bank)
            id_banks.append(identifiers)
            key_banks.append(keys)
            set_ids.append(
                canonical_fingerprint(
                    {
                        "kind": "mapped-root-source-entity-set",
                        "source": source,
                        "revision": revision,
                        "dimension": dimension,
                        "mapped_root_set": domain.entity_set_id(dimension),
                        "plc_source_set": reference.scope_entity_set_id(dimension),
                        "root_relation": relation.relation_id,
                        "source_entities": tuple(
                            (name, occurrence, identifier)
                            for (name, occurrence), identifier in zip(
                                keys, identifiers.tolist(), strict=True
                            )
                        ),
                    }
                )
            )
        self.reference = reference
        self.domain = domain
        self.root_relation = relation
        self.scope_entity_id_banks = tuple(id_banks)
        self.scope_entity_key_banks = tuple(key_banks)
        self.scope_entity_set_ids = tuple(set_ids)
        self.source_id, self.source_revision = source, revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-mapped-hex-source",
                "reference": reference.binding_id,
                "mapped_domain": domain.domain_id,
                "root_relation": relation.relation_id,
                "scope_entity_set_ids": self.scope_entity_set_ids,
            }
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Mapped source dimension must be in 0..3.")
        return self.scope_entity_id_banks[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Mapped source dimension must be in 0..3.")
        return self.scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Mapped source dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("entity_id must be one stable integer scope ID.")
        identifiers = self.scope_entity_id_banks[dimension]
        position = int(np.searchsorted(identifiers, entity_id))
        if position >= identifiers.size or int(identifiers[position]) != entity_id:
            raise ValueError("entity_id is absent from this mapped source dimension.")
        return self.scope_entity_key_banks[dimension][position]

    @classmethod
    def from_stored_fields(
        cls: type[NativeMappedHexSource], value: NativeMappedHexSource, /
    ) -> NativeMappedHexSource:
        if type(value) is not cls:
            raise TypeError(
                "stored_fields must have the registered mapped-hex source type."
            )
        from ...geometry._mesh_certificates import _same_tree_content

        fresh = cls(
            value.reference,
            value.domain,
            plc_vertex_root_global_ids=value.root_relation.vertex_root_global_ids,
        )
        if (
            fresh.source_id,
            fresh.source_revision,
            fresh.binding_id,
            fresh.root_relation.relation_id,
        ) != (
            value.source_id,
            value.source_revision,
            value.binding_id,
            value.root_relation.relation_id,
        ) or not _same_tree_content(fresh.root_relation, value.root_relation):
            raise ValueError(
                "Stored mapped PLC/root entity relation no longer validates."
            )
        return fresh

    def validate_restored(self, /) -> None:
        """Rebuild and validate the bound PLC-to-root entity relation."""
        type(self).from_stored_fields(self)

    @property
    def complex(self) -> PiecewiseLinearComplex:
        """Return the exact PLC source retained by this mapped root authority."""
        return self.reference.complex

    def boundary_query(
        self,
        /,
        *,
        face_scope_keys: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
        edge_scope_keys: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    ) -> MappedReferenceBoundarySource:
        """Return exact original-source charts over the declared root-cell map."""
        from ...geometry._mapped_reference_boundary import (
            MappedPlcRootAuthority,
            MappedReferenceBoundarySource,
        )

        if not isinstance(self, MappedPlcRootAuthority):
            raise TypeError("Mapped boundary query requires a complete root authority.")

        return MappedReferenceBoundarySource(
            self,
            face_scope_keys=face_scope_keys,
            edge_scope_keys=edge_scope_keys,
        )


@final
class NativeImplicitSource(StrictModule, NonTrainableState):
    """Bind implicit geometry and discovery data to stable source strata."""

    geometry: CompiledGeometry
    grid: PreparedTensorGrid | None
    discovery: AdaptiveImplicitSurface | None
    authoring_profile: AnalyticImplicitProfile | None
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        geometry: CompiledGeometry,
        source_id: str,
        source_revision: str,
        /,
        *,
        grid: PreparedTensorGrid | None = None,
        discovery: AdaptiveImplicitSurface | None = None,
        authoring_profile: AnalyticImplicitProfile | None = None,
    ) -> None:
        if geometry.ambient_dimension != 3 or geometry.kind is not GeometryKind.REGION:
            raise ValueError("NativeImplicitSource requires a bounded 3D region source.")
        if grid is not None and not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid or None.")
        if discovery is not None and not isinstance(discovery, AdaptiveImplicitSurface):
            raise TypeError("discovery must be an AdaptiveImplicitSurface or None.")
        if grid is None and discovery is None:
            raise ValueError(
                "NativeImplicitSource requires a fixed grid or adaptive discovery."
            )
        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        if authoring_profile is not None:
            if not isinstance(authoring_profile, AnalyticImplicitProfile):
                raise TypeError(
                    "authoring_profile must be AnalyticImplicitProfile or None."
                )
            if (
                authoring_profile.source_id,
                authoring_profile.source_revision,
            ) != (source, revision):
                raise ValueError(
                    "authoring_profile must bind the NativeImplicitSource identity."
                )
            authoring_profile.require_bound(
                geometry, authoring_profile.coordinate_contract
            )
        if discovery is not None:
            if discovery.source_id != source:
                raise ValueError("discovery.source_id must match source_id.")
            if array_tree_fingerprint(discovery.geometry) != array_tree_fingerprint(
                geometry
            ):
                raise ValueError("discovery must retain the same compiled geometry.")
        surface_key = source_entity_id(source, revision, "implicit-surface", "zero-set")
        surface_id = _numeric_entity_ids(
            "native-implicit-surface", (surface_key,), ((),)
        )[0]
        region_key = source_entity_id(source, revision, "implicit-region", "interior")
        region_id = _numeric_entity_ids("native-implicit-region", (region_key,), ((),))[0]
        raw_banks = (
            (),
            (),
            ((surface_key, (), surface_id),),
            ((region_key, (), region_id),),
        )
        id_banks: list[np.ndarray] = []
        key_banks: list[tuple[tuple[str, tuple[str, ...]], ...]] = []
        set_ids: list[str] = []
        for dimension, bank in enumerate(raw_banks):
            identifiers = np.asarray([row[2] for row in bank], dtype=np.int64)
            identifiers.setflags(write=False)
            id_banks.append(identifiers)
            key_banks.append(tuple((name, path) for name, path, _ in bank))
            set_ids.append(_plc_scope_set_id(source, revision, dimension, bank))
        banks = (tuple(id_banks), tuple(key_banks), tuple(set_ids))
        self.geometry = geometry
        self.grid = grid
        self.discovery = discovery
        self.authoring_profile = authoring_profile
        (
            self.scope_entity_id_banks,
            self.scope_entity_key_banks,
            self.scope_entity_set_ids,
        ) = banks
        self.source_id, self.source_revision = source, revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-implicit-source",
                "source": source,
                "revision": revision,
                "geometry": array_tree_fingerprint(geometry),
                "grid": None if grid is None else array_tree_fingerprint(grid),
                "discovery": None
                if discovery is None
                else discovery.evidence.evidence_id,
                "authoring_profile": (
                    None if authoring_profile is None else authoring_profile.profile_id
                ),
                "scope_entity_key_banks": self.scope_entity_key_banks,
                "scope_entity_set_ids": self.scope_entity_set_ids,
            }
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Implicit source dimension must be in 0..3.")
        return self.scope_entity_id_banks[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Implicit source dimension must be in 0..3.")
        return self.scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Implicit source dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("entity_id must be one stable integer scope ID.")
        identifiers = self.scope_entity_id_banks[dimension]
        position = int(np.searchsorted(identifiers, entity_id))
        if position >= identifiers.size or int(identifiers[position]) != entity_id:
            raise ValueError("entity_id is absent from this implicit source dimension.")
        return self.scope_entity_key_banks[dimension][position]


def _source_deviation(value: float | None, /) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("maximum_deviation must be a nonnegative physical tolerance.")
    deviation = float(value)
    if not np.isfinite(deviation) or deviation < 0.0:
        raise ValueError("maximum_deviation must be finite and nonnegative.")
    return deviation


def _bound_source_query(
    query: SourceBoundaryQuery, source_id: str, source_revision: str, /
) -> None:
    if not isinstance(query, SourceBoundaryQuery):
        raise TypeError("fidelity_source must implement SourceBoundaryQuery.")
    if (query.source_id, query.source_revision) != (source_id, source_revision):
        raise ValueError("fidelity_source must bind source_id and source_revision.")


def _block_region_bindings(
    block_regions: tuple[tuple[str, str], ...],
    block_names: tuple[str, ...],
    /,
) -> tuple[tuple[str, str], ...]:
    bindings = tuple(
        sorted(
            (
                normalized_identifier(block, "block_name"),
                normalized_identifier(region, "region_name"),
            )
            for block, region in block_regions
        )
    )
    if len({block for block, _ in bindings}) != len(bindings):
        raise ValueError("block_regions must contain at most one region per block.")
    if any(block not in block_names for block, _ in bindings):
        raise ValueError("block_regions may name only declared blocks.")
    return bindings


@final
class NativeStructuredSource(StrictModule, NonTrainableState):
    """Bind transfinite blocks and explicit gluing to one source boundary."""

    blocks: tuple[TransfiniteBlock, ...]
    interfaces: tuple[BlockInterfaceControl, ...]
    fidelity_source: SourceBoundaryQuery
    maximum_deviation: float | None = eqx.field(static=True)
    domain: PiecewiseLinearDomain | None
    block_regions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        blocks: tuple[TransfiniteBlock, ...],
        interfaces: tuple[BlockInterfaceControl, ...],
        source_id: str,
        source_revision: str,
        /,
        *,
        fidelity_source: SourceBoundaryQuery,
        maximum_deviation: float | None = None,
        domain: PiecewiseLinearDomain | None = None,
        block_regions: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if not blocks or any(not isinstance(block, TransfiniteBlock) for block in blocks):
            raise TypeError("blocks must contain at least one TransfiniteBlock.")
        if any(
            not isinstance(interface, BlockInterfaceControl) for interface in interfaces
        ):
            raise TypeError("interfaces must contain BlockInterfaceControl values.")
        names = tuple(sorted(block.name for block in blocks))
        if len(set(names)) != len(names):
            raise ValueError("TransfiniteBlock names must be unique.")
        if any(
            interface.first_block not in names or interface.second_block not in names
            for interface in interfaces
        ):
            raise ValueError("Every block interface must name declared blocks.")
        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        _bound_source_query(fidelity_source, source, revision)
        if domain is not None and (
            domain.source_id != source or domain.source_revision != revision
        ):
            raise ValueError("domain must bind source_id and source_revision.")
        deviation = _source_deviation(maximum_deviation)
        regions = _block_region_bindings(block_regions, names)
        ordered_blocks = tuple(sorted(blocks, key=lambda block: block.name))
        ordered_interfaces = tuple(
            sorted(interfaces, key=lambda interface: interface.control_id)
        )
        block_dimensions = {len(block.intervals) for block in ordered_blocks}
        if len(block_dimensions) != 1:
            raise ValueError(
                "Structured source blocks must share one topological dimension."
            )
        block_dimension = next(iter(block_dimensions))
        domain_regions = tuple(domain.region_ids) if domain is not None else ()
        region_map = dict(regions)
        if domain_regions:
            if not region_map and len(domain_regions) == 1:
                region_map = {block.name: domain_regions[0] for block in ordered_blocks}
            elif set(region_map) != set(names):
                raise ValueError(
                    "Every structured block must bind one source region when its domain has multiple regions."
                )
            if any(region not in domain_regions for region in region_map.values()):
                raise ValueError(
                    "Structured block regions must exist in the source domain."
                )
        interface_by_face: dict[tuple[str, int], BlockInterfaceControl] = {}
        for interface in ordered_interfaces:
            first = (interface.first_block, interface.first_face)
            second = (interface.second_block, interface.second_face)
            if first in interface_by_face or second in interface_by_face:
                raise ValueError(
                    "A logical block face cannot have multiple interface declarations."
                )
            interface_by_face[first] = interface
            interface_by_face[second] = interface
        if block_dimension == 2:
            curve_entities = tuple(
                sorted(
                    {
                        boundary.source_id
                        for block in ordered_blocks
                        for boundary in block.boundaries
                    }
                )
            )
            curve_records = tuple(
                ("curve", identifier, ()) for identifier in curve_entities
            )
            surface_records = tuple(
                ("surface", block.name, ()) for block in ordered_blocks
            )
        else:
            curve_records = ()
            face_records: list[tuple[str, _ScopeEntityIdentifier, tuple[str, ...]]] = []
            for block in ordered_blocks:
                for face, boundary in enumerate(block.boundaries):
                    interface = interface_by_face.get((block.name, face))
                    if interface is not None:
                        first_region = region_map.get(interface.first_block)
                        second_region = region_map.get(interface.second_block)
                        if first_region is not None and first_region == second_region:
                            continue
                    face_records.append(
                        (
                            "surface",
                            boundary.source_id,
                            (block.name, f"face:{face}"),
                        )
                    )
            surface_records = tuple(face_records)
        region_entities = domain_regions if block_dimension == 3 else ()
        if block_dimension == 3 and not region_entities:
            region_entities = tuple(sorted(set(region_map.values())))
        scope_records = (
            (),
            curve_records,
            surface_records,
            tuple(("region", region, ()) for region in region_entities),
        )
        provided_banks = _provided_source_scope_banks(fidelity_source)
        if provided_banks is not None:
            scope_id_banks, scope_key_banks, scope_set_ids = provided_banks
        else:
            source_banks = tuple(
                _prepare_projected_source_scope_bank(source, revision, dimension, entries)
                for dimension, entries in enumerate(scope_records)
            )
            scope_id_banks = tuple(bank[0] for bank in source_banks)
            scope_key_banks = tuple(bank[1] for bank in source_banks)
            scope_set_ids = tuple(bank[2] for bank in source_banks)
        self.blocks = ordered_blocks
        self.interfaces = ordered_interfaces
        self.fidelity_source = fidelity_source
        self.maximum_deviation = deviation
        self.domain = domain
        self.block_regions = regions
        self.scope_entity_id_banks = scope_id_banks
        self.scope_entity_key_banks = scope_key_banks
        self.scope_entity_set_ids = scope_set_ids
        self.source_id, self.source_revision = source, revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-structured-source",
                "source": source,
                "revision": revision,
                "blocks": tuple(block.block_id for block in ordered_blocks),
                "interfaces": tuple(
                    interface.control_id for interface in ordered_interfaces
                ),
                "maximum_deviation": deviation,
                "domain": None if domain is None else domain.domain_id,
                "block_regions": regions,
                "scope_entity_key_banks": scope_key_banks,
                "scope_entity_set_ids": scope_set_ids,
            }
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Structured source dimension must be in 0..3.")
        return self.scope_entity_id_banks[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Structured source dimension must be in 0..3.")
        return self.scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Structured source dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("entity_id must be one stable integer scope ID.")
        identifiers = self.scope_entity_id_banks[dimension]
        position = int(np.searchsorted(identifiers, entity_id))
        if position >= identifiers.size or int(identifiers[position]) != entity_id:
            raise ValueError("entity_id is absent from this structured source dimension.")
        return self.scope_entity_key_banks[dimension][position]


def _profile_boundary_edge_ids(profile: CellMesh, /) -> np.ndarray:
    """Return authored profile edge IDs incident to exactly one surface cell."""
    edge_ids = np.asarray(profile.entity_set(1).entity_ids, dtype=np.int64)
    if len(profile.topology.incidences) < 2:
        raise ValueError("A surface sweep profile requires edge-to-face incidence.")
    relation = profile.topology.incidences[1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    incident_edges = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    if incident_edges.size and (
        np.any(incident_edges < 0) or np.any(incident_edges >= edge_ids.size)
    ):
        raise ValueError("Profile edge incidence references absent source entities.")
    counts = np.bincount(incident_edges, minlength=edge_ids.size)
    if counts.shape != (edge_ids.size,) or np.any(counts == 0) or np.any(counts > 2):
        raise ValueError("Sweep profile must have manifold edge incidence.")
    return edge_ids[counts == 1]


def _profile_boundary_vertex_ids(
    profile: CellMesh, boundary_edge_ids: np.ndarray, /
) -> np.ndarray:
    edge_ids = np.asarray(profile.entity_set(1).entity_ids, dtype=np.int64)
    boundary_positions = np.flatnonzero(np.isin(edge_ids, boundary_edge_ids))
    relation = profile.topology.incidences[0].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    vertex_positions = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    edge_positions = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    selected = np.isin(edge_positions, boundary_positions)
    vertex_ids = np.asarray(profile.entity_set(0).entity_ids, dtype=np.int64)
    return np.unique(vertex_ids[vertex_positions[selected]])


def _sweep_source_scope_banks(
    profile: CellMesh,
    control: SweepControl,
    source_id: str,
    source_revision: str,
    scope_owner: SourceBoundaryQuery,
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    block_regions: tuple[tuple[str, str], ...],
    /,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[tuple[tuple[str, tuple[str, ...]], ...], ...],
    tuple[str, ...],
]:
    provided_banks = _provided_source_scope_banks(scope_owner)
    if provided_banks is not None:
        return provided_banks
    profile_vertices = tuple(
        int(value) for value in np.asarray(profile.entity_set(0).entity_ids).tolist()
    )
    profile_edges = tuple(
        int(value) for value in np.asarray(profile.entity_set(1).entity_ids).tolist()
    )
    profile_faces = tuple(
        int(value) for value in np.asarray(profile.entity_set(2).entity_ids).tolist()
    )
    boundary_edges = tuple(
        int(value) for value in _profile_boundary_edge_ids(profile).tolist()
    )
    boundary_vertices = tuple(
        int(value)
        for value in _profile_boundary_vertex_ids(
            profile, np.asarray(boundary_edges, dtype=np.int64)
        ).tolist()
    )
    vertex_records = (
        ()
        if control.closed
        else tuple(
            ("sweep-vertex", (vertex, side), ())
            for vertex in profile_vertices
            for side in ("start", "end")
        )
    )
    edge_records = tuple(
        ("sweep-trajectory", vertex, ()) for vertex in boundary_vertices
    ) + (
        ()
        if control.closed
        else tuple(
            ("sweep-cap-edge", (edge, side), ())
            for edge in profile_edges
            for side in ("start", "end")
        )
    )
    face_records = tuple(("sweep-side", edge, ()) for edge in boundary_edges) + (
        ()
        if control.closed
        else tuple(
            ("sweep-cap", (face, side), ())
            for face in profile_faces
            for side in ("start", "end")
        )
    )
    if isinstance(domain, MappedReferenceDomain):
        region_names = tuple(domain.reference_domain.region_ids)
    elif domain is not None:
        region_names = tuple(domain.region_ids)
    else:
        region_names = tuple(sorted({region for _, region in block_regions}))
    mapped_blocks = tuple(f"sweep:{block.name}" for block in profile.blocks)
    region_map = dict(block_regions)
    if region_names:
        if region_map and set(region_map) != set(mapped_blocks):
            raise ValueError("Every swept block needs one declared source region.")
        if not region_map and len(region_names) == 1:
            region_map = {name: region_names[0] for name in mapped_blocks}
        if (len(region_names) > 1 and set(region_map) != set(mapped_blocks)) or any(
            region not in region_names for region in region_map.values()
        ):
            raise ValueError("Sweep block regions must bind declared source regions.")
    region_records = tuple(("region", region, ()) for region in region_names)
    records_by_dimension = (
        vertex_records,
        edge_records,
        face_records,
        region_records,
    )
    banks = tuple(
        _prepare_projected_source_scope_bank(
            source_id, source_revision, dimension, records
        )
        for dimension, records in enumerate(records_by_dimension)
    )
    return (
        tuple(bank[0] for bank in banks),
        tuple(bank[1] for bank in banks),
        tuple(bank[2] for bank in banks),
    )


@final
class NativeSweepSource(StrictModule, NonTrainableState):
    """Bind a profile mesh, station map, and source-fidelity authority."""

    profile: CellMesh
    profile_geometry: cell_geometry_owner.CellGeometrySpec
    control: SweepControl
    fidelity_source: SourceBoundaryQuery
    maximum_deviation: float | None = eqx.field(static=True)
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None
    root_correspondence: cell_geometry_owner.CellGeometryRestrictionSource | None
    block_regions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    domain_vertex_map: tuple[tuple[int, int], ...] | None = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        profile: CellMesh,
        control: SweepControl,
        source_id: str,
        source_revision: str,
        /,
        *,
        fidelity_source: SourceBoundaryQuery,
        maximum_deviation: float | None = None,
        domain: PiecewiseLinearDomain | MappedReferenceDomain | None = None,
        block_regions: tuple[tuple[str, str], ...] = (),
        domain_vertex_map: ArrayLike | None = None,
        profile_geometry: cell_geometry_owner.CellGeometrySpec | None = None,
        root_correspondence: cell_geometry_owner.CellGeometryRestrictionSource
        | None = None,
    ) -> None:
        if profile.topological_dimension != 2 or profile.ambient_dimension != 3:
            raise ValueError("sweep profile must be an ambient-3 surface mesh.")
        geometry = (
            cell_geometry_owner.CellGeometrySpec.affine(profile)
            if profile_geometry is None
            else profile_geometry
        )
        if not isinstance(geometry, cell_geometry_owner.CellGeometrySpec):
            raise TypeError("profile_geometry must be a CellGeometrySpec.")
        geometry.resolve(profile)
        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        _bound_source_query(fidelity_source, source, revision)
        if domain is not None and (
            domain.source_id != source or domain.source_revision != revision
        ):
            raise ValueError("domain must bind source_id and source_revision.")
        if isinstance(domain, MappedReferenceDomain):
            if root_correspondence is None:
                raise TypeError("Mapped sweeps require an explicit root_correspondence.")
            expected_geometry = cell_geometry_id(domain.source_geometry)
            if (
                root_correspondence.source_geometry_id != expected_geometry
                or root_correspondence.source_topology_id
                != domain.reference_mesh.topology_id
            ):
                raise ValueError(
                    "root_correspondence must bind the mapped source geometry and reference topology."
                )
            expected_names = tuple(
                sorted(f"sweep:{block.name}" for block in profile.blocks)
            )
            if root_correspondence.block_names != expected_names:
                raise ValueError(
                    "root_correspondence must name every generated sweep block."
                )
        elif root_correspondence is not None:
            raise ValueError("root_correspondence requires a MappedReferenceDomain.")
        block_names = tuple(f"sweep:{block.name}" for block in profile.blocks)
        regions = _block_region_bindings(block_regions, block_names)
        domain_vertex_map_: tuple[tuple[int, int], ...] | None = None
        if domain_vertex_map is not None:
            if not isinstance(domain, PiecewiseLinearDomain):
                raise ValueError("domain_vertex_map requires a PiecewiseLinearDomain.")
            raw_vertex_map = np.asarray(domain_vertex_map)
            if raw_vertex_map.dtype.kind not in "iu" or raw_vertex_map.shape != (
                domain.vertices.shape[0],
                2,
            ):
                raise TypeError(
                    "domain_vertex_map must contain one (profile vertex ID, station index) pair per domain vertex."
                )
            if raw_vertex_map.dtype.kind == "u" and np.any(
                raw_vertex_map > np.iinfo(np.int64).max
            ):
                raise ValueError("domain_vertex_map values must fit int64.")
            vertex_map = np.asarray(raw_vertex_map, dtype=np.int64)
            profile_vertex_ids = np.asarray(profile.vertex_global_ids, dtype=np.int64)
            if np.any(~np.isin(vertex_map[:, 0], profile_vertex_ids)):
                raise ValueError(
                    "domain_vertex_map must reference retained profile vertex IDs."
                )
            station_count = (
                control.schedule.layer_count
                if control.closed
                else control.schedule.layer_count + 1
            )
            if np.any(vertex_map[:, 1] < 0) or np.any(vertex_map[:, 1] >= station_count):
                raise ValueError(
                    "domain_vertex_map station indices lie outside the sweep node layers."
                )
            domain_vertex_map_ = tuple((int(row[0]), int(row[1])) for row in vertex_map)
            if len(set(domain_vertex_map_)) != len(domain_vertex_map_):
                raise ValueError(
                    "domain_vertex_map must not merge source vertex identities."
                )
        if (
            domain_vertex_map_ is not None
            and isinstance(domain, PiecewiseLinearDomain)
            and len(domain_vertex_map_) != domain.vertices.shape[0]
        ):
            raise ValueError("domain_vertex_map must cover every source-domain vertex.")
        deviation = _source_deviation(maximum_deviation)
        scope_id_banks, scope_key_banks, scope_set_ids = _sweep_source_scope_banks(
            profile, control, source, revision, fidelity_source, domain, regions
        )
        self.profile = profile
        self.profile_geometry = geometry
        self.control = control
        self.fidelity_source = fidelity_source
        self.maximum_deviation = deviation
        self.domain = domain
        self.root_correspondence = root_correspondence
        self.domain_vertex_map = domain_vertex_map_
        self.block_regions = regions
        self.scope_entity_id_banks = scope_id_banks
        self.scope_entity_key_banks = scope_key_banks
        self.scope_entity_set_ids = scope_set_ids
        self.source_id, self.source_revision = source, revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-sweep-source",
                "source": source,
                "revision": revision,
                "profile": profile.mesh_id,
                "profile_geometry": array_tree_fingerprint(geometry),
                "control": control.control_id,
                "maximum_deviation": deviation,
                "domain": None if domain is None else domain.domain_id,
                "root_correspondence": None
                if root_correspondence is None
                else root_correspondence.restriction_source_id,
                "domain_vertex_map": domain_vertex_map_,
                "block_regions": regions,
                "scope_entity_key_banks": scope_key_banks,
                "scope_entity_set_ids": scope_set_ids,
            }
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Sweep source dimension must be in 0..3.")
        return self.scope_entity_id_banks[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Sweep source dimension must be in 0..3.")
        return self.scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Sweep source dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("entity_id must be one stable integer scope ID.")
        identifiers = self.scope_entity_id_banks[dimension]
        position = int(np.searchsorted(identifiers, entity_id))
        if position >= identifiers.size or int(identifiers[position]) != entity_id:
            raise ValueError("entity_id is absent from this sweep source dimension.")
        return self.scope_entity_key_banks[dimension][position]


class _StoredOwnerDefault:
    """Distinguish an omitted restored owner from an explicit None."""


_STORED_OWNER = _StoredOwnerDefault()


@final
class NativeLayerCoreSource(StrictModule, NonTrainableState):
    """Bind accepted layers and a PLC core by source and material ancestry.

    Wall charts are rooted on the actual authored wall; boundary charts retain
    the complete fidelity-query atlas and may have a different mesh and scope.
    Projected wall rows retain their original B-Rep query and association policy
    so their parameters and residuals can be independently reissued. Conforming
    execution retains the exact control and policy that grew the wall.
    ``noncap_refinement="conforming"`` permits source-bound core boundary
    subdivision while preserving a fixed cap when ``cap_refinement="fixed"``.
    """

    layers: BoundaryLayerMesh
    layer_control: BoundaryLayerControl | None
    layer_policy: BoundaryLayerPolicy | None
    complex: PiecewiseLinearComplex
    core_power_source: NativePolyhedralSource | None
    vertex_layer_ids: np.ndarray
    cap_polygon_ids: np.ndarray
    cap_refinement: Literal["fixed", "conforming"] = eqx.field(static=True)
    noncap_refinement: Literal["fixed", "conforming"] = eqx.field(static=True)
    layer_regions: np.ndarray
    region_ids: tuple[str, ...] = eqx.field(static=True)
    region_entity_ids: np.ndarray
    region_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    core_region_map: np.ndarray
    source_boundary_scope: MeshingScope | None
    source_domain: MeshingDomain | PiecewiseLinearDomain | None
    source_wall_charts: SurfaceSourceCharts | None
    source_boundary_charts: SurfaceSourceCharts | None
    source_boundary_schedule: NativeSurfaceSchedule | None
    source_wall_projection: NativeBRepProjection | None
    source_wall_association_policy: AssociationPropagationPolicy | None
    maximum_deviation: float | None = eqx.field(static=True)
    fidelity_source: SourceBoundaryQuery | None
    core_facet_source_ids: tuple[str | None, ...] = eqx.field(static=True)
    core_facet_source_occurrence_paths: tuple[tuple[str, ...] | None, ...] = eqx.field(
        static=True
    )
    core_vertex_representatives: np.ndarray | None
    core_vertex_shifts: np.ndarray | None
    core_seam_polygon_pairs: np.ndarray | None
    periodic_topology_id: str | None = eqx.field(static=True)
    reference_source: NativeLayerCoreSource | None
    mapped_domain: MappedReferenceDomain | None
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        layers: BoundaryLayerMesh,
        complex_: PiecewiseLinearComplex,
        source_id: str,
        source_revision: str,
        /,
        *,
        vertex_layer_ids: ArrayLike,
        cap_polygon_ids: ArrayLike,
        layer_regions: ArrayLike,
        region_entity_ids: ArrayLike,
        region_occurrence_paths: tuple[tuple[str, ...], ...],
        region_ids: tuple[str, ...] | None = None,
        core_region_map: ArrayLike | None = None,
        source_boundary_scope: MeshingScope | None = None,
        source_domain: MeshingDomain | PiecewiseLinearDomain | None = None,
        fidelity_source: SourceBoundaryQuery | None = None,
        source_wall_charts: SurfaceSourceCharts | None = None,
        source_boundary_charts: SurfaceSourceCharts | None = None,
        source_boundary_schedule: NativeSurfaceSchedule | None = None,
        source_wall_projection: NativeBRepProjection | None = None,
        source_wall_association_policy: AssociationPropagationPolicy | None = None,
        cap_refinement: Literal["fixed", "conforming"] = "fixed",
        noncap_refinement: Literal["fixed", "conforming"] = "fixed",
        layer_control: BoundaryLayerControl | None = None,
        layer_policy: BoundaryLayerPolicy | None = None,
        maximum_deviation: float | None = None,
        core_facet_source_ids: tuple[str | None, ...] | None = None,
        core_facet_source_occurrence_paths: tuple[tuple[str, ...] | None, ...]
        | None = None,
        core_vertex_representatives: ArrayLike | None = None,
        core_vertex_shifts: ArrayLike | None = None,
        core_seam_polygon_pairs: ArrayLike | None = None,
        reference_source: NativeLayerCoreSource | None = None,
        mapped_domain: MappedReferenceDomain | None = None,
        core_power_source: NativePolyhedralSource | None = None,
    ) -> None:
        source_wall = layers.source_wall
        wall_association = layers.wall_association
        column_index = np.asarray(layers.column_index)
        if not isinstance(source_wall, CellMesh):
            raise TypeError("source_wall must be the original wall CellMesh.")
        if source_wall.topological_dimension != 2 or source_wall.ambient_dimension != 3:
            raise ValueError("source_wall must be an ambient-3 surface mesh.")
        if wall_association is not None and not isinstance(
            wall_association, GeometryAssociation
        ):
            raise TypeError("wall_association must be GeometryAssociation or None.")
        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        vertex_map = np.asarray(vertex_layer_ids)
        polygon_map = np.asarray(cap_polygon_ids)
        regions_input = np.asarray(layer_regions)
        columns = np.asarray(column_index)
        vertex_count = len(complex_.vertex_ids)
        layer_cell_count = sum(block.cell_count for block in layers.mesh.blocks)
        layer_vertex_count = layers.mesh.coordinates.shape[0]
        cap_cell_count = (
            0
            if layers.cap is None
            else sum(block.cell_count for block in layers.cap.blocks)
        )
        if (
            vertex_map.shape != (vertex_count,)
            or not np.issubdtype(vertex_map.dtype, np.integer)
            or np.any(vertex_map < -1)
            or np.any(vertex_map >= layer_vertex_count)
        ):
            raise ValueError(
                "vertex_layer_ids must map PLC vertices into layer vertices or -1."
            )
        polygon_count = np.asarray(complex_.polygon_facets).shape[0]
        if (
            polygon_map.shape != (cap_cell_count,)
            or not np.issubdtype(polygon_map.dtype, np.integer)
            or np.any(polygon_map < 0)
            or np.any(polygon_map >= polygon_count)
            or np.unique(polygon_map).size != polygon_map.size
        ):
            raise ValueError(
                "cap_polygon_ids must map each cap cell to a distinct PLC polygon."
            )
        if regions_input.shape != (layer_cell_count,) or not np.issubdtype(
            regions_input.dtype, np.integer
        ):
            raise ValueError(
                "layer_regions must contain one integer material per layer cell."
            )
        names, core_map = prepare_layer_region_identity(
            complex_,
            np.asarray(regions_input, dtype=np.int64),
            region_ids,
            core_region_map,
        )
        region_entity_ids_array = np.asarray(region_entity_ids)
        region_paths = tuple(
            tuple(normalized_identifier(part, "region_occurrence") for part in path)
            for path in region_occurrence_paths
        )
        if (
            region_entity_ids_array.shape != (len(names),)
            or not np.issubdtype(region_entity_ids_array.dtype, np.integer)
            or np.any(region_entity_ids_array < 0)
            or np.unique(region_entity_ids_array).size != len(names)
            or len(region_paths) != len(names)
        ):
            raise ValueError(
                "region_entity_ids and region_occurrence_paths must align with authored composite region IDs."
            )
        region_entity_ids_array = np.array(
            region_entity_ids_array, dtype=np.int64, copy=True
        )
        region_entity_ids_array.setflags(write=False)
        plc_id_banks, plc_key_banks, plc_set_ids = _prepare_plc_scope_banks(
            complex_, source, revision
        )
        region_order = np.argsort(region_entity_ids_array, kind="stable")
        region_scope_ids = np.array(
            region_entity_ids_array[region_order], dtype=np.int64, copy=True
        )
        region_scope_ids.setflags(write=False)
        region_scope_keys = tuple(
            (names[int(row)], region_paths[int(row)]) for row in region_order.tolist()
        )
        region_scope_set_id = canonical_fingerprint(
            {
                "kind": "layer-core-region-entity-set",
                "source_id": source,
                "source_revision": revision,
                "entities": region_scope_keys,
            }
        )
        scope_id_banks = (*plc_id_banks[:3], region_scope_ids)
        scope_key_banks = (*plc_key_banks[:3], region_scope_keys)
        scope_set_ids = (*plc_set_ids[:3], region_scope_set_id)
        if (
            columns.shape != (layer_cell_count,)
            or not np.issubdtype(columns.dtype, np.integer)
            or np.any(columns < 0)
        ):
            raise ValueError(
                "column_index must give one stable nonnegative column ID per layer cell."
            )
        if source_boundary_scope is not None and (
            source_boundary_scope.source_id != source
            or source_boundary_scope.source_revision != revision
        ):
            raise ValueError(
                "source_boundary_scope must bind source_id and source_revision."
            )
        if source_domain is not None and (
            source_domain.source_id != source or source_domain.source_revision != revision
        ):
            raise ValueError("source_domain must bind source_id and source_revision.")
        if fidelity_source is not None:
            _bound_source_query(fidelity_source, source, revision)
        if source_wall_charts is not None:
            if not isinstance(source_wall_charts, SurfaceSourceCharts):
                raise TypeError("source_wall_charts must be SurfaceSourceCharts or None.")
            if (source_wall_charts.source_id, source_wall_charts.source_revision) != (
                source,
                revision,
            ):
                raise ValueError("source_wall_charts must bind source identity.")
            source_wall_charts.require_root(source_wall, source_wall_charts.geometry)
            if source_domain is not None and (
                not isinstance(source_domain, MeshingDomain)
                or source_wall_charts.domain.domain_id != source_domain.domain_id
            ):
                raise ValueError(
                    "source_wall_charts must retain the original source domain."
                )
        if source_boundary_charts is not None:
            if not isinstance(source_boundary_charts, SurfaceSourceCharts):
                raise TypeError(
                    "source_boundary_charts must be SurfaceSourceCharts or None."
                )
            if (
                fidelity_source is None
                or (
                    source_domain is not None
                    and (
                        not isinstance(source_domain, MeshingDomain)
                        or source_boundary_charts.domain.domain_id
                        != source_domain.domain_id
                    )
                )
                or (
                    source_wall_charts is not None
                    and source_wall_charts.domain.domain_id
                    != source_boundary_charts.domain.domain_id
                )
            ):
                raise ValueError(
                    "Full boundary charts require their original domain and query."
                )
            source_boundary_charts.require_root(
                source_boundary_charts.mesh, source_boundary_charts.geometry
            )
            source_boundary_charts.require_bound(
                source_boundary_charts.mesh,
                source_boundary_charts.geometry,
                fidelity_source,
            )
        if source_boundary_schedule is not None:
            if not isinstance(source_boundary_schedule, NativeSurfaceSchedule):
                raise TypeError(
                    "source_boundary_schedule must be NativeSurfaceSchedule or None."
                )
            if source_boundary_charts is None:
                raise ValueError(
                    "An original surface schedule requires its source boundary atlas."
                )
            stored_curve = source_boundary_schedule.curve_schedule
            curve_schedule = NativeCurveSchedule(
                quadrature_intervals=stored_curve.quadrature_intervals,
                quadrature_order=stored_curve.quadrature_order,
                fidelity_samples=stored_curve.fidelity_samples,
            )
            schedule = NativeSurfaceSchedule(
                quality_angle_degrees=source_boundary_schedule.quality_angle_degrees,
                spacing_fraction=source_boundary_schedule.spacing_fraction,
                minimum_edge_samples=source_boundary_schedule.minimum_edge_samples,
                normal_deviation_aim=source_boundary_schedule.normal_deviation_aim,
                curve_schedule=curve_schedule,
            )
            if (
                curve_schedule.schedule_id != stored_curve.schedule_id
                or schedule.schedule_id != source_boundary_schedule.schedule_id
            ):
                raise ValueError("The original boundary surface schedule changed.")
        deviation = _source_deviation(maximum_deviation)
        if wall_association is not None and (
            wall_association.source_id != source
            or wall_association.source_revision != revision
        ):
            raise ValueError("wall_association must bind source_id and source_revision.")
        if (source_wall_projection is None) != (source_wall_association_policy is None):
            raise ValueError(
                "Wall projection and association policy must be retained together."
            )
        if source_wall_projection is not None:
            if not isinstance(
                source_wall_projection, NativeBRepProjection
            ) or not isinstance(
                source_wall_association_policy, AssociationPropagationPolicy
            ):
                raise TypeError(
                    "Projected wall authority requires its native BRep query and association policy."
                )
            if (
                wall_association is None
                or wall_association.association_kind is not GeometryAssociationKind.BREP
                or not isinstance(source_domain, MeshingDomain)
                or source_domain.brep_authority is None
                or source_wall_projection.model_id
                != source_domain.brep_authority.model_id
                or source_wall_projection.source_id != source
                or source_wall_projection.source_revision != revision
                or source_wall_projection.ambient_dimension != 3
            ):
                raise ValueError(
                    "Wall projection must bind the original BRep model and wall association."
                )
            stored_projection_policy = source_wall_projection.policy
            projection_policy = BRepProjectionPolicy(
                ambiguity_tolerance=stored_projection_policy.ambiguity_tolerance,
                parametric_tolerance=stored_projection_policy.parametric_tolerance,
                classifier_tolerance=stored_projection_policy.classifier_tolerance,
            )
            association_policy = AssociationPropagationPolicy(
                classification_tolerance=source_wall_association_policy.classification_tolerance,
                maximum_residual=source_wall_association_policy.maximum_residual,
            )
            if (
                projection_policy.policy_id != stored_projection_policy.policy_id
                or association_policy.policy_id
                != source_wall_association_policy.policy_id
            ):
                raise ValueError("Wall projection or association policy changed.")
        if cap_refinement not in ("fixed", "conforming"):
            raise ValueError("cap_refinement must be 'fixed' or 'conforming'.")
        if noncap_refinement not in ("fixed", "conforming"):
            raise ValueError("noncap_refinement must be 'fixed' or 'conforming'.")
        if (layer_control is None) != (layer_policy is None):
            raise ValueError(
                "Authored layer control and policy must be retained together."
            )
        if (cap_refinement == "conforming" or noncap_refinement == "conforming") and layer_control is None:
            raise ValueError(
                "Conforming execution requires the authored layer control and policy."
            )
        if layer_control is not None:
            if not isinstance(layer_control, BoundaryLayerControl) or not isinstance(
                layer_policy, BoundaryLayerPolicy
            ):
                raise TypeError(
                    "Authored layers require BoundaryLayerControl and BoundaryLayerPolicy."
                )
            model = (
                source_domain.brep_authority
                if isinstance(source_domain, MeshingDomain)
                else None
            )
            control = BoundaryLayerControl.from_stored_fields(
                layer_control, source_model=model
            )
            from ...discretization._cell_geometry_validity import CellValidityPolicy

            validity = CellValidityPolicy(
                relative_determinant_floor=layer_policy.validity.relative_determinant_floor,
                relative_planarity_tolerance=layer_policy.validity.relative_planarity_tolerance,
            )
            policy = BoundaryLayerPolicy(
                proximity_samples=layer_policy.proximity_samples,
                proximity_fraction=layer_policy.proximity_fraction,
                curvature_fraction=layer_policy.curvature_fraction,
                medial_tolerance=layer_policy.medial_tolerance,
                reduction_factor=layer_policy.reduction_factor,
                simplex_cap=layer_policy.simplex_cap,
                validity=validity,
            )
            if (
                control.control_id != layers.control_id
                or validity.policy_id != layer_policy.validity.policy_id
                or policy.policy_id != layer_policy.policy_id
                or policy.policy_id != layers.policy_id
            ):
                raise ValueError(
                    "Authored control and policy must match the accepted layers."
                )
        if (
            reference_source is not None
            and (reference_source.cap_refinement != cap_refinement
                 or reference_source.noncap_refinement != noncap_refinement)
        ):
            raise ValueError(
                "Mapped physical and reference sources must share cap refinement."
            )
        if (reference_source is None) != (mapped_domain is None):
            raise ValueError(
                "reference_source and mapped_domain must be supplied together."
            )
        if mapped_domain is not None and (
            mapped_domain.source_id != source or mapped_domain.source_revision != revision
        ):
            raise ValueError("mapped_domain must bind source_id and source_revision.")
        if core_power_source is not None:
            if not isinstance(core_power_source, NativePolyhedralSource):
                raise TypeError(
                    "core_power_source must be NativePolyhedralSource or None."
                )
            if (
                (core_power_source.source_id, core_power_source.source_revision)
                != (source, revision)
                or core_power_source.complex.complex_id != complex_.complex_id
                or (
                    None
                    if core_power_source.mapped_domain is None
                    else core_power_source.mapped_domain.domain_id
                )
                != (None if mapped_domain is None else mapped_domain.domain_id)
            ):
                raise ValueError(
                    "core_power_source must bind this fixed core PLC and mapped domain."
                )
        facet_sources = (
            tuple(complex_.facet_ids)
            if core_facet_source_ids is None
            else tuple(core_facet_source_ids)
        )
        if len(facet_sources) != len(complex_.facet_ids) or any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in facet_sources
        ):
            raise ValueError(
                "core_facet_source_ids must align with PLC facet identities."
            )
        facet_paths = (
            tuple(None for _ in facet_sources)
            if core_facet_source_occurrence_paths is None
            else tuple(
                None
                if path is None
                else tuple(
                    normalized_identifier(part, "source_occurrence") for part in path
                )
                for path in core_facet_source_occurrence_paths
            )
        )
        if (
            len(facet_paths) != len(facet_sources)
            or any(
                (facet is None) != (path is None)
                for facet, path in zip(facet_sources, facet_paths, strict=True)
            )
            or (
                source_domain is not None
                and any(
                    facet is not None and path is None
                    for facet, path in zip(facet_sources, facet_paths, strict=True)
                )
            )
        ):
            raise ValueError(
                "Core facet source IDs and occurrence paths must align exactly with source-domain incidence."
            )
        periodic_values = (
            core_vertex_representatives,
            core_vertex_shifts,
            core_seam_polygon_pairs,
        )
        periodic_topology = layers.mesh.periodic_topology
        periodic_topology_id = (
            None if periodic_topology is None else periodic_topology.periodic_topology_id
        )
        if all(value is None for value in periodic_values):
            representatives = shifts = seam_pairs = None
        elif any(value is None for value in periodic_values):
            raise ValueError(
                "Periodic core representatives, shifts, and seam pairs are joint data."
            )
        else:
            if periodic_topology is None:
                raise ValueError(
                    "Periodic core ancestry requires a periodic accepted layer mesh."
                )
            representatives = np.asarray(core_vertex_representatives)
            shifts = np.asarray(core_vertex_shifts)
            seam_pairs = np.asarray(core_seam_polygon_pairs)
            polygon_count = np.asarray(complex_.polygon_facets).shape[0]
            if (
                representatives.shape != (vertex_count,)
                or not np.issubdtype(representatives.dtype, np.integer)
                or np.any(representatives < 0)
                or np.any(representatives >= vertex_count)
                or shifts.shape != (vertex_count, periodic_topology.cell.rank)
                or not np.issubdtype(shifts.dtype, np.integer)
                or seam_pairs.ndim != 2
                or seam_pairs.shape[1] != 2
                or not np.issubdtype(seam_pairs.dtype, np.integer)
                or np.any(seam_pairs < 0)
                or np.any(seam_pairs >= polygon_count)
                or np.any(seam_pairs[:, 0] == seam_pairs[:, 1])
            ):
                raise ValueError(
                    "Periodic core ancestry must align with PLC vertices, the layer orbit rank, and polygons."
                )
            representatives = np.array(representatives, dtype=np.int64, copy=True)
            shifts = np.array(shifts, dtype=np.int64, copy=True)
            seam_pairs = np.array(seam_pairs, dtype=np.int64, copy=True)
            for array in (representatives, shifts, seam_pairs):
                array.setflags(write=False)
        vertex_map = np.array(vertex_map, dtype=np.int64, copy=True)
        polygon_map = np.array(polygon_map, dtype=np.int64, copy=True)
        regions_array = np.array(regions_input, dtype=np.int64, copy=True)
        for array in (vertex_map, polygon_map, regions_array, core_map):
            array.setflags(write=False)
        self.layers = layers
        self.layer_control = layer_control
        self.layer_policy = layer_policy
        self.complex = complex_
        self.vertex_layer_ids = vertex_map
        self.cap_polygon_ids = polygon_map
        self.cap_refinement = cap_refinement
        self.noncap_refinement = noncap_refinement
        self.layer_regions = regions_array
        self.region_ids = names
        self.region_entity_ids = region_entity_ids_array
        self.region_occurrence_paths = region_paths
        self.core_region_map = core_map
        self.source_boundary_scope = source_boundary_scope
        self.source_domain = source_domain
        self.fidelity_source = fidelity_source
        self.source_wall_charts = source_wall_charts
        self.source_boundary_charts = source_boundary_charts
        self.source_boundary_schedule = source_boundary_schedule
        self.source_wall_projection = source_wall_projection
        self.source_wall_association_policy = source_wall_association_policy
        self.maximum_deviation = deviation
        self.core_facet_source_ids = facet_sources
        self.core_facet_source_occurrence_paths = facet_paths
        self.core_vertex_representatives = representatives
        self.core_vertex_shifts = shifts
        self.core_seam_polygon_pairs = seam_pairs
        self.periodic_topology_id = periodic_topology_id
        self.reference_source = reference_source
        self.mapped_domain = mapped_domain
        self.core_power_source = core_power_source
        self.source_id, self.source_revision = source, revision
        self.scope_entity_id_banks = scope_id_banks
        self.scope_entity_key_banks = scope_key_banks
        self.scope_entity_set_ids = scope_set_ids
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-layer-core-source",
                "source": source,
                "revision": revision,
                "layers": layers.result_id,
                "complex": complex_.complex_id,
                "cap_refinement": cap_refinement,
                **({"noncap_refinement": noncap_refinement} if noncap_refinement != "fixed" else {}),
                "layer_control": None
                if layer_control is None
                else layer_control.control_id,
                "layer_policy": None if layer_policy is None else layer_policy.policy_id,
                "identity_maps": array_tree_fingerprint(
                    (vertex_map, polygon_map, regions_array)
                ),
                "region_ids": names,
                "region_entity_ids": array_tree_fingerprint(region_entity_ids_array),
                "region_occurrence_paths": region_paths,
                "core_region_map": array_tree_fingerprint(core_map),
                "original_boundary": None
                if source_boundary_scope is None
                else source_boundary_scope.scope_id,
                "original_domain": None
                if source_domain is None
                else source_domain.domain_id,
                "fidelity_source": None
                if fidelity_source is None
                else (fidelity_source.source_id, fidelity_source.source_revision),
                "source_wall_charts": None
                if source_wall_charts is None
                else source_wall_charts.source_chart_id,
                "source_boundary_charts": None
                if source_boundary_charts is None
                else source_boundary_charts.source_chart_id,
                "source_boundary_schedule": None
                if source_boundary_schedule is None
                else source_boundary_schedule.schedule_id,
                "source_wall_projection": None
                if source_wall_projection is None
                else source_wall_projection.projection_id,
                "source_wall_association_policy": None
                if source_wall_association_policy is None
                else source_wall_association_policy.policy_id,
                "maximum_deviation": deviation,
                "core_facet_source_ids": facet_sources,
                "core_facet_source_occurrence_paths": facet_paths,
                "core_power_source": None
                if core_power_source is None
                else core_power_source.binding_id,
                "periodic_topology_id": periodic_topology_id,
                "core_periodic_ancestry": array_tree_fingerprint(
                    (representatives, shifts, seam_pairs)
                ),
                "reference_source": None
                if reference_source is None
                else reference_source.binding_id,
                "mapped_domain": None
                if mapped_domain is None
                else mapped_domain.domain_id,
            }
        )

    @classmethod
    def fixed_core(
        cls: type[NativeLayerCoreSource],
        layers: BoundaryLayerMesh,
        outer: SurfaceModel,
        coordinate_contract: SpatialCoordinateContract,
        source_id: str,
        source_revision: str,
        /,
        *,
        region_id: str,
        region_entity_id: int,
        region_occurrence_path: tuple[str, ...],
        wall_occurrence_path: tuple[str, ...],
        outer_occurrence_path: tuple[str, ...],
        layer_control: BoundaryLayerControl,
        layer_policy: BoundaryLayerPolicy,
    ) -> NativeLayerCoreSource:
        """Bind a closed accepted layer cap to an independently authored outer shell."""
        from ...geometry._mesh_certificates import _same_tree_content
        from .._layer_core_topology import closed_fixed_core_source

        if type(layers) is not BoundaryLayerMesh or type(outer) is not SurfaceModel:
            raise TypeError("Fixed core requires accepted layers and an authoritative outer surface.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if not _same_tree_content(coordinate_contract, outer.metadata.coordinate_contract):
            raise ValueError("Outer surface and fixed core must use the same coordinate contract.")
        if not isinstance(layers.source_wall, CellMesh):
            raise ValueError("Fixed core must retain the actual authored layer wall.")
        if isinstance(region_entity_id, bool) or not isinstance(region_entity_id, int):
            raise TypeError("region_entity_id must be an integer.")
        if region_entity_id < 0:
            raise ValueError("region_entity_id must be non-negative.")
        if (
            not isinstance(wall_occurrence_path, tuple)
            or not isinstance(outer_occurrence_path, tuple)
            or not wall_occurrence_path
            or not outer_occurrence_path
        ):
            raise ValueError("Fixed core requires explicit wall and outer occurrence paths.")
        wall_path = (layers.source_wall.mesh_id, layers.source_wall.numeric_version, *wall_occurrence_path)
        outer_path = (outer.metadata.source_id, outer.metadata.source_revision, *outer_occurrence_path)
        return closed_fixed_core_source(
            layers, outer.mesh, coordinate_contract, source_id, source_revision,
            region_id=region_id, region_entity_id=region_entity_id,
            region_occurrence_path=region_occurrence_path,
            wall_occurrence_path=wall_path,
            outer_occurrence_path=outer_path,
            layer_control=layer_control, layer_policy=layer_policy,
        )

    @property
    def conforming_execution(self) -> bool:
        """Whether the core uses the source-authoritative conforming carrier."""
        return self.cap_refinement == "conforming" or self.noncap_refinement == "conforming"

    @classmethod
    def from_stored_fields(
        cls: type[NativeLayerCoreSource],
        value: NativeLayerCoreSource,
        /,
        *,
        layers: BoundaryLayerMesh | _StoredOwnerDefault = _STORED_OWNER,
        complex_: PiecewiseLinearComplex | _StoredOwnerDefault = _STORED_OWNER,
        layer_control: BoundaryLayerControl | None | _StoredOwnerDefault = _STORED_OWNER,
        layer_policy: BoundaryLayerPolicy | None | _StoredOwnerDefault = _STORED_OWNER,
        source_domain: MeshingDomain
        | PiecewiseLinearDomain
        | None
        | _StoredOwnerDefault = _STORED_OWNER,
        fidelity_source: SourceBoundaryQuery | None | _StoredOwnerDefault = _STORED_OWNER,
        source_wall_charts: SurfaceSourceCharts
        | None
        | _StoredOwnerDefault = _STORED_OWNER,
        source_boundary_charts: SurfaceSourceCharts
        | None
        | _StoredOwnerDefault = _STORED_OWNER,
        source_boundary_schedule: NativeSurfaceSchedule
        | None
        | _StoredOwnerDefault = _STORED_OWNER,
        source_wall_projection: NativeBRepProjection
        | None
        | _StoredOwnerDefault = _STORED_OWNER,
        source_wall_association_policy: AssociationPropagationPolicy
        | None
        | _StoredOwnerDefault = _STORED_OWNER,
        mapped_domain: MappedReferenceDomain | None | _StoredOwnerDefault = _STORED_OWNER,
        core_power_source: NativePolyhedralSource
        | None
        | _StoredOwnerDefault = _STORED_OWNER,
        reference_source: NativeLayerCoreSource
        | None
        | _StoredOwnerDefault = _STORED_OWNER,
    ) -> NativeLayerCoreSource:
        """Reconstruct the source from its retained owners and verify its identity."""
        if type(value) is not cls:
            raise TypeError(
                "stored_fields must have the registered layer/core source type."
            )
        reference = (
            value.reference_source
            if isinstance(reference_source, _StoredOwnerDefault)
            else reference_source
        )
        if reference is not None:
            if type(reference) is not cls or reference.reference_source is not None:
                raise ValueError(
                    "Mapped layer/core reference ancestry must not be recursive."
                )
            reference = cls.from_stored_fields(reference)
        stored_projection = (
            value.source_wall_projection
            if isinstance(source_wall_projection, _StoredOwnerDefault)
            else source_wall_projection
        )
        if stored_projection is not None:
            if not isinstance(stored_projection, NativeBRepProjection):
                raise TypeError("Stored wall projection must be NativeBRepProjection.")
            restored_domain = (
                value.source_domain
                if isinstance(source_domain, _StoredOwnerDefault)
                else source_domain
            )
            if (
                not isinstance(restored_domain, MeshingDomain)
                or restored_domain.brep_authority is None
            ):
                raise ValueError(
                    "Stored wall projection requires its original BRep model."
                )
            restored_projection = prepare_brep_projection(
                restored_domain.brep_authority,
                policy=stored_projection.policy,
                embedding=stored_projection.embedding,
            )
            if restored_projection.projection_id != stored_projection.projection_id:
                raise ValueError("Stored wall projection inputs changed identity.")
        else:
            restored_projection = None
        fresh = cls(
            value.layers if isinstance(layers, _StoredOwnerDefault) else layers,
            value.complex if isinstance(complex_, _StoredOwnerDefault) else complex_,
            value.source_id,
            value.source_revision,
            vertex_layer_ids=value.vertex_layer_ids,
            cap_polygon_ids=value.cap_polygon_ids,
            layer_regions=value.layer_regions,
            region_entity_ids=value.region_entity_ids,
            region_occurrence_paths=value.region_occurrence_paths,
            region_ids=value.region_ids,
            core_region_map=value.core_region_map,
            source_boundary_scope=value.source_boundary_scope,
            source_domain=value.source_domain
            if isinstance(source_domain, _StoredOwnerDefault)
            else source_domain,
            fidelity_source=value.fidelity_source
            if isinstance(fidelity_source, _StoredOwnerDefault)
            else fidelity_source,
            source_wall_charts=value.source_wall_charts
            if isinstance(source_wall_charts, _StoredOwnerDefault)
            else source_wall_charts,
            source_boundary_charts=value.source_boundary_charts
            if isinstance(source_boundary_charts, _StoredOwnerDefault)
            else source_boundary_charts,
            source_boundary_schedule=value.source_boundary_schedule
            if isinstance(source_boundary_schedule, _StoredOwnerDefault)
            else source_boundary_schedule,
            source_wall_projection=restored_projection,
            source_wall_association_policy=value.source_wall_association_policy
            if isinstance(source_wall_association_policy, _StoredOwnerDefault)
            else source_wall_association_policy,
            layer_control=value.layer_control
            if isinstance(layer_control, _StoredOwnerDefault)
            else layer_control,
            layer_policy=value.layer_policy
            if isinstance(layer_policy, _StoredOwnerDefault)
            else layer_policy,
            cap_refinement=value.cap_refinement,
            noncap_refinement=value.noncap_refinement,
            maximum_deviation=value.maximum_deviation,
            core_facet_source_ids=value.core_facet_source_ids,
            core_facet_source_occurrence_paths=value.core_facet_source_occurrence_paths,
            core_vertex_representatives=value.core_vertex_representatives,
            core_vertex_shifts=value.core_vertex_shifts,
            core_seam_polygon_pairs=value.core_seam_polygon_pairs,
            reference_source=reference,
            mapped_domain=value.mapped_domain
            if isinstance(mapped_domain, _StoredOwnerDefault)
            else mapped_domain,
            core_power_source=value.core_power_source
            if isinstance(core_power_source, _StoredOwnerDefault)
            else core_power_source,
        )
        if (
            fresh.binding_id != value.binding_id
            or fresh.periodic_topology_id != value.periodic_topology_id
            or fresh.scope_entity_key_banks != value.scope_entity_key_banks
            or fresh.scope_entity_set_ids != value.scope_entity_set_ids
            or len(fresh.scope_entity_id_banks) != len(value.scope_entity_id_banks)
            or any(
                not np.array_equal(current, stored)
                for current, stored in zip(
                    fresh.scope_entity_id_banks, value.scope_entity_id_banks, strict=True
                )
            )
        ):
            raise ValueError("Stored layer/core source identity no longer validates.")
        from ...geometry._mapped_reference_boundary import MappedReferenceBoundarySource

        query = fresh.fidelity_source
        if isinstance(query, MappedReferenceBoundarySource):
            query.require_current()
            if fresh.mapped_domain is None or not isinstance(
                query.authority, (NativePlcSource, NativeMappedHexSource)
            ):
                raise ValueError(
                    "Mapped layer/core query requires its full PLC authority."
                )
            _mapped_layer_core_facet_relation(fresh, query.authority)
        return fresh

    def validate_restored(self, /) -> None:
        """Reconstruct and validate the retained layer/core source authority."""
        type(self).from_stored_fields(self)

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Layer source dimension must be in 0..3.")
        return self.scope_entity_id_banks[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Layer source dimension must be in 0..3.")
        return self.scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Layer source dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("entity_id must be one stable integer scope ID.")
        ids = self.scope_entity_id_banks[dimension]
        position = int(np.searchsorted(ids, entity_id))
        if position >= ids.size or int(ids[position]) != entity_id:
            raise ValueError("entity_id is absent from the layer source dimension.")
        return self.scope_entity_key_banks[dimension][position]


def _mapped_layer_core_facet_relation(
    source: NativeLayerCoreSource, authority: NativePlcSource | NativeMappedHexSource, /
) -> tuple[int, ...]:
    """Prove fixed-core polygon ancestry in the independently authored full PLC.

    Returns the full-PLC polygon row for each core polygon, or -1 for an
    internal same-material polygon. No vertex identity is inferred from position.
    """
    from .. import _layer_core_topology as topology

    reference = source.reference_source
    domain = source.mapped_domain
    authority_domain = (
        authority.mapped_domain
        if isinstance(authority, NativePlcSource)
        else authority.domain
    )
    if reference is None or domain is None or authority_domain is None:
        raise ValueError("Mapped core facet ancestry requires both retained authorities.")
    if (
        source.core_facet_source_ids != reference.core_facet_source_ids
        or source.core_facet_source_occurrence_paths
        != reference.core_facet_source_occurrence_paths
        or source.region_ids != reference.region_ids
        or source.region_occurrence_paths != reference.region_occurrence_paths
        or not np.array_equal(source.core_region_map, reference.core_region_map)
    ):
        raise ValueError("Mapped core and reference source ancestry differs.")
    if authority_domain.domain_id != domain.domain_id:
        raise ValueError("Mapped core and full PLC have different root domains.")
    authority.validate_restored()
    mapping, cap_ids, polygons, points = topology.prepare_identity(
        reference.layers,
        reference.complex,
        reference.vertex_layer_ids,
        reference.cap_polygon_ids,
    )
    combined, layer_faces = topology.combined_domain(
        reference.layers,
        reference.complex,
        np.asarray(reference.layer_regions, dtype=np.int64),
        mapping,
        cap_ids,
        polygons,
        points,
        source.source_id,
        reference.region_ids,
        np.asarray(reference.core_region_map, dtype=np.int64),
        reference.region_occurrence_paths,
    )
    if combined.domain_id != domain.reference_domain.domain_id:
        raise ValueError(
            "Mapped root domain is not the independently reconstructed layer/core domain."
        )
    roots = domain.reference_domain_vertex_root_ids
    if roots is None or len(roots) != len(points):
        raise ValueError(
            "Mapped core requires complete declared domain/root vertex ancestry."
        )
    core_roots = np.asarray(roots, dtype=np.int64)[mapping]
    relation = authority.root_relation
    if relation is None:
        raise ValueError("Full PLC authority has no mapped root relation.")
    full = authority.complex
    full_offsets = np.asarray(full.polygon_offsets, dtype=np.int64)
    full_vertices = np.asarray(full.polygon_vertices, dtype=np.int64)
    full_facets = np.asarray(full.polygon_facets, dtype=np.int64)
    full_roots = np.asarray(relation.vertex_root_global_ids, dtype=np.int64)
    full_rows: dict[tuple[int, ...], int] = {}
    for row in range(len(full_facets)):
        vertices = full_vertices[full_offsets[row] : full_offsets[row + 1]]
        triangle = tuple(int(full_roots[int(vertex)]) for vertex in vertices)
        key = tuple(sorted(triangle))
        if len(triangle) != 3 or key in full_rows:
            raise ValueError("Full mapped PLC has ambiguous triangular ancestry.")
        full_rows[key] = row
    cap_set = set(int(value) for value in cap_ids)
    region_keys = tuple(
        zip(reference.region_ids, reference.region_occurrence_paths, strict=True)
    )
    full_region_keys = tuple(
        zip(full.region_ids, full.region_occurrence_paths, strict=True)
    )
    output: list[int] = []
    matched_rows: set[int] = set()
    for row, polygon in enumerate(polygons):
        facet = int(reference.complex.polygon_facets[row])
        positive, negative = (
            int(value) for value in reference.complex.facet_regions[facet]
        )
        core_map = reference.core_region_map
        positive = int(core_map[positive]) if positive >= 0 else -1
        negative = int(core_map[negative]) if negative >= 0 else -1
        if row in cap_set:
            incidents = layer_faces[
                tuple(sorted(int(value) for value in mapping[polygon]))
            ]
            if len(incidents) != 1:
                raise ValueError("Core cap must have exactly one accepted layer side.")
            negative = incidents[0].region
        if positive == negative:
            output.append(-1)
            continue
        triangle = tuple(int(core_roots[int(vertex)]) for vertex in polygon)
        full_row = full_rows.get(tuple(sorted(triangle)))
        if full_row is None:
            raise ValueError("Core source polygon has no full PLC root triangle.")
        if full_row in matched_rows:
            raise ValueError("Multiple core polygons claim the same full PLC triangle.")
        matched_rows.add(full_row)
        source_key = (
            reference.core_facet_source_ids[facet],
            reference.core_facet_source_occurrence_paths[facet],
        )
        full_key = authority.source_entity_key_for_scope(
            2, int(full.facet_entity_ids[int(full_facets[full_row])])
        )
        if source_key != full_key:
            raise ValueError(
                "Core facet changes its independently authored full source identity."
            )
        full_triangle = tuple(
            int(full_roots[int(vertex)])
            for vertex in full_vertices[
                full_offsets[full_row] : full_offsets[full_row + 1]
            ]
        )
        same = _mapped_triangle_key(triangle) == _mapped_triangle_key(full_triangle)
        expected = (positive, negative) if same else (negative, positive)
        actual = tuple(
            -1 if int(value) < 0 else region_keys.index(full_region_keys[int(value)])
            for value in full.facet_regions[full_facets[full_row]]
        )
        if expected != actual:
            raise ValueError("Core/full PLC facet orientation or material sides differ.")
        output.append(full_row)
    return tuple(output)


__all__ = [
    "NativeCurveSource",
    "NativeImplicitSource",
    "NativePlanarSource",
    "NativePlcSource",
    "NativePolyhedralSource",
    "NativeLayerCoreSource",
    "NativeMappedHexSource",
    "NativeStructuredSource",
    "NativeSweepSource",
]
