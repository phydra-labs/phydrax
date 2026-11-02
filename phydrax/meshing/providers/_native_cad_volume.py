from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import finite_real_scalar
from ...discretization._cell_complex import PolygonalConnectivity
from ...discretization._cell_geometry import CellGeometrySpec
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_mesh import SimplicialConnectivity
from ...discretization._coordinate_enclosure import ExactScalar
from ...discretization._reference_cell import reference_cell_topology
from ...geometry._domain_boundary import (
    exact_plane_chart,
    ExactPlaneChart,
    MeshingDomainBoundarySource,
)
from ...geometry._mesh_certificates import _same_tree_content
from ...geometry._meshing_domain import _PlacedSurfacePatch, MeshingDomain, PatchCurveUse
from ...geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
    SourceBoundaryQuery,
)
from ...geometry._surface_source_support import SurfaceSourceCharts
from ...geometry.brep._model import BRepGeometry, BRepModel
from ...geometry.brep._patches import PlanePatch
from ...geometry.brep._projection_contracts import brep_entity_id
from ...typing import checked, parse
from .._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from .._certification_schedule import MeshCertificationSchedule
from .._contracts import VolumeMeshingSpec
from .._controls import (
    FeatureKind,
    HoleSeed,
    PatchControl,
    ProtectedFeature,
    RegionControl,
    RegionSeed,
)
from .._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from .._plc_source import _certify_cad_surface, PiecewiseLinearComplex, PlcBoundaryPolicy
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import UniformSizeControl
from .._volume_generation import (
    generate_plc_volume,
    NativeVolumeSchedule,
    VolumeConstruction,
)
from ._native_options import NativeMeshingOptions
from ._native_plc_geometry import (
    physical_plc_tetrahedra,
    simplex_volume_entities,
    volume_cell_entity_rows,
)
from ._native_sources import NativePlcSource
from ._native_volume import (
    _organization as _plc_organization,
    _target_support,
    _volume_compliance,
    volume_support_issues,
)


def _readonly(value: np.ndarray, dtype: np.dtype, /) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _domain_stratum(
    domain: MeshingDomain, dimension: int, /
) -> tuple[tuple[int, ...], tuple[tuple[str, ...], ...]]:
    if dimension == 3:
        return domain.region_source_indices, domain.region_source_occurrences
    return domain.source_indices[dimension], domain.source_occurrences[dimension]


def _resolve_brep_row(
    domain: MeshingDomain,
    dimension: int,
    source_index: int,
    occurrence_path: tuple[str, ...],
    /,
) -> tuple[int, str]:
    indices, paths = _domain_stratum(domain, dimension)
    matches = tuple(
        row
        for row, (index, path) in enumerate(zip(indices, paths, strict=True))
        if index == source_index and path == occurrence_path
    )
    if len(matches) != 1:
        raise ValueError(
            "A surface B-Rep association must resolve through one exact domain source occurrence."
        )
    row = matches[0]
    source_key = domain.entity_id(dimension, row)
    if (
        domain.resolve_entity_id(dimension, source_key, occurrence_path=occurrence_path)
        != row
    ):
        raise ValueError("MeshingDomain source resolution changed a B-Rep occurrence.")
    return row, source_key


class _BRepRowResolver:
    """Resolve B-Rep occurrences of one domain once per distinct occurrence.

    Association rows of one pass repeat a few source entities; each distinct
    ``(dimension, index, occurrence)`` is resolved and keyed once, and the
    resolver dies with the pass.
    """

    __slots__ = ("_domain", "_resolved")

    def __init__(self, domain: MeshingDomain, /) -> None:
        self._domain = domain
        self._resolved: dict[tuple[int, int, tuple[str, ...]], tuple[int, str, str]] = {}

    def __call__(
        self, dimension: int, source_index: int, occurrence_path: tuple[str, ...], /
    ) -> tuple[int, str, str]:
        """Return the domain row, domain source key and B-Rep entity id."""
        key = (dimension, source_index, occurrence_path)
        resolved = self._resolved.get(key)
        if resolved is None:
            domain = self._domain
            row, source_key = _resolve_brep_row(
                domain, dimension, source_index, occurrence_path
            )
            resolved = (
                row,
                source_key,
                brep_entity_id(
                    domain.source_id,
                    domain.source_revision,
                    dimension,
                    source_index,
                    occurrence_path=occurrence_path,
                ),
            )
            self._resolved[key] = resolved
        return resolved


def _cad_curve_parameter_sense(domain: MeshingDomain, curve: int, /) -> int:
    """Relate the owner's normalized chain gauge to the native B-Rep gauge."""
    if not 0 <= curve < len(domain.curves):
        raise ValueError("CAD curve ancestry names an absent source curve.")
    patch, loop, position = (int(value) for value in domain.curve_owners[curve])
    if min(patch, loop, position) < 0:
        raise ValueError("CAD curve has no unique owner gauge.")
    try:
        use = domain.patches[patch].loops[loop][position]
    except IndexError as error:
        raise ValueError("CAD curve has no unique owner gauge.") from error
    if not isinstance(use, PatchCurveUse) or use.curve != curve or use.first == use.last:
        raise ValueError("CAD curve has no unique nondegenerate owner gauge.")
    return 1 if use.last > use.first else -1


def _association_for(
    surface: CellMeshingResult | SurfaceSourceCharts, dimension: int, /, *, required: bool
) -> GeometryAssociation | None:
    entity_set = surface.mesh.entity_set(dimension)
    expected_kind = (
        GeometryAssociationKind.CURVE if dimension == 1 else GeometryAssociationKind.BREP
    )
    matches = tuple(
        association
        for association in surface.associations
        if association.association_kind is expected_kind
        and association.target_entity_set_id == entity_set.entity_set_id
    )
    if len(matches) > 1 or (required and len(matches) != 1):
        raise ValueError(
            f"The accepted CAD surface must carry one {expected_kind.value} association for dimension {dimension}."
        )
    if not matches:
        return None
    association = matches[0]
    identifiers = np.asarray(association.target_global_ids, dtype=np.int64)
    entity_ids = np.asarray(entity_set.entity_ids, dtype=np.int64)
    target_coverage = (
        identifiers.ndim == 1
        and np.unique(identifiers).size == identifiers.size
        and bool(np.all(np.isin(identifiers, entity_ids)))
        if dimension == 1
        else np.array_equal(np.sort(identifiers), np.sort(entity_ids))
    )
    if (
        not association.source_id
        or not association.source_revision
        or not target_coverage
        or not np.all(np.asarray(association.resolved, dtype=np.bool_))
        or np.any(np.asarray(association.ambiguous, dtype=np.bool_))
    ):
        raise ValueError(
            "Accepted CAD surface associations must resolve every target entity."
        )
    if dimension == 1:
        dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
        if (
            dimensions.shape != identifiers.shape
            or np.any(dimensions != 1)
            or any(
                role is not GeometrySourceEntityRole.EDGE
                for role in association.source_roles
            )
            or any(value is None for value in association.source_parameter_intervals)
            or any(value is not None for value in association.source_parameter_points)
        ):
            raise ValueError(
                "Accepted CAD curve-edge associations must retain one exact interval and B-Rep occurrence per target edge."
            )
    return association


def _association_row_map(association: GeometryAssociation, /) -> dict[int, int]:
    return {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(association.target_global_ids, dtype=np.int64).tolist()
        )
    }


def _exact_surface_points(
    surface: CellMeshingResult | SurfaceSourceCharts, /
) -> tuple[np.ndarray, tuple[tuple[ExactScalar, ...], ...]]:
    mesh = surface.mesh
    if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
        raise ValueError(
            "CAD tetrahedral filling requires an accepted triangular surface mesh in 3D."
        )
    point_bank: list[tuple[ExactScalar, ...] | None] = [None] * (
        mesh.coordinates.shape[0]
    )
    for block in mesh.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Every CAD surface source cell must be a triangle.")
        references = tuple(
            tuple(Fraction.from_float(float(value)) for value in vertex)
            for vertex in reference_cell_topology("triangle").vertices
        )
        cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        corners = surface.geometry.evaluate_exact(
            mesh, cell_ids, (references,) * cell_ids.size
        )
        cells = np.asarray(block.vertices, dtype=np.int64)
        for cell, vertices in zip(corners, cells.tolist(), strict=True):
            if len(cell) != 3:
                raise ValueError(
                    "Surface geometry evaluation must return three triangle corners."
                )
            for vertex, point in zip(vertices, cell, strict=True):
                exact = tuple(point)
                previous = point_bank[int(vertex)]
                if previous is not None and previous != exact:
                    raise ValueError(
                        "CAD surface geometry is discontinuous at a shared mesh vertex."
                    )
                point_bank[int(vertex)] = exact
    if any(point is None for point in point_bank):
        raise ValueError(
            "Every accepted CAD surface vertex must belong to a source triangle."
        )
    exact_points = tuple(point for point in point_bank if point is not None)
    points = np.asarray(
        [[float(component) for component in point] for point in exact_points],
        dtype=np.float64,
    )
    if points.shape != (mesh.coordinates.shape[0], 3) or not np.all(np.isfinite(points)):
        raise ValueError("CAD surface source coordinates must be finite three-vectors.")
    return points, exact_points


def _plc_source_triangle_id(
    triangle: tuple[int, int, int],
    facet: int,
    vertex_ids: tuple[str, ...],
    vertex_occurrence_paths: tuple[tuple[str, ...], ...],
    facet_ids: tuple[str, ...],
    facet_paths: tuple[tuple[str, ...], ...],
) -> str:
    vertices = tuple(
        (vertex_ids[index], vertex_occurrence_paths[index]) for index in triangle
    )
    rotations = tuple(vertices[offset:] + vertices[:offset] for offset in range(3))
    return canonical_fingerprint(
        {
            "kind": "plc-source-triangle",
            "facet": (facet_ids[facet], facet_paths[facet]),
            "oriented_vertices": min(rotations),
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class _CadPlcData:
    complex: PiecewiseLinearComplex
    vertex_brep_dimensions: np.ndarray
    vertex_brep_indices: np.ndarray
    vertex_parameters: np.ndarray
    segment_curve_rows: np.ndarray
    segment_brep_indices: np.ndarray
    source_triangle_ids: tuple[str, ...]
    source_triangle_cell_ids: np.ndarray
    source_triangle_patches: np.ndarray
    source_triangle_parameters: np.ndarray
    source_triangle_parameter_exact: tuple[tuple[tuple[Fraction, Fraction], ...], ...]


def _cad_plc_data(
    domain: MeshingDomain,
    surface: CellMeshingResult,
    surface_source: SurfaceSourceCharts,
    boundary: PlcBoundaryPolicy,
    deviation: float | None,
    /,
) -> _CadPlcData:
    """PLC of an accepted CAD surface mesh; facet rows are source patch rows.

    Exact planar patches declare their facet plane. A vertex of an accepted
    surface lies within the source ``deviation`` bound of its patch, which is
    the declared plane tolerance (zero without a bound: rounding only).
    """
    domain.require_current(domain.source_id, domain.source_revision)
    if domain.brep_authority is None or domain.source_kinds != ("vertex", "edge", "face"):
        raise ValueError(
            "CAD volume preparation requires a BRep-authoritative MeshingDomain."
        )
    active_curve_rows = {
        row for row in range(len(domain.curves)) if domain.curve_uses(row)
    }
    vertex_association = _association_for(surface_source, 0, required=True)
    edge_association = _association_for(
        surface_source, 1, required=bool(active_curve_rows)
    )
    face_association = _association_for(surface_source, 2, required=True)
    if vertex_association is None or face_association is None:
        raise ValueError("CAD source charts require vertex and face associations.")
    for dimension, chart_association in enumerate(
        (vertex_association, edge_association, face_association)
    ):
        accepted = _association_for(
            surface, dimension, required=chart_association is not None
        )
        if (accepted is None) != (chart_association is None) or (
            accepted is not None
            and chart_association is not None
            and accepted.association_id != chart_association.association_id
        ):
            raise ValueError(
                "Accepted CAD surface associations differ from their bound source charts."
            )
    for association in (vertex_association, edge_association, face_association):
        if association is None:
            continue
        if (
            association.source_id != domain.source_id
            or association.source_revision != domain.source_revision
        ):
            raise ValueError(
                "Accepted CAD surface associations must bind the same BRep revision."
            )

    _, exact_points = _exact_surface_points(surface)
    if not isinstance(surface_source, SurfaceSourceCharts):
        raise TypeError("CAD volume construction requires retained SurfaceSourceCharts.")
    source_chart_cell_ids = np.asarray(surface_source.cell_global_ids, dtype=np.int64)
    source_chart_patches = np.asarray(surface_source.root_patches, dtype=np.int64)
    source_chart_parameters = np.asarray(surface_source.root_parameters, dtype=np.float64)
    source_chart_parameter_exact = surface_source.root_parameter_exact
    chart_row_by_cell_id = {
        int(identifier): row
        for row, identifier in enumerate(source_chart_cell_ids.tolist())
    }
    if (
        source_chart_parameters.shape != (source_chart_cell_ids.size, 3, 2)
        or source_chart_patches.shape != source_chart_cell_ids.shape
        or len(source_chart_parameter_exact) != source_chart_cell_ids.size
        or len(chart_row_by_cell_id) != source_chart_cell_ids.size
    ):
        raise ValueError("CAD source root charts lost their original triangle ancestry.")
    mesh = surface.mesh
    mesh_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    vertex_association_rows = _association_row_map(vertex_association)
    vertex_dimensions = np.asarray(vertex_association.source_dimensions, dtype=np.int64)
    vertex_indices = np.asarray(vertex_association.source_indices, dtype=np.int64)
    vertex_parameters = np.asarray(vertex_association.parameters, dtype=np.float64)
    vertex_paths = vertex_association.source_occurrence_paths
    vertex_ids_by_surface_row: list[str] = []
    vertex_paths_by_surface_row: list[tuple[str, ...]] = []
    vertex_dimensions_by_surface_row: list[int] = []
    vertex_indices_by_surface_row: list[int] = []
    vertex_parameters_by_surface_row: list[np.ndarray] = []
    resolve = _BRepRowResolver(domain)
    for row, global_id in enumerate(mesh_vertex_ids.tolist()):
        try:
            association_row = vertex_association_rows[int(global_id)]
        except KeyError as error:
            raise ValueError(
                "CAD surface vertex association lost a mesh entity."
            ) from error
        dimension = int(vertex_dimensions[association_row])
        source_index = int(vertex_indices[association_row])
        if dimension not in (0, 1, 2):
            raise ValueError(
                "A CAD boundary vertex must resolve to a source vertex, curve, or face."
            )
        path = tuple(vertex_paths[association_row])
        _, domain_key, expected_brep_id = resolve(dimension, source_index, path)
        if vertex_association.source_entity_ids[association_row] != expected_brep_id:
            raise ValueError(
                "CAD surface vertex association changed its B-Rep entity identity."
            )
        parameters = np.asarray(vertex_parameters[association_row], dtype=np.float64)
        if dimension == 1 and not np.isfinite(parameters[0]):
            raise ValueError(
                "CAD source-edge vertices require a finite source curve parameter."
            )
        if dimension == 2 and not np.all(np.isfinite(parameters)):
            raise ValueError("CAD source-face vertices require finite chart parameters.")
        parameter_key = (
            ()
            if dimension == 0
            else tuple(
                float(value).hex() if np.isfinite(value) else None
                for value in parameters.tolist()
            )
        )
        identity = canonical_fingerprint(
            {
                "kind": "cad-plc-source-vertex",
                "source_entity": domain_key,
                "source_dimension": dimension,
                "source_parameters": parameter_key,
            }
        )
        vertex_ids_by_surface_row.append(identity)
        vertex_paths_by_surface_row.append(path)
        vertex_dimensions_by_surface_row.append(dimension)
        vertex_indices_by_surface_row.append(source_index)
        vertex_parameters_by_surface_row.append(parameters)

    unique_vertex_rows: dict[tuple[str, tuple[str, ...]], int] = {}
    plc_points: list[np.ndarray] = []
    plc_vertex_ids: list[str] = []
    plc_vertex_paths: list[tuple[str, ...]] = []
    plc_vertex_dimensions: list[int] = []
    plc_vertex_indices: list[int] = []
    plc_vertex_parameters: list[np.ndarray] = []
    mesh_row_to_plc: list[int] = []
    for row, (identifier, path) in enumerate(
        zip(vertex_ids_by_surface_row, vertex_paths_by_surface_row, strict=True)
    ):
        key = (identifier, path)
        existing = unique_vertex_rows.get(key)
        if existing is None:
            existing = len(plc_points)
            unique_vertex_rows[key] = existing
            plc_points.append(np.asarray(exact_points[row], dtype=object))
            plc_vertex_ids.append(identifier)
            plc_vertex_paths.append(path)
            plc_vertex_dimensions.append(vertex_dimensions_by_surface_row[row])
            plc_vertex_indices.append(vertex_indices_by_surface_row[row])
            plc_vertex_parameters.append(vertex_parameters_by_surface_row[row])
        elif tuple(plc_points[existing].tolist()) != exact_points[row]:
            raise ValueError(
                "One CAD source vertex identity maps to different exact points."
            )
        mesh_row_to_plc.append(existing)

    face_association_rows = _association_row_map(face_association)
    face_dimensions = np.asarray(face_association.source_dimensions, dtype=np.int64)
    face_indices = np.asarray(face_association.source_indices, dtype=np.int64)
    face_paths = face_association.source_occurrence_paths
    face_orientations = np.asarray(face_association.orientations, dtype=np.int64)
    face_domain_rows: dict[int, int] = {}
    resolve = _BRepRowResolver(domain)
    for global_id, association_row in face_association_rows.items():
        dimension = int(face_dimensions[association_row])
        if dimension != 2:
            raise ValueError(
                "Every accepted surface triangle must have one B-Rep face source."
            )
        source_index = int(face_indices[association_row])
        path = tuple(face_paths[association_row])
        domain_row, _, expected_brep_id = resolve(2, source_index, path)
        if face_association.source_entity_ids[association_row] != expected_brep_id:
            raise ValueError("CAD surface face association changed its B-Rep identity.")
        orientation = int(face_orientations[association_row])
        if orientation not in (-1, 1):
            raise ValueError(
                "CAD surface triangle orientation must be resolved against its source face."
            )
        face_domain_rows[int(global_id)] = domain_row

    triangles: list[tuple[int, int, int]] = []
    triangle_facets: list[int] = []
    triangle_cell_ids: list[int] = []
    triangle_parameters: list[np.ndarray] = []
    triangle_parameter_exact: list[tuple[tuple[Fraction, Fraction], ...]] = []
    covered_patches: set[int] = set()
    for block in mesh.blocks:
        cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        cells = np.asarray(block.vertices, dtype=np.int64)
        for cell_id, cell in zip(cell_ids.tolist(), cells.tolist(), strict=True):
            try:
                association_row = face_association_rows[int(cell_id)]
                patch = face_domain_rows[int(cell_id)]
                chart_row = chart_row_by_cell_id[int(cell_id)]
            except KeyError as error:
                raise ValueError(
                    "CAD surface triangle association lost its source chart ancestry."
                ) from error
            if int(source_chart_patches[chart_row]) != patch:
                raise ValueError(
                    "CAD surface chart and source-face patch ancestry disagree."
                )
            orientation = int(face_orientations[association_row])
            if len(cell) != 3:
                raise ValueError("Every accepted CAD source cell must be a triangle.")
            polygon = (
                mesh_row_to_plc[int(cell[0])],
                mesh_row_to_plc[int(cell[1])],
                mesh_row_to_plc[int(cell[2])],
            )
            parameters = np.array(source_chart_parameters[chart_row], copy=True)
            exact_parameters = surface_source.root_parameter_chart(chart_row)
            exact_numeric = np.asarray(
                [[float(value) for value in point] for point in exact_parameters],
                dtype=np.float64,
            )
            if not np.array_equal(exact_numeric, source_chart_parameters[chart_row]):
                raise ValueError(
                    "CAD source UV values differ from their exact root chart bank."
                )
            if orientation < 0:
                polygon = (polygon[0], polygon[2], polygon[1])
                parameters = parameters[[0, 2, 1]]
                exact_parameters = (
                    exact_parameters[0],
                    exact_parameters[2],
                    exact_parameters[1],
                )
            triangles.append(polygon)
            triangle_facets.append(patch)
            triangle_cell_ids.append(int(cell_id))
            triangle_parameters.append(parameters)
            triangle_parameter_exact.append(exact_parameters)
            covered_patches.add(patch)
    if covered_patches != set(range(len(domain.patches))):
        raise ValueError(
            "Accepted surface boundary must cover every declared CAD source patch."
        )

    facet_regions = np.full((len(domain.patches), 2), -1, dtype=np.int64)
    for region_row, region in enumerate(domain.regions):
        for patch, sign in region.boundary:
            # The accepted loop is already oriented to the source patch normal,
            # including patch.orientation; do not apply that sign again here.
            # A positive region boundary sign puts the region on the opposite
            # (negative) side in the CAPI (positive, negative) pair.
            side = 1 if sign > 0 else 0
            if facet_regions[patch, side] >= 0:
                raise ValueError(
                    "CAD source patch has ambiguous oriented material-side incidence."
                )
            facet_regions[patch, side] = region_row
    if np.any(np.all(facet_regions < 0, axis=1)):
        raise ValueError(
            "Every CAD source patch must bound at least one material region."
        )

    segments: list[tuple[int, int]] = []
    segment_ids: list[str] = []
    segment_paths: list[tuple[str, ...]] = []
    segment_curve_rows: list[int] = []
    segment_edge_ids: list[int] = []
    segment_brep_indices: list[int] = []
    segment_surface_edge_ids: list[int] = []
    segment_intervals: list[tuple[Fraction, Fraction]] = []
    if edge_association is not None:
        edge_rows = _association_row_map(edge_association)
        edge_dimensions = np.asarray(edge_association.source_dimensions, dtype=np.int64)
        edge_indices = np.asarray(edge_association.source_indices, dtype=np.int64)
        edge_paths = edge_association.source_occurrence_paths
        edge_orientations = np.asarray(edge_association.orientations, dtype=np.int64)
        connectivity = mesh.connectivity
        if isinstance(connectivity, PolygonalConnectivity):
            connectivity_edges = np.asarray(connectivity.edges, dtype=np.int64)
        elif isinstance(connectivity, SimplicialConnectivity):
            connectivity_edges = np.asarray(connectivity.entities[1], dtype=np.int64)
        else:
            raise ValueError(
                "CAD boundary charts require canonical surface edge incidence."
            )
        source_edge_keys: dict[tuple[int, int], int] = {}
        edge_global_ids = np.asarray(mesh.entity_set(1).entity_ids)
        resolve = _BRepRowResolver(domain)
        for edge_row, edge in enumerate(connectivity_edges.tolist()):
            global_id = int(edge_global_ids[edge_row])
            association_row = edge_rows.get(global_id)
            if association_row is None or int(edge_dimensions[association_row]) != 1:
                continue
            source_index = int(edge_indices[association_row])
            path = tuple(edge_paths[association_row])
            curve_row, curve_key, expected_brep_id = resolve(1, source_index, path)
            if edge_association.source_entity_ids[association_row] != expected_brep_id:
                raise ValueError(
                    "CAD surface edge association changed its source curve identity."
                )
            orientation = int(edge_orientations[association_row])
            if orientation not in (-1, 1):
                raise ValueError(
                    "CAD source curve segments require a resolved orientation."
                )
            orientation *= _cad_curve_parameter_sense(domain, curve_row)
            first, second = (int(edge[0]), int(edge[1]))
            if orientation < 0:
                first, second = second, first
            plc_first, plc_second = mesh_row_to_plc[first], mesh_row_to_plc[second]
            key = (min(plc_first, plc_second), max(plc_first, plc_second))
            identity = canonical_fingerprint(
                {
                    "kind": "cad-plc-source-segment",
                    "source_curve": curve_key,
                    "occurrence_path": path,
                    "oriented_source_vertices": (
                        plc_vertex_ids[plc_first],
                        plc_vertex_ids[plc_second],
                    ),
                }
            )
            interval = edge_association.source_parameter_intervals[association_row]
            if interval is None or not 0 <= interval[0] < interval[1] <= 1:
                raise ValueError(
                    "CAD source segment lacks an exact normalized curve interval."
                )
            existing = source_edge_keys.get(key)
            if existing is not None:
                if (
                    segment_ids[existing] != identity
                    or segment_paths[existing] != path
                    or segment_intervals[existing] != interval
                ):
                    raise ValueError("One PLC edge has conflicting CAD curve ancestry.")
                continue
            source_edge_keys[key] = len(segments)
            segments.append((plc_first, plc_second))
            segment_ids.append(identity)
            segment_paths.append(path)
            segment_curve_rows.append(curve_row)
            segment_edge_ids.append(global_id)
            segment_brep_indices.append(source_index)
            segment_surface_edge_ids.append(global_id)
            segment_intervals.append(interval)
    if set(segment_curve_rows) != active_curve_rows:
        raise ValueError(
            "Accepted surface mesh must retain every active source CAD curve stratum."
        )

    vertex_ids = tuple(plc_vertex_ids)
    vertex_occurrence_paths = tuple(plc_vertex_paths)
    facet_ids = tuple(domain.entity_id(2, row) for row in range(len(domain.patches)))
    facet_paths = tuple(domain.source_occurrences[2])
    region_ids = tuple(domain.entity_id(3, row) for row in range(len(domain.regions)))
    region_paths = tuple(domain.region_source_occurrences)
    source_triangle_ids = tuple(
        _plc_source_triangle_id(
            triangle,
            facet,
            vertex_ids,
            vertex_occurrence_paths,
            facet_ids,
            facet_paths,
        )
        for triangle, facet in zip(triangles, triangle_facets, strict=True)
    )
    if len(set(source_triangle_ids)) != len(source_triangle_ids):
        raise ValueError("CAD source surface triangles have ambiguous PLC ancestry.")
    facet_planes = np.full((len(domain.patches), 3, 3), np.nan, dtype=np.float64)
    for patch in range(len(domain.patches)):
        plane = _exact_patch_plane(domain, patch)
        if plane is not None:
            facet_planes[patch] = plane
    report = surface.certification
    if report is None:
        fidelity_tolerance = 0.0 if deviation is None else deviation
        normal_deviation = None
        embedding, fidelity = _certify_cad_surface(
            surface_source, fidelity_tolerance, normal_deviation, None
        )
    else:
        certification_inputs = report.inputs
        fidelity_tolerance = certification_inputs.fidelity_tolerance
        if fidelity_tolerance is None:
            fidelity_tolerance = 0.0 if deviation is None else deviation
        normal_deviation = certification_inputs.maximum_normal_deviation
        embedding = report.embedding
        fidelity = report.fidelity
        if fidelity is None:
            _, fidelity = _certify_cad_surface(
                surface_source,
                fidelity_tolerance,
                normal_deviation,
                certification_inputs.junction_vertices,
            )
    complex_ = PiecewiseLinearComplex(
        np.asarray(
            [[float(value) for value in point] for point in plc_points], dtype=np.float64
        ),
        triangles,
        np.asarray(triangle_facets, dtype=np.int64),
        facet_regions,
        region_ids,
        segments=np.asarray(segments, dtype=np.int64).reshape((-1, 2)),
        vertex_ids=vertex_ids,
        vertex_occurrence_paths=vertex_occurrence_paths,
        facet_ids=facet_ids,
        facet_occurrence_paths=facet_paths,
        segment_ids=tuple(segment_ids),
        segment_occurrence_paths=tuple(segment_paths),
        region_occurrence_paths=region_paths,
        cad_source_charts=surface_source,
        cad_fidelity_tolerance=fidelity_tolerance,
        cad_maximum_normal_deviation=normal_deviation,
        cad_embedding=embedding,
        cad_fidelity=fidelity,
        cad_vertex_rows=tuple(mesh_row_to_plc),
        cad_segment_edge_ids=tuple(segment_edge_ids),
        cad_polygon_cell_ids=tuple(triangle_cell_ids),
        boundary=boundary,
        facet_planes=facet_planes,
        facet_plane_tolerances=np.full(
            (len(domain.patches),), 0.0 if deviation is None else deviation
        ),
        cad_fidelity_tolerance=fidelity_tolerance,
        cad_maximum_normal_deviation=normal_deviation,
        cad_embedding=embedding,
        cad_fidelity=fidelity,
        cad_source_charts=surface_source,
        cad_vertex_rows=np.asarray(mesh_row_to_plc, dtype=np.int64),
        cad_segment_edge_ids=np.asarray(segment_surface_edge_ids, dtype=np.int64),
        cad_polygon_cell_ids=np.asarray(triangle_cell_ids, dtype=np.int64),
    )
    return _CadPlcData(
        complex_,
        _readonly(np.asarray(plc_vertex_dimensions), np.dtype(np.int8)),
        _readonly(np.asarray(plc_vertex_indices), np.dtype(np.int32)),
        _readonly(np.asarray(plc_vertex_parameters), np.dtype(np.float64)),
        _readonly(np.asarray(segment_curve_rows), np.dtype(np.int32)),
        _readonly(np.asarray(segment_brep_indices), np.dtype(np.int32)),
        source_triangle_ids,
        _readonly(np.asarray(triangle_cell_ids), np.dtype(np.int64)),
        _readonly(np.asarray(triangle_facets), np.dtype(np.int64)),
        _readonly(np.asarray(triangle_parameters), np.dtype(np.float64)),
        tuple(triangle_parameter_exact),
    )


def _surface_source_authorities(
    surface: CellMeshingResult, domain: MeshingDomain, /
) -> tuple[SurfaceSourceCharts, SourceBoundaryQuery]:
    surface_source = getattr(surface, "surface_source", None)
    if not isinstance(surface_source, SurfaceSourceCharts):
        raise ValueError(
            "CAD volume source acceptance requires retained SurfaceSourceCharts ancestry."
        )
    surface_source.require_root(surface.mesh, surface.geometry)
    chart_query = surface_source.boundary_source
    if surface_source is not None:
        chart_domain = getattr(surface_source, "domain", None)
        if (
            chart_domain is None
            or chart_domain.domain_id != domain.domain_id
            or getattr(surface_source, "source_id", None) != domain.source_id
            or getattr(surface_source, "source_revision", None) != domain.source_revision
        ):
            raise ValueError(
                "Accepted surface source charts must bind the exact CAD source domain."
            )
        chart_id = getattr(surface_source, "source_chart_id", None)
        if not isinstance(chart_id, str) or not chart_id:
            raise ValueError(
                "Accepted CAD surface charts require a stable chart identity."
            )
        cell_ids = np.asarray(
            getattr(surface_source, "cell_global_ids", ()), dtype=np.int64
        )
        expected_cell_ids = np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in surface.mesh.blocks
            ]
        )
        root_parameters = np.asarray(
            getattr(surface_source, "root_parameters", ()), dtype=np.float64
        )
        root_patches = np.asarray(
            getattr(surface_source, "root_patches", ()), dtype=np.int64
        )
        if (
            not np.array_equal(cell_ids, expected_cell_ids)
            or root_parameters.shape != (cell_ids.size, 3, 2)
            or not np.all(np.isfinite(root_parameters))
            or root_patches.shape != cell_ids.shape
            or not np.all((root_patches >= 0) & (root_patches < len(domain.patches)))
            or set(root_patches.tolist()) != set(range(len(domain.patches)))
        ):
            raise ValueError(
                "Accepted surface source chart cell IDs and patch parameters must cover its mesh."
            )
        face_association = _association_for(surface, 2, required=True)
        if face_association is None:
            raise ValueError("CAD surface requires a source-face association.")
        association_rows = _association_row_map(face_association)
        dimensions = np.asarray(face_association.source_dimensions, dtype=np.int64)
        indices = np.asarray(face_association.source_indices, dtype=np.int64)
        paths = face_association.source_occurrence_paths
        resolve = _BRepRowResolver(domain)
        for cell_id, patch in zip(cell_ids.tolist(), root_patches.tolist(), strict=True):
            association_row = association_rows[int(cell_id)]
            if int(dimensions[association_row]) != 2:
                raise ValueError(
                    "Surface chart cells must retain their B-Rep face association."
                )
            resolved_patch, _, _ = resolve(
                2,
                int(indices[association_row]),
                tuple(paths[association_row]),
            )
            if resolved_patch != int(patch):
                raise ValueError(
                    "Surface chart patch rows disagree with B-Rep association ancestry."
                )
        chart_query = getattr(surface_source, "boundary_source", None)
        if chart_query is not None and not isinstance(chart_query, SourceBoundaryQuery):
            raise TypeError(
                "Surface source boundary authority must implement SourceBoundaryQuery."
            )
    certification = getattr(surface, "certification", None)
    inputs = None if certification is None else getattr(certification, "inputs", None)
    query = None if inputs is None else getattr(inputs, "source", None)
    if query is not None and not isinstance(query, SourceBoundaryQuery):
        raise TypeError(
            "Accepted surface certification source must implement SourceBoundaryQuery."
        )
    if (
        query is not None
        and chart_query is not None
        and (
            query.source_id != chart_query.source_id
            or query.source_revision != chart_query.source_revision
        )
    ):
        raise ValueError(
            "Surface charts and certification query must share one CAD revision."
        )
    query = query if query is not None else chart_query
    for candidate in (chart_query, query):
        if (
            isinstance(candidate, MeshingDomainBoundarySource)
            and candidate.domain.domain_id != domain.domain_id
        ):
            raise ValueError(
                "CAD boundary queries must use the exact accepted MeshingDomain."
            )
    if query is not None and (
        query.source_id != domain.source_id
        or query.source_revision != domain.source_revision
        or query.ambient_dimension != 3
    ):
        raise ValueError(
            "Accepted surface CAD fidelity query changed its source revision."
        )
    if not isinstance(query, MeshingDomainBoundarySource):
        raise ValueError(
            "CAD volume source charts require their original MeshingDomainBoundarySource."
        )
    surface_source.require_bound(surface.mesh, surface.geometry, query)
    if isinstance(query, MeshingDomainBoundarySource) and set(query.patches) != set(
        range(len(domain.patches))
    ):
        raise ValueError("CAD volume fidelity queries must cover every source patch.")
    return surface_source, query


@final
class NativeCadVolumeSource(StrictModule, NonTrainableState):
    """A source-bound CAD shell plus its accepted surface mesh and PLC ancestry."""

    domain: MeshingDomain
    surface: CellMeshingResult
    surface_source: SurfaceSourceCharts
    fidelity_source: SourceBoundaryQuery
    plc_source: NativePlcSource
    vertex_brep_dimensions: np.ndarray
    vertex_brep_indices: np.ndarray
    vertex_parameters: np.ndarray
    segment_curve_rows: np.ndarray
    segment_brep_indices: np.ndarray
    source_triangle_ids: tuple[str, ...]
    source_triangle_cell_ids: np.ndarray
    source_triangle_patches: np.ndarray
    source_triangle_parameters: np.ndarray
    source_triangle_parameter_exact: tuple[tuple[tuple[Fraction, Fraction], ...], ...]
    maximum_deviation: float | None = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: MeshingDomain | BRepModel,
        surface: CellMeshingResult,
        /,
        *,
        fidelity_source: SourceBoundaryQuery | None = None,
        maximum_deviation: float | None = None,
        boundary: PlcBoundaryPolicy = "fixed",
    ) -> None:
        if isinstance(domain, BRepModel):
            domain = MeshingDomain.from_brep(domain)
        if not isinstance(domain, MeshingDomain):
            raise TypeError("domain must be BRepModel or authoritative MeshingDomain.")
        boundary = parse(boundary, PlcBoundaryPolicy, "boundary")
        surface_source, accepted_query = _surface_source_authorities(surface, domain)
        if (
            fidelity_source is not None
            and accepted_query is not None
            and (
                fidelity_source.source_id != accepted_query.source_id
                or fidelity_source.source_revision != accepted_query.source_revision
            )
        ):
            raise ValueError(
                "Explicit and accepted CAD fidelity queries must bind the same source revision."
            )
        query = (
            accepted_query
            if accepted_query is not None
            else fidelity_source
            if fidelity_source is not None
            else MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))
        )
        if (
            isinstance(query, MeshingDomainBoundarySource)
            and query.domain.domain_id != domain.domain_id
        ):
            raise ValueError(
                "CAD volume source queries must bind the exact MeshingDomain."
            )
        if isinstance(query, MeshingDomainBoundarySource) and set(query.patches) != set(
            range(len(domain.patches))
        ):
            raise ValueError("CAD volume fidelity queries must cover every source patch.")
        if not isinstance(query, SourceBoundaryQuery):
            raise TypeError("fidelity_source must implement SourceBoundaryQuery.")
        if not isinstance(query, MeshingDomainBoundarySource):
            raise TypeError(
                "CAD volume source fidelity requires MeshingDomainBoundarySource."
            )
        if (
            query.source_id != domain.source_id
            or query.source_revision != domain.source_revision
            or query.ambient_dimension != 3
        ):
            raise ValueError(
                "CAD source fidelity queries must bind the same 3D BRep revision."
            )
        deviation = None
        if maximum_deviation is not None:
            deviation = finite_real_scalar(float(maximum_deviation), "maximum_deviation")
            if deviation < 0.0:
                raise ValueError("maximum_deviation must be non-negative.")
        data = _cad_plc_data(domain, surface, surface_source, boundary, deviation)
        plc_source = NativePlcSource(
            data.complex, domain.source_id, domain.source_revision
        )
        self.domain = domain
        self.surface_source = surface_source
        self.surface = surface
        self.fidelity_source = query
        self.plc_source = plc_source
        self.vertex_brep_dimensions = data.vertex_brep_dimensions
        self.vertex_brep_indices = data.vertex_brep_indices
        self.vertex_parameters = data.vertex_parameters
        self.segment_curve_rows = data.segment_curve_rows
        self.source_triangle_ids = data.source_triangle_ids
        self.source_triangle_cell_ids = data.source_triangle_cell_ids
        self.source_triangle_patches = data.source_triangle_patches
        self.source_triangle_parameters = data.source_triangle_parameters
        self.source_triangle_parameter_exact = data.source_triangle_parameter_exact
        self.segment_brep_indices = data.segment_brep_indices
        self.maximum_deviation = deviation
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-cad-volume-source",
                "domain": domain.domain_id,
                "surface": surface.result_id,
                "surface_source_chart": surface_source.source_chart_id,
                "plc": plc_source.binding_id,
                "fidelity_source": (query.source_id, query.source_revision),
                "maximum_deviation": deviation,
                "boundary": boundary,
            }
        )


def _domain_scope_rows(
    domain: MeshingDomain, scope: MeshingScope, dimension: int, /
) -> np.ndarray:
    if (
        scope.source_id != domain.source_id
        or scope.source_revision != domain.source_revision
        or scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != dimension
        or scope.entity_set_id != domain.scope_entity_set_id(dimension)
    ):
        raise ValueError("CAD volume controls must bind their exact source entity set.")
    if scope.side_records:
        if domain.brep_authority is None:
            raise ValueError("CAD side scopes require their authoritative B-Rep model.")
        MeshingScope.from_stored_fields(scope, source_model=domain.brep_authority)
    rows = np.asarray(
        [
            domain.resolve_entity_id(dimension, key, occurrence_path=path)
            for identifier in np.asarray(scope.entity_ids, dtype=np.int64)
            for key, path in (
                domain.source_entity_key_for_scope(dimension, int(identifier)),
            )
        ],
        dtype=np.int64,
    )
    if np.unique(rows).size != rows.size:
        raise ValueError("CAD volume scope repeats one source occurrence.")
    return rows


def _plc_scope(
    source: NativeCadVolumeSource,
    dimension: int,
    identifiers: np.ndarray,
    /,
) -> MeshingScope:
    return MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        dimension,
        source.plc_source.scope_entity_set_id(dimension),
        np.asarray(identifiers, dtype=np.int64),
    )


def _region_row(domain: MeshingDomain, name: str, /) -> int:
    matches = tuple(
        row for row, region in enumerate(domain.regions) if region.name == name
    )
    if len(matches) != 1:
        raise ValueError("CAD region names must resolve one exact source occurrence.")
    return matches[0]


def _plc_feature_ids(
    source: NativeCadVolumeSource,
    scope: MeshingScope,
    feature_kind: FeatureKind,
    /,
) -> tuple[int, np.ndarray]:
    dimension = scope.entity_dimension
    domain = source.domain
    rows = _domain_scope_rows(domain, scope, dimension)
    complex_ = source.plc_source.complex
    if dimension == 0 and feature_kind is FeatureKind.CORNER:
        source_indices = np.asarray(domain.source_indices[0], dtype=np.int64)
        paths = domain.source_occurrences[0]
        selected = {(int(source_indices[row]), paths[int(row)]) for row in rows.tolist()}
        output = [
            complex_.vertex_entity_ids[row]
            for row, (source_dimension, source_index, path) in enumerate(
                zip(
                    source.vertex_brep_dimensions.tolist(),
                    source.vertex_brep_indices.tolist(),
                    complex_.vertex_occurrence_paths,
                    strict=True,
                )
            )
            if source_dimension == 0 and (int(source_index), path) in selected
        ]
        return 0, np.asarray(output, dtype=np.int64)
    if dimension == 1 and feature_kind is FeatureKind.CURVE:
        source_indices = np.asarray(domain.source_indices[1], dtype=np.int64)
        paths = domain.source_occurrences[1]
        selected = {(int(source_indices[row]), paths[int(row)]) for row in rows.tolist()}
        plc_edge_row_by_identity = {
            (identifier, tuple(path)): row
            for row, (identifier, path) in enumerate(
                zip(
                    complex_.plc_edge_ids,
                    complex_.plc_edge_occurrence_paths,
                    strict=True,
                )
            )
        }
        output: list[int] = []
        for segment_row, (curve_row, identifier, occurrence_path) in enumerate(
            zip(
                source.segment_curve_rows.tolist(),
                complex_.segment_ids,
                complex_.segment_occurrence_paths,
                strict=True,
            )
        ):
            if (
                int(source.segment_brep_indices[segment_row]),
                domain.source_occurrences[1][int(curve_row)],
            ) not in selected:
                continue
            plc_edge_row = plc_edge_row_by_identity.get(
                (identifier, tuple(occurrence_path))
            )
            if plc_edge_row is None:
                raise ValueError(
                    "An authored CAD curve is absent from its canonical PLC edge bank."
                )
            output.append(int(complex_.plc_edge_entity_ids[plc_edge_row]))
        return 1, np.asarray(output, dtype=np.int64)
    if dimension == 2 and feature_kind in (
        FeatureKind.SURFACE,
        FeatureKind.MATERIAL_INTERFACE,
    ):
        if feature_kind is FeatureKind.MATERIAL_INTERFACE and any(
            not np.all(complex_.facet_regions[int(row)] >= 0) for row in rows.tolist()
        ):
            raise ValueError(
                "Material-interface controls must select only CAD interfaces."
            )
        return 2, np.asarray(complex_.facet_entity_ids, dtype=np.int64)[rows]
    raise ValueError("CAD protected feature dimension and kind are incompatible.")


def _translate_specification(
    source: NativeCadVolumeSource, specification: VolumeMeshingSpec, /
) -> VolumeMeshingSpec:
    domain = source.domain
    complex_ = source.plc_source.complex
    boundary_scope = _plc_scope(
        source,
        2,
        np.asarray(complex_.facet_entity_ids, dtype=np.int64),
    )
    size_controls: list[UniformSizeControl] = []
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            raise ValueError(
                "CAD PLC filling supports uniform source-patch size controls only."
            )
        rows = _domain_scope_rows(domain, control.scope, 2)
        scope = _plc_scope(
            source, 2, np.asarray(complex_.facet_entity_ids, dtype=np.int64)[rows]
        )
        size_controls.append(
            UniformSizeControl(
                scope,
                control.target_size,
                minimum_size=control.minimum_size,
                maximum_size=control.maximum_size,
                maximum_growth_rate=control.maximum_growth_rate,
                strength=control.strength,
                priority=control.priority,
            )
        )
    features: list[ProtectedFeature] = []
    for feature in specification.protected_features:
        dimension, identifiers = _plc_feature_ids(
            source, feature.scope, feature.feature_kind
        )
        if identifiers.size == 0:
            raise ValueError(
                "A CAD protected feature must select represented PLC entities."
            )
        features.append(
            ProtectedFeature(
                _plc_scope(source, dimension, identifiers),
                feature.feature_kind,
                maximum_deviation=feature.maximum_deviation,
                hard=feature.hard,
            )
        )
    region_controls: list[RegionControl] = []
    region_names = [region.name for region in domain.regions]
    for control in specification.region_controls:
        selected_rows = _domain_scope_rows(domain, control.scope, 3)
        if selected_rows.size != 1:
            raise ValueError(
                "CAD region control scope must identify exactly one source region."
            )
        row = int(selected_rows[0])
        region_names[row] = control.region_name
        region_controls.append(
            RegionControl(
                _plc_scope(
                    source,
                    3,
                    np.asarray((complex_.region_entity_ids[row],), dtype=np.int64),
                ),
                complex_.region_ids[row],
                control.material_id,
                control.role,
                meshing_enabled=control.meshing_enabled,
            )
        )
    patch_controls: list[PatchControl] = []
    for control in specification.patch_controls:
        rows = _domain_scope_rows(domain, control.scope, 2)
        if control.scope.side_records:
            unqualified = set(control.scope.unqualified_entity_ids)
            side_counts: dict[int, int] = {}
            for side in control.scope.side_records:
                side_counts[side.entity_id] = side_counts.get(side.entity_id, 0) + 1
            for identifier, row in zip(
                np.asarray(control.scope.entity_ids), rows, strict=True
            ):
                if int(identifier) in unqualified:
                    continue
                adjacent_count = sum(
                    int(region) >= 0 for region in complex_.facet_regions[int(row)]
                )
                if side_counts.get(int(identifier), 0) != adjacent_count:
                    raise ValueError(
                        "A single interface side cannot be flattened into a patch control; "
                        "mesh the shared facet and resolve its sided integration route."
                    )
        expected_names = set(control.adjacent_region_names)
        adjacent_rows = sorted(
            {
                int(region)
                for facet in rows.tolist()
                for region in complex_.facet_regions[int(facet)]
                if int(region) >= 0
            }
        )
        for facet in rows.tolist():
            facet_names = {
                region_names[int(region)]
                for region in complex_.facet_regions[int(facet)]
                if int(region) >= 0
            }
            if facet_names != expected_names:
                raise ValueError(
                    "CAD patch control adjacency differs from its selected source facets."
                )
        adjacent = tuple(complex_.region_ids[row] for row in adjacent_rows)
        patch_controls.append(
            PatchControl(
                control.name,
                _plc_scope(
                    source,
                    2,
                    np.asarray(complex_.facet_entity_ids, dtype=np.int64)[rows],
                ),
                adjacent,
                required=control.required,
            )
        )
    region_seeds: list[RegionSeed] = []
    for seed in specification.region_seeds:
        row = _region_row(domain, seed.region_name)
        region_seeds.append(
            RegionSeed(
                np.asarray(seed.point, dtype=np.float64),
                complex_.region_ids[row],
                seed.material_id,
                seed.role,
            )
        )
    hole_seeds: list[HoleSeed] = []
    for seed in specification.hole_seeds:
        rows = _domain_scope_rows(domain, seed.scope, 3)
        hole_seeds.append(
            HoleSeed(
                np.asarray(seed.point, dtype=np.float64),
                _plc_scope(
                    source,
                    3,
                    np.asarray(complex_.region_entity_ids, dtype=np.int64)[rows],
                ),
            )
        )
    return VolumeMeshingSpec(
        specification.target,
        boundary_scope,
        specification.fill_strategy,
        size_controls=tuple(size_controls),
        protected_features=tuple(features),
        region_controls=tuple(region_controls),
        patch_controls=tuple(patch_controls),
        region_seeds=tuple(region_seeds),
        hole_seeds=tuple(hole_seeds),
        periodic_constraints=specification.periodic_constraints,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        maximum_deviation=specification.maximum_deviation,
        deterministic=specification.deterministic,
    )


def _cad_source_tolerance(
    source: NativeCadVolumeSource, specification: VolumeMeshingSpec, /
) -> float:
    tolerance = (
        specification.maximum_deviation
        if specification.maximum_deviation is not None
        else source.maximum_deviation
    )
    if tolerance is None:
        raise ValueError(
            "CAD tetrahedral filling requires an explicit source deviation tolerance."
        )
    if (
        specification.maximum_deviation is not None
        and source.maximum_deviation is not None
        and specification.maximum_deviation > source.maximum_deviation
    ):
        raise ValueError(
            "Requested CAD deviation tolerance exceeds its source-declared bound."
        )
    return float(tolerance)


def cad_volume_support_issues(
    source: NativeCadVolumeSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    if not isinstance(source, NativeCadVolumeSource):
        raise TypeError("source must be NativeCadVolumeSource.")
    issues = list(_target_support(specification))
    domain = source.domain
    boundary = specification.boundary_scope
    expected_scope = np.asarray(domain.scope_entity_ids(2), dtype=np.int64)
    if (
        boundary.source_id != domain.source_id
        or boundary.source_revision != domain.source_revision
        or boundary.entity_kind is not MeshingEntityKind.GEOMETRY
        or boundary.entity_dimension != 2
        or boundary.entity_set_id != domain.scope_entity_set_id(2)
        or not np.array_equal(
            np.sort(np.asarray(boundary.entity_ids, dtype=np.int64)),
            np.sort(expected_scope),
        )
    ):
        issues.append("a whole-source boundary scope for every CAD face occurrence")
    if specification.periodic_constraints:
        issues.append("periodic CAD volume constraints")
    if specification.layer_controls:
        issues.append("CAD boundary-layer controls")
    if (
        source.fidelity_source.source_id != domain.source_id
        or source.fidelity_source.source_revision != domain.source_revision
    ):
        issues.append("a source-bound original BRep boundary query")
    if source.fidelity_source.ambient_dimension != 3:
        issues.append("a three-dimensional original BRep boundary query")
    try:
        _cad_source_tolerance(source, specification)
    except ValueError:
        issues.append("an explicit CAD maximum_deviation no weaker than source authority")
    try:
        translated = _translate_specification(source, specification)
    except (TypeError, ValueError, IndexError) as error:
        issues.append(
            f"CAD source controls resolvable through exact domain strata: {error}"
        )
    else:
        issues.extend(volume_support_issues(source.plc_source, translated))
    return issues


_CadEntityKey = tuple[str, tuple[str, ...]]
_CadEntityAncestor = tuple[int, int, tuple[str, ...]]
_CadEntityMaps = tuple[
    dict[_CadEntityKey, _CadEntityAncestor],
    dict[_CadEntityKey, _CadEntityAncestor],
    dict[_CadEntityKey, _CadEntityAncestor],
    dict[_CadEntityKey, _CadEntityAncestor],
]


def _cad_entity_maps(source: NativeCadVolumeSource, /) -> _CadEntityMaps:
    complex_ = source.plc_source.complex
    domain = source.domain
    maps: list[dict[_CadEntityKey, _CadEntityAncestor]] = [{}, {}, {}, {}]
    for row, key in enumerate(
        zip(complex_.vertex_ids, complex_.vertex_occurrence_paths, strict=True)
    ):
        maps[0][key] = (
            int(source.vertex_brep_dimensions[row]),
            int(source.vertex_brep_indices[row]),
            key[1],
        )
    for row, key in enumerate(
        zip(complex_.facet_ids, complex_.facet_occurrence_paths, strict=True)
    ):
        maps[2][key] = (2, int(domain.source_indices[2][row]), key[1])
    for row, key in enumerate(
        zip(complex_.region_ids, complex_.region_occurrence_paths, strict=True)
    ):
        maps[3][key] = (3, int(domain.region_source_indices[row]), key[1])
    source_segment_row_by_key = {
        (identifier, tuple(path)): row
        for row, (identifier, path) in enumerate(
            zip(complex_.segment_ids, complex_.segment_occurrence_paths, strict=True)
        )
    }
    segment_curve_rows = np.asarray(source.segment_curve_rows, dtype=np.int64)
    segment_brep_indices = np.asarray(source.segment_brep_indices, dtype=np.int64)
    if (
        segment_curve_rows.shape != (len(complex_.segment_ids),)
        or segment_brep_indices.shape != segment_curve_rows.shape
    ):
        raise ValueError(
            "CAD source curve rows and B-Rep edge identities are misaligned."
        )
    for row, key in enumerate(
        zip(complex_.plc_edge_ids, complex_.plc_edge_occurrence_paths, strict=True)
    ):
        source_segment_row = source_segment_row_by_key.get(key)
        if source_segment_row is not None:
            curve_row = int(segment_curve_rows[source_segment_row])
            source_index = int(domain.source_indices[1][curve_row])
            source_path = domain.source_occurrences[1][curve_row]
            if (
                source_index != int(segment_brep_indices[source_segment_row])
                or source_path != key[1]
            ):
                raise ValueError(
                    "CAD segment source lineage changed its curve occurrence."
                )
            maps[1][key] = (1, source_index, source_path)
            continue
        # A synthesized one-facet PLC edge maps to its BRep face, not a CAD curve.
        facets = complex_.plc_edge_facets[row]
        if len(facets) != 1:
            raise ValueError(
                "A CAD patch seam requires its authored source curve identity."
            )
        facet = int(facets[0])
        maps[1][key] = (
            2,
            int(domain.source_indices[2][facet]),
            domain.source_occurrences[2][facet],
        )
    expected_counts = (
        len(complex_.vertex_ids),
        len(complex_.plc_edge_ids),
        len(complex_.facet_ids),
        len(complex_.region_ids),
    )
    if tuple(len(bank) for bank in maps) != expected_counts:
        raise ValueError(
            "CAD PLC ancestry maps must uniquely cover every authored source entity."
        )
    return maps[0], maps[1], maps[2], maps[3]


def _source_entity_for_plc(
    source_maps: _CadEntityMaps,
    dimension: int,
    identifier: str,
    path: tuple[str, ...],
    /,
) -> _CadEntityAncestor:
    try:
        return source_maps[dimension][(identifier, path)]
    except (IndexError, KeyError) as error:
        raise ValueError(
            "CAD PLC ancestry must resolve one stable source ID and occurrence."
        ) from error


def _cad_uv_triangle_orientation(
    parameters: tuple[tuple[Fraction, Fraction], ...], /
) -> int:
    if len(parameters) != 3 or any(len(point) != 2 for point in parameters):
        raise ValueError("CAD source chart orientation requires one exact UV triangle.")
    points = parameters
    area = (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) - (
        points[1][1] - points[0][1]
    ) * (points[2][0] - points[0][0])
    if area == 0:
        raise ValueError("CAD source chart ancestry contains a degenerate UV triangle.")
    return 1 if area > 0 else -1


type _ExactPlaneChart = tuple[
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
]


def _exact_patch_chart(domain: MeshingDomain, patch: int, /) -> _ExactPlaneChart | None:
    """Exact placed ``(origin, first_axis, second_axis)`` of a planar patch chart.

    Only a (placed) ``PlanePatch`` has one; the exact affine placement maps
    its binary64 origin and axes without rounding.
    """
    surface = domain.patches[patch].surface
    placement = None
    if isinstance(surface, _PlacedSurfacePatch):
        surface, placement = surface.source, surface.placement
    if not isinstance(surface, PlanePatch):
        return None
    rows = [
        tuple(Fraction.from_float(float(value)) for value in np.asarray(vector))
        for vector in (surface.origin, surface.first_axis, surface.second_axis)
    ]
    if placement is not None:
        matrix = placement.matrix
        rows = [
            tuple(
                sum(
                    (matrix[axis][column] * row[column] for column in range(3)),
                    Fraction(0),
                )
                + (matrix[axis][3] if index == 0 else Fraction(0))
                for axis in range(3)
            )
            for index, row in enumerate(rows)
        ]
    origin, first, second = rows
    return (
        (origin[0], origin[1], origin[2]),
        (first[0], first[1], first[2]),
        (second[0], second[1], second[2]),
    )


def _exact_patch_plane(domain: MeshingDomain, patch: int, /) -> np.ndarray | None:
    """Binary64 rows of a patch's exact plane, each component rounded once.

    The plane is the affine chart of a ``PlanePatch`` or, for a patch proved
    planar by its carrier, its `exact_plane_chart`.
    """
    chart = _exact_patch_chart(domain, patch)
    if chart is None:
        plane = exact_plane_chart(domain, patch)
        if plane is None:
            return None
        chart = (plane.origin, plane.first_axis, plane.second_axis)
    return np.asarray(
        [[float(value) for value in row] for row in chart], dtype=np.float64
    )


def _cad_plane_witness_uv(
    chart: _ExactPlaneChart, plane: np.ndarray, parameters: np.ndarray, /
) -> tuple[Fraction, Fraction]:
    """Exact chart parameters of a declared-plane (stratum-3) witness point.

    The witness is ``origin + s * first_axis + t * second_axis`` of the
    declared binary64 plane rows; its parameters are those of its orthogonal
    projection onto the exact patch chart (the declared rows round the chart,
    so they are ``(s, t)`` whenever the rounding is exact).
    """
    rows = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in plane
    )
    s, t = (Fraction.from_float(float(value)) for value in parameters)
    origin, first, second = chart
    delta = tuple(
        rows[0][axis] + s * rows[1][axis] + t * rows[2][axis] - origin[axis]
        for axis in range(3)
    )

    def dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    first_first, first_second, second_second = (
        dot(first, first),
        dot(first, second),
        dot(second, second),
    )
    determinant = first_first * second_second - first_second * first_second
    if determinant == 0:
        raise ValueError("CAD planar patch chart axes are parallel.")
    first_delta, second_delta = dot(first, delta), dot(second, delta)
    return (
        (first_delta * second_second - second_delta * first_second) / determinant,
        (first_first * second_delta - first_second * first_delta) / determinant,
    )


def _cad_curve_association(
    source: NativeCadVolumeSource, construction: VolumeConstruction, /
) -> GeometryAssociation | None:
    """Restrict original exact curve intervals along proven PLC segment ancestry."""
    source.domain.require_current(source.source_id, source.source_revision)
    source.surface_source.require_root(source.surface.mesh, source.surface.geometry)
    source.surface_source.require_bound(
        source.surface.mesh, source.surface.geometry, source.fidelity_source
    )
    complex_ = PiecewiseLinearComplex.from_stored_fields(source.plc_source.complex)
    if (
        complex_.cad_source_charts is None
        or complex_.cad_source_charts.source_chart_id
        != source.surface_source.source_chart_id
        or construction.complex_id != complex_.complex_id
        or construction.source_id != source.source_id
        or construction.source_revision != source.source_revision
    ):
        raise ValueError(
            "CAD volume construction changed its source PLC or chart binding."
        )
    original = _association_for(complex_.cad_source_charts, 1, required=False)
    if complex_.cad_segment_edge_ids is None:
        raise ValueError("CAD PLC lost its original surface edge mapping.")
    segment_ids = np.asarray(complex_.cad_segment_edge_ids, dtype=np.int64)
    authored = np.asarray(complex_.segments, dtype=np.int64).reshape((-1, 2))
    if segment_ids.shape != (authored.shape[0],):
        raise ValueError("CAD source curve segment and surface edge rows are misaligned.")
    if not segment_ids.size:
        return None
    if original is None:
        raise ValueError("CAD source curve intervals are no longer available.")
    original_rows = _association_row_map(original)
    edge_key_to_authored = {
        (identifier, tuple(path)): row
        for row, (identifier, path) in enumerate(
            zip(complex_.segment_ids, complex_.segment_occurrence_paths, strict=True)
        )
    }
    if len(edge_key_to_authored) != segment_ids.size:
        raise ValueError("CAD source curve segment identities are ambiguous.")
    source_edges = np.asarray(construction.source_segment_vertices, dtype=np.int64)
    complex_rows = np.asarray(construction.source_segment_complex_rows, dtype=np.int64)
    if (
        source_edges.shape != (complex_rows.size, 2)
        or len(construction.segment_ids) != complex_rows.size
        or len(construction.segment_occurrence_paths) != complex_rows.size
    ):
        raise ValueError("CAD source segment ancestry banks are misaligned.")
    source_to_authored: dict[int, int] = {}
    canonical_edges = np.asarray(complex_.plc_edge_vertices, dtype=np.int64)
    for row, (identifier, path) in enumerate(
        zip(construction.segment_ids, construction.segment_occurrence_paths, strict=True)
    ):
        authored_row = edge_key_to_authored.get((identifier, tuple(path)))
        if authored_row is None:
            continue
        complex_row = int(complex_rows[row])
        if (
            not 0 <= complex_row < canonical_edges.shape[0]
            or not np.array_equal(source_edges[row], authored[authored_row])
            or not np.array_equal(canonical_edges[complex_row], authored[authored_row])
            or complex_.plc_edge_ids[complex_row] != identifier
            or tuple(complex_.plc_edge_occurrence_paths[complex_row]) != tuple(path)
        ):
            raise ValueError("CAD source curve segment changed its canonical endpoints.")
        source_to_authored[row] = authored_row
    if len(source_to_authored) != segment_ids.size or set(
        source_to_authored.values()
    ) != set(range(segment_ids.size)):
        raise ValueError("CAD source curve segments lost their canonical PLC ancestry.")
    mesh = construction.mesh
    edges, _ = simplex_volume_entities(mesh)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_rows = {tuple(sorted(map(int, edge))): row for row, edge in enumerate(edges)}
    if len(edge_rows) != edge_ids.size:
        raise ValueError("CAD volume mesh edge incidence is ambiguous.")
    input_vertices = np.asarray(construction.input_vertices, dtype=np.int64)
    strata = np.asarray(construction.witness_strata, dtype=np.int64)
    entities = np.asarray(construction.witness_entities, dtype=np.int64)
    parameters = np.asarray(construction.witness_parameters, dtype=np.float64)
    deviations = np.asarray(construction.witness_deviations, dtype=np.float64)
    if (
        input_vertices.shape != (mesh.coordinates.shape[0],)
        or strata.shape != input_vertices.shape
        or entities.shape != input_vertices.shape
        or parameters.shape != (input_vertices.size, 2)
        or deviations.shape != input_vertices.shape
    ):
        raise ValueError("CAD curve endpoint witness banks are misaligned.")
    segments = np.asarray(construction.segments, dtype=np.int64).reshape((-1, 2))
    segment_sources = np.asarray(construction.segment_source_edges, dtype=np.int64)
    if (
        segment_sources.shape != (segments.shape[0],)
        or len(construction.segment_source_ids) != segments.shape[0]
        or len(construction.segment_source_occurrence_paths) != segments.shape[0]
    ):
        raise ValueError("CAD target segment ancestry banks are misaligned.")
    targets: list[int] = []
    source_ids: list[str] = []
    parent_ids: list[int] = []
    source_indices: list[int] = []
    paths: list[tuple[str, ...]] = []
    intervals: list[tuple[Fraction, Fraction]] = []
    orientations: list[int] = []
    residuals: list[float] = []
    parts: dict[int, list[tuple[Fraction, Fraction]]] = {}
    seen: set[int] = set()
    old_indices = np.asarray(original.source_indices, dtype=np.int64)
    old_residuals = np.asarray(original.residuals, dtype=np.float64)
    resolve = _BRepRowResolver(source.domain)
    senses: list[int] = []
    for surface_edge_id in segment_ids.tolist():
        original_row = original_rows.get(int(surface_edge_id))
        if original_row is None:
            raise ValueError("CAD source segment lost its original curve association.")
        path = tuple(original.source_occurrence_paths[original_row])
        curve, _, expected_id = resolve(1, int(old_indices[original_row]), path)
        if original.source_entity_ids[original_row] != expected_id:
            raise ValueError("CAD source segment changed its B-Rep curve identity.")
        senses.append(_cad_curve_parameter_sense(source.domain, curve))
    for segment_row, (segment, raw_parent) in enumerate(
        zip(segments, segment_sources, strict=True)
    ):
        parent = int(raw_parent)
        if not 0 <= parent < source_edges.shape[0]:
            raise ValueError("CAD target segment names an absent source edge.")
        authored_row = source_to_authored.get(parent)
        if authored_row is None:
            continue
        if construction.segment_source_ids[segment_row] != construction.segment_ids[
            parent
        ] or tuple(construction.segment_source_occurrence_paths[segment_row]) != tuple(
            construction.segment_occurrence_paths[parent]
        ):
            raise ValueError("CAD target segment changed its source identity.")
        edge_row = edge_rows.get(tuple(sorted(map(int, segment))))
        if edge_row is None or edge_row in seen:
            raise ValueError("CAD target segment does not identify one unique mesh edge.")
        seen.add(edge_row)
        original_row = original_rows.get(int(segment_ids[authored_row]))
        if original_row is None:
            raise ValueError("CAD source segment lost its original curve association.")
        interval = original.source_parameter_intervals[original_row]
        if interval is None:
            raise ValueError("CAD source segment lacks an exact curve interval.")
        fractions: list[Fraction] = []
        for raw_vertex in edges[edge_row]:
            vertex = int(raw_vertex)
            if not 0 <= vertex < input_vertices.size:
                raise ValueError("CAD target edge names an absent vertex.")
            input_vertex = int(input_vertices[vertex])
            if input_vertex == int(source_edges[parent, 0]):
                fraction = Fraction(0)
            elif input_vertex == int(source_edges[parent, 1]):
                fraction = Fraction(1)
            elif input_vertex < 0 and strata[vertex] == 1 and entities[vertex] == parent:
                value = float(parameters[vertex, 0])
                if not np.isfinite(value):
                    raise ValueError("CAD edge witness has no finite source parameter.")
                fraction = Fraction.from_float(value)
            else:
                raise ValueError("CAD curve endpoint lacks its source segment witness.")
            if not 0 <= fraction <= 1:
                raise ValueError("CAD curve endpoint lies outside its source segment.")
            fractions.append(fraction)
        if fractions[0] == fractions[1]:
            raise ValueError("CAD target curve edge has a degenerate source interval.")
        endpoints = tuple(
            interval[0] + value * (interval[1] - interval[0]) for value in fractions
        )
        child = (min(endpoints), max(endpoints))
        parts.setdefault(authored_row, []).append(child)
        targets.append(int(edge_ids[edge_row]))
        parent_ids.append(int(segment_ids[authored_row]))
        source_ids.append(original.source_entity_ids[original_row])
        source_indices.append(int(old_indices[original_row]))
        paths.append(tuple(original.source_occurrence_paths[original_row]))
        intervals.append(child)
        orientations.append(
            (1 if endpoints[1] > endpoints[0] else -1) * senses[authored_row]
        )
        # The output edge is bounded from its PLC parent and the parent from
        # the source curve; use the triangle inequality, rounded outward.
        bound = float(old_residuals[original_row]) + float(
            np.max(deviations[edges[edge_row]])
        )
        if not np.isfinite(bound) or bound < 0:
            raise ValueError("CAD curve edge has no finite source residual bound.")
        residuals.append(0.0 if bound == 0.0 else float(np.nextafter(bound, np.inf)))
    for authored_row, surface_edge_id in enumerate(segment_ids.tolist()):
        original_row = original_rows[int(surface_edge_id)]
        root = original.source_parameter_intervals[original_row]
        children = sorted(parts.get(authored_row, ()))
        if (
            root is None
            or not children
            or children[0][0] != root[0]
            or children[-1][1] != root[1]
            or any(left[1] != right[0] for left, right in zip(children, children[1:]))
        ):
            raise ValueError(
                "CAD target curve edges do not exactly partition their source interval."
            )
    return GeometryAssociation(
        GeometryAssociationKind.CURVE,
        source.source_id,
        source.source_revision,
        mesh.entity_set(1).entity_set_id,
        np.asarray(targets, dtype=np.int64),
        tuple(source_ids),
        np.asarray(residuals, dtype=np.float64),
        source_dimensions=np.ones(len(targets), dtype=np.int8),
        source_indices=np.asarray(source_indices, dtype=np.int32),
        source_roles=(GeometrySourceEntityRole.EDGE,) * len(targets),
        source_occurrence_paths=tuple(paths),
        source_parameter_intervals=tuple(intervals),
        orientations=np.asarray(orientations, dtype=np.int8),
        parent_dimensions=np.ones(len(targets), dtype=np.int8),
        parent_ids=np.asarray(parent_ids, dtype=np.int64),
        parent_association_id=original.association_id,
        provenance=GeometryAssociationProvenance.LINEAGE,
        exact=False,
    )


def _cad_volume_chart_witness(
    source: NativeCadVolumeSource,
    construction: VolumeConstruction,
    geometry: CellGeometrySpec,
    curve_association: GeometryAssociation | None,
    /,
) -> SourceBoundaryChartWitness:
    from ._native_volume import _boundary_face_entities

    domain = source.domain
    query = source.fidelity_source
    surface_source = source.surface_source
    if not isinstance(query, MeshingDomainBoundarySource):
        raise TypeError("CAD chart witnesses require the original domain boundary query.")
    query.require_current()
    surface_source.require_root(source.surface.mesh, source.surface.geometry)
    surface_source.require_bound(source.surface.mesh, source.surface.geometry, query)

    source_triangle_ids = source.source_triangle_ids
    source_triangle_cell_ids = np.asarray(source.source_triangle_cell_ids, dtype=np.int64)
    source_triangle_patches = np.asarray(source.source_triangle_patches, dtype=np.int64)
    source_triangle_parameters = np.asarray(
        source.source_triangle_parameters, dtype=np.float64
    )
    source_triangle_parameter_exact = source.source_triangle_parameter_exact
    if (
        source_triangle_cell_ids.shape != (len(source_triangle_ids),)
        or np.unique(source_triangle_cell_ids).size != source_triangle_cell_ids.size
        or source_triangle_patches.shape != source_triangle_cell_ids.shape
        or source_triangle_parameters.shape != (source_triangle_cell_ids.size, 3, 2)
        or len(source_triangle_parameter_exact) != source_triangle_cell_ids.size
        or any(
            len(triangle) != 3 or any(len(point) != 2 for point in triangle)
            for triangle in source_triangle_parameter_exact
        )
        or not np.all(np.isfinite(source_triangle_parameters))
        or len(set(source_triangle_ids)) != len(source_triangle_ids)
    ):
        raise ValueError("CAD source triangle cell IDs and UV roots are not aligned.")
    if any(
        not np.array_equal(
            np.asarray(
                [[float(value) for value in point] for point in exact_triangle],
                dtype=np.float64,
            ),
            source_triangle_parameters[row],
        )
        for row, exact_triangle in enumerate(source_triangle_parameter_exact)
    ):
        raise ValueError(
            "CAD UV values differ from their retained exact surface chart roots."
        )
    if set(source_triangle_cell_ids.tolist()) != set(surface_source.cell_global_ids):
        raise ValueError(
            "CAD volume source roots omit an original surface triangle cell."
        )
    complex_ = source.plc_source.complex
    if (
        complex_.cad_polygon_cell_ids is None
        or not np.array_equal(
            source_triangle_cell_ids,
            np.asarray(complex_.cad_polygon_cell_ids, dtype=np.int64),
        )
        or not np.array_equal(
            source_triangle_patches,
            np.asarray(complex_.polygon_facets, dtype=np.int64),
        )
    ):
        raise ValueError("CAD source triangles changed their canonical PLC row bindings.")
    root_rows = {
        int(cell_id): row for row, cell_id in enumerate(surface_source.cell_global_ids)
    }
    face_association = _association_for(surface_source, 2, required=True)
    if face_association is None:
        raise ValueError("CAD source charts require a source-face association.")
    face_rows = _association_row_map(face_association)
    face_orientations = np.asarray(face_association.orientations, dtype=np.int8)
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    for row, cell_id in enumerate(source_triangle_cell_ids.tolist()):
        root_row = root_rows[int(cell_id)]
        root = surface_source.root_parameter_chart(root_row)
        if int(face_orientations[face_rows[int(cell_id)]]) < 0:
            root = (root[0], root[2], root[1])
        polygon = tuple(int(value) for value in vertices[offsets[row] : offsets[row + 1]])
        if (
            len(polygon) != 3
            or source_triangle_parameter_exact[row] != root
            or int(surface_source.root_patches[root_row])
            != int(source_triangle_patches[row])
            or source_triangle_ids[row]
            != _plc_source_triangle_id(
                polygon,
                int(source_triangle_patches[row]),
                complex_.vertex_ids,
                complex_.vertex_occurrence_paths,
                complex_.facet_ids,
                complex_.facet_occurrence_paths,
            )
        ):
            raise ValueError("CAD source triangle changed its exact root chart ancestry.")
    source_chart_by_face_id = {
        identifier: row for row, identifier in enumerate(source_triangle_ids)
    }

    mesh = construction.mesh
    if (
        mesh.topological_dimension != 3
        or len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind not in ("simplex:3", "tetrahedron")
        or mesh.blocks[0].arity != 4
    ):
        raise ValueError("CAD source chart witnesses require tetrahedral volume cells.")
    block = mesh.blocks[0]
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    cells = np.asarray(block.vertices, dtype=np.int64)
    cell_regions = np.asarray(construction.cell_regions, dtype=np.int64)
    _, cell_faces = volume_cell_entity_rows(mesh, cell_ids)
    mesh_face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    boundary_mask = np.asarray(mesh.boundary_masks[2], dtype=np.bool_)
    reference = reference_cell_topology("tetrahedron")
    reference_vertices = np.asarray(reference.vertices, dtype=np.float64)
    reference_faces = tuple(
        tuple(int(value) for value in row) for row in reference.entities[2]
    )
    if (
        cells.shape != (cell_ids.size, 4)
        or cell_regions.shape != cell_ids.shape
        or cell_faces.shape != (cell_ids.size, 4)
        or boundary_mask.shape != mesh_face_ids.shape
        or reference_vertices.shape != (4, 3)
    ):
        raise ValueError("CAD volume cell and facet ancestry arrays are misaligned.")
    face_row_by_id = {
        int(identifier): row for row, identifier in enumerate(mesh_face_ids.tolist())
    }
    owners_by_face: dict[int, list[tuple[int, int, int, int]]] = {}
    for cell_row, cell_id in enumerate(cell_ids.tolist()):
        region = int(cell_regions[cell_row])
        for local_facet, face_row in enumerate(cell_faces[cell_row].tolist()):
            face_id = int(mesh_face_ids[int(face_row)])
            owners_by_face.setdefault(face_id, []).append(
                (int(cell_id), cell_row, local_facet, region)
            )

    boundary_faces = np.asarray(construction.boundary_faces, dtype=np.int64)
    target_face_ids = np.asarray(
        _boundary_face_entities(mesh, construction), dtype=np.int64
    )
    face_source_facets = np.asarray(construction.face_source_facets, dtype=np.int64)
    face_sources = np.asarray(construction.face_sources, dtype=np.int64)
    face_orientations = np.asarray(construction.face_orientations, dtype=np.int64)
    source_faces = np.asarray(construction.source_faces, dtype=np.int64)
    source_face_facets = np.asarray(construction.source_face_facets, dtype=np.int64)
    source_face_ids = tuple(construction.source_face_ids)
    source_face_paths = tuple(construction.source_face_occurrence_paths)
    if (
        boundary_faces.ndim != 2
        or boundary_faces.shape[1] != 3
        or target_face_ids.shape != (boundary_faces.shape[0],)
        or face_source_facets.shape != target_face_ids.shape
        or face_sources.shape != target_face_ids.shape
        or face_orientations.shape != target_face_ids.shape
        or len(construction.face_source_ids) != target_face_ids.size
        or len(construction.face_source_occurrence_paths) != target_face_ids.size
        or len(set(target_face_ids.tolist())) != target_face_ids.size
        or source_faces.ndim != 2
        or source_faces.shape[1] != 3
        or source_face_facets.shape != (source_faces.shape[0],)
        or len(source_face_ids) != source_faces.shape[0]
        or len(source_face_paths) != source_faces.shape[0]
        or len(set(source_face_ids)) != source_faces.shape[0]
    ):
        raise ValueError("CAD boundary/source facet ancestry arrays are misaligned.")
    if set(source_face_ids) != set(source_chart_by_face_id):
        raise ValueError("Volume source facets lost original CAD surface triangle IDs.")
    target_face_id_set = set(target_face_ids.tolist())
    required_boundary_faces = set(mesh_face_ids[boundary_mask].tolist())
    for face_id, owners in owners_by_face.items():
        if len(owners) > 2:
            raise ValueError("CAD volume contains a nonmanifold target face.")
        if len(owners) == 2 and owners[0][3] != owners[1][3]:
            required_boundary_faces.add(face_id)
    if not required_boundary_faces.issubset(target_face_id_set):
        raise ValueError(
            "CAD source chart witness omits an output exterior or material-interface face."
        )

    input_vertices = np.asarray(construction.input_vertices, dtype=np.int64)
    witness_strata = np.asarray(construction.witness_strata, dtype=np.int64)
    witness_entities = np.asarray(construction.witness_entities, dtype=np.int64)
    witness_parameters = np.asarray(construction.witness_parameters, dtype=np.float64)
    source_segment_vertices = np.asarray(
        construction.source_segment_vertices, dtype=np.int64
    )
    if (
        input_vertices.shape != (mesh.coordinates.shape[0],)
        or witness_strata.shape != input_vertices.shape
        or witness_entities.shape != input_vertices.shape
        or witness_parameters.shape != (input_vertices.size, 2)
        or source_segment_vertices.ndim != 2
        or source_segment_vertices.shape[1] != 2
    ):
        raise ValueError("CAD volume vertices lost their source witness parameters.")

    chart_parameters_by_source_face: list[tuple[tuple[Fraction, Fraction], ...]] = []
    triangles_by_patch: dict[int, set[int]] = {}
    triangles_by_vertex: dict[int, set[int]] = {}
    triangles_by_edge: dict[tuple[int, int], set[int]] = {}
    patch_vertex_parameters: dict[int, dict[int, set[tuple[Fraction, Fraction]]]] = {}
    patch_root_orientations: dict[int, set[int]] = {}
    for row, (face_id, facet) in enumerate(
        zip(source_face_ids, source_face_facets.tolist(), strict=True)
    ):
        chart_row = source_chart_by_face_id[face_id]
        if int(source_triangle_patches[chart_row]) != int(facet):
            raise ValueError("CAD source triangle ID changed its B-Rep patch.")
        if source_face_paths[row] != domain.source_occurrences[2][int(facet)]:
            raise ValueError("CAD source triangle changed its patch occurrence path.")
        chart_parameters_by_source_face.append(source_triangle_parameter_exact[chart_row])
        triangle = (
            int(source_faces[row, 0]),
            int(source_faces[row, 1]),
            int(source_faces[row, 2]),
        )
        if len(set(triangle)) != 3:
            raise ValueError("CAD source triangle ancestry is degenerate.")
        patch_root_orientations.setdefault(int(facet), set()).add(
            _cad_uv_triangle_orientation(chart_parameters_by_source_face[row])
        )
        patch_vertices = patch_vertex_parameters.setdefault(int(facet), {})
        for vertex, uv in zip(
            triangle,
            chart_parameters_by_source_face[row],
            strict=True,
        ):
            patch_vertices.setdefault(vertex, set()).add(uv)
        triangles_by_patch.setdefault(int(facet), set()).add(row)
        for vertex in triangle:
            triangles_by_vertex.setdefault(vertex, set()).add(row)
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_key = (min(first, second), max(first, second))
            triangles_by_edge.setdefault(edge_key, set()).add(row)

    region_signs_by_patch: list[dict[int, int]] = [{} for _ in range(len(domain.patches))]
    for region_row, region in enumerate(domain.regions):
        for patch, sign in region.boundary:
            if sign not in (-1, 1) or patch < 0 or patch >= len(domain.patches):
                raise ValueError(
                    "CAD region boundary has invalid oriented patch incidence."
                )
            if region_row in region_signs_by_patch[patch]:
                raise ValueError("CAD region repeats one oriented source patch.")
            region_signs_by_patch[patch][region_row] = int(sign)
    for patch, signs in enumerate(region_signs_by_patch):
        patch_orientation = int(domain.patches[patch].orientation)
        if (
            patch_orientation not in (-1, 1)
            or set(signs) != set(domain.patch_regions[patch])
            or not signs
            or len(signs) > 2
        ):
            raise ValueError("CAD patch material incidence is incomplete or ambiguous.")
        if len(signs) == 2 and {patch_orientation * sign for sign in signs.values()} != {
            -1,
            1,
        }:
            raise ValueError(
                "CAD material-interface sides do not have opposite orientation."
            )

    reference_rows: list[np.ndarray] = []
    reference_rational_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    source_rows: list[np.ndarray] = []
    source_rational_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    parent_cells: list[int] = []
    root_cells: list[int] = []
    local_facets: list[int] = []
    source_chart_ids: list[str] = []
    source_entity_ids: list[str] = []
    occurrence_paths: list[tuple[str, ...]] = []
    orientations: list[int] = []
    declared_planes = source.plc_source.complex.facet_planes
    exact_charts: dict[int, _ExactPlaneChart | None] = {}
    plane_charts: dict[int, ExactPlaneChart | None] = {}

    def revolution_plane(patch: int, /) -> ExactPlaneChart | None:
        if patch not in plane_charts:
            plane_charts[patch] = exact_plane_chart(domain, patch)
        return plane_charts[patch]

    def patch_witness_row(vertex: int, patch: int, /) -> int:
        row = int(witness_entities[vertex])
        if not 0 <= row < source_faces.shape[0]:
            raise ValueError("CAD facet witness names an absent source triangle.")
        if int(source_face_facets[row]) != patch:
            raise ValueError("CAD facet witness changed its source patch.")
        return row

    def patch_input_uv(
        input_vertex: int, patch: int, plane: ExactPlaneChart | None, /
    ) -> tuple[Fraction, Fraction]:
        parameter_candidates = patch_vertex_parameters.get(patch, {}).get(
            input_vertex, set()
        )
        if plane is not None:
            # Seam copies of one authored vertex meet in the plane chart.
            parameter_candidates = {plane.chart_point(uv) for uv in parameter_candidates}
        if input_vertex < 0 or len(parameter_candidates) != 1:
            raise ValueError(
                "CAD planar patch lacks one exact root parameter per output vertex."
            )
        return next(iter(parameter_candidates))

    def planar_vertex_uv(
        vertex: int,
        patch: int,
        chart: _ExactPlaneChart,
        plane: ExactPlaneChart | None,
        /,
    ) -> tuple[Fraction, Fraction]:
        """Chart parameters of a face vertex in its planar patch's one exact chart.

        The chart is the patch's own affine chart, or with ``plane`` its exact
        plane chart, where straight source segments and triangles stay affine.
        """
        input_vertex = int(input_vertices[vertex])
        stratum = int(witness_strata[vertex])
        if input_vertex >= 0:
            return patch_input_uv(input_vertex, patch, plane)
        if stratum == 1:
            edge_row = int(witness_entities[vertex])
            if not 0 <= edge_row < source_segment_vertices.shape[0]:
                raise ValueError("CAD edge witness names an absent source segment.")
            first, second = (
                patch_input_uv(int(value), patch, plane)
                for value in source_segment_vertices[edge_row]
            )
            parameter = Fraction.from_float(float(witness_parameters[vertex, 0]))
            if not 0 <= parameter <= 1:
                raise ValueError("CAD edge witness lies outside its source segment.")
            return (
                first[0] + parameter * (second[0] - first[0]),
                first[1] + parameter * (second[1] - first[1]),
            )
        if stratum == 2:
            root = chart_parameters_by_source_face[patch_witness_row(vertex, patch)]
            if plane is not None:
                root = tuple(plane.chart_point(uv) for uv in root)
            u, v = (
                Fraction.from_float(float(value)) for value in witness_parameters[vertex]
            )
            if u < 0 or v < 0 or u + v > 1:
                raise ValueError("CAD facet witness lies outside its source triangle.")
            return (
                root[0][0]
                + u * (root[1][0] - root[0][0])
                + v * (root[2][0] - root[0][0]),
                root[0][1]
                + u * (root[1][1] - root[0][1])
                + v * (root[2][1] - root[0][1]),
            )
        if stratum == 3:
            patch_witness_row(vertex, patch)
            plane_rows = (
                None if declared_planes is None else np.asarray(declared_planes[patch])
            )
            if plane_rows is None or not np.all(np.isfinite(plane_rows)):
                raise ValueError("CAD plane witness lies on an undeclared source patch.")
            return _cad_plane_witness_uv(chart, plane_rows, witness_parameters[vertex])
        raise ValueError(
            "CAD boundary vertex has no exact source vertex, edge, facet, or plane witness."
        )

    for face_row in range(boundary_faces.shape[0]):
        face = (
            int(boundary_faces[face_row, 0]),
            int(boundary_faces[face_row, 1]),
            int(boundary_faces[face_row, 2]),
        )
        target_face_id = int(target_face_ids[face_row])
        patch = int(face_source_facets[face_row])
        if not 0 <= patch < len(domain.patches):
            raise ValueError("CAD output facet names an undeclared source patch.")
        entity_id = domain.entity_id(2, patch)
        occurrence = domain.source_occurrences[2][patch]
        complex_ = source.plc_source.complex
        if (
            int(face_sources[face_row]) != int(complex_.facet_entity_ids[patch])
            or construction.face_source_ids[face_row] != entity_id
            or tuple(construction.face_source_occurrence_paths[face_row]) != occurrence
        ):
            raise ValueError("CAD output face changed its source patch identity or path.")
        face_orientation = int(face_orientations[face_row])
        if face_orientation not in (-1, 1):
            raise ValueError("CAD output face has unresolved source-facet orientation.")
        owners = owners_by_face.get(int(target_face_id), [])
        expected_region_signs = region_signs_by_patch[patch]
        if (
            len(owners) != len(expected_region_signs)
            or {owner[3] for owner in owners} != set(expected_region_signs)
            or len({owner[3] for owner in owners}) != len(owners)
        ):
            raise ValueError(
                "CAD output face does not retain its material-side incidence."
            )
        target_row = face_row_by_id.get(int(target_face_id))
        if target_row is None or bool(boundary_mask[target_row]) != (
            len(expected_region_signs) == 1
        ):
            raise ValueError(
                "CAD source face is not an actual exterior or interface facet."
            )
        if len(expected_region_signs) == 1:
            selected_region = next(iter(expected_region_signs))
        else:
            positive_regions = tuple(
                region
                for region, sign in expected_region_signs.items()
                if int(domain.patches[patch].orientation) * sign == 1
            )
            if len(positive_regions) != 1:
                raise ValueError(
                    "CAD material interface has no unique positive source side."
                )
            selected_region = positive_regions[0]
        selected_owners = tuple(owner for owner in owners if owner[3] == selected_region)
        if len(selected_owners) != 1:
            raise ValueError("CAD source patch has no unique material-side volume owner.")
        parent_id, cell_row, local_facet, region = selected_owners[0]
        local_face_vertices = reference_faces[local_facet]
        cell_vertices = tuple(int(value) for value in cells[cell_row])
        positions_by_vertex = {
            vertex: position for position, vertex in enumerate(cell_vertices)
        }
        if set(int(value) for value in face) != {
            cell_vertices[position] for position in local_face_vertices
        }:
            raise ValueError("CAD target face and selected local volume facet disagree.")
        local_positions = tuple(positions_by_vertex[int(value)] for value in face)
        local_permutation = tuple(
            local_face_vertices.index(position) for position in local_positions
        )
        inversions = sum(
            local_permutation[first] > local_permutation[second]
            for first in range(3)
            for second in range(first + 1, 3)
        )
        target_orientation = -1 if inversions % 2 else 1
        region_sign = expected_region_signs[region]
        if target_orientation != face_orientation * region_sign:
            raise ValueError(
                "CAD target facet orientation contradicts its source patch and material side."
            )

        face_vertex_rows = tuple(int(value) for value in face)
        vertex_candidate_sets: list[set[int]] = []
        for vertex in face_vertex_rows:
            input_vertex = int(input_vertices[vertex])
            if input_vertex >= 0:
                vertex_candidates = triangles_by_vertex.get(input_vertex, set())
            elif int(witness_strata[vertex]) == 2:
                source_face_row = int(witness_entities[vertex])
                if not 0 <= source_face_row < source_faces.shape[0]:
                    raise ValueError("CAD facet witness names an absent source triangle.")
                vertex_candidates = {source_face_row}
            elif int(witness_strata[vertex]) == 1:
                edge_row = int(witness_entities[vertex])
                if not 0 <= edge_row < source_segment_vertices.shape[0]:
                    raise ValueError("CAD edge witness names an absent source edge.")
                edge_vertices = tuple(
                    int(value) for value in source_segment_vertices[edge_row]
                )
                edge_inputs = edge_vertices
                if min(edge_inputs) < 0 or edge_inputs[0] == edge_inputs[1]:
                    raise ValueError(
                        "CAD source edge lost its original endpoint ancestry."
                    )
                vertex_candidates = triangles_by_edge.get(
                    tuple(sorted(edge_inputs)), set()
                )
            elif int(witness_strata[vertex]) == 3:
                # A point on its patch's declared plane is not bound to one
                # source triangle: its face takes the patch's planar chart.
                vertex_candidates = set()
            else:
                raise ValueError(
                    "CAD boundary vertex has no exact source vertex, edge, facet, or plane witness."
                )
            vertex_candidate_sets.append(vertex_candidates)
        smallest_candidates = min(vertex_candidate_sets, key=len)
        candidates = set(smallest_candidates)
        for vertex_candidates in vertex_candidate_sets:
            candidates.intersection_update(vertex_candidates)
        candidates.intersection_update(triangles_by_patch.get(patch, ()))
        if len(candidates) > 1:
            raise ValueError(
                "CAD output face has ambiguous original surface triangle ancestry."
            )
        patch_orientation = int(domain.patches[patch].orientation)
        planar_parameters: list[tuple[Fraction, Fraction]] | None = None
        # Every face of a patch with an exact plane chart takes that chart, so
        # its target triangles partition one chart of the patch.
        plane = revolution_plane(patch)
        if candidates and plane is None:
            source_face_row = next(iter(candidates))
            root_parameters = chart_parameters_by_source_face[source_face_row]
            root_orientation = _cad_uv_triangle_orientation(root_parameters)
            source_triangle = tuple(int(value) for value in source_faces[source_face_row])
        else:
            if plane is not None:
                chart = (plane.origin, plane.first_axis, plane.second_axis)
            else:
                if patch not in exact_charts:
                    exact_charts[patch] = _exact_patch_chart(domain, patch)
                exact_chart = exact_charts[patch]
                if exact_chart is None:
                    raise ValueError(
                        "CAD output face spans source triangles without an exact planar patch chart."
                    )
                chart = exact_chart
            if patch_root_orientations.get(patch) != {patch_orientation}:
                raise ValueError(
                    "CAD planar patch source triangles do not share one exact oriented chart."
                )
            planar_parameters = [
                planar_vertex_uv(vertex, patch, chart, plane)
                for vertex in face_vertex_rows
            ]
            source_triangle = ()
            root_parameters = ()
            root_orientation = patch_orientation
        if root_orientation != patch_orientation:
            raise ValueError(
                "CAD source UV roots do not retain their authored patch orientation."
            )
        source_corner_by_vertex = {
            vertex: corner for corner, vertex in enumerate(source_triangle)
        }
        # A face of a planar chart already carries its vertices' chart
        # parameters; a face inside one source triangle restricts each vertex to
        # that triangle's roots.
        child_parameters = [] if planar_parameters is None else planar_parameters
        triangle_vertex_rows = face_vertex_rows if planar_parameters is None else ()
        for vertex in triangle_vertex_rows:
            input_vertex = int(input_vertices[vertex])
            if input_vertex >= 0:
                corner = source_corner_by_vertex.get(input_vertex)
                if corner is None:
                    raise ValueError(
                        "CAD target vertex is absent from its source triangle ancestry."
                    )
                uv = tuple(root_parameters[corner])
            elif int(witness_strata[vertex]) == 2:
                if int(witness_entities[vertex]) != source_face_row:
                    raise ValueError(
                        "CAD facet witness changed its source triangle parent."
                    )
                u, v = (
                    Fraction.from_float(float(value))
                    for value in witness_parameters[vertex]
                )
                if u < 0 or v < 0 or u + v > 1:
                    raise ValueError(
                        "CAD facet witness lies outside its source triangle."
                    )
                uv = (
                    root_parameters[0][0]
                    + u * (root_parameters[1][0] - root_parameters[0][0])
                    + v * (root_parameters[2][0] - root_parameters[0][0]),
                    root_parameters[0][1]
                    + u * (root_parameters[1][1] - root_parameters[0][1])
                    + v * (root_parameters[2][1] - root_parameters[0][1]),
                )
            else:
                edge_row = int(witness_entities[vertex])
                if not 0 <= edge_row < source_segment_vertices.shape[0]:
                    raise ValueError("CAD edge witness names an absent source segment.")
                edge_vertices = tuple(
                    int(value) for value in source_segment_vertices[edge_row]
                )
                edge_inputs = edge_vertices
                first_corner = source_corner_by_vertex.get(edge_inputs[0])
                second_corner = source_corner_by_vertex.get(edge_inputs[1])
                parameter = Fraction.from_float(float(witness_parameters[vertex, 0]))
                if (
                    first_corner is None
                    or second_corner is None
                    or not 0 <= parameter <= 1
                ):
                    raise ValueError(
                        "CAD edge witness cannot be restricted to its source triangle."
                    )
                uv = (
                    root_parameters[first_corner][0]
                    + parameter
                    * (
                        root_parameters[second_corner][0]
                        - root_parameters[first_corner][0]
                    ),
                    root_parameters[first_corner][1]
                    + parameter
                    * (
                        root_parameters[second_corner][1]
                        - root_parameters[first_corner][1]
                    ),
                )
            child_parameters.append(uv)
        child_parameter_array = np.asarray(
            [tuple(float(value) for value in uv) for uv in child_parameters],
            dtype=np.float64,
        )
        source_orientation = _cad_uv_triangle_orientation(tuple(child_parameters))
        if source_orientation != face_orientation * root_orientation:
            raise ValueError(
                "CAD output UV triangle contradicts its retained source-facet witness."
            )
        if source_orientation * target_orientation != patch_orientation * region_sign:
            raise ValueError(
                "CAD witness orientation contradicts source patch material incidence."
            )
        reference_points = reference_vertices[np.asarray(local_positions, dtype=np.int64)]
        reference_rows.append(reference_points)
        if not np.all(reference_points == np.floor(reference_points)):
            raise ValueError("Tetrahedral reference vertices must be exactly integral.")
        reference_rational_rows.append(
            tuple(
                tuple(Fraction(int(value)) for value in point)
                for point in reference_points
            )
        )
        source_rows.append(child_parameter_array)
        source_rational_rows.append(tuple(child_parameters))
        parent_cells.append(parent_id)
        local_facets.append(local_facet)
        source_chart_ids.append(entity_id if plane is None else plane.chart_id)
        source_entity_ids.append(entity_id)
        occurrence_paths.append(occurrence)
        orientations.append(source_orientation * target_orientation)
        root_cells.append(
            int(
                source_triangle_cell_ids[
                    source_chart_by_face_id[source_face_ids[next(iter(candidates))]]
                ]
            )
            if plane is None and len(candidates) == 1
            else -1
        )

    if not parent_cells:
        raise ValueError("CAD source chart witness cannot be empty.")
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=query.source_scope_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(parent_cells, dtype=np.int64),
        entity_global_ids=target_face_ids,
        local_entity_indices=np.asarray(local_facets, dtype=np.int32),
        target_reference_vertices=np.asarray(reference_rows, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(reference_rational_rows),
        source_chart_ids=tuple(source_chart_ids),
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrence_paths),
        source_parameters=np.asarray(source_rows, dtype=np.float64),
        source_parameter_rational_tokens=rational_chart_tokens(source_rational_rows),
        orientations=np.asarray(orientations, dtype=np.int8),
        cad_plc=source.plc_source.complex,
        cad_curve_association=curve_association,
        cad_root_cell_ids=np.asarray(root_cells, dtype=np.int64),
    )


def _cad_surface_feature_witness(
    domain: MeshingDomain,
    feature: ProtectedFeature,
    full_witness: SourceBoundaryChartWitness,
    /,
) -> tuple[MeshingDomainBoundarySource, SourceBoundaryChartWitness]:
    if feature.scope.entity_dimension != 2:
        raise ValueError("CAD surface features must name source dimension-2 scopes.")
    keys = tuple(
        domain.source_entity_key_for_scope(2, int(identifier))
        for identifier in np.asarray(feature.scope.entity_ids, dtype=np.int64).tolist()
    )
    patch_rows = tuple(
        sorted(
            {
                domain.resolve_entity_id(2, entity_id, occurrence_path=path)
                for entity_id, path in keys
            }
        )
    )
    query = MeshingDomainBoundarySource(domain, patch_rows)
    selected = set(keys)
    rows = np.asarray(
        [
            row
            for row, (entity_id, path) in enumerate(
                zip(
                    full_witness.source_entity_ids,
                    full_witness.occurrence_paths,
                    strict=True,
                )
            )
            if (entity_id, path) in selected
        ],
        dtype=np.int64,
    )
    if rows.size == 0:
        raise ValueError(
            "CAD surface feature has no witnessed source-face output facets."
        )
    witness = SourceBoundaryChartWitness(
        mesh_id=full_witness.mesh_id,
        geometry_id=full_witness.geometry_id,
        topology_id=full_witness.topology_id,
        source_id=full_witness.source_id,
        source_revision=full_witness.source_revision,
        source_query_id=query.source_scope_id,
        chart_dimension=full_witness.chart_dimension,
        reference_dimension=full_witness.reference_dimension,
        cell_global_ids=np.asarray(full_witness.cell_global_ids, dtype=np.int64)[rows],
        entity_global_ids=np.asarray(full_witness.entity_global_ids, dtype=np.int64)[
            rows
        ],
        local_entity_indices=np.asarray(
            full_witness.local_entity_indices, dtype=np.int32
        )[rows],
        target_reference_vertices=np.asarray(full_witness.target_reference_vertices)[
            rows
        ],
        target_reference_rational_tokens=tuple(
            full_witness.target_reference_rational_tokens[int(row)] for row in rows
        ),
        source_chart_ids=tuple(full_witness.source_chart_ids[int(row)] for row in rows),
        source_entity_ids=tuple(full_witness.source_entity_ids[int(row)] for row in rows),
        occurrence_paths=tuple(full_witness.occurrence_paths[int(row)] for row in rows),
        source_parameters=np.asarray(full_witness.source_parameters)[rows],
        source_parameter_rational_tokens=tuple(
            full_witness.source_parameter_rational_tokens[int(row)] for row in rows
        ),
        orientations=np.asarray(full_witness.orientations, dtype=np.int8)[rows],
        cad_plc=full_witness.cad_plc,
        cad_curve_association=full_witness.cad_curve_association,
        cad_root_cell_ids=np.asarray(full_witness.cad_root_cell_ids, dtype=np.int64)[
            rows
        ],
    )
    return query, witness


def _cad_corner_feature_witness(
    source: NativeCadVolumeSource,
    construction: VolumeConstruction,
    geometry: CellGeometrySpec,
    feature: ProtectedFeature,
    /,
) -> SourceBoundaryPointWitness:
    domain = source.domain
    query = source.fidelity_source
    if not isinstance(query, MeshingDomainBoundarySource):
        raise TypeError(
            "CAD corner witnesses require the original domain boundary query."
        )
    identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
    if (
        feature.scope.entity_dimension != 0
        or identifiers.ndim != 1
        or identifiers.size == 0
    ):
        raise ValueError("CAD corner features must name authored source point scopes.")
    mesh = construction.mesh
    input_vertices = np.asarray(construction.input_vertices, dtype=np.int64)
    complex_ = source.plc_source.complex
    target_ids = []
    source_ids = []
    occurrences = []
    for identifier in identifiers.tolist():
        entity_id, occurrence = domain.source_entity_key_for_scope(0, int(identifier))
        source_row = domain.resolve_entity_id(0, entity_id, occurrence_path=occurrence)
        source_index = int(domain.source_indices[0][source_row])
        source_occurrence = domain.source_occurrences[0][source_row]
        plc_rows = tuple(
            row
            for row, (dimension, index, path) in enumerate(
                zip(
                    np.asarray(source.vertex_brep_dimensions, dtype=np.int64).tolist(),
                    np.asarray(source.vertex_brep_indices, dtype=np.int64).tolist(),
                    complex_.vertex_occurrence_paths,
                    strict=True,
                )
            )
            if dimension == 0
            and index == source_index
            and tuple(path) == source_occurrence
        )
        if len(plc_rows) != 1:
            raise ValueError(
                "CAD source corner has no unique original PLC vertex ancestry."
            )
        target_rows = np.flatnonzero(input_vertices == plc_rows[0])
        if target_rows.size != 1:
            raise ValueError(
                "CAD source corner does not map to one retained volume vertex."
            )
        target_ids.append(
            int(np.asarray(mesh.vertex_global_ids, dtype=np.int64)[target_rows[0]])
        )
        source_ids.append(entity_id)
        occurrences.append(occurrence)
    return SourceBoundaryPointWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_scope_id,
        vertex_global_ids=np.asarray(target_ids, dtype=np.int64),
        source_entity_ids=tuple(source_ids),
        occurrence_paths=tuple(occurrences),
    )


def _cad_curve_feature_witness(
    source: NativeCadVolumeSource,
    construction: VolumeConstruction,
    geometry: CellGeometrySpec,
    feature: ProtectedFeature,
    surface_witness: SourceBoundaryChartWitness,
    /,
) -> tuple[SourceBoundaryQuery, SourceBoundaryChartWitness]:
    from ._native_surface import _domain_curve_source_query
    from ._native_volume import _boundary_face_entities

    domain = source.domain
    identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
    if (
        feature.scope.entity_dimension != 1
        or identifiers.ndim != 1
        or identifiers.size == 0
    ):
        raise ValueError("CAD curve features must name authored source-edge scopes.")
    query = _domain_curve_source_query(domain, identifiers)
    domain.validate_curve_boundary_query(query, identifiers)
    from ...geometry._curve_boundary import ParametricCurveBoundarySource

    if not isinstance(query, ParametricCurveBoundarySource):
        raise TypeError(
            "CAD curve feature preparation requires ParametricCurveBoundarySource."
        )

    scope_id_by_key = {
        domain.source_entity_key_for_scope(1, int(scope_id)): int(scope_id)
        for scope_id in domain.scope_entity_ids(1).tolist()
    }
    chart_rows: dict[str, tuple[int, int, PatchCurveUse, int]] = {}
    query_rows = {chart_id: row for row, chart_id in enumerate(query.chart_ids)}
    for chart_id, (entity_id, occurrence) in zip(
        query.chart_ids,
        zip(query.curve_entity_ids, query.occurrence_paths, strict=True),
        strict=True,
    ):
        numeric_id = scope_id_by_key[(entity_id, occurrence)]
        uses = tuple(
            value
            for value in domain.source_curve_chart_uses(numeric_id)
            if value[0] == chart_id
        )
        if len(uses) != 1:
            raise ValueError("CAD curve query chart has no unique authored trim use.")
        _, patch_row, _, _, use = uses[0]
        chart_rows[chart_id] = (patch_row, use.curve, use, query_rows[chart_id])
    if set(chart_rows) != set(query.chart_ids):
        raise ValueError("CAD curve query omitted an authored trim-use chart.")

    complex_ = source.plc_source.complex
    source_segment_vertices_all = np.asarray(
        construction.source_segment_vertices, dtype=np.int64
    )
    source_segment_complex_rows = np.asarray(
        construction.source_segment_complex_rows, dtype=np.int64
    )
    canonical_edges = np.asarray(complex_.plc_edge_vertices, dtype=np.int64).reshape(
        (-1, 2)
    )
    canonical_edge_row_by_key = {
        (identifier, tuple(path)): row
        for row, (identifier, path) in enumerate(
            zip(
                complex_.plc_edge_ids,
                complex_.plc_edge_occurrence_paths,
                strict=True,
            )
        )
    }
    source_bank_row_by_complex_edge = {
        int(complex_row): row
        for row, complex_row in enumerate(source_segment_complex_rows.tolist())
    }
    curve_rows = np.asarray(source.segment_curve_rows, dtype=np.int64)
    brep_edge_rows = np.asarray(source.segment_brep_indices, dtype=np.int64)
    complex_segments = np.asarray(complex_.segments, dtype=np.int64)
    if (
        len(canonical_edge_row_by_key) != canonical_edges.shape[0]
        or source_segment_vertices_all.ndim != 2
        or source_segment_vertices_all.shape[1] != 2
        or source_segment_complex_rows.shape != (source_segment_vertices_all.shape[0],)
        or len(source_bank_row_by_complex_edge) != source_segment_complex_rows.size
        or complex_segments.shape != (len(complex_.segment_ids), 2)
        or curve_rows.shape != (len(complex_.segment_ids),)
        or brep_edge_rows.shape != curve_rows.shape
    ):
        raise ValueError("CAD curve and canonical PLC source-edge banks are misaligned.")
    # Curve-feature witnesses deliberately include only authored CAD segment IDs.
    actual_segment_source_rows: list[int] = []
    segment_curves: list[int] = []
    segment_edges: list[int] = []
    segment_occurrence_paths: list[tuple[str, ...]] = []
    for curve_segment_row, (identifier, path) in enumerate(
        zip(complex_.segment_ids, complex_.segment_occurrence_paths, strict=True)
    ):
        key = (identifier, tuple(path))
        complex_edge_row = canonical_edge_row_by_key.get(key)
        if complex_edge_row is None:
            raise ValueError(
                "An authored CAD curve is absent from its canonical PLC edge bank."
            )
        source_row = source_bank_row_by_complex_edge.get(complex_edge_row)
        if source_row is None:
            raise ValueError(
                "An authored CAD curve is absent from the canonical recovery bank."
            )
        if tuple(int(value) for value in complex_segments[curve_segment_row]) != tuple(
            int(value) for value in canonical_edges[complex_edge_row]
        ):
            raise ValueError(
                "CAD curve identity and PLC edge have different oriented endpoint rows."
            )
        actual_segment_source_rows.append(source_row)
        segment_curves.append(int(curve_rows[curve_segment_row]))
        segment_edges.append(int(brep_edge_rows[curve_segment_row]))
        segment_occurrence_paths.append(tuple(path))
    if len(set(actual_segment_source_rows)) != len(actual_segment_source_rows):
        raise ValueError("CAD curve identities do not map to distinct PLC edge rows.")
    segment_source_rows = np.asarray(actual_segment_source_rows, dtype=np.int64)
    segment_vertices = source_segment_vertices_all[segment_source_rows]
    segment_row_by_source_row = {
        int(source_row): row for row, source_row in enumerate(actual_segment_source_rows)
    }
    segment_curve_rows = np.asarray(segment_curves, dtype=np.int64)
    segment_brep_edge_rows = np.asarray(segment_edges, dtype=np.int64)
    segment_occurrence_path_rows = tuple(segment_occurrence_paths)
    input_vertices = np.asarray(construction.input_vertices, dtype=np.int64)
    strata = np.asarray(construction.witness_strata, dtype=np.int64)
    entities = np.asarray(construction.witness_entities, dtype=np.int64)
    parameters = np.asarray(construction.witness_parameters, dtype=np.float64)
    if (
        segment_vertices.ndim != 2
        or segment_vertices.shape[1] != 2
        or segment_curve_rows.shape != (segment_vertices.shape[0],)
        or segment_brep_edge_rows.shape != segment_curve_rows.shape
        or input_vertices.shape != (construction.mesh.coordinates.shape[0],)
        or strata.shape != input_vertices.shape
        or entities.shape != input_vertices.shape
        or parameters.shape != (input_vertices.size, 2)
    ):
        raise ValueError("CAD curve witness ancestry arrays are misaligned.")
    authority = domain.brep_authority
    if authority is None:
        raise ValueError("CAD curve witnesses require the authoritative BRep source.")
    brep = authority.geometry if isinstance(authority, BRepModel) else authority
    edge_vertices = tuple(
        tuple(int(value) for value in pair) for pair in brep.edge_vertices
    )

    def edge_parameter_range(
        edge_index: int, occurrence_path: tuple[str, ...], /
    ) -> tuple[float, float]:
        if isinstance(authority, BRepModel):
            bounds = authority.source_curve_parameter_domain(
                edge_index, occurrence_path=occurrence_path
            )
        elif isinstance(authority, BRepGeometry):
            if occurrence_path:
                raise ValueError(
                    "Placed BRep curve intervals require their BRepModel owner."
                )
            bounds = authority.source_curve_parameter_domain(edge_index)
        else:
            raise TypeError(
                "MeshingDomain.brep_authority has no exact BRep curve interval."
            )
        first, last = bounds
        return float(first), float(last)

    query_key_rows = {
        (entity_id, path): row
        for row, (entity_id, path) in enumerate(
            zip(query.curve_entity_ids, query.occurrence_paths, strict=True)
        )
    }
    if complex_.cad_source_charts is None or complex_.cad_segment_edge_ids is None:
        raise ValueError("CAD curve witness lost its source chart and edge bindings.")
    original_curves = _association_for(complex_.cad_source_charts, 1, required=True)
    if original_curves is None:
        raise ValueError("CAD source charts require a source-curve association.")
    original_curve_rows = _association_row_map(original_curves)
    source_edge_ids = np.asarray(complex_.cad_segment_edge_ids, dtype=np.int64)
    if source_edge_ids.shape != segment_curve_rows.shape:
        raise ValueError("CAD curve source edge and segment rows are misaligned.")
    segment_parameters: dict[int, tuple[Fraction, Fraction]] = {}
    segment_rows_by_curve: dict[int, list[int]] = {}
    vertex_segments: dict[int, set[int]] = {}
    for segment_row, (curve_row, edge_index) in enumerate(
        zip(segment_curve_rows.tolist(), segment_brep_edge_rows.tolist(), strict=True)
    ):
        if not 0 <= curve_row < len(domain.curves) or not 0 <= edge_index < len(
            edge_vertices
        ):
            raise ValueError("CAD PLC segment names an absent BRep edge definition.")
        source_occurrence = domain.source_occurrences[1][curve_row]
        if (
            int(domain.source_indices[1][curve_row]) != edge_index
            or segment_occurrence_path_rows[segment_row] != source_occurrence
        ):
            raise ValueError("CAD PLC segment changed its source edge occurrence.")
        query_row = query_key_rows.get(
            (domain.entity_id(1, curve_row), source_occurrence)
        )
        if query_row is None:
            continue
        association_row = original_curve_rows.get(int(source_edge_ids[segment_row]))
        if association_row is None:
            raise ValueError("CAD curve segment lost its original source interval.")
        interval = original_curves.source_parameter_intervals[association_row]
        if interval is None:
            raise ValueError("CAD curve segment lacks an exact source interval.")
        patch, loop, position = (int(value) for value in domain.curve_owners[curve_row])
        _cad_curve_parameter_sense(domain, curve_row)
        owner = domain.patches[patch].loops[loop][position]
        first = Fraction.from_float(float(owner.first))
        width = Fraction.from_float(float(owner.last)) - first
        first_parameter, last_parameter = (first + value * width for value in interval)
        segment_parameters[segment_row] = (first_parameter, last_parameter)
        segment_rows_by_curve.setdefault(curve_row, []).append(segment_row)
        for plc_vertex in segment_vertices[segment_row].tolist():
            vertex_segments.setdefault(int(plc_vertex), set()).add(segment_row)

    vertex_segment_rows: list[set[int]] = []
    for vertex_row, input_vertex in enumerate(input_vertices.tolist()):
        if input_vertex >= 0:
            vertex_segment_rows.append(set(vertex_segments.get(int(input_vertex), ())))
        elif int(strata[vertex_row]) == 1:
            source_segment_row = int(entities[vertex_row])
            if not 0 <= source_segment_row < source_segment_vertices_all.shape[0]:
                raise ValueError(
                    "CAD edge witness names an absent canonical source segment."
                )
            actual_segment_row = segment_row_by_source_row.get(source_segment_row)
            vertex_segment_rows.append(
                set() if actual_segment_row is None else {actual_segment_row}
            )
        else:
            vertex_segment_rows.append(set())

    def vertex_parameter_on_segment(vertex_row: int, segment_row: int, /) -> Fraction:
        first_parameter, last_parameter = segment_parameters[segment_row]
        input_vertex = int(input_vertices[vertex_row])
        endpoints = tuple(int(value) for value in segment_vertices[segment_row])
        if input_vertex >= 0:
            if input_vertex == endpoints[0]:
                return first_parameter
            if input_vertex == endpoints[1]:
                return last_parameter
            raise ValueError("CAD target edge vertex is not a source segment endpoint.")
        if int(strata[vertex_row]) != 1:
            raise ValueError("CAD target curve vertex lacks source-edge ancestry.")
        source_segment_row = int(entities[vertex_row])
        if source_segment_row != int(segment_source_rows[segment_row]):
            raise ValueError("CAD generated curve vertex changed its source segment.")
        value = float(parameters[vertex_row, 0])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "CAD generated curve vertex lies outside its source segment."
            )
        fraction = Fraction.from_float(value)
        return first_parameter + fraction * (last_parameter - first_parameter)

    mesh = construction.mesh
    target_edges, _ = simplex_volume_entities(mesh)
    target_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    if target_edges.shape != (target_edge_ids.size, 2):
        raise ValueError("CAD target edge identities are misaligned.")
    boundary_faces = np.asarray(construction.boundary_faces, dtype=np.int64)
    boundary_face_ids = np.asarray(
        _boundary_face_entities(mesh, construction), dtype=np.int64
    )
    face_patches = np.asarray(construction.face_source_facets, dtype=np.int64)
    if (
        boundary_faces.ndim != 2
        or boundary_faces.shape[1] != 3
        or boundary_face_ids.shape != (boundary_faces.shape[0],)
        or face_patches.shape != boundary_face_ids.shape
    ):
        raise ValueError("CAD source-face ancestry cannot bind curve feature edges.")
    edge_face_patches: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_row, face in enumerate(boundary_faces.tolist()):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_key = (min(int(first), int(second)), max(int(first), int(second)))
            edge_face_patches.setdefault(edge_key, []).append(
                (int(boundary_face_ids[face_row]), int(face_patches[face_row]))
            )
    surface_face_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(surface_witness.entity_global_ids, dtype=np.int64).tolist()
        )
    }
    cell_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.blocks[0].global_ids, dtype=np.int64).tolist()
        )
    }
    tetrahedron = reference_cell_topology("tetrahedron")
    reference_vertices = np.asarray(tetrahedron.vertices, dtype=np.float64)
    local_edges = tuple(
        tuple(int(value) for value in pair) for pair in tetrahedron.entities[1]
    )
    source_chart_ids = []
    source_entity_ids = []
    occurrences = []
    source_parameters = []
    source_parameter_rational_rows = []
    target_references = []
    target_reference_rational_rows = []
    target_entity_ids = []
    target_cell_ids = []
    local_edge_indices = []
    orientations = []
    for chart_id, (patch_row, curve_row, use, query_row) in chart_rows.items():
        edge_index = int(domain.source_indices[1][curve_row])
        edge_range = edge_parameter_range(
            edge_index, tuple(domain.source_occurrences[1][curve_row])
        )
        query_range = tuple(float(value) for value in query.parameter_ranges[query_row])
        if tuple(sorted(query_range)) != tuple(sorted(edge_range)):
            raise ValueError(
                "CAD p-curve and BRep edge parameter intervals lack a shared exact scale."
            )
        if curve_row not in segment_rows_by_curve:
            raise ValueError(
                "CAD curve feature has no retained source PLC segment chain."
            )
        entity_id = query.curve_entity_ids[query_row]
        occurrence = query.occurrence_paths[query_row]
        for target_edge_row, target_pair in enumerate(target_edges.tolist()):
            first_vertex, second_vertex = (int(value) for value in target_pair)
            candidates = (
                vertex_segment_rows[first_vertex] & vertex_segment_rows[second_vertex]
            )
            candidates = {
                segment_row
                for segment_row in candidates
                if segment_row in segment_parameters
                and int(segment_curves[segment_row]) == curve_row
            }
            if not candidates:
                continue
            edge_key = tuple(sorted((first_vertex, second_vertex)))
            adjacent = tuple(
                face_id
                for face_id, face_patch in edge_face_patches.get(edge_key, ())
                if face_patch == patch_row
            )
            if not adjacent:
                continue
            if len(candidates) != 1:
                raise ValueError(
                    "CAD target edge has ambiguous source curve segment ancestry."
                )
            segment_row = next(iter(candidates))
            face_id = min(adjacent)
            try:
                face_witness_row = surface_face_rows[face_id]
                parent_id = int(surface_witness.cell_global_ids[face_witness_row])
                parent_row = cell_rows[parent_id]
            except KeyError as error:
                raise ValueError(
                    "CAD curve edge lacks a selected material-side volume cell."
                ) from error
            cell_vertices = tuple(
                int(value)
                for value in np.asarray(mesh.blocks[0].vertices[parent_row]).tolist()
            )
            local_matches = tuple(
                (local_edge, pair)
                for local_edge, pair in enumerate(local_edges)
                if tuple(sorted((cell_vertices[pair[0]], cell_vertices[pair[1]])))
                == edge_key
            )
            if len(local_matches) != 1:
                raise ValueError(
                    "CAD target curve edge has no unique local tetrahedron edge."
                )
            local_edge, local_pair = local_matches[0]
            first_target, last_target = local_pair
            edge_range_low = min(edge_range)
            edge_range_high = max(edge_range)
            edge_width = Fraction.from_float(edge_range_high) - Fraction.from_float(
                edge_range_low
            )
            if edge_width <= 0:
                raise ValueError(
                    "CAD source curve has a degenerate edge parameter range."
                )
            query_first = Fraction.from_float(query_range[0])
            query_last = Fraction.from_float(query_range[1])

            def chart_parameter(native_parameter: Fraction, /) -> Fraction:
                fraction = (
                    native_parameter - Fraction.from_float(edge_range_low)
                ) / edge_width
                if use.sense < 0:
                    fraction = 1 - fraction
                return query_first + fraction * (query_last - query_first)

            edge_parameters = tuple(
                chart_parameter(value)
                for value in (
                    vertex_parameter_on_segment(cell_vertices[first_target], segment_row),
                    vertex_parameter_on_segment(cell_vertices[last_target], segment_row),
                )
            )
            if any(
                not min(edge_range) <= value <= max(edge_range)
                for value in edge_parameters
            ):
                raise ValueError(
                    "CAD curve witness parameters leave the authored BRep edge interval."
                )
            source_chart_ids.append(chart_id)
            source_entity_ids.append(entity_id)
            occurrences.append(tuple(occurrence))
            source_parameters.append(tuple(float(value) for value in edge_parameters))
            source_parameter_rational_rows.append(
                tuple((value,) for value in edge_parameters)
            )
            target_reference_row = reference_vertices[
                np.asarray(local_pair, dtype=np.int64)
            ]
            target_references.append(target_reference_row)
            target_reference_rational_rows.append(
                tuple(
                    tuple(Fraction.from_float(float(value)) for value in point)
                    for point in target_reference_row
                )
            )
            target_entity_ids.append(int(target_edge_ids[target_edge_row]))
            target_cell_ids.append(parent_id)
            local_edge_indices.append(local_edge)
            orientations.append(1 if edge_parameters[1] > edge_parameters[0] else -1)
    if not target_cell_ids:
        raise ValueError("CAD curve feature has no mapped output boundary edges.")
    return query, SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.query_id,
        chart_dimension=1,
        reference_dimension=3,
        cell_global_ids=np.asarray(target_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(target_entity_ids, dtype=np.int64),
        local_entity_indices=np.asarray(local_edge_indices, dtype=np.int32),
        target_reference_vertices=np.asarray(target_references, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(
            target_reference_rational_rows
        ),
        source_chart_ids=tuple(source_chart_ids),
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrences),
        source_parameters=np.asarray(source_parameters, dtype=np.float64)[:, :, None],
        source_parameter_rational_tokens=rational_chart_tokens(
            source_parameter_rational_rows
        ),
        orientations=np.asarray(orientations, dtype=np.int8),
    )


def _brep_associations(
    source: NativeCadVolumeSource,
    construction: VolumeConstruction,
    physical_points: np.ndarray,
    /,
) -> tuple[GeometryAssociation, ...]:
    from ._native_volume import _source_associations

    mesh = construction.mesh
    native = _source_associations(source.plc_source, construction)
    source_maps = _cad_entity_maps(source)
    vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        )
    }
    vertex_distances = np.zeros((mesh.coordinates.shape[0],), dtype=np.float64)
    boundary_rows = np.unique(np.asarray(construction.boundary_faces, dtype=np.int64))
    if boundary_rows.size:
        distance = source.fidelity_source.boundary_distance(
            np.asarray(physical_points, dtype=np.float64)[boundary_rows]
        )
        upper = np.asarray(distance.upper, dtype=np.float64)
        if upper.shape != boundary_rows.shape:
            raise ValueError(
                "CAD boundary query returned a misaligned vertex distance bank."
            )
        vertex_distances[boundary_rows] = upper

    connectivity_edges, connectivity_faces = simplex_volume_entities(mesh)
    edge_vertices = {
        int(identifier): np.asarray(vertices, dtype=np.int64)
        for identifier, vertices in zip(
            np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64),
            connectivity_edges,
            strict=True,
        )
    }
    face_vertices = {
        int(identifier): np.asarray(vertices, dtype=np.int64)
        for identifier, vertices in zip(
            np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64),
            connectivity_faces,
            strict=True,
        )
    }
    # Host banks indexed per association row below (device arrays would issue
    # one eager gather per row).
    input_vertices = np.asarray(construction.input_vertices, dtype=np.int64)
    vertex_brep_dimensions = np.asarray(source.vertex_brep_dimensions, dtype=np.int64)
    vertex_brep_indices = np.asarray(source.vertex_brep_indices, dtype=np.int64)
    vertex_parameters = np.asarray(source.vertex_parameters, dtype=np.float64)
    result: list[GeometryAssociation] = []
    resolve = _BRepRowResolver(source.domain)
    association_ids: dict[tuple[int, int, tuple[str, ...]], str] = {}
    for association in native:
        target_dimension = next(
            dimension
            for dimension in range(mesh.topological_dimension + 1)
            if mesh.entity_set(dimension).entity_set_id
            == association.target_entity_set_id
        )
        source_dimensions: list[int] = []
        source_indices: list[int] = []
        source_ids: list[str] = []
        occurrence_paths: list[tuple[str, ...]] = []
        source_roles: list[GeometrySourceEntityRole] = []
        parameters = np.full(
            (len(association.source_entity_ids), 2), np.nan, dtype=np.float64
        )
        source_orientations = np.asarray(association.orientations, dtype=np.int8)
        orientations = source_orientations.copy()
        target_ids = np.asarray(association.target_global_ids, dtype=np.int64)
        residuals = np.asarray(association.residuals, dtype=np.float64).copy()
        resolved = np.ones((len(association.source_entity_ids),), dtype=np.bool_)
        region_sides = tuple(association.source_region_sides)
        for row, (identifier, path, role) in enumerate(
            zip(
                association.source_entity_ids,
                association.source_occurrence_paths,
                association.source_roles,
                strict=True,
            )
        ):
            if role is None:
                raise ValueError("Native PLC ancestry must name one source stratum role.")
            plc_dimension = {
                GeometrySourceEntityRole.VERTEX: 0,
                GeometrySourceEntityRole.EDGE: 1,
                GeometrySourceEntityRole.FACET: 2,
                GeometrySourceEntityRole.REGION: 3,
            }[role]
            target_id = int(target_ids[row])
            if target_dimension == 0:
                mesh_vertex = vertex_rows[target_id]
                input_vertex = int(input_vertices[mesh_vertex])
                if input_vertex >= 0:
                    dimension = int(vertex_brep_dimensions[input_vertex])
                    source_index = int(vertex_brep_indices[input_vertex])
                    entity_path = source.plc_source.complex.vertex_occurrence_paths[
                        input_vertex
                    ]
                    if dimension in (1, 2):
                        parameters[row] = vertex_parameters[input_vertex]
                else:
                    dimension, source_index, entity_path = _source_entity_for_plc(
                        source_maps, plc_dimension, identifier, path
                    )
                residuals[row] = vertex_distances[mesh_vertex] if dimension < 3 else 0.0
                resolved[row] = bool(np.isfinite(residuals[row]))
            else:
                dimension, source_index, entity_path = _source_entity_for_plc(
                    source_maps, plc_dimension, identifier, path
                )
                if target_dimension == 1:
                    entity_vertices = edge_vertices[target_id]
                elif target_dimension == 2:
                    entity_vertices = face_vertices[target_id]
                else:
                    entity_vertices = np.empty((0,), dtype=np.int64)
                if dimension < 3 and entity_vertices.size:
                    residuals[row] = float(
                        np.max(vertex_distances[entity_vertices], initial=0.0)
                    )
                elif dimension == 3:
                    residuals[row] = 0.0
                resolved[row] = bool(np.isfinite(residuals[row]))
            source_dimensions.append(dimension)
            source_indices.append(source_index)
            association_key = (dimension, source_index, entity_path)
            association_id = association_ids.get(association_key)
            if association_id is None:
                association_id = association_ids[association_key] = brep_entity_id(
                    source.source_id,
                    source.source_revision,
                    dimension,
                    source_index,
                    occurrence_path=entity_path,
                )
            source_ids.append(association_id)
            occurrence_paths.append(entity_path)
            source_roles.append(
                {
                    0: GeometrySourceEntityRole.VERTEX,
                    1: GeometrySourceEntityRole.EDGE,
                    2: GeometrySourceEntityRole.FACET,
                    3: GeometrySourceEntityRole.REGION,
                }[dimension]
            )
            if target_dimension == 2 and dimension == 2:
                orientations[row] = int(source_orientations[row])
            elif target_dimension == 1 and dimension == 1:
                curve_row, _, _ = resolve(1, source_index, entity_path)
                orientations[row] = int(source_orientations[row]) * (
                    _cad_curve_parameter_sense(source.domain, curve_row)
                )
            else:
                orientations[row] = 0
        for dimension, source_index, path in dict.fromkeys(
            zip(source_dimensions, source_indices, occurrence_paths, strict=True)
        ):
            resolve(dimension, source_index, path)
        if target_dimension != 2:
            region_sides = tuple((None, None) for _ in source_ids)
        result.append(
            GeometryAssociation(
                GeometryAssociationKind.BREP,
                source.source_id,
                source.source_revision,
                association.target_entity_set_id,
                np.asarray(association.target_global_ids, dtype=np.int64),
                tuple(source_ids),
                residuals,
                resolved=resolved,
                source_dimensions=np.asarray(source_dimensions, dtype=np.int8),
                source_indices=np.asarray(source_indices, dtype=np.int32),
                source_roles=tuple(source_roles),
                source_occurrence_paths=tuple(occurrence_paths),
                source_region_sides=region_sides,
                parameters=parameters,
                orientations=orientations,
                exact=False,
            )
        )
    return tuple(result)


def _compliance_for_request(
    original: VolumeMeshingSpec,
    translated: VolumeMeshingSpec,
    report: MeshingComplianceReport,
    /,
) -> MeshingComplianceReport:
    replacements = {
        **{
            translated_control.control_id: original_control.control_id
            for original_control, translated_control in zip(
                original.size_controls, translated.size_controls, strict=True
            )
        },
        **{
            translated_feature.feature_id: original_feature.feature_id
            for original_feature, translated_feature in zip(
                original.protected_features,
                translated.protected_features,
                strict=True,
            )
        },
    }

    def rename(entries: tuple[tuple[str, float], ...]) -> tuple[tuple[str, float], ...]:
        result: list[tuple[str, float]] = []
        for key, value in entries:
            for old, new in replacements.items():
                key = key.replace(old, new)
            result.append((key, value))
        return tuple(result)

    issues = tuple(
        next(
            (key.replace(old, new) for old, new in replacements.items() if old in key),
            key,
        )
        for key in report.issues
    )
    return MeshingComplianceReport(
        original.specification_id,
        issues=issues,
        requested=rename(report.requested),
        achieved=rename(report.achieved),
    )


@final
class PreparedCadVolume(StrictModule, NonTrainableState):
    source: NativeCadVolumeSource
    specification: VolumeMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    schedule: NativeVolumeSchedule
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: NativeCadVolumeSource,
        specification: VolumeMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.schedule = schedule
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "native-cad-volume-preparation",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
            }
        )

    def execute(self) -> NativeMeshConstruction:
        self.source.domain.require_current(
            self.source.source_id, self.source.source_revision
        )
        require_current = getattr(self.source.fidelity_source, "require_current", None)
        if require_current is not None:
            require_current()
        translated = _translate_specification(self.source, self.specification)
        plc_complex = self.source.plc_source.complex
        plc_edge_row_by_key: dict[tuple[str, tuple[str, ...]], int] = {}
        for row, (identifier, path) in enumerate(
            zip(
                plc_complex.plc_edge_ids,
                plc_complex.plc_edge_occurrence_paths,
                strict=True,
            )
        ):
            key = (identifier, tuple(path))
            if key in plc_edge_row_by_key:
                raise ValueError("CAD PLC edge identity/occurrence rows are ambiguous.")
            plc_edge_row_by_key[key] = row
        segment_keys = tuple(
            (identifier, tuple(path))
            for identifier, path in zip(
                plc_complex.segment_ids,
                plc_complex.segment_occurrence_paths,
                strict=True,
            )
        )
        segment_edges = np.asarray(plc_complex.segments, dtype=np.int64)
        plc_edge_vertices = np.asarray(
            plc_complex.plc_edge_vertices, dtype=np.int64
        ).reshape((-1, 2))
        if segment_edges.shape != (len(segment_keys), 2):
            raise ValueError("CAD source segment identity rows are misaligned.")
        try:
            segment_edge_rows = tuple(plc_edge_row_by_key[key] for key in segment_keys)
        except KeyError as error:
            raise ValueError(
                "A CAD curve identity is absent from its canonical PLC source edge bank."
            ) from error
        for row, edge_row in enumerate(segment_edge_rows):
            if tuple(int(value) for value in segment_edges[row]) != tuple(
                int(value) for value in plc_edge_vertices[edge_row]
            ):
                raise ValueError(
                    "CAD curve identity and PLC edge have different oriented endpoint rows."
                )
        # The full canonical bank supports raw identity translation, not blanket
        # 1D promotion. PLC facet rows are patch rows: exact planar patches are
        # declared facet planes and group natively; every other patch keeps its
        # authored triangle edges.
        construction = generate_plc_volume(
            plc_complex,
            translated,
            self.schedule,
            source_id=self.source.source_id,
            source_revision=self.source.source_revision,
            source_map_id="",
            frame_id=self.coordinate_contract.spatial_id,
        )
        physical = physical_plc_tetrahedra(self.source.plc_source, construction)
        native_compliance = _volume_compliance(
            self.source.plc_source,
            translated,
            self.schedule,
            construction,
            physical.points,
        )
        compliance = _compliance_for_request(
            self.specification, translated, native_compliance
        )
        fidelity_tolerance = _cad_source_tolerance(self.source, self.specification)
        organization = _plc_organization(
            self.source.plc_source,
            translated,
            construction,
            source_scope_owner=self.source.domain,
            region_controls=self.specification.region_controls,
        )
        associations = _brep_associations(self.source, construction, physical.points)
        curve_association = _cad_curve_association(self.source, construction)
        if curve_association is not None:
            associations += (curve_association,)
        organization = NativeMeshOrganization(
            boundary=organization.boundary,
            patches=organization.patches,
            zones=organization.zones,
            labels=organization.labels,
            attributes=organization.attributes,
            associations=associations,
            region_boundary_evidence=organization.region_boundary_evidence,
            surface_source=self.source.surface_source,
        )
        fidelity_witness = _cad_volume_chart_witness(
            self.source, construction, physical.geometry, curve_association
        )
        scoped_fidelity = []
        for feature in self.specification.protected_features:
            if not feature.hard:
                continue
            if feature.feature_kind in (
                FeatureKind.SURFACE,
                FeatureKind.MATERIAL_INTERFACE,
            ):
                feature_query, feature_witness = _cad_surface_feature_witness(
                    self.source.domain, feature, fidelity_witness
                )
            elif feature.feature_kind is FeatureKind.CORNER:
                feature_query = self.source.fidelity_source
                feature_witness = _cad_corner_feature_witness(
                    self.source,
                    construction,
                    physical.geometry,
                    feature,
                )
            elif feature.feature_kind is FeatureKind.CURVE:
                feature_query, feature_witness = _cad_curve_feature_witness(
                    self.source,
                    construction,
                    physical.geometry,
                    feature,
                    fidelity_witness,
                )
            else:
                raise ValueError(
                    "CAD feature has no supported source chart witness route."
                )
            scoped_fidelity.append(
                (feature_query, feature_witness, feature.maximum_deviation)
            )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("volume_cad"),
            self.source.source_id,
            self.source.source_revision,
            domain=self.source.domain,
            cell_regions=construction.cell_regions,
            fidelity_source=self.source.fidelity_source,
            fidelity_tolerance=fidelity_tolerance,
            scoped_fidelity=tuple(scoped_fidelity),
            protected_features=self.specification.protected_features,
            fidelity_witness=fidelity_witness,
            feature_source=self.source.domain,
        )
        return NativeMeshConstruction(
            construction.mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=physical.geometry,
            organization=organization,
            stages=construction.stages,
        )


def prepare_native_cad_volume(
    source: NativeCadVolumeSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> PreparedCadVolume:
    if options.route != "cad_tetrahedral":
        raise ValueError(
            "Native CAD volume preparation requires the cad_tetrahedral route."
        )
    issues = cad_volume_support_issues(source, specification)
    if issues:
        raise ValueError("Unsupported native CAD volume request: " + "; ".join(issues))
    schedule = options.volume_schedule
    if schedule is None:
        raise ValueError(
            "Native CAD volume route requires its numerical volume schedule."
        )
    if source.surface.coordinate_contract.spatial_id != coordinate_contract.spatial_id:
        raise ValueError("CAD surface and volume coordinate contracts must agree.")
    return PreparedCadVolume(source, specification, coordinate_contract, schedule)


__all__ = [
    "NativeCadVolumeSource",
    "PreparedCadVolume",
    "cad_volume_support_issues",
    "prepare_native_cad_volume",
]
