from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CellGeometrySpec,
    CellMesh,
    CellValidityCertificate,
    certify_cell_geometry_validity,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import (
    compose_expression,
    evaluate_expression,
    Expression,
)
from ..discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from ..discretization._nested_reference import (
    nested_cell_reference_maps,
    NestedCellReferenceMap,
)
from ..discretization._nested_reference_types import (
    freeze_reference_chart_arguments,
    NestedReferenceWitnesses,
    reference_chart_arguments_identity,
    ReferenceCellMapWitnesses,
    validate_nested_reference_witnesses,
    validate_reference_cell_map_witnesses,
)
from ..discretization.fem import lagrange_element
from ..geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ..geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryProjectionQuery,
    SourceBoundaryQuery,
)
from ..geometry._source_fidelity import certify_source_fidelity, SourceFidelityCertificate
from ..geometry._source_identity import source_entity_id
from ..geometry._source_interpolation import exact_surface_coordinate_expressions
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.implicit._source_charts import (
    AdaptiveImplicitSourceQuery,
    ImplicitRootChartEvidence,
    prepare_adaptive_implicit_root_charts,
)
from ..meshing._boundary_layer import _FACE_ROUTES, BoundaryLayerMesh
from ..meshing._contracts import MeshingFailure, MeshingFailureCategory
from ..meshing._curving import (
    _distortion_blocks,
    _geometry_with_exact_coordinate_bank,
    _sampled_quality,
    _straight_geometry,
    HighOrderCurvingPolicy,
)
from ..meshing._layer_core_topology import LayerCoreConstruction
from ._mixed_adaptation import (
    MixedCoarseningMap,
    _MixedReferenceAffineMap,
    _mixed_coarsening_reference_transforms,
    _parent_to_child_chart,
)
from ._layer_curving_records import (
    _adaptive_implicit_source_query,
    _barycentric_3d,
    _BarycentricSolver3,
    _evaluate_cell_geometry_point,
    _evaluate_mapped_chart_point,
    _exact2,
    _exact3,
    _exact_reference_cell_vertices,
    _face_reference_element,
    _fraction3,
    _fraction_chart2,
    _fraction_chart3,
    _implicit_cell_key,
    _implicit_root_edge_row,
    _implicit_root_witness_row,
    _ImplicitCellKey,
    _ImplicitRootEdgeRow,
    _mapped_chart_reference_contexts,
    _mapped_physical_distance_bounds,
    _mapped_reference_chart_rows,
    _NodalReferenceElement,
    _prepare_barycentric_solver_3d,
    _reference_expression,
    _reference_face_dofs,
    _validate_parametric_chart_geometry,
    LayerColumnProjectionIdentity,
    LayerCurvingImplicitRootTrace,
    LayerCurvingMappedChartTrace,
    LayerCurvingProjectionTrace,
    LayerCurvingSourceEvidence,
)
from ._lineage import EntityLineageKind, MeshLineage


class LayerCurvingStatus(StrEnum):
    CURVED = "curved"
    ROLLED_BACK_INVALID = "rolled_back_invalid"
    ROLLED_BACK_EMBEDDING = "rolled_back_embedding"
    ROLLED_BACK_SOURCE_FIDELITY = "rolled_back_source_fidelity"
    ROLLED_BACK_SOURCE_PROJECTION = "rolled_back_source_projection"
    UNRESOLVED_SOURCE_MAP = "unresolved_source_map"


class LayerCurvingConstructionResult(StrictModule, NonTrainableState):
    """Source-aware high-order candidate and its all-or-nothing acceptance."""

    status: LayerCurvingStatus = eqx.field(static=True)
    construction: LayerCoreConstruction
    candidate_geometry: CellGeometrySpec | None
    validity: CellValidityCertificate | None
    embedding: GlobalEmbeddingCertificate | None
    source_evidence: LayerCurvingSourceEvidence | None
    projection_trace: LayerCurvingProjectionTrace | None
    implicit_root_trace: LayerCurvingImplicitRootTrace | None
    mapped_chart_trace: LayerCurvingMappedChartTrace | None
    implicit_root_chart_witness: SourceBoundaryChartWitness | None
    implicit_map_witness_id: str | None = eqx.field(static=True)
    implicit_chart_evidence: tuple[ImplicitRootChartEvidence, ...] = eqx.field(static=True)
    chart_witness: SourceBoundaryChartWitness | None
    source_fidelity: SourceFidelityCertificate | None
    failure: str | None = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: LayerCurvingStatus,
        construction: LayerCoreConstruction,
        candidate_geometry: CellGeometrySpec | None,
        validity: CellValidityCertificate | None,
        embedding: GlobalEmbeddingCertificate | None,
        source_evidence: LayerCurvingSourceEvidence | None,
        failure: str | None,
        /,
        *,
        projection_trace: LayerCurvingProjectionTrace | None = None,
        implicit_root_trace: LayerCurvingImplicitRootTrace | None = None,
        mapped_chart_trace: LayerCurvingMappedChartTrace | None = None,
        implicit_root_chart_witness: SourceBoundaryChartWitness | None = None,
        implicit_map_witness_id: str | None = None,
        implicit_chart_evidence: tuple[ImplicitRootChartEvidence, ...] = (),
        chart_witness: SourceBoundaryChartWitness | None = None,
        source_fidelity: SourceFidelityCertificate | None = None,
    ) -> None:
        if not isinstance(status, LayerCurvingStatus):
            raise TypeError("status must be LayerCurvingStatus.")
        implicit_evidence = tuple(implicit_chart_evidence)
        if status is LayerCurvingStatus.CURVED:
            if (
                candidate_geometry is None
                or validity is None
                or not validity.all_certified
                or embedding is None
                or embedding.status != "certified"
                or source_evidence is None
                or (
                    source_evidence is not None
                    and (
                        source_evidence.projection_trace is not projection_trace
                        or source_evidence.implicit_root_trace is not implicit_root_trace
                        or source_evidence.mapped_chart_trace is not mapped_chart_trace
                        or source_evidence.chart_witness is not chart_witness
                        or source_evidence.fidelity is not source_fidelity
                        or source_evidence.implicit_root_chart_witness
                        is not implicit_root_chart_witness
                        or source_evidence.implicit_map_witness_id
                        != implicit_map_witness_id
                        or source_evidence.implicit_chart_evidence != implicit_evidence
                    )
                )
                or construction.geometry.geometry_layout_id != candidate_geometry.geometry_layout_id
                or (
                    source_evidence is not None
                    and source_evidence.source_surface_charts is None
                    and sum(
                        trace is not None
                        for trace in (
                            projection_trace,
                            implicit_root_trace,
                            mapped_chart_trace,
                        )
                    )
                    != 1
                )
                or (
                    source_evidence is not None
                    and source_evidence.source_surface_charts is not None
                    and any(
                        trace is not None
                        for trace in (
                            projection_trace,
                            implicit_root_trace,
                            mapped_chart_trace,
                        )
                    )
                )
                or (projection_trace is not None and not projection_trace.resolved)
                or (implicit_root_trace is not None and not implicit_root_trace.resolved)
                or (mapped_chart_trace is not None and not mapped_chart_trace.resolved)
                or (implicit_root_trace is not None and implicit_root_chart_witness is None)
                or chart_witness is None
                or source_fidelity is None
                or source_fidelity.status != "certified"
            ):
                raise ValueError("A curved layer construction requires current full-map evidence.")
            if implicit_root_trace is not None and (
                implicit_map_witness_id != implicit_root_trace.map_witness_id
                or implicit_evidence != implicit_root_trace.chart_evidence
            ):
                raise ValueError("Accepted implicit layer root-map evidence is incomplete.")
        elif failure is None:
            raise ValueError("A rolled-back layer-curving result must retain its failure.")
        self.status = status
        self.construction = construction
        self.candidate_geometry = candidate_geometry
        self.validity = validity
        self.embedding = embedding
        self.source_evidence = source_evidence
        self.projection_trace = projection_trace
        self.implicit_root_trace = implicit_root_trace
        self.mapped_chart_trace = mapped_chart_trace
        self.implicit_root_chart_witness = implicit_root_chart_witness
        self.implicit_map_witness_id = implicit_map_witness_id
        self.implicit_chart_evidence = implicit_evidence
        self.chart_witness = chart_witness
        self.source_fidelity = source_fidelity
        self.failure = failure
        self.result_id = canonical_fingerprint(
            {
                "kind": "layer-curving-construction-result",
                "status": status.value,
                "construction": construction.mesh.mesh_id,
                "geometry": None if candidate_geometry is None else candidate_geometry.geometry_layout_id,
                "validity": None if validity is None else validity.certificate_id,
                "embedding": None if embedding is None else embedding.certificate_id,
                "source_evidence": None if source_evidence is None else source_evidence.evidence_id,
                "failure": failure,
                "projection_trace": (
                    None if projection_trace is None else projection_trace.trace_id
                ),
                "implicit_root_trace": (
                    None if implicit_root_trace is None else implicit_root_trace.trace_id
                ),
                "mapped_chart_trace": (
                    None if mapped_chart_trace is None else mapped_chart_trace.trace_id
                ),
                "implicit_map": implicit_map_witness_id,
                "implicit_root_chart_witness": (
                    None
                    if implicit_root_chart_witness is None
                    else implicit_root_chart_witness.witness_id
                ),
                "implicit_chart_evidence": tuple(
                    (
                        row.source_cell_key,
                        row.root_edge_keys,
                        row.status,
                        row.field_residual_bound,
                        row.target_to_root_lower_bound,
                        row.target_to_root_bound,
                        row.source_normal_deviation_bounds,
                        row.chart_jacobian_lower,
                        row.chart_injectivity_margin,
                        row.source_gradient_lower,
                    )
                    for row in implicit_evidence
                ),
                "chart_witness": None if chart_witness is None else chart_witness.witness_id,
                "source_fidelity": (
                    None if source_fidelity is None else source_fidelity.certificate_id
                ),
            }
        )


class LayerCurvingFailure(MeshingFailure):
    """Domain failure retaining the rejected candidate and its full source authority."""

    def __init__(self, result: LayerCurvingConstructionResult, /) -> None:
        if not isinstance(result, LayerCurvingConstructionResult):
            raise TypeError("result must be LayerCurvingConstructionResult.")
        category = {
            LayerCurvingStatus.ROLLED_BACK_INVALID: MeshingFailureCategory.AUDIT_FAILED,
            LayerCurvingStatus.ROLLED_BACK_EMBEDDING: MeshingFailureCategory.AUDIT_FAILED,
            LayerCurvingStatus.ROLLED_BACK_SOURCE_FIDELITY: MeshingFailureCategory.QUALITY_REJECTED,
            LayerCurvingStatus.ROLLED_BACK_SOURCE_PROJECTION: MeshingFailureCategory.INVALID_SOURCE,
            LayerCurvingStatus.UNRESOLVED_SOURCE_MAP: MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
        }[result.status]
        trace = result.projection_trace or result.implicit_root_trace or result.mapped_chart_trace
        if trace is not None:
            cells = tuple(sorted(set(int(value) for value in trace.target_cell_global_ids)))
        elif result.chart_witness is not None:
            cells = tuple(sorted(set(int(value) for value in result.chart_witness.cell_global_ids)))
        else:
            cells = ()
        detail = result.failure or result.status.value
        super().__init__(
            category,
            f"Native source-aware layer curving failed ({result.status.value}): {detail}",
            provider_code="native_layer_curving",
            stage="layer_curving",
            entity_ids=cells,
        )
        self.result = result
        self.projection_trace = result.projection_trace
        self.implicit_root_trace = result.implicit_root_trace
        self.mapped_chart_trace = result.mapped_chart_trace


def require_layer_core_construction(
    result: LayerCurvingConstructionResult, /
) -> LayerCoreConstruction:
    """Return only a fully accepted curved construction; retain rejection evidence on failure."""
    if not isinstance(result, LayerCurvingConstructionResult):
        raise TypeError("result must be LayerCurvingConstructionResult.")
    if result.status is not LayerCurvingStatus.CURVED:
        raise LayerCurvingFailure(result)
    return result.construction





@dataclass(frozen=True, slots=True)
class _SourceFace:
    chart_row: int
    source_cell_id: int
    source_patch: int
    source_entity_id: str
    occurrence_path: tuple[str, ...]
    source_root_cell_id: int
    source_piece_id: tuple[int, ...]
    target_block_row: int
    target_cell_row: int
    target_cell_id: int
    target_local_face: int
    target_face_global_id: int
    target_vertex_rows: tuple[int, int, int]
    target_reference_corners: tuple[tuple[Fraction, Fraction, Fraction], ...]
    source_parameter_corners: tuple[tuple[Fraction, Fraction], ...]
    layer_index: int = -1
    layer_column_id: int = -1
    layer_front_row_identity: str = ""
    source_column_ids: tuple[int, ...] = ()
    source_column_identities: tuple[str, ...] = ()
    source_column_root_vertex_ids: tuple[int, ...] = ()
    implicit_chart_id: str = ""
    implicit_cell_key: tuple[int, ...] = ()
    implicit_root_edge_keys: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    implicit_root_cell_vertex_global_ids: tuple[int, ...] = ()


def _canonical_source_faces(faces: tuple[_SourceFace, ...], /) -> tuple[_SourceFace, ...]:
    """Order layer ancestry by the exact source-witness canonical row identity."""
    target_parameters = tuple(face.target_reference_corners for face in faces)
    source_parameters = tuple(face.source_parameter_corners for face in faces)
    target_tokens = rational_chart_tokens(target_parameters)
    source_tokens = rational_chart_tokens(source_parameters)
    order = sorted(
        range(len(faces)),
        key=lambda row: (
            faces[row].target_cell_id,
            faces[row].target_local_face,
            faces[row].target_face_global_id,
            faces[row].implicit_chart_id or faces[row].source_entity_id,
            target_tokens[row],
            tuple(
                float(value)
                for point in target_parameters[row]
                for value in point
            ),
            (),
            tuple(
                float(value)
                for point in source_parameters[row]
                for value in point
            ),
            source_tokens[row],
            faces[row].source_entity_id,
            faces[row].occurrence_path,
        ),
    )
    return tuple(faces[row] for row in order)

@dataclass(frozen=True, slots=True)
class _MappedLayerNodeRecord:
    geometry_node_row: int
    target_cell_id: int
    local_dof: int
    chart_row: int
    target_reference: tuple[ExactScalar, ExactScalar, ExactScalar]
    root_reference: tuple[ExactScalar, ExactScalar, ExactScalar]
    source_parameter: tuple[ExactScalar, ExactScalar]
    target_point: NDArray[np.float64]
    tangents: NDArray[np.float64]
    normal: NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class _MappedTransitionNodeRecord:
    geometry_node_row: int
    target_cell_id: int
    local_dof: int
    chart_row: int
    target_reference: tuple[ExactScalar, ExactScalar, ExactScalar]
    root_reference: tuple[ExactScalar, ExactScalar, ExactScalar]
    source_parameter: tuple[ExactScalar, ExactScalar]
    target_point: NDArray[np.float64]
    source_point: NDArray[np.float64]
    tangents: NDArray[np.float64]
    normal: NDArray[np.float64]
    distance_lower: float
    distance_upper: float
    root_face_global_id: int
    root_cell_global_id: int
    root_local_face_index: int

def _int3(values: ArrayLike | Sequence[int], /) -> tuple[int, int, int]:
    exact = np.asarray(values, dtype=np.int64)
    if exact.shape != (3,):
        raise ValueError("A triangular source face must retain three vertex identities.")
    return (int(exact[0]), int(exact[1]), int(exact[2]))
def _face_key(vertex_ids: np.ndarray, rows: tuple[int, ...], /) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in vertex_ids[np.asarray(rows, dtype=np.int64)]))


def _face_table(
    mesh: CellMesh,
) -> tuple[dict[tuple[int, ...], list[tuple[int, int, int, tuple[int, ...]]]], dict[tuple[int, ...], int]]:
    from ..meshing._curving import _entity_key_table

    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    target: dict[tuple[int, ...], list[tuple[int, int, int, tuple[int, ...]]]] = {}
    for block_row, block in enumerate(mesh.blocks):
        routes = _FACE_ROUTES.get(block.cell_kind)
        if routes is None:
            raise ValueError(f"Layer curving does not support {block.cell_kind!r} cells.")
        cells = np.asarray(block.vertices, dtype=np.int64).reshape((block.cell_count, -1))
        for cell_row in range(block.cell_count):
            cell = cells[cell_row]
            for local_face, local_vertices in enumerate(routes):
                if len(local_vertices) != 3:
                    continue
                rows = tuple(int(cell[index]) for index in local_vertices)
                key = _face_key(vertex_ids, rows)
                target.setdefault(key, []).append((block_row, cell_row, local_face, rows))
    face_entities = _entity_key_table(mesh, 2)
    entity_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_ids = {key: int(entity_ids[row]) for key, row in face_entities.items()}
    return target, face_ids


def _source_faces(
    construction: LayerCoreConstruction,
    layers: BoundaryLayerMesh,
    charts: SurfaceSourceCharts,
    /,
) -> tuple[_SourceFace, ...]:
    wall = layers.source_wall
    charts.require_root(wall, charts.geometry)
    charts.require_bound(wall, charts.geometry, charts.boundary_source)
    if (construction.source_id, construction.source_revision) != (
        charts.source_id,
        charts.source_revision,
    ):
        raise ValueError("Layer construction and exact source charts bind different revisions.")
    layer_count = int(layers.mesh.coordinates.shape[0])
    if (
        construction.mesh.coordinates.shape[0] < layer_count
        or not np.array_equal(
            np.asarray(construction.mesh.vertex_global_ids, dtype=np.int64)[:layer_count],
            np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64),
        )
        or not np.array_equal(
            np.asarray(construction.mesh.coordinates, dtype=np.float64)[:layer_count].view(np.uint64),
            np.asarray(layers.mesh.coordinates, dtype=np.float64).view(np.uint64),
        )
    ):
        raise ValueError("Layer/core composition changed exact layer vertex identity or coordinates.")
    wall_map = np.asarray(layers.wall_vertices, dtype=np.int64)
    if wall_map.shape != (wall.coordinates.shape[0],):
        raise ValueError("Boundary-layer root vertex map does not bind its source wall.")
    source_vertex_to_layer_id = {
        int(source_row): int(np.asarray(layers.mesh.vertex_global_ids)[int(layer_row)])
        for source_row, layer_row in enumerate(wall_map)
        if layer_row >= 0
    }
    source_cell_rows: dict[int, tuple[int, tuple[int, int, int]]] = {}
    for block in wall.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Exact source chart curving currently requires triangular root charts.")
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            source_cell_rows[int(identifier)] = (
                row,
                _int3(np.asarray(block.vertices[row], dtype=np.int64)),
            )
    target_faces, target_face_ids = _face_table(construction.mesh)
    layer_cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in layers.mesh.blocks)
    )
    if (
        not np.array_equal(layer_cell_ids, np.sort(layer_cell_ids))
        or np.asarray(layers.layer_index).shape != layer_cell_ids.shape
        or np.asarray(layers.column_index).shape != layer_cell_ids.shape
    ):
        raise ValueError("Layer cell, schedule, and front-column identities are not aligned.")
    layer_row_by_cell_id = {
        int(identifier): row for row, identifier in enumerate(layer_cell_ids)
    }
    rows_by_cell_id = {
        int(identifier): row
        for row, identifier in enumerate(charts.cell_global_ids)
    }
    query_patches = set(int(value) for value in charts.boundary_source.patches)
    layer_values = np.asarray(layers.layer_index, dtype=np.int64)
    front_row_values = np.asarray(layers.column_index, dtype=np.int64)
    front_columns = np.asarray(layers.front_row_column_ids, dtype=np.int64)
    column_root_vertices = np.asarray(layers.column_vertex_sources, dtype=np.int64)
    result: list[_SourceFace] = []
    for chart_row, cell_id in enumerate(charts.cell_global_ids):
        cell_identifier = int(cell_id)
        patch = int(charts.root_patches[chart_row])
        if patch not in query_patches:
            continue
        if cell_identifier not in source_cell_rows:
            raise ValueError("Exact source chart references a missing source-wall cell.")
        _, source_vertices = source_cell_rows[cell_identifier]
        try:
            mapped_ids = tuple(source_vertex_to_layer_id[value] for value in source_vertices)
        except KeyError as error:
            raise ValueError("Source chart facet lost one of its accepted layer vertices.") from error
        key = tuple(sorted(mapped_ids))
        candidates = target_faces.get(key, ())
        if len(candidates) != 1:
            raise ValueError("A source chart does not bind one unique exterior layer/core facet.")
        block_row, cell_row, local_face, _ = candidates[0]
        target_block = construction.mesh.blocks[block_row]
        target_cell_id = int(np.asarray(target_block.global_ids, dtype=np.int64)[cell_row])
        layer_row = layer_row_by_cell_id.get(target_cell_id)
        if layer_row is None:
            raise ValueError("A source-bound exterior facet is not owned by an accepted layer cell.")
        front_row_id = int(front_row_values[layer_row])
        if front_row_id < 0 or front_row_id >= len(layers.front_row_identity):
            raise ValueError("Layer cell front-row ancestry names an absent source row.")
        source_columns = tuple(
            int(value) for value in front_columns[front_row_id] if int(value) >= 0
        )
        source_column_identities = tuple(
            layers.column_vertex_identity[value] for value in source_columns
        )
        source_column_roots = tuple(
            int(column_root_vertices[value]) for value in source_columns
        )
        target_ids = np.asarray(construction.mesh.vertex_global_ids, dtype=np.int64)
        source_row_by_layer_vertex_id = {
            int(layer_vertex_id): int(source_row)
            for source_row, layer_vertex_id in source_vertex_to_layer_id.items()
        }
        source_corner_by_row = {
            source_row: corner for corner, source_row in enumerate(source_vertices)
        }
        exact_parameters = _fraction_chart2(
            charts.root_parameter_chart(rows_by_cell_id[cell_identifier])
        )
        local_vertices = _FACE_ROUTES[target_block.cell_kind][local_face]
        reference_vertices = _exact_reference_cell_vertices(target_block.cell_kind)
        target_reference = _fraction_chart3(
            tuple(reference_vertices[index] for index in local_vertices)
        )
        target_rows_ordered = _int3(
            tuple(
                int(np.asarray(target_block.vertices, dtype=np.int64)[cell_row, index])
                for index in local_vertices
            )
        )
        source_parameters: list[tuple[Fraction, Fraction]] = []
        for target_row in target_rows_ordered:
            vertex_id = int(target_ids[target_row])
            try:
                source_row = source_row_by_layer_vertex_id[vertex_id]
                corner = source_corner_by_row[source_row]
            except KeyError as error:
                raise ValueError("Layer boundary face changed source-wall vertex ancestry.") from error
            source_parameters.append(exact_parameters[corner])
        source_entity_id = charts.domain.entity_id(2, patch)
        occurrence = tuple(charts.domain.source_occurrences[2][patch])
        uv = _fraction_chart2(source_parameters)
        area = (
            (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1])
            - (uv[1][1] - uv[0][1]) * (uv[2][0] - uv[0][0])
        )
        if area == 0:
            raise ValueError("Source chart degenerates on the realized layer boundary face.")
        result.append(
            _SourceFace(
                chart_row,
                cell_identifier,
                patch,
                source_entity_id,
                occurrence,
                int(np.asarray(charts.source_root_cell_ids)[chart_row]),
                tuple(charts.source_piece_ids[chart_row]),
                block_row,
                cell_row,
                target_cell_id,
                local_face,
                target_face_ids[key],
                target_rows_ordered,
                target_reference,
                uv,
                int(layer_values[layer_row]),
                front_row_id,
                layers.front_row_identity[front_row_id],
                source_columns,
                source_column_identities,
                source_column_roots,
            )
        )
    if not result:
        raise ValueError("Source query has no charted boundary facets in the layer/core mesh.")
    return _canonical_source_faces(tuple(result))
def _implicit_query_scope_id(source_query: AdaptiveImplicitSourceQuery, /) -> str:
    scope_id = getattr(source_query, "boundary_source_revision", source_query.source_revision)
    if not isinstance(scope_id, str) or not scope_id:
        raise ValueError("Adaptive implicit source query has no stable source-scope identity.")
    return scope_id
def _implicit_source_faces(
    mesh: CellMesh,
    layers: BoundaryLayerMesh,
    source_query: AdaptiveImplicitSourceQuery,
    root_witness: SourceBoundaryChartWitness,
    /,
) -> tuple[_SourceFace, ...]:
    """Transfer adaptive root charts through source-cell and exact vertex ancestry."""
    wall = layers.source_wall
    implicit_entity_id = source_entity_id(
        source_query.source_id,
        source_query.source_revision,
        "implicit-surface",
        "zero-set",
    )
    if (
        not isinstance(root_witness, SourceBoundaryChartWitness)
        or root_witness.mesh_id != wall.mesh_id
        or root_witness.topology_id != wall.topology_id
        or root_witness.chart_dimension != 2
        or root_witness.reference_dimension != 2
        or root_witness.source_id != source_query.source_id
        or root_witness.source_revision != source_query.source_revision
        or root_witness.source_query_id != _implicit_query_scope_id(source_query)
    ):
        raise ValueError("Adaptive root witness is not bound to the current source-wall mesh and query.")
    if (
        root_witness.implicit_cell_keys.shape != (len(root_witness.source_chart_ids), 4)
        or root_witness.implicit_root_edge_keys.shape
        != (len(root_witness.source_chart_ids), 3, 2, 3)
        or root_witness.source_entity_ids
        != tuple(implicit_entity_id for _ in root_witness.source_chart_ids)
        or root_witness.occurrence_paths != tuple(() for _ in root_witness.source_chart_ids)
    ):
        raise ValueError("Adaptive root witness lost exact cell, edge, or source identity.")
    standard_chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    root_parameters = tuple(
        _fraction_chart2(root_witness.source_parameter_chart(row))
        for row in range(len(root_witness.source_chart_ids))
    )
    if root_parameters != tuple(standard_chart for _ in root_parameters):
        raise ValueError("Adaptive implicit parameters must retain the canonical standard root triangle.")
    layer_count = int(layers.mesh.coordinates.shape[0])
    if (
        mesh.coordinates.shape[0] < layer_count
        or not np.array_equal(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64)[:layer_count],
            np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64),
        )
        or not np.array_equal(
            np.asarray(mesh.coordinates, dtype=np.float64)[:layer_count].view(np.uint64),
            np.asarray(layers.mesh.coordinates, dtype=np.float64).view(np.uint64),
        )
    ):
        raise ValueError("Layer/core composition changed exact layer vertex identity or coordinates.")
    wall_map = np.asarray(layers.wall_vertices, dtype=np.int64)
    if wall_map.shape != (wall.coordinates.shape[0],) or np.any(wall_map < 0):
        raise ValueError("Adaptive source-wall triangles lost accepted layer vertex ancestry.")
    source_cells: dict[int, tuple[int, int, int]] = {}
    for block in wall.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Adaptive implicit layer roots require triangular source-wall cells.")
        for row, cell_id in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            source_cells[int(cell_id)] = _int3(block.vertices[row])
    root_cell_ids = tuple(int(value) for value in root_witness.cell_global_ids)
    if (
        len(root_cell_ids) != len(source_cells)
        or len(set(root_cell_ids)) != len(root_cell_ids)
        or set(root_cell_ids) != set(source_cells)
        or not np.array_equal(root_witness.entity_global_ids, root_witness.cell_global_ids)
        or np.any(root_witness.local_entity_indices != -1)
    ):
        raise ValueError("Adaptive root witness does not cover each source-wall triangle exactly once.")
    target_faces, target_face_ids = _face_table(mesh)
    layer_cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in layers.mesh.blocks)
    )
    if (
        not np.array_equal(layer_cell_ids, np.sort(layer_cell_ids))
        or np.asarray(layers.layer_index).shape != layer_cell_ids.shape
        or np.asarray(layers.column_index).shape != layer_cell_ids.shape
    ):
        raise ValueError("Layer cell, schedule, and front-column identities are not aligned.")
    layer_row_by_cell_id = {
        int(identifier): row for row, identifier in enumerate(layer_cell_ids)
    }
    layer_values = np.asarray(layers.layer_index, dtype=np.int64)
    front_rows = np.asarray(layers.column_index, dtype=np.int64)
    front_columns = np.asarray(layers.front_row_column_ids, dtype=np.int64)
    column_roots = np.asarray(layers.column_vertex_sources, dtype=np.int64)
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    result: list[_SourceFace] = []
    for chart_row, source_cell_id in enumerate(root_cell_ids):
        source_vertices = source_cells[source_cell_id]
        target_vertex_rows = _int3(
            tuple(int(wall_map[row]) for row in source_vertices)
        )
        key = tuple(sorted(int(target_ids[row]) for row in target_vertex_rows))
        candidates = target_faces.get(key, ())
        if len(candidates) != 1:
            raise ValueError("An adaptive root chart does not bind one unique exterior layer/core facet.")
        block_row, cell_row, local_face, _ = candidates[0]
        block = mesh.blocks[block_row]
        target_cell_id = int(np.asarray(block.global_ids, dtype=np.int64)[cell_row])
        layer_row = layer_row_by_cell_id.get(target_cell_id)
        if layer_row is None:
            raise ValueError("An adaptive source-bound facet is not owned by an accepted layer cell.")
        front_row = int(front_rows[layer_row])
        if front_row < 0 or front_row >= len(layers.front_row_identity):
            raise ValueError("Layer cell front-row ancestry names an absent source row.")
        source_columns = tuple(int(value) for value in front_columns[front_row] if int(value) >= 0)
        source_column_names = tuple(layers.column_vertex_identity[value] for value in source_columns)
        source_column_roots = tuple(int(column_roots[value]) for value in source_columns)
        local_vertices = _FACE_ROUTES[block.cell_kind][local_face]
        reference_vertices = _exact_reference_cell_vertices(block.cell_kind)
        cell_vertices = np.asarray(block.vertices, dtype=np.int64)[cell_row]
        reference_by_source_row: dict[int, tuple[Fraction, Fraction, Fraction]] = {}
        for source_row, target_row in zip(source_vertices, target_vertex_rows, strict=True):
            matches = [
                int(local_index)
                for local_index in local_vertices
                if int(cell_vertices[int(local_index)]) == target_row
            ]
            if len(matches) != 1:
                raise ValueError("Adaptive root chart vertex has no unique exact target-local reference.")
            reference_by_source_row[source_row] = reference_vertices[matches[0]]
        target_reference = _fraction_chart3(
            tuple(reference_by_source_row[row] for row in source_vertices)
        )
        root_key = tuple(
            int(value) for value in root_witness.implicit_cell_keys[chart_row]
        )
        root_edges = _implicit_root_edge_row(
            root_witness.implicit_root_edge_keys[chart_row]
        )
        result.append(
            _SourceFace(
                chart_row=chart_row,
                source_cell_id=source_cell_id,
                source_patch=-1,
                source_entity_id=implicit_entity_id,
                occurrence_path=(),
                source_root_cell_id=source_cell_id,
                source_piece_id=root_key,
                target_block_row=block_row,
                target_cell_row=cell_row,
                target_cell_id=target_cell_id,
                target_local_face=local_face,
                target_face_global_id=target_face_ids[key],
                target_vertex_rows=target_vertex_rows,
                target_reference_corners=target_reference,
                source_parameter_corners=root_parameters[chart_row],
                layer_index=int(layer_values[layer_row]),
                layer_column_id=front_row,
                layer_front_row_identity=layers.front_row_identity[front_row],
                source_column_ids=source_columns,
                source_column_identities=source_column_names,
                source_column_root_vertex_ids=source_column_roots,
                implicit_chart_id=root_witness.source_chart_ids[chart_row],
                implicit_cell_key=root_key,
                implicit_root_edge_keys=root_edges,
                implicit_root_cell_vertex_global_ids=_int3(
                    tuple(int(wall.vertex_global_ids[row]) for row in source_vertices)
                ),
            )
        )
    if not result:
        raise ValueError("Adaptive implicit query has no charted source-wall facets.")
    return _canonical_source_faces(tuple(result))


def _implicit_chart_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source_query: AdaptiveImplicitSourceQuery,
    faces: tuple[_SourceFace, ...],
    /,
) -> SourceBoundaryChartWitness:
    target_parameters = tuple(face.target_reference_corners for face in faces)
    source_parameters = tuple(face.source_parameter_corners for face in faces)
    implicit_entity_id = source_entity_id(
        source_query.source_id,
        source_query.source_revision,
        "implicit-surface",
        "zero-set",
    )

    def build(orientations: np.ndarray) -> SourceBoundaryChartWitness:
        return SourceBoundaryChartWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            source_id=source_query.source_id,
            source_revision=source_query.source_revision,
            source_query_id=_implicit_query_scope_id(source_query),
            chart_dimension=2,
            reference_dimension=3,
            cell_global_ids=np.asarray([face.target_cell_id for face in faces], dtype=np.int64),
            entity_global_ids=np.asarray(
                [face.target_face_global_id for face in faces], dtype=np.int64
            ),
            local_entity_indices=np.asarray(
                [face.target_local_face for face in faces], dtype=np.int32
            ),
            target_reference_vertices=np.asarray(
                [[[float(value) for value in point] for point in row] for row in target_parameters],
                dtype=np.float64,
            ),
            target_reference_rational_tokens=rational_chart_tokens(target_parameters),
            source_chart_ids=tuple(face.implicit_chart_id for face in faces),
            source_entity_ids=tuple(implicit_entity_id for _ in faces),
            occurrence_paths=tuple(() for _ in faces),
            source_parameters=np.asarray(
                [[[float(value) for value in point] for point in row] for row in source_parameters],
                dtype=np.float64,
            ),
            source_parameter_rational_tokens=rational_chart_tokens(source_parameters),
            orientations=orientations,
            implicit_cell_keys=np.asarray(
                [face.implicit_cell_key for face in faces], dtype=np.int64
            ),
            implicit_root_edge_keys=np.asarray(
                [face.implicit_root_edge_keys for face in faces], dtype=np.int64
            ),
        )

    provisional = build(np.ones((len(faces),), dtype=np.int8))
    from ..geometry._source_fidelity import _surface_target_chart_rows

    _, _, orientations = _surface_target_chart_rows(
        mesh, geometry, None, provisional, full_source=False
    )
    return build(np.asarray(orientations, dtype=np.int8))


def _source_parameter(
    point: tuple[ExactScalar, ExactScalar, ExactScalar],
    face: _SourceFace,
    solver: _BarycentricSolver3,
    /,
) -> tuple[ExactScalar, ExactScalar]:
    weights = _barycentric_3d(
        point, face.target_reference_corners, solver=solver
    )
    return (
        sum(
            (
                weights[index] * face.source_parameter_corners[index][0]
                for index in range(3)
            ),
            Fraction(0),
        ),
        sum(
            (
                weights[index] * face.source_parameter_corners[index][1]
                for index in range(3)
            ),
            Fraction(0),
        ),
    )


def _column_identity_for_faces(
    faces: tuple[_SourceFace, ...],
    /,
) -> LayerColumnProjectionIdentity:
    if not faces:
        raise ValueError("Layer source-chart ancestry cannot be empty.")
    return LayerColumnProjectionIdentity(
        np.asarray([face.layer_index for face in faces], dtype=np.int64),
        np.asarray([face.source_cell_id for face in faces], dtype=np.int64),
        np.asarray([face.layer_column_id for face in faces], dtype=np.int64),
        tuple(face.layer_front_row_identity for face in faces),
        tuple(face.source_column_ids for face in faces),
        tuple(face.source_column_identities for face in faces),
        tuple(face.source_column_root_vertex_ids for face in faces),
    )
def _chart_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    charts: SurfaceSourceCharts,
    faces: tuple[_SourceFace, ...],
    /,
) -> SourceBoundaryChartWitness:
    from ..geometry._source_boundary_records import rational_chart_tokens

    source_parameters = tuple(face.source_parameter_corners for face in faces)
    target_parameters = tuple(face.target_reference_corners for face in faces)
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=charts.source_id,
        source_revision=charts.source_revision,
        source_query_id=charts.boundary_source.source_scope_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray([face.target_cell_id for face in faces], dtype=np.int64),
        entity_global_ids=np.asarray([face.target_face_global_id for face in faces], dtype=np.int64),
        local_entity_indices=np.asarray([face.target_local_face for face in faces], dtype=np.int32),
        target_reference_vertices=np.asarray(
            [[[float(value) for value in point] for point in row] for row in target_parameters],
            dtype=np.float64,
        ),
        target_reference_rational_tokens=rational_chart_tokens(target_parameters),
        source_chart_ids=tuple(face.source_entity_id for face in faces),
        source_entity_ids=tuple(face.source_entity_id for face in faces),
        occurrence_paths=tuple(face.occurrence_path for face in faces),
        source_parameters=np.asarray(
            [[[float(value) for value in point] for point in row] for row in source_parameters],
            dtype=np.float64,
        ),
        source_parameter_rational_tokens=rational_chart_tokens(source_parameters),
        orientations=np.asarray(
            [
                1
                if (
                    (face.source_parameter_corners[1][0] - face.source_parameter_corners[0][0])
                    * (face.source_parameter_corners[2][1] - face.source_parameter_corners[0][1])
                    - (face.source_parameter_corners[1][1] - face.source_parameter_corners[0][1])
                    * (face.source_parameter_corners[2][0] - face.source_parameter_corners[0][0])
                ) > 0
                else -1
                for face in faces
            ],
            dtype=np.int8,
        ),
    )




def _construction_with_geometry(
    construction: LayerCoreConstruction,
    geometry: CellGeometrySpec,
    /,
) -> LayerCoreConstruction:
    return LayerCoreConstruction(
        construction.mesh,
        construction.domain,
        construction.cell_regions,
        construction.zones,
        construction.patches,
        construction.labels,
        construction.attributes,
        construction.associations,
        geometry,
        construction.core,
        construction.core_vertex_map,
        construction.core_cell_ids,
        construction.region_ids,
        construction.source_id,
        construction.source_revision,
        construction.periodic_topology,
        construction.periodic_requested,
        construction.periodic_achieved,
        construction.reference_maps,
        construction.source_mesh,
        construction.source_geometry,
        construction.reference_mesh,
        construction.reference_geometry,
        construction.polyhedral_core,
    )


def _rollback(
    status: LayerCurvingStatus,
    construction: LayerCoreConstruction,
    failure: str,
    /,
    *,
    candidate: CellGeometrySpec | None = None,
    validity: CellValidityCertificate | None = None,
    embedding: GlobalEmbeddingCertificate | None = None,
    evidence: LayerCurvingSourceEvidence | None = None,
    projection_trace: LayerCurvingProjectionTrace | None = None,
    implicit_root_trace: LayerCurvingImplicitRootTrace | None = None,
    mapped_chart_trace: LayerCurvingMappedChartTrace | None = None,
    implicit_root_chart_witness: SourceBoundaryChartWitness | None = None,
    implicit_map_witness_id: str | None = None,
    implicit_chart_evidence: tuple[ImplicitRootChartEvidence, ...] = (),
    chart_witness: SourceBoundaryChartWitness | None = None,
    source_fidelity: SourceFidelityCertificate | None = None,
) -> LayerCurvingConstructionResult:
    return LayerCurvingConstructionResult(
        status,
        construction,
        candidate,
        validity,
        embedding,
        evidence,
        failure,
        projection_trace=projection_trace,
        mapped_chart_trace=mapped_chart_trace,
        implicit_root_trace=implicit_root_trace,
        implicit_root_chart_witness=implicit_root_chart_witness,
        implicit_map_witness_id=implicit_map_witness_id,
        implicit_chart_evidence=implicit_chart_evidence,
        chart_witness=chart_witness,
        source_fidelity=source_fidelity,
    )


def _curve_adaptive_implicit_layer_core(
    construction: LayerCoreConstruction,
    layers: BoundaryLayerMesh,
    source_query: AdaptiveImplicitSourceQuery,
    root_witness: SourceBoundaryChartWitness,
    policy: HighOrderCurvingPolicy,
    frame_id: str,
    fidelity_tolerance: float,
    maximum_normal_deviation: float | None,
    /,
) -> LayerCurvingConstructionResult:
    """Create and continuously certify mixed-cell maps over adaptive implicit root charts."""
    try:
        if (construction.source_id, construction.source_revision) != (
            source_query.source_id,
            source_query.source_revision,
        ):
            raise ValueError("Layer construction and adaptive root query bind different source revisions.")
        faces = _implicit_source_faces(
            construction.mesh, layers, source_query, root_witness
        )
        straight = _straight_geometry(construction.mesh, policy.degree)
        witness = _implicit_chart_witness(construction.mesh, straight, source_query, faces)
        root_map = prepare_adaptive_implicit_root_charts(
            construction.mesh,
            straight,
            source_query,
            witness,
            frame_id=frame_id,
            tolerance=None,
            maximum_normal_deviation=None,
        )
    except (ArithmeticError, IndexError, TypeError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
            construction,
            f"Adaptive implicit target-to-root map could not be prepared: {error}",
            implicit_root_chart_witness=root_witness,
        )
    root_evidence = tuple(root_map.chart_evidence)
    if root_map.status != "certified" or root_map.witness_id is None:
        details = tuple(row.detail for row in root_evidence if row.status != "certified")
        return _rollback(
            LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
            construction,
            "; ".join(details) or "Adaptive implicit target-to-root chart is unresolved.",
            candidate=straight,
            implicit_root_chart_witness=root_witness,
            implicit_map_witness_id=root_map.witness_id,
            implicit_chart_evidence=root_evidence,
            chart_witness=witness,
        )

    root_map_id = root_map.witness_id
    candidate: CellGeometrySpec | None = None
    witness: SourceBoundaryChartWitness | None = None
    final_map_id = root_map_id
    final_chart_evidence = root_evidence
    try:
        elements, routes, coordinates = straight.resolve(construction.mesh)
        candidate_coordinates = np.array(coordinates, dtype=np.float64, copy=True)
        assigned: dict[int, np.ndarray] = {}
        node_records: list[
            tuple[int, int, int, int, _SourceFace, tuple[ExactScalar, ExactScalar, ExactScalar]]
        ] = []
        for face in faces:
            element = _face_reference_element(
                elements[face.target_block_row],
                construction.mesh.blocks[face.target_block_row].cell_kind,
            )
            route = np.asarray(routes[face.target_block_row], dtype=np.int64)
            face_nodes = _reference_face_dofs(
                element, face.target_reference_corners
            )
            for local_dof, reference in face_nodes:
                point = root_map.project_target(face.chart_row, reference)
                geometry_row = int(route[face.target_cell_row, local_dof])
                if geometry_row < 0:
                    raise ValueError("An implicit source-bound reference node has no map row.")
                previous = assigned.get(geometry_row)
                if previous is not None and not np.array_equal(
                    previous.view(np.uint64), point.point.view(np.uint64)
                ):
                    raise ValueError(
                        "Adjacent implicit root charts disagree on a shared geometry node."
                    )
                assigned[geometry_row] = np.array(point.point, copy=True)
                if geometry_row < construction.mesh.coordinates.shape[0]:
                    physical_vertex = np.asarray(
                        construction.mesh.coordinates, dtype=np.float64
                    )[geometry_row]
                    if not np.array_equal(physical_vertex.view(np.uint64), point.point.view(np.uint64)):
                        raise ValueError(
                            "Implicit root chart changes an immutable layer/core mesh vertex."
                        )
                else:
                    candidate_coordinates[geometry_row] = point.point
                node_records.append(
                    (
                        geometry_row,
                        face.target_cell_id,
                        local_dof,
                        face.chart_row,
                        face,
                        reference,
                    )
                )
        if not node_records:
            raise ValueError("Adaptive implicit charts constrain no high-order layer boundary DOFs.")
        layout = (
            {block.name: element for block, element in zip(construction.mesh.blocks, elements, strict=True)},
            {block.name: route for block, route in zip(construction.mesh.blocks, routes, strict=True)},
        )
        candidate = _geometry_with_exact_coordinate_bank(
            construction.mesh,
            layout[0],
            layout[1],
            straight,
            candidate_coordinates,
            identity_context=(
                "adaptive-implicit-layer",
                frame_id,
                source_query.source_id,
                source_query.source_revision,
                root_map_id,
            ),
        )
        witness = _implicit_chart_witness(construction.mesh, candidate, source_query, faces)
        final_map = prepare_adaptive_implicit_root_charts(
            construction.mesh,
            candidate,
            source_query,
            witness,
            frame_id=frame_id,
            tolerance=fidelity_tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
        )
        final_chart_evidence = tuple(final_map.chart_evidence)
        if final_map.status != "certified" or final_map.witness_id is None:
            return _rollback(
                LayerCurvingStatus.ROLLED_BACK_SOURCE_FIDELITY,
                construction,
                "Adaptive implicit root-map regularity or continuous tolerance failed after full-map construction.",
                candidate=candidate,
                implicit_root_chart_witness=root_witness,
                implicit_map_witness_id=final_map.witness_id,
                implicit_chart_evidence=final_chart_evidence,
                chart_witness=witness,
            )
        final_map_id = final_map.witness_id
        geometry_rows: list[int] = []
        target_cells: list[int] = []
        local_dofs: list[int] = []
        chart_rows: list[int] = []
        reference_points: list[tuple[ExactScalar, ExactScalar, ExactScalar]] = []
        target_points: list[np.ndarray] = []
        source_points: list[np.ndarray] = []
        displacement_upper: list[float] = []
        root_error: list[float] = []
        cell_keys: list[_ImplicitCellKey] = []
        root_edges: list[_ImplicitRootEdgeRow] = []
        statuses: list[str] = []
        layer_indices: list[int] = []
        source_wall_cell_ids: list[int] = []
        front_row_ids: list[int] = []
        front_row_identities: list[str] = []
        source_column_ids: list[tuple[int, ...]] = []
        source_column_identities: list[tuple[str, ...]] = []
        source_column_root_vertex_ids: list[tuple[int, ...]] = []
        for geometry_row, cell_id, local_dof, chart_row, face, reference in node_records:
            point = final_map.project_target(chart_row, reference)
            geometry_rows.append(geometry_row)
            target_cells.append(cell_id)
            local_dofs.append(local_dof)
            chart_rows.append(chart_row)
            reference_points.append(reference)
            target_points.append(np.asarray(candidate_coordinates[geometry_row], dtype=np.float64))
            source_points.append(np.asarray(point.point, dtype=np.float64))
            displacement_upper.append(point.displacement_upper)
            root_error.append(point.root_error)
            cell_keys.append(_implicit_cell_key(point.source_cell_key))
            root_edges.append(_implicit_root_edge_row(point.root_edge_keys))
            statuses.append(point.status)
            layer_indices.append(face.layer_index)
            source_wall_cell_ids.append(face.source_cell_id)
            front_row_ids.append(face.layer_column_id)
            front_row_identities.append(face.layer_front_row_identity)
            source_column_ids.append(face.source_column_ids)
            source_column_identities.append(face.source_column_identities)
            source_column_root_vertex_ids.append(face.source_column_root_vertex_ids)
        column_identity = LayerColumnProjectionIdentity(
            np.asarray(layer_indices, dtype=np.int64),
            np.asarray(source_wall_cell_ids, dtype=np.int64),
            np.asarray(front_row_ids, dtype=np.int64),
            tuple(front_row_identities),
            tuple(source_column_ids),
            tuple(source_column_identities),
            tuple(source_column_root_vertex_ids),
        )
        root_trace = LayerCurvingImplicitRootTrace(
            source_query.source_id,
            source_query.source_revision,
            frame_id,
            final_map_id,
            witness.witness_id,
            np.asarray(geometry_rows, dtype=np.int64),
            np.asarray(target_cells, dtype=np.int64),
            np.asarray(local_dofs, dtype=np.int32),
            np.asarray(chart_rows, dtype=np.int64),
            tuple(reference_points),
            np.asarray(target_points, dtype=np.float64),
            np.asarray(source_points, dtype=np.float64),
            np.asarray(displacement_upper, dtype=np.float64),
            np.asarray(root_error, dtype=np.float64),
            tuple(cell_keys),
            tuple(root_edges),
            tuple(statuses),
            final_chart_evidence,
            column_identity,
        )
    except (ArithmeticError, IndexError, TypeError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_PROJECTION,
            construction,
            f"Adaptive implicit root projection failed: {error}",
            implicit_root_chart_witness=root_witness,
            candidate=candidate,
            implicit_map_witness_id=final_map_id,
            implicit_chart_evidence=final_chart_evidence,
            chart_witness=witness,
        )

    try:
        validity = certify_cell_geometry_validity(candidate, mesh=construction.mesh, policy=policy.validity)
    except (ArithmeticError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_INVALID,
            construction,
            f"Whole-cell validity certification failed: {error}",
            candidate=candidate,
            implicit_root_chart_witness=root_witness,
            implicit_root_trace=root_trace,
            implicit_map_witness_id=final_map_id,
            implicit_chart_evidence=final_chart_evidence,
            chart_witness=witness,
        )
    if not validity.all_certified:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_INVALID,
            construction,
            "Whole-cell Bernstein validity did not certify every layer/core cell.",
            implicit_root_chart_witness=root_witness,
            candidate=candidate,
            validity=validity,
            implicit_root_trace=root_trace,
            implicit_map_witness_id=final_map_id,
            implicit_chart_evidence=final_chart_evidence,
            chart_witness=witness,
        )
    try:
        embedding = certify_global_embedding(construction.mesh, candidate, validity)
    except (ArithmeticError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_EMBEDDING,
            construction,
            f"Global embedding certification failed: {error}",
            implicit_root_chart_witness=root_witness,
            candidate=candidate,
            validity=validity,
            implicit_root_trace=root_trace,
            implicit_map_witness_id=final_map_id,
            implicit_chart_evidence=final_chart_evidence,
            chart_witness=witness,
        )
    if embedding.status != "certified":
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_EMBEDDING,
            construction,
            "Global embedding evidence did not certify the curved layer/core map.",
            implicit_root_chart_witness=root_witness,
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            implicit_root_trace=root_trace,
            implicit_map_witness_id=final_map_id,
            implicit_chart_evidence=final_chart_evidence,
            chart_witness=witness,
        )
    if not np.array_equal(
        np.asarray(construction.mesh.coordinates, dtype=np.float64).view(np.uint64),
        np.asarray(candidate.coordinates, dtype=np.float64)[
            : construction.mesh.coordinates.shape[0]
        ].view(np.uint64),
    ):
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_PROJECTION,
            construction,
            "High-order curving changed layer/core mesh vertices or physical column heights.",
            implicit_root_chart_witness=root_witness,
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            implicit_root_trace=root_trace,
            implicit_map_witness_id=final_map_id,
            implicit_chart_evidence=final_chart_evidence,
            chart_witness=witness,
        )
    try:
        fidelity = certify_source_fidelity(
            construction.mesh,
            candidate,
            source_query,
            tolerance=fidelity_tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
            witness=witness,
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_FIDELITY,
            construction,
            f"Continuous adaptive implicit source-fidelity certification failed: {error}",
            implicit_root_chart_witness=root_witness,
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            implicit_root_trace=root_trace,
            implicit_map_witness_id=final_map_id,
            implicit_chart_evidence=final_chart_evidence,
            chart_witness=witness,
        )
    if fidelity.status != "certified":
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_FIDELITY,
            construction,
            "Continuous adaptive implicit source-fidelity evidence did not certify the map.",
            implicit_root_chart_witness=root_witness,
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            implicit_root_trace=root_trace,
            implicit_map_witness_id=final_map_id,
            implicit_chart_evidence=final_chart_evidence,
            chart_witness=witness,
            source_fidelity=fidelity,
        )
    try:
        blocks = _distortion_blocks(construction.mesh, candidate, policy.degree)
        minimum_scaled, maximum_distortion = _sampled_quality(
            blocks, np.asarray(candidate.coordinates, dtype=np.float64)
        )
        evidence = LayerCurvingSourceEvidence(
            None,
            source_query,
            layers.result_id,
            layers.evidence,
            witness,
            fidelity,
            minimum_scaled,
            maximum_distortion,
            implicit_root_trace=root_trace,
            implicit_map_witness_id=final_map_id,
            implicit_chart_evidence=final_chart_evidence,
            implicit_root_chart_witness=root_witness,
            implicit_root_cell_vertex_global_ids=np.asarray(
                [face.implicit_root_cell_vertex_global_ids for face in faces],
                dtype=np.int64,
            ),
            chart_column_identity=_column_identity_for_faces(faces),
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_INVALID,
            construction,
            f"Adaptive implicit source evidence construction failed: {error}",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            implicit_root_trace=root_trace,
            implicit_map_witness_id=final_map_id,
            implicit_chart_evidence=final_chart_evidence,
            chart_witness=witness,
            source_fidelity=fidelity,
        )
    return LayerCurvingConstructionResult(
        LayerCurvingStatus.CURVED,
        _construction_with_geometry(construction, candidate),
        candidate,
        validity,
        embedding,
        evidence,
        None,
        implicit_root_trace=root_trace,
        implicit_root_chart_witness=root_witness,
        implicit_map_witness_id=final_map_id,
        implicit_chart_evidence=final_chart_evidence,
        chart_witness=witness,
        source_fidelity=fidelity,
    )

def _mapped_layer_column_identity(
    construction: LayerCoreConstruction,
    layers: BoundaryLayerMesh,
    witness: SourceBoundaryChartWitness,
    /,
) -> LayerColumnProjectionIdentity:
    wall = layers.source_wall
    wall_map = np.asarray(layers.wall_vertices, dtype=np.int64)
    mesh = construction.mesh
    target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if wall_map.shape != np.asarray(wall.vertex_global_ids).shape:
        raise ValueError("Mapped layer source-wall vertex ancestry is incomplete.")
    target_faces, target_face_ids = _face_table(mesh)
    layer_cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in layers.mesh.blocks)
    )
    layer_rows = {int(identifier): row for row, identifier in enumerate(layer_cell_ids)}
    layer_indices = np.asarray(layers.layer_index, dtype=np.int64)
    front_rows = np.asarray(layers.column_index, dtype=np.int64)
    front_columns = np.asarray(layers.front_row_column_ids, dtype=np.int64)
    column_sources = np.asarray(layers.column_vertex_sources, dtype=np.int64)
    provenance: dict[int, tuple[int, int, int, str, tuple[int, ...], tuple[str, ...], tuple[int, ...]]] = {}
    for block in wall.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Mapped layer source-wall ancestry requires triangular wall facets.")
        for source_row, source_cell_id in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            source_vertices = np.asarray(block.vertices[source_row], dtype=np.int64)
            target_rows = wall_map[source_vertices]
            if np.any(target_rows < 0) or np.any(target_rows >= target_vertex_ids.size):
                raise ValueError("Mapped layer source-wall facet lost its output vertices.")
            key = tuple(sorted(int(value) for value in target_vertex_ids[target_rows]))
            candidates = target_faces.get(key, ())
            if len(candidates) != 1:
                raise ValueError("Mapped source-wall facet does not bind one exact output boundary face.")
            block_row, cell_row, _, _ = candidates[0]
            target_cell_id = int(np.asarray(mesh.blocks[block_row].global_ids)[cell_row])
            layer_row = layer_rows.get(target_cell_id)
            if layer_row is None:
                raise ValueError("Mapped source-wall facet is not owned by a retained layer cell.")
            front_row = int(front_rows[layer_row])
            if front_row < 0 or front_row >= front_columns.shape[0]:
                raise ValueError("Mapped layer cell names an absent source front row.")
            columns = tuple(int(value) for value in front_columns[front_row] if int(value) >= 0)
            if not columns:
                raise ValueError("Mapped layer front row lost its source columns.")
            face_id = target_face_ids[key]
            values = (
                int(source_cell_id),
                int(layer_indices[layer_row]),
                front_row,
                layers.front_row_identity[front_row],
                columns,
                tuple(layers.column_vertex_identity[value] for value in columns),
                tuple(int(column_sources[value]) for value in columns),
            )
            if face_id in provenance:
                raise ValueError("Mapped source-wall ancestry repeats an output boundary face.")
            provenance[face_id] = values
    records = []
    for face_id in np.asarray(witness.entity_global_ids, dtype=np.int64):
        try:
            records.append(provenance[int(face_id)])
        except KeyError as error:
            raise ValueError("Mapped source chart lost its exact layer/source-column ancestry.") from error
    return LayerColumnProjectionIdentity(
        np.asarray([row[1] for row in records], dtype=np.int64),
        np.asarray([row[0] for row in records], dtype=np.int64),
        np.asarray([row[2] for row in records], dtype=np.int64),
        tuple(row[3] for row in records),
        tuple(row[4] for row in records),
        tuple(row[5] for row in records),
        tuple(row[6] for row in records),
    )


def _rebind_chart_witness_geometry(
    witness: SourceBoundaryChartWitness,
    geometry: CellGeometrySpec,
    /,
) -> SourceBoundaryChartWitness:
    return SourceBoundaryChartWitness(
        mesh_id=witness.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=witness.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        chart_dimension=witness.chart_dimension,
        reference_dimension=witness.reference_dimension,
        cell_global_ids=witness.cell_global_ids,
        entity_global_ids=witness.entity_global_ids,
        local_entity_indices=witness.local_entity_indices,
        target_reference_vertices=witness.target_reference_vertices,
        target_reference_rational_tokens=witness.target_reference_rational_tokens,
        target_face_vertex_global_ids=witness.target_face_vertex_global_ids,
        source_chart_ids=witness.source_chart_ids,
        source_entity_ids=witness.source_entity_ids,
        occurrence_paths=witness.occurrence_paths,
        source_parameters=witness.source_parameters,
        source_parameter_rational_tokens=witness.source_parameter_rational_tokens,
        orientations=witness.orientations,
        implicit_cell_keys=(
            witness.implicit_cell_keys
            if witness.implicit_cell_keys.shape[0]
            else None
        ),
        implicit_root_edge_keys=(
            witness.implicit_root_edge_keys
            if witness.implicit_root_edge_keys.shape[0]
            else None
        ),
    )


def _curve_mapped_layer_core(
    construction: LayerCoreConstruction,
    layers: BoundaryLayerMesh,
    query: MappedReferenceBoundarySource,
    witness: SourceBoundaryChartWitness,
    policy: HighOrderCurvingPolicy,
    frame_id: str,
    tolerance: float,
    maximum_normal_deviation: float | None,
    /,
) -> LayerCurvingConstructionResult:
    trace: LayerCurvingMappedChartTrace | None = None
    candidate: CellGeometrySpec | None = None
    validity: CellValidityCertificate | None = None
    embedding: GlobalEmbeddingCertificate | None = None
    fidelity: SourceFidelityCertificate | None = None
    try:
        if construction.reference_mesh is None or construction.reference_geometry is None:
            raise ValueError("Mapped layer construction lost its exact root reference geometry.")
        if (
            construction.source_id != query.source_id
            or construction.source_revision != query.source_revision
            or frame_id != query.coordinate_contract.spatial_id
            or witness.mesh_id != construction.mesh.mesh_id
            or witness.geometry_id != cell_geometry_id(construction.geometry)
            or witness.source_query_id != query.source_query_id
        ):
            raise ValueError("Mapped layer chart witness changes source, frame, or construction identity.")
        query.require_current()
        mapped_rows = _mapped_reference_chart_rows(
            query, witness, construction.reference_maps
        )
        chart_identity = _mapped_layer_column_identity(construction, layers, witness)
        straight = _straight_geometry(construction.mesh, policy.degree)
        elements, routes, coordinates = straight.resolve(construction.mesh)
        candidate_coordinates = np.array(coordinates, dtype=np.float64, copy=True)
        block_rows_by_cell = {
            int(identifier): (block_row, cell_row)
            for block_row, block in enumerate(construction.mesh.blocks)
            for cell_row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
        }
        assigned: dict[int, np.ndarray] = {}
        records: list[_MappedLayerNodeRecord] = []
        covered: dict[tuple[int, int], set[int]] = {}
        face_dofs: dict[tuple[int, int], set[int]] = {}
        for chart_row, (_, _, _, _, mapping) in enumerate(mapped_rows):
            cell_id = int(witness.cell_global_ids[chart_row])
            block_row, cell_row = block_rows_by_cell[cell_id]
            element = _face_reference_element(
                elements[block_row], construction.mesh.blocks[block_row].cell_kind
            )
            local_face = int(witness.local_entity_indices[chart_row])
            face_key = (cell_id, local_face)
            route = np.asarray(routes[block_row], dtype=np.int64)
            target_corners = _fraction_chart3(
                witness.target_reference_chart(chart_row)
            )
            face_nodes = _reference_face_dofs(element, target_corners)
            dofs = tuple(local_dof for local_dof, _ in face_nodes)
            face_dofs.setdefault(face_key, set()).update(dofs)
            covered.setdefault(face_key, set())
            contexts = _mapped_chart_reference_contexts(
                query,
                witness,
                chart_row,
                mapping,
                tuple(reference for _, reference in face_nodes),
            )
            for (local_dof, reference), (
                root_reference,
                source_parameter,
                source_chart_uv,
                source_chart_solver,
            ) in zip(face_nodes, contexts, strict=True):
                covered[face_key].add(local_dof)
                root_reference, source_parameter, point, tangents, normal = (
                    _evaluate_mapped_chart_point(
                        query,
                        witness,
                        chart_row,
                        reference,
                        mapping,
                        root_reference,
                        source_parameter,
                        source_chart_uv,
                        source_chart_solver,
                    )
                )
                geometry_row = int(route[cell_row, local_dof])
                if geometry_row < 0:
                    raise ValueError("Mapped high-order source chart has no coordinate-map node.")
                previous = assigned.get(geometry_row)
                if previous is not None and not np.array_equal(previous.view(np.uint64), point.view(np.uint64)):
                    raise ValueError("Adjacent mapped source charts disagree at a shared high-order node.")
                assigned[geometry_row] = np.array(point, copy=True)
                if geometry_row < construction.mesh.coordinates.shape[0]:
                    physical_vertex = np.asarray(construction.mesh.coordinates, dtype=np.float64)[geometry_row]
                    if not np.array_equal(physical_vertex.view(np.uint64), point.view(np.uint64)):
                        raise ValueError("Mapped source root changes an immutable layer/core physical vertex.")
                else:
                    candidate_coordinates[geometry_row] = point
                records.append(
                    _MappedLayerNodeRecord(
                        geometry_row,
                        cell_id,
                        local_dof,
                        chart_row,
                        reference,
                        root_reference,
                        source_parameter,
                        point,
                        tangents,
                        normal,
                    )
                )
        if not records:
            raise ValueError("Mapped source charts constrain no mixed-cell high-order layer DOFs.")
        for face_key, expected_dofs in face_dofs.items():
            if covered.get(face_key, set()) != expected_dofs:
                raise ValueError("Mapped source chart partition omits a target boundary reference DOF.")
        layout = (
            {block.name: element for block, element in zip(construction.mesh.blocks, elements, strict=True)},
            {block.name: route for block, route in zip(construction.mesh.blocks, routes, strict=True)},
        )
        candidate = _geometry_with_exact_coordinate_bank(
            construction.mesh,
            layout[0],
            layout[1],
            straight,
            candidate_coordinates,
            identity_context=(
                "mapped-reference-layer",
                query.source_query_id,
                witness.witness_id,
            ),
        )
        target_witness = _rebind_chart_witness_geometry(witness, candidate)
        reference_maps = tuple(construction.reference_maps)
        trace_chart_rows = np.asarray(
            [record.chart_row for record in records], dtype=np.int64
        )
        trace_columns = LayerColumnProjectionIdentity(
            np.asarray(
                [chart_identity.layer_indices[row] for row in trace_chart_rows],
                dtype=np.int64,
            ),
            np.asarray(
                [chart_identity.source_wall_cell_ids[row] for row in trace_chart_rows],
                dtype=np.int64,
            ),
            np.asarray(
                [chart_identity.front_row_ids[row] for row in trace_chart_rows],
                dtype=np.int64,
            ),
            tuple(chart_identity.front_row_identities[row] for row in trace_chart_rows),
            tuple(chart_identity.source_column_ids[row] for row in trace_chart_rows),
            tuple(chart_identity.source_column_identities[row] for row in trace_chart_rows),
            tuple(chart_identity.source_column_root_vertex_ids[row] for row in trace_chart_rows),
        )
        physical_distance_bounds = tuple(
            _mapped_physical_distance_bounds(
                query,
                witness.source_chart_ids[record.chart_row],
                record.source_parameter,
                record.target_point,
            )
            for record in records
        )
        trace = LayerCurvingMappedChartTrace(
            query.source_id,
            query.source_revision,
            frame_id,
            query.source_query_id,
            np.asarray([record.geometry_node_row for record in records], dtype=np.int64),
            np.asarray([record.target_cell_id for record in records], dtype=np.int64),
            np.asarray([record.local_dof for record in records], dtype=np.int32),
            trace_chart_rows,
            tuple(record.target_reference for record in records),
            tuple(record.root_reference for record in records),
            tuple(record.source_parameter for record in records),
            tuple(witness.source_chart_ids[record.chart_row] for record in records),
            tuple(witness.source_entity_ids[record.chart_row] for record in records),
            tuple(witness.occurrence_paths[record.chart_row] for record in records),
            np.asarray(
                [mapped_rows[record.chart_row][1] for record in records],
                dtype=np.int64,
            ),
            np.asarray(
                [mapped_rows[record.chart_row][2] for record in records],
                dtype=np.int64,
            ),
            np.asarray(
                [mapped_rows[record.chart_row][3] for record in records],
                dtype=np.int32,
            ),
            np.asarray([record.target_point for record in records], dtype=np.float64),
            np.asarray([record.target_point for record in records], dtype=np.float64),
            np.asarray([bounds[0] for bounds in physical_distance_bounds], dtype=np.float64),
            np.asarray([bounds[1] for bounds in physical_distance_bounds], dtype=np.float64),
            np.asarray([record.normal for record in records], dtype=np.float64),
            np.asarray([record.tangents for record in records], dtype=np.float64),
            trace_columns,
        )
    except (ArithmeticError, IndexError, TypeError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
            construction,
            str(error),
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            mapped_chart_trace=trace,
            chart_witness=witness if candidate is not None else None,
            source_fidelity=fidelity,
        )
    assert candidate is not None and trace is not None
    if not trace.resolved:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_FIDELITY,
            construction,
            "Mapped exact root-chart lift produced unresolved source evidence.",
            candidate=candidate,
            mapped_chart_trace=trace,
            chart_witness=target_witness,
        )
    try:
        validity = certify_cell_geometry_validity(
            candidate, mesh=construction.mesh, policy=policy.validity
        )
    except (ArithmeticError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_INVALID,
            construction,
            f"Whole-cell mapped layer validity certification failed: {error}",
            candidate=candidate,
            mapped_chart_trace=trace,
            chart_witness=target_witness,
        )
    if not validity.all_certified:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_INVALID,
            construction,
            "Whole-cell Bernstein validity did not certify every mapped layer/core cell.",
            candidate=candidate,
            validity=validity,
            mapped_chart_trace=trace,
            chart_witness=target_witness,
        )
    try:
        embedding = certify_global_embedding(construction.mesh, candidate, validity)
    except (ArithmeticError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_EMBEDDING,
            construction,
            f"Mapped layer global embedding certification failed: {error}",
            candidate=candidate,
            validity=validity,
            mapped_chart_trace=trace,
            chart_witness=target_witness,
        )
    if embedding.status != "certified":
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_EMBEDDING,
            construction,
            "Global embedding evidence did not certify the mapped layer/core map.",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            mapped_chart_trace=trace,
            chart_witness=target_witness,
        )
    if not np.array_equal(
        np.asarray(construction.mesh.coordinates, dtype=np.float64).view(np.uint64),
        np.asarray(candidate.coordinates, dtype=np.float64)[
            : construction.mesh.coordinates.shape[0]
        ].view(np.uint64),
    ):
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_PROJECTION,
            construction,
            "Mapped high-order curving changed immutable layer/core vertices.",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            mapped_chart_trace=trace,
            chart_witness=target_witness,
        )
    try:
        fidelity = certify_source_fidelity(
            construction.mesh,
            candidate,
            query,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
            witness=target_witness,
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_FIDELITY,
            construction,
            f"Continuous mapped source-fidelity certification failed: {error}",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            mapped_chart_trace=trace,
            chart_witness=target_witness,
        )
    if fidelity.status != "certified":
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_FIDELITY,
            construction,
            "Continuous mapped source-fidelity evidence did not meet declared physical bounds.",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            mapped_chart_trace=trace,
            chart_witness=target_witness,
            source_fidelity=fidelity,
        )
    try:
        blocks = _distortion_blocks(construction.mesh, candidate, policy.degree)
        minimum_scaled, maximum_distortion = _sampled_quality(
            blocks, np.asarray(candidate.coordinates, dtype=np.float64)
        )
        evidence = LayerCurvingSourceEvidence(
            None,
            query,
            layers.result_id,
            layers.evidence,
            target_witness,
            fidelity,
            minimum_scaled,
            maximum_distortion,
            mapped_chart_trace=trace,
            mapped_reference_maps=reference_maps,
            chart_column_identity=chart_identity,
            frame_id=frame_id,
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_INVALID,
            construction,
            f"Mapped layer source-evidence construction failed: {error}",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            mapped_chart_trace=trace,
            chart_witness=target_witness,
            source_fidelity=fidelity,
        )
    return LayerCurvingConstructionResult(
        LayerCurvingStatus.CURVED,
        _construction_with_geometry(construction, candidate),
        candidate,
        validity,
        embedding,
        evidence,
        None,
        mapped_chart_trace=trace,
        chart_witness=target_witness,
        source_fidelity=fidelity,
    )


def curve_layer_core_construction(
    construction: LayerCoreConstruction,
    layers: BoundaryLayerMesh,
    source_surface_charts: SurfaceSourceCharts | None,
    /,
    *,
    policy: HighOrderCurvingPolicy,
    fidelity_tolerance: float | None = None,
    maximum_normal_deviation: float | None = None,
    fidelity_source: SourceBoundaryQuery | None = None,
    implicit_root_witness: SourceBoundaryChartWitness | None = None,
    mapped_chart_witness: SourceBoundaryChartWitness | None = None,
    frame_id: str | None = None,
) -> LayerCurvingConstructionResult:
    """Build and certify a full mixed-cell high-order layer/core coordinate map.

    Parametric sources evaluate exact retained root charts at every source-wall
    face DOF and retain source/column ancestry without point-projection traces.
    Adaptive implicit sources retain exact root-cell/edge identity and certified root
    lifts. Mapped-reference sources evaluate their exact original root-cell map and
    chart derivatives.
    Every route certifies the complete mixed-cell map, source fidelity, validity,
    and global embedding without treating mapped sources as projection queries.
    """
    if not isinstance(construction, LayerCoreConstruction):
        raise TypeError("construction must be LayerCoreConstruction.")
    if not isinstance(layers, BoundaryLayerMesh):
        raise TypeError("layers must be BoundaryLayerMesh.")
    if not isinstance(policy, HighOrderCurvingPolicy):
        raise TypeError("policy must be HighOrderCurvingPolicy.")
    if policy.relaxation_rounds != 0:
        raise ValueError("Exact source-bound layer maps do not admit relaxation rounds.")
    if policy.degree < 2:
        raise ValueError("Layer curving requires geometry degree two or higher.")
    if fidelity_tolerance is not None and (
        not np.isfinite(float(fidelity_tolerance)) or float(fidelity_tolerance) < 0.0
    ):
        raise ValueError("fidelity_tolerance must be None or finite and nonnegative.")
    if maximum_normal_deviation is not None and (
        not np.isfinite(float(maximum_normal_deviation))
        or not 0.0 <= float(maximum_normal_deviation) <= np.pi
    ):
        raise ValueError("maximum_normal_deviation must be None or radians in [0, pi].")
    if source_surface_charts is None and isinstance(
        fidelity_source, MappedReferenceBoundarySource
    ):
        if mapped_chart_witness is None or implicit_root_witness is not None:
            return _rollback(
                LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
                construction,
                "Mapped high-order layer curving requires its exact target-local source chart witness.",
                chart_witness=mapped_chart_witness,
            )
        selected_frame = frame_id or fidelity_source.coordinate_contract.spatial_id
        if selected_frame != fidelity_source.coordinate_contract.spatial_id:
            return _rollback(
                LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
                construction,
                "Mapped layer source and exact target map use different physical frames.",
                chart_witness=mapped_chart_witness,
            )
        selected_tolerance = (
            policy.residual_tolerance
            if fidelity_tolerance is None
            else float(fidelity_tolerance)
        )
        if selected_tolerance <= 0.0:
            return _rollback(
                LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
                construction,
                "Mapped layer fidelity requires a positive physical tolerance.",
                chart_witness=mapped_chart_witness,
            )
        return _curve_mapped_layer_core(
            construction,
            layers,
            fidelity_source,
            mapped_chart_witness,
            policy,
            selected_frame,
            selected_tolerance,
            maximum_normal_deviation,
        )
    if source_surface_charts is None:
        if mapped_chart_witness is not None:
            return _rollback(
                LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
                construction,
                "Mapped source chart witness requires MappedReferenceBoundarySource authority.",
                chart_witness=mapped_chart_witness,
            )
        if fidelity_source is None or implicit_root_witness is None:
            return _rollback(
                LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
                construction,
                "High-order curving has no exact source-reference map and continuous source-fidelity witness.",
                implicit_root_chart_witness=implicit_root_witness,
            )
        if not isinstance(fidelity_source, SourceBoundaryProjectionQuery):
            raise TypeError("fidelity_source must implement SourceBoundaryProjectionQuery.")
        try:
            adaptive_source = _adaptive_implicit_source_query(fidelity_source)
        except TypeError as error:
            return _rollback(
                LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
                construction,
                str(error),
                implicit_root_chart_witness=implicit_root_witness,
            )
        selected_frame = frame_id or adaptive_source.frame_id
        if not selected_frame or selected_frame != adaptive_source.frame_id:
            return _rollback(
                LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
                construction,
                "Adaptive implicit curving requires the exact source-bound spatial frame.",
                implicit_root_chart_witness=implicit_root_witness,
            )
        selected_tolerance = (
            policy.residual_tolerance
            if fidelity_tolerance is None
            else float(fidelity_tolerance)
        )
        if selected_tolerance <= 0.0:
            return _rollback(
                LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
                construction,
                "Adaptive implicit fidelity requires a positive physical tolerance.",
                implicit_root_chart_witness=implicit_root_witness,
            )
        return _curve_adaptive_implicit_layer_core(
            construction,
            layers,
            adaptive_source,
            implicit_root_witness,
            policy,
            selected_frame,
            selected_tolerance,
            maximum_normal_deviation,
        )
    if not isinstance(source_surface_charts, SurfaceSourceCharts):
        raise TypeError("source_surface_charts must be SurfaceSourceCharts or None.")
    if fidelity_source is not None or implicit_root_witness is not None or frame_id is not None:
        raise ValueError("Parametric source charts cannot be combined with implicit root operands.")
    candidate: CellGeometrySpec | None = None
    witness: SourceBoundaryChartWitness | None = None
    projection_trace: LayerCurvingProjectionTrace | None = None
    try:
        source_faces = _source_faces(construction, layers, source_surface_charts)
        straight = _straight_geometry(construction.mesh, policy.degree)
        elements, routes, coordinates = straight.resolve(construction.mesh)
        candidate_coordinates = np.array(coordinates, dtype=np.float64, copy=True)
        assigned: dict[int, np.ndarray] = {}
        assigned_exact: dict[int, tuple[ExactScalar, ...]] = {}
        candidate_exact_overrides: dict[int, tuple[ExactScalar, ...]] = {}
        exact_surface_maps: dict[int, tuple[Expression, ...]] = {}
        for face in source_faces:
            block = construction.mesh.blocks[face.target_block_row]
            element = _face_reference_element(
                elements[face.target_block_row], block.cell_kind
            )
            route = np.asarray(routes[face.target_block_row], dtype=np.int64)
            target_solver = _prepare_barycentric_solver_3d(
                face.target_reference_corners
            )
            face_nodes = _reference_face_dofs(
                element, face.target_reference_corners, solver=target_solver
            )
            if not face_nodes:
                continue
            dofs = tuple(local_dof for local_dof, _ in face_nodes)
            exact_parameters = tuple(
                _source_parameter(reference, face, target_solver)
                for _, reference in face_nodes
            )
            exact_map = exact_surface_maps.get(face.source_patch)
            if exact_map is None:
                source_map = exact_surface_coordinate_expressions(
                    source_surface_charts.domain.patches[face.source_patch].surface
                )
                if source_map is None:
                    raise ValueError(
                        "Parametric layer curving requires an exact source-surface map."
                    )
                exact_map = tuple(source_map)
                exact_surface_maps[face.source_patch] = exact_map
            if len(exact_map) != construction.mesh.ambient_dimension:
                raise ValueError(
                    "Exact source-surface map has the wrong physical dimension."
                )
            exact_source_points = tuple(
                tuple(evaluate_expression(expression, parameter) for expression in exact_map)
                for parameter in exact_parameters
            )
            if any(
                not isinstance(value, (Fraction, RadicalElement))
                for point in exact_source_points
                for value in point
            ):
                raise ValueError(
                    "Exact source-surface evaluation must return canonical exact scalars."
                )
            source_points = np.asarray(
                tuple(
                    tuple(float(value) for value in point)
                    for point in exact_source_points
                ),
                dtype=np.float64,
            )
            if source_points.shape != (len(dofs), construction.mesh.ambient_dimension) or not np.all(
                np.isfinite(source_points)
            ):
                raise ValueError("Exact source patch evaluation returned invalid physical coordinates.")
            for (local_dof, _), point, exact_point in zip(
                face_nodes,
                source_points,
                exact_source_points,
                strict=True,
            ):
                geometry_row = int(route[face.target_cell_row, local_dof])
                if geometry_row < 0:
                    raise ValueError("A source-bound reference node has no coordinate-map row.")
                previous = assigned.get(geometry_row)
                if previous is not None and not np.array_equal(
                    previous.view(np.uint64), point.view(np.uint64)
                ):
                    raise ValueError(
                        "Adjacent exact source charts disagree on a shared geometry node."
                    )
                previous_exact = assigned_exact.get(geometry_row)
                if previous_exact is not None and previous_exact != exact_point:
                    raise ValueError(
                        "Adjacent exact source charts disagree on the exact coordinate of a shared geometry node."
                    )
                assigned[geometry_row] = np.array(point, copy=True)
                assigned_exact[geometry_row] = exact_point
                candidate_exact_overrides[geometry_row] = exact_point
                if geometry_row < construction.mesh.coordinates.shape[0]:
                    physical_vertex = np.asarray(
                        construction.mesh.coordinates, dtype=np.float64
                    )[geometry_row]
                    if not np.array_equal(
                        physical_vertex.view(np.uint64), point.view(np.uint64)
                    ):
                        raise ValueError(
                            "Exact source chart changes an immutable layer/core mesh vertex."
                        )
                else:
                    candidate_coordinates[geometry_row] = point
        if not assigned:
            raise ValueError("Exact source charts constrain no high-order layer boundary DOFs.")
        layout = (
            {block.name: element for block, element in zip(construction.mesh.blocks, elements, strict=True)},
            {block.name: route for block, route in zip(construction.mesh.blocks, routes, strict=True)},
        )
        candidate = _geometry_with_exact_coordinate_bank(
            construction.mesh,
            layout[0],
            layout[1],
            straight,
            candidate_coordinates,
            identity_context=(
                "parametric-layer-source-chart",
                source_surface_charts.source_chart_id,
                construction.geometry.geometry_layout_id,
            ),
            exact_overrides=candidate_exact_overrides,
        )
        witness = _chart_witness(
            construction.mesh, candidate, source_surface_charts, source_faces
        )
        _validate_parametric_chart_geometry(
            construction.mesh,
            candidate,
            source_surface_charts,
            witness,
            _column_identity_for_faces(source_faces),
            0.0,
        )
    except (ArithmeticError, IndexError, TypeError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.UNRESOLVED_SOURCE_MAP,
            construction,
            str(error),
            candidate=candidate,
            chart_witness=witness,
        )
    if candidate is None or witness is None:
        raise RuntimeError("Exact source-chart construction returned without its retained target map.")
    try:
        validity = certify_cell_geometry_validity(
            candidate, mesh=construction.mesh, policy=policy.validity
        )
    except (ArithmeticError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_INVALID,
            construction,
            f"Whole-cell validity certification failed: {error}",
            candidate=candidate,
            projection_trace=projection_trace,
            chart_witness=witness,
        )
    if not validity.all_certified:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_INVALID,
            construction,
            "Whole-cell Bernstein validity did not certify every layer/core cell.",
            candidate=candidate,
            validity=validity,
            projection_trace=projection_trace,
            chart_witness=witness,
        )
    try:
        embedding = certify_global_embedding(construction.mesh, candidate, validity)
    except (ArithmeticError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_EMBEDDING,
            construction,
            f"Global embedding certification failed: {error}",
            candidate=candidate,
            validity=validity,
            projection_trace=projection_trace,
            chart_witness=witness,
        )
    if embedding.status != "certified":
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_EMBEDDING,
            construction,
            "Global embedding evidence did not certify the curved layer/core map.",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            projection_trace=projection_trace,
            chart_witness=witness,
        )
    if not np.array_equal(
        np.asarray(construction.mesh.coordinates, dtype=np.float64).view(np.uint64),
        np.asarray(candidate.coordinates, dtype=np.float64)[
            : construction.mesh.coordinates.shape[0]
        ].view(np.uint64),
    ):
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_PROJECTION,
            construction,
            "High-order curving changed layer/core mesh vertices or physical column heights.",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            projection_trace=projection_trace,
            chart_witness=witness,
        )
    tolerance = 0.0 if fidelity_tolerance is None else float(fidelity_tolerance)
    try:
        fidelity = certify_source_fidelity(
            construction.mesh,
            candidate,
            source_surface_charts.boundary_source,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
            witness=witness,
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_FIDELITY,
            construction,
            f"Continuous source-fidelity certification failed: {error}",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            projection_trace=projection_trace,
            chart_witness=witness,
        )
    if fidelity.status != "certified":
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_SOURCE_FIDELITY,
            construction,
            "Continuous source-fidelity evidence did not meet the declared physical tolerance.",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            projection_trace=projection_trace,
            chart_witness=witness,
            source_fidelity=fidelity,
        )
    try:
        blocks = _distortion_blocks(construction.mesh, candidate, policy.degree)
        minimum_scaled, maximum_distortion = _sampled_quality(
            blocks, np.asarray(candidate.coordinates, dtype=np.float64)
        )
        evidence = LayerCurvingSourceEvidence(
            source_surface_charts,
            source_surface_charts.boundary_source,
            layers.result_id,
            layers.evidence,
            witness,
            fidelity,
            minimum_scaled,
            maximum_distortion,
            frame_id=source_surface_charts.coordinate_contract.spatial_id,
            projection_trace=projection_trace,
            chart_column_identity=_column_identity_for_faces(source_faces),
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        return _rollback(
            LayerCurvingStatus.ROLLED_BACK_INVALID,
            construction,
            f"Curved-map evidence construction failed: {error}",
            candidate=candidate,
            validity=validity,
            embedding=embedding,
            projection_trace=projection_trace,
            chart_witness=witness,
            source_fidelity=fidelity,
        )
    return LayerCurvingConstructionResult(
        LayerCurvingStatus.CURVED,
        _construction_with_geometry(construction, candidate),
        candidate,
        validity,
        embedding,
        evidence,
        None,
        projection_trace=projection_trace,
        chart_witness=witness,
        source_fidelity=fidelity,
    )





def _transition_reference_maps(
    reference_maps: tuple[ReferenceCellMapWitnesses, ...],
    /,
) -> dict[tuple[int, int], NestedCellReferenceMap]:
    result: dict[tuple[int, int], NestedCellReferenceMap] = {}
    identities: dict[tuple[int, int], tuple[object, ...] | None] = {}
    for witnesses in reference_maps:
        validate_reference_cell_map_witnesses(witnesses)
        for mapping in nested_cell_reference_maps(witnesses):
            if mapping.arguments is None:
                raise ValueError("Exact reference transition omitted its source-local expression bank.")
            key = (int(mapping.target_cell_id), int(mapping.source_cell_id))
            identity = reference_chart_arguments_identity((mapping.arguments,))
            previous = identities.get(key)
            if key in identities and previous != identity:
                raise ValueError("Reference transition repeats one target/source chart with conflicting expressions.")
            result[key] = mapping
            identities[key] = identity
    return result


def _mixed_coarsening_reference_maps(
    source_mesh: CellMesh,
    maps: tuple[MixedCoarseningMap, ...],
    /,
) -> dict[tuple[int, int], _MixedReferenceAffineMap]:
    transforms = _mixed_coarsening_reference_transforms(maps)
    source_kinds = {
        int(identifier): block.cell_kind
        for block in source_mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    for record in maps:
        child_ids = np.asarray(record.child_cell_ids, dtype=np.int64)
        if child_ids.size != len(record.child_reference_vertices):
            raise ValueError("Mixed coarsening charts do not align with child IDs.")
        if any(
            source_kinds.get(int(child_id)) != record.parent_cell_kind
            for child_id in child_ids
        ):
            raise ValueError("Mixed coarsening changes the layer source cell family.")
    return transforms


def _transition_source_chart_rows(
    source_evidence: LayerCurvingSourceEvidence,
    target_witness: SourceBoundaryChartWitness,
    lineage: MeshLineage,
    reference_maps: dict[tuple[int, int], NestedCellReferenceMap],
    mixed_reference_maps: dict[tuple[int, int], _MixedReferenceAffineMap],
    /,
) -> tuple[int, ...]:
    old_witness = source_evidence.chart_witness
    cell_lineage = lineage.entity_lineage(3)
    relation = {
        (int(source_id), int(target_id)): int(kind)
        for source_id, target_id, kind in zip(
            np.asarray(cell_lineage.source_global_ids, dtype=np.int64),
            np.asarray(cell_lineage.target_global_ids, dtype=np.int64),
            np.asarray(cell_lineage.relation_kinds, dtype=np.int32),
            strict=True,
        )
    }
    allowed = {
        int(EntityLineageKind.PRESERVED),
        int(EntityLineageKind.REFINED_FROM),
        int(EntityLineageKind.COARSENED_INTO),
        int(EntityLineageKind.RELOCATED),
    }
    result: list[int] = []
    for target_row in range(len(target_witness.source_chart_ids)):
        target_cell_id = int(target_witness.cell_global_ids[target_row])
        target_reference = _fraction_chart3(
            target_witness.target_reference_chart(target_row)
        )
        target_parameters = _fraction_chart2(
            target_witness.source_parameter_chart(target_row)
        )
        candidates: list[int] = []
        for old_row in range(len(old_witness.source_chart_ids)):
            if (
                old_witness.source_chart_ids[old_row]
                != target_witness.source_chart_ids[target_row]
                or old_witness.source_entity_ids[old_row]
                != target_witness.source_entity_ids[target_row]
                or old_witness.occurrence_paths[old_row]
                != target_witness.occurrence_paths[target_row]
            ):
                continue
            source_cell_id = int(old_witness.cell_global_ids[old_row])
            key = (target_cell_id, source_cell_id)
            mapping = reference_maps.get(key)
            mixed_mapping = mixed_reference_maps.get(key)
            if mapping is None and mixed_mapping is None:
                continue
            relation_kind = relation.get((source_cell_id, target_cell_id))
            if relation_kind not in allowed:
                raise ValueError("Source chart reference map disagrees with exact cell lineage.")
            old_reference = _fraction_chart3(
                old_witness.target_reference_chart(old_row)
            )
            old_parameters = _fraction_chart2(
                old_witness.source_parameter_chart(old_row)
            )
            if mapping is not None:
                if mapping.arguments is None or len(mapping.arguments) != 3:
                    raise ValueError(
                        "Exact source chart transition omitted its full reference-map expression."
                    )
                reference_chart = old_reference
            else:
                if relation_kind != int(EntityLineageKind.COARSENED_INTO):
                    raise ValueError(
                        "Mixed coarsening charts must bind a coarsened child relation."
                    )
                matrix, offset = mixed_mapping
                reference_chart = tuple(
                    tuple(
                        offset[axis]
                        + sum(
                            (
                                matrix[axis][dimension] * point[dimension]
                                for dimension in range(3)
                            ),
                            Fraction(0),
                        )
                        for axis in range(3)
                    )
                    for point in old_reference
                )
            reference_solver = _prepare_barycentric_solver_3d(reference_chart)
            matches = True
            for target_point, parameter_point in zip(
                target_reference, target_parameters, strict=True
            ):
                source_reference = (
                    _exact3(mapping.evaluate(target_point))
                    if mapping is not None
                    else target_point
                )
                weights = _barycentric_3d(
                    source_reference,
                    reference_chart,
                    solver=reference_solver,
                )
                if any(value < 0 for value in weights):
                    matches = False
                    break
                mapped_parameters = _exact2(
                    tuple(
                        sum(
                            (
                                weights[corner]
                                * old_parameters[corner][axis]
                                for corner in range(3)
                            ),
                            Fraction(0),
                        )
                        for axis in range(2)
                    )
                )
                if mapped_parameters != parameter_point:
                    matches = False
                    break
            if matches:
                candidates.append(old_row)
        if len(candidates) != 1:
            raise ValueError(
                "Transitioned source chart has no unique exact source-face/reference-map ancestor."
            )
        result.append(candidates[0])
    return tuple(result)


def _transition_source_faces(
    source_evidence: LayerCurvingSourceEvidence,
    target_mesh: CellMesh,
    target_witness: SourceBoundaryChartWitness,
    old_chart_rows: tuple[int, ...],
    /,
) -> tuple[_SourceFace, ...]:
    charts = source_evidence.source_surface_charts
    root_witness = source_evidence.implicit_root_chart_witness
    root_vertices = source_evidence.implicit_root_cell_vertex_global_ids
    if charts is None and (
        root_witness is None or root_vertices is None
    ):
        raise ValueError("Layer source evidence lost its parametric or implicit root authority.")
    cells = {
        int(cell_id): (block_row, cell_row, block)
        for block_row, block in enumerate(target_mesh.blocks)
        for cell_row, cell_id in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    chart_cell_rows = (
        {}
        if charts is None
        else {int(cell_id): row for row, cell_id in enumerate(charts.cell_global_ids)}
    )
    root_rows = (
        ()
        if root_witness is None
        else tuple(
            _implicit_root_witness_row(root_witness, target_witness, row)
            for row in range(len(target_witness.source_chart_ids))
        )
    )
    result: list[_SourceFace] = []
    column = source_evidence.chart_column_identity
    for chart_row, old_row in enumerate(old_chart_rows):
        target_cell_id = int(target_witness.cell_global_ids[chart_row])
        try:
            block_row, cell_row, block = cells[target_cell_id]
        except KeyError as error:
            raise ValueError("Transitioned source chart names an absent target cell.") from error
        local_face = int(target_witness.local_entity_indices[chart_row])
        routes = _FACE_ROUTES.get(block.cell_kind)
        if routes is None or local_face < 0 or local_face >= len(routes) or len(routes[local_face]) < 3:
            raise ValueError("Transitioned layer source chart requires a target facet with at least three vertices.")
        cell_vertices = np.asarray(block.vertices, dtype=np.int64)[cell_row]
        target_vertex_rows = _int3(
            tuple(int(cell_vertices[index]) for index in routes[local_face][:3])
        )
        source_cell_id = int(column.source_wall_cell_ids[old_row])
        if charts is not None:
            source_chart_row = chart_cell_rows.get(source_cell_id)
            if source_chart_row is None:
                raise ValueError("Transitioned parametric chart lost its source-wall cell ancestry.")
            patch = int(charts.root_patches[source_chart_row])
            entity_id = target_witness.source_entity_ids[chart_row]
            occurrence = tuple(target_witness.occurrence_paths[chart_row])
            if (
                charts.domain.entity_id(2, patch) != entity_id
                or tuple(charts.domain.source_occurrences[2][patch]) != occurrence
            ):
                raise ValueError("Transitioned parametric chart changed its source entity occurrence.")
            root_cell_id = int(charts.source_root_cell_ids[source_chart_row])
            piece_id = tuple(charts.source_piece_ids[source_chart_row])
            source_patch = patch
            implicit_chart_id = ""
            implicit_cell_key: tuple[int, ...] = ()
            implicit_root_edges: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
            implicit_vertices: tuple[int, ...] = ()
        else:
            assert root_witness is not None and root_vertices is not None
            root_row = root_rows[chart_row]
            if source_cell_id != int(root_witness.cell_global_ids[root_row]):
                raise ValueError("Implicit chart column ancestry changed its source-wall cell.")
            root_cell_id = source_cell_id
            implicit_cell_key = tuple(
                int(value) for value in root_witness.implicit_cell_keys[root_row]
            )
            implicit_root_edges = _implicit_root_edge_row(
                root_witness.implicit_root_edge_keys[root_row]
            )
            piece_id = implicit_cell_key
            entity_id = target_witness.source_entity_ids[chart_row]
            occurrence = tuple(target_witness.occurrence_paths[chart_row])
            source_patch = -1
            implicit_chart_id = root_witness.source_chart_ids[root_row]
            implicit_vertices = _int3(root_vertices[root_row])
        result.append(
            _SourceFace(
                chart_row=chart_row,
                source_cell_id=source_cell_id,
                source_patch=source_patch,
                source_entity_id=entity_id,
                occurrence_path=occurrence,
                source_root_cell_id=root_cell_id,
                source_piece_id=piece_id,
                target_block_row=block_row,
                target_cell_row=cell_row,
                target_cell_id=target_cell_id,
                target_local_face=local_face,
                target_face_global_id=int(target_witness.entity_global_ids[chart_row]),
                target_vertex_rows=target_vertex_rows,
                target_reference_corners=_fraction_chart3(
                    target_witness.target_reference_chart(chart_row)
                ),
                source_parameter_corners=_fraction_chart2(
                    target_witness.source_parameter_chart(chart_row)
                ),
                layer_index=int(column.layer_indices[old_row]),
                layer_column_id=int(column.front_row_ids[old_row]),
                layer_front_row_identity=column.front_row_identities[old_row],
                source_column_ids=column.source_column_ids[old_row],
                source_column_identities=column.source_column_identities[old_row],
                source_column_root_vertex_ids=column.source_column_root_vertex_ids[old_row],
                implicit_chart_id=implicit_chart_id,
                implicit_cell_key=implicit_cell_key,
                implicit_root_edge_keys=implicit_root_edges,
                implicit_root_cell_vertex_global_ids=implicit_vertices,
            )
        )
    # Rows were constructed directly from the already-canonical target witness;
    # preserve that exact ordering, including its target face-vertex tie-breaker.
    return tuple(result)


def _transition_face_reference_element(
    block_kind: str, element: object, /
) -> _NodalReferenceElement:
    if isinstance(element, _NodalReferenceElement):
        return element
    degree = getattr(element, "degree", None)
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise ValueError("Transitioned source map has no supported nodal reference order.")
    return lagrange_element(block_kind, int(degree))



def _select_column_identity(
    identity: LayerColumnProjectionIdentity,
    rows: tuple[int, ...],
    /,
) -> LayerColumnProjectionIdentity:
    return LayerColumnProjectionIdentity(
        np.asarray([identity.layer_indices[row] for row in rows], dtype=np.int64),
        np.asarray([identity.source_wall_cell_ids[row] for row in rows], dtype=np.int64),
        np.asarray([identity.front_row_ids[row] for row in rows], dtype=np.int64),
        tuple(identity.front_row_identities[row] for row in rows),
        tuple(identity.source_column_ids[row] for row in rows),
        tuple(identity.source_column_identities[row] for row in rows),
        tuple(identity.source_column_root_vertex_ids[row] for row in rows),
    )


@dataclass(frozen=True, slots=True)
class _MappedReferenceTransitionRecord:
    source_cell_id: int
    signature: tuple[object, ...] | None
    exact_vertices: tuple[tuple[Fraction, Fraction, Fraction], ...]
    arguments: tuple[Expression, ...]


def _compose_mapped_reference_maps(
    transition_maps: tuple[ReferenceCellMapWitnesses, ...],
    source_maps: tuple[ReferenceCellMapWitnesses, ...],
    /,
    *,
    mixed_coarsening_maps: tuple[MixedCoarseningMap, ...] = (),
    mixed_reference_maps: dict[tuple[int, int], _MixedReferenceAffineMap] | None = None,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    source_by_cell: dict[int, NestedCellReferenceMap] = {}
    source_signatures: dict[int, tuple[object, ...] | None] = {}
    for bank in source_maps:
        validate_reference_cell_map_witnesses(bank)
        for mapping in nested_cell_reference_maps(bank):
            arguments = mapping.arguments
            if arguments is None:
                raise ValueError("Mapped source evidence omitted its exact root-cell expression.")
            signature = (
                mapping.source_cell_id,
                mapping.source_cell_kind,
                reference_chart_arguments_identity((arguments,)),
            )
            previous = source_signatures.get(mapping.target_cell_id)
            if mapping.target_cell_id in source_signatures and previous != signature:
                raise ValueError("Mapped source evidence repeats conflicting root-cell maps.")
            source_by_cell[mapping.target_cell_id] = mapping
            source_signatures[mapping.target_cell_id] = signature
    groups: dict[
        tuple[str, str], dict[int, _MappedReferenceTransitionRecord]
    ] = {}
    for bank in transition_maps:
        validate_reference_cell_map_witnesses(bank)
        for mapping in nested_cell_reference_maps(bank):
            root_map = source_by_cell.get(mapping.source_cell_id)
            if root_map is None:
                continue
            root_arguments = root_map.arguments
            transition_arguments = mapping.arguments
            if (
                root_arguments is None
                or transition_arguments is None
                or mapping.source_cell_kind != root_map.target_cell_kind
            ):
                raise ValueError("Mapped transition lacks compatible exact target-to-root expressions.")
            exact_root_arguments = tuple(
                _reference_expression(expression) for expression in root_arguments
            )
            exact_transition_arguments = tuple(
                _reference_expression(expression) for expression in transition_arguments
            )
            arguments = tuple(
                compose_expression(expression, exact_transition_arguments)
                for expression in exact_root_arguments
            )
            corners = _exact_reference_cell_vertices(mapping.target_cell_kind)
            exact_vertices = tuple(
                _fraction3(
                    tuple(evaluate_expression(expression, point) for expression in arguments)
                )
                for point in corners
            )
            key = (mapping.target_cell_kind, root_map.source_cell_kind)
            signature = reference_chart_arguments_identity((arguments,))
            records = groups.setdefault(key, {})
            record = _MappedReferenceTransitionRecord(
                root_map.source_cell_id,
                signature,
                exact_vertices,
                arguments,
            )
            previous = records.get(mapping.target_cell_id)
            if previous is not None and previous != record:
                raise ValueError("Mapped target cell has conflicting root-reference compositions.")
            records[mapping.target_cell_id] = record
    mixed_transforms = (
        _mixed_coarsening_reference_transforms(mixed_coarsening_maps)
        if mixed_reference_maps is None
        else mixed_reference_maps
    )
    for mixed_map in mixed_coarsening_maps:
        child_ids = np.asarray(mixed_map.child_cell_ids, dtype=np.int64)
        if child_ids.size != len(mixed_map.child_reference_vertices):
            raise ValueError("Mixed coarsening maps do not align with their child IDs.")
        for child_id_raw in child_ids:
            child_id = int(child_id_raw)
            root_map = source_by_cell.get(child_id)
            if (
                root_map is None
                or root_map.arguments is None
                or root_map.target_cell_kind != mixed_map.parent_cell_kind
            ):
                raise ValueError(
                    "Mixed mapped coarsening lacks compatible child-to-root ancestry."
                )
            transform = mixed_transforms.get(
                (mixed_map.parent_cell_id, child_id)
            )
            if transform is None:
                raise ValueError("Mixed mapped coarsening omitted an exact child chart.")
            parent_to_child = _parent_to_child_chart(*transform)
            root_arguments = tuple(
                _reference_expression(expression) for expression in root_map.arguments
            )
            arguments = tuple(
                compose_expression(expression, parent_to_child)
                for expression in root_arguments
            )
            corners = _exact_reference_cell_vertices(mixed_map.parent_cell_kind)
            exact_vertices = tuple(
                _fraction3(
                    tuple(
                        evaluate_expression(expression, point)
                        for expression in arguments
                    )
                )
                for point in corners
            )
            key = (
                mixed_map.parent_cell_kind,
                root_map.source_cell_kind,
            )
            signature = reference_chart_arguments_identity((arguments,))
            records = groups.setdefault(key, {})
            record = _MappedReferenceTransitionRecord(
                root_map.source_cell_id,
                signature,
                exact_vertices,
                arguments,
            )
            previous = records.get(mixed_map.parent_cell_id)
            if previous is not None and previous != record:
                raise ValueError(
                    "Mixed coarsened children disagree on their exact root map."
                )
            records[mixed_map.parent_cell_id] = record
    if not groups:
        raise ValueError("Mapped source transition retained no exact target-to-root cell maps.")
    result: list[ReferenceCellMapWitnesses] = []
    for (target_kind, root_kind), records in sorted(groups.items()):
        ordered = tuple(sorted(records.items()))
        target_ids = np.asarray([target_id for target_id, _ in ordered], dtype=np.int64)
        root_ids = np.asarray(
            [record.source_cell_id for _, record in ordered], dtype=np.int64
        )
        exact_bank = tuple(record.exact_vertices for _, record in ordered)
        execution_bank = np.asarray(
            [[[float(value) for value in point] for point in cell] for cell in exact_bank],
            dtype=np.float64,
        )
        arguments = freeze_reference_chart_arguments(
            tuple(record.arguments for _, record in ordered)
        )
        for value in (target_ids, root_ids, execution_bank):
            value.setflags(write=False)
        bank = ReferenceCellMapWitnesses(
            target_ids,
            root_ids,
            target_kind,
            root_kind,
            execution_bank,
            exact_bank,
            arguments,
        )
        validate_reference_cell_map_witnesses(bank)
        result.append(bank)
    return tuple(result)



def _renew_mapped_layer_source_evidence(
    source_evidence: LayerCurvingSourceEvidence,
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    target_chart_witness: SourceBoundaryChartWitness,
    lineage: MeshLineage,
    transition_maps: tuple[ReferenceCellMapWitnesses, ...],
    transition_maps_by_pair: dict[tuple[int, int], NestedCellReferenceMap],
    mixed_coarsening_maps: tuple[MixedCoarseningMap, ...],
    mixed_reference_maps: dict[tuple[int, int], _MixedReferenceAffineMap],
    /,
) -> LayerCurvingSourceEvidence:
    query = source_evidence.source_query
    if (
        not isinstance(query, MappedReferenceBoundarySource)
        or source_evidence.mapped_chart_trace is None
        or source_evidence.source_surface_charts is not None
        or source_evidence.implicit_root_trace is not None
        or source_evidence.frame_id != query.coordinate_contract.spatial_id
        or target_chart_witness.source_query_id != query.source_query_id
    ):
        raise ValueError("Mapped layer renewal lost its original source/frame/root-chart authority.")
    source_evidence.require_current(source_mesh, source_geometry)
    old_chart_rows = _transition_source_chart_rows(
        source_evidence,
        target_chart_witness,
        lineage,
        transition_maps_by_pair,
        mixed_reference_maps,
    )
    mapped_maps = _compose_mapped_reference_maps(
        transition_maps,
        source_evidence.mapped_reference_maps,
        mixed_coarsening_maps=mixed_coarsening_maps,
        mixed_reference_maps=mixed_reference_maps,
    )
    mapped_rows = _mapped_reference_chart_rows(query, target_chart_witness, mapped_maps)
    chart_identity = _select_column_identity(
        source_evidence.chart_column_identity, old_chart_rows
    )
    elements, routes, coordinates = target_geometry.resolve(target_mesh)
    cell_rows = {
        int(cell_id): (block_row, cell_row)
        for block_row, block in enumerate(target_mesh.blocks)
        for cell_row, cell_id in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    records: list[_MappedTransitionNodeRecord] = []
    expected: dict[tuple[int, int], set[int]] = {}
    covered: dict[tuple[int, int], set[int]] = {}
    for chart_row, (_, root_face, root_cell, root_local, mapping) in enumerate(mapped_rows):
        target_cell_id = int(target_chart_witness.cell_global_ids[chart_row])
        block_row, cell_row = cell_rows[target_cell_id]
        cell_kind = target_mesh.blocks[block_row].cell_kind
        face_element = _transition_face_reference_element(
            cell_kind, elements[block_row]
        )
        local_face = int(target_chart_witness.local_entity_indices[chart_row])
        facet_key = (target_cell_id, local_face)
        route = np.asarray(routes[block_row], dtype=np.int64)
        has_geometry_nodes = isinstance(elements[block_row], _NodalReferenceElement)
        target_corners = _fraction_chart3(
            target_chart_witness.target_reference_chart(chart_row)
        )
        face_nodes = _reference_face_dofs(face_element, target_corners)
        dofs = tuple(local_dof for local_dof, _ in face_nodes)
        expected.setdefault(facet_key, set()).update(dofs)
        covered.setdefault(facet_key, set())
        contexts = _mapped_chart_reference_contexts(
            query,
            target_chart_witness,
            chart_row,
            mapping,
            tuple(reference for _, reference in face_nodes),
        )
        for (local_dof, target_reference), (
            root_reference,
            source_parameter,
            source_chart_uv,
            source_chart_solver,
        ) in zip(face_nodes, contexts, strict=True):
            covered[facet_key].add(local_dof)
            node_row = int(route[cell_row, local_dof]) if has_geometry_nodes else -1
            if node_row >= 0:
                target_point = np.asarray(coordinates[node_row], dtype=np.float64)
            elif node_row == -1:
                target_point = _evaluate_cell_geometry_point(
                    elements,
                    routes,
                    coordinates,
                    block_row,
                    cell_row,
                    target_reference,
                )
            else:
                raise ValueError("Mapped transition has an invalid target geometry-node sentinel.")
            root_reference, source_parameter, source_point, tangents, normal = (
                _evaluate_mapped_chart_point(
                    query,
                    target_chart_witness,
                    chart_row,
                    target_reference,
                    mapping,
                    root_reference,
                    source_parameter,
                    source_chart_uv,
                    source_chart_solver,
                )
            )
            distance_lower, distance_upper = _mapped_physical_distance_bounds(
                query,
                target_chart_witness.source_chart_ids[chart_row],
                source_parameter,
                target_point,
            )
            records.append(
                _MappedTransitionNodeRecord(
                    node_row,
                    target_cell_id,
                    local_dof,
                    chart_row,
                    target_reference,
                    root_reference,
                    source_parameter,
                    target_point,
                    source_point,
                    tangents,
                    normal,
                    distance_lower,
                    distance_upper,
                    root_face,
                    root_cell,
                    root_local,
                )
            )
    if not records or any(covered.get(key, set()) != values for key, values in expected.items()):
        raise ValueError("Mapped transition chart rows omit target high-order boundary DOFs.")
    trace_chart_rows = tuple(record.chart_row for record in records)
    trace_columns = _select_column_identity(chart_identity, trace_chart_rows)
    trace = LayerCurvingMappedChartTrace(
        query.source_id,
        query.source_revision,
        source_evidence.frame_id,
        query.source_query_id,
        np.asarray([record.geometry_node_row for record in records], dtype=np.int64),
        np.asarray([record.target_cell_id for record in records], dtype=np.int64),
        np.asarray([record.local_dof for record in records], dtype=np.int32),
        np.asarray(trace_chart_rows, dtype=np.int64),
        tuple(record.target_reference for record in records),
        tuple(record.root_reference for record in records),
        tuple(record.source_parameter for record in records),
        tuple(target_chart_witness.source_chart_ids[row] for row in trace_chart_rows),
        tuple(target_chart_witness.source_entity_ids[row] for row in trace_chart_rows),
        tuple(target_chart_witness.occurrence_paths[row] for row in trace_chart_rows),
        np.asarray(
            [record.root_face_global_id for record in records], dtype=np.int64
        ),
        np.asarray(
            [record.root_cell_global_id for record in records], dtype=np.int64
        ),
        np.asarray(
            [record.root_local_face_index for record in records], dtype=np.int32
        ),
        np.asarray([record.target_point for record in records], dtype=np.float64),
        np.asarray([record.source_point for record in records], dtype=np.float64),
        np.asarray([record.distance_lower for record in records], dtype=np.float64),
        np.asarray([record.distance_upper for record in records], dtype=np.float64),
        np.asarray([record.normal for record in records], dtype=np.float64),
        np.asarray([record.tangents for record in records], dtype=np.float64),
        trace_columns,
    )
    if not trace.resolved:
        raise ValueError("Mapped transition source trace is unresolved.")
    fidelity = certify_source_fidelity(
        target_mesh,
        target_geometry,
        query,
        tolerance=source_evidence.fidelity.tolerance,
        maximum_normal_deviation=source_evidence.fidelity.maximum_normal_deviation,
        witness=target_chart_witness,
    )
    if fidelity.status != "certified":
        raise ValueError("Mapped transition did not retain continuous source fidelity.")
    degree = max(int(getattr(element, "degree", 1)) for element in elements)
    quality_blocks = _distortion_blocks(target_mesh, target_geometry, degree)
    minimum_scaled, maximum_distortion = _sampled_quality(
        quality_blocks, np.asarray(coordinates, dtype=np.float64)
    )
    return LayerCurvingSourceEvidence(
        None,
        query,
        source_evidence.layer_result_id,
        source_evidence.layer_evidence,
        target_chart_witness,
        fidelity,
        minimum_scaled,
        maximum_distortion,
        mapped_chart_trace=trace,
        mapped_reference_maps=mapped_maps,
        chart_column_identity=chart_identity,
        frame_id=source_evidence.frame_id,
    )
def renew_layer_curving_source_evidence(
    source_evidence: LayerCurvingSourceEvidence,
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    target_chart_witness: SourceBoundaryChartWitness,
    lineage: MeshLineage,
    reference_maps: tuple[ReferenceCellMapWitnesses, ...],
    /,
    *,
    coarsening: NestedReferenceWitnesses | None = None,
    mixed_coarsening_maps: tuple[MixedCoarseningMap, ...] = (),
) -> LayerCurvingSourceEvidence:
    """Recompute layer source, node, and column evidence through exact transitions.

    Parametric and adaptive-implicit traces are rebuilt from their native
    authority. Mapped-root transitions compose exact target-to-current and
    current-to-root charts, then reevaluate every mapped boundary DOF and
    derivative without copying stale source or column evidence.
    """
    if not isinstance(source_evidence, LayerCurvingSourceEvidence):
        raise TypeError("source_evidence must be LayerCurvingSourceEvidence.")
    if (
        not isinstance(source_mesh, CellMesh)
        or not isinstance(source_geometry, CellGeometrySpec)
        or not isinstance(target_mesh, CellMesh)
        or not isinstance(target_geometry, CellGeometrySpec)
        or not isinstance(target_chart_witness, SourceBoundaryChartWitness)
        or not isinstance(lineage, MeshLineage)
    ):
        raise TypeError("Layer evidence renewal requires canonical source/target geometry endpoints.")
    if source_mesh.topological_dimension != 3 or target_mesh.topological_dimension != 3:
        raise ValueError("Layer source evidence renewal requires volume-cell endpoints.")
    source_evidence.require_current(source_mesh, source_geometry)
    if (
        lineage.source_topology_id != source_mesh.topology_id
        or lineage.target_topology_id != target_mesh.topology_id
        or target_chart_witness.mesh_id != target_mesh.mesh_id
        or target_chart_witness.topology_id != target_mesh.topology_id
        or target_chart_witness.geometry_id != cell_geometry_id(target_geometry)
        or target_chart_witness.source_id != source_evidence.source_id
        or target_chart_witness.source_revision != source_evidence.source_revision
    ):
        raise ValueError("Layer source chart transition endpoints do not bind the accepted source.")
    if coarsening is not None:
        validate_nested_reference_witnesses(coarsening, dimension=3)
    reference_maps_by_pair = _transition_reference_maps(reference_maps)
    mixed_reference_maps = _mixed_coarsening_reference_maps(
        source_mesh, mixed_coarsening_maps
    )
    if set(reference_maps_by_pair) & set(mixed_reference_maps):
        raise ValueError("Nested and mixed coarsening maps repeat a source chart transition.")
    if source_evidence.mapped_chart_trace is not None:
        return _renew_mapped_layer_source_evidence(
            source_evidence,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            target_chart_witness,
            lineage,
            reference_maps,
            reference_maps_by_pair,
            mixed_coarsening_maps,
            mixed_reference_maps,
        )
    old_chart_rows = _transition_source_chart_rows(
        source_evidence,
        target_chart_witness,
        lineage,
        reference_maps_by_pair,
        mixed_reference_maps,
    )
    faces = _transition_source_faces(
        source_evidence, target_mesh, target_chart_witness, old_chart_rows
    )
    elements, routes, coordinates = target_geometry.resolve(target_mesh)
    implicit_records: list[
        tuple[int, int, int, int, _SourceFace, tuple[ExactScalar, ExactScalar, ExactScalar]]
    ] = []
    covered_dofs: dict[tuple[int, int], set[int]] = {}
    expected_dofs: dict[tuple[int, int], set[int]] = {}
    for face in faces:
        element = _transition_face_reference_element(
            target_mesh.blocks[face.target_block_row].cell_kind,
            elements[face.target_block_row],
        )
        route = np.asarray(routes[face.target_block_row], dtype=np.int64)
        facet_key = (face.target_cell_id, face.target_local_face)
        coordinate_element = elements[face.target_block_row]
        has_geometry_nodes = isinstance(
            coordinate_element, _NodalReferenceElement
        )
        face_nodes = _reference_face_dofs(element, face.target_reference_corners)
        dofs = tuple(local_dof for local_dof, _ in face_nodes)
        expected_dofs.setdefault(facet_key, set()).update(dofs)
        for local_dof, reference in face_nodes:
            if has_geometry_nodes:
                geometry_row = int(route[face.target_cell_row, local_dof])
                if geometry_row < 0:
                    raise ValueError("Transitioned source-face DOF has no coordinate-map row.")
            else:
                geometry_row = -1
            covered_dofs.setdefault(facet_key, set()).add(local_dof)
            if source_evidence.source_surface_charts is None:
                implicit_records.append(
                    (
                        geometry_row,
                        face.target_cell_id,
                        local_dof,
                        face.chart_row,
                        face,
                        reference,
                    )
                )
    if any(covered_dofs.get(key, set()) != value for key, value in expected_dofs.items()):
        raise ValueError("Transitioned source chart rows do not cover every target boundary geometry DOF.")

    projection_trace: LayerCurvingProjectionTrace | None = None
    implicit_root_trace: LayerCurvingImplicitRootTrace | None = None
    implicit_map_witness_id: str | None = None
    implicit_chart_evidence: tuple[ImplicitRootChartEvidence, ...] = ()
    implicit_root_chart_witness = source_evidence.implicit_root_chart_witness
    implicit_root_vertices = source_evidence.implicit_root_cell_vertex_global_ids
    if source_evidence.source_surface_charts is not None:
        _validate_parametric_chart_geometry(
            target_mesh,
            target_geometry,
            source_evidence.source_surface_charts,
            target_chart_witness,
            _column_identity_for_faces(faces),
            source_evidence.fidelity.tolerance,
        )
    else:
        if implicit_root_chart_witness is None or implicit_root_vertices is None:
            raise ValueError("Implicit transition lost source-wall root chart authority.")
        if not isinstance(source_evidence.source_query, SourceBoundaryProjectionQuery):
            raise ValueError("Implicit source transition requires its native projection query.")
        adaptive_source = _adaptive_implicit_source_query(source_evidence.source_query)
        root_map = prepare_adaptive_implicit_root_charts(
            target_mesh,
            target_geometry,
            adaptive_source,
            target_chart_witness,
            frame_id=source_evidence.frame_id,
            tolerance=source_evidence.fidelity.tolerance,
            maximum_normal_deviation=source_evidence.fidelity.maximum_normal_deviation,
        )
        if root_map.status != "certified" or root_map.witness_id is None:
            details = tuple(
                item.detail for item in root_map.chart_evidence if item.status != "certified"
            )
            raise ValueError(
                "; ".join(details)
                or "Transitioned adaptive implicit root charts are unresolved."
            )
        implicit_chart_evidence = tuple(root_map.chart_evidence)
        node_rows: list[int] = []
        cell_ids: list[int] = []
        local_dofs: list[int] = []
        chart_rows: list[int] = []
        reference_points: list[tuple[ExactScalar, ExactScalar, ExactScalar]] = []
        target_points: list[np.ndarray] = []
        source_points: list[np.ndarray] = []
        displacement: list[float] = []
        root_errors: list[float] = []
        source_cell_keys: list[tuple[int, ...]] = []
        root_edges: list[tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]] = []
        statuses: list[str] = []
        row_faces: list[_SourceFace] = []
        for geometry_row, cell_id, local_dof, chart_row, face, reference in implicit_records:
            point = root_map.project_target(chart_row, reference)
            node_rows.append(geometry_row)
            cell_ids.append(cell_id)
            local_dofs.append(local_dof)
            chart_rows.append(chart_row)
            reference_points.append(reference)
            target_point = (
                np.asarray(coordinates[geometry_row], dtype=np.float64)
                if geometry_row >= 0
                else _evaluate_cell_geometry_point(
                    elements,
                    routes,
                    coordinates,
                    face.target_block_row,
                    face.target_cell_row,
                    reference,
                )
            )
            target_points.append(target_point)
            source_points.append(np.asarray(point.point, dtype=np.float64))
            displacement.append(point.displacement_upper)
            root_errors.append(point.root_error)
            source_cell_keys.append(point.source_cell_key)
            root_edges.append(point.root_edge_keys)
            statuses.append(point.status)
            row_faces.append(face)
        implicit_root_trace = LayerCurvingImplicitRootTrace(
            source_evidence.source_id,
            source_evidence.source_revision,
            source_evidence.frame_id,
            root_map.witness_id,
            target_chart_witness.witness_id,
            np.asarray(node_rows, dtype=np.int64),
            np.asarray(cell_ids, dtype=np.int64),
            np.asarray(local_dofs, dtype=np.int32),
            np.asarray(chart_rows, dtype=np.int64),
            tuple(reference_points),
            np.asarray(target_points, dtype=np.float64),
            np.asarray(source_points, dtype=np.float64),
            np.asarray(displacement, dtype=np.float64),
            np.asarray(root_errors, dtype=np.float64),
            tuple(source_cell_keys),
            tuple(root_edges),
            tuple(statuses),
            implicit_chart_evidence,
            _column_identity_for_faces(tuple(row_faces)),
        )
        if not implicit_root_trace.resolved:
            raise ValueError("Transitioned adaptive implicit root trace is unresolved.")
        implicit_map_witness_id = root_map.witness_id
    fidelity = certify_source_fidelity(
        target_mesh,
        target_geometry,
        source_evidence.source_query,
        tolerance=source_evidence.fidelity.tolerance,
        maximum_normal_deviation=source_evidence.fidelity.maximum_normal_deviation,
        witness=target_chart_witness,
    )
    if fidelity.status != "certified":
        raise ValueError("Transitioned layer map did not retain certified continuous source fidelity.")
    degree = max(int(getattr(element, "degree", 1)) for element in elements)
    quality_blocks = _distortion_blocks(target_mesh, target_geometry, degree)
    minimum_scaled, maximum_distortion = _sampled_quality(
        quality_blocks, np.asarray(coordinates, dtype=np.float64)
    )
    return LayerCurvingSourceEvidence(
        source_evidence.source_surface_charts,
        source_evidence.source_query,
        source_evidence.layer_result_id,
        source_evidence.layer_evidence,
        target_chart_witness,
        fidelity,
        minimum_scaled,
        maximum_distortion,
        projection_trace=projection_trace,
        implicit_root_trace=implicit_root_trace,
        implicit_map_witness_id=implicit_map_witness_id,
        implicit_chart_evidence=implicit_chart_evidence,
        implicit_root_chart_witness=implicit_root_chart_witness,
        implicit_root_cell_vertex_global_ids=implicit_root_vertices,
        chart_column_identity=_column_identity_for_faces(faces),
    )


__all__ = [
    "LayerCurvingConstructionResult",
    "LayerCurvingFailure",
    "require_layer_core_construction",
    "LayerCurvingStatus",
    "curve_layer_core_construction",
    "renew_layer_curving_source_evidence",
]
