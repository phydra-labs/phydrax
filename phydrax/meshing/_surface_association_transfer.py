# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Propagate parametric surface support through source topology lineage."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.meshing._surface_boundary_operations as surface_boundary_operations_owner
import phydrax.meshing._surface_boundary_sources as surface_boundary_sources_owner
from phydrax.discretization._sphere_chart_records import PreparedSphereChartDeformation
from phydrax.discretization._surface_chart_records import (
    PreparedSurfaceChartDeformation,
    SurfaceChartWitness,
)
from phydrax.meshing._association_state import (
    entity_rows,
    GeometryAssociation,
    GeometryAssociationProvenance,
    target_dimension,
)

from .._fingerprint import array_tree_fingerprint
from .._geometry_predicates import orient2d, PredicateMode
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry_transition_types import MaterialChartDeformation
from ..discretization._chart_geometry import area
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    expression_evaluate,
    require_rational_reference_scalar,
)
from ..geometry._domain_boundary import MeshingDomainBoundarySource
from ..geometry._domain_coverage import DomainCoverageCertificate
from ..geometry._global_embedding import GlobalEmbeddingCertificate
from ..geometry._meshing_domain import PatchCurveUse, PatchPoleUse
from ..geometry._surface_source_support import (
    curve_owner,
    PreparedSurfaceSourceSupport,
    SurfaceNativeRestrictionBoundarySource,
    SurfaceRestrictionRows,
    SurfaceSourceReceipts,
)
from ..typing import checked
from ._association_routes import (
    association_for_dimension,
    association_lookup,
    AssociationPropagationError,
    child_sources,
    incidence_pairs,
    require_transfer_lineage,
    surface_source_association_kind,
)
from ._certification_inputs import MeshCertificationInputs
from ._entity_keys import key_rows
from ._lineage_records import EntityLineageKind, MeshLineage
from ._result import CellMeshingResult
from ._surface_boundary_operations import (
    created_weights,
    lineage_ancestry,
    subdivision_proof,
)


if TYPE_CHECKING:
    from ..geometry._sphere_material_atlas import SphereMaterialCellAtlas


class SurfaceEntityClasses(NamedTuple):
    """Retain surface strata and occurrence-aware native codes."""

    dimensions: np.ndarray
    indices: np.ndarray
    resolved: np.ndarray
    codes: np.ndarray


class SurfaceChartAssociationRows(NamedTuple):
    cell_ids: np.ndarray
    patches: np.ndarray
    vertex_ids: tuple[tuple[int, ...], ...]
    source_corners: tuple[tuple[tuple[Fraction, ...], ...], ...]
    source_bounds: np.ndarray


class _TrimChartPhase(StrictModule):
    coordinates: np.ndarray
    triangles: np.ndarray
    physical_vertices: dict[int, int]
    edges: dict[tuple[int, int], list[tuple[int, int]]]


class _TrimBoundaryPhase(StrictModule):
    edges: np.ndarray
    provenance: np.ndarray


def _prepare_trim_chart(
    uv_cells: np.ndarray,
    cells: np.ndarray,
    budget: CoordinateEnclosureBudget,
) -> _TrimChartPhase:
    uv, inverse = np.unique(uv_cells.reshape(-1, 2), axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    physical_vertices: dict[int, int] = {}
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    orientation = orient2d(
        uv_cells[:, 0],
        uv_cells[:, 1],
        uv_cells[:, 2],
        mode=PredicateMode.EXACT,
        budget=budget,
    )
    signs = np.asarray(orientation.signs, dtype=np.int8)
    if np.any(signs == 0):
        raise ValueError(
            "SurfaceCurveCover.charts must have nonzero reference orientation; got a singular trim triangle"
        )
    for row, (triangle, physical) in enumerate(zip(triangles, cells, strict=True)):
        if signs[row] < 0:
            triangle = triangle[[0, 2, 1]]
            physical = physical[[0, 2, 1]]
            triangles[row] = triangle
        for vertex, original in zip(triangle, physical, strict=True):
            previous = physical_vertices.get(int(vertex))
            if previous is not None and previous != int(original):
                raise ValueError(
                    "SurfaceCurveCover.vertices must retain distinct physical vertex identities; got merged physical vertices"
                )
            physical_vertices[int(vertex)] = int(original)
        for first, last in ((0, 1), (1, 2), (2, 0)):
            a, b = int(triangle[first]), int(triangle[last])
            edges.setdefault((min(a, b), max(a, b)), []).append((a, b))
    return _TrimChartPhase(
        coordinates=uv,
        triangles=triangles,
        physical_vertices=physical_vertices,
        edges=edges,
    )


def _prepare_trim_boundary(
    witness: PreparedSurfaceCurveWitness,
    patch: int,
    chart: _TrimChartPhase,
    kinds: np.ndarray,
    rows: np.ndarray,
    values: np.ndarray,
    budget: CoordinateEnclosureBudget,
) -> _TrimBoundaryPhase:
    domain = witness.support.domain
    uv, physical_vertices, edges = chart.coordinates, chart.physical_vertices, chart.edges
    boundary, provenance = [], []
    for occurrences in edges.values():
        if len(occurrences) == 2:
            continue
        if len(occurrences) != 1:
            raise ValueError(
                "SurfaceCurveCover.edges must have one boundary or two interior occurrences; got a nonmanifold trim edge"
            )
        a, b = occurrences[0]
        endpoints = np.asarray(
            (physical_vertices[a], physical_vertices[b]), dtype=np.int64
        )
        curve, first, last = witness.edge_interval(
            kinds[endpoints], rows[endpoints], values[endpoints]
        )
        matches = []
        for loop_index, loop in enumerate(domain.patches[int(patch)].loops):
            for use_index, use in enumerate(loop):
                if not isinstance(use, PatchCurveUse) or use.curve != curve:
                    continue
                budget.reserve_geometry_queries(2)
                source_charts = np.asarray(
                    use.charts(np.asarray((first, last), dtype=np.float64))
                )
                if np.array_equal(source_charts, uv[[a, b]]):
                    matches.append((loop_index, use_index, first, last))
        if len(matches) != 1:
            raise ValueError(
                "SurfaceCurveCover.coedges must identify one coedge occurrence; got ambiguous boundary provenance"
            )
        boundary.append((a, b))
        provenance.append(matches[0])
    return _TrimBoundaryPhase(
        edges=np.asarray(boundary, dtype=np.int64).reshape(-1, 2),
        provenance=np.asarray(provenance, dtype=np.float64).reshape(-1, 4),
    )


@checked
def prepare_surface_curve_chart_cover(
    witness: PreparedSurfaceCurveWitness,
    mesh: cell_mesh_owner.CellMesh,
    charts: SurfaceChartWitness,
    dimensions: np.ndarray,
    indices: np.ndarray,
    parameters: np.ndarray,
    /,
    *,
    source_chart_cover: MeshingDomainBoundarySource,
    budget: CoordinateEnclosureBudget,
) -> MeshingDomainBoundarySource:
    """Retain current coedge parameters for the trim-ribbon proof.

    Born vertex tokens come from the source-bound controller, never a physical
    inverse or generic native feature label. These are evidence inputs: the
    chart-cover owner independently certifies continuous trims and ribbons.
    """

    witness.support.require_current()
    domain = witness.support.domain
    if source_chart_cover.domain.domain_id != domain.domain_id or set(
        source_chart_cover.patches
    ) != set(map(int, np.asarray(charts.patches))):
        raise ValueError(
            "SurfaceCurveCover.source must preserve the retained source cover domain and patches; got different source cover"
        )
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    if charts.topology_id != mesh.topology_id or charts.domain_id != domain.domain_id:
        raise ValueError(
            "SurfaceCurveCover.topology must bind the chart topology and domain; got different chart binding"
        )
    kinds, rows, values = (
        np.asarray(dimensions),
        np.asarray(indices),
        np.asarray(parameters),
    )
    if (
        kinds.shape != (mesh.coordinates.shape[0],)
        or rows.shape != kinds.shape
        or values.shape != (kinds.size, 2)
    ):
        raise ValueError(
            "SurfaceCurveCover.strata must contain one stratum token per physical vertex; got misaligned stratum arrays"
        )
    declared = np.asarray(charts.cell_global_ids)
    order = np.searchsorted(declared, ids)
    if np.any(order >= declared.size) or not np.array_equal(declared[order], ids):
        raise ValueError(
            "SurfaceCurveCover.cells must represent every physical cell; got an omitted cell ID"
        )
    records = []
    for patch in np.unique(np.asarray(charts.patches)):
        selected = np.flatnonzero(np.asarray(charts.patches)[order] == patch)
        uv_cells = np.asarray(charts.charts)[order[selected]]
        chart = _prepare_trim_chart(uv_cells, cells[selected], budget)
        boundary = _prepare_trim_boundary(
            witness, int(patch), chart, kinds, rows, values, budget
        )
        uv, triangles = chart.coordinates, chart.triangles
        physical = domain.evaluate(np.full(uv.shape[0], int(patch), dtype=np.int32), uv)
        records.append(
            (
                int(patch),
                uv,
                triangles,
                physical,
                boundary.edges,
                boundary.provenance,
                np.zeros((uv.shape[0],), dtype=np.bool_),
                np.empty((0,), dtype=np.int64),
                np.empty((0, 2), dtype=np.int64),
                np.empty((0, 2), dtype=np.int64),
            )
        )
    return MeshingDomainBoundarySource(
        domain,
        tuple(record[0] for record in records),
        resolution=source_chart_cover.resolution,
        chart_triangulations=tuple(records),
    )


class _CurveParameterResolution(StrictModule):
    parameter: float | None = eqx.field(static=True)
    reason: str = eqx.field(static=True)


class _CurveIntervalResolution(StrictModule):
    interval: tuple[int, float, float] | None = eqx.field(static=True)
    reason: str = eqx.field(static=True)


class PreparedSurfaceCurveWitness(StrictModule, NonTrainableState):
    """Retain source vertex strata and incident coedge parameter occurrences.

    Parameters use each coedge interval, including both uses on a periodic seam.
    """

    support: PreparedSurfaceSourceSupport
    source_mesh: cell_mesh_owner.CellMesh
    source_association: GeometryAssociation
    topology_id: str = eqx.field(static=True)
    vertex_global_ids: Array
    source_dimensions: Array
    source_indices: Array
    source_parameters: Array
    source_entity_ids: tuple[str, ...] = eqx.field(static=True)
    source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)

    def validate_restored(self) -> None:
        """Authenticate lowered rows against the retained source proof."""
        self.support.require_current()
        mesh, association = self.source_mesh, self.source_association
        domain = self.support.domain
        if not isinstance(mesh, cell_mesh_owner.CellMesh) or not isinstance(
            association, GeometryAssociation
        ):
            raise TypeError(
                "SurfaceCurveWitness.source must retain a mesh and geometry association; got incompatible restored record kinds"
            )
        if mesh.topology_id != self.topology_id or not np.array_equal(
            np.asarray(mesh.vertex_global_ids), np.asarray(self.vertex_global_ids)
        ):
            raise ValueError(
                "SurfaceCurveWitness.topology must preserve source topology and vertex IDs; got different restored vertex topology"
            )
        if (
            association.association_kind
            is not surface_source_association_kind(domain.source_kinds)
            or association.source_id != domain.source_id
            or association.source_revision != domain.source_revision
            or not association.complete
            or target_dimension(mesh, association) != 0
        ):
            raise ValueError(
                "SurfaceCurveWitness.association must retain source identity and vertex association coverage; got different restored source association"
            )
        rows = association.target_rows(np.asarray(mesh.vertex_global_ids, dtype=np.int64))
        kinds = np.asarray(association.source_dimensions)[rows]
        indices = np.asarray(
            [
                self.support.domain_row(
                    domain,
                    int(association.source_dimensions[row]),
                    int(association.source_indices[row]),
                    association.source_entities.occurrence_path(row),
                )
                for row in rows
            ],
            dtype=np.int32,
        )
        entities = tuple(association.source_entities.entity_id(row) for row in rows)
        occurrences = tuple(
            association.source_entities.occurrence_path(row) for row in rows
        )

        original_arrays = (
            np.asarray(mesh.vertex_global_ids),
            kinds,
            indices,
            np.asarray(association.parameters)[rows],
        )
        lowered_arrays = (
            self.vertex_global_ids,
            self.source_dimensions,
            self.source_indices,
            self.source_parameters,
        )
        if array_tree_fingerprint(lowered_arrays) != array_tree_fingerprint(
            original_arrays
        ):
            raise ValueError(
                "SurfaceCurveWitness.arrays must preserve source array types, shapes and parameter bits; got different restored arrays"
            )
        if (
            self.source_entity_ids != entities
            or self.source_occurrence_paths != occurrences
        ):
            raise ValueError(
                "SurfaceCurveWitness.rows must preserve source strata, parameters, entities and occurrences; got different restored source rows"
            )
        if any(
            entity != domain.entity_id(int(kind), int(index))
            or occurrence != domain.source_occurrences[int(kind)][int(index)]
            for entity, occurrence, kind, index in zip(
                entities, occurrences, kinds, indices, strict=True
            )
        ):
            raise ValueError(
                "SurfaceCurveWitness.entities must name declared source entities; got different entity bindings"
            )

    def require_bound(self, mesh: cell_mesh_owner.CellMesh, /) -> None:
        self.validate_restored()
        if mesh.topology_id != self.topology_id or not np.array_equal(
            np.asarray(mesh.vertex_global_ids), np.asarray(self.vertex_global_ids)
        ):
            raise ValueError(
                "SurfaceCurveWitness.topology must bind the mesh vertex topology; got stale topology binding"
            )

    def evaluate(
        self,
        vertex_global_id: int,
        parameter: float,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[np.ndarray, tuple[tuple[int, int, np.ndarray], ...]]:
        """Evaluate a retained curve parameter and all its source uses."""
        self.support.require_current()
        rows = np.flatnonzero(np.asarray(self.vertex_global_ids) == vertex_global_id)
        if rows.size != 1:
            raise ValueError(
                "SurfaceCurveWitness.vertex_global_id must name a retained source vertex; got an absent vertex ID"
            )
        row = int(rows[0])
        if int(self.source_dimensions[row]) != 1:
            raise ValueError(
                "SurfaceCurveWitness.vertex_stratum must be a curve stratum for sliding; got a noncurve vertex"
            )
        curve = int(self.source_indices[row])
        return self.evaluate_curve(curve, parameter, budget=budget)

    def evaluate_curve(
        self,
        curve: int,
        parameter: float,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[np.ndarray, tuple[tuple[int, int, np.ndarray], ...]]:
        """Evaluate source-bound born nodes as well as retained source vertices."""

        self.support.require_current()
        if not 0 <= curve < len(self.support.domain.curves):
            raise ValueError(
                "SurfaceCurveWitness.curve must name an authored source curve; got an absent curve index"
            )
        domain = self.support.domain
        owner = curve_owner(domain, curve)
        value = float(parameter)
        if not np.isfinite(value) or not min(owner.first, owner.last) <= value <= max(
            owner.first, owner.last
        ):
            raise ValueError(
                "SurfaceCurveWitness.parameter must lie in the authored interval; got out-of-interval proposal"
            )
        fraction = (value - owner.first) / (owner.last - owner.first)
        uses = domain.curve_uses(curve)
        budget.reserve_geometry_queries(1 + len(uses))
        point = np.asarray(
            domain.curve_atlas.map(
                jnp.asarray((curve,), dtype=jnp.int32),
                jnp.asarray(((fraction,),), dtype=jnp.float64),
            )
        )[0]
        charts = tuple(
            (
                patch,
                occurrence,
                np.asarray(
                    use.charts(
                        np.asarray((value,), dtype=np.float64),
                    )
                )[0],
            )
            for occurrence, (patch, use) in enumerate(uses)
        )
        if not np.all(np.isfinite(point)) or any(
            not np.all(np.isfinite(chart)) for _, _, chart in charts
        ):
            raise ValueError(
                "SurfaceCurveWitness.proposal must contain finite source curve and chart values; got nonfinite proposal"
            )
        return point, charts

    @checked
    def edge_interval(
        self: PreparedSurfaceCurveWitness,
        dimensions: np.ndarray,
        indices: np.ndarray,
        parameters: np.ndarray,
        /,
    ) -> tuple[int, float, float]:
        """Return a source curve interval, refusing ambiguous or collapsed endpoints."""
        result = self._resolve_edge_interval(dimensions, indices, parameters)
        if result.interval is None:
            raise ValueError(result.reason)
        return result.interval

    @checked
    def admissible_edge_interval(
        self: PreparedSurfaceCurveWitness,
        dimensions: np.ndarray,
        indices: np.ndarray,
        parameters: np.ndarray,
        /,
    ) -> tuple[int, float, float] | None:
        """Return an eligible source interval, or None for unsupported endpoint tokens."""
        return self._resolve_edge_interval(dimensions, indices, parameters).interval

    def _resolve_edge_interval(
        self, dimensions: np.ndarray, indices: np.ndarray, parameters: np.ndarray, /
    ) -> _CurveIntervalResolution:
        """Resolve two retained source-stratum tokens, including born nodes.

        Junction/corner-only ambiguity is refused instead of selecting a curve
        by a native feature code. Closed seam endpoints need an interval side.
        """
        kinds, rows, values = (
            np.asarray(dimensions),
            np.asarray(indices),
            np.asarray(parameters),
        )
        if kinds.shape != (2,) or rows.shape != (2,) or values.shape != (2, 2):
            return _CurveIntervalResolution(
                interval=None,
                reason="SurfaceCurveWitness.edge_tokens must contain two source-stratum tokens; got misaligned token arrays",
            )
        curves = {
            int(index) for kind, index in zip(kinds, rows, strict=True) if kind == 1
        }
        if not curves and np.all(kinds == 0) and (rows[0] != rows[1]):
            curves = {
                curve
                for curve, authority in enumerate(self.support.domain.curves)
                if {authority.start, authority.end} == {int(rows[0]), int(rows[1])}
            }
        if len(curves) != 1 or np.any((kinds != 0) & (kinds != 1)):
            return _CurveIntervalResolution(
                interval=None,
                reason="SurfaceCurveWitness.edge_interval must identify one authored curve; got ambiguous or unsupported endpoint strata",
            )
        curve = curves.pop()
        left_result = _resolve_curve_parameter(
            self.support,
            curve,
            int(kinds[0]),
            int(rows[0]),
            values[0],
            float(values[1, 0]) if kinds[1] == 1 else None,
        )
        if left_result.parameter is None:
            return _CurveIntervalResolution(interval=None, reason=left_result.reason)
        left = left_result.parameter
        right_result = _resolve_curve_parameter(
            self.support, curve, int(kinds[1]), int(rows[1]), values[1], left
        )
        if right_result.parameter is None:
            return _CurveIntervalResolution(interval=None, reason=right_result.reason)
        right = right_result.parameter
        owner = curve_owner(self.support.domain, curve)
        if (
            not all(
                (
                    np.isfinite(value)
                    and min(owner.first, owner.last)
                    <= value
                    <= max(owner.first, owner.last)
                    for value in (left, right)
                )
            )
            or left == right
        ):
            return _CurveIntervalResolution(
                interval=None,
                reason="SurfaceCurveWitness.edge_interval must be noncollapsed and lie in the curve interval; got collapsed or out-of-interval endpoints",
            )
        return _CurveIntervalResolution(interval=(curve, left, right), reason="")


@checked
def prepare_surface_curve_witness(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PreparedSurfaceCurveWitness:
    """Lower source associations without inventing a curve identity."""
    vertex, _ = source_associations(support, source, budget=budget)
    ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    rows = vertex.target_rows(ids)
    dimensions = np.asarray(vertex.source_dimensions)[rows]
    indices = np.asarray(
        [
            support.domain_row(
                support.domain,
                int(vertex.source_dimensions[row]),
                int(vertex.source_indices[row]),
                vertex.source_entities.occurrence_path(row),
            )
            for row in rows
        ],
        dtype=np.int32,
    )
    return PreparedSurfaceCurveWitness(
        support=support,
        topology_id=source.mesh.topology_id,
        source_mesh=source.mesh,
        source_association=vertex,
        vertex_global_ids=jnp.asarray(ids),
        source_dimensions=jnp.asarray(dimensions),
        source_indices=jnp.asarray(indices),
        source_parameters=vertex.parameters[rows],
        source_entity_ids=tuple(vertex.source_entities.entity_id(row) for row in rows),
        source_occurrence_paths=tuple(
            vertex.source_entities.occurrence_path(row) for row in rows
        ),
    )


def _chart_association_rows(
    mesh: cell_mesh_owner.CellMesh,
    witness: SurfaceChartWitness,
    bounds: np.ndarray,
    /,
) -> SurfaceChartAssociationRows:

    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    witness_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    rows = key_rows(witness_ids[:, None], ids[:, None])
    if witness_ids.size != ids.size or np.any(rows < 0):
        raise ValueError(
            "SurfaceAssociation.charts must bind every mesh cell ID; got different chart cell IDs"
        )
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = tuple(
        tuple(tuple(Fraction(float(value)) for value in point) for point in cell)
        for cell in np.asarray(witness.charts)[rows]
    )
    return SurfaceChartAssociationRows(
        ids,
        np.asarray(witness.patches, dtype=np.int64)[rows],
        tuple(tuple(int(value) for value in row) for row in vertices[cells]),
        corners,
        bounds,
    )


def _sphere_corner_stratum(
    atlas: SphereMaterialCellAtlas,
    row: int,
    local: int,
    /,
) -> tuple[int, int, np.ndarray, tuple[Fraction, Fraction]]:
    """Read authored strata; radial references are never called UV charts."""

    dimension, source_index, path, exact_parameters = atlas.corner_tokens[row][local]
    domain = atlas.domain
    index = PreparedSurfaceSourceSupport.domain_row(domain, dimension, source_index, path)
    parameters = np.zeros((2,), dtype=np.float64)
    parameters[: len(exact_parameters)] = tuple(
        float(value) for value in exact_parameters
    )
    patch = int(np.asarray(atlas.patches)[row])
    if dimension == 0:
        poles = [
            use
            for loop in domain.patches[patch].loops
            for use in loop
            if isinstance(use, PatchPoleUse) and use.corner == index
        ]
        if not poles:
            raise ValueError(
                "SurfaceAssociation.sphere_pole must retain an authored pole use; got missing pole use"
            )
        uv = np.asarray(poles[0].start, dtype=np.float64)
    elif dimension == 1:
        uses = [use for owner, use in domain.curve_uses(index) if owner == patch]
        if not uses:
            raise ValueError(
                "SurfaceAssociation.sphere_seam must retain an authored source curve use; got missing seam use"
            )
        uv = uses[0].charts(np.asarray((parameters[0],), dtype=np.float64))[0]
    elif dimension == 2:
        if index != patch:
            raise ValueError(
                "SurfaceAssociation.sphere_patch must preserve the source patch; got changed sphere patch"
            )
        uv = parameters
    else:
        raise ValueError(
            "SurfaceAssociation.sphere_strata must use source dimensions zero through two; got unsupported source dimension"
        )
    return dimension, index, parameters, (Fraction(float(uv[0])), Fraction(float(uv[1])))


def _sphere_chart_association_rows(
    mesh: cell_mesh_owner.CellMesh,
    atlas: SphereMaterialCellAtlas,
    bounds: np.ndarray,
    /,
) -> SurfaceChartAssociationRows:
    """Authored parameter representatives for metadata, not a UV coverage proof."""
    atlas.require_bound(
        atlas.domain, mesh, atlas.source_geometry, atlas.coordinate_contract
    )
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    global_vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[cells]
    rows = [atlas.cell_row(int(identifier)) for identifier in ids]
    return SurfaceChartAssociationRows(
        ids,
        np.asarray(atlas.patches, dtype=np.int64)[rows],
        tuple(tuple(int(value) for value in vertices) for vertices in global_vertices),
        tuple(
            tuple(_sphere_corner_stratum(atlas, row, local)[3] for local in range(3))
            for row in rows
        ),
        bounds,
    )


def _sphere_vertex_bindings(
    atlas: SphereMaterialCellAtlas,
    /,
) -> dict[int, tuple[int, int, np.ndarray]]:
    values: dict[int, tuple[int, int, np.ndarray]] = {}
    vertex_ids = np.asarray(atlas.physical_corner_global_ids, dtype=np.int64)
    for row in range(atlas.num_charts):
        for local in range(3):
            identifier = int(vertex_ids[row, local])
            dimension, index, parameters, _ = _sphere_corner_stratum(atlas, row, local)
            previous = values.get(int(identifier))
            if previous is not None and (
                previous[:2] != (dimension, index)
                or not np.array_equal(previous[2], parameters)
            ):
                raise ValueError(
                    "SurfaceAssociation.sphere_vertices must agree on source strata across occurrences; got inconsistent sphere vertex strata"
                )
            values[int(identifier)] = dimension, index, parameters
    return values


def _sphere_vertex_weights(
    deformation: PreparedSphereChartDeformation,
    maximum_visits: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> dict[int, surface_boundary_operations_owner.SurfaceVertexWeights]:
    """Return rational source-reference support from native pieces."""

    vertices = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    candidates: dict[
        int, tuple[int, surface_boundary_operations_owner.SurfaceVertexWeights]
    ] = {}
    visits = 0
    for piece in deformation.pieces:
        new_row = deformation.target_atlas.cell_row(piece.target_cell_global_id)
        old_row = deformation.source_atlas.cell_row(piece.source_cell_global_id)
        inverse = deformation.target_atlas.projective_reference_map(
            piece.target_cell_global_id,
            deformation.source_atlas,
            piece.source_cell_global_id,
            budget=budget,
        )
        arguments = inverse.reference_expressions()
        triangle = piece.exact_target_reference_vertices
        sign = area(triangle)
        for local, point in enumerate(vertices):
            visits += 1
            if visits > maximum_visits:
                raise AssociationPropagationError(
                    "SurfaceAssociation.sphere_queries must fit the source query allowance; got exhausted query allowance",
                    np.asarray((piece.target_cell_global_id,), dtype=np.int64),
                )
            if any(
                area((triangle[index], triangle[(index + 1) % 3], point)) * sign < 0
                for index in range(3)
            ):
                continue
            u, v = (
                require_rational_reference_scalar(
                    expression_evaluate(argument, point, budget=budget)
                )
                for argument in arguments
            )
            barycentric = (1 - u - v, u, v)
            if any(value < 0 for value in barycentric):
                raise ValueError(
                    "SurfaceAssociation.sphere_functional must lie in its source reference cell; got off-cell birth functional"
                )
            identifier = int(
                np.asarray(deformation.target_atlas.physical_corner_global_ids)[
                    new_row, local
                ]
            )
            parents = np.asarray(deformation.source_atlas.physical_corner_global_ids)[
                old_row
            ]
            weights = tuple(
                (int(parent), weight)
                for parent, weight in zip(parents, barycentric, strict=True)
                if weight
            )
            previous = candidates.get(identifier)
            if previous is None or piece.source_cell_global_id < previous[0]:
                candidates[identifier] = piece.source_cell_global_id, weights
    target_ids = np.asarray(deformation.target_mesh.vertex_global_ids, dtype=np.int64)
    if set(candidates) != set(target_ids.tolist()):
        raise ValueError(
            "SurfaceAssociation.sphere_pieces must support all target vertex functionals; got missing sphere support"
        )
    return {identifier: value[1] for identifier, value in candidates.items()}


def _surface_vertex_weights(
    deformation: PreparedSurfaceChartDeformation, /
) -> dict[int, surface_boundary_operations_owner.SurfaceVertexWeights]:
    """Read vertex functionals from retained rational material overlap pieces."""
    references = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )

    def cells(mesh: cell_mesh_owner.CellMesh, /) -> dict[int, tuple[int, int, int]]:
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        result: dict[int, tuple[int, int, int]] = {}
        for block in mesh.blocks:
            for identifier, vertices in zip(
                np.asarray(block.global_ids),
                np.asarray(block.vertices),
                strict=True,
            ):
                values = vertex_ids[vertices]
                if values.shape != (3,):
                    raise ValueError(
                        "SurfaceAssociation.material_cells must remain triangles; got different cell kind"
                    )
                result[int(identifier)] = (
                    int(values[0]),
                    int(values[1]),
                    int(values[2]),
                )
        return result

    source_cells = cells(deformation.source_mesh)
    target_cells = cells(deformation.target_mesh)
    candidates: dict[
        int, tuple[int, surface_boundary_operations_owner.SurfaceVertexWeights]
    ] = {}
    for occurrence in deformation.occurrences:
        for piece in occurrence.pieces:
            source_vertices = source_cells[piece.source_cell_global_id]
            target_vertices = target_cells[piece.target_cell_global_id]
            for local, reference in enumerate(references):
                slots = [
                    slot
                    for slot, point in enumerate(piece.exact_target_reference_vertices)
                    if point == reference
                ]
                if not slots:
                    continue
                source_point = piece.exact_source_reference_vertices[slots[0]]
                u, v = source_point
                barycentric = (1 - u - v, u, v)
                if any(value < 0 for value in barycentric):
                    raise ValueError(
                        "SurfaceAssociation.material_functional must lie in its source material cell; got off-cell birth functional"
                    )
                weights = tuple(
                    (identifier, weight)
                    for identifier, weight in zip(
                        source_vertices, barycentric, strict=True
                    )
                    if weight
                )
                target_vertex = target_vertices[local]
                previous = candidates.get(target_vertex)
                if previous is None or piece.source_cell_global_id < previous[0]:
                    candidates[target_vertex] = (piece.source_cell_global_id, weights)
    target_ids = set(map(int, np.asarray(deformation.target_mesh.vertex_global_ids)))
    if set(candidates) != target_ids:
        raise ValueError(
            "SurfaceAssociation.material_pieces must support every target vertex functional; got missing material support"
        )
    return {identifier: value[1] for identifier, value in candidates.items()}


@checked
def source_associations(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[GeometryAssociation, tuple[int, ...]]:

    if source.coordinate_contract.spatial_id != support.coordinate_contract.spatial_id:
        raise ValueError(
            "SurfaceAssociation.coordinate_contract must preserve the source coordinate and unit contract; got different spatial ID"
        )
    levels = []
    vertex = None
    expected_kind = surface_source_association_kind(support.domain.source_kinds)
    for value in source.associations:
        if (
            value.association_kind is not expected_kind
            or value.source_id != support.domain.source_id
            or value.source_revision != support.domain.source_revision
            or not value.complete
        ):
            raise ValueError(
                "SurfaceAssociation.namespace must preserve source identity and association coverage; got different association namespace"
            )
        dimension = target_dimension(source.mesh, value)
        for row in range(value.target_global_ids.shape[0]):
            kind = int(value.source_dimensions[row])
            index = support.domain_row(
                support.domain,
                kind,
                int(value.source_indices[row]),
                value.source_entities.occurrence_path(row),
            )
            if value.source_entities.entity_id(row) != support.domain.entity_id(
                kind, index
            ):
                raise ValueError(
                    "SurfaceAssociation.entity_id must name its typed source stratum; got different association entity ID"
                )
        if dimension == 0:
            if vertex is not None:
                raise ValueError(
                    "SurfaceAssociation.tables must contain one vertex association; got multiple vertex associations"
                )
            vertex = value
        else:
            levels.append(dimension)
    if vertex is None or len(set(levels)) != len(levels) or 2 not in levels:
        raise ValueError(
            "SurfaceAssociation.tables must contain source vertex and cell associations; got missing or repeated tables"
        )
    vertex.target_rows(np.asarray(source.mesh.entity_set(0).entity_ids, dtype=np.int64))
    cells = association_for_dimension(source, 2)
    cells.target_rows(np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64))
    return vertex, tuple(sorted(levels))


def _codes(
    support: PreparedSurfaceSourceSupport, dimensions: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    offsets = np.cumsum((0, *(len(row) for row in support.domain.source_indices[:-1])))
    if np.any(dimensions < 0) or np.any(indices < 0):
        raise ValueError(
            "SurfaceAssociation.strata must identify one source stratum; got negative dimension or source index"
        )
    return offsets[dimensions] + indices


@checked
def classes(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[SurfaceEntityClasses, ...]:
    source_associations(support, source, budget=budget)
    mesh = source.mesh
    result = []
    associations = association_lookup(source)
    for dimension in range(3):
        identifiers = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        dims = np.full(identifiers.shape, -1, dtype=np.int64)
        indices = np.full(identifiers.shape, -1, dtype=np.int64)
        if dimension == 1:
            links = incidence_pairs(mesh, 1, 2)
            cell = associations[2]
            cell_rows = cell.target_rows(
                np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
            )
            patch_ids = np.asarray(
                [
                    support.domain_row(
                        support.domain,
                        int(cell.source_dimensions[row]),
                        int(cell.source_indices[row]),
                        cell.source_entities.occurrence_path(row),
                    )
                    for row in cell_rows
                ],
                dtype=np.int64,
            )
            for edge in range(identifiers.size):
                adjacent = links[links[:, 0] == edge, 1]
                patches = np.unique(patch_ids[adjacent])
                if patches.size == 1:
                    dims[edge], indices[edge] = 2, patches[0]
        value = associations.get(dimension)
        if value is not None:
            target_rows = entity_rows(
                mesh, dimension, np.asarray(value.target_global_ids, dtype=np.int64)
            )
            if np.any(target_rows < 0):
                raise ValueError(
                    "SurfaceAssociation.target_ids must name mesh entities; got unavailable mesh entity IDs"
                )
            for source_row, row in enumerate(target_rows.tolist()):
                kind = int(value.source_dimensions[source_row])
                dims[row] = kind
                indices[row] = support.domain_row(
                    support.domain,
                    kind,
                    int(value.source_indices[source_row]),
                    value.source_entities.occurrence_path(source_row),
                )
        if dimension == 2 and np.any(dims != 2):
            raise ValueError(
                "SurfaceAssociation.cells must retain patch strata; got a nonpatch cell stratum"
            )
        if dimension == 1 and np.any(dims < 1):
            raise ValueError(
                "SurfaceAssociation.edges must retain curve or patch support; got unresolved coedge stratum"
            )
        result.append(
            SurfaceEntityClasses(
                dims,
                indices,
                np.ones(dims.shape, dtype=np.bool_),
                _codes(support, dims, indices),
            )
        )
    return tuple(result)


@checked
def protected_edges(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    /,
    *,
    midpoint_required: bool = True,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    source_associations(support, source, budget=budget)
    links = incidence_pairs(source.mesh, 0, 1)
    identifiers = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
    count = source.mesh.entity_set(1).count
    protected = np.zeros(count, dtype=np.bool_)
    for edge in range(count):
        vertices = links[links[:, 1] == edge, 0]
        if vertices.size != 2 or identifiers[vertices[0]] == identifiers[vertices[1]]:
            raise ValueError(
                "SurfaceAssociation.edges must have two distinct vertex identities; got collapsed or invalid edge incidence"
            )
        if midpoint_required:
            point = np.sum(coordinates[vertices] * np.float64(0.5), axis=0)
            protected[edge] = np.array_equal(
                point, coordinates[vertices[0]]
            ) or np.array_equal(point, coordinates[vertices[1]])
    # Source homotopy and reevaluated curve residuals remain attached to curved
    # feature chords through native subdivision.
    return protected


def _target_uv(
    rows: SurfaceRestrictionRows | SurfaceChartAssociationRows, vertex_id: int, patch: int
) -> np.ndarray:
    candidates = []
    for cell_id, cell_patch, vertices, charts in zip(
        rows.cell_ids.tolist(),
        rows.patches.tolist(),
        rows.vertex_ids,
        rows.source_corners,
        strict=True,
    ):
        if cell_patch != patch or vertex_id not in vertices:
            continue
        candidates.append((cell_id, charts[vertices.index(vertex_id)]))
    if not candidates:
        raise AssociationPropagationError(
            "SurfaceAssociation.charts must retain a source chart restriction for every vertex; got a missing vertex chart",
            np.asarray((vertex_id,), dtype=np.int64),
        )
    # Source seam/pole copies are chart gauges. The selected cell ID defines
    # the numerical representative without welding coordinate occurrences.
    return np.asarray(
        tuple(float(value) for value in min(candidates, key=lambda value: value[0])[1]),
        dtype=np.float64,
    )


def _resolve_curve_parameter(
    support: PreparedSurfaceSourceSupport,
    curve: int,
    kind: int,
    index: int,
    parameters: np.ndarray,
    other: float | None,
) -> _CurveParameterResolution:
    if kind == 1:
        if index != curve:
            return _CurveParameterResolution(
                parameter=None,
                reason="SurfaceCurve.parameter must retain one source curve; got different endpoint curves",
            )
        return _CurveParameterResolution(parameter=float(parameters[0]), reason="")
    if kind != 0:
        return _CurveParameterResolution(
            parameter=None,
            reason="SurfaceCurve.endpoint must lie in its curve closure; got unsupported endpoint stratum",
        )
    authority = support.domain.curves[curve]
    owner = curve_owner(support.domain, curve)
    first, last = (owner.first, owner.last)
    if index == authority.start and index == authority.end:
        if other is None or (other - first) / (last - first) == 0.5:
            return _CurveParameterResolution(
                parameter=None,
                reason="SurfaceCurve.endpoint must retain an unambiguous closed-curve interval side; got missing interval-side parameter",
            )
        return _CurveParameterResolution(
            parameter=first if (other - first) / (last - first) < 0.5 else last, reason=""
        )
    if index == authority.start:
        return _CurveParameterResolution(parameter=first, reason="")
    if index == authority.end:
        return _CurveParameterResolution(parameter=last, reason="")
    return _CurveParameterResolution(
        parameter=None,
        reason="SurfaceCurve.endpoint must name a corner of its source curve; got an unrelated corner",
    )


@checked
def curve_parameter(
    support: PreparedSurfaceSourceSupport,
    curve: int,
    kind: int,
    index: int,
    parameters: np.ndarray,
    other: float | None,
) -> float:
    result = _resolve_curve_parameter(support, curve, kind, index, parameters, other)
    if result.parameter is None:
        raise ValueError(result.reason)
    return result.parameter


def _residual(
    support: PreparedSurfaceSourceSupport,
    kind: int,
    index: int,
    parameters: np.ndarray,
    point: np.ndarray,
) -> float:
    if kind == 0:
        value = support.domain.corner_points[index]
    elif kind == 1:
        owner = curve_owner(support.domain, index)
        fraction = (float(parameters[0]) - owner.first) / (owner.last - owner.first)
        value = np.asarray(
            support.domain.curve_atlas.map(
                jnp.asarray((index,), dtype=jnp.int32),
                jnp.asarray(((fraction,),), dtype=jnp.float64),
            )
        )[0]
    else:
        value = support.domain.evaluate(
            np.asarray((index,), dtype=np.int64), parameters[None, :2]
        )[0]
    return float(np.linalg.norm(np.asarray(value) - point))


def _curve_parameter_from_functional(
    support: PreparedSurfaceSourceSupport,
    source: cell_mesh_owner.CellMesh,
    vertex: GeometryAssociation,
    classes: SurfaceEntityClasses,
    vertex_rows: np.ndarray,
    curve: int,
    key: surface_boundary_operations_owner.SurfaceVertexWeights,
    /,
) -> float:
    """Retain an chord's parameter functional and its nonzero residual."""
    if len(key) not in (1, 2):
        raise ValueError(
            "SurfaceAssociation.feature_functional must retain one endpoint or two edge endpoints; got unsupported functional support size"
        )
    identifiers = np.asarray([identifier for identifier, _ in key], dtype=np.int64)
    rows = entity_rows(source, 0, identifiers)
    if np.any(rows < 0):
        raise ValueError(
            "SurfaceAssociation.feature_functional must reference declared source vertices; got an absent source vertex"
        )
    values = np.asarray(vertex.parameters)[vertex_rows[rows]]
    if len(key) == 1:
        row = int(rows[0])
        parameter = curve_parameter(
            support,
            curve,
            int(classes.dimensions[row]),
            int(classes.indices[row]),
            values[0],
            None,
        )
    else:
        a, b = map(int, rows)
        first_hint = float(values[1, 0]) if classes.dimensions[b] == 1 else None
        last_hint = float(values[0, 0]) if classes.dimensions[a] == 1 else None
        first = curve_parameter(
            support,
            curve,
            int(classes.dimensions[a]),
            int(classes.indices[a]),
            values[0],
            first_hint,
        )
        last = curve_parameter(
            support,
            curve,
            int(classes.dimensions[b]),
            int(classes.indices[b]),
            values[1],
            last_hint,
        )
        parameter = float(key[0][1] * Fraction(first) + key[1][1] * Fraction(last))
    owner = curve_owner(support.domain, curve)
    if not np.isfinite(parameter) or not min(owner.first, owner.last) <= parameter <= max(
        owner.first, owner.last
    ):
        raise ValueError(
            "SurfaceAssociation.feature_parameter must lie in the authored curve interval; got out-of-interval parameter"
        )
    return parameter


@checked
def source_proof(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    receipts: SurfaceSourceReceipts | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> SurfaceRestrictionRows:
    """Root-identity proof of the transfer source, through its certified serial chain if any."""
    mesh = source.mesh
    if (
        mesh.topology_id == support.root_topology_id
        or mesh.storage is not None
        or source.geometry.restriction_source is not None
    ):
        return support.prove_native_restrictions(
            mesh, source.geometry, receipts=receipts, budget=budget
        )
    chain = None if source.certification is None else source.certification.request.source
    if (
        not isinstance(
            chain, surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource
        )
        or chain.support.support_id != support.support_id
        or chain.targets[-1].topology_id != mesh.topology_id
    ):
        raise ValueError(
            "SurfaceAssociation.subdivision must retain the certified root lineage chain; got missing or differently bound subdivision"
        )
    return subdivision_proof(
        support, chain.lineages, chain.targets, source.geometry, budget=budget
    )[0]


class _SurfaceProofPhase(StrictModule):
    source: SurfaceRestrictionRows | SurfaceChartAssociationRows
    target: SurfaceRestrictionRows | SurfaceChartAssociationRows
    weights: dict[int, surface_boundary_operations_owner.SurfaceVertexWeights]
    sphere_bindings: dict[int, tuple[int, int, np.ndarray]] | None


class _SurfaceVertexPhase(StrictModule):
    identifiers: np.ndarray
    dimensions: np.ndarray
    indices: np.ndarray
    parameters: np.ndarray
    parent_dimensions: np.ndarray
    parent_ids: np.ndarray
    source_rows: np.ndarray


class _SurfaceEntityPhase(StrictModule):
    identifiers: np.ndarray
    dimensions: np.ndarray
    indices: np.ndarray
    parameters: np.ndarray
    residuals: np.ndarray
    parent_dimensions: np.ndarray
    parent_ids: np.ndarray
    source_ids: np.ndarray
    target_ids: np.ndarray


def _prepare_native_surface_proof(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    lineage: MeshLineage,
    target: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    receipts: SurfaceSourceReceipts | None,
    budget: CoordinateEnclosureBudget,
) -> _SurfaceProofPhase:
    local_source_proof = source_proof(support, source, receipts, budget=budget)
    weights = created_weights(lineage, source.mesh, target)
    serial = target.storage is None and geometry.restriction_source is None
    ancestry = (
        lineage_ancestry(local_source_proof, lineage, target, weights) if serial else None
    )
    target_proof = support.prove_native_restrictions(
        target,
        geometry,
        receipts=receipts,
        ancestry=ancestry,
        budget=budget,
    )
    source_roots = dict(
        zip(
            local_source_proof.cell_ids.tolist(),
            local_source_proof.root_cell_ids.tolist(),
            strict=True,
        )
    )
    target_roots = dict(
        zip(
            target_proof.cell_ids.tolist(),
            target_proof.root_cell_ids.tolist(),
            strict=True,
        )
    )
    cell_record = lineage.entity_lineage(2)
    kinds = np.asarray(cell_record.relation_kinds, dtype=np.int32)
    allowed = (
        EntityLineageKind.PRESERVED,
        EntityLineageKind.REFINED_FROM,
        EntityLineageKind.SPLIT_FROM,
        EntityLineageKind.COARSENED_INTO,
        EntityLineageKind.MERGED_INTO,
    )
    if np.any(~np.isin(kinds, tuple(int(value) for value in allowed))):
        raise AssociationPropagationError(
            "SurfaceAssociation.lineage must contain supported source-preserving cell relations; got unknown or remeshed relation",
            np.asarray(cell_record.target_global_ids),
        )
    for parent, child in zip(
        np.asarray(cell_record.source_global_ids).tolist(),
        np.asarray(cell_record.target_global_ids).tolist(),
        strict=True,
    ):
        if (
            parent not in source_roots
            or child not in target_roots
            or source_roots[parent] != target_roots[child]
        ):
            raise AssociationPropagationError(
                "SurfaceAssociation.lineage must preserve source root cell identity; got different source and target roots",
                np.asarray((child,), dtype=np.int64),
            )
    return _SurfaceProofPhase(
        source=local_source_proof,
        target=target_proof,
        weights=weights,
        sphere_bindings=None,
    )


def _prepare_chart_surface_proof(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    target: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    deformation: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation,
    budget: CoordinateEnclosureBudget,
) -> _SurfaceProofPhase:
    sphere_bindings = None
    deformation.require_bound(source.mesh, source.geometry, target, geometry)
    if deformation.domain_id != support.domain.domain_id:
        raise ValueError(
            "SurfaceAssociation.deformation must preserve the support domain ID; got different deformation domain"
        )
    if isinstance(deformation, PreparedSphereChartDeformation):
        local_source_proof = _sphere_chart_association_rows(
            source.mesh,
            deformation.source_atlas,
            np.asarray(deformation.source_fidelity_bounds),
        )
        target_proof = _sphere_chart_association_rows(
            target,
            deformation.target_atlas,
            np.asarray(deformation.target_fidelity_bounds),
        )
        weights = _sphere_vertex_weights(
            deformation, support.maximum_support_queries, budget=budget
        )
        sphere_bindings = _sphere_vertex_bindings(deformation.target_atlas)
    else:
        local_source_proof = _chart_association_rows(
            source.mesh,
            deformation.source_witness,
            np.asarray(deformation.source_fidelity_bounds),
        )
        target_proof = _chart_association_rows(
            target,
            deformation.target_witness,
            np.asarray(deformation.target_fidelity_bounds),
        )
        weights = _surface_vertex_weights(deformation)
    return _SurfaceProofPhase(
        source=local_source_proof,
        target=target_proof,
        weights=weights,
        sphere_bindings=sphere_bindings,
    )


class _CreatedSurfaceSupport(StrictModule):
    dimension: int = eqx.field(static=True)
    row: int = eqx.field(static=True)
    identifier: int = eqx.field(static=True)


def _prepare_created_vertex_support(
    proof: _SurfaceProofPhase,
    target_ids: np.ndarray,
    row: int,
    source_dimension: int,
    source_row: int,
    transported: bool,
) -> _CreatedSurfaceSupport:
    local_source_proof, weights = proof.source, proof.weights
    if transported and (source_dimension < 1 or source_row < 0):
        parents = {parent for parent, _ in weights[int(target_ids[row])]}
        candidates = [
            cell
            for cell, vertices in enumerate(local_source_proof.vertex_ids)
            if parents <= set(vertices)
        ]
        if candidates:
            source_dimension, source_row = 2, candidates[0]
    identifier = int(target_ids[row])
    if (
        source_dimension not in (1, 2)
        or source_row < 0
        or (not transported and len(weights.get(identifier, ())) != source_dimension + 1)
    ):
        raise AssociationPropagationError(
            "SurfaceAssociation.created_vertices must retain source edge or cell lineage support; got missing source support",
            target_ids[row : row + 1],
        )
    return _CreatedSurfaceSupport(
        dimension=source_dimension, row=source_row, identifier=identifier
    )


def _prepare_created_surface_vertices(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    lineage: MeshLineage,
    target: cell_mesh_owner.CellMesh,
    vertex: GeometryAssociation,
    old: tuple[SurfaceEntityClasses, ...],
    proof: _SurfaceProofPhase,
    deformation: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation | None,
    phase: _SurfaceVertexPhase,
    previous_rows: np.ndarray,
) -> _SurfaceVertexPhase:
    target_proof = proof.target
    weights, sphere_bindings = proof.weights, proof.sphere_bindings
    target_ids, vertex_rows = phase.identifiers, phase.source_rows
    dims, indices, parameters = phase.dimensions, phase.indices, phase.parameters
    parent_dims, parent_ids = phase.parent_dimensions, phase.parent_ids
    created_rows = np.flatnonzero(previous_rows < 0)
    if created_rows.size:
        ordering = np.argsort(target_ids[created_rows], kind="stable")
        created_rows = created_rows[ordering]
        source_dims, source_rows = child_sources(
            lineage, source.mesh, target, target_ids[created_rows]
        )
        for row, source_dimension, source_row in zip(
            created_rows.tolist(), source_dims.tolist(), source_rows.tolist(), strict=True
        ):
            source_support = _prepare_created_vertex_support(
                proof,
                target_ids,
                row,
                source_dimension,
                source_row,
                deformation is not None,
            )
            source_dimension, source_row = source_support.dimension, source_support.row
            identifier = source_support.identifier
            dims[row], indices[row] = (
                old[source_dimension].dimensions[source_row],
                old[source_dimension].indices[source_row],
            )
            parent_dims[row] = source_dimension
            parent_ids[row] = np.asarray(
                source.mesh.entity_set(source_dimension).entity_ids
            )[source_row]
            if sphere_bindings is not None:
                dims[row], indices[row], parameters[row] = sphere_bindings[identifier]
                continue
            if dims[row] == 1:
                curve = int(indices[row])
                parameters[row, 0] = _curve_parameter_from_functional(
                    support,
                    source.mesh,
                    vertex,
                    old[0],
                    vertex_rows,
                    curve,
                    weights[identifier],
                )
            elif dims[row] == 2:
                parameters[row] = _target_uv(
                    target_proof, int(target_ids[row]), int(indices[row])
                )
            else:
                raise AssociationPropagationError(
                    "SurfaceAssociation.created_vertices must retain curve or patch support; got a new corner stratum",
                    target_ids[row : row + 1],
                )
    return phase


def _prepare_surface_vertices(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    lineage: MeshLineage,
    target: cell_mesh_owner.CellMesh,
    vertex: GeometryAssociation,
    old: tuple[SurfaceEntityClasses, ...],
    proof: _SurfaceProofPhase,
    deformation: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation | None,
) -> _SurfaceVertexPhase:
    target_proof = proof.target
    sphere_bindings = proof.sphere_bindings
    target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    vertex_rows = vertex.target_rows(
        np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    )
    previous_rows = entity_rows(source.mesh, 0, target_ids)
    dims = np.full(target_ids.shape, -1, dtype=np.int64)
    indices = np.full(target_ids.shape, -1, dtype=np.int64)
    parameters = np.zeros((target_ids.size, 2), dtype=np.float64)
    parent_dims = np.full(target_ids.shape, -1, dtype=np.int8)
    parent_ids = np.full(target_ids.shape, -1, dtype=np.int64)
    for row in np.flatnonzero(previous_rows >= 0).tolist():
        parent = int(previous_rows[row])
        original = int(vertex_rows[parent])
        if deformation is None:
            if not np.array_equal(
                np.asarray(target.coordinates)[row],
                np.asarray(source.mesh.coordinates)[parent],
            ):
                raise AssociationPropagationError(
                    "SurfaceAssociation.preserved_vertices must preserve the source coordinate representative; got changed preserved coordinates",
                    target_ids[row : row + 1],
                )
        dims[row], indices[row] = old[0].dimensions[parent], old[0].indices[parent]
        parameters[row] = (
            _target_uv(target_proof, int(target_ids[row]), int(indices[row]))
            if deformation is not None and dims[row] == 2
            else np.asarray(vertex.parameters[original])[:2]
        )
        parent_dims[row], parent_ids[row] = 0, target_ids[row]
        if sphere_bindings is not None:
            kind, index, value = sphere_bindings[int(target_ids[row])]
            if (kind, index) != (int(dims[row]), int(indices[row])):
                raise AssociationPropagationError(
                    "SurfaceAssociation.preserved_vertices must preserve sphere source strata; got changed sphere stratum",
                    target_ids[row : row + 1],
                )
            parameters[row] = value
    phase = _SurfaceVertexPhase(
        identifiers=target_ids,
        dimensions=dims,
        indices=indices,
        parameters=parameters,
        parent_dimensions=parent_dims,
        parent_ids=parent_ids,
        source_rows=vertex_rows,
    )
    phase = _prepare_created_surface_vertices(
        support,
        source,
        lineage,
        target,
        vertex,
        old,
        proof,
        deformation,
        phase,
        previous_rows,
    )
    if np.any(dims < 0):
        raise AssociationPropagationError(
            "SurfaceAssociation.vertices must resolve every source stratum; got negative source dimension",
            target_ids[dims < 0],
        )
    return phase


def _certify_surface_features(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    vertex: GeometryAssociation,
    old: tuple[SurfaceEntityClasses, ...],
    proof: _SurfaceProofPhase,
    vertices: _SurfaceVertexPhase,
    deformation: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation | None,
) -> _SurfaceVertexPhase:
    target_proof, weights = proof.target, proof.weights
    dims, indices, parameters = vertices.dimensions, vertices.indices, vertices.parameters
    target_ids, vertex_rows = vertices.identifiers, vertices.source_rows
    if isinstance(deformation, PreparedSurfaceChartDeformation):
        for row in np.flatnonzero(dims == 1).tolist():
            curve = int(indices[row])
            incident = [
                cell
                for cell, vertices in enumerate(target_proof.vertex_ids)
                if int(target_ids[row]) in vertices
            ]
            patch = int(target_proof.patches[incident[0]])
            uses = [
                use for owner, use in support.domain.curve_uses(curve) if owner == patch
            ]
            if not uses:
                raise AssociationPropagationError(
                    "SurfaceAssociation.transported_features must retain source chart uses; got missing feature chart use",
                    target_ids[row : row + 1],
                )
            # The boundary chord retains its source functional; the curve
            # discrepancy is measured below and coverage retains the trim ribbon.
            parameter = _curve_parameter_from_functional(
                support,
                source.mesh,
                vertex,
                old[0],
                vertex_rows,
                curve,
                weights[int(target_ids[row])],
            )
            if not any(
                min(use.first, use.last) <= parameter <= max(use.first, use.last)
                for use in uses
            ):
                raise AssociationPropagationError(
                    "SurfaceAssociation.feature_parameter must lie in a retained curve use; got out-of-use feature parameter",
                    target_ids[row : row + 1],
                )
            parameters[row, 0] = parameter
    return vertices


def _publish_surface_vertices(
    support: PreparedSurfaceSourceSupport,
    target: cell_mesh_owner.CellMesh,
    vertex: GeometryAssociation,
    vertices: _SurfaceVertexPhase,
) -> GeometryAssociation:
    dims, indices, parameters = vertices.dimensions, vertices.indices, vertices.parameters
    target_ids = vertices.identifiers
    parent_dims, parent_ids = vertices.parent_dimensions, vertices.parent_ids
    residuals = np.asarray(
        [
            _residual(support, int(kind), int(index), value, point)
            for kind, index, value, point in zip(
                dims.tolist(),
                indices.tolist(),
                parameters,
                np.asarray(target.coordinates),
                strict=True,
            )
        ]
    )
    paths = tuple(
        support.domain.source_occurrences[int(kind)][int(index)]
        for kind, index in zip(dims, indices, strict=True)
    )
    labels = np.asarray(
        [
            support.domain.source_indices[int(kind)][int(index)]
            for kind, index in zip(dims, indices, strict=True)
        ],
        dtype=np.int64,
    )
    return GeometryAssociation(
        vertex.association_kind,
        support.domain.source_id,
        support.domain.source_revision,
        target.entity_set(0).entity_set_id,
        target_ids,
        tuple(
            support.domain.entity_id(int(kind), int(index))
            for kind, index in zip(dims, indices, strict=True)
        ),
        residuals,
        source_dimensions=dims,
        source_indices=labels,
        source_occurrence_paths=paths,
        parameters=parameters,
        exact=False,
        parent_dimensions=parent_dims,
        parent_ids=parent_ids,
        parent_association_id=vertex.association_id,
        provenance=GeometryAssociationProvenance.LINEAGE,
    )


def _prepare_surface_entity(
    lineage: MeshLineage,
    target: cell_mesh_owner.CellMesh,
    dimension: int,
) -> _SurfaceEntityPhase:
    ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
    kinds = np.full(ids.shape, 2, dtype=np.int64)
    source_indices = np.full(ids.shape, -1, dtype=np.int64)
    params = np.zeros((ids.size, 2), dtype=np.float64)
    errors = np.zeros(ids.shape, dtype=np.float64)
    parent_dimensions = np.full(ids.shape, dimension, dtype=np.int8)
    parent_identifiers = np.full(ids.shape, -1, dtype=np.int64)
    record = lineage.entity_lineage(dimension)
    old_ids, new_ids = (
        np.asarray(record.source_global_ids, dtype=np.int64),
        np.asarray(record.target_global_ids, dtype=np.int64),
    )
    return _SurfaceEntityPhase(
        identifiers=ids,
        dimensions=kinds,
        indices=source_indices,
        parameters=params,
        residuals=errors,
        parent_dimensions=parent_dimensions,
        parent_ids=parent_identifiers,
        source_ids=old_ids,
        target_ids=new_ids,
    )


def _prepare_surface_cells(
    proof: SurfaceRestrictionRows | SurfaceChartAssociationRows,
    rows: _SurfaceEntityPhase,
) -> _SurfaceEntityPhase:
    target_proof = proof
    patch_by_cell = dict(
        zip(proof.cell_ids.tolist(), proof.patches.tolist(), strict=True)
    )
    bound_by_cell = dict(
        zip(proof.cell_ids.tolist(), proof.source_bounds.tolist(), strict=True)
    )
    ids, source_indices, params = rows.identifiers, rows.indices, rows.parameters
    errors, parent_identifiers = rows.residuals, rows.parent_ids
    old_ids, new_ids = rows.source_ids, rows.target_ids
    for row, identifier in enumerate(ids.tolist()):
        if identifier not in patch_by_cell:
            raise AssociationPropagationError(
                "SurfaceAssociation.cells must retain a source restriction; got a missing restriction row",
                ids[row : row + 1],
            )
        source_indices[row] = patch_by_cell[identifier]
        proof_row = int(np.flatnonzero(target_proof.cell_ids == identifier)[0])
        params[row] = np.asarray(
            tuple(
                float(
                    sum(
                        (point[axis] for point in target_proof.source_corners[proof_row]),
                        Fraction(0),
                    )
                    / 3
                )
                for axis in range(2)
            )
        )
        errors[row] = bound_by_cell[identifier]
        ancestors = old_ids[new_ids == identifier]
        if not ancestors.size:
            raise AssociationPropagationError(
                "SurfaceAssociation.cells must retain source-cell ancestry; got a cell without ancestors",
                ids[row : row + 1],
            )
        parent_identifiers[row] = np.min(ancestors)
    return rows


def _prepare_edge_parameters(
    support: PreparedSurfaceSourceSupport,
    target: cell_mesh_owner.CellMesh,
    target_proof: SurfaceRestrictionRows | SurfaceChartAssociationRows,
    vertices_phase: _SurfaceVertexPhase,
    rows: _SurfaceEntityPhase,
    row: int,
    adjacent: np.ndarray,
    links: np.ndarray,
    current_cells: np.ndarray,
    patch_by_cell: Mapping[int, int],
    proof_rows: Mapping[int, int],
) -> _SurfaceEntityPhase:
    ids, kinds, source_indices = rows.identifiers, rows.dimensions, rows.indices
    params, errors = rows.parameters, rows.residuals
    target_ids = vertices_phase.identifiers
    dims, indices, parameters = (
        vertices_phase.dimensions,
        vertices_phase.indices,
        vertices_phase.parameters,
    )
    vertices = links[links[:, 1] == row, 0]
    if vertices.size != 2:
        raise ValueError(
            "SurfaceAssociation.edge_vertices must retain two vertex identities; got different incidence count"
        )
    midpoint = np.sum(np.asarray(target.coordinates)[vertices] * 0.5, axis=0)
    if kinds[row] == 1:
        curve = int(source_indices[row])
        a, b = vertices.tolist()
        av = float(parameters[a, 0]) if dims[a] == 1 else None
        bv = float(parameters[b, 0]) if dims[b] == 1 else None
        first = curve_parameter(
            support, curve, int(dims[a]), int(indices[a]), parameters[a], bv
        )
        last = curve_parameter(
            support, curve, int(dims[b]), int(indices[b]), parameters[b], av
        )
        params[row, 0] = float((Fraction(first) + Fraction(last)) / 2)
        errors[row] = _residual(support, 1, curve, params[row], midpoint)
    else:
        patch = int(source_indices[row])
        # Seam and pole copies are chart gauges: both endpoints are
        # read in one adjacent cell's chart, never across gauges.
        chart_cells = sorted(
            int(current_cells[cell])
            for cell in adjacent
            if patch_by_cell[int(current_cells[cell])] == patch
        )
        if not chart_cells:
            raise AssociationPropagationError(
                "SurfaceAssociation.edge_chart must have an adjacent source patch chart; got missing adjacent chart",
                ids[row : row + 1],
            )
        chart_row = proof_rows[chart_cells[0]]
        corners = target_proof.vertex_ids[chart_row]
        charts = target_proof.source_corners[chart_row]
        ends = tuple(
            charts[corners.index(int(target_ids[vertex]))] for vertex in vertices.tolist()
        )
        params[row] = np.asarray(
            tuple(float((ends[0][axis] + ends[1][axis]) / 2) for axis in range(2)),
            dtype=np.float64,
        )
        errors[row] = _residual(support, 2, patch, params[row], midpoint)
    return rows


def _prepare_surface_edges(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    lineage: MeshLineage,
    target: cell_mesh_owner.CellMesh,
    old: tuple[SurfaceEntityClasses, ...],
    proof: SurfaceRestrictionRows | SurfaceChartAssociationRows,
    vertices_phase: _SurfaceVertexPhase,
    rows: _SurfaceEntityPhase,
) -> _SurfaceEntityPhase:
    target_proof = proof
    patch_by_cell = dict(
        zip(proof.cell_ids.tolist(), proof.patches.tolist(), strict=True)
    )
    cell_record = lineage.entity_lineage(2)
    ids, kinds, source_indices = rows.identifiers, rows.dimensions, rows.indices
    parent_dimensions, parent_identifiers = rows.parent_dimensions, rows.parent_ids
    old_ids, new_ids = rows.source_ids, rows.target_ids
    links = incidence_pairs(target, 0, 1)
    cell_links = incidence_pairs(target, 1, 2)
    current_cells = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)
    proof_rows = {
        cell: proof_row for proof_row, cell in enumerate(target_proof.cell_ids.tolist())
    }
    for row, identifier in enumerate(ids.tolist()):
        ancestors = old_ids[new_ids == identifier]
        ancestor_rows = (
            entity_rows(source.mesh, 1, ancestors)
            if ancestors.size
            else np.empty((0,), dtype=np.int64)
        )
        if np.any(ancestor_rows < 0):
            raise AssociationPropagationError(
                "SurfaceAssociation.edge_lineage must name declared source edges; got an absent source edge",
                ids[row : row + 1],
            )
        parent_classes = (
            np.unique(
                np.stack(
                    (
                        old[1].dimensions[ancestor_rows],
                        old[1].indices[ancestor_rows],
                    ),
                    axis=1,
                ),
                axis=0,
            )
            if ancestor_rows.size
            else np.empty((0, 2), dtype=np.int64)
        )
        if parent_classes.shape[0] == 1:
            kinds[row], source_indices[row] = parent_classes[0]
            parent_identifiers[row] = np.min(ancestors)
        adjacent = cell_links[cell_links[:, 0] == row, 1]
        if parent_classes.shape[0] != 1:
            patches = np.unique(
                [patch_by_cell[int(current_cells[cell])] for cell in adjacent]
            )
            if patches.size != 1:
                raise AssociationPropagationError(
                    "SurfaceAssociation.interface_edges must retain source-curve lineage or one source patch; got ambiguous interface support",
                    ids[row : row + 1],
                )
            source_indices[row] = patches[0]
            parent_dimensions[row] = 2
            source_cells = np.asarray(cell_record.source_global_ids, dtype=np.int64)
            target_cells = np.asarray(cell_record.target_global_ids, dtype=np.int64)
            parents = source_cells[np.isin(target_cells, current_cells[adjacent])]
            if not parents.size:
                raise AssociationPropagationError(
                    "SurfaceAssociation.interior_edges must retain source-cell lineage; got an edge without parent cells",
                    ids[row : row + 1],
                )
            parent_identifiers[row] = np.min(parents)
        rows = _prepare_edge_parameters(
            support,
            target,
            target_proof,
            vertices_phase,
            rows,
            row,
            adjacent,
            links,
            current_cells,
            patch_by_cell,
            proof_rows,
        )
    return rows


def _publish_surface_entity(
    support: PreparedSurfaceSourceSupport,
    target: cell_mesh_owner.CellMesh,
    vertex: GeometryAssociation,
    previous: GeometryAssociation,
    dimension: int,
    rows: _SurfaceEntityPhase,
) -> GeometryAssociation:
    ids, kinds, source_indices = rows.identifiers, rows.dimensions, rows.indices
    params, errors = rows.parameters, rows.residuals
    parent_dimensions, parent_identifiers = rows.parent_dimensions, rows.parent_ids
    if np.any(source_indices < 0):
        raise AssociationPropagationError(
            "SurfaceAssociation.entities must resolve every source index; got negative source indices",
            ids[source_indices < 0],
        )
    return GeometryAssociation(
        vertex.association_kind,
        support.domain.source_id,
        support.domain.source_revision,
        target.entity_set(dimension).entity_set_id,
        ids,
        tuple(
            support.domain.entity_id(int(kind), int(index))
            for kind, index in zip(kinds, source_indices, strict=True)
        ),
        errors,
        source_dimensions=kinds,
        source_indices=np.asarray(
            [
                support.domain.source_indices[int(kind)][int(index)]
                for kind, index in zip(kinds, source_indices, strict=True)
            ],
            dtype=np.int64,
        ),
        source_occurrence_paths=tuple(
            support.domain.source_occurrences[int(kind)][int(index)]
            for kind, index in zip(kinds, source_indices, strict=True)
        ),
        parameters=params,
        exact=False,
        parent_dimensions=parent_dimensions,
        parent_ids=parent_identifiers,
        parent_association_id=previous.association_id,
        provenance=GeometryAssociationProvenance.LINEAGE,
    )


def _supported_surface_deformation(
    deformation: MaterialChartDeformation | None,
) -> PreparedSurfaceChartDeformation | PreparedSphereChartDeformation | None:
    """Select the supported surface correspondence records."""
    match deformation:
        case None:
            return None
        case PreparedSurfaceChartDeformation() | PreparedSphereChartDeformation():
            return deformation
        case unsupported:
            raise TypeError(
                "SurfaceAssociation.deformation must be a surface or sphere correspondence; "
                f"got {type(unsupported).__name__}"
            )


@checked
def propagate(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    lineage: MeshLineage,
    target: cell_mesh_owner.CellMesh,
    /,
    *,
    geometry: cell_geometry_owner.CellGeometrySpec,
    receipts: SurfaceSourceReceipts | None = None,
    embedding: GlobalEmbeddingCertificate | None = None,
    coverage: DomainCoverageCertificate | None = None,
    deformation: MaterialChartDeformation | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[GeometryAssociation, ...]:
    deformation = _supported_surface_deformation(deformation)

    vertex, dimensions = source_associations(support, source, budget=budget)
    require_transfer_lineage(
        source.mesh,
        lineage,
        target,
        message="SurfaceAssociation.lineage must bind source and target topology IDs; got different topology lineage",
    )
    if target.storage is not None:
        if embedding is not None:
            embedding.binding.require(target, geometry)
            if embedding.status != "certified":
                raise ValueError(
                    "SurfaceAssociation.embedding must be certified for the collective target; got unresolved embedding"
                )
        if coverage is not None:
            coverage.binding.require(target, geometry)
            if coverage.status != "certified":
                raise ValueError(
                    "SurfaceAssociation.coverage must be certified for the collective target; got unresolved subdivision coverage"
                )
    old = classes(support, source, budget=budget)
    if deformation is None:
        proof = _prepare_native_surface_proof(
            support, source, lineage, target, geometry, receipts, budget
        )
    else:
        proof = _prepare_chart_surface_proof(
            support, source, target, geometry, deformation, budget
        )
    vertices = _prepare_surface_vertices(
        support, source, lineage, target, vertex, old, proof, deformation
    )
    vertices = _certify_surface_features(
        support, source, vertex, old, proof, vertices, deformation
    )
    output = [_publish_surface_vertices(support, target, vertex, vertices)]
    for dimension in dimensions:
        previous = association_for_dimension(source, dimension)
        rows = _prepare_surface_entity(lineage, target, dimension)
        match dimension:
            case 2:
                rows = _prepare_surface_cells(proof.target, rows)
            case 1:
                rows = _prepare_surface_edges(
                    support, source, lineage, target, old, proof.target, vertices, rows
                )
            case _:
                raise ValueError(
                    "SurfaceAssociation.dimension must be one or two for nonvertex entities; got unsupported entity dimension"
                )
        output.append(
            _publish_surface_entity(support, target, vertex, previous, dimension, rows)
        )
    return tuple(output)


@checked
def successor_certification_inputs(
    support: PreparedSurfaceSourceSupport | None,
    source: CellMeshingResult,
    inputs: MeshCertificationInputs,
    lineage: MeshLineage,
    target: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    deformation: MaterialChartDeformation | None = None,
    budget: CoordinateEnclosureBudget,
) -> MeshCertificationInputs:
    """Bind a serial successor's fidelity to its certified root chart subdivision chain.

    CAD association propagation supplies no surface support. Prepare it from the
    accepted chart root once, then retain it across affine subdivision epochs.
    Every other retained certification input is unchanged.
    """

    deformation = _supported_surface_deformation(deformation)
    retained = inputs.source
    if support is None:
        charts = source.surface_source
        if charts is None:
            return inputs
        match retained:
            case MeshingDomainBoundarySource():
                charts.require_root(source.mesh, source.geometry)
                support = PreparedSurfaceSourceSupport(
                    charts.domain,
                    source,
                    charts.root_parameters,
                    charts.boundary_source,
                    maximum_support_queries=charts.maximum_support_queries,
                    budget=budget,
                )
            case surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource():
                support = retained.support
            case _:
                return inputs
    if deformation is not None:
        match retained:
            case (
                surface_boundary_sources_owner.SurfaceChartBoundarySource()
                | SurfaceNativeRestrictionBoundarySource()
            ):
                root_candidate = retained.root
            case _:
                root_candidate = retained
        match root_candidate:
            case (
                MeshingDomainBoundarySource()
                | surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource()
            ) as root:
                pass
            case _:
                raise ValueError(
                    "SurfaceAssociation.certification.source must retain a continuous domain or subdivision boundary source; got unsupported source renewal"
                )
        if source.certification is None or source.certification.fidelity is None:
            raise ValueError(
                "SurfaceAssociation.certification.fidelity must retain predecessor fidelity evidence; got missing fidelity certificate"
            )
        chart_source = surface_boundary_sources_owner.SurfaceChartBoundarySource(
            root, support, deformation, source.certification.fidelity
        )
        return MeshCertificationInputs(
            source.mesh,
            source.geometry,
            inputs.schedule,
            domain=inputs.domain,
            cell_regions=inputs.cell_regions,
            source=chart_source,
            fidelity_tolerance=inputs.fidelity_tolerance,
            fidelity_sample_order=inputs.fidelity_sample_order,
            limits=inputs.limits,
            junction_vertices=inputs.junction_vertices,
            budget=budget,
        )
    if target.storage is not None:
        return inputs
    match retained:
        case MeshingDomainBoundarySource():
            subdivision = surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource(
                retained, support, (lineage,), (target,), geometry
            )
        case surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource() if (
            retained.support.support_id == support.support_id
        ):
            subdivision = surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource(
                retained.root,
                support,
                (*retained.lineages, lineage),
                (*retained.targets, target),
                geometry,
            )
        case _:
            return inputs
    return MeshCertificationInputs(
        source.mesh,
        source.geometry,
        inputs.schedule,
        domain=inputs.domain,
        cell_regions=inputs.cell_regions,
        source=subdivision,
        fidelity_tolerance=inputs.fidelity_tolerance,
        fidelity_sample_order=inputs.fidelity_sample_order,
        limits=inputs.limits,
        junction_vertices=inputs.junction_vertices,
        budget=budget,
    )


__all__ = [
    "SurfaceEntityClasses",
    "classes",
    "propagate",
    "protected_edges",
    "source_associations",
    "successor_certification_inputs",
    "PreparedSurfaceCurveWitness",
    "prepare_surface_curve_witness",
    "prepare_surface_curve_chart_cover",
]
