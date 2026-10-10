# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Lineage-only parametric surface support; no synthetic PLC/B-Rep authority."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._surface_chart_deformation import (
    PreparedSurfaceChartDeformation,
    SurfaceChartWitness,
)
from ..geometry._mesh_certificates import (
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    MeshCertificateFinding,
    SourceBoundaryChartCover,
    SourceBoundaryDistance,
    SourceBoundarySamples,
    SourceFidelityCertificate,
)
from ..geometry._meshing_domain import MeshingDomainBoundarySource
from ..geometry._surface_source_support import (
    _upper,
    curve_owner,
    PreparedSurfaceSourceSupport,
    SurfaceCellAncestry,
    SurfaceNativeRestrictionBoundarySource,
    SurfaceRestrictionRows,
    SurfaceSourceReceipts,
)
from ._association import (
    _child_sources,
    _entity_rows,
    _incidence_pairs,
    _surface_source_association_kind,
    _target_dimension,
    AssociationPropagationError,
    GeometryAssociation,
    GeometryAssociationProvenance,
)


if TYPE_CHECKING:
    from ..discretization._sphere_chart_deformation import PreparedSphereChartDeformation
    from ..geometry._sphere_material_atlas import SphereMaterialCellAtlas
    from ._certification_inputs import MeshCertificationInputs
    from ._lineage import MeshLineage
    from ._result import CellMeshingResult


class SurfaceEntityClasses(NamedTuple):
    """Authoritative original surface strata and occurrence-aware native codes."""

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


def prepare_surface_curve_chart_cover(
    witness: PreparedSurfaceCurveWitness,
    mesh: CellMesh,
    charts: SurfaceChartWitness,
    dimensions: np.ndarray,
    indices: np.ndarray,
    parameters: np.ndarray,
    /,
    *,
    source_chart_cover: MeshingDomainBoundarySource,
) -> MeshingDomainBoundarySource:
    """Retain actual current coedge parameters for the owning trim-ribbon proof.

    Born vertex tokens come from the source-bound controller, never a physical
    inverse or generic native feature label. These are evidence inputs: the
    chart-cover owner independently certifies continuous trims and ribbons.
    """
    from ..geometry._meshing_domain import PatchCurveUse

    witness.support.require_current()
    domain = witness.support.domain
    if (
        not isinstance(source_chart_cover, MeshingDomainBoundarySource)
        or source_chart_cover.domain.domain_id != domain.domain_id
        or set(source_chart_cover.patches) != set(map(int, np.asarray(charts.patches)))
    ):
        raise ValueError(
            "Actual target trim records change the original retained source-cover authority."
        )
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    if charts.topology_id != mesh.topology_id or charts.domain_id != domain.domain_id:
        raise ValueError(
            "Current trim records require their actual source-bound chart topology."
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
            "Current trim records require one complete retained stratum token per physical vertex."
        )
    declared = np.asarray(charts.cell_global_ids)
    order = np.searchsorted(declared, ids)
    if np.any(order >= declared.size) or not np.array_equal(declared[order], ids):
        raise ValueError("Current trim records omit a scientific physical cell.")
    records = []
    for patch in np.unique(np.asarray(charts.patches)):
        selected = np.flatnonzero(np.asarray(charts.patches)[order] == patch)
        uv_cells = np.asarray(charts.charts)[order[selected]]
        uv, inverse = np.unique(uv_cells.reshape(-1, 2), axis=0, return_inverse=True)
        triangles = inverse.reshape(-1, 3)
        physical_vertices: dict[int, int] = {}
        edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
        from .._meshcore import charge_native_geometry_queries, exact_orient2d

        signs = exact_orient2d(uv_cells[:, 0], uv_cells[:, 1], uv_cells[:, 2])
        if np.any(signs == 0):
            raise ValueError("An actual target trim chart is singular.")
        for row, (triangle, physical) in enumerate(
            zip(triangles, cells[selected], strict=True)
        ):
            if signs[row] < 0:
                triangle = triangle[[0, 2, 1]]
                physical = physical[[0, 2, 1]]
                triangles[row] = triangle
            for vertex, original in zip(triangle, physical, strict=True):
                previous = physical_vertices.get(int(vertex))
                if previous is not None and previous != int(original):
                    raise ValueError(
                        "One trim-chart occurrence merges distinct scientific physical vertices."
                    )
                physical_vertices[int(vertex)] = int(original)
            for first, last in ((0, 1), (1, 2), (2, 0)):
                a, b = int(triangle[first]), int(triangle[last])
                edges.setdefault((min(a, b), max(a, b)), []).append((a, b))
        boundary, provenance = [], []
        for occurrences in edges.values():
            if len(occurrences) == 2:
                continue
            if len(occurrences) != 1:
                raise ValueError("The actual trim chart has a nonmanifold edge.")
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
                    charge_native_geometry_queries(2)
                    source_charts = np.asarray(
                        use.charts(np.asarray((first, last), dtype=np.float64))
                    )
                    if np.array_equal(source_charts, uv[[a, b]]):
                        matches.append((loop_index, use_index, first, last))
            if len(matches) != 1:
                raise ValueError(
                    "A current boundary edge lacks one unambiguous authoritative coedge occurrence."
                )
            boundary.append((a, b))
            provenance.append(matches[0])
        physical = domain.evaluate(np.full(uv.shape[0], int(patch), dtype=np.int32), uv)
        records.append(
            (
                int(patch),
                uv,
                triangles,
                physical,
                np.asarray(boundary, dtype=np.int64).reshape(-1, 2),
                np.asarray(provenance, dtype=np.float64).reshape(-1, 4),
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


class PreparedSurfaceCurveWitness(StrictModule, NonTrainableState):
    """Actual scientific vertex strata, not inferred native feature labels.

    Parameters use the owning coedge's original interval. Every incident
    occurrence is returned separately, including two uses on a periodic seam.
    """

    support: PreparedSurfaceSourceSupport
    source_mesh: CellMesh
    source_association: GeometryAssociation
    topology_id: str = eqx.field(static=True)
    vertex_global_ids: Array
    source_dimensions: Array
    source_indices: Array
    source_parameters: Array
    source_entity_ids: tuple[str, ...] = eqx.field(static=True)
    source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)

    def validate_restored(self) -> None:
        """Authenticate lowered rows against the retained actual source proof."""
        self.support.require_current()
        mesh, association = self.source_mesh, self.source_association
        domain = self.support.domain
        if not isinstance(mesh, CellMesh) or not isinstance(
            association, GeometryAssociation
        ):
            raise TypeError(
                "Curve witness requires its actual original mesh and scientific association."
            )
        if mesh.topology_id != self.topology_id or not np.array_equal(
            np.asarray(mesh.vertex_global_ids), np.asarray(self.vertex_global_ids)
        ):
            raise ValueError(
                "Curve witness changes its original scientific vertex topology."
            )
        if (
            association.association_kind
            is not _surface_source_association_kind(domain.source_kinds)
            or association.source_id != domain.source_id
            or association.source_revision != domain.source_revision
            or not association.complete
            or _target_dimension(mesh, association) != 0
        ):
            raise ValueError(
                "Curve witness changes its original complete source association authority."
            )
        rows = association.target_rows(np.asarray(mesh.vertex_global_ids, dtype=np.int64))
        kinds = np.asarray(association.source_dimensions)[rows]
        indices = np.asarray(
            [
                self.support._domain_row(
                    domain,
                    int(association.source_dimensions[row]),
                    int(association.source_indices[row]),
                    association.source_occurrence_paths[row],
                )
                for row in rows
            ],
            dtype=np.int32,
        )
        entities = tuple(association.source_entity_ids[row] for row in rows)
        occurrences = tuple(association.source_occurrence_paths[row] for row in rows)
        from .._fingerprint import array_tree_fingerprint

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
                "Curve witness changes original scientific array types, shapes or source parameter bits."
            )
        if (
            self.source_entity_ids != entities
            or self.source_occurrence_paths != occurrences
        ):
            raise ValueError(
                "Curve witness alters original source strata, parameters, entities or occurrences."
            )
        if any(
            entity != domain.entity_id(int(kind), int(index))
            or occurrence != domain.source_occurrences[int(kind)][int(index)]
            for entity, occurrence, kind, index in zip(
                entities, occurrences, kinds, indices, strict=True
            )
        ):
            raise ValueError(
                "Curve witness source rows do not name canonical original source entities."
            )

    def require_bound(self, mesh: CellMesh, /) -> None:
        self.validate_restored()
        if mesh.topology_id != self.topology_id or not np.array_equal(
            np.asarray(mesh.vertex_global_ids), np.asarray(self.vertex_global_ids)
        ):
            raise ValueError("Curve witness is stale for the scientific vertex topology.")

    def evaluate(
        self,
        vertex_global_id: int,
        parameter: float,
        /,
    ) -> tuple[np.ndarray, tuple[tuple[int, int, np.ndarray], ...]]:
        """Evaluate a retained curve parameter and all its actual source uses."""
        self.support.require_current()
        rows = np.flatnonzero(np.asarray(self.vertex_global_ids) == vertex_global_id)
        if rows.size != 1:
            raise ValueError("Curve proposal names an absent scientific vertex.")
        row = int(rows[0])
        if int(self.source_dimensions[row]) != 1:
            raise ValueError("Only an authored curve stratum can slide.")
        curve = int(self.source_indices[row])
        return self.evaluate_curve(curve, parameter)

    def evaluate_curve(
        self,
        curve: int,
        parameter: float,
        /,
    ) -> tuple[np.ndarray, tuple[tuple[int, int, np.ndarray], ...]]:
        """Evaluate source-bound born nodes as well as retained source vertices."""
        from .._meshcore import charge_native_geometry_queries

        self.support.require_current()
        if not 0 <= curve < len(self.support.domain.curves):
            raise ValueError("Curve proposal names an absent original source curve.")
        domain = self.support.domain
        owner = curve_owner(domain, curve)
        value = float(parameter)
        if not np.isfinite(value) or not min(owner.first, owner.last) <= value <= max(
            owner.first, owner.last
        ):
            raise ValueError(
                "Curve proposal leaves its authoritative parameter interval."
            )
        fraction = (value - owner.first) / (owner.last - owner.first)
        uses = domain.curve_uses(curve)
        charge_native_geometry_queries(1 + len(uses))
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
            raise ValueError("The original curve owner returned a nonfinite proposal.")
        return point, charts

    def edge_interval(
        self,
        dimensions: np.ndarray,
        indices: np.ndarray,
        parameters: np.ndarray,
        /,
    ) -> tuple[int, float, float]:
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
            raise ValueError(
                "Curve interval requires two complete source-stratum tokens."
            )
        curves = {
            int(index) for kind, index in zip(kinds, rows, strict=True) if kind == 1
        }
        if not curves and np.all(kinds == 0) and rows[0] != rows[1]:
            curves = {
                curve
                for curve, authority in enumerate(self.support.domain.curves)
                if {authority.start, authority.end} == {int(rows[0]), int(rows[1])}
            }
        if len(curves) != 1 or np.any((kinds != 0) & (kinds != 1)):
            raise ValueError("A feature interval lacks one unambiguous authored curve.")
        curve = curves.pop()
        left = _curve_parameter(
            self.support,
            curve,
            int(kinds[0]),
            int(rows[0]),
            values[0],
            float(values[1, 0]) if kinds[1] == 1 else None,
        )
        right = _curve_parameter(
            self.support, curve, int(kinds[1]), int(rows[1]), values[1], left
        )
        owner = curve_owner(self.support.domain, curve)
        if (
            not all(
                np.isfinite(value)
                and min(owner.first, owner.last) <= value <= max(owner.first, owner.last)
                for value in (left, right)
            )
            or left == right
        ):
            raise ValueError(
                "A feature interval is collapsed or leaves its actual curve owner."
            )
        return curve, left, right


def prepare_surface_curve_witness(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    /,
) -> PreparedSurfaceCurveWitness:
    """Lower complete source associations without inventing a curve identity."""
    vertex, _ = source_associations(support, source)
    ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    rows = vertex.target_rows(ids)
    dimensions = np.asarray(vertex.source_dimensions)[rows]
    indices = np.asarray(
        [
            support._domain_row(
                support.domain,
                int(vertex.source_dimensions[row]),
                int(vertex.source_indices[row]),
                vertex.source_occurrence_paths[row],
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
        source_entity_ids=tuple(vertex.source_entity_ids[row] for row in rows),
        source_occurrence_paths=tuple(
            vertex.source_occurrence_paths[row] for row in rows
        ),
    )


def _chart_association_rows(
    mesh: CellMesh,
    witness: SurfaceChartWitness,
    bounds: np.ndarray,
    /,
) -> SurfaceChartAssociationRows:
    from ._topology_edit import key_rows

    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    witness_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    rows = key_rows(witness_ids[:, None], ids[:, None])
    if witness_ids.size != ids.size or np.any(rows < 0):
        raise ValueError("Chart associations require the exact scientific cell witness.")
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
    """Read original authored strata; radial references are never called UV charts."""
    from ..geometry._meshing_domain import PatchPoleUse

    dimension, source_index, path, exact_parameters = atlas.corner_tokens[row][local]
    domain = atlas.domain
    index = PreparedSurfaceSourceSupport._domain_row(
        domain, dimension, source_index, path
    )
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
                "A sphere pole association lacks its actual authored source use."
            )
        uv = np.asarray(poles[0].start, dtype=np.float64)
    elif dimension == 1:
        uses = [use for owner, use in domain.curve_uses(index) if owner == patch]
        if not uses:
            raise ValueError(
                "A sphere seam association lacks its actual authored source curve use."
            )
        uv = uses[0].charts(np.asarray((parameters[0],), dtype=np.float64))[0]
    elif dimension == 2:
        if index != patch:
            raise ValueError(
                "A sphere surface association changes its actual source patch."
            )
        uv = parameters
    else:
        raise ValueError(
            "Sphere corner associations require original source dimensions zero through two."
        )
    return dimension, index, parameters, (Fraction(float(uv[0])), Fraction(float(uv[1])))


def _sphere_chart_association_rows(
    mesh: CellMesh,
    atlas: SphereMaterialCellAtlas,
    bounds: np.ndarray,
    /,
) -> SurfaceChartAssociationRows:
    """Authored parameter representatives for metadata, not a UV coverage theorem."""
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
                    "One physical sphere vertex has inconsistent authored source strata."
                )
            values[int(identifier)] = dimension, index, parameters
    return values


def _sphere_vertex_weights(
    deformation: PreparedSphereChartDeformation,
    maximum_visits: int,
    /,
) -> dict[int, _Weights]:
    """Exact original-reference source support from actual complete native pieces."""
    from ..discretization._coordinate_enclosure import expression_evaluate
    from ..discretization._sphere_chart_deformation import _area

    vertices = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    candidates: dict[int, tuple[int, _Weights]] = {}
    visits = 0
    for piece in deformation.pieces:
        new_row = deformation.target_atlas.cell_row(piece.target_cell_global_id)
        old_row = deformation.source_atlas.cell_row(piece.source_cell_global_id)
        inverse = deformation.target_atlas.projective_reference_map(
            piece.target_cell_global_id,
            deformation.source_atlas,
            piece.source_cell_global_id,
        )
        arguments = inverse.reference_expressions()
        triangle = piece.exact_target_reference_vertices
        sign = _area(triangle)
        for local, point in enumerate(vertices):
            visits += 1
            if visits > maximum_visits:
                raise AssociationPropagationError(
                    "Sphere source support exhausts its original query allowance.",
                    np.asarray((piece.target_cell_global_id,), dtype=np.int64),
                )
            if any(
                _area((triangle[index], triangle[(index + 1) % 3], point)) * sign < 0
                for index in range(3)
            ):
                continue
            u, v = (expression_evaluate(argument, point) for argument in arguments)
            barycentric = (1 - u - v, u, v)
            if any(value < 0 for value in barycentric):
                raise ValueError(
                    "A sphere birth functional leaves its actual original source reference cell."
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
            "Complete sphere pieces do not support all actual target vertex functionals."
        )
    return {identifier: value[1] for identifier, value in candidates.items()}


def _surface_vertex_weights(
    deformation: PreparedSurfaceChartDeformation, /
) -> dict[int, _Weights]:
    """Read vertex functionals from retained exact material overlap pieces."""
    references = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )

    def cells(mesh: CellMesh, /) -> dict[int, tuple[int, int, int]]:
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
                    raise ValueError("Surface material cells must remain triangles.")
                result[int(identifier)] = (
                    int(values[0]),
                    int(values[1]),
                    int(values[2]),
                )
        return result

    source_cells = cells(deformation.source_mesh)
    target_cells = cells(deformation.target_mesh)
    candidates: dict[int, tuple[int, _Weights]] = {}
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
                        "A surface birth functional leaves its original material cell."
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
            "Complete surface pieces do not support every target vertex functional."
        )
    return {identifier: value[1] for identifier, value in candidates.items()}


def source_associations(
    support: PreparedSurfaceSourceSupport, source: CellMeshingResult, /
) -> tuple[GeometryAssociation, tuple[int, ...]]:
    from ._result import CellMeshingResult

    if not isinstance(source, CellMeshingResult):
        raise TypeError(
            "Surface source transfer requires an actual accepted CellMeshingResult."
        )
    if source.coordinate_contract.spatial_id != support.coordinate_contract.spatial_id:
        raise ValueError(
            "Surface source transfer changes the original coordinate/unit contract."
        )
    levels = []
    vertex = None
    expected_kind = _surface_source_association_kind(support.domain.source_kinds)
    for value in source.associations:
        if (
            value.association_kind is not expected_kind
            or value.source_id != support.domain.source_id
            or value.source_revision != support.domain.source_revision
            or not value.complete
        ):
            raise ValueError(
                "Surface source transfer preserves the complete original association namespace."
            )
        dimension = _target_dimension(source.mesh, value)
        for row in range(value.target_global_ids.shape[0]):
            kind = int(value.source_dimensions[row])
            index = support._domain_row(
                support.domain,
                kind,
                int(value.source_indices[row]),
                value.source_occurrence_paths[row],
            )
            if value.source_entity_ids[row] != support.domain.entity_id(kind, index):
                raise ValueError(
                    "A surface association ID does not name its actual original typed source stratum."
                )
        if dimension == 0:
            if vertex is not None:
                raise ValueError(
                    "Surface source transfer requires one vertex association."
                )
            vertex = value
        else:
            levels.append(dimension)
    if vertex is None or len(set(levels)) != len(levels) or 2 not in levels:
        raise ValueError(
            "Surface source transfer requires original vertex and cell associations."
        )
    vertex.target_rows(np.asarray(source.mesh.entity_set(0).entity_ids, dtype=np.int64))
    cells = next(
        value
        for value in source.associations
        if _target_dimension(source.mesh, value) == 2
    )
    cells.target_rows(np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64))
    return vertex, tuple(sorted(levels))


def _codes(
    support: PreparedSurfaceSourceSupport, dimensions: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    offsets = np.cumsum((0, *(len(row) for row in support.domain.source_indices[:-1])))
    if np.any(dimensions < 0) or np.any(indices < 0):
        raise ValueError(
            "A current native surface entity has no unambiguous original source stratum."
        )
    return offsets[dimensions] + indices


def classes(
    support: PreparedSurfaceSourceSupport, source: CellMeshingResult, /
) -> tuple[SurfaceEntityClasses, ...]:
    source_associations(support, source)
    mesh = source.mesh
    result = []
    associations = {
        _target_dimension(mesh, value): value for value in source.associations
    }
    for dimension in range(3):
        identifiers = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        dims = np.full(identifiers.shape, -1, dtype=np.int64)
        indices = np.full(identifiers.shape, -1, dtype=np.int64)
        if dimension == 1:
            links = _incidence_pairs(mesh, 1, 2)
            cell = associations[2]
            cell_rows = cell.target_rows(
                np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
            )
            patch_ids = np.asarray(
                [
                    support._domain_row(
                        support.domain,
                        int(cell.source_dimensions[row]),
                        int(cell.source_indices[row]),
                        cell.source_occurrence_paths[row],
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
            target_rows = _entity_rows(
                mesh, dimension, np.asarray(value.target_global_ids, dtype=np.int64)
            )
            if np.any(target_rows < 0):
                raise ValueError(
                    "Source surface association names unavailable current mesh entities."
                )
            for source_row, row in enumerate(target_rows.tolist()):
                kind = int(value.source_dimensions[source_row])
                dims[row] = kind
                indices[row] = support._domain_row(
                    support.domain,
                    kind,
                    int(value.source_indices[source_row]),
                    value.source_occurrence_paths[source_row],
                )
        if dimension == 2 and np.any(dims != 2):
            raise ValueError(
                "Native physical surface cells must retain their original patch stratum."
            )
        if dimension == 1 and np.any(dims < 1):
            raise ValueError(
                "Surface interface edges require their actual original coedge authority."
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


def protected_edges(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    /,
    *,
    midpoint_required: bool = True,
) -> np.ndarray:
    source_associations(support, source)
    links = _incidence_pairs(source.mesh, 0, 1)
    identifiers = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
    count = source.mesh.entity_set(1).count
    protected = np.zeros(count, dtype=np.bool_)
    for edge in range(count):
        vertices = links[links[:, 1] == edge, 0]
        if vertices.size != 2 or identifiers[vertices[0]] == identifiers[vertices[1]]:
            raise ValueError(
                "A native physical surface edge must have two distinct current identities."
            )
        if midpoint_required:
            point = np.sum(coordinates[vertices] * np.float64(0.5), axis=0)
            protected[edge] = np.array_equal(
                point, coordinates[vertices[0]]
            ) or np.array_equal(point, coordinates[vertices[1]])
    # Curved feature chords are not falsely treated as exact analytic curves:
    # their native subdivisions inherit the original source homotopy and retain
    # a freshly evaluated, generally nonzero source residual.
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
            "Current vertex lacks an actual original-chart restriction witness.",
            np.asarray((vertex_id,), dtype=np.int64),
        )
    # Actual source seam/pole copies are chart gauges, not coordinate welding.
    # The selected scientific cell ID defines the numeric representative.
    return np.asarray(
        tuple(float(value) for value in min(candidates, key=lambda value: value[0])[1]),
        dtype=np.float64,
    )


def _curve_parameter(
    support: PreparedSurfaceSourceSupport,
    curve: int,
    kind: int,
    index: int,
    parameters: np.ndarray,
    other: float | None,
) -> float:
    if kind == 1:
        if index != curve:
            raise ValueError("A feature midpoint mixes different original source curves.")
        return float(parameters[0])
    if kind != 0:
        raise ValueError(
            "A feature midpoint endpoint is not in its original curve closure."
        )
    authority = support.domain.curves[curve]
    owner = curve_owner(support.domain, curve)
    first, last = owner.first, owner.last
    if index == authority.start and index == authority.end:
        if other is None or (other - first) / (last - first) == 0.5:
            raise ValueError(
                "A closed feature endpoint requires its actual unambiguous source interval side."
            )
        return first if (other - first) / (last - first) < 0.5 else last
    if index == authority.start:
        return first
    if index == authority.end:
        return last
    raise ValueError("A feature midpoint names an unrelated original source corner.")


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


type _Weights = tuple[tuple[int, Fraction], ...]


def _curve_parameter_from_functional(
    support: PreparedSurfaceSourceSupport,
    source: CellMesh,
    vertex: GeometryAssociation,
    classes: SurfaceEntityClasses,
    vertex_rows: np.ndarray,
    curve: int,
    key: _Weights,
    /,
) -> float:
    """Retain an original chord's parameter functional and its nonzero residual."""
    if len(key) not in (1, 2):
        raise ValueError(
            "A source-bound feature must retain its original edge functional."
        )
    identifiers = np.asarray([identifier for identifier, _ in key], dtype=np.int64)
    rows = _entity_rows(source, 0, identifiers)
    if np.any(rows < 0):
        raise ValueError(
            "A feature functional references an absent scientific source vertex."
        )
    values = np.asarray(vertex.parameters)[vertex_rows[rows]]
    if len(key) == 1:
        row = int(rows[0])
        parameter = _curve_parameter(
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
        first = _curve_parameter(
            support,
            curve,
            int(classes.dimensions[a]),
            int(classes.indices[a]),
            values[0],
            first_hint,
        )
        last = _curve_parameter(
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
            "A feature functional leaves its original authored curve interval."
        )
    return parameter


def _dot(first: tuple[Fraction, ...], second: tuple[Fraction, ...], /) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _support_weights(
    points: np.ndarray, point: np.ndarray, /
) -> tuple[Fraction, ...] | None:
    """Exact least-squares barycentric weights of ``point`` on a segment or triangle support."""
    origin = tuple(Fraction(float(value)) for value in points[0])
    edges = tuple(
        tuple(
            Fraction(float(value)) - base for value, base in zip(row, origin, strict=True)
        )
        for row in points[1:]
    )
    offset = tuple(
        Fraction(float(value)) - base for value, base in zip(point, origin, strict=True)
    )
    gram = tuple(tuple(_dot(a, b) for b in edges) for a in edges)
    rhs = tuple(_dot(edge, offset) for edge in edges)
    if len(edges) == 1:
        if gram[0][0] == 0:
            return None
        local: tuple[Fraction, ...] = (rhs[0] / gram[0][0],)
    else:
        determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
        if determinant == 0:
            return None
        local = (
            (rhs[0] * gram[1][1] - gram[0][1] * rhs[1]) / determinant,
            (gram[0][0] * rhs[1] - gram[1][0] * rhs[0]) / determinant,
        )
    return (1 - sum(local, Fraction(0)), *local)


def _created_weights(
    lineage: MeshLineage, source_mesh: CellMesh, target: CellMesh, /
) -> dict[int, _Weights]:
    from ._lineage import EntityLineageKind

    """Position of every created vertex in its actual split-lineage support simplex.

    Lineage alone names the support (a source edge or cell); the exact weights
    locate the actual target carrier point in it. Any off-support carrier
    deviation is measured afresh by the restriction proof.
    """
    record = lineage.entity_lineage(0)
    split = np.asarray(record.relation_kinds, dtype=np.int32) == int(
        EntityLineageKind.SPLIT_FROM
    )
    supports: dict[int, set[int]] = {}
    for parent, child in zip(
        np.asarray(record.source_global_ids, dtype=np.int64)[split].tolist(),
        np.asarray(record.target_global_ids, dtype=np.int64)[split].tolist(),
        strict=True,
    ):
        supports.setdefault(child, set()).add(parent)
    children = np.asarray(sorted(supports), dtype=np.int64)
    child_rows = _entity_rows(target, 0, children)
    if np.any(child_rows < 0):
        raise AssociationPropagationError(
            "Split lineage names vertices absent from the current target.",
            children[child_rows < 0],
        )
    source_coordinates = np.asarray(source_mesh.coordinates, dtype=np.float64)
    target_coordinates = np.asarray(target.coordinates, dtype=np.float64)
    weights: dict[int, _Weights] = {}
    for child, child_row in zip(children.tolist(), child_rows.tolist(), strict=True):
        witness = np.asarray((child,), dtype=np.int64)
        ordered = np.asarray(sorted(supports[child]), dtype=np.int64)
        rows = _entity_rows(source_mesh, 0, ordered)
        if ordered.size not in (2, 3) or np.any(rows < 0):
            raise AssociationPropagationError(
                "A created surface vertex lacks a source segment or triangle lineage support.",
                witness,
            )
        local = _support_weights(source_coordinates[rows], target_coordinates[child_row])
        if local is None or any(value < 0 for value in local):
            raise AssociationPropagationError(
                "A created surface vertex leaves its split-lineage support simplex.",
                witness,
            )
        weights[child] = tuple(zip(ordered.tolist(), local, strict=True))
    return weights


def _lineage_ancestry(
    source_proof: SurfaceRestrictionRows,
    lineage: MeshLineage,
    target: CellMesh,
    weights: Mapping[int, _Weights],
    /,
) -> dict[int, SurfaceCellAncestry]:
    """Serial target roots and root-reference corners from actual topology lineage.

    Preserved vertices keep their proved source-cell reference corner; a created
    vertex is the weighted reference point of its split-lineage support in the
    same scientific root cell. Coordinates never select a root.
    """
    known = {
        cell: (root, dict(zip(vertices, corners, strict=True)))
        for cell, root, vertices, corners in zip(
            source_proof.cell_ids.tolist(),
            source_proof.root_cell_ids.tolist(),
            source_proof.vertex_ids,
            source_proof.reference_corners,
            strict=True,
        )
    }
    cell_record = lineage.entity_lineage(2)
    cell_parents: dict[int, set[int]] = {}
    for parent, child in zip(
        np.asarray(cell_record.source_global_ids, dtype=np.int64).tolist(),
        np.asarray(cell_record.target_global_ids, dtype=np.int64).tolist(),
        strict=True,
    ):
        cell_parents.setdefault(child, set()).add(parent)
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    ancestry: dict[int, SurfaceCellAncestry] = {}
    for block in target.blocks:
        for identifier, vertices in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(),
            np.asarray(block.vertices),
            strict=True,
        ):
            witness = np.asarray((identifier,), dtype=np.int64)
            parents = sorted(cell_parents.get(identifier, (identifier,)))
            if any(parent not in known for parent in parents):
                raise AssociationPropagationError(
                    "A serial surface cell lacks proved source-cell lineage.", witness
                )
            roots = {known[parent][0] for parent in parents}
            if len(roots) != 1:
                raise AssociationPropagationError(
                    "A serial surface cell merges different scientific source roots.",
                    witness,
                )
            positions: dict[int, tuple[Fraction, Fraction]] = {}
            for parent in parents:
                for vertex, corner in known[parent][1].items():
                    if positions.setdefault(vertex, corner) != corner:
                        raise AssociationPropagationError(
                            "Sibling lineage disagrees on a root-reference vertex position.",
                            witness,
                        )
            corners: list[tuple[Fraction, Fraction]] = []
            for vertex in target_vertex_ids[vertices].tolist():
                if vertex in positions:
                    corners.append(positions[vertex])
                    continue
                support = weights.get(vertex, ())
                if not support or any(end not in positions for end, _ in support):
                    raise AssociationPropagationError(
                        "A created serial vertex has no split-lineage support in its root cell.",
                        witness,
                    )
                corners.append(
                    (
                        sum(
                            (weight * positions[end][0] for end, weight in support),
                            Fraction(0),
                        ),
                        sum(
                            (weight * positions[end][1] for end, weight in support),
                            Fraction(0),
                        ),
                    )
                )
            ancestry[identifier] = SurfaceCellAncestry(roots.pop(), tuple(corners))
    return ancestry


def _source_proof(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    receipts: SurfaceSourceReceipts | None,
    /,
) -> SurfaceRestrictionRows:
    """Root-identity proof of the transfer source, through its certified serial chain if any."""
    mesh = source.mesh
    if (
        mesh.topology_id == support.root_topology_id
        or mesh.storage is not None
        or source.geometry.restriction_source is not None
    ):
        return support.prove_native_restrictions(mesh, source.geometry, receipts=receipts)
    chain = None if source.certification is None else source.certification.request.source
    if (
        not isinstance(chain, SurfaceSubdivisionBoundarySource)
        or chain.support.support_id != support.support_id
        or chain.targets[-1].topology_id != mesh.topology_id
    ):
        raise ValueError(
            "A serial surface descendant lacks its certified root subdivision lineage chain."
        )
    return _subdivision_proof(support, chain.lineages, chain.targets, source.geometry)[0]


def propagate(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    lineage: MeshLineage,
    target: CellMesh,
    /,
    *,
    geometry: CellGeometrySpec,
    receipts: SurfaceSourceReceipts | None = None,
    embedding: GlobalEmbeddingCertificate | None = None,
    coverage: DomainCoverageCertificate | None = None,
    deformation: PreparedSurfaceChartDeformation
    | PreparedSphereChartDeformation
    | None = None,
) -> tuple[GeometryAssociation, ...]:
    from ._lineage import EntityLineageKind, MeshLineage

    vertex, dimensions = source_associations(support, source)
    from ..discretization._sphere_chart_deformation import PreparedSphereChartDeformation

    sphere_bindings: dict[int, tuple[int, int, np.ndarray]] | None = None
    if (
        not isinstance(lineage, MeshLineage)
        or lineage.source_topology_id != source.mesh.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(
            "Surface source transfer requires actual source-to-target native topology lineage."
        )
    if target.storage is not None:
        if embedding is not None:
            embedding.binding.require(target, geometry)
            if embedding.status != "certified":
                raise ValueError(
                    "Current collective surface embedding remains unresolved."
                )
        if coverage is not None:
            coverage.binding.require(target, geometry)
            if coverage.status != "certified":
                raise ValueError(
                    "Current collective native source subdivision coverage remains unresolved."
                )
    old = classes(support, source)
    if deformation is None:
        source_proof = _source_proof(support, source, receipts)
        weights = _created_weights(lineage, source.mesh, target)
        serial = target.storage is None and geometry.restriction_source is None
        ancestry = (
            _lineage_ancestry(source_proof, lineage, target, weights) if serial else None
        )
        target_proof = support.prove_native_restrictions(
            target, geometry, receipts=receipts, ancestry=ancestry
        )
        source_roots = dict(
            zip(
                source_proof.cell_ids.tolist(),
                source_proof.root_cell_ids.tolist(),
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
                "Unknown/remeshed surface lineage cannot preserve source support.",
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
                    "Native surface lineage changes scientific source root identity.",
                    np.asarray((child,), dtype=np.int64),
                )
    else:
        deformation.require_bound(source.mesh, source.geometry, target, geometry)
        if deformation.domain_id != support.domain.domain_id:
            raise ValueError("Chart transport changes original source authority.")
        if isinstance(deformation, PreparedSphereChartDeformation):
            source_proof = _sphere_chart_association_rows(
                source.mesh,
                deformation.source_atlas,
                np.asarray(deformation.source_fidelity_bounds),
            )
            target_proof = _sphere_chart_association_rows(
                target,
                deformation.target_atlas,
                np.asarray(deformation.target_fidelity_bounds),
            )
            weights = _sphere_vertex_weights(deformation, support.maximum_support_queries)
            sphere_bindings = _sphere_vertex_bindings(deformation.target_atlas)
        else:
            source_proof = _chart_association_rows(
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
        cell_record = lineage.entity_lineage(2)
    patch_by_cell = dict(
        zip(target_proof.cell_ids.tolist(), target_proof.patches.tolist(), strict=True)
    )
    bound_by_cell = dict(
        zip(
            target_proof.cell_ids.tolist(),
            target_proof.source_bounds.tolist(),
            strict=True,
        )
    )
    target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    vertex_rows = vertex.target_rows(
        np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    )
    previous_rows = _entity_rows(source.mesh, 0, target_ids)
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
                    "Preserved native surface vertex changes its actual coordinate representative.",
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
                    "A preserved sphere vertex changes its original source stratum.",
                    target_ids[row : row + 1],
                )
            parameters[row] = value
    created_rows = np.flatnonzero(previous_rows < 0)
    if created_rows.size:
        ordering = np.argsort(target_ids[created_rows], kind="stable")
        created_rows = created_rows[ordering]
        source_dims, source_rows = _child_sources(
            lineage, source.mesh, target, target_ids[created_rows]
        )
        for row, source_dimension, source_row in zip(
            created_rows.tolist(), source_dims.tolist(), source_rows.tolist(), strict=True
        ):
            if deformation is not None and (source_dimension < 1 or source_row < 0):
                parents = {parent for parent, _ in weights[int(target_ids[row])]}
                candidates = [
                    cell
                    for cell, vertices in enumerate(source_proof.vertex_ids)
                    if parents <= set(vertices)
                ]
                if candidates:
                    source_dimension, source_row = 2, candidates[0]
            identifier = int(target_ids[row])
            if (
                source_dimension not in (1, 2)
                or source_row < 0
                or (
                    deformation is None
                    and len(weights.get(identifier, ())) != source_dimension + 1
                )
            ):
                raise AssociationPropagationError(
                    "A created native surface vertex lacks its source edge or cell lineage support.",
                    target_ids[row : row + 1],
                )
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
                    "Native midpoint cannot be a new original source corner.",
                    target_ids[row : row + 1],
                )
    if np.any(dims < 0):
        raise AssociationPropagationError(
            "Current vertices have unresolved original source strata.",
            target_ids[dims < 0],
        )
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
                    "A transported feature lacks its original chart use.",
                    target_ids[row : row + 1],
                )
            # A native boundary chord is not the analytic pcurve. Its original
            # source functional is retained and the genuine curve discrepancy
            # is exposed below; COMPLETE chart coverage retains the trim ribbon.
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
                    "A feature leaves its actual original curve use.",
                    target_ids[row : row + 1],
                )
            parameters[row, 0] = parameter
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
    output = [
        GeometryAssociation(
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
    ]
    for dimension in dimensions:
        previous = next(
            value
            for value in source.associations
            if _target_dimension(source.mesh, value) == dimension
        )
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
        if dimension == 2:
            for row, identifier in enumerate(ids.tolist()):
                if identifier not in patch_by_cell:
                    raise AssociationPropagationError(
                        "Current physical cell has no actual original source restriction.",
                        ids[row : row + 1],
                    )
                source_indices[row] = patch_by_cell[identifier]
                proof_row = int(np.flatnonzero(target_proof.cell_ids == identifier)[0])
                params[row] = np.asarray(
                    tuple(
                        float(
                            sum(
                                (
                                    point[axis]
                                    for point in target_proof.source_corners[proof_row]
                                ),
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
                        "Current cell has no declared native source-cell lineage.",
                        ids[row : row + 1],
                    )
                parent_identifiers[row] = np.min(ancestors)
        elif dimension == 1:
            links = _incidence_pairs(target, 0, 1)
            cell_links = _incidence_pairs(target, 1, 2)
            current_cells = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)
            proof_rows = {
                cell: proof_row
                for proof_row, cell in enumerate(target_proof.cell_ids.tolist())
            }
            for row, identifier in enumerate(ids.tolist()):
                ancestors = old_ids[new_ids == identifier]
                ancestor_rows = (
                    _entity_rows(source.mesh, 1, ancestors)
                    if ancestors.size
                    else np.empty((0,), dtype=np.int64)
                )
                if np.any(ancestor_rows < 0):
                    raise AssociationPropagationError(
                        "Current edge lineage names undeclared source edges.",
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
                            "New surface interface edge lacks actual source-curve lineage.",
                            ids[row : row + 1],
                        )
                    source_indices[row] = patches[0]
                    parent_dimensions[row] = 2
                    source_cells = np.asarray(
                        cell_record.source_global_ids, dtype=np.int64
                    )
                    target_cells = np.asarray(
                        cell_record.target_global_ids, dtype=np.int64
                    )
                    parents = source_cells[np.isin(target_cells, current_cells[adjacent])]
                    if not parents.size:
                        raise AssociationPropagationError(
                            "A new interior edge lacks its actual source-cell lineage.",
                            ids[row : row + 1],
                        )
                    parent_identifiers[row] = np.min(parents)
                vertices = links[links[:, 1] == row, 0]
                if vertices.size != 2:
                    raise ValueError(
                        "A current physical surface edge must retain two vertex identities."
                    )
                midpoint = np.sum(np.asarray(target.coordinates)[vertices] * 0.5, axis=0)
                if kinds[row] == 1:
                    curve = int(source_indices[row])
                    a, b = vertices.tolist()
                    av = float(parameters[a, 0]) if dims[a] == 1 else None
                    bv = float(parameters[b, 0]) if dims[b] == 1 else None
                    first = _curve_parameter(
                        support, curve, int(dims[a]), int(indices[a]), parameters[a], bv
                    )
                    last = _curve_parameter(
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
                            "A surface edge has no adjacent cell chart on its source patch.",
                            ids[row : row + 1],
                        )
                    chart_row = proof_rows[chart_cells[0]]
                    corners = target_proof.vertex_ids[chart_row]
                    charts = target_proof.source_corners[chart_row]
                    ends = tuple(
                        charts[corners.index(int(target_ids[vertex]))]
                        for vertex in vertices.tolist()
                    )
                    params[row] = np.asarray(
                        tuple(
                            float((ends[0][axis] + ends[1][axis]) / 2)
                            for axis in range(2)
                        ),
                        dtype=np.float64,
                    )
                    errors[row] = _residual(support, 2, patch, params[row], midpoint)
        else:
            raise ValueError(
                "Native surface source transfer has no higher-dimensional physical cells."
            )
        if np.any(source_indices < 0):
            raise AssociationPropagationError(
                "Current surface entities lack original source authority.",
                ids[source_indices < 0],
            )
        output.append(
            GeometryAssociation(
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
        )
    return tuple(output)


_REFERENCE_EDGES = (
    ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))),
    ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
    ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(0))),
)


def _edge_parameter(
    edge: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    point: tuple[Fraction, Fraction],
    /,
) -> Fraction | None:
    """Exact parameter of ``point`` on one oriented unit-triangle edge, or None off it."""
    (x0, y0), (x1, y1) = edge
    dx, dy, px, py = x1 - x0, y1 - y0, point[0] - x0, point[1] - y0
    if px * dy - py * dx != 0:
        return None
    parameter = (px * dx + py * dy) / (dx * dx + dy * dy)
    return parameter if 0 <= parameter <= 1 else None


def _tiles_reference_triangle(
    children: list[tuple[tuple[Fraction, Fraction], ...]], /
) -> bool:
    """Exact oriented-chain proof that positively oriented children tile the unit triangle.

    Interior edges must cancel and the remaining boundary chain must be exactly
    the oriented unit-triangle boundary. A planar 2-chain with zero boundary
    vanishes, so the positively oriented children cover the root exactly once.
    """
    net: dict[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], int] = {}
    for corners in children:
        for k in range(3):
            first, second = corners[k], corners[(k + 1) % 3]
            net[(first, second)] = net.get((first, second), 0) + 1
            net[(second, first)] = net.get((second, first), 0) - 1
    if any(count > 1 for count in net.values()):
        return False
    spans: tuple[list[tuple[Fraction, Fraction]], ...] = ([], [], [])
    for (first, second), count in net.items():
        if count != 1:
            continue
        for edge, edge_spans in zip(_REFERENCE_EDGES, spans, strict=True):
            start, end = _edge_parameter(edge, first), _edge_parameter(edge, second)
            if start is not None and end is not None and start < end:
                edge_spans.append((start, end))
                break
        else:
            return False
    for edge_spans in spans:
        position = Fraction(0)
        for start, end in sorted(edge_spans):
            if start != position:
                return False
            position = end
        if position != 1:
            return False
    return True


def _l1_upper(first: np.ndarray, second: tuple[Fraction, ...], /) -> Fraction:
    return sum(
        (abs(Fraction(float(a)) - b) for a, b in zip(first, second, strict=True)),
        Fraction(0),
    )


def _subdivision_proof(
    support: PreparedSurfaceSourceSupport,
    lineages: tuple[MeshLineage, ...],
    targets: tuple[CellMesh, ...],
    geometry: CellGeometrySpec,
    /,
) -> tuple[SurfaceRestrictionRows, tuple[dict[int, _Weights], ...]]:
    """Compose serial lineage epochs from the accepted root to the final target.

    Each epoch's ancestry is taken from the previous epoch's proved root
    references, so every proof stays in exact root-reference coordinates.
    """
    from ._result import CellMeshingResult

    root = support.original
    if not isinstance(root, CellMeshingResult):
        raise ValueError(
            "Serial subdivision proofs require the accepted serial surface root."
        )
    proof = support.prove_native_restrictions(root.mesh, root.geometry)
    previous = root.mesh
    epochs: list[dict[int, _Weights]] = []
    for index, (lineage, target) in enumerate(zip(lineages, targets, strict=True)):
        weights = _created_weights(lineage, previous, target)
        ancestry = _lineage_ancestry(proof, lineage, target, weights)
        epoch_geometry = (
            geometry if index == len(targets) - 1 else CellGeometrySpec.affine(target)
        )
        if index == len(targets) - 1 and epoch_geometry.restriction_source is not None:
            proof = support.prove_native_restrictions(target, epoch_geometry)
            restricted_roots = dict(
                zip(proof.cell_ids.tolist(), proof.root_cell_ids.tolist(), strict=True)
            )
            if set(restricted_roots) != set(ancestry) or any(
                restricted_roots[identifier] != witness.root_cell_id
                for identifier, witness in ancestry.items()
            ):
                raise ValueError(
                    "Restricted surface roots disagree with the serial lineage ancestry."
                )
        else:
            proof = support.prove_native_restrictions(
                target, epoch_geometry, ancestry=ancestry
            )
        epochs.append(weights)
        previous = target
    return proof, tuple(epochs)


@final
class SurfaceSubdivisionBoundarySource(StrictModule, NonTrainableState):
    """Certified chart cover of serial native subdivisions of an accepted surface root.

    The root chart chain is re-verified by its owning source. The retained
    lineage chain proves each target cell's scientific root and exact
    root-reference corners; the cells of every root must tile its reference
    triangle exactly. A cell chord inherits its root's two-sided chart bound
    plus its exactly measured deviation from the restricted root chord.
    Collapsed source wedges follow the split lineage of their root edge. Point
    queries remain the root source's.
    """

    root: MeshingDomainBoundarySource
    support: PreparedSurfaceSourceSupport
    lineages: tuple[MeshLineage, ...]
    targets: tuple[CellMesh, ...]
    geometry: CellGeometrySpec
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    def __init__(
        self,
        root: MeshingDomainBoundarySource,
        support: PreparedSurfaceSourceSupport,
        lineages: tuple[MeshLineage, ...],
        targets: tuple[CellMesh, ...],
        geometry: CellGeometrySpec,
        /,
    ) -> None:
        from ._lineage import MeshLineage
        from ._result import CellMeshingResult

        if (
            not isinstance(root, MeshingDomainBoundarySource)
            or not isinstance(support, PreparedSurfaceSourceSupport)
            or not isinstance(geometry, CellGeometrySpec)
            or any(not isinstance(value, MeshLineage) for value in lineages)
            or any(not isinstance(value, CellMesh) for value in targets)
        ):
            raise TypeError(
                "Subdivision fidelity requires the root chart source, original support, lineages, target meshes and map."
            )
        original = support.original
        if (
            not isinstance(original, CellMeshingResult)
            or original.mesh.storage is not None
        ):
            raise ValueError(
                "Subdivision fidelity requires a serial accepted surface root."
            )
        if (root.source_id, root.source_revision) != (
            support.domain.source_id,
            support.domain.source_revision,
        ) or root.domain.domain_id != support.domain.domain_id:
            raise ValueError(
                "The root chart source does not bind the original surface support."
            )
        if not lineages or len(lineages) != len(targets):
            raise ValueError(
                "Subdivision fidelity requires one target mesh per serial lineage epoch."
            )
        topology = support.root_topology_id
        for lineage, target in zip(lineages, targets, strict=True):
            if (
                lineage.source_topology_id != topology
                or lineage.target_topology_id != target.topology_id
                or target.storage is not None
            ):
                raise ValueError(
                    "Subdivision fidelity lineages must chain serially from the accepted root topology."
                )
            topology = target.topology_id
        restriction = geometry.restriction_source
        if restriction is not None:
            from ..discretization._cell_geometry_validity import cell_geometry_id

            if (
                restriction.source_topology_id != original.mesh.topology_id
                or restriction.source_geometry_id != cell_geometry_id(original.geometry)
            ):
                raise ValueError(
                    "Subdivision fidelity restriction must retain the accepted "
                    "serial root coordinate authority."
                )
        self.root, self.support, self.lineages, self.targets, self.geometry = (
            root,
            support,
            tuple(lineages),
            tuple(targets),
            geometry,
        )
        self.source_id, self.source_revision = root.source_id, root.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.root.ambient_dimension

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return self.root.boundary_distance(points)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        return self.root.boundary_samples(maximum_samples)

    def require_current(self) -> None:
        """Re-establish original support identity and every retained binding after restoration."""
        self.support.require_current()
        rebuilt = SurfaceSubdivisionBoundarySource(
            self.root, self.support, self.lineages, self.targets, self.geometry
        )
        if (rebuilt.source_id, rebuilt.source_revision) != (
            self.source_id,
            self.source_revision,
        ):
            raise ValueError(
                "Subdivision fidelity source identity differs from its retained root chart source."
            )

    def boundary_chart_cover(self, maximum_patches: int, /) -> SourceBoundaryChartCover:
        cover = self.root.boundary_chart_cover(maximum_patches)
        if cover.semantics != "certified" or not cover.complete:
            return cover
        from ._result import CellMeshingResult

        root = self.support.original
        if not isinstance(root, CellMeshingResult):
            raise ValueError(
                "Subdivision fidelity lost its serial accepted surface root."
            )
        proof, epochs = _subdivision_proof(
            self.support, self.lineages, self.targets, self.geometry
        )
        findings = list(cover.findings)
        if proof.cell_ids.size + cover.simplices.shape[0] > maximum_patches:
            findings.append(
                MeshCertificateFinding(
                    "source_chart_cover_capacity", "unresolved", "mesh"
                )
            )
            return SourceBoundaryChartCover(
                cover.simplices,
                cover.deviation_bounds,
                "sampled",
                False,
                self.source_id,
                self.source_revision,
                findings=tuple(findings),
                resource_counts=cover.resource_counts,
            )
        root_coordinates = np.asarray(root.mesh.coordinates, dtype=np.float64)
        root_vertex_ids = np.asarray(root.mesh.vertex_global_ids, dtype=np.int64).tolist()
        # Chart-cover pieces carry only physical corners; they bind root mesh
        # entities exactly as the fidelity certifier binds them to mesh maps.
        vertex_keys: dict[tuple[float, ...], int | None] = {}
        for identifier, point in zip(
            root_vertex_ids, root_coordinates.tolist(), strict=True
        ):
            key = tuple(point)
            vertex_keys[key] = identifier if key not in vertex_keys else None
        cell_keys: dict[tuple[tuple[float, ...], ...], int | None] = {}
        for block in root.mesh.blocks:
            for identifier, vertices in zip(
                np.asarray(block.global_ids, dtype=np.int64).tolist(),
                np.asarray(block.vertices),
                strict=True,
            ):
                key = tuple(
                    sorted(tuple(row) for row in root_coordinates[vertices].tolist())
                )
                cell_keys[key] = identifier if key not in cell_keys else None
        cell_bounds: dict[int, float] = {}
        collapsed: list[tuple[tuple[int, ...], float]] = []
        complete = True
        for simplex, bound in zip(
            np.asarray(cover.simplices).tolist(),
            np.asarray(cover.deviation_bounds).tolist(),
            strict=True,
        ):
            corners = sorted({tuple(row) for row in simplex})
            if len(corners) == 3:
                cell = cell_keys.get(tuple(corners))
                if cell is None:
                    complete = False
                    continue
                cell_bounds[cell] = max(bound, cell_bounds.get(cell, 0.0))
            else:
                ends = tuple(vertex_keys.get(corner) for corner in corners)
                if any(end is None for end in ends):
                    complete = False
                    continue
                collapsed.append((tuple(end for end in ends if end is not None), bound))
        children: dict[int, list[tuple[tuple[Fraction, Fraction], ...]]] = {}
        for root_id, corners in zip(
            proof.root_cell_ids.tolist(), proof.reference_corners, strict=True
        ):
            children.setdefault(root_id, []).append(corners)
        complete &= set(children) == set(cell_bounds) and all(
            _tiles_reference_triangle(value) for value in children.values()
        )
        target = self.targets[-1]
        target_coordinates = np.asarray(target.coordinates, dtype=np.float64)
        target_rows = {
            identifier: row
            for row, identifier in enumerate(
                np.asarray(target.vertex_global_ids, dtype=np.int64).tolist()
            )
        }
        images: list[np.ndarray] = []
        bounds: list[float] = []
        for vertices, root_id, deviation in zip(
            proof.vertex_ids,
            proof.root_cell_ids.tolist(),
            proof.restriction_deviations.tolist(),
            strict=True,
        ):
            if root_id not in cell_bounds:
                continue
            images.append(
                target_coordinates[[target_rows[vertex] for vertex in vertices]]
            )
            bounds.append(_upper(Fraction(cell_bounds[root_id]) + Fraction(deviation)))
        for ends, bound in collapsed:
            chain = self._collapsed_chain(
                ends,
                epochs,
                root_coordinates,
                root_vertex_ids,
                target_coordinates,
                target_rows,
            )
            if chain is None:
                complete = False
                continue
            for first, second, deviation in chain:
                images.append(
                    np.stack(
                        (
                            target_coordinates[first],
                            target_coordinates[first],
                            target_coordinates[second],
                        )
                    )
                )
                bounds.append(_upper(Fraction(bound) + deviation))
        if not complete:
            findings.append(
                MeshCertificateFinding(
                    "source_trim_coverage_premise", "unresolved", "mesh"
                )
            )
        return SourceBoundaryChartCover(
            np.stack(images) if images else np.empty((0, 3, 3)),
            np.asarray(bounds, dtype=np.float64),
            "certified" if complete else "sampled",
            complete,
            self.source_id,
            self.source_revision,
            findings=tuple(findings),
            resource_counts=cover.resource_counts,
        )

    @staticmethod
    def _collapsed_chain(
        ends: tuple[int, ...],
        epochs: tuple[Mapping[int, _Weights], ...],
        root_coordinates: np.ndarray,
        root_vertex_ids: list[int],
        target_coordinates: np.ndarray,
        target_rows: Mapping[int, int],
        /,
    ) -> list[tuple[int, int, Fraction]] | None:
        """Target rows and endpoint deviations of a collapsed root wedge's split segment."""
        if any(end not in target_rows for end in ends):
            return None
        if len(ends) == 1:
            row = target_rows[ends[0]]
            return [(row, row, Fraction(0))]
        chain: list[tuple[Fraction, int]] = [
            (Fraction(0), ends[0]),
            (Fraction(1), ends[1]),
        ]
        for weights in epochs:
            splits: dict[frozenset[int], list[tuple[int, dict[int, Fraction]]]] = {}
            for vertex, support in weights.items():
                shares = dict(support)
                splits.setdefault(frozenset(shares), []).append((vertex, shares))
            inserted = list(chain)
            for (first_t, first), (second_t, second) in zip(
                chain[:-1], chain[1:], strict=True
            ):
                for vertex, shares in splits.get(frozenset((first, second)), ()):
                    inserted.append(
                        (first_t + shares[second] * (second_t - first_t), vertex)
                    )
            chain = sorted(inserted)
        if any(vertex not in target_rows for _, vertex in chain):
            return None
        start, end = (root_coordinates[root_vertex_ids.index(value)] for value in ends)
        points: list[tuple[int, Fraction]] = []
        for parameter, vertex in chain:
            exact = tuple(
                (1 - parameter) * Fraction(float(a)) + parameter * Fraction(float(b))
                for a, b in zip(start, end, strict=True)
            )
            row = target_rows[vertex]
            points.append((row, _l1_upper(target_coordinates[row], exact)))
        return [
            (first[0], second[0], max(first[1], second[1]))
            for first, second in zip(points[:-1], points[1:], strict=True)
        ]


@final
class SurfaceChartBoundarySource(StrictModule, NonTrainableState):
    """Original continuous source theorem plus a bound accepted chart epoch.

    The old/new physical maps remain distinct. This retains the independently
    verified original source cover, not a chord cover declared equal to either
    coordinate map, and the actual chart/measure correspondence for later use.
    """

    root: MeshingDomainBoundarySource | SurfaceSubdivisionBoundarySource
    support: PreparedSurfaceSourceSupport
    deformation: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation
    predecessor_fidelity: SourceFidelityCertificate

    def __init__(
        self,
        root: MeshingDomainBoundarySource | SurfaceSubdivisionBoundarySource,
        support: PreparedSurfaceSourceSupport,
        deformation: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation,
        predecessor_fidelity: SourceFidelityCertificate,
        /,
    ) -> None:
        from ..discretization._sphere_chart_deformation import (
            PreparedSphereChartDeformation,
        )

        if not isinstance(
            root, (MeshingDomainBoundarySource, SurfaceSubdivisionBoundarySource)
        ):
            raise TypeError(
                "Chart fidelity requires the retained original continuous source owner."
            )
        if not isinstance(support, PreparedSurfaceSourceSupport) or not isinstance(
            deformation, (PreparedSurfaceChartDeformation, PreparedSphereChartDeformation)
        ):
            raise TypeError(
                "Chart fidelity requires original support and its actual deformation certificate."
            )
        support.require_current()
        deformation.require_bound(
            deformation.source_mesh,
            deformation.source_geometry,
            deformation.target_mesh,
            deformation.target_geometry,
        )
        if not isinstance(predecessor_fidelity, SourceFidelityCertificate):
            raise TypeError(
                "Chart renewal requires the actual predecessor continuous fidelity certificate."
            )
        predecessor_fidelity.binding.require(
            deformation.source_mesh, deformation.source_geometry
        )
        if (
            predecessor_fidelity.status != "certified"
            or predecessor_fidelity.mesh_to_source_semantics != "certified"
            or predecessor_fidelity.source_to_mesh_semantics != "certified"
        ):
            raise ValueError(
                "The predecessor lacks a two-sided continuous source theorem."
            )
        if (root.source_id, root.source_revision) != (
            support.domain.source_id,
            support.domain.source_revision,
        ) or deformation.domain_id != support.domain.domain_id:
            raise ValueError(
                "Chart fidelity changes the original scientific source authority."
            )
        self.root, self.support, self.deformation = root, support, deformation
        self.predecessor_fidelity = predecessor_fidelity

    @property
    def source_id(self) -> str:
        return self.root.source_id

    @property
    def source_revision(self) -> str:
        return self.root.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.root.ambient_dimension

    def require_current(self) -> None:
        self.support.require_current()
        self.deformation.require_bound(
            self.deformation.source_mesh,
            self.deformation.source_geometry,
            self.deformation.target_mesh,
            self.deformation.target_geometry,
        )
        self.predecessor_fidelity.binding.require(
            self.deformation.source_mesh, self.deformation.source_geometry
        )

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return self.root.boundary_distance(points)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        return self.root.boundary_samples(maximum_samples)

    def boundary_chart_cover(self, maximum_patches: int, /) -> SourceBoundaryChartCover:
        self.require_current()
        from ..discretization._cell_geometry_transfer import (
            _mapped_coordinate_expressions,
            _mapped_geometry_cells,
        )
        from ..geometry._mesh_certificates import _fidelity_proxy, _FidelityMap

        original = self.predecessor_fidelity.chart_coverage
        exact_predecessor = (
            original is None
            and self.predecessor_fidelity.mesh_to_source_upper == 0.0
            and self.predecessor_fidelity.mesh_to_source_lower == 0.0
            and self.predecessor_fidelity.source_to_mesh_upper == 0.0
            and self.predecessor_fidelity.source_to_mesh_lower == 0.0
            and self.deformation.source_domain_coverage == "certified"
            and self.deformation.target_domain_coverage == "certified"
        )
        if not exact_predecessor:
            if original is None:
                original = self.root.boundary_chart_cover(maximum_patches)
            elif (
                original.source_id,
                original.source_revision,
            ) != (self.source_id, self.source_revision):
                raise ValueError(
                    "Retained predecessor chart coverage changes source authority."
                )
            if not original.complete or original.semantics != "certified":
                return original
            findings = original.findings
            resource_counts = original.resource_counts
        else:
            findings = ()
            resource_counts = ()
        mesh, geometry = self.deformation.target_mesh, self.deformation.target_geometry
        from ..discretization._sphere_chart_deformation import (
            PreparedSphereChartDeformation,
        )

        if mesh.entity_set(2).count > maximum_patches:
            return SourceBoundaryChartCover(
                np.empty((0, 3, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                "sampled",
                False,
                self.source_id,
                self.source_revision,
                findings=(
                    MeshCertificateFinding(
                        "source_chart_cover_capacity", "unresolved", "mesh"
                    ),
                ),
            )
        proxies = [
            _fidelity_proxy(
                _FidelityMap(_mapped_coordinate_expressions(element, local), 2, row)
            )
            for row, (element, local) in enumerate(_mapped_geometry_cells(mesh, geometry))
        ]
        # Complete material-chart coverage carries every old map point to a
        # target map point within the certified displacement bound. Compose
        # that correspondence with the predecessor's two-sided source theorem
        # and the exact target-map-minus-chord enclosures; do not claim equality
        # of the old and new physical surfaces.
        rounding = max((Fraction(proxy[2]) for proxy in proxies), default=Fraction(0))
        chord = max((Fraction(proxy[1]) for proxy in proxies), default=Fraction(0))
        forward = max(
            (
                Fraction(float(value))
                for value in np.asarray(self.deformation.target_fidelity_bounds)
            ),
            default=Fraction(0),
        )
        if isinstance(self.deformation, PreparedSphereChartDeformation):
            # The genuine target radial atlas independently covers the ORIGINAL
            # authored sphere. Its pointwise full-map fidelity bound is
            # two-sided; old-to-new displacement is a different theorem and
            # must not be added to that independent source error.
            atlas = self.deformation.target_atlas
            atlas.require_bound(atlas.domain, mesh, geometry, atlas.coordinate_contract)
            if (
                atlas.coverage_status != "certified"
                or self.deformation.target_domain_coverage != "certified"
            ):
                raise ValueError(
                    "Sphere renewal lacks its independent complete original-source material cover."
                )
            bound = _upper(forward + chord + rounding)
        else:
            backward = Fraction(
                self.predecessor_fidelity.source_to_mesh_upper
            ) + Fraction(self.deformation.maximum_displacement_bound)
            bound = _upper(max(forward, backward) + chord + rounding)
        return SourceBoundaryChartCover(
            np.asarray([proxy[0] for proxy in proxies], dtype=np.float64),
            np.full((len(proxies),), bound, dtype=np.float64),
            "certified",
            True,
            self.source_id,
            self.source_revision,
            findings=findings,
            resource_counts=resource_counts,
        )


def successor_certification_inputs(
    support: PreparedSurfaceSourceSupport,
    source: CellMeshingResult,
    inputs: MeshCertificationInputs,
    lineage: MeshLineage,
    target: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    deformation: PreparedSurfaceChartDeformation
    | PreparedSphereChartDeformation
    | None = None,
) -> MeshCertificationInputs:
    """Bind a serial successor's fidelity to its certified root chart subdivision chain.

    Every other retained certification input is unchanged; renewal still runs
    the complete certifier on the actual target.
    """
    from ._certification_inputs import MeshCertificationInputs

    if not isinstance(inputs, MeshCertificationInputs):
        raise TypeError(
            "Successor certification requires the source's retained MeshCertificationInputs."
        )
    retained = inputs.source
    if deformation is not None:
        root = (
            retained.root
            if isinstance(
                retained,
                (SurfaceChartBoundarySource, SurfaceNativeRestrictionBoundarySource),
            )
            else retained
        )
        if not isinstance(
            root, (MeshingDomainBoundarySource, SurfaceSubdivisionBoundarySource)
        ):
            raise ValueError(
                "Source-chart renewal lacks the original independent continuous source theorem."
            )
        if source.certification is None or source.certification.fidelity is None:
            raise ValueError(
                "Chart source renewal requires actual predecessor fidelity evidence."
            )
        chart_source = SurfaceChartBoundarySource(
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
        )
    if target.storage is not None:
        return inputs
    if isinstance(retained, MeshingDomainBoundarySource):
        subdivision = SurfaceSubdivisionBoundarySource(
            retained, support, (lineage,), (target,), geometry
        )
    elif (
        isinstance(retained, SurfaceSubdivisionBoundarySource)
        and retained.support.support_id == support.support_id
    ):
        subdivision = SurfaceSubdivisionBoundarySource(
            retained.root,
            support,
            (*retained.lineages, lineage),
            (*retained.targets, target),
            geometry,
        )
    else:
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
    )


__all__ = [
    "SurfaceEntityClasses",
    "SurfaceSubdivisionBoundarySource",
    "classes",
    "propagate",
    "protected_edges",
    "source_associations",
    "successor_certification_inputs",
    "PreparedSurfaceCurveWitness",
    "prepare_surface_curve_witness",
    "prepare_surface_curve_chart_cover",
]
