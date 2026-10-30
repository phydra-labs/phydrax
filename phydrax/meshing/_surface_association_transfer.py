#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transfer trimmed-surface source strata through exact mesh lineage."""

from __future__ import annotations

from fractions import Fraction
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellMesh
from ..geometry._meshing_domain import MeshingDomain, PatchCurveUse
from ..geometry.surface import SurfaceMetadata, SurfaceModel
from ..typing import checked
from ._association import (
    _child_sources,
    AssociationEntityClasses,
    AssociationPropagationError,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from ._association_routes import (
    associations_for_namespace,
    entity_rows,
    incidence_pairs,
    lineage_association_rows,
    require_transfer_lineage,
    target_dimension,
)
from ._lineage import MeshLineage, VertexInterpolationStencil
from ._result import CellMeshingResult


def _domain_row(
    domain: MeshingDomain, dimension: int, index: int, path: tuple[str, ...], /
) -> int:
    matches = tuple(
        row
        for row, (source_index, occurrence) in enumerate(
            zip(
                domain.source_indices[dimension],
                domain.source_occurrences[dimension],
                strict=True,
            )
        )
        if source_index == index and occurrence == path
    )
    if len(matches) != 1:
        raise ValueError(
            "Surface association source stratum must resolve to one occurrence-qualified domain row."
        )
    return matches[0]


def _class_code(
    dimension: int, index: int, path: tuple[str, ...], /
) -> tuple[int, int, tuple[str, ...]]:
    return dimension, index, path


def _simplex_edges(mesh: CellMesh, /) -> np.ndarray:
    """Edge vertex rows aligned with the mesh's canonical edge entity set."""
    from ._topology_edit import entity_keys

    if any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError("Surface association transfer requires triangular incidence.")
    keys = np.asarray(entity_keys(mesh, 1), dtype=np.int64)
    vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64).tolist()
        )
    }
    return np.asarray(
        [[vertex_rows[int(value)] for value in key] for key in keys.tolist()],
        dtype=np.int64,
    ).reshape((-1, 2))


def _target_edge_parameter(
    target_vertex_id: int,
    source_endpoint_ids: tuple[int, int],
    stencil: VertexInterpolationStencil,
    /,
) -> Fraction:
    if target_vertex_id == source_endpoint_ids[0]:
        return Fraction(0)
    if target_vertex_id == source_endpoint_ids[1]:
        return Fraction(1)
    stencil_ids = np.asarray(stencil.target_global_ids, dtype=np.int64)
    rows = np.flatnonzero(stencil_ids == target_vertex_id)
    if rows.size != 1:
        raise ValueError(
            "A split trim-edge vertex requires one exact interpolation-stencil row."
        )
    row = int(rows[0])
    source_ids = np.asarray(stencil.source_global_ids, dtype=np.int64)[row]
    weights = np.asarray(stencil.weights, dtype=np.float64)[row]
    valid = np.asarray(stencil.valid, dtype=np.bool_)[row] & (weights != 0.0)
    support = source_ids[valid]
    coefficients = weights[valid]
    if not np.all(np.isin(support, np.asarray(source_endpoint_ids, dtype=np.int64))):
        raise ValueError(
            "A trim-edge split stencil must remain on its exact source edge."
        )
    parameter = sum(
        (
            Fraction.from_float(float(weight))
            for identifier, weight in zip(
                support.tolist(), coefficients.tolist(), strict=True
            )
            if identifier == source_endpoint_ids[1]
        ),
        Fraction(0),
    )
    if parameter <= 0 or parameter >= 1:
        raise ValueError(
            "A trim-edge split parameter must lie strictly inside its source interval."
        )
    return parameter


def _trim_child_parameter(
    edge_table: GeometryAssociation,
    table_row: int,
    vertex_id: int,
    source_mesh: CellMesh,
    source_edge_row: int,
    stencil: VertexInterpolationStencil,
    /,
) -> Fraction:
    """Exact source-curve parameter of a vertex created on one trim edge."""
    interval = edge_table.source_parameter_intervals[table_row]
    if interval is None:
        raise ValueError(
            "Trimmed curve-edge association requires exact source parameter intervals."
        )
    sign = int(np.asarray(edge_table.orientations, dtype=np.int8)[table_row])
    if sign not in (-1, 1):
        raise ValueError("Trimmed curve-edge parents must retain source orientation.")
    endpoints = _simplex_edges(source_mesh)[source_edge_row]
    source_vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    fraction = _target_edge_parameter(
        vertex_id,
        (int(source_vertex_ids[endpoints[0]]), int(source_vertex_ids[endpoints[1]])),
        stencil,
    )
    if sign < 0:
        fraction = 1 - fraction
    return interval[0] + fraction * (interval[1] - interval[0])


def _trim_curve_point(
    domain: MeshingDomain, curve: int, parameter: Fraction, /
) -> np.ndarray:
    from .providers._native_surface import _source_curve_point

    return np.asarray(_source_curve_point(domain, curve, parameter), dtype=np.float64)


def _use_chart(use: PatchCurveUse, parameter: float, /) -> np.ndarray:
    native = use.first + parameter * (use.last - use.first)
    return np.asarray(
        use.pcurve.evaluate(jnp.asarray(native, dtype=jnp.float64)), dtype=np.float64
    ).reshape((2,))


def _source_vertex_chart(
    domain: MeshingDomain,
    patch: int,
    dimension: int,
    index: int,
    parameters: np.ndarray,
    /,
) -> np.ndarray:
    """One chart point of a source vertex on ``patch``, or NaN when not unique.

    Patch vertices keep their UV; curve and corner vertices are mapped through
    the patch's trim p-curves at their canonical curve parameter or endpoint.
    """
    missing = np.full((2,), np.nan, dtype=np.float64)
    if dimension == 2:
        return np.asarray(parameters, dtype=np.float64).reshape((2,))
    uses = tuple(
        use
        for loop in domain.patches[patch].loops
        for use in loop
        if isinstance(use, PatchCurveUse)
    )
    if dimension == 1:
        candidates = [
            _use_chart(use, float(parameters[0])) for use in uses if use.curve == index
        ]
    elif dimension == 0:
        candidates = [
            _use_chart(use, endpoint)
            for use in uses
            for endpoint, corner in (
                (0.0, domain.curves[use.curve].start),
                (1.0, domain.curves[use.curve].end),
            )
            if corner == index
        ]
    else:
        return missing
    if not candidates or any(
        not np.array_equal(candidates[0], candidate) for candidate in candidates[1:]
    ):
        return missing
    return candidates[0]


@final
class SurfaceAssociationTransfer(StrictModule, NonTrainableState):
    """Carry occurrence-qualified face and trim-curve classes on a source chart."""

    domain: MeshingDomain
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(self, domain: MeshingDomain, /) -> None:
        if not isinstance(domain, MeshingDomain):
            raise TypeError("domain must be MeshingDomain.")
        self.domain = domain
        domain.require_current(domain.source_id, domain.source_revision)
        self.transfer_id = canonical_fingerprint(
            {"kind": "surface-association-transfer", "domain": domain.domain_id}
        )

    @property
    def source_namespaces(self) -> tuple[tuple[str, str], ...]:
        return ((self.domain.source_id, self.domain.source_revision),)

    def validate_source(self, source: CellMeshingResult, /) -> None:
        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        self.domain.require_current(self.domain.source_id, self.domain.source_revision)
        if source.mesh.topological_dimension != 2 or source.mesh.ambient_dimension != 3:
            raise ValueError(
                "Surface association transfer requires a two-dimensional sheet in three-space."
            )
        if (
            source.surface_source is None
            or source.surface_source.domain.domain_id != self.domain.domain_id
        ):
            raise ValueError(
                "Surface associations require the exact trimmed source chart carried by the result."
            )
        source.surface_source.require_current()
        tables = associations_for_namespace(
            source,
            self.source_namespaces[0],
            kinds=(GeometryAssociationKind.SURFACE, GeometryAssociationKind.CURVE),
        )
        by_dimension = {target_dimension(source.mesh, value): value for value in tables}
        if 0 not in by_dimension or 2 not in by_dimension:
            raise ValueError(
                "Trimmed surface source tables must include vertex and triangle associations."
            )
        by_dimension[0].target_rows(
            np.asarray(source.mesh.entity_set(0).entity_ids, dtype=np.int64)
        )
        by_dimension[2].target_rows(
            np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
        )
        if by_dimension[0].association_kind is not GeometryAssociationKind.SURFACE:
            raise ValueError(
                "Surface vertex rows must retain the surface source namespace."
            )
        for association in tables:
            dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
            indices = np.asarray(association.source_indices, dtype=np.int64)
            for row, (dimension, index) in enumerate(
                zip(dimensions.tolist(), indices.tolist(), strict=True)
            ):
                if dimension < 0 or index < 0:
                    raise ValueError(
                        "Trimmed surface associations must name explicit source strata."
                    )
                expected = self.domain.entity_id(
                    dimension,
                    _domain_row(
                        self.domain,
                        dimension,
                        index,
                        association.source_occurrence_paths[row],
                    ),
                )
                if association.source_entity_ids[row] != expected:
                    raise ValueError(
                        "Trimmed surface association entity IDs must match the exact source occurrence."
                    )
                if (
                    association.association_kind is GeometryAssociationKind.CURVE
                    and dimension != 1
                ):
                    raise ValueError(
                        "Trimmed curve-edge rows must identify source curves."
                    )
                if (
                    association.association_kind is GeometryAssociationKind.SURFACE
                    and target_dimension(source.mesh, association) == 2
                    and dimension != 2
                ):
                    raise ValueError(
                        "Trimmed surface cells must retain source patch strata."
                    )

    def classes(
        self, source: CellMeshingResult, /
    ) -> tuple[AssociationEntityClasses, ...]:
        self.validate_source(source)
        mesh = source.mesh
        tables = {
            target_dimension(mesh, association): association
            for association in associations_for_namespace(
                source,
                self.source_namespaces[0],
                kinds=(GeometryAssociationKind.SURFACE, GeometryAssociationKind.CURVE),
            )
        }
        result = []
        for dimension in range(3):
            count = mesh.entity_set(dimension).count
            dimensions = np.full((count,), -1, dtype=np.int64)
            indices = np.full((count,), -1, dtype=np.int64)
            resolved = np.zeros((count,), dtype=np.bool_)
            paths: list[tuple[str, ...] | None] = [None] * count
            association = tables.get(dimension)
            if association is not None:
                rows = entity_rows(
                    mesh,
                    dimension,
                    np.asarray(association.target_global_ids, dtype=np.int64),
                )
                valid = rows >= 0
                dimensions[rows[valid]] = np.asarray(
                    association.source_dimensions, dtype=np.int64
                )[valid]
                indices[rows[valid]] = np.asarray(
                    association.source_indices, dtype=np.int64
                )[valid]
                resolved[rows[valid]] = np.asarray(association.resolved, dtype=np.bool_)[
                    valid
                ]
                for source_row, mesh_row in enumerate(rows.tolist()):
                    if mesh_row >= 0:
                        paths[mesh_row] = association.source_occurrence_paths[source_row]
            if dimension == 1:
                cell_association = tables[2]
                cells = entity_rows(
                    mesh,
                    2,
                    np.asarray(cell_association.target_global_ids, dtype=np.int64),
                )
                cell_dimensions = np.asarray(
                    cell_association.source_dimensions, dtype=np.int64
                )
                cell_indices = np.asarray(cell_association.source_indices, dtype=np.int64)
                cell_paths = cell_association.source_occurrence_paths
                adjacency = incidence_pairs(mesh, 1, 2)
                for edge in np.flatnonzero(~resolved).tolist():
                    incident = adjacency[adjacency[:, 0] == edge, 1]
                    values = []
                    for cell_row in incident.tolist():
                        position = np.flatnonzero(cells == cell_row)
                        if position.size != 1:
                            raise ValueError(
                                "Trimmed surface cell association must cover every incident triangle."
                            )
                        row = int(position[0])
                        values.append(
                            (
                                int(cell_dimensions[row]),
                                int(cell_indices[row]),
                                cell_paths[row],
                            )
                        )
                    if values and len(set(values)) == 1:
                        dimensions[edge], indices[edge], paths[edge] = values[0]
                        resolved[edge] = True
            if not np.all(resolved):
                raise AssociationPropagationError(
                    "Trimmed surface carrier entities must resolve to one source stratum.",
                    np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)[
                        ~resolved
                    ],
                )
            codes = np.full((count,), -1, dtype=np.int64)
            keys = tuple(
                sorted(
                    {
                        _class_code(
                            int(dimensions[row]), int(indices[row]), paths[row] or ()
                        )
                        for row in np.flatnonzero(resolved).tolist()
                    }
                )
            )
            code_by_key = {key: code for code, key in enumerate(keys)}
            for row in np.flatnonzero(resolved).tolist():
                codes[row] = code_by_key[
                    _class_code(int(dimensions[row]), int(indices[row]), paths[row] or ())
                ]
            result.append(AssociationEntityClasses(dimensions, indices, resolved, codes))
        return tuple(result)

    @checked
    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        vertex_stencil: VertexInterpolationStencil | None,
    ) -> tuple[GeometryAssociation, ...]:
        self.validate_source(source)
        if not isinstance(lineage, MeshLineage) or not isinstance(target, CellMesh):
            raise TypeError("lineage and target must be MeshLineage and CellMesh values.")
        if vertex_stencil is None or not isinstance(
            vertex_stencil, VertexInterpolationStencil
        ):
            raise TypeError(
                "Surface association transfer requires the exact vertex interpolation stencil."
            )
        if target.topological_dimension != 2 or target.ambient_dimension != 3:
            raise ValueError(
                "Surface association transfer requires a triangular target sheet in three-space."
            )
        if any(block.cell_kind != "triangle" for block in target.blocks):
            raise ValueError(
                "Surface association transfer requires triangular target cells."
            )
        require_transfer_lineage(source.mesh, lineage, target, dimensions=(0, 1, 2))
        tables = {
            target_dimension(source.mesh, association): association
            for association in associations_for_namespace(
                source,
                self.source_namespaces[0],
                kinds=(GeometryAssociationKind.SURFACE, GeometryAssociationKind.CURVE),
            )
        }
        output = [self._surface_vertices(source, lineage, target, vertex_stencil, tables)]
        output.append(self._surface_cells(source, lineage, target, tables[2]))
        if 1 in tables:
            curve_edges = self._curve_edges(
                source, lineage, target, vertex_stencil, tables[1]
            )
            if curve_edges is not None:
                output.append(curve_edges)
        return tuple(output)

    def remap_boundary(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
    ) -> SurfaceModel | None:
        """Carry the trimmed-sheet boundary model through the same cell lineage."""
        boundary = source.boundary
        if boundary is None:
            return None
        self.validate_source(source)
        if not isinstance(lineage, MeshLineage) or not isinstance(target, CellMesh):
            raise TypeError("lineage and target must be MeshLineage and CellMesh values.")
        metadata = boundary.metadata
        if (
            metadata.source_id,
            metadata.source_revision,
        ) != self.source_namespaces[0]:
            raise ValueError(
                "Trimmed surface boundary metadata must match its association namespace."
            )
        if (
            metadata.coordinate_contract.spatial_id
            != source.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Trimmed surface boundary must preserve the source coordinate contract."
            )
        if boundary.mesh.topology_id != source.mesh.topology_id:
            raise ValueError(
                "Trimmed surface boundary must bind the source sheet topology."
            )
        if (
            boundary.selections
            or boundary.interfaces
            or boundary.orientation_repair is not None
        ):
            raise ValueError(
                "Surface boundary remapping refuses unresolved selections, interfaces, or orientation repairs."
            )
        require_transfer_lineage(source.mesh, lineage, target, dimensions=(2,))
        source_tags = boundary.metadata.cell_tags
        source_block_ids = np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in boundary.mesh.blocks
            ]
        )
        if source_tags and len(source_tags) != source_block_ids.size:
            raise ValueError(
                "Trimmed surface tags must align with source boundary cells."
            )
        tag_by_id = (
            dict(zip(source_block_ids.tolist(), source_tags, strict=True))
            if source_tags
            else {}
        )
        record = lineage.entity_lineage(2)
        old_ids = np.asarray(record.source_global_ids, dtype=np.int64)
        new_ids = np.asarray(record.target_global_ids, dtype=np.int64)
        target_block_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in target.blocks]
        )
        target_tags: list[str] = []
        for identifier in target_block_ids.tolist():
            parents = old_ids[new_ids == identifier]
            inherited = {
                tag_by_id[int(parent)]
                for parent in parents.tolist()
                if int(parent) in tag_by_id
            }
            if not inherited and identifier in tag_by_id:
                inherited = {tag_by_id[identifier]}
            if len(inherited) > 1:
                raise ValueError("Trimmed surface tags conflict across lineage parents.")
            if source_tags and not inherited:
                raise ValueError("Trimmed surface tag lineage omitted a target cell.")
            if inherited:
                target_tags.append(next(iter(inherited)))
        metadata = SurfaceMetadata(
            source_id=boundary.metadata.source_id,
            source_revision=boundary.metadata.source_revision,
            coordinate_contract=boundary.metadata.coordinate_contract,
            provenance=(*boundary.metadata.provenance, lineage.lineage_id),
            cell_tags=tuple(target_tags) if source_tags else (),
        )
        return SurfaceModel(target, metadata)

    def _surface_vertices(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        stencil: VertexInterpolationStencil,
        tables: dict[int, GeometryAssociation],
        /,
    ) -> GeometryAssociation:
        old = tables[0]
        source_mesh, target_mesh = source.mesh, target
        source_vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
        target_vertex_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
        source_rows = {
            int(identifier): row
            for row, identifier in enumerate(source_vertex_ids.tolist())
        }
        vertex_dimensions = np.full(target_vertex_ids.shape, -1, dtype=np.int8)
        vertex_indices = np.full(target_vertex_ids.shape, -1, dtype=np.int64)
        residuals = np.zeros(target_vertex_ids.shape, dtype=np.float64)
        parameters = np.full((target_vertex_ids.size, 2), np.nan, dtype=np.float64)
        source_ids: list[str] = [""] * target_vertex_ids.size
        paths: list[tuple[str, ...]] = [()] * target_vertex_ids.size
        roles: list[GeometrySourceEntityRole | None] = [None] * target_vertex_ids.size
        parent_dimensions = np.full(target_vertex_ids.shape, -1, dtype=np.int8)
        parent_ids = np.full(target_vertex_ids.shape, -1, dtype=np.int64)
        old_lookup = {
            int(identifier): row
            for row, identifier in enumerate(
                np.asarray(old.target_global_ids, dtype=np.int64).tolist()
            )
        }
        created_mask = ~np.isin(target_vertex_ids, source_vertex_ids)
        created_ids = np.sort(target_vertex_ids[created_mask])
        child_dimensions, child_rows = _child_sources(
            lineage, source_mesh, target_mesh, created_ids, stencil
        )
        child_position = {
            int(identifier): row for row, identifier in enumerate(created_ids.tolist())
        }
        edge_table = tables.get(1)
        cell_table = tables[2]
        cell_source_dimensions = np.asarray(cell_table.source_dimensions, dtype=np.int64)
        cell_source_indices = np.asarray(cell_table.source_indices, dtype=np.int64)
        edge_ids = np.asarray(source_mesh.entity_set(1).entity_ids, dtype=np.int64)
        cell_ids = np.asarray(source_mesh.entity_set(2).entity_ids, dtype=np.int64)
        edge_assoc_rows = (
            {}
            if edge_table is None
            else {
                int(identifier): row
                for row, identifier in enumerate(
                    np.asarray(edge_table.target_global_ids, dtype=np.int64).tolist()
                )
            }
        )
        cell_assoc_rows = {
            int(identifier): row
            for row, identifier in enumerate(
                np.asarray(cell_table.target_global_ids, dtype=np.int64).tolist()
            )
        }
        projected_rows: list[int] = []
        projected_patches: list[int] = []
        projected_seeds: list[tuple[float, float]] = []
        for target_row, identifier in enumerate(target_vertex_ids.tolist()):
            if identifier in source_rows and identifier in old_lookup:
                parent_row = old_lookup[identifier]
                vertex_dimensions[target_row] = np.asarray(
                    old.source_dimensions, dtype=np.int8
                )[parent_row]
                vertex_indices[target_row] = np.asarray(
                    old.source_indices, dtype=np.int64
                )[parent_row]
                residuals[target_row] = np.asarray(old.residuals, dtype=np.float64)[
                    parent_row
                ]
                parameters[target_row] = np.asarray(old.parameters, dtype=np.float64)[
                    parent_row
                ]
                source_ids[target_row] = old.source_entity_ids[parent_row]
                paths[target_row] = old.source_occurrence_paths[parent_row]
                roles[target_row] = old.source_roles[parent_row]
                parent_dimensions[target_row] = 0
                parent_ids[target_row] = identifier
                continue
            position = child_position.get(identifier)
            if position is None or child_dimensions[position] < 1:
                raise AssociationPropagationError(
                    "Created trim-sheet vertices require exact edge or face source support.",
                    np.asarray((identifier,), dtype=np.int64),
                )
            source_dimension = int(child_dimensions[position])
            source_entity_row = int(child_rows[position])
            selected_table = None
            selected_table_row = -1
            if source_dimension == 1 and edge_table is not None:
                source_edge_id = int(edge_ids[source_entity_row])
                selected_table_row = edge_assoc_rows.get(source_edge_id, -1)
                if selected_table_row >= 0:
                    selected_table = edge_table
            if selected_table is None:
                if source_dimension == 1:
                    incident = incidence_pairs(source_mesh, 1, 2)
                    cell_rows = incident[incident[:, 0] == source_entity_row, 1]
                    candidates = [
                        cell_assoc_rows[int(cell_ids[row])] for row in cell_rows.tolist()
                    ]
                else:
                    candidates = [cell_assoc_rows[int(cell_ids[source_entity_row])]]
                candidate_rows = tuple(dict.fromkeys(candidates))
                identities = {
                    (
                        cell_table.source_entity_ids[row],
                        int(cell_source_dimensions[row]),
                        int(cell_source_indices[row]),
                        cell_table.source_occurrence_paths[row],
                    )
                    for row in candidate_rows
                }
                if len(identities) != 1:
                    raise AssociationPropagationError(
                        "Created trim-sheet vertices cross source patch strata.",
                        np.asarray((identifier,), dtype=np.int64),
                    )
                selected_table = cell_table
                selected_table_row = candidate_rows[0]
            vertex_dimensions[target_row] = np.asarray(
                selected_table.source_dimensions, dtype=np.int8
            )[selected_table_row]
            vertex_indices[target_row] = np.asarray(
                selected_table.source_indices, dtype=np.int64
            )[selected_table_row]
            source_ids[target_row] = selected_table.source_entity_ids[selected_table_row]
            paths[target_row] = selected_table.source_occurrence_paths[selected_table_row]
            roles[target_row] = selected_table.source_roles[selected_table_row]
            parent_dimensions[target_row] = source_dimension
            parent_ids[target_row] = int(
                source_mesh.entity_set(source_dimension).entity_ids[source_entity_row]
            )
            if selected_table is edge_table and edge_table is not None:
                # A bisected trim-edge child lies on the chord of its source
                # edge, not on the patch: the trim curve continues past the
                # chart boundary, so a closest-point query on the patch lands
                # outside the trim. Its stratum is the source curve at the exact
                # split parameter of the parent edge's source interval.
                curve_parameter = _trim_child_parameter(
                    edge_table,
                    selected_table_row,
                    identifier,
                    source_mesh,
                    source_entity_row,
                    stencil,
                )
                curve_point = _trim_curve_point(
                    self.domain, int(vertex_indices[target_row]), curve_parameter
                )
                parameters[target_row] = (float(curve_parameter), np.nan)
                residuals[target_row] = float(
                    np.linalg.norm(
                        np.asarray(target_mesh.coordinates, dtype=np.float64)[target_row]
                        - curve_point
                    )
                )
                continue
            patch = self._patch_index(
                int(vertex_dimensions[target_row]),
                int(vertex_indices[target_row]),
                paths[target_row],
            )
            stencil_row = int(
                np.flatnonzero(
                    np.asarray(stencil.target_global_ids, dtype=np.int64) == identifier
                )[0]
            )
            stencil_sources = np.asarray(stencil.source_global_ids, dtype=np.int64)[
                stencil_row
            ]
            stencil_weights = np.asarray(stencil.weights, dtype=np.float64)[stencil_row]
            stencil_valid = np.asarray(stencil.valid, dtype=np.bool_)[stencil_row]
            source_vertex_rows = entity_rows(
                source_mesh, 0, stencil_sources[stencil_valid]
            )
            if np.any(source_vertex_rows < 0):
                raise ValueError(
                    "Surface vertex stencil references unknown source vertices."
                )
            source_parent_rows = old.target_rows(stencil_sources[stencil_valid])
            uv = np.stack(
                [
                    _source_vertex_chart(
                        self.domain,
                        patch,
                        int(np.asarray(old.source_dimensions)[int(parent_row)]),
                        int(np.asarray(old.source_indices)[int(parent_row)]),
                        np.asarray(old.parameters, dtype=np.float64)[int(parent_row)],
                    )
                    for parent_row in np.asarray(source_parent_rows).tolist()
                ]
            )
            # The seed is the stencil mean of the parents' charts on this patch;
            # parents without one patch chart point are omitted, and the
            # projection itself is verified below.
            weights = stencil_weights[stencil_valid]
            finite = np.all(np.isfinite(uv), axis=1)
            if not np.any(finite) or np.sum(weights[finite]) <= 0.0:
                raise AssociationPropagationError(
                    "Trimmed surface vertex has no patch-chart stencil sample to seed its projection.",
                    np.asarray((identifier,), dtype=np.int64),
                )
            seed = np.sum(uv[finite] * weights[finite, None], axis=0) / np.sum(
                weights[finite]
            )
            projected_rows.append(target_row)
            projected_patches.append(patch)
            projected_seeds.append((float(seed[0]), float(seed[1])))
        if projected_rows:
            rows = np.asarray(projected_rows, dtype=np.int64)
            projected = self.domain.project(
                np.asarray(target_mesh.coordinates, dtype=np.float64)[rows],
                np.asarray(projected_patches, dtype=np.int32),
                np.asarray(projected_seeds, dtype=np.float64),
            )
            if not np.all(np.asarray(projected.converged, dtype=np.bool_)):
                raise AssociationPropagationError(
                    "Trimmed surface source projection could not verify inherited patch classes.",
                    target_vertex_ids[
                        rows[~np.asarray(projected.converged, dtype=np.bool_)]
                    ],
                )
            parameters[rows] = np.asarray(projected.parameters, dtype=np.float64)
            residuals[rows] = np.asarray(projected.distances, dtype=np.float64)
        if np.any(vertex_dimensions < 0) or any(not value for value in source_ids):
            raise ValueError("Every target surface vertex must retain a source stratum.")
        return GeometryAssociation(
            GeometryAssociationKind.SURFACE,
            self.domain.source_id,
            self.domain.source_revision,
            target_mesh.entity_set(0).entity_set_id,
            target_vertex_ids,
            tuple(source_ids),
            residuals,
            source_dimensions=vertex_dimensions,
            source_indices=vertex_indices,
            source_roles=tuple(roles),
            source_occurrence_paths=tuple(paths),
            parameters=parameters,
            parent_dimensions=parent_dimensions,
            parent_ids=parent_ids,
            exact=False,
            provenance=GeometryAssociationProvenance.LINEAGE,
            parent_association_id=old.association_id,
        )

    def _patch_index(self, dimension: int, index: int, path: tuple[str, ...], /) -> int:
        if dimension == 2:
            return _domain_row(self.domain, 2, index, path)
        if dimension == 1:
            curve = _domain_row(self.domain, 1, index, path)
            return int(self.domain.curve_owners[curve, 0])
        raise ValueError(
            "A created trim-sheet vertex must inherit a source curve or patch."
        )

    def _surface_cells(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        parent: GeometryAssociation,
        /,
    ) -> GeometryAssociation:
        rows = lineage_association_rows(parent, source.mesh, lineage, target, 2)
        if rows.target_ids.size != target.entity_set(2).count:
            raise AssociationPropagationError(
                "Every target trim-sheet cell must inherit one source patch.",
                np.setdiff1d(
                    np.asarray(target.entity_set(2).entity_ids), rows.target_ids
                ),
            )
        source_rows = rows.association_rows
        target_ids = rows.target_ids
        cell_vertices: dict[int, list[int]] = {}
        for block in target.blocks:
            if not isinstance(block, CellBlock) or block.cell_kind != "triangle":
                raise ValueError("Trimmed surface target must retain triangle blocks.")
            for identifier, vertices in zip(
                np.asarray(block.global_ids, dtype=np.int64).tolist(),
                np.asarray(block.vertices, dtype=np.int64).tolist(),
                strict=True,
            ):
                cell_vertices[int(identifier)] = vertices
        cells = np.asarray(
            [cell_vertices[int(identifier)] for identifier in target_ids.tolist()],
            dtype=np.int64,
        )
        centroid = np.mean(
            np.asarray(target.coordinates, dtype=np.float64)[cells], axis=1
        )
        patch_rows = np.asarray(
            [
                _domain_row(
                    self.domain,
                    int(parent.source_dimensions[row]),
                    int(parent.source_indices[row]),
                    parent.source_occurrence_paths[row],
                )
                for row in source_rows.tolist()
            ],
            dtype=np.int32,
        )
        seeds = np.asarray(parent.parameters, dtype=np.float64)[source_rows]
        projected = self.domain.project(centroid, patch_rows, seeds)
        if not np.all(np.asarray(projected.converged, dtype=np.bool_)):
            raise AssociationPropagationError(
                "Trimmed surface cell projection could not verify inherited patch classes.",
                target_ids[~np.asarray(projected.converged, dtype=np.bool_)],
            )
        return GeometryAssociation(
            GeometryAssociationKind.SURFACE,
            parent.source_id,
            parent.source_revision,
            target.entity_set(2).entity_set_id,
            target_ids,
            tuple(parent.source_entity_ids[int(row)] for row in source_rows.tolist()),
            np.asarray(projected.distances, dtype=np.float64),
            source_dimensions=np.asarray(parent.source_dimensions, dtype=np.int8)[
                source_rows
            ],
            source_indices=np.asarray(parent.source_indices, dtype=np.int64)[source_rows],
            source_roles=tuple(
                parent.source_roles[int(row)] for row in source_rows.tolist()
            ),
            source_occurrence_paths=tuple(
                parent.source_occurrence_paths[int(row)] for row in source_rows.tolist()
            ),
            parameters=np.asarray(projected.parameters, dtype=np.float64),
            parent_dimensions=np.full(target_ids.shape, 2, dtype=np.int8),
            parent_ids=rows.parent_ids,
            exact=False,
            provenance=GeometryAssociationProvenance.LINEAGE,
            parent_association_id=parent.association_id,
        )

    def _curve_edges(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        stencil: VertexInterpolationStencil,
        parent: GeometryAssociation,
        /,
    ) -> GeometryAssociation | None:
        rows = lineage_association_rows(parent, source.mesh, lineage, target, 1)
        if rows.target_ids.size == 0:
            return None
        source_edges = _simplex_edges(source.mesh)
        target_edges = _simplex_edges(target)
        source_vertex_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
        target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
        parent_edge_row = entity_rows(source.mesh, 1, rows.parent_ids)
        target_edge_row = entity_rows(target, 1, rows.target_ids)
        source_assoc_rows = rows.association_rows
        target_parameters: list[tuple[Fraction, Fraction] | None] = []
        orientations = np.empty(rows.target_ids.shape, dtype=np.int8)
        old_orientations = np.asarray(parent.orientations, dtype=np.int8)
        old_intervals = parent.source_parameter_intervals
        for output_row, (target_row, source_row, edge_row) in enumerate(
            zip(
                target_edge_row.tolist(),
                source_assoc_rows.tolist(),
                parent_edge_row.tolist(),
                strict=True,
            )
        ):
            interval = old_intervals[source_row]
            if interval is None:
                raise ValueError(
                    "Trimmed curve-edge association requires exact source parameter intervals."
                )
            if edge_row < 0:
                raise ValueError(
                    "Trimmed curve-edge lineage must identify its source parent edge."
                )
            source_edge_vertices = source_edges[edge_row]
            source_endpoints = (
                int(source_vertex_ids[source_edge_vertices[0]]),
                int(source_vertex_ids[source_edge_vertices[1]]),
            )
            target_edge_vertices = target_edges[target_row]
            target_endpoints = (
                int(target_vertex_ids[target_edge_vertices[0]]),
                int(target_vertex_ids[target_edge_vertices[1]]),
            )
            sign = int(old_orientations[source_row])
            if sign not in (-1, 1):
                raise ValueError(
                    "Trimmed curve-edge parents must retain source orientation."
                )
            first_fraction = _target_edge_parameter(
                target_endpoints[0], source_endpoints, stencil
            )
            last_fraction = _target_edge_parameter(
                target_endpoints[1], source_endpoints, stencil
            )
            if sign < 0:
                first_fraction, last_fraction = 1 - first_fraction, 1 - last_fraction
            first_parameter = interval[0] + first_fraction * (interval[1] - interval[0])
            last_parameter = interval[0] + last_fraction * (interval[1] - interval[0])
            if first_parameter == last_parameter:
                raise ValueError(
                    "Trimmed curve-edge child must retain a nonzero exact source interval."
                )
            target_parameters.append(
                (
                    min(first_parameter, last_parameter),
                    max(first_parameter, last_parameter),
                )
            )
            orientations[output_row] = 1 if last_parameter > first_parameter else -1
        return GeometryAssociation(
            GeometryAssociationKind.CURVE,
            parent.source_id,
            parent.source_revision,
            target.entity_set(1).entity_set_id,
            rows.target_ids,
            tuple(
                parent.source_entity_ids[int(row)] for row in source_assoc_rows.tolist()
            ),
            np.asarray(parent.residuals, dtype=np.float64)[source_assoc_rows],
            source_dimensions=np.asarray(parent.source_dimensions, dtype=np.int8)[
                source_assoc_rows
            ],
            source_indices=np.asarray(parent.source_indices, dtype=np.int64)[
                source_assoc_rows
            ],
            source_roles=tuple(
                parent.source_roles[int(row)] for row in source_assoc_rows.tolist()
            ),
            source_occurrence_paths=tuple(
                parent.source_occurrence_paths[int(row)]
                for row in source_assoc_rows.tolist()
            ),
            source_parameter_intervals=tuple(target_parameters),
            orientations=orientations,
            parent_dimensions=np.ones(rows.target_ids.shape, dtype=np.int8),
            parent_ids=rows.parent_ids,
            exact=False,
            provenance=GeometryAssociationProvenance.LINEAGE,
            parent_association_id=parent.association_id,
        )


__all__ = ["SurfaceAssociationTransfer"]
