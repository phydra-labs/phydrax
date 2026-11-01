# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Exact chart-witness transport for serial surface subdivision."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

import numpy as np

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.geometry._surface_source_support as surface_support_owner
import phydrax.meshing._association as association_owner
import phydrax.meshing._lineage as lineage_owner
import phydrax.meshing._result as result_owner

from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._surface_chart_records import (
    rational_chart_tokens,
    SurfaceChartWitness,
    SurfaceRootMeshCellBinding,
)
from ..geometry._rational import rational_dot
from ..linalg._exact import solve_exact
from ..typing import checked


type SurfaceVertexWeights = tuple[tuple[int, Fraction], ...]


def _support_weights(
    points: np.ndarray, point: np.ndarray, /
) -> tuple[Fraction, ...] | None:
    """Return exact affine coordinates in a segment or triangle's span."""
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
    gram = tuple(tuple(rational_dot(a, b) for b in edges) for a in edges)
    rhs = tuple(rational_dot(edge, offset) for edge in edges)
    try:
        solution = solve_exact(
            gram,
            tuple((value,) for value in rhs),
            singular_message="Surface subdivision support is rank deficient.",
        )
    except ValueError:
        return None
    local = tuple(row[0] for row in solution)
    return (1 - sum(local, Fraction(0)), *local)


def _entity_rows(
    mesh: cell_mesh_owner.CellMesh,
    dimension: int,
    identifiers: np.ndarray,
    /,
) -> np.ndarray:
    """Resolve globally identified mesh entities to rows without positional identity."""
    ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    values = np.asarray(identifiers, dtype=np.int64)
    if not ids.size:
        return np.full(values.shape, -1, dtype=np.int64)
    positions = np.searchsorted(sorted_ids, values)
    present = positions < sorted_ids.size
    present[present] &= sorted_ids[positions[present]] == values[present]

    rows = np.full(values.shape, -1, dtype=np.int64)
    rows[present] = order[positions[present]]
    return rows


@checked
def created_weights(
    vertex_stencil: lineage_owner.VertexInterpolationStencil,
    source_mesh: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    /,
) -> dict[int, SurfaceVertexWeights]:
    """Read exact created-vertex support from the accepted interpolation stencil."""
    if (
        vertex_stencil.source_entity_set_id != source_mesh.entity_set(0).entity_set_id
        or vertex_stencil.target_entity_set_id != target.entity_set(0).entity_set_id
    ):
        raise ValueError("Surface subdivision stencil endpoints do not match its meshes.")
    source_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    stencil_target_ids = np.asarray(vertex_stencil.target_global_ids, dtype=np.int64)
    if not np.array_equal(
        np.sort(stencil_target_ids, kind="stable"),
        np.sort(target_ids, kind="stable"),
    ):
        raise ValueError("Surface subdivision stencil must cover the target vertices.")
    source_id_set = set(source_ids.tolist())
    created = sorted(set(target_ids.tolist()) - source_id_set)
    if not created:
        return {}
    stencil_row_by_id = {
        int(identifier): row for row, identifier in enumerate(stencil_target_ids.tolist())
    }
    stencil_sources = np.asarray(vertex_stencil.source_global_ids, dtype=np.int64)
    stencil_coefficients = np.asarray(vertex_stencil.weights, dtype=np.float64)
    stencil_valid = np.asarray(vertex_stencil.valid, dtype=np.bool_)
    source_coordinates = np.asarray(source_mesh.coordinates, dtype=np.float64)
    target_coordinates = np.asarray(target.coordinates, dtype=np.float64)
    weights: dict[int, SurfaceVertexWeights] = {}
    for child in created:
        witness = np.asarray((child,), dtype=np.int64)
        child_rows = _entity_rows(target, 0, witness)
        source_rows = np.flatnonzero(source_ids == child)
        if child_rows.size != 1 or source_rows.size:
            raise association_owner.AssociationPropagationError(
                "Surface subdivision target vertex identity is malformed.", witness
            )
        stencil_row = stencil_row_by_id[child]
        support_rows = np.flatnonzero(
            stencil_valid[stencil_row] & (stencil_coefficients[stencil_row] != 0.0)
        )
        combined: dict[int, Fraction] = {}
        for column in support_rows.tolist():
            source_id = int(stencil_sources[stencil_row, column])
            coefficient = Fraction.from_float(
                float(stencil_coefficients[stencil_row, column])
            )
            combined[source_id] = combined.get(source_id, Fraction(0)) + coefficient
        ordered = sorted(combined)
        if len(ordered) not in (2, 3):
            raise association_owner.AssociationPropagationError(
                "Surface subdivision vertices require source segment or triangle support.",
                witness,
            )
        rows = _entity_rows(source_mesh, 0, np.asarray(ordered, dtype=np.int64))
        if (
            np.any(rows < 0)
            or sum((combined[identifier] for identifier in ordered), Fraction(0)) != 1
        ):
            raise association_owner.AssociationPropagationError(
                "Surface subdivision stencil does not provide exact source support.",
                witness,
            )
        exact = tuple(combined[identifier] for identifier in ordered)
        local = _support_weights(
            source_coordinates[rows], target_coordinates[int(child_rows[0])]
        )
        if local is None or any(value < 0 for value in local) or local != exact:
            raise association_owner.AssociationPropagationError(
                "Surface subdivision stencil does not reproduce its target point in the source simplex.",
                witness,
            )
        weights[child] = tuple(zip(ordered, exact, strict=True))
    return weights


@checked
def lineage_chart_witness(
    source_witness: SurfaceChartWitness,
    source_mesh: cell_mesh_owner.CellMesh,
    lineage: lineage_owner.MeshLineage,
    target: cell_mesh_owner.CellMesh,
    weights: Mapping[int, SurfaceVertexWeights],
    /,
    *,
    source_geometry_id: str,
    geometry_id: str,
) -> SurfaceChartWitness:
    """Transport exact source charts and all root identities through cell lineage."""
    if (
        source_witness.topology_id != source_mesh.topology_id
        or source_witness.geometry_id != source_geometry_id
        or lineage.source_topology_id != source_mesh.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(
            "Surface subdivision lineage and chart witness must bind their exact source and target meshes."
        )
    source_cell_ids = np.asarray(source_witness.cell_global_ids, dtype=np.int64)
    if set(source_cell_ids.tolist()) != {
        int(value) for block in source_mesh.blocks for value in block.global_ids
    }:
        raise ValueError(
            "Surface source chart witness must cover the source mesh by cell ID."
        )
    witness_row_by_cell = {
        int(identifier): row for row, identifier in enumerate(source_cell_ids.tolist())
    }
    source_vertices_by_cell: dict[int, tuple[int, ...]] = {}
    for block in source_mesh.blocks:
        for identifier, local_vertices in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(),
            np.asarray(block.vertices, dtype=np.int64),
            strict=True,
        ):
            source_vertices_by_cell[identifier] = tuple(
                int(value)
                for value in np.asarray(source_mesh.vertex_global_ids)[local_vertices]
            )
    cell_record = lineage.entity_lineage(2)
    cell_parents: dict[int, set[int]] = {}
    for parent, child in zip(
        np.asarray(cell_record.source_global_ids, dtype=np.int64).tolist(),
        np.asarray(cell_record.target_global_ids, dtype=np.int64).tolist(),
        strict=True,
    ):
        cell_parents.setdefault(child, set()).add(parent)
    target_cell_ids = tuple(
        int(value)
        for block in target.blocks
        for value in np.asarray(block.global_ids, dtype=np.int64)
    )
    if set(cell_parents) != set(target_cell_ids):
        raise ValueError(
            "Surface subdivision lineage must cover every target cell exactly by explicit parent IDs."
        )
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    root_mesh_cell_bindings: list[SurfaceRootMeshCellBinding | None] = []
    source_root_cell_ids: list[int] = []
    source_piece_ids: list[tuple[int, ...]] = []
    patches: list[int] = []
    charts: list[tuple[tuple[Fraction, Fraction], ...]] = []
    entities: list[str] = []
    occurrences: list[tuple[str, ...]] = []
    for block in target.blocks:
        for identifier, local_vertices in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(),
            np.asarray(block.vertices, dtype=np.int64),
            strict=True,
        ):
            witness_ids = np.asarray((identifier,), dtype=np.int64)
            parents = cell_parents.get(identifier, set())
            if len(parents) != 1:
                raise association_owner.AssociationPropagationError(
                    "Surface subdivision cells must retain one explicit parent; got merged or missing lineage.",
                    witness_ids,
                )
            parent = next(iter(parents))
            source_row = witness_row_by_cell.get(parent)
            if source_row is None:
                raise association_owner.AssociationPropagationError(
                    "Surface subdivision cells must name a chart-witnessed source parent.",
                    witness_ids,
                )
            source_vertex_ids = source_vertices_by_cell[parent]
            positions = dict(
                zip(
                    source_vertex_ids, source_witness.exact_chart(source_row), strict=True
                )
            )
            target_corners: list[tuple[Fraction, Fraction]] = []
            for vertex in target_vertex_ids[local_vertices].tolist():
                if vertex in positions:
                    target_corners.append(positions[vertex])
                    continue
                support = weights.get(vertex, ())
                if (
                    not support
                    or any(parent_vertex not in positions for parent_vertex, _ in support)
                    or sum((value for _, value in support), Fraction(0)) != 1
                ):
                    raise association_owner.AssociationPropagationError(
                        "Surface subdivision vertices must retain exact support in their source chart cell.",
                        witness_ids,
                    )
                target_corners.append(
                    (
                        sum(
                            (
                                value * positions[parent_vertex][0]
                                for parent_vertex, value in support
                            ),
                            Fraction(0),
                        ),
                        sum(
                            (
                                value * positions[parent_vertex][1]
                                for parent_vertex, value in support
                            ),
                            Fraction(0),
                        ),
                    )
                )
            root_mesh_cell_bindings.append(
                source_witness.root_mesh_cell_bindings[source_row]
            )
            source_root_cell_ids.append(
                int(source_witness.source_root_cell_ids[source_row])
            )
            source_piece_ids.append(source_witness.source_piece_ids[source_row])
            patches.append(int(np.asarray(source_witness.patches)[source_row]))
            charts.append(tuple(target_corners))
            entities.append(source_witness.geometry_entity_ids[source_row])
            occurrences.append(source_witness.occurrence_paths[source_row])
    return SurfaceChartWitness(
        geometry_id=geometry_id,
        topology_id=target.topology_id,
        domain_id=source_witness.domain_id,
        cell_global_ids=np.asarray(target_cell_ids, dtype=np.int64),
        patches=np.asarray(patches, dtype=np.int64),
        charts=np.asarray(
            [[[float(value) for value in point] for point in chart] for chart in charts],
            dtype=np.float64,
        ),
        source_root_cell_ids=np.asarray(source_root_cell_ids, dtype=np.int64),
        root_mesh_cell_bindings=tuple(root_mesh_cell_bindings),
        source_piece_ids=tuple(source_piece_ids),
        geometry_entity_ids=tuple(entities),
        occurrence_paths=tuple(occurrences),
        chart_rational_tokens=rational_chart_tokens(charts),
    )


@checked
def subdivision_proof(
    support: surface_support_owner.PreparedSurfaceSourceSupport,
    lineages: tuple[lineage_owner.MeshLineage, ...],
    targets: tuple[cell_mesh_owner.CellMesh, ...],
    vertex_stencils: tuple[lineage_owner.VertexInterpolationStencil, ...],
    geometry: cell_geometry_owner.CellGeometrySpec,
    /,
) -> tuple[
    surface_support_owner.SurfaceRestrictionRows,
    tuple[dict[int, SurfaceVertexWeights], ...],
]:
    """Compose serial surface restrictions through canonical chart witnesses."""
    root = support.original
    if not isinstance(root, result_owner.CellMeshingResult):
        raise ValueError(
            "SurfaceSubdivision.support.original must be an accepted serial surface result; got an incompatible root carrier."
        )
    if not lineages or not (len(lineages) == len(targets) == len(vertex_stencils)):
        raise ValueError(
            "SurfaceSubdivision targets, lineages, and vertex stencils must align."
        )
    source_charts = root.surface_source
    if not isinstance(source_charts, surface_support_owner.SurfaceSourceCharts):
        raise ValueError(
            "SurfaceSubdivision.support.original must retain canonical SurfaceSourceCharts."
        )
    source_charts.require_root(root.mesh, root.geometry)
    source_witness = source_charts.chart_witness
    proof = support.prove_native_restrictions(
        root.mesh, root.geometry, chart_witness=source_witness
    )
    previous = root.mesh
    previous_geometry = root.geometry
    epochs: list[dict[int, SurfaceVertexWeights]] = []
    for index, (lineage, target, vertex_stencil) in enumerate(
        zip(lineages, targets, vertex_stencils, strict=True)
    ):
        weights = created_weights(vertex_stencil, previous, target)
        epoch_geometry = (
            geometry
            if index == len(targets) - 1
            else cell_geometry_owner.CellGeometrySpec.affine(target)
        )
        target_witness = lineage_chart_witness(
            source_witness,
            previous,
            lineage,
            target,
            weights,
            source_geometry_id=cell_geometry_id(previous_geometry),
            geometry_id=cell_geometry_id(epoch_geometry),
        )
        proof = support.prove_native_restrictions(
            target, epoch_geometry, chart_witness=target_witness
        )
        epochs.append(weights)
        source_witness = target_witness
        previous = target
        previous_geometry = epoch_geometry
    return proof, tuple(epochs)


__all__ = [
    "SurfaceVertexWeights",
    "created_weights",
    "lineage_chart_witness",
    "subdivision_proof",
]
