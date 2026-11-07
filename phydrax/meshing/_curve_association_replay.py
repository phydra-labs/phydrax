#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..discretization import CellGeometrySpec, CellMesh
from ..geometry._curve_boundary import ParametricCurveBoundarySource
from ..geometry._source_boundary_records import SourceBoundaryChartWitness
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)


type _PointKey = tuple[str, str, int, str, tuple[str, ...], Fraction]


def is_curve_vertex_closure(
    mesh: CellMesh, association: GeometryAssociation, row: int, /
) -> bool:
    """Identify a point restriction, never a second whole-edge assignment."""
    return (
        mesh.topological_dimension == 1
        and association.association_kind is GeometryAssociationKind.CURVE
        and association.target_entity_set_id == mesh.entity_set(0).entity_set_id
        and association.source_roles[row] is GeometrySourceEntityRole.EDGE
        and association.source_parameter_points[row] is not None
    )


def _witness_curve_endpoints(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validated: tuple[ValidatedSourceFidelityWitness, ...],
    /,
) -> set[_PointKey]:
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    cells = {
        int(identifier): tuple(int(vertex_ids[vertex]) for vertex in vertices)
        for block in mesh.blocks
        if block.cell_kind == "interval"
        for identifier, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    endpoints: set[_PointKey] = set()
    for token in validated:
        witness = token.witness
        if (
            not token.binds(mesh, geometry, token.source)
            or not isinstance(token.source, ParametricCurveBoundarySource)
            or not isinstance(witness, SourceBoundaryChartWitness)
            or witness.chart_dimension != 1
        ):
            continue
        for row, identifier in enumerate(witness.cell_global_ids):
            vertices = cells.get(int(identifier))
            if vertices is None:
                continue
            for target, source in zip(
                witness.target_reference_rational_tokens[row],
                witness.source_parameter_rational_tokens[row],
                strict=True,
            ):
                reference = Fraction(*target[0])
                if reference not in (0, 1):
                    continue
                endpoints.add(
                    (
                        witness.source_id,
                        witness.source_revision,
                        vertices[int(reference)],
                        witness.source_entity_ids[row],
                        witness.occurrence_paths[row],
                        Fraction(*source[0]),
                    )
                )
    return endpoints


def curve_vertex_closure_issues(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    associations: tuple[GeometryAssociation, ...],
    validated: tuple[ValidatedSourceFidelityWitness, ...],
    /,
) -> tuple[str, ...]:
    """Prove every incident source restriction when one vertex joins curves."""
    if mesh.topological_dimension != 1:
        return ()
    groups: dict[tuple[str, str, int], set[_PointKey]] = {}
    for association in associations:
        resolved = np.asarray(association.resolved, dtype=np.bool_)
        for row, identifier in enumerate(np.asarray(association.target_global_ids)):
            if not resolved[row] or not is_curve_vertex_closure(mesh, association, row):
                continue
            parameter = association.source_parameter_points[row]
            if parameter is None:
                raise ValueError("A curve vertex closure requires an exact parameter.")
            key = (association.source_id, association.source_revision, int(identifier))
            groups.setdefault(key, set()).add(
                (
                    *key,
                    association.source_entity_ids[row],
                    association.source_occurrence_paths[row],
                    parameter,
                )
            )
    shared = set().union(
        *(
            points
            for points in groups.values()
            if len({(point[3], point[4]) for point in points}) > 1
        )
    )
    if not shared:
        return ()
    proven = _witness_curve_endpoints(mesh, geometry, validated)
    return () if shared <= proven else ("association_curve_vertex_closure",)
