#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Topology and source-namespace operations shared by association transfers."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np
from jax.typing import ArrayLike

from ..discretization import CellMesh
from ..typing import checked
from ._association import (
    _boundary_mask,
    _entity_rows,
    _incidence_pairs,
    _ordered_vertices,
    _target_dimension,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
)
from ._lineage import EntityLineageKind, MeshLineage


class AssociationSource(Protocol):
    mesh: CellMesh
    associations: tuple[GeometryAssociation, ...]


class AssociationLineageRows(NamedTuple):
    target_ids: np.ndarray
    parent_ids: np.ndarray
    association_rows: np.ndarray


def target_dimension(mesh: CellMesh, association: GeometryAssociation, /) -> int:
    """Resolve a source association's exact target mesh dimension."""
    return _target_dimension(mesh, association)


def entity_rows(mesh: CellMesh, dimension: int, identifiers: ArrayLike, /) -> np.ndarray:
    """Resolve global entity IDs to mesh rows, returning ``-1`` for absent IDs."""
    return _entity_rows(mesh, dimension, np.asarray(identifiers, dtype=np.int64))


def incidence_pairs(mesh: CellMesh, low: int, high: int, /) -> np.ndarray:
    """Return mesh incidence pairs with the lower entity in the upper closure."""
    return _incidence_pairs(mesh, low, high)


def boundary_mask(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Return entities in the closure of the mesh boundary."""
    return _boundary_mask(mesh, dimension)


def ordered_vertices(mesh: CellMesh, dimension: int, /) -> np.ndarray | None:
    """Return the canonical simplex vertex order for an entity dimension."""
    return _ordered_vertices(mesh, dimension)


def association_lookup(
    source: AssociationSource,
    /,
    *,
    kind: GeometryAssociationKind | None = None,
    namespace: tuple[str, str] | None = None,
) -> dict[int, GeometryAssociation]:
    """Index namespace-matching tables by target dimension."""
    output: dict[int, GeometryAssociation] = {}
    for association in source.associations:
        if kind is not None and association.association_kind is not kind:
            continue
        if (
            namespace is not None
            and (association.source_id, association.source_revision) != namespace
        ):
            continue
        dimension = target_dimension(source.mesh, association)
        if dimension in output:
            raise ValueError(
                "A source carries one association table per target dimension and namespace."
            )
        output[dimension] = association
    return output


def associations_for_namespace(
    source: AssociationSource,
    namespace: tuple[str, str],
    /,
    *,
    kinds: tuple[GeometryAssociationKind, ...],
) -> tuple[GeometryAssociation, ...]:
    """Return source tables for one exact source namespace and permitted kinds."""
    values = tuple(
        association
        for association in source.associations
        if (association.source_id, association.source_revision) == namespace
        and association.association_kind in kinds
    )
    if not values:
        raise ValueError(
            "Association transfer requires rows for its exact source namespace."
        )
    dimensions = tuple(target_dimension(source.mesh, value) for value in values)
    if len(set(dimensions)) != len(dimensions):
        raise ValueError(
            "A source carries one association table per target dimension and namespace."
        )
    for association, dimension in zip(values, dimensions, strict=True):
        association.validate_target(source.mesh.entity_set(dimension))
        if not association.complete:
            raise ValueError(
                "Source association transfer requires resolved, unambiguous rows."
            )
    return tuple(sorted(values, key=lambda value: target_dimension(source.mesh, value)))


def require_transfer_lineage(
    source: CellMesh,
    lineage: MeshLineage,
    target: CellMesh,
    /,
    *,
    dimensions: tuple[int, ...],
) -> None:
    """Validate topology identity and reject unknown relation rows on used dimensions."""
    if (
        lineage.source_topology_id != source.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(
            "Association transfer lineage does not bind source and target topology."
        )
    for dimension in dimensions:
        if dimension > target.topological_dimension:
            continue
        record = lineage.entity_lineage(dimension)
        if (
            record.source_entity_set_id != source.entity_set(dimension).entity_set_id
            or record.target_entity_set_id != target.entity_set(dimension).entity_set_id
        ):
            raise ValueError(
                "Association transfer lineage does not bind source and target entity sets."
            )
        kinds = np.asarray(record.relation_kinds, dtype=np.int32)
        if np.any(kinds == int(EntityLineageKind.UNKNOWN)):
            raise ValueError(
                "Association transfer lineage contains unknown source relations."
            )


def _row_identity(association: GeometryAssociation, row: int, /) -> tuple[object, ...]:
    """Stable source class payload used to verify merged lineage parents."""
    return (
        association.association_kind,
        association.source_id,
        association.source_revision,
        association.source_entity_ids[row],
        association.source_roles[row],
        association.source_occurrence_paths[row],
        association.source_region_sides[row],
        int(np.asarray(association.source_dimensions)[row]),
        int(np.asarray(association.source_indices)[row]),
        association.source_parameter_intervals[row],
        association.source_parameter_points[row],
    )


@checked
def lineage_association_rows(
    association: GeometryAssociation,
    source: CellMesh,
    lineage: MeshLineage,
    target: CellMesh,
    dimension: int,
    /,
    *,
    merge_coarsened_intervals: bool = False,
) -> AssociationLineageRows:
    """Map lineage-supported target entities to one agreeing source association row.

    ``merge_coarsened_intervals`` admits target entities whose parents are all
    coarsened into them and agree up to their parameter intervals and points;
    the caller owns the merged parameterization.
    """
    if not isinstance(source, CellMesh) or not isinstance(target, CellMesh):
        raise TypeError("source and target must be CellMesh values.")
    if association.target_entity_set_id != source.entity_set(dimension).entity_set_id:
        raise ValueError(
            "Association lineage table targets the wrong source entity dimension."
        )
    require_transfer_lineage(source, lineage, target, dimensions=(dimension,))
    record = lineage.entity_lineage(dimension)
    source_ids = np.asarray(record.source_global_ids, dtype=np.int64)
    target_ids = np.asarray(record.target_global_ids, dtype=np.int64)
    relation_kinds = np.asarray(record.relation_kinds, dtype=np.int32)
    source_table_ids = np.asarray(association.target_global_ids, dtype=np.int64)
    source_row_by_id = {
        int(identifier): row for row, identifier in enumerate(source_table_ids.tolist())
    }
    target_entity_ids = np.asarray(
        target.entity_set(dimension).entity_ids, dtype=np.int64
    )
    output_targets: list[int] = []
    output_parents: list[int] = []
    output_rows: list[int] = []
    for target_row, target_id in enumerate(target_entity_ids.tolist()):
        relation_rows = np.flatnonzero(target_ids == target_id)
        parents = source_ids[relation_rows]
        parent_rows = [
            source_row_by_id[int(identifier)]
            for identifier in parents
            if int(identifier) in source_row_by_id
        ]
        if not parent_rows and target_id in source_row_by_id:
            parent_rows = [source_row_by_id[target_id]]
            parents = np.asarray((target_id,), dtype=np.int64)
        if not parent_rows:
            continue
        merged = (
            merge_coarsened_intervals
            and len(parent_rows) > 1
            and bool(
                np.all(
                    relation_kinds[relation_rows] == int(EntityLineageKind.COARSENED_INTO)
                )
            )
        )
        # Interval and parameter point are the last two identity entries.
        width = -2 if merged else None
        identities = {_row_identity(association, row)[:width] for row in parent_rows}
        if len(identities) != 1:
            raise ValueError(
                f"Association transfer cannot merge distinct source strata onto target entity {target_id}."
            )
        first = parent_rows[0]
        matching_parent_ids = [
            int(identifier)
            for identifier, parent_row in zip(
                parents.tolist(),
                [source_row_by_id.get(int(value), -1) for value in parents.tolist()],
                strict=True,
            )
            if parent_row >= 0
            and _row_identity(association, parent_row)[:width]
            == _row_identity(association, first)[:width]
        ]
        output_targets.append(int(target_id))
        output_parents.append(
            min(matching_parent_ids)
            if matching_parent_ids
            else int(source_table_ids[first])
        )
        output_rows.append(first)
    return AssociationLineageRows(
        np.asarray(output_targets, dtype=np.int64),
        np.asarray(output_parents, dtype=np.int64),
        np.asarray(output_rows, dtype=np.int64),
    )


def inherit_association_table(
    association: GeometryAssociation,
    source: CellMesh,
    lineage: MeshLineage,
    target: CellMesh,
    dimension: int,
    /,
    *,
    residuals: ArrayLike | None = None,
    exact: bool | None = None,
    provenance: GeometryAssociationProvenance = GeometryAssociationProvenance.LINEAGE,
    parent_association_id: str | None = None,
) -> GeometryAssociation | None:
    """Carry exact source class rows through same-stratum entity lineage."""
    rows = lineage_association_rows(association, source, lineage, target, dimension)
    if rows.target_ids.size == 0:
        return None
    parent_rows = rows.association_rows
    residual = (
        np.asarray(association.residuals, dtype=np.float64)[parent_rows]
        if residuals is None
        else np.asarray(residuals, dtype=np.float64)
    )
    if residual.shape != rows.target_ids.shape:
        raise ValueError(
            "Association transfer residuals must align with inherited target rows."
        )
    inherited_exact = (
        association.exact
        and np.array_equal(rows.target_ids, rows.parent_ids)
        and bool(np.all(residual == 0.0))
        if exact is None
        else exact
    )
    source_dimensions = np.asarray(association.source_dimensions, dtype=np.int8)[
        parent_rows
    ]
    source_indices = np.asarray(association.source_indices, dtype=np.int64)[parent_rows]
    if association.association_kind is not GeometryAssociationKind.BREP and not np.any(
        source_dimensions >= 0
    ):
        source_dimensions = None
        source_indices = None
    return GeometryAssociation(
        association.association_kind,
        association.source_id,
        association.source_revision,
        target.entity_set(dimension).entity_set_id,
        rows.target_ids,
        tuple(association.source_entity_ids[int(row)] for row in parent_rows.tolist()),
        residual,
        resolved=np.asarray(association.resolved, dtype=np.bool_)[parent_rows],
        ambiguous=np.asarray(association.ambiguous, dtype=np.bool_)[parent_rows],
        exact=inherited_exact,
        source_dimensions=source_dimensions,
        source_indices=source_indices,
        source_roles=tuple(
            association.source_roles[int(row)] for row in parent_rows.tolist()
        ),
        source_occurrence_paths=tuple(
            association.source_occurrence_paths[int(row)] for row in parent_rows.tolist()
        ),
        source_region_sides=tuple(
            association.source_region_sides[int(row)] for row in parent_rows.tolist()
        ),
        source_parameter_intervals=tuple(
            association.source_parameter_intervals[int(row)]
            for row in parent_rows.tolist()
        ),
        source_parameter_points=tuple(
            association.source_parameter_points[int(row)] for row in parent_rows.tolist()
        ),
        parameters=np.asarray(association.parameters, dtype=np.float64)[parent_rows],
        orientations=np.asarray(association.orientations, dtype=np.int8)[parent_rows],
        parent_dimensions=np.full(rows.target_ids.shape, dimension, dtype=np.int8),
        parent_ids=rows.parent_ids,
        provenance=provenance,
        parent_association_id=association.association_id
        if parent_association_id is None
        else parent_association_id,
    )


def association_classes(
    source: AssociationSource,
    associations: tuple[GeometryAssociation, ...],
    /,
    *,
    topological_dimension: int | None = None,
    default_dimension: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """Lower source tables to dimension/index/resolution/code arrays by mesh row."""
    mesh = source.mesh
    top = (
        mesh.topological_dimension
        if topological_dimension is None
        else topological_dimension
    )
    result = []
    for dimension in range(top + 1):
        count = mesh.entity_set(dimension).count
        dimensions = np.full(count, -1, dtype=np.int64)
        indices = np.full(count, -1, dtype=np.int64)
        resolved = np.zeros(count, dtype=np.bool_)
        keys: list[tuple[object, ...] | None] = [None] * count
        for association in associations:
            if target_dimension(mesh, association) != dimension:
                continue
            rows = entity_rows(
                mesh, dimension, np.asarray(association.target_global_ids, dtype=np.int64)
            )
            valid = rows >= 0
            source_dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
            source_indices = np.asarray(association.source_indices, dtype=np.int64)
            dimensions[rows[valid]] = source_dimensions[valid]
            indices[rows[valid]] = source_indices[valid]
            resolved[rows[valid]] = np.asarray(association.resolved, dtype=np.bool_)[
                valid
            ]
            for association_row, mesh_row in enumerate(rows.tolist()):
                if mesh_row >= 0:
                    keys[mesh_row] = _row_identity(association, association_row)
        if default_dimension is not None:
            missing = ~resolved
            dimensions[missing] = default_dimension
            indices[missing] = 0
            resolved[missing] = True
            for row in np.flatnonzero(missing).tolist():
                keys[row] = ("default", default_dimension)
        distinct = tuple(sorted({key for key in keys if key is not None}, key=repr))
        code_by_key = {key: code for code, key in enumerate(distinct)}
        codes = np.asarray(
            [code_by_key[key] if key is not None else -1 for key in keys], dtype=np.int64
        )
        result.append((dimensions, indices, resolved, codes))
    return tuple(result)


__all__ = [
    "AssociationLineageRows",
    "AssociationSource",
    "association_classes",
    "association_lookup",
    "associations_for_namespace",
    "boundary_mask",
    "entity_rows",
    "incidence_pairs",
    "inherit_association_table",
    "lineage_association_rows",
    "ordered_vertices",
    "require_transfer_lineage",
    "target_dimension",
]
