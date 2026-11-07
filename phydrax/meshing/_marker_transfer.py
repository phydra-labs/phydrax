#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transfer declared categorical cell markers through exact cell ancestry.

Categorical cell markers are integer MARKER or MATERIAL attributes. Layer
column markers identify the authored source column.  They are not the
refined column grouping identifiers carried by MixedLayerColumns.
"""

from __future__ import annotations

import numpy as np

from ..discretization import CellMesh
from ._association_routes import require_transfer_lineage
from ._layer_core_topology import mesh_entity_scope
from ._lineage import EntityLineageKind, MeshLineage
from ._organization import MeshAttribute, MeshAttributeRole
from ._scope import resolve_mesh_scope


def validate_cell_markers(
    mesh: CellMesh, attributes: tuple[MeshAttribute, ...], /
) -> None:
    """Require explicit, complete, dimensionless integer categorical cell semantics."""
    dimension = mesh.topological_dimension
    entities = mesh.entity_set(dimension)
    active = np.asarray(entities.active_mask, dtype=np.bool_)
    cell_ids = np.asarray(entities.entity_ids, dtype=np.int64)[active]
    names: set[str] = set()
    for attribute in attributes:
        if (
            attribute.role not in (MeshAttributeRole.MARKER, MeshAttributeRole.MATERIAL)
            or attribute.scope.entity_dimension != dimension
            or attribute.unit is not None
            or attribute.component_shape
            or np.asarray(attribute.values).dtype.kind not in "iu"
        ):
            raise ValueError("Native adaptation requires complete integer cell MARKER or MATERIAL attributes with no units.")
        if attribute.name in names:
            raise ValueError("Native adaptation requires uniquely named cell markers.")
        names.add(attribute.name)
        resolve_mesh_scope(mesh, attribute.scope)
        if not np.array_equal(np.asarray(attribute.scope.entity_ids), np.sort(cell_ids)):
            raise ValueError("Native adaptation requires each cell marker on every active source cell.")
    by_name = {attribute.name: attribute for attribute in attributes}
    if "layer_index" in by_name or "layer_column" in by_name:
        if not {"layer_index", "layer_column"} <= by_name.keys():
            raise ValueError("Layer index and source-column markers must be declared together.")
        intervals = np.asarray(by_name["layer_index"].values)
        columns = np.asarray(by_name["layer_column"].values)
        if (
            np.any(intervals < -1)
            or np.any(columns < -1)
            or not np.array_equal(intervals == -1, columns == -1)
        ):
            raise ValueError("Layer index and source-column markers disagree on core cells.")


def transfer_cell_markers(
    source: CellMesh,
    attributes: tuple[MeshAttribute, ...],
    lineage: MeshLineage,
    target: CellMesh,
    /,
) -> tuple[MeshAttribute, ...]:
    """Inherit each categorical value only from compatible exact cell parents."""
    if not attributes:
        return ()
    validate_cell_markers(source, attributes)
    dimension = source.topological_dimension
    if target.topological_dimension != dimension:
        raise ValueError("Cell-marker transfer requires equal mesh dimensions.")
    require_transfer_lineage(source, lineage, target, dimensions=(dimension,))
    record = lineage.entity_lineage(dimension)
    old_ids = np.asarray(record.source_global_ids, dtype=np.int64)
    new_ids = np.asarray(record.target_global_ids, dtype=np.int64)
    kinds = np.asarray(record.relation_kinds, dtype=np.int32)
    allowed = (int(EntityLineageKind.PRESERVED), int(EntityLineageKind.REFINED_FROM), int(EntityLineageKind.COARSENED_INTO))
    if np.any(~np.isin(kinds, allowed)):
        raise ValueError("Cell-marker transfer requires preserved, refined, or coarsened ancestry.")
    source_entities = source.entity_set(dimension)
    source_ids = np.asarray(source_entities.entity_ids, dtype=np.int64)[np.asarray(source_entities.active_mask, dtype=np.bool_)]
    target_entities = target.entity_set(dimension)
    target_ids = np.sort(np.asarray(target_entities.entity_ids, dtype=np.int64)[np.asarray(target_entities.active_mask, dtype=np.bool_)])
    if not np.all(np.isin(old_ids, source_ids)) or not np.all(np.isin(new_ids, target_ids)):
        raise ValueError("Cell-marker lineage references inactive or unknown cells.")
    if np.unique(target_ids).size != target_ids.size or not np.array_equal(np.unique(new_ids), target_ids):
        raise ValueError("Cell-marker lineage must cover every active target cell.")
    parents: dict[int, list[int]] = {int(identifier): [] for identifier in target_ids}
    for old, new in zip(old_ids.tolist(), new_ids.tolist(), strict=True):
        parents[new].append(old)
    output = []
    for attribute in attributes:
        values = np.asarray(attribute.values)
        lookup = dict(zip(np.asarray(attribute.scope.entity_ids, dtype=np.int64).tolist(), values.tolist(), strict=True))
        inherited = []
        for identifier in target_ids.tolist():
            candidates = {lookup[parent] for parent in parents[identifier]}
            if len(candidates) != 1:
                raise ValueError(f"Cell marker {attribute.name!r} conflicts across exact ancestry.")
            inherited.append(next(iter(candidates)))
        output.append(MeshAttribute(attribute.name, attribute.role, mesh_entity_scope(target, dimension, target_ids), np.asarray(inherited, dtype=values.dtype), unit=attribute.unit))
    return tuple(output)
