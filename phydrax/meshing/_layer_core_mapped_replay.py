#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reconstruct every interior mapped-root stratum from exact volume charts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_transition_types import (
    _require_exact_reference_group,
)
from ..discretization._nested_reference import nested_cell_reference_maps
from ..discretization._nested_reference_types import exact_reference_cell_vertices
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from ._layer_source_association_replay import _maps, _witness
from ._volume_entity_containment import reference_entity_containment


if TYPE_CHECKING:
    from ._contracts import VolumeMeshingSpec
    from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
    from .providers._native_sources import NativeLayerCoreSource


def replay_mapped_root_associations(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    source: NativeLayerCoreSource | None = None,
    specification: VolumeMeshingSpec | None = None,
    root: LayerCoreRootGeometry | None = None,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> tuple[GeometryAssociation, ...]:
    """Bind cells and uniquely root-interior vertices, edges and faces.

    Every retained current-to-root chart is checked against the full coordinate
    map, and exact reference containment—not source output labels—decides whether
    a lower-dimensional entity belongs to a unique root interior.
    """
    if (
        mesh.topological_dimension != 3
        or domain.reference_mesh.topological_dimension != 3
    ):
        raise ValueError("Mapped volume source replay requires three-dimensional cells.")
    root_proved = False
    if computed_root is not None:
        from ._layer_core_root_geometry import (
            ComputedLayerCoreRoot,
            LayerCoreRootGeometry,
        )

        if (
            type(computed_root) is not ComputedLayerCoreRoot
            or not isinstance(root, LayerCoreRootGeometry)
            or source is None
            or specification is None
            or source.mapped_domain is not domain
        ):
            raise ValueError("Mapped replay requires its actual computed root operands.")
        computed_root.require(source, specification, root.mesh, root.geometry, root)
        root_proved = (
            root.mesh is mesh
            and root.geometry is geometry
            and computed_root.mapped_chart_bank_equal
        )
    root_kinds = {
        int(identifier): block.cell_kind
        for block in domain.reference_mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    declared_roots = set(domain.root_cell_ids)
    restriction = geometry.restriction_source
    direct_root = restriction is not None and (
        restriction.source_topology_id == domain.reference_mesh.topology_id
        and restriction.source_geometry_id == domain.reference_geometry_id
    )
    groups = []
    if direct_root:
        assert restriction is not None
        block_maps = restriction.block_parent_reference_maps
        if set(block_maps) != {block.name for block in mesh.blocks}:
            raise ValueError("Mapped root charts must cover every target block.")
        if not root_proved:
            root_maps = _maps(domain.reference_mesh, domain.source_geometry)
            target_maps = _maps(mesh, geometry)
        for block in mesh.blocks:
            bank = block_maps[block.name]
            if bank is None:
                raise ValueError("Mapped root charts cannot omit an endpoint block.")
            rows = nested_cell_reference_maps(bank)
            if len(rows) != block.cell_count:
                raise ValueError("Mapped root chart rows do not cover an endpoint block.")
            for row, chart in enumerate(rows):
                if (
                    chart.target_cell_id != row
                    or chart.target_cell_kind != block.cell_kind
                    or chart.arguments is None
                    or chart.source_cell_id not in declared_roots
                    or root_kinds.get(chart.source_cell_id) != chart.source_cell_kind
                ):
                    raise ValueError(
                        "Mapped root chart changes its source or target cell identity."
                    )
                group = _witness(
                    int(block.global_ids[row]),
                    chart.source_cell_id,
                    block.cell_kind,
                    chart.source_cell_kind,
                    chart.source_reference_vertices,
                    chart.arguments,
                )
                if not root_proved:
                    _require_exact_reference_group(
                        root_maps,
                        target_maps,
                        group,
                        exact_reference_cell_vertices(group),
                    )
                groups.append(group)
    else:
        if source is None or root is None or source.mapped_domain is not domain:
            raise ValueError(
                "Mapped descendant replay requires its actual retained source and root."
            )
        from ._layer_core_mapped_segment_overlay import mapped_local_reference_maps
        from ._layer_source_association_replay import _chart_corners

        local_maps = mapped_local_reference_maps(
            source, mesh, geometry, root, computed_root=computed_root
        )
        target_ids = {
            int(identifier)
            for block in mesh.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64)
        }
        if set(local_maps) != target_ids:
            raise ValueError(
                "Mapped descendant charts do not cover the actual target cells."
            )
        for block in mesh.blocks:
            for identifier in np.asarray(block.global_ids, dtype=np.int64):
                cell = int(identifier)
                parent, arguments = local_maps[cell]
                if parent not in declared_roots or parent not in root_kinds:
                    raise ValueError(
                        "Mapped descendant chart changed its original root identity."
                    )
                groups.append(
                    _witness(
                        cell,
                        parent,
                        block.cell_kind,
                        root_kinds[parent],
                        _chart_corners(block.cell_kind, arguments),
                        arguments,
                    )
                )
    containment = reference_entity_containment(mesh, domain.reference_mesh, tuple(groups))
    owners: dict[tuple[int, int], set[int]] = {}
    boundary: set[tuple[int, int]] = set()
    for row in containment:
        key = (row.fine_dimension, row.fine_id)
        if row.coarse_dimension == 3:
            owners.setdefault(key, set()).add(row.coarse_cell_id)
        else:
            boundary.add(key)
    paths = dict(
        zip(domain.root_cell_ids, domain.root_cell_occurrence_paths, strict=True)
    )
    output = []
    for dimension in range(4):
        entity_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        values: list[tuple[int, int]] = []
        for identifier in entity_ids.tolist():
            key = (dimension, int(identifier))
            roots = owners.get(key, set())
            if dimension == 3 and len(roots) != 1:
                raise ValueError(
                    "Every mapped volume cell must retain one exact source root."
                )
            if len(roots) == 1 and (dimension == 3 or key not in boundary):
                values.append((int(identifier), next(iter(roots))))
        if not values:
            continue
        values.sort()
        targets = np.asarray([value[0] for value in values], dtype=np.int64)
        parents = np.asarray([value[1] for value in values], dtype=np.int64)
        output.append(
            GeometryAssociation(
                GeometryAssociationKind.MAPPED_REFERENCE,
                domain.source_id,
                domain.source_revision,
                mesh.entity_set(dimension).entity_set_id,
                targets,
                tuple(domain.image_entity_id(3, int(parent)) for parent in parents),
                np.zeros(targets.shape, dtype=np.float64),
                exact=True,
                source_dimensions=np.full(targets.shape, 3, dtype=np.int8),
                source_indices=parents,
                source_roles=(GeometrySourceEntityRole.REGION,) * targets.size,
                source_occurrence_paths=tuple(paths[int(parent)] for parent in parents),
                parent_dimensions=np.full(targets.shape, 3, dtype=np.int8),
                parent_ids=parents,
            )
        )
    return tuple(output)


__all__ = ["replay_mapped_root_associations"]
