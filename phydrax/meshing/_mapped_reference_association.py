#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Classify and transfer exact mapped-reference root identities."""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..typing import checked
from ._association import (
    AssociationEntityClasses,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from ._association_routes import (
    associations_for_namespace,
    entity_rows,
    inherit_association_table,
    target_dimension,
)
from ._lineage import MeshLineage, VertexInterpolationStencil
from ._result import CellMeshingResult


def _restriction_parent_cells(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> np.ndarray:
    root_dimension = domain.reference_mesh.topological_dimension
    if mesh.topology_id == domain.reference_mesh.topology_id and cell_geometry_id(
        geometry
    ) == cell_geometry_id(domain.source_geometry):
        root_ids = np.asarray(mesh.entity_set(root_dimension).entity_ids, dtype=np.int64)
        if not np.all(
            np.isin(root_ids, np.asarray(domain.root_cell_ids, dtype=np.int64))
        ):
            raise ValueError(
                "Mapped-reference root cells must belong to the declared chart domain."
            )
        return root_ids
    restriction = geometry.restriction_source
    if restriction is None:
        raise ValueError(
            "Mapped-reference association requires exact source-cell restriction ancestry."
        )
    if (
        restriction.source_geometry_id != cell_geometry_id(domain.source_geometry)
        or restriction.source_topology_id != domain.reference_mesh.topology_id
    ):
        raise ValueError(
            "Mapped-reference restriction ancestry binds a different root chart."
        )
    parent_by_target: dict[int, int] = {}
    block_parent = dict(
        zip(restriction.block_names, restriction.parent_cell_ids, strict=True)
    )
    if set(block_parent) != {block.name for block in mesh.blocks}:
        raise ValueError(
            "Mapped-reference restriction blocks do not match the target mesh."
        )
    for block in mesh.blocks:
        parent_ids = np.asarray(block_parent[block.name], dtype=np.int64)
        target_ids = np.asarray(block.global_ids, dtype=np.int64)
        if parent_ids.shape != target_ids.shape:
            raise ValueError(
                "Mapped-reference parent cell IDs must align with target block cells."
            )
        parent_by_target.update(
            (int(target_id), int(parent_id))
            for target_id, parent_id in zip(
                target_ids.tolist(), parent_ids.tolist(), strict=True
            )
        )
    identifiers = np.asarray(
        mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
    )
    if set(parent_by_target) != set(identifiers.tolist()):
        raise ValueError(
            "Mapped-reference restriction ancestry must cover every target cell exactly once."
        )
    parents = np.asarray(
        [parent_by_target[int(identifier)] for identifier in identifiers], dtype=np.int64
    )
    if not np.all(np.isin(parents, np.asarray(domain.root_cell_ids, dtype=np.int64))):
        raise ValueError(
            "Mapped-reference parent cells must belong to the declared root mesh."
        )
    return parents


@checked
def mapped_reference_associations(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> tuple[GeometryAssociation, ...]:
    """Bind every exactly restricted target cell to its unique mapped root region."""
    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError("mesh and geometry must be CellMesh and CellGeometrySpec.")
    if mesh.topological_dimension != domain.reference_mesh.topological_dimension:
        raise ValueError("Mapped-reference target and root dimensions must match.")
    geometry.resolve(mesh)
    root_cells = _restriction_parent_cells(domain, mesh, geometry)
    root_dimension = domain.reference_mesh.topological_dimension
    entity_set = mesh.entity_set(root_dimension)
    identifiers = np.asarray(entity_set.entity_ids, dtype=np.int64)
    root_paths = dict(
        zip(domain.root_cell_ids, domain.root_cell_occurrence_paths, strict=True)
    )
    role = (
        GeometrySourceEntityRole.REGION
        if root_dimension == 3
        else GeometrySourceEntityRole.FACET
    )
    return (
        GeometryAssociation(
            GeometryAssociationKind.MAPPED_REFERENCE,
            domain.source_id,
            domain.source_revision,
            entity_set.entity_set_id,
            identifiers,
            tuple(
                domain.image_entity_id(root_dimension, int(root_id))
                for root_id in root_cells.tolist()
            ),
            np.zeros(identifiers.shape, dtype=np.float64),
            exact=True,
            source_dimensions=np.full(identifiers.shape, root_dimension, dtype=np.int8),
            source_indices=root_cells,
            source_roles=(role,) * identifiers.size,
            source_occurrence_paths=tuple(
                root_paths[int(root_id)] for root_id in root_cells.tolist()
            ),
            parent_dimensions=np.full(identifiers.shape, root_dimension, dtype=np.int8),
            parent_ids=root_cells,
        ),
    )


@final
class MappedReferenceAssociationTransfer(StrictModule, NonTrainableState):
    """Preserve exact mapped-root identities through explicit cell lineage."""

    domain: MappedReferenceDomain
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(self, domain: MappedReferenceDomain, /) -> None:
        self.domain = domain
        self.transfer_id = canonical_fingerprint(
            {"kind": "mapped-reference-association-transfer", "domain": domain.domain_id}
        )

    @property
    def source_namespaces(self) -> tuple[tuple[str, str], ...]:
        return ((self.domain.source_id, self.domain.source_revision),)

    def validate_source(self, source: CellMeshingResult, /) -> None:
        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        tables = associations_for_namespace(
            source,
            self.source_namespaces[0],
            kinds=(GeometryAssociationKind.MAPPED_REFERENCE,),
        )
        root_dimension = self.domain.reference_mesh.topological_dimension
        declared_roots = set(self.domain.root_cell_ids)
        for association in tables:
            dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
            indices = np.asarray(association.source_indices, dtype=np.int64)
            if np.any(dimensions < 0) or np.any(indices < 0):
                raise ValueError(
                    "Mapped-reference source rows must name explicit root strata."
                )
            for row, (dimension, index) in enumerate(
                zip(dimensions.tolist(), indices.tolist(), strict=True)
            ):
                if dimension == root_dimension and index not in declared_roots:
                    raise ValueError(
                        "Mapped-reference source cell IDs must name declared root cells."
                    )
                if association.source_entity_ids[row] != self.domain.image_entity_id(
                    dimension, index
                ):
                    raise ValueError(
                        "Mapped-reference source entity IDs must match exact root-image identity."
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
                kinds=(GeometryAssociationKind.MAPPED_REFERENCE,),
            )
        }
        result = []
        for dimension in range(mesh.topological_dimension + 1):
            count = mesh.entity_set(dimension).count
            dimensions = np.full((count,), -1, dtype=np.int64)
            indices = np.full((count,), -1, dtype=np.int64)
            resolved = np.zeros((count,), dtype=np.bool_)
            codes = np.full((count,), -1, dtype=np.int64)
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
                keys = tuple(
                    (
                        association.source_entity_ids[row],
                        association.source_occurrence_paths[row],
                    )
                    for row in np.flatnonzero(valid).tolist()
                )
                unique = {key: index for index, key in enumerate(sorted(set(keys)))}
                for association_row, mesh_row in enumerate(rows.tolist()):
                    if mesh_row >= 0:
                        codes[mesh_row] = unique[
                            (
                                association.source_entity_ids[association_row],
                                association.source_occurrence_paths[association_row],
                            )
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
        if vertex_stencil is not None and not isinstance(
            vertex_stencil, VertexInterpolationStencil
        ):
            raise TypeError("vertex_stencil must be VertexInterpolationStencil or None.")
        output = []
        for association in associations_for_namespace(
            source,
            self.source_namespaces[0],
            kinds=(GeometryAssociationKind.MAPPED_REFERENCE,),
        ):
            dimension = target_dimension(source.mesh, association)
            inherited = inherit_association_table(
                association,
                source.mesh,
                lineage,
                target,
                dimension,
                exact=False,
                provenance=GeometryAssociationProvenance.LINEAGE,
            )
            if inherited is not None:
                output.append(inherited)
        return tuple(output)


__all__ = ["MappedReferenceAssociationTransfer", "mapped_reference_associations"]
