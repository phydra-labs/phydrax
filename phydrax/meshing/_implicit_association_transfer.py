#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Implicit source identity and boundary-model lineage transport."""

from __future__ import annotations

from typing import final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.meshing._association_state import (
    entity_rows,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
)

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_transfer import is_affine_cell_geometry
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..geometry._source_fidelity import ImplicitProjectionBoundarySource
from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ..geometry.surface import (
    SurfaceInterface,
    SurfaceMetadata,
    SurfaceModel,
    SurfaceSelection,
)
from ..typing import checked
from ._association_routes import (
    boundary_mask,
    ordered_vertices,
    require_transfer_lineage,
)
from ._lineage import inherit_scope
from ._lineage_records import EntityLineageKind, MeshLineage
from ._result import CellMeshingResult
from ._scope import MeshingEntityKind, MeshingScope


class ImplicitEntityClasses(NamedTuple):
    dimensions: np.ndarray
    indices: np.ndarray
    resolved: np.ndarray
    codes: np.ndarray


def _inherited_surface_ids(
    source: CellMesh,
    target: CellMesh,
    lineage: MeshLineage,
    identifiers: ArrayLike,
    name: str,
) -> Array:
    scope = MeshingScope(
        source.mesh_id,
        source.numeric_version,
        MeshingEntityKind.MESH,
        2,
        source.entity_set(2).entity_set_id,
        identifiers,
    )
    return inherit_scope(scope, lineage, target, name).global_entity_ids


def _implicit_face_ids(mesh: CellMesh, /) -> np.ndarray:
    identifiers = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    return (
        identifiers
        if mesh.topological_dimension == 2
        else identifiers[boundary_mask(mesh, 2)]
    )


@final
class ImplicitAssociationTransfer(StrictModule, NonTrainableState):
    """Carry certified analytic implicit boundaries through native carrier edits.

    The scalar source, state, and revision retain their declared identities.
    The successor source-fidelity certificate admits the target. Surface metadata
    and volume boundary associations follow explicit face lineage.
    """

    profile: AnalyticImplicitProfile
    maximum_queries: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, profile: AnalyticImplicitProfile, /, *, maximum_queries: int
    ) -> None:
        if isinstance(maximum_queries, bool) or not isinstance(maximum_queries, int):
            raise TypeError(
                f"ImplicitAssociationTransfer.maximum_queries must be an integer; got {maximum_queries!r}"
            )
        if maximum_queries < 1:
            raise ValueError(
                f"ImplicitAssociationTransfer.maximum_queries must be positive; got {maximum_queries}"
            )
        profile.validate_source_integrity()
        self.profile = profile
        self.maximum_queries = maximum_queries
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "implicit-association-transfer",
                "profile": profile.profile_id,
                "maximum_queries": maximum_queries,
            }
        )

    @checked
    def source_associations(
        self, source: CellMeshingResult, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        self.profile.require_bound(
            self.profile.geometry, source.coordinate_contract, budget=budget
        )
        mesh = source.mesh
        top = mesh.topological_dimension
        family = "triangle" if top == 2 else "tetrahedron"
        if (top, mesh.ambient_dimension) not in ((2, 3), (3, 3)) or any(
            block.cell_kind != family for block in mesh.blocks
        ):
            raise ValueError(
                f"ImplicitAssociationTransfer.source_mesh must be a triangular surface or tetrahedral volume in three dimensions; got dimensions {(top, mesh.ambient_dimension)!r}, cell kinds {tuple(block.cell_kind for block in mesh.blocks)!r}"
            )
        if not is_affine_cell_geometry(mesh, source.geometry, budget=budget):
            raise ValueError(
                f"ImplicitAssociationTransfer.source_geometry must be affine; got geometry {source.geometry.geometry_layout_id!r} on mesh {mesh.mesh_id!r}"
            )
        boundary = source.boundary
        if top == 2 and (boundary is None or boundary.mesh.mesh_id != mesh.mesh_id):
            raise ValueError(
                f"ImplicitAssociationTransfer.boundary must bind the source mesh; got boundary mesh {None if boundary is None else boundary.mesh.mesh_id!r}, expected {mesh.mesh_id!r}"
            )
        if boundary is not None and (
            boundary.metadata.source_id,
            boundary.metadata.source_revision,
        ) != (
            self.profile.source_id,
            self.profile.source_revision,
        ):
            raise ValueError(
                f"ImplicitAssociationTransfer.boundary.metadata must identify the profile source and revision; got {(boundary.metadata.source_id, boundary.metadata.source_revision)!r}, expected {(self.profile.source_id, self.profile.source_revision)!r}"
            )
        if len(source.associations) != 1:
            raise ValueError(
                f"ImplicitAssociationTransfer.source_associations must contain one face table; got count {len(source.associations)}"
            )
        association = source.associations[0]
        if association.association_kind is not GeometryAssociationKind.IMPLICIT or (
            association.source_id,
            association.source_revision,
        ) != (self.profile.source_id, self.profile.source_revision):
            raise ValueError(
                f"ImplicitAssociationTransfer.source_association must identify the profile zero set; got kind {association.association_kind!r}, source {(association.source_id, association.source_revision)!r}"
            )
        association.validate_target(mesh.entity_set(2))
        expected = _implicit_face_ids(mesh)
        if (
            not association.complete
            or association.source_entities.entity_ids()
            != ("implicit-zero-set",) * association.target_global_ids.size
            or not np.array_equal(
                np.sort(np.asarray(association.target_global_ids)), np.sort(expected)
            )
        ):
            raise ValueError(
                f"ImplicitAssociationTransfer.source_association must resolve all cells as implicit-zero-set; got coverage {association.complete}, source entities {association.source_entities.entity_ids()!r}"
            )
        if source.certification is None or source.certification.request.source is None:
            raise ValueError(
                f"ImplicitAssociationTransfer.source_certification must retain source-fidelity inputs; got certification present {source.certification is not None}, source input present {source.certification is not None and source.certification.request.source is not None}"
            )
        query = source.certification.request.source
        if (query.source_id, query.source_revision) != (
            self.profile.source_id,
            self.profile.source_revision,
        ):
            raise ValueError(
                f"ImplicitAssociationTransfer.fidelity_source must identify the profile source and revision; got {(query.source_id, query.source_revision)!r}, expected {(self.profile.source_id, self.profile.source_revision)!r}"
            )
        match query:
            case ImplicitProjectionBoundarySource():
                actual_profile = query.profile
            case _:
                actual_profile = query
        match actual_profile:
            case AnalyticImplicitProfile():
                pass
            case _:
                raise TypeError(
                    f"ImplicitAssociationTransfer.fidelity_source must be an AnalyticImplicitProfile; got {type(actual_profile).__name__}"
                )
        actual_profile.require_bound(
            self.profile.geometry, source.coordinate_contract, budget=budget
        )
        if actual_profile.profile_id != self.profile.profile_id:
            raise ValueError(
                f"ImplicitAssociationTransfer.profile_id must match the fidelity profile; got {actual_profile.profile_id!r}, expected {self.profile.profile_id!r}"
            )
        return association, (2,)

    def classes(
        self, source: CellMeshingResult, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> tuple[ImplicitEntityClasses, ...]:
        self.source_associations(source, budget=budget)
        mesh = source.mesh
        result = []
        for degree in range(mesh.topological_dimension + 1):
            count = mesh.entity_set(degree).count
            dimensions = (
                np.full(count, 2, dtype=np.int64)
                if mesh.topological_dimension == 2
                else np.where(boundary_mask(mesh, degree), 2, 3).astype(np.int64)
            )
            result.append(
                ImplicitEntityClasses(
                    dimensions,
                    np.zeros(count, dtype=np.int64),
                    np.ones(count, dtype=np.bool_),
                    np.zeros(count, dtype=np.int64),
                )
            )
        return tuple(result)

    def _require_lineage(
        self, source: CellMeshingResult, lineage: MeshLineage, target: CellMesh,
        *, budget: CoordinateEnclosureBudget,
    ) -> GeometryAssociation:
        association, _ = self.source_associations(source, budget=budget)
        record = lineage.entity_lineage(2)
        require_transfer_lineage(
            source.mesh,
            lineage,
            target,
            message=f"ImplicitAssociationTransfer.lineage.topology_ids must bind source and target meshes; got lineage {(lineage.source_topology_id, lineage.target_topology_id)!r}, meshes {(source.mesh.topology_id, target.topology_id)!r}",
        )
        if (
            record.source_entity_set_id != source.mesh.entity_set(2).entity_set_id
            or record.target_entity_set_id != target.entity_set(2).entity_set_id
        ):
            raise ValueError(
                f"ImplicitAssociationTransfer.face_lineage must bind the source and target face sets; got {(record.source_entity_set_id, record.target_entity_set_id)!r}, expected {(source.mesh.entity_set(2).entity_set_id, target.entity_set(2).entity_set_id)!r}"
            )
        target_ids = _implicit_face_ids(target)
        relevant = np.isin(np.asarray(record.target_global_ids), target_ids)
        source_ids = np.asarray(association.target_global_ids)
        created_boundary = np.intersect1d(
            np.asarray(record.created_target_ids), target_ids
        )
        foreign_parents = np.setdiff1d(
            np.asarray(record.source_global_ids)[relevant], source_ids
        )
        if created_boundary.size or foreign_parents.size:
            raise ValueError(
                f"ImplicitAssociationTransfer.face_lineage must derive every target boundary face from declared source boundary parents; got created boundary IDs {created_boundary!r}, foreign source IDs {foreign_parents!r}; target boundary faces={target_ids.size}, total target faces={target.entity_set(2).count}, total created faces={np.asarray(record.created_target_ids).size}"
            )
        supported = np.asarray(
            (
                EntityLineageKind.PRESERVED,
                EntityLineageKind.REFINED_FROM,
                EntityLineageKind.COARSENED_INTO,
                EntityLineageKind.COLLAPSED_INTO,
                EntityLineageKind.RELOCATED,
            ),
            dtype=np.int32,
        )
        if not np.all(np.isin(np.asarray(record.relation_kinds)[relevant], supported)):
            raise ValueError(
                f"ImplicitAssociationTransfer.face_lineage.relation_kinds must preserve, refine, coarsen, collapse, or relocate faces; got {np.asarray(record.relation_kinds)!r}"
            )
        if not np.array_equal(
            np.unique(np.asarray(record.target_global_ids)[relevant]),
            np.sort(target_ids),
        ):
            raise ValueError(
                f"ImplicitAssociationTransfer.face_lineage.target_global_ids must cover every target face; got {np.asarray(record.target_global_ids)!r}, expected {np.asarray(target.entity_set(2).entity_ids)!r}"
            )
        return association

    @checked
    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[GeometryAssociation, ...]:
        association = self._require_lineage(source, lineage, target, budget=budget)
        if not is_affine_cell_geometry(target, geometry, budget=budget):
            raise ValueError(
                f"ImplicitAssociationTransfer.target_geometry must be affine; got layout {geometry.geometry_layout_id!r}"
            )
        identifiers = _implicit_face_ids(target)
        count = identifiers.size
        if count > self.maximum_queries:
            raise ValueError(
                f"ImplicitAssociationTransfer.maximum_queries must admit target faces; got count {count}, limit {self.maximum_queries}"
            )
        # Each target face centre evaluates the source boundary field once.
        budget.reserve_geometry_queries(count)
        record = lineage.entity_lineage(2)
        relevant = np.isin(np.asarray(record.target_global_ids), identifiers)
        source_ids, target_ids = (
            np.asarray(record.source_global_ids)[relevant],
            np.asarray(record.target_global_ids)[relevant],
        )
        association.target_rows(source_ids)
        parents = np.full(count, -1, dtype=np.int64)
        for row, identifier in enumerate(identifiers):
            actual = np.unique(source_ids[target_ids == identifier])
            if actual.size == 1:
                parents[row] = actual[0]
        faces = ordered_vertices(target, 2)
        if faces is None:
            raise ValueError(
                "ImplicitAssociationTransfer.target_faces must provide simplex face incidence; got unsupported topology"
            )
        cells = faces[entity_rows(target, 2, identifiers)]
        centers = jnp.mean(target.coordinates[cells], axis=1)
        residuals = jnp.abs(self.profile.geometry.boundary_field(centers))
        return (
            GeometryAssociation(
                GeometryAssociationKind.IMPLICIT,
                self.profile.source_id,
                self.profile.source_revision,
                target.entity_set(2).entity_set_id,
                identifiers,
                ("implicit-zero-set",) * count,
                residuals,
                resolved=np.ones(count, dtype=np.bool_),
                exact=False,
                source_dimensions=np.full(count, 2, dtype=np.int32),
                source_indices=np.zeros(count, dtype=np.int64),
                parent_dimensions=np.where(parents >= 0, 2, -1).astype(np.int32),
                parent_ids=parents,
                provenance=GeometryAssociationProvenance.LINEAGE,
                parent_association_id=association.association_id,
            ),
        )

    @checked
    def remap_boundary(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        budget: CoordinateEnclosureBudget,
    ) -> SurfaceModel:
        self._require_lineage(source, lineage, target, budget=budget)
        if not is_affine_cell_geometry(target, geometry, budget=budget):
            raise ValueError(
                f"ImplicitAssociationTransfer.target_geometry must be affine for boundary remapping; got layout {geometry.geometry_layout_id!r}"
            )
        boundary = source.boundary
        if boundary is None:
            raise ValueError(
                f"ImplicitAssociationTransfer.boundary must retain a SurfaceModel; got {boundary!r}"
            )
        target_ids = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)
        tags: list[str | None] = [None] * target_ids.size
        if boundary.metadata.cell_tags:
            raw_ids = np.concatenate(
                [np.asarray(block.global_ids) for block in boundary.mesh.blocks]
            )
            for tag in dict.fromkeys(boundary.metadata.cell_tags):
                selected = raw_ids[
                    np.asarray(
                        [value == tag for value in boundary.metadata.cell_tags],
                        dtype=np.bool_,
                    )
                ]
                inherited = np.asarray(
                    _inherited_surface_ids(source.mesh, target, lineage, selected, tag)
                )
                for row in np.flatnonzero(np.isin(target_ids, inherited)):
                    if tags[row] is not None:
                        raise ValueError(
                            f"ImplicitAssociationTransfer.boundary.cell_tags must remain disjoint after lineage inheritance; got target row {row}, previous tag {tags[row]!r}, inherited tag {tag!r}"
                        )
                    tags[row] = tag
            if any(tag is None for tag in tags):
                raise ValueError(
                    f"ImplicitAssociationTransfer.boundary.cell_tags must cover every target cell; got uncovered rows {tuple(row for row, tag in enumerate(tags) if tag is None)!r}"
                )
        ordered_tags = tuple(tag for tag in tags if tag is not None)
        raw_target_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in target.blocks]
        )
        if ordered_tags:
            order = np.argsort(target_ids, kind="stable")
            sorted_ids = target_ids[order]
            positions = np.searchsorted(sorted_ids, raw_target_ids)
            if (
                raw_target_ids.size != target_ids.size
                or np.unique(raw_target_ids).size != target_ids.size
                or np.unique(target_ids).size != target_ids.size
                or np.any(positions >= sorted_ids.size)
            ):
                raise ValueError(
                    f"ImplicitAssociationTransfer.target_cell_ids must uniquely cover the target face IDs; got block IDs {raw_target_ids!r}, face IDs {target_ids!r}"
                )
            if not np.array_equal(sorted_ids[positions], raw_target_ids):
                raise ValueError(
                    f"ImplicitAssociationTransfer.target_cell_ids must belong to the target face set; got block IDs {raw_target_ids!r}, face IDs {target_ids!r}"
                )
            block_tags = tuple(ordered_tags[row] for row in order[positions])
        else:
            block_tags = ()
        metadata = SurfaceMetadata(
            source_id=boundary.metadata.source_id,
            source_revision=boundary.metadata.source_revision,
            coordinate_contract=boundary.metadata.coordinate_contract,
            provenance=(
                *boundary.metadata.provenance,
                boundary.model_id,
                lineage.lineage_id,
            ),
            cell_tags=block_tags,
        )

        def selection(value: SurfaceSelection) -> SurfaceSelection:
            return SurfaceSelection(
                value.name,
                _inherited_surface_ids(
                    source.mesh, target, lineage, value.cell_global_ids, value.name
                ),
                cell_entity_set_id=target.entity_set(2).entity_set_id,
                role=value.role,
            )

        selections = tuple(selection(value) for value in boundary.selections)
        interfaces = tuple(
            SurfaceInterface(
                value.name,
                selection(value.support),
                minus_region=value.minus_region,
                plus_region=value.plus_region,
            )
            for value in boundary.interfaces
        )
        # Orientation repair retains the source provenance, not an
        # invented target repair. Native lineage carries the descendant relations.
        return SurfaceModel(
            target,
            metadata,
            selections=selections,
            interfaces=interfaces,
            orientation_repair=boundary.orientation_repair,
        )


__all__ = ["ImplicitAssociationTransfer"]
