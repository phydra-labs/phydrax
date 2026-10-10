#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Original implicit-source authority and boundary-model lineage transport."""

from __future__ import annotations

from typing import final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_transfer import is_affine_cell_geometry
from ..discretization._cell_mesh import CellMesh
from ..geometry._mesh_certificates import ImplicitProjectionBoundarySource
from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ..geometry.surface import (
    SurfaceInterface,
    SurfaceMetadata,
    SurfaceModel,
    SurfaceSelection,
)
from ._association import (
    _entity_rows,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
)
from ._lineage import EntityLineageKind, inherit_scope, MeshLineage
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


@final
class ImplicitAssociationTransfer(StrictModule, NonTrainableState):
    """Carry a certified analytic implicit surface through native carrier edits.

    The scalar source, state and revision remain authoritative. The independent
    successor source-fidelity certificate, not sampled residuals, admits the
    target. Surface metadata follows explicit cell lineage, never point matches.
    """

    profile: AnalyticImplicitProfile
    maximum_queries: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self, profile: AnalyticImplicitProfile, /, *, maximum_queries: int
    ) -> None:
        if not isinstance(profile, AnalyticImplicitProfile):
            raise TypeError(
                "Implicit association transport requires its owning analytic source profile."
            )
        if isinstance(maximum_queries, bool) or not isinstance(maximum_queries, int):
            raise TypeError("maximum_queries must be an integer.")
        if maximum_queries < 1:
            raise ValueError(
                "Implicit association transport requires a positive authored query budget."
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

    def source_associations(
        self, source: CellMeshingResult, /
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        self.profile.require_bound(self.profile.geometry, source.coordinate_contract)
        mesh = source.mesh
        if (mesh.topological_dimension, mesh.ambient_dimension) != (2, 3) or any(
            block.cell_kind != "triangle" for block in mesh.blocks
        ):
            raise ValueError(
                "Implicit boundary transport requires its actual triangular surface carrier."
            )
        if not is_affine_cell_geometry(mesh, source.geometry):
            raise ValueError(
                "An affine SurfaceModel cannot represent a curved implicit carrier."
            )
        boundary = source.boundary
        if boundary is None or boundary.mesh.mesh_id != mesh.mesh_id:
            raise ValueError(
                "Implicit boundary transport requires its exact original SurfaceModel carrier."
            )
        if (boundary.metadata.source_id, boundary.metadata.source_revision) != (
            self.profile.source_id,
            self.profile.source_revision,
        ):
            raise ValueError(
                "Implicit SurfaceModel refers to a different scalar source or revision."
            )
        if len(source.associations) != 1:
            raise ValueError(
                "Implicit surface transport requires one complete original face association."
            )
        association = source.associations[0]
        if association.association_kind is not GeometryAssociationKind.IMPLICIT or (
            association.source_id,
            association.source_revision,
        ) != (self.profile.source_id, self.profile.source_revision):
            raise ValueError(
                "Implicit association transport cannot replace source authority."
            )
        association.validate_target(mesh.entity_set(2))
        if (
            not association.complete
            or association.source_entity_ids
            != ("implicit-zero-set",) * association.target_global_ids.size
        ):
            raise ValueError(
                "Implicit face authority must cover every original carrier cell."
            )
        if source.certification is None or source.certification.request.source is None:
            raise ValueError(
                "Implicit source transport requires original source-fidelity inputs."
            )
        query = source.certification.request.source
        if (query.source_id, query.source_revision) != (
            self.profile.source_id,
            self.profile.source_revision,
        ):
            raise ValueError(
                "Implicit fidelity and face authority use different scalar sources."
            )
        actual_profile = (
            query.profile
            if isinstance(query, ImplicitProjectionBoundarySource)
            else query
        )
        if not isinstance(actual_profile, AnalyticImplicitProfile):
            raise TypeError(
                "Implicit face transport requires the original source-owned analytic query."
            )
        actual_profile.require_bound(self.profile.geometry, source.coordinate_contract)
        if actual_profile.profile_id != self.profile.profile_id:
            raise ValueError(
                "Implicit transfer uses a different prepared scalar-source proof."
            )
        return association, (2,)

    def classes(self, source: CellMeshingResult, /) -> tuple[ImplicitEntityClasses, ...]:
        self.source_associations(source)
        return tuple(
            ImplicitEntityClasses(
                np.full(source.mesh.entity_set(degree).count, 2, dtype=np.int64),
                np.zeros(source.mesh.entity_set(degree).count, dtype=np.int64),
                np.ones(source.mesh.entity_set(degree).count, dtype=np.bool_),
                np.zeros(source.mesh.entity_set(degree).count, dtype=np.int64),
            )
            for degree in range(3)
        )

    def _require_lineage(
        self, source: CellMeshingResult, lineage: MeshLineage, target: CellMesh
    ) -> None:
        self.source_associations(source)
        record = lineage.entity_lineage(2)
        if (
            lineage.source_topology_id != source.mesh.topology_id
            or lineage.target_topology_id != target.topology_id
            or record.source_entity_set_id != source.mesh.entity_set(2).entity_set_id
            or record.target_entity_set_id != target.entity_set(2).entity_set_id
        ):
            raise ValueError(
                "Implicit transport requires the exact oriented source and target face lineage."
            )
        if record.created_target_ids.size:
            raise ValueError(
                "Implicit target faces require declared original-source lineage, not guessed ancestry."
            )
        supported = np.asarray(
            (
                EntityLineageKind.PRESERVED,
                EntityLineageKind.REFINED_FROM,
                EntityLineageKind.COARSENED_INTO,
                EntityLineageKind.RELOCATED,
            ),
            dtype=np.int32,
        )
        if not np.all(np.isin(np.asarray(record.relation_kinds), supported)):
            raise ValueError(
                "Implicit carrier transport lacks a supported face-preservation relation."
            )
        if not np.array_equal(
            np.unique(np.asarray(record.target_global_ids)),
            np.sort(np.asarray(target.entity_set(2).entity_ids)),
        ):
            raise ValueError("Implicit face lineage leaves target authority incomplete.")

    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
    ) -> tuple[GeometryAssociation, ...]:
        self._require_lineage(source, lineage, target)
        if not is_affine_cell_geometry(target, geometry):
            raise ValueError(
                "An affine implicit boundary model cannot certify a curved successor."
            )
        count = target.entity_set(2).count
        if count > self.maximum_queries:
            raise ValueError(
                "Implicit association queries exceed their authored preallocation budget."
            )
        association, _ = self.source_associations(source)
        record = lineage.entity_lineage(2)
        identifiers = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)
        source_ids, target_ids = (
            np.asarray(record.source_global_ids),
            np.asarray(record.target_global_ids),
        )
        association.target_rows(source_ids)
        parents = np.full(count, -1, dtype=np.int64)
        for row, identifier in enumerate(identifiers):
            actual = np.unique(source_ids[target_ids == identifier])
            if actual.size == 1:
                parents[row] = actual[0]
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int32) for block in target.blocks]
        )
        centers = jnp.mean(target.coordinates[cells], axis=1)
        block_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in target.blocks]
        )
        residuals = jnp.abs(self.profile.geometry.boundary_field(centers))[
            _entity_rows(target, 2, block_ids)
        ]
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

    def remap_boundary(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
    ) -> SurfaceModel:
        self._require_lineage(source, lineage, target)
        if not is_affine_cell_geometry(target, geometry):
            raise ValueError(
                "An affine SurfaceModel cannot replace the actual curved successor map."
            )
        boundary = source.boundary
        if boundary is None:
            raise ValueError(
                "Implicit surface transport lost its original boundary model."
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
                            "Implicit boundary tags overlap after a source-lineage merge."
                        )
                    tags[row] = tag
            if any(tag is None for tag in tags):
                raise ValueError(
                    "Implicit boundary lineage leaves original cell tags uncovered."
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
                    "Implicit target cell presentation does not cover its face SCI identities."
                )
            if not np.array_equal(sorted_ids[positions], raw_target_ids):
                raise ValueError(
                    "Implicit target cell presentation contains foreign face SCI identities."
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
        # Orientation repair remains the original source provenance, not an
        # invented target repair. The exact native lineage carries its descendants.
        return SurfaceModel(
            target,
            metadata,
            selections=selections,
            interfaces=interfaces,
            orientation_repair=boundary.orientation_repair,
        )


__all__ = ["ImplicitAssociationTransfer"]
