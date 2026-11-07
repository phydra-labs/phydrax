#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Preserve implicit source strata through carrier topology edits."""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._cell_complex import TetrahedralConnectivity
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..discretization._cell_mesh import SimplicialConnectivity
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryFaceImageWitness,
    SourceBoundaryProjectionQuery,
    SourceProjectionStatus,
)
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ..geometry.surface import SurfaceMetadata, SurfaceModel
from ..typing import checked
from ._certification_inputs import ScopedFidelityInput
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
    boundary_mask,
    entity_rows,
    incidence_pairs,
    require_transfer_lineage,
    target_dimension,
)
from ._lineage import EntityLineageKind, MeshLineage, VertexInterpolationStencil
from ._result import CellMeshingResult
from ._topology_edit import key_rows
from .providers._native_sources import NativeImplicitSource


# Outward local faces (opposite vertices 0..3) of a positively oriented tetrahedron.
_OUTWARD_TETRAHEDRON_FACES = np.asarray(
    ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64
)


@final
class ImplicitAssociationTransfer(StrictModule, NonTrainableState):
    """Carry an analytic or explicitly authored native implicit namespace.

    Boundary strata remain the declared zero-set and are verified by the
    profile or retained source-bound boundary query after edits. Interior
    regions contain that surface in their closure; topology edits cannot replace
    this classification with a nearest-source guess.
    """

    profile: AnalyticImplicitProfile | None
    native_source: NativeImplicitSource | None
    native_source_binding_id: str | None = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        profile: AnalyticImplicitProfile | NativeImplicitSource,
        /,
    ) -> None:
        if isinstance(profile, NativeImplicitSource):
            native_source = profile
            profile_ = native_source.authoring_profile
            if profile_ is not None and not isinstance(profile_, AnalyticImplicitProfile):
                raise TypeError(
                    "NativeImplicitSource.authoring_profile must be AnalyticImplicitProfile or None."
                )
            if profile_ is None and native_source.discovery is None:
                raise ValueError(
                    "NativeImplicitSource must retain an authoring profile or adaptive discovery."
                )
            namespace = (native_source.source_id, native_source.source_revision)
        elif isinstance(profile, AnalyticImplicitProfile):
            native_source = None
            profile_ = profile
            namespace = (profile.source_id, profile.source_revision)
        else:
            raise TypeError(
                "profile must be AnalyticImplicitProfile or NativeImplicitSource."
            )
        if profile_ is not None:
            profile_.validate_source_integrity()
        self.profile = profile_
        self.native_source = native_source
        self.native_source_binding_id = (
            None if native_source is None else native_source.binding_id
        )
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "implicit-association-transfer",
                "source": namespace,
                "profile": None if profile_ is None else profile_.profile_id,
                "native_source_binding": self.native_source_binding_id,
            }
        )

    @property
    def source_namespaces(self) -> tuple[tuple[str, str], ...]:
        if self.profile is not None:
            return ((self.profile.source_id, self.profile.source_revision),)
        if self.native_source is not None:
            return ((self.native_source.source_id, self.native_source.source_revision),)
        raise RuntimeError("Implicit association transfer lost its source authority.")

    def _source_entity_key_for_scope(
        self, dimension: int, /
    ) -> tuple[str, tuple[str, ...]]:
        authority = self.profile if self.profile is not None else self.native_source
        if authority is None:
            raise RuntimeError("Implicit association transfer lost its source authority.")
        identifiers = authority.scope_entity_ids(dimension)
        if identifiers.size != 1:
            raise ValueError(
                "Implicit source authority must retain one entity in each boundary or region stratum."
            )
        return authority.source_entity_key_for_scope(dimension, int(identifiers[0]))

    def _project_boundary(
        self, source: CellMeshingResult, points: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.profile is not None:
            projection = self.profile.project_boundary(points)
        else:
            if source.certification is None:
                raise ValueError(
                    "Native implicit projection requires accepted source certification."
                )
            query = source.certification.inputs.source
            if not isinstance(query, SourceBoundaryProjectionQuery):
                raise ValueError(
                    "Native implicit projection requires its typed boundary projection query."
                )
            if (query.source_id, query.source_revision) != self.source_namespaces[
                0
            ] or query.frame_id != source.coordinate_contract.spatial_id:
                raise ValueError(
                    "Implicit boundary query must bind the exact source revision and frame."
                )
            entity_id, occurrence_path = self._source_entity_key_for_scope(2)
            projection = query.project_boundary(
                points,
                entity_ids=(entity_id,) * points.shape[0],
                occurrence_paths=(occurrence_path,) * points.shape[0],
            )
        status = np.asarray(projection.status, dtype=np.int8)
        resolved = np.isin(
            status,
            (SourceProjectionStatus.UNIQUE, SourceProjectionStatus.SEAM),
        )
        return (
            resolved,
            np.asarray(projection.residuals, dtype=np.float64),
            np.asarray(projection.parameters, dtype=np.float64),
        )

    def validate_source(self, source: CellMeshingResult, /) -> None:
        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        namespace = self.source_namespaces[0]
        if self.profile is not None:
            self.profile.require_bound(self.profile.geometry, source.coordinate_contract)
        if source.certification is None:
            if self.profile is None:
                raise ValueError(
                    "Native implicit association requires accepted source certification."
                )
        else:
            inputs = source.certification.inputs
            authority = inputs.feature_source
            if self.native_source_binding_id is not None and (
                not isinstance(authority, NativeImplicitSource)
                or authority.binding_id != self.native_source_binding_id
            ):
                raise ValueError(
                    "Native implicit result must retain the exact NativeImplicitSource binding."
                )
            if isinstance(authority, NativeImplicitSource):
                if (authority.source_id, authority.source_revision) != namespace:
                    raise ValueError(
                        "Native implicit authoring source identity differs from its associations."
                    )
                if self.profile is not None:
                    retained_profile = authority.authoring_profile
                    if not isinstance(retained_profile, AnalyticImplicitProfile):
                        raise ValueError(
                            "Native implicit result must retain its typed authoring profile."
                        )
                    if retained_profile.profile_id != self.profile.profile_id:
                        raise ValueError(
                            "Implicit transfer profile differs from the retained authoring profile."
                        )
                    self.profile.require_bound(
                        authority.geometry, source.coordinate_contract
                    )
                elif authority.discovery is None:
                    raise ValueError(
                        "Native implicit source must retain exact adaptive discovery."
                    )
            elif isinstance(authority, AnalyticImplicitProfile):
                if (
                    self.profile is None
                    or authority.profile_id != self.profile.profile_id
                ):
                    raise ValueError(
                        "Implicit transfer requires the exact certified authoring profile."
                    )
                self.profile.require_bound(authority.geometry, source.coordinate_contract)
            elif self.profile is not None and isinstance(
                inputs.domain, AnalyticImplicitProfile
            ):
                if inputs.domain.profile_id != self.profile.profile_id:
                    raise ValueError(
                        "Implicit certification domain differs from the transfer authoring profile."
                    )
            else:
                raise ValueError(
                    "Accepted implicit results must retain a typed authoring source."
                )
            if self.profile is None:
                query = inputs.source
                if (
                    not isinstance(query, SourceBoundaryProjectionQuery)
                    or (query.source_id, query.source_revision) != namespace
                    or query.frame_id != source.coordinate_contract.spatial_id
                ):
                    raise ValueError(
                        "Native implicit boundary query must bind the source identity and frame."
                    )
        mesh = source.mesh
        if mesh.ambient_dimension != 3 or mesh.topological_dimension not in (2, 3):
            raise ValueError(
                "Implicit association transfer requires a surface or volume carrier in three dimensions."
            )
        expected = (
            ("triangle",)
            if mesh.topological_dimension == 2
            else ("tetrahedron", "simplex:3")
        )
        if not mesh.blocks or any(
            block.cell_kind not in expected for block in mesh.blocks
        ):
            raise ValueError(
                "Implicit association transfer requires triangular or tetrahedral simplex blocks."
            )
        tables = associations_for_namespace(
            source,
            namespace,
            kinds=(GeometryAssociationKind.IMPLICIT,),
        )
        by_dimension = {
            target_dimension(mesh, association): association for association in tables
        }
        if mesh.topological_dimension not in by_dimension:
            raise ValueError(
                "Implicit source associations must classify every carrier cell."
            )
        cell_table = by_dimension[mesh.topological_dimension]
        expected_cells = np.asarray(
            mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
        )
        if not np.array_equal(
            np.sort(np.asarray(cell_table.target_global_ids, dtype=np.int64)),
            np.sort(expected_cells),
        ):
            raise ValueError(
                "Implicit source cell associations must cover the complete carrier."
            )
        expected_role = (
            GeometrySourceEntityRole.FACET
            if mesh.topological_dimension == 2
            else GeometrySourceEntityRole.REGION
        )
        if any(role is not expected_role for role in cell_table.source_roles):
            raise ValueError(
                "Implicit source cell associations must retain their zero-set or interior-region role."
            )
        for association in tables:
            if not association.complete:
                raise ValueError(
                    "Implicit source association tables must be resolved and unambiguous."
                )
            if any(
                role
                not in (GeometrySourceEntityRole.FACET, GeometrySourceEntityRole.REGION)
                for role in association.source_roles
            ):
                raise ValueError(
                    "Implicit source associations must use facet and region strata only."
                )

    def classes(
        self, source: CellMeshingResult, /
    ) -> tuple[AssociationEntityClasses, ...]:
        self.validate_source(source)
        mesh = source.mesh
        tables = {
            target_dimension(mesh, association): association
            for association in source.associations
            if association.association_kind is GeometryAssociationKind.IMPLICIT
            and (association.source_id, association.source_revision)
            == self.source_namespaces[0]
        }
        has_surface_source = any(
            role is GeometrySourceEntityRole.FACET
            for association in tables.values()
            for role in association.source_roles
        )
        result: list[AssociationEntityClasses] = []
        for dimension in range(mesh.topological_dimension + 1):
            identifiers = np.asarray(
                mesh.entity_set(dimension).entity_ids, dtype=np.int64
            )
            if mesh.topological_dimension == 2:
                strata = np.full(identifiers.shape, 2, dtype=np.int64)
            else:
                strata = np.full(identifiers.shape, 3, dtype=np.int64)
                if has_surface_source and dimension < mesh.topological_dimension:
                    strata[boundary_mask(mesh, dimension)] = 2
                if dimension == 3:
                    strata.fill(3)
            resolved = np.ones(identifiers.shape, dtype=np.bool_)
            table = tables.get(dimension)
            if table is not None:
                rows = entity_rows(
                    mesh, dimension, np.asarray(table.target_global_ids, dtype=np.int64)
                )
                if np.any(rows < 0):
                    raise ValueError(
                        "Implicit association table names entities outside its carrier."
                    )
                table_strata = np.asarray(
                    [
                        2 if role is GeometrySourceEntityRole.FACET else 3
                        for role in table.source_roles
                    ],
                    dtype=np.int64,
                )
                if not np.array_equal(table_strata, strata[rows]):
                    raise ValueError(
                        "Implicit source associations disagree with carrier stratum closure."
                    )
            codes = np.where(strata == 2, 1, 2).astype(np.int64)
            result.append(
                AssociationEntityClasses(
                    strata,
                    np.zeros(identifiers.shape, dtype=np.int64),
                    resolved,
                    codes,
                )
            )
        return tuple(result)

    def _check_collapses(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        classes: tuple[AssociationEntityClasses, ...],
        target: CellMesh,
        target_classes: tuple[AssociationEntityClasses, ...],
        /,
    ) -> None:
        record = lineage.entity_lineage(0)
        kinds = np.asarray(record.relation_kinds, dtype=np.int32)
        collapsed = kinds == int(EntityLineageKind.COLLAPSED_INTO)
        if not np.any(collapsed):
            return
        removed_ids = np.asarray(record.source_global_ids, dtype=np.int64)[collapsed]
        kept_ids = np.asarray(record.target_global_ids, dtype=np.int64)[collapsed]
        removed_rows = entity_rows(source.mesh, 0, removed_ids)
        kept_rows = entity_rows(source.mesh, 0, kept_ids)
        kept_target_rows = entity_rows(target, 0, kept_ids)
        if np.any(removed_rows < 0) or np.any(kept_target_rows < 0):
            raise ValueError(
                "Implicit vertex-collapse lineage must name a source vertex and a target vertex."
            )
        # A vertex may collapse into one born during adaptation (a split point); the
        # born vertex's stratum is its target classification.
        retained = kept_rows >= 0
        dimensions = classes[0].dimensions
        kept_dimensions = np.where(
            retained,
            dimensions[np.maximum(kept_rows, 0)],
            target_classes[0].dimensions[kept_target_rows],
        )
        legal = dimensions[removed_rows] >= kept_dimensions
        associations = tuple(
            value
            for value in source.associations
            if value.association_kind is GeometryAssociationKind.IMPLICIT
            and (value.source_id, value.source_revision) == self.source_namespaces[0]
            and target_dimension(source.mesh, value) == 0
        )
        if associations and np.any(retained):
            # One implicit namespace retains one entity per stratum, so only kept
            # source vertices carry an occurrence path to compare.
            association = associations[0]
            removed_assoc_rows = association.target_rows(removed_ids[retained])
            kept_assoc_rows = association.target_rows(kept_ids[retained])
            legal[retained] &= np.asarray(
                [
                    association.source_occurrence_paths[int(first)]
                    == association.source_occurrence_paths[int(second)]
                    for first, second in zip(
                        removed_assoc_rows, kept_assoc_rows, strict=True
                    )
                ],
                dtype=np.bool_,
            )
        if not np.all(legal):
            raise AssociationPropagationError(
                "Implicit collapse violates source-stratum closure.", kept_ids[~legal]
            )

    @checked
    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        vertex_stencil: VertexInterpolationStencil | None,
        geometry_transition: CellGeometryTransition | None,
        target_fidelity_witness: SourceBoundaryChartWitness | SourceBoundaryFaceImageWitness | None,
        target_scoped_fidelity: tuple[ScopedFidelityInput, ...],
        validated_fidelity: tuple[ValidatedSourceFidelityWitness, ...],
    ) -> tuple[GeometryAssociation, ...]:
        self.validate_source(source)
        if not isinstance(lineage, MeshLineage) or not isinstance(target, CellMesh):
            raise TypeError("lineage and target must be MeshLineage and CellMesh values.")
        if (target.ambient_dimension, target.topological_dimension) not in (
            (3, 2),
            (3, 3),
        ):
            raise ValueError(
                "Implicit association transfer requires a surface or volume carrier in three dimensions."
            )
        target_kinds = (
            ("triangle",)
            if target.topological_dimension == 2
            else ("tetrahedron", "simplex:3")
        )
        if not target.blocks or any(
            block.cell_kind not in target_kinds for block in target.blocks
        ):
            raise ValueError(
                "Implicit association transfer requires triangular or tetrahedral simplex blocks."
            )
        require_transfer_lineage(
            source.mesh,
            lineage,
            target,
            dimensions=tuple(range(target.topological_dimension + 1)),
        )
        if vertex_stencil is not None and not isinstance(
            vertex_stencil, VertexInterpolationStencil
        ):
            raise TypeError("vertex_stencil must be VertexInterpolationStencil or None.")
        namespace = self.source_namespaces[0]
        source_classes = self.classes(source)
        source_tables = {
            target_dimension(source.mesh, association): association
            for association in source.associations
            if association.association_kind is GeometryAssociationKind.IMPLICIT
            and (association.source_id, association.source_revision) == namespace
        }
        has_surface_source = any(
            role is GeometrySourceEntityRole.FACET
            for association in source_tables.values()
            for role in association.source_roles
        )
        target_classes = self._classes_for_mesh(target, has_surface_source)
        self._check_collapses(source, lineage, source_classes, target, target_classes)
        source_keys: dict[GeometrySourceEntityRole, tuple[str, tuple[str, ...]]] = {}
        for association in source_tables.values():
            for entity_id, role, path in zip(
                association.source_entity_ids,
                association.source_roles,
                association.source_occurrence_paths,
                strict=True,
            ):
                if role is None:
                    raise ValueError(
                        "Implicit association rows must retain an explicit source role."
                    )
                key = (entity_id, path)
                previous = source_keys.get(role)
                if previous is None:
                    source_keys[role] = key
                elif previous != key:
                    raise ValueError(
                        "One implicit namespace must retain one entity per source stratum."
                    )
        surface_key = source_keys.get(GeometrySourceEntityRole.FACET)
        region_key = source_keys.get(GeometrySourceEntityRole.REGION)
        if surface_key is None:
            surface_key = self._source_entity_key_for_scope(2)
        if region_key is None:
            region_key = self._source_entity_key_for_scope(3)
        outputs: list[GeometryAssociation] = []
        for dimension, classes in enumerate(target_classes):
            entity_set = target.entity_set(dimension)
            identifiers = np.asarray(entity_set.entity_ids, dtype=np.int64)
            roles = tuple(
                GeometrySourceEntityRole.FACET
                if value == 2
                else GeometrySourceEntityRole.REGION
                for value in classes.dimensions.tolist()
            )
            keys = tuple(
                surface_key if role is GeometrySourceEntityRole.FACET else region_key
                for role in roles
            )
            parent_dimensions, parent_ids = self._parents(
                source.mesh, target, lineage, dimension, identifiers, vertex_stencil
            )
            points = self._classification_points(target, dimension, classes)
            residuals = np.zeros(identifiers.shape, dtype=np.float64)
            parameters = np.full((identifiers.size, 2), np.nan, dtype=np.float64)
            resolved = np.ones(identifiers.shape, dtype=np.bool_)
            surface_rows = np.flatnonzero(classes.dimensions == 2)
            if surface_rows.size:
                projection_resolved, projection_residuals, projection_parameters = (
                    self._project_boundary(source, points[surface_rows])
                )
                if not np.all(projection_resolved):
                    raise AssociationPropagationError(
                        "Implicit source projection could not verify target zero-set classes.",
                        identifiers[surface_rows[~projection_resolved]],
                    )
                residuals[surface_rows] = projection_residuals
                parameters[surface_rows] = projection_parameters
            outputs.append(
                GeometryAssociation(
                    GeometryAssociationKind.IMPLICIT,
                    namespace[0],
                    namespace[1],
                    entity_set.entity_set_id,
                    identifiers,
                    tuple(value[0] for value in keys),
                    residuals,
                    resolved=resolved,
                    exact=bool(np.all(classes.dimensions == 3)),
                    source_dimensions=classes.dimensions.astype(np.int8),
                    source_indices=np.zeros(identifiers.shape, dtype=np.int64),
                    source_roles=roles,
                    source_occurrence_paths=tuple(value[1] for value in keys),
                    parent_dimensions=parent_dimensions,
                    parent_ids=parent_ids,
                    parameters=parameters,
                    provenance=GeometryAssociationProvenance.LINEAGE,
                    parent_association_id=(
                        source_tables[dimension].association_id
                        if dimension in source_tables
                        else next(iter(source_tables.values())).association_id
                    ),
                )
            )
        return tuple(outputs)

    def remap_boundary(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
    ) -> SurfaceModel | None:
        """Rebuild the analytic surface carrier on the target mesh."""
        boundary = source.boundary
        if boundary is None:
            return None
        self.validate_source(source)
        if not isinstance(lineage, MeshLineage) or not isinstance(target, CellMesh):
            raise TypeError("lineage and target must be MeshLineage and CellMesh values.")
        metadata = boundary.metadata
        if (metadata.source_id, metadata.source_revision) != self.source_namespaces[0]:
            raise ValueError(
                "Implicit boundary metadata must identify the transfer source."
            )
        if (
            boundary.selections
            or boundary.interfaces
            or boundary.orientation_repair is not None
        ):
            raise ValueError(
                "Implicit boundary remapping refuses selections, interfaces, or orientation repairs."
            )
        if target.topological_dimension == 2:
            if boundary.mesh.topology_id != source.mesh.topology_id:
                raise ValueError(
                    "Implicit surface boundary must bind the source sheet topology."
                )
            target_ids = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)
            source_ids = np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
            parent_rows = {
                int(identifier): row for row, identifier in enumerate(source_ids.tolist())
            }
            source_tags = metadata.cell_tags
            if source_tags and len(source_tags) != source_ids.size:
                raise ValueError(
                    "Implicit source boundary tags must align with source cells."
                )
            record = lineage.entity_lineage(2)
            old_ids = np.asarray(record.source_global_ids, dtype=np.int64)
            new_ids = np.asarray(record.target_global_ids, dtype=np.int64)
            # Lineage parents of each target cell, in lineage-row order.
            lineage_parents: dict[int, list[int]] = {}
            for parent, child in zip(old_ids.tolist(), new_ids.tolist(), strict=True):
                lineage_parents.setdefault(child, []).append(parent)
            tags_by_id: dict[int, str] = {}
            for identifier in target_ids.tolist():
                rows = tuple(
                    parent_rows[parent]
                    for parent in lineage_parents.get(identifier, ())
                    if parent in parent_rows
                )
                if not rows and identifier in parent_rows:
                    rows = (parent_rows[identifier],)
                inherited = {source_tags[row] for row in rows} if source_tags else set()
                if len(inherited) > 1:
                    raise ValueError(
                        "Implicit surface cell tags conflict across lineage parents."
                    )
                if inherited:
                    tags_by_id[identifier] = next(iter(inherited))
            block_ids = np.concatenate(
                [np.asarray(block.global_ids, dtype=np.int64) for block in target.blocks]
            )
            if source_tags and set(tags_by_id) != set(target_ids.tolist()):
                raise ValueError(
                    "Implicit surface boundary tags must cover every target cell."
                )
            cell_tags = (
                tuple(tags_by_id[int(identifier)] for identifier in block_ids.tolist())
                if source_tags
                else ()
            )
            surface_mesh = target
        elif target.topological_dimension == 3:
            match target.connectivity:
                case TetrahedralConnectivity() as connectivity:
                    face_rows = np.asarray(connectivity.faces, dtype=np.int64)
                case SimplicialConnectivity() as connectivity:
                    face_rows = np.asarray(connectivity.entities[2], dtype=np.int64)
                case _:
                    raise ValueError(
                        "Implicit volume boundary remapping requires tetrahedral face incidence."
                    )
            boundary_rows = np.flatnonzero(boundary_mask(target, 2))
            # Canonical face rows are vertex sorted; orient each boundary face
            # outward from its unique, positively oriented owner tetrahedron.
            cells = np.concatenate(
                [np.asarray(block.vertices, dtype=np.int64) for block in target.blocks]
            )
            local_faces = cells[:, _OUTWARD_TETRAHEDRON_FACES].reshape((-1, 3))
            owners = key_rows(
                np.sort(local_faces, axis=1), np.sort(face_rows[boundary_rows], axis=1)
            )
            if np.any(owners < 0):
                raise ValueError("Implicit target boundary faces must bound one cell.")
            faces = local_faces[owners]
            face_ids = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)[
                boundary_rows
            ]
            if faces.size == 0:
                raise ValueError(
                    "Implicit target volume must retain its source boundary faces."
                )
            used = np.unique(faces)
            inverse = np.full(target.coordinates.shape[0], -1, dtype=np.int64)
            inverse[used] = np.arange(used.size, dtype=np.int64)
            surface_mesh = CellMesh.from_triangles(
                np.asarray(target.coordinates, dtype=np.float64)[used],
                inverse[faces],
                vertex_global_ids=np.asarray(target.vertex_global_ids, dtype=np.int64)[
                    used
                ],
                cell_global_ids=face_ids,
                numeric_version=target.numeric_version,
            )
            source_tags = metadata.cell_tags
            if source_tags and len(set(source_tags)) != 1:
                raise ValueError(
                    "Implicit volume boundary remapping requires one source zero-set tag."
                )
            cell_tags = (source_tags[0],) * faces.shape[0] if source_tags else ()
        else:
            raise ValueError(
                "Implicit boundary remapping requires a surface or volume target."
            )
        target_metadata = SurfaceMetadata(
            source_id=metadata.source_id,
            source_revision=metadata.source_revision,
            coordinate_contract=metadata.coordinate_contract,
            provenance=(*metadata.provenance, lineage.lineage_id),
            cell_tags=cell_tags,
        )
        return SurfaceModel(surface_mesh, target_metadata)

    def _classes_for_mesh(
        self, mesh: CellMesh, has_surface_source: bool, /
    ) -> tuple[AssociationEntityClasses, ...]:
        result = []
        for dimension in range(mesh.topological_dimension + 1):
            count = mesh.entity_set(dimension).count
            if mesh.topological_dimension == 2:
                strata = np.full((count,), 2, dtype=np.int64)
            else:
                strata = np.full((count,), 3, dtype=np.int64)
                if has_surface_source and dimension < mesh.topological_dimension:
                    strata[boundary_mask(mesh, dimension)] = 2
                if dimension == 3:
                    strata.fill(3)
            result.append(
                AssociationEntityClasses(
                    strata,
                    np.zeros((count,), dtype=np.int64),
                    np.ones((count,), dtype=np.bool_),
                    np.where(strata == 2, 1, 2).astype(np.int64),
                )
            )
        return tuple(result)

    def _parents(
        self,
        source: CellMesh,
        target: CellMesh,
        lineage: MeshLineage,
        dimension: int,
        identifiers: np.ndarray,
        vertex_stencil: VertexInterpolationStencil | None,
        /,
    ) -> tuple[np.ndarray, np.ndarray]:
        record = lineage.entity_lineage(dimension)
        old_ids = np.asarray(record.source_global_ids, dtype=np.int64)
        new_ids = np.asarray(record.target_global_ids, dtype=np.int64)
        parent_dimensions = np.full(identifiers.shape, -1, dtype=np.int8)
        parent_ids = np.full(identifiers.shape, -1, dtype=np.int64)
        # Each target's parent is its least lineage source present in the source
        # mesh, else itself when retained; one sorted pass serves every target.
        present = entity_rows(source, dimension, old_ids) >= 0
        lineage_new = new_ids[present]
        lineage_old = old_ids[present]
        order = np.lexsort((lineage_old, lineage_new))
        lineage_new = lineage_new[order]
        lineage_old = lineage_old[order]
        positions = np.searchsorted(lineage_new, identifiers, side="left")
        inside = positions < lineage_new.size
        has_parent = np.zeros(identifiers.shape, dtype=np.bool_)
        has_parent[inside] = lineage_new[positions[inside]] == identifiers[inside]
        parent_dimensions[has_parent] = dimension
        parent_ids[has_parent] = lineage_old[positions[has_parent]]
        retained = ~has_parent & (entity_rows(source, dimension, identifiers) >= 0)
        parent_dimensions[retained] = dimension
        parent_ids[retained] = identifiers[retained]
        if dimension == 0:
            created_rows = np.flatnonzero(entity_rows(source, 0, identifiers) < 0)
            if created_rows.size:
                if vertex_stencil is None:
                    raise AssociationPropagationError(
                        "Created implicit vertices require an exact source interpolation stencil.",
                        identifiers[created_rows],
                    )
                created_ids = np.sort(identifiers[created_rows])
                child_dimensions, child_rows = _child_sources(
                    lineage, source, target, created_ids, vertex_stencil
                )
                target_rows = entity_rows(target, 0, created_ids)
                for identifier, target_row, source_dimension, source_row in zip(
                    created_ids.tolist(),
                    target_rows.tolist(),
                    child_dimensions.tolist(),
                    child_rows.tolist(),
                    strict=True,
                ):
                    if source_dimension < 0 or source_row < 0:
                        raise AssociationPropagationError(
                            "Created implicit vertices require exact source-stratum support.",
                            np.asarray((identifier,), dtype=np.int64),
                        )
                    parent_dimensions[target_row] = source_dimension
                    parent_ids[target_row] = int(
                        source.entity_set(source_dimension).entity_ids[source_row]
                    )
        return parent_dimensions, parent_ids

    def _classification_points(
        self, mesh: CellMesh, dimension: int, classes: AssociationEntityClasses, /
    ) -> np.ndarray:
        coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
        if dimension == 0:
            return coordinates
        pairs = incidence_pairs(mesh, 0, dimension)
        count = mesh.entity_set(dimension).count
        sums = np.zeros((count, 3), dtype=np.float64)
        np.add.at(sums, pairs[:, 1], coordinates[pairs[:, 0]])
        weights = np.bincount(pairs[:, 1], minlength=count)
        if np.any(weights == 0):
            raise ValueError(
                "Every associated implicit entity must contain mesh vertices."
            )
        return sums / weights[:, None]


__all__ = ["ImplicitAssociationTransfer"]
