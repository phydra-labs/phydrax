#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Revision-bound association transfer for native layer/core volumes."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import final, TYPE_CHECKING

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryFaceImageWitness,
)
from ..geometry._source_fidelity import (
    validate_source_fidelity_witness,
    ValidatedSourceFidelityWitness,
)
from ._association import (
    AssociationEntityClasses,
    GeometryAssociation,
    GeometryAssociationArchiveRecipe,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
    source_boundary_association_piece_rows,
    source_boundary_image_association_piece_rows,
    SourceBoundaryTargetPieceRow,
)
from ._association_routes import (
    incidence_pairs,
    require_transfer_lineage,
    target_dimension,
)
from ._certification_inputs import (
    _validate_scoped_fidelity,
    ScopedFidelityInput,
    validated_fidelity_witness,
)
from ._contracts import VolumeMeshingSpec
from ._layer_core_boundary_replay import (
    _require_layer_fidelity_token,
    core_adjacent_face_ids,
    core_source_facet_keys,
    is_internal_core_chart,
    mapped_cap_edge_owner,
    require_scoped_cap_coverage,
    replay_composite_regions,
    replay_core_boundary_faces,
    replay_core_facet_edges,
    replay_core_interface_faces,
    replay_layer_region_vertices,
)
from ._layer_core_carrier_witness import (
    NativeLayerCoreCarrierWitness,
    require_layer_core_specification,
)
from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
from ._layer_source_association_replay import replay_layer_source_associations
from ._lineage import MeshLineage, VertexInterpolationStencil
from ._layer_core_segment_overlay import replay_source_edge_face_overlays
from ._layer_core_mapped_replay import replay_mapped_root_associations
from ._result import CellMeshingResult
from .providers._native_sources import NativeLayerCoreSource

if TYPE_CHECKING:
    from ._association import _LayerImageSupportPreparation

type _SemanticKey = tuple[str, str, GeometryAssociationKind, int, int]
type _SemanticRow = tuple[
    str,
    GeometrySourceEntityRole | None,
    tuple[str, ...],
    tuple[str | None, str | None],
    tuple[Fraction, Fraction] | None,
    Fraction | None,
    int,
    int,
    tuple[float | None, ...],
    int,
    float,
    bool,
    bool,
    tuple[Fraction, Fraction] | None,
    SourceBoundaryTargetPieceRow | None,
    str | None,
    int,
    int,
    GeometryAssociationProvenance,
    str | None,
    bool,
]


def _semantic_rows(
    mesh: CellMesh, records: tuple[GeometryAssociation, ...], /
) -> dict[_SemanticKey, tuple[_SemanticRow, ...]]:
    """Normalize every retained row without discarding same-dimension tables."""
    groups: dict[_SemanticKey, list[_SemanticRow]] = defaultdict(list)
    for association in records:
        dimension = target_dimension(mesh, association)
        association.validate_target(mesh.entity_set(dimension))
        ids = np.asarray(association.target_global_ids, dtype=np.int64)
        dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
        indices = np.asarray(association.source_indices, dtype=np.int64)
        residuals = np.asarray(association.residuals, dtype=np.float64)
        parameters = np.asarray(association.parameters, dtype=np.float64)
        orientations = np.asarray(association.orientations, dtype=np.int8)
        resolved = np.asarray(association.resolved, dtype=np.bool_)
        ambiguous = np.asarray(association.ambiguous, dtype=np.bool_)
        parent_dimensions = np.asarray(association.parent_dimensions, dtype=np.int64)
        parent_ids = np.asarray(association.parent_ids, dtype=np.int64)
        for row, identifier in enumerate(ids.tolist()):
            key = (
                association.source_id,
                association.source_revision,
                association.association_kind,
                dimension,
                identifier,
            )
            parameter = tuple(
                None if np.isnan(value) else float(value) for value in parameters[row]
            )
            groups[key].append(
                (
                    association.source_entity_ids[row],
                    association.source_roles[row],
                    association.source_occurrence_paths[row],
                    association.source_region_sides[row],
                    association.source_parameter_intervals[row],
                    association.source_parameter_points[row],
                    int(dimensions[row]),
                    int(indices[row]),
                    parameter,
                    int(orientations[row]),
                    float(residuals[row]),
                    bool(resolved[row]),
                    bool(ambiguous[row]),
                    association.target_parameter_intervals[row],
                    association.target_reference_pieces[row],
                    association.source_boundary_witness_id,
                    int(parent_dimensions[row]),
                    int(parent_ids[row]),
                    association.provenance,
                    association.parent_association_id,
                    association.exact,
                )
            )
    return {key: tuple(sorted(values, key=repr)) for key, values in groups.items()}


def _source_face_witness(
    source: CellMeshingResult, /
) -> SourceBoundaryChartWitness | SourceBoundaryFaceImageWitness:
    certification = source.certification
    if certification is None:
        raise ValueError(
            "Layer/core association transfer requires certified source authority."
        )
    witness = certification.inputs.fidelity_witness
    if not isinstance(
        witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
    ):
        raise ValueError(
            "Layer/core association transfer requires an exact source face witness."
        )
    return witness


def _merge_face_tables(
    mesh: CellMesh, tables: tuple[GeometryAssociation, ...], /
) -> tuple[GeometryAssociation, ...]:
    if not tables:
        return ()
    rows = _semantic_rows(mesh, tables)
    first = tables[0]
    selected: dict[
        tuple[int, str, GeometrySourceEntityRole | None, tuple[str, ...]], _SemanticRow
    ] = {}
    for key, payloads in rows.items():
        if key[:4] != (first.source_id, first.source_revision, first.association_kind, 2):
            raise ValueError("Core face tables bind incompatible source namespaces.")
        for payload in payloads:
            identity = (int(key[4]), payload[0], payload[1], payload[2])
            previous = selected.get(identity)
            if previous is not None:
                if previous[:14] != payload[:14] or previous[16:] != payload[16:]:
                    raise ValueError(
                        "Core source face pieces disagree on side or orientation."
                    )
                if (
                    previous[14] is not None
                    and payload[14] is not None
                    and previous[14:] != payload[14:]
                ):
                    raise ValueError(
                        "Core source face pieces bind conflicting source witnesses."
                    )
                if previous[14] is not None:
                    continue
            selected[identity] = payload
    groups: dict[
        tuple[
            str,
            GeometrySourceEntityRole | None,
            tuple[str, ...],
            str | None,
            GeometryAssociationProvenance,
            str | None,
            bool,
        ],
        dict[int, _SemanticRow],
    ] = {}
    for identity, payload in selected.items():
        group = (
            payload[0],
            payload[1],
            payload[2],
            payload[15],
            payload[18],
            payload[19],
            payload[20],
        )
        groups.setdefault(group, {})[identity[0]] = payload
    output = []
    for _, by_face in sorted(groups.items(), key=repr):
        ordered = sorted(by_face.items())
        identifiers = np.asarray([item[0] for item in ordered], dtype=np.int64)
        values = tuple(item[1] for item in ordered)
        output.append(
            GeometryAssociation(
                first.association_kind,
                first.source_id,
                first.source_revision,
                mesh.entity_set(2).entity_set_id,
                identifiers,
                tuple(value[0] for value in values),
                np.asarray([value[10] for value in values], dtype=np.float64),
                exact=values[0][20],
                source_roles=tuple(value[1] for value in values),
                source_occurrence_paths=tuple(value[2] for value in values),
                source_region_sides=tuple(value[3] for value in values),
                orientations=np.asarray([value[9] for value in values], dtype=np.int8),
                target_reference_pieces=tuple(value[14] for value in values),
                source_boundary_witness_id=values[0][15],
                parent_dimensions=np.asarray(
                    [value[16] for value in values], dtype=np.int8
                ),
                parent_ids=np.asarray([value[17] for value in values], dtype=np.int64),
                parent_association_id=values[0][19],
                provenance=values[0][18],
            )
        )
    return tuple(output)


def _mapped_endpoint_regions(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    table: GeometryAssociation,
    /,
) -> tuple[int, ...]:
    """Decode the replayed material table without inferring a physical region."""
    from ._layer_core_association import composite_region_identity

    authored = source.reference_source
    if authored is None:
        raise ValueError("Mapped region replay requires its original reference source.")
    keys = tuple(
        (composite_region_identity(authored, index), tuple(path))
        for index, path in enumerate(authored.region_occurrence_paths)
    )
    if len(set(keys)) != len(keys):
        raise ValueError("Authored mapped material identities are ambiguous.")
    by_key = {key: index for index, key in enumerate(keys)}
    table.validate_target(mesh.entity_set(3))
    ids = np.asarray(table.target_global_ids, dtype=np.int64)
    expected = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    if (
        (table.source_id, table.source_revision)
        != (authored.source_id, authored.source_revision)
        or ids.shape != expected.shape
        or not np.array_equal(ids, expected)
        or any(role is not GeometrySourceEntityRole.REGION for role in table.source_roles)
    ):
        raise ValueError(
            "Mapped material replay does not cover the actual endpoint cells."
        )
    try:
        return tuple(
            by_key[(identifier, tuple(path))]
            for identifier, path in zip(
                table.source_entity_ids, table.source_occurrence_paths, strict=True
            )
        )
    except KeyError as error:
        raise ValueError(
            "Mapped material replay names an unknown source occurrence."
        ) from error


@final
class NativeLayerCoreAssociationTransfer(StrictModule, NonTrainableState):
    """Replay mixed-volume source associations against original authoring inputs.

    Bind the actual layer/core source and the accepted volume request. Native
    mixed refinement and complete-partition coarsening require endpoint-bound
    geometry transitions, current source face witnesses and the accepted
    native carrier inventory; conforming caps also require the original root
    geometry and a separately proved scoped cap witness.
    """

    source: NativeLayerCoreSource
    specification: VolumeMeshingSpec
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self, source: NativeLayerCoreSource, specification: VolumeMeshingSpec, /
    ) -> None:
        if type(source) is not NativeLayerCoreSource:
            raise TypeError("source must be NativeLayerCoreSource.")
        if type(specification) is not VolumeMeshingSpec:
            raise TypeError("specification must be VolumeMeshingSpec.")
        require_layer_core_specification(source, specification)
        source.validate_restored()
        query = source.fidelity_source
        if query is None:
            raise ValueError(
                "Layer/core association transfer requires its original source query."
            )
        if not isinstance(query, MappedReferenceBoundarySource):
            require_current = getattr(query, "require_current", None)
            if not callable(require_current):
                raise TypeError(
                    "Layer/core source query must support currentness validation."
                )
            require_current()
            if source.source_boundary_charts is None:
                from ._layer_core_topology import validate_staged_layer_domain

                validate_staged_layer_domain(source)
            else:
                source.source_boundary_charts.require_current()
        if source.source_wall_charts is not None:
            source.source_wall_charts.require_current()
        self.source = source
        self.specification = specification
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "native-layer-core-association-transfer",
                "source": source.binding_id,
                "specification": specification.specification_id,
            }
        )

    @property
    def source_namespaces(self) -> tuple[tuple[str, str], ...]:
        authored = (
            self.source
            if self.source.reference_source is None
            else self.source.reference_source
        )
        namespaces = {
            (authored.source_id, authored.source_revision),
            (authored.layers.result_id, authored.layers.result_id),
        }
        wall = authored.layers.wall_association
        if wall is not None:
            namespaces.add((wall.source_id, wall.source_revision))
        if self.source.mapped_domain is not None:
            namespaces.add(
                (
                    self.source.mapped_domain.source_id,
                    self.source.mapped_domain.source_revision,
                )
            )
        return tuple(sorted(namespaces))

    def _endpoint(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        witness: SourceBoundaryChartWitness | SourceBoundaryFaceImageWitness,
        region_table: GeometryAssociation,
        carrier: NativeLayerCoreCarrierWitness,
        root: LayerCoreRootGeometry | None,
        scoped_fidelity: tuple[ScopedFidelityInput, ...],
        validated_fidelity: tuple[ValidatedSourceFidelityWitness, ...],
        /,
        *,
        computed_root: ComputedLayerCoreRoot | None = None,
        image_preparation: _LayerImageSupportPreparation | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        from ._association import (
            _LayerImageSupportPreparation,
            _prepare_layer_image_support,
        )
        from ._layer_core_strata_replay import _replay_core_strata

        query = self.source.fidelity_source
        if query is None:
            raise ValueError("The original layer/core source query is absent.")
        primary_token = validate_source_fidelity_witness(
            mesh,
            geometry,
            query,
            validated_fidelity_witness(
                validated_fidelity, mesh, geometry, query, witness
            ),
        )
        validated_scoped = tuple(
            token
            for _, token, _ in _validate_scoped_fidelity(
                mesh, geometry, scoped_fidelity, validated_fidelity
            )
        )
        cap_coverage_proved = self.source.cap_refinement == "conforming"
        if cap_coverage_proved and self.source.mapped_domain is None:
            require_scoped_cap_coverage(mesh, geometry, validated_scoped, self.source)
        source_facet_keys = core_source_facet_keys(self.source)
        contexts: list[
            tuple[
                ValidatedSourceFidelityWitness,
                SourceBoundaryChartWitness | SourceBoundaryFaceImageWitness,
            ]
        ] = [(primary_token, witness)]
        for token in validated_scoped:
            if not isinstance(
                token.witness,
                (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness),
            ) or (token.source.source_id, token.source.source_revision) != (
                self.source.source_id,
                self.source.source_revision,
            ):
                continue
            keys = set(
                zip(
                    token.witness.source_entity_ids,
                    token.witness.occurrence_paths,
                    strict=True,
                )
            )
            if (
                keys & source_facet_keys
                and token.witness.witness_id != witness.witness_id
            ):
                scoped_witness = _require_layer_fidelity_token(
                    mesh, geometry, token, self.source
                )
                assert isinstance(
                    scoped_witness,
                    (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness),
                )
                contexts.append((token, scoped_witness))
        if root is not None and computed_root is None:
            computed_root = ComputedLayerCoreRoot.reprove(
                self.source, self.specification, root
            )
        layer = replay_layer_source_associations(
            self.source, mesh, geometry, root, computed_root=computed_root
        )
        cap_owner = mapped_cap_edge_owner(
            self.source, mesh, geometry, layer, validated_scoped
        )
        if self.source.conforming_execution and self.source.mapped_domain is not None:
            if root is None:
                raise ValueError("Mapped conforming replay requires its original root.")
            regions = _mapped_endpoint_regions(self.source, mesh, region_table)
            if (
                computed_root is not None
                and mesh is root.mesh
                and geometry is root.geometry
            ):
                # The private token is tied to these exact operands and its
                # root constructor has already replayed the mapped controls.
                computed_root.require(
                    self.source, self.specification, mesh, geometry, root
                )
                if regions != root.reference_cell_regions:
                    raise ValueError(
                        "Mapped root material replay differs from its reference proof."
                    )
            else:
                from ._layer_core_mapped import _reprove_mapped_layer_endpoint

                _reprove_mapped_layer_endpoint(
                    self.source,
                    self.specification,
                    root,
                    mesh,
                    geometry,
                    regions,
                    layer,
                    computed_root=computed_root,
                )
        if image_preparation is None:
            image_preparation = _prepare_layer_image_support(
                self.source, mesh, geometry, layer.core_cell_ids,
                root=root, computed_root=computed_root,
            )
        elif type(image_preparation) is _LayerImageSupportPreparation:
            image_preparation.require(
                self.source, mesh, geometry, layer.core_cell_ids,
                root=root, computed_root=computed_root,
            )
        else:
            raise TypeError(
                "image_preparation must be a call-local layer support preparation."
            )
        core_ids = set(layer.core_cell_ids)
        core_faces = core_adjacent_face_ids(mesh, core_ids)
        eligible = frozenset(
            int(face_id)
            for cell_id, face_id in zip(
                witness.cell_global_ids, witness.entity_global_ids, strict=True
            )
            if int(cell_id) in core_ids
        )
        collected_faces = list(
            replay_core_boundary_faces(
                self.source, mesh, geometry, primary_token, eligible
            )
        )
        for scoped_token, scoped_witness in contexts[1:]:
            if is_internal_core_chart(scoped_token):
                continue
            scoped_ids = frozenset(
                int(face_id)
                for face_id, identity, occurrence in zip(
                    scoped_witness.entity_global_ids,
                    scoped_witness.source_entity_ids,
                    scoped_witness.occurrence_paths,
                    strict=True,
                )
                if int(face_id) in core_faces
                and (identity, occurrence) in source_facet_keys
            )
            collected_faces.extend(
                replay_core_boundary_faces(
                    self.source, mesh, geometry, scoped_token, scoped_ids
                )
            )
        cap_faces = replay_core_interface_faces(
            self.source, mesh, layer, require_coverage=not cap_coverage_proved
        )
        face_tables = _merge_face_tables(
            mesh, (*collected_faces, *((cap_faces,) if cap_faces is not None else ()))
        )
        chart_pieces = {
            item.witness_id: (
                source_boundary_image_association_piece_rows(item)
                if isinstance(item, SourceBoundaryFaceImageWitness)
                else source_boundary_association_piece_rows(item)
            )
            for _, item in contexts
        }
        for table in face_tables:
            if table.source_boundary_witness_id is None:
                continue
            if table.source_boundary_witness_id not in chart_pieces or (
                table.source_id,
                table.source_revision,
            ) != (self.source.source_id, self.source.source_revision):
                raise ValueError("Core face support changes its source witness identity.")
            for identifier, identity, occurrence, pieces in zip(
                table.target_global_ids,
                table.source_entity_ids,
                table.source_occurrence_paths,
                table.target_reference_pieces,
                strict=True,
            ):
                if (
                    chart_pieces[table.source_boundary_witness_id].get(
                        (int(identifier), identity, occurrence)
                    )
                    != pieces
                ):
                    raise ValueError(
                        "Core face support differs from its exact source chart."
                    )
        core_records = _replay_core_strata(
            self.source,
            self.specification,
            mesh,
            geometry,
            primary_token,
            layer,
            face_tables,
            region_table,
            carrier,
            validated_scoped,
            root,
            image_preparation,
            computed_root=computed_root,
        )
        facet_edges = tuple(
            association
            for index, (chart_token, chart_witness) in enumerate(contexts)
            if not is_internal_core_chart(chart_token)
            for association in replay_core_facet_edges(
                self.source,
                mesh,
                geometry,
                chart_witness,
                layer,
                face_tables,
                preparation=image_preparation,
                include_cap=index == 0 and not cap_coverage_proved,
                cap_coverage_proved=cap_coverage_proved,
                cap_owner=cap_owner,
            )
        )
        edge_overlays = replay_source_edge_face_overlays(
            self.source,
            mesh,
            geometry,
            tuple(token for token, _ in contexts),
            core_records,
            carrier.original_source_edge_rows,
            root,
            computed_root=computed_root,
        )
        records = [
            *layer.associations,
            *face_tables,
            *core_records,
            *facet_edges,
            *edge_overlays,
        ]
        layer_regions = replay_layer_region_vertices(
            self.source, mesh, layer, region_table, core_records
        )
        if layer_regions is not None:
            records.append(layer_regions)
        authored = (
            self.source
            if self.source.reference_source is None
            else self.source.reference_source
        )
        if authored.layers.wall_association is not None:
            from ..geometry._domain_boundary import MeshingDomainBoundarySource
            from ..geometry._mesh_certificates import _same_tree_content
            from ._layer_curving import layer_wall_boundary_query
            from ._layer_wall_source_replay import replay_layer_wall_associations

            wall_query = (
                self.source.fidelity_source
                if authored.source_wall_charts is None
                else layer_wall_boundary_query(authored.layers, authored.source_wall_charts)
            )
            wall_tokens = tuple(
                token
                for token in (primary_token, *validated_scoped)
                if isinstance(token.witness, SourceBoundaryChartWitness)
                and wall_query is not None
                and (
                    token.source is wall_query
                    or (
                        isinstance(token.source, MeshingDomainBoundarySource)
                        and isinstance(wall_query, MeshingDomainBoundarySource)
                        and token.source.source_scope_id == wall_query.source_scope_id
                        and token.witness.source_query_id == wall_query.source_scope_id
                        and _same_tree_content(token.source, wall_query)
                    )
                )
                and token.binds(mesh, geometry, token.source)
            )
            if not wall_tokens:
                raise ValueError(
                    "Layer wall replay requires its independently validated source chart."
                )
            wall_witness = wall_tokens[0].witness
            assert isinstance(wall_witness, SourceBoundaryChartWitness)
            records.extend(
                replay_layer_wall_associations(
                    self.source,
                    mesh,
                    geometry,
                    wall_witness,
                    layer,
                    self.specification.maximum_deviation,
                )
            )
        domain = self.source.mapped_domain
        if domain is not None:
            records.extend(
                replay_mapped_root_associations(
                    domain, mesh, geometry,
                    source=self.source,
                    specification=self.specification,
                    root=root,
                    computed_root=computed_root,
                )
            )
        return tuple(records)

    def validate_source(self, source: CellMeshingResult, /) -> None:
        self._validate_source(source, computed_root=None)

    def _validate_source(
        self,
        source: CellMeshingResult,
        /,
        *,
        computed_root: ComputedLayerCoreRoot | None,
    ) -> None:
        from ._adaptation import _identity_lineage

        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        self.source.validate_restored()
        require_layer_core_specification(self.source, self.specification)
        query = self.source.fidelity_source
        if query is None:
            raise ValueError("The original layer/core source query is absent.")
        if not isinstance(query, MappedReferenceBoundarySource):
            require_current = getattr(query, "require_current", None)
            if not callable(require_current):
                raise TypeError(
                    "Layer/core source query must support currentness validation."
                )
            require_current()
        witness = _source_face_witness(source)
        carrier = source.layer_core_carrier_witness
        if carrier is None:
            raise ValueError(
                "Layer/core source lacks its accepted native carrier inventory."
            )
        carrier.require_current(self.source, self.specification)
        if self.source.conforming_execution:
            stencils = source.layer_core_overlap_stencils
            columns = source.layer_core_column_certification
            root = source.layer_core_root_geometry
            if root is None:
                raise ValueError(
                    "Conforming layer/core source lacks its original root proof."
                )
            if self.source.mapped_domain is not None:
                if stencils is not None or columns is not None:
                    raise ValueError(
                        "Mapped physical results cannot carry reference-only stencil or column evidence."
                    )
                # The endpoint replay below independently proves mapped
                # physical controls, source/periodic ancestry and coverage.
            else:
                if stencils is None or columns is None:
                    raise ValueError(
                        "Conforming layer/core source lacks its original stencil and column proofs."
                    )
                if (
                    stencils.source_binding_id != self.source.binding_id
                    or columns.source_binding_id != self.source.binding_id
                    or columns.specification_id != self.specification.specification_id
                    or root.overlap_stencils.witness_id != stencils.witness_id
                ):
                    raise ValueError(
                        "Conforming layer/core evidence binds another source or request."
                    )
                from ._layer_core_subdivision import (
                    require_descendant_overlap_source_provenance,
                    require_refined_column_certification,
                )

            if source.mesh.mesh_id == root.mesh.mesh_id:
                if cell_geometry_id(source.geometry) != cell_geometry_id(root.geometry):
                    raise ValueError(
                        "Conforming root differs from its retained source proof."
                    )
                if self.source.mapped_domain is None and (
                    stencils != root.overlap_stencils
                    or columns != root.column_certification
                ):
                    raise ValueError(
                        "Conforming root differs from its retained source proof."
                    )
            else:
                restriction = source.geometry.restriction_source
                if restriction is None or (
                    restriction.source_topology_id != root.mesh.topology_id
                    or restriction.source_geometry_id != cell_geometry_id(root.geometry)
                ):
                    raise ValueError(
                        "Conforming descendant lacks its exact original-root map."
                    )
                if self.source.mapped_domain is None:
                    if columns is None:
                        raise ValueError("Unmapped descendant lacks its column evidence.")
                    require_descendant_overlap_source_provenance(self.source, source)
                    require_refined_column_certification(
                        self.source,
                        source.mesh,
                        source.geometry,
                        self.specification,
                        columns,
                    )
        regions, _, validated_source = replay_composite_regions(
            self.source, source, source.mesh, _identity_lineage(source.mesh, source.mesh)
        )
        assert source.certification is not None
        expected = list(
            self._endpoint(
                source.mesh,
                source.geometry,
                witness,
                regions,
                carrier,
                source.layer_core_root_geometry,
                source.certification.inputs.scoped_fidelity,
                validated_source,
                computed_root=computed_root,
            )
        )
        expected.append(regions)
        namespaces = set(self.source_namespaces)
        source_tables = tuple(
            table
            for table in source.associations
            if (table.source_id, table.source_revision) in namespaces
        )
        for table in source_tables:
            GeometryAssociationArchiveRecipe.from_association(table).restore()
        actual_rows = _semantic_rows(source.mesh, source_tables)
        expected_rows = _semantic_rows(source.mesh, tuple(expected))
        if actual_rows != expected_rows:
            raise ValueError(
                "Layer/core association rows differ from independently replayed source strata."
            )

    def classes(
        self, source: CellMeshingResult, /
    ) -> tuple[AssociationEntityClasses, ...]:
        return self._classes(source, computed_root=None)

    def _classes(
        self,
        source: CellMeshingResult,
        /,
        *,
        computed_root: ComputedLayerCoreRoot | None,
    ) -> tuple[AssociationEntityClasses, ...]:
        if computed_root is not None:
            root = source.layer_core_root_geometry
            if type(computed_root) is not ComputedLayerCoreRoot or root is None:
                raise ValueError("Layer association classes require their actual computed root.")
            computed_root.require(
                self.source, self.specification, root.mesh, root.geometry, root
            )
        self._validate_source(source, computed_root=computed_root)
        role_dimensions = {
            GeometrySourceEntityRole.VERTEX: 0,
            GeometrySourceEntityRole.EDGE: 1,
            GeometrySourceEntityRole.FACET: 2,
            GeometrySourceEntityRole.REGION: 3,
        }
        namespaces = set(self.source_namespaces)
        records = _semantic_rows(
            source.mesh,
            tuple(
                table
                for table in source.associations
                if (table.source_id, table.source_revision) in namespaces
            ),
        )
        result = []
        for dimension in range(4):
            ids = np.asarray(source.mesh.entity_set(dimension).entity_ids, dtype=np.int64)
            positions = {int(identifier): row for row, identifier in enumerate(ids)}
            keys: list[list[tuple[object, ...]]] = [[] for _ in ids]
            dims = np.full(ids.shape, -1, dtype=np.int64)
            indices = np.full(ids.shape, -1, dtype=np.int64)
            resolved = np.zeros(ids.shape, dtype=np.bool_)
            partial_edges: set[int] = set()
            partial_facets: dict[int, list[tuple[_SemanticKey, _SemanticRow]]] = (
                defaultdict(list)
            )
            for key, payloads in records.items():
                if key[3] != dimension:
                    continue
                row = positions[int(key[4])]
                for payload in payloads:
                    if dimension == 1 and payload[13] is not None:
                        partial_edges.add(row)
                        if payload[1] is GeometrySourceEntityRole.FACET:
                            partial_facets[row].append((key, payload))
                        continue
                    if (
                        dimension == 2
                        and payload[1] is GeometrySourceEntityRole.EDGE
                        and payload[14] is not None
                    ):
                        continue
                    declared = int(payload[6])
                    source_dimension = (
                        declared if declared >= 0 else role_dimensions.get(payload[1], -1)
                    )
                    if source_dimension < 0 or not payload[11] or payload[12]:
                        raise ValueError(
                            "Layer/core association contains an unresolved source class."
                        )
                    if dims[row] < 0 or source_dimension < dims[row]:
                        dims[row], indices[row] = source_dimension, int(payload[7])
                    resolved[row] = True
                    keys[row].append(
                        (
                            key[0],
                            key[1],
                            key[2],
                            payload[0],
                            payload[1],
                            payload[2],
                            payload[3],
                        )
                    )
            if dimension == 1:
                face_ids = np.asarray(
                    source.mesh.entity_set(2).entity_ids, dtype=np.int64
                )
                associated_faces = {
                    int(key[4])
                    for key, payloads in records.items()
                    if key[3] == 2
                    and any(
                        payload[1] is not GeometrySourceEntityRole.EDGE
                        or payload[14] is None
                        for payload in payloads
                    )
                }
                face_incidence = incidence_pairs(source.mesh, 1, 2)
                cell_incidence = incidence_pairs(source.mesh, 1, 3)
                cell_ids = np.asarray(
                    source.mesh.entity_set(3).entity_ids, dtype=np.int64
                )
                region_by_cell: dict[int, set[tuple[object, ...]]] = defaultdict(set)
                for key, payloads in records.items():
                    if key[3] != 3:
                        continue
                    for payload in payloads:
                        if (
                            payload[1] is GeometrySourceEntityRole.REGION
                            and int(payload[6]) < 0
                        ):
                            region_by_cell[int(key[4])].add(
                                (key[0], key[1], payload[0], payload[2])
                            )
                boundary_edges = {
                    int(edge)
                    for edge, face in face_incidence
                    if int(face_ids[face]) in associated_faces
                }
                edge_regions: dict[int, set[tuple[object, ...]]] = defaultdict(set)
                invalid_edges: set[int] = set()
                for edge, cell in cell_incidence:
                    values = region_by_cell.get(int(cell_ids[cell]), set())
                    if len(values) != 1:
                        invalid_edges.add(int(edge))
                    edge_regions[int(edge)].update(values)
                for row in np.flatnonzero(~resolved).tolist():
                    regions = edge_regions.get(row, set())
                    if row in invalid_edges or not regions:
                        continue
                    if row in partial_edges:
                        parts = partial_facets.get(row, [])
                        ordered = sorted(parts, key=lambda value: value[1][13] or ())
                        cursor = Fraction(0)
                        region_ids = {value[2] for value in regions}
                        for _, payload in ordered:
                            interval = payload[13]
                            if (
                                interval is None
                                or interval[0] != cursor
                                or {side for side in payload[3] if side is not None}
                                != region_ids
                            ):
                                break
                            cursor = interval[1]
                        else:
                            if ordered and cursor == Fraction(1):
                                dims[row] = 2
                                resolved[row] = True
                                keys[row].append(
                                    (
                                        "exact-facet-cover",
                                        tuple(
                                            (
                                                key[0],
                                                key[1],
                                                payload[0],
                                                payload[2],
                                                payload[3],
                                                payload[13],
                                            )
                                            for key, payload in ordered
                                        ),
                                    )
                                )
                        continue
                    if row in boundary_edges or len(regions) != 1:
                        continue
                    dims[row] = 3
                    resolved[row] = True
                    keys[row].append(next(iter(regions)))
            canonical = [tuple(sorted(value, key=repr)) for value in keys]
            codebook = {
                key: index for index, key in enumerate(sorted(set(canonical), key=repr))
            }
            codes = np.asarray(
                [codebook[key] if key else -1 for key in canonical], dtype=np.int64
            )
            result.append(AssociationEntityClasses(dims, indices, resolved, codes))
        return tuple(result)

    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        vertex_stencil: VertexInterpolationStencil | None,
        geometry_transition: CellGeometryTransition | None,
        target_fidelity_witness: SourceBoundaryChartWitness
        | SourceBoundaryFaceImageWitness
        | None,
        target_scoped_fidelity: tuple[ScopedFidelityInput, ...],
        validated_fidelity: tuple[ValidatedSourceFidelityWitness, ...],
    ) -> tuple[GeometryAssociation, ...]:
        return self._propagate_computed(
            source,
            lineage,
            target,
            vertex_stencil=vertex_stencil,
            geometry_transition=geometry_transition,
            target_fidelity_witness=target_fidelity_witness,
            target_scoped_fidelity=target_scoped_fidelity,
            validated_fidelity=validated_fidelity,
            computed_root=None,
        )

    def _propagate_computed(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        vertex_stencil: VertexInterpolationStencil | None,
        geometry_transition: CellGeometryTransition | None,
        target_fidelity_witness: SourceBoundaryChartWitness
        | SourceBoundaryFaceImageWitness
        | None,
        target_scoped_fidelity: tuple[ScopedFidelityInput, ...],
        validated_fidelity: tuple[ValidatedSourceFidelityWitness, ...],
        computed_root: ComputedLayerCoreRoot | None,
    ) -> tuple[GeometryAssociation, ...]:
        root = source.layer_core_root_geometry
        if root is not None:
            if computed_root is None:
                computed_root = ComputedLayerCoreRoot.reprove(
                    self.source, self.specification, root
                )
            else:
                computed_root.require(
                    self.source, self.specification, root.mesh, root.geometry, root
                )
        elif computed_root is not None:
            raise ValueError("A computed root requires its actual retained root.")
        self._validate_source(source, computed_root=computed_root)
        if geometry_transition is None or target_fidelity_witness is None:
            raise ValueError(
                "Layer/core transfer requires the exact transition and renewed source face witness."
            )
        carrier = source.layer_core_carrier_witness
        if carrier is None:
            raise ValueError(
                "Layer/core source lacks its accepted native carrier inventory."
            )
        geometry_transition.require_bound(
            source.mesh, source.geometry, target, geometry_transition.target_geometry
        )
        require_transfer_lineage(source.mesh, lineage, target, dimensions=(0, 1, 2, 3))
        _, regions, _ = replay_composite_regions(self.source, source, target, lineage)
        records = list(
            self._endpoint(
                target,
                geometry_transition.target_geometry,
                target_fidelity_witness,
                regions,
                carrier,
                source.layer_core_root_geometry,
                target_scoped_fidelity,
                validated_fidelity,
                computed_root=computed_root,
            )
        )
        records.append(regions)
        return tuple(
            sorted(
                records,
                key=lambda value: (
                    value.source_id,
                    value.source_revision,
                    value.target_entity_set_id,
                    value.association_id,
                ),
            )
        )


__all__ = ["NativeLayerCoreAssociationTransfer"]
