#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source- and request-bound native core carrier inventory."""

from __future__ import annotations

from typing import final, TYPE_CHECKING

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


if TYPE_CHECKING:
    from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
    from ._association import _LayerImageSupportPreparation
    from ._contracts import VolumeMeshingSpec
    from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
    from ._layer_core_topology import LayerCoreConstruction
    from ._plc_source import PiecewiseLinearComplex
    from ._volume_generation import VolumeConstruction
    from .providers._native_sources import NativeLayerCoreSource


def require_layer_core_specification(
    source: NativeLayerCoreSource, specification: VolumeMeshingSpec, /
) -> None:
    """Revalidate the request against the retained B-Rep authority when present."""
    from ..geometry._meshing_domain import MeshingDomain
    from ._contracts import VolumeMeshingSpec

    domain = source.source_domain
    model = domain.brep_authority if isinstance(domain, MeshingDomain) else None
    VolumeMeshingSpec.from_stored_fields(specification, source_model=model)


def _native_recovered_rows(
    core: VolumeConstruction, execution: PiecewiseLinearComplex, /
) -> tuple[int, ...]:
    """Verify native carrier rows against the actual PLC passed to recovery."""
    active = np.asarray(core.source_segment_complex_rows, dtype=np.int64)
    raw_to_active = np.asarray(core.plc_edge_source_rows, dtype=np.int64)
    raw_edges = np.asarray(core.plc_edges, dtype=np.int64)
    expected_edges = np.asarray(execution.plc_edge_vertices, dtype=np.int64).reshape(
        (-1, 2)
    )
    if (
        core.complex_id != execution.complex_id
        or active.ndim != 1
        or np.unique(active).size != active.size
        or np.any(active < 0)
        or np.any(active >= expected_edges.shape[0])
        or raw_edges.shape != (active.size, 2)
        or raw_to_active.shape != (active.size,)
        or not np.array_equal(
            np.sort(raw_to_active), np.arange(active.size, dtype=np.int64)
        )
    ):
        raise ValueError("Native recovery carriers do not bind their execution PLC.")
    for raw_row, active_row in enumerate(raw_to_active.tolist()):
        if set(raw_edges[raw_row].tolist()) != set(
            expected_edges[active[active_row]].tolist()
        ):
            raise ValueError(
                "Native recovery carrier changed its execution PLC endpoints."
            )
    segment_sources = np.asarray(core.segment_source_edges, dtype=np.int64)
    segments = np.asarray(core.segments, dtype=np.int64)
    if (
        segments.ndim != 2
        or segments.shape[1] != 2
        or segment_sources.shape != (segments.shape[0],)
        or np.any(segment_sources < 0)
        or np.any(segment_sources >= active.size)
        or set(segment_sources.tolist()) != set(range(active.size))
    ):
        raise ValueError("Native recovery did not retain every execution carrier chain.")
    return tuple(sorted(int(value) for value in active))


def _conforming_recovered_source_rows(
    source: NativeLayerCoreSource,
    construction: LayerCoreConstruction,
    primary: ValidatedSourceFidelityWitness,
    scoped: tuple[ValidatedSourceFidelityWitness, ...],
    root: LayerCoreRootGeometry,
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
    preparation: _LayerImageSupportPreparation | None = None,
) -> tuple[int, ...]:
    """Map actual recovered target segments to proved original source edges."""
    from ..geometry._source_boundary_records import (
        SourceBoundaryChartWitness,
        SourceBoundaryFaceImageWitness,
    )
    from ._association import _prepare_layer_image_support
    from ._layer_core_boundary_replay import (
        _require_layer_fidelity_token,
        core_cap_source_keys,
        replay_core_boundary_faces,
        replay_core_interface_faces,
        require_scoped_cap_coverage,
    )
    from ._layer_core_strata_replay import (
        _source_face_support,
        incident_core_source_facets,
        mapped_internal_facet_carriers,
        match_core_source_edges,
    )
    from ._layer_core_subdivision import require_overlap_source_provenance
    from ._layer_source_association_replay import replay_layer_source_associations
    from ._topology_edit import entity_keys
    from ._volume_generation import VolumeConstruction

    mesh, geometry, core = construction.mesh, construction.geometry, construction.core
    if not isinstance(core, VolumeConstruction):
        raise TypeError("Conforming source carrier proof requires a native simplex core.")
    if source.mapped_domain is None:
        require_overlap_source_provenance(source, construction)
    elif (
        root.reference_mesh is not construction.reference_mesh
        or root.reference_geometry is not construction.reference_geometry
        or root.overlap_exact_geometry is not construction.overlap_exact_geometry
        or root.overlap_stencils is not construction.overlap_source_stencils
        or root.column_certification is not construction.layer_core_column_certification
        or root.layer_root_partition is not construction.layer_root_partition
    ):
        raise ValueError(
            "Mapped carrier construction changed its independently proved reference operands."
        )
    if source.cap_refinement == "conforming":
        require_scoped_cap_coverage(mesh, geometry, scoped, source)
    primary_witness = _require_layer_fidelity_token(mesh, geometry, primary, source)
    if not isinstance(
        primary_witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
    ):
        raise ValueError("Conforming carrier requires an exact source face witness.")
    layer = replay_layer_source_associations(
        source, mesh, geometry, root, computed_root=computed_root
    )
    if preparation is None:
        preparation = _prepare_layer_image_support(
            source,
            mesh,
            geometry,
            layer.core_cell_ids,
            root=root,
            computed_root=computed_root,
        )
    else:
        preparation.require(
            source,
            mesh,
            geometry,
            layer.core_cell_ids,
            root=root,
            computed_root=computed_root,
        )
    core_ids = set(layer.core_cell_ids)
    eligible = frozenset(
        int(face)
        for cell, face in zip(
            primary_witness.cell_global_ids,
            primary_witness.entity_global_ids,
            strict=True,
        )
        if int(cell) in core_ids
    )
    face_tables = list(
        replay_core_boundary_faces(source, mesh, geometry, primary, eligible)
    )
    chart_support = _source_face_support(mesh, geometry, primary_witness, preparation)
    cap_keys = core_cap_source_keys(source)
    for token in scoped:
        if not isinstance(
            token.witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        ):
            continue
        keys = set(
            zip(
                token.witness.source_entity_ids,
                token.witness.occurrence_paths,
                strict=True,
            )
        )
        if not keys & cap_keys:
            continue
        witness = _require_layer_fidelity_token(mesh, geometry, token, source)
        assert isinstance(
            witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        )
        cap_faces = frozenset(
            int(face)
            for face, identity, path in zip(
                witness.entity_global_ids,
                witness.source_entity_ids,
                witness.occurrence_paths,
                strict=True,
            )
            if (identity, path) in cap_keys
        )
        face_tables.extend(
            replay_core_boundary_faces(source, mesh, geometry, token, cap_faces)
        )
        for key, (vertices, edges) in _source_face_support(
            mesh, geometry, witness, preparation
        ).items():
            current = chart_support.setdefault(key, (set(), set()))
            current[0].update(vertices)
            current[1].update(edges)
    cap = replay_core_interface_faces(
        source, mesh, layer, require_coverage=source.cap_refinement != "conforming"
    )
    if cap is not None:
        face_tables.append(cap)
    carriers = incident_core_source_facets(
        source, mesh, layer, tuple(face_tables), chart_support
    )
    for edge, facets in mapped_internal_facet_carriers(
        source, mesh, geometry, layer, root, scoped, computed_root=computed_root
    ).items():
        carriers[edge] = carriers.get(edge, frozenset()) | facets
    matches = match_core_source_edges(source, preparation.images, carriers)
    vertex_map = np.asarray(construction.core_vertex_map, dtype=np.int64)
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    segments = np.asarray(core.segments, dtype=np.int64)
    if (
        vertex_map.shape != (core.mesh.coordinates.shape[0],)
        or np.any(vertex_map < 0)
        or np.any(vertex_map >= target_ids.size)
        or np.any(segments < 0)
        or np.any(segments >= vertex_map.size)
    ):
        raise ValueError(
            "Native recovered segment vertices lack exact combined-mesh ancestry."
        )
    edge_by_key = {
        tuple(int(value) for value in key if value >= 0): int(identifier)
        for key, identifier in zip(
            entity_keys(mesh, 1), mesh.entity_set(1).entity_ids, strict=True
        )
    }
    recovered: set[int] = set()
    for segment in segments:
        key = tuple(sorted(int(value) for value in target_ids[vertex_map[segment]]))
        edge_id = edge_by_key.get(key)
        if edge_id is None:
            raise ValueError(
                "A native recovered carrier is absent from the combined mesh."
            )
        recovered.update(source_row for source_row, _, _ in matches.get(edge_id, ()))
    return tuple(sorted(recovered))


@final
class NativeLayerCoreCarrierWitness(StrictModule, NonTrainableState):
    """Retain source-derived original and required native core edge inventories.

    Endpoint consumers independently reprove actual support and complete
    coverage. Native recovery findings remain operation-local evidence.
    """

    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    original_source_edge_rows: tuple[int, ...] = eqx.field(static=True)
    required_source_edge_rows: tuple[int, ...] = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        /,
    ) -> None:
        from ._contracts import VolumeMeshingSpec
        from ._layer_core_edge_policy import required_core_source_edge_rows
        from .providers._native_sources import NativeLayerCoreSource

        if (
            type(source) is not NativeLayerCoreSource
            or type(specification) is not VolumeMeshingSpec
        ):
            raise TypeError(
                "Carrier inventory requires the exact layer/core source and volume request."
            )
        source.validate_restored()
        require_layer_core_specification(source, specification)
        authored = source if source.reference_source is None else source.reference_source
        required_rows = required_core_source_edge_rows(source, specification)
        self.source_binding_id = source.binding_id
        self.specification_id = specification.specification_id
        self.original_source_edge_rows = tuple(range(len(authored.complex.plc_edge_ids)))
        self.required_source_edge_rows = required_rows
        self.witness_id = canonical_fingerprint(
            {
                "kind": "native-layer-core-carrier-witness",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "original_source_edge_rows": self.original_source_edge_rows,
                "required_source_edge_rows": required_rows,
            }
        )

    @classmethod
    def from_construction(
        cls,
        source: NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        construction: LayerCoreConstruction,
        primary: ValidatedSourceFidelityWitness,
        scoped: tuple[ValidatedSourceFidelityWitness, ...],
        root: LayerCoreRootGeometry | None,
        /,
        *,
        computed_root: ComputedLayerCoreRoot | None = None,
    ) -> NativeLayerCoreCarrierWitness:
        """Bind the native recovery carrier/source relation at construction."""
        return cls._from_construction(
            source,
            specification,
            construction,
            primary,
            scoped,
            root,
            computed_root=computed_root,
            preparation=None,
        )

    @classmethod
    def _from_construction(
        cls,
        source: NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        construction: LayerCoreConstruction,
        primary: ValidatedSourceFidelityWitness,
        scoped: tuple[ValidatedSourceFidelityWitness, ...],
        root: LayerCoreRootGeometry | None,
        /,
        *,
        computed_root: ComputedLayerCoreRoot | None,
        preparation: _LayerImageSupportPreparation | None,
    ) -> NativeLayerCoreCarrierWitness:
        """Private provider worker sharing only a live, operand-bound preparation."""
        from ._layer_core_execution import LayerCoreExecution
        from ._layer_core_topology import LayerCoreConstruction
        from ._volume_generation import VolumeConstruction

        if not isinstance(construction, LayerCoreConstruction):
            raise TypeError("Core carrier issuance requires layer/core construction.")
        core = construction.core
        if not isinstance(core, VolumeConstruction):
            raise TypeError(
                "Core carrier issuance requires native simplex volume construction."
            )
        authored = source if source.reference_source is None else source.reference_source
        if source.conforming_execution:
            execution = construction.execution_carrier
            if not isinstance(execution, LayerCoreExecution) or root is None:
                raise ValueError(
                    "Conforming carrier issuance requires its execution and root authorities."
                )
            _native_recovered_rows(core, execution.complex)
            active = _conforming_recovered_source_rows(
                source,
                construction,
                primary,
                scoped,
                root,
                computed_root=computed_root,
                preparation=preparation,
            )
        else:
            if computed_root is not None:
                raise ValueError(
                    "Fixed carrier construction cannot consume a conforming root proof."
                )
            if preparation is not None:
                raise ValueError(
                    "Fixed carrier construction cannot consume layer image preparation."
                )
            active = _native_recovered_rows(core, authored.complex)
        from ._layer_core_edge_policy import required_core_source_edge_rows

        if not set(required_core_source_edge_rows(source, specification)) <= set(active):
            raise ValueError("Native recovery omitted an enforced original source edge.")
        return cls(source, specification)

    def require_current(
        self, source: NativeLayerCoreSource, specification: VolumeMeshingSpec, /
    ) -> None:
        """Revalidate the source, request, inventory identity and authored row bounds."""
        rebuilt = type(self)(source, specification)
        if (
            self.source_binding_id != source.binding_id
            or self.specification_id != specification.specification_id
            or rebuilt.original_source_edge_rows != self.original_source_edge_rows
            or rebuilt.required_source_edge_rows != self.required_source_edge_rows
            or rebuilt.witness_id != self.witness_id
        ):
            raise ValueError("Native core carrier inventory is stale or inconsistent.")


__all__ = ["NativeLayerCoreCarrierWitness"]
