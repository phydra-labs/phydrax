#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native material volume routes through the canonical PLC fill."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import final

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._compartments import CompartmentMeshingSource
from ...geometry._source_identity import source_entity_id
from ...geometry.multiregion_surface import (
    LabelFieldSurfaceExtractionResult,
    LabelFieldSurfaceExtractionStatus,
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    MultiRegionSurfaceStatus,
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
)
from ...typing import checked
from .._compartments import CompartmentPlcPreparation, prepare_compartment_plc
from .._contracts import MeshingFailure, MeshingFailureCategory, VolumeMeshingSpec
from .._native_construction import NativeMeshConstruction
from .._plc_source import PiecewiseLinearComplex
from .._volume_generation import NativeVolumeSchedule
from ._native_options import NativeMeshingOptions
from ._native_sources import NativePlcSource
from ._native_volume import PreparedPlcVolume, volume_support_issues


@dataclass(frozen=True, slots=True)
class _LabelFieldPlcNormalization:
    source: NativePlcSource
    facet_source_region_ids: tuple[tuple[str, str], ...]
    boundary_region_id: str


def _label_field_failure(message: str, /) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.INVALID_SOURCE,
        message,
        provider_code="native_compartment_label_source",
        stage="source_inspection",
    )


def _source_region_key(
    source: CompartmentMeshingSource | NativePlcSource,
    region_id: str | None,
    /,
) -> tuple[str, tuple[str, ...]] | None:
    if region_id is None:
        return None
    matches = tuple(
        key
        for identifier in source.scope_entity_ids(3).tolist()
        if (key := source.source_entity_key_for_scope(3, int(identifier)))[0]
        == region_id
    )
    if len(matches) != 1:
        raise ValueError(
            "A source-adjacent region must resolve through exactly one source-owned scope key."
        )
    return matches[0]


def _normalize_label_field_source(
    result: LabelFieldSurfaceExtractionResult, /
) -> _LabelFieldPlcNormalization:
    """Bind accepted surface witnesses to an explicit, lineage-keyed PLC."""
    if not isinstance(result, LabelFieldSurfaceExtractionResult):
        raise TypeError("result must be a LabelFieldSurfaceExtractionResult.")
    seed = result.candidate_seed
    topology = result.topology
    state = result.state
    surface = result.surface
    lineage = result.lineage
    evidence = result.evidence
    validation = evidence.validation
    if (
        evidence.status is not LabelFieldSurfaceExtractionStatus.ACCEPTED
        or not evidence.accepted
        or not evidence.collision_certified
        or seed is None
        or topology is None
        or state is None
        or surface is None
        or validation is None
        or validation.status is not MultiRegionSurfaceStatus.ACCEPTED
        or not validation.accepted
        or not validation.self_intersection_checked
        or validation.intersecting_pair_count != 0
        or validation.uncertain_pair_count != 0
        or evidence.lineage_id != lineage.lineage_id
        or lineage.candidate_seed_id != seed.seed_id
        or topology.region_ids != lineage.region_ids
        or topology.region_ids != seed.region_ids
        or topology.region_kinds != seed.region_kinds
        or topology.epoch != lineage.source_epoch
        or validation.topology_id != topology.topology_id
        or validation.lineage_id != topology.lineage_id
        or surface.topology.topology_id != topology.topology_id
        or not surface.evidence.accepted
        or surface.evidence.evidence_id != validation.evidence_id
        or not isinstance(seed, MultiRegionSurfaceSeed)
        or not isinstance(topology, MultiRegionSurfaceTopology)
        or not isinstance(state, MultiRegionSurfaceState)
        or not isinstance(surface, PreparedMultiRegionSurface)
    ):
        raise _label_field_failure(
            "Label-field PLC input must retain one accepted collision-certified surface, "
            "state, topology, validation, and lineage."
        )
    try:
        state.require_topology(topology)
        seed_topology = seed.topology(topology.plan, epoch=topology.epoch)
    except (TypeError, ValueError) as error:
        raise _label_field_failure(
            "Label-field surface state and seed must match the certified topology."
        ) from error
    if (
        seed_topology.topology_id != topology.topology_id
        or seed_topology.lineage_id != topology.lineage_id
    ):
        raise _label_field_failure(
            "Label-field topology no longer matches its retained source seed."
        )

    positions = np.asarray(
        state.positions[: topology.vertex_count], dtype=np.float64
    )
    certified_seed_positions = np.asarray(
        seed.positions, dtype=np.dtype(topology.plan.coordinate_dtype)
    ).astype(np.float64)
    if not np.array_equal(positions, certified_seed_positions):
        raise _label_field_failure(
            "Label-field state positions differ from the collision-certified source seed."
        )
    geometry_id = canonical_fingerprint(
        {
            "kind": "multiregion-surface-geometry",
            "topology": topology.topology_id,
            "positions": array_tree_fingerprint(positions),
        }
    )
    if validation.geometry_id != geometry_id:
        raise _label_field_failure(
            "Label-field validation does not certify the retained source positions."
        )

    source_keys = seed.vertex_source_keys
    if source_keys is None or len(source_keys) != topology.vertex_count:
        raise _label_field_failure(
            "Label-field PLC normalization requires retained stable source keys for every vertex."
        )
    boundary_indices = tuple(
        index
        for index, kind in enumerate(topology.region_kinds)
        if kind == "boundary"
    )
    if (
        len(boundary_indices) != 1
        or topology.region_ids[boundary_indices[0]] != lineage.boundary_region_id
    ):
        raise _label_field_failure(
            "Label-field lineage must identify the topology's unique boundary region."
        )
    boundary_index = boundary_indices[0]
    boundary_source_index = (
        lineage.source_label_ids.index(lineage.boundary_region_id)
        if lineage.boundary_region_id in lineage.source_label_ids
        else -1
    )
    if (
        lineage.region_source_label_indices[boundary_index]
        != boundary_source_index
        or any(
            source_index < 0
            or source_index >= len(lineage.source_label_ids)
            or lineage.source_label_ids[source_index] != region_id
            for region_index, (region_id, source_index) in enumerate(
                zip(
                    topology.region_ids,
                    lineage.region_source_label_indices,
                    strict=True,
                )
            )
            if region_index != boundary_index
        )
    ):
        raise _label_field_failure(
            "Label-field source labels do not preserve their region lineage."
        )
    finite_region_ids = tuple(
        identifier
        for identifier, kind in zip(
            topology.region_ids, topology.region_kinds, strict=True
        )
        if kind == "finite"
    )
    if not finite_region_ids:
        raise _label_field_failure(
            "Label-field PLC normalization requires at least one finite material region."
        )

    source_id = canonical_fingerprint(
        {
            "kind": "accepted-label-field-source",
            "source_binding": lineage.source_binding_id,
            "source_route": lineage.source_route_id,
            "source_prepared": lineage.source_prepared_id,
            "extraction_prepared": lineage.prepared_id,
            "source_sites": lineage.source_site_id,
            "source_label_ids": lineage.source_label_ids,
            "region_ids": lineage.region_ids,
            "region_source_label_indices": lineage.region_source_label_indices,
        }
    )
    source_revision = canonical_fingerprint(
        {
            "kind": "accepted-label-field-surface-revision",
            "source": source_id,
            "source_state": lineage.source_state_id,
            "lineage": lineage.lineage_id,
            "topology": topology.topology_id,
            "topology_lineage": topology.lineage_id,
            "surface_validation": validation.evidence_id,
            "extraction_evidence": evidence.evidence_id,
            "surface_geometry": geometry_id,
            "source_keys": source_keys,
        }
    )
    vertex_ids = tuple(
        source_entity_id(
            source_id,
            source_revision,
            "label-field-source-vertex",
            canonical_fingerprint({"source_key": key}),
        )
        for key in source_keys
    )
    faces = topology.host_faces()
    face_labels = topology.host_face_labels()
    region_to_plc = {
        identifier: index for index, identifier in enumerate(finite_region_ids)
    }
    plc_regions = np.empty((topology.face_count, 2), dtype=np.int64)
    source_region_pairs: list[tuple[str, str]] = []
    facet_ids: list[str] = []
    for row, (face, labels) in enumerate(
        zip(faces, face_labels, strict=True)
    ):
        left_id = topology.region_ids[int(labels[0])]
        right_id = topology.region_ids[int(labels[1])]
        left = region_to_plc.get(left_id, -1)
        right = region_to_plc.get(right_id, -1)
        if left < 0 and right < 0:
            raise _label_field_failure(
                "A label-field interface cannot separate two non-material boundary regions."
            )
        # Surface normals point out of the left label; PLC positive sides are
        # along the loop normal, so its incidence order is (right, left).
        plc_regions[row] = (right, left)
        source_region_pairs.append((left_id, right_id))
        oriented_keys = tuple(source_keys[int(vertex)] for vertex in face)
        rotations = tuple(
            oriented_keys[offset:] + oriented_keys[:offset]
            for offset in range(3)
        )
        facet_key = canonical_fingerprint(
            {
                "oriented_source_vertices": min(rotations),
                "source_region_ids": (left_id, right_id),
            }
        )
        facet_ids.append(
            source_entity_id(
                source_id, source_revision, "label-field-source-facet", facet_key
            )
        )

    complex_ = PiecewiseLinearComplex(
        positions,
        tuple(faces[row] for row in range(topology.face_count)),
        np.arange(topology.face_count, dtype=np.int64),
        plc_regions,
        finite_region_ids,
        vertex_ids=vertex_ids,
        vertex_occurrence_paths=((),) * len(vertex_ids),
        facet_ids=tuple(facet_ids),
        facet_occurrence_paths=((),) * len(facet_ids),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),) * len(finite_region_ids),
        boundary="conforming",
    )
    return _LabelFieldPlcNormalization(
        NativePlcSource(complex_, source_id, source_revision),
        tuple(source_region_pairs),
        lineage.boundary_region_id,
    )




@final
class _PreparedCompartmentRoute(StrictModule, NonTrainableState):
    """Prepared categorical or accepted-surface PLC request."""

    source: CompartmentMeshingSource | LabelFieldSurfaceExtractionResult
    preparation: CompartmentPlcPreparation | None
    label_field: _LabelFieldPlcNormalization | None
    volume: PreparedPlcVolume

    def __init__(
        self,
        source: CompartmentMeshingSource | LabelFieldSurfaceExtractionResult,
        preparation: CompartmentPlcPreparation | None,
        label_field: _LabelFieldPlcNormalization | None,
        volume: PreparedPlcVolume,
        /,
    ) -> None:
        self.source = source
        self.preparation = preparation
        self.label_field = label_field
        self.volume = volume

    def execute(self, /) -> NativeMeshConstruction:
        construction = self.volume.execute()
        if self.label_field is not None:
            facet_source_region_ids = dict(
                zip(
                    self.label_field.source.complex.facet_ids,
                    self.label_field.facet_source_region_ids,
                    strict=True,
                )
            )
            patches = []
            represented_facets: set[str] = set()
            for patch in construction.organization.patches:
                matches = tuple(
                    (facet_id, source_pair)
                    for facet_id, source_pair in facet_source_region_ids.items()
                    if patch.name.startswith(f"source-facet:{facet_id}:")
                )
                if len(matches) > 1:
                    raise ValueError(
                        "One native PLC patch ambiguously represents multiple label-field faces."
                    )
                if not matches:
                    patches.append(patch)
                    continue
                facet_id, source_pair = matches[0]
                if facet_id in represented_facets:
                    raise ValueError(
                        "One label-field face must map to one canonical PLC patch."
                    )
                represented_facets.add(facet_id)
                left_id, right_id = source_pair
                source_region_keys = (
                    None
                    if left_id == self.label_field.boundary_region_id
                    else _source_region_key(self.label_field.source, left_id),
                    None
                    if right_id == self.label_field.boundary_region_id
                    else _source_region_key(self.label_field.source, right_id),
                )
                patches.append(
                    type(patch)(
                        patch.name,
                        patch.scope,
                        connected=patch.connected,
                        adjacent_zone_ids=patch.adjacent_zone_ids,
                        source_adjacent_region_ids=(
                            None if source_region_keys[0] is None else source_region_keys[0][0],
                            None if source_region_keys[1] is None else source_region_keys[1][0],
                        ),
                        source_adjacent_region_occurrence_paths=(
                            None if source_region_keys[0] is None else source_region_keys[0][1],
                            None if source_region_keys[1] is None else source_region_keys[1][1],
                        ),
                    )
                )
            if represented_facets != set(facet_source_region_ids):
                raise ValueError(
                    "Native PLC patches do not preserve every accepted label-field face."
                )
            organization = replace(
                construction.organization,
                patches=tuple(patches),
            )
            return replace(construction, organization=organization)

        if not isinstance(self.source, CompartmentMeshingSource) or not isinstance(
            self.preparation, CompartmentPlcPreparation
        ):
            raise RuntimeError("Prepared compartment route lost its source preparation.")
        organization = construction.organization
        facet_interface_sides = {
            facet_id: (interface_id, source_pair)
            for facet_id, interface_id, source_pair in zip(
                self.preparation.complex.facet_ids,
                self.preparation.facet_interface_ids,
                self.preparation.facet_region_ids,
                strict=True,
            )
            if interface_id is not None
        }
        definitions = {
            value.interface_id: value
            for value in self.source.compartments.interfaces
        }
        patches = []
        renamed = set()
        for patch in organization.patches:
            matches = tuple(
                (facet_id, interface_id, source_pair)
                for facet_id, (interface_id, source_pair) in facet_interface_sides.items()
                if patch.name.startswith(f"source-facet:{facet_id}:")
            )
            if len(matches) > 1:
                raise ValueError(
                    "One native PLC patch ambiguously represents multiple source material interfaces."
                )
            if not matches:
                patches.append(patch)
                continue
            _, interface_id, source_pair = matches[0]
            definition = definitions[interface_id]
            if set(source_pair) != {
                definition.first_compartment_id,
                definition.second_compartment_id,
            }:
                raise ValueError(
                    "Compartment interface source sides disagree with their declared adjacency."
                )
            if interface_id in renamed:
                raise ValueError(
                    "A material interface must map to one canonical PLC patch."
                )
            renamed.add(interface_id)
            source_region_keys = (
                _source_region_key(self.source, source_pair[0]),
                _source_region_key(self.source, source_pair[1]),
            )
            patches.append(
                type(patch)(
                    interface_id,
                    patch.scope,
                    connected=patch.connected,
                    adjacent_zone_ids=patch.adjacent_zone_ids,
                    source_adjacent_region_ids=(
                        None if source_region_keys[0] is None else source_region_keys[0][0],
                        None if source_region_keys[1] is None else source_region_keys[1][0],
                    ),
                    source_adjacent_region_occurrence_paths=(
                        None if source_region_keys[0] is None else source_region_keys[0][1],
                        None if source_region_keys[1] is None else source_region_keys[1][1],
                    ),
                )
            )
        if len({patch.name for patch in patches}) != len(patches):
            raise ValueError(
                "Compartment interface identifiers cannot collide with other output patch names."
            )
        organization = replace(
            organization,
            boundary=self.source.outer_surface,
            patches=tuple(patches),
        )
        return replace(construction, organization=organization)


@checked
def prepare_native_compartment(
    source: CompartmentMeshingSource | LabelFieldSurfaceExtractionResult,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedCompartmentRoute:
    """Normalize categorical labels or accepted material surfaces to a PLC fill."""
    if not isinstance(
        source, (CompartmentMeshingSource, LabelFieldSurfaceExtractionResult)
    ):
        raise TypeError(
            "source must be a CompartmentMeshingSource or accepted label-field extraction."
        )
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("specification must be a VolumeMeshingSpec.")
    if not isinstance(options, NativeMeshingOptions):
        raise TypeError("options must be NativeMeshingOptions.")
    if isinstance(source, CompartmentMeshingSource):
        if options.route != "image_material_tetrahedral":
            raise ValueError(
                "Native compartment sources require the image material route."
            )
    elif options.route not in (
        "image_material_tetrahedral",
        "plc_tetrahedral",
    ):
        raise ValueError(
            "Label-field material normalization requires an explicit PLC volume route."
        )
    if isinstance(source, CompartmentMeshingSource):
        if source.coordinate_contract.spatial_id != coordinate_contract.spatial_id:
            raise ValueError(
                "The coordinate contract must bind the categorical image source."
            )
        preparation = prepare_compartment_plc(source)
        label_field = None
        plc_source = NativePlcSource(
            preparation.complex,
            source.source_id,
            source.source_revision,
        )
    else:
        label_field = _normalize_label_field_source(source)
        preparation = None
        plc_source = label_field.source
    if specification.boundary_scope.source_id != plc_source.source_id or (
        specification.boundary_scope.source_revision != plc_source.source_revision
    ):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Compartment boundary scope must bind the exact normalized source revision.",
            provider_code="native_compartment_scope",
            stage="source_inspection",
        )
    schedule = options.volume_schedule
    if not isinstance(schedule, NativeVolumeSchedule):
        raise TypeError(
            "The selected explicit material route requires a NativeVolumeSchedule."
        )

    unsupported = volume_support_issues(plc_source, specification)
    if unsupported:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "; ".join(unsupported),
            provider_code="native_compartment_preflight",
            stage="source_inspection",
        )
    volume = PreparedPlcVolume(
        plc_source,
        specification,
        coordinate_contract,
        schedule,
    )
    return _PreparedCompartmentRoute(
        source, preparation, label_field, volume
    )




__all__ = ["prepare_native_compartment"]
