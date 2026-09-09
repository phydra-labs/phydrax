#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Audited segmentation processing and exact voxel-interface extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy import ndimage

from .._fingerprint import canonical_fingerprint
from ..geometry import (
    CompartmentComplex,
    CompartmentDefinition,
    CompartmentInterfaceDefinition,
)
from ..geometry.surface import SurfaceMetadata, SurfaceModel
from ..measurement import DataStage, DerivationRecord
from ._core import LabelVolume, MedicalImageAsset


class SegmentationOperationKind(StrEnum):
    KEEP_LARGEST_COMPONENT = "keep_largest_component"
    REMOVE_SMALL_COMPONENTS = "remove_small_components"
    FILL_HOLES = "fill_holes"


@dataclass(frozen=True, slots=True)
class SegmentationOperation:
    kind: SegmentationOperationKind
    label_id: str
    replacement_label_id: str
    minimum_volume: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SegmentationOperationKind):
            raise TypeError("kind must be SegmentationOperationKind.")
        if (
            not isinstance(self.label_id, str)
            or not self.label_id
            or self.label_id != self.label_id.strip()
            or not isinstance(self.replacement_label_id, str)
            or not self.replacement_label_id
            or self.replacement_label_id != self.replacement_label_id.strip()
        ):
            raise ValueError(
                "Operation label IDs must be canonical non-empty identifiers."
            )
        minimum = float(self.minimum_volume)
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_volume must be finite and non-negative.")
        if (
            self.kind is SegmentationOperationKind.REMOVE_SMALL_COMPONENTS
            and minimum <= 0.0
        ):
            raise ValueError("REMOVE_SMALL_COMPONENTS requires positive minimum_volume.")
        object.__setattr__(self, "minimum_volume", minimum)


@dataclass(frozen=True, slots=True)
class SegmentationOperationReport:
    operation: SegmentationOperation
    changed_voxels: int
    changed_measure: float
    components_before: int
    components_after: int
    report_id: str


@dataclass(frozen=True, slots=True)
class SegmentationTransition:
    source: LabelVolume
    target: LabelVolume
    reports: tuple[SegmentationOperationReport, ...]
    transition_id: str


@dataclass(frozen=True, slots=True)
class SegmentationProcessingPlan:
    source: LabelVolume
    operations: tuple[SegmentationOperation, ...]
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, LabelVolume):
            raise TypeError("source must be LabelVolume.")
        if not self.operations or any(
            not isinstance(value, SegmentationOperation) for value in self.operations
        ):
            raise ValueError("operations must contain SegmentationOperation values.")
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "segmentation-processing-plan",
                    "source": self.source.label_volume_id,
                    "operations": [
                        {
                            "kind": value.kind.value,
                            "label": value.label_id,
                            "replacement": value.replacement_label_id,
                            "minimum_volume": value.minimum_volume,
                        }
                        for value in self.operations
                    ],
                }
            ),
        )

    def execute(self) -> SegmentationTransition:
        labels_by_id = {
            value.label_id: value.value for value in self.source.ontology.labels
        }
        data = np.array(self.source.asset.values, copy=True)
        voxel_measure = abs(
            float(np.linalg.det(self.source.asset.spatial_affine.matrix[:3, :3]))
        )
        reports = []
        structure = ndimage.generate_binary_structure(3, 1)
        for operation in self.operations:
            if (
                operation.label_id not in labels_by_id
                or operation.replacement_label_id not in labels_by_id
            ):
                raise ValueError(
                    "Segmentation operation labels must belong to the source ontology."
                )
            value = labels_by_id[operation.label_id]
            replacement = labels_by_id[operation.replacement_label_id]
            before = data == value
            components, before_count = ndimage.label(before, structure=structure)
            after = before.copy()
            if operation.kind is SegmentationOperationKind.KEEP_LARGEST_COMPONENT:
                sizes = np.bincount(components.reshape((-1,)))
                keep = 0 if len(sizes) <= 1 else int(np.argmax(sizes[1:]) + 1)
                after = components == keep
            elif operation.kind is SegmentationOperationKind.REMOVE_SMALL_COMPONENTS:
                sizes = np.bincount(components.reshape((-1,)))
                minimum_count = int(np.ceil(operation.minimum_volume / voxel_measure))
                retained = np.flatnonzero(sizes >= minimum_count)
                retained = retained[retained != 0]
                after = np.isin(components, retained)
            else:
                after = ndimage.binary_fill_holes(before)
            changed = before ^ after
            data[before & ~after] = replacement
            data[after] = value
            _, after_count = ndimage.label(after, structure=structure)
            changed_count = int(np.count_nonzero(changed))
            report_id = canonical_fingerprint(
                {
                    "kind": "segmentation-operation-report",
                    "plan": self.plan_id,
                    "operation": operation.kind.value,
                    "changed": changed_count,
                    "components_before": int(before_count),
                    "components_after": int(after_count),
                }
            )
            reports.append(
                SegmentationOperationReport(
                    operation,
                    changed_count,
                    changed_count * voxel_measure,
                    int(before_count),
                    int(after_count),
                    report_id,
                )
            )
        asset = self.source.asset
        target_asset = MedicalImageAsset(
            f"{asset.asset_id}:segmentation:{self.plan_id[:12]}",
            asset.modality,
            data,
            asset.spatial_affine,
            asset.spec,
            asset.deidentification,
            asset.reference,
            DerivationRecord(
                asset.derivation.origin,
                DataStage.DERIVED,
                (asset.content_id,),
                self.plan_id,
            ),
            asset.time_axis,
            asset.valid_mask,
            asset.acquisition,
            {} if asset.metadata is None else dict(asset.metadata),
            asset.intended_use,
        )
        target = LabelVolume(target_asset, self.source.ontology)
        transition_id = canonical_fingerprint(
            {
                "kind": "segmentation-transition",
                "source": self.source.label_volume_id,
                "target": target.label_volume_id,
                "reports": [value.report_id for value in reports],
            }
        )
        return SegmentationTransition(self.source, target, tuple(reports), transition_id)


@dataclass(frozen=True, slots=True)
class CompartmentSurface:
    definition: CompartmentInterfaceDefinition
    surface: SurfaceModel
    lower_label_face_count: int
    surface_id: str


@dataclass(frozen=True, slots=True)
class CompartmentSurfaceResult:
    complex: CompartmentComplex
    surfaces: tuple[CompartmentSurface, ...]
    extraction_id: str


def build_compartment_complex(
    labels: LabelVolume,
    compartments: tuple[CompartmentDefinition, ...],
    interfaces: tuple[CompartmentInterfaceDefinition, ...],
    /,
) -> CompartmentComplex:
    if not isinstance(labels, LabelVolume):
        raise TypeError("labels must be LabelVolume.")
    ontology = {value.label_id: value.value for value in labels.ontology.labels}
    value_to_compartment: dict[int, str] = {}
    for compartment in compartments:
        for label_id in compartment.label_ids:
            if label_id not in ontology:
                raise ValueError(
                    f"Unknown ontology label {label_id!r} in compartment definition."
                )
            value = ontology[label_id]
            if value in value_to_compartment:
                raise ValueError("Ontology labels may belong to only one compartment.")
            value_to_compartment[value] = compartment.compartment_id
    data = np.asarray(labels.asset.values)
    valid = np.asarray(labels.asset.valid_mask)
    voxel_measure = abs(float(np.linalg.det(labels.asset.spatial_affine.matrix[:3, :3])))
    measures = []
    for compartment in compartments:
        values = [ontology[value] for value in compartment.label_ids]
        count = int(np.count_nonzero(valid & np.isin(data, values)))
        if count == 0:
            raise ValueError(f"Compartment {compartment.compartment_id!r} is empty.")
        measures.append((compartment.compartment_id, count * voxel_measure))
    observed: set[tuple[str, str]] = set()
    for axis in range(3):
        lower_slice = [slice(None)] * 3
        upper_slice = [slice(None)] * 3
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)
        lower = data[tuple(lower_slice)]
        upper = data[tuple(upper_slice)]
        supported = (
            valid[tuple(lower_slice)] & valid[tuple(upper_slice)] & (lower != upper)
        )
        for first_value, second_value in zip(
            lower[supported], upper[supported], strict=True
        ):
            first = value_to_compartment.get(int(first_value))
            second = value_to_compartment.get(int(second_value))
            if first is not None and second is not None and first != second:
                observed.add((first, second) if first < second else (second, first))
    return CompartmentComplex(
        labels.label_volume_id,
        compartments,
        interfaces,
        tuple(measures),
        tuple(sorted(observed)),
    )


def extract_compartment_surfaces(
    labels: LabelVolume,
    complex: CompartmentComplex,
    /,
) -> CompartmentSurfaceResult:
    if complex.source_revision != labels.label_volume_id:
        raise ValueError("Compartment complex and label volume revisions differ.")
    ontology = {value.label_id: value.value for value in labels.ontology.labels}
    value_to_compartment = {
        ontology[label_id]: compartment.compartment_id
        for compartment in complex.compartments
        for label_id in compartment.label_ids
    }
    compartment_ids = tuple(
        compartment.compartment_id for compartment in complex.compartments
    )
    compartment_code = np.full(labels.asset.values.shape, -1, dtype=np.int32)
    for code, compartment_id in enumerate(compartment_ids):
        source_values = [
            value
            for value, assigned in value_to_compartment.items()
            if assigned == compartment_id
        ]
        compartment_code[np.isin(labels.asset.values, source_values)] = code
    interfaces = {value.ordered_pair: value for value in complex.interfaces}
    loops: dict[str, list[tuple[tuple[int, int, int], ...]]] = {
        value.interface_id: [] for value in complex.interfaces
    }
    lower_counts = {value.interface_id: 0 for value in complex.interfaces}
    data = np.asarray(labels.asset.values)
    valid = np.asarray(labels.asset.valid_mask)
    shape = data.shape
    # Doubled voxel-index coordinates make half-index boundary vertices integral.
    axes = ((1, 2), (2, 0), (0, 1))
    reflected_world_frame = (
        np.linalg.det(labels.asset.spatial_affine.matrix[:3, :3]) < 0.0
    )
    for axis in range(3):
        first_axis, second_axis = axes[axis]
        lower_slice = [slice(None)] * 3
        upper_slice = [slice(None)] * 3
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)
        lower = compartment_code[tuple(lower_slice)]
        upper = compartment_code[tuple(upper_slice)]
        supported = (
            valid[tuple(lower_slice)]
            & valid[tuple(upper_slice)]
            & (lower >= 0)
            & (upper >= 0)
            & (lower != upper)
        )
        for index_array in np.argwhere(supported):
            index = tuple(int(value) for value in index_array)
            first = compartment_ids[int(lower[index])]
            second = compartment_ids[int(upper[index])]
            definition = interfaces.get(tuple(sorted((first, second))))
            if definition is None:
                continue
            center = np.asarray(index, dtype=int) * 2
            center[axis] += 1
            corner_offsets = []
            for first_sign, second_sign in (
                (-1, -1),
                (1, -1),
                (1, 1),
                (-1, 1),
            ):
                offset = np.zeros(3, dtype=int)
                offset[first_axis] = first_sign
                offset[second_axis] = second_sign
                corner_offsets.append(tuple((center + offset).tolist()))
            loop = tuple(corner_offsets)
            if first == definition.first_compartment_id:
                lower_counts[definition.interface_id] += 1
            reverse = (first != definition.first_compartment_id) ^ reflected_world_frame
            if reverse:
                loop = tuple(reversed(loop))
            loops[definition.interface_id].append(loop)
    surfaces = []
    for definition in complex.interfaces:
        face_loops = loops[definition.interface_id]
        if not face_loops:
            if definition.required:
                raise ValueError(
                    f"Required interface {definition.interface_id!r} has no voxel faces."
                )
            continue
        keys = sorted({point for loop in face_loops for point in loop})
        key_to_index = {key: index for index, key in enumerate(keys)}
        index_coordinates = np.asarray(keys, dtype=float) / 2.0
        coordinates = labels.asset.spatial_affine.index_to_world(index_coordinates)
        triangles = []
        for loop in face_loops:
            indices = tuple(key_to_index[key] for key in loop)
            triangles.extend(
                (
                    (indices[0], indices[1], indices[2]),
                    (indices[0], indices[2], indices[3]),
                )
            )
        surface = SurfaceModel.from_triangles(
            coordinates,
            np.asarray(triangles, dtype=np.int64),
            SurfaceMetadata(
                source_id=labels.asset.asset_id,
                source_revision=labels.label_volume_id,
                coordinate_contract=labels.asset.spatial_affine.coordinate_contract,
                provenance=("voxel-compartment-interface", definition.interface_id),
            ),
        )
        surface_id = canonical_fingerprint(
            {
                "kind": "compartment-surface",
                "interface": definition.interface_id,
                "source": labels.label_volume_id,
                "mesh": surface.mesh.mesh_id,
            }
        )
        surfaces.append(
            CompartmentSurface(
                definition,
                surface,
                lower_counts[definition.interface_id],
                surface_id,
            )
        )
    extraction_id = canonical_fingerprint(
        {
            "kind": "compartment-surface-result",
            "complex": complex.complex_id,
            "surfaces": [value.surface_id for value in surfaces],
        }
    )
    return CompartmentSurfaceResult(complex, tuple(surfaces), extraction_id)


__all__ = [
    "CompartmentSurface",
    "CompartmentSurfaceResult",
    "SegmentationOperation",
    "SegmentationOperationKind",
    "SegmentationOperationReport",
    "SegmentationProcessingPlan",
    "SegmentationTransition",
    "build_compartment_complex",
    "extract_compartment_surfaces",
]
