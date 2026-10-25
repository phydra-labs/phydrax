#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Audited segmentation processing and exact voxel-interface extraction."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy import ndimage

from .._fingerprint import (
    _canonical_json_text_upper_bytes,
    _retained_text_storage_upper_bytes,
    canonical_fingerprint,
)
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateTemporaryStorage,
)
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
            asset.references,
            DerivationRecord(
                asset.derivation.origin,
                DataStage.DERIVED,
                (asset.content_id,),
                self.plan_id,
            ),
            time_axis=asset.time_axis,
            valid_mask=asset.valid_mask,
            uncertainty=None,
            quality_flags=asset.quality_flags,
            acquisition=asset.acquisition,
            metadata={} if asset.metadata is None else dict(asset.metadata),
            intended_use=asset.intended_use,
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


def _admit_compartment_source_phase(
    budget: CoordinateEnclosureBudget,
    work_units: int,
    scratch_bytes: int,
    result_bytes: int,
    retained_work_units: int,
    /,
) -> None:
    budget.ticket.checkpoint()
    budget.admit_work_bound(
        work_units + retained_work_units,
        storage_upper=scratch_bytes + result_bytes,
    )
    budget.reserve(work_units, scratch_bytes + result_bytes)


def _finish_compartment_source_phase(
    budget: CoordinateEnclosureBudget,
    lifetime: CoordinateTemporaryStorage,
    result: object,
    scratch_bytes: int,
    result_bytes: int,
    /,
    *,
    retain_result: bool,
) -> None:
    budget.ticket.checkpoint()
    if retain_result:
        with budget.live_storage() as storage:
            storage.promote_temporary_storage(lifetime, result_bytes)
            budget.retain_owner(result, temporary_storage_credit=storage)
    budget.release_temporary_storage(scratch_bytes)


def _compartment_complex_text_upper_bytes(
    labels: LabelVolume,
    compartments: tuple[CompartmentDefinition, ...],
    interfaces: tuple[CompartmentInterfaceDefinition, ...],
    /,
) -> tuple[int, int, int]:
    region_count = len(compartments)

    def report_texts() -> Iterator[str]:
        # Expected and missing each repeat every required interface pair;
        # observed and forbidden each fit the full unordered-pair set.
        yield from ("kind", "compartment-adjacency-report")
        yield from ("expected", "observed", "missing", "forbidden")
        for interface in interfaces:
            if interface.required:
                for _ in range(2):
                    yield interface.first_compartment_id
                    yield interface.second_compartment_id
        for compartment in compartments:
            for _ in range(2 * max(0, region_count - 1)):
                yield compartment.compartment_id

    def complex_texts() -> Iterator[str]:
        # Complex identity recipe, including all declaration fields and
        # measure names; keys are included alongside the variable values.
        yield from (
            "kind",
            "compartment-complex",
            "source_revision",
            labels.label_volume_id,
            "compartments",
            "interfaces",
            "measures",
            "adjacency",
        )
        for compartment in compartments:
            yield from ("id", compartment.compartment_id)
            yield "labels"
            yield from compartment.label_ids
            yield from ("role", compartment.material_role, "parent")
            if compartment.containment_parent_id is not None:
                yield compartment.containment_parent_id
            yield "neighbors"
            yield from compartment.allowed_neighbor_ids
        for interface in interfaces:
            yield from (
                "id",
                interface.interface_id,
                "pair",
                interface.first_compartment_id,
                interface.second_compartment_id,
                "role",
                interface.interface_role,
                "required",
            )
        for compartment in compartments:
            yield compartment.compartment_id

    def retained_texts() -> Iterator[str]:
        yield labels.label_volume_id
        for compartment in compartments:
            yield compartment.compartment_id
            yield from compartment.label_ids
            yield compartment.material_role
            if compartment.containment_parent_id is not None:
                yield compartment.containment_parent_id
            yield from compartment.allowed_neighbor_ids
        for interface in interfaces:
            yield interface.interface_id
            yield interface.first_compartment_id
            yield interface.second_compartment_id
            yield interface.interface_role

    return (
        _canonical_json_text_upper_bytes(report_texts()),
        _canonical_json_text_upper_bytes(complex_texts()),
        _retained_text_storage_upper_bytes(retained_texts()),
    )




def _compartment_surface_text_bounds(
    labels: LabelVolume,
    complex: CompartmentComplex,
    face_count: int,
    /,
) -> tuple[int, int, int]:
    result_text_upper = _canonical_json_text_upper_bytes(
        ("kind", "compartment-surface-result", "complex", "surfaces")
    )
    text_work_upper = result_text_upper
    text_peak_upper = result_text_upper
    if face_count:
        for definition in complex.interfaces:
            metadata_text_upper = _canonical_json_text_upper_bytes(
                (
                    "kind",
                    "surface-metadata",
                    "source_id",
                    labels.asset.asset_id,
                    "source_revision",
                    labels.label_volume_id,
                    "coordinate_contract",
                    labels.asset.spatial_affine.coordinate_contract.spatial_id,
                    "provenance",
                    "voxel-compartment-interface",
                    definition.interface_id,
                    "cell_tags",
                )
            )
            surface_text_upper = _canonical_json_text_upper_bytes(
                (
                    "kind",
                    "compartment-surface",
                    "interface",
                    definition.interface_id,
                    "source",
                    labels.label_volume_id,
                    "mesh",
                )
            )
            text_work_upper += metadata_text_upper + surface_text_upper
            text_peak_upper = max(
                text_peak_upper, metadata_text_upper, surface_text_upper
            )

    _, _, complex_text_storage_upper = _compartment_complex_text_upper_bytes(
        labels, complex.compartments, complex.interfaces
    )
    surface_text_storage_upper = 0
    if face_count and complex.interfaces:
        contract = labels.asset.spatial_affine.coordinate_contract

        def retained_surface_texts() -> Iterator[str]:
            yield labels.asset.asset_id
            yield contract.length_coordinate_kind
            yield contract.coordinate_system
            yield contract.reference_frame
            yield contract.length_unit.symbol
            yield contract.length_unit.reference_system_id
            yield from (
                name for name, _, _ in contract.length_unit.dimension.terms
            )

        surface_text_storage_upper = _retained_text_storage_upper_bytes(
            retained_surface_texts()
        )
    return (
        text_work_upper,
        2 * text_peak_upper,
        complex_text_storage_upper + surface_text_storage_upper,
    )


def build_compartment_complex(
    labels: LabelVolume,
    compartments: tuple[CompartmentDefinition, ...],
    interfaces: tuple[CompartmentInterfaceDefinition, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> CompartmentComplex:
    if budget is None:
        return _build_compartment_complex_core(labels, compartments, interfaces)
    with budget.temporary_scope() as lifetime:
        return _build_compartment_complex(
            labels, compartments, interfaces, budget=budget, lifetime=lifetime,
            retain_result=True,
        )


def _build_compartment_complex(
    labels: LabelVolume,
    compartments: tuple[CompartmentDefinition, ...],
    interfaces: tuple[CompartmentInterfaceDefinition, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
    lifetime: CoordinateTemporaryStorage,
    retain_result: bool,
) -> CompartmentComplex:
    if not isinstance(labels, LabelVolume):
        raise TypeError("labels must be LabelVolume.")
    voxel_count = labels.asset.values.size
    region_count = len(compartments)
    label_count = sum(len(value.label_ids) for value in compartments)
    interface_count = len(interfaces)
    pair_capacity = region_count * max(0, region_count - 1) // 2
    itemsize = labels.asset.values.dtype.itemsize
    report_text_upper, complex_text_upper, retained_text_bytes = (
        _compartment_complex_text_upper_bytes(
            labels, compartments, interfaces
        )
    )
    text_work_upper = report_text_upper + complex_text_upper
    # The fingerprints run sequentially, so scratch holds only the larger
    # escaped payload while work sums both complete serializer traversals.
    text_scratch_upper = 2 * max(report_text_upper, complex_text_upper)
    work_units = (
        voxel_count * (region_count + 6)
        + label_count
        + region_count * region_count
        + interface_count
        + 128
        + text_work_upper
    )
    # Two selected label arrays and concurrent masks bound the volume
    # temporary; the text term is the larger simultaneous fingerprint payload.
    scratch_bytes = (
        voxel_count * (2 * itemsize + 16)
        + 512 * (region_count * region_count + label_count + interface_count + 1)
        + text_scratch_upper
    )
    # Retained text is credited only for published results; temporary
    # revalidation shares the existing source owners.
    result_bytes = (
        8192
        + 512 * (region_count + interface_count + pair_capacity + 1)
        + (retained_text_bytes if retain_result else 0)
    )
    retained_work_units = (
        64 + region_count * region_count + label_count + interface_count
    )
    _admit_compartment_source_phase(
        budget, work_units, scratch_bytes, result_bytes, retained_work_units
    )
    result = _build_compartment_complex_core(labels, compartments, interfaces)
    _finish_compartment_source_phase(
        budget, lifetime, result, scratch_bytes, result_bytes,
        retain_result=retain_result,
    )
    return result


def _build_compartment_complex_core(
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
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> CompartmentSurfaceResult:
    if complex.source_revision != labels.label_volume_id:
        raise ValueError("Compartment complex and label volume revisions differ.")
    if budget is None:
        compartment_code, compartment_ids, _ = _compartment_code(labels, complex)
        return _extract_compartment_surfaces_core(
            labels, complex, compartment_code, compartment_ids
        )
    return _extract_compartment_surfaces(
        labels, complex, budget=budget, lifetime=None, retain_result=True
    )


def _compartment_code(
    labels: LabelVolume,
    complex: CompartmentComplex,
    /,
) -> tuple[np.ndarray, tuple[str, ...], dict[int, str]]:
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
    return compartment_code, compartment_ids, value_to_compartment


def _count_compartment_faces(
    labels: LabelVolume,
    compartment_code: np.ndarray,
    /,
) -> int:
    valid = np.asarray(labels.asset.valid_mask)
    face_count = 0
    for axis in range(3):
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
        face_count += int(np.count_nonzero(supported))
    return face_count


def _extract_compartment_surfaces(
    labels: LabelVolume,
    complex: CompartmentComplex,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    lifetime: CoordinateTemporaryStorage | None,
    retain_result: bool,
) -> CompartmentSurfaceResult:
    if complex.source_revision != labels.label_volume_id:
        raise ValueError("Compartment complex and label volume revisions differ.")
    voxel_count = labels.asset.values.size
    region_count = len(complex.compartments)
    label_count = sum(len(value.label_ids) for value in complex.compartments)
    interface_count = len(complex.interfaces)
    code_bytes = (
        4 * voxel_count + 1024 + 512 * (region_count + label_count + 1)
    )
    # One visit per region-volume pass and each of three face-count traversals.
    code_work = (
        voxel_count * (region_count + 3)
        + label_count
        + region_count
        + interface_count
        + 128
    )
    code_scratch = 16 * voxel_count + 512 * (
        region_count + label_count + 1
    )
    with budget.live_storage() as code_storage:
        code_storage.set_bound(code_bytes)
        with budget.temporary_scope():
            budget.ticket.checkpoint()
            budget.admit_work_bound(code_work, storage_upper=code_scratch)
            budget.reserve(code_work, code_scratch)
            compartment_code, compartment_ids, value_to_compartment = (
                _compartment_code(labels, complex)
            )
            face_count = _count_compartment_faces(labels, compartment_code)
            budget.ticket.checkpoint()
        sort_work = 24 * face_count * (4 * face_count + 1).bit_length()
        text_work_upper, text_scratch_upper, retained_text_bytes = (
            _compartment_surface_text_bounds(labels, complex, face_count)
        )
        work_units = (
            3 * voxel_count
            + face_count * 32
            + sort_work
            + interface_count
            + 128
            + text_work_upper
        )
        # Surface metadata and identities serialize once per emitted interface;
        # only the largest simultaneous JSON string and encoded copy add peak
        # scratch, while work sums every possible interface fingerprint.
        scratch_bytes = (
            256 * voxel_count
            + 512 * (interface_count + 1)
            + text_scratch_upper
        )
        # Published output ownership includes strings from the complex and
        # surfaces; revalidation shares those source owners with its caller.
        result_bytes = (
            8192 * face_count
            + 4096 * (interface_count + 1)
            + 8192
            + (retained_text_bytes if retain_result else 0)
        )
        retained_work_units = 128 + 16 * interface_count
        try:
            if lifetime is None:
                with budget.temporary_scope() as result_lifetime:
                    _admit_compartment_source_phase(
                        budget,
                        work_units,
                        scratch_bytes,
                        result_bytes,
                        retained_work_units if retain_result else 0,
                    )
                    result = _extract_compartment_surfaces_core(
                        labels, complex, compartment_code, compartment_ids
                    )
                    _finish_compartment_source_phase(
                        budget,
                        result_lifetime,
                        result,
                        scratch_bytes,
                        result_bytes,
                        retain_result=retain_result,
                    )
            else:
                _admit_compartment_source_phase(
                    budget,
                    work_units,
                    scratch_bytes,
                    result_bytes,
                    retained_work_units if retain_result else 0,
                )
                result = _extract_compartment_surfaces_core(
                    labels, complex, compartment_code, compartment_ids
                )
                _finish_compartment_source_phase(
                    budget,
                    lifetime,
                    result,
                    scratch_bytes,
                    result_bytes,
                    retain_result=retain_result,
                )
        finally:
            del compartment_code, compartment_ids, value_to_compartment
    return result


def _extract_compartment_surfaces_core(
    labels: LabelVolume,
    complex: CompartmentComplex,
    compartment_code: np.ndarray,
    compartment_ids: tuple[str, ...],
    /,
) -> CompartmentSurfaceResult:
    interfaces = {value.ordered_pair: value for value in complex.interfaces}
    loops: dict[str, list[tuple[tuple[int, int, int], ...]]] = {
        value.interface_id: [] for value in complex.interfaces
    }
    lower_counts = {value.interface_id: 0 for value in complex.interfaces}
    np.asarray(labels.asset.values)
    valid = np.asarray(labels.asset.valid_mask)
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
            index = tuple(index_array)
            first = compartment_ids[int(lower[index])]
            second = compartment_ids[int(upper[index])]
            definition = interfaces.get(tuple(sorted((first, second))))
            if definition is None:
                continue
            center = np.asarray(index, dtype=np.int64) * 2
            center[axis] += 1
            corner_offsets = []
            for first_sign, second_sign in (
                (-1, -1),
                (1, -1),
                (1, 1),
                (-1, 1),
            ):
                offset = np.zeros(3, dtype=np.int64)
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
        index_coordinates = np.asarray(keys, dtype=np.float64) / 2.0
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


def _rebuild_compartment_source_facts(
    labels: LabelVolume,
    compartments: CompartmentComplex,
    interfaces: CompartmentSurfaceResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> None:
    if budget is None:
        expected_complex = build_compartment_complex(
            labels, compartments.compartments, compartments.interfaces
        )
        if expected_complex.complex_id != compartments.complex_id:
            raise ValueError(
                "CompartmentMeshingSource.compartments must match image measures and adjacency; "
                f"got {compartments.complex_id!r} != {expected_complex.complex_id!r}"
            )
        expected_interfaces = extract_compartment_surfaces(labels, compartments)
    else:
        with budget.temporary_scope() as lifetime:
            expected_complex = _build_compartment_complex(
                labels,
                compartments.compartments,
                compartments.interfaces,
                budget=budget,
                lifetime=lifetime,
                retain_result=False,
            )
            try:
                if expected_complex.complex_id != compartments.complex_id:
                    raise ValueError(
                        "CompartmentMeshingSource.compartments must match image measures and adjacency; "
                        f"got {compartments.complex_id!r} != {expected_complex.complex_id!r}"
                    )
                expected_interfaces = _extract_compartment_surfaces(
                    labels,
                    compartments,
                    budget=budget,
                    lifetime=lifetime,
                    retain_result=False,
                )
                try:
                    if expected_interfaces.extraction_id != interfaces.extraction_id:
                        raise ValueError(
                            "CompartmentMeshingSource.interfaces must represent the occupied image cells; "
                            f"got {interfaces.extraction_id!r} != {expected_interfaces.extraction_id!r}"
                        )
                    if tuple(
                        (value.definition.interface_id, value.surface.mesh.mesh_id)
                        for value in interfaces.surfaces
                    ) != tuple(
                        (value.definition.interface_id, value.surface.mesh.mesh_id)
                        for value in expected_interfaces.surfaces
                    ):
                        raise ValueError(
                            "CompartmentMeshingSource.interfaces.surfaces must match the extraction binding; "
                            "got different interface or mesh IDs"
                        )
                finally:
                    del expected_interfaces
            finally:
                del expected_complex
        return
    if expected_interfaces.extraction_id != interfaces.extraction_id:
        raise ValueError(
            "CompartmentMeshingSource.interfaces must represent the occupied image cells; "
            f"got {interfaces.extraction_id!r} != {expected_interfaces.extraction_id!r}"
        )
    if tuple(
        (value.definition.interface_id, value.surface.mesh.mesh_id)
        for value in interfaces.surfaces
    ) != tuple(
        (value.definition.interface_id, value.surface.mesh.mesh_id)
        for value in expected_interfaces.surfaces
    ):
        raise ValueError(
            "CompartmentMeshingSource.interfaces.surfaces must match the extraction binding; "
            "got different interface or mesh IDs"
        )


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
