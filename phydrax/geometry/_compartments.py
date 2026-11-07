#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Semantic volume compartments and their oriented adjacency complex."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING, TypeAlias

import numpy as np

from .._fingerprint import canonical_fingerprint
from ._source_identity import source_entity_id


if TYPE_CHECKING:
    from .._physical import SpatialCoordinateContract
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..imaging._core import LabelVolume
    from ..imaging._segmentation import CompartmentSurfaceResult
    from .surface._model import SurfaceModel


CompartmentImageInterpretation: TypeAlias = Literal["occupied-voxel-cells"]


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty identifier.")
    return value


def _ordered_pair(first: str, second: str, /) -> tuple[str, str]:
    return (first, second) if first < second else (second, first)


def _compartment_scope_banks(
    labels: LabelVolume,
    compartments: CompartmentComplex,
    source_id: str,
    source_revision: str,
    outer_surface_id: str,
    /,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[tuple[tuple[str, tuple[str, ...]], ...], ...],
    tuple[str, ...],
]:
    """Derive PLC-compatible stable scope banks from image face incidence.

    Voxel coordinates are doubled integer indices, so image-face corners are
    exact lattice keys even under an oblique physical affine. Facet names bind
    the declared interface (or outer source plus owning compartment), while
    edge identities use the same vertex/facet incidence witness as the native
    PLC source.
    """
    ontology = {value.label_id: value.value for value in labels.ontology.labels}
    compartment_ids = tuple(value.compartment_id for value in compartments.compartments)
    value_to_slot: dict[int, int] = {}
    for index, definition in enumerate(compartments.compartments):
        for label_id in definition.label_ids:
            value_to_slot[ontology[label_id]] = index

    data = np.asarray(labels.asset.values)
    valid = np.asarray(labels.asset.valid_mask, dtype=np.bool_)
    codes = np.full(data.shape, -1, dtype=np.int64)
    for value, index in value_to_slot.items():
        codes[valid & (data == value)] = index
    interface_by_pair = {value.ordered_pair: value for value in compartments.interfaces}
    in_plane_axes = ((1, 2), (2, 0), (0, 1))
    face_loops: dict[str, list[tuple[tuple[int, int, int], ...]]] = {}
    lattice_vertices: set[tuple[int, int, int]] = set()

    def corners(center: np.ndarray, axis: int, /) -> tuple[tuple[int, int, int], ...]:
        first_axis, second_axis = in_plane_axes[axis]
        result = []
        for first_sign, second_sign in (
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1),
        ):
            point = center.copy()
            point[first_axis] += first_sign
            point[second_axis] += second_sign
            result.append((int(point[0]), int(point[1]), int(point[2])))
        return tuple(result)

    def append_face(
        center: np.ndarray,
        axis: int,
        facet_name: str,
        /,
    ) -> None:
        loop = corners(center, axis)
        face_loops.setdefault(facet_name, []).append(loop)
        lattice_vertices.update(loop)

    shape = tuple(int(value) for value in data.shape)
    for index in np.ndindex(shape):
        slot = int(codes[index])
        if slot < 0:
            continue
        region_id = compartment_ids[slot]
        for axis in range(3):
            for side in (-1, 1):
                neighbor = list(index)
                neighbor[axis] += side
                if 0 <= neighbor[axis] < shape[axis]:
                    neighbor_slot = int(codes[tuple(neighbor)])
                    if neighbor_slot == slot:
                        continue
                    if neighbor_slot >= 0:
                        if side < 0:
                            continue
                        other_region = compartment_ids[neighbor_slot]
                        pair = _ordered_pair(region_id, other_region)
                        definition = interface_by_pair.get(pair)
                        interface_key = (
                            definition.interface_id
                            if definition is not None
                            else json.dumps(
                                ["undeclared-interface-pair", pair],
                                separators=(",", ":"),
                                ensure_ascii=False,
                            )
                        )
                        facet_name = source_entity_id(
                            source_id,
                            source_revision,
                            "compartment-interface-facet",
                            interface_key,
                        )
                        center = 2 * np.asarray(index, dtype=np.int64)
                        center[axis] += 1
                    else:
                        facet_name = source_entity_id(
                            source_id,
                            source_revision,
                            "compartment-outer-facet",
                            json.dumps(
                                (outer_surface_id, region_id),
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        )
                        center = 2 * np.asarray(index, dtype=np.int64)
                        center[axis] += side
                else:
                    facet_name = source_entity_id(
                        source_id,
                        source_revision,
                        "compartment-outer-facet",
                        json.dumps(
                            (outer_surface_id, region_id),
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                    )
                    center = 2 * np.asarray(index, dtype=np.int64)
                    center[axis] += side
                append_face(center, axis, facet_name)

    vertex_names = tuple(
        source_entity_id(
            source_id,
            source_revision,
            "lattice-vertex",
            json.dumps(key, separators=(",", ":")),
        )
        for key in sorted(lattice_vertices)
    )
    vertex_name_by_key = dict(zip(sorted(lattice_vertices), vertex_names, strict=True))
    edge_facets: dict[tuple[str, str], set[str]] = {}
    for facet_name, loops in face_loops.items():
        for loop in loops:
            names = tuple(vertex_name_by_key[key] for key in loop)
            for index, first in enumerate(names):
                second = names[(index + 1) % len(names)]
                edge = (first, second) if first < second else (second, first)
                edge_facets.setdefault(edge, set()).add(facet_name)
    edge_names = tuple(
        canonical_fingerprint(
            {
                "kind": "plc-facet-edge",
                "vertices": tuple(sorted(((first, ()), (second, ())))),
                "facets": tuple(sorted((facet_name, ()) for facet_name in facets)),
                "occurrence_path": (),
            }
        )
        for (first, second), facets in sorted(edge_facets.items())
    )
    facet_names = tuple(sorted(face_loops))
    region_names = tuple(sorted(compartment_ids))
    name_banks = (vertex_names, edge_names, facet_names, region_names)
    plc_kinds = ("plc-vertex", "plc-edge", "plc-facet", "plc-region")
    id_banks: list[np.ndarray] = []
    key_banks: list[tuple[tuple[str, tuple[str, ...]], ...]] = []
    set_ids: list[str] = []
    maximum = (1 << 63) - 1
    for dimension, (names, plc_kind) in enumerate(
        zip(name_banks, plc_kinds, strict=True)
    ):
        keyed = tuple((name, ()) for name in names)
        numeric = tuple(
            int(canonical_fingerprint({"kind": plc_kind, "value": key})[:16], 16)
            & maximum
            for key in keyed
        )
        if len(set(numeric)) != len(numeric):
            raise ValueError(
                f"Compartment dimension-{dimension} source scope IDs collide in int64."
            )
        order = np.argsort(np.asarray(numeric, dtype=np.int64), kind="stable")
        identifiers = np.asarray(numeric, dtype=np.int64)[order]
        identifiers.setflags(write=False)
        ordered_keys = tuple(keyed[int(index)] for index in order)
        id_banks.append(identifiers)
        key_banks.append(ordered_keys)
        set_ids.append(
            canonical_fingerprint(
                {
                    "kind": "plc-meshing-entity-set",
                    "source_id": source_id,
                    "source_revision": source_revision,
                    "dimension": dimension,
                    "entities": ordered_keys,
                }
            )
        )
    return tuple(id_banks), tuple(key_banks), tuple(set_ids)


@dataclass(frozen=True, slots=True)
class CompartmentDefinition:
    compartment_id: str
    label_ids: tuple[str, ...]
    material_role: str
    containment_parent_id: str | None = None
    allowed_neighbor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "compartment_id", _identifier(self.compartment_id, "compartment_id")
        )
        labels = tuple(sorted(_identifier(value, "label_id") for value in self.label_ids))
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("label_ids must be unique and non-empty.")
        neighbors = tuple(
            sorted(
                _identifier(value, "allowed_neighbor_id")
                for value in self.allowed_neighbor_ids
            )
        )
        if len(set(neighbors)) != len(neighbors) or self.compartment_id in neighbors:
            raise ValueError("allowed_neighbor_ids must be unique and exclude self.")
        object.__setattr__(self, "label_ids", labels)
        object.__setattr__(
            self, "material_role", _identifier(self.material_role, "material_role")
        )
        object.__setattr__(self, "allowed_neighbor_ids", neighbors)
        if self.containment_parent_id is not None:
            object.__setattr__(
                self,
                "containment_parent_id",
                _identifier(self.containment_parent_id, "containment_parent_id"),
            )


@dataclass(frozen=True, slots=True)
class CompartmentInterfaceDefinition:
    interface_id: str
    first_compartment_id: str
    second_compartment_id: str
    interface_role: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "interface_id", _identifier(self.interface_id, "interface_id")
        )
        object.__setattr__(
            self,
            "first_compartment_id",
            _identifier(self.first_compartment_id, "first_compartment_id"),
        )
        object.__setattr__(
            self,
            "second_compartment_id",
            _identifier(self.second_compartment_id, "second_compartment_id"),
        )
        object.__setattr__(
            self, "interface_role", _identifier(self.interface_role, "interface_role")
        )
        if self.first_compartment_id == self.second_compartment_id:
            raise ValueError("An interface must connect distinct compartments.")
        if not isinstance(self.required, bool):
            raise TypeError("required must be boolean.")

    @property
    def ordered_pair(self) -> tuple[str, str]:
        return _ordered_pair(self.first_compartment_id, self.second_compartment_id)


@dataclass(frozen=True, slots=True)
class CompartmentAdjacencyReport:
    expected_pairs: tuple[tuple[str, str], ...]
    observed_pairs: tuple[tuple[str, str], ...]
    missing_pairs: tuple[tuple[str, str], ...]
    forbidden_pairs: tuple[tuple[str, str], ...]
    successful: bool
    report_id: str


@dataclass(frozen=True, slots=True)
class CompartmentComplex:
    source_revision: str
    compartments: tuple[CompartmentDefinition, ...]
    interfaces: tuple[CompartmentInterfaceDefinition, ...]
    compartment_measures: tuple[tuple[str, float], ...]
    observed_adjacencies: tuple[tuple[str, str], ...]
    adjacency: CompartmentAdjacencyReport = field(init=False)
    complex_id: str = field(init=False)

    def __post_init__(self) -> None:
        revision = _identifier(self.source_revision, "source_revision")
        if not self.compartments or any(
            not isinstance(value, CompartmentDefinition) for value in self.compartments
        ):
            raise ValueError("compartments must contain CompartmentDefinition values.")
        identifiers = tuple(value.compartment_id for value in self.compartments)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Compartment identifiers must be unique.")
        known = set(identifiers)
        for compartment in self.compartments:
            if (
                compartment.containment_parent_id is not None
                and compartment.containment_parent_id not in known
            ):
                raise ValueError(
                    "Containment parents must identify declared compartments."
                )
            if any(
                neighbor not in known for neighbor in compartment.allowed_neighbor_ids
            ):
                raise ValueError("Allowed neighbors must identify declared compartments.")
        if any(
            not isinstance(value, CompartmentInterfaceDefinition)
            for value in self.interfaces
        ):
            raise TypeError(
                "interfaces must contain CompartmentInterfaceDefinition values."
            )
        interface_ids = [value.interface_id for value in self.interfaces]
        interface_pairs = tuple(value.ordered_pair for value in self.interfaces)
        expected = tuple(
            sorted(value.ordered_pair for value in self.interfaces if value.required)
        )
        if len(set(interface_ids)) != len(interface_ids) or len(set(expected)) != len(
            expected
        ):
            raise ValueError(
                "Interface identifiers and required compartment pairs must be unique."
            )
        if any(
            first not in known or second not in known for first, second in interface_pairs
        ):
            raise ValueError("Interface endpoints must identify declared compartments.")
        measures = tuple(
            sorted((str(name), float(value)) for name, value in self.compartment_measures)
        )
        if {name for name, _ in measures} != known or any(
            not np.isfinite(value) or value <= 0.0 for _, value in measures
        ):
            raise ValueError(
                "compartment_measures must provide one positive measure per compartment."
            )
        observed = tuple(
            sorted(_ordered_pair(pair[0], pair[1]) for pair in self.observed_adjacencies)
        )
        if len(set(observed)) != len(observed) or any(
            len(pair) != 2
            or pair[0] == pair[1]
            or pair[0] not in known
            or pair[1] not in known
            for pair in observed
        ):
            raise ValueError(
                "observed_adjacencies must contain unique valid compartment pairs."
            )
        allowed = {
            _ordered_pair(compartment.compartment_id, neighbor)
            for compartment in self.compartments
            for neighbor in compartment.allowed_neighbor_ids
        }
        allowed.update(value.ordered_pair for value in self.interfaces)
        missing = tuple(sorted(set(expected) - set(observed)))
        forbidden = tuple(sorted(set(observed) - allowed))
        report_id = canonical_fingerprint(
            {
                "kind": "compartment-adjacency-report",
                "expected": [list(pair) for pair in expected],
                "observed": [list(pair) for pair in observed],
                "missing": [list(pair) for pair in missing],
                "forbidden": [list(pair) for pair in forbidden],
            }
        )
        adjacency = CompartmentAdjacencyReport(
            expected,
            observed,
            missing,
            forbidden,
            not missing and not forbidden,
            report_id,
        )
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "compartment_measures", measures)
        object.__setattr__(self, "observed_adjacencies", observed)
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(
            self,
            "complex_id",
            canonical_fingerprint(
                {
                    "kind": "compartment-complex",
                    "source_revision": revision,
                    "compartments": [
                        {
                            "id": value.compartment_id,
                            "labels": list(value.label_ids),
                            "role": value.material_role,
                            "parent": value.containment_parent_id,
                            "neighbors": list(value.allowed_neighbor_ids),
                        }
                        for value in self.compartments
                    ],
                    "interfaces": [
                        {
                            "id": value.interface_id,
                            "pair": list(value.ordered_pair),
                            "role": value.interface_role,
                            "required": value.required,
                        }
                        for value in self.interfaces
                    ],
                    "measures": [list(value) for value in measures],
                    "adjacency": report_id,
                }
            ),
        )

    def require_valid_adjacency(self) -> None:
        if not self.adjacency.successful:
            raise ValueError(
                f"Compartment adjacency is invalid: missing={self.adjacency.missing_pairs}, "
                f"forbidden={self.adjacency.forbidden_pairs}."
            )


@dataclass(frozen=True, slots=True)
class CompartmentMeshingSource:
    """Bind categorical voxel compartments and interface sources for meshing.

    A valid image sample owns the physical image of its index cell
    ``i + [-1/2, 1/2]^3``. The full image affine, category ontology,
    compartment complex, extracted interface sources and explicit outer
    boundary are retained as one revision-bound source. Invalid samples and
    ontology categories not assigned to a compartment are outside the finite
    compartment union. This descriptor does not certify that the supplied
    outer surface bounds that union; the native volume route must prove its
    coverage and orientation.
    """

    labels: LabelVolume
    compartments: CompartmentComplex
    outer_surface: SurfaceModel
    interfaces: CompartmentSurfaceResult
    interpretation: CompartmentImageInterpretation = "occupied-voxel-cells"
    outer_geometry: CellGeometrySpec | None = None
    source_id: str = field(init=False)
    source_revision: str = field(init=False)
    coordinate_contract: SpatialCoordinateContract = field(init=False)
    interface_definitions: tuple[tuple[str, str, str, str, bool], ...] = field(init=False)
    scope_entity_id_banks: tuple[np.ndarray, ...] = field(
        init=False, repr=False, compare=False
    )
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...] = field(
        init=False, repr=False, compare=False
    )
    scope_entity_set_ids: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        from ..discretization._cell_geometry import CellGeometrySpec
        from ..discretization._cell_geometry_validity import cell_geometry_id
        from ..imaging._core import LabelVolume
        from ..imaging._segmentation import (
            build_compartment_complex,
            CompartmentSurfaceResult,
            extract_compartment_surfaces,
        )
        from .surface._model import SurfaceModel

        if not isinstance(self.labels, LabelVolume):
            raise TypeError("labels must be a LabelVolume.")
        if not isinstance(self.compartments, CompartmentComplex):
            raise TypeError("compartments must be a CompartmentComplex.")
        if not isinstance(self.outer_surface, SurfaceModel):
            raise TypeError("outer_surface must be a SurfaceModel.")
        if not isinstance(self.interfaces, CompartmentSurfaceResult):
            raise TypeError("interfaces must be a CompartmentSurfaceResult.")
        if self.outer_geometry is not None and not isinstance(
            self.outer_geometry, CellGeometrySpec
        ):
            raise TypeError("outer_geometry must be a CellGeometrySpec or None.")
        if self.interpretation != "occupied-voxel-cells":
            raise ValueError(
                "interpretation must select the supported occupied-voxel-cells model."
            )

        revision = self.labels.label_volume_id
        if self.compartments.source_revision != revision:
            raise ValueError(
                "Compartment complex revision must match the categorical label volume."
            )
        self.compartments.require_valid_adjacency()
        if self.interfaces.complex.complex_id != self.compartments.complex_id:
            raise ValueError(
                "Extracted interfaces must belong to the declared compartment complex."
            )
        contract = self.labels.asset.spatial_affine.coordinate_contract
        if self.labels.asset.values.ndim != 3:
            raise ValueError("Compartment meshing requires a 3D categorical label image.")

        surfaces = (self.outer_surface,) + tuple(
            value.surface for value in self.interfaces.surfaces
        )
        if any(
            surface.metadata.coordinate_contract.spatial_id != contract.spatial_id
            for surface in surfaces
        ):
            raise ValueError(
                "Outer and interface surfaces must share the label image's spatial contract."
            )
        definitions = {
            value.interface_id: value for value in self.compartments.interfaces
        }
        surface_ids: set[str] = set()
        for value in self.interfaces.surfaces:
            definition = definitions.get(value.definition.interface_id)
            if definition != value.definition:
                raise ValueError(
                    "Every extracted compartment interface must match its declared "
                    "source definition."
                )
            if value.definition.interface_id in surface_ids:
                raise ValueError("Extracted compartment interface IDs must be unique.")
            surface_ids.add(value.definition.interface_id)
            if value.surface.metadata.source_revision != revision:
                raise ValueError(
                    "Every extracted interface revision must match the label volume."
                )
        missing = tuple(
            sorted(
                value.interface_id
                for value in self.compartments.interfaces
                if value.required and value.interface_id not in surface_ids
            )
        )
        if missing:
            raise ValueError(
                f"Required compartment interfaces are absent from the source: {missing}."
            )

        rebuilt = build_compartment_complex(
            self.labels,
            self.compartments.compartments,
            self.compartments.interfaces,
        )
        if rebuilt.complex_id != self.compartments.complex_id:
            raise ValueError(
                "Compartment measures or observed adjacency do not match the source labels."
            )
        rebuilt_interfaces = extract_compartment_surfaces(self.labels, rebuilt)
        if rebuilt_interfaces.extraction_id != self.interfaces.extraction_id:
            raise ValueError(
                "Extracted interface geometry does not match the source label revision."
            )

        outer_geometry_id: str | None = None
        if self.outer_geometry is not None:
            if self.outer_geometry.source_mesh_id != self.outer_surface.mesh.mesh_id:
                raise ValueError(
                    "outer_geometry must be bound to the declared outer surface mesh."
                )
            outer_geometry_id = cell_geometry_id(self.outer_geometry)

        interface_definitions = tuple(
            (
                value.interface_id,
                value.first_compartment_id,
                value.second_compartment_id,
                value.interface_role,
                value.required,
            )
            for value in sorted(
                self.compartments.interfaces, key=lambda value: value.interface_id
            )
        )
        source_id = canonical_fingerprint(
            {
                "kind": "compartment-meshing-source",
                "labels": revision,
                "complex": self.compartments.complex_id,
                "outer_surface": self.outer_surface.model_id,
                "interfaces": self.interfaces.extraction_id,
                "interpretation": self.interpretation,
                "coordinates": contract.spatial_id,
                "outer_geometry": outer_geometry_id,
            }
        )
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "coordinate_contract", contract)
        object.__setattr__(self, "interface_definitions", interface_definitions)
        object.__setattr__(self, "source_id", source_id)
        scope_id_banks, scope_key_banks, scope_set_ids = _compartment_scope_banks(
            self.labels,
            self.compartments,
            source_id,
            revision,
            self.outer_surface.model_id,
        )
        object.__setattr__(self, "scope_entity_id_banks", scope_id_banks)
        object.__setattr__(self, "scope_entity_key_banks", scope_key_banks)
        object.__setattr__(self, "scope_entity_set_ids", scope_set_ids)

    @classmethod
    def from_archive_inputs(
        cls,
        labels: LabelVolume,
        compartments: tuple[CompartmentDefinition, ...],
        interfaces: tuple[CompartmentInterfaceDefinition, ...],
        outer_surface: SurfaceModel,
        interpretation: CompartmentImageInterpretation,
        outer_geometry: CellGeometrySpec | None,
        /,
        *,
        expected_source_id: str,
        expected_source_revision: str,
        expected_scope_ids: tuple[str, ...],
    ) -> CompartmentMeshingSource:
        """Rebuild image-derived authority and compare its stored namespace."""
        from .._physical import SpatialCoordinateContract
        from ..discretization._cell_geometry import CellGeometrySpec
        from ..discretization._cell_mesh import CellMesh
        from ..imaging._core import LabelVolume
        from ..imaging._segmentation import build_compartment_complex, extract_compartment_surfaces
        from .surface._contracts import (
            SurfaceInterface, SurfaceMetadata, SurfaceOrientationRepair, SurfaceSelection,
        )
        from .surface._model import SurfaceModel

        labels = LabelVolume.from_stored_fields(labels)
        mesh = CellMesh.from_stored_fields(outer_surface.mesh)
        stored_metadata = outer_surface.metadata
        contract = SpatialCoordinateContract.from_dict(stored_metadata.coordinate_contract.to_dict())
        metadata = SurfaceMetadata(
            source_id=stored_metadata.source_id,
            source_revision=stored_metadata.source_revision,
            coordinate_contract=contract,
            provenance=stored_metadata.provenance,
            cell_tags=stored_metadata.cell_tags,
        )
        if metadata.metadata_id != stored_metadata.metadata_id:
            raise ValueError("Stored outer surface metadata identity changed.")

        def selection(value: SurfaceSelection) -> SurfaceSelection:
            rebuilt = SurfaceSelection(
                value.name, value.cell_global_ids,
                cell_entity_set_id=value.cell_entity_set_id, role=value.role,
            )
            if rebuilt.selection_id != value.selection_id:
                raise ValueError("Stored surface selection identity changed.")
            return rebuilt

        selections = tuple(selection(value) for value in outer_surface.selections)
        restored_interfaces = []
        for value in outer_surface.interfaces:
            item = SurfaceInterface(
                value.name, selection(value.support),
                minus_region=value.minus_region, plus_region=value.plus_region,
            )
            if item.interface_id != value.interface_id:
                raise ValueError("Stored surface interface identity changed.")
            restored_interfaces.append(item)
        repair = outer_surface.orientation_repair
        if repair is not None:
            rebuilt_repair = SurfaceOrientationRepair(
                source_face_indices=repair.source_face_indices,
                orientation_signs=repair.orientation_signs,
                component_ids=repair.component_ids,
                source_topology_id=repair.source_topology_id,
                repaired_topology_id=repair.repaired_topology_id,
            )
            if rebuilt_repair.repair_id != repair.repair_id:
                raise ValueError("Stored surface orientation repair identity changed.")
            repair = rebuilt_repair
        rebuilt_surface = SurfaceModel(
            mesh, metadata, selections=selections,
            interfaces=tuple(restored_interfaces), orientation_repair=repair,
        )
        if rebuilt_surface.model_id != outer_surface.model_id:
            raise ValueError("Stored outer surface identity changed.")
        if outer_geometry is not None:
            outer_geometry = CellGeometrySpec.from_stored_fields(
                outer_geometry,
                source_mesh=mesh,
                source_map=outer_geometry.source_map,
                restriction_source=outer_geometry.restriction_source,
                elements=dict(zip(outer_geometry.block_names, outer_geometry.elements, strict=True)),
            )
        restored_compartments = tuple(
            CompartmentDefinition(
                value.compartment_id, value.label_ids, value.material_role,
                value.containment_parent_id, value.allowed_neighbor_ids,
            )
            for value in compartments
        )
        restored_definitions = tuple(
            CompartmentInterfaceDefinition(
                value.interface_id, value.first_compartment_id,
                value.second_compartment_id, value.interface_role, value.required,
            )
            for value in interfaces
        )
        if restored_compartments != compartments or restored_definitions != interfaces:
            raise ValueError("Stored compartment declarations are not canonical.")
        complex_ = build_compartment_complex(labels, restored_compartments, restored_definitions)
        extracted = extract_compartment_surfaces(labels, complex_)
        source = cls(
            labels, complex_, rebuilt_surface, extracted, interpretation,
            outer_geometry=outer_geometry,
        )
        if (
            source.source_id != expected_source_id
            or source.source_revision != expected_source_revision
            or source.scope_entity_set_ids != expected_scope_ids
        ):
            raise ValueError("Stored compartment source identity or namespace changed.")
        return source

    def _scope_index(self, dimension: int, /) -> int:
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
            raise TypeError("dimension must be an integer.")
        index = int(dimension)
        if index not in (0, 1, 2, 3):
            raise ValueError("Compartment source dimensions must lie in 0..3.")
        return index

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        """Stable int64 source IDs for one PLC entity dimension."""
        return self.scope_entity_id_banks[self._scope_index(dimension)]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        """Stable entity-set identity bound to source revision and exact keys."""
        return self.scope_entity_set_ids[self._scope_index(dimension)]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        """Resolve a numeric scope ID to its exact source key and path."""
        index = self._scope_index(dimension)
        if isinstance(entity_id, bool) or not isinstance(entity_id, (int, np.integer)):
            raise TypeError("entity_id must be an integer source scope ID.")
        identifiers = self.scope_entity_id_banks[index]
        position = int(np.searchsorted(identifiers, int(entity_id)))
        if position >= identifiers.size or int(identifiers[position]) != int(entity_id):
            raise ValueError(
                "entity_id is absent from this compartment source dimension."
            )
        return self.scope_entity_key_banks[index][position]

    def validate_source_integrity(self, /) -> None:
        """Rebuild source bindings and reject stale image/complex/interface content."""
        expected = CompartmentMeshingSource(
            self.labels,
            self.compartments,
            self.outer_surface,
            self.interfaces,
            self.interpretation,
            outer_geometry=self.outer_geometry,
        )
        if (
            expected.source_id != self.source_id
            or expected.source_revision != self.source_revision
            or expected.coordinate_contract.spatial_id
            != self.coordinate_contract.spatial_id
            or expected.interface_definitions != self.interface_definitions
            or expected.scope_entity_set_ids != self.scope_entity_set_ids
            or expected.scope_entity_key_banks != self.scope_entity_key_banks
            or any(
                not np.array_equal(left, right)
                for left, right in zip(
                    expected.scope_entity_id_banks,
                    self.scope_entity_id_banks,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "CompartmentMeshingSource identity must match its retained source content."
            )


__all__ = [
    "CompartmentAdjacencyReport",
    "CompartmentComplex",
    "CompartmentDefinition",
    "CompartmentImageInterpretation",
    "CompartmentInterfaceDefinition",
    "CompartmentMeshingSource",
]
