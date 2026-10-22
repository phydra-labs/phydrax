#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Semantic volume compartments and their oriented adjacency complex."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import equinox as eqx
import numpy as np

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_geometry_validity as cell_geometry_validity_owner
import phydrax.geometry.surface as surface_owner
import phydrax.imaging as imaging_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._validation import canonical_identifier
from ..typing import checked, parse


CompartmentImageInterpretation: TypeAlias = Literal["occupied-voxel-cells"]


def _ordered_pair(first: str, second: str, /) -> tuple[str, str]:
    return (first, second) if first < second else (second, first)


@dataclass(frozen=True, slots=True)
class CompartmentDefinition:
    compartment_id: str
    label_ids: tuple[str, ...]
    material_role: str
    containment_parent_id: str | None = None
    allowed_neighbor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compartment_id",
            canonical_identifier(self.compartment_id, "compartment_id"),
        )
        labels = tuple(
            sorted(canonical_identifier(value, "label_id") for value in self.label_ids)
        )
        if not labels or len(set(labels)) != len(labels):
            raise ValueError(
                f"CompartmentDefinition.label_ids must be unique and nonempty; got {labels!r}"
            )
        neighbors = tuple(
            sorted(
                canonical_identifier(value, "allowed_neighbor_id")
                for value in self.allowed_neighbor_ids
            )
        )
        if len(set(neighbors)) != len(neighbors) or self.compartment_id in neighbors:
            raise ValueError(
                f"CompartmentDefinition.allowed_neighbor_ids must be unique and exclude self; got {neighbors!r}"
            )
        object.__setattr__(self, "label_ids", labels)
        object.__setattr__(
            self,
            "material_role",
            canonical_identifier(self.material_role, "material_role"),
        )
        object.__setattr__(self, "allowed_neighbor_ids", neighbors)
        if self.containment_parent_id is not None:
            object.__setattr__(
                self,
                "containment_parent_id",
                canonical_identifier(self.containment_parent_id, "containment_parent_id"),
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
            self, "interface_id", canonical_identifier(self.interface_id, "interface_id")
        )
        object.__setattr__(
            self,
            "first_compartment_id",
            canonical_identifier(self.first_compartment_id, "first_compartment_id"),
        )
        object.__setattr__(
            self,
            "second_compartment_id",
            canonical_identifier(self.second_compartment_id, "second_compartment_id"),
        )
        object.__setattr__(
            self,
            "interface_role",
            canonical_identifier(self.interface_role, "interface_role"),
        )
        if self.first_compartment_id == self.second_compartment_id:
            raise ValueError(
                f"CompartmentInterfaceDefinition.second_compartment_id must differ from first_compartment_id; got {self.second_compartment_id!r}"
            )
        if not isinstance(self.required, bool):
            raise TypeError(
                f"CompartmentInterfaceDefinition.required must be boolean; got {self.required!r}"
            )

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


class _ComplexDeclarations(StrictModule):
    """Declared compartments and required interface pairs admitted in source order."""

    known: frozenset[str] = eqx.field(static=True)
    expected: tuple[tuple[str, str], ...] = eqx.field(static=True)


def _validate_complex_declarations(
    compartments: tuple[CompartmentDefinition, ...],
    interfaces: tuple[CompartmentInterfaceDefinition, ...],
    /,
) -> _ComplexDeclarations:
    if not compartments or any(
        not isinstance(value, CompartmentDefinition) for value in compartments
    ):
        raise ValueError(
            f"CompartmentComplex.compartments must contain CompartmentDefinition values; got {compartments!r}"
        )
    identifiers = tuple(value.compartment_id for value in compartments)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            f"CompartmentComplex.compartments must have unique identifiers; got {identifiers!r}"
        )
    known = frozenset(identifiers)
    for compartment in compartments:
        if (
            compartment.containment_parent_id is not None
            and compartment.containment_parent_id not in known
        ):
            raise ValueError(
                f"CompartmentComplex.containment_parent_id must identify a declared compartment; got {compartment.containment_parent_id!r}"
            )
        if any(neighbor not in known for neighbor in compartment.allowed_neighbor_ids):
            raise ValueError(
                f"CompartmentComplex.allowed_neighbor_ids must identify declared compartments; got {compartment.allowed_neighbor_ids!r}"
            )
    if any(not isinstance(value, CompartmentInterfaceDefinition) for value in interfaces):
        raise TypeError(
            f"CompartmentComplex.interfaces must contain CompartmentInterfaceDefinition values; got {interfaces!r}"
        )
    interface_ids = [value.interface_id for value in interfaces]
    interface_pairs = tuple(value.ordered_pair for value in interfaces)
    expected = tuple(sorted(value.ordered_pair for value in interfaces if value.required))
    if len(set(interface_ids)) != len(interface_ids) or len(set(expected)) != len(
        expected
    ):
        raise ValueError(
            f"CompartmentComplex.interfaces must have unique identifiers and required pairs; got ids {interface_ids!r}, pairs {expected!r}"
        )
    if any(
        first not in known or second not in known for first, second in interface_pairs
    ):
        raise ValueError(
            f"CompartmentComplex.interfaces must connect declared compartments; got {interface_pairs!r}"
        )
    return _ComplexDeclarations(known, expected)


class _ComplexPreparation(StrictModule):
    """Normalized measures and the adjacency report of one compartment complex."""

    measures: tuple[tuple[str, float], ...] = eqx.field(static=True)
    observed: tuple[tuple[str, str], ...] = eqx.field(static=True)
    adjacency: CompartmentAdjacencyReport = eqx.field(static=True)


def _prepare_complex_adjacency(
    compartments: tuple[CompartmentDefinition, ...],
    interfaces: tuple[CompartmentInterfaceDefinition, ...],
    measures: tuple[tuple[str, float], ...],
    observed_adjacencies: tuple[tuple[str, str], ...],
    declarations: _ComplexDeclarations,
    /,
) -> _ComplexPreparation:
    measures_ = tuple(sorted((str(name), float(value)) for name, value in measures))
    if {name for name, _ in measures_} != declarations.known or any(
        not np.isfinite(value) or value <= 0.0 for _, value in measures_
    ):
        raise ValueError(
            f"CompartmentComplex.compartment_measures must provide a positive measure for each declared compartment; got {measures_!r}"
        )
    observed = tuple(
        sorted(_ordered_pair(pair[0], pair[1]) for pair in observed_adjacencies)
    )
    if len(set(observed)) != len(observed) or any(
        len(pair) != 2
        or pair[0] == pair[1]
        or pair[0] not in declarations.known
        or pair[1] not in declarations.known
        for pair in observed
    ):
        raise ValueError(
            f"CompartmentComplex.observed_adjacencies must contain unique declared compartment pairs; got {observed!r}"
        )
    allowed = {
        _ordered_pair(compartment.compartment_id, neighbor)
        for compartment in compartments
        for neighbor in compartment.allowed_neighbor_ids
    }
    allowed.update(value.ordered_pair for value in interfaces)
    expected = declarations.expected
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
        expected, observed, missing, forbidden, not missing and not forbidden, report_id
    )
    return _ComplexPreparation(measures_, observed, adjacency)


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
        revision = canonical_identifier(self.source_revision, "source_revision")
        declarations = _validate_complex_declarations(self.compartments, self.interfaces)
        prepared = _prepare_complex_adjacency(
            self.compartments,
            self.interfaces,
            self.compartment_measures,
            self.observed_adjacencies,
            declarations,
        )
        measures = prepared.measures
        observed = prepared.observed
        adjacency = prepared.adjacency
        report_id = adjacency.report_id
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
                f"CompartmentComplex.adjacency must satisfy declared interfaces and neighbors; got "
                f"missing={self.adjacency.missing_pairs}, forbidden={self.adjacency.forbidden_pairs}"
            )


@checked
def complete_interface_definitions(
    complex_: CompartmentComplex, /
) -> tuple[tuple[str, str, str, bool], ...]:
    """Bind observed pairs to declared interface IDs or deterministic adjacency IDs."""
    definitions = {
        value.ordered_pair: (
            value.interface_id,
            value.first_compartment_id,
            value.second_compartment_id,
            value.required,
        )
        for value in complex_.interfaces
    }
    if len(definitions) != len(complex_.interfaces):
        raise ValueError(
            "CompartmentComplex.interfaces must assign one identity per compartment pair; got contradictory interface IDs"
        )
    for first, second in complex_.observed_adjacencies:
        pair = (first, second)
        if pair not in definitions:
            identifier = canonical_fingerprint(
                {
                    "kind": "compartment-material-adjacency",
                    "regions": pair,
                }
            )
            definitions[pair] = (f"adjacency:{identifier}", first, second, True)
    return tuple(sorted(definitions.values()))


class _CompartmentSourceBinding(StrictModule):
    """Image-derived identity and interfaces admitted for one meshing source."""

    interpretation: CompartmentImageInterpretation = eqx.field(static=True)
    revision: str = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract = eqx.field(static=True)
    interface_definitions: tuple[tuple[str, str, str, bool], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)


def _prepare_compartment_source(
    labels: imaging_owner.LabelVolume,
    compartments: CompartmentComplex,
    outer_surface: surface_owner.SurfaceModel,
    interfaces: imaging_owner.CompartmentSurfaceResult,
    interpretation: CompartmentImageInterpretation,
    /,
    *,
    outer_geometry: cell_geometry_owner.CellGeometrySpec | None,
) -> _CompartmentSourceBinding:
    interpretation_ = parse(
        interpretation, CompartmentImageInterpretation, "interpretation"
    )
    revision = labels.label_volume_id
    if compartments.source_revision != revision:
        raise ValueError(
            f"CompartmentMeshingSource.compartments.source_revision must match labels; got {compartments.source_revision!r} != {revision!r}"
        )
    compartments.require_valid_adjacency()
    if interfaces.complex.complex_id != compartments.complex_id:
        raise ValueError(
            f"CompartmentMeshingSource.interfaces.complex must match compartments; got {interfaces.complex.complex_id!r}"
        )
    contract = labels.asset.spatial_affine.coordinate_contract
    surfaces = (outer_surface,) + tuple(value.surface for value in interfaces.surfaces)
    if any(
        value.metadata.coordinate_contract.spatial_id != contract.spatial_id
        for value in surfaces
    ):
        raise ValueError(
            f"CompartmentMeshingSource.geometry must share the image spatial contract; got expected {contract.spatial_id!r}"
        )
    if labels.asset.values.ndim != 3:
        raise ValueError(
            f"CompartmentMeshingSource.labels must be three-dimensional; got rank {labels.asset.values.ndim}"
        )
    for value in interfaces.surfaces:
        if value.surface.metadata.source_revision != revision:
            raise ValueError(
                f"CompartmentMeshingSource.interfaces.source_revision must match labels; got {value.surface.metadata.source_revision!r} != {revision!r}"
            )
    expected_complex = imaging_owner.build_compartment_complex(
        labels, compartments.compartments, compartments.interfaces
    )
    if expected_complex.complex_id != compartments.complex_id:
        raise ValueError(
            f"CompartmentMeshingSource.compartments must match image measures and adjacency; got {compartments.complex_id!r} != {expected_complex.complex_id!r}"
        )
    expected_interfaces = imaging_owner.extract_compartment_surfaces(labels, compartments)
    if expected_interfaces.extraction_id != interfaces.extraction_id:
        raise ValueError(
            f"CompartmentMeshingSource.interfaces must represent the occupied image cells; got {interfaces.extraction_id!r} != {expected_interfaces.extraction_id!r}"
        )
    if tuple(
        (value.definition.interface_id, value.surface.mesh.mesh_id)
        for value in interfaces.surfaces
    ) != tuple(
        (value.definition.interface_id, value.surface.mesh.mesh_id)
        for value in expected_interfaces.surfaces
    ):
        raise ValueError(
            "CompartmentMeshingSource.interfaces.surfaces must match the extraction binding; got different interface or mesh IDs"
        )
    definitions = complete_interface_definitions(compartments)
    if outer_geometry is not None:
        mesh = outer_surface.mesh
        if (
            outer_geometry.block_names != tuple(sorted(block.name for block in mesh.blocks))
            or not np.array_equal(
                np.asarray(outer_geometry.coordinates), np.asarray(mesh.coordinates)
            )
        ):
            raise ValueError(
                "CompartmentMeshingSource.outer_geometry must bind the declared outer mesh; got different coordinates or blocks"
            )
        blocks = {block.name: block for block in mesh.blocks}
        for name, element, route in zip(
            outer_geometry.block_names,
            outer_geometry.elements,
            outer_geometry.geometry_dofs,
            strict=True,
        ):
            block = blocks[name]
            if (
                element.cell_kind != block.cell_kind
                or element.element_id
                != cell_geometry_owner.coordinate_lagrange_element(block.cell_kind, 1).element_id
                or not np.array_equal(np.asarray(route), np.asarray(block.vertices))
            ):
                raise ValueError(
                    "CompartmentMeshingSource.outer_geometry must bind affine facet source laws; got different elements or routes"
                )
    source_id = canonical_fingerprint(
        {
            "kind": "compartment-meshing-source",
            "labels": revision,
            "complex": compartments.complex_id,
            "outer": outer_surface.model_id,
            "interfaces": interfaces.extraction_id,
            "interpretation": interpretation_,
            "coordinates": contract.spatial_id,
            **(
                {
                    "outer_geometry": cell_geometry_validity_owner.cell_geometry_id(outer_geometry),
                    "outer_geometry_content": array_tree_fingerprint(outer_geometry),
                }
                if outer_geometry is not None
                else {}
            ),
        }
    )
    return _CompartmentSourceBinding(
        interpretation_, revision, contract, definitions, source_id
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class CompartmentMeshingSource:
    """Bind a voxel-cell material domain to its source revision and spatial contract.

    A sample at index ``i`` owns the affine image of ``i + [-.5, .5]^3``.
    Invalid samples and ontology labels absent from the complex are exterior.
    The supplied outer surface must bound exactly that domain; generation
    independently certifies this obligation. Interfaces retain their source
    identity and orientation rather than supplying independent overlapping
    region shells.
    """

    labels: imaging_owner.LabelVolume
    compartments: CompartmentComplex
    outer_surface: surface_owner.SurfaceModel
    interfaces: imaging_owner.CompartmentSurfaceResult
    interpretation: CompartmentImageInterpretation = "occupied-voxel-cells"
    outer_geometry: cell_geometry_owner.CellGeometrySpec | None = None
    source_id: str = field(init=False)
    source_revision: str = field(init=False)
    coordinate_contract: SpatialCoordinateContract = field(init=False)
    interface_definitions: tuple[tuple[str, str, str, bool], ...] = field(init=False)

    @checked
    def __init__(
        self,
        labels: imaging_owner.LabelVolume,
        compartments: CompartmentComplex,
        outer_surface: surface_owner.SurfaceModel,
        interfaces: imaging_owner.CompartmentSurfaceResult,
        interpretation: CompartmentImageInterpretation = "occupied-voxel-cells",
        *,
        outer_geometry: cell_geometry_owner.CellGeometrySpec | None = None,
    ) -> None:
        binding = _prepare_compartment_source(
            labels, compartments, outer_surface, interfaces, interpretation,
            outer_geometry=outer_geometry,
        )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "compartments", compartments)
        object.__setattr__(self, "outer_surface", outer_surface)
        object.__setattr__(self, "outer_geometry", outer_geometry)
        object.__setattr__(self, "interfaces", interfaces)
        object.__setattr__(self, "interpretation", binding.interpretation)
        object.__setattr__(self, "interface_definitions", binding.interface_definitions)
        object.__setattr__(self, "source_revision", binding.revision)
        object.__setattr__(self, "coordinate_contract", binding.coordinate_contract)
        object.__setattr__(self, "source_id", binding.source_id)

    def validate_source_integrity(self) -> None:
        """Re-admit the retained segmentation, complex and extracted interface source."""
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
        ):
            raise ValueError(
                f"CompartmentMeshingSource.source_id must match retained source content; got {self.source_id!r} != {expected.source_id!r}"
            )


__all__ = [
    "CompartmentAdjacencyReport",
    "CompartmentComplex",
    "CompartmentDefinition",
    "CompartmentInterfaceDefinition",
    "CompartmentImageInterpretation",
    "CompartmentMeshingSource",
]
