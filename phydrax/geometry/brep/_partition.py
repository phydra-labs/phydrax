#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import product
from math import prod
from pathlib import Path

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    CADOccurrence,
    CADRevision,
    CADSelectionSet,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._boolean import (
    _assemble,
    _boundary,
    _components,
    _material_owner,
    _membership,
    _rectangles,
    _rectilinear_family,
    _RegionSelection,
    _source_revision,
    BRepBooleanFailure,
    BRepBooleanPolicy,
)
from ._boolean_full_overlay import full_overlay_partition_brep
from ._constructors import BRepTessellationPolicy
from ._intersection_curve import original_trim_intersection_preparation
from ._model import BRepEntityId, BRepModel
from ._projection_contracts import brep_entity_id, BRepEntityDimension
from ._sewing import BRepSewingPolicy


def _text(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _partition_dimension(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Partition topological_dimension must be an integer.")
    if value not in (2, 3):
        raise ValueError("Partition topological_dimension must be 2 or 3.")
    return value


def _solid_occurrence_id(index: int) -> str:
    return f"solid:{index}"


def _face_occurrence_id(solid_index: int, face_index: int) -> str:
    return f"solid:{solid_index}/face:{face_index}"


def _edge_occurrence_id(face_index: int, edge_index: int) -> str:
    return f"face:{face_index}/edge:{edge_index}"


def _entity_text(entity: BRepEntityId) -> str:
    match entity.kind:
        case "vertex":
            dimension = BRepEntityDimension.VERTEX
        case "edge":
            dimension = BRepEntityDimension.EDGE
        case "face":
            dimension = BRepEntityDimension.FACE
        case "solid":
            dimension = BRepEntityDimension.SOLID
        case _:
            raise ValueError(f"Unsupported B-Rep entity kind {entity.kind!r}.")
    return brep_entity_id(
        entity.source_revision,
        dimension,
        entity.index,
        occurrence_path=entity.occurrence_path,
    )


def cad_revision_from_brep_model(model: BRepModel, /) -> CADRevision:
    """Create the exact region and boundary occurrence inventory of a B-Rep."""

    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    occurrences: list[CADOccurrence] = []
    if model.topology.num_solids:
        for solid_index, (face_indices, orientations) in enumerate(
            zip(
                model.topology.solid_faces,
                model.topology.solid_face_orientations,
                strict=True,
            )
        ):
            solid_occurrence_id = _solid_occurrence_id(solid_index)
            occurrences.append(
                CADOccurrence(
                    model.source_revision,
                    solid_occurrence_id,
                    _entity_text(model.solid_ids[solid_index]),
                    "solid",
                    (solid_occurrence_id,),
                )
            )
            for face_index, orientation in zip(face_indices, orientations, strict=True):
                face_occurrence_id = _face_occurrence_id(solid_index, face_index)
                occurrences.append(
                    CADOccurrence(
                        model.source_revision,
                        face_occurrence_id,
                        _entity_text(model.face_ids[face_index]),
                        "face",
                        (solid_occurrence_id, face_occurrence_id),
                        solid_occurrence_id,
                        orientation,
                    )
                )
    else:
        for face_index, wires in enumerate(model.topology.face_wires):
            face_occurrence_id = f"face:{face_index}"
            occurrences.append(
                CADOccurrence(
                    model.source_revision,
                    face_occurrence_id,
                    _entity_text(model.face_ids[face_index]),
                    "face",
                    (face_occurrence_id,),
                )
            )
            edge_orientations: dict[int, int] = {}
            for wire in wires:
                for signed_edge_index in wire:
                    edge_index = abs(signed_edge_index) - 1
                    if edge_index in edge_orientations:
                        raise ValueError(
                            "A planar B-Rep face cannot repeat an edge occurrence."
                        )
                    edge_orientations[edge_index] = 1 if signed_edge_index > 0 else -1
            if set(edge_orientations) != set(model.topology.face_edges[face_index]):
                raise ValueError(
                    "Planar B-Rep face-edge incidence is internally inconsistent."
                )
            for edge_index in model.topology.face_edges[face_index]:
                edge_occurrence_id = _edge_occurrence_id(face_index, edge_index)
                occurrences.append(
                    CADOccurrence(
                        model.source_revision,
                        edge_occurrence_id,
                        _entity_text(model.edge_ids[edge_index]),
                        "edge",
                        (face_occurrence_id, edge_occurrence_id),
                        face_occurrence_id,
                        edge_orientations[edge_index],
                    )
                )
    return CADRevision(
        model.source_revision,
        model.source_id,
        tuple(occurrences),
        model.model_id,
    )


def all_brep_solids(model: BRepModel, /) -> CADSelectionSet:
    """Select every solid occurrence from one semantic B-Rep revision."""

    revision = cad_revision_from_brep_model(model)
    selection = CADSelectionSet.from_revision(
        revision,
        tuple(
            occurrence.occurrence_id
            for occurrence in revision.occurrences
            if occurrence.kind == "solid"
        ),
    )
    if not selection.selectors:
        raise ValueError("The B-Rep model contains no solid occurrences.")
    return selection


def all_brep_faces(model: BRepModel, /) -> CADSelectionSet:
    """Select every root face occurrence from one planar semantic B-Rep revision."""

    revision = cad_revision_from_brep_model(model)
    selection = CADSelectionSet.from_revision(
        revision,
        tuple(
            occurrence.occurrence_id
            for occurrence in revision.occurrences
            if occurrence.kind == "face" and occurrence.parent_occurrence_id is None
        ),
    )
    if not selection.selectors:
        raise ValueError("The B-Rep model contains no root face occurrences.")
    return selection


class BRepPartitionRole(str, Enum):
    """Geometry-only role of a partition operand."""

    REGION = "region"
    VOID = "void"


@dataclass(frozen=True, slots=True)
class BRepPartitionOperand:
    """One persisted exact solid selection participating in a partition."""

    operand_id: str
    model: BRepModel
    role: BRepPartitionRole
    selection: CADSelectionSet | None = None
    target_region_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        operand_id = _text(self.operand_id, "operand_id")
        if not isinstance(self.model, BRepModel):
            raise TypeError("model must be a BRepModel.")
        role = BRepPartitionRole(self.role)
        revision = cad_revision_from_brep_model(self.model)
        selection = self.selection
        if selection is None:
            selection = all_brep_solids(self.model)
        if not isinstance(selection, CADSelectionSet):
            raise TypeError("selection must be a CADSelectionSet or None.")
        selection.require_kind("solid")
        if not selection.selectors:
            raise ValueError("A partition operand must select at least one solid.")
        if selection.revision_id != revision.revision_id:
            raise ValueError("Partition selection belongs to another B-Rep revision.")
        for selector in selection.selectors:
            if revision.select(selector.occurrence_id) != selector:
                raise ValueError(
                    "Partition selection does not match the B-Rep occurrence inventory."
                )
        target_region_ids = tuple(
            _text(value, "target_region_id") for value in self.target_region_ids
        )
        if len(set(target_region_ids)) != len(target_region_ids):
            raise ValueError("A void cannot repeat a target region.")
        if role is BRepPartitionRole.REGION and target_region_ids:
            raise ValueError("Region operands cannot declare void targets.")
        if role is BRepPartitionRole.VOID and not target_region_ids:
            raise ValueError("Void operands require at least one target region.")
        object.__setattr__(self, "operand_id", operand_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "target_region_ids", target_region_ids)


def _operand_selection(operand: BRepPartitionOperand) -> CADSelectionSet:
    selection = operand.selection
    # BRepPartitionOperand.__post_init__ replaces a None selection with all solids.
    if not (selection is not None):
        raise RuntimeError("Internal invariant failed: selection is not None.")
    return selection


@dataclass(frozen=True, slots=True)
class BRepPartitionPolicy:
    """Total region precedence and publication behavior for exact cell selection."""

    region_precedence: tuple[str, ...]
    overwrite: bool = False
    run_parallel: bool = False
    maximum_cells: int = 100_000
    maximum_faces: int = 100_000
    sewing: BRepSewingPolicy = field(default_factory=BRepSewingPolicy)

    def __post_init__(self) -> None:
        precedence = tuple(
            _text(value, "region_precedence entry") for value in self.region_precedence
        )
        if not precedence:
            raise ValueError("Partition policy requires explicit region precedence.")
        if len(set(precedence)) != len(precedence):
            raise ValueError("Region precedence cannot repeat a region.")
        if not isinstance(self.overwrite, bool) or not isinstance(
            self.run_parallel, bool
        ):
            raise TypeError("overwrite and run_parallel must be boolean.")
        for name, value in (
            ("maximum_cells", self.maximum_cells),
            ("maximum_faces", self.maximum_faces),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not isinstance(self.sewing, BRepSewingPolicy):
            raise TypeError("sewing must be a BRepSewingPolicy.")
        object.__setattr__(self, "region_precedence", precedence)


@dataclass(frozen=True, slots=True)
class BRepPartitionPlan:
    """Closed, material-neutral recipe for a host-side exact CAD partition."""

    coordinate_contract: SpatialCoordinateContract
    operands: tuple[BRepPartitionOperand, ...]
    policy: BRepPartitionPolicy
    plan_id: str = field(init=False)
    topological_dimension: int = field(init=False, default=3)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        operands = tuple(self.operands)
        if not operands or any(
            not isinstance(value, BRepPartitionOperand) for value in operands
        ):
            raise TypeError("operands must contain at least one BRepPartitionOperand.")
        if not isinstance(self.policy, BRepPartitionPolicy):
            raise TypeError("policy must be a BRepPartitionPolicy.")
        operand_ids = tuple(value.operand_id for value in operands)
        if len(set(operand_ids)) != len(operand_ids):
            raise ValueError("Partition operand IDs must be unique.")
        region_ids = {
            value.operand_id
            for value in operands
            if value.role is BRepPartitionRole.REGION
        }
        if set(self.policy.region_precedence) != region_ids:
            raise ValueError(
                "region_precedence must name every region operand exactly once."
            )
        for operand in operands:
            if (
                operand.model.coordinate_contract.spatial_id
                != self.coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "Every partition operand must use the plan coordinate contract."
                )
            if (
                operand.role is BRepPartitionRole.VOID
                and not set(operand.target_region_ids) <= region_ids
            ):
                raise ValueError("A void targets an unknown partition region.")
        plan_id = canonical_fingerprint(
            {
                "kind": "brep-partition-plan",
                "coordinate_contract": self.coordinate_contract.spatial_id,
                "topological_dimension": self.topological_dimension,
                "operands": sorted(
                    (
                        value.operand_id,
                        value.model.model_id,
                        _operand_selection(value).selection_id,
                        value.role.value,
                        sorted(value.target_region_ids),
                    )
                    for value in operands
                ),
                "region_precedence": self.policy.region_precedence,
                "overwrite": self.policy.overwrite,
                "run_parallel": self.policy.run_parallel,
                "maximum_cells": self.policy.maximum_cells,
                "maximum_faces": self.policy.maximum_faces,
                "sewing_tolerance": self.policy.sewing.tolerance,
                "maximum_coedges": self.policy.sewing.maximum_coedges,
            }
        )
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "plan_id", plan_id)


@dataclass(frozen=True, slots=True)
class BRepPartitionRegion:
    """Named final region represented by exact semantic B-Rep entity IDs."""

    name: str
    entity_ids: tuple[BRepEntityId, ...]
    topological_dimension: int = 3
    entity_kind: str = field(init=False)

    def __post_init__(self) -> None:
        name = _text(self.name, "region name")
        dimension = _partition_dimension(self.topological_dimension)
        expected_kind = "face" if dimension == 2 else "solid"
        entity_ids = tuple(self.entity_ids)
        if not entity_ids or any(
            not isinstance(value, BRepEntityId) or value.kind != expected_kind
            for value in entity_ids
        ):
            raise TypeError(
                f"A {dimension}D partition region requires BRep {expected_kind} entity IDs."
            )
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("A partition region cannot repeat an entity ID.")
        if len({value.source_revision for value in entity_ids}) != 1:
            raise ValueError(
                "Partition region entities must share one semantic revision."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "entity_ids", entity_ids)
        object.__setattr__(self, "topological_dimension", dimension)
        object.__setattr__(self, "entity_kind", expected_kind)


@dataclass(frozen=True, slots=True)
class BRepPartitionPatch:
    """Named final boundary set with exact one- or two-sided region incidence."""

    name: str
    entity_ids: tuple[BRepEntityId, ...]
    adjacent_region_ids: tuple[str, ...]
    topological_dimension: int = 3
    entity_kind: str = field(init=False)

    def __post_init__(self) -> None:
        name = _text(self.name, "patch name")
        dimension = _partition_dimension(self.topological_dimension)
        expected_kind = "edge" if dimension == 2 else "face"
        entity_ids = tuple(self.entity_ids)
        adjacent = tuple(
            _text(value, "adjacent_region_id") for value in self.adjacent_region_ids
        )
        if not entity_ids or any(
            not isinstance(value, BRepEntityId) or value.kind != expected_kind
            for value in entity_ids
        ):
            raise TypeError(
                f"A {dimension}D partition patch requires BRep {expected_kind} entity IDs."
            )
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("A partition patch cannot repeat an entity ID.")
        if len({value.source_revision for value in entity_ids}) != 1:
            raise ValueError("Partition patch entities must share one semantic revision.")
        if len(adjacent) not in (1, 2):
            raise ValueError("A partition patch must have one or two incident regions.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "entity_ids", entity_ids)
        object.__setattr__(self, "adjacent_region_ids", adjacent)
        object.__setattr__(self, "topological_dimension", dimension)
        object.__setattr__(self, "entity_kind", expected_kind)


@dataclass(frozen=True, slots=True)
class BRepPartitionReport:
    """Exact history, inventory, and publication evidence for one partition."""

    plan_id: str
    model_id: str
    source_revision_id: str
    target_revision_id: str
    history_certificate_id: str
    source_solid_occurrences: int
    source_face_occurrences: int
    target_solids: int
    target_faces: int
    deleted_source_occurrences: int
    created_target_occurrences: int
    source_edge_occurrences: int = 0
    target_edges: int = 0
    topological_dimension: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "source_revision_id",
            _text(self.source_revision_id, "source_revision_id"),
        )
        object.__setattr__(
            self,
            "target_revision_id",
            _text(self.target_revision_id, "target_revision_id"),
        )
        object.__setattr__(
            self,
            "history_certificate_id",
            _text(self.history_certificate_id, "history_certificate_id"),
        )
        counts = (
            self.source_solid_occurrences,
            self.source_face_occurrences,
            self.target_solids,
            self.target_faces,
            self.deleted_source_occurrences,
            self.created_target_occurrences,
            self.source_edge_occurrences,
            self.target_edges,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("Partition report counts must be non-negative integers.")
        dimension = _partition_dimension(self.topological_dimension)
        object.__setattr__(self, "topological_dimension", dimension)

    @property
    def source_region_occurrences(self) -> int:
        return (
            self.source_face_occurrences
            if self.topological_dimension == 2
            else self.source_solid_occurrences
        )

    @property
    def source_patch_occurrences(self) -> int:
        return (
            self.source_edge_occurrences
            if self.topological_dimension == 2
            else self.source_face_occurrences
        )

    @property
    def target_regions(self) -> int:
        return (
            self.target_faces if self.topological_dimension == 2 else self.target_solids
        )

    @property
    def target_patches(self) -> int:
        return self.target_edges if self.topological_dimension == 2 else self.target_faces


@dataclass(frozen=True, slots=True)
class BRepPartitionResult:
    """Published partition plus authoritative exact history and named entity sets."""

    model: BRepModel
    revision: CADRevision
    association_graph: AssociationGraph
    regions: tuple[BRepPartitionRegion, ...]
    patches: tuple[BRepPartitionPatch, ...]
    report: BRepPartitionReport
    topological_dimension: int = 3
    region_entity_kind: str = field(init=False)
    patch_entity_kind: str = field(init=False)
    named_region_entity_ids: tuple[tuple[str, tuple[BRepEntityId, ...]], ...] = field(
        init=False
    )
    named_patch_entity_ids: tuple[tuple[str, tuple[BRepEntityId, ...]], ...] = field(
        init=False
    )
    named_solid_entity_ids: tuple[tuple[str, tuple[BRepEntityId, ...]], ...] = field(
        init=False
    )
    named_face_entity_ids: tuple[tuple[str, tuple[BRepEntityId, ...]], ...] = field(
        init=False
    )
    named_edge_entity_ids: tuple[tuple[str, tuple[BRepEntityId, ...]], ...] = field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model, BRepModel):
            raise TypeError("model must be a BRepModel.")
        if not isinstance(self.revision, CADRevision):
            raise TypeError("revision must be a CADRevision.")
        if not isinstance(self.association_graph, AssociationGraph):
            raise TypeError("association_graph must be an AssociationGraph.")
        if not isinstance(self.report, BRepPartitionReport):
            raise TypeError("report must be a BRepPartitionReport.")
        dimension = _partition_dimension(self.topological_dimension)
        if self.report.topological_dimension != dimension:
            raise ValueError("Partition result and report dimensions must match.")
        regions = tuple(self.regions)
        patches = tuple(self.patches)
        if not regions or any(
            not isinstance(value, BRepPartitionRegion) for value in regions
        ):
            raise TypeError("regions must contain BRepPartitionRegion values.")
        if not patches or any(
            not isinstance(value, BRepPartitionPatch) for value in patches
        ):
            raise TypeError("patches must contain BRepPartitionPatch values.")
        if any(
            value.topological_dimension != dimension for value in (*regions, *patches)
        ):
            raise ValueError("Every region and patch must use the result dimension.")
        if len({value.name for value in regions}) != len(regions):
            raise ValueError("Partition region names must be unique.")
        if len({value.name for value in patches}) != len(patches):
            raise ValueError("Partition patch names must be unique.")
        if self.revision.revision_id != self.model.source_revision:
            raise ValueError("Result revision must use the model semantic revision.")
        if self.association_graph.target_revision != self.revision:
            raise ValueError("Association graph must terminate at the result revision.")
        if (
            self.report.model_id != self.model.model_id
            or self.report.target_revision_id != self.revision.revision_id
            or self.report.source_revision_id
            != self.association_graph.source_revision.revision_id
            or self.report.history_certificate_id
            != self.association_graph.transaction.coverage.certificate_id
        ):
            raise ValueError("Partition report identity does not match the result.")
        if (
            self.report.target_solids != self.model.topology.num_solids
            or self.report.target_faces != self.model.topology.num_faces
            or self.report.target_edges
            != (self.model.topology.num_edges if dimension == 2 else 0)
        ):
            raise ValueError(
                "Partition report inventory does not match the result model."
            )

        region_entities = self.model.face_ids if dimension == 2 else self.model.solid_ids
        patch_entities = self.model.edge_ids if dimension == 2 else self.model.face_ids
        if dimension == 2 and self.model.topology.num_solids:
            raise ValueError("A planar partition model cannot contain solids.")
        region_owner: dict[BRepEntityId, str] = {}
        for region in regions:
            for entity_id in region.entity_ids:
                if entity_id in region_owner:
                    raise ValueError("Partition regions must be disjoint.")
                region_owner[entity_id] = region.name
        if set(region_owner) != set(region_entities):
            raise ValueError("Partition regions must exhaust the result region entities.")
        patch_owner: dict[BRepEntityId, tuple[str, ...]] = {}
        for patch in patches:
            for entity_id in patch.entity_ids:
                if entity_id in patch_owner:
                    raise ValueError("Partition patches must be disjoint.")
                patch_owner[entity_id] = patch.adjacent_region_ids
        if set(patch_owner) != set(patch_entities):
            raise ValueError(
                "Partition patches must exhaust the result boundary entities."
            )

        region_rank = {value.name: index for index, value in enumerate(regions)}
        incidence = (
            self.model.topology.edge_faces
            if dimension == 2
            else self.model.topology.face_solids
        )
        for patch_index, region_indices in enumerate(incidence):
            if len(region_indices) not in (1, 2):
                raise ValueError(
                    "Every partition patch must be incident to one or two regions."
                )
            adjacent = tuple(
                sorted(
                    (region_owner[region_entities[index]] for index in region_indices),
                    key=region_rank.__getitem__,
                )
            )
            if patch_owner[patch_entities[patch_index]] != adjacent:
                raise ValueError("Partition patch incidence does not match the B-Rep.")
            if dimension == 2 and len(region_indices) == 2:
                signs = []
                for face_index in region_indices:
                    matches = tuple(
                        signed_edge
                        for wire in self.model.topology.face_wires[face_index]
                        for signed_edge in wire
                        if abs(signed_edge) - 1 == patch_index
                    )
                    if len(matches) != 1:
                        raise ValueError(
                            "A planar face must contain each incident edge exactly once."
                        )
                    signs.append(1 if matches[0] > 0 else -1)
                if signs[0] == signs[1]:
                    raise ValueError(
                        "A shared planar edge must have opposite face incidence."
                    )

        named_regions = tuple((value.name, value.entity_ids) for value in regions)
        named_patches = tuple((value.name, value.entity_ids) for value in patches)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "patches", patches)
        object.__setattr__(self, "topological_dimension", dimension)
        object.__setattr__(self, "named_region_entity_ids", named_regions)
        object.__setattr__(self, "named_patch_entity_ids", named_patches)
        object.__setattr__(
            self,
            "region_entity_kind",
            "face" if dimension == 2 else "solid",
        )
        object.__setattr__(
            self,
            "patch_entity_kind",
            "edge" if dimension == 2 else "face",
        )
        object.__setattr__(
            self,
            "named_solid_entity_ids",
            named_regions if dimension == 3 else (),
        )
        object.__setattr__(
            self,
            "named_face_entity_ids",
            named_patches if dimension == 3 else named_regions,
        )
        object.__setattr__(
            self,
            "named_edge_entity_ids",
            named_patches if dimension == 2 else (),
        )

    def region(self, name: str, /) -> BRepPartitionRegion:
        expected = _text(name, "region name")
        for region in self.regions:
            if region.name == expected:
                return region
        raise KeyError(f"Unknown partition region {expected!r}.")

    def patch(self, name: str, /) -> BRepPartitionPatch:
        expected = _text(name, "patch name")
        for patch in self.patches:
            if patch.name == expected:
                return patch
        raise KeyError(f"Unknown partition patch {expected!r}.")


class BRepPartitionHistoryError(RuntimeError):
    """Raised when native CAD cannot certify exhaustive region and patch history."""


def _publish_staged(staging: Path, target: Path, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        os.replace(staging, target)
    else:
        os.link(staging, target)
        staging.unlink()
    directory_descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


@dataclass(frozen=True, slots=True)
class _NativePartition:
    """Route-independent exact partition before persistence.

    Lineage pairs address native ``model`` solid and face indices; persistence
    maps them through the certified round-trip correspondence.
    """

    models: tuple[BRepModel, ...]
    model: BRepModel
    solid_regions: tuple[str, ...]
    solid_pairs: tuple[tuple[str, int], ...]
    face_pairs: tuple[tuple[str, int, int], ...]
    certificate: str
    coverage_kind: str


def _partition_selection(
    plan: BRepPartitionPlan,
    operands: tuple[BRepPartitionOperand, ...],
) -> _RegionSelection:
    solids: list[tuple[int, int]] = []
    for position, operand in enumerate(operands):
        selected = {
            selector.occurrence_id for selector in _operand_selection(operand).selectors
        }
        solids.extend(
            (position, solid)
            for solid in range(operand.model.topology.num_solids)
            if _solid_occurrence_id(solid) in selected
        )
    return _RegionSelection(
        tuple(operand.operand_id for operand in operands),
        plan.policy.region_precedence,
        tuple(
            (operand.operand_id, tuple(sorted(operand.target_region_ids)))
            for operand in operands
            if operand.role is BRepPartitionRole.VOID
        ),
        tuple(solids),
    )


def _rectilinear_partition(
    plan: BRepPartitionPlan,
    models: tuple[BRepModel, ...],
    selection: _RegionSelection,
    policy: BRepBooleanPolicy,
) -> _NativePartition:
    """Exact source-coordinate cells of axis-aligned rectangular solids."""
    selected_solids = tuple(
        frozenset(solid for operand, solid in selection.solids if operand == index)
        for index in range(len(models))
    )
    rectangles = _rectangles(models, selected_solids)
    coordinates = tuple(
        tuple(
            sorted(
                {rectangle.lower[d] for rectangle in rectangles}
                | {rectangle.upper[d] for rectangle in rectangles}
            )
        )
        for d in range(3)
    )
    count = prod(len(values) - 1 for values in coordinates)
    if count > policy.maximum_cells:
        raise BRepBooleanFailure("partition arrangement cell budget exhausted")
    membership: dict[tuple[int, int, int], tuple[tuple[int, int], ...]] = {}
    region_cells: dict[str, set[tuple[int, int, int]]] = {
        region: set() for region in plan.policy.region_precedence
    }
    for i, j, k in product(*(range(len(values) - 1) for values in coordinates)):
        cell = i, j, k
        point = (
            (Fraction(coordinates[0][i]) + Fraction(coordinates[0][i + 1])) / 2,
            (Fraction(coordinates[1][j]) + Fraction(coordinates[1][j + 1])) / 2,
            (Fraction(coordinates[2][k]) + Fraction(coordinates[2][k + 1])) / 2,
        )
        members = _membership(point, rectangles, models)
        membership[cell] = members
        owner = _material_owner(selection, members)
        if owner is not None:
            region_cells[owner].add(cell)
    components = []
    owners = []
    for region in plan.policy.region_precedence:
        parts = _components(region_cells[region])
        components.extend(parts)
        owners.extend(region for _ in parts)
    if not components:
        raise BRepBooleanFailure("partition has no surviving material region")
    components_ = tuple(components)
    faces = _boundary(components_, coordinates, rectangles, policy)
    certificate = canonical_fingerprint(
        {
            "kind": "native-cad-partition-arrangement",
            "plan": plan.plan_id,
            "coordinates": coordinates,
            "components": components_,
            "owners": owners,
        }
    )
    model = _assemble(models, coordinates, faces, policy, certificate, shared_faces=True)
    solid_pairs = {
        (f"operand:{operand}/solid:{parent_solid}", solid)
        for solid, component in enumerate(components_)
        for operand, parent_solid in {
            ancestor for cell in component for ancestor in membership[cell]
        }
    }
    # Shared faces are numbered by first occurrence, as `_assemble` allocates them.
    indices: dict[tuple[tuple[int, int, int], ...], int] = {}
    face_pairs: set[tuple[str, int, int]] = set()
    for face in faces:
        index = indices.setdefault(tuple(sorted(face.vertices)), len(indices))
        for operand, parent_solid, parent_face in face.sources:
            face_pairs.add(
                (
                    f"operand:{operand}/solid:{parent_solid}/face:{parent_face}",
                    face.solid,
                    index,
                )
            )
    return _NativePartition(
        models,
        model,
        tuple(owners),
        tuple(sorted(solid_pairs)),
        tuple(sorted(face_pairs)),
        certificate,
        "native-rectilinear-cad-partition",
    )


def _native_arrangement(
    plan: BRepPartitionPlan, policy: BRepBooleanPolicy
) -> _NativePartition:
    """Dispatch exact rectilinear cells or the complete curved face-cell overlay.

    Both routes resolve every open material cell with the same region/void
    owner rule; neither tessellates, samples, or refits a source.
    """
    operands = tuple(sorted(plan.operands, key=lambda operand: operand.operand_id))
    models = tuple(operand.model for operand in operands)
    selection = _partition_selection(plan, operands)
    for model in models:
        if model.geometry is None:
            raise BRepBooleanFailure(
                "authoritative closed-solid geometry is required", (model.model_id,)
            )
    if _rectilinear_family(models):
        return _rectilinear_partition(plan, models, selection, policy)
    overlay = full_overlay_partition_brep(models, selection, policy)
    return _NativePartition(
        models,
        overlay.model,
        overlay.solid_regions,
        overlay.solid_pairs,
        overlay.face_pairs,
        overlay.certificate_id,
        "native-complete-face-cell-material-partition",
    )


def _face_signature(model: BRepModel, face: int) -> tuple[tuple[float, ...], ...]:
    geometry = model.geometry
    if geometry is None:
        raise BRepPartitionHistoryError(
            "Native partition persistence lost exact geometry."
        )
    points = np.asarray(geometry.vertex_points)
    vertices = {
        vertex
        for edge in model.topology.face_edges[face]
        for vertex in geometry.edge_vertices[edge]
    }
    return tuple(
        sorted(tuple(float(value) for value in points[index]) for index in vertices)
    )


def _native_roundtrip_maps(
    original: BRepModel,
    reopened: BRepModel,
    exported: tuple[tuple[str, str], ...] | None = None,
    imported: tuple[tuple[str, str], ...] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if exported is None and imported is None:
        if original.model_id != reopened.model_id:
            raise BRepPartitionHistoryError(
                "Native archive changed its registered model identity."
            )
        face_map = tuple(reopened.face_ids.index(entity) for entity in original.face_ids)
        solid_map = tuple(
            reopened.solid_ids.index(entity) for entity in original.solid_ids
        )
    else:
        if exported is None or imported is None or not exported or not imported:
            raise BRepPartitionHistoryError(
                "External roundtrip identity provenance is unavailable."
            )
        source_refs = dict(exported)
        target_refs = dict(imported)
        if len(source_refs) != len(exported) or len(target_refs) != len(imported):
            raise BRepPartitionHistoryError(
                "Codec provenance repeats a native entity label."
            )
        face_map = _codec_entity_map(
            source_refs,
            target_refs,
            "face",
            original.topology.num_faces,
            reopened.topology.num_faces,
        )
        solid_map = _codec_entity_map(
            source_refs,
            target_refs,
            "solid",
            original.topology.num_solids,
            reopened.topology.num_solids,
        )
    for face, target_face in enumerate(face_map):
        if _face_signature(original, face) != _face_signature(reopened, target_face):
            raise BRepPartitionHistoryError(
                "Persisted face geometry changed after reference mapping."
            )
    first_solids = tuple(
        frozenset(face_map[face] for face in faces)
        for faces in original.topology.solid_faces
    )
    if any(
        first_solids[solid] != frozenset(reopened.topology.solid_faces[target])
        for solid, target in enumerate(solid_map)
    ):
        raise BRepPartitionHistoryError(
            "Persisted solid incidence changed after reference mapping."
        )
    for solid, faces in enumerate(original.topology.solid_faces):
        target = solid_map[solid]
        original_signs = original.topology.solid_face_orientations[solid]
        target_signs = dict(
            zip(
                reopened.topology.solid_faces[target],
                reopened.topology.solid_face_orientations[target],
                strict=True,
            )
        )
        for face, sign in zip(faces, original_signs, strict=True):
            if target_signs[face_map[face]] != sign:
                raise BRepPartitionHistoryError("Persisted solid orientation changed.")
    return solid_map, face_map


def _codec_entity_map(
    exported: dict[str, str],
    imported: dict[str, str],
    kind: str,
    source_count: int,
    target_count: int,
) -> tuple[int, ...]:
    if source_count != target_count:
        raise BRepPartitionHistoryError("External entity inventory cardinality changed.")
    target = {}
    for index in range(target_count):
        label = f"{kind}:{index}"
        if label not in imported or imported[label] in target:
            raise BRepPartitionHistoryError(
                "Reader identity provenance is missing or ambiguous."
            )
        target[imported[label]] = index
    result = []
    for index in range(source_count):
        label = f"{kind}:{index}"
        if label not in exported or exported[label] not in target:
            raise BRepPartitionHistoryError(
                "Writer reference has no exact reader descendant."
            )
        result.append(target[exported[label]])
    if len(set(result)) != source_count:
        raise BRepPartitionHistoryError("External reference mapping is not bijective.")
    return tuple(result)


def _native_partition_graph(
    native: _NativePartition,
    model: BRepModel,
    solid_map: tuple[int, ...],
    face_map: tuple[int, ...],
) -> AssociationGraph:
    source = _source_revision(native.models, native.certificate)
    target = cad_revision_from_brep_model(model)
    pairs = {
        (parent, f"solid:{solid_map[solid]}") for parent, solid in native.solid_pairs
    } | {
        (parent, f"solid:{solid_map[solid]}/face:{face_map[face]}")
        for parent, solid, face in native.face_pairs
    }
    unmapped = {occurrence.occurrence_id for occurrence in target.occurrences} - {
        second for _, second in pairs
    }
    if unmapped:
        raise BRepPartitionHistoryError(
            f"Partition target occurrences have no source ancestry: {sorted(unmapped)!r}."
        )
    transaction = OccurrenceCorrespondenceTransaction(
        native.certificate,
        source.revision_id,
        target.revision_id,
        tuple(
            OccurrenceCorrespondence(first, second, native.certificate)
            for first, second in sorted(pairs)
        ),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(True, True, native.certificate, native.coverage_kind),
    )
    return AssociationGraph(source, target, transaction)


def _native_named_sets(
    plan: BRepPartitionPlan,
    native: _NativePartition,
    model: BRepModel,
    solid_map: tuple[int, ...],
) -> tuple[tuple[BRepPartitionRegion, ...], tuple[BRepPartitionPatch, ...]]:
    owners = [""] * len(native.solid_regions)
    for index, owner in enumerate(native.solid_regions):
        owners[solid_map[index]] = owner
    regions = tuple(
        BRepPartitionRegion(
            region,
            tuple(
                model.solid_ids[index]
                for index, owner in enumerate(owners)
                if owner == region
            ),
        )
        for region in plan.policy.region_precedence
        if region in owners
    )
    rank = {region: index for index, region in enumerate(plan.policy.region_precedence)}
    grouped: dict[tuple[str, ...], list[BRepEntityId]] = {}
    for index, solids in enumerate(model.topology.face_solids):
        adjacent = tuple(
            sorted({owners[solid] for solid in solids}, key=rank.__getitem__)
        )
        grouped.setdefault(adjacent, []).append(model.face_ids[index])
    patches = tuple(
        BRepPartitionPatch(
            f"boundary:{adjacent[0]}"
            if len(adjacent) == 1
            else f"interface:{adjacent[0]}:{adjacent[1]}",
            tuple(entities),
            adjacent,
        )
        for adjacent, entities in sorted(grouped.items())
    )
    return regions, patches


def partition_brep(
    plan: BRepPartitionPlan,
    /,
    *,
    destination: str | Path,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
) -> BRepPartitionResult:
    """Exact native region partition, certified round-trip, and atomic commit.

    Axis-aligned rectangular solids use exact source-coordinate cells; other
    admitted solids use the complete native face-cell overlay, with interface
    faces shared by both owning regions. ``.phx`` persists every exact carrier,
    including intersection branches. External B-Rep text admits only carriers
    it represents exactly and refuses intersection branches before publication.
    """
    # One branch preparation scope covers construction and the round-trip
    # verification, whose reloaded branches share exact identical-query results.
    with original_trim_intersection_preparation(()):
        return _partition_brep(
            plan,
            destination,
            linear_deflection,
            angular_deflection,
            trim_samples_per_edge,
        )


def _partition_brep(
    plan: BRepPartitionPlan,
    destination: str | Path,
    linear_deflection: float,
    angular_deflection: float,
    trim_samples_per_edge: int,
    /,
) -> BRepPartitionResult:
    from dataclasses import replace

    from ..._external_resource import ResourceLimits
    from ...interchange._cad import CadImportPolicy
    from ...interchange._cad_archive import load_brep_archive, save_brep_archive
    from ...interchange._cad_brep_text import read_brep_text, write_brep_text

    if not isinstance(plan, BRepPartitionPlan):
        raise TypeError("plan must be a BRepPartitionPlan.")
    if plan.policy.run_parallel:
        raise ValueError("Native CAD partition does not yet admit parallel execution.")
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() not in {".brep", ".brp", ".phx"}:
        raise ValueError("Partition destinations require .brep, .brp, or .phx.")
    if target.exists() and not plan.policy.overwrite:
        raise FileExistsError(target)
    tessellation = BRepTessellationPolicy(
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
    )
    native = _native_arrangement(
        plan,
        BRepBooleanPolicy(
            maximum_cells=plan.policy.maximum_cells,
            maximum_faces=plan.policy.maximum_faces,
            sewing=plan.policy.sewing,
            tessellation=tessellation,
        ),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.partition-", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    staging = Path(name)
    staging.unlink()
    try:
        if target.suffix.lower() == ".phx":
            save_brep_archive(native.model, staging)
            model = load_brep_archive(staging)
            exported_provenance = imported_provenance = None
        else:
            exported_provenance = write_brep_text(native.model, staging).provenance
            policy = CadImportPolicy(
                plan.coordinate_contract,
                ResourceLimits(64 * 1024 * 1024, 128, 1_000_000, 10_000_000, 0),
                tessellation=tessellation,
            )
            decoded = read_brep_text(
                staging,
                policy,
                trusted_root=target.parent,
                source_length_unit=plan.coordinate_contract.length_unit,
            )
            restored = decoded.model
            imported_provenance = decoded.provenance
            model = BRepModel(
                patches=restored.patches,
                parameter_bounds=restored.parameter_bounds,
                orientation=restored.orientation,
                trim_domains=restored.trim_domains,
                topology=restored.topology,
                coordinate_contract=restored.coordinate_contract,
                mesh_vertices=restored.mesh_vertices,
                mesh_faces=restored.mesh_faces,
                triangle_face_ids=restored.triangle_face_ids,
                triangle_parameters=restored.triangle_parameters,
                tessellation_deviation_bounds=restored.tessellation_deviation_bounds,
                tessellation_normal_bounds=restored.tessellation_normal_bounds,
                mesh_vertex_source_dimensions=restored.mesh_vertex_source_dimensions,
                mesh_vertex_source_indices=restored.mesh_vertex_source_indices,
                mesh_vertex_parameters=restored.mesh_vertex_parameters,
                mesh_chart_restriction_vertices=restored.mesh_chart_restriction_vertices,
                mesh_chart_restriction_edges=restored.mesh_chart_restriction_edges,
                mesh_chart_restriction_endpoint_parameters=(
                    restored.mesh_chart_restriction_endpoint_parameters
                ),
                mesh_chart_restriction_parameters=(
                    restored.mesh_chart_restriction_parameters
                ),
                coedge_deviation_bounds=restored.coedge_deviation_bounds,
                triangle_occurrence_ids=restored.triangle_occurrence_ids,
                vertex_occurrence_ids=restored.vertex_occurrence_ids,
                physical_tags=restored.physical_tags,
                report=replace(restored.report, source_id=str(target)),
                geometry=restored.geometry,
            )
        solid_map, face_map = _native_roundtrip_maps(
            native.model,
            model,
            exported_provenance,
            imported_provenance,
        )
        graph = _native_partition_graph(native, model, solid_map, face_map)
        regions, patches = _native_named_sets(plan, native, model, solid_map)
        source_occurrences = graph.source_revision.occurrences
        mapped_source = {
            edge.source_occurrence_id for edge in graph.transaction.correspondences
        }
        mapped_target = {
            edge.target_occurrence_id for edge in graph.transaction.correspondences
        }
        report = BRepPartitionReport(
            plan.plan_id,
            model.model_id,
            graph.source_revision.revision_id,
            graph.target_revision.revision_id,
            native.certificate,
            sum(occurrence.kind == "solid" for occurrence in source_occurrences),
            sum(occurrence.kind == "face" for occurrence in source_occurrences),
            model.topology.num_solids,
            model.topology.num_faces,
            sum(
                occurrence.occurrence_id not in mapped_source
                for occurrence in source_occurrences
            ),
            sum(
                occurrence.occurrence_id not in mapped_target
                for occurrence in graph.target_revision.occurrences
            ),
        )
        result = BRepPartitionResult(
            model, graph.target_revision, graph, regions, patches, report
        )
        _publish_staged(staging, target, plan.policy.overwrite)
        return result
    finally:
        staging.unlink(missing_ok=True)


__all__ = [
    "BRepPartitionHistoryError",
    "BRepPartitionOperand",
    "BRepPartitionPatch",
    "BRepPartitionPlan",
    "BRepPartitionPolicy",
    "BRepPartitionRegion",
    "BRepPartitionReport",
    "BRepPartitionResult",
    "BRepPartitionRole",
    "all_brep_faces",
    "all_brep_solids",
    "cad_revision_from_brep_model",
    "partition_brep",
]
