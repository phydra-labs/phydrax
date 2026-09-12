#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from OCP.BOPAlgo import BOPAlgo_CellsBuilder  # ty: ignore[unresolved-import]
from OCP.TopAbs import (  # ty: ignore[unresolved-import]
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SOLID,
)
from OCP.TopExp import TopExp_Explorer  # ty: ignore[unresolved-import]
from OCP.TopoDS import TopoDS, TopoDS_Shape  # ty: ignore[unresolved-import]
from OCP.TopTools import TopTools_ListOfShape  # ty: ignore[unresolved-import]

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
from ._model import BRepEntityId, BRepModel
from ._occt import (
    _shape_digest,
    import_brep,
    persist_occt_shape,
    read_occt_shape,
)


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
    return f"{entity.source_revision}:{entity.kind}:{entity.index}"


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
    if not occurrences:
        raise ValueError("A partition operand B-Rep must contain a solid or face.")
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


@dataclass(frozen=True, slots=True)
class BRepPartitionPolicy:
    """Total region precedence and publication behavior for exact cell selection."""

    region_precedence: tuple[str, ...]
    overwrite: bool = False
    run_parallel: bool = False

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
                        value.selection.selection_id,
                        value.role.value,
                        sorted(value.target_region_ids),
                    )
                    for value in operands
                ),
                "region_precedence": self.policy.region_precedence,
                "overwrite": self.policy.overwrite,
                "run_parallel": self.policy.run_parallel,
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
    """Raised when OCCT cannot certify exhaustive region and patch history."""


@dataclass(frozen=True, slots=True)
class _LoadedModel:
    model: BRepModel
    revision: CADRevision
    shape: TopoDS_Shape
    solids: tuple[Any, ...]
    faces: tuple[Any, ...]
    edges: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _SourceFace:
    occurrence: CADOccurrence
    shape: Any


@dataclass(frozen=True, slots=True)
class _SourceSolid:
    operand_id: str
    occurrence: CADOccurrence
    shape: Any
    faces: tuple[_SourceFace, ...]


@dataclass(frozen=True, slots=True)
class _History:
    target_indices: tuple[int, ...]
    modified_count: int
    generated_count: int
    deleted: bool


def _explore_unique(shape: Any, kind: Any, caster: Any) -> tuple[Any, ...]:
    explorer = TopExp_Explorer(shape, kind)
    entities: list[Any] = []
    while explorer.More():
        candidate = caster(explorer.Current())
        if not any(value.IsSame(candidate) for value in entities):
            entities.append(candidate)
        explorer.Next()
    return tuple(entities)


def _shape_index(entities: Sequence[Any], candidate: Any) -> int:
    for index, entity in enumerate(entities):
        if entity.IsSame(candidate):
            return index
    raise BRepPartitionHistoryError(
        "OCCT history referenced an entity outside the exact partition inventory."
    )


def _oriented_face(solid: Any, face: Any) -> Any:
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    while explorer.More():
        candidate = TopoDS.Face_s(explorer.Current())
        if candidate.IsSame(face):
            return candidate
        explorer.Next()
    raise BRepPartitionHistoryError(
        "Persisted solid incidence disagrees with its exact face inventory."
    )


def _shape_list(shapes: Sequence[Any]) -> TopTools_ListOfShape:
    result = TopTools_ListOfShape()
    for shape in shapes:
        result.Append(shape)
    return result


def _members(
    builder: BOPAlgo_CellsBuilder,
    shape: Any,
    atoms: tuple[Any, ...],
) -> frozenset[int]:
    builder.RemoveAllFromResult()
    builder.AddToResult(_shape_list((shape,)), TopTools_ListOfShape())
    selected = _explore_unique(builder.Shape(), TopAbs_SOLID, TopoDS.Solid_s)
    return frozenset(_shape_index(atoms, value) for value in selected)


def _flatten_history(values: Sequence[Any], kind: Any, caster: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        candidates = (
            (caster(value),)
            if value.ShapeType() == kind
            else _explore_unique(value, kind, caster)
        )
        for candidate in candidates:
            if not any(existing.IsSame(candidate) for existing in result):
                result.append(candidate)
    return tuple(result)


def _capture_history(
    builder: BOPAlgo_CellsBuilder,
    source: Any,
    targets: tuple[Any, ...],
    kind: Any,
    caster: Any,
) -> _History:
    modified = tuple(builder.Modified(source))
    generated = tuple(builder.Generated(source))
    deleted = bool(builder.IsDeleted(source))
    candidates = list(_flatten_history(modified, kind, caster))
    candidates.extend(_flatten_history(generated, kind, caster))
    if any(value.IsSame(source) for value in targets):
        candidates.append(caster(source))
    indices = tuple(
        sorted(
            {
                _shape_index(targets, candidate)
                for candidate in candidates
                if any(target.IsSame(candidate) for target in targets)
            }
        )
    )
    history_entities = _flatten_history((*modified, *generated), kind, caster)
    if any(
        not any(target.IsSame(candidate) for target in targets)
        for candidate in history_entities
    ):
        raise BRepPartitionHistoryError(
            "OCCT returned non-final history without exhaustive final resolution."
        )
    if not indices and not deleted:
        raise BRepPartitionHistoryError(
            "OCCT supplied neither exact descendants nor a deletion decision."
        )
    return _History(indices, len(modified), len(generated), deleted)


def _load_model(model: BRepModel) -> _LoadedModel:
    source = Path(model.source_id).expanduser().resolve()
    if model.report.source_format != "brep" or not source.is_file():
        raise ValueError("Partition operands must be persisted native BREP models.")
    shape, source_format, source_digest = read_occt_shape(source)
    if source_format != "brep" or source_digest != model.source_digest:
        raise ValueError("Persisted partition operand identity has changed.")
    solids = _explore_unique(shape, TopAbs_SOLID, TopoDS.Solid_s)
    faces = _explore_unique(shape, TopAbs_FACE, TopoDS.Face_s)
    edges = _explore_unique(shape, TopAbs_EDGE, TopoDS.Edge_s)
    if (
        len(solids) != model.topology.num_solids
        or len(faces) != len(model.face_ids)
        or len(edges) != len(model.edge_ids)
    ):
        raise ValueError("Persisted BREP inventory no longer matches its model.")
    return _LoadedModel(
        model,
        cad_revision_from_brep_model(model),
        shape,
        solids,
        faces,
        edges,
    )


def _composite_sources(
    plan: BRepPartitionPlan,
    loaded: dict[tuple[str, str], _LoadedModel],
) -> tuple[CADRevision, tuple[_SourceSolid, ...], dict[str, tuple[Any, ...]]]:
    revision_id = canonical_fingerprint(
        {
            "kind": "brep-partition-source-revision",
            "plan_id": plan.plan_id,
        }
    )
    occurrences: list[CADOccurrence] = []
    source_solids: list[_SourceSolid] = []
    operand_shapes: dict[str, tuple[Any, ...]] = {}
    for operand in sorted(plan.operands, key=lambda value: value.operand_id):
        item = loaded[(operand.model.source_id, operand.model.source_digest)]
        selected_shapes: list[Any] = []
        for selector in operand.selection.selectors:
            source_solid_index = next(
                index
                for index in range(len(item.solids))
                if item.revision.select(_solid_occurrence_id(index)) == selector
            )
            solid_occurrence_id = (
                f"operand:{operand.operand_id}:solid:{source_solid_index}"
            )
            solid_occurrence = CADOccurrence(
                revision_id,
                solid_occurrence_id,
                selector.entity_id,
                "solid",
                (solid_occurrence_id,),
            )
            occurrences.append(solid_occurrence)
            faces: list[_SourceFace] = []
            for face_index, orientation in zip(
                item.model.topology.solid_faces[source_solid_index],
                item.model.topology.solid_face_orientations[source_solid_index],
                strict=True,
            ):
                face_occurrence_id = f"{solid_occurrence_id}/face:{face_index}"
                original = item.revision.select(
                    _face_occurrence_id(source_solid_index, face_index)
                )
                occurrence = CADOccurrence(
                    revision_id,
                    face_occurrence_id,
                    original.entity_id,
                    "face",
                    (solid_occurrence_id, face_occurrence_id),
                    solid_occurrence_id,
                    orientation,
                )
                occurrences.append(occurrence)
                faces.append(
                    _SourceFace(
                        occurrence,
                        _oriented_face(
                            item.solids[source_solid_index], item.faces[face_index]
                        ),
                    )
                )
            shape = item.solids[source_solid_index]
            selected_shapes.append(shape)
            source_solids.append(
                _SourceSolid(
                    operand.operand_id,
                    solid_occurrence,
                    shape,
                    tuple(faces),
                )
            )
        operand_shapes[operand.operand_id] = tuple(selected_shapes)
    if len({value.occurrence_id for value in occurrences}) != len(occurrences):
        raise ValueError("Partition operand IDs produce ambiguous CAD occurrences.")
    revision = CADRevision(
        revision_id,
        f"brep-partition-input:{plan.plan_id}",
        tuple(occurrences),
        plan.plan_id,
    )
    return revision, tuple(source_solids), operand_shapes


def _classify_atoms(
    plan: BRepPartitionPlan,
    operand_members: dict[str, frozenset[int]],
    atom_count: int,
) -> tuple[str | None, ...]:
    voids = tuple(
        value for value in plan.operands if value.role is BRepPartitionRole.VOID
    )
    owners: list[str | None] = []
    for atom_index in range(atom_count):
        owner = next(
            (
                region_id
                for region_id in plan.policy.region_precedence
                if atom_index in operand_members[region_id]
            ),
            None,
        )
        if owner is not None and any(
            owner in void.target_region_ids
            and atom_index in operand_members[void.operand_id]
            for void in voids
        ):
            owner = None
        owners.append(owner)
    if not any(value is not None for value in owners):
        raise ValueError("Partition policy removes every result solid.")
    if {value for value in owners if value is not None} != set(
        plan.policy.region_precedence
    ):
        raise ValueError("Every declared partition region must retain a solid.")
    return tuple(owners)


def _build_final_shape(
    plan: BRepPartitionPlan,
    builder: BOPAlgo_CellsBuilder,
    operand_shapes: dict[str, tuple[Any, ...]],
) -> TopoDS_Shape:
    voids = tuple(
        value for value in plan.operands if value.role is BRepPartitionRole.VOID
    )
    builder.RemoveAllFromResult()
    for rank, region_id in enumerate(plan.policy.region_precedence):
        higher = plan.policy.region_precedence[:rank]
        avoid = tuple(
            shape for higher_id in higher for shape in operand_shapes[higher_id]
        ) + tuple(
            shape
            for void in voids
            if region_id in void.target_region_ids
            for shape in operand_shapes[void.operand_id]
        )
        for shape in operand_shapes[region_id]:
            builder.AddToResult(
                _shape_list((shape,)),
                _shape_list(avoid),
                rank + 1,
            )
    builder.RemoveInternalBoundaries()
    result = builder.Shape()
    if result.IsNull():
        raise ValueError("OCCT produced an empty partition result.")
    return result


def _final_solid_owners(
    final_solids: tuple[Any, ...],
    atoms: tuple[Any, ...],
    atom_owners: tuple[str | None, ...],
) -> tuple[str, ...]:
    oriented_atom_faces = tuple(
        (owner, _explore_unique(atom, TopAbs_FACE, TopoDS.Face_s))
        for atom, owner in zip(atoms, atom_owners, strict=True)
        if owner is not None
    )
    owners: list[str] = []
    for solid in final_solids:
        evidence: set[str] = set()
        explorer = TopExp_Explorer(solid, TopAbs_FACE)
        while explorer.More():
            final_face = TopoDS.Face_s(explorer.Current())
            for owner, faces in oriented_atom_faces:
                if any(final_face.IsEqual(face) for face in faces):
                    evidence.add(owner)
            explorer.Next()
        if len(evidence) != 1:
            raise BRepPartitionHistoryError(
                "Final solid material ownership lacks unique exact cell incidence."
            )
        owners.append(evidence.pop())
    if set(owners) != {value for value in atom_owners if value is not None}:
        raise BRepPartitionHistoryError(
            "Final solid ownership does not exhaust retained exact cells."
        )
    return tuple(owners)


def _exact_roundtrip_map(
    original: tuple[Any, ...], reopened: tuple[Any, ...], kind: str
) -> tuple[int, ...]:
    if len(original) != len(reopened):
        raise BRepPartitionHistoryError(
            f"Persisted partition changed the exact {kind} inventory."
        )
    original_digests = tuple(_shape_digest(value) for value in original)
    reopened_digests = tuple(_shape_digest(value) for value in reopened)
    if len(set(original_digests)) != len(original_digests) or len(
        set(reopened_digests)
    ) != len(reopened_digests):
        raise BRepPartitionHistoryError(
            f"Exact {kind} identity is ambiguous after persistence."
        )
    if set(original_digests) != set(reopened_digests):
        raise BRepPartitionHistoryError(
            f"Persisted partition changed exact {kind} topology or geometry."
        )
    return tuple(reopened_digests.index(value) for value in original_digests)


def _verify_roundtrip(
    original_shape: Any,
    reopened_shape: Any,
    model: BRepModel,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[int, ...], tuple[int, ...]]:
    original_solids = _explore_unique(original_shape, TopAbs_SOLID, TopoDS.Solid_s)
    original_faces = _explore_unique(original_shape, TopAbs_FACE, TopoDS.Face_s)
    reopened_solids = _explore_unique(reopened_shape, TopAbs_SOLID, TopoDS.Solid_s)
    reopened_faces = _explore_unique(reopened_shape, TopAbs_FACE, TopoDS.Face_s)
    solid_map = _exact_roundtrip_map(original_solids, reopened_solids, "solid")
    face_map = _exact_roundtrip_map(original_faces, reopened_faces, "face")
    if (
        len(reopened_solids) != model.topology.num_solids
        or len(reopened_faces) != model.topology.num_faces
    ):
        raise BRepPartitionHistoryError(
            "Reopened partition model does not match its native BREP inventory."
        )
    for original_solid_index, solid in enumerate(original_solids):
        mapped_faces: list[int] = []
        mapped_orientations: list[int] = []
        explorer = TopExp_Explorer(solid, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            original_face_index = _shape_index(original_faces, face)
            mapped_faces.append(face_map[original_face_index])
            global_face = original_faces[original_face_index]
            mapped_orientations.append(
                1 if face.Orientation() == global_face.Orientation() else -1
            )
            explorer.Next()
        reopened_solid_index = solid_map[original_solid_index]
        if (
            tuple(mapped_faces) != model.topology.solid_faces[reopened_solid_index]
            or tuple(mapped_orientations)
            != model.topology.solid_face_orientations[reopened_solid_index]
        ):
            raise BRepPartitionHistoryError(
                "Reopened partition changed exact solid-face incidence."
            )
    return original_solids, original_faces, solid_map, face_map


def _identity_association_graph(
    plan: BRepPartitionPlan,
    source_revision: CADRevision,
    source_solids: tuple[_SourceSolid, ...],
    target_model: BRepModel,
    final_solids: tuple[Any, ...],
    final_faces: tuple[Any, ...],
    solid_map: tuple[int, ...],
    face_map: tuple[int, ...],
    /,
) -> tuple[AssociationGraph, str]:
    """Certify a one-solid no-op partition through exact persisted topology."""

    if len(source_solids) != 1 or len(final_solids) != 1 or len(solid_map) != 1:
        raise BRepPartitionHistoryError(
            "Identity BRep partition requires exactly one source and target solid."
        )
    target_revision = cad_revision_from_brep_model(target_model)
    source_solid = source_solids[0]
    target_solid_index = solid_map[0]
    correspondences = [
        OccurrenceCorrespondence(
            source_solid.occurrence.occurrence_id,
            _solid_occurrence_id(target_solid_index),
            canonical_fingerprint(
                {
                    "kind": "exact-brep-identity-solid",
                    "plan_id": plan.plan_id,
                    "source": source_solid.occurrence.occurrence_id,
                    "target": _solid_occurrence_id(target_solid_index),
                }
            ),
        )
    ]
    for source_face in source_solid.faces:
        original_face_index = _shape_index(final_faces, source_face.shape)
        target_face_index = face_map[original_face_index]
        target_solids = target_model.topology.face_solids[target_face_index]
        if target_solids != (target_solid_index,):
            raise BRepPartitionHistoryError(
                "Identity BRep partition changed exact solid-face incidence."
            )
        target_face_id = _face_occurrence_id(target_solid_index, target_face_index)
        correspondences.append(
            OccurrenceCorrespondence(
                source_face.occurrence.occurrence_id,
                target_face_id,
                canonical_fingerprint(
                    {
                        "kind": "exact-brep-identity-face",
                        "plan_id": plan.plan_id,
                        "source": source_face.occurrence.occurrence_id,
                        "target": target_face_id,
                    }
                ),
            )
        )
    target_ids = {occurrence.occurrence_id for occurrence in target_revision.occurrences}
    correspondence_targets = {
        correspondence.target_occurrence_id for correspondence in correspondences
    }
    if correspondence_targets != target_ids:
        raise BRepPartitionHistoryError(
            "Identity BRep partition does not exhaust its target topology."
        )
    certificate_id = canonical_fingerprint(
        {
            "kind": "exact-brep-identity-persistence",
            "plan_id": plan.plan_id,
            "source_revision": source_revision.revision_id,
            "target_revision": target_revision.revision_id,
            "correspondences": sorted(
                (
                    correspondence.source_occurrence_id,
                    correspondence.target_occurrence_id,
                    correspondence.evidence_id,
                )
                for correspondence in correspondences
            ),
        }
    )
    transaction = OccurrenceCorrespondenceTransaction(
        canonical_fingerprint(
            {
                "kind": "brep-identity-correspondence-transaction",
                "plan_id": plan.plan_id,
                "certificate": certificate_id,
            }
        ),
        source_revision.revision_id,
        target_revision.revision_id,
        tuple(correspondences),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True,
            True,
            certificate_id,
            "OCP.exact-persistence-identity",
        ),
    )
    return AssociationGraph(source_revision, target_revision, transaction), certificate_id


def _association_graph(
    plan: BRepPartitionPlan,
    source_revision: CADRevision,
    source_solids: tuple[_SourceSolid, ...],
    target_model: BRepModel,
    final_solids: tuple[Any, ...],
    final_faces: tuple[Any, ...],
    solid_map: tuple[int, ...],
    face_map: tuple[int, ...],
    builder: BOPAlgo_CellsBuilder,
) -> tuple[AssociationGraph, str]:
    if not builder.HasHistory():
        raise BRepPartitionHistoryError(
            "OCCT did not provide the required live Boolean history."
        )
    target_revision = cad_revision_from_brep_model(target_model)
    edges: list[OccurrenceCorrespondence] = []
    evidence_rows: list[tuple[object, ...]] = []
    for source_solid in source_solids:
        solid_history = _capture_history(
            builder,
            source_solid.shape,
            final_solids,
            TopAbs_SOLID,
            TopoDS.Solid_s,
        )
        mapped_solids = tuple(solid_map[index] for index in solid_history.target_indices)
        evidence_rows.append(
            (
                source_solid.occurrence.occurrence_id,
                mapped_solids,
                solid_history.modified_count,
                solid_history.generated_count,
                solid_history.deleted,
            )
        )
        for target_solid_index in mapped_solids:
            target_id = _solid_occurrence_id(target_solid_index)
            evidence_id = canonical_fingerprint(
                {
                    "kind": "occt-exact-solid-history",
                    "plan_id": plan.plan_id,
                    "source": source_solid.occurrence.occurrence_id,
                    "target": target_id,
                }
            )
            edges.append(
                OccurrenceCorrespondence(
                    source_solid.occurrence.occurrence_id,
                    target_id,
                    evidence_id,
                )
            )
        for source_face in source_solid.faces:
            face_history = _capture_history(
                builder,
                source_face.shape,
                final_faces,
                TopAbs_FACE,
                TopoDS.Face_s,
            )
            mapped_faces = tuple(face_map[index] for index in face_history.target_indices)
            evidence_rows.append(
                (
                    source_face.occurrence.occurrence_id,
                    mapped_faces,
                    face_history.modified_count,
                    face_history.generated_count,
                    face_history.deleted,
                )
            )
            for target_face_index in mapped_faces:
                for target_solid_index in target_model.topology.face_solids[
                    target_face_index
                ]:
                    target_id = _face_occurrence_id(target_solid_index, target_face_index)
                    evidence_id = canonical_fingerprint(
                        {
                            "kind": "occt-exact-face-history",
                            "plan_id": plan.plan_id,
                            "source": source_face.occurrence.occurrence_id,
                            "target": target_id,
                        }
                    )
                    edges.append(
                        OccurrenceCorrespondence(
                            source_face.occurrence.occurrence_id,
                            target_id,
                            evidence_id,
                        )
                    )
    pairs = {
        (value.source_occurrence_id, value.target_occurrence_id): value for value in edges
    }
    edges = sorted(
        pairs.values(),
        key=lambda value: (
            value.source_occurrence_id,
            value.target_occurrence_id,
        ),
    )
    edge_sources = {value.source_occurrence_id for value in edges}
    edge_targets = {value.target_occurrence_id for value in edges}
    target_ids = {value.occurrence_id for value in target_revision.occurrences}
    source_ids = {value.occurrence_id for value in source_revision.occurrences}
    if (
        not {_solid_occurrence_id(index) for index in range(len(final_solids))}
        <= edge_targets
    ):
        raise BRepPartitionHistoryError(
            "OCCT solid history does not exhaust the final solid inventory."
        )
    certificate_id = canonical_fingerprint(
        {
            "kind": "occt-cells-builder-exhaustive-history",
            "plan_id": plan.plan_id,
            "source_revision": source_revision.revision_id,
            "target_revision": target_revision.revision_id,
            "history": sorted(evidence_rows),
            "edges": sorted(pairs),
            "deleted": sorted(source_ids - edge_sources),
            "created": sorted(target_ids - edge_targets),
        }
    )
    coverage = AssociationCoverageEvidence(
        True,
        True,
        certificate_id,
        "OCP.BOPAlgo_CellsBuilder",
    )
    transaction_id = canonical_fingerprint(
        {
            "kind": "brep-partition-correspondence-transaction",
            "plan_id": plan.plan_id,
            "certificate": certificate_id,
        }
    )
    transaction = OccurrenceCorrespondenceTransaction(
        transaction_id,
        source_revision.revision_id,
        target_revision.revision_id,
        tuple(edges),
        frozenset(),
        frozenset(),
        coverage,
    )
    return AssociationGraph(source_revision, target_revision, transaction), certificate_id


def _regions_and_patches(
    plan: BRepPartitionPlan,
    model: BRepModel,
    original_owners: tuple[str, ...],
    solid_map: tuple[int, ...],
) -> tuple[tuple[BRepPartitionRegion, ...], tuple[BRepPartitionPatch, ...]]:
    model_owners = [""] * len(original_owners)
    for original_index, model_index in enumerate(solid_map):
        model_owners[model_index] = original_owners[original_index]
    regions = tuple(
        BRepPartitionRegion(
            region_id,
            tuple(
                model.solid_ids[index]
                for index, owner in enumerate(model_owners)
                if owner == region_id
            ),
        )
        for region_id in plan.policy.region_precedence
    )
    rank = {
        region_id: index for index, region_id in enumerate(plan.policy.region_precedence)
    }
    groups: dict[tuple[str, ...], list[BRepEntityId]] = {}
    for face_index, solid_indices in enumerate(model.topology.face_solids):
        if len(solid_indices) not in (1, 2):
            raise BRepPartitionHistoryError(
                "Final partition face is not exactly one- or two-sided."
            )
        adjacent = tuple(
            sorted(
                (model_owners[index] for index in solid_indices),
                key=rank.__getitem__,
            )
        )
        groups.setdefault(adjacent, []).append(model.face_ids[face_index])
    patches: list[BRepPartitionPatch] = []
    for adjacent in sorted(groups, key=lambda value: tuple(rank[item] for item in value)):
        if len(adjacent) == 1:
            name = f"boundary:{adjacent[0]}"
        elif adjacent[0] == adjacent[1]:
            name = f"internal:{adjacent[0]}"
        else:
            name = f"interface:{adjacent[0]}:{adjacent[1]}"
        patches.append(BRepPartitionPatch(name, tuple(groups[adjacent]), adjacent))
    return regions, tuple(patches)


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


def partition_brep(
    plan: BRepPartitionPlan,
    /,
    *,
    destination: str | Path,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
) -> BRepPartitionResult:
    """Execute, certify, persist, reopen, and atomically publish an exact partition."""

    if not isinstance(plan, BRepPartitionPlan):
        raise TypeError("plan must be a BRepPartitionPlan.")
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() not in {".brep", ".brp"}:
        raise ValueError("A partition destination requires a .brep or .brp suffix.")
    if target.exists() and not plan.policy.overwrite:
        raise FileExistsError(target)
    loaded: dict[tuple[str, str], _LoadedModel] = {}
    for operand in plan.operands:
        key = (operand.model.source_id, operand.model.source_digest)
        if key not in loaded:
            loaded[key] = _load_model(operand.model)
    source_revision, source_solids, operand_shapes = _composite_sources(plan, loaded)

    arguments: list[Any] = []
    for source_solid in source_solids:
        if not any(value.IsSame(source_solid.shape) for value in arguments):
            arguments.append(source_solid.shape)
    identity_partition = len(arguments) == 1 and len(source_solids) == 1
    if identity_partition:
        builder = None
        atoms = (source_solids[0].shape,)
        operand_members = {
            operand_id: frozenset(
                0 for shape in shapes if shape.IsSame(source_solids[0].shape)
            )
            for operand_id, shapes in operand_shapes.items()
        }
        final_shape = source_solids[0].shape
    else:
        builder = BOPAlgo_CellsBuilder()
        builder.SetRunParallel(plan.policy.run_parallel)
        builder.SetNonDestructive(True)
        builder.SetToFillHistory(True)
        for argument in arguments:
            builder.AddArgument(argument)
        builder.Perform()
        if builder.HasErrors():
            raise RuntimeError("OCCT failed to construct the exact partition cells.")
        atoms = _explore_unique(builder.GetAllParts(), TopAbs_SOLID, TopoDS.Solid_s)
        if not atoms:
            raise RuntimeError("OCCT produced no solid partition cells.")
        operand_members = {
            operand_id: frozenset(
                atom_index
                for shape in shapes
                for atom_index in _members(builder, shape, atoms)
            )
            for operand_id, shapes in operand_shapes.items()
        }
        final_shape = _build_final_shape(plan, builder, operand_shapes)
    atom_owners = _classify_atoms(plan, operand_members, len(atoms))
    final_solids = _explore_unique(final_shape, TopAbs_SOLID, TopoDS.Solid_s)
    final_faces = _explore_unique(final_shape, TopAbs_FACE, TopoDS.Face_s)
    if not final_solids or not final_faces:
        raise RuntimeError("OCCT produced an incomplete partition result.")
    original_owners = _final_solid_owners(final_solids, atoms, atom_owners)

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.partition-",
        suffix=".brep",
        dir=target.parent,
    )
    os.close(descriptor)
    staging = Path(staging_name)
    staging.unlink()
    try:
        staged_model = persist_occt_shape(
            final_shape,
            staging,
            coordinate_contract=plan.coordinate_contract,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            trim_samples_per_edge=trim_samples_per_edge,
        )
        reopened_shape, source_format, source_digest = read_occt_shape(staging)
        if source_format != "brep" or source_digest != staged_model.source_digest:
            raise BRepPartitionHistoryError(
                "Staged partition artifact failed exact identity verification."
            )
        (
            verified_solids,
            verified_faces,
            solid_map,
            face_map,
        ) = _verify_roundtrip(final_shape, reopened_shape, staged_model)
        if identity_partition:
            association_graph, history_certificate_id = _identity_association_graph(
                plan,
                source_revision,
                source_solids,
                staged_model,
                verified_solids,
                verified_faces,
                solid_map,
                face_map,
            )
        else:
            association_graph, history_certificate_id = _association_graph(
                plan,
                source_revision,
                source_solids,
                staged_model,
                verified_solids,
                verified_faces,
                solid_map,
                face_map,
                builder,
            )
        regions, patches = _regions_and_patches(
            plan,
            staged_model,
            original_owners,
            solid_map,
        )
        edge_sources = {
            value.source_occurrence_id
            for value in association_graph.transaction.correspondences
        }
        edge_targets = {
            value.target_occurrence_id
            for value in association_graph.transaction.correspondences
        }
        report = BRepPartitionReport(
            plan.plan_id,
            staged_model.model_id,
            source_revision.revision_id,
            staged_model.source_revision,
            history_certificate_id,
            sum(value.kind == "solid" for value in source_revision.occurrences),
            sum(value.kind == "face" for value in source_revision.occurrences),
            staged_model.topology.num_solids,
            staged_model.topology.num_faces,
            sum(
                value.occurrence_id not in edge_sources
                for value in source_revision.occurrences
            ),
            sum(
                value.occurrence_id not in edge_targets
                for value in association_graph.target_revision.occurrences
            ),
        )
        BRepPartitionResult(
            staged_model,
            association_graph.target_revision,
            association_graph,
            regions,
            patches,
            report,
        )
        if target.exists() and not plan.policy.overwrite:
            raise FileExistsError(target)
        _publish_staged(staging, target, plan.policy.overwrite)
        final_model = import_brep(
            target,
            coordinate_contract=plan.coordinate_contract,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            trim_samples_per_edge=trim_samples_per_edge,
        )
        if (
            final_model.source_revision != staged_model.source_revision
            or final_model.model_id != staged_model.model_id
        ):
            raise RuntimeError(
                "Published partition differs from its verified staged artifact."
            )
        final_revision = cad_revision_from_brep_model(final_model)
        final_graph = AssociationGraph(
            association_graph.source_revision,
            final_revision,
            association_graph.transaction,
        )
        return BRepPartitionResult(
            final_model,
            final_revision,
            final_graph,
            regions,
            patches,
            report,
        )
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
