#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

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
from ._projection_contracts import brep_entity_id


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


def _entity_text(entity: BRepEntityId, /) -> str:
    dimensions = {"vertex": 0, "edge": 1, "face": 2, "solid": 3}
    return brep_entity_id(
        entity.source_id,
        entity.source_revision,
        dimensions[entity.kind],
        entity.index,
        occurrence_path=entity.occurrence_path,
    )


def _entity_occurrence_id(entity: BRepEntityId, /) -> str:
    return canonical_fingerprint(
        {"kind": "cad-qualified-entity-occurrence", "entity": _entity_text(entity)}
    )


def _assembly_namespace_occurrence_id(
    source_id: str, source_revision: str, path: tuple[str, ...], /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "cad-instance-namespace",
            "source_id": source_id,
            "source_revision": source_revision,
            "path": path,
        }
    )


def cad_revision_from_brep_model(model: BRepModel, /) -> CADRevision:
    """Create the exhaustive occurrence-qualified source entity inventory."""
    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    geometry = model.geometry
    records = geometry.qualified_entity_incidence(model.source_id, model.source_revision)
    entities = tuple(sorted({record.member for record in records}))
    namespaces = tuple(
        sorted(
            {entity.occurrence_path for entity in entities if entity.occurrence_path}
            | {container.path for container in geometry.iter_assembly_containers()}
            | {occurrence.path for occurrence in geometry.iter_occurrences()}
        )
    )
    by_namespace = {
        namespace: _assembly_namespace_occurrence_id(
            model.source_id, model.source_revision, namespace
        )
        for namespace in namespaces
    }
    parents = {
        child: container.path
        for container in geometry.iter_assembly_containers()
        for child in (*container.member_paths, *container.child_paths)
    }
    namespace_paths: dict[tuple[str, ...], tuple[str, ...]] = {}

    def namespace_path(namespace: tuple[str, ...], /) -> tuple[str, ...]:
        if namespace not in namespace_paths:
            chain: list[tuple[str, ...]] = []
            active: set[tuple[str, ...]] = set()
            current = namespace
            while current not in namespace_paths:
                if current in active:
                    raise ValueError(
                        "Assembly container paths must form an acyclic forest."
                    )
                active.add(current)
                chain.append(current)
                parent = parents.get(current)
                if parent is None:
                    break
                current = parent
            prefix = namespace_paths.get(current, ())
            for item in reversed(chain):
                prefix = (*prefix, by_namespace[item])
                namespace_paths[item] = prefix
        return namespace_paths[namespace]

    occurrences: list[CADOccurrence] = []
    for namespace in namespaces:
        identifier = by_namespace[namespace]
        parent = parents.get(namespace)
        occurrences.append(
            CADOccurrence(
                model.source_revision,
                identifier,
                identifier,
                "assembly",
                namespace_path(namespace),
                None if parent is None else by_namespace[parent],
            )
        )
    for entity in entities:
        identifier = _entity_occurrence_id(entity)
        parent = by_namespace.get(entity.occurrence_path)
        path = (
            (identifier,)
            if parent is None
            else (*namespace_path(entity.occurrence_path), identifier)
        )
        occurrences.append(
            CADOccurrence(
                model.source_revision,
                identifier,
                _entity_text(entity),
                entity.kind,
                path,
                parent,
            )
        )
    return CADRevision(
        model.source_revision, model.source_id, tuple(occurrences), geometry.geometry_id
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
    """Declare total region precedence for native partition cells."""

    region_precedence: tuple[str, ...]

    def __post_init__(self) -> None:
        precedence = tuple(
            _text(value, "region_precedence entry") for value in self.region_precedence
        )
        if not precedence:
            raise ValueError("Partition policy requires explicit region precedence.")
        if len(set(precedence)) != len(precedence):
            raise ValueError("Region precedence cannot repeat a region.")
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
    operand_association_graphs: tuple[tuple[str, AssociationGraph], ...] = ()
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
        operand_graphs = tuple(self.operand_association_graphs)
        if len({name for name, _ in operand_graphs}) != len(operand_graphs):
            raise ValueError("Partition operand history names must be unique.")
        for name, operand_graph in operand_graphs:
            _text(name, "operand history name")
            if not isinstance(operand_graph, AssociationGraph):
                raise TypeError("Partition operand history must be an AssociationGraph.")
            if operand_graph.target_revision != self.association_graph.source_revision:
                raise ValueError(
                    "Operand history must terminate at the partition source revision."
                )
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
                        self.model.geometry.coedge_senses[coedge]
                        for loop in self.model.geometry.face_loops[face_index]
                        for coedge in loop
                        if self.model.geometry.coedge_edges[coedge] == patch_index
                    )
                    if len(matches) != 1:
                        raise ValueError(
                            "A planar face must contain each incident edge exactly once."
                        )
                    signs.append(matches[0])
                if signs[0] == signs[1]:
                    raise ValueError(
                        "A shared planar edge must have opposite face incidence."
                    )

        named_regions = tuple((value.name, value.entity_ids) for value in regions)
        named_patches = tuple((value.name, value.entity_ids) for value in patches)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "patches", patches)
        object.__setattr__(self, "topological_dimension", dimension)
        object.__setattr__(self, "operand_association_graphs", operand_graphs)
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

    def operand_association_graph(self, operand_id: str, /) -> AssociationGraph:
        """Return the construction-certified operand-to-composite correspondence."""
        expected = _text(operand_id, "operand history name")
        for name, graph in self.operand_association_graphs:
            if name == expected:
                return graph
        raise KeyError(f"Partition has no exact history for operand {expected!r}.")


class BRepPartitionResultArchiveRecipe(NamedTuple):
    """Constructor-owned partition state with a PyTree-visible B-Rep."""

    model: BRepModel
    revision: CADRevision
    association_graph: AssociationGraph
    regions: tuple[BRepPartitionRegion, ...]
    patches: tuple[BRepPartitionPatch, ...]
    report: BRepPartitionReport
    topological_dimension: int
    operand_association_graphs: tuple[tuple[str, AssociationGraph], ...]


class BRepPartitionHistoryError(RuntimeError):
    """Report unresolved native source correspondence or partition history."""


def _operand_composite_graph(
    source: CADRevision,
    target: CADRevision,
    mapping: dict[str, str],
    operand_id: str,
    /,
) -> AssociationGraph:
    """Certify the construction's exact occurrence-preserving composite embedding."""
    source_ids = {item.occurrence_id for item in source.occurrences}
    if not set(mapping) <= source_ids:
        raise BRepPartitionHistoryError(
            "Operand composite mapping has foreign occurrences."
        )
    if len(set(mapping.values())) != len(mapping):
        raise BRepPartitionHistoryError("Operand composite mapping aliases occurrences.")
    for item in source.occurrences:
        if item.occurrence_id not in mapping:
            continue
        mapped = target.occurrence(mapping[item.occurrence_id])
        if (
            mapped.entity_id != item.entity_id
            or mapped.kind != item.kind
            or mapped.orientation != item.orientation
            or any(part not in mapping for part in item.path)
            or mapped.path != tuple(mapping[part] for part in item.path)
        ):
            raise BRepPartitionHistoryError(
                "Operand composite mapping changes occurrence identity."
            )
    certificate = canonical_fingerprint(
        {
            "kind": "native-brep-operand-composite-history",
            "operand": operand_id,
            "source": source.revision_id,
            "target": target.revision_id,
            "pairs": tuple(sorted(mapping.items())),
        }
    )
    mapped_targets = set(mapping.values())
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision_id,
        target.revision_id,
        tuple(
            OccurrenceCorrespondence(first, second, certificate)
            for first, second in sorted(mapping.items())
        ),
        frozenset(source_ids - set(mapping)),
        frozenset(
            item.occurrence_id
            for item in target.occurrences
            if item.occurrence_id not in mapped_targets
        ),
        AssociationCoverageEvidence(
            True, True, certificate, "native-partition-composite"
        ),
    )
    return AssociationGraph(source, target, transaction)


def _partition_source_revision(
    plan: BRepPartitionPlan,
    operands: tuple[BRepPartitionOperand, ...],
) -> tuple[
    CADRevision,
    tuple[dict[str, str], ...],
    tuple[dict[str, str], ...],
    tuple[dict[str, str], ...],
]:
    revision_id = canonical_fingerprint(
        {
            "kind": "native-brep-partition-source-revision",
            "plan": plan.plan_id,
            "operands": tuple(
                (operand.operand_id, operand.model.source_revision)
                for operand in operands
            ),
        }
    )
    occurrences: list[CADOccurrence] = []
    source_entities: list[dict[str, str]] = []
    source_namespaces: list[dict[str, str]] = []
    source_mappings: list[dict[str, str]] = []
    for operand in operands:
        model = operand.model
        geometry = model.geometry
        local_revision = cad_revision_from_brep_model(model)
        local_to_partition = {
            item.occurrence_id: canonical_fingerprint(
                {
                    "kind": "native-brep-partition-source-occurrence",
                    "operand_id": operand.operand_id,
                    "source_occurrence_id": item.occurrence_id,
                }
            )
            for item in local_revision.occurrences
        }
        source_mappings.append(local_to_partition)
        occurrences.extend(
            CADOccurrence(
                revision_id,
                local_to_partition[item.occurrence_id],
                item.entity_id,
                item.kind,
                tuple(local_to_partition[parent] for parent in item.path),
                (
                    None
                    if item.parent_occurrence_id is None
                    else local_to_partition[item.parent_occurrence_id]
                ),
                orientation=item.orientation,
            )
            for item in local_revision.occurrences
        )
        source_by_entity: dict[str, str] = {}
        for record in geometry.qualified_entity_incidence(
            model.source_id, model.source_revision
        ):
            entity = record.member
            if record.container != entity or entity.kind not in (
                "solid",
                "face",
                "edge",
                "vertex",
            ):
                continue
            local_id = _entity_occurrence_id(entity)
            occurrence_id = local_to_partition.get(local_id)
            if occurrence_id is None:
                raise BRepPartitionHistoryError(
                    "Partition source entity is absent from its canonical CAD revision."
                )
            source_by_entity[local_id] = occurrence_id
        source_entities.append(source_by_entity)
        source_namespaces.append(
            {
                item.entity_id: local_to_partition[item.occurrence_id]
                for item in local_revision.occurrences
                if item.kind == "assembly"
            }
        )
    return (
        CADRevision(
            revision_id,
            f"native-brep-partition-sources:{plan.plan_id}",
            tuple(occurrences),
            plan.plan_id,
        ),
        tuple(source_entities),
        tuple(source_namespaces),
        tuple(source_mappings),
    )


def _full_overlay_partition_result(
    plan: BRepPartitionPlan,
    operands: tuple[BRepPartitionOperand, ...],
    *,
    tessellation: Any | None,
) -> BRepPartitionResult:
    from ._assembly import BRepAffinePlacement
    from ._boolean_core import BRepBooleanPolicy, RegionSelection
    from ._boolean_full_overlay import full_overlay_partition_brep
    from ._placement import entity_text, materialize_brep_occurrences

    source, source_occurrences, source_namespaces, source_mappings = (
        _partition_source_revision(plan, operands)
    )
    operand_graphs = tuple(
        (
            operand.operand_id,
            _operand_composite_graph(
                cad_revision_from_brep_model(operand.model),
                source,
                mapping,
                operand.operand_id,
            ),
        )
        for operand, mapping in zip(operands, source_mappings, strict=True)
    )
    identity = BRepAffinePlacement.identity().identity_payload()
    model_list: list[BRepModel] = []
    entity_maps: list[tuple[tuple[BRepEntityId, ...], tuple[BRepEntityId, ...]]] = []
    for operand in operands:
        geometry = operand.model.geometry
        leaves = tuple(geometry.assembly_graph.iter_occurrences())
        if (
            len(leaves) == 1
            and leaves[0].solid_index == 0
            and operand.model.topology.num_solids == 1
            and leaves[0].placement.identity_payload() == identity
        ):
            entities = tuple(
                sorted(
                    {
                        record.member
                        for record in geometry.qualified_entity_incidence(
                            operand.model.source_id, operand.model.source_revision
                        )
                    }
                )
            )
            model_list.append(operand.model)
            entity_maps.append((entities, entities))
        else:
            placed = materialize_brep_occurrences(operand.model)
            model_list.append(placed.model)
            entity_maps.append((placed.source_entities, placed.target_entities))

    selected_solids: list[tuple[int, int]] = []
    target_sources: list[dict[BRepEntityId, str]] = []
    target_source_entities: list[dict[BRepEntityId, BRepEntityId]] = []
    for position, (operand, (source_entities, target_entities)) in enumerate(
        zip(operands, entity_maps, strict=True)
    ):
        source_to_target = dict(zip(source_entities, target_entities, strict=True))
        source_by_text = {entity_text(entity): entity for entity in source_entities}
        for selector in _operand_selection(operand).selectors:
            source_entity = source_by_text.get(selector.entity_id)
            if source_entity is None or source_entity.kind != "solid":
                raise BRepPartitionHistoryError(
                    "Partition selection does not resolve to a source-solid occurrence."
                )
            target_entity = source_to_target[source_entity]
            selected_solids.append((position, target_entity.index))
        target_to_source: dict[BRepEntityId, str] = {}
        for source_entity, target_entity in zip(
            source_entities, target_entities, strict=True
        ):
            if source_entity.kind not in ("solid", "face", "edge", "vertex"):
                continue
            source_occurrence_id = source_occurrences[position].get(
                _entity_occurrence_id(source_entity)
            )
            if source_occurrence_id is None:
                raise BRepPartitionHistoryError(
                    "Partition input has no exact source occurrence binding."
                )
            target_to_source[target_entity] = source_occurrence_id
        target_sources.append(target_to_source)
        target_source_entities.append(
            {
                target_entity: source_entity
                for source_entity, target_entity in zip(
                    source_entities, target_entities, strict=True
                )
            }
        )

    models = tuple(model_list)

    selection = RegionSelection(
        tuple(operand.operand_id for operand in operands),
        plan.policy.region_precedence,
        tuple(
            (operand.operand_id, tuple(sorted(operand.target_region_ids)))
            for operand in operands
            if operand.role is BRepPartitionRole.VOID
        ),
        tuple(sorted(set(selected_solids))),
    )
    native = full_overlay_partition_brep(
        models,
        selection,
        BRepBooleanPolicy(tessellation=tessellation),
    )

    model = native.model
    target_revision = cad_revision_from_brep_model(model)
    geometry = model.geometry
    target_entities = {
        _entity_text(record.member): record.member
        for record in geometry.qualified_entity_incidence(
            model.source_id, model.source_revision
        )
    }
    target_occurrences: dict[BRepEntityId, str] = {}
    target_occurrence_records: dict[str, CADOccurrence] = {}
    for occurrence in target_revision.occurrences:
        target_occurrence_records[occurrence.occurrence_id] = occurrence
        entity = target_entities.get(occurrence.entity_id)
        if entity is not None:
            target_occurrences[entity] = occurrence.occurrence_id
    pairs: set[tuple[str, str]] = set()

    def add_entity_correspondence(
        operand_index: int,
        source_entity: BRepEntityId,
        target_entity: BRepEntityId,
        kind: str,
        /,
    ) -> None:
        source_occurrence_id = target_sources[operand_index].get(source_entity)
        target_id = target_occurrences.get(target_entity)
        original_entity = target_source_entities[operand_index].get(source_entity)
        if source_occurrence_id is None or target_id is None or original_entity is None:
            raise BRepPartitionHistoryError(
                f"Partition {kind} correspondence must bind exact source and target occurrences."
            )
        pairs.add((source_occurrence_id, target_id))
        namespace = _assembly_namespace_occurrence_id(
            original_entity.source_id,
            original_entity.source_revision,
            original_entity.occurrence_path,
        )
        source_assembly_id = source_namespaces[operand_index].get(namespace)
        if source_assembly_id is None:
            raise BRepPartitionHistoryError(
                f"Partition {kind} source must retain its exact assembly occurrence."
            )
        source_assembly_chain = []
        current_source_assembly = source_assembly_id
        while current_source_assembly is not None:
            source_occurrence = source.occurrence(current_source_assembly)
            if source_occurrence.kind != "assembly":
                raise BRepPartitionHistoryError(
                    "Partition source namespace ancestry must contain only assemblies."
                )
            source_assembly_chain.append(current_source_assembly)
            current_source_assembly = source_occurrence.parent_occurrence_id
        target_assembly_chain = []
        current_target_assembly = target_occurrence_records[
            target_id
        ].parent_occurrence_id
        while current_target_assembly is not None:
            target_occurrence = target_occurrence_records.get(current_target_assembly)
            if target_occurrence is None or target_occurrence.kind != "assembly":
                raise BRepPartitionHistoryError(
                    "Partition target namespace ancestry must contain only assemblies."
                )
            target_assembly_chain.append(current_target_assembly)
            current_target_assembly = target_occurrence.parent_occurrence_id
        if source_assembly_chain and not target_assembly_chain:
            raise BRepPartitionHistoryError(
                "Partition target entity must retain its containing assembly occurrence."
            )
        for depth, source_assembly in enumerate(source_assembly_chain):
            target_assembly = target_assembly_chain[
                min(depth, len(target_assembly_chain) - 1)
            ]
            pairs.add((source_assembly, target_assembly))

    for operand_index, source_entity, target_entity in native.solid_pairs:
        add_entity_correspondence(operand_index, source_entity, target_entity, "solid")
    for operand_index, source_entity, target_entity in native.face_pairs:
        add_entity_correspondence(operand_index, source_entity, target_entity, "face")
    for operand_index, source_entity, target_entity in native.edge_pairs:
        add_entity_correspondence(operand_index, source_entity, target_entity, "edge")
    for operand_index, source_entity, target_entity in native.vertex_pairs:
        add_entity_correspondence(operand_index, source_entity, target_entity, "vertex")

    certificate_id = canonical_fingerprint(
        {
            "kind": "native-complete-face-cell-partition-history",
            "plan": plan.plan_id,
            "overlay": native.certificate_id,
            "source_revision": source.revision_id,
            "target_revision": target_revision.revision_id,
            "correspondences": tuple(sorted(pairs)),
        }
    )
    correspondences = tuple(
        OccurrenceCorrespondence(first, second, certificate_id)
        for first, second in sorted(pairs)
    )
    mapped_source = {first for first, _ in pairs}
    mapped_target = {second for _, second in pairs}
    target_ids = {value.occurrence_id for value in target_revision.occurrences}
    transaction = OccurrenceCorrespondenceTransaction(
        certificate_id,
        source.revision_id,
        target_revision.revision_id,
        correspondences,
        frozenset(),
        frozenset(target_ids - mapped_target),
        AssociationCoverageEvidence(
            True,
            True,
            certificate_id,
            "native-complete-face-cell-partition",
        ),
    )
    graph = AssociationGraph(source, target_revision, transaction)
    rank = {name: index for index, name in enumerate(plan.policy.region_precedence)}
    owners = tuple(native.solid_regions)
    regions = tuple(
        BRepPartitionRegion(
            name,
            tuple(
                model.solid_ids[index]
                for index, owner in enumerate(owners)
                if owner == name
            ),
        )
        for name in plan.policy.region_precedence
        if name in owners
    )
    grouped_faces: dict[tuple[str, ...], list[BRepEntityId]] = {}
    for face, solids in enumerate(model.topology.face_solids):
        adjacent = tuple(
            sorted((owners[solid] for solid in solids), key=rank.__getitem__)
        )
        if len(adjacent) not in (1, 2):
            raise BRepPartitionHistoryError(
                "Native partition produced a nonmanifold face-region incidence."
            )
        grouped_faces.setdefault(adjacent, []).append(model.face_ids[face])
    patches = tuple(
        BRepPartitionPatch(
            f"boundary:{adjacent[0]}"
            if len(adjacent) == 1
            else f"interface:{adjacent[0]}:{adjacent[1]}",
            tuple(entity_ids),
            adjacent,
        )
        for adjacent, entity_ids in sorted(
            grouped_faces.items(),
            key=lambda item: tuple(rank[value] for value in item[0]),
        )
    )
    report = BRepPartitionReport(
        plan.plan_id,
        model.model_id,
        source.revision_id,
        target_revision.revision_id,
        certificate_id,
        sum(value.kind == "solid" for value in source.occurrences),
        sum(value.kind == "face" for value in source.occurrences),
        model.topology.num_solids,
        model.topology.num_faces,
        sum(value.occurrence_id not in mapped_source for value in source.occurrences),
        len(target_ids - mapped_target),
    )
    return BRepPartitionResult(
        model,
        target_revision,
        graph,
        regions,
        patches,
        report,
        operand_association_graphs=operand_graphs,
    )


def partition_brep(
    plan: BRepPartitionPlan,
    /,
    *,
    tessellation: Any | None = None,
) -> BRepPartitionResult:
    """Execute an exact in-memory arrangement and publish exhaustive lineage."""
    if not isinstance(plan, BRepPartitionPlan):
        raise TypeError("plan must be a BRepPartitionPlan.")
    if tessellation is not None:
        from ._constructors import BRepTessellationPolicy

        if not isinstance(tessellation, BRepTessellationPolicy):
            raise TypeError("tessellation must be a BRepTessellationPolicy or None.")
    operands = tuple(sorted(plan.operands, key=lambda operand: operand.operand_id))
    from ._native_box_partition import partition_native_boxes

    try:
        return partition_native_boxes(plan, tessellation=tessellation)
    except NotImplementedError:
        pass
    return _full_overlay_partition_result(plan, operands, tessellation=tessellation)


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
