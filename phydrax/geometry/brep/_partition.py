#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
from itertools import product
from math import prod
from pathlib import Path
from typing import assert_never, Literal, TypeAlias

import numpy as np

import phydrax.interchange._cad as cad_owner
import phydrax.interchange._cad_archive as cad_archive_owner
import phydrax.interchange._cad_brep_text as cad_brep_text_owner

from ..._external_resource import ResourceLimits
from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._validation import normalized_identifier
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, parse
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    CADRevision,
    CADSelectionSet,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._boolean_core import (
    assemble,
    boundary,
    BRepBooleanFailure,
    BRepBooleanPolicy,
    components,
    material_owner,
    membership,
    rectangles,
    rectilinear_family,
    RegionSelection,
    source_revision,
)
from ._boolean_full_overlay import full_overlay_partition_brep
from ._constructors import BRepTessellationPolicy
from ._coordinate_request import coordinate_source_request
from ._intersection_curve import original_trim_intersection_preparation
from ._model import BRepEntityId, BRepModel
from ._request_preparations import TrimRequestPreparations
from ._revision import (
    _solid_occurrence_id,
    cad_revision_from_brep_model,
)
from ._sewing import BRepSewingPolicy


BRepPartitionDimension: TypeAlias = Literal[2, 3]
PartitionSuffix: TypeAlias = Literal[".brep", ".brp", ".phx"]


def _partition_dimension(value: int, /) -> BRepPartitionDimension:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "BRepPartition.topological_dimension must be an integer; got a noninteger or bool"
        )
    return parse(value, BRepPartitionDimension, "BRepPartition.topological_dimension")


@checked
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
        raise ValueError("BRepPartition.model must contain solid occurrences; got none")
    return selection


@checked
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
        raise ValueError(
            "BRepPartition.model must contain root face occurrences; got none"
        )
    return selection


class BRepPartitionRole(str, Enum):
    """Geometry-only role of a partition operand."""

    REGION = "region"
    VOID = "void"


@dataclass(frozen=True, slots=True)
class BRepPartitionOperand:
    """Declare one persisted solid selection participating in a partition."""

    operand_id: str
    model: BRepModel
    role: BRepPartitionRole
    selection: CADSelectionSet | None = None
    target_region_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        operand_id = normalized_identifier(str(self.operand_id), "operand_id")
        if not isinstance(self.model, BRepModel):
            raise TypeError("BRep.model must be a BRepModel; got another object kind")
        role = parse(self.role, BRepPartitionRole, "BRepPartitionOperand.role")
        revision = cad_revision_from_brep_model(self.model)
        selection = self.selection
        if selection is None:
            selection = all_brep_solids(self.model)
        if not isinstance(selection, CADSelectionSet):
            raise TypeError(
                "BRepPartitionOperand.selection must be a CADSelectionSet or None; got another object kind"
            )
        selection.require_kind("solid")
        if not selection.selectors:
            raise ValueError(
                "BRepPartitionOperand.selection must select at least one solid; got an empty selection"
            )
        if selection.revision_id != revision.revision_id:
            raise ValueError(
                "BRepPartitionOperand.selection.revision_id must match the model revision; got another revision"
            )
        for selector in selection.selectors:
            if revision.select(selector.occurrence_id) != selector:
                raise ValueError(
                    "BRepPartitionOperand.selection must match the B-Rep occurrence inventory; got a mismatched selector"
                )
        target_region_ids = tuple(
            normalized_identifier(str(value), "target_region_id")
            for value in self.target_region_ids
        )
        if len(set(target_region_ids)) != len(target_region_ids):
            raise ValueError(
                "BRepPartitionOperand.target_region_ids must be unique; got repeated target regions"
            )
        if role is BRepPartitionRole.REGION and target_region_ids:
            raise ValueError(
                "BRepPartitionOperand.target_region_ids must be empty for a region; got void targets"
            )
        if role is BRepPartitionRole.VOID and not target_region_ids:
            raise ValueError(
                "BRepPartitionOperand.target_region_ids must be nonempty for a void; got no targets"
            )
        object.__setattr__(self, "operand_id", operand_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "target_region_ids", target_region_ids)


def _operand_selection(operand: BRepPartitionOperand) -> CADSelectionSet:
    selection = operand.selection
    # BRepPartitionOperand.__post_init__ replaces a None selection with all solids.
    if not (selection is not None):
        raise RuntimeError(
            "BRepPartitionOperand.selection must be normalized before consumption; got None"
        )
    return selection


@dataclass(frozen=True, slots=True)
class BRepPartitionPolicy:
    """Declare total region precedence and partition publication limits."""

    region_precedence: tuple[str, ...]
    overwrite: bool = False
    run_parallel: bool = False
    maximum_cells: int = 100_000
    maximum_faces: int = 100_000
    sewing: BRepSewingPolicy = field(default_factory=BRepSewingPolicy)

    def __post_init__(self) -> None:
        precedence = tuple(
            normalized_identifier(str(value), "region_precedence entry")
            for value in self.region_precedence
        )
        if not precedence:
            raise ValueError(
                "BRepPartitionPolicy.region_precedence must be nonempty; got no regions"
            )
        if len(set(precedence)) != len(precedence):
            raise ValueError(
                "BRepPartitionPolicy.region_precedence must be unique; got repeated regions"
            )
        for name, value in (
            ("BRepPartitionPolicy.overwrite", self.overwrite),
            ("BRepPartitionPolicy.run_parallel", self.run_parallel),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool value; got {value!r}")
        for name, value in (
            ("BRepPartitionPolicy.maximum_cells", self.maximum_cells),
            ("BRepPartitionPolicy.maximum_faces", self.maximum_faces),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer; got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value!r}")
        if not isinstance(self.sewing, BRepSewingPolicy):
            raise TypeError(
                "BRep.sewing must be a BRepSewingPolicy; got another object kind"
            )
        object.__setattr__(self, "region_precedence", precedence)


@dataclass(frozen=True, slots=True)
class BRepPartitionPlan:
    """Declare a material-neutral recipe for a host-side CAD partition."""

    coordinate_contract: SpatialCoordinateContract
    operands: tuple[BRepPartitionOperand, ...]
    policy: BRepPartitionPolicy
    plan_id: str = field(init=False)
    topological_dimension: int = field(init=False, default=3)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError(
                "BRepPartitionPlan.coordinate_contract must be a SpatialCoordinateContract; got another object kind"
            )
        operands = tuple(self.operands)
        if not operands or any(
            not isinstance(value, BRepPartitionOperand) for value in operands
        ):
            raise TypeError(
                "BRepPartitionPlan.operands must contain at least one BRepPartitionOperand; got an empty or invalid operand collection"
            )
        if not isinstance(self.policy, BRepPartitionPolicy):
            raise TypeError(
                "BRepPartitionPlan.policy must be a BRepPartitionPolicy; got another object kind"
            )
        operand_ids = tuple(value.operand_id for value in operands)
        if len(set(operand_ids)) != len(operand_ids):
            raise ValueError(
                "BRepPartitionPlan.operands.operand_id must be unique; got duplicate operand IDs"
            )
        region_ids = {
            value.operand_id
            for value in operands
            if value.role is BRepPartitionRole.REGION
        }
        if set(self.policy.region_precedence) != region_ids:
            raise ValueError(
                "BRepPartitionPlan.policy.region_precedence must name every region operand once; got missing or repeated region IDs"
            )
        for operand in operands:
            if (
                operand.model.coordinate_contract.spatial_id
                != self.coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "BRepPartitionPlan.operands.coordinate_contract must match the plan spatial identity; got another coordinate contract"
                )
            if (
                operand.role is BRepPartitionRole.VOID
                and not set(operand.target_region_ids) <= region_ids
            ):
                raise ValueError(
                    "BRepPartitionPlan.void.target_region_ids must name declared region operands; got an unknown target region"
                )
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
    """Return a named region represented by semantic B-Rep entity IDs."""

    name: str
    entity_ids: tuple[BRepEntityId, ...]
    topological_dimension: int = 3
    entity_kind: str = field(init=False)

    def __post_init__(self) -> None:
        name = normalized_identifier(str(self.name), "region name")
        dimension = _partition_dimension(self.topological_dimension)
        expected_kind = "face" if dimension == 2 else "solid"
        entity_ids = tuple(self.entity_ids)
        if not entity_ids or any(
            not isinstance(value, BRepEntityId) or value.kind != expected_kind
            for value in entity_ids
        ):
            raise TypeError(
                f"BRepPartitionRegion.entity_ids must contain BRep {expected_kind} IDs for dimension {dimension}; got empty or incompatible IDs"
            )
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError(
                "BRepPartitionRegion.entity_ids must be unique; got a repeated entity ID"
            )
        if len({value.source_revision for value in entity_ids}) != 1:
            raise ValueError(
                "BRepPartitionRegion.entity_ids must share one semantic revision; got multiple revisions"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "entity_ids", entity_ids)
        object.__setattr__(self, "topological_dimension", dimension)
        object.__setattr__(self, "entity_kind", expected_kind)


@dataclass(frozen=True, slots=True)
class BRepPartitionPatch:
    """Return a named boundary set with one- or two-sided region incidence."""

    name: str
    entity_ids: tuple[BRepEntityId, ...]
    adjacent_region_ids: tuple[str, ...]
    topological_dimension: int = 3
    entity_kind: str = field(init=False)

    def __post_init__(self) -> None:
        name = normalized_identifier(str(self.name), "patch name")
        dimension = _partition_dimension(self.topological_dimension)
        expected_kind = "edge" if dimension == 2 else "face"
        entity_ids = tuple(self.entity_ids)
        adjacent = tuple(
            normalized_identifier(str(value), "adjacent_region_id")
            for value in self.adjacent_region_ids
        )
        if not entity_ids or any(
            not isinstance(value, BRepEntityId) or value.kind != expected_kind
            for value in entity_ids
        ):
            raise TypeError(
                f"BRepPartitionPatch.entity_ids must contain BRep {expected_kind} IDs for dimension {dimension}; got empty or incompatible IDs"
            )
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError(
                "BRepPartitionPatch.entity_ids must be unique; got a repeated entity ID"
            )
        if len({value.source_revision for value in entity_ids}) != 1:
            raise ValueError(
                "BRepPartitionPatch.entity_ids must share one semantic revision; got multiple revisions"
            )
        if len(adjacent) not in (1, 2):
            raise ValueError(
                "BRepPartitionPatch.adjacent_region_ids must contain one or two regions; got another cardinality"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "entity_ids", entity_ids)
        object.__setattr__(self, "adjacent_region_ids", adjacent)
        object.__setattr__(self, "topological_dimension", dimension)
        object.__setattr__(self, "entity_kind", expected_kind)


@dataclass(frozen=True, slots=True)
class BRepPartitionReport:
    """Report history, inventory, and publication evidence for a partition."""

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
        object.__setattr__(
            self, "plan_id", normalized_identifier(str(self.plan_id), "plan_id")
        )
        object.__setattr__(
            self, "model_id", normalized_identifier(str(self.model_id), "model_id")
        )
        object.__setattr__(
            self,
            "source_revision_id",
            normalized_identifier(str(self.source_revision_id), "source_revision_id"),
        )
        object.__setattr__(
            self,
            "target_revision_id",
            normalized_identifier(str(self.target_revision_id), "target_revision_id"),
        )
        object.__setattr__(
            self,
            "history_certificate_id",
            normalized_identifier(
                str(self.history_certificate_id), "history_certificate_id"
            ),
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
            raise ValueError(
                "BRepPartitionReport.counts must be nonnegative integers; got a negative, bool, or noninteger count"
            )
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
    """Return a partition with certified history and named entity sets."""

    model: BRepModel
    revision: CADRevision
    association_graph: AssociationGraph
    regions: tuple[BRepPartitionRegion, ...]
    patches: tuple[BRepPartitionPatch, ...]
    report: BRepPartitionReport
    topological_dimension: int = 3
    named_region_entity_ids: tuple[tuple[str, tuple[BRepEntityId, ...]], ...] = field(
        init=False
    )
    named_patch_entity_ids: tuple[tuple[str, tuple[BRepEntityId, ...]], ...] = field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model, BRepModel):
            raise TypeError("BRep.model must be a BRepModel; got another object kind")
        if not isinstance(self.revision, CADRevision):
            raise TypeError(
                "BRepPartitionResult.revision must be a CADRevision; got another object kind"
            )
        if not isinstance(self.association_graph, AssociationGraph):
            raise TypeError(
                "BRepPartitionResult.association_graph must be an AssociationGraph; got another object kind"
            )
        if not isinstance(self.report, BRepPartitionReport):
            raise TypeError(
                "BRepPartitionResult.report must be a BRepPartitionReport; got another object kind"
            )
        dimension = _partition_dimension(self.topological_dimension)
        if self.report.topological_dimension != dimension:
            raise ValueError(
                "BRepPartitionResult.topological_dimension must match report.topological_dimension; got different dimensions"
            )
        regions = tuple(self.regions)
        patches = tuple(self.patches)
        if not regions or any(
            not isinstance(value, BRepPartitionRegion) for value in regions
        ):
            raise TypeError(
                "BRepPartitionResult.regions must contain BRepPartitionRegion values; got an empty or invalid region collection"
            )
        if not patches or any(
            not isinstance(value, BRepPartitionPatch) for value in patches
        ):
            raise TypeError(
                "BRepPartitionResult.patches must contain BRepPartitionPatch values; got an empty or invalid patch collection"
            )
        if any(
            value.topological_dimension != dimension for value in (*regions, *patches)
        ):
            raise ValueError(
                "BRepPartitionResult.topological_dimension must match every region and patch dimension; got a mismatched dimension"
            )
        if len({value.name for value in regions}) != len(regions):
            raise ValueError(
                "BRepPartitionResult.regions.name must be unique; got duplicate names"
            )
        if len({value.name for value in patches}) != len(patches):
            raise ValueError(
                "BRepPartitionResult.patches.name must be unique; got duplicate names"
            )
        if self.revision.revision_id != self.model.source_revision:
            raise ValueError(
                "BRepPartitionResult.revision must use model.source_revision; got a different revision"
            )
        if self.association_graph.target_revision != self.revision:
            raise ValueError(
                "BRepPartitionResult.association_graph must terminate at revision; got a different target"
            )
        self._validate_report_inventory(dimension)
        self._validate_entity_incidence(regions, patches, dimension)

        named_regions = tuple((value.name, value.entity_ids) for value in regions)
        named_patches = tuple((value.name, value.entity_ids) for value in patches)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "patches", patches)
        object.__setattr__(self, "topological_dimension", dimension)
        object.__setattr__(self, "named_region_entity_ids", named_regions)
        object.__setattr__(self, "named_patch_entity_ids", named_patches)

    @property
    def region_entity_kind(self) -> str:
        return "face" if self.topological_dimension == 2 else "solid"

    @property
    def patch_entity_kind(self) -> str:
        return "edge" if self.topological_dimension == 2 else "face"

    @property
    def named_solid_entity_ids(self) -> tuple[tuple[str, tuple[BRepEntityId, ...]], ...]:
        return self.named_region_entity_ids if self.topological_dimension == 3 else ()

    @property
    def named_face_entity_ids(self) -> tuple[tuple[str, tuple[BRepEntityId, ...]], ...]:
        return (
            self.named_patch_entity_ids
            if self.topological_dimension == 3
            else self.named_region_entity_ids
        )

    @property
    def named_edge_entity_ids(self) -> tuple[tuple[str, tuple[BRepEntityId, ...]], ...]:
        return self.named_patch_entity_ids if self.topological_dimension == 2 else ()

    def _validate_report_inventory(self, dimension: int) -> None:
        if (
            self.report.model_id != self.model.model_id
            or self.report.target_revision_id != self.revision.revision_id
            or self.report.source_revision_id
            != self.association_graph.source_revision.revision_id
            or self.report.history_certificate_id
            != self.association_graph.transaction.coverage.certificate_id
        ):
            raise ValueError(
                "BRepPartitionResult.report must match model, revision, and history identities; got a mismatched identity"
            )
        if (
            self.report.target_solids != self.model.topology.num_solids
            or self.report.target_faces != self.model.topology.num_faces
            or self.report.target_edges
            != (self.model.topology.num_edges if dimension == 2 else 0)
        ):
            raise ValueError(
                "BRepPartitionResult.report must match the model entity inventory; got different counts"
            )

    def _validate_entity_incidence(
        self,
        regions: tuple[BRepPartitionRegion, ...],
        patches: tuple[BRepPartitionPatch, ...],
        dimension: int,
    ) -> None:
        region_entities = self.model.face_ids if dimension == 2 else self.model.solid_ids
        patch_entities = self.model.edge_ids if dimension == 2 else self.model.face_ids
        if dimension == 2 and self.model.topology.num_solids:
            raise ValueError(
                "BRepPartitionResult.model.topology must contain no solids for dimension 2; got solids"
            )
        region_owner: dict[BRepEntityId, str] = {}
        for region in regions:
            for entity_id in region.entity_ids:
                if entity_id in region_owner:
                    raise ValueError(
                        "BRepPartitionResult.regions.entity_ids must be disjoint; got a repeated entity"
                    )
                region_owner[entity_id] = region.name
        if set(region_owner) != set(region_entities):
            raise ValueError(
                "BRepPartitionResult.regions.entity_ids must exhaust model region entities; got missing or extra entities"
            )
        patch_owner: dict[BRepEntityId, tuple[str, ...]] = {}
        for patch in patches:
            for entity_id in patch.entity_ids:
                if entity_id in patch_owner:
                    raise ValueError(
                        "BRepPartitionResult.patches.entity_ids must be disjoint; got a repeated entity"
                    )
                patch_owner[entity_id] = patch.adjacent_region_ids
        if set(patch_owner) != set(patch_entities):
            raise ValueError(
                "BRepPartitionResult.patches.entity_ids must exhaust model boundary entities; got missing or extra entities"
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
                    "BRepPartitionResult.patch.incidence must contain one or two regions; got another cardinality"
                )
            adjacent = tuple(
                sorted(
                    (region_owner[region_entities[index]] for index in region_indices),
                    key=region_rank.__getitem__,
                )
            )
            if patch_owner[patch_entities[patch_index]] != adjacent:
                raise ValueError(
                    "BRepPartitionResult.patch.adjacent_region_ids must match model incidence; got different adjacent regions"
                )
            if dimension == 2 and len(region_indices) == 2:
                self._validate_shared_planar_edge(patch_index, region_indices)

    def _validate_shared_planar_edge(
        self, patch_index: int, region_indices: tuple[int, ...]
    ) -> None:
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
                    "BRepPartitionResult.face_wires must contain each incident edge once; got missing or repeated edge incidence"
                )
            signs.append(1 if matches[0] > 0 else -1)
        if signs[0] == signs[1]:
            raise ValueError(
                "BRepPartitionResult.shared_edge must have opposite face incidence; got equal signs"
            )

    def region(self, name: str, /) -> BRepPartitionRegion:
        expected = normalized_identifier(str(name), "region name")
        for region in self.regions:
            if region.name == expected:
                return region
        raise KeyError(
            f"BRepPartitionResult.region.name must identify a declared region; got {expected!r}"
        )

    def patch(self, name: str, /) -> BRepPartitionPatch:
        expected = normalized_identifier(str(name), "patch name")
        for patch in self.patches:
            if patch.name == expected:
                return patch
        raise KeyError(
            f"BRepPartitionResult.patch.name must identify a declared patch; got {expected!r}"
        )


class BRepPartitionHistoryError(RuntimeError):
    """Raised when native CAD cannot certify exhaustive region and patch history."""


def publish_staged(staging: Path, target: Path, overwrite: bool) -> None:
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
class NativePartition:
    """Retain a route-independent partition before persistence.

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
) -> RegionSelection:
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
    return RegionSelection(
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
    selection: RegionSelection,
    policy: BRepBooleanPolicy,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> NativePartition:
    """Partition axis-aligned rectangular solids using rational source coordinates."""
    selected_solids = tuple(
        frozenset(solid for operand, solid in selection.solids if operand == index)
        for index in range(len(models))
    )
    rectangles_bank = rectangles(models, selected_solids)
    coordinates = tuple(
        tuple(
            sorted(
                {rectangle.lower[d] for rectangle in rectangles_bank}
                | {rectangle.upper[d] for rectangle in rectangles_bank}
            )
        )
        for d in range(3)
    )
    count = prod(len(values) - 1 for values in coordinates)
    if count > policy.maximum_cells:
        raise BRepBooleanFailure(
            "BRepPartition.arrangement.cells must fit maximum_cells; got partition arrangement cell budget exhausted"
        )
    membership_bank: dict[tuple[int, int, int], tuple[tuple[int, int], ...]] = {}
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
        members = membership(point, rectangles_bank, models)
        membership_bank[cell] = members
        owner = material_owner(selection, members)
        if owner is not None:
            region_cells[owner].add(cell)
    components_bank = []
    owners = []
    for region in plan.policy.region_precedence:
        parts = components(region_cells[region])
        components_bank.extend(parts)
        owners.extend(region for _ in parts)
    if not components_bank:
        raise BRepBooleanFailure(
            "BRepPartition.regions must contain surviving material; got no surviving material region"
        )
    components_ = tuple(components_bank)
    faces = boundary(components_, coordinates, rectangles_bank, policy)
    certificate = canonical_fingerprint(
        {
            "kind": "native-cad-partition-arrangement",
            "plan": plan.plan_id,
            "coordinates": coordinates,
            "components": components_,
            "owners": owners,
        }
    )
    model = assemble(
        models,
        coordinates,
        faces,
        policy,
        certificate,
        shared_faces=True,
        preparations=preparations,
        budget=budget,
    )
    solid_pairs = {
        (f"operand:{operand}/solid:{parent_solid}", solid)
        for solid, component in enumerate(components_)
        for operand, parent_solid in {
            ancestor for cell in component for ancestor in membership_bank[cell]
        }
    }
    # Shared faces are numbered by first occurrence, as assemble allocates them.
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
    return NativePartition(
        models,
        model,
        tuple(owners),
        tuple(sorted(solid_pairs)),
        tuple(sorted(face_pairs)),
        certificate,
        "native-rectilinear-cad-partition",
    )


@checked
def native_arrangement(
    plan: BRepPartitionPlan,
    policy: BRepBooleanPolicy,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> NativePartition:
    """Dispatch rectilinear cells or the exhaustive curved face-cell overlay.

    Both routes resolve every open material cell with the same region/void
    owner rule; neither tessellates, samples, or refits a source.
    """
    operands = tuple(sorted(plan.operands, key=lambda operand: operand.operand_id))
    models = tuple(operand.model for operand in operands)
    selection = _partition_selection(plan, operands)
    for model in models:
        if model.geometry is None:
            raise BRepBooleanFailure(
                "BRepBoolean.model.geometry must define closed-solid shells; got missing geometry or shells",
                (model.model_id,),
            )
    if rectilinear_family(models):
        return _rectilinear_partition(
            plan, models, selection, policy, preparations=preparations, budget=budget
        )
    overlay = full_overlay_partition_brep(
        models, selection, policy, preparations=preparations, budget=budget
    )
    return NativePartition(
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
            "BRepPartition.persistence.model.geometry must be defined; got None"
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


def native_roundtrip_maps(
    original: BRepModel,
    reopened: BRepModel,
    exported: tuple[tuple[str, str], ...] | None = None,
    imported: tuple[tuple[str, str], ...] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if exported is None and imported is None:
        if original.model_id != reopened.model_id:
            raise BRepPartitionHistoryError(
                "BRepPartition.persistence.archive must preserve model_id; got a different model identity"
            )
        face_map = tuple(reopened.face_ids.index(entity) for entity in original.face_ids)
        solid_map = tuple(
            reopened.solid_ids.index(entity) for entity in original.solid_ids
        )
    else:
        if exported is None or imported is None or not exported or not imported:
            raise BRepPartitionHistoryError(
                "BRepPartition.persistence.roundtrip must retain codec identity provenance; got no provenance"
            )
        source_refs = dict(exported)
        target_refs = dict(imported)
        if len(source_refs) != len(exported) or len(target_refs) != len(imported):
            raise BRepPartitionHistoryError(
                "BRepPartition.persistence.codec_labels must be unique; got repeated labels"
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
                "BRepPartition.persistence.faces must preserve mapped source geometry; got a changed face"
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
            "BRepPartition.persistence.solid_incidence must match mapped source incidence; got different incidence"
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
                raise BRepPartitionHistoryError(
                    "BRepPartition.persistence.solid_orientation must preserve source orientation; got a different orientation"
                )
    return solid_map, face_map


def _codec_entity_map(
    exported: dict[str, str],
    imported: dict[str, str],
    kind: str,
    source_count: int,
    target_count: int,
) -> tuple[int, ...]:
    if source_count != target_count:
        raise BRepPartitionHistoryError(
            "BRepPartition.persistence.entity_inventory must preserve cardinality; got different entity counts"
        )
    target = {}
    for index in range(target_count):
        label = f"{kind}:{index}"
        if label not in imported or imported[label] in target:
            raise BRepPartitionHistoryError(
                "BRepPartition.persistence.reader_labels must identify every entity uniquely; got missing or ambiguous labels"
            )
        target[imported[label]] = index
    result = []
    for index in range(source_count):
        label = f"{kind}:{index}"
        if label not in exported or exported[label] not in target:
            raise BRepPartitionHistoryError(
                "BRepPartition.persistence.writer_reference must have a reader descendant; got no descendant"
            )
        result.append(target[exported[label]])
    if len(set(result)) != source_count:
        raise BRepPartitionHistoryError(
            "BRepPartition.persistence.reference_mapping must be bijective; got duplicate or missing references"
        )
    return tuple(result)


def _native_partition_graph(
    native: NativePartition,
    model: BRepModel,
    solid_map: tuple[int, ...],
    face_map: tuple[int, ...],
) -> AssociationGraph:
    source = source_revision(native.models, native.certificate)
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
            f"BRepPartition.history.target_occurrences must have source ancestry; got unmapped {sorted(unmapped)!r}"
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
    native: NativePartition,
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


@checked
def partition_brep(
    plan: BRepPartitionPlan,
    /,
    *,
    destination: str | Path,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepPartitionResult:
    """Partition regions, certify the round-trip, and commit publication atomically.

    Axis-aligned rectangular solids use source-coordinate cells; other
    admitted solids use the exhaustive native face-cell overlay, with interface
    faces shared by both adjacent regions. ``.phx`` persists every carrier,
    including intersection branches. External B-Rep text admits only carriers
    it represents exactly and refuses intersection branches before publication.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request((plan,)) as budget:
            return partition_brep(
                plan,
                destination=destination,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
                trim_samples_per_edge=trim_samples_per_edge,
                preparations=preparations,
                budget=budget,
            )
    # One branch preparation scope covers construction and the round-trip
    # verification, whose reloaded branches share identical-query results.
    with original_trim_intersection_preparation(
        (), preparations=preparations
    ) as preparations:
        return _partition_brep(
            plan,
            destination,
            linear_deflection,
            angular_deflection,
            trim_samples_per_edge,
            preparations=preparations,
            budget=budget,
        )


def _partition_brep(
    plan: BRepPartitionPlan,
    destination: str | Path,
    linear_deflection: float,
    angular_deflection: float,
    trim_samples_per_edge: int,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> BRepPartitionResult:

    if plan.policy.run_parallel:
        raise ValueError(
            "partition_brep.plan.policy.run_parallel must be False; got True"
        )
    target = Path(destination).expanduser().resolve()
    suffix = parse(
        target.suffix.lower(), PartitionSuffix, "partition_brep.destination.suffix"
    )
    if target.exists() and not plan.policy.overwrite:
        raise FileExistsError(target)
    tessellation = BRepTessellationPolicy(
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
    )
    native = native_arrangement(
        plan,
        BRepBooleanPolicy(
            maximum_cells=plan.policy.maximum_cells,
            maximum_faces=plan.policy.maximum_faces,
            sewing=plan.policy.sewing,
            tessellation=tessellation,
        ),
        preparations=preparations,
        budget=budget,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.partition-", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    staging = Path(name)
    staging.unlink()
    try:
        match suffix:
            case ".phx":
                cad_archive_owner.save_brep_archive(native.model, staging)
                model = cad_archive_owner.load_brep_archive(
                    staging, preparations=preparations
                )
                exported_provenance = imported_provenance = None
            case ".brep" | ".brp":
                exported_provenance = cad_brep_text_owner.write_brep_text(
                    native.model, staging
                ).provenance
                policy = cad_owner.CadImportPolicy(
                    plan.coordinate_contract,
                    ResourceLimits(64 * 1024 * 1024, 128, 1_000_000, 10_000_000, 0),
                    tessellation=tessellation,
                )
                decoded = cad_brep_text_owner.read_brep_text(
                    staging,
                    policy,
                    trusted_root=target.parent,
                    source_length_unit=plan.coordinate_contract.length_unit,
                )
                restored = decoded.model
                imported_provenance = decoded.provenance
                model = restored.evolve(
                    report=replace(restored.report, source_id=str(target)),
                    budget=budget,
                    preparations=preparations,
                )
            case _:
                assert_never(suffix)
        solid_map, face_map = native_roundtrip_maps(
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
        publish_staged(staging, target, plan.policy.overwrite)
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
