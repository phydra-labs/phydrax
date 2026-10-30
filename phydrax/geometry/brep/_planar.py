#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native planar partition requests and in-memory execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from .._cad_revision import CADSelectionSet
from .._planar_embedding import PlanarEmbedding
from ..simplicial import PlanarMeshRegion
from ._constructors import BRepTessellationPolicy
from ._model import BRepModel
from ._partition import (
    _text,
    all_brep_faces,
    BRepPartitionPolicy,
    BRepPartitionResult,
    BRepPartitionRole,
    cad_revision_from_brep_model,
)


@dataclass(frozen=True, slots=True)
class PlanarPartitionOperand:
    """One planar mesh region or exact root-face selection in a 2D partition."""

    operand_id: str
    source: PlanarMeshRegion | BRepModel
    role: BRepPartitionRole
    selection: CADSelectionSet | None = None
    target_region_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        operand_id = _text(self.operand_id, "operand_id")
        role = BRepPartitionRole(self.role)
        source = self.source
        selection = self.selection
        if isinstance(source, PlanarMeshRegion):
            if selection is not None:
                raise ValueError("Planar mesh operands do not accept CAD selections.")
        elif isinstance(source, BRepModel):
            if source.topology.num_solids:
                raise ValueError(
                    "Persisted planar operands must contain root faces and zero solids."
                )
            revision = cad_revision_from_brep_model(source)
            if selection is None:
                selection = all_brep_faces(source)
            if not isinstance(selection, CADSelectionSet):
                raise TypeError("selection must be a CADSelectionSet or None.")
            selection.require_kind("face")
            if not selection.selectors:
                raise ValueError("A planar B-Rep operand must select at least one face.")
            if selection.revision_id != revision.revision_id:
                raise ValueError("Planar selection belongs to another B-Rep revision.")
            for selector in selection.selectors:
                occurrence = revision.occurrence(selector.occurrence_id)
                if occurrence.parent_occurrence_id is not None:
                    raise ValueError("Planar selections must contain only root faces.")
                if revision.select(selector.occurrence_id) != selector:
                    raise ValueError(
                        "Planar selection does not match the B-Rep occurrence inventory."
                    )
        else:
            raise TypeError("source must be a PlanarMeshRegion or persisted BRepModel.")
        target_region_ids = tuple(
            _text(value, "target_region_id") for value in self.target_region_ids
        )
        if len(set(target_region_ids)) != len(target_region_ids):
            raise ValueError("A planar void cannot repeat a target region.")
        if role is BRepPartitionRole.REGION and target_region_ids:
            raise ValueError("Planar region operands cannot declare void targets.")
        if role is BRepPartitionRole.VOID and not target_region_ids:
            raise ValueError("Planar void operands require at least one target region.")
        object.__setattr__(self, "operand_id", operand_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "target_region_ids", target_region_ids)


@dataclass(frozen=True, slots=True)
class PlanarPartitionPlan:
    """Closed exact 2D CAD partition recipe in one physical world embedding."""

    coordinate_contract: SpatialCoordinateContract
    embedding: PlanarEmbedding
    operands: tuple[PlanarPartitionOperand, ...]
    policy: BRepPartitionPolicy
    plan_id: str = field(init=False)
    topological_dimension: int = field(init=False, default=2)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        if self.coordinate_contract.coordinate_system != "cartesian":
            raise ValueError("Planar embeddings require a Cartesian coordinate contract.")
        if not isinstance(self.embedding, PlanarEmbedding):
            raise TypeError("embedding must be a PlanarEmbedding.")
        operands = tuple(self.operands)
        if not operands or any(
            not isinstance(value, PlanarPartitionOperand) for value in operands
        ):
            raise TypeError("operands must contain at least one PlanarPartitionOperand.")
        if not isinstance(self.policy, BRepPartitionPolicy):
            raise TypeError("policy must be a BRepPartitionPolicy.")
        operand_ids = tuple(value.operand_id for value in operands)
        if len(set(operand_ids)) != len(operand_ids):
            raise ValueError("Planar partition operand IDs must be unique.")
        region_ids = {
            value.operand_id
            for value in operands
            if value.role is BRepPartitionRole.REGION
        }
        if set(self.policy.region_precedence) != region_ids:
            raise ValueError(
                "region_precedence must name every planar region exactly once."
            )
        for operand in operands:
            if (
                isinstance(operand.source, BRepModel)
                and operand.source.coordinate_contract.spatial_id
                != self.coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "Every persisted planar operand must use the plan coordinate contract."
                )
            if (
                operand.role is BRepPartitionRole.VOID
                and not set(operand.target_region_ids) <= region_ids
            ):
                raise ValueError("A planar void targets an unknown partition region.")
        plan_id = canonical_fingerprint(
            {
                "kind": "planar-brep-partition-plan",
                "coordinate_contract": self.coordinate_contract.spatial_id,
                "embedding": self.embedding.embedding_id,
                "topological_dimension": self.topological_dimension,
                "operands": sorted(_operand_descriptor(value) for value in operands),
                "region_precedence": self.policy.region_precedence,
            }
        )
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "plan_id", plan_id)


def _operand_descriptor(operand: PlanarPartitionOperand) -> tuple[object, ...]:
    source = operand.source
    if isinstance(source, BRepModel):
        selection = operand.selection
        if selection is None:
            raise RuntimeError("A planar B-Rep operand lost its exact face selection.")
        source_descriptor: object = (
            "brep",
            source.model_id,
            selection.selection_id,
        )
    else:
        source_descriptor = (
            "planar-mesh-region",
            source.feature_id,
            np.asarray(source.vertices),
            np.asarray(source.edges),
            np.asarray(source.loop_offsets),
        )
    return (
        operand.operand_id,
        source_descriptor,
        operand.role.value,
        tuple(sorted(operand.target_region_ids)),
    )


def partition_planar(
    plan: PlanarPartitionPlan,
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
) -> BRepPartitionResult:
    """Execute an exact supported planar arrangement entirely in memory."""
    if not isinstance(plan, PlanarPartitionPlan):
        raise TypeError("plan must be a PlanarPartitionPlan.")
    if tessellation is not None and not isinstance(tessellation, BRepTessellationPolicy):
        raise TypeError("tessellation must be a BRepTessellationPolicy or None.")
    from ._native_planar_partition import partition_native_planar_arrangement

    return partition_native_planar_arrangement(plan, tessellation=tessellation)


__all__ = [
    "PlanarPartitionOperand",
    "PlanarPartitionPlan",
    "partition_planar",
]
