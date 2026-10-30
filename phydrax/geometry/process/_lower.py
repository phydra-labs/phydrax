#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lower final-state planar stack extrusions through the exact CAD partition seam."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..._fingerprint import canonical_fingerprint
from .._cad_revision import AssociationGraph, CADRevision
from ..brep import (
    BRepEntityId,
    BRepModel,
    BRepPartitionOperand,
    BRepPartitionPatch,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionRegion,
    BRepPartitionReport,
    BRepPartitionResult,
    BRepPartitionRole,
    partition_brep,
)
from ..brep._constructors import (
    brep_extrusion,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from ..simplicial import PlanarMeshRegion
from ._stack import ProcessStack, ZInterval


@dataclass(frozen=True, slots=True)
class ProcessStackResult:
    """In-memory operands and authoritative final CAD partition history."""

    stack: ProcessStack
    partition: BRepPartitionResult
    operand_models: tuple[tuple[str, BRepModel], ...]
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.stack, ProcessStack):
            raise TypeError("stack must be a ProcessStack.")
        if not isinstance(self.partition, BRepPartitionResult):
            raise TypeError("partition must be a BRepPartitionResult.")
        operands = tuple(self.operand_models)
        expected_ids = tuple(value.region_id for value in self.stack.regions) + tuple(
            value.void_id for value in self.stack.voids
        )
        if tuple(name for name, _ in operands) != expected_ids:
            raise ValueError("operand_models must follow the stack operand order.")
        if not all(isinstance(model, BRepModel) for _, model in operands):
            raise TypeError("operand_models must contain BRepModel values.")
        spatial_id = self.stack.coordinate_contract.spatial_id
        if any(
            model.coordinate_contract.spatial_id != spatial_id for _, model in operands
        ):
            raise ValueError(
                "Every in-memory operand must use the stack coordinate contract."
            )
        if self.partition.topological_dimension != 3:
            raise ValueError("Process stack lowering must produce a 3D partition.")
        if self.partition.model.coordinate_contract.spatial_id != spatial_id:
            raise ValueError(
                "The partition model must use the stack coordinate contract."
            )
        if tuple(value.name for value in self.partition.regions) != tuple(
            value.region_id
            for value in sorted(
                self.stack.regions,
                key=lambda region: region.precedence,
                reverse=True,
            )
        ):
            raise ValueError("Partition regions must preserve explicit stack precedence.")
        object.__setattr__(self, "operand_models", operands)
        object.__setattr__(
            self,
            "result_id",
            canonical_fingerprint(
                {
                    "kind": "process-stack-result",
                    "stack": self.stack.stack_id,
                    "partition_model": self.partition.model.model_id,
                    "partition_revision": self.partition.revision.revision_id,
                    "association_graph": self.partition.association_graph.graph_id,
                    "operands": [[name, model.model_id] for name, model in operands],
                }
            ),
        )

    @property
    def model(self) -> BRepModel:
        return self.partition.model

    @property
    def revision(self) -> CADRevision:
        return self.partition.revision

    @property
    def association_graph(self) -> AssociationGraph:
        return self.partition.association_graph

    @property
    def regions(self) -> tuple[BRepPartitionRegion, ...]:
        return self.partition.regions

    @property
    def patches(self) -> tuple[BRepPartitionPatch, ...]:
        return self.partition.patches

    @property
    def report(self) -> BRepPartitionReport:
        return self.partition.report

    @property
    def named_solid_entity_ids(
        self,
    ) -> tuple[tuple[str, tuple[BRepEntityId, ...]], ...]:
        return self.partition.named_solid_entity_ids

    @property
    def named_face_entity_ids(
        self,
    ) -> tuple[tuple[str, tuple[BRepEntityId, ...]], ...]:
        return self.partition.named_face_entity_ids


def _footprint_loops(footprint: PlanarMeshRegion, /) -> tuple[np.ndarray, ...]:
    vertices = np.asarray(footprint.vertices, dtype=np.float64)
    edges = np.asarray(footprint.edges, dtype=np.int64)
    offsets = np.asarray(footprint.loop_offsets, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.all(np.isfinite(vertices)):
        raise ValueError("Process stack footprints require finite shape (N, 2).")
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != edges.shape[0]
        or np.any(offsets[1:] <= offsets[:-1])
    ):
        raise ValueError("Process stack footprint loop topology is malformed.")
    loops: list[np.ndarray] = []
    for index in range(offsets.size - 1):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        loop_edges = edges[start:stop]
        vertex_ids = loop_edges[:, 0]
        if (
            vertex_ids.size < 3
            or np.any(loop_edges[:, 1] != np.roll(vertex_ids, -1))
            or np.any(vertex_ids < 0)
            or np.any(vertex_ids >= vertices.shape[0])
        ):
            raise ValueError("Process stack footprint loops must be closed edge cycles.")
        loops.append(vertices[vertex_ids])
    return tuple(loops)


def _in_memory_operand(
    operand_id: str,
    footprint: PlanarMeshRegion,
    interval: ZInterval,
    stack: ProcessStack,
    tessellation: BRepTessellationPolicy,
    /,
) -> BRepModel:
    loops = tuple(ProfileLoop.polygon(points) for points in _footprint_loops(footprint))
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, interval.lower)),
        loops[0],
        holes=loops[1:],
    )
    model = brep_extrusion(
        profile,
        (0.0, 0.0, interval.height),
        coordinate_contract=stack.coordinate_contract,
        tessellation=tessellation,
        source_id=f"process-stack:{stack.stack_id}:{operand_id}",
    )
    if model.topology.num_solids != 1:
        raise ValueError("A process stack operand must construct exactly one solid.")
    return model


def lower_process_stack(
    stack: ProcessStack,
    /,
    *,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
) -> ProcessStackResult:
    """Construct process operands and partition them without persisting files."""

    if not isinstance(stack, ProcessStack):
        raise TypeError("stack must be a ProcessStack.")
    tessellation = BRepTessellationPolicy(
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
    )

    operands: list[tuple[str, BRepModel]] = []
    partition_operands: list[BRepPartitionOperand] = []
    for region in stack.regions:
        model = _in_memory_operand(
            region.region_id,
            region.footprint,
            region.z_interval,
            stack,
            tessellation,
        )
        operands.append((region.region_id, model))
        partition_operands.append(
            BRepPartitionOperand(
                region.region_id,
                model,
                BRepPartitionRole.REGION,
            )
        )
    for void in stack.voids:
        model = _in_memory_operand(
            void.void_id,
            void.footprint,
            void.z_interval,
            stack,
            tessellation,
        )
        operands.append((void.void_id, model))
        partition_operands.append(
            BRepPartitionOperand(
                void.void_id,
                model,
                BRepPartitionRole.VOID,
                target_region_ids=void.target_region_ids,
            )
        )

    plan = BRepPartitionPlan(
        stack.coordinate_contract,
        tuple(partition_operands),
        BRepPartitionPolicy(
            stack.region_precedence,
        ),
    )
    result = partition_brep(plan, tessellation=tessellation)
    return ProcessStackResult(stack, result, tuple(operands))


__all__ = ["ProcessStackResult", "lower_process_stack"]
