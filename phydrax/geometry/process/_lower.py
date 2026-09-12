#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lower final-state planar stack extrusions through the exact CAD partition seam."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from OCP.BRepBuilderAPI import (  # ty: ignore[unresolved-import]
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
)
from OCP.BRepCheck import BRepCheck_Analyzer  # ty: ignore[unresolved-import]
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism  # ty: ignore[unresolved-import]
from OCP.gp import gp_Pnt, gp_Vec  # ty: ignore[unresolved-import]

from ..._fingerprint import canonical_fingerprint
from .._cad_revision import AssociationGraph, CADRevision
from ..brep._model import BRepEntityId, BRepModel
from ..brep._occt import persist_occt_shape
from ..brep._partition import (
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
from ..simplicial import PlanarMeshRegion
from ._stack import ProcessStack, ZInterval


@dataclass(frozen=True, slots=True)
class ProcessStackResult:
    """Persisted operands and authoritative final CAD partition history."""

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
                "Every persisted operand must use the stack coordinate contract."
            )
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
    vertices = np.asarray(footprint.vertices, dtype=float)
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


def _wire(points: np.ndarray, z: float, /):
    builder = BRepBuilderAPI_MakePolygon()
    for x, y in points:
        builder.Add(gp_Pnt(float(x), float(y), z))
    builder.Close()
    if not builder.IsDone():
        raise ValueError("OCCT could not construct a process footprint wire.")
    return builder.Wire()


def _vertical_extrusion(footprint: PlanarMeshRegion, interval: ZInterval, /):
    loops = _footprint_loops(footprint)
    face_builder = BRepBuilderAPI_MakeFace(_wire(loops[0], interval.lower), True)
    for hole in loops[1:]:
        face_builder.Add(_wire(hole, interval.lower))
    if not face_builder.IsDone():
        raise ValueError("OCCT could not construct a process footprint face.")
    prism = BRepPrimAPI_MakePrism(
        face_builder.Face(),
        gp_Vec(0.0, 0.0, interval.height),
        True,
    )
    if not prism.IsDone():
        raise ValueError("OCCT could not construct a vertical process extrusion.")
    shape = prism.Shape()
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("Vertical process extrusion is not a valid OCCT solid.")
    return shape


def _persist_operand(
    operand_id: str,
    fingerprint: str,
    footprint: PlanarMeshRegion,
    interval: ZInterval,
    stack: ProcessStack,
    directory: Path,
    index: int,
    *,
    overwrite: bool,
    linear_deflection: float,
    angular_deflection: float,
    trim_samples_per_edge: int,
) -> tuple[str, BRepModel]:
    destination = directory / f"{index:04d}-{fingerprint[:16]}.brep"
    model = persist_occt_shape(
        _vertical_extrusion(footprint, interval),
        destination,
        coordinate_contract=stack.coordinate_contract,
        overwrite=overwrite,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
    )
    if model.topology.num_solids != 1:
        raise RuntimeError("A persisted process operand must contain exactly one solid.")
    return operand_id, model


def lower_process_stack(
    stack: ProcessStack,
    destination: str | Path,
    /,
    *,
    operand_directory: str | Path | None = None,
    overwrite: bool = False,
    run_parallel: bool = False,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
) -> ProcessStackResult:
    """Persist exact extrusions, partition them, and return exact named CAD history."""

    if not isinstance(stack, ProcessStack):
        raise TypeError("stack must be a ProcessStack.")
    if not isinstance(overwrite, bool) or not isinstance(run_parallel, bool):
        raise TypeError("overwrite and run_parallel must be boolean.")
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() not in {".brep", ".brp"}:
        raise ValueError("A process stack destination requires a .brep or .brp suffix.")
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    operand_root = (
        target.parent / f".{target.stem}-operands"
        if operand_directory is None
        else Path(operand_directory).expanduser().resolve()
    )
    if operand_root == target:
        raise ValueError("operand_directory must differ from the result destination.")

    persisted: list[tuple[str, BRepModel]] = []
    partition_operands: list[BRepPartitionOperand] = []
    for index, region in enumerate(stack.regions):
        operand_id, model = _persist_operand(
            region.region_id,
            region.stack_region_id,
            region.footprint,
            region.z_interval,
            stack,
            operand_root,
            index,
            overwrite=overwrite,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            trim_samples_per_edge=trim_samples_per_edge,
        )
        persisted.append((operand_id, model))
        partition_operands.append(
            BRepPartitionOperand(
                operand_id,
                model,
                BRepPartitionRole.REGION,
            )
        )
    for void_index, void in enumerate(stack.voids, start=len(stack.regions)):
        operand_id, model = _persist_operand(
            void.void_id,
            void.stack_void_id,
            void.footprint,
            void.z_interval,
            stack,
            operand_root,
            void_index,
            overwrite=overwrite,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            trim_samples_per_edge=trim_samples_per_edge,
        )
        persisted.append((operand_id, model))
        partition_operands.append(
            BRepPartitionOperand(
                operand_id,
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
            overwrite=overwrite,
            run_parallel=run_parallel,
        ),
    )
    result = partition_brep(
        plan,
        destination=target,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
    )
    return ProcessStackResult(stack, result, tuple(persisted))


__all__ = ["ProcessStackResult", "lower_process_stack"]
