#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lower final-state planar stack extrusions through CAD partitioning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ...interchange._cad_archive import load_brep_archive, save_brep_archive
from ...typing import checked
from .._cad_revision import AssociationGraph, CADRevision
from ..brep._constructors import (
    brep_extrusion,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from ..brep._model import BRepEntityId, BRepModel
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
from ..design._naming import name_partition_entities, RecipeEntityName, RecipeEntityNames
from ..simplicial import PlanarMeshRegion
from ._stack import ProcessStack, ZInterval


@dataclass(frozen=True, slots=True)
class ProcessStackResult:
    """Persisted operand locations and relocation-invariant CAD partition history."""

    stack: ProcessStack
    partition: BRepPartitionResult
    operand_models: tuple[tuple[str, BRepModel], ...]
    operand_artifact_paths: tuple[tuple[str, Path], ...]
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.stack, ProcessStack):
            raise TypeError(
                f"ProcessStackResult.stack must be a ProcessStack; got {type(self.stack).__name__}"
            )
        if not isinstance(self.partition, BRepPartitionResult):
            raise TypeError(
                f"ProcessStackResult.partition must be a BRepPartitionResult; got {type(self.partition).__name__}"
            )
        operands = tuple(self.operand_models)
        expected_ids = tuple(value.region_id for value in self.stack.regions) + tuple(
            value.void_id for value in self.stack.voids
        )
        if tuple(name for name, _ in operands) != expected_ids:
            raise ValueError(
                f"ProcessStackResult.operand_models must follow stack operand order {expected_ids!r}; got {tuple(name for name, _ in operands)!r}"
            )
        if not all(isinstance(model, BRepModel) for _, model in operands):
            raise TypeError(
                f"ProcessStackResult.operand_models must contain BRepModel values; got {tuple(type(model).__name__ for _, model in operands)!r}"
            )
        artifacts = tuple(self.operand_artifact_paths)
        if tuple(name for name, _ in artifacts) != expected_ids:
            raise ValueError(
                "ProcessStackResult.operand_artifact_paths must follow the stack operand order; "
                f"got {tuple(name for name, _ in artifacts)}"
            )
        if not all(isinstance(path, Path) for _, path in artifacts):
            raise TypeError(
                "ProcessStackResult.operand_artifact_paths must contain Path values; "
                f"got {tuple(type(path).__name__ for _, path in artifacts)}"
            )
        spatial_id = self.stack.coordinate_contract.spatial_id
        if any(
            model.coordinate_contract.spatial_id != spatial_id for _, model in operands
        ):
            raise ValueError(
                f"ProcessStackResult.operand_models must use stack coordinate contract {spatial_id!r}; got inconsistent operand contracts"
            )
        if self.partition.model.coordinate_contract.spatial_id != spatial_id:
            raise ValueError(
                f"ProcessStackResult.partition must use stack coordinate contract {spatial_id!r}; got {self.partition.model.coordinate_contract.spatial_id!r}"
            )
        if tuple(value.name for value in self.partition.regions) != tuple(
            value.region_id
            for value in sorted(
                self.stack.regions,
                key=lambda region: region.precedence,
                reverse=True,
            )
        ):
            raise ValueError(
                f"ProcessStackResult.partition must preserve stack region precedence; got {tuple(value.name for value in self.partition.regions)!r}"
            )
        object.__setattr__(self, "operand_models", operands)
        object.__setattr__(self, "operand_artifact_paths", artifacts)
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

    def recipe_names(
        self, source_names: Mapping[str, RecipeEntityName], /
    ) -> RecipeEntityNames:
        """Bind authored recipe names through exhaustive partition lineage.

        Source keys are occurrences in ``association_graph.source_revision``.
        Three-dimensional partition edge ancestry is unsupported and explicitly
        refused in the returned binding record.
        """
        return name_partition_entities(self.partition, source_names)


def _footprint_loops(footprint: PlanarMeshRegion, /) -> tuple[np.ndarray, ...]:
    vertices = np.asarray(footprint.vertices, dtype=np.float64)
    edges = np.asarray(footprint.edges, dtype=np.int64)
    offsets = np.asarray(footprint.loop_offsets, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.all(np.isfinite(vertices)):
        raise ValueError(
            f"ProcessStack.footprint_vertices must be finite with shape (N, 2); got shape {vertices.shape}"
        )
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != edges.shape[0]
        or np.any(offsets[1:] <= offsets[:-1])
    ):
        raise ValueError(
            f"ProcessStack.footprint_topology must have aligned edge and strictly increasing loop-offset arrays; got shapes {(edges.shape, offsets.shape)}"
        )
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
            raise ValueError(
                f"ProcessStack.footprint_loops must be closed edge cycles with at least three valid vertices; got invalid loop {index}"
            )
        loops.append(vertices[vertex_ids])
    return tuple(loops)


def _vertical_extrusion(
    footprint: PlanarMeshRegion,
    interval: ZInterval,
    /,
    *,
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy,
    source_id: str,
) -> BRepModel:
    loops = _footprint_loops(footprint)
    profile = PlanarProfile(
        ProfilePlane(origin=(0.0, 0.0, interval.lower)),
        ProfileLoop.polygon(loops[0]),
        tuple(ProfileLoop.polygon(hole) for hole in loops[1:]),
    )
    return brep_extrusion(
        profile,
        (0.0, 0.0, interval.height),
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
    )


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
) -> tuple[str, BRepModel, Path]:
    destination = directory / f"{index:04d}-{fingerprint[:16]}.phx"
    tessellation = BRepTessellationPolicy(
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
    )
    model = _vertical_extrusion(
        footprint,
        interval,
        coordinate_contract=stack.coordinate_contract,
        tessellation=tessellation,
        source_id=f"process-operand:{fingerprint}",
    )
    receipt = save_brep_archive(
        model,
        destination,
        mode="atomic_replace" if overwrite else "exclusive",
    )
    model = load_brep_archive(receipt.path)
    if model.topology.num_solids != 1:
        raise RuntimeError(
            f"ProcessStack.operand_model must contain one solid; got {model.topology.num_solids}"
        )
    return operand_id, model, receipt.path


@checked
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
    """Persist planar extrusions, partition them, and return named CAD history.

    Operands are lossless native ``.phx`` B-Rep archives in ``operand_directory``;
    ``operand_artifact_paths`` records their locations independently of model
    source identities. The returned operand models are restored from those
    archives. Coordinates, wall/cap/solid topology and source identities
    retain the stack's physical coordinate contract. The final destination is
    external B-Rep text.
    """

    if not isinstance(overwrite, bool) or not isinstance(run_parallel, bool):
        raise TypeError(
            f"lower_process_stack.options must have boolean overwrite and run_parallel; got {(type(overwrite).__name__, type(run_parallel).__name__)}"
        )
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() not in {".brep", ".brp"}:
        raise ValueError(
            f"lower_process_stack.destination must have a .brep or .brp suffix; got {target.suffix!r}"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    operand_root = (
        target.parent / f".{target.stem}-operands"
        if operand_directory is None
        else Path(operand_directory).expanduser().resolve()
    )
    if operand_root == target:
        raise ValueError(
            f"lower_process_stack.operand_directory must differ from the result destination; got {operand_root!s}"
        )

    persisted: list[tuple[str, BRepModel]] = []
    artifacts: list[tuple[str, Path]] = []
    partition_operands: list[BRepPartitionOperand] = []
    for index, region in enumerate(stack.regions):
        operand_id, model, artifact_path = _persist_operand(
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
        artifacts.append((operand_id, artifact_path))
        partition_operands.append(
            BRepPartitionOperand(
                operand_id,
                model,
                BRepPartitionRole.REGION,
            )
        )
    for void_index, void in enumerate(stack.voids, start=len(stack.regions)):
        operand_id, model, artifact_path = _persist_operand(
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
        artifacts.append((operand_id, artifact_path))
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
    return ProcessStackResult(stack, result, tuple(persisted), tuple(artifacts))


__all__ = ["ProcessStackResult", "lower_process_stack"]
