#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider route of owner-local planar generation on a fixed tile complex."""

from __future__ import annotations

from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ...typing import checked
from .._contracts import MeshingProviderInfo, SurfaceMeshingSpec
from .._distributed_planar_generation import (
    execute_distributed_planar_initial,
    PreparedDistributedPlanarGeneration,
)
from .._initial_certification import publish_distributed_planar_initial
from .._measurements import measure_phase, NativeMeshingPhaseRecorder
from .._result import CellMeshingResult
from . import _native_sources as sources_owner


@checked
def execute_distributed_planar_route(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedDistributedPlanarGeneration,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    """Refine owned tiles, then certify and publish them collectively.

    The plan's request scope and workspace stay open around the owner-local
    native scope; every owner refuses together before any result exists.
    """
    del scope
    prepared.require_binding(source, specification)
    workspace.retain_owner(prepared.tiles)
    with measure_phase(record_phase, "construction"):
        generated = execute_distributed_planar_initial(prepared)
    quality = specification.quality_target
    required_angle = 0.0 if quality is None or not quality.hard else quality.minimum_angle
    with measure_phase(record_phase, "certification"):
        return publish_distributed_planar_initial(
            prepared,
            generated,
            coordinate_contract,
            provider,
            minimum_angle=required_angle,
            plan_id=plan_id,
        )


__all__ = ["execute_distributed_planar_route"]
