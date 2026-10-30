from __future__ import annotations

import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellMesh
from phydrax.meshing._adaptation import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.meshing._canonical import certify_cell_mesh

from .._runtime import require_meshcore, runtime_identity


def _linear(points: np.ndarray, /) -> np.ndarray:
    return 2.0 + 3.0 * points[:, 0] - 1.5 * points[:, 1]


def _p1_integral(mesh: CellMesh, values: np.ndarray, /) -> float:
    triangles = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks], axis=0
    )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    corners = points[triangles]
    areas = 0.5 * np.abs(
        (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
        - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0])
    )
    return float(np.sum(areas * np.mean(values[triangles], axis=1)))


def native_transfer() -> dict[str, object]:
    meshcore = require_meshcore()
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "unit-square",
                "triangle",
                np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
            ),
        ),
        numeric_version="native-transfer-source",
    )
    source = certify_cell_mesh(mesh, SpatialCoordinateContract.si())
    marks = np.asarray((mesh.blocks[0].global_ids[0],), dtype=np.int64)
    prepared = prepare_mesh_adaptation(
        source,
        MarkedMeshAdaptation(marks),
        policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION),
    )
    adaptation = execute_mesh_adaptation(prepared)
    target = adaptation.target
    transition = adaptation.transition
    if transition is None:
        raise RuntimeError(
            "Native bisection did not retain its accepted mesh transition."
        )
    stencil = transition.vertex_stencil
    if stencil is None:
        raise RuntimeError("Native bisection did not retain its vertex transfer stencil.")
    source_values = _linear(np.asarray(source.mesh.coordinates, dtype=np.float64))
    transferred = np.asarray(
        stencil.apply(
            np.asarray(source.mesh.vertex_global_ids, dtype=np.int64), source_values
        ),
        dtype=np.float64,
    )
    target_values = _linear(np.asarray(target.mesh.coordinates, dtype=np.float64))
    reproduction_error = float(np.max(np.abs(transferred - target_values)))
    source_integral = _p1_integral(source.mesh, source_values)
    target_integral = _p1_integral(target.mesh, transferred)
    integral_error = abs(target_integral - source_integral)
    if reproduction_error > 1.0e-12 or integral_error > 1.0e-12:
        raise RuntimeError(
            f"Native P1 transfer failed linear/conservative oracle: {reproduction_error=}, {integral_error=}."
        )
    if (
        not target.audit.passed
        or target.certification is None
        or not target.certification.passed
    ):
        raise RuntimeError("Native transfer target lost its accepted mesh/certification.")
    return {
        "status": "passed",
        "route": MeshAdaptationRoute.NATIVE_BISECTION.value,
        "source_result_id": source.result_id,
        "target_result_id": target.result_id,
        "source_topology_id": source.mesh.topology_id,
        "target_topology_id": target.mesh.topology_id,
        "source_integral": source_integral,
        "target_integral": target_integral,
        "integral_absolute_error": integral_error,
        "linear_reproduction_error": reproduction_error,
        "transfer_id": stencil.stencil_id,
        "target_cells": sum(block.cell_count for block in target.mesh.blocks),
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
