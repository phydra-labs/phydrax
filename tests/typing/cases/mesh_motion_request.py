"""Motion assessments accept the actual optional request ledger and expose settlement."""

from typing import assert_type

from jax.typing import ArrayLike

from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.meshing import (
    advance_mesh_motion,
    CellMeshingResult,
    MeshAdaptationPolicy,
    MeshMotionAdvance,
    MeshMotionAssessment,
    MeshMotionMonitor,
    MeshOptimizationResult,
    optimize_cell_mesh,
    TargetMatrixOptimizationPlan,
)
from phydrax.meshing._measurements import NativeExecutionRecord


def motion_request_contract(
    monitor: MeshMotionMonitor,
    source: CellMeshingResult,
    coordinates: ArrayLike,
    policy: MeshAdaptationPolicy,
    budget: CoordinateEnclosureBudget,
) -> None:
    assert_type(
        monitor.assess(coordinates, boundary_residual=0.0), MeshMotionAssessment
    )
    assert_type(
        monitor.assess(coordinates, boundary_residual=0.0, budget=budget),
        MeshMotionAssessment,
    )
    advance = advance_mesh_motion(
        monitor,
        source,
        coordinates,
        boundary_residual=0.0,
        adaptation_policy=policy,
    )
    assert_type(advance, MeshMotionAdvance)
    assert_type(advance.execution_evidence, NativeExecutionRecord | None)
    assert_type(
        advance_mesh_motion(monitor, source, coordinates, boundary_residual=0.0),
        MeshMotionAdvance,
    )


def relocation_request_contract(
    plan: TargetMatrixOptimizationPlan,
    source: CellMeshingResult,
    budget: CoordinateEnclosureBudget,
) -> None:
    assert_type(
        optimize_cell_mesh(plan, source.coordinate_contract, budget=budget),
        MeshOptimizationResult,
    )
    assert_type(
        optimize_cell_mesh(plan, source.coordinate_contract),
        MeshOptimizationResult,
    )
