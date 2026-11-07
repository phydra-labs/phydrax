"""Periodic cell homogenization and lattice design realization consumer types."""

from typing import assert_type

from jax import Array

import phydrax as phx


def periodic_cell_problems(mesh: phx.discretization.CellMesh) -> None:
    space = phx.discretization.fem.PeriodicSimplexSpace(mesh)
    assert_type(space.component_winding_ranks, tuple[int, ...])
    pattern = phx.discretization.fem.PeriodicSimplexOperatorPattern(space, components=3)
    assert_type(pattern.free_count, int)
    problem = phx.applications.solid_mechanics.PeriodicCellProblem(
        space, physics="elasticity"
    )
    stiffness = phx.applications.solid_mechanics.isotropic_elastic_stiffness(1.0, 0.3)
    assert_type(stiffness, Array)
    result = problem.solve(stiffness)
    assert_type(result, phx.applications.solid_mechanics.PeriodicHomogenizationResult)
    assert_type(result.effective_tensor, Array)
    assert_type(result.cubic_symmetry_defect, Array)
    assert_type(
        result.hill_mandel,
        tuple[phx.applications.solid_mechanics.HillMandelEvidence, ...],
    )
    assert_type(
        problem.cases,
        tuple[phx.applications.solid_mechanics.PeriodicHomogenizationCase, ...],
    )
    phx.applications.solid_mechanics.PeriodicCellProblem(space, physics="flow")  # ty: ignore[invalid-argument-type]


def lattice_design(
    mesh: phx.discretization.CellMesh, source: phx.geometry.TrigonometricLevelSet
) -> None:
    plan = phx.meshing.prepare_periodic_implicit_realization(
        mesh, source, region_blocks=("network",)
    )
    assert_type(plan, phx.meshing.PeriodicImplicitRealizationPlan)
    state = plan.base_state.updated({plan.parameter_id("level"): 0.1})
    realization = plan.realize(state)
    assert_type(realization, phx.meshing.PeriodicImplicitRealization)
    assert_type(
        realization.require_sensitivities(), phx.meshing.PeriodicImplicitRealization
    )
    assert_type(realization.volume_fraction, Array)
    assert_type(plan.assess(realization), phx.meshing.MeshMotionAssessment)
    problem = phx.applications.solid_mechanics.PeriodicCellProblem(
        mesh, physics="conductivity"
    )
    assert_type(
        problem.solve_realization(2.0, realization),
        phx.applications.solid_mechanics.PeriodicHomogenizationResult,
    )
