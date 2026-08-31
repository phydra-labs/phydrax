#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax as phx


def _parts(*, assignment=None, boundary="reject", cell_primary=False):
    axis_type = (
        phx.discretization.UniformCellAxisSpec
        if cell_primary
        else phx.discretization.UniformAxisSpec
    )
    axes = tuple(
        axis_type(10, periodic=True, **({} if cell_primary else {"endpoint": False}))
        for _ in range(2)
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y")).prepare(
        jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), jnp.ones((3,)), ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=(
            phx.discretization.TensorBSplineSplatAssignment(2)
            if assignment is None
            else assignment
        ),
        boundary=boundary,
    ).prepare(particles)
    domain = phx.discretization.MPMParticleDomainPlan(
        jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
        periodic=(True, True),
        support_margin=0.0,
    )
    problem = phx.equations.MaterialPointProblemIR(
        "compile-test",
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
    )
    return problem, particles, splat, domain


def test_material_point_compiler_records_complete_dependency_bundle():
    problem, particles, splat, domain = _parts()
    compiled = phx.equations.compile_material_point_problem(
        problem,
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        domain,
    )

    assert len(compiled.discretization_bundle.records) == 3
    assert compiled.dynamics.splat.prepared_id == splat.prepared_id
    assert compiled.dynamics.particles.prepared_id == particles.prepared_id
    assert compiled.dynamics.resource_evidence.step_workspace_bytes > 0
    assert compiled.compilation_id


@pytest.mark.parametrize(
    ("assignment", "message"),
    [
        (phx.discretization.MultilinearSplatAssignment(), "TensorBSpline"),
        (phx.discretization.TensorBSplineSplatAssignment(3), "quadratic"),
    ],
)
def test_material_point_compiler_rejects_unqualified_assignments(assignment, message):
    problem, particles, splat, domain = _parts(assignment=assignment)
    with pytest.raises((TypeError, ValueError), match=message):
        phx.equations.compile_material_point_problem(
            problem,
            particles,
            splat,
            phx.discretization.ExplicitMPMMethodPlan(),
            domain,
        )


def test_material_point_compiler_rejects_cell_targets_and_drop_boundaries():
    problem, particles, splat, domain = _parts(cell_primary=True)
    with pytest.raises(ValueError, match="nodal"):
        phx.equations.compile_material_point_problem(
            problem,
            particles,
            splat,
            phx.discretization.ExplicitMPMMethodPlan(),
            domain,
        )

    problem, particles, splat, domain = _parts(boundary="drop")
    with pytest.raises(ValueError, match="boundary='reject'"):
        phx.equations.compile_material_point_problem(
            problem,
            particles,
            splat,
            phx.discretization.ExplicitMPMMethodPlan(),
            domain,
        )


def test_nonperiodic_compilation_requires_declared_complete_halo():
    axes = tuple(phx.discretization.UniformAxisSpec(13) for _ in range(2))
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y")).prepare(
        jnp.asarray([[-0.1, -0.1], [1.1, 1.1]])
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(particles)
    domain = phx.discretization.MPMParticleDomainPlan(
        jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
        support_margin=0.1,
    )
    problem = phx.equations.MaterialPointProblemIR(
        "halo-test",
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
    )

    with pytest.raises(ValueError, match="complete declared support halo"):
        phx.equations.compile_material_point_problem(
            problem,
            particles,
            splat,
            phx.discretization.ExplicitMPMMethodPlan(),
            domain,
        )
