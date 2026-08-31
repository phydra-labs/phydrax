#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _compiled_dem():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]), jnp.asarray([1.0, 1.0]), ambient_dimension=2
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.1, 0.1]), jnp.asarray([0, 0])
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([1.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.9]]),
        jnp.asarray([[0.0]]),
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e3)
        )
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "coupled", materials, gravity=jnp.zeros((2,))
    )
    compiled = phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[-0.25, 0.0], [0.25, 0.0]]),
        jnp.zeros((2, 2)),
    )
    transfer = phx.discretization.ConservativeParticleGridTransferPlan(
        jnp.asarray([[-0.25, 0.0], [0.25, 0.0]]),
        jnp.ones((2,)),
        0.3,
        1,
    ).prepare(particles)
    return compiled, state, transfer


def test_particle_grid_transfer_and_unresolved_coupling_conserve_content():
    compiled, state, transfer = _compiled_dem()
    relation = transfer.relation(state.kinematics.position)
    content = jnp.asarray([2.0, 3.0])
    deposited = transfer.deposit_particle_content(relation, content)
    gathered = transfer.gather(relation, deposited)
    assert relation.successful
    assert jnp.isclose(jnp.sum(deposited), jnp.sum(content))
    assert jnp.allclose(gathered, content)

    coupling = phx.equations.UnresolvedCFDEMCouplingPlan(
        compiled.dynamics,
        transfer,
        phx.equations.StokesDragPlan(maximum_reynolds=1.0),
    )
    evaluation = phx.equations.evaluate_unresolved_cfd_dem(
        coupling,
        state,
        jnp.zeros((2, 2)),
        jnp.ones((2,)),
        jnp.full((2,), 10.0),
        jnp.zeros((2, 2)),
        jnp.full((2,), 0.01),
        jnp.asarray(1.0e-3),
    )
    assert evaluation.successful
    assert jnp.allclose(evaluation.particle_force, 0.0)
    assert jnp.allclose(evaluation.momentum_residual, 0.0)


def test_multirate_window_commits_equal_opposite_impulses_atomically():
    compiled, state, transfer = _compiled_dem()
    coupling = phx.equations.UnresolvedCFDEMCouplingPlan(
        compiled.dynamics,
        transfer,
        phx.equations.StokesDragPlan(maximum_reynolds=1.0),
    )
    coupled_state = phx.solver.CFDEMCouplingState(
        state,
        jnp.zeros((2, 2)),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.zeros(()),
        jnp.zeros((), dtype=jnp.int32),
    )
    result = phx.solver.advance_cfd_dem_window(
        coupling,
        phx.solver.CFDEMCouplingSchedulePlan(2),
        coupled_state,
        jnp.full((2, 2), 0.01),
        jnp.ones((2,)),
        jnp.full((2,), 10.0),
        jnp.zeros((2, 2)),
        jnp.full((2,), 0.01),
        jnp.asarray(0.0),
        jnp.asarray(2.0e-4),
        lambda fluid, impulse, dt: fluid + impulse,
    )
    assert result.successful
    assert jnp.linalg.norm(result.momentum_residual) < 1.0e-10
    assert result.accepted_state.accepted_windows == 1


def test_resolved_ib_work_adjoint_is_conservative():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.asarray([1.0]), ambient_dimension=2
    ).prepare()
    bodies = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.1]), jnp.asarray([0])
    ).prepare(particles)
    kinematics = bodies.kinematics(jnp.asarray([[0.0, 0.0]]), jnp.zeros((1, 2)))
    geometry = phx.equations.ResolvedIBGeometryPlan(
        jnp.asarray([[0.0, 0.0]]),
        jnp.asarray([0]),
        jnp.asarray([1.0]),
        jnp.asarray([[0.0, 0.0]]),
        0.5,
        1,
    )
    ib = phx.equations.ResolvedIBCFDEMCouplingPlan(
        bodies, geometry, phx.equations.IBConstraintPlan(10.0)
    )
    evaluation = phx.equations.evaluate_resolved_ib_cfd_dem(
        ib, kinematics, jnp.zeros((1, 2))
    )
    assert evaluation.successful
    assert jnp.isclose(evaluation.work_adjoint_residual, 0.0)
