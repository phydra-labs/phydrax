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
    mesh = phx.discretization.CellMesh(
        jnp.asarray(
            (
                (-0.4, -0.1),
                (-0.1, -0.1),
                (-0.25, 0.2),
                (0.1, -0.1),
                (0.4, -0.1),
                (0.25, 0.2),
            )
        ),
        (
            phx.discretization.CellBlock(
                "cells",
                "triangle",
                jnp.asarray(((0, 1, 2), (3, 4, 5)), dtype=jnp.int32),
            ),
        ),
    )
    measure = phx.discretization.DiscreteMeasure(
        "cell_volume",
        mesh.support.support_id,
        mesh.topology.entities(2).entity_set_id,
        jnp.ones((2,)),
    )
    transfer = phx.discretization.MeshCompactKernelSplatAssignment(
        0.3, 1, partition_policy="normalize"
    ).prepare(
        phx.discretization.MeshSplatTarget(mesh, entity_dimension=2, measure=measure),
        state.kinematics.position,
        state.body_properties.active,
        particles.particle_ids,
    )
    return compiled, state, transfer


def test_particle_grid_transfer_and_unresolved_coupling_conserve_content():
    compiled, state, transfer = _compiled_dem()
    relation = transfer.routes(state.kinematics.position, state.body_properties.active)
    content = jnp.asarray([2.0, 3.0])
    deposited = transfer.deposit(
        state.kinematics.position, state.body_properties.active, content
    )
    gathered = transfer.gather(
        state.kinematics.position,
        state.body_properties.active,
        deposited.content,
    )
    assert jnp.all(relation.evidence.complete)
    assert jnp.isclose(jnp.sum(deposited.content), jnp.sum(content))
    assert jnp.allclose(gathered.values, content)

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


def test_mac_penalty_ib_window_preserves_zero_load_and_projection():
    dem, dem_state, _ = _compiled_dem()
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(6, periodic=True) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[-0.5, -0.5], [0.5, 0.5]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    fluid = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        momentum,
        phx.solver.MACPressureProjectionPlan(operators, solve_method="transform"),
    )
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.asarray([0, 1]),
        jnp.zeros((2, 2)),
        jnp.ones((2,)),
    ).prepare()
    face_transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    coupling = phx.equations.MACPenaltyIBCFDEMCouplingPlan(
        fluid,
        dem.dynamics,
        jnp.asarray([0, 1]),
        phx.equations.IBPenaltyPlan(1.0),
        face_transfer,
    )
    zero_velocity = tuple(
        jnp.zeros(layout.shape) for layout in finite_volume.face_layouts
    )
    evaluation = phx.equations.evaluate_mac_penalty_ib_cfd_dem(
        coupling, dem_state.kinematics, zero_velocity, jnp.asarray(1.0e-4)
    )
    coupled_state = phx.solver.MACPenaltyIBCouplingState.initialize(
        coupling, dem_state, fluid.project_state(zero_velocity)
    )
    step = phx.solver.advance_mac_penalty_ib_cfd_dem_window(
        coupling,
        phx.solver.MACPenaltyIBCouplingSchedulePlan(1),
        coupled_state,
        jnp.asarray(0.0),
        jnp.asarray(1.0e-4),
    )

    assert evaluation.successful
    assert jnp.abs(evaluation.work_adjoint_residual) < 1e-10
    assert step.successful
    accepted_velocity = fluid.unpack_velocity(step.accepted_state.fluid_state)
    assert jnp.linalg.norm(operators.divergence(accepted_velocity)) < 1e-8
