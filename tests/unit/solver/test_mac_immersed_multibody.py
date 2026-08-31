#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _flow(count=6):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=boundaries
    ).prepare()
    pressure = phx.solver.MACPressureProjectionPlan(
        operators, boundaries=boundaries, solve_method="transform"
    )
    dynamics = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01), momentum, pressure
    )
    zero = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    return finite_volume, operators, boundaries, dynamics, zero


def test_free_rigid_projection_preserves_zero_coupled_state():
    finite_volume, operators, _, dynamics, zero = _flow()
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.asarray([1.0]), ambient_dimension=2
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.asarray([0]), jnp.asarray([0.1])
    ).prepare(particles)
    offsets = jnp.asarray(
        [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]
    )
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(4), offsets, jnp.full((4,), 0.25)
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(
        operators, markers
    ).prepare()
    rigid_map = phx.discretization.RigidMarkerMapPlan(
        markers, bodies, jnp.zeros((4,), dtype=jnp.int32)
    ).prepare()
    kinematics = bodies.kinematics(
        jnp.asarray([[0.5, 0.5]]),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
    )
    projection = phx.solver.MACRigidImmersedProjectionPlan(
        dynamics,
        rigid_map,
        transfer,
        constraint_length=1.0 / finite_volume.cell_shape[0],
        tolerance=1.0e-8,
    )
    result = projection.project(zero, kinematics, 1.0e-3)
    method = phx.solver.MACRigidImmersedEulerMethod(
        dynamics, projection, 1.0e-3
    )
    coupled = method.step(
        0.0, dynamics.project_state(zero), kinematics
    )

    assert result.successful
    assert jnp.linalg.norm(result.divergence) < 1.0e-8
    assert jnp.linalg.norm(result.marker_slip) < 1.0e-8
    assert jnp.linalg.norm(result.body_kinematics.velocity) < 1.0e-10
    assert coupled.successful
    assert jnp.linalg.norm(coupled.projection.marker_slip) < 1.0e-8


def test_deformable_backward_euler_preserves_zero_state_and_energy():
    finite_volume, operators, boundaries, dynamics, zero = _flow()
    marker_position = jnp.asarray([[0.35, 0.5], [0.65, 0.5]])
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(2), marker_position, jnp.asarray([0.5, 0.5])
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(
        operators, markers
    ).prepare()
    exact = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators, transfer, boundaries=boundaries, tolerance=1.0e-8
    )
    configuration_space = phx.linalg.ArraySpace((4,))
    marker_map = phx.discretization.FiniteElementImmersedMarkerMapPlan(
        markers, configuration_space, jnp.eye(4)
    ).prepare()
    structure = phx.dynamics.SecondOrderDifferentialSystem(
        lambda _time, _q, _v, acceleration, _args: acceleration,
        state_shape=(4,),
        system_id="zero-marker-mass",
    )
    method = phx.solver.MACDeformableImmersedBackwardEulerMethod(
        dynamics,
        exact,
        marker_map,
        structure,
        lambda q, _args: 0.5 * jnp.sum(q * 0.0),
        1.0e-3,
        energy_id="zero-energy",
    )
    fluid_state = dynamics.project_state(zero)
    configuration = marker_position.reshape((-1,))
    state = method.initialize(fluid_state, configuration, jnp.zeros_like(configuration))
    result = method.step(0.0, state)

    assert result.successful
    assert result.route_unchanged
    assert jnp.linalg.norm(result.divergence) < 1.0e-8
    assert jnp.linalg.norm(result.marker_slip) < 1.0e-8
    assert jnp.abs(result.energy.coupling_power_residual) < 1.0e-8
