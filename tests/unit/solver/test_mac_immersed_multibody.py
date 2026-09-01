#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _flow(count=6):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(2)
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
    offsets = jnp.asarray([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(4), offsets, jnp.full((4,), 0.25)
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
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
    method = phx.solver.MACRigidImmersedEulerMethod(dynamics, projection, 1.0e-3)
    coupled = method.step(0.0, dynamics.project_state(zero), kinematics)
    backward = phx.solver.MACRigidImmersedBackwardEulerMethod(
        method, maximum_iterations=1, tolerance=1.0e-8
    )
    midpoint = phx.solver.MACRigidImmersedMidpointMethod(backward).step(
        0.0, dynamics.project_state(zero), kinematics
    )
    hard_contact = phx.discretization.HardContactRoutePlan(
        jnp.asarray([0]),
        jnp.asarray([-1]),
        jnp.asarray([17]),
        position_stabilization=0.0,
    ).prepare(bodies)

    def separated_geometry(_kinematics):
        normal = jnp.asarray([[0.0, 1.0]])
        zero_vector = jnp.zeros((1, 2))
        zero_scalar = jnp.zeros((1,))
        zero_angular = jnp.zeros((1, 1))
        zero_index = jnp.zeros((1,), dtype=jnp.int32)
        return phx.discretization.RigidContactGeometry(
            normal,
            jnp.asarray([0.1]),
            zero_scalar,
            jnp.ones((1,)),
            zero_vector,
            zero_vector,
            zero_vector,
            zero_vector,
            zero_vector,
            zero_vector,
            zero_scalar,
            zero_vector,
            zero_angular,
            zero_angular,
            jnp.asarray([17], dtype=jnp.int32),
            zero_index,
            zero_index,
            jnp.asarray([True]),
            zero_index,
            jnp.asarray([0.1]),
            jnp.asarray(True),
            "separated-rigid-contact",
        )

    contacted = phx.solver.MACRigidImmersedContactMethod(
        backward, hard_contact, maximum_iterations=1, tolerance=1.0e-8
    ).step(
        0.0,
        dynamics.project_state(zero),
        kinematics,
        hard_contact.initial_state(),
        separated_geometry,
    )

    assert result.successful
    assert jnp.linalg.norm(result.divergence) < 1.0e-8
    assert jnp.linalg.norm(result.marker_slip) < 1.0e-8
    assert jnp.linalg.norm(result.body_kinematics.velocity) < 1.0e-10
    assert coupled.successful
    assert jnp.linalg.norm(coupled.projection.marker_slip) < 1.0e-8
    assert contacted.accepted
    assert contacted.contact.successful
    assert midpoint.accepted
    assert jnp.linalg.norm(midpoint.projection.marker_slip) < 1.0e-8


def test_deformable_backward_euler_preserves_zero_state_and_energy():
    finite_volume, operators, boundaries, dynamics, zero = _flow()
    marker_position = jnp.asarray([[0.35, 0.5], [0.65, 0.5]])
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(2), marker_position, jnp.asarray([0.5, 0.5])
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
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
    newmark = phx.solver.MACDeformableImmersedNewmarkMethod(method)
    newmark_state = newmark.initialize(
        fluid_state, configuration, jnp.zeros_like(configuration)
    )
    newmark_result = newmark.step(0.0, newmark_state)

    assert result.successful
    assert result.route_unchanged
    assert jnp.linalg.norm(result.divergence) < 1.0e-8
    assert jnp.linalg.norm(result.marker_slip) < 1.0e-8
    assert jnp.abs(result.energy.coupling_power_residual) < 1.0e-8
    assert newmark_result.successful
    assert newmark_result.route_unchanged
    assert jnp.linalg.norm(newmark_result.divergence) < 1.0e-8
    assert jnp.linalg.norm(newmark_result.marker_slip) < 1.0e-8
