#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._reduced_articulation import (
    ReducedArticulationPlan,
    ReducedArticulationState,
)
from phydrax.discretization.particle._reduced_dynamics import (
    reduced_bias_terms,
    reduced_energy,
    reduced_forward_dynamics,
    reduced_inverse_dynamics,
    reduced_mass_matrix,
    reduced_semi_implicit_velocity_euler_step,
    ReducedDynamicsStatus,
    ReducedSemiImplicitVelocityEulerStepPolicy,
)
from phydrax.discretization.particle._rigid_body import (
    quaternion_rotation_matrix,
    RigidBodyLoad,
    RigidBodySetPlan,
)
from phydrax.discretization.particle._rigid_joints import (
    HingeJointSetPlan,
    PrismaticJointSetPlan,
    RigidJointGraphPlan,
)
from phydrax.discretization.particle._rigid_parameters import (
    RigidInertialParameterization,
)


def _single_axis_articulation(kind: str):
    body_ids = jnp.asarray([10, 11], dtype=jnp.int64)
    particles = ParticleSetPlan(
        body_ids,
        jnp.asarray([1.0, 2.0]),
        ambient_dimension=3,
    ).prepare()
    inertia = jnp.stack(
        (
            jnp.eye(3),
            jnp.diag(jnp.asarray([1.0, 1.0, 3.0])),
        )
    )
    bodies = RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        inertia,
        fixed_mask=jnp.asarray([True, False]),
    ).prepare(particles)
    position = (
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        if kind == "hinge"
        else jnp.zeros((2, 3))
    )
    reference = bodies.kinematics(
        position,
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        jnp.zeros((2, 3)),
    )
    if kind == "hinge":
        joint = HingeJointSetPlan(
            jnp.asarray([20]),
            body_ids[:1],
            body_ids[1:],
            jnp.asarray([[0.0, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.0, 1.0]]),
        )
        graph = RigidJointGraphPlan(hinge=joint).prepare(bodies, reference)
    else:
        joint = PrismaticJointSetPlan(
            jnp.asarray([20]),
            body_ids[:1],
            body_ids[1:],
            jnp.asarray([[0.0, 0.0, 0.0]]),
            jnp.asarray([[1.0, 0.0, 0.0]]),
        )
        graph = RigidJointGraphPlan(prismatic=joint).prepare(bodies, reference)
    articulation = ReducedArticulationPlan(
        10,
        jnp.asarray([20]),
        jnp.asarray([10]),
        jnp.asarray([11]),
    ).prepare(graph, reference)
    return articulation


def test_single_hinge_has_analytic_mass_gravity_and_acceleration():
    articulation = _single_axis_articulation("hinge")
    q = jnp.asarray([0.0])
    v = jnp.asarray([0.0])
    gravity = jnp.asarray([0.0, -9.81, 0.0])

    mass = reduced_mass_matrix(articulation, q)
    bias = reduced_bias_terms(articulation, q, v, gravity)
    forward = reduced_forward_dynamics(
        articulation, q, v, jnp.asarray([0.0]), gravity
    )

    assert mass.successful
    assert mass.positive_definite
    assert jnp.allclose(mass.matrix, mass.matrix.T, atol=1.0e-7)
    assert jnp.allclose(mass.matrix, jnp.asarray([[5.0]]), rtol=2.0e-6)
    assert jnp.allclose(mass.operator.mv(jnp.asarray([2.0])), jnp.asarray([10.0]))
    assert bias.successful
    assert jnp.allclose(bias.velocity, 0.0, atol=1.0e-7)
    assert jnp.allclose(bias.gravity, jnp.asarray([19.62]), rtol=2.0e-6)
    assert forward.successful
    assert jnp.allclose(
        forward.acceleration, jnp.asarray([-19.62 / 5.0]), rtol=2.0e-5
    )


def test_single_prismatic_has_analytic_mass_gravity_and_acceleration():
    articulation = _single_axis_articulation("prismatic")
    q = jnp.asarray([0.0])
    v = jnp.asarray([0.0])
    gravity = jnp.asarray([-9.81, 0.0, 0.0])

    mass = reduced_mass_matrix(articulation, q)
    bias = reduced_bias_terms(articulation, q, v, gravity)
    forward = reduced_forward_dynamics(
        articulation, q, v, jnp.asarray([0.0]), gravity
    )

    assert mass.successful
    assert jnp.allclose(mass.matrix, jnp.asarray([[2.0]]), rtol=2.0e-6)
    assert jnp.allclose(bias.gravity, jnp.asarray([19.62]), rtol=2.0e-6)
    assert forward.successful
    assert jnp.allclose(forward.acceleration, jnp.asarray([-9.81]), rtol=2.0e-5)


def test_inverse_then_forward_dynamics_round_trips_with_residual_evidence():
    articulation = _single_axis_articulation("hinge")
    q = jnp.asarray([0.31])
    v = jnp.asarray([0.47])
    acceleration = jnp.asarray([-0.73])
    gravity = jnp.asarray([0.0, -9.81, 0.0])

    inverse = reduced_inverse_dynamics(
        articulation, q, v, acceleration, gravity
    )
    forward = reduced_forward_dynamics(
        articulation, q, v, inverse.generalized_effort, gravity
    )

    assert inverse.successful
    assert forward.successful
    assert jnp.allclose(forward.acceleration, acceleration, rtol=2.0e-5, atol=2.0e-6)
    assert forward.relative_inverse_forward_residual <= 1.0e-6


def test_external_body_load_pullback_preserves_power():
    articulation = _single_axis_articulation("hinge")
    q = jnp.asarray([0.0])
    v = jnp.asarray([0.5])
    zero = jnp.asarray([0.0, 0.0, 0.0])
    load = RigidBodyLoad(
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )

    inverse = reduced_inverse_dynamics(
        articulation,
        q,
        v,
        jnp.asarray([0.0]),
        zero,
        external_load=load,
    )

    assert inverse.successful
    assert jnp.allclose(inverse.external_effort, jnp.asarray([3.0]))
    assert jnp.abs(inverse.external_power_residual) <= 1.0e-7


def test_zero_force_velocity_euler_step_reports_and_respects_energy_bound():
    articulation = _single_axis_articulation("hinge")
    state = ReducedArticulationState(jnp.asarray([0.2]), jnp.asarray([0.3]))
    zero_gravity = jnp.zeros((3,))
    initial = reduced_energy(
        articulation, state.configuration, state.velocity, zero_gravity
    )
    result = reduced_semi_implicit_velocity_euler_step(
        articulation,
        state,
        jnp.asarray([0.0]),
        zero_gravity,
        jnp.asarray(1.0e-2),
    )
    accepted = reduced_energy(
        articulation,
        result.accepted_state.configuration,
        result.accepted_state.velocity,
        zero_gravity,
    )

    assert result.successful
    assert jnp.abs(result.diagnostics.energy_defect) <= (
        result.diagnostics.allowed_energy_defect
    )
    assert jnp.allclose(accepted.total, initial.total, rtol=2.0e-6)
    assert jnp.allclose(
        result.accepted_state.configuration, result.candidate_state.configuration
    )


def test_failed_step_rolls_back_candidate_state():
    articulation = _single_axis_articulation("prismatic")
    state = ReducedArticulationState(jnp.asarray([0.2]), jnp.asarray([0.3]))
    result = reduced_semi_implicit_velocity_euler_step(
        articulation,
        state,
        jnp.asarray([1.0]),
        jnp.zeros((3,)),
        jnp.asarray(0.2),
        policy=ReducedSemiImplicitVelocityEulerStepPolicy(maximum_step_size=0.1),
    )

    assert not result.successful
    assert result.status == int(ReducedDynamicsStatus.STEP_SIZE_REJECTED)
    assert jnp.allclose(result.accepted_state.configuration, state.configuration)
    assert jnp.allclose(result.accepted_state.velocity, state.velocity)
    assert not jnp.allclose(
        result.candidate_state.configuration, state.configuration
    )

    nonfinite = reduced_semi_implicit_velocity_euler_step(
        articulation,
        state,
        jnp.asarray([jnp.nan]),
        jnp.zeros((3,)),
        jnp.asarray(1.0e-2),
    )
    assert not nonfinite.successful
    assert nonfinite.status == int(ReducedDynamicsStatus.NONFINITE_INPUT)
    assert jnp.allclose(
        nonfinite.accepted_state.configuration, state.configuration
    )
    assert jnp.allclose(nonfinite.accepted_state.velocity, state.velocity)


def test_rebased_nonzero_com_energy_matches_maximal_com_evaluation():
    source_articulation = _single_axis_articulation("hinge")
    baseline = source_articulation.graph.bodies
    source = RigidBodySetPlan(
        baseline.material_ids,
        jnp.stack(
            (
                jnp.eye(3),
                jnp.diag(jnp.asarray([1.0, 1.2, 1.5])),
            )
        ),
        fixed_mask=baseline.fixed_mask,
    ).prepare(baseline.particles)
    source_reference = source.kinematics(
        jnp.asarray([[1.0, -0.5, 0.2], [2.0, -0.5, 0.2]]),
        jnp.zeros((2, 3)),
        jnp.asarray(
            [
                [0.9238795325, 0.0, 0.3826834324, 0.0],
                [0.9238795325, 0.0, 0.3826834324, 0.0],
            ]
        ),
        jnp.zeros((2, 3)),
    )
    offsets = jnp.asarray([[0.1, -0.2, 0.05], [-0.25, 0.15, 0.3]])
    parameterization = RigidInertialParameterization(source)
    realization = parameterization.realize(parameterization.inverse(offsets))
    target = realization.rigid_body_plan.prepare(realization.particle_plan.prepare())
    target_reference = realization.reference_frame_rebase.rebase_kinematics(
        source_reference, source, target
    )
    graph = source_articulation.graph.plan.prepare(target, target_reference)
    articulation = source_articulation.plan.prepare(graph, target_reference)
    configuration = jnp.asarray([0.37])
    velocity = jnp.asarray([-0.61])

    reduced = reduced_energy(
        articulation, configuration, velocity, jnp.zeros((3,))
    )
    kinematics = articulation.forward_kinematics(
        configuration, velocity
    ).bodies
    rotation = quaternion_rotation_matrix(kinematics.orientation)
    world_inertia = (
        rotation
        @ target.mass_properties.inertia_com
        @ jnp.swapaxes(rotation, -1, -2)
    )
    angular_momentum = (
        world_inertia @ kinematics.angular_velocity[..., None]
    )[..., 0]
    maximal_kinetic = 0.5 * jnp.sum(
        target.mass_properties.masses
        * jnp.sum(kinematics.velocity * kinematics.velocity, axis=-1)
    ) + 0.5 * jnp.sum(
        kinematics.angular_velocity * angular_momentum
    )

    assert reduced.successful
    assert jnp.allclose(reduced.kinetic, maximal_kinetic, rtol=2.0e-6)
