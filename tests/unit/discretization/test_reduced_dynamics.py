#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.particle._articulated_mass import (
    ArticulatedMassPolicy,
    ArticulatedMassStatus,
    PreparedArticulatedMass,
)
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
    FixedJointSetPlan,
    HingeJointSetPlan,
    PrismaticJointSetPlan,
    RigidJointGraphPlan,
)
from phydrax.discretization.particle._rigid_parameters import (
    RigidInertialParameterization,
)


def _single_axis_articulation(kind: str) -> Any:
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


def test_reduced_dynamics_scenario_1() -> None:
    articulation = _single_axis_articulation("hinge")
    q = jnp.asarray([0.0])
    v = jnp.asarray([0.0])
    gravity = jnp.asarray([0.0, -9.81, 0.0])

    mass = reduced_mass_matrix(articulation, q)
    bias = reduced_bias_terms(articulation, q, v, gravity)
    forward = reduced_forward_dynamics(articulation, q, v, jnp.asarray([0.0]), gravity)

    assert mass.successful
    assert mass.positive_definite
    assert jnp.allclose(mass.matrix, mass.matrix.T, atol=1.0e-7)
    assert jnp.allclose(mass.matrix, jnp.asarray([[5.0]]), rtol=2.0e-6)
    assert jnp.allclose(mass.operator.mv(jnp.asarray([2.0])), jnp.asarray([10.0]))
    assert bias.successful
    assert jnp.allclose(bias.velocity, 0.0, atol=1.0e-7)
    assert jnp.allclose(bias.gravity, jnp.asarray([19.62]), rtol=2.0e-6)
    assert forward.successful
    assert jnp.allclose(forward.acceleration, jnp.asarray([-19.62 / 5.0]), rtol=2.0e-5)
    articulation = _single_axis_articulation("prismatic")
    q = jnp.asarray([0.0])
    v = jnp.asarray([0.0])
    gravity = jnp.asarray([-9.81, 0.0, 0.0])

    mass = reduced_mass_matrix(articulation, q)
    bias = reduced_bias_terms(articulation, q, v, gravity)
    forward = reduced_forward_dynamics(articulation, q, v, jnp.asarray([0.0]), gravity)

    assert mass.successful
    assert jnp.allclose(mass.matrix, jnp.asarray([[2.0]]), rtol=2.0e-6)
    assert jnp.allclose(bias.gravity, jnp.asarray([19.62]), rtol=2.0e-6)
    assert forward.successful
    assert jnp.allclose(forward.acceleration, jnp.asarray([-9.81]), rtol=2.0e-5)
    articulation = _single_axis_articulation("hinge")
    q = jnp.asarray([0.31])
    v = jnp.asarray([0.47])
    acceleration = jnp.asarray([-0.73])
    gravity = jnp.asarray([0.0, -9.81, 0.0])

    inverse = reduced_inverse_dynamics(articulation, q, v, acceleration, gravity)
    forward = reduced_forward_dynamics(
        articulation, q, v, inverse.generalized_effort, gravity
    )

    assert inverse.successful
    assert forward.successful
    assert jnp.allclose(forward.acceleration, acceleration, rtol=2.0e-5, atol=2.0e-6)
    assert forward.relative_inverse_forward_residual <= 1.0e-6
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


def test_reduced_dynamics_scenario_2() -> None:
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
    assert not jnp.allclose(result.candidate_state.configuration, state.configuration)

    nonfinite = reduced_semi_implicit_velocity_euler_step(
        articulation,
        state,
        jnp.asarray([jnp.nan]),
        jnp.zeros((3,)),
        jnp.asarray(1.0e-2),
    )
    assert not nonfinite.successful
    assert nonfinite.status == int(ReducedDynamicsStatus.NONFINITE_INPUT)
    assert jnp.allclose(nonfinite.accepted_state.configuration, state.configuration)
    assert jnp.allclose(nonfinite.accepted_state.velocity, state.velocity)
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

    reduced = reduced_energy(articulation, configuration, velocity, jnp.zeros((3,)))
    kinematics = articulation.forward_kinematics(configuration, velocity).bodies
    rotation = quaternion_rotation_matrix(kinematics.orientation)
    world_inertia = (
        rotation @ target.mass_properties.inertia_com @ jnp.swapaxes(rotation, -1, -2)
    )
    angular_momentum = (world_inertia @ kinematics.angular_velocity[..., None])[..., 0]
    maximal_kinetic = 0.5 * jnp.sum(
        target.mass_properties.masses
        * jnp.sum(kinematics.velocity * kinematics.velocity, axis=-1)
    ) + 0.5 * jnp.sum(kinematics.angular_velocity * angular_momentum)

    assert reduced.successful
    assert jnp.allclose(reduced.kinetic, maximal_kinetic, rtol=2.0e-6)


def test_forward_dynamics_reuses_its_bound_articulated_mass() -> None:
    articulation = _single_axis_articulation("hinge")
    q = jnp.asarray([0.37])
    v = jnp.asarray([-0.61])
    effort = jnp.asarray([0.8])
    gravity = jnp.asarray([0.0, -9.81, 0.0])

    forward = reduced_forward_dynamics(articulation, q, v, effort, gravity)
    bias = reduced_bias_terms(articulation, q, v, gravity)
    mass = forward.mass
    response = mass.apply_inverse(articulation, q, effort - bias.total)

    assert forward.successful
    assert isinstance(mass, PreparedArticulatedMass)
    assert mass.is_current(articulation, q)
    assert response.successful
    assert jnp.allclose(response.value, forward.acceleration, rtol=1.0e-12)
    assert jnp.allclose(forward.minimum_articulated_inertia, 5.0, rtol=1.0e-12)
    stale = mass.apply_inverse(articulation, q + 0.1, effort)
    assert stale.status == int(ArticulatedMassStatus.STALE_BINDING)
    step = reduced_semi_implicit_velocity_euler_step(
        articulation,
        ReducedArticulationState(q, v),
        effort,
        gravity,
        jnp.asarray(1.0e-3),
    )
    assert step.dynamics.mass.is_current(articulation, q)


def test_forward_dynamics_compiles_with_closed_over_articulation() -> None:
    articulation = _single_axis_articulation("prismatic")
    gravity = jnp.asarray([-9.81, 0.0, 0.0])

    def acceleration(point: jax.Array, rate: jax.Array) -> jax.Array:
        return reduced_forward_dynamics(
            articulation, point, rate, jnp.zeros((1,)), gravity
        ).acceleration

    compiled = jax.jit(acceleration)(jnp.asarray([0.2]), jnp.asarray([0.1]))
    assert jnp.allclose(compiled, jnp.asarray([-9.81]), rtol=1.0e-12)


def test_forward_dynamics_keeps_legacy_pivot_acceptance_without_mass_policy() -> None:
    articulation = _single_axis_articulation("hinge")
    q = jnp.asarray([0.2])
    v = jnp.asarray([0.1])
    effort = jnp.asarray([0.5])
    zero_gravity = jnp.zeros((3,))
    strict = ArticulatedMassPolicy(relative_pivot_tolerance=1.5)

    legacy = reduced_forward_dynamics(articulation, q, v, effort, zero_gravity)
    explicit = reduced_forward_dynamics(
        articulation, q, v, effort, zero_gravity, mass_policy=strict
    )
    step = reduced_semi_implicit_velocity_euler_step(
        articulation,
        ReducedArticulationState(q, v),
        effort,
        zero_gravity,
        jnp.asarray(1.0e-3),
        mass_policy=strict,
    )

    assert legacy.successful
    assert legacy.mass.successful
    assert not explicit.successful
    assert explicit.status == int(ReducedDynamicsStatus.MASS_NOT_POSITIVE_DEFINITE)
    assert explicit.mass.status == int(ArticulatedMassStatus.PIVOT_NOT_POSITIVE)
    assert jnp.all(explicit.acceleration == 0.0)
    assert not step.successful
    assert jnp.allclose(step.accepted_state.configuration, q)


_BRANCHED_BODY_IDS = (100, 101, 102, 103, 104)
_BRANCHED_POSITIONS = (
    (0.0, 0.0, 0.0),
    (0.4, 0.0, 0.1),
    (0.8, 0.3, 0.1),
    (0.4, -0.5, 0.2),
    (0.6, -0.9, 0.0),
)
_BRANCHED_ORIENTATIONS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.95, 0.1, 0.2, 0.05),
    (1.0, 0.0, 0.0, 0.0),
    (0.9, 0.0, 0.3, 0.3),
    (0.97, -0.1, 0.0, 0.2),
)


def _branched_articulation() -> Any:
    """Root, hinge 31 with a prismatic 32 branch and a hinge 33 → fixed 34 branch."""

    body_ids = jnp.asarray(_BRANCHED_BODY_IDS, dtype=jnp.int64)
    particles = ParticleSetPlan(
        body_ids, jnp.asarray([1.0, 1.3, 0.7, 0.9, 0.5]), ambient_dimension=3
    ).prepare()
    inertia = np.stack(
        [
            np.asarray([[a, d, 0.0], [d, b, 0.5 * d], [0.0, 0.5 * d, c]])
            for a, b, c, d in (
                (0.2, 0.25, 0.3, 0.02),
                (0.1, 0.12, 0.09, 0.01),
                (0.05, 0.04, 0.06, 0.005),
                (0.08, 0.1, 0.07, -0.01),
                (0.03, 0.02, 0.025, 0.003),
            )
        ]
    )
    bodies = RigidBodySetPlan(
        jnp.zeros((5,), dtype=jnp.int32),
        jnp.asarray(inertia),
        fixed_mask=jnp.asarray([True, False, False, False, False]),
    ).prepare(particles)
    quaternions = np.asarray(_BRANCHED_ORIENTATIONS)
    reference = bodies.kinematics(
        jnp.asarray(_BRANCHED_POSITIONS),
        jnp.zeros((5, 3)),
        jnp.asarray(quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)),
        jnp.zeros((5, 3)),
    )
    hinge_axes = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.5, 0.0]])
    prismatic_axis = np.asarray([[0.3, 1.0, 0.2]])
    graph = RigidJointGraphPlan(
        fixed=FixedJointSetPlan(
            jnp.asarray([34]), jnp.asarray([103]), jnp.asarray([104])
        ),
        hinge=HingeJointSetPlan(
            jnp.asarray([31, 33]),
            jnp.asarray([100, 101]),
            jnp.asarray([101, 103]),
            jnp.asarray([[0.1, 0.0, 0.05], [0.4, -0.2, 0.15]]),
            jnp.asarray(hinge_axes / np.linalg.norm(hinge_axes, axis=-1, keepdims=True)),
        ),
        prismatic=PrismaticJointSetPlan(
            jnp.asarray([32]),
            jnp.asarray([101]),
            jnp.asarray([102]),
            jnp.asarray([[0.8, 0.3, 0.1]]),
            jnp.asarray(prismatic_axis / np.linalg.norm(prismatic_axis)),
        ),
    ).prepare(bodies, reference)
    return ReducedArticulationPlan(
        100,
        jnp.asarray([31, 32, 33, 34]),
        jnp.asarray([100, 101, 101, 103]),
        jnp.asarray([101, 102, 103, 104]),
    ).prepare(graph, reference)


def _fixed_only_articulation() -> Any:
    body_ids = jnp.asarray([5, 6], dtype=jnp.int64)
    particles = ParticleSetPlan(
        body_ids, jnp.asarray([1.0, 2.0]), ambient_dimension=3
    ).prepare()
    bodies = RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        fixed_mask=jnp.asarray([True, False]),
    ).prepare(particles)
    reference = bodies.kinematics(
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        jnp.zeros((2, 3)),
    )
    graph = RigidJointGraphPlan(
        fixed=FixedJointSetPlan(jnp.asarray([8]), body_ids[:1], body_ids[1:])
    ).prepare(bodies, reference)
    return ReducedArticulationPlan(
        5, jnp.asarray([8]), jnp.asarray([5]), jnp.asarray([6])
    ).prepare(graph, reference)


_BRANCHED_Q = jnp.asarray([0.41, -0.23, 0.77])
_BRANCHED_V = jnp.asarray([-0.6, 0.35, 1.1])
_BRANCHED_A = jnp.asarray([0.9, -1.4, 0.25])
_BRANCHED_GRAVITY = jnp.asarray([0.3, -9.81, 1.2])


def _branched_load() -> RigidBodyLoad:
    return RigidBodyLoad(
        jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.4, -1.1, 0.3],
                [0.0, 0.0, 0.0],
                [-0.7, 0.2, 0.9],
            ]
        ),
        jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, -0.2],
                [0.0, 0.0, 0.0],
                [0.3, 0.5, -0.4],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def _lagrangian_references(
    articulation: Any, q: jax.Array, v: jax.Array, gravity: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Independent ``M``, ``c`` and ``g`` from the energy Lagrangian, not RNEA."""

    zero_gravity = jnp.zeros((3,))

    def kinetic(point: jax.Array, rate: jax.Array) -> jax.Array:
        return reduced_energy(articulation, point, rate, zero_gravity).kinetic

    def potential(point: jax.Array) -> jax.Array:
        return reduced_energy(articulation, point, jnp.zeros_like(v), gravity).potential

    momentum = jax.grad(kinetic, argnums=1)
    mass = jax.jacfwd(momentum, argnums=1)(q, v)
    bias = jax.jacfwd(momentum, argnums=0)(q, v) @ v - jax.grad(kinetic)(q, v)
    return mass, bias, jax.grad(potential)(q)


def _external_reference(
    articulation: Any, q: jax.Array, load: RigidBodyLoad
) -> jax.Array:
    jacobian = jax.jacfwd(lambda rate: articulation.body_velocity_action(q, rate))(
        jnp.zeros((articulation.nv,))
    )
    wrench = jnp.concatenate((load.force, load.torque), axis=-1)
    return jnp.einsum("bkn,bk->n", jacobian, wrench)


def test_branched_inverse_dynamics_columns_match_lagrangian_references() -> None:
    articulation = _branched_articulation()
    q, v, a = _BRANCHED_Q, _BRANCHED_V, _BRANCHED_A
    gravity = _BRANCHED_GRAVITY
    load = _branched_load()
    mass, bias, generalized_gravity = _lagrangian_references(articulation, q, v, gravity)
    external = _external_reference(articulation, q, load)

    inverse = reduced_inverse_dynamics(articulation, q, v, a, gravity, external_load=load)
    terms = reduced_bias_terms(articulation, q, v, gravity)

    assert articulation.nv == 3
    assert inverse.successful
    assert inverse.status == int(ReducedDynamicsStatus.SUCCESS)
    assert inverse.external_duality.valid
    assert jnp.abs(inverse.external_power_residual) <= 1.0e-12
    assert inverse.decomposition_residual <= 1.0e-12
    assert jnp.allclose(inverse.mass_effort, mass @ a, rtol=1.0e-10, atol=1.0e-10)
    assert jnp.allclose(inverse.bias_effort, bias, rtol=1.0e-10, atol=1.0e-10)
    assert jnp.allclose(
        inverse.gravity_effort, generalized_gravity, rtol=1.0e-10, atol=1.0e-10
    )
    assert jnp.allclose(inverse.external_effort, external, rtol=1.0e-10, atol=1.0e-10)
    assert jnp.allclose(
        inverse.generalized_effort,
        mass @ a + bias + generalized_gravity - external,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert terms.successful
    assert jnp.allclose(terms.velocity, bias, rtol=1.0e-10, atol=1.0e-10)
    assert jnp.allclose(terms.gravity, generalized_gravity, rtol=1.0e-10, atol=1.0e-10)
    assert jnp.allclose(terms.velocity, inverse.bias_effort, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.allclose(terms.gravity, inverse.gravity_effort, rtol=1.0e-12, atol=1.0e-12)

    effort = jnp.asarray([0.8, -0.3, 0.45])
    forward = reduced_forward_dynamics(
        articulation, q, v, effort, gravity, external_load=load
    )
    expected = jnp.linalg.solve(mass, effort + external - bias - generalized_gravity)
    assert forward.successful
    assert jnp.allclose(forward.acceleration, expected, rtol=1.0e-10, atol=1.0e-10)
    assert forward.relative_inverse_forward_residual <= 1.0e-12
    assert jnp.allclose(forward.reconstructed_effort, effort, rtol=1.0e-10, atol=1.0e-10)


def test_branched_inverse_dynamics_derivatives_compile_and_batch() -> None:
    articulation = _branched_articulation()
    q, v, a = _BRANCHED_Q, _BRANCHED_V, _BRANCHED_A
    gravity = _BRANCHED_GRAVITY
    load = _branched_load()
    mass, _, _ = _lagrangian_references(articulation, q, v, gravity)

    def candidate(point: jax.Array, rate: jax.Array, change: jax.Array) -> jax.Array:
        return reduced_inverse_dynamics(
            articulation, point, rate, change, gravity, external_load=load
        ).candidate_effort

    def gravity_effort(point: jax.Array) -> jax.Array:
        return reduced_inverse_dynamics(articulation, point, v, a, gravity).gravity_effort

    def potential(point: jax.Array) -> jax.Array:
        return reduced_energy(articulation, point, jnp.zeros_like(v), gravity).potential

    stiffness = jax.hessian(potential)(q)
    assert jnp.allclose(
        jax.jacrev(candidate, argnums=2)(q, v, a), mass, rtol=1.0e-10, atol=1.0e-10
    )
    assert jnp.allclose(
        jax.jacfwd(candidate, argnums=2)(q, v, a), mass, rtol=1.0e-10, atol=1.0e-10
    )
    assert jnp.allclose(
        jax.jacfwd(gravity_effort)(q), stiffness, rtol=1.0e-9, atol=1.0e-9
    )
    assert jnp.allclose(
        jax.jacrev(gravity_effort)(q), stiffness, rtol=1.0e-9, atol=1.0e-9
    )
    forward_q = jax.jacfwd(candidate, argnums=0)(q, v, a)
    reverse_q = jax.jacrev(candidate, argnums=0)(q, v, a)
    forward_v = jax.jacfwd(candidate, argnums=1)(q, v, a)
    reverse_v = jax.jacrev(candidate, argnums=1)(q, v, a)
    assert jnp.all(jnp.isfinite(forward_q)) and jnp.all(jnp.isfinite(forward_v))
    assert jnp.allclose(forward_q, reverse_q, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.allclose(forward_v, reverse_v, rtol=1.0e-12, atol=1.0e-12)

    compiled = jax.jit(candidate)(q, v, a)
    assert jnp.allclose(compiled, candidate(q, v, a), rtol=1.0e-12, atol=1.0e-12)
    points = jnp.stack((q, q + 0.2, q - 0.35))
    batched = jax.vmap(candidate, in_axes=(0, None, None))(points, v, a)
    looped = jnp.stack([candidate(point, v, a) for point in points])
    assert jnp.allclose(batched, looped, rtol=1.0e-12, atol=1.0e-12)


def test_inverse_dynamics_columns_stay_independent_under_nonfinite_velocity() -> None:
    articulation = _branched_articulation()
    q, v, a = _BRANCHED_Q, _BRANCHED_V, _BRANCHED_A
    gravity = _BRANCHED_GRAVITY
    finite = reduced_inverse_dynamics(articulation, q, v, a, gravity)
    dof_joint_ids = np.asarray(articulation.joint_ids)[
        np.asarray(articulation.dof_joint_indices)
    ]
    root_dof = int(np.flatnonzero(dof_joint_ids == 31)[0])
    nonfinite_v = v.at[root_dof].set(jnp.nan)

    inverse = reduced_inverse_dynamics(articulation, q, nonfinite_v, a, gravity)

    assert not inverse.successful
    assert inverse.status == int(ReducedDynamicsStatus.NONFINITE_INPUT)
    assert jnp.all(inverse.generalized_effort == 0.0)
    assert jnp.array_equal(inverse.mass_effort, finite.mass_effort)
    assert jnp.array_equal(inverse.gravity_effort, finite.gravity_effort)
    assert jnp.all(jnp.isnan(inverse.bias_effort))
    assert jnp.all(jnp.isnan(inverse.candidate_effort))


def test_nonfinite_velocity_keeps_configuration_bound_mass_usable() -> None:
    articulation = _single_axis_articulation("hinge")
    q = jnp.asarray([0.3], dtype=jnp.float64)
    result = reduced_forward_dynamics(
        articulation,
        q,
        jnp.asarray([jnp.nan], dtype=jnp.float64),
        jnp.zeros((1,), dtype=jnp.float64),
        jnp.zeros((3,), dtype=jnp.float64),
    )

    assert not result.successful
    assert result.status == int(ReducedDynamicsStatus.NONFINITE_INPUT)
    assert result.mass.successful
    response = result.mass.apply_inverse(
        articulation, q, jnp.asarray([2.5], dtype=jnp.float64)
    )
    assert response.successful
    # The axis inertia is I_zz + mass * radius**2 = 3 + 2.
    np.testing.assert_allclose(response.value, [0.5], rtol=1.0e-12, atol=1.0e-12)


def test_zero_rate_mass_preparation_rejects_nonfinite_reference_pose() -> None:
    articulation = _single_axis_articulation("hinge")
    invalid = eqx.tree_at(
        lambda value: value.reference_orientation,
        articulation,
        articulation.reference_orientation.at[articulation.root_index, 0].set(jnp.nan),
    )
    zero = jnp.zeros((1,), dtype=jnp.float64)
    result = reduced_forward_dynamics(
        invalid, zero, zero, zero, jnp.zeros((3,), dtype=jnp.float64)
    )

    assert not result.successful
    assert not result.mass.successful
    assert not result.mass.evidence.factor_finite


def test_nonfinite_inertia_propagates_through_zero_motion_columns() -> None:
    articulation = _branched_articulation()
    leaf = articulation.graph.bodies.mass_properties.inertia_com
    infinite = eqx.tree_at(
        lambda value: value.graph.bodies.mass_properties.inertia_com,
        articulation,
        leaf.at[4].set(jnp.full((3, 3), jnp.inf)),
    )
    zero = jnp.zeros((3,))
    dof_joint_ids = np.asarray(infinite.joint_ids)[np.asarray(infinite.dof_joint_indices)]
    ancestor = np.isin(dof_joint_ids, (31, 33))

    inverse = reduced_inverse_dynamics(infinite, _BRANCHED_Q, zero, zero, zero)
    terms = reduced_bias_terms(infinite, _BRANCHED_Q, zero, zero)

    assert not inverse.successful
    assert inverse.status == int(ReducedDynamicsStatus.NONFINITE_OUTPUT)
    assert jnp.all(inverse.generalized_effort == 0.0)
    for component in (
        inverse.mass_effort,
        inverse.bias_effort,
        inverse.gravity_effort,
        inverse.candidate_effort,
    ):
        assert jnp.all(jnp.isnan(component[ancestor]))
        assert jnp.all(jnp.isfinite(component[~ancestor]))
    assert not terms.successful
    assert terms.status == int(ReducedDynamicsStatus.NONFINITE_OUTPUT)


def test_zero_dof_inverse_bias_and_forward_dynamics_are_exact() -> None:
    articulation = _fixed_only_articulation()
    empty = jnp.zeros((0,))
    gravity = jnp.asarray([0.0, -9.81, 0.0])

    inverse = reduced_inverse_dynamics(articulation, empty, empty, empty, gravity)
    terms = reduced_bias_terms(articulation, empty, empty, gravity)
    forward = reduced_forward_dynamics(articulation, empty, empty, empty, gravity)

    assert articulation.nv == 0
    assert inverse.successful and terms.successful and forward.successful
    assert inverse.candidate_effort.shape == (0,)
    assert inverse.mass_effort.shape == (0,)
    assert terms.total.shape == (0,)
    assert forward.acceleration.shape == (0,)
    assert inverse.decomposition_residual == 0.0
