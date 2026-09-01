#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.applications.solid_mechanics._fem_dynamics import (
    FiniteElementDynamicsState,
    prepare_finite_element_dynamics,
    prepare_finite_element_dynamics_step,
    solve_finite_element_dynamics_step,
)
from phydrax.discretization.fem._rigid_coupling import (
    prepare_finite_element_point_interpolation,
    PreparedFiniteElementPointInterpolation,
    RigidDeformableAttachmentPlan,
)


def _tetrahedral_elasticity():
    points = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        points,
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("tetrahedron", 1),
            component_shape=(3,),
        ),
    ).prepare()
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.fem.linear_elasticity_form("u", 2.0, 1.0),
        discretization,
    )
    return discretization, compiled


def _rigid_translation_state(*, materials=None):
    displacement = jnp.zeros((4, 3))
    velocity = jnp.broadcast_to(jnp.asarray((0.4, -0.2, 0.1)), displacement.shape)
    return FiniteElementDynamicsState(
        displacement,
        velocity,
        jnp.zeros_like(displacement),
        materials=materials,
    )


def _elastic_potential(compiled):
    def energy(time, displacement, args):
        del time, args
        return 0.5 * jnp.sum(displacement * compiled.residual(displacement))

    return energy


def _zero_work(previous, candidate, args):
    del previous, candidate, args
    return jnp.asarray(0.0)


def test_manufactured_rigid_translation_newmark_step_has_zero_strain_and_energy_defect():
    _, compiled = _tetrahedral_elasticity()
    state = _rigid_translation_state()
    rigid_displacement = jnp.broadcast_to(
        jnp.asarray((0.3, -0.1, 0.2)), state.displacement.shape
    )
    assert jnp.max(jnp.abs(compiled.residual(rigid_displacement))) < 1.0e-12

    plan = prepare_finite_element_dynamics(
        compiled,
        state,
        0.1,
        potential_energy=_elastic_potential(compiled),
        potential_energy_id="linear-elastic-strain-energy",
        external_work=_zero_work,
        external_work_id="zero-external-work",
    )
    result = solve_finite_element_dynamics_step(
        prepare_finite_element_dynamics_step(plan, state, 0.1)
    )

    assert bool(result.accepted)
    assert jnp.allclose(
        result.accepted_state.displacement,
        0.1 * state.velocity,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    assert jnp.allclose(result.accepted_state.velocity, state.velocity, atol=1.0e-10)
    assert jnp.max(jnp.abs(result.accepted_state.acceleration)) < 1.0e-9
    assert bool(result.candidate.energy.available)
    assert bool(result.candidate.energy.finite)
    assert bool(result.candidate.energy.balanced)


def test_implicit_newmark_root_exposes_velocity_derivative():
    _, compiled = _tetrahedral_elasticity()
    template_state = FiniteElementDynamicsState(
        jnp.zeros((4, 3)),
        jnp.zeros((4, 3)),
        jnp.zeros((4, 3)),
    )
    plan = prepare_finite_element_dynamics(compiled, template_state, 0.05)
    translation = jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0)), (4, 3))

    def endpoint(scale):
        state = FiniteElementDynamicsState(
            template_state.displacement,
            scale * translation,
            template_state.acceleration,
        )
        result = solve_finite_element_dynamics_step(
            prepare_finite_element_dynamics_step(plan, state, 0.05)
        )
        return result.accepted_state.displacement[0, 0]

    assert jnp.isclose(jax.grad(endpoint)(jnp.asarray(0.7)), 0.05, rtol=2.0e-6)


def test_failed_inversion_hook_rolls_back_every_kinematic_field():
    _, compiled = _tetrahedral_elasticity()
    state = _rigid_translation_state()

    def inverted(time, displacement, args):
        del time, displacement, args
        return jnp.asarray((-1.0,))

    plan = prepare_finite_element_dynamics(
        compiled,
        state,
        0.1,
        determinant_evaluator=inverted,
        determinant_id="manufactured-inverted-cell",
        minimum_jacobian=0.0,
    )
    result = solve_finite_element_dynamics_step(
        prepare_finite_element_dynamics_step(plan, state, 0.1)
    )

    assert not bool(result.accepted)
    assert bool(result.rollback_applied)
    assert jnp.array_equal(result.accepted_state.displacement, state.displacement)
    assert jnp.array_equal(result.accepted_state.velocity, state.velocity)
    assert jnp.array_equal(result.accepted_state.acceleration, state.acceleration)
    assert result.accepted_state.time == state.time
    assert result.accepted_state.state_version == state.state_version
    assert not bool(result.candidate.admissibility.jacobian_valid)


def test_accepted_step_commits_material_history_atomically():
    _, compiled = _tetrahedral_elasticity()
    material = phx.equations.FiniteElementMaterialTransaction(
        (
            phx.equations.FiniteElementMaterialState(
                "history",
                jnp.zeros((1, 2)),
            ),
        )
    )
    state = _rigid_translation_state(materials=material)

    def update(displacement, velocity, acceleration, time, dt, previous, args):
        del displacement, velocity, acceleration, time, args
        return previous.with_trials({"history": previous.states[0].committed + dt})

    plan = prepare_finite_element_dynamics(
        compiled,
        state,
        0.1,
        material_update=update,
        material_update_id="constant-history-increment",
    )
    result = solve_finite_element_dynamics_step(
        prepare_finite_element_dynamics_step(plan, state, 0.1)
    )
    promoted = result.promote()

    assert bool(result.accepted)
    assert promoted.materials is not None
    assert jnp.allclose(
        promoted.materials.states[0].committed,
        jnp.full((1, 2), 0.1),
    )
    assert jnp.array_equal(
        promoted.materials.states[0].trial,
        promoted.materials.states[0].committed,
    )
    assert promoted.materials.states[0].state_version == (
        material.states[0].state_version + 1
    )
    assert promoted.state_version == state.state_version + 1


def test_prepared_interpolation_and_transpose_scatter_are_exact_duals():
    discretization, _ = _tetrahedral_elasticity()
    interpolation = prepare_finite_element_point_interpolation(
        discretization,
        "u",
        "tetrahedra",
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray(((0.2, 0.3, 0.1),)),
    )
    displacement = jnp.arange(12.0).reshape((4, 3)) / 7.0
    point_dual = jnp.asarray(((0.7, -0.4, 0.2),))
    evidence = interpolation.duality_evidence(displacement, point_dual)

    assert bool(evidence.valid)
    assert jnp.abs(evidence.residual) < 1.0e-12
    assert jnp.allclose(
        jnp.sum(interpolation.transpose_scatter(point_dual), axis=0),
        point_dual[0],
    )


def _one_rigid_body(position):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((7,), dtype=jnp.int64),
        jnp.ones((1,)),
        ambient_dimension=3,
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.eye(3)[None, :, :],
    ).prepare(particles)
    kinematics = bodies.kinematics(
        jnp.asarray(position)[None, :],
        jnp.zeros((1, 3)),
        jnp.asarray(((1.0, 0.0, 0.0, 0.0),)),
        jnp.zeros((1, 3)),
    )
    return bodies, kinematics


def test_attachment_kkt_loads_are_action_reaction_and_moment_balanced():
    discretization, _ = _tetrahedral_elasticity()
    interpolation = prepare_finite_element_point_interpolation(
        discretization,
        "u",
        "tetrahedra",
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray(((0.2, 0.3, 0.1),)),
    )
    bodies, kinematics = _one_rigid_body(interpolation.reference_positions[0])
    attachment = RigidDeformableAttachmentPlan(
        interpolation,
        bodies,
        jnp.asarray((7,), dtype=jnp.int64),
        jnp.zeros((1, 3)),
    )
    payload = attachment.kkt_payload(
        jnp.zeros((4, 3)),
        kinematics,
        jnp.asarray(((1.2, -0.4, 0.7),)),
    )
    increments = (
        jnp.arange(12.0).reshape((4, 3)) / 11.0,
        jnp.asarray(((0.2, -0.1, 0.3),)),
        jnp.asarray(((-0.4, 0.6, 0.1),)),
    )
    image = payload.operator.mv(increments)
    transpose = payload.operator.transpose_mv(payload.multiplier)
    kkt_duality = jnp.sum(image * payload.multiplier) - sum(
        jnp.sum(value * dual) for value, dual in zip(increments, transpose, strict=True)
    )

    assert jnp.max(jnp.abs(payload.constraint_residual)) < 1.0e-12
    assert bool(payload.certificate.valid)
    assert jnp.max(jnp.abs(payload.certificate.force_balance)) < 1.0e-12
    assert jnp.max(jnp.abs(payload.certificate.moment_balance)) < 1.0e-12
    assert jnp.abs(kkt_duality) < 1.0e-12


def test_duplicate_attachment_rows_fail_rank_preparation():
    discretization, _ = _tetrahedral_elasticity()
    single = prepare_finite_element_point_interpolation(
        discretization,
        "u",
        "tetrahedra",
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray(((0.2, 0.3, 0.1),)),
    )
    interpolation = PreparedFiniteElementPointInterpolation(
        discretization,
        "u",
        jnp.repeat(single.dof_routes, 2, axis=0),
        jnp.repeat(single.weights, 2, axis=0),
        jnp.repeat(single.reference_positions, 2, axis=0),
        single.dof_reference_positions,
    )
    bodies, _ = _one_rigid_body(single.reference_positions[0])

    with pytest.raises(ValueError, match="duplicate or rank deficient"):
        RigidDeformableAttachmentPlan(
            interpolation,
            bodies,
            jnp.asarray((7, 7), dtype=jnp.int64),
            jnp.zeros((2, 3)),
        )
