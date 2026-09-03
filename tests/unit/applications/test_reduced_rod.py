from __future__ import annotations

import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_dynamics import (
    prepare_rod,
    rod_potential_energy,
    RodPlan,
)
from phydrax.applications.solid_mechanics._rod_reduction import (
    evaluate_reduced_rod,
    lift_reduced_rod_state,
    prepare_reduced_rod,
    pullback_reduced_rod_loads,
    reduced_rod_power_evidence,
    ReducedRodPlan,
    ReducedRodState,
)


def _rod(*, inextensible: bool = False):
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
            jnp.asarray((1.0, 1.5, 1.0)),
            jnp.asarray((0.2, 0.3)),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((100.0, 30.0))), (2, 2, 2)
            ),
            jnp.asarray((((5.0,),),)),
            inextensible=inextensible,
        )
    )


def _reduction(*, rod=None):
    prepared_rod = _rod() if rod is None else rod
    stretch_basis = jnp.asarray(
        (
            ((1.0, 0.0), (0.0, 0.25)),
            ((0.5, 0.0), (0.0, -0.25)),
        )
    )
    bend_basis = jnp.asarray((((0.0, 1.0),),))
    return prepare_reduced_rod(
        prepared_rod,
        ReducedRodPlan(stretch_basis, bend_basis),
    )


def test_zero_coefficients_lift_exactly_to_native_rest_state():
    rod = _rod()
    reduction = _reduction(rod=rod)
    reduced_state = ReducedRodState(jnp.zeros((2,)), jnp.zeros((2,)))

    lifted = lift_reduced_rod_state(reduction, reduced_state)
    rest = rod.initialize_state()
    evaluation = evaluate_reduced_rod(reduction, reduced_state)

    assert reduced_state.values.shape == reduction.state_layout.shape == (4,)
    assert reduction.configuration_slice == slice(0, 2)
    assert reduction.velocity_slice == slice(2, 4)
    assert jnp.array_equal(lifted.positions, rest.positions)
    assert jnp.array_equal(lifted.orientations, rest.orientations)
    assert jnp.array_equal(lifted.velocities, rest.velocities)
    assert jnp.array_equal(lifted.angular_velocities, rest.angular_velocities)
    assert evaluation.valid
    assert evaluation.strain_evidence.valid
    assert evaluation.potential_energy == pytest.approx(0.0, abs=2.0e-6)


def test_velocity_jvp_matches_a_local_finite_displacement():
    reduction = _reduction()
    coefficients = jnp.asarray((0.08, 0.12))
    coefficient_velocities = jnp.asarray((-0.17, 0.21))
    state = ReducedRodState(coefficients, coefficient_velocities)
    lifted = lift_reduced_rod_state(reduction, state)
    epsilon = 2.0e-4
    displaced = lift_reduced_rod_state(
        reduction,
        ReducedRodState(
            coefficients + epsilon * coefficient_velocities,
            jnp.zeros_like(coefficient_velocities),
        ),
    )
    reference = lift_reduced_rod_state(
        reduction,
        ReducedRodState(coefficients, jnp.zeros_like(coefficient_velocities)),
    )

    finite_position_velocity = (displaced.positions - reference.positions) / epsilon
    finite_angular_velocity = (
        displaced.orientations - reference.orientations
    ) / epsilon

    assert jnp.allclose(
        lifted.velocities, finite_position_velocity, rtol=2.0e-3, atol=2.0e-4
    )
    assert jnp.allclose(
        lifted.angular_velocities,
        finite_angular_velocity,
        rtol=2.0e-3,
        atol=2.0e-4,
    )


def test_load_vjp_satisfies_native_reduced_power_duality():
    reduction = _reduction()
    coefficients = jnp.asarray((0.06, -0.14))
    coefficient_velocities = jnp.asarray((0.23, -0.19))
    native_forces = jnp.asarray(((0.4, -0.7), (0.3, 0.2), (-0.1, 0.8)))
    native_moments = jnp.asarray((0.35, -0.22))
    lifted = lift_reduced_rod_state(
        reduction, ReducedRodState(coefficients, coefficient_velocities)
    )
    generalized = pullback_reduced_rod_loads(
        reduction, coefficients, native_forces, native_moments
    )
    native_power = jnp.sum(native_forces * lifted.velocities) + jnp.sum(
        native_moments * lifted.angular_velocities
    )
    reduced_power = jnp.sum(generalized * coefficient_velocities)
    evidence = reduced_rod_power_evidence(
        reduction,
        coefficients,
        coefficient_velocities,
        native_forces,
        native_moments,
    )

    assert reduced_power == pytest.approx(native_power, rel=2.0e-6, abs=2.0e-6)
    assert evidence.valid
    assert evidence.absolute_residual == pytest.approx(0.0, abs=2.0e-6)


def test_reduced_potential_is_native_potential_of_lifted_state():
    reduction = _reduction()
    state = ReducedRodState(jnp.asarray((0.11, 0.18)), jnp.asarray((-0.2, 0.3)))
    evaluation = evaluate_reduced_rod(reduction, state)
    native_potential = rod_potential_energy(
        reduction.rod,
        evaluation.native_state.positions,
        evaluation.native_state.orientations,
    )

    assert evaluation.potential_energy == pytest.approx(
        native_potential, rel=2.0e-6, abs=2.0e-6
    )
    assert evaluation.quadrature_valid
    assert evaluation.strain_evidence.valid
    assert evaluation.native_evaluation.valid


def test_fixed_base_pose_rigidly_places_the_zero_strain_rod():
    rod = _rod()
    angle = jnp.asarray(0.4)
    position = jnp.asarray((2.0, -1.0))
    stretch_basis = jnp.asarray(
        (
            ((1.0, 0.0), (0.0, 0.25)),
            ((0.5, 0.0), (0.0, -0.25)),
        )
    )
    bend_basis = jnp.asarray((((0.0, 1.0),),))
    reduction = prepare_reduced_rod(
        rod,
        ReducedRodPlan(
            stretch_basis,
            bend_basis,
            fixed_base_position=position,
            fixed_base_orientation=angle,
        ),
    )
    lifted = reduction.lift(reduction.rest_state())
    evaluation = reduction.evaluate(reduction.rest_state())

    assert jnp.allclose(lifted.positions[0], position, atol=2.0e-6)
    assert lifted.orientations[0] == pytest.approx(angle, abs=2.0e-6)
    assert evaluation.lift_evidence.valid
    assert evaluation.potential_energy == pytest.approx(0.0, abs=2.0e-5)


def test_rank_deficient_incompatible_and_inextensible_reductions_reject():
    duplicate_stretch = jnp.asarray(
        (
            ((1.0, 1.0), (0.0, 0.0)),
            ((0.0, 0.0), (0.0, 0.0)),
        )
    )
    duplicate_bend = jnp.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="full column rank"):
        ReducedRodPlan(duplicate_stretch, duplicate_bend)

    incompatible_stretch = jnp.zeros((3, 2, 2)).at[0, 0, 0].set(1.0)
    incompatible_stretch = incompatible_stretch.at[1, 1, 1].set(1.0)
    incompatible_bend = jnp.zeros((2, 1, 2))
    incompatible_plan = ReducedRodPlan(
        incompatible_stretch, incompatible_bend
    )
    with pytest.raises(ValueError, match="incompatible"):
        prepare_reduced_rod(_rod(), incompatible_plan)

    compatible_plan = ReducedRodPlan(
        jnp.asarray(
            (
                ((1.0, 0.0), (0.0, 0.0)),
                ((0.0, 0.0), (0.0, 1.0)),
            )
        ),
        jnp.zeros((1, 1, 2)),
    )
    with pytest.raises(ValueError, match="Inextensible rods are unsupported"):
        prepare_reduced_rod(_rod(inextensible=True), compatible_plan)
