#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native coupled articulated contact/limit step against analytic mechanics."""

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.contact._articulated_step import (
    ArticulatedContactStepStatus,
    PreparedArticulatedContactStep,
)
from phydrax.applications.contact._cone import ContactConeBranch
from phydrax.applications.contact._participant_adapters import (
    ArticulatedPrimitiveContactParticipant,
)
from phydrax.discretization.particle._reduced_articulation import (
    ReducedArticulationState,
)
from phydrax.discretization.particle._reduced_dynamics import (
    reduced_semi_implicit_velocity_euler_step,
)
from phydrax.discretization.particle._rigid_joint_coordinates import (
    LiftedJointCoordinateState,
)
from phydrax.linalg import MaterializationPolicy
from phydrax.linalg.svd import DenseSVD, SVDSolvePolicy
from tests._support.native_articulated import (
    capsule_bar_model,
    contact_step,
    dof,
    pendulum_model,
    planar_slider_model,
    slider_model,
    standard_gravity,
    step_policy,
    WORK_TOLERANCE,
    zero_gravity,
)


def _run(
    step: PreparedArticulatedContactStep,
    source: Any,
    duration: float,
    gravity: jax.Array,
) -> Any:
    return step.step(source, duration, jnp.zeros((0,)), gravity)


def _source(
    step: PreparedArticulatedContactStep, configuration: Any, velocity: Any
) -> Any:
    lifted = jnp.asarray(configuration, dtype=step.dtype)
    principal, winding, valid = step.coordinates.split(lifted)
    assert bool(valid)
    return step.initial_source(principal, winding.winding, jnp.asarray(velocity))


def _trees_equal(left: Any, right: Any) -> bool:
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_contact_free_step_is_exact_native_euler() -> None:
    prepared, step = contact_step(slider_model())
    source = _source(step, [0.5], [0.0])
    result = _run(step, source, 0.01, standard_gravity())
    native = reduced_semi_implicit_velocity_euler_step(
        prepared.articulation,
        ReducedArticulationState(jnp.asarray([0.5]), jnp.asarray([0.0])),
        jnp.zeros((1,)),
        standard_gravity(),
        0.01,
        policy=step.policy.smooth,
    )

    assert bool(result.successful) and bool(result.free_bypass)
    assert int(result.attempted_segments) == 1
    assert jnp.array_equal(
        result.accepted.configuration, native.candidate_state.configuration
    )
    assert jnp.array_equal(result.accepted.velocity, native.candidate_state.velocity)
    assert float(result.ledger.correction_loss) == 0.0
    assert float(result.ledger.contact_work) == 0.0


def test_oblique_separated_features_with_overlapping_boxes_stay_exact() -> None:
    # Tip ends at angle 1.0, obstacle at 1.21: physical gap 2 sin(0.105) - 0.2 > 0
    # while both world boxes overlap in x and y.
    prepared, step = contact_step(pendulum_model(obstacle_angle=1.21))
    source = _source(step, [0.95], [1.0])
    result = _run(step, source, 0.05, zero_gravity())
    native = reduced_semi_implicit_velocity_euler_step(
        prepared.articulation,
        ReducedArticulationState(jnp.asarray([0.95]), jnp.asarray([1.0])),
        jnp.zeros((1,)),
        zero_gravity(),
        0.05,
        policy=step.policy.smooth,
    )
    end_gap = 2.0 * math.sin(0.5 * (1.21 - 1.0)) - 0.2

    assert end_gap > 0.0
    assert bool(result.successful) and bool(result.free_bypass)
    assert jnp.array_equal(result.accepted.velocity, native.candidate_state.velocity)
    assert jnp.array_equal(
        result.accepted.configuration, native.candidate_state.configuration
    )


def test_initially_touching_approaching_slider_is_perfectly_inelastic() -> None:
    mass, speed = 2.0, 1.0
    _, step = contact_step(slider_model(mass=mass))
    result = _run(step, _source(step, [0.0], [-speed]), 0.01, zero_gravity())
    segment = result.last_segment
    ledger = result.ledger

    assert bool(result.successful)
    assert not bool(result.free_bypass)
    assert abs(float(result.accepted.velocity[0])) <= 1.0e-9
    assert abs(float(result.accepted.configuration[0])) <= 1.0e-9
    # p = m V, W_contact = p . G(q1) v1 = 0, D_contact = m V^2 / 2.
    normal = float(jnp.sum(jnp.where(segment.block_valid, segment.impulse[:, 0], 0.0)))
    assert normal == pytest.approx(mass * speed, rel=1.0e-9)
    assert abs(float(ledger.contact_work)) <= 1.0e-8
    assert float(ledger.correction_loss) == pytest.approx(
        0.5 * mass * speed**2, rel=1.0e-9
    )
    assert float(ledger.final_energy - ledger.initial_energy) == pytest.approx(
        -0.5 * mass * speed**2, rel=1.0e-9
    )
    assert abs(float(ledger.identity_residual)) <= 1.0e-10
    # The law normal velocity g(q1)/h is distinct from the measured velocity.
    valid = np.asarray(segment.block_valid)
    assert np.all(np.abs(np.asarray(segment.law_velocity)[valid, 0]) <= 1.0e-6)
    # A newly activated contact is not even a fixed-branch derivative candidate.
    assert not bool(result.fixed_branch_candidate)


def test_resting_contact_supports_gravity_with_explicit_correction_loss() -> None:
    mass, step_size, gravity = 2.0, 0.01, 9.81
    _, step = contact_step(slider_model(mass=mass))
    result = _run(step, _source(step, [0.0], [0.0]), step_size, standard_gravity())
    segment = result.last_segment

    assert bool(result.successful)
    assert abs(float(result.accepted.velocity[0])) <= 1.0e-9
    normal = float(jnp.sum(jnp.where(segment.block_valid, segment.impulse[:, 0], 0.0)))
    assert normal == pytest.approx(mass * gravity * step_size, rel=1.0e-9)
    assert float(result.ledger.correction_loss) == pytest.approx(
        0.5 * mass * (gravity * step_size) ** 2, rel=1.0e-8
    )
    assert float(result.ledger.contact_work) <= 1.0e-8
    assert int(segment.active_contact_count) == 1


def test_sliding_friction_matches_dynamic_coulomb_law() -> None:
    carriage, slider, mu_dynamic = 3.0, 1.0, 0.5
    step_size, gravity, speed = 0.01, 9.81, 1.0
    model = planar_slider_model(
        carriage_mass=carriage, slider_mass=slider, friction=(0.8, mu_dynamic)
    )
    prepared, step = contact_step(model)
    drive, lift = dof(prepared, "drive"), dof(prepared, "lift")
    velocity = jnp.zeros((2,)).at[drive].set(speed)
    result = _run(
        step, _source(step, jnp.zeros((2,)), velocity), step_size, standard_gravity()
    )
    normal = slider * gravity * step_size
    expected = speed - mu_dynamic * normal / (carriage + slider)
    segment = result.last_segment

    assert bool(result.successful)
    assert float(result.accepted.velocity[drive]) == pytest.approx(expected, rel=1.0e-8)
    assert abs(float(result.accepted.velocity[lift])) <= 1.0e-9
    active = np.asarray(segment.block_valid) & (np.asarray(segment.impulse)[:, 0] > 0.0)
    assert np.asarray(segment.branch)[active].tolist() == [int(ContactConeBranch.SLIDING)]
    tangent = np.linalg.norm(np.asarray(segment.impulse)[active, 1:])
    assert tangent == pytest.approx(mu_dynamic * normal, rel=1.0e-8)
    assert float(result.ledger.friction_work) < 0.0


def test_small_tangential_velocity_sticks_within_static_cone() -> None:
    carriage, slider, mu_static = 3.0, 1.0, 0.8
    model = planar_slider_model(
        carriage_mass=carriage, slider_mass=slider, friction=(mu_static, 0.5)
    )
    prepared, step = contact_step(model)
    drive = dof(prepared, "drive")
    velocity = jnp.zeros((2,)).at[drive].set(0.01)
    result = _run(
        step, _source(step, jnp.zeros((2,)), velocity), 0.01, standard_gravity()
    )
    segment = result.last_segment
    active = np.asarray(segment.block_valid) & (np.asarray(segment.impulse)[:, 0] > 0.0)

    assert bool(result.successful)
    assert abs(float(result.accepted.velocity[drive])) <= 1.0e-9
    assert np.asarray(segment.branch)[active].tolist() == [
        int(ContactConeBranch.STICKING)
    ]
    tangent = np.linalg.norm(np.asarray(segment.impulse)[active, 1:])
    assert tangent == pytest.approx((carriage + slider) * 0.01, rel=1.0e-8)
    assert tangent <= mu_static * float(np.asarray(segment.impulse)[active, 0][0])


def test_joint_stop_reaction_uses_generalized_impulse_units() -> None:
    mass, speed = 2.0, 1.0
    _, step = contact_step(slider_model(mass=mass, stop_bounds=(-1.0, 0.5)))
    result = _run(step, _source(step, [0.5], [speed]), 0.01, zero_gravity())

    assert bool(result.successful)
    assert abs(float(result.accepted.velocity[0])) <= 1.0e-9
    # Rows are (lower, upper); the upper stop reacts with p = m V in N s.
    assert float(result.accepted.stop_impulse[1]) == pytest.approx(
        mass * speed, rel=1.0e-9
    )
    assert abs(float(result.accepted.stop_impulse[0])) <= 1.0e-9
    assert int(result.accepted.stop_branch[1]) == int(ContactConeBranch.NORMAL_ACTIVE)


def test_simultaneous_stop_and_contact_share_one_coupled_response() -> None:
    carriage, slider, speed = 3.0, 1.0, 0.5
    model = planar_slider_model(
        carriage_mass=carriage, slider_mass=slider, drive_bounds=(-1.0, 0.0)
    )
    prepared, step = contact_step(model)
    drive, lift = dof(prepared, "drive"), dof(prepared, "lift")
    velocity = jnp.zeros((2,)).at[drive].set(speed)
    result = _run(
        step, _source(step, jnp.zeros((2,)), velocity), 0.01, standard_gravity()
    )
    segment = result.last_segment
    contacts = step.contact_capacity
    contact_valid = np.asarray(segment.block_valid)[:contacts]
    friction_x = -(
        np.sum(np.asarray(segment.impulse)[:contacts][contact_valid, 1:] ** 2) ** 0.5
    )
    stop = float(result.accepted.stop_impulse[1])

    assert bool(result.successful)
    assert abs(float(result.accepted.velocity[drive])) <= 1.0e-9
    assert abs(float(result.accepted.velocity[lift])) <= 1.0e-9
    # Stop and friction impulses together remove the carriage momentum.
    assert stop - friction_x == pytest.approx((carriage + slider) * speed, rel=1.0e-8)
    assert stop > 0.0


def test_capsule_on_plane_emits_two_endpoint_supports() -> None:
    mass, step_size, gravity = 2.0, 0.01, 9.81
    _, step = contact_step(capsule_bar_model(mass=mass))
    result = _run(step, _source(step, [0.0], [0.0]), step_size, standard_gravity())
    segment = result.last_segment
    active = np.asarray(segment.block_valid) & (np.asarray(segment.impulse)[:, 0] > 0.0)
    normals = np.asarray(segment.impulse)[active, 0]

    assert bool(result.successful)
    assert normals.size == 2
    assert float(np.sum(normals)) == pytest.approx(mass * gravity * step_size, rel=1.0e-9)
    assert normals[0] == pytest.approx(normals[1], rel=1.0e-6)


def test_full_turn_without_obstacle_advances_winding_exactly() -> None:
    step_size = 0.1
    rate = (2.0 * math.pi + 0.1) / step_size
    _, step = contact_step(pendulum_model(obstacle_angle=None))
    result = _run(step, _source(step, [0.0], [rate]), step_size, zero_gravity())

    assert bool(result.successful) and bool(result.free_bypass)
    assert int(result.accepted.winding[0]) == 1
    lifted = step.coordinates.lift(
        result.accepted.configuration,
        LiftedJointCoordinateState(result.accepted.winding),
    )
    assert float(lifted[0]) == pytest.approx(2.0 * math.pi + 0.1, rel=1.0e-12)


def test_full_turn_through_obstacle_with_clear_endpoints_is_caught() -> None:
    step_size = 0.1
    rate = (2.0 * math.pi + 0.1) / step_size
    _, step = contact_step(pendulum_model(obstacle_angle=0.5 * math.pi))
    result = _run(step, _source(step, [0.0], [rate]), step_size, zero_gravity())

    assert not bool(result.free_bypass)
    assert bool(result.successful)
    assert int(result.accepted.winding[0]) == 0
    touching = 0.5 * math.pi - 2.0 * math.asin(0.1)
    assert float(result.accepted.configuration[0]) <= touching + 1.0e-6
    assert float(result.covered_fraction) == 1.0
    assert int(result.accepted_segments) >= 2


def test_initial_penetration_refuses_and_rolls_back_every_leaf() -> None:
    _, step = contact_step(slider_model())
    source = _source(step, [-0.05], [0.0])
    result = _run(step, source, 0.01, standard_gravity())

    assert not bool(result.successful)
    assert int(result.status) == int(ArticulatedContactStepStatus.INITIALLY_INADMISSIBLE)
    assert _trees_equal(result.accepted, source)


@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        (
            {"maximum_segment_attempts": 1},
            ArticulatedContactStepStatus.SUBSTEP_BUDGET_EXHAUSTED,
        ),
        (
            {"maximum_subdivision_depth": 0},
            ArticulatedContactStepStatus.MINIMUM_SEGMENT_REACHED,
        ),
    ],
)
def test_subdivision_budgets_are_distinct_failures_without_partial_time(
    overrides: dict[str, int], status: ArticulatedContactStepStatus
) -> None:
    step_size = 0.1
    rate = (2.0 * math.pi + 0.1) / step_size
    _, step = contact_step(
        pendulum_model(obstacle_angle=0.5 * math.pi), step_policy(**overrides)
    )
    source = _source(step, [0.0], [rate])
    result = _run(step, source, step_size, zero_gravity())

    assert not bool(result.successful)
    assert int(result.status) == int(status)
    assert _trees_equal(result.accepted, source)


def test_casewise_batches_isolate_failures_under_vmap() -> None:
    _, step = contact_step(slider_model())
    sources = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        _source(step, [0.5], [0.0]),
        _source(step, [-0.05], [0.0]),
    )

    def advance(source: Any) -> Any:
        return step.step(source, 0.01, jnp.zeros((0,)), standard_gravity())

    result = eqx.filter_jit(jax.vmap(advance))(sources)

    assert np.asarray(result.successful).tolist() == [True, False]
    assert int(result.status[1]) == int(
        ArticulatedContactStepStatus.INITIALLY_INADMISSIBLE
    )
    assert jnp.array_equal(result.accepted.configuration[1], sources.configuration[1])
    repeated = eqx.filter_jit(jax.vmap(advance))(sources)
    assert _trees_equal(repeated.accepted, result.accepted)


def test_free_fall_jvp_matches_analytic_gravity_sensitivity() -> None:
    step_size = 0.01
    _, step = contact_step(slider_model())
    result = _run(step, _source(step, [0.5], [0.0]), step_size, standard_gravity())
    direction = jnp.asarray([0.0, 0.0, 1.0])
    jvp = step.step_jvp(
        result,
        jnp.zeros((0,)),
        standard_gravity(),
        (None, jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((0,)), direction),
    )

    assert bool(result.fixed_branch_candidate) and bool(jvp.successful)
    assert bool(jvp.binding_current) and bool(jvp.coupled_regular)
    assert float(jvp.velocity_tangent[0]) == pytest.approx(step_size, rel=1.0e-12)
    assert float(jvp.configuration_tangent[0]) == pytest.approx(step_size**2, rel=1.0e-12)


def _gravity_jvp(
    step: PreparedArticulatedContactStep, result: Any, gravity: jax.Array
) -> Any:
    return step.step_jvp(
        result,
        jnp.zeros((0,)),
        gravity,
        (None, jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((0,)), _UP),
    )


def _impulse_vjp(
    step: PreparedArticulatedContactStep, result: Any, gravity: jax.Array
) -> Any:
    valid = jnp.asarray(result.last_segment.block_valid, dtype=step.dtype)
    impulse_bar = jnp.zeros_like(result.last_segment.impulse).at[:, 0].set(valid)
    return step.step_vjp(
        result,
        jnp.zeros((0,)),
        gravity,
        (jnp.zeros((1,)), jnp.zeros((1,)), impulse_bar),
    ), impulse_bar


_UP = jnp.asarray([0.0, 0.0, 1.0])


def test_resting_stop_jvp_and_vjp_are_dual_regular_and_match_differences() -> None:
    # A slider without collision geometry resting on its lower joint stop: the
    # normal-only stop row is the only active block, so the full coupled
    # fixed-branch Jacobian is certifiably regular.
    mass, step_size = 2.0, 0.01
    _, step = contact_step(slider_model(mass=mass, floor=False, stop_bounds=(0.0, 1.0)))
    first = _run(step, _source(step, [0.0], [0.0]), step_size, standard_gravity())
    assert bool(first.successful)
    assert not bool(first.fixed_branch_candidate)  # stop activation is discrete
    result = _run(step, first.accepted, step_size, standard_gravity())
    jvp = _gravity_jvp(step, result, standard_gravity())
    valid = np.asarray(result.last_segment.block_valid)

    assert bool(result.fixed_branch_candidate)
    assert bool(jvp.successful) and bool(jvp.coupled_regular)
    # p_lower = -m g_z h on the fixed resting branch; v1 stays zero.
    assert float(jvp.impulse_tangent[0, 0]) == pytest.approx(
        -mass * step_size, rel=1.0e-7
    )
    assert abs(float(jvp.velocity_tangent[0])) <= 1.0e-10

    epsilon = 1.0e-6
    shifted = _run(step, first.accepted, step_size, standard_gravity() + epsilon * _UP)
    difference = (
        np.sum(np.asarray(shifted.last_segment.impulse)[valid, 0])
        - np.sum(np.asarray(result.last_segment.impulse)[valid, 0])
    ) / epsilon
    assert difference == pytest.approx(
        float(np.sum(np.asarray(jvp.impulse_tangent)[valid, 0])), rel=1.0e-5
    )

    vjp, impulse_bar = _impulse_vjp(step, result, standard_gravity())
    assert bool(vjp.successful) and bool(vjp.coupled_regular)
    assert float(jnp.vdot(vjp.gravity_cotangent, _UP)) == pytest.approx(
        float(jnp.vdot(impulse_bar, jvp.impulse_tangent)), rel=1.0e-8
    )
    without_friction = step.step_vjp(
        result,
        jnp.zeros((0,)),
        standard_gravity(),
        (jnp.zeros((1,)), jnp.zeros((1,)), impulse_bar),
        friction_cotangent=False,
    )
    assert bool(without_friction.successful)
    assert without_friction.step_cotangent.static_friction_table is None
    assert without_friction.step_cotangent.dynamic_friction_table is None
    assert jnp.array_equal(without_friction.gravity_cotangent, vjp.gravity_cotangent)


def test_sliding_contact_friction_sensitivity_matches_mechanics_and_differences() -> None:
    carriage, slider, mu = 3.0, 1.0, 0.5
    duration = 0.01
    prepared, step = contact_step(
        planar_slider_model(
            carriage_mass=carriage, slider_mass=slider, friction=(0.8, mu)
        )
    )
    drive, lift = dof(prepared, "drive"), dof(prepared, "lift")
    velocity = jnp.zeros((2,), dtype=step.dtype).at[drive].set(1.0)
    first = _run(
        step,
        _source(step, jnp.zeros_like(velocity), velocity),
        duration,
        standard_gravity(),
    )
    assert bool(first.successful)
    result = _run(step, first.accepted, duration, standard_gravity())
    assert bool(result.successful) and bool(result.fixed_branch_candidate)
    zero_step = jax.tree.map(jnp.zeros_like, eqx.filter(step, eqx.is_inexact_array))
    step_dot = eqx.tree_at(
        lambda value: value.dynamic_friction_table,
        zero_step,
        step.dynamic_friction_table,
    )
    zero_velocity = jnp.zeros_like(velocity)
    command = jnp.zeros((0,), dtype=step.dtype)
    tangent = step.step_jvp(
        result,
        command,
        standard_gravity(),
        (step_dot, zero_velocity, zero_velocity, command, zero_gravity()),
    )
    assert bool(tangent.successful) and bool(tangent.coupled_regular)
    normal_impulse = slider * 9.81 * duration
    expected = -mu * normal_impulse / (carriage + slider)
    assert float(tangent.velocity_tangent[drive]) == pytest.approx(expected, abs=1.0e-9)
    assert abs(float(tangent.velocity_tangent[lift])) <= 1.0e-9
    assert float(tangent.configuration_tangent[drive]) == pytest.approx(
        duration * expected, abs=1.0e-10
    )
    assert float(jnp.linalg.norm(tangent.impulse_tangent[:, 1:])) == pytest.approx(
        mu * normal_impulse, abs=1.0e-9
    )

    epsilon = 2.0e-5
    plus = eqx.tree_at(
        lambda value: value.dynamic_friction_table,
        step,
        (1.0 + epsilon) * step.dynamic_friction_table,
    )
    minus = eqx.tree_at(
        lambda value: value.dynamic_friction_table,
        step,
        (1.0 - epsilon) * step.dynamic_friction_table,
    )
    forward = _run(plus, result.source, duration, standard_gravity())
    backward = _run(minus, result.source, duration, standard_gravity())
    assert bool(forward.successful) and bool(backward.successful)
    difference = (forward.accepted.velocity - backward.accepted.velocity) / (
        2.0 * epsilon
    )
    np.testing.assert_allclose(difference, tangent.velocity_tangent, atol=1.0e-7)

    velocity_bar = zero_velocity.at[drive].set(1.0)
    pullback = step.step_vjp(
        result,
        command,
        standard_gravity(),
        (zero_velocity, velocity_bar, jnp.zeros_like(result.last_segment.impulse)),
    )
    assert bool(pullback.successful) and bool(pullback.coupled_regular)
    assert pullback.step_cotangent.dynamic_friction_table is not None
    dual = jnp.vdot(
        pullback.step_cotangent.dynamic_friction_table, step.dynamic_friction_table
    )
    assert float(dual) == pytest.approx(expected, abs=1.0e-9)


def test_contact_plane_translation_sensitivity_preserves_relative_mechanics() -> None:
    duration = 0.01
    prepared, step = contact_step(planar_slider_model())
    drive, lift = dof(prepared, "drive"), dof(prepared, "lift")
    velocity = jnp.zeros((2,), dtype=step.dtype).at[drive].set(1.0)
    first = _run(
        step,
        _source(step, jnp.zeros_like(velocity), velocity),
        duration,
        standard_gravity(),
    )
    assert bool(first.successful)
    result = _run(step, first.accepted, duration, standard_gravity())
    assert bool(result.successful) and bool(result.fixed_branch_candidate)

    def plane_offset(value: PreparedArticulatedContactStep) -> jax.Array:
        assert value.scene is not None
        return value.scene.planes[0].offset

    zero_step = jax.tree.map(jnp.zeros_like, eqx.filter(step, eqx.is_inexact_array))
    step_dot = eqx.tree_at(plane_offset, zero_step, jnp.ones_like(plane_offset(step)))
    q_dot = jnp.zeros_like(velocity).at[lift].set(1.0)
    zero_velocity = jnp.zeros_like(velocity)
    command = jnp.zeros((0,), dtype=step.dtype)
    tangent = step.step_jvp(
        result,
        command,
        standard_gravity(),
        (step_dot, q_dot, zero_velocity, command, zero_gravity()),
    )
    assert bool(tangent.successful) and bool(tangent.coupled_regular)
    np.testing.assert_allclose(tangent.configuration_tangent, q_dot, atol=1.0e-9)
    np.testing.assert_allclose(tangent.velocity_tangent, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(tangent.impulse_tangent, 0.0, atol=1.0e-9)

    def translated(delta: float) -> Any:
        changed = eqx.tree_at(plane_offset, step, plane_offset(step) + delta)
        source = eqx.tree_at(
            lambda value: value.configuration,
            result.source,
            result.source.configuration + delta * q_dot,
        )
        return _run(changed, source, duration, standard_gravity())

    epsilon = 2.0e-5
    forward, backward = translated(epsilon), translated(-epsilon)
    assert bool(forward.successful) and bool(backward.successful)
    difference = (forward.accepted.configuration - backward.accepted.configuration) / (
        2.0 * epsilon
    )
    np.testing.assert_allclose(difference, tangent.configuration_tangent, atol=1.0e-7)

    pullback = step.step_vjp(
        result,
        command,
        standard_gravity(),
        (q_dot, zero_velocity, jnp.zeros_like(result.last_segment.impulse)),
    )
    assert bool(pullback.successful) and bool(pullback.coupled_regular)
    dual = plane_offset(pullback.step_cotangent) + jnp.vdot(
        pullback.configuration_cotangent, q_dot
    )
    assert float(dual) == pytest.approx(1.0, abs=1.0e-9)


def _scene_attachment_points(value: PreparedArticulatedContactStep) -> tuple[Any, Any]:
    """Primitive attachment points held by the step's canonical contact scene."""

    assert value.scene is not None
    participant = value.scene.participants[0]
    assert isinstance(participant, ArticulatedPrimitiveContactParticipant)
    return participant.geometry.local_first, participant.geometry.local_second


def _attached_at(
    step: PreparedArticulatedContactStep, points: jax.Array
) -> PreparedArticulatedContactStep:
    return eqx.tree_at(_scene_attachment_points, step, (points, points))


def test_scene_geometry_update_reaches_route_and_trajectory_kinematics() -> None:
    # Pendulum link: hinge about z at the origin, COM at (1, 0, 0), tip sphere at
    # its COM. Moving the tip point through the canonical scene must move the
    # step's route kinematics and contact trajectory and carry their tangent:
    # x(q) = Rz(q) (e1 + p).
    _, step = contact_step(pendulum_model(obstacle_angle=None))
    dtype = step.dtype
    configuration = jnp.asarray([0.3], dtype=dtype)
    velocity = jnp.asarray([2.0], dtype=dtype)
    step_size = jnp.asarray(0.1, dtype=dtype)
    angle = 0.3 + 0.5 * 0.1 * 2.0

    def tip(points: jax.Array) -> tuple[jax.Array, jax.Array]:
        moved = _attached_at(step, points)
        participant, geometry = moved.participant, moved.geometry
        assert participant is not None and geometry is not None
        trajectory = participant.trajectory(configuration, velocity, step_size)
        route = geometry.vertex_positions(
            moved.articulation, jnp.asarray([angle], dtype=dtype)
        )
        return trajectory.evaluate(0.5)[0], route[0]

    offset = jnp.asarray([[0.0, 0.25, 0.0]], dtype=dtype)
    direction = jnp.asarray([[1.0, 0.0, 0.0]], dtype=dtype)
    positions, tangents = jax.jvp(tip, (offset,), (direction,))

    cosine, sine = math.cos(angle), math.sin(angle)
    for position, tangent in zip(positions, tangents, strict=True):
        np.testing.assert_allclose(
            position, [cosine - 0.25 * sine, sine + 0.25 * cosine, 0.0], atol=1.0e-12
        )
        np.testing.assert_allclose(tangent, [cosine, sine, 0.0], atol=1.0e-12)


def test_scene_geometry_update_governs_the_contact_step() -> None:
    # The slider sphere touches the floor at q = 0. Raising its attachment point
    # through the scene leaves clearance, so the step is the exact native free
    # fall; lowering it starts in penetration, which the step must refuse.
    prepared, step = contact_step(slider_model())
    source = _source(step, [0.0], [0.0])

    raised = _attached_at(step, jnp.asarray([[0.0, 0.0, 0.2]], dtype=step.dtype))
    free = _run(raised, source, 0.01, standard_gravity())
    native = reduced_semi_implicit_velocity_euler_step(
        prepared.articulation,
        ReducedArticulationState(jnp.asarray([0.0]), jnp.asarray([0.0])),
        jnp.zeros((1,)),
        standard_gravity(),
        0.01,
        policy=step.policy.smooth,
    )
    assert bool(free.successful) and bool(free.free_bypass)
    assert jnp.array_equal(free.accepted.velocity, native.candidate_state.velocity)
    np.testing.assert_allclose(
        free.accepted.configuration, native.candidate_state.configuration, atol=1.0e-14
    )

    lowered = _attached_at(step, jnp.asarray([[0.0, 0.0, -0.05]], dtype=step.dtype))
    refused = _run(lowered, source, 0.01, standard_gravity())
    assert not bool(refused.successful)
    assert int(refused.status) == int(ArticulatedContactStepStatus.INITIALLY_INADMISSIBLE)
    assert _trees_equal(refused.accepted, source)


@pytest.mark.parametrize("model", [slider_model(), capsule_bar_model()])
def test_redundant_resting_contacts_refuse_impulse_derivatives(model: Any) -> None:
    # Sticking on a vertical slider leaves identically zero tangential response
    # rows; a resting capsule's two endpoint supports share one normal DOF. The
    # primal is certified, but the impulse derivative is not unique.
    _, step = contact_step(model)
    first = _run(step, _source(step, [0.0], [0.0]), 0.01, standard_gravity())
    result = _run(step, first.accepted, 0.01, standard_gravity())
    jvp = _gravity_jvp(step, result, standard_gravity())
    vjp, _ = _impulse_vjp(step, result, standard_gravity())

    assert bool(result.successful)
    assert bool(jvp.binding_current) and bool(vjp.binding_current)
    assert not bool(jvp.successful) and not bool(vjp.successful)
    assert not np.any(np.asarray(jvp.impulse_tangent))
    assert not np.any(np.asarray(vjp.gravity_cotangent))


def test_coupled_regularity_budget_refuses_before_materialization() -> None:
    # The coupled (v1, p) Jacobian of the regular resting stop has
    # (1 + 3 * 2)**2 = 49 entries; a 16-entry budget is refused by the native
    # rank owner's plan before any SVD or materialization runs.
    bounded = SVDSolvePolicy(
        DenseSVD(algorithm="qr"), materialization=MaterializationPolicy(max_entries=16)
    )
    _, step = contact_step(
        slider_model(floor=False, stop_bounds=(0.0, 1.0)),
        step_policy(derivative_regularity=bounded),
    )
    first = _run(step, _source(step, [0.0], [0.0]), 0.01, standard_gravity())
    result = _run(step, first.accepted, 0.01, standard_gravity())
    jvp = _gravity_jvp(step, result, standard_gravity())
    vjp, _ = _impulse_vjp(step, result, standard_gravity())

    assert bool(result.fixed_branch_candidate) and bool(jvp.branch_admitted)
    for derivative in (jvp, vjp):
        assert derivative.coupled_regularity is not None
        assert derivative.coupled_regularity.route == "unavailable"
        assert not bool(derivative.coupled_regular)
        assert not bool(derivative.linear_successful)
        assert not bool(derivative.successful)


def _perturbed_inertia(
    step: PreparedArticulatedContactStep,
) -> PreparedArticulatedContactStep:
    return eqx.tree_at(
        lambda value: value.articulation,
        step,
        jax.tree.map(
            lambda leaf: leaf * 1.01 if eqx.is_inexact_array(leaf) else leaf,
            step.articulation,
        ),
    )


@pytest.mark.parametrize(
    ("model", "configuration"),
    [
        (slider_model(), [0.5]),  # contact-free smooth route
        (slider_model(floor=False, stop_bounds=(0.0, 1.0)), [0.0]),  # resting stop
    ],
)
@pytest.mark.parametrize("stale", ["gravity", "inertia"])
def test_derivatives_refuse_inputs_that_did_not_produce_the_result(
    model: Any, configuration: list[float], stale: str
) -> None:
    _, step = contact_step(model)
    first = _run(step, _source(step, configuration, [0.0]), 0.01, standard_gravity())
    result = _run(step, first.accepted, 0.01, standard_gravity())
    assert bool(_gravity_jvp(step, result, standard_gravity()).successful)
    other_step, gravity = (
        (step, standard_gravity() + 0.5 * _UP)
        if stale == "gravity"
        else (_perturbed_inertia(step), standard_gravity())
    )

    jvp = _gravity_jvp(other_step, result, gravity)
    vjp, _ = _impulse_vjp(other_step, result, gravity)

    assert not bool(jvp.binding_current) and not bool(jvp.successful)
    assert not bool(vjp.binding_current) and not bool(vjp.successful)
    assert not np.any(np.asarray(jvp.velocity_tangent))
    assert not np.any(np.asarray(vjp.gravity_cotangent))


def test_endpoint_law_is_certified_at_the_accepted_velocity() -> None:
    # A loose coupled-root tolerance can stop at v1 = -5.96e-8 m/s: the
    # response map's own velocity satisfies the law, but the accepted state
    # has |p_n g(q1)/h| = 1.19e-7 J, above the 1e-8 J complementarity budget.
    _, step = contact_step(
        slider_model(mass=2.0),
        step_policy(root_damping=0.5, root_absolute_tolerance=1.0e-7),
    )
    source = _source(step, [0.0], [-1.0])
    result = _run(step, source, 0.01, zero_gravity())
    segment = result.last_segment
    valid = np.asarray(segment.block_valid)
    duration = float(segment.duration)
    actual = np.abs(
        np.asarray(segment.impulse)[valid, 0] * np.asarray(segment.gap)[valid] / duration
    )
    budget = WORK_TOLERANCE * float(segment.fraction)

    assert np.all(actual > budget)
    assert not bool(segment.cone_successful)
    assert np.allclose(
        np.asarray(segment.law_velocity)[valid, 0],
        np.asarray(segment.gap)[valid] / duration,
        rtol=1.0e-6,
        atol=1.0e-14,
    )
    assert not bool(result.successful)
    assert _trees_equal(result.accepted, source)


@pytest.mark.parametrize(
    ("model", "configuration", "velocity", "gravity"),
    [
        (slider_model(), [0.0], [-1.0], zero_gravity()),
        (slider_model(), [0.0], [0.0], standard_gravity()),
        (capsule_bar_model(), [0.0], [0.0], standard_gravity()),
        (pendulum_model(obstacle_angle=1.21), [0.95], [1.0], zero_gravity()),
    ],
)
def test_device_lbvh_route_matches_exhaustive_authority(
    model: Any, configuration: list[float], velocity: list[float], gravity: jax.Array
) -> None:
    _, exhaustive = contact_step(model)
    _, device = contact_step(model, search_backend="lbvh")
    reference = _run(
        exhaustive, _source(exhaustive, configuration, velocity), 0.01, gravity
    )
    result = _run(device, _source(device, configuration, velocity), 0.01, gravity)

    assert bool(reference.successful) and bool(result.successful)
    assert bool(result.free_bypass) == bool(reference.free_bypass)
    assert np.allclose(
        np.asarray(result.accepted.velocity),
        np.asarray(reference.accepted.velocity),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    reference_keys = np.sort(
        np.asarray(reference.last_segment.block_keys)[
            np.asarray(reference.last_segment.block_valid)
        ]
    )
    device_keys = np.sort(
        np.asarray(result.last_segment.block_keys)[
            np.asarray(result.last_segment.block_valid)
        ]
    )
    assert device_keys.tolist() == reference_keys.tolist()


def test_geometry_without_possible_routes_runs_contact_free_with_zero_capacity() -> None:
    prepared, step = contact_step(slider_model(floor=False))
    result = _run(step, _source(step, [0.0], [0.0]), 0.01, standard_gravity())
    native = reduced_semi_implicit_velocity_euler_step(
        prepared.articulation,
        ReducedArticulationState(jnp.asarray([0.0]), jnp.asarray([0.0])),
        jnp.zeros((1,)),
        standard_gravity(),
        0.01,
        policy=step.policy.smooth,
    )

    assert step.contact_capacity == 0 and step.search is None
    assert bool(result.successful) and bool(result.free_bypass)
    assert jnp.array_equal(result.accepted.velocity, native.candidate_state.velocity)
    with pytest.raises(ValueError, match="Zero route capacity"):
        contact_step(slider_model(), route_capacities=(0, 0, 0, 0, 0, 0))
