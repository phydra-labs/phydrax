#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_h2_and_hinfinity_state_feedback_stabilize_scalar_systems() -> None:
    h2 = phx.control.h2_state_feedback(
        jnp.asarray([[0.9]]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
    )
    hinfinity = phx.control.hinfinity_state_feedback(
        jnp.asarray([[-1.0]]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[1.0]]),
        2.0,
    )

    assert h2.stable and h2.certified
    assert h2.performance >= 0.0
    assert hinfinity.stable and hinfinity.certified


def test_tube_mpc_tightens_state_and_control_bounds() -> None:
    plan = phx.control.prepare_tube_mpc(
        jnp.asarray([[0.8]]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.3]]),
        jnp.asarray([0.1]),
        jnp.asarray([-2.0]),
        jnp.asarray([2.0]),
        jnp.asarray([-1.0]),
        jnp.asarray([1.0]),
    )

    assert plan.error_radius[0] > 0.1
    assert plan.tightened_state_lower[0] > -2.0
    assert plan.tightened_state_upper[0] < 2.0
    assert jnp.abs(plan.control(jnp.asarray([0.0]), jnp.asarray([1.0]))[0]) <= 1.0


def test_gaussian_chance_constraint_uses_covariance_margin() -> None:
    constraint = phx.control.gaussian_chance_constraint(
        jnp.asarray([1.0, 0.0]), 2.0, 0.95
    )

    margin = constraint.margin(jnp.asarray([1.0, 0.0]), jnp.eye(2) * 0.04)

    assert margin > 0.0
    assert margin < 1.0


def test_linear_moving_horizon_recovers_noise_free_trajectory() -> None:
    measurements = jnp.asarray([[1.0], [2.0], [4.0]])
    result = phx.control.linear_moving_horizon_estimate(
        jnp.asarray([[2.0]]),
        jnp.asarray([[1.0]]),
        measurements,
        jnp.asarray([1.0]),
        jnp.asarray([[1.0e6]]),
        jnp.asarray([[1.0e6]]),
        jnp.asarray([[1.0e6]]),
    )

    assert bool(result.successful)
    assert jnp.allclose(result.states[:, 0], measurements[:, 0], atol=2.0e-6)


def test_safety_projection_and_reachable_box_are_fail_closed() -> None:
    filtered = phx.control.project_control_halfspaces(
        jnp.asarray([2.0, -2.0]),
        jnp.asarray([[1.0, 0.0], [0.0, -1.0]]),
        jnp.asarray([1.0, 1.0]),
    )
    reachable = phx.control.propagate_linear_reachable_box(
        jnp.eye(2),
        jnp.eye(2),
        phx.control.LinearReachableBox(jnp.zeros(2), jnp.ones(2)),
        phx.control.LinearReachableBox(jnp.zeros(2), jnp.full((2,), 0.5)),
        jnp.full((2,), 0.25),
    )

    assert bool(filtered.successful)
    assert jnp.all(filtered.control <= 1.0)
    assert jnp.all(filtered.control >= -1.0)
    assert jnp.allclose(reachable.radius, 1.75)
