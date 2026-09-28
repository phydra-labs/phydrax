#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tests._support.assertions import assert_tree_equal


def _prepared() -> phx.solver.PreparedSHAKERATTLEPlan:
    plan = phx.solver.SHAKERATTLEPlan(
        maximum_projection_steps=8,
        constraint_tolerance=1.0e-10,
        condition_limit=1.0e10,
    )
    return plan.prepare(
        jnp.ones((2,), dtype=jnp.float64),
        lambda configuration, _: configuration,
        lambda configuration, _: jnp.asarray(
            [jnp.vdot(configuration, configuration) - 1.0]
        ),
    )


def test_shake_rattle_contracts() -> None:
    state = phx.solver.ConstrainedMechanicalState(
        jnp.asarray((1.0, 0.0), dtype=jnp.float64),
        jnp.asarray((0.0, 1.0), dtype=jnp.float64),
    )
    prepared = _prepared()

    result = jax.jit(lambda value: prepared.step(value, 0.05))(state)

    assert bool(result.accepted)
    assert int(result.evidence.status) == phx.solver.ConstrainedMechanicsStatus.SUCCESS
    assert result.evidence.position_residual < 1.0e-10
    assert result.evidence.velocity_residual < 1.0e-10
    assert int(result.evidence.constraint_rank) == 1
    assert np.isfinite(float(result.evidence.constraint_condition))
    assert float(result.evidence.projection_work) <= 1.0e-14
    assert jnp.allclose(
        jnp.vdot(result.state.configuration, result.state.configuration), 1.0
    )
    assert jnp.allclose(
        jnp.vdot(result.state.configuration, result.state.momentum),
        0.0,
        atol=1.0e-10,
    )


def test_constraint_operator_exposes_matching_jvp_and_transpose() -> None:
    prepared = _prepared()
    configuration = jnp.asarray((0.6, 0.8), dtype=jnp.float64)
    operator = prepared.constraint_operator(configuration)
    tangent = jnp.asarray((0.3, -0.2), dtype=jnp.float64)
    covector = jnp.asarray((1.7,), dtype=jnp.float64)

    primal_pair = jnp.vdot(operator.mv(tangent), covector)
    transpose_pair = jnp.vdot(tangent, operator.transpose_mv(covector))

    np.testing.assert_allclose(primal_pair, transpose_pair, rtol=1.0e-14)


def test_rank_deficient_projection_rolls_back_with_evidence() -> None:
    plan = phx.solver.SHAKERATTLEPlan(maximum_projection_steps=4)
    prepared = plan.prepare(
        jnp.ones((2,), dtype=jnp.float64),
        lambda configuration, _: jnp.zeros_like(configuration),
        lambda configuration, _: jnp.asarray([configuration[0], 2.0 * configuration[0]]),
    )
    state = phx.solver.ConstrainedMechanicalState(
        jnp.asarray((0.0, 1.0), dtype=jnp.float64),
        jnp.asarray((1.0, 0.0), dtype=jnp.float64),
    )

    result = prepared.step(state, 0.1)

    assert not bool(result.accepted)
    assert int(result.evidence.constraint_rank) == 1
    assert int(result.evidence.status) in (
        phx.solver.ConstrainedMechanicsStatus.POSITION_PROJECTION_FAILED,
        phx.solver.ConstrainedMechanicsStatus.VELOCITY_PROJECTION_FAILED,
        phx.solver.ConstrainedMechanicsStatus.RANK_DEFICIENT,
    )
    assert_tree_equal(result.state, state)


def test_ill_conditioned_projection_rolls_back_with_condition_status() -> None:
    plan = phx.solver.SHAKERATTLEPlan(condition_limit=1.0e6)
    prepared = plan.prepare(
        jnp.ones((2,), dtype=jnp.float64),
        lambda configuration, _: jnp.zeros_like(configuration),
        lambda configuration, _: jnp.asarray(
            [configuration[0], 1.0e-4 * configuration[1]]
        ),
    )
    state = phx.solver.ConstrainedMechanicalState(
        jnp.zeros((2,), dtype=jnp.float64),
        jnp.ones((2,), dtype=jnp.float64),
    )

    result = prepared.step(state, 0.1)

    assert not bool(result.accepted)
    assert (
        int(result.evidence.status)
        == phx.solver.ConstrainedMechanicsStatus.CONDITION_LIMIT_REACHED
    )
    assert float(result.evidence.constraint_condition) > 1.0e6
    assert_tree_equal(result.state, state)


def test_constraint_gram_resource_limit_reports_linear_status() -> None:
    plan = phx.solver.SHAKERATTLEPlan(
        maximum_projection_steps=2,
        maximum_constraint_entries=1,
    )
    prepared = plan.prepare(
        jnp.ones((64,), dtype=jnp.float64),
        lambda configuration, _: jnp.zeros_like(configuration),
        lambda configuration, _: configuration[:2],
    )
    state = phx.solver.ConstrainedMechanicalState(
        jnp.zeros((64,), dtype=jnp.float64),
        jnp.ones((64,), dtype=jnp.float64),
    )

    result = jax.jit(lambda value: prepared.step(value, 0.1))(state)

    assert not bool(result.accepted)
    assert (
        int(result.evidence.position_linear_status)
        == phx.linalg.LinearSolveStatus.CAPABILITY_REJECTED
    )
    assert int(result.evidence.constraint_rank) == 0
    assert np.isinf(float(result.evidence.constraint_condition))


def test_nonpositive_step_rolls_back_without_committing() -> None:
    state = phx.solver.ConstrainedMechanicalState(
        jnp.asarray((1.0, 0.0), dtype=jnp.float64),
        jnp.asarray((0.0, 1.0), dtype=jnp.float64),
    )

    result = _prepared().step(state, 0.0)

    assert not bool(result.accepted)
    assert (
        int(result.evidence.status) == phx.solver.ConstrainedMechanicsStatus.INVALID_STEP
    )
    assert_tree_equal(result.state, state)


def test_fixed_constraint_step_jvp_matches_finite_difference() -> None:
    plan = phx.solver.SHAKERATTLEPlan()
    prepared = plan.prepare(
        jnp.ones((2,), dtype=jnp.float64),
        lambda configuration, _: jnp.zeros_like(configuration),
        lambda configuration, _: configuration[:1],
    )
    configuration = jnp.asarray((0.0, 0.2), dtype=jnp.float64)
    base = jnp.asarray((0.0, 0.4), dtype=jnp.float64)
    tangent = jnp.asarray((0.0, -0.7), dtype=jnp.float64)

    def advance(momentum: jax.Array) -> jax.Array:
        state = phx.solver.ConstrainedMechanicalState(configuration, momentum)
        return prepared.step(state, 0.05).state.configuration

    _, jvp = jax.jvp(advance, (base,), (tangent,))
    epsilon = 1.0e-6
    finite_difference = (
        advance(base + epsilon * tangent) - advance(base - epsilon * tangent)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(jvp, finite_difference, rtol=1.0e-8, atol=1.0e-10)
