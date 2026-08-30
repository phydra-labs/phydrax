#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _operator():
    space = phx.linalg.ArraySpace((2,))
    return phx.linalg.DenseLinearOperator(
        jnp.asarray([[2.0, 0.0], [0.0, 3.0]]), source=space, target=space
    )


def test_projection_history_exactly_recovers_rhs_span():
    operator = _operator()
    policy = phx.linalg.LinearSolveHistoryPolicy("projection", capacity=3)
    history = phx.linalg.LinearSolveHistory.empty(operator, policy, "shifted-family")
    first = jnp.asarray([1.0, 0.0])
    second = jnp.asarray([0.0, 2.0])
    history = history.update(operator, first, time=0.0)
    history = history.update(operator, second, time=1.0)
    rhs = operator.mv(2.0 * first + 3.0 * second)

    guess, diagnostics = history.initial_guess(rhs)

    assert jnp.allclose(guess, jnp.asarray([2.0, 6.0]))
    assert diagnostics.effective_dimension == 2
    assert diagnostics.rank == 2
    assert diagnostics.projection_residual_norm < 1.0e-12


def test_rolling_qr_history_recovers_span_after_eviction():
    operator = _operator()
    history = phx.linalg.LinearSolveHistory.empty(
        operator,
        phx.linalg.LinearSolveHistoryPolicy("rolling-qr", capacity=2),
        "family",
    )
    history = history.update(operator, jnp.asarray([1.0, 1.0]), time=0.0)
    history = history.update(operator, jnp.asarray([1.0, 0.0]), time=1.0)
    history = history.update(operator, jnp.asarray([0.0, 2.0]), time=2.0)

    guess, diagnostics = history.initial_guess(operator.mv(jnp.asarray([3.0, 4.0])))

    assert jnp.allclose(guess, jnp.asarray([3.0, 4.0]), atol=1.0e-10)
    assert diagnostics.rank == 2
    assert history.effective_dimension == 2


def test_rejected_history_update_is_bitwise_inert():
    operator = _operator()
    history = phx.linalg.LinearSolveHistory.empty(
        operator,
        phx.linalg.LinearSolveHistoryPolicy("last-solution", capacity=2),
        "family",
    )
    rejected = history.update(operator, jnp.ones((2,)), accepted=False)

    assert rejected.history_id == history.history_id
    assert rejected.effective_dimension == 0
    assert rejected.update_count == 0
    assert jnp.array_equal(rejected.solution_basis, history.solution_basis)


def test_stabilized_extrapolation_reproduces_linear_history():
    operator = _operator()
    history = phx.linalg.LinearSolveHistory.empty(
        operator,
        phx.linalg.LinearSolveHistoryPolicy(
            "stabilized-extrapolation", capacity=3, extrapolation_degree=1
        ),
        "family",
    )
    history = history.update(operator, jnp.asarray([0.0, 1.0]), time=0.0)
    history = history.update(operator, jnp.asarray([1.0, 3.0]), time=1.0)

    guess, diagnostics = history.initial_guess(operator.mv(jnp.zeros((2,))), time=2.0)

    assert jnp.allclose(guess, jnp.asarray([2.0, 5.0]), atol=1.0e-12)
    assert diagnostics.rank == 2
