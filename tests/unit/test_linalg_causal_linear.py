#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.linalg._causal_linear import (
    associative_affine_solve,
    associative_transpose_solve,
    causal_linearized_residual,
    solve_causal_least_squares,
)


def _dense_operator(transitions):
    num_steps, state_size, _ = transitions.shape
    rows = []
    for time in range(num_steps):
        blocks = []
        for source in range(num_steps):
            if source == time:
                block = jnp.eye(state_size, dtype=transitions.dtype)
            elif source == time - 1:
                block = -transitions[time]
            else:
                block = jnp.zeros((state_size, state_size), dtype=transitions.dtype)
            blocks.append(block)
        rows.append(jnp.concatenate(blocks, axis=1))
    return jnp.concatenate(rows, axis=0)


def _problem(dtype=jnp.float64):
    transitions = jnp.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.2, -0.1], [0.05, 0.3]],
            [[-0.15, 0.08], [0.1, 0.25]],
            [[0.12, 0.03], [-0.05, 0.18]],
        ],
        dtype=dtype,
    )
    residuals = jnp.linspace(-0.4, 0.5, 8, dtype=dtype).reshape((4, 2))
    return transitions, residuals


@pytest.mark.parametrize("damping", (0.0, 0.3))
def test_causal_least_squares_matches_dense_normal_equations(damping):
    transitions, residuals = _problem()
    operator = _dense_operator(transitions)
    expected = jnp.linalg.solve(
        operator.T @ operator + damping * jnp.eye(operator.shape[1]),
        -operator.T @ residuals.reshape((-1,)),
    ).reshape(residuals.shape)

    actual = jax.jit(solve_causal_least_squares)(
        transitions,
        residuals,
        jnp.asarray(damping),
    )

    assert jnp.allclose(actual, expected, atol=1e-11, rtol=1e-11)
    assert jnp.allclose(
        causal_linearized_residual(transitions, residuals, actual).reshape((-1,)),
        residuals.reshape((-1,)) + operator @ actual.reshape((-1,)),
    )


def test_affine_and_transpose_scans_match_dense_triangular_solves():
    transitions, right = _problem()
    operator = _dense_operator(transitions)

    forward = associative_affine_solve(transitions, right)
    transpose = associative_transpose_solve(transitions, right)

    assert jnp.allclose(
        forward.reshape((-1,)),
        jnp.linalg.solve(operator, right.reshape((-1,))),
    )
    assert jnp.allclose(
        transpose.reshape((-1,)),
        jnp.linalg.solve(operator.T, right.reshape((-1,))),
    )


def test_causal_linear_solves_have_correct_reverse_derivatives():
    transitions, right = _problem()
    operator = _dense_operator(transitions)
    weights = jnp.arange(right.size, dtype=right.dtype).reshape(right.shape)

    gradient = jax.grad(
        lambda rhs: jnp.sum(associative_affine_solve(transitions, rhs) * weights)
    )(right)
    expected = jnp.linalg.solve(operator.T, weights.reshape((-1,))).reshape(right.shape)

    assert jnp.allclose(gradient, expected)


def test_causal_linear_contract_rejects_invalid_shapes_and_damping():
    transitions, residuals = _problem()
    with pytest.raises(ValueError, match="transitions"):
        associative_affine_solve(transitions[:, 0], residuals)
    with pytest.raises(ValueError, match="offsets"):
        associative_affine_solve(transitions, residuals[:, :1])
    with pytest.raises(Exception, match="nonnegative"):
        solve_causal_least_squares(transitions, residuals, jnp.asarray(-1.0))
