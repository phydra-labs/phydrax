#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.optim._kfac._blocks import (
    estimate_kron_factors,
    initialize_block_state,
    kron_dense_matrix,
    kron_diagonal,
    kron_matvec,
    preconditioned_conjugate_gradient,
    solve_block_direction,
    update_block_state,
)
from phydrax.optim._kfac._types import (
    AffineBlockSpec,
    BlockCurvatureState,
    DenseFactorState,
    KronFactorState,
    ParameterLayout,
    UncoveredBlockSpec,
)


def _affine_block():
    return AffineBlockSpec(
        name="layer",
        indices=(0, 1, 2, 3, 4, 5),
        output_size=2,
        input_size=3,
        has_bias=True,
    )


@pytest.mark.parametrize("approximation", ["expand", "reduce"])
def test_kron_factor_strategies_are_psd_and_preserve_curvature_trace(approximation):
    block = _affine_block()
    jacobian = jnp.asarray(
        [
            [1.0, -0.3, 0.2, 0.5, 0.1, -0.4],
            [-0.2, 0.7, 0.4, 0.3, -0.8, 0.6],
            [0.9, 0.2, -0.5, -0.1, 0.4, 0.8],
        ]
    )
    activation, sensitivity = estimate_kron_factors(
        jacobian,
        block,
        approximation=approximation,
    )

    assert activation.shape == (3, 3)
    assert sensitivity.shape == (2, 2)
    assert jnp.min(jnp.linalg.eigvalsh(activation)) >= -1e-12
    assert jnp.min(jnp.linalg.eigvalsh(sensitivity)) >= -1e-12
    assert jnp.allclose(
        jnp.trace(activation) * jnp.trace(sensitivity),
        jnp.sum(jnp.square(jacobian)),
        rtol=1e-10,
        atol=1e-10,
    )


def test_single_rank_one_event_kron_block_matches_exact_ggn():
    block = _affine_block()
    activation_vector = jnp.asarray([0.4, -0.2, 1.0])
    sensitivity_vector = jnp.asarray([0.7, -0.3])
    jacobian = jnp.outer(sensitivity_vector, activation_vector).reshape((1, -1))
    activation, sensitivity = estimate_kron_factors(
        jacobian,
        block,
        approximation="expand",
    )
    factors = (KronFactorState(activation, sensitivity, jnp.asarray(True)),)

    assert jnp.allclose(
        kron_dense_matrix(factors),
        jacobian.T @ jacobian,
        rtol=1e-10,
        atol=1e-10,
    )


def test_kronecker_sum_matvec_and_diagonal_match_dense_oracle():
    block = _affine_block()
    factors = (
        KronFactorState(
            jnp.asarray([[2.0, 0.2, 0.1], [0.2, 1.5, -0.1], [0.1, -0.1, 1.0]]),
            jnp.asarray([[1.2, 0.1], [0.1, 0.8]]),
            jnp.asarray(True),
        ),
        KronFactorState(
            jnp.asarray([[0.5, 0.0, 0.1], [0.0, 0.7, 0.0], [0.1, 0.0, 0.9]]),
            jnp.asarray([[0.4, -0.05], [-0.05, 0.6]]),
            jnp.asarray(True),
        ),
    )
    vector = jnp.linspace(-0.4, 0.6, 6)
    damping = 0.03
    dense = kron_dense_matrix(factors, damping=damping)

    assert jnp.allclose(
        kron_matvec(factors, vector, block, damping=damping),
        dense @ vector,
        rtol=1e-12,
        atol=1e-12,
    )
    assert jnp.allclose(
        kron_diagonal(factors, block, damping=damping),
        jnp.diag(dense),
        rtol=1e-12,
        atol=1e-12,
    )


def test_preconditioned_conjugate_gradient_matches_dense_solve_and_zero_rhs():
    matrix = jnp.asarray([[4.0, 0.5, 0.2], [0.5, 3.0, -0.1], [0.2, -0.1, 2.0]])
    rhs = jnp.asarray([1.0, -0.5, 0.25])
    solution, iterations, relative_residual = preconditioned_conjugate_gradient(
        lambda vector: matrix @ vector,
        rhs,
        jnp.diag(matrix),
        max_steps=10,
        relative_tolerance=1e-12,
    )
    zero, zero_iterations, zero_residual = preconditioned_conjugate_gradient(
        lambda vector: matrix @ vector,
        jnp.zeros_like(rhs),
        jnp.diag(matrix),
        max_steps=10,
        relative_tolerance=1e-12,
    )
    small_rhs = 1e-10 * rhs
    small_solution, _, small_relative_residual = preconditioned_conjugate_gradient(
        lambda vector: matrix @ vector,
        small_rhs,
        jnp.diag(matrix),
        max_steps=10,
        relative_tolerance=1e-8,
    )

    assert iterations <= 3
    assert relative_residual < 1e-10
    assert jnp.allclose(solution, jnp.linalg.solve(matrix, rhs), rtol=1e-10, atol=1e-10)
    assert small_relative_residual < 1e-8
    assert jnp.allclose(
        small_solution,
        jnp.linalg.solve(matrix, small_rhs),
        rtol=1e-8,
        atol=1e-20,
    )
    assert zero_iterations == 0
    assert zero_residual == 0.0
    assert jnp.allclose(zero, 0.0)


def test_per_term_factor_ema_initializes_from_first_observation():
    block = _affine_block()
    layout = ParameterLayout((block,), None, 6)
    state = initialize_block_state(layout, num_terms=2, dtype=jnp.float64)
    jacobians = (
        jnp.asarray([[1.0, 0.2, -0.1, 0.3, 0.4, -0.2]]),
        jnp.asarray([[0.5, -0.2, 0.4, -0.1, 0.7, 0.3]]),
    )
    updated = update_block_state(
        state,
        layout,
        jacobians,
        approximation="expand",
        factor_decay=0.9,
    )

    assert len(updated.affine[0]) == 2
    for term_index, factor in enumerate(updated.affine[0]):
        expected_a, expected_g = estimate_kron_factors(
            jacobians[term_index],
            block,
            approximation="expand",
        )
        assert factor.initialized
        assert jnp.allclose(factor.activation, expected_a)
        assert jnp.allclose(factor.sensitivity, expected_g)


def test_exact_and_diagonal_uncovered_blocks_match_dense_oracles():
    gradient = jnp.asarray([1.0, -2.0, 0.5])
    curvature = jnp.asarray([[3.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 1.5]])
    damping = 0.1
    exact_spec = UncoveredBlockSpec("uncovered", (0, 1, 2), "exact")
    exact_layout = ParameterLayout((), exact_spec, 3)
    exact_state = BlockCurvatureState(
        (),
        (DenseFactorState(curvature, jnp.asarray(True)),),
    )
    exact_direction, _, _ = solve_block_direction(
        exact_state,
        exact_layout,
        gradient,
        damping=damping,
        cg_max_steps=10,
        cg_relative_tolerance=1e-10,
    )

    diagonal_spec = UncoveredBlockSpec("uncovered", (0, 1, 2), "diagonal")
    diagonal_layout = ParameterLayout((), diagonal_spec, 3)
    diagonal = jnp.diag(curvature)
    diagonal_state = BlockCurvatureState(
        (),
        (DenseFactorState(diagonal, jnp.asarray(True)),),
    )
    diagonal_direction, _, _ = solve_block_direction(
        diagonal_state,
        diagonal_layout,
        gradient,
        damping=damping,
        cg_max_steps=10,
        cg_relative_tolerance=1e-10,
    )

    assert jnp.allclose(
        exact_direction,
        jnp.linalg.solve(curvature + damping * jnp.eye(3), gradient),
        rtol=1e-12,
        atol=1e-12,
    )
    assert jnp.allclose(
        diagonal_direction,
        gradient / (diagonal + damping),
        rtol=1e-12,
        atol=1e-12,
    )
