#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from phydrax.control import (
    continuous_controllability_gramian,
    continuous_controllability_gramian_action,
    continuous_lyapunov_solution,
    continuous_observability_gramian,
    continuous_observability_gramian_action,
    discrete_controllability_gramian,
    discrete_controllability_gramian_action,
    discrete_lyapunov_solution,
    discrete_observability_gramian,
    discrete_observability_gramian_action,
    finite_continuous_lyapunov,
    finite_discrete_lyapunov,
    LinearMatrixEquationStatus,
    solve_continuous_lyapunov,
    solve_continuous_lyapunov_krylov,
    solve_discrete_lyapunov,
    solve_discrete_lyapunov_krylov,
)


def test_dense_continuous_and_discrete_lyapunov_match_scipy_references():
    continuous_matrix = np.array([[-1.0, 1.7], [-0.4, -2.0]])
    source = np.array([[1.0, 0.2], [0.2, 2.0]])
    continuous = solve_continuous_lyapunov(continuous_matrix, source, tolerance=1e-10)
    continuous_reference = scipy.linalg.solve_continuous_lyapunov(
        continuous_matrix, -source
    )
    np.testing.assert_allclose(
        continuous.value, continuous_reference, rtol=2e-11, atol=2e-11
    )
    assert bool(continuous.diagnostics.stable)
    assert bool(continuous.diagnostics.converged)
    assert continuous.diagnostics.relative_residual < 1e-11
    assert np.isfinite(float(continuous.diagnostics.condition_number))

    discrete_matrix = np.array([[0.6, 0.2], [0.0, 0.25]])
    discrete = solve_discrete_lyapunov(discrete_matrix, source, tolerance=1e-10)
    discrete_reference = scipy.linalg.solve_discrete_lyapunov(discrete_matrix, source)
    np.testing.assert_allclose(discrete.value, discrete_reference, rtol=2e-11, atol=2e-11)
    assert bool(discrete.diagnostics.stable)
    assert bool(discrete.diagnostics.converged)
    assert discrete.diagnostics.method == "bilinear-bartels-stewart"


def test_lyapunov_equations_diagnose_unstable_and_singular_marginal_systems():
    source = jnp.eye(2)
    unstable_continuous = solve_continuous_lyapunov(
        jnp.diag(jnp.array([0.4, -1.0])), source, tolerance=1e-10
    )
    continuous_reference = scipy.linalg.solve_continuous_lyapunov(
        np.diag([0.4, -1.0]), -np.eye(2)
    )
    np.testing.assert_allclose(
        unstable_continuous.value, continuous_reference, rtol=1e-10, atol=1e-10
    )
    assert not bool(unstable_continuous.diagnostics.stable)
    assert bool(unstable_continuous.diagnostics.converged)

    unstable_discrete = solve_discrete_lyapunov(
        jnp.diag(jnp.array([1.2, 0.4])), source, tolerance=1e-10
    )
    discrete_reference = scipy.linalg.solve_discrete_lyapunov(
        np.diag([1.2, 0.4]), np.eye(2)
    )
    np.testing.assert_allclose(
        unstable_discrete.value, discrete_reference, rtol=1e-10, atol=1e-10
    )
    assert not bool(unstable_discrete.diagnostics.stable)
    assert bool(unstable_discrete.diagnostics.converged)

    marginal = solve_continuous_lyapunov(jnp.zeros((1, 1)), jnp.ones((1, 1)))
    assert not bool(marginal.diagnostics.converged)
    assert int(marginal.diagnostics.status) == int(
        LinearMatrixEquationStatus.SINGULAR_EQUATION
    )
    assert marginal.diagnostics.spectral_separation == 0.0


def test_finite_lyapunov_integrals_and_sums_match_scalar_analytic_values():
    horizon = 1.3
    source = jnp.ones((1, 1))
    stable = finite_continuous_lyapunov(jnp.array([[-2.0]]), source, horizon)
    marginal = finite_continuous_lyapunov(jnp.zeros((1, 1)), source, horizon)
    unstable = finite_continuous_lyapunov(jnp.array([[0.5]]), source, horizon)
    np.testing.assert_allclose(stable.value, [[(1.0 - np.exp(-4.0 * horizon)) / 4.0]])
    np.testing.assert_allclose(marginal.value, [[horizon]])
    np.testing.assert_allclose(unstable.value, [[np.exp(horizon) - 1.0]])
    assert bool(stable.diagnostics.converged)
    assert bool(marginal.diagnostics.converged)
    assert bool(unstable.diagnostics.converged)

    matrix = jnp.array([[0.5]])
    steps = 7
    discrete = finite_discrete_lyapunov(matrix, source, steps)
    np.testing.assert_allclose(discrete.value, [[sum(0.25**k for k in range(steps))]])
    assert int(discrete.diagnostics.iterations) == steps
    assert bool(discrete.diagnostics.converged)


def test_infinite_gramians_require_stability_but_finite_gramians_do_not():
    factor = jnp.ones((1, 1))
    stable = continuous_controllability_gramian(jnp.array([[-1.0]]), factor)
    unstable = continuous_controllability_gramian(jnp.array([[0.2]]), factor)
    marginal = continuous_controllability_gramian(jnp.zeros((1, 1)), factor)
    assert bool(stable.diagnostics.converged)
    assert int(stable.diagnostics.status) == LinearMatrixEquationStatus.CONVERGED
    assert not bool(unstable.diagnostics.converged)
    assert int(unstable.diagnostics.status) == LinearMatrixEquationStatus.UNSTABLE_SYSTEM
    assert not bool(marginal.diagnostics.converged)
    assert int(marginal.diagnostics.status) == LinearMatrixEquationStatus.MARGINAL_SYSTEM

    finite_unstable = continuous_controllability_gramian(
        jnp.array([[0.2]]), factor, horizon=2.0
    )
    assert bool(finite_unstable.diagnostics.converged)
    assert bool(finite_unstable.diagnostics.positive_semidefinite)

    discrete_unstable = discrete_controllability_gramian(jnp.array([[1.1]]), factor)
    discrete_marginal = discrete_controllability_gramian(jnp.array([[1.0]]), factor)
    assert int(discrete_unstable.diagnostics.status) == (
        LinearMatrixEquationStatus.UNSTABLE_SYSTEM
    )
    assert int(discrete_marginal.diagnostics.status) == (
        LinearMatrixEquationStatus.MARGINAL_SYSTEM
    )


def test_infinite_gramian_nonfinite_inputs_take_status_precedence():
    result = continuous_controllability_gramian(
        jnp.array([[jnp.nan]]),
        jnp.ones((1, 1)),
    )

    assert not bool(result.diagnostics.finite)
    assert not bool(result.diagnostics.converged)
    assert int(result.diagnostics.status) == LinearMatrixEquationStatus.NONFINITE


def test_continuous_gramian_action_rejects_self_certifying_order_one():
    with pytest.raises(ValueError, match="quadrature_order must be at least 2"):
        continuous_controllability_gramian_action(
            lambda value: value,
            lambda value: value,
            lambda value: value,
            lambda value: value,
            jnp.ones((1,)),
            1.0,
            quadrature_order=1,
            krylov_dimension=1,
        )


def test_long_discrete_gramian_action_uses_nonrecursive_iteration():
    steps = 2048
    vector = jnp.array([1.5])
    result = discrete_controllability_gramian_action(
        lambda value: value,
        lambda value: value,
        lambda value: value,
        lambda value: value,
        vector,
        steps,
    )

    np.testing.assert_allclose(result.value, steps * vector)
    assert int(result.diagnostics.terms) == steps
    assert bool(result.diagnostics.converged)


def test_singular_reachability_and_observability_are_preserved_and_reported():
    continuous_matrix = jnp.diag(jnp.array([-1.0, -2.0]))
    input_matrix = jnp.array([[1.0], [0.0]])
    output_matrix = jnp.array([[0.0, 1.0]])
    controllability = continuous_controllability_gramian(continuous_matrix, input_matrix)
    observability = continuous_observability_gramian(continuous_matrix, output_matrix)
    np.testing.assert_allclose(controllability.value, [[0.5, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(observability.value, [[0.0, 0.0], [0.0, 0.25]])
    for result in (controllability, observability):
        assert bool(result.diagnostics.positive_semidefinite)
        assert bool(result.diagnostics.singular)
        assert int(result.diagnostics.rank) == 1
        assert np.isinf(float(result.diagnostics.gramian_condition_number))
        assert bool(result.diagnostics.converged)

    discrete_matrix = jnp.diag(jnp.array([0.5, 0.25]))
    discrete = discrete_observability_gramian(discrete_matrix, output_matrix)
    np.testing.assert_allclose(discrete.value, [[0.0, 0.0], [0.0, 16.0 / 15.0]])
    assert int(discrete.diagnostics.rank) == 1


def test_complex_lyapunov_and_gramian_use_conjugate_transposes():
    matrix = np.array(
        [[-1.0 + 0.4j, 0.2 - 0.1j], [0.0, -2.0 - 0.3j]],
        dtype=np.complex128,
    )
    source = np.array([[1.0, 0.2j], [-0.2j, 0.7]], dtype=np.complex128)
    result = solve_continuous_lyapunov(matrix, source, tolerance=1e-10)
    reference = scipy.linalg.solve_continuous_lyapunov(matrix, -source)
    np.testing.assert_allclose(result.value, reference, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        result.value, np.asarray(result.value).conj().T, atol=1e-12
    )

    input_matrix = jnp.array([[1.0 + 0.2j], [0.3 - 0.4j]])
    gramian = continuous_controllability_gramian(matrix, input_matrix)
    gramian_reference = scipy.linalg.solve_continuous_lyapunov(
        matrix, -np.asarray(input_matrix @ jnp.conj(input_matrix.T))
    )
    np.testing.assert_allclose(gramian.value, gramian_reference, rtol=3e-11, atol=3e-11)
    assert bool(gramian.diagnostics.positive_semidefinite)


def test_operator_krylov_lyapunov_solvers_match_dense_without_kronecker_matrices():
    continuous_matrix = jnp.array([[-1.0, 0.3], [-0.2, -2.0]])
    source = jnp.array([[1.0, 0.1], [0.1, 0.7]])
    continuous_dense = solve_continuous_lyapunov(continuous_matrix, source)
    continuous_action = solve_continuous_lyapunov_krylov(
        lambda vector: continuous_matrix @ vector,
        source,
        tolerance=1e-10,
        restart=4,
        max_steps=4,
    )
    np.testing.assert_allclose(
        continuous_action.value, continuous_dense.value, rtol=2e-8, atol=2e-8
    )
    assert bool(continuous_action.diagnostics.converged)
    assert np.isnan(float(continuous_action.diagnostics.condition_number))

    discrete_matrix = jnp.array([[0.5, 0.1], [0.0, 0.2]])
    discrete_dense = solve_discrete_lyapunov(discrete_matrix, source)
    discrete_action = solve_discrete_lyapunov_krylov(
        lambda vector: discrete_matrix @ vector,
        source,
        tolerance=1e-10,
        restart=4,
        max_steps=4,
    )
    np.testing.assert_allclose(
        discrete_action.value, discrete_dense.value, rtol=2e-8, atol=2e-8
    )
    assert bool(discrete_action.diagnostics.converged)


def test_dense_and_matrix_free_gramian_actions_agree_for_all_four_variants():
    continuous_matrix = jnp.array([[-1.0, 0.2], [0.0, -2.0]])
    discrete_matrix = jnp.array([[0.5, 0.1], [0.0, 0.2]])
    input_matrix = jnp.array([[1.0], [0.5]])
    output_matrix = jnp.array([[1.0, -0.3]])
    vector = jnp.array([0.4, -0.7])
    horizon = 1.2
    steps = 6

    continuous_control = continuous_controllability_gramian(
        continuous_matrix, input_matrix, horizon=horizon
    )
    continuous_control_action = continuous_controllability_gramian_action(
        lambda value: continuous_matrix @ value,
        lambda value: continuous_matrix.T @ value,
        lambda value: input_matrix @ value,
        lambda value: input_matrix.T @ value,
        vector,
        horizon,
        quadrature_order=12,
        krylov_dimension=2,
    )
    np.testing.assert_allclose(
        continuous_control_action.value,
        continuous_control.value @ vector,
        rtol=2e-9,
        atol=2e-9,
    )

    continuous_observation = continuous_observability_gramian(
        continuous_matrix, output_matrix, horizon=horizon
    )
    continuous_observation_action = continuous_observability_gramian_action(
        lambda value: continuous_matrix @ value,
        lambda value: continuous_matrix.T @ value,
        lambda value: output_matrix @ value,
        lambda value: output_matrix.T @ value,
        vector,
        horizon,
        quadrature_order=12,
        krylov_dimension=2,
    )
    np.testing.assert_allclose(
        continuous_observation_action.value,
        continuous_observation.value @ vector,
        rtol=2e-9,
        atol=2e-9,
    )

    discrete_control = discrete_controllability_gramian(
        discrete_matrix, input_matrix, steps=steps
    )
    discrete_control_action = discrete_controllability_gramian_action(
        lambda value: discrete_matrix @ value,
        lambda value: discrete_matrix.T @ value,
        lambda value: input_matrix @ value,
        lambda value: input_matrix.T @ value,
        vector,
        steps,
    )
    np.testing.assert_allclose(
        discrete_control_action.value, discrete_control.value @ vector, atol=1e-12
    )

    discrete_observation = discrete_observability_gramian(
        discrete_matrix, output_matrix, steps=steps
    )
    discrete_observation_action = discrete_observability_gramian_action(
        lambda value: discrete_matrix @ value,
        lambda value: discrete_matrix.T @ value,
        lambda value: output_matrix @ value,
        lambda value: output_matrix.T @ value,
        vector,
        steps,
    )
    np.testing.assert_allclose(
        discrete_observation_action.value,
        discrete_observation.value @ vector,
        atol=1e-12,
    )


def test_complex_matrix_free_gramian_actions_agree_with_dense_values():
    matrix = jnp.array([[-1.0 + 0.4j, 0.2 - 0.1j], [0.0, -2.0 - 0.3j]])
    input_matrix = jnp.array([[1.0 + 0.2j], [0.3 - 0.4j]])
    vector = jnp.array([0.4 - 0.1j, -0.7 + 0.3j])
    horizon = 0.8
    dense = continuous_controllability_gramian(matrix, input_matrix, horizon=horizon)
    action = continuous_controllability_gramian_action(
        lambda value: matrix @ value,
        lambda value: jnp.conj(matrix.T) @ value,
        lambda value: input_matrix @ value,
        lambda value: jnp.conj(input_matrix.T) @ value,
        vector,
        horizon,
        quadrature_order=12,
        krylov_dimension=2,
    )
    np.testing.assert_allclose(action.value, dense.value @ vector, rtol=2e-9, atol=2e-9)


def test_dense_dimension_limits_are_explicit_and_actions_remain_available():
    matrix = jnp.diag(jnp.array([-1.0, -2.0, -3.0]))
    source = jnp.eye(3)
    with pytest.raises(ValueError, match="max_dimension=2"):
        solve_continuous_lyapunov(matrix, source, max_dimension=2)
    with pytest.raises(ValueError, match="max_dimension=2"):
        continuous_controllability_gramian(matrix, source, max_dimension=2)

    action = solve_continuous_lyapunov_krylov(
        lambda vector: matrix @ vector,
        source,
        tolerance=1e-8,
        restart=9,
        max_steps=4,
    )
    assert action.value.shape == (3, 3)
    assert bool(action.diagnostics.converged)


def test_lyapunov_primitives_are_jittable_and_have_implicit_gradients():
    source = jnp.ones((1, 1))
    continuous_jit = jax.jit(continuous_lyapunov_solution)
    discrete_jit = jax.jit(discrete_lyapunov_solution)
    np.testing.assert_allclose(continuous_jit(jnp.array([[-2.0]]), source), [[0.25]])
    np.testing.assert_allclose(
        discrete_jit(jnp.array([[0.4]]), source), [[1.0 / (1.0 - 0.4**2)]]
    )

    continuous_gradient = jax.grad(
        lambda rate: continuous_lyapunov_solution(rate.reshape(1, 1), source)[0, 0]
    )(jnp.asarray(-2.0))
    discrete_gradient = jax.grad(
        lambda rate: discrete_lyapunov_solution(rate.reshape(1, 1), source)[0, 0]
    )(jnp.asarray(0.4))
    np.testing.assert_allclose(continuous_gradient, 1.0 / 8.0, rtol=2e-9)
    np.testing.assert_allclose(
        discrete_gradient,
        2.0 * 0.4 / (1.0 - 0.4**2) ** 2,
        rtol=2e-9,
    )

    finite_value = jax.jit(
        lambda rate: (
            continuous_controllability_gramian(
                rate.reshape(1, 1), source, horizon=1.0
            ).value
        )
    )(jnp.asarray(-1.0))
    np.testing.assert_allclose(finite_value, [[(1.0 - np.exp(-2.0)) / 2.0]])
