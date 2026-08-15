#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest
import scipy.linalg as spla

import phydrax as phx


la = phx.linalg
eigen = la.eigen


def _self_adjoint_properties(*, positive_definite=False):
    evidence = {"self_adjoint": "construction"}
    if positive_definite:
        evidence.update(
            {
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            }
        )
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=positive_definite,
        evidence=evidence,
    )


def _standard_problem(matrix):
    return eigen.Eigenproblem(
        la.DenseLinearOperator(
            matrix,
            properties=_self_adjoint_properties(),
        )
    )


def _generalized_problem(operator_matrix, metric_matrix):
    return eigen.GeneralizedEigenproblem(
        la.DenseLinearOperator(
            operator_matrix,
            properties=_self_adjoint_properties(),
        ),
        la.DenseLinearOperator(
            metric_matrix,
            properties=_self_adjoint_properties(positive_definite=True),
        ),
    )


def test_polynomial_spectral_operator_matches_direct_matrix_polynomial_and_reuses_spectrum():
    matrix = jnp.asarray(
        [
            [2.0, 0.4, 0.0, 0.0],
            [0.4, 3.0, 0.2, 0.0],
            [0.0, 0.2, 5.0, 0.3],
            [0.0, 0.0, 0.3, 7.0],
        ]
    )
    coefficients = jnp.asarray([1.5, -0.4, 0.25, 0.1])
    prepared = eigen.prepare_self_adjoint_spectrum(_standard_problem(matrix))
    function = eigen.PolynomialSpectralFunction(coefficients)
    result = eigen.self_adjoint_spectral_operator(prepared, function)
    identity = jnp.eye(4)
    reference = (
        coefficients[0] * identity
        + coefficients[1] * matrix
        + coefficients[2] * matrix @ matrix
        + coefficients[3] * matrix @ matrix @ matrix
    )

    assert bool(result.successful)
    assert jnp.allclose(result.operator, reference, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(result.density_kernel, reference, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(result.trace, jnp.trace(reference), atol=1e-11)
    assert result.diagnostics.reconstruction_residual < 1e-10
    assert result.provenance.spectrum_plan_id == prepared.plan.plan_id


def test_fermi_dirac_values_and_trainable_parameters_match_scalar_reference():
    diagonal = jnp.asarray([-2.0, -0.5, 1.0, 3.0])
    chemical_potential = jnp.asarray(0.3)
    temperature = jnp.asarray(0.7)
    function = eigen.FermiDiracSpectralFunction(
        chemical_potential,
        temperature,
    )
    result = eigen.self_adjoint_spectral_operator(
        _standard_problem(jnp.diag(diagonal)),
        function,
    )
    reference = jax.nn.sigmoid((chemical_potential - diagonal) / temperature)

    assert bool(result.successful)
    assert jnp.allclose(jnp.diag(result.operator), reference, atol=1e-12)
    assert jnp.allclose(result.diagnostics.function_values, reference, atol=1e-12)
    assert jnp.allclose(result.trace, jnp.sum(reference), atol=1e-12)
    policy = eigen.SelfAdjointSpectralOperatorPolicy(differentiation="frechet")

    def trace(current_chemical_potential, current_temperature):
        return eigen.self_adjoint_spectral_operator(
            _standard_problem(jnp.diag(diagonal)),
            eigen.FermiDiracSpectralFunction(
                current_chemical_potential,
                current_temperature,
            ),
            policy=policy,
        ).trace

    parameter_gradient = jax.jit(jax.grad(trace, argnums=(0, 1)))(
        chemical_potential,
        temperature,
    )
    reference_gradient = jax.grad(
        lambda current_chemical_potential, current_temperature: jnp.sum(
            jax.nn.sigmoid(
                (current_chemical_potential - diagonal) / current_temperature
            )
        ),
        argnums=(0, 1),
    )(chemical_potential, temperature)
    low_temperature = eigen.self_adjoint_spectral_operator(
        _standard_problem(jnp.diag(diagonal)),
        eigen.FermiDiracSpectralFunction(
            chemical_potential,
            jnp.asarray(1e-4),
        ),
    )

    assert jnp.allclose(parameter_gradient[0], reference_gradient[0], atol=1e-12)
    assert jnp.allclose(parameter_gradient[1], reference_gradient[1], atol=1e-12)
    assert jnp.array_equal(
        jnp.diag(low_temperature.operator),
        (diagonal < chemical_potential).astype(diagonal.dtype),
    )
    with pytest.raises(Exception, match="temperature"):
        eigen.FermiDiracSpectralFunction(jnp.asarray(0.0), jnp.asarray(0.0))


def test_loewner_derivative_is_finite_at_repeated_eigenvalues_and_matches_finite_difference():
    matrix = jnp.asarray(
        [
            [1.0, 0.0, 0.1, 0.0],
            [0.0, 1.0, -0.2, 0.0],
            [0.1, -0.2, 4.0, 0.3],
            [0.0, 0.0, 0.3, 7.0],
        ]
    )
    perturbation = jnp.asarray(
        [
            [0.2, 0.3, -0.1, 0.0],
            [0.3, -0.2, 0.2, 0.1],
            [-0.1, 0.2, 0.1, -0.3],
            [0.0, 0.1, -0.3, -0.1],
        ]
    )
    coefficients = jnp.asarray([0.5, -0.3, 0.2, 0.04])
    coefficient_tangent = jnp.asarray([0.1, -0.05, 0.03, -0.01])
    policy = eigen.SelfAdjointSpectralOperatorPolicy(differentiation="frechet")

    def operator(current_matrix, current_coefficients):
        return eigen.self_adjoint_spectral_operator(
            _standard_problem(current_matrix),
            eigen.PolynomialSpectralFunction(current_coefficients),
            policy=policy,
        ).operator

    primal, tangent = jax.jit(
        lambda current_matrix, current_coefficients, matrix_tangent, parameter_tangent: (
            jax.jvp(
                operator,
                (current_matrix, current_coefficients),
                (matrix_tangent, parameter_tangent),
            )
        )
    )(matrix, coefficients, perturbation, coefficient_tangent)
    step = 1e-5
    finite_difference = (
        operator(
            matrix + step * perturbation,
            coefficients + step * coefficient_tangent,
        )
        - operator(
            matrix - step * perturbation,
            coefficients - step * coefficient_tangent,
        )
    ) / (2 * step)
    cotangent = jnp.asarray(
        [
            [0.2, 0.1, -0.1, 0.0],
            [0.1, -0.3, 0.2, 0.1],
            [-0.1, 0.2, 0.4, -0.2],
            [0.0, 0.1, -0.2, 0.1],
        ]
    )
    reverse_matrix, reverse_coefficients = jax.grad(
        lambda current_matrix, current_coefficients: jnp.sum(
            operator(current_matrix, current_coefficients) * cotangent
        ),
        argnums=(0, 1),
    )(matrix, coefficients)

    assert jnp.all(jnp.isfinite(primal))
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.allclose(tangent, finite_difference, rtol=5e-6, atol=5e-7)
    assert jnp.allclose(
        jnp.sum(reverse_matrix * perturbation)
        + jnp.sum(reverse_coefficients * coefficient_tangent),
        jnp.sum(cotangent * tangent),
        rtol=1e-9,
        atol=1e-10,
    )


def test_generalized_loewner_derivative_and_density_include_metric_tangent():
    operator = jnp.diag(jnp.asarray([1.0, 4.0, 12.0, 28.0]))
    metric = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    operator_tangent = jnp.asarray(
        [
            [0.1, 0.2, 0.0, 0.0],
            [0.2, -0.1, 0.3, 0.0],
            [0.0, 0.3, 0.2, 0.1],
            [0.0, 0.0, 0.1, -0.2],
        ]
    )
    metric_tangent = jnp.asarray(
        [
            [0.02, -0.01, 0.0, 0.0],
            [-0.01, 0.03, 0.02, 0.0],
            [0.0, 0.02, -0.01, 0.01],
            [0.0, 0.0, 0.01, 0.02],
        ]
    )
    function = eigen.FermiDiracSpectralFunction(jnp.asarray(2.0), jnp.asarray(0.8))
    policy = eigen.SelfAdjointSpectralOperatorPolicy(differentiation="frechet")

    def outputs(current_operator, current_metric):
        result = eigen.self_adjoint_spectral_operator(
            _generalized_problem(current_operator, current_metric),
            function,
            policy=policy,
        )
        return result.operator, result.density_kernel

    _, tangent = jax.jvp(
        outputs,
        (operator, metric),
        (operator_tangent, metric_tangent),
    )
    step = 1e-5
    plus = outputs(
        operator + step * operator_tangent,
        metric + step * metric_tangent,
    )
    minus = outputs(
        operator - step * operator_tangent,
        metric - step * metric_tangent,
    )
    finite_difference = tuple(
        (upper - lower) / (2 * step) for upper, lower in zip(plus, minus, strict=True)
    )

    assert jnp.allclose(tangent[0], finite_difference[0], rtol=5e-6, atol=5e-7)
    assert jnp.allclose(tangent[1], finite_difference[1], rtol=5e-6, atol=5e-7)


def test_builtin_spectral_functions_match_scipy_references():
    matrix = jnp.asarray(
        [
            [2.0, 0.3, 0.0],
            [0.3, 3.0, 0.2],
            [0.0, 0.2, 5.0],
        ]
    )
    cases = (
        (eigen.ExponentialSpectralFunction(), spla.expm(matrix)),
        (eigen.LogarithmSpectralFunction(), spla.logm(matrix)),
        (eigen.SquareRootSpectralFunction(), spla.fractional_matrix_power(matrix, 0.5)),
        (
            eigen.InverseSquareRootSpectralFunction(),
            spla.fractional_matrix_power(matrix, -0.5),
        ),
        (
            eigen.FractionalPowerSpectralFunction(1.5),
            spla.fractional_matrix_power(matrix, 1.5),
        ),
        (
            eigen.ResolventSpectralFunction(jnp.asarray(-1.0)),
            jnp.linalg.inv(matrix + jnp.eye(3)),
        ),
    )

    for function, reference in cases:
        result = eigen.self_adjoint_spectral_operator(
            _standard_problem(matrix),
            function,
        )
        assert bool(result.successful)
        assert jnp.allclose(result.operator, reference, rtol=1e-10, atol=1e-10)


def test_invalid_spectral_domains_report_status_without_clipping():
    matrix = jnp.diag(jnp.asarray([-1.0, 1.0, 3.0]))
    logarithm = eigen.self_adjoint_spectral_operator(
        _standard_problem(matrix),
        eigen.LogarithmSpectralFunction(),
    )
    square_root = eigen.self_adjoint_spectral_operator(
        _standard_problem(matrix),
        eigen.SquareRootSpectralFunction(),
    )
    pole = eigen.self_adjoint_spectral_operator(
        _standard_problem(matrix),
        eigen.ResolventSpectralFunction(jnp.asarray(1.0)),
    )

    assert logarithm.status == int(eigen.SelfAdjointSpectralOperatorStatus.DOMAIN_ERROR)
    assert square_root.status == int(eigen.SelfAdjointSpectralOperatorStatus.DOMAIN_ERROR)
    assert pole.status == int(eigen.SelfAdjointSpectralOperatorStatus.DOMAIN_ERROR)
    assert not bool(logarithm.diagnostics.domain_valid)
    assert not bool(square_root.diagnostics.domain_valid)
    assert not bool(pole.diagnostics.domain_valid)


def test_batched_spectral_functions_preserve_batch_axes_mixed_status_and_loewner_derivatives():
    positive = jnp.asarray(
        [
            [[1.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 4.0]],
            [[2.0, -0.2, 0.1], [-0.2, 3.0, 0.0], [0.1, 0.0, 5.0]],
        ]
    )
    perturbation = jnp.asarray(
        [
            [[0.1, 0.2, 0.3], [0.2, -0.1, 0.1], [0.3, 0.1, 0.2]],
            [[0.2, 0.1, -0.2], [0.1, 0.3, 0.2], [-0.2, 0.2, -0.1]],
        ]
    )
    coefficients = jnp.asarray([0.5, -0.3, 0.2])
    coefficient_tangent = jnp.asarray([0.1, -0.05, 0.03])
    policy = eigen.SelfAdjointSpectralOperatorPolicy(differentiation="frechet")

    def outputs(matrices, polynomial_coefficients):
        problem = eigen.Eigenproblem(
            la.DenseLinearOperator(
                matrices,
                properties=_self_adjoint_properties(),
            )
        )
        result = eigen.self_adjoint_spectral_operator(
            problem,
            eigen.PolynomialSpectralFunction(polynomial_coefficients),
            policy=policy,
        )
        return result.operator, result.density_kernel, result.trace

    primal, tangent = jax.jit(
        lambda matrices, polynomial_coefficients, matrix_tangent, parameter_tangent: (
            jax.jvp(
                outputs,
                (matrices, polynomial_coefficients),
                (matrix_tangent, parameter_tangent),
            )
        )
    )(positive, coefficients, perturbation, coefficient_tangent)
    step = 1e-5
    plus = outputs(
        positive + step * perturbation,
        coefficients + step * coefficient_tangent,
    )
    minus = outputs(
        positive - step * perturbation,
        coefficients - step * coefficient_tangent,
    )
    finite_difference = tuple(
        (upper - lower) / (2 * step) for upper, lower in zip(plus, minus, strict=True)
    )

    assert primal[0].shape == (2, 3, 3)
    assert primal[1].shape == (2, 3, 3)
    assert primal[2].shape == (2,)
    assert jnp.allclose(tangent[0], finite_difference[0], rtol=5e-6, atol=5e-7)
    assert jnp.allclose(tangent[1], finite_difference[1], rtol=5e-6, atol=5e-7)
    assert jnp.allclose(tangent[2], finite_difference[2], rtol=5e-6, atol=5e-7)

    mixed_matrices = positive.at[1].set(
        jnp.diag(jnp.asarray([-1.0, 2.0, 4.0]))
    )
    mixed_problem = eigen.Eigenproblem(
        la.DenseLinearOperator(
            mixed_matrices,
            properties=_self_adjoint_properties(),
        )
    )
    mixed = eigen.self_adjoint_spectral_operator(
        mixed_problem,
        eigen.LogarithmSpectralFunction(),
    )

    assert mixed.status.shape == (2,)
    assert jnp.array_equal(
        mixed.status,
        jnp.asarray(
            [
                int(eigen.SelfAdjointSpectralOperatorStatus.SUCCESS),
                int(eigen.SelfAdjointSpectralOperatorStatus.DOMAIN_ERROR),
            ]
        ),
    )
    assert jnp.array_equal(
        mixed.diagnostics.domain_valid,
        jnp.asarray([True, False]),
    )
