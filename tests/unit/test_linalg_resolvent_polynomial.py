import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_resolvent_scan_matches_normal_distance_and_marks_singular_shift():
    operator = phx.linalg.DenseLinearOperator(jnp.diag(jnp.asarray([1.0, 3.0])))
    result = phx.linalg.eigen.resolvent_scan(
        phx.linalg.eigen.ResolventScanProblem(
            operator,
            jnp.asarray([0.0 + 0.0j, 2.0 + 0.0j, 1.0 + 0.0j]),
        )
    )

    np.testing.assert_allclose(result.minimum_singular_values, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(result.resolvent_norms[:2], [1.0, 1.0])
    assert jnp.isinf(result.resolvent_norms[2])
    assert bool(result.diagnostics.singular_mask[2])
    assert bool(result.successful)


def test_nonnormal_resolvent_exceeds_inverse_eigenvalue_distance():
    operator = phx.linalg.DenseLinearOperator(jnp.asarray([[0.0, 20.0], [0.0, 0.0]]))
    result = phx.linalg.eigen.resolvent_scan(
        phx.linalg.eigen.ResolventScanProblem(operator, jnp.asarray([1.0 + 0.0j]))
    )

    assert result.resolvent_norms[0] > 10.0


def test_quadratic_eigenproblem_returns_original_residual_certificates():
    constant = phx.linalg.DenseLinearOperator(jnp.asarray([[-1.0]]))
    linear = phx.linalg.DenseLinearOperator(jnp.asarray([[0.0]]))
    quadratic = phx.linalg.DenseLinearOperator(jnp.asarray([[1.0]]))
    result = phx.linalg.eigen.polynomial_eigensolve(
        phx.linalg.eigen.PolynomialEigenproblem((constant, linear, quadratic))
    )

    np.testing.assert_allclose(jnp.sort(jnp.real(result.eigenvalues)), [-1.0, 1.0])
    assert jnp.max(result.diagnostics.original_relative_residuals) < 1e-12
    assert jnp.all(result.diagnostics.converged_mask)
    assert bool(result.successful)


def test_singular_leading_polynomial_reports_infinite_mode():
    constant = phx.linalg.DenseLinearOperator(jnp.asarray([[-2.0]]))
    linear = phx.linalg.DenseLinearOperator(jnp.asarray([[1.0]]))
    quadratic = phx.linalg.DenseLinearOperator(jnp.asarray([[0.0]]))
    result = phx.linalg.eigen.polynomial_eigensolve(
        phx.linalg.eigen.PolynomialEigenproblem((constant, linear, quadratic))
    )

    assert jnp.count_nonzero(result.diagnostics.finite_mask) == 1
    assert jnp.count_nonzero(result.diagnostics.infinite_mask) == 1
    finite = result.eigenvalues[result.diagnostics.finite_mask]
    np.testing.assert_allclose(finite, [2.0], atol=1e-12)
    assert jnp.max(result.diagnostics.original_relative_residuals) < 1e-12
    infinite_index = int(np.flatnonzero(np.asarray(result.diagnostics.infinite_mask))[0])
    assert result.diagnostics.right_extraction_blocks[infinite_index] == 1
    np.testing.assert_allclose(
        jnp.linalg.norm(result.right_eigenvectors[:, infinite_index]),
        1.0,
        atol=1e-12,
    )
    assert result.diagnostics.right_extraction_norms[infinite_index] > 0.0


def test_resolvent_refresh_preserves_identity_and_rejects_new_operator_identity():
    shifts = jnp.asarray([0.0 + 0.0j])

    def problem(diagonal, operator_id):
        return phx.linalg.eigen.ResolventScanProblem(
            phx.linalg.DenseLinearOperator(
                jnp.diag(jnp.asarray(diagonal)),
                operator_id=operator_id,
            ),
            shifts,
            problem_id="resolvent-refresh-problem",
        )

    prepared = phx.linalg.eigen.prepare_resolvent_scan(
        problem([1.0, 3.0], "resolvent-refresh-operator")
    )
    refreshed = phx.linalg.eigen.refresh_resolvent_scan(
        prepared,
        problem([2.0, 4.0], "resolvent-refresh-operator"),
    )
    result = phx.linalg.eigen.resolvent_scan(refreshed)

    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == 1
    np.testing.assert_allclose(result.resolvent_norms, [0.5], atol=1e-12)
    with pytest.raises(ValueError, match="symbolic plan"):
        phx.linalg.eigen.refresh_resolvent_scan(
            prepared,
            problem([2.0, 4.0], "different-resolvent-operator"),
        )


def test_polynomial_refresh_preserves_identity_and_rejects_new_coefficients():
    def problem(constant, *, constant_id="polynomial-constant"):
        return phx.linalg.eigen.PolynomialEigenproblem(
            (
                phx.linalg.DenseLinearOperator(
                    jnp.asarray([[constant]]),
                    operator_id=constant_id,
                ),
                phx.linalg.DenseLinearOperator(
                    jnp.asarray([[0.0]]),
                    operator_id="polynomial-linear",
                ),
                phx.linalg.DenseLinearOperator(
                    jnp.asarray([[1.0]]),
                    operator_id="polynomial-quadratic",
                ),
            ),
            problem_id="polynomial-refresh-problem",
        )

    prepared = phx.linalg.eigen.prepare_polynomial_eigensolve(problem(-1.0))
    refreshed = phx.linalg.eigen.refresh_polynomial_eigensolve(
        prepared,
        problem(-4.0),
    )
    result = phx.linalg.eigen.polynomial_eigensolve(refreshed)

    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == 1
    np.testing.assert_allclose(
        jnp.sort(jnp.real(result.eigenvalues)),
        [-2.0, 2.0],
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="incompatible"):
        phx.linalg.eigen.refresh_polynomial_eigensolve(
            prepared,
            problem(-4.0, constant_id="different-polynomial-constant"),
        )


def test_generalized_pencil_pseudospectrum_is_projective_and_handles_infinity():
    operator = phx.linalg.DenseLinearOperator(
        jnp.diag(jnp.asarray([2.0, 5.0], dtype=jnp.complex128))
    )
    mass = phx.linalg.DenseLinearOperator(
        jnp.diag(jnp.asarray([1.0, 2.0], dtype=jnp.complex128))
    )
    eigenproblem = phx.linalg.eigen.GeneralEigenproblem(operator, mass)
    norm = phx.linalg.eigen.PencilPerturbationNorm(2.0, 3.0)
    shifts = jnp.asarray([[2.0, 1.0], [1.0, 0.0], [6.0, 3.0]])
    result = phx.linalg.eigen.pencil_pseudospectrum(
        phx.linalg.eigen.PencilPseudospectrumProblem(eigenproblem, shifts, norm)
    )

    np.testing.assert_allclose(result.backward_errors[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.backward_errors[0], result.backward_errors[2], atol=1e-12
    )
    np.testing.assert_allclose(
        result.minimum_singular_values[1] / 3.0,
        result.backward_errors[1],
        atol=1e-12,
    )
    assert bool(result.successful)
    assert result.diagnostics.decomposition_count == 1


def test_pencil_pseudospectrum_frozen_direction_and_invalid_norm_fail_closed():
    operator = phx.linalg.DenseLinearOperator(jnp.asarray([[2.0 + 0.0j]]))
    problem = phx.linalg.eigen.GeneralEigenproblem(operator)
    frozen = phx.linalg.eigen.PencilPseudospectrumProblem(
        problem,
        jnp.asarray([[0.0, 1.0]]),
        phx.linalg.eigen.PencilPerturbationNorm(0.0, 1.0),
    )
    result = phx.linalg.eigen.pencil_pseudospectrum(frozen)
    assert bool(jnp.isinf(result.backward_errors[0]))
    assert bool(result.diagnostics.frozen_direction_mask[0])
    with pytest.raises(ValueError, match="cannot both be zero"):
        phx.linalg.eigen.PencilPerturbationNorm(0.0, 0.0)
