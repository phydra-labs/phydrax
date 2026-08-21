#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


class _NoMaterializeStableOperator(la.AbstractLinearOperator):
    matrix: jax.Array

    def __init__(self, matrix, /, *, operator_id="matrix-free-stable"):
        matrix = jnp.asarray(matrix)
        space = la.ArraySpace((matrix.shape[0],), dtype=matrix.dtype)
        self.matrix = matrix
        self.source = space
        self.target = space
        self.properties = la.OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "verified"},
        )
        self.capabilities = la.OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    def mv(self, vector, /):
        return self.matrix @ vector

    def transpose_mv(self, vector, /):
        return self.matrix.T @ vector

    def adjoint_mv(self, vector, /):
        return jnp.conj(self.matrix.T) @ vector

    def _materialize(self, /):
        raise AssertionError("The factored path must not materialize the operator.")


def _problem(*, problem_id="factored-reference"):
    matrix = jnp.diag(jnp.asarray([-1.0, -2.0, -4.0, -8.0]))
    source_factor = jnp.asarray([[1.0], [0.75], [-0.5], [0.25]])
    operator = _NoMaterializeStableOperator(matrix, operator_id=problem_id)
    return (
        matrix,
        source_factor,
        la.factored_continuous_lyapunov_equation(
            operator,
            source_factor,
            problem_id=problem_id,
        ),
    )


def _policy(*, maximum_rank=None, residual_tolerance=1e-8, max_dimension=4):
    return la.FactoredMatrixEquationPolicy(
        (-1.0, -2.0, -4.0, -8.0),
        shifted=la.ShiftedSolvePolicy(
            "lanczos",
            max_dimension=max_dimension,
            relative_tolerance=1e-11,
            absolute_tolerance=1e-12,
        ),
        relative_truncation_tolerance=0.0,
        absolute_truncation_tolerance=0.0,
        maximum_rank=maximum_rank,
        relative_residual_tolerance=residual_tolerance,
        absolute_residual_tolerance=1e-12,
    )


def test_factored_solution_rank_masks_unused_fixed_capacity_columns():
    left = jnp.asarray([[1.0, 100.0], [2.0, 200.0]])
    right = jnp.asarray([[3.0, 300.0], [4.0, 400.0], [5.0, 500.0]])
    general = la.FactoredMatrixSolution(left, right, rank=1)
    psd = la.FactoredMatrixSolution(
        left,
        rank=1,
        hermitian_positive_semidefinite=True,
    )

    assert jnp.allclose(general.to_dense(), jnp.outer(left[:, 0], right[:, 0]))
    assert jnp.allclose(psd.to_dense(), jnp.outer(left[:, 0], left[:, 0]))


def test_matrix_free_factored_lyapunov_matches_existing_exact_reference_without_materializing():
    matrix, source_factor, problem = _problem()
    result = la.solve_factored_matrix_equation(problem, policy=_policy())
    forcing = source_factor @ source_factor.T
    exact = la.solve_matrix_equation(
        la.continuous_lyapunov_equation(matrix, forcing),
        policy=la.MatrixEquationPolicy(
            linear=la.LinearSolvePolicy(la.DenseLU()),
        ),
    )
    reconstructed = result.solution.to_dense()
    residual = matrix @ reconstructed + reconstructed @ matrix.T + forcing

    assert result.status == int(la.FactoredMatrixEquationStatus.SUCCESS)
    assert result.successful
    assert result.solution.hermitian_positive_semidefinite
    assert jnp.allclose(reconstructed, exact.value, rtol=2e-8, atol=2e-9)
    assert jnp.linalg.norm(residual) < 2e-8
    assert jnp.all(result.shifted_statuses == int(la.ShiftedSolveStatus.SUCCESS))
    assert jnp.allclose(
        result.certificate.residual_norm,
        jnp.linalg.norm(residual),
        rtol=2e-7,
        atol=2e-10,
    )
    assert result.certificate.exact
    assert result.provenance.operator_materialized is False
    assert result.provenance.solution_materialized is False


def test_factored_truncation_reports_rank_loss_storage_and_original_residual_certificate():
    matrix, source_factor, problem = _problem(problem_id="factored-truncated")
    result = la.solve_factored_matrix_equation(
        problem,
        policy=_policy(maximum_rank=2, residual_tolerance=1.0),
    )
    reconstructed = result.solution.to_dense()
    forcing = source_factor @ source_factor.T
    residual = matrix @ reconstructed + reconstructed @ matrix.T + forcing
    expected_allocated_bytes = (
        problem.dimension
        * len(result.provenance.shifts)
        * problem.source_rank
        * result.solution.factor.dtype.itemsize
    )
    expected_retained_bytes = (
        problem.dimension * 2 * result.solution.factor.dtype.itemsize
    )

    assert result.rank == 2
    assert result.diagnostics.raw_rank >= result.rank
    assert result.truncation_loss > 0.0
    assert result.diagnostics.relative_truncation_loss > 0.0
    assert result.solution.factor.shape == (problem.dimension, 4)
    assert result.cost.raw_rank_capacity == 4
    assert result.cost.factor_storage_bytes == expected_allocated_bytes
    assert result.diagnostics.factor_storage_bytes == expected_allocated_bytes
    assert result.diagnostics.retained_factor_storage_bytes == expected_retained_bytes
    assert result.cost.explicit_solution_bytes == (
        problem.dimension**2 * result.solution.factor.dtype.itemsize
    )
    assert jnp.allclose(
        result.certificate.residual_norm,
        jnp.linalg.norm(residual),
        rtol=2e-7,
        atol=2e-10,
    )


def test_general_factored_solution_contract_reconstructs_u_v_adjoint():
    left = jnp.asarray([[1.0, 0.5], [-2.0, 1.0], [0.25, -0.75]])
    right = jnp.asarray([[1.0 + 0.5j, -1.0j], [0.25, 2.0], [-0.5j, 0.75], [1.5, -0.25j]])
    solution = la.FactoredMatrixSolution(left, right)

    assert solution.form == "general"
    assert not solution.hermitian_positive_semidefinite
    assert solution.rank == 2
    assert jnp.allclose(solution.to_dense(), left @ jnp.conj(right.T))


def test_shifted_failure_propagates_to_factored_status_and_per_shift_evidence():
    _matrix, _source_factor, problem = _problem(problem_id="factored-failure")
    failed = la.solve_factored_matrix_equation(
        problem,
        policy=_policy(max_dimension=1, residual_tolerance=1.0),
    )

    assert failed.status == int(la.FactoredMatrixEquationStatus.SHIFTED_SOLVE_FAILURE)
    assert not failed.successful
    assert jnp.any(failed.shifted_statuses != int(la.ShiftedSolveStatus.SUCCESS))
    assert failed.shifted_statuses.shape == (4, 1)
    assert not failed.diagnostics.converged


def test_factored_public_lifecycle_refreshes_and_unsupported_dense_structures_are_rejected():
    matrix, source_factor, first = _problem(problem_id="factored-refresh")
    policy = _policy()
    plan = la.plan_factored_matrix_equation(first, policy)
    prepared = la.prepare_factored_matrix_equation(first, plan)
    second = la.factored_continuous_lyapunov_equation(
        _NoMaterializeStableOperator(
            1.1 * matrix,
            operator_id="factored-refresh",
        ),
        0.5 * source_factor,
        problem_id="factored-refresh",
    )
    refreshed = la.refresh_factored_matrix_equation(prepared, second)

    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == 1
    refreshed_result = la.solve_factored_matrix_equation(refreshed)
    jitted_result = jax.jit(la.solve_factored_matrix_equation)(refreshed)
    assert refreshed_result.provenance.numeric_version == 1
    assert jitted_result.status == refreshed_result.status
    assert jnp.allclose(
        jitted_result.solution.to_dense(),
        refreshed_result.solution.to_dense(),
    )

    dense_sylvester = la.sylvester_equation(jnp.eye(2), jnp.eye(2), jnp.eye(2))
    with pytest.raises(NotImplementedError, match="no dense fallback|not converted"):
        la.solve_factored_matrix_equation(dense_sylvester, policy=policy)

    public = (
        "FactoredMatrixEquationProblem",
        "FactoredMatrixSolution",
        "FactoredMatrixEquationPolicy",
        "FactoredMatrixEquationPlan",
        "PreparedFactoredMatrixEquation",
        "FactoredMatrixEquationResult",
        "factored_continuous_lyapunov_equation",
        "plan_factored_matrix_equation",
        "prepare_factored_matrix_equation",
        "refresh_factored_matrix_equation",
        "solve_factored_matrix_equation",
    )
    assert all(name in la.__all__ for name in public)
