#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


la = phx.linalg
eig = la.eigen


def _problem(matrix, *, operator_id="general-schur"):
    return eig.SchurEigenproblem(
        la.DenseLinearOperator(jnp.asarray(matrix), operator_id=operator_id)
    )


def test_complex_schur_reconstructs_nonnormal_operator_and_reports_spectrum():
    matrix = jnp.asarray(
        [
            [1.0, 8.0, 0.0],
            [-2.0, 1.0, 3.0],
            [0.0, 0.0, 4.0],
        ]
    )
    result = eig.schur_eigensolve(_problem(matrix))

    reconstructed = (
        result.schur_vectors @ result.schur_form @ jnp.conj(result.schur_vectors.T)
    )
    expected_eigenvalues = np.sort_complex(np.asarray(jnp.linalg.eigvals(matrix)))
    actual_eigenvalues = np.sort_complex(np.asarray(result.eigenvalues))

    assert result.status == int(eig.SchurSolveStatus.SUCCESS)
    assert result.successful
    assert jnp.allclose(reconstructed, matrix, rtol=1e-11, atol=1e-12)
    assert np.allclose(actual_eigenvalues, expected_eigenvalues, rtol=1e-11, atol=1e-12)
    assert result.diagnostics.departure_from_normality > 0.5
    assert result.provenance.ordering == "backend Schur order; no eigenvalue reordering"


def test_schur_relation_and_column_diagnostics_hold_for_complex_operator():
    matrix = jnp.asarray(
        [
            [1.0 + 2.0j, 3.0 - 1.0j, 0.5j],
            [0.0, -2.0 + 0.25j, 4.0],
            [1.0j, 0.0, 0.5 - 1.0j],
        ]
    )
    result = eig.schur_eigensolve(_problem(matrix, operator_id="complex-schur"))
    relation = matrix @ result.schur_vectors - result.schur_vectors @ result.schur_form

    assert result.status == int(eig.SchurSolveStatus.SUCCESS)
    assert jnp.allclose(
        result.diagnostics.column_residual_norms,
        jnp.linalg.norm(relation, axis=0),
    )
    assert result.diagnostics.residual_norm == pytest.approx(
        float(jnp.linalg.norm(relation))
    )
    assert result.diagnostics.unitarity_error < 1e-12


def test_defective_jordan_block_preserves_schur_semantics_without_eigenvectors():
    matrix = jnp.asarray(
        [
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 2.0],
        ]
    )
    result = eig.schur_eigensolve(_problem(matrix, operator_id="jordan-schur"))

    assert result.status == int(eig.SchurSolveStatus.SUCCESS)
    assert jnp.allclose(result.eigenvalues, 2.0)
    assert jnp.allclose(result.diagnostics.eigenvalue_separation, 0.0)
    assert result.diagnostics.departure_from_normality > 0.0
    assert result.schur_vectors.shape == (3, 3)


def test_prepared_schur_is_jittable_and_refreshes_under_same_symbolic_plan():
    first_matrix = jnp.asarray([[1.0, 3.0], [-2.0, 4.0]])
    second_matrix = jnp.asarray([[2.0, -1.0], [5.0, 3.0]])
    first_problem = _problem(first_matrix, operator_id="refreshable-schur")
    second_problem = _problem(second_matrix, operator_id="refreshable-schur")
    plan = eig.plan_schur_eigensolve(first_problem)
    prepared = eig.prepare_schur_eigensolve(first_problem, plan)

    compiled = jax.jit(eig.schur_eigensolve)
    first = compiled(prepared)
    refreshed = eig.refresh_schur_eigensolve(prepared, second_problem)
    second = compiled(refreshed)

    assert first.status == int(eig.SchurSolveStatus.SUCCESS)
    assert second.status == int(eig.SchurSolveStatus.SUCCESS)
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == 1
    assert refreshed.refresh_count == 1
    assert second.provenance.numeric_version == 1
    assert np.allclose(
        np.sort_complex(np.asarray(second.eigenvalues)),
        np.sort_complex(np.asarray(jnp.linalg.eigvals(second_matrix))),
    )


def test_schur_planning_enforces_materialization_and_resource_budgets():
    problem = _problem(jnp.eye(4), operator_id="budgeted-schur")
    materialization_policy = eig.SchurSolvePolicy(
        materialization=la.MaterializationPolicy(max_entries=15)
    )
    prepared = eig.prepare_schur_eigensolve
    with pytest.raises(la.LinearCapabilityError, match="exceeding"):
        prepared(problem, materialization_policy)

    resource_policy = eig.SchurSolvePolicy(
        resources=eig.SchurResourcePolicy(preparation_bytes=127)
    )
    with pytest.raises(ValueError, match="preparation estimate"):
        eig.plan_schur_eigensolve(problem, resource_policy)


def test_schur_status_reports_an_unsatisfied_zero_tolerance():
    matrix = jnp.asarray(
        [
            [1.0, 8.0, 0.0],
            [-2.0, 1.0, 3.0],
            [0.0, 0.0, 4.0],
        ]
    )
    policy = eig.SchurSolvePolicy(
        tolerance=eig.SchurTolerancePolicy(relative=0.0, absolute=0.0)
    )
    result = eig.schur_eigensolve(
        _problem(matrix, operator_id="strict-schur"), policy=policy
    )

    assert result.status == int(eig.SchurSolveStatus.RESIDUAL_TOLERANCE_NOT_MET)
    assert not result.successful
    assert not result.diagnostics.converged


def test_schur_rejects_nonfinite_inputs_and_incompatible_refreshes():
    with pytest.raises(ValueError, match="finite"):
        eig.prepare_schur_eigensolve(
            _problem(jnp.asarray([[1.0, jnp.nan], [0.0, 2.0]]), operator_id="nan-schur")
        )

    prepared = eig.prepare_schur_eigensolve(
        _problem(jnp.eye(2), operator_id="first-schur")
    )
    with pytest.raises(ValueError, match="different symbolic"):
        eig.refresh_schur_eigensolve(
            prepared,
            _problem(2.0 * jnp.eye(2), operator_id="second-schur"),
        )
