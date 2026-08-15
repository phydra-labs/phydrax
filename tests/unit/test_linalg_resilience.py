#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _positive_definite_properties():
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )


def _status_policy(equilibration, *, refinement_steps=3):
    return la.ResilientSolvePolicy(
        la.LinearSolvePolicy(
            la.DenseLU(),
            failure=la.FailurePolicy("status"),
        ),
        equilibration=equilibration,
        refinement=la.RefinementPolicy(
            max_steps=refinement_steps,
            tolerance=la.TolerancePolicy(relative=2e-8, absolute=1e-12),
        ),
        failure=la.FailurePolicy("status"),
    )


def test_two_sided_scaled_operator_matches_dense_actions_and_congruence_claims():
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    base = la.DenseLinearOperator(matrix, properties=_positive_definite_properties())
    left = jnp.asarray([2.0, 0.5])
    right = jnp.asarray([0.25, 3.0])
    transformed = la.TwoSidedScaledLinearOperator(base, left, right)
    dense = left[:, None] * matrix * right[None, :]
    vector = jnp.asarray([1.0, -2.0])

    assert jnp.allclose(transformed.mv(vector), dense @ vector)
    assert jnp.allclose(transformed.transpose_mv(vector), dense.T @ vector)
    assert jnp.allclose(transformed.adjoint_mv(vector), dense.T @ vector)
    assert jnp.allclose(la.assemble_diagonal(transformed), jnp.diag(dense))
    assert not transformed.properties.certifies("self_adjoint")
    cost = la.estimate_operator_action_cost(transformed)
    assert cost.exact
    assert cost.operation_class == "two-sided-scaled-action"

    congruence = la.TwoSidedScaledLinearOperator(
        base,
        left,
        congruence=True,
    )
    assert congruence.properties.certifies("self_adjoint")
    assert congruence.properties.certifies("positive_definite")
    assert jnp.allclose(
        la.materialize(congruence, la.MaterializationPolicy(max_entries=4)),
        left[:, None] * matrix * left[None, :],
    )


def test_ruiz_equilibration_reduces_condition_and_refines_original_residual():
    matrix = jnp.asarray([[1e-6, 2e-3], [3e2, 4e6]])
    operator = la.DenseLinearOperator(matrix, operator_id="ill-scaled-system")
    problem = la.LinearSystem(operator, problem_id="ill-scaled-problem")
    policy = _status_policy(
        la.EquilibrationPolicy(
            "ruiz",
            max_steps=12,
            tolerance=5e-2,
            diagnose_condition=True,
        )
    )
    plan = la.plan_resilient_solve(problem, policy)
    prepared = la.prepare_resilient_solve(problem, plan)
    rhs = jnp.asarray([[1.0, 2.0], [2.0, -1.0]])
    layout = la.RHSLayout((2,))

    result = la.solve_resilient(prepared, rhs, rhs_layout=layout)
    compiled = jax.jit(la.solve_resilient)(prepared, rhs, rhs_layout=layout)
    expected = jnp.linalg.solve(matrix, rhs)

    assert prepared.transform.converged
    assert prepared.transform.row_spread < 1.1
    assert prepared.transform.column_spread < 1.1
    assert prepared.condition_after < prepared.condition_before
    assert jnp.all(result.successful)
    assert jnp.all(
        result.diagnostics.residual_norm <= result.diagnostics.initial_residual_norm
    )
    assert jnp.allclose(result.value, expected, rtol=1e-9, atol=1e-9)
    assert jnp.allclose(compiled.value, expected, rtol=1e-9, atol=1e-9)
    assert result.provenance.plan_id == plan.plan_id

    updated_matrix = jnp.asarray([[2e-6, 1e-3], [5e2, 3e6]])
    updated_problem = la.LinearSystem(
        la.DenseLinearOperator(updated_matrix, operator_id="ill-scaled-system"),
        problem_id="ill-scaled-problem",
    )
    refreshed = la.refresh_resilient_solve(prepared, updated_problem)
    refreshed_result = la.solve_resilient(refreshed, rhs, rhs_layout=layout)

    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.numeric_version == 1
    assert refreshed.base_prepared.numeric_version == 1
    assert jnp.allclose(
        refreshed_result.value,
        jnp.linalg.solve(updated_matrix, rhs),
        rtol=1e-9,
        atol=1e-9,
    )


def test_symmetric_ruiz_preserves_certificates_for_native_pcg():
    matrix = jnp.asarray([[1e-6, 1e-3], [1e-3, 2e3]])
    operator = la.DenseLinearOperator(
        matrix,
        properties=_positive_definite_properties(),
    )
    problem = la.LinearSystem(operator)
    policy = la.ResilientSolvePolicy(
        la.LinearSolvePolicy(
            la.PCG(),
            tolerance=la.TolerancePolicy(relative=1e-12, max_steps=10),
            failure=la.FailurePolicy("status"),
        ),
        equilibration=la.EquilibrationPolicy("symmetric-ruiz", max_steps=10),
        failure=la.FailurePolicy("status"),
    )

    prepared = la.prepare_resilient_solve(problem, policy)
    result = la.solve_resilient(prepared, jnp.asarray([1.0, 2.0]))

    assert prepared.transformed_operator.congruence
    assert prepared.transformed_operator.properties.certifies("self_adjoint")
    assert prepared.transformed_operator.properties.certifies("positive_definite")
    assert prepared.base_prepared.plan.method == "pcg"
    assert result.successful
    assert jnp.allclose(result.value, jnp.linalg.solve(matrix, jnp.asarray([1.0, 2.0])))


def test_explicit_equilibration_supports_a_matrix_free_base():
    matrix = jnp.asarray([[4.0, 1.0], [2.0, 3.0]])
    space = la.ArraySpace((2,), dtype=jnp.float64)
    operator = la.FunctionLinearOperator(
        lambda value: matrix @ value,
        source=space,
        target=space,
        transpose_action=lambda value: matrix.T @ value,
    )
    problem = la.LinearSystem(operator)
    policy = la.ResilientSolvePolicy(
        la.LinearSolvePolicy(
            la.GMRES(),
            tolerance=la.TolerancePolicy(relative=1e-12, max_steps=10),
            failure=la.FailurePolicy("status"),
        ),
        equilibration=la.EquilibrationPolicy(
            "explicit",
            left_scale=jnp.asarray([0.25, 0.5]),
            right_scale=jnp.asarray([2.0, 0.5]),
        ),
        failure=la.FailurePolicy("status"),
    )
    rhs = jnp.asarray([1.0, -2.0])

    prepared = la.prepare_resilient_solve(problem, policy)
    result = la.solve_resilient(prepared, rhs)

    assert prepared.plan.cost.materialization_bytes == 0
    assert jnp.isnan(prepared.condition_before)
    assert prepared.base_prepared.plan.backend == "native-krylov"
    assert result.successful
    assert jnp.allclose(result.value, jnp.linalg.solve(matrix, rhs), atol=1e-11)


def test_resilient_one_shot_solve_has_the_dense_mathematical_derivative():
    rhs = jnp.asarray([1.0, -2.0])
    policy = la.ResilientSolvePolicy(
        la.LinearSolvePolicy(
            la.DenseLU(),
            differentiation=la.DifferentiationPolicy("mathematical"),
            failure=la.FailurePolicy("status"),
        ),
        refinement=la.RefinementPolicy(max_steps=0),
        failure=la.FailurePolicy("status"),
    )

    def resilient(diagonal):
        matrix = jnp.asarray([[diagonal[0], 0.25], [0.5, diagonal[1]]])
        problem = la.LinearSystem(la.DenseLinearOperator(matrix))
        return jnp.sum(la.solve_resilient(problem, rhs, policy).value)

    def dense(diagonal):
        matrix = jnp.asarray([[diagonal[0], 0.25], [0.5, diagonal[1]]])
        return jnp.sum(jnp.linalg.solve(matrix, rhs))

    diagonal = jnp.asarray([3.0, 4.0])
    actual = jax.jit(jax.grad(resilient))(diagonal)
    expected = jax.grad(dense)(diagonal)

    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_resilient_planning_rejects_unavailable_materialization_and_workspace():
    matrix = jnp.eye(3)
    space = la.ArraySpace((3,), dtype=jnp.float64)
    matrix_free = la.FunctionLinearOperator(
        lambda value: matrix @ value,
        source=space,
        target=space,
        transpose_action=lambda value: matrix.T @ value,
    )
    with pytest.raises(ValueError, match="exceeds max_entries"):
        la.plan_resilient_solve(
            la.LinearSystem(matrix_free),
            _status_policy(
                la.EquilibrationPolicy(
                    "ruiz",
                    materialization=la.MaterializationPolicy(max_entries=1),
                )
            ),
        )
    problem = la.LinearSystem(la.DenseLinearOperator(matrix))
    policy = la.ResilientSolvePolicy(
        la.LinearSolvePolicy(la.DenseLU()),
        equilibration=la.EquilibrationPolicy("ruiz"),
        resources=la.ResilienceResourcePolicy(max_workspace_bytes=1),
    )
    with pytest.raises(ValueError, match="workspace exceeds"):
        la.plan_resilient_solve(problem, policy)
