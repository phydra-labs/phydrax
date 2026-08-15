#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _dense_rational_action(matrix, vector, function):
    value = jnp.zeros_like(
        vector,
        dtype=jnp.result_type(matrix, vector, function.poles),
    )
    power = vector
    for index, coefficient in enumerate(function.polynomial_coefficients):
        if index:
            power = matrix @ power
        value = value + coefficient * power
    identity = jnp.eye(matrix.shape[0], dtype=value.dtype)
    for pole, residue in zip(function.poles, function.residues, strict=True):
        value = value + residue * jnp.linalg.solve(pole * identity - matrix, vector)
    return value


def test_partial_fraction_action_matches_dense_polynomial_and_resolvents():
    matrix = jnp.asarray(
        [
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 4.0],
        ]
    )
    vector = jnp.asarray([1.0, -2.0, 0.5])
    function = la.PartialFractionRationalFunction(
        jnp.asarray([6.0 + 0.5j, 7.0 - 0.25j]),
        jnp.asarray([1.5 - 0.2j, -0.25 + 0.1j]),
        polynomial_coefficients=jnp.asarray([0.3, -0.2, 0.1]),
    )
    operator = la.DenseLinearOperator(matrix, operator_id="rational-dense")
    policy = la.RationalFunctionPolicy(shifted=la.ShiftedSolvePolicy(max_dimension=3))

    result = la.rational_function_action(operator, vector, function, policy=policy)
    expected = _dense_rational_action(matrix, vector, function)

    assert result.status == int(la.RationalFunctionStatus.SUCCESS)
    assert result.successful
    assert jnp.allclose(result.value, expected, rtol=1e-11, atol=1e-12)
    assert result.diagnostics.polynomial_matvec_count == 2
    assert result.diagnostics.solve_matvec_count == 0
    assert result.diagnostics.residual_indicator < 1e-12
    assert "r_j/(p_j-z)" in result.provenance.convention


def test_partial_fraction_scalar_evaluation_uses_pole_minus_argument_convention():
    function = la.PartialFractionRationalFunction(
        jnp.asarray([3.0, 5.0]),
        jnp.asarray([2.0, -1.0]),
        polynomial_coefficients=jnp.asarray([1.0, 0.5]),
    )
    values = jnp.asarray([-1.0, 0.25, 2.0])
    expected = 1.0 + 0.5 * values + 2.0 / (3.0 - values) - 1.0 / (5.0 - values)

    assert jnp.allclose(function(values), expected)


def test_prepared_rational_action_is_jittable_and_refreshes_numeric_state():
    first_matrix = jnp.asarray([[2.0, 0.5], [0.5, 3.0]])
    second_matrix = jnp.asarray([[1.5, -0.25], [-0.25, 2.5]])
    first_vector = jnp.asarray([1.0, -1.0])
    second_vector = jnp.asarray([0.5, 2.0])
    first_function = la.PartialFractionRationalFunction(
        jnp.asarray([4.0, 5.0]),
        jnp.asarray([1.0, -0.5]),
        polynomial_coefficients=jnp.asarray([0.25, 0.1]),
    )
    second_function = la.PartialFractionRationalFunction(
        jnp.asarray([4.5, 6.0]),
        jnp.asarray([-0.75, 0.2]),
        polynomial_coefficients=jnp.asarray([-0.1, 0.3]),
    )
    first_operator = la.DenseLinearOperator(first_matrix, operator_id="refresh-rational")
    second_operator = la.DenseLinearOperator(
        second_matrix, operator_id="refresh-rational"
    )
    plan = la.plan_rational_function_action(first_operator, first_function)
    prepared = la.prepare_rational_function_action(
        first_operator,
        first_vector,
        first_function,
        plan,
    )
    compiled = jax.jit(la.rational_function_action)

    first = compiled(prepared)
    refreshed = la.refresh_rational_function_action(
        prepared,
        second_operator,
        second_function,
        second_vector,
    )
    second = compiled(refreshed)

    assert first.status == int(la.RationalFunctionStatus.SUCCESS)
    assert second.status == int(la.RationalFunctionStatus.SUCCESS)
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == 1
    assert refreshed.refresh_count == 1
    assert second.provenance.numeric_version == 1
    assert jnp.allclose(
        second.value,
        _dense_rational_action(second_matrix, second_vector, second_function),
        rtol=1e-11,
        atol=1e-12,
    )


def test_rational_coefficients_are_differentiable_on_a_reusable_projection():
    matrix = jnp.asarray([[2.0, 0.5], [0.5, 3.0]])
    vector = jnp.asarray([1.0, -1.0])
    function = la.PartialFractionRationalFunction(
        jnp.asarray([4.0, 5.0]),
        jnp.asarray([1.0, -0.5]),
        polynomial_coefficients=jnp.asarray([0.25, 0.1]),
    )
    operator = la.DenseLinearOperator(matrix, operator_id="differentiable-rational")
    prepared = la.prepare_rational_function_action(operator, vector, function)

    def prepared_objective(residues):
        updated = eqx.tree_at(lambda state: state.function.residues, prepared, residues)
        return jnp.real(jnp.sum(la.rational_function_action(updated).value ** 2))

    def dense_objective(residues):
        updated = eqx.tree_at(lambda state: state.residues, function, residues)
        return jnp.real(jnp.sum(_dense_rational_action(matrix, vector, updated) ** 2))

    actual = jax.jit(jax.grad(prepared_objective))(function.residues)
    expected = jax.grad(dense_objective)(function.residues)
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_zero_residue_masks_an_irrelevant_singular_shift():
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0]))
    vector = jnp.asarray([1.0, 2.0, -1.0])
    function = la.PartialFractionRationalFunction(
        jnp.asarray([1.0, 5.0]),
        jnp.asarray([0.0, 2.0]),
        polynomial_coefficients=jnp.asarray([0.5]),
    )
    operator = la.DenseLinearOperator(matrix, operator_id="masked-pole")
    result = la.rational_function_action(operator, vector, function)
    expected = 0.5 * vector + 2.0 * jnp.linalg.solve(5.0 * jnp.eye(3) - matrix, vector)

    assert result.diagnostics.shifted_status[0] == int(la.ShiftedSolveStatus.SINGULAR)
    assert not result.diagnostics.active_poles[0]
    assert result.status == int(la.RationalFunctionStatus.SUCCESS)
    assert jnp.all(jnp.isfinite(result.value))
    assert jnp.allclose(result.value, expected, rtol=1e-11, atol=1e-12)


def test_active_singular_pole_propagates_failure_status():
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0]))
    function = la.PartialFractionRationalFunction(
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
    )
    result = la.rational_function_action(
        la.DenseLinearOperator(matrix, operator_id="singular-pole"),
        jnp.asarray([1.0, 0.0, 0.0]),
        function,
    )

    assert result.status == int(la.RationalFunctionStatus.SHIFTED_SOLVE_FAILURE)
    assert not result.diagnostics.converged


def test_rational_plan_enforces_aggregate_matvec_and_structure_contracts():
    operator = la.DenseLinearOperator(jnp.eye(3), operator_id="budget-rational")
    function = la.PartialFractionRationalFunction(
        jnp.asarray([4.0]),
        jnp.asarray([1.0]),
        polynomial_coefficients=jnp.asarray([1.0, 2.0, 3.0]),
    )
    policy = la.RationalFunctionPolicy(
        resources=la.RationalFunctionResourcePolicy(max_matvec_count=4)
    )
    with pytest.raises(ValueError, match="matvecs"):
        la.plan_rational_function_action(operator, function, policy)

    plan = la.plan_rational_function_action(operator, function)
    incompatible = la.PartialFractionRationalFunction(
        jnp.asarray([4.0, 5.0]),
        jnp.asarray([1.0, 1.0]),
        polynomial_coefficients=jnp.asarray([1.0, 2.0, 3.0]),
    )
    with pytest.raises(ValueError, match="structure"):
        la.prepare_rational_function_action(
            operator,
            jnp.ones(3),
            incompatible,
            plan,
        )
