#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _self_adjoint_properties():
    return la.OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "construction"},
    )


def test_shifted_family_solves_many_systems_from_one_shared_basis():
    matrix = jnp.asarray([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    operator = la.DenseLinearOperator(
        matrix,
        properties=_self_adjoint_properties(),
        operator_id="shared-shifted-operator",
    )
    family = la.ShiftedLinearSystemFamily(operator, jnp.asarray([5.0, 6.0, 7.0]))
    rhs = jnp.asarray([1.0, -2.0, 0.5])
    plan = la.plan_shifted_solve(
        family,
        la.ShiftedSolvePolicy(max_dimension=3),
    )
    prepared = la.prepare_shifted_solve(family, rhs, plan)
    result = la.solve_shifted(prepared)
    expected = jax.vmap(lambda shift: jnp.linalg.solve(shift * jnp.eye(3) - matrix, rhs))(
        family.shifts
    )

    assert plan.selected_method == "lanczos"
    assert plan.cost.matvec_count == 3
    assert result.all_successful
    assert jnp.allclose(result.value, expected, rtol=1e-11, atol=1e-11)
    assert result.diagnostics.setup_matvec_count == 3
    assert result.diagnostics.solve_matvec_count == 0
    assert jnp.all(result.diagnostics.residual_norm < 1e-11)
    assert jnp.allclose(result.solution(1), expected[1])


def test_shifted_family_supports_complex_shifts_and_jitted_re_evaluation():
    matrix = jnp.asarray([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    operator = la.DenseLinearOperator(
        matrix,
        properties=_self_adjoint_properties(),
        operator_id="complex-shifted-operator",
    )
    shifts = jnp.asarray([2.0 + 1.0j, 5.0 + 2.0j])
    family = la.ShiftedLinearSystemFamily(operator, shifts)
    rhs = jnp.asarray([1.0, -2.0, 0.5])
    prepared = la.prepare_shifted_solve(
        family,
        rhs,
        la.ShiftedSolvePolicy(max_dimension=3),
    )
    result = jax.jit(lambda state: la.solve_shifted(state))(prepared)
    expected = jax.vmap(
        lambda shift: jnp.linalg.solve(
            shift * jnp.eye(3, dtype=shift.dtype) - matrix,
            rhs.astype(shift.dtype),
        )
    )(shifts)

    assert jnp.issubdtype(result.value.dtype, jnp.complexfloating)
    assert result.all_successful
    assert jnp.allclose(result.value, expected, rtol=1e-11, atol=1e-11)


def test_shifted_truncation_reports_true_physical_residuals_without_false_success():
    matrix = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    operator = la.DenseLinearOperator(
        matrix,
        properties=_self_adjoint_properties(),
    )
    family = la.ShiftedLinearSystemFamily(operator, jnp.asarray([5.0, 6.0]))
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0])
    result = la.solve_shifted(
        family,
        rhs,
        policy=la.ShiftedSolvePolicy(
            max_dimension=2,
            relative_tolerance=1e-14,
            absolute_tolerance=1e-14,
        ),
    )
    physical_residuals = jax.vmap(
        lambda shift, value: jnp.linalg.norm(shift * value - matrix @ value - rhs)
    )(family.shifts, result.value)

    assert jnp.all(result.status == int(la.ShiftedSolveStatus.MAX_DIMENSION_REACHED))
    assert not result.all_successful
    assert jnp.allclose(
        result.diagnostics.residual_norm,
        physical_residuals,
        rtol=1e-10,
        atol=1e-12,
    )


def test_shifted_family_exposes_singular_systems_and_accepts_zero_rhs():
    operator = la.DiagonalLinearOperator(
        jnp.asarray([1.0, 2.0, 3.0]),
        operator_id="singular-shifted-operator",
    )
    family = la.ShiftedLinearSystemFamily(operator, jnp.asarray([2.0, 4.0]))
    policy = la.ShiftedSolvePolicy(max_dimension=3)
    singular = la.solve_shifted(family, jnp.ones(3), policy=policy)
    zero = la.solve_shifted(family, jnp.zeros(3), policy=policy)

    assert singular.status[0] == int(la.ShiftedSolveStatus.SINGULAR)
    assert singular.diagnostics.rank[0] == 2
    assert singular.diagnostics.residual_norm[0] > 0.9
    assert singular.status[1] == int(la.ShiftedSolveStatus.SUCCESS)
    assert zero.all_successful
    assert jnp.array_equal(zero.value, jnp.zeros((2, 3)))
    assert jnp.array_equal(zero.diagnostics.residual_norm, jnp.zeros(2))


def test_shifted_plan_enforces_whole_family_resource_budgets():
    operator = la.DiagonalLinearOperator(jnp.arange(1.0, 6.0))
    family = la.ShiftedLinearSystemFamily(operator, jnp.arange(6.0, 10.0))
    plan = la.plan_shifted_solve(
        family,
        la.ShiftedSolvePolicy(max_dimension=4),
    )
    constrained = la.ShiftedSolvePolicy(
        max_dimension=4,
        resources=la.ShiftedSolveResourcePolicy(
            max_storage_bytes=plan.cost.total_storage_bytes - 1,
        ),
    )

    assert plan.cost.num_shifts == 4
    assert plan.cost.solution_storage_bytes == 4 * 5 * jnp.dtype(jnp.float64).itemsize
    assert plan.cost.total_storage_bytes > plan.cost.solution_storage_bytes
    with pytest.raises(ValueError, match="storage estimate"):
        la.plan_shifted_solve(family, constrained)


def test_shifted_refresh_preserves_plan_and_rebuilds_operator_rhs_and_shifts():
    first_matrix = jnp.asarray([[3.0, 0.5], [0.5, 2.0]])
    first_operator = la.DenseLinearOperator(
        first_matrix,
        properties=_self_adjoint_properties(),
        operator_id="refreshable-shifted-operator",
    )
    first_family = la.ShiftedLinearSystemFamily(
        first_operator,
        jnp.asarray([4.0, 5.0]),
    )
    prepared = la.prepare_shifted_solve(
        first_family,
        jnp.asarray([1.0, -2.0]),
        la.ShiftedSolvePolicy(max_dimension=2),
    )
    second_matrix = jnp.asarray([[2.5, -0.25], [-0.25, 1.5]])
    second_operator = la.DenseLinearOperator(
        second_matrix,
        properties=_self_adjoint_properties(),
        operator_id="refreshable-shifted-operator",
    )
    second_family = la.ShiftedLinearSystemFamily(
        second_operator,
        jnp.asarray([3.5, 6.0]),
    )
    rhs = jnp.asarray([-1.0, 0.75])
    refreshed = la.refresh_shifted_solve(prepared, second_family, rhs)
    result = la.solve_shifted(refreshed)
    expected = jax.vmap(
        lambda shift: jnp.linalg.solve(shift * jnp.eye(2) - second_matrix, rhs)
    )(second_family.shifts)

    assert refreshed.plan is prepared.plan
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == 1
    assert refreshed.refresh_count == 1
    assert refreshed.projection.numeric_version == 1
    assert jnp.allclose(result.value, expected, rtol=1e-11, atol=1e-11)


def test_shifted_solutions_differentiate_with_respect_to_runtime_shifts():
    matrix = jnp.asarray([[3.0, 0.5], [0.5, 2.0]])
    operator = la.DenseLinearOperator(
        matrix,
        properties=_self_adjoint_properties(),
    )
    family = la.ShiftedLinearSystemFamily(operator, jnp.asarray([4.0, 5.0]))
    rhs = jnp.asarray([1.0, -2.0])
    prepared = la.prepare_shifted_solve(
        family,
        rhs,
        la.ShiftedSolvePolicy(max_dimension=2),
    )

    def specialized(shifts):
        values = la.solve_shifted(prepared, shifts=shifts).value
        return jnp.sum(values**2)

    def dense(shifts):
        values = jax.vmap(
            lambda shift: jnp.linalg.solve(shift * jnp.eye(2) - matrix, rhs)
        )(shifts)
        return jnp.sum(values**2)

    actual = jax.jit(jax.grad(specialized))(family.shifts)
    expected = jax.grad(dense)(family.shifts)
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-11)
