#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


def _clarabel_policy():
    return phx.optim.ConvexSolvePolicy(phx.optim.ClarabelInteriorPoint(presolve=False))


def _solve(binding):
    return phx.optim.solve_convex_program(binding.prepare(_clarabel_policy())).result


def test_cvxpy_sparse_import_parameter_refresh_and_inverse_maps():
    cp = pytest.importorskip("cvxpy")
    parameter = cp.Parameter(nonneg=True, value=1.0)
    x = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(x[0] + 2.0 * x[1]), [x >= 0, cp.sum(x) == parameter])
    binding = phx.optim.import_cvxpy_problem(problem)
    assert binding.program.constraint_is_sparse
    assert binding.constraint_slices
    parameter.value = 2.0
    refreshed = phx.optim.refresh_cvxpy_program(binding)
    assert refreshed.program.structure_id == binding.program.structure_id
    assert refreshed.constraint_slices == binding.constraint_slices
    assert refreshed.binding_id == binding.binding_id
    assert refreshed.numeric_version == binding.numeric_version + 1
    assert refreshed.numeric_fingerprint != binding.numeric_fingerprint
    assert refreshed.numeric_binding_id != binding.numeric_binding_id


def test_cvxpy_export_rejects_mismatched_result_binding():
    cp = pytest.importorskip("cvxpy")
    x = cp.Variable(boolean=True)
    problem = cp.Problem(cp.Minimize(x), [x >= 0])
    with pytest.raises(TypeError, match="Mixed-integer"):
        phx.optim.import_cvxpy_problem(problem)


def test_cvxpy_solution_restores_primal_and_constraint_duals():
    cp = pytest.importorskip("cvxpy")
    pytest.importorskip("clarabel")
    x = cp.Variable(2)
    constraints = [x >= 0, cp.sum(x) == 1.0]
    problem = cp.Problem(cp.Minimize(x[0] + 2.0 * x[1]), constraints)
    binding = phx.optim.import_cvxpy_problem(problem)
    result = _solve(binding)
    assert int(result.provenance.numeric_version) == binding.numeric_version
    assert result.provenance.numeric_binding_id == binding.numeric_binding_id
    restored = phx.optim.restore_cvxpy_solution(binding, result)
    assert restored
    assert x.value is not None
    assert all(constraint.dual_value is not None for constraint in constraints)
    assert jnp.allclose(jnp.asarray(x.value), jnp.asarray([1.0, 0.0]), atol=1e-6)


def test_cvxpy_solution_rejects_nonoptimal_result_without_mutation():
    cp = pytest.importorskip("cvxpy")
    pytest.importorskip("clarabel")
    x = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(x[0] + 2.0 * x[1]), [x >= 0, cp.sum(x) == 1.0])
    binding = phx.optim.import_cvxpy_problem(problem)
    result = _solve(binding)
    failed = eqx.tree_at(
        lambda value: value.status,
        result,
        jnp.asarray(
            int(phx.optim.ConvexProgramStatus.ITERATION_LIMIT),
            dtype=result.status.dtype,
        ),
    )

    with pytest.raises(ValueError, match="successful optimal result"):
        phx.optim.restore_cvxpy_solution(binding, failed)

    assert x.value is None
    assert problem.status is None


def test_cvxpy_solution_rejects_pre_refresh_result_before_mutation():
    cp = pytest.importorskip("cvxpy")
    pytest.importorskip("clarabel")
    parameter = cp.Parameter(nonneg=True, value=1.0)
    x = cp.Variable(2)
    constraint = cp.sum(x) == parameter
    problem = cp.Problem(cp.Minimize(x[0] + 2.0 * x[1]), [x >= 0, constraint])
    binding = phx.optim.import_cvxpy_problem(problem)
    stale_result = _solve(binding)

    parameter.value = 2.0
    refreshed = phx.optim.refresh_cvxpy_program(binding)
    with pytest.raises(ValueError, match="numeric binding"):
        phx.optim.restore_cvxpy_solution(refreshed, stale_result)

    assert x.value is None
    assert constraint.dual_value is None
    assert problem.status is None

    restored = phx.optim.restore_cvxpy_solution(refreshed, _solve(refreshed))
    assert restored
    assert jnp.allclose(jnp.sum(jnp.asarray(x.value)), 2.0, atol=1e-6)


def test_cvxpy_solution_rejects_result_from_different_identical_problem():
    cp = pytest.importorskip("cvxpy")
    pytest.importorskip("clarabel")
    x = cp.Variable(2)
    first = cp.Problem(cp.Minimize(x[0] + 2.0 * x[1]), [x >= 0, cp.sum(x) == 1.0])
    first_binding = phx.optim.import_cvxpy_problem(first)
    result = _solve(first_binding)

    y = cp.Variable(2)
    constraints = [y >= 0, cp.sum(y) == 1.0]
    second = cp.Problem(cp.Minimize(y[0] + 2.0 * y[1]), constraints)
    second_binding = phx.optim.import_cvxpy_problem(second)
    assert second_binding.program.structure_id == first_binding.program.structure_id

    with pytest.raises(ValueError, match="numeric binding"):
        phx.optim.restore_cvxpy_solution(second_binding, result)

    assert y.value is None
    assert all(constraint.dual_value is None for constraint in constraints)
    assert second.status is None
