#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _acceptance_policy():
    return phx.optim.StateAcceptancePolicy(
        state_relative_tolerance=0.0,
        state_absolute_tolerance=1.0e-6,
        adjoint_relative_tolerance=0.0,
        adjoint_absolute_tolerance=1.0e-6,
    )


def _state_design_problem(*, bounds=None):
    return phx.optim.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: jnp.sum((state - 2.0) ** 2) + 0.1 * jnp.sum(design**2),
        acceptance_policy=_acceptance_policy(),
        design_bounds=bounds,
        problem_id="linear-state-design",
    )


def _termination(*, tolerance=1e-6, steps=30):
    return phx.optim.OptimizationTermination(
        absolute_optimality=tolerance,
        relative_optimality=0.0,
        maximum_steps=steps,
    )


def test_least_squares_state_solver_satisfies_frozen_state_equation():
    problem = _state_design_problem()
    result = problem.solve_state(jnp.array([1.25]), jnp.array([0.0]))

    np.testing.assert_allclose(result.state, jnp.array([1.25]), atol=1e-10)
    assert result.successful
    assert result.residual_norm < 1e-12
    assert result.diagnostics.residual_evaluations > 0


@pytest.mark.parametrize(
    "method",
    [phx.optim.ReducedAdjoint(), phx.optim.SimultaneousKKT()],
)
def test_state_design_methods_recover_analytic_kkt_solution(method):
    result = phx.optim.solve_state_design(
        _state_design_problem(),
        jnp.array([0.0]),
        jnp.array([0.0]),
        method=method,
        termination=_termination(),
    )
    expected = jnp.array([4.0 / 2.2])

    np.testing.assert_allclose(result.state, expected, atol=2e-5)
    np.testing.assert_allclose(result.design, expected, atol=2e-5)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.diagnostics.primal_feasibility < 1e-6
    assert result.diagnostics.dual_feasibility < 1e-6
    assert result.adjoint is not None
    assert result.provenance.matrix_free


def test_reduced_adjoint_projects_bound_constrained_design():
    problem = _state_design_problem(bounds=phx.optim.Bounds(0.0, 1.0))
    result = phx.optim.solve_state_design(
        problem,
        jnp.array([0.0]),
        jnp.array([-1.0]),
        method=phx.optim.ReducedAdjoint(),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.state, jnp.array([1.0]), atol=2e-5)
    np.testing.assert_allclose(result.design, jnp.array([1.0]), atol=2e-5)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.diagnostics.primal_feasibility < 1e-6


def test_simultaneous_kkt_rejects_unmodeled_bound_complementarity():
    with pytest.raises(ValueError, match="unconstrained design"):
        phx.optim.solve_state_design(
            _state_design_problem(bounds=phx.optim.Bounds(0.0, 1.0)),
            jnp.array([0.0]),
            jnp.array([0.0]),
            method=phx.optim.SimultaneousKKT(),
            termination=_termination(),
        )


def _nested_state_design_problem():
    def residual(state, design, _):
        return {"field": state["field"] - design["controls"][0]}

    def objective(state, design, target):
        return jnp.sum((state["field"] - target) ** 2) + 0.1 * jnp.sum(
            design["controls"][0] ** 2
        )

    return phx.optim.StateDesignProblem(
        residual,
        objective,
        acceptance_policy=_acceptance_policy(),
        problem_id="nested-linear-state-design",
    )


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.ReducedAdjoint(),
        phx.optim.SimultaneousKKT(),
        phx.optim.ReducedNewtonKrylov(),
    ],
)
def test_state_design_methods_eager_and_jit_agree_with_large_static_budget(method):
    problem = _nested_state_design_problem()
    initial_state = {"field": jnp.array([0.0])}
    initial_design = {"controls": (jnp.array([0.0]),)}
    termination = _termination(tolerance=1e-7, steps=100_000)

    def solve(target):
        return phx.optim.solve_state_design(
            problem,
            initial_state,
            initial_design,
            method=method,
            termination=termination,
            args=target,
        )

    eager = solve(jnp.array([2.0]))
    compiled = eqx.filter_jit(solve)(jnp.array([2.0]))
    expected = jnp.array([4.0 / 2.2])

    np.testing.assert_allclose(compiled.state["field"], expected, atol=2e-5)
    np.testing.assert_allclose(
        compiled.design["controls"][0],
        expected,
        atol=2e-5,
    )
    for compiled_leaf, eager_leaf in zip(
        jax.tree.leaves((compiled.state, compiled.design)),
        jax.tree.leaves((eager.state, eager.design)),
        strict=True,
    ):
        np.testing.assert_allclose(compiled_leaf, eager_leaf, atol=1e-9)
    assert (
        int(compiled.status)
        == int(eager.status)
        == int(phx.optim.OptimizationStatus.SUCCESS)
    )
    assert int(compiled.diagnostics.iterations) == int(eager.diagnostics.iterations)
    assert int(compiled.diagnostics.accepted_steps) == int(
        eager.diagnostics.accepted_steps
    )
    assert int(compiled.diagnostics.rejected_steps) == int(
        eager.diagnostics.rejected_steps
    )


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.ReducedAdjoint(),
        phx.optim.SimultaneousKKT(),
        phx.optim.ReducedNewtonKrylov(),
    ],
)
def test_state_design_methods_support_jvp_vmap_and_nested_pytrees(method):
    problem = _nested_state_design_problem()
    initial_state = {"field": jnp.array([0.0])}
    initial_design = {"controls": (jnp.array([0.0]),)}
    termination = _termination(tolerance=1e-7, steps=40)

    def solution(target):
        result = phx.optim.solve_state_design(
            problem,
            initial_state,
            initial_design,
            method=method,
            termination=termination,
            args=target,
        )
        return result.design["controls"][0][0]

    targets = jnp.array([1.0, 2.0])
    mapped = jax.vmap(solution)(targets)
    value, derivative = jax.jvp(
        solution,
        (jnp.array(1.5),),
        (jnp.array(0.25),),
    )

    np.testing.assert_allclose(mapped, targets / 1.1, atol=2e-5)
    np.testing.assert_allclose(value, 1.5 / 1.1, atol=2e-5)
    np.testing.assert_allclose(derivative, 0.25 / 1.1, atol=2e-5)


def test_reduced_adjoint_rejected_trial_preserves_last_accepted_pair():
    problem = _state_design_problem()
    result = phx.optim.solve_state_design(
        problem,
        jnp.array([0.0]),
        jnp.array([0.0]),
        method=phx.optim.ReducedAdjoint(
            line_search=phx.optim.ArmijoLineSearch(maximum_steps=1)
        ),
        termination=_termination(steps=5),
    )

    np.testing.assert_array_equal(result.state, jnp.array([0.0]))
    np.testing.assert_array_equal(result.design, jnp.array([0.0]))
    np.testing.assert_allclose(result.objective, 4.0)
    assert int(result.status) == int(phx.optim.OptimizationStatus.LINE_SEARCH_FAILED)
    assert int(result.diagnostics.iterations) == 1
    assert int(result.diagnostics.accepted_steps) == 0
    assert int(result.diagnostics.rejected_steps) == 1
    assert int(result.diagnostics.constraint_evaluations) == 2
    assert float(result.diagnostics.accepted_step_size) == 0.0
    assert float(result.diagnostics.final_step_norm) == 0.0


def test_reduced_adjoint_evaluation_budget_gates_whole_outer_iterations():
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=1e-7,
        relative_optimality=0.0,
        maximum_steps=100_000,
        maximum_evaluations=1,
    )
    solve = eqx.filter_jit(
        lambda target: phx.optim.solve_state_design(
            _state_design_problem(),
            jnp.array([0.0]),
            jnp.array([0.0]),
            method=phx.optim.ReducedAdjoint(),
            termination=termination,
            args=target,
        )
    )

    result = solve(jnp.array([2.0]))

    np.testing.assert_array_equal(result.state, jnp.array([0.0]))
    np.testing.assert_array_equal(result.design, jnp.array([0.0]))
    assert int(result.status) == int(
        phx.optim.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
    )
    assert int(result.diagnostics.iterations) == 0
    assert int(result.diagnostics.constraint_evaluations) == 1
    assert int(result.diagnostics.objective_evaluations) > 1


def test_state_and_adjoint_acceptance_use_measured_defects_not_solver_status_alone():
    policy = phx.optim.StateAcceptancePolicy(
        state_relative_tolerance=0.0,
        state_absolute_tolerance=1e-6,
        adjoint_relative_tolerance=0.0,
        adjoint_absolute_tolerance=1e-6,
    )
    state_evidence = policy.state_evidence(
        jnp.asarray((1.0,)),
        jnp.asarray((1e-3,)),
        phx.optim.OptimizationStatus.SUCCESS,
        reference_norm=1.0,
        admissible=True,
        realization_matches=True,
    )
    adjoint_evidence = policy.adjoint_evidence(
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0 + 1e-3,)),
        phx.linalg.LinearSolveStatus.SUCCESS,
        admissible=True,
        realization_matches=True,
    )

    assert state_evidence.status_accepted
    assert state_evidence.residual_norm > state_evidence.threshold
    assert not state_evidence.accepted
    assert adjoint_evidence.status_accepted
    assert adjoint_evidence.transpose_defect_norm > adjoint_evidence.threshold
    assert not adjoint_evidence.accepted
