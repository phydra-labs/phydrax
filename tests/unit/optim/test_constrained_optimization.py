#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _termination(*, steps=50, tolerance=1e-7):
    return phx.optim.OptimizationTermination(
        absolute_optimality=tolerance,
        relative_optimality=0.0,
        maximum_steps=steps,
    )


def test_bounds_broadcast_over_pytrees_and_report_activity():
    parameters = {"a": jnp.array([-2.0, 0.5]), "b": jnp.array(3.0)}
    bounds = phx.optim.Bounds(
        {"a": jnp.array([-1.0, 0.0]), "b": jnp.array(-2.0)},
        {"a": jnp.array([1.0, 1.0]), "b": jnp.array(2.0)},
    )
    projected = bounds.project(parameters)

    np.testing.assert_allclose(projected["a"], jnp.array([-1.0, 0.5]))
    np.testing.assert_allclose(projected["b"], 2.0)
    assert bounds.contains(projected)
    assert bounds.violation(parameters) == 1.0

    gradient = {"a": jnp.array([1.0, 0.0]), "b": jnp.array(-1.0)}
    active = bounds.active_mask(projected, gradient)
    np.testing.assert_array_equal(active["a"], jnp.array([True, False]))
    assert active["b"]


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.ProjectedGradient(),
        phx.optim.ActiveSetNewton(),
        phx.optim.ProjectedLBFGS(),
    ],
)
def test_bound_methods_converge_to_active_corner_with_feasible_iterates(method):
    bounds = phx.optim.Bounds(
        jnp.array([0.0, -1.0]),
        jnp.array([1.0, 2.0]),
    )
    result = phx.optim.minimize(
        lambda value, _: jnp.sum((value - jnp.array([2.0, -3.0])) ** 2),
        jnp.array([-4.0, 4.0]),
        bounds=bounds,
        method=method,
        termination=_termination(steps=100),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([1.0, -1.0]), atol=1e-7)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.diagnostics.primal_feasibility == 0.0
    assert result.diagnostics.dual_feasibility < 1e-7
    assert result.diagnostics.complementarity < 1e-10
    assert result.diagnostics.active_constraints == 2
    assert result.provenance.globalization == "projected-armijo"


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.ProjectedGradient(),
        phx.optim.ActiveSetNewton(),
        phx.optim.ProjectedLBFGS(),
    ],
)
def test_bound_methods_stage_large_budget_and_backtracked_step(method):
    problem = phx.optim.MinimizationProblem(
        lambda value, target: 5.0 * jnp.sum((value - target) ** 2),
        bounds=phx.optim.Bounds(-20.0, 20.0),
    )
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=1e-8,
        relative_optimality=0.0,
        maximum_steps=1_000_000,
    )

    def solve(target):
        return phx.optim.minimize(
            problem,
            jnp.array([0.0]),
            method=method,
            termination=termination,
            args=target,
        )

    target = jnp.array([1.0])
    eager = solve(target)
    compiled = eqx.filter_jit(solve)(target)

    np.testing.assert_allclose(compiled.parameters, eager.parameters, atol=1e-8)
    np.testing.assert_allclose(compiled.objective, eager.objective, atol=1e-8)
    assert (
        int(compiled.status)
        == int(eager.status)
        == int(phx.optim.OptimizationStatus.SUCCESS)
    )
    compiled_diagnostics = jax.tree.leaves(compiled.diagnostics)
    eager_diagnostics = jax.tree.leaves(eager.diagnostics)
    assert len(compiled_diagnostics) == len(eager_diagnostics)
    for compiled_value, eager_value in zip(
        compiled_diagnostics,
        eager_diagnostics,
        strict=True,
    ):
        np.testing.assert_allclose(compiled_value, eager_value, atol=1e-8)
    assert int(compiled.diagnostics.accepted_steps) >= 1
    if not isinstance(method, phx.optim.ActiveSetNewton):
        assert int(compiled.diagnostics.globalization_evaluations) > int(
            compiled.diagnostics.iterations
        )


def test_bound_method_can_reject_infeasible_initial_point_by_policy():
    problem = phx.optim.MinimizationProblem(
        lambda value, _: jnp.sum(value**2),
        bounds=phx.optim.Bounds(0.0, 1.0),
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([-2.0]),
        method=phx.optim.ProjectedGradient(project_initial=False),
    )

    assert result.status == phx.optim.OptimizationStatus.INFEASIBLE
    assert result.diagnostics.primal_feasibility == 2.0


def _mixed_constraint_problem():
    constraints = (
        phx.optim.NonlinearConstraint(
            lambda value, _: jnp.array([value[0] + value[1]]),
            lower=1.0,
            upper=1.0,
            constraint_id="sum-one",
        ),
        phx.optim.NonlinearConstraint(
            lambda value, _: {"nonnegative": value[0]},
            lower={"nonnegative": 0.0},
            constraint_id="nonnegative-first",
        ),
    )
    return phx.optim.MinimizationProblem(
        lambda value, _: jnp.sum((value - jnp.array([1.0, 2.0])) ** 2),
        constraints=constraints,
        problem_id="mixed-constrained-quadratic",
    )


@pytest.mark.parametrize(
    ("method", "tolerance"),
    [
        (phx.optim.AugmentedLagrangian(inner_maximum_steps=40), 1e-4),
        (phx.optim.SQP(), 1e-6),
    ],
)
def test_nonlinear_constrained_methods_satisfy_kkt_system(method, tolerance):
    result = phx.optim.minimize(
        _mixed_constraint_problem(),
        jnp.array([0.5, 0.5]),
        method=method,
        termination=_termination(steps=25, tolerance=tolerance),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([0.0, 1.0]), atol=5e-4)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.diagnostics.primal_feasibility <= tolerance
    assert result.diagnostics.dual_feasibility <= tolerance
    assert result.diagnostics.complementarity <= tolerance
    assert result.diagnostics.active_constraints >= 1
    if isinstance(method, phx.optim.AugmentedLagrangian):
        assert result.diagnostics.setup_refreshes > 0
        assert result.diagnostics.numeric_refreshes > 0
        assert result.diagnostics.hvp_evaluations > 0


def test_sqp_reports_failed_restoration_for_infeasible_nonlinear_equation():
    problem = phx.optim.MinimizationProblem(
        lambda value, _: jnp.sum(value**2),
        constraints=(
            phx.optim.NonlinearConstraint(
                lambda value, _: value**2 + 1.0,
                lower=0.0,
                upper=0.0,
            ),
        ),
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([0.0]),
        method=phx.optim.SQP(),
        termination=_termination(steps=5),
    )

    assert result.status in (
        phx.optim.OptimizationStatus.RESTORATION_FAILED,
        phx.optim.OptimizationStatus.MAXIMUM_STEPS_REACHED,
    )
    assert result.diagnostics.primal_feasibility >= 1.0
    assert result.diagnostics.direction_fallbacks >= 1


def _assert_constrained_diagnostics_match(compiled, eager):
    for field in (
        "iterations",
        "accepted_steps",
        "rejected_steps",
        "objective_evaluations",
        "gradient_evaluations",
        "constraint_evaluations",
        "globalization_evaluations",
        "direction_fallbacks",
        "active_constraints",
    ):
        np.testing.assert_array_equal(
            getattr(compiled.diagnostics, field),
            getattr(eager.diagnostics, field),
        )
    for field in (
        "initial_optimality_norm",
        "final_optimality_norm",
        "final_step_norm",
        "accepted_step_size",
        "damping",
        "primal_feasibility",
        "dual_feasibility",
        "complementarity",
    ):
        np.testing.assert_allclose(
            getattr(compiled.diagnostics, field),
            getattr(eager.diagnostics, field),
            equal_nan=True,
        )


@pytest.mark.parametrize(
    ("method", "tolerance"),
    [
        (
            phx.optim.AugmentedLagrangian(
                maximum_outer_steps=100_000,
                inner_maximum_steps=40,
            ),
            1e-4,
        ),
        (phx.optim.SQP(), 1e-6),
    ],
)
def test_constrained_native_methods_filtered_jit_match_large_budget_eager(
    method,
    tolerance,
):
    problem = _mixed_constraint_problem()
    termination = _termination(steps=100_000, tolerance=tolerance)

    def solve(initial):
        return phx.optim.minimize(
            problem,
            initial,
            method=method,
            termination=termination,
        )

    initial = jnp.array([0.5, 0.5])
    eager = solve(initial)
    compiled = eqx.filter_jit(solve)(initial)

    assert compiled.status.shape == ()
    assert jnp.issubdtype(compiled.status.dtype, jnp.integer)
    assert (
        int(compiled.status)
        == int(eager.status)
        == int(phx.optim.OptimizationStatus.SUCCESS)
    )
    np.testing.assert_allclose(compiled.parameters, eager.parameters, atol=tolerance)
    np.testing.assert_allclose(compiled.objective, eager.objective, atol=tolerance)
    _assert_constrained_diagnostics_match(compiled, eager)


def test_sqp_filtered_jit_restoration_failure_preserves_accepted_iterate():
    problem = phx.optim.MinimizationProblem(
        lambda value, _: jnp.sum(value**2),
        constraints=(
            phx.optim.NonlinearConstraint(
                lambda value, _: value**2 + 1.0,
                lower=0.0,
                upper=0.0,
            ),
        ),
    )
    method = phx.optim.SQP()
    termination = _termination(steps=100_000)

    def solve(initial):
        return phx.optim.minimize(
            problem,
            initial,
            method=method,
            termination=termination,
        )

    initial = jnp.array([0.0])
    eager = solve(initial)
    compiled = eqx.filter_jit(solve)(initial)

    assert (
        int(compiled.status)
        == int(eager.status)
        == int(phx.optim.OptimizationStatus.RESTORATION_FAILED)
    )
    np.testing.assert_array_equal(compiled.parameters, initial)
    np.testing.assert_array_equal(eager.parameters, initial)
    assert int(compiled.diagnostics.rejected_steps) == 1
    assert int(compiled.diagnostics.direction_fallbacks) >= 1
    _assert_constrained_diagnostics_match(compiled, eager)


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.AugmentedLagrangian(
            maximum_outer_steps=12,
            inner_maximum_steps=30,
        ),
        phx.optim.SQP(),
        phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-centered",
        ),
    ],
)
def test_native_constrained_methods_support_jvp_vmap_and_pytree_parameters(method):
    constraint = phx.optim.NonlinearConstraint(
        lambda parameters, target: parameters["state"] - target,
        lower=0.0,
        upper=0.0,
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, target: jnp.sum((parameters["state"] - target) ** 2),
        constraints=(constraint,),
    )
    termination = _termination(steps=40, tolerance=1e-8)

    def solution(target):
        return phx.optim.minimize(
            problem,
            {"state": jnp.array([0.0])},
            method=method,
            termination=termination,
            args=target,
        ).parameters["state"][0]

    targets = jnp.array([1.0, 1.5])
    mapped = jax.vmap(solution)(targets)
    value, derivative = jax.jvp(
        solution,
        (jnp.array(1.25),),
        (jnp.array(0.3),),
    )

    np.testing.assert_allclose(mapped, targets, atol=2e-5)
    np.testing.assert_allclose(value, 1.25, atol=2e-5)
    np.testing.assert_allclose(derivative, 0.3, atol=2e-5)


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.ProjectedGradient(),
        phx.optim.ProjectedLBFGS(),
        phx.optim.ActiveSetNewton(),
    ],
)
def test_native_bound_methods_support_jvp_vmap_and_pytree_parameters(method):
    problem = phx.optim.MinimizationProblem(
        lambda parameters, target: jnp.sum((parameters["state"] - target) ** 2),
        bounds=phx.optim.Bounds(-2.0, 2.0),
    )
    termination = _termination(steps=30, tolerance=1e-8)

    def solution(target):
        return phx.optim.minimize(
            problem,
            {"state": jnp.array([0.0])},
            method=method,
            termination=termination,
            args=target,
        ).parameters["state"][0]

    targets = jnp.array([0.5, 1.0])
    mapped = jax.vmap(solution)(targets)
    value, derivative = jax.jvp(
        solution,
        (jnp.array(0.75),),
        (jnp.array(0.2),),
    )

    np.testing.assert_allclose(mapped, targets, atol=2e-6)
    np.testing.assert_allclose(value, 0.75, atol=2e-6)
    np.testing.assert_allclose(derivative, 0.2, atol=2e-6)
