#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _termination(*, steps: Any = 50, tolerance: Any = 1e-7) -> Any:
    return phx.optim.OptimizationTermination(
        absolute_optimality=tolerance,
        relative_optimality=0.0,
        maximum_steps=steps,
    )


def test_constrained_optimization_scenario_1() -> None:
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
    for method in [
        phx.optim.ProjectedGradient(),
        phx.optim.ActiveSetNewton(),
        phx.optim.ProjectedLBFGS(),
    ]:
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
    for method, tolerance in [
        (phx.optim.AugmentedLagrangian(inner_maximum_steps=40), 1e-4),
        (phx.optim.SQP(), 1e-6),
    ]:
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


def test_bound_methods_stage_large_budget_and_backtracked_step() -> None:
    for method in [
        phx.optim.ProjectedGradient(),
        phx.optim.ActiveSetNewton(),
        phx.optim.ProjectedLBFGS(),
    ]:
        problem = phx.optim.MinimizationProblem(
            lambda value, target: 5.0 * jnp.sum((value - target) ** 2),
            bounds=phx.optim.Bounds(-20.0, 20.0),
        )
        termination = phx.optim.OptimizationTermination(
            absolute_optimality=1e-8,
            relative_optimality=0.0,
            maximum_steps=1_000_000,
        )

        def solve(target: Any) -> Any:
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


def _mixed_constraint_problem() -> Any:
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


def test_constrained_optimization_scenario_2() -> None:
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
    problem = phx.optim.MinimizationProblem(
        lambda value, _: jnp.sum((value - 2.0) ** 2),
        constraints=(
            phx.optim.NonlinearConstraint(
                lambda value, _: value,
                lower=0.0,
                constraint_id="nonnegative",
            ),
        ),
    )
    result = phx.optim.minimize(
        problem,
        jnp.asarray([1.0]),
        method=phx.optim.AugmentedLagrangian(
            maximum_outer_steps=4,
            inner_maximum_steps=8,
        ),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=0.0,
            relative_optimality=0.0,
            maximum_steps=4,
            maximum_evaluations=1,
        ),
    )

    assert int(result.status) == int(
        phx.optim.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
    )
    assert int(result.diagnostics.iterations) == 0


def _assert_constrained_diagnostics_match(compiled: Any, eager: Any) -> None:
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


def test_constrained_native_methods_filtered_jit_match_large_budget_eager() -> None:
    for method, tolerance in [
        (
            phx.optim.AugmentedLagrangian(
                maximum_outer_steps=100_000,
                inner_maximum_steps=40,
            ),
            1e-4,
        ),
        (phx.optim.SQP(), 1e-6),
    ]:
        problem = _mixed_constraint_problem()
        termination = _termination(steps=100_000, tolerance=tolerance)

        def solve(initial: Any) -> Any:
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


def test_sqp_filtered_jit_restoration_failure_preserves_accepted_iterate() -> None:
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

    def solve(initial: Any) -> Any:
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


def test_native_constrained_methods_support_jvp_vmap_and_pytree_parameters() -> None:
    for method in [
        phx.optim.AugmentedLagrangian(
            maximum_outer_steps=12,
            inner_maximum_steps=30,
        ),
        phx.optim.SQP(),
        phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-centered",
        ),
    ]:
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

        def solution(target: Any) -> Any:
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


def test_native_bound_methods_support_jvp_vmap_and_pytree_parameters() -> None:
    for method in [
        phx.optim.ProjectedGradient(),
        phx.optim.ProjectedLBFGS(),
        phx.optim.ActiveSetNewton(),
    ]:
        problem = phx.optim.MinimizationProblem(
            lambda parameters, target: jnp.sum((parameters["state"] - target) ** 2),
            bounds=phx.optim.Bounds(-2.0, 2.0),
        )
        termination = _termination(steps=30, tolerance=1e-8)

        def solution(target: Any) -> Any:
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


def test_projected_lbfgs_admits_an_unbounded_domain_without_a_finite_box() -> None:
    def quadratic(x: jax.Array, _: None) -> jax.Array:
        return (1.0 - x[0]) ** 2 + 100.0 * (1.0 - x[1]) ** 2

    result = phx.optim.minimize(
        quadratic,
        jnp.array([-1.2, 1.0], dtype=jnp.float64),
        method=phx.optim.ProjectedLBFGS(),
        bounds=phx.optim.Bounds(),
        termination=_termination(steps=500, tolerance=1e-8),
    )

    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    np.testing.assert_allclose(result.parameters, jnp.array([1.0, 1.0]), atol=1e-6)
    assert result.diagnostics.primal_feasibility == 0.0
    assert result.diagnostics.complementarity == 0.0
    assert int(result.diagnostics.active_constraints) == 0
    assert result.certificate is not None
    assert result.certificate.inequality_multipliers.size == 0
    assert result.certificate.equality_multipliers.size == 0


@pytest.mark.strict_jax
def test_bound_activity_handles_absent_and_fixed_constraints_without_nan() -> None:
    bounds = phx.optim.Bounds(
        jnp.asarray([-jnp.inf, 0.0, 1.0, -jnp.inf], dtype=jnp.float64),
        jnp.asarray([jnp.inf, jnp.inf, 1.0, 0.0], dtype=jnp.float64),
    )
    values = jnp.asarray([0.0, 0.0, 1.0, 0.0], dtype=jnp.float64)
    gradient = jnp.asarray([1.0, 1.0, -1.0, -1.0], dtype=jnp.float64)
    with jax.debug_nans(True):
        active = bounds.active_mask(values, gradient)
    np.testing.assert_array_equal(
        active, np.asarray([False, True, True, True], dtype=np.bool_)
    )


@pytest.mark.strict_jax
def test_lbfgs_history_ring_matches_dense_inverse_bfgs_in_every_group() -> None:
    from phydrax.optim._lbfgs_kernel import (
        admit_lbfgs_pair,
        append_lbfgs_pair,
        initial_lbfgs_history,
        lbfgs_direction,
        lbfgs_pair_curvature,
    )

    rng = np.random.default_rng(7)
    capacity = 3
    groups = (np.array([0, 1]), np.array([2, 3]))
    layout = phx.optim.OptimizerGroupLayout(jnp.array([0, 0, 1, 1]), group_count=2)
    hessians = []
    for _ in groups:
        factor = rng.normal(size=(4, 4))
        hessians.append(factor @ factor.T + 4.0 * np.eye(4))
    history = initial_lbfgs_history(jnp.zeros((4, 2)), capacity, layout, jnp.float64)
    stored: list[list[tuple[np.ndarray, np.ndarray]]] = [[], []]
    for step_index in range(5):
        step = rng.normal(size=(4, 2))
        change = np.zeros_like(step)
        for group, rows in enumerate(groups):
            change[rows] = (hessians[group] @ step[rows].reshape(-1)).reshape(2, 2)
        curvature, step_squared, change_squared = lbfgs_pair_curvature(
            jnp.asarray(step), jnp.asarray(change), layout
        )
        # Group 1 skips odd pairs, so the two rings hold different histories and
        # only group 0 wraps around its capacity.
        admitted = jnp.array([True, step_index % 2 == 0]) & admit_lbfgs_pair(
            curvature, step_squared, change_squared, 1e-10
        )
        history = append_lbfgs_pair(
            history,
            jnp.asarray(step),
            jnp.asarray(change),
            curvature,
            change_squared,
            admitted,
            layout,
        )
        for group, rows in enumerate(groups):
            if bool(admitted[group]):
                stored[group].append((step[rows].reshape(-1), change[rows].reshape(-1)))
    np.testing.assert_array_equal(history.count, [capacity, capacity])

    gradient = rng.normal(size=(4, 2))
    direction = np.asarray(lbfgs_direction(jnp.asarray(gradient), history, layout))
    newest_changes = np.zeros_like(gradient)
    for group, rows in enumerate(groups):
        pairs = stored[group][-capacity:]
        newest_step, newest_change = pairs[-1]
        inverse = float(
            newest_step @ newest_change / (newest_change @ newest_change)
        ) * np.eye(4)
        for step, change in pairs:
            rho = 1.0 / float(change @ step)
            update = np.eye(4) - rho * np.outer(change, step)
            inverse = update.T @ inverse @ update + rho * np.outer(step, step)
        np.testing.assert_allclose(
            direction[rows].reshape(-1),
            -inverse @ gradient[rows].reshape(-1),
            rtol=1e-10,
        )
        newest_changes[rows] = newest_change.reshape(2, 2)

    # The newest secant equation H y = s holds independently in each group.
    secant = np.asarray(lbfgs_direction(jnp.asarray(newest_changes), history, layout))
    for group, rows in enumerate(groups):
        np.testing.assert_allclose(
            secant[rows].reshape(-1), -stored[group][-1][0], rtol=1e-10
        )
