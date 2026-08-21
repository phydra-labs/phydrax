#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg
nl = phx.nonlinear


def _gmres_policy(*, relative: float = 1e-12, maximum_steps: int):
    return la.LinearSolvePolicy(
        la.GMRES(restart=maximum_steps),
        tolerance=la.TolerancePolicy(
            relative=relative,
            absolute=0.0,
            max_steps=maximum_steps,
        ),
    )


def test_eisenstat_walker_changes_inner_work_and_records_final_forcing():
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 4.0, 8.0]))
    expected = jnp.ones((4,))
    target = matrix @ expected
    problem = nl.NonlinearSystemProblem(
        lambda state, args: matrix @ state - target,
        problem_id="forcing-work",
    )
    linear_policy = _gmres_policy(relative=1e-10, maximum_steps=4)
    refresh = nl.JacobianRefreshPolicy("periodic", period=100)
    termination = nl.NonlinearTermination(
        absolute_residual=1e-9,
        relative_residual=1e-9,
        maximum_steps=20,
    )

    constant = nl.NewtonKrylov(
        linear_policy=linear_policy,
        forcing_policy=nl.NewtonForcingPolicy("constant"),
        jacobian_refresh=refresh,
    ).solve(problem, jnp.zeros((4,)), termination=termination)
    adaptive = nl.NewtonKrylov(
        linear_policy=linear_policy,
        forcing_policy=nl.NewtonForcingPolicy(
            "eisenstat-walker",
            initial=0.5,
            minimum=1e-10,
            maximum=0.8,
            gamma=0.9,
            exponent=1.5,
        ),
        jacobian_refresh=refresh,
    ).solve(problem, jnp.zeros((4,)), termination=termination)

    assert bool(constant.successful)
    assert bool(adaptive.successful)
    assert jnp.allclose(constant.state, expected, atol=1e-7)
    assert jnp.allclose(adaptive.state, expected, atol=1e-7)
    assert int(constant.diagnostics.iterations) < int(adaptive.diagnostics.iterations)
    assert int(constant.diagnostics.linear_iterations) < int(
        adaptive.diagnostics.linear_iterations
    )
    assert float(constant.diagnostics.final_forcing) == pytest.approx(1e-10)
    assert 1e-10 <= float(adaptive.diagnostics.final_forcing) < 0.5


def test_jacobian_refresh_policies_change_preparation_counts_without_losing_root():
    problem = nl.NonlinearSystemProblem(
        lambda state, args: state**2 - 2.0,
        problem_id="refresh-counts",
    )
    termination = nl.NonlinearTermination(
        absolute_residual=1e-6,
        relative_residual=1e-6,
        maximum_steps=20,
    )
    policies = {
        "every-step": nl.JacobianRefreshPolicy("every-step"),
        "stagnation": nl.JacobianRefreshPolicy("stagnation", residual_reduction=0.01),
        "periodic": nl.JacobianRefreshPolicy("periodic", period=3),
    }

    results = {
        name: nl.NewtonKrylov(
            forcing_policy=nl.NewtonForcingPolicy("constant"),
            jacobian_refresh=policy,
        ).solve(problem, jnp.asarray([1.0]), termination=termination)
        for name, policy in policies.items()
    }

    for result in results.values():
        assert bool(result.successful)
        assert jnp.allclose(result.state, jnp.sqrt(2.0), atol=2e-6)
    assert {
        name: int(result.diagnostics.jacobian_preparations)
        for name, result in results.items()
    } == {"every-step": 4, "stagnation": 3, "periodic": 2}


def test_rejection_refresh_reprepares_after_rejected_trust_steps_and_converges():
    problem = nl.NonlinearSystemProblem(
        lambda state, args: state**2 - 2.0,
        problem_id="rejection-refresh",
    )
    method = nl.NewtonTrustRegion(
        linear_policy=_gmres_policy(maximum_steps=1),
        forcing_policy=nl.NewtonForcingPolicy("constant"),
        jacobian_refresh=nl.JacobianRefreshPolicy("rejection"),
        trust_region=nl.RootTrustRegion(
            initial_radius=10.0,
            maximum_radius=10.0,
            maximum_attempts=1,
        ),
    )

    result = method.solve(
        problem,
        jnp.asarray([0.1]),
        termination=nl.NonlinearTermination(
            absolute_residual=1e-8,
            relative_residual=1e-8,
            maximum_steps=60,
        ),
    )

    rejected = int(result.diagnostics.rejected_steps)
    assert bool(result.successful)
    assert jnp.allclose(result.state, jnp.sqrt(2.0), atol=1e-7)
    assert rejected >= 2
    assert int(result.diagnostics.jacobian_preparations) == rejected + 1


def test_trust_region_dogleg_accepts_boundary_step_before_full_newton_fits():
    matrix = jnp.diag(jnp.asarray([1.0, 4.0]))
    expected = jnp.asarray([2.0, 1.0])
    target = matrix @ expected
    problem = nl.NonlinearSystemProblem(
        lambda state, args: matrix @ state - target,
        problem_id="dogleg-boundary",
    )
    method = nl.NewtonTrustRegion(
        linear_policy=_gmres_policy(maximum_steps=2),
        forcing_policy=nl.NewtonForcingPolicy("constant"),
        trust_region=nl.RootTrustRegion(initial_radius=1.5),
    )
    initial = jnp.zeros((2,))

    boundary = method.solve(
        problem,
        initial,
        termination=nl.NonlinearTermination(
            absolute_residual=1e-12,
            relative_residual=1e-12,
            maximum_steps=1,
        ),
    )
    converged = method.solve(
        problem,
        initial,
        termination=nl.NonlinearTermination(
            absolute_residual=1e-12,
            relative_residual=1e-12,
            maximum_steps=5,
        ),
    )

    assert int(boundary.status) == int(nl.NonlinearStatus.MAXIMUM_STEPS_REACHED)
    assert int(boundary.diagnostics.accepted_steps) == 1
    assert int(boundary.diagnostics.rejected_steps) == 0
    assert jnp.allclose(jnp.linalg.norm(boundary.state), 1.5, atol=1e-6)
    assert not jnp.allclose(boundary.state, expected)
    assert float(boundary.diagnostics.final_residual_norm) < float(
        boundary.diagnostics.initial_residual_norm
    )
    assert float(boundary.diagnostics.final_trust_radius) > 1.5
    assert bool(converged.successful)
    assert int(converged.diagnostics.accepted_steps) == 2
    assert jnp.allclose(converged.state, expected, atol=1e-8)
    assert jnp.allclose(converged.residual, jnp.zeros((2,)), atol=1e-10)


def test_residual_evaluation_cap_is_hard_under_jit():
    problem = nl.NonlinearSystemProblem(
        lambda state, args: state - 2.0,
        validity=lambda state, residual, auxiliary, args: jnp.all(state <= 0.0),
        problem_id="evaluation-cap",
    )
    method = nl.NewtonKrylov(line_search=nl.RootLineSearch(maximum_steps=12))
    termination = nl.NonlinearTermination(
        maximum_steps=10,
        maximum_evaluations=2,
    )

    def solve(initial):
        result = method.solve(problem, initial, termination=termination)
        return (
            result.status,
            result.diagnostics.residual_evaluations,
            result.diagnostics.rejected_steps,
            result.state,
            result.residual,
        )

    status, evaluations, rejected, state, residual = jax.jit(solve)(jnp.asarray([0.0]))

    assert int(status) == int(nl.NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert int(evaluations) == termination.maximum_evaluations
    assert int(evaluations) <= termination.maximum_evaluations
    assert int(rejected) == 1
    assert jnp.allclose(state, jnp.asarray([0.0]))
    assert jnp.allclose(residual, jnp.asarray([-2.0]))


def test_total_inner_iteration_cap_is_hard_under_jit():
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 4.0, 8.0]))
    expected = jnp.ones((4,))
    target = matrix @ expected
    problem = nl.NonlinearSystemProblem(
        lambda state, args: matrix @ state - target,
        problem_id="linear-iteration-cap",
    )
    method = nl.NewtonKrylov(
        linear_policy=_gmres_policy(maximum_steps=4),
        forcing_policy=nl.NewtonForcingPolicy("constant"),
        jacobian_refresh=nl.JacobianRefreshPolicy("periodic", period=100),
    )
    termination = nl.NonlinearTermination(
        absolute_residual=1e-12,
        relative_residual=1e-12,
        maximum_steps=10,
        maximum_linear_iterations=3,
    )

    def solve(initial):
        result = method.solve(problem, initial, termination=termination)
        return (
            result.status,
            result.diagnostics.linear_iterations,
            result.state,
            result.diagnostics.final_residual_norm,
        )

    status, iterations, state, residual_norm = jax.jit(solve)(jnp.zeros((4,)))

    assert int(status) == int(nl.NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED)
    assert int(iterations) == termination.maximum_linear_iterations
    assert int(iterations) <= termination.maximum_linear_iterations
    assert not jnp.allclose(state, expected, atol=1e-8)
    assert float(residual_norm) > termination.residual_threshold(jnp.linalg.norm(target))
