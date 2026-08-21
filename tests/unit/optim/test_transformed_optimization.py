#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import linalg as la


class _WrappedParameters(eqx.Module):
    weight: jax.Array
    scale: float = eqx.field(static=True)


def _termination(*, maximum_steps=30, maximum_evaluations=None):
    return phx.optim.OptimizationTermination(
        absolute_optimality=1e-10,
        relative_optimality=0.0,
        maximum_steps=maximum_steps,
        maximum_evaluations=maximum_evaluations,
    )


def test_newton_krylov_eager_and_compiled_results_agree():
    problem = phx.optim.MinimizationProblem(
        lambda parameters, target: jnp.sum((parameters - target) ** 2)
    )
    method = phx.optim.NewtonKrylov()
    termination = _termination()

    def solve(target):
        return phx.optim.minimize(
            problem,
            jnp.array([-2.0, 4.0]),
            method=method,
            termination=termination,
            args=target,
        )

    target = jnp.array([1.5, -0.5])
    eager = solve(target)
    compiled = eqx.filter_jit(solve)(target)

    np.testing.assert_allclose(compiled.parameters, eager.parameters, atol=1e-9)
    assert (
        int(compiled.status)
        == int(eager.status)
        == int(phx.optim.OptimizationStatus.SUCCESS)
    )
    assert int(compiled.diagnostics.iterations) == int(eager.diagnostics.iterations)
    assert int(compiled.diagnostics.accepted_steps) == int(
        eager.diagnostics.accepted_steps
    )


def test_native_methods_compile_with_nested_parameters():
    initial = {"left": jnp.array([-1.0]), "right": (jnp.array([3.0]),)}
    target = {"left": jnp.array([2.0]), "right": (jnp.array([-4.0]),)}

    def objective(parameters, desired):
        return jnp.sum((parameters["left"] - desired["left"]) ** 2) + jnp.sum(
            (parameters["right"][0] - desired["right"][0]) ** 2
        )

    solve = eqx.filter_jit(
        lambda desired: phx.optim.minimize(
            objective,
            initial,
            method=phx.optim.NewtonKrylov(),
            termination=_termination(),
            args=desired,
        )
    )
    result = solve(target)

    np.testing.assert_allclose(result.parameters["left"], target["left"], atol=1e-8)
    np.testing.assert_allclose(
        result.parameters["right"][0],
        target["right"][0],
        atol=1e-8,
    )


def test_native_solver_compiles_with_partitioned_equinox_parameters():
    wrapped = _WrappedParameters(jnp.array([0.0]), 2.0)
    initial, static = eqx.partition(wrapped, eqx.is_inexact_array)

    def objective(parameters, target):
        model = eqx.combine(parameters, static)
        return jnp.sum((model.scale * model.weight - target) ** 2)

    result = eqx.filter_jit(
        lambda target: phx.optim.minimize(
            objective,
            initial,
            method=phx.optim.NewtonKrylov(),
            termination=_termination(maximum_steps=100_000),
            args=target,
        )
    )(jnp.array([6.0]))

    fitted = eqx.combine(result.parameters, static)
    np.testing.assert_allclose(fitted.weight, jnp.array([3.0]), atol=1e-8)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)


def test_gauss_newton_and_lm_compile_with_dynamic_data():
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda parameters, target: jnp.array(
            [parameters[0] - target, 2.0 * parameters[0] - 2.0 * target]
        )
    )

    for method in (phx.optim.GaussNewton(), phx.optim.LevenbergMarquardt()):
        solve = eqx.filter_jit(
            lambda target: phx.optim.least_squares(
                problem,
                jnp.array([0.0]),
                method=method,
                termination=_termination(),
                args=target,
            )
        )
        result = solve(jnp.array(2.5))
        np.testing.assert_allclose(result.parameters, jnp.array([2.5]), atol=1e-8)
        assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)


def test_native_solvers_vmap_over_dynamic_problem_data():
    scalar_problem = phx.optim.MinimizationProblem(
        lambda parameters, target: jnp.sum((parameters - target) ** 2)
    )
    residual_problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda parameters, target: parameters - target
    )
    targets = jnp.array([[1.0], [2.0], [-3.0]])

    scalar = eqx.filter_jit(
        jax.vmap(
            lambda target: (
                phx.optim.minimize(
                    scalar_problem,
                    jnp.array([0.0]),
                    method=phx.optim.NewtonKrylov(),
                    termination=_termination(),
                    args=target,
                ).parameters
            )
        )
    )(targets)
    residual = eqx.filter_jit(
        jax.vmap(
            lambda target: (
                phx.optim.least_squares(
                    residual_problem,
                    jnp.array([0.0]),
                    method=phx.optim.GaussNewton(),
                    termination=_termination(),
                    args=target,
                ).parameters
            )
        )
    )(targets)

    np.testing.assert_allclose(scalar, targets, atol=1e-8)
    np.testing.assert_allclose(residual, targets, atol=1e-8)


def test_native_solvers_jvp_matches_dynamic_solution_map():
    termination = _termination()
    scalar_problem = phx.optim.MinimizationProblem(
        lambda parameters, target: jnp.sum((parameters - target) ** 2)
    )
    residual_problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda parameters, target: parameters - target
    )

    scalar_map = lambda target: (
        phx.optim.minimize(
            scalar_problem,
            jnp.array([0.0]),
            method=phx.optim.NewtonKrylov(),
            termination=termination,
            args=target,
        ).parameters
    )
    residual_map = lambda target: (
        phx.optim.least_squares(
            residual_problem,
            jnp.array([0.0]),
            method=phx.optim.GaussNewton(),
            termination=termination,
            args=target,
        ).parameters
    )
    target = jnp.array([2.0])
    tangent = jnp.array([0.75])

    for solution_map in (scalar_map, residual_map):
        value, derivative = jax.jvp(solution_map, (target,), (tangent,))
        np.testing.assert_allclose(value, target, atol=1e-8)
        np.testing.assert_allclose(derivative, tangent, atol=1e-8)


def test_compiled_termination_statuses_are_array_driven():
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: jnp.sum((parameters - 3.0) ** 4)
    )
    result = eqx.filter_jit(
        lambda initial: phx.optim.minimize(
            problem,
            initial,
            method=phx.optim.NewtonKrylov(),
            termination=_termination(maximum_steps=1),
        )
    )(jnp.array([0.0]))

    assert int(result.status) == int(phx.optim.OptimizationStatus.MAXIMUM_STEPS_REACHED)
    assert int(result.diagnostics.iterations) == 1


def test_compiled_nonfinite_input_is_reported_without_tracer_conversion():
    problem = phx.optim.MinimizationProblem(lambda parameters, _: jnp.sum(parameters**2))
    result = eqx.filter_jit(
        lambda initial: phx.optim.minimize(
            problem,
            initial,
            method=phx.optim.NewtonKrylov(),
            termination=_termination(),
        )
    )(jnp.array([jnp.nan]))

    assert int(result.status) == int(phx.optim.OptimizationStatus.NONFINITE_INPUT)
    assert int(result.diagnostics.iterations) == 0


def test_lm_rejected_trials_preserve_the_last_accepted_parameters():
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda parameters, _: jnp.where(
            parameters[0] > 0.0,
            jnp.array([jnp.nan]),
            jnp.array([parameters[0] - 1.0]),
        )
    )
    result = eqx.filter_jit(
        lambda initial: phx.optim.least_squares(
            problem,
            initial,
            method=phx.optim.LevenbergMarquardt(maximum_trials=4),
            termination=_termination(maximum_steps=5),
        )
    )(jnp.array([0.0]))

    np.testing.assert_array_equal(result.parameters, jnp.array([0.0]))
    assert int(result.status) == int(phx.optim.OptimizationStatus.NONFINITE_EVALUATION)
    assert int(result.diagnostics.accepted_steps) == 0
    assert int(result.diagnostics.rejected_steps) == 1


def test_least_squares_initial_setup_respects_evaluation_budget_status():
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda parameters, _: parameters - 2.0
    )
    result = eqx.filter_jit(
        lambda initial: phx.optim.least_squares(
            problem,
            initial,
            method=phx.optim.GaussNewton(),
            termination=_termination(
                maximum_steps=100_000,
                maximum_evaluations=1,
            ),
        )
    )(jnp.array([0.0]))

    np.testing.assert_array_equal(result.parameters, jnp.array([0.0]))
    assert int(result.status) == int(
        phx.optim.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
    )
    assert int(result.diagnostics.accepted_steps) == 0


def test_native_curvature_methods_report_observed_prepared_refresh_reuse():
    methods = (
        phx.optim.NewtonKrylov(),
        phx.optim.GaussNewton(),
        phx.optim.LevenbergMarquardt(),
        phx.optim.GeneralizedGaussNewton(),
    )

    assert all(method.capabilities.matrix_free for method in methods)
    assert all(method.capabilities.prepared_refresh for method in methods)
    assert all(method.capabilities.implicit_differentiation for method in methods[:3])


def test_prepared_refresh_diagnostics_match_linear_solve_lifecycle():
    termination = _termination()
    scalar = phx.optim.minimize(
        lambda parameters, _: jnp.sum((parameters - 2.0) ** 2),
        jnp.array([0.0]),
        method=phx.optim.NewtonKrylov(),
        termination=termination,
    )
    residual_problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda parameters, _: jnp.array([parameters[0] - 2.0, 2.0 * parameters[0] - 4.0])
    )
    residual_results = tuple(
        phx.optim.least_squares(
            residual_problem,
            jnp.array([0.0]),
            method=method,
            termination=termination,
        )
        for method in (phx.optim.GaussNewton(), phx.optim.LevenbergMarquardt())
    )
    composite = phx.optim.composite_least_squares(
        phx.optim.CompositeLeastSquaresProblem(
            lambda parameters, _: parameters - 2.0,
            lambda parameters, _: 0.1 * jnp.sum(parameters**2),
        ),
        jnp.array([0.0]),
        termination=termination,
    )

    for result in (scalar, *residual_results, composite):
        assert int(result.diagnostics.setup_refreshes) == 1
        assert int(result.diagnostics.numeric_refreshes) == (
            int(result.diagnostics.linear_solves) + 1
        )


def test_callable_preconditioner_stays_outside_staged_refresh_carries():
    parameters = jnp.array([0.0])
    space = la.PyTreeSpace(parameters)
    inverse = la.FunctionLinearOperator(
        lambda vector: vector,
        source=space,
        target=space,
        operator_id="optimizer-test-identity-inverse",
    )
    policy = la.LinearSolvePolicy(
        la.MINRES(),
        tolerance=la.TolerancePolicy(
            absolute=1e-12,
            relative=1e-10,
            max_steps=8,
        ),
        preconditioning=la.PreconditioningPolicy(
            la.OperatorPreconditioner(inverse, positive_definite=True)
        ),
    )
    termination = _termination(maximum_steps=12)
    scalar_method = phx.optim.NewtonKrylov(linear_policy=policy)
    state = scalar_method.init(parameters)
    assert all(eqx.is_array(leaf) for leaf in jax.tree.leaves(state))

    scalar = eqx.filter_jit(
        lambda initial: phx.optim.minimize(
            phx.optim.MinimizationProblem(lambda value, _: jnp.sum((value - 2.0) ** 2)),
            initial,
            method=scalar_method,
            termination=termination,
        )
    )(parameters)
    composite = eqx.filter_jit(
        lambda initial: phx.optim.composite_least_squares(
            phx.optim.CompositeLeastSquaresProblem(
                lambda value, _: value - 2.0,
                lambda value, _: 0.1 * jnp.sum(value**2),
            ),
            initial,
            method=phx.optim.GeneralizedGaussNewton(linear_policy=policy),
            termination=termination,
        )
    )(parameters)

    np.testing.assert_allclose(scalar.parameters, jnp.array([2.0]), atol=1e-8)
    np.testing.assert_allclose(
        composite.parameters,
        jnp.array([2.0 / 1.2]),
        atol=1e-7,
    )
    for result in (scalar, composite):
        assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
        assert int(result.diagnostics.numeric_refreshes) == (
            int(result.diagnostics.linear_solves) + 1
        )


def test_native_prepared_methods_preserve_float32_parameter_carries():
    initial = jnp.array([0.0], dtype=jnp.float32)
    target = jnp.array([2.0], dtype=jnp.float32)
    termination = _termination(maximum_steps=12)

    scalar = eqx.filter_jit(
        lambda value: phx.optim.minimize(
            phx.optim.MinimizationProblem(
                lambda parameters, desired: jnp.sum((parameters - desired) ** 2)
            ),
            value,
            method=phx.optim.NewtonKrylov(),
            termination=termination,
            args=target,
        )
    )(initial)
    residual = tuple(
        eqx.filter_jit(
            lambda value, method=method: phx.optim.least_squares(
                phx.optim.NonlinearLeastSquaresProblem(
                    lambda parameters, desired: parameters - desired
                ),
                value,
                method=method,
                termination=termination,
                args=target,
            )
        )(initial)
        for method in (
            phx.optim.GaussNewton(),
            phx.optim.LevenbergMarquardt(),
        )
    )
    composite = eqx.filter_jit(
        lambda value: phx.optim.composite_least_squares(
            phx.optim.CompositeLeastSquaresProblem(
                lambda parameters, desired: parameters - desired,
                lambda parameters, _: 0.05 * jnp.sum(parameters**2),
            ),
            value,
            method=phx.optim.GeneralizedGaussNewton(),
            termination=termination,
            args=target,
        )
    )(initial)

    for result in (scalar, *residual, composite):
        assert result.parameters.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(result.parameters))
