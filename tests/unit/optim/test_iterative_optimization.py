#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

import phydrax as phx


def _termination(*, steps=50, tolerance=1e-8):
    return phx.optim.OptimizationTermination(
        absolute_optimality=tolerance,
        relative_optimality=0.0,
        maximum_steps=steps,
    )


def test_minimization_problem_auxiliary_status_and_provenance_contracts():
    problem = phx.optim.MinimizationProblem(
        lambda value, shift: (jnp.sum((value - shift) ** 2), {"shift": shift}),
        has_aux=True,
        problem_id="auxiliary-quadratic",
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([0.0, 0.0]),
        method=phx.optim.OptimistixMethod(optx.BFGS(rtol=1e-10, atol=1e-10)),
        termination=_termination(),
        args=jnp.array([2.0, -1.0]),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([2.0, -1.0]), atol=1e-7)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.successful
    assert result.provenance.problem_id == "auxiliary-quadratic"
    assert result.provenance.backend == "optimistix"
    np.testing.assert_allclose(result.auxiliary["shift"], jnp.array([2.0, -1.0]))
    assert phx.optim.optimization_status_message(result.status) == "success"


def test_optimistix_adapter_supports_filtered_jit_status_paths():
    problem = phx.optim.MinimizationProblem(
        lambda value, target: jnp.sum((value - target) ** 2)
    )
    method = phx.optim.OptimistixMethod(optx.BFGS(rtol=1e-10, atol=1e-10))
    solve = eqx.filter_jit(
        lambda initial, target: phx.optim.minimize(
            problem,
            initial,
            method=method,
            termination=_termination(),
            args=target,
        )
    )

    successful = solve(jnp.array([0.0]), jnp.array([2.0]))
    nonfinite = solve(jnp.array([jnp.nan]), jnp.array([2.0]))

    np.testing.assert_allclose(successful.parameters, jnp.array([2.0]), atol=1e-7)
    assert int(successful.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert int(successful.diagnostics.accepted_steps) == -1
    assert int(nonfinite.status) == int(phx.optim.OptimizationStatus.NONFINITE_INPUT)
    assert jnp.isnan(nonfinite.parameters[0])


def test_optimistix_adapter_rejects_unenforceable_evaluation_budget():
    problem = phx.optim.MinimizationProblem(lambda value, _: jnp.sum(value**2))
    termination = phx.optim.OptimizationTermination(
        maximum_steps=10,
        maximum_evaluations=10,
    )

    with pytest.raises(ValueError, match="cannot enforce maximum_evaluations"):
        phx.optim.minimize(
            problem,
            jnp.array([1.0]),
            method=phx.optim.OptimistixMethod(optx.BFGS(rtol=1e-8, atol=1e-8)),
            termination=termination,
        )


def test_minimization_problem_rejects_non_scalar_and_constrained_backend_mismatch():
    vector_problem = phx.optim.MinimizationProblem(lambda value, _: value)
    with pytest.raises(TypeError, match="one real scalar"):
        vector_problem.value(jnp.ones(2))

    constrained = phx.optim.MinimizationProblem(
        lambda value, _: jnp.sum(value**2),
        bounds=phx.optim.Bounds(0.0, 1.0),
    )
    with pytest.raises(ValueError, match="does not translate"):
        phx.optim.minimize(
            constrained,
            jnp.array([0.5]),
            method=phx.optim.OptimistixMethod(optx.BFGS(rtol=1e-8, atol=1e-8)),
        )
    with pytest.raises(ValueError, match="unconstrained"):
        phx.optim.minimize(
            constrained,
            jnp.array([0.5]),
            method=phx.optim.NewtonKrylov(),
        )


@pytest.mark.parametrize(
    "method",
    [phx.optim.GaussNewton(), phx.optim.LevenbergMarquardt()],
)
def test_native_nonlinear_least_squares_methods_solve_nonlinear_residual(method):
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, _: jnp.array([value[0] ** 2 - 4.0, value[1] - 3.0]),
        problem_id="two-residuals",
    )
    result = phx.optim.least_squares(
        problem,
        jnp.array([1.0, 0.0]),
        method=method,
        termination=_termination(steps=30),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([2.0, 3.0]), atol=1e-6)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.objective < 1e-12
    assert result.diagnostics.residual_evaluations > 0
    assert result.provenance.matrix_free


def test_gauss_newton_handles_rectangular_rank_deficient_residual():
    result = phx.optim.least_squares(
        lambda value, _: jnp.array([value[0] + value[1] - 2.0]),
        jnp.array([0.0, 0.0]),
        method=phx.optim.GaussNewton(),
        termination=_termination(steps=20),
    )

    np.testing.assert_allclose(jnp.sum(result.parameters), 2.0, atol=1e-7)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.diagnostics.linear_solves >= 1


def test_newton_krylov_uses_descent_fallback_for_indefinite_hessian():
    result = phx.optim.minimize(
        lambda value, _: jnp.sum(value**4 - value**2),
        jnp.array([0.2]),
        method=phx.optim.NewtonKrylov(),
        termination=_termination(steps=40),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([2.0**-0.5]), atol=1e-5)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.diagnostics.direction_fallbacks >= 1
    assert result.diagnostics.hvp_evaluations >= 1
    assert result.provenance.method == "newton-krylov"


def test_newton_krylov_forcing_controls_inner_accuracy_under_jit():
    diagonal = jnp.logspace(0.0, 6.0, 24)

    def objective(value, _):
        return 0.5 * jnp.sum(diagonal * value**2)

    termination = _termination(steps=1, tolerance=0.0)
    initial = jnp.ones((24,))
    loose_method = phx.optim.NewtonKrylov(
        minimum_forcing=0.5,
        maximum_forcing=0.5,
    )
    tight_method = phx.optim.NewtonKrylov(
        minimum_forcing=1e-8,
        maximum_forcing=1e-8,
    )

    def solve(method):
        return phx.optim.minimize(
            objective,
            initial,
            method=method,
            termination=termination,
        )

    loose = solve(loose_method)
    tight = solve(tight_method)
    compiled_loose = eqx.filter_jit(lambda: solve(loose_method))()
    compiled_tight = eqx.filter_jit(lambda: solve(tight_method))()

    assert int(loose.diagnostics.linear_iterations) < int(
        tight.diagnostics.linear_iterations
    )
    assert float(tight.diagnostics.final_optimality_norm) < float(
        loose.diagnostics.final_optimality_norm
    )
    assert int(compiled_loose.diagnostics.linear_iterations) == int(
        loose.diagnostics.linear_iterations
    )
    assert int(compiled_tight.diagnostics.linear_iterations) == int(
        tight.diagnostics.linear_iterations
    )


def test_nonfinite_initial_parameters_return_typed_status():
    result = phx.optim.minimize(
        lambda value, _: jnp.sum(value**2),
        jnp.array([jnp.nan]),
        method=phx.optim.NewtonKrylov(),
        termination=_termination(),
    )

    assert result.status == phx.optim.OptimizationStatus.NONFINITE_INPUT
    assert not result.successful


def _finite_only_at_nonpositive_parameters(parameters):
    value = parameters[0]
    return jnp.where(value <= 0.0, (value - 1.0) ** 2, jnp.nan)


def _finite_only_at_nonpositive_residual(parameters):
    value = parameters[0]
    return jnp.where(
        value <= 0.0,
        jnp.asarray([value - 1.0]),
        jnp.asarray([jnp.nan]),
    )


def test_scalar_and_bound_line_searches_report_all_nonfinite_trials():
    search = phx.optim.ArmijoLineSearch(maximum_steps=2)
    unconstrained = phx.optim.minimize(
        lambda parameters, _: _finite_only_at_nonpositive_parameters(parameters),
        jnp.array([0.0]),
        method=phx.optim.NewtonKrylov(line_search=search),
        termination=_termination(steps=2, tolerance=0.0),
    )
    bounded = phx.optim.minimize(
        phx.optim.MinimizationProblem(
            lambda parameters, _: _finite_only_at_nonpositive_parameters(parameters),
            bounds=phx.optim.Bounds(-1.0, 2.0),
        ),
        jnp.array([0.0]),
        method=phx.optim.ProjectedGradient(line_search=search),
        termination=_termination(steps=2, tolerance=0.0),
    )

    for result in (unconstrained, bounded):
        assert result.status == phx.optim.OptimizationStatus.NONFINITE_EVALUATION
        np.testing.assert_array_equal(result.parameters, jnp.array([0.0]))
        assert result.diagnostics.rejected_steps == 1


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.GaussNewton(line_search=phx.optim.ArmijoLineSearch(maximum_steps=2)),
        phx.optim.LevenbergMarquardt(maximum_trials=2),
    ],
)
def test_least_squares_methods_report_all_nonfinite_trials(method):
    result = phx.optim.least_squares(
        lambda parameters, _: _finite_only_at_nonpositive_residual(parameters),
        jnp.array([0.0]),
        method=method,
        termination=_termination(steps=2, tolerance=0.0),
    )

    assert result.status == phx.optim.OptimizationStatus.NONFINITE_EVALUATION
    np.testing.assert_array_equal(result.parameters, jnp.array([0.0]))
    assert result.diagnostics.rejected_steps == 1


def test_composite_line_search_reports_all_nonfinite_trials():
    result = phx.optim.composite_least_squares(
        phx.optim.CompositeLeastSquaresProblem(
            lambda parameters, _: _finite_only_at_nonpositive_residual(parameters),
            lambda parameters, _: 0.05 * jnp.sum(parameters**2),
        ),
        jnp.array([0.0]),
        method=phx.optim.GeneralizedGaussNewton(
            line_search=phx.optim.ArmijoLineSearch(maximum_steps=2)
        ),
        termination=_termination(steps=2, tolerance=0.0),
    )

    assert result.status == phx.optim.OptimizationStatus.NONFINITE_EVALUATION
    np.testing.assert_array_equal(result.parameters, jnp.array([0.0]))
    assert result.diagnostics.rejected_steps == 1
