#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _termination(*, steps=100, evaluations=1000, tolerance=1e-6):
    return phx.optim.OptimizationTermination(
        absolute_optimality=tolerance,
        relative_optimality=0.0,
        absolute_step=1e-12,
        relative_step=1e-12,
        maximum_steps=steps,
        maximum_evaluations=evaluations,
    )


def test_strong_wolfe_reports_and_satisfies_both_inequalities():
    def objective(parameters):
        return 0.5 * jnp.sum((parameters - 3.0) ** 2)

    parameters = jnp.array([0.0, 1.0])
    value, gradient = jax.value_and_grad(objective)(parameters)
    direction = -gradient
    policy = phx.optim.StrongWolfeLineSearch(
        sufficient_decrease=1e-4,
        curvature=0.8,
    )
    result = phx.optim.strong_wolfe_line_search(
        jax.value_and_grad(objective),
        parameters,
        value,
        gradient,
        direction,
        step=lambda base, tangent, rate: base + rate * tangent,
        contains=lambda candidate: jnp.all(jnp.isfinite(candidate)),
        policy=policy,
    )

    initial_directional = jnp.vdot(gradient, direction).real
    candidate_directional = jnp.vdot(result.gradient, direction).real
    assert bool(result.accepted)
    assert bool(result.sufficient_decrease_satisfied)
    assert bool(result.curvature_satisfied)
    assert (
        result.value
        <= value + policy.sufficient_decrease * result.rate * initial_directional
    )
    assert jnp.abs(candidate_directional) <= policy.curvature * jnp.abs(
        initial_directional
    )


def test_newton_trust_region_preserves_pytree_and_reports_ratio_radius():
    target = {"left": jnp.array([1.0, -2.0]), "right": jnp.array([0.5])}

    def objective(parameters, args):
        del args
        return 0.5 * sum(
            jnp.sum((leaf - target[name]) ** 2) for name, leaf in parameters.items()
        )

    initial = {"left": jnp.array([5.0, 3.0]), "right": jnp.array([-4.0])}
    result = phx.optim.minimize(
        objective,
        initial,
        method=phx.optim.NewtonTrustRegion(initial_radius=0.5),
        termination=_termination(),
    )

    assert bool(result.successful)
    assert jax.tree.structure(result.parameters) == jax.tree.structure(initial)
    assert jnp.allclose(result.parameters["left"], target["left"], atol=1e-5)
    assert jnp.isfinite(result.diagnostics.reduction_ratio)
    assert result.diagnostics.damping > 0.0
    assert result.provenance.globalization == "trust-region-ratio"


@pytest.mark.parametrize(
    "beta_method",
    [
        "fletcher-reeves",
        "polak-ribiere+",
        "hestenes-stiefel+",
        "dai-yuan",
    ],
)
def test_nonlinear_conjugate_gradient_uses_selected_beta_without_restart(
    beta_method,
):
    def objective(parameters):
        x, y = parameters
        return 0.5 * x**2 + y**2 + 1e-3 * (x**4 + y**4)

    method = phx.optim.NonlinearConjugateGradient(
        beta_method=beta_method,
        orthogonality_restart=0.99,
        descent_safeguard=1e-12,
        line_search=phx.optim.StrongWolfeLineSearch(
            curvature=1e-3,
            maximum_steps=60,
        ),
    )
    parameters = jnp.array([0.25, 1.0])
    state = method.prepare_state(objective, parameters)
    next_parameters, next_state, _ = method.step(
        objective,
        parameters,
        state,
        termination=None,
    )

    old_gradient = state.gradient
    new_gradient = next_state.gradient
    difference = new_gradient - old_gradient
    new_squared = jnp.vdot(new_gradient, new_gradient).real
    old_squared = jnp.vdot(old_gradient, old_gradient).real
    numerator = jnp.vdot(new_gradient, difference).real
    denominator = jnp.vdot(state.direction, difference).real
    if beta_method == "fletcher-reeves":
        expected_beta = new_squared / old_squared
    elif beta_method == "polak-ribiere+":
        expected_beta = jnp.maximum(0.0, numerator / old_squared)
    elif beta_method == "hestenes-stiefel+":
        expected_beta = jnp.maximum(0.0, numerator / denominator)
    else:
        expected_beta = new_squared / denominator
    expected_direction = -new_gradient + expected_beta * state.direction

    assert bool(next_state.metrics.accepted)
    assert not bool(next_state.metrics.direction_fallback)
    assert not jnp.allclose(new_gradient, old_gradient)
    assert jnp.abs(expected_beta) > 1e-8
    assert jnp.allclose(next_state.direction, expected_direction, rtol=2e-5)
    assert not jnp.allclose(next_parameters, parameters)


def test_nonlinear_conjugate_gradient_forced_restart_uses_steepest_descent():
    def objective(parameters):
        return 0.25 * jnp.sum(parameters**4) + 0.5 * jnp.sum(parameters**2)

    method = phx.optim.NonlinearConjugateGradient(
        beta_method="fletcher-reeves",
        orthogonality_restart=0.0,
    )
    parameters = jnp.array([1.25, -0.5])
    state = method.prepare_state(objective, parameters)
    _, next_state, _ = method.step(
        objective,
        parameters,
        state,
        termination=None,
    )

    assert bool(next_state.metrics.accepted)
    assert bool(next_state.metrics.direction_fallback)
    assert jnp.allclose(next_state.direction, -next_state.gradient)


def test_builtin_proximal_functionals_have_exact_observable_maps():
    vector = {"x": jnp.array([-2.0, -0.25, 3.0])}
    assert jnp.allclose(
        phx.optim.L1Functional(1.0).proximal(vector, 0.5)["x"],
        jnp.array([-1.5, 0.0, 2.5]),
    )
    assert jnp.allclose(
        phx.optim.ElasticNetFunctional(1.0, 2.0).proximal(vector, 0.5)["x"],
        jnp.array([-0.75, 0.0, 1.25]),
    )

    box = phx.optim.BoxIndicator(-1.0, 1.0)
    projected_box = box.proximal(vector, 1.0)
    assert jnp.allclose(projected_box["x"], jnp.array([-1.0, -0.25, 1.0]))
    assert jnp.isfinite(box.value(projected_box))
    assert jnp.isinf(box.value(vector))

    generic = phx.optim.IndicatorFunctional(
        lambda tree: jax.tree.map(jnp.maximum, tree, {"x": jnp.zeros(3)}),
        lambda tree: jnp.all(tree["x"] >= 0.0),
    )
    assert jnp.all(generic.proximal(vector, 1.0)["x"] >= 0.0)
    assert jnp.isinf(generic.value(vector))

    simplex = phx.optim.SimplexIndicator()
    simplex_point = simplex.proximal({"x": jnp.array([-1.0, 0.2, 2.0])}, 1.0)
    assert jnp.all(simplex_point["x"] >= 0.0)
    assert jnp.allclose(jnp.sum(simplex_point["x"]), 1.0)
    assert jnp.isfinite(simplex.value(simplex_point))

    grouped = phx.optim.GroupLassoFunctional(1.0, axis=-1).proximal(
        {"x": jnp.array([[3.0, 4.0], [0.0, 0.0]])},
        1.0,
    )
    assert jnp.allclose(grouped["x"][0], jnp.array([2.4, 3.2]))
    assert jnp.allclose(grouped["x"][1], 0.0)

    matrix = {"x": jnp.diag(jnp.array([2.0, 0.5]))}
    nuclear = phx.optim.NuclearNormFunctional(1.0)
    assert jnp.allclose(
        nuclear.proximal(matrix, 1.0)["x"],
        jnp.diag(jnp.array([1.0, 0.0])),
        atol=1e-6,
    )
    assert jnp.allclose(nuclear.value(matrix), 2.5)


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.ProximalGradient(),
        phx.optim.AcceleratedProximalGradient(),
        phx.optim.ProximalNewton(inner_steps=30),
    ],
)
def test_proximal_methods_report_composite_stationarity(method):
    target = jnp.array([2.0, -1.0, 0.1])

    def smooth(parameters, args):
        del args
        return 0.5 * jnp.vdot(parameters - target, parameters - target).real

    problem = phx.optim.ProximalProblem(
        smooth,
        phx.optim.L1Functional(0.25),
        problem_id="lasso-quadratic",
    )
    result = phx.optim.proximal_minimize(
        problem,
        jnp.zeros_like(target),
        method=method,
        termination=_termination(steps=200, evaluations=5000, tolerance=2e-5),
    )

    expected = jnp.sign(target) * jnp.maximum(jnp.abs(target) - 0.25, 0.0)
    assert bool(result.successful)
    assert jnp.allclose(result.parameters, expected, atol=2e-4)
    assert result.composite_stationarity <= 2e-5
    assert jnp.allclose(
        result.composite_stationarity,
        result.diagnostics.final_optimality_norm,
    )


def test_finite_difference_gauss_newton_works_with_stopped_residual_derivatives():
    def residual(parameters, args):
        del args
        return jax.lax.stop_gradient(parameters * parameters - 4.0)

    result = phx.optim.least_squares(
        residual,
        jnp.array([3.0]),
        method=phx.optim.FiniteDifferenceGaussNewton(),
        termination=_termination(steps=50, evaluations=1000, tolerance=2e-4),
    )

    assert bool(result.successful)
    assert jnp.allclose(result.parameters, jnp.array([2.0]), atol=2e-3)
    assert result.diagnostics.jvp_evaluations == 0
    assert result.diagnostics.vjp_evaluations == 0
    assert result.diagnostics.jacobian_evaluations >= 1
    assert not result.provenance.implicit_differentiation


def test_filter_and_soc_accept_full_step_rejected_by_plain_merit_model():
    filter_policy = phx.optim.FilterGlobalization(
        objective_margin=0.9,
        violation_margin=1e-4,
    )
    filter_objectives = jnp.array([0.0, jnp.inf])
    filter_violations = jnp.array([0.0, jnp.inf])
    assert not bool(
        filter_policy.acceptable(
            -2.0,
            4.0,
            filter_objectives,
            filter_violations,
            1,
        )
    )
    assert bool(
        filter_policy.acceptable(
            -1.2,
            0.8,
            filter_objectives,
            filter_violations,
            1,
        )
    )
    assert -2.0 + 10.0 * 4.0 > 0.0

    circle = phx.optim.NonlinearConstraint(
        lambda parameters, args: jnp.array([jnp.vdot(parameters, parameters).real]),
        lower=jnp.array([1.0]),
        upper=jnp.array([1.0]),
        constraint_id="unit-circle",
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, args: parameters[0],
        constraints=(circle,),
        problem_id="maratos-full-step",
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([0.0, 1.0]),
        method=phx.optim.SQP(
            filter_globalization=filter_policy,
            second_order_correction=True,
            hessian_scale=0.5,
        ),
        termination=_termination(steps=1, evaluations=200, tolerance=1e-10),
    )

    assert result.diagnostics.accepted_steps == 1
    assert jnp.allclose(result.diagnostics.accepted_step_size, 1.0)
    assert result.provenance.globalization == "objective-feasibility-filter-with-soc"
    assert result.diagnostics.primal_feasibility < 4.0
    assert jnp.allclose(
        result.diagnostics.final_step_norm,
        jnp.linalg.norm(result.parameters - jnp.array([0.0, 1.0])),
    )


def test_predictor_corrector_reports_all_kkt_residuals():
    constraint = phx.optim.NonlinearConstraint(
        lambda parameters, args: parameters,
        lower=jnp.array([1.0]),
        upper=jnp.array([jnp.inf]),
        constraint_id="lower-one",
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, args: 0.5 * jnp.sum((parameters - 2.0) ** 2),
        constraints=(constraint,),
        problem_id="predictor-corrector-quadratic",
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([1.5]),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-predictor-corrector",
        ),
        termination=_termination(steps=100, evaluations=2000, tolerance=2e-5),
    )

    assert bool(result.successful)
    assert result.certificate is not None
    assert jnp.isfinite(result.diagnostics.primal_feasibility)
    assert jnp.isfinite(result.diagnostics.dual_feasibility)
    assert jnp.isfinite(result.diagnostics.complementarity)
    assert jnp.allclose(
        result.diagnostics.primal_feasibility,
        result.certificate.primal_feasibility,
    )
    assert jnp.allclose(
        result.diagnostics.dual_feasibility,
        result.certificate.dual_feasibility,
    )
    assert jnp.allclose(
        result.diagnostics.complementarity,
        result.certificate.complementarity,
    )
    assert result.diagnostics.linear_solves >= 2
    assert result.provenance.globalization == "mehrotra-predictor-corrector-residual"


def test_predictor_corrector_rejects_nonfinite_and_infeasible_inputs_explicitly():
    finite_constraint = phx.optim.NonlinearConstraint(
        lambda parameters, args: parameters,
        lower=jnp.array([0.0]),
        upper=jnp.array([jnp.inf]),
    )
    nonfinite_problem = phx.optim.MinimizationProblem(
        lambda parameters, args: jnp.sum(parameters * parameters),
        constraints=(finite_constraint,),
    )
    nonfinite = phx.optim.minimize(
        nonfinite_problem,
        jnp.array([jnp.nan]),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-predictor-corrector",
        ),
        termination=_termination(steps=2, evaluations=100),
    )
    assert nonfinite.status == int(phx.optim.OptimizationStatus.NONFINITE_INPUT)

    lower = phx.optim.NonlinearConstraint(
        lambda parameters, args: parameters,
        lower=jnp.array([1.0]),
        upper=jnp.array([jnp.inf]),
        constraint_id="lower-one",
    )
    upper = phx.optim.NonlinearConstraint(
        lambda parameters, args: parameters,
        lower=jnp.array([-jnp.inf]),
        upper=jnp.array([0.0]),
        constraint_id="upper-zero",
    )
    infeasible_problem = phx.optim.MinimizationProblem(
        lambda parameters, args: jnp.sum(parameters * parameters),
        constraints=(lower, upper),
        problem_id="inconsistent-interval",
    )
    infeasible = phx.optim.minimize(
        infeasible_problem,
        jnp.array([0.5]),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-predictor-corrector",
            maximum_restoration_steps=4,
        ),
        termination=_termination(steps=10, evaluations=500),
    )
    assert infeasible.status == int(phx.optim.OptimizationStatus.INFEASIBLE)


def test_constrained_solver_accepts_traced_constraint_and_parameter_bounds():
    @eqx.filter_jit
    def solve(
        constraint_lower,
        constraint_upper,
        parameter_lower,
        parameter_upper,
        initial,
    ):
        constraint = phx.optim.NonlinearConstraint(
            lambda parameters, args: parameters,
            lower=constraint_lower,
            upper=constraint_upper,
            constraint_id="dynamic-interval",
        )
        problem = phx.optim.MinimizationProblem(
            lambda parameters, args: jnp.sum((parameters - 0.25) ** 2),
            constraints=(constraint,),
            bounds=phx.optim.Bounds(parameter_lower, parameter_upper),
        )
        return phx.optim.minimize(
            problem,
            initial,
            method=phx.optim.SQP(),
            termination=_termination(steps=3, evaluations=100),
        )

    result = solve(
        jnp.array([0.0]),
        jnp.array([1.0]),
        jnp.array([-2.0]),
        jnp.array([2.0]),
        jnp.array([0.5]),
    )

    assert jnp.all(jnp.isfinite(result.parameters))
    assert result.certificate is not None
    assert result.certificate.inequality_sources == (
        "constraint:0:0:lower",
        "bound:0:lower",
        "constraint:0:0:upper",
        "bound:0:upper",
    )


def test_jitted_dynamic_lower_excludes_known_infinite_upper_from_sqp():
    @eqx.filter_jit
    def solve(lower, initial):
        constraint = phx.optim.NonlinearConstraint(
            lambda parameters, args: parameters,
            lower=lower,
            upper=jnp.inf,
            constraint_id="dynamic-lower",
        )
        problem = phx.optim.MinimizationProblem(
            lambda parameters, args: jnp.sum((parameters - 2.0) ** 2),
            constraints=(constraint,),
        )
        return phx.optim.minimize(
            problem,
            initial,
            method=phx.optim.SQP(),
            termination=_termination(steps=20, evaluations=300),
        )

    result = solve(jnp.array([0.0]), jnp.array([1.0]))

    assert result.certificate is not None
    assert result.certificate.inequality_sources == ("constraint:0:0:lower",)
    assert jnp.all(jnp.isfinite(result.certificate.inequality_multipliers))
    assert jnp.all(jnp.isfinite(result.certificate.slacks))
    assert jnp.isfinite(result.diagnostics.complementarity)


def test_jitted_dynamic_upper_excludes_known_infinite_lower_from_primal_dual():
    @eqx.filter_jit
    def solve(upper, initial):
        constraint = phx.optim.NonlinearConstraint(
            lambda parameters, args: parameters,
            lower=-jnp.inf,
            upper=upper,
            constraint_id="dynamic-upper",
        )
        problem = phx.optim.MinimizationProblem(
            lambda parameters, args: jnp.sum((parameters - 2.0) ** 2),
            constraints=(constraint,),
        )
        return phx.optim.minimize(
            problem,
            initial,
            method=phx.optim.PrimalDualInteriorPoint(
                mode="matrix-free-predictor-corrector",
            ),
            termination=_termination(
                steps=100,
                evaluations=2000,
                tolerance=2e-5,
            ),
        )

    result = solve(jnp.array([1.0]), jnp.array([0.0]))

    assert result.certificate is not None
    assert result.certificate.inequality_sources == ("constraint:0:0:upper",)
    assert jnp.all(jnp.isfinite(result.certificate.inequality_multipliers))
    assert jnp.all(jnp.isfinite(result.certificate.slacks))
    assert jnp.isfinite(result.diagnostics.complementarity)


def test_filter_switches_to_armijo_at_feasible_point():
    inactive = phx.optim.NonlinearConstraint(
        lambda parameters, args: parameters,
        lower=jnp.array([-100.0]),
        upper=jnp.array([100.0]),
        constraint_id="inactive-interval",
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, args: jnp.sum(parameters**4),
        constraints=(inactive,),
    )
    initial = jnp.array([1.0])
    result = phx.optim.minimize(
        problem,
        initial,
        method=phx.optim.SQP(
            filter_globalization=phx.optim.FilterGlobalization(),
            hessian_scale=1.0,
        ),
        termination=_termination(steps=1, evaluations=100, tolerance=1e-12),
    )

    assert result.diagnostics.accepted_steps == 1
    assert result.diagnostics.accepted_step_size < 1.0
    assert result.objective < 1.0


def test_finite_difference_gauss_newton_honors_exact_remaining_budget():
    result = phx.optim.least_squares(
        lambda parameters, args: parameters - 1.0,
        jnp.array([3.0, -2.0]),
        method=phx.optim.FiniteDifferenceGaussNewton(),
        termination=_termination(steps=5, evaluations=16, tolerance=1e-12),
    )

    assert result.status == int(phx.optim.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert result.diagnostics.residual_evaluations == 16
    assert result.diagnostics.globalization_evaluations == 1


def test_monotone_fista_recomputes_worse_extrapolated_candidate():
    problem = phx.optim.ProximalProblem(
        lambda parameters, args: 0.5 * jnp.sum(parameters**2),
        phx.optim.L1Functional(0.0),
    )
    method = phx.optim.AcceleratedProximalGradient(initial_step_size=0.5)
    parameters = jnp.array([1.0])
    state = method.prepare_state(problem, parameters, args=None)
    state = eqx.tree_at(
        lambda current: current.extrapolated,
        state,
        jnp.array([10.0]),
    )
    next_parameters, next_state = method.step(
        problem,
        parameters,
        state,
        termination=_termination(steps=5, evaluations=100),
        args=None,
    )

    assert bool(next_state.metrics.accepted)
    assert bool(next_state.metrics.direction_fallback)
    assert next_state.objective <= state.objective
    assert jnp.allclose(next_parameters, jnp.array([0.5]))
    assert next_state.objective_evaluations - state.objective_evaluations == 4


def test_predictor_corrector_rejects_equality_only_problem():
    equality = phx.optim.NonlinearConstraint(
        lambda parameters, args: parameters,
        lower=jnp.array([0.0]),
        upper=jnp.array([0.0]),
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, args: jnp.sum(parameters**2),
        constraints=(equality,),
    )

    with pytest.raises(ValueError, match="requires at least one inequality"):
        phx.optim.minimize(
            problem,
            jnp.array([0.0]),
            method=phx.optim.PrimalDualInteriorPoint(
                mode="matrix-free-predictor-corrector",
            ),
            termination=_termination(steps=2, evaluations=100),
        )


def test_predictor_corrector_reports_unusable_kkt_direction():
    dtype = jnp.asarray(0.0).dtype
    kkt_space = phx.linalg.BlockSpace(
        (
            phx.linalg.ArraySpace((1,), dtype=dtype),
            phx.linalg.ArraySpace((0,), dtype=dtype),
        )
    )
    invalid_inverse = phx.linalg.FunctionLinearOperator(
        lambda blocks: (
            jnp.full_like(blocks[0], jnp.nan),
            jnp.full_like(blocks[1], jnp.nan),
        ),
        source=kkt_space,
        target=kkt_space,
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.MINRES(),
        preconditioning=phx.linalg.PreconditioningPolicy(
            phx.linalg.OperatorPreconditioner(
                invalid_inverse,
                positive_definite=True,
            )
        ),
        differentiation=phx.linalg.DifferentiationPolicy("none"),
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, args: jnp.sum((parameters - 2.0) ** 2),
        bounds=phx.optim.Bounds(0.0, jnp.inf),
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([1.0]),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-predictor-corrector",
            linear_policy=policy,
        ),
        termination=_termination(steps=2, evaluations=100),
    )

    assert result.status == int(phx.optim.OptimizationStatus.LINEAR_SOLVE_FAILED)
    assert result.status != int(phx.optim.OptimizationStatus.INFEASIBLE)


def test_predictor_corrector_reports_exhausted_line_search():
    curved = phx.optim.NonlinearConstraint(
        lambda parameters, args: parameters**2,
        lower=jnp.array([0.25]),
        upper=jnp.array([jnp.inf]),
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, args: 0.5 * jnp.sum((parameters - 2.0) ** 2),
        constraints=(curved,),
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([1.0]),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-predictor-corrector",
            sufficient_decrease=0.999999,
            maximum_line_search_steps=1,
        ),
        termination=_termination(steps=2, evaluations=100),
    )

    assert result.status == int(phx.optim.OptimizationStatus.LINE_SEARCH_FAILED)
    assert result.status != int(phx.optim.OptimizationStatus.INFEASIBLE)
