#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

import phydrax as phx


opt = phx.optim


def _termination(maximum_steps: Any = 100) -> Any:
    return opt.OptimizationTermination(
        absolute_optimality=1e-6,
        relative_optimality=0.0,
        maximum_steps=maximum_steps,
        maximum_evaluations=5000,
    )


def test_best_in_class_nonlinear_optimization_scenario_1() -> None:
    parameter = opt.ParameterBlock(
        lambda value: value["x"],
        lambda value, replacement: {"x": replacement},
        block_id="x",
    )
    residual = opt.ResidualBlock(
        lambda values, target: values[0] - target,
        ("x",),
        weight=2.0,
        loss=opt.HuberLoss(1.0),
        block_id="fit",
    )
    graph = opt.ResidualGraphProblem((parameter,), (residual,))
    prepared = opt.prepare_residual_graph(
        graph,
        {"x": jnp.zeros((2,))},
        args=jnp.asarray([1.0, 2.0]),
    )
    result = opt.least_squares(
        graph.as_least_squares_problem(),
        {"x": jnp.zeros((2,))},
        args=jnp.asarray([1.0, 2.0]),
        method=opt.LevenbergMarquardt(),
        termination=_termination(),
    )
    certificate = opt.factor_graph_certificate(
        graph,
        result.parameters,
        jnp.asarray([1.0, 2.0]),
    )

    assert prepared.adjacency.tolist() == [[True]]
    assert bool(result.successful)
    assert bool(certificate.certified)
    assert opt.CauchyLoss().evaluate(4.0).first < 1.0
    assert opt.TukeyLoss().evaluate(4.0).first == 0.0
    eliminated = opt.ParameterBlock(
        lambda value: value[:1],
        lambda value, replacement: value.at[:1].set(replacement),
        block_id="eliminated",
        elimination_group=0,
    )
    retained = opt.ParameterBlock(
        lambda value: value[1:],
        lambda value, replacement: value.at[1:].set(replacement),
        block_id="retained",
        elimination_group=1,
    )
    residual = opt.ResidualBlock(
        lambda values, args: jnp.asarray([values[0][0] + values[1][0] - 1.0]),
        ("eliminated", "retained"),
        block_id="factor",
    )
    graph = opt.prepare_residual_graph(
        opt.ResidualGraphProblem((eliminated, retained), (residual,)),
        jnp.zeros((2,)),
    )
    route = opt.plan_least_squares_route(
        graph,
        policy=opt.LeastSquaresRoutePolicy(dense_dimension=1),
    )
    schur = opt.prepare_schur_plan(graph)
    step = opt.solve_schur_system(
        jnp.asarray([[2.0, 1.0], [1.0, 3.0]]),
        jnp.asarray([-1.0, -2.0]),
        schur,
    )

    assert route.route == "schur"
    assert jnp.allclose(step, jnp.asarray([0.2, 0.6]))
    eliminated = opt.ParameterBlock(
        lambda value: value[:1],
        lambda value, replacement: value.at[:1].set(replacement),
        block_id="eliminated",
        elimination_group=0,
    )
    retained = opt.ParameterBlock(
        lambda value: value[1:],
        lambda value, replacement: value.at[1:].set(replacement),
        block_id="retained",
        elimination_group=1,
    )
    coupled = opt.ResidualBlock(
        lambda values, args: values[0] + values[1] - 1.0,
        ("eliminated", "retained"),
        block_id="coupled",
    )
    anchor = opt.ResidualBlock(
        lambda values, args: values[0] - 0.25,
        ("eliminated",),
        block_id="anchor",
    )
    graph = opt.ResidualGraphProblem(
        (eliminated, retained),
        (coupled, anchor),
        problem_id="routed-graph",
    )
    result = opt.solve_residual_graph(
        graph,
        jnp.zeros(2),
        termination=opt.OptimizationTermination(
            absolute_optimality=1e-8,
            relative_optimality=0.0,
            maximum_steps=20,
        ),
        route_policy=opt.LeastSquaresRoutePolicy(dense_dimension=1),
    )

    robust_parameter = opt.ParameterBlock(
        lambda value: value,
        lambda value, replacement: replacement,
        block_id="parameter",
    )
    robust_block = opt.ResidualBlock(
        lambda values, args: jnp.asarray(
            [values[0][0] - 10.0, 2.0 * values[0][0] - 20.0]
        ),
        ("parameter",),
        loss=opt.CauchyLoss(1.0),
        block_id="outlier",
    )
    robust_model = opt.linearize_residual_graph(
        opt.ResidualGraphProblem(
            (robust_parameter,),
            (robust_block,),
        ),
        jnp.zeros(1),
    )

    assert bool(result.successful)
    assert jnp.allclose(result.parameters, jnp.asarray([0.25, 0.75]), atol=1e-6)
    assert result.method_evidence.route == "schur"
    assert result.method_evidence.schur_plan_id
    assert int(result.method_evidence.linear_solves) > 0
    assert int(robust_model.robust_blocks) == 1
    assert int(robust_model.clipped_curvature_blocks) == 1
    assert float(jnp.min(jnp.linalg.eigvalsh(robust_model.curvature))) >= -1e-12


def test_best_in_class_nonlinear_optimization_scenario_2() -> None:
    for method in [
        opt.DoglegLeastSquares(),
        opt.DoglegLeastSquares("subspace"),
        opt.DoglegLeastSquares("dogbox"),
    ]:
        problem = opt.NonlinearLeastSquaresProblem(
            lambda parameters, target: parameters - target,
            bounds=(opt.Bounds(0.0, 1.0) if method.mode == "dogbox" else None),
        )
        target = (
            jnp.asarray([2.0, -0.5])
            if method.mode == "dogbox"
            else jnp.asarray([1.0, 0.5])
        )
        result = opt.least_squares(
            problem,
            jnp.asarray([0.2, 0.2]),
            args=target,
            method=method,
            termination=_termination(),
        )
        expected = jnp.asarray([1.0, 0.0]) if method.mode == "dogbox" else target
        assert bool(result.successful)
        assert jnp.allclose(result.parameters, expected, atol=1e-6)
    time = jnp.linspace(0.0, 1.0, 20)
    observations = 3.0 * jnp.exp(-2.0 * time)
    problem = opt.VariableProjectionProblem(
        lambda nonlinear, args: jnp.exp(-nonlinear[0] * time)[:, None],
        observations,
    )
    result = opt.variable_projection(
        problem,
        jnp.asarray([1.0]),
        termination=_termination(),
    )
    assert bool(result.successful)
    assert jnp.allclose(result.nonlinear_parameters, jnp.asarray([2.0]), atol=1e-5)
    assert jnp.allclose(result.linear_parameters, jnp.asarray([3.0]), atol=1e-5)
    problem = opt.NonlinearLeastSquaresProblem(
        lambda parameters, target: jnp.asarray(
            [parameters[0] ** 2 - target[0], parameters[1] - target[1]]
        ),
        bounds=opt.Bounds(jnp.asarray([0.0, -5.0]), jnp.asarray([5.0, 5.0])),
    )
    result = opt.least_squares(
        problem,
        jnp.asarray([1.0, 0.0]),
        args=jnp.asarray([4.0, 3.0]),
        method=opt.POUNDERS(initial_radius=0.5),
        termination=_termination(),
    )
    assert bool(result.successful)
    assert jnp.allclose(result.parameters, jnp.asarray([2.0, 3.0]), atol=1e-4)
    problem = opt.NonlinearLeastSquaresProblem(
        lambda parameters, args: jnp.asarray(
            [
                parameters[0] + parameters[1] - 1.0,
                2.0 * parameters[0] + 2.0 * parameters[1] - 2.0,
            ]
        ),
        problem_id="rank-deficient",
    )
    result = opt.least_squares(
        problem,
        jnp.zeros(2),
        method=opt.POUNDERS(initial_radius=0.25),
        termination=opt.OptimizationTermination(
            absolute_optimality=1e-6,
            relative_optimality=0.0,
            maximum_steps=200,
            maximum_evaluations=5000,
        ),
    )

    assert result.status == int(opt.OptimizationStatus.CERTIFICATION_FAILED)
    # ty: ignore[unresolved-attribute]
    assert result.status_evidence.internal_status == int(opt.OptimizationStatus.SUCCESS)
    # ty: ignore[unresolved-attribute]
    assert bool(result.status_evidence.demoted)
    # ty: ignore[unresolved-attribute]
    assert not bool(result.optimality_certificate.certified)
    # ty: ignore[unresolved-attribute]
    assert float(result.optimality_certificate.optimality_norm) > 1e-6
    # ty: ignore[unresolved-attribute]
    assert result.optimality_certificate.kind == "derivative-free-stationarity"
    assert int(result.method_evidence.interpolation_rank) < 6


def test_least_squares_step_without_termination_uses_default_policy() -> None:
    for method in [opt.DoglegLeastSquares(), opt.POUNDERS(initial_radius=0.5)]:
        target = jnp.asarray([1.0, 0.5])

        def residual(parameters: Any) -> Any:
            return parameters - target

        parameters = jnp.asarray([0.2, 0.2])
        state = method.prepare_state(residual, parameters)
        next_parameters, next_state, objective = method.step(
            residual,
            parameters,
            state,
            termination=None,
        )

        assert int(next_state.iteration) == 1
        assert bool(jnp.isfinite(objective))
        assert float(jnp.linalg.norm(residual(next_parameters))) < float(
            jnp.linalg.norm(residual(parameters))
        )


def test_physical_stationarity_certificate_never_steps_outside_narrow_bounds() -> None:
    evaluated = []

    def residual(parameters: Any, args: Any) -> Any:
        evaluated.append(float(parameters[0]))
        return parameters - 0.5e-6

    problem = opt.NonlinearLeastSquaresProblem(
        residual,
        bounds=opt.Bounds(0.0, 1e-6),
        problem_id="narrow-bounds",
    )
    certificate = opt.certify_least_squares_physical(
        problem,
        jnp.asarray([0.0]),
        None,
        _termination(),
        certificate_step=1e-4,
    )

    assert bool(certificate.finite)
    assert evaluated
    assert min(evaluated) >= 0.0
    assert max(evaluated) <= 1e-6


def test_best_in_class_nonlinear_optimization_scenario_3() -> None:
    point = jnp.asarray([1.0, 0.0, 0.0])
    geometry = opt.ParameterGeometry(
        point,
        {"<root>": phx.metrix.SphereManifold(3)},
    )
    parameter = opt.ParameterBlock(
        lambda value: value,
        lambda value, replacement: replacement,
        geometry=geometry,
        block_id="sphere",
    )
    factor = opt.ResidualBlock(
        lambda values, target: values[0] - target,
        ("sphere",),
        block_id="target",
    )
    graph = opt.ResidualGraphProblem((parameter,), (factor,))
    moved = graph.retract(point, {"sphere": jnp.asarray([0.0, 0.1, 0.0])})
    incremental = opt.prepare_incremental_factor_graph(
        graph,
        point,
        args=point,
    )
    updated, evidence = opt.update_incremental_factor_graph(
        incremental,
        graph,
        moved,
        changed_factors=("target",),
        args=point,
    )

    assert jnp.allclose(jnp.linalg.norm(moved), 1.0)
    assert bool(graph.manifold_valid(moved))
    assert bool(evidence.affected_parameters[0])
    assert int(updated.update_count) == 1
    problem = _constrained_problem()
    target = jnp.asarray([0.2, 0.8])
    prepared = opt.prepare_constrained_model(
        problem,
        jnp.asarray([0.5, 0.5]),
        args=target,
    )
    evaluation = prepared.evaluate(jnp.asarray([0.5, 0.5]), target)
    assert float(evaluation.primal_feasibility) == 0.0

    for method in (
        opt.SQP(
            hessian_update="sr1",
            filter_globalization=opt.FilterGlobalization(),
        ),
        opt.SQP(
            hessian_update="exact",
            filter_globalization=opt.FilterGlobalization(),
        ),
    ):
        result = opt.minimize(
            problem,
            jnp.asarray([0.5, 0.5]),
            args=target,
            method=method,
            termination=_termination(),
        )
        assert bool(result.successful)
        assert jnp.allclose(result.parameters, target, atol=2e-5)

    interior_point = opt.minimize(
        problem,
        jnp.asarray([0.5, 0.5]),
        args=target,
        method=opt.PrimalDualInteriorPoint(
            mode="dense-filter",
        ),
        termination=_termination(),
    )
    assert interior_point.status == int(opt.OptimizationStatus.CERTIFICATION_FAILED)
    # ty: ignore[unresolved-attribute]
    assert interior_point.status_evidence.internal_status == int(
        opt.OptimizationStatus.SUCCESS
    )
    # ty: ignore[unresolved-attribute]
    assert bool(interior_point.status_evidence.demoted)
    # ty: ignore[unresolved-attribute]
    assert interior_point.optimality_certificate.kind == "active-kkt"
    # ty: ignore[unresolved-attribute]
    assert not bool(interior_point.optimality_certificate.certified)
    assert jnp.allclose(interior_point.parameters, target, atol=2e-5)
    assert int(interior_point.method_evidence.kkt_rhs_solves) == 2 * int(
        interior_point.method_evidence.kkt_factorizations
    )
    assert int(interior_point.method_evidence.kkt_factorization_reuses) == int(
        interior_point.method_evidence.kkt_factorizations
    )

    plan = opt.plan_kkt(2, 1)
    kkt = opt.solve_kkt(
        jnp.diag(jnp.asarray([2.0, 4.0])),
        jnp.asarray([[1.0, 1.0]]),
        jnp.asarray([-2.0, -4.0]),
        jnp.asarray([0.0]),
        plan,
    )
    assert bool(kkt.inertia_matches)
    assert int(kkt.inertia.positive) == 2
    assert int(kkt.inertia.negative) == 1
    factorization = opt.factor_kkt(
        jnp.diag(jnp.asarray([2.0, 4.0])),
        jnp.asarray([[1.0, 1.0]]),
        plan,
    )
    reused = opt.solve_factored_kkt(
        factorization,
        jnp.asarray([1.0, -1.0]),
        jnp.asarray([0.5]),
    )
    assert float(reused.residual_norm) <= 1e-12
    problem = _constrained_problem()
    parameters = jnp.asarray([0.2, 0.8])
    args = jnp.asarray([0.2, 0.8])
    tangent = jnp.asarray([1.0, 0.0])
    forward = opt.constrained_solution_jvp(
        problem,
        parameters,
        args,
        tangent,
    )
    reverse = opt.constrained_solution_vjp(
        problem,
        parameters,
        args,
        tangent,
    )
    assert bool(forward.regular)
    assert bool(reverse.regular)
    assert jnp.allclose(forward.value, jnp.asarray([0.5, -0.5]))
    assert jnp.allclose(reverse.value, jnp.asarray([0.5, -0.5]))
    problem = _constrained_problem()
    with pytest.raises(ValueError, match="parameter PyTree"):
        opt.constrained_solution_vjp(
            problem,
            jnp.asarray([0.2, 0.8]),
            jnp.asarray([0.2, 0.8]),
            jnp.asarray([1.0]),
        )


def _constrained_problem() -> Any:
    constraint = opt.NonlinearConstraint(
        lambda parameters, args: jnp.asarray([jnp.sum(parameters)]),
        lower=1.0,
        upper=1.0,
        constraint_id="sum",
    )
    return opt.MinimizationProblem(
        lambda parameters, target: jnp.sum((parameters - target) ** 2),
        bounds=opt.Bounds(0.0, jnp.inf),
        constraints=(constraint,),
        problem_id="constrained-quadratic",
    )


def test_best_in_class_nonlinear_optimization_scenario_4() -> None:
    bounds_problem = opt.MinimizationProblem(
        lambda parameters, target: jnp.sum((parameters - target) ** 2),
        bounds=opt.Bounds(-2.0, 2.0),
    )
    bobyqa = opt.minimize(
        bounds_problem,
        jnp.zeros((2,)),
        args=jnp.asarray([1.0, -1.0]),
        method=opt.BOBYQA(initial_radius=0.5),
        termination=_termination(200),
    )
    multistart = opt.multistart_minimize(
        opt.MinimizationProblem(
            lambda parameters, args: jnp.sum((parameters * parameters - 1.0) ** 2),
            bounds=opt.Bounds(-2.0, 2.0),
        ),
        jnp.asarray([0.1, 0.1]),
        policy=opt.MultiStartPolicy(count=4, seed=3),
        termination=opt.OptimizationTermination(
            maximum_steps=400,
            maximum_evaluations=4000,
        ),
    )
    scipy = opt.minimize(
        opt.MinimizationProblem(
            lambda parameters, target: jnp.sum((parameters - target) ** 2)
        ),
        jnp.zeros((2,)),
        args=jnp.asarray([1.0, 2.0]),
        method=opt.SciPyMinimize("BFGS", options={"gtol": 1e-10}),
        termination=_termination(),
    )
    ceres = opt.ceres_least_squares(
        opt.NonlinearLeastSquaresProblem(lambda parameters, target: parameters - target),
        jnp.zeros((2,)),
        lambda problem, parameters, args: (args, True, {"iterations": 1}),
        args=jnp.asarray([1.0, 2.0]),
    )

    assert bool(bobyqa.successful)
    assert bool(multistart.successful)
    assert bool(scipy.successful)
    assert bool(ceres.successful)
    problem = opt.NonlinearLeastSquaresProblem(
        lambda value, _: value - 2.0,
        bounds=opt.Bounds(0.0, 1.0),
    )
    with pytest.raises(ValueError, match="constrained KKT"):
        opt.implicit_least_squares(problem, jnp.asarray([0.5]))
    problem = opt.NonlinearLeastSquaresProblem(
        lambda value, _: value - 2.0,
        bounds=opt.Bounds(0.0, 1.0),
    )
    result = opt.ceres_least_squares(
        problem,
        jnp.asarray([0.5]),
        lambda _problem, _parameters, _args: (
            jnp.asarray([2.0]),
            True,
            "forged-success",
        ),
    )

    assert not bool(result.successful)
    assert result.diagnostics.primal_feasibility > 0.0


def test_cobyqa_concrete_host_bounds_and_nonlinear_values() -> None:
    calls = {"objective": 0, "constraint": 0}

    def objective(parameters: dict[str, object], args: object) -> float:
        calls["objective"] += 1
        coordinates = np.asarray(parameters["x"])
        assert np.all(np.abs(coordinates) <= 1.0)
        return float(np.sum((coordinates - np.asarray([0.25, -0.5])) ** 2))

    def constraint(
        parameters: dict[str, object], args: object
    ) -> dict[str, NDArray[np.float64]]:
        calls["constraint"] += 1
        return {"sum": np.asarray(np.sum(np.asarray(parameters["x"])), dtype=np.float64)}

    result = opt.minimize(
        opt.MinimizationProblem(
            objective,
            bounds=opt.Bounds(-1.0, 1.0),
            constraints=(opt.NonlinearConstraint(constraint, lower=-1.0, upper=1.0),),
        ),
        {"x": jnp.asarray([3.0, -3.0])},
        method=opt.COBYQA(initial_radius=0.5),
        termination=_termination(200),
    )

    assert bool(result.successful)
    assert np.allclose(np.asarray(result.parameters["x"]), [0.25, -0.5], atol=1e-4)
    assert int(result.diagnostics.objective_evaluations) == calls["objective"]
    assert int(result.diagnostics.constraint_evaluations) == calls["constraint"]
    assert float(result.diagnostics.primal_feasibility) == 0.0


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 6])
def test_model_based_hard_initial_and_poll_allowance(budget: int) -> None:
    calls = 0

    def objective(parameters: object, args: object) -> float:
        nonlocal calls
        calls += 1
        coordinate = float(np.asarray(parameters)[0])
        return 0.0 if coordinate == 0.0 else 1.0 + 0.1 * coordinate

    result = opt.minimize(
        opt.MinimizationProblem(objective, bounds=opt.Bounds(-1.0, 1.0)),
        jnp.zeros((1,)),
        method=opt.BOBYQA(),
        termination=opt.OptimizationTermination(
            absolute_optimality=1e-6,
            relative_optimality=0.0,
            maximum_evaluations=budget,
        ),
    )

    assert calls <= budget
    assert int(result.diagnostics.objective_evaluations) == calls
    assert int(result.status) == int(opt.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert np.isnan(float(result.diagnostics.final_optimality_norm))
    assert float(result.objective) == 0.0
    assert np.all(np.abs(np.asarray(result.parameters)) <= 1.0)


@pytest.mark.parametrize(
    "budget, expected_calls, successful", [(4, 3, False), (5, 5, True)]
)
def test_model_based_terminal_gradient_requires_complete_allowance(
    budget: int, expected_calls: int, successful: bool
) -> None:
    calls = 0

    def objective(parameters: object, args: object) -> float:
        nonlocal calls
        calls += 1
        return float(np.asarray(parameters)[0] ** 2)

    result = opt.minimize(
        opt.MinimizationProblem(objective, bounds=opt.Bounds(-1.0, 1.0)),
        jnp.zeros((1,)),
        method=opt.BOBYQA(),
        termination=opt.OptimizationTermination(
            absolute_optimality=1e-6,
            relative_optimality=0.0,
            maximum_evaluations=budget,
        ),
    )

    assert calls == expected_calls
    assert int(result.diagnostics.objective_evaluations) == calls
    assert bool(result.successful) is successful
    if successful:
        assert float(result.diagnostics.final_optimality_norm) <= 1e-6
    else:
        assert int(result.status) == int(
            opt.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
        )
        assert np.isnan(float(result.diagnostics.final_optimality_norm))


@pytest.mark.parametrize("enforce_equality", [False, True])
def test_cobyqa_exhausted_initial_allowance_retains_best_feasible_state(
    enforce_equality: bool,
) -> None:
    result = opt.minimize(
        opt.MinimizationProblem(
            lambda parameters, args: float((np.asarray(parameters)[0] - 0.25) ** 2),
            bounds=opt.Bounds(-1.0, 1.0),
            constraints=(
                (
                    opt.NonlinearConstraint(
                        lambda parameters, args: {"equality": np.asarray(parameters)},
                        lower=0.0,
                        upper=0.0,
                    ),
                )
                if enforce_equality
                else ()
            ),
        ),
        jnp.zeros((1,)),
        method=opt.COBYQA(),
        termination=opt.OptimizationTermination(
            absolute_optimality=1e-6, maximum_evaluations=2
        ),
    )

    assert int(result.status) == int(opt.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert np.array_equal(
        np.asarray(result.parameters), [0.0 if enforce_equality else 0.25]
    )
    assert float(result.objective) == (0.0625 if enforce_equality else 0.0)
    assert float(result.diagnostics.primal_feasibility) == 0.0
    assert int(result.diagnostics.objective_evaluations) == 2
    assert int(result.diagnostics.constraint_evaluations) == (
        2 if enforce_equality else 0
    )


def test_model_based_nonfinite_terminal_check_cannot_report_success() -> None:
    calls = 0

    def objective(parameters: object, args: object) -> float:
        nonlocal calls
        calls += 1
        return float("inf") if calls == 4 else float(np.asarray(parameters)[0] ** 2)

    result = opt.minimize(
        opt.MinimizationProblem(objective, bounds=opt.Bounds(-1.0, 1.0)),
        jnp.zeros((1,)),
        method=opt.BOBYQA(),
        termination=opt.OptimizationTermination(
            absolute_optimality=1e-6, maximum_evaluations=5
        ),
    )

    assert calls == int(result.diagnostics.objective_evaluations) == 4
    assert int(result.status) == int(opt.OptimizationStatus.NONFINITE_EVALUATION)
    assert np.array_equal(np.asarray(result.parameters), [0.0])
    assert float(result.objective) == 0.0
    assert np.isnan(float(result.diagnostics.final_optimality_norm))


@pytest.mark.parametrize("failure_call", [1, 2, 4, 5])
@pytest.mark.parametrize("nonfinite_constraint", [False, True])
def test_cobyqa_nonfinite_evaluation_refuses_and_retains_finite_feasible_state(
    failure_call: int, nonfinite_constraint: bool
) -> None:
    calls = {"objective": 0, "constraint": 0}

    def objective(parameters: object, args: object) -> float:
        calls["objective"] += 1
        if calls["objective"] == failure_call and not nonfinite_constraint:
            return float("inf")
        coordinate = float(np.asarray(parameters)[0])
        return 0.0 if coordinate == 0.0 else 1.0 + 0.1 * coordinate

    def constraint(parameters: object, args: object) -> NDArray[np.float64]:
        calls["constraint"] += 1
        return np.asarray(
            [
                np.nan
                if calls["constraint"] == failure_call and nonfinite_constraint
                else 0.0
            ],
            dtype=np.float64,
        )

    result = opt.minimize(
        opt.MinimizationProblem(
            objective,
            bounds=opt.Bounds(-1.0, 1.0),
            constraints=(opt.NonlinearConstraint(constraint, upper=0.0),),
        ),
        jnp.zeros((1,)),
        method=opt.COBYQA(),
        termination=opt.OptimizationTermination(
            absolute_optimality=1e-6, maximum_evaluations=20
        ),
    )

    assert int(result.status) == int(opt.OptimizationStatus.NONFINITE_EVALUATION)
    assert calls["objective"] == calls["constraint"] == failure_call
    assert int(result.diagnostics.objective_evaluations) == failure_call
    assert int(result.diagnostics.constraint_evaluations) == failure_call
    assert np.isnan(float(result.diagnostics.final_optimality_norm))
    assert not bool(result.successful)
    if failure_call > 1:
        assert np.isfinite(float(result.objective))
        assert float(result.diagnostics.primal_feasibility) == 0.0
        assert np.array_equal(np.asarray(result.parameters), [0.0])


def test_best_in_class_nonlinear_optimization_scenario_5() -> None:
    problem = opt.MinimizationProblem(lambda value, _: jnp.sum(value**2))
    with pytest.raises(ValueError, match="one local step"):
        opt.multistart_minimize(
            problem,
            jnp.asarray([1.0]),
            policy=opt.MultiStartPolicy(count=4, generator="normal"),
            termination=opt.OptimizationTermination(maximum_steps=3),
        )
    problem = opt.MinimizationProblem(
        lambda value, _: jnp.where(
            value[0] == 0.0,
            jnp.asarray(2.0e12),
            jnp.asarray(jnp.nan),
        ),
        bounds=opt.Bounds(-1.0, 1.0),
    )
    result = opt.multistart_minimize(
        problem,
        jnp.asarray([0.0]),
        policy=opt.MultiStartPolicy(
            local_method=opt.ProjectedGradient(),
            count=4,
            generator="normal",
            seed=5,
        ),
        termination=opt.OptimizationTermination(
            maximum_steps=4,
            maximum_evaluations=40,
        ),
    )

    assert bool(result.successful)
    assert int(result.best_index) == 0
    for method in (
        opt.DoglegLeastSquares(),
        opt.BoundedNewtonTrustRegion(),
    ):
        if isinstance(method, opt.DoglegLeastSquares):
            result = opt.least_squares(
                opt.NonlinearLeastSquaresProblem(lambda value, _: value - 3.0),
                jnp.asarray([0.0]),
                method=method,
                termination=opt.OptimizationTermination(
                    absolute_optimality=0.0,
                    relative_optimality=0.0,
                    maximum_steps=20,
                    maximum_evaluations=1,
                ),
            )
        else:
            result = opt.minimize(
                opt.MinimizationProblem(
                    lambda value, _: jnp.sum((value - 3.0) ** 2),
                    bounds=opt.Bounds(-1.0, 1.0),
                ),
                jnp.asarray([0.0]),
                method=method,
                termination=opt.OptimizationTermination(
                    absolute_optimality=0.0,
                    relative_optimality=0.0,
                    maximum_steps=20,
                    maximum_evaluations=1,
                ),
            )

        assert int(result.status) == int(
            opt.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
        )
    from phydrax.optim._external_backends import _certify_minimization

    problem = opt.MinimizationProblem(
        lambda value, _: value[0],
        constraints=(
            opt.NonlinearConstraint(
                lambda value, _: value,
                upper=0.0,
                constraint_id="upper-zero",
            ),
        ),
    )
    *_, certified = _certify_minimization(
        problem,
        jnp.asarray([0.0]),
        None,
        opt.OptimizationTermination(
            absolute_optimality=1e-8,
            relative_optimality=0.0,
        ),
        True,
    )

    assert not bool(certified)


def test_best_in_class_nonlinear_optimization_scenario_6() -> None:
    result = opt.minimize(
        _constrained_problem(),
        jnp.asarray([0.5, 0.5]),
        args=jnp.asarray([0.2, 0.8]),
        method=opt.PrimalDualInteriorPoint(
            mode="dense-filter",
            maximum_line_search_steps=4,
        ),
        termination=opt.OptimizationTermination(
            absolute_optimality=0.0,
            relative_optimality=0.0,
            maximum_steps=10,
            maximum_evaluations=1,
        ),
    )

    assert int(result.status) == int(opt.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert int(result.diagnostics.iterations) == 0
    assert int(result.diagnostics.objective_evaluations) == 1
    assert int(result.diagnostics.globalization_evaluations) == 0
    result = opt.minimize(
        _constrained_problem(),
        jnp.asarray([0.5, 0.5]),
        args=jnp.asarray([0.2, 0.8]),
        method=opt.PrimalDualInteriorPoint(
            mode="dense-filter",
            maximum_line_search_steps=4,
        ),
        termination=opt.OptimizationTermination(
            absolute_optimality=0.0,
            relative_optimality=0.0,
            maximum_steps=10,
            maximum_evaluations=4,
        ),
    )

    assert int(result.status) == int(opt.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert int(result.diagnostics.objective_evaluations) == 1
    assert int(result.diagnostics.globalization_evaluations) == 0
    result = opt.minimize(
        _constrained_problem(),
        jnp.asarray([0.5, 0.5]),
        args=jnp.asarray([0.1, 0.9]),
        method=opt.PrimalDualInteriorPoint(
            mode="dense-filter",
            maximum_line_search_steps=4,
        ),
        termination=opt.OptimizationTermination(
            absolute_optimality=0.0,
            relative_optimality=0.0,
            maximum_steps=1,
            maximum_evaluations=100,
        ),
    )

    assert int(result.diagnostics.globalization_evaluations) == 4
    method = opt.BoundedLevenbergMarquardt()
    residual = opt.BoundedResidualFunction(
        lambda value: value - 2.0,
        opt.Bounds(-5.0, 5.0),
    )
    parameters = jnp.asarray([0.0])
    base = method.prepare_state(residual, parameters)
    small = eqx.tree_at(
        lambda state: (state.damping, state.metrics.damping),
        base,
        (jnp.asarray(1e-6), jnp.asarray(1e-3)),
    )
    large = eqx.tree_at(
        lambda state: (state.damping, state.metrics.damping),
        base,
        (jnp.asarray(1e3), jnp.asarray(1.0)),
    )
    termination = opt.OptimizationTermination(
        absolute_optimality=0.0,
        relative_optimality=0.0,
        maximum_steps=10,
        maximum_evaluations=100,
    )

    small_parameters, small_state, _ = method.step(
        residual,
        parameters,
        small,
        termination=termination,
    )
    large_parameters, large_state, _ = method.step(
        residual,
        parameters,
        large,
        termination=termination,
    )

    assert int(small_state.iteration) == 1
    assert int(large_state.iteration) == 1
    assert not jnp.allclose(small_parameters, large_parameters)
    from phydrax._nonlinear_precision import NonlinearPrecisionPolicy
    from phydrax.optim._constrained_sensitivity import _sensitivity_system

    parameters = jnp.asarray([0.25, 0.75])
    barrier = 1e-4
    target = parameters - 0.5 * barrier / parameters
    _, initial, residual, *_ = _sensitivity_system(
        _constrained_problem(),
        parameters,
        target,
        "barrier",
        1e-7,
        barrier,
        None,
        NonlinearPrecisionPolicy(),
    )

    assert jnp.linalg.norm(residual(initial, target), ord=jnp.inf) < 1e-6


def test_constrained_sensitivity_rejects_nonconverged_linear_result(
    monkeypatch: Any,
) -> None:
    import phydrax.optim._constrained_sensitivity as sensitivity_module

    original = sensitivity_module.solve_linear

    def failed_solve(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        return eqx.tree_at(
            lambda value: (value.status, value.diagnostics.converged),
            result,
            (
                jnp.asarray(
                    int(phx.linalg.LinearSolveStatus.MAXIMUM_STEPS_REACHED),
                    dtype=jnp.int32,
                ),
                jnp.asarray(False),
            ),
        )

    monkeypatch.setattr(sensitivity_module, "solve_linear", failed_solve)
    result = opt.constrained_solution_jvp(
        _constrained_problem(),
        jnp.asarray([0.2, 0.8]),
        jnp.asarray([0.2, 0.8]),
        jnp.asarray([1.0, 0.0]),
    )

    assert not bool(result.regular)
    assert jnp.all(jnp.isnan(result.value))
