#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


opt = phx.optim


def _termination(maximum_steps=100):
    return opt.OptimizationTermination(
        absolute_optimality=1e-6,
        relative_optimality=0.0,
        maximum_steps=maximum_steps,
        maximum_evaluations=5000,
    )


def test_residual_graph_robust_losses_and_certificate():
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


def test_route_and_schur_planning_solve_declared_partition():
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


def test_graph_route_executes_schur_and_clips_nonconvex_robust_curvature():
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


@pytest.mark.parametrize(
    "method",
    [
        opt.DoglegLeastSquares(),
        opt.DoglegLeastSquares("subspace"),
        opt.DoglegLeastSquares("dogbox"),
    ],
)
def test_least_squares_trust_family(method):
    problem = opt.NonlinearLeastSquaresProblem(
        lambda parameters, target: parameters - target,
        bounds=(opt.Bounds(0.0, 1.0) if method.mode == "dogbox" else None),
    )
    target = (
        jnp.asarray([2.0, -0.5]) if method.mode == "dogbox" else jnp.asarray([1.0, 0.5])
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


def test_variable_projection_recovers_linear_and_nonlinear_blocks():
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


def test_pounders_solves_bound_constrained_black_box_residual():
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


def test_pounders_demotes_model_success_when_physical_stationarity_fails():
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
    assert result.status_evidence.internal_status == int(opt.OptimizationStatus.SUCCESS)
    assert bool(result.status_evidence.demoted)
    assert not bool(result.optimality_certificate.certified)
    assert float(result.optimality_certificate.optimality_norm) > 1e-6
    assert result.optimality_certificate.kind == "derivative-free-stationarity"
    assert int(result.method_evidence.interpolation_rank) < 6


def test_physical_stationarity_certificate_never_steps_outside_narrow_bounds():
    evaluated = []

    def residual(parameters, args):
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


def test_manifold_and_incremental_factor_graph_lifecycle():
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


def _constrained_problem():
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


def test_scaled_constrained_model_sqp_ipm_and_kkt_inertia():
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
        method=opt.FilterInteriorPoint(),
        termination=_termination(),
    )
    assert interior_point.status == int(opt.OptimizationStatus.CERTIFICATION_FAILED)
    assert interior_point.status_evidence.internal_status == int(
        opt.OptimizationStatus.SUCCESS
    )
    assert bool(interior_point.status_evidence.demoted)
    assert interior_point.optimality_certificate.kind == "active-kkt"
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


def test_constrained_forward_reverse_solution_maps():
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


def test_model_based_multistart_and_external_recertification():
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
