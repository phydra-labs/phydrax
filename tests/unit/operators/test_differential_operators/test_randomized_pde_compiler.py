import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.equations._randomized_compile import (
    analyze_randomized_compilation,
    compile_pde_randomized_term,
    RandomizedDifferentialPlan,
)
from phydrax.operators.differential._dimension_estimators import (
    DimensionSamplingPolicy,
)


def _problem(dimension, expression, *, rhs=0.0):
    return phx.equations.PDEProblemIR(
        coordinates=(
            phx.equations.PDECoordinate(
                "x",
                "space",
                size=dimension,
                bounds=(-1.0, 1.0),
            ),
        ),
        fields=(phx.equations.PDEField("u", coordinates=("x",)),),
        equations=(
            phx.equations.PDEEquation(
                "governing",
                expression,
                phx.equations.PDEExpression.constant(rhs),
            ),
        ),
    )


def _domain(dimension):
    return phx.domain.HyperRectangle(
        jnp.full((dimension,), -1.0),
        jnp.full((dimension,), 1.0),
        label="x",
    )


def _compile(problem, domain, plan, *, num_points=32):
    return compile_pde_randomized_term(
        problem,
        "governing",
        plan,
        component=domain.component(),
        sampling=phx.domain.PointSampling(num_points, layout=phx.domain.SampleLayout((("x",),))),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(19),
    )


def test_analysis_reports_stable_randomized_paths_and_methods():
    field = phx.equations.PDEExpression.field("u")
    problem = _problem(7, field.laplacian("x") + field.derivative("x", axis=0))
    plan = RandomizedDifferentialPlan(
        "hutchinson",
        trace_policy=phx.operators.StochasticTracePolicy(8),
    )

    first = analyze_randomized_compilation(problem, "governing", plan)
    replay = analyze_randomized_compilation(problem, "governing", plan)

    assert first.supported
    assert first == replay
    assert first.randomized_node_paths == ("root.args[0].args[0]",)
    assert first.node_methods == (("root.args[0].args[0]", "hutchinson"),)
    assert first.plan_id == plan.plan_id


def test_analysis_rejects_biased_nonlinear_and_product_lowerings():
    field = phx.equations.PDEExpression.field("u")
    expressions = (
        field.laplacian("x").exp(),
        field.laplacian("x") * field.laplacian("x"),
    )
    plan = RandomizedDifferentialPlan()

    for expression in expressions:
        report = analyze_randomized_compilation(
            _problem(4, expression),
            "governing",
            plan,
        )
        assert not report.supported
        assert report.rejection_reasons


def test_hutchinson_compilation_evaluates_laplacian_without_dense_hessian():
    dimension = 20
    field = phx.equations.PDEExpression.field("u")
    problem = _problem(dimension, field.laplacian("x"), rhs=2.0 * dimension)
    domain = _domain(dimension)
    plan = RandomizedDifferentialPlan(
        trace_policy=phx.operators.StochasticTracePolicy(16),
    )
    compiled = _compile(problem, domain, plan, num_points=12)
    function = domain.Function("x")(lambda x: jnp.dot(x, x))
    batch = compiled.term.sample(key=jr.key(3))

    loss = compiled.term.loss({"u": function}, batch=batch)
    jitted = eqx.filter_jit(
        lambda current: compiled.term.loss({"u": current}, batch=batch)
    )(function)

    assert compiled.report.supported
    assert jnp.allclose(loss, 0.0)
    assert jnp.allclose(jitted, 0.0)


def test_randomized_compiler_preserves_parameter_gradients():
    dimension = 8
    field = phx.equations.PDEExpression.field("u")
    problem = _problem(dimension, field.laplacian("x"), rhs=dimension)
    domain = _domain(dimension)
    compiled = _compile(
        problem,
        domain,
        RandomizedDifferentialPlan(
            trace_policy=phx.operators.StochasticTracePolicy(8),
        ),
    )
    batch = compiled.term.sample(key=jr.key(5))

    def loss(coefficient):
        function = domain.Function("x")(
            lambda x: coefficient * jnp.dot(x, x)
        )
        return compiled.term.loss({"u": function}, batch=batch)

    coefficient = jnp.asarray(0.2)
    value, gradient = jax.value_and_grad(loss)(coefficient)

    assert jnp.allclose(value, (2.0 * dimension * coefficient - dimension) ** 2)
    assert jnp.allclose(
        gradient,
        4.0 * dimension * (2.0 * dimension * coefficient - dimension),
    )


def test_dimension_compilation_runs_in_dimension_1000_with_independent_products():
    dimension = 1000
    field = phx.equations.PDEExpression.field("u")
    problem = _problem(dimension, field.laplacian("x"), rhs=2.0 * dimension)
    domain = _domain(dimension)
    plan = RandomizedDifferentialPlan(
        "dimension",
        dimension_policy=DimensionSamplingPolicy(dimension, 8),
        loss_mode="independent_product",
    )
    compiled = _compile(problem, domain, plan, num_points=2)
    function = domain.Function("x")(lambda x: jnp.dot(x, x))

    diagnostics = compiled.term.diagnostics({"u": function}, key=jr.key(7))

    assert diagnostics.num_realizations == 8
    assert diagnostics.finite
    assert jnp.allclose(diagnostics.objective, 0.0)


def test_exact_first_report_rejects_objectives_with_no_randomized_node():
    coordinate = phx.equations.PDEExpression.coordinate_value("x")
    expression = coordinate.dot(coordinate).laplacian("x")
    problem = _problem(3, expression, rhs=6.0)
    report = analyze_randomized_compilation(
        problem,
        "governing",
        RandomizedDifferentialPlan(),
    )

    assert not report.supported
    assert report.exact_node_paths
    assert "deterministic PDE compiler" in report.rejection_reasons[-1]


def test_dimension_u_statistic_rejects_dependent_without_replacement_draws():
    with pytest.raises(ValueError, match="independent coordinate draws"):
        RandomizedDifferentialPlan(
            "dimension",
            dimension_policy=DimensionSamplingPolicy(10, 4),
            loss_mode="u_statistic",
        )
