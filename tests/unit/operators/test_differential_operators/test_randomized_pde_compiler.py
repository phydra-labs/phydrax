from typing import Any

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


def _problem(dimension: Any, expression: Any, *, rhs: Any = 0.0) -> Any:
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


def _domain(dimension: Any) -> Any:
    return phx.domain.HyperRectangle(
        jnp.full((dimension,), -1.0),
        jnp.full((dimension,), 1.0),
        label="x",
    )


def _compile(problem: Any, domain: Any, plan: Any, *, num_points: Any = 32) -> Any:
    return compile_pde_randomized_term(
        problem,
        "governing",
        plan,
        component=domain.component(),
        sampling=phx.domain.PointSampling(
            num_points, layout=phx.domain.SampleLayout((("x",),))
        ),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(19),
    )


def test_randomized_pde_compiler_scenario_1() -> None:
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
    with pytest.raises(ValueError):
        RandomizedDifferentialPlan(
            "dimension",
            dimension_policy=DimensionSamplingPolicy(10, 4),
            loss_mode="u_statistic",
        )


def test_randomized_compiler_preserves_parameter_gradients() -> None:
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

    def loss(coefficient: Any) -> Any:
        function = domain.Function("x")(lambda x: coefficient * jnp.dot(x, x))
        return compiled.term.loss({"u": function}, batch=batch)

    coefficient = jnp.asarray(0.2)
    value, gradient = jax.value_and_grad(loss)(coefficient)

    assert jnp.allclose(value, (2.0 * dimension * coefficient - dimension) ** 2)
    assert jnp.allclose(
        gradient,
        4.0 * dimension * (2.0 * dimension * coefficient - dimension),
    )


def test_randomized_pde_compiler_scenario_2() -> None:
    coordinate = phx.equations.PDEExpression.coordinate_value("x")
    expression = coordinate.dot(coordinate).laplacian("x")
    problem = _problem(3, expression, rhs=6.0)
    report = analyze_randomized_compilation(
        problem,
        "governing",
        RandomizedDifferentialPlan(),
    )

    assert not report.supported
    dimension = 3
    field = phx.equations.PDEExpression.field("u")
    coordinate = phx.equations.PDEExpression.coordinate_value("x")
    expression = (field.laplacian("x") * coordinate).component(0)
    domain = _domain(dimension)
    compiled = _compile(
        _problem(dimension, expression),
        domain,
        RandomizedDifferentialPlan(
            trace_policy=phx.operators.StochasticTracePolicy(5),
        ),
        num_points=4,
    )
    function = domain.Function("x")(lambda x: jnp.dot(x, x))

    diagnostics = compiled.term.diagnostics({"u": function}, key=jr.key(29))

    assert diagnostics.num_realizations == 5
    assert bool(diagnostics.finite)
