import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict


def _polar_problem():
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(2.0, 0.0), side=2.0).compile()
    )
    chart = phx.metrix.CoordinateChart("polar", ("r", "theta"))
    metric = phx.metrix.diagonal_metric(
        lambda q: jnp.array([1.0, q[0] ** 2]),
        chart=chart,
    )
    return domain, metric


def _points(values):
    return frozendict(
        {
            "x": cx.Field(
                jnp.asarray(values, dtype=float),
                dims=("sample", None),
            )
        }
    )


def test_domain_function_riemannian_operators_match_polar_identities():
    domain, metric = _polar_problem()
    scalar = domain.Function("x")(lambda x: x[0] ** 2)
    vector = domain.Function("x")(lambda x: jnp.array([x[0], 0.0]))
    metric_field = domain.Function("x")(lambda x: metric(x))
    inverse_metric = domain.Function("x")(lambda x: metric.inverse(x))
    points = _points([[1.3, 0.2], [2.0, -0.4], [2.7, 0.7]])

    gradient = phx.operators.riemannian_grad(scalar, metric, var="x")
    divergence = phx.operators.riemannian_div(vector, metric, var="x")
    hessian = phx.operators.covariant_hessian(scalar, metric, var="x")
    laplacian = phx.operators.laplace_beltrami(scalar, metric, var="x")
    metric_derivative = phx.operators.covariant_derivative(
        metric_field,
        metric,
        phx.metrix.TensorType(("covariant", "covariant")),
        var="x",
    )
    inverse_divergence = phx.operators.riemannian_div_tensor(
        inverse_metric,
        metric,
        var="x",
    )

    radii = jnp.asarray(points["x"].data)[..., 0]
    assert jnp.allclose(
        jnp.asarray(gradient(points).data),
        jnp.stack((2.0 * radii, jnp.zeros_like(radii)), axis=-1),
    )
    assert jnp.allclose(jnp.asarray(divergence(points).data), 2.0)
    assert jnp.allclose(jnp.asarray(laplacian(points).data), 4.0)
    assert jnp.allclose(jnp.asarray(hessian(points).data)[..., 0, 0], 2.0)
    assert jnp.allclose(jnp.asarray(hessian(points).data)[..., 1, 1], 2.0 * radii**2)
    assert jnp.allclose(jnp.asarray(metric_derivative(points).data), 0.0, atol=1e-9)
    assert jnp.allclose(jnp.asarray(inverse_divergence(points).data), 0.0, atol=1e-9)


def test_laplace_beltrami_matches_unit_sphere_eigenfunction():
    chart = phx.metrix.CoordinateChart("sphere", ("theta", "phi"))
    embedded = phx.metrix.EmbeddedChart(
        chart,
        lambda q: jnp.array(
            [
                jnp.sin(q[0]) * jnp.cos(q[1]),
                jnp.sin(q[0]) * jnp.sin(q[1]),
                jnp.cos(q[0]),
            ]
        ),
        3,
    )
    metric = embedded.induced_metric()
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(1.5, 0.0), side=2.0).compile()
    )
    scalar = domain.Function("x")(lambda q: jnp.sin(q[0]) * jnp.cos(q[1]))
    points = _points([[0.6, -0.7], [1.1, 0.4], [2.2, 0.8]])

    laplacian = phx.operators.laplace_beltrami(scalar, metric, var="x")
    values = jnp.asarray(laplacian(points).data)
    expected = (
        -2.0
        * jnp.sin(jnp.asarray(points["x"].data)[:, 0])
        * jnp.cos(jnp.asarray(points["x"].data)[:, 1])
    )

    assert jnp.allclose(values, expected, atol=1e-9)
    with pytest.raises(TypeError, match="RiemannianMetric"):
        phx.operators.laplace_beltrami(scalar, domain.component(), var="x")


def test_riemannian_measure_multiplies_existing_component_weights():
    domain, metric = _polar_problem()
    component = phx.domain.with_riemannian_measure(
        domain.component(),
        metric,
        var="x",
    )
    points = _points([[1.5, 0.1], [2.0, -0.2], [2.5, 0.4]])

    weight = component.weight_all
    assert weight is not None
    weights = jnp.asarray(weight(points).data)
    assert jnp.allclose(weights, jnp.array([1.5, 2.0, 2.5]))


def test_metric_aware_domain_stochastic_operators_use_riemannian_volume():
    domain, metric = _polar_problem()
    density = domain.Function("x")(lambda x: x[0] ** 2)
    coordinate_drift = domain.Function("x")(lambda x: jnp.array([0.5 / x[0], 0.0]))
    covariance = domain.Function("x")(lambda x: metric.inverse(x))
    points = _points([[1.4, 0.2], [2.2, -0.3]])

    backward = phx.operators.kolmogorov_generator(
        density,
        coordinate_drift,
        covariance=covariance,
        metric=metric,
        var="x",
    )
    forward = phx.operators.fokker_planck_operator(
        density,
        coordinate_drift,
        covariance=covariance,
        metric=metric,
        var="x",
    )

    assert jnp.allclose(jnp.asarray(backward(points).data), 2.0, atol=1e-9)
    assert jnp.allclose(jnp.asarray(forward(points).data), 2.0, atol=1e-9)


def test_metric_aware_fokker_planck_constraint_threads_metric_to_residual():
    domain, metric = _polar_problem()
    density = domain.Function("x")(1.0)
    coordinate_drift = domain.Function("x")(lambda x: jnp.array([0.5 / x[0], 0.0]))
    covariance = domain.Function("x")(lambda x: metric.inverse(x))
    condition = phx.conditions.stochastic.FokkerPlanck(
        "p",
        domain.component(),
        drift=coordinate_drift,
        evolution_var=None,
        covariance=covariance,
        metric=metric,
    )
    target = phx.integration.mean_over(condition.on)
    realization = phx.integration.materialize(
        target,
        phx.integration.MonteCarloPlan(20),
        key=jr.key(5),
    )
    constraint = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(realization),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"p": density},
        terms=(constraint,),
    )

    assert solver.loss(key=jr.key(6)) < 1e-20


def test_riemannian_residual_runs_through_functional_solver():
    domain, metric = _polar_problem()
    field = domain.Function("x")(lambda x: x[0] ** 2)
    condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda u: phx.operators.laplace_beltrami(u, metric, var="x") - 4.0,
    )
    target = phx.integration.mean_over(condition.on)
    realization = phx.integration.materialize(
        target,
        phx.integration.MonteCarloPlan(24),
        key=jr.key(3),
    )
    constraint = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(realization),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(constraint,),
    )

    assert solver.loss(key=jr.key(4)) < 1e-20
