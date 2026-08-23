#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Interval1d, TimeInterval
from phydrax.operators.differential import fractional_laplacian
from phydrax.operators.integral import (
    integral,
    local_integral,
    local_integral_ball,
    mean,
    nonlocal_integral,
    spatial_integral,
    time_convolution,
)


def _interval_rule(count, lower=0.0, upper=1.0):
    step = (upper - lower) / count
    points = lower + step * (jnp.arange(count, dtype=float) + 0.5)
    return {
        "points": points[:, None],
        "weights": jnp.full((count,), step),
    }


def _ball_rule(radius, dimension, count):
    if dimension == 1:
        rule = _interval_rule(count, -radius, radius)
        return {"offsets": rule["points"], "weights": rule["weights"]}
    if dimension != 2:
        raise ValueError("Test helper supports one- and two-dimensional balls.")
    index = jnp.arange(count, dtype=float)
    radial = radius * jnp.sqrt((index + 0.5) / count)
    angle = index * jnp.pi * (3.0 - jnp.sqrt(5.0))
    offsets = jnp.stack((radial * jnp.cos(angle), radial * jnp.sin(angle)), axis=1)
    weights = jnp.full((count,), jnp.pi * radius**2 / count)
    return {"offsets": offsets, "weights": weights}


def test_integral_and_mean_delegate_to_typed_integration_api():
    domain = Interval1d(0.0, 1.0)
    target = phx.integration.over(domain.component())
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8))
    function = DomainFunction(domain=domain, deps=(), func=2.0)

    integrated = integral(function, target, plan)
    averaged = mean(function, target, plan)

    assert jnp.allclose(jnp.asarray(integrated.data), 2.0, atol=1e-12)
    assert jnp.allclose(jnp.asarray(averaged.data), 2.0, atol=1e-12)


def test_spatial_integral_nonlocal_kernel_converges_under_rule_refinement():
    domain = Interval1d(0.0, 1.0)

    @domain.Function("x")
    def function(x):
        return jnp.sin(jnp.pi * x[0])

    def kernel(pair):
        x, y = pair[0], pair[1]
        return jnp.exp(-((x - y) ** 2))

    coarse = spatial_integral(function, quad=_interval_rule(512), kernel=kernel)
    fine = spatial_integral(function, quad=_interval_rule(4096), kernel=kernel)
    points = frozendict(
        {"x": cx.Field(jnp.linspace(0.0, 1.0, 5)[:, None], dims=("n", None))}
    )
    coarse_values = jnp.asarray(coarse(points).data)
    fine_values = jnp.asarray(fine(points).data)

    relative_error = jnp.linalg.norm(coarse_values - fine_values) / jnp.linalg.norm(
        fine_values
    )
    assert relative_error < 1e-4


def test_time_convolution_exp_sin_closed_form():
    domain = TimeInterval(0.0, 2.0)

    @domain.Function("t")
    def function(t):
        return jnp.sin(t)

    convolution = time_convolution(
        lambda lag: jnp.exp(-lag),
        function,
        rule=phx.integration.GaussLegendreRule(64),
    )
    times = jnp.linspace(0.0, 2.0, 25)
    values = jnp.asarray(
        convolution(frozendict({"t": cx.Field(times, dims=("t",))})).data
    )
    exact = 0.5 * (jnp.sin(times) - jnp.cos(times) + jnp.exp(-times))
    assert jnp.max(jnp.abs(values - exact)) < 2e-3


def test_time_convolution_is_exact_zero_at_nonzero_domain_start():
    domain = TimeInterval(2.0, 3.0)
    function = domain.Function("t")(lambda time: jnp.stack((time, time**2)))
    convolution = time_convolution(lambda lag: jnp.exp(-lag), function)
    start = frozendict({"t": cx.Field(jnp.array(2.0), dims=())})

    value = jnp.asarray(convolution(start).data)

    assert jnp.array_equal(value, jnp.zeros((2,)))
    assert convolution.metadata["integral_randomized"] is False
    assert convolution.metadata["integral_rule"] == "GaussLegendreRule"


def test_time_convolution_nonzero_start_and_clustered_rule():
    domain = TimeInterval(2.0, 3.0)
    function = domain.Function("t")(lambda time: (time - 2.0) ** 2)
    convolution = time_convolution(
        lambda lag: jnp.ones_like(lag),
        function,
        rule=phx.integration.GaussLegendreRule(32),
        cluster_exponent=2.0,
    )
    endpoint = frozendict({"t": cx.Field(jnp.array(3.0), dims=())})

    assert jnp.allclose(convolution(endpoint).data, 1.0 / 3.0, atol=1e-12)


def test_fractional_laplacian_constant_zero():
    domain = Interval1d(-1.0, 1.0)
    function = DomainFunction(domain=domain, deps=(), func=jnp.array(3.14))
    operator = fractional_laplacian(function, alpha=1.2)
    points = jnp.linspace(-0.8, 0.8, 7)[:, None]
    values = jnp.asarray(
        operator(frozendict({"x": cx.Field(points, dims=("n", None))})).data
    )
    assert jnp.max(jnp.abs(values)) < 1e-12


def test_nonlocal_integral_zero_field_zero_result():
    domain = Interval1d(0.0, 1.0)
    function = DomainFunction(domain=domain, deps=(), func=jnp.array(0.0))

    def integrand(delta_value, displacement):
        return (jnp.abs(displacement[0]) < 0.25).astype(float) * delta_value

    operator = nonlocal_integral(function, integrand=integrand, quad=_interval_rule(512))
    points = jnp.linspace(0.1, 0.9, 5)[:, None]
    values = jnp.asarray(
        operator(frozendict({"x": cx.Field(points, dims=("n", None))})).data
    )
    assert jnp.max(jnp.abs(values)) < 1e-12


def test_nonlocal_integral_time_dependent_field_integrates_to_time(sample_batch):
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)

    @domain.Function("t")
    def function(time):
        return time

    operator = nonlocal_integral(
        function,
        integrand=lambda value: value,
        quad=_interval_rule(1024),
        time_var="t",
    )
    component = domain.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=4)
    output = jnp.asarray(operator(batch).data)

    assert output.shape == (3, 4)
    assert jnp.allclose(output, batch.points["t"].data[None, :], atol=1e-12)


def test_nonlocal_integral_context_parameter_receives_full_context():
    domain = Interval1d(0.0, 1.0)
    function = domain.Function("x")(lambda x: x[0])
    operator = nonlocal_integral(
        function,
        integrand=lambda context: context["uy"] + 0.0 * context["xi"][0],
        quad=_interval_rule(128),
    )
    points = frozendict({"x": cx.Field(jnp.array([[0.25], [0.75]]), dims=("n", None))})

    assert jnp.allclose(jnp.asarray(operator(points).data), 0.5, atol=1e-12)


def test_local_integral_constant_field_equals_ball_volume():
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    function = DomainFunction(domain=domain, deps=(), func=jnp.array(2.5))
    radius = 0.4
    operator = local_integral(
        function,
        integrand=lambda value: value,
        ball_quad=_ball_rule(radius, 2, 2048),
    )

    value = jnp.asarray(
        operator(frozendict({"x": cx.Field(jnp.array([0.1, -0.2]), dims=(None,))})).data
    )

    assert jnp.allclose(value, jnp.pi * radius**2 * 2.5, atol=1e-12)


def test_local_integral_zero_and_linear_symmetry():
    interval = Interval1d(-1.0, 1.0)
    zero = DomainFunction(domain=interval, deps=(), func=jnp.array(0.0))
    rule_1d = _ball_rule(0.2, 1, 1024)
    zero_operator = local_integral(
        zero,
        integrand=lambda delta, displacement: delta * displacement[0],
        ball_quad=rule_1d,
    )
    points = jnp.linspace(-0.3, 0.3, 7)[:, None]
    values = jnp.asarray(
        zero_operator(frozendict({"x": cx.Field(points, dims=("n", None))})).data
    )
    assert jnp.max(jnp.abs(values)) < 1e-12

    square = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @square.Function("x")
    def linear(x):
        return x[0]

    ball_operator = local_integral_ball(
        linear,
        f_bond=lambda delta, displacement: delta,
        ball_quad=_ball_rule(0.25, 2, 4096),
    )
    point = frozendict({"x": cx.Field(jnp.array([0.1, -0.2]), dims=(None,))})
    assert jnp.abs(jnp.asarray(ball_operator(point).data)) < 5e-4
