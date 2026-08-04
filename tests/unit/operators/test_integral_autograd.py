#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import Boundary, DomainFunction, Interval1d, Square, TimeInterval
from phydrax.operators.integral import (
    integral,
    nonlocal_integral,
    spatial_integral,
    time_convolution,
)


def _interval_rule(count):
    points = (jnp.arange(count, dtype=float) + 0.5) / count
    return {
        "points": points[:, None],
        "weights": jnp.full((count,), 1.0 / count),
    }


def _square_rule(order):
    axis = -0.5 + (jnp.arange(order, dtype=float) + 0.5) / order
    first, second = jnp.meshgrid(axis, axis, indexing="ij")
    points = jnp.stack((first, second), axis=-1).reshape((-1, 2))
    return {
        "points": points,
        "weights": jnp.full((order**2,), 1.0 / order**2),
    }


def test_fixed_integral_grad_has_finite_parameter_shape():
    geometry = Square(center=(0.0, 0.0), side=1.0)
    target = phx.integration.over(geometry.component())
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(16))

    def loss(parameter):
        function = geometry.Function("x")(lambda x: jnp.dot(parameter, x) ** 2)
        return integral(function, target, plan).data

    parameter = jnp.array([0.3, -0.2])
    gradient = jax.grad(loss)(parameter)

    assert gradient.shape == parameter.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_sampled_boundary_integral_grad_has_finite_parameter_shape(sample_batch):
    geometry = Square(center=(0.0, 0.0), side=1.0)
    component = geometry.component({"x": Boundary()})
    points = sample_batch(component, blocks=(("x",),), num_points=2048, key=1)
    realization = phx.integration.from_samples(phx.integration.over(component), points)

    def loss(parameter):
        function = geometry.Function("x")(lambda x: jnp.dot(parameter, x) ** 2)
        return integral(function, realization).data

    parameter = jnp.array([0.1, 0.4])
    gradient = jax.grad(loss)(parameter)

    assert gradient.shape == parameter.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_spatial_integral_grad_has_finite_parameter_shape():
    geometry = Square(center=(0.0, 0.0), side=1.0)
    quadrature = _square_rule(48)

    def loss(parameter):
        function = geometry.Function("x")(lambda x: jnp.dot(parameter, x))
        operator = spatial_integral(function, quad=quadrature)
        point = frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})
        return jnp.asarray(operator(point).data)

    parameter = jnp.array([0.2, -0.5])
    gradient = jax.grad(loss)(parameter)

    assert gradient.shape == parameter.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_nonlocal_integral_grad_matches_analytic():
    geometry = Interval1d(0.0, 1.0)
    quadrature = _interval_rule(4096)

    def loss(parameter):
        function = DomainFunction(
            domain=geometry,
            deps=("x",),
            func=lambda x: parameter * x[0],
        )
        operator = nonlocal_integral(
            function,
            integrand=lambda delta, displacement: delta * delta,
            quad=quadrature,
        )
        point = frozendict({"x": cx.Field(jnp.array([0.5]), dims=(None,))})
        return jnp.asarray(operator(point).data)

    parameter = jnp.array(1.2)
    gradient = jax.grad(loss)(parameter)

    assert jnp.allclose(gradient, parameter / 6.0, rtol=1e-3, atol=1e-5)


def test_time_convolution_grad_matches_closed_form():
    domain = TimeInterval(0.0, 2.0)

    def loss(parameter):
        function = domain.Function("t")(lambda time: parameter * jnp.sin(time))
        convolution = time_convolution(lambda lag: jnp.exp(-lag), function, order=64)
        return jnp.asarray(
            convolution(frozendict({"t": cx.Field(jnp.array(1.234), dims=())})).data
        )

    gradient = jax.grad(loss)(jnp.array(0.9))
    time = 1.234
    exact = 0.5 * (jnp.sin(time) - jnp.cos(time) + jnp.exp(-time))

    assert jnp.allclose(gradient, exact, atol=3e-3, rtol=0.0)
