#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import Boundary, DomainFunction, Interval1d, Square, TimeInterval
from phydrax.operators.integral import (
    integral,
    nonlocal_integral,
    spatial_integral,
    time_convolution,
)
from phydrax.operators.integral._quadrature import build_quadrature as build_mc_quadrature


def test_ad_integral_interior_grad_finite_shape(sample_batch):
    geom = Square(center=(0.0, 0.0), side=1.0)
    component = geom.component()
    batch = sample_batch(component, blocks=(("x",),), num_points=2048, key=0)

    def loss(theta):
        @geom.Function("x")
        def f(x):
            return jnp.dot(theta, x) ** 2

        return integral(f, batch, component=component, over="x").data

    theta0 = jnp.array([0.3, -0.2])
    g = jax.grad(loss)(theta0)
    assert g.shape == theta0.shape
    assert jnp.all(jnp.isfinite(g))


def test_ad_integral_boundary_grad_finite_shape(sample_batch):
    geom = Square(center=(0.0, 0.0), side=1.0)
    component = geom.component({"x": Boundary()})
    batch = sample_batch(component, blocks=(("x",),), num_points=2048, key=1)

    def loss(theta):
        @geom.Function("x")
        def f(x):
            return jnp.dot(theta, x) ** 2

        return integral(f, batch, component=component, over="x").data

    theta0 = jnp.array([0.1, 0.4])
    g = jax.grad(loss)(theta0)
    assert g.shape == theta0.shape
    assert jnp.all(jnp.isfinite(g))


def test_ad_spatial_integral_grad_finite_shape():
    geom = Square(center=(0.0, 0.0), side=1.0)
    quad = build_mc_quadrature(geom, num_points=4096, seed=0)

    def loss(theta):
        @geom.Function("x")
        def u(x):
            return jnp.dot(theta, x)

        S = spatial_integral(u, quad=quad)
        x0 = frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})
        return jnp.asarray(S(x0).data)

    theta0 = jnp.array([0.2, -0.5])
    g = jax.grad(loss)(theta0)
    assert g.shape == theta0.shape
    assert jnp.all(jnp.isfinite(g))


def test_ad_nonlocal_integral_grad_matches_analytic():
    # Domain [0,1], x=0.5, u(y)=θ*y, integrand=du^2 with du=θ(y-0.5)
    # ∫_0^1 du^2 dy = θ^2/12, so d/dθ = θ/6.
    geom = Interval1d(0.0, 1.0)
    quad = build_mc_quadrature(geom, num_points=4096, seed=0)

    def loss(theta):
        u = DomainFunction(domain=geom, deps=("x",), func=lambda x: theta * x[0])

        def integrand(du, xi):
            return du * du

        NLI = nonlocal_integral(u, integrand=integrand, quad=quad)
        x0 = frozendict({"x": cx.Field(jnp.array([0.5]), dims=(None,))})
        return jnp.asarray(NLI(x0).data)

    theta0 = jnp.array(1.2)
    g = jax.grad(loss)(theta0)
    expected = theta0 / 6.0
    assert jnp.isfinite(g)
    assert jnp.allclose(g, expected, rtol=0.05, atol=2e-2)


def test_ad_time_convolution_grad_matches_closed_form():
    # k(τ)=e^{-τ}, u(t)=θ sin t ⇒ d/dθ (k⋆u)(t) = 0.5*(sin t - cos t + e^{-t})
    dom = TimeInterval(0.0, 2.0)
    k = lambda tau: jnp.exp(-tau)

    def loss(theta):
        @dom.Function("t")
        def u(t):
            return theta * jnp.sin(t)

        conv = time_convolution(k, u, order=64)
        t0 = 1.234
        return jnp.asarray(conv(frozendict({"t": cx.Field(jnp.array(t0), dims=())})).data)

    theta0 = jnp.array(0.9)
    g = jax.grad(loss)(theta0)
    t = 1.234
    exact = 0.5 * (jnp.sin(t) - jnp.cos(t) + jnp.exp(-t))
    assert jnp.isfinite(g)
    assert jnp.allclose(g, exact, atol=3e-3, rtol=0.0)
