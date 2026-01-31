#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Interval1d, Square, TimeInterval
from phydrax.operators.differential import fractional_laplacian
from phydrax.operators.integral import (
    build_ball_quadrature,
    integral,
    local_integral,
    local_integral_ball,
    mean,
    nonlocal_integral,
    spatial_integral,
    time_convolution,
)
from phydrax.operators.integral._quadrature import build_quadrature as build_mc_quadrature


def test_integral_and_mean_constant_1d(sample_batch):
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    batch = sample_batch(component, blocks=(("x",),), num_points=4096, key=0)

    I1 = jnp.asarray(integral(1.0, batch, component=component, over="x").data)
    assert jnp.allclose(I1, 1.0, atol=1e-12, rtol=0.0)

    u = DomainFunction(domain=geom, deps=(), func=2.0)
    m = jnp.asarray(mean(u, batch, component=component, over="x").data)
    assert jnp.allclose(m, 2.0, atol=1e-12, rtol=0.0)


def test_integral_constant_2d_square(sample_batch):
    geom = Square(center=(0.0, 0.0), side=2.0)  # area=4
    component = geom.component()
    batch = sample_batch(component, blocks=(("x",),), num_points=8192, key=1)
    val = jnp.asarray(integral(1.0, batch, component=component, over="x").data)
    assert jnp.allclose(val, 4.0, atol=1e-2, rtol=0.0)


def test_spatial_integral_nonlocal_kernel_matches_dense_mc_1d():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.sin(jnp.pi * x[0])

    def K_xy(z):
        x, y = z[0], z[1]
        return jnp.exp(-((x - y) ** 2))

    quad_coarse = build_mc_quadrature(geom, num_points=4096, seed=1)
    quad_dense = build_mc_quadrature(geom, num_points=20000, seed=1)
    I_coarse = spatial_integral(u, quad=quad_coarse, kernel=K_xy)
    I_dense = spatial_integral(u, quad=quad_dense, kernel=K_xy)

    xs = jnp.linspace(0.0, 1.0, 5)[:, None]
    pts = frozendict({"x": cx.Field(xs, dims=("n", None))})
    v_coarse = jnp.asarray(I_coarse(pts).data)
    v_dense = jnp.asarray(I_dense(pts).data)

    rel = jnp.linalg.norm(v_coarse - v_dense) / (jnp.linalg.norm(v_dense) + 1e-12)
    assert rel < 0.1


def test_time_convolution_exp_sin_closed_form():
    # k(τ)=e^{-τ}, u(t)=sin t ⇒ (k⋆u)(t)=0.5*(sin t - cos t + e^{-t})
    dom = TimeInterval(0.0, 2.0)

    @dom.Function("t")
    def u(t):
        return jnp.sin(t)

    k = lambda tau: jnp.exp(-tau)
    conv = time_convolution(k, u, order=64)

    ts = jnp.linspace(0.0, 2.0, 25)
    vals = jnp.asarray(conv(frozendict({"t": cx.Field(ts, dims=("t",))})).data)
    exact = 0.5 * (jnp.sin(ts) - jnp.cos(ts) + jnp.exp(-ts))
    err = jnp.max(jnp.abs(vals - exact))
    assert err < 2e-3


def test_fractional_laplacian_constant_zero():
    geom = Interval1d(-1.0, 1.0)
    u = DomainFunction(domain=geom, deps=(), func=jnp.array(3.14))
    fl = fractional_laplacian(u, alpha=1.2)
    xs = jnp.linspace(-0.8, 0.8, 7)[:, None]
    vals = jnp.asarray(fl(frozendict({"x": cx.Field(xs, dims=("n", None))})).data)
    assert jnp.max(jnp.abs(vals)) < 1e-12


def test_nonlocal_integral_zero_field_zero_result():
    geom = Interval1d(0.0, 1.0)
    u = DomainFunction(domain=geom, deps=(), func=jnp.array(0.0))

    delta = 0.25

    def integrand(du, xi):
        mask = (jnp.abs(xi[0]) < delta).astype(float)
        return mask * du

    quad = build_mc_quadrature(geom, num_points=2048, seed=0)
    op = nonlocal_integral(u, integrand=integrand, quad=quad)
    xs = jnp.linspace(0.1, 0.9, 5)[:, None]
    vals = jnp.asarray(op(frozendict({"x": cx.Field(xs, dims=("n", None))})).data)
    assert jnp.max(jnp.abs(vals)) < 1e-12


def test_nonlocal_integral_time_dependent_u_integrates_to_t(sample_batch):
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def u(t):
        return t

    quad = build_mc_quadrature(dom.factor("x"), num_points=4096, seed=0)
    op = nonlocal_integral(u, integrand=lambda uy: uy, quad=quad, time_var="t")

    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=4)
    out = jnp.asarray(op(batch).data)
    assert out.shape == (3, 4)

    t = jnp.asarray(batch.points["t"].data)
    assert jnp.allclose(out, t[None, :], atol=1e-12, rtol=0.0)


def test_build_mc_quadrature_subset_measure_estimate_and_cache():
    geom = Square(center=(0.0, 0.0), side=2.0)  # area=4
    where = lambda p: p[0] > 0.0  # half-space in x, expected area 2

    quad1 = build_mc_quadrature(
        geom,
        over="interior",
        num_points=5000,
        where=where,
        use_subset_measure=True,
        seed=0,
        num_subset_samples=5000,
    )
    wsum1 = float(jnp.sum(jnp.asarray(quad1["weights"])))
    assert abs(wsum1 - 2.0) / 2.0 < 0.1

    quad2 = build_mc_quadrature(
        geom,
        over="interior",
        num_points=4000,
        where=where,
        use_subset_measure=True,
        seed=123,
    )
    wsum2 = float(jnp.sum(jnp.asarray(quad2["weights"])))
    assert abs(wsum2 - 2.0) / 2.0 < 0.1

    quad3 = build_mc_quadrature(
        geom,
        over="interior",
        num_points=1000,
        where=where,
        use_subset_measure=True,
        subset_measure=2.0,
        seed=999,
    )
    wsum3 = float(jnp.sum(jnp.asarray(quad3["weights"])))
    assert abs(wsum3 - 2.0) / 2.0 < 0.1


def test_build_ball_quadrature_volume_sum():
    quad = build_ball_quadrature(radius=1.0, dim=2, num_points=8000, seed=0)
    wsum = float(jnp.sum(jnp.asarray(quad["weights"])))
    assert abs(wsum - jnp.pi) / float(jnp.pi) < 0.02


def test_local_integral_constant_field_equals_ball_volume():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(domain=geom, deps=(), func=jnp.array(2.5))

    R = 0.4
    bq = build_ball_quadrature(radius=R, dim=2, num_points=6000, seed=0)
    op = local_integral(u, integrand=lambda uy: uy, ball_quad=bq)

    val = jnp.asarray(
        op(frozendict({"x": cx.Field(jnp.array([0.1, -0.2]), dims=(None,))})).data
    )
    exact = float(jnp.pi * R * R * 2.5)
    rel = abs(float(val.item()) - exact) / exact
    assert rel < 0.03


def test_local_integral_zero_field_zero_result():
    geom = Interval1d(-1.0, 1.0)
    u = DomainFunction(domain=geom, deps=(), func=jnp.array(0.0))
    bq = build_ball_quadrature(radius=0.2, dim=1, num_points=4000, seed=0)

    def integrand(du, xi):
        return du * xi[0]

    op = local_integral(u, integrand=integrand, ball_quad=bq)
    xs = jnp.linspace(-0.3, 0.3, 7)[:, None]
    vals = jnp.asarray(op(frozendict({"x": cx.Field(xs, dims=("n", None))})).data)
    assert jnp.max(jnp.abs(vals)) < 1e-12


def test_local_integral_ball_zero_field_and_linear_field():
    geom = Square(center=(0.0, 0.0), side=2.0)
    bq = build_ball_quadrature(radius=0.25, dim=2, num_points=6000, seed=42)

    u0 = DomainFunction(domain=geom, deps=(), func=jnp.array(0.0))
    op0 = local_integral_ball(u0, f_bond=lambda du, xi: du, ball_quad=bq)
    x = frozendict({"x": cx.Field(jnp.array([0.1, -0.2]), dims=(None,))})
    assert abs(float(jnp.asarray(op0(x).data).item())) < 1e-12

    @geom.Function("x")
    def u_lin(x):
        return x[0]

    op1 = local_integral_ball(u_lin, f_bond=lambda du, xi: du, ball_quad=bq)
    assert abs(float(jnp.asarray(op1(x).data).item())) < 5e-3
