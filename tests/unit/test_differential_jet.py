#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.domain import Interval1d, Square, TimeInterval
from phydrax.operators.differential import (
    bilaplacian,
    div_diag_k_grad,
    dt_n,
    laplacian,
    partial_n,
)


def test_dt_n_second_derivative_matches_ad_and_closed_form():
    time = TimeInterval(0.0, 1.0)

    @time.Function("t")
    def u(t):
        return t**3

    d2_ad = dt_n(u, var="t", order=2, backend="ad")
    d2_jet = dt_n(u, var="t", order=2, backend="jet")

    t = jnp.asarray(0.37)
    expected = 6.0 * t
    assert jnp.allclose(d2_jet.func(t), expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(d2_ad.func(t), expected, rtol=1e-6, atol=1e-6)


def test_partial_n_coord_separable_second_derivative_matches_closed_form():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 3

    d2_jet = partial_n(u, var="x", order=2, backend="jet")

    coords = jnp.linspace(0.0, 1.0, 8)
    out = d2_jet.func((coords,))
    assert jnp.allclose(out, 6.0 * coords, rtol=1e-6, atol=1e-6)


def test_laplacian_backend_jet_matches_ad():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 3

    lap_ad = laplacian(u, var="x", backend="ad")
    lap_jet = laplacian(u, var="x", backend="jet")

    x = jnp.asarray([0.2])
    expected = 6.0 * x[0]
    assert jnp.allclose(lap_ad.func(x), expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(lap_jet.func(x), expected, rtol=1e-6, atol=1e-6)

    coords = jnp.linspace(0.0, 1.0, 5)
    assert jnp.allclose(lap_jet.func((coords,)), 6.0 * coords, rtol=1e-6, atol=1e-6)


def test_div_diag_k_grad_backend_jet_constant_k_matches_laplacian():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 3

    @geom.Function("x")
    def k_vec(x):
        return jnp.array([1.0], dtype=float)

    op = div_diag_k_grad(u, k_vec, var="x", backend="jet")
    lap = laplacian(u, var="x", backend="jet")

    x = jnp.asarray([0.7])
    assert jnp.allclose(op.func(x), lap.func(x), rtol=1e-6, atol=1e-6)

    coords = jnp.linspace(0.0, 1.0, 6)
    expected = 6.0 * coords
    assert jnp.allclose(op.func((coords,)), expected, rtol=1e-6, atol=1e-6)


def test_bilaplacian_backend_jet_matches_ad_1d():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 4

    bilap_ad = bilaplacian(u, var="x", backend="ad")
    bilap_jet = bilaplacian(u, var="x", backend="jet")

    x = jnp.asarray([0.2])
    expected = 24.0
    assert jnp.allclose(bilap_ad.func(x), expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(bilap_jet.func(x), expected, rtol=1e-6, atol=1e-6)

    coords = jnp.linspace(0.0, 1.0, 7)
    assert jnp.allclose(bilap_jet.func((coords,)), expected, rtol=1e-6, atol=1e-6)


def test_bilaplacian_backend_jet_matches_ad_2d_mixed_terms():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 4 + x[1] ** 4 + 2.0 * x[0] ** 2 * x[1] ** 2

    bilap_ad = bilaplacian(u, var="x", backend="ad")
    bilap_jet = bilaplacian(u, var="x", backend="jet")

    x = jnp.asarray([0.2, -0.1])
    expected = 64.0
    assert jnp.allclose(bilap_ad.func(x), expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(bilap_jet.func(x), expected, rtol=1e-6, atol=1e-6)

    coords_x = jnp.linspace(-1.0, 1.0, 4)
    coords_y = jnp.linspace(-0.5, 0.5, 3)
    assert jnp.allclose(
        bilap_jet.func((coords_x, coords_y)), expected, rtol=1e-6, atol=1e-6
    )
