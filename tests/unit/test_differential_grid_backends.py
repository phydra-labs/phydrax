#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.domain import (
    CosineAxisSpec,
    FourierAxisSpec,
    Interval1d,
    LegendreAxisSpec,
    SineAxisSpec,
)
from phydrax.operators.differential import laplacian, partial_n


def test_partial_n_basis_fourier_matches_closed_form_periodic():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.sin(2.0 * jnp.pi * x[0])

    d2 = partial_n(
        u,
        var="x",
        order=2,
        backend="basis",
        basis="fourier",
        periodic=True,
    )
    coords = FourierAxisSpec(64).materialize(jnp.array(0.0), jnp.array(1.0)).nodes
    out = d2.func((coords,))
    expected = -((2.0 * jnp.pi) ** 2) * jnp.sin(2.0 * jnp.pi * coords)
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_partial_n_fd_matches_closed_form_periodic_to_tolerance():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.sin(2.0 * jnp.pi * x[0])

    d1 = partial_n(
        u,
        var="x",
        order=1,
        backend="fd",
        periodic=True,
    )
    coords = FourierAxisSpec(256).materialize(jnp.array(0.0), jnp.array(1.0)).nodes
    out = d1.func((coords,))
    expected = 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * coords)
    assert jnp.allclose(out, expected, rtol=2e-3, atol=2e-3)


def test_partial_n_basis_poly_on_legendre_nodes_matches_closed_form():
    geom = Interval1d(-1.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 3

    d2 = partial_n(
        u,
        var="x",
        order=2,
        backend="basis",
        basis="poly",
    )
    coords = LegendreAxisSpec(12).materialize(jnp.array(-1.0), jnp.array(1.0)).nodes
    out = d2.func((coords,))
    expected = 6.0 * coords
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_laplacian_basis_matches_second_partial_in_1d():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.sin(2.0 * jnp.pi * x[0])

    lap = laplacian(
        u,
        var="x",
        backend="basis",
        basis="fourier",
        periodic=True,
    )
    d2 = partial_n(
        u,
        var="x",
        order=2,
        backend="basis",
        basis="fourier",
        periodic=True,
    )
    coords = FourierAxisSpec(64).materialize(jnp.array(0.0), jnp.array(1.0)).nodes
    out_lap = lap.func((coords,))
    out_d2 = d2.func((coords,))
    assert jnp.allclose(out_lap, out_d2, rtol=1e-8, atol=1e-8)


def test_partial_n_basis_cosine_matches_closed_form():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.cos(3.0 * jnp.pi * x[0])

    d1 = partial_n(
        u,
        var="x",
        order=1,
        backend="basis",
        basis="cosine",
    )
    coords = CosineAxisSpec(129).materialize(jnp.array(0.0), jnp.array(1.0)).nodes
    out = d1.func((coords,))
    expected = -3.0 * jnp.pi * jnp.sin(3.0 * jnp.pi * coords)
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_partial_n_basis_sine_matches_closed_form():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.sin(4.0 * jnp.pi * x[0])

    d1 = partial_n(
        u,
        var="x",
        order=1,
        backend="basis",
        basis="sine",
    )
    coords = SineAxisSpec(128).materialize(jnp.array(0.0), jnp.array(1.0)).nodes
    out = d1.func((coords,))
    expected = 4.0 * jnp.pi * jnp.cos(4.0 * jnp.pi * coords)
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)
