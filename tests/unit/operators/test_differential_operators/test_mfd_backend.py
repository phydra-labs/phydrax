#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.domain import FourierAxisSpec, Interval1d
from phydrax.operators.differential import laplacian, partial_n


def test_partial_n_mfd_matches_closed_form_periodic():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.sin(2.0 * jnp.pi * x[0])

    d1 = partial_n(
        u,
        var="x",
        order=1,
        backend="mfd",
        periodic=True,
    )
    coords = FourierAxisSpec(256).materialize(jnp.array(0.0), jnp.array(1.0)).nodes
    out = d1.func((coords,))
    expected = 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * coords)
    assert jnp.allclose(out, expected, rtol=2e-3, atol=2e-3)


def test_partial_n_mfd_biased_boundary_polynomial_exact():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 3

    d1 = partial_n(
        u,
        var="x",
        order=1,
        backend="mfd",
        periodic=False,
    )
    coords = jnp.linspace(0.0, 1.0, 64)
    out = d1.func(
        (coords,),
        mfd_boundary_mode="biased",
        mfd_stencil_size=5,
    )
    expected = 3.0 * coords**2
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_laplacian_mfd_matches_second_partial_in_1d_periodic():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.sin(2.0 * jnp.pi * x[0])

    lap = laplacian(
        u,
        var="x",
        backend="mfd",
        periodic=True,
    )
    d2 = partial_n(
        u,
        var="x",
        order=2,
        backend="mfd",
        periodic=True,
    )
    coords = FourierAxisSpec(128).materialize(jnp.array(0.0), jnp.array(1.0)).nodes
    out_lap = lap.func((coords,))
    out_d2 = d2.func((coords,))
    assert jnp.allclose(out_lap, out_d2, rtol=1e-6, atol=1e-6)


def test_partial_n_mfd_point_matches_closed_form():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 3

    d1 = partial_n(u, var="x", order=1, backend="mfd")
    x = jnp.array([0.4], dtype=float)
    out = jnp.asarray(d1.func(x, mfd_step=1e-4, mfd_stencil_size=5))
    expected = jnp.asarray(3.0 * x[0] ** 2)
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)


def test_partial_n_mfd_point_biased_boundary_polynomial_exact():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 3

    d1 = partial_n(u, var="x", order=1, backend="mfd")
    x = jnp.array([1e-3], dtype=float)
    out = jnp.asarray(
        d1.func(
            x,
            mfd_boundary_mode="biased",
            mfd_step=1e-2,
            mfd_stencil_size=5,
        )
    )
    expected = jnp.asarray(3.0 * x[0] ** 2)
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_partial_n_mfd_point_vectorized_input_matches_closed_form():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        x_arr = jnp.asarray(x)
        if x_arr.ndim == 1 and int(x_arr.shape[0]) == 1:
            xx = x_arr[0]
        else:
            xx = x_arr
        return xx**3

    d1 = partial_n(u, var="x", order=1, backend="mfd")
    x = jnp.linspace(0.0, 1.0, 64)
    out = jnp.asarray(
        d1.func(
            x,
            mfd_boundary_mode="biased",
            mfd_step=1.0 / 64.0,
            mfd_stencil_size=5,
        )
    )
    expected = 3.0 * x**2
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)


def test_partial_n_mfd_rejects_ad_engine_override():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    with pytest.raises(ValueError, match="backend='ad'"):
        partial_n(u, var="x", order=1, backend="mfd", ad_engine="jvp")


def test_partial_n_mfd_tuple_preserves_trailing_channels():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        xx = x[0]
        return jnp.stack(
            (
                xx**3,
                jnp.sin(2.0 * jnp.pi * xx),
            ),
            axis=-1,
        )

    d1 = partial_n(u, var="x", order=1, backend="mfd")
    coords = jnp.linspace(0.0, 1.0, 128)
    out = d1.func(
        (coords,),
        mfd_boundary_mode="ghost",
        mfd_stencil_size=5,
    )
    expected = jnp.stack(
        (
            3.0 * coords**2,
            2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * coords),
        ),
        axis=-1,
    )

    assert out.shape == expected.shape
    assert jnp.allclose(out[2:-2, 0], expected[2:-2, 0], rtol=1e-6, atol=1e-6)
    assert jnp.allclose(out[2:-2, 1], expected[2:-2, 1], rtol=2e-3, atol=2e-3)


def test_partial_n_mfd_tuple_channels_biased_boundary_polynomial_exact():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        xx = x[0]
        return jnp.stack(
            (
                xx**3,
                jnp.sin(2.0 * jnp.pi * xx),
            ),
            axis=-1,
        )

    d1 = partial_n(u, var="x", order=1, backend="mfd")
    coords = jnp.linspace(0.0, 1.0, 128)
    out = d1.func(
        (coords,),
        mfd_boundary_mode="biased",
        mfd_stencil_size=5,
    )
    expected_poly = 3.0 * coords**2
    expected_sin = 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * coords)

    assert out.shape == (coords.shape[0], 2)
    assert jnp.allclose(out[:, 0], expected_poly, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(out[2:-2, 1], expected_sin[2:-2], rtol=2e-3, atol=2e-3)


def test_partial_n_mfd_tuple_hybrid_matches_ghost():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 4 + x[0]

    d1 = partial_n(u, var="x", order=1, backend="mfd")
    coords = jnp.linspace(0.0, 1.0, 96)

    out_ghost = d1.func(
        (coords,),
        mfd_boundary_mode="ghost",
        mfd_stencil_size=5,
    )
    out_hybrid = d1.func(
        (coords,),
        mfd_boundary_mode="hybrid",
        mfd_stencil_size=5,
    )
    assert jnp.allclose(out_hybrid, out_ghost, rtol=0.0, atol=0.0)
