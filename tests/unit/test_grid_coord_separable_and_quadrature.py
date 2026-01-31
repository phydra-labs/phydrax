#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.domain import FourierAxisSpec, Interval1d, LegendreAxisSpec
from phydrax.operators.integral import integral


def test_coord_separable_fourier_axis_spec_interval_discretization_attached():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    batch = component.sample_coord_separable({"x": FourierAxisSpec(8)})
    (axis,) = batch.coord_axes_by_label["x"]

    x_field = batch.points["x"][0]
    assert x_field.dims == (axis,)
    assert x_field.data.shape == (8,)

    disc = batch.axis_discretization_by_axis[axis]
    assert disc.basis == "fourier"
    assert bool(disc.periodic) is True
    assert disc.quad_weights is not None
    assert disc.nodes.shape == x_field.data.shape
    assert jnp.allclose(disc.nodes, jnp.asarray(x_field.data, dtype=float))


def test_coord_separable_legendre_axis_spec_integral_matches_closed_form():
    geom = Interval1d(-1.0, 2.0)
    component = geom.component()

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    batch = component.sample_coord_separable({"x": LegendreAxisSpec(6)})
    out = integral(u, batch, component=component)
    expected = (2.0**3 - (-1.0) ** 3) / 3.0
    assert jnp.allclose(jnp.asarray(out.data), expected, rtol=1e-7, atol=1e-7)


def test_sdf_domain_function_evaluates_on_coord_separable_batch():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    batch = component.sample_coord_separable({"x": FourierAxisSpec(8)})
    (axis,) = batch.coord_axes_by_label["x"]

    phi = component.sdf(var="x")
    out = phi(batch)
    assert out.dims == (axis,)
    values = jnp.asarray(out.data, dtype=float)
    assert jnp.all(values <= 0.0 + 1e-8)
    assert jnp.any(values < 0.0)
