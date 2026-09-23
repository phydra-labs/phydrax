#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.discretization import FourierAxisSpec, LegendreAxisSpec
from phydrax.domain import (
    Interval1d,
    SampleLayout,
    TimeInterval,
)
from phydrax.integration import from_samples, over
from phydrax.operators.integral import integral


def test_coord_separable_fourier_axis_spec_interval_discretization_attached():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    batch = component.sample(phx.domain.GridSampling({"x": FourierAxisSpec(8)}))
    (axis,) = batch.coord_axes_by_label["x"]

    x_field = batch.points["x"][0]
    assert x_field.dims == (axis,)
    assert x_field.data.shape == (8,)

    disc = batch.axis_discretization_by_axis[axis]
    assert disc.basis == "fourier"
    assert bool(disc.periodic) is True
    assert disc.quad_weights.shape == (8,)
    assert jnp.all(jnp.isfinite(disc.quad_weights))
    assert jnp.allclose(disc.quad_weights, jnp.full((8,), 1.0 / 8.0))
    assert jnp.sum(disc.quad_weights) == pytest.approx(1.0, abs=1e-12)
    assert disc.nodes.shape == x_field.data.shape
    assert jnp.allclose(disc.nodes, jnp.asarray(x_field.data, dtype="float64"))


def test_coord_separable_legendre_axis_spec_integral_matches_closed_form():
    geom = Interval1d(-1.0, 2.0)
    component = geom.component()

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    batch = component.sample(phx.domain.GridSampling({"x": LegendreAxisSpec(6)}))
    realization = from_samples(over(component), batch)
    out = integral(u, realization)
    expected = (2.0**3 - (-1.0) ** 3) / 3.0
    assert jnp.allclose(jnp.asarray(out.data), expected, rtol=1e-7, atol=1e-7)


def test_legendre_axis_endpoint_rules_and_validation():
    lower = jnp.asarray(-2.0)
    upper = jnp.asarray(3.0)
    radau = LegendreAxisSpec(5, kind="radau").materialize(lower, upper)
    lobatto = LegendreAxisSpec(5, kind="lobatto").materialize(lower, upper)

    assert radau.nodes[0] == lower
    assert lobatto.nodes[0] == lower
    assert lobatto.nodes[-1] == upper
    assert jnp.sum(radau.quad_weights) == pytest.approx(5.0, abs=1e-12)
    assert jnp.sum(lobatto.quad_weights) == pytest.approx(5.0, abs=1e-12)
    with pytest.raises(ValueError, match="kind"):
        LegendreAxisSpec(4, kind="typo")
    with pytest.raises(ValueError, match="at least two"):
        LegendreAxisSpec(1, kind="lobatto")


def test_sdf_domain_function_preserves_interval_sign_and_distance():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    batch = component.points({"x": jnp.asarray([[-0.25], [0.0], [0.25], [1.0], [1.25]])})

    values = jnp.asarray(component.sdf(var="x")(batch).data, dtype="float64")
    assert jnp.allclose(values, jnp.asarray([0.25, 0.0, -0.25, 0.0, 0.25]))


def test_coord_separable_scalar_time_axis_integral_constant():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 2.0)
    domain = geom @ time
    component = domain.component()

    @domain.Function("x", "t")
    def u(x, t):
        del x, t
        return 1.0

    batch = component.sample(
        phx.domain.GridSampling(
            {"t": LegendreAxisSpec(8)},
            dense=phx.domain.PointSampling(5, layout=SampleLayout((("x",),))),
        )
    )
    realization = from_samples(over(component), batch)
    out = integral(u, realization)
    assert jnp.allclose(jnp.asarray(out.data), 2.0, rtol=1e-7, atol=1e-7)
