#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

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


def test_grid_coord_separable_and_quadrature_scenario_1() -> None:
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
    # ty: ignore[unresolved-attribute]
    assert disc.quad_weights.shape == (8,)
    # ty: ignore[invalid-argument-type]
    assert jnp.all(jnp.isfinite(disc.quad_weights))
    # ty: ignore[invalid-argument-type]
    assert jnp.allclose(disc.quad_weights, jnp.full((8,), 1.0 / 8.0))
    # ty: ignore[invalid-argument-type]
    assert jnp.sum(disc.quad_weights) == pytest.approx(1.0, abs=1e-12)
    assert disc.nodes.shape == x_field.data.shape
    assert jnp.allclose(disc.nodes, jnp.asarray(x_field.data, dtype="float64"))
    lower = jnp.asarray(-2.0)
    upper = jnp.asarray(3.0)
    radau = LegendreAxisSpec(5, kind="radau").materialize(lower, upper)
    lobatto = LegendreAxisSpec(5, kind="lobatto").materialize(lower, upper)

    assert radau.nodes[0] == lower
    assert lobatto.nodes[0] == lower
    assert lobatto.nodes[-1] == upper
    # ty: ignore[invalid-argument-type]
    assert jnp.sum(radau.quad_weights) == pytest.approx(5.0, abs=1e-12)
    # ty: ignore[invalid-argument-type]
    assert jnp.sum(lobatto.quad_weights) == pytest.approx(5.0, abs=1e-12)
    with pytest.raises(ValueError, match="kind"):
        # ty: ignore[invalid-argument-type]
        LegendreAxisSpec(4, kind="typo")
    with pytest.raises(ValueError, match="at least two"):
        LegendreAxisSpec(1, kind="lobatto")
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    epsilon = 1e-4
    points = jnp.asarray(
        [-0.25, -epsilon, 0.0, epsilon, 0.5, 1.0 - epsilon, 1.0, 1.0 + epsilon, 1.25]
    )
    batch = component.points({"x": points[:, None]})

    values = jnp.asarray(component.sdf(var="x")(batch).data, dtype="float64")
    assert jnp.array_equal(
        jnp.sign(values), jnp.asarray([1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0])
    )
    near_boundary = jnp.asarray([1, 3, 5, 7])
    assert jnp.allclose(jnp.abs(values[near_boundary]), epsilon, rtol=1e-3)


def test_coord_separable_legendre_axis_spec_integral_matches_closed_form() -> None:
    geom = Interval1d(-1.0, 2.0)
    component = geom.component()

    @geom.Function("x")
    def u(x: Any) -> Any:
        return x[0] ** 2

    batch = component.sample(phx.domain.GridSampling({"x": LegendreAxisSpec(6)}))
    realization = from_samples(over(component), batch)
    out = integral(u, realization)
    expected = (2.0**3 - (-1.0) ** 3) / 3.0
    assert jnp.allclose(jnp.asarray(out.data), expected, rtol=1e-7, atol=1e-7)


def test_coord_separable_scalar_time_axis_integral_constant() -> None:
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 2.0)
    domain = geom @ time
    component = domain.component()

    @domain.Function("x", "t")
    def u(x: Any, t: Any) -> float:
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
