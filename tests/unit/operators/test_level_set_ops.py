#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict


def _points_1d(values):
    return frozendict({"x": cx.Field(values[:, None], dims=("point", None))})


def test_compact_level_set_regularizations_are_complementary_and_differentiable():
    domain = phx.domain.Interval1d(-1.0, 1.0)

    @domain.Function("x")
    def level_set(point):
        return point[0]

    width = 0.2
    values = jnp.asarray((-0.5, -0.1, 0.0, 0.1, 0.5))
    batch = _points_1d(values)
    positive = phx.operators.regularized_heaviside(level_set, width=width)
    negative = phx.operators.level_set_phase_indicator(
        level_set,
        width=width,
        phase="inside",
    )
    delta = phx.operators.regularized_delta(level_set, width=width)
    derivative = phx.operators.grad(positive, var="x")

    positive_values = jnp.asarray(positive(batch).data)
    negative_values = jnp.asarray(negative(batch).data)
    delta_values = jnp.asarray(delta(batch).data)
    derivative_values = jnp.asarray(derivative(batch).data)[..., 0]

    np.testing.assert_allclose(positive_values + negative_values, 1.0, atol=1.0e-14)
    np.testing.assert_allclose(derivative_values, delta_values, atol=1.0e-12)
    assert positive_values[0] == 0.0
    assert positive_values[-1] == 1.0
    assert delta_values[0] == 0.0
    assert delta_values[-1] == 0.0
    assert delta_values[2] == pytest.approx(1.0 / width)


def test_circle_level_set_geometry_and_motion_are_recovered():
    spatial = phx.domain.GeometryDomain(
        phx.geometry.Circle(center=(0.0, 0.0), radius=3.0).compile()
    )
    domain = spatial @ phx.domain.TimeInterval(0.0, 1.0)

    @domain.Function("x", "t")
    def level_set(point, time):
        radius = 1.0 + time
        return jnp.sum(point * point) - radius * radius

    times = jnp.asarray((0.0, 0.25, 0.5, 0.75))
    angles = jnp.asarray((0.2, 1.1, 2.3, 5.0))
    radii = 1.0 + times
    points = radii[:, None] * jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
    batch = frozendict(
        {
            "x": cx.Field(points, dims=("point", None)),
            "t": cx.Field(times, dims=("point",)),
        }
    )

    normal = phx.operators.level_set_normal(level_set, var="x")
    curvature = phx.operators.level_set_curvature(level_set, var="x")
    velocity = phx.operators.level_set_normal_velocity(
        level_set,
        spatial_var="x",
        time_var="t",
    )
    coarea = phx.operators.level_set_coarea_density(
        level_set,
        width=0.1,
        var="x",
    )

    np.testing.assert_allclose(
        normal(batch).data,
        points / radii[:, None],
        atol=1.0e-11,
    )
    np.testing.assert_allclose(curvature(batch).data, 1.0 / radii, atol=1.0e-10)
    np.testing.assert_allclose(velocity(batch).data, jnp.ones_like(times), atol=1.0e-11)
    np.testing.assert_allclose(coarea(batch).data, 20.0 * radii, atol=1.0e-10)


def test_level_set_regularization_rejects_ambiguous_policy_values():
    domain = phx.domain.Interval1d(-1.0, 1.0)

    @domain.Function("x")
    def level_set(point):
        return point[0]

    with pytest.raises(ValueError, match="width must be finite and positive"):
        phx.operators.regularized_delta(level_set, width=0.0)
    with pytest.raises(ValueError, match="phase must be"):
        phx.operators.level_set_phase_indicator(
            level_set,
            width=0.1,
            phase="unknown",
        )
