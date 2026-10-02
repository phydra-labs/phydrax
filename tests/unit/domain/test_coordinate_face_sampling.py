#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coordinate-face tensor grids and reduced tangential quadrature.

Oracles are the declared box bounds: a face ``x_c = a_c`` of ``prod_k [a_k, b_k]``
is the transverse box ``prod_{k != c} [a_k, b_k]``, so face integrals are ordinary
lower-dimensional integrals with ``x_c`` substituted, and a zero-dimensional face
integral is the point value.
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import UniformAxisSpec


_BOX3 = phx.domain.HyperRectangle(np.zeros(3), np.asarray([2.0, 3.0, 5.0]))
_GAUSS = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4))


def test_coordinate_face_grid_fixes_face_and_discretizes_tangential_axes() -> None:
    face = _BOX3.component({"x": phx.domain.CoordinateFace(1, "upper")})
    batch = face.sample(
        phx.domain.GridSampling({"x": (UniformAxisSpec(3), UniformAxisSpec(4))}),
        key=jr.key(0),
    )

    assert isinstance(batch, phx.domain.GridBatch)
    axes = batch.coord_axes_by_label["x"]
    assert len(axes) == 3
    first, normal, third = (np.asarray(field.data) for field in batch.points["x"])
    np.testing.assert_allclose(first, np.linspace(0.0, 2.0, 3))
    np.testing.assert_array_equal(normal, [3.0])
    np.testing.assert_allclose(third, np.linspace(0.0, 5.0, 4))
    assert batch.coord_mask_by_label["x"].data.shape == (3, 1, 4)
    assert set(batch.axis_discretization_by_axis) == {axes[0], axes[2]}

    values = _BOX3.Function("x")(lambda x: x[1])(batch)
    np.testing.assert_array_equal(np.asarray(values.data), np.full((3, 1, 4), 3.0))


def test_coordinate_face_grid_counts_cover_tangential_axes_only() -> None:
    face = _BOX3.component({"x": phx.domain.CoordinateFace(0, "lower")})
    batch = face.sample(phx.domain.GridSampling({"x": (6, 7)}), key=jr.key(1))
    first, second, third = (np.asarray(field.data) for field in batch.points["x"])

    np.testing.assert_array_equal(first, [0.0])
    assert second.shape == (6,) and np.all((second >= 0.0) & (second <= 3.0))
    assert third.shape == (7,) and np.all((third >= 0.0) & (third <= 5.0))
    with pytest.raises(ValueError, match="2 tangential components"):
        face.sample(phx.domain.GridSampling({"x": (6, 7, 8)}), key=jr.key(1))


def test_interval_endpoint_face_grid_is_one_point() -> None:
    interval = phx.domain.Interval1d(0.0, 2.0)
    endpoint = interval.component({"x": phx.domain.CoordinateFace(0, "upper")})
    batch = endpoint.sample(phx.domain.GridSampling({"x": 5}), key=jr.key(2))

    (coordinate,) = batch.points["x"]
    np.testing.assert_array_equal(np.asarray(coordinate.data), [2.0])


@pytest.mark.parametrize(
    "plan",
    (
        pytest.param(_GAUSS, id="gauss-legendre"),
        pytest.param(
            phx.integration.FixedQuadraturePlan(
                phx.integration.CubatureRule("tensor", 5, dimension=1)
            ),
            id="tensor-cubature",
        ),
    ),
)
@pytest.mark.parametrize(
    ("axis", "side", "integrand", "mean", "expected"),
    (
        # y = 2 face of [0, 1] x [0, 2]: integral of x^2 over [0, 1].
        pytest.param(1, "upper", lambda x: x[0] ** 2, False, 1.0 / 3.0, id="y-face"),
        # x = 1 face: mean of y^2 over [0, 2] is (8 / 3) / 2.
        pytest.param(0, "upper", lambda x: x[1] ** 2, True, 4.0 / 3.0, id="x-face"),
        # x = 0 face: x is substituted, so x + y integrates y over [0, 2].
        pytest.param(0, "lower", lambda x: x[0] + x[1], False, 2.0, id="x0-face"),
    ),
)
def test_fixed_quadrature_on_2d_box_face_matches_line_integral(
    plan: Any,
    axis: int,
    side: Any,
    integrand: Callable[[Any], Any],
    mean: bool,
    expected: float,
) -> None:
    box = phx.domain.HyperRectangle(np.zeros(2), np.asarray([1.0, 2.0]))
    face = box.component({"x": phx.domain.CoordinateFace(axis, side)})
    target = (phx.integration.mean_over if mean else phx.integration.over)(face)
    estimate = phx.integration.integrate(box.Function("x")(integrand), target, plan)

    np.testing.assert_allclose(estimate.value.data, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "plan",
    (
        pytest.param(_GAUSS, id="gauss-legendre"),
        pytest.param(
            phx.integration.FixedQuadraturePlan(
                phx.integration.CubatureRule("tensor", 5, dimension=2)
            ),
            id="tensor-cubature",
        ),
    ),
)
def test_fixed_quadrature_on_3d_box_face_matches_surface_integral(plan: Any) -> None:
    face = _BOX3.component({"x": phx.domain.CoordinateFace(1, "upper")})
    function = _BOX3.Function("x")(lambda x: x[0] * x[2] ** 2 + x[1])
    # int_0^2 int_0^5 (x z^2 + 3) dz dx = 2 * 125 / 3 + 3 * 10
    expected = 250.0 / 3.0 + 30.0

    integral = phx.integration.integrate(function, phx.integration.over(face), plan)
    mean = phx.integration.integrate(function, phx.integration.mean_over(face), plan)

    np.testing.assert_allclose(integral.value.data, expected, rtol=1e-12)
    np.testing.assert_allclose(mean.value.data, expected / 10.0, rtol=1e-12)


@pytest.mark.parametrize(
    "plan",
    (
        pytest.param(_GAUSS, id="gauss-legendre"),
        pytest.param(
            phx.integration.FixedQuadraturePlan(
                phx.integration.CubatureRule("tensor", 5, dimension=1)
            ),
            id="tensor-cubature",
        ),
    ),
)
@pytest.mark.parametrize(("side", "point"), (("lower", 0.5), ("upper", 2.0)))
def test_interval_endpoint_face_integral_is_point_value(
    plan: Any, side: Any, point: float
) -> None:
    interval = phx.domain.Interval1d(0.5, 2.0)
    endpoint = interval.component({"x": phx.domain.CoordinateFace(0, side)})
    function = interval.Function("x")(lambda x: jnp.sin(x[0]) + 3.0)

    estimate = phx.integration.integrate(function, phx.integration.over(endpoint), plan)

    np.testing.assert_allclose(estimate.value.data, np.sin(point) + 3.0, rtol=1e-14)


def test_coordinate_face_cubature_requires_matching_tensor_rule() -> None:
    face = _BOX3.component({"x": phx.domain.CoordinateFace(2, "lower")})
    plan = phx.integration.FixedQuadraturePlan(
        phx.integration.CubatureRule("tensor", 5, dimension=3)
    )

    with pytest.raises(ValueError, match="2-dimensional tensor rule"):
        phx.integration.integrate(1.0, phx.integration.over(face), plan)
