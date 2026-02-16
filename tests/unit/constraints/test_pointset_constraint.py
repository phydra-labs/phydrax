#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.constraints import PointSetConstraint
from phydrax.domain import Interval1d


def test_pointset_penalty_mean_and_sum():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    points = {"x": jnp.array([[0.0], [0.5], [1.0]], dtype=float)}
    u = geom.Function()(0.0)

    def residual(functions):
        return functions["u"] - 1.0

    c_mean = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        reduction="mean",
        weight=2.0,
    )
    loss_mean = c_mean.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_mean, 2.0)

    c_sum = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        reduction="sum",
        weight=2.0,
    )
    loss_sum = c_sum.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_sum, 6.0)


def test_pointset_domainfunction_weight_mean_and_sum():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    points = {"x": jnp.array([[0.0], [0.5], [1.0]], dtype=float)}
    u = geom.Function()(0.0)

    @geom.Function("x")
    def w(x):
        xx = _x_values(x)
        return xx + 1.0

    def residual(functions):
        return functions["u"] - 1.0

    c_mean = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        reduction="mean",
        weight=w,
    )
    loss_mean = c_mean.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_mean, 1.5)

    c_sum = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        reduction="sum",
        weight=w,
    )
    loss_sum = c_sum.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_sum, 4.5)


def test_pointset_domainfunction_weight_must_be_scalar_per_point():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    points = {"x": jnp.array([[0.0], [0.5], [1.0]], dtype=float)}
    u = geom.Function()(0.0)

    @geom.Function("x")
    def bad_w(x):
        xx = _x_values(x)
        return jnp.stack((xx, xx + 1.0), axis=-1)

    constraint = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=lambda fns: fns["u"] - 1.0,
        reduction="mean",
        weight=bad_w,
    )
    with pytest.raises(ValueError, match="scalar per point"):
        _ = constraint.loss({"u": u}, key=jr.key(0))


def _x_values(x):
    x_arr = jnp.asarray(x, dtype=float)
    if x_arr.ndim == 0:
        return x_arr.reshape(())
    if x_arr.ndim == 1:
        if int(x_arr.shape[0]) == 1:
            return x_arr[0]
        return x_arr
    if x_arr.ndim == 2 and int(x_arr.shape[1]) == 1:
        return x_arr[:, 0]
    raise ValueError(f"Unsupported x shape {x_arr.shape}.")
