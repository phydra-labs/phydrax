#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.conditions import Residual
from phydrax.domain import Interval1d
from phydrax.terms import ResidualPenalty


def _fixed_source(component, values):
    batch = component.points(
        {"x": jnp.asarray(values, dtype=float).reshape((-1, 1))}
    )
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component),
        batch,
    )
    return phx.integration.fixed(realization)


def test_pointset_penalty_mean_and_sum():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    source = _fixed_source(component, [0.0, 0.5, 1.0])
    u = geom.Function()(0.0)
    condition = Residual("u", component, lambda value: value - 1.0)

    mean_term = ResidualPenalty(condition, source, scale=2.0)
    loss_mean = mean_term.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_mean, 2.0)

    sum_term = ResidualPenalty(condition, source, scale=6.0)
    loss_sum = sum_term.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_sum, 6.0)


def test_pointset_domainfunction_weight_mean_and_sum():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    source = _fixed_source(component, [0.0, 0.5, 1.0])
    u = geom.Function()(0.0)

    @geom.Function("x")
    def density(x):
        xx = _x_values(x)
        return xx + 1.0

    condition = Residual("u", component, lambda value: value - 1.0)
    mean_term = ResidualPenalty(condition, source, density=density)
    loss_mean = mean_term.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_mean, 1.5)

    sum_term = ResidualPenalty(condition, source, scale=3.0, density=density)
    loss_sum = sum_term.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_sum, 4.5)


def test_pointset_domainfunction_weight_must_be_scalar_per_point():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    source = _fixed_source(component, [0.0, 0.5, 1.0])
    u = geom.Function()(0.0)

    @geom.Function("x")
    def bad_density(x):
        xx = _x_values(x)
        return jnp.stack((xx, xx + 1.0), axis=-1)

    condition = Residual("u", component, lambda value: value - 1.0)
    term = ResidualPenalty(condition, source, density=bad_density)
    with pytest.raises(ValueError, match="scalar"):
        _ = term.loss({"u": u}, key=jr.key(0))


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
