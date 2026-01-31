#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from phydrax.constraints._discrete_interior import DiscreteInteriorDataConstraint
from phydrax.domain import Interval1d, ProductStructure, TimeInterval


def _xt_domain():
    return Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)


def test_sensor_tracks_hermite_interpolates_linear_time():
    domain = _xt_domain()
    structure = ProductStructure((("x", "t"),))

    @domain.Function("x", "t")
    def u(x, t):
        return 2.0 * t

    constraint = DiscreteInteriorDataConstraint(
        "u",
        domain,
        sensors=jnp.array([[0.0]], dtype=float),
        times=jnp.array([0.0, 1.0], dtype=float),
        sensor_values=jnp.array([[0.0, 2.0]], dtype=float),
        num_points=16,
        structure=structure,
    )
    loss_fn = eqx.filter_jit(lambda k: constraint.loss({"u": u}, key=k))
    assert loss_fn(jr.key(0)) < 1e-6


def test_sensor_tracks_single_time_constant():
    domain = _xt_domain()
    structure = ProductStructure((("x", "t"),))

    @domain.Function("x", "t")
    def u(x, t):
        return 3.0

    constraint = DiscreteInteriorDataConstraint(
        "u",
        domain,
        sensors=jnp.array([[0.0]], dtype=float),
        times=jnp.array([0.0], dtype=float),
        sensor_values=jnp.array([[3.0]], dtype=float),
        num_points=16,
        structure=structure,
    )
    loss_fn = eqx.filter_jit(lambda k: constraint.loss({"u": u}, key=k))
    assert loss_fn(jr.key(0)) < 1e-6
