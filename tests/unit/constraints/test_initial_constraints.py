#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.constraints import ContinuousInitialConstraint
from phydrax.domain import FixedStart, Interval1d, SampleLayout, TimeInterval


def test_continuous_initial_constraint_zero_when_satisfied():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    dom = geom @ time

    component = dom.component({"t": FixedStart()})
    structure = SampleLayout((("x",),))

    u = dom.Function()(1.0)
    c = ContinuousInitialConstraint(
        "u",
        component,
        func=1.0,
        sampling=phx.domain.PointSampling(8, layout=structure),
    )
    loss = c.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)


def test_continuous_initial_constraint_requires_fixed_start():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    dom = geom @ time

    component = dom.component()
    structure = SampleLayout((("x", "t"),))
    u = dom.Function()(0.0)

    with pytest.raises(ValueError):
        ContinuousInitialConstraint(
            "u",
            component,
            func=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
        )
