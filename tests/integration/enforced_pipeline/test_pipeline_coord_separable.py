#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx

# Enforced pipeline integration tests.
from phydrax.domain import Interval1d, SampleLayout
from phydrax.operators.differential import partial_x


def test_coord_separable_matches_dense_partial_x():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    sep = component.sample(phx.domain.GridSampling({"x": 8}, design="latin_hypercube"))
    dense = component.sample(phx.domain.PointSampling(8, layout=SampleLayout((("x",),))))

    du = partial_x(u, var="x")
    sep_val = jnp.asarray(du(sep).data).reshape((-1,))
    dense_val = jnp.asarray(du(dense).data).reshape((-1,))

    assert jnp.allclose(sep_val, dense_val, atol=1e-5)
