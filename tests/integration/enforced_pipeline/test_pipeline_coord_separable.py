#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

# Enforced pipeline integration tests.
from phydrax.domain import Interval1d, ProductStructure
from phydrax.operators.differential import partial_x


def test_coord_separable_matches_dense_partial_x():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    sep = component.sample_coord_separable(
        {"x": 8}, num_points=(), sampler="latin_hypercube"
    )
    dense = component.sample(8, structure=ProductStructure((("x",),)))

    du = partial_x(u, var="x")
    sep_val = jnp.asarray(du(sep).data).reshape((-1,))
    dense_val = jnp.asarray(du(dense).data).reshape((-1,))

    assert jnp.allclose(sep_val, dense_val, atol=1e-5)
