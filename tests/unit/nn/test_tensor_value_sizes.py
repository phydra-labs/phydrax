#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.layers import Linear, RandomFourierFeatureEmbeddings
from phydrax.nn.models import MLP


def test_tensor_value_sizes_scenario_1() -> None:
    layer = Linear(in_size=(2, 2), out_size=(3, 1), key=jr.key(0))
    assert layer.in_size == (2, 2)
    assert layer.out_size == (3, 1)

    x = jnp.ones((2, 2))
    y = layer(x)
    assert y.shape == (3, 1)

    xb = jnp.ones((5, 2, 2))
    yb = layer(xb)
    assert yb.shape == (5, 3, 1)
    layer = Linear(in_size=(2, 2), out_size="scalar", key=jr.key(1))
    x = jnp.ones((2, 2))
    y = layer(x)
    assert y.shape == ()

    xb = jnp.ones((7, 2, 2))
    yb = layer(xb)
    assert yb.shape == (7,)
    for scan in (False, True):
        model = MLP(
            in_size=2, out_size=(2, 2), hidden_sizes=(8,), scan=scan, key=jr.key(2)
        )
        x = jnp.ones((2,))
        y = model(x)
        assert y.shape == (2, 2)

        xb = jnp.ones((4, 2))
        yb = model(xb)
        assert yb.shape == (4, 2, 2)
    emb = RandomFourierFeatureEmbeddings(in_size=(2, 2), out_size=8, key=jr.key(3))
    x = jnp.ones((2, 2))
    y = emb(x)
    assert y.shape == (8,)
