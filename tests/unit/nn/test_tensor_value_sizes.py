#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.models import Linear, MLP, RandomFourierFeatureEmbeddings


def test_linear_tensor_in_tensor_out_shapes():
    layer = Linear(in_size=(2, 2), out_size=(3, 1), key=jr.key(0))
    assert layer.in_size == (2, 2)
    assert layer.out_size == (3, 1)

    x = jnp.ones((2, 2))
    y = layer(x)
    assert y.shape == (3, 1)

    xb = jnp.ones((5, 2, 2))
    yb = layer(xb)
    assert yb.shape == (5, 3, 1)


def test_linear_tensor_in_scalar_out_shapes():
    layer = Linear(in_size=(2, 2), out_size="scalar", key=jr.key(1))
    x = jnp.ones((2, 2))
    y = layer(x)
    assert y.shape == ()

    xb = jnp.ones((7, 2, 2))
    yb = layer(xb)
    assert yb.shape == (7,)


def test_mlp_tensor_out_shapes():
    model = MLP(in_size=2, out_size=(2, 2), hidden_sizes=(8,), key=jr.key(2))
    x = jnp.ones((2,))
    y = model(x)
    assert y.shape == (2, 2)

    xb = jnp.ones((4, 2))
    yb = model(xb)
    assert yb.shape == (4, 2, 2)


def test_random_fourier_tensor_in_size():
    emb = RandomFourierFeatureEmbeddings(in_size=(2, 2), out_size=8, key=jr.key(3))
    x = jnp.ones((2, 2))
    y = emb(x)
    assert y.shape == (8,)
