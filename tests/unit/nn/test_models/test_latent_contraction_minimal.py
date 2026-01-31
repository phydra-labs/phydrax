#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.models import LatentContractionModel, MLP, Separable


def test_latent_contraction_regular_and_factorwise():
    key = jr.key(0)
    # space model: in_size=2 -> out=latent*out=8
    space_model = MLP(in_size=2, out_size=8, width_size=8, depth=1, key=key)
    # time model: scalar -> latent=4
    time_model = MLP(in_size="scalar", out_size=4, width_size=8, depth=1, key=jr.key(1))

    model = LatentContractionModel(
        out_size=2,
        latent_size=4,
        x=space_model,
        t=time_model,
    )

    # regular
    xs = jnp.stack([jnp.array([0.1, 0.2, 0.3]), jnp.array([0.2, 0.3, 0.4])], axis=0)
    y = jax.vmap(model)(xs)
    assert y.shape == (2, 2)

    # factor-wise
    x_space = (jnp.array([0.1, 0.2]), jnp.array([0.2, 0.3]))
    x_time = (jnp.array([0.3, 0.4]),)
    yts = model({"x": x_space, "t": x_time})
    assert yts.shape == (2, 2, 2, 2)

    # factor count mismatch
    with pytest.raises(ValueError):
        _ = model((jnp.array([0.1, 0.2]),))


def test_separable_wrapper_regular_and_separable():
    key = jr.key(42)
    # Two scalar models for 2D
    m1 = MLP(in_size="scalar", out_size=8, width_size=8, depth=1, key=key)
    m2 = MLP(in_size="scalar", out_size=8, width_size=8, depth=1, key=jr.key(43))

    model = Separable(in_size=2, out_size=2, latent_size=4, models=(m1, m2))

    xs = jnp.array([[0.1, 0.2], [0.2, 0.4]])
    y = jax.vmap(model)(xs)
    assert y.shape == (2, 2)

    # separable input
    x1 = jnp.array([0.1, 0.2])
    x2 = jnp.array([0.2, 0.4])
    ys = model((x1, x2))
    assert ys.shape == (2, 2, 2)
