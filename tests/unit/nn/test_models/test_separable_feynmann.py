#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.models import SeparableFeynmaNN


def test_separable_feynmann_vector_input_shape():
    model = SeparableFeynmaNN(
        in_size=2,
        out_size=3,
        latent_size=4,
        width_size=8,
        depth=2,
        num_paths=2,
        key=jr.key(0),
    )
    x = jr.normal(jr.key(1), (2,))
    y = model(x)
    assert y.shape == (3,)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))


def test_separable_feynmann_coord_separable_shape():
    model = SeparableFeynmaNN(
        in_size=2,
        out_size="scalar",
        latent_size=3,
        width_size=8,
        depth=2,
        num_paths=2,
        key=jr.key(2),
    )
    x0 = jnp.linspace(0.0, 1.0, 5)
    x1 = jnp.linspace(-1.0, 1.0, 6)
    y = model((x0, x1))
    assert y.shape == (5, 6)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))


def test_separable_feynmann_scalar_requires_split_input():
    with pytest.raises(ValueError, match="requires in_size >= 2"):
        _ = SeparableFeynmaNN(
            in_size="scalar",
            out_size="scalar",
            width_size=8,
            depth=1,
            num_paths=2,
        )


def test_separable_feynmann_scalar_with_split_input():
    model = SeparableFeynmaNN(
        in_size="scalar",
        out_size="scalar",
        latent_size=2,
        split_input=2,
        width_size=6,
        depth=1,
        num_paths=2,
        key=jr.key(3),
    )
    x = jnp.asarray(0.25)
    y = model(x)
    assert y.shape == ()
    assert jnp.isfinite(y)
