#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.models import MLP, RandomFourierFeatureEmbeddings, Sequential


def test_sequential_requires_at_least_one_model():
    with pytest.raises(ValueError, match="at least one model"):
        _ = Sequential(())


def test_sequential_rff_then_mlp_shape_and_metadata():
    model = Sequential(
        (
            RandomFourierFeatureEmbeddings(
                in_size="scalar",
                out_size=16,
                key=jr.key(0),
            ),
            MLP(
                in_size=16,
                out_size=3,
                width_size=8,
                depth=1,
                key=jr.key(1),
            ),
        )
    )
    y = model(jnp.asarray(0.25), key=jr.key(2))
    assert model.in_size == "scalar"
    assert model.out_size == 3
    assert y.shape == (3,)


def test_sequential_single_stage_matches_wrapped_model():
    mlp = MLP(
        in_size=2,
        out_size="scalar",
        width_size=16,
        depth=2,
        key=jr.key(0),
    )
    model = Sequential((mlp,))
    x = jnp.asarray([0.1, -0.3], dtype=float)
    y_ref = mlp(x, key=jr.key(3))
    y_seq = model(x, key=jr.key(3))
    assert y_seq.shape == ()
    assert jnp.allclose(y_seq, y_ref)


def test_sequential_rejects_adjacent_size_mismatch():
    m1 = MLP(in_size=2, out_size=3, width_size=8, depth=1, key=jr.key(0))
    m2 = MLP(in_size=4, out_size=1, width_size=8, depth=1, key=jr.key(1))
    with pytest.raises(ValueError, match="Sequential size mismatch"):
        _ = Sequential((m1, m2))


def test_sequential_tuple_input_requires_structured_first_stage():
    m1 = MLP(in_size=2, out_size=2, width_size=8, depth=1, key=jr.key(0))
    m2 = MLP(in_size=2, out_size=1, width_size=8, depth=1, key=jr.key(1))
    model = Sequential((m1, m2))
    with pytest.raises(TypeError, match="tuple input"):
        _ = model((jnp.asarray(0.1), jnp.asarray(0.2)), key=jr.key(2))
