#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.models import EquinoxModel, EquinoxStructuredModel


def test_equinox_model_value_layout_tensor_io():
    module = eqx.nn.MLP(in_size=4, out_size=6, width_size=8, depth=2, key=jr.key(0))
    model = EquinoxModel(module, in_size=(2, 2), out_size=(3, 2))

    x = jnp.ones((2, 2))
    y = model(x)
    assert y.shape == (3, 2)


def test_equinox_model_value_layout_scalar_io():
    module = eqx.nn.MLP(
        in_size="scalar",
        out_size="scalar",
        width_size=8,
        depth=2,
        key=jr.key(1),
    )
    model = EquinoxModel(module, in_size="scalar", out_size="scalar")

    x = jnp.array(1.25)
    y = model(x)
    assert y.shape == ()


def test_equinox_model_passthrough_forwards_key_and_kwargs():
    module = eqx.nn.Dropout(p=0.5, inference=False)
    model = EquinoxModel(module, in_size=4, out_size=4, layout="passthrough")

    x = jnp.ones((4,))
    y = model(x, key=jr.key(2), inference=False)
    assert y.shape == x.shape


def test_equinox_structured_model_passthrough_tuple_input():
    class TupleModule(eqx.Module):
        def __call__(self, x, *, key=None):
            del key
            a, b = x
            return a + 2.0 * b

    model = EquinoxStructuredModel(
        TupleModule(), in_size="scalar", out_size="scalar", layout="passthrough"
    )
    y = model((jnp.array(1.0), jnp.array(2.0)))
    assert y.shape == ()
    assert jnp.isclose(y, 5.0)


def test_equinox_structured_model_value_layout_concatenates_tuple():
    module = eqx.nn.MLP(
        in_size=2, out_size="scalar", width_size=8, depth=2, key=jr.key(3)
    )
    model = EquinoxStructuredModel(module, in_size=2, out_size="scalar", layout="value")

    y = model((jnp.array(1.0), jnp.array(2.0)))
    assert y.shape == ()
