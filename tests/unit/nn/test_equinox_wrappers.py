#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.layers import Dropout, inference_mode
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


def test_equinox_wrappers_declare_wrapped_module_arrays_as_parameters():
    module = eqx.nn.MLP(in_size=2, out_size=3, width_size=4, depth=1, key=jr.key(4))
    arrays = jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_inexact_array))

    with pytest.raises(ValueError, match="EquinoxModel") as raised:
        phx.require_parameter_roles({"model": module}, context="raw module")
    assert "['model'].layers[0].weight" in str(raised.value)

    for model in (
        EquinoxModel(module, in_size=2, out_size=3),
        EquinoxStructuredModel(module, in_size=2, out_size=3, layout="value"),
    ):
        parameters, model_state, fixed = phx.partition_parameters({"model": model})
        assert jax.tree_util.tree_leaves(model_state) == []
        assert not any(
            eqx.is_inexact_array(leaf) for leaf in jax.tree_util.tree_leaves(fixed)
        )
        leaves = jax.tree_util.tree_leaves(parameters)
        assert len(leaves) == len(arrays)
        assert all(left is right for left, right in zip(leaves, arrays, strict=True))


def test_equinox_wrappers_reject_stateful_modules():
    module, _ = eqx.nn.make_with_state(eqx.nn.BatchNorm)(3, axis_name="batch")

    with pytest.raises(TypeError, match="stateful Equinox modules"):
        EquinoxModel(module, in_size=3, out_size=3)
    with pytest.raises(TypeError, match="stateful Equinox modules"):
        EquinoxStructuredModel(module, in_size=3, out_size=3)


def test_inference_mode_switches_mixed_phydrax_and_equinox_tree():
    tree = {
        "phydrax": Dropout(4, p=0.5, inference=False),
        "equinox": eqx.nn.Dropout(p=0.5, inference=False),
        "label": "fixed",
    }

    converted = inference_mode(tree)
    restored = inference_mode(converted, False)
    values = jnp.ones((4,))

    assert converted is not tree
    assert converted["phydrax"].inference
    assert converted["equinox"].inference
    assert not tree["phydrax"].inference
    assert not tree["equinox"].inference
    assert converted["label"] == "fixed"
    assert jnp.array_equal(converted["phydrax"](values), values)
    assert jnp.array_equal(converted["equinox"](values), values)
    assert not restored["phydrax"].inference
    assert not restored["equinox"].inference
