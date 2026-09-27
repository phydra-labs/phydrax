#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.layers import Dropout, inference_mode
from phydrax.nn.models import EquinoxModel, EquinoxStructuredModel, FunctionalJAXAdapter


def test_equinox_wrappers_scenario_1() -> None:
    module = eqx.nn.MLP(in_size=4, out_size=6, width_size=8, depth=2, key=jr.key(0))
    model = EquinoxModel(module, in_size=(2, 2), out_size=(3, 2))

    x = jnp.ones((2, 2))
    y = model(x)
    assert y.shape == (3, 2)
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
    module = eqx.nn.Dropout(p=0.5, inference=False)
    model = EquinoxModel(module, in_size=4, out_size=4, layout="passthrough")

    x = jnp.ones((4,))
    y = model(x, key=jr.key(2), inference=False)
    assert y.shape == x.shape
    module = eqx.nn.MLP(
        in_size=2, out_size="scalar", width_size=8, depth=2, key=jr.key(3)
    )
    model = EquinoxStructuredModel(module, in_size=2, out_size="scalar", layout="value")

    y = model((jnp.array(1.0), jnp.array(2.0)))
    assert y.shape == ()
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
    module, _ = eqx.nn.make_with_state(eqx.nn.BatchNorm)(3, axis_name="batch")

    with pytest.raises(TypeError, match="stateful Equinox modules.*FunctionalJAXAdapter"):
        EquinoxModel(module, in_size=3, out_size=3)
    with pytest.raises(TypeError, match="stateful Equinox modules.*FunctionalJAXAdapter"):
        EquinoxStructuredModel(module, in_size=3, out_size=3)
    adapter = _running_mean_adapter(inference=False)
    x = jnp.asarray([1.0, 2.0, 3.0])

    output, advanced = adapter.transition(x)
    assert jnp.allclose(output, 2.0 * (x - 2.0))
    assert float(advanced.model_state["mean"]) == 1.0
    assert float(adapter.model_state["mean"]) == 0.0
    assert jnp.allclose(adapter(x), output)
    assert float(adapter.model_state["mean"]) == 0.0

    evaluated = inference_mode(advanced)
    assert evaluated.inference and not advanced.inference
    assert jnp.allclose(evaluated(x), 2.0 * (x - 1.0))
    assert float(evaluated.transition(x)[1].model_state["mean"]) == 1.0

    parameters, model_state, _ = phx.partition_parameters(adapter)
    assert [float(leaf) for leaf in jax.tree_util.tree_leaves(parameters)] == [2.0]
    assert [float(leaf) for leaf in jax.tree_util.tree_leaves(model_state)] == [0.0]
    execution = adapter.model_execution_contract().execution
    assert (execution.tier, execution.stateful, execution.jit) == (
        "functional-jax",
        True,
        True,
    )
    assert jnp.allclose(eqx.filter_jit(lambda model, v: model(v))(adapter, x), output)
    adapter = FunctionalJAXAdapter(
        lambda parameters, model_state, x, key, *, inference: (
            x,
            {"mean": jnp.zeros(2)},
        ),
        {"scale": jnp.asarray(2.0)},
        {"mean": jnp.asarray(0.0)},
        in_size=3,
        out_size=3,
        inference=False,
    )
    with pytest.raises(ValueError, match="next_model_state"):
        adapter(jnp.ones(3))
    with pytest.raises(TypeError, match="inference must be bool"):
        _running_mean_adapter(inference=1)
    tree = {
        "phydrax": Dropout(4, p=0.5, inference=False),
        "equinox": eqx.nn.Dropout(p=0.5, inference=False),
        "label": "fixed",
    }

    converted = inference_mode(tree)
    restored = inference_mode(converted, False)
    values = jnp.ones((4,))

    assert converted is not tree
    # ty: ignore[unresolved-attribute]
    assert converted["phydrax"].inference
    # ty: ignore[unresolved-attribute]
    assert converted["equinox"].inference
    assert not tree["phydrax"].inference
    assert not tree["equinox"].inference
    assert converted["label"] == "fixed"
    # ty: ignore[call-non-callable]
    assert jnp.array_equal(converted["phydrax"](values), values)
    # ty: ignore[call-non-callable]
    assert jnp.array_equal(converted["equinox"](values), values)
    # ty: ignore[unresolved-attribute]
    assert not restored["phydrax"].inference
    # ty: ignore[unresolved-attribute]
    assert not restored["equinox"].inference


def test_equinox_structured_model_passthrough_tuple_input() -> None:
    class TupleModule(eqx.Module):
        def __call__(self, x: Any, *, key: Any = None) -> Any:
            del key
            a, b = x
            return a + 2.0 * b

    model = EquinoxStructuredModel(
        TupleModule(), in_size="scalar", out_size="scalar", layout="passthrough"
    )
    y = model((jnp.array(1.0), jnp.array(2.0)))
    assert y.shape == ()
    assert jnp.isclose(y, 5.0)


def _running_mean(
    parameters: Any, model_state: Any, x: Any, key: Any, *, inference: Any
) -> Any:
    del key
    if inference:
        return parameters["scale"] * (x - model_state["mean"]), model_state
    mean = jnp.mean(x)
    next_state = {"mean": 0.5 * model_state["mean"] + 0.5 * mean}
    return parameters["scale"] * (x - mean), next_state


def _running_mean_adapter(**options: Any) -> Any:
    return FunctionalJAXAdapter(
        _running_mean,
        {"scale": jnp.asarray(2.0)},
        {"mean": jnp.asarray(0.0)},
        in_size=3,
        out_size=3,
        **options,
    )
