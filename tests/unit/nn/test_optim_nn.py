#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import cast

import jax
import jax.numpy as jnp
import optax

from phydrax.nn import optim as nn_optim


def _value_fn(params):
    total = jnp.array(0.0)
    for leaf in jax.tree_util.tree_leaves(params):
        total = total + jnp.sum(jnp.asarray(leaf) ** 2)
    return total


def test_nn_optim_exports_and_updates():
    params = {"w": jnp.array([1.0, -2.0])}
    value = _value_fn(params)
    grads = jax.grad(_value_fn)(params)

    opts = (
        nn_optim.lbfgs_sw(learning_rate=1.0),
        nn_optim.bfgs_sw(learning_rate=1.0),
        nn_optim.ssbroyden(learning_rate=1.0),
    )

    for opt in opts:
        assert isinstance(opt, optax.GradientTransformationExtraArgs)
        state = opt.init(params)
        updates, state = opt.update(
            grads,
            state,
            params,
            value=value,
            grad=grads,
            value_fn=_value_fn,
        )
        new_params = cast(dict[str, jax.Array], optax.apply_updates(params, updates))
        assert new_params["w"].shape == params["w"].shape
