#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.domain import Interval1d, ProductStructure, TimeInterval
from phydrax.operators.differential import grad, hessian


def test_parameter_broadcasts_over_product_domain():
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    lam = dom.Parameter(2.5)
    assert lam.deps == ()

    component = dom.component()
    structure = ProductStructure((("x",), ("t",)))
    batch = component.sample((3, 4), structure=structure, key=jr.key(0))
    out = jnp.asarray(lam(batch).data)
    assert out.shape == (3, 4)
    assert jnp.allclose(out, 2.5)


def test_parameter_is_inexact_and_trainable_leaf():
    dom = Interval1d(0.0, 1.0)
    lam = dom.Parameter(1)

    params, _ = eqx.partition(lam.func, eqx.is_inexact_array)
    leaves = [x for x in jax.tree_util.tree_leaves(params) if eqx.is_inexact_array(x)]
    assert len(leaves) == 1
    assert jnp.issubdtype(leaves[0].dtype, jnp.inexact)


def test_parameter_transform_applies_and_trains_raw():
    dom = Interval1d(0.0, 1.0)
    lam = dom.Parameter(-1.0, transform=jax.nn.softplus)

    params, _ = eqx.partition(lam.func, eqx.is_inexact_array)
    leaves = [x for x in jax.tree_util.tree_leaves(params) if eqx.is_inexact_array(x)]
    assert len(leaves) == 1
    assert jnp.allclose(leaves[0], -1.0)

    component = dom.component()
    structure = ProductStructure((("x",),))
    batch = component.sample(5, structure=structure, key=jr.key(0))
    out = jnp.asarray(lam(batch).data)
    assert jnp.all(out > 0.0)


def test_parameter_grad_and_hessian_are_zero():
    dom = Interval1d(0.0, 1.0)
    lam = dom.Parameter(2.5)

    component = dom.component()
    structure = ProductStructure((("x",),))
    batch = component.sample(7, structure=structure, key=jr.key(0))

    g = jnp.asarray(grad(lam, var="x")(batch).data)
    H = jnp.asarray(hessian(lam, var="x")(batch).data)
    assert g.shape == (7, 1)
    assert H.shape == (7, 1, 1)
    assert jnp.allclose(g, 0.0)
    assert jnp.allclose(H, 0.0)
