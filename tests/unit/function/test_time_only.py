#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.domain import Interval1d, ProductStructure, TimeInterval


def test_time_only_function_broadcasts_over_space_and_time_axes():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    dom = geom @ time

    @dom.Function("t")
    def f(t):
        return 3.0 * t

    component = dom.component()
    structure = ProductStructure((("x",), ("t",)))
    batch = component.sample((7, 9), structure=structure, key=jr.key(0))

    out = f(batch)

    axis_x = batch.structure.axis_for("x")
    axis_t = batch.structure.axis_for("t")
    assert axis_x is not None and axis_t is not None
    assert set(out.named_dims) == {axis_x, axis_t}

    t_vals = jnp.asarray(batch["t"].data, dtype=float).reshape((-1,))
    n_x = int(batch["x"].data.shape[0])
    if out.dims == (axis_t, axis_x):
        expected = jnp.broadcast_to(3.0 * t_vals[:, None], (t_vals.shape[0], n_x))
    elif out.dims == (axis_x, axis_t):
        expected = jnp.broadcast_to(3.0 * t_vals[None, :], (n_x, t_vals.shape[0]))
    else:
        raise AssertionError(f"Unexpected dims for broadcasted output: {out.dims}")

    assert jnp.allclose(out.data, expected)
