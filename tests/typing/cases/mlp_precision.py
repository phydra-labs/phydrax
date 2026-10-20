"""`Linear`, `MLP` and `HyperRectangle` accept an optional `DTypeLike` dtype."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import assert_type

from phydrax.domain import HyperRectangle
from phydrax.nn.layers import Linear
from phydrax.nn.models import MLP


key = jax.random.key(0)

assert_type(Linear(in_size=2, out_size=3, dtype=jnp.float32, key=key), Linear)
assert_type(Linear(in_size=2, out_size=3, dtype="float32", key=key), Linear)
assert_type(Linear(in_size=2, out_size=3, dtype=None, key=key), Linear)
model = MLP(
    in_size=2,
    out_size="scalar",
    width_size=4,
    depth=1,
    dtype=np.dtype("float32"),
    key=key,
)
assert_type(model, MLP)
assert_type(model(jnp.ones((2,), dtype=jnp.float32)), jax.Array)
assert_type(MLP(in_size=2, out_size=1, hidden_sizes=(3,), dtype=np.float32), MLP)
assert_type(HyperRectangle(np.zeros(2), np.ones(2), dtype=jnp.float32), HyperRectangle)

Linear(in_size=2, out_size=3, dtype=32)  # ty: ignore[invalid-argument-type]
MLP(
    in_size=2,
    out_size=1,
    hidden_sizes=(4,),
    dtype=1.5,  # ty: ignore[invalid-argument-type]
)
