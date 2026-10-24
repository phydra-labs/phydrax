"""The measure-aware attention family accepts an optional `DTypeLike` dtype."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import assert_type

from phydrax.nn.layers import MeasureAwareAttention
from phydrax.nn.operator.architectures import (
    LatentTokenProcessor,
    SpectralConvND,
    Transolver,
    UPT,
)
from phydrax.nn.operator.layers import OperatorAttention


key = jax.random.key(0)

attention = MeasureAwareAttention(
    source_channels=2,
    query_channels=2,
    out_channels=2,
    num_heads=1,
    head_dim=2,
    dtype=jnp.float32,
    key=key,
)
assert_type(attention, MeasureAwareAttention)
assert_type(
    attention(
        jnp.ones((1, 3, 2), dtype=jnp.float32),
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        jnp.ones((1, 3), dtype=jnp.float32),
    ),
    jax.Array,
)
assert_type(
    OperatorAttention(source_channels=2, dtype="float32", key=key), OperatorAttention
)
assert_type(
    LatentTokenProcessor(4, 1, num_heads=2, dtype=np.dtype("float32"), key=key),
    LatentTokenProcessor,
)
assert_type(
    SpectralConvND(in_channels=2, out_channels=2, n_modes=2, dtype=np.float32, key=key),
    SpectralConvND,
)
assert_type(UPT(in_channels=1, out_channels=1, coord_dim=1, dtype=None), UPT)
assert_type(Transolver(coord_dim=1, dtype=jnp.float32), Transolver)

MeasureAwareAttention(
    source_channels=2,
    query_channels=2,
    out_channels=2,
    num_heads=1,
    head_dim=2,
    dtype=32,  # ty: ignore[invalid-argument-type]
)
UPT(
    in_channels=1,
    out_channels=1,
    coord_dim=1,
    dtype=1.5,  # ty: ignore[invalid-argument-type]
)
