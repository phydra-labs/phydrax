# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Literal, TypeAlias

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from ..typing import PRNGKey


ProbeRefresh: TypeAlias = Literal["reuse", "redraw"]


def random_probes(key: PRNGKey, dimension: int, count: int, dtype: jnp.dtype, /) -> Array:
    """Unnormalized Gaussian columns, preserving the native Nyström stream."""
    real_dtype = jnp.empty((), dtype=dtype).real.dtype
    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_key, imag_key = jr.split(key)
        real = jr.normal(real_key, (dimension, count), dtype=real_dtype).astype(dtype)
        imaginary = jr.normal(imag_key, (dimension, count), dtype=real_dtype).astype(
            dtype
        )
        unit = jnp.asarray(1j, dtype=dtype)
        normalizer = jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype)).astype(dtype)
        return (real + unit * imaginary) / normalizer
    return jr.normal(key, (dimension, count), dtype=dtype)


__all__ = ["ProbeRefresh", "random_probes"]
