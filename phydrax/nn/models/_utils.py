#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import string

import jax.numpy as jnp
from jax import Array


def _stack_separable(coords: tuple[Array, ...], /) -> Array:
    arrs = [jnp.asarray(c) for c in coords]
    grids = jnp.meshgrid(*arrs, indexing="ij")
    flat = [g.reshape(-1) for g in grids]
    return jnp.stack(flat, axis=1)


def _contract_str(n: int, /) -> str:
    letters = string.ascii_lowercase.replace("l", "").replace("o", "")
    if n < 2 or n > len(letters):
        raise ValueError(f"n must be in [2, {len(letters)}], got {n}.")
    left = ",".join(f"{letters[i]}lo" for i in range(n))
    right = "".join(letters[:n]) + "o"
    return f"{left}->{right}"
