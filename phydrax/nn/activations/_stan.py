#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule


class Stan(StrictModule):
    r"""Self-scalable tanh (Stan) activation.

    Applies

    $$
    \text{Stan}_\beta(x)=\tanh(x)\,(1+\beta x),
    $$

    with trainable $\beta$ (scalar or broadcastable array).
    """

    beta: Array

    def __init__(
        self,
        shape: int | Sequence[int] | None = None,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""**Arguments:**

        - `shape`: Shape of $\beta$ (use `None` for a scalar).
        - `key`: PRNG key (unused; included for API compatibility).
        """
        shape = shape or ()
        self.beta = jnp.ones(shape=shape, dtype=float)

    def __call__(self, x: Array, /) -> Array:
        r"""Apply $\text{Stan}_\beta$ to `x`.

        Computes $\tanh(x)\,(1+\beta x)$ with broadcasting over the shape of
        `beta`.
        """
        return jnp.tanh(x) * (1.0 + self.beta * x)
