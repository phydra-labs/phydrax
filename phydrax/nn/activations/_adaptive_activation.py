#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule


class AdaptiveActivation(StrictModule):
    r"""Adaptive activation wrapper.

    Wraps an activation $\sigma$ as

    $$
    x\mapsto\sigma(a x),
    $$

    where $a$ is a trainable scalar (layer-wise) or broadcastable vector (neuron-wise).
    """

    fn: Callable[[Array], Array]
    alpha: Array

    def __init__(
        self,
        fn: Callable[[Array], Array],
        /,
        *,
        shape: int | Sequence[int] | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""**Arguments:**

        - `fn`: Base activation function $\sigma$.
        - `shape`: Shape of the trainable coefficient $a$ (use `None` for a scalar).
        - `key`: PRNG key (unused; included for API compatibility).
        """
        shape = shape or ()
        self.fn = fn
        self.alpha = jnp.ones(shape, dtype=float)

    def __call__(self, x: Array, /) -> Array:
        r"""Apply the adaptive activation to `x`.

        Computes $\sigma(a x)$ where $\sigma$ is the wrapped `fn` and $a$ is the
        trainable coefficient.
        """
        return self.fn(self.alpha * x)
