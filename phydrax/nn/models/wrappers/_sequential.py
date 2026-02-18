#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _canonical_size
from ..core._base import _AbstractBaseModel, _AbstractStructuredInputModel


class Sequential(_AbstractStructuredInputModel):
    r"""Compose models in sequence.

    Given models $(m_1,\dots,m_K)$, this wrapper evaluates

    $$
    y(x)=m_K(\cdots m_2(m_1(x))\cdots).
    $$

    Adjacent models must have compatible sizes:
    `canonical(prev.out_size) == canonical(next.in_size)`.
    """

    models: tuple[_AbstractBaseModel, ...]
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(self, models: Sequence[_AbstractBaseModel]):
        if not models:
            raise ValueError("Sequential requires at least one model.")

        models_t = tuple(models)
        for idx in range(1, len(models_t)):
            prev = models_t[idx - 1]
            curr = models_t[idx]
            prev_out = _canonical_size(prev.out_size)
            curr_in = _canonical_size(curr.in_size)
            if prev_out != curr_in:
                raise ValueError(
                    "Sequential size mismatch at stage "
                    f"{idx - 1}->{idx}: prev.out_size={prev.out_size!r} "
                    f"is not compatible with curr.in_size={curr.in_size!r}."
                )

        self.models = models_t
        self.in_size = _canonical_size(models_t[0].in_size)
        self.out_size = _canonical_size(models_t[-1].out_size)

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Evaluate the model pipeline.

        If tuple input is provided, the first stage must support structured input.
        """
        if len(self.models) == 1:
            keys = (key,)
        else:
            keys = tuple(jr.split(key, len(self.models)))

        first_model = self.models[0]
        if isinstance(x, tuple):
            if not isinstance(first_model, _AbstractStructuredInputModel):
                raise TypeError(
                    "Sequential received tuple input, but the first stage does not "
                    "support structured inputs."
                )
            y = first_model(x, key=keys[0])
        else:
            y = first_model(jnp.asarray(x), key=keys[0])

        for model, subkey in zip(self.models[1:], keys[1:], strict=True):
            y = model(jnp.asarray(y), key=subkey)
        return jnp.asarray(y)


__all__ = ["Sequential"]
