#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _get_size
from ..core._base import _AbstractBaseModel, _AbstractStructuredInputModel


class ConcatenatedModel(_AbstractStructuredInputModel):
    r"""Concatenate outputs from multiple models.

    Given models $\{m_k\}_{k=1}^K$ with shared input space, this wrapper returns

    $$
    y(x)=\operatorname{concat}\big(m_1(x),\dots,m_K(x)\big),
    $$

    where concatenation is performed along `axis` (by default the feature axis).
    Scalar outputs (`out_size="scalar"`) are treated as length-1 feature vectors
    for concatenation.
    """

    models: tuple[_AbstractBaseModel, ...]
    axis: int
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(self, models: Sequence[_AbstractBaseModel], *, axis: int = -1):
        if not models:
            raise ValueError("ConcatenatedModel requires at least one model.")
        first_in_size = models[0].in_size
        for model in models[1:]:
            if model.in_size != first_in_size:
                raise ValueError("All models must share the same in_size.")

        self.models = tuple(models)
        self.axis = int(axis)
        self.in_size = first_in_size
        out_dim = sum(_get_size(model.out_size) for model in self.models)
        if out_dim == 1 and all(model.out_size == "scalar" for model in self.models):
            self.out_size = "scalar"
        else:
            self.out_size = out_dim

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Evaluate all child models at `x` and concatenate their outputs.

        If more than one model is present, the input `key` is split so each child
        receives its own subkey.
        """
        if isinstance(x, tuple):
            structured_models: list[_AbstractStructuredInputModel] = []
            for model in self.models:
                if not isinstance(model, _AbstractStructuredInputModel):
                    raise TypeError(
                        "ConcatenatedModel received tuple input, but at least one "
                        "child model does not support structured inputs."
                    )
                structured_models.append(model)
            if len(structured_models) == 1:
                outputs = (structured_models[0](x, key=key),)
            else:
                keys = jr.split(key, len(structured_models))
                outputs = tuple(
                    model(x, key=subkey) for model, subkey in zip(structured_models, keys)
                )
        else:
            x_arr = jnp.asarray(x)
            if len(self.models) == 1:
                outputs = (self.models[0](x_arr, key=key),)
            else:
                keys = jr.split(key, len(self.models))
                outputs = tuple(
                    model(x_arr, key=subkey) for model, subkey in zip(self.models, keys)
                )
        outputs = tuple(
            jnp.expand_dims(o, axis=-1) if self.models[i].out_size == "scalar" else o
            for i, o in enumerate(outputs)
        )
        return jnp.concatenate(outputs, axis=self.axis)
