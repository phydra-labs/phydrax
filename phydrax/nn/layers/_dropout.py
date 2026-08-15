#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from typing import cast, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from .._base import _AbstractBaseModel
from .._keys import EvalKey, require_eval_key
from .._utils import _canonical_size, _get_value_shape, SizeLike


def _dropout_probabilities(
    dropout: float | Sequence[float],
    count: int,
) -> tuple[float, ...]:
    if isinstance(dropout, Sequence) and not isinstance(dropout, (str, bytes)):
        probabilities = tuple(float(p) for p in cast(Sequence[float], dropout))
        if len(probabilities) != count:
            raise ValueError(
                f"Expected {count} dropout probabilities, got {len(probabilities)}."
            )
    else:
        probability = float(dropout)
        if count == 0 and probability != 0.0:
            raise ValueError("Dropout requires at least one hidden layer.")
        probabilities = (probability,) * count
    return probabilities


class Dropout(_AbstractBaseModel):
    """Inverted dropout with explicit elementwise or feature-mask semantics."""

    p: float
    mode: Literal["elementwise", "feature"]
    inference: bool
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    _value_shape: tuple[int, ...]

    def __init__(
        self,
        size: SizeLike,
        /,
        *,
        p: float,
        mode: Literal["elementwise", "feature"] = "feature",
        inference: bool = False,
    ):
        probability = float(p)
        if not 0.0 <= probability < 1.0:
            raise ValueError("Dropout p must satisfy 0 <= p < 1.")
        if mode not in ("elementwise", "feature"):
            raise ValueError("Dropout mode must be 'elementwise' or 'feature'.")
        size_c = _canonical_size(size)
        self.p = probability
        self.mode = mode
        self.inference = bool(inference)
        self.in_size = size_c
        self.out_size = size_c
        self._value_shape = _get_value_shape(size_c)

    def __call__(self, x: Array, /, *, key: EvalKey = None) -> Array:
        value = jnp.asarray(x)
        if self.inference or self.p == 0.0:
            return value
        rng = require_eval_key(key, owner="active Dropout")
        if self.mode == "elementwise":
            mask_shape = value.shape
        else:
            value_rank = len(self._value_shape)
            if value_rank and (
                value.ndim < value_rank or value.shape[-value_rank:] != self._value_shape
            ):
                raise ValueError(
                    "Feature dropout expected trailing value shape "
                    f"{self._value_shape}, got {value.shape}."
                )
            if value_rank:
                mask_shape = (1,) * (value.ndim - value_rank) + self._value_shape
            else:
                mask_shape = (1,) * value.ndim
        keep = jr.bernoulli(rng, p=1.0 - self.p, shape=mask_shape)
        return jnp.where(keep, value / (1.0 - self.p), jnp.zeros((), value.dtype))


def inference_mode(tree, value: bool = True):
    """Switch every inference-aware Equinox or Phydrax leaf in a PyTree."""

    return eqx.nn.inference_mode(tree, value)


__all__ = ["Dropout", "inference_mode"]
