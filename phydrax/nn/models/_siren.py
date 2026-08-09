# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from .._base import _AbstractBaseModel
from .._keys import EvalKey, fold_in_eval_key
from .._utils import _canonical_size, _identity, SizeLike
from ..layers._linear import Linear
from ..layers._sine import SineLayer


class SIREN(_AbstractBaseModel):
    r"""Sinusoidal representation network with paper-faithful initialization."""

    layers: tuple[SineLayer, ...]
    projection: Linear
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    final_activation: Callable
    first_omega: float
    hidden_omega: float

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        width_size: int = 128,
        depth: int = 3,
        first_omega: float = 30.0,
        hidden_omega: float = 1.0,
        final_activation: Callable | None = None,
        use_bias: bool = True,
        use_final_bias: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        in_size_c = _canonical_size(in_size)
        out_size_c = _canonical_size(out_size)
        width = int(width_size)
        hidden_depth = int(depth)
        first_frequency = float(first_omega)
        hidden_frequency = float(hidden_omega)
        if width <= 0 or hidden_depth <= 0:
            raise ValueError("width_size and depth must be positive.")
        if (
            not math.isfinite(first_frequency)
            or first_frequency <= 0.0
            or not math.isfinite(hidden_frequency)
            or hidden_frequency <= 0.0
        ):
            raise ValueError("first_omega and hidden_omega must be finite and positive.")

        keys = jr.split(key, hidden_depth + 1)
        hidden: list[SineLayer] = [
            SineLayer(
                in_size=in_size_c,
                out_size=width,
                omega=first_frequency,
                is_first=True,
                use_bias=use_bias,
                key=keys[0],
            )
        ]
        hidden.extend(
            SineLayer(
                in_size=width,
                out_size=width,
                omega=hidden_frequency,
                is_first=False,
                use_bias=use_bias,
                key=layer_key,
            )
            for layer_key in keys[1:-1]
        )
        output_bound = math.sqrt(6.0 / width) / hidden_frequency
        projection = Linear(
            in_size=width,
            out_size=out_size_c,
            activation=None,
            rwf=False,
            use_bias=use_final_bias,
            bias_init_lim=output_bound,
            key=keys[-1],
        )
        output_key, _, _ = jr.split(keys[-1], 3)
        self.projection = eqx.tree_at(
            lambda layer: layer.weight,
            projection,
            jr.uniform(
                output_key,
                projection.weight.shape,
                minval=-output_bound,
                maxval=output_bound,
                dtype=projection.weight.dtype,
            ),
        )
        self.layers = tuple(hidden)
        self.in_size = in_size_c
        self.out_size = out_size_c
        self.final_activation = (
            _identity if final_activation is None else final_activation
        )
        self.first_omega = first_frequency
        self.hidden_omega = hidden_frequency

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, key=fold_in_eval_key(key, index))
        output = self.projection(hidden, key=fold_in_eval_key(key, len(self.layers)))
        return self.final_activation(output)


__all__ = ["SIREN"]
