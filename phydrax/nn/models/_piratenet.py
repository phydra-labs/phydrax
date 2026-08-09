# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .._base import _AbstractBaseModel
from .._keys import EvalKey, fold_in_eval_key
from .._utils import _canonical_size, _identity, SizeLike
from ..layers._adaptive_residual import AdaptiveResidual
from ..layers._linear import Linear


class _PirateBranch(StrictModule):
    layers: tuple[Linear, Linear, Linear]

    def __init__(
        self,
        *,
        width: int,
        activation: Callable,
        rwf: bool | tuple[float, float],
        use_bias: bool,
        initializer: str,
        key: Key[Array, ""],
    ):
        keys = jr.split(key, 3)
        self.layers = tuple(
            Linear(
                in_size=width,
                out_size=width,
                activation=activation,
                rwf=rwf,
                use_bias=use_bias,
                initializer=initializer,
                key=layer_key,
            )
            for layer_key in keys
        )

    @staticmethod
    def _gate(gate: Array, encoder_u: Array, encoder_v: Array, /) -> Array:
        return gate * encoder_u + (1.0 - gate) * encoder_v

    def __call__(
        self,
        hidden: Array,
        encoder_u: Array,
        encoder_v: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        first = self.layers[0](hidden, key=fold_in_eval_key(key, 0))
        first = self._gate(first, encoder_u, encoder_v)
        second = self.layers[1](first, key=fold_in_eval_key(key, 1))
        second = self._gate(second, encoder_u, encoder_v)
        return self.layers[2](second, key=fold_in_eval_key(key, 2))


class PirateNet(_AbstractBaseModel):
    r"""Physics-informed residual network with identity-start adaptive depth.

    An optional existing Phydrax embedding first maps the coordinates. Two
    persistent nonlinear encoders are then reused by every three-layer Pirate
    branch. Each branch is wrapped by :class:`AdaptiveResidual`; with the
    default zero gates the complete deep body is exactly the identity and the
    initialized model is a linear map of its embedding.
    """

    embedding: _AbstractBaseModel | None
    lift: Linear
    encoder_u: Linear
    encoder_v: Linear
    blocks: tuple[AdaptiveResidual, ...]
    projection: Linear
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    final_activation: Callable
    width_size: int

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        width_size: int = 128,
        depth: int = 4,
        embedding: _AbstractBaseModel | None = None,
        activation: Callable = jax.nn.tanh,
        final_activation: Callable | None = None,
        initial_alpha: float = 0.0,
        rwf: bool | tuple[float, float] = False,
        use_bias: bool = True,
        use_final_bias: bool = True,
        initializer: str = "glorot_normal",
        key: Key[Array, ""] = DOC_KEY0,
    ):
        in_size_c = _canonical_size(in_size)
        out_size_c = _canonical_size(out_size)
        width = int(width_size)
        block_count = int(depth)
        if width <= 0 or block_count <= 0:
            raise ValueError("width_size and depth must be positive.")
        if embedding is not None:
            if not isinstance(embedding, _AbstractBaseModel):
                raise TypeError("embedding must be a Phydrax neural model or None.")
            if embedding.in_size != in_size_c:
                raise ValueError("embedding input size must match PirateNet in_size.")
            embedded_size = embedding.out_size
        else:
            embedded_size = in_size_c
        alpha_value = float(initial_alpha)
        if not jnp.isfinite(alpha_value):
            raise ValueError("initial_alpha must be finite.")

        keys = jr.split(key, block_count + 4)
        self.embedding = embedding
        self.lift = Linear(
            in_size=embedded_size,
            out_size=width,
            activation=None,
            rwf=rwf,
            use_bias=use_bias,
            initializer=initializer,
            key=keys[0],
        )
        self.encoder_u = Linear(
            in_size=embedded_size,
            out_size=width,
            activation=activation,
            rwf=rwf,
            use_bias=use_bias,
            initializer=initializer,
            key=keys[1],
        )
        self.encoder_v = Linear(
            in_size=embedded_size,
            out_size=width,
            activation=activation,
            rwf=rwf,
            use_bias=use_bias,
            initializer=initializer,
            key=keys[2],
        )
        self.blocks = tuple(
            AdaptiveResidual(
                _PirateBranch(
                    width=width,
                    activation=activation,
                    rwf=rwf,
                    use_bias=use_bias,
                    initializer=initializer,
                    key=block_key,
                ),
                initial_alpha=alpha_value,
            )
            for block_key in keys[3:-1]
        )
        self.projection = Linear(
            in_size=width,
            out_size=out_size_c,
            activation=None,
            rwf=rwf,
            use_bias=use_final_bias,
            initializer=initializer,
            key=keys[-1],
        )
        self.in_size = in_size_c
        self.out_size = out_size_c
        self.final_activation = (
            _identity if final_activation is None else final_activation
        )
        self.width_size = width

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        features = (
            x
            if self.embedding is None
            else self.embedding(x, key=fold_in_eval_key(key, 0))
        )
        hidden = self.lift(features, key=fold_in_eval_key(key, 1))
        encoder_u = self.encoder_u(features, key=fold_in_eval_key(key, 2))
        encoder_v = self.encoder_v(features, key=fold_in_eval_key(key, 3))
        for index, block in enumerate(self.blocks, start=4):
            hidden = block(
                hidden,
                encoder_u,
                encoder_v,
                key=fold_in_eval_key(key, index),
            )
        output = self.projection(hidden, key=fold_in_eval_key(key, len(self.blocks) + 4))
        return self.final_activation(output)


__all__ = ["PirateNet"]
