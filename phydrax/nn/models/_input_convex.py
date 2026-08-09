# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from .._base import _AbstractBaseModel, _AbstractStructuredInputModel
from .._keys import EvalKey, fold_in_eval_key
from .._utils import _canonical_size, SizeLike
from ..layers._linear import Linear
from ..parameters import PositiveTransform


ConvexActivation = Literal["softplus", "relu", "squared_relu"]


def _convex_activation(name: ConvexActivation, values: Array, /) -> Array:
    if name == "softplus":
        return jax.nn.softplus(values)
    if name == "relu":
        return jax.nn.relu(values)
    if name == "squared_relu":
        positive = jax.nn.relu(values)
        return positive * positive
    raise ValueError("activation must be 'softplus', 'relu', or 'squared_relu'.")


def _positive_linear(
    *,
    in_size: SizeLike,
    out_size: SizeLike,
    key: Key[Array, ""],
) -> Linear:
    return Linear(
        in_size=in_size,
        out_size=out_size,
        activation=None,
        rwf=False,
        use_bias=False,
        weight_transform=PositiveTransform(),
        key=key,
    )


class InputConvexNetwork(_AbstractBaseModel):
    r"""Scalar input-convex potential with positive hidden-state couplings."""

    input_layers: tuple[Linear, ...]
    state_layers: tuple[Linear, ...]
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: Literal["scalar"]
    width_size: int
    activation: ConvexActivation

    def __init__(
        self,
        *,
        in_size: SizeLike,
        width_size: int = 64,
        depth: int = 3,
        activation: ConvexActivation = "softplus",
        use_bias: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        in_size_c = _canonical_size(in_size)
        width = int(width_size)
        hidden_depth = int(depth)
        if width <= 0 or hidden_depth <= 0:
            raise ValueError("width_size and depth must be positive.")
        _convex_activation(activation, jnp.asarray(0.0))
        keys = jr.split(key, 2 * hidden_depth + 1)
        self.input_layers = tuple(
            Linear(
                in_size=in_size_c,
                out_size=width if index < hidden_depth else "scalar",
                activation=None,
                rwf=False,
                use_bias=use_bias,
                key=keys[index],
            )
            for index in range(hidden_depth + 1)
        )
        self.state_layers = tuple(
            _positive_linear(
                in_size=width,
                out_size=width if index < hidden_depth - 1 else "scalar",
                key=keys[hidden_depth + 1 + index],
            )
            for index in range(hidden_depth)
        )
        self.in_size = in_size_c
        self.out_size = "scalar"
        self.width_size = width
        self.activation = activation

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        hidden = _convex_activation(
            self.activation,
            self.input_layers[0](x, key=fold_in_eval_key(key, 0)),
        )
        for index in range(1, len(self.input_layers) - 1):
            hidden = _convex_activation(
                self.activation,
                self.input_layers[index](x, key=fold_in_eval_key(key, 2 * index))
                + self.state_layers[index - 1](
                    hidden, key=fold_in_eval_key(key, 2 * index + 1)
                ),
            )
        return self.input_layers[-1](
            x, key=fold_in_eval_key(key, 2 * len(self.input_layers))
        ) + self.state_layers[-1](
            hidden, key=fold_in_eval_key(key, 2 * len(self.input_layers) + 1)
        )

    def gradient(self, x: Array, /) -> Array:
        """Return the monotone map induced by this convex potential."""
        return jax.grad(lambda value: self(value))(x)

    def hessian(self, x: Array, /) -> Array:
        """Return the input Hessian of this scalar potential."""
        return jax.hessian(lambda value: self(value))(x)


class PartiallyInputConvexNetwork(_AbstractStructuredInputModel):
    r"""Potential convex in its second input for every fixed context.

    The model accepts ``(context, convex_input)``. Context enters through an
    unconstrained nonlinear feature map and additive affine terms; all recurrent
    hidden-state weights remain positive. Therefore context dependence is
    unrestricted while convexity in the designated second input is structural.
    """

    context_lift: Linear
    convex_input_layers: tuple[Linear, ...]
    context_layers: tuple[Linear, ...]
    state_layers: tuple[Linear, ...]
    in_size: tuple[int | tuple[int, ...] | Literal["scalar"], ...]
    out_size: Literal["scalar"]
    width_size: int
    activation: ConvexActivation

    def __init__(
        self,
        *,
        context_size: SizeLike,
        convex_size: SizeLike,
        width_size: int = 64,
        depth: int = 3,
        activation: ConvexActivation = "softplus",
        use_bias: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        context_size_c = _canonical_size(context_size)
        convex_size_c = _canonical_size(convex_size)
        width = int(width_size)
        hidden_depth = int(depth)
        if width <= 0 or hidden_depth <= 0:
            raise ValueError("width_size and depth must be positive.")
        _convex_activation(activation, jnp.asarray(0.0))
        keys = jr.split(key, 3 * hidden_depth + 3)
        self.context_lift = Linear(
            in_size=context_size_c,
            out_size=width,
            activation=jax.nn.tanh,
            rwf=False,
            use_bias=use_bias,
            key=keys[0],
        )
        self.convex_input_layers = tuple(
            Linear(
                in_size=convex_size_c,
                out_size=width if index < hidden_depth else "scalar",
                activation=None,
                rwf=False,
                use_bias=use_bias,
                key=keys[1 + index],
            )
            for index in range(hidden_depth + 1)
        )
        context_offset = hidden_depth + 2
        self.context_layers = tuple(
            Linear(
                in_size=width,
                out_size=width if index < hidden_depth else "scalar",
                activation=None,
                rwf=False,
                use_bias=False,
                key=keys[context_offset + index],
            )
            for index in range(hidden_depth + 1)
        )
        state_offset = context_offset + hidden_depth + 1
        self.state_layers = tuple(
            _positive_linear(
                in_size=width,
                out_size=width if index < hidden_depth - 1 else "scalar",
                key=keys[state_offset + index],
            )
            for index in range(hidden_depth)
        )
        self.in_size = (context_size_c, convex_size_c)
        self.out_size = "scalar"
        self.width_size = width
        self.activation = activation

    def __call__(
        self,
        x: tuple[Array, Array],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, tuple) or len(x) != 2:
            raise TypeError(
                "PartiallyInputConvexNetwork requires (context, convex_input)."
            )
        context, convex_input = x
        context_features = self.context_lift(context, key=fold_in_eval_key(key, 0))
        hidden = _convex_activation(
            self.activation,
            self.convex_input_layers[0](convex_input, key=fold_in_eval_key(key, 1))
            + self.context_layers[0](context_features, key=fold_in_eval_key(key, 2)),
        )
        for index in range(1, len(self.convex_input_layers) - 1):
            site = 3 * index
            hidden = _convex_activation(
                self.activation,
                self.convex_input_layers[index](
                    convex_input, key=fold_in_eval_key(key, site)
                )
                + self.context_layers[index](
                    context_features, key=fold_in_eval_key(key, site + 1)
                )
                + self.state_layers[index - 1](
                    hidden, key=fold_in_eval_key(key, site + 2)
                ),
            )
        return (
            self.convex_input_layers[-1](
                convex_input, key=fold_in_eval_key(key, 3 * len(self.state_layers))
            )
            + self.context_layers[-1](
                context_features,
                key=fold_in_eval_key(key, 3 * len(self.state_layers) + 1),
            )
            + self.state_layers[-1](
                hidden, key=fold_in_eval_key(key, 3 * len(self.state_layers) + 2)
            )
        )

    def convex_gradient(self, context: Array, convex_input: Array, /) -> Array:
        """Return the gradient with respect to the structurally convex input."""
        return jax.grad(lambda value: self((context, value)))(convex_input)

    def convex_hessian(self, context: Array, convex_input: Array, /) -> Array:
        """Return the Hessian with respect to the structurally convex input."""
        return jax.hessian(lambda value: self((context, value)))(convex_input)


__all__ = ["InputConvexNetwork", "PartiallyInputConvexNetwork"]
