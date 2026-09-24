# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._differentiation import AbstractConstructionCertificate
from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._model import INPUT_CONVEX_CERTIFICATE_KEY
from .._base import _AbstractBaseModel, _AbstractStructuredInputModel
from .._keys import EvalKey, fold_in_eval_key
from .._utils import _canonical_size, SizeLike
from ..layers._linear import Linear
from ..parameters import PositiveTransform


ConvexActivation = Literal["softplus", "relu", "squared_relu"]
InputConvexConstruction = Literal[
    "input-convex-network", "partially-input-convex-network"
]
_CanonicalSize: TypeAlias = int | tuple[int, ...] | Literal["scalar"]


def _convex_activation(name: ConvexActivation, values: Array, /) -> Array:
    if name == "softplus":
        return jax.nn.softplus(values)
    if name == "relu":
        return jax.nn.relu(values)
    if name == "squared_relu":
        positive = jax.nn.relu(values)
        return positive * positive
    raise ValueError("activation must be 'softplus', 'relu', or 'squared_relu'.")


def _size_payload(size: _CanonicalSize | None, /) -> Any:
    return list(size) if isinstance(size, tuple) else size


class InputConvexCertificate(AbstractConstructionCertificate):
    """Construction evidence that a scalar potential is convex in its input.

    Convexity is structural and holds for every parameter value: the convex
    input enters only through affine maps, every hidden-to-hidden and final
    hidden coupling is a positive-transformed linear map, and the activation is
    convex and nondecreasing. `context_size` is `None` for a potential that is
    jointly convex in its whole input; otherwise the potential is convex in its
    second argument for every fixed context of that size.
    """

    capability_id: ClassVar[str] = "input-convex"
    construction: InputConvexConstruction = eqx.field(static=True)
    convex_input_size: _CanonicalSize = eqx.field(static=True)
    context_size: _CanonicalSize | None = eqx.field(static=True)
    activation: ConvexActivation = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        construction: InputConvexConstruction,
        convex_input_size: SizeLike,
        context_size: SizeLike | None,
        activation: ConvexActivation,
        depth: int,
        width_size: int,
    ):
        if construction not in (
            "input-convex-network",
            "partially-input-convex-network",
        ):
            raise ValueError("Unknown input-convex construction.")
        if (context_size is None) != (construction == "input-convex-network"):
            raise ValueError(
                "Only partially input-convex constructions declare a context size."
            )
        if activation not in ("softplus", "relu", "squared_relu"):
            raise ValueError("activation must be 'softplus', 'relu', or 'squared_relu'.")
        depth_ = int(depth)
        width = int(width_size)
        if depth_ <= 0 or width <= 0:
            raise ValueError("Input-convex depth and width_size must be positive.")
        convex_size = _canonical_size(convex_input_size)
        context = None if context_size is None else _canonical_size(context_size)
        self.construction = construction
        self.convex_input_size = convex_size
        self.context_size = context
        self.activation = activation
        self.depth = depth_
        self.width_size = width
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "input-convex-certificate",
                "construction": construction,
                "convex_input_size": _size_payload(convex_size),
                "context_size": _size_payload(context),
                "activation": activation,
                "depth": depth_,
                "width_size": width,
                "hidden_coupling": "positive-transform",
            }
        )


def _require_positive_couplings(layers: tuple[Linear, ...], /) -> None:
    if any(not isinstance(layer.weight_transform, PositiveTransform) for layer in layers):
        raise ValueError(
            "Input-convex certificates require positive hidden-state couplings."
        )


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

    def input_convex_certificate(self) -> InputConvexCertificate:
        """Return the construction certificate of joint input convexity."""
        _require_positive_couplings(self.state_layers)
        return InputConvexCertificate(
            construction="input-convex-network",
            convex_input_size=self.in_size,
            context_size=None,
            activation=self.activation,
            depth=len(self.state_layers),
            width_size=self.width_size,
        )

    def model_metadata(self) -> Mapping[str, Any]:
        """Attach the input-convex certificate to a bound domain function."""
        return {INPUT_CONVEX_CERTIFICATE_KEY: self.input_convex_certificate()}


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

    def input_convex_certificate(self) -> InputConvexCertificate:
        """Return the construction certificate of convexity in the second input."""
        _require_positive_couplings(self.state_layers)
        context_size, convex_size = self.in_size
        return InputConvexCertificate(
            construction="partially-input-convex-network",
            convex_input_size=convex_size,
            context_size=context_size,
            activation=self.activation,
            depth=len(self.state_layers),
            width_size=self.width_size,
        )

    def model_metadata(self) -> Mapping[str, Any]:
        """Attach the input-convex certificate to a bound domain function."""
        return {INPUT_CONVEX_CERTIFICATE_KEY: self.input_convex_certificate()}


__all__ = [
    "InputConvexCertificate",
    "InputConvexNetwork",
    "PartiallyInputConvexNetwork",
]
