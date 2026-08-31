#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key
from opt_einsum import contract

from ..._doc import DOC_KEY0
from .._base import _AbstractBaseModel
from .._initializers import _initializer_dict
from .._keys import EvalKey
from .._utils import (
    _canonical_size,
    _get_size,
    _get_value_shape,
    _identity,
    SizeLike,
)
from ..parameters import AbstractParameterTransform, LowRankUpdate


_key = DOC_KEY0


class Linear(_AbstractBaseModel):
    r"""Affine layer with optional activation.

    Computes

    $$
    y=\phi(Wx+b),
    $$

    where $\phi$ is `activation` (or the identity). If *Random Weight
    Factorization* (RWF) is enabled, parameters are represented as an
    unscaled weight matrix $V$ and per-output log-scales $s$, and the layer
    applies

    $$
    y=\phi\!\left(\operatorname{diag}(e^s)\,Vx + b\right).
    $$

    A shape-preserving `weight_transform` may map raw trainable weights to
    physically constrained effective weights at evaluation time. A
    `LowRankUpdate` instead applies a frozen-base adaptation without
    materializing its dense update.
    """

    weight: Array | LowRankUpdate  # In RWF mode, this stores V (unscaled weights)
    bias: Array | None
    activation: Callable
    weight_transform: AbstractParameterTransform | None

    # Random Weight Factorization (RWF) params
    random_weight_factorization: bool
    rwf_log_scales: Array | None

    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]

    _in_value_shape: tuple[int, ...]
    _out_value_shape: tuple[int, ...]

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        activation: Callable | None = None,
        initializer: str = "glorot_normal",
        rwf: bool | tuple[float, float] = True,
        use_random_weight_factorization: bool | None = None,
        use_bias: bool = True,
        bias_init_lim: float = 1.0,
        weight_transform: AbstractParameterTransform | None = None,
        key: Key[Array, ""] = _key,
    ):
        # Initialise the weight matrix and (optionally) RWF scales and bias
        in_size_c = _canonical_size(in_size)
        out_size_c = _canonical_size(out_size)
        in_shape = _get_value_shape(in_size_c)
        out_shape = _get_value_shape(out_size_c)
        out_size_ = _get_size(out_size_c)
        in_size_ = _get_size(in_size_c)

        # Split keys: weight, scales (RWF), bias
        wkey, skey, bkey = jr.split(key, 3)

        # Base weights (V in RWF; full W otherwise)
        self.weight = _initializer_dict[initializer](
            wkey,
            (out_size_, in_size_),
        )

        # Random Weight Factorization setup
        if use_random_weight_factorization is not None and rwf is True:
            # Backward-compat for older flag; prefer explicit `rwf` if provided as tuple
            rwf_value: bool | tuple[float, float] | None = bool(
                use_random_weight_factorization
            )
        else:
            rwf_value = rwf

        if rwf_value is False or rwf_value is None:
            rwf_params = None
        elif rwf_value is True:
            rwf_params = (1.0, 0.1)
        else:
            mu, sigma = rwf_value
            rwf_params = (float(mu), float(sigma))

        self.random_weight_factorization = rwf_params is not None
        if rwf_params is not None:
            mu, sigma = rwf_params
            self.rwf_log_scales = mu + sigma * jr.normal(skey, shape=(out_size_,))
        else:
            self.rwf_log_scales = None
        if weight_transform is not None:
            if not isinstance(weight_transform, AbstractParameterTransform):
                raise TypeError(
                    "weight_transform must be an AbstractParameterTransform or None."
                )
            if not weight_transform.preserves_shape:
                raise ValueError("Linear requires a shape-preserving weight transform.")
            if self.random_weight_factorization:
                raise ValueError(
                    "weight_transform and random weight factorization are mutually exclusive."
                )

        if use_bias:
            self.bias = jr.uniform(
                bkey,
                (out_size_,),
                minval=-bias_init_lim,
                maxval=bias_init_lim,
            )
        else:
            self.bias = None

        self.in_size = in_size_c
        self.out_size = out_size_c
        self._in_value_shape = in_shape
        self._out_value_shape = out_shape
        if activation is None:
            act = _identity
        else:
            act = activation
        self.activation = act
        self.weight_transform = weight_transform

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = _key,
    ) -> Array:
        x_arr = jnp.asarray(x)
        in_shape = self._in_value_shape
        if in_shape:
            if x_arr.ndim < len(in_shape) or x_arr.shape[-len(in_shape) :] != in_shape:
                raise ValueError(
                    f"`x` must have trailing shape {in_shape}, got {x_arr.shape}."
                )
            leading_shape = x_arr.shape[: -len(in_shape)]
            x_flat = x_arr.reshape(leading_shape + (_get_size(in_shape),))
        else:
            if x_arr.shape == () or x_arr.shape == (1,):
                leading_shape = ()
                x_flat = x_arr.reshape((1,))
            else:
                leading_shape = x_arr.shape
                x_flat = x_arr.reshape(leading_shape + (1,))

        if isinstance(self.weight, LowRankUpdate):
            if self.weight_transform is not None:
                raise ValueError(
                    "Low-rank weights are incompatible with weight transforms."
                )
            x_flat = self.weight.apply(x_flat)
        else:
            # Map raw weights into their physical parameter space when requested.
            w = (
                self.weight
                if self.weight_transform is None
                else self.weight_transform(self.weight)
            )
            # Support both vector input (in,) and batched input (..., in).
            # Weight is shaped (out, in); contract over the last dim of x.
            x_flat = contract("oi,...i->...o", w, x_flat)
        if self.random_weight_factorization and self.rwf_log_scales is not None:
            x_flat = jnp.exp(self.rwf_log_scales) * x_flat
        if self.bias is not None:
            x_flat = x_flat + self.bias

        x_flat = jnp.real(x_flat)

        out_shape = self._out_value_shape
        if out_shape:
            y = x_flat.reshape(leading_shape + out_shape)
        else:
            if int(x_flat.shape[-1]) != 1:
                raise ValueError(
                    "Scalar out_size requires a single output feature, got shape "
                    f"{x_flat.shape}."
                )
            y = jnp.squeeze(x_flat, axis=-1)

        # Apply the activation after shaping to preserve value dimensions
        return self.activation(y)
