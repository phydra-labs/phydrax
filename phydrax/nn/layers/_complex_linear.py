#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax.ein import contract

from ..._doc import DOC_KEY0
from .._base import _AbstractBaseModel
from .._keys import EvalKey
from .._utils import _canonical_size, _get_size, _get_value_shape, SizeLike


class ComplexLinear(_AbstractBaseModel):
    """Complex affine map represented by real Cartesian trainable leaves."""

    weight_real: Array
    weight_imag: Array
    bias_real: Array | None
    bias_imag: Array | None
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    _in_value_shape: tuple[int, ...]
    _out_value_shape: tuple[int, ...]

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        use_bias: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        in_size_ = _canonical_size(in_size)
        out_size_ = _canonical_size(out_size)
        in_shape = _get_value_shape(in_size_)
        out_shape = _get_value_shape(out_size_)
        input_count = _get_size(in_size_)
        output_count = _get_size(out_size_)
        weight_real_key, weight_imag_key = jr.split(key)
        component_scale = 1.0 / math.sqrt(float(input_count + output_count))
        shape = (output_count, input_count)
        self.weight_real = component_scale * jr.normal(weight_real_key, shape)
        self.weight_imag = component_scale * jr.normal(weight_imag_key, shape)
        self.bias_real = jnp.zeros((output_count,)) if use_bias else None
        self.bias_imag = jnp.zeros((output_count,)) if use_bias else None
        self.in_size = in_size_
        self.out_size = out_size_
        self._in_value_shape = in_shape
        self._out_value_shape = out_shape

    @property
    def weight(self) -> Array:
        return self.weight_real + 1j * self.weight_imag

    @property
    def bias(self) -> Array | None:
        if self.bias_real is None or self.bias_imag is None:
            return None
        return self.bias_real + 1j * self.bias_imag

    def __call__(
        self,
        value: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        array = jnp.asarray(value)
        in_shape = self._in_value_shape
        if in_shape:
            if array.ndim < len(in_shape) or array.shape[-len(in_shape) :] != in_shape:
                raise ValueError(
                    f"ComplexLinear expected trailing shape {in_shape}; got {array.shape}."
                )
            leading = array.shape[: -len(in_shape)]
            flattened = array.reshape(leading + (_get_size(in_shape),))
        else:
            if array.shape == () or array.shape == (1,):
                leading = ()
                flattened = array.reshape((1,))
            else:
                leading = array.shape
                flattened = array.reshape(leading + (1,))
        output = contract("oi,...i->...o", self.weight, flattened)
        bias = self.bias
        if bias is not None:
            output = output + bias
        out_shape = self._out_value_shape
        if out_shape:
            return output.reshape(leading + out_shape)
        if int(output.shape[-1]) != 1:
            raise ValueError("Scalar ComplexLinear output requires one feature.")
        return jnp.squeeze(output, axis=-1)


__all__ = ["ComplexLinear"]
