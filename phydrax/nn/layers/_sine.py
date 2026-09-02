# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import math
from typing import Any, Literal

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax.ein import contract

from ..._doc import DOC_KEY0
from .._base import _AbstractBaseModel
from .._keys import EvalKey
from .._utils import _canonical_size, _get_size, _get_value_shape, SizeLike


class SineLayer(_AbstractBaseModel):
    r"""SIREN affine-sinusoidal layer with frequency-aware initialization."""

    weight: Array
    bias: Array | None
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    _in_value_shape: tuple[int, ...]
    _out_value_shape: tuple[int, ...]
    omega: float
    is_first: bool

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        omega: float = 1.0,
        is_first: bool = False,
        use_bias: bool = True,
        dtype: Any | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        omega_value = float(omega)
        if not math.isfinite(omega_value) or omega_value <= 0.0:
            raise ValueError("omega must be finite and positive.")
        in_size_c = _canonical_size(in_size)
        out_size_c = _canonical_size(out_size)
        in_count = _get_size(in_size_c)
        out_count = _get_size(out_size_c)
        bound = 1.0 / in_count if is_first else math.sqrt(6.0 / in_count) / omega_value
        weight_key, bias_key = jr.split(key)
        weight_dtype = jnp.asarray(0.0).dtype if dtype is None else jnp.dtype(dtype)
        self.weight = jr.uniform(
            weight_key,
            (out_count, in_count),
            minval=-bound,
            maxval=bound,
            dtype=weight_dtype,
        )
        self.bias = (
            jr.uniform(
                bias_key,
                (out_count,),
                minval=-bound,
                maxval=bound,
                dtype=weight_dtype,
            )
            if use_bias
            else None
        )
        self.in_size = in_size_c
        self.out_size = out_size_c
        self._in_value_shape = _get_value_shape(in_size_c)
        self._out_value_shape = _get_value_shape(out_size_c)
        self.omega = omega_value
        self.is_first = bool(is_first)

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        value = jnp.asarray(x)
        if self._in_value_shape:
            if (
                value.ndim < len(self._in_value_shape)
                or value.shape[-len(self._in_value_shape) :] != self._in_value_shape
            ):
                raise ValueError(
                    "`x` must have trailing shape "
                    f"{self._in_value_shape}, got {value.shape}."
                )
            leading_shape = value.shape[: -len(self._in_value_shape)]
            flat = value.reshape(leading_shape + (_get_size(self._in_value_shape),))
        elif value.shape in ((), (1,)):
            leading_shape = ()
            flat = value.reshape((1,))
        else:
            leading_shape = value.shape
            flat = value.reshape(leading_shape + (1,))
        affine = contract("oi,...i->...o", self.weight, flat)
        if self.bias is not None:
            affine = affine + self.bias
        affine = jnp.real(affine)
        if self._out_value_shape:
            output = affine.reshape(leading_shape + self._out_value_shape)
        else:
            if int(affine.shape[-1]) != 1:
                raise ValueError(
                    "Scalar out_size requires a single output feature, got shape "
                    f"{affine.shape}."
                )
            output = jnp.squeeze(affine, axis=-1)
        return jnp.sin(self.omega * output)


__all__ = ["SineLayer"]
