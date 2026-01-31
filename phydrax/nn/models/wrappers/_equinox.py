#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array, Key

from ...._callable import _ensure_special_kwonly_args
from ...._doc import DOC_KEY0
from ..._utils import _canonical_size, _get_size, _get_value_shape, SizeLike
from ..core._base import _AbstractBaseModel, _AbstractStructuredInputModel


_Layout = Literal["value", "passthrough"]


def _flatten_value(x: Array, /, *, in_size: int | tuple[int, ...] | Literal["scalar"]):
    x_arr = jnp.asarray(x)
    if in_size == "scalar":
        if x_arr.shape == ():
            return x_arr, ()
        if x_arr.shape == (1,):
            return x_arr.reshape(()), ()
        raise ValueError(f"`x` must have scalar shape () (or (1,)), got {x_arr.shape}.")

    in_shape = _get_value_shape(in_size)
    if x_arr.shape != in_shape:
        raise ValueError(f"`x` must have shape {in_shape}, got {x_arr.shape}.")
    x_flat = x_arr.reshape((_get_size(in_size),))
    return x_flat, ()


def _reshape_value(
    y_flat: Array,
    /,
    *,
    leading_shape: tuple[int, ...],
    out_size: int | tuple[int, ...] | Literal["scalar"],
) -> Array:
    y_arr = jnp.asarray(y_flat)
    if out_size == "scalar":
        if leading_shape:
            raise ValueError(
                "Expected scalar output with no leading axes; got "
                f"leading_shape={leading_shape}."
            )
        if y_arr.shape == ():
            return y_arr
        if y_arr.shape == (1,):
            return jnp.squeeze(y_arr, axis=0)
        raise ValueError(
            "Wrapped module returned the wrong shape for scalar out_size. Expected "
            f"() or (1,), got {y_arr.shape}."
        )

    out_numel = _get_size(out_size)
    out_shape = _get_value_shape(out_size)
    if leading_shape:
        raise ValueError(
            "Expected unbatched output (no leading axes) for layout='value'. "
            f"Got leading_shape={leading_shape}."
        )
    if y_arr.shape != (out_numel,):
        raise ValueError(
            "Wrapped module returned the wrong shape. Expected "
            f"({out_numel},) (to reshape to {out_shape}), got {y_arr.shape}."
        )
    return y_arr.reshape(out_shape)


class EquinoxModel(_AbstractBaseModel):
    """Adapter for arbitrary Equinox/JAX callables with Phydrax model metadata.

    Default `layout="value"` treats `in_size/out_size` as **value shapes** and performs:
    (flatten value axes) -> (call wrapped module) -> (reshape back to value axes).

    Use `layout="passthrough"` to forward inputs/outputs unchanged.
    """

    module: Any
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    layout: _Layout

    def __init__(
        self,
        module: Any,
        /,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        layout: _Layout = "value",
    ):
        self.module = _ensure_special_kwonly_args(module)
        self.in_size = _canonical_size(in_size)
        self.out_size = _canonical_size(out_size)
        self.layout = layout

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_
        if self.layout == "passthrough":
            return self.module(x, key=key, **kwargs)

        x_flat, leading_shape = _flatten_value(x, in_size=self.in_size)
        y_flat = self.module(x_flat, key=key, **kwargs)
        return _reshape_value(y_flat, leading_shape=leading_shape, out_size=self.out_size)


class EquinoxStructuredModel(_AbstractStructuredInputModel):
    """Equinox/JAX callable adapter that supports structured (tuple) inputs."""

    module: Any
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    layout: _Layout

    def __init__(
        self,
        module: Any,
        /,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        layout: _Layout = "passthrough",
    ):
        self.module = _ensure_special_kwonly_args(module)
        self.in_size = _canonical_size(in_size)
        self.out_size = _canonical_size(out_size)
        self.layout = layout

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_
        if self.layout == "passthrough":
            if isinstance(x, tuple):
                x_in = tuple(jnp.asarray(a) for a in x)
            else:
                x_in = jnp.asarray(x)
            return self.module(x_in, key=key, **kwargs)

        if isinstance(x, tuple):
            parts = []
            for a in x:
                arr = jnp.asarray(a)
                if arr.ndim > 1:
                    raise ValueError(
                        "EquinoxStructuredModel(layout='value') expects tuple inputs to be "
                        "scalars/vectors (ndim<=1). Got an input with shape "
                        f"{arr.shape}."
                    )
                parts.append(arr.reshape((-1,)))
            x_arr = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)
        else:
            x_arr = jnp.asarray(x)

        x_flat, leading_shape = _flatten_value(x_arr, in_size=self.in_size)
        y_flat = self.module(x_flat, key=key, **kwargs)
        return _reshape_value(y_flat, leading_shape=leading_shape, out_size=self.out_size)


__all__ = ["EquinoxModel", "EquinoxStructuredModel"]
