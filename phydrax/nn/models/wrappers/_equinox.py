#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import inspect
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...._callable import _ensure_special_kwonly_args, _KeyIterAdapter
from ...._differentiation import DerivativeRegularity
from ...._doc import DOC_KEY0
from ...._model import AbstractArrayModel
from ..._base import _AbstractBaseModel, _AbstractStructuredInputModel
from ..._contracts import AFFINE, compose_regularity, model_regularity, SMOOTH
from ..._keys import EvalKey
from ..._utils import _canonical_size, _get_size, _get_value_shape, SizeLike
from ...activations import activation_regularity


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


_EQX_MLP_IDENTITY = (
    inspect.signature(eqx.nn.MLP.__init__).parameters["final_activation"].default
)
_EQX_AFFINE_LAYERS = (
    eqx.nn.Linear,
    eqx.nn.Conv,
    eqx.nn.ConvTranspose,
    eqx.nn.Identity,
    eqx.nn.Dropout,
    eqx.nn.AvgPool1d,
    eqx.nn.AvgPool2d,
    eqx.nn.AvgPool3d,
    eqx.nn.AdaptiveAvgPool1d,
    eqx.nn.AdaptiveAvgPool2d,
    eqx.nn.AdaptiveAvgPool3d,
)
_EQX_PIECEWISE_LINEAR_LAYERS = (
    eqx.nn.PReLU,
    eqx.nn.MaxPool1d,
    eqx.nn.MaxPool2d,
    eqx.nn.MaxPool3d,
    eqx.nn.AdaptiveMaxPool1d,
    eqx.nn.AdaptiveMaxPool2d,
    eqx.nn.AdaptiveMaxPool3d,
)
_EQX_NORMALIZATIONS = (eqx.nn.LayerNorm, eqx.nn.RMSNorm, eqx.nn.GroupNorm)


def _activation(fn: Any, /) -> DerivativeRegularity | None:
    if fn is _EQX_MLP_IDENTITY:
        return AFFINE
    return _module_regularity(fn)


def _module_regularity(module: Any, /) -> DerivativeRegularity | None:
    """Structural value regularity of a known Equinox layer tree (`None` if unknown).

    Affine layers (linear, convolution, identity, average pooling, inference-mode
    dropout) are degree-1 polynomials; `PReLU` and max pooling are C0 piecewise
    linear; normalizations with a positive epsilon are smooth; `Sequential` and
    `MLP` compose their stages; `Lambda` and bare callables use
    `activation_regularity`.
    """
    if isinstance(module, _KeyIterAdapter):
        return _module_regularity(module.func)
    if isinstance(module, _EQX_AFFINE_LAYERS):
        return AFFINE
    if isinstance(module, _EQX_PIECEWISE_LINEAR_LAYERS):
        return DerivativeRegularity.piecewise_polynomial(continuity=0, degree_bound=1)
    if isinstance(module, _EQX_NORMALIZATIONS):
        return SMOOTH if module.eps > 0.0 else None
    if isinstance(module, eqx.nn.Sequential):
        return compose_regularity(*(_module_regularity(layer) for layer in module.layers))
    if isinstance(module, eqx.nn.MLP):
        stages = []
        for layer in module.layers[:-1]:
            stages += [_module_regularity(layer), _activation(module.activation)]
        stages += [
            _module_regularity(module.layers[-1]),
            _activation(module.final_activation),
        ]
        return compose_regularity(*stages)
    if isinstance(module, eqx.nn.Lambda):
        return activation_regularity(module.fn)
    if isinstance(module, AbstractArrayModel):
        return model_regularity(module)
    return activation_regularity(module)


def _stateless_module(module: Any, /, *, wrapper: str) -> Any:
    """Wrap `module` for keyword dispatch after rejecting Equinox state."""
    if any(
        isinstance(node, eqx.nn.StateIndex)
        for node in jax.tree_util.tree_leaves(
            module, is_leaf=lambda node: isinstance(node, eqx.nn.StateIndex)
        )
    ):
        raise TypeError(
            f"stateful Equinox modules are not supported by {wrapper}; "
            "declare model state explicitly"
        )
    return _ensure_special_kwonly_args(module)


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
                f"Expected scalar output with no leading axes; got leading_shape={leading_shape}."
            )
        if y_arr.shape == ():
            return y_arr
        if y_arr.shape == (1,):
            return jnp.squeeze(y_arr, axis=0)
        raise ValueError(
            f"Wrapped module returned the wrong shape for scalar out_size. Expected () or (1,), got {y_arr.shape}."
        )

    out_numel = _get_size(out_size)
    out_shape = _get_value_shape(out_size)
    if leading_shape:
        raise ValueError(
            f"Expected unbatched output (no leading axes) for layout='value'. Got leading_shape={leading_shape}."
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

    The wrapper is a `ParameterOwner`: every inexact array of `module` is a
    PARAMETER. Modules holding `eqx.nn.StateIndex` state are rejected.
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
        self.module = _stateless_module(module, wrapper=type(self).__name__)
        self.in_size = _canonical_size(in_size)
        self.out_size = _canonical_size(out_size)
        self.layout = layout

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_
        if self.layout == "passthrough":
            return self.module(x, key=key, **kwargs)

        x_flat, leading_shape = _flatten_value(x, in_size=self.in_size)
        y_flat = self.module(x_flat, key=key, **kwargs)
        return _reshape_value(y_flat, leading_shape=leading_shape, out_size=self.out_size)

    def _value_regularity(self) -> DerivativeRegularity | None:
        return _module_regularity(self.module)


class EquinoxStructuredModel(_AbstractStructuredInputModel):
    """Equinox/JAX callable adapter that supports structured (tuple) inputs.

    Like `EquinoxModel`, the module's inexact arrays are PARAMETER and stateful
    modules are rejected.
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
        layout: _Layout = "passthrough",
    ):
        self.module = _stateless_module(module, wrapper=type(self).__name__)
        self.in_size = _canonical_size(in_size)
        self.out_size = _canonical_size(out_size)
        self.layout = layout

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: EvalKey = DOC_KEY0,
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

    def _value_regularity(self) -> DerivativeRegularity | None:
        return _module_regularity(self.module)


__all__ = ["EquinoxModel", "EquinoxStructuredModel"]
