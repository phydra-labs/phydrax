#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from .._callable import _ensure_special_kwonly_args
from .._strict import StrictModule
from ..nn.models.core._base import _AbstractBaseModel


class StructuredCallable(StrictModule):
    """Wrapper marking a callable as accepting structured (tuple) inputs."""

    func: Callable

    def __init__(self, func: Callable, /):
        self.func = _ensure_special_kwonly_args(func)

    def __call__(self, x: Any, /, *, key=None, iter_=None, **kwargs: Any):
        return self.func(x, key=key, iter_=iter_, **kwargs)


def structured(func: Callable, /) -> StructuredCallable:
    """Mark a callable as supporting structured (tuple) inputs."""
    return StructuredCallable(func)


class _ConcatenatedModelCallable(StrictModule):
    raw_model: Any
    model: Callable
    supports_structured_input: bool
    supports_blockwise_input: bool
    warn_on_auto_fallback: bool

    def __init__(self, model: Callable, /):
        self.raw_model = model
        supports_structured_input = isinstance(model, StructuredCallable)
        supports_blockwise_input = False
        warn_on_auto_fallback = False
        if isinstance(model, _AbstractBaseModel) and model.supports_structured_input():
            supports_structured_input = True
            supports_blockwise_input = model.supports_blockwise_input()
            warn_on_auto_fallback = model.warn_on_auto_fallback()
        self.supports_structured_input = bool(supports_structured_input)
        self.supports_blockwise_input = bool(supports_blockwise_input)
        self.warn_on_auto_fallback = bool(warn_on_auto_fallback)
        self.model = _ensure_special_kwonly_args(model)

    def emit_auto_fallback_warning(self, message: str, /) -> None:
        if self.warn_on_auto_fallback:
            warnings.warn(message, UserWarning, stacklevel=3)

    def __call__(self, *args: Any, key=None, iter_=None, **kwargs: Any):
        if not args:
            raise ValueError("Model callable requires at least one positional input.")

        if self.supports_structured_input:

            def _as_array_or_tuple(value: Any):
                if isinstance(value, tuple):
                    return tuple(jnp.asarray(v) for v in value)
                return jnp.asarray(value)

            if len(args) == 1:
                x_in = _as_array_or_tuple(args[0])
            else:
                packed: list[Any] = []
                for value in args:
                    if isinstance(value, tuple):
                        packed.extend(jnp.asarray(v) for v in value)
                    else:
                        packed.append(jnp.asarray(value))
                x_in = tuple(packed)
            return self.model(x_in, key=key, iter_=iter_, **kwargs)

        for a in args:
            if isinstance(a, tuple):
                raise ValueError(
                    "Model callable does not support structured inputs; got a tuple argument. "
                    "Use a model that supports_structured_input() or explicitly materialize the grid."
                )
            arr = jnp.asarray(a)
            if arr.ndim > 1:
                raise ValueError(
                    "Model callable does not support batched/structured inputs; got input with shape "
                    f"{arr.shape}. Use a model that supports_structured_input() or explicitly materialize the grid."
                )

        parts = [jnp.asarray(x).reshape((-1,)) for x in args]
        x_in = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)
        return self.model(x_in, key=key, iter_=iter_, **kwargs)
