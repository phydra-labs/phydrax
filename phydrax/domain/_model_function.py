#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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
    model: Callable
    supports_structured_input: bool

    def __init__(self, model: Callable, /):
        supports_structured_input = isinstance(model, StructuredCallable)
        if isinstance(model, _AbstractBaseModel) and model.supports_structured_input():
            supports_structured_input = True
        self.supports_structured_input = bool(supports_structured_input)
        self.model = _ensure_special_kwonly_args(model)

    def __call__(self, *args: Any, key=None, iter_=None, **kwargs: Any):
        if not args:
            raise ValueError("Model callable requires at least one positional input.")

        if self.supports_structured_input:
            if len(args) == 1:
                x_in = args[0]
            else:
                flat: list[Any] = []
                for a in args:
                    if isinstance(a, tuple):
                        flat.extend(a)
                    else:
                        flat.append(a)
                x_in = tuple(jnp.asarray(x) for x in flat)
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
