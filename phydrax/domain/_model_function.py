#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from .._callable import _ensure_special_kwonly_args
from .._strict import StrictModule
from ..nn.models.core._base import DomainInputMode
from ..nn.models.core._loss import model_domain_metadata


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
    input_mode: DomainInputMode
    supports_structured_input: bool
    supports_blockwise_input: bool
    warn_on_auto_fallback: bool
    _call_has_var_kwargs: bool
    _call_has_key: bool
    _call_has_iter: bool

    def __init__(
        self,
        model: Callable,
        /,
        *,
        input_mode: DomainInputMode | None = None,
    ):
        self.raw_model = model
        supports_structured_input = isinstance(model, StructuredCallable)
        supports_blockwise_input = False
        warn_on_auto_fallback = False
        inferred_mode: DomainInputMode = (
            "structured" if supports_structured_input else "flat"
        )
        metadata = model_domain_metadata(model)
        if metadata is not None:
            (
                inferred_mode,
                supports_structured_input,
                supports_blockwise_input,
                warn_on_auto_fallback,
            ) = metadata
        mode = inferred_mode if input_mode is None else input_mode
        if mode not in ("flat", "structured"):
            raise ValueError("input_mode must be either 'flat' or 'structured'.")
        if mode == "structured" and not supports_structured_input:
            raise ValueError(
                "input_mode='structured' requires a model that supports structured inputs. "
                "Use structured=True for plain callables that intentionally accept tuple inputs."
            )
        self.input_mode = mode
        self.supports_structured_input = bool(supports_structured_input)
        self.supports_blockwise_input = bool(supports_blockwise_input)
        self.warn_on_auto_fallback = bool(warn_on_auto_fallback)
        sig = inspect.signature(model)
        params = sig.parameters
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        has_key = "key" in params
        has_iter = "iter_" in params
        if has_key and params["key"].kind != inspect.Parameter.KEYWORD_ONLY:
            raise TypeError("`key` must be a keyword-only argument for model calls.")
        if has_iter and params["iter_"].kind != inspect.Parameter.KEYWORD_ONLY:
            raise TypeError("`iter_` must be a keyword-only argument for model calls.")
        self._call_has_var_kwargs = bool(has_var_kwargs)
        self._call_has_key = bool(has_key)
        self._call_has_iter = bool(has_iter)

    def emit_auto_fallback_warning(self, message: str, /) -> None:
        if self.warn_on_auto_fallback:
            warnings.warn(message, UserWarning, stacklevel=3)

    def _call_model(self, x: Any, /, *, key=None, iter_=None, **kwargs: Any):
        out_kwargs = kwargs
        if self._call_has_key or self._call_has_var_kwargs:
            out_kwargs = dict(out_kwargs)
            out_kwargs["key"] = key
        if iter_ is not None and (self._call_has_iter or self._call_has_var_kwargs):
            if out_kwargs is kwargs:
                out_kwargs = dict(out_kwargs)
            out_kwargs["iter_"] = iter_
        return self.raw_model(x, **out_kwargs)

    def __call__(self, *args: Any, key=None, iter_=None, **kwargs: Any):
        if not args:
            raise ValueError("Model callable requires at least one positional input.")

        input_mode = kwargs.pop("_phydrax_model_input_mode", self.input_mode)
        if input_mode not in ("flat", "structured"):
            raise ValueError(
                "_phydrax_model_input_mode must be either 'flat' or 'structured'."
            )
        if input_mode == "structured" and not self.supports_structured_input:
            raise ValueError(
                "Structured model input was requested, but this model does not "
                "support structured inputs."
            )

        if input_mode == "structured":

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
            return self._call_model(x_in, key=key, iter_=iter_, **kwargs)

        arrays: list[Any] = []
        for value in args:
            if isinstance(value, tuple):
                raise ValueError(
                    "Model callable does not support structured inputs; got a tuple argument. "
                    "Use a model that supports_structured_input() or explicitly materialize the grid."
                )
            arrays.append(jnp.asarray(value))

        if len(arrays) == 1:
            x_in = arrays[0]
            return self._call_model(x_in, key=key, iter_=iter_, **kwargs)

        leading_shape: tuple[int, ...] | None = None
        for arr in arrays:
            if arr.ndim < 2:
                continue
            candidate = tuple(int(i) for i in arr.shape[:-1])
            if leading_shape is None:
                leading_shape = candidate
                continue
            if candidate != leading_shape:
                raise ValueError(
                    "Flat model packing requires batched inputs to share leading "
                    f"shape; got {candidate} and {leading_shape}."
                )

        if leading_shape is None:
            parts = [arr.reshape((-1,)) for arr in arrays]
            x_in = jnp.concatenate(parts, axis=0)
            return self._call_model(x_in, key=key, iter_=iter_, **kwargs)

        parts = []
        for arr in arrays:
            if arr.ndim == 0:
                part = jnp.broadcast_to(arr, leading_shape + (1,))
            elif tuple(int(i) for i in arr.shape) == leading_shape:
                part = arr.reshape(leading_shape + (1,))
            elif tuple(int(i) for i in arr.shape[:-1]) == leading_shape:
                part = arr.reshape(leading_shape + (int(arr.shape[-1]),))
            else:
                raise ValueError(
                    "Flat model packing could not align input with shape "
                    f"{arr.shape} to leading batch shape {leading_shape}."
                )
            parts.append(part)
        x_in = jnp.concatenate(parts, axis=-1)
        return self._call_model(x_in, key=key, iter_=iter_, **kwargs)
