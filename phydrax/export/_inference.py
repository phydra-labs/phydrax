#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from phydrax.domain import DomainFunction

from .._callable import _ensure_special_kwonly_args


def array_inference_callable(
    function: Callable[..., Any] | DomainFunction,
    /,
) -> Callable[..., Any]:
    """Normalize one callable learned inference boundary."""

    if isinstance(function, DomainFunction):
        return _ensure_special_kwonly_args(function.func)
    if not callable(function):
        raise TypeError(
            "Inference export expects a callable or DomainFunction; "
            f"got {type(function).__name__}."
        )
    return _ensure_special_kwonly_args(function)


def _as_args(value: Any, /) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def make_inference_export_callable(
    function: Callable[..., Any] | DomainFunction,
    /,
    *,
    key: Any,
    preprocess: Callable[..., Any] | None,
    postprocess: Callable[..., Any] | None,
    vectorize: bool,
) -> Callable[..., Any]:
    """Compose deterministic preprocessing, inference, and postprocessing."""

    callable_function = array_inference_callable(function)

    def call_model(*model_args: Any) -> Any:
        if not vectorize:
            return callable_function(*model_args, key=key)

        def row_call(*row_args: Any) -> Any:
            return callable_function(*row_args, key=key)

        return jax.vmap(row_call)(*model_args)

    def exported(*args: Any) -> Any:
        model_args = args if preprocess is None else _as_args(preprocess(*args))
        result = call_model(*model_args)
        return result if postprocess is None else postprocess(result)

    return exported


__all__ = ["array_inference_callable", "make_inference_export_callable"]
