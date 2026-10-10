#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical precision formats, evidence, bounded rewriting, and selection."""

from importlib import import_module
from typing import Any, Literal

import jax
import jax.numpy as jnp

from ._dtype_names import __all__ as _dtype_all
from ._precision import (
    __all__ as _format_all,
    dequantize_mx as _dequantize_mx,
    MicroscaledArray,
    MicroscalingFormat,
    quantize_mx as _quantize_mx,
)
from ._precision_rewrite import __all__ as _rewrite_all


def quantize_mx(
    value: Any,
    format: MicroscalingFormat,
    /,
    *,
    overflow: Literal["error", "saturate"] = "error",
) -> MicroscaledArray:
    """Quantize with deterministic RNE, E8M0 scales, and explicit overflow."""
    return _quantize_mx(value, format, overflow=overflow)


def dequantize_mx(
    value: MicroscaledArray,
    /,
    *,
    dtype: Any = jnp.float32,
) -> jax.Array:
    """Decode a portable microscaling payload into explicit compute precision."""
    return _dequantize_mx(value, dtype=dtype)


_FACADE_EXPORT_MODULES = ("._dtype_names", "._precision", "._precision_rewrite")


def __getattr__(name: str) -> Any:
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = sorted({*_dtype_all, *_format_all, *_rewrite_all})
