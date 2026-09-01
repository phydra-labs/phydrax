#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint


def nonempty_string(name: str, value: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def string_tuple(
    name: str,
    values: Sequence[str],
    /,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain only non-empty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def labels_tuple(
    name: str,
    values: Sequence[str] | None,
    size: int,
    /,
) -> tuple[str, ...]:
    if values is None:
        return ()
    result = string_tuple(name, values)
    if len(result) != size:
        raise ValueError(f"{name} must have length {size}; got {len(result)}.")
    return result


def integer_array(
    name: str,
    value: ArrayLike,
    /,
    *,
    ndim: int,
) -> Array:
    host = np.asarray(value)
    if host.ndim != ndim:
        raise ValueError(f"{name} must be rank-{ndim}; got shape {host.shape}.")
    if not np.issubdtype(host.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    int32 = np.iinfo(np.int32)
    if host.size and (np.any(host < int32.min) or np.any(host > int32.max)):
        raise ValueError(f"{name} values must be representable as int32.")
    return jnp.asarray(host, dtype=jnp.int32)


def boolean_array(
    name: str,
    value: ArrayLike | None,
    shape: tuple[int, ...],
    /,
    *,
    default: bool,
) -> Array:
    if value is None:
        return jnp.full(shape, default, dtype=bool)
    host = np.asarray(value)
    if host.dtype != np.dtype(bool):
        raise TypeError(f"{name} must have a boolean dtype.")
    if host.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {host.shape}.")
    return jnp.asarray(host, dtype=bool)


def confidence_array(
    name: str,
    value: ArrayLike | None,
    shape: tuple[int, ...],
    /,
) -> Array:
    if value is None:
        return jnp.ones(shape, dtype=jnp.float32)
    host = np.asarray(value)
    if not np.issubdtype(host.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype.")
    if host.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {host.shape}.")
    if np.any(~np.isfinite(host)) or np.any((host < 0.0) | (host > 1.0)):
        raise ValueError(f"{name} must contain finite values in [0, 1].")
    return jnp.asarray(host, dtype=jnp.float32)


def content_id(kind: str, static: dict[str, Any], arrays: Any, /) -> str:
    return canonical_fingerprint(
        {
            "arrays": array_tree_fingerprint(arrays),
            "kind": kind,
            "static": static,
        }
    )
