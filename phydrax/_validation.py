#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import SupportsFloat, SupportsIndex


def is_canonical_identifier(value: str, /) -> bool:
    """Return whether `value` is non-empty and free of surrounding whitespace."""
    return bool(value) and value == value.strip()


def canonical_identifier(value: object, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not is_canonical_identifier(value):
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def normalized_identifier(value: object, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def optional_identifier(value: object | None, name: str, /) -> str | None:
    return None if value is None else canonical_identifier(value, name)


def unique_identifiers(
    values: Sequence[object],
    name: str,
    /,
    *,
    allow_empty: bool = False,
    sort: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    result = tuple(canonical_identifier(value, name) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must be non-empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique identifiers.")
    return tuple(sorted(result)) if sort else result


def finite_real_scalar(value: object, name: str, /) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def positive_finite_float(
    value: SupportsFloat | SupportsIndex | str, name: str, /
) -> float:
    """Convert `value` with `float` and require a finite, strictly positive result."""
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def positive_integer(value: object, name: str, /) -> int:
    """Require an integral, non-boolean value of at least one."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def nonnegative_integer(value: object, name: str, /) -> int:
    """Require an integral, non-boolean value of at least zero."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


__all__ = [
    "canonical_identifier",
    "finite_real_scalar",
    "is_canonical_identifier",
    "nonnegative_integer",
    "normalized_identifier",
    "optional_identifier",
    "positive_finite_float",
    "positive_integer",
    "unique_identifiers",
]
