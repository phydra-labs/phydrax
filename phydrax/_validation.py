#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


def canonical_identifier(value: object, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
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


__all__ = [
    "canonical_identifier",
    "finite_real_scalar",
    "normalized_identifier",
    "optional_identifier",
    "unique_identifiers",
]
