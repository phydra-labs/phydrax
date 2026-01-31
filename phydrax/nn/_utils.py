#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias


SizeLike: TypeAlias = int | Sequence[int] | Literal["scalar"]


def _identity(*args: Any, **kwargs: Any) -> Any:
    del kwargs
    if not args:
        return None
    if len(args) == 1:
        return args[0]
    return args


def _get_value_shape(size: SizeLike) -> tuple[int, ...]:
    """Convert a value-size specification to a trailing value shape.

    Rules:
    - `"scalar"` -> `()`
    - `k` -> `(k,)`
    - `(s1, s2, ...)` -> `(s1, s2, ...)`

    `()` is treated as scalar.
    """
    if size == "scalar":
        return ()
    if isinstance(size, int):
        k = int(size)
        if k <= 0:
            raise ValueError(f"Size entries must be positive; got {size!r}.")
        return (k,)
    shape = tuple(int(s) for s in size)
    if not shape:
        return ()
    if any(s <= 0 for s in shape):
        raise ValueError(f"Shape entries must be positive; got {shape!r}.")
    return shape


def _canonical_size(size: SizeLike) -> int | tuple[int, ...] | Literal["scalar"]:
    """Normalize a size-like spec to a stable representation.

    - scalar -> `"scalar"`
    - 1D -> `int`
    - rank>=2 -> `tuple[int, ...]`
    """
    shape = _get_value_shape(size)
    if not shape:
        return "scalar"
    if len(shape) == 1:
        return int(shape[0])
    return shape


def _get_size(size: SizeLike) -> int:
    """Convert a size/shape specification to an integer feature count.

    - `"scalar"` -> 1
    - `k` -> `k`
    - `(s1, ..., sr)` -> `∏ si`
    """
    if size == "scalar":
        return 1
    if isinstance(size, int):
        k = int(size)
        if k <= 0:
            raise ValueError(f"Size entries must be positive; got {size!r}.")
        return k
    shape = tuple(int(s) for s in size)
    if not shape:
        return 1
    n = 1
    for s in shape:
        if s <= 0:
            raise ValueError(f"Shape entries must be positive; got {shape!r}.")
        n *= s
    return int(n)
