"""Key annotations and closed selector aliases must be valid type forms."""

from __future__ import annotations

from typing import Literal, TypeAlias

from jax import Array
from jaxtyping import Key


Mode: TypeAlias = Literal["dense", "sparse"]
UndeclaredMode: type = Literal["dense", "sparse"]  # ty: ignore[invalid-assignment]


def draw(key: Key[Array, ""], /) -> Array:
    return key


def bare(key: Key, /) -> None: ...  # ty: ignore[invalid-type-form]


def select(mode: Mode, /) -> None: ...


def undeclared(mode: UndeclaredMode, /) -> None: ...  # ty: ignore[invalid-type-form]


select("dense")
select("matrix-free")  # ty: ignore[invalid-argument-type]
