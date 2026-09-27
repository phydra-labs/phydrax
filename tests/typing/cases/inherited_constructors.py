"""A concrete module that inherits an Equinox custom constructor keeps its signature."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
from typing_extensions import assert_type

from phydrax import StrictModule


class AbstractScale(StrictModule):
    factor: float = eqx.field(static=True)

    def __init__(self, *, factor: float = 1.0) -> None:
        self.factor = factor


class Doubling(AbstractScale):
    if TYPE_CHECKING:
        __init__ = AbstractScale.__init__


assert_type(Doubling(), Doubling)
assert_type(Doubling(factor=2.0), Doubling)

Doubling(factor="two")  # ty: ignore[invalid-argument-type]
Doubling(scale=2.0)  # ty: ignore[unknown-argument]
Doubling(2.0)  # ty: ignore[too-many-positional-arguments]
