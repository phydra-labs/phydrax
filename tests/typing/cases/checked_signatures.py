"""`phydrax.typing.checked` preserves the static signature it decorates."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from typing_extensions import assert_type, Self

import phydrax.typing as pt
from phydrax import StrictModule


class Interval(StrictModule):
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)

    @pt.checked
    def __init__(self, lower: float, upper: float, /, *, closed: bool = True) -> None:
        del closed
        self.lower = lower
        self.upper = upper

    @pt.checked
    def widen(self, other: Interval, *, scale: float = 1.0) -> Interval:
        return Interval(
            min(self.lower, other.lower), scale * max(self.upper, other.upper)
        )

    @classmethod
    @pt.checked
    def unit(cls) -> Self:
        return cls(0.0, 1.0)

    @staticmethod
    @pt.checked
    def length(interval: Interval) -> float:
        return interval.upper - interval.lower

    @property
    @pt.checked
    def width(self) -> float:
        return self.upper - self.lower


@pt.checked
def scaled[T](values: Array, scale: float, combine: Callable[[Array], T], /) -> T:
    return combine(scale * values)


unit = Interval(0.0, 1.0)
assert_type(unit, Interval)
assert_type(Interval.unit(), Interval)
assert_type(unit.widen(unit, scale=2.0), Interval)
assert_type(Interval.length(unit), float)
assert_type(unit.width, float)
assert_type(scaled(jnp.ones(3), 2.0, jnp.sum), Array)
assert_type(scaled(jnp.ones(3), 2.0, len), int)

Interval(0.0)  # ty: ignore[missing-argument]
Interval(0.0, "one")  # ty: ignore[invalid-argument-type]
Interval(lower=0.0, upper=1.0)  # ty: ignore[positional-only-parameter-as-kwarg]
unit.widen(1.0)  # ty: ignore[invalid-argument-type]
unit.widen(unit, factor=2.0)  # ty: ignore[unknown-argument]
scaled(jnp.ones(3), 2.0)  # ty: ignore[missing-argument]
