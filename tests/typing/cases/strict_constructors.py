"""StrictModule construction is typed by each concrete class constructor."""

from __future__ import annotations

from typing import TypeVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from typing_extensions import assert_type, Self

from phydrax import StrictModule


class Sample(StrictModule):
    values: Array
    label: str = eqx.field(static=True)


class Interval(StrictModule):
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)

    def __init__(self, lower: float, upper: float, /, *, closed: bool = True) -> None:
        del closed
        self.lower = lower
        self.upper = upper

    @classmethod
    def unit(cls) -> Self:
        return cls(0.0, 1.0)

    @classmethod
    def malformed(cls) -> Self:
        return cls(0.0, "one")  # ty: ignore[invalid-argument-type]


_IntervalT = TypeVar("_IntervalT", bound=Interval)


def build(kind: type[_IntervalT], /) -> _IntervalT:
    return kind(0.0, 1.0, closed=False)


def build_incomplete(kind: type[_IntervalT], /) -> _IntervalT:
    return kind(0.0)  # ty: ignore[missing-argument]


assert_type(Sample(jnp.zeros(3), "sample"), Sample)
assert_type(Interval(0.0, 1.0), Interval)
assert_type(Interval.unit(), Interval)
assert_type(build(Interval), Interval)

Sample(jnp.zeros(3), 3)  # ty: ignore[invalid-argument-type]
Sample(jnp.zeros(3), label="sample", unit="m")  # ty: ignore[unknown-argument]
Interval(0.0)  # ty: ignore[missing-argument]
Interval(0.0, 1.0, open=True)  # ty: ignore[unknown-argument]
Interval(lower=0.0, upper=1.0)  # ty: ignore[positional-only-parameter-as-kwarg]
