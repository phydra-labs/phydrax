#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from operator import index

import equinox as eqx

from ._core import AbstractFiniteRealAlgebraSpec
from ._resources import AlgebraResourceBudget


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _subtract(left, right):
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _negate(value):
    return tuple(-entry for entry in value)


def _conjugate(level: int, value, /):
    if level == 0:
        return tuple(value)
    half = 1 << (level - 1)
    return _conjugate(level - 1, value[:half]) + _negate(value[half:])


def _multiply(level: int, left, right, /):
    if level == 0:
        return (left[0] * right[0],)
    half = 1 << (level - 1)
    a, b = left[:half], left[half:]
    c, d = right[:half], right[half:]
    first = _subtract(
        _multiply(level - 1, a, c),
        _multiply(level - 1, _conjugate(level - 1, d), b),
    )
    second = _add(
        _multiply(level - 1, d, a),
        _multiply(level - 1, b, _conjugate(level - 1, c)),
    )
    return first + second


def _basis(dimension: int, position: int, /):
    return tuple(Fraction(int(index_ == position)) for index_ in range(dimension))


def _labels(level: int, /) -> tuple[str, ...]:
    if level == 0:
        return ("1",)
    if level == 1:
        return ("1", "i")
    if level == 2:
        return ("1", "i", "j", "k")
    if level == 3:
        return ("1", "i", "j", "k", "l", "il", "jl", "kl")
    return ("1",) + tuple(f"e{index_}" for index_ in range(1, 1 << level))


def _terms(level: int, /):
    dimension = 1 << level
    basis = tuple(_basis(dimension, position) for position in range(dimension))
    terms = []
    for left in range(dimension):
        for right in range(dimension):
            product = _multiply(level, basis[left], basis[right])
            for output, coefficient in enumerate(product):
                if coefficient:
                    terms.append(
                        (
                            left,
                            right,
                            output,
                            coefficient.numerator,
                            coefficient.denominator,
                        )
                    )
    return tuple(terms)


def _family_claims(level: int, labels: tuple[str, ...], /):
    proven = ("proven", "family_construction", ())
    disproven_zero = (
        "disproven",
        "family_construction",
        ("no-zero-divisor-family-proof",),
    )
    claims = {
        "positive_norm": proven,
        "division_algebra": proven
        if level <= 3
        else (
            "disproven",
            "explicit_witness",
            ("cayley-dickson-zero-divisor",),
        ),
        "has_zero_divisors": disproven_zero
        if level <= 3
        else (
            "proven",
            "explicit_witness",
            ("cayley-dickson-zero-divisor",),
        ),
        "norm_multiplicative": proven
        if level <= 3
        else (
            "disproven",
            "family_construction",
            ("level>=4",),
        ),
    }
    if level == 2:
        claims["commutative"] = (
            "disproven",
            "explicit_witness",
            (labels[1], labels[2]),
        )
    return claims


def _initialize_cayley_dickson(
    instance: AbstractFiniteRealAlgebraSpec,
    level: int,
    budget: AlgebraResourceBudget | None,
    /,
) -> None:
    if isinstance(level, bool):
        raise TypeError("Cayley-Dickson level must be an integer.")
    level_ = index(level)
    if level_ < 0:
        raise ValueError("Cayley-Dickson level must be nonnegative.")
    dimension = 1 << level_
    budget_ = AlgebraResourceBudget() if budget is None else budget
    budget_.admit_coordinates(dimension)
    labels = _labels(level_)
    conjugation = tuple(
        tuple(
            1 if row == column == 0 else -1 if row == column else 0
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    family = {
        0: "real",
        1: "complex",
        2: "quaternion",
        3: "octonion",
    }.get(level_, f"cayley-dickson-{level_}")
    AbstractFiniteRealAlgebraSpec.__init__(
        instance,
        family,
        labels,
        _terms(level_),
        (1,) + (0,) * (dimension - 1),
        conjugation,
        convention={
            "kind": "cayley-dickson-left-v1",
            "level": level_,
            "pair_product": "(ac-conj(d)b,da+bconj(c))",
        },
        family_claims=_family_claims(level_, labels),
        budget=budget_,
    )
    instance.level = level_


class CayleyDicksonAlgebraSpec(AbstractFiniteRealAlgebraSpec):
    """Canonical finite Cayley-Dickson algebra under one fixed pair convention."""

    level: int = eqx.field(static=True)

    def __init__(
        self,
        level: int,
        /,
        *,
        budget: AlgebraResourceBudget | None = None,
    ):
        _initialize_cayley_dickson(self, level, budget)

    def _family_marker(self) -> str:
        return self.family


class RealAlgebraSpec(AbstractFiniteRealAlgebraSpec):
    level: int = eqx.field(static=True)

    def __init__(self, *, budget: AlgebraResourceBudget | None = None):
        _initialize_cayley_dickson(self, 0, budget)

    def _family_marker(self) -> str:
        return self.family


class ComplexAlgebraSpec(AbstractFiniteRealAlgebraSpec):
    level: int = eqx.field(static=True)

    def __init__(self, *, budget: AlgebraResourceBudget | None = None):
        _initialize_cayley_dickson(self, 1, budget)

    def _family_marker(self) -> str:
        return self.family


class QuaternionAlgebraSpec(AbstractFiniteRealAlgebraSpec):
    level: int = eqx.field(static=True)

    def __init__(self, *, budget: AlgebraResourceBudget | None = None):
        _initialize_cayley_dickson(self, 2, budget)

    def _family_marker(self) -> str:
        return self.family


class OctonionAlgebraSpec(AbstractFiniteRealAlgebraSpec):
    level: int = eqx.field(static=True)

    def __init__(self, *, budget: AlgebraResourceBudget | None = None):
        _initialize_cayley_dickson(self, 3, budget)

    def _family_marker(self) -> str:
        return self.family


__all__ = [
    "CayleyDicksonAlgebraSpec",
    "ComplexAlgebraSpec",
    "OctonionAlgebraSpec",
    "QuaternionAlgebraSpec",
    "RealAlgebraSpec",
]
