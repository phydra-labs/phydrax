from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction
from numbers import Integral
from typing import Any, TypeAlias

import equinox as eqx

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState


Exponent: TypeAlias = int | Fraction
DimensionTerms: TypeAlias = Mapping[str, Exponent] | Iterable[tuple[str, Exponent]]


def _exponent(value: Exponent) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
        raise TypeError("dimension exponents must be integers or fractions")
    return Fraction(value)


class DimensionSignature(StrictModule, NonTrainableState):
    """Canonical exact powers of named physical-dimension axes."""

    terms: tuple[tuple[str, int, int], ...] = eqx.field(static=True)
    dimension_id: str = eqx.field(static=True)

    def __init__(self, terms: DimensionTerms = ()) -> None:
        items = terms.items() if isinstance(terms, Mapping) else terms
        accumulated: dict[str, Fraction] = {}
        for axis, exponent in items:
            if not isinstance(axis, str) or not axis or axis.strip() != axis:
                raise ValueError(
                    "dimension axis names must be non-empty stripped strings"
                )
            accumulated[axis] = accumulated.get(axis, Fraction(0)) + _exponent(exponent)
        canonical = tuple(
            (axis, exponent.numerator, exponent.denominator)
            for axis, exponent in sorted(accumulated.items())
            if exponent
        )
        self.terms = canonical
        self.dimension_id = canonical_fingerprint(
            {"kind": "dimension_signature", "terms": canonical}
        )

    @property
    def is_dimensionless(self) -> bool:
        return not self.terms

    def exponent(self, axis: str) -> Fraction:
        for name, numerator, denominator in self.terms:
            if name == axis:
                return Fraction(numerator, denominator)
        return Fraction(0)

    def multiply(self, other: DimensionSignature) -> DimensionSignature:
        if not isinstance(other, DimensionSignature):
            raise TypeError("dimension multiplication requires a DimensionSignature")
        return DimensionSignature((*self._fraction_terms(), *other._fraction_terms()))

    def divide(self, other: DimensionSignature) -> DimensionSignature:
        if not isinstance(other, DimensionSignature):
            raise TypeError("dimension division requires a DimensionSignature")
        return DimensionSignature(
            (
                *self._fraction_terms(),
                *((axis, -power) for axis, power in other._fraction_terms()),
            )
        )

    def power(self, exponent: Exponent) -> DimensionSignature:
        power = _exponent(exponent)
        return DimensionSignature(
            (axis, value * power) for axis, value in self._fraction_terms()
        )

    def _fraction_terms(self) -> tuple[tuple[str, Fraction], ...]:
        return tuple(
            (axis, Fraction(numerator, denominator))
            for axis, numerator, denominator in self.terms
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": [
                {
                    "axis": axis,
                    "numerator": numerator,
                    "denominator": denominator,
                }
                for axis, numerator, denominator in self.terms
            ],
            "dimension_id": self.dimension_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DimensionSignature:
        if not isinstance(payload, Mapping):
            raise TypeError("dimension payload must be a mapping")
        expected = {"terms", "dimension_id"}
        if set(payload) != expected:
            raise ValueError("dimension payload must use the canonical fields")
        raw_terms = payload.get("terms")
        if not isinstance(raw_terms, list):
            raise TypeError("dimension payload terms must be a list")
        terms: list[tuple[str, Fraction]] = []
        for raw_term in raw_terms:
            if not isinstance(raw_term, Mapping):
                raise TypeError("dimension payload terms must be mappings")
            if set(raw_term) != {"axis", "numerator", "denominator"}:
                raise ValueError("dimension terms must use the canonical fields")
            axis = raw_term.get("axis")
            numerator = raw_term.get("numerator")
            denominator = raw_term.get("denominator")
            if not isinstance(axis, str):
                raise TypeError("dimension payload axis must be a string")
            if isinstance(numerator, bool) or not isinstance(numerator, Integral):
                raise TypeError("dimension payload numerator must be an integer")
            if isinstance(denominator, bool) or not isinstance(denominator, Integral):
                raise TypeError("dimension payload denominator must be an integer")
            terms.append((axis, Fraction(int(numerator), int(denominator))))
        dimension = cls(terms)
        claimed_id = payload.get("dimension_id")
        if not isinstance(claimed_id, str):
            raise TypeError("dimension payload dimension_id must be a string")
        if claimed_id != dimension.dimension_id:
            raise ValueError("dimension payload fingerprint does not match its content")
        return dimension

    def __mul__(self, other: DimensionSignature) -> DimensionSignature:
        return self.multiply(other)

    def __truediv__(self, other: DimensionSignature) -> DimensionSignature:
        return self.divide(other)

    def __pow__(self, exponent: Exponent) -> DimensionSignature:
        return self.power(exponent)

    def __hash__(self) -> int:
        return hash(self.terms)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DimensionSignature) and self.terms == other.terms


DIMENSIONLESS = DimensionSignature()
LENGTH = DimensionSignature({"length": 1})
MASS = DimensionSignature({"mass": 1})
TIME = DimensionSignature({"time": 1})
CHARGE = DimensionSignature({"charge": 1})
TEMPERATURE = DimensionSignature({"temperature": 1})
AMOUNT = DimensionSignature({"amount": 1})
ANGLE = DimensionSignature({"angle": 1})

CURRENT = CHARGE / TIME
AREA = LENGTH**2
VOLUME = LENGTH**3
VELOCITY = LENGTH / TIME
ACCELERATION = LENGTH / TIME**2
MOMENTUM = MASS * VELOCITY
ENERGY = MASS * LENGTH**2 / TIME**2
FORCE = ENERGY / LENGTH
PRESSURE = ENERGY / VOLUME
FREQUENCY = DIMENSIONLESS / TIME
VOLTAGE = ENERGY / CHARGE
CAPACITANCE = CHARGE / VOLTAGE
CONDUCTANCE = CURRENT / VOLTAGE
RESISTANCE = VOLTAGE / CURRENT
CONCENTRATION = AMOUNT / VOLUME
