#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isqrt
from typing import TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


_MAX_PRIME = 2_147_483_647


def _is_prime(value: int, /) -> bool:
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    candidate = 5
    step = 2
    limit = isqrt(value)
    while candidate <= limit:
        if value % candidate == 0:
            return False
        candidate += step
        step = 6 - step
    return True


class PrimeField(StrictModule, NonTrainableState):
    """Exact host arithmetic over one explicitly declared prime field."""

    modulus: int = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(self, modulus: int, /):
        value = int(modulus)
        if value > _MAX_PRIME:
            raise ValueError(
                f"Prime-field modulus must not exceed {_MAX_PRIME}; got {value}."
            )
        if not _is_prime(value):
            raise ValueError("Prime-field modulus must be prime.")
        self.modulus = value
        self.field_id = canonical_fingerprint({"kind": "prime-field", "modulus": value})

    def normalize(self, value: int, /) -> int:
        return int(value) % self.modulus

    def add(self, left: int, right: int, /) -> int:
        return (int(left) + int(right)) % self.modulus

    def subtract(self, left: int, right: int, /) -> int:
        return (int(left) - int(right)) % self.modulus

    def multiply(self, left: int, right: int, /) -> int:
        return (int(left) * int(right)) % self.modulus

    def inverse(self, value: int, /) -> int:
        normalized = self.normalize(value)
        if normalized == 0:
            raise ZeroDivisionError("Zero has no multiplicative inverse in a field.")
        return pow(normalized, self.modulus - 2, self.modulus)

    def divide(self, numerator: int, denominator: int, /) -> int:
        return self.multiply(numerator, self.inverse(denominator))


class RationalField(StrictModule, NonTrainableState):
    """Exact rational coefficient marker for rank-only Betti analysis."""

    field_id: str = eqx.field(static=True)

    def __init__(self):
        self.field_id = canonical_fingerprint({"kind": "rational-field"})


CoefficientDomain: TypeAlias = PrimeField | RationalField


__all__ = ["CoefficientDomain", "PrimeField", "RationalField"]
