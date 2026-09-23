#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical finite products of integral and modular Abelian charges."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


AbelianCharge: TypeAlias = tuple[int, ...]


class AbelianGroup(StrictModule):
    """Finite direct product of U(1) and cyclic Abelian charge components."""

    components: tuple[int | None, ...] = eqx.field(static=True)
    group_id: str = eqx.field(static=True)

    def __init__(self, components: Sequence[int | None], /):
        values = tuple(components)
        if not values:
            raise ValueError("An Abelian group requires at least one component.")
        normalized = []
        for component in values:
            if component is None:
                normalized.append(None)
            else:
                if isinstance(component, bool) or int(component) < 2:
                    raise ValueError(
                        "Cyclic Abelian moduli must be integers at least two."
                    )
                normalized.append(int(component))
        self.components = tuple(normalized)
        self.group_id = canonical_fingerprint(
            {"kind": "abelian-group", "components": self.components}
        )

    @property
    def zero(self) -> AbelianCharge:
        return (0,) * len(self.components)

    def normalize(self, charge: Sequence[int], /) -> AbelianCharge:
        values = tuple(charge)
        if len(values) != len(self.components):
            raise ValueError("Charge component count does not match the Abelian group.")
        output = []
        for value, modulus in zip(values, self.components, strict=True):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Abelian charges must contain integers.")
            output.append(value if modulus is None else value % modulus)
        return tuple(output)

    def add(self, *charges: Sequence[int]) -> AbelianCharge:
        total = [0] * len(self.components)
        for charge in charges:
            values = self.normalize(charge)
            total = [left + right for left, right in zip(total, values, strict=True)]
        return self.normalize(total)

    def negate(self, charge: Sequence[int], /) -> AbelianCharge:
        return self.normalize(tuple(-value for value in self.normalize(charge)))

    def subtract(self, left: Sequence[int], right: Sequence[int], /) -> AbelianCharge:
        return self.add(left, self.negate(right))


__all__ = ["AbelianCharge", "AbelianGroup"]
