#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index

import equinox as eqx

from ._core import AbstractFiniteRealAlgebraSpec
from ._resources import AlgebraResourceBudget


class MulticomplexAlgebraSpec(AbstractFiniteRealAlgebraSpec):
    """Commutative associative algebra with independent commuting imaginary units."""

    rank: int = eqx.field(static=True)

    def __init__(
        self,
        rank: int,
        /,
        *,
        budget: AlgebraResourceBudget | None = None,
    ):
        if isinstance(rank, bool):
            raise TypeError("Multicomplex rank must be an integer.")
        rank_ = index(rank)
        if rank_ < 1:
            raise ValueError("Multicomplex rank must be positive.")
        dimension = 1 << rank_
        budget_ = AlgebraResourceBudget() if budget is None else budget
        budget_.admit_coordinates(dimension)
        labels = ("1",) + tuple(
            "*".join(f"i{axis + 1}" for axis in range(rank_) if bitmap & (1 << axis))
            for bitmap in range(1, dimension)
        )
        terms = []
        for left in range(dimension):
            for right in range(dimension):
                repeated = left & right
                sign = -1 if repeated.bit_count() % 2 else 1
                terms.append((left, right, left ^ right, sign, 1))
        conjugation = tuple(
            tuple(
                (-1 if row.bit_count() % 2 else 1) if row == column else 0
                for column in range(dimension)
            )
            for row in range(dimension)
        )
        proven = ("proven", "family_construction", ())
        claims = {
            "commutative": proven,
            "associative": proven,
            "alternative": proven,
            "left_alternative": proven,
            "right_alternative": proven,
            "flexible": proven,
            "power_associative": proven,
            "division_algebra": proven
            if rank_ == 1
            else (
                "disproven",
                "explicit_witness",
                ("multicomplex-zero-divisor",),
            ),
            "has_zero_divisors": (
                "disproven",
                "family_construction",
                ("complex-division",),
            )
            if rank_ == 1
            else (
                "proven",
                "explicit_witness",
                ("multicomplex-zero-divisor",),
            ),
            "positive_norm": proven
            if rank_ == 1
            else (
                "unknown",
                "unavailable",
                (),
            ),
            "norm_multiplicative": proven
            if rank_ == 1
            else (
                "disproven",
                "family_construction",
                ("rank>=2",),
            ),
        }
        AbstractFiniteRealAlgebraSpec.__init__(
            self,
            f"multicomplex-{rank_}",
            labels,
            terms,
            (1,) + (0,) * (dimension - 1),
            conjugation,
            convention={
                "kind": "multicomplex-commuting-v1",
                "rank": rank_,
                "generator_square": -1,
            },
            family_claims=claims,
            budget=budget_,
        )
        self.rank = rank_

    def _family_marker(self) -> str:
        return self.family


__all__ = ["MulticomplexAlgebraSpec"]
