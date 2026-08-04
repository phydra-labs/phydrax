#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, Literal, TypeAlias

import equinox as eqx

from ..._numerics import normalize_anisotropy
from ..._strict import StrictModule


SmolyakInterpolationRule: TypeAlias = Literal[
    "auto",
    "leja",
    "clenshaw-curtis",
    "gauss-hermite",
]


class SmolyakInterpolationPlan(StrictModule):
    """Static weighted-index and per-axis rule specification for interpolation."""

    dimension: int = eqx.field(static=True)
    level: int = eqx.field(static=True)
    anisotropy: tuple[float, ...] = eqx.field(static=True)
    axis_rules: tuple[SmolyakInterpolationRule, ...] = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        level: int,
        /,
        *,
        anisotropy: Sequence[float] | None = None,
        axis_rules: (
            SmolyakInterpolationRule | Sequence[SmolyakInterpolationRule] | None
        ) = "auto",
    ):
        dimension_ = int(dimension)
        level_ = int(level)
        if dimension_ < 1 or level_ < 1:
            raise ValueError(
                "Smolyak interpolation dimension and level must be positive."
            )
        if axis_rules is None:
            rules: tuple[str, ...] = ("auto",) * dimension_
        elif isinstance(axis_rules, str):
            rules = (axis_rules,) * dimension_
        else:
            rules = tuple(str(rule) for rule in axis_rules)
        if len(rules) != dimension_:
            raise ValueError("axis_rules must contain one rule per dimension.")
        allowed = ("auto", "leja", "clenshaw-curtis", "gauss-hermite")
        invalid = tuple(rule for rule in rules if rule not in allowed)
        if invalid:
            choices = ", ".join(repr(rule) for rule in allowed)
            raise ValueError(
                f"Unsupported interpolation axis rule {invalid[0]!r}; expected {choices}."
            )
        self.dimension = dimension_
        self.level = level_
        self.anisotropy = normalize_anisotropy(dimension_, anisotropy)
        self.axis_rules = cast(tuple[SmolyakInterpolationRule, ...], rules)


__all__ = ["SmolyakInterpolationPlan", "SmolyakInterpolationRule"]
