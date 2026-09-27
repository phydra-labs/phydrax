#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx

from ..._numerics import normalize_anisotropy
from ..._strict import StrictModule
from ...typing import parse


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
    ) -> None:
        dimension_ = int(dimension)
        level_ = int(level)
        if dimension_ < 1 or level_ < 1:
            raise ValueError(
                "Smolyak interpolation dimension and level must be positive."
            )
        raw_rules: tuple[object, ...]
        if axis_rules is None:
            raw_rules = ("auto",) * dimension_
        elif isinstance(axis_rules, str):
            raw_rules = (axis_rules,) * dimension_
        else:
            raw_rules = tuple(axis_rules)
        if len(raw_rules) != dimension_:
            raise ValueError("axis_rules must contain one rule per dimension.")
        rules = tuple(
            parse(rule, SmolyakInterpolationRule, "axis_rules") for rule in raw_rules
        )
        self.dimension = dimension_
        self.level = level_
        self.anisotropy = normalize_anisotropy(dimension_, anisotropy)
        self.axis_rules = rules


class AdaptiveSmolyakInterpolationPlan(StrictModule):
    """Eager dimension-adaptive Smolyak interpolation preparation."""

    dimension: int = eqx.field(static=True)
    initial_level: int = eqx.field(static=True)
    anisotropy: tuple[float, ...] = eqx.field(static=True)
    axis_rules: tuple[SmolyakInterpolationRule, ...] = eqx.field(static=True)
    indicator_norm: Literal["max", "weighted-l2"] = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    max_indices: int = eqx.field(static=True)
    max_nodes: int = eqx.field(static=True)
    max_rounds: int = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        initial_level: int = 1,
        /,
        *,
        anisotropy: Sequence[float] | None = None,
        axis_rules: SmolyakInterpolationRule
        | Sequence[SmolyakInterpolationRule] = "auto",
        indicator_norm: Literal["max", "weighted-l2"] = "max",
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 1.0e-6,
        max_indices: int = 64,
        max_nodes: int = 100_000,
        max_rounds: int = 32,
    ) -> None:
        fixed = SmolyakInterpolationPlan(
            dimension,
            initial_level,
            anisotropy=anisotropy,
            axis_rules=axis_rules,
        )
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if absolute < 0.0 or relative < 0.0:
            raise ValueError("Adaptive interpolation tolerances must be nonnegative.")
        if indicator_norm not in ("max", "weighted-l2"):
            raise ValueError("indicator_norm must be 'max' or 'weighted-l2'.")
        capacities = (int(max_indices), int(max_nodes), int(max_rounds))
        if any(value < 1 for value in capacities):
            raise ValueError("Adaptive interpolation capacities must be positive.")
        self.dimension = fixed.dimension
        self.initial_level = fixed.level
        self.anisotropy = fixed.anisotropy
        self.axis_rules = fixed.axis_rules
        self.indicator_norm = indicator_norm
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.max_indices, self.max_nodes, self.max_rounds = capacities


__all__ = [
    "AdaptiveSmolyakInterpolationPlan",
    "SmolyakInterpolationPlan",
    "SmolyakInterpolationRule",
]
