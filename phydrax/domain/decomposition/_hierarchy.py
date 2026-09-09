#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx

from ..._strict import StrictModule
from ._fields import LocalFieldFamily, partition_of_unity_field


class SubdomainLevel(StrictModule):
    """One additive partition-of-unity correction level."""

    family: LocalFieldFamily
    coefficient: float = eqx.field(static=True)
    level_id: str = eqx.field(static=True)

    def __init__(
        self,
        level_id: str,
        family: LocalFieldFamily,
        /,
        *,
        coefficient: float = 1.0,
    ):
        name = str(level_id)
        coefficient_ = float(coefficient)
        if not name:
            raise ValueError("level_id must be non-empty.")
        if not math.isfinite(coefficient_) or coefficient_ == 0.0:
            raise ValueError("coefficient must be finite and nonzero.")
        if not isinstance(family, LocalFieldFamily):
            raise TypeError("family must be a LocalFieldFamily.")
        self.level_id = name
        self.family = family
        self.coefficient = coefficient_

    def field(self):
        return self.coefficient * partition_of_unity_field(self.family)


class SubdomainHierarchy(StrictModule):
    """Ordered additive local correction levels over one ambient domain."""

    levels: tuple[SubdomainLevel, ...]
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        levels: Sequence[SubdomainLevel],
        /,
        *,
        hierarchy_id: str,
    ):
        levels_ = tuple(levels)
        if not levels_ or any(not isinstance(level, SubdomainLevel) for level in levels_):
            raise TypeError(
                "levels must be a non-empty sequence of SubdomainLevel values."
            )
        identities = tuple(level.level_id for level in levels_)
        if len(set(identities)) != len(identities):
            raise ValueError("Subdomain hierarchy level IDs must be unique.")
        ambient = levels_[0].family.cover.ambient
        for level in levels_[1:]:
            if not level.family.cover.ambient.same_support(ambient):
                raise ValueError("Every hierarchy level must share one ambient domain.")
        identifier = str(hierarchy_id)
        if not identifier:
            raise ValueError("hierarchy_id must be non-empty.")
        self.levels = levels_
        self.hierarchy_id = identifier

    @property
    def ambient(self):
        return self.levels[0].family.cover.ambient

    def level(self, level_id: str, /) -> SubdomainLevel:
        for level in self.levels:
            if level.level_id == level_id:
                return level
        raise KeyError(f"Unknown subdomain hierarchy level {level_id!r}.")

    def correction(self):
        result = self.levels[0].field()
        for level in self.levels[1:]:
            result = result + level.field()
        return result.with_metadata(
            hierarchy_id=self.hierarchy_id,
            hierarchy_levels=tuple(level.level_id for level in self.levels),
        )


__all__ = ["SubdomainHierarchy", "SubdomainLevel"]
