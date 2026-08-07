#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

from .._frozendict import frozendict
from .._strict import StrictModule
from ._domain import JointFactor
from ._measure import BaseMeasure
from ._selection import Selection


class FactorComponent(StrictModule):
    """A joint factor with validated coordinate selections and base measure."""

    factor: JointFactor
    selections: frozendict[str, Selection]
    measure: BaseMeasure

    def __init__(
        self,
        *,
        factor: JointFactor,
        selections: Mapping[str, Selection],
        measure: BaseMeasure,
    ):
        if not isinstance(factor, JointFactor):
            raise TypeError("FactorComponent.factor must be a JointFactor.")
        resolved = dict(selections)
        if tuple(resolved) != factor.labels:
            raise ValueError(
                "FactorComponent selections must contain every factor label in order; "
                f"expected {factor.labels}, got {tuple(resolved)}."
            )
        if any(not isinstance(selection, Selection) for selection in resolved.values()):
            raise TypeError("FactorComponent selections must be Selection values.")
        if not isinstance(measure, BaseMeasure):
            raise TypeError("FactorComponent.measure must be a BaseMeasure.")
        self.factor = factor
        self.selections = frozendict(resolved)
        self.measure = measure


__all__ = ["FactorComponent"]
