#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Closed validity ranges of individual material-property correlations.

A range states where one published correlation was fitted or qualified. It is
independent of any application-level model envelope (for example battery
equation envelopes), which aggregates geometry, protocol, and replay policy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import parse


ValidityExceedance: TypeAlias = Literal["refuse", "terminal-domain-exit"]


@final
class PropertyValidityRange(StrictModule, NonTrainableState):
    """Per-variable closed intervals, SI units, and the correlation source.

    ``contains`` evaluates on device and returns the elementwise conjunction of
    every declared interval; every declared variable must be supplied, so a
    caller cannot silently skip a coordinate. ``exceedance`` records how an
    owning workflow must treat a violation; the range itself never clips or
    extrapolates.
    """

    variables: tuple[str, ...] = eqx.field(static=True)
    lower: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    units: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    exceedance: ValidityExceedance = eqx.field(static=True)
    range_id: str = eqx.field(static=True)

    def __init__(
        self,
        bounds: Mapping[str, tuple[float, float, str]],
        /,
        *,
        source_id: str,
        exceedance: ValidityExceedance = "terminal-domain-exit",
    ) -> None:
        if not isinstance(bounds, Mapping) or not bounds:
            raise TypeError("bounds must be a non-empty mapping.")
        if not isinstance(source_id, str):
            raise TypeError("source_id must be a string.")
        source = source_id
        if not source:
            raise ValueError("source_id must be non-empty.")
        policy = parse(exceedance, ValidityExceedance, "exceedance")
        if any(not isinstance(name, str) for name in bounds):
            raise TypeError("Validity variable names must be strings.")
        names = tuple(sorted(bounds))
        if any(not name for name in names):
            raise ValueError("Validity variable names must be non-empty.")
        lower: list[float] = []
        upper: list[float] = []
        units: list[str] = []
        for name in names:
            entry = bounds[name]
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise TypeError(f"Validity bound {name!r} must be (lower, upper, unit).")
            low, high, unit = float(entry[0]), float(entry[1]), str(entry[2])
            if (
                math.isnan(low)
                or math.isnan(high)
                or low == math.inf
                or high == -math.inf
                or not low <= high
                or not unit
            ):
                raise ValueError(
                    f"Validity bound {name!r} requires ordered limits and a unit."
                )
            lower.append(low)
            upper.append(high)
            units.append(unit)
        self.variables = names
        self.lower = tuple(lower)
        self.upper = tuple(upper)
        self.units = tuple(units)
        self.source_id = source
        self.exceedance = policy
        self.range_id = canonical_fingerprint(
            {
                "kind": "property-validity-range",
                "variables": list(names),
                "lower": [value if math.isfinite(value) else None for value in lower],
                "upper": [value if math.isfinite(value) else None for value in upper],
                "units": units,
                "source": source,
                "exceedance": policy,
            }
        )

    def contains(self, **values: ArrayLike) -> Array:
        """Elementwise membership of broadcast-compatible variable arrays."""

        supplied = tuple(sorted(values))
        if supplied != self.variables:
            raise ValueError(
                "Validity evaluation requires exactly the declared variables "
                f"{self.variables}; received {supplied}."
            )
        inside = jnp.asarray(True)
        for name, low, high in zip(self.variables, self.lower, self.upper, strict=True):
            value = jnp.asarray(values[name])
            inside = inside & jnp.isfinite(value) & (value >= low) & (value <= high)
        return inside

    def bound(self, variable: str, /) -> tuple[float, float, str]:
        """Declared ``(lower, upper, unit)`` of one variable."""

        name = str(variable)
        if name not in self.variables:
            raise KeyError(f"{name!r} is not a declared validity variable.")
        index = self.variables.index(name)
        return self.lower[index], self.upper[index], self.units[index]


__all__ = ["PropertyValidityRange", "ValidityExceedance"]
