#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit mappings between acquisition clock domains."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import conversion_factor, TIME, UnitDefinition
from ._quantity import canonical_quantity_text


@dataclass(frozen=True, slots=True)
class ClockIdentity:
    clock_id: str
    time_unit: UnitDefinition
    basis: str
    origin: str | None = None
    identity_id: str = field(init=False)

    def __post_init__(self) -> None:
        clock = canonical_quantity_text(self.clock_id, "clock_id")
        if (
            not isinstance(self.time_unit, UnitDefinition)
            or self.time_unit.dimension != TIME
        ):
            raise ValueError("time_unit must be a time UnitDefinition.")
        basis = canonical_quantity_text(self.basis, "basis")
        origin = (
            None
            if self.origin is None
            else canonical_quantity_text(self.origin, "origin")
        )
        if basis == "absolute" and origin is None:
            raise ValueError("Absolute clocks require an origin.")
        object.__setattr__(self, "clock_id", clock)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(
            self,
            "identity_id",
            canonical_fingerprint(
                {
                    "kind": "clock-identity",
                    "clock": clock,
                    "unit": self.time_unit.unit_id,
                    "basis": basis,
                    "origin": origin,
                }
            ),
        )


class ClockMappingEvidence(StrictModule, NonTrainableState):
    in_domain: Array
    finite: Array
    extrapolated: Array
    maximum_fit_residual: Array
    successful: Array
    mapping_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class AffineClockMap:
    source: ClockIdentity
    target: ClockIdentity
    offset: float
    rate: float
    calibration_id: str
    valid_interval_source: tuple[float, float]
    maximum_fit_residual: float = 0.0
    mapping_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, ClockIdentity) or not isinstance(
            self.target, ClockIdentity
        ):
            raise TypeError("source and target must be ClockIdentity values.")
        if self.source.clock_id == self.target.clock_id:
            raise ValueError("Clock maps require distinct domains.")
        offset, rate = float(self.offset), float(self.rate)
        interval = tuple(float(value) for value in self.valid_interval_source)
        residual = float(self.maximum_fit_residual)
        if (
            not np.isfinite(offset)
            or not np.isfinite(rate)
            or rate <= 0.0
            or len(interval) != 2
            or not np.all(np.isfinite(interval))
            or interval[1] <= interval[0]
            or not np.isfinite(residual)
            or residual < 0.0
        ):
            raise ValueError(
                "Affine clock parameters must be finite, monotone, and bounded."
            )
        calibration = canonical_quantity_text(self.calibration_id, "calibration_id")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "valid_interval_source", interval)
        object.__setattr__(self, "maximum_fit_residual", residual)
        object.__setattr__(self, "calibration_id", calibration)
        object.__setattr__(
            self,
            "mapping_id",
            canonical_fingerprint(
                {
                    "kind": "affine-clock-map",
                    "source": self.source.identity_id,
                    "target": self.target.identity_id,
                    "offset": offset,
                    "rate": rate,
                    "interval": list(interval),
                    "residual": residual,
                    "calibration": calibration,
                }
            ),
        )

    def prepare(self) -> PreparedClockMap:
        source_to_target = float(
            conversion_factor(self.source.time_unit, self.target.time_unit)
        )
        return PreparedClockMap(
            jnp.asarray((self.valid_interval_source[0], self.valid_interval_source[1])),
            jnp.asarray((self.offset,)),
            jnp.asarray((self.rate * source_to_target,)),
            jnp.asarray((self.maximum_fit_residual,)),
            self.mapping_id,
        )


@dataclass(frozen=True, slots=True)
class PiecewiseClockMap:
    source: ClockIdentity
    target: ClockIdentity
    source_knots: np.ndarray
    target_knots: np.ndarray
    calibration_id: str
    maximum_fit_residual: float = 0.0
    mapping_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, ClockIdentity) or not isinstance(
            self.target, ClockIdentity
        ):
            raise TypeError("source and target must be ClockIdentity values.")
        source = np.array(self.source_knots, dtype=float, copy=True)
        target = np.array(self.target_knots, dtype=float, copy=True)
        if (
            source.ndim != 1
            or source.shape != target.shape
            or source.size < 2
            or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(target))
            or not np.all(np.diff(source) > 0.0)
            or not np.all(np.diff(target) > 0.0)
        ):
            raise ValueError(
                "Clock knots must be finite, equally sized, and strictly increasing."
            )
        target = target * float(
            conversion_factor(self.target.time_unit, self.source.time_unit)
        )
        residual = float(self.maximum_fit_residual)
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("maximum_fit_residual must be finite and non-negative.")
        calibration = canonical_quantity_text(self.calibration_id, "calibration_id")
        source.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "source_knots", source)
        object.__setattr__(self, "target_knots", target)
        object.__setattr__(self, "maximum_fit_residual", residual)
        object.__setattr__(self, "calibration_id", calibration)
        object.__setattr__(
            self,
            "mapping_id",
            canonical_fingerprint(
                {
                    "kind": "piecewise-clock-map",
                    "source": self.source.identity_id,
                    "target": self.target.identity_id,
                    "source_knots": array_tree_fingerprint(source),
                    "target_knots_in_source_unit": array_tree_fingerprint(target),
                    "residual": residual,
                    "calibration": calibration,
                }
            ),
        )

    def prepare(self) -> PreparedClockMap:
        slopes = np.diff(self.target_knots) / np.diff(self.source_knots)
        offsets = self.target_knots[:-1] - slopes * self.source_knots[:-1]
        target_factor = float(
            conversion_factor(self.source.time_unit, self.target.time_unit)
        )
        return PreparedClockMap(
            jnp.asarray(self.source_knots),
            jnp.asarray(offsets * target_factor),
            jnp.asarray(slopes * target_factor),
            jnp.full((slopes.size,), self.maximum_fit_residual * target_factor),
            self.mapping_id,
        )


class PreparedClockMap(StrictModule, NonTrainableState):
    source_knots: Array
    offsets: Array
    rates: Array
    fit_residuals: Array
    mapping_id: str = eqx.field(static=True)

    def map(
        self, source_times: ArrayLike, /, *, allow_extrapolation: bool = False
    ) -> tuple[Array, ClockMappingEvidence]:
        times = jnp.asarray(source_times)
        lower, upper = self.source_knots[0], self.source_knots[-1]
        in_domain = (times >= lower) & (times <= upper)
        interval_count = self.rates.shape[0]
        if interval_count == 1:
            indices = jnp.zeros(times.shape, dtype=jnp.int32)
        else:
            indices = jnp.clip(
                jnp.searchsorted(self.source_knots[1:-1], times, side="right"),
                0,
                interval_count - 1,
            )
        mapped = self.offsets[indices] + self.rates[indices] * times
        finite = jnp.all(jnp.isfinite(mapped))
        extrapolated = ~in_domain
        accepted = jnp.asarray(bool(allow_extrapolation)) | in_domain
        output = jnp.where(accepted, mapped, jnp.nan)
        residual = jnp.max(self.fit_residuals[indices])
        successful = finite & jnp.all(accepted)
        return output, ClockMappingEvidence(
            in_domain,
            finite,
            extrapolated,
            residual,
            successful,
            self.mapping_id,
        )


__all__ = [
    "AffineClockMap",
    "ClockIdentity",
    "ClockMappingEvidence",
    "PiecewiseClockMap",
    "PreparedClockMap",
]
