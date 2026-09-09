#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sample-time and temporal-support contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral

import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..units import conversion_factor, TIME, UnitDefinition
from ._quantity import canonical_quantity_text


def _readonly_real(value: ArrayLike, name: str, /) -> np.ndarray:
    original = np.asarray(value)
    array = np.array(value, dtype=np.result_type(original.dtype, np.float64), copy=True)
    if not np.issubdtype(array.dtype, np.floating) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and real-valued.")
    array.setflags(write=False)
    return array


class TimeBasis(StrEnum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class TemporalSamplingKind(StrEnum):
    INSTANTANEOUS = "instantaneous"
    INTERVAL_MEAN = "interval_mean"
    INTERVAL_INTEGRAL = "interval_integral"
    CUMULATIVE = "cumulative"


@dataclass(frozen=True, slots=True)
class SampleTimeAxis:
    """Strictly ordered sample times with an explicit basis and origin."""

    label: str
    sample_times: np.ndarray
    time_unit: UnitDefinition
    basis: TimeBasis = TimeBasis.RELATIVE
    origin: str | None = None
    time_axis_id: str = field(init=False)

    def __post_init__(self) -> None:
        label = canonical_quantity_text(self.label, "label")
        times = _readonly_real(self.sample_times, "sample_times")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("sample_times must be a non-empty rank-one array.")
        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("sample_times must be strictly increasing.")
        if (
            not isinstance(self.time_unit, UnitDefinition)
            or self.time_unit.dimension != TIME
        ):
            raise ValueError("time_unit must be a time UnitDefinition.")
        if not isinstance(self.basis, TimeBasis):
            raise TypeError("basis must be TimeBasis.")
        origin = (
            None
            if self.origin is None
            else canonical_quantity_text(self.origin, "origin")
        )
        if self.basis is TimeBasis.ABSOLUTE and origin is None:
            raise ValueError("Absolute sample time axes require an origin.")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "sample_times", times)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(
            self,
            "time_axis_id",
            canonical_fingerprint(
                {
                    "kind": "sample-time-axis",
                    "label": label,
                    "samples": array_tree_fingerprint(times),
                    "unit": self.time_unit.unit_id,
                    "basis": self.basis.value,
                    "origin": origin,
                }
            ),
        )

    @classmethod
    def uniform(
        cls,
        label: str,
        sample_count: int,
        interval: float,
        time_unit: UnitDefinition,
        /,
        *,
        origin: float = 0.0,
    ) -> SampleTimeAxis:
        if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
            raise TypeError("sample_count must be an integer.")
        count, width, start = int(sample_count), float(interval), float(origin)
        if count < 1 or not np.isfinite(width) or width <= 0.0 or not np.isfinite(start):
            raise ValueError("sample_count and interval must be positive and finite.")
        return cls(label, start + width * np.arange(count), time_unit)

    @property
    def axis_label(self) -> str:
        return self.label

    @property
    def sample_count(self) -> int:
        return int(self.sample_times.size)

    @property
    def is_uniform(self) -> bool:
        if self.sample_count <= 2:
            return True
        delta = np.diff(self.sample_times)
        tolerance = (
            64.0 * np.finfo(delta.dtype).eps * max(1.0, float(np.max(np.abs(delta))))
        )
        return bool(np.all(np.abs(delta - delta[0]) <= tolerance))

    def values_in(self, unit: UnitDefinition, /) -> np.ndarray:
        return self.sample_times * float(conversion_factor(self.time_unit, unit))

    def interval_in(self, unit: UnitDefinition, /) -> float | None:
        if self.sample_count < 2 or not self.is_uniform:
            return None
        values = self.values_in(unit)
        return float(values[1] - values[0])

    def duration_in(self, unit: UnitDefinition, /) -> float:
        values = self.values_in(unit)
        return float(values[-1] - values[0])


@dataclass(frozen=True, slots=True)
class TemporalSampling:
    """Meaning of each sample over time, independent of sample timestamps."""

    kind: TemporalSamplingKind = TemporalSamplingKind.INSTANTANEOUS
    interval_bounds: np.ndarray | None = None
    interval_unit: UnitDefinition | None = None
    origin: str | None = None
    temporal_sampling_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TemporalSamplingKind):
            raise TypeError("kind must be TemporalSamplingKind.")
        bounds = None
        if self.interval_bounds is not None:
            bounds = _readonly_real(self.interval_bounds, "interval_bounds")
            if bounds.ndim != 2 or bounds.shape[1] != 2:
                raise ValueError("interval_bounds must have shape (sample_count, 2).")
            if np.any(bounds[:, 1] <= bounds[:, 0]):
                raise ValueError("Every temporal interval must have positive width.")
        interval_kind = self.kind in {
            TemporalSamplingKind.INTERVAL_MEAN,
            TemporalSamplingKind.INTERVAL_INTEGRAL,
        }
        if interval_kind and bounds is None:
            raise ValueError("Interval means and integrals require interval_bounds.")
        if not interval_kind and bounds is not None:
            raise ValueError(
                "interval_bounds are only valid for interval means or integrals."
            )
        if interval_kind:
            if (
                not isinstance(self.interval_unit, UnitDefinition)
                or self.interval_unit.dimension != TIME
            ):
                raise ValueError(
                    "interval_unit must be a time unit for interval samples."
                )
        elif self.interval_unit is not None:
            raise ValueError("interval_unit is only valid for interval samples.")
        origin = (
            None
            if self.origin is None
            else canonical_quantity_text(self.origin, "origin")
        )
        if self.kind is TemporalSamplingKind.CUMULATIVE and origin is None:
            raise ValueError("Cumulative temporal sampling requires an origin.")
        object.__setattr__(self, "interval_bounds", bounds)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(
            self,
            "temporal_sampling_id",
            canonical_fingerprint(
                {
                    "kind": "temporal-sampling",
                    "sampling_kind": self.kind.value,
                    "bounds": None if bounds is None else array_tree_fingerprint(bounds),
                    "interval_unit": None
                    if self.interval_unit is None
                    else self.interval_unit.unit_id,
                    "origin": origin,
                }
            ),
        )


__all__ = [
    "SampleTimeAxis",
    "TemporalSampling",
    "TemporalSamplingKind",
    "TimeBasis",
]
