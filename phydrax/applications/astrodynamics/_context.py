#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import DimensionalScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


AstrodynamicsTimeScale: TypeAlias = Literal[
    "UTC", "TAI", "GPS", "TT", "TCG", "TDB", "TCB", "UT1"
]


AstrodynamicsScaleContract = DimensionalScaleContract


class JulianDate(StrictModule, NonTrainableState):
    """Normalized two-part Julian date without an attached time scale."""

    high: float = eqx.field(static=True)
    low: float = eqx.field(static=True)
    date_id: str = eqx.field(static=True)

    def __init__(self, high: float, low: float = 0.0, /):
        high_ = float(high)
        low_ = float(low)
        if not np.isfinite(high_) or not np.isfinite(low_):
            raise ValueError("Julian-date parts must be finite.")
        carry = float(np.floor(low_ + 0.5))
        normalized_high = high_ + carry
        normalized_low = low_ - carry
        self.high = normalized_high
        self.low = normalized_low
        self.date_id = canonical_fingerprint(
            {
                "kind": "julian-date",
                "high": normalized_high,
                "low": normalized_low,
            }
        )

    def difference_seconds(self, other: JulianDate, /) -> float:
        if not isinstance(other, JulianDate):
            raise TypeError("other must be a JulianDate.")
        return ((self.high - other.high) + (self.low - other.low)) * 86400.0


class TimeInstant(StrictModule, NonTrainableState):
    """One physical instant represented in an explicit astronomical time scale."""

    julian_date: JulianDate
    scale: AstrodynamicsTimeScale = eqx.field(static=True)
    instant_id: str = eqx.field(static=True)

    def __init__(self, julian_date: JulianDate, scale: AstrodynamicsTimeScale, /):
        if not isinstance(julian_date, JulianDate):
            raise TypeError("julian_date must be a JulianDate.")
        scale_ = str(scale).upper()
        if scale_ not in ("UTC", "TAI", "GPS", "TT", "TCG", "TDB", "TCB", "UT1"):
            raise ValueError("Unknown astronomical time scale.")
        self.julian_date = julian_date
        self.scale = scale_  # type: ignore[assignment]
        self.instant_id = canonical_fingerprint(
            {
                "kind": "time-instant",
                "date": julian_date.date_id,
                "scale": scale_,
            }
        )


class ReferenceEpoch(StrictModule, NonTrainableState):
    """Exact origin for a continuous solver or an explicitly noncontinuous model."""

    instant: TimeInstant
    continuous: bool = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(self, instant: TimeInstant, /, *, continuous: bool = True):
        if not isinstance(instant, TimeInstant):
            raise TypeError("instant must be a TimeInstant.")
        if not isinstance(continuous, bool):
            raise TypeError("continuous must be a bool.")
        if continuous and instant.scale == "UTC":
            raise ValueError("UTC cannot be used as a continuous solver epoch.")
        self.instant = instant
        self.continuous = continuous
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-reference-epoch",
                "instant": instant.instant_id,
                "continuous": continuous,
            }
        )

    @property
    def time_scale(self) -> AstrodynamicsTimeScale:
        return self.instant.scale


class FrameDefinition(StrictModule, NonTrainableState):
    """Reference-frame identity for Cartesian astrodynamics states."""

    origin_id: str = eqx.field(static=True)
    orientation_id: str = eqx.field(static=True)
    pseudo_inertial: bool = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        origin_id: str,
        orientation_id: str,
        /,
        *,
        pseudo_inertial: bool,
    ):
        origin = str(origin_id).strip()
        orientation = str(orientation_id).strip()
        if not origin or not orientation:
            raise ValueError("Astrodynamics frame identifiers must be non-empty.")
        if not isinstance(pseudo_inertial, bool):
            raise TypeError("pseudo_inertial must be a bool.")
        self.origin_id = origin
        self.orientation_id = orientation
        self.pseudo_inertial = pseudo_inertial
        self.frame_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-frame",
                "origin": origin,
                "orientation": orientation,
                "pseudo_inertial": pseudo_inertial,
            }
        )


class AstrodynamicsContext(StrictModule, NonTrainableState):
    """Static scale, epoch, and frame contract for one astrodynamics problem."""

    scale: AstrodynamicsScaleContract
    epoch: ReferenceEpoch
    frame: FrameDefinition
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AstrodynamicsScaleContract,
        epoch: ReferenceEpoch,
        frame: FrameDefinition,
        /,
    ):
        if not isinstance(scale, AstrodynamicsScaleContract):
            raise TypeError("scale must be an AstrodynamicsScaleContract.")
        if not isinstance(epoch, ReferenceEpoch):
            raise TypeError("epoch must be a ReferenceEpoch.")
        if not isinstance(frame, FrameDefinition):
            raise TypeError("frame must be a FrameDefinition.")
        self.scale = scale
        self.epoch = epoch
        self.frame = frame
        self.context_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-context",
                "scale": scale.scale_id,
                "epoch": epoch.epoch_id,
                "frame": frame.frame_id,
            }
        )

    def require_compatible(self, other: AstrodynamicsContext, /) -> None:
        if not isinstance(other, AstrodynamicsContext):
            raise TypeError("other must be an AstrodynamicsContext.")
        if self.context_id != other.context_id:
            raise ValueError("Astrodynamics contexts are incompatible.")


__all__ = [
    "AstrodynamicsContext",
    "FrameDefinition",
    "AstrodynamicsScaleContract",
    "AstrodynamicsTimeScale",
    "JulianDate",
    "ReferenceEpoch",
    "TimeInstant",
]
