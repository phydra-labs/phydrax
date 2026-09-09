#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._resource import ResourceManifest


TimeScale = Literal["tai", "gps", "utc", "instrument", "source-relative"]


class LeapSecondTable(StrictModule, NonTrainableState):
    """Pinned UTC transition table in nominal non-leap-counting seconds.

    ``transition_utc_seconds`` and ``tai_minus_utc_after`` are absolute values
    relative to a caller-declared canonical origin. The table does not download,
    extrapolate, or infer future leap seconds.
    """

    transition_utc_seconds: Array
    tai_minus_utc_after: Array
    initial_tai_minus_utc: float = eqx.field(static=True)
    source: ResourceManifest
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition_utc_seconds: ArrayLike,
        tai_minus_utc_after: ArrayLike,
        initial_tai_minus_utc: float,
        source: ResourceManifest,
        /,
    ):
        transitions = np.asarray(transition_utc_seconds, dtype=float)
        offsets = np.asarray(tai_minus_utc_after, dtype=float)
        initial = float(initial_tai_minus_utc)
        if transitions.ndim != 1 or offsets.shape != transitions.shape:
            raise ValueError("Leap transitions and offsets must be matching vectors.")
        if (
            np.any(~np.isfinite(transitions))
            or np.any(~np.isfinite(offsets))
            or not np.isfinite(initial)
            or np.any(np.diff(transitions) <= 0)
            or np.any(np.diff(np.concatenate(([initial], offsets))) < 0)
        ):
            raise ValueError(
                "Leap-second transitions and offsets must be finite and ordered."
            )
        if not isinstance(source, ResourceManifest):
            raise TypeError("Leap-second table requires an exact source manifest.")
        self.transition_utc_seconds = jnp.asarray(transitions)
        self.tai_minus_utc_after = jnp.asarray(offsets)
        self.initial_tai_minus_utc = initial
        self.source = source
        self.table_id = canonical_fingerprint(
            {
                "kind": "leap-second-table",
                "transitions": transitions,
                "offsets": offsets,
                "initial": initial,
                "source": source.manifest_id,
            }
        )

    def offset_for_utc(self, nominal_utc_seconds: ArrayLike) -> Array:
        values = jnp.asarray(nominal_utc_seconds)
        if self.transition_utc_seconds.size == 0:
            return jnp.full_like(values, self.initial_tai_minus_utc)
        indices = jnp.searchsorted(self.transition_utc_seconds, values, side="right") - 1
        selected = self.tai_minus_utc_after[jnp.maximum(indices, 0)]
        return jnp.where(indices >= 0, selected, self.initial_tai_minus_utc)

    def offset_for_tai(self, tai_seconds: ArrayLike) -> Array:
        values = jnp.asarray(tai_seconds)
        if self.transition_utc_seconds.size == 0:
            return jnp.full_like(values, self.initial_tai_minus_utc)
        transition_tai = self.transition_utc_seconds + self.tai_minus_utc_after
        indices = jnp.searchsorted(transition_tai, values, side="right") - 1
        selected = self.tai_minus_utc_after[jnp.maximum(indices, 0)]
        return jnp.where(indices >= 0, selected, self.initial_tai_minus_utc)

    def tai_in_positive_leap(self, tai_seconds: ArrayLike) -> Array:
        values = jnp.asarray(tai_seconds)
        if self.transition_utc_seconds.size == 0:
            return jnp.zeros_like(values, dtype=bool)
        previous = jnp.concatenate(
            (
                jnp.asarray([self.initial_tai_minus_utc]),
                self.tai_minus_utc_after[:-1],
            )
        )
        starts = self.transition_utc_seconds + previous
        ends = self.transition_utc_seconds + self.tai_minus_utc_after
        return jnp.any(
            (values[..., None] >= starts) & (values[..., None] < ends), axis=-1
        )


class TimeReferenceContract(StrictModule, NonTrainableState):
    """One continuous sampled clock tied explicitly to absolute TAI.

    Recorded values obey ``recorded = ideal * (1 + drift) + offset``. Conversion
    first removes the declared offset/drift, then maps the ideal source scale to
    TAI. UTC uses a pinned leap table; TAI, GPS, instrument, and source-relative
    clocks use the exact TAI instant of their epoch.
    """

    scale: TimeScale = eqx.field(static=True)
    epoch_label: str = eqx.field(static=True)
    epoch_tai_seconds: float = eqx.field(static=True)
    epoch_nominal_seconds: float | None = eqx.field(static=True)
    clock_offset_seconds: float = eqx.field(static=True)
    drift_fraction: float = eqx.field(static=True)
    leap_seconds: LeapSecondTable | None
    correction_resources: tuple[ResourceManifest, ...] = eqx.field(static=True)
    time_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: TimeScale,
        epoch_label: str,
        epoch_tai_seconds: float,
        /,
        *,
        epoch_nominal_seconds: float | None = None,
        clock_offset_seconds: float = 0.0,
        drift_ppm: float = 0.0,
        leap_seconds: LeapSecondTable | None = None,
        correction_resources: Sequence[ResourceManifest] = (),
    ):
        if scale not in ("tai", "gps", "utc", "instrument", "source-relative"):
            raise ValueError("Unsupported time scale.")
        label = str(epoch_label).strip()
        tai_epoch = float(epoch_tai_seconds)
        nominal = None if epoch_nominal_seconds is None else float(epoch_nominal_seconds)
        offset = float(clock_offset_seconds)
        drift = float(drift_ppm) * 1e-6
        resources = tuple(correction_resources)
        if (
            not label
            or not np.isfinite(tai_epoch)
            or (nominal is not None and not np.isfinite(nominal))
            or not np.isfinite(offset)
            or not np.isfinite(drift)
            or 1.0 + drift <= 0
        ):
            raise ValueError(
                "Time reference values must be finite and clock rate positive."
            )
        if scale == "utc":
            if nominal is None or not isinstance(leap_seconds, LeapSecondTable):
                raise ValueError(
                    "UTC requires nominal epoch seconds and a pinned leap-second table."
                )
            transitions = np.asarray(leap_seconds.transition_utc_seconds)
            offsets = np.asarray(leap_seconds.tai_minus_utc_after)
            index = int(np.searchsorted(transitions, nominal, side="right") - 1)
            epoch_offset = (
                leap_seconds.initial_tai_minus_utc if index < 0 else float(offsets[index])
            )
            if not np.isclose(tai_epoch, nominal + epoch_offset, rtol=0.0, atol=1e-9):
                raise ValueError(
                    "UTC nominal and TAI epoch values disagree with the leap table."
                )
        elif leap_seconds is not None:
            raise ValueError("Leap-second tables belong only to UTC contracts.")
        if any(not isinstance(item, ResourceManifest) for item in resources):
            raise TypeError("Clock correction resources must be ResourceManifest values.")
        self.scale = scale
        self.epoch_label = label
        self.epoch_tai_seconds = tai_epoch
        self.epoch_nominal_seconds = nominal
        self.clock_offset_seconds = offset
        self.drift_fraction = drift
        self.leap_seconds = leap_seconds
        self.correction_resources = resources
        self.time_id = canonical_fingerprint(
            {
                "kind": "time-reference-contract",
                "scale": scale,
                "epoch_label": label,
                "epoch_tai_seconds": tai_epoch,
                "epoch_nominal_seconds": nominal,
                "clock_offset_seconds": offset,
                "drift_fraction": drift,
                "leap_seconds": None if leap_seconds is None else leap_seconds.table_id,
                "resources": [item.manifest_id for item in resources],
            }
        )

    def ideal_seconds(self, recorded_seconds: ArrayLike) -> Array:
        values = jnp.asarray(recorded_seconds)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Recorded clock values must be finite.",
        )
        return (values - self.clock_offset_seconds) / (1.0 + self.drift_fraction)

    def to_tai(self, recorded_seconds: ArrayLike) -> Array:
        ideal = self.ideal_seconds(recorded_seconds)
        if self.scale != "utc":
            return self.epoch_tai_seconds + ideal
        if self.epoch_nominal_seconds is None or self.leap_seconds is None:
            raise RuntimeError("UTC contract lost its required epoch or leap table.")
        nominal = self.epoch_nominal_seconds + ideal
        return nominal + self.leap_seconds.offset_for_utc(nominal)

    def from_tai(self, tai_seconds: ArrayLike) -> Array:
        values = jnp.asarray(tai_seconds)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "TAI values must be finite.",
        )
        if self.scale == "utc":
            if self.epoch_nominal_seconds is None or self.leap_seconds is None:
                raise RuntimeError("UTC contract lost its required epoch or leap table.")
            values = eqx.error_if(
                values,
                self.leap_seconds.tai_in_positive_leap(values),
                "TAI instant lies inside a positive leap second that nominal UTC cannot represent.",
            )
            nominal = values - self.leap_seconds.offset_for_tai(values)
            ideal = nominal - self.epoch_nominal_seconds
        else:
            ideal = values - self.epoch_tai_seconds
        return ideal * (1.0 + self.drift_fraction) + self.clock_offset_seconds


class TimeTransform(StrictModule, NonTrainableState):
    source_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    resources: tuple[ResourceManifest, ...] = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(self, source: TimeReferenceContract, target: TimeReferenceContract, /):
        if not isinstance(source, TimeReferenceContract) or not isinstance(
            target, TimeReferenceContract
        ):
            raise TypeError("Time transform endpoints must be time contracts.")
        resources = tuple(
            sorted(
                (*source.correction_resources, *target.correction_resources),
                key=lambda item: item.manifest_id,
            )
        )
        leap_resources = tuple(
            item.source
            for item in (source.leap_seconds, target.leap_seconds)
            if item is not None
        )
        resources = tuple(
            {item.manifest_id: item for item in (*resources, *leap_resources)}.values()
        )
        self.source_id, self.target_id = source.time_id, target.time_id
        self.resources = resources
        self.transform_id = canonical_fingerprint(
            {
                "kind": "time-transform",
                "source": source.time_id,
                "target": target.time_id,
                "resources": [item.manifest_id for item in resources],
            }
        )


def convert_time(
    values: ArrayLike,
    source: TimeReferenceContract,
    target: TimeReferenceContract,
    /,
) -> tuple[Array, TimeTransform]:
    if not isinstance(source, TimeReferenceContract) or not isinstance(
        target, TimeReferenceContract
    ):
        raise TypeError("convert_time requires source and target time contracts.")
    return target.from_tai(source.to_tai(values)), TimeTransform(source, target)


__all__ = [
    "LeapSecondTable",
    "TimeReferenceContract",
    "TimeScale",
    "TimeTransform",
    "convert_time",
]
