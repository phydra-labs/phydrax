#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import AstrodynamicsTimeScale, JulianDate, TimeInstant
from ._data import AstrodynamicsDataProvenance
from ._status import AstrodynamicsStatus


TimeScaleName: TypeAlias = AstrodynamicsTimeScale
TimeInterpolation: TypeAlias = Literal["constant", "linear", "step"]

_LG = 6.969290134e-10
_LB = 1.550519768e-8
_TDB0 = -6.55e-5
_REFERENCE_JD = 2443144.5003725


class TimeScaleTransformResult(StrictModule):
    relative_seconds: Array
    offset_seconds: Array
    valid: Array
    status: Array
    transform_id: str = eqx.field(static=True)


class TimeScaleTransform(StrictModule, NonTrainableState):
    """Prepared offset table mapping relative seconds between named time scales."""

    nodes: Array
    offsets: Array
    provenance: AstrodynamicsDataProvenance
    source_scale: TimeScaleName = eqx.field(static=True)
    target_scale: TimeScaleName = eqx.field(static=True)
    interpolation: TimeInterpolation = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_scale: TimeScaleName,
        target_scale: TimeScaleName,
        nodes: ArrayLike,
        offsets: ArrayLike,
        provenance: AstrodynamicsDataProvenance,
        /,
        *,
        interpolation: TimeInterpolation,
    ):
        source = str(source_scale).upper()
        target = str(target_scale).upper()
        supported = ("UTC", "TAI", "GPS", "TT", "TCG", "TDB", "TCB", "UT1")
        if source not in supported or target not in supported:
            raise ValueError("Unknown astrodynamics time scale.")
        if source == target:
            raise ValueError("Time-scale transform endpoints must differ.")
        if interpolation not in ("constant", "linear", "step"):
            raise ValueError("Unknown time offset interpolation policy.")
        if not isinstance(provenance, AstrodynamicsDataProvenance):
            raise TypeError("provenance must be AstrodynamicsDataProvenance.")
        nodes_host = np.asarray(nodes, dtype=float)
        offsets_host = np.asarray(offsets, dtype=float)
        if (
            nodes_host.ndim != 1
            or nodes_host.size == 0
            or offsets_host.shape != nodes_host.shape
            or np.any(~np.isfinite(nodes_host))
            or np.any(~np.isfinite(offsets_host))
            or np.any(np.diff(nodes_host) <= 0.0)
        ):
            raise ValueError(
                "Time transform nodes/offsets must be finite monotone vectors."
            )
        if interpolation == "constant" and nodes_host.size != 1:
            raise ValueError("A constant time transform requires one node and offset.")
        self.nodes = jnp.asarray(nodes_host)
        self.offsets = jnp.asarray(offsets_host)
        self.provenance = provenance
        self.source_scale = source  # type: ignore[assignment]
        self.target_scale = target  # type: ignore[assignment]
        self.interpolation = interpolation
        self.transform_id = canonical_fingerprint(
            {
                "kind": "time-scale-transform",
                "source": source,
                "target": target,
                "interpolation": interpolation,
                "nodes": nodes_host.tolist(),
                "offsets": offsets_host.tolist(),
                "provenance": provenance.provenance_id,
            }
        )

    def apply(self, relative_seconds: ArrayLike, /) -> TimeScaleTransformResult:
        query = jnp.asarray(relative_seconds)
        finite = jnp.isfinite(query)
        if self.interpolation == "constant":
            offset = jnp.broadcast_to(self.offsets[0], query.shape)
            support = jnp.ones_like(finite)
        else:
            support = (query >= self.nodes[0]) & (query <= self.nodes[-1])
            if self.interpolation == "linear":
                offset = jnp.interp(query, self.nodes, self.offsets)
            else:
                index = jnp.searchsorted(self.nodes, query, side="right") - 1
                offset = self.offsets[jnp.clip(index, 0, int(self.nodes.size) - 1)]
        valid = finite & support
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                support,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.INVALID_DOMAIN),
            ),
        ).astype(jnp.int32)
        offset = jnp.where(valid, offset, 0.0)
        return TimeScaleTransformResult(
            query + offset, offset, valid, status, self.transform_id
        )

    def inverse(self) -> TimeScaleTransform:
        return TimeScaleTransform(
            self.target_scale,
            self.source_scale,
            self.nodes + self.offsets,
            -self.offsets,
            self.provenance,
            interpolation=self.interpolation,
        )

    @classmethod
    def constant(
        cls,
        source_scale: TimeScaleName,
        target_scale: TimeScaleName,
        offset_seconds: float,
        provenance: AstrodynamicsDataProvenance,
        /,
    ) -> TimeScaleTransform:
        return cls(
            source_scale,
            target_scale,
            jnp.asarray((0.0,)),
            jnp.asarray((offset_seconds,)),
            provenance,
            interpolation="constant",
        )

    @classmethod
    def tai_to_tt(cls, provenance: AstrodynamicsDataProvenance, /) -> TimeScaleTransform:
        return cls.constant("TAI", "TT", 32.184, provenance)

    @classmethod
    def gps_to_tai(cls, provenance: AstrodynamicsDataProvenance, /) -> TimeScaleTransform:
        return cls.constant("GPS", "TAI", 19.0, provenance)


class LeapSecondTable(StrictModule, NonTrainableState):
    """UTC transition epochs and resulting TAI minus UTC offsets."""

    transition_seconds: Array
    tai_minus_utc: Array
    provenance: AstrodynamicsDataProvenance
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition_seconds: ArrayLike,
        tai_minus_utc: ArrayLike,
        provenance: AstrodynamicsDataProvenance,
        /,
    ):
        transitions = np.asarray(transition_seconds, dtype=float)
        offsets = np.asarray(tai_minus_utc, dtype=float)
        if (
            transitions.ndim != 1
            or transitions.size == 0
            or offsets.shape != transitions.shape
            or np.any(np.diff(transitions) <= 0.0)
            or np.any(np.diff(offsets) < 0.0)
            or np.any(~np.isfinite(transitions))
            or np.any(~np.isfinite(offsets))
        ):
            raise ValueError("Leap-second table is invalid.")
        self.transition_seconds = jnp.asarray(transitions)
        self.tai_minus_utc = jnp.asarray(offsets)
        self.provenance = provenance
        self.table_id = canonical_fingerprint(
            {
                "kind": "leap-second-table",
                "transitions": transitions.tolist(),
                "offsets": offsets.tolist(),
                "provenance": provenance.provenance_id,
            }
        )

    def utc_to_tai(self) -> TimeScaleTransform:
        return TimeScaleTransform(
            "UTC",
            "TAI",
            self.transition_seconds,
            self.tai_minus_utc,
            self.provenance,
            interpolation="step",
        )


class PreparedTimeRoute(StrictModule, NonTrainableState):
    """One statically compiled route through astronomical time scales."""

    transforms: tuple[TimeScaleTransform, ...]
    source_scale: TimeScaleName = eqx.field(static=True)
    target_scale: TimeScaleName = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(self, transforms: tuple[TimeScaleTransform, ...], /):
        items = tuple(transforms)
        if not items:
            raise ValueError("Prepared time route requires at least one transform.")
        for left, right in zip(items[:-1], items[1:], strict=True):
            if left.target_scale != right.source_scale:
                raise ValueError("Prepared time route is disconnected.")
        self.transforms = items
        self.source_scale = items[0].source_scale
        self.target_scale = items[-1].target_scale
        self.route_id = canonical_fingerprint(
            {"kind": "prepared-time-route", "transforms": [x.transform_id for x in items]}
        )

    def apply(self, relative_seconds: ArrayLike, /) -> TimeScaleTransformResult:
        value = jnp.asarray(relative_seconds)
        total_offset = jnp.zeros_like(value)
        valid = jnp.ones_like(value, dtype=bool)
        status = jnp.zeros_like(value, dtype=jnp.int32)
        for transform in self.transforms:
            result = transform.apply(value)
            value = result.relative_seconds
            total_offset = total_offset + result.offset_seconds
            valid = valid & result.valid
            status = jnp.where(valid, result.status, status)
        return TimeScaleTransformResult(value, total_offset, valid, status, self.route_id)


def relativistic_linear_transform(
    source_scale: TimeScaleName,
    target_scale: TimeScaleName,
    reference_jd: float,
    provenance: AstrodynamicsDataProvenance,
    /,
) -> TimeScaleTransform:
    """Construct IAU linear TCG/TCB scale transformations around one reference JD."""

    elapsed = (float(reference_jd) - _REFERENCE_JD) * 86400.0
    if (source_scale, target_scale) == ("TT", "TCG"):
        offset = _LG * elapsed
    elif (source_scale, target_scale) == ("TDB", "TCB"):
        offset = _LB * elapsed - _TDB0
    else:
        raise ValueError("Requested relativistic linear route is unsupported.")
    return TimeScaleTransform.constant(source_scale, target_scale, offset, provenance)


def convert_instant(
    instant: TimeInstant,
    route: PreparedTimeRoute,
    /,
) -> TimeInstant:
    """Host conversion of one exact instant through a prepared route."""

    if not isinstance(instant, TimeInstant) or not isinstance(route, PreparedTimeRoute):
        raise TypeError("instant and route have incompatible types.")
    if instant.scale != route.source_scale:
        raise ValueError("Time instant scale does not match route source.")
    result = route.apply(jnp.asarray(0.0))
    if not bool(result.valid):
        raise ValueError("Time instant lies outside route coverage.")
    offset_days = float(result.offset_seconds) / 86400.0
    return TimeInstant(
        JulianDate(instant.julian_date.high, instant.julian_date.low + offset_days),
        route.target_scale,
    )


__all__ = [
    "LeapSecondTable",
    "PreparedTimeRoute",
    "TimeInterpolation",
    "TimeScaleName",
    "TimeScaleTransform",
    "TimeScaleTransformResult",
    "convert_instant",
    "relativistic_linear_transform",
]
