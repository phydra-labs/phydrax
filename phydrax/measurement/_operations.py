#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class OperationalCoordinate(StrictModule, NonTrainableState):
    """Namespaced ordered coordinate such as run/luminosity block or run/spill."""

    namespace: str = eqx.field(static=True)
    axes: tuple[tuple[str, int], ...] = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)

    def __init__(self, namespace: str, axes: Mapping[str, int], /):
        namespace_ = _identifier(namespace, "Operational namespace")
        if not isinstance(axes, Mapping) or not axes:
            raise TypeError("axes must be a non-empty mapping.")
        axes_ = tuple(
            (_identifier(name, "Operational axis"), int(value))
            for name, value in axes.items()
        )
        names = tuple(name for name, _ in axes_)
        if len(set(names)) != len(names) or any(value < 0 for _, value in axes_):
            raise ValueError("Operational axes must be unique and nonnegative.")
        self.namespace = namespace_
        self.axes = axes_
        self.coordinate_id = canonical_fingerprint(
            {
                "kind": "operational-coordinate",
                "namespace": namespace_,
                "axes": list(axes_),
            }
        )

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(value for _, value in self.axes)


class OperationalInterval(StrictModule, NonTrainableState):
    """Half-open validity interval over one exact ordered coordinate space."""

    start: OperationalCoordinate
    end: OperationalCoordinate
    interval_id: str = eqx.field(static=True)

    def __init__(self, start: OperationalCoordinate, end: OperationalCoordinate, /):
        if not isinstance(start, OperationalCoordinate) or not isinstance(
            end, OperationalCoordinate
        ):
            raise TypeError("start and end must be OperationalCoordinate values.")
        if start.namespace != end.namespace or tuple(
            name for name, _ in start.axes
        ) != tuple(name for name, _ in end.axes):
            raise ValueError(
                "Operational interval endpoints must share namespace and axes."
            )
        if start.values >= end.values:
            raise ValueError("Operational interval must be strictly increasing.")
        self.start = start
        self.end = end
        self.interval_id = canonical_fingerprint(
            {
                "kind": "operational-interval",
                "start": start.coordinate_id,
                "end": end.coordinate_id,
            }
        )

    def contains(self, coordinate: OperationalCoordinate, /) -> bool:
        if not isinstance(coordinate, OperationalCoordinate):
            raise TypeError("coordinate must be OperationalCoordinate.")
        if coordinate.namespace != self.start.namespace or tuple(
            name for name, _ in coordinate.axes
        ) != tuple(name for name, _ in self.start.axes):
            return False
        return self.start.values <= coordinate.values < self.end.values


class ConditionPayloadReference(StrictModule, NonTrainableState):
    semantic_name: str = eqx.field(static=True)
    authority: str = eqx.field(static=True)
    external_tag: str = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    source_uri: str = eqx.field(static=True)
    unit_ids: tuple[str, ...] = eqx.field(static=True)
    dependency_ids: tuple[str, ...] = eqx.field(static=True)
    interval: OperationalInterval
    payload_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        semantic_name: str,
        authority: str,
        external_tag: str,
        checksum: str,
        source_uri: str,
        interval: OperationalInterval,
        unit_ids: Sequence[str] = (),
        dependency_ids: Sequence[str] = (),
    ):
        if not isinstance(interval, OperationalInterval):
            raise TypeError("interval must be OperationalInterval.")
        values = tuple(
            _identifier(value, name)
            for value, name in (
                (semantic_name, "Semantic name"),
                (authority, "Authority"),
                (external_tag, "External tag"),
                (checksum, "Checksum"),
                (source_uri, "Source URI"),
            )
        )
        units = tuple(sorted(_identifier(value, "Unit ID") for value in unit_ids))
        dependencies = tuple(
            sorted(_identifier(value, "Dependency ID") for value in dependency_ids)
        )
        if len(set(units)) != len(units) or len(set(dependencies)) != len(dependencies):
            raise ValueError("Condition units and dependencies must be unique.")
        (
            self.semantic_name,
            self.authority,
            self.external_tag,
            self.checksum,
            self.source_uri,
        ) = values
        self.unit_ids = units
        self.dependency_ids = dependencies
        self.interval = interval
        self.payload_id = canonical_fingerprint(
            {
                "kind": "condition-payload-reference",
                "values": list(values),
                "units": list(units),
                "dependencies": list(dependencies),
                "interval": interval.interval_id,
            }
        )


class ResolvedConditionSnapshot(StrictModule, NonTrainableState):
    coordinate: OperationalCoordinate
    payloads: tuple[ConditionPayloadReference, ...]
    resolution_time: int = eqx.field(static=True)
    resolver_id: str = eqx.field(static=True)
    missing_names: tuple[str, ...] = eqx.field(static=True)
    overlapping_names: tuple[str, ...] = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate: OperationalCoordinate,
        payloads: Sequence[ConditionPayloadReference],
        /,
        *,
        resolution_time: int,
        resolver_id: str,
        required_names: Sequence[str] = (),
    ):
        if not isinstance(coordinate, OperationalCoordinate):
            raise TypeError("coordinate must be OperationalCoordinate.")
        payloads_ = tuple(payloads)
        if any(not isinstance(value, ConditionPayloadReference) for value in payloads_):
            raise TypeError("payloads must contain ConditionPayloadReference values.")
        if any(not value.interval.contains(coordinate) for value in payloads_):
            raise ValueError(
                "Every resolved condition payload must contain the coordinate."
            )
        names = tuple(value.semantic_name for value in payloads_)
        overlapping = tuple(sorted(name for name in set(names) if names.count(name) > 1))
        required = tuple(
            sorted(_identifier(value, "Required condition") for value in required_names)
        )
        missing = tuple(sorted(set(required) - set(names)))
        resolution = int(resolution_time)
        if resolution < 0:
            raise ValueError("resolution_time must be nonnegative.")
        self.coordinate = coordinate
        self.payloads = tuple(
            sorted(payloads_, key=lambda value: (value.semantic_name, value.payload_id))
        )
        self.resolution_time = resolution
        self.resolver_id = _identifier(resolver_id, "Resolver ID")
        self.missing_names = missing
        self.overlapping_names = overlapping
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "resolved-condition-snapshot",
                "coordinate": coordinate.coordinate_id,
                "payloads": [value.payload_id for value in self.payloads],
                "resolution_time": resolution,
                "resolver": self.resolver_id,
                "missing": list(missing),
                "overlapping": list(overlapping),
            }
        )

    @property
    def successful(self) -> bool:
        return not self.missing_names and not self.overlapping_names


class ExposureKind(StrEnum):
    DELIVERED_LUMINOSITY = "delivered-luminosity"
    RECORDED_LUMINOSITY = "recorded-luminosity"
    CERTIFIED_LUMINOSITY = "certified-luminosity"
    PROTONS_ON_TARGET = "protons-on-target"
    LIVE_TIME = "live-time"
    TARGET_MASS = "target-mass"
    DETECTOR_EXPOSURE = "detector-exposure"


class ExposureRecord(StrictModule, NonTrainableState):
    kind: ExposureKind = eqx.field(static=True)
    value: float = eqx.field(static=True)
    uncertainty: float = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    authority: str = eqx.field(static=True)
    correlation_id: str = eqx.field(static=True)
    interval: OperationalInterval
    exposure_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ExposureKind,
        value: float,
        uncertainty: float,
        unit_id: str,
        interval: OperationalInterval,
        /,
        *,
        authority: str,
        correlation_id: str,
    ):
        if not isinstance(kind, ExposureKind) or not isinstance(
            interval, OperationalInterval
        ):
            raise TypeError("kind and interval must use operational measurement types.")
        value_ = float(value)
        uncertainty_ = float(uncertainty)
        if (
            not math.isfinite(value_)
            or value_ < 0.0
            or not math.isfinite(uncertainty_)
            or uncertainty_ < 0.0
        ):
            raise ValueError(
                "Exposure value and uncertainty must be finite and nonnegative."
            )
        self.kind = kind
        self.value = value_
        self.uncertainty = uncertainty_
        self.unit_id = _identifier(unit_id, "Unit ID")
        self.authority = _identifier(authority, "Authority")
        self.correlation_id = _identifier(correlation_id, "Correlation ID")
        self.interval = interval
        self.exposure_id = canonical_fingerprint(
            {
                "kind": "exposure-record",
                "exposure_kind": kind.value,
                "value": value_,
                "uncertainty": uncertainty_,
                "unit": self.unit_id,
                "authority": self.authority,
                "correlation": self.correlation_id,
                "interval": interval.interval_id,
            }
        )


class DataQualityAnnotation(StrictModule, NonTrainableState):
    interval: OperationalInterval
    defect_ids: tuple[str, ...] = eqx.field(static=True)
    certified: bool = eqx.field(static=True)
    authority: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    annotation_id: str = eqx.field(static=True)

    def __init__(
        self,
        interval: OperationalInterval,
        /,
        *,
        defect_ids: Sequence[str] = (),
        certified: bool,
        authority: str,
        evidence_ids: Sequence[str],
    ):
        if not isinstance(interval, OperationalInterval):
            raise TypeError("interval must be OperationalInterval.")
        defects = tuple(sorted(_identifier(value, "Defect ID") for value in defect_ids))
        evidence = tuple(
            sorted(_identifier(value, "Evidence ID") for value in evidence_ids)
        )
        if (
            len(set(defects)) != len(defects)
            or len(set(evidence)) != len(evidence)
            or not evidence
        ):
            raise ValueError(
                "Defect and evidence identities must be unique; evidence is required."
            )
        self.interval = interval
        self.defect_ids = defects
        self.certified = bool(certified)
        self.authority = _identifier(authority, "Authority")
        self.evidence_ids = evidence
        self.annotation_id = canonical_fingerprint(
            {
                "kind": "data-quality-annotation",
                "interval": interval.interval_id,
                "defects": list(defects),
                "certified": bool(certified),
                "authority": self.authority,
                "evidence": list(evidence),
            }
        )


__all__ = [
    "ConditionPayloadReference",
    "DataQualityAnnotation",
    "ExposureKind",
    "ExposureRecord",
    "OperationalCoordinate",
    "OperationalInterval",
    "ResolvedConditionSnapshot",
]
