#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..._fingerprint import canonical_fingerprint


class FragmentStatus(StrEnum):
    COMPLETE = "complete"
    CORRUPT = "corrupt"


@dataclass(frozen=True, slots=True)
class RawFragmentRecord:
    event_id: int
    source_id: str
    orbit: int
    bunch_crossing_id: int
    timestamp: int
    payload_checksum: str
    payload_size_bytes: int
    status: FragmentStatus

    def __post_init__(self) -> None:
        source = str(self.source_id).strip()
        checksum = str(self.payload_checksum).strip()
        values = tuple(
            map(
                int,
                (
                    self.event_id,
                    self.orbit,
                    self.bunch_crossing_id,
                    self.timestamp,
                    self.payload_size_bytes,
                ),
            )
        )
        if (
            not source
            or not checksum
            or any(value < 0 for value in values)
            or not isinstance(self.status, FragmentStatus)
        ):
            raise ValueError(
                "Raw fragment identity, payload, timing, and status are invalid."
            )
        object.__setattr__(self, "source_id", source)
        object.__setattr__(self, "payload_checksum", checksum)
        (
            event_id,
            orbit,
            bunch_crossing_id,
            timestamp,
            payload_size_bytes,
        ) = values
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "orbit", orbit)
        object.__setattr__(self, "bunch_crossing_id", bunch_crossing_id)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "payload_size_bytes", payload_size_bytes)


class EventBuildStatus(StrEnum):
    COMPLETE = "complete"
    MISSING_SOURCE = "missing-source"
    DUPLICATE_SOURCE = "duplicate-source"
    CORRUPT_FRAGMENT = "corrupt-fragment"
    CLOCK_MISMATCH = "clock-mismatch"


@dataclass(frozen=True, slots=True)
class BuiltEventRecord:
    event_id: int
    orbit: int
    bunch_crossing_id: int
    fragment_source_ids: tuple[str, ...]
    payload_size_bytes: int
    status: EventBuildStatus
    evidence_ids: tuple[str, ...]
    record_id: str


@dataclass(frozen=True, slots=True)
class EventBuildingPlan:
    required_source_ids: tuple[str, ...]
    maximum_timestamp_spread: int
    plan_id: str

    def __init__(self, required_source_ids, /, *, maximum_timestamp_spread: int):
        sources = tuple(sorted(str(value).strip() for value in required_source_ids))
        spread = int(maximum_timestamp_spread)
        if (
            not sources
            or any(not value for value in sources)
            or len(set(sources)) != len(sources)
            or spread < 0
        ):
            raise ValueError("Event-building sources and timestamp policy are invalid.")
        object.__setattr__(self, "required_source_ids", sources)
        object.__setattr__(self, "maximum_timestamp_spread", spread)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "trigger-event-building-plan",
                    "sources": list(sources),
                    "maximum_timestamp_spread": spread,
                }
            ),
        )


def build_fragment_events(
    plan: EventBuildingPlan,
    fragments: tuple[RawFragmentRecord, ...],
    /,
) -> tuple[BuiltEventRecord, ...]:
    """Replay deterministic fragment grouping without accessing live readout systems."""
    if not isinstance(plan, EventBuildingPlan):
        raise TypeError("plan must be EventBuildingPlan.")
    fragments_ = tuple(fragments)
    if any(not isinstance(value, RawFragmentRecord) for value in fragments_):
        raise TypeError("fragments must contain RawFragmentRecord values.")
    event_ids = tuple(sorted({value.event_id for value in fragments_}))
    output: list[BuiltEventRecord] = []
    for event_id in event_ids:
        event_fragments = tuple(
            value for value in fragments_ if value.event_id == event_id
        )
        sources = tuple(value.source_id for value in event_fragments)
        source_set = set(sources)
        orbit_values = {value.orbit for value in event_fragments}
        bunch_values = {value.bunch_crossing_id for value in event_fragments}
        timestamp_values = tuple(value.timestamp for value in event_fragments)
        if any(value.status is FragmentStatus.CORRUPT for value in event_fragments):
            status = EventBuildStatus.CORRUPT_FRAGMENT
        elif len(source_set) != len(sources):
            status = EventBuildStatus.DUPLICATE_SOURCE
        elif source_set != set(plan.required_source_ids):
            status = EventBuildStatus.MISSING_SOURCE
        elif (
            len(orbit_values) != 1
            or len(bunch_values) != 1
            or max(timestamp_values) - min(timestamp_values)
            > plan.maximum_timestamp_spread
        ):
            status = EventBuildStatus.CLOCK_MISMATCH
        else:
            status = EventBuildStatus.COMPLETE
        evidence = tuple(sorted(value.payload_checksum for value in event_fragments))
        orbit = min(orbit_values) if orbit_values else 0
        bunch = min(bunch_values) if bunch_values else 0
        size = sum(value.payload_size_bytes for value in event_fragments)
        record_id = canonical_fingerprint(
            {
                "kind": "built-trigger-event",
                "event_id": event_id,
                "orbit": orbit,
                "bunch_crossing": bunch,
                "sources": sorted(sources),
                "size": size,
                "status": status.value,
                "evidence": list(evidence),
                "plan": plan.plan_id,
            }
        )
        output.append(
            BuiltEventRecord(
                event_id,
                orbit,
                bunch,
                tuple(sorted(sources)),
                size,
                status,
                evidence,
                record_id,
            )
        )
    return tuple(output)


__all__ = [
    "BuiltEventRecord",
    "EventBuildingPlan",
    "EventBuildStatus",
    "FragmentStatus",
    "RawFragmentRecord",
    "build_fragment_events",
]
