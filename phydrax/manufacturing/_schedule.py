#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from .._fingerprint import canonical_fingerprint
from ._path import ToolpathEvent


@dataclass(frozen=True, slots=True)
class ProcessSchedule:
    events: tuple[ToolpathEvent, ...]

    @classmethod
    def create(cls, events: Iterable[ToolpathEvent]) -> Self:
        return cls(tuple(sorted(events, key=lambda x: (x.start_time_s, x.event_id))))

    def __post_init__(self) -> None:
        canonical = tuple(sorted(self.events, key=lambda x: (x.start_time_s, x.event_id)))
        if (
            not self.events
            or any(not isinstance(event, ToolpathEvent) for event in self.events)
            or len({x.event_id for x in self.events}) != len(self.events)
            or self.events != canonical
        ):
            raise ValueError("Schedule events must be nonempty, unique, and canonical.")
        if any(
            b.start_time_s < a.end_time_s
            for a, b in zip(self.events, self.events[1:], strict=False)
        ):
            raise ValueError("Bounded schedule requires nonoverlap.")

    @property
    def schedule_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "process-schedule",
                "events": [x.event_fingerprint for x in self.events],
            }
        )


__all__ = ["ProcessSchedule"]
