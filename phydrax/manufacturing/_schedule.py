#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

from .._fingerprint import canonical_fingerprint
from ._path import ToolpathEvent


@dataclass(frozen=True, slots=True)
class ProcessSchedule:
    events: tuple[ToolpathEvent, ...]

    @classmethod
    def create(cls, events):
        return cls(tuple(sorted(events, key=lambda x: (x.start_time_s, x.event_id))))

    def __post_init__(self):
        if not self.events or len({x.event_id for x in self.events}) != len(self.events):
            raise ValueError("Schedule events must be nonempty and unique.")
        if any(
            b.start_time_s < a.end_time_s
            for a, b in zip(self.events, self.events[1:], strict=False)
        ):
            raise ValueError("Bounded schedule requires nonoverlap.")

    @property
    def schedule_id(self):
        return canonical_fingerprint(
            {
                "kind": "process-schedule",
                "events": [x.event_fingerprint for x in self.events],
            }
        )


__all__ = ["ProcessSchedule"]
