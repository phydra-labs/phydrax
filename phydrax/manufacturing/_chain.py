#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ManufacturingStage:
    stage_id: str
    predecessor_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.stage_id or self.stage_id in self.predecessor_ids:
            raise ValueError("Manufacturing stage identity/dependencies invalid.")


@dataclass(frozen=True, slots=True)
class ManufacturingProcessGraph:
    stages: tuple[ManufacturingStage, ...]

    @classmethod
    def create(cls, stages):
        return cls(tuple(stages))

    def __post_init__(self):
        ids = {s.stage_id for s in self.stages}
        if len(ids) != len(self.stages) or any(
            set(s.predecessor_ids) - ids for s in self.stages
        ):
            raise ValueError("Manufacturing graph has duplicate/unknown stages.")
        visiting = set()
        done = set()
        by = {s.stage_id: s for s in self.stages}

        def visit(name):
            if name in visiting:
                raise ValueError("Manufacturing graph contains a cycle.")
            if name in done:
                return
            visiting.add(name)
            for parent in by[name].predecessor_ids:
                visit(parent)
            visiting.remove(name)
            done.add(name)

        for name in ids:
            visit(name)


__all__ = ["ManufacturingProcessGraph", "ManufacturingStage"]
