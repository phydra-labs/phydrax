#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from phydrax import logging as pxlogging


@dataclass(frozen=True, slots=True)
class CapturedPhydraxEvents:
    path: Path

    def records(self, event: str | None = None) -> list[dict[str, object]]:
        records = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
        ]
        if event is None:
            return records
        return [record for record in records if record["event"] == event]


@pytest.fixture
def phydrax_events(tmp_path: Path) -> Iterator[CapturedPhydraxEvents]:
    path = tmp_path / "events.jsonl"
    handler_id = pxlogging.add_json_sink(path, level="TRACE")
    pxlogging.enable()
    try:
        yield CapturedPhydraxEvents(path)
    finally:
        pxlogging.remove_sink(handler_id)
        pxlogging.disable()
