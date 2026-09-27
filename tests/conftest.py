from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import settings


settings.register_profile(
    "phydrax",
    deadline=None,
    max_examples=25,
    print_blob=True,
)
settings.register_profile(
    "phydrax-stress",
    deadline=None,
    max_examples=200,
    print_blob=True,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "phydrax"))


@pytest.fixture(autouse=True)
def _strict_jax_numerics(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.node.get_closest_marker("strict_jax") is None:
        yield
        return

    import jax

    with (
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        yield


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
    from phydrax import logging as pxlogging

    path = tmp_path / "events.jsonl"
    handler_id = pxlogging.add_json_sink(path, level="TRACE")
    pxlogging.enable()
    try:
        yield CapturedPhydraxEvents(path)
    finally:
        pxlogging.remove_sink(handler_id)
        pxlogging.disable()
